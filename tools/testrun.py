"""Run a test suite with each test's outcome and time, always logged, and gate it on exact test IDs.

    python -X utf8 -m tools.testrun SUITE [--gpu amd|nvidia] [--shift METRES] [-k PATTERN] [--list]

Suites: assets, checks, site and tools (the tests under tools/) run on the CPU; render, smoke (the render tests tagged
@smoke), ui and probe drive Edge and need --gpu. A browser suite holds that GPU's machine-wide lock, and every suite registers as heavy work, so it waits
while a timing run holds the quiet lock (tools.locks). --shift raises the world by that many metres
(HOG_DATUM_SHIFT_M, assets only). Each run writes build/logs/<suite>[+<shift>m][~<filter>]-<gpu or cpu>-<page tag>.log,
the unittest transcript followed by a table of every test, and a .json of the same.

Each test ends as one of:
  pass, fail, error;
  expected-fail: listed in its module's EXPECTED_FAILURES, and failing or erroring;
  skipped;
  blocked: skipped for a reason starting "blocked:", such as an unavailable GPU or tool, or its class's setup was.
A listed test that passes, or that did not run, fails the gate however the counts compare. The verdict is error if
any test errored, else fail, else blocked (also when nothing ran), else pass. Exit status: 0 pass, 1 fail or error,
3 blocked, 2 usage.

A run without -k resumes where an earlier one stopped. Each test class that passes (every test a pass, a skip or a
listed failure, its class fixtures clean) is recorded as soon as its tearDownClass has run, or, in a module with a
tearDownModule or module cleanups, once those have run clean, with its tests' outcomes and times and its part of the
transcript, under the suite's input key (suite_key, the key of its S3 or S4 stage, which also tells the GPUs apart) in
build/logs/testrun/ (Store). A run under the same key runs only the classes without a record, so a pause or a crash
loses only the class it was in (in a module with a tearDownModule or module cleanups, the module's classes run so far),
and its log and report still cover every test, the recorded classes' parts first. A
class that fails, errors or is blocked is never recorded and always runs again, as does one whose tests all skip; a
changed input changes the key and reruns every class. A -k run neither skips nor records.
"""
import argparse
from collections import Counter
import fnmatch
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import unittest
from unittest.suite import _ErrorHolder

from tools import locks
from tools.inputs import Blocked
from tools.paths import LOGS, ROOT, SRC, page_tag, write_atomic

SUITES = {
    "assets": {"patterns": ("test_assets_*.py",)},
    "uncached": {"modules": ("test_uncached_build",)},
    "checks": {"modules": ("test_materials", "test_assemble", "test_shaders")},
    "site": {"modules": ("test_site",)},
    "render": {"patterns": ("test_render_*.py",), "gpu": True},
    "smoke": {"patterns": ("test_render_*.py",), "gpu": True, "smoke": True},
    "ui": {"modules": ("test_ui",), "gpu": True},
    "probe": {"modules": ("test_mesh_probe",), "gpu": True},
    "tools": {"patterns": ("test_*.py",), "dir": "tools"},
}
RANK = {"pass": 0, "skipped": 1, "blocked": 2, "fail": 3, "error": 4}
EXIT = {"pass": 0, "fail": 1, "error": 1, "blocked": 3}


def tests_in(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from tests_in(item)
        else:
            yield item


def class_id(cls):
    """A test class's ID, the prefix of its tests' IDs."""
    return f"{cls.__module__}.{cls.__qualname__}"


IMPORT_FAILURES = ("_FailedTest", "ModuleImportFailure")      # what discover makes of a module it cannot import


def load(name, pattern=None):
    """(the selected tests, the import failures, every test the suite holds before -k and the smoke tag select)."""
    spec = SUITES[name]
    loader = unittest.TestLoader()
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    suite = unittest.TestSuite()
    start = ROOT / spec["dir"] if "dir" in spec else SRC
    top = ROOT if "dir" in spec and start.is_relative_to(ROOT) else start     # a folder elsewhere is its own top level
    if str(top) not in sys.path:
        sys.path.insert(0, str(top))
    for p in spec.get("patterns", ()):
        suite.addTests(loader.discover(str(start), pattern=p, top_level_dir=str(top)))
    for module in spec.get("modules", ()):
        try:
            suite.addTests(loader.loadTestsFromName(module))
        except Exception:                       # as discover does: the import error becomes a failing test
            failed_import, _ = unittest.loader._make_failed_import_test(module, unittest.TestSuite)
            suite.addTests(failed_import)
    everything = list(tests_in(suite))
    failed = [t for t in everything if type(t).__name__ in IMPORT_FAILURES]
    match = pattern if pattern and "*" in pattern else f"*{pattern}*"

    def selected(t):
        if type(t).__name__ in IMPORT_FAILURES:
            return True
        if spec.get("smoke") and not getattr(getattr(t, t._testMethodName), "smoke", False):
            return False
        return not pattern or fnmatch.fnmatchcase(t.id(), match)
    return unittest.TestSuite(t for t in everything if selected(t)), failed, everything


def expected_failures(tests):
    """Every module's EXPECTED_FAILURES (class.method IDs), qualified with the module's name."""
    modules = {type(t).__module__ for t in tests}
    return {f"{m}.{rel}" for m in modules for rel in getattr(sys.modules[m], "EXPECTED_FAILURES", ())}


class Transcribing:
    """A result's stream that also keeps what it writes during each class's tests, by class (Recorder.transcripts)."""

    def __init__(self, stream, result):
        self.stream, self.result = stream, result

    def write(self, text):
        if self.result.current:
            self.result.transcripts.setdefault(self.result.current, []).append(text)
        return self.stream.write(text)

    def writeln(self, text=None):
        self.write(f"{text}\n" if text else "\n")

    def __getattr__(self, name):
        return getattr(self.stream, name)


class Recorder(unittest.TextTestResult):
    """Each test's worst outcome over its subtests, its time, the class or module fixtures that failed, and each class's
    part of the transcript."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records, self.fixtures, self.adapters, self.loads, self._t0 = {}, [], set(), {}, {}
        self.transcripts, self.current = {}, None
        self.stream = Transcribing(self.stream, self)

    def _set(self, test, outcome, detail):
        record = self.records.setdefault(test.id(), {"outcome": "pass", "detail": ""})
        if RANK[outcome] > RANK[record["outcome"]]:
            record["outcome"] = outcome
        record["detail"] += detail

    def _fixture(self, holder, outcome, detail):
        fixture, _, scope = holder.description.partition(" (")
        self.fixtures.append({"fixture": fixture, "scope": scope.rstrip(")"), "outcome": outcome, "detail": detail})

    def startTest(self, test):
        self.current = class_id(type(test))
        super().startTest(test)
        self._t0[test.id()] = time.perf_counter()

    def stopTest(self, test):
        super().stopTest(test)
        record = self.records.setdefault(test.id(), {"outcome": "pass", "detail": ""})
        record["seconds"] = round(time.perf_counter() - self._t0.pop(test.id()), 3)
        cls = type(test)
        if getattr(cls, "adapter", None):
            self.adapters.add(cls.adapter)
        if getattr(cls, "load_s", None) is not None:
            self.loads[class_id(cls)] = round(cls.load_s, 3)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._set(test, "fail", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        if isinstance(test, _ErrorHolder):
            self._fixture(test, "error", self._exc_info_to_string(err, test))
        else:
            self._set(test, "error", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        outcome = "blocked" if reason.startswith("blocked:") else "skipped"
        if isinstance(test, _ErrorHolder):
            self._fixture(test, outcome, reason)
        else:                                       # a skipped subtest leaves its whole test short of a pass
            self._set(getattr(test, "test_case", test), outcome, reason)

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            failed = issubclass(err[0], test.failureException)
            self._set(test, "fail" if failed else "error", f"[{subtest._subDescription()}] " + self._exc_info_to_string(err, test))

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._set(test, "fail", "(unittest.expectedFailure: list it in EXPECTED_FAILURES instead)\n")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._set(test, "fail", "(unittest.expectedFailure: list it in EXPECTED_FAILURES instead)\n")


def gate(ids, records, fixtures, expected, every=None):
    """Final outcomes of the tests `ids` that ran, problems outside any one test, and the verdict. `every` holds all the
    suite's test IDs before -k or the smoke tag selected `ids` (by default, `ids`)."""
    every = set(ids) if every is None else set(every)
    final, problems = {}, []
    for test_id in ids:
        ran = test_id in records
        record = dict(records.get(test_id) or {"outcome": None, "detail": "", "seconds": 0.0})
        if not ran:                              # its class or module setup failed or was skipped
            cause = next((f for f in fixtures if f["fixture"].startswith("setUp") and test_id.startswith(f["scope"] + ".")),
                         None)
            record.update(outcome=cause["outcome"] if cause else "error",
                          detail=cause["detail"] if cause else "did not run")
        if test_id in expected:                  # only a test that ran and failed is an expected failure
            outcome = record["outcome"]
            if ran and outcome in ("fail", "error"):
                record["outcome"] = "expected-fail"
            elif outcome in ("pass", "skipped"):
                why = "passed: remove it from the list" if outcome == "pass" else "was skipped, so it did not run"
                record.update(outcome="fail", detail=f"listed in EXPECTED_FAILURES but {why}\n" + record["detail"])
        final[test_id] = record
    problems += [f"{test_id} is listed in EXPECTED_FAILURES but there is no such test" for test_id in sorted(expected - every)]
    problems += [f"{f['fixture']} ({f['scope']}): {f['detail'].strip()}" for f in fixtures if not f["fixture"].startswith("setUp")]
    counts = Counter(r["outcome"] for r in final.values())
    if counts["error"] or any("tearDown" in p for p in problems):
        verdict = "error"
    elif counts["fail"] or problems:
        verdict = "fail"
    elif counts["blocked"] or not (counts["pass"] or counts["expected-fail"]):
        verdict = "blocked"
    else:
        verdict = "pass"
    return final, problems, verdict


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def git_head():
    run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return run.stdout.strip() or None


class ByClass(unittest.TestSuite):
    """A flat suite that calls ended(cls, result) once each test class has run its last test and its tearDownClass,
    and module_ended(module, result) once each module's tearDownModule and module cleanups have run."""

    def __init__(self, tests, ended, module_ended):
        super().__init__(tests)
        self.ended, self.module_ended = ended, module_ended

    def _tearDownPreviousClass(self, test, result):     # unittest calls it before each test, and once more at the end
        previous = getattr(result, "_previousTestClass", None)
        super()._tearDownPreviousClass(test, result)
        if previous is not None and previous is not test.__class__:
            self.ended(previous, result)

    def _handleModuleTearDown(self, result):            # when the module changes, and at the end, after the class's
        module = self._get_previous_module(result)
        super()._handleModuleTearDown(result)
        if module is not None:
            self.module_ended(module, result)


def torn_down_by_module(cls):
    """Whether a class's module tears down after it (tearDownModule, or module cleanups registered), so the class counts
    only once that has run clean."""
    return getattr(sys.modules.get(cls.__module__), "tearDownModule", None) is not None \
        or bool(unittest.case._module_cleanups)


def suite_key(name, gpu=None, shift=None):
    """The suite's input key, as its S3 or S4 stage computes it (tools.stages.suite_inputs, which takes in the GPU's
    adapter, Edge and its flags); Blocked when the suite cannot run."""
    from tools import stages                         # noqa: PLC0415 (stages reads SUITES from this module)
    return stages.key_of(stages.suite_inputs(name, gpu, shift))


class Store:
    """The records of a suite's classes that passed under one input key, one JSON file each, in
    LOGS/testrun/<suite>[+<shift>m]-<gpu or cpu>/<key>/. Opening it removes the suite's records under every other
    key."""

    def __init__(self, stem, key):
        self.key, self.folder = key, LOGS / "testrun" / stem / key[:16]
        if self.folder.parent.is_dir():
            for other in self.folder.parent.iterdir():
                if other != self.folder:
                    shutil.rmtree(other, ignore_errors=True)

    def reusable(self, tests):
        """{class ID: record} of the classes among `tests`, in their order, recorded with exactly the tests they
        hold."""
        held = {}
        for t in tests:
            held.setdefault(class_id(type(t)), set()).add(t.id())
        recorded = {}
        for path in sorted(self.folder.glob("*.json")) if self.folder.is_dir() else ():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if record.get("key") == self.key:
                recorded[record["class"]] = record
        return {c: recorded[c] for c, ids in held.items() if c in recorded and set(recorded[c]["tests"]) == ids}

    def save(self, cls, record):
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", cls)
        try:
            write_atomic(self.folder / f"{name}.json", json.dumps({"class": cls, "key": self.key, "time": time.time(),
                                                                   **record}, indent=1).encode("utf-8"))
        except OSError as e:
            print(f"note: {cls} was not recorded ({e}), so a rerun runs it again", file=sys.stderr, flush=True)


def set_environment(gpu, shift):
    """HOG_GPU and HOG_DATUM_SHIFT_M for this run, from its arguments: never inherited from the shell."""
    os.environ["HOG_GPU"] = gpu or ""
    if shift:
        os.environ["HOG_DATUM_SHIFT_M"] = str(shift)
    else:
        os.environ.pop("HOG_DATUM_SHIFT_M", None)


def run(name, gpu=None, shift=None, pattern=None):
    """Run suite `name`, resuming an earlier run under the same key (Store) unless `pattern` filters it; (verdict, path
    of the JSON report)."""
    spec = SUITES[name]
    if spec.get("gpu") and gpu is None:
        raise SystemExit(f"the {name} suite drives Edge: pass --gpu amd or --gpu nvidia")
    set_environment(gpu, shift)
    only = f"~{re.sub(r'[^A-Za-z0-9_.-]+', '_', pattern)}" if pattern else ""
    suite_tag = f"{name}{f'{float(shift):+g}m' if shift else ''}"
    stem = f"{suite_tag}{only}-{gpu or 'cpu'}-{page_tag()}"
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path, json_path = LOGS / f"{stem}.log", LOGS / f"{stem}.json"
    label = f"{stem} in {ROOT.name}"
    say = lambda message: print(message, file=sys.stderr, flush=True)
    started, head = time.time(), git_head()
    with open(log_path, "w", encoding="utf-8", newline="\n") as log, \
            locks.gpu_work(gpu, label, say) if spec.get("gpu") else locks.heavy(label, say):
        stream = Tee(sys.stderr, log)
        store = None
        if not pattern:
            try:
                store = Store(f"{suite_tag}-{gpu or 'cpu'}", suite_key(name, gpu, shift))
            except Blocked as e:
                stream.write(f"note: this run records no classes, since the suite's key cannot be made: {e}\n\n")
        suite, broken, everything = load(name, pattern)
        tests = list(tests_in(suite))
        ids = [t.id() for t in tests]
        expected = expected_failures(everything)
        reused = store.reusable(tests) if store else {}
        if reused:
            stream.write(f"The {len(reused)} classes that passed under this suite's key ({store.key[:12]}) before "
                         "this run was resumed, not run again:\n")
            for record in reused.values():
                stream.write(record["transcript"])
            stream.write("\nThis run:\n")
        todo = [t for t in tests if class_id(type(t)) not in reused]
        planned, held = {}, {}
        for t in todo:
            planned.setdefault(class_id(type(t)), []).append(t.id())

        def ended(cls, result):
            """Record a class that passed as soon as its tearDownClass has run, or, when its module tears down after it,
            hold it until that has run (module_ended)."""
            own, transcript = planned.get(class_id(cls), []), "".join(result.transcripts.pop(class_id(cls), []))
            result.current = None
            if store is None or not own:
                return
            fixtures = [f for f in result.fixtures if f["scope"] == class_id(cls)]
            if gate(own, result.records, fixtures, expected & set(own), own)[2] == "pass":
                record = {"suite": name, "gpu": gpu, "shift_m": shift, "head": head, "worktree": str(ROOT),
                          "tests": {i: result.records[i] for i in own}, "transcript": transcript,
                          "page_load_s": result.loads.get(class_id(cls)), "adapter": getattr(cls, "adapter", None)}
                if torn_down_by_module(cls):
                    held.setdefault(cls.__module__, []).append((class_id(cls), record))
                else:
                    store.save(class_id(cls), record)

        def module_ended(module, result):
            """Record the module's held classes, unless its setUpModule, tearDownModule or module cleanups failed."""
            waiting = held.pop(module, [])
            if not any(f["scope"] == module for f in result.fixtures):
                for cls, record in waiting:
                    store.save(cls, record)
        result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=Recorder).run(
            ByClass(todo, ended, module_ended))
        records = {i: r for record in reused.values() for i, r in record["tests"].items()} | result.records
        final, problems, verdict = gate(ids, records, result.fixtures, expected, {t.id() for t in everything})
        if broken:
            verdict = "error"
        loads = {c: r["page_load_s"] for c, r in reused.items() if r.get("page_load_s") is not None} | result.loads
        adapters = sorted(result.adapters | {r["adapter"] for r in reused.values() if r.get("adapter")})
        wall = time.time() - started
        stream.write(f"\n{'outcome':14s} {'seconds':>8s}  test\n")
        for test_id, r in final.items():
            stream.write(f"{r['outcome']:14s} {r.get('seconds', 0.0):8.1f}  {test_id}\n")
        for cls, seconds in loads.items():
            stream.write(f"{'page load':14s} {seconds:8.1f}  {cls}\n")
        for p in problems:
            stream.write(f"PROBLEM: {p}\n")
        counts = Counter(r["outcome"] for r in final.values())
        reused_s = round(sum(r.get("seconds", 0.0) for c in reused.values() for r in c["tests"].values()), 1)
        resumed = (f", resuming {len(reused)} classes that passed under this key before ({reused_s:.1f} s of tests)"
                   if reused else "")
        stream.write(f"\n{name}: {verdict.upper()} ({', '.join(f'{v} {k}' for k, v in sorted(counts.items()))}) in "
                     f"{wall:.1f} s{resumed}; log {log_path}\n")
    report = {"suite": name, "gpu": gpu, "shift_m": shift, "pattern": pattern, "verdict": verdict, "counts": dict(counts),
              "wall_s": round(wall, 1), "started": started, "page_tag": page_tag(), "head": head,
              "worktree": str(ROOT), "python": platform.python_version(), "adapters": adapters,
              "page_loads_s": loads, "problems": problems, "tests": final, "log": str(log_path),
              "key": store.key if store else None, "resumed": {"classes": list(reused), "seconds": reused_s}}
    write_atomic(json_path, json.dumps(report, indent=1).encode("utf-8"))
    return verdict, json_path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.testrun", description=__doc__.splitlines()[0])
    parser.add_argument("suite", choices=sorted(SUITES))
    parser.add_argument("--gpu", choices=locks.GPUS)
    parser.add_argument("--shift", type=float, help="raise the world by this many metres (assets)")
    parser.add_argument("-k", dest="pattern", help="only tests whose name matches (unittest -k)")
    parser.add_argument("--list", action="store_true", help="print the suite's test IDs and exit")
    args = parser.parse_args(argv)
    if args.list:
        os.environ["HOG_GPU"] = args.gpu or ""
        suite, broken, _ = load(args.suite, args.pattern)
        for t in [*tests_in(suite)]:
            print(t.id())
        return 1 if broken else 0
    verdict, _ = run(args.suite, args.gpu, args.shift, args.pattern)
    return EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())
