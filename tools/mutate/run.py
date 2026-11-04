"""Mutation checks: each recorded mutation of a family's source must make the named tests fail.

    python -X utf8 -m tools.mutate.run FAMILY [FAMILY ...]

FAMILY names a module in tools/mutate holding MUTATIONS. Every mutation runs in a temporary copy of the worktree
(src, tools, the generated files and the root inputs), never in the worktree itself. With `rebuild` the copy's page is
rebuilt from the mutated source before its tests run, and its payload too unless the mutation is of a page template or
shader, which only the page reads; a mutation that breaks the build counts as caught.
The same tests must first pass on an unmutated copy, so a failure that is not the mutation's (an import error, a
missing file, an unavailable browser) reports INVALID instead of caught. Tests that their module lists in
EXPECTED_FAILURES are left out of both runs: they fail either way, so they can neither invalidate a mutation's run nor
catch it, and a mutation that names only such tests is INVALID. A module's INPUTS names the pinned inputs (tools.inputs)
its tests read from build/inputs; each copy gets them verified, and without them its mutations report BLOCKED. Exit
status 1 when any mutation survives or is invalid or blocked.
Each mutation registers as heavy work while it is judged (tools.locks), so a timing run waits for the one under way and
can start before the next. A mutation whose tests drive a browser (tests of one of tools.testrun's GPU suites) is judged
on the GPU that HOG_GPU names, holding that GPU's lock (locks.gpu_work) through its copy, rebuild and tests so that a
suite on it waits for it rather than overlapping it. Without a valid HOG_GPU, or when its tests are skipped as blocked
because Edge renders on another GPU, it reports BLOCKED.
"""
import fnmatch
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

from tools import inputs, locks
from tools.mutate.runner import BLOCKED, NO_TESTS
from tools.paths import ROOT

COPIED = ("build/data.js", "build/meshes.js", "hogwarts-flyover.html", "georef.json", "feature_offsets.json")
PAGE_ONLY = (".html", ".glsl")      # page templates and shaders: assemble.py reads them, the payload's build never does
RUNNER = Path(__file__).with_name("runner.py")
ONLY_LISTED = "INVALID (every test it names is listed in EXPECTED_FAILURES)"
SKIPPED_BLOCKED = "BLOCKED (its tests were skipped as blocked: Edge renders on another GPU than HOG_GPU names)"


class Mutation(NamedTuple):
    name: str
    file: str                  # path under src/
    old: str                   # text that must occur exactly once
    new: str
    tests: tuple               # unittest names, such as "test_assets_grounds.HutTests"
    rebuild: bool = True       # rebuild the page (and the payload, see PAGE_ONLY) from the mutated source first


def copy_worktree(target, kept=()):
    shutil.copytree(ROOT / "src", target / "src", ignore=shutil.ignore_patterns("__pycache__"))
    if (ROOT / "tools").is_dir():               # so tests that import tools (test_shaders, test_site) import the copy's
        shutil.copytree(ROOT / "tools", target / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    for rel in COPIED:
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target / rel)
    for name in kept:
        path = target / "build" / "inputs" / inputs.spec(name)["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inputs.require(name), path)


def python(*args, cwd):
    return subprocess.run([sys.executable, "-X", "utf8", *args], cwd=cwd, capture_output=True, text=True)


def run(m, mutate, kept=()):
    """'pass', 'fail', 'build failed', 'no tests' or 'blocked': the tests of `m` on a fresh copy holding the inputs `kept`
    (its module's INPUTS), with or without its mutation."""
    with tempfile.TemporaryDirectory(prefix="hog-mutate-") as folder:
        root = Path(folder)
        copy_worktree(root, kept)
        if mutate:
            source = root / "src" / m.file
            source.write_text(source.read_text(encoding="utf-8").replace(m.old, m.new), encoding="utf-8", newline="\n")
        if m.rebuild:
            steps = ("src/assemble.py",) if m.file.endswith(PAGE_ONLY) else ("src/build_meshes.py", "src/assemble.py")
            for step in steps:
                if python(step, cwd=root).returncode:
                    return "build failed"
        status = python(str(RUNNER), *m.tests, cwd=root / "src").returncode
        return {0: "pass", NO_TESTS: "no tests", BLOCKED: "blocked"}.get(status, "fail")


def check_anchors(mutations):
    for m in mutations:
        found = (ROOT / "src" / m.file).read_text(encoding="utf-8").count(m.old)
        if found != 1:
            raise ValueError(f"{m.name}: the text to mutate occurs {found} times in {m.file}, not once")


def drives_browser(tests):
    """Whether any of the unittest names `tests` is in one of tools.testrun's GPU suites, whose tests drive Edge."""
    from tools.testrun import SUITES            # noqa: PLC0415
    suites = [s for s in SUITES.values() if s.get("gpu")]
    return any(module in s.get("modules", ()) or any(fnmatch.fnmatch(f"{module}.py", p) for p in s.get("patterns", ()))
               for module in (t.split(".")[0] for t in tests) for s in suites)


def judge(m, kept, clean):
    """The verdict on mutation `m`, its module's inputs `kept`; `clean` caches the unmutated runs by tests, rebuild and
    inputs."""
    key = (m.tests, m.rebuild, kept)
    if key not in clean:
        clean[key] = run(m, mutate=False, kept=kept)
    if clean[key] == "no tests":
        return ONLY_LISTED
    if clean[key] == "blocked":
        return SKIPPED_BLOCKED
    if clean[key] != "pass":
        return f"INVALID (the tests do not pass without the mutation: {clean[key]})"
    outcome = run(m, mutate=True, kept=kept)
    return {"fail": "caught", "build failed": "caught (build failed)", "pass": "SURVIVED", "no tests": ONLY_LISTED,
            "blocked": SKIPPED_BLOCKED}[outcome]


def blocked(names):
    """Why the inputs `names` are not available, or None."""
    try:
        for name in names:
            inputs.require(name)
    except inputs.Blocked as e:
        return str(e)
    return None


def main(families):
    modules = {f: importlib.import_module(f"tools.mutate.{f}") for f in families}
    plans = [(f, m) for f in families for m in modules[f].MUTATIONS]
    check_anchors([m for _, m in plans])
    kept = {f: tuple(getattr(modules[f], "INPUTS", ())) for f in families}
    missing = {f: blocked(kept[f]) for f in families}
    gpu = os.environ.get("HOG_GPU", "").lower()
    clean, bad = {}, 0
    for family, m in plans:
        browser = drives_browser(m.tests)
        if missing[family]:
            verdict = f"BLOCKED ({missing[family]})"
        elif browser and gpu not in locks.GPUS:
            verdict = f"BLOCKED (its tests drive a browser: set HOG_GPU to one of {', '.join(locks.GPUS)})"
        else:
            label = f"mutation check {family}: {m.name}"
            with locks.gpu_work(gpu, label) if browser else locks.heavy(label):
                verdict = judge(m, kept[family], clean)
        bad += not verdict.startswith("caught")
        print(f"{family}: {m.name}: {verdict}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) if len(sys.argv) > 1 else print(__doc__) or 2)
