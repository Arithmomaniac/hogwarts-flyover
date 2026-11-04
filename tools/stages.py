"""The pipeline stages S0-S6 (handoff, "Pipeline stages and checkpoints"), their cache keys and the machine-wide ledger.
dodo.py makes them doit tasks; REBUILD.md says when to run which.

A stage's key hashes everything it reads:
- its code: the static import closure of its scripts, with whole packages;
- its data files;
- pinned external inputs, by their verified hashes;
- the locked package versions and the Python version;
- for browser stages, also the page, the GPU adapter and driver, the Edge version, HOG_GPU and Edge's launch flags.

Each run appends its verdict to <state>/ledger.jsonl, which every worktree on the machine shares. A stage is skipped
when its key already has a passing verdict and the files it wrote still hold what that verdict recorded. Family parts
live in the shared cache, so a part built in one worktree serves the others. Fail, error and blocked verdicts are
recorded but never reused. A stage that cannot run because an input, a tool or a GPU is missing is blocked, which is
never a pass.
"""
import ast
from dataclasses import dataclass, field
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Callable

from filelock import FileLock

from tools import inputs, locks
from tools.inputs import Blocked, HashCache
from tools.paths import LOGS, PAGE, ROOT, SRC, cache_dir, page_tag, replace, state_dir, write_atomic

DATA_JS, MESHES_JS, MANIFEST = ROOT / "build" / "data.js", ROOT / "build" / "meshes.js", SRC / "mesh_scene.json"
GENERATED = ("build/data.js", "build/meshes.js", "hogwarts-flyover.html", "src/mesh_scene.json")
PAGE_SOURCES = ("src/template.html", "src/template_b.html", "src/template_c.html",
                *(f"src/frag{i}.glsl" for i in range(1, 6)))
RENDER_TESTS = ("src/pagetest.py", "src/render_testing.py", "src/shaders.py")
ROOT_DATA = ("georef.json", "feature_offsets.json")     # data files at the root that src modules read
GPU_FLAGS = {"amd": ["--force-low-power-gpu"], "nvidia": ["--force-high-performance-gpu"]}   # as pagetest passes them
EXIT = {"pass": 0, "fail": 1, "error": 1, "blocked": 3}
VERDICTS = {0: "pass", 3: "blocked"}                 # other exit codes are fail


# ---------------------------------------------------------------- what a key hashes
def rel(path):
    """A repository path relative to the root, with forward slashes; any other path in full."""
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def files(*paths):
    """{path: sha256} of the files, None for a missing one."""
    with HashCache() as hashes:
        return {rel(p): hashes.sha256(ROOT / p) if (ROOT / p).is_file() else None for p in paths}


def tree(folder):
    """{path: sha256} of every file under a folder, caches aside."""
    return files(*sorted(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts))


def _module_file(name):
    """The repository file of an importable module name, or None for anything outside the repository."""
    head = name.split(".")
    for base in (SRC, ROOT):
        path = base.joinpath(*head)
        if path.with_suffix(".py").is_file():
            return path.with_suffix(".py")
        if (path / "__init__.py").is_file():
            return path / "__init__.py"
    return None


@functools.lru_cache(maxsize=None)
def _closure(script):
    seen, todo = set(), [ROOT / script]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                found = _module_file(name)
                if found:
                    todo.append(found)
    return tuple(sorted(rel(p) for p in seen))


def code(*scripts):
    """{path: sha256} of the scripts and every repository module they import, statically, through any depth. Modules
    loaded by name (src/materials's families, the family builders, tools/manifest/families) are added by the caller."""
    return files(*sorted({p for s in scripts for p in _closure(s)}))


def environment():
    return {"python": platform.python_version(), "packages": files("requirements.lock.txt")}


@functools.lru_cache(maxsize=None)
def machine():
    """The display adapters (name, driver version) and the installed Edge's version, from Windows."""
    script = ("$a = Get-CimInstance Win32_VideoController | ForEach-Object { @{name = $_.Name; driver = $_.DriverVersion} }; "
              "$e = Get-Item \"${env:ProgramFiles(x86)}\\Microsoft\\Edge\\Application\\msedge.exe\", "
              "\"$env:ProgramFiles\\Microsoft\\Edge\\Application\\msedge.exe\" -ErrorAction SilentlyContinue | Select-Object -First 1; "
              "@{adapters = @($a); edge = $e.VersionInfo.ProductVersion} | ConvertTo-Json -Compress -Depth 3")
    run = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True)
    found = json.loads(run.stdout or "{}")
    return {"adapters": [(a["name"], a["driver"]) for a in found.get("adapters") or []], "edge": found.get("edge")}


def browser(gpu):
    """What a browser stage's verdict depends on beyond the page and the tests: blocked when the GPU is not there."""
    words = {"amd": ("amd", "radeon"), "nvidia": ("nvidia",)}[gpu]
    adapter = [a for a in machine()["adapters"] if any(w in a[0].lower() for w in words)]
    if not adapter:
        raise Blocked(f"this machine shows no {gpu} display adapter ({machine()['adapters']})")
    if not machine()["edge"]:
        raise Blocked("Microsoft Edge is not installed where Playwright's msedge channel looks for it")
    return {"gpu": gpu, "adapter": adapter[0], "edge": machine()["edge"], "flags": GPU_FLAGS[gpu]}


def key_of(parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- the ledger
def ledger_file():
    return state_dir() / "ledger.jsonl"


def record(entry):
    with FileLock(str(ledger_file()) + ".lock"):
        with open(ledger_file(), "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")


def history(stage=None, key=None):
    try:
        lines = ledger_file().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if (stage is None or entry["stage"] == stage) and (key is None or entry["key"] == key):
            out.append(entry)
    return out


def reusable(stage, key):
    """The latest passing verdict for this stage and key whose outputs still hold the recorded bytes, or None."""
    for entry in reversed(history(stage, key)):
        if entry["verdict"] == "pass":
            outputs = entry.get("outputs", {})
            current = files(*outputs) if outputs else {}
            if all(current.get(rel(ROOT / p)) == h for p, h in outputs.items()):
                return entry
    return None


# ---------------------------------------------------------------- running a stage
@dataclass
class Stage:
    name: str
    inputs: Callable[[], dict]                      # the key's parts; raises Blocked when the stage cannot run
    run: Callable[[str], tuple]                     # (key) -> (verdict, log path or None, {output path: sha256})
    cached: bool = True                             # timing measures the machine, so its verdicts are not reused
    extra: dict = field(default_factory=dict)


def head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None


def execute(stage):
    """Run a stage unless its key has a reusable pass; True when it passed."""
    started = time.time()
    try:
        key = key_of(stage.inputs())
    except Blocked as e:
        return _finish(stage, None, "blocked", None, {}, started, str(e))
    hit = reusable(stage.name, key) if stage.cached else None
    if hit:
        print(f"{stage.name}: pass, reused ({key[:12]}, {time.strftime('%Y-%m-%d %H:%M', time.localtime(hit['time']))}"
              f" in {hit['worktree']}; log {hit.get('log')})", flush=True)
        return True
    try:
        verdict, log, outputs = stage.run(key)
        note = None
    except Blocked as e:
        verdict, log, outputs, note = "blocked", None, {}, str(e)
    return _finish(stage, key, verdict, log, outputs, started, note)


def _finish(stage, key, verdict, log, outputs, started, note):
    entry = {"stage": stage.name, "key": key, "verdict": verdict, "time": time.time(), "seconds": round(time.time() - started, 1),
             "worktree": str(ROOT), "head": head(), "log": str(log) if log else None, "outputs": outputs, **stage.extra,
             **({"note": note} if note else {})}
    if key is not None or verdict == "blocked":
        record(entry)
    print(f"{stage.name}: {verdict.upper()}" + (f" ({note})" if note else "") + (f"; log {log}" if log else ""), flush=True)
    return verdict == "pass"


def python(*args, log, env=None):
    """Run the venv's Python on args from the repository root, appending its output to `log`; its exit code."""
    with open(log, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"$ python {' '.join(map(str, args))}\n")
        f.flush()
        run = subprocess.run([sys.executable, "-X", "utf8", *map(str, args)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONUTF8": "1", **(env or {})})
    return run.returncode


def log_path(name, key):
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / f"{name.replace(':', '-')}-{key[:12]}.log"
    path.write_text("", encoding="utf-8")
    return path


# ---------------------------------------------------------------- S0-S2: the generated files
def families():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["families"]


def manifest_stage():
    def inputs_():
        return {"code": code("tools/manifest/gen.py"), "tree": tree("tools/manifest"),
                "data": files("build/data.js", "tools/whcache.lock.json"), **environment()}

    def run(key):
        log = log_path("manifest", key)
        with heavy("manifest"):
            fetched = python("-m", "tools.whcache", "fetch", log=log)
            if fetched:
                return ("blocked" if fetched == 3 else "error"), log, {}
            if python("-m", "tools.manifest.gen", log=log):
                return "error", log, {}
            if python("-m", "tools.manifest.verify", log=log):
                return "fail", log, {}
        return "pass", log, files("src/mesh_scene.json")
    return Stage("manifest", inputs_, run)


def s0_stage():
    """S0 from the pinned data.js of before the railway regrade: fix_berth.py (idempotent), then the harbour mouth
    (fix_harbour.py) and the railway regrade (fix_railway.py), as tools.site.build finishes it."""
    def inputs_():
        inputs.require("data-before-railway-x")
        return {"source": inputs.spec("data-before-railway-x")["sha256"], "code": code("src/fix_berth.py", "src/fix_harbour.py", "src/fix_railway.py"),
                "data": files("src/mesh_scene.json", "georef.json"), **environment()}

    def run(key):
        source, log = inputs.require("data-before-railway-x"), log_path("s0", key)
        with tempfile.TemporaryDirectory(prefix="hog-s0-") as folder, heavy("s0"):
            work = Path(folder) / "data.js"
            shutil.copyfile(source, work)
            for script in ("src/fix_berth.py", "src/fix_harbour.py", "src/fix_railway.py"):
                if python(script, work, log=log):
                    return "error", log, {}
            if not DATA_JS.is_file() or work.read_bytes() != DATA_JS.read_bytes():
                write_atomic(DATA_JS, work.read_bytes())
        return "pass", log, files("build/data.js")
    return Stage("s0", inputs_, run)


def part_inputs(family, shift=None):
    """What a family's part reads; for a part on the world raised `shift` metres, also the shift and how it is applied
    (the unshifted parts' keys do not change)."""
    module = f"src/mesh_{family}.py"
    parts = {"family": family, "code": code("src/build_meshes.py", *([module] if (ROOT / module).is_file() else [])),
             "materials": tree("src/materials"), "assets": tree(f"src/assets/{family}"),
             "data": files("build/data.js", "src/mesh_scene.json", "georef.json"), **environment()}
    if shift:
        parts.update(shift=float(shift), shifting=code("tools/shifted_part.py"))
    return parts


def part_label(family, shift=None):
    return f"{family}{float(shift):+g}m" if shift else family


def part_file(family, key):
    return cache_dir("parts") / f"{family}-{key[:16]}.json"


def s1_stage(family, shift=None):
    """A family's part, on the registered world or, for the asset tests' shifted runs, on one raised `shift` metres."""
    label = part_label(family, shift)

    def run(key):
        part, log = part_file(label, key), log_path(f"s1-{label}", key)
        with heavy(f"s1:{label}"):
            temp = part.with_name(part.name + f".{os.getpid()}.tmp")
            build = (("-m", "tools.shifted_part", family, shift, temp) if shift else
                     ("src/build_meshes.py", "--family", family, "--part", temp))
            if python(*build, log=log):
                return "error", log, {}
            replace(temp, part)                       # another worktree's S2 may be reading the part
        return "pass", log, {str(part): files(part)[str(part)]}
    return Stage(f"s1:{label}", lambda: part_inputs(family, shift), run)


def shifted_parts(shift):
    """Every family's part on the world raised `shift` metres, built first where its key has no reusable pass: the asset
    tests' shifted runs merge their payload from these."""
    paths = []
    for family in families():
        stage = s1_stage(family, shift)
        if not execute(stage):
            raise RuntimeError(f"{stage.name} did not pass: see its log under {LOGS}")
        paths.append(part_file(part_label(family, shift), key_of(stage.inputs())))
    return paths


def s2_stage():
    def inputs_():
        return {"parts": {f: key_of(part_inputs(f)) for f in families()}, "code": code("src/build_meshes.py", "src/assemble.py"),
                "sources": files(*PAGE_SOURCES), "materials": tree("src/materials"),
                "data": files("build/data.js", "src/mesh_scene.json"), **environment()}

    def run(key):
        log = log_path("s2", key)
        with tempfile.TemporaryDirectory(prefix="hog-s2-") as folder, heavy("s2"):
            for f in families():
                part = part_file(f, key_of(part_inputs(f)))
                if not part.is_file():
                    with open(log, "a", encoding="utf-8") as out:
                        out.write(f"no {f} part for its current inputs ({part}): run doit s1 first\n")
                    return "error", log, {}
                shutil.copyfile(part, Path(folder) / f"{f}.json")
            merged, page = Path(folder) / "meshes.js", Path(folder) / "page.html"
            if python("src/build_meshes.py", "--parts", folder, merged, log=log) or \
                    python("src/assemble.py", page, "--meshes", merged, log=log):
                return "error", log, {}
            for target, built in ((MESHES_JS, merged), (PAGE, page)):     # only once both are built
                if not target.is_file() or built.read_bytes() != target.read_bytes():
                    write_atomic(target, built.read_bytes())
        return "pass", log, files("build/meshes.js", "hogwarts-flyover.html")
    return Stage("s2", inputs_, run)


def stale_stage():
    """The staleness gate: the generated files in HEAD equal what the stages just rebuilt from HEAD's sources. Any
    change outside the generated files, untracked files included, fails it: the rebuild would not be HEAD's."""
    def run(key):
        log = log_path("stale", key)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT,
                                capture_output=True, text=True)
        diff = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", *GENERATED], cwd=ROOT, capture_output=True,
                              text=True)
        failed = [result for result in (status, diff) if result.returncode]
        if failed:                                   # no output then means git could not look, not a clean tree
            detail = "".join(result.stderr for result in failed).strip()
            log.write_text(f"git failed, so the gate cannot tell:\n{detail}\n", encoding="utf-8")
            print(f"  git failed: {detail.splitlines()[0] if detail else 'no message'}", flush=True)
            return "error", log, {}
        sources = [line[3:] for line in status.stdout.splitlines() if line[3:] not in GENERATED]
        changed = diff.stdout.split()
        lines = ([f"sources differ from HEAD (commit them with their rebuilt outputs, or remove them): {sources}"]
                 if sources else [])
        lines += [f"{p} differs from HEAD's after a rebuild: commit the rebuilt file with its sources" for p in changed]
        log.write_text("\n".join(lines or ["the generated files match a rebuild of HEAD"]) + "\n", encoding="utf-8")
        for line in lines:
            print("  " + line, flush=True)
        return ("fail" if changed or sources else "pass"), log, {}
    return Stage("stale", lambda: {"head": head(), **files(*GENERATED)}, run, cached=False)


# ---------------------------------------------------------------- S3-S4: test suites through tools.testrun
def suite_inputs(suite, gpu=None, shift=None):
    from tools.testrun import SUITES                 # noqa: PLC0415 (keeps doit's task listing light)
    spec = SUITES[suite]
    base = ROOT / spec.get("dir", "src")
    tests = sorted({p for pattern in spec.get("patterns", ()) for p in base.glob(pattern)}
                   | {SRC / f"{m}.py" for m in spec.get("modules", ())})
    parts = {"suite": suite, "shift": shift, "tests": code(*(rel(t) for t in tests), "tools/testrun.py"),
             "generated": files(*GENERATED), "data": files(*ROOT_DATA), **environment()}
    if suite in ("assets", "uncached", "checks", "site", "tools"):
        parts["src"] = tree("src")                   # CPU gates: any source, builders loaded by name included
    if suite == "assets" and shift:
        parts["parts"] = {f: key_of(part_inputs(f, shift)) for f in families()}    # its payload is merged from these
    if suite == "assets":                            # test_assets_grounds reads S0's input by path
        parts["s0_input"] = inputs.spec("data-before-railway-x")["sha256"]
    if suite == "tools":
        parts["tools"] = tree("tools")               # its tests run tools and src scripts as subprocesses, and copy both
        from tools.blender import fetch              # noqa: PLC0415
        parts["blender"] = fetch.VERSION if fetch.cached() else None    # test_blender's Blender tests skip without it
    if suite in ("render", "smoke"):
        parts["page_code"] = {**files(*PAGE_SOURCES, *RENDER_TESTS), **tree("src/materials")}
    if suite == "checks":
        from tools import glslang                   # noqa: PLC0415
        parts["glslang"] = glslang.VERSION
    if suite == "site":                              # test_site also reads these by path
        parts["site"] = {**tree("tools/site"), **files("tools/inputs.lock.json", "tools/original-intermediates.lock.json",
                                                       "tools/stages.py")}
    if gpu:
        parts["browser"] = browser(gpu)
    return parts


def suite_stage(name, suite, gpu=None, shift=None):
    def run(key):
        args = ["-m", "tools.testrun", suite, *(["--gpu", gpu] if gpu else []), *(["--shift", shift] if shift else [])]
        log, started = log_path(name, key), time.time()
        status = python(*args, log=log)
        report = LOGS / f"{suite}{f'{float(shift):+g}m' if shift else ''}-{gpu or 'cpu'}-{page_tag()}.json"
        found = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
        if status not in (0, 1, 3) or found.get("started", 0) < started:
            return "error", log, {}
        return found["verdict"], Path(found["log"]), {}
    return Stage(name, lambda: suite_inputs(suite, gpu, shift), run)


# ---------------------------------------------------------------- S5-S6: review sheets and timing
def review_stage(family, gpu="nvidia"):
    def inputs_():
        return {"family": family, "page": files("hogwarts-flyover.html"),
                "code": code("tools/review/sheet.py", f"tools/review/{family}.py"), "browser": browser(gpu), **environment()}

    def run(key):
        log, out = log_path(f"s5-{family}", key), ROOT / "build" / "review" / f"{family}-{page_tag()}.png"
        # the sheet takes its GPU's lock and then registers its heavy work itself (locks.gpu_work): registered work
        # around it would hold up a timing run while the sheet waits for the GPU
        status = python("-m", "tools.review.sheet", family, "--gpu", gpu, "--out", out, log=log)
        return VERDICTS.get(status, "fail"), log, files(out) if status == 0 else {}
    return Stage(f"s5:{family}", inputs_, run)


def timing_stage(name, gpu, command, baseline=True):
    """A timing run alone on the machine: it holds the quiet lock, which waits for heavy work to end. With `baseline` it
    is blocked while the frozen baseline carries other data than the page: the tour's camera follows the ground and the
    trees, so the two would show different views in every window."""
    def inputs_():
        found = {"page": files("hogwarts-flyover.html"),
                 "code": code("src/mesh_probe.py", "tools/timing/passes.py"), "browser": browser(gpu), **environment()}
        if baseline:
            from tools.site import baseline as frozen     # noqa: PLC0415 (keeps doit's task listing light)
            frozen.check_baseline(inputs.require("baseline-x"), ROOT / "hogwarts-flyover.html")
            found["baseline"] = inputs.spec("baseline-x")["sha256"]
        return found

    def run(key):
        log = log_path(name, key)
        with locks.quiet(f"{name} in {ROOT.name}") as token:
            status = python(*command(), log=log, env={locks.TOKEN: token})
        return VERDICTS.get(status, "fail"), log, {}
    return Stage(name, inputs_, run, cached=False)


def probe_command(gpu):
    out = LOGS / f"timing-{gpu}-{page_tag()}.json"         # probe-*.json are the probe tests' reports
    return ["src/mesh_probe.py", "--gpu", gpu, "--baseline", inputs.require("baseline-x"), "--out", out]


def heavy(label):
    return locks.heavy(f"{label} in {ROOT.name}")


def summary(names):
    """name, verdict, seconds, log of each stage's latest ledger entry in this worktree."""
    rows = []
    for name in names:
        mine = [e for e in history(name) if e["worktree"] == str(ROOT)]
        if mine:
            e = mine[-1]
            rows.append((name, e["verdict"], e["seconds"], time.strftime("%Y-%m-%d %H:%M", time.localtime(e["time"])),
                         e.get("log") or e.get("note") or ""))
    return rows
