"""doit tasks for the pipeline stages (tools/stages.py); REBUILD.md says when to run which. From the repository root,
with the venv's doit (.venv\\Scripts\\doit.exe):

    doit list                  # the tasks
    doit                       # the CPU gates (s3), rebuilding first whatever they read that is out of date
    doit s2                    # the generated files: manifest, s0, one part per family, then the payload and page
    doit stale                 # the staleness gate: HEAD's generated files equal a rebuild of HEAD
    doit uncached              # the payload rebuilt without the part cache equals S2's, for trunk merges
    doit -n 4 s4               # rendering and UI tests on both GPUs, one worker per suite (REBUILD.md says why)
    doit smoke:nvidia          # the smoke subset on NVIDIA, for lane milestones
    doit s5:castle             # a family's review sheet
    doit s6:amd                # frame-time probe, alone on the machine (it waits for the quiet lock)
    doit passes_candidate:amd  # GPU time per pass, landscape against integrated, alone on the machine
    doit ledger                # the latest verdict of each stage in this worktree

A stage whose inputs have a passing verdict in the machine-wide ledger is reused rather than rerun.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import stages  # noqa: E402
from tools.paths import ROOT  # noqa: E402

DOIT_CONFIG = {"default_tasks": ["s3"], "verbosity": 2, "dep_file": "build/.doit.json", "backend": "json",
               "par_type": "thread"}
GPUS = ("amd", "nvidia")
# the datum shifts of the asset tests: site X raises the lake 131.0 m and the castle platform 131.6 m, and an exact shift
# must be a whole number of every height layer's quantum (down to 0.5 m)
SHIFT_M = 140
SHEETS = sorted(p.stem for p in (ROOT / "tools" / "review").glob("*.py") if p.stem not in ("__init__", "sheet"))


def run(stage):
    return {"actions": [(stages.execute, [stage])], "verbosity": 2}


def task_manifest():
    """Generate src/mesh_scene.json from tools/manifest and check its quotations."""
    return {**run(stages.manifest_stage()), "doc": task_manifest.__doc__}


def task_s0():
    """S0: build/data.js from the pinned data.js of before the railway regrade (fix_berth, fix_harbour, fix_railway)."""
    return {**run(stages.s0_stage()), "task_dep": ["manifest"], "doc": task_s0.__doc__}


def task_s1():
    """S1: one payload part per family, in parallel, kept in the machine-wide cache. The families are read from the
    manifest when the task runs, after the manifest task, so a family the manifest has just gained gets its part."""
    def build():
        with ThreadPoolExecutor(max_workers=4) as pool:
            return all(list(pool.map(stages.execute, [stages.s1_stage(f) for f in stages.families()])))
    return {"actions": [build], "task_dep": ["s0"], "verbosity": 2, "doc": task_s1.__doc__}


def task_s2():
    """S2: build/meshes.js from the parts, then hogwarts-flyover.html."""
    return {**run(stages.s2_stage()), "task_dep": ["s1"], "doc": task_s2.__doc__}


def task_stale():
    """The staleness gate: the committed generated files equal a fresh rebuild."""
    return {**run(stages.stale_stage()), "task_dep": ["s2"], "doc": task_stale.__doc__}


def task_uncached():
    """The payload rebuilt from the sources without the part cache equals S2's: run at trunk merges."""
    return {**run(stages.suite_stage("uncached", "uncached")), "task_dep": ["s2"], "doc": task_uncached.__doc__}


def task_s3():
    """S3: the CPU gates (asset tests, also on a world raised and lowered 140 m; page checks; site checks; tool tests)."""
    for name, suite, shift in (("assets", "assets", None), ("assets+140", "assets", SHIFT_M), ("assets-140", "assets", -SHIFT_M),
                               ("checks", "checks", None), ("site", "site", None), ("tools", "tools", None)):
        yield {**run(stages.suite_stage(f"s3:{name}", suite, shift=shift)), "name": name, "task_dep": ["s2"]}


def task_s4():
    """S4: the rendering and UI tests on each GPU."""
    for suite in ("render", "ui"):
        for gpu in GPUS:
            yield {**run(stages.suite_stage(f"s4:{suite}:{gpu}", suite, gpu)), "name": f"{suite}:{gpu}", "task_dep": ["s2"]}


def task_smoke():
    """The @smoke rendering tests, for lane milestones (NVIDIA)."""
    for gpu in GPUS:
        yield {**run(stages.suite_stage(f"smoke:{gpu}", "smoke", gpu)), "name": gpu, "task_dep": ["s2"]}


def task_s5():
    """S5: labelled review sheets on NVIDIA, under build/review."""
    for family in SHEETS:
        yield {**run(stages.review_stage(family)), "name": family, "task_dep": ["s2"]}


def task_s6():
    """S6: the frame-time probe against the frozen baseline, alone on the machine (never reused)."""
    for gpu in GPUS:
        stage = stages.timing_stage(f"s6:{gpu}", gpu, lambda gpu=gpu: stages.probe_command(gpu))
        yield {**run(stage), "name": gpu, "task_dep": ["s2"]}


def task_passes():
    """GPU time per render pass for the baseline and the page, alone on the machine (never reused)."""
    for gpu in GPUS:
        stage = stages.timing_stage(f"passes:{gpu}", gpu,
                                    lambda gpu=gpu: ["-m", "tools.timing.passes", "--gpu", gpu, "--mode", "pages"])
        yield {**run(stage), "name": gpu, "task_dep": ["s2"]}


def task_passes_candidate():
    """GPU time per render pass and CPU time of the mesh draws, landscape against integrated within the page alone:
    no baseline needed (alone on the machine, never reused)."""
    for gpu in GPUS:
        stage = stages.timing_stage(f"passes_candidate:{gpu}", gpu,
                                    lambda gpu=gpu: ["-m", "tools.timing.passes", "--gpu", gpu, "--mode", "candidate"],
                                    baseline=False)
        yield {**run(stage), "name": gpu, "task_dep": ["s2"]}


def task_ledger():
    """The latest verdict of each stage in this worktree."""
    def show():
        names = ["manifest", "s0", *(f"s1:{f}" for f in stages.families()), "s2", "stale", "uncached",
                 *(f"s3:{n}" for n in ("assets", "assets+140", "assets-140", "checks", "site", "tools")),
                 *(f"s1:{stages.part_label(f, s)}" for s in (SHIFT_M, -SHIFT_M) for f in stages.families()),
                 *(f"s4:{s}:{g}" for s in ("render", "ui") for g in GPUS), *(f"smoke:{g}" for g in GPUS),
                 *(f"s5:{f}" for f in SHEETS), *(f"s6:{g}" for g in GPUS), *(f"passes:{g}" for g in GPUS),
                 *(f"passes_candidate:{g}" for g in GPUS)]
        for name, verdict, seconds, when, where in stages.summary(names):
            print(f"{name:18s} {verdict:8s} {seconds:8.1f} s  {when}  {where}")
    return {"actions": [show], "verbosity": 2, "uptodate": [False]}
