"""Build a castle site's data from its pinned inputs, and keep the site's S0 input.

    python -X utf8 -m tools.site.build SITE [--dem MODEL]       # SITE: x or scardroy; MODEL: the site's "dem" by default

It stages the site's inputs (tools.site.stage), then runs the data pipeline: src/features.py, build_final.py,
build_extras.py, build_regional.py, pack.py and fix_berth.py. The resulting build/data.js, the data of before the railway
regrade, is kept in build/inputs/ as the site's S0 input and checked against tools/inputs.lock.json; S0 (doit s0) starts
from it. The FINISH steps, fix_harbour.py and fix_railway.py, then finish build/data.js as S0 does. A build on another elevation model than the site's
(--dem, to compare models) leaves the S0 input as it was. The site's sources must be fetched first
(python -X utf8 -m tools.site.sources fetch); the build takes a minute or two.
"""
import argparse
import shutil
import subprocess
import sys

from tools import inputs, locks
from tools.paths import BUILD, ROOT, sha256
from tools.site import stage as STAGE
from tools.site import terrain

S0_INPUTS = {"x": "data-before-railway-x"}         # Scardroy's pinned data-before-railway predates extras.place and .ref
STEPS = ("features", "build_final", "build_extras", "build_regional", "pack", "fix_berth")
FINISH = ("fix_harbour", "fix_railway")              # after the S0 input, in S0's order (tools/stages.py)


def run(step):
    subprocess.run([sys.executable, "-X", "utf8", f"{step}.py"], cwd=ROOT / "src", check=True)


def build(site, dem=None):
    """Stage the site and run the pipeline, keeping the site's S0 input if it has one; (its name, the kept file), or
    (None, None). Data built on another elevation model than the site's is not kept: S0 starts from the site's own."""
    STAGE.stage(site, dem)
    (ROOT / "view").mkdir(exist_ok=True)            # the steps write preview images there
    for step in STEPS:
        run(step)
    own_model = dem in (None, STAGE.SRC.SITES[site]["dem"])
    name, kept = S0_INPUTS.get(site) if own_model else None, None
    if name:
        kept = BUILD / "inputs" / f"{name}.js"      # where tools.inputs looks for the input's "file" in a worktree
        kept.parent.mkdir(exist_ok=True)
        shutil.copyfile(BUILD / "data.js", kept)
    for step in FINISH:
        run(step)
    return name, kept


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site", choices=sorted(STAGE.SRC.SITES))
    parser.add_argument("--dem", choices=list(terrain.MODELS), help="the elevation model (default: the site's)")
    args = parser.parse_args(argv)
    with locks.heavy(f"tools.site.build {args.site}"):
        name, kept = build(args.site, args.dem)
    if name is None:
        held = S0_INPUTS.get(args.site)
        print(f"build/data.js is {args.site}'s data on {args.dem}; {held}, which the site's own model builds, is left as "
              f"it was" if held else f"build/data.js is {args.site}'s data; the site has no S0 input")
        return 0
    pin = f"size {kept.stat().st_size} and sha256 {sha256(kept)}"
    if name not in inputs.entries():
        print(f"{kept.relative_to(ROOT)} holds {name}, which tools/inputs.lock.json does not pin yet: {pin}")
        return 1
    found = inputs.problems(name, kept)
    if found:
        print(f"{kept.relative_to(ROOT)} differs from the pinned {name}: {found[0]}. If the change is intended, pin "
              f"{name} in tools/inputs.lock.json with {pin}.")
        return 1
    print(f"{kept.relative_to(ROOT)} holds the pinned {name}; build/data.js is the site's data")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
