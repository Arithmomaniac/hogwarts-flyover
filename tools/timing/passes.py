"""GPU time per render pass (shader program) through timer queries, for the frame-time probe's windows: where time goes.

    python -X utf8 -m tools.timing.passes --gpu amd|nvidia [--mode pages|candidate] [--out PATH]

`pages` times the frozen baseline (input baseline-x) and the candidate page for each probe configuration, in two browser
sessions that open them in opposite orders, since a second live page runs slower on the AMD iGPU;
`candidate` alternates landscape and integrated rounds within the candidate page alone. Timing runs only in a quiet
window: run it through the stage runner, which holds the quiet lock.
"""
import argparse
import asyncio
import contextlib
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

from tools import inputs, locks
from tools.paths import LOGS, PAGE, page_tag
from tools.site import baseline as frozen
import mesh_probe as mp

PASSES = r"""
  const passExt = gl.getExtension("EXT_disjoint_timer_query_webgl2");
  const passNames = new Map();
  for (const [n, p] of Object.entries({ p1: prog, up, taa, tree: treeP, cone: coneP, sun: sunP, far: farP, classify: clsP,
      shadow: shP, shade: p3P, ...(typeof meshP === "undefined" ? {} : { mesh: meshP, meshDepth: meshDP, resolve: resP,
      meshShadow: shDepthP, refl: reflP, glass: glassP }), ...(typeof lakeP === "undefined" ? {} : { lake: lakeP }) }))
    passNames.set(p, n);
  let passOn = false, passActive = null, passFrame = [];
  const passRows = [], passUse = gl.useProgram.bind(gl);
  const passEnd = () => { if (passActive) { gl.endQuery(passExt.TIME_ELAPSED_EXT); passFrame.push(passActive); passActive = null; } };
  gl.useProgram = (p) => {
    if (passOn) { passEnd(); const q = gl.createQuery(); gl.beginQuery(passExt.TIME_ELAPSED_EXT, q); passActive = { q, name: passNames.get(p) || "other" }; }
    return passUse(p);
  };
  const passRender = render;
  render = function (pos, f) { const L = passRender(pos, f); if (passOn) { passEnd(); passRows.push(passFrame); passFrame = []; } return L; };
  // CPU time of the mesh draws (submission, not GPU work): what batching or culling could save
  const passCpu = {}, passTime = (name, f) => (...a) => {
    if (!passOn) return f(...a);
    const t = performance.now(), r = f(...a);
    passCpu[name] = (passCpu[name] || 0) + performance.now() - t;
    return r;
  };
  if (typeof meshShadows === "function") meshShadows = passTime("meshShadow", meshShadows);
  if (typeof drawMeshDepth === "function") drawMeshDepth = passTime("meshDepth", drawMeshDepth);
  if (typeof drawMeshes === "function") drawMeshes = passTime("mesh", drawMeshes);
  if (typeof drawReflections === "function") drawReflections = passTime("refl", drawReflections);
  if (typeof drawGlass === "function") drawGlass = passTime("glass", drawGlass);
  window.__pass = {
    start() { passOn = true; passRows.length = 0; for (const k in passCpu) delete passCpu[k]; gl.getParameter(passExt.GPU_DISJOINT_EXT); },
    async collect() {
      passOn = false;
      for (let i = 0; i < 400; i++) {
        await new Promise((r) => setTimeout(r, 5));
        if (passRows.every((fr) => fr.every((x) => gl.getQueryParameter(x.q, gl.QUERY_RESULT_AVAILABLE)))) break;
      }
      const disjoint = gl.getParameter(passExt.GPU_DISJOINT_EXT), ms = {}, cpu = {};
      for (const fr of passRows) for (const x of fr) { ms[x.name] = (ms[x.name] || 0) + gl.getQueryParameter(x.q, gl.QUERY_RESULT) / 1e6; gl.deleteQuery(x.q); }
      for (const k in ms) ms[k] /= passRows.length;
      for (const k in passCpu) cpu[k] = passCpu[k] / passRows.length;
      return { frames: passRows.length, disjoint, ms, cpu };
    },
  };
"""
PAGE_WINDOWS = ["castle", "trees-65", "lake-ship", "hogsmeade"]
CANDIDATE_CONFIGS = {"landscape": (False, 0), "integrated": (True, 0)}


def instrumented(source, target):
    mp.instrument(source, target)
    text = target.read_text(encoding="utf-8")
    target.write_text(text.replace(mp.API, PASSES + mp.API, 1), encoding="utf-8", newline="\n")


async def check_adapter(page, gpu):
    if gpu not in (await page.evaluate("document.getElementById('gpu-info').textContent")).lower():
        raise RuntimeError(f"the page is not running on the {gpu} adapter")


@contextlib.asynccontextmanager
async def browser_session(gpu, pages):
    """A browser on `gpu` with `pages`, (role, instrumented page) pairs, opened in that order and each checked to run on
    that GPU: {role: page}."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="msedge", headless=True, chromium_sandbox=True, args=mp.GPU_ARGS[gpu])
        try:
            context = await browser.new_context(viewport=mp.VIEWPORT, device_scale_factor=1, offline=True)
            opened = {}
            for role, path in pages:
                page = await context.new_page()
                await mp.open_page(page, path, meshes=role == "candidate")
                await check_adapter(page, gpu)
                opened[role] = page
            yield opened
        finally:
            await browser.close()


async def time_session(pages):
    """Each window's GPU time per pass, mesh-draw CPU time and wall time per frame, for every probe configuration."""
    out = {}
    for window in PAGE_WINDOWS:
        frames = mp.schedule(mp.WINDOWS[window], window in mp.PANNED)
        for name, (role, meshes, dbg) in mp.CONFIGS.items():
            page, settings = pages[role], {**mp.SETTINGS, "dbg": dbg, "meshes": meshes}
            await mp.block(page, settings, frames)                       # warm caches
            await page.evaluate("__pass.start()")
            b = await mp.block(page, settings, frames)
            r = await page.evaluate("__pass.collect()")
            r["wallMs"] = sum(b["batches"][mp.PREROLL // mp.IN_FLIGHT:]) / mp.COUNT
            out[f"{window}|{name}"] = r
    return out


def geometric_mean(values):
    return math.exp(statistics.fmean(map(math.log, values))) if all(v > 0 for v in values) else statistics.fmean(values)


def combine(runs):
    """One record per window and configuration from the sessions' records, each time the geometric mean of theirs: a
    factor that slows whichever page was opened second then cancels between the pages when both share it."""
    out = {}
    for key in runs[0]:
        rs = [run[key] for run in runs]
        out[key] = {"frames": sum(r["frames"] for r in rs), "disjoint": any(r["disjoint"] for r in rs),
                    **{part: {k: geometric_mean([r[part].get(k, 0.0) for r in rs])
                              for k in set().union(*(r[part] for r in rs))} for part in ("ms", "cpu")},
                    "wallMs": geometric_mean([r["wallMs"] for r in rs]), "sessions": rs}
    return out


def report(gpu, out):
    for window in PAGE_WINDOWS:
        print(f"== {gpu} {window}")
        rows = {c: out[f"{window}|{c}"] for c in mp.CONFIGS}
        names = sorted({k for r in rows.values() for k in r["ms"]}, key=lambda k: -rows["integrated"]["ms"].get(k, 0))
        print(f"{'pass':12}" + "".join(f"{c:>11}" for c in mp.CONFIGS))
        for k in names:
            print(f"{k:12}" + "".join(f"{rows[c]['ms'].get(k, 0):11.2f}" for c in mp.CONFIGS))
        print(f"{'gpu sum':12}" + "".join(f"{sum(rows[c]['ms'].values()):11.2f}" for c in mp.CONFIGS))
        print(f"{'wall':12}" + "".join(f"{rows[c]['wallMs']:11.2f}" for c in mp.CONFIGS),
              " disjoint" if any(r["disjoint"] for r in rows.values()) else "")
        print(f"{'mesh cpu':12}" + "".join(f"{sum(rows[c]['cpu'].values()):11.2f}" for c in mp.CONFIGS))


async def time_pages(gpu, baseline, candidate, session=browser_session):
    """Every probe configuration's times in PAGE_WINDOWS, for the baseline and the candidate. On the AMD iGPU a second
    live page runs slower whatever it is, so, as the probe does, each window is timed in one browser session per order
    of mesh_probe.LOAD_ORDERS, and `combine` takes each time's geometric mean across them. A slowdown that differs
    between the two pages does not cancel: the square root of their ratio remains when their times are compared. The
    pass timer only reports; S6 judges a window by both load orders' bounds (mesh_probe.summarize)."""
    with tempfile.TemporaryDirectory() as folder:
        targets = {}
        for role, source in (("baseline", baseline), ("candidate", candidate)):
            targets[role] = Path(folder) / f"{role}.html"
            instrumented(source, targets[role])
        runs = []
        for order in mp.LOAD_ORDERS:
            async with session(gpu, [(role, targets[role]) for role in order]) as pages:
                runs.append(await time_session(pages))
    out = combine(runs)
    report(gpu, out)
    return out


async def time_candidate(gpu, candidate):
    out = {}
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "candidate.html"
        instrumented(candidate, target)
        async with async_playwright() as p:
            browser = await p.chromium.launch(channel="msedge", headless=True, chromium_sandbox=True, args=mp.GPU_ARGS[gpu])
            page = await (await browser.new_context(viewport=mp.VIEWPORT, device_scale_factor=1, offline=True)).new_page()
            await mp.open_page(page, target, meshes=True)
            await check_adapter(page, gpu)
            for window in mp.WINDOWS:
                frames = mp.schedule(mp.WINDOWS[window], window in mp.PANNED)
                runs = {c: [] for c in CANDIDATE_CONFIGS}
                for rnd in range(4):
                    for name in (list(CANDIDATE_CONFIGS) if rnd % 2 == 0 else list(CANDIDATE_CONFIGS)[::-1]):
                        meshes, dbg = CANDIDATE_CONFIGS[name]
                        settings = {**mp.SETTINGS, "dbg": dbg, "meshes": meshes}
                        await mp.block(page, settings, frames)
                        await page.evaluate("__pass.start()")
                        await mp.block(page, settings, frames)
                        runs[name].append(await page.evaluate("__pass.collect()"))
                med = {c: {k: statistics.median(r["ms"].get(k, 0) for r in rs) for k in set().union(*(r["ms"] for r in rs))}
                       for c, rs in runs.items()}
                cpu = {c: {k: statistics.median(r["cpu"].get(k, 0) for r in rs) for k in set().union(*(r["cpu"] for r in rs))}
                       for c, rs in runs.items()}
                tot = {c: statistics.median(sum(r["ms"].values()) for r in rs) for c, rs in runs.items()}
                ratio = [sum(a["ms"].values()) / sum(b["ms"].values()) for a, b in zip(runs["integrated"], runs["landscape"])]
                out[window] = {"median": med, "cpu": cpu, "total": tot, "ratioToLandscape": ratio}
                print(f"{gpu} {window:14} GPU ms landscape {tot['landscape']:6.2f}  integrated {tot['integrated']:6.2f}"
                      f" | integrated/landscape {statistics.median(ratio):.3f} [{min(ratio):.2f}-{max(ratio):.2f}]", flush=True)
                mesh = sorted(med["integrated"].items(), key=lambda kv: -kv[1])
                print(f"    integrated GPU ms: " + ", ".join(f"{k} {v:.2f}" for k, v in mesh), flush=True)
                print(f"    integrated CPU ms: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(cpu["integrated"].items())),
                      flush=True)
            await browser.close()
    return out


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", choices=sorted(mp.GPU_ARGS), required=True)
    parser.add_argument("--mode", choices=("pages", "candidate"), default="candidate")
    parser.add_argument("--candidate", type=Path, default=PAGE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    locks.require_quiet("tools.timing.passes")
    if args.mode == "pages":
        try:
            baseline = inputs.require("baseline-x")
            frozen.check_baseline(baseline, args.candidate)
        except inputs.Blocked as e:
            print(f"blocked: {e}", file=sys.stderr)
            return 3
        result = asyncio.run(time_pages(args.gpu, baseline, args.candidate))
    else:
        result = asyncio.run(time_candidate(args.gpu, args.candidate))
    out = args.out or LOGS / f"passes-{args.mode}-{args.gpu}-{page_tag(args.candidate)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
