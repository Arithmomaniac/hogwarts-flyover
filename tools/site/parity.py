"""Parity of the landscape: a candidate page with its meshes off against the frozen baseline at the site, at several tour
times and suns.

    python -X utf8 -m tools.site.parity OUT_DIR [--baseline PAGE] [--candidate PAGE] [--gpu nvidia|amd]

The baseline defaults to the input baseline-x, the frozen renderer re-frozen at site X (tools.site.baseline), which must
carry the candidate's data; the candidate defaults to the page. Both render at full internal resolution, never the
page's adaptive scale. For each view it prints the per-pixel difference, the largest over the colour channels: its mean,
99th percentile and maximum, and the share of pixels over DIFF_LEVEL. A view with more than DIFF_SHARE of them over
DIFF_LEVEL also gets an image of the difference in OUT_DIR. It runs in headless Edge under the heavy and GPU locks.
Since the meshes replaced the raymarched objects, the baseline still draws objects, with their shadows and reflections,
that the candidate draws in no pass, so those pixels differ by design.
"""
import argparse
import asyncio
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

from tools import inputs, locks
from tools.paths import PAGE
from tools.site import baseline as frozen
from tools.site.baseline import same_data
from mesh_probe import GPU_ARGS

VIEWS = ((0, 18.5), (40, 18.5), (60, 18.5), (95, 18.5), (140, 18.5), (60, 12.0), (60, 20.0), (60, 7.0))  # (s, hour)
QUERY = "?t=0&pause=1&hud=0&scale=1&meshes=0"        # the landscape alone, at a fixed internal scale
W, H, SETTLE_FRAMES = 1280, 720, 160
DIFF_LEVEL, DIFF_SHARE = 8, 0.001


def difference(a, b):
    """(per-pixel difference image, its mean, 99th percentile, maximum, share over DIFF_LEVEL) of two RGB images."""
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).max(axis=2)
    return d, float(d.mean()), float(np.percentile(d, 99)), int(d.max()), float(np.mean(d > DIFF_LEVEL))


async def shots(path, gpu):
    out = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(channel="msedge", headless=True, args=GPU_ARGS[gpu])
        page = await (await browser.new_context(viewport={"width": W, "height": H}, offline=True)).new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(Path(path).resolve().as_uri() + QUERY)
        await page.wait_for_function("window.__hog && (__hog.error || (__hog.ready && __hog.frames >= 10))",
                                     timeout=300000)
        failure = await page.evaluate("__hog.error || null")
        if failure:
            raise RuntimeError(f"{path}: {failure}")
        total = await page.evaluate("__hog.total")
        for t, hour in VIEWS:
            frames = await page.evaluate("([t, h]) => { __hog.set(t); __hog.tod(h); return __hog.frames; }",
                                         [min(t, total - 1), hour])
            await page.wait_for_function(f"__hog.frames >= {frames + SETTLE_FRAMES}", timeout=180000)
            out.append(Image.open(io.BytesIO(await page.screenshot())).convert("RGB"))
        await browser.close()
        if errors:
            raise RuntimeError(f"{path}: page errors {errors}")
    return out


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path, default=PAGE)
    parser.add_argument("--gpu", choices=sorted(GPU_ARGS), default="nvidia")
    args = parser.parse_args(argv)
    baseline = args.baseline or inputs.require(frozen.NAME)
    if args.baseline is None and not same_data(baseline, args.candidate):
        parser.error(f"{baseline} carries other data than {args.candidate}: make it again with"
                     f" python -X utf8 -m tools.site.baseline, and pin it")
    args.out.mkdir(parents=True, exist_ok=True)
    with locks.gpu_work(args.gpu, "site parity"):
        base = asyncio.run(shots(baseline, args.gpu))
        cand = asyncio.run(shots(args.candidate, args.gpu))
    for (t, hour), a, b in zip(VIEWS, base, cand):
        d, mean, p99, peak, share = difference(a, b)
        print(f"t={t:<4} h={hour:<5} mean {mean:6.3f}  p99 {p99:5.1f}  max {peak:4d}  >{DIFF_LEVEL}: {share:.4f}")
        if share > DIFF_SHARE:
            Image.fromarray(np.clip(d * 8, 0, 255).astype(np.uint8)).save(args.out / f"diff-t{t:g}-h{hour:g}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
