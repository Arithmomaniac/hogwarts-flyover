"""The frame-time probe's windows moved onto a changed tour: each window keeps its place on the path. Its start time on
the old page puts the camera at some point; its new time is when the new tour passes nearest that point, searched
within 15% of the tour around the same share of it, so a tour that passes a place twice keeps each window's pass.

    python -X utf8 -m tools.timing.windows OLD_PAGE [NEW_PAGE] [--gpu nvidia|amd]

It prints mesh_probe.WINDOWS as they should read for the new page (the worktree's page unless NEW_PAGE says otherwise),
with each window's miss: how far the new tour passes from the old window's point. WINDOWS must still hold the old page's
times. The run fails, and prints no WINDOWS, when the new tour passes no nearer than MAX_MISS_M to a window's point: the
tour has left that place, or WINDOWS already hold the new page's times.
"""
import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
from playwright.async_api import async_playwright

from tools import locks
from tools.paths import PAGE
from tools.review import tour
from mesh_probe import GPU_ARGS, WINDOWS

SEARCH = 0.15
MAX_MISS_M = 10.0               # a sample every STEP_S at the tour's top speed, 68 m/s, lies within 1.7 m of its path


def remap(windows, old_positions, old_total, new_positions, new_total, step=tour.STEP_S):
    """{window: (new start time, miss in metres)} for `windows` ({window: old start time}) between tours sampled every
    `step` seconds."""
    old_positions, new_positions = np.asarray(old_positions, float), np.asarray(new_positions, float)
    times = np.arange(len(new_positions)) * step
    out = {}
    for name, t in windows.items():
        x, _, z = old_positions[min(int(round(t / step)), len(old_positions) - 1)]
        near = np.abs(times - t / old_total * new_total) < SEARCH * new_total
        d = np.where(near, np.hypot(new_positions[:, 0] - x, new_positions[:, 2] - z), np.inf)
        j = int(np.argmin(d))
        out[name] = (round(float(times[j]), 2), float(d[j]))
    return out


async def track(pw, path, gpu):
    browser = await pw.chromium.launch(channel="msedge", headless=True, args=GPU_ARGS[gpu])
    page = await (await browser.new_context(offline=True)).new_page()
    await page.goto(Path(path).resolve().as_uri() + "?t=0&pause=1&hud=0&scale=0.25")
    await page.wait_for_function("window.__hog && (__hog.error || (__hog.ready && __hog.frames >= 10))", timeout=300000)
    total, positions = await page.evaluate(tour.TRACK, tour.STEP_S)
    await browser.close()
    return total, positions


async def tracks(old, new, gpu):
    async with async_playwright() as pw:
        return await track(pw, old, gpu), await track(pw, new, gpu)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path, nargs="?", default=PAGE)
    parser.add_argument("--gpu", choices=sorted(GPU_ARGS), default="nvidia")
    args = parser.parse_args(argv)
    with locks.gpu_work(args.gpu, "probe windows onto a changed tour"):
        (old_total, old_positions), (new_total, new_positions) = asyncio.run(tracks(args.old, args.new, args.gpu))
    moved = remap(WINDOWS, old_positions, old_total, new_positions, new_total)
    print(f"tour {old_total:.1f} s -> {new_total:.1f} s")
    for name, (t, miss) in moved.items():
        print(f"  {name:14s} {WINDOWS[name]:7.2f} s -> {t:7.2f} s (miss {miss:.2f} m)")
    if lost := [name for name, (_, miss) in moved.items() if miss > MAX_MISS_M]:
        print(f"the new tour passes no nearer than {MAX_MISS_M:g} m to where {', '.join(lost)} started: set them by hand,"
              " or run this against the page whose times WINDOWS hold")
        return 1
    print("WINDOWS = {" + ", ".join(f'"{name}": {t:g}' for name, (t, _) in moved.items()) + "}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
