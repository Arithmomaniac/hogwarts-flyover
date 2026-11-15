"""Continuous-motion evidence for the qualification: short stretches of the tour where the plan expects motion faults
(depth edges, glass, levels of detail, reflections, thin parts, near shadows), each frame rendered explicitly and
captured as the page displays it, then checked for ghosts, pops and shimmer.

    python -X utf8 -m tools.inspection.motion [--gpu nvidia|amd] [--only NAME,...] [--page PAGE] [--out DIR]

The page runs from a copy whose animation loop stops (instrumented, as src/mesh_probe.py's adapter does), so each step
renders exactly one frame at its tour time, with the history the temporal filter keeps from the frame before, as at 30
fps in the flyover; a screenshot between frames adds none. Each sequence starts PREROLL frames early to fill that
history, at 1920 x 1080, DPR 1 and internal scale 0.5, at the page's own hour (the tour's 18:30).

For each sequence it writes, under build/inspection/motion/: every frame (JPEG), an animated GIF at half size, a strip of
the eight frames round the largest change between neighbours, a chart of that change through the sequence with its pops
marked (changes: the mean absolute difference of consecutive frames in grey levels; a pop: a change that exceeds
POP_FACTOR times the sequence's median, recorded as the frame it shows in), and a JSON record.
"""
import argparse
import asyncio
import io
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools import locks
from tools.inspection import inventory
from tools.paths import PAGE, page_tag
from tools.review import sheet, tour

FPS, PREROLL, SECONDS = 30, 16, 3.0
W, H = 1920, 1080
POP_FACTOR, POP_FLOOR = 3.0, 0.5    # a pop changes more than this many times the median change, which is at least this
STRIP = 8
OUT = inventory.OUT_DIR / "motion"
LOOP, API, RENDER = "\n  function loop(now) {\n", "\n  window.__hog = {\n", "\n  function render(pos, f) {\n"
ADAPTER = r"""
  // tools/inspection/motion.py's adapter, only ever inserted into a temporary copy of the page
  let motionStop = false;
  const motionChannel = new MessageChannel(), motionWaiting = [];
  motionChannel.port1.onmessage = () => motionWaiting.shift()();
  const motionYield = () => new Promise((resolve) => { motionWaiting.push(resolve); motionChannel.port2.postMessage(0); });
  async function motionDone() {                    // a fence's status only changes between tasks: yield, unthrottled
    const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    try {
      for (;;) {
        const status = gl.clientWaitSync(fence, 0, 0);
        if (status === gl.WAIT_FAILED) throw new Error("GPU_COMPLETION");
        if (status !== gl.TIMEOUT_EXPIRED) break;
        await motionYield();
      }
    } finally { gl.deleteSync(fence); }
    if (gl.isContextLost() || gl.getError() !== gl.NO_ERROR) throw new Error("WEBGL_ERROR");
  }
  window.__motion = {
    async stop() { motionStop = true; await motionDone(); },
    reset() { histOk = false; jIdx = 0; },
    async frame(t) { const p = pathAt(t); simT = t; render(p.pos, p.fwd); await motionDone(); },
  };
"""
# name: (the waypoint whose pass it shows, which pass of a caption the tour repeats (1 the first; None for a caption it
# flies past once), seconds from that pass to its start, what it checks)
SEQUENCES = {
    "depth-edges": ("Gryffindor Tower", None, -1.5, "the castle's silhouette against the sky and the terrain"),
    "glass": ("Greenhouses", None, -1.5, "the greenhouses' glass and the plants behind it as the tour passes over"),
    "levels-of-detail": ("Hogsmeade", None, -1.5, "Hogsmeade's levels of detail as the tour comes in"),
    "reflections": ("The Lake", None, -1.5, "the castle's reflection in the lake"),
    "thin-parts": ("Quidditch pitch", 1, -1.5, "the goal hoops' poles as the tour drops past the nearer goal"),
    "near-shadows": ("Hagrid's hut", None, -1.5, "the hut's near shadows at its close pass"),
}


def pass_time(wps, times, caption, occurrence=None):
    """The tour time at which the flyover passes a captioned waypoint (wps and times as tools.review.tour gives them):
    the `occurrence`-th waypoint with that caption, counting from 1. A caption the tour repeats must say which pass,
    and one it lacks, or lacks that often, is refused, so a change to the tour cannot move a sequence silently."""
    hits = [t for w, t in zip(wps, times) if w[6] == caption]
    if occurrence is None and len(hits) > 1:
        raise ValueError(f"the tour passes {caption!r} {len(hits)} times: say which")
    k = (occurrence or 1) - 1
    if not 0 <= k < len(hits):
        raise ValueError(f"the tour passes {caption!r} {len(hits)} times, not {occurrence or 1}")
    return hits[k]


def schedule(start, seconds=SECONDS, fps=FPS, preroll=PREROLL):
    """The tour times of a sequence's frames: `preroll` frames before `start`, then `seconds` of frames at `fps`."""
    return [start + (i - preroll) / fps for i in range(preroll + round(seconds * fps))]


def instrumented(text):
    """The page's text with the motion adapter: its loop stops on request and __motion renders one frame at a time.
    Refused unless the loop, the render function and the API each occur once."""
    if any(text.count(anchor) != 1 for anchor in (LOOP, API, RENDER)):
        raise ValueError("the page lacks the motion adapter's anchors; the adapter needs updating for this page")
    return text.replace(LOOP, LOOP + "    if (motionStop) return;\n", 1).replace(API, ADAPTER + API, 1)


def changes(frames):
    """The mean absolute difference, in grey levels, between each frame and the one before."""
    return [float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32)))) for a, b in zip(frames, frames[1:])]


def pops(series, factor=POP_FACTOR, floor=POP_FLOOR):
    """Indices of the changes that exceed `factor` times the sequence's median change (at least `floor`)."""
    if not series:
        return []
    limit = factor * max(statistics.median(series), floor)
    return [k for k, v in enumerate(series) if v > limit]


def pop_frames(series):
    """The frames that pop: change k runs from frame k to frame k + 1, so it shows in frame k + 1."""
    return [k + 1 for k in pops(series)]


def chart(path, series, flagged, title):
    w, h, pad = 960, 300, 40
    img = Image.new("RGB", (w, h), (24, 24, 28))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 16)
    limit = POP_FACTOR * max(statistics.median(series), POP_FLOOR) if series else POP_FLOOR
    top = max(series + [limit]) * 1.1
    x = lambda k: pad + k * (w - 2 * pad) / max(1, len(series) - 1)          # noqa: E731
    y = lambda v: h - pad - v / top * (h - 2 * pad)                          # noqa: E731
    for k in range(0, w - 2 * pad, 12):                                      # the pop threshold, dashed
        draw.line([(pad + k, y(limit)), (pad + k + 6, y(limit))], fill=(150, 90, 90), width=1)
    draw.line([(x(k), y(v)) for k, v in enumerate(series)], fill=(120, 200, 255), width=2)
    for k in flagged:
        draw.ellipse([x(k) - 5, y(series[k]) - 5, x(k) + 5, y(series[k]) + 5], outline=(255, 90, 90), width=2)
    peak = max(series) if series else 0.0
    draw.text((pad, 8), f"{title}: change between frames, grey levels (median {statistics.median(series or [0]):.2f}, "
                        f"peak {peak:.2f}; dashed: pop threshold {limit:.2f}); pops circled", fill=(235, 235, 235),
              font=font)
    img.save(path)


async def record(page, name, times, out, preroll=PREROLL):
    """Render a sequence and write its frames, GIF, strip, chart and record; returns the record."""
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    await page.evaluate("() => __motion.reset()")
    grey, files = [], []
    for k, t in enumerate(times):
        await page.evaluate("(t) => __motion.frame(t)", t)
        if k < preroll:
            continue
        shot = Image.open(io.BytesIO(await page.screenshot(type="jpeg", quality=92))).convert("RGB")
        files.append(f"{name}/frame-{k - preroll:03d}.jpg")
        shot.save(out / files[-1], quality=92)
        grey.append(np.asarray(shot.convert("L").resize((480, 270))))
    series = changes(grey)
    flagged = pops(series)
    frames = [Image.open(out / f) for f in files]
    half = [f.resize((W // 2, H // 2)) for f in frames]
    half[0].save(out / f"{name}.gif", save_all=True, append_images=half[1:], duration=round(1000 / FPS), loop=0)
    peak = int(np.argmax(series)) + 1 if series else 0
    first = max(0, min(peak - STRIP // 2, len(frames) - STRIP))
    strip = Image.new("RGB", (4 * 480, 2 * 270))
    for i, f in enumerate(frames[first:first + STRIP]):
        strip.paste(f.resize((480, 270)), ((i % 4) * 480, (i // 4) * 270))
    strip.save(out / f"{name}-strip.png")
    chart(out / f"{name}-changes.png", series, flagged, name)
    return {"sequence": name, "times": times[preroll:], "frames": files, "changes": series,
            "pops": pop_frames(series), "strip": f"{name}-strip.png", "strip_first_frame": first,
            "gif": f"{name}.gif", "chart": f"{name}-changes.png"}


async def run(page_path, names, gpu, out):
    from playwright.async_api import async_playwright
    copy = out / "motion-page.html"
    copy.write_text(instrumented(Path(page_path).read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    records = {}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(channel="msedge", headless=True, args=sheet.GPU_ARGS[gpu])
        context = await browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1, offline=True)
        page = await context.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(copy.resolve().as_uri() + "?t=0&pause=1&hud=0&scale=0.5&meshes=1")
        await page.wait_for_function("window.__hog && (__hog.error || (__hog.ready && __hog.frames >= 10))",
                                     timeout=300000)
        failure = await page.evaluate("__hog.error || null")
        if failure:
            raise RuntimeError(failure)
        adapter = await page.evaluate("document.getElementById('gpu-info').textContent")
        if gpu not in adapter.lower():
            raise RuntimeError(f"blocked: asked for {gpu}, but Edge renders on {adapter!r}")
        _, positions = await page.evaluate(tour.TRACK, tour.STEP_S)
        wps = tour.waypoints(Path(page_path).read_text(encoding="utf-8", errors="replace"))
        times = tour.waypoint_times(positions, wps)
        await page.evaluate("() => __motion.stop()")
        for name in names:
            caption, occurrence, lead, what = SEQUENCES[name]
            rec = await record(page, name, schedule(pass_time(wps, times, caption, occurrence) + lead), out)
            rec.update(checks=what, waypoint=caption, occurrence=occurrence or 1, page=str(page_path),
                       page_hash=page_tag(page_path), gpu=gpu, adapter=adapter)
            (out / f"{name}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            records[name] = rec
            print(f"{name}: {len(rec['frames'])} frames, pops at {rec['pops']}", flush=True)
        await browser.close()
    if errors:
        raise RuntimeError(f"page errors: {errors}")
    return records


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", choices=sorted(sheet.GPU_ARGS), default="nvidia")
    parser.add_argument("--page", type=Path, default=PAGE)
    parser.add_argument("--only", type=lambda s: [p for p in s.split(",") if p])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    names = [n for n in SEQUENCES if not args.only or n in args.only]
    args.out.mkdir(parents=True, exist_ok=True)
    with locks.gpu_work(args.gpu, f"inspection motion ({len(names)} sequences)"):
        asyncio.run(run(args.page, names, args.gpu, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
