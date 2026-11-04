"""Labelled review sheets: chosen views of the page rendered in Edge on one GPU, a column per configuration.

    python -X utf8 -m tools.review.sheet FAMILY [--page PAGE] [--before PAGE] [--gpu nvidia|amd] [--views 0,2,5]
                                         [--scale 1] [--out PATH]

FAMILY names a module in tools/review holding VIEWS and COLUMNS, or one that renders itself through its own make() (the
tour). A view is (label, camera x, height above ground, z, target x, target height above ground, target z, hour); x None
puts the camera over the lake's centre, and the hour may be "civil dusk" or "night". A column is (title, query); --before
adds the same columns for an earlier page first.
The sheet goes to build/review/<family>-<page hash>.png unless --out says otherwise. A run holds its GPU's lock and counts
as heavy work, so a suite on the same GPU and a timing run wait for it.
"""
import argparse
import asyncio
import importlib
import io
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import async_playwright

from tools import locks
from tools.paths import BUILD, PAGE, ROOT, page_tag
from mesh_probe import GPU_ARGS

MESHES = "?t=60&pause=1&hud=0&scale={scale}"                      # the scene as the page shows it
LANDSCAPE = "?t=60&pause=1&hud=0&scale={scale}&meshes=0"           # the terrain and trees alone (a diagnostic)
W, H, THUMB = 1280, 720, (800, 450)
SETTLE_FRAMES = 160


async def hours_by_name(page):
    """The first hour from 17:00 (three-minute steps) at civil dusk (sun 0.5 degrees down) and at full night."""
    named = {}
    for h in [x / 20 for x in range(17 * 20, 23 * 20)]:
        light = await page.evaluate("(h) => __hog.light(h)", h)
        if "civil dusk" not in named and light["el"] < -0.5:
            named["civil dusk"] = h
        if "night" not in named and light["dusk"] > 0.99:
            named["night"] = h
    return named


async def render(page_path, query, views, gpu):
    shots = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(channel="msedge", headless=True, args=GPU_ARGS[gpu])
        page = await (await browser.new_context(viewport={"width": W, "height": H}, offline=True)).new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(Path(page_path).resolve().as_uri() + query)
        await page.wait_for_function("window.__hog && (__hog.error || (__hog.ready && __hog.frames >= 10))",
                                     timeout=180000)
        failure = await page.evaluate("__hog.error || null")
        if failure:
            raise RuntimeError(failure)
        named = await hours_by_name(page)
        lake = await page.evaluate("DATA.meta.extras.lake_c")

        async def ground(x, z):
            return await page.evaluate("([x, z]) => __hog.ground(x, z)[1]", [x, z])

        for label, x, y, z, tx, ty, tz, hour in views:
            if x is None:
                x, z = lake
            p = [x, await ground(x, z) + y, z]
            q = [tx, await ground(tx, tz) + ty, tz]
            d = [q[i] - p[i] for i in range(3)]
            h = named[hour] if isinstance(hour, str) else hour
            frames = await page.evaluate(
                "([p, yaw, pitch, h]) => { __hog.camera(p, yaw, pitch); __hog.tod(h); return __hog.frames; }",
                [p, math.atan2(d[0], d[2]), math.atan2(d[1], math.hypot(d[0], d[2])), h])
            await page.wait_for_function(f"__hog.frames >= {frames + SETTLE_FRAMES}", timeout=180000)
            shots.append(Image.open(io.BytesIO(await page.screenshot())).convert("RGB"))
        await browser.close()
        if errors:
            raise RuntimeError(f"page errors: {errors}")
    return shots


def compose(out, views, columns):
    font = ImageFont.truetype("arial.ttf", 22)
    pad, label_h, head_h = 12, 30, 40
    img = Image.new("RGB", (len(columns) * (THUMB[0] + pad) + pad, head_h + len(views) * (THUMB[1] + label_h + pad) + pad),
                    (24, 24, 28))
    draw = ImageDraw.Draw(img)
    for c, (title, shots) in enumerate(columns):
        x = pad + c * (THUMB[0] + pad)
        draw.text((x, 8), title, fill=(240, 240, 240), font=font)
        for r, shot in enumerate(shots):
            y = head_h + r * (THUMB[1] + label_h + pad)
            draw.text((x, y + 2), f"{r + 1}. {views[r][0]}", fill=(210, 210, 210), font=font)
            img.paste(shot.resize(THUMB, Image.LANCZOS), (x, y + label_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


async def make(family, page, before=None, gpu="nvidia", views=None, scale=1, out=None):
    spec = importlib.import_module(f"tools.review.{family}")
    if hasattr(spec, "make"):                       # a sheet whose views are not fixed cameras renders itself
        return await spec.make(page, before, gpu, views, scale, out)
    chosen = [spec.VIEWS[i] for i in views] if views else spec.VIEWS
    pages = ([("Before", before)] if before else []) + [("After" if before else None, page)]
    columns = []
    for role, path in pages:
        for title, query in spec.COLUMNS:
            shots = await render(path, query.format(scale=f"{scale:g}"), chosen, gpu)
            columns.append((f"{role}: {title}" if role else title, shots))
    out = Path(out) if out else BUILD / "review" / f"{family}-{page_tag(page)}.png"
    compose(out, chosen, columns)
    return out


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("family")
    parser.add_argument("--page", type=Path, default=PAGE)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--gpu", choices=sorted(GPU_ARGS), default="nvidia")
    parser.add_argument("--views", type=lambda s: [int(i) for i in s.split(",")])
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    label = f"review sheet {args.family} in {ROOT.name}"
    with locks.gpu_work(args.gpu, label):
        out = asyncio.run(make(args.family, args.page, args.before, args.gpu, args.views, args.scale, args.out))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
