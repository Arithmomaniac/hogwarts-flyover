"""The tour's contact sheet: a still of the page at each of the tour's waypoints, at the moment the flyover passes
nearest it, with the meshes on, rendered in Edge on one GPU. It shows the integrated scene, so a trunk merge's sheet,
set beside the previous merge's, shows what the merge broke.

    python -X utf8 -m tools.review.sheet tour [--page PAGE] [--before PAGE] [--gpu nvidia|amd] [--scale 1] [--out PATH]

doit s5:tour makes the trunk page's sheet, build/review/tour-<page hash>.png. With --before, each waypoint's still on the
earlier page stands to the left of the page's, the two matched by caption, or by camera point for unnamed waypoints
(match), each under its own label.
"""
import ast
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.async_api import async_playwright

from tools.paths import BUILD, page_tag
from tools.review import sheet
from mesh_probe import GPU_ARGS

QUERY = "?t=0&pause=1&hud=0&scale={scale}&meshes=1"   # asked for: a page from before the integration has them off
HOUR = 15.0                     # mid-afternoon, so no view is lost to a low sun
STEP_S = 0.05                   # the spacing of the tour's samples searched for each waypoint
# the tour's length and its camera positions every `step` seconds, as the page flies it
TRACK = """(step) => {
    const n = Math.floor(__hog.total / step), pos = [];
    for (let i = 0; i < n; i++) pos.push(__hog.path(i * step).pos);
    return [__hog.total, pos]; }"""
THUMB = (640, 360)
PAIRS_PER_ROW, SINGLES_PER_ROW = 2, 4
BACKGROUND = (24, 24, 28)


def waypoints(page_text):
    """The tour's waypoints as the page defines them: (camera E, N, height, look E, N, height, caption or None)."""
    start = page_text.index("const WP = [")
    block = page_text[start + len("const WP = "): page_text.index("];", start) + 1]
    return ast.literal_eval(block.replace("null", "None"))


def waypoint_times(positions, wps, step=STEP_S):
    """The tour time at which the camera (positions: (n, 3) samples `step` s apart) passes nearest each waypoint's
    camera point, each found after the one before, so a tour that passes a waypoint twice keeps its order."""
    positions, times, after = np.asarray(positions, float), [], 0
    for w in wps:
        d = np.hypot(positions[:, 0] - w[0], positions[:, 2] - w[1])
        d[:after] = np.inf
        after = int(np.argmin(d))
        times.append(after * step)
    return times


async def stills(page_path, gpu, scale=1, only=None):
    """{waypoint index: (label, image)} of the page at each waypoint, or at the waypoints whose indices are in `only`."""
    if only is not None and not only:
        return {}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(channel="msedge", headless=True, args=GPU_ARGS[gpu])
        page = await (await browser.new_context(viewport={"width": sheet.W, "height": sheet.H}, offline=True)).new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        await page.goto(Path(page_path).resolve().as_uri() + QUERY.format(scale=f"{scale:g}"))
        await page.wait_for_function("window.__hog && (__hog.error || (__hog.ready && __hog.frames >= 10))", timeout=300000)
        failure = await page.evaluate("__hog.error || null")
        if failure:
            raise RuntimeError(failure)
        total, positions = await page.evaluate(TRACK, STEP_S)
        wps = waypoints(Path(page_path).read_text(encoding="utf-8", errors="replace"))
        out = {}
        for k, (w, t) in enumerate(zip(wps, waypoint_times(positions, wps))):
            if only is not None and k not in only:
                continue
            frames = await page.evaluate("([t, h]) => { __hog.set(t); __hog.tod(h); return __hog.frames; }", [t, HOUR])
            await page.wait_for_function(f"__hog.frames >= {frames + sheet.SETTLE_FRAMES}", timeout=180000)
            shot = Image.open(io.BytesIO(await page.screenshot())).convert("RGB")
            out[k] = (f"{k + 1}. {w[6] or '(no caption)'}, {t:.0f} s of {total:.0f}", shot)
        await browser.close()
        if errors:
            raise RuntimeError(f"page errors: {errors}")
    return out


def keys(wps, other):
    """Each of a page's waypoints `wps` named for matching it with the `other` page's: by its caption, or, when it has
    none or either page uses its caption more than once, by its caption and camera point."""
    def repeated(ws):
        captions = [w[6] for w in ws]
        return {c for c in captions if captions.count(c) > 1}
    shared = {None} | repeated(wps) | repeated(other)
    return [(w[6], w[0], w[1]) if w[6] in shared else w[6] for w in wps]


def match(before, after):
    """[(index on the earlier page or None, index on the later page or None)], one per waypoint of either page's tour:
    the later page's waypoints in their order, each with the earlier page's of the same name (keys), then the earlier
    page's waypoints that the later lacks. Unnamed waypoints match by camera point, so adding or dropping one does not
    set different places side by side; one that moves stands unmatched on each side."""
    where = {k: i for i, k in enumerate(keys(before, after))}
    pairs = [(where.get(k), j) for j, k in enumerate(keys(after, before))]
    matched = {i for i, _ in pairs}
    return pairs + [(i, None) for i in range(len(before)) if i not in matched]


def compose(out, titles, cells):
    """A contact sheet of `cells` in reading order: each a waypoint's stills (label, image), one under each of `titles`,
    so with two each waypoint's earlier still stands to the left of its later one. None marks a page that lacks it."""
    font = ImageFont.truetype("arial.ttf", 20)
    pad, label_h, head_h = 10, 28, 36
    per_row = PAIRS_PER_ROW if len(titles) > 1 else SINGLES_PER_ROW
    rows = -(-len(cells) // per_row)
    cell_w = len(titles) * (THUMB[0] + pad)
    img = Image.new("RGB", (per_row * cell_w + pad, head_h + rows * (THUMB[1] + label_h + pad) + pad), BACKGROUND)
    draw = ImageDraw.Draw(img)
    draw.text((pad, 8), "   |   ".join(titles), fill=(240, 240, 240), font=font)
    for k, cell in enumerate(cells):
        x0, y = pad + (k % per_row) * cell_w, head_h + (k // per_row) * (THUMB[1] + label_h + pad)
        for c, still in enumerate(cell):
            x = x0 + c * (THUMB[0] + pad)
            if still is None:
                draw.text((x, y + 2), "(not on this page)", fill=(150, 150, 150), font=font)
                continue
            draw.text((x, y + 2), still[0], fill=(210, 210, 210), font=font)
            img.paste(still[1].resize(THUMB, Image.LANCZOS), (x, y + label_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


async def make(page, before=None, gpu="nvidia", views=None, scale=1, out=None):
    """tools.review.sheet's entry for this sheet, whose views are the tour's waypoints rather than fixed cameras. `views`
    picks the page's waypoints by index, each beside the earlier page's of the same name (match) with --before."""
    read = lambda path: waypoints(Path(path).read_text(encoding="utf-8", errors="replace"))
    wps = read(page)
    pairs = match(read(before), wps) if before else [(None, j) for j in range(len(wps))]
    if views:
        pairs = [(i, j) for i, j in pairs if j in set(views)]
    late = await stills(page, gpu, scale, {j for _, j in pairs if j is not None})
    path = Path(out) if out else BUILD / "review" / f"tour-{page_tag(page)}.png"
    if not before:
        return compose(path, [f"Page ({page_tag(page)})"], [[late[j]] for _, j in pairs])
    early = await stills(before, gpu, scale, {i for i, _ in pairs if i is not None})
    cells = [[None if i is None else early[i], None if j is None else late[j]] for i, j in pairs]
    return compose(path, [f"Before ({page_tag(before)})", f"After ({page_tag(page)})"], cells)
