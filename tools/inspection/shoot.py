"""The inspection's shots: each inventory object's close-up and context view, and each tour waypoint's still, as final
displayed frames of the finished page in Edge on one GPU.

    python -X utf8 -m tools.inspection.shoot [--gpu nvidia|amd] [--only ID,FAMILY/,PREFIX*,...] [--baseline PAGE]
                                             [--detail ID,...] [--page PAGE] [--out DIR]

Frames are 1920 x 1080 at device pixel ratio 1 with the page's internal scale fixed at 0.5 (adaptive scaling off), its
default quality and detail, paused and with no HUD, as the plan's qualification requires; `--detail` adds a close-up at
internal scale 1 for the objects it names. An object with no views of its own (Obj.views) is framed: for a payload
object, each side frame.candidates offers, sun first, is rendered and probed on the framing test's grid until one shows
its subject clearly (frame.score, frame.choose); landscape objects take the sun's side, or their own. Its context view
stands further back on the close-up's side unless it gives its own. Tour stills come first, at the page's own hour,
before any view takes the camera off the tour. `--baseline` shoots each payload object's close-up again on that page
(baseline-x, the frozen page at site X), for the checklist's comparison with the baseline's low-detail version; its
__hog has no camera or light, so it is shot from a copy given those two hooks (hooked_baseline).

Everything goes to build/inspection/shots/: a PNG per shot and a JSON record per object, written as soon as the object
is shot, with its views, hours, the scores of every candidate tried, and what the browser reported (adapter, device
pixel ratio, canvas and internal sizes, page hash). The run holds its GPU's lock and registers as heavy work
(locks.gpu_work).
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from tools import locks
from tools.inspection import frame, inventory
from tools.paths import PAGE, page_tag
from tools.review import sheet, tour

W, H, SCALE = 1920, 1080, 0.5
QUERY = "?t=0&pause=1&hud=0&scale={scale}&meshes=1"
PROBE_FRAMES = 12                   # frames a candidate renders before its G-buffer is probed (as the framing test)
SETTLE_FRAMES = sheet.SETTLE_FRAMES # frames a shot renders before it is taken: the temporal history converges
CLEAR_M = 2.0                       # a framed camera stands at least this far above the ground under it
NIGHT_SUN_DEG = -12.0               # night: the sun this far down. The review sheets' "night", every lamp lit, comes with
                                    # the sun only 2.5 degrees down, in a bright twilight
MESH_KINDS = ("asset", "part", "sample", "outlier")
OUT = inventory.OUT_DIR / "shots"
# the framing test's grid (src/test_render_tour.py), read in one pass: every 4th internal pixel's hit distance and
# material
GRID = """() => {
    const view = __hog.probe(0, 0), [w, h] = view.res, all = __hog.probeRect(0, 0, w, h), out = [];
    for (let y = 2; y < h; y += 4) for (let x = 2; x < w; x += 4) { const r = all[y * w + x]; out.push([x, y, r[5], r[0]]); }
    return { view, out }; }"""
OBSERVED = """() => { const cv = document.querySelector('canvas'), v = __hog.probe ? __hog.probe(0, 0) : null;
    return { adapter: document.getElementById('gpu-info').textContent, dpr: window.devicePixelRatio,
             canvas: [cv.width, cv.height], internal: v ? v.res : null, viewport: [innerWidth, innerHeight] }; }"""
# baseline-x's __hog has no camera or light; its script keeps the same free camera and lighting(), so a copy of it gets
# the two hooks after this anchor, as the page's own camera and light are written
BASELINE_ANCHOR = "path: pathAt, ground,"
BASELINE_NEEDS = ("let free = false, fp = [0, 0, 0], fyaw = 0, fpitch = 0", "function lighting(")
BASELINE_HOOKS = "camera(pos, yaw, pitch) { free = true; fp = pos.slice(); fyaw = yaw; fpitch = pitch; }, light: lighting,"
HOOKED = inventory.OUT_DIR / "baseline-hooked.html"


def hooked_baseline(text):
    """The baseline page's text with the camera and light hooks the shots need; refused unless the anchor occurs once
    and the page declares the free camera's variables and lighting()."""
    if text.count(BASELINE_ANCHOR) != 1 or any(n not in text for n in BASELINE_NEEDS):
        raise ValueError("the baseline page lacks the anchor or the free camera the shots' hooks need")
    return text.replace(BASELINE_ANCHOR, f"{BASELINE_ANCHOR} {BASELINE_HOOKS}")


def file_name(object_id, role):
    return f"{object_id.replace('/', '__')}-{role}.png"


def record_name(object_id):
    return f"{object_id.replace('/', '__')}.json"


def resolve(view, ground, lake):
    """(eye, target) of a view in world (east, height, north): its heights above the ground under each point ("own"),
    or both above the ground under its target ("target") with the camera at least CLEAR_M above the ground under it.
    A camera with no east or north stands over the lake's centre `lake`, as in tools.review.sheet."""
    (x, y, z), (tx, ty, tz) = view.eye, view.target
    if x is None:
        x, z = lake
    if view.rel == "own":
        return (x, ground(x, z) + y, z), (tx, ground(tx, tz) + ty, tz)
    base = ground(tx, tz)
    return (x, max(base + y, ground(x, z) + CLEAR_M), z), (tx, base + ty, tz)


def context_view(obj, close, ground, lake):
    """An object's context view: its own second view, or one framed from further back on its close-up's side."""
    if len(obj.views) > 1:
        return obj.views[1]
    eye, target = resolve(close, ground, lake)
    side = frame.bearing(frame.View(close.label, eye, target, close.hour))
    return frame.framed(f"{obj.title}: context", obj.at, obj.size, side, close.hour, context=True)


def subject_box(obj, ground):
    """The world box a view of an object is scored on: its own, or a cube of its size round its target, cut off just
    under the ground there (a ray into the cube below the ground would end on the ground in front of it, as if
    hidden)."""
    if obj.box:
        return tuple(obj.box)
    x, h, z = obj.at
    g = ground(x, z)
    y, s = g + h, obj.size / 2
    return (x - s, max(y - s, g - 0.5), z - s, x + s, y + s, z + s)


class Session:
    """One page in Edge on one GPU, and what the inspection asks of it."""

    def __init__(self, page, named, lake, scale):
        self.page, self.named, self.lake, self.scale = page, named, lake, scale
        self._ground = {}

    def ground(self, x, z):
        key = (round(x, 3), round(z, 3))
        if key not in self._ground:
            raise KeyError(key)
        return self._ground[key]

    async def fetch_ground(self, *points):
        missing = [(round(x, 3), round(z, 3)) for x, z in points if (round(x, 3), round(z, 3)) not in self._ground]
        if missing:
            heights = await self.page.evaluate("(ps) => ps.map(([x, z]) => __hog.ground(x, z)[1])", missing)
            self._ground.update(zip(missing, heights))

    async def place(self, view):
        """Set the camera and hour of a view; (eye, target) as resolved."""
        x, z = view.eye[0], view.eye[2]
        if x is None:
            x, z = self.lake
        await self.fetch_ground((x, z), (view.target[0], view.target[2]))
        eye, target = resolve(view, self.ground, self.lake)
        yaw, pitch = frame.aim(eye, target)
        hour = self.named[view.hour] if isinstance(view.hour, str) else view.hour
        frames = await self.page.evaluate(
            "([p, yaw, pitch, h]) => { __hog.camera(p, yaw, pitch); __hog.tod(h); return __hog.frames; }",
            [list(eye), yaw, pitch, hour])
        return eye, target, frames

    async def settle(self, frames, n):
        await self.page.wait_for_function(f"__hog.frames >= {frames + n}", timeout=180000)

    async def score(self, view, box):
        _, _, frames = await self.place(view)
        await self.settle(frames, PROBE_FRAMES)
        grid = await self.page.evaluate(GRID)
        return frame.score(grid["view"], grid["out"], box)

    async def shot(self, view, path, scale=None):
        if scale is not None:
            await self.page.evaluate("(s) => __hog.scale(s)", scale)
        eye, target, frames = await self.place(view)
        await self.settle(frames, SETTLE_FRAMES)
        await self.page.screenshot(path=str(path))
        if scale is not None:
            await self.page.evaluate("(s) => __hog.scale(s)", self.scale)
        return eye, target


async def open_page(pw, page_path, gpu, scale):
    browser = await pw.chromium.launch(channel="msedge", headless=True, args=sheet.GPU_ARGS[gpu])
    context = await browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=1, offline=True)
    page = await context.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    await page.goto(Path(page_path).resolve().as_uri() + QUERY.format(scale=f"{scale:g}"))
    await page.wait_for_function("window.__hog && (__hog.error || (__hog.ready && __hog.frames >= 10))",
                                 timeout=300000)
    failure = await page.evaluate("__hog.error || null")
    if failure:
        raise RuntimeError(failure)
    observed = await page.evaluate(OBSERVED)
    if gpu not in observed["adapter"].lower():
        raise RuntimeError(f"blocked: asked for {gpu}, but Edge renders on {observed['adapter']!r}")
    named = await hours_by_name(page)
    lake = tuple(await page.evaluate("DATA.meta.extras.lake_c"))
    return browser, Session(page, named, lake, scale), observed, errors


async def hours_by_name(page):
    """The review sheets' named hours (tools.review.sheet.hours_by_name), with night when the sun is NIGHT_SUN_DEG
    down."""
    named = await sheet.hours_by_name(page)
    for h in [x / 20 for x in range(17 * 20, 24 * 20)]:
        if (await page.evaluate("(h) => __hog.light(h)", h))["el"] < NIGHT_SUN_DEG:
            named["night"] = h
            break
    return named


async def frame_close(session, obj, sun_azimuth):
    """(close-up view, [(azimuth, visible, occluded)] of each candidate tried) of an object with no views of its own."""
    hour = obj.hour
    if obj.kind not in MESH_KINDS:
        azimuth = sun_azimuth if obj.azimuth is None else obj.azimuth
        return frame.framed(obj.title, obj.at, obj.size, azimuth, hour, elevation=obj.elevation), []
    await session.fetch_ground((obj.at[0], obj.at[2]))
    box = subject_box(obj, session.ground)
    sides = [obj.azimuth] if obj.azimuth is not None else [az for az, _ in frame.candidates(sun_azimuth)]
    tried, views = [], []
    for azimuth in sides:
        view = frame.framed(obj.title, obj.at, obj.size, azimuth, hour, elevation=obj.elevation)
        visible, occluded = await session.score(view, box)
        tried.append((round(azimuth, 1), round(visible, 3), round(occluded, 3)))
        views.append(view)
        if visible >= frame.MIN_VISIBLE and occluded <= frame.MAX_OCCLUDED:
            break
    return views[frame.choose([(v, o) for _, v, o in tried])], tried


async def shoot_page(page_path, objects, gpu, out, scale=SCALE, detail=()):
    """Shoot every object on the page; returns {object id: record}."""
    from playwright.async_api import async_playwright
    records, tag = {}, page_tag(page_path)
    async with async_playwright() as pw:
        browser, session, observed, errors = await open_page(pw, page_path, gpu, scale)
        stills = [o for o in objects if o.kind == "tour"]
        if stills:
            _, positions = await session.page.evaluate(tour.TRACK, tour.STEP_S)
            wps = tour.waypoints(Path(page_path).read_text(encoding="utf-8", errors="replace"))
            times = tour.waypoint_times(positions, wps)
            for o in stills:
                k = int(o.id.split("/")[1].split("-")[0]) - 1
                frames = await session.page.evaluate("(t) => { __hog.set(t); return __hog.frames; }", times[k])
                await session.settle(frames, SETTLE_FRAMES)
                name = file_name(o.id, "tour")
                await session.page.screenshot(path=str(out / name))
                records[o.id] = write_record(out, {
                    "object": o.id, "page": str(page_path), "page_hash": tag, "gpu": gpu, "scale": scale,
                    "observed": observed, "shots": [{"file": name, "role": "tour", "time": times[k],
                                                     "hour": "the page's own (18.5)"}]})
        sun = (await session.page.evaluate("(h) => __hog.light(h)", inventory.DAY))["az"]
        for o in objects:
            if o.kind == "tour":
                continue
            close, tried = (o.views[0], []) if o.views else await frame_close(session, o, sun)
            shots = []
            eye, target = await session.shot(close, out / file_name(o.id, "close"))
            shots.append({"file": file_name(o.id, "close"), "role": "close", "view": close, "eye": eye,
                          "target": target, "candidates": tried})
            if o.id in detail:
                eye, target = await session.shot(close, out / file_name(o.id, "detail"), scale=1.0)
                shots.append({"file": file_name(o.id, "detail"), "role": "detail", "view": close, "eye": eye,
                              "target": target, "scale": 1.0})
            context = context_view(o, close, session.ground, session.lake)
            eye, target = await session.shot(context, out / file_name(o.id, "context"))
            shots.append({"file": file_name(o.id, "context"), "role": "context", "view": context, "eye": eye,
                          "target": target})
            records[o.id] = write_record(out, {"object": o.id, "page": str(page_path), "page_hash": tag, "gpu": gpu,
                                               "scale": scale, "observed": observed, "shots": shots})
            print(f"{o.id}: {', '.join(s['file'] for s in shots)}" + (f" (tried {tried})" if tried else ""),
                  flush=True)
        await browser.close()
    if errors:
        raise RuntimeError(f"page errors: {errors}")
    return records


def write_record(out, record):
    """Write an object's record beside its shots as soon as it has them, so a later failure keeps them."""
    (out / record_name(record["object"])).write_text(json.dumps(record, indent=1, default=list), encoding="utf-8")
    return record


async def shoot_baseline(hooked, objects, records, gpu, out, scale=SCALE):
    """Each payload object's close-up again on the baseline page (its hooked copy), added to its record."""
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        browser, session, observed, errors = await open_page(pw, hooked, gpu, scale)
        for o in objects:
            if o.kind not in MESH_KINDS or o.id not in records:
                continue
            close = frame.View(*records[o.id]["shots"][0]["view"])
            await session.shot(close, out / file_name(o.id, "baseline"))
            records[o.id]["shots"].append({"file": file_name(o.id, "baseline"), "role": "baseline", "view": close,
                                           "page": str(hooked), "page_hash": page_tag(hooked), "observed": observed})
            write_record(out, records[o.id])
        await browser.close()
    if errors:
        raise RuntimeError(f"baseline page errors: {errors}")


def chosen(objects, only):
    """The objects `only` names, each selector an exact object id, or a prefix when it ends in "/" (a family, as
    "tour/") or "*" (as "castle/court-house-*")."""
    if not only:
        return objects

    def named(oid, p):
        if p.endswith("*"):
            return oid.startswith(p[:-1])
        return oid.startswith(p) if p.endswith("/") else oid == p
    return [o for o in objects if any(named(o.id, p) for p in only)]


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", choices=sorted(sheet.GPU_ARGS), default="nvidia")
    parser.add_argument("--page", type=Path, default=PAGE)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--only", type=lambda s: [p for p in s.split(",") if p])
    parser.add_argument("--detail", type=lambda s: {p for p in s.split(",") if p}, default=set())
    parser.add_argument("--inventory", type=Path, default=inventory.OUT_DIR / "inventory.json")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    objects = chosen(inventory.load(args.inventory), args.only)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.baseline:               # hooked before the long main pass, so a page the hooks cannot fit stops it at once
        HOOKED.parent.mkdir(parents=True, exist_ok=True)
        HOOKED.write_text(hooked_baseline(args.baseline.read_text(encoding="utf-8")), encoding="utf-8")
    with locks.gpu_work(args.gpu, f"inspection shots ({len(objects)} objects)"):
        records = asyncio.run(shoot_page(args.page, objects, args.gpu, args.out, detail=args.detail))
        if args.baseline:
            asyncio.run(shoot_baseline(HOOKED, objects, records, args.gpu, args.out))
    print(f"wrote {len(records)} records to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
