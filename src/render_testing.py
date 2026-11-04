"""Shared helpers of the rendering tests (test_render_*.py): diagnostic payloads, the probed frame's rays, and a page
test case with views and shading probes that several families use.

Diagnostic boxes go through the real encoder, the page's decode/upload, the depth resolve and the mesh draw; each
probed G-buffer pixel is compared with an analytic ray-box intersection along that frame's jittered ray."""
import functools
import json
import math
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import box as rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import build_meshes  # noqa: E402
from assets_testing import levels  # noqa: E402
from mesh_source import load_data  # noqa: E402
from pagetest import PageTestCase  # noqa: E402

SCENE = {"assets": [{"id": "diagnostic"}]}
WALL = (6, 1)                     # boundary-wall material: plain albedo, no window bands
MAT_SKY, MAT_GROUND, MAT_WATER = 0, 1, 2
EAST, WEST = math.pi / 2, -math.pi / 2
# open lawn between the castle and the hut, clear of every mesh but the test's own
LAWN = (-260.0, -60.0)
SUNLIT = (-150.0, -350.0)         # open grass south of the castle that the midday sun reaches
EYE_HEIGHT_M = 25.0
EDGE_MARGIN_M = 0.03              # rays this close to a box edge may land either side after quantization
JITTER_CYCLE = 8                  # frames before the Halton camera jitter repeats
FRAME_TIMEOUT_S = 30.0            # the longest wait for a page's next frame, as for Playwright's waits

# In-page JS: `await nextFrame(ms)` is true once the page has rendered a new frame, which always takes the next jitter
# phase (the jittered path needs EXT_color_buffer_float), or false after `ms` without one, even if animation frames
# stop. A still view stops rendering once it has converged (template_c.html's IDLE_N frames) while __hog.frames counts
# on, so a loop that waits on __hog.frames can probe one frame over and over.
NEXT_FRAME = """const nextFrame = async (ms) => {
  const j = __hog.probe(0, 0).jit, end = performance.now() + ms;
  while (performance.now() < end) {
    await new Promise((done) => { requestAnimationFrame(done); setTimeout(done, Math.max(0, end - performance.now())); });
    const v = __hog.probe(0, 0).jit;
    if (v[0] !== j[0] || v[1] !== j[1]) return true;
  }
  return false;
};"""
# In-page JS: restarts the still-view count, as replacing the meshes does, so the next ~95 frames all render. It works
# through the meshes' generation, which the still-view check sees only while the meshes are on.
REARM = """if (!__hog.meshStats().on) throw new Error("REARM needs the meshes on: with them off, nothing restarts a still view");
__hog.meshes(true);"""


def payload(*boxes, panes=(), walls=(), attr=None):
    """Axis-aligned boxes (x0, y0, z0, x1, y1, z1), glazing slabs and (footprint, y0, y1) walls, as a mesh payload
    encoded as build_meshes.py does; boxes take `attr` (material, aux), boundary walls by default."""
    meshes = [build_meshes.prism(rectangle(x0, z0, x1, z1), y0, y1, a)
              for a, group in ((attr or WALL, boxes), ((build_meshes.MAT_GLASS, 0), panes))
              for x0, y0, z0, x1, y1, z1 in group]
    meshes += [build_meshes.prism(footprint, y0, y1, WALL) for footprint, y0, y1 in walls]
    return json.loads(build_meshes.strip_js(build_meshes.encode_payload({"diagnostic": meshes}, SCENE)))

def ray(view, px, py):
    """The ray P1 and the mesh projection both use for internal pixel (px, py) in the probed frame."""
    w, h = view["res"]
    u = (2 * (px + 0.5 + view["jit"][0]) - w) / h * view["tanH"]
    v = (2 * (py + 0.5 + view["jit"][1]) - h) / h * view["tanH"]
    d = np.array(view["rot"]).reshape(3, 3).T @ np.array([u, v, 1.0])
    return np.array(view["pos"]), d / np.linalg.norm(d)

def box_hit(origin, direction, box, grow=0.0):
    lo, hi = np.array(box[:3]) - grow, np.array(box[3:]) + grow
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = (lo - origin) / direction, (hi - origin) / direction
    near, far = np.minimum(a, b).max(), np.maximum(a, b).min()
    return near if 0 < near <= far else None

def ahead(view, distance, size, lift=0.0):
    """A cube of edge `size` centred `distance` metres along the probed frame's centre ray, raised by `lift`."""
    w, h = view["res"]
    o, d = ray(view, w // 2, h // 2)
    c = o + d * distance + np.array([0.0, lift, 0.0])
    return (*(c - size / 2), *(c + size / 2))

def off_screen(view, box):
    """True when one frustum plane has every corner of the box outside it."""
    w, h = view["res"]
    rot = np.array(view["rot"]).reshape(3, 3)
    corners = np.array([[box[3 * (k & 1) + 0], box[3 * (k >> 1 & 1) + 1], box[3 * (k >> 2) + 2]] for k in range(8)])
    x, y, z = rot @ (corners - np.array(view["pos"])).T
    tx, ty = view["tanH"] * w / h, view["tanH"]
    return any(np.all(c) for c in (z < 0, x > z * tx, x < -z * tx, y > z * ty, y < -z * ty))

@functools.cache
def castle_floor():
    """The castle platform of the world the page shows (never datum-shifted): castle walls stand on it."""
    return levels(load_data())["platform"]


class RenderTestCase(PageTestCase):
    """A shared page with the meshes on, and helpers to show a diagnostic payload in place of the scene's meshes from a
    chosen view and probe what it draws."""

    async def show(self, boxes, at=LAWN, yaw=EAST, pitch=-0.25, height=EYE_HEIGHT_M, panes=(), walls=(), shadow=(), attr=None):
        x, z = at
        ground = (await self.page.evaluate("([x, z]) => __hog.ground(x, z)", [x, z]))[1]
        body = payload(*boxes, panes=panes, walls=walls, attr=attr) if boxes or panes or walls else None
        if shadow:
            body["shadow"] = [[float(v) for v in box] for box in shadow]
        frames = await self.page.evaluate(
            "async ([b, pos, yaw, pitch]) => { __hog.camera(pos, yaw, pitch); await __hog.loadMeshes(b); return __hog.frames; }",
            [body, [x, ground + height, z], yaw, pitch])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 3}")

    async def centre(self):
        return await self.page.evaluate("() => { const r = __hog.probe(0, 0).res; return __hog.probe(r[0] >> 1, r[1] >> 1); }")

    async def each_frame(self, probe, arg=None, n=JITTER_CYCLE):
        """[probe(arg) in each of the page's next n rendered frames], `probe` being a JS function's source. The frames are
        gathered inside the page after REARM, since round trips that a busy machine slows down could otherwise run past
        a still view's convergence; a frame that does not come within FRAME_TIMEOUT_S fails the test."""
        frames = await self.page.evaluate(f"""async ([arg, n, ms]) => {{
            {NEXT_FRAME}
            const probe = {probe}, out = [];
            {REARM}
            while (out.length < n && await nextFrame(ms)) out.push(probe(arg));
            return out; }}""", [arg, n, FRAME_TIMEOUT_S * 1000])
        self.assertEqual(len(frames), n, f"frame {len(frames) + 1} of {n} did not render within {FRAME_TIMEOUT_S:g} s")
        return frames

    async def ground(self, x, z):
        return (await self.page.evaluate("([x, z]) => __hog.ground(x, z)", [x, z]))[1]

    async def lake_view(self, pitch, toward_castle=False):
        """Camera 30 m over the lake; dbg 128 makes the shading pass report (mesh reflected?, reflected distance, rmat)."""
        lake = await self.page.evaluate("DATA.meta.extras.lake_c")
        yaw = math.atan2(-lake[0], -lake[1]) if toward_castle else 0.0
        await self.page.evaluate("__hog.dbg(128)")
        return {"at": tuple(lake), "yaw": yaw, "pitch": pitch, "height": 30.0}


    async def reflected(self):
        """Mean colour of the mirrored meshes over the 49 x 49 pixels round the centre, through one jitter cycle: a frame's
        jitter phase moves the edges between faces across the reflection's half-resolution texels, which shifts one
        frame's mean by several per cent."""
        frames = await self.each_frame("""() => {
            const [w, h] = __hog.probe(0, 0).res, out = [];
            for (let y = (h >> 1) - 24; y <= (h >> 1) + 24; y += 2) for (let x = (w >> 1) - 24; x <= (w >> 1) + 24; x += 2) {
              const r = __hog.probeRefl(x, y);
              if (r[3] > 0) out.push(r.slice(0, 3));
            }
            return out; }""")
        return np.array([colour for frame in frames for colour in frame], float).reshape(-1, 3).mean(axis=0)

    async def castle_box(self, attr, hour, sunlit=True):
        """Shading-pass colours on the top and the (sunlit, unless not `sunlit`) west wall of a 20 x 24 m box, 22 m tall and
        tagged `attr`, 30 m east of LAWN at `hour`; the castle base there gives class 2 (range) walls, which P3 paints with
        window bands up to 1.5 m under the range's top. A castle code's box, and the camera with it, rises to the castle's
        floor where that stands above the lawn, as the castle does: P3 paints its windows and shades its foot from there."""
        x, z = LAWN
        ground = (await self.page.evaluate("([x, z]) => __hog.ground(x, z)", [x, z]))[1]
        lift = max(castle_floor() - ground, 0.0) if attr[0] == build_meshes.MAT_CASTLE else 0.0
        frames = await self.page.evaluate("(h) => { __hog.tod(h); return __hog.frames; }", hour)
        await self.page.wait_for_function(f"__hog.frames >= {frames + 10}")
        box = (x + 30.0, ground + lift - 1.0, z - 12.0, x + 50.0, ground + lift + 22.0, z + 12.0)
        await self.show([box], at=(x - 15.0, z), height=45.0 + lift, pitch=-0.5, attr=attr)
        faces = await self.page.evaluate("""([mat, sunlit]) => {
            const [w, h] = __hog.probe(0, 0).res, out = { top: [], wall: [] };
            for (let y = 1; y < h; y += 3) for (let x = 1; x < w; x += 3) {
              const p = __hog.probe(x, y);
              if (p.mat !== mat || (sunlit && __hog.probeSun(x, y) < 0.95)) continue;
              if (p.n[1] > 0.9) out.top.push(__hog.probeColor(x, y).slice(0, 3));
              else if (p.n[0] < -0.9) out.wall.push(__hog.probeColor(x, y).slice(0, 3));
            }
            return out; }""", [attr[0], sunlit])
        looks = {k: np.array(v, float).reshape(-1, 3) for k, v in faces.items()}
        self.assertGreater(len(looks["wall"]), 200)
        self.assertGreater(len(looks["top"]), 50)
        return looks

    async def light_at(self, test):
        """The first hour from 17:00 (in steps of three minutes) whose lighting passes `test`."""
        for hour in np.arange(17.0, 23.0, 0.05):
            if test(await self.page.evaluate("(h) => __hog.light(h)", float(hour))):
                return float(hour)
        self.fail("no hour has that light")

    async def screen_box(self, attr, hour):
        """Displayed colours (0-255: after exposure, tone mapping and TAA) of castle_box's west wall, tagged `attr`."""
        await self.castle_box(attr, hour, sunlit=False)
        frames = await self.page.evaluate("__hog.frames")
        await self.page.wait_for_function(f"__hog.frames >= {frames + 40}")
        return np.array(await self.page.evaluate("""(mat) => {
            const [w, h] = __hog.probe(0, 0).res, out = [];
            for (let y = 2; y < h - 2; y += 3) for (let x = 2; x < w - 2; x += 3) {
              const p = __hog.probe(x, y);
              if (p.mat === mat && p.n[0] < -0.9) out.push(__hog.probeScreen((x + 0.5) / w, (y + 0.5) / h));
            }
            return out; }""", attr[0]), float).reshape(-1, 3)
