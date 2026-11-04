"""The integrated scene: the meshes are the scene's objects. With no meshes loaded, nothing raymarched stands in their
place in any pass: the primary and far-tile marches, the cone pre-pass, the shadow and sun marches and the lake
reflections. The meshes' own shadows and reflections replace the raymarched objects', and the page opens with the
meshes on. The trees follow the loaded payload's footprint layer: none stands on ground it marks as built. The goals'
poles stay whole from afar, widened to a pixel, and keep their true size close up."""
import io
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assets_testing as at  # noqa: E402
import build_meshes  # noqa: E402
import mesh_footprint  # noqa: E402
import mesh_grounds  # noqa: E402
from mesh_geometry import bar, box, concat, lathe  # noqa: E402
from materials import code  # noqa: E402
from mesh_source import CHANNELS, Grid, decode, load_data, sd_bytes  # noqa: E402
from pagetest import PAGE, smoke  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402
from render_testing import (FRAME_TIMEOUT_S, JITTER_CYCLE, MAT_GROUND, NEXT_FRAME, SCENE, SUNLIT, WALL,  # noqa: E402
                            RenderTestCase, payload, ray)
from test_site import class_colours  # noqa: E402

OBJECT_MATS = (4, 5, 6, 7)                  # castle, building, wall or hedge, set piece: what objects write
MAT_TREE = 3
RMAT_RAYMARCHED_CASTLE, RMAT_MESH = 4, 5    # the reflection codes the shading pass reports under debug bit 128
DBG_REFLECTIONS = 128
DBG_NO_CONE = 512                           # the primary march starts at the camera, without the cone pre-pass
# active samplers per program, on both GPUs (limit 16): the raymarched objects' four (the castle's two distance fields,
# the building codes and the boundary walls) left every raymarching program when the meshes replaced them, the
# surface class moved into locB's third channel, read with texelFetch, and the mesh casters' shadow layers moved from
# the shadow pass to the shading pass, which looks them up for each pixel
SAMPLERS = {"p1": 10, "far": 10, "shadow": 11, "shade": 15, "cone": 9, "sun": 10, "classify": 1, "mesh": 1, "resolve": 1}
GRID = """(step) => {
    const v = __hog.probe(0, 0), [w, h] = v.res, out = [];
    for (let y = 1; y < h; y += step) for (let x = 1; x < w; x += step) { const p = __hog.probe(x, y); out.push([p.mat, p.t]); }
    return out; }"""
# ground in a court of the castle that at 16:30 only the castle shades, 30 m from any ground that does not qualify
# (found on the page before integration: shaded with the castle's meshes or the raymarched castle, sunlit with neither)
CASTLE_SHADE_HOUR = 16.5
CASTLE_SHADE = (18.8, 1.1)
CLEARING_M = 30.0                           # a test footprint: a disc of wooded ground marked as built
CROWN_M = 4.9                               # frag1.glsl treeH's widest crown, which may lean in over a clearing's edge
FAR_FROM_MESHES_M = 60.0
DBG_CLEAN_EDGES = 2048                      # the surface classes' edges without their noise
DBG_NO_CLOUD_SHADOWS = 1024                 # the ground lit as under a clear sky
# a class's own texel or its neighbour's: seen from CLASS_VIEW_M straight above (2 m texels about 9 internal pixels
# across), ground pixels between CLASS_EDGE_TEXELS of a texel from an edge with the other class must show their own
# texel's class; the two classes' colours must stand CLASS_SEPARATION apart in log chromaticity
CLASS_VIEW_M = 60.0
CLASS_EDGE_TEXELS = (0.05, 0.4)
CLASS_AGREE = 0.98
CLASS_SEPARATION = 0.2
SHINGLE = 7                                 # the page re-draws the shingle's texels at load (template_b.html)
# dirty state: a box mirrored in the lake's middle fills MIRRORED_MIN of the lake within MIRROR_PROBE_PX internal
# pixels of the centre, and MIRROR_AWAY_M east none of it; a post POST_M tall and POST_WIDE_M across, seen from
# SUN_VIEW_M straight above, shades the ground at mid-height below SHADE_MAX of the sun, which elsewhere is above
# SUNLIT_MIN; a still view converges within STILL_WAIT frames (template_c.html's IDLE_N is 96)
MIRROR_PROBE_PX = 10
MIRRORED_MIN = 0.8
MIRROR_AWAY_M = 400.0
POST_M, POST_WIDE_M = 12.0, 4.0
SUN_VIEW_M = 60.0
SUN_HOURS = (11.0, 14.5)
SHADE_MAX, SUNLIT_MIN = 0.3, 0.7
STILL_WAIT = 110
# a payload replaced: the old post's top, GHOST_CONTRAST (0-255) off the ground, must be within GHOST_LEFT of the
# settled ground two frames later; the new post stands GHOST_AWAY_M east
GHOST_HOUR = 12.5
GHOST_AWAY_M = 15.0
GHOST_CONTRAST, GHOST_LEFT = 40.0, 12.0
SHORE_HEIGHT_M = 35.0
SHORE_SPAN_M = 3.0                          # the beach's edge is judged against its own running mean over this much shore
SHORE_EDGE_RMS_M = 0.25                     # the 2 m texels' staircase measured 0.39 m, the smooth edge 0.15 m
SHORE_KINK_STEP_M = 0.5
# nor may it turn at texel corners: the 90th percentile of its second difference over 0.5 m steps measured 1.0 m on the
# staircase, 0.45 m with the shingle's field unblurred or the shingle kept in the class map, 0.18 m smoothed
SHORE_KINK_P90_M = 0.3
BANK_NY = 0.95                              # sloped ground: the bank
BANK_ANGLE_P95_DEG = 8.0                    # neighbouring bank normals 0.26 m apart: 15.9 degrees cell by cell, 3.5 smoothed
NORMALS = """() => { const [w, h] = __hog.probe(0, 0).res, x0 = (w / 10) | 0;
    return [w - 2 * x0, __hog.probeRect(x0, 2, w - 2 * x0, h - 4)]; }"""
DBG_THIN_OFF = 4096                         # thin parts keep their true size (template_c.html thinPerM)
DBG_EVERY_REFLECTION = 8192                 # the mirror draws every chunk its frustum holds (template_c.html seenInLake)
LAKE_CULL_BEARING_DEG = 240.0               # from over the lake's middle: the mirror's frustum holds about twice the chunks
                                            # whose reflections could land on its water
# two half-float steps of the shading pass's reflection record (its reflected distance, metres): run to run, mirroring
# every chunk moved up to 6 of about 15,000 lake values by that much (2 of 25 runs on NVIDIA), and the culled mirror
# moved none
MIRROR_RT_RTOL = 2.0 ** -9
LAKE_PIXELS_MIN = 500
DBG_NO_LAKE_QUERY = 32768                   # the mirror draws what its cull passes, lake water in view or not
# the mirror waits for lake water in view: LAKE_EYE_M over the lake's middle, a wall WALL_AHEAD_M ahead hides all of
# it, while a tower TOWER_AHEAD_M ahead in the water passes the mirror's cull; a still view's lake query has answered
# LAKE_ANSWER frames on; a jump of LAKE_JUMP_M past the wall, or a turn of LAKE_TURN_RAD, shows the water
LAKE_EYE_M, LAKE_PITCH = 30.0, -0.1
WALL_AHEAD_M, WALL_HALF_M, WALL_TOP_M = 20.0, 60.0, 120.0
TOWER_AHEAD_M, TOWER_HALF_M, TOWER_TOP_M = 200.0, 15.0, 60.0
LAKE_ANSWER = 20
LAKE_JUMP_M, LAKE_TURN_RAD = 60.0, math.pi / 2
GOLD = code("gold")                         # the goals' poles and hoops: mesh_thin.POLICY's thin parts
POLE_FOOT_M = 2.0                           # a pole's span is judged from this over the pitch to its top
POLE_BACK_M = tuple(d + mesh_grounds.HOOP_U_M for d in (250.0, 450.0))   # from the pitch's centre: the near goal's poles
                                            # about 250 m and 450 m off (0.2 and 0.1 internal pixels), the far goal's
                                            # 130 m beyond them
POLE_UP_M, POLE_LOOK_M = 40.0, 7.0          # over the stands, at the poles' middle
POLE_BEARING_DEG = 15.0                     # off the pitch's axis: each goal's three poles stand 3 pixels or more apart
THIN_PAD_PX = 1.0                           # a gold fragment counts for a pole's row this close to its projected axis
POLE_IN_FRONT_M = 3.0                       # a row whose axis pixel is this much nearer than the pole has it hidden
POLE_ROWS_MIN = 4                           # the far goal's poles from about 580 m measured 5
POLE_SHARE_UNWIDENED = 0.5                  # true-size poles, for the measure to tell: measured 0.15 and 0.09
GOAL_NEAR_M, GOAL_NEAR_UP_M, GOAL_NEAR_BEARING_DEG = 20.0, 5.0, 45.0
GOAL_FAR_M = 60.0                           # the near goal's pixels, left of the far goal's


def goal_poles(meta):
    """Each goal pole's (foot, top) as world points, POLE_FOOT_M over the pitch and at its top, where
    mesh_grounds.hoops stands them: the eastern goal's three first."""
    p = mesh_grounds.pitch_layout(meta["pitch"])
    top = p.level + mesh_grounds.HOOP_HEIGHT - mesh_grounds.HOOP_ACROSS
    return [(np.array([x, p.level + POLE_FOOT_M, z]), np.array([x, top, z]))
            for end in (1.0, -1.0) for k in (-1.0, 0.0, 1.0)
            for x, z in [p.at(end * mesh_grounds.HOOP_U_M, k * mesh_grounds.HOOP_SPACING)]]


def to_pixel(view, point):
    """(x, y, distance): the internal pixel whose centre `point` projects to in the probed frame, as `ray` casts, and
    its distance from the camera, as the G-buffer records it."""
    w, h = view["res"]
    d = np.asarray(point, float) - np.array(view["pos"])
    c = np.array(view["rot"]).reshape(3, 3) @ d
    return ((c[0] / c[2] / view["tanH"] * h + w) / 2 - 0.5 - view["jit"][0],
            (c[1] / c[2] / view["tanH"] * h + h) / 2 - 0.5 - view["jit"][1], float(np.linalg.norm(d)))


def aim(eye, target):
    d = np.asarray(target, float) - np.asarray(eye, float)
    return math.atan2(d[0], d[2]), math.atan2(d[1], math.hypot(d[0], d[2]))


def embedded_footprint():
    """The page's own payload's footprint layer, as bytes on the local grid."""
    payload = json.loads(build_meshes.strip_js((ROOT / "build" / "meshes.js").read_text(encoding="ascii")))
    return mesh_footprint.decoded(payload["footprint"])


def wooded_spot():
    """World (east, north) of the densest wood over a CLEARING_M disc on the local grid, FAR_FROM_MESHES_M from every
    standing mesh and well inside the grid (where only the local layers count)."""
    data = load_data()
    grid = Grid(data["meta"]["local"])
    trees = decode(data["tex"]["locA"])[..., CHANNELS["locA"].index("trees")] / 255.0
    texel = float(np.mean(grid.texel_size()))
    r = int(CLEARING_M / texel)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = (np.hypot(yy, xx) <= r).astype(float)
    density = ndi.convolve(trees, disc / disc.sum(), mode="constant")
    meshes = sd_bytes(embedded_footprint()) < mesh_footprint.OPEN_M
    ok = ndi.distance_transform_edt(~meshes) * texel > FAR_FROM_MESHES_M
    inner = np.zeros_like(ok)
    inner[100:-100, 100:-100] = True
    i, j = np.unravel_index(np.argmax(np.where(ok & inner, density, -1.0)), density.shape)
    return tuple(float(v) for v in grid.to_world(i, j))


def with_clearing(spot):
    """The embedded payload's footprint entry with a CLEARING_M disc of built ground round `spot` added."""
    grid = Grid(load_data()["meta"]["local"])
    ii, jj = np.mgrid[0:grid.ny, 0:grid.nx]
    x, z = grid.to_world(ii, jj)
    disc = np.clip(np.floor(128 + 10 * (np.hypot(x - spot[0], z - spot[1]) - CLEARING_M)), 0, 255).astype(np.uint8)
    return mesh_footprint.encode(np.minimum(embedded_footprint(), disc))


def class_edge():
    """(class map, its grid, (a, b), world (east, north)): the pair of surface classes that meets most on open ground,
    of those with one colour each (test_site.class_colours) but the shingle, and the middle of the 25-texel window
    where they meet most. Open ground is level, dry and clear of roads, and far enough from trees and meshes for their
    noon shadows, at least 120 m inside the local grid, where its layers alone colour the ground (frag1.glsl's lw)."""
    data = load_data()
    grid = Grid(data["meta"]["local"])
    cls = decode(data["tex"]["locN"])[..., CHANNELS["locN"].index("surface")].astype(int)
    la = decode(data["tex"]["locA"])
    texel = float(np.mean(grid.texel_size()))
    rise = np.hypot(*np.gradient(decode(data["tex"]["locBed"]).astype(float), texel))
    near = (la[..., CHANNELS["locA"].index("road")] > 10) | (decode(data["tex"]["locW"]) > 0) | (rise > 0.1)
    tall = (la[..., CHANNELS["locA"].index("trees")] > 10) | (sd_bytes(embedded_footprint()) < mesh_footprint.OPEN_M)
    open_ = ~(ndi.binary_dilation(near, iterations=3) | ndi.binary_dilation(tall, iterations=8))
    inset = int(math.ceil(120.0 / texel))
    open_[:inset] = open_[-inset:] = False
    open_[:, :inset] = open_[:, -inset:] = False
    plain = [k for k, (colours, noise) in class_colours().items() if len(colours) == 1 and not noise and k != SHINGLE]
    keys = []
    for a, b, ok in ((cls[:-1], cls[1:], open_[:-1] & open_[1:]), (cls[:, :-1], cls[:, 1:], open_[:, :-1] & open_[:, 1:])):
        m = ok & (a != b) & np.isin(a, plain) & np.isin(b, plain)
        keys.append(np.minimum(a[m], b[m]) * 16 + np.maximum(a[m], b[m]))
    a, b = divmod(int(np.bincount(np.concatenate(keys)).argmax()), 16)
    cross = ndi.generate_binary_structure(2, 1)
    edge = open_ & (((cls == a) & ndi.binary_dilation(cls == b, cross)) | ((cls == b) & ndi.binary_dilation(cls == a, cross)))
    i, j = np.unravel_index(np.argmax(ndi.uniform_filter(edge.astype(float), 25)), edge.shape)
    return cls, grid, (a, b), tuple(float(v) for v in grid.to_world(i, j))


class IntegratedSceneTests(RenderTestCase):
    QUERY = "?t=60&pause=1&scale=0.35&hud=0"   # the page's own defaults: no meshes or debug parameters
    DBG = 0

    async def look(self, target, back, up, bearing):
        """The camera `back` metres from the target along `bearing` (degrees from north) and `up` metres over the
        ground there, looking at it; a few frames to settle."""
        b = math.radians(bearing)
        x, z = target[0] + back * math.sin(b), target[2] + back * math.cos(b)
        eye = [x, await self.ground(x, z) + up, z]
        yaw, pitch = aim(eye, target)
        frames = await self.page.evaluate("([p, y, q]) => { __hog.camera(p, y, q); return __hog.frames; }", [eye, yaw, pitch])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 12}", timeout=FRAME_TIMEOUT_S * 1000)
        return eye

    async def at_ground(self, x, z, lift=0.0):
        return [x, await self.ground(x, z) + lift, z]

    async def views(self):
        """(name, target, back, up, bearing) for each object the raymarcher drew, from DATA and the manifest."""
        meta = await self.page.evaluate("DATA.meta")
        extras, scene = meta["extras"], at.scene()
        claimed = lambda asset: next(a for a in scene["assets"] if a["id"] == asset)["components"][0]["at"]
        towers = np.array([t[:2] for t in meta["towers"]])
        mid = lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        brg = math.radians(meta["pitch"]["brg"])
        goal = (meta["pitch"]["c"][0] + 61.0 * math.sin(brg), meta["pitch"]["c"][1] + 61.0 * math.cos(brg))
        hedge, wall = self.layer_points()
        ship, lake = mid(extras["ship"]["a"], extras["ship"]["b"]), extras["lake_c"]
        lake_side = math.degrees(math.atan2(lake[0] - ship[0], lake[1] - ship[1]))   # look at the ship from the lake
        castle = await self.at_ground(*towers.mean(axis=0), 30.0)
        out = [("castle", castle, 380.0, 140.0, 225.0)]
        for name, (x, z), back, up, bearing, lift in (
                ("High Street", claimed("three-broomsticks"), 70.0, 30.0, 135.0, 3.0),
                ("hut", claimed("hut"), 40.0, 45.0, 90.0, 2.0),                 # over the trees round it
                ("greenhouses", claimed("greenhouses-north"), 50.0, 55.0, 45.0, 2.0),
                ("gate pillars", mid(*extras["gates"]), 35.0, 10.0, 180.0, 3.0),
                ("tomb", extras["tomb"], 25.0, 8.0, 0.0, 1.0),
                ("train", mid(*meta["train"]["carriages"]), 90.0, 30.0, 90.0, 2.0),
                ("engine", mid(*meta["train"]["engine"]), 45.0, 15.0, 90.0, 2.0),
                ("ship", mid(extras["ship"]["a"], extras["ship"]["b"]), 140.0, 90.0, lake_side, 10.0),
                ("goal", goal, 50.0, 15.0, 180.0, 8.0),
                ("Willow", extras["willow"], 40.0, 10.0, 180.0, 5.0),
                ("hedge", hedge, 30.0, 10.0, 0.0, 1.0),
                ("boundary wall", wall, 30.0, 10.0, 0.0, 1.0)):
            out.append((name, await self.at_ground(x, z, lift), back, up, bearing))
        out.append(("castle beyond the 3D trees", castle, 3200.0, 500.0, 225.0))
        return out

    @staticmethod
    def layer_points():
        """A raymarched hedge's and a boundary wall's middle texel, in world (east, north), from the data's layers."""
        data = load_data()
        grid = Grid(data["meta"]["local"])
        a, b = decode(data["tex"]["locA"]), decode(data["tex"]["locB"])
        out = []
        for field, level in ((sd_bytes(a[..., CHANNELS["locA"].index("hedge")]), 0.45),
                             (sd_bytes(b[..., CHANNELS["locB"].index("wall")]), -1.05)):
            i, j = np.nonzero(field < level)
            k = len(i) // 2
            out.append(tuple(float(v) for v in grid.to_world(i[k], j[k])))
        return out

    @smoke
    async def test_every_program_uses_its_pinned_samplers(self):
        audit = await self.page.evaluate("__hog.samplers()")
        self.assertEqual(audit["counts"], SAMPLERS)
        self.assertLessEqual(max(SAMPLERS.values()), audit["limit"])

    @smoke
    async def test_the_page_opens_with_the_meshes_on(self):
        page = await self.context.new_page()
        try:
            await page.goto(PAGE.as_uri() + self.QUERY)
            await page.wait_for_function("window.__hog && (__hog.error || (__hog.ready && __hog.frames >= 10))",
                                         timeout=120000)
            stats = await page.evaluate("__hog.error || __hog.meshStats()")
        finally:
            await page.close()
        self.assertTrue(stats["on"])
        self.assertGreater(stats["drawn"], 0)

    async def test_no_object_stands_where_the_meshes_are_gone(self):
        await self.page.evaluate("__hog.loadMeshes(null)")
        for name, target, back, up, bearing in await self.views():
            eye = await self.look(target, back, up, bearing)
            mats = np.array(await self.page.evaluate(GRID, 3))
            with self.subTest(view=name):
                objects = int(np.isin(mats[:, 0], OBJECT_MATS).sum())
                self.assertEqual(objects, 0, f"{objects} of {len(mats)} probed pixels are objects")
                centre = await self.centre()
                # the view sees the place, not a hill in front of it
                self.assertGreater(centre["t"], 0.6 * math.dist(eye, target), "the target is hidden")

    async def at_castle_shade(self, payload):
        """The castle-shaded court at CASTLE_SHADE_HOUR, with the embedded meshes or none."""
        await self.page.evaluate("async ([b, h]) => { __hog.tod(h); await __hog.loadMeshes(b ? MESHES : null); }",
                                 [payload, CASTLE_SHADE_HOUR])
        await self.look(await self.at_ground(*CASTLE_SHADE), 20.0, 160.0, 200.0)     # from above, over the court's walls

    @smoke
    async def test_the_castles_shadow_is_its_meshs_in_the_shadow_pass(self):
        looks = {}
        for payload in (True, False):
            await self.at_castle_shade(payload)
            looks[payload] = await self.each_frame("""() => { const [w, h] = __hog.probe(0, 0).res;
                const x = 4 * (w >> 3) + 2, y = 4 * (h >> 3) + 2, p = __hog.probe(x, y);
                return [p.mat, __hog.probeSun(x, y)]; }""")      # the shadow-pass block whose centre pixel sees the court
        for mat, _ in looks[True] + looks[False]:
            self.assertEqual(mat, MAT_GROUND)
        self.assertLess(max(s for _, s in looks[True]), 0.3, "the castle's meshes should shade the court")
        self.assertGreater(min(s for _, s in looks[False]), 0.9, "with no meshes nothing should shade the court")

    async def test_the_castles_shadow_is_its_meshs_in_the_sun_cache(self):
        # the sun cache (tTree's alpha, per 7 m cell) is what the march reads beyond SUN_NEAR, 3.2 km
        cache = {}
        for payload in (True, False):
            await self.at_castle_shade(payload)
            cache[payload] = await self.page.evaluate("([x, z]) => __hog.probeSunCache(x, z)", list(CASTLE_SHADE))
        self.assertLess(cache[True], 0.3, "the castle's meshes should shade the court's cell")
        self.assertGreater(cache[False], 0.9, "with no meshes nothing should shade the court's cell")

    async def lake_codes(self, payload):
        lake = await self.page.evaluate("DATA.meta.extras.lake_c")
        await self.page.evaluate("async (b) => { __hog.dbg(%d); await __hog.loadMeshes(b ? MESHES : null); }" % DBG_REFLECTIONS,
                                 payload)
        eye = await self.at_ground(lake[0], lake[1], 30.0)
        yaw = math.atan2(-lake[0], -lake[1])
        frames = await self.page.evaluate("([p, y]) => { __hog.camera(p, y, -0.12); return __hog.frames; }", [eye, yaw])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 12}")
        return np.array(await self.page.evaluate("""() => { const [w, h] = __hog.probe(0, 0).res, out = [];
            for (let y = 2; y < h / 2; y += 3) for (let x = 2; x < w; x += 3)
              if (__hog.probe(x, y).mat === 2) out.push(__hog.probeColor(x, y)[2]);
            return out; }"""))

    async def trees_round(self, spot):
        """(tree pixels, probed pixels) within CLEARING_M - CROWN_M of `spot` on the ground, seen from 150 m above it."""
        eye = [spot[0], await self.ground(*spot) + 150.0, spot[1] - 10.0]      # pitched 86 degrees down, north
        frames = await self.page.evaluate("([p]) => { __hog.camera(p, 0, -1.5); return __hog.frames; }", [eye])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 12}", timeout=FRAME_TIMEOUT_S * 1000)
        view, samples = await self.page.evaluate("""(step) => { const v = __hog.probe(0, 0), [w, h] = v.res, out = [];
            for (let y = h >> 2; y < 3 * h >> 2; y += step) for (let x = (w - (h >> 1)) >> 1; x < (w + (h >> 1)) >> 1; x += step) {
              const p = __hog.probe(x, y); out.push([x, y, p.mat, p.t]); }
            return [v, out]; }""", 3)
        trees = probed = 0
        for x, y, mat, t in samples:
            o, d = ray(view, x, y)
            hit = o + d * t
            if math.hypot(hit[0] - spot[0], hit[2] - spot[1]) < CLEARING_M - CROWN_M:
                probed += 1
                trees += mat == MAT_TREE
        return trees, probed

    async def test_the_trees_follow_the_payloads_footprint(self):
        spot = wooded_spot()
        await self.page.evaluate("async (f) => { await __hog.loadMeshes({ ...MESHES, footprint: f }); }", with_clearing(spot))
        cleared, probed = await self.trees_round(spot)
        await self.page.evaluate("async () => { await __hog.loadMeshes(MESHES); }")
        back, probed_back = await self.trees_round(spot)
        self.assertGreater(min(probed, probed_back), 100)
        self.assertEqual(cleared, 0, "trees stand on ground the payload's footprint marks as built")
        self.assertGreater(back, 0.25 * probed_back, "the wood should come back with the embedded payload's footprint")

    async def shore_view(self):
        """Look straight down on the shore 40 m along it from the tomb, the lawn at the bottom of the frame and the lake
        at the top, with the classes' edges free of their noise: metres per screen pixel on the ground."""
        extras = await self.page.evaluate("DATA.meta.extras")
        tomb, lake = np.array(extras["tomb"], float), np.array(extras["lake_c"], float)
        out = (lake - tomb) / np.linalg.norm(lake - tomb)
        p = tomb + 40.0 * np.array([out[1], -out[0]])
        for _ in range(120):                                          # out to the waterline
            bed, top = await self.page.evaluate("([x, z]) => __hog.ground(x, z)", [float(p[0]), float(p[1])])
            if top > bed + 0.05:
                break
            p = p + 0.5 * out
        else:
            self.fail("no waterline found beside the tomb")
        p = p - 3.0 * out                                             # over the beach
        eye = [float(p[0]), await self.ground(*p) + SHORE_HEIGHT_M, float(p[1])]
        frames = await self.page.evaluate("([e, y]) => { __hog.tod(12.5); __hog.dbg(%d); __hog.camera(e, y, -1.5); "
                                          "return __hog.frames; }" % DBG_CLEAN_EDGES, [eye, math.atan2(out[0], out[1])])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 60}", timeout=FRAME_TIMEOUT_S * 1000)
        return 2.0 * SHORE_HEIGHT_M * (await self.centre())["tanH"] / self.VIEWPORT["height"]

    async def test_the_beach_follows_the_shore_not_the_texel_grid(self):
        metres = await self.shore_view()
        image = np.asarray(Image.open(io.BytesIO(await self.page.screenshot())).convert("RGB"), float)
        green = image[..., 1] - 0.5 * (image[..., 0] + image[..., 2])
        lawn = float(np.median(green[-image.shape[0] // 10:]))         # the bottom of the frame is lawn
        self.assertGreater(lawn, 10.0, "the bottom of the frame is not lawn")
        edge = []                                                     # rows up from the lawn to the beach
        for column in green.T[image.shape[1] // 10:-(image.shape[1] // 10)]:   # a tree stands at the frame's edge
            up = np.nonzero(column[::-1] < 0.5 * lawn)[0]
            edge.append(len(column) - 1 - up[0] if len(up) and 10 < up[0] < len(column) - 10 else np.nan)
        span = int(round(SHORE_SPAN_M / metres)) | 1
        runs = [r[~np.isnan(r)] for r in np.split(np.array(edge), np.nonzero(np.isnan(edge))[0])]
        runs = [r for r in runs if len(r) > 4 * span]
        self.assertTrue(runs, "the beach's edge does not cross the frame")
        residual = []
        for r in runs:                                                # each against its own running mean
            mean = np.convolve(r, np.ones(span) / span, "valid")
            residual.append(r[span // 2:span // 2 + len(mean)] - mean)
        with self.subTest(measure="teeth"):
            self.assertLess(float(np.sqrt(np.mean(np.concatenate(residual) ** 2))) * metres, SHORE_EDGE_RMS_M)
        k = int(round(SHORE_KINK_STEP_M / metres))
        bends = np.concatenate([np.abs(r[2 * k:] - 2 * r[k:-k] + r[:-2 * k]) for r in runs]) * metres
        with self.subTest(measure="corners"):
            self.assertLess(float(np.percentile(bends, 90)), SHORE_KINK_P90_M)

    async def test_the_bank_shades_as_one_slope_not_cell_by_cell(self):
        await self.shore_view()
        width, probed = await self.page.evaluate(NORMALS)
        g = np.array(probed, float).reshape(-1, width, 6)             # material, normal, aux, distance
        a, b = g[:, :-2], g[:, 2:]                                    # each pixel and the one two to its right
        bank = (a[..., 0] == MAT_GROUND) & (b[..., 0] == MAT_GROUND) & (np.minimum(a[..., 2], b[..., 2]) < BANK_NY)
        n, m = a[bank][:, 1:4], b[bank][:, 1:4]
        cos = np.einsum("ij,ij->i", n, m) / (np.linalg.norm(n, axis=1) * np.linalg.norm(m, axis=1))
        angles = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        self.assertGreater(len(angles), 200, "too little bank in the frame")
        self.assertLess(float(np.percentile(angles, 95)), BANK_ANGLE_P95_DEG)

    async def test_the_ground_takes_its_class_from_the_texel_nearest_filtering_takes(self):
        # the shading pass reads the surface class with texelFetch, from the texel nearest filtering would take: with
        # the classes' noise off, a ground pixel near an edge between two classes shows its own texel's class, which
        # a read half a texel off would swap for its neighbour's
        cls, grid, (a, b), (x, z) = class_edge()
        top = await self.ground(x, z)
        frames = await self.page.evaluate("([p]) => { __hog.dbg(%d); __hog.tod(12.5); __hog.camera(p, 0, -1.5); "
                                          "return __hog.frames; }" % (DBG_CLEAN_EDGES | DBG_NO_CLOUD_SHADOWS),
                                          [[x, top + CLASS_VIEW_M, z]])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 4}", timeout=FRAME_TIMEOUT_S * 1000)
        view, found = await self.page.evaluate("""() => {
            const v = __hog.probe(0, 0), [w, h] = v.res, g = __hog.probeRect(0, 0, w, h), c = __hog.probeColorRect(0, 0, w, h), out = [];
            for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
              const i = y * w + x;
              if (g[i][0] === %d) out.push([x, y, g[i][5], c[4 * i], c[4 * i + 1], c[4 * i + 2]]);
            }
            return [v, out]; }""" % MAT_GROUND)
        px = np.array(found, float)
        w, h = view["res"]
        rays = np.column_stack([(2 * (px[:, 0] + 0.5 + view["jit"][0]) - w) / h * view["tanH"],
                                (2 * (px[:, 1] + 0.5 + view["jit"][1]) - h) / h * view["tanH"], np.ones(len(px))])
        rays = rays @ np.array(view["rot"]).reshape(3, 3)                 # as `ray` turns them into the world
        world = np.array(view["pos"]) + rays / np.linalg.norm(rays, axis=1, keepdims=True) * px[:, 2:3]
        row, col = grid.to_texel(world[:, 0], world[:, 2])
        rr, cc = row + 0.5, col + 0.5                                      # texel corners at whole numbers
        i, j = np.floor(rr).astype(int), np.floor(cc).astype(int)
        own, fi, fj = cls[i, j], rr - i, cc - j
        across = [(cls[np.where(fi < 0.5, i - 1, i + 1), j], np.minimum(fi, 1 - fi)),
                  (cls[i, np.where(fj < 0.5, j - 1, j + 1)], np.minimum(fj, 1 - fj))]
        pair = np.isin(own, (a, b))
        edge = pair & np.any([(n != own) & np.isin(n, (a, b)) & (d > CLASS_EDGE_TEXELS[0]) & (d < CLASS_EDGE_TEXELS[1])
                              for n, d in across], axis=0)
        inner = pair & np.all([(n == own) & (d > 0.25) for n, d in across], axis=0)
        chroma = np.log(np.maximum(px[:, [3, 5]], 1e-4)) - np.log(np.maximum(px[:, [4]], 1e-4))   # R/G, B/G
        ref = {k: np.median(chroma[inner & (own == k)], axis=0) for k in (a, b)}
        self.assertGreater(min(np.sum(inner & (own == k)) for k in (a, b)), 100, f"too little of classes {a} and {b} in view")
        self.assertGreater(float(np.linalg.norm(ref[a] - ref[b])), CLASS_SEPARATION, "the two classes look alike")
        self.assertGreater(int(edge.sum()), 200, "too little edge in view")
        shown = np.where(np.linalg.norm(chroma - ref[a], axis=1) < np.linalg.norm(chroma - ref[b], axis=1), a, b)
        self.assertGreaterEqual(float(np.mean(shown[edge] == own[edge])), CLASS_AGREE)

    async def goal_view(self, target, back, up, bearing):
        """The camera `back` metres from `target` along `bearing` (degrees from north) and `up` metres over the ground,
        looking at it: the probed view and each goal pole's (foot, top) as world points."""
        meta = await self.page.evaluate("DATA.meta")
        await self.look(target, back, up, bearing)
        return await self.centre(), goal_poles(meta)

    async def pole_rows(self, back, dbg):
        """Per frame of a jitter cycle, per goal pole: (rows of its projected span the pole is in front of, rows of those
        holding a gold fragment within THIN_PAD_PX of its projected axis), seen from `back` metres with debug bits `dbg`."""
        await self.page.evaluate(f"__hog.dbg({dbg})")
        meta = await self.page.evaluate("DATA.meta")
        p = mesh_grounds.pitch_layout(meta["pitch"])
        view, poles = await self.goal_view([float(p.c[0]), p.level + POLE_LOOK_M, float(p.c[1])], back, POLE_UP_M,
                                           meta["pitch"]["brg"] + POLE_BEARING_DEG)
        w, h = view["res"]
        spans = [[to_pixel(view, end) for end in pole] for pole in poles]
        rects = []
        for (x0, y0, _), (x1, y1, _) in spans:
            left, bottom = max(int(math.floor(min(x0, x1))) - 3, 0), max(int(math.floor(y0)) - 1, 0)
            rects.append([left, bottom, min(int(math.ceil(max(x0, x1))) + 4, w) - left, min(int(math.ceil(y1)) + 2, h) - bottom])
        frames = await self.each_frame("""(rects) => [__hog.probe(0, 0), rects.map(([x, y, w, h]) => __hog.probeRect(x, y, w, h))]""",
                                       rects)
        out = []
        for v, probed in frames:
            counts = []
            for (x, y, rw, rh), (foot, top), got in zip(rects, poles, probed):
                g = np.array(got, float).reshape(rh, rw, 6)
                (fx, fy, fd), (tx, ty, td) = to_pixel(v, foot), to_pixel(v, top)
                seen = covered = 0
                for row in range(int(math.ceil(fy)), int(math.floor(ty)) + 1):
                    f = (row - fy) / (ty - fy)
                    axis, depth = fx + f * (tx - fx), fd + f * (td - fd)
                    cols = [c for c in range(rw) if abs(x + c - axis) <= THIN_PAD_PX]
                    nearest = g[row - y, int(round(axis)) - x]
                    if nearest[5] < depth - POLE_IN_FRONT_M:             # the stands or a hoop stand before it
                        continue
                    seen += 1
                    covered += any(g[row - y, c, 0] == GOLD[0] and g[row - y, c, 4] == GOLD[1] for c in cols)
                counts.append((seen, covered))
            out.append(counts)
        return out

    async def test_the_goals_poles_stay_whole_from_afar(self):
        # PS 8: "three golden poles with hoops on the end ... fifty feet high". From 250 m a 0.18 m pole is about 0.2
        # internal pixels: only the jittered frames whose samples land on it draw it, and the temporal filter's clamp
        # then fades it. Widened to THIN_PX it covers a pixel in every row of every frame.
        for back in POLE_BACK_M:
            for dbg, whole in ((0, True), (DBG_THIN_OFF, False)):
                with self.subTest(back=back, widened=whole):
                    frames = await self.pole_rows(back, dbg)
                    seen = np.array([[s for s, _ in f] for f in frames])
                    covered = np.array([[c for _, c in f] for f in frames])
                    self.assertTrue((seen >= POLE_ROWS_MIN).all(), f"too few rows of some pole in view: {seen.min(axis=0)}")
                    share = covered.sum() / seen.sum()
                    if whole:
                        self.assertEqual(share, 1.0, f"rows without the pole, per frame and pole: {(seen - covered).tolist()}")
                    else:
                        self.assertLess(share, POLE_SHARE_UNWIDENED, "the measure cannot tell sub-pixel poles from whole ones")

    async def test_thin_parts_keep_their_true_size_close_up(self):
        # from 20 m the hoops' 0.12 m tubes span more than THIN_PX either side: widening may not touch them. Pixels
        # farther than GOAL_FAR_M are left out: the far goal's poles behind may widen.
        records = {}
        for dbg in (0, DBG_THIN_OFF):
            await self.page.evaluate(f"__hog.dbg({dbg})")
            meta = await self.page.evaluate("DATA.meta")
            foot, top = goal_poles(meta)[1]                        # the eastern goal's middle pole
            view, _ = await self.goal_view([float(v) for v in top], GOAL_NEAR_M, GOAL_NEAR_UP_M,
                                           meta["pitch"]["brg"] + GOAL_NEAR_BEARING_DEG)
            steady = {**view, "jit": [0.0, 0.0]}          # the same rectangle in both runs, whatever their jitter phase
            (fx, fy, _), (tx, ty, _) = to_pixel(steady, foot), to_pixel(steady, top)
            w, h = view["res"]
            rect = [max(int(min(fx, tx)) - 30, 0), max(int(fy), 0)]
            rect += [min(int(max(fx, tx)) + 30, w) - rect[0], min(int(ty) + 40, h) - rect[1]]
            frames = await self.each_frame("""(r) => [__hog.probe(0, 0).jit, __hog.probeRect(...r)]""", rect)
            records[dbg] = {tuple(j): np.array(g, float) for j, g in frames}
        gold = lambda g: int(((g[:, 0] == GOLD[0]) & (g[:, 4] == GOLD[1])).sum())
        pairs = [(records[0][j], records[DBG_THIN_OFF][j]) for j in records[0] if j in records[DBG_THIN_OFF]]
        self.assertEqual(len(pairs), JITTER_CYCLE)
        self.assertGreater(min(gold(a) for a, _ in pairs), 50, "the near goal is not in the frame")
        for a, b in pairs:
            near = (a[:, 5] < GOAL_FAR_M) | (b[:, 5] < GOAL_FAR_M)
            np.testing.assert_array_equal(a[near], b[near])

    async def test_the_lake_mirrors_the_meshes_and_nothing_raymarched(self):
        bare, meshes = await self.lake_codes(False), await self.lake_codes(True)
        self.assertGreater(len(bare), 1000)
        self.assertEqual(int((bare == RMAT_RAYMARCHED_CASTLE).sum()), 0, "a raymarched castle in the lake")
        self.assertEqual(int((bare == RMAT_MESH).sum()), 0)
        self.assertGreater(int((meshes == RMAT_MESH).sum()), 100, "the castle's meshes should show in the lake")
        self.assertEqual(int((meshes == RMAT_RAYMARCHED_CASTLE).sum()), 0)

    async def mirror_run(self, eye, yaw, pitch, dbg):
        """{jitter: (chunks the mirror drew, the shading pass's reflection record at every third lake pixel)} over a jitter
        cycle, seen from `eye` with debug bits DBG_REFLECTIONS | `dbg`."""
        await self.page.evaluate(f"__hog.dbg({DBG_REFLECTIONS | dbg})")
        frames = await self.page.evaluate("([p, y, q]) => { __hog.camera(p, y, q); return __hog.frames; }", [eye, yaw, pitch])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 12}", timeout=FRAME_TIMEOUT_S * 1000)
        frames = await self.each_frame("""() => {
            const v = __hog.probe(0, 0), [w, h] = v.res, g = __hog.probeRect(0, 0, w, h), c = __hog.probeColorRect(0, 0, w, h), out = [];
            for (let y = 1; y < h; y += 3) for (let x = 1; x < w; x += 3) {
              const i = y * w + x;
              if (g[i][0] === 2) out.push([c[4 * i], c[4 * i + 1], c[4 * i + 2]]);
            }
            return [v.jit, __hog.meshStats().reflected, out]; }""")
        return {tuple(j): (n, np.array(out, float).reshape(-1, 3)) for j, n, out in frames}

    async def test_the_mirror_draws_only_what_the_lake_can_show(self):
        # with the lake query off, the cull alone decides what the mirror draws in every frame: with it on, a view showing
        # no water skips the mirror once the query answers, whatever the cull passes. From Hogsmeade, looking away from
        # the lake, no mirror image can show through its water
        lake = await self.page.evaluate("DATA.meta.extras.lake_c")
        village = next(a for a in at.scene()["assets"] if a["id"] == "three-broomsticks")["components"][0]["at"]
        away = math.atan2(village[0] - lake[0], village[1] - lake[1])
        eye = await self.at_ground(*village, 40.0)
        culled, every = [await self.mirror_run(eye, away, -0.15, dbg)
                         for dbg in (DBG_NO_LAKE_QUERY, DBG_EVERY_REFLECTION)]
        with self.subTest(view="Hogsmeade"):
            self.assertEqual({n for n, _ in culled.values()}, {0})
            self.assertGreater(min(n for n, _ in every.values()), 0, "the mirror's frustum holds no chunk here")
        # from over the lake's middle, the mirror leaves out reflections that would land on dry ground, and each lake
        # pixel reflects what it did with every chunk mirrored, ripples and all: the same flag and code exactly, the
        # same reflected distance within MIRROR_RT_RTOL. Mirroring every chunk (the debug path) moves a few of its
        # distances by that much between identical runs, which the culled mirror never does
        eye = await self.at_ground(*lake, 30.0)
        culled, every = [await self.mirror_run(eye, math.radians(LAKE_CULL_BEARING_DEG), -0.12, dbg)
                         for dbg in (DBG_NO_LAKE_QUERY, DBG_EVERY_REFLECTION)]
        pairs = [(culled[j], every[j]) for j in culled if j in every]
        with self.subTest(view="over the lake"):
            self.assertEqual(len(pairs), JITTER_CYCLE)
            self.assertTrue(all(a[0] < b[0] for a, b in pairs), "the mirror leaves nothing out here")
            self.assertGreater(min(len(a[1]) for a, _ in pairs), LAKE_PIXELS_MIN, "too little lake in view")
            for a, b in pairs:
                np.testing.assert_array_equal(a[1][:, [0, 2]], b[1][:, [0, 2]])
                np.testing.assert_allclose(a[1][:, 1], b[1][:, 1], rtol=MIRROR_RT_RTOL, atol=0.0)

    async def test_the_cone_pre_pass_leaves_the_scene_as_the_full_march_finds_it(self):
        # the cone pre-pass only starts each block's march later, through air it has proved empty; resuming with the
        # centre ray's step count moves where a grazing ray runs out of steps, so 1-2% of samples differ (test_shaders
        # pins the two copies of the step rule as equal)
        await self.page.evaluate("__hog.loadMeshes(null)")
        meta = await self.page.evaluate("DATA.meta")
        towers = np.array([t[:2] for t in meta["towers"]])
        castle = await self.at_ground(*towers.mean(axis=0), 30.0)
        for name, target, back, up, bearing in (("castle", castle, 500.0, 120.0, 225.0),
                                                ("far hills", await self.at_ground(-5000.0, 900.0), 4000.0, 300.0, 90.0)):
            await self.look(target, back, up, bearing)
            runs = {}
            for dbg in (0, DBG_NO_CONE):
                await self.page.evaluate(f"__hog.dbg({dbg})")
                frames = await self.each_frame(f"""() => {{ const v = __hog.probe(0, 0), [w, h] = v.res, out = [];
                    for (let y = 3; y < h; y += 8) for (let x = 3; x < w; x += 8) {{ const p = __hog.probe(x, y); out.push([p.mat, p.t]); }}
                    return [v.jit, out]; }}""")                  # a whole jitter cycle each, so their phases pair up
                runs[dbg] = {tuple(j): np.array(out) for j, out in frames}
            pairs = [(runs[0][j], runs[DBG_NO_CONE][j]) for j in runs[0] if j in runs[DBG_NO_CONE]]
            with self.subTest(view=name):
                self.assertTrue(pairs, "no jitter phase in both runs")
                for a, b in pairs:
                    self.assertGreater(np.mean(a[:, 0] == b[:, 0]), 0.97)
                    hit = (a[:, 0] == b[:, 0]) & (a[:, 0] > 0)
                    self.assertGreater(np.mean(np.abs(a[hit, 1] - b[hit, 1]) <= 0.002 * b[hit, 1] + 0.05), 0.97)


LEVEL_BAND = float(re.search(r"const LEVEL_BAND = ([\d.]+);", (ROOT / "src" / "template_c.html").read_text(encoding="utf-8"))[1])
STILL_FRAMES = 90                           # of the 96 a still view renders before it stops (template_c.html IDLE_N)


def coarsest(g, d, scale, band):
    """The coarsest level of group g (a __hog.groups() record) whose switch distance, scaled, `band` times over, d has
    passed."""
    return max([0] + [k for k in (1, 2) if g["has"][k] and d >= g["switch"][k - 1] * scale * band])


class LevelOfDetailTests(RenderTestCase):
    """Levels of detail on the page: each view picks a group's level from its own output pixels (the mirror at half
    resolution, never the internal scale), keeps it while the camera waits inside the band past a switch distance,
    draws one level in the depth pre-pass and the G-buffer pass alike, and the shadow layers draw the casters."""

    async def tree(self):
        """(index, record) of the Willow's group, which has every level."""
        groups = await self.page.evaluate("__hog.groups()")
        i = next(i for i, g in enumerate(groups) if g["asset"] == "willow" and all(g["has"]))
        return i, groups[i]

    async def view_from(self, g, d):
        """The camera `d` metres from group g's box, as the page measures it (to the box's nearest point), over the box
        and the ground, looking at the box's middle; a few frames to pick."""
        lo, hi = np.array(g["lo"]), np.array(g["hi"])
        mid, up = (lo + hi) / 2, 30.0
        while True:
            eye = [float(mid[0]), float(hi[1] + up), float(lo[2] - math.sqrt(max(d * d - up * up, 1.0)))]
            if eye[1] > await self.ground(eye[0], eye[2]) + 2.0:
                break
            up += 10.0
        yaw, pitch = aim(eye, mid)
        frames = await self.page.evaluate("([p, y, q]) => { __hog.camera(p, y, q); return __hog.frames; }", [eye, yaw, pitch])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 4}", timeout=FRAME_TIMEOUT_S * 1000)

    async def picks(self, i):
        g = (await self.page.evaluate("__hog.groups()"))[i]
        return g["main"], g["mirror"]

    async def test_each_view_picks_its_level_from_its_own_output_pixels(self):
        i, g = await self.tree()
        scale = self.VIEWPORT["height"] / g["refRows"]             # the switch distances are for refRows output rows
        main = g["switch"][0] * scale
        await self.view_from(g, 0.2 * main)                         # from the near side, so each step goes coarser
        for d in (0.75 * main, 3.0 * g["switch"][1] * scale):
            await self.view_from(g, d)
            with self.subTest(distance=round(d)):
                self.assertEqual(await self.picks(i), (coarsest(g, d, scale, LEVEL_BAND), coarsest(g, d, scale / 2, LEVEL_BAND)))
        await self.view_from(g, 0.2 * main)
        await self.view_from(g, 0.75 * main)
        self.assertEqual(await self.picks(i), (0, 1), "between its switch distances the mirror is coarser")
        for s in (0.7, 0.25):                                       # adaptive scaling changes the internal pixels only
            try:
                frames = await self.page.evaluate(f"() => {{ __hog.scale({s}); return __hog.frames; }}")
                await self.page.wait_for_function(f"__hog.frames >= {frames + 4}", timeout=FRAME_TIMEOUT_S * 1000)
                with self.subTest(scale=s):
                    self.assertEqual(await self.picks(i), (0, 1))
            finally:                                                # the page is shared: back to QUERY's scale
                await self.page.evaluate(f"__hog.scale({re.search(r'scale=([0-9.]+)', self.QUERY).group(1)})")

    async def test_a_camera_waiting_past_a_switch_distance_keeps_its_level(self):
        i, g = await self.tree()
        main = g["switch"][0] * self.VIEWPORT["height"] / g["refRows"]
        for start, level in ((0.4 * main, 0), (1.6 * main, 1)):     # arriving from nearer, then from farther
            await self.view_from(g, start)
            await self.view_from(g, 1.1 * main)                     # inside the band: either level may stand
            with self.subTest(arriving="from farther" if level else "from nearer"):
                found = await self.each_frame(f"() => __hog.groups()[{i}].main", n=STILL_FRAMES)
                self.assertEqual(set(found), {level})

    async def test_the_pre_pass_and_the_g_buffer_pass_draw_one_level(self):
        # just past the Willow's switch, the pass that ends the march at the nearest mesh and the pass that writes it
        # must draw the same triangles: where they differ, the march gives way to a mesh the G-buffer never gets, and
        # leaves sky below the horizon
        i, g = await self.tree()
        main = g["switch"][0] * self.VIEWPORT["height"] / g["refRows"]
        await self.view_from(g, 0.4 * main)
        await self.view_from(g, 1.25 * main)
        self.assertEqual((await self.picks(i))[0], 1)
        frames = await self.each_frame("""() => { const v = __hog.probe(0, 0), [w, h] = v.res, g = __hog.probeRect(0, 0, w, h);
            let holes = 0, objects = 0;
            for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
              const p = g[y * w + x], below = (2 * (y + 0.5 + v.jit[1]) - h) / h * v.tanH * v.rot[4] + v.rot[7] < -0.05;
              holes += p[0] === 0 && below; objects += p[0] >= 4;
            }
            return [holes, objects]; }""")
        self.assertGreater(min(o for _, o in frames), 100, "the Willow is not in the frame")
        self.assertEqual([h for h, _ in frames], [0] * len(frames))

    async def test_the_shadow_layers_draw_the_shadow_casters(self):
        groups = await self.page.evaluate("__hog.groups()")
        casters = sum(1 for g in groups if g["has"][3])
        self.assertGreater(casters, 0)
        frames = await self.page.evaluate("() => { __hog.tod(15.25); return __hog.frames; }")   # a new sun: layers drawn again
        await self.page.wait_for_function(f"__hog.frames >= {frames + 3}", timeout=FRAME_TIMEOUT_S * 1000)
        drawn = await self.page.evaluate("__hog.meshStats().casters")
        self.assertGreaterEqual(drawn, casters)


class DirtyStateTests(RenderTestCase):
    """What a change of state must refresh: a resize the mesh targets, a new sun the mesh shadow layers, reflections
    turned back on the mirror, a new payload its shadows and the temporal history. A level crossing is
    LevelOfDetailTests', a still view and its resumption test_ui's."""

    async def lake_box(self):
        """The lake view (debug bit 128 set), and a 30 m box where the centre pixel's reflected ray meets it 100 m past
        the water, as test_render_castle's reflection test places one."""
        view = await self.lake_view(-0.6)
        await self.show([], **view)
        water = await self.centre()
        w, h = water["res"]
        origin, direction = ray(water, w // 2, h // 2)
        centre = origin + direction * water["t"] + direction * np.array([1.0, -1.0, 1.0]) * 100.0
        return view, (*(centre - 15.0), *(centre + 15.0))

    async def mirrored(self):
        """Through a jitter cycle: the least share of lake pixels within MIRROR_PROBE_PX of the centre whose shading
        pass reports a mirrored mesh (debug bit 128), and the most holes (sky below the horizon) over the frame."""
        frames = await self.each_frame("""() => {
            const v = __hog.probe(0, 0), [w, h] = v.res, g = __hog.probeRect(0, 0, w, h), c = __hog.probeColorRect(0, 0, w, h);
            let lake = 0, mirrored = 0, holes = 0;
            for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
              const i = y * w + x, below = (2 * (y + 0.5 + v.jit[1]) - h) / h * v.tanH * v.rot[4] + v.rot[7] < -0.05;
              holes += g[i][0] === 0 && below;
              if (g[i][0] === 2 && Math.abs(x + 0.5 - w / 2) < %d && Math.abs(y + 0.5 - h / 2) < %d) { lake++; mirrored += c[4 * i] > 0.5; }
            }
            return [lake ? mirrored / lake : -1, holes]; }""" % (MIRROR_PROBE_PX, MIRROR_PROBE_PX))
        return min(s for s, _ in frames), max(n for _, n in frames)

    async def from_above(self, at, posts, hour, regions):
        """Over `at` (east, north), looking down from SUN_VIEW_M, with `posts` (east, north) POST_M tall and POST_WIDE_M
        across, a shadow layer of `regions`, and the sun at `hour`: the probed view."""
        ground = await self.ground(*at)
        half = POST_WIDE_M / 2
        boxes = [(x - half, ground - 1.0, z - half, x + half, ground + POST_M, z + half) for x, z in posts]
        frames = await self.page.evaluate("(h) => { __hog.tod(h); return __hog.frames; }", hour)
        await self.page.wait_for_function(f"__hog.frames >= {frames + 2}", timeout=FRAME_TIMEOUT_S * 1000)
        await self.show(boxes, at=at, yaw=0.0, pitch=-1.5, height=SUN_VIEW_M, shadow=regions)
        return await self.centre()

    async def shade(self, view, at, hour):
        """The shadow pass's sun term where a POST_M post standing at `at` shades the ground at mid-height at `hour`."""
        ground = await self.ground(*at)
        sun = np.array((await self.page.evaluate("(h) => __hog.light(h)", hour))["dir"])
        spot = np.array([at[0], ground + POST_M / 2, at[1]]) - sun * (POST_M / 2) / sun[1]
        x, y, _ = to_pixel(view, spot)
        return await self.page.evaluate("([x, y]) => __hog.probeSun(x, y)", [int(round(x)), int(round(y))])

    def regions(self, at, ground):
        """A fine shadow layer round `at` listed before a scene-wide one, so a 2 m post's shadow is sharp."""
        mid = np.array([at[0], ground, at[1]])
        return [(*(mid - 80.0), *(mid + 80.0)), (*(mid - 3000.0), *(mid + 3000.0))]

    async def still(self):
        """Waits until the view has converged (the page's IDLE_N frames and some)."""
        frames = await self.page.evaluate("__hog.frames")
        await self.page.wait_for_function(f"__hog.frames >= {frames + STILL_WAIT}", timeout=FRAME_TIMEOUT_S * 1000)

    async def test_a_resize_rebuilds_the_mesh_distance_and_the_mirror(self):
        # the targets the mesh passes draw into follow the internal size: a mesh distance left at another size leaves
        # holes, a mirror left at another size reflects the box where its lookups no longer land
        view, box = await self.lake_box()
        found = {}
        for size in (self.VIEWPORT, {"width": 800, "height": 600}, {"width": 1280, "height": 600}):
            await self.page.set_viewport_size(size)
            await self.show([box], **view)
            found[size["width"]] = await self.mirrored()
        for width, (share, holes) in found.items():
            with self.subTest(width=width):
                self.assertGreater(share, MIRRORED_MIN)
                self.assertEqual(holes, 0)

    async def test_a_new_sun_moves_the_mesh_shadow(self):
        # the mesh shadow layers are drawn again when the sun moves: left as they were, the post's shadow stays where
        # the first sun cast it
        at = SUNLIT
        view = await self.from_above(at, [at], SUN_HOURS[0], self.regions(at, await self.ground(*at)))
        for hour in SUN_HOURS:                                     # the payload stays: only the sun moves
            frames = await self.page.evaluate("(h) => { __hog.tod(h); return __hog.frames; }", hour)
            await self.page.wait_for_function(f"__hog.frames >= {frames + 3}", timeout=FRAME_TIMEOUT_S * 1000)
            with self.subTest(hour=hour):
                for other in SUN_HOURS:
                    sun = await self.shade(view, at, other)
                    if other == hour:
                        self.assertLess(sun, SHADE_MAX)
                    else:
                        self.assertGreater(sun, SUNLIT_MIN)

    async def test_reflections_turned_back_on_mirror_the_scene_as_it_is(self):
        # with reflections off the lake shows no mesh; turned back on after the box has moved out of the mirror's
        # middle, the mirror must have been drawn again, not kept from before
        view, box = await self.lake_box()
        await self.show([box], **view)
        self.assertGreater((await self.mirrored())[0], MIRRORED_MIN)
        frames = await self.page.evaluate("() => { __hog.q(0); return __hog.frames; }")
        await self.page.wait_for_function(f"__hog.frames >= {frames + 3}", timeout=FRAME_TIMEOUT_S * 1000)
        with self.subTest(reflections="off"):
            self.assertEqual((await self.mirrored())[0], 0.0)
        await self.show([tuple(v + (MIRROR_AWAY_M if k in (0, 3) else 0.0) for k, v in enumerate(box))], **view)
        frames = await self.page.evaluate("() => { __hog.q(1); return __hog.frames; }")
        await self.page.wait_for_function(f"__hog.frames >= {frames + 3}", timeout=FRAME_TIMEOUT_S * 1000)
        with self.subTest(reflections="on again"):
            self.assertEqual((await self.mirrored())[0], 0.0)

    async def answered(self):
        """[lake pixels, chunks the mirror drew] in the latest frame, LAKE_ANSWER frames on: the lake query answered."""
        frames = await self.page.evaluate("__hog.frames")
        await self.page.wait_for_function(f"__hog.frames >= {frames + LAKE_ANSWER}", timeout=FRAME_TIMEOUT_S * 1000)
        return await self.page.evaluate("""() => { const [w, h] = __hog.probe(0, 0).res;
            return [__hog.probeRect(0, 0, w, h).filter((p) => p[0] === 2).length, __hog.meshStats().reflected]; }""")

    async def first_frame(self, change, arg):
        """The chunks the mirror drew in the first frame after `change(arg)`, an async JS function's source."""
        return await self.page.evaluate(f"""async ([arg, ms]) => {{
            {NEXT_FRAME}
            await ({change})(arg);
            return await nextFrame(ms) ? __hog.meshStats().reflected : -1; }}""", [arg, FRAME_TIMEOUT_S * 1000])

    async def test_the_mirror_is_drawn_only_while_lake_water_shows(self):
        # over the lake's middle a wall just ahead hides all its water, while the tower beyond passes the mirror's own
        # cull: once the lake query has answered for that still view, the mirror is not drawn. A jump, a turn or a new
        # payload that shows the water draws it in its first frame, before any answer, and it stays drawn
        lake = await self.page.evaluate("DATA.meta.extras.lake_c")
        water = await self.ground(*lake)
        x, z = lake
        wall = (x - WALL_HALF_M, water - 5.0, z + WALL_AHEAD_M, x + WALL_HALF_M, water + WALL_TOP_M, z + WALL_AHEAD_M + 2.0)
        tower = (x - TOWER_HALF_M, water - 5.0, z + TOWER_AHEAD_M - TOWER_HALF_M,
                 x + TOWER_HALF_M, water + TOWER_TOP_M, z + TOWER_AHEAD_M + TOWER_HALF_M)
        hide = lambda: self.show([wall, tower], at=lake, yaw=0.0, pitch=LAKE_PITCH, height=LAKE_EYE_M)
        try:
            await self.page.evaluate(f"__hog.dbg({DBG_NO_LAKE_QUERY})")
            await hide()
            culled = (await self.answered())[1]
        finally:
            await self.page.evaluate("__hog.dbg(0)")
        with self.subTest(view="hidden"):
            self.assertGreater(culled, 0, "the mirror's cull leaves nothing to skip here")
            self.assertEqual(await self.answered(), [0, 0])
        camera = "([p, y, q]) => __hog.camera(p, y, q)"
        changes = {"a jump": (camera, [[x, water + LAKE_EYE_M, z + LAKE_JUMP_M], 0.0, LAKE_PITCH]),
                   "a turn": (camera, [[x, water + LAKE_EYE_M, z], LAKE_TURN_RAD, LAKE_PITCH]),
                   "a new payload": ("(b) => __hog.loadMeshes(b)", payload(tower))}
        for name, (change, arg) in changes.items():
            await hide()
            hidden = await self.answered()
            first = await self.first_frame(change, arg)
            shown = await self.answered()
            with self.subTest(change=name):
                self.assertEqual(hidden, [0, 0])
                self.assertGreater(first, 0, "the mirror waited for the lake query")
                self.assertGreater(shown[0], LAKE_PIXELS_MIN)
                self.assertGreater(shown[1], 0, "the mirror stayed off with the lake in view")

    async def test_a_new_payload_brings_its_own_shadow_and_no_ghost_of_the_last(self):
        # a payload replaced while the view is still: the shadow layers are drawn for the new casters, and the temporal
        # history starts again, so the old post neither shades the ground nor lingers on screen
        at, away = SUNLIT, (SUNLIT[0] + GHOST_AWAY_M, SUNLIT[1])
        ground = await self.ground(*at)
        regions = self.regions(at, ground)
        view = await self.from_above(at, [at], GHOST_HOUR, regions)
        await self.still()
        x, y, _ = to_pixel(view, (at[0], ground + POST_M, at[1]))          # the post's top, seen from above
        w, h = view["res"]
        spot = [(x + 0.5) / w, (y + 0.5) / h]
        post = np.array(await self.page.evaluate("(f) => __hog.probeScreen(...f)", spot), float)
        self.assertLess(await self.shade(view, at, GHOST_HOUR), SHADE_MAX)
        half = POST_WIDE_M / 2
        body = payload((away[0] - half, ground - 1.0, away[1] - half, away[0] + half, ground + POST_M, away[1] + half))
        body["shadow"] = [[float(v) for v in r] for r in regions]
        first = await self.page.evaluate("""async ([b, f, ms]) => {
            %s
            await __hog.loadMeshes(b);
            await nextFrame(ms);
            const history = __hog.history();
            await nextFrame(ms);
            return [history, __hog.probeScreen(...f)]; }""" % NEXT_FRAME, [body, spot, FRAME_TIMEOUT_S * 1000])
        history, first = first[0], np.array(first[1], float)
        await self.still()
        settled = np.array(await self.page.evaluate("(f) => __hog.probeScreen(...f)", spot), float)
        self.assertGreater(float(np.abs(post - settled).max()), GHOST_CONTRAST, "the post does not stand out from the ground")
        with self.subTest(check="history"):
            # the first frame after the swap starts the history again; the neighbourhood clamp would hide a ghost
            # either way, but a swap taken for motion keeps blending the old frames for a while
            self.assertLessEqual(history, 2)
            self.assertLess(float(np.abs(first - settled).max()), GHOST_LEFT)
        with self.subTest(check="shadows"):
            self.assertGreater(await self.shade(view, at, GHOST_HOUR), SUNLIT_MIN)
            self.assertLess(await self.shade(view, away, GHOST_HOUR), SHADE_MAX)


# the near shadow layer (template_b.html nearShadow): a wall NEAR_WALL_M long and tall at SUNLIT, facing NEAR_SUN_OFF_DEG
# off the sun at NEAR_HOUR, with an eave EAVE_OUT_M out and EAVE_THICK_M thick along its top; a post NEAR_POST_M away
# spreads the scene-wide layer to about 0.75 m texels, so only the near layer can draw the eave's shadow, a band about
# 0.6 m deep. Seen from NEAR_EYE_M, the wall from SHADE_FROM_M under the eave to EDGE_M short of the eave's shadow line
# must lie in shade, and from WALL_LIT_M down in sun, also when GRAZING_OFF_DEG off the sun; from FAR_EYE_M the near
# layer does not reach the wall
NEAR_HOUR = 13.0
NEAR_SUN_OFF_DEG, GRAZING_OFF_DEG = 30.0, 80.0
NEAR_WALL_M = 20.0
EAVE_OUT_M, EAVE_THICK_M = 0.6, 0.2
NEAR_POST_M = 1500.0
NEAR_EYE_M, FAR_EYE_M = 25.0, 150.0
WALL_LIT_M = (2.0, 6.0)
NEAR_SHADE_MAX, NEAR_LIT_MIN = 0.3, 0.9
NEAR_STILL_FRAMES = 30
WALL_END_M = 3.0                            # samples this far from either end: the sun's slant cuts the band there
# the near layer's bias, seen close up: from CLOSE_EYE_M out and CLOSE_UNDER_M under the eave, looking at CLOSE_AIM_M under
# it, at internal scale 1 (a pixel under a centimetre of wall; the shading pass compares each one). The sampled pixels
# from SHADE_FROM_M under the eave down to EDGE_M short of the eave's shadow line must all lie in shade, and from
# CLOSE_LIT_M past the line in sun. Percentage-closer filtering blurs each edge of the band over about 0.3 m of this wall;
# at 13:00 (sun 41 degrees) the shipped bias (5 cm and 1.5 texels) measured the sun term crossing 0.3 about 0.215 m under
# the eave, the static layers' (10 cm and 2 texels) about 0.28 m, and the band's lower edge crossing it again about 0.17 m
# short of the line
CLOSE_EYE_M, CLOSE_UNDER_M, CLOSE_AIM_M = 6.0, 1.0, 0.4
SHADE_FROM_M = 0.245
EDGE_M = 0.2
CLOSE_LIT_M = (0.3, 1.5)


class NearShadowTests(RenderTestCase):
    """The near shadow layer: about 9 cm texels round the camera's ground point, drawn again when the camera's cell, the
    sun or the payload changes, never while the view stands still."""

    async def eave_wall(self, eye_m, off_deg=NEAR_SUN_OFF_DEG, eave=True):
        """Shows the wall, with its eave or without, facing `off_deg` off the sun at NEAR_HOUR, from `eye_m` in front of
        it: the view, the wall face's outward normal, its top's world point and the direction along it."""
        frames = await self.page.evaluate("(h) => { __hog.tod(h); return __hog.frames; }", NEAR_HOUR)
        await self.page.wait_for_function(f"__hog.frames >= {frames + 2}", timeout=FRAME_TIMEOUT_S * 1000)
        sun = np.array((await self.page.evaluate("(h) => __hog.light(h)", NEAR_HOUR))["dir"])
        facing = math.atan2(sun[0], sun[2]) + math.radians(off_deg)
        normal = np.array([math.sin(facing), 0.0, math.cos(facing)])
        along = np.array([normal[2], 0.0, -normal[0]])
        x, z = SUNLIT
        ground = await self.ground(x, z)
        base = np.array([x, ground, z])
        top = ground + NEAR_WALL_M
        rect = lambda inner, outer: Polygon([(c[0], c[2]) for c in (base + along * a + normal * b for a, b in (
            (-NEAR_WALL_M / 2, inner), (NEAR_WALL_M / 2, inner), (NEAR_WALL_M / 2, outer), (-NEAR_WALL_M / 2, outer)))])
        walls = [(rect(-0.5, 0.0), ground - 1.0, top)] + [(rect(-0.5, EAVE_OUT_M), top, top + EAVE_THICK_M)] * eave
        p = base + np.array([sun[2], 0.0, -sun[0]]) / math.hypot(sun[0], sun[2]) * NEAR_POST_M
        post = (p[0] - 1.0, ground, p[2] - 1.0, p[0] + 1.0, ground + 10.0, p[2] + 1.0)
        eye = base + normal * eye_m
        await self.show([post], at=(eye[0], eye[2]), yaw=math.atan2(-normal[0], -normal[2]), pitch=0.15,
                        height=NEAR_WALL_M * 0.5 + ground - await self.ground(eye[0], eye[2]), walls=walls)
        return await self.centre(), normal, np.array([x, top, z]), along

    async def step_to(self, eye_m, normal, top, under=NEAR_WALL_M * 0.5, pitch=0.15):
        """The camera alone moved to `eye_m` in front of the wall shown last and `under` metres under its top, looking
        `pitch` up, the payload and the sun as they were: the view."""
        eye = np.array([top[0], top[1] - under, top[2]]) + normal * eye_m
        frames = await self.page.evaluate("([p, y, q]) => { __hog.camera(p, y, q); return __hog.frames; }",
                                          [eye.tolist(), math.atan2(-normal[0], -normal[2]), pitch])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 3}", timeout=FRAME_TIMEOUT_S * 1000)
        return await self.centre()

    async def wall_sun(self, view, normal, top, along):
        """(depth under the eave, sun term) of the wall face's pixels, every other internal pixel, away from its
        ends."""
        found = await self.page.evaluate("""([mat]) => {
            const v = __hog.probe(0, 0), [w, h] = v.res, g = __hog.probeRect(0, 0, w, h), out = [];
            for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
              const p = g[y * w + x];
              if (p[0] === mat) out.push([x, y, p[1], p[2], p[3], p[5], __hog.probeSun(x, y)]);
            }
            return out; }""", [WALL[0]])
        out = []
        for x, y, nx, ny, nz, t, sun in found:
            if np.dot([nx, ny, nz], normal) > 0.99:
                origin, direction = ray(view, x, y)
                point = origin + direction * t
                if abs(np.dot(point - top, along)) < NEAR_WALL_M / 2 - WALL_END_M:
                    out.append((float(top[1] - point[1]), sun))
        return out

    async def eave_line(self):
        """How far under the eave its shadow line falls on the wall at NEAR_HOUR."""
        sun = np.array((await self.page.evaluate("(h) => __hog.light(h)", NEAR_HOUR))["dir"])
        return EAVE_OUT_M * math.tan(math.asin(sun[1])) / math.cos(math.radians(NEAR_SUN_OFF_DEG))

    async def test_a_near_eave_shades_the_wall_beneath_it(self):
        # only the near layer draws the eave's shadow, and its bias lets the shade reach to within SHADE_FROM_M of the
        # eave: seen close up, the wall from there to the eave's shadow line is shaded, and the wall past it sunlit
        _, normal, top, along = await self.eave_wall(NEAR_EYE_M)
        line = await self.eave_line()
        try:
            await self.page.evaluate("__hog.scale(1)")
            view = await self.step_to(CLOSE_EYE_M, normal, top, CLOSE_UNDER_M,
                                      math.atan2(CLOSE_UNDER_M - CLOSE_AIM_M, CLOSE_EYE_M))
            samples = await self.wall_sun(view, normal, top, along)
        finally:                                                    # the page is shared: back to QUERY's scale
            await self.page.evaluate(f"__hog.scale({re.search(r'scale=([0-9.]+)', self.QUERY).group(1)})")
        band = [(round(d, 3), s) for d, s in samples if SHADE_FROM_M <= d < line - EDGE_M]
        past = [s for d, s in samples if line + CLOSE_LIT_M[0] < d < line + CLOSE_LIT_M[1]]
        self.assertGreater(len(band), 100, "too little of the band under the eave in view")
        self.assertGreater(len(past), 100)
        self.assertEqual([x for x in band if x[1] >= NEAR_SHADE_MAX], [], "sunlit under the eave")
        self.assertGreater(min(past), NEAR_LIT_MIN)

    async def test_a_near_wall_lit_at_a_grazing_angle_does_not_shade_itself(self):
        # the near layer's bias is small enough for the band under an eave, and must still keep a face the sun only
        # grazes from comparing against its own texels: without its eave nothing else can shade this wall
        view, normal, top, along = await self.eave_wall(NEAR_EYE_M, GRAZING_OFF_DEG, eave=False)
        below = [s for d, s in await self.wall_sun(view, normal, top, along) if WALL_LIT_M[0] < d < WALL_LIT_M[1]]
        self.assertGreater(len(below), 50)
        self.assertGreater(min(below), NEAR_LIT_MIN)

    async def test_the_near_layer_follows_the_camera_and_rests_with_it(self):
        # from afar the near layer lies round the camera, short of the wall; the camera alone walked up to it (the
        # payload and the sun as they were, so only its cell can call for the layer), the layer is drawn once for the
        # new cell, then not again while the view stands still
        _, normal, top, along = await self.eave_wall(FAR_EYE_M)
        before = (await self.page.evaluate("__hog.meshStats()"))["near"]
        view = await self.step_to(NEAR_EYE_M, normal, top)
        drawn = (await self.page.evaluate("__hog.meshStats()"))["near"]
        frames = await self.page.evaluate("__hog.frames")
        await self.page.wait_for_function(f"__hog.frames >= {frames + NEAR_STILL_FRAMES}", timeout=FRAME_TIMEOUT_S * 1000)
        with self.subTest(check="drawn for the new cell"):
            self.assertGreater(drawn, before)
        with self.subTest(check="still"):
            self.assertEqual((await self.page.evaluate("__hog.meshStats()"))["near"], drawn)
        line = await self.eave_line()
        band = [(round(d, 3), s) for d, s in await self.wall_sun(view, normal, top, along)
                if SHADE_FROM_M <= d < line - EDGE_M]
        with self.subTest(check="shades the wall"):
            self.assertGreater(len(band), 20, "too little of the band under the eave in view")
            self.assertEqual([x for x in band if x[1] >= NEAR_SHADE_MAX], [], "sunlit under the eave")


# thin casters' sun shadows: a pole's shadow on open ground has no gap along its length, though it is narrower than a
# shadow-pass block (the shading pass looks the mesh casters up for each pixel), than a near-layer texel or than a static
# layer's (each layer draws a part narrower than its texels as lines too). The pole stands THIN_POLE_M tall at SUNLIT,
# upright or leaning THIN_LEAN_M across the sun, with posts NEAR_POST_M / 2 away on either side spreading the scene-wide
# layer to about 0.75 m texels round it (a layer leaves out a few metres at its edges, so the pole stands in its middle);
# seen from THIN_VIEW_M straight above at THIN_HOUR, its shadow's centre line over THIN_SPAN of its length
# is stepped every THIN_STEP_PX internal pixels, and each step's darkest open ground within THIN_PAD_PX must be shaded
# below THIN_GAP_SUN in the sun term the shading pass uses there. A lattice of THIN_LATTICE_BARS such poles NEAR_TEXEL_BAR_M
# thick, THIN_LATTICE_PITCH_M apart (under two near-layer texels, as a gate's pickets are), covers about a quarter of the
# band it shades: its open ground there, over THIN_SPAN of its length and THIN_PAD_PX in from its outer poles' shadows,
# must average a sun term above THIN_LATTICE_SUN_MIN, which lines a texel wide would merge it far below
THIN_HOUR = 14.0
THIN_POLE_M, THIN_LEAN_M = 9.0, 3.0
THIN_VIEW_M = 40.0
THIN_SPAN = (0.15, 0.85)
THIN_STEP_PX, THIN_PAD_PX = 0.5, 1
THIN_GAP_SUN = 0.75
GOAL_POLE_R_M = 0.09                        # mesh_grounds.hoops: the goals' poles
NEAR_TEXEL_BAR_M = 0.03                     # under a third of a near-layer texel (template_b.html NEAR_M / SH_RES)
THIN_LATTICE_BARS, THIN_LATTICE_PITCH_M = 15, 0.13     # mesh_grounds' gates: pickets 0.135 m apart
THIN_LATTICE_SUN_MIN = 0.55
DBG_NO_NEAR_LAYER = 16384
DBG_SUN = 64                                # the shading pass shows its sun term


class ThinShadowTests(RenderTestCase):
    """Thin casters' sun shadows stay whole: a goal pole's (PS 8: "three golden poles ... fifty feet high"), whose shadow
    is a line narrower than a shadow-pass block, and poles thinner than a near-layer texel or than a static layer's."""

    async def shadow_frame(self, caster, dbg=0):
        """`caster` (a function of (base, sun direction, sideways) giving its mesh and points on it) shown at SUNLIT
        and seen from above its shadow with debug bits `dbg`: the view, each internal pixel's material and the shading
        pass's sun term (debug bit DBG_SUN), and the pixels where the sun casts the points onto the ground."""
        frames = await self.page.evaluate("([h, d]) => { __hog.tod(h); __hog.dbg(d); return __hog.frames; }", [THIN_HOUR, dbg])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 2}", timeout=FRAME_TIMEOUT_S * 1000)
        sun = np.array((await self.page.evaluate("(h) => __hog.light(h)", THIN_HOUR))["dir"])
        side = np.array([sun[2], 0.0, -sun[0]]) / math.hypot(sun[0], sun[2])
        x, z = SUNLIT
        base = np.array([x, await self.ground(x, z), z])
        mesh, ends = caster(base, sun, side)
        posts = [box(q[0] - 1.0, q[0] + 1.0, q[1], q[1] + 10.0, q[2] - 1.0, q[2] + 1.0, WALL)
                 for q in (base + side * NEAR_POST_M / 2, base - side * NEAR_POST_M / 2)]
        body = json.loads(build_meshes.strip_js(build_meshes.encode_payload({"diagnostic": [mesh, *posts]}, SCENE)))

        async def landing(point):                  # where the sun ray through `point` meets the ground
            q = point
            for _ in range(3):
                g = await self.ground(q[0], q[2])
                q = point - sun * (point[1] - g) / sun[1]
            return q

        landed = [await landing(q) for q in ends]
        a, b = landed[0], landed[1]
        mid, along = (a + b) / 2, b - a
        eye = [float(mid[0]), float(mid[1] + THIN_VIEW_M), float(mid[2])]
        yaw = math.atan2(along[0], along[2]) + math.radians(30.0)  # the shadow runs across the pixel grid
        frames = await self.page.evaluate(
            "async ([b, p, y]) => { __hog.camera(p, y, -1.5); await __hog.loadMeshes(b); return __hog.frames; }",
            [body, eye, yaw])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 3}", timeout=FRAME_TIMEOUT_S * 1000)
        frames = await self.page.evaluate(f"() => {{ __hog.dbg({dbg | DBG_SUN}); return __hog.frames; }}")
        await self.page.wait_for_function(f"__hog.frames >= {frames + 2}", timeout=FRAME_TIMEOUT_S * 1000)
        view, mats, suns = await self.page.evaluate("""() => {
            const v = __hog.probe(0, 0), [w, h] = v.res, g = __hog.probeRect(0, 0, w, h), c = __hog.probeColorRect(0, 0, w, h);
            return [v, g.map((p) => p[0]), Array.from({ length: w * h }, (_, i) => c[4 * i])]; }""")
        w, h = view["res"]
        return (view, np.array(mats).reshape(h, w), np.array(suns, float).reshape(h, w),
                [np.array(to_pixel(view, q)[:2]) for q in landed])

    async def pole_shadow(self, pole, dbg=0):
        """For each step along `pole`'s shadow (`pole` as shadow_frame's caster, its points its centre line's two
        ends), the sun term at the darkest open ground within THIN_PAD_PX, seen with debug bits `dbg`."""
        _, mats, suns, (p0, p1) = await self.shadow_frame(pole, dbg)
        n = int(np.hypot(*(p1 - p0)) / THIN_STEP_PX)
        out = []
        for k in range(n + 1):
            px, py = p0 + (THIN_SPAN[0] + (THIN_SPAN[1] - THIN_SPAN[0]) * k / max(n, 1)) * (p1 - p0)
            x0, y0 = int(px) - THIN_PAD_PX, int(py) - THIN_PAD_PX
            m = mats[max(y0, 0):y0 + 2 * THIN_PAD_PX + 1, max(x0, 0):x0 + 2 * THIN_PAD_PX + 1]
            s = suns[max(y0, 0):y0 + 2 * THIN_PAD_PX + 1, max(x0, 0):x0 + 2 * THIN_PAD_PX + 1]
            if (m == MAT_GROUND).any():
                out.append(float(s[m == MAT_GROUND].min()))
        self.assertGreater(len(out), 40, "too little of the shadow in view")
        return np.array(out)

    def assertWhole(self, darkest):
        gaps = np.flatnonzero(darkest >= THIN_GAP_SUN)
        self.assertEqual(len(gaps), 0, f"lit at {len(gaps)} of {len(darkest)} steps along the shadow (darkest step "
                                       f"{darkest.min():.2f}, brightest {darkest.max():.2f})")

    async def test_a_goal_poles_shadow_narrower_than_a_shadow_block_has_no_gap(self):
        # an upright goal pole, two near-layer texels across: the near layer holds it whole, but its shadow seen from
        # above is a line a pixel or two wide, which a quarter-resolution shadow pass sees only where it crosses a
        # block's centre
        pole = lambda base, sun, side: (lathe(base[0], base[2], [(GOAL_POLE_R_M, base[1] - 0.5),
                                                                 (GOAL_POLE_R_M, base[1] + THIN_POLE_M)], WALL, 12, cap=False),
                                        (base, base + np.array([0.0, THIN_POLE_M, 0.0])))
        self.assertWhole(await self.pole_shadow(pole))

    async def test_a_bars_shadow_thinner_than_a_near_layer_texel_has_no_gap(self):
        def pole(base, sun, side):
            top = base + np.array([0.0, THIN_POLE_M, 0.0]) + side * THIN_LEAN_M
            return bar(base - np.array([0.0, 0.5, 0.0]), top, NEAR_TEXEL_BAR_M, WALL), (base, top)
        self.assertWhole(await self.pole_shadow(pole))

    async def test_a_goal_poles_shadow_thinner_than_a_static_layer_texel_has_no_gap(self):
        # past the near layer (here: with it left out) the scene-wide layer's texels are about four poles across
        def pole(base, sun, side):
            top = base + np.array([0.0, THIN_POLE_M, 0.0]) + side * THIN_LEAN_M
            return bar(base - np.array([0.0, 0.5, 0.0]), top, 2 * GOAL_POLE_R_M, WALL), (base, top)
        self.assertWhole(await self.pole_shadow(pole, DBG_NO_NEAR_LAYER))

    async def test_a_lattice_finer_than_two_near_layer_texels_keeps_its_average_shade(self):
        # a gate's pickets: lines a near-layer texel wide would merge their shadows into a solid one, so a lattice's
        # parts draw none (mesh_thin.CROWD_M), and the layers' own coverage of them keeps the band as dark as it is on
        # average
        offsets = (np.arange(THIN_LATTICE_BARS) - (THIN_LATTICE_BARS - 1) / 2) * THIN_LATTICE_PITCH_M

        def lattice(base, sun, side):
            top = base + np.array([0.0, THIN_POLE_M, 0.0]) + side * THIN_LEAN_M
            bars = [bar(base - np.array([0.0, 0.5, 0.0]) + side * o, top + side * o, NEAR_TEXEL_BAR_M, WALL) for o in offsets]
            return concat(bars), [base, top] + [q + side * o for o, q in ((offsets[0], base), (offsets[0], top),
                                                                          (offsets[-1], top), (offsets[-1], base))]
        _, mats, suns, (_, _, b0, t0, t1, b1) = await self.shadow_frame(lattice)
        across = lambda f: [b0 + f * (t0 - b0), b1 + f * (t1 - b1)]
        band = Polygon([*across(THIN_SPAN[0]), *across(THIN_SPAN[1])[::-1]]).buffer(-THIN_PAD_PX)
        ys, xs = np.mgrid[0:suns.shape[0], 0:suns.shape[1]]
        open_ground = shapely.contains_xy(band, xs, ys) & (mats == MAT_GROUND)
        self.assertGreater(open_ground.sum(), 100, "too little of the lattice's shadow in view")
        self.assertGreater(suns[open_ground].mean(), THIN_LATTICE_SUN_MIN)
