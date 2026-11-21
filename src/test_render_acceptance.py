"""Acceptance checks for claims of the plan's acceptance table that no other test covers: foreground masonry hides the
plants behind glass, panes or none (GH2); a pause past glass and the camera moving on leave no persistent ghost of it
(GH4); the harbour mouth reads as a recess from oblique views (HB1); the page's programs link and draw every path
without GL or page errors on the adapter it runs on, which the render suite's runs on both GPUs make per adapter
(PO1/PO2); and resizing, a new sun, a new quality and a converged pause keep the tour and its controls (SC7). Each has a
fault it must catch in tools/mutate/acceptance.py."""
import functools
import json
import math
from pathlib import Path
import re
import sys
import unittest

import numpy as np
from playwright.async_api import expect
from shapely.geometry import box as rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))

import build_meshes  # noqa: E402
import mesh_grounds  # noqa: E402
import test_render_core  # noqa: E402   (a module, not its classes: their tests stay theirs)
from mesh_greenhouses import FOLIAGE, WALL_STONE  # noqa: E402
from mesh_source import load_data  # noqa: E402
from render_testing import (EDGE_MARGIN_M, FRAME_TIMEOUT_S, JITTER_CYCLE, LAWN, SCENE, WEST, RenderTestCase,  # noqa: E402
                            ahead, box_hit, ray)
from tools.inspection.frame import rays  # noqa: E402

# the opening is 4.5 m wide and 2 m high under its ivy (harbour-heads-down): from 40 degrees off its facing and 80 m
# out, half the rays through it meet the water, the ivy or a channel wall within about a metre of the cliff face beside
# it, far short of the recess's 8.5 m behind that face. Lane A is rebuilding the harbour mouth.
EXPECTED_FAILURES = ("HarbourMouthViewTests.test_the_harbour_mouth_reads_as_recessed_from_oblique_views",)

IDLE_N = int(re.search(r"const IDLE_N = (\d+);", (ROOT / "src" / "template_c.html").read_text(encoding="utf-8"))[1])
# GH2: the probed patch round the centre (inside both the masonry, about 27 internal pixels across, and the plant, about
# 21) and its distance to the analytic hit, as test_mesh_in_front_of_terrain_wins_with_the_object_record takes it
PATCH_PX = 9
HIT_M = 0.02
# GH4: as test_moving_past_glass_leaves_no_ghost, which slides SLIDE_M north at a metre a frame; the pause comes
# PAUSE_AT_M in, with the panes still over the right part of the probed fifth, and the camera then moves on MOVE_ON_M
# past the slide's end. The panes tint the fifth by TINT_MIN or more (0-255) and may leave GHOST_SHARE of that tint;
# while paused they cover PAUSED_SHARE of it at least (measured: 0.44)
SLIDE_M, PAUSE_AT_M, MOVE_ON_M = 12, 6, 12
TINT_MIN, GHOST_SHARE, PAUSED_SHARE = 10.0, 0.05, 0.2
# HB1: two cameras OBLIQUE_DEG either side of the mouth's facing, HARBOUR_VIEW_M out and HARBOUR_EYE_M over the lake's
# level, as the tour views its subjects (its waypoints stand 23 to 180 m from theirs, 20 to 45 m up; 100 m out on the
# north side the shore's trees stand in the way). Rays through the opening are judged EDGE_IN_M inside its outline, the
# cliff face beside it over BESIDE_M either side; "most of the recess's depth" is RECESS_SHARE of it, and FRONT_M
# nearer than the face means something stands in front of the opening
OBLIQUE_DEG = 40.0
HARBOUR_VIEW_M, HARBOUR_EYE_M = 80.0, 20.0
EDGE_IN_M, BESIDE_M = 0.1, 2.0
RECESS_SHARE = 0.5
FRONT_M, BLOCKED_SHARE = 1.0, 0.05
OPENING_MIN, BESIDE_MIN = 100, 100        # samples over a jitter cycle: measured about 440 and 465
# PO1: a view renders all its passes in each of its first frames; the far-tile view is test_mesh_against_sky_and_far_tiles'
VIEW_FRAMES = JITTER_CYCLE
FAR_VIEW = {"at": LAWN, "yaw": WEST, "pitch": -0.05, "height": 400.0}
GREENHOUSES = ((30.0, 305.0, 26.0), (-20.0, 245.0, 2.0))   # the tour's waypoint: eye and target (east, north, metres up)
LAKE_EYE_M, CASTLE_LOOK_M = 30.0, 40.0                     # over the lake's middle, at the castle (as the tour's opening)
# SC7: the tour plays LEAD_S before the changes and pauses PAUSE_S at least, so a restart or a clock running through the
# pause shows; the rendered camera's tour time is the last rendered frame's (dt at most 0.1 s a frame), found in steps of
# TOUR_STEP_S, so it may run CLOCK_SLACK_S ahead of the clock, and lies on the tour within TOUR_OFF_M (68 m/s at most)
CHOSEN_HOUR, CHOSEN_TEXT = 9.75, "09:45"
CHOSEN_QUALITY = ("high", 2)              # one press of Q from the page's medium
RESIZED = ({"width": 900, "height": 600}, {"width": 1280, "height": 640})
TOUR_SCALE = 0.35                         # PageTestCase's internal scale
LEAD_S, PAUSE_S = 3.0, 3.0
TOUR_STEP_S, CLOCK_SLACK_S, TOUR_OFF_M = 0.005, 0.25, 0.5

# records, from the page's start, each program it links, with its shaders' sources, and each one a draw call uses, since
# the start and since the test last emptied `recent`
RECORD_PROGRAMS = """(() => {
  const proto = WebGL2RenderingContext.prototype;
  const log = window.__programs = { linked: [], drawn: new Set(), recent: new Set(), current: null };
  const native = Object.fromEntries(["shaderSource", "attachShader", "linkProgram", "useProgram", "drawArrays",
                                     "drawElements", "drawArraysInstanced", "drawElementsInstanced"].map((k) => [k, proto[k]]));
  const draw = () => { log.drawn.add(log.current); log.recent.add(log.current); };
  proto.shaderSource = function (s, src) { s.__src = src; return native.shaderSource.call(this, s, src); };
  proto.attachShader = function (p, s) {
    (p.__src = p.__src || []).push(s.__src);
    return native.attachShader.call(this, p, s);
  };
  proto.linkProgram = function (p) { log.linked.push(p); return native.linkProgram.call(this, p); };
  proto.useProgram = function (p) { log.current = p; return native.useProgram.call(this, p); };
  proto.drawArrays = function (...a) { draw(); return native.drawArrays.apply(this, a); };
  proto.drawElements = function (...a) { draw(); return native.drawElements.apply(this, a); };
  proto.drawArraysInstanced = function (...a) { draw(); return native.drawArraysInstanced.apply(this, a); };
  proto.drawElementsInstanced = function (...a) { draw(); return native.drawElementsInstanced.apply(this, a); };
})();"""
# every GL error flag raised since the page started (getError's first call), then the programs' states, each named by
# its fragment shader's first lines; the page's own context, as getContext returns it to any caller
GL_STATE = """() => {
  const gl = document.getElementById("c").getContext("webgl2"), log = __programs, errors = [];
  for (let e = gl.getError(); e !== gl.NO_ERROR && errors.length < 8; e = gl.getError()) errors.push("0x" + e.toString(16));
  const name = (p) => ((p.__src || [])[1] || "?").split("\\n").slice(1, 3).join(" ").slice(0, 72);
  return { errors, lost: gl.isContextLost(), error: __hog.error || null, linked: log.linked.length,
           unlinked: log.linked.filter((p) => !gl.getProgramParameter(p, gl.LINK_STATUS)).map(name),
           undrawn: log.linked.filter((p) => !log.drawn.has(p)).map(name) }; }"""
RECT = "([x, y, w, h]) => ({ view: __hog.probe(0, 0), rect: __hog.probeRect(x, y, w, h) })"
SHOWN_SECOND = "+(document.getElementById('status').textContent.match(/flyover (\\d+)/) || [0, 0])[1]"
# the camera's tour time: the tour's time nearest the status line's second at which its path passes the last rendered
# frame's camera, with how far from it the path passes there
TOUR_TIME = """(step) => {
  const v = __hog.probe(0, 0), m = document.getElementById("status").textContent.match(/flyover (\\d+) \\//);
  let best = [Infinity, NaN];
  for (let t = (m ? +m[1] : 0) - 2; t <= (m ? +m[1] : 0) + 3; t += step) {
    const p = __hog.path(t).pos, d = Math.hypot(p[0] - v.pos[0], p[1] - v.pos[1], p[2] - v.pos[2]);
    if (d < best[0]) best = [d, t];
  }
  return { shown: m ? +m[1] : null, t: best[1], off: best[0], ms: performance.now() }; }"""
# the last rendered frame's sun and size, the canvas's, and the quality each program that has drawn since `recent` was
# emptied and reads one holds. The mesh trees' passes are left out: they read the quality only through frag1.glsl's
# ZERO, min(uQual, 0), which is 0 whatever the quality, and the page sets it in none of them
APPLIED = """(hour) => {
  const v = __hog.probe(0, 0), cv = document.getElementById("c"), gl = cv.getContext("webgl2");
  const trees = (p) => /^#define (TINST_PASS|TREE_DRAW|TREE_MESH) 1$/m.test((p.__src || [])[1] || "");
  const quality = [...__programs.recent].filter((p) => !trees(p)).map((p) => [p, gl.getUniformLocation(p, "uQual")])
                                         .filter(([, u]) => u !== null).map(([p, u]) => gl.getUniform(p, u));
  return { sun: v.sun, want: __hog.light(hour).dir, res: v.res, canvas: [cv.width, cv.height], quality }; }"""


def scene(*parts, panes=()):
    """Boxes (x0, y0, z0, x1, y1, z1), each with its (material, aux), and glazing slabs, as a payload encoded as
    build_meshes.py does: render_testing.payload with a material for each box."""
    meshes = [build_meshes.prism(rectangle(x0, z0, x1, z1), y0, y1, attr) for (x0, y0, z0, x1, y1, z1), attr in parts]
    meshes += [build_meshes.prism(rectangle(x0, z0, x1, z1), y0, y1, (build_meshes.MAT_GLASS, 0))
               for x0, y0, z0, x1, y1, z1 in panes]
    return json.loads(build_meshes.strip_js(build_meshes.encode_payload({"diagnostic": meshes}, SCENE)))


@functools.cache
def harbour():
    """(waterline origin (x, z), unit axis into the cliff, unit across, lake level, the back wall's distance in) of the
    harbour mouth in the page's data (meta.extras.harbour), as mesh_grounds.harbour_frame builds on it."""
    meta = load_data()["meta"]
    rec = meta["extras"]["harbour"]
    u = np.asarray(rec["axis"], float)
    return (np.asarray(rec["mouth"], float), u, np.array([-u[1], u[0]]), float(meta["lake"]),
            rec["depth"] - mesh_grounds.BACK_IN_M)


class GlassTests(RenderTestCase):
    settled_screen = test_render_core.MeshOcclusionTests.settled_screen

    async def load(self, body):
        frames = await self.page.evaluate("async (b) => { await __hog.loadMeshes(b); return __hog.frames; }", body)
        await self.page.wait_for_function(f"__hog.frames >= {frames + 3}")

    async def surfaces(self, front, behind):
        """[(material, aux, distance, analytic distance to `front`, to `behind`)] over the PATCH_PX square round the
        centre in each frame of a jitter cycle, for the pixels whose ray meets box `front` well inside its edges."""
        x0, y0 = (r // 2 - PATCH_PX // 2 for r in (await self.centre())["res"])
        out = []
        for frame in await self.each_frame(RECT, [x0, y0, PATCH_PX, PATCH_PX]):
            for i, (mat, _, _, _, aux, t) in enumerate(frame["rect"]):
                origin, direction = ray(frame["view"], x0 + i % PATCH_PX, y0 + i // PATCH_PX)
                if box_hit(origin, direction, front, -EDGE_MARGIN_M) is not None:
                    out.append((int(mat), round(aux), t,
                                box_hit(origin, direction, front), box_hit(origin, direction, behind)))
        return out

    async def test_foreground_masonry_hides_the_plants_behind_glass(self):
        """GH2 Glass/history: foreground masonry occludes the plants behind glass. Set as in
        test_glass_tints_what_it_covers_and_leaves_no_trace, with the greenhouses' foliage and the stone of their dwarf
        walls: through two panes the probed surface is the plant; with the masonry in front of it, every probed pixel
        the masonry covers is the masonry, at its own distance, nearer than the plant, and of its material, with or
        without the panes, and none is the plant."""
        await self.show([])
        view = await self.centre()
        w, h = view["res"]
        origin, direction = ray(view, w // 2, h // 2)
        plant, masonry = ahead(view, 0.5 * view["t"], 4.0), ahead(view, 0.2 * view["t"], 2.0)

        def pane(s):                                       # a 6 m slab across the centre ray, facing the camera (east)
            c = origin + direction * s
            return (c[0], c[1] - 3.0, c[2] - 3.0, c[0] + 0.1, c[1] + 3.0, c[2] + 3.0)
        panes = (pane(0.35 * view["t"]), pane(0.3 * view["t"]))
        for glazed in (False, True):
            for front in (False, True):
                # the masonry comes first in the payload, so a pass that let later fragments win would show the plant
                parts = ([(masonry, WALL_STONE)] if front else []) + [(plant, FOLIAGE)]
                await self.load(scene(*parts, panes=panes if glazed else ()))
                found = await self.surfaces(masonry if front else plant, plant)
                attr, name = (WALL_STONE, "masonry") if front else (FOLIAGE, "plant")
                with self.subTest(masonry=front, panes=glazed):
                    self.assertGreater(len(found), JITTER_CYCLE * PATCH_PX ** 2 // 2)
                    wrong = [(mat, aux, round(t, 3), round(want, 3)) for mat, aux, t, want, _ in found
                             if (mat, aux) != tuple(attr) or abs(t - want) > HIT_M]
                    self.assertEqual(wrong, [], f"{len(wrong)} of {len(found)} samples are not the {name}")
                    if front:
                        self.assertTrue(all(plant_t is None or t < plant_t - 1.0 for _, _, t, _, plant_t in found))

    async def pause_on_the_way(self):
        """(the converged screen at the pause, the settled screen after moving on, the same view from a fresh history):
        PAUSE_AT_M of test_moving_past_glass_leaves_no_ghost's slide north, a pause until the still view has
        converged, then the rest of the slide and MOVE_ON_M more, a metre a frame."""
        start = (await self.centre())["pos"]
        slide = """async ([p, first, last]) => {
            for (let k = first; k <= last; k++) {
              const f = __hog.frames;
              __hog.camera([p[0], p[1], p[2] + k], Math.PI / 2, -0.25);
              while (__hog.frames === f) await new Promise(requestAnimationFrame);
            } }"""
        await self.page.evaluate(slide, [start, 1, PAUSE_AT_M])
        paused = await self.settled_screen()
        self.assertGreaterEqual(await self.page.evaluate("__hog.history()"), IDLE_N, "the paused view did not converge")
        await self.page.evaluate(slide, [start, PAUSE_AT_M + 1, SLIDE_M + MOVE_ON_M])
        moved = await self.settled_screen()
        await self.page.evaluate("__hog.meshes(false); __hog.meshes(true)")
        return paused, moved, await self.settled_screen()

    async def test_a_pause_past_glass_leaves_no_ghost_once_the_camera_moves_on(self):
        """GH4 Glass/history: pause and resume leave no persistent glass ghost. Sliding past three panes as
        test_moving_past_glass_leaves_no_ghost does, the camera pauses halfway, the panes still over the probed fifth,
        until its still view has converged, then moves on. Once settled, the view differs from one rendered from a
        fresh history by no more than the same pause and resumption leave without the panes, within that test's
        tolerance of 5% of the panes' tint."""
        await self.show([])
        view = await self.centre()
        w, h = view["res"]
        origin, direction = ray(view, w // 2, h // 2)
        panes = []
        for s in (28.0, 30.0, 32.0):                                    # three 10 m slabs facing the camera (east)
            c = origin + direction * s
            panes.append((c[0], c[1] - 5.0, c[2] - 5.0, c[0] + 0.1, c[1] + 5.0, c[2] + 5.0))
        bare = await self.settled_screen()
        await self.show([], panes=panes)
        glazed = await self.settled_screen()
        tint = np.abs(glazed - bare).mean()
        self.assertGreater(tint, TINT_MIN)                              # the panes visibly tint the probed fifth
        paused, moved, fresh = await self.pause_on_the_way()
        self.assertGreater(np.abs(moved - glazed).mean(), TINT_MIN)     # moving on carries their image clear of it
        await self.show([])
        await self.settled_screen()
        paused_bare, moved_bare, fresh_bare = await self.pause_on_the_way()
        self.assertGreater(np.abs(paused - paused_bare).mean(), PAUSED_SHARE * tint)   # the pause held the panes in view
        excess = np.abs(moved - fresh).mean() - np.abs(moved_bare - fresh_bare).mean()
        self.assertLess(excess, GHOST_SHARE * tint)


class HarbourMouthViewTests(RenderTestCase):
    async def recess(self, eye, target):
        """(how much farther than the cliff face beside it each ray through the opening ends, the recess's depth behind
        that face, the face's samples) from `eye`, looking at `target`, over a jitter cycle. A ray goes through the
        opening when it crosses its plane (mesh_grounds.PORTAL_M in) inside the arch, and the face beside it is the plane
        through the hits of the rays crossing that plane BESIDE_M either side of the arch, at the arch's heights."""
        m, u, side, lake, back = harbour()
        half_w, crown, arch = mesh_grounds.CHANNEL_HALF_M, mesh_grounds.CROWN_M, mesh_grounds.vault
        d = target - eye
        frames = await self.page.evaluate("([p, y, q]) => { __hog.camera(p, y, q); return __hog.frames; }",
                                          [eye.tolist(), math.atan2(d[0], d[2]), math.atan2(d[1], math.hypot(d[0], d[2]))])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 3}")
        view = await self.centre()
        w, h = view["res"]
        per_m = h / (2.0 * float(np.linalg.norm(d)) * view["tanH"])
        half = (math.ceil(1.5 * (half_w + BESIDE_M) * per_m) + 2, math.ceil(1.5 * crown * per_m) + 2)
        x0, y0 = w // 2 - half[0], h // 2 - half[1]
        through, beside = [], []
        for frame in await self.each_frame(RECT, [x0, y0, 2 * half[0], 2 * half[1]]):
            view = frame["view"]
            rows = np.array(frame["rect"], float).reshape(-1, 6)
            ys, xs = np.mgrid[y0:y0 + 2 * half[1], x0:x0 + 2 * half[0]]
            dirs = rays(view, np.stack([xs.ravel(), ys.ravel()], 1).astype(float))
            o = np.array(view["pos"], float)
            rel, flat = o[[0, 2]] - m, dirs[:, [0, 2]]
            s = (mesh_grounds.PORTAL_M - rel @ u) / (flat @ u)            # along each ray to the opening's plane
            across, up = rel @ side + s * (flat @ side), o[1] + s * dirs[:, 1] - lake
            inner = (s > 0) & (np.abs(across) < half_w - EDGE_IN_M) & (up > EDGE_IN_M) & (up < arch(across) - EDGE_IN_M)
            face = ((s > 0) & (np.abs(across) > half_w + EDGE_IN_M) & (np.abs(across) < half_w + BESIDE_M)
                    & (up > EDGE_IN_M) & (up < crown))
            through += [(o, dirs[k], rows[k, 5]) for k in np.flatnonzero(inner)]
            beside += [o + dirs[k] * rows[k, 5] for k in np.flatnonzero(face)]
        if len(beside) < 3:
            return np.array([]), 0.0, beside
        points = np.array(beside)
        centre = points.mean(axis=0)
        normal = np.linalg.svd(points - centre)[2][-1]
        excess = np.array([t - normal @ (centre - o) / (normal @ dr) for o, dr, t in through])
        return excess, back - float(np.median((points[:, [0, 2]] - m) @ u)), beside

    async def test_the_harbour_mouth_reads_as_recessed_from_oblique_views(self):
        """HB1 Harbour: the harbour mouth is visibly recessed from oblique views. From OBLIQUE_DEG either side of the
        mouth's facing, HARBOUR_VIEW_M out and HARBOUR_EYE_M over the lake's level, with nothing standing in front of
        the opening, the median hit through the opening lies farther than the cliff face beside it by most
        (RECESS_SHARE) of the recess's depth behind that face."""
        m, u, side, lake, _ = harbour()
        foot = m + mesh_grounds.PORTAL_M * u
        target = np.array([foot[0], lake + mesh_grounds.CROWN_M / 2, foot[1]])
        facing = math.atan2(-u[0], -u[1])
        for turn in (-OBLIQUE_DEG, OBLIQUE_DEG):
            bearing = facing + math.radians(turn)
            eye = np.array([target[0] + HARBOUR_VIEW_M * math.sin(bearing), lake + HARBOUR_EYE_M,
                            target[2] + HARBOUR_VIEW_M * math.cos(bearing)])
            excess, depth, beside = await self.recess(eye, target)
            with self.subTest(off_facing_deg=turn):
                self.assertGreater(len(beside), BESIDE_MIN, "the cliff face beside the opening is out of view")
                self.assertGreater(len(excess), OPENING_MIN, "the opening is out of view")
                self.assertLess(float(np.mean(excess < -FRONT_M)), BLOCKED_SHARE, "something stands in front of the opening")
                self.assertGreater(float(np.median(excess)), RECESS_SHARE * depth,
                                   f"through the opening, hits lie {np.percentile(excess, [25, 50, 75]).round(2)} m "
                                   f"(quartiles) behind the face beside it, in front of a recess {depth:.1f} m deep")


class ProgramTests(RenderTestCase):
    INIT_SCRIPTS = (RECORD_PROGRAMS,)

    async def frames(self, n=VIEW_FRAMES):
        frames = await self.page.evaluate("__hog.frames")
        await self.page.wait_for_function(f"__hog.frames >= {frames + n}", timeout=FRAME_TIMEOUT_S * 1000)

    async def above(self, x, z, up):
        return [x, await self.ground(x, z) + up, z]

    async def look(self, eye, target):
        """The camera at `eye` looking at `target` for VIEW_FRAMES frames; the page's mesh counters then."""
        d = np.subtract(target, eye)
        await self.page.evaluate("([p, y, q]) => __hog.camera(p, y, q)",
                                 [list(eye), math.atan2(d[0], d[2]), math.atan2(d[1], math.hypot(d[0], d[2]))])
        await self.frames()
        return await self.page.evaluate("__hog.meshStats()")

    async def test_every_program_links_and_draws_every_path_without_gl_or_page_errors(self):
        """PO1/PO2 Platform/output: the required programs link and draw without GL or page errors on the actual adapter.
        Frames that take every path (the meshes and their shadow layers, the near layer included; glass; the lake's
        reflections; the cone and far-tile passes, over distant terrain too; each quality; dusk) leave no GL error flag
        raised since the page started, no page error and no __hog.error, and every program the page linked has linked
        and drawn."""
        near = (await self.page.evaluate("__hog.meshStats()"))["near"]
        opening = await self.page.evaluate(f"__hog.path({self.OPENING_T})")
        mirrored = "the lake toward the castle"
        views = {"the tour's opening": (opening["pos"], np.add(opening["pos"], opening["fwd"])),
                 mirrored: (await self.above(*await self.page.evaluate("DATA.meta.extras.lake_c"), LAKE_EYE_M),
                            await self.above(0.0, 0.0, CASTLE_LOOK_M)),
                 "the greenhouses' glass": [await self.above(*p) for p in GREENHOUSES]}
        far = await self.above(*FAR_VIEW["at"], FAR_VIEW["height"])
        for hour in (self.TOD, await self.light_at(lambda L: L["dusk"] > 0.99)):
            await self.page.evaluate("(h) => __hog.tod(h)", hour)
            for name, (eye, target) in views.items():
                stats = await self.look(eye, target)
                with self.subTest(hour=hour, view=name):
                    self.assertGreater(stats["drawn"], 0)
                    if name == mirrored:
                        self.assertGreater(stats["reflected"], 0, "the mirror drew nothing")
            if hour == self.TOD:
                for quality in (0, 2, 1):
                    await self.page.evaluate("(q) => __hog.q(q)", quality)
                    await self.frames()
                await self.page.evaluate("([p, y, q]) => __hog.camera(p, y, q)", [far, FAR_VIEW["yaw"], FAR_VIEW["pitch"]])
                await self.frames()
                stats = await self.page.evaluate("__hog.meshStats()")
                self.assertGreater(stats["casters"], 0, "no shadow caster in the static layers")
                self.assertGreater(stats["near"], near, "the near layer was never drawn")
        state = await self.page.evaluate(GL_STATE)
        self.assertEqual(state["errors"], [], f"GL errors on {self.adapter}")
        self.assertFalse(state["lost"])
        self.assertIsNone(state["error"])
        self.assertEqual(self.errors, [])
        self.assertGreater(state["linked"], 0)
        self.assertEqual(state["unlinked"], [])
        self.assertEqual(state["undrawn"], [], "programs that never drew")


class TourStateTests(RenderTestCase):
    QUERY = f"?t=60&pause=1&scale={TOUR_SCALE:g}"      # the page's overlay shown, as a visitor sees its controls
    INIT_SCRIPTS = (RECORD_PROGRAMS,)

    async def tour(self):
        return await self.page.evaluate(TOUR_TIME, TOUR_STEP_S)

    async def test_resizes_a_new_sun_and_quality_and_a_converged_pause_keep_the_tour_and_its_controls(self):
        """SC7 State changes: resize, sun and quality changes and the paused convergence preserve the tour and the
        controls. During playback the viewport is resized, the time of day set on its slider, the quality cycled with
        Q and the viewport resized again; the tour is then paused until its view has converged, and resumed. The tour
        time keeps advancing from where it was, no faster than the clock and not while paused, with the camera on the
        tour; and the controls still show the chosen hour, quality and playback, which the frames apply: the rendered
        sun is the chosen hour's, every program that reads the quality draws at the chosen one, and the canvas and its
        internal image have the last viewport's size."""
        page = self.page
        status, play = page.locator("#status"), page.locator("#play")
        slider = page.get_by_role("slider", name="Time of day")
        canvas = page.locator("#c")
        await canvas.focus()
        await page.keyboard.press("f")                # setUp left the camera in __hog.camera's free flight
        await expect(status).to_contain_text(f"flyover {self.OPENING_T} /")
        await play.click()
        await expect(play).to_have_text("Pause tour")
        await page.wait_for_function(f"{SHOWN_SECOND} >= {self.OPENING_T + LEAD_S}", timeout=FRAME_TIMEOUT_S * 1000)
        before = await self.tour()
        await page.set_viewport_size(RESIZED[0])
        await slider.fill(f"{CHOSEN_HOUR:g}")
        await canvas.focus()
        await page.keyboard.press("q")
        await page.set_viewport_size(RESIZED[1])
        await page.wait_for_function(f"__hog.frames >= {await page.evaluate('__hog.frames') + VIEW_FRAMES}")
        after = await self.tour()
        await play.click()
        await expect(play).to_have_text("Resume tour")
        paused = await self.tour()
        await expect(status).to_contain_text("paused · still", timeout=FRAME_TIMEOUT_S * 1000)
        await page.wait_for_function(f"performance.now() >= {paused['ms'] + PAUSE_S * 1000}")
        held = await self.tour()
        await play.click()
        resumed_ms = await page.evaluate("() => { __programs.recent.clear(); return performance.now(); }")
        await page.wait_for_function(f"__hog.frames >= {await page.evaluate('__hog.frames') + VIEW_FRAMES}")
        resumed = await self.tour()
        with self.subTest(tour="through the changes"):
            self.assertGreater(after["t"], before["t"])
            self.assertLessEqual(after["t"] - before["t"], (after["ms"] - before["ms"]) / 1000 + CLOCK_SLACK_S)
        with self.subTest(tour="paused"):
            self.assertAlmostEqual(held["t"], paused["t"], delta=TOUR_STEP_S)
        with self.subTest(tour="resumed"):
            self.assertGreater(resumed["t"], paused["t"])
            self.assertLessEqual(resumed["t"] - paused["t"], (resumed["ms"] - resumed_ms) / 1000 + CLOCK_SLACK_S)
        with self.subTest(tour="the camera on it"):
            self.assertLess(max(r["off"] for r in (before, after, paused, held, resumed)), TOUR_OFF_M)
        with self.subTest(controls="shown"):
            await expect(slider).to_have_value(f"{CHOSEN_HOUR:g}")
            await expect(page.locator("#todTxt")).to_have_text(CHOSEN_TEXT)
            await page.get_by_text("Stats", exact=True).click()
            await expect(page.locator("#frame-stats")).to_contain_text(f"quality {CHOSEN_QUALITY[0]}")
            await expect(play).to_have_text("Pause tour")
            await expect(status).not_to_contain_text("paused")
        applied = await page.evaluate(APPLIED, CHOSEN_HOUR)
        with self.subTest(controls="applied"):
            np.testing.assert_allclose(applied["sun"], applied["want"], atol=1e-9)
            self.assertTrue(applied["quality"])
            self.assertEqual(set(applied["quality"]), {CHOSEN_QUALITY[1]})
            size = RESIZED[1]
            self.assertEqual(applied["canvas"], [size["width"], size["height"]])
            self.assertEqual(applied["res"], [round(size["width"] * TOUR_SCALE), round(size["height"] * TOUR_SCALE)])


if __name__ == "__main__":
    unittest.main()
