"""Production-path checks that meshes and the raymarched scene resolve to the correct front surface: occlusion against
terrain, sky and far tiles, every jitter phase, sun shadows, payload loading, exposure, lake reflections, glass, resizing,
lamplight, and the programs the page compiles."""
import base64
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest
import zlib

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import box as rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import build_meshes  # noqa: E402
import shaders  # noqa: E402
from mesh_geometry import Mesh  # noqa: E402
from mesh_source import load_data  # noqa: E402
from pagetest import smoke  # noqa: E402
from render_testing import (EDGE_MARGIN_M, FRAME_TIMEOUT_S, JITTER_CYCLE, LAWN, MAT_GROUND, MAT_SKY, MAT_WATER,  # noqa: E402
                            NEXT_FRAME, REARM, SCENE, SUNLIT, WALL, WEST, RenderTestCase, ahead, box_hit, off_screen,
                            payload, ray)

DBG_NO_CLOUD_SHADOWS = 1024                 # the ground lit as under a clear sky
METERING_SPREAD = 1.03                      # metered exposure keeps level sunlit ground this steady through the day

# records each shader source the page hands to WebGL, before the page's own script runs
RECORD_SHADERS = """(() => {
  const sources = window.__shaderSources = [], native = WebGL2RenderingContext.prototype.shaderSource;
  WebGL2RenderingContext.prototype.shaderSource = function (shader, source) { sources.push(source); return native.call(this, shader, source); };
})();"""


def first_difference(got, expected):
    a, b = got.splitlines(), expected.splitlines()
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"line {i + 1}: page {x!r}, shaders.py {y!r}"
    return f"the page has {len(a)} lines, shaders.py {len(b)}"


def slab(centre, half, faces):
    """A horizontal slab of no thickness round `centre` ((x, y, z)), 2 x `half` metres square, as the payload's body: its
    faces ("up", "down") in that order in one chunk, as a coarser level leaves a thin slab it flattened."""
    x, y, z = centre
    quad = np.array([[x - half, y, z - half], [x + half, y, z - half], [x + half, y, z + half], [x - half, y, z + half]])
    pieces = [Mesh(quad, np.tile([0.0, 1.0 if face == "up" else -1.0, 0.0], (4, 1)), np.tile(np.array(WALL, np.uint8), (4, 1)),
                   np.array([[0, 2, 1], [0, 3, 2]] if face == "up" else [[0, 1, 2], [0, 2, 3]], np.uint32)) for face in faces]
    return json.loads(build_meshes.strip_js(build_meshes.encode_payload({"diagnostic": pieces}, SCENE)))


class MeshOcclusionTests(RenderTestCase):
    INIT_SCRIPTS = (RECORD_SHADERS,)

    @smoke
    async def test_mesh_in_front_of_terrain_wins_with_the_object_record(self):
        await self.show([])
        view = await self.centre()
        self.assertEqual(view["mat"], MAT_GROUND)
        cube = ahead(view, 0.5 * view["t"], 4.0)
        await self.show([cube])
        hit = await self.centre()
        self.assertEqual((hit["mat"], hit["aux"]), WALL)
        self.assertAlmostEqual(hit["t"], box_hit(*ray(hit, *[r // 2 for r in hit["res"]]), cube), delta=0.02)
        self.assertLess(hit["t"], view["t"])
        self.assertGreater(-np.dot(hit["n"], ray(hit, *[r // 2 for r in hit["res"]])[1]), 0.5)

    @smoke
    async def test_terrain_in_front_of_mesh_wins(self):
        await self.show([])
        view = await self.centre()
        buried = ahead(view, view["t"] + 6.0, 4.0)       # the ray meets the ground before reaching the cube
        await self.show([buried])
        hit = await self.centre()
        self.assertEqual(hit["mat"], MAT_GROUND)
        # jitter moves a grazing ground hit by decimetres between frames, so compare with this frame's ray
        self.assertLess(hit["t"], box_hit(*ray(hit, *[r // 2 for r in hit["res"]]), buried) - 1.0)

    async def test_raymarch_ends_at_the_nearest_mesh_but_finds_terrain_in_front(self):
        # debug bit 256 keeps the mesh depth pre-pass and the march's limit but skips the meshes' G-buffer draw, so the
        # probe shows the record P1 itself left: the sky where it gave way to a nearer mesh
        await self.show([])
        view = await self.centre()
        self.assertEqual(view["mat"], MAT_GROUND)
        await self.page.evaluate("__hog.dbg(256)")
        await self.show([ahead(view, 0.5 * view["t"], 4.0)])
        self.assertEqual((await self.centre())["mat"], MAT_SKY)
        await self.show([ahead(view, view["t"] + 6.0, 4.0)])          # behind the ground the ray meets first
        self.assertEqual((await self.centre())["mat"], MAT_GROUND)

    async def test_mesh_against_sky_and_far_tiles(self):
        await self.show([], yaw=WEST, pitch=0.3)
        sky = await self.centre()
        self.assertEqual(sky["mat"], MAT_SKY)
        await self.show([ahead(sky, 120.0, 6.0)], yaw=WEST, pitch=0.3)
        self.assertEqual((await self.centre())["mat"], WALL[0])
        # distant terrain puts its 8x8 blocks on the far-tile stencil path; a nearer mesh must still win there
        await self.show([], yaw=WEST, pitch=-0.05, height=400.0)
        far = await self.centre()
        self.assertGreater(far["t"], 3000.0)
        await self.show([ahead(far, 0.7 * far["t"], 0.012 * far["t"])], yaw=WEST, pitch=-0.05, height=400.0)
        hit = await self.centre()
        self.assertEqual(hit["mat"], WALL[0])
        self.assertLess(hit["t"], far["t"])

    async def test_off_axis_silhouette_follows_every_jitter_phase(self):
        await self.show([])
        view = await self.centre()
        w, h = view["res"]
        o, d = ray(view, int(w * 0.8), h // 2)
        target = o + d * 40.0
        cube = (*(target - 3.0), *(target + 3.0))
        await self.show([cube])
        row = [x for x in range(int(w * 0.6), w)]
        frames = await self.each_frame("([xs, y]) => xs.map((x) => __hog.probe(x, y))", [row, h // 2])
        phases, checked = set(), 0
        for frame in frames:
            phases.add(tuple(frame[0]["jit"]))
            for x, probe in zip(row, frame):
                origin, direction = ray(probe, x, h // 2)
                inside, outside = box_hit(origin, direction, cube, -EDGE_MARGIN_M), box_hit(origin, direction, cube, EDGE_MARGIN_M)
                if (inside is None) != (outside is None):
                    continue
                with self.subTest(x=x, jitter=probe["jit"]):
                    self.assertEqual(probe["mat"] == WALL[0], inside is not None)
                    checked += 1
        self.assertGreaterEqual(len(phases), 4)
        self.assertGreater(checked, JITTER_CYCLE * len(row) // 2)

    @smoke
    async def test_every_program_fits_the_sampler_limit(self):
        audit = await self.page.evaluate("__hog.samplers()")
        for name, count in audit["counts"].items():
            with self.subTest(program=name):
                self.assertLessEqual(count, audit["limit"])

    @smoke
    async def test_the_page_compiles_the_cpu_checked_programs(self):
        # test_shaders validates shaders.py's sources with glslang: they must be exactly what the page compiles
        defs = await self.page.evaluate("__hog.defs")
        expected = [s for p in shaders.programs(defs) for s in (p.vertex, p.fragment)]
        recorded = await self.page.evaluate("window.__shaderSources")
        self.assertEqual(len(recorded), len(expected))
        for i, (got, want) in enumerate(zip(recorded, expected)):
            if got != want:
                self.fail(f"shader source {i}: {first_difference(got, want)}")
        cpu = shaders.cpu_defs(load_data(), float_linear="MANUAL_BILERP" not in defs)
        sampled = set(shaders.GROUND_KEYS)
        self.assertEqual({k: v for k, v in cpu.items() if k not in sampled}, {k: v for k, v in defs.items() if k not in sampled})
        self.assertEqual(list(cpu), list(defs))

    @smoke
    async def test_off_screen_caster_shades_visible_ground_until_removed(self):
        await self.show([])
        sun = np.array((await self.centre())["sun"])
        self.assertGreater(sun[1], 0.15)
        away = math.atan2(-sun[0], -sun[2])              # face away from the sun so its casters sit behind the camera
        view = {"at": SUNLIT, "yaw": away, "pitch": -0.35, "height": 40.0}
        await self.show([], **view)
        w, h = (await self.centre())["res"]
        # shadow-pass block centres (4 bx + 2, 4 by + 2) in the lower half that see lit open ground in one frame
        open_ground = await self.page.evaluate("""([w, h]) => {
            const out = [];
            for (let by = 1; 4 * by + 2 < h / 2; by += 2) for (let bx = 2; 4 * bx + 6 < w; bx += 3) {
              const x = 4 * bx + 2, y = 4 * by + 2;
              if (__hog.probe(x, y).mat === 1 && __hog.probeSun(x, y) > 0.8) out.push([x, y]);
            }
            return out; }""", [w, h])
        self.assertTrue(open_ground)
        x, y = open_ground[len(open_ground) // 2]
        lit = await self.page.evaluate("([x, y]) => [__hog.probe(x, y), __hog.probeSun(x, y)]", [x, y])
        self.assertEqual(lit[0]["mat"], MAT_GROUND)
        self.assertGreater(lit[1], 0.7)
        origin, direction = ray(lit[0], x, y)
        centre = origin + direction * lit[0]["t"] + sun * 120.0
        caster = (*(centre - 6.0), *(centre + 6.0))
        self.assertTrue(off_screen(lit[0], caster))
        await self.show([caster], **view)
        self.assertLess(await self.page.evaluate("([x, y]) => __hog.probeSun(x, y)", [x, y]), 0.3)
        await self.show([], **view)                        # an empty payload must not keep the old caster
        self.assertGreater(await self.page.evaluate("([x, y]) => __hog.probeSun(x, y)", [x, y]), 0.7)

    async def test_grazing_sun_does_not_shadow_a_mesh_face_with_itself(self):
        # 400 m up no terrain or tree shades the wall, so only the mesh shadow layer can darken it
        samples = await self.lit_wall(angle=80.0, lift=400.0, post=500.0)
        middle = [s for _, _, along, up, s in samples if abs(along) < 25.0 and abs(up) < 6.0]
        self.assertGreater(len(middle), 30)
        self.assertGreater(min(middle), 0.9, f"{sum(s < 0.9 for s in middle)} of {len(middle)} samples self-shadowed")

    async def test_sun_leaving_the_local_grid_still_reaches_a_mesh(self):
        # 40-70 m up, the low sun's rays leave the local grid westwards: its edge may bound the march step, not dim the sun
        samples = await self.lit_wall(angle=0.0, lift=55.0)
        self.assertGreater(len(samples), 100)
        self.assertGreater(min(s for *_, s in samples), 0.9)

    async def test_the_finest_shadow_layer_shades_a_point(self):
        await self.show([])
        sun = np.array((await self.centre())["sun"])
        level = math.hypot(sun[0], sun[2])
        away, side = -np.array([sun[0], 0.0, sun[2]]) / level, np.array([-sun[2], 0.0, sun[0]]) / level
        x, z = SUNLIT
        foot = np.array([x, (await self.page.evaluate("([x, z]) => __hog.ground(x, z)", [x, z]))[1], z])
        post = (x - 1.5, foot[1] - 1.0, z - 1.5, x + 1.5, foot[1] + 20.0, z + 1.5)
        length = 20.0 * level / sun[1]                     # the post's shadow on the ground
        mid = foot + away * 0.5 * length
        # a 160 m region round the shadow listed before a 6 km one: the 3 m shadow must come from the finer texels
        regions = [(*(mid - 80.0), *(mid + 80.0)), (*(mid - 3000.0), *(mid + 3000.0))]
        eye = mid + side * 40.0
        eye_ground = (await self.page.evaluate("([x, z]) => __hog.ground(x, z)", [eye[0], eye[2]]))[1]
        view = {"at": (eye[0], eye[2]), "yaw": math.atan2(-side[0], -side[2]), "pitch": -0.9,
                "height": foot[1] + 50.0 - eye_ground}
        await self.show([], **view)
        frame = await self.page.evaluate("""() => {
            const view = __hog.probe(0, 0), [w, h] = view.res, out = [];
            for (let y = 2; y < h; y += 4) for (let x = 2; x < w; x += 4) {
              const p = __hog.probe(x, y);
              if (p.mat === 1) out.push([x, y, p.t, __hog.probeSun(x, y)]);
            }
            return { view, out }; }""")
        core = []                                          # sunlit ground deep inside where the post's shadow will fall
        for px, py, t, s in frame["out"]:
            origin, direction = ray(frame["view"], px, py)
            if s > 0.7 and box_hit(origin + direction * t + sun * 0.05, sun, post, -0.75) is not None:
                core.append([px, py])
        self.assertGreater(len(core), 5)
        await self.show([post], shadow=regions, **view)
        self.assertLess(max(await self.page.evaluate("(xy) => xy.map(([x, y]) => __hog.probeSun(x, y))", core)), 0.3)

    @smoke
    async def test_rejected_payloads_leave_the_loaded_meshes(self):
        far = await self.ground(-900.0, -900.0)
        good = payload(ahead(await self.centre(), 30.0, 4.0), (-900.0, far, -900.0, -896.0, far + 4.0, -896.0))
        self.assertEqual(len(good["chunks"]), 2)
        await self.page.evaluate("(b) => __hog.loadMeshes(b)", good)
        disc = lambda cx: Polygon([(cx + 5.0 * math.cos(a), 5.0 * math.sin(a))
                                   for a in np.linspace(0.0, 2.0 * math.pi, 8000, endpoint=False)])
        ground = await self.ground(-295.0, 0.0)
        walls = [build_meshes.prism(disc(cx), ground, ground + 1.0, WALL) for cx in (-300.0, -290.0)]
        # two 40k-vertex chunks, one tile, side by side in the streams: one unit's full level, without coarser levels
        big = json.loads(build_meshes.strip_js(build_meshes.merge(
            build_meshes.encode_assets({"diagnostic": walls}, {"diagnostic": [{0: walls}]}))))
        self.assertEqual([(c["group"], c["level"]) for c in big["chunks"]], [(0, 0), (0, 0)])
        def broken(body, change):
            body = json.loads(json.dumps(body))
            change(body["chunks"])
            return body
        def recoded(body, key, change):                    # one stream's compressed bytes changed, still valid zlib
            body = json.loads(json.dumps(body))
            inner = change(zlib.decompress(base64.b64decode(body["streams"][key])))
            body["streams"][key] = base64.b64encode(zlib.compress(inner)).decode("ascii")
            return body
        def inverted(chunks):
            chunks[0]["box"] = chunks[0]["box"][3:] + chunks[0]["box"][:3]
        def overrun(chunks):                               # the same vertex total, but chunk 0 now indexes past its own
            chunks[0]["verts"] -= 1
            chunks[1]["verts"] += 1
        def unbalanced(chunks):
            chunks[0]["verts"] += 1
        def merged(chunks):                                # consistent streams and indices, past 16-bit indexing
            chunks[0]["verts"] += chunks[1]["verts"]
            chunks[0]["tris"] += chunks[1]["tris"]
            del chunks[1]
        def orphaned(chunks):                              # its group's only chunk, now a coarser level of it
            chunks[0]["level"] = 1
        def strayed(chunks):
            chunks[0]["group"] = len(good["groups"])
        def overcounted(chunks):                           # narrow runs past the chunk's own triangles
            chunks[0]["narrow"] = [[chunks[0]["tris"] + 1, 50]]
        def unsorted(chunks):                              # narrow runs not narrowest first
            chunks[0]["narrow"] = [[1, 50], [1, 40]]
        old = {**good, "version": 1, "streams": {k: base64.b64encode(zlib.compress(v, 9)).decode("ascii")
                                                 for k, v in build_meshes.decode_streams(good).items()}}
        for name, body in (("inverted box", broken(good, inverted)), ("index past its vertices", broken(good, overrun)),
                           ("streams shorter than chunks", broken(good, unbalanced)),
                           ("more vertices than 16-bit indices reach", broken(big, merged)),
                           ("a group without its full level", broken(good, orphaned)),
                           ("a chunk outside every group", broken(good, strayed)),
                           ("narrow runs past its triangles", broken(good, overcounted)),
                           ("narrow runs out of order", broken(good, unsorted)),
                           ("switch distances for no view", {k: v for k, v in good.items() if k != "levels"}),
                           ("the version-1 format", old), ("the version-2 format", {**good, "version": 2}),
                           ("truncated position codec", recoded(good, "pos", lambda s: s[:-4])),
                           ("truncated index codec", recoded(good, "idx", lambda s: s[:-2]))):
            with self.subTest(case=name):
                outcome = await self.page.evaluate(
                    "async (b) => { try { await __hog.loadMeshes(b); return 'loaded'; } catch (e) { return String(e.message || e); } }",
                    body)
                self.assertNotEqual(outcome, "loaded")
                self.assertEqual((await self.page.evaluate("__hog.meshStats()"))["chunks"], 2)
        self.assertIsNone(await self.page.evaluate("__hog.error || null"))

    async def test_a_flattened_slab_shades_as_the_face_the_camera_sees(self):
        # a coarser level can flatten a slab thinner than its budget, such as a wall's coping, into twins: the same
        # triangles facing up, then down. The G-buffer pass takes the later of equal depths, so seen from above the
        # slab would shade as its underside
        x, z = LAWN
        top = await self.ground(x, z) + 30.0                         # over the tallest tree (25.5 m)
        frames = await self.page.evaluate(
            "async ([b, pos]) => { __hog.camera(pos, 0, -1.5); await __hog.loadMeshes(b); return __hog.frames; }",
            [slab((x, top, z), 4.0, ("up", "down")), [x, top + 15.0, z]])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 3}")
        found = await self.each_frame("""() => { const r = __hog.probe(0, 0).res, p = __hog.probe(r[0] >> 1, r[1] >> 1);
            return [p.mat, p.n[1]]; }""")
        self.assertEqual({mat for mat, _ in found}, {WALL[0]})
        self.assertGreater(min(up for _, up in found), 0.9)

    async def test_the_lake_mirrors_a_flattened_slab_as_its_underside(self):
        # the mirror pass keeps the first of equal depths, so there the upper twin would win: the lake must show the
        # slab's underside, lit as the underside alone is
        view = await self.lake_view(-0.6)
        await self.show([], **view)
        water = await self.centre()
        w, h = water["res"]
        origin, direction = ray(water, w // 2, h // 2)
        centre = origin + direction * water["t"] + direction * np.array([1.0, -1.0, 1.0]) * 100.0
        x, z = view["at"]
        eye = [x, await self.ground(x, z) + view["height"], z]
        colours = {}
        for faces in (("up", "down"), ("down",)):
            frames = await self.page.evaluate(
                "async ([b, pos, yaw, pitch]) => { __hog.camera(pos, yaw, pitch); await __hog.loadMeshes(b); return __hog.frames; }",
                [slab(centre, 15.0, faces), eye, view["yaw"], view["pitch"]])
            await self.page.wait_for_function(f"__hog.frames >= {frames + 3}")
            colours[faces] = await self.reflected()
        self.assertTrue(np.isfinite(colours[("down",)]).all(), "the slab is not in the lake")
        np.testing.assert_allclose(colours[("up", "down")], colours[("down",)], rtol=0.05)

    @smoke
    async def test_the_page_decodes_the_streams_the_encoder_quantized(self):
        # SHA256 of each stream the page decodes: a diagnostic payload against the encoder's raw streams, and the
        # embedded payload against what the build's own decoder makes of it
        digests = """async (b) => {
          const body = b || MESHES, s = await meshStreams(body, body.chunks.reduce((n, c) => n + c.verts, 0),
                                                          body.chunks.reduce((n, c) => n + c.tris, 0)), out = {};
          for (const k of Object.keys(s))
            out[k] = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", s[k])), (x) => x.toString(16).padStart(2, "0")).join("");
          return out; }"""
        sha = lambda streams: {k: hashlib.sha256(v).hexdigest() for k, v in streams.items()}
        ground = await self.ground(-260.0, -60.0)
        boxes = [(-262.0, ground, -62.0, -258.0, ground + 4.0, -58.0), (500.0, ground, 500.0, 507.0, ground + 1.0, 503.0)]
        meshes = {"diagnostic": [build_meshes.prism(rectangle(x0, z0, x1, z1), y0, y1, WALL) for x0, y0, z0, x1, y1, z1 in boxes]}
        encoded = build_meshes.encode_assets(meshes)
        with self.subTest(payload="diagnostic"):
            self.assertEqual(await self.page.evaluate(digests, payload(*boxes)),
                             sha({k: encoded["diagnostic"]["streams"][k] for k in build_meshes.RAW}))
        with self.subTest(payload="embedded"):
            embedded = json.loads(build_meshes.strip_js((ROOT / "build" / "meshes.js").read_text(encoding="utf-8")))
            self.assertEqual(await self.page.evaluate(digests, None), sha(build_meshes.decode_streams(embedded)))

    async def test_metered_exposure_keeps_sunlit_ground_steady_through_the_day(self):
        # under a clear sky: at t=60 a cloud's shadow falls on this spot at 09:30, 13% darker, as the sky toward the sun
        # shows; with the clouds' shadows off the three hours measured within 0.8% of each other
        x, z = -220.0, -110.0       # open, level lawn south-west of the castle, clear of meshes and shadows 09:30-15:30
        height = lambda dx, dz: self.page.evaluate("([x, z]) => __hog.ground(x, z)[1]", [x + dx, z + dz])
        ground = await height(0.0, 0.0)
        rise = math.hypot(await height(5.0, 0.0) - await height(-5.0, 0.0), await height(0.0, 5.0) - await height(0.0, -5.0))
        # on a slope the light the ground takes changes with the sun's bearing too, which metering cannot follow
        self.assertLess(math.degrees(math.atan(rise / 10.0)), 2.0, f"the ground at {(x, z)} is not level")
        looks = {}
        for hour in (9.5, 12.5, 15.5):
            frames = await self.page.evaluate("([p, h, d]) => { __hog.camera(p, 0, -1.5); __hog.tod(h); __hog.dbg(d); return __hog.frames; }",
                                              [[x, ground + 40.0, z], hour, DBG_NO_CLOUD_SHADOWS])
            await self.page.wait_for_function(f"__hog.frames >= {frames + 150}", timeout=60000)
            looks[hour] = await self.page.evaluate("""() => {
                const [w, h] = __hog.probe(0, 0).res, out = [];
                for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) {
                  const x = (w >> 1) + 8 * i, y = (h >> 1) + 8 * j;
                  out.push([__hog.probe(x, y).mat, __hog.probeSun(x, y), __hog.probeScreen(x / w, y / h)]);
                }
                return out; }""")
        for hour, samples in looks.items():
            with self.subTest(hour=hour):                  # open, sunlit ground under the camera at every hour
                self.assertTrue(all(mat == MAT_GROUND and sun > 0.9 for mat, sun, _ in samples))
        brightness = {h: float(np.mean([np.dot(rgb, [0.2126, 0.7152, 0.0722]) for *_, rgb in s])) for h, s in looks.items()}
        self.assertLess(max(brightness.values()) / min(brightness.values()), METERING_SPREAD, brightness)

    @smoke
    async def test_off_screen_mesh_appears_in_the_lake(self):
        view = await self.lake_view(-0.6)
        await self.show([], **view)
        water = await self.centre()
        self.assertEqual(water["mat"], MAT_WATER)
        w, h = water["res"]
        origin, direction = ray(water, w // 2, h // 2)
        mirrored = direction * np.array([1.0, -1.0, 1.0])
        centre = origin + direction * water["t"] + mirrored * 100.0
        box = (*(centre - 15.0), *(centre + 15.0))
        self.assertTrue(off_screen(water, box))
        await self.show([box], **view)
        # ripples bend the reflection by several degrees here, as they bend the raymarched one: look around the centre
        reflected = await self.page.evaluate("""() => { const [w, h] = __hog.probe(0, 0).res; let n = 0;
            for (let y = (h >> 1) - 16; y <= (h >> 1) + 16; y += 4) for (let x = (w >> 1) - 16; x <= (w >> 1) + 16; x += 4)
              if (__hog.probe(x, y).mat === 2 && __hog.probeColor(x, y)[0] === 1) n++;
            return n; }""")
        self.assertGreater(reflected, 0)

    async def test_nearer_reflected_terrain_hides_a_mesh(self):
        view = await self.lake_view(-0.12, toward_castle=True)
        await self.show([], **view)
        w, h = (await self.centre())["res"]
        # over one 8-frame jitter cycle, keep water pixels whose reflection meets the ground at a steady distance;
        # a pixel picked from a single frame can skim a far shore that other jitter phases miss
        frames, steady = await self.page.evaluate(f"""async ([w, h, ms]) => {{
            {NEXT_FRAME}
            const seen = new Map();
            let frames = 0;
            {REARM}
            for (; frames < {JITTER_CYCLE} && await nextFrame(ms); frames++) {{
              for (let y = 4; y < h / 2; y += 6) for (let x = 4; x < w - 4; x += 6) {{
                const g = __hog.probe(x, y), c = __hog.probeColor(x, y), s = seen.get(x + "," + y) || {{ x, y, g, t: [] }};
                if (g.mat === 2 && c[2] === 1 && c[1] > 60) s.t.push(c[1]); else s.miss = true;
                seen.set(x + "," + y, s);
              }}
            }}
            return [frames, [...seen.values()].filter((s) => !s.miss && Math.max(...s.t) < 1.1 * Math.min(...s.t))
                                               .map((s) => [s.x, s.y, s.g, Math.min(...s.t)])]; }}""",
                                                  [w, h, FRAME_TIMEOUT_S * 1000])
        self.assertEqual(frames, JITTER_CYCLE)
        self.assertTrue(steady)
        x, y, probe, reflected_t = steady[len(steady) // 2]
        origin, direction = ray(probe, x, y)
        water = origin + direction * probe["t"]
        mirrored = direction * np.array([1.0, -1.0, 1.0])
        for s, expected in ((reflected_t + 100.0, 0.0), (0.4 * reflected_t, 1.0)):
            centre = water + mirrored * s
            await self.show([(*(centre - 10.0), *(centre + 10.0))], **view)
            verdicts = await self.each_frame("([x, y]) => __hog.probeColor(x, y)[0]", [x, y])
            with self.subTest(distance=s):
                self.assertEqual(verdicts, [expected] * JITTER_CYCLE)

    async def test_moving_past_glass_leaves_no_ghost(self):
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
        self.assertGreater(tint, 10.0)                                  # the panes visibly tint the probed fifth
        moved, fresh = await self.slide_and_settle()                   # the slide carries their image clear of it
        self.assertGreater(np.abs(moved - glazed).mean(), 10.0)
        # the temporal filter carries a share of motion-time history into its settled still, glass or not: the panes
        # may leave nothing persistent beyond what the same slide leaves without them
        await self.show([])
        await self.settled_screen()
        moved_bare, fresh_bare = await self.slide_and_settle()
        excess = np.abs(moved - fresh).mean() - np.abs(moved_bare - fresh_bare).mean()
        self.assertLess(excess, 0.05 * tint)

    @smoke
    async def test_glass_tints_what_it_covers_and_leaves_no_trace(self):
        await self.show([])
        view = await self.centre()
        w, h = view["res"]
        origin, direction = ray(view, w // 2, h // 2)
        plant = ahead(view, 0.5 * view["t"], 4.0)
        def pane(s):                                       # a 6 m slab across the centre ray, facing the camera (east)
            c = origin + direction * s
            return (c[0], c[1] - 3.0, c[2] - 3.0, c[0] + 0.1, c[1] + 3.0, c[2] + 3.0)
        near, nearer, masonry = pane(0.35 * view["t"]), pane(0.3 * view["t"]), ahead(view, 0.2 * view["t"], 2.0)
        looks = {}
        for name, boxes, panes in (("bare", [plant], ()), ("one", [plant], (near,)), ("two", [plant], (near, nearer)),
                                   ("masonry", [plant, masonry], ()), ("masonry+glass", [plant, masonry], (near, nearer)),
                                   ("bare again", [plant], ())):
            looks[name] = await self.converged(boxes, panes)
        self.assertEqual(looks["two"][0]["mat"], looks["bare"][0]["mat"])   # glazing never enters the G-buffer
        self.assertAlmostEqual(looks["two"][0]["t"], looks["bare"][0]["t"], delta=0.1)
        brightness = {k: float(np.mean(v[1])) for k, v in looks.items()}
        self.assertLess(brightness["two"], brightness["one"])            # overlapping panes both tint
        self.assertLess(brightness["one"], brightness["bare"])
        self.assertGreater(brightness["two"], 0.5 * brightness["bare"])  # and the plant still shows through
        difference = lambda a, b: max(abs(x - y) for x, y in zip(looks[a][1], looks[b][1]))
        self.assertLessEqual(difference("masonry+glass", "masonry"), 2)  # nearer masonry hides the panes
        self.assertLessEqual(difference("bare again", "bare"), 2)        # removing the panes leaves no ghost

    async def test_lamplit_windows_look_warm_not_white(self):
        lum = lambda c: float(np.asarray(c) @ [0.2126, 0.7152, 0.0722])
        hours = {"civil dusk": await self.light_at(lambda L: L["el"] < -0.5),
                 "night": await self.light_at(lambda L: L["dusk"] > 0.99)}
        looks = {}
        for when, hour in hours.items():
            for attr in ((4, 41), (4, 45), (7, 13), (7, 14)):
                looks[when, attr] = np.median(await self.screen_box(attr, hour), axis=0)
                r, g, b = looks[when, attr]
                with self.subTest(at=when, code=attr):
                    self.assertGreater(r, 120)                     # clearly lit
                    self.assertLess(b, 0.6 * r)                    # warm lamplight, not blown out to white
                    self.assertLess(g, 0.92 * r)
            with self.subTest(at=when):
                self.assertLess(lum(looks[when, (4, 45)]), 0.85 * lum(looks[when, (4, 41)]))   # some rooms dimmer
                self.assertLess(lum(looks[when, (7, 14)]), 0.85 * lum(looks[when, (7, 13)]))   # the station's "dim light"

    async def test_lit_windows_shine_in_the_lake_at_night(self):
        night = await self.light_at(lambda L: L["dusk"] > 0.99)
        frames = await self.page.evaluate("(h) => { __hog.tod(h); return __hog.frames; }", night)
        await self.page.wait_for_function(f"__hog.frames >= {frames + 10}")
        view = await self.lake_view(-0.6)
        await self.show([], **view)
        water = await self.centre()
        w, h = water["res"]
        origin, direction = ray(water, w // 2, h // 2)
        centre = origin + direction * water["t"] + direction * np.array([1.0, -1.0, 1.0]) * 100.0
        box = (*(centre - 15.0), *(centre + 15.0))
        colours = {}
        for attr in ((7, 11), (7, 13), (4, 41)):
            await self.show([box], attr=attr, **view)
            colours[attr] = await self.reflected()
        for attr in ((7, 13), (4, 41)):
            with self.subTest(code=attr):
                self.assertGreater(colours[attr].sum(), 4.0 * colours[(7, 11)].sum())   # lamps, not dark glass
                r, g, b = colours[attr]
                self.assertTrue(r > g > b, colours[attr])

    async def test_resize_rebuilds_the_mesh_targets(self):
        await self.show([])
        cube = ahead(await self.centre(), 30.0, 4.0)
        for size in ({"width": 640, "height": 480}, {"width": 1280, "height": 600}):
            await self.page.set_viewport_size(size)
            await self.show([cube])
            with self.subTest(size=size):
                hit = await self.centre()
                self.assertEqual(hit["mat"], WALL[0])
                self.assertAlmostEqual(hit["res"][0] / hit["res"][1], size["width"] / size["height"], delta=0.02)
                self.assertIsNone(await self.page.evaluate("__hog.error || null"))

    async def lit_wall(self, angle, lift, post=0.0):
        """Sun-term samples (x, y, along, up, sun) on the sunlit face of a 60 x 30 m wall floating `lift` m over SUNLIT,
        turned so the sun strikes it `angle` degrees off its normal (facing the sun's bearing when the sun stands
        higher than that). A small post `post` m across the sun's path widens the shadow layer, and with it the texels,
        to the castle's scale."""
        await self.show([])
        sun = np.array((await self.centre())["sun"])
        level = math.hypot(sun[0], sun[2])
        facing = math.atan2(sun[0], sun[2]) + math.acos(min(math.cos(math.radians(angle)) / level, 1.0))
        normal = np.array([math.sin(facing), 0.0, math.cos(facing)])
        along = np.array([normal[2], 0.0, -normal[0]])
        x, z = SUNLIT
        ground = (await self.page.evaluate("([x, z]) => __hog.ground(x, z)", [x, z]))[1]
        centre = np.array([x, ground + lift, z])
        corners = [centre + along * a + normal * b for a, b in ((-30, -2), (30, -2), (30, 2), (-30, 2))]
        wall = (Polygon([(c[0], c[2]) for c in corners]), centre[1] - 15.0, centre[1] + 15.0)
        boxes = []
        if post:
            p = centre + np.array([sun[2], 0.0, -sun[0]]) / level * post
            boxes.append((p[0] - 1, ground, p[2] - 1, p[0] + 1, ground + 10, p[2] + 1))
        eye = centre + normal * 70.0
        eye_ground = (await self.page.evaluate("([x, z]) => __hog.ground(x, z)", [eye[0], eye[2]]))[1]
        await self.show(boxes, at=(eye[0], eye[2]), yaw=math.atan2(-normal[0], -normal[2]), pitch=0.0,
                        height=centre[1] - eye_ground, walls=[wall])
        frame = await self.page.evaluate("""() => {
            const view = __hog.probe(0, 0), [w, h] = view.res, out = [];
            for (let y = 2; y < h; y += 8) for (let x = 2; x < w; x += 8) {
              const p = __hog.probe(x, y);
              if (p.mat === 6) out.push([x, y, p.t, p.n, __hog.probeSun(x, y)]);
            }
            return { view, out }; }""")
        samples = []
        for x, y, t, n, sun_term in frame["out"]:
            if np.dot(n, normal) > 0.99:
                origin, direction = ray(frame["view"], x, y)
                offset = origin + direction * t - centre
                samples.append((x, y, float(np.dot(offset, along)), float(offset[1]), sun_term))
        return samples

    async def converged(self, boxes, panes=()):
        """The centre G-buffer record and displayed RGB once the temporal filter has settled on a still view."""
        await self.show(boxes, panes=panes)
        await self.page.wait_for_function(f"__hog.frames >= {await self.page.evaluate('__hog.frames') + 120}", timeout=60000)
        return await self.page.evaluate(
            "() => { const r = __hog.probe(0, 0).res; return [__hog.probe(r[0] >> 1, r[1] >> 1), __hog.probeScreen(0.5, 0.5)]; }")

    async def settled_screen(self):
        """Displayed RGB over the central fifth of the canvas once the temporal filter has settled."""
        await self.page.wait_for_function(f"__hog.frames >= {await self.page.evaluate('__hog.frames') + 150}", timeout=60000)
        return np.array(await self.page.evaluate(
            "() => { const out = []; for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++)"
            " out.push(__hog.probeScreen(0.4 + 0.2 * i / 6, 0.4 + 0.2 * j / 6)); return out; }"), float)

    async def slide_and_settle(self):
        """The settled screen after sliding 12 m north over twelve frames, and the same view from a fresh history."""
        await self.page.evaluate("""async () => {
            const v = __hog.probe(0, 0), yaw = Math.PI / 2, pitch = -0.25;
            for (let k = 1; k <= 12; k++) {
              const f = __hog.frames;
              __hog.camera([v.pos[0], v.pos[1], v.pos[2] + k], yaw, pitch);
              while (__hog.frames === f) await new Promise(requestAnimationFrame);
            } }""")
        moved = await self.settled_screen()
        await self.page.evaluate("__hog.meshes(false); __hog.meshes(true)")
        return moved, await self.settled_screen()


if __name__ == "__main__":
    unittest.main()
