"""Site lane checks: pinned geographic sources, the site tools and the terrain built at the chosen site.

    python -X utf8 -m unittest discover -s src -p test_site.py
"""
import base64
import contextlib
import filecmp
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import inputs  # noqa: E402
from tools.site import layout, sources, sun, terrain  # noqa: E402
import common  # noqa: E402

X = common.SITES["x"]["anchor"]
# The location review's results at X, kept as data: real-world measurements, not levels of the scene.
REVIEW_AT_X = json.loads((ROOT / "tools" / "site" / "review_at_x.json").read_text(encoding="utf-8"))["results"]


def overpass_payload(elements, stamp):
    return json.dumps({"version": 0.6, "osm3s": {"timestamp_osm_base": stamp}, "elements": elements}).encode()


def packed_meta():
    """The metadata in the committed build/data.js."""
    text = (ROOT / "build" / "data.js").read_text(encoding="ascii")
    return json.loads(text[text.index("{"): text.rindex("}") + 1])["meta"]


LUMA = np.array([0.3, 0.55, 0.15])     # the luminance weights frag4.glsl's groundAlb gives the ground's colour


def class_colours(text=None):
    """classCol by surface class, in GLSL `text` (src/frag4.glsl by default): {class: (the RGB colours its albedo mixes,
    the frequencies per metre of the noise in it)}."""
    text = (ROOT / "src" / "frag4.glsl").read_text(encoding="utf-8") if text is None else text
    start = text.index("vec3 classCol(")
    body = text[text.index("{", start) + 1: text.index("\n}\n", start)]
    code = " ".join(line.split("//")[0] for line in body.splitlines())
    out, after = {}, 0
    for statement in filter(str.strip, code.split(";")):
        branch = re.fullmatch(r"\s*if \(c < (\d+)\.5\) return (.*)", statement)
        last = re.fullmatch(r"\s*return (.*)", statement)
        cls, expression = (int(branch[1]), branch[2]) if branch else (after, last[1])
        after = cls + 1
        number = r"\s*(-?[\d.]+)\s*"
        out[cls] = ([tuple(float(v) for v in m) for m in re.findall(rf"vec3\({number},{number},{number}\)", expression)],
                    [float(f) for f in re.findall(r"vnoise\(xz \* ([\d.]+)", expression)])
    return out


GLIMPSE = 0.05         # through the trees, at most this share of the points on the castle's upper parts shows
SEATED_EYE_M, STANDING_EYE_M = 2.25, 2.7     # passengers' eyes above the rail top: mesh_transport.coach_body's
                                             # floor stands 1.05 m above it and its windows 1.95-2.8 m


def railway_point(line, s):
    """(x, z, formation level) of a railway.Line at arclength s."""
    return float(np.interp(s, line.s, line.xz[:, 0])), float(np.interp(s, line.s, line.xz[:, 1])), \
        float(np.interp(s, line.s, line.level))


class SightLines:
    """Lines of sight over a build's local layer, past its ground and the renderer's 3D trees (frag1.glsl: treeDens,
    treeH, treeSD): in each 7 m cell a tree stands at the cell's hashed spot where its hash falls below the tree density
    there (on dry ground, off the roads, clear of buildings); near the station it is a pine, a cone on its trunk, and
    elsewhere mostly a broadleaf, an ellipsoid crown on its trunk."""
    CELL_M, STEP_M, SHORT_M = 7.0, 0.5, 25.0     # sightlines stop short of the castle's own mound

    def __init__(self, ms, data, lo, hi):
        tex, self.ms, self.grid = data["tex"], ms, ms.Grid(data["meta"]["local"])
        self.bed = ms.decode(tex["locBed"])
        self.la, self.water = ms.decode(tex["locA"]) / 255.0, ms.decode(tex["locW"])
        poi = {p["name"]: np.asarray(p["p"], float) for p in data["meta"]["extras"]["poi"]}
        self.c0 = np.floor(np.asarray(lo, float) / self.CELL_M) - 2
        c1 = np.floor(np.asarray(hi, float) / self.CELL_M) + 2
        cz, cx = np.mgrid[self.c0[1]:c1[1] + 1, self.c0[0]:c1[0] + 1]
        cells = np.stack([cx.ravel(), cz.ravel()], 1).astype(np.float32)
        h = self.hash42(cells).astype(np.float64)
        ctr = (cells + 0.25 + 0.5 * h[:, 1:3]) * self.CELL_M
        step = self.smooth
        dens = self.density(ctr)
        con_f = np.maximum.reduce([np.full(len(ctr), 0.12),
                                   1.0 - step(120.0, 230.0, np.hypot(*(ctr - poi["Hogsmeade station"]).T)),
                                   0.85 * (1.0 - step(260.0, 480.0, np.hypot(*(ctr - poi["Grawp's clearing"]).T)))])
        con = h[:, 3] < con_f
        hh = h[:, 3] * 13.7 % 1.0
        height = np.where(con, 13.0 + 8.0 * hh, 10.0 + 9.0 * hh) * (0.72 + 0.28 * step(0.1, 0.8, dens))
        self.trees = {"ctr": ctr, "H": np.where(h[:, 0] < dens, height, np.nan), "con": con,
                      "r": np.where(con, 0.23 * height, 3.0 + 1.9 * (hh * 7.3 % 1.0))}
        self.shape = cz.shape

    @staticmethod
    def smooth(a, b, x):
        t = np.clip((x - a) / (b - a), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def density(self, p):
        """frag1.glsl's tree density at (x, z) points `p` ((n, 2)), read bilinearly as the shader does: a cell's tree
        stands at its hashed spot where the cell's hash falls below the density there."""
        at = lambda layer: self.ms.sample(layer, self.grid, p[:, 0], p[:, 1])
        return at(self.la[..., 1]) * (at(self.water) <= -0.9) * (1.0 - at(self.la[..., 0])) * \
            self.smooth(0.5, 2.5, at(self.la[..., 3]) * 25.5 - 12.8)

    @staticmethod
    def hash42(p):
        """frag1.glsl's hash42 in float32."""
        f32, fract = np.float32, lambda v: v - np.floor(v)
        p4 = fract(p[:, [0, 1, 0, 1]] * np.array([0.1031, 0.1030, 0.0973, 0.1099], f32)).astype(f32)
        p4 = (p4 + (p4 * (p4[:, [3, 2, 0, 1]] + f32(33.33))).sum(1, dtype=f32)[:, None]).astype(f32)
        return fract((p4[:, [0, 0, 1, 2]] + p4[:, [1, 2, 2, 3]]) * p4[:, [2, 1, 3, 0]]).astype(f32)

    def seen(self, eye, targets, short_m=None):
        """How many of `targets` ((n, 3): x, height, z) are in sight from `eye` (x, height, z), the lines stopping
        `short_m` short of them (by default SHORT_M, short of the castle's own mound)."""
        eye, targets = np.asarray(eye, float), np.asarray(targets, float)
        short = self.SHORT_M if short_m is None else short_m
        v = targets - eye
        length = np.hypot(v[:, 0], v[:, 2])
        s = np.arange(1.0, float(length.max()) - short, self.STEP_M)
        use = s[None, :] < (length - short)[:, None]
        p = eye + (s[None, :, None] / length[:, None, None]) * v[:, None, :]
        ground = self.ms.sample(self.bed, self.grid, p[..., 0], p[..., 2])
        blocked = (use & (p[..., 1] < ground)).any(axis=1)
        cell = np.floor(p[..., [0, 2]] / self.CELL_M) - self.c0
        tr = self.trees
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                ix, iz = (cell[..., 0] + dx).astype(int), (cell[..., 1] + dz).astype(int)
                inside = (ix >= 0) & (iz >= 0) & (ix < self.shape[1]) & (iz < self.shape[0])
                k = np.where(inside, iz * self.shape[1] + ix, 0)
                H, r, con = tr["H"][k], tr["r"][k], tr["con"][k]
                q = np.hypot(p[..., 0] - tr["ctr"][k, 0], p[..., 2] - tr["ctr"][k, 1])
                hy = p[..., 1] - ground
                a = 0.36 * H
                with np.errstate(invalid="ignore"):
                    cone = con & (hy >= 0.12 * H) & (hy <= H) & (q <= r * (H - hy) / (0.88 * H))
                    crown = ~con & ((q / r) ** 2 + ((hy - (H - a)) / a) ** 2 <= 1.0)
                    trunk = (q <= 0.25 + 0.012 * H) & (hy >= -0.5) & (hy <= np.where(con, 0.5 * H, H - a))
                blocked |= (use & inside & ~np.isnan(H) & (cone | crown | trunk)).any(axis=1)
        return int((~blocked).sum())


class SiteTests(unittest.TestCase):
    def test_staging_glo30_reproduces_the_original_mosaic(self):
        from tools.site import stage

        np.testing.assert_array_equal(stage.dem_mosaic("glo30"),
                                      np.load(inputs.require("original-intermediates") / "dem" / "mosaic.npy"))

    def test_the_regional_imagery_and_plantations_reproduce_the_original_build(self):
        import build_regional

        original = inputs.require("original-intermediates")
        built = original / "build"
        np.testing.assert_array_equal(build_regional.satellite_B(common.SITES["scardroy"]["anchor"], original / "s2tiles"),
                                      np.load(built / "S_B.npy"))
        for name in build_regional.PLANTATIONS:
            got = build_regional.plantations(name, np.load(built / f"S_{name}.npy"), np.load(built / f"R_{name}_bed.npy"),
                                             np.load(built / f"R_{name}_surf.npy"))
            np.testing.assert_array_equal(got, np.load(built / f"SF_{name}.npy"), name)

    def test_an_exact_channel_is_continuous_at_any_angle(self):
        import build_final
        from scipy import ndimage as ndi

        for angle in (0.0, 17.0, 45.0, 63.0, 90.0):
            t = np.linspace(-60.0, 60.0, 13)
            course = (100.0 + t * np.cos(np.radians(angle)), 100.0 + t * np.sin(np.radians(angle)), 2.5)
            sd = build_final.channel_sd([course], (200, 200))
            channel = sd < 0.0
            _, pieces = ndi.label(channel, structure=np.ones((3, 3)))
            ii, jj = np.nonzero(channel)
            self.assertEqual(pieces, 1, angle)
            self.assertGreater(np.hypot(np.ptp(ii), np.ptp(jj)), 115.0, angle)     # the whole 120-texel course
            self.assertAlmostEqual(sd[100, 100], -1.25, delta=0.1)                 # half the width on the line
        across = build_final.channel_sd([(np.array([20.0, 180.0]), np.array([50.0, 50.0]), 2.5)], (200, 200))
        self.assertLess(across[50, 100], 0.0)              # a course along row 50, with columns 20 to 180
        self.assertGreater(across[100, 50], 10.0)

    def test_a_graded_channel_never_rises_along_its_course(self):
        import build_final

        rng = np.random.default_rng(3)
        ground = 260.0 - 0.05 * np.arange(200)[None, :] + rng.normal(0.0, 1.5, (40, 200))   # falling east, bumpy
        ground[:, 90:110] = np.inf                                                         # a culvert: not ground
        course = (np.linspace(5.0, 195.0, 60), np.full(60, 20.0))                          # eastwards, in texels
        level = build_final.downhill([course], ground)
        along = level[20, 5:196]
        self.assertTrue(np.all(np.diff(along) <= 0.0))
        self.assertTrue(np.all(along[:85] <= ground[20, 5:90]))
        self.assertEqual(level[23, 150], level[20, 150])          # beside the line: its nearest point's level

    def test_the_site_built_is_the_one_whose_inputs_are_staged(self):
        with tempfile.TemporaryDirectory() as folder:
            stamp = Path(folder, "stage.json")
            for site in common.SITES:
                stamp.write_text(json.dumps({"site": site, "dem": "glo30"}), encoding="utf-8")
                self.assertEqual(common.site_name(stamp), site)

    def test_a_stamp_naming_no_known_site_is_refused_with_its_remedy(self):
        with tempfile.TemporaryDirectory() as folder:
            stamp = Path(folder, "stage.json")
            for text in (json.dumps({"site": "y"}), "{not json", json.dumps(["x"])):
                stamp.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "tools.site.stage"):
                    common.site_name(stamp)

    def test_the_page_credits_the_elevation_model_staged_not_the_sites_default(self):
        """A build on another model (tools.site.build x --dem glo30) is credited to that model."""
        with tempfile.TemporaryDirectory() as folder:
            stamp = Path(folder, "stage.json")
            self.assertEqual(common.site_dem(stamp), common.SITES[common.site_name(stamp)]["dem"])    # nothing staged
            for model in terrain.MODELS:
                stamp.write_text(json.dumps({"site": "x", "dem": model}), encoding="utf-8")
                self.assertEqual(common.site_dem(stamp), model)

    def test_only_a_build_step_needs_a_finished_stage(self):
        """A build step's `from common import *` takes the staged site, and refuses an unfinished stage; the site tools
        import common's constants alone, so they can stage again."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "src").mkdir()
            (root / "build").mkdir()
            shutil.copy(ROOT / "src" / "common.py", root / "src")
            shutil.copy(ROOT / "georef.json", root)
            build_step = "from common import *; import json; print(json.dumps([SITE['lake'], CASTLE_E, CASTLE_N]))"
            run = lambda code: subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=root / "src",
                                              capture_output=True, text=True)
            for name, site in common.SITES.items():
                (root / "build" / "stage.json").write_text(json.dumps({"site": name, "dem": "glo30"}), encoding="utf-8")
                self.assertEqual(json.loads(run(build_step).stdout), [site["lake"], *site["anchor"]], name)
            (root / "build" / "stage.json").write_text(json.dumps({"staging": "x", "dem": "glo30"}), encoding="utf-8")
            refused = run(build_step)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("tools.site.stage x", refused.stderr)
            tool = run("from common import SITES, TM, to_ll, world_to_ff")
            self.assertEqual(tool.returncode, 0, tool.stderr)

    def test_without_staged_inputs_the_site_is_the_one_the_packed_page_shows(self):
        import build_extras

        name = common.site_name(ROOT / "absent-stage.json")
        anchor = common.SITES[name]["anchor"]
        lon, lat = common.to_ll.transform(*anchor)
        extras = packed_meta()["extras"]
        np.testing.assert_allclose(extras["castle_ll"], [lat, lon], atol=1e-9)
        self.assertEqual((extras["ref"], extras["place"]), (build_extras.reference(anchor), common.SITES[name]["place"]))

    def test_the_location_note_gives_the_essays_reference_point_from_the_castle(self):
        import build_extras

        self.assertEqual(build_extras.reference(common.SITES["scardroy"]["anchor"]), {"km": 2.5, "dir": "SSW"})
        self.assertEqual(build_extras.reference(common.SITES["x"]["anchor"]), {"km": 6.5, "dir": "SE"})
        self.assertEqual([common.compass16(e, n) for e, n in ((0, 1), (1, 0.01), (0.2, -1), (-0.19, 1))],
                         ["N", "E", "SSE", "N"])


OSTN15, OSTN15_FILE = "site-ostn15", "uk_os_OSTN15_NTv2_OSGBtoETRS.tif"


@contextlib.contextmanager
def sandbox(name):
    """A private state folder and copies of the input lock and of one source's pins and recipe record, so fetching and
    pinning run for real without touching this machine's state or inputs or the repository's pins; the sandbox's root."""
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        locks = root / "tools" / "site" / "locks"
        locks.mkdir(parents=True)
        shutil.copy(inputs.LOCK, root / "tools" / inputs.LOCK.name)
        for suffix in (".json", ".recipe.json"):
            shutil.copy(sources.LOCKS / f"{name}{suffix}", locks)
        environ = {k: v for k, v in os.environ.items() if not k.startswith("HOG_INPUT_")}
        with mock.patch.dict(os.environ, environ | {"HOG_STATE_DIR": str(root / "state")}, clear=True), \
                mock.patch.object(inputs, "LOCK", root / "tools" / inputs.LOCK.name), \
                mock.patch.object(inputs, "BUILD", root / "build"), \
                mock.patch.object(sources, "LOCKS", locks), contextlib.redirect_stdout(io.StringIO()):
            yield root


class SourceTests(unittest.TestCase):
    def test_canonical_extracts_ignore_the_server_header_order_and_formatting(self):
        a = {"type": "way", "id": 7, "tags": {"name": "Allt Mhartuin", "waterway": "river"}, "geometry": []}
        b = {"type": "node", "id": 3, "lat": 57.5, "lon": -5.1, "tags": {"place": "village", "name": "Achnasheen"}}
        first = sources.canonical_osm(overpass_payload([a, b], "2026-10-31T21:10:02Z"))
        second = sources.canonical_osm(json.dumps({"elements": [dict(reversed(list(b.items()))), a],
                                                   "osm3s": {"timestamp_osm_base": "2026-11-02T08:00:00Z"}},
                                                  indent=2).encode())
        self.assertEqual(first, second)
        a["tags"]["waterway"] = "stream"
        self.assertNotEqual(sources.canonical_osm(overpass_payload([a, b], "2026-10-31T21:10:02Z")), first)

    def test_the_tiles_cover_every_imagery_texel_of_both_layers(self):
        recipe = sources.SOURCES["site-s2-x"].recipe
        for (z, x0, x1, y0, y1), half in zip(recipe.ranges, (12000.0 + 15.0 + 7.5, 36000.0 + 60.0 + 30.0)):
            west, south, east, north = sources.bounds(sources.X, half)
            for lon, lat in ((west, north), (east, south), (west, south), (east, north)):
                tx, ty = sources.tile_xy(lon, lat, z)
                self.assertTrue(x0 <= tx < x1 + 1 and y0 <= ty < y1 + 1, (z, lon, lat))

    def test_each_source_pins_exactly_the_files_its_recipe_writes(self):
        def written(recipe):
            if isinstance(recipe, sources.Http):
                return {rel for rel, _ in recipe.files}
            if isinstance(recipe, sources.ZipMembers):
                return set(recipe.members)
            if isinstance(recipe, sources.Window):
                return {f"{recipe.stem}.npy", f"{recipe.stem}.json"}
            if isinstance(recipe, sources.Overpass):
                return {recipe.file}
            return set(sources.tile_names(recipe))

        for name, source in sources.SOURCES.items():
            self.assertEqual(set(inputs.tree_manifest(name)), written(source.recipe), name)

    def test_a_seed_is_copied_only_when_it_holds_the_pinned_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            seed = Path(folder) / "seed"
            (seed / "dem").mkdir(parents=True)
            good = b"the pinned tile"
            pin = (len(good), hashlib.sha256(good).hexdigest())
            target = Path(folder) / "out" / "tile.tif"
            with mock.patch.dict(os.environ, {inputs.env_name("original-intermediates"): str(seed)}):
                (seed / "dem" / "tile.tif").write_bytes(b"other bytes of a tile")
                self.assertFalse(sources._seed(("original-intermediates", "dem"), "tile.tif", target, pin))
                self.assertFalse(target.exists())
                (seed / "dem" / "tile.tif").write_bytes(good)
                self.assertTrue(sources._seed(("original-intermediates", "dem"), "tile.tif", target, pin))
                self.assertEqual(target.read_bytes(), good)

    def test_the_gedtm30_window_holds_metres(self):
        import rasterio

        ged = np.load(inputs.require("site-gedtm30") / "gedtm30_n57_w006_w004.npy")
        with rasterio.open(inputs.require("site-glo30") / "Copernicus_DSM_COG_10_N57_00_W006_00_DEM.tif") as ds:
            glo = ds.read(1)
        # both put pixel centres on whole arc-seconds from 58 N, 6 W: GLO-30 every 1.5" east, GEDTM30 every 1"
        a, b = glo[:, ::2], ged[:, ::3][:, :glo.shape[1] // 2]
        land = (a > 50) & np.isfinite(b)
        self.assertLess(abs(float(np.median(b[land] - a[land]))), 3.0)

    def test_overpass_errors_are_refused(self):
        payload = json.loads(overpass_payload([], "2026-10-31T21:10:02Z"))
        payload["remark"] = "runtime error: Query timed out in \"query\" at line 1 after 901 seconds."
        with self.assertRaises(ValueError):
            sources.canonical_osm(json.dumps(payload).encode())

    def test_each_source_is_pinned_from_its_current_recipe(self):
        for name, source in sources.SOURCES.items():
            recorded = json.loads((sources.LOCKS / f"{name}.recipe.json").read_text(encoding="utf-8"))
            self.assertEqual(recorded, sources.recipe_record(source.recipe), name)

    def test_a_fetch_refuses_a_recipe_changed_since_pinning(self):
        fetched = []
        stale = sources.recipe_record(sources.SOURCES[OSTN15].recipe) | {"files": [["grid.tif", "https://old/grid.tif"]]}
        with sandbox(OSTN15), mock.patch.dict(sources.FETCH, {sources.Http: lambda *args: fetched.append(args)}):
            (sources.LOCKS / f"{OSTN15}.recipe.json").write_text(json.dumps(stale), encoding="utf-8")
            with self.assertRaises(inputs.Blocked):
                sources.fetch(OSTN15)
        self.assertEqual(fetched, [])

    def test_pinning_fetches_afresh_without_seeds(self):
        seen = []

        def fake_http(recipe, pins, out, seeding):
            seen.append(((out / OSTN15_FILE).exists(), pins, seeding))
            (out / OSTN15_FILE).write_bytes(b"fresh bytes")

        with sandbox(OSTN15), mock.patch.dict(sources.FETCH, {sources.Http: fake_http}):
            (sources.folder(OSTN15) / OSTN15_FILE).write_bytes(b"stale bytes already cached")
            sources.fetch(OSTN15, pin=True)
            self.assertEqual(seen, [(False, {}, False)])
            self.assertEqual((sources.folder(OSTN15) / OSTN15_FILE).read_bytes(), b"fresh bytes")

    def test_pinning_sets_aside_a_pinned_copy_whose_pins_change(self):
        real = inputs.require(OSTN15) / OSTN15_FILE

        def next_release(recipe, pins, out, seeding):
            (out / OSTN15_FILE).write_bytes(b"the publisher's next release")

        with sandbox(OSTN15), mock.patch.dict(sources.FETCH, {sources.Http: next_release}):
            shutil.copy(real, sources.folder(OSTN15))
            sources.fetch(OSTN15, pin=True)
            [aside] = sources.folder(OSTN15).parent.glob(f"{OSTN15}.pinned-*")
            self.assertTrue(filecmp.cmp(aside / OSTN15_FILE, real, shallow=False))

    def test_pinning_to_the_same_pins_keeps_a_single_copy(self):
        real = inputs.require(OSTN15) / OSTN15_FILE
        with sandbox(OSTN15), \
                mock.patch.dict(sources.FETCH, {sources.Http: lambda recipe, pins, out, seeding: shutil.copy(real, out)}):
            shutil.copy(real, sources.folder(OSTN15))
            sources.fetch(OSTN15, pin=True)
            self.assertEqual([p.name for p in sources.folder(OSTN15).parent.iterdir()], [OSTN15])

    def test_a_copy_that_cannot_be_set_aside_keeps_its_pins(self):
        real = inputs.require(OSTN15) / OSTN15_FILE

        def next_release(recipe, pins, out, seeding):
            (out / OSTN15_FILE).write_bytes(b"the publisher's next release")

        with sandbox(OSTN15), mock.patch.dict(sources.FETCH, {sources.Http: next_release}):
            shutil.copy(real, sources.folder(OSTN15))
            pinned = inputs.tree_manifest(OSTN15)
            with mock.patch.object(Path, "rename", side_effect=PermissionError("a file in it is open")), \
                    self.assertRaises(PermissionError):
                sources.fetch(OSTN15, pin=True)
            self.assertEqual(inputs.tree_manifest(OSTN15), pinned)
            self.assertTrue(filecmp.cmp(sources.folder(OSTN15) / OSTN15_FILE, real, shallow=False))

    def test_a_fetch_uses_a_verified_kept_copy_without_downloading(self):
        real, fetched = inputs.require(OSTN15) / OSTN15_FILE, []
        with sandbox(OSTN15) as root, mock.patch.dict(sources.FETCH, {sources.Http: lambda *args: fetched.append(args)}):
            kept = root / "kept"
            kept.mkdir()
            shutil.copy(real, kept)
            inputs.main(["set", OSTN15, str(kept)])
            self.assertEqual(sources.fetch(OSTN15).resolve(), kept.resolve())
        self.assertEqual(fetched, [])

    def test_pinning_everything_leaves_out_sources_whose_publisher_serves_only_the_current_release(self):
        calls = []
        with mock.patch.object(sources, "fetch", lambda name, pin=False: calls.append(name)), \
                contextlib.redirect_stdout(io.StringIO()):
            sources.main(["pin"])
            self.assertEqual(calls, [n for n in sources.SOURCES if n != "site-os50"])
            calls.clear()
            sources.main(["pin", "site-os50"])
            self.assertEqual(calls, ["site-os50"])

    def test_one_failing_source_does_not_stop_the_rest(self):
        calls = []

        def fetch(name, pin=False):
            calls.append(name)
            if name == "site-os50":
                raise ValueError("SHA256 hash of downloaded file does not match the known hash")

        with mock.patch.object(sources, "fetch", fetch), contextlib.redirect_stderr(io.StringIO()):
            status = sources.main(["fetch"])
        self.assertEqual((status, calls), (1, list(sources.SOURCES)))

    def test_overpass_boxes_are_centred_on_the_site(self):
        lon, lat = sources.to_ll.transform(*sources.X)
        number = r"(-?\d+\.\d+)"
        for name in ("site-osm-x", "site-osm-water-x"):
            boxes = set(re.findall(rf"\({number},{number},{number},{number}\)", sources.SOURCES[name].recipe.body))
            self.assertTrue(boxes)
            for s, w, n, e in ((float(v) for v in box) for box in boxes):
                self.assertLess(abs((s + n) / 2 - lat), 0.01, name)
                self.assertLess(abs((w + e) / 2 - lon), 0.02, name)


class TerrainTests(unittest.TestCase):
    def test_a_lattice_reads_its_edge_pixels_out_to_their_outer_edges(self):
        heights = np.arange(12, dtype=np.float32).reshape(3, 4)
        lattice = (heights, -6.0, 58.0, 0.5, 0.5)        # centres at 58, 57.5 and 57 N, and every 0.5 degrees from 6 W
        lon = np.array([-4.5, -6.24, -6.0, -5.0])
        lat = np.array([57.0 - 1e-12, 56.76, 56.74, 58.24])     # a hair past the last centre, inside, beyond, inside
        got = terrain._sample_lattice(lattice, lon, lat)
        np.testing.assert_array_equal(got[[0, 1, 3]], heights[[2, 2, 0], [3, 0, 2]])
        self.assertTrue(np.isnan(got[2]))

    def test_staging_refuses_a_model_that_leaves_heights_missing_on_the_lattice(self):
        from tools.site import stage

        def one_missing_per_call(model, lon, lat):
            z = np.zeros(np.broadcast(lon, lat).shape, np.float32)
            z[-1, -1] = np.nan
            return z

        calls = -(-stage.LATTICE[0] // 300)
        with mock.patch.object(stage.T, "heights_ll", one_missing_per_call), \
                self.assertRaisesRegex(ValueError, f"fabdem: {calls} heights missing"):
            stage.dem_mosaic("fabdem")

    def test_heights_by_longitude_and_latitude_broadcast_for_every_model(self):
        lon0, lat0 = terrain.ll(*X)
        lon = lon0 + np.array([[-0.01, 0.0, 0.01]])
        lat = lat0 + np.array([[-0.005], [0.005]])
        E, N = terrain.tm(*np.broadcast_arrays(lon, lat))
        for model in terrain.MODELS:
            got = terrain.heights_ll(model, lon, lat)
            self.assertEqual(got.shape, (2, 3), model)
            np.testing.assert_allclose(got, terrain.heights(model, E, N), atol=0.05, err_msg=model)

    def test_restaging_that_fails_part_way_leaves_a_stamp_the_build_refuses(self):
        from tools.site import stage

        with tempfile.TemporaryDirectory() as folder:
            stamp = Path(folder, "stage.json")
            stamp.write_text(json.dumps({"site": "scardroy", "dem": "glo30"}), encoding="utf-8")
            with mock.patch.object(stage, "STAMP", stamp), \
                    mock.patch.object(stage, "require", side_effect=inputs.Blocked("no original intermediates")), \
                    self.assertRaises(inputs.Blocked):
                stage.stage("x")
            with self.assertRaisesRegex(RuntimeError, "tools.site.stage x"):     # neither site: stage x again
                common.site_name(stamp)

    def test_filling_raises_closed_depressions_to_their_spill_level(self):
        z = np.full((9, 9), 10.0)
        z[2:7, 2:7] = 12.0                    # a rim round a pit, open to nothing
        z[4, 4] = 9.0
        z[3:6, 3:6] = np.minimum(z[3:6, 3:6], 11.0)
        z[8, 4] = 8.0                         # a notch draining the outer ground off the edge
        filled = common.fill(z)
        self.assertEqual(filled[4, 4], 12.0)                        # spills over the lowest point of its rim
        np.testing.assert_array_equal(filled[z == 10.0], 10.0)     # open ground stays as it is
        self.assertTrue(np.all(filled >= z))

    def test_an_outlet_drains_what_reaches_it(self):
        z = np.full((9, 9), 20.0)
        z[3:6, 3:6] = 5.0                     # a basin...
        outlet = np.zeros(z.shape, bool)
        outlet[4, 4] = True                   # ...that drains away, like a lake, so stays as it is
        np.testing.assert_array_equal(common.fill(z, outlet), z)
        self.assertEqual(common.fill(z)[4, 4], 20.0)

    def test_national_grid_goes_through_ostn15_not_a_ballpark(self):
        from pyproj import Transformer

        lon, lat = terrain.ll(*X)
        helmert = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True).transform(lon, lat)
        ballpark = Transformer.from_crs(common.TM, "EPSG:27700", always_xy=True).transform(*X)
        ostn15 = terrain.bng(*X)
        self.assertLess(np.hypot(*np.subtract(ostn15, helmert)), 3.0)          # the 7-parameter fit is good to 2 m
        self.assertGreater(np.hypot(*np.subtract(ostn15, ballpark)), 50.0)     # no datum shift at all is 80 m out

    def test_national_grid_applies_the_ostn15_grid_shift(self):
        from pyproj import Transformer

        grid = inputs.require("site-ostn15") / "uk_os_OSTN15_NTv2_OSGBtoETRS.tif"
        explicit = Transformer.from_pipeline(
            "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad "
            f"+step +inv +proj=hgridshift +grids={grid.as_posix()} "
            "+step +proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=airy")
        rng = np.random.default_rng(7)
        E, N = X[0] + rng.uniform(-36000, 36000, 50), X[1] + rng.uniform(-36000, 36000, 50)
        np.testing.assert_allclose(np.column_stack(terrain.bng(E, N)),
                                   np.column_stack(explicit.transform(*terrain.ll(E, N))), atol=0.05)

    def test_glo30_reads_as_the_production_sampler_does(self):
        root = inputs.require("original-intermediates") / "dem"
        mosaic = np.concatenate([np.load(root / "W006.npy"), np.load(root / "W005.npy")], axis=1)
        rng = np.random.default_rng(3)
        E, N = X[0] + rng.uniform(-36000, 36000, 500), X[1] + rng.uniform(-36000, 36000, 500)
        np.testing.assert_allclose(terrain.heights("glo30", E, N), common.dem_sample(mosaic, E, N), atol=1e-3)

    def test_os50_lines_up_with_glo30(self):
        from tools.site import registration

        tiles = np.array(registration.tile_offsets(X, "os50", reach=6000.0, spacing=4000.0))
        self.assertGreaterEqual(len(tiles), 4)
        # read half a post out, OS Terrain 50 lies 35 m from GLO-30 (25 m on each axis)
        self.assertLess(np.hypot(np.median(tiles[:, 2]), np.median(tiles[:, 3])), 5.0)

    def test_bare_earth_models_stay_within_metres_of_the_surface_model(self):
        rng = np.random.default_rng(5)
        E, N = X[0] + rng.uniform(-30000, 30000, 4000), X[1] + rng.uniform(-30000, 30000, 4000)
        glo = terrain.heights("glo30", E, N)
        for model in ("fabdem", "gedtm30", "os50"):
            other = terrain.heights(model, E, N)
            land = (glo > 50) & np.isfinite(other)
            self.assertLess(abs(float(np.median(other[land] - glo[land]))), 3.0, model)


class ReviewPortTests(unittest.TestCase):
    """The ported location-review model must reproduce the review's own numbers at X."""

    @classmethod
    def setUpClass(cls):
        E0, N0, n = terrain.square(X, 33000, 25.0)
        dem = terrain.model_grid("glo30", E0, N0, 25.0, n, n)
        g = layout.evaluate(dem, terrain.water_fraction(dem), [X], step=10)
        h = sun.site_metrics(dem, np.array([X[0]]), np.array([X[1]]), g["L"])
        cls.got = {k: float(v[0]) for k, v in (g | h).items() if np.ndim(v)}

    def test_the_lake_level_sun_skyline_and_ordering_match_exactly(self):
        for key in ("L", "Ls", "set_0606", "set_0621", "rise_0331", "rise_0504", "rise_0622", "north_max", "south_max",
                    "o_hagrid", "village_slope"):
            self.assertAlmostEqual(self.got[key], REVIEW_AT_X[key], delta=0.001, msg=key)

    def test_the_earthworks_match_within_the_water_rasterisation(self):
        # the review rasterised water at 10 m first; a burn's channel polygon under the castle then covers a few more
        # 25 m cells, whose bed the model puts 10 m down: 0.3 % of the volume and 2.6 m of the worst fill
        self.assertAlmostEqual(self.got["V_total"] / REVIEW_AT_X["V_total"], 1.0, delta=0.01)
        self.assertLess(abs(self.got["max_fill"] - REVIEW_AT_X["max_fill"]), 3.0)


class SunTests(unittest.TestCase):
    def test_twilight_cannot_stand_in_for_sunset(self):
        # just off X, the formula's spike in civil twilight once became this day's sunset, 54 minutes late
        lon, lat = terrain.ll(X[0] - 1000.0, X[1] + 500.0)
        official, _, _ = sun.rise_set("1994-06-06", lat, lon, "set")
        self.assertTrue(np.datetime64("1994-06-06T21:00") < official < np.datetime64("1994-06-06T21:30"), official)

    def test_sea_horizon_times_change_smoothly_through_a_year(self):
        lon, lat = terrain.ll(*X)
        for kind in ("rise", "set"):
            days = np.arange(np.datetime64("2026-01-01"), np.datetime64("2027-01-01"))
            minutes = [((sun.rise_set(str(d), lat, lon, kind)[0] - d.astype("datetime64[s]")).astype(float) / 60) for d in days]
            self.assertLess(np.abs(np.diff(minutes)).max(), 5.0, kind)


class RegistrationTests(unittest.TestCase):
    def test_a_known_shift_is_recovered_within_two_metres(self):
        from tools.site import registration

        step = registration.STEP
        E0, N0, n = terrain.square((X[0] + 4000, X[1] - 4000), registration.TILE_M / 2, step)
        E, N = np.meshgrid(E0 + step * np.arange(n), N0 + step * np.arange(n))
        reference = terrain.heights("glo30", E, N)
        for dE, dN in ((25.0, 0.0), (-12.5, 37.5), (60.0, 45.0)):
            moved = terrain.heights("glo30", E - dE, N - dN)     # its features lie (dE, dN) from the reference's
            got = registration.offset(reference, moved, step)
            self.assertLess(np.hypot(got[0] - dE, got[1] - dN), 2.0, (dE, dN, got))

    def test_a_known_shift_is_recovered_from_a_differently_textured_model(self):
        from scipy import ndimage as ndi
        from tools.site import registration

        step = registration.STEP
        E0, N0, n = terrain.square((X[0] + 4000, X[1] + 4000), registration.TILE_M / 2, step)
        E, N = np.meshgrid(E0 + step * np.arange(n), N0 + step * np.arange(n))
        reference = terrain.heights("glo30", E, N)
        # another model's view of the same ground: smoother, with its own errors varying over its 50 m posts
        other = ndi.gaussian_filter(terrain.heights("glo30", E - 20.0, N + 10.0), 25.0 / step)
        errors = np.random.default_rng(2).normal(0.0, 1.0, (n // 10 + 2, n // 10 + 2))
        other += ndi.zoom(errors, 10, order=1)[:n, :n]
        got = registration.offset(reference, other, step)
        self.assertLess(np.hypot(got[0] - 20.0, got[1] + 10.0), 2.0, got)


def glsl_calls(source, name):
    """The arguments of every call to `name` in GLSL source (not its definition), split at top-level commas."""
    calls = []
    for match in re.finditer(rf"(?<!float )\b{name}\(", source):
        depth, start, i, args = 1, match.end(), match.end(), []
        while depth:
            depth += {"(": 1, ")": -1}.get(source[i], 0)
            if source[i] == "," and depth == 1 or depth == 0:
                args.append(source[start:i].strip())
                start = i + 1
            i += 1
        calls.append(args)
    return calls


def level(expr, lake):
    """A GLSL height in metres, a literal or LAKE plus or minus one, with LAKE set; None for any other expression."""
    match = re.fullmatch(r"\(?(?:LAKE\s*([+-])\s*)?(\d+\.\d*)\)?", expr.strip())
    if match is None:
        return None
    sign, value = match.groups()
    return float(value) if sign is None else lake + (float(value) if sign == "+" else -float(value))


def lake_relative(test, source, bands=2):
    """Assert that the castle's window bands (`bands` of them), its shading ramp and the fog in GLSL `source` stand at
    Scardroy's levels with its lake and move with any other lake. The frozen baseline has two bands, the shading pass's
    and its raymarched castle's reflection; the page has the first alone, since the meshes replaced that castle."""
    windows = [args[2] for args in glsl_calls(source, "windowMask") if level(args[2], 0.0) is not None]
    shading = [args[:2] for args in glsl_calls(source, "smoothstep") if args[2] == "p.y"]
    fog = re.findall(r"ro\.y - (\([^()]*\)|[^()\s]+)\)", source)
    levels = ([(w, 192.0) for w in windows] + [(a, 188.0) for a, _ in shading] + [(b, 200.0) for _, b in shading]
              + [(f, 150.0) for f in fog if level(f, 0.0) is not None])
    test.assertEqual(len(levels), bands + 3)          # the castle's window bands, its shading ramp (two levels), the fog
    scardroy, x = common.SITES["scardroy"]["lake"], common.SITES["x"]["lake"]
    for expr, at_scardroy in levels:
        test.assertEqual(level(expr, scardroy), at_scardroy, expr)
        test.assertEqual(level(expr, x), at_scardroy + x - scardroy, expr)


class BaselineTests(unittest.TestCase):
    """tools/site/baseline.py: the frozen baseline page re-frozen at a site."""

    def test_the_baseline_moves_the_castle_and_fog_with_the_lake_colours_its_ground_as_the_page_and_carries_the_sites_data(
            self):
        """The frozen renderer, its levels made lake-relative, keeps its own colours for the surface classes it had and
        takes the page's (src/frag4.glsl classCol) for those the site's data adds, the overgrown garden's; nothing else
        of it changes but its data."""
        from tools.site import baseline

        try:
            page = inputs.require("baseline-s4-idle").read_bytes().decode("utf-8")
        except inputs.Blocked as e:
            raise unittest.SkipTest(f"blocked: {e}") from e
        data = (ROOT / "build" / "data.js").read_bytes().decode("ascii")
        out = baseline.rebased(page, data)
        lake_relative(self, out)
        self.assertEqual(class_colours(out), {**class_colours(), **class_colours(page)})

        def split(text):            # (before the DATA statement's object, the object, after it)
            start = text.index(baseline.DATA) + len(baseline.DATA)
            obj, end = json.JSONDecoder().raw_decode(text, start)
            return text[:start], obj, text[end:]

        expected = page
        for old, new in baseline.LEVELS:
            expected = expected.replace(old, new)
        expected = expected.replace(*(baseline.function(t, baseline.CLASS_COLOURS) for t in (page, out)))
        before, obj, after = split(out)
        self.assertEqual(obj, json.loads(data[data.index("{"): data.rindex("}") + 1]))
        self.assertEqual((before, after), split(expected)[::2])     # and nothing else changes

    def test_the_baseline_keeps_its_own_ground_colours_and_takes_only_the_classes_it_lacks(self):
        """Parity still shows a change to the colour of a class the frozen renderer had: of the page's classCol it
        takes only the lines for the classes after its last, whose own line becomes a branch."""
        from tools.site import baseline

        frozen = ("vec3 classCol(float c, vec2 xz) {\n  if (c < 0.5) return vec3(0.1, 0.2, 0.1);   // meadow\n"
                  "  return vec3(0.3, 0.3, 0.3);   // shingle\n}\n")
        page = ("vec3 classCol(float c, vec2 xz) {\n  if (c < 0.5) return vec3(0.9, 0.2, 0.9);   // meadow, recoloured\n"
                "  if (c < 7.5) return vec3(0.3, 0.3, 0.3);   // shingle\n"
                "  return mix(vec3(0.0, 0.1, 0.0), vec3(0.1, 0.1, 0.0), vnoise(xz * 0.5));   // added\n}\n")
        merged = baseline.with_added_classes(frozen, page)
        self.assertEqual(class_colours(merged), {0: ([(0.1, 0.2, 0.1)], []), 7: ([(0.3, 0.3, 0.3)], []),
                                                 8: ([(0.0, 0.1, 0.0), (0.1, 0.1, 0.0)], [0.5])})
        self.assertEqual(baseline.with_added_classes(frozen, frozen), frozen)

    def test_a_page_without_the_frozen_levels_is_refused(self):
        from tools.site import baseline

        page = "\n".join(old for old, _ in baseline.LEVELS[1:]) + '\nconst DATA={"a":1};</script>\n'
        with self.assertRaises(ValueError):
            baseline.rebased(page, 'const DATA={"a":2};\n')

    def test_the_baseline_flies_the_pages_tour(self):
        # S6 sets the baseline and the page at the same tour times, so the two must fly one tour
        from tools.site import baseline

        tour = lambda look: (f'const WP = [\n    [1100, -1500, 300, 0, 0, {look}, "Hogwarts Castle"],\n  ];\n'
                             '  function pathAt(t) { return t; }\n\n  ')
        section = "// ---------------------------------------------------------------- HUD text\n"
        colours = "vec3 classCol(float c, vec2 xz) {\n  return vec3(0.3, 0.3, 0.3);\n}\n"
        frozen = ("<script>" + "\n".join(old for old, _ in baseline.LEVELS) + "\n" + colours + 'const DATA={"a":1};\n  '
                  + tour(40) + section + "</script>")
        page = frozen.replace(tour(40), tour(60))
        out = baseline.rebased(frozen, 'const DATA={"a":2};\n', tour=baseline.tour_text(page), shader=colours)
        self.assertEqual(baseline.tour_text(out), tour(60))
        frozen_tour = baseline.rebased(frozen, 'const DATA={"a":2};\n', shader=colours)
        self.assertEqual(out.replace(tour(60), tour(40)), frozen_tour)

    def test_a_page_without_its_one_class_colours_is_refused(self):
        from tools.site import baseline

        page = "\n".join(old for old, _ in baseline.LEVELS) + '\nconst DATA={"a":1};</script>\n'
        with self.assertRaises(ValueError):
            baseline.rebased(page, 'const DATA={"a":2};\n')


class ParityTests(unittest.TestCase):
    def test_a_pixels_difference_is_its_largest_channels_and_only_those_over_the_level_count(self):
        from tools.site import parity

        a = np.zeros((10, 10, 3), np.uint8)
        b = a.copy()
        b[0, :5, 1] = parity.DIFF_LEVEL + 1
        b[1, 0, 2] = parity.DIFF_LEVEL
        half = parity.DIFF_LEVEL // 2 + 1
        b[2, 0, :2] = half                                  # two channels under the level, together over it
        _, mean, _, peak, share = parity.difference(a, b)
        self.assertEqual((peak, share), (parity.DIFF_LEVEL + 1, 0.05))
        self.assertAlmostEqual(mean, (5 * (parity.DIFF_LEVEL + 1) + parity.DIFF_LEVEL + half) / 100)
        self.assertEqual(parity.difference(a, a)[1:], (0.0, 0.0, 0, 0.0))

    def test_parity_runs_only_against_a_baseline_carrying_the_pages_data(self):
        import tempfile
        from tools.site import parity

        with tempfile.TemporaryDirectory() as folder:
            pages = [Path(folder) / f"{name}.html" for name in "abc"]
            for path, value in zip(pages, (1, 1, 2)):
                path.write_text(f'<p></p>\n<script>const DATA={{"lake":{value}}};</script>\n', encoding="utf-8")
            self.assertTrue(parity.same_data(pages[0], pages[1]))
            self.assertFalse(parity.same_data(pages[0], pages[2]))


class BuildTests(unittest.TestCase):
    def test_s0_starts_from_the_default_sites_input(self):
        from tools.site import build

        name = build.S0_INPUTS[common.site_name(ROOT / "absent-stage.json")]
        stages = (ROOT / "tools" / "stages.py").read_text(encoding="utf-8")
        self.assertEqual(set(re.findall(r'"(data-before-railway[^"]*)"', stages)), {name})
        self.assertEqual(inputs.spec(name)["file"], f"{name}.js")      # where tools.site.build keeps it

    def test_the_s0_input_is_kept_between_fix_berth_and_fix_railway(self):
        from tools.site import build

        ran = []

        def step(name):
            ran.append(name)
            (out / "build" / "data.js").write_text(" ".join(ran), encoding="ascii")

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "build").mkdir()
            with mock.patch.object(build.STAGE, "stage") as stage, mock.patch.object(build, "run", step), \
                    mock.patch.object(build, "BUILD", out / "build"), mock.patch.object(build, "ROOT", out):
                name, kept = build.build("x")
            stage.assert_called_once_with("x", None)
            self.assertEqual(ran[-4:], ["pack", "fix_berth", "fix_harbour", "fix_railway"])
            self.assertEqual((name, kept), ("data-before-railway-x", out / "build" / "inputs" / "data-before-railway-x.js"))
            self.assertEqual(kept.read_text(encoding="ascii"), " ".join(ran[:-2]))
            self.assertEqual((out / "build" / "data.js").read_text(encoding="ascii"), " ".join(ran))

    def test_s0_finishes_the_data_as_the_site_build_does(self):
        # the steps after the kept S0 input: S0 reruns fix_berth (idempotent) on it, then the same steps in order
        from tools.site import build

        stages = (ROOT / "tools" / "stages.py").read_text(encoding="utf-8")
        steps = re.search(r'for script in \(([^)]*)\)', stages).group(1)
        self.assertEqual(re.findall(r'"src/(\w+)\.py"', steps), ["fix_berth", *build.FINISH])

    def test_another_elevation_model_leaves_the_s0_input_alone(self):
        from tools.site import build

        own = common.SITES["x"]["dem"]
        other = next(model for model in terrain.MODELS if model != own)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            kept = out / "build" / "inputs" / "data-before-railway-x.js"
            kept.parent.mkdir(parents=True)
            kept.write_text("pinned", encoding="ascii")
            step = lambda name: (out / "build" / "data.js").write_text(name, encoding="ascii")
            with mock.patch.object(build.STAGE, "stage"), mock.patch.object(build, "run", step), \
                    mock.patch.object(build, "BUILD", out / "build"), mock.patch.object(build, "ROOT", out):
                self.assertEqual(build.build("x", other), (None, None))
                self.assertEqual(kept.read_text(encoding="ascii"), "pinned")
                build.build("x", own)                                   # naming the site's own model keeps it
                self.assertEqual(kept.read_text(encoding="ascii"), "fix_berth")


class ForestExtentTests(unittest.TestCase):
    """Where White Hound's Forest stops short of real features, and the corridors cleared through it
    (src/forest_extent.py), on a synthetic 8 m layer turned 25 degrees, as the Forest map's is."""
    TEXEL = 8.0

    @classmethod
    def setUpClass(cls):
        import forest_extent

        cls.fx = forest_extent
        a = np.radians(25.0)
        cls.ex, cls.ey = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        cls.origin = np.array([0.0, 1000.0]) - 2400.0 * (cls.ex + cls.ey)     # a 4.8 km square round (0, 1000)
        j, i = np.meshgrid(np.arange(600) + 0.5, np.arange(600) + 0.5)
        cls.E = cls.origin[0] + cls.TEXEL * (j * cls.ex[0] + i * cls.ey[0])
        cls.N = cls.origin[1] + cls.TEXEL * (j * cls.ex[1] + i * cls.ey[1])

    def read(self, layer, e, n):
        """The layer as the shader reads it: bilinear between texel centres."""
        from scipy import ndimage as ndi

        de, dn = np.atleast_1d(np.asarray(e, float)) - self.origin[0], np.atleast_1d(np.asarray(n, float)) - self.origin[1]
        j = (de * self.ex[0] + dn * self.ex[1]) / self.TEXEL - 0.5
        i = (de * self.ey[0] + dn * self.ey[1]) / self.TEXEL - 0.5
        return ndi.map_coordinates(layer, [i, j], order=1, mode="nearest")

    @staticmethod
    def house(e, n):
        return {"building": "house"}, np.array([e, e + 12.0, e + 12.0, e, e]), np.array([n, n, n + 9.0, n + 9.0, n])

    def test_corridors_keep_the_forest_off_roads_railways_and_buildings_as_the_shader_reads_it(self):
        road = ({"highway": "primary", "ref": "A890"}, np.array([-1700.0, 1700.0]), np.array([300.0, 700.0]))
        rail = ({"railway": "rail"}, np.array([-1700.0, 1700.0]), np.array([1400.0, 1200.0]))
        house = self.house(400.0, 2000.0)
        barn = ({"building": "barn"}, np.array([-700.0, -620.0, -620.0, -700.0, -700.0]),
                np.array([2000.0, 2000.0, 2080.0, 2080.0, 2000.0]))
        keep = self.fx.keep({}, self.E, self.N, [road, rail, house, barn], self.TEXEL)
        for (tags, e, n), half in ((road, self.fx.A_ROAD_M), (rail, self.fx.RAILWAY_M), (house, self.fx.BUILDING_M)):
            self.assertEqual(float(self.read(keep, *PackedSiteTests.bands(e, n, half)).max()), 0.0, tags)
        self.assertEqual(float(self.read(keep, -660.0, 2040.0)[0]), 0.0)                 # inside the barn, 40 m in
        np.testing.assert_array_equal(self.read(keep, np.linspace(-1700.0, 1700.0, 50), np.full(50, -500.0)), 1.0)

    def test_the_forest_ends_short_of_a_road_it_stops_at_and_of_the_settlement_along_it(self):
        road = ({"highway": "primary", "ref": "A832"}, np.array([-4000.0, 4000.0]), np.array([2000.0, 2000.0]))
        village, lone = self.house(100.0, 1900.0), self.house(-900.0, 600.0)
        keep = self.fx.keep({"ends_at": ("A832",)}, self.E, self.N, [road, village, lone], self.TEXEL)
        e, n = np.meshgrid(np.linspace(-1700.0, 1700.0, 69), np.linspace(2000.0 - self.fx.SETBACK_MIN_M, 2700.0, 30))
        self.assertEqual(float(self.read(keep, e, n).max()), 0.0)                    # beyond, and the setback
        between = PackedSiteTests.bands(np.array([106.0, 106.0]), np.array([1904.5, 2000.0]), 0.0)
        self.assertEqual(float(self.read(keep, *between).max()), 0.0)                # no Forest between village and road
        self.assertEqual(float(self.read(keep, *PackedSiteTests.bands(*village[1:], self.fx.VILLAGE_MIN_M)).max()), 0.0)
        e, n = np.meshgrid(np.linspace(-1700.0, 1700.0, 69), np.linspace(-700.0, 1600.0, 47))
        away = np.hypot(e + 894.0, n - 604.5) > 100.0                               # beyond the lone house's ground,
        np.testing.assert_array_equal(self.read(keep, e[away], n[away]), 1.0)       # elsewhere as drawn
        self.assertEqual(float(self.read(keep, -894.0, 604.5)[0]), 0.0)             # the lone house: a clearing

    def test_a_building_the_forest_covers_stands_in_its_homesteads_clearing_not_a_cutout(self):
        """Its yard is clear, BUILDING_M round it, and the clearing spreads on beyond as its ground: at least
        clearings.MIN_M2 in area and MIN_WIDE_M across, its edge wandering, the Forest thickening back gradually."""
        import clearings

        house = self.house(400.0, 2000.0)
        home = self.fx.homesteads(self.E, self.N, [house], self.TEXEL)
        self.assertEqual(float(self.read(home, *PackedSiteTests.bands(*house[1:], self.fx.BUILDING_M)).max()), 0.0)
        (poly,) = clearings.outlines(home, self.E, self.N)
        self.assertGreaterEqual(poly.area, clearings.MIN_M2)
        self.assertFalse(poly.buffer(-clearings.MIN_WIDE_M / 2).is_empty)
        ClearingTests.unlike_a_cutout(self, poly.exterior.coords)
        self.assertGreaterEqual(ClearingTests.feather(home, self.E, self.N), self.fx.EDGE_M / 3)
        np.testing.assert_array_equal(self.fx.keep({}, self.E, self.N, [house], self.TEXEL), home)

    def test_a_road_the_forest_ends_at_must_cut_something_off(self):
        road = ({"highway": "primary", "ref": "A832"}, np.array([-1000.0, 1000.0]), np.array([2000.0, 2000.0]))
        with self.assertRaisesRegex(ValueError, "A832"):
            self.fx.keep({"ends_at": ("A832",)}, self.E, self.N, [road], self.TEXEL)

    def test_a_site_without_forest_rules_keeps_white_hounds_forest_as_drawn(self):
        road = ({"highway": "primary", "ref": "A832"}, np.array([-1800.0, 1800.0]), np.array([1000.0, 1000.0]))
        np.testing.assert_array_equal(self.fx.keep(None, self.E, self.N, [road], self.TEXEL), 1.0)


class CastleBaseTests(unittest.TestCase):
    """The castle's base (src/castle_base.py) on a synthetic site: level ground 70 m below the platform, the castle a
    120 by 80 m block whose south face stands 20 m from the lake's shore, and Hagrid's cabin and the pitch to the west."""
    LAKE = 100.0                                                        # any datum: only differences matter
    PLATFORM, GROUND = LAKE + 36.0, LAKE - 34.0

    @classmethod
    def setUpClass(cls):
        import castle_base

        cls.cb = castle_base
        c = (np.arange(400) - 199.5) * castle_base.TEXEL_M
        cls.E, cls.N = np.meshgrid(c, c[::-1])                          # rows from north to south
        cls.castle = (np.abs(cls.E) <= 60.0) & (np.abs(cls.N) <= 40.0)
        cls.lake_sd = cls.N + 60.0                                      # the shore runs east-west, 20 m south of it
        cls.ground = np.full(cls.E.shape, cls.GROUND)
        cls.bases = {kind: castle_base.shape(kind, cls.ground, cls.E, cls.N, cls.castle, cls.lake_sd, cls.PLATFORM,
                                             cls.LAKE, np.array([-1.0, 0.0]))[0] for kind in ("mound", "crag")}

    def grade(self, z):
        gn, ge = np.gradient(z, self.cb.TEXEL_M)
        return np.hypot(gn, ge)

    def test_a_base_only_adds_ground_and_holds_the_castle_level_on_its_platform(self):
        for kind, z in self.bases.items():
            self.assertGreaterEqual(float((z - self.ground).min()), 0.0, kind)
            np.testing.assert_array_equal(z[self.castle], self.PLATFORM, err_msg=kind)

    def test_the_mounds_lawns_can_be_walked_and_fall_most_gently_toward_the_cabin_and_pitch(self):
        g, z = self.grade(self.bases["mound"]), self.bases["mound"]
        far = (self.lake_sd > 150.0) & (z - self.ground > 0.5)          # raised, and clear of the lake front
        west = np.degrees(np.arctan2(np.abs(self.N), -self.E)) < self.cb.TOWARD_HALF_DEG["mound"] - 20.0
        self.assertLessEqual(float(g[far].max()), 0.45)                                      # "walked up or down"
        self.assertLessEqual(float(g[far & west].max()), self.cb.LAWN_GRADE["toward"] + 0.02)
        self.assertLess(float(g[far & west].max()), float(g[far & (self.E > 0.0)].max()) - 0.1)   # gentlest there

    def test_between_the_castle_and_the_lake_a_path_slope_tops_a_cliff(self):
        column = (self.N[:, 200] <= -40.0) & (self.N[:, 200] >= -62.0)    # from the south face to the water's edge
        profile = self.bases["mound"][:, 200][column]
        steps = -np.diff(profile) / self.cb.TEXEL_M
        self.assertGreaterEqual(float(profile[0] - profile[-1]), self.PLATFORM - self.LAKE - 1.0)
        self.assertGreaterEqual(float(steps.max()), 0.8 * self.cb.FACE_GRADE)              # the face, near vertical
        self.assertLessEqual(float(steps[:4].max()), self.cb.PATH_GRADE + 0.02)            # the path slope above it

    def test_the_crag_shows_rock_faces_away_from_the_cabin_but_lawns_toward_it(self):
        g, z = self.grade(self.bases["crag"]), self.bases["crag"]
        north = (self.N > 44.0) & (self.N < 100.0) & (np.abs(self.E) < 50.0)              # beside its north face
        self.assertGreaterEqual(float(g[north].max()), 0.8 * self.cb.CRAG_FACE_GRADE)
        west = np.degrees(np.arctan2(np.abs(self.N), -self.E)) < self.cb.TOWARD_HALF_DEG["crag"] - 20.0
        far = (self.lake_sd > 150.0) & (z - self.ground > 0.5)
        self.assertLessEqual(float(g[far & west].max()), self.cb.LAWN_GRADE["toward"] + 0.02)

    def test_an_unknown_base_is_refused(self):
        with self.assertRaisesRegex(ValueError, "plinth"):
            self.cb.shape("plinth", self.ground, self.E, self.N, self.castle, self.lake_sd, self.PLATFORM, self.LAKE,
                          np.array([-1.0, 0.0]))

    def test_the_vegetable_patch_lies_level_in_its_own_little_hollow_below_the_castle(self):
        """White Hound: "The vegetable patch must be either quite low-lying relative to the rest of the grounds, or in its
        own little hollow, because it can flood in heavy rain" (OotP: "splashed across the flooded vegetable patch")."""
        from scipy import ndimage as ndi

        cb = self.cb
        garden = np.hypot(self.E, self.N - 105.0) <= 35.0                               # 30 to 100 m north of it
        shed = (np.abs(self.E - 10.0) <= 6.0) & (np.abs(self.N - 105.0) <= 4.0)            # a greenhouse in it
        z, _, hollow = cb.shape("mound", self.ground, self.E, self.N, self.castle, self.lake_sd, self.PLATFORM, self.LAKE,
                                np.array([-1.0, 0.0]), garden, shed)
        floor = z[hollow & (ndi.distance_transform_edt(hollow) * cb.TEXEL_M > cb.GARDEN_RIM_M)]
        self.assertLess(float(np.ptp(floor)), 0.01)                                        # level
        pond = common.fill(z) - z                                                         # rain, run off as it would
        self.assertGreaterEqual(float(pond[hollow & (z <= floor.max())].min()), cb.GARDEN_HOLLOW_M - 0.01)   # holds it
        self.assertEqual(float(np.ptp(z[shed])), 0.0)                                     # the greenhouse stands level,
        self.assertGreaterEqual(float(z[shed].min() - floor.max()), cb.GARDEN_HOLLOW_M - 0.01)    # above the flood
        self.assertLess(float(floor.max()), self.PLATFORM - 1.0)                           # below the castle
        g = self.grade(z)
        between = (np.abs(self.E) <= 25.0) & (self.N > 44.0) & (self.N < 70.0)             # the bank up to the castle
        self.assertLessEqual(float(g[between].max()), cb.GARDEN_BANK_GRADE + 0.02)
        np.testing.assert_array_equal(z[self.castle], self.PLATFORM)


class GroundsGradeTests(unittest.TestCase):
    """grounds_grade on synthetic ground: 2 m texels, rows running south."""

    @staticmethod
    def grid(half=200.0):
        n = int(2 * half / 2.0)
        i, j = np.mgrid[0:n, 0:n]
        return j * 2.0 - half + 1.0, half - 1.0 - i * 2.0

    def test_the_drive_follows_the_road_round_a_bend_rather_than_cutting_across(self):
        import grounds_grade

        E, N = self.grid(60.0)
        road = np.zeros(E.shape)
        road[5, 5:55] = 1.0                      # east along the top row, then south down a column
        road[5:55, 54] = 1.0
        line = grounds_grade.drive_line(road, E, N, (E[5, 5], N[5, 5]), (E[54, 54], N[54, 54]))
        rows, cols = np.rint((59.0 - line[:, 1]) / 2.0).astype(int), np.rint((line[:, 0] + 59.0) / 2.0).astype(int)
        self.assertEqual(float(road[rows, cols].mean()), 1.0)

    def test_a_knoll_lifts_its_tree_with_walkable_flanks_and_meets_the_ground(self):
        import grounds_grade

        E, N = self.grid(120.0)
        F = np.full(E.shape, 5.0, np.float32)
        out = grounds_grade.knoll(F, E, N, (0.0, 0.0), 24.0)
        centre = np.unravel_index(int(np.argmin(np.hypot(E, N))), E.shape)
        self.assertAlmostEqual(float(out[centre]), 29.0, delta=0.5)
        self.assertGreaterEqual(float((out - F).min()), 0.0)              # it only ever raises the ground
        slope = np.hypot(*np.gradient(out, 2.0))
        self.assertLess(float(slope.max()), 1 / 2.6 + 0.04)               # flanks no steeper than 1 in 2.6
        self.assertLess(float((out - F)[np.hypot(E, N) > 100.0].max()), 0.1)

    def test_the_cabins_terrace_stands_above_the_gates_with_the_drive_climbing_to_it(self):
        import grounds_grade

        E, N = self.grid()
        F = (10.0 + 0.1 * np.abs(E)).astype(np.float32)        # a valley with the cabin in its bottom
        cabin, rise = (0.0, 0.0), 4.0
        drive = np.column_stack([np.arange(-190.0, 191.0, 2.0), np.zeros(191)])
        far = np.full(E.shape, 1000.0)
        out = grounds_grade.cabin_terrace(F, E, N, cabin, drive, rise, far, 0.0)
        at = lambda z, p: float(z[int(np.argmin(np.abs(N[:, 0] - p[1]))), int(np.argmin(np.abs(E[0] - p[0])))])
        gates = at(F, drive[0])
        self.assertAlmostEqual(at(out, cabin), gates + rise, delta=0.05)
        self.assertGreaterEqual(float((out - F).min()), 0.0)              # it only ever raises the ground
        z = [at(out, p) for p in drive if np.hypot(*p) >= grounds_grade.TERRACE_REACH_M - 2.0 and p[0] < 0]
        # the drive never falls on its way up, but for a few centimetres where it leaves the gates' own ground
        self.assertGreater(float(np.diff(z).min()), -0.1)
        shore = np.clip(60.0 - E, -50.0, None)                            # a lake beyond 60 m east, its surface at 0
        capped = grounds_grade.cabin_terrace(F, E, N, cabin, drive, rise, shore, 0.0)
        bank = (shore > 0.0) & (shore < 30.0)
        limit = np.maximum(F, grounds_grade.SHORE_M + grounds_grade.SHORE_GRADE * shore)
        self.assertLess(float((capped - limit)[bank].max()), 0.05)


    def test_the_station_wood_rises_behind_the_wall_and_west_of_the_walk_but_leaves_the_walk_its_ground(self):
        import grounds_grade
        import shapely

        E, N = self.grid()
        F = np.zeros(E.shape, np.float32)
        terrace = shapely.box(-50.0, -20.0, 50.0, 20.0)                  # the track along its north side
        walk = np.array([[0.0, -20.0], [0.0, -190.0]])
        road = (np.abs(E + 25.0) < 1.5) & (N < -20.0)                     # a road leaving the terrace beyond its wall
        ridge = ((-60.0, -60.0), (-60.0, -190.0))
        level, rise, crest = 10.0, 3.0, 8.0
        out, wood = grounds_grade.station_wood(F, E, N, terrace, (0.0, -1.0), (0.0, 20.0), level, rise, crest,
                                               np.ones(E.shape, bool), grounds_grade.room(E, N, walk, road),
                                               ridge=ridge)
        at = lambda z, x, n: float(z[int(np.argmin(np.abs(N[:, 0] - n))), int(np.argmin(np.abs(E[0] - x)))])
        wall = -20.0                                                      # texel centres lie on odd metres
        self.assertAlmostEqual(at(out, 41.0, wall - 19.0), level + rise * 19.0 / grounds_grade.SHELF_M, delta=0.01)
        self.assertAlmostEqual(at(out, 41.0, wall - grounds_grade.SHELF_M - 11.0),
                               level + rise - grounds_grade.SHELF_FALL * 11.0, delta=0.01)
        self.assertAlmostEqual(at(out, -59.0, -121.0), level + crest, delta=0.05)
        near = (np.abs(E) < grounds_grade.PATH_CLEAR_M - 1.0) & (N < wall - 1.0)
        self.assertEqual(float(np.abs(out - F)[near].max()), 0.0)        # the walk keeps its own ground,
        for off in (7.0, 9.0, 13.0):                                      # the shelf climbing from it no more steeply
            self.assertLessEqual(at(out, off, -61.0),                     # than its batter
                                 grounds_grade.RISE_BATTER * (off - grounds_grade.PATH_CLEAR_M) + 0.01)
        self.assertEqual(float(np.abs(out - F)[(N > 22.0) | (np.abs(E) < 48.0) & (np.abs(N) < 18.0)].max()), 0.0)
        self.assertGreaterEqual(float((out - F).min()), 0.0)              # it only ever raises the ground
        self.assertTrue(wood[int(np.argmin(np.abs(N[:, 0] + 40.0))), int(np.argmin(np.abs(E[0] - 40.0)))])
        self.assertFalse(wood[np.abs(E) < grounds_grade.PATH_CLEAR_M].any())
        self.assertEqual(float((out - F)[road].max()), 0.0)              # nor does the road
        self.assertEqual(float((out - F)[E > 50.0 + grounds_grade.SHELF_M].max()), 0.0)   # only alongside the terrace,
        self.assertLess(at(out, 61.0, -45.0), 0.5 * at(out, 41.0, -45.0))                # easing out past its end


    def test_the_trains_screen_rises_on_the_castles_side_of_the_line_and_no_further(self):
        import grounds_grade

        E, N = self.grid()
        F = np.zeros(E.shape, np.float32)
        route = np.column_stack([np.zeros(301), np.arange(-150.0, 150.5, 1.0)])     # a line running north at x = 0
        formation = np.full(301, 2.0)
        keep = np.ones(E.shape, bool)
        everywhere = np.full(E.shape, np.inf)
        out, wood = grounds_grade.train_screen(F, E, N, route, formation, (-180.0, 0.0), 8.0, keep, everywhere)
        at = lambda z, x, n: float(z[int(np.argmin(np.abs(N[:, 0] - n))), int(np.argmin(np.abs(E[0] - x)))])
        out_m = grounds_grade.SCREEN_OUT_M
        self.assertAlmostEqual(at(out, -out_m - 1.0, 1.0), 10.0, delta=0.1)          # its crest, west towards the castle
        self.assertEqual(float(out[E > 0.0].max()), 0.0)                               # nothing on the line's far side
        self.assertEqual(float(out[np.abs(N) > 152.0].max()), 0.0)                     # nor past the route's ends
        self.assertEqual(float(out[np.abs(E + out_m) > grounds_grade.SCREEN_HALF_M + 1.0].max()), 0.0)
        self.assertGreaterEqual(float((out - F).min()), 0.0)                           # it only ever raises the ground
        self.assertTrue(wood[int(np.argmin(np.abs(N[:, 0] - 1.0))), int(np.argmin(np.abs(E[0] + out_m + 1.0)))])
        self.assertFalse(wood[np.abs(E) < grounds_grade.SCREEN_KERB_M].any())       # no crown reaches the train
        held = grounds_grade.train_screen(F, E, N, route, formation, (-180.0, 0.0), 8.0, E > -out_m, everywhere)[0]
        self.assertEqual(float(held[E < -out_m].max()), 0.0)                           # and only where it may stand,
        walk = np.array([[-out_m, 150.0], [-out_m, 40.0]])                             # leaving a walk across it its ground
        eased = grounds_grade.train_screen(F, E, N, route, formation, (-180.0, 0.0), 8.0, keep,
                                           grounds_grade.room(E, N, walk))[0]
        on_walk = (np.abs(E + out_m) < grounds_grade.PATH_CLEAR_M - 1.0) & (N > 41.0)
        self.assertEqual(float(eased[on_walk].max()), 0.0)
        for off in (7.0, 9.0, 13.0):                                                   # rising no more steeply than
            self.assertLessEqual(at(eased, -out_m + off, 101.0),                       # its batter beside the walk
                                 grounds_grade.RISE_BATTER * (off - grounds_grade.PATH_CLEAR_M) + 0.01)
        self.assertAlmostEqual(at(eased, -out_m - 1.0, 1.0), 10.0, delta=0.1)          # but standing full height beyond


    def test_the_hollows_the_rises_enclose_are_made_up_where_they_may_be_and_nothing_else_changes(self):
        """Made up to the level they would spill at, but for ground that keeps its hollow. Their trees stand on as drawn
        (made_up never sees them): a meadow cut into the wood there would be a clearing no visitor could read."""
        import grounds_grade

        E, N = self.grid(40.0)
        F = np.full(E.shape, 10.0, np.float32)
        F[10:30] = 5.0                                                    # a valley the rises close, spilling at 8 m
        filled = np.maximum(F, 8.0)
        may = np.ones(E.shape, bool)
        may[10:30, 35:] = False                                           # but for ground that keeps its hollow
        ground = grounds_grade.made_up(F, filled, may)
        self.assertEqual(float(np.abs(ground[10:30, :35] - 8.0).max()), 0.0)
        rest = np.ones(E.shape, bool)
        rest[10:30, :35] = False
        self.assertEqual(float(np.abs(ground - F)[rest].max()), 0.0)


class ClearingTests(unittest.TestCase):
    """src/clearings.py on synthetic ground: 2 m texels, rows running south."""

    def glade(self):
        import clearings

        E, N = GroundsGradeTests.grid(150.0)
        core = (np.abs(N) <= 1.0) & (np.abs(E) < 60.0)         # the ground under a straight line of sight
        return clearings, E, N, core, clearings.glade(core, 2.0, 1)

    @staticmethod
    def feather(m, E, N):
        """How far, on average, a clearing's multiplier `m` takes to rise from 0.1 to 0.9 across its edge: the area
        between those two outlines over the length of the half-way one."""
        import clearings

        (lo,), (mid,), (hi,) = (clearings.outlines(m, E, N, level) for level in (0.1, 0.5, 0.9))
        return (hi.area - lo.area) / mid.exterior.length

    @staticmethod
    def unlike_a_cutout(case, ring):
        """No straight stretch of the outline is clearings.LONG_M long, and no two straight stretches run parallel and
        face each other across it."""
        import clearings

        longest = max((s[0] for s in clearings.straight_edges(ring)), default=0.0)
        case.assertLess(longest, clearings.LONG_M)
        case.assertEqual(clearings.parallel_edges(ring), [])

    def test_a_clearing_keeps_its_core_clear_wanders_at_its_edge_and_thickens_back_gradually(self):
        """No crown reaches the core; though the core is a straight line, the clearing round it is wide, its edge
        wanders, and the wood thickens back gradually rather than at a cut."""
        from scipy import ndimage as ndi

        clearings, E, N, core, m = self.glade()
        d = ndi.distance_transform_edt(~core) * 2.0
        self.assertEqual(float(m[d <= clearings.SAFE_M].max()), 0.0)
        (poly,) = clearings.outlines(m, E, N)
        self.assertGreaterEqual(poly.area, clearings.MIN_M2)
        self.assertFalse(poly.buffer(-clearings.MIN_WIDE_M / 2).is_empty)
        self.unlike_a_cutout(self, poly.exterior.coords)
        self.assertGreaterEqual(self.feather(m, E, N), clearings.FEATHER_M / 3)

    def test_a_few_trees_stand_scattered_in_a_clearing_clear_of_its_core_where_trees_may_stand(self):
        from scipy import ndimage as ndi

        clearings, E, N, core, m = self.glade()
        may = (E < -20.0) | (E > 0.0)                           # say, a track across the clearing
        trees = clearings.scattered(m, core, E, N, 2.0, 3, 2, may)
        _, count = ndi.label(trees > 0)
        self.assertEqual(count, 3)
        d = ndi.distance_transform_edt(~core) * 2.0
        self.assertGreaterEqual(float(d[trees > 0].min()), clearings.SAFE_M)
        self.assertTrue(may[trees > 0].all())

    def test_a_strips_parallel_edges_and_a_wedges_long_side_are_found_but_not_a_curve(self):
        import clearings

        strip = np.array([[0.0, 0.0], [60.0, 0.0], [60.0, 25.0], [0.0, 25.0], [0.0, 0.0]])
        self.assertEqual(len(clearings.parallel_edges(strip)), 1)
        wedge = np.array([[0.0, 0.0], [70.0, -12.0], [70.0, 12.0], [0.0, 0.0]])
        self.assertGreaterEqual(max(s[0] for s in clearings.straight_edges(wedge)), clearings.LONG_M)
        self.assertEqual(clearings.parallel_edges(wedge), [])
        turn = np.linspace(0.0, 2 * np.pi, 400)
        self.assertEqual(clearings.straight_edges(np.column_stack([40.0 * np.cos(turn), 40.0 * np.sin(turn)])), [])

    def test_trees_removed_where_no_rule_may_remove_them_are_found(self):
        import clearings

        drawn, final, accounted = np.full((3, 3), 0.8), np.full((3, 3), 0.8), np.zeros((3, 3), bool)
        final[0, 0], final[1, 1], final[2, 2] = 0.0, 0.0, 0.78
        accounted[0, 0] = True
        np.testing.assert_array_equal(np.argwhere(clearings.unaccounted(drawn, final, accounted)), [[1, 1]])

    def test_a_clearings_ground_gives_way_to_the_woods_evenly_across_its_outline(self):
        """Its own ground covers it to half ECOTONE_M inside its outline, and none lies half ECOTONE_M outside (to a
        texel). Between, its share falls evenly with the distance across the outline, as each band's mean distance has
        it, over 32 seeds, to within the rounding of a distance measured between texel centres: not all of it on one
        side of a line and none on the other, which would miss by a quarter."""
        import shapely

        clearings, E, N, _, m = self.glade()
        own = np.array([clearings.ecotone(m, 2.0, seed) for seed in range(1, 33)])
        (poly,) = clearings.outlines(m, E, N)
        across = shapely.distance(poly.exterior, shapely.points(E, N)) * np.where(m < 0.5, -1.0, 1.0)
        half = clearings.ECOTONE_M / 2
        self.assertTrue(own[:, across <= -half - 2.0].all())
        self.assertFalse(own[:, across >= half + 2.0].any())
        for lo, hi in ((-0.8, -0.2), (0.2, 0.8)):
            band = (across >= lo * half) & (across < hi * half)
            expected = float(np.mean(0.5 - across[band] / clearings.ECOTONE_M))
            self.assertAlmostEqual(float(own[:, band].mean()), expected, delta=0.08)

    @staticmethod
    def interleaved(own, other=None):
        """Whether, along each row of `own` (its own ground first, then the ground beyond its edge; `other`, by default
        all but its own), the two grounds interleave: some of its own lies beyond the first of the other's. Only rows
        holding both count."""
        other = ~own if other is None else other
        first_other = np.argmax(other, axis=1)
        last_own = own.shape[1] - 1 - np.argmax(own[:, ::-1], axis=1)
        return (last_own > first_other)[own.any(axis=1) & other.any(axis=1)]

    def test_a_clearings_ground_and_the_ground_round_it_interleave_rather_than_meet_at_a_wavy_line(self):
        """Across a straight edge, more than twice clearings.INTERLEAVE of the lines straight across it (over five
        seeds) find its ground and the ground beyond interleaved, in tufts, rather than meeting once at a line that only
        wavers, as bigger clumps would."""
        import clearings

        m = np.repeat(np.where(np.arange(40) < 20, 0.0, 1.0)[None, :], 200, axis=0)
        own = np.concatenate([clearings.ecotone(m, 2.0, seed) for seed in range(1, 6)])
        self.assertGreater(float(self.interleaved(own).mean()), 2 * clearings.INTERLEAVE)

    def test_a_patch_of_ground_blends_into_the_soft_ground_round_it_but_not_across_a_wall_nor_onto_a_path(self):
        """A patch of scrub in meadow takes in meadow beyond its edge and gives up its own to it, in tufts within half
        ECOTONE_M of its edge, as a clearing's ground does; but along a wall its edge keeps its line, and a path across
        it stays a path."""
        from scipy import ndimage as ndi

        import clearings

        E, N = GroundsGradeTests.grid(150.0)
        sc = np.zeros(E.shape, np.uint8)
        sc[(np.abs(E) < 40.0) & (np.abs(N) < 40.0)] = clearings.SCRUB
        path = np.abs(N) < 2.0
        sc[path] = 3
        wall = np.abs(E - 40.0) < 4.0
        out = clearings.blend(sc, clearings.SCRUB, 2.0, 5, (0,), wall)
        patch = sc == clearings.SCRUB
        reach = np.minimum(ndi.distance_transform_edt(patch), ndi.distance_transform_edt(~patch)) * 2.0
        self.assertTrue((out[path] == 3).all())
        self.assertTrue((out == sc)[(E >= 35.0) & (np.abs(N) < 30.0)].all())          # along the wall
        self.assertTrue((out == sc)[reach > clearings.ECOTONE_M / 2 + 2.0].all())
        self.assertGreater(int(((sc == 0) & (out == clearings.SCRUB)).sum()), 0)
        self.assertGreater(int((patch & (out == 0)).sum()), 0)

    def test_a_patch_whose_edge_meets_a_wall_at_a_slant_keeps_its_ground_along_the_wall(self):
        """Where a patch's edge meets a wall at a slant, the ground across its edge from the texels beside the wall lies
        beyond the wall's band; still none of its ground within the band gives way."""
        import clearings

        E, N = GroundsGradeTests.grid(150.0)
        sc = np.where(N < 0.3 * E, clearings.SCRUB, 0).astype(np.uint8)
        wall = np.abs(N) < 4.0
        for seed in (5, 6, 7):
            out = clearings.blend(sc, clearings.SCRUB, 2.0, seed, (0,), wall)
            with self.subTest(seed=seed):
                self.assertTrue((out == sc)[wall].all())

    def test_a_patch_of_ground_takes_in_ground_only_where_it_may(self):
        """The woods' floor spreads only under their canopy, not over the open lawn beyond it: a patch takes in soft
        ground only where `may` lets it, though it still gives up its own across its edge."""
        import clearings

        E, N = GroundsGradeTests.grid(150.0)
        sc = np.ones(E.shape, np.uint8)
        sc[(np.abs(E) < 40.0) & (np.abs(N) < 40.0)] = 5
        out = clearings.blend(sc, 5, 2.0, 6, (1,), may=E < 0.0)
        gained = (sc == 1) & (out == 5)
        self.assertEqual(int(gained[E >= 0.0].sum()), 0)
        self.assertGreater(int(gained[E < 0.0].sum()), 0)
        self.assertGreater(int(((sc == 5) & (out == 1))[E >= 0.0].sum()), 0)

    def test_a_wood_whose_trees_end_at_a_line_gives_way_within_it(self):
        """A wood whose trees stand dense to a line, as the station wood's do, has no canopy beyond that line for its
        floor to spread under. Blended inset by half clearings.ECOTONE_M, its floor takes in no ground beyond its edge and
        gives way to the ground round it in tufts across the ECOTONE_M inside it: half that band stays its own, and the
        two interleave along more than twice clearings.INTERLEAVE of the lines across it (over five seeds)."""
        import clearings

        depth = (19.5 - np.arange(40)) * 2.0              # metres inside the wood's edge, negative beyond it
        sc = np.repeat(np.where(depth > 0.0, 5, 1)[None, :], 200, axis=0).astype(np.uint8)
        own = np.concatenate([clearings.blend(sc, 5, 2.0, seed, (1,), inset_m=clearings.ECOTONE_M / 2)
                              for seed in range(1, 6)]) == 5
        self.assertFalse(own[:, depth < 0.0].any())
        self.assertTrue(own[:, depth > clearings.ECOTONE_M + 2.0].all())
        self.assertAlmostEqual(float(own[:, (depth > 0.0) & (depth < clearings.ECOTONE_M)].mean()), 0.5, delta=0.08)
        self.assertGreater(float(self.interleaved(own).mean()), 2 * clearings.INTERLEAVE)

    def test_a_wood_gives_way_from_its_open_edge_whatever_wood_it_runs_on_into(self):
        """A planting that runs on into a wood (`within`) gives way only from its open edge: inset by half
        clearings.ECOTONE_M, its floor's share rises evenly with the distance in from that edge, over ECOTONE_M (over
        five seeds), however near the wood it runs on into lies, and that wood's ground stays as it was."""
        import clearings

        x = (np.arange(60) - 9.5) * 2.0                   # metres in from the lawn's edge: 20 m of planting, then a wood
        sc = np.repeat(np.where(x < 0.0, 1, np.where(x < 20.0, 9, 5))[None, :], 200, axis=0).astype(np.uint8)
        sc[::7, 30::5] = 1                                 # the wood's own tufts of lawn
        within = np.repeat((x > 20.0)[None, :], 200, axis=0)
        outs = [clearings.blend(sc, 9, 2.0, seed, (1,), within=within, inset_m=clearings.ECOTONE_M / 2)
                for seed in range(1, 6)]
        for out in outs:
            np.testing.assert_array_equal(out[within], sc[within])
        own = np.concatenate(outs) == 9
        for lo in (0.0, 4.0, 8.0, 12.0):
            band = (x > lo) & (x < lo + 4.0)
            with self.subTest(metres_in=lo):
                self.assertAlmostEqual(float(own[:, band].mean()), float(np.mean(x[band])) / clearings.ECOTONE_M,
                                       delta=0.08)


class ForestPlacesTests(unittest.TestCase):
    """src/forest_places.py on synthetic ground: 2 m texels, rows running south."""

    def test_a_hollow_has_a_level_floor_below_its_lowest_rim_and_steep_sides_up_to_the_ground_at_its_rim(self):
        """Cut into a hillside, the hollow's floor lies forest_places.DEPTH_M below the lowest ground on its rim, level;
        its sides rise from it at SIDE_DEG, with no cliff at the rim; it only cuts, and nothing beyond its rim
        changes."""
        import forest_places as fp

        E, N = GroundsGradeTests.grid(120.0)
        slope = lambda e, n: 200.0 + 0.12 * e + 0.05 * n          # about 7 degrees, falling to the south-west
        F, at = slope(E, N), (10.0, -6.0)
        out = fp.hollow(F, E, N, at, slope)
        r = np.hypot(E - at[0], N - at[1])
        a = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
        floor = float(slope(at[0] + fp.RIM_M * np.cos(a), at[1] + fp.RIM_M * np.sin(a)).min()) - fp.DEPTH_M
        self.assertAlmostEqual(float(out[r <= fp.RIM_M].min()), floor, places=2)
        np.testing.assert_allclose(out[r < 15.0], floor, atol=1e-3)
        self.assertTrue((out <= F + 1e-4).all())
        np.testing.assert_array_equal(out[r > fp.RIM_M], F[r > fp.RIM_M])
        gn, ge = np.gradient(out, -2.0, 2.0)
        side = (r > 20.0) & (r < fp.RIM_M - 3.0) & (out > floor + 1.0)
        self.assertGreater(int(side.sum()), 100)
        self.assertAlmostEqual(float(np.median(np.hypot(ge, gn)[side])), np.tan(np.radians(fp.SIDE_DEG)), delta=0.1)
        step = max(float(np.abs(np.diff(out, axis=0)).max()), float(np.abs(np.diff(out, axis=1)).max()))
        self.assertLessEqual(step, 2.0 * np.tan(np.radians(fp.SIDE_DEG)) * 1.25)

    def test_the_bare_patch_white_hound_draws_round_his_crater_takes_the_forest_round_it(self):
        """The crater's patch, the bare texels holding his drawn spot and those within forest_places.CRATER_EDGE_M of
        them, where the layer's sampling of his map blurs its edge, takes the Forest round it, and no field tint: no
        texel of it is thinner than the Forest round it. No other ground changes, and a spot in open country, far wider
        than a crater, is refused."""
        import forest_places as fp

        E, N = GroundsGradeTests.grid(200.0)
        texture = np.where((np.arange(E.size).reshape(E.shape) % 7) == 0, 0.6, 0.85)
        r = np.hypot(E - 30.0, N + 20.0)
        t = np.clip((r - 40.0) / 8.0, 0.0, 1.0)                     # its edge blurred over 8 m
        density, field = texture * t, 1.0 - t
        density[np.hypot(E + 120.0, N - 100.0) < 30.0] = 0.0         # other bare ground
        patch = fp.crater(density, E, N, (30.0, -20.0))
        self.assertTrue(patch[r <= 50.0].all())
        self.assertFalse(patch[r >= 56.0].any())
        d, f = fp.restored(density, field, patch)
        self.assertGreaterEqual(float(d[patch].min()), 0.85)
        self.assertEqual(float(f[patch].max()), 0.0)
        np.testing.assert_array_equal(d[~patch], density[~patch])
        np.testing.assert_array_equal(f[~patch], field[~patch])
        with self.assertRaises(ValueError):
            fp.crater(np.where(E < 100.0, 0.0, 0.85), E, N, (30.0, -20.0))

    def test_grawps_clearing_keeps_a_few_small_pines_in_its_core_and_the_wood_round_it(self):
        """In its core only the small pines' density (forest_places.PINES) stands, never more than the wood drawn there;
        the wood thickens back beyond it to the density drawn."""
        import forest_places as fp

        E, N = GroundsGradeTests.grid(150.0)
        m = fp.grawp_clearing(E, N, (4.0, -2.0), 2.0)
        r = np.hypot(E - 4.0, N + 2.0)
        for drawn in (0.8, 0.05):
            out = fp.with_pines(np.full(E.shape, drawn), m)
            with self.subTest(drawn=drawn):
                np.testing.assert_allclose(out[r <= fp.GRAWP_CORE_M], min(drawn, fp.PINES))
                np.testing.assert_allclose(out[r > 120.0], drawn)
                self.assertTrue((out <= drawn + 1e-9).all())

    def test_each_place_stands_where_its_site_moves_it_or_where_white_hound_drew_it(self):
        import forest_places as fp

        for name, key in (("Aragog's hollow", "aragog_hollow"), ("Grawp's clearing", "grawp_clearing")):
            drawn = np.array(common.ff_to_world(*fp.DRAWN[name]), float)
            with self.subTest(place=name):
                np.testing.assert_allclose(fp.where(common.SITES["scardroy"], name), drawn)
                np.testing.assert_allclose(fp.where(common.SITES["x"], name), common.SITES["x"][key])
                self.assertGreater(float(np.hypot(*(np.asarray(common.SITES["x"][key]) - drawn))), 100.0)


class GroundColourTests(unittest.TestCase):
    """frag4.glsl's colours for the surface classes of the 2 m layer."""

    def test_scrub_and_an_overgrown_garden_are_dull_greens_and_browns_with_a_shrubby_texture(self):
        """The maps' scrub, which their legend calls "Rough scrubland, not suitable for farming", reads as scrub, and the
        Shack's "dank overgrown garden" (PoA 14) as rank grass, nettles and bramble: neither as heather in bloom nor as a
        stain on the meadow round them. Each colour their ground mixes is a dull green or brown: a hue from orange-brown
        to green, no more saturated than the rough meadow's. Their mean is darker than the meadow, but by no more than
        two fifths, and noise finer than 2 m, a shrub's scale, textures it."""
        import colorsys

        import clearings

        colours = class_colours()
        (meadow,), _ = colours[0]
        for cls in (clearings.SCRUB, clearings.OVERGROWN):
            mixed, frequencies = colours[cls]
            with self.subTest(surface_class=cls):
                self.assertGreaterEqual(len(mixed), 2)
                for rgb in mixed:
                    hue, saturation, _ = colorsys.rgb_to_hsv(*rgb)
                    self.assertTrue(35.0 <= hue * 360.0 <= 110.0, (rgb, hue * 360.0))
                    self.assertLessEqual(saturation, colorsys.rgb_to_hsv(*meadow)[1], rgb)
                self.assertTrue(0.6 <= float(np.mean(mixed, axis=0) @ LUMA) / float(np.asarray(meadow) @ LUMA) < 1.0)
                self.assertGreater(max(frequencies), 0.5)


class RailwayEarthworksTests(unittest.TestCase):
    """fix_railway's rules on synthetic ground: a second run must decide as the first did."""

    def test_an_embankment_across_a_valley_makes_up_the_hollow_it_encloses(self):
        import fix_railway

        old = np.full((40, 40), 10.0)
        old[15:26, 5:] = 5.0 + 0.02 * (39 - np.arange(5, 40))       # a valley draining out of the east edge
        old[2:5, 2:5] = 9.0                                          # and a natural pit, in the hills
        cut, fill = np.full(old.shape, np.inf), np.full(old.shape, -np.inf)
        fill[15:26, 30:33] = 8.0                                     # an embankment across the valley
        new, hollow = fix_railway.graded(old, cut, fill, np.zeros(old.shape, bool), 0.02)
        self.assertEqual(float(new[15:26, 30:33].min()), 8.0)
        self.assertEqual(float(np.abs(new[15:26, 5:30] - 8.0).max()), 0.0)       # made up to where it spills
        self.assertTrue(hollow[15:26, 5:30].all())
        self.assertEqual(float(new[2:5, 2:5].max()), 9.0)                          # the natural pit is left alone
        self.assertLess(float(np.abs(new - old)[15:26, 33:].max()), 1e-9)         # and the ground below the dam

    def test_the_pine_bank_rule_decides_the_same_on_the_ground_it_raised(self):
        import fix_railway

        quantum = 0.02
        fill = np.full(3, 10.011)                       # rounds up by 0.009 m to the height quantum
        cut = fill + np.array([0.055, 0.15, 0.3])
        old, pines, onbed = np.full(3, 9.0), np.ones(3, bool), np.zeros(3, bool)
        first = fix_railway.kept_pines(pines, cut, fill, onbed, cut < old + fix_railway.TOL_M)
        raised = np.round(np.maximum(old, fill) / quantum) * quantum
        second = fix_railway.kept_pines(pines, cut, fill, onbed, cut < raised + fix_railway.TOL_M)
        self.assertEqual(first.tolist(), second.tolist())
        self.assertTrue(first[2])                       # pines on a true embankment slope are kept


class PackedSiteTests(unittest.TestCase):
    """The committed build/data.js, checked against the configuration of the site it shows."""

    @classmethod
    def setUpClass(cls):
        import mesh_source

        cls.ms = mesh_source
        cls.data = mesh_source.load_data()
        cls.meta = cls.data["meta"]
        cls.site = common.SITES[common.site_name(ROOT / "absent-stage.json")]
        cls.grid = mesh_source.Grid(cls.meta["local"])

    def layer(self, name):
        return self.ms.decode(self.data["tex"][name])

    def imagery(self, name):
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(base64.b64decode(self.data["tex"][name]["data"]))))

    def osm_courses(self):
        """The carved OpenStreetMap watercourses through the local frame, each as castle-relative (E, N) every 2 m."""
        source = "site-osm-x" if self.site is common.SITES["x"] else "original-intermediates"
        osm = json.loads((inputs.require(source) / "osm.json").read_text(encoding="utf-8"))
        for el in osm["elements"]:
            if el["type"] != "way" or el.get("tags", {}).get("waterway") not in self.site["carve"]:
                continue
            g = el["geometry"]
            e, n = common.to_tm.transform([p["lon"] for p in g], [p["lat"] for p in g])
            e, n = np.asarray(e) - self.site["anchor"][0], np.asarray(n) - self.site["anchor"][1]
            s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(e), np.diff(n)))])
            t = np.arange(0.0, s[-1], 2.0)
            yield el["id"], np.interp(t, s, e), np.interp(t, s, n)

    def test_every_layer_has_its_grids_shape(self):
        import build_regional

        shapes = {"loc": (self.grid.ny, self.grid.nx), "cas": (self.meta["castle"]["ny"], self.meta["castle"]["nx"]),
                  "forest": (self.meta["forest"]["ny"], self.meta["forest"]["nx"])}
        for level, (R, st) in build_regional.LEVELS.items():
            n = build_regional.grid(R, st)[2]
            shapes[level] = (n, n)
        for name, packed in self.data["tex"].items():
            grid = next(prefix for prefix in sorted(shapes, key=len, reverse=True) if name.startswith(prefix))
            self.assertEqual((packed["h"], packed["w"]), shapes[grid], name)

    def test_the_far_layer_stays_well_below_the_ceiling(self):
        ceiling = float(re.search(r"#define CEIL ([\d.]+)", PageTests.FRAG).group(1))
        self.assertLess(float((self.layer("BBed") + np.maximum(self.layer("BW"), 0.0)).max()), ceiling - 40.0)

    def test_no_imagery_tile_is_missing(self):
        from scipy import ndimage as ndi

        for name in ("M", "A", "B"):        # a missing tile is a black block on land; dark water is fine
            labels, n = ndi.label((self.imagery(name + "Sat").max(axis=-1) <= 3) & (self.layer(name + "W") <= 0.0))
            self.assertLess(int(np.bincount(labels.ravel())[1:].max(initial=0)), 100, name)

    def test_the_lake_lies_at_the_sites_level(self):
        from scipy import ndimage as ndi

        self.assertEqual(self.meta["lake"], self.site["lake"])
        bed, w = self.layer("locBed"), self.layer("locW")
        labels, _ = ndi.label(w > 0.5)
        lake = labels == np.argmax(np.bincount(labels[labels > 0]))
        np.testing.assert_allclose((bed + w)[lake], self.site["lake"], atol=0.021)

    def test_the_castle_stands_on_its_platform(self):
        towers = np.asarray(self.meta["towers"])
        base = self.ms.sample(self.layer("locCB"), self.grid, towers[:, 0], towers[:, 1])
        self.assertGreaterEqual(float(base.min()), self.site["lake"] + 36.0 - 0.02)

    def test_the_regional_water_is_sane(self):
        for level in ("M", "A", "B"):
            bed, w = self.layer(level + "Bed"), self.layer(level + "W")
            wet = w > 0.0
            self.assertTrue(0.01 < wet.mean() < 0.1, level)
            sea_level = -self.data["tex"][level + "Bed"]["q"] / 2.0 - 0.02       # within the bed's height quantum
            self.assertGreaterEqual(float((bed + w)[wet].min()), sea_level, level)

    def test_every_configured_place_is_found(self):
        found = set(self.meta["extras"]["real"])
        for spellings in self.site["names"]:
            self.assertTrue(found & set(spellings), spellings)

    def test_the_page_names_the_sites_glen(self):
        self.assertEqual(self.meta["extras"]["glen"], self.site["glen"])

    def test_the_page_credits_the_elevation_model_the_data_was_built_from(self):
        self.assertEqual(self.meta["extras"]["dem"], self.site["dem"])

    def test_the_lake_is_held_by_its_banks(self):
        from scipy import ndimage as ndi

        bed, w = self.layer("locBed"), self.layer("locW")
        lake = (w > 0.0) & (np.abs(bed + w - self.site["lake"]) < 0.05)
        beside = ndi.binary_dilation(lake, structure=np.ones((3, 3), bool)) & (w <= 0.0)
        low = beside & (bed < self.site["lake"] - 0.5)          # a dry texel below the water beside it: a wall of water
        where = [[round(float(v)) for v in self.grid.to_world(i, j)] for i, j in np.argwhere(low)[:5]]
        self.assertEqual((int(low.sum()), where), (0, []))

    def test_no_stream_stands_above_the_lake_on_the_ground_raised_round_the_castle(self):
        """White Hound's outflow runs "from the lake north and down through the forest": where the castle's base buries
        its first stretch it runs under the base and comes out at its foot, never perched above the lake."""
        if not self.site["graded"]:
            self.skipTest("the site keeps the original carve")
        bed, w = self.layer("locBed"), self.layer("locW")
        stream = self.ms.sd_bytes(self.layer("locB")[..., 1])
        i, j = np.nonzero((stream < 0.0) & (w > 0.0))
        E, N = self.grid.to_world(i, j)
        natural = terrain.heights(self.site["dem"], E + self.site["anchor"][0], N + self.site["anchor"][1])
        perched = (bed[i, j] - natural > 3.0) & ((bed + w)[i, j] > self.site["lake"]) & (np.hypot(E, N) < 400.0)
        where = [[round(float(e)), round(float(n))] for e, n in zip(E[perched][:5], N[perched][:5])]
        self.assertEqual((int(perched.sum()), where), (0, []))

    def test_trees_grow_over_the_buried_outflow_as_round_it(self):
        """Where the castle's base buries the lake's outflow it runs under the base, so the grounds map's water ink there
        draws no stream above ground and leaves no treeless strip through the woods round it."""
        if not self.site["graded"]:
            self.skipTest("the site keeps the original carve")
        from scipy import ndimage as ndi
        import build_final

        water = np.load(inputs.require("original-intermediates") / "build" / "cls_g.npy") == common.C_WATER
        lab, _ = ndi.label(water)
        lake = lab == 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
        v, u = np.nonzero(build_final.drop_small(ndi.binary_opening(water & ~lake, iterations=1), 80))   # its streams
        E, N = (np.asarray(c, float) for c in build_final.g2w(u + 0.5, v + 0.5))
        near = (np.abs(E) < 1000.0) & (np.abs(N) < 1000.0)
        E, N = E[near], N[near]
        at = lambda field, e, n: self.ms.sample(field, self.grid, e, n)
        trees = self.layer("locA")[..., 1] / 255.0
        stream = self.ms.sd_bytes(self.layer("locB")[..., 1])
        shore = ndi.distance_transform_edt(self.layer("locW") <= 0.0) * self.grid.texel_size()[0]
        buried = (at(stream, E, N) >= 1.0) & (at(shore, E, N) > 6.0)        # a stream the build no longer draws there
        ring = np.mean([at(trees, E + 8.0 * np.cos(a), N + 8.0 * np.sin(a)) for a in np.arange(8) * np.pi / 4], axis=0)
        wooded = buried & (ring > 0.1)
        bare = wooded & (at(trees, E, N) < 0.05)
        self.assertGreater(int(wooded.sum()), 300)
        self.assertLess(float(bare.sum() / wooded.sum()), 0.2)

    def grounds_points(self):
        """Hagrid's cabin, the gates' centre and the drive from them to the castle's front, in castle-relative metres."""
        import grounds_grade

        scene = json.loads((ROOT / "src" / "mesh_scene.json").read_text(encoding="utf-8"))
        cabin = np.array(next(a for a in scene["assets"] if a["id"] == "hut")["components"][0]["at"], float)
        gates = np.mean(np.array(self.meta["extras"]["gates"], float), axis=0)
        named = json.loads((inputs.require("original-intermediates") / "build" / "castle_named.json").read_text(
            encoding="utf-8"))
        i, j = np.mgrid[0:self.grid.ny, 0:self.grid.nx]
        E, N = self.grid.to_world(i, j)
        road = self.layer("locA")[..., 0] / 255.0
        drive = grounds_grade.drive_line(road, E, N, gates, named["EH"])
        return cabin, gates, drive, road

    def test_hagrids_cabin_stands_above_the_gates_and_the_lake(self):
        """White Hound: "The front gates are lower down than Hagrid's cabin." His page quotes the books: "up through the
        grounds to Hagrid's cabin" from the gates (GoF 24), and Harry "scrambling up the grassy slope" from the lake."""
        if not self.site["cabin_terrace"]:
            self.skipTest("the site's cabin keeps its own ground")
        cabin, gates, _, _ = self.grounds_points()
        at = lambda p: float(self.ms.sample(self.layer("locBed"), self.grid, *p))
        self.assertGreater(at(cabin) - at(gates), 3.0)
        self.assertGreater(at(cabin) - self.site["lake"], 15.0)

    def test_the_drive_climbs_from_the_gates_to_hagrids_cabin(self):
        """Walking from the gates to the cabin is "up through the grounds" all the way, never down into a hollow first."""
        if not self.site["cabin_terrace"]:
            self.skipTest("the site's cabin keeps its own ground")
        import grounds_grade

        cabin, gates, drive, _ = self.grounds_points()
        reach = np.hypot(*(drive - cabin).T) <= grounds_grade.TERRACE_REACH_M
        upto = drive[:int(np.argmax(reach)) + 1]
        z = self.ms.sample(self.layer("locBed"), self.grid, upto[:, 0], upto[:, 1])
        self.assertGreater(len(upto), 50)
        self.assertGreater(float(z.min() - z[0]), -0.5)
        self.assertGreater(float(z[-1] - z[0]), 3.0)

    def test_the_lawns_fall_from_the_castles_front_to_hagrids_cabin(self):
        """The books: "they went down the sloping lawns to Hagrid's hut" (PoA 6), and from the foot of the castle's steps
        the Gryffindors head "down the sloping lawn towards Hagrid's small wooden cabin" (GoF 13). Raised on its terrace,
        the cabin must not be reached by climbing out of a hollow below it."""
        if not self.site["cabin_terrace"]:
            self.skipTest("the site's cabin keeps its own ground")
        cabin, _, drive, road = self.grounds_points()
        on_road = self.ms.sample(road, self.grid, drive[:, 0], drive[:, 1]) > 0.3
        front = drive[int(np.nonzero(on_road)[0][-1])]                   # where the drive meets the castle's front
        bed = self.layer("locBed")
        level = float(self.ms.sample(bed, self.grid, *cabin))
        L = float(np.hypot(*(front - cabin)))
        across = np.array([-(front - cabin)[1], (front - cabin)[0]]) / L
        lowest = {}
        for off in (-15.0, 0.0, 15.0):
            line = cabin + off * across + np.outer(np.arange(0.0, L, 2.0) / L, front - cabin)
            lowest[off] = round(float(self.ms.sample(bed, self.grid, line[:, 0], line[:, 1]).min()) - level, 1)
        self.assertTrue(all(v > -1.0 for v in lowest.values()), lowest)

    def test_white_hounds_sight_lines_clear_the_ground(self):
        """White Hound: "There is a clear line of sight from Hagrid's cabin to the castle doors, from at or near the front
        gates to Hagrid's cabin and from the castle doors to the front gates"."""
        if not self.site["cabin_terrace"]:
            self.skipTest("the site's cabin keeps its own ground")
        cabin, gates, drive, road = self.grounds_points()
        on_road = self.ms.sample(road, self.grid, drive[:, 0], drive[:, 1]) > 0.3
        doors = drive[int(np.nonzero(on_road)[0][-1])]                  # where the drive meets the castle's front
        bed = self.layer("locBed")

        def clear(a, ha, b, hb):
            L = float(np.hypot(*(b - a)))
            u = np.arange(6.0, L - 6.0, 1.0) / L
            za, zb = (float(self.ms.sample(bed, self.grid, *p)) + h for p, h in ((a, ha), (b, hb)))
            ground = self.ms.sample(bed, self.grid, *(a + np.outer(u, b - a)).T)
            return round(float((za + u * (zb - za) - ground).min()), 1)
        seen = {"cabin to doors": clear(cabin, 2.0, doors, 1.7), "gates to cabin": clear(gates, 2.5, cabin, 2.0),
                "doors to gates": clear(doors, 1.7, gates, 2.0)}
        self.assertTrue(all(v > 0.0 for v in seen.values()), seen)

    def willow_and_greenhouses(self):
        """The Whomping Willow, the northern greenhouses and the main group (5b), in castle-relative metres."""
        scene = json.loads((ROOT / "src" / "mesh_scene.json").read_text(encoding="utf-8"))
        group = lambda name: np.array([c["at"] for c in next(a for a in scene["assets"] if a["id"] == name)["components"]],
                                      float)
        return np.array(self.meta["extras"]["willow"], float), group("greenhouses-north"), group("greenhouses-5b")

    def test_the_whomping_willow_stands_on_its_own_knoll(self):
        """The Willow "stood alone in the middle of the grounds" (PoA 9); at X its ground lay 10 m below the lake, its
        crown hidden from every greenhouse, so it stands on a knoll of its own."""
        if not self.site["willow_knoll"]:
            self.skipTest("the site's Willow keeps its own ground")
        willow, _, _ = self.willow_and_greenhouses()
        bed = self.layer("locBed")
        turn = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
        ring = self.ms.sample(bed, self.grid, willow[0] + 60.0 * np.cos(turn), willow[1] + 60.0 * np.sin(turn))
        top = float(self.ms.sample(bed, self.grid, *willow))
        self.assertGreater(top - self.site["lake"], 10.0)
        self.assertGreater(top - float(ring.mean()), 8.0)

    def test_from_the_greenhouses_the_willows_crown_shows_but_not_a_person_at_its_foot(self):
        """White Hound: "From the greenhouses it is possible to see at least the upper part of the Whomping Willow", "yet
        not possible to see a person standing at its base"; and "you wouldn't expect there to be other trees near it to
        obscure it from view", so the lawn opened to it from the northern greenhouses leaves no wood on their line."""
        import assets_testing

        if not self.site["willow_lawn"]:
            self.skipTest("the site keeps the grounds' trees as drawn")
        willow, north, main = self.willow_and_greenhouses()
        bed, trees = self.layer("locBed"), self.layer("locA")[..., 1] / 255.0
        at = lambda field, p: self.ms.sample(field, self.grid, p[..., 0], p[..., 1])
        foot = float(at(bed, willow))
        pos = np.concatenate([c["pos"] for c in assets_testing.payload()["chunks"] if c["asset"] == "willow"])
        top = float(pos[:, 1].max())

        def clear(a, height):                   # over the ground, short of the knoll's own rounded top
            L = float(np.hypot(*(willow - a)))
            u = np.arange(6.0, L - 30.0, 1.0) / L
            za = float(at(bed, a)) + 1.7
            return float((za + u * (foot + height - za) - at(bed, a + np.outer(u, willow - a))).min())
        crown = [clear(a, top - foot) for a in north]          # the top of the Willow's own mesh
        person = [clear(a, 1.8) for a in np.vstack([north, main])]
        self.assertGreaterEqual(sum(c > 0.0 for c in crown), 4, np.round(crown, 1).tolist())
        self.assertLess(max(person), 0.0)
        centre = north.mean(axis=0)
        L = float(np.hypot(*(willow - centre)))
        axis = centre + np.outer(np.arange(10.0, L - 20.0, 2.0) / L, willow - centre)
        across = np.array([-(willow - centre)[1], (willow - centre)[0]]) / L
        wood = max(float(at(trees, axis + off * across).max()) for off in (-8.0, 0.0, 8.0))
        self.assertLess(wood, 0.05)

    def test_the_trees_hide_little_of_the_willows_crown_that_the_ground_leaves_in_sight_of_each_greenhouse(self):
        """White Hound: "From the greenhouses it is possible to see at least the upper part of the Whomping Willow", and
        "you wouldn't expect there to be other trees near it to obscure it from view": from each northern greenhouse,
        past the renderer's trees, at least four fifths of the points on the top quarter of the Willow's own mesh that
        the ground leaves in sight still show."""
        import assets_testing

        if not self.site["willow_lawn"]:
            self.skipTest("the site keeps the grounds' trees as drawn")
        willow, north, _ = self.willow_and_greenhouses()
        bed = self.layer("locBed")
        foot = float(self.ms.sample(bed, self.grid, *willow))
        pos = np.concatenate([c["pos"] for c in assets_testing.payload()["chunks"] if c["asset"] == "willow"])
        upper = pos[pos[:, 1] > foot + 0.75 * (float(pos[:, 1].max()) - foot)]
        crown = upper[np.random.default_rng(0).choice(len(upper), 60, False)]
        eyes = np.array([(p[0], float(self.ms.sample(bed, self.grid, *p)) + 1.7, p[1]) for p in north])
        lines = self.sight_lines(eyes, crown)
        seen = [lines.seen(e, crown, short_m=6.0) for e in eyes]       # up to the crown, not short of a castle's mound
        lines.trees["H"][:] = np.nan
        ground = [lines.seen(e, crown, short_m=6.0) for e in eyes]
        self.assertTrue(all(s >= 0.8 * g for s, g in zip(seen, ground)), (seen, ground))

    def test_the_tunnel_portal_stands_where_the_hill_first_covers_the_tunnel(self):
        """White Hound's line is hidden from the station "once it has rounded the bend", into the tunnel: its portal
        stands where the ground before the railway's earthworks first rises a headwall and a margin over the formation,
        across the headwall's width, not on low ground nor at the end of a needless cutting."""
        if not self.site["portal_m"]:
            self.skipTest("the site keeps the original portal")
        import railway
        from tools.site import build

        before = self.ms.load_data(str(inputs.require(build.S0_INPUTS[common.site_name(ROOT / "absent-stage.json")])))
        grid, bed = self.ms.Grid(before["meta"]["local"]), self.ms.decode(before["tex"]["locBed"])
        al = railway.alignment(json.loads((ROOT / "src" / "mesh_scene.json").read_text(encoding="utf-8")), before["meta"],
                               self.ms.load_georef())
        half = railway.FORMATION_M + railway.HEADWALL_M / railway.CUT
        need = railway.HEADWALL_M + 2.0

        def cover(s):           # the least height of the ground over the formation across the headwall's width
            p, t = railway.point_at(al.main.xz, s), railway.tangent_at(al.main.xz, s)
            q = p + np.outer(np.linspace(-half, half, 9), [-t[1], t[0]])
            return float(self.ms.sample(bed, grid, q[:, 0], q[:, 1]).min()) - (
                al.level + railway.GRADE * max(s - railway.TURNOUT_M, 0.0))
        self.assertEqual(al.portal_s, self.site["portal_m"])
        self.assertGreaterEqual(min(cover(al.portal_s + a) for a in (0.0, 3.0, 6.0)), need)
        self.assertLess(cover(al.portal_s - 10.0), need)

    def test_trees_flank_the_first_years_path_down_to_the_lake(self):
        """PS 6: "they followed Hagrid down what seemed to be a steep, narrow path. It was so dark either side of them that
        Harry thought there must be thick trees there." Where the path falls from the station terrace towards the lake,
        White Hound's pines stand beside it, within 5 m: the wooded rises' trees, standing further off, do not count."""
        import railway

        al = railway.alignment(json.loads((ROOT / "src" / "mesh_scene.json").read_text(encoding="utf-8")), self.meta,
                               self.ms.load_georef())
        trees = self.layer("locA")[..., 1] / 255.0
        xz = al.path.xz
        down = self.ms.sample(self.layer("locBed"), self.grid, xz[:, 0], xz[:, 1]) < al.level - 0.5
        flanked = []
        for p, s in zip(xz[down], al.path.s[down]):
            t = railway.tangent_at(xz, s)
            n = np.array([-t[1], t[0]])
            sides = [max(float(self.ms.sample(trees, self.grid, *(p + side * o * n))) for o in (3.0, 5.0))
                     for side in (-1.0, 1.0)]
            flanked.append(max(sides) > 0.3)
        self.assertGreater(len(flanked), 20)
        self.assertGreaterEqual(float(np.mean(flanked)), 0.8)

    def railway_and_castle(self):
        """(the railway's alignment, the walk from the platform to the lake, 150 points on the castle's upper parts)."""
        import assets_testing
        import railway

        scene, georef = json.loads((ROOT / "src" / "mesh_scene.json").read_text(encoding="utf-8")), self.ms.load_georef()
        al = railway.alignment(scene, self.meta, georef)
        pos = np.concatenate([c["pos"] for c in assets_testing.payload()["chunks"] if c["asset"] == "castle"])
        upper = pos[pos[:, 1] > self.meta["lake"] + 36.0 + 10.0]
        return al, railway.lake_walk(al, scene, georef), upper[np.random.default_rng(0).choice(len(upper), 150, False)]

    def sight_lines(self, eyes, targets):
        points = np.vstack([np.asarray(eyes)[:, [0, 2]], targets[:, [0, 2]]])
        return SightLines(self.ms, self.data, points.min(axis=0) - 10.0, points.max(axis=0) + 10.0)

    def test_the_castle_is_first_seen_where_the_walk_to_the_lake_opens_on_to_it(self):
        """PS 6: "'Yeh'll get yer firs' sight o' Hogwarts in a sec,' Hagrid called over his shoulder, 'jus' round this bend
        here.' ... The narrow path had opened suddenly on to the edge of a great black lake." From the platform and all
        along the walk down to the lake, short of its last bend, at most a glimpse of the castle shows through the trees;
        where the walk opens on to the lake, the castle stands in full view."""
        import mesh_transport

        al, walk, targets = self.railway_and_castle()
        origin, along, west = al.frame
        top = al.level + mesh_transport.RAIL_TOP_M + mesh_transport.PLATFORM_HEIGHT_M + 1.6
        platform = [(*(origin + t * along + (mesh_transport.PLATFORM_BACK_M - 1.0) * west), top)
                    for t in np.arange(al.platform[0] + 5.0, al.platform[1] - 4.0, 15.0)]
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(walk, axis=0).T))])
        bed = self.layer("locBed")
        ground = lambda p: float(self.ms.sample(bed, self.grid, p[0], p[1])) + 1.6
        path = [(*walk[int(np.searchsorted(s, d))], ground(walk[int(np.searchsorted(s, d))]))
                for d in np.arange(8.0, s[-1] - 40.0, 12.0)]                 # its last 28 m run west from the bend
        eyes = np.array([(x, y, z) for x, z, y in platform + path + [(*walk[-1], ground(walk[-1]))]])
        lines = self.sight_lines(eyes, targets)
        seen = [lines.seen(eye, targets) for eye in eyes]
        self.assertLessEqual(max(seen[:-1]), GLIMPSE * len(targets), seen)
        self.assertGreaterEqual(seen[-1], 0.3 * len(targets))

    def test_beside_the_walk_to_the_lake_no_rise_climbs_more_steeply_than_the_railways_rock_cuttings(self):
        """The wooded rises keeping PS 6's first sight of the castle for the walk's end leave the walk its own ground and
        stand no wall beside it: within 20 m of the walk the ground nowhere climbs more steeply than the railway's rock
        cuttings."""
        import grounds_grade
        import railway

        if not (self.site.get("station_wood") or self.site.get("train_screen")):
            self.skipTest("this site raises no ground beside the walk")
        scene, georef = json.loads((ROOT / "src" / "mesh_scene.json").read_text(encoding="utf-8")), self.ms.load_georef()
        walk = grounds_grade.resampled(railway.lake_walk(railway.alignment(scene, self.meta, georef), scene, georef))
        rows, cols = self.grid.to_texel(walk[:, 0], walk[:, 1])
        across, down = self.grid.texel_size()
        pad = int(np.ceil(25.0 / min(across, down)))
        i0, j0 = max(int(rows.min()) - pad, 0), max(int(cols.min()) - pad, 0)
        i, j = np.mgrid[i0:min(int(rows.max()) + pad + 1, self.grid.ny), j0:min(int(cols.max()) + pad + 1, self.grid.nx)]
        slope = np.hypot(*np.gradient(self.layer("locBed")[i, j].astype(float), down, across))
        E, N = self.grid.to_world(i, j)
        near = grounds_grade.line_distance(E, N, walk)[0] <= 20.0
        worst = np.argmax(np.where(near, slope, -1.0))
        self.assertLessEqual(float(slope.flat[worst]), railway.CUT + 0.1, (float(E.flat[worst]), float(N.flat[worst])))

    def test_the_castle_is_hidden_from_the_trains_last_approach(self):
        """White Hound: "Since this is the first glimpse the students get of the castle there must be a barrier \u2013 wall,
        trees or rising land \u2013 preventing the castle from being seen from the train as it approaches the station."
        From a seated and a standing passenger's eyes at the carriages' windows, from the tunnel's portal down the main
        line, through the junction and along the loop to the platform's end, at most a glimpse of the castle shows."""
        import mesh_transport

        al, _, targets = self.railway_and_castle()
        for eye in (SEATED_EYE_M, STANDING_EYE_M):
            up = mesh_transport.RAIL_TOP_M + eye
            main = [(*railway_point(al.main, s), up) for s in np.arange(al.portal_s, -0.1, -15.0)]
            loop = [(*railway_point(al.loop, s), up) for s in np.arange(15.0, al.coaches[0], 15.0)]
            eyes = np.array([(x, level + u, z) for x, z, level, u in main + loop])
            lines = self.sight_lines(eyes, targets)
            seen = [lines.seen(e, targets) for e in eyes]
            with self.subTest(eye=eye):
                self.assertLessEqual(max(seen), GLIMPSE * len(targets), seen)

    def test_no_trees_crown_reaches_the_train(self):
        """No tree the renderer could grow, whatever its cell's hash, comes with its crown within a metre of a coach's
        side all along the open line: the loop and the main line up to its tunnel. frag1.glsl stands a cell's tree
        anywhere in the middle half of its 7 m cell where the density, read bilinearly up to a texel diagonal into
        cleared ground, exceeds the hash, and its crown spreads up to TREE_CROWN_M."""
        import mesh_transport
        from scipy.spatial import cKDTree

        al, _, _ = self.railway_and_castle()
        line = np.concatenate([al.loop.xz, al.main.xz[al.main.s <= al.portal_s]])
        reach = mesh_transport.COACH_W / 2 + 1.0 + common.TREE_CROWN_M
        lo, hi = line.min(axis=0) - reach, line.max(axis=0) + reach
        x, z = np.meshgrid(np.arange(lo[0], hi[0], 0.25), np.arange(lo[1], hi[1], 0.25))
        p = np.column_stack([x.ravel(), z.ravel()])
        near = cKDTree(line).query(p, distance_upper_bound=reach)[0] < reach
        spot = p[near][np.all(np.abs(p[near] / SightLines.CELL_M % 1.0 - 0.5) < 0.25, axis=1)]
        grown = SightLines(self.ms, self.data, lo, hi).density(spot) > 0.0
        self.assertFalse(grown.any(), spot[grown][:5])

    def test_the_railway_clears_every_tree_within_reach_of_the_open_line(self):
        """fix_railway clears the open line's lineside whatever grows there: run again on this build, wooded all over, it
        leaves no tree within LINE_CLEAR_M of the loop or of the main line up to its portal, and the wood beyond."""
        import fix_railway
        import railway
        from scipy.spatial import cKDTree

        la, lb, ln = (self.layer(k) for k in ("locA", "locB", "locN"))
        scene, georef = json.loads((ROOT / "src" / "mesh_scene.json").read_text(encoding="utf-8")), self.ms.load_georef()
        wooded = np.full(la.shape[:2], 255, la.dtype)
        bed, _, trees, *_ = fix_railway.fix_railway(scene, self.meta, georef, self.layer("locBed"), self.layer("locW"),
                                                    wooded, ln[..., 0], lb[..., self.ms.CHANNELS["locB"].index("stream")],
                                                    self.data["tex"]["locBed"]["q"])
        self.assertEqual(float(np.abs(bed - self.layer("locBed")).max()), 0.0)        # the works stand as built
        al = railway.alignment(scene, self.meta, georef)
        rows, cols = fix_railway.window(self.grid, al)
        x, z = self.grid.to_world(*np.mgrid[rows, cols])
        line = np.concatenate([al.loop.xz, al.main.xz[al.main.s <= al.portal_s]])
        d = cKDTree(line).query(np.column_stack([x.ravel(), z.ravel()]))[0].reshape(x.shape)
        held = trees[rows, cols]
        self.assertEqual(int(held[d < fix_railway.LINE_CLEAR_M - 1.0].max()), 0)
        self.assertGreater(float((held[d > 20.0] > 0).mean()), 0.9)

    def test_the_castle_shows_through_the_shacks_one_unblocked_window(self):
        """DH 32: "all the windows boarded except for one", and "The single unblocked window revealed distant bursts of
        light where the castle stood". From that window, glazed on the Shack's ground floor, most of the points on the
        castle's upper parts are in sight, past the trees on the Shack's hill."""
        import assets_testing
        import mesh_village
        from materials import code

        if not self.site["shack_garden"]:
            self.skipTest("the site keeps the trees on the Shack's hill as drawn")
        tris, attr, _ = assets_testing.payload_part("shrieking-shack")
        glazed = np.any([(attr == code(g)).all(axis=1) for g in mesh_village.GLASS], axis=0)
        window = tris[glazed].reshape(-1, 3).mean(axis=0)
        _, _, targets = self.railway_and_castle()
        seen = self.sight_lines(window[None, :], targets).seen(window, targets)
        self.assertGreater(seen, len(targets) / 2)

    def test_every_clearing_is_large_and_wide_with_no_long_straight_or_parallel_edges(self):
        """The user: "make sure that the clearings, wherever they are, are large enough to be plausible and not look
        suspiciously like cutouts created just for line of sight purposes". Each clearing the build records, every one
        that follows no linear feature (clearings.LINEAR lists those), is at least clearings.MIN_M2 in area and
        MIN_WIDE_M across, with no straight stretch of its edge LONG_M long and no two that run parallel and face each
        other across it."""
        import clearings

        records = self.meta.get("clearings", [])
        configured = {"willow-lawn": self.site["willow_lawn"], "shack-garden": self.site["shack_garden"],
                      "homestead": self.site.get("forest"), "aragogs-hollow": self.site["aragog_hollow"],
                      "grawps-clearing": self.site["grawp_clearing"]}
        self.assertEqual({r["id"] for r in records}, {k for k, on in configured.items() if on})
        for r in records:
            poly = clearings.polygon(r)
            with self.subTest(clearing=r["id"], at=np.round(poly.centroid.coords[0]).tolist()):
                self.assertGreaterEqual(poly.area, clearings.MIN_M2)
                self.assertFalse(poly.buffer(-clearings.MIN_WIDE_M / 2).is_empty)
                ClearingTests.unlike_a_cutout(self, r["outline"])

    def test_in_each_clearing_a_few_trees_stand_and_the_wood_thins_towards_its_edge(self):
        """In each clearing of the local layer at least two trees stand scattered, where the renderer grows a tree at
        full density; and where its edge cuts through wood along 40 m or more, as the Willow's lawn does, the trees thin
        out towards it over a third of clearings.FEATHER_M or more (along the edge's normals, from a tenth to nine tenths
        of their density beyond it), not stopping at a cut, which the bilinear read spreads over a texel."""
        import clearings
        import shapely

        trees = self.layer("locA")[..., 1] / 255.0
        F = clearings.FEATHER_M
        out = np.arange(-F, 1.5 * F + 0.5, 1.0)

        def rise(p, top):                       # from the last point below a tenth to the first at nine tenths
            full = int(np.argmax(p >= 0.9 * top))
            low = np.nonzero(p[:full] < 0.1 * top)[0]
            return out[full] - out[low[-1] + 1] if len(low) else np.nan
        thinned = []
        for r in (r for r in self.meta.get("clearings", []) if r["layer"] == "local"):
            ring = np.asarray(r["outline"], float)
            poly = clearings.polygon(r)
            lines = SightLines(self.ms, self.data, ring.min(axis=0), ring.max(axis=0))
            ctr = lines.trees["ctr"]
            standing = ~np.isnan(lines.trees["H"]) & shapely.contains_xy(poly, *ctr.T) & (lines.density(ctr) >= 0.9)
            pts, normal = self.across_edge(ring)
            q = pts[:, None, :] + out[None, :, None] * normal[:, None, :]
            profile = self.ms.sample(trees, self.grid, q[..., 0], q[..., 1])
            top = profile[:, -1]
            cut = (top >= 0.25) & (profile[:, int(np.argmin(np.abs(out)))] >= 0.25 * top)   # the edge in wood
            rises = np.array([rise(p, t) for p, t in zip(profile[cut], top[cut])])
            rises = rises[~np.isnan(rises)]
            with self.subTest(clearing=r["id"]):
                self.assertGreaterEqual(int(standing.sum()), 2)
                if len(rises) >= 20:                    # points 2 m apart
                    self.assertGreaterEqual(float(np.median(rises)), F / 3)
                    thinned.append(r["id"])
        if self.site["willow_lawn"]:
            self.assertIn("willow-lawn", thinned)

    def test_each_tree_left_standing_in_a_clearing_grows_and_none_on_the_vegetable_patch(self):
        """Every texel of the local layer at full density inside a clearing, which only its scattered trees hold, is
        one where the renderer grows a tree at full density: off the water, the roads and the buildings. None lies on
        the vegetable patch, whose beds and hollow, which floods in heavy rain, take no trees."""
        from scipy import ndimage as ndi

        trees = self.layer("locA")[..., 1] / 255.0
        patch = ndi.binary_closing(self.layer("locN")[..., self.ms.CHANNELS["locN"].index("surface")] == 4,
                                   iterations=3)
        for r in (r for r in self.meta.get("clearings", []) if r["layer"] == "local"):
            rows, cols = self.inner_texels(r, self.grid, 0.0)
            full = trees[rows, cols] >= 1.0
            rows, cols = rows[full], cols[full]
            ring = np.asarray(r["outline"], float)
            grows = SightLines(self.ms, self.data, ring.min(axis=0), ring.max(axis=0)).density(
                np.column_stack(self.grid.to_world(rows, cols)))
            with self.subTest(clearing=r["id"]):
                self.assertGreater(len(rows), 0)
                self.assertGreaterEqual(float(grows.min()), 0.9)
                self.assertFalse(patch[rows, cols].any())

    def test_every_clearings_reason_is_a_decision_in_the_manifest(self):
        import clearings

        decisions = json.loads((ROOT / "src" / "mesh_scene.json").read_text(encoding="utf-8"))["decisions"]
        reasons = {r["decision"] for r in self.meta.get("clearings", [])} | set(clearings.LINEAR.values())
        self.assertEqual(reasons - set(decisions), set())

    def inner_texels(self, record, grid, margin):
        """The (rows, columns) of a grid's texels whose centres lie inside a recorded clearing's outline, `margin`
        metres or more in from it."""
        import clearings
        import shapely

        inner = clearings.polygon(record).buffer(-margin)
        i, j = np.mgrid[0:grid.ny, 0:grid.nx]
        e, n = grid.to_world(i, j)
        lo, hi = np.asarray(inner.bounds[:2]), np.asarray(inner.bounds[2:])
        near = (e >= lo[0]) & (e <= hi[0]) & (n >= lo[1]) & (n <= hi[1])
        inside = np.zeros(e.shape, bool)
        inside[near] = shapely.contains_xy(inner, e[near], n[near])
        return np.nonzero(inside)

    @staticmethod
    def across_edge(ring, step=2.0, inside=None):
        """Points every `step` metres along an outline `ring`, and the unit normals there that point away from the patch
        it bounds, whose inside `inside` tells for (n, 2) points (by default, the ring's own polygon's)."""
        import shapely

        line = shapely.LineString(ring)
        if inside is None:
            shell = shapely.Polygon(ring)
            inside = lambda p: shapely.contains_xy(shell, *p.T)
        s = np.arange(0.0, line.length, step)
        pts = shapely.get_coordinates(shapely.line_interpolate_point(line, s))
        ahead = shapely.get_coordinates(shapely.line_interpolate_point(line, s + 1.0)) - pts
        size = np.hypot(*ahead.T)
        pts, ahead, size = pts[size > 1e-6], ahead[size > 1e-6], size[size > 1e-6]
        normal = np.column_stack([ahead[:, 1], -ahead[:, 0]]) / size[:, None]
        normal[inside(pts + normal)] *= -1.0
        return pts, normal

    def test_each_clearing_is_cleared_in_the_packed_layers_but_for_its_scattered_trees(self):
        """Inside each recorded clearing's outline, where its multiplier falls below a half, no texel of the packed
        layer holds more than half a tree's density (to its 1/15 steps), but where a tree stands scattered at full
        density: the records describe the layers. The Forest layer's texels are read a texel and a half in, past
        pack.py's smoothing."""
        layers = {"local": (self.grid, self.layer("locA")[..., 1] / 255.0, 0.0),
                  "forest": (self.ms.Grid(self.meta["forest"]), self.layer("forest")[..., 0] / 255.0, 1.5)}
        for r in self.meta.get("clearings", []):
            grid, trees, texels = layers[r["layer"]]
            rows, cols = self.inner_texels(r, grid, texels * grid.texel_size()[0])
            held = trees[rows, cols]
            with self.subTest(clearing=r["id"], at=np.round(np.mean(r["outline"], axis=0)).tolist()):
                self.assertGreater(len(rows), 0)
                self.assertLessEqual(float(held[held < 1.0].max(initial=0.0)), 8 / 15 + 1e-6)
                if r["layer"] == "forest":
                    self.assertLess(float(held.max()), 1.0)

    def test_the_ground_each_clearing_bares_reads_as_its_place(self):
        """The Willow's lawn is lawn, not the forest floor, but for the vegetable patch's beds that it takes in round
        the northern greenhouses; the Shack's "dank overgrown garden" (PoA 14) is rank grass, nettles and bramble
        (clearings.OVERGROWN). Earth (its paths) may cross either. Each is read further in than the band, half
        clearings.ECOTONE_M either side of its outline, where its ground gives way to the ground round it."""
        import clearings

        place = {"willow-lawn": {1, 3, 4}, "shack-garden": {clearings.OVERGROWN, 3}}
        surface = self.layer("locN")[..., self.ms.CHANNELS["locN"].index("surface")]
        for r in (r for r in self.meta.get("clearings", []) if r["layer"] == "local"):
            rows, cols = self.inner_texels(r, self.grid, clearings.ECOTONE_M / 2 + 2.0)
            with self.subTest(clearing=r["id"]):
                self.assertGreater(len(rows), 0)
                self.assertLessEqual(set(np.unique(surface[rows, cols]).tolist()), place[r["id"]])

    CONTRAST = 0.01             # a change in the ground's albedo this big, in any channel, shows (meadow to lawn: 0.02)
    OWN_EDGES = (3, 4, 6, 7)    # earth, beds, the pitch and shingle: paths, plots and shores, which keep their own edges
    STRETCH = 20                # points 2 m apart along an edge: 40 m

    def hard_ground(self):
        """The texels whose own lines explain the edges of the ground beside them: paths, beds, the pitch and the shore
        (OWN_EDGES), water, buildings, walls, and the railway's earthworks and cleared lineside."""
        import fix_railway

        scene = json.loads((ROOT / "src" / "mesh_scene.json").read_text(encoding="utf-8"))
        georef = self.ms.load_georef()
        hard = np.isin(self.layer("locN")[..., 0], self.OWN_EDGES) | (self.layer("locW") > -0.9)
        hard |= self.ms.sd_bytes(self.layer("locA")[..., 3]) < 1.0
        hard |= self.ms.sd_bytes(self.layer("locB")[..., 0]) < 1.5
        hard |= fix_railway.footprint(scene, self.meta, georef, self.layer("locBed").astype(np.float64))
        (rows, cols), _, _, lineside = fix_railway.surfaces(scene, self.meta, georef)
        hard[rows, cols] |= lineside
        return hard

    def across_stretches(self, lines, inside, hard, out):
        """For each 40 m stretch of the outlines `lines` ((n, 2) world points each, the matching `inside` telling their
        patch's side, as across_edge takes it) that does not run along `hard` ground (a third of its points or more
        within 4 m of it): (its first point, the surface classes along its lines across the edge at offsets `out`, one
        row a line, and whether each lies off hard ground)."""
        from scipy import ndimage as ndi

        surface = self.layer("locN")[..., 0]
        near = ndi.distance_transform_edt(~hard) * self.grid.texel_size()[0]
        for line, side in zip(lines, inside):
            pts, normal = self.across_edge(line, inside=side)
            q = pts[:, None, :] + out[None, :, None] * normal[:, None, :]
            i, j = (np.clip(np.rint(a).astype(int), 0, size - 1)
                    for a, size in zip(self.grid.to_texel(q[..., 0], q[..., 1]), (self.grid.ny, self.grid.nx)))
            along = self.ms.sample(near, self.grid, *pts.T) < 4.0
            for k in range(0, len(pts) - self.STRETCH + 1, self.STRETCH):
                if along[k: k + self.STRETCH].mean() < 1 / 3:
                    s = slice(k, k + self.STRETCH)
                    yield pts[k], surface[i[s], j[s]], ~hard[i[s], j[s]]

    def ground_edges(self, lines, inside, hard):
        """(where, width) for each 40 m stretch of the outlines (across_stretches) across which the ground's colour (each
        class's mean colour in frag4.glsl's classCol) differs perceptibly 10 to 16 m in and out: the metres over which
        the share of that change, averaged along the stretch, goes from a tenth to nine tenths, its samples on hard
        ground left out."""
        colours = class_colours()
        palette = np.array([np.mean(colours[k][0], axis=0) for k in range(max(colours) + 1)])
        out = np.arange(-16.0, 16.5, 1.0)
        found = []
        for where, classes, kept in self.across_stretches(lines, inside, hard, out):
            n = kept.sum(axis=0)
            profile = (palette[classes] * kept[..., None]).sum(axis=0) / np.maximum(n, 1)[:, None]
            profile[n == 0] = np.nan
            inner, outer = profile[(out <= -10.0) & (n > 0)], profile[(out >= 10.0) & (n > 0)]
            if not len(inner) or not len(outer):
                continue
            change = outer.mean(axis=0) - inner.mean(axis=0)
            if np.abs(change).max() < self.CONTRAST:
                continue
            share = (profile - inner.mean(axis=0)) @ change / float(change @ change)
            found.append((where, float(((share > 0.1) & (share < 0.9)).sum())))
        return found

    def test_each_clearings_ground_fades_into_the_ground_round_it_over_several_metres(self):
        """A clearing's ground does not end at a line, which reads as a cutout or a stain: along each 40 m of its edge
        (ground_edges) its ground's colour changes over clearings.BLEND_M or more."""
        import clearings

        hard, tested = self.hard_ground(), set()
        for r in (r for r in self.meta.get("clearings", []) if r["layer"] == "local"):
            edges = self.ground_edges([np.asarray(r["outline"], float)], [None], hard)
            with self.subTest(clearing=r["id"]):
                self.assertEqual([(np.round(p).tolist(), w) for p, w in edges if w < clearings.BLEND_M], [])
            tested |= {r["id"]} if edges else set()
        if self.site["shack_garden"]:                # its ground is not the meadow's, so its edge shows
            self.assertIn("shack-garden", tested)

    def test_every_patch_of_ground_fades_into_the_ground_round_it(self):
        """No patch of ground at the site ends at a line, straight or wavering, which reads as a cutout or a stain: not
        the maps' scrub, the woods' floor, the lawns, nor a clearing's ground. Each class's patches are outlined where
        their share, smoothed over half clearings.ECOTONE_M to pass over the tufts their edges blend in, crosses a half.
        Along each 40 m of those outlines (across_stretches, lines from half ECOTONE_M and 2 m inside to as far outside,
        their samples on hard ground left out), at least clearings.INTERLEAVE of the lines that cross between the class's
        ground and other ground find the two interleaved (ClearingTests.interleaved), but where paths, beds, the pitch,
        the shore, water, buildings, walls or the railway's works run along the stretch (hard_ground)."""
        import contourpy
        from scipy import ndimage as ndi

        import clearings

        surface, hard = self.layer("locN")[..., 0], self.hard_ground()
        reach = clearings.ECOTONE_M / 2 + 2.0
        out = np.arange(-reach, reach + 0.5, 1.0)
        for cls in (1, clearings.SCRUB, 5, clearings.OVERGROWN):
            share = ndi.gaussian_filter((surface == cls).astype(np.float64),
                                        clearings.ECOTONE_M / 2 / self.grid.texel_size()[0])
            lines = [np.column_stack(self.grid.to_world(xy[:, 1], xy[:, 0]))
                     for xy in contourpy.contour_generator(z=share).lines(0.5) if len(xy) > 2]
            inside = lambda p, share=share: self.ms.sample(share, self.grid, *p.T) > 0.5
            crisp, tested = [], 0
            for where, classes, kept in self.across_stretches(lines, [inside] * len(lines), hard, out):
                mixed = ClearingTests.interleaved((classes == cls) & kept, (classes != cls) & kept)
                if len(mixed) >= self.STRETCH // 2:
                    tested += 1
                    if mixed.mean() < clearings.INTERLEAVE:
                        crisp.append((np.round(where).tolist(), round(float(mixed.mean()), 2)))
            with self.subTest(surface_class=cls):
                if cls in (clearings.SCRUB, 5):
                    self.assertGreater(tested, 20)
                self.assertEqual(crisp, [])

    def test_the_woods_floor_lies_under_their_canopy(self):
        """The woods' floor, their litter, lies under their canopy (a tree density over 0.05): beyond it, its tufts would
        read as specks on open ground. So a wood whose trees end at a line gives way to the ground round it within that
        line, and scrub gives none of its ground up to the floor. The floor may lie bare only under water, where trees
        are kept off the streams, and where a clearing's ground gives way to the woods' across its thinning edge
        (within half clearings.ECOTONE_M and a texel of its outline or an island of wood it leaves standing)."""
        import shapely

        import clearings

        surface, trees = self.layer("locN")[..., 0], self.layer("locA")[..., 1] / 255.0
        rows, cols = np.nonzero((surface == 5) & (trees <= 0.05) & (self.layer("locW") <= -0.9))
        at = shapely.points(*self.grid.to_world(rows, cols))
        edge = np.zeros(len(at), bool)
        for r in (r for r in self.meta.get("clearings", []) if r["layer"] == "local"):
            edge |= shapely.distance(clearings.polygon(r).boundary, at) <= clearings.ECOTONE_M / 2 + \
                self.grid.texel_size()[0]
        self.assertEqual(np.round(shapely.get_coordinates(at[~edge])).tolist(), [])

    def test_a_wood_that_thins_out_keeps_its_floor_past_its_edges_band(self):
        """A wood's floor gives way to the ground round it across clearings.ECOTONE_M centred on its dense canopy's edge
        (a tree density over 0.55), as every patch's does, so half ECOTONE_M in where the wood thins out. Only where its
        trees end at a line (full trees, 0.95 or more, beside a texel of 0.55 or less), as the station wood's and the
        train's screen's do, does it give way within that line, up to ECOTONE_M in. So deeper under the dense canopy than
        half ECOTONE_M and a texel, and further than ECOTONE_M and a texel from such a line, the ground is all floor, off
        hard ground (hard_ground)."""
        from scipy import ndimage as ndi

        import clearings

        surface, trees = self.layer("locN")[..., 0], self.layer("locA")[..., 1] / 255.0
        texel = self.grid.texel_size()[0]
        dense = trees > 0.55
        line = (trees >= 0.95) & ndi.binary_dilation(~dense, structure=ndi.generate_binary_structure(2, 1))
        deep = dense & (ndi.distance_transform_edt(dense) * texel > clearings.ECOTONE_M / 2 + texel) & \
            (ndi.distance_transform_edt(~line) * texel > clearings.ECOTONE_M + texel) & ~self.hard_ground()
        self.assertGreater(int(deep.sum()), 10000)
        rows, cols = np.nonzero(deep & (surface != 5))
        self.assertEqual(np.round(np.column_stack(self.grid.to_world(rows, cols))).tolist()[:10], [])

    def test_the_lake_front_wall_stands_at_the_top_of_a_cliff(self):
        """White Hound's 2e, "wall along edge of cliff overlooking lake": the castle's base carries it on the cliff's
        top, its upper part "sloping back towards the base of the castle", its lower part near "vertical"."""
        if not self.site["base"]:
            self.skipTest("the site keeps the original build's plinth")
        from scipy import ndimage as ndi

        scene = json.loads((ROOT / "src" / "mesh_scene.json").read_text(encoding="utf-8"))
        walls = [p["at"] for a in scene["assets"] for p in a.get("castle_pieces", []) if "(2e)" in p.get("note", "")]
        bed, wet = self.layer("locBed"), self.layer("locW") > 0.0
        idx = ndi.distance_transform_edt(~wet, return_distances=False, return_indices=True)
        lake = self.site["lake"]
        self.assertEqual(len(walls), 3)
        for x, z in walls:
            i, j = (int(round(float(v))) for v in self.grid.to_texel(x, z))
            water = np.array(self.grid.to_world(idx[0][i, j], idx[1][i, j]), float)
            run = float(np.hypot(water[0] - x, water[1] - z))
            self.assertGreaterEqual(float(bed[i, j]) - lake, 20.0, (x, z))                 # on the cliff's top
            self.assertGreaterEqual((float(bed[i, j]) - lake) / max(run, 1.0), 2.0, (x, z))    # a drop, not a slope

    def test_the_vegetable_patch_floods_in_heavy_rain(self):
        """OotP: they "splashed across the flooded vegetable patch to double Herbology"; White Hound: it lies "quite
        low-lying relative to the rest of the grounds, or in its own little hollow, because it can flood in heavy rain"."""
        if not self.site["base"]:
            self.skipTest("the site keeps the original build's plinth")
        from scipy import ndimage as ndi

        bed, w = self.layer("locBed").astype(np.float64), self.layer("locW")
        patch = ndi.binary_closing(self.layer("locN")[..., 0] == 4, iterations=3)       # its hatching, as an area
        platform = np.abs(bed - self.site["lake"] - 36.0) < 0.02
        near = patch & (ndi.distance_transform_edt(~platform) * self.grid.texel_size()[0] < 150.0)
        ii, jj = np.nonzero(near)
        box = (slice(ii.min() - 40, ii.max() + 41), slice(jj.min() - 40, jj.max() + 41))
        pond = common.fill(bed[box], w[box] > 0.0) - bed[box]
        flooded = float(((pond >= 0.1) & near[box]).sum() * np.prod(self.grid.texel_size()))     # deep enough to splash
        self.assertGreaterEqual(flooded, 10000.0)           # the patch's hollow, 1.15 ha at X, not what is left of it

    def test_the_greenhouses_stand_below_the_castle_and_above_hagrids_cabin(self):
        """White Hound: the greenhouses are "probably themselves somewhat lower than the castle", and "Hagrid's cabin is
        downslope from the castle, and probably also downslope from the greenhouses"."""
        scene = json.loads((ROOT / "src" / "mesh_scene.json").read_text(encoding="utf-8"))
        houses = np.array([c["at"] for a in scene["assets"] if a["family"] == "greenhouses" and a["id"] != "court-glasshouse"
                           for c in a.get("components", [])])
        hut = next(p["p"] for p in self.meta["extras"]["poi"] if p["name"] == "Hagrid's hut")
        bed = self.layer("locBed")
        ground = self.ms.sample(bed, self.grid, houses[:, 0], houses[:, 1])
        cabin = float(self.ms.sample(bed, self.grid, np.array([hut[0]]), np.array([hut[1]]))[0])
        self.assertLess(float(ground.max()), self.site["lake"] + 36.0)
        self.assertGreater(float(ground.min()), cabin)

    def test_the_northern_greenhouses_stand_together_on_the_patchs_terrace(self):
        """The vegetable patch and its greenhouses lie on one level terrace below the castle; a greenhouse reaching past
        the terrace's edge stands wholly on it, not half down the mound's flank below its neighbours."""
        if not self.site["base"]:
            self.skipTest("the site keeps the original build's plinth")
        scene = json.loads((ROOT / "src" / "mesh_scene.json").read_text(encoding="utf-8"))
        north = np.array([c["at"] for c in next(a for a in scene["assets"] if a["id"] == "greenhouses-north")["components"]])
        ground = self.ms.sample(self.layer("locBed"), self.grid, north[:, 0], north[:, 1])
        self.assertLess(float(ground.max() - ground.min()), 1.0, ground.round(1).tolist())

    def test_graded_channels_fall_along_their_course(self):
        if not self.site["graded"]:
            self.skipTest("the site keeps the original carve")
        bed, w, stream = self.layer("locBed"), self.layer("locW"), self.ms.sd_bytes(self.layer("locB")[..., 1])
        georef = self.ms.load_georef()
        total = carved = 0
        for way, E, N in self.osm_courses():
            i, j = (np.round(v).astype(int) for v in self.grid.to_texel(E, N))
            inside = (i >= 0) & (i < bed.shape[0]) & (j >= 0) & (j < bed.shape[1])
            if self.site["carve_in_maps"]:
                course = inside
            else:              # inside White Hound's maps, its own streams may cross a real course
                course = inside & np.array([self.ms.map_region(e, n, georef) == "outside" for e, n in zip(E, N)])
            i, j = i[course], j[course]
            on = (stream[i, j] < 0.0) & (w[i, j] > 0.0) & (w[i, j] < 2.0)
            total += len(on)
            carved += on.sum()
            self.assertTrue(np.all(np.diff((bed + w)[i[on], j[on]]) <= 0.021), way)
        self.assertGreater(total, 0)
        self.assertGreaterEqual(carved / total, 0.95)

    def test_real_watercourses_stay_out_of_white_hounds_maps_unless_configured(self):
        if self.site["carve_in_maps"]:
            self.skipTest("the site carves its real watercourses inside the maps too")
        stream = self.ms.sd_bytes(self.layer("locB")[..., 1])
        georef = self.ms.load_georef()
        for way, E, N in self.osm_courses():
            i, j = (np.round(v).astype(int) for v in self.grid.to_texel(E, N))
            inside = (i >= 0) & (i < stream.shape[0]) & (j >= 0) & (j < stream.shape[1])
            mapped = np.array([self.ms.map_region(e, n, georef) != "outside" for e, n in zip(E[inside], N[inside])])
            if mapped.any():
                # White Hound's own streams may cross a real course, but do not follow it
                self.assertLess(float(np.mean(stream[i[inside], j[inside]][mapped] < 0.0)), 0.1, way)

    def test_the_earthworks_enclose_no_dry_hollows(self):
        if not self.site["fill_pits"]:
            self.skipTest("the site keeps the original earthworks, hollows and all")
        from scipy import ndimage as ndi

        bed, w = self.layer("locBed")[::2, ::2], self.layer("locW")[::2, ::2]
        ii, jj = np.mgrid[0:self.grid.ny:2, 0:self.grid.nx:2]
        E, N = self.grid.to_world(ii, jj)
        natural = ndi.gaussian_filter(terrain.heights(self.site["dem"], E + self.site["anchor"][0],
                                                      N + self.site["anchor"][1]), 1.5)
        wet = w > 0.0                          # water drains whatever reaches it
        pits = (common.fill(bed, wet) - bed > 1.0) & ~(common.fill(natural, wet) - natural > 1.0)
        labels, n = ndi.label(pits)
        sizes = np.bincount(labels.ravel())[1:]
        big = [(int(s) * 16, [round(float(v)) for v in self.grid.to_world(*np.argwhere(labels == k + 1).mean(0) * 2)])
               for k, s in enumerate(sizes) if s >= 10]
        self.assertEqual(big, [])              # (m², castle-relative centre) of each hollow over 160 m²

    def pitch_points(self, beyond_m):
        """Castle-relative (E, N) every half metre inside the registered outer edge of the pitch's stands, or within
        `beyond_m` metres outside it."""
        import shapely

        P = self.meta["pitch"]
        ax = np.radians(P["brg"])
        along, across = np.array([np.sin(ax), np.cos(ax)]), np.array([-np.cos(ax), np.sin(ax)])
        t = np.linspace(-np.pi, np.pi, 4000)
        region = shapely.Polygon(np.column_stack([P["a"] * np.cos(t), P["b"] * np.sin(t)])).buffer(beyond_m)
        x0, y0, x1, y1 = region.bounds
        u, v = (g.ravel() for g in np.meshgrid(np.arange(x0, x1, 0.5), np.arange(y0, y1, 0.5)))
        keep = shapely.contains_xy(region, u, v)
        u, v = u[keep], v[keep]
        return P["c"][0] + u * along[0] + v * across[0], P["c"][1] + u * along[1] + v * across[1]

    def test_no_tree_grows_on_the_pitch_or_within_a_crown_of_its_stands(self):
        import build_final

        E, N = self.pitch_points(build_final.TREE_CROWN_M)
        trees = self.ms.sample(self.layer("locA")[..., 1].astype(np.float64), self.grid, E, N)
        self.assertEqual(float(trees.max()), 0.0)

    def test_the_ground_under_the_pitch_and_its_stands_lies_at_the_playing_level(self):
        E, N = self.pitch_points(0.0)
        bed = self.ms.sample(self.layer("locBed"), self.grid, E, N)
        np.testing.assert_allclose(bed, self.meta["pitch"]["level"], atol=self.data["tex"]["locBed"]["q"])

    def test_the_forest_maps_lettering_and_compass_rose_grow_no_wood(self):
        """The ink of the Forest map's lettering, and of its compass rose's N, W, cross, E and S, whose extents in map
        pixels (x0, y0, x1, y1) are listed here, is as sparsely treed as the open moor round it (a mean tree density of up
        to 0.07, from the map's scattered trees), not wooded."""
        classes = np.load(inputs.require("original-intermediates") / "build" / "cls_ff.npy")
        density, grid = self.layer("forest")[..., 0] / 255.0, self.ms.Grid(self.meta["forest"])
        for x0, y0, x1, y1 in ((2092, 3313, 2849, 3372), (2739, 2849, 2771, 2883), (2538, 2915, 2577, 2952),
                               (2571, 2881, 2811, 3123), (2812, 3048, 2842, 3081), (2612, 3126, 2638, 3155)):
            v, u = np.nonzero(classes[y0:y1, x0:x1] == common.C_BLACK)
            ink = self.ms.sample(density, grid, *common.ff_to_world(u + x0 + 0.5, v + y0 + 0.5))
            self.assertLess(float(ink.mean()), 0.1, (x0, y0))

    def test_the_forest_maps_paths_and_rivers_banks_grow_no_wood(self):
        """Away from the Forest map's woods (its green), its black ink draws paths and tracks, and the banks of its rivers:
        as sparsely treed as the open moor, not rows of trees."""
        from scipy import ndimage as ndi

        classes = np.load(inputs.require("original-intermediates") / "build" / "cls_ff.npy")
        woods = ndi.binary_dilation((classes == common.C_DENSE) | (classes == common.C_UNDER), iterations=6)
        banks = ndi.binary_dilation(classes == common.C_WATER, iterations=3)
        ink = (classes == common.C_BLACK) & ~woods
        density, grid = self.layer("forest")[..., 0] / 255.0, self.ms.Grid(self.meta["forest"])
        for name, where in (("paths and tracks", ink & ~banks), ("rivers' banks", ink & banks)):
            v, u = np.nonzero(where)
            self.assertLess(float(self.ms.sample(density, grid, *common.ff_to_world(u + 0.5, v + 0.5)).mean()), 0.1, name)

    @staticmethod
    def bands(e, n, half, step=2.0):
        """Points every `step` metres along a polyline and out to `half` metres either side of it."""
        out_e, out_n = [], []
        for k in range(len(e) - 1):
            length = np.hypot(e[k + 1] - e[k], n[k + 1] - n[k])
            if length == 0:
                continue
            t = np.linspace(0.0, 1.0, int(length / step) + 2)[:, None]
            off = np.linspace(-half, half, int(2 * half / step) + 1)[None, :]
            ue, un = (e[k + 1] - e[k]) / length, (n[k + 1] - n[k]) / length
            out_e.append((e[k] + t * (e[k + 1] - e[k]) - off * un).ravel())
            out_n.append((n[k] + t * (n[k + 1] - n[k]) + off * ue).ravel())
        return np.concatenate(out_e), np.concatenate(out_n)

    def real_ways(self, select):
        """The site's OpenStreetMap ways whose tags `select` picks: (tags, castle-relative E, N)."""
        source = "site-osm-x" if self.site is common.SITES["x"] else "original-intermediates"
        osm = json.loads((inputs.require(source) / "osm.json").read_text(encoding="utf-8"))
        for el in osm["elements"]:
            tags = el.get("tags", {})
            if el["type"] == "way" and el.get("geometry") and select(tags):
                g = el["geometry"]
                e, n = common.to_tm.transform([p["lon"] for p in g], [p["lat"] for p in g])
                yield tags, np.asarray(e) - self.site["anchor"][0], np.asarray(n) - self.site["anchor"][1]

    def forest_read(self):
        """White Hound's Forest density as the shader reads it at world points, and nothing outside its layer."""
        density, grid = self.layer("forest")[..., 0] / 255.0, self.ms.Grid(self.meta["forest"])

        def read(e, n):
            i, j = grid.to_texel(e, n)
            inside = (i >= 0) & (i <= grid.ny - 1) & (j >= 0) & (j <= grid.nx - 1)
            return np.where(inside, self.ms.sample(density, grid, e, n), 0.0)
        return read

    def regional_read(self, name):
        """A 10 m regional layer (MBed, MW) read bilinearly at world points, as frag1.glsl's regionalH reads it within
        4.7 km of the castle (texel centres 10 m apart from -5000 m, rows running north)."""
        from scipy import ndimage as ndi

        layer = self.layer(name).astype(np.float64)
        return lambda e, n: ndi.map_coordinates(layer, [(np.asarray(n) + 5000.0) / 10.0, (np.asarray(e) + 5000.0) / 10.0],
                                                order=1, mode="nearest")

    def far_trees(self, lo, hi):
        """The renderer's trees (frag1.glsl treeSD, treeDens, treeH) in the 7 m cells from `lo` to `hi` (east, north),
        where they are White Hound's Forest layer's: off the local layer, within 2.5 km of the castle, where no
        plantation is read, and far inside his layer. (centres, density there, whether a tree stands, whether it is a
        pine, its height)."""
        import shapely

        step = SightLines.smooth
        c0, c1 = np.floor(np.asarray(lo, float) / 7.0) - 1, np.floor(np.asarray(hi, float) / 7.0) + 1
        cz, cx = np.mgrid[c0[1]:c1[1] + 1, c0[0]:c1[0] + 1]
        cells = np.stack([cx.ravel(), cz.ravel()], 1).astype(np.float32)
        h = SightLines.hash42(cells).astype(np.float64)
        ctr = (cells + 0.25 + 0.5 * h[:, 1:3]) * 7.0
        e, n = self.grid.to_world(np.array([0, 0, self.grid.ny - 1, self.grid.ny - 1]),
                                  np.array([0, self.grid.nx - 1, 0, self.grid.nx - 1]))
        local = shapely.Polygon(np.column_stack([e, n])[[0, 1, 3, 2]]).buffer(80.0)
        self.assertFalse(shapely.intersects_xy(local, *ctr.T).any())
        self.assertLess(float(np.hypot(*ctr.T).max()), 2500.0)
        forest = self.ms.Grid(self.meta["forest"])
        i, j = forest.to_texel(*ctr.T)
        edge = np.minimum.reduce([i, forest.ny - 1 - i, j, forest.nx - 1 - j]) * forest.texel_size()[0]
        self.assertGreater(float(edge.min()), 300.0)
        dens = self.forest_read()(*ctr.T) * (self.regional_read("MW")(*ctr.T) <= -0.9)
        poi = {p["name"]: np.asarray(p["p"], float) for p in self.meta["extras"]["poi"]}
        con = h[:, 3] < np.maximum.reduce([np.full(len(ctr), 0.12),
                                           1.0 - step(120.0, 230.0, np.hypot(*(ctr - poi["Hogsmeade station"]).T)),
                                           0.85 * (1.0 - step(260.0, 480.0, np.hypot(*(ctr - poi["Grawp's clearing"]).T)))])
        hh = h[:, 3] * 13.7 % 1.0
        height = np.where(con, 13.0 + 8.0 * hh, 10.0 + 9.0 * hh) * (0.72 + 0.28 * step(0.1, 0.8, dens))
        return ctr, dens, h[:, 0] < dens, con, height

    def test_white_hounds_forest_grows_no_tree_on_the_real_roads_railway_or_buildings(self):
        import forest_extent as fx

        if self.site.get("forest") is None:
            self.skipTest("this site keeps White Hound's Forest as drawn")
        read = self.forest_read()
        for name, half, select in (("A-roads", fx.A_ROAD_M, fx.is_a_road), ("railway", fx.RAILWAY_M, fx.is_railway),
                                   ("buildings", fx.BUILDING_M, fx.is_building)):
            worst = max(float(read(*self.bands(e, n, half)).max()) for _, e, n in self.real_ways(select))
            self.assertEqual(worst, 0.0, name)

    def test_the_forest_ends_short_of_the_roads_it_stops_at_and_of_the_settlements_along_them(self):
        """Seen from the castle, nothing beyond such a road is Forest, nor anything short of it by the least setback; a
        building near it stands outside the Forest, with no Forest between it and the road."""
        import forest_extent as fx
        from scipy.spatial import cKDTree

        rules = self.site.get("forest") or {}
        if not rules.get("ends_at"):
            self.skipTest("the Forest ends at no road at this site")
        read = self.forest_read()
        roads = [self.bands(e, n, 0.0, 10.0) for _, e, n in self.real_ways(lambda t: fx.ends_here(t, rules))]
        re_, rn = np.concatenate([r[0] for r in roads]), np.concatenate([r[1] for r in roads])
        s = np.arange(-fx.SETBACK_MIN_M, 1500.0, 8.0)[None, :]
        r = np.hypot(re_, rn)[:, None]
        self.assertEqual(float(read(re_[:, None] * (1 + s / r), rn[:, None] * (1 + s / r)).max()), 0.0)
        near = cKDTree(np.column_stack([re_, rn]))
        settled = 0
        for _, e, n in self.real_ways(fx.is_building):
            d, k = near.query(np.column_stack([e, n]))
            if d.min() > fx.VILLAGE_REACH_M:
                continue
            settled += 1
            ce, cn = e[:-1].mean(), n[:-1].mean()
            to_road = self.bands(np.array([ce, re_[k[d.argmin()]]]), np.array([cn, rn[k[d.argmin()]]]), 0.0)
            self.assertEqual(float(read(*to_road).max()), 0.0, (ce, cn))
            self.assertEqual(float(read(*self.bands(e, n, fx.VILLAGE_MIN_M)).max()), 0.0, (ce, cn))
        self.assertGreater(settled, 0)

    def test_the_forest_keeps_white_hounds_scale_with_aragogs_nest_in_its_heart(self):
        """White Hound: the Forest "is probably about four miles across, at least in the south/north direction" and "needs
        to be three or four miles across", "about two miles across on a south-east to north-west axis"; Aragog's nest
        is in its heart "in the sense that it is in the dense inner area as opposed to the outskirts"."""
        import forest_extent as fx
        from scipy import ndimage as ndi
        from scipy.spatial import cKDTree

        if self.site.get("forest") is None:
            self.skipTest("this site keeps White Hound's Forest as drawn")
        mile = 1609.344
        density, grid = self.layer("forest")[..., 0] / 255.0, self.ms.Grid(self.meta["forest"])
        texel = grid.texel_size()[0]
        E, N = grid.to_world(*np.mgrid[0:grid.ny, 0:grid.nx])
        labels, _ = ndi.label(ndi.binary_closing(density > 0.5, iterations=3))
        sizes = np.bincount(labels.ravel()) * texel * texel
        body = np.isin(labels, [k for k in np.nonzero(sizes >= 1e6)[0] if k > 0])    # its parts of a km² or more
        self.assertTrue(3.0 <= np.ptp(N[body]) / mile <= 4.5, np.ptp(N[body]) / mile)
        self.assertGreaterEqual(np.ptp((N[body] - E[body]) / np.sqrt(2.0)) / mile, 2.0)
        # the open ground beyond the Forest: treeless, joined to the layer's edge, not counting the corridors
        pts = [self.bands(e, n, 0.0) for kind in (fx.is_a_road, fx.is_railway, fx.is_building)
               for _, e, n in self.real_ways(kind)]
        reach = cKDTree(np.column_stack([np.concatenate([p[0] for p in pts]), np.concatenate([p[1] for p in pts])]))
        corridor = reach.query(np.column_stack([E.ravel(), N.ravel()]), distance_upper_bound=40.0)[0].reshape(E.shape) < 40.0
        open_, _ = ndi.label((density < 0.15) & ~corridor)
        edge = np.unique(np.r_[open_[0], open_[-1], open_[:, 0], open_[:, -1]])
        beyond = np.isin(open_, edge[edge > 0])
        nest = next(p["p"] for p in self.meta["extras"]["poi"] if p["name"] == "Aragog's hollow")
        i, j = np.round(grid.to_texel(*nest)).astype(int)
        self.assertGreater(ndi.distance_transform_edt(~beyond)[i, j] * texel, mile / 2)

    def places(self):
        return {p["name"]: np.asarray(p["p"], float) for p in self.meta["extras"]["poi"]}

    def water_distance(self, at):
        """Metres from `at` to the nearest texel centre of water in the 10 m regional layer, and that water's surface."""
        bed, wd = self.layer("MBed").astype(np.float64), self.layer("MW").astype(np.float64)
        ii, jj = np.nonzero(wd > -0.9)
        d = np.hypot(jj * 10.0 - 5000.0 - at[0], ii * 10.0 - 5000.0 - at[1])
        k = int(np.argmin(d))
        return float(d[k]), float(bed[ii[k], jj[k]] + wd[ii[k], jj[k]])

    def line_distance(self, at, select):
        """Metres from `at` to the site's real ways that `select` picks."""
        import shapely

        lines = [shapely.LineString(np.column_stack([e, n])) for _, e, n in self.real_ways(select)]
        return float(shapely.union_all(lines).distance(shapely.Point(*at)))

    def test_aragogs_hollow_is_a_vast_steep_sided_crater_dry_above_the_loch(self):
        """CoS 15: Harry, Ron and Fang "had reached the rim of a vast hollow", and were carried "down the steep slope" into
        it; White Hound: it is "steep-sided ... at least fifty yards across and probably bigger", near a water supply
        "but not so close that it floods". In the packed 10 m ground (MBed) the hollow is 80 to 120 m across (as a disc
        of the area that lies 2 m or more below the plane through the hillside round it, 10 to 30 m beyond its rim), and
        8 m or more deep where it would spill; along three in four of the lines in from its rim the ground falls 30
        degrees or more over 10 m; no water lies in it, and its floor stands 3 m or more above the surface of the water
        nearest it."""
        import forest_places as fp
        from scipy import ndimage as ndi

        if not self.site["aragog_hollow"]:
            self.skipTest("this site keeps White Hound's crater as drawn, a patch bare of trees")
        at = np.asarray(self.site["aragog_hollow"], float)
        j0, i0 = (int(round(v)) for v in (at + 5000.0) / 10.0)
        w = (slice(i0 - 30, i0 + 31), slice(j0 - 30, j0 + 31))
        bed, wd = self.layer("MBed")[w].astype(np.float64), self.layer("MW")[w].astype(np.float64)
        depth = common.fill(bed) - bed
        labels, _ = ndi.label(depth > 1.0)
        self.assertGreater(labels[30, 30], 0)
        hollow = labels == labels[30, 30]
        self.assertGreaterEqual(float(depth[30, 30]), 8.0)
        N, E = np.mgrid[i0 - 30:i0 + 31, j0 - 30:j0 + 31] * 10.0 - 5000.0
        r = np.hypot(E - at[0], N - at[1])
        hillside = (r >= fp.RIM_M + 10.0) & (r <= fp.RIM_M + 30.0)
        coef = np.linalg.lstsq(np.column_stack([np.ones(hillside.sum()), E[hillside], N[hillside]]), bed[hillside],
                               rcond=None)[0]
        labels, _ = ndi.label(bed < coef[0] + coef[1] * E + coef[2] * N - 2.0)
        self.assertGreater(labels[30, 30], 0)
        across = 2.0 * np.sqrt((labels == labels[30, 30]).sum() * 100.0 / np.pi)
        self.assertTrue(80.0 <= across <= 120.0, across)
        read, s = self.regional_read("MBed"), np.arange(0.0, fp.RIM_M + 15.0, 1.0)
        rays = [read(at[0] + s * np.sin(a), at[1] + s * np.cos(a)) for a in np.radians(np.arange(0.0, 360.0, 10.0))]
        steepest = [np.degrees(np.arctan(float((z[10:] - z[:-10]).max()) / 10.0)) for z in rays]
        self.assertGreaterEqual(float(np.percentile(steepest, 25)), 30.0)
        self.assertTrue((wd[hollow] <= -0.9).all())
        _, surface = self.water_distance(at)
        self.assertGreaterEqual(float(bed[hollow].min()), surface + 3.0)

    def test_aragogs_hollow_lies_where_white_hound_reasons_the_nest_lies_cleared_of_trees(self):
        """White Hound: the nest lies "about one and a third miles in a straightish line from the edge of the Forest" and
        "cannot be much more than a mile and a half into the woods, and really should be less than that"; "to the right
        and north because the forest slopes down in that direction"; "fairly close to a water supply"; "in the dense inner
        area as opposed to the outskirts". So it lies 1 to 1.5 miles from the path's start by Hagrid's hut, within 45
        degrees of north of the castle, with water within 200 m of its rim and the railway 100 m or more beyond it, and
        the Forest round it, from 30 to 100 m beyond its rim, is 0.6 dense or more. No tree can stand in it, where the
        renderer reads no density: "a hollow which had been cleared of trees" (CoS 15)."""
        import forest_extent as fx
        import forest_places as fp

        if not self.site["aragog_hollow"]:
            self.skipTest("this site keeps White Hound's crater as drawn, a patch bare of trees")
        poi, mile = self.places(), 1609.344
        at = poi["Aragog's hollow"]
        self.assertTrue(1.0 <= np.hypot(*(at - poi["Hagrid's hut"])) / mile <= 1.5)
        self.assertLessEqual(abs(float(np.degrees(np.arctan2(at[0], at[1])))), 45.0)
        self.assertLessEqual(self.water_distance(at)[0], fp.RIM_M + 200.0)
        self.assertGreaterEqual(self.line_distance(at, fx.is_railway), fp.RIM_M + 100.0)
        r, a = np.meshgrid(np.arange(fp.RIM_M + 30.0, fp.RIM_M + 101.0, 10.0), np.radians(np.arange(0.0, 360.0, 5.0)))
        self.assertGreaterEqual(float(self.forest_read()(at[0] + r * np.sin(a), at[1] + r * np.cos(a)).mean()), 0.6)
        r, a = np.meshgrid(np.arange(0.0, fp.RIM_M + 0.5, 2.0), np.radians(np.arange(0.0, 360.0, 2.0)))
        self.assertEqual(float(self.forest_read()(at[0] + r * np.sin(a), at[1] + r * np.cos(a)).max()), 0.0)

    def test_the_forest_stands_again_on_the_crater_white_hound_drew(self):
        """Where his map draws the crater bare (6g), the hollow standing elsewhere at this site, the Forest is as dense as
        round it, all but on the railway's corridor and the hollow's own clearing: nineteen in twenty of its points
        0.5 or denser, with no field tint. His crater and its edge, which the layer's sampling of his map blurs, reach
        86 m from his spot, and the next lawn his map draws lies 99 m from it."""
        import clearings
        import forest_extent as fx
        import forest_places as fp
        import shapely

        if not self.site["aragog_hollow"]:
            self.skipTest("this site keeps White Hound's crater as drawn, a patch bare of trees")
        drawn = np.array(common.ff_to_world(*fp.DRAWN["Aragog's hollow"]), float)
        r, a = np.meshgrid(np.arange(0.0, 86.0, 4.0), np.radians(np.arange(0.0, 360.0, 5.0)))
        e, n = (drawn[0] + r * np.sin(a)).ravel(), (drawn[1] + r * np.cos(a)).ravel()
        rail = shapely.union_all([shapely.LineString(np.column_stack(w[1:])) for w in self.real_ways(fx.is_railway)])
        (hollow,) = [clearings.polygon(c) for c in self.meta["clearings"] if c["id"] == "aragogs-hollow"]
        off = shapely.distance(rail, shapely.points(e, n)) > fx.RAILWAY_M + 2.0 * 8.0 * np.sqrt(2.0)
        off &= ~shapely.contains_xy(hollow.buffer(fp.HOLLOW_FEATHER_M + 8.0 * np.sqrt(2.0)), e, n)
        self.assertGreater(int(off.sum()), 500)
        self.assertGreaterEqual(float(np.percentile(self.forest_read()(e[off], n[off]), 5)), 0.5)
        grid = self.ms.Grid(self.meta["forest"])
        field = self.ms.sample(self.layer("forest")[..., 1] / 255.0, grid, e[off], n[off])
        self.assertLessEqual(float(field.max()), 0.1)

    def test_grawps_clearing_is_a_small_artificial_clearing_with_a_few_small_pines_standing(self):
        """White Hound: "an apparently artificial clearing where Grawp is shackled, surrounded by small pine trees"; OotP
        30: "The trees to which the other ends of the ropes around Grawp's wrists and ankles were attached creaked
        ominously". Its recorded outline holds 0.3 to 0.4 ha. In its core, 12 m and more in from its outline, past the
        band where the wood thickens back, stand 3 to 10 small pines as the renderer grows them (far_trees), each where
        the density is 0.3 or less, so under four fifths of a full-grown pine's height, and no bigger tree."""
        import clearings
        import shapely

        if not self.site["grawp_clearing"]:
            self.skipTest("this site leaves White Hound's spot for Grawp's clearing in the trees")
        (record,) = [r for r in self.meta.get("clearings", []) if r["id"] == "grawps-clearing"]
        poly = clearings.polygon(record)
        self.assertTrue(3000.0 <= poly.area <= 4000.0, poly.area)
        ctr, dens, stands, pine, _ = self.far_trees(np.asarray(poly.bounds[:2]), np.asarray(poly.bounds[2:]))
        core = stands & shapely.contains_xy(poly.buffer(-12.0), *ctr.T)
        small = int((core & pine & (dens <= 0.3)).sum())
        self.assertTrue(3 <= small <= 10, small)
        self.assertLessEqual(float(dens[core].max(initial=0.0)), 0.3)

    def test_grawps_clearing_lies_in_dry_forest_in_front_of_the_castle(self):
        """White Hound: from "the front steps of the castle, which faces somewhat north of west", Harry can see the area
        where Grawp is confined, so "a substantial chunk of the Forest must be broadly in front of the castle"; Hagrid
        led them in by the path and then "About 600 yards after leaving the path". The clearing lies within 22.5 degrees
        of west-north-west of the Entrance Hall, 0.8 to 1.6 miles from the path's start by Hagrid's hut, 200 m or more
        from any water and 150 m or more from the railway and the A890."""
        import forest_extent as fx

        if not self.site["grawp_clearing"]:
            self.skipTest("this site leaves White Hound's spot for Grawp's clearing in the trees")
        poi, mile = self.places(), 1609.344
        at = poi["Grawp's clearing"]
        doors = np.degrees(np.arctan2(*(at - np.asarray(self.meta["named"]["EH"], float)))) % 360.0
        self.assertLessEqual(abs(float(doors) - 292.5), 22.5)
        self.assertTrue(0.8 <= np.hypot(*(at - poi["Hagrid's hut"])) / mile <= 1.6)
        self.assertGreaterEqual(self.water_distance(at)[0], 200.0)
        for select in (fx.is_railway, fx.is_a_road):
            self.assertGreaterEqual(self.line_distance(at, select), 150.0)

    def test_the_extras_name_each_forest_place_where_it_stands(self):
        import forest_places as fp

        poi = self.places()
        for name in fp.DRAWN:
            np.testing.assert_allclose(poi[name], fp.where(self.site, name), atol=0.05)


class PageTests(unittest.TestCase):
    FRAG = "".join((ROOT / "src" / f"frag{i}.glsl").read_text(encoding="utf-8") for i in range(1, 6))

    def test_castle_and_fog_heights_move_with_the_lake(self):
        lake_relative(self, self.FRAG, bands=1)

    def test_the_page_names_its_site_from_the_data(self):
        """The title, the loading screen and the heading name the site's glen from the packed extras, never a site's
        glen written into the page."""
        page = (ROOT / "src" / "template.html").read_text(encoding="utf-8")
        script = page + (ROOT / "src" / "template_c.html").read_text(encoding="utf-8")
        for site in common.SITES.values():
            self.assertNotIn(site["glen"], script)
        for slot in ("load-title", "glen"):
            self.assertIn(f'id="{slot}"', page)
            self.assertRegex(script, rf'\$\("{slot}"\)\.textContent = `[^`]*\$\{{X\.glen\}}')
        self.assertRegex(script, r"document\.title = `[^`]*\$\{X\.glen\}")

    def test_every_elevation_model_has_its_terrain_credit(self):
        script = (ROOT / "src" / "template_c.html").read_text(encoding="utf-8")
        table = script[script.index("const DEM_CREDITS = {"):]
        table = table[:table.index("};")]
        self.assertEqual(set(re.findall(r"^\s*(\w+): \[", table, re.M)), set(terrain.MODELS))
        page = (ROOT / "src" / "template.html").read_text(encoding="utf-8")
        for slot in ("dem-short", "dem-credit"):
            self.assertIn(f'id="{slot}"', page)


if __name__ == "__main__":
    unittest.main()
