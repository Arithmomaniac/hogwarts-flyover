"""The Willow family's codes in the production path: bark, knot, tunnel mouth and leaves."""
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import build_meshes  # noqa: E402
import mesh_willow  # noqa: E402
from mesh_geometry import panel  # noqa: E402
from materials import code  # noqa: E402
from materials.willow import BARK_AXES, TRUNK_AXIS  # noqa: E402
from pagetest import smoke  # noqa: E402
from render_testing import EAST, LAWN, SCENE, RenderTestCase  # noqa: E402


GRAIN_COV = 0.03                # the ridges' brightest quarter varies by at least this (std over mean) close up:
                                # 0.036-0.040 with its grain, 0.013-0.022 without


class WillowRenderTests(RenderTestCase):
    async def look(self, pos, yaw, pitch, hour, meshes=None):
        """The camera at world `pos` facing `yaw`, `pitch` at `hour`, over the scene's meshes or the diagnostic `meshes`."""
        frames = await self.page.evaluate("(h) => { __hog.tod(h); return __hog.frames; }", hour)
        await self.page.wait_for_function(f"__hog.frames >= {frames + 3}")
        body = None
        if meshes is not None:
            body = json.loads(build_meshes.strip_js(build_meshes.encode_payload({"diagnostic": meshes}, SCENE)))
        frames = await self.page.evaluate(
            "async ([b, pos, yaw, pitch]) => { __hog.camera(pos, yaw, pitch); if (b) await __hog.loadMeshes(b); return __hog.frames; }",
            [body, [float(v) for v in pos], yaw, pitch])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 10}")

    async def wood(self, auxes, facing):
        """Luminance on an internal-pixel grid over the middle half of the frame's height (rows up the frame): NaN but
        where it shows material 7 with an aux in `auxes`, in sunlight, facing within 45 degrees of `facing`."""
        found = await self.page.evaluate("""([auxes, f]) => {
            const [w, h] = __hog.probe(0, 0).res, y0 = h >> 2, H = h >> 1;
            const g = __hog.probeRect(0, y0, w, H), c = __hog.probeColorRect(0, y0, w, H), out = [];
            for (let i = 0; i < w * H; i++) {
              const r = g[i], ok = r[0] === 7 && auxes.includes(Math.round(r[4])) && r[1] * f[0] + r[2] * f[1] + r[3] * f[2] > 0.7
                && __hog.probeSun(i % w, y0 + Math.floor(i / w)) > 0.95;
              out.push(ok ? 0.2126 * c[4 * i] + 0.7152 * c[4 * i + 1] + 0.0722 * c[4 * i + 2] : null);
            }
            return { w, lum: out }; }""", [list(auxes), [float(v) for v in facing]])
        return np.array([np.nan if v is None else v for v in found["lum"]], float).reshape(-1, found["w"])

    async def trunk_and_root(self, attr_of, hour=16.0):
        """{"upright": (grid, axis up its furrows), "lying": ..., "wall": ...}: diagnostic wood tagged attr_of(code), seen
        close up from its sunlit side: a trunk on the Willow's own axis (its bark wraps round that axis), with no canopy to
        shade it, a root (a lying tube 20 degrees off its code's axis) and a wall like the hollow's."""
        lit = await self.page.evaluate("(h) => __hog.light(h).dir", hour)
        sun = np.array([lit[0], 0.0, lit[2]]) / math.hypot(lit[0], lit[2])
        x0, y0, z0, lx, lz = TRUNK_AXIS
        base = np.array([x0, y0 + await self.page.evaluate("DATA.meta.lake"), z0])
        up = np.array([lx, 1.0, lz])
        out = {}
        # the trunk, then tubes as wide as its flare where its cells crossfade between two rungs, and at the next rung
        for name, r in (("upright", 1.0), ("crossfaded", 1.1), ("flared", 1.5)):
            trunk = [mesh_willow.tube(np.array([base - up, base + 4.0 * up]), np.array([r, r]), attr_of(120), 64, cap=False)]
            await self.look(base + 1.7 * up + (r + 1.5) * sun, math.atan2(-sun[0], -sun[2]), 0.12, hour, trunk)
            out[name] = (await self.wood([attr_of(120)[1]], sun), 0)
        lying = next(aux for aux, axis in BARK_AXES.items() if abs(axis[2]) > 0.99)   # furrows along z: across the view
        x, z = LAWN[0] + 26.0, LAWN[1]
        ground = await self.ground(x, z)
        a = np.array([math.sin(math.radians(20.0)), 0.0, math.cos(math.radians(20.0))])
        mid = np.array([x + 1.2 + 0.36, ground + 1.2, z])
        root = mesh_willow.tube(np.array([mid - 2.5 * a, mid + 2.5 * a]), np.array([0.36, 0.36]), attr_of(lying), 32)
        await self.look((x, ground + 1.2, z), EAST, 0.0, hour, [root])
        out["lying"] = (await self.wood([attr_of(lying)[1]], (-1.0, 0.0, 0.0)), 1)
        # a wall of the hollow: a plate of the trunk's bark on a line out from the trunk's axis, facing round the axis as
        # the walls do (its bearing round the axis barely changes across it), out on the lawn and facing the sun
        gap = mesh_willow.gap_bearing(await self.page.evaluate("({extras: DATA.meta.extras})"))
        out_dir = max((np.array([-sun[2], 0.0, sun[0]]) * s for s in (1, -1)),
                      key=lambda v: v[0] * math.cos(gap) + v[2] * math.sin(gap))
        c = base + 14.0 * out_dir
        c[1] = await self.ground(c[0], c[2]) + 1.3
        across, rise = out_dir * 0.8, np.array([0.0, 0.8, 0.0])
        if np.cross(across, rise) @ sun < 0:
            across = -across
        plate = panel([c - across - rise, c + across - rise, c + across + rise, c - across + rise], attr_of(120))
        await self.look(c + 1.8 * sun, math.atan2(-sun[0], -sun[2]), 0.0, hour, [plate])
        out["wall"] = (await self.wood([attr_of(120)[1]], sun), 0)
        return out

    async def test_bark_close_up_shows_lit_plates_and_dark_furrows_along_the_wood(self):
        # the trunk, burl and roots read as smooth charcoal close up unless the bark has grey-brown plates split by deep
        # furrows; the furrows run along the wood, up the trunk and along a root that bends away from its code, as long
        # spindles, not polygons cut across them; a sailcloth trunk and root in their place set the light
        import scipy.ndimage as ndi
        steps = lambda grid, axis: np.nanmean(np.abs(np.diff(grid, axis=axis)))   # axis 0: up the frame, 1: across it
        bark = await self.trunk_and_root(lambda aux: (7, aux))
        sail = await self.trunk_and_root(lambda aux: (7, 24))
        for name, (grid, along) in bark.items():
            lum = grid[np.isfinite(grid)]
            with self.subTest(bark=name):
                self.assertGreater(lum.size, 3000, "too little of the wood is lit and facing the camera")
                self.assertGreater(lum.mean(), 0.62 * np.nanmean(sail[name][0]))       # grey-brown, not charcoal
                self.assertLess(np.percentile(lum, 10), 0.5 * np.percentile(lum, 75))  # deep furrows between plates
                self.assertLess(steps(grid, along), 0.65 * steps(grid, 1 - along))     # smoother along the furrows
                labels, n = ndi.label(np.nan_to_num(grid, nan=np.inf) < 0.5 * np.percentile(lum, 75))
                spans = [(sl[along].stop - sl[along].start) / (sl[1 - along].stop - sl[1 - along].start)
                         for sl, size in zip(ndi.find_objects(labels), np.bincount(labels.ravel())[1:]) if size >= 40]
                self.assertGreater(float(np.median(spans)), 4.0, len(spans))          # long spindles, not polygons
                ridge = lum[lum > np.percentile(lum, 75)]
                if name != "crossfaded":                # two spaces' ridges mix there, each with its own grain
                    self.assertGreater(float(ridge.std() / ridge.mean()), GRAIN_COV, "flat ridges, with no grain close up")

        def crossings(grid):
            """Furrow edges per pixel across the middle third of the frame (the wood about face-on)."""
            w = grid.shape[1]
            g = grid[:, w // 3:2 * w // 3]
            dark = g < 0.5 * np.nanpercentile(g, 75)
            both = np.isfinite(g[:, 1:]) & np.isfinite(g[:, :-1])
            return np.sum((dark[:, 1:] != dark[:, :-1]) & both) / max(np.sum(both), 1)
        # the flared base's finer cells keep its furrows as wide as the trunk's above it, seen from as far off
        ratio = crossings(bark["flared"][0]) / crossings(bark["upright"][0])
        self.assertTrue(0.8 < ratio < 1.25, f"the flare's furrows {1 / ratio:.2f} times the trunk's width")

    @smoke
    async def test_willow_codes_shade_as_bark_knot_mouth_and_leaves(self):
        # without their own cases these codes fall through to the sails' pale canvas
        lum = lambda rgb: np.asarray(rgb) @ [0.2126, 0.7152, 0.0722]
        codes = {"sail": (7, 24), **{name: code(f"willow-{name}") for name in ("bark", "knot", "root-hole", "leaves")}}
        looks = {name: await self.castle_box(attr, 16.0) for name, attr in codes.items()}
        wall = {name: np.median(v["wall"], axis=0) for name, v in looks.items()}
        self.assertLess(lum(wall["bark"]), 0.95 * lum(wall["sail"]))                   # wood darker than sailcloth
        self.assertGreater(wall["bark"][0], wall["bark"][2])                           # grey-brown, not blue
        self.assertLess(lum(wall["knot"]), 0.75 * lum(wall["bark"]))                   # the knot reads darker
        self.assertLess(lum(wall["root-hole"]), 0.5 * lum(wall["knot"]))               # the tunnel mouth darker still
        r, g, b = wall["leaves"]
        self.assertTrue(g > r and g > b, wall["leaves"])                                # green leaves


if __name__ == "__main__":
    unittest.main()
