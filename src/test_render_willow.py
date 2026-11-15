"""The Willow family's codes in the production path: bark, knot, tunnel mouth and leaves."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from materials import code  # noqa: E402
from materials.willow import BARK_AXES  # noqa: E402
from pagetest import smoke  # noqa: E402
from render_testing import EAST, LAWN, RenderTestCase  # noqa: E402


class WillowRenderTests(RenderTestCase):
    async def close_wall(self, attr, hour=16.0):
        """Luminance (an internal-pixel grid, NaN off the wall) of the sunlit west face of a box tagged `attr`, 3.5 m in
        front of a camera at eye height: close enough to see the bark."""
        x, z = LAWN[0] + 26.0, LAWN[1]           # where castle_box's sunlit west wall stands: no shadow falls there
        ground = await self.ground(x, z)
        frames = await self.page.evaluate("(h) => { __hog.tod(h); return __hog.frames; }", hour)
        await self.page.wait_for_function(f"__hog.frames >= {frames + 10}")
        await self.show([(x + 3.5, ground - 0.5, z - 2.0, x + 4.5, ground + 4.0, z + 2.0)], at=(x, z), yaw=EAST,
                        pitch=0.15, height=1.6, attr=attr)
        found = await self.page.evaluate("""([aux]) => {
            const [w, h] = __hog.probe(0, 0).res, s = Math.min(w, h) >> 2, x0 = (w >> 1) - s, y0 = (h >> 1) - s;
            const g = __hog.probeRect(x0, y0, 2 * s, 2 * s), c = __hog.probeColorRect(x0, y0, 2 * s, 2 * s), out = [];
            let lit = 0, shown = 0;
            for (let i = 0; i < 4 * s * s; i++) {
              const ok = g[i][0] === 7 && Math.abs(g[i][4] - aux) < 0.5 && g[i][1] < -0.9;
              out.push(ok ? 0.2126 * c[4 * i] + 0.7152 * c[4 * i + 1] + 0.0722 * c[4 * i + 2] : NaN);
              if (ok && i % 97 === 0) { shown++; lit += __hog.probeSun(x0 + i % (2 * s), y0 + Math.floor(i / (2 * s))) > 0.95; }
            }
            const mats = {}; for (const r of g) mats[r[0] + ":" + Math.round(r[4]) + ":" + Math.round(r[1])] = 1;
            return { side: 2 * s, lum: out, lit: lit / Math.max(shown, 1), shown, mats: Object.keys(mats).slice(0, 12) }; }""",
            [attr[1]])
        self.assertGreater(found["lit"], 0.95, f"the test wall is not in sunlight: {found['shown']} samples, {found['mats']}")
        grid = np.array([np.nan if v is None else v for v in found["lum"]], float).reshape(found["side"], -1)
        self.assertGreater(np.isfinite(grid).mean(), 0.5, "the test wall does not fill the probed window")
        return grid

    async def test_bark_close_up_shows_lit_plates_and_dark_furrows_along_the_wood(self):
        # the trunk, buttresses and roots read as smooth charcoal close up unless the bark has grey-brown plates split by
        # deep furrows, and the furrows run along the wood: up an upright trunk, along a lying root
        steps = lambda grid, axis: np.nanmean(np.abs(np.diff(grid, axis=axis)))   # axis 0: up the wall, 1: across it
        sail = np.nanmean(await self.close_wall((7, 24)))
        lying = next(aux for aux, axis in BARK_AXES.items() if abs(axis[2]) > 0.99)   # furrows along z: across the view
        for name, attr, along in (("upright", code("willow-bark"), 0), ("lying", (7, lying), 1)):
            grid = await self.close_wall(attr)
            lum = grid[np.isfinite(grid)]
            with self.subTest(bark=name):
                self.assertGreater(lum.mean(), 0.62 * sail)                            # grey-brown, not charcoal
                self.assertLess(np.percentile(lum, 10), 0.5 * np.percentile(lum, 75))  # deep furrows between plates
                self.assertLess(steps(grid, along), 0.65 * steps(grid, 1 - along))     # smoother along the furrows

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
