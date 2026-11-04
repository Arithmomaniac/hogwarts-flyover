"""The castle's codes in the production path: masonry with painted or real windows, slate, glass and oak; lit
panes at dusk; and the same materials mirrored in the lake."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pagetest import smoke  # noqa: E402
from render_testing import LAWN, ray, RenderTestCase  # noqa: E402


class CastleRenderTests(RenderTestCase):
    @smoke
    async def test_castle_codes_shade_as_masonry_slate_glass_and_oak(self):
        lum = lambda rgb: rgb @ [0.2126, 0.7152, 0.0722]
        dark = lambda wall: float(np.mean(lum(wall) < 0.4 * np.median(lum(wall))))
        looks = {attr: await self.castle_box(attr, 16.0) for attr in ((4, 12), (4, 22), (4, 40), (4, 42), (4, 43), (4, 44))}
        painted, real = looks[(4, 12)], looks[(4, 22)]
        self.assertGreater(dark(painted["wall"]), 0.05)        # the legacy class code keeps its painted windows
        with self.subTest(code="class with real openings"):   # the same masonry and slate roof, without painted windows
            self.assertLess(dark(real["wall"]), 0.005)
            self.assertAlmostEqual(np.median(lum(real["wall"])) / np.median(lum(painted["wall"])), 1.0, delta=0.1)
            np.testing.assert_allclose(real["top"].mean(axis=0), painted["top"].mean(axis=0), rtol=0.1)
        masonry = np.median(lum(looks[(4, 43)]["wall"]))
        with self.subTest(code="reveal masonry"):            # never slate on top, never painted windows
            self.assertLess(dark(looks[(4, 43)]["wall"]), 0.005)
            top = looks[(4, 43)]["top"].mean(axis=0)
            self.assertGreater(top[0], top[2])
        with self.subTest(code="exposed masonry"):           # the same, without the recess's loss of sky
            self.assertLess(dark(looks[(4, 44)]["wall"]), 0.005)
            top = looks[(4, 44)]["top"].mean(axis=0)
            self.assertGreater(top[0], top[2])
            self.assertGreater(np.median(lum(looks[(4, 44)]["wall"])), 1.02 * masonry)
        with self.subTest(code="glass"):                     # dark, but for the Fresnel sky it reflects
            self.assertLess(np.median(lum(looks[(4, 40)]["wall"])), 0.3 * masonry)
        with self.subTest(code="oak"):
            oak = np.median(looks[(4, 42)]["wall"], axis=0)
            self.assertTrue(oak[0] > oak[1] > oak[2], oak)
            self.assertLess(lum(oak), 0.6 * masonry)

    async def test_lit_panes_glow_at_dusk(self):
        for hour in np.arange(19.0, 23.0, 0.25):
            if await self.page.evaluate("(h) => __hog.light(h).dusk", float(hour)) > 0.99:
                break
        x, z = LAWN
        ground = (await self.page.evaluate("([x, z]) => __hog.ground(x, z)", [x, z]))[1]
        frames = await self.page.evaluate("(h) => { __hog.tod(h); return __hog.frames; }", float(hour))
        await self.page.wait_for_function(f"__hog.frames >= {frames + 10}")
        box = (x + 30.0, ground - 1.0, z - 12.0, x + 50.0, ground + 22.0, z + 12.0)
        glow = {}
        for attr in ((4, 40), (4, 41)):
            await self.show([box], at=(x - 15.0, z), height=45.0, pitch=-0.5, attr=attr)
            glow[attr] = await self.page.evaluate("""() => {
                const [w, h] = __hog.probe(0, 0).res, out = [];
                for (let y = 1; y < h; y += 3) for (let x = 1; x < w; x += 3) {
                  const p = __hog.probe(x, y), c = __hog.probeColor(x, y);
                  if (p.mat === 4 && p.n[0] < -0.9) out.push(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]);
                }
                return out; }""")
            self.assertGreater(len(glow[attr]), 200)
        self.assertGreater(np.median(glow[(4, 41)]), 4.0 * np.median(glow[(4, 40)]))

    async def test_reflections_take_the_castle_materials(self):
        view = await self.lake_view(-0.6)
        await self.show([], **view)
        water = await self.centre()
        w, h = water["res"]
        origin, direction = ray(water, w // 2, h // 2)
        centre = origin + direction * water["t"] + direction * np.array([1.0, -1.0, 1.0]) * 100.0
        box = (*(centre - 15.0), *(centre + 15.0))
        colours = {}
        for attr in ((4, 12), (4, 22), (4, 40), (4, 43)):
            await self.show([box], attr=attr, **view)
            colours[attr] = await self.reflected()
        np.testing.assert_allclose(colours[(4, 22)], colours[(4, 12)], rtol=0.05)   # the same mirrored masonry
        np.testing.assert_allclose(colours[(4, 43)], colours[(4, 12)], rtol=0.05)
        self.assertLess(colours[(4, 40)].sum(), 0.2 * colours[(4, 43)].sum())       # glass is dark


if __name__ == "__main__":
    unittest.main()
