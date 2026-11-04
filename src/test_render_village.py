"""The village family's codes in the production path: thatch, slate, harl, rubble and paint by day, and its windows'
lamplight at dusk."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from materials import code  # noqa: E402
from pagetest import smoke  # noqa: E402
from render_testing import RenderTestCase  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722])


class VillageRenderTests(RenderTestCase):
    @smoke
    async def test_village_codes_shade_as_thatch_slate_harl_rubble_and_paint(self):
        names = ("village-thatch", "village-slate", "harl-white", "harl-ochre", "harl-blue", "harl-terracotta",
                 "rubble", "door-red", "sail")
        wall = {n: np.median((await self.castle_box(code(n), 16.0))["wall"], axis=0) for n in names}
        lum = {n: float(c @ LUM) for n, c in wall.items()}
        for n in names[:-1]:
            with self.subTest(code=n):
                self.assertFalse(np.allclose(wall[n], wall["sail"], rtol=0.05))     # not the unregistered fallback
        self.assertEqual(max(lum[n] for n in names[2:6]), lum["harl-white"])          # white harl is the brightest wall
        r, g, b = wall["harl-ochre"]
        self.assertTrue(r > g > b, wall["harl-ochre"])                                 # yellow ochre
        self.assertGreater(wall["harl-blue"][2], wall["harl-blue"][0])                 # blue
        self.assertGreater(wall["harl-terracotta"][0], 1.4 * wall["harl-terracotta"][1])
        self.assertGreater(wall["door-red"][0], 3.0 * wall["door-red"][1])            # a red door
        self.assertLess(lum["village-slate"], lum["rubble"])                           # dark slate over grey stone
        self.assertGreater(wall["village-thatch"][0], wall["village-thatch"][2])      # warm straw
        self.assertLess(lum["village-thatch"], lum["harl-white"])

    async def test_village_windows_glow_with_lamplight_at_dusk_and_shops_brightest(self):
        hour = await self.light_at(lambda light: light["dusk"] > 0.99)
        lum = {n: float(np.median((await self.castle_box(code(n), hour, sunlit=False))["wall"], axis=0) @ LUM)
               for n in ("glass-dark", "glass-lit", "glass-shop")}
        self.assertGreater(lum["glass-lit"], 4.0 * lum["glass-dark"])
        self.assertGreater(lum["glass-shop"], lum["glass-lit"])


if __name__ == "__main__":
    unittest.main()
