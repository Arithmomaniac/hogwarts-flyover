"""The transport family's codes in the production path: paint, iron, glass, steel and stone; the carriage,
station and ship lights at dusk; and their reflections."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pagetest import smoke  # noqa: E402
from render_testing import ray, RenderTestCase  # noqa: E402


class TransportRenderTests(RenderTestCase):
    @smoke
    async def test_transport_codes_shade_as_paint_iron_glass_steel_and_stone(self):
        lum = lambda rgb: np.asarray(rgb) @ [0.2126, 0.7152, 0.0722]
        codes = {"scarlet": (7, 10), "iron": (7, 11), "glass": (7, 13), "rail": (7, 15), "ballast": (7, 17),
                 "paving": (7, 18), "boarding": (7, 19), "slate": (7, 20), "door": (7, 21), "void": (7, 22),
                 "port": (7, 23), "sail": (7, 24), "trim": (7, 25)}
        looks = {name: await self.castle_box(attr, 16.0) for name, attr in codes.items()}
        wall = {name: np.median(v["wall"], axis=0) for name, v in looks.items()}
        top = {name: np.median(v["top"], axis=0) for name, v in looks.items()}
        r, g, b = wall["scarlet"]
        self.assertTrue(r > 4 * g and r > 4 * b, wall["scarlet"])                        # scarlet livery
        bright = lum(wall["boarding"])
        self.assertTrue(wall["boarding"][0] > wall["boarding"][2], wall["boarding"])       # cream boarding
        for name in ("iron", "glass", "void", "port"):
            with self.subTest(code=name):
                self.assertLess(lum(wall[name]), 0.2 * bright)
        self.assertLess(lum(wall["void"]), 0.5 * lum(wall["iron"]))                        # the tunnel's dark mouth
        self.assertGreater(lum(top["rail"]), 2.0 * lum(wall["rail"]))                      # polished heads, rusty sides
        for name in ("door", "trim"):
            with self.subTest(code=name):
                self.assertTrue(wall[name][1] > wall[name][0] and wall[name][1] > wall[name][2], wall[name])   # green paint
        self.assertGreater(top["slate"][2], top["slate"][0])                               # blue-grey slate
        for name in ("ballast", "paving", "sail"):
            with self.subTest(code=name):
                self.assertTrue(lum(wall["iron"]) < lum(wall[name]) < bright, (name, wall[name]))

    async def test_carriage_station_and_ship_lights_glow_at_dusk(self):
        for hour in np.arange(19.0, 23.0, 0.25):
            if await self.page.evaluate("(h) => __hog.light(h).dusk", float(hour)) > 0.99:
                break
        lum = lambda rgb: np.asarray(rgb) @ [0.2126, 0.7152, 0.0722]
        looks = {attr: lum(np.median((await self.castle_box(attr, float(hour), sunlit=False))["wall"], axis=0))
                 for attr in ((7, 11), (7, 13), (7, 23))}
        self.assertGreater(looks[(7, 13)], 4.0 * looks[(7, 11)])          # lit windows
        self.assertGreater(looks[(7, 23)], 4.0 * looks[(7, 11)])          # the ship's misty ports

    async def test_reflections_take_the_transport_materials(self):
        view = await self.lake_view(-0.6)
        await self.show([], **view)
        water = await self.centre()
        w, h = water["res"]
        origin, direction = ray(water, w // 2, h // 2)
        centre = origin + direction * water["t"] + direction * np.array([1.0, -1.0, 1.0]) * 100.0
        box = (*(centre - 15.0), *(centre + 15.0))
        colours = {}
        for attr in ((7, 10), (7, 13), (7, 14), (7, 19)):
            await self.show([box], attr=attr, **view)
            colours[attr] = await self.reflected()
        r, g, b = colours[(7, 10)]
        self.assertTrue(r > 3 * g and r > 3 * b, colours[(7, 10)])                         # a scarlet train mirrors red
        for glass in ((7, 13), (7, 14)):                                                    # lamplit and dimly lit
            with self.subTest(code=glass):
                self.assertLess(colours[glass].sum(), 0.2 * colours[(7, 19)].sum())       # glass is dark


if __name__ == "__main__":
    unittest.main()
