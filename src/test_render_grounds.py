"""The grounds family's codes in the production path: creosote, shingles, pumpkin and earth."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from materials import code  # noqa: E402
from pagetest import smoke  # noqa: E402
from render_testing import RenderTestCase  # noqa: E402


class GroundsRenderTests(RenderTestCase):
    @smoke
    async def test_grounds_codes_shade_as_creosote_shingles_pumpkin_and_earth(self):
        # without their own cases these codes fall through to the sails' pale canvas
        lum = lambda rgb: np.asarray(rgb) @ [0.2126, 0.7152, 0.0722]
        codes = {"sail": (7, 24), "boards": (7, 26), "shingles": (7, 27), "pumpkin": (7, 28), "earth": (7, 29),
                 "planks": (7, 30)}
        looks = {name: await self.castle_box(attr, 16.0) for name, attr in codes.items()}
        wall = {name: np.median(v["wall"], axis=0) for name, v in looks.items()}
        r, g, b = wall["pumpkin"]
        self.assertTrue(r > 1.3 * g and g > 1.3 * b, wall["pumpkin"])                  # orange
        for name in ("boards", "shingles", "earth"):
            with self.subTest(code=name):
                self.assertLess(lum(wall[name]), 0.8 * lum(wall["sail"]))              # dark timber and soil
                self.assertGreater(wall[name][0], wall[name][2])                        # warm browns
        self.assertGreater(lum(wall["shingles"]), lum(wall["boards"]))                  # weathered shingles over creosote
        self.assertGreater(lum(wall["planks"]), 1.5 * lum(wall["boards"]))             # doors that read against the boards

    async def test_pitch_codes_shade_as_gold_whitewash_and_green_doors(self):
        # pitch-changing-room-materials : whitewashed boarding and dark green doors; the hoops' gold
        lum = lambda rgb: np.asarray(rgb) @ [0.2126, 0.7152, 0.0722]
        codes = {"gold": code("gold"), "planks": code("weathered-planks"), "boarding": code("changing-room-boarding"),
                 "door": code("changing-room-door")}
        looks = {name: await self.castle_box(attr, 16.0) for name, attr in codes.items()}
        wall = {name: np.median(v["wall"], axis=0) for name, v in looks.items()}
        r, g, b = wall["gold"]
        self.assertTrue(r > g > 1.5 * b, wall["gold"])                                   # yellow metal
        self.assertGreater(lum(wall["boarding"]), 1.5 * lum(wall["planks"]))            # whitewash over bare timber
        r, g, b = wall["door"]
        self.assertTrue(g > r and g > b, wall["door"])                                   # painted green
        self.assertLess(lum(wall["door"]), lum(wall["planks"]))

    async def test_the_boars_shade_as_weathered_stone_beside_their_pillars(self):
        # gates-pillars-boars : the boars a darker, weathered grey than the pale stone of the pillars
        lum = lambda rgb: np.asarray(rgb) @ [0.2126, 0.7152, 0.0722]
        looks = {name: await self.castle_box(code(name), 16.0) for name in ("pale-stone", "boar-stone", "sail")}
        wall = {name: np.median(v["wall"], axis=0) for name, v in looks.items()}
        r, g, b = wall["boar-stone"]
        self.assertLess(max(r, g, b) - min(r, g, b), 0.25 * max(r, g, b), wall)       # a neutral grey, not a colour
        self.assertLess(lum(wall["boar-stone"]), lum(wall["pale-stone"]), wall)
        cool = lambda rgb: rgb[2] / rgb[0]
        self.assertGreater(cool(wall["boar-stone"]), cool(wall["sail"]) + 0.06, wall)  # not the warm fallback (sail)

    async def test_the_harbour_and_bridge_codes_shade_as_rock_ivy_and_shadow(self):
        lum = lambda rgb: np.asarray(rgb) @ [0.2126, 0.7152, 0.0722]
        looks = {name: await self.castle_box(code(name), 16.0) for name in ("harbour-rock", "ivy", "arch-shadow", "sail")}
        wall = {name: np.median(v["wall"], axis=0) for name, v in looks.items()}
        r, g, b = wall["harbour-rock"]
        self.assertLess(max(r, g, b) - min(r, g, b), 0.3 * max(r, g, b), wall)        # a grey rock
        r, g, b = wall["ivy"]
        self.assertTrue(g > r and g > b, wall)                                           # green leaves
        self.assertLess(lum(wall["arch-shadow"]), 0.2 * lum(wall["harbour-rock"]), wall)   # the dark under an arch
        self.assertLess(lum(wall["harbour-rock"]), lum(wall["sail"]), wall)              # its own code, not the fallback


if __name__ == "__main__":
    unittest.main()
