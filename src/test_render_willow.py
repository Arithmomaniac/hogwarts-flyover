"""The Willow family's codes in the production path: bark, knot, tunnel mouth and leaves."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from materials import code  # noqa: E402
from pagetest import smoke  # noqa: E402
from render_testing import RenderTestCase  # noqa: E402


class WillowRenderTests(RenderTestCase):
    @smoke
    async def test_willow_codes_shade_as_bark_knot_mouth_and_leaves(self):
        # without their own cases these codes fall through to the sails' pale canvas
        lum = lambda rgb: np.asarray(rgb) @ [0.2126, 0.7152, 0.0722]
        codes = {"sail": (7, 24), **{name: code(f"willow-{name}") for name in ("bark", "knot", "root-hole", "leaves")}}
        looks = {name: await self.castle_box(attr, 16.0) for name, attr in codes.items()}
        wall = {name: np.median(v["wall"], axis=0) for name, v in looks.items()}
        self.assertLess(lum(wall["bark"]), 0.8 * lum(wall["sail"]))                    # dark wood
        self.assertGreater(wall["bark"][0], wall["bark"][2])                           # grey-brown, not blue
        self.assertLess(lum(wall["knot"]), 0.75 * lum(wall["bark"]))                   # the knot reads darker
        self.assertLess(lum(wall["root-hole"]), 0.5 * lum(wall["knot"]))               # the tunnel mouth darker still
        r, g, b = wall["leaves"]
        self.assertTrue(g > r and g > b, wall["leaves"])                                # green leaves


if __name__ == "__main__":
    unittest.main()
