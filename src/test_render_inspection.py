"""The qualification's shots are what the page displays (the plan's platform/output case): at internal scale 0.5 the
shot tool's 1920 x 1080 frame at device pixel ratio 1 is drawn from a 960 x 540 internal image, and its screenshot is
the displayed frame, pixel for pixel, after the temporal filter, not an internal buffer."""
import io
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))

from pagetest import PageTestCase  # noqa: E402
from tools.inspection import shoot  # noqa: E402

SPOTS = [(fx, fy) for fx in (0.1, 0.3, 0.5, 0.7, 0.9) for fy in (0.1, 0.3, 0.5, 0.7, 0.9)]
MATCH = 3                           # grey levels: the screenshot and the read-back frame are the same pixels


class DisplayedShotTests(PageTestCase):
    QUERY = shoot.QUERY.format(scale=f"{shoot.SCALE:g}")
    VIEWPORT = {"width": shoot.W, "height": shoot.H}

    async def test_a_half_scale_shot_is_the_full_size_displayed_frame(self):
        seen = await self.page.evaluate(shoot.OBSERVED)
        self.assertEqual((seen["dpr"], seen["viewport"], seen["canvas"]), (1, [shoot.W, shoot.H], [shoot.W, shoot.H]))
        self.assertEqual(seen["internal"], [round(shoot.W * shoot.SCALE), round(shoot.H * shoot.SCALE)])
        frames = await self.page.evaluate("__hog.frames")
        await self.page.wait_for_function(f"__hog.frames >= {frames + shoot.SETTLE_FRAMES}", timeout=180000)
        shot = np.asarray(Image.open(io.BytesIO(await self.page.screenshot())).convert("RGB"), int)
        self.assertEqual(shot.shape[:2], (shoot.H, shoot.W))
        shown = await self.page.evaluate("(spots) => spots.map(([x, y]) => __hog.probeScreen(x, y))", SPOTS)
        for (fx, fy), rgb in zip(SPOTS, shown):
            with self.subTest(spot=(fx, fy)):          # probeScreen counts rows from the bottom, as GL does
                pixel = shot[shoot.H - 1 - int(fy * shoot.H), int(fx * shoot.W)]
                self.assertLessEqual(int(np.abs(pixel - np.array(rgb)).max()), MATCH, f"{pixel} against {rgb}")


if __name__ == "__main__":
    unittest.main()
