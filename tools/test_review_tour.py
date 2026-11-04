"""tools.review.tour: the waypoints as the page defines them, the moment the tour passes each, which waypoints of two
pages are the same, and where each still goes on the contact sheet."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from tools.paths import PAGE
from tools.review import tour

WP_TEXT = """const DATA = 1;
  const WP = [
    [1100, -1500, 300, 0, 0, 40, "Hogwarts Castle"],
    [230, -70, 70, 10, 30, 45, null],
  ];
  const K = WP.map((w) => w);"""


def wp(caption, east):
    return [east, 0, 30, 0, 0, 0, caption]


class TourTests(unittest.TestCase):
    def test_the_waypoints_are_read_as_the_page_defines_them(self):
        self.assertEqual(tour.waypoints(WP_TEXT), [[1100, -1500, 300, 0, 0, 40, "Hogwarts Castle"],
                                                   [230, -70, 70, 10, 30, 45, None]])
        for w in tour.waypoints(PAGE.read_text(encoding="utf-8", errors="replace")):
            self.assertEqual(len(w), 7)
            self.assertTrue(w[6] is None or isinstance(w[6], str))

    def test_the_sheet_asks_for_the_meshes(self):
        # a page from before the integration leaves them off unless asked, and its stills would set its raymarched
        # objects beside the later page's meshes
        self.assertIn("&meshes=1", tour.QUERY)

    def test_each_waypoint_is_found_after_the_one_before(self):
        # out along x to 100 and back: the second waypoint, x = 20, is passed on the way out too
        x = np.concatenate([np.arange(0.0, 101.0), np.arange(99.0, -1.0, -1.0)])
        positions = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
        wps = [[80, 0, 10, 0, 0, 0, "far"], [20, 0, 10, 0, 0, 0, "back"]]
        self.assertEqual(tour.waypoint_times(positions, wps, step=0.5), [80 * 0.5, 180 * 0.5])

    def test_the_sheet_sets_each_waypoint_in_reading_order_with_its_earlier_still_on_the_left(self):
        def still(k, column):           # each still its own colour
            return f"{k + 1}. view", Image.new("RGB", (64, 36), (40 * k, 100 * column, 50))
        thumb_w, thumb_h = tour.THUMB
        cell_h = thumb_h + 28 + 10
        with tempfile.TemporaryDirectory(prefix="hog-tour-") as folder:
            single = Image.open(tour.compose(Path(folder) / "single.png", ["Page"],
                                             [[still(k, 0)] for k in range(5)])).copy()
            pairs = Image.open(tour.compose(Path(folder) / "pairs.png", ["Before", "After"],
                                            [[still(0, 1), still(0, 2)], [None, still(1, 2)],
                                             [still(2, 1), still(2, 2)]])).copy()
        self.assertEqual(single.size, (tour.SINGLES_PER_ROW * (thumb_w + 10) + 10, 36 + 2 * cell_h + 10))
        for k in range(5):
            self.assertEqual(single.getpixel((10 + (k % 4) * (thumb_w + 10) + thumb_w // 2,
                                              36 + (k // 4) * cell_h + 28 + thumb_h // 2)), (40 * k, 0, 50))
        middle = lambda k, c: (10 + (k % 2) * 2 * (thumb_w + 10) + c * (thumb_w + 10) + thumb_w // 2,
                               36 + (k // 2) * cell_h + 28 + thumb_h // 2)
        for k in (0, 2):
            self.assertEqual(pairs.getpixel(middle(k, 0)), (40 * k, 100, 50))
        for k in range(3):
            self.assertEqual(pairs.getpixel(middle(k, 1)), (40 * k, 200, 50))
        self.assertEqual(pairs.getpixel(middle(1, 0)), tour.BACKGROUND)     # the earlier page lacks the second

    def test_the_two_pages_waypoints_pair_by_caption_and_unnamed_or_repeated_ones_by_camera_point(self):
        # a waypoint added to or dropped from the tour, named or not, must not set different places side by side
        before = [wp("Castle", 0), wp(None, 10), wp("Willow", 20), wp("Shack", 30), wp(None, 40)]
        after = [wp("Castle", 0), wp(None, 10), wp("Willow", 20), wp("Hut", 35), wp(None, 38), wp("Willow", 39),
                 wp(None, 40)]
        self.assertEqual(tour.match(before, after),
                         [(0, 0), (1, 1), (2, 2), (None, 3), (None, 4), (None, 5), (4, 6), (3, None)])

    def test_chosen_views_are_the_later_pages_waypoints_each_beside_its_earlier_still(self):
        # the later page adds a waypoint before the Willow, so its second and third waypoints are the earlier page's
        # missing one and its second
        rendered = {}

        async def stills(path, gpu, scale=1, only=None):
            rendered[Path(path).name] = only
            return {k: (f"{Path(path).name} {k}", None) for k in only}
        with tempfile.TemporaryDirectory(prefix="hog-tour-") as folder:
            pages = {}
            for name, wps in (("before.html", [wp("Castle", 0), wp("Willow", 20), wp("Shack", 30)]),
                              ("after.html", [wp("Castle", 0), wp("Hut", 10), wp("Willow", 20)])):
                pages[name] = Path(folder) / name
                pages[name].write_text(f"const WP = {json.dumps(wps)};\n", encoding="utf-8")
            with mock.patch.object(tour, "stills", side_effect=stills), \
                    mock.patch.object(tour, "compose", side_effect=lambda out, titles, cells: cells):
                cells = asyncio.run(tour.make(pages["after.html"], pages["before.html"], views=[1, 2]))
        self.assertEqual(rendered, {"before.html": {1}, "after.html": {1, 2}})
        self.assertEqual(cells, [[None, ("after.html 1", None)], [("before.html 1", None), ("after.html 2", None)]])


if __name__ == "__main__":
    unittest.main()
