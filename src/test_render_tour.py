"""The tour frames what its captions name: at each waypoint whose caption names an object, that object is the visible
surface over at least MIN_SHARE of the frame as the flyover shows it, so trees, terrain and buildings in front of it
count against it. A caption that names a clearing in the Forest names its ground within the clearing's recorded outline
(meta.clearings), which, being dry ground in the Forest, must also show more than any water in its frame; and one that
names a spot the scene does not model, Sirius's cave, is held to showing that spot: in the frame, EDGE of it inside
each edge, and not hidden. A caption the test does not know fails, so a new waypoint says what it shows. The path's
clearance follows the ground and the trees, not the meshes, so the camera is also kept out of every building's box."""
from pathlib import Path
import sys
from typing import NamedTuple
import unittest

import numpy as np
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))

import assets_testing as at  # noqa: E402
from pagetest import PAGE, PageTestCase  # noqa: E402
from tools.manifest.families.village import SHOPS  # noqa: E402
from tools.review import tour  # noqa: E402


class Clearing(NamedTuple):
    """A clearing's ground, by its decision in meta.clearings."""
    decision: str


MIN_SHARE = 0.0025                  # about a 48 px square at 1280 x 720
# a higher bar where the object must stand out from its like: the Willow is a tree among trees
MINIMUM = {"Whomping Willow": 0.01}
VILLAGE = tuple(a["id"] for a in at.scene()["assets"] if a["family"] == "village")
# what each caption names: a material wherever it shows (the castle and its towers; the lake; the forest), payload
# assets, or a clearing's ground
SUBJECTS = {
    "Hogwarts Castle": "castle", "The Lake": "lake", "Durmstrang ship": ("ship",), "Gryffindor Tower": "castle",
    "Hogsmeade station": ("station",), "North Tower": "castle", "Greenhouses": ("greenhouses-north", "greenhouses-5b"),
    "Whomping Willow": ("willow",), "Forbidden Forest": "forest", "Aragog's hollow": Clearing("aragogs-hollow"),
    "Grawp's clearing": Clearing("grawps-clearing"), "Entrance Hall & Great Hall": "castle",
    "West Tower (Owlery)": "castle", "Hagrid's hut": ("hut",), "Astronomy Tower": "castle",
    "Quidditch pitch": ("pitch",), "Winged-boar gates": ("gates",), "Hogsmeade": VILLAGE,
    "Hogsmeade High Street": tuple(id for id, _, _ in SHOPS),
    "Shrieking Shack": ("shrieking-shack",), "Dumbledore's tomb": ("tomb",),
}
SPOTS = {"Sirius's cave"}           # points of interest (meta.extras.poi) on the ground that the scene does not model
EDGE = 0.1                          # a spot lies at least this share of the frame inside each of its edges
HIDDEN = 0.05                       # and the first surface along its ray within this share of its distance
MATERIALS = {"castle": 4, "lake": 2, "forest": 3}
GROUND = 1
NOT_AN_OBJECT = (0, 1, 2, 3)        # sky, ground, water and trees: an asset's box holds some of them around it
CLEAR_M = 3.0                       # how near a building's box the camera may pass
# an asset spread wider than this is many small pieces (walls, fences, bridges, track, houses), its box mostly empty
SPREAD_M = 400.0
GRID = """() => {
    const view = __hog.probe(0, 0), [w, h] = view.res, out = [];
    for (let y = 2; y < h; y += 4) for (let x = 2; x < w; x += 4) { const p = __hog.probe(x, y); out.push([x, y, p.t, p.mat]); }
    return { view, out }; }"""


def points(view, samples):
    """World points of probed (x, y, hit distance) samples, along the rays render_testing.ray gives."""
    w, h = view["res"]
    u = (2 * (samples[:, 0] + 0.5 + view["jit"][0]) - w) / h * view["tanH"]
    v = (2 * (samples[:, 1] + 0.5 + view["jit"][1]) - h) / h * view["tanH"]
    d = np.stack([u, v, np.ones_like(u)], axis=1) @ np.array(view["rot"]).reshape(3, 3)
    return np.array(view["pos"]) + d / np.linalg.norm(d, axis=1, keepdims=True) * samples[:, 2:3]


def projected(view, world):
    """The internal pixel (x, y) a world point lies at in a probed view, and its depth along the view (points
    inverted)."""
    w, h = view["res"]
    c = np.array(view["rot"], float).reshape(3, 3) @ (np.asarray(world, float) - np.asarray(view["pos"], float))
    return ((c[0] / c[2] / view["tanH"] * h + w) / 2 - 0.5 - view["jit"][0],
            (c[1] / c[2] / view["tanH"] * h + h) / 2 - 0.5 - view["jit"][1], float(c[2]))


def outline(meta, decision):
    """A clearing's outlines as recorded (meta.clearings), the islands of wood it leaves standing cut out, or None."""
    parts = [shapely.Polygon(c["outline"], c.get("islands", [])) for c in meta.get("clearings", [])
             if c["decision"] == decision]
    return shapely.union_all(parts) if parts else None


def share(frame, subject):
    """The share of the probed frame whose visible surface is `subject`: a material, payload assets, or the ground
    within an outline (a shapely geometry in east and north)."""
    samples = np.array(frame["out"], dtype=float)
    mats = samples[:, 3].astype(int)
    if isinstance(subject, str):
        return float(np.mean(mats == MATERIALS[subject]))
    p = points(frame["view"], samples)
    if isinstance(subject, shapely.Geometry):
        return float(np.mean(shapely.contains_xy(subject, p[:, 0], p[:, 2]) & (mats == GROUND)))
    inside = np.zeros(len(samples), bool)
    for asset in subject:
        corners = at.payload_part(asset)[0].reshape(-1, 3)
        inside |= np.all((p >= corners.min(0) - 0.3) & (p <= corners.max(0) + 0.3), axis=1)
    return float(np.mean(inside & ~np.isin(mats, NOT_AN_OBJECT)))


class TourFramingTests(PageTestCase):
    QUERY = "?t=0&pause=1&scale=0.35&hud=0"
    OPENING_T = 0                               # the tour's start, as the flyover shows it

    async def spot(self, meta, name):
        """Where a point of interest on the ground lies in the frame, as shares of the frame's width and height (None
        behind the camera), and the distance to the first surface along its ray over the spot's (None outside the
        frame)."""
        e, n = next(x["p"] for x in meta["extras"]["poi"] if x["name"] == name)
        point = [e, await self.page.evaluate("([x, z]) => __hog.ground(x, z)[1]", [e, n]), n]
        view = await self.page.evaluate("__hog.probe(0, 0)")
        (w, h), (x, y, depth) = view["res"], projected(view, point)
        if depth <= 0:
            return None, None, None
        if not (0 <= round(x) < w and 0 <= round(y) < h):
            return x / w, y / h, None
        hit = await self.page.evaluate("([x, y]) => __hog.probe(x, y).t", [round(x), round(y)])
        return x / w, y / h, hit / float(np.linalg.norm(np.subtract(point, view["pos"])))

    async def test_each_captioned_object_covers_its_share_of_the_frame(self):
        wps = tour.waypoints(PAGE.read_text(encoding="utf-8", errors="replace"))
        _, positions = await self.page.evaluate(tour.TRACK, tour.STEP_S)
        meta = await self.page.evaluate("DATA.meta")
        for k, (w, t) in enumerate(zip(wps, tour.waypoint_times(positions, wps))):
            caption = w[6]
            if caption is None:
                continue
            with self.subTest(waypoint=k + 1, caption=caption):
                self.assertIn(caption, SUBJECTS.keys() | SPOTS, "say what this caption names")
                frames = await self.page.evaluate("""(t) => {
                    const p = __hog.path(t);
                    __hog.camera(p.pos, Math.atan2(p.fwd[0], p.fwd[2]), Math.asin(p.fwd[1]));
                    return __hog.frames; }""", t)
                await self.page.wait_for_function(f"__hog.frames >= {frames + 12}", timeout=60000)
                if caption in SPOTS:
                    fx, fy, seen = await self.spot(meta, caption)
                    self.assertIsNotNone(fx, f"{caption} lies behind the camera at {t:.0f} s")
                    self.assertTrue(EDGE <= fx <= 1 - EDGE and EDGE <= fy <= 1 - EDGE,
                                    f"{caption} lies at ({fx:.2f}, {fy:.2f}) of the frame at {t:.0f} s")
                    self.assertGreaterEqual(seen, 1 - HIDDEN, f"something nearer hides {caption} at {t:.0f} s")
                    continue
                subject = SUBJECTS[caption]
                if isinstance(subject, Clearing):
                    subject = outline(meta, subject.decision)
                    self.assertIsNotNone(subject, f"no clearing is recorded for {caption}")
                frame = await self.page.evaluate(GRID)
                got, least = share(frame, subject), MINIMUM.get(caption, MIN_SHARE)
                self.assertGreaterEqual(got, least, f"{caption} covers {100 * got:.2f} % of the frame at {t:.0f} s")
                if isinstance(SUBJECTS[caption], Clearing):     # dry ground in the Forest, not a loch beside it
                    water = share(frame, "lake")
                    self.assertLess(water, got, f"{caption}'s frame shows more water ({100 * water:.2f} %) than its "
                                                f"ground ({100 * got:.2f} %) at {t:.0f} s")

    async def test_the_camera_keeps_out_of_every_building(self):
        _, positions = await self.page.evaluate(tour.TRACK, tour.STEP_S)
        p = np.asarray(positions, float)
        for asset in at.scene()["assets"]:
            corners = at.payload_part(asset["id"])[0].reshape(-1, 3)
            lo, hi = corners.min(0), corners.max(0)
            if max(hi[0] - lo[0], hi[2] - lo[2]) > SPREAD_M:
                continue
            with self.subTest(asset=asset["id"]):
                inside = np.flatnonzero(np.all((p >= lo - CLEAR_M) & (p <= hi + CLEAR_M), axis=1)) * tour.STEP_S
                self.assertEqual(inside.size, 0, f"the camera is inside {asset['id']}'s box"
                                 + (f" from {inside[0]:.1f} s to {inside[-1]:.1f} s" if inside.size else ""))


if __name__ == "__main__":
    unittest.main()
