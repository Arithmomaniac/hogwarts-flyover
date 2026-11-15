"""The greenhouse family: at least three glasshouses with frames, benches and plants, and fittings in plain materials."""

from pathlib import Path
import sys
import unittest

import numpy as np
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assets_testing as world  # noqa: E402
import mesh_greenhouses  # noqa: E402
from assets_testing import triangle_areas  # noqa: E402

# the faults trimesh finds in this family's decoded meshes today (assets_testing.mesh_faults), pinned: a new fault
# fails, and so does a fix until this is updated
MESH_FAULTS = {"winding": {}, "collapsed": {}}


class GreenhouseTests(unittest.TestCase):
    @staticmethod
    def stone_tops(opaque):
        """{height: area} of the stone's upward faces: the floor and the dwarf wall's top."""
        first = opaque.idx[:, 0]
        stone_up = np.flatnonzero((opaque.nrm[first, 1] > 0.99) & np.all(opaque.attr[first] == mesh_greenhouses.WALL_STONE, axis=1))
        heights = np.round(opaque.pos[first[stone_up], 1], 4)
        return {float(y): float(triangle_areas(opaque, stone_up[heights == y]).sum()) for y in np.unique(heights)}

    def test_benches_stand_on_a_floor_inside_a_dwarf_wall(self):
        footprint, ground = Polygon([(0.0, 0.0), (20.0, 0.0), (20.0, 6.0), (0.0, 6.0)]), world.datum()
        opaque, _ = mesh_greenhouses.greenhouse([footprint], lambda x, z: np.full(np.shape(x), ground))
        tops = self.stone_tops(opaque)
        floor = max(tops, key=tops.get)
        self.assertGreater(tops[floor], 0.5 * footprint.area)                               # the floor the benches stand on
        self.assertLess(tops.get(round(floor + mesh_greenhouses.DWARF_M, 4), 0.0), 0.25 * footprint.area)   # a wall ring
        self.assertAlmostEqual(floor, ground + mesh_greenhouses.FLOOR_GAP_M, places=6)

    def test_the_floor_stands_clear_of_the_ground_under_it(self):
        # the floor is level with the highest ground inside the dwarf wall, a little above it. Laid at the corners'
        # median ground, it lay under the lawn where the ground rose inside, which then showed through it, and where the
        # two met (a terrace graded to it) each showed in turn, in rings
        footprint = Polygon([(0.0, 0.0), (20.0, 0.0), (20.0, 6.0), (0.0, 6.0)])
        base = world.datum()
        ground = lambda x, z: base + 0.1 * np.asarray(x) + 0.3 * np.exp(-((np.asarray(x) - 12.0) ** 2 + (np.asarray(z) - 3.0) ** 2) / 4.0)
        opaque, _ = mesh_greenhouses.greenhouse([footprint], ground)
        tops = self.stone_tops(opaque)
        floor = max(tops, key=tops.get)
        x, z = np.meshgrid(np.linspace(0.3, 19.7, 98), np.linspace(0.3, 5.7, 28))
        self.assertLessEqual(float(ground(x, z).max()), floor - 0.5 * mesh_greenhouses.FLOOR_GAP_M)
        self.assertGreater(float(ground(x, z).max()), floor - 2.0 * mesh_greenhouses.FLOOR_GAP_M, "no higher than it needs")

    def test_the_dwarf_wall_meets_the_ground_under_a_raised_floor(self):
        # on a slope and over a hump the floor stands level with the highest ground inside, up to 0.6 m above the lowest:
        # the dwarf wall round it still runs down into the ground along every face, so no gap opens beneath it
        footprint = Polygon([(0.0, 0.0), (20.0, 0.0), (20.0, 6.0), (0.0, 6.0)])
        base = world.datum()
        ground = lambda x, z: base + 0.015 * np.asarray(x) + 0.3 * np.exp(-((np.asarray(x) - 12.0) ** 2 + (np.asarray(z) - 3.0) ** 2) / 4.0)
        opaque, _ = mesh_greenhouses.greenhouse([footprint], ground)
        tops = self.stone_tops(opaque)
        floor = max(tops, key=tops.get)
        self.assertGreater(floor - float(ground(np.array([0.0]), np.array([0.0]))[0]), 0.5, "the floor is not raised here")
        tri = opaque.pos[opaque.idx]
        stone = np.all(opaque.attr[opaque.idx[:, 0]] == mesh_greenhouses.WALL_STONE, axis=1)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        side = stone & (np.abs(n[:, 1]) < 1e-6 * np.linalg.norm(n, axis=1))
        u, v = np.meshgrid(np.linspace(0.0, 1.0, 41), np.linspace(0.0, 1.0, 41))
        inside = (u + v <= 1.0).ravel()
        u, v = u.ravel()[inside, None, None], v.ravel()[inside, None, None]
        f = tri[side]
        pts = (f[None, :, 0] + u * (f[None, :, 1] - f[None, :, 0]) + v * (f[None, :, 2] - f[None, :, 0])).reshape(-1, 3)
        cell = np.round(pts[:, [0, 2]] / 0.25).astype(np.int64)            # each 0.25 m of the upright faces' plan
        keys, at = np.unique(cell, axis=0, return_inverse=True)
        bottom = np.full(len(keys), np.inf)
        np.minimum.at(bottom, at.ravel(), pts[:, 1])
        xz = keys * 0.25
        self.assertLess(float(np.max(bottom - ground(xz[:, 0], xz[:, 1]))), -0.5)   # each runs down into the ground

    def test_walls_on_a_steep_slope_wind_with_their_normals(self):
        # on a steep slope the sunk bases of the wall and floor run far below them downhill and close under them uphill
        footprint, ground = Polygon([(0.0, 0.0), (20.0, 0.0), (20.0, 6.0), (0.0, 6.0)]), world.datum()
        opaque, _ = mesh_greenhouses.greenhouse([footprint], lambda x, z: ground + 0.5 * np.asarray(x))   # 1 in 2
        self.assertEqual(int(world.turned(opaque.pos, opaque.idx, opaque.nrm).sum()), 0)

    def test_a_potting_shed_on_a_steep_slope_winds_with_its_normals(self):
        # a shed's floor sits at its corners' median ground too, so uphill its sunk walls rose above their eaves
        footprint, ground = Polygon([(0.0, 0.0), (20.0, 0.0), (20.0, 6.0), (0.0, 6.0)]), world.datum()
        shed = mesh_greenhouses.shed([footprint], lambda x, z: ground + 0.5 * np.asarray(x))   # 1 in 2
        self.assertEqual(int(world.turned(shed.pos, shed.idx, shed.nrm).sum()), 0)

    def test_fittings_take_no_building_windows_or_roofs(self):
        # P3 paints windows on building walls (material 5 except the glasshouse kind) and slate on their upper faces
        footprint, ground = Polygon([(0.0, 0.0), (20.0, 0.0), (20.0, 6.0), (0.0, 6.0)]), world.datum()
        opaque, _ = mesh_greenhouses.greenhouse([footprint], lambda x, z: np.full(np.shape(x), ground))
        self.assertEqual({tuple(int(v) for v in a) for a in opaque.attr if a[0] == 5}, {mesh_greenhouses.FRAME})

    def test_at_least_three_glasshouses_with_frames_benches_and_plants(self):
        chunks = [c for c in world.payload()["chunks"]
                  if c["asset"] in world.family_assets("greenhouses")]
        panes = np.concatenate([c["pos"][c["idx"]] for c in chunks if c.get("pass") == "glass"])
        boxes = np.concatenate([panes[:, :, [0, 2]].min(axis=1) - 0.1, panes[:, :, [0, 2]].max(axis=1) + 0.1], axis=1)
        house = list(range(len(boxes)))                     # panes whose footprints touch belong to one house
        def root(i):
            while house[i] != i:
                i = house[i]
            return i
        for i, j in zip(*np.nonzero((boxes[:, None, 0] <= boxes[None, :, 2]) & (boxes[None, :, 0] <= boxes[:, None, 2])
                                    & (boxes[:, None, 1] <= boxes[None, :, 3]) & (boxes[None, :, 1] <= boxes[:, None, 3]))):
            house[root(i)] = root(j)
        houses = {}
        for i in range(len(boxes)):
            b = houses.setdefault(root(i), boxes[i].copy())
            b[:2], b[2:] = np.minimum(b[:2], boxes[i, :2]), np.maximum(b[2:], boxes[i, 2:])
        self.assertGreaterEqual(len(houses), 3)
        for x0, z0, x1, z1 in houses.values():
            found = {tuple(int(v) for v in a) for c in chunks if c.get("pass") != "glass"
                     for a in c["attr"][(c["pos"][:, 0] >= x0) & (c["pos"][:, 0] <= x1)
                                        & (c["pos"][:, 2] >= z0) & (c["pos"][:, 2] <= z1)]}
            with self.subTest(house=(round(x0), round(z0))):
                self.assertLessEqual({mesh_greenhouses.FRAME, mesh_greenhouses.BENCH, mesh_greenhouses.FOLIAGE}, found)


class GreenhouseMeshOracleTests(unittest.TestCase):
    """trimesh, as an independent oracle on the decoded payload."""

    def test_the_decoded_meshes_have_only_the_known_faults(self):
        self.assertEqual(world.mesh_faults("greenhouses"), MESH_FAULTS)


if __name__ == "__main__":
    unittest.main()
