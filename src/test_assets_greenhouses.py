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
    def test_benches_stand_on_a_floor_inside_a_dwarf_wall(self):
        footprint, ground = Polygon([(0.0, 0.0), (20.0, 0.0), (20.0, 6.0), (0.0, 6.0)]), world.datum()
        opaque, _ = mesh_greenhouses.greenhouse([footprint], lambda x, z: np.full(np.shape(x), ground))
        first = opaque.idx[:, 0]
        stone_up = np.flatnonzero((opaque.nrm[first, 1] > 0.99) & np.all(opaque.attr[first] == mesh_greenhouses.WALL_STONE, axis=1))
        top_at = lambda y: triangle_areas(opaque, stone_up[np.isclose(opaque.pos[first[stone_up], 1], y)]).sum()
        self.assertLess(top_at(ground + mesh_greenhouses.DWARF_M), 0.25 * footprint.area)   # a wall ring, not a slab
        self.assertGreater(top_at(ground), 0.5 * footprint.area)                            # the floor the benches stand on

    def test_walls_on_a_steep_slope_wind_with_their_normals(self):
        # the floor sits at the corners' median ground, so uphill the sunk base of the wall and floor rises above them
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
