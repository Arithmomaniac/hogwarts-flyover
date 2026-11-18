"""The greenhouse family: at least three glasshouses with frames, benches and plants, and fittings in plain materials."""

from pathlib import Path
import sys
import unittest

import numpy as np
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assets_testing as world  # noqa: E402
import mesh_greenhouses  # noqa: E402
import railway  # noqa: E402
from assets_testing import triangle_areas  # noqa: E402
from mesh_source import building_components, load_georef  # noqa: E402

SHED = Polygon([(0.0, 0.0), (8.0, 0.0), (8.0, 5.0), (0.0, 5.0)])
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

    def test_benches_stand_on_legs_under_rows_of_plants_and_the_glass_reads_as_panes(self):
        # the benches read as black slabs with no legible plants, the glazing as an empty skeleton (greenhouses-5b-10)
        footprint, ground = Polygon([(0.0, 0.0), (20.0, 0.0), (20.0, 6.0), (0.0, 6.0)]), world.datum()
        opaque, _ = mesh_greenhouses.greenhouse([footprint], lambda x, z: np.full(np.shape(x), ground))
        tris, attr, n, area = faces(opaque)
        bench = np.all(attr == mesh_greenhouses.BENCH, axis=1)
        self.assertLess(area[bench & (np.abs(n[:, 1]) < 0.1)].sum(), 0.3 * 2 * 19.0)    # open under the top: legs, no sides
        plants = np.unique(tris[np.all(attr == mesh_greenhouses.FOLIAGE, axis=1)].reshape(-1, 3), axis=0)
        flat = np.hypot(plants[:, None, 0] - plants[None, :, 0], plants[:, None, 2] - plants[None, :, 2])
        apex = plants[~np.any((flat < 0.3) & (plants[None, :, 1] > plants[:, None, 1] + 1e-6), axis=1)]   # each plant's crown
        self.assertGreaterEqual(len({round(float(z), 1) for z in apex[:, 2]}), 4)          # two rows on each bench
        self.assertGreater(float(np.ptp(apex[:, 1])), 0.2)                                # of plants of different heights
        frame = np.all(attr == mesh_greenhouses.FRAME, axis=1)
        rails = {round(float(t[:, 1].mean()) - ground, 1) for t in tris[frame] if np.ptp(t[:, 0]) > 15.0}
        self.assertGreaterEqual(len(rails), 5)                                            # rails and purlins, not just eaves and ridge

    def test_each_bench_leg_stands_under_its_bench_top(self):
        # the legs stood just outside the top, the outer row in the dwarf wall and over its sill (code review, item 22)
        footprint, ground = Polygon([(0.0, 0.0), (20.0, 0.0), (20.0, 6.0), (0.0, 6.0)]), world.datum()
        opaque, _ = mesh_greenhouses.greenhouse([footprint], lambda x, z: np.full(np.shape(x), ground))
        tris, attr, _, _ = faces(opaque)
        bench = tris[np.all(attr == mesh_greenhouses.BENCH, axis=1)]
        low = bench[:, :, 1].max(axis=1) < ground + mesh_greenhouses.BENCH_H - mesh_greenhouses.BENCH_TOP_M + 1e-6
        tops = bench[~low].reshape(-1, 3)
        for side in (tops[:, 2] < 3.0, tops[:, 2] > 3.0):
            z0, z1 = tops[side, 2].min(), tops[side, 2].max()
            legs = bench[low].reshape(-1, 3)
            legs = legs[(legs[:, 2] - 3.0) * (z0 - 3.0) > 0]
            with self.subTest(side=round(float(z0), 1)):
                self.assertTrue(np.all((legs[:, 2] >= z0) & (legs[:, 2] <= z1)))

    def test_a_wing_runs_towards_the_range_it_joins_and_opens_into_it(self):
        ground = world.datum()
        wing = Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)])           # wider than it is long
        _, glass = mesh_greenhouses.greenhouse([wing], lambda x, z: np.full(np.shape(x), ground), open_towards=(2.5, 20.0))
        tris, _, n, area = faces(glass)
        ridge = glass.pos[glass.pos[:, 1] > glass.pos[:, 1].max() - 1e-6]
        self.assertTrue(np.allclose(ridge[:, 0], 2.5) and np.ptp(ridge[:, 2]) > 3.9)   # the ridge runs to the range
        far = (np.abs(n[:, 2]) > 0.9) & np.all(tris[:, :, 2] > 3.9, axis=1)
        # the range's side glazing closes the end below its eave; above, the wing's gable stood open (code review)
        self.assertAlmostEqual(float(area[far].sum()), 2.5 * 2.5 * mesh_greenhouses.PITCH, places=6)
        self.assertGreaterEqual(float(tris[far][:, :, 1].min()), ground + mesh_greenhouses.EAVE_M - 1e-6)

    def test_the_court_glasshouse_is_the_t_the_map_draws(self):
        # it was one rectangle turned across the lawn (court-glasshouse, round 1)
        traced = next(a for a in world.scene()["assets"] if a["id"] == "court-glasshouse")["traced"][0]["plan_px"]
        rng, wing = (Polygon(railway.grounds_world(np.asarray(traced[k], np.float64), load_georef())) for k in ("range", "wing"))
        panes = np.concatenate([c["pos"][c["idx"]] for c in world.payload()["chunks"]
                                if c["asset"] == "court-glasshouse" and c.get("pass") == "glass"])
        n = np.cross(panes[:, 1] - panes[:, 0], panes[:, 2] - panes[:, 0])
        roof = panes[np.abs(n[:, 1]) > 0.3 * np.linalg.norm(n, axis=1)]
        covered = lambda x, z: bool(np.any([Polygon(t[:, [0, 2]]).buffer(1e-6).contains(Point(x, z)) for t in roof]))
        far = np.asarray(wing.exterior.coords[:4])
        tip = far[np.argsort([Point(*p).distance(rng) for p in far])[-2:]].mean(axis=0)   # the wing's outer end
        tip = tip + (np.asarray(wing.centroid.coords[0]) - tip) * 0.3
        for name, (x, z) in {"range": rng.centroid.coords[0], "wing": tip}.items():
            with self.subTest(part=name):
                self.assertTrue(covered(x, z))
        self.assertFalse(covered(*(np.asarray(rng.centroid.coords[0]) + 2.0 * (np.asarray(rng.centroid.coords[0]) - tip))))

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


def faces(mesh):
    """(triangles, attrs, unit normals, areas) of a built mesh."""
    tris = mesh.pos[mesh.idx]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = np.linalg.norm(n, axis=1)
    return tris, mesh.attr[mesh.idx[:, 0]], n / area[:, None], area / 2


class PottingShedTests(unittest.TestCase):
    def test_a_shed_has_a_door_towards_the_glasshouses_and_windows_of_its_own(self):
        # building walls take the shading pass's painted windows, which the eaves cut to slits wherever the ground falls
        # (potting-shed-1) and which left a thatched shed a blank box with dark slashes (greenhouses-north-4)
        ground = world.datum()
        shed = mesh_greenhouses.shed([SHED], lambda x, z: np.full(np.shape(x), ground), towards=(4.0, 30.0))
        tris, attr, n, area = faces(shed)
        self.assertFalse(np.any((attr[:, 0] == 5) & (np.abs(n[:, 1]) < 0.25)), "a building kind's painted walls")
        glass = np.all(attr == mesh_greenhouses.SHED_GLASS, axis=1)
        for side in (1.0, -1.0):
            with self.subTest(side=side):
                self.assertGreaterEqual(area[glass & (n[:, 2] * side > 0.99)].sum(),
                                        2 * mesh_greenhouses.WINDOW_W * mesh_greenhouses.WINDOW_H - 1e-6)
        timber = np.all(attr == mesh_greenhouses.BENCH, axis=1)
        low = tris[:, :, 1].min(axis=1) < ground + 1e-6
        door = timber & low & (n[:, 2] > 0.99)
        self.assertGreaterEqual(float(np.ptp(tris[door][:, :, 1])), mesh_greenhouses.DOOR_MIN_M, "no door towards +z")
        self.assertFalse(np.any(timber & low & (n[:, 2] < -0.99)), "a door away from the glasshouses")

    def test_on_falling_ground_each_window_is_whole_and_clear_of_the_ground(self):
        base = world.datum()
        ground = lambda x, z: base + 0.12 * np.asarray(x)                    # 1 in 8 along the long walls
        shed = mesh_greenhouses.shed([SHED], ground)
        tris, attr, n, area = faces(shed)
        glass = np.all(attr == mesh_greenhouses.SHED_GLASS, axis=1)
        panes = area[glass].sum() / (mesh_greenhouses.WINDOW_W * mesh_greenhouses.WINDOW_H)
        self.assertGreaterEqual(panes, 4 - 1e-6)
        rect = mesh_greenhouses.fit_rectangle(SHED)
        head = mesh_greenhouses.floor_level(rect, ground) + mesh_greenhouses.SHED_EAVE_M - mesh_greenhouses.HEAD_M
        self.assertLessEqual(float(tris[glass][:, :, 1].max()), head + 1e-6)   # masonry over every head, under the eaves
        corners = tris[glass].reshape(-1, 3)
        self.assertGreaterEqual(float(np.min(corners[:, 1] - ground(corners[:, 0], corners[:, 2]))),
                                mesh_greenhouses.WINDOW_SILL_M - 1e-6)
        roof = (attr[:, 0] == 5) & (n[:, 1] > 0.25)
        self.assertLess(float(corners[:, 1].max()), float(tris[roof][:, :, 1].min()))

    def test_every_claimed_footprint_stands_as_its_kind(self):
        # greenhouses-north-4 is the group's one thatched footprint, its potting shed: the smallest glasshouse is -6
        kinds = {c["id"]: c["kind"] for c in building_components()}
        chunks = [c for c in world.payload()["chunks"] if c["asset"] in world.family_assets("greenhouses")]
        def found(attr, x, z, glass):                    # a face of `attr` spans the footprint's centre in plan
            for c in chunks:
                if (c.get("pass") == "glass") != glass:
                    continue
                tris = c["pos"][c["idx"][np.all(c["attr"][c["idx"][:, 0]] == attr, axis=1)]][:, :, [0, 2]]
                if np.any(np.all((tris.min(axis=1) - 0.5 <= (x, z)) & ((x, z) <= tris.max(axis=1) + 0.5), axis=1)):
                    return True
            return False
        claimed = [c for a in world.scene()["assets"] if a["family"] == "greenhouses" for c in a.get("components", [])]
        self.assertIn("thatch", {kinds[c["id"]] for c in claimed})
        for c in claimed:
            with self.subTest(component=c["id"], kind=kinds[c["id"]]):
                if kinds[c["id"]] == "glass":
                    self.assertTrue(found(mesh_greenhouses.GLASS, *c["at"], True))
                else:
                    roof = mesh_greenhouses.THATCH if kinds[c["id"]] == "thatch" else mesh_greenhouses.SLATE
                    self.assertTrue(found(roof, *c["at"], False))
                    self.assertTrue(any(np.any(np.all(k["attr"] == mesh_greenhouses.SHED_GLASS, axis=1)
                                               & (np.hypot(*(k["pos"][:, [0, 2]] - c["at"]).T) < 10.0)) for k in chunks))

    def test_no_wall_in_the_family_takes_painted_windows(self):
        for c in world.payload()["chunks"]:
            if c["asset"] not in world.family_assets("greenhouses") or c.get("pass") == "glass":
                continue
            tris = c["pos"][c["idx"]]
            n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            upright = np.abs(n[:, 1]) < 0.25 * np.linalg.norm(n, axis=1)
            attr = c["attr"][c["idx"][:, 0]]
            with self.subTest(asset=c["asset"]):
                self.assertLessEqual({tuple(int(v) for v in a) for a in attr[upright & (attr[:, 0] == 5)]},
                                     {mesh_greenhouses.FRAME})

class GreenhouseMeshOracleTests(unittest.TestCase):
    """trimesh, as an independent oracle on the decoded payload."""

    def test_the_decoded_meshes_have_only_the_known_faults(self):
        self.assertEqual(world.mesh_faults("greenhouses"), MESH_FAULTS)


if __name__ == "__main__":
    unittest.main()
