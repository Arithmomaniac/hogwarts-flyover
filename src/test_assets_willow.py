"""The Willow family: a static, distinctive Whomping Willow from a seeded branch generator.

White Hound (the grounds essay) and JKR's sketch set what is checked here: a thick trunk; six or seven thick, gnarled,
sinuous limbs that start low and clench like knuckles; a large gap in the roots, at least one root curling out of the
ground; a knot low on the trunk; lower branches that do not reach the ground; asymmetric finer growth."""

from pathlib import Path
import math
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assets_testing as world  # noqa: E402
import mesh_willow  # noqa: E402
from materials import code  # noqa: E402
from mesh_source import CHANNELS, Grid, decode, sample  # noqa: E402

# the faults trimesh finds in this family's decoded meshes (assets_testing.mesh_faults), pinned: none
MESH_FAULTS = {"winding": {}, "collapsed": {}}
BARK, KNOT, HOLE, LEAVES = (code(n) for n in ("willow-bark", "willow-knot", "willow-root-hole", "willow-leaves"))
TRIANGLE_BUDGET = 50_000


def flat(x, z):
    return np.zeros(np.shape(x))


def poi(name):
    return np.array(next(p["p"] for p in world.data()["meta"]["extras"]["poi"] if p["name"] == name), float)


class Placed:
    """The decoded Willow with every vertex's height over the sampled ground."""

    def __init__(self):
        data = world.data()
        bed, grid = decode(data["tex"]["locBed"]), Grid(data["meta"]["local"])
        self.ground = lambda x, z: sample(bed, grid, x, z)
        self.at = np.array(data["meta"]["extras"]["willow"], float)
        found = world.chunks("willow")
        self.pos = np.concatenate([c["pos"] for c in found])
        self.attr = np.concatenate([c["attr"] for c in found])
        self.tris = np.concatenate([c["pos"][c["idx"]] for c in found])
        self.tri_attr = np.concatenate([c["attr"][c["idx"][:, 0]] for c in found])
        self.above = self.pos[:, 1] - self.ground(self.pos[:, 0], self.pos[:, 2])
        self.out = np.hypot(self.pos[:, 0] - self.at[0], self.pos[:, 2] - self.at[1])

    def hits(self, start, direction):
        """(distances, triangle rows) of every triangle the ray meets."""
        a, e1, e2 = self.tris[:, 0], self.tris[:, 1] - self.tris[:, 0], self.tris[:, 2] - self.tris[:, 0]
        p = np.cross(direction, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-12
        inv = 1.0 / np.where(ok, det, 1.0)
        s = start - a
        u = np.einsum("ij,ij->i", s, p) * inv
        qv = np.cross(s, e1)
        v = (qv @ direction) * inv
        t = np.einsum("ij,ij->i", e2, qv) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        return t[hit], np.flatnonzero(hit)

    def first_hit(self, start, direction):
        """The (material, aux) of the first triangle the ray meets, or None."""
        t, rows = self.hits(start, direction)
        return tuple(int(k) for k in self.tri_attr[rows[np.argmin(t)]]) if len(t) else None

    def first_distance(self, start, direction):
        t, _ = self.hits(start, direction)
        return float(t.min()) if len(t) else math.inf


_PLACED = None


def placed():
    global _PLACED
    if _PLACED is None:
        _PLACED = Placed()
    return _PLACED


def bearing(v):
    return math.atan2(v[0], v[1])           # clockwise from north, (x east, z north)


def wrapped(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def opposed_faces(mesh):
    """Faces larger than a sliver whose winding opposes their vertices' mean normal."""
    tris = mesh.pos[mesh.idx]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    big = np.linalg.norm(fn, axis=1) / 2 > world.SLIVER_M2
    return int((big & (np.einsum("ij,ij->i", fn, mesh.nrm[mesh.idx].mean(axis=1)) < 0)).sum())


class WillowSkeletonTests(unittest.TestCase):
    """The branch structure, from the generator on level ground."""

    @classmethod
    def setUpClass(cls):
        cls.tree = mesh_willow.tree(flat)
        cls.limbs = [b for b in cls.tree if b.level == 1]

    def test_six_or_seven_thick_limbs_start_low(self):
        # "thick, gnarled branches ... which probably start quite near the ground"; JKR draws six or seven
        self.assertIn(len(self.limbs), (6, 7))
        for i, limb in enumerate(self.limbs):
            with self.subTest(limb=i):
                self.assertLess(limb.pts[0, 1], 3.0)
                self.assertGreaterEqual(limb.r[0], 0.25)              # "a branch as thick as a python"
        trunk = next(b for b in self.tree if b.level == 0)
        self.assertGreaterEqual(trunk.r[np.argmin(np.abs(trunk.pts[:, 1] - 1.5))], 0.7)   # "a thick trunk"

    def test_every_limb_is_sinuous_and_knuckled(self):
        # a knuckle: a bulge in the limb's radius where the limb bends sharply
        for i, limb in enumerate(self.limbs):
            s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(limb.pts, axis=0), axis=1))])
            trend = np.interp(s, [s[0], s[-1]], [limb.r[0], limb.r[-1]])
            knuckles, sides = 0, []
            for k in range(1, len(s) - 1):
                if not (limb.r[k] >= limb.r[k - 1] and limb.r[k] >= limb.r[k + 1] and limb.r[k] > 1.15 * trend[k]):
                    continue
                before = limb.pts[k] - limb.pts[np.searchsorted(s, s[k] - 0.8)]
                after = limb.pts[min(np.searchsorted(s, s[k] + 0.8), len(s) - 1)] - limb.pts[k]
                turn = math.degrees(math.acos(np.clip(before @ after / np.linalg.norm(before) / np.linalg.norm(after), -1, 1)))
                if turn >= 20.0:
                    knuckles += 1
                    sides.append(np.sign(np.cross(before[[0, 2]], after[[0, 2]])))
            with self.subTest(limb=i):
                self.assertGreaterEqual(knuckles, 2)
                self.assertGreater(len(set(sides) - {0.0}), 1)       # it bends both ways: sinuous, not a hook

    def test_the_growth_is_asymmetric(self):
        # wood volume by bearing round the trunk: a lopsided old tree, not a radial sprig
        weights, bearings = [], []
        for b in self.tree:
            if b.level < 1:
                continue
            mid = (b.pts[1:] + b.pts[:-1]) / 2
            seg = np.linalg.norm(np.diff(b.pts, axis=0), axis=1) * ((b.r[1:] + b.r[:-1]) / 2) ** 2
            weights.append(seg)
            bearings.append(np.arctan2(mid[:, 0], mid[:, 2]))
        w, a = np.concatenate(weights), np.concatenate(bearings)
        self.assertGreater(abs(np.sum(w * np.exp(1j * a))) / np.sum(w), 0.1)

    def test_the_tree_is_reproducible(self):
        again = mesh_willow.tree(flat)
        self.assertEqual(len(again), len(self.tree))
        for a, b in zip(again, self.tree):
            np.testing.assert_array_equal(a.pts, b.pts)
            np.testing.assert_array_equal(a.r, b.r)

    def test_the_roots_open_ends_are_hidden_in_the_trunk(self):
        # a root's tube is open where it leaves the trunk; nothing culls back faces, so that end must stay inside
        trunk = next(b for b in self.tree if b.level == 0)
        lean = trunk.pts[-1][[0, 2]] / trunk.pts[-1][1]
        for i, root in enumerate(b for b in self.tree if b.level == -1):
            t = root.pts[1] - root.pts[0]
            t /= np.linalg.norm(t)
            u = np.cross(t, [0.0, 1.0, 0.0])
            u /= np.linalg.norm(u)
            v = np.cross(t, u)
            a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            ring = root.pts[0] + root.r[0] * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)
            off = ring[:, [0, 2]] - ring[:, 1:2] * lean
            inside = np.hypot(*off.T) < mesh_willow.trunk_radius(np.arctan2(off[:, 1], off[:, 0]), ring[:, 1], -math.pi / 2)
            with self.subTest(root=i):
                self.assertTrue(np.all(inside | (ring[:, 1] < 0.0)))

    def test_its_pieces_are_wound_with_their_normals_on_level_ground(self):
        # the payload's oracle sees one ground and one bearing; a fold elsewhere would show at another site
        for i, piece in enumerate(mesh_willow.meshes(self.tree, flat, -math.pi / 2)):
            with self.subTest(piece=i):
                self.assertEqual(opposed_faces(piece), 0)


class WillowTests(unittest.TestCase):
    """The built Willow in the payload, on the registered ground."""

    def test_it_stands_at_its_mapped_place_on_the_ground(self):
        w = placed()
        trunk = (w.attr[:, 1] == BARK[1]) & (w.out < 2.5) & (w.above > 1.2) & (w.above < 2.0)
        self.assertGreater(trunk.sum(), 20)
        centre = w.pos[trunk][:, [0, 2]].mean(axis=0)
        self.assertLess(np.hypot(*(centre - w.at)), 0.5)
        self.assertLess(w.above[(w.attr[:, 1] == BARK[1]) & (w.out < 1.5)].min(), -0.2)   # rooted below the ground

    def test_its_lower_branches_do_not_reach_the_ground(self):
        # "Nor do the lower branches hang right down to the ground": everything beyond the roots stands clear
        w = placed()
        beyond = w.out > mesh_willow.ROOT_ZONE_M
        self.assertGreater(beyond.sum(), 0)
        self.assertGreaterEqual(w.above[beyond].min(), 1.0)

    def test_it_stands_alone_within_its_clearing(self):
        w = placed()
        self.assertLessEqual(w.out.max(), 12.0)
        self.assertGreater(w.out.max(), 7.0)                        # a great spreading tree, not a sapling
        self.assertTrue(9.0 <= w.above.max() <= 16.0, w.above.max())

    def test_no_forest_tree_grows_into_its_crown(self):
        # "The Whomping Willow stands alone in the grounds": the map's tree density is empty as far as its crown
        # reaches plus a forest tree's crown (at most 4.9 m, frag1.glsl) from anywhere in its 7 m jittered cell
        data = world.data()
        trees = decode(data["tex"]["locA"])[..., CHANNELS["locA"].index("trees")].astype(float)
        w = placed()
        reach = w.out.max() + 4.9 + 7.0 * math.sqrt(2) / 2
        r, a = np.meshgrid(np.arange(0.0, reach, 0.5), np.linspace(0, 2 * np.pi, 96, endpoint=False))
        density = sample(trees, Grid(data["meta"]["local"]), w.at[0] + r * np.sin(a), w.at[1] + r * np.cos(a))
        self.assertEqual(float(density.max()), 0.0)

    def test_it_uses_its_own_bark_knot_hole_and_leaves(self):
        found = {tuple(int(v) for v in a) for a in placed().attr}
        self.assertEqual(found, {BARK, KNOT, HOLE, LEAVES})

    def test_the_gap_in_its_roots_faces_the_lawn_between_the_doors_and_the_hut(self):
        w = placed()
        hole = w.pos[np.all(w.attr == HOLE, axis=1)]
        facing = bearing(hole[:, [0, 2]].mean(axis=0) - w.at)
        doors, hut = (bearing(poi(n) - w.at) for n in ("Entrance Hall & Great Hall", "Hagrid's hut"))
        wedge = wrapped(hut - doors)
        self.assertTrue(0 <= wrapped(facing - doors) * np.sign(wedge) <= abs(wedge),
                        (math.degrees(facing), math.degrees(doors), math.degrees(hut)))

    def test_the_gap_admits_a_dog_dragging_a_boy_but_squeezes_an_adult(self):
        # "wide enough to admit a large dog dragging a tallish fourteen-year-old boy, but small enough to be a tight
        # squeeze for a young adult": open over at least 0.6 m, closed again within 1.1 m
        w = placed()
        hole = w.pos[np.all(w.attr == HOLE, axis=1)]
        out = hole[:, [0, 2]].mean(axis=0) - w.at
        out /= np.linalg.norm(out)
        side = np.array([-out[1], out[0]])
        for offset, expected in ((0.0, True), (0.3, True), (-0.3, True), (0.55, False), (-0.55, False)):
            x, z = w.at + 4.0 * out + offset * side
            start = np.array([x, self_ground(w, x, z) + 0.3, z])
            target = np.array([w.at[0] + offset * side[0], self_ground(w, *w.at) + 0.3, w.at[1] + offset * side[1]])
            d = (target - start) / np.linalg.norm(target - start)
            with self.subTest(offset=offset):
                self.assertEqual(w.first_hit(start, d) == HOLE, expected)

    def test_its_knot_is_low_enough_for_a_cat_or_a_rat(self):
        # "at a height accessible to a large domestic cat or an agile rat"
        w = placed()
        knot = np.all(w.attr == KNOT, axis=1)
        self.assertGreater(knot.sum(), 0)
        self.assertTrue(0.15 <= np.median(w.above[knot]) <= 0.7, np.median(w.above[knot]))
        self.assertLess(w.out[knot].max(), 2.0)

    def test_a_cat_on_the_lawn_can_reach_the_knot(self):
        # nothing (a root, say) buries the knot: seen square-on from the lawn at each point's own height, much of it shows
        w = placed()
        knot = w.pos[np.all(w.attr == KNOT, axis=1)]
        seen = 0
        for p in knot:
            out = p[[0, 2]] - w.at
            out /= np.linalg.norm(out)
            start = p + 4.0 * np.array([out[0], 0.0, out[1]])
            d = p - start
            seen += w.first_distance(start, d / np.linalg.norm(d)) > np.linalg.norm(d) - 0.02
        self.assertGreaterEqual(seen / len(knot), 0.3)

    def test_a_root_curls_out_of_the_ground(self):
        # "at least one evidently curls right out of the ground": daylight under a root beside the trunk
        w = placed()
        tris = w.tris[np.all(w.tri_attr == BARK, axis=1)]
        mid = tris.mean(axis=1)
        out = np.hypot(mid[:, 0] - w.at[0], mid[:, 2] - w.at[1])
        arch = False
        for ring in np.arange(1.7, mesh_willow.ROOT_ZONE_M, 0.2):
            near = tris[np.abs(out - ring) < 0.1]
            for a in np.linspace(0, 2 * np.pi, 180, endpoint=False):
                x, z = w.at + ring * np.array([math.sin(a), math.cos(a)])
                g = self_ground(w, x, z)
                heights = world.surface_heights(near, x, z) - g
                # a whole root over daylight: its underside clear of the ground, its top a root's thickness above
                if len(heights) and heights.min() > 0.08 and heights.max() < 1.0 and np.ptp(heights) > 0.15:
                    arch = True
        self.assertTrue(arch)

    def test_it_stays_within_its_triangle_budget(self):
        self.assertLessEqual(len(placed().tris), TRIANGLE_BUDGET)


def self_ground(w, x, z):
    return float(w.ground(np.array([x]), np.array([z]))[0])


def silhouette(mesh, axes, cell=0.1, extent=13.0):
    """Occupancy of the mesh's triangles projected onto the plane of two directions, on a cell-metre grid."""
    uv = np.stack([mesh.pos @ a for a in axes], axis=1)[mesh.idx]          # (t, 3, 2)
    n = int(2 * extent / cell)
    grid = np.zeros((n, n), bool)
    lo = np.floor((uv.min(axis=1) + extent) / cell).astype(int).clip(0, n - 1)
    hi = np.ceil((uv.max(axis=1) + extent) / cell).astype(int).clip(0, n - 1)
    span = hi - lo + 1
    a, b, c = uv[:, 0], uv[:, 1], uv[:, 2]
    d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(d) > 1e-12
    for size in np.unique(span[ok], axis=0):           # triangles with equal boxes, rasterized together
        rows = np.flatnonzero(ok & np.all(span == size, axis=1))
        di, dj = np.mgrid[0:size[0], 0:size[1]]
        i, j = lo[rows, 0, None, None] + di, lo[rows, 1, None, None] + dj
        px, py = i * cell - extent + cell / 2, j * cell - extent + cell / 2
        ar, br, cr, dr = (v[rows, ..., None, None] for v in (a, b, c, d))
        l1 = ((br[:, 1] - cr[:, 1]) * (px - cr[:, 0]) + (cr[:, 0] - br[:, 0]) * (py - cr[:, 1])) / dr
        l2 = ((cr[:, 1] - ar[:, 1]) * (px - cr[:, 0]) + (ar[:, 0] - cr[:, 0]) * (py - cr[:, 1])) / dr
        inside = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        grid[i[inside], j[inside]] = True
    return grid


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


class WillowLodTests(unittest.TestCase):
    """The coarser levels of detail and the shadow caster, from the same tree on level ground (the payload carries
    them all, and the page draws one per view)."""

    X, Y, Z = np.eye(3)

    @classmethod
    def setUpClass(cls):
        from mesh_geometry import concat
        cls.levels = {k: concat(v) for k, v in mesh_willow.levels(flat).items()}
        cls.tris = {k: len(m.idx) for k, m in cls.levels.items()}
        # seen along a low afternoon sun from the south-west
        sun = np.array([-0.55, 0.5, -0.67])
        sun /= np.linalg.norm(sun)
        u = np.cross(sun, cls.Y)
        u /= np.linalg.norm(u)
        cls.views = {"east": (cls.Z, cls.Y), "north": (cls.X, cls.Y), "above": (cls.X, cls.Z), "sun": (u, np.cross(u, sun))}
        cls.full = {name: silhouette(cls.levels["lod0"], axes) for name, axes in cls.views.items()}   # drawn once: slow

    def test_each_level_costs_a_fraction_of_the_full_tree(self):
        full = self.tris["lod0"]
        self.assertLessEqual(self.tris["lod1"], 0.35 * full)
        self.assertLessEqual(self.tris["lod2"], 0.10 * full)
        self.assertLessEqual(self.tris["shadow"], min(6000, self.tris["lod1"]))

    def test_coarser_levels_keep_the_silhouette(self):
        for level, floor in (("lod1", 0.85), ("lod2", 0.7)):
            for name in ("east", "north", "above"):
                with self.subTest(level=level, view=name):
                    self.assertGreaterEqual(iou(silhouette(self.levels[level], self.views[name]), self.full[name]),
                                            floor)

    def test_the_shadow_caster_casts_the_trees_shadow(self):
        # seen along the sun, the caster covers what the full tree covers
        self.assertGreaterEqual(iou(silhouette(self.levels["shadow"], self.views["sun"]), self.full["sun"]), 0.8)

    def test_every_level_is_wound_with_its_normals_in_its_own_codes(self):
        codes = {BARK, KNOT, HOLE, LEAVES}
        for level, m in self.levels.items():
            with self.subTest(level=level):
                self.assertEqual(opposed_faces(m), 0)
                self.assertLessEqual({tuple(int(v) for v in a) for a in m.attr}, codes)

    def test_the_levels_are_reproducible(self):
        from mesh_geometry import concat
        again = {k: concat(v) for k, v in mesh_willow.levels(flat).items()}
        for level, m in self.levels.items():
            with self.subTest(level=level):
                np.testing.assert_array_equal(again[level].pos, m.pos)
                np.testing.assert_array_equal(again[level].idx, m.idx)


class WillowMeshOracleTests(unittest.TestCase):
    """trimesh, as an independent oracle on the decoded payload."""

    def test_the_decoded_meshes_have_only_the_known_faults(self):
        self.assertEqual(world.mesh_faults("willow"), MESH_FAULTS)


if __name__ == "__main__":
    unittest.main()
