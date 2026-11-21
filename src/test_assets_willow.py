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
import materials.willow as bark  # noqa: E402
from materials.willow import BARK_AXES, LYING  # noqa: E402
from mesh_source import CHANNELS, Grid, decode, sample  # noqa: E402

# the faults trimesh finds in this family's decoded meshes (assets_testing.mesh_faults), pinned: none
MESH_FAULTS = {"winding": {}, "collapsed": {}}
BARK, KNOT, HOLE, LEAVES = (code(n) for n in ("willow-bark", "willow-knot", "willow-root-hole", "willow-leaves"))
BARKS = {(7, aux) for aux in BARK_AXES}         # the bark, once for each direction its furrows run in


def is_bark(attr):
    return (attr[..., 0] == 7) & np.isin(attr[..., 1], list(BARK_AXES))
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

    def test_the_knots_dark_wood_fills_its_eye(self):
        # the dark knot-hole is the burl's sunken middle, not a patch beside it, however the trunk leans
        gap = -math.pi / 2
        trunk = next(b for b in self.tree if b.level == 0)
        site = mesh_willow.knot_site(flat, gap)
        m = mesh_willow.knot_eye(trunk, gap, site)
        lean = trunk.pts[-1] / trunk.pts[-1][1]
        knot = m.pos[np.all(m.attr == KNOT, axis=1)]
        ring = knot @ lean / (lean @ lean)                     # each vertex's height along the leaning axis
        off = knot - ring[:, None] * lean
        turn = wrapped(np.arctan2(off[:, 2], off[:, 0]) - site[0])
        self.assertLess(abs(ring.mean() - site[1]), 0.01)
        self.assertLess(abs(turn.mean()), math.radians(1))

    def test_the_eye_is_a_smooth_dish(self):
        # close round its middle the eye's floor is the same depth whichever way you look, not a crumpled star
        gap = -math.pi / 2
        site = mesh_willow.knot_site(flat, gap)
        reach = mesh_willow.TRUNK_R * (1 + mesh_willow.FLARE * math.exp(-site[1] / mesh_willow.FLARE_M))
        for d in (0.005, 0.02):
            a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            r = mesh_willow.trunk_radius(site[0] + d * np.cos(a) / reach, site[1] + d * np.sin(a), gap, site)
            # the trunk's flare and ridges tilt the floor (a plane over a small circle); a crumple is what is left
            basis = np.stack([np.ones_like(a), np.cos(a), np.sin(a)], axis=1)
            residual = r - basis @ np.linalg.lstsq(basis, r, rcond=None)[0]
            with self.subTest(d=d):
                self.assertLess(np.ptp(residual), 0.005)

    def test_the_burl_swells_round_a_sunken_knot(self):
        # a healed branch scar: the trunk's own wood swells into a burl, and the knot sits sunk in its middle
        gap = -math.pi / 2
        trunk = next(b for b in self.tree if b.level == 0)
        site = mesh_willow.knot_site(flat, gap)
        m = mesh_willow.trunk_mesh(trunk, gap, knot=site)
        lean = trunk.pts[-1][[0, 2]] / trunk.pts[-1][1]
        rel = m.pos[:, [0, 2]] - m.pos[:, 1:2] * lean                      # off the leaning axis, at each height
        out, heading = np.hypot(*rel.T), np.arctan2(rel[:, 0], rel[:, 1])
        centre = mesh_willow.knot_eye(trunk, gap, site).pos[0]              # the eye's middle
        crel = centre[[0, 2]] - centre[1] * lean
        away = -np.sign(wrapped(math.atan2(math.cos(gap), math.sin(gap)) - math.atan2(*crel)))
        turn = wrapped(heading - math.atan2(*crel)) * away
        level = np.abs(m.pos[:, 1] - centre[1]) < 0.08
        side = np.median(out[level & (turn > math.radians(45)) & (turn < math.radians(90))])
        rim = out[level & (np.linalg.norm(m.pos - centre, axis=1) < 0.3)].max()
        eye = math.hypot(*crel)
        self.assertGreaterEqual(rim - side, 0.06)                                      # the burl stands proud
        # and the knot is sunk in it: its middle stands lower than the burl around it (the lobes averaged away)
        reach = mesh_willow.TRUNK_R * (1 + mesh_willow.FLARE * math.exp(-site[1] / mesh_willow.FLARE_M))
        relief = lambda th, y: mesh_willow.trunk_radius(th, y, gap, site) - mesh_willow.trunk_radius(th, y, gap)
        a, d = np.meshgrid(np.linspace(0, 2 * np.pi, 36, endpoint=False), (0.08, 0.1, 0.12, 0.15))
        around = relief(site[0] + d * np.cos(a) / reach, site[1] + d * np.sin(a)).mean()
        self.assertGreaterEqual(around - float(relief(np.array([site[0]]), np.array([site[1]]))[0]), 0.01)

    def test_its_pieces_are_wound_with_their_normals_on_level_ground(self):
        # the payload's oracle sees one ground and one bearing; a fold elsewhere would show at another site
        for i, piece in enumerate(mesh_willow.meshes(self.tree, flat, -math.pi / 2)):
            with self.subTest(piece=i):
                self.assertEqual(opposed_faces(piece), 0)


class WillowTests(unittest.TestCase):
    """The built Willow in the payload, on the registered ground."""

    def test_it_stands_at_its_mapped_place_on_the_ground(self):
        w = placed()
        trunk = is_bark(w.attr) & (w.out < 2.5) & (w.above > 1.2) & (w.above < 2.0)
        self.assertGreater(trunk.sum(), 20)
        centre = w.pos[trunk][:, [0, 2]].mean(axis=0)
        self.assertLess(np.hypot(*(centre - w.at)), 0.5)
        self.assertLess(w.above[is_bark(w.attr) & (w.out < 1.5)].min(), -0.2)            # rooted below the ground

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
        self.assertLessEqual({BARK, KNOT, HOLE, LEAVES}, found)
        self.assertLessEqual(found, BARKS | {KNOT, HOLE, LEAVES})

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
        tris = w.tris[is_bark(w.tri_attr)]
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

    def test_the_knot_lies_flush_in_the_trunks_eye(self):
        # "a knot on the trunk", "the knot in the bark": the dark knot-hole lies in the burl's sunken eye, within a couple
        # of centimetres of the bark around it, not a ball set against the trunk
        w = placed()
        knot = w.pos[np.all(w.attr == KNOT, axis=1)]
        bark = w.pos[is_bark(w.attr) & (w.out < 2.0) & (w.above < 1.5)]
        self.assertGreater(len(knot), 0)
        nearest = np.array([np.linalg.norm(bark - k, axis=1).min() for k in knot])
        self.assertLess(nearest.max(), 0.035)

    def test_the_knots_outline_is_round(self):
        # seen square-on, the dark eye is a round hole, not a patch of the mesh's cells with straight sides
        w = placed()
        knot = np.all(w.attr == KNOT, axis=1)
        centre = w.pos[knot].mean(axis=0)
        out = np.array([centre[0] - w.at[0], 0.0, centre[2] - w.at[1]])
        out /= np.linalg.norm(out)
        side, up = np.array([-out[2], 0.0, out[0]]), np.array([0.0, 1.0, 0.0])
        near = np.linalg.norm(w.tris.mean(axis=1) - centre, axis=1) < 0.4
        tris, attr = w.tris[near], w.tri_attr[near]
        a, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
        step = 0.003
        s, v = np.meshgrid(np.arange(-0.1, 0.1 + 1e-9, step), np.arange(-0.1, 0.1 + 1e-9, step))
        covered = np.zeros(s.shape, bool)
        p = np.cross(-out, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-12
        inv = 1.0 / np.where(ok, det, 1.0)
        for idx in np.ndindex(s.shape):
            start = centre + 0.5 * out + s[idx] * side + v[idx] * up
            q = start - a
            u = np.einsum("ij,ij->i", q, p) * inv
            qv = np.cross(q, e1)
            vv = (qv @ -out) * inv
            tt = np.einsum("ij,ij->i", e2, qv) * inv
            hit = ok & (u >= 0) & (vv >= 0) & (u + vv <= 1) & (tt > 0)
            if hit.any():
                covered[idx] = np.all(attr[np.flatnonzero(hit)[np.argmin(tt[hit])]] == KNOT)
        self.assertGreater(covered.sum(), 50)
        r, ang = np.hypot(s, v)[covered], np.arctan2(v, s)[covered]
        bins = np.floor((ang + np.pi) / (2 * np.pi) * 16).astype(int) % 16
        reach = np.array([r[bins == b].max() for b in range(16) if (bins == b).any()])
        self.assertEqual(len(reach), 16)
        self.assertLess(reach.max() / reach.min(), 1.15, np.round(reach, 3))

    def test_the_roots_bark_runs_along_them(self):
        # each bark vertex's code names the direction its furrows run; on the roots, that is out from the trunk
        w = placed()
        roots = is_bark(w.attr) & (w.out > 1.7) & (w.out < mesh_willow.ROOT_ZONE_M) & (w.above < 0.7)
        self.assertGreater(roots.sum(), 200)
        radial = np.stack([w.pos[roots, 0] - w.at[0], np.zeros(roots.sum()), w.pos[roots, 2] - w.at[1]], axis=1)
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        axes = np.array([BARK_AXES[int(a)] for a in w.attr[roots, 1]])
        off = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", radial, axes)), 0, 1)))
        self.assertLessEqual(np.percentile(off, 90), 25.0)


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
        codes = BARKS | {KNOT, HOLE, LEAVES}
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


def jitter(q):
    """The bark shader's cell centres (materials.willow._JITTER): each cell's centre, relative to the cell's corner."""
    hq = (q * np.array([0.1031, 0.103, 0.0973])) % 1.0
    hq = hq + np.sum(hq * (hq[:, [1, 0, 2]] + 33.33), axis=1, keepdims=True)
    return 0.5 + bark.JITTER_AMP * (((hq[:, [0, 0, 1]] + hq[:, [1, 0, 0]]) * hq[:, [2, 1, 0]]) % 1.0 - 0.5)


def border(m1, rs):
    """The distance from the point to the nearest border of the cell whose centre is m1 against the centres rs."""
    b = rs - m1[:, None]
    bb = np.einsum("nij,nij->ni", b, b)
    eb = np.einsum("nij,nij->ni", 0.5 * (m1[:, None] + rs), b) / np.sqrt(np.maximum(bb, 1e-12))
    return np.min(np.where(bb > 1e-6, eb, np.inf), axis=1)


def bark_net(S, reach=1):
    """The bark shader's net (materials.willow.BARK_P3) at points S in its cell space: each point's distance to the
    nearest cell border, in cells, found as the shader does (the nearest centre, then the nearest border round it).
    `reach` 2 searches 5x5x5 cells both times: the exact net, for any jitter."""
    span = np.arange(-reach, reach + 1, dtype=float)
    offsets = np.stack(np.meshgrid(span, span, span, indexing="ij"), -1).reshape(-1, 3)
    c = np.floor(S)
    f = S - c
    rs = np.stack([o + jitter(c + o) - f for o in offsets], axis=1)
    near = np.argmin(np.einsum("nij,nij->ni", rs, rs), axis=1)
    m1, mo = rs[np.arange(len(S)), near], offsets[near]
    if reach > 1:
        return border(m1, rs)
    return border(m1, np.stack([mo + o + jitter(c + mo + o) - f for o in offsets], axis=1))


def ridge(e):
    """The bark's height over the furrows' floor (0) to the ridges' crests (1), from the distance to the border."""
    w = np.clip((e - bark.RIDGE_E) / bark.RIDGE_W, 0, 1)
    return 1 - w * w * (3 - 2 * w)


def bark_space(p, n, aux):
    """The bark shader's cell space (materials.willow.BARK_P3) at points p with normals n on wood of codes aux: round
    the trunk's axis on the trunk, round the normal on every other piece of wood, and along the code's direction."""
    dot = lambda a, b: np.einsum("ij,ij->i", a, b)
    aux = np.asarray(aux)
    d = np.array([BARK_AXES[int(a)] for a in aux])
    e1 = np.cross(d, np.where(np.abs(d[:, 1:2]) > 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]))
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d, e1)
    x0, y0, z0, lx, lz = bark.TRUNK_AXIS
    y0 += world.levels()["lake"]                         # as the shader takes it from LAKE
    trunk = aux == BARK[1]
    o = np.where(trunk[:, None], p - [x0, y0, z0] - np.outer(p[:, 1] - y0, [lx, 1.0, lz]), n - d * dot(n, d)[:, None])
    o /= np.maximum(np.linalg.norm(o, axis=1, keepdims=True), 1e-4)
    th, hy = np.arctan2(o[:, 2], o[:, 0]), p[:, 1] - y0
    wobble = 1 + sum(a * np.sin(n * th + k * hy + ph) for a, n, k, ph in bark.WOBBLE)
    r = np.where(trunk, bark.ROUND[BARK[1]] * wobble, bark.ROUND["wood"])
    along = dot(p - [x0, y0, z0], d)
    shift = np.where(trunk, 0.0, bark.LIMB_SHIFT)
    return (np.stack([r * dot(o, e1), along * bark.STRETCH, r * dot(o, e2)], axis=1) * bark.CELL_PER_M
            + shift[:, None] * np.stack([dot(p - [x0, y0, z0], e1), np.zeros(len(p)), dot(p - [x0, y0, z0], e2)], axis=1))


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def bark_lattices(p, n, aux):
    """The bark shader's cell spaces at points p with normals n on wood of codes aux, each with its weight: on the trunk,
    its own space at the two rungs of the ladder its girth lies between, and the hollow's walls' space (their distance
    from the axis across them, their height along them) as the wood turns side-on to the axis; on every other piece of
    wood, its own space alone."""
    dot = lambda a, b: np.einsum("ij,ij->i", a, b)
    S = bark_space(p, n, aux)
    x0, y0, z0, lx, lz = bark.TRUNK_AXIS
    y0 += world.levels()["lake"]
    o = p - [x0, y0, z0] - np.outer(p[:, 1] - y0, [lx, 1.0, lz])
    rho = np.linalg.norm(o, axis=1)
    oh = o / np.maximum(rho, 1e-4)[:, None]
    twist = np.abs(dot(n, np.cross([0.0, 1.0, 0.0], oh))) / np.maximum(np.abs(dot(n, oh)), 1e-3)
    trunk = np.asarray(aux) == BARK[1]
    wall = np.where(trunk, smoothstep(*bark.WALL_TWIST, twist), 0.0)
    sharp = 1 + bark.WALL_SHARP * smoothstep(*bark.WALL_SLOPE, np.abs(n[:, 1]))
    wall = np.clip((wall - 0.5) * sharp + 0.5, 0, 1)
    L = np.clip(np.log(np.maximum(rho / bark.GIRTH_M, 1.0)) / math.log(bark.RUNG), 0, bark.RUNGS)
    i = np.where(trunk, np.minimum(np.floor(L), bark.RUNGS - 1), 0.0)
    f = np.where(trunk, smoothstep(*bark.RUNG_BAND, L - i), 0.0)
    walls = np.stack([rho * bark.CELL_PER_M, S[:, 1], np.arctan2(oh[:, 2], oh[:, 0]) * bark.WALL_SPREAD], axis=1)
    rung = lambda s: S * np.array([s, 1.0, s])
    # one space to each rung (so a triangle's corners either side of a rung's edge read the same space)
    return [(rung(bark.RUNG ** r), (1 - wall) * np.where(i == r, 1 - f, np.where(i + 1 == r, f, 0.0)))
            for r in range(bark.RUNGS + 1)] + [(walls, wall)]


def bark_height(p, n, aux):
    """The bark's height over the furrows' floor (0) to the ridges' crests (1) at points p (as bark_space), close up:
    where cell spaces crossfade, each one's borders where it is nearer than the others by MERGE times its share."""
    E, top = np.full(len(p), np.inf), np.zeros(len(p))
    for S, w in bark_lattices(p, n, aux):
        on = w > 1e-3
        e = np.full(len(p), np.inf)
        e[on] = bark_net(S[on])
        E = np.minimum(E, e + (1 - w) * bark.MERGE)
        top = np.maximum(top, np.where(on, w, 0.0))
    return ridge(E - (1 - top) * bark.MERGE)


def round_trunk(a, y, r=1.0):
    """Points (and their normals) on a circular trunk r metres round the Willow's axis, at bearings a, y over its base."""
    x0, y0, z0, lx, lz = bark.TRUNK_AXIS
    n = np.stack([np.cos(a), np.zeros_like(a), np.sin(a)], -1)
    return np.array([x0, y0 + world.levels()["lake"], z0]) + np.outer(y, [lx, 1.0, lz]) + np.asarray(r)[..., None] * n, n


def bark_patch(width_m=1.0, height_m=2.4, res_m=0.005, start=0.0):
    """The bark's height over a patch of an upright trunk 1 m in radius, from bearing `start`: rows up the trunk,
    columns round it."""
    a = np.arange(start, start + width_m, res_m)
    y = np.arange(0.3, 0.3 + height_m, res_m)
    A, Y = np.meshgrid(a, y)
    p, n = round_trunk(A.ravel(), Y.ravel())
    return bark_height(p, n, np.full(len(p), BARK[1])).reshape(A.shape)


_WOOD = None


def placed_wood():
    """The Willow's bark triangles from the generator, placed as the page places them: (corners (t, 3, 3), corner
    normals (t, 3, 3), codes (t,), heights of their middles over the ground (t,), distances from the trunk's axis (t,))."""
    global _WOOD
    if _WOOD is None:
        from mesh_geometry import concat
        w = placed()
        x0, z0 = w.at
        y0 = self_ground(w, x0, z0)
        ground = lambda x, z: w.ground(np.asarray(x) + x0, np.asarray(z) + z0) - y0
        gap = mesh_willow.gap_bearing(world.data()["meta"])
        m = concat(mesh_willow.meshes(mesh_willow.tree(ground, gap), ground, gap))
        keep = is_bark(m.attr[m.idx[:, 0]])
        tris = m.pos[m.idx[keep]] + [x0, y0, z0]
        mid = tris.mean(axis=1)
        _WOOD = (tris, m.nrm[m.idx[keep]], m.attr[m.idx[keep, 0], 1].astype(int),
                 mid[:, 1] - w.ground(mid[:, 0], mid[:, 2]), np.hypot(mid[:, 0] - x0, mid[:, 2] - z0))
    return _WOOD


def surface_map(tris, S):
    """Each triangle's area, its two directions in the world (t1, t2), and the map from steps along them to the change of
    cell coordinates S (at its corners): (t, 2, 3)."""
    u, v = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    fn = np.cross(u, v)
    area = np.linalg.norm(fn, axis=1) / 2
    t1 = u / np.linalg.norm(u, axis=1, keepdims=True)
    t2 = np.cross(fn / (2 * area[:, None]), t1)
    edges = np.stack([np.stack([np.einsum("ij,ij->i", e, t) for t in (t1, t2)], -1)
                      for e in (u, v)], axis=1)
    return area, t1, t2, np.linalg.solve(edges, np.stack([S[:, 1] - S[:, 0], S[:, 2] - S[:, 0]], axis=1))


def cells_on(tris, nrms, aux):
    """Each triangle's area, and the bark's cells per metre over it across (the most) and along (the least) its
    furrows, from the cell space at its corners (the one weighted most over them, where spaces crossfade)."""
    spaces = [bark_lattices(tris[:, k], nrms[:, k], aux) for k in range(3)]
    main = np.argmax(np.stack([sum(spaces[k][j][1] for k in range(3)) for j in range(len(spaces[0]))], axis=1), axis=1)
    S = np.stack([np.choose(main[:, None], [space[0] for space in spaces[k]]) for k in range(3)], axis=1)
    area, _, _, J = surface_map(tris, S)
    sv = np.linalg.svd(J, compute_uv=False)
    return area, sv[:, 0], sv[:, 1]


def furrows_on(tris, nrms, aux):
    """For each of bark_lattices' cell spaces, each triangle's furrow direction in the world: the way its cells change least."""
    spaces = [bark_lattices(tris[:, k], nrms[:, k], aux) for k in range(3)]
    out = []
    for j in range(len(spaces[0])):
        _, t1, t2, J = surface_map(tris, np.stack([spaces[k][j][0] for k in range(3)], axis=1))
        least = np.linalg.svd(J)[0][:, :, 1]
        out.append(least[:, :1] * t1 + least[:, 1:] * t2)
    return out


def weighted_median(x, w):
    order = np.argsort(x)
    return float(x[order][np.searchsorted(np.cumsum(w[order]), w.sum() / 2)])


class WillowBarkTests(unittest.TestCase):
    def test_the_bark_fades_to_its_own_mean_far_off(self):
        # past the fade the shader uses FAR_H in place of the ridges and furrows; were it not what the
        # pattern averages to, the whole tree would brighten or darken as the camera draws back
        rng = np.random.default_rng(7)
        n = 40000
        # a long span of wood: a trunk's cells wrap round it, some 90 to a ring, so a short one samples too few
        a, y = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 30.0, n)
        p, nrm = round_trunk(a, y, rng.uniform(0.7, 1.6, n))
        lying = BARK_AXES[LYING]
        e1 = np.cross(lying, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        nl = np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * np.cross(lying, e1)
        pl = np.array([-254.3, world.datum() + 2.0, 184.2]) + rng.uniform(-15, 15, n)[:, None] * lying + 0.3 * nl
        for name, p, nrm, aux in (("trunk", p, nrm, BARK[1]), ("lying", pl, nl, LYING)):
            with self.subTest(wood=name):
                self.assertAlmostEqual(float(np.mean(bark_height(p, nrm, np.full(n, aux)))), bark.FAR_H, delta=0.03)

    def test_the_ridges_run_unbroken_round_the_furrows(self):
        # the shader's 3x3x3 search round the nearest centre must find the border it lies by; where it misses one, the
        # ridge steps to the next border's height at a hard edge
        S = np.random.default_rng(3).uniform(0, 1, (30000, 3)) * [60.0, 20.0, 60.0] + [6600.0, 1500.0, 4800.0]
        shader, exact = bark_net(S), bark_net(S, reach=2)
        self.assertLess(float(np.mean(np.abs(shader - exact) > 0.01)), 0.002)

    def test_the_trunks_ridges_part_and_meet_round_its_furrows(self):
        # cells wrapped round the trunk at one depth line up as straight, parallel stripes, their edges all along the
        # wood; old willow bark's ridges wander, part and meet again round each furrow, so some of their edges lean across
        # it. Pooled over four patches round the trunk: the share of the edges leaning more than 10 degrees
        leaning = []
        for a in (0.0, 1.6, 3.2, 4.8):
            rows, cols = np.gradient(bark_patch(start=a, res_m=0.004))      # up the trunk, round it
            g = np.hypot(rows, cols)
            edge = g > np.percentile(g, 80)
            leaning.append(np.mean(np.degrees(np.arctan2(np.abs(rows[edge]), np.abs(cols[edge]))) > 10))
        self.assertGreater(float(np.mean(leaning)), 0.006)

    def test_the_furrows_are_long_spindles_between_interlacing_ridges(self):
        # old willow bark: dark furrows, each a separate spindle ten or more times as long as it is wide, between long
        # grey ridges that part and meet round them, not a dark net round pale scales (which reads as snakeskin)
        import scipy.ndimage as ndi
        h = bark_patch()
        furrow = h < 0.5
        self.assertTrue(0.3 < furrow.mean() < 0.6, furrow.mean())          # wide furrows, not cracks
        labels, n = ndi.label(furrow)
        sizes = np.bincount(labels.ravel())[1:]
        self.assertLess(sizes.max() / sizes.sum(), 0.25)                 # separate spindles, not one dark net
        long = []
        for k, sl in enumerate(ndi.find_objects(labels), 1):
            if sizes[k - 1] >= 30:
                long.append((sl[0].stop - sl[0].start) / (sl[1].stop - sl[1].start))
        self.assertGreater(float(np.median(long)), 6.0)


class WillowBarkSpaceTests(unittest.TestCase):
    """Where the bark's cells lie on the built tree's wood. A pixel 4 m off (PIX_M wide) shows the pattern where it
    spans fewer than FADE[0] cells and no less than COARSE[0] of the trunk's."""

    PIX_M = 0.0064

    @classmethod
    def setUpClass(cls):
        tris, nrms, cls.aux, cls.above, cls.out = placed_wood()
        cls.tris, cls.nrms = tris, nrms
        cls.area, cls.across, cls.along = cells_on(tris, nrms, cls.aux)
        shown = (cls.across * cls.PIX_M < bark.FADE[0]) & (cls.across > bark.COARSE[0] * bark.CELL_PER_M)
        cls.shown = shown & (cls.above > 0.02)                  # above the ground
        # each triangle's bearing round the trunk's axis from the hollow's, and how far its face turns round the trunk
        w = placed()
        mid = tris.mean(axis=1)
        _, _, _, lx, lz = bark.TRUNK_AXIS
        rise = mid[:, 1] - self_ground(w, *w.at)
        off = mid[:, [0, 2]] - w.at - np.outer(rise, [lx, lz])
        theta = np.arctan2(off[:, 1], off[:, 0])
        gap = mesh_willow.gap_bearing(world.data()["meta"])
        cls.from_gap = np.abs(np.degrees(wrapped(theta - gap)))
        cls.in_gap = mesh_willow.gap_cut(theta, rise, gap) > 0.02
        face = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        face /= np.linalg.norm(face, axis=1, keepdims=True)
        cls.side_on = np.abs(face[:, 0] * -np.sin(theta) + face[:, 2] * np.cos(theta))

    def width(self, m):
        """The furrows' width over the triangles m (area-weighted median), metres."""
        return weighted_median(1 / self.across[m], self.area[m])

    def trunk(self):
        return self.shown & (self.aux == BARK[1]) & (self.above > 1.0) & (self.above < 2.0)

    def test_the_trunk_axis_is_the_built_trunks(self):
        w = placed()
        x0, z0 = w.at
        y0 = self_ground(w, x0, z0)
        ground = lambda x, z: w.ground(np.asarray(x) + x0, np.asarray(z) + z0) - y0
        trunk = next(b for b in mesh_willow.tree(ground, mesh_willow.gap_bearing(world.data()["meta"])) if b.level == 0)
        # from the ground to the fork, the trunk keeps within 12 cm of the bark's axis (the bearings round it then vary
        # the furrows' width round the trunk by under a quarter): an edit to the ground under the Willow fails here
        pts = trunk.pts[(trunk.pts[:, 1] >= 0) & (trunk.pts[:, 1] <= mesh_willow.FORK_M)] + [x0, y0, z0]
        self.assertGreater(len(pts), 10)
        ax, ay, az, lx, lz = bark.TRUNK_AXIS
        ay += world.levels()["lake"]
        miss = np.hypot(pts[:, 0] - ax - (pts[:, 1] - ay) * lx, pts[:, 2] - az - (pts[:, 1] - ay) * lz)
        self.assertLess(float(miss.max()), 0.12)

    def test_the_furrows_run_along_every_piece_of_wood(self):
        # cells fixed in the world, long along a code's direction, are cut crosswise into polygons wherever the wood
        # tilts from it: on the burl, the flare and the root feet, and on the limbs and roots that bend away from their
        # codes. Seen from 4 m, almost all the bark shows furrows at least five times as long as they are wide
        above = self.above > 0.02
        self.assertGreater(self.area[self.shown].sum() / self.area[above].sum(), 0.7)
        long = self.across / np.maximum(self.along, 1e-9) >= 5
        self.assertGreater(self.area[self.shown & long].sum() / self.area[self.shown].sum(), 0.97)

    def test_the_low_wood_has_the_trunks_cells(self):
        # the burl, the root feet and the roots show furrows about as wide as the trunk's above them
        trunk = self.shown & (self.aux == BARK[1]) & (self.above > 1.0) & (self.above < 2.0)
        low = self.shown & (self.above < 1.2) & (self.out < mesh_willow.ROOT_ZONE_M)
        ratio = weighted_median(1 / self.across[low], self.area[low]) / weighted_median(1 / self.across[trunk],
                                                                                         self.area[trunk])
        self.assertTrue(0.67 < ratio < 1.5, ratio)

    def test_the_hollows_walls_have_the_trunks_cells(self):
        # the hollow's walls face round the trunk, not out from it, so their bearing round its axis barely changes across
        # them: cells wrapped round the axis grow several times the trunk's width there and smear into blotches
        walls = self.shown & (self.aux == BARK[1]) & self.in_gap & (self.side_on > 0.7)
        self.assertGreater(self.area[walls].sum(), 0.4)
        ratio = self.width(walls) / self.width(self.trunk())
        self.assertTrue(0.67 < ratio < 1.5, ratio)
        long = self.across / np.maximum(self.along, 1e-9) >= 5
        self.assertGreater(self.area[walls & long].sum() / self.area[walls].sum(), 0.9)

    def test_the_flared_base_has_the_trunks_cells(self):
        # the trunk flares to 1.6 times its girth at the ground; the same cells round it there would be as much wider
        flare = self.shown & (self.aux == BARK[1]) & (self.above < 0.6) & (self.from_gap > 80)
        self.assertGreater(self.area[flare].sum(), 2.0)
        ratio = self.width(flare) / self.width(self.trunk())
        self.assertTrue(0.83 < ratio < 1.2, ratio)

    def test_limbs_side_by_side_show_different_cells(self):
        # a limb's cells lie round its normal and along its code's direction, so two limbs of one code side by side show
        # the same cells (the same pattern repeated) unless each is shifted by its own place: the share of the limbs' and
        # roots' bark with a twin, within a quarter of a cell, on wood of its code more than half a metre away
        from scipy.spatial import cKDTree
        mid = self.tris.mean(axis=1)
        nrm = self.nrms.mean(axis=1)
        S = bark_space(mid, nrm / np.linalg.norm(nrm, axis=1, keepdims=True), self.aux)
        wood = self.shown & (self.aux != BARK[1])
        twin = np.zeros(len(mid), bool)
        for aux in np.unique(self.aux[wood]):
            rows = np.flatnonzero(wood & (self.aux == aux))
            for i, j in cKDTree(S[rows]).query_pairs(0.25):
                if np.linalg.norm(mid[rows[i]] - mid[rows[j]]) > 0.5:
                    twin[rows[[i, j]]] = True
        self.assertLess(self.area[twin].sum() / self.area[wood].sum(), 0.05)

    def test_the_hollows_walls_and_the_trunk_crossfade_without_a_net(self):
        # where the wood slopes, the walls' furrows follow the slope round the arch and the trunk's run out from its axis:
        # where both show, their borders cross in a net of short cells. The share of the hollow's bark where the two
        # largest cell spaces both weigh over 0.2 and their furrows cross by more than 20 degrees
        gap = np.flatnonzero(self.shown & (self.aux == BARK[1]) & self.in_gap)
        tris, nrms, aux = self.tris[gap], self.nrms[gap], self.aux[gap]
        dirs = np.stack(furrows_on(tris, nrms, aux), axis=1)                       # (triangle, space, 3)
        rng = np.random.default_rng(5)
        k = 16
        bc = rng.dirichlet(np.ones(3), (len(gap), k))
        p = np.einsum("tsk,tkj->tsj", bc, tris).reshape(-1, 3)
        n = np.einsum("tsk,tkj->tsj", bc, nrms).reshape(-1, 3)
        w = np.stack([w for _, w in bark_lattices(p, n / np.linalg.norm(n, axis=1, keepdims=True), np.repeat(aux, k))], 1)
        top = np.argsort(-w, axis=1)[:, :2]
        row = np.repeat(np.arange(len(gap)), k)
        cos = np.abs(np.einsum("ij,ij->i", dirs[row, top[:, 0]], dirs[row, top[:, 1]]))
        net = (np.take_along_axis(w, top[:, 1:], 1)[:, 0] > 0.2) & (cos < math.cos(math.radians(20)))
        share = np.sum(net * np.repeat(self.area[gap], k)) / (k * self.area[gap].sum())
        self.assertLess(share, 0.012)


class WillowMeshOracleTests(unittest.TestCase):
    """trimesh, as an independent oracle on the decoded payload."""

    def test_the_decoded_meshes_have_only_the_known_faults(self):
        self.assertEqual(world.mesh_faults("willow"), MESH_FAULTS)


if __name__ == "__main__":
    unittest.main()
