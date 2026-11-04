"""The village family (Hogsmeade): every mapped footprint regularized and split into buildings, each built from its style
record with straight-skeleton roofs and manifold3d openings.

The village is built once per test process and shared, read-only, by every class that needs it (shared()): every
footprint, or with VILLAGE_SAMPLE=high-street only the footprints along the High Street, the named buildings and the
houses with garden walls, as the village's quick mutation runs ask (tools/mutate/village.py)."""
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assets_testing as world  # noqa: E402
import mesh_village_plan as plan  # noqa: E402
import mesh_village_roofs as roofs  # noqa: E402
from mesh_source import Grid, field_polygons, load_georef, map_region  # noqa: E402

# the faults trimesh finds in this family's decoded meshes (assets_testing.mesh_faults), pinned: a new fault fails, and
# so does a fix until this is updated
MESH_FAULTS = {"winding": {}, "collapsed": {}}
WALL_LEVEL = -0.3     # build_meshes.BUILDING_WALL_LEVEL: walls stand 0.3 m inside the footprint distance
HIGH_STREET_M = 14.0  # the High Street's footprints: those within this of its centreline
SAMPLE = os.environ.get("VILLAGE_SAMPLE") == "high-street"
_SHARED = {}


def high_street(v, mv):
    """The High Street's footprints, the named buildings' and those of the houses with garden walls."""
    import mesh_village_small as small
    gardens = {min(v.outlines, key=lambda cid: v.outlines[cid].distance(Point(m.pos[:, [0, 2]].mean(axis=0))))
               for m in small.garden_walls(v)}
    return gardens | {cid for cid in v.built
                      if v.outlines[cid].distance(v.streets.high) < HIGH_STREET_M or v.scene.owners[cid] in mv.NAMED}


def shared():
    """The village as built, once per test process: its scene, the Village, the footprints in scope, each building
    group's (units, shell, fittings) and each unit's roof skin as its shell was built (skin, the height it was built
    from)."""
    if not _SHARED:
        from unittest import mock
        import build_meshes
        import mesh_village as mv

        scene = build_meshes.Scene(world.data(), world.scene())
        v = mv.Village(scene)
        scope = high_street(v, mv) if SAMPLE else set(v.built)
        skins, made = {}, mv._Building.__init__

        def keep_skin(b, village, fp, u, y_ref):
            made(b, village, fp, u, y_ref)
            skins[(u.cid, u.part, u.k)] = (b.skin, y_ref)

        shells = {}
        with mock.patch.object(mv._Building, "__init__", keep_skin):
            for cid in sorted(scope):
                for part, fp in enumerate(v.parts[cid]):
                    units = [u for u in v.units[cid] if u.part == part]
                    for asset in sorted({u.asset for u in units}):
                        group = [u for u in units if u.asset == asset]
                        shells[(cid, part, asset)] = (group, *v._build(fp, group))
        for u in (u for us in v.units.values() for u in us):
            v.attic(u), v.ridge(u)                # cached now, before any test builds under a mocked roofer
        _SHARED.update(mv=mv, scene=scene, v=v, scope=scope, shells=shells, skins=skins)
    return _SHARED


class VillageMeshOracleTests(unittest.TestCase):
    """trimesh, as an independent oracle on the decoded payload."""

    def test_the_decoded_meshes_have_only_the_known_faults(self):
        self.assertEqual(world.mesh_faults("village"), MESH_FAULTS)

    def test_nothing_stands_on_the_hamlet_left_unbuilt_for_the_shack(self):
        """The four footprints of the hamlet that the manifest leaves unbuilt, so that the Shrieking Shack stands above
        the rest of the village (the decision "shack-above-the-village"), get neither the village's meshes nor the
        generic massing: no triangle of the payload stands over them."""
        import build_meshes
        scene = world.scene()
        hamlet = [c["id"] for c in scene["excluded"]["components"] if c.get("decision") == "shack-above-the-village"]
        self.assertEqual(len(hamlet), 4)
        s = build_meshes.Scene(world.data(), scene)
        found = {c["id"]: c for c in s.labelled[1]}
        mapped = shapely.union_all([p for cid in hamlet for p in s.footprint(found[cid])])
        x0, z0, x1, z1 = mapped.bounds
        for chunk in world.payload()["chunks"]:
            xz = chunk["pos"][chunk["idx"]].mean(axis=1)[:, [0, 2]]
            near = (xz[:, 0] > x0) & (xz[:, 0] < x1) & (xz[:, 1] > z0) & (xz[:, 1] < z1)
            with self.subTest(asset=chunk["asset"]):
                self.assertFalse(shapely.contains_xy(mapped, xz[near, 0], xz[near, 1]).any())


def traced(shape, n=48, texel=2.0, angle_deg=world.GRID_DEG):
    """The footprints a mapped building `shape` (world polygon near the origin) leaves after the pipeline: its signed
    distance sampled on a rotated 2 m grid and contoured as Scene.footprint contours it."""
    c, s, w = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg)), n * texel
    grid = Grid({"nx": n, "ny": n, "aff": [[c / w, s / w, 0.5], [s / w, -c / w, 0.5]]})
    ii, jj = np.mgrid[0:n, 0:n]
    x, z = grid.to_world(ii, jj)
    d = shapely.distance(shapely.points(x.ravel(), z.ravel()), shape.boundary).reshape(x.shape)
    inside = shapely.contains_xy(shape, x.ravel(), z.ravel()).reshape(x.shape)
    return field_polygons(np.where(inside, -d, d), grid, WALL_LEVEL)


def rect(cx, cz, length, depth, deg):
    """A world rectangle centred on (cx, cz), `length` along the bearing `deg` (from east, anticlockwise)."""
    a = math.radians(deg)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c = np.array([cx, cz])
    return Polygon([c + u * su * length / 2 + v * sv * depth / 2 for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def first_hit(origins, dirs, tris, near=0.02, block=1_000_000):
    """Each ray's distance to the first of `tris` (t, 3, 3) it crosses beyond `near` (Moller-Trumbore), or inf; the rays
    are taken together, in batches of about `block` ray-triangle pairs."""
    out = np.full(len(origins), np.inf)
    if not len(tris):
        return out
    origins, dirs = np.asarray(origins, np.float64), np.asarray(dirs, np.float64)
    v0 = tris[:, 0]
    e1, e2 = tris[:, 1] - v0, tris[:, 2] - v0
    step = max(1, block // len(tris))
    for lo in range(0, len(origins), step):
        o, d = origins[lo:lo + step, None, :], dirs[lo:lo + step, None, :]
        p = np.cross(d, e2)                                     # (rays, triangles, 3)
        det = np.einsum("tj,rtj->rt", e1, p)
        ok = np.abs(det) > 1e-12
        inv = 1.0 / np.where(ok, det, 1.0)
        s = o - v0
        u = np.einsum("rtj,rtj->rt", s, p) * inv
        q = np.cross(s, e1)
        v = np.einsum("rtj,rj->rt", q, d[:, 0]) * inv
        t = np.einsum("tj,rtj->rt", e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > near)
        out[lo:lo + step] = np.where(hit, t, np.inf).min(axis=1)
    return out


def cracks(shell):
    """The edges of a shell (its corners joined where they coincide) that border a crack: edges shared by an odd number
    of triangles. Such edges along one line, under 0.1 mm off it, where every corner meets an even number of them, are
    where triangles with no area were left out, not cracks."""
    pos, ids = np.unique(np.round(shell.pos * 1e5).astype(np.int64), axis=0, return_inverse=True)
    f = ids.ravel()[shell.idx.astype(np.int64)]
    edges, counts = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1),
                              axis=0, return_counts=True)
    odd = [tuple(int(x) for x in e) for e in edges[counts % 2 == 1]]
    ends, p = sorted({x for e in odd for x in e}), pos.astype(np.float64)

    def off(x, a, b):
        """How far x lies off the line through a and b: their triangle's height over its longest side."""
        side = max(np.linalg.norm(p[b] - p[a]), np.linalg.norm(p[x] - p[a]), np.linalg.norm(p[x] - p[b]))
        return np.linalg.norm(np.cross(p[b] - p[a], p[x] - p[a])) / side if side else 0.0

    cracked = []
    for a, b in odd:
        near = [x for x in ends if off(x, a, b) < 10]
        q, r = max(((x, y) for x in near for y in near), key=lambda e: np.linalg.norm(p[e[1]] - p[e[0]]))
        line = {x for x in near if off(x, q, r) < 10}
        meets = np.bincount([x for e in odd if set(e) <= line for x in e])
        if (meets % 2).any():
            cracked.append((a, b))
    return cracked


class VillagePlanTests(unittest.TestCase):
    """Footprints traced on the 2 m grid come back as straight walls and separate buildings."""

    def regular(self, shape, seed=1):
        (trace,) = traced(shape)
        (fp,) = plan.plan_footprint(trace, seed)
        return fp, trace

    def test_the_plan_frame_is_a_proper_rotation_that_round_trips(self):
        f = plan.Frame(-812.5, -1190.25, 0.61)
        self.assertAlmostEqual(np.linalg.det(f.rot3), 1.0)
        p = np.array([[-800.0, -1200.0], [-820.5, -1185.0]])
        np.testing.assert_allclose(f.world(f.plan(p)), p, atol=1e-9)
        np.testing.assert_allclose(f.world3(np.column_stack([f.plan(p), [2.0, 3.0]]), 100.0)[:, [0, 2]], p, atol=1e-9)

    def test_a_moved_part_turns_about_its_components_centre_onto_its_new_place(self):
        """MOVED turns a footprint part's frame `turn` degrees about its component's decoded centre and carries that
        point to its new place: every plan point lands where its old world point goes under that motion. A component
        no longer decoded there is refused rather than moved from wherever it now is."""
        import mesh_village as mv
        frame, c, at, to = plan.Frame(3.0, -4.0, 0.7), {"id": 9, "x": 1.0, "z": 2.0}, (1.0, 2.0), (40.0, -25.0)
        t = math.radians(155.0)
        turn = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        XY = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 8.0], [-2.0, 3.0]])
        moved = mv._moved(frame, c, at, to, 155.0)
        np.testing.assert_allclose(moved.world(XY), (frame.world(XY) - at) @ turn.T + to, atol=1e-9)
        with self.assertRaises(ValueError):
            mv._moved(frame, {**c, "x": 3.0}, at, to, 155.0)

    def test_a_traced_rectangle_comes_back_square_at_its_own_angle(self):
        for deg in (37.0, -12.0, world.GRID_DEG):
            with self.subTest(deg=deg):
                fp, trace = self.regular(rect(1.0, -2.0, 11.0, 7.0, deg))
                self.assertTrue(fp.regular)
                self.assertEqual(len(fp.outline.exterior.coords), 5)          # four corners
                x0, y0, x1, y1 = fp.outline.bounds
                sides = sorted((x1 - x0, y1 - y0))
                self.assertAlmostEqual(sides[0], 7.0 - 0.6, delta=0.35)       # walls 0.3 m inside the footprint
                self.assertAlmostEqual(sides[1], 11.0 - 0.6, delta=0.35)
                world_outline = fp.frame.to_world(fp.outline)
                self.assertLess(world_outline.hausdorff_distance(rect(1.0, -2.0, 10.4, 6.4, deg)), 0.35)
                self.assertEqual(len(fp.units), 1)

    def test_offset_cottages_in_a_row_become_separate_buildings(self):
        a = math.radians(30.0)
        along, shift = np.array([math.cos(a), math.sin(a)]) * 7.5, np.array([-math.sin(a), math.cos(a)]) * 1.6
        row = shapely.union_all([rect(*-along, 7.8, 6.0, 30.0), rect(*shift, 7.8, 6.0, 30.0), rect(*along, 7.8, 6.0, 30.0)])
        fp, _ = self.regular(row)
        self.assertEqual(len(fp.units), 3)
        middle = fp.frame.to_world(sorted(fp.units, key=lambda u: fp.frame.to_world(u).centroid.distance(Point(0, 0)))[0])
        self.assertLess(middle.centroid.distance(Point(*shift)), 0.8)

    def test_a_courtyard_block_keeps_its_courtyard_and_splits_into_ranges(self):
        block = rect(0.0, 0.0, 26.0, 22.0, 15.0).difference(rect(0.0, 0.0, 12.0, 9.0, 15.0))
        fp, _ = self.regular(block)
        self.assertEqual(len(fp.outline.interiors), 1)
        self.assertGreaterEqual(len(fp.units), 4)
        self.assertAlmostEqual(shapely.union_all(fp.units).area, fp.outline.area, delta=1e-6)
        for u in fp.units:
            x0, y0, x1, y1 = u.bounds
            self.assertGreaterEqual(min(x1 - x0, y1 - y0), plan.MIN_SLAB_M - 1e-9)

    def test_a_long_row_is_cut_into_plots(self):
        fp, _ = self.regular(rect(0.0, 0.0, 36.0, 7.0, -20.0), seed=4)
        self.assertGreaterEqual(len(fp.units), 4)
        for u in fp.units:
            x0, y0, x1, y1 = u.bounds
            self.assertGreater(max(x1 - x0, y1 - y0), plan.PLOT_M[0] * 0.8)
            self.assertLess(max(x1 - x0, y1 - y0), plan.PLOT_M[1] * 1.2)
        self.assertAlmostEqual(sum(u.area for u in fp.units), fp.outline.area, delta=1e-6)

    def test_an_irregular_footprint_keeps_its_straightened_trace(self):
        wedge = Polygon([(-9.0, -6.0), (9.0, -6.0), (-3.0, 8.0)])
        (trace,) = traced(wedge)
        (fp,) = plan.plan_footprint(trace, 1)
        self.assertFalse(fp.regular)
        self.assertEqual(fp.units, (fp.outline,))
        self.assertLess(fp.frame.to_world(fp.outline).hausdorff_distance(trace), 0.8)

    def test_planning_is_deterministic(self):
        block = rect(3.0, 1.0, 26.0, 9.0, 52.0).union(rect(10.0, 6.0, 8.0, 12.0, 52.0))
        (trace,) = traced(block)
        a, b = plan.plan_footprint(trace, 9), plan.plan_footprint(trace, 9)
        self.assertEqual([u.wkb for f in a for u in f.units], [u.wkb for f in b for u in f.units])

    def test_hogsmeade_map_pixels_land_where_the_registration_puts_them(self):
        georef = load_georef()
        for px in ((361.0, 1148.0), (544.0, 1506.5), (100.0, 2000.0)):
            with self.subTest(px=px):
                x, z = plan.hogs_world(np.array(px), georef)
                self.assertEqual(map_region(x, z, georef), "hogs")
                r = math.radians(georef["rot_deg"])
                x0, x1, y0, y1 = georef["hogs_rect"]
                u = georef["U0"] + (x * math.cos(r) + z * math.sin(r)) / georef["mFF"]
                v = georef["V0"] + (x * math.sin(r) - z * math.cos(r)) / georef["mFF"]
                self.assertAlmostEqual((u - x0) * 1735 / (x1 - x0), px[0], places=6)
                self.assertAlmostEqual((v - y0) * 2261 / (y1 - y0), px[1], places=6)

class VillageRoofTests(unittest.TestCase):
    """roof(): closed solids over plan outlines, a gable where an edge's angle is 90 and a hip elsewhere, from either
    backend."""

    def solid(self, outline, angles, backend=None):
        import trimesh

        v, f = roofs.roof(orient(outline, 1.0), angles, backend=backend)
        m = trimesh.Trimesh(v, f, process=False, validate=False)
        self.assertTrue(m.is_watertight)
        self.assertTrue(m.is_winding_consistent)
        self.assertGreater(m.volume, 0.0)
        up = m.face_normals[:, 2] > 1e-6
        self.assertAlmostEqual(float((m.area_faces * m.face_normals[:, 2])[up].sum()), outline.area, delta=1e-6 * outline.area)
        return v, f, m

    def ridge(self, v):
        top = v[v[:, 2] > v[:, 2].max() - 1e-6]
        return top[np.lexsort((top[:, 1], top[:, 0]))]

    def test_gable_ends_carry_the_ridge_to_the_end_walls(self):
        outline = orient(box(0.0, 0.0, 10.0, 6.0), 1.0)
        pts = np.asarray(outline.exterior.coords)[:-1]
        angles = [90.0 if abs(b[0] - a[0]) < 1e-9 else 45.0 for a, b in zip(pts, np.roll(pts, -1, 0))]
        for backend in roofs.BACKENDS:
            with self.subTest(backend=backend):
                v, _, _ = self.solid(outline, angles, backend)
                r = self.ridge(v)
                np.testing.assert_allclose(r[[0, -1]], [[0.0, 3.0, 3.0], [10.0, 3.0, 3.0]], atol=1e-6)

    def test_hip_ends_stop_the_ridge_short_of_the_end_walls(self):
        v, _, _ = self.solid(box(0.0, 0.0, 10.0, 6.0), 45.0)
        np.testing.assert_allclose(self.ridge(v)[[0, -1]], [[3.0, 3.0, 3.0], [7.0, 3.0, 3.0]], atol=1e-6)

    def test_each_angle_applies_to_the_edge_from_its_point_to_the_next(self):
        outline = orient(box(0.0, 0.0, 10.0, 6.0), 1.0)
        pts = np.asarray(outline.exterior.coords)[:-1]
        angles = [90.0 if abs(b[0] - a[0]) < 1e-9 and a[0] == 0.0 else 45.0 for a, b in zip(pts, np.roll(pts, -1, 0))]
        v, _, _ = self.solid(outline, angles)
        r = self.ridge(v)
        np.testing.assert_allclose(r[[0, -1]], [[0.0, 3.0, 3.0], [7.0, 3.0, 3.0]], atol=1e-6)   # gable at x = 0 only

    def test_an_l_shaped_roof_and_a_courtyard_roof_are_closed_and_cover_their_plans_once(self):
        ell = shapely.union_all([box(0.0, 0.0, 14.0, 6.0), box(8.0, 0.0, 14.0, 12.0)])
        self.solid(ell, 45.0)
        ring = box(0.0, 0.0, 24.0, 20.0).difference(box(6.0, 6.0, 18.0, 14.0))
        _, _, m = self.solid(ring, 45.0)
        centres = m.triangles_center
        self.assertFalse(any(box(6.1, 6.1, 17.9, 13.9).contains(Point(x, y)) for x, y, _ in centres))

    def test_the_baseline_roofs_any_outline_with_any_pitches(self):
        """A triangle, and an irregular plan with two pitches and a gable, come out closed and cover their plans once;
        a sloping edge rises at its own pitch, a gable straight up."""
        tri = Polygon([(0.0, 0.0), (9.0, 0.0), (3.0, 7.0)])
        v, _, _ = self.solid(tri, 45.0, backend="inrepo")
        self.assertAlmostEqual(float(v[:, 2].max()), shapely.maximum_inscribed_circle(tri, 1e-4).length, delta=0.02)
        odd = orient(Polygon([(0.0, 0.0), (11.0, 0.0), (12.5, 4.0), (7.0, 8.5), (0.0, 6.0)]), 1.0)
        angles = [40.0, 50.0, 90.0, 50.0, 40.0]
        v, f, m = self.solid(odd, angles, backend="inrepo")
        near = v[(np.abs(v[:, 1]) < 1e-9) & (v[:, 2] > 1e-9)]                   # nothing rises off the 40 degree base
        self.assertEqual(len(near), 0)
        ring = np.asarray(odd.exterior.coords)[:-1]
        k = next(i for i in range(5) if np.allclose(ring[i], (12.5, 4.0)))
        gable = LineString([ring[k], ring[(k + 1) % 5]])
        tops = [p for p in v if p[2] > 1e-6 and gable.distance(Point(p[:2])) < 1e-6]
        self.assertTrue(tops, "the gable rises straight up from its wall")

    def test_roofs_are_deterministic(self):
        ell = orient(shapely.union_all([box(0.0, 0.0, 14.0, 6.0), box(8.0, 0.0, 14.0, 12.0)]), 1.0)
        a, b = roofs.roof(ell, 45.0), roofs.roof(ell, 45.0)
        self.assertEqual(a[0].tobytes() + a[1].tobytes(), b[0].tobytes() + b[1].tobytes())

    def test_a_gable_beside_an_inside_corner_still_roofs_its_plan_once(self):
        """CGAL's skeleton folds this L's roof over itself, its gable standing beside the inside corner; roof() builds
        such a roof with the baseline instead, closed and covering its plan once."""
        ell = Polygon([(-4.92, 0.52), (-4.92, -1.39), (-2.53, -1.39), (-2.53, -4.44), (3.04, -4.44), (3.04, 0.52)])
        self.solid(ell, [49.2, 49.2, 90.0, 49.2, 49.2, 90.0])

class VillageBuildTests(unittest.TestCase):
    """The village as built, every footprint of it (shared()): who is where, which way they face, how they meet the
    ground and the sky, and the village's share of the payload."""
    BUDGET = 200_000
    SHACK_ABOVE_M = 2.0     # the Shrieking Shack's floor stands at least this far above every other building's
    MOVED_GARDEN_M = 14.0   # a cottage moved for the Shack stands this far from any other, as the stile lane's do
    FACING_DEG = 20.0       # ...its front within this of facing the nearest road straight on, as along its lane
    TREE_OFF_M = 2.5        # frag1.glsl's vegMask grows no tree centred nearer than this to a mapped footprint

    @classmethod
    def setUpClass(cls):
        found = shared()
        cls.mv, cls.scene, cls.v = found["mv"], found["scene"], found["v"]
        cls.scope, cls.shells, cls.skins = found["scope"], found["shells"], found["skins"]

    def units(self):
        return [u for cid in sorted(self.scope) for u in self.v.units[cid]]

    def test_the_village_keeps_to_its_share_of_the_payload(self):
        """Its buildings and their fittings come to at most BUDGET triangles: at under 9 bytes each, less the generic
        massing they replace, the 1.5 MB the village may add to the payload."""
        if SAMPLE:
            self.skipTest("the budget is the whole village's")
        total = sum(len(shell.idx) + sum(len(m.idx) for m in fittings) for _, shell, fittings in self.shells.values())
        self.assertLessEqual(total, self.BUDGET)

    def front(self, u):
        """(midpoint, outward unit normal, endpoints) of a unit's front, in world (x, z)."""
        a, b = plan.edges(u.outline)[u.front]
        f = u.fp.frame
        mid = f.world((a + b) / 2)
        out = f.world((a + b) / 2 + plan.outward(a, b)) - mid
        return mid, out / np.linalg.norm(out), (f.world(a), f.world(b))

    def faces(self, mesh, name):
        from materials import code
        sel = (mesh.attr[mesh.idx[:, 0]] == code(name)).all(axis=1)
        return mesh.pos[mesh.idx[sel]]

    def test_every_footprint_is_built_but_the_specks(self):
        """Every village footprint is built, the named shops among them, but one whose plan is only specks (a sliver of
        a wing), which keeps the generic massing."""
        owners = self.scene.owners
        for cid in self.v.outlines:
            specks = max((fp.outline.area for fp in self.v.parts[cid]), default=0.0) < plan.MIN_PART_M2
            with self.subTest(component=cid):
                self.assertEqual(cid in self.v.built, not specks)
        self.assertEqual({owners[c] for c in self.v.built} & set(self.mv.SHOPS), set(self.mv.SHOPS))

    def test_the_shack_stands_a_little_way_above_the_rest_of_the_village(self):
        """PoA 14: "It stood a little way above the rest of the village". The Shrieking Shack's floor stands at least
        SHACK_ABOVE_M above every other building's in the whole village, whatever the scope (the manifest decision
        "shack-above-the-village")."""
        everyone = [u for cid in sorted(self.v.units) for u in self.v.units[cid]]
        (shack,) = [u for u in everyone if u.asset == "shrieking-shack"]
        floor = {(u.cid, u.part, u.k): self.mv.floor_level(self.scene, u) for u in everyone if u is not shack}
        highest = max(floor, key=floor.get)
        self.assertGreaterEqual(self.mv.floor_level(self.scene, shack) - floor[highest], self.SHACK_ABOVE_M, highest)

    def test_the_cottages_moved_for_the_shack_stand_across_their_lane_clear_of_trees(self):
        """Cottages 23 and 28 stand across their lane from where the map draws them (MOVED; the decision
        "shack-above-the-village"): the lane runs between the mapped footprint and the built one, which stands off the
        road with its front looking squarely across at the lane, the nearest road within FACING_DEG of straight ahead,
        and at least MOVED_GARDEN_M from any other building. The raymarcher grows trees from the site's tree layer,
        which keeps them off the mapped footprints only: no tree texel lies within TREE_OFF_M of the moved walls and a
        texel's diagonal more, so no tree stands nearer them than one may stand to a mapped footprint."""
        from common import TEXEL_REACH_M
        from mesh_source import CHANNELS
        streets = self.v.streets
        trees = self.scene.locA[..., CHANNELS["locA"].index("trees")]
        ti, tj = np.nonzero(trees > 0)
        tree_xz = np.column_stack(self.scene.local.to_world(ti, tj))
        self.assertEqual(set(self.mv.MOVED), {23, 28})
        for cid in sorted(self.mv.MOVED):
            mapped = shapely.union_all(self.scene.footprint(self.v.components[cid]))
            built = self.v.outlines[cid]
            a, b = (np.asarray(g.centroid.coords[0]) for g in (mapped, built))
            x0, z0, x1, z1 = built.bounds
            gx, gz = np.meshgrid(np.arange(x0, x1, 0.5), np.arange(z0, z1, 0.5))
            inside = np.column_stack([gx.ravel(), gz.ravel()])
            inside = inside[shapely.contains_xy(built, inside[:, 0], inside[:, 1])]
            near = tree_xz[(np.abs(tree_xz - b) < 40.0).all(axis=1)]
            with self.subTest(component=cid):
                self.assertIn(cid, self.v.built)
                self.assertGreater(streets.road_share(a + (b - a) * np.linspace(0.0, 1.0, 200)[:, None]), 0.0)
                self.assertEqual(streets.road_share(inside), 0.0)
                self.assertGreaterEqual(min(built.distance(o) for c, o in self.v.outlines.items() if c != cid),
                                        self.MOVED_GARDEN_M)
                self.assertGreater(float(shapely.distance(built, shapely.points(near)).min()),
                                   self.TREE_OFF_M + TEXEL_REACH_M)
                for u in self.v.units[cid]:
                    mid, out, _ = self.front(u)
                    _, i = streets.tree.query(mid)
                    ahead = streets.tree.data[i] - mid
                    self.assertGreater(float(out @ ahead) / float(np.linalg.norm(ahead)),
                                       math.cos(math.radians(self.FACING_DEG)))

    def test_each_named_shop_is_the_building_its_label_points_to(self):
        labels = self.v._labels()
        for name in sorted({u.asset for u in self.units() if u.asset in self.mv.SHOPS}):
            with self.subTest(shop=name):
                building = shapely.union_all([u.fp.frame.to_world(u.outline) for u in self.units() if u.asset == name])
                self.assertLess(building.distance(Point(labels[name])), 1.5)

    def test_the_three_broomsticks_is_an_old_low_end_and_a_later_two_storey_inn(self):
        """White Hound: "The pub called the Three Broomsticks is probably the oldest building in the village", perhaps
        after the Cramond Inn, "one end of which is very old, although the rest is late 18th C": a low rubble end
        under thatch beside a two-storey inn under slate, sharing a wall, both fronting the High Street; Rosmerta
        sees the street from her bedroom over the pub (HBP 27), and light spills from its windows (DH 28)."""
        parts = [u for u in self.units() if u.asset == "three-broomsticks"]
        self.assertEqual(len(parts), 2)
        old, new = sorted(parts, key=lambda u: u.style.storeys)
        self.assertEqual((old.style.storeys < 2.0, old.style.wall, old.style.roof), (True, "rubble", "thatch"))
        self.assertEqual((new.style.storeys, new.style.roof), (2.0, "slate"))
        self.assertTrue(old.high and new.high)
        self.assertGreater(old.outline.boundary.intersection(new.outline.buffer(0.05)).length, 3.0)
        _, shell, _ = self.shells[(new.cid, new.part, "three-broomsticks")]
        mid, out, _ = self.front(new)
        glass = np.concatenate([self.faces(shell, g) for g in self.mv.GLASS])
        glass = glass[np.abs((glass.mean(axis=1)[:, [0, 2]] - mid) @ out) < 0.5]
        upper = self.mv.floor_level(self.scene, new) + self.mv.FLOOR_OVER_M + self.mv.STOREY_M
        self.assertTrue((glass[:, :, 1].min(axis=1) > upper).any() and (glass[:, :, 1].max(axis=1) < upper).any())

    def frontage(self, u, k):
        """(share of the samples in front of edge k on the road, its length, its world midpoint, its outward normal)."""
        a, b = plan.edges(u.outline)[k]
        n = plan.outward(a, b)
        f = u.fp.frame
        pts = f.world(np.array([a + (b - a) * t + n * d for t in (0.2, 0.35, 0.5, 0.65, 0.8) for d in (2.0, 4.0, 6.0)]))
        mid = f.world((a + b) / 2)
        out = f.world((a + b) / 2 + n) - mid
        return self.v.streets.road_share(pts), float(np.linalg.norm(b - a)), mid, out, pts

    def faces_the_high_street(self, u, k):
        """Whether edge k is a wall at least 2.5 m long looking across the road at the High Street: road in front of it,
        the centreline within 12 m of its samples and straight ahead of it."""
        share, length, mid, out, pts = self.frontage(u, k)
        line = self.v.streets.high
        if share == 0.0 or length < 2.5 or min(line.distance(Point(p)) for p in pts) >= 12.0:
            return False
        to = np.asarray(line.interpolate(line.project(Point(mid))).coords[0]) - mid
        return float(out @ to) > 0.7 * float(np.linalg.norm(to))

    def free_edges(self, u):
        return [k for k in range(len(plan.edges(u.outline))) if k not in u.party]

    def test_buildings_facing_the_high_street_front_it(self):
        """A building with a wall looking across the road at the High Street is a High Street building, fronting it
        with such a wall; no other building is."""
        for u in self.units():
            facing = [k for k in self.free_edges(u) if self.faces_the_high_street(u, k)]
            with self.subTest(component=u.cid, part=u.part, unit=u.k):
                self.assertEqual(u.high, bool(facing))
                if facing:
                    self.assertIn(u.front, facing)

    def test_no_front_is_a_step_beside_a_longer_street_wall(self):
        """A front is long enough for a door and a window if any free wall is, and away from the High Street it has at
        least half the road frontage (road share in front of a wall times its length) of the building's best wall of
        those that could front it."""
        for u in self.units():
            free = self.free_edges(u)
            if not free:
                continue
            scores = {k: self.frontage(u, k)[:2] for k in free}
            fronts = {k: sc for k, sc in scores.items() if sc[1] >= 2.5} or scores
            share, length = scores[u.front]
            with self.subTest(component=u.cid, part=u.part, unit=u.k):
                if any(n >= 2.5 for _, n in scores.values()):
                    self.assertGreaterEqual(length, 2.5)
                if not u.high:
                    self.assertGreaterEqual(share * length, 0.5 * max(s * n for s, n in fronts.values()))

    def test_the_three_broomsticks_and_honeydukes_face_each_other_across_the_high_street(self):
        named = {u.asset: u for u in self.units() if u.asset in ("three-broomsticks", "honeydukes")}
        (m1, o1, _), (m2, o2, _) = self.front(named["three-broomsticks"]), self.front(named["honeydukes"])
        self.assertTrue(self.v.streets.high.distance(Point(m1)) < 15.0 and self.v.streets.high.distance(Point(m2)) < 15.0)
        # each faces the High Street's centreline from its own side of it ("on the opposite side of the road")
        line = self.v.streets.high
        sides = []
        for m, o in ((m1, o1), (m2, o2)):
            s = line.project(Point(m))
            near = np.asarray(line.interpolate(s).coords[0])
            tangent = np.asarray(line.interpolate(s + 1.0).coords[0]) - np.asarray(line.interpolate(s - 1.0).coords[0])
            self.assertGreater(float((near - m) @ o), 0.0)
            sides.append(np.sign(tangent[0] * (m - near)[1] - tangent[1] * (m - near)[0]))
        self.assertNotEqual(sides[0], sides[1])

    def test_the_hogs_head_faces_south_east_and_is_slated(self):
        (u,) = [u for u in self.units() if u.asset == "hogs-head"]
        _, out, _ = self.front(u)
        south_east = np.array([1.0, -1.0]) / math.sqrt(2.0)          # x east, z north
        self.assertGreater(float(out @ south_east), math.cos(math.radians(45.0)))
        self.assertEqual(u.style.roof, "slate")
        (group, shell, _), = [v for k, v in self.shells.items() if k[2] == "hogs-head"]
        self.assertEqual(len(self.faces(shell, "village-thatch")) + len(self.faces(shell, "village-thatch-new")), 0)

    def hogs_head(self):
        """The Hog's Head: (its unit, shell, fittings, its front's world midpoint and outward normal, the viewer's
        right as they face the front, the front's length), in world (x east, z north)."""
        (u,) = [u for u in self.units() if u.asset == "hogs-head"]
        _, shell, fittings = self.shells[(u.cid, u.part, "hogs-head")]
        mid, out, (a, b) = self.front(u)
        return u, shell, fittings, mid, out, np.array([-out[1], out[0]]), float(np.linalg.norm(b - a))

    @staticmethod
    def plan_normals(tris):
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])[:, [0, 2]]
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def test_the_hogs_head_has_bay_windows_on_its_front_and_its_left_side(self):
        """OotP 16: "The bay windows were so encrusted with grime that very little daylight could permeate the room."
        As White Hound's plan has them, canted bays stand out from the bar's front, just right of its middle as you
        face it, and from its left side wall near the front: each glazed in grimy glass on its face and on both canted
        sides."""
        u, shell, _, mid, out, right, length = self.hogs_head()
        glass = self.faces(shell, "glass-dark")
        rel, n = glass.mean(axis=1)[:, [0, 2]] - mid, self.plan_normals(glass)
        for name, wall, along, plane, lo, hi, placed in (
                ("front", out, right, 0.0, -length / 2, length / 2, lambda x: x > 0.0),
                ("left side", -right, -out, length / 2, 0.0, 5.0, lambda x: True)):
            proud = rel @ wall - plane
            bay = (proud > 0.1) & (proud < 1.0) & (rel @ along > lo) & (rel @ along < hi)
            with self.subTest(bay=name):
                facing = n[bay] @ wall
                self.assertTrue((facing > 0.95).any())
                self.assertTrue(placed(float((rel[bay][facing > 0.95] @ along).mean())))
                canted = n[bay][(facing > 0.4) & (facing < 0.9)] @ along
                self.assertTrue((canted > 0.3).any() and (canted < -0.3).any())

    def test_the_hogs_head_sign_hangs_over_its_door_at_the_front_left(self):
        """OotP 16: "A battered wooden sign hung from a rusty bracket over the door, with a picture on it of a wild
        boar's severed head, leaking blood on to the white cloth around it." The door is at the left of the front as
        you face it (White Hound's plan); the sign hangs out from the wall straight over it, above its head, and on
        each face of the board a white cloth carries the boar's dark head and its blood."""
        from materials import code
        u, shell, fittings, mid, out, right, length = self.hogs_head()
        door = self.faces(shell, u.style.door)
        door = door[np.abs((door.mean(axis=1)[:, [0, 2]] - mid) @ out) < 0.5]
        along = float((door.reshape(-1, 3)[:, [0, 2]].mean(axis=0) - mid) @ right)
        self.assertLess(along, -length / 6)
        boards = [m for m in fittings if (m.attr[0] == code("sign-board")).all()]
        self.assertEqual(len(boards), 1)
        board = boards[0].pos
        centre = board[:, [0, 2]].mean(axis=0)
        self.assertLess(abs(float((centre - mid) @ right) - along), 0.1)
        self.assertTrue(0.3 < float((centre - mid) @ out) < 1.2)
        self.assertGreater(float(board[:, 1].min()), float(door[:, :, 1].max()))
        for paint in ("sign-cloth", "margin-black", "margin-oxblood"):
            sides = [float((m.pos[:, [0, 2]].mean(axis=0) - centre) @ right) for m in fittings
                     if (m.attr[0] == code(paint)).all()]
            with self.subTest(paint=paint):
                self.assertTrue(any(s > 0.0 for s in sides) and any(s < 0.0 for s in sides), sides)

    def test_the_hogs_head_has_eaves_to_its_street_and_its_stack_on_its_right_gable(self):
        """Its roof's eaves run along its front and its ends are gables. White Hound's plan puts every fireplace on its
        right side wall as you face the front, so the one stack it has rises from that gable."""
        u, shell, _, mid, out, right, length = self.hogs_head()
        angles = self.mv.roof_angles(u)
        self.assertLess(angles[u.front], 89.999)
        self.assertTrue(all(angles[(u.front + j) % 4] >= 89.999 for j in (-1, 1)))
        (stack,) = self.stacks(shell)
        self.assertGreater(float((stack[0] - mid) @ right), length / 2 - 1.0)

    def shack(self):
        (u,) = [u for u in self.units() if u.asset == "shrieking-shack"]
        _, shell, fittings = self.shells[(u.cid, u.part, "shrieking-shack")]
        return u, shell, fittings

    def lamps(self):
        """The High Street's lamps (mesh_village_small), built once per test process."""
        import mesh_village_small as small
        found = shared()
        if "lamps" not in found:
            found["lamps"] = small.lamps(self.v)
        return found["lamps"]

    @staticmethod
    def material(mesh, name):
        from materials import code
        return mesh.pos[(mesh.attr == code(name)).all(axis=1)]

    def test_a_few_streetlamps_stand_along_the_high_street_only(self):
        """HBP 27: "the darkness complete but for a few streetlamps and lit upper windows"; White Hound: "In DH the
        side-road to the Hog's Head is not lit". Each lamp stands on the High Street's pavement, a pace or two in
        front of a High Street building and clear of its door, the lamps alternately on either side and no two within
        15 m."""
        lamps = self.lamps()
        self.assertGreaterEqual(len(lamps), 10)
        line = self.v.streets.high
        axis = np.asarray(line.coords[-1]) - np.asarray(line.coords[0])
        left = np.array([-axis[1], axis[0]])
        fronts = [self.front(u) for cid in sorted(self.v.units) for u in self.v.units[cid] if u.high]
        doors = np.concatenate([self.faces(shell, d) for _, shell, _ in self.shells.values() for d in self.mv.DOORS])
        doors = doors.mean(axis=1)[:, [0, 2]]
        centres, sides = [], []
        for m in lamps:
            c = self.material(m, "lamp-glass")[:, [0, 2]].mean(axis=0)
            _, n, (a, b) = min(fronts, key=lambda f: LineString(f[2]).distance(Point(c)))
            t = (b - a) / np.linalg.norm(b - a)
            near = doors[np.abs((doors - a) @ n) < 0.5]
            with self.subTest(lamp=tuple(np.round(c, 1))):
                self.assertTrue(1.0 < LineString([a, b]).distance(Point(c)) < 2.5)
                self.assertGreater(float((c - a) @ n), 0.0)
                self.assertTrue(np.all(np.abs((near - c) @ t) > 0.8))
            centres.append(c)
            sides.append(float(n @ left) > 0.0)
        self.assertGreaterEqual(min(sum(sides), len(sides) - sum(sides)), 0.4 * len(sides))
        gaps = np.linalg.norm(np.asarray(centres)[:, None] - np.asarray(centres)[None], axis=2)
        self.assertGreaterEqual(float((gaps + np.eye(len(centres)) * 1e9).min()), 15.0)

    def test_streetlamps_are_cast_iron_on_stone_plinths_clear_of_the_buildings(self):
        """White Hound: "ornate Victorian cast-iron jobs". Each lamp's glass lantern stands on a cast-iron post over a
        stone plinth sunk in the ground, clear of every building."""
        for m in self.lamps():
            c = m.pos[:, [0, 2]].mean(axis=0)
            g = float(self.scene.ground(*c))
            stone, iron, glass = (self.material(m, n) for n in ("pale-stone", "black-iron", "lamp-glass"))
            with self.subTest(lamp=tuple(np.round(c, 1))):
                self.assertLess(float(stone[:, 1].min()), g)
                self.assertTrue(0.4 < float(stone[:, 1].max()) - g < 0.8)
                self.assertGreater(len(iron), 0)
                self.assertTrue(3.0 < float(glass[:, 1].min()) - g and float(glass[:, 1].max()) - g < 4.6)
                self.assertGreater(min(o.distance(Point(c)) for o in self.v.outlines.values()), 0.5)

    def test_the_shack_is_two_storeys_of_weatherboard_under_corrugated_iron(self):
        """White Hound's Shack: "a timber frame clad with weatherboarding", "possible that it has a corrugated-iron
        roof", two storeys."""
        u, shell, _ = self.shack()
        self.assertEqual(u.style.storeys, 2.0)
        self.assertGreater(len(self.faces(shell, "weatherboard")), 0)
        self.assertGreater(len(self.faces(shell, "corrugated-iron")), 0)
        for roof in ("village-thatch", "village-thatch-new", "village-slate"):
            self.assertEqual(len(self.faces(shell, roof)), 0)

    def test_every_window_of_the_shack_is_boarded_but_one_towards_the_castle(self):
        """PoA 14: "even in daylight was slightly creepy, with its boarded windows and dank overgrown garden"; DH 32:
        "all the windows boarded except for one", and "The single unblocked window revealed distant bursts of light
        where the castle stood". The castle stands at the world's origin."""
        u, shell, _ = self.shack()
        glass = np.concatenate([self.faces(shell, g) for g in self.mv.GLASS])
        self.assertGreater(len(glass), 0)
        c = glass.mean(axis=1)[:, [0, 2]]
        self.assertLess(float(np.ptp(c, axis=0).max()), 1.5)
        towards = -c.mean(axis=0) / np.linalg.norm(c.mean(axis=0))
        self.assertTrue(np.all(self.plan_normals(glass) @ towards > 0.7))
        boards = self.faces(shell, "boarding")
        walls = {tuple(np.round(n, 1)) for n in self.plan_normals(boards)}
        self.assertGreaterEqual(len(boards) // 2, 6)
        self.assertGreaterEqual(len(walls), 2)

    def test_a_low_fence_rings_the_shacks_garden(self):
        """PoA 14: "as they leaned on the fence, looking up at it": a fence about a metre high rings the Shack's garden
        a few metres out from its walls, broken only where a path runs through it."""
        from materials import code
        u, _, fittings = self.shack()
        fence = [m for m in fittings if (m.attr[0] == code("boarding")).all()]
        self.assertTrue(fence)
        pts = np.concatenate([m.pos for m in fence])
        house = u.fp.frame.to_world(u.outline)
        away = shapely.distance(house, shapely.points(pts[:, [0, 2]]))
        self.assertTrue(np.all((away > 2.0) & (away < 8.0)), (away.min(), away.max()))
        centre = np.asarray(house.centroid.coords[0])
        angle = np.arctan2(pts[:, 2] - centre[1], pts[:, 0] - centre[0])
        self.assertGreaterEqual(len(np.unique(np.floor((angle + math.pi) / (2 * math.pi) * 24))), 20)
        above = pts[:, 1] - self.scene.ground(pts[:, 0], pts[:, 2])
        self.assertLess(float(above.max()), 1.3)
        self.assertGreater(float(np.percentile(above, 90)), 0.8)
        for m in fence:
            tris = m.pos[m.idx]
            self.assertEqual(self.v.streets.road_share(tris.mean(axis=1)[:, [0, 2]]), 0.0)

    def test_the_shacks_one_chimney_stack_is_stone(self):
        """The Shack's one stack is rubble, like its footing, not weatherboard."""
        from materials import code
        _, shell, _ = self.shack()
        (stack,) = self.stacks(shell)
        _, top, corners = stack
        lo, hi = corners.min(axis=0) - 0.05, corners.max(axis=0) + 0.05
        c = shell.pos[shell.idx].mean(axis=1)
        side = np.all((c[:, [0, 2]] >= lo) & (c[:, [0, 2]] <= hi), axis=1) & (c[:, 1] > top - 0.9) & (c[:, 1] < top - 0.05)
        self.assertTrue(side.any())
        self.assertTrue((shell.attr[shell.idx[side, 0]] == code("rubble")).all())

    def kirk(self):
        """The kirk: (its nave, its porch, their shell, their fittings)."""
        nave, porch = sorted((u for u in self.units() if u.asset == "kirk"), key=lambda u: -u.outline.area)
        _, shell, fittings = self.shells[(nave.cid, nave.part, "kirk")]
        return nave, porch, shell, fittings

    @staticmethod
    def walls(u):
        """Each of a unit's walls: (edge, start, end, outward normal, length), in world (x, z)."""
        f, out = u.fp.frame, []
        for k, (a, b) in enumerate(plan.edges(u.outline)):
            n = f.world((a + b) / 2 + plan.outward(a, b)) - f.world((a + b) / 2)
            out.append((k, f.world(a), f.world(b), n / np.linalg.norm(n), float(np.linalg.norm(b - a))))
        return out

    def test_the_kirk_is_a_rubble_nave_under_steep_slate_with_a_gabled_side_porch(self):
        """White Hound: "something like the modern kirk at Balquhidder": a rubble nave under a steep slate roof, its
        eaves along its long sides and gables at its ends, a gabled porch against one long side, and no chimney."""
        nave, porch, shell, _ = self.kirk()
        self.assertEqual(len([u for u in self.units() if u.asset == "kirk"]), 2)
        for u in (nave, porch):
            self.assertEqual((u.style.wall, u.style.roof), ("rubble", "slate"))
            self.assertGreaterEqual(u.style.pitch, 45.0)
        walls, angles = self.walls(nave), self.mv.roof_angles(nave)
        longest = max(w[-1] for w in walls)
        sides = {k for k, *_, length in walls if length > longest - 0.01}
        self.assertEqual({k for k in range(len(walls)) if angles[k] < 89.999}, sides)
        porch_plan = porch.fp.frame.to_world(porch.outline).buffer(0.05)
        against = {k for k, a, b, _, _ in walls if LineString([a, b]).intersection(porch_plan).length > 2.0}
        self.assertTrue(against and against <= sides)
        gables = {k for k, angle in enumerate(self.mv.roof_angles(porch)) if angle >= 89.999}
        self.assertEqual(gables, {porch.front, (porch.front + 2) % 4})          # its door in its front gable
        self.assertEqual(self.stacks(shell), [])

    def test_the_kirks_long_walls_have_pointed_lancets_between_buttresses(self):
        """The Balquhidder kirk's side walls (White Hound's model): tall pointed lancets, with buttresses standing out
        from the wall between them."""
        nave, porch, shell, _ = self.kirk()
        glass = np.concatenate([self.faces(shell, g) for g in self.mv.GLASS])
        verts = shell.pos[shell.idx].reshape(-1, 3)
        eave = self.eave(nave)
        walls = self.walls(nave)
        longest = max(w[-1] for w in walls)
        for k, a, b, n, length in (w for w in walls if w[-1] > longest - 0.01):
            t = (b - a) / length
            depth = (glass.mean(axis=1)[:, [0, 2]] - a) @ n
            lights = self.panes(glass[np.abs(depth + 0.16) < 0.08], a, t)
            out = (verts[:, [0, 2]] - a) @ n
            along = (verts[:, [0, 2]] - a) @ t
            proud = np.sort(along[(out > 0.3) & (out < 0.8) & (along > 0) & (along < length) & (verts[:, 1] < eave)])
            buttresses = np.split(proud, np.flatnonzero(np.diff(proud) > 1.0) + 1) if len(proud) else []
            centres = [float(np.mean(x)) for x in buttresses]
            beside = porch.fp.frame.to_world(porch.outline)
            span = None
            if LineString([a, b]).distance(beside) < 0.05:          # the porch stands out from the wall as well
                p = (np.asarray(beside.exterior.coords) - a) @ t
                span = (float(p.min()), float(p.max()))
            with self.subTest(wall=k, porch=span is not None):
                if span:                                             # none of them behind it or crowding it
                    self.assertFalse(any(span[0] - 0.35 < c < span[1] + 0.35 for c in centres))
                    self.assertFalse(any(lo < span[1] and span[0] < hi for lo, hi, _, _ in lights))
            centres += list(span or [])
            with self.subTest(wall=k):
                self.assertGreaterEqual(len(lights), 2)
                self.assertGreaterEqual(len(centres), 2)
                for lo, hi, bottom, top in lights:
                    self.assertGreater(top - bottom, 2.5 * (hi - lo))
                    light = glass[np.abs(depth + 0.16) < 0.08]
                    light = light[((light.mean(axis=1)[:, [0, 2]] - a) @ t > lo - 0.01) &
                                  ((light.mean(axis=1)[:, [0, 2]] - a) @ t < hi + 0.01)].reshape(-1, 3)
                    apex = light[np.argmax(light[:, 1])]
                    self.assertLess(abs(float((apex[[0, 2]] - a) @ t) - (lo + hi) / 2), 0.1 * (hi - lo))
                    corners = light[np.abs(np.abs((light[:, [0, 2]] - a) @ t - (lo + hi) / 2) - (hi - lo) / 2) < 0.01]
                    self.assertLess(float(corners[:, 1].max()), top - 0.15)
                    self.assertTrue(any(c < lo for c in centres) and any(c > hi for c in centres))

    def test_the_kirk_carries_a_lantern_bell_turret_on_its_west_gable(self):
        """White Hound: "something like the modern kirk at Balquhidder but with this traditional, lantern-like
        decoration": on the apex of its westward gable an open turret, a cap on posts over a stone base, rises well
        above the ridge; its other gable is lit by three lancets, the middle one tallest."""
        nave, porch, shell, fittings = self.kirk()
        walls = self.walls(nave)
        longest = max(w[-1] for w in walls)
        west, east = sorted((w for w in walls if w[-1] < longest - 0.01), key=lambda w: -float(w[3] @ [-1.0, 0.0]))
        ridge = self.v.ridge(nave)
        for gable, turret in ((west, True), (east, False)):
            _, a, b, n, length = gable
            apex = (a + b) / 2
            near = [m for m in fittings if np.linalg.norm(m.pos[:, [0, 2]].mean(axis=0) - apex) < 1.5
                    and m.pos[:, 1].max() > ridge + 0.3]
            with self.subTest(gable="west" if turret else "east"):
                if not turret:
                    self.assertEqual(near, [])
                    continue
                self.assertGreater(max(float(m.pos[:, 1].max()) for m in near), ridge + 1.5)
                posts = [m for m in near if np.ptp(m.pos[:, 0]) < 0.3 and np.ptp(m.pos[:, 2]) < 0.3
                         and np.ptp(m.pos[:, 1]) > 0.6]
                self.assertGreaterEqual(len(posts), 4)
        _, a, b, n, length = east
        t = (b - a) / length
        glass = np.concatenate([self.faces(shell, g) for g in self.mv.GLASS])
        on = np.abs((glass.mean(axis=1)[:, [0, 2]] - a) @ n + 0.16) < 0.08
        lights = sorted(self.panes(glass[on], a, t))
        self.assertEqual(len(lights), 3)
        heights = [top - bottom for _, _, bottom, top in lights]
        self.assertEqual(int(np.argmax(heights)), 1)

    def kirkyard(self):
        """The kirkyard's wall and headstones (mesh_village_small), built once per test process."""
        import mesh_village_small as small
        found = shared()
        if "kirkyard" not in found:
            found["kirkyard"] = small.kirkyard(self.v)
        return found["kirkyard"]

    def test_a_low_rubble_wall_rings_the_kirkyard_open_where_the_path_comes_in(self):
        """The user's choice of Hogsmeade's small structures includes a kirkyard wall: a low rubble wall rings the
        kirk some metres out, stepping with the ground, broken only where a road or a building comes near it."""
        from materials import code
        nave, porch, _, _ = self.kirk()
        kirk = shapely.union_all([u.fp.frame.to_world(u.outline) for u in (nave, porch)])
        walls = [m for m in self.kirkyard() if (m.attr == code("rubble")).all()]
        self.assertTrue(walls)
        pts = np.concatenate([m.pos for m in walls])
        away = shapely.distance(kirk, shapely.points(pts[:, [0, 2]]))
        self.assertTrue(np.all((away > 4.0) & (away < 12.0)), (away.min(), away.max()))
        centre = np.asarray(kirk.centroid.coords[0])
        angle = np.arctan2(pts[:, 2] - centre[1], pts[:, 0] - centre[0])
        self.assertGreaterEqual(len(np.unique(np.floor((angle + math.pi) / (2 * math.pi) * 24))), 18)
        above = pts[:, 1] - self.scene.ground(pts[:, 0], pts[:, 2])
        self.assertLess(float(above.max()), 1.6)
        self.assertGreater(float(np.percentile(above, 90)), 0.8)
        for m in walls:
            tris = m.pos[m.idx]
            self.assertEqual(self.v.streets.road_share(tris.mean(axis=1)[:, [0, 2]]), 0.0)

    def test_a_few_headstones_stand_in_the_kirkyard(self):
        """The user's choice also gives the kirkyard a few headstones: upright stone slabs in rows between the kirk and
        its wall, on the nave's side away from its porch, clear of the kirk and of the paths."""
        from materials import code
        nave, porch, _, _ = self.kirk()
        kirk = shapely.union_all([u.fp.frame.to_world(u.outline) for u in (nave, porch)])
        stones = [m for m in self.kirkyard() if (m.attr == code("pale-stone")).all()]
        self.assertTrue(4 <= len(stones) <= 16, len(stones))
        middle = np.asarray(nave.fp.frame.to_world(nave.outline).centroid.coords[0])
        towards_porch = np.asarray(porch.fp.frame.to_world(porch.outline).centroid.coords[0]) - middle
        for m in stones:
            c = m.pos[:, [0, 2]].mean(axis=0)
            g = float(self.scene.ground(*c))
            with self.subTest(stone=tuple(np.round(c, 1))):
                self.assertLess(float((c - middle) @ towards_porch), 0.0)
                self.assertTrue(1.5 < kirk.distance(Point(c)) < 8.0)
                self.assertTrue(0.5 < float(m.pos[:, 1].max()) - g < 1.3 and float(m.pos[:, 1].min()) < g)
                self.assertLess(min(np.ptp(m.pos[:, 0]), np.ptp(m.pos[:, 2])), 0.8)
                self.assertEqual(self.v.streets.road_share([c]), 0.0)

    def gardens(self):
        """The garden walls and gates near the High Street (mesh_village_small), built once per test process."""
        import mesh_village_small as small
        found = shared()
        if "gardens" not in found:
            found["gardens"] = small.garden_walls(self.v)
        return found["gardens"]

    def test_garden_walls_with_gates_front_houses_near_the_high_street(self):
        """The user's choice of Hogsmeade's small structures includes garden walls and gates near the High Street: in
        front of houses near it that stand back from their road, a low rubble wall runs along the road, clear of it and
        of every building, with a painted gate between stone piers straight in front of the house's door, a garden's
        depth (GARDEN 'deep' less 'back') out from it."""
        import mesh_village_small as small
        gardens = self.gardens()
        self.assertGreaterEqual(len(gardens), 8)
        doors = np.concatenate([self.faces(shell, d) for _, shell, _ in self.shells.values() for d in self.mv.DOORS])
        doors = doors.mean(axis=1)[:, [0, 2]]
        for m in gardens:
            wall, gate, piers = (self.material(m, n) for n in ("rubble", "door-green", "pale-stone"))
            c = gate[:, [0, 2]].mean(axis=0)
            with self.subTest(garden=tuple(np.round(c, 1))):
                self.assertTrue(len(wall) and len(gate) and len(piers))
                self.assertLess(self.v.streets.high.distance(Point(c)), 60.0)
                above = wall[:, 1] - self.scene.ground(wall[:, 0], wall[:, 2])
                self.assertTrue(above.max() < 1.6 and np.percentile(above, 90) > 0.7)
                self.assertEqual(self.v.streets.road_share(m.pos[m.idx].mean(axis=1)[:, [0, 2]]), 0.0)
                self.assertGreater(min(o.distance(Point(p)) for o in self.v.outlines.values()
                                       for p in m.pos[:, [0, 2]][::4]), 0.2)
                along = wall[:, [0, 2]] - c
                t = np.linalg.svd(along - along.mean(axis=0))[2][0]            # the wall's direction
                door = doors[np.argmin(np.linalg.norm(doors - c, axis=1))]
                self.assertLess(float(np.linalg.norm(door - c)), 10.0)
                self.assertGreater(float(np.linalg.norm(door - c)), small.GARDEN["deep"] - small.GARDEN["back"])
                self.assertLess(abs(float((door - c) @ t)), 0.6)

    def named_units(self, asset):
        return [u for u in self.units() if u.asset == asset]

    def tall_windows(self, u, shell, wall=None):
        """The separate windows in a unit's front (or wall k) as (lo, hi, bottom, top) panes."""
        _, a, b, n, length = self.walls(u)[u.front if wall is None else wall]
        glass = np.concatenate([self.faces(shell, g) for g in self.mv.GLASS])
        c = glass.mean(axis=1)[:, [0, 2]] - a
        depth, along = c @ n, c @ (b - a) / length
        return self.panes(glass[(depth < -0.05) & (depth > -0.3) & (along > 0) & (along < length) &
                                (self.plan_normals(glass) @ n > 0.9)], a, (b - a) / length)

    def lantern_posts(self, fittings, near, above):
        """The thin upright pieces of a lantern turret within 2 m of world point `near`, rising above height `above`."""
        return [m for m in fittings if np.linalg.norm(m.pos[:, [0, 2]].mean(axis=0) - near) < 2.0
                and m.pos[:, 1].max() > above and np.ptp(m.pos[:, 0]) < 0.3 and np.ptp(m.pos[:, 2]) < 0.3
                and np.ptp(m.pos[:, 1]) > 0.6]

    def test_the_school_is_a_tall_schoolroom_with_a_bell_beside_its_schoolhouse(self):
        """White Hound's map has a primary school but no design of it (inferred): a tall one-storey schoolroom of rubble
        under slate, lit by tall windows, a bell turret on its free gable, where no chimney stands, beside a two-storey
        schoolhouse."""
        units = self.named_units("school")
        room = max(units, key=lambda u: u.outline.area)
        self.assertEqual((room.style.storeys, room.style.wall, room.style.roof), (1.0, "rubble", "slate"))
        self.assertTrue(len(units) > 1 and all(u.style.storeys == 2.0 for u in units if u is not room))
        _, shell, fittings = self.shells[(room.cid, room.part, "school")]
        windows = self.tall_windows(room, shell)
        self.assertGreaterEqual(len(windows), 3)
        self.assertTrue(all(top - bottom > 2.0 for _, _, bottom, top in windows))
        gables = [w for w in self.walls(room) if self.mv.roof_angles(room)[w[0]] >= 89.999 and w[0] not in room.party]
        bells = [(a + b) / 2 for _, a, b, _, _ in gables
                 if len(self.lantern_posts(fittings, (a + b) / 2, self.v.ridge(room))) >= 4]
        self.assertTrue(bells)
        self.assertFalse([c for c, _, _ in self.stacks(shell) if min(np.linalg.norm(c - m) for m in bells) < 1.5])

    def test_the_distillery_dries_its_malt_in_a_kiln_under_a_pagoda_roof(self):
        """White Hound's map has a whisky distillery but no design of it (inferred): whitewashed buildings under slate,
        one a kiln under a steep pyramid roof crowned with a ventilator, the pagoda of a Highland distillery."""
        units = self.named_units("distillery")
        (kiln,) = [u for u in units if u.style.family == "kiln"]
        for u in units:
            self.assertEqual((u.style.wall, u.style.roof), ("harl-white", "slate"))
        angles = self.mv.roof_angles(kiln)
        self.assertTrue(all(angle < 89.999 for angle in angles))                       # hipped all round, a pyramid
        self.assertGreaterEqual(kiln.style.pitch, 50.0)
        sides = [float(np.linalg.norm(q - p)) for p, q in plan.edges(kiln.outline)]
        self.assertGreater(min(sides) / max(sides), 0.8)                                  # a kiln is square
        _, shell, fittings = self.shells[(kiln.cid, kiln.part, "distillery")]
        inside = kiln.fp.frame.to_world(kiln.outline).buffer(-0.3)
        self.assertFalse([c for c, _, _ in self.stacks(shell) if inside.contains(Point(c))])   # its smoke is vented
        top, ridge = kiln.fp.frame.world(np.asarray(kiln.outline.centroid.coords[0])), self.v.ridge(kiln)
        (vent,) = [m for m in self.made_of(fittings, "boarding") if np.linalg.norm(m.pos[:, [0, 2]].mean(axis=0) - top)
                   < 2.5 and m.pos[:, 1].max() > ridge + 0.5]
        _, a, b, n, length = self.walls(kiln)[kiln.front]
        across = np.ptp(vent.pos[:, [0, 2]] @ n)
        self.assertLess(float(vent.pos[:, 1].min()), ridge - across / 2 * math.tan(math.radians(kiln.style.pitch)))

    def test_the_village_hall_is_lit_by_tall_windows(self):
        """White Hound's map has a village hall but no design of it (inferred): a tall one-storey hall under slate, its
        door in the middle of its front between tall windows, and tall windows along its back."""
        hall = max(self.named_units("village-hall"), key=lambda u: u.outline.area)
        self.assertEqual((hall.style.storeys, hall.style.roof), (1.0, "slate"))
        _, shell, _ = self.shells[(hall.cid, hall.part, "village-hall")]
        windows = self.tall_windows(hall, shell)
        self.assertGreaterEqual(len(windows), 2)
        self.assertTrue(all(top - bottom > 2.0 for _, _, bottom, top in windows))
        _, a, b, n, length = self.walls(hall)[hall.front]
        doors = np.concatenate([self.faces(shell, d) for d in self.mv.DOORS])
        depth, along = (doors.mean(axis=1)[:, [0, 2]] - a) @ n, (doors.mean(axis=1)[:, [0, 2]] - a) @ (b - a) / length
        ((lo, hi, _, _),) = self.panes(doors[(depth < -0.05) & (depth > -0.3) & (along > 0) & (along < length)], a,
                                       (b - a) / length)
        self.assertTrue(min(w[0] for w in windows) < lo and max(w[1] for w in windows) > hi)
        back = min(self.walls(hall), key=lambda w: float(w[3] @ n))[0]
        windows = self.tall_windows(hall, shell, wall=back)
        self.assertGreaterEqual(len(windows), 2)
        self.assertTrue(all(top - bottom > 2.0 for _, _, bottom, top in windows))

    def test_the_carriage_hangar_is_one_long_shed_with_carriage_doors(self):
        """White Hound's map has a hangar for the Thestral-drawn carriages but no design of it (inferred): one long
        weatherboarded shed under slate, its front a row of doors wide and high enough for a carriage."""
        (shed,) = self.named_units("carriage-hangar")
        self.assertEqual((shed.style.wall, shed.style.roof), ("weatherboard", "slate"))
        _, shell, _ = self.shells[(shed.cid, shed.part, "carriage-hangar")]
        _, a, b, n, length = self.walls(shed)[shed.front]
        doors = np.concatenate([self.faces(shell, d) for d in self.mv.DOORS])
        depth = (doors.mean(axis=1)[:, [0, 2]] - a) @ n
        wide = self.panes(doors[(depth < -0.05) & (depth > -0.3)], a, (b - a) / length)
        self.assertGreaterEqual(len(wide), 4)
        self.assertTrue(all(hi - lo > 2.8 and top - bottom > 3.0 for lo, hi, bottom, top in wide))
        for lo, hi, bottom, _ in wide:                       # a carriage rolls in: each opens at the ground before it
            p = a + (b - a) / length * (lo + hi) / 2 + n * 0.6
            self.assertLess(bottom - float(self.scene.ground(*p)), 0.35)
        self.assertEqual(self.stacks(shell), [])

    def shop(self, asset):
        """A named shop's one unit, its shell and its fittings."""
        (u,) = self.named_units(asset)
        _, shell, fittings = self.shells[(u.cid, u.part, asset)]
        return u, shell, fittings

    def made_of(self, fittings, name):
        """The fittings wholly of material `name`."""
        return [m for m in fittings if len(m.pos) and len(self.material(m, name)) == len(m.pos)]

    def test_the_post_office_has_an_owl_loft_astride_its_ridge(self):
        """OotP 16: "the post office, from which owls issued at regular intervals". A weatherboarded owl loft stands
        along the middle of the post office's longest ridge, reaching down to meet the roof's slopes, with a row of
        flight holes over the roof along each of its long sides."""
        u, shell, fittings = self.shop("post-office")
        ridge = self.v.ridge(u)
        (loft,) = [m for m in self.made_of(fittings, "weatherboard") if m.pos[:, 1].max() > ridge + 1.0]
        v, f = roofs.roof(u.outline, self.mv.roof_angles(u))
        top = v[:, 2] >= v[:, 2].max() - 1e-6
        p, q = max(((v[i, :2], v[j, :2]) for tri in f for i, j in zip(tri, np.roll(tri, -1)) if top[i] and top[j]),
                   key=lambda e: float(np.linalg.norm(e[1] - e[0])))
        p, q = u.fp.frame.world(p), u.fp.frame.world(q)
        d = (q - p) / np.linalg.norm(q - p)
        across = np.array([-d[1], d[0]])
        xz = loft.pos[:, [0, 2]]
        mid = xz.mean(axis=0)
        self.assertLess(LineString([p, q]).distance(Point(mid)), 0.3)
        self.assertLess(abs(float((mid - (p + q) / 2) @ d)), 0.5)
        self.assertGreater(np.ptp(xz @ d), np.ptp(xz @ across))
        reach = ridge - np.ptp(xz @ across) / 2 * math.tan(math.radians(u.style.pitch))
        self.assertLess(float(loft.pos[:, 1].min()), reach)
        holes = [m for m in self.made_of(fittings, "margin-black") if m.pos[:, 1].min() > ridge]
        sides = [float(np.sign((m.pos[:, [0, 2]].mean(axis=0) - mid) @ across)) for m in holes]
        self.assertGreaterEqual(min(sides.count(1.0), sides.count(-1.0)), 2)

    def test_scrivenshafts_gables_are_crow_stepped(self):
        """White Hound: Scrivenshaft's is "A sort of Mediaeval-style stationers", in a village "probably sixteenth to
        eighteenth century in style". It is slated and gabled, and up both rakes of each free gable stone crow steps rise
        one above another to its apex, each standing clear above the roof beside it."""
        u, shell, fittings = self.shop("scrivenshafts")
        self.assertEqual(u.style.roof, "slate")
        angles, attic, eave = self.mv.roof_angles(u), self.v.attic(u), self.v.eave(u)
        gables = [w for w in self.walls(u) if angles[w[0]] >= 89.999 and w[0] not in u.party]
        self.assertTrue(gables)
        stones = self.made_of(fittings, "pale-stone")
        for k, a, b, n, length in gables:
            t = (b - a) / length
            near = [m for m in stones if abs(float((m.pos[:, [0, 2]].mean(axis=0) - a) @ n)) < 0.4
                    and m.pos[:, 1].min() > eave - 1.0]
            for end, way in ((0.0, 1.0), (length, -1.0)):
                with self.subTest(gable=k, end=end):
                    rake = sorted((float(((m.pos[:, [0, 2]].mean(axis=0) - a) @ t - end) * way), m) for m in near)
                    rake = [(x, m) for x, m in rake if 0.0 < x < length / 2 - 0.2]
                    self.assertGreaterEqual(len(rake), 3)
                    tops = [float(m.pos[:, 1].max()) for _, m in rake]
                    self.assertTrue(all(lo < hi for lo, hi in zip(tops, tops[1:])), tops)
                    for x, m in rake:
                        along = (m.pos[:, [0, 2]] - a) @ t
                        upper = float(along.max() if way > 0 else along.min())         # its end nearer the apex
                        inside = u.fp.frame.plan((a + t * upper - n * 0.3)[None])
                        roof = eave + float(attic(inside)[0])
                        self.assertGreater(float(m.pos[:, 1].max()), roof + 0.05)

    def test_madam_puddifoots_shop_windows_are_steamed_up(self):
        """OotP 25: "a small teashop that Harry had never noticed before. It was a cramped, steamy little place where
        everything seemed to have been decorated with frills or bows. Harry was reminded unpleasantly of Umbridge's
        office." Its front is harled pink, and a misted pane stands in front of the glass of each of its shop windows,
        and of nothing else."""
        u, shell, fittings = self.shop("puddifoots")
        self.assertEqual(u.style.wall, "harl-pink")
        _, a, b, n, length = self.walls(u)[u.front]
        t = (b - a) / length
        glass = self.faces(shell, "glass-shop")
        c = glass.mean(axis=1)[:, [0, 2]] - a
        windows = self.panes(glass[(c @ n < -0.05) & (c @ n > -0.3) & (c @ t > 0) & (c @ t < length)], a, t)
        self.assertTrue(windows)
        panes = [(float((x @ t).min()), float((x @ t).max()), float(m.pos[:, 1].min()), float(m.pos[:, 1].max()),
                  float((x @ n).max())) for m in self.made_of(fittings, "glass-steamed")
                 for x in [m.pos[:, [0, 2]] - a]]
        self.assertEqual(len(panes), len(windows))
        for lo, hi, bottom, top in windows:
            self.assertTrue(any(plo <= lo + 0.01 and phi >= hi - 0.01 and pb <= bottom + 0.01 and pt >= top - 0.01
                                and out < 0.0 for plo, phi, pb, pt, out in panes))

    def test_gladrags_has_a_canvas_blind_over_its_shopfront(self):
        """Gladrags Wizardwear, an outfitter of "London, Paris, Hogsmeade" (GoF 8): a canvas blind is pulled out over
        its shopfront from under its fascia, sloping down to a valance well over a passer-by's head."""
        u, shell, fittings = self.shop("gladrags")
        _, a, b, n, length = self.walls(u)[u.front]
        cloth = np.concatenate([m.pos for m in self.made_of(fittings, "shop-blind")])
        out, along = (cloth[:, [0, 2]] - a) @ n, (cloth[:, [0, 2]] - a) @ ((b - a) / length)
        self.assertGreater(out.max(), 1.2)
        self.assertGreater(np.ptp(along), 0.7 * length)
        self.assertGreater(cloth[out < 0.3, 1].min(), cloth[out > out.max() - 0.1, 1].max())
        self.assertGreater(cloth[:, 1].min() - (self.v.eave(u) - u.style.eaves), 2.0)

    def test_zonkos_is_open(self):
        """HBP 12: "Zonko's Joke Shop had been boarded up". The flyover's Hogsmeade is that of PoA to OotP, where it is
        open: its shop windows are glazed and nothing of it is boarded."""
        u, shell, fittings = self.shop("zonkos")
        _, a, b, n, length = self.walls(u)[u.front]
        glass = self.faces(shell, "glass-shop")
        d = (glass.mean(axis=1)[:, [0, 2]] - a) @ n
        self.assertGreaterEqual(len(self.panes(glass[(d < -0.05) & (d > -0.3)], a, (b - a) / length)), 2)
        self.assertEqual(len(self.faces(shell, "boarding")), 0)
        self.assertFalse(self.made_of(fittings, "boarding"))

    def test_dervish_and_banges_hangs_out_a_bracket_clock(self):
        """PoA 5: "They sell that sort of thing in Dervish and Banges, magical instruments and stuff". Like an
        instrument maker's, it hangs a clock out over the pavement on an iron bracket, over its shopfront and under its
        eaves: a white face on either side of a drum, looking up and down the street."""
        u, shell, fittings = self.shop("dervish-and-banges")
        _, a, b, n, length = self.walls(u)[u.front]
        t = (b - a) / length
        floor, eave = self.v.eave(u) - u.style.eaves, self.v.eave(u)
        faces = self.made_of(fittings, "clock-face")
        self.assertEqual(len(faces), 2)
        for m in faces:
            x = m.pos[:, [0, 2]] - a
            self.assertLess(np.ptp(x @ t), 0.02)
            self.assertGreater(min(np.ptp(x @ n), np.ptp(m.pos[:, 1])), 0.5)
            self.assertGreater(float((x @ n).min()), 0.3)
            self.assertGreater(float(m.pos[:, 1].min()), floor + 3.2)
            self.assertLess(float(m.pos[:, 1].max()), eave - 0.3)
        c = [m.pos.mean(axis=0) for m in faces]
        self.assertGreater(abs(float((c[0] - c[1])[[0, 2]] @ t)), 0.1)
        bracket = [m for m in self.made_of(fittings, "black-iron")
                   if ((m.pos[:, [0, 2]] - a) @ n).min() < 0.05 and ((m.pos[:, [0, 2]] - a) @ n).max() > 0.3
                   and abs(float(m.pos[:, 1].mean()) - (c[0][1] + c[1][1]) / 2) < 0.3]
        self.assertTrue(bracket)

    def test_every_door_opens_on_its_buildings_front_at_its_floor(self):
        """Every building but an outshot has a door in its front wall, at its floor; each of its front doors stands
        clear of the ground before it, and if raised has steps up to it: a block that rises from the ground before the
        door to its sill. The kirk's nave is entered through its porch."""
        for (cid, part, asset), (group, shell, fittings) in self.shells.items():
            doors = np.concatenate([self.faces(shell, d) for d in self.mv.DOORS])
            blocks = [(shapely.convex_hull(shapely.multipoints(m.pos[:, [0, 2]])), float(m.pos[:, 1].min()),
                       float(m.pos[:, 1].max())) for m in fittings if len(m.pos)]
            for u in group:
                if u.style.family in ("outshot", "kirk"):
                    continue
                mid, out, (a, b) = self.front(u)
                length = float(np.linalg.norm(b - a))
                t = (b - a) / length
                c = doors.mean(axis=1)
                depth, along = (c[:, [0, 2]] - a) @ out, (c[:, [0, 2]] - a) @ t
                on = (depth < -0.05) & (depth > -0.3) & (along > 0) & (along < length)
                with self.subTest(component=cid, part=u.part, unit=u.k):
                    self.assertTrue(on.any(), "no door recessed in the front wall")
                    for lo, hi, sill, _ in self.panes(doors[on], a, t):
                        p = a + t * (lo + hi) / 2 + out * 0.6
                        g = float(self.scene.ground(*p))
                        self.assertGreater(sill - g, 0.05)                                  # never buried
                        if sill - g > 0.35:                                                   # then steps lead up
                            self.assertTrue(any(hull.distance(Point(p)) < 0.6 and low < g and top > sill - 0.25
                                                for hull, low, top in blocks), "a raised door without steps")

    def test_a_base_course_keeps_to_the_foot_of_its_walls(self):
        """A dark-painted base course rises no more than 0.5 m above the lowest ground under its own building's walls,
        however steep the street: never a storey-high dark band on the downhill side, nor on the lower houses of a
        terrace stepping down a slope."""
        from materials import code
        dark = code("plinth-dark")
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            based = [u for u in group if u.style.plinth != u.style.wall]
            n = shell.nrm[shell.idx[:, 0]]
            sel = (shell.attr[shell.idx[:, 0]] == dark).all(axis=1) & (np.abs(n[:, 1]) < 0.3)
            if not based or not sel.any():
                continue
            tris = shell.pos[shell.idx[sel]]
            frame = based[0].fp.frame
            behind = frame.plan(tris.mean(axis=1)[:, [0, 2]] - 0.05 * n[sel][:, [0, 2]])        # a face's own building
            owner = np.full(len(tris), -1)
            for i, u in enumerate(based):
                owner[shapely.contains_xy(u.outline, behind[:, 0], behind[:, 1])] = i
            lowest = np.array([float(np.min(self.scene.ground(*u.fp.frame.world(np.asarray(u.outline.exterior.coords)).T)))
                               for u in based] + [-np.inf])                     # a face behind no building fails
            over = tris[:, :, 1].max(axis=1) - (lowest[owner] + 0.5)
            with self.subTest(component=cid, asset=asset):
                self.assertLessEqual(float(over.max()), 0.01)

    def test_a_wall_of_its_own_stone_down_to_its_foot_is_one_face(self):
        """A building whose base is of its wall's own stone is walled in one piece: no seam runs round its free corners
        between the lowest ground and its eaves (whose overhang meets its walls at most 0.75 m below them)."""
        if "_outline_tree" not in type(self).__dict__:
            type(self)._outline_tree = (shapely.STRtree(list(self.v.outlines.values())), list(self.v.outlines))
        tree, cids = self._outline_tree
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            for u in group:
                if u.style.plinth != u.style.wall:
                    continue
                f = u.fp.frame
                edges = plan.edges(u.outline)
                corners = f.world(np.array([a for a, _ in edges]))
                eave = self.eave(u)
                lowest = float(np.min(self.scene.ground(*corners.T)))
                own = [w.fp.frame.to_world(w.outline) for w in self.v.units[cid] if w is not u]
                for k, xz in enumerate(corners):
                    if k in u.party or (k - 1) % len(edges) in u.party:
                        continue
                    if any(cids[i] != cid for i in tree.query(Point(xz), predicate="dwithin", distance=0.5)) \
                            or any(o.distance(Point(xz)) < 0.5 for o in own):
                        continue                                  # another building stands at this corner
                    at = shell.pos[np.hypot(shell.pos[:, 0] - xz[0], shell.pos[:, 2] - xz[1]) < 0.01][:, 1]
                    seam = at[(at > lowest + 0.01) & (at < eave - 0.75)]
                    with self.subTest(component=cid, part=u.part, unit=u.k, corner=k):
                        self.assertEqual(len(seam), 0, np.round(seam, 2).tolist())

    def test_walls_reach_below_the_ground_all_round(self):
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            for u in group:
                ring = u.fp.frame.world(np.asarray(u.outline.exterior.coords)[:-1])
                for x, z in ring:
                    near = shell.pos[np.hypot(shell.pos[:, 0] - x, shell.pos[:, 2] - z) < 0.5]
                    with self.subTest(component=cid, part=u.part, unit=u.k, corner=(round(x, 1), round(z, 1))):
                        self.assertLess(float(near[:, 1].min()), float(self.scene.ground(x, z)) - 0.5)

    def test_no_window_sinks_into_the_ground_before_it(self):
        """Every upright pane of glass stands at least 0.3 m above the ground 0.9 m in front of it, past its reveal. A
        floor follows the highest ground along its building's front, and on a slope the ground can rise higher behind
        or beside the building, where its other windows are."""
        from materials import code
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            attr = shell.attr[shell.idx[:, 0]]
            n = shell.nrm[shell.idx[:, 0]]
            glass = np.zeros(len(attr), bool)
            for name in self.mv.GLASS:
                glass |= (attr == code(name)).all(axis=1)
            glass &= np.abs(n[:, 1]) < 0.3
            if not glass.any():
                continue
            tris = shell.pos[shell.idx[glass]]
            out = n[glass][:, [0, 2]] / np.linalg.norm(n[glass][:, [0, 2]], axis=1, keepdims=True)
            at = tris.mean(axis=1)[:, [0, 2]] + 0.9 * out
            clear = tris[:, :, 1].min(axis=1) - self.scene.ground(at[:, 0], at[:, 1])
            with self.subTest(component=cid, part=part, asset=asset):
                self.assertGreaterEqual(float(clear.min()), 0.3)

    def test_windows_are_recessed_behind_their_walls(self):
        """Glass, or the planks boarding a window up, stands behind its wall's face (a bay's glass behind the bay's
        face, 0.6 m out from its wall), and every front over 4 m with room ahead of two of its bays has a ground-floor
        window beside its door, or a bay (a carriage shed's front is its doors)."""
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            glass = np.concatenate([self.faces(shell, g) for g in self.mv.GLASS + ("boarding",)])
            facing = np.cross(glass[:, 1] - glass[:, 0], glass[:, 2] - glass[:, 0])[:, [0, 2]] if len(glass) \
                else np.zeros((0, 2))
            for u in group:
                mid, out, (a, b) = self.front(u)
                length = float(np.linalg.norm(b - a))
                c = glass.mean(axis=1)[:, [0, 2]] if len(glass) else np.zeros((0, 2))
                depth = (c - a) @ out
                along = (c - a) @ ((b - a) / length)
                ahead = facing @ out > 0.9 * np.linalg.norm(facing, axis=1)
                front = ahead & (depth > -0.5) & (depth < 1.0) & (along > 0) & (along < length)
                on = front & (depth < 0.3)
                floor = self.mv.floor_level(self.scene, u) + self.mv.FLOOR_OVER_M
                ground = front & (glass[:, :, 1].min(axis=1) < floor + 1.2) if len(glass) else front
                is_open = self.v.opener(u.fp.frame, u.outline)
                room = sum(is_open(u.front, s, 1.0) for s in self.mv.bays(length)[0])
                with self.subTest(component=cid, part=u.part, unit=u.k):
                    if u.style.family not in ("outshot", "hangar") and length > 4.0 and room >= 2:
                        self.assertTrue(ground.any(), "a front without a ground-floor window")
                    self.assertTrue(np.all(depth[on] < -0.05), "glass standing proud of its wall")

    def test_a_back_wall_off_the_high_street_has_one_window(self):
        """Off the High Street, each of a house's walls facing away from its front holds at most one window; the named
        buildings are lit as their own designs say."""
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            glass = np.concatenate([self.faces(shell, g) for g in self.mv.GLASS])
            if not len(glass):
                continue
            facing = np.cross(glass[:, 1] - glass[:, 0], glass[:, 2] - glass[:, 0])[:, [0, 2]]
            facing /= np.linalg.norm(facing, axis=1, keepdims=True)
            c = glass.mean(axis=1)[:, [0, 2]]
            for u in group:
                if u.high or u.asset in self.mv.NAMED:
                    continue
                f = u.fp.frame
                edges = plan.edges(u.outline)
                ahead = plan.outward(*edges[u.front])
                for k, (a, b) in enumerate(edges):
                    if k in u.party or float(plan.outward(a, b) @ ahead) > -0.9:
                        continue
                    wa, wb = f.world(a), f.world(b)
                    length = float(np.linalg.norm(wb - wa))
                    t = (wb - wa) / length
                    out = f.world((a + b) / 2 + plan.outward(a, b)) - f.world((a + b) / 2)
                    rel = c - wa
                    on = (facing @ out > 0.9) & (np.abs(rel @ out) < 0.5) & (rel @ t > 0) & (rel @ t < length)
                    with self.subTest(component=cid, part=u.part, unit=u.k, edge=k):
                        self.assertLessEqual(len(self.spans(glass[on], wa, t)), 1)

    def neighbour_tris(self, cid, part, asset, reach=4.0):
        """The triangles of every other built shell within `reach` of this one."""
        if "_boxes" not in type(self).__dict__:
            type(self)._boxes = {k: (v[1].pos[:, [0, 2]].min(axis=0), v[1].pos[:, [0, 2]].max(axis=0))
                                 for k, v in self.shells.items()}
        lo, hi = self._boxes[(cid, part, asset)]
        out = [v[1].pos[v[1].idx] for k, v in self.shells.items() if k != (cid, part, asset)
               and np.all(self._boxes[k][0] <= hi + reach) and np.all(self._boxes[k][1] >= lo - reach)]
        return np.concatenate(out) if out else np.zeros((0, 3, 3))

    def test_every_shell_is_closed(self):
        """Each building shell, its corners joined where they coincide, shares every edge between an even number of
        triangles (two, or four where two walls touch along a corner): no edge borders a crack where a thin triangle was
        dropped. Triangles with no area, left out as the payload's encoder leaves them out, leave such edges along one
        line, under 0.1 mm off it, where every corner meets an even number of them, and no crack: those are let be."""
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            with self.subTest(component=cid, asset=asset):
                self.assertEqual(len(cracks(shell)), 0)

    def test_hardly_a_triangle_is_laid_flat(self):
        """The tidy lays a sliver flat only where splitting cannot remove it, for that leaves a seam the payload's
        quantization can open by a hair: hardly a shell triangle is laid flat, each leaving three unshared edges."""
        flat = total = 0
        for group, shell, _ in self.shells.values():
            _, ids = np.unique(np.round(shell.pos * 1e5).astype(np.int64), axis=0, return_inverse=True)
            f = ids.ravel()[shell.idx.astype(np.int64)]
            _, counts = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1),
                                  axis=0, return_counts=True)
            flat += int((counts % 2).sum()) / 3
            total += len(f)
        self.assertLessEqual(flat, 0.0005 * total)

    def test_every_normal_is_a_unit_vector(self):
        """Every corner of every shell and fitting carries a finite unit normal, thatch smoothed or not: a triangle
        with no area has no normal to give its corners."""
        for (cid, part, asset), (group, shell, fittings) in self.shells.items():
            for k, m in enumerate([shell] + fittings):
                length = np.linalg.norm(m.nrm, axis=1)
                with self.subTest(component=cid, asset=asset, piece=k):
                    self.assertTrue(np.all(np.abs(length - 1.0) < 1e-6), int((~(np.abs(length - 1.0) < 1e-6)).sum()))

    def test_no_shell_triangle_is_a_sliver(self):
        """No triangle of a building's shell is thinner than SLIVER_M, which 16-bit quantization could turn over."""
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            tris = shell.pos[shell.idx]
            longest = np.max([np.linalg.norm(tris[:, (i + 1) % 3] - tris[:, i], axis=1) for i in range(3)], axis=0)
            area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
            with self.subTest(component=cid, part=part, asset=asset):
                self.assertEqual(int((area <= self.mv.SLIVER_M * longest).sum()), 0)

    def test_no_triangle_thicker_than_a_sliver_turns_over_in_the_payload(self):
        """The payload stores each chunk's corners to 16 bits across its box, which moves a corner at most half the
        box's diagonal over 65535, so a corner moves at most the diagonal over 65535 off the side facing it. That is
        under SLIVER_M for every chunk of every village asset, so a triangle that is no sliver cannot turn over."""
        import build_meshes
        pieces = {}
        for (cid, part, asset), (group, shell, fittings) in self.shells.items():
            pieces.setdefault(asset, []).extend([shell] + fittings)
        for asset, extra in self.v.small().items():
            pieces.setdefault(asset, []).extend(extra)
        widest = max(float(np.linalg.norm(m.pos.max(axis=0) - m.pos.min(axis=0)))
                     for asset in pieces for _, m in build_meshes.chunks_of(pieces[asset]))
        self.assertLess(widest / 65535, self.mv.SLIVER_M)

    def test_every_window_and_door_looks_out_unobstructed(self):
        """Seen from its middle or near any of its corners, nothing (its own building's eaves or hoods, or another
        building) stands within 3 m in front of any glass, or within 0.6 m of a door."""
        from materials import code
        corners = np.array([[1, 1, 1], [8, 1, 1], [1, 8, 1], [1, 1, 8]], np.float64)
        corners /= corners.sum(axis=1, keepdims=True)
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            tris = shell.pos[shell.idx]
            attr = shell.attr[shell.idx[:, 0]]
            glass = np.zeros(len(attr), bool)
            for name in self.mv.GLASS:
                glass |= (attr == code(name)).all(axis=1)
            door = np.zeros(len(attr), bool)
            for name in self.mv.DOORS:
                door |= (attr == code(name)).all(axis=1)
            looks = glass | door
            origins = np.einsum("sk,tkd->tsd", corners, tris[looks]).reshape(-1, 3)
            dirs = shell.nrm[shell.idx[looks, 0]]
            dirs = np.repeat(dirs / np.linalg.norm(dirs, axis=1, keepdims=True), len(corners), axis=0)
            clear = np.repeat(np.where(glass[looks], 3.0, 0.6), len(corners))
            hits = np.minimum(first_hit(origins, dirs, tris),
                              first_hit(origins, dirs, self.neighbour_tris(cid, part, asset)))
            with self.subTest(component=cid, asset=asset):
                self.assertTrue(np.all(hits > clear), np.round(origins[hits <= clear], 1).tolist()[:6])

    @staticmethod
    def projection(solid):
        """A manifold3d solid's shadow on the plan, as a shapely geometry."""
        rings = [np.asarray(c) for c in solid.project().to_polygons() if len(c) >= 3]
        signed = [float(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])) for c in rings]
        outer = shapely.union_all([Polygon(c) for c, s in zip(rings, signed) if s > 0])
        return outer.difference(shapely.union_all([Polygon(c) for c, s in zip(rings, signed) if s < 0]))

    def skin(self, u):
        """A unit's roof skin as its shell was built: (skin, the height its plan was built from)."""
        return self.skins[(u.cid, u.part, u.k)]

    def test_no_roof_reaches_over_another_building(self):
        """A building's eaves and verges never reach over another building's footprint: its party walls stop them."""
        for cid in sorted(self.scope):
            near = [w for c, w in self.v.outlines.items() if c != cid and w.distance(self.v.outlines[cid]) < 3.0]
            for u in self.v.units[cid]:
                skin, _ = self.skin(u)
                if skin is None:
                    continue
                cover = u.fp.frame.to_world(self.projection(skin))
                others = near + [w.fp.frame.to_world(w.outline) for w in self.v.units[cid] if w is not u]
                over = sum(cover.intersection(o.buffer(-0.05)).area for o in others)
                with self.subTest(component=cid, part=u.part, unit=u.k):
                    self.assertLess(over, 0.05)

    def skin_below_attic(self, u, skin, gap=0.005, y_ref=0.0, attic=None):
        """Where a unit's skin rides less than `gap` above its attic (`attic`, else the village's own), looking straight
        down 0.3 m or more inside its walls at points 1 m apart: (plan x, y, how far above the attic), for a unit built
        from height y_ref."""
        inner = u.outline.buffer(-0.3)
        x0, y0, x1, y1 = inner.bounds if not inner.is_empty else (0, 0, 0, 0)
        pts = np.array([(x, y) for x in np.arange(x0, x1, 1.0) for y in np.arange(y0, y1, 1.0)
                        if inner.contains(Point(x, y))]).reshape(-1, 2)
        if not len(pts):
            return np.zeros((0, 3))
        m = skin.to_mesh()
        tris = np.asarray(m.vert_properties, np.float64)[:, :3][np.asarray(m.tri_verts)]
        attic = (attic or self.v.attic(u))(pts) + self.eave(u)
        met = y_ref + 1e4 - first_hit(np.column_stack([pts, np.full(len(pts), 1e4)]),
                                      np.tile([0.0, 0.0, -1.0], (len(pts), 1)), tris)
        low = met < attic + gap
        return np.column_stack([pts[low], met[low] - attic[low]])

    def test_every_roof_skin_rides_its_attic_by_eps(self):
        """Looking straight down inside a building's walls, its roof skin rides EPS_M above its attic: one roofer
        builds both, the skin's outline pushed out for the eaves and lowered by their rise, so its slopes and its cap
        run on from the attic's. (The attic raised by half EPS_M, which the skin also covers, shows only where the
        roofers differ: test_a_skin_covers_its_attic_whichever_roofer_builds_them. The skin is probed as built, before
        the shell's tidy, which may join a skin's corner to the attic's beneath it.)"""
        for u in self.units():
            skin, y_ref = self.skin(u)
            with self.subTest(component=u.cid, part=u.part, unit=u.k):
                self.assertIsNotNone(skin)
                low = self.skin_below_attic(u, skin, self.mv.EPS_M - 0.001, y_ref)
                self.assertEqual(len(low), 0, np.round(low, 3).tolist()[:4])

    def test_a_skin_covers_its_attic_whichever_roofer_builds_them(self):
        """Should CGAL's roof over an attic be refused, alone or with the one over its skin, the in-repo roofer's slabs
        need not run on from CGAL's slopes, nor from its own over the outline pushed out for the eaves: the skin still
        covers the attic. Checked on every plan along the High Street but a rectangle, which both roofers roof alike: a
        check of the roofers' fallback, not of the village's plans."""
        from unittest import mock
        real = roofs._stands
        for u in (u for cid in sorted(self.scope & high_street(self.v, self.mv)) for u in self.v.units[cid]):
            x0, y0, x1, y1 = u.outline.bounds
            if abs(u.outline.area - (x1 - x0) * (y1 - y0)) < 1e-6:
                continue
            for refused in ("attic", "every roof"):
                stands = (lambda v, f, area: False) if refused == "every roof" else \
                    (lambda v, f, area, a=u.outline.area: abs(area - a) > 1e-9 and real(v, f, area))
                with mock.patch.object(roofs, "_stands", stands):
                    skin = self.mv._Building(self.v, u.fp, u, 0.0).skin
                    with self.subTest(component=u.cid, part=u.part, unit=u.k, refused=refused):
                        self.assertIsNotNone(skin)
                        attic = self.mv.attic_top(u.outline, self.mv.roof_angles(u))      # as the mocked roofer builds it
                        low = self.skin_below_attic(u, skin, attic=attic)
                        self.assertEqual(len(low), 0, np.round(low, 2).tolist()[:4])

    def openings_on_front(self, u, mesh, names):
        """The faces of `names` in a unit's front wall: (triangles, their distance along the front, the front's length,
        its outward normal, its start, its direction)."""
        _, out, (a, b) = self.front(u)
        length = float(np.linalg.norm(b - a))
        t = (b - a) / length
        faces = np.concatenate([self.faces(mesh, n) for n in names])
        c = faces.mean(axis=1)[:, [0, 2]] if len(faces) else np.zeros((0, 2))
        along = (c - a) @ t
        on = (np.abs((c - a) @ out) < 0.5) & (along > 0) & (along < length)
        return faces[on], along[on], length, out, a, t

    def eave(self, u):
        return self.mv.floor_level(self.scene, u) + self.mv.FLOOR_OVER_M + u.style.eaves

    def test_attic_storeys_have_windows_above_their_eaves(self):
        """A storey-and-a-half building lights its attic from its front: through dormers rising through the eaves, or
        a window in a gable front; the High Street has a good few of them."""
        lit = 0
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            for u in group:
                if u.style.storeys != 1.5:
                    continue
                glass, _, length, _, _, _ = self.openings_on_front(u, shell, self.mv.GLASS)
                if length < 4.0:
                    continue
                with self.subTest(component=cid, part=u.part, unit=u.k):
                    self.assertTrue(len(glass) and glass[:, :, 1].max() > self.eave(u) + 0.3)
                    lit += 1
        self.assertGreaterEqual(lit, 5)

    def dormers(self, u, shell):
        """A unit's dormer windows (glass rising through its eaves, where a gable's window stands wholly above them),
        as their extents (lo, hi) along its front, with the front's outward normal, start and direction, and the unit's
        eaves' height."""
        glass, _, _, out, a, t = self.openings_on_front(u, shell, self.mv.GLASS)
        eave = self.eave(u)
        y = glass[:, :, 1]
        across = (y.min(axis=1) < eave - 0.05) & (y.max(axis=1) > eave + 0.3) if len(glass) else np.zeros(0, bool)
        return self.spans(glass[across], a, t), out, a, t, eave

    def test_dormers_run_back_into_their_roofs(self):
        """Nothing upright within a dormer's depth behind it faces back up its roof: its hood and cheeks run into the
        slope rather than ending in a gable standing out of it, short of the ridge or beside a hipped end."""
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            tris = shell.pos[shell.idx]
            n = shell.nrm[shell.idx[:, 0]]
            for u in group:
                if u.style.storeys != 1.5:
                    continue
                spans, out, a, t, eave = self.dormers(u, shell)
                c = tris.mean(axis=1)
                depth = (a - c[:, [0, 2]]) @ out
                pos = (c[:, [0, 2]] - a) @ t
                upright_back = (n[:, [0, 2]] @ -out > 0.7) & (np.abs(n[:, 1]) < 0.3) & (c[:, 1] > eave + 0.2)
                for lo, hi in spans:
                    wd = hi - lo + 2 * self.mv.DORMER["cheek"]
                    reach = self.mv.dormer_depth(u.style.pitch, wd) + 0.3
                    apex = self.mv.DORMER["window"] - self.mv.DORMER["below"] + self.mv.DORMER["lintel"] \
                        + wd / 2 * math.tan(math.radians(u.style.pitch))           # the hood's top, above the eaves
                    near = upright_back & (depth > 0.05) & (depth < reach) & (pos > lo - 0.35) & (pos < hi + 0.35) \
                        & (c[:, 1] < eave + apex + 0.2)
                    with self.subTest(component=cid, part=u.part, unit=u.k, dormer=round((lo + hi) / 2, 1)):
                        self.assertFalse(near.any(), np.round(c[near], 1).tolist()[:4])

    def test_dormers_break_through_their_eaves(self):
        """The eave stops either side of a dormer: nothing of its building stands just in front of a dormer's face."""
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            tris = shell.pos[shell.idx]
            for u in group:
                if u.style.storeys != 1.5:
                    continue
                spans, out, a, t, eave = self.dormers(u, shell)
                for lo, hi in spans:
                    xz = np.array([a + t * (lo + (hi - lo) * f) + out * d
                                   for f in (0.25, 0.5, 0.75) for d in (0.15, 0.35)])
                    origins = np.column_stack([xz[:, 0], np.full(len(xz), eave + 3.0), xz[:, 1]])
                    hits = first_hit(origins, np.tile([0.0, -1.0, 0.0], (len(xz), 1)), tris)
                    with self.subTest(component=cid, part=u.part, unit=u.k, dormer=round((lo + hi) / 2, 1)):
                        self.assertTrue(np.isinf(hits).all(), np.round(origins[~np.isinf(hits)], 1).tolist()[:3])

    @staticmethod
    def spans(faces, a, t):
        """The extents (lo, hi) along a front (from a, along t) of the separate openings among `faces`: triangles whose
        extents overlap belong to one opening."""
        out = []
        for p in sorted(((f[:, [0, 2]] - a) @ t for f in faces), key=lambda p: p.min()):
            if out and p.min() <= out[-1][1] + 0.05:
                out[-1][1] = max(out[-1][1], float(p.max()))
            else:
                out.append([float(p.min()), float(p.max())])
        return out

    @staticmethod
    def panes(faces, a, t):
        """Each separate opening among `faces` (triangles in a front, from a along t) as (lo, hi, bottom, top): triangles
        whose extents overlap both along the front and in height belong to one opening."""
        boxes = [[float(p.min()), float(p.max()), float(f[:, 1].min()), float(f[:, 1].max())]
                 for f, p in ((f, (f[:, [0, 2]] - a) @ t) for f in faces)]
        joined = True
        while joined:
            joined = False
            for i, j in ((i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
                x, y = boxes[i], boxes[j]
                if x[0] <= y[1] + 0.05 and y[0] <= x[1] + 0.05 and x[2] <= y[3] + 0.05 and y[2] <= x[3] + 0.05:
                    boxes[i] = [min(x[0], y[0]), max(x[1], y[1]), min(x[2], y[2]), max(x[3], y[3])]
                    del boxes[j]
                    joined = True
                    break
        return boxes

    def test_honeydukes_windows_are_mullioned(self):
        """PoA 10: "nodding through the mullioned windows". Every window in Honeydukes' front, the shop's and those of
        the owners' rooms over it, is divided by a narrow upright bar standing in it from sill to head."""
        (u,) = [u for u in self.units() if u.asset == "honeydukes"]
        _, shell, fittings = self.shells[(u.cid, u.part, "honeydukes")]
        _, out, (a, b) = self.front(u)
        t = (b - a) / np.linalg.norm(b - a)
        glass = np.concatenate([self.faces(shell, g) for g in self.mv.GLASS])
        depth = (glass.mean(axis=1)[:, [0, 2]] - a) @ out
        windows = self.panes(glass[(depth < -0.05) & (depth > -0.3) & (self.plan_normals(glass) @ out > 0.9)], a, t)
        self.assertGreaterEqual(len(windows), 4)
        bars = []
        for m in fittings:
            along, inside = (m.pos[:, [0, 2]] - a) @ t, (m.pos[:, [0, 2]] - a) @ out
            if np.ptp(along) < 0.15 and inside.max() < 0.0 and inside.min() > -0.3:
                bars.append((along.min(), along.max(), m.pos[:, 1].min(), m.pos[:, 1].max()))
        for lo, hi, bottom, top in windows:
            with self.subTest(window=(round((lo + hi) / 2, 1), round(bottom, 1))):
                self.assertTrue(any(lo < b0 and b1 < hi and z0 <= bottom + 0.05 and z1 >= top - 0.05
                                    for b0, b1, z0, z1 in bars))

    def test_no_opening_cuts_into_its_eaves(self):
        """An opening's cut stays inside its wall: no downward face of anything but roofing stands out from a wall
        below its eaves (as the head of a cut into an eave's underside would)."""
        from materials import code
        roofing = np.array([code(name) for name in self.mv.ROOFING])
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            tris = shell.pos[shell.idx]
            c = tris.mean(axis=1)
            n = shell.nrm[shell.idx[:, 0]]
            attr = shell.attr[shell.idx[:, 0]]
            down = (n[:, 1] < -0.9) & ~(attr[:, None, :] == roofing[None]).all(axis=2).any(axis=1)
            for u in group:
                eave = self.eave(u)
                f = u.fp.frame
                for k, (a, b) in enumerate(plan.edges(u.outline)):
                    if k in u.party:
                        continue
                    wa, wb = f.world(a), f.world(b)
                    length = float(np.linalg.norm(wb - wa))
                    t = (wb - wa) / length
                    out = f.world((a + b) / 2 + plan.outward(a, b)) - f.world((a + b) / 2)
                    rel = c[:, [0, 2]] - wa
                    bad = down & (rel @ out > 0.02) & (rel @ out < 0.5) & (rel @ t > 0) & (rel @ t < length) \
                        & (c[:, 1] > eave - 1.5) & (c[:, 1] < eave + 0.1)
                    with self.subTest(component=cid, part=u.part, unit=u.k, edge=k):
                        self.assertFalse(bad.any(), np.round(c[bad], 2).tolist()[:4])

    def test_a_wall_against_other_buildings_for_most_of_its_length_is_a_party_wall(self):
        """A wall standing against other buildings (within 5 cm of its own footprint's, 35 cm of others') for more than
        half its length is a party wall, and one standing against them for less is not."""
        for cid, us in ((c, self.v.units[c]) for c in sorted(self.scope)):
            for u in us:
                f = u.fp.frame
                world = f.to_world(u.outline)
                beside = shapely.union_all(
                    [w.fp.frame.to_world(w.outline).buffer(0.05) for w in us if w is not u and w.fp is u.fp]
                    + [w.fp.frame.to_world(w.fp.outline).buffer(0.35) for w in us if w.fp is not u.fp]
                    + [o.buffer(0.35) for c, o in self.v.outlines.items() if c != cid and o.distance(world) < 1.0])
                for e, (a, b) in enumerate(plan.edges(u.outline)):
                    edge = LineString(f.world(np.array([a, b])))
                    share = edge.intersection(beside).length / edge.length
                    if abs(share - 0.5) > 0.05:
                        with self.subTest(component=cid, part=u.part, unit=u.k, edge=e, share=round(share, 2)):
                            self.assertEqual(e in u.party, share > 0.5)

    def test_chimney_stacks_stand_on_their_walls(self):
        """Every stack top's corners lie over its buildings' walls (give or take 3 cm): a stack on a party wall stands
        where both buildings meet, never out over open ground."""
        from materials import code
        stone = code("pale-stone")
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            tris = shell.pos[shell.idx]
            n = shell.nrm[shell.idx[:, 0]]
            cope = (shell.attr[shell.idx[:, 0]] == stone).all(axis=1) & (n[:, 1] > 0.9)
            standing = shapely.union_all([u.fp.frame.to_world(u.outline) for u in self.v.units[cid]
                                          if u.part == part]).buffer(0.03)
            corners = tris[cope].reshape(-1, 3)[:, [0, 2]]
            outside = ~shapely.contains_xy(standing, corners[:, 0], corners[:, 1])
            with self.subTest(component=cid, asset=asset):
                self.assertFalse(outside.any(), np.round(corners[outside], 2).tolist()[:4])

    def stacks(self, shell):
        """Each chimney stack in a shell, from its pale stone top (top triangles sharing corners are one stack's): (its
        world centre (x, z), its top, its top's corners (x, z))."""
        from materials import code
        tris = shell.pos[shell.idx]
        n = shell.nrm[shell.idx[:, 0]]
        tops = tris[(shell.attr[shell.idx[:, 0]] == code("pale-stone")).all(axis=1) & (n[:, 1] > 0.9)]
        _, corner = np.unique(np.round(tops.reshape(-1, 3) * 1000).astype(np.int64), axis=0, return_inverse=True)
        corner = corner.reshape(-1, 3)
        root = list(range(len(tops)))

        def find(i):
            while root[i] != i:
                i = root[i]
            return i
        first = {}
        for t, cs in enumerate(corner):
            for c in cs:
                a, b = find(t), find(first.setdefault(int(c), t))
                root[max(a, b)] = min(a, b)
        out = []
        for r in sorted({find(t) for t in range(len(tops))}):
            pts = tops[[t for t in range(len(tops)) if find(t) == r]].reshape(-1, 3)
            xz = pts[:, [0, 2]]
            out.append(((xz.min(axis=0) + xz.max(axis=0)) / 2, float(pts[:, 1].max()), np.unique(np.round(xz, 3), axis=0)))
        return out

    def test_chimney_stacks_sit_down_in_the_roofs_under_them(self):
        """No stack hangs in the air: straight down from just under any point of its pale stone top, its shell runs on
        at least to the roof of the building it stands over there (a stack on a party wall beside a lower neighbour
        starts down in the neighbour's roof)."""
        from materials import code
        stone = code("pale-stone")
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            tris = shell.pos[shell.idx]
            n = shell.nrm[shell.idx[:, 0]]
            tops = tris[(shell.attr[shell.idx[:, 0]] == stone).all(axis=1) & (n[:, 1] > 0.9)]
            if not len(tops):
                continue
            centre = tops.mean(axis=1, keepdims=True)
            pts = np.concatenate([centre, centre + 0.8 * (tops - centre)], axis=1).reshape(-1, 3)   # inside each top
            origins = pts - [0.0, 0.01, 0.0]
            ends = origins[:, 1] - first_hit(origins, np.tile([0.0, -1.0, 0.0], (len(pts), 1)), tris)
            local = group[0].fp.frame.plan(pts[:, [0, 2]])
            roof = np.full(len(pts), -np.inf)
            for u in (w for w in self.v.units[cid] if w.part == part):
                inside = shapely.contains_xy(u.outline.buffer(0.01), local[:, 0], local[:, 1])
                if inside.any():
                    height = self.v.attic(u)(local) + self.eave(u)
                    roof = np.where(inside, np.maximum(roof, height), roof)
            bad = ends > roof
            with self.subTest(component=cid, asset=asset):
                self.assertFalse(bad.any(), np.round(np.column_stack([pts[bad][:, [0, 2]], (ends - roof)[bad]]), 2).tolist()[:6])

    def test_a_stack_off_the_high_street_carries_one_pot(self):
        """A stack carries a clay pot, or over a High Street building a pot or two."""
        from materials import code
        pot = code("chimney-pot")
        for (cid, part, asset), (group, shell, fittings) in self.shells.items():
            found = self.stacks(shell)
            if not found:
                continue
            centres = np.array([c for c, _, _ in found])
            count = np.zeros(len(found), int)
            for m in fittings:
                if (m.attr[0] == pot).all():
                    count[int(np.argmin(np.linalg.norm(centres - m.pos[:, [0, 2]].mean(axis=0), axis=1)))] += 1
            frame = group[0].fp.frame
            for (c, _, corners), pots in zip(found, count):
                local = frame.plan(corners)
                high = any(u.high and shapely.contains_xy(u.outline.buffer(0.01), local[:, 0], local[:, 1]).any()
                           for u in self.v.units[cid] if u.part == part)
                with self.subTest(component=cid, asset=asset, stack=tuple(np.round(c, 1))):
                    self.assertIn(int(pots), (1, 2) if high else (1,))

    def test_chimney_stacks_stand_on_gables_or_straddle_ridges(self):
        """Every stack rises from a gable wall, free or party, or straddles its building's ridge; none stands on a hip
        or an eave."""
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            for c, _, _ in self.stacks(shell):
                p = Point(group[0].fp.frame.plan(c[None])[0])
                ok = False
                for u in group:
                    angles = self.mv.roof_angles(u)
                    gables = [LineString([a, b]) for k, (a, b) in enumerate(plan.edges(u.outline)) if angles[k] >= 89.999]
                    ok |= any(g.distance(p) < self.mv.STACK["across"] / 2 + 0.1 for g in gables)
                    apex = self.v.ridge(u) - self.eave(u)
                    ok |= bool(self.v.attic(u)(np.array([p.coords[0]]))[0] >= apex - 0.25)
                with self.subTest(component=cid, asset=asset, stack=tuple(np.round(c, 1))):
                    self.assertTrue(ok)

    def test_chimney_stacks_clear_the_roofs_round_them(self):
        """Every stack's top stands at least 0.6 m above any roof within 2.3 m of it, so smoke clears the ridge beside
        it."""
        for (cid, part, asset), (group, shell, _) in self.shells.items():
            roof = np.concatenate([self.faces(shell, r) for r in self.mv.ROOFING])
            rc = roof.mean(axis=1)
            for c, top, _ in self.stacks(shell):
                near = roof[np.hypot(rc[:, 0] - c[0], rc[:, 2] - c[1]) < 2.3]
                with self.subTest(component=cid, asset=asset, stack=tuple(np.round(c, 1))):
                    if len(near):
                        self.assertGreaterEqual(top, float(near[:, :, 1].max()) + 0.6)

    def test_the_baseline_roofer_builds_every_building(self):
        """With the in-repo roofer in place of CGAL, every building still builds, with its skin."""
        from unittest import mock
        with mock.patch.object(roofs, "BACKEND", "inrepo"):
            for cid in sorted(self.scope):
                for u in self.v.units[cid]:
                    with self.subTest(component=cid, part=u.part, unit=u.k):
                        self.assertIsNotNone(self.mv._Building(self.v, u.fp, u, 0.0).skin)

    def test_party_walls_are_gables(self):
        """A building's roof meets its neighbours' in gables over its party walls, unless they enclose it entirely, or
        it is the distillery's kiln, a pyramid that stands up against its still house's gable."""
        for u in self.units():
            if len(u.party) == len(plan.edges(u.outline)) or u.style.family == "kiln":
                continue
            angles = self.mv.roof_angles(u)
            with self.subTest(component=u.cid, part=u.part, unit=u.k):
                self.assertTrue(all(angles[e] >= 89.999 for e in u.party))

    def test_a_building_whose_eaves_cannot_be_pushed_out_keeps_its_skin_above_its_attic(self):
        """If its outline cannot be pushed out for its eaves, a building's skin rides its attic without them, EPS_M
        above it: not lowered by the eaves' rise as if it had them."""
        from unittest import mock
        u = max(self.units(), key=lambda w: w.outline.area)
        with mock.patch.object(self.mv._Building, "_outset", lambda b, poly, angles, dims, extra=0.0: (poly, False)):
            b = self.mv._Building(self.v, u.fp, u, 0.0)
        m = b.skin.to_mesh()
        tris = np.asarray(m.vert_properties, np.float64)[:, :3][np.asarray(m.tri_verts)]
        inner = u.outline.buffer(-0.3)
        x0, y0, x1, y1 = inner.bounds
        pts = np.array([(x, y) for x in np.arange(x0, x1, 1.0) for y in np.arange(y0, y1, 1.0)
                        if inner.contains(Point(x, y))])
        attic = self.mv.attic_top(u.outline, b.angles)(pts) + b.z_eave
        met = 1e4 - first_hit(np.column_stack([pts, np.full(len(pts), 1e4)]), np.tile([0.0, 0.0, -1.0], (len(pts), 1)),
                              tris)
        self.assertTrue(np.all(met >= attic + self.mv.EPS_M - 0.001))

    def test_neighbours_in_a_row_never_share_their_colour_and_height(self):
        """Houses side by side in a row differ in colour or height; a named building's own parts (the kirk's porch
        against its nave) need not."""
        for cid, us in ((c, self.v.units[c]) for c in sorted(self.scope)):
            for a, b in zip(us[:-1], us[1:]):
                if a.part == b.part and b.party and not (a.asset == b.asset and a.asset in self.mv.NAMED):
                    with self.subTest(component=cid, units=(a.k, b.k)):
                        self.assertNotEqual((a.style.wall, a.style.storeys), (b.style.wall, b.style.storeys))

    def test_signs_hang_out_over_the_street(self):
        from materials import code
        for (cid, part, asset), (group, shell, fittings) in self.shells.items():
            for m in fittings:
                if (m.attr[0] == code("sign-board")).all():
                    c = m.pos.mean(axis=0)[[0, 2]]
                    dist = [float((c - mid) @ out) for mid, out, _ in (self.front(u) for u in group)]
                    with self.subTest(component=cid):
                        self.assertTrue(any(0.3 < d < 1.2 for d in dist), dist)

    def test_each_building_keeps_to_its_triangle_budget(self):
        """With its fittings a building averages under 360 triangles; no house takes over 900, and no named building,
        with its own detail (bays, signs, a fence), over 4000."""
        sizes = [(len(shell.idx) + sum(len(m.idx) for m in fittings), len(group), asset in self.mv.NAMED)
                 for (_, _, asset), (group, shell, fittings) in self.shells.items()]
        self.assertLess(float(np.mean([n / units for n, units, _ in sizes])), 360.0)
        self.assertLess(max(n / units for n, units, named in sizes if not named), 900.0)
        self.assertLess(max(n for n, _, named in sizes if named), 4000.0)

    def test_the_village_builds_identically_twice(self):
        """Every level of detail of a footprint's buildings, built again from a fresh Village, comes out byte for byte the
        same, its shell and every piece."""
        cid = min(self.scope)
        again = self.mv.Village(self.scene)
        for name, level in self.mv.LEVELS.items():
            for part, fp in enumerate(self.v.parts[cid]):
                group = [u for u in self.v.units[cid] if u.part == part]
                shell, pieces = self.v._build(fp, group, level)
                shell2, pieces2 = again._build(again.parts[cid][part], [u for u in again.units[cid] if u.part == part],
                                               level)
                with self.subTest(level=name, part=part):
                    self.assertEqual(len(pieces), len(pieces2))
                    for m, m2 in zip([shell] + pieces, [shell2] + pieces2):
                        for x, y in zip(m, m2):
                            self.assertEqual(np.asarray(x).tobytes(), np.asarray(y).tobytes())


LOD_CELL_M = 0.25       # silhouettes are compared on cells this wide: what lod1 keeps spans a few of them


def occupancy(groups, axes, cell=LOD_CELL_M):
    """Each group's meshes (one per level, the same levels for every group) projected onto the plane of two directions:
    the cells of a cell-metre grid fitted round the group whose centres they cover. Per level, every group's grid laid
    end to end as one flat boolean array, each row closed by a spare cell; and where each group's grid starts in them,
    with their end. Each triangle is filled row by row: the span of a row's centre line it covers."""
    basis = np.column_stack(axes)
    uv = [[m.pos @ basis for m in meshes] for meshes in groups]
    lo = np.array([np.min([p.min(axis=0) for p in g], axis=0) for g in uv])
    hi = np.array([np.max([p.max(axis=0) for p in g], axis=0) for g in uv])
    size = np.floor((hi - lo) / cell).astype(np.int64) + 1
    width = size[:, 1] + 1                  # the spare cell: no row's coverage runs on into the next
    start = np.concatenate([[0], np.cumsum(size[:, 0] * width)])
    grids = []
    for level in range(len(groups[0])):
        tris = np.concatenate([(uv[g][level][groups[g][level].idx] - lo[g]) / cell for g in range(len(groups))])
        owner = np.concatenate([np.full(len(groups[g][level].idx), g) for g in range(len(groups))])
        first = np.maximum(np.ceil(tris[:, :, 0].min(axis=1) - 0.5), 0).astype(np.int64)
        last = np.minimum(np.floor(tris[:, :, 0].max(axis=1) - 0.5).astype(np.int64), size[owner, 0] - 1)
        rows = np.maximum(last - first + 1, 0)
        t = np.repeat(np.arange(len(tris)), rows)
        i = first[t] + np.arange(len(t)) - np.repeat(np.cumsum(rows) - rows, rows)
        x = i + 0.5
        y0, y1 = np.full(len(t), np.inf), np.full(len(t), -np.inf)
        for e in range(3):                  # where the row's centre line crosses each edge
            p, q = tris[t, e], tris[t, (e + 1) % 3]
            dx = q[:, 0] - p[:, 0]
            s = np.divide(x - p[:, 0], dx, out=np.full(len(t), -1.0), where=dx != 0)
            on = (s >= 0.0) & (s <= 1.0)
            y = p[:, 1] + np.clip(s, 0.0, 1.0) * (q[:, 1] - p[:, 1])
            y0, y1 = np.where(on, np.minimum(y0, y), y0), np.where(on, np.maximum(y1, y), y1)
        crossed = np.isfinite(y0)
        j0 = np.maximum(np.ceil(np.where(crossed, y0, 1.0) - 0.5), 0).astype(np.int64)
        j1 = np.minimum(np.floor(np.where(crossed, y1, 0.0) - 0.5).astype(np.int64), size[owner[t], 1] - 1)
        keep = crossed & (j0 <= j1)
        row = start[owner[t]] + i * width[owner[t]]
        n = int(start[-1])
        steps = np.bincount((row + j0)[keep], minlength=n) - np.bincount((row + j1 + 1)[keep], minlength=n)
        grids.append(np.cumsum(steps[:n]) > 0)
    return grids, start


def ious(a, b, start):
    """Two occupancies' intersection over union for each group (occupancy), and for all of them together."""
    inter = np.add.reduceat((a & b).astype(np.int64), start[:-1])
    union = np.add.reduceat((a | b).astype(np.int64), start[:-1])
    return inter / np.maximum(union, 1), float(inter.sum() / union.sum())


def codes(mesh):
    return {tuple(int(x) for x in a) for a in np.unique(mesh.attr, axis=0)}


def plan_triangles(meshes):
    """The meshes' triangles seen from above as world (x, z) polygons, but for those seen edge on."""
    tris = np.concatenate([m.pos[m.idx][:, :, [0, 2]] for m in meshes]) if meshes else np.zeros((0, 3, 2))
    a = (tris[:, 1, 0] - tris[:, 0, 0]) * (tris[:, 2, 1] - tris[:, 0, 1]) \
        - (tris[:, 1, 1] - tris[:, 0, 1]) * (tris[:, 2, 0] - tris[:, 0, 0])
    tris = tris[np.abs(a) > 1e-9]
    return shapely.polygons(np.concatenate([tris, tris[:, :1]], axis=1))


def areas(mesh, rows):
    """The areas of a mesh's triangles `rows` (a mask or indices)."""
    tris = mesh.pos[mesh.idx[rows]]
    return 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)


def shared_planes(shell, piece, gap=0.005):
    """[(area, piece triangle)] where a triangle of `piece` lies within `gap` of the plane of a face of `shell` facing the
    same way, over the part of that face it overlaps: there the two fight in the depth buffer."""
    def faces(m):
        tris = m.pos[m.idx]
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        length = np.linalg.norm(n, axis=1)
        keep = length > 1e-12
        return tris[keep], n[keep] / length[keep, None]
    s, ns = faces(shell)
    p, np_ = faces(piece)
    i, j = np.nonzero(np_ @ ns.T > 0.999)
    off = np.einsum("pkd,pd->pk", p[i] - s[j, 0][:, None, :], ns[j])    # each corner's height over the shell face
    near = np.all(np.abs(off) < gap, axis=1)
    out = []
    for a, b in zip(i[near], j[near]):
        u = np.cross(np_[a], [0.0, 1.0, 0.0] if abs(np_[a][1]) < 0.9 else [1.0, 0.0, 0.0])
        basis = np.column_stack([u / np.linalg.norm(u), np.cross(np_[a], u / np.linalg.norm(u))])
        area = Polygon(p[a] @ basis).intersection(Polygon(s[b] @ basis)).area
        if area > 1e-6:
            out.append((area, p[a]))
    return out


class VillageLodTests(unittest.TestCase):
    """The village's coarser levels of detail and its shadow caster, built from the same building records as its full
    level (shared()), each building group's beside its full level's. The payload carries each that passes its guards
    and saves enough triangles, and the page draws one per view."""
    X, Y, Z = np.eye(3)
    COVER_M = 0.75          # seen from above, a part this near the shadow caster casts its shadow with it...
    COVER_AREA_M2 = 1.0     # ...and no patch farther than this

    @classmethod
    def setUpClass(cls):
        from mesh_geometry import concat
        found = shared()
        cls.mv, cls.scene, cls.v, cls.shells = found["mv"], found["scene"], found["v"], found["shells"]
        cls.keys = sorted(cls.shells)
        if "levels" not in found:
            built = {"lod0": {k: [shell] + fittings for k, (_, shell, fittings) in cls.shells.items()}}
            for name in ("lod1", "lod2", "shadow"):
                built[name] = {}
                for k in cls.keys:
                    shell, pieces = cls.v._build(cls.v.parts[k[0]][k[1]], cls.shells[k][0], cls.mv.LEVELS[name])
                    built[name][k] = [shell] + pieces
            found["levels"] = built
        cls.levels = found["levels"]
        cls.whole = {name: [concat(cls.levels[name][k]) for k in cls.keys] for name in cls.levels}
        sun = np.array([-0.55, 0.5, -0.67])               # a low afternoon sun from the south-west, as the Willow's
        sun /= np.linalg.norm(sun)
        u = np.cross(sun, cls.Y)
        u /= np.linalg.norm(u)
        cls.views = {"east": (cls.Z, cls.Y), "north": (cls.X, cls.Y), "above": (cls.X, cls.Z),
                     "sun": (u, np.cross(u, sun))}

    def seen(self, view, names):
        """{level: occupancy} of every building group seen from `view`, and the groups' starts (occupancy)."""
        grids, start = occupancy([[self.whole[n][g] for n in names] for g in range(len(self.keys))], self.views[view])
        return dict(zip(names, grids)), start

    def test_each_level_costs_a_fraction_of_the_full_village(self):
        """With the small structures it keeps, lod1 costs at most 0.42 of the full village's triangles, lod2 0.2 and the
        shadow caster 0.13."""
        tris = {name: sum(len(m.idx) for m in self.whole[name])
                + sum(len(m.idx) for ms in self.v.small(self.mv.LEVELS[name]).values() for m in ms)
                for name in self.mv.LEVELS}
        self.assertLessEqual(tris["lod1"], 0.42 * tris["lod0"])
        self.assertLessEqual(tris["lod2"], 0.2 * tris["lod0"])
        self.assertLessEqual(tris["shadow"], 0.13 * tris["lod0"])

    def test_coarser_levels_keep_each_buildings_silhouette(self):
        """Seen from the east, the north and above, each building group's lod1 and lod2 cover at least 0.8 of what they
        and its full level cover between them (their intersection over union), and the whole village's at least
        0.95."""
        for view in ("east", "north", "above"):
            grids, start = self.seen(view, ("lod0", "lod1", "lod2"))
            for level in ("lod1", "lod2"):
                each, pooled = ious(grids["lod0"], grids[level], start)
                worst = int(np.argmin(each))
                with self.subTest(view=view, level=level):
                    self.assertGreaterEqual(pooled, 0.95)
                    self.assertGreaterEqual(each[worst], 0.8, self.keys[worst])

    def test_the_shadow_caster_casts_each_buildings_shadow(self):
        """Seen along the sun, each building group's shadow caster covers at least 0.8 of what it and its full level
        cover between them, and the whole village's at least 0.94."""
        grids, start = self.seen("sun", ("lod0", "shadow"))
        each, pooled = ious(grids["lod0"], grids["shadow"], start)
        worst = int(np.argmin(each))
        self.assertGreaterEqual(pooled, 0.94)
        self.assertGreaterEqual(each[worst], 0.8, self.keys[worst])

    def test_the_shadow_caster_covers_every_part_seen_from_above(self):
        """Seen from above, the shadow caster leaves no patch of the full village's parts (each building group's shell
        and fittings, and the small structures) larger than COVER_AREA_M2 farther than COVER_M from it: what it leaves
        out casts no sun shadow on the page, and a patch that size is more than a few steps' or lamps' plinths. COVER_M
        lets it leave out the eaves and verges (at most 0.45 m), which the page's shadow texels (about 0.7 m) hardly
        show. A fitting standing out farther than that from its building group's own walls over more than
        COVER_AREA_M2 casts as built: no other piece near it stands in for it."""
        shadow = self.mv.LEVELS["shadow"]
        casters = [shapely.union_all(plan_triangles(self.levels["shadow"][k])) for k in self.keys]
        casters += [shapely.union_all(plan_triangles([m])) for ms in self.v.small(shadow).values() for m in ms]
        casters = [c for c in casters if not c.is_empty]
        tree = shapely.STRtree(casters)
        parts = [self.levels["lod0"][k] for k in self.keys]
        parts += [[m] for ms in self.v.small(self.mv.FULL).values() for m in ms]
        left = []
        for meshes in parts:
            tris = plan_triangles(meshes)
            if not len(tris):
                continue
            near = tree.query(shapely.box(*shapely.total_bounds(tris)), predicate="dwithin", distance=self.COVER_M)
            cover = shapely.union_all([casters[j] for j in near]).buffer(self.COVER_M, quad_segs=4)
            shapely.prepare(cover)
            out = tris[~shapely.covered_by(tris, cover)]
            left += list(shapely.difference(out, cover))
        patches = shapely.get_parts(shapely.union_all(left)) if left else []
        big = sorted(((round(float(g.area), 2), np.round(shapely.get_coordinates(g.centroid)[0], 1).tolist())
                      for g in patches if g.area > self.COVER_AREA_M2), reverse=True)
        self.assertEqual(big, [])
        standing = 0
        for k in self.keys:
            walls = shapely.union_all(plan_triangles(self.levels["shadow"][k][:1])).buffer(self.COVER_M, quad_segs=4)
            shapely.prepare(walls)
            cast = {m.pos.tobytes() for m in self.levels["shadow"][k][1:]}
            for m in self.levels["lod0"][k][1:]:
                tris = plan_triangles([m])
                out = tris[~shapely.covered_by(tris, walls)]
                if shapely.area(out).sum() <= self.COVER_AREA_M2 \
                        or shapely.union_all(out).difference(walls).area <= self.COVER_AREA_M2:
                    continue
                standing += 1
                with self.subTest(group=k, piece=np.round(m.pos.mean(axis=0), 1).tolist()):
                    self.assertIn(m.pos.tobytes(), cast)
        self.assertGreater(standing, 0)

    def test_every_level_is_wound_with_unit_normals_in_its_buildings_own_codes(self):
        """Every piece of every coarser level winds with its normals, each a unit vector, in (material, aux) codes its
        building group's full level uses; no triangle of a coarser level's shell is a sliver (SLIVER_M)."""
        for name in ("lod1", "lod2", "shadow"):
            for k in self.keys:
                full = set().union(*(codes(m) for m in self.levels["lod0"][k]))
                shell = self.levels[name][k][0]
                tris = shell.pos[shell.idx]
                longest = np.max([np.linalg.norm(tris[:, (i + 1) % 3] - tris[:, i], axis=1) for i in range(3)], axis=0)
                area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
                with self.subTest(level=name, group=k):
                    self.assertEqual(int((area <= self.mv.SLIVER_M * longest).sum()), 0)
                    for m in self.levels[name][k]:
                        self.assertEqual(int(world.turned(m.pos, m.idx, m.nrm).sum()), 0)
                        self.assertTrue(np.all(np.abs(np.linalg.norm(m.nrm, axis=1) - 1.0) < 1e-6))
                        self.assertLessEqual(codes(m), full)

    def test_each_level_cuts_its_parts_from_the_buildings_records(self):
        """Each level cuts its parts from the same building records (_Building.at). lod1 joins every wall prism, the base
        course apart, and every feature solid (dormers, bays, buttresses), joins no stack, cuts nothing and keeps every
        featured fitting (a named building's roof features) but no other; lod2 and the shadow caster join no base course,
        feature or stack and cut nothing, lod2 keeps no piece and the shadow caster only the fittings hung to cast
        (_Building.cast)."""
        for (cid, part, asset), (group, _, _) in self.shells.items():
            for b in self.v.buildings(self.v.parts[cid][part], group)[1]:
                tagged = lambda *roles: {id(s) for s, _, role in b.solids if role in roles}
                for name in ("lod1", "lod2", "shadow"):
                    solids, cutters, pieces = b.at(self.mv.LEVELS[name])
                    joined = {id(s) for s, _ in solids}
                    with self.subTest(level=name, component=cid, part=part, unit=b.u.k):
                        self.assertEqual(cutters, [])
                        if name == "lod1":
                            self.assertLessEqual(tagged("walls", "feature"), joined)
                            self.assertFalse(tagged("stack") & joined)
                            self.assertEqual({id(m) for m in pieces} & {id(m) for m in b.fittings},
                                             {id(m) for m in b.featured})
                        else:
                            self.assertFalse(tagged("walls", "feature", "stack") & joined)
                            self.assertEqual([id(m) for m in pieces],
                                             [id(m) for m in b.casting] if name == "shadow" else [])

    def test_lod1_keeps_the_fittings_crowning_its_roofs_as_built(self):
        """lod1 keeps, byte for byte, every fitting of the full level that rises over 0.3 m above the ridge of the
        building under it (within half a metre), but for those in the fine fittings' codes (chimney pots, a turret's
        bell, the owl loft's flight holes): the turrets' stones and caps, the ventilator and its cap, the owl loft with
        its cap and ledges, the crow steps' top stones."""
        from materials import code
        fine = {code(n) for n in ("chimney-pot", "black-iron") + self.mv.MARGINS}
        crowning = 0
        for k in self.keys:
            group = self.shells[k][0]
            kept = {m.pos.tobytes() for m in self.levels["lod1"][k][1:]}
            for m in self.levels["lod0"][k][1:]:
                c = m.pos.mean(axis=0)
                under = [self.v.ridge(u) for u in group
                         if u.fp.frame.to_world(u.outline).buffer(0.5).contains(Point(c[0], c[2]))]
                if not under or m.pos[:, 1].max() <= max(under) + 0.3 or codes(m) & fine:
                    continue
                crowning += 1
                with self.subTest(group=k, piece=np.round(c, 1).tolist()):
                    self.assertIn(m.pos.tobytes(), kept)
        self.assertGreater(crowning, 0)

    def test_no_piece_lies_in_the_plane_of_a_face_of_its_building(self):
        """No piece standing apart from a building group's shell, at the full level or lod1 (its fittings, flat
        openings, stack boxes and roof features), lies within 5 mm of the plane of a face of the shell that faces the
        same way, over that face; nor, at lod1, of a face of another of its pieces in other materials: there the two would
        fight in the depth buffer."""
        from mesh_geometry import concat
        for name in ("lod0", "lod1"):
            for k in self.keys:
                shell, *pieces = self.levels[name][k]
                found = shared_planes(shell, concat(pieces)) if pieces else []
                if name == "lod1":
                    kinds = [codes(m) for m in pieces]
                    for m, kind in zip(pieces, kinds):
                        others = [m2 for m2, kind2 in zip(pieces, kinds) if kind2 != kind]
                        found += shared_planes(concat(others), m) if others else []
                with self.subTest(level=name, group=k):
                    self.assertFalse(found, [(round(a, 3), np.round(t.mean(axis=0), 1).tolist()) for a, t in found[:4]])

    def test_coarser_roofs_stop_at_the_buildings_round_them(self):
        """A coarser level's roof, the skin's outer solid (lod1's notched for its dormers and trimmed back to its gables,
        lod2's whole), never reaches over another building's footprint either: it stops where the skin does."""
        projection = VillageBuildTests.projection
        for (cid, part, asset), (group, _, _) in self.shells.items():
            near = [w for c, w in self.v.outlines.items() if c != cid and w.distance(self.v.outlines[cid]) < 3.0]
            for b in self.v.buildings(self.v.parts[cid][part], group)[1]:
                if b.plain_skin is None:
                    continue
                others = near + [w.fp.frame.to_world(w.outline) for w in self.v.units[cid] if w is not b.u]
                for name, hull in (("lod1", b._hull(True, True)), ("lod2", b._hull(False, False))):
                    cover = b.u.fp.frame.to_world(projection(hull))
                    with self.subTest(level=name, component=cid, part=part, unit=b.u.k):
                        self.assertLess(sum(cover.intersection(o.buffer(-0.05)).area for o in others), 0.05)

    def test_every_coarser_shell_is_closed(self):
        """Each coarser level's shell is closed, as its full level's is (VillageBuildTests.test_every_shell_is_closed):
        its flat openings and stack boxes stand apart from it."""
        for name in ("lod1", "lod2", "shadow"):
            for k in self.keys:
                with self.subTest(level=name, group=k):
                    self.assertEqual(len(cracks(self.levels[name][k][0])), 0)

    def test_lod1_lays_every_window_and_door_flat_on_its_wall(self):
        """lod1 shows as much of each glass and door paint as its full level does (there a cut's back face), on flat
        panels standing DEPTH_GAP_M proud of the walls they open in (give or take the WELD_M the tidy may move a wall's
        corner), nothing of their building in front of them, seen from their middles or near any of their corners."""
        from materials import code
        from mesh_geometry import concat
        shown = [code(n) for n in self.mv.GLASS + self.mv.DOORS]
        corners = np.array([[1, 1, 1], [8, 1, 1], [1, 8, 1], [1, 1, 8]], np.float64)
        corners /= corners.sum(axis=1, keepdims=True)
        for k in self.keys:
            full, flat = concat(self.levels["lod0"][k]), concat(self.levels["lod1"][k])
            area = lambda m, c: float(areas(m, (m.attr[m.idx[:, 0]] == c).all(axis=1)).sum())
            for c in shown:
                with self.subTest(group=k, code=c):
                    self.assertAlmostEqual(area(flat, c), area(full, c), delta=0.01 * area(full, c) + 0.01)
            sel = np.zeros(len(flat.idx), bool)
            for c in shown:
                sel |= (flat.attr[flat.idx[:, 0]] == c).all(axis=1)
            tris = flat.pos[flat.idx[sel]]
            n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            n = np.repeat(n / np.linalg.norm(n, axis=1, keepdims=True), len(corners), axis=0)
            origins = np.einsum("sk,tkd->tsd", corners, tris).reshape(-1, 3)
            shell = self.levels["lod1"][k][0]
            behind = first_hit(origins, -n, shell.pos[shell.idx])
            ahead = first_hit(origins, n, flat.pos[flat.idx], near=0.001)
            with self.subTest(group=k):
                off = np.abs(behind - self.mv.DEPTH_GAP_M) >= self.mv.WELD_M
                self.assertFalse(off.any(), np.round(behind[off], 3).tolist()[:6])
                self.assertTrue(np.all(ahead > 0.5), np.round(ahead[ahead <= 0.5], 3).tolist()[:6])

    def test_lod1_keeps_the_stacks_and_roof_features_but_not_the_fittings(self):
        """lod1 keeps every chimney stack and a named building's roof features: as much of each (material, aux) faces
        the sky over the ridge of the building under it (within half a metre) as in its full level (the stacks' caps,
        a turret's stones, crow steps, the ventilator's and the owl loft's tops), but for the roofs' own (a flat-capped
        roof still shows under a stack standing in it as a box) and for the codes that only the fine fittings, the
        paint round openings and shops' fascias carry, of which it has none. Its stacks stand apart from its shell, as
        boxes integration can drop: no stack's cap is part of the shell."""
        from materials import code
        fine = {code(n) for n in ("chimney-pot", "sign-board", "sign-cloth", "black-iron", "rust-iron", "glass-steamed",
                                  "shop-blind", "clock-face") + self.mv.MARGINS + self.mv.FASCIAS}
        skip = fine | {code(n) for n in self.mv.ROOFING}

        def sky(pieces, group):
            found = {}
            for m in pieces:
                tris = m.pos[m.idx]
                n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
                up = n[:, 1] > 0.99 * np.linalg.norm(n, axis=1)
                mid = tris.mean(axis=1)
                for u in group:
                    under = shapely.contains_xy(u.fp.frame.to_world(u.outline).buffer(0.5), mid[:, 0], mid[:, 2])
                    keep = up & under & (tris[:, :, 1].min(axis=1) > self.v.ridge(u))
                    for c, a in zip(m.attr[m.idx[keep, 0]], areas(m, keep)):
                        found[tuple(int(x) for x in c)] = found.get(tuple(int(x) for x in c), 0.0) + float(a)
            return {c: a for c, a in found.items() if c not in skip}
        for k in self.keys:
            group = self.shells[k][0]
            full, flat = sky(self.levels["lod0"][k], group), sky(self.levels["lod1"][k], group)
            with self.subTest(group=k):
                self.assertEqual(set(flat), set(full))
                for c, a in full.items():
                    self.assertAlmostEqual(flat[c], a, delta=0.01 * a + 1e-3, msg=c)
                self.assertFalse(set().union(*(codes(m) for m in self.levels["lod1"][k])) & fine)
                self.assertNotIn(code("pale-stone"), sky(self.levels["lod1"][k][:1], group))

    def test_lod2_and_the_shadow_caster_are_each_buildings_walls_and_roofs_alone(self):
        """lod2 and the shadow caster are each building group's shell alone: its walls in their own material from foot to
        eaves, with no base course, under their roofs; no openings, stacks, dormers or fittings, but that the shadow
        caster keeps, as built, the fittings standing out beyond the eaves that cast shadows of their own: Gladrags'
        blind and the Shack's fence."""
        from materials import code
        casting = set()
        for k in self.keys:
            group = self.shells[k][0]
            allowed = {code(u.style.wall) for u in group} | {code(u.style.roof_material) for u in group}
            full = {m.pos.tobytes() for m in self.levels["lod0"][k][1:]}
            hung = sum(len(b.casting) for b in self.v.buildings(self.v.parts[k[0]][k[1]], group)[1])
            casting |= {k[2]} if hung else set()
            for name in ("lod2", "shadow"):
                shell, *pieces = self.levels[name][k]
                with self.subTest(level=name, group=k):
                    self.assertLessEqual(codes(shell), allowed)
                    self.assertEqual(len(pieces), hung if name == "shadow" else 0)
                    self.assertTrue(all(m.pos.tobytes() in full for m in pieces))
        self.assertEqual(casting, {"gladrags", "shrieking-shack"})

    def test_lod1_and_the_shadow_caster_keep_the_kirkyard_and_garden_walls_and_lod2_no_small_structures(self):
        """Of the small structures, lod1 keeps the kirkyard's wall and the gardens' walls and gates as built, but not the
        lamps or the headstones; the shadow caster keeps the kirkyard's wall as built and, for each garden's wall and
        gate, one slab along them (garden_walls(whole=True)), for they cast shadows of their own beyond the eaves;
        lod2 keeps none."""
        import mesh_village_small as small
        from materials import code
        full = self.v.small(self.mv.LEVELS["lod0"])
        kirkyard = [m for m in full["kirk"] if (m.attr == code("rubble")).all()]
        stand_ins = small.garden_walls(self.v, whole=True)
        for name, expected in (("lod1", {"street-furniture": [], "kirk": kirkyard, self.mv.HOUSE: full[self.mv.HOUSE]}),
                               ("shadow", {"street-furniture": [], "kirk": kirkyard, self.mv.HOUSE: stand_ins})):
            found = self.v.small(self.mv.LEVELS[name])
            self.assertEqual(set(found), set(expected))
            for asset, pieces in expected.items():
                with self.subTest(level=name, asset=asset):
                    self.assertEqual(len(found[asset]), len(pieces))
                    for m, m2 in zip(found[asset], pieces):
                        self.assertEqual(m.pos.tobytes(), m2.pos.tobytes())
                        self.assertEqual(m.attr.tobytes(), m2.attr.tobytes())
        self.assertTrue(kirkyard and len(kirkyard) < len(full["kirk"]))
        self.assertEqual(len(stand_ins), len(full[self.mv.HOUSE]))
        self.assertTrue(all(len(m.idx) == 10 for m in stand_ins))
        self.assertFalse([m for ms in self.v.small(self.mv.LEVELS["lod2"]).values() for m in ms])

    def test_levels_gather_every_built_footprint_and_the_small_structures_at_each_level(self):
        """mesh_village.levels(village) holds, for each level, every built footprint's pieces at that level and the
        small structures it keeps, by asset."""
        from unittest import mock
        component = lambda v, cid, level: {"house": [("built", cid, level)]}
        small = lambda v, level: {"house": [("small", level)], "lamps": []}
        with mock.patch.object(self.mv.Village, "component", component), \
                mock.patch.object(self.mv.Village, "small", small):
            found = self.mv.levels(self.v)
        self.assertEqual(set(found), set(self.mv.LEVELS))
        for name, level in self.mv.LEVELS.items():
            with self.subTest(level=name):
                self.assertEqual(found[name], {"house": [("built", cid, level) for cid in sorted(self.v.built)]
                                               + [("small", level)], "lamps": []})


class VillageAtticTests(unittest.TestCase):
    """Where a storey-and-a-half building's attic windows go, on plans the High Street may not have."""

    @staticmethod
    def unit(outline, front, roof="thatch", pitch=50.0, party=frozenset(), margins=None, hips=False, high=False):
        import mesh_village as mv
        st = mv.Style(family="cottage-dormer", storeys=1.5, eaves=3.1, wall="rubble", plinth="rubble", roof=roof,
                      roof_material="village-thatch" if roof == "thatch" else "village-slate", pitch=pitch, hips=hips,
                      shopfront=None, margins=margins, door="door-green", sign=False)
        return mv.Unit(cid=0, part=0, k=0, fp=None, outline=orient(outline, 1.0), party=party, front=front, high=high,
                       asset="village-house", style=st)

    def window(self, u):
        import mesh_village as mv
        return mv.attic_window(u, mv.roof_angles(u))

    def test_a_short_hipped_attic_front_becomes_a_gable_with_its_window(self):
        """A hipped storey-and-a-half cottage's 5 m end, too short for a dormer between its hips, fronting the street
        is a gable instead, with its attic window in it."""
        import mesh_village as mv
        u = self.unit(box(0.0, 0.0, 5.0, 8.0), front=3, hips=True)                       # its south end
        self.assertGreaterEqual(mv.roof_angles(u)[3], 89.999)
        self.assertEqual(self.window(u)[0], "gable")

    def test_a_short_front_beside_an_inside_corner_keeps_its_eaves(self):
        """An L-shaped cottage's 3 m wall in the angle of its L, too short for a dormer, is no gable: one there would
        stand beside the inside corner, and the roof beyond it would have to fold back over itself."""
        import mesh_village as mv
        ell = Polygon([(-4.92, 0.52), (-4.92, -1.39), (-2.53, -1.39), (-2.53, -4.44), (3.04, -4.44), (3.04, 0.52)])
        self.assertLess(mv.roof_angles(self.unit(ell, front=2))[2], 89.999)

    def test_a_short_front_on_a_skewed_plan_keeps_its_eaves(self):
        """Only a square-cornered plan turns a short attic front into a gable: on a skewed one the slopes beside the
        gable run askew, and the window in it would meet them."""
        import mesh_village as mv
        u = self.unit(Polygon([(0.0, 0.0), (5.0, 0.0), (5.6, 8.0), (0.0, 8.0)]), front=0, hips=True)
        self.assertLess(mv.roof_angles(u)[0], 89.999)

    def test_dormers_stand_only_before_roof_deep_enough_to_bury_them(self):
        """Along an L-shaped cottage's 16 m front, no dormer stands before its 3 m deep wing."""
        how = self.window(self.unit(Polygon([(0, 0), (16.27, 0), (16.27, 2.95), (4.27, 2.95), (4.27, 5.6), (0, 5.6)]),
                                    front=0))
        if how is not None:
            kind, picks, w = how
            self.assertEqual(kind, "dormers")
            self.assertTrue(all(s + w / 2 <= 4.27 for s in picks), picks)

    def test_a_courtyard_house_roofs_its_courtyard_side_too(self):
        """A unit round a courtyard has a pitch for each courtyard wall as well, and its roof builds."""
        import mesh_village as mv
        u = self.unit(Polygon([(0, 0), (12, 0), (12, 12), (0, 12)], [[(4, 4), (4, 8), (8, 8), (8, 4)]]), front=0)
        angles = mv.roof_angles(u)
        self.assertEqual(len(angles), 8)
        v, f = roofs.roof(u.outline, angles)
        self.assertGreater(len(f), 0)

    def test_a_plain_cottage_takes_two_dormers_and_a_shallow_one_none(self):
        """A 10 m cottage front on the High Street takes two dormers; elsewhere one, over its middle; a 3 m deep
        cottage none."""
        kind, picks, w = self.window(self.unit(box(0.0, 0.0, 10.0, 6.0), front=3, high=True))   # its south side
        self.assertEqual((kind, len(picks)), ("dormers", 2))
        kind, picks, w = self.window(self.unit(box(0.0, 0.0, 10.0, 6.0), front=3))
        self.assertEqual((kind, picks), ("dormers", [5.0]))
        self.assertIsNone(self.window(self.unit(box(0.0, 0.0, 10.0, 3.0), front=3)))

    def test_a_gable_window_keeps_under_its_verge(self):
        """A 4 m gable front's attic window, with its painted margin, keeps under the underside of the verge above it."""
        import mesh_village as mv
        u = self.unit(box(0.0, 0.0, 4.0, 9.0), front=3, roof="slate", pitch=40.0, margins="margin-black")
        kind, s, z0, w, h = self.window(u)
        self.assertEqual(kind, "gable")
        drop = mv.SLATE["thick"] / math.cos(math.radians(40.0))
        corner = (2.0 - w / 2 - mv.MARGIN_M) * math.tan(math.radians(40.0))      # the gable's slope over the margin
        self.assertLessEqual(z0 + h + mv.MARGIN_M, corner - drop + mv.EPS_M - mv.SOFFIT_GAP_M + 1e-9)


class VillageTidyTests(unittest.TestCase):
    """_tidy and _shaded: slivers joined and split away without opening a crack; triangles with no area left out."""

    @staticmethod
    def tetra(base, apex):
        pts = np.vstack([base, [apex]])
        centre = pts.mean(axis=0)
        faces = []
        for f in ((0, 1, 2), (0, 1, 3), (1, 2, 3), (2, 0, 3)):
            p = pts[list(f)]
            out = np.cross(p[1] - p[0], p[2] - p[0]) @ (p.mean(axis=0) - centre) > 0
            faces.append(f if out else (f[0], f[2], f[1]))
        return pts, np.array(faces)

    def test_triangles_joined_onto_the_same_corners_go_only_in_pairs_wound_opposite_ways(self):
        """Three solids on one face, their corners a millimetre or two apart: once the corners are joined, two of the
        three faces, wound opposite ways, go and the third stays, so no edge is left odd."""
        import mesh_village as mv
        base = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        parts = [self.tetra(base, (1.0, 1.0, -2.0)), self.tetra(base + [0.0, 0.0, 0.001], (1.0, 1.0, 2.0)),
                 self.tetra(base + [0.0, 0.0, 0.002], (0.8, 0.8, 3.0))]
        v = np.vstack([p for p, _ in parts])
        f = np.vstack([faces + 4 * i for i, (_, faces) in enumerate(parts)])
        _, f, _ = mv._tidy(v, f, np.zeros(len(f), np.int64))
        _, counts = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0,
                              return_counts=True)
        self.assertEqual(len(f), 10)
        self.assertFalse((counts % 2).any())

    def test_a_sliver_splitting_cannot_remove_is_laid_flat_on_its_longest_side(self):
        """A sliver with no triangle across its longest side, its apex 0.5 mm off that side: the apex moves onto the side,
        so the sliver covers nothing and _shaded leaves it out. The corners handed in stay as they were."""
        import mesh_village as mv
        v = np.array([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.068, 0.0005, 0.0)])
        out, f, _ = mv._tidy(v, np.array([(0, 1, 2)]), np.zeros(1, np.int64))
        self.assertEqual(f.tolist(), [[0, 1, 2]])
        np.testing.assert_allclose(out[2], (0.068, 0.0, 0.0), atol=1e-12)
        self.assertEqual(v[2].tolist(), [0.068, 0.0005, 0.0])

    def test_a_slivers_apex_stays_put_where_laying_it_flat_would_turn_another_triangle_over(self):
        """Two slivers share an apex, 0.6 mm over the first one's longest side and 1 mm over the second one's, which
        lies beyond the first: the apex moves onto the first one's side, and no further, which would turn it over. The
        pair lies at an odd angle, so the first, laid flat, has a normal of rounding noise, not of exact zeros."""
        import mesh_village as mv
        turn = math.radians(23.0)
        turn = np.array([[math.cos(turn), -math.sin(turn), 0.0], [math.sin(turn), math.cos(turn), 0.0], [0.0, 0.0, 1.0]])
        v = np.array([(1.0, 0.0004, 0.0), (3.0, 0.0004, 0.0), (2.0, 0.001, 0.0), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0)])
        v = v @ turn.T
        out, _, _ = mv._tidy(v, np.array([(0, 1, 2), (3, 4, 2)]), np.zeros(2, np.int64))
        side = out[1] - out[0]
        self.assertLess(np.linalg.norm(np.cross(out[2] - out[0], side)) / np.linalg.norm(side), 1e-9)

    def test_a_sliver_left_by_laying_another_flat_is_laid_flat_in_turn(self):
        """Laying one sliver flat moves its apex, a corner of a second triangle, which that thins into a sliver: the tidy
        goes round again and lays the second flat as well."""
        import mesh_village as mv
        v = np.array([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (2.0, 0.003, 0.0), (12.0, 0.003, 0.0), (7.0, -0.002, 0.0)])
        out, _, _ = mv._tidy(v, np.array([(0, 1, 2), (2, 3, 4)]), np.zeros(2, np.int64))
        np.testing.assert_allclose(out[2], (2.0, 0.0, 0.0), atol=1e-12)
        side = out[3] - out[2]
        self.assertLess(np.linalg.norm(np.cross(out[4] - out[2], side)) / np.linalg.norm(side), 1e-9)

    def test_a_slivers_apex_stays_put_beside_a_triangle_with_no_area(self):
        """A triangle at the apex with no area (its corners in line, at an odd angle, so its normal is rounding noise)
        faces no known way, and moving the apex would give it area either way round: the apex stays put."""
        import mesh_village as mv
        turn = math.radians(9.0)
        turn = np.array([[math.cos(turn), -math.sin(turn), 0.0], [math.sin(turn), math.cos(turn), 0.0], [0.0, 0.0, 1.0]])
        v = np.array([(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (2.0, 0.001, 0.0), (3.0, 1.001, 0.0), (4.0, 2.001, 0.0)]) @ turn.T
        out, _, _ = mv._tidy(v, np.array([(0, 1, 2), (2, 3, 4)]), np.zeros(2, np.int64))
        self.assertEqual(out[2].tolist(), v[2].tolist())

    def test_the_apex_of_two_slivers_whose_longest_sides_meet_joins_the_corner_where_they_meet(self):
        """Two slivers share an apex in the narrow wedge their longest sides make, 2.5 degrees apart, as on a terrace's
        party gable at site X: laid on either side, the apex lifts off the other, and it would only creep towards their
        corner, a hair a pass. It joins that corner, the one place on both sides: both slivers go, the two triangles
        beside them close over where they were, and every corner left stays where it was."""
        import mesh_village as mv
        wedge, turn = math.radians(2.5), math.radians(17.0)
        turn = np.array([[math.cos(turn), -math.sin(turn), 0.0], [math.sin(turn), math.cos(turn), 0.0], [0.0, 0.0, 1.0]])
        corner, far, beyond = (0.0, 0.0, 0.0), (4.76, 0.0, 0.0), (4.7, 0.02, 0.0)
        near = (0.15 * math.cos(wedge), 0.15 * math.sin(wedge), 0.0)
        v = np.array([corner, far, beyond, near, (0.077, 0.0012, 0.0)]) @ turn.T
        out, f, _ = mv._tidy(v, np.array([(4, 1, 2), (4, 2, 3), (4, 3, 0), (4, 0, 1)]), np.zeros(4, np.int64))
        self.assertEqual(f.tolist(), [[0, 1, 2], [0, 2, 3]])
        np.testing.assert_array_equal(out[:4], v[:4])

    def test_a_triangle_with_no_area_is_left_out(self):
        """A triangle whose corners lie on one line covers nothing and has no normal to give them: _shaded drops it
        and keeps the rest."""
        import mesh_village as mv
        import mesh_village_plan as plan
        v = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
        f = np.array([[0, 1, 2], [0, 3, 1]])
        with np.errstate(invalid="ignore", divide="ignore"):
            mesh = mv._shaded(v, f, np.zeros(len(f), np.int64), {0: mv._material("rubble")}, plan.Frame(0.0, 0.0, 0.0),
                              0.0)
        self.assertEqual(len(mesh.idx), 1)
        self.assertTrue(np.all(np.isfinite(mesh.nrm)))


class VillageSmallTests(unittest.TestCase):
    """The small structures' builders on ground of the test's own making."""

    def test_a_garden_wall_steps_down_a_slope(self):
        """On a slope of 1 in 5 a garden wall steps down in rubble lengths of at most GARDEN['step'], each GARDEN['high']
        over the ground at its upper end, so nowhere higher over the ground than that and one length's fall. The High
        Street's gardens at site X stand on ground too level to show it."""
        import mesh_village_small as small
        from materials import code
        t = np.array([math.cos(math.radians(23.0)), math.sin(math.radians(23.0))])

        class Slope:
            @staticmethod
            def ground(x, z):
                return 0.2 * (np.asarray(x) * t[0] + np.asarray(z) * t[1])

        p0 = np.array([5.0, -2.0])
        m = small._garden(Slope, p0, p0 + 3.3 * t, p0 + 4.7 * t, p0 + 8.0 * t, t)
        wall = m.pos[(m.attr == code("rubble")).all(axis=1)]
        above = wall[:, 1] - Slope.ground(wall[:, 0], wall[:, 2])
        top = above[above > 0.0]
        self.assertAlmostEqual(float(top.min()), small.GARDEN["high"], delta=1e-9)
        self.assertLessEqual(float(top.max()), small.GARDEN["high"] + 0.2 * small.GARDEN["step"] + 1e-9)

    def test_a_gardens_stand_in_is_one_slab_round_its_wall(self):
        """The shadow caster's stand-in for a garden's wall and gate (whole) is one rubble slab along them, on a slope of
        1 in 5: from no higher than the lowest foot of the wall's lengths to no lower than their highest top, and seen
        from above over all the wall but where its piers stand wider than it."""
        import mesh_village_small as small
        from materials import code
        t = np.array([math.cos(math.radians(23.0)), math.sin(math.radians(23.0))])

        class Slope:
            @staticmethod
            def ground(x, z):
                return 0.2 * (np.asarray(x) * t[0] + np.asarray(z) * t[1])

        p0 = np.array([5.0, -2.0])
        args = (Slope, p0, p0 + 3.3 * t, p0 + 4.7 * t, p0 + 8.0 * t, t)
        wall, slab = small._garden(*args), small._garden(*args, whole=True)
        lengths = wall.pos[(wall.attr == code("rubble")).all(axis=1)][:, 1]
        self.assertEqual((len(slab.idx), codes(slab)), (10, {code("rubble")}))
        self.assertLessEqual(float(slab.pos[:, 1].min()), float(lengths.min()) + 1e-9)
        self.assertGreaterEqual(float(slab.pos[:, 1].max()), float(lengths.max()) - 1e-9)
        seen = lambda m: shapely.union_all(plan_triangles([m]))
        wider = 2 * (small.GARDEN["pier"] - small.GARDEN["thick"]) * small.GARDEN["pier"]     # the two piers' sides
        self.assertLess(seen(wall).difference(seen(slab)).area, wider + 1e-6)


class VillageStyleTests(unittest.TestCase):
    """choose_style: how a unit's looks vary along a row."""

    def test_a_unit_never_shares_its_colour_and_height_with_the_one_before_it(self):
        """However the dice fall; an outshot, always one storey of rubble, behind another is harled instead."""
        import mesh_village as mv
        for family in ("shop", "terrace", "cottage-dormer", "cottage", "villa", "outshot"):
            for seed in range(40):
                rng = np.random.default_rng(seed)
                previous = mv.choose_style(rng, family)
                style = mv.choose_style(rng, family, previous=previous)
                with self.subTest(family=family, seed=seed):
                    self.assertNotEqual((style.wall, style.storeys), (previous.wall, previous.storeys))


class VillageOpeningTests(unittest.TestCase):
    """Where the village leaves room for a window or a door."""

    def test_a_front_from_4_m_has_room_for_a_window_beside_its_door(self):
        """A 4.3 m front has two bays, a door's and a window's; a 3.5 m front a door's alone."""
        import mesh_village as mv
        self.assertEqual(len(mv.bays(4.3)[0]), 2)
        self.assertEqual(len(mv.bays(3.5)[0]), 1)

    def test_an_opening_needs_the_way_ahead_open_across_its_whole_width(self):
        """With a building 2 m in front of the right-hand part of a house's front, a window there is blocked, even
        one whose middle looks past the building's corner, and a window further left is not."""
        import types
        import mesh_village as mv
        wall = box(2.0, -4.0, 6.0, -2.0)
        village = types.SimpleNamespace(_standing=[wall], _standing_index=shapely.STRtree([wall]))
        village.open_ahead = lambda xz, out, clear=mv.CLEAR_M: mv.Village.open_ahead(village, xz, out, clear)
        house = orient(box(-4.0, 0.0, 4.0, 6.0), 1.0)
        is_open = mv.Village.opener(village, types.SimpleNamespace(world=lambda p: np.asarray(p, np.float64)), house)
        front = 3                                                     # (-4, 0) to (4, 0), facing south
        self.assertFalse(is_open(front, 5.5, 1.0))                   # its right-hand side looks at the building
        self.assertTrue(is_open(front, 2.0, 1.0))
        self.assertTrue(is_open(front, 5.5, 1.0, clear=1.5))         # a door needs less room


class VillageStreetTests(unittest.TestCase):
    """A building fronts the street beside it, and only High Street fronts count as the High Street."""

    def test_the_front_faces_the_road_and_the_high_street_is_recognised(self):
        house = orient(box(0.0, 0.0, 8.0, 6.0), 1.0)
        road = [(x, z) for x in np.arange(-20.0, 30.0, 2.0) for z in (-3.0, -5.0)]          # a road south of it
        streets = plan.Streets(road, [(-20.0, -4.0), (30.0, -4.0)])
        k, high = plan.front_edge(house, set(), streets)
        a, b = plan.edges(house)[k]
        np.testing.assert_allclose(plan.outward(a, b), [0.0, -1.0], atol=1e-9)
        self.assertTrue(high)
        elsewhere = plan.Streets(road, [(-20.0, -60.0), (30.0, -60.0)])
        self.assertFalse(plan.front_edge(house, set(), elsewhere)[1])

    def test_a_party_wall_is_never_the_front(self):
        house = orient(box(0.0, 0.0, 8.0, 6.0), 1.0)
        road = [(x, z) for x in np.arange(-20.0, 30.0, 2.0) for z in (-3.0, -5.0)]
        k, _ = plan.front_edge(house, set(), plan.Streets(road, [(0.0, 50.0), (1.0, 50.0)]))
        k2, _ = plan.front_edge(house, {k}, plan.Streets(road, [(0.0, 50.0), (1.0, 50.0)]))
        self.assertNotEqual(k, k2)

    def test_a_short_step_never_outranks_a_long_street_wall(self):
        """A 13 m wall with two thirds of its samples on the road fronts the house, not the 3 m step beyond it with
        all of them on it."""
        house = orient(Polygon([(0, 0), (13, 0), (13, -2), (16, -2), (16, 6), (0, 6)]), 1.0)
        road = [(x, z) for x in np.arange(-20.0, 30.0, 2.0) for z in (-5.0, -7.0)]
        k, high = plan.front_edge(house, set(), plan.Streets(road, [(0.0, 50.0), (1.0, 50.0)]))
        a, b = plan.edges(house)[k]
        self.assertAlmostEqual(float(np.linalg.norm(b - a)), 13.0)
        self.assertFalse(high)

    def test_a_corner_house_fronts_the_high_street(self):
        """A corner house fronts the High Street with its 5 m end rather than the side road with its 10 m flank."""
        house = orient(box(0.0, 0.0, 10.0, 5.0), 1.0)
        road = [(x, z) for x in (13.0, 15.0) for z in np.arange(-20.0, 30.0, 2.0)] + \
               [(x, z) for x in np.arange(-20.0, 12.0, 2.0) for z in (-3.0, -5.0)]
        k, high = plan.front_edge(house, set(), plan.Streets(road, [(14.0, -50.0), (14.0, 50.0)]))
        a, b = plan.edges(house)[k]
        np.testing.assert_allclose(plan.outward(a, b), [1.0, 0.0], atol=1e-9)
        self.assertTrue(high)

    def test_a_front_holds_a_door_and_a_window_where_any_wall_can(self):
        """A 1.5 m return with the only road beside it never fronts a house with longer walls: the wall facing the
        road fronts it."""
        house = orient(Polygon([(0, 0), (8, 0), (8, 4), (6.5, 4), (6.5, 6), (0, 6)]), 1.0)
        road = [(x, z) for x in (6.75, 7.75) for z in (6.0, 7.0, 8.0, 9.0, 10.0)]
        k, _ = plan.front_edge(house, set(), plan.Streets(road, [(0.0, 50.0), (1.0, 50.0)]))
        a, b = plan.edges(house)[k]
        self.assertGreaterEqual(float(np.linalg.norm(b - a)), 2.5)


if __name__ == "__main__":
    unittest.main()
