"""The grounds family: Hagrid's hut and garden, and the Quidditch pitch (whose tests fail until it is built)."""

import math
import os
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, Point, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assets_testing as world  # noqa: E402
import fix_harbour  # noqa: E402
import mesh_geometry  # noqa: E402
import mesh_grounds  # noqa: E402
import mesh_source  # noqa: E402
import railway  # noqa: E402
from materials import code  # noqa: E402
from mesh_source import Grid, decode, load_georef, sample  # noqa: E402
from assets_testing import surface_heights, FT, payload_part  # noqa: E402

# the faults trimesh finds in this family's decoded meshes today (assets_testing.mesh_faults), pinned: a new fault
# fails, and so does a fix until this is updated
MESH_FAULTS = {"winding": {}, "collapsed": {}}


def hut_frame(tris, attr, normal):
    """The built hut's centre (x, z), frontage and depth, unit front (toward the castle at the origin) and right (as
    you face the front), and floor level (the foot of the front door)."""
    walls = tris[np.all(attr == mesh_grounds.WEATHERBOARD, axis=1)].reshape(-1, 3)
    cx, cz, width, depth, axis = mesh_geometry.fit_rectangle(shapely.MultiPoint(walls[:, [0, 2]]).convex_hull)
    centre = np.array([cx, cz])
    front = np.array([-axis[1], axis[0]])
    front = front if front @ -centre > 0 else -front
    right = np.array([-front[1], front[0]])                  # the viewer's right as they face the front (x east, z north)
    u, w = (tris[..., [0, 2]] - centre) @ right, (tris[..., [0, 2]] - centre) @ front
    door = (np.all(attr == mesh_grounds.PLANKS, axis=1) & (np.abs(normal[:, [0, 2]] @ front) > 0.95)
            & (np.abs(u).max(axis=1) < 1.0) & (np.abs(np.abs(w.mean(axis=1)) - depth / 2) < 0.4))
    return centre, width, depth, front, right, float(tris[door][:, :, 1].min())


class HutTests(unittest.TestCase):
    """Hagrid's hut as White Hound measures and draws it (hut-plan, hut-front-gable, timber-hut): a weatherboarded house
    about 35 by 21 ft facing the castle, its front a gable end with the low eaves at the sides, a 5 ft door between two
    sash windows front and back, a stovepipe over a front corner, and rain-barrels right of the front door and back
    left."""

    @classmethod
    def setUpClass(cls):
        cls.tris, cls.attr, cls.normal = payload_part("hut")
        cls.centre, cls.width, cls.depth, cls.front, cls.right, cls.floor = hut_frame(cls.tris, cls.attr, cls.normal)

    def test_the_openings_are_cased_and_the_windows_silled(self):
        # casings proud of the boards at both edges of the door and each window, front and back, not decals on them
        timber = self.of(mesh_grounds.TIMBER)[0]
        c = timber.mean(axis=1)
        u_, w_ = (c[:, [0, 2]] - self.centre) @ self.right, (c[:, [0, 2]] - self.centre) @ self.front
        y = c[:, 1] - self.floor
        proud = np.abs(np.abs(w_) - self.depth / 2 - mesh_grounds.CASING_PROUD_M) < 0.02   # their faces, proud of the boards
        facing = np.abs(self.of(mesh_grounds.TIMBER)[1][:, [0, 2]] @ self.front) > 0.9
        openings = [(0.0, mesh_grounds.DOOR_W / 2)] + [(a, mesh_grounds.WINDOW_W / 2)
                                                       for a in (-mesh_grounds.WINDOW_AT, mesh_grounds.WINDOW_AT)]
        for face in (1.0, -1.0):
            for a, hw in openings:
                for s in (-1.0, 1.0):
                    with self.subTest(face=face, opening=a, side=s):
                        out = s * (u_ - a) - hw                                    # beyond the opening's edge
                        near = proud & facing & (np.sign(w_) == face) & (out > 0.0) & (out < mesh_grounds.CASING_M)
                        self.assertTrue((near & (y > 0.5) & (y < 3.0)).any())
            sill = (np.sign(w_) == face) & (np.abs(np.abs(w_) - self.depth / 2) < 0.12) \
                & (np.abs(y - mesh_grounds.SILL_M) < 0.1)
            self.assertGreater(int(sill.sum()), 4)

    def of(self, attr):
        keep = np.all(self.attr == attr, axis=1)
        return self.tris[keep], self.normal[keep]

    def local(self, pts):
        """(across the front from the middle, out of the front from the middle, above the floor)."""
        xz = pts[..., [0, 2]] - self.centre
        return xz @ self.right, xz @ self.front, pts[..., 1] - self.floor

    def test_a_35_by_21_ft_house_faces_the_castle(self):
        self.assertAlmostEqual(self.width, 35 * FT, delta=0.15)
        self.assertAlmostEqual(self.depth, 21 * FT, delta=0.15)
        towards = -self.centre / np.linalg.norm(self.centre)
        self.assertLess(math.degrees(math.acos(float(self.front @ towards))), 20.0)

    def test_eaves_10_ft_at_the_sides_and_a_15_ft_ridge_running_front_to_back(self):
        walls, n = self.of(mesh_grounds.WEATHERBOARD)
        _, _, y = self.local(walls)
        sides = np.abs(n[:, [0, 2]] @ self.right) > 0.95
        gables = np.abs(n[:, [0, 2]] @ self.front) > 0.95
        self.assertAlmostEqual(float(y[sides].max()), 10 * FT, delta=0.05)
        self.assertAlmostEqual(float(y[gables].max()), 15 * FT, delta=0.05)
        roof, n = self.of(mesh_grounds.SHINGLES)
        u, w, y = self.local(roof[n[:, 1] > 0.2].reshape(-1, 3))
        ridge = y > y.max() - 0.02
        self.assertLess(float(np.abs(u[ridge]).max()), 0.3)                          # along the middle
        self.assertGreater(float(w[ridge].max() - w[ridge].min()), 21 * FT)           # from the front to the back
        self.assertLess(float(y[np.abs(u) > self.width / 2].max()), 10 * FT + 0.25)   # low at the sides

    def test_a_5_ft_door_between_two_4_ft_sash_windows_front_and_back(self):
        planks, n = self.of(mesh_grounds.PLANKS)
        glass, _ = self.of(mesh_grounds.GLASS)
        sashes, _ = self.of(mesh_grounds.SASH)
        for side in (1.0, -1.0):
            with self.subTest(side="front" if side > 0 else "back"):
                u, w, y = self.local(planks)
                door = ((np.abs(n[:, [0, 2]] @ self.front) > 0.95) & (np.abs(u).max(axis=1) < 1.0)
                        & (np.abs(side * w.mean(axis=1) - self.depth / 2) < 0.4))
                self.assertAlmostEqual(float(np.ptp(u[door])), 5 * FT, delta=0.05)
                self.assertAlmostEqual(float(y[door].min()), 0.0, delta=0.01)
                u, w, y = self.local(glass)
                mine = side * w.mean(axis=1) > 0
                for hand in (-1.0, 1.0):
                    pane = mine & (hand * u.mean(axis=1) > 1.0)
                    self.assertAlmostEqual(float(np.ptp(u[pane])), 4 * FT, delta=0.05)
                    self.assertAlmostEqual(float(y[pane].min()), 3 * FT, delta=0.05)
                    self.assertAlmostEqual(float(y[pane].max()), 11 * FT, delta=0.05)
                    su, sw, sy = self.local(sashes)
                    bars = (side * sw.mean(axis=1) > 0) & (np.abs(su.mean(axis=1) - u[pane].mean()) < 4 * FT / 2 + 0.05)
                    across = np.unique(np.round(sy[bars & (np.ptp(sy, axis=1) < 0.1)].mean(axis=1) / FT))
                    self.assertTrue({5.0, 7.0, 9.0} <= set(across))               # square 2 ft panes up the sash

    def test_a_stovepipe_over_a_front_corner_and_barrels_right_of_the_front_door_and_back_left(self):
        iron, _ = self.of(mesh_grounds.IRON)
        roof, _ = self.of(mesh_grounds.SHINGLES)
        u, w, y = self.local(iron.reshape(-1, 3))
        pipe = y > 15 * FT                                                           # whatever stands over the ridge
        self.assertTrue(pipe.any())
        self.assertLess(float(u[pipe].mean()), -self.width / 4)                      # on the left as you face it
        self.assertGreater(float(w[pipe].mean()), 0.0)                               # over the front half
        top = iron.reshape(-1, 3)[pipe][np.argmax(y[pipe])]
        self.assertGreater(float(top[1] - surface_heights(roof, top[0], top[2]).max()), 0.8)
        timber, _ = self.of(mesh_grounds.TIMBER)
        u, w, y = self.local(timber.mean(axis=1))
        outside = (np.abs(u) > self.width / 2 + 0.05) | (np.abs(w) > self.depth / 2 + 0.05)
        for hand, side in ((1.0, 1.0), (-1.0, -1.0)):
            with self.subTest(barrel="front right" if side > 0 else "back left"):
                barrel = outside & (hand * u > self.width / 4) & (side * w > self.depth / 2)
                self.assertTrue(barrel.any())
                self.assertGreater(float(np.ptp(self.local(timber[barrel].reshape(-1, 3))[2])), 0.7)


class HutGardenTests(unittest.TestCase):
    """The garden behind the hut inside White Hound's traced outline (4b): a vaultable fence with a gate and a growing
    tree in it (garden-fence), a dozen boulder-sized pumpkins near the back door, runner beans, a potato bed and a
    compost heap (garden-contents)."""

    @classmethod
    def setUpClass(cls):
        cls.tris, cls.attr, cls.normal = payload_part("hut-garden")
        cls.hut = hut_frame(*payload_part("hut"))
        scene = world.scene()
        traced = next(a for a in scene["assets"] if a["id"] == "hut-garden")["traced"][0]["plan_px"]
        cls.outline = Polygon(railway.grounds_world(np.asarray(traced, np.float64), load_georef()))
        data = world.data()
        cls.grid, cls.bed = Grid(data["meta"]["local"]), decode(data["tex"]["locBed"])
        centre, width, depth, front, right, _ = cls.hut
        corner = lambda s, t: centre + right * s * width / 2 + front * t * depth / 2
        cls.house = Polygon([corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)])
        cls.back_door = centre - front * depth / 2

    def of(self, attr):
        keep = np.all(self.attr == attr, axis=1)
        return self.tris[keep]

    def ground(self, xz):
        return sample(self.bed, self.grid, xz[:, 0], xz[:, 1])

    def posts(self):
        """Centres (x, z) and tops of the fence's posts: thin upright timber."""
        timber = self.of(mesh_grounds.TIMBER)
        groups = {}
        for t in timber:
            key = tuple(np.round(t.mean(axis=0)[[0, 2]] / 0.3).astype(int))
            groups.setdefault(key, []).append(t)
        out = []
        for g in groups.values():
            p = np.concatenate(g)
            if np.ptp(p[:, 1]) > 1.0 and max(np.ptp(p[:, 0]), np.ptp(p[:, 2])) < 0.25:
                out.append((*p[:, [0, 2]].mean(axis=0), p[:, 1].max()))
        return np.array(out)

    def test_the_garden_lies_behind_the_house_inside_white_hounds_fence_line(self):
        centre, _, _, front, _, _ = self.hut
        self.assertLess(float((np.asarray(self.outline.centroid.coords[0]) - centre) @ front), 0.0)
        self.assertGreater(self.outline.area, 40 * FT * 60 * FT)
        posts = self.posts()
        self.assertGreater(len(posts), 20)
        ring = self.outline.exterior
        for x, z, top in posts:
            self.assertLess(ring.distance(Point(x, z)), 0.8)
        heights = posts[:, 2] - self.ground(posts[:, :2])
        self.assertTrue(np.all((heights > 1.0) & (heights < 1.4)), heights)          # low enough to vault

    def test_the_fence_has_a_gate_and_a_growing_tree(self):
        ring = self.outline.exterior
        timber = self.of(mesh_grounds.TIMBER)
        rails = timber[(np.ptp(timber[:, :, [0, 2]], axis=1).max(axis=1) > 0.5) & (np.ptp(timber[:, :, 1], axis=1) < 1.0)]
        spans, L = [], ring.length                    # each rail face as an interval along the outline
        for t in rails:
            near = [ring.project(Point(x, z)) for x, _, z in t if ring.distance(Point(x, z)) < 0.8]
            if len(near) == 3:
                a, b = min(near), max(near)
                spans += [(b, L), (0.0, a)] if b - a > L / 2 else [(a, b)]   # a face across the outline's start
        spans.sort()
        gaps, reach = [], spans[0][1]
        for a, b in spans[1:]:
            if a - reach > 1.0:
                gaps.append((reach, a))
            reach = max(reach, b)
        away = [g for g in gaps if self.house.distance(ring.interpolate((g[0] + g[1]) / 2)) > 1.5]
        self.assertTrue(away, "no gate or opening in the fence away from the house")
        bark = self.of(mesh_grounds.BARK).reshape(-1, 3)
        trunk = bark[[ring.distance(Point(x, z)) < 1.0 for x, _, z in bark]]
        self.assertGreater(len(trunk), 0)                                            # a tree standing in the fence line
        foot = trunk[np.argmin(trunk[:, 1])]
        crown = self.of(mesh_grounds.FOLIAGE).reshape(-1, 3)
        near = np.hypot(crown[:, 0] - foot[0], crown[:, 2] - foot[2]) < 8.0
        self.assertGreater(float(crown[near][:, 1].max() - foot[1]), 6.0)

    def fence_runs(self):
        """The fence's timber in plan, as connected runs (posts, rails and the gate's leaf: thin bars, not the hen-coop)."""
        timber = self.of(mesh_grounds.TIMBER)
        timber = timber[0.5 * np.linalg.norm(np.cross(timber[:, 1] - timber[:, 0], timber[:, 2] - timber[:, 0]), axis=1) < 0.3]
        near = np.array([self.outline.exterior.distance(Point(*t.mean(axis=0)[[0, 2]])) < 3.5 for t in timber])
        plan = shapely.union_all([shapely.MultiPoint(t[:, [0, 2]]).convex_hull.buffer(0.15) for t in timber[near]])
        return list(getattr(plan, "geoms", [plan]))

    def test_the_back_door_opens_onto_the_garden_not_onto_rails(self):
        _, _, _, front, _, _ = self.hut
        step = self.back_door - front * 1.2                                         # just outside the back door
        for run in self.fence_runs():
            self.assertGreater(run.distance(Point(*step)), 1.5)

    def test_no_fence_section_stands_loose(self):
        for run in self.fence_runs():
            e = np.asarray(run.minimum_rotated_rectangle.exterior.coords)
            with self.subTest(at=np.round(run.centroid.coords[0], 1).tolist()):
                self.assertTrue(max(np.hypot(*(e[1] - e[0])), np.hypot(*(e[2] - e[1]))) > 3.0
                                or run.distance(self.house) < 0.5)

    def test_the_fence_line_carries_no_hedge(self):
        # the map's thin line round the garden is the fence White Hound describes; the hedge heuristic had read it as a
        # hedge and planted one beside the fence
        tris, _, _ = payload_part("field-boundaries")
        ring = self.outline.exterior
        self.assertEqual(sum(ring.distance(Point(x, z)) < 1.5 for x, _, z in tris.mean(axis=1)), 0)

    def test_a_dozen_boulder_sized_pumpkins_grow_near_the_back_door(self):
        pumpkins = self.of(mesh_grounds.PUMPKIN)
        centres = pumpkins.mean(axis=1)
        from scipy.spatial import cKDTree
        pairs = np.array(sorted(cKDTree(centres).query_pairs(0.6)))
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(centres),) * 2)
        count, label = connected_components(graph, directed=False)
        self.assertEqual(count, 12)
        far = max(np.hypot(*(np.asarray(p) - self.back_door)) for p in self.outline.exterior.coords)
        for k in range(count):
            p = pumpkins[label == k].reshape(-1, 3)
            with self.subTest(pumpkin=k):
                self.assertTrue(1.2 < max(np.ptp(p[:, 0]), np.ptp(p[:, 2])) < 2.6)   # the size of a large boulder
                self.assertTrue(self.outline.contains(Point(*p[:, [0, 2]].mean(axis=0))))
                self.assertLess(float(np.hypot(*(p[:, [0, 2]].mean(axis=0) - self.back_door))), 0.65 * far)

    def pumpkins(self):
        """Each pumpkin's points, grouped by the nearness of its faces."""
        from scipy.spatial import cKDTree
        pumpkins = self.of(mesh_grounds.PUMPKIN)
        centres = pumpkins.mean(axis=1)
        pairs = np.array(sorted(cKDTree(centres).query_pairs(0.6)))
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(centres),) * 2)
        count, label = connected_components(graph, directed=False)
        return [pumpkins[label == k].reshape(-1, 3) for k in range(count)]

    def test_the_pumpkins_are_ribbed_and_lie_among_their_leaves(self):
        foliage = self.of(mesh_grounds.FOLIAGE)
        n = np.cross(foliage[:, 1] - foliage[:, 0], foliage[:, 2] - foliage[:, 0])
        area = 0.5 * np.linalg.norm(n, axis=1)
        broad = (n[:, 1] > 0.7 * np.maximum(2 * area, 1e-12)) & (area > 0.01)  # leaves' faces, not the runners' bars
        leaves = foliage[broad].reshape(-1, 3)
        low = leaves[leaves[:, 1] - self.ground(leaves[:, [0, 2]]) < 0.6]
        for k, p in enumerate(self.pumpkins()):
            with self.subTest(pumpkin=k):
                c = p[:, [0, 2]].mean(axis=0)
                y0, y1 = p[:, 1].min(), p[:, 1].max()
                belt = p[np.abs(p[:, 1] - (y0 + 0.45 * (y1 - y0))) < 0.08 * (y1 - y0)]
                r = np.hypot(*(belt[:, [0, 2]] - c).T)
                self.assertGreater(float(np.ptp(r) / r.mean()), 0.12)                # deeply ribbed, not a smooth ball
                reach = np.hypot(*(low[:, [0, 2]] - c).T)
                self.assertGreater(int((reach < np.ptp(p[:, 0]) / 2 + 1.2).sum()), 12)   # leaves round its foot

    def test_the_pumpkins_sprawl_rather_than_stand_in_a_grid(self):
        from scipy.spatial import cKDTree
        centres = np.array([p[:, [0, 2]].mean(axis=0) for p in self.pumpkins()])
        d = cKDTree(centres).query(centres, k=2)[0][:, 1]
        self.assertGreater(float(d.std() / d.mean()), 0.12)

    def test_the_pumpkin_patch_has_a_ragged_edge(self):
        earth = self.of(mesh_grounds.EARTH)
        flat = earth[(np.ptp(earth[:, :, 1], axis=1) < 0.15)
                     & (earth[:, :, 1].mean(axis=1) - self.ground(earth.mean(axis=1)[:, [0, 2]]) < 0.15)]
        plan = shapely.union_all([Polygon(t[:, [0, 2]]).buffer(0.01) for t in flat])
        patch = max(getattr(plan, "geoms", [plan]), key=lambda g: g.area)
        self.assertLess(patch.area / patch.convex_hull.area, 0.93)                # lobed, not a cut polygon

    def test_the_potato_ridges_carry_their_plants(self):
        earth = self.of(mesh_grounds.EARTH)
        sloped = earth[np.abs(np.cross(earth[:, 1] - earth[:, 0], earth[:, 2] - earth[:, 0])[:, 1]) <
                       0.95 * np.linalg.norm(np.cross(earth[:, 1] - earth[:, 0], earth[:, 2] - earth[:, 0]), axis=1)]
        ridges = sloped[(sloped[:, :, 1].max(axis=1) - self.ground(sloped.mean(axis=1)[:, [0, 2]])) < 0.45]
        bed = shapely.union_all([shapely.MultiPoint(t[:, [0, 2]]).convex_hull.buffer(0.1) for t in ridges])
        leaves = self.of(mesh_grounds.FOLIAGE).reshape(-1, 3)
        on = shapely.contains_xy(bed, leaves[:, 0], leaves[:, 2])
        self.assertGreater(int(on.sum()), 40)

    def test_runner_beans_a_potato_bed_and_a_compost_heap_fill_the_rest(self):
        earth = self.of(mesh_grounds.EARTH).reshape(-1, 3)
        inside = np.array([self.outline.contains(Point(x, z)) for x, _, z in earth])
        self.assertGreater(inside.mean(), 0.95)
        lift = earth[:, 1] - self.ground(earth[:, [0, 2]])
        self.assertGreater(float(lift.max()), 0.8)                                   # a heap Hagrid can collapse onto
        leaves = self.of(mesh_grounds.FOLIAGE).reshape(-1, 3)
        low = leaves[leaves[:, 1] - self.ground(leaves[:, [0, 2]]) < 3.0]
        rows = low[[self.outline.contains(Point(x, z)) for x, _, z in low]]
        self.assertGreater(len(rows), 0)
        self.assertGreater(float((rows[:, 1] - self.ground(rows[:, [0, 2]])).max()), 1.8)   # beans up their poles


class PitchStairTests(unittest.TestCase):
    """Each stair's handrails end on newel posts, not in mid-air (inspection round 2, pitch-stair)."""

    def test_the_handrails_end_on_newels(self):
        p = mesh_grounds.pitch_layout(world.data()["meta"]["pitch"])
        ground = lambda x, z: np.full(np.shape(x), p.level - 0.3)          # the lawn about the playing level
        mesh = mesh_geometry.concat(mesh_grounds.stair(p, mesh_grounds.STAIRS_DEG[0], ground))
        tris = mesh.pos[mesh.idx]
        timber = tris[np.all(mesh.attr[mesh.idx[:, 0]] == mesh_grounds.TIMBER, axis=1)]
        pts = timber.reshape(-1, 3)
        key = np.round(pts[:, [0, 2]] / 0.2).astype(int)
        posts = []                                                                 # upright posts: (x, z, bottom, top)
        for k in np.unique(key, axis=0):
            q = pts[np.all(key == k, axis=1)]
            if np.ptp(q[:, 1]) > 1.0 and np.ptp(q[:, [0, 2]], axis=0).max() < 0.25:
                posts.append((*q[:, [0, 2]].mean(axis=0), q[:, 1].min(), q[:, 1].max()))
        posts = np.array(posts)
        treads = tris[np.all(mesh.attr[mesh.idx[:, 0]] == mesh_grounds.PLANKS, axis=1)].reshape(-1, 3)
        centre = p.c
        lawn = treads[treads[:, 1] < treads[:, 1].min() + 0.05]                    # the outer flight's foot, on the lawn
        reach = np.hypot(*(treads[:, [0, 2]] - centre).T)
        walk = treads[reach < reach.min() + 0.3]                                   # the inner flight's foot, at the walkway
        for name, foot in (("lawn", lawn), ("walkway", walk)):
            with self.subTest(foot=name):
                y = float(foot[:, 1].max())
                d = np.min(np.hypot(*(posts[:, None, :2] - foot[None, :, [0, 2]]).transpose(2, 0, 1)), axis=1)
                newel = (d < 1.2) & (posts[:, 2] < y + 0.1) & (posts[:, 3] > y + 0.75)
                self.assertGreaterEqual(int(newel.sum()), 2)                      # one each side, rail high


class PitchTests(unittest.TestCase):
    """The Quidditch pitch as White Hound plans it (pitch-layout-wh, stands-full-ring, pitch-goals-equal-height): three
    staggered, hollow timber tiers of single rows behind a barrier low enough to vault and a walkway Hagrid can walk round,
    four 15 ft stairs cutting through them, a 10 ft entrance on the castle side, the changing-room block interrupting the
    stands, a freestanding podium midway along the castle side, and six golden hoops 50 ft high."""

    @classmethod
    def setUpClass(cls):
        cls.tris, cls.attr, cls.normal = payload_part("pitch")
        cls.P = world.data()["meta"]["pitch"]
        ax = math.radians(cls.P["brg"])
        cls.along, cls.across = np.array([math.sin(ax), math.cos(ax)]), np.array([-math.cos(ax), math.sin(ax)])
        t = np.linspace(-math.pi, math.pi, 4000)
        cls.outer = LineString(np.column_stack([cls.P["a"] * np.cos(t), cls.P["b"] * np.sin(t)]))
        data = world.data()
        cls.grid, cls.bed = Grid(data["meta"]["local"]), decode(data["tex"]["locBed"])

    def uv(self, pts):
        xz = np.asarray(pts)[..., [0, 2]] - self.P["c"]
        return xz @ self.along, xz @ self.across

    def angle(self, u, v):
        """The registered ellipse's parameter angle (degrees): 0 at the west end, -90 the north side, 90 the south."""
        return np.degrees(np.arctan2(v / self.P["b"], u / self.P["a"]))

    def inset(self, pts):
        """Metres inside the registered outer edge of the stands."""
        u, v = self.uv(pts)
        flat = [self.outer.distance(Point(a, b)) for a, b in zip(np.ravel(u), np.ravel(v))]
        return np.array(flat).reshape(np.shape(u))

    def inward(self, pts):
        """Unit (x, z) directions towards the pitch, normal to the stands' rings."""
        u, v = self.uv(pts)
        g = -(u[..., None] / self.P["a"] ** 2 * self.along + v[..., None] / self.P["b"] ** 2 * self.across)
        return g / np.linalg.norm(g, axis=-1, keepdims=True)

    def faces(self, attr, up=None):
        keep = np.all(self.attr == attr, axis=1)
        if up is not None:
            keep &= (self.normal[:, 1] > 0.95) if up else (np.abs(self.normal[:, 1]) < 0.1)
        return self.tris[keep], self.normal[keep]

    def ground(self, pts):
        pts = np.asarray(pts).reshape(-1, 3)
        return sample(self.bed, self.grid, pts[:, 0], pts[:, 2])

    def edge_frame(self, deg):
        """(world (x, z), unit inward normal, unit tangent) of the registered outer edge at parameter angle `deg`."""
        t = math.radians(deg)
        edge = np.asarray(self.P["c"]) + self.P["a"] * math.cos(t) * self.along + self.P["b"] * math.sin(t) * self.across
        n = -(math.cos(t) / self.P["a"] * self.along + math.sin(t) / self.P["b"] * self.across)
        n /= np.linalg.norm(n)
        return edge, n, np.array([-n[1], n[0]])

    def decks(self):
        """Upward plank faces deeper than a bench, outside the changing-room block: the tiers' decks."""
        tris, _ = self.faces(mesh_grounds.PLANKS, up=True)
        depth = np.ptp(self.inset(tris), axis=1)
        a = self.angle(*self.uv(tris.mean(axis=1)))
        keep = (depth > 1.0) & (np.abs(a - 90.0) > 20.0)
        return tris[keep]

    def test_three_staggered_tiers_rise_behind_a_barrier_and_a_walkway(self):
        decks = self.decks()
        height = decks[:, :, 1].mean(axis=1) - self.P["level"]
        depth = self.inset(decks.mean(axis=1))
        levels = np.unique(np.round(height / 0.3))
        self.assertEqual(len(levels), 3)                                           # "probably only three"
        tiers = [depth[np.round(height / 0.3) == k].mean() for k in levels]
        self.assertTrue(all(a > b for a, b in zip(tiers, tiers[1:])))              # each higher tier further out
        self.assertTrue(all(np.diff(levels * 0.3) >= 0.9))                         # rows see over the heads below
        walls, n = self.faces(mesh_grounds.PLANKS, up=False)
        centre = walls.mean(axis=1)
        facing = np.einsum("ij,ij->i", n[:, [0, 2]], self.inward(centre))
        barrier = (np.abs(facing) > 0.9) & (self.inset(centre) > max(tiers) + 2.0) & (self.inset(centre) < max(tiers) + 4.5)
        self.assertTrue(barrier.any())
        top = walls[barrier][:, :, 1].max() - self.P["level"]
        self.assertTrue(0.9 < top < 1.3, top)                                      # low enough to vault over
        walk = self.inset(walls[barrier].mean(axis=1)).min() - (max(tiers) + 0.85)
        self.assertGreater(walk, 2.0)                                              # room for Hagrid to walk round

    def test_the_stands_are_hollow_and_open_between_the_tiers(self):
        tris, n = self.tris, self.normal
        centre = tris.mean(axis=1)
        a = self.angle(*self.uv(centre))
        stands = (np.abs(a - 90.0) > 20.0) & (self.inset(centre) < 7.0)
        across = np.abs(np.einsum("ij,ij->i", n[:, [0, 2]], self.inward(centre))) > 0.9
        wide = np.ptp(tris[:, :, [0, 2]], axis=1).max(axis=1) > 0.5
        tall = np.ptp(tris[:, :, 1], axis=1) > 0.6
        self.assertEqual(int((stands & across & wide & tall & (np.abs(n[:, 1]) < 0.1)).sum()), 0)   # no solid risers

    def test_four_15_ft_stairs_cut_through_the_stands(self):
        tris, _ = self.faces(mesh_grounds.PLANKS, up=True)
        centre = tris.mean(axis=1)
        tangent = self.inward(centre) @ np.array([[0.0, -1.0], [1.0, 0.0]])
        wide = np.array([np.ptp(t[:, [0, 2]] @ d) for t, d in zip(tris, tangent)])
        deep = np.ptp(self.inset(tris), axis=1)
        treads = tris[(wide > 15 * FT - 0.3) & (deep < 0.4)]
        angles = self.angle(*self.uv(treads.mean(axis=1)))
        for at in (-140.0, -40.0, 40.0, 140.0):
            with self.subTest(stair=at):
                mine = treads[np.abs(angles - at) < 10.0]
                self.assertGreater(len(mine), 6)
                rise = mine[:, :, 1].mean(axis=1) - self.P["level"]
                self.assertLess(rise.min(), 0.6)                                   # entered from the ground
                self.assertGreater(rise.max(), 3.0)                                # up to the highest tier
                edge, n, side = self.edge_frame(at)                                # and the tiers are cut away
                rel = centre[:, [0, 2]] - edge
                path = (np.abs(rel @ side) < 15 * FT / 2 - 0.3) & (rel @ n > 0.3) & (rel @ n < 4.5)
                self.assertTrue(path.any())
                self.assertTrue(np.all((wide[path] > 15 * FT - 0.3) & (deep[path] < 0.4)), "decks or seats in the stair")

    def test_the_stairs_climb_from_the_lawn_behind_the_stands(self):
        # pitch-stairs-from-lawn: "the exit from the stair opens onto the lawn, not inside the stadium itself"
        tris, _ = self.faces(mesh_grounds.PLANKS, up=True)
        centre = tris.mean(axis=1)
        u, v = self.uv(centre)
        outside = (u / self.P["a"]) ** 2 + (v / self.P["b"]) ** 2 > 1.0
        tangent = self.inward(centre) @ np.array([[0.0, -1.0], [1.0, 0.0]])
        wide = np.array([np.ptp(t[:, [0, 2]] @ d) for t, d in zip(tris, tangent)]) > 15 * FT - 0.3
        top = self.decks()[:, :, 1].mean(axis=1).max()
        for at in (-140.0, -40.0, 40.0, 140.0):
            with self.subTest(stair=at):
                mine = outside & wide & (np.abs(self.angle(u, v) - at) < 10.0)
                self.assertGreater(int(mine.sum()), 1)
                foot = centre[mine][np.argmax(self.inset(centre[mine]))]               # the tread furthest out
                ground = float(sample(self.bed, self.grid, foot[[0]], foot[[2]])[0])
                self.assertLess(abs(foot[1] - ground), 0.35)                            # it meets the lawn
                self.assertGreater(centre[mine][:, 1].max(), top - 0.3)                 # and the top tier behind

    def test_a_10_ft_entrance_opens_on_the_castle_side(self):
        u, v = 0.0, -self.P["b"]                                                   # the middle of the north side
        gap = self.P["c"] + u * self.along + v * self.across
        towards = -np.asarray(self.P["c"]) / np.linalg.norm(self.P["c"])
        self.assertGreater(float((gap - self.P["c"]) @ towards), 0.0)              # upslope, towards the castle
        centre = self.tris.mean(axis=1)
        side = (centre[:, [0, 2]] - gap) @ self.along
        north = self.uv(centre)[1] < 0.0                                           # the castle side only
        slot = (np.abs(side) < 10 * FT / 2 - 0.2) & (self.inset(centre) < 9.0) & (centre[:, 1] > self.P["level"] + 0.2)
        self.assertEqual(int((slot & north).sum()), 0)

    def test_six_golden_hoops_50_ft_high_and_17_ft_apart(self):
        gold, _ = self.faces(mesh_grounds.GOLD)
        pts = gold.reshape(-1, 3)
        u, v = self.uv(pts)
        for end in (1.0, -1.0):
            with self.subTest(end="west" if end > 0 else "east"):
                mine = end * u > 0
                posts = np.unique(np.round(v[mine] / 0.5)) * 0.5
                groups = [g for g in np.split(np.sort(v[mine]), np.flatnonzero(np.diff(np.sort(v[mine])) > 1.0) + 1)]
                self.assertEqual(len(groups), 3, posts)
                centres = [g.mean() for g in groups]
                self.assertTrue(np.allclose(np.diff(centres), 17 * FT, atol=0.1), centres)
                self.assertAlmostEqual(float(pts[mine][:, 1].max() - self.P["level"]), 50 * FT, delta=0.1)
                hoop = pts[mine & (pts[:, 1] > self.P["level"] + 45 * FT + 0.02) & (np.abs(v - centres[1]) < 1.5)]
                self.assertAlmostEqual(float(np.ptp(hoop[:, 1])), 5 * FT, delta=0.1)
                self.assertLess(float(np.abs(u[mine]).max()), 250 * FT - 5.0)        # some distance in from the ends

    def test_the_changing_rooms_interrupt_the_stands_with_doors_onto_the_pitch(self):
        walls, _ = self.faces(mesh_grounds.BOARDING)
        centre = walls.reshape(-1, 3).mean(axis=0)
        self.assertLess(abs(float(self.angle(*self.uv(centre)) - 90.0)), 10.0)    # where the map draws the block
        pts = walls.reshape(-1, 3)
        t = self.inward(centre)[None, :] @ np.array([[0.0, -1.0], [1.0, 0.0]])
        length = float(np.ptp(pts[:, [0, 2]] @ t[0]))
        self.assertAlmostEqual(length, 85 * FT, delta=1.0)
        doors, n = self.faces(mesh_grounds.DOOR)
        onto = np.einsum("ij,ij->i", n[:, [0, 2]], self.inward(doors.mean(axis=1))) > 0.9
        spots = np.sort(doors[onto].mean(axis=1)[:, [0, 2]] @ t[0])
        self.assertEqual(len(np.split(spots, np.flatnonzero(np.diff(spots) > 0.8) + 1)), 5)   # four rooms and Hooch
        tris, _ = self.faces(mesh_grounds.PLANKS, up=True)                          # deck-deep planks, block included
        deep = np.ptp(self.inset(tris), axis=1) > 1.0
        a = self.angle(*self.uv(tris.mean(axis=1)))
        self.assertFalse((deep & (np.abs(a - 90.0) < 4.0)).any())                  # no seats run through the block
        walls, wn = self.faces(mesh_grounds.PLANKS, up=False)                       # nor a barrier across its doors
        wc = walls.mean(axis=1)
        u, v = self.uv(wc)
        radial = np.abs(np.einsum("ij,ij->i", wn[:, [0, 2]], self.inward(wc))) > 0.9
        across = radial & (v > 0.0) & (np.abs(u) < 85 * FT / 2 - 0.5) & (self.inset(wc) > 18 * FT + 0.3)
        self.assertFalse(across.any(), "the doors do not open directly onto the pitch")

    def test_every_changing_room_has_its_drawn_door_onto_the_lawn(self):
        # White Hound's changing_rooms.jpg draws a door on each long side of every room
        doors, n = self.faces(mesh_grounds.DOOR)
        block = doors.reshape(-1, 3).mean(axis=0)
        t = self.inward(block)[None, :] @ np.array([[0.0, -1.0], [1.0, 0.0]])
        out = np.einsum("ij,ij->i", n[:, [0, 2]], self.inward(doors.mean(axis=1))) < -0.9
        out &= self.inset(doors.mean(axis=1)) < 2.0                                # in the block's outer wall
        spots = np.sort(doors[out].mean(axis=1)[:, [0, 2]] @ t[0] - block[[0, 2]] @ t[0])
        bay = 85 * FT / 5
        self.assertEqual({int(np.floor(s / bay + 2.5)) for s in spots}, {0, 1, 2, 3, 4})

    def test_the_castle_doors_are_in_sight_from_the_pitch(self):
        # "they saw the front doors of the castle open in the distance" (Wood's team, on the pitch)
        tris, attr, _ = payload_part("castle")
        oak = tris[np.all(attr == code("castle-oak"), axis=1)].reshape(-1, 3)
        named = np.asarray(world.data()["meta"]["named"]["EH"], np.float64)
        door = oak[np.hypot(*(oak[:, [0, 2]] - named).T) < 40.0]
        top = np.array([door[:, 0].mean(), door[:, 1].min() + 0.6 * np.ptp(door[:, 1]), door[:, 2].mean()])
        blockers = np.concatenate([self.tris] + [payload_part(a)[0] for a in ("gates", "grounds-walls", "field-boundaries")])
        for du, dv in ((0.0, 0.0), (-30.0, 0.0), (30.0, 0.0)):                       # along the middle of the pitch
            with self.subTest(at=(du, dv)):
                eye = np.asarray(self.P["c"]) + du * self.along + dv * self.across
                start = np.array([eye[0], self.P["level"] + 1.7, eye[1]])
                v = top - start
                length = float(np.linalg.norm(v))
                s = np.linspace(1.0, length - 2.0, 2000)
                ray = start + s[:, None] * v / length
                self.assertFalse((sample(self.bed, self.grid, ray[:, 0], ray[:, 2]) > ray[:, 1]).any())   # over the land
                self.assertFalse(world.segment_hits(start[None], (v / length)[None], length - 12.0, blockers)[0])

    def test_every_changing_room_has_a_window(self):
        # "Ron strode across to the window and stared out into the rain" (in the Gryffindor changing room), and
        # White Hound's "at least one window per changing room": every room of the block, whichever is Gryffindor's
        glass, n = self.faces(mesh_grounds.GLASS)
        doors, _ = self.faces(mesh_grounds.DOOR)
        block = doors.reshape(-1, 3).mean(axis=0)
        t = self.inward(block)[None, :] @ np.array([[0.0, -1.0], [1.0, 0.0]])
        near = np.hypot(*(glass.mean(axis=1)[:, [0, 2]] - block[[0, 2]]).T) < 20.0
        spots = np.sort(glass[near].mean(axis=1)[:, [0, 2]] @ t[0] - block[[0, 2]] @ t[0])
        bay = 85 * FT / 5
        rooms = {int(np.floor(s / bay + 2.5)) for s in spots}
        self.assertEqual(rooms, {0, 1, 2, 3, 4})

    def test_the_changing_rooms_doors_stand_on_the_ground(self):
        doors, n = self.faces(mesh_grounds.DOOR)
        onto = np.einsum("ij,ij->i", n[:, [0, 2]], self.inward(doors.mean(axis=1))) > 0.9
        pts = doors[onto].reshape(-1, 3)
        clearance = pts[:, 1] - self.ground(pts)
        self.assertGreater(float(clearance.min()), -0.02)                           # not sunk into the grass
        self.assertLess(float(clearance.min()), 0.25)                               # nor a step up to reach

    def test_the_decks_stand_clear_of_the_ground(self):
        pts = self.decks().reshape(-1, 3)
        self.assertGreater(float((pts[:, 1] - self.ground(pts)).min()), 0.0)

    def test_a_freestanding_podium_midway_along_the_castle_side(self):
        tris, _ = self.faces(mesh_grounds.PLANKS, up=True)
        centre = tris.mean(axis=1)
        a, depth = self.angle(*self.uv(centre)), self.inset(centre)
        height = centre[:, 1] - self.P["level"]
        top = tris[(np.abs(a + 90.0) < 20.0) & (depth > 9.0) & (height > 2.0) & (height < 4.5)]
        self.assertGreater(len(top), 0)
        walls, n = self.faces(mesh_grounds.PLANKS, up=False)
        wc = walls.mean(axis=1)
        barrier = walls[(np.abs(np.einsum("ij,ij->i", n[:, [0, 2]], self.inward(wc))) > 0.9) & (self.inset(wc) > 6.5)
                        & (self.inset(wc) < 8.5)].reshape(-1, 3)
        gap = min(np.hypot(*(barrier[:, [0, 2]] - p[[0, 2]]).T).min() for p in top.reshape(-1, 3))
        self.assertGreater(gap, 1.0)                                               # not built into the stands


class GatesTests(unittest.TestCase):
    """The front gates (gates-pillars-boars): set into the boundary wall where it opens for the road (the mapped gap,
    meta.extras.gates), a pair of tall stone pillars either side of the road, each topped by a winged boar facing out of
    the grounds, and high, paired gates of wrought iron, closed, between them; the wall runs on unbroken from each pillar
    to the boundary wall."""

    @classmethod
    def setUpClass(cls):
        cls.tris, cls.attr, cls.normal = payload_part("gates")
        data = world.data()
        cls.A, cls.B = (np.asarray(p, np.float64) for p in data["meta"]["extras"]["gates"])
        cls.d = (cls.B - cls.A) / np.linalg.norm(cls.B - cls.A)
        cls.grid, cls.bed = Grid(data["meta"]["local"]), decode(data["tex"]["locBed"])
        mid = (cls.A + cls.B) / 2
        tops = []
        for tris, attr, normal in (payload_part("grounds-walls"), (cls.tris, cls.attr, cls.normal)):
            keep = (normal[:, 1] > 0.95) & (np.hypot(*(tris.mean(axis=1)[:, [0, 2]] - mid).T) < 40.0)
            keep &= np.all(attr == mesh_grounds.WALL_STONE, axis=1) | np.all(attr == code("pale-stone"), axis=1)
            tops += [Polygon(t[:, [0, 2]]) for t in tris[keep]]
        cls.walls = shapely.union_all([t.buffer(0.02) for t in tops if t.area > 1e-6])

    def of(self, attr):
        return self.tris[np.all(self.attr == attr, axis=1)]

    def ground(self, xz):
        xz = np.asarray(xz).reshape(-1, 2)
        return sample(self.bed, self.grid, xz[:, 0], xz[:, 1])

    def pillars(self):
        """(centre (x, z), foot, top) of each pillar, ordered along the gap: the pale cornices high over the ground (the
        walls' coping is far lower), split where the way opens, each up to the top of its cornice."""
        stone = self.of(code("pale-stone"))
        pts = stone.reshape(-1, 3)
        shafts = pts[pts[:, 1] - self.ground(pts[:, [0, 2]]) > 3.5]
        s = (shafts[:, [0, 2]] - self.A) @ self.d
        order = np.sort(np.unique(np.round(s, 1)))
        cut = order[np.argmax(np.diff(order))] + 0.05
        everything = stone.reshape(-1, 3)
        out = []
        for keep in (s < cut, s >= cut):
            p = shafts[keep]
            c = (p[:, [0, 2]].min(axis=0) + p[:, [0, 2]].max(axis=0)) / 2
            top = everything[np.hypot(*(everything[:, [0, 2]] - c).T) < 0.9][:, 1].max()
            out.append((c, float(self.ground(c)[0]), float(top)))
        return out

    def axis(self):
        """(first pillar's centre, unit direction to the second, their distance)."""
        (c0, _, _), (c1, _, _) = self.pillars()
        return c0, (c1 - c0) / np.linalg.norm(c1 - c0), float(np.linalg.norm(c1 - c0))

    def across(self, pts):
        """Metres along the line from the first pillar to the second, and metres off it."""
        c0, u, _ = self.axis()
        rel = np.asarray(pts)[..., [0, 2]] - c0
        return rel @ u, rel @ np.array([-u[1], u[0]])

    def test_two_tall_stone_pillars_stand_either_side_of_the_road(self):
        _, _, width = self.axis()
        for c, foot, top in self.pillars():
            self.assertGreater(top - foot, 4.5)                                   # "tall stone pillars"
        self.assertTrue(4.5 < width < 12.0, width)                                # room for the carriages, and no more

    def test_the_wall_runs_on_from_each_pillar_to_the_boundary_wall(self):
        c0, u, width = self.axis()
        for c, sign in ((c0, -1.0), (c0 + u * width, 1.0)):                       # outwards from each pillar
            with self.subTest(pillar=c.round(1).tolist()):
                line = [Point(*(c + sign * u * t)) for t in np.arange(0.6, 12.0, 0.25)]
                gaps = [round(float(t), 2) for t, q in zip(np.arange(0.6, 12.0, 0.25), line) if self.walls.distance(q) > 0.35]
                self.assertEqual(gaps, [])                                        # nowhere to walk round the gates
        centre = self.tris.mean(axis=1)[:, [0, 2]]
        pier = np.min([np.max(np.abs(centre - c), axis=1) for c, _, _ in self.pillars()], axis=0) < mesh_grounds.PIER_M
        stone = np.all(self.attr == mesh_grounds.WALL_STONE, axis=1) & ~pier          # the walls, not the pillars
        s, _ = self.across(self.tris[stone].reshape(-1, 3))
        self.assertFalse(((s > 0.6) & (s < width - 0.6)).any())                   # but not across the way
        self.assertEqual(int((stone & (self.normal[:, 1] > 0.95)).sum()), 0)       # and coped like the rest

    def test_a_winged_boar_tops_each_pillar_facing_out_of_the_grounds(self):
        boars = self.of(code("boar-stone")).reshape(-1, 3)
        for c, _, top in self.pillars():
            with self.subTest(pillar=c.round(1).tolist()):
                mine = boars[np.hypot(*(boars[:, [0, 2]] - c).T) < 1.6]
                self.assertGreater(len(mine), 100)
                self.assertAlmostEqual(float(mine[:, 1].min()), top, delta=0.05)  # standing on it
                h = mine[:, 1] - mine[:, 1].min()
                self.assertGreater(np.ptp(h), 1.0)
                out = c / np.linalg.norm(c)                                       # away from the castle (the origin)
                side = np.array([-out[1], out[0]])
                low, high = mine[h < 0.45 * h.max()], mine[h > 0.6 * h.max()]
                body = np.ptp(low[:, [0, 2]] @ side)
                self.assertGreater(np.ptp(high[:, [0, 2]] @ side), max(1.0, body + 0.3))   # wings spread over the body
                head = mine[(h > 0.15 * h.max()) & (h < 0.45 * h.max())]
                reach = (head[:, [0, 2]] - c) @ out
                self.assertGreater(reach.max(), -reach.min() + 0.2)               # its snout towards the road outside

    def test_the_boar_is_stocky_its_deep_body_low_on_short_legs(self):
        boar = np.load(ROOT / "src" / "assets" / "grounds" / "boar.npz")
        pos = boar["pos"].astype(np.float64)
        tri = pos[boar["idx"]]
        cen = tri.mean(axis=1)
        mid = cen[(np.abs(cen[:, 0]) < 0.15) & (np.abs(cen[:, 2]) < 0.12)]       # across the middle of the body
        self.assertLess(float(mid[:, 1].min()), 0.35)                            # the belly low over short legs
        self.assertGreater(float(np.ptp(mid[mid[:, 1] < 1.3][:, 1])), 0.75)      # under a deep barrel of a body

    def test_high_paired_iron_gates_close_the_way_between_the_pillars(self):
        _, _, width = self.axis()
        iron = self.of(code("black-iron")).reshape(-1, 3)
        s, off = self.across(iron)
        between = (s > 0.6) & (s < width - 0.6)
        self.assertLess(float(np.abs(off[between]).max()), 0.3)                  # closed, across the way
        bins = np.floor(s[between] / 0.25)
        self.assertGreater(len(np.unique(bins)) * 0.25, 0.9 * (width - 1.2))     # from pillar to pillar
        pts = iron[between]
        clearance = pts[:, 1] - self.ground(pts[:, [0, 2]])
        lowest = [clearance[bins == k].min() for k in np.unique(bins)]
        highest = [clearance[bins == k].max() for k in np.unique(bins)]
        self.assertGreater(float(min(highest)), 2.5)                              # "high", all the way across
        self.assertLess(float(max(lowest)), 0.2)                                  # and no way under them


class WallsTests(unittest.TestCase):
    """The boundary walls (3a) under a coping course, and the wall at the edge of the vegetable patch (5d), White
    Hound's "low wall", the remnant of the former curtain wall: thick, low and broken-topped (walls-coped,
    vegetable-patch-wall-ruin)."""

    @classmethod
    def setUpClass(cls):
        cls.tris, cls.attr, cls.normal = payload_part("grounds-walls")
        data = world.data()
        cls.grid, cls.bed = Grid(data["meta"]["local"]), decode(data["tex"]["locBed"])
        asset = next(a for a in world.scene()["assets"] if a["id"] == "grounds-walls")
        cls.patch = [np.asarray(p["at"]) for p in asset["castle_pieces"] if "(5d)" in p["note"]]
        centre = cls.tris.mean(axis=1)[:, [0, 2]]
        cls.near_patch = np.min([np.hypot(*(centre - p).T) for p in cls.patch], axis=0) < 32.0
        cliff = [np.asarray(p["at"]) for p in asset["castle_pieces"] if "(2e)" in p["note"]]
        cls.near_cliff = np.min([np.hypot(*(centre - p).T) for p in cliff], axis=0) < 100.0
        cls.boundary = np.all(cls.attr == mesh_grounds.WALL_STONE, axis=1) | np.all(cls.attr == code("pale-stone"), axis=1)

    def height(self, tris):
        pts = tris.reshape(-1, 3)
        return pts[:, 1] - sample(self.bed, self.grid, pts[:, 0], pts[:, 2])

    def test_the_boundary_walls_carry_a_coping(self):
        from scipy.spatial import cKDTree
        keep = self.boundary & ~self.near_patch & ~self.near_cliff
        up = keep & (self.normal[:, 1] > 0.95)
        stone = np.all(self.attr == mesh_grounds.WALL_STONE, axis=1)
        self.assertGreater(int((up & ~stone).sum()), 100)
        self.assertEqual(int((up & stone).sum()), 0)                               # every top is coped
        coping = self.height(self.tris[up & ~stone])
        self.assertGreater(float(np.median(coping)), 2.0)
        cap = self.tris[keep & ~stone].reshape(-1, 3)                               # and the stone stops under it
        body = self.tris[keep & stone].reshape(-1, 3)
        tree = cKDTree(cap[:, [0, 2]])
        near = tree.query_ball_point(body[:, [0, 2]], 1.0)
        over = np.array([cap[i, 1].max() - y if i else -1.0 for i, y in zip(near, body[:, 1])])
        self.assertGreater(float(over.min()), 0.05)                               # within 1 m, on slopes

    def test_the_boundary_walls_run_on_across_breaks_narrower_than_a_gateway(self):
        keep = self.boundary & ~self.near_patch & ~self.near_cliff & (self.normal[:, 1] > 0.2)   # coping tops, tilted on banks too
        plan = shapely.union_all([Polygon(t[:, [0, 2]]).buffer(0.05) for t in self.tris[keep]])
        pieces = list(getattr(plan, "geoms", [plan]))
        tree = shapely.STRtree(pieces)
        for i, p in enumerate(pieces):
            gap = min((p.distance(pieces[j]) for j in tree.query(p.buffer(4.0)) if j != i), default=np.inf)
            with self.subTest(at=np.round(p.centroid.coords[0]).tolist()):     # walls-coped: breaks under 3.5 m closed
                self.assertGreater(gap, 3.5 - 2 * mesh_grounds.COPING_OVER_M - 0.15)   # less the copings and buffer

    def test_the_vegetable_patch_wall_is_a_low_ruin(self):
        mine = self.near_patch & (self.normal[:, 1] > 0.95)
        self.assertGreater(int(mine.sum()), 10)
        tops = self.height(self.tris[mine])
        self.assertTrue(0.7 < float(tops.min()) and float(tops.max()) < 1.6, (tops.min(), tops.max()))   # "low"
        self.assertGreater(float(np.ptp(tops)), 0.25)                              # broken, not coped level
        for p in self.patch:                                                       # a wall, not the map's broad ink
            near = self.tris[mine & (np.hypot(*(self.tris.mean(axis=1)[:, [0, 2]] - p).T) < 20.0)]
            foot = shapely.union_all([Polygon(t[:, [0, 2]]).buffer(0.01) for t in near])
            rect = foot.minimum_rotated_rectangle
            e = np.asarray(rect.exterior.coords)
            width = min(np.hypot(*(e[1] - e[0])), np.hypot(*(e[2] - e[1])))
            self.assertLess(width, 1.6)


    def test_the_cliff_edge_wall_is_a_low_safety_wall_on_the_lake_side(self):
        # cliff-edge-safety-wall: "(2e) wall along edge of cliff overlooking lake", "(2f) path along edge of cliff,
        # behind safety wall", not a masonry block from the cliff's foot to the castle platform
        asset = next(a for a in world.scene()["assets"] if a["id"] == "grounds-walls")
        cliff = [np.asarray(p["at"]) for p in asset["castle_pieces"] if "(2e)" in p["note"]]
        centre = self.tris.mean(axis=1)
        mine = np.min([np.hypot(*(centre[:, [0, 2]] - p).T) for p in cliff], axis=0) < 100.0
        self.assertGreater(int(mine.sum()), 50)
        self.assertFalse(np.any(self.attr[mine, 0] == 4))                          # no castle massing left
        tops = mine & (self.normal[:, 1] > 0.7)                                    # it climbs along the cliff top
        mid = centre[tops]
        for p in cliff:                                                            # along the whole of every piece
            with self.subTest(piece=p.tolist()):
                self.assertTrue((np.hypot(*(mid[:, [0, 2]] - p).T) < 25.0).any())
                from scipy.sparse.csgraph import minimum_spanning_tree
                faces = self.tris[tops & (np.hypot(*(centre[:, [0, 2]] - p).T) < 100.0)]
                plan = shapely.union_all([Polygon(t[:, [0, 2]]).buffer(0.3) for t in faces])
                parts = list(getattr(plan, "geoms", [plan]))
                gaps = np.array([[a_.distance(b_) for b_ in parts] for a_ in parts])
                widest = minimum_spanning_tree(gaps).toarray().max() if len(parts) > 1 else 0.0
                self.assertLess(float(widest), 12.0)                               # no stretch left unwalled
        ring = 0.8 * np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0.0, 2 * math.pi, 16, endpoint=False)])
        behind = np.max([sample(self.bed, self.grid, *(mid[:, [0, 2]] + o).T) for o in ring], axis=0)
        heights = mid[:, 1] - behind                                               # as it stands to the path behind it
        self.assertTrue(0.8 < float(np.median(heights)) < 1.5, np.median(heights))  # a "lower safety-wall"
        tops = mid
        away = tops[:, [0, 2]] / np.linalg.norm(tops[:, [0, 2]], axis=1, keepdims=True)   # away from the castle
        g = lambda xz: sample(self.bed, self.grid, xz[:, 0], xz[:, 1])
        drop = g(tops[:, [0, 2]]) - g(tops[:, [0, 2]] + 3.0 * away)
        self.assertGreater(float(np.median(drop)), 0.3)                            # the drop is beyond the wall
        lv = world.levels()                                                        # and it runs along the cliff's top
        rise = (np.median(tops[:, 1]) - lv["lake"]) / (lv["platform"] - lv["lake"])
        self.assertGreater(float(rise), 0.4)
        inward = -away                                                             # on the brink, not down the face:
        rise = g(tops[:, [0, 2]] + 3.0 * inward) - g(tops[:, [0, 2]] + 0.5 * inward)
        self.assertLess(float(np.percentile(rise, 90)), 2.0)                       # the path behind climbs gently
        fall = g(tops[:, [0, 2]]) - g(tops[:, [0, 2]] + 3.0 * away)
        self.assertGreater(float(np.percentile(fall, 20)), 1.5)                    # and the drop is right beyond it
        cap = self.tris[mine & (self.normal[:, 1] > 0.7) & np.all(self.attr == mesh_grounds.PALE_STONE, axis=1)]
        plan = shapely.union_all([Polygon(t[:, [0, 2]]).buffer(0.02) for t in cap])
        knots = [r for g_ in getattr(plan, "geoms", [plan]) for r in g_.interiors if Polygon(r).area > 0.2]
        self.assertEqual(len(knots), 0)                                            # one wall, not knotted loops

    def test_the_cliff_wall_s_coping_sits_on_its_stone(self):
        # under the coping course, everywhere along the wall: no stone through it, no slit beneath it
        mine = self.near_cliff
        stone = self.tris[mine & np.all(self.attr == mesh_grounds.WALL_STONE, axis=1)].reshape(-1, 3)
        cap = self.tris[mine & np.all(self.attr == mesh_grounds.PALE_STONE, axis=1) & (self.normal[:, 1] > 0.7)]
        from scipy.spatial import cKDTree
        xz = np.round(stone[:, [0, 2]], 3)
        _, inv = np.unique(xz, axis=0, return_inverse=True)
        top = np.full(inv.max() + 1, -np.inf)
        np.maximum.at(top, inv.ravel(), stone[:, 1])
        tops = stone[stone[:, 1] >= top[inv.ravel()] - 1e-6]                       # the stone's top at each corner
        a, b, c = cap[:, 0][:, [0, 2]], cap[:, 1][:, [0, 2]], cap[:, 2][:, [0, 2]]
        tree = cKDTree(cap.mean(axis=1)[:, [0, 2]])
        reach = float(np.max(np.linalg.norm(cap[:, :, [0, 2]] - cap.mean(axis=1)[:, None, [0, 2]], axis=2)))
        gaps = []
        for p in tops:
            for i in tree.query_ball_point(p[[0, 2]], reach + 1e-3):
                m = np.column_stack([b[i] - a[i], c[i] - a[i]])
                if abs(np.linalg.det(m)) < 1e-12:
                    continue
                s, t = np.linalg.solve(m, p[[0, 2]] - a[i])
                if s >= -1e-6 and t >= -1e-6 and s + t <= 1 + 1e-6:
                    y = cap[i, 0, 1] + s * (cap[i, 1, 1] - cap[i, 0, 1]) + t * (cap[i, 2, 1] - cap[i, 0, 1])
                    gaps.append(y - p[1])
                    break
        gaps = np.array(gaps)
        self.assertGreater(len(gaps), 0.9 * len(tops))
        self.assertGreater(float(gaps.min()), 0.05)                                # no stone through the coping
        self.assertLess(float(gaps.max()), mesh_grounds.COPING_M + 0.15)           # nor a slit under it

    def test_the_cliff_wall_runs_on_over_the_harbour_roof(self):
        # the harbour's rock roof carries the cliff top over the recess cut beneath it, so the wall continues across
        meta = world.data()["meta"]
        rec = meta["extras"]["harbour"]
        m, u = np.asarray(rec["mouth"], np.float64), np.asarray(rec["axis"], np.float64)
        cap = self.tris[np.all(self.attr == mesh_grounds.PALE_STONE, axis=1) & (self.normal[:, 1] > 0.7)]
        side = np.array([-u[1], u[0]])
        rel = cap[..., [0, 2]] - m
        local = np.stack([rel @ u, rel @ side], -1)                                # (along the axis, across)
        half, deep = mesh_grounds.CHANNEL_HALF_M, rec["depth"] + mesh_grounds.ROOF_PAST_M
        band = shapely.box(0.0, -half, deep, half)
        hits = [i for i, t in enumerate(local) if Polygon(t).buffer(0.01).intersects(band)]
        self.assertTrue(hits)
        plan = shapely.union_all([Polygon(local[i]).buffer(0.01) for i in hits]).intersection(band)
        lo, hi = plan.bounds[1], plan.bounds[3]
        self.assertTrue(lo < -half + 0.1 and hi > half - 0.1, (lo, hi))           # right across the channel
        self.assertGreater(float(cap[hits][..., 1].min()) - float(meta["lake"]), mesh_grounds.CROWN_M + 1.0)   # on the roof

    def test_the_cliff_wall_hangs_nowhere_down_the_cliff_face(self):
        # away from the harbour's roof its stone is founded under the brink, not hung in teeth down the face below it
        rec = world.data()["meta"]["extras"]["harbour"]
        m, u = np.asarray(rec["mouth"], np.float64), np.asarray(rec["axis"], np.float64)
        stone = self.tris[self.near_cliff & np.all(self.attr == mesh_grounds.WALL_STONE, axis=1)].reshape(-1, 3)
        rel = stone[:, [0, 2]] - m
        a, w = rel @ u, rel @ np.array([-u[1], u[0]])
        roof = (a > mesh_grounds.PORTAL_M - 3.0) & (a < rec["depth"] + mesh_grounds.ROOF_PAST_M + 3.0) \
            & (np.abs(w) < mesh_grounds.SHELL_HALF_M + 3.0)
        below = stone[~roof, 1] - sample(self.bed, self.grid, stone[~roof, 0], stone[~roof, 2])
        import build_meshes
        self.assertGreater(float(below.min()), -build_meshes.SINK_M - 0.3)          # sunk as every wall is, no deeper
        from scipy.spatial import cKDTree
        cap = self.tris[self.near_cliff & np.all(self.attr == mesh_grounds.PALE_STONE, axis=1) & (self.normal[:, 1] > 0.7)]
        cap = cap.reshape(-1, 3)
        _, i = cKDTree(cap[:, [0, 2]]).query(stone[~roof][:, [0, 2]])
        depth = cap[i, 1] - stone[~roof, 1]                                        # stone under its coping's top:
        self.assertLess(float(depth.max()), mesh_grounds.SAFETY_WALL_M + 1.5)      # a low wall, no teeth down the face

    def test_the_cliff_wall_reaches_the_ground_beside_the_harbour_roof(self):
        # beside the roof, a ray across the wall just above the ground meets its stone: no slot under it where the cut's
        # bank dips beside the shell (further out the wall stands on the open cliff's brink, its foot under the path)
        rec = world.data()["meta"]["extras"]["harbour"]
        m, u = np.asarray(rec["mouth"], np.float64), np.asarray(rec["axis"], np.float64)
        side = np.array([-u[1], u[0]])
        stone = self.tris[np.all(self.attr == mesh_grounds.WALL_STONE, axis=1)]
        cap = self.tris[np.all(self.attr == mesh_grounds.PALE_STONE, axis=1) & (self.normal[:, 1] > 0.7)]
        rel = cap[..., [0, 2]] - m
        local = np.stack([rel @ u, rel @ side], -1)
        near = (np.abs(local[..., 1]).min(axis=1) < 16.0) & (local[..., 0].min(axis=1) < rec["depth"] + 10.0)
        plan = shapely.union_all([Polygon(t).buffer(0.01) for t in local[near]])
        starts = []
        reach = mesh_grounds.SHELL_HALF_M + 2.0
        for w in np.arange(-reach, reach, 0.25):
            if abs(w) <= mesh_grounds.SHELL_HALF_M:
                continue                                                           # on the roof, over the channel
            cut = plan.intersection(LineString([(-5.0, w), (rec["depth"] + 10.0, w)]))
            if cut.is_empty:
                continue
            a = 0.5 * (cut.bounds[0] + cut.bounds[2])
            p = m + (a - 1.5) * u + w * side
            ground = float(sample(self.bed, self.grid, np.array([m[0] + a * u[0] + w * side[0]]),
                                  np.array([m[1] + a * u[1] + w * side[1]]))[0])
            starts.append([p[0], ground + 0.3, p[1]])
        self.assertGreater(len(starts), 8)
        starts = np.array(starts)
        hits = world.segment_hits(starts, np.tile([u[0], 0.0, u[1]], (len(starts), 1)), 3.0, stone)
        self.assertTrue(hits.all(), starts[~hits][:4])

    @staticmethod
    def cliff(brink_at, terrace=False):
        """A synthetic cliff top falling 30 m to the lake (+z) from a brink at z = brink_at(x), with a 1 m terrace step
        4.5 m in from the mapped edge (z = 0) if asked; the mapped edge along x with outward normals +z."""
        def ground(x, z):
            x, z = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(z, np.float64))
            g = np.interp(z - brink_at(x), [-10.0, 0.0, 7.5, 20.0], [30.0, 30.0, 0.0, 0.0])
            return g + (np.interp(z, [-4.75, -4.25], [1.0, 0.0]) if terrace else 0.0)
        scene = types.SimpleNamespace(meta={"extras": {}}, ground=ground)
        p = np.column_stack([np.arange(0.0, 96.0, 0.5), np.zeros(192)])
        return scene, p, np.tile([0.0, 1.0], (len(p), 1))

    def test_the_cliff_wall_takes_the_brink_nearest_the_mapped_edge(self):
        scene, p, n = self.cliff(lambda x: 0.5 + 0 * x, terrace=True)              # not the terrace step further in
        on, by = mesh_grounds.brink(scene, p, n)
        expect = 0.5 - mesh_grounds.SAFETY_THICK_M / 2 - mesh_grounds.SAFETY_INSET_M
        self.assertTrue(np.allclose(on[:, 1], expect, atol=0.15), (on[:, 1].min(), on[:, 1].max()))
        self.assertTrue(np.allclose(by[:, 1], expect - mesh_grounds.SAFETY_PATH_M, atol=0.15))

    def test_the_cliff_wall_s_outer_face_stands_on_the_brink(self):
        # seen from the lake, its coping marks the brink itself: not set back on the cliff top behind it
        scene, p, n = self.cliff(lambda x: 0.5 + 0 * x)
        on, _ = mesh_grounds.brink(scene, p, n)
        outer = on[:, 1] + mesh_grounds.SAFETY_THICK_M / 2
        self.assertTrue(np.allclose(outer, 0.5, atol=0.3), (outer.min(), outer.max()))

    def test_the_cliff_wall_eases_round_steps_in_the_brink(self):
        step = lambda x: np.where((np.asarray(x) // 16.0) % 2 == 0, 0.5, -1.5)     # the brink jogs 2 m every 16 m
        scene, p, n = self.cliff(step)
        on, _ = mesh_grounds.brink(scene, p, n)
        expect = step(on[:, 0]) - mesh_grounds.SAFETY_THICK_M / 2 - mesh_grounds.SAFETY_INSET_M
        mid = np.abs((on[:, 0] % 16.0) - 8.0) < 1.0
        self.assertTrue(np.allclose(on[mid, 1], expect[mid], atol=0.2))             # it follows the brink
        line = LineString(on)
        q = np.array([line.interpolate(t).coords[0] for t in np.arange(0.0, line.length, 1.0)])
        heading = np.unwrap(np.arctan2(*np.diff(q, axis=0).T[::-1]))
        self.assertLess(float(np.degrees(np.abs(np.diff(heading)).max())), 30.0)   # in an easy line, not jogging


class HedgesTests(unittest.TestCase):
    """The field hedges: continuous rows trimmed to about 1.5 m, rounded over the top with irregular crowns, joined across
    the breaks narrower than a gateway that the map's ink leaves at bends and along the rows (hedge-crowns)."""

    @classmethod
    def setUpClass(cls):
        cls.tris, cls.attr, cls.normal = payload_part("field-boundaries")
        data = world.data()
        cls.grid, cls.bed = Grid(data["meta"]["local"]), decode(data["tex"]["locBed"])
        leaves = cls.tris[np.all(cls.attr == mesh_grounds.FOLIAGE, axis=1)]
        cls.plan = shapely.union_all([shapely.MultiPoint(t[:, [0, 2]]).convex_hull.buffer(0.05) for t in leaves])
        cls.pieces = list(getattr(cls.plan, "geoms", [cls.plan]))

    @classmethod
    def ink(cls):
        if not hasattr(cls, "_ink"):
            data = world.data()
            road = decode(data["tex"]["locA"])[..., 0] / 255.0
            cls._ink = shapely.union_all(mesh_source.field_polygons(-road, cls.grid, -0.5))
        return cls._ink

    @staticmethod
    def row(p):
        e = np.asarray(p.minimum_rotated_rectangle.exterior.coords)
        return max(np.hypot(*(e[1] - e[0])), np.hypot(*(e[2] - e[1]))) > 4.0

    def gateway(self, a, b):
        """Whether a road or path runs through the break between two pieces."""
        from shapely.ops import nearest_points
        return LineString(nearest_points(a, b)).buffer(0.5).intersects(self.ink())

    def above_ground(self, pts):
        return pts[:, 1] - sample(self.bed, self.grid, pts[:, 0], pts[:, 2])

    def crest(self):
        """The top's points along the crest, more than 0.42 m in from the hedge's sides, and those within 0.15 m of
        them (the plan is buffered 0.05 m)."""
        up = self.tris[np.all(self.attr == mesh_grounds.FOLIAGE, axis=1) & (self.normal[:, 1] > 0.3)].reshape(-1, 3)
        edge = shapely.distance(self.plan.boundary, shapely.points(up[:, [0, 2]]))
        return up[edge > 0.42], up[edge < 0.15]

    def test_the_hedges_have_irregular_crowns(self):
        mid, _ = self.crest()
        h = self.above_ground(mid)
        self.assertTrue(1.2 < float(np.percentile(h, 5)) and float(np.percentile(h, 95)) < 1.9)
        self.assertGreater(float(np.percentile(h, 90) - np.percentile(h, 10)), 0.15)   # not trimmed flat

    def test_the_hedges_are_rounded_over_the_top(self):
        mid, side = self.crest()
        self.assertGreater(float(np.median(self.above_ground(mid)) - np.median(self.above_ground(side))), 0.2)
        up = self.normal[np.all(self.attr == mesh_grounds.FOLIAGE, axis=1) & (self.normal[:, 1] > 0.3)]
        self.assertLess(float((up[:, 1] > 0.995).mean()), 0.25)                    # no flat lids

    def test_the_rows_run_on_across_breaks_narrower_than_a_gateway(self):
        tree = shapely.STRtree(self.pieces)
        for i, p in enumerate(self.pieces):
            if not self.row(p):
                continue                                                           # a round shrub, not a row
            near = [j for j in tree.query(p.buffer(4.5)) if j != i and self.row(self.pieces[j])
                    and not self.gateway(p, self.pieces[j])]
            gap = min((p.distance(self.pieces[j]) for j in near), default=np.inf)
            with self.subTest(at=np.round(p.centroid.coords[0]).tolist()):     # hedge-crowns: breaks under 4 m closed
                self.assertGreater(gap, 4.0 - 0.15)                                # no dashed rows (plans buffered)

    def test_no_hedge_stands_alone_as_a_block(self):
        for p in self.pieces:                                                      # a row, or else a round shrub
            rect = p.minimum_rotated_rectangle
            e = np.asarray(rect.exterior.coords)
            with self.subTest(at=np.round(p.centroid.coords[0]).tolist()):
                self.assertTrue(max(np.hypot(*(e[1] - e[0])), np.hypot(*(e[2] - e[1]))) > 4.0 or p.area / rect.area < 0.85)


class TombTests(unittest.TestCase):
    """Dumbledore's white marble tomb on the lakeside promontory (meta.extras.tomb), tall enough to reflect in the lake:
    a chest tomb on a stepped base (tomb-chest)."""

    def test_a_white_marble_chest_tomb_on_steps(self):
        tris, attr, normal = payload_part("tomb")
        at = np.asarray(world.data()["meta"]["extras"]["tomb"])
        marble = np.all(attr == code("white-marble"), axis=1)
        self.assertGreater(float(marble.mean()), 0.95)
        pts = tris[marble].reshape(-1, 3)
        self.assertLess(float(np.hypot(*(pts[:, [0, 2]].mean(axis=0) - at))), 1.0)
        rect = shapely.MultiPoint(pts[:, [0, 2]]).minimum_rotated_rectangle
        e = np.asarray(rect.exterior.coords)
        length, width = sorted((np.hypot(*(e[1] - e[0])), np.hypot(*(e[2] - e[1]))), reverse=True)
        self.assertTrue(2.8 < length < 4.2 and 1.4 < width < 2.6, (length, width))
        data = world.data()
        ground = sample(decode(data["tex"]["locBed"]), Grid(data["meta"]["local"]), pts[:, 0], pts[:, 2])
        self.assertTrue(1.4 < float((pts[:, 1] - ground).max()) < 2.4)              # tall enough to reflect
        self.assertLess(float((pts[:, 1] - ground).min()), 0.0)                     # standing on the ground
        up = tris[marble & (normal[:, 1] > 0.999)]
        levels = np.round(up[:, :, 1].mean(axis=1) - float(up[:, :, 1].min()), 2)
        span = lambda sel: np.ptp(sel.reshape(-1, 3)[:, [0, 2]], axis=0)
        base, chest = up[levels == levels.min()], up[levels > 1.0]
        self.assertTrue(np.all(span(base) > span(chest) + 0.6), (span(base), span(chest)))   # on a stepped base


class LakesidePathTests(unittest.TestCase):
    """"while they took another long walk around the lake": a walkable band of ground near the shore (gentler than 0.6
    in 1, within 40 m of the water) runs right round the lake, and none of the grounds' walls, hedges, fences, gates,
    stands or tomb cuts it."""

    def test_the_lake_can_be_walked_right_round(self):
        data = world.data()
        grid, bed, wd = Grid(data["meta"]["local"]), decode(data["tex"]["locBed"]), decode(data["tex"]["locW"])
        from scipy import ndimage as ndi
        lab, _ = ndi.label(wd > 0)
        lake = ndi.binary_fill_holes(lab == np.bincount(lab.ravel())[1:].argmax() + 1)
        spacing = float(np.hypot(*np.subtract(grid.to_world(0, 1), grid.to_world(0, 0))))
        dist = ndi.distance_transform_edt(~lake) * spacing
        gz, gx = np.gradient(bed, spacing)
        band = (dist > 1.0) & (dist < 40.0) & (np.hypot(gx, gz) < 0.6) & (wd <= 0)
        blocked = np.zeros_like(band)
        for asset in ("grounds-walls", "field-boundaries", "gates", "hut-garden", "pitch", "tomb"):
            tris = payload_part(asset)[0]
            edge = np.linalg.norm(tris - np.roll(tris, 1, axis=1), axis=2).max(axis=1)
            for k in np.unique(np.clip(np.ceil(edge / (spacing / 2)), 1, 64)).astype(int):
                a, b = np.meshgrid(np.arange(k + 1), np.arange(k + 1))      # barycentric points every half texel
                keep = a + b <= k
                w1, w2 = a[keep] / k, b[keep] / k
                sel = tris[np.clip(np.ceil(edge / (spacing / 2)), 1, 64).astype(int) == k]
                pts = (sel[:, None, 0] * (1 - w1 - w2)[None, :, None] + sel[:, None, 1] * w1[None, :, None]
                       + sel[:, None, 2] * w2[None, :, None]).reshape(-1, 3)
                i, j = grid.to_texel(pts[:, 0], pts[:, 2])
                ok = (i >= 0) & (i < grid.ny - 0.5) & (j >= 0) & (j < grid.nx - 0.5)
                blocked[np.round(i[ok]).astype(int), np.round(j[ok]).astype(int)] = True
        free = band & ~ndi.binary_dilation(blocked, iterations=1)
        lab, _ = ndi.label(free, np.ones((3, 3), bool))
        ring = lab == np.bincount(lab.ravel())[1:].argmax() + 1
        inside = ndi.binary_fill_holes(ring) & ~ring
        self.assertGreater(float((lake & inside).sum()) / float(lake.sum()), 0.99)   # the walk closes round it


class HarbourMouthTests(unittest.TestCase):
    """The harbour mouth's terrain step (src/fix_harbour.py, harbour-local-cutout): found from the data at the cliff
    foot nearest White Hound's mapped mouth, a water-filled recess cut only inside its declared bounds, and nothing
    changed by a second run."""

    def mouth(self):
        """(record, waterline origin, unit axis into the cliff, unit across, lake level) of the cut in the data."""
        meta = world.data()["meta"]
        rec = meta["extras"]["harbour"]
        u = np.asarray(rec["axis"], np.float64)
        return rec, np.asarray(rec["mouth"], np.float64), u, np.array([-u[1], u[0]]), float(meta["lake"])

    def shell(self):
        tris, attr, normal = payload_part("harbour-mouth")
        return tris, attr, normal

    def local(self, pts):
        _, m, u, side, lake = self.mouth()
        rel = np.asarray(pts)[..., [0, 2]] - m
        return rel @ u, rel @ side, np.asarray(pts)[..., 1] - lake

    def test_the_channel_runs_square_into_the_cliff(self):
        """Its axis is square to the lake's edge beside the mouth, so the portal faces the lake across the cliff's line
        whatever stands on the cliff above it (the castle's walls hold the cliff top's level: castle_base.HOLD_M)."""
        from scipy import ndimage as ndi

        rec, m, u, _, _ = self.mouth()
        data = world.data()
        grid = Grid(data["meta"]["local"])
        e, n = grid.to_world(*np.mgrid[0:grid.ny, 0:grid.nx])
        wet = decode(data["tex"]["locW"]) > 0
        recess = shapely.contains_xy(Polygon(rec["bounds"]).buffer(2.0), e, n)       # cut since: left out
        shore = np.stack([e, n], -1)[wet & ~ndi.binary_erosion(wet) & (np.hypot(e - m[0], n - m[1]) < 15.0) & ~recess]
        along = np.linalg.svd(shore - shore.mean(axis=0), full_matrices=False)[2][0]
        self.assertLess(float(np.degrees(np.arcsin(abs(float(u @ along))))), 6.0)

    def test_the_shell_is_rock_and_ivy_inside_its_bounds_not_a_boathouse(self):
        tris, attr, _ = self.shell()
        kinds = {tuple(a) for a in np.unique(attr, axis=0)}
        self.assertEqual(kinds, {mesh_grounds.HARBOUR_ROCK, mesh_grounds.IVY})         # no timber, slate or glass
        rec = self.mouth()[0]
        c = tris.mean(axis=1)
        self.assertTrue(shapely.contains_xy(Polygon(rec["bounds"]).buffer(4.0), c[:, 0], c[:, 2]).all())

    def test_a_boat_passes_in_under_the_ivy_with_heads_bent(self):
        # "'Heads down!' yelled Hagrid as the first boats reached the cliff; they all bent their heads and the little
        # boats carried them through a curtain of ivy which hid a wide opening in the cliff face"
        tris, attr, normal = self.shell()
        rock = tris[np.all(attr == mesh_grounds.HARBOUR_ROCK, axis=1)]
        ivy = tris[np.all(attr == mesh_grounds.IVY, axis=1)]
        _, m, u, side, lake = self.mouth()
        for y in (0.3, 0.9):                                                       # a boat's gunwale and a bent head
            start = np.array([*(m - 3.0 * u)[:1], lake + y, *(m - 3.0 * u)[1:]])
            ray = np.array([u[0], 0.0, u[1]])
            self.assertFalse(world.segment_hits(start[None], ray[None], 10.0, rock)[0], y)   # open into the cliff
        a, w, h = self.local(rock.reshape(-1, 3))
        over = (np.abs(w) < 1.0) & (a > 1.0) & (a < 5.0) & (h > 0.0)
        headroom = float(h[over].min())
        self.assertTrue(1.6 < headroom < 2.3, headroom)                            # low: heads bent, not a boathouse
        a, w, h = self.local(ivy.reshape(-1, 3))
        hanging = np.abs(w) < 2.0
        self.assertTrue(0.7 < float(h[hanging].min()) < 1.3, h[hanging].min())     # the ivy hangs to head height
        self.assertGreater(float(np.ptp(w[hanging])), 2 * mesh_grounds.CHANNEL_HALF_M - 0.8)   # across the opening

    def test_the_channel_is_open_to_the_back_wall_across_its_width(self):
        # nothing of any asset stands in the boats' way, from outside the mouth to the back wall
        rec, m, u, side, lake = self.mouth()
        reach = 3.0 + rec["depth"] - mesh_grounds.BACK_IN_M - 0.3
        found = []
        for c in world.payload()["chunks"]:
            t = c["pos"][c["idx"]]
            solid = ~np.all(c["attr"][c["idx"][:, 0]] == mesh_grounds.IVY, axis=1)   # the boats pass through the ivy
            rel = t.mean(axis=1)[:, [0, 2]] - m
            near = (np.hypot(*rel.T) < rec["depth"] + 20.0) & solid
            if near.any():
                found.append((c["asset"], t[near]))
        self.assertTrue(any(a == "harbour-mouth" for a, _ in found))
        half = mesh_grounds.CHANNEL_HALF_M - 0.2
        starts, labels = [], []
        for w in np.linspace(-half, half, 9):
            for y in (0.2, 0.6, 1.0):                                              # gunwale to a bent head
                p = m - 3.0 * u + w * side
                starts.append([p[0], lake + y, p[1]])
                labels.append((round(float(w), 2), y))
        starts = np.array(starts)
        ray = np.tile([u[0], 0.0, u[1]], (len(starts), 1))
        for asset, tris in found:
            hits = world.segment_hits(starts, ray, reach, tris)
            self.assertFalse(hits.any(), (asset, [l for l, h in zip(labels, hits) if h][:5]))

    def test_the_mouth_rock_takes_the_terrain_cliffs_colour(self):
        # the cut is filled back to the cliff's surface in mesh: it must not read as a column of other stone
        # it calls the terrain's own rock function, so the two cannot drift apart
        import materials
        mat = next(m for m in materials.materials() if m.name == "harbour-rock")
        self.assertIsNotNone(mat.p3)
        self.assertRegex(mat.p3, r"\balb = rockAlb\(p, n\);")
        frag4 = (ROOT / "src" / "frag4.glsl").read_text(encoding="utf-8")
        self.assertIn("vec3 rockAlb(vec3 p, vec3 n) {", frag4)
        self.assertRegex(frag4.split("vec3 groundAlb(", 1)[1], r"mix\(a, rockAlb\(p, n\), rock \* 0\.9\)")

    def test_the_roof_is_shaded_smoothly_as_the_cliff(self):
        found = [c for c in world.chunks("harbour-mouth")]
        flat = []
        for c in found:
            rock = np.all(c["attr"][c["idx"][:, 0]] == mesh_grounds.HARBOUR_ROCK, axis=1)
            t = c["pos"][c["idx"]]
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            up = rock & (n[:, 1] > 0.3)
            vn = c["nrm"][c["idx"][up]]
            vn = vn / np.maximum(np.linalg.norm(vn, axis=2, keepdims=True), 1e-12)
            flat.append(np.all(np.einsum("tvk,tk->tv", vn, n[up]) > 0.9998, axis=1))
        flat = np.concatenate(flat)
        self.assertLess(float(flat.mean()), 0.5)                                   # not faceted, spike by spike

    def test_ivy_curtains_the_whole_opening_and_drapes_the_face(self):
        tris, attr, _ = self.shell()
        ivy = tris[np.all(attr == mesh_grounds.IVY, axis=1)].reshape(-1, 3)
        a, w, h = self.local(ivy)
        self.assertGreater(float(np.ptp(w)), 2 * mesh_grounds.CHANNEL_HALF_M + 1.0)   # wider than the opening
        self.assertGreater(float(h.max()), mesh_grounds.CROWN_M + 1.5)               # over the face above it
        hanging = np.abs(w) < mesh_grounds.CHANNEL_HALF_M - 0.3
        self.assertLess(float(h[hanging].min()), 1.0)                                # and low across the opening

    def test_the_recess_is_roofed_over_and_closed_at_the_back(self):
        tris, attr, _ = self.shell()
        rock = tris[np.all(attr == mesh_grounds.HARBOUR_ROCK, axis=1)]
        rec, m, u, side, lake = self.mouth()
        for a in (0.5, 3.0, 6.0, 9.0):                                             # no open slot seen from above,
            for w in (-2.5, 0.0, 2.5):                                             # from the waterline in
                p = m + a * u + w * side
                start = np.array([p[0], lake + 0.5, p[1]])
                self.assertTrue(world.segment_hits(start[None], np.array([[0.0, 1.0, 0.0]]), 60.0, rock)[0], (a, w))
                if a < 2.0:                                                        # the lintel: low over the water
                    continue
                above = np.array([p[0], lake + 80.0, p[1]])                        # the cliff's surface closes over it
                self.assertTrue(world.segment_hits(above[None], np.array([[0.0, -1.0, 0.0]]), 72.0, rock)[0], (a, w))
        start = np.array([*(m + 2.0 * u)[:1], lake + 0.5, *(m + 2.0 * u)[1:]])
        ray = np.array([[u[0], 0.0, u[1]]])
        self.assertTrue(world.segment_hits(start[None], ray, rec["depth"] + 2.0, rock)[0])   # a closed back, no tunnel
        self.assertFalse(world.segment_hits(start[None], ray, 6.0, rock)[0])                # visibly recessed: 8 m in

    def test_the_arch_stands_clear_of_the_channel_banks(self):
        # the cut is made texel by texel, so its banks ramp up inside the rectangle; the arch must span only water
        data = world.data()
        grid, bed = Grid(data["meta"]["local"]), decode(data["tex"]["locBed"])
        rec, m, u, side, lake = self.mouth()
        half = mesh_grounds.CHANNEL_HALF_M
        a, w = np.meshgrid(np.arange(mesh_grounds.PORTAL_M, rec["depth"] - mesh_grounds.BACK_IN_M, 0.5),
                           np.linspace(-half + 0.2, half - 0.2, 9), indexing="ij")
        p = m + a[..., None] * u + w[..., None] * side
        ground = sample(bed, grid, p[..., 0], p[..., 1]) - lake
        self.assertLess(float((ground - mesh_grounds.vault(w)).max()), -0.2)        # the vault clears the banks
        self.assertGreater(2 * half, 4.0)                                          # and the opening is still wide

    def synthetic(self, cliff=True):
        """A lake at the datum meeting a straight cliff 36 m high at x = 10 (or no cliff), on a rotated grid."""
        grid = world.rotated_grid(60)
        e, n = grid.to_world(*np.mgrid[0:grid.ny, 0:grid.nx])
        lake = world.datum()
        bed = np.where(e > 10.0, lake + (36.0 if cliff else 1.0), lake - 4.0)
        wd = np.where(bed < lake, lake - bed, -8.0)
        surface = np.where(wd > 0, 3, 1).astype(np.uint8)
        return {"lake": lake}, grid, bed, wd, surface, e, n

    def check_bounded_and_idempotent(self, meta, grid, bed, wd, surface, mapped):
        b2, w2, s2, changed, record = fix_harbour.fix_harbour(meta, grid, bed, wd, surface, mapped)
        self.assertGreater(changed, 0)
        e, n = grid.to_world(*np.mgrid[0:grid.ny, 0:grid.nx])
        inside = shapely.contains_xy(Polygon(record["bounds"]), e, n)
        moved = (b2 != bed) | (w2 != wd) | (s2 != surface)
        self.assertFalse((moved & ~inside).any())                                  # nothing outside the bounds
        lake = meta["lake"]
        self.assertTrue(np.all(w2[inside] > 0))                                     # a water-filled channel
        self.assertLessEqual(float(b2[inside].max()), lake - fix_harbour.CHANNEL_M + 0.011)
        again = fix_harbour.fix_harbour(meta, grid, b2, w2, s2, mapped, record)
        self.assertEqual(again[3], 0)                                               # idempotent
        for a, b in zip(again[:3], (b2, w2, s2)):
            self.assertTrue(np.array_equal(a, b))
        self.assertLessEqual(Polygon(record["bounds"]).area, 10.0 * 15.0)          # tightly bounded
        return b2, w2, record

    def test_a_synthetic_cliff_is_cut_where_the_mapped_mouth_points(self):
        meta, grid, bed, wd, surface, e, n = self.synthetic()
        b2, w2, record = self.check_bounded_and_idempotent(meta, grid, bed, wd, surface, np.array([-20.0, 5.0]))
        mouth, axis = np.asarray(record["mouth"]), np.asarray(record["axis"])
        self.assertLess(abs(mouth[0] - 10.0), 3.0)                                 # on the cliff face
        self.assertGreater(float(axis @ np.array([1.0, 0.0])), 0.95)              # running into it
        cut = (w2 > 0) & (wd <= 0)
        self.assertGreater(float(e[cut].max()), 10.0 + 8.0)                        # deep into the cliff

    def test_a_taller_bank_beside_the_mouth_does_not_turn_the_cut(self):
        # as where the castle's walls hold the cliff's top level beside the mouth: the bed's slope at the foot turns
        # towards the taller bank, the lake's edge not
        meta, grid, bed, wd, surface, e, n = self.synthetic()
        bed = np.where((e > 10.0) & (n < 5.0), meta["lake"] + 16.0, bed)
        _, _, record = self.check_bounded_and_idempotent(meta, grid, bed, wd, surface, np.array([-20.0, 5.0]))
        self.assertGreater(float(np.asarray(record["axis"]) @ np.array([1.0, 0.0])), 0.995)   # square to the shore

    def test_without_a_cliff_near_the_mapped_mouth_nothing_is_cut(self):
        meta, grid, bed, wd, surface, *_ = self.synthetic(cliff=False)
        with self.assertRaises(ValueError):
            fix_harbour.fix_harbour(meta, grid, bed, wd, surface, np.array([-20.0, 5.0]))

    def test_the_registered_cut_is_the_one_found_from_the_mapped_mouth(self):
        from datum_shift import shifted
        from mesh_source import load_data
        path = Path(os.environ.get("HOG_INPUT_DATA_BEFORE_RAILWAY_X") or ROOT / "build" / "inputs" / "data-before-railway-x.js")
        if not path.is_file():
            raise unittest.SkipTest(f"blocked: S0's input {path} is missing: run python -X utf8 -m tools.site.build x")
        before = load_data(path)                                                   # as S0 starts, before the cut
        before = shifted(before, world.SHIFT_M) if world.SHIFT_M else before
        meta, grid = before["meta"], Grid(before["meta"]["local"])
        bed, wd = decode(before["tex"]["locBed"]), decode(before["tex"]["locW"])
        surface = decode(before["tex"]["locN"])[..., 0]
        mapped = fix_harbour.mapped_mouth(world.scene(), load_georef())
        _, _, record = self.check_bounded_and_idempotent(meta, grid, bed, wd, surface, mapped)
        self.assertEqual(record, world.data()["meta"]["extras"]["harbour"])

    def test_the_registered_cliff_is_recessed_at_the_mapped_mouth(self):
        data = world.data()
        meta, grid = data["meta"], Grid(data["meta"]["local"])
        bed, wd, surface = decode(data["tex"]["locBed"]), decode(data["tex"]["locW"]), decode(data["tex"]["locN"])[..., 0]
        mapped = fix_harbour.mapped_mouth(world.scene(), load_georef())
        rec = meta["extras"]["harbour"]
        again = fix_harbour.fix_harbour(meta, grid, bed, wd, surface, mapped, rec)
        self.assertEqual(again[3], 0)                                               # the data holds the cut
        for a, b in zip(again[:3], (bed, wd, surface)):
            self.assertTrue(np.array_equal(a, b))
        mouth, axis = np.asarray(rec["mouth"]), np.asarray(rec["axis"])
        self.assertLess(float(np.hypot(*(mouth - mapped))), fix_harbour.SEARCH_M)
        g = lambda p, f: float(sample(f, grid, np.array([p[0]]), np.array([p[1]]))[0])
        back = mouth + axis * (fix_harbour.DEPTH_M + 4.0)
        self.assertGreater(g(back, bed) - meta["lake"], fix_harbour.CLIFF_M)       # a recess in a cliff, not a bay
        self.assertGreater(g(mouth - axis * (fix_harbour.APRON_M + 3.0), wd), 0.0)   # opening onto the lake
        self.assertGreater(g(mouth + axis * 8.0, wd), 0.0)                          # water well inside it


class BridgesTests(unittest.TestCase):
    """Bridges where X's roads and paths cross its burns (bridges): wherever a road runs between two pieces of a
    stream, a small stone bridge carries it across: coped parapets either side of the road, leaving the road clear,
    over an arched opening in each face for the water ("the stream probably does pass under the High Street")."""

    @classmethod
    def setUpClass(cls):
        from scipy import ndimage as ndi
        from scipy.spatial import cKDTree
        cls.tris, cls.attr, cls.normal = payload_part("bridges")
        data = world.data()
        cls.grid, cls.bed = Grid(data["meta"]["local"]), decode(data["tex"]["locBed"])
        road = decode(data["tex"]["locA"])[..., 0] / 255.0
        stream = mesh_source.sd_bytes(decode(data["tex"]["locB"])[..., 1])
        e, n = cls.grid.to_world(*np.mgrid[0:cls.grid.ny, 0:cls.grid.nx])
        lab, k = ndi.label(stream < 0, np.ones((3, 3), bool))
        pieces = [np.column_stack([e[lab == c], n[lab == c]]) for c in range(1, k + 1)]
        cls.road, cls.crossings = np.column_stack([e[road > 0.5], n[road > 0.5]]), []
        for a in range(k):
            for b in range(a + 1, k):
                dist, at = cKDTree(pieces[b]).query(pieces[a])
                j = int(np.argmin(dist))
                if dist[j] > 25.0:
                    continue
                p, q = pieces[a][j], pieces[b][at[j]]
                line = p + np.linspace(0.0, 1.0, 60)[:, None] * (q - p)
                on = sample(road, cls.grid, line[:, 0], line[:, 1]) >= 0.5
                if on.any():                                                       # a road between the two pieces
                    cls.crossings.append(line[on].mean(axis=0))

    def road_axis(self, c):
        near = self.road[np.hypot(*(self.road - c).T) < 8.0]
        return np.linalg.svd(near - near.mean(axis=0))[2][0]

    def test_a_bridge_at_every_crossing_and_nowhere_else(self):
        self.assertGreater(len(self.crossings), 4)
        centre = self.tris.mean(axis=1)[:, [0, 2]]
        nearest = np.min([np.hypot(*(centre - c).T) for c in self.crossings], axis=0)
        self.assertLess(float(nearest.max()), mesh_grounds.CROSSING_GAP_M)        # nothing away from a crossing
        for c in self.crossings:
            with self.subTest(at=np.round(c).tolist()):
                self.assertGreater(int((np.hypot(*(centre - c).T) < 8.0).sum()), 20)

    def near(self, c, reach=30.0):
        return np.hypot(*(self.tris.mean(axis=1)[:, [0, 2]] - c).T) < reach

    def of(self, name, up=False):
        keep = np.all(self.attr == code(name), axis=1)
        return keep & (self.normal[:, 1] > 0.8) if up else keep

    def deck_level(self, c):
        """The walkable level at c: the bridge's deck where it has one over the road, else the ground."""
        deck = self.tris[self.of("bridge-deck", up=True) & self.near(c, 12.0)]
        hit = [t for t in deck if Polygon(t[:, [0, 2]]).buffer(1e-6).contains(Point(*c))]
        ground = float(sample(self.bed, self.grid, c[:1], c[1:])[0])
        return max(ground, float(np.mean([t[:, 1].mean() for t in hit]))) if hit else ground

    def test_each_bridge_is_one_rubble_stone_structure_carrying_the_road(self):
        kinds = {tuple(a) for a in np.unique(self.attr, axis=0)}
        self.assertEqual(kinds, {code("grounds-rubble"), code("pale-stone"), code("arch-shadow"), code("bridge-deck")})
        for c in self.crossings:
            with self.subTest(at=np.round(c).tolist()):
                plan = shapely.union_all([shapely.MultiPoint(t[:, [0, 2]]).convex_hull.buffer(0.05)
                                          for t in self.tris[self.near(c)]])
                self.assertEqual(len(getattr(plan, "geoms", [plan])), 1)            # one structure, not pieces
                self.assertTrue(plan.contains(Point(*c)))                          # over the road

    def test_coped_parapets_either_side_leave_the_road_clear(self):
        stone = self.of("grounds-rubble") | self.of("pale-stone")
        for c in self.crossings:
            with self.subTest(at=np.round(c).tolist()):
                ax = self.road_axis(c)
                side = np.array([-ax[1], ax[0]])
                mine = stone & self.near(c, 12.0)
                pts = self.tris[mine & self.of("pale-stone", up=True)].reshape(-1, 3)
                level = self.deck_level(c)
                over = np.abs((pts[:, [0, 2]] - c) @ ax) < 3.0
                self.assertTrue(0.8 < float(np.median(pts[over, 1] - level)) < 1.4, np.median(pts[over, 1] - level))
                across = (pts[:, [0, 2]] - c) @ side
                self.assertTrue((across > 0.5).any() and (across < -0.5).any())     # on both sides of the road
                tall = self.tris[mine].reshape(-1, 3)
                lane = np.abs((tall[:, [0, 2]] - c) @ side) < max(0.6, float(np.abs(across).min()) - 0.4)
                deck = self.tris[self.of("bridge-deck", up=True) & self.near(c)].reshape(-1, 3)
                da = np.round((deck[:, [0, 2]] - c) @ ax, 1)
                cols = np.unique(da)
                walk = np.interp((tall[:, [0, 2]] - c) @ ax, cols, [deck[da == v, 1].max() for v in cols])
                walk = np.maximum(walk, sample(self.bed, self.grid, tall[:, 0], tall[:, 2]))
                self.assertFalse(np.any(lane & (tall[:, 1] > walk + 0.3)))          # the road itself kept clear

    def test_the_arch_stands_in_each_face_beneath_its_parapet(self):
        dark = self.of("arch-shadow")
        for c in self.crossings:
            ax = self.road_axis(c)
            side = np.array([-ax[1], ax[0]])
            cap = self.tris[self.near(c) & self.of("pale-stone", up=True)].reshape(-1, 3)
            capA, capS = (cap[:, [0, 2]] - c) @ ax, (cap[:, [0, 2]] - c) @ side
            void = self.tris[self.near(c) & dark].reshape(-1, 3)
            vA, vS = (void[:, [0, 2]] - c) @ ax, (void[:, [0, 2]] - c) @ side
            for s in (1.0, -1.0):
                with self.subTest(at=np.round(c).tolist(), side=s):
                    mine, rail = vS * s > 0, capS * s > 0
                    self.assertTrue(mine.any() and rail.any())
                    self.assertLess(abs(float(np.max(s * vS[mine]) - np.max(s * capS[rail]))), 0.6)   # in the face
                    self.assertTrue(capA[rail].min() < vA[mine].min() and vA[mine].max() < capA[rail].max())

    def test_each_arch_rises_beneath_its_coping_not_through_it(self):
        ring = self.of("pale-stone") & (np.abs(self.normal[:, 1]) < 0.2)
        for c in self.crossings:
            ax = self.road_axis(c)
            side = np.array([-ax[1], ax[0]])
            cap = self.tris[self.near(c) & self.of("pale-stone") & (self.normal[:, 1] < -0.8)].reshape(-1, 3)   # undersides
            arc = self.tris[self.near(c) & ring]
            for s in (1.0, -1.0):
                with self.subTest(at=np.round(c).tolist(), side=s):
                    rail = cap[((cap[:, [0, 2]] - c) @ side) * s > 0]
                    ca = (rail[:, [0, 2]] - c) @ ax
                    level = lambda p: np.interp((p[..., [0, 2]] - c) @ ax, *zip(*sorted(zip(ca, rail[:, 1]))))
                    mine = arc[(((arc.mean(axis=1)[:, [0, 2]] - c) @ side) * s > 0)
                               & ((arc[..., 1] - level(arc)).min(axis=1) < -0.05)]   # the ring, not the coping's edge
                    self.assertGreater(len(mine), 0)
                    self.assertGreater(float((level(mine) - mine[..., 1]).min()), 0.5)   # with stone over the ring

    def test_each_face_stands_where_its_burn_leaves_the_road(self):
        # not at the road's edge with the burn coming out of the grass metres off: where the burn's water ends within
        # 7.5 m of the road's edge (Hogsmeade's crossing on the tour ends 7 m off), the face stands at it
        reach = 7.5
        data = world.data()
        stream = mesh_source.sd_bytes(decode(data["tex"]["locB"])[..., 1])
        road = decode(data["tex"]["locA"])[..., 0] / 255.0
        e, n = self.grid.to_world(*np.mgrid[0:self.grid.ny, 0:self.grid.nx])
        wet = np.column_stack([e[stream < 0], n[stream < 0]])
        dark = self.of("arch-shadow")
        checked = 0
        for c in self.crossings:
            ax = self.road_axis(c)
            side = np.array([-ax[1], ax[0]])
            void = self.tris[self.near(c) & dark].reshape(-1, 3)
            vS = (void[:, [0, 2]] - c) @ side
            near = wet[np.hypot(*(wet - c).T) < 30.0]
            wa, ws = (near - c) @ ax, (near - c) @ side
            for s in (1.0, -1.0):
                zs = np.arange(0.0, 8.0, 0.25)
                ink = sample(road, self.grid, *(c + s * zs[:, None] * side).T) >= 0.5
                edge = float(zs[np.argmax(~ink)]) if (~ink).any() else 8.0
                mine = (ws * s > 0) & (np.abs(wa) < 6.0)
                if not mine.any():
                    continue
                end = float(np.min(np.abs(ws[mine])))
                if end > edge + reach:
                    continue
                checked += 1
                with self.subTest(at=np.round(c).tolist(), side=s):
                    self.assertGreater(float(np.max(s * vS[vS * s > 0])), end - 1.0)
        self.assertGreater(checked, len(self.crossings))

    def test_each_arch_spans_its_burn(self):
        data = world.data()
        stream = mesh_source.sd_bytes(decode(data["tex"]["locB"])[..., 1])
        dark = self.of("arch-shadow")
        checked = 0
        for c in self.crossings:
            ax = self.road_axis(c)
            side = np.array([-ax[1], ax[0]])
            void = self.tris[self.near(c) & dark].reshape(-1, 3)
            vA, vS = (void[:, [0, 2]] - c) @ ax, (void[:, [0, 2]] - c) @ side
            for s in (1.0, -1.0):
                mine = vS * s > 0
                outer = float(np.max(s * vS[mine]))
                xs = np.arange(-10.0, 10.0, 0.25)
                wet = np.zeros_like(xs, bool)
                for off in (0.75, 1.5):
                    pts = c + xs[:, None] * ax + s * (outer + off) * side
                    wet |= sample(stream, self.grid, pts[:, 0], pts[:, 1]) < 0
                if not wet.any():
                    continue
                checked += 1
                with self.subTest(at=np.round(c).tolist(), side=s):            # the water comes out of the arch
                    self.assertTrue(vA[mine].min() - 0.5 <= xs[wet].min() and xs[wet].max() <= vA[mine].max() + 0.5,
                                    (xs[wet].min(), xs[wet].max(), vA[mine].min(), vA[mine].max()))
        self.assertGreater(checked, len(self.crossings))

    def test_the_parapets_run_unbroken_either_side(self):
        # alike either side in the open; in a street each side runs between its own houses
        bsd = self.bsd()
        for c in self.crossings:
            ax = self.road_axis(c)
            side = np.array([-ax[1], ax[0]])
            X, Z = np.meshgrid(np.arange(-16.0, 16.1, 0.5), np.arange(-10.0, 10.1, 0.5), indexing="ij")
            P = c + X[..., None] * ax + Z[..., None] * side
            street = sample(bsd, self.grid, P[..., 0].ravel(), P[..., 1].ravel()).min() <= mesh_grounds.STREET_M
            cap = self.tris[self.near(c) & self.of("pale-stone", up=True)]
            s_ = (cap.mean(axis=1)[:, [0, 2]] - c) @ side
            reach = []
            for s in (1.0, -1.0):
                with self.subTest(at=np.round(c).tolist(), side=s):
                    plan = shapely.union_all([Polygon(t[:, [0, 2]]).buffer(0.02) for t in cap[s_ * s > 0]])
                    self.assertEqual(len(getattr(plan, "geoms", [plan])), 1)          # no gaps, no offsets
                    a = (cap[s_ * s > 0].reshape(-1, 3)[:, [0, 2]] - c) @ ax
                    reach.append((float(a.min()), float(a.max())))
            if not street:
                self.assertLess(float(np.abs(np.subtract(*reach)).max()), 0.5)       # the two sides alike

    def test_a_raised_deck_ramps_gently_down_to_the_road(self):
        deck = self.of("bridge-deck", up=True)
        for c in self.crossings:
            with self.subTest(at=np.round(c).tolist()):
                ax = self.road_axis(c)
                side = np.array([-ax[1], ax[0]])
                pts = self.tris[deck & self.near(c)].reshape(-1, 3)
                self.assertGreater(len(pts), 0)
                a = np.round((pts[:, [0, 2]] - c) @ ax, 1)
                order = np.unique(a)
                lift = pts[:, 1] - sample(self.bed, self.grid, pts[:, 0], pts[:, 2])
                w = np.abs((pts[:, [0, 2]] - c) @ side)
                rise = np.array([lift[a == v][np.argmin(w[a == v])] for v in order])   # along the road's middle
                self.assertLess(float(np.max(np.abs(np.diff(rise) / np.diff(order)))), 0.2)   # no steeper than 1 in 5
                for v in (a.min(), a.max()):
                    end = pts[a == v]
                    ground = sample(self.bed, self.grid, end[:, 0], end[:, 2])
                    self.assertLess(float(np.max(np.abs(end[:, 1] - ground))), 0.3)   # meeting the road at its ends

    def test_a_raised_deck_is_walled_both_sides(self):
        # wherever the deck stands more than 0.3 m over the ground there is coping on both sides of it; in a street
        # a house front across a side, where its parapet stops short of the house, closes that side
        deck = self.tris[self.of("bridge-deck", up=True)].reshape(-1, 3)
        cap = self.tris[self.of("pale-stone", up=True)].mean(axis=1)
        bsd = self.bsd()
        for c in self.crossings:
            ax = self.road_axis(c)
            side = np.array([-ax[1], ax[0]])
            pts = deck[np.hypot(*(deck[:, [0, 2]] - c).T) < 30.0]
            lift = pts[:, 1] - sample(self.bed, self.grid, pts[:, 0], pts[:, 2])
            street = (sample(bsd, self.grid, pts[:, 0], pts[:, 2]) < mesh_grounds.STREET_M).any()
            a = np.round((pts[:, [0, 2]] - c) @ ax, 1)
            mine = cap[np.hypot(*(cap[:, [0, 2]] - c).T) < 30.0]
            ca, cs = (mine[:, [0, 2]] - c) @ ax, (mine[:, [0, 2]] - c) @ side
            across = np.arange(0.0, mesh_grounds.DECK_HALF_M[1] + mesh_grounds.FACE_REACH_M + 0.01, 0.25)  # to the faces
            for v in np.unique(a[lift > 0.3]):
                with self.subTest(at=np.round(c).tolist(), station=float(v)):
                    near = np.abs(ca - v) < 0.6
                    for s in (1.0, -1.0):                                          # (the wall stops 0.5 m short)
                        q = (c + v * ax + s * across[:, None] * side)[None] + np.array([-1.0, 0.0, 1.0])[:, None, None] * ax
                        q = q.reshape(-1, 2)
                        housed = street and sample(bsd, self.grid, q[:, 0], q[:, 1]).min() < 0.0   # its front closes it
                        self.assertTrue((near & (cs * s > 0)).any() or housed, s)

    def test_an_arched_opening_in_each_face_lets_the_water_through(self):
        dark = self.of("arch-shadow")
        for c in self.crossings:
            with self.subTest(at=np.round(c).tolist()):
                ax = self.road_axis(c)
                side = np.array([-ax[1], ax[0]])
                mine = dark & self.near(c, mesh_grounds.CROSSING_GAP_M)
                facing = self.normal[mine][:, [0, 2]] @ side
                self.assertTrue((facing > 0.9).any() and (facing < -0.9).any())     # both faces, across the road
                t = self.tris[mine]
                area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
                centre = t.mean(axis=1)
                above = centre[:, 1] > sample(self.bed, self.grid, centre[:, 0], centre[:, 2])
                self.assertGreater(float(area[above].sum()), 0.4 * float(area.sum()))   # open, not buried

    def test_the_parapets_keep_off_the_road_ink(self):
        # except where the deck stands proud, which must be walled (a path meeting a raised ramp meets its parapet)
        data = world.data()
        road = decode(data["tex"]["locA"])[..., 0] / 255.0
        tops = self.tris[(self.of("grounds-rubble") | self.of("pale-stone")) & (self.normal[:, 1] > 0.8)].mean(axis=1)
        deck = self.tris[self.of("bridge-deck", up=True)].reshape(-1, 3)
        lift = deck[:, 1] - sample(self.bed, self.grid, deck[:, 0], deck[:, 2])
        from scipy.spatial import cKDTree
        raised = lift[cKDTree(deck[:, [0, 2]]).query(tops[:, [0, 2]])[1]] > 0.3
        ink = sample(road, self.grid, tops[:, 0], tops[:, 2])
        self.assertLess(float(((ink >= 0.65) & ~raised).mean()), 0.02)            # no parapet across a road or path

    @staticmethod
    def synthetic_bridge(extra, ground=None):
        """The copings' plan of a bridge on a synthetic road along x (|z| < 2.5) over a burn 3 m below at x = 0 (or on
        `ground(x, z)`), with `extra(e, n)` more road ink, and that ink."""
        grid = world.rotated_grid(40)
        e, n = grid.to_world(*np.mgrid[0:grid.ny, 0:grid.nx])
        more = extra(e, n).astype(np.float64)
        locA = np.zeros((grid.ny, grid.nx, len(mesh_source.CHANNELS["locA"])), np.uint8)
        locA[..., mesh_source.CHANNELS["locA"].index("road")] = 255 * np.maximum(np.abs(n) < 2.5, more)
        ground = ground or (lambda x, z: np.where(np.abs(np.asarray(z)) > 6.0, -3.0, 0.0) + 0.0 * np.asarray(x))
        scene = types.SimpleNamespace(locA=locA, local=grid, ground=ground, sunk=lambda x, z: ground(x, z) - 0.5,
                                      meta={"extras": {}})
        mesh = mesh_geometry.concat(mesh_grounds.bridge(scene, np.zeros(2), np.array([1.0, 0.0]), 2.5,
                                                        [np.array([0.0, 7.0]), np.array([0.0, -7.0])]))
        tris = mesh.pos[mesh.idx]
        attr = mesh.attr[mesh.idx[:, 0]]
        normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        up = normal[:, 1] > 0.8 * np.maximum(np.linalg.norm(normal, axis=1), 1e-12)
        return tris[np.all(attr == code("pale-stone"), axis=1) & up], (grid, more)

    def test_a_parapet_stops_where_a_path_leaves_the_road(self):
        # a path leaving northwards (+z) from x = 1.8 m, within the walls' reach over the arch's abutment
        cap, (grid, branch) = self.synthetic_bridge(lambda e, n: (e > 1.8) & (e < 5.8) & (n > 2.5))
        for s in (1.0, -1.0):
            plan = shapely.union_all([Polygon(t[:, [0, 2]]).buffer(0.01) for t in cap[cap.mean(axis=1)[:, 2] * s > 0]])
            self.assertEqual(len(getattr(plan, "geoms", [plan])), 1)                 # a parapet each side, unbroken
            x0, z0, x1, z1 = plan.bounds
            gx, gz = np.meshgrid(np.arange(x0, x1, 0.1), np.arange(z0, z1, 0.1))
            inside = shapely.contains_xy(plan, gx, gz)
            on_path = sample(branch, grid, gx[inside], gz[inside]) >= 0.65
            self.assertFalse(on_path.any(), np.column_stack([gx[inside], gz[inside]])[on_path][:3])   # off the path
        self.assertLess(float(cap[..., 0].max()), 1.8)                             # both walls stop short of it

    def test_the_roads_ragged_edge_is_no_path(self):
        # a bump on the road's edge, reaching out to the faces (7 m) but not past them, leaves the walls their full reach
        full, _ = self.synthetic_bridge(lambda e, n: np.zeros_like(e, bool))
        cap, _ = self.synthetic_bridge(lambda e, n: (e > 1.6) & (e < 2.6) & (n > 2.5) & (n < 7.4))
        self.assertAlmostEqual(float(cap[..., 0].max()), float(full[..., 0].max()), delta=0.05)

    def test_a_path_meeting_a_raised_deck_meets_its_parapet(self):
        # where the road lies level with the burn's banks the deck humps over the arches and ramps down at 1 in
        # RAMP_GRADE: a path leaving it while it stands proud does not open the parapet over the drop
        level = lambda x, z: 0.0 * np.asarray(x) + 0.0 * np.asarray(z)
        path = lambda e, n: (e > 1.8) & (e < 5.8) & (n > 2.5)
        full, _ = self.synthetic_bridge(lambda e, n: np.zeros_like(e, bool), level)
        cap, _ = self.synthetic_bridge(path, level)
        self.assertGreater(float(full[..., 0].max()), 4.0)                         # raised well past the path
        self.assertAlmostEqual(float(cap[..., 0].max()), float(full[..., 0].max()), delta=0.05)

    def bsd(self):
        data = world.data()
        return mesh_source.sd_bytes(decode(data["tex"]["locA"])[..., mesh_source.CHANNELS["locA"].index("bsd")])

    def test_no_bridge_stands_in_a_house_or_on_its_footway(self):
        # in Hogsmeade the burns pass under the High Street between its houses: the bridges stand in the gaps, clear of
        # the house fronts and their doors, and off the flagstone footway before them
        bsd = self.bsd()
        deck = self.of("bridge-deck")
        pts = self.tris[~deck].reshape(-1, 3)
        clear = sample(bsd, self.grid, pts[:, 0], pts[:, 2])
        self.assertGreater(float(clear.min()), 0.5, np.round(pts[np.argmin(clear)], 1))   # walls clear of the fronts
        lay = self.tris[deck].reshape(-1, 3)
        door = sample(bsd, self.grid, lay[:, 0], lay[:, 2]) < 1.0                 # lane B's doorstep (FOOTWAY "door")
        lift = lay[door, 1] - sample(self.bed, self.grid, lay[door, 0], lay[door, 2])
        self.assertLess(float(lift.max(initial=0.0)), 0.2)                         # and the doorsteps: a kerb, no more
        flags, fattr, fnormal = payload_part("street-furniture")
        flags = flags[np.all(fattr == code("flagstone"), axis=1) & (fnormal[:, 1] > 0.5)]
        way = shapely.union_all([Polygon(t[:, [0, 2]]) for t in flags if Polygon(t[:, [0, 2]]).area > 1e-6])
        self.assertGreater(way.area, 100.0)
        for c in self.crossings:
            walls = self.tris[self.near(c) & ~deck]
            plan = shapely.union_all([Polygon(t[:, [0, 2]]).buffer(0.02) for t in walls])
            under = self.tris[self.near(c) & deck].reshape(-1, 3)
            on = shapely.contains_xy(way, under[:, 0], under[:, 2])
            lift = under[on, 1] - sample(self.bed, self.grid, under[on, 0], under[on, 2])
            with self.subTest(at=np.round(c).tolist()):
                self.assertLess(plan.intersection(way).area, 0.5)                  # no wall through the flags
                self.assertLess(float(lift.max(initial=0.0)), 0.1)                 # nor deck over them: under the kerb

    def test_a_street_bridge_humps_low_over_its_burn(self):
        # where houses line the road, the burn passes under the street on its lowest arch, the setts on its ring: a
        # low hump, not a humpbacked bridge rising before the houses' doors
        bsd = self.bsd()
        streets = 0
        for c in self.crossings:
            ax = self.road_axis(c)
            side = np.array([-ax[1], ax[0]])
            X, Z = np.meshgrid(np.arange(-16.0, 16.1, 0.5), np.arange(-10.0, 10.1, 0.5), indexing="ij")
            P = c + X[..., None] * ax + Z[..., None] * side
            if sample(bsd, self.grid, P[..., 0].ravel(), P[..., 1].ravel()).min() > 4.0:
                continue
            streets += 1
            deck = self.tris[self.near(c, 20.0) & self.of("bridge-deck", up=True)].reshape(-1, 3)
            above = deck[:, 1] - sample(self.bed, self.grid, deck[:, 0], deck[:, 2])
            on_road = np.abs((deck[:, [0, 2]] - c) @ side) < 2.0
            with self.subTest(at=np.round(c).tolist()):
                self.assertLess(float(above[on_road].max()), 0.75)
        self.assertGreaterEqual(streets, 1)                                        # the High Street's, at least


class ArchOpeningTests(unittest.TestCase):
    """The opening beneath a bridge's arch is one simple outline, so the dark beneath it can be triangulated wherever
    the burn's bed meets the arch's springing: the grounds failed to build 140 m down, at the bridges near (-256, 314)
    and (377, -9), when the arch's computed ends missed their springing by an ulp and crossed the bed lying there."""

    def test_the_arch_meets_a_bed_lying_at_its_springing(self):
        x0, half, rise = -1.25, 1.25, 0.55                                 # the arch near (-256, 314)
        radius = (half ** 2 + rise ** 2) / (2 * rise)
        xs = np.linspace(x0 - half, x0 + half, 9)
        drop = np.array([0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 1.0, 0.97, 0.73])  # its bed beneath the springing, along the face
        for end, below in (("left", drop), ("right", drop[::-1])):
            crossed = []
            for spring in 114.43250644458794 + 0.0137 * np.arange(50):   # its springing 140 m down, and more levels
                opening = mesh_grounds.arch_opening(x0, half, spring, radius, spring + rise - radius, (xs, spring - below))
                if not opening.is_valid:
                    crossed.append((float(spring), shapely.is_valid_reason(opening)))
            with self.subTest(bed_at_springing=end):
                self.assertEqual(crossed, [])


class GroundsStoneTests(unittest.TestCase):
    """The grounds' walls, gate pillars and bridges are rubble stone in rough courses (the village's rubble, so one
    shading branch), not the smooth shared boundary-wall stone, which read as render or concrete close up."""

    def test_the_stonework_is_coursed_rubble(self):
        import materials
        mats = {m.name: m for m in materials.materials()}
        self.assertEqual(mats["grounds-rubble"].p3, mats["rubble"].p3)            # the village's body, shared
        for asset in ("grounds-walls", "gates", "bridges"):
            with self.subTest(asset=asset):
                attr = payload_part(asset)[1]
                self.assertFalse(np.all(attr == code("wall-stone"), axis=1).any())
                self.assertTrue(np.all(attr == code("grounds-rubble"), axis=1).any())

    def test_the_gate_pillars_shafts_are_rubble_under_pale_caps(self):
        tris, attr, normal = payload_part("gates")
        side = np.abs(normal[:, 1]) < 0.2
        rubble, pale = np.all(attr == code("grounds-rubble"), axis=1), np.all(attr == code("pale-stone"), axis=1)
        gates = GatesTests()
        GatesTests.setUpClass()
        for c, foot, top in gates.pillars():
            mine = np.max(np.abs(tris.mean(axis=1)[:, [0, 2]] - c), axis=1) < mesh_grounds.PIER_M / 2 + 0.15
            with self.subTest(pillar=np.round(c, 1).tolist()):
                self.assertGreater(float(tris[mine & side & rubble][..., 1].max()), foot + mesh_grounds.PIER_H_M - 0.5)
                low_pale = tris[mine & side & pale][..., 1].min(axis=1)
                shaft = (low_pale > foot + mesh_grounds.WALL_HEIGHT_M + 0.3) & (low_pale < foot + mesh_grounds.PIER_H_M - 0.1)
                self.assertFalse(shaft.any())                                     # pale only at the cornice, over the walls


class GroundsMeshOracleTests(unittest.TestCase):
    """trimesh, as an independent oracle on the decoded payload."""

    def test_the_decoded_meshes_have_only_the_known_faults(self):
        self.assertEqual(world.mesh_faults("grounds"), MESH_FAULTS)


if __name__ == "__main__":
    unittest.main()
