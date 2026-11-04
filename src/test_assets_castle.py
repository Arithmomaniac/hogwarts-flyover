"""The castle family: real openings, open air in front of every pane, the doors, the pend, the Head's tower, the court
houses and the skyline."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import shapely
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assets_testing as world  # noqa: E402
import mesh_castle  # noqa: E402
import mesh_openings  # noqa: E402
from mesh_source import CASTLE_CLASSES, Grid, castle_class_sd, castle_piece_labels, decode, sample  # noqa: E402
from assets_testing import surface_heights, segment_hits  # noqa: E402

# the faults trimesh finds in this family's decoded meshes today (assets_testing.mesh_faults), pinned: a new fault
# fails, and so does a fix until this is updated
MESH_FAULTS = {"winding": {}, "collapsed": {}}


class FacadeTests(unittest.TestCase):
    """Walls with deep round-arched openings (castle-norman-openings): holes, reveals and fills behind them."""
    WALL, REVEAL, GLASS = (4, 21), (4, 43), (4, 40)
    WINDOW = dict(sill=101.0, width=1.0, height=2.2, depth=0.7, fill=(4, 40))

    def build(self, openings, **kw):
        # a 10 m wall from x = 0 to 10 along z = 0 between y = 100 and 108; it faces south (-z), right of its direction
        return mesh_openings.facade((0.0, 0.0), (10.0, 0.0), 100.0, 108.0, openings, self.WALL, reveal=self.REVEAL, **kw)

    def faces(self, mesh, attr):
        first = mesh.idx[:, 0]
        return mesh.idx[np.all(mesh.attr[first] == attr, axis=1)]

    def area(self, mesh, attr):
        a, b, c = (mesh.pos[self.faces(mesh, attr)[:, k]] for k in range(3))
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()

    def test_openings_cut_the_wall_and_are_closed_from_behind(self):
        windows = [mesh_openings.Opening(u=3.0, **self.WINDOW), mesh_openings.Opening(u=6.5, **self.WINDOW)]
        mesh, used = self.build(windows)
        self.assertEqual(used, windows)
        wall = self.faces(mesh, self.WALL)
        hole = Polygon(mesh_openings.outline(windows[0]))
        self.assertAlmostEqual(self.area(mesh, self.WALL), 80.0 - 2 * hole.area, places=6)
        corners = mesh.pos[wall][:, :, [0, 1]]                           # (u, y): the wall runs along x
        for o in windows:
            centre = Point(o.u, o.sill + o.height / 2)
            self.assertFalse(any(Polygon(t).contains(centre) for t in corners))
        np.testing.assert_allclose(mesh.nrm[wall.ravel()], np.tile([0.0, 0.0, -1.0], (wall.size, 1)), atol=1e-9)
        glass = self.faces(mesh, self.GLASS)
        np.testing.assert_allclose(mesh.pos[glass.ravel(), 2], 0.7)     # the depth behind the wall plane
        np.testing.assert_allclose(mesh.nrm[glass.ravel()], np.tile([0.0, 0.0, -1.0], (glass.size, 1)), atol=1e-9)
        self.assertAlmostEqual(self.area(mesh, self.GLASS), 2 * hole.area, places=6)

    def test_reveals_line_each_opening_and_face_into_it(self):
        window = mesh_openings.Opening(u=5.0, **self.WINDOW)
        mesh, _ = self.build([window])
        reveal = self.faces(mesh, self.REVEAL)
        self.assertEqual(len(reveal), 2 * len(mesh_openings.outline(window)))
        axis = np.array([window.u, window.sill + window.height - window.width / 2])
        for tri in reveal:
            p, n = mesh.pos[tri], mesh.nrm[tri[0]]
            self.assertAlmostEqual(n[2], 0.0)
            self.assertLessEqual(p[:, 2].max(), 0.7 + 1e-9)
            self.assertGreaterEqual(p[:, 2].min(), -1e-9)
            centre = p.mean(axis=0)
            towards = np.array([axis[0], max(min(centre[1], axis[1]), window.sill + 0.5)]) - centre[[0, 1]]
            self.assertGreater(n[[0, 1]] @ towards, 0.0)
            self.assertGreater(np.cross(p[1] - p[0], p[2] - p[0]) @ n, 0.0)

    def test_the_head_is_a_round_arch(self):
        window = mesh_openings.Opening(u=5.0, **self.WINDOW)
        ring = mesh_openings.outline(window)
        spring = window.sill + window.height - window.width / 2
        head = ring[ring[:, 1] >= spring - 1e-9]
        np.testing.assert_allclose(np.hypot(head[:, 0] - window.u, head[:, 1] - spring), window.width / 2)
        top = np.sort(ring[:, 1])[-2:]
        self.assertAlmostEqual(top[0], top[1])                          # a level crown facet, not a Gothic point
        self.assertLessEqual(top[1], window.sill + window.height)
        self.assertGreater(top[1], window.sill + window.height - 0.05)

    def test_openings_that_do_not_fit_are_left_out(self):
        fits = mesh_openings.Opening(u=3.0, **self.WINDOW)
        too_high = fits._replace(u=6.0, sill=106.0)                    # its crown would pierce the wall top
        crowding = fits._replace(u=3.8)                                 # closer to the first than the least masonry
        mesh, used = self.build([fits, too_high, crowding])
        self.assertEqual(used, [fits])

    def test_open_passages_have_no_fill(self):
        passage = mesh_openings.Opening(u=5.0, sill=100.4, width=4.0, height=5.0, depth=6.0, fill=None)
        mesh, used = self.build([passage])
        self.assertEqual(used, [passage])
        self.assertEqual(len(self.faces(mesh, self.GLASS)), 0)

    def test_tower_facets_shade_round(self):
        mesh, _ = mesh_openings.facade((5.0, 0.0), (0.0, 5.0), 100.0, 108.0, [], self.WALL, centre=(0.0, 0.0))
        radial = mesh.pos[:, [0, 2]] / np.linalg.norm(mesh.pos[:, [0, 2]], axis=1, keepdims=True)
        np.testing.assert_allclose(mesh.nrm[:, [0, 2]], radial, atol=1e-9)

    def test_doorway_orders_step_in_to_the_door(self):
        # a 4 m doorway 0.4 m deep with two inner orders, each 0.3 m in and 0.3 m deeper, closed by the door at 1.0 m
        door = mesh_openings.Opening(u=5.0, sill=100.5, width=4.0, height=6.0, depth=0.4, fill=(4, 42),
                                     orders=((0.3, 0.3), (0.3, 0.3)))
        mesh, used = self.build([door])
        self.assertEqual(used, [door])
        oak = self.faces(mesh, (4, 42))
        np.testing.assert_allclose(mesh.pos[oak.ravel(), 2], 1.0)
        inner = door._replace(width=2.8, height=5.4)
        self.assertAlmostEqual(self.area(mesh, (4, 42)), Polygon(mesh_openings.outline(inner)).area, places=6)
        reveal = self.faces(mesh, self.REVEAL)
        facing = mesh.nrm[reveal[:, 0]]
        rings = reveal[np.isclose(facing[:, 2], -1.0)]                 # the faces of the orders, facing out like the wall
        self.assertEqual(sorted(set(np.round(mesh.pos[rings.ravel(), 2], 6))), [0.4, 0.7])
        step = lambda d: Polygon(mesh_openings.outline(door._replace(width=4.0 - 0.6 * d, height=6.0 - 0.3 * d)))
        self.assertAlmostEqual(self.area(mesh._replace(idx=rings), self.REVEAL),
                               step(0).area - step(1).area + step(1).area - step(2).area, places=6)

    def test_cuts_open_the_wall_and_windows_give_way_to_them(self):
        cut = Polygon([(4.0, 100.5), (8.0, 100.5), (8.0, 105.0), (4.0, 105.0)])
        crowding = mesh_openings.Opening(u=3.5, **self.WINDOW)          # 0.5 m from the cut's left side
        clear = mesh_openings.Opening(u=1.5, **self.WINDOW)
        mesh, used = self.build([crowding, clear], cuts=[cut])
        self.assertEqual(used, [clear])
        hole = Polygon(mesh_openings.outline(clear))
        self.assertAlmostEqual(self.area(mesh, self.WALL), 80.0 - cut.area - hole.area, places=6)
        self.assertEqual(len(self.faces(mesh, self.REVEAL)), 2 * len(mesh_openings.outline(clear)))   # cuts are unlined

    def test_a_tube_lines_the_way_between_two_rings_facing_in(self):
        ring = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [-1.0, 2.0, 0.0]])
        far = ring + [0.0, 0.0, 5.0]
        mesh = mesh_openings.tube(ring, far, self.REVEAL)
        self.assertAlmostEqual(self.area(mesh, self.REVEAL), 5.0 * 8.0, places=6)
        axis = np.array([0.0, 1.0])
        for tri in mesh.idx:
            p, n = mesh.pos[tri], mesh.nrm[tri[0]]
            self.assertGreater(n[[0, 1]] @ (axis - p.mean(axis=0)[[0, 1]]), 0.0)
            self.assertGreater(np.cross(p[1] - p[0], p[2] - p[0]) @ n, 0.0)


class CastleOpeningTests(unittest.TestCase):
    """The castle's walls and towers have real recessed windows (castle-norman-openings) instead of painted ones, and
    every window looks out into the open air: not into the ground, a neighbouring wing or a tower."""
    CLEAR_M = 3.0                       # open air in front of each pane

    @classmethod
    def setUpClass(cls):
        chunks = [c for c in world.payload()["chunks"]
                  if c["asset"] == "castle"]
        cls.tris = np.concatenate([c["pos"][c["idx"]] for c in chunks])
        cls.attr = np.concatenate([c["attr"][c["idx"][:, 0]] for c in chunks])
        glass = np.flatnonzero((cls.attr[:, 0] == 4) & np.isin(cls.attr[:, 1], mesh_castle.GLASS_PANES))
        corners = np.round(cls.tris[glass].reshape(-1, 3) * 1000).astype(np.int64)
        _, vertex = np.unique(corners, axis=0, return_inverse=True)
        incidence = coo_matrix((np.ones(corners.shape[0]), (np.repeat(np.arange(len(glass)), 3), vertex.ravel())))
        _, pane = connected_components(bmat([[None, incidence], [incidence.T, None]]), directed=False)
        pane = pane[:len(glass)]                       # triangles sharing a corner make up one pane
        a, b, c = (cls.tris[glass, k] for k in range(3))
        normal = np.cross(b - a, c - a)
        cls.centre, cls.normal = [], []
        for p in np.unique(pane):
            mine = pane == p
            pts = cls.tris[glass[mine]].reshape(-1, 3)
            n = normal[mine].sum(axis=0)
            cls.centre.append((pts.min(axis=0) + pts.max(axis=0)) / 2)
            cls.normal.append(n / np.linalg.norm(n))
        cls.centre, cls.normal = np.array(cls.centre), np.array(cls.normal)
        data = world.data()
        cls.meta = data["meta"]
        cls.local, cls.castle = Grid(cls.meta["local"]), Grid(cls.meta["castle"])
        cls.bed = decode(data["tex"]["locBed"])
        cls.csd = castle_class_sd(data)

    def test_castle_faces_carry_real_openings_instead_of_painted_windows(self):
        painted = range(mesh_castle.CASTLE_CLASS_AUX, mesh_castle.CASTLE_CLASS_AUX + len(self.meta["castle"]["cls"]))
        self.assertFalse(np.isin(self.attr[:, 1], painted).any())
        self.assertGreater(len(self.centre), 1000)

    def test_lit_rooms_vary_some_bright_some_dim_most_dark(self):
        panes = self.attr[(self.attr[:, 0] == 4) & np.isin(self.attr[:, 1], mesh_castle.GLASS_PANES), 1]
        share = {code: float(np.mean(panes == code)) for code in mesh_castle.GLASS_PANES}
        self.assertGreater(share[mesh_castle.GLASS], 0.5)                          # most rooms dark
        self.assertGreater(share[mesh_castle.GLASS_LIT], 0.1)
        self.assertGreater(share[mesh_castle.GLASS_DIM], 0.1)                      # a candle further back in the room

    def test_every_pane_looks_out_into_the_open_air(self):
        start = self.centre + self.normal * 0.01           # clear of the pane's own 16-bit quantized triangles
        along = start[:, None, :] + self.normal[:, None, :] * np.linspace(0.0, self.CLEAR_M, 13)[None, :, None]
        ground = sample(self.bed, self.local, along[..., 0].ravel(), along[..., 2].ravel()).reshape(along.shape[:2])
        buried = np.flatnonzero(np.any(along[..., 1] < ground + 0.05, axis=1))
        self.assertEqual(len(buried), 0, f"panes facing the ground at {np.round(self.centre[buried[:5]], 1).tolist()}")
        blocked = np.flatnonzero(segment_hits(start, self.normal, self.CLEAR_M, self.tris))
        self.assertEqual(len(blocked), 0, f"{len(blocked)} of {len(start)} panes face masonry, e.g. at "
                                          f"{np.round(self.centre[blocked[:5]], 1).tolist()}")

    def test_every_storeyed_mass_and_named_tower_has_windows(self):
        inside = self.centre - self.normal * 1.5                   # a step behind the glass, inside the building
        for k in (CASTLE_CLASSES.index(name) for name in ("range", "block", "great_hall", "keep")):
            with self.subTest(mass=CASTLE_CLASSES[k]):
                self.assertGreaterEqual(int((sample(self.csd[..., k], self.castle, inside[:, 0], inside[:, 2]) < 0).sum()), 10)
        towers = self.meta["towers"]
        for name in ("AT", "RT", "GT", "NT"):
            x, z, r, *_ = min(towers, key=lambda t: math.dist(t[:2], self.meta["named"][name]))
            with self.subTest(tower=name):
                self.assertGreaterEqual(int((np.hypot(self.centre[:, 0] - x, self.centre[:, 2] - z) < 1.3 * r + 1.0).sum()), 3)


class OpenAirTests(unittest.TestCase):
    """A window's reveal must stay inside its own wing: a courtyard close behind the wall would open through it."""

    def test_reveals_stay_inside_their_own_wing(self):
        # a 20 x 12 m wing whose courtyard comes within 0.4 m of its south wall along x = 2..9
        wing = orient(Polygon([(0, 0), (20, 0), (20, 12), (0, 12)], [[(2, 0.4), (9, 0.4), (9, 8), (2, 8)]]), 1.0)
        mass = mesh_castle.Mass(2, wing, lambda x, z: np.full(np.shape(x), 24.0), lambda x, z: np.full(np.shape(x), 28.0))
        air = mesh_castle.Air([mass], [], lambda x, z: np.zeros(np.shape(x)))
        x, z = np.array([5.0, 15.0]), np.zeros(2)
        south = np.tile([0.0, -1.0], (2, 1))
        sill = np.full(2, 5.0)
        self.assertEqual(air(x, z, south, sill, sill + 2.2, np.full(2, 0.7), mass=np.zeros(2, int)).tolist(), [False, True])


class CastleEntranceTests(unittest.TestCase):
    """White Hound: "a flight of stone steps" up to "the huge, oak front door", beside the Astronomy Tower (on its
    right as you face the doors), the doors facing as near west as still commands a view of the gates."""

    @classmethod
    def setUpClass(cls):
        chunks = [c for c in world.payload()["chunks"]
                  if c["asset"] == "castle"]
        cls.tris = np.concatenate([c["pos"][c["idx"]] for c in chunks])
        cls.attr = np.concatenate([c["attr"][c["idx"][:, 0]] for c in chunks])
        n = np.cross(cls.tris[:, 1] - cls.tris[:, 0], cls.tris[:, 2] - cls.tris[:, 0])
        cls.normal = n / np.linalg.norm(n, axis=1, keepdims=True)
        data = world.data()
        cls.meta = data["meta"]
        cls.local, cls.bed = Grid(cls.meta["local"]), decode(data["tex"]["locBed"])
        cls.at = min(cls.meta["towers"], key=lambda t: math.dist(t[:2], cls.meta["named"]["AT"]))[:2]
        centre = cls.tris.mean(axis=1)
        oak = np.all(cls.attr == (4, mesh_castle.OAK), axis=1) & (np.hypot(centre[:, 0] - cls.at[0], centre[:, 2] - cls.at[1]) < 30.0)
        cls.door = cls.tris[oak].reshape(-1, 3)
        cls.out = cls.normal[oak].mean(axis=0)
        cls.out /= np.linalg.norm(cls.out)

    def ground(self, p):
        return float(sample(self.bed, self.local, np.array([p[0]]), np.array([p[2]]))[0])

    def test_oak_doors_face_west_beside_the_astronomy_tower(self):
        self.assertGreater(len(self.door), 0)
        self.assertGreater(-self.out[0], math.cos(math.radians(30)))      # within 30 degrees of west
        centre = (self.door.min(axis=0) + self.door.max(axis=0)) / 2
        x, z = self.at
        towards = np.array([x - centre[0], z - centre[2]])
        self.assertLess(np.hypot(*towards), 20.0)
        facing = -self.out[[0, 2]]
        self.assertGreater(towards @ np.array([facing[1], -facing[0]]), 0.0)   # on the right as you face the doors

    def test_a_flight_of_steps_climbs_from_the_lawn_to_the_doors(self):
        foot = self.door[:, 1].min()
        centre = (self.door.min(axis=0) + self.door.max(axis=0)) / 2
        self.assertGreater(foot - self.ground(centre + self.out * 1.5), 0.8)   # a raised threshold
        side = np.array([-self.out[2], 0.0, self.out[0]])
        rel = self.tris.mean(axis=1) - centre
        treads = np.all(self.attr == (4, mesh_castle.PLAIN), axis=1) & (self.normal[:, 1] > 0.99) \
            & (np.abs(rel @ side) < 2.0) & (rel @ self.out > 0.0) & (rel @ self.out < 30.0)
        levels = sorted({round(float(y), 2) for y in self.tris[treads][:, :, 1].ravel()}, reverse=True)
        self.assertAlmostEqual(levels[0], foot, delta=0.05)                  # the top step is the threshold
        risers = -np.diff(levels)
        self.assertGreaterEqual(len(risers), 3)
        self.assertTrue(np.all((risers > 0.1) & (risers < 0.2)), risers)
        last = self.tris[treads][np.isclose(self.tris[treads][:, :, 1].mean(axis=1), levels[-1], atol=0.01)]
        front = last.reshape(-1, 3)[np.argmax((last.reshape(-1, 3) - centre) @ self.out)]
        self.assertLess(levels[-1] - self.ground(front + self.out * 0.2), 0.4)   # one short step down to the grass


class CastlePendTests(unittest.TestCase):
    """White Hound: from the front lawn "in through the pend which leads to the broomshed", into a court which
    "evidently can be barred": a lined passage through the wing, with gates."""

    @classmethod
    def setUpClass(cls):
        chunks = [c for c in world.payload()["chunks"]
                  if c["asset"] == "castle"]
        cls.tris = np.concatenate([c["pos"][c["idx"]] for c in chunks])
        cls.attr = np.concatenate([c["attr"][c["idx"][:, 0]] for c in chunks])
        data = world.data()
        meta = data["meta"]
        scene = world.scene()
        traced = {t["id"]: t for t in next(a for a in scene["assets"] if a["id"] == "castle")["traced"]}
        plan, _ = mesh_castle.plan_fit(traced["plan-fit"]["plan_px"], {k: tuple(v) for k, v in meta["named"].items()})
        ends = traced["broomshed-pend"]["plan_px"]
        cls.lawn, cls.court = plan(*ends["lawn"]), plan(*ends["court"])
        floor = float(sample(decode(data["tex"]["locBed"]), Grid(meta["local"]), *(np.array([v]) for v in cls.lawn))[0])
        cls.floor = floor

    def sight_lines(self):
        """Horizontal 25 m rays from the lawn into the court, swept round the traced line and across it."""
        axis = (self.court - self.lawn) / np.linalg.norm(self.court - self.lawn)
        starts, dirs = [], []
        for turn in np.radians(np.arange(-15, 16, 1)):
            d = np.array([axis[0] * math.cos(turn) - axis[1] * math.sin(turn), axis[0] * math.sin(turn) + axis[1] * math.cos(turn)])
            for shift in np.arange(-3.0, 3.01, 0.5):
                o = self.lawn - axis * 9.0 + np.array([-axis[1], axis[0]]) * shift
                for lift in (1.5, 2.5, 3.5):
                    starts.append([o[0], self.floor + lift, o[1]])
                    dirs.append([d[0], 0.0, d[1]])
        return np.array(starts), np.array(dirs)

    def test_a_lined_passage_runs_from_the_lawn_into_the_court(self):
        starts, dirs = self.sight_lines()
        clear = np.flatnonzero(~segment_hits(starts, dirs, 25.0, self.tris))
        self.assertGreater(len(clear), 0)                                 # a straight way through
        k = clear[len(clear) // 2]
        middle = starts[k] + dirs[k] * 9.0 + dirs[k] * np.linalg.norm(self.court - self.lawn) / 2
        side = np.array([dirs[k][2], 0.0, -dirs[k][0]])
        lined = segment_hits(np.tile(middle, (3, 1)), np.array([side, -side, [0.0, 1.0, 0.0]]), 3.5, self.tris)
        self.assertTrue(lined.all(), lined)                                # walls each side and a vault overhead

    def test_open_gates_could_bar_the_passage(self):
        mid = (self.lawn + self.court) / 2
        centre = self.tris.mean(axis=1)
        near = (np.hypot(centre[:, 0] - mid[0], centre[:, 2] - mid[1]) < 12.0) & np.all(self.attr == mesh_castle.TIMBER, axis=1)
        a, b, c = (self.tris[near, k] for k in range(3))
        timber = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2      # two faces to each leaf
        self.assertGreater(timber, mesh_castle.PEND_WIDTH_M * mesh_castle.GATE_M)


class HeadsTowerTests(unittest.TestCase):
    """White Hound: "the Head's tower is joined on to the end of the hospital wing", with "two turrets running up the
    side" holding its stairs, and "an external stair ... leads down to an enclosed corner of garden"."""

    @classmethod
    def setUpClass(cls):
        payload = world.payload()
        chunks = [c for c in payload["chunks"] if c["asset"] == "castle"]
        cls.tris = np.concatenate([c["pos"][c["idx"]] for c in chunks])
        cls.attr = np.concatenate([c["attr"][c["idx"][:, 0]] for c in chunks])
        n = np.cross(cls.tris[:, 1] - cls.tris[:, 0], cls.tris[:, 2] - cls.tris[:, 0])
        with np.errstate(invalid="ignore"):
            cls.normal = n / np.linalg.norm(n, axis=1, keepdims=True)
        data = world.data()
        cls.meta = data["meta"]
        cls.local, cls.bed = Grid(cls.meta["local"]), decode(data["tex"]["locBed"])
        scene = world.scene()
        traced = {t["id"]: t for t in next(a for a in scene["assets"] if a["id"] == "castle")["traced"]}
        plan, _ = mesh_castle.plan_fit(traced["plan-fit"]["plan_px"], {k: tuple(v) for k, v in cls.meta["named"].items()})
        cls.ho = plan(*traced["heads-tower"]["plan_px"])
        index = {name: min(range(len(cls.meta["towers"])), key=lambda i: math.dist(cls.meta["towers"][i][:2], cls.meta["named"][name]))
                 for name in ("AT", "RT", "GT")}
        towers = cls.attr[:, 1].astype(int) - mesh_castle.MESH_TOWER_AUX
        cls.named_tops = {name: float(cls.tris[(cls.attr[:, 0] == 4) & (towers == i)][:, :, 1].max()) for name, i in index.items()}
        pts = cls.tris.reshape(-1, 3)
        cls.reach = np.hypot(pts[:, 0] - cls.ho[0], pts[:, 2] - cls.ho[1])
        cls.pts = pts
        cls.top = surface_heights(cls.tris, *cls.ho).max()                         # the tower's roof walk at its axis

    def ground(self, x, z):
        return float(sample(self.bed, self.local, np.array([x]), np.array([z]))[0])

    def test_a_round_tower_rises_from_the_end_of_the_hospital_wing(self):
        self.assertGreater(self.top, self.ground(*self.ho) + 30.0)                  # a tower over the wing's 24 m
        self.assertLess(self.top, min(self.named_tops["RT"], self.named_tops["GT"]))   # but not one of the high towers
        rim = self.pts[np.abs(self.reach - mesh_castle.HEAD_R) < 0.3]
        angles = np.sort(np.degrees(np.arctan2(rim[:, 2] - self.ho[1], rim[:, 0] - self.ho[0])))
        self.assertLess(np.diff(np.concatenate([angles, angles[:1] + 360])).max(), 40.0)   # round, all the way round

    def test_two_stair_turrets_run_up_its_side(self):
        caps = self.pts[(self.reach > mesh_castle.HEAD_R + 0.2) & (self.reach < mesh_castle.HEAD_R + 4.0)
                        & (self.pts[:, 1] > self.top + mesh_castle.CROWN + 0.5)]       # above the battlements
        angles = np.sort(np.degrees(np.arctan2(caps[:, 2] - self.ho[1], caps[:, 0] - self.ho[0])))
        gaps = np.diff(np.concatenate([angles, angles[:1] + 360]))
        self.assertEqual(int((gaps > 30.0).sum()), 2)                              # two separate turret caps
        for a in angles[np.flatnonzero(gaps > 30.0)]:
            d = np.radians(a)
            foot = [self.ho[0] + math.cos(d) * (mesh_castle.HEAD_R + 1.0), self.ho[1] + math.sin(d) * (mesh_castle.HEAD_R + 1.0)]
            column = self.pts[np.hypot(self.pts[:, 0] - foot[0], self.pts[:, 2] - foot[1]) < 2.5]
            self.assertLess(column[:, 1].min(), self.ground(*foot) + 0.5)         # running up from the ground

    def test_an_external_stair_leads_down_into_an_enclosed_garden(self):
        centre = self.tris.mean(axis=1)
        reach = np.hypot(centre[:, 0] - self.ho[0], centre[:, 2] - self.ho[1])
        treads = np.all(self.attr == (4, mesh_castle.PLAIN), axis=1) & (self.normal[:, 1] > 0.99) & (reach < 20.0) \
            & (self.tris[:, :, 1].max(axis=1) < self.ground(*self.ho) + 4.0)         # at its foot, not chimney tops
        levels = sorted({round(float(y), 2) for y in self.tris[treads][:, :, 1].ravel()}, reverse=True)
        self.assertGreaterEqual(len(levels), 4)
        self.assertTrue(np.all((-np.diff(levels) > 0.1) & (-np.diff(levels) < 0.2)), levels)
        beds = np.all(self.attr == mesh_castle.HERBS, axis=1) & (self.normal[:, 1] > 0.9) & (reach < 30.0)
        self.assertGreaterEqual(len(self.tris[beds]), 8)                             # herb beds
        middle = self.tris[beds].reshape(-1, 3).mean(axis=0)[[0, 2]]
        level = self.ground(*middle)
        near = np.hypot(*(centre[:, [0, 2]] - middle).T) < 25.0
        bounding = near & ((self.tris[:, :, 1].max(axis=1) > level + 1.0) | np.all(self.attr == mesh_castle.WALL_STONE, axis=1))
        walls = shapely.union_all([g for g in (Polygon(t[:, [0, 2]]) for t in self.tris[bounding]) if g.area > 1e-6])
        turns = np.radians(np.arange(0, 360, 22.5))
        ways_out = [t for t in turns if not shapely.LineString([middle, middle + 15.0 * np.array([math.cos(t), math.sin(t)])]).intersects(walls)]
        self.assertLessEqual(len(ways_out), 2, np.degrees(ways_out))               # walled round in plan, but for a gate


class CourtHouseTests(unittest.TestCase):
    """court-buildings-as-houses: "domestic, house-like buildings ... within a more stark curtain wall", with pitched
    roofs and chimneys; and sparing chimneys on the ranges and blocks."""

    @classmethod
    def setUpClass(cls):
        chunks = [c for c in world.payload()["chunks"]
                  if c["asset"] == "castle"]
        cls.tris = np.concatenate([c["pos"][c["idx"]] for c in chunks])
        cls.attr = np.concatenate([c["attr"][c["idx"][:, 0]] for c in chunks])
        n = np.cross(cls.tris[:, 1] - cls.tris[:, 0], cls.tris[:, 2] - cls.tris[:, 0])
        with np.errstate(invalid="ignore"):
            cls.normal = n / np.linalg.norm(n, axis=1, keepdims=True)
        cls.centre = cls.tris.mean(axis=1)
        data = world.data()
        cls.meta = data["meta"]
        image, pieces = castle_piece_labels(data)
        grid = Grid(cls.meta["castle"])
        at = lambda x, z: image[tuple(int(round(float(v))) for v in grid.to_texel(x, z))]
        towers = {at(x, z) for x, z, *_ in cls.meta["towers"]}
        cls.houses = [p for p in pieces if p["role"] == "court" and p["area"] < mesh_castle.HOUSE_MAX_M2 and p["id"] not in towers]

    def test_court_pieces_that_are_not_towers_are_gabled_houses_with_chimneys(self):
        self.assertGreaterEqual(len(self.houses), 3)
        for p in self.houses:
            with self.subTest(piece=p["id"]):
                near = np.hypot(self.centre[:, 0] - p["x"], self.centre[:, 2] - p["z"]) < 0.8 * math.sqrt(p["area"]) + 1.0
                self.assertGreater(int((near & (self.normal[:, 1] > 0.5) & (self.normal[:, 1] < 0.9)).sum()), 0)   # pitched
                stack = near & np.all(self.attr == (4, mesh_castle.PLAIN), axis=1)
                self.assertGreater(int(stack.sum()), 0)
                top = self.tris[stack].reshape(-1, 3)
                x, _, z = top[np.argmax(top[:, 1])]
                roofs = surface_heights(self.tris[~np.all(self.attr == (4, mesh_castle.PLAIN), axis=1)], x, z)
                self.assertGreater(top[:, 1].max(), roofs.max() + 0.5)               # a chimney standing over its roof

    def test_ranges_and_blocks_carry_a_few_chimneys(self):
        stacks = np.all(self.attr == (4, mesh_castle.PLAIN), axis=1) & (self.normal[:, 1] > 0.99)
        tops = self.centre[stacks]
        eaves = world.levels()["platform"] + self.meta["castle"]["cls"][CASTLE_CLASSES.index("range")][0]
        tops = tops[tops[:, 1] > eaves]                                            # over the ranges' eaves
        clusters = {(round(x / 3), round(z / 3)) for x, _, z in tops}
        self.assertGreaterEqual(len(clusters), 10)
        self.assertLessEqual(len(clusters), 120)                                   # sparing


class CastleSkylineTests(unittest.TestCase):
    """Astronomy is the highest point of the castle, and Ravenclaw and Gryffindor the next two towers (wh-skyline)."""

    def test_castle_skyline_puts_astronomy_first(self):
        meta = world.data()["meta"]
        index = {n: min(range(len(meta["towers"])), key=lambda i: math.dist(meta["towers"][i][:2], meta["named"][n]))
                 for n in ("AT", "RT", "GT")}
        tops, body = {}, -math.inf
        for chunk in world.payload()["chunks"]:
            if chunk["asset"] != "castle":
                continue
            for aux in np.unique(chunk["attr"][:, 1]):
                y = float(chunk["pos"][chunk["attr"][:, 1] == aux, 1].max())
                if aux >= mesh_castle.MESH_TOWER_AUX:
                    tops[aux - mesh_castle.MESH_TOWER_AUX] = max(tops.get(aux - mesh_castle.MESH_TOWER_AUX, -math.inf), y)
                else:
                    body = max(body, y)
        astronomy, ravenclaw, gryffindor = (tops[index[n]] for n in ("AT", "RT", "GT"))
        others = [y for i, y in tops.items() if i not in index.values()]
        self.assertGreater(astronomy, max(body, ravenclaw, gryffindor, *others))   # "the highest of the castle"
        self.assertGreater(min(ravenclaw, gryffindor), max(others))                # "the three highest Towers"

    def test_ravenclaw_and_gryffindor_drums_clear_the_main_buildings(self):
        # on a level platform Gryffindor's 30 m drum sat below the 32 m blocks round it, its turrets alone clearing them
        base = world.datum()
        named = {"AT": (0.0, 0.0), "RT": (50.0, 0.0), "GT": (0.0, 50.0), "NT": (-50.0, 0.0)}
        towers = [(0.0, 0.0, 3.6, base + 58.0, mesh_castle.ASTRONOMY), (50.0, 0.0, 5.2, base + 45.0, 1),
                  (0.0, 50.0, 7.0, base + 30.0, mesh_castle.GRYFFINDOR), (-50.0, 0.0, 3.6, base + 54.0, mesh_castle.NORTH),
                  (90.0, 90.0, 4.0, base + 20.0, 0)]
        body_top = base + 40.0
        kept, _ = mesh_castle.skyline(towers, named, body_top, set())
        drums = {t["i"]: t["top"] for t in kept}
        self.assertGreaterEqual(min(drums[1], drums[2]), body_top + mesh_castle.RANK_CLEARANCE_M)


class CastleMeshOracleTests(unittest.TestCase):
    """trimesh, as an independent oracle on the decoded payload."""

    def test_the_decoded_meshes_have_only_the_known_faults(self):
        self.assertEqual(world.mesh_faults("castle"), MESH_FAULTS)


if __name__ == "__main__":
    unittest.main()
