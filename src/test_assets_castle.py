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
from build_meshes import piece_owners  # noqa: E402

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
        cls.centre, cls.normal, cls.size = [], [], []
        for p in np.unique(pane):
            mine = pane == p
            pts = cls.tris[glass[mine]].reshape(-1, 3)
            n = normal[mine].sum(axis=0)
            cls.centre.append((pts.min(axis=0) + pts.max(axis=0)) / 2)
            cls.normal.append(n / np.linalg.norm(n))
            cls.size.append((float(np.ptp(pts[:, [0, 2]] @ np.array([-n[2], n[0]]) / np.hypot(n[0], n[2]))),
                             float(np.ptp(pts[:, 1]))))
        cls.centre, cls.normal, cls.size = np.array(cls.centre), np.array(cls.normal), np.array(cls.size)
        data = world.data()
        cls.meta = data["meta"]
        cls.local, cls.castle = Grid(cls.meta["local"]), Grid(cls.meta["castle"])
        cls.bed = decode(data["tex"]["locBed"])
        cls.csd = castle_class_sd(data)

    def test_castle_faces_carry_real_openings_instead_of_painted_windows(self):
        painted = range(mesh_castle.CASTLE_CLASS_AUX, mesh_castle.CASTLE_CLASS_AUX + len(self.meta["castle"]["cls"]))
        self.assertFalse(np.isin(self.attr[:, 1], painted).any())
        self.assertGreater(len(self.centre), 1000)

    def test_the_storeys_windows_differ_and_the_wings_spacing_varies(self):
        # the west front read as one grid of identical windows: storeyed masses' windows come in several sizes, the
        # principal storey's tallest and the attic's smallest, and wings space them differently
        inside = self.centre - self.normal * 1.5
        k = [CASTLE_CLASSES.index(n) for n in ("range", "block")]
        storeyed = np.min([sample(self.csd[..., j], self.castle, inside[:, 0], inside[:, 2]) for j in k], axis=0) < 0
        heights = self.size[storeyed, 1]
        for storey, (_, height) in mesh_castle.STOREYS.items():
            with self.subTest(storey=storey):
                self.assertGreater(int((np.abs(heights - height) < 0.05).sum()), 50)
        rows = {}
        c, n = self.centre[storeyed], self.normal[storeyed]
        t = np.column_stack([-n[:, 2], n[:, 0]]) / np.hypot(n[:, 0], n[:, 2])[:, None]     # along each wall
        key = np.column_stack([np.round(np.degrees(np.arctan2(n[:, 2], n[:, 0]))),
                               np.round(np.einsum("ij,ij->i", c[:, [0, 2]], n[:, [0, 2]])), np.round(c[:, 1])])
        _, row = np.unique(key, axis=0, return_inverse=True)
        along = np.einsum("ij,ij->i", c[:, [0, 2]], t)
        for r in np.unique(row):                                # each row's usual step, as one of the wings' spacings
            steps = np.diff(np.sort(along[row == r]))
            if len(steps) >= 3:
                usual = float(np.median(steps))
                nearest = min(mesh_castle.SPACINGS, key=lambda spacing: abs(spacing - usual))
                if abs(nearest - usual) < 0.1:
                    rows[nearest] = rows.get(nearest, 0) + 1
        self.assertGreaterEqual(sum(count >= 5 for count in rows.values()), 3, rows)

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

    def test_gryffindors_turrets_and_the_thin_columns_have_windows(self):
        towers = self.meta["towers"]
        pick = lambda name: min(range(len(towers)), key=lambda i: math.dist(towers[i][:2], self.meta["named"][name]))
        reach = lambda i: np.hypot(self.centre[:, 0] - towers[i][0], self.centre[:, 2] - towers[i][1])
        gt = pick("GT")
        top = float(self.tris[np.all(self.attr == (4, mesh_castle.MESH_TOWER_AUX + gt), axis=1)][:, :, 1].max())
        turrets = (reach(gt) < 1.3 * towers[gt][2]) & (self.centre[:, 1] > top - mesh_castle.TURRET_M - mesh_castle.CROWN + 1.0)
        self.assertGreaterEqual(int(turrets.sum()), 3)                   # "three thinner turrets", lived in
        for name in ("AT", "NT"):                                        # "a long thin column topped by a wider ... top"
            i = pick(name)
            column = reach(i) < mesh_castle.COLUMN * towers[i][2] + 0.5
            with self.subTest(tower=name):
                self.assertGreaterEqual(int(column.sum()), 3)


class PartyWallTests(unittest.TestCase):
    """A wall is left out as hidden only where taller masonry covers it along its whole length: one whose middle
    touches a drum but whose end faces open air keeps its facade."""

    def test_a_wall_only_partly_against_a_drum_stays(self):
        cls = [[2.2, 4], [13.0, 0], [24.0, 1], [32.0, 1], [21.0, 3], [27.0, 0]]
        # a range whose east wall at x = 6 runs from z = -4 to z = 4, beside a drum of radius 6 centred at (12, 0): its middle against the drum, its ends in the open
        rng = orient(Polygon([(0, -4), (6, -4), (6, 4), (0, 4)]), 1.0)
        drum = (Point(12.0, 0.0).buffer(6.05, 64), world.datum() + 60.0)
        base = lambda x, z: np.full(np.shape(x), world.datum())
        [(hidden, _)] = mesh_castle.party_walls([(2, rng)], cls, [drum], None, None, base=base)
        walls = mesh_castle.edges(rng)
        east = next(i for i, (p0, p1) in enumerate(walls) if abs(p0[0] - 6) < 1e-9 and abs(p1[0] - 6) < 1e-9)
        self.assertFalse(hidden[east])


class StraightenTests(unittest.TestCase):
    """Straightening each class's traced outline moves its walls by up to STRAIGHTEN_M; where two masses met along a
    raster edge, the slit that opens between them is filled again."""

    def test_the_traced_masses_meet_without_slits(self):
        from mesh_source import field_polygons

        data = world.data()
        meta = data["meta"]
        grid = Grid(meta["castle"])
        image, pieces = castle_piece_labels(data)
        owners = piece_owners(world.scene(), pieces)
        keep = np.isin(image, [p for p, a in owners.items() if a == "castle"])
        csd = castle_class_sd(data)
        cls = meta["castle"]["cls"]
        raws = [(k, p) for k in range(len(cls)) for p in field_polygons(csd[..., k], grid, 0.0, keep)]
        out = mesh_castle.straighten(raws, cls)
        lost = shapely.union_all([p for _, p in raws]).difference(shapely.union_all([p for _, p in out]))
        tree = shapely.STRtree([p for _, p in out])
        slits = [g for g in getattr(lost, "geoms", [lost]) if g.geom_type == "Polygon" and g.area > 0.5
                 and len(tree.query(g.buffer(0.05), predicate="intersects")) >= 2
                 and g.boundary.difference(shapely.union_all([p for _, p in out]).buffer(0.05)).length < 0.3 * g.length]
        self.assertLess(sum(g.area for g in slits), 50.0, [tuple(np.round(g.centroid.coords[0], 1)) for g in slits[:5]])

    def test_a_small_wall_stub_left_against_a_taller_mass_is_dropped(self):
        # a 4 m stub of battlemented wall left against the east wing's block when the cliff took the rest of its wall
        # stood proud of the block's face, its merlons and coping showing as pale outlines two-thirds down (east-wing)
        cls = [[2.2, 4], [13.0, 0], [24.0, 1], [32.0, 1], [21.0, 3], [27.0, 0]]
        k_wall, k_block = (CASTLE_CLASSES.index(n) for n in ("wall", "block"))
        block = Polygon([(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)])
        stub = Polygon([(30.0, 10.0), (34.0, 10.0), (34.0, 14.0), (30.0, 14.0)])
        out = mesh_castle.straighten([(k_block, block), (k_wall, stub)], cls)
        self.assertEqual([k for k, _ in out], [k_block])
        curtain = Polygon([(30.0, 10.0), (60.0, 10.0), (60.0, 13.0), (30.0, 13.0)])   # a curtain wall running off stays
        out = mesh_castle.straighten([(k_block, block), (k_wall, curtain)], cls)
        self.assertEqual(sorted(k for k, _ in out), sorted([k_block, k_wall]))


class DrumCutTests(unittest.TestCase):
    """A mass cut round a tower's drum keeps no thin crescent wrapped round the drum's far side: its corner stood out
    past the drum's face, with a gap at its foot (north-battlements, round 3)."""

    def test_a_crescent_left_round_a_drum_goes_unless_it_joins_another_mass(self):
        drum = shapely.Point(0.0, 0.0).buffer(5.0)
        rng = Polygon([(-6.5, -6.0), (30.0, -6.0), (30.0, 6.0), (-6.5, 6.0)]).difference(drum)
        out = mesh_castle.unwrapped(rng, (0.0, 0.0), 5.0, Polygon())
        self.assertLess(out.intersection(shapely.box(-10.0, -10.0, -3.0, 10.0)).area, 0.05)   # the far side is clear
        self.assertGreater(out.area, rng.area - 45.0)                                       # the range itself stays
        through = Polygon([(-6.5, -6.0), (60.0, -6.0), (60.0, 6.0), (-6.5, 6.0)]).difference(drum)
        self.assertTrue(mesh_castle.unwrapped(through.union(shapely.box(-40.0, -6.0, -6.5, 6.0)), (0.0, 0.0), 5.0,
                                              Polygon()).buffer(0.05).contains(shapely.Point(-30.0, 0.0)))  # a range running on past it stays
        wall = Polygon([(-30.0, -1.0), (-6.4, -1.0), (-6.4, 1.0), (-30.0, 1.0)])           # a wall meeting it there
        out = mesh_castle.unwrapped(rng, (0.0, 0.0), 5.0, wall)
        self.assertTrue(out.buffer(0.05).intersects(wall))
        split = rng.difference(shapely.box(-10.0, -0.5, 0.0, 0.5))                         # two pieces, one met by a wall
        out = mesh_castle.unwrapped(split, (0.0, 0.0), 5.0, shapely.box(-30.0, -5.0, -6.4, -4.0))
        self.assertLess(out.intersection(shapely.box(-10.0, 0.0, -3.0, 10.0)).area, 0.05)
        self.assertGreater(out.intersection(shapely.box(-10.0, -10.0, -3.0, 0.0)).area, 10.0)


    def test_a_face_a_drum_straddles_keeps_its_length(self):
        block = shapely.box(-40.0, -30.0, 40.0, 3.0)                                        # the drum 3 m inside its face
        o = block.difference(shapely.Point(0.0, 0.0).buffer(5.0))
        out = mesh_castle.unwrapped(o, (0.0, 0.0), 5.0, Polygon())
        self.assertAlmostEqual(out.intersection(shapely.box(-40.0, 0.0, -10.0, 3.0)).area, 90.0, delta=0.5)

    def test_a_mass_a_tower_overhangs_is_cut_back_square_not_notched_round_it(self):
        # the block round the North Tower was notched round its clearance, so the drum read as built into it (round 3)
        disc = shapely.Point(0.0, 0.0).buffer(6.0, 64)
        for block in (shapely.box(-10.0, -20.0, 10.0, 2.0), shapely.box(-10.0, -20.0, 10.0, -3.0)):    # over its axis, or short
            out = mesh_castle.cut_back(block, disc)
            self.assertLess(out.bounds[3], -6.0 + 1e-3)                                       # one straight wall ...
            self.assertGreater(out.area, 20.0 * 13.0 - 1.0)                                   # ... and no more taken
        self.assertTrue(mesh_castle.cut_back(shapely.box(-20.0, -20.0, 20.0, 20.0), disc).contains(shapely.Point(0.0, 9.0)))
        through = mesh_castle.cut_back(shapely.box(-30.0, -3.0, 20.0, 3.0), shapely.Point(0.0, 0.0).buffer(8.0, 64))
        self.assertAlmostEqual(through.intersection(shapely.box(8.0, -3.0, 20.0, 3.0)).area, 72.0, delta=1.0)   # a range runs on


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

    def test_the_oak_doors_are_boarded_and_bound_in_iron(self):
        centre = (self.door.min(axis=0) + self.door.max(axis=0)) / 2
        width = float(np.ptp(self.door[:, [0, 2]] @ np.array([-self.out[2], self.out[0]])))
        mid = self.tris.mean(axis=1)
        iron = np.all(self.attr == (4, mesh_castle.IRON), axis=1) & (np.linalg.norm(mid - centre, axis=1) < 5.0)
        side = np.array([-self.out[2], 0.0, self.out[0]])
        straps = {round(float(y), 1) for t in self.tris[iron] if np.ptp(t @ side) > 0.3 * width for y in [t[:, 1].mean()]}
        self.assertGreaterEqual(len({round(y) for y in straps}), 3)         # straps across the leaves at three heights
        seams = np.all(self.attr == (4, mesh_castle.IRON), axis=1) & (np.linalg.norm(mid - centre, axis=1) < 5.0)
        uprights = {round(float(t[:, [0, 2]].mean(axis=0) @ side[[0, 2]] - centre[[0, 2]] @ side[[0, 2]]), 1)
                    for t in self.tris[seams] if np.ptp(t[:, 1]) > 2.0}
        self.assertGreaterEqual(len(uprights), 6)                            # the boards' joints
        self.assertEqual({abs(u) for u in uprights if u > 0}, {abs(u) for u in uprights if u < 0})   # the leaves mirror

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
        walls = segment_hits(np.tile(middle, (2, 1)), np.array([side, -side]), 3.5, self.tris)
        # The vault is judged from the floor, not from the chosen ray's lift, which shifts with any change to the wing.
        low = middle + np.array([0.0, self.floor + 1.5 - middle[1], 0.0])
        vault = segment_hits(low[None], np.array([[0.0, 1.0, 0.0]]), 5.5, self.tris)
        self.assertTrue(walls.all() and vault.all(), (walls, vault))       # walls each side and a vault overhead

    def test_open_gates_could_bar_the_passage(self):
        mid = (self.lawn + self.court) / 2
        centre = self.tris.mean(axis=1)
        near = (np.hypot(centre[:, 0] - mid[0], centre[:, 2] - mid[1]) < 12.0) & np.all(self.attr == (4, mesh_castle.OAK), axis=1)
        a, b, c = (self.tris[near, k] for k in range(3))
        oak = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2      # two faces to each leaf
        self.assertGreater(oak, mesh_castle.PEND_WIDTH_M * mesh_castle.GATE_M)   # oak leaves, as the doors


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


class FacadeDetailTests(unittest.TestCase):
    """The ranges' and blocks' walls are banded by string courses between storeys and a cornice under the eaves,
    projecting from the wall, so a wing reads as storeys of a castle rather than a flat grid."""

    def test_storeyed_walls_carry_projecting_string_courses(self):
        tris, attr, normal = world.payload_part("castle")
        course = np.all(attr == (4, mesh_castle.COURSE), axis=1)
        a = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
        tops = course & (normal[:, 1] > 0.99)
        self.assertGreater(float(a[tops].sum()), 500.0)                 # in plan: a 0.12 m band along kilometres of wall
        levels = np.round(tris[tops][:, :, 1].mean(axis=1) - world.levels()["platform"], 0)
        self.assertGreaterEqual(len(np.unique(levels)), 3)              # several storey lines and the eaves


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
                pots = near & np.all(self.attr == (4, mesh_castle.IRON), axis=1)
                self.assertGreater(float(self.tris[pots][:, :, 1].max(initial=-np.inf)), top[:, 1].max() + 0.3)   # sooted pots on it

    def test_court_houses_have_a_door_and_windows_on_the_court(self):
        panes = np.isin(self.attr[:, 1], mesh_castle.GLASS_PANES) & (self.attr[:, 0] == 4)
        for p in self.houses:
            with self.subTest(piece=p["id"]):
                near = np.hypot(self.centre[:, 0] - p["x"], self.centre[:, 2] - p["z"]) < 0.8 * math.sqrt(p["area"]) + 1.0
                door = near & np.all(self.attr == (4, mesh_castle.OAK), axis=1)
                self.assertGreater(int(door.sum()), 0)                                   # a door, at the foot of a wall
                self.assertLess(float(self.tris[door][:, :, 1].min()) - float(self.tris[near][:, :, 1].min()), 2.0)
                centres = {(round(x, 0), round(y, 0), round(z, 0)) for x, y, z in self.centre[near & panes]}
                self.assertGreaterEqual(len(centres), 4)                                 # windows on two storeys
                self.assertGreater(float(np.ptp([c[1] for c in centres])), 2.0)

    def test_ranges_and_blocks_carry_a_few_chimneys(self):
        stacks = np.all(self.attr == (4, mesh_castle.PLAIN), axis=1) & (self.normal[:, 1] > 0.99)
        tops = self.centre[stacks]
        eaves = world.levels()["platform"] + self.meta["castle"]["cls"][CASTLE_CLASSES.index("range")][0]
        tops = tops[tops[:, 1] > eaves]                                            # over the ranges' eaves
        clusters = {(round(x / 3), round(z / 3)) for x, _, z in tops}
        self.assertGreaterEqual(len(clusters), 10)
        self.assertLessEqual(len(clusters), 120)                                   # sparing


def top_down(tris, lo, cell, shape):
    """The highest surface over each cell of a grid starting at `lo` (x, z): -inf where none passes; triangles seen
    edge-on from above are left out."""
    out = np.full(shape, -np.inf)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    tris = tris[np.abs(n[:, 1]) > 0.05 * np.linalg.norm(n, axis=1)]
    for t in tris:
        (c0, r0), (c1, r1) = (np.floor((f(t[:, [0, 2]], axis=0) - lo) / cell).astype(int) for f in (np.min, np.max))
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, shape[1] - 1), min(r1, shape[0] - 1)
        if c1 < c0 or r1 < r0:
            continue
        x, z = np.meshgrid(lo[0] + (np.arange(c0, c1 + 1) + 0.5) * cell, lo[1] + (np.arange(r0, r1 + 1) + 0.5) * cell)
        a, b, c = t
        v0, v1 = (b - a)[[0, 2]], (c - a)[[0, 2]]
        den = v0[0] * v1[1] - v0[1] * v1[0]
        px, pz = x - a[0], z - a[2]
        v = (px * v1[1] - pz * v1[0]) / den
        w = (v0[0] * pz - v0[1] * px) / den
        inside = (v >= 0) & (w >= 0) & (v + w <= 1)
        if not inside.any():                      # smaller than a cell: its centre stands for it
            m = t.mean(axis=0)
            r, k = (int((m[2] - lo[1]) // cell), int((m[0] - lo[0]) // cell))
            if 0 <= r < shape[0] and 0 <= k < shape[1]:
                out[r, k] = max(out[r, k], t[:, 1].max())
            continue
        y = a[1] + v * (b - a)[1] + w * (c - a)[1]
        block = out[r0:r1 + 1, c0:c1 + 1]
        block[inside] = np.maximum(block[inside], y[inside])
    return out


class CastleMassingTests(unittest.TestCase):
    """The castle's masses meet cleanly: no lower mass's roof or wall pokes out beside a taller one's wall or a tower's
    drum as a sliver, and the hip roofs are planar slopes at their pitch, meeting in straight hips and ridges."""
    CELL = 0.25                     # m
    THIN = 2                        # cells: what a disc of this radius cannot pass through is a sliver
    RISE = 1.0                      # m: a taller wall rises at least this far over the sliver within REACH
    REACH = 4                       # cells

    @classmethod
    def setUpClass(cls):
        from scipy import ndimage

        cls.tris, cls.attr, cls.normal = world.payload_part("castle")
        cls.meta = world.data()["meta"]
        lo = cls.tris[:, :, [0, 2]].reshape(-1, 2).min(axis=0) - 2.0
        hi = cls.tris[:, :, [0, 2]].reshape(-1, 2).max(axis=0) + 2.0
        shape = tuple(np.ceil((hi - lo) / cls.CELL).astype(int)[::-1])
        r = cls.THIN
        disc = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) <= r + 0.01
        traced = {t["id"]: t for t in next(a for a in world.scene()["assets"] if a["id"] == "castle")["traced"]}
        plan, _ = mesh_castle.plan_fit(traced["plan-fit"]["plan_px"], {k: tuple(v) for k, v in cls.meta["named"].items()})
        ho = plan(*traced["heads-tower"]["plan_px"])
        zz, xx = np.mgrid[:shape[0], :shape[1]]
        head = np.hypot(lo[0] + (xx + 0.5) * cls.CELL - ho[0], lo[1] + (zz + 0.5) * cls.CELL - ho[1]) \
            < mesh_castle.HEAD_R + 2 * mesh_castle.STAIR_TURRET_R + 1.0
        cls.slivers, cls.proud = {}, {}
        for k, name in enumerate(CASTLE_CLASSES):
            mine = np.all(cls.attr == (4, mesh_castle.REAL_AUX + k), axis=1)
            if name == "low_wall" or not mine.any():
                continue
            own, others = top_down(cls.tris[mine], lo, cls.CELL, shape), top_down(cls.tris[~mine], lo, cls.CELL, shape)
            covered = np.isfinite(own) & (own >= others - 0.05)        # seen from above, not buried in other masonry
            thin = covered & ~ndimage.binary_opening(covered, disc)
            taller = ndimage.maximum_filter(others, size=2 * cls.REACH + 1) > own + cls.RISE
            parts, n = ndimage.label(thin & taller, np.ones((3, 3)))
            cls.slivers[name] = ndimage.sum(thin & taller, parts, range(1, n + 1)).max(initial=0) * cls.CELL ** 2
            if int(cls.meta["castle"]["cls"][k][1]) != mesh_castle.FLAT:
                continue
            # a parapet or merlon (standing over its own walk) beside taller masonry: one along a wall hidden in it;
            # the Head's tower (keep masonry) rightly has its parapet beside its own stair turrets
            raised = covered & (own > ndimage.minimum_filter(np.where(covered, own, np.inf), size=9) + 0.5) & ~head
            parts, n = ndimage.label(raised & ndimage.maximum_filter(others > own + cls.RISE, size=5), np.ones((3, 3)))
            cls.proud[name] = ndimage.sum(parts > 0, parts, range(1, n + 1)).max(initial=0) * cls.CELL ** 2

    def test_lower_masses_stop_at_the_drums_as_the_skyline_ranks_them(self):
        towers = self.meta["towers"]
        mass = (self.attr[:, 0] == 4) & (self.attr[:, 1] >= mesh_castle.REAL_AUX) & (self.attr[:, 1] < mesh_castle.REAL_AUX + 6)
        centre = self.tris.mean(axis=1)
        for name in ("GT", "RT", "AT"):
            i = min(range(len(towers)), key=lambda j: math.dist(towers[j][:2], self.meta["named"][name]))
            x, z, r = towers[i][:3]
            mine = np.all(self.attr == (4, mesh_castle.MESH_TOWER_AUX + i), axis=1)
            top = float(self.tris[mine][:, :, 1].max())
            radius = (mesh_castle.COLUMN if name == "AT" else 1.0) * r
            inside = mass & (np.hypot(centre[:, 0] - x, centre[:, 2] - z) < radius - 0.5) & (self.tris[:, :, 1].max(axis=1) < top - 3.0)
            a = 0.5 * np.linalg.norm(np.cross(self.tris[inside, 1] - self.tris[inside, 0], self.tris[inside, 2] - self.tris[inside, 0]), axis=1)
            with self.subTest(tower=name):
                self.assertLess(float(a.sum()), 1.0)

    def test_no_parapet_stands_against_a_taller_wall(self):
        for name, area in self.proud.items():           # the end of one meeting it is about a quarter of a square metre
            with self.subTest(mass=name):
                self.assertLess(area, 1.5)

    def test_no_lower_mass_shows_as_a_sliver_beside_a_taller_one(self):
        for name, area in self.slivers.items():          # the largest; masonry meeting at a slant leaves smaller ones
            with self.subTest(mass=name):
                self.assertLess(area, 3.0)

    def test_hip_roofs_slope_at_their_pitch(self):
        for k, (_, rt) in enumerate(self.meta["castle"]["cls"]):
            if int(rt) not in mesh_castle.ROOFS:
                continue
            pitch = mesh_castle.ROOFS[int(rt)][1]
            mine = np.all(self.attr == (4, mesh_castle.REAL_AUX + k), axis=1) & (self.normal[:, 1] > 0.2)
            t = self.tris[mine]
            area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
            ny = self.normal[mine, 1]
            slope = np.sqrt(np.maximum(1.0 - ny ** 2, 0.0)) / ny
            # the pitch (the castle base's own fall tilts it a little), a flat top, or the turrets' steep caps
            off = (slope > 0.05) & (np.abs(slope / pitch - 1.0) > 0.1) & (slope < 1.5)
            with self.subTest(mass=CASTLE_CLASSES[k]):
                self.assertLess(area[off].sum() / area.sum(), 0.02)

    def test_the_courts_stay_enclosed(self):
        from scipy import ndimage

        data = world.data()
        grid = Grid(self.meta["castle"])
        image, pieces = castle_piece_labels(data)
        owners = piece_owners(world.scene(), pieces)
        raw = (castle_class_sd(data) < 0).any(axis=-1) & np.isin(image, [p for p, a in owners.items() if a == "castle"])
        lo = self.tris[:, :, [0, 2]].reshape(-1, 2).min(axis=0) - 4.0
        hi = self.tris[:, :, [0, 2]].reshape(-1, 2).max(axis=0) + 4.0
        shape = tuple(np.ceil((hi - lo) / 0.5).astype(int)[::-1])
        free, _ = ndimage.label(~np.isfinite(top_down(self.tris, lo, 0.5, shape)))
        meshed = np.bincount(free.ravel()) * 0.25
        texel = float(np.prod(grid.texel_size()))
        courts, n = ndimage.label(~raw)
        sizes = np.bincount(courts.ravel()) * texel
        outside = courts[0, 0]
        for c in range(1, n + 1):
            if c == outside or sizes[c] < 300.0:
                continue
            i, j = np.unravel_index(np.argmax(ndimage.distance_transform_edt(courts == c)), raw.shape)
            x, z = grid.to_world(i, j)
            label = free[int((z - lo[1]) // 0.5), int((x - lo[0]) // 0.5)]
            with self.subTest(court=c, at=(round(float(x)), round(float(z)))):
                self.assertNotEqual(label, 0)
                self.assertLess(meshed[label], 1.3 * sizes[c] + 300.0)          # not opened to another court or out

    def test_the_north_towers_top_stands_clear_of_its_neighbours(self):
        towers = self.meta["towers"]
        i = min(range(len(towers)), key=lambda j: math.dist(towers[j][:2], self.meta["named"]["NT"]))
        x, z, r = towers[i][:3]
        mine = np.all(self.attr == (4, mesh_castle.MESH_TOWER_AUX + i), axis=1)
        pts = self.tris[mine].reshape(-1, 3)
        head = mesh_castle.OVERHANG * r
        foot = pts[np.abs(np.hypot(pts[:, 0] - x, pts[:, 2] - z) - head) < 0.2, 1].min()   # the overhanging top's foot
        rest = self.tris[~mine].reshape(-1, 3)
        reach = np.hypot(rest[:, 0] - x, rest[:, 2] - z)
        near = (reach > head + 0.05) & (reach < head + 1.0) & (rest[:, 1] > foot - 1.0)   # its own windows lie within
        self.assertFalse(near.any(), np.round(rest[near][:3], 1).tolist())   # a tower beside its neighbours, not fused

    def test_the_north_tower_stands_clear_of_the_block_beside_it_not_in_a_notch(self):
        # the block beside it was cut round it in a circular notch over its whole height, so its drum read as built into
        # that block, and the dome's flank hung over the block's roof (north-tower, round 3)
        towers = self.meta["towers"]
        i = min(range(len(towers)), key=lambda j: math.dist(towers[j][:2], self.meta["named"]["NT"]))
        x, z, r = towers[i][:3]
        head = mesh_castle.OVERHANG * r
        clear = max(rr for rr, _ in mesh_castle.dome(r)) + mesh_castle.NT_CLEAR_M      # from the dome's widest
        own = np.all(self.attr == (4, mesh_castle.MESH_TOWER_AUX + i), axis=1)
        top = float(self.tris[own][:, :, 1].max()) - mesh_castle.dome(r)[-1][1]              # under its finial
        rest = self.tris[~own].reshape(-1, 3)
        reach = np.hypot(rest[:, 0] - x, rest[:, 2] - z)
        rest, reach = rest[reach > head + 0.3], reach[reach > head + 0.3]               # past its own windows' reveals
        high = rest[:, 1] > top - mesh_castle.DROP_M - mesh_castle.NT_CLEAR_M             # as high as its overhang
        self.assertGreater(float(reach[high].min()), clear - mesh_castle.COURSE_OUT_M - 0.05,     # less a cornice's jut
                           np.round(rest[high][np.argmin(reach[high])], 1).tolist())
        bearing = np.degrees(np.arctan2(rest[:, 2] - z, rest[:, 0] - x))[reach < clear + 2.0]
        if len(bearing):                                                             # beside a face, not wrapped round
            ordered = np.sort(bearing)
            self.assertLess(360.0 - float(np.diff(np.append(ordered, ordered[0] + 360.0)).max()), 120.0)


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


class CastleFootingTests(unittest.TestCase):
    """The castle stands on its base. A wall's foot follows the ground under it, SINK_M below it, so where the ground
    under a wall falls away down the lake's cliff, its face hangs down the cliff as a pale wedge (an inspection defect:
    up to 39 m below the castle at the lake's edge). Under the castle's walls the ground is its platform or the skirt
    of lawn round it, at most a few metres lower. And the ground that holds their level stays clear of the lake: its
    edge is drawn between texels, where the depth of a wet texel and a dry one, read bilinearly, crosses zero, and
    pack.py keeps every dry depth below -1 m as -8 m, so beside a bank as tall as the cliff's sheer drop the lake's
    edge would stand metres above it, a dark tooth up the cliff's foot (as the castle-cliff fix first made it)."""

    HANG_M = 5.0                                    # the deepest a foot may reach below the castle's base
    REACH_M, LEVEL_M = 30.0, 1.0                    # within this of the castle, the lake's edge so near its level

    def test_no_castle_masonry_hangs_down_the_cliff(self):
        data = world.data()
        cbase, local = decode(data["tex"]["locCB"]), Grid(data["meta"]["local"])
        hung = {}
        for chunk in world.payload()["chunks"]:
            if chunk["asset"] != "castle":
                continue
            depth = sample(cbase, local, chunk["pos"][:, 0], chunk["pos"][:, 2]) - chunk["pos"][:, 1]
            for (x, _, z), d in zip(chunk["pos"][depth > self.HANG_M], depth[depth > self.HANG_M]):
                cell = (int(round(x / 10.0)) * 10, int(round(z / 10.0)) * 10)
                hung[cell] = max(hung.get(cell, 0.0), round(float(d), 1))
        self.assertEqual(hung, {}, "castle masonry hangs this far (m) below the castle's base, by 10 m cell (east, "
                                   "north)")

    def test_the_lake_lies_level_at_the_foot_of_the_castles_cliff(self):
        from scipy import ndimage as ndi

        data = world.data()
        bed, wd = decode(data["tex"]["locBed"]), decode(data["tex"]["locW"])
        local, lake = Grid(data["meta"]["local"]), float(data["meta"]["lake"])
        castle = np.concatenate([c["pos"][:, [0, 2]] for c in world.payload()["chunks"] if c["asset"] == "castle"])
        i, j = (np.clip(np.round(v).astype(int), 0, n - 1)
                for v, n in zip(local.to_texel(castle[:, 0], castle[:, 1]), (local.ny, local.nx)))
        near = np.zeros(bed.shape, bool)
        near[i, j] = True
        near = ndi.binary_dilation(near, iterations=round(self.REACH_M / 2.0))
        e, n = local.to_world(*np.mgrid[0:local.ny, 0:local.nx])
        harbour = Polygon(data["meta"]["extras"]["harbour"]["bounds"]).buffer(2.0)    # its recess: inside its shell
        near &= ~shapely.contains_xy(harbour, e, n)
        high = {}
        rows, cols = bed.shape
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):                # each pair of neighbouring texels, both ways
            a = (slice(max(-di, 0), rows - max(di, 0)), slice(max(-dj, 0), cols - max(dj, 0)))
            b = (slice(max(di, 0), rows - max(-di, 0)), slice(max(dj, 0), cols - max(-dj, 0)))
            for wet, dry in ((a, b), (b, a)):
                edge = (wd[wet] > 0) & (wd[dry] <= 0) & near[wet]
                t = wd[wet][edge] / (wd[wet][edge] - wd[dry][edge])
                rise = bed[wet][edge] + t * (bed[dry][edge] - bed[wet][edge]) - lake
                for x, z, r in zip(e[wet][edge][rise > self.LEVEL_M], n[wet][edge][rise > self.LEVEL_M],
                                   rise[rise > self.LEVEL_M]):
                    cell = (int(round(x / 10.0)) * 10, int(round(z / 10.0)) * 10)
                    high[cell] = max(high.get(cell, 0.0), round(float(r), 1))
        self.assertEqual(high, {}, "the lake's edge stands this far (m) above the lake, by 10 m cell (east, north)")

    PAVING = 3                                      # the surface classes' earth and paving (build_final.py)
    BLOT_M = 4.0                                    # how near a wall's foot a lone texel of paving blots it

    def test_no_lone_texel_of_paving_blots_the_walls_foot(self):
        # the grounds map draws the castle's outline in the earth colour, and read texel by texel it left chains of
        # single texels of paving round the castle, each a blot at its walls' foot (a hard class keeps a sharp edge,
        # frag4.glsl classMix): the footing broken into patches between the ground's triangles (inspection round 3,
        # tour/04). The paving at the walls is the courtyards' and the paths', in blocks two texels across or more
        from scipy import ndimage as ndi
        from scipy.spatial import cKDTree

        data = world.data()
        local, bed = Grid(data["meta"]["local"]), decode(data["tex"]["locBed"])
        paved = decode(data["tex"]["locN"])[..., 0] == self.PAVING
        lone = paved & ~ndi.binary_opening(paved, structure=np.ones((2, 2), bool))     # in no 2 x 2 block of paving
        tris, _, normal = world.payload_part("castle")
        upright = tris[np.abs(normal[:, 1]) < 0.05]
        order = np.argsort(upright[..., 1], axis=1)
        a, b = (upright[np.arange(len(upright)), order[:, k]] for k in (0, 1))   # each face's two lowest corners
        foot = np.abs(a[:, 1] - sample(bed, local, a[:, 0], a[:, 2])) < 3.0          # a face standing on the ground
        a, b = a[foot][:, [0, 2]], b[foot][:, [0, 2]]
        k = np.maximum(np.ceil(np.hypot(*(b - a).T)).astype(int), 1) + 1              # a point a metre or less apart
        seg = np.repeat(np.arange(len(a)), k)
        t = (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(k - 1, k)
        feet = cKDTree(a[seg] + t[:, None] * (b[seg] - a[seg]))
        e, n = local.to_world(*np.nonzero(lone))
        blots = np.isfinite(feet.query(np.column_stack([e, n]), distance_upper_bound=self.BLOT_M)[0])
        cells = {}
        for x, z in zip(e[blots], n[blots]):
            cell = (int(round(x / 10.0)) * 10, int(round(z / 10.0)) * 10)
            cells[cell] = cells.get(cell, 0) + 1
        self.assertEqual(cells, {}, "lone texels of paving at the castle walls' foot, by 10 m cell (east, north)")


if __name__ == "__main__":
    unittest.main()
