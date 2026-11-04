"""The integrated scene: every object the raymarcher drew has a mesh, so removing the raymarched objects from every pass
leaves nothing out. Walls, hedges and claimed building footprints each lie within REACH_M of a mesh seen from above, and
nearly all of each castle piece does; the set pieces (gate piers, tomb, Willow, train, ship, goals) each stand inside
or beside their asset. The only raymarched footprints with no mesh are the ones the manifest leaves unbuilt (its excluded
components) and the old sites of the village's moved cottages.

The ground follows the meshes too: the payload's footprint layer, which the page reads in place of the data's building
distances, marks the ground under every standing mesh as built and leaves the unbuilt footprints open."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import shapely
from shapely.geometry import MultiPoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assets_testing as at  # noqa: E402
import build_meshes  # noqa: E402
import mesh_footprint  # noqa: E402
import mesh_grounds  # noqa: E402
import mesh_levels  # noqa: E402
import mesh_thin  # noqa: E402
import mesh_village  # noqa: E402
import shaders  # noqa: E402
from materials import code  # noqa: E402
from mesh_geometry import Mesh, bar, concat, lathe  # noqa: E402
from mesh_source import (CHANNELS, Grid, building_labels, castle_piece_labels, decode, sample,  # noqa: E402
                         sd_bytes)

REACH_M = 4.0            # a raymarched surface counts as replaced when a mesh stands within this, seen from above: 2 m
                         # texels put a footprint's outer texel centres up to 1.4 m beyond its outline, and the castle's
                         # and the village's plans straighten traced outlines by up to 1 m (measured: 3.6 m at most)
CELL_M = 1.0             # the cells meshes are marked on
SAMPLE_M = 0.5           # triangles are sampled at least this densely, so a marked cell lies within it of the surface
WALL_IN = 1.05           # frag2.glsl's wallD: a boundary wall where its field is below -WALL_IN
HEDGE_M = 0.45           # wallD: a hedge where its field is below this
BUILDING_LEVEL = build_meshes.BUILDING_WALL_LEVEL       # bldD's walls
SET_PIECE_REACH_M = 15.0  # each set piece's asset reaches within this of the point the raymarcher drew it at
PIECE_SHARE = 0.95       # of a castle piece's texels within REACH_M of a mesh: the grounds' ruined and coped walls leave
                         # gaps on purpose, up to 10 m wide (the outer walls' pieces measured 99.1%)
TREES_FROM_M, TREES_FULL_M = mesh_footprint.TREES_FROM_M, mesh_footprint.TREES_FULL_M
RELIEF_FULL_M = 3.0      # frag1.glsl detailH: the micro-relief is whole from this far from a wall
TEXEL_REACH_M = 2.0 * math.sqrt(2.0)    # common.TEXEL_REACH_M: a bilinear lookup reads texel centres a diagonal away
# the assets the footprint layer leaves off on purpose: linear features (a strip cleared along them would read as a
# cutout), the train and the ship, the Willow (a tree among trees) and the street furniture (in streets, where no tree
# grows)
UNMARKED = {"track", "train", "ship", "grounds-walls", "field-boundaries", "bridges", "willow", "street-furniture"}
QUANTIZED = 1.02         # the payload's 16-bit positions move a level's measured deviation by under 1% (measured 0.7%)
STRUCTURE_M = 64.0       # the widest unit a family that builds its own levels makes: the village's widest building, 54 m
LONG_THIN_M = (0.5, 0.25)   # a long thin part, found apart from mesh_thin: at least this long, at most this across


def surface_points(chunks, glass=False):
    """Batches of world (east, north) points on the chunks' triangles seen from above, at most SAMPLE_M apart, glazing
    included only with `glass`."""
    for c in chunks:
        if (c.get("pass") == "glass" and not glass) or not len(c["idx"]):
            continue
        tri = c["pos"][c["idx"]][:, :, [0, 2]]
        edges = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]], 1)
        n = np.maximum(np.ceil(np.linalg.norm(edges, axis=2).max(axis=1) / SAMPLE_M), 1).astype(int)
        for k in np.unique(n):
            sel = tri[n == k]
            i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
            keep = i + j <= k
            u, v = i[keep] / k, j[keep] / k
            yield (sel[:, 0, None] + u[None, :, None] * (sel[:, 1] - sel[:, 0])[:, None]
                   + v[None, :, None] * (sel[:, 2] - sel[:, 0])[:, None]).reshape(-1, 2)


def footprint(chunks, x0, z0, nx, nz, glass=False):
    """Cells (CELL_M) that some triangle passes over (glazing only with `glass`), sampled at most SAMPLE_M apart."""
    marked = np.zeros((nz, nx), bool)
    for pts in surface_points(chunks, glass):
        cx = np.floor((pts[:, 0] - x0) / CELL_M).astype(int)
        cz = np.floor((pts[:, 1] - z0) / CELL_M).astype(int)
        ok = (cx >= 0) & (cz >= 0) & (cx < nx) & (cz < nz)
        marked[cz[ok], cx[ok]] = True
    return marked


class RaymarchedObjectsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.scene = at.data(), at.scene()
        cls.local, cls.castle = Grid(cls.data["meta"]["local"]), Grid(cls.data["meta"]["castle"])
        corners = np.array([cls.local.to_world(i, j) for i in (0, cls.local.ny) for j in (0, cls.local.nx)])
        cls.x0, cls.z0 = corners.min(axis=0) - 20.0
        x1, z1 = corners.max(axis=0) + 20.0
        nx, nz = int((x1 - cls.x0) / CELL_M) + 1, int((z1 - cls.z0) / CELL_M) + 1
        cls.reach = ndi.distance_transform_edt(~footprint(at.payload()["chunks"], cls.x0, cls.z0, nx, nz)) * CELL_M
        cls.shape = (nz, nx)

    def distance(self, grid, mask):
        """Metres from each texel of `mask` (on `grid`) to the nearest mesh, seen from above."""
        i, j = np.nonzero(mask)
        x, z = grid.to_world(i, j)
        cx = np.clip(np.floor((x - self.x0) / CELL_M).astype(int), 0, self.shape[1] - 1)
        cz = np.clip(np.floor((z - self.z0) / CELL_M).astype(int), 0, self.shape[0] - 1)
        return self.reach[cz, cx]

    def assert_replaced(self, name, grid, mask):
        d = self.distance(grid, mask)
        self.assertGreater(len(d), 0, f"no raymarched {name} to check")
        self.assertEqual(int((d > REACH_M).sum()), 0,
                         f"{(d > REACH_M).sum()} of {len(d)} raymarched {name} texels have no mesh within {REACH_M} m")

    def test_every_boundary_wall_and_hedge_has_a_mesh(self):
        a, b = decode(self.data["tex"]["locA"]), decode(self.data["tex"]["locB"])
        self.assert_replaced("boundary-wall", self.local, sd_bytes(b[..., CHANNELS["locB"].index("wall")]) < -WALL_IN)
        self.assert_replaced("hedge", self.local, sd_bytes(a[..., CHANNELS["locA"].index("hedge")]) < HEDGE_M)

    def test_every_castle_piece_has_its_mesh(self):
        image, pieces = castle_piece_labels(self.data)
        self.assertTrue(pieces)
        for piece in pieces:
            with self.subTest(piece=piece["id"], role=piece["role"]):
                d = self.distance(self.castle, image == piece["id"])
                self.assertGreater(float(np.mean(d <= REACH_M)), PIECE_SHARE)

    def test_every_claimed_building_footprint_has_a_mesh(self):
        # the manifest's excluded components and the moved cottages' old sites are left unbuilt on purpose
        labels, _ = building_labels(self.data)
        unbuilt = {c["id"] for c in self.scene["excluded"]["components"]} | set(mesh_village.MOVED)
        bsd = sd_bytes(decode(self.data["tex"]["locA"])[..., CHANNELS["locA"].index("bsd")])
        self.assert_replaced("building", self.local, (bsd < BUILDING_LEVEL) & ~np.isin(labels, list(unbuilt)))

    def test_every_set_piece_has_its_asset(self):
        meta, extras = self.data["meta"], self.data["meta"]["extras"]
        brg = math.radians(meta["pitch"]["brg"])
        axis = np.array([math.sin(brg), math.cos(brg)])
        pitch = np.array(meta["pitch"]["c"])
        places = {
            "gates": [extras["gates"][0], extras["gates"][1]],
            "tomb": [extras["tomb"]], "willow": [extras["willow"]],
            "train": [meta["train"]["carriages"][0], meta["train"]["carriages"][1], meta["train"]["engine"][0],
                      meta["train"]["engine"][1]],
            "ship": [extras["ship"]["a"], extras["ship"]["b"]],
            "pitch": [pitch + 61.0 * axis, pitch - 61.0 * axis],             # frag2.glsl's goalD: 61 m out on the axis
        }
        for asset, points in places.items():
            tris = at.payload_part(asset)[0].reshape(-1, 3)[:, [0, 2]]
            for p in points:
                with self.subTest(asset=asset, at=tuple(np.round(p, 1))):
                    self.assertLess(float(np.min(np.hypot(*(tris - np.asarray(p, float)).T))), SET_PIECE_REACH_M)


def under(asset):
    """World (east, north) points under an asset's meshes seen from above, glazing included, at most SAMPLE_M apart;
    for an enclosure, also every point of its convex hull on a SAMPLE_M lattice."""
    found = at.chunks(asset)
    pts = np.concatenate(list(surface_points(found, glass=True)))
    if mesh_footprint.POLICY[asset] == mesh_footprint.ENCLOSURE:
        hull = MultiPoint(np.concatenate([c["pos"][:, [0, 2]] for c in found])).convex_hull
        x0, z0, x1, z1 = hull.bounds
        X, Z = np.meshgrid(np.arange(x0, x1, SAMPLE_M), np.arange(z0, z1, SAMPLE_M))
        inside = shapely.contains_xy(hull, X, Z)
        pts = np.concatenate([pts, np.stack([X[inside], Z[inside]], 1)])
    return pts


class FootprintLayerTests(unittest.TestCase):
    """The footprint layer the page reads in place of the data's building distances (mesh_footprint): built ground under
    every standing mesh, open ground everywhere else, so no tree grows inside a mesh and no clearing opens away from
    one."""

    @classmethod
    def setUpClass(cls):
        cls.data, cls.scene = at.data(), at.scene()
        cls.grid = Grid(cls.data["meta"]["local"])
        a = decode(cls.data["tex"]["locA"])
        cls.trees, cls.road = (a[..., CHANNELS["locA"].index(k)] / 255.0 for k in ("trees", "road"))
        cls.data_sd = sd_bytes(a[..., CHANNELS["locA"].index("bsd")])
        entry = at.payload().get("footprint")
        cls.sd = sd_bytes(mesh_footprint.decoded(entry)) if entry else cls.data_sd     # what the page reads
        cls.standing = {asset for asset, kind in mesh_footprint.POLICY.items() if kind and at.chunks(asset)}
        corners = np.array([cls.grid.to_world(i, j) for i in (0, cls.grid.ny) for j in (0, cls.grid.nx)])
        cls.x0, cls.z0 = corners.min(axis=0) - 20.0
        x1, z1 = corners.max(axis=0) + 20.0
        nx, nz = int((x1 - cls.x0) / CELL_M) + 1, int((z1 - cls.z0) / CELL_M) + 1
        marked = np.zeros((nz, nx), bool)
        for asset in cls.standing:
            pts = under(asset)
            cx, cz = (np.floor((pts[:, k] - (cls.x0, cls.z0)[k]) / CELL_M).astype(int) for k in (0, 1))
            marked[cz, cx] = True
        cls.reach = ndi.distance_transform_edt(~marked) * CELL_M

    def page(self, field, pts):
        """The page's bilinear lookup of a texel field at world points."""
        return sample(field, self.grid, pts[:, 0], pts[:, 1])

    def distance(self, mask):
        """Metres from each texel of `mask` to the nearest standing mesh, seen from above."""
        x, z = self.grid.to_world(*np.nonzero(mask))
        return self.reach[np.floor((z - self.z0) / CELL_M).astype(int), np.floor((x - self.x0) / CELL_M).astype(int)]

    def test_every_manifest_asset_has_a_footprint_policy(self):
        self.assertEqual(set(mesh_footprint.POLICY), {a["id"] for a in self.scene["assets"]})
        self.assertLessEqual(set(mesh_footprint.POLICY.values()), {mesh_footprint.BUILT, mesh_footprint.ENCLOSURE, None})
        kinds = {kind: {a for a, k in mesh_footprint.POLICY.items() if k == kind} for kind in (None, mesh_footprint.ENCLOSURE)}
        self.assertEqual(kinds, {None: UNMARKED, mesh_footprint.ENCLOSURE: {"hut-garden"}})

    def test_the_hut_garden_is_built_ground_inside_its_fence(self):
        garden = build_meshes.Scene(self.data, self.scene).garden.buffer(-1.0)
        x0, z0, x1, z1 = garden.bounds
        X, Z = np.meshgrid(np.arange(x0, x1, SAMPLE_M), np.arange(z0, z1, SAMPLE_M))
        inside = shapely.contains_xy(garden, X, Z)
        sd = self.page(self.sd, np.stack([X[inside], Z[inside]], 1))
        self.assertGreater(len(sd), 1000)
        self.assertEqual(int((sd > TREES_FROM_M).sum()), 0, f"up to {sd.max():.2f} m from built ground inside the fence")

    def test_the_ground_under_every_standing_mesh_reads_as_built(self):
        self.assertTrue(self.standing)
        for asset in sorted(self.standing):
            with self.subTest(asset=asset):
                sd = self.page(self.sd, under(asset))
                self.assertEqual(int((sd > TREES_FROM_M).sum()), 0,
                                 f"{(sd > TREES_FROM_M).sum()} of {len(sd)} points under it read as open ground "
                                 f"(up to {sd.max():.2f} m from a wall)")

    def test_no_tree_grows_inside_a_standing_mesh(self):
        for asset in sorted(self.standing):
            with self.subTest(asset=asset):
                pts = under(asset)
                density = (self.page(self.trees, pts) * (1.0 - self.page(self.road, pts))
                           * mesh_footprint.vegetation(self.page(self.sd, pts)))
                self.assertEqual(int((density > 0).sum()), 0, f"trees of density up to {density.max():.2f} inside it")

    def test_clearings_open_only_beside_standing_meshes(self):
        thinned = self.sd < TREES_FULL_M
        self.assertTrue(thinned.any())
        far = self.distance(thinned) > TREES_FULL_M + TEXEL_REACH_M
        self.assertEqual(int(far.sum()), 0, f"{far.sum()} texels thin the trees with no standing mesh near them")

    def test_the_unbuilt_footprints_are_open_ground(self):
        labels, _ = building_labels(self.data)
        unbuilt = np.isin(labels, [c["id"] for c in self.scene["excluded"]["components"]] + list(mesh_village.MOVED))
        site = unbuilt & (self.distance(np.ones_like(unbuilt)).reshape(unbuilt.shape) > RELIEF_FULL_M + TEXEL_REACH_M)
        self.assertGreater(int(site.sum()), 100)
        self.assertEqual(int((self.sd[site] < RELIEF_FULL_M).sum()), 0)


class FootprintEncodingTests(unittest.TestCase):
    """The layer's encoding keeps every point under a mesh under the tree line however narrow the mesh: a 0.3 m rail
    running diagonally between texel centres and a 0.3 m pier midway between four of them read below TREES_FROM_M all
    over, and the ground beyond the clearing reads as open."""

    def test_narrow_meshes_read_as_built_all_over(self):
        grid = at.rotated_grid(40)
        a, b = np.array([-21.0, -17.0]), np.array([19.0, 14.0])
        side = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
        rail = np.array([a - 0.15 * side, b - 0.15 * side, b + 0.15 * side, a + 0.15 * side])
        mid = np.array(grid.to_world(30.5, 9.5))                     # equidistant from four texel centres
        pier = mid + 0.15 * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        quads = np.concatenate([rail, pier])
        corners = np.column_stack([quads[:, 0], np.zeros(8), quads[:, 1]])
        tris = corners[np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])]
        sd = sd_bytes(mesh_footprint.layer({"rail and pier": mesh_footprint.mark(tris, grid)}, grid))
        t = np.linspace(0.0, 1.0, 400)[:, None]
        on = np.concatenate([a + t * (b - a), mid + np.random.default_rng(0).uniform(-0.15, 0.15, (50, 2))])
        self.assertLessEqual(float(sample(sd, grid, on[:, 0], on[:, 1]).max()), TREES_FROM_M)
        off = (a + t * (b - a) + side * (TREES_FULL_M + 2.0 + TEXEL_REACH_M))[50:-50]
        self.assertGreaterEqual(float(sample(sd, grid, off[:, 0], off[:, 1]).min()), TREES_FULL_M)

    def test_a_silhouette_off_the_local_grid_marks_nothing(self):
        grid, n = at.rotated_grid(40), 4 * 40                         # raster cells per side
        inside = (60, 60, np.ones((12, 20), bool))
        beyond = [(-20, 60, np.ones((12, 20), bool)), (60, -30, np.ones((12, 20), bool)),     # just past each edge
                  (n + 5, 60, np.ones((12, 20), bool)), (60, n + 5, np.ones((12, 20), bool))]
        alone = mesh_footprint.layer({"inside": inside}, grid)
        self.assertTrue((alone < 255).any())
        with_beyond = mesh_footprint.layer({"inside": inside, **{f"beyond {k}": s for k, s in enumerate(beyond)}}, grid)
        np.testing.assert_array_equal(with_beyond, alone)
        open_ground = mesh_footprint.layer({f"beyond {k}": s for k, s in enumerate(beyond)}, grid)
        self.assertTrue((open_ground == 255).all())


class ThinPartTests(unittest.TestCase):
    """Thin set pieces (mesh_thin): a round part's radius comes from its own normals, exactly for its polygon; flat
    faces, unlisted materials and rounder surfaces get none; the shipped payload gives the goals' poles and hoops theirs
    and no other vertex any; and the page reads the radii in the encoder's unit."""
    GOLD = code("gold")
    POLE_M = 0.09                    # mesh_grounds.hoops' poles: 0.18 m across

    def pole(self, r):
        return lathe(0.0, 0.0, [(r, 0.0), (r, 15.0)], self.GOLD, 12, cap=False)

    def test_a_round_part_measures_its_own_radius(self):
        hoop = mesh_grounds.ring(0.0, 14.5, 0.0, (1.0, 0.0), 0.7, mesh_grounds.HOOP_TUBE_M, self.GOLD)
        for name, part, r in (("pole", self.pole(self.POLE_M), self.POLE_M), ("hoop", hoop, mesh_grounds.HOOP_TUBE_M)):
            with self.subTest(part=name):
                np.testing.assert_allclose(mesh_thin.radii(part, [self.GOLD]), r, rtol=1e-9)

    def test_only_listed_round_parts_get_a_radius(self):
        for name, part, listed in (("an unlisted material", self.pole(self.POLE_M), [code("dark-timber")]),
                                   ("a square bar", bar((0.0, 0.0, 0.0), (0.0, 10.0, 0.0), 0.1, self.GOLD), [self.GOLD]),
                                   ("a rounder drum", self.pole(1.5 * mesh_thin.MAX_M), [self.GOLD])):
            with self.subTest(part=name):
                self.assertFalse(mesh_thin.radii(part, listed).any())

    def test_the_shipped_payload_widens_the_goals_and_nothing_else(self):
        payload = build_meshes.decode_payload((ROOT / "build" / "meshes.js").read_text(encoding="ascii"))
        listed = {(asset, code(name)) for asset, names in mesh_thin.POLICY.items() for name in names}
        gold = []
        for c in payload["chunks"]:
            carrying = {(c["asset"], tuple(a)) for a in c["attr"][c["thin"] > 0].tolist()}
            self.assertLessEqual(carrying, listed, f"radii outside mesh_thin.POLICY in a {c['asset']} chunk")
            if c["asset"] == "pitch":
                gold.append(c["thin"][np.all(c["attr"] == self.GOLD, axis=1)])
        gold = np.concatenate(gold)
        self.assertEqual(sorted(np.unique(np.round(gold, 4)).tolist()), [mesh_grounds.HOOP_TUBE_M, self.POLE_M])

    def test_the_page_reads_radii_in_the_encoders_unit(self):
        self.assertEqual(shaders.Source().number("THIN_UNIT_M"), mesh_thin.UNIT_M)


def thin_parts(chunk, across_m):
    """(triangle indices, greatest extent across, centre, principal axis, length) of each part of a decoded chunk (its
    triangles joined through shared points) at most `across_m` across its principal axis."""
    pos, idx = chunk["pos"], chunk["idx"].astype(np.int64)
    if not len(idx):
        return
    _, point = np.unique(pos, axis=0, return_inverse=True)
    point = point.ravel()[idx]
    n = int(point.max()) + 1
    links = coo_matrix((np.ones(2 * len(idx)), (np.concatenate([point[:, 0], point[:, 1]]),
                                                np.concatenate([point[:, 1], point[:, 2]]))), shape=(n, n))
    part = connected_components(links, directed=False)[1][point[:, 0]]
    order = np.argsort(part, kind="stable")
    for tris in np.split(order, np.flatnonzero(np.diff(part[order])) + 1):
        p = pos[np.unique(idx[tris])]
        q = p - p.mean(axis=0)
        axes = np.linalg.svd(q, full_matrices=False)[2]
        extent = np.ptp(q @ axes.T, axis=0)
        if extent[1:].max() <= across_m:
            yield tris, float(extent[1:].max()), p.mean(axis=0), axes[0], float(extent[0])


class NarrowCasterTests(unittest.TestCase):
    """Thin casters in the sun-shadow layers (mesh_thin.widths): a long thin part, a stick or a round tube, measures its
    least width across its length, and a wide, short or round-ended part or a lattice's none; the shipped payload's
    casters hold every long thin part outside a lattice in their narrow runs, narrowest first, so that a layer whose
    texels are wider draws it as lines too."""
    GOLD, IRON, STONE = code("gold"), code("black-iron"), code("pale-stone")

    @staticmethod
    def widths(parts):
        m = concat(parts)
        found = mesh_thin.widths(m, np.round(m.pos / 1e-5).astype(np.int64))
        return np.split(found, np.cumsum([len(p.idx) for p in parts])[:-1])

    def test_a_long_thin_part_measures_its_least_width_across_its_length(self):
        # a square bar's faces each have their own vertices but meet at shared points: one part 0.1 m across, not four
        # flat strips of no width. The pole is 0.18 m across its corners, the hoop's tube round
        parts = {"square bar": (bar((0.0, 0.0, 0.0), (3.0, 8.0, 1.0), 0.1, self.IRON), 0.1),
                 "12-sided pole": (lathe(5.0, 0.0, [(0.09, 0.0), (0.09, 15.0)], self.GOLD, 12, cap=False),
                                   2 * 0.09 * math.cos(math.pi / 12)),
                 "hoop": (mesh_grounds.ring(10.0, 14.5, 0.0, (1.0, 0.0), 0.7, mesh_grounds.HOOP_TUBE_M, self.GOLD),
                          2 * mesh_grounds.HOOP_TUBE_M)}
        for (name, (_, want)), found in zip(parts.items(), self.widths([m for m, _ in parts.values()])):
            with self.subTest(part=name):
                np.testing.assert_allclose(found, want, rtol=1e-6)

    def test_wide_short_and_round_ended_parts_are_not_narrow(self):
        ball = [(0.0, 0.0), (0.14, 0.06), (0.2, 0.2), (0.14, 0.34), (0.0, 0.4)]
        parts = {"wall": build_meshes.prism(shapely.box(0.0, 0.0, 5.0, 0.3), 0.0, 3.0, self.STONE),
                 "block": build_meshes.prism(shapely.box(10.0, 0.0, 10.4, 0.4), 0.0, 0.5, self.STONE),
                 "drum": lathe(20.0, 0.0, [(0.4, 0.0), (0.4, 3.0)], self.STONE, 24),
                 "ball": lathe(30.0, 0.0, ball, self.GOLD, 12, cap=False)}
        for name, found in zip(parts, self.widths(list(parts.values()))):
            with self.subTest(part=name):
                self.assertTrue(np.isinf(found).all())

    def test_a_lattice_denser_than_two_near_layer_texels_is_not_narrow(self):
        # a gate's pickets 0.13 m apart: lines a near-layer texel wide would merge their shadows into a solid one. Spread
        # 0.5 m apart they are narrow, and so is a rail crossing posts, which is no lattice
        row = lambda pitch: [bar((k * pitch, 0.0, 0.0), (k * pitch + 0.5, 3.0, 0.0), 0.03, self.IRON) for k in range(5)]
        fence = [bar((k * 2.0, 0.0, 5.0), (k * 2.0, 1.2, 5.0), 0.1, self.IRON) for k in range(3)] + [
            bar((-0.1, 1.0, 5.05), (4.1, 1.0, 5.05), 0.06, self.IRON)]
        for name, parts, narrow in (("pickets 0.13 m apart", row(0.13), False), ("pickets 0.5 m apart", row(0.5), True),
                                    ("a rail across posts", fence, True)):
            with self.subTest(parts=name):
                self.assertEqual({bool(np.isfinite(w).all()) for w in self.widths(parts)}, {narrow})

    def test_the_shipped_casters_hold_every_long_thin_part_in_their_narrow_runs(self):
        # the static layers draw each group's caster (level 3) or its full level, the near layer its full level: there,
        # every long thin part lies in its chunk's narrow runs, no wider than it measures, unless a parallel one runs
        # alongside it within mesh_thin.CROWD_M (a lattice, the gates' pickets). Among them are the goals' poles (PS 8:
        # "three golden poles"), the street lamps, the gates' frames, the ship's stays, the Shack's fence, the glazing
        # bars and the inn signs' brackets; the goals' hoops are round tubes, narrow too
        payload = build_meshes.decode_payload((ROOT / "build" / "meshes.js").read_text(encoding="ascii"))
        found = set()
        for c in payload["chunks"]:
            runs = c.get("narrow", [])
            with self.subTest(asset=c["asset"], group=c["group"], level=c["level"]):
                if runs:
                    self.assertIn(c["level"], (0, 3))
                    self.assertNotEqual(c.get("pass"), "glass")
                    self.assertEqual([w for _, w in runs], sorted({w for _, w in runs}))
                    self.assertTrue(all(n > 0 for n, _ in runs))
                if c["level"] not in (0, 3) or c.get("pass") == "glass":
                    continue
                width = np.full(c["tris"], np.inf)
                width[:sum(n for n, _ in runs)] = np.repeat([w / 1000 for _, w in runs], [n for n, _ in runs])
                parts = list(thin_parts(c, mesh_thin.CAST_M))
                for k, (tris, across, centre, axis, length) in enumerate(parts):
                    if length < LONG_THIN_M[0] or across > LONG_THIN_M[1]:
                        continue
                    if np.isfinite(width[tris]).all():
                        self.assertLessEqual(width[tris].max(), across + 0.001)
                        found.add((c["asset"], tuple(c["attr"][c["idx"][tris[0], 0]].tolist())))
                        continue
                    alongside = []                    # a lattice: thin parts parallel alongside it, how far across
                    for j, (_, _, other, a, _) in enumerate(parts):
                        d = other - centre
                        if j != k and abs(a @ axis) > mesh_thin.PARALLEL and abs(d @ axis) < length / 2:
                            alongside.append(np.linalg.norm(d - (d @ axis) * axis))
                    self.assertLess(min(alongside, default=np.inf), mesh_thin.CROWD_M, "a long thin part left out")
                if c["asset"] == "pitch":
                    gold = np.all(c["attr"][c["idx"][:, 0]] == self.GOLD, axis=1)
                    self.assertLessEqual(width[gold].max(initial=0.0), 2 * ThinPartTests.POLE_M)
        for asset, name in (("pitch", "gold"), ("street-furniture", "black-iron"), ("gates", "black-iron"),
                            ("ship", "rigging"), ("shrieking-shack", "boarding"), ("greenhouses-north", "building-glass"),
                            ("village-house", "black-iron"), ("hogs-head", "rust-iron")):
            self.assertIn((asset, code(name)), found)


class LevelTests(unittest.TestCase):
    """Levels of detail (mesh_levels): every group keeps its full level, each coarser level costs less than the one
    before and passes its guards from its switch distance on, the village and the Willow use their lanes' own levels,
    thin parts are whole at every level and glazing has its full level only."""

    @classmethod
    def setUpClass(cls):
        payload = at.levelled()
        cls.groups = {}
        for c in payload["chunks"]:
            cls.groups.setdefault(c["group"], {}).setdefault(c["level"], []).append(c)
        cls.switch = [g["switch"] for g in payload["groups"]]
        cls.family = {a["id"]: a["family"] for a in at.scene()["assets"]}

    @staticmethod
    def mesh(chunks):
        out, n = [], 0
        for c in chunks:
            out.append(Mesh(c["pos"], c["nrm"], c["attr"], c["idx"].astype(np.uint32) + n))
            n += len(c["pos"])
        return Mesh(np.concatenate([m.pos for m in out]), np.concatenate([m.nrm for m in out]),
                    np.concatenate([m.attr for m in out]), np.concatenate([m.idx for m in out]))

    def test_every_group_keeps_its_full_level_and_each_coarser_level_costs_less(self):
        for g, levels in self.groups.items():
            with self.subTest(group=g, asset=levels[0][0]["asset"] if 0 in levels else None):
                self.assertIn(0, levels)
                previous = sum(c["tris"] for c in levels[0])
                for level in (1, 2):
                    if level in levels:
                        tris = sum(c["tris"] for c in levels[level])
                        self.assertLessEqual(tris, mesh_levels.SAVING * previous)
                        self.assertIsNotNone(self.switch[g][level - 1])
                        previous = tris

    def test_a_group_of_a_familys_own_levels_spans_a_tile_and_a_structure_at_most(self):
        # build_meshes.unit_groups files every level of a unit with the group of its full level's tile, so such a group
        # spans at most a tile and its widest unit: one unit of every one of a kind across the village (the High Street's
        # lamps, the gardens' walls) drags them all into one group, whose coarser levels then fail their guards
        for g, levels in self.groups.items():
            asset = levels[0][0]["asset"]
            if self.family[asset] not in mesh_levels.OWN_LEVELS:
                continue
            pos = np.concatenate([c["pos"] for chunks in levels.values() for c in chunks])
            with self.subTest(group=g, asset=asset):
                self.assertLessEqual(float(np.ptp(pos[:, [0, 2]], axis=0).max()), build_meshes.TILE_M + STRUCTURE_M)

    def test_coarser_levels_pass_their_guards_from_their_switch_distances(self):
        # the silhouette guard from the east, the north and above at the switch distance and twice it; the lanes'
        # levels within the deviation guard there too (the simplifier's levels within its budget, mesh_levels)
        for g, levels in self.groups.items():
            asset = levels[0][0]["asset"]
            for level in (1, 2):
                if level not in levels:
                    continue
                full, coarse, d = self.mesh(levels[0]), self.mesh(levels[level]), self.switch[g][level - 1]
                with self.subTest(group=g, asset=asset, level=level, switch=d):
                    for far in (d, 2 * d):
                        self.assertIs(mesh_levels.alike(full, coarse, far * mesh_levels.PIX_REF), True)
                    if self.family[asset] in mesh_levels.OWN_LEVELS:     # the payload's quantized levels
                        self.assertLessEqual(mesh_levels.deviation(full, coarse),
                                             QUANTIZED * mesh_levels.DEVIATION_PX * mesh_levels.PIX_REF * d)

    def test_the_village_and_the_willow_use_their_lanes_levels(self):
        found = {f: {k: sum(c["tris"] for g in self.groups.values() for c in g.get(k, [])
                            if self.family[c["asset"]] == f) for k in range(4)} for f in ("village", "willow")}
        for f in found:
            with self.subTest(family=f):
                self.assertGreater(found[f][3], 0, "no shadow caster: only the lanes' levels have one")
                self.assertGreater(found[f][1], 0)

    def test_thin_parts_are_whole_at_every_level(self):
        for g, levels in self.groups.items():
            if levels[0][0]["asset"] not in mesh_thin.POLICY:
                continue
            thin = {k: np.sort(np.concatenate([c["thin"][c["thin"] > 0] for c in chunks])) for k, chunks in levels.items()}
            for k in thin:
                with self.subTest(group=g, level=k):
                    np.testing.assert_array_equal(thin[k], thin[0])

    def test_glazing_has_its_full_level_only(self):
        levels = {c["level"] for g in self.groups.values() for chunks in g.values() for c in chunks if c.get("pass") == "glass"}
        self.assertEqual(levels, {0})


if __name__ == "__main__":
    unittest.main()
