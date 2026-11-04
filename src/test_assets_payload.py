"""The mesh payload: footprint extraction keeps its topology, the encoding round-trips safely and deterministically, and the
checked-in payload carries every claimed asset. That it is reproducible without the part cache is test_uncached_build's
check, which the trunk keeper runs at merges."""

import base64
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assets_testing as world  # noqa: E402
import build_meshes  # noqa: E402
import meshopt  # noqa: E402
import mesh_thin  # noqa: E402
from materials import code  # noqa: E402
from mesh_geometry import bar, lathe  # noqa: E402
from mesh_source import field_polygons  # noqa: E402
from assets_testing import GRID_DEG, rotated_grid, box_field, triangle_areas  # noqa: E402


class FootprintTopologyTests(unittest.TestCase):
    def test_ring_footprint_keeps_its_courtyard(self):
        grid = rotated_grid()
        field = np.maximum(box_field(grid, (0.0, 0.0), 10.0, 10.0), -box_field(grid, (0.0, 0.0), 4.0, 4.0))
        polygons = field_polygons(field, grid, 0.0)
        self.assertEqual([len(p.interiors) for p in polygons], [1])
        ring = polygons[0]
        self.assertAlmostEqual(ring.area, 20 * 20 - 8 * 8, delta=0.05 * 336)
        mesh = build_meshes.prism(ring, 100.0, 110.0, (4, 0))
        caps = np.flatnonzero(mesh.nrm[mesh.idx[:, 0], 1] > 0.99)
        self.assertAlmostEqual(triangle_areas(mesh, caps).sum(), ring.area, places=6)
        hole = Polygon(ring.interiors[0])
        centres = mesh.pos[mesh.idx[caps]].mean(axis=1)
        self.assertFalse(any(hole.contains(Point(x, z)) for x, _, z in centres))

    def test_detached_pieces_stay_separate(self):
        grid = rotated_grid()
        c, s = math.cos(math.radians(GRID_DEG)), math.sin(math.radians(GRID_DEG))
        # two blocks whose corner texels touch diagonally, as village wings meet their parents; the 2.8 m half
        # size keeps every texel centre off the block edges
        field = np.minimum(box_field(grid, (-3.0 * c, -3.0 * s), 2.8, 2.8),
                           box_field(grid, (3.0 * c + 5.0 * s, 3.0 * s - 5.0 * c), 2.8, 2.8))
        labels, count = ndi.label(field < 0.0)
        self.assertEqual(count, 2)
        pieces = [field_polygons(field, grid, 0.0, keep=labels == k) for k in (1, 2)]
        self.assertEqual([len(p) for p in pieces], [1, 1])
        self.assertLess(pieces[0][0].intersection(pieces[1][0]).area, 1e-6)

    def test_extraction_keeps_registration(self):
        grid, centre = rotated_grid(), (13.0, -7.0)
        (polygon,) = field_polygons(box_field(grid, centre, 5.0, 3.0), grid, 0.0)
        self.assertLess(math.dist(polygon.centroid.coords[0], centre), 0.1)
        # bilinear 2 m sampling chamfers the corners exactly as the raymarcher draws them
        self.assertAlmostEqual(polygon.area, 60.0, delta=0.1 * 60)
        corners = np.array(polygon.minimum_rotated_rectangle.exterior.coords[:3])
        edges = np.diff(corners, axis=0)
        long_edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        bearing = math.degrees(math.atan2(long_edge[1], long_edge[0])) % 180.0
        self.assertAlmostEqual(bearing, GRID_DEG, delta=1.0)

    def test_prism_walls_face_out_of_the_solid(self):
        grid = rotated_grid()
        ring = field_polygons(np.maximum(box_field(grid, (0.0, 0.0), 10.0, 10.0),
                                         -box_field(grid, (0.0, 0.0), 4.0, 4.0)), grid, 0.0)[0]
        # shapely's set operations (crenellated parapets, split pieces) hand back clockwise outlines
        for winding in (1.0, -1.0):
            with self.subTest(winding=winding):
                mesh = build_meshes.prism(orient(ring, winding), 100.0, 110.0, (4, 0))
                walls = np.flatnonzero(np.abs(mesh.nrm[mesh.idx[:, 0], 1]) < 1e-6)
                self.assertTrue(len(walls))
                centres = mesh.pos[mesh.idx[walls]].mean(axis=1)[:, [0, 2]]
                normals = mesh.nrm[mesh.idx[walls, 0]][:, [0, 2]]
                for centre, normal in zip(centres, normals):
                    self.assertFalse(ring.contains(Point(*(centre + 0.05 * normal))))
                    self.assertTrue(ring.contains(Point(*(centre - 0.05 * normal))))


class MeshPayloadTests(unittest.TestCase):
    SCENE = {"assets": [{"id": "hut", "components": [{"id": 22, "at": [0.0, 0.0]}]}, {"id": "tomb"}, {"id": "pitch"}]}
    GOLD = code("gold")

    def hut(self):
        grid, ground = rotated_grid(), world.datum()
        return build_meshes.prism(field_polygons(box_field(grid, (2.0, 3.0), 5.0, 3.0), grid, 0.0)[0],
                                  ground, ground + 3.0, (5, 3))

    def test_payload_drops_zero_area_triangles(self):
        # a potted plant: a lathe whose profile starts and ends on its axis collapses half its pole quads
        y = world.datum() + 3.0
        plant = lathe(2.0, 3.0, [(0.0, y), (0.22, y + 0.12), (0.27, y + 0.3), (0.15, y + 0.46), (0.0, y + 0.52)], (6, 0), 5,
                      cap=False)
        corners = plant.pos[plant.idx]
        solid = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) > 1e-12
        self.assertFalse(solid.all())
        (chunk,) = self.full_level(build_meshes.encode_payload({"hut": [plant]}, self.SCENE))
        self.assertEqual(chunk["tris"], solid.sum())
        corners = chunk["pos"][chunk["idx"]]
        self.assertGreater(np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1).min(), 0.0)
        self.assertEqual(chunk["verts"], len(np.unique(plant.idx[solid])))

    def test_payload_drops_triangles_that_quantization_collapses(self):
        # a 0.3 mm wide sliver keeps its area in metres but not at the chunk box's 16-bit quantum (about 0.15 mm here)
        ground = world.datum()
        pos = np.array([[0.0, ground, 0.0], [10.0, ground, 0.0], [0.0, ground + 10.0, 1.0],
                        [1.0, ground + 1.0, 0.5], [2.0, ground + 1.0, 0.5], [1.5, ground + 1.0 + 3e-6, 0.5]])
        nrm = np.tile([0.0, 0.0, 1.0], (6, 1))
        sliver = build_meshes.Mesh(pos, nrm, np.tile([5, 3], (6, 1)).astype(np.uint8), np.array([[0, 1, 2], [3, 4, 5]], np.uint32))
        a, b, c = pos[3:]
        self.assertGreater(np.linalg.norm(np.cross(b - a, c - a)) / 2, 1e-7)    # a triangle in metres, kept before quantizing
        (chunk,) = build_meshes.decode_payload(build_meshes.encode_payload({"hut": [sliver]}, self.SCENE))["chunks"]
        self.assertEqual((chunk["tris"], chunk["verts"]), (1, 3))          # the sliver and the vertices only it used go
        corners = chunk["pos"][chunk["idx"]]
        self.assertGreater(np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1).min(), 0.0)

    def test_payload_round_trips_deterministically(self):
        mesh = self.hut()
        text = build_meshes.encode_payload({"hut": [mesh]}, self.SCENE)
        self.assertEqual(text, build_meshes.encode_payload({"hut": [mesh]}, self.SCENE))
        (chunk,) = self.full_level(text)
        self.assertEqual(chunk["asset"], "hut")
        np.testing.assert_array_equal(chunk["idx"], mesh.idx)
        np.testing.assert_array_equal(chunk["attr"], mesh.attr)
        extent = mesh.pos.max(axis=0) - mesh.pos.min(axis=0)
        np.testing.assert_array_less(np.abs(chunk["pos"] - mesh.pos).max(axis=0), extent / 65535 + 1e-9)
        np.testing.assert_allclose(chunk["nrm"], mesh.nrm, atol=0.02)

    @staticmethod
    def full_level(text):
        return [c for c in build_meshes.decode_payload(text)["chunks"] if c["level"] == 0]

    def raw_streams(self, meshes):
        """The raw streams as the encoder quantizes them, before any compression: what decoding must give back."""
        encoded = build_meshes.encode_assets(meshes)
        return {k: b"".join(encoded[a]["streams"][k] for a in sorted(encoded)) for k in build_meshes.RAW}

    def several_chunks(self):
        y = world.datum()
        plant = lathe(2.0, 3.0, [(0.0, y), (0.22, y + 0.12), (0.27, y + 0.3), (0.15, y + 0.46), (0.0, y + 0.52)], (6, 0), 40)
        far = build_meshes.prism(Polygon([(300.0, 0.0), (304.0, 0.0), (304.0, 3.0), (300.0, 3.0)]), y, y + 3.0, (5, 3))
        pane = build_meshes.prism(Polygon([(0.0, 9.0), (4.0, 9.0), (4.0, 9.1), (0.0, 9.1)]), y, y + 2.0, (build_meshes.MAT_GLASS, 0))
        pole = lathe(40.0, 40.0, [(0.09, y), (0.09, y + 14.0)], self.GOLD, 12, cap=False)
        return {"hut": [self.hut(), plant, pane], "tomb": [far], "pitch": [pole]}

    def test_streams_decode_to_exactly_what_the_encoder_quantized(self):
        meshes = self.several_chunks()
        payload = json.loads(build_meshes.strip_js(build_meshes.encode_payload(meshes, self.SCENE)))
        self.assertGreater(len(payload["chunks"]), 2)
        raw = self.raw_streams(meshes)
        self.assertTrue(any(raw["thin"]), "no vertex carries a thin radius, so its stream goes untested")
        self.assertEqual(build_meshes.decode_streams(payload), raw)

    def test_thin_parts_carry_their_radius_through_the_payload(self):
        # the same golden pole: the pitch lists its gold as thin (mesh_thin.POLICY), the hut does not
        y = world.datum()
        pole = lambda x: lathe(x, 40.0, [(0.09, y), (0.09, y + 14.0)], self.GOLD, 12, cap=False)
        text = build_meshes.encode_payload({"pitch": [pole(40.0)], "hut": [self.hut(), pole(60.0)]}, self.SCENE)
        chunks = {c["asset"]: c for c in build_meshes.decode_payload(text)["chunks"]}
        np.testing.assert_allclose(chunks["pitch"]["thin"], 0.09, atol=mesh_thin.UNIT_M / 2)
        self.assertFalse(chunks["hut"]["thin"].any())

    def test_narrow_parts_lead_their_chunk_narrowest_first(self):
        # a hut with a black-iron bar 0.1 m square and a 12-sided pole 0.1 m across its corners: the chunk's triangles
        # start with the pole's (the narrower across its flats), then the bar's, then the hut's in their own order, and
        # its narrow runs count them with their widths in millimetres (mesh_thin.widths)
        y, iron = world.datum(), code("black-iron")
        hut = self.hut()
        rod = bar((2.0, y, 3.0), (2.0, y + 6.0, 3.0), 0.1, iron)
        pole = lathe(4.0, 3.0, [(0.05, y), (0.05, y + 6.0)], iron, 12, cap=False)
        (chunk,) = self.full_level(build_meshes.encode_payload({"hut": [hut, rod, pole]}, self.SCENE))
        across_flats = 2 * 0.05 * math.cos(math.pi / 12)
        self.assertEqual(chunk["narrow"], [[len(pole.idx), round(1000 * across_flats)], [len(rod.idx), 100]])
        first, second = len(pole.idx), len(pole.idx) + len(rod.idx)
        np.testing.assert_array_equal(chunk["idx"][:first], pole.idx + len(hut.pos) + len(rod.pos))
        np.testing.assert_array_equal(chunk["idx"][first:second], rod.idx + len(hut.pos))
        np.testing.assert_array_equal(chunk["idx"][second:], hut.idx)

    def test_payload_holds_meshoptimizer_streams(self):
        payload = json.loads(build_meshes.strip_js(build_meshes.encode_payload(self.several_chunks(), self.SCENE)))
        self.assertEqual(payload["version"], 4)
        first = {k: zlib.decompress(base64.b64decode(v))[0] for k, v in payload["streams"].items()}
        self.assertEqual((first["pos"], first["idx"]), (CodecTests.VERTEX_V1, CodecTests.INDEX_SEQUENCE_V1))

    def test_old_and_damaged_payloads_are_rejected(self):
        meshes = {"hut": [self.hut()]}
        good = json.loads(build_meshes.strip_js(build_meshes.encode_payload(meshes, self.SCENE)))
        old = {**good, "version": 1,
               "streams": {k: base64.b64encode(zlib.compress(v, 9)).decode("ascii") for k, v in self.raw_streams(meshes).items()}}
        def recoded(key, change):
            body = json.loads(json.dumps(good))
            body["streams"][key] = base64.b64encode(zlib.compress(change(zlib.decompress(base64.b64decode(body["streams"][key]))))).decode("ascii")
            return body
        cases = {"the version-1 format": old, "the version-2 format": {**good, "version": 2},
                 "truncated positions": recoded("pos", lambda s: s[:-4]),
                 "truncated indices": recoded("idx", lambda s: s[:-2]), "a normal short": recoded("nrm", lambda s: s[:-2]),
                 "a (material, aux) pair short": recoded("attr", lambda s: s[:-2])}
        for case, body in cases.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                build_meshes.decode_payload(build_meshes.wrap_js(body))

    def test_payload_rejects_unsafe_geometry(self):
        mesh = self.hut()
        cases = {
            "index past the vertices": {"hut": [mesh._replace(idx=np.where(mesh.idx == 0, len(mesh.pos), mesh.idx))]},
            "non-finite position": {"hut": [mesh._replace(pos=np.where(mesh.pos == mesh.pos.max(), np.nan, mesh.pos))]},
            "asset missing from the manifest": {"hut": [mesh], "phantom": [mesh]},
            "claimed footprint without geometry": {"tomb": [mesh]},
        }
        for case, meshes in cases.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                build_meshes.encode_payload(meshes, self.SCENE)
        payload = json.loads(build_meshes.strip_js(build_meshes.encode_payload({"hut": [mesh]}, self.SCENE)))
        payload["chunks"][0]["verts"] -= 1
        with self.assertRaises(ValueError):
            build_meshes.decode_payload(build_meshes.wrap_js(payload))


class CodecTests(unittest.TestCase):
    """meshoptimizer's codecs, run from the vendored files, give back exactly what they were given."""
    RNG = np.random.default_rng(7)
    VERTEX_V1, INDEX_SEQUENCE_V1 = 0xA1, 0xD1        # the codecs' header bytes for bitstream version 1

    def positions(self, n):
        """u16 positions padded to 8-byte vertices, as the payload's position stream is: a random walk, like a mesh."""
        walk = np.cumsum(self.RNG.integers(-40, 41, (n, 3)), axis=0) % 65536
        out = np.zeros((n, 4), "<u2")
        out[:, :3] = walk
        return out.tobytes()

    def test_vendored_files_match_their_pins(self):
        for name in meshopt.PINNED:
            with self.subTest(file=name):
                meshopt.vendored(name)
        with tempfile.TemporaryDirectory() as folder:
            for name in meshopt.PINNED:
                data = bytearray((meshopt.VENDOR / name).read_bytes())
                data[len(data) // 2] ^= 1
                (Path(folder) / name).write_bytes(bytes(data))
            with mock.patch.object(meshopt, "VENDOR", Path(folder)):
                for name in meshopt.PINNED:
                    with self.subTest(tampered=name), self.assertRaises(ValueError):
                        meshopt.vendored(name)

    def test_vertices_round_trip_exactly_and_deterministically(self):
        for n in (0, 1, 3, 257, 20000):
            with self.subTest(vertices=n):
                raw = self.positions(n)
                packed = meshopt.encode_vertices(raw, 8)
                self.assertEqual(packed[0], self.VERTEX_V1)
                self.assertEqual(packed, meshopt.encode_vertices(raw, 8))
                self.assertEqual(meshopt.decode_vertices(packed, n, 8), raw)
        noise = self.RNG.integers(0, 256, 4 * 1000, dtype=np.uint8).tobytes()
        self.assertEqual(meshopt.decode_vertices(meshopt.encode_vertices(noise, 4), 1000, 4), noise)

    def test_indices_round_trip_exactly_without_rotating_triangles(self):
        # triangles whose smallest index comes last: the triangle codec would rotate them, the sequence codec must not
        tris = np.sort(self.RNG.integers(0, 65536, (5000, 3)), axis=1)[:, ::-1]
        for idx in (np.zeros((0, 3), "<u2"), tris.astype("<u2"), np.arange(30000).reshape(-1, 3).astype("<u2")):
            with self.subTest(indices=idx.size):
                packed = meshopt.encode_indices(idx)
                self.assertEqual(packed[0], self.INDEX_SEQUENCE_V1)
                self.assertEqual(packed, meshopt.encode_indices(idx))
                self.assertEqual(meshopt.decode_indices(packed, idx.size), idx.tobytes())
        with self.assertRaises(ValueError):
            meshopt.encode_indices(np.array([0, 1, 65536]))

    def test_malformed_streams_and_wrong_index_counts_raise(self):
        # the vertex codec cannot vouch for its count (see meshopt.decode_vertices); the index codec can, smooth or not
        raw, smooth = self.positions(300), np.repeat(np.arange(300), 3).astype("<u2")
        vertices, indices = meshopt.encode_vertices(raw, 8), meshopt.encode_indices(smooth)
        cases = {"truncated vertices": lambda: meshopt.decode_vertices(vertices[:-5], 300, 8),
                 "truncated indices": lambda: meshopt.decode_indices(indices[:-3], 900),
                 "fewer indices than encoded": lambda: meshopt.decode_indices(indices, 897),
                 "more indices than encoded": lambda: meshopt.decode_indices(indices, 903),
                 "a stride that is not a multiple of 4": lambda: meshopt.encode_vertices(raw, 6)}
        for case, call in cases.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                call()

    def test_the_simplifier_keeps_open_borders_and_reports_its_error_in_metres(self):
        # a gently bowed 10 m sheet of 800 triangles, its bow 5 cm high: within 1 mm it keeps some of its curve, within
        # 0.1 m it may flatten it (budgets in metres, ErrorAbsolute), and either way every vertex on its open border
        # stays (LockBorder)
        n = 21
        x, z = (a.ravel() for a in np.meshgrid(np.linspace(0.0, 10.0, n), np.linspace(0.0, 10.0, n)))
        pos = np.column_stack([x, 0.002 * (x - 5.0) ** 2, z])
        k = np.arange(n * n).reshape(n, n)[:-1, :-1].ravel()
        tris = np.concatenate([np.column_stack([k, k + n, k + 1]), np.column_stack([k + 1, k + n, k + n + 1])])
        border = set(np.flatnonzero((x == 0.0) | (x == 10.0) | (z == 0.0) | (z == 10.0)).tolist())
        found = {}
        for budget in (0.001, 0.1):
            kept, error = meshopt.simplify(tris, pos, 0, budget, ("LockBorder", "ErrorAbsolute"))
            found[budget] = len(kept)
            with self.subTest(budget=budget):
                self.assertLessEqual(error, budget)
                self.assertLessEqual(border, set(np.unique(kept).tolist()))
        self.assertGreater(found[0.001], found[0.1])
        self.assertLess(found[0.1], len(tris) / 4)


class UnitLevelTests(unittest.TestCase):
    """A family's own levels of detail (build_meshes.unit_groups): a piece its levels leave out stands in with its full
    level at each coarser level and in the shadow caster of its group."""

    def test_a_piece_left_at_its_full_level_stays_at_the_coarser_level_and_casts_its_shadow(self):
        # a lane's round tower, with a coarser level and a shadow caster, beside a piece its levels leave out (the
        # generic massing of a footprint the village does not build), in one tile
        ring = lambda n: Polygon([(10.0 + 3.0 * math.cos(a), 10.0 + 3.0 * math.sin(a))
                                  for a in np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)])
        tower, coarse, caster = (build_meshes.prism(ring(n), 0.0, 12.0, (5, 0)) for n in (64, 16, 8))
        left = build_meshes.prism(Polygon([(20.0, 0.0), (27.0, 0.0), (27.0, 18.0), (20.0, 18.0)]), 0.0, 5.6, (5, 0))
        units = [{0: [tower], 1: [coarse], 3: [caster]}, {0: [left]}]
        encoded = build_meshes.encode_assets({"diagnostic": [tower, left]}, {"diagnostic": units})["diagnostic"]
        tris = {}
        for c in encoded["chunks"]:
            tris[c["level"]] = tris.get(c["level"], 0) + c["tris"]
        self.assertEqual(tris.get(1), len(coarse.idx) + len(left.idx))
        self.assertEqual(tris.get(3), len(caster.idx) + len(left.idx))


class PartMergeTests(unittest.TestCase):
    """A family's part, merged, gives the same bytes as building every family at once (stages S1 and S2)."""
    SCENE = {"families": ["grounds"], "assets": [{"id": "hut", "family": "grounds"}, {"id": "tomb", "family": "grounds"}]}

    def test_a_part_with_an_empty_asset_and_no_mesh_tower_merges_to_the_full_build(self):
        ground = world.datum()
        meshes = {"hut": [build_meshes.prism(Polygon([(0, 0), (4, 0), (4, 3), (0, 3)]), ground, ground + 3.0, (5, 3))],
                  "tomb": []}
        tops = [0.0, 0.0]
        local = {"nx": 40, "ny": 40, "aff": world.rotated_grid(40).aff.tolist()}
        marked = build_meshes.footprint_marks(build_meshes.encode_assets(meshes), local)
        self.assertEqual(set(marked), {"hut"})
        full = build_meshes.encode_payload(meshes, self.SCENE, build_meshes.source_digests(),
                                           build_meshes.shadow_regions(build_meshes.bounds(meshes), self.SCENE), tops,
                                           build_meshes.footprint_entry(marked, local))
        with mock.patch.object(build_meshes, "massing", return_value=(meshes, tops, [])):
            part = json.loads(json.dumps(build_meshes.encode_part("grounds", data={"meta": {"local": local}},
                                                                  scene=self.SCENE)))
        self.assertEqual(build_meshes.merge_parts([part], self.SCENE), full)

    def test_the_parts_folder_may_be_written_with_a_trailing_slash(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(build_meshes, "merge_parts", return_value="merged") as merge, \
                mock.patch.object(build_meshes, "write") as write:
            out = str(Path(folder) / "meshes.js")
            build_meshes.main(["--parts", folder + "/", out])
        merge.assert_called_once_with([])
        write.assert_called_once_with(out, "merged")


class BuiltPayloadTests(unittest.TestCase):
    def test_payload_carries_every_claimed_asset(self):
        scene = world.scene()
        built = {c["asset"] for c in world.payload()["chunks"]}
        claiming = {a["id"] for a in scene["assets"] if {"components", "castle_pieces", "rule"} & set(a)}
        self.assertLessEqual(claiming, built)
        self.assertLessEqual(built, {a["id"] for a in scene["assets"]})


class PayloadOracleTests(unittest.TestCase):
    """trimesh, as an independent oracle on the decoded payload: each family's test module checks its own meshes."""

    def test_every_family_with_geometry_pins_its_mesh_faults(self):
        built = {c["asset"] for c in world.payload()["chunks"]}
        for family in sorted({a["family"] for a in world.scene()["assets"] if a["id"] in built}):
            with self.subTest(family=family):
                module = importlib.import_module(f"test_assets_{family}")
                self.assertEqual(set(module.MESH_FAULTS), {"winding", "collapsed"})


if __name__ == "__main__":
    unittest.main()
