"""The trees family: the species kit for the Forest's instanced mesh trees (mesh_trees.py).

The renderer's contract (lane T) sets what is checked here: two conifers and three broadleaves, two or three seeded
variants each; three levels of detail per variant within their triangle and vertex budgets, in tree-local metres with
the base at the origin; a recorded crown radius and height that no vertex exceeds, fitted to the placements' crown
ratios; leaf cards with their own corners' UVs; and per-species colours. White Hound's Forest sets the species, its
"well-grown and dark" trees, and the visible trunks under their crowns."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mesh_trees  # noqa: E402
from mesh_trees import BARK, CARD, FOLIAGE  # noqa: E402

BUDGETS = ((1500, 1100), (300, 220), (60, 45))       # (triangles, vertices) at each level: the renderer's contract
CONIFER_RATIO = 0.23                                 # the placements' conifer crown radius over its height
CONIFERS, BROADLEAVES = {"scots-pine", "yew"}, {"oak", "beech", "birch"}
# a leaf's length in life, metres (a pine's or yew's: a spray of needles)
LIFE_SIZE_M = {"oak": (0.06, 0.14), "beech": (0.05, 0.11), "birch": (0.03, 0.07), "scots-pine": (0.06, 0.16),
               "yew": (0.04, 0.12)}


def luminance(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def leaves(m):
    return m.attr[:, 0] != BARK


def side_coverage(m, crown_r, base, height, size=96):
    """The fraction of the crown's outline seen from the south (the convex hull of its leaves) that they cover, the
    cards drawn whole."""
    tris = m.pos[m.idx[leaves(m)[m.idx].all(axis=1)]]
    sx = (tris[..., 0] / crown_r + 1) / 2 * size
    sy = (height - tris[..., 1]) / (height - base) * size
    leaf, hull = Image.new("L", (size, size), 0), Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(leaf)
    for xs, ys in zip(sx, sy):
        draw.polygon(list(zip(xs, ys)), fill=255)
    pts = np.stack([sx.ravel(), sy.ravel()], 1)
    ImageDraw.Draw(hull).polygon([tuple(p) for p in pts[ConvexHull(pts).vertices]], fill=255)
    inside = np.asarray(hull) > 0
    return float(np.mean(np.asarray(leaf)[inside] > 0))


def leaf_cover(m, turn, mm=5):
    """The leaves' silhouette seen from the side at `turn` (radians about the trunk), in square metres, rasterised
    independently of the builder at `mm`-millimetre pixels: solid leaves cover their outline, and each leaf card (its two
    triangles drawn as one quad) mesh_trees.card_fill at each point of it, independently of the others."""
    per_m = 1000.0 / mm
    tri_leaf = m.attr[m.idx[:, 0], 0] != BARK
    tris = m.idx[tri_leaf]
    u = (np.cos(turn) * m.pos[:, 0] + np.sin(turn) * m.pos[:, 2]) * per_m
    v = m.pos[:, 1] * per_m
    u0, v0 = int(np.floor(u.min())) - 2, int(np.floor(v.min())) - 2
    w, h = int(np.ceil(u.max())) - u0 + 3, int(np.ceil(v.max())) - v0 + 3
    solid, clear = Image.new("1", (w, h), 0), np.ones((h, w))
    draw = ImageDraw.Draw(solid)
    card = m.attr[tris[:, 0], 0] == CARD
    for t in tris[~card]:
        draw.polygon([(u[i] - u0, v[i] - v0) for i in t], fill=1)
    for first in np.unique(tris[card].min(axis=1)):           # a card's four corners follow its first
        q = np.arange(first, first + 4)
        x0, y0 = int(np.floor(u[q].min())) - u0, int(np.floor(v[q].min())) - v0
        sub = Image.new("1", (int(np.ceil(u[q].max())) - u0 - x0 + 2, int(np.ceil(v[q].max())) - v0 - y0 + 2), 0)
        ImageDraw.Draw(sub).polygon([(u[i] - u0 - x0, v[i] - v0 - y0) for i in q], fill=1)
        a = np.asarray(sub)
        # the card's own coordinates at each pixel's centre: s along its first edge, t along its last
        pu, pv = np.meshgrid(np.arange(a.shape[1]) + x0 + u0 + 0.5, np.arange(a.shape[0]) + y0 + v0 + 0.5)
        e, f = np.array([u[q[1]] - u[q[0]], v[q[1]] - v[q[0]]]), np.array([u[q[3]] - u[q[0]], v[q[3]] - v[q[0]]])
        det = e[0] * f[1] - e[1] * f[0]
        if abs(det) < 1e-9:
            s = t_ = np.full(a.shape, 0.5)
        else:
            du, dv = pu - u[q[0]], pv - v[q[0]]
            s, t_ = (du * f[1] - dv * f[0]) / det, (e[0] * dv - e[1] * du) / det
        clear[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] *= np.where(a, 1 - mesh_trees.card_fill(s, t_), 1.0)
    cover = np.where(np.asarray(solid), 1.0, 1 - clear)
    return float(cover.sum()) / per_m ** 2


def foot_radius(m, height):
    """How far the bark reaches from the axis at the foot of the trunk: its lowest ring, sunk under the origin."""
    bark = m.attr[:, 0] == BARK
    bark &= m.pos[:, 1] < m.pos[bark, 1].min() + 0.005 * height
    return float(np.hypot(m.pos[bark, 0], m.pos[bark, 2]).max())


class TreeKitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kit = mesh_trees.kit()

    def test_two_conifers_and_three_broadleaves_each_in_two_or_three_variants(self):
        species = {v.species: v.conifer for v in self.kit}
        self.assertEqual({s for s, c in species.items() if c}, CONIFERS)
        self.assertEqual({s for s, c in species.items() if not c}, BROADLEAVES)
        for s in species:
            with self.subTest(species=s):
                self.assertIn(sum(v.species == s for v in self.kit), (2, 3))
        self.assertEqual(len({v.name for v in self.kit}), len(self.kit))

    def test_each_classs_shares_sum_to_one(self):
        for conifer in (True, False):
            with self.subTest(conifer=conifer):
                shares = [v.share for v in self.kit if v.conifer == conifer]
                self.assertAlmostEqual(sum(shares), 1.0)
                self.assertGreater(min(shares), 0)

    def test_each_level_keeps_the_renderers_budget(self):
        for v in self.kit:
            self.assertEqual(len(v.levels), 3)
            for level, (m, (tris, verts)) in enumerate(zip(v.levels, BUDGETS)):
                with self.subTest(variant=v.name, level=level):
                    self.assertLessEqual(len(m.idx), tris)
                    self.assertLessEqual(len(m.pos), verts)

    def test_its_arrays_are_the_renderers(self):
        for v in self.kit:
            for level, m in enumerate(v.levels):
                with self.subTest(variant=v.name, level=level):
                    n = len(m.pos)
                    self.assertEqual((m.pos.shape, m.nrm.shape, m.attr.shape, m.uv.shape), ((n, 3), (n, 3), (n, 2), (n, 2)))
                    self.assertEqual((m.attr.dtype, m.idx.dtype, m.uv.dtype), (np.uint8, np.uint32, np.float32))
                    self.assertLess(int(m.idx.max()), n)
                    np.testing.assert_allclose(np.linalg.norm(m.nrm, axis=1), 1.0, atol=1e-6)
                    self.assertLessEqual(set(np.unique(m.attr[:, 0])), {BARK, FOLIAGE, CARD})

    def test_leaf_cards_span_a_square_of_their_uv(self):
        # each card's corners in order round a square of the uv's unit square, so its leaves keep their shape
        for v in self.kit:
            for level, m in enumerate(v.levels):
                with self.subTest(variant=v.name, level=level):
                    card = m.attr[:, 0] == CARD
                    np.testing.assert_array_equal(m.uv[~card], 0)
                    if level < 2:
                        self.assertTrue(card.any())
                    for quad in m.uv[card].reshape(-1, 4, 2).astype(float):
                        lo, span = quad.min(axis=0), np.ptp(quad, axis=0)
                        self.assertAlmostEqual(span[0], span[1], places=5)
                        self.assertTrue(span[0] > 0 and lo.min() >= 0 and (lo + span).max() <= 1 + 1e-6)
                        self.assertEqual({tuple(np.round((c - lo) / span[0]).astype(int)) for c in quad},
                                         {(0, 0), (1, 0), (1, 1), (0, 1)})

    def test_near_trees_leaves_are_about_life_size(self):
        # the page cuts LEAF_GRID leaves per unit of a card's uv (times LEAF_TILE), each LEAF_CELLS cells long: a card
        # with the whole uv square shows leaves a third of its side, tens of centimetres on cards a metre or more across
        for v in self.kit:
            m = v.levels[0]
            card = np.flatnonzero(m.attr[:, 0] == CARD).reshape(-1, 4)
            side = np.linalg.norm(m.pos[card[:, 1]] - m.pos[card[:, 0]], axis=1)
            span = np.ptp(m.uv[card].astype(float), axis=1)[:, 0]
            leaf = side * mesh_trees.LEAF_CELLS / (mesh_trees.LEAF_GRID * mesh_trees.LEAF_TILE * span)
            with self.subTest(variant=v.name):
                lo, hi = LIFE_SIZE_M[v.species]
                self.assertTrue(lo <= float(np.median(leaf)) <= hi, float(np.median(leaf)))
                self.assertLess(float(np.percentile(leaf, 90)), 2 * hi)

    def test_near_crowns_are_full_not_half_bare(self):
        # cards a clump's size leave an airy crown (a birch's, a young pine's) half bare inside its outline, its limbs
        # showing between a few clumps: from two sides, the leaves cover at least half the outline of the crown
        for v in self.kit:
            m = v.levels[0]
            leaf = mesh_trees.drawn(m, leaves_only=True)          # the circles its cards' leaves fill, not their corners
            with self.subTest(variant=v.name):
                full = []
                for turn in (0.0, math.pi / 2):
                    outline = np.stack([np.cos(turn) * leaf[:, 0] + np.sin(turn) * leaf[:, 2], leaf[:, 1]], 1)
                    full.append(leaf_cover(m, turn, mm=20) / ConvexHull(outline).volume)
                self.assertGreater(float(np.mean(full)), 0.5)

    def test_every_triangle_is_wound_along_its_normals(self):
        for v in self.kit:
            for level, m in enumerate(v.levels):
                with self.subTest(variant=v.name, level=level):
                    t = m.pos[m.idx]
                    face = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
                    self.assertGreater(float(np.linalg.norm(face, axis=1).min()), 1e-8)
                    self.assertTrue(np.all(np.einsum("ij,ikj->ik", face, m.nrm[m.idx]) > 0))    # each vertex's

    def test_conifers_and_broadleaves_fit_their_placements_crowns(self):
        ratios = {v.name: v.crown_r / v.height for v in self.kit}
        for v in self.kit:
            with self.subTest(variant=v.name):
                if v.conifer:
                    self.assertAlmostEqual(ratios[v.name], CONIFER_RATIO, delta=0.02)
                else:
                    self.assertTrue(0.18 <= ratios[v.name] <= 0.65)
        broad = [ratios[v.name] for v in self.kit if not v.conifer]
        self.assertLessEqual(min(broad), 0.22)
        self.assertGreaterEqual(max(broad), 0.55)

    def test_no_vertex_lies_beyond_its_crown_radius_or_height(self):
        for v in self.kit:
            with self.subTest(variant=v.name):
                for m in v.levels:
                    # what a level draws: its cards' corners, beyond the circles their leaves fill, draw nothing
                    p = mesh_trees.drawn(m)
                    self.assertLessEqual(float(np.hypot(p[:, 0], p[:, 2]).max()), v.crown_r + 1e-9)
                    self.assertLessEqual(float(p[:, 1].max()), v.height + 1e-9)
                # the leaves, not a stray twig or a lump of one level, set the full level's crown
                lv = mesh_trees.drawn(v.levels[0], leaves_only=True)
                self.assertGreater(float(np.hypot(lv[:, 0], lv[:, 2]).max()), 0.9 * v.crown_r)
                self.assertGreater(float(lv[:, 1].max()), 0.97 * v.height)

    def test_it_stands_on_the_origin(self):
        for v in self.kit:
            with self.subTest(variant=v.name):
                m = v.levels[0]
                low = m.pos[:, 1].min()
                foot = m.pos[(m.attr[:, 0] == BARK) & (m.pos[:, 1] < low + 0.01 * v.height)]   # the trunk's first ring
                self.assertGreaterEqual(len(foot), 5)
                self.assertLess(float(foot[:, 1].max()), -0.01 * v.height)     # it sinks into a slope
                centre = foot[:, [0, 2]].mean(axis=0)
                self.assertLess(float(np.hypot(*centre)), 0.1 * v.trunk_r)
                self.assertAlmostEqual(float(np.median(np.linalg.norm(foot[:, [0, 2]] - centre, axis=1))) / v.trunk_r,
                                       1.0, delta=0.25)

    def test_its_trunk_shows_under_its_crown(self):
        for v in self.kit:
            with self.subTest(variant=v.name):
                m = v.levels[0]
                # where its leaves start as drawn: a card's leaves fill the circle its edges touch, not its corners
                self.assertAlmostEqual(v.crown_base, float(mesh_trees.drawn(m, leaves_only=True)[:, 1].min()))
                self.assertGreater(v.crown_base, 0.05 * v.height)
                if v.name.startswith("scots-pine-mature"):         # a tall, bare stem under a high crown
                    self.assertGreater(v.crown_base, 0.45 * v.height)
                if v.species == "yew" or v.name.startswith("scots-pine-young"):    # leaves almost to the ground
                    self.assertLess(v.crown_base, 0.2 * v.height)

    def test_its_trunk_reaches_into_its_crown_at_every_level(self):
        for v in self.kit:
            for level, m in enumerate(v.levels):
                with self.subTest(variant=v.name, level=level):
                    lv = leaves(m)
                    low = float(m.pos[lv, 1].min())
                    self.assertGreater(float(m.pos[~lv, 1].max()), low + 0.05 * (v.height - low))

    def test_the_crowns_have_gaps_yet_read_as_crowns(self):
        for v in self.kit:
            with self.subTest(variant=v.name):
                covered = side_coverage(v.levels[0], v.crown_r, v.crown_base, v.height)
                self.assertGreater(covered, 0.45)
                self.assertLess(covered, 0.97)

    def test_the_levels_keep_one_outline(self):
        for v in self.kit:
            full = v.levels[0]
            for level, m in enumerate(v.levels[1:], 1):
                with self.subTest(variant=v.name, level=level):
                    lv = leaves(m)
                    # an airy crown's coarsest lumps (a birch's, a young pine's cone) are narrower than its crown, to
                    # cover no more of it than its sparse cards do (test_a_tree_neither_thins_nor_thickens_...)
                    narrowest = 0.3 if level == 2 else 0.75
                    self.assertGreater(float(np.hypot(m.pos[lv, 0], m.pos[lv, 2]).max()), narrowest * v.crown_r)
                    self.assertGreater(float(m.pos[lv, 1].max()), 0.85 * v.height)
                    self.assertLess(abs(float(m.pos[lv, 1].min()) - v.crown_base), 0.15 * v.height)
                    self.assertLess(abs(float(m.pos[lv, 1].min()) - float(full.pos[leaves(full), 1].min())),
                                    0.15 * v.height)

    def test_a_tree_neither_thins_nor_thickens_as_it_changes_level(self):
        # the renderer switches levels with distance: a coarser level whose leaves cover less of the crown, or more, makes
        # each tree thin out or fill in at the switch (seen from four sides between the two the builder matches on)
        turns = [k * np.pi / 4 + np.pi / 8 for k in range(4)]
        for v in self.kit:
            full = np.mean([leaf_cover(v.levels[0], t) for t in turns])
            for level, m in enumerate(v.levels[1:], 1):
                with self.subTest(variant=v.name, level=level):
                    self.assertAlmostEqual(np.mean([leaf_cover(m, t) for t in turns]) / full, 1.0, delta=0.2)

    def test_the_trunk_keeps_its_girth_at_every_level(self):
        # a coarser level shrunk to the full level's crown would thin its trunk, which shows at the switch
        for v in self.kit:
            full = foot_radius(v.levels[0], v.height)
            for level, m in enumerate(v.levels[1:], 1):
                with self.subTest(variant=v.name, level=level):
                    self.assertAlmostEqual(foot_radius(m, v.height) / full, 1.0, delta=0.1)

    def test_leaves_light_as_one_crown(self):
        for v in self.kit:
            with self.subTest(variant=v.name):
                m = v.levels[0]
                lv = np.flatnonzero(leaves(m))
                order = lv[np.argsort(m.pos[lv, 1])]
                tenth = max(len(order) // 10, 1)
                self.assertGreater(float(m.nrm[order[-tenth:], 1].mean()), 0.4)     # its top faces the sky
                self.assertLess(float(m.nrm[order[:tenth], 1].mean()), 0.0)        # its underside the ground

    def test_the_inside_of_a_crown_is_marked_deeper(self):
        for v in self.kit:
            with self.subTest(variant=v.name):
                m = v.levels[0]
                lv = leaves(m)
                out = np.hypot(m.pos[:, 0], m.pos[:, 2]) / v.crown_r
                self.assertGreater(float(m.attr[lv & (out < 0.4), 1].mean()), float(m.attr[lv & (out > 0.8), 1].mean()) + 40)

    def test_bark_turns_up_the_stem_only_where_the_species_does(self):
        for v in self.kit:
            with self.subTest(variant=v.name):
                m = v.levels[0]
                b = m.attr[:, 0] == BARK
                foot, stem = b & (m.pos[:, 1] < 0.02 * v.height), b & (m.pos[:, 1] > 0.6 * v.height)
                if v.species in ("birch", "scots-pine"):
                    self.assertLess(float(m.attr[foot, 1].max()), 60)
                    self.assertGreater(float(m.attr[stem, 1].min()), 200)
                else:
                    self.assertEqual(int(m.attr[b, 1].max()), 0)

    def test_a_cards_leaves_thin_to_none_at_the_circle_its_edges_touch(self):
        # as the page cuts them (frag5.glsl leafEdge): its corners and its edges' midpoints carry no leaves
        self.assertAlmostEqual(float(mesh_trees.card_fill(0.5, 0.5)), mesh_trees.CARD_FILL)
        for s, t in ((0.0, 0.0), (1.0, 1.0), (0.5, 0.0), (1.0, 0.5)):
            self.assertEqual(float(mesh_trees.card_fill(s, t)), 0.0)

    def test_a_birchs_foot_stays_dark_between_its_rings(self):
        # the page interpolates the bark's turn between the trunk's rings: with none between the sunk foot and a ring
        # metres up, a birch greys from the ground. Its turn as drawn up the trunk's first 5% stays dark at the full and
        # the middle level
        for v in self.kit:
            if v.species != "birch":
                continue
            for level in (0, 1):
                m = v.levels[level]
                trunk = (m.attr[:, 0] == BARK) & (np.hypot(m.pos[:, 0], m.pos[:, 2]) < 2.5 * v.trunk_r) & (
                    m.pos[:, 1] < 0.4 * v.height)
                ys, at = m.pos[trunk, 1], m.attr[trunk, 1].astype(float)
                rings = np.unique(np.round(ys, 2))
                turn = np.interp(np.linspace(0, 0.05 * v.height, 20), rings,
                                 [at[np.round(ys, 2) == y].mean() for y in rings])
                with self.subTest(variant=v.name, level=level):
                    self.assertLess(float(turn.max()), 10)

    def test_the_kit_is_reproducible(self):
        again = [mesh_trees.variant(sp, k, seed, ratio) for sp in mesh_trees.SPECIES
                 for k, (seed, ratio) in enumerate(zip(sp.seeds, sp.ratios))]
        for v, w in zip(self.kit, again):
            for level, (a, b) in enumerate(zip(v.levels, w.levels)):
                with self.subTest(variant=v.name, level=level):
                    for field in a._fields:
                        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


class TreeColourTests(unittest.TestCase):
    def test_every_species_has_its_colours(self):
        colours = mesh_trees.kit_colours()
        self.assertEqual(set(colours), CONIFERS | BROADLEAVES)
        for name, c in colours.items():
            with self.subTest(species=name):
                for part in ("bark", "bark_upper", "foliage"):
                    self.assertEqual(len(c[part]), 3)
                    self.assertTrue(all(0 < x < 1 for x in c[part]))
                self.assertLess(luminance(c["foliage"]), 0.15)       # "well-grown and dark"
                self.assertEqual(c["autumn_share"], 0.0)             # in leaf, as the scene is

    def test_birch_whitens_and_a_pine_reddens_up_the_stem(self):
        colours = mesh_trees.kit_colours()
        birch, pine = colours["birch"], colours["scots-pine"]
        self.assertGreater(luminance(birch["bark_upper"]), 0.5)
        self.assertLess(luminance(birch["bark"]), 0.1)
        self.assertGreater(pine["bark_upper"][0], 2 * pine["bark_upper"][1])
        self.assertGreater(pine["bark_upper"][0], pine["bark"][0])


if __name__ == "__main__":
    unittest.main()
