"""Mesh trees (the default; ?trees=raymarched marches the raymarched trees instead, for comparison): the tree table's
trees drawn by their own pass in place of the primary march's 3D trees, each ray cast within a screen quad round it
(frag5.glsl TREE_DRAW). The march marches none of them inside the table (vegetation mode 3) and ends at their depth, as
it ends at a mesh's; they stand where the raymarched trees stood, in the same shapes, and write the record those wrote,
so the shading pass shades them alike. Debug bit 1048576 brings the raymarched trees back within the same page, the
other arm of the within-page A/B.

With a payload that carries a tree kit, the trees near the camera are its species meshes instead (TREE_MESH), each
fitted to its tree and handed over to its drawn crown by a dither across a band of distance, farther for a tree that
stands apart with the sparse dial on (as the page opens; ?treesparse=0 turns it off); and the lake's mirror draws every
tree's crown, which the march's reflections in the main lake then leave out (TREE_REFL). The drawn crowns reach
TREE_FAR (2.6 km), and the march draws the raymarched trees' canopy sheet beyond, as it does for those; debug bit
2097152 takes the crowns to the table's edge.

The default page's programs, the mesh trees' among them, are test_render_core's
test_the_page_compiles_the_cpu_checked_programs; the raymarched dial's page compiling none of this, with no MESH_TREES
define, is RaymarchedDialTests'."""
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import shaders  # noqa: E402
from mesh_source import load_data  # noqa: E402
from pagetest import PageTestCase, smoke  # noqa: E402
from render_testing import FRAME_TIMEOUT_S, LAWN, MAT_SKY, MAT_WATER, NEXT_FRAME, REARM, WEST, RenderTestCase, ray  # noqa: E402
from test_render_core import RECORD_SHADERS, first_difference  # noqa: E402

DBG_P1_RECORD = 256                 # template_c.html drawMeshes: P1's own record, no mesh or mesh-tree G-buffer draw
DBG_RAYMARCHED_TREES = 1048576      # the raymarched trees in a mesh-tree page (frag1.glsl VM_TREES)
DBG_CROWNS_TO_EDGE = 2097152        # the drawn crowns to the table's edge, no canopy sheet within it (frag3.glsl evalTop)
DBG_REFLECTIONS = 128               # the shading pass reports each pixel's reflection: (mirror?, distance, material)
RMAT_TREE, RMAT_MIRROR = 3, 5       # ... its material: a tree the march's reflection met, or the mirror's image
MAT_TREE = 3
CROWN, TRUNK, SHEET = 1.0, 2.0, -1.0  # a 3D tree's part in its record (tp.w), and the far canopy sheet's
LEAVES, BARK = 3.0, 4.0             # ... and a species mesh's
TREE_FAR_M = 2600.0                 # the drawn crowns' reach, and the raymarched 3D trees' (frag1.glsl TREE_FAR)
P1_STEPS = 128                      # P1's record counts its march's iterations in 128s of its code; a draw's has none
FOREST_CELL_M = 100.0               # the forest views look at the densest square of this side
FOREST_REACH_M = 1500.0             # ... within this of the castle
MIRROR_NEAR_M = 1500.0              # trees this near the lake, and their reflections this long, lie inside the table
FINEST_M = 600.0                    # the species' reach (their last switch) that brings both levels over the forest view

# In-page JS: every `step`-th internal pixel of a rectangle as [x, y, code, t, tc (4), tp (4)] (probeTreeRect's rows)
SAMPLE = """([x0, y0, w, h, step]) => {
  const out = [], rows = __hog.probeTreeRect(x0, y0, w, h);
  for (let y = 0; y < h; y += step) for (let x = 0; x < w; x += step) out.push([x0 + x, y0 + y, ...rows[y * w + x]]);
  return out; }"""
# In-page JS: as SAMPLE, rows [x, y, mat, the shading pass's report under DBG_REFLECTIONS (3)]
REFLECTION = """([x0, y0, w, h, step]) => {
  const g = __hog.probeRect(x0, y0, w, h), c = __hog.probeColorRect(x0, y0, w, h), out = [];
  for (let y = 0; y < h; y += step) for (let x = 0; x < w; x += step) {
    const i = y * w + x;
    out.push([x0 + x, y0 + y, g[i][0], c[4 * i], c[4 * i + 1], c[4 * i + 2]]);
  }
  return out; }"""


def is_tree(cells):
    """Pixels of a 3D tree's record: material 3 with a crown or trunk part (not the far sheet)."""
    return ((cells[:, 2].astype(int) & 7) == MAT_TREE) & np.isin(cells[:, 11], (CROWN, TRUNK))


def is_species(cells):
    """Pixels of a species mesh's record: material 3 with its leaves or bark."""
    return ((cells[:, 2].astype(int) & 7) == MAT_TREE) & np.isin(cells[:, 11], (LEAVES, BARK))


class MeshTreeTests(RenderTestCase):
    @classmethod
    async def _open(cls):
        await super()._open()
        cls.opened = await cls.page.evaluate("__hog.treeStats()")      # the species meshes' dials as the page opens

    def setUp(self):
        super().setUp()
        # the species meshes' detail and sparse dials, which tests change and the shared restore leaves alone: as the page
        # opened
        self.loop.run_until_complete(self.page.evaluate("([d, s]) => { __hog.treeDetail(d); __hog.treeSparse(s); }",
                                                        [self.opened["detail"], self.opened["sparse"]]))

    async def forest(self, pitch=-0.6, height=60.0, back=150.0, kit=False, apart=False):
        """Looks north at the densest FOREST_CELL_M square of trees (of trees that stand apart, when `apart`) within
        FOREST_REACH_M of the castle, from `back` metres south of its centre and `height` metres over the ground: without
        meshes, or with the embedded payload's and the tree kit it carries when `kit` (blocked when it carries none).
        Returns the square's centre (x, z)."""
        if not kit:
            await self.show([])
        key = await self.page.evaluate("""([cell, reach, apart]) => {
            const n = new Map();
            for (const t of __hog.treeList(-reach, -reach, reach, reach)) {
              if (apart && !t.apart) continue;
              const k = `${Math.floor(t.x / cell)},${Math.floor(t.z / cell)}`;
              n.set(k, (n.get(k) || 0) + 1);
            }
            return [...n].sort((a, b) => b[1] - a[1])[0][0]; }""", [FOREST_CELL_M, FOREST_REACH_M, apart])
        x, z = ((int(v) + 0.5) * FOREST_CELL_M for v in key.split(","))
        if kit:
            await self.look([x, await self.ground(x, z - back) + height, z - back], 0.0, pitch)
            if not (await self.page.evaluate("__hog.treeStats()"))["kit"]:
                self.skipTest("blocked: the embedded payload carries no tree kit")
        else:
            await self.show([], at=(x, z - back), yaw=0.0, pitch=pitch, height=height)
        return x, z

    async def look(self, eye, yaw, pitch):
        """The camera at `eye`, the payload as it is."""
        frames = await self.page.evaluate("([p, y, q]) => { __hog.camera(p, y, q); return __hog.frames; }", [eye, yaw, pitch])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 3}")

    async def instances(self, centres):
        """The instances (treeList's) centred at each of `centres` (n x 2), in their order."""
        unique, back = np.unique(centres, axis=0, return_inverse=True)
        lo, hi = unique.min(axis=0) - 1.0, unique.max(axis=0) + 1.0
        found = await self.page.evaluate("([a, b, c, d]) => __hog.treeList(a, b, c, d)", [*lo, *hi])
        index = {(t["x"], t["z"]): t for t in found}
        trees = [index[(x, z)] for x, z in unique.tolist()]
        return [trees[i] for i in back.ravel()]

    async def sample(self, dbg, step=2, phase=None, probe=SAMPLE):
        """`probe` (SAMPLE's by default) over the whole internal frame in a frame drawn with debug bits `dbg`, at jitter
        `phase` when given (so that two calls see the same rays), with that frame's probe (its view)."""
        found = await self.page.evaluate(f"""async ([dbg, step, phase, ms]) => {{
            {NEXT_FRAME}
            const sample = {probe};
            __hog.dbg(dbg);
            {REARM}
            for (let i = 0; i < 4 * 8; i++) {{
              if (!await nextFrame(ms)) break;
              const view = __hog.probe(0, 0), j = view.jit;
              if (i >= 1 && (!phase || (j[0] === phase[0] && j[1] === phase[1])))
                return {{ view, cells: sample([0, 0, view.res[0], view.res[1], step]) }};
            }}
            return null; }}""", [dbg, step, phase, FRAME_TIMEOUT_S * 1000])
        self.assertIsNotNone(found, f"no frame at jitter phase {phase} with debug bits {dbg}")
        return found["view"], np.array(found["cells"], dtype=np.float64)

    @smoke
    async def test_the_page_opens_with_the_mesh_trees_on(self):
        # the sampler audit lists the mesh trees' programs only while they are on, and test_render_core's
        # test_every_program_fits_the_sampler_limit checks every program it lists
        audit = await self.page.evaluate("__hog.samplers()")
        self.assertLessEqual({"trees", "treeInstances", "treeMeshes", "treeMirror"}, set(audit["counts"]))

    async def test_mesh_trees_are_the_raymarched_trees(self):
        # one rule: every raymarched tree a forest view shows is an instance, its centre, size, kind and hue the same;
        # the mesh trees then show the same trees
        await self.forest()
        view, marched = await self.sample(DBG_RAYMARCHED_TREES)
        stats = await self.page.evaluate("__hog.treeStats()")
        self.assertFalse(stats["shown"])
        found = marched[is_tree(marched)]
        self.assertGreater(len(found), 500)
        self.assertTrue(np.all(found[:, 2] >= P1_STEPS))         # marched, not drawn
        records = np.unique(found[:, 4:11], axis=0)      # tc and tp's kind, half-height and hue: one row per tree
        lo, hi = records[:, :2].min(axis=0) - 1.0, records[:, :2].max(axis=0) + 1.0
        inst = await self.page.evaluate("([a, b, c, d]) => __hog.treeList(a, b, c, d)", [lo[0], lo[1], hi[0], hi[1]])
        centres = np.array([[t["x"], t["z"]] for t in inst])
        for rec in records:
            k = int(np.argmin(np.abs(centres - rec[:2]).sum(axis=1)))
            t = inst[k]
            with self.subTest(tree=(round(t["x"], 2), round(t["z"], 2))):
                self.assertLess(np.abs(centres[k] - rec[:2]).max(), 1e-3)
                self.assertAlmostEqual(rec[2], t["r"], delta=1e-4 * t["r"])
                self.assertAlmostEqual(rec[3], t["H"], delta=1e-4 * t["H"])
                self.assertEqual(rec[4] > 0.5, t["con"])
                self.assertAlmostEqual(rec[6], t["hue"], delta=1e-5)
        _, drawn = await self.sample(0, phase=view["jit"])
        self.assertTrue((await self.page.evaluate("__hog.treeStats()"))["shown"])
        seen = [{tuple(c) for c in np.round(cells[is_tree(cells)][:, 4:6], 1)} for cells in (marched, drawn)]
        self.assertGreater(len(seen[0] & seen[1]), 0.9 * len(seen[0] | seen[1]))

    async def test_the_march_draws_no_trees_and_ends_at_theirs(self):
        # with mesh trees on, P1's own record (debug bit 256) holds no tree inside the table: where a mesh tree is the
        # front surface, P1 gave way to it (the sky's record) or found nothing beyond it
        await self.forest()
        view, drawn = await self.sample(0)
        _, own = await self.sample(DBG_P1_RECORD, phase=view["jit"])
        _, marched = await self.sample(DBG_P1_RECORD | DBG_RAYMARCHED_TREES, phase=view["jit"])
        self.assertGreater(is_tree(marched).sum(), 500)            # the view has trees for the march to draw
        self.assertEqual(int(((own[:, 2].astype(int) & 7) == MAT_TREE).sum()), 0)
        trees = is_tree(drawn)
        self.assertGreater(trees.sum(), 500)
        self.assertTrue(np.all(drawn[trees, 2] < P1_STEPS))       # drawn, not marched
        p1_mat, p1_t, tree_t = own[trees, 2].astype(int) & 7, own[trees, 3], drawn[trees, 3]
        # P1 bisects terrain that the step past a tree's depth crossed before it checks that depth, as at a mesh's
        # (trace(): terrain first), so a rare pixel's ground lies up to a step beyond the tree; the G-buffer pass's
        # depth test keeps the tree there. Marching on past the trees would put thousands beyond them
        beyond = (p1_mat != MAT_SKY) & (p1_t > tree_t * 1.001 + 0.05)
        self.assertLessEqual(int(beyond.sum()), max(3, int(0.001 * trees.sum())), f"P1 marched past {int(beyond.sum())} mesh-tree pixels")
        self.assertGreater((p1_mat == MAT_SKY).mean(), 0.95)

    async def test_the_march_finds_no_tree_at_the_table_edge(self):
        # with the drawn crowns to the table's edge (by default the march draws the canopy sheet from TREE_FAR on and
        # never reaches this bound), inside the tree table the march's bound on the trees is the distance to the table's
        # edge, which only bounds its steps: a ray crossing the edge (the horizon's band, seen from anywhere near the
        # castle) passes it. Were the bound a hit as it reaches 0, a sample within the march's epsilon of the edge would
        # end the ray as a tree with no tree in its record. From 80 m over the lawn, looking west, the nearest edge 4.8 km
        # off under the ceiling
        await self.show([], at=LAWN, yaw=WEST, pitch=0.06, height=80.0)
        _, cells = await self.sample(DBG_CROWNS_TO_EDGE, step=1)
        mat = cells[:, 2].astype(int) & 7
        self.assertGreater((mat == MAT_SKY).mean(), 0.1)        # rays that cross the edge and run on into the sky
        bogus = (mat == MAT_TREE) & np.all(cells[:, 4:12] == 0, axis=1)
        self.assertEqual(int(bogus.sum()), 0, f"{int(bogus.sum())} pixels end at the table's edge as a tree")

    async def test_the_drawn_crowns_give_way_to_the_canopy_sheet_at_their_reach(self):
        # the drawn crowns reach TREE_FAR, as the raymarched 3D trees do, and beyond it the march draws the canopy sheet
        # the raymarched trees give way to: there both show the same sheet, and no mesh tree's crown or trunk lies beyond.
        # Debug bit 2097152 takes the crowns to the table's edge. From 400 m over the lawn, toward the eighth of the
        # compass with the most trees 2.8 to 4.5 km off, looking at their ground 3.4 km out
        best, bed = await self.page.evaluate("""([cx, cz]) => {
            const n = new Array(8).fill(0), beds = [...Array(8)].map(() => []);
            for (const t of __hog.treeList(cx - 4500, cz - 4500, cx + 4500, cz + 4500)) {
              const dx = t.x - cx, dz = t.z - cz, d = Math.hypot(dx, dz);
              if (d <= 2800 || d >= 4500) continue;
              const k = ((Math.round(Math.atan2(dx, dz) / (Math.PI / 4)) % 8) + 8) % 8;
              n[k]++; beds[k].push(t.bed);
            }
            const best = n.indexOf(Math.max(...n)), b = beds[best].sort((p, q) => p - q);
            return [best, b[b.length >> 1]]; }""", list(LAWN))
        eye = await self.ground(*LAWN) + 400.0
        await self.show([], at=LAWN, yaw=best * math.pi / 4, pitch=math.atan2(bed - eye, 3400.0), height=400.0)
        view, marched = await self.sample(DBG_RAYMARCHED_TREES, step=1)
        _, drawn = await self.sample(0, step=1, phase=view["jit"])
        _, edge = await self.sample(DBG_CROWNS_TO_EDGE, step=1, phase=view["jit"])

        def sheet(cells):
            return ((cells[:, 2].astype(int) & 7) == MAT_TREE) & (cells[:, 11] == SHEET)

        far = sheet(marched) & (marched[:, 3] > TREE_FAR_M + 50.0)
        self.assertGreater(int(far.sum()), 500, "too little far canopy in view")
        self.assertGreater(sheet(drawn)[far].mean(), 0.95)
        self.assertEqual(int((is_tree(drawn) & (drawn[:, 3] >= TREE_FAR_M)).sum()), 0, "a drawn crown beyond TREE_FAR")
        self.assertGreater(int((is_tree(edge) & far).sum()), 0.5 * far.sum(), "debug bit 2097152 leaves the far canopy a sheet")

    async def test_a_tree_beside_the_camera_puts_no_fragment_in_the_crown_pass(self):
        # a crown's quad covers what its box reaches beyond the camera's plane, never the whole view: a tree across that
        # plane to one side, off the view, covers nothing, and one across it that reaches into the view covers part of it.
        # Were a box near the camera drawn over the whole view, the forest's trees beside the camera would each ray cast
        # every pixel, unseen but for the time (156 of them at the greenhouses). The tree nearest the lawn, the camera
        # level with its middle, looking north
        found = await self.page.evaluate("([x, z]) => __hog.treeList(x - 300, z - 300, x + 300, z + 300)", list(LAWN))
        tree = min(found, key=lambda t: math.hypot(t["x"] - LAWN[0], t["z"] - LAWN[1]))
        middle = tree["bed"] + tree["relief"] + tree["near"] + 0.5 * tree["H"]

        async def cover(side):
            await self.look([tree["x"] + side, middle, tree["z"]], 0.0, 0.0)
            return await self.page.evaluate("([x, z]) => __hog.crownCover(x, z)", [tree["x"], tree["z"]])

        self.assertEqual((await cover(3.0 * tree["r"] + 10.0))["covered"], 0, "a tree beside the view covers it")
        self.assertGreater(middle, await self.ground(tree["x"] + 1.5 * tree["r"], tree["z"]) + 1.5)   # the camera stays put
        across = await cover(1.5 * tree["r"])
        self.assertGreater(across["covered"], 0, "a tree across the camera's plane, in view, covers none of it")
        self.assertLess(across["covered"], 0.5 * across["pixels"], "a tree across the camera's plane covers the view")

    async def test_mesh_trees_take_the_raymarched_trees_shapes(self):
        # like for like: the same rays meet a tree in both, at about the same distance (the march stops within its
        # epsilon of a crown, and a mesh tree shears to a plane through the ground rather than every bump of it)
        await self.forest()
        view, marched = await self.sample(DBG_RAYMARCHED_TREES)
        _, drawn = await self.sample(0, phase=view["jit"])
        a, b = is_tree(marched), is_tree(drawn)
        self.assertGreater(a.sum(), 500)
        self.assertLess((a ^ b).sum(), 0.15 * (a | b).sum())
        same = a & b & np.all(np.abs(marched[:, 4:6] - drawn[:, 4:6]) < 1e-3, axis=1)
        self.assertGreater(same.sum(), 0.75 * (a | b).sum())
        np.testing.assert_allclose(drawn[same, 4:8], marched[same, 4:8], atol=1e-3)    # tc: centre, crown radius, height
        np.testing.assert_allclose(drawn[same, 8:11], marched[same, 8:11], atol=1e-4)  # tp: conifer, half-height, hue
        dt = np.abs(marched[same, 3] - drawn[same, 3])
        self.assertLess(np.percentile(dt / (0.3 + 0.002 * drawn[same, 3]), 95), 1.0)
        self.assertGreater(np.mean(marched[same, 11] == drawn[same, 11]), 0.95)   # crown and trunk alike

    async def test_the_kit_draws_the_near_trees_and_hands_them_over_to_their_crowns(self):
        # with the payload's kit, trees nearer than the last switch's band are species meshes and trees beyond it drawn
        # crowns; across the band each tree's pixels go over from its mesh to its crown, by a dither, so the nearer
        # band trees show mostly mesh and the farther mostly crown. The detail dial puts the band across the densest
        # square, so that its trees fall on both sides and in it; at 0 it leaves only drawn crowns. Every tree takes the
        # forest's band: the sparse dial off
        await self.page.evaluate("__hog.treeSparse(0)")
        x, z = await self.forest(kit=True)
        view = await self.centre()
        stats = await self.page.evaluate("__hog.treeStats()")
        square = math.dist(view["pos"], (x, await self.ground(x, z), z))
        await self.page.evaluate(f"__hog.treeDetail({stats['detail'] * square / stats['meshes']['lod'][-1]})")
        view, cells = await self.sample(0, step=1)
        band = (await self.page.evaluate("__hog.treeStats()"))["meshes"]["band"]
        cells = cells[is_tree(cells) | is_species(cells)]
        trees = await self.instances(cells[:, 4:6])
        dist = np.linalg.norm(np.array([[t["x"], t["bed"], t["z"]] for t in trees]) - view["pos"], axis=1)
        mesh = is_species(cells)
        near, far = dist < band[0], dist > band[1]
        self.assertGreater(int(near.sum()), 300)
        self.assertGreater(int(far.sum()), 300)
        self.assertEqual(int((~mesh[near]).sum()), 0, "drawn crowns nearer than the band")
        self.assertEqual(int(mesh[far].sum()), 0, "species meshes beyond the band")
        into = (dist - band[0]) / (band[1] - band[0])            # how far into the band each pixel's tree stands
        shares = {}
        for k in np.flatnonzero(~near & ~far):
            shares.setdefault(tuple(cells[k, 4:6]), []).append((into[k], mesh[k]))
        shares = np.array([(v[0][0], np.mean([m for _, m in v])) for v in shares.values() if len(v) >= 10])
        nearer, farther = shares[shares[:, 0] < 1 / 3, 1], shares[shares[:, 0] > 2 / 3, 1]
        self.assertGreaterEqual(min(len(nearer), len(farther)), 3, "too few trees in the band's thirds")
        self.assertGreater(nearer.mean(), 0.6)
        self.assertLess(farther.mean(), 0.4)
        await self.page.evaluate("__hog.treeDetail(0)")
        _, cells = await self.sample(0)
        self.assertEqual(int(is_species(cells).sum()), 0)
        self.assertGreater(int(is_tree(cells).sum()), 500)
        self.assertIsNone((await self.page.evaluate("__hog.treeStats()"))["meshes"])

    async def test_trees_that_stand_apart_keep_their_species_to_their_own_reach(self):
        # a tree with few others near it (treeStats' apartRule: n or fewer within r m) stands apart, as a lawn's trees do.
        # With the sparse dial on (as the page opens) it keeps its species mesh beyond the forest's reach: past the
        # forest's hand-over band and short of its own, a tree that stands apart is all species mesh, and the forest's
        # trees there all drawn crowns; across its own band it goes over to its crown. With the dial off it hands over
        # with the forest: the same trees are all drawn crowns, none left undrawn. From 130 m south of the densest square
        # of such trees, 30 m up
        x, z = await self.forest(pitch=-0.25, height=30.0, back=130.0, kit=True, apart=True)
        rule, half = (await self.page.evaluate("__hog.treeStats()"))["apartRule"], FOREST_CELL_M / 2
        box = half + rule["r"] + 1.0
        listed = await self.page.evaluate("([a, b, c, d]) => __hog.treeList(a, b, c, d)", [x - box, z - box, x + box, z + box])
        p, flagged = np.array([[t["x"], t["z"]] for t in listed]), np.array([t["apart"] for t in listed])
        inside = np.all(np.abs(p - [x, z]) <= half, axis=1)
        others = (((p[inside, None, :] - p[None, :, :]) ** 2).sum(axis=2) < rule["r"] ** 2).sum(axis=1) - 1
        np.testing.assert_array_equal(flagged[inside], others <= rule["n"])

        async def trees(reach):                             # each tree pixel nearer than reach: a species mesh's?, its
            view, cells = await self.sample(0, step=1)      # tree apart?, the tree's distance
            cells = cells[(is_tree(cells) | is_species(cells)) & (cells[:, 3] < reach)]
            found = await self.instances(cells[:, 4:6])
            dist = np.linalg.norm(np.array([[t["x"], t["bed"], t["z"]] for t in found]) - view["pos"], axis=1)
            return is_species(cells), np.array([t["apart"] for t in found]), dist, cells

        meshes = (await self.page.evaluate("__hog.treeStats()"))["meshes"]      # the sparse dial on, as the page opens
        band, own = meshes["band"], meshes["sparseBand"]
        mesh, apart, dist, cells = await trees(1.5 * own[1])
        between = apart & (dist > band[1]) & (dist < own[0])
        found = {tuple(c) for c in cells[between, 4:6]}
        self.assertGreater(int(between.sum()), 300)
        self.assertGreaterEqual(len(found), 3, "too few trees apart between the bands")
        self.assertEqual(int((~mesh[between]).sum()), 0, "a drawn crown for a tree apart, within its own reach")
        self.assertEqual(int(mesh[~apart & (dist > band[1])].sum()), 0, "a species mesh for a forest tree beyond its reach")
        into = (dist - own[0]) / (own[1] - own[0])          # across their own band they go over to their crowns
        shares = {}
        for k in np.flatnonzero(apart & (into > 0) & (into < 1)):
            shares.setdefault(tuple(cells[k, 4:6]), []).append((into[k], mesh[k]))
        shares = np.array([(v[0][0], np.mean([m for _, m in v])) for v in shares.values() if len(v) >= 10]).reshape(-1, 2)
        nearer, farther = shares[shares[:, 0] < 0.5, 1], shares[shares[:, 0] >= 0.5, 1]
        self.assertGreaterEqual(min(len(nearer), len(farther)), 3, "too few trees apart in their band's halves")
        self.assertGreater(nearer.mean(), 0.55)
        self.assertLess(farther.mean(), 0.45)
        await self.page.evaluate("__hog.treeSparse(0)")
        mesh, apart, dist, cells = await trees(1.5 * own[1])
        between = apart & (dist > band[1]) & (dist < own[0])
        self.assertEqual(int(mesh[between].sum()), 0, "with the dial off, a species mesh beyond the forest's reach")
        shown = {tuple(c) for c in cells[between, 4:6]}
        self.assertGreater(len(found & shown), 0.9 * len(found), "with the dial off, trees apart that nothing draws")

    async def test_species_meshes_stand_within_their_trees(self):
        # each tree near the camera is a variant of its own kind fitted to it: every pixel of its mesh lies within the
        # upright cylinder of its crown radius about its centre, from its ground (its trunk sunk a little) to its height,
        # and carries its record (centre, size, kind and hue). The dial brings both levels, leaf cards and all, over the
        # view
        await self.forest(kit=True)
        stats = await self.page.evaluate("__hog.treeStats()")
        await self.page.evaluate(f"__hog.treeDetail({stats['detail'] * FINEST_M / stats['meshes']['lod'][-1]})")
        view, cells = await self.sample(0, step=1)
        conifer = {v["species"]: v["conifer"] for v in stats["kit"]}
        mesh = cells[is_species(cells)]
        self.assertGreater(int((mesh[:, 11] == LEAVES).sum()), 2000)
        self.assertGreater(int((mesh[:, 11] == BARK).sum()), 20)
        trees = await self.instances(mesh[:, 4:6])
        r, H = np.array([t["r"] for t in trees]), np.array([t["H"] for t in trees])
        con = np.array([t["con"] for t in trees])
        np.testing.assert_array_equal(mesh[:, 6:8], np.stack([r, H], axis=1))
        np.testing.assert_array_equal(mesh[:, 8] > 0.5, con)
        np.testing.assert_allclose(mesh[:, 10], [t["hue"] for t in trees], atol=1e-5)
        np.testing.assert_array_equal([conifer[int(s)] for s in mesh[:, 9]], con)   # a species of its kind
        hits = np.array([o + d * t for o, d, t in ((*ray(view, int(px), int(py)), t) for px, py, t in mesh[:, [0, 1, 3]])])
        slack = 0.02 + 2e-4 * mesh[:, 3]
        out = np.hypot(hits[:, 0] - mesh[:, 4], hits[:, 2] - mesh[:, 5]) - r
        self.assertLess(out.max(), slack[np.argmax(out)], "a species mesh wider than its tree's crown")
        ground = np.array([t["bed"] + t["relief"] for t in trees])
        reach = np.abs([t["near"] for t in trees]) + slack           # its near octave, as its distance gives it
        above = hits[:, 1] - ground
        self.assertTrue(np.all(above > -0.03 * H - reach), "a species mesh below its tree's ground")
        self.assertTrue(np.all(above < H + reach), "a species mesh taller than its tree")

    async def test_the_mirror_draws_the_trees_the_lake_reflections_leave_out(self):
        # with mesh trees, the march's reflections in the main lake leave the table's trees out and the mirror draws
        # them: where the raymarched trees' reflection meets a tree near the lake, the mirror's image shows instead.
        # From 30 m over the lake's middle, toward the eighth of the compass with the most trees near its shores
        lake = await self.page.evaluate("DATA.meta.extras.lake_c")
        found = await self.page.evaluate("([a, b, c, d]) => __hog.treeList(a, b, c, d)",
                                         [lake[0] - MIRROR_NEAR_M, lake[1] - MIRROR_NEAR_M, lake[0] + MIRROR_NEAR_M, lake[1] + MIRROR_NEAR_M])
        d = np.array([[t["x"] - lake[0], t["z"] - lake[1]] for t in found])
        dist, bearing = np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 0], d[:, 1])
        yaws = np.arange(8) * math.pi / 4
        count = [int(((dist > 150) & (dist < MIRROR_NEAR_M) & (np.abs(np.angle(np.exp(1j * (bearing - y)))) < math.pi / 8)).sum())
                 for y in yaws]
        await self.look([lake[0], await self.ground(*lake) + 30.0, lake[1]], float(yaws[int(np.argmax(count))]), -0.1)
        view, marched = await self.sample(DBG_REFLECTIONS | DBG_RAYMARCHED_TREES, step=1, probe=REFLECTION)
        _, drawn = await self.sample(DBG_REFLECTIONS, step=1, phase=view["jit"], probe=REFLECTION)
        self.assertGreater((await self.page.evaluate("__hog.treeStats()"))["mirrored"], 0)
        water = marched[:, 2] == MAT_WATER
        trees = water & (marched[:, 5] == RMAT_TREE) & (marched[:, 4] < MIRROR_NEAR_M)
        self.assertGreater(int(trees.sum()), 200, "the lake reflects no trees from here")
        left = water & (drawn[:, 5] == RMAT_TREE) & (drawn[:, 4] < MIRROR_NEAR_M)
        self.assertEqual(int(left.sum()), 0, "the march's reflections kept trees the mirror draws")
        # the shading pass looks the rippled ray up in the half-resolution flat mirror, where the camera sees its point
        # at the flat image's distance: about half the pixels where the rippled march met a tree's crown show the mirror's
        # tree (0.47 on AMD), the rest its neighbours, as the ripples scatter both. A mirror with no trees shows almost none
        mirrored = drawn[trees, 5] == RMAT_MIRROR
        cover = int((water & (drawn[:, 5] == RMAT_MIRROR) & (drawn[:, 4] < MIRROR_NEAR_M)).sum()) / int(trees.sum())
        self.assertGreater(mirrored.mean(), 0.3, f"the mirror shows {mirrored.mean():.2f} of the trees (its cover {cover:.2f})")


class RaymarchedDialTests(RenderTestCase):
    """?trees=raymarched: the page as it was before the mesh trees, its trees raymarched, for comparison."""
    QUERY = PageTestCase.QUERY + "&trees=raymarched"
    INIT_SCRIPTS = (RECORD_SHADERS,)

    @smoke
    async def test_the_raymarched_dial_compiles_the_cpu_checked_programs_without_mesh_trees(self):
        defs = await self.page.evaluate("__hog.defs")
        cpu = shaders.cpu_defs(load_data(), float_linear="MANUAL_BILERP" not in defs, trees=False)
        self.assertEqual(list(cpu.items()), list(defs.items()))
        expected = [s for p in shaders.programs(defs) for s in (p.vertex, p.fragment)]
        recorded = await self.page.evaluate("window.__shaderSources")
        self.assertEqual(len(recorded), len(expected))
        for i, (got, want) in enumerate(zip(recorded, expected)):
            if got != want:
                self.fail(f"shader source {i}: {first_difference(got, want)}")


if __name__ == "__main__":
    import unittest
    unittest.main()
