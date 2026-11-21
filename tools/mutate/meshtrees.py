"""Mutations that the mesh-tree tests must catch: the mesh trees off by default or the raymarched dial ignored, the march
keeping its trees (vegetation mode 1), the A/B's raymarched arm keeping the mesh trees, the instance pass putting a tree
in every cell or reading another cell's density, a tree's crown sheared against the ground's slope or its record's kind
lost, the trees left out of the depth pre-pass or of the G-buffer pass, and the table's edge taken for a tree. With the
payload's kit: a species mesh not fitted to its tree, two levels keeping each other's share of a tree, the drawn crowns
fading in the wrong way across the hand-over band, a tree that stands apart keeping the forest's reach (its drawn
crown's or its species'), the sparse dial always on, off as the page opens, or its drawn crowns' reach kept with the
dial off, its count of neighbours missing a row of cells, the lake's reflections keeping the trees the mirror draws, or
the mirror drawing none. And the drawn crowns' reach: the crowns drawn past it, the march's canopy sheet left out within
the table, or the lake's reflections marching the sheet within the crowns' reach. And the crowns' quads covering the
whole view when a box comes within 1 m of the camera (the clip's loss), which changes no pixel, only the time:
__hog.crownCover counts a quad's pixels.

They rebuild the copy's page and need a browser (set HOG_GPU):

    python -X utf8 -m tools.mutate.run meshtrees
"""
from tools.mutate.run import Mutation

TREES = "test_render_trees.MeshTreeTests."
ONE_RULE = (TREES + "test_mesh_trees_are_the_raymarched_trees",)
MARCH = (TREES + "test_the_march_draws_no_trees_and_ends_at_theirs",)
SHAPES = (TREES + "test_mesh_trees_take_the_raymarched_trees_shapes",)
HAND_OVER = (TREES + "test_the_kit_draws_the_near_trees_and_hands_them_over_to_their_crowns",)
FIT = (TREES + "test_species_meshes_stand_within_their_trees",)
APART = (TREES + "test_trees_that_stand_apart_keep_their_species_to_their_own_reach",)
MIRROR = (TREES + "test_the_mirror_draws_the_trees_the_lake_reflections_leave_out",)
EDGE = (TREES + "test_the_march_finds_no_tree_at_the_table_edge",)
CUT = (TREES + "test_the_drawn_crowns_give_way_to_the_canopy_sheet_at_their_reach",)
CLIP = (TREES + "test_a_tree_beside_the_camera_puts_no_fragment_in_the_crown_pass",)
FLAG = '  const TREES_ON = Q.get("trees") !== "raymarched";'
MUTATIONS = [
    Mutation("mesh trees off by default", "template_b.html", FLAG, '  const TREES_ON = Q.get("trees") === "mesh";',
             ("test_render_core.MeshOcclusionTests.test_the_page_compiles_the_cpu_checked_programs",
              TREES + "test_the_page_opens_with_the_mesh_trees_on")),
    Mutation("the raymarched dial ignored", "template_b.html", FLAG, "  const TREES_ON = true;",
             ("test_render_trees.RaymarchedDialTests."
              "test_the_raymarched_dial_compiles_the_cpu_checked_programs_without_mesh_trees",)),
    Mutation("the march keeps its trees", "frag1.glsl",
             "#define VM_TREES ((uDbg & 1048576) != 0 ? 1 : 3)", "#define VM_TREES 1", MARCH),
    Mutation("the raymarched trees' arm keeps the mesh trees", "template_c.html",
             "detail === 1 && !(dbg & (8 | 1048576));", "detail === 1 && !(dbg & 8);", ONE_RULE),
    Mutation("a tree in every cell", "frag5.glsl", "  if (h.x >= v.x) return;\n", "", ONE_RULE),
    Mutation("the instance pass reads the next cell's density", "frag5.glsl",
             "  vec2 v = texelFetch(tTree, ic, 0).rg;", "  vec2 v = texelFetch(tTree, ic + ivec2(1, 0), 0).rg;", ONE_RULE),
    Mutation("crowns sheared against the ground's slope", "frag5.glsl",
             "  o.y -= dot(vB.yz, o.xz); d.y -= dot(vB.yz, d.xz);", "  o.y += dot(vB.yz, o.xz); d.y += dot(vB.yz, d.xz);",
             SHAPES),
    Mutation("every mesh tree recorded as a broadleaf", "frag5.glsl",
             "  gOut3 = vec4(con ? 1. : 0., con ? 0. : 0.36 * H,", "  gOut3 = vec4(0., 0.36 * H,", SHAPES),
    Mutation("the trees left out of the depth pre-pass", "template_c.html",
             "    if (trees) drawTrees(pos, rot, tanH, jit, sw, sh, sunDir);\n", "", MARCH),
    Mutation("the trees left out of the G-buffer pass", "template_c.html",
             "    if (trees) drawTrees(pos, rot, tanH, jit, sw, sh, sunDir);   // the same program", "    // the same program",
             SHAPES),
    Mutation("the table's edge taken for a tree", "frag3.glsl", "{ s.vd = max(te, 1.); return s; }",
             "{ s.vd = te; return s; }", EDGE),
    Mutation("species meshes at their own size", "template_b.html",
             "  vec2 g = aC.xy + aC.zw * f, k = vec2(aA.z / uFit.x, aA.w / uFit.y);", "  vec2 g = aC.xy + aC.zw * f, k = vec2(1.);",
             FIT),
    Mutation("two levels keep each other's share", "frag5.glsl",
             "  if (vI.y > 0. && (vI.z < 0.5) == (hash12(", "  if (vI.y > 0. && (vI.z > 0.5) == (hash12(", HAND_OVER),
    Mutation("the drawn crowns fade the wrong way", "template_b.html",
             "  vF = band.y > 0. ? clamp(", "  vF = band.y > 0. ? 1. - clamp(", HAND_OVER),
    Mutation("a tree apart's drawn crown keeps the forest's band", "template_b.html",
             "  vec2 band = aS > 0.5 ? uSparseBand : uMeshBand;", "  vec2 band = uMeshBand;", APART),
    Mutation("a tree apart's species keep the forest's reach", "template_c.html", "L = apart[at] > 0 ? Ds : D;", "L = D;", APART),
    Mutation("the sparse dial always on", "template_c.html", "(treeSparse ? TREE_LOD_SPARSE : TREE_LOD)", "TREE_LOD_SPARSE", APART),
    Mutation("the sparse dial off as the page opens", "template_c.html",
             'let treeSparse = Q.get("treesparse") !== "0";', 'let treeSparse = Q.get("treesparse") === "1";', APART),
    Mutation("trees apart's drawn crowns keep their reach with the dial off", "template_c.html",
             "sparseBand: [Ds[1] * (1 - TREE_BAND), Ds[1] * (1 + TREE_BAND)]",
             "sparseBand: [TREE_LOD_SPARSE[1] * k * (1 - TREE_BAND), TREE_LOD_SPARSE[1] * k * (1 + TREE_BAND)]", APART),
    Mutation("trees apart counted without the next row of cells", "template_b.html",
             "j <= Math.min(cz + 1, gz - 1); j++)", "j <= Math.min(cz, gz - 1); j++)", APART),
    Mutation("the lake's reflections keep the trees the mirror draws", "frag1.glsl",
             "#define VM_REFL (VM_TREES == 3 && uTreeRefl != 0 && abs(r.s.wy - LAKE) < 0.25 ? 3 : 2)", "#define VM_REFL 2",
             MIRROR),
    Mutation("the mirror draws no trees", "template_c.html",
             "    if (treesShown) drawTreeMirror(pos, rot, tanH, jit, sw, sh, L, lake);", "", MIRROR),
    Mutation("the drawn crowns past their reach", "frag5.glsl",
             "((uDbg & 2097152) == 0 && t >= TREE_FAR)) discard;", "((uDbg & 2097152) != 0 && t >= TREE_FAR)) discard;", CUT),
    Mutation("the march's canopy sheet left out within the table", "frag3.glsl",
             "(tp < TREE_FAR || (uDbg & 2097152) != 0)", "(tp < TREE_FAR || (uDbg & 2097152) == 0)", CUT),
    Mutation("the lake's reflections march the sheet within the crowns' reach", "frag3.glsl",
             "float tp = td < 1e5 ? td : distance(p, vec3(uCamPos.x, 2. * LAKE - uCamPos.y, uCamPos.z));", "float tp = td;",
             MIRROR),
    Mutation("the crown quads cover the whole view near the camera", "template_b.html",
             "  lo = max(lo - j - px, -1.); hi = min(hi - j + px, 1.);\n  bool off = vF <= 0. ||",
             "  bool near = false, ahead = false;\n"
             "  for (int k = 0; k < 8; k++) { near = near || c[k].z < ${fl(MESH_NEAR)}; ahead = ahead || c[k].z > 0.; }\n"
             "  lo = near ? vec2(-1.) : max(lo - j - px, -1.); hi = near ? vec2(1.) : min(hi - j + px, 1.);\n"
             "  bool off = !ahead || vF <= 0. ||", CLIP),
]
