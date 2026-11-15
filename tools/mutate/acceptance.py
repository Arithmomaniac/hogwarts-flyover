"""Mutations of the page that the acceptance checks (src/test_render_acceptance.py) must catch: the G-buffer pass ignoring
the depth its pre-pass left, so a farther mesh drawn later shows through nearer masonry (GH2); a converged still view
that does not wake when the camera moves on, leaving the paused glass on screen (GH4); a GL error on each path the page
takes: glass, the static and near shadow layers, the lake's reflections and the cone pre-pass (PO1/PO2); a resize that
restarts the tour or puts the hour back, and raymarching passes that ignore the quality (SC7); and a harbour recess
closed just behind its portal (HB1), judged once its test no longer fails as expected.

The page mutations rebuild the copy's page and need a browser (set HOG_GPU):

    python -X utf8 -m tools.mutate.run acceptance
"""
from tools.mutate.run import Mutation

GLASS = "test_render_acceptance.GlassTests."
MASONRY = (GLASS + "test_foreground_masonry_hides_the_plants_behind_glass",)
PAUSE = (GLASS + "test_a_pause_past_glass_leaves_no_ghost_once_the_camera_moves_on",)
HARBOUR = ("test_render_acceptance.HarbourMouthViewTests",)
PROGRAMS = ("test_render_acceptance.ProgramTests",)
TOUR = ("test_render_acceptance.TourStateTests",)
MUTATIONS = [
    Mutation("the G-buffer pass ignores the pre-pass's depth", "template_c.html",
             "gl.bindFramebuffer(gl.FRAMEBUFFER, gFbo); gl.depthFunc(gl.LEQUAL);",
             "gl.bindFramebuffer(gl.FRAMEBUFFER, gFbo); gl.depthFunc(gl.ALWAYS);", MASONRY),
    Mutation("a converged still view does not wake when the camera moves", "template_c.html",
             "const sig = `${pos.join()}|${Array.from(f).join()}|", "const sig = `${Array.from(f).join()}|", PAUSE),
    Mutation("the glass pass sets its pass as a float", "template_c.html",
             "gl.uniform1i(UG.uGlassPass, k);", "gl.uniform1f(UG.uGlassPass, k);", PROGRAMS),
    Mutation("the static shadow layers attach layers past their array", "template_b.html",
             "gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, shTexM, 0, k);",
             "gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, shTexM, 0, k + SH_LAYERS);", PROGRAMS),
    Mutation("the near layer attaches a layer past its array", "template_b.html",
             "gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, shTexM, 0, shN);",
             "gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, shTexM, 0, SH_LAYERS);", PROGRAMS),
    Mutation("the mirror sets its dusk as an integer", "template_c.html",
             "gl.uniform1f(UL.uDusk, L.dusk);", "gl.uniform1i(UL.uDusk, L.dusk);", PROGRAMS),
    Mutation("the cone pre-pass copies past the tree table", "template_c.html",
             "gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, TREE_N, 0, 0, cw, ch);",
             "gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, TREE_N, 0, 0, cw + TREE_N, ch);", PROGRAMS),
    Mutation("a resize restarts the tour", "template_c.html",
             "const dpr = Math.min(devicePixelRatio || 1, 1);",
             'simT = +(Q.get("t") || 0); const dpr = Math.min(devicePixelRatio || 1, 1);', TOUR),
    Mutation("a resize puts the hour back to the opening's", "template_c.html",
             "const dpr = Math.min(devicePixelRatio || 1, 1);",
             'tod = +(Q.get("tod") || 18.5); const dpr = Math.min(devicePixelRatio || 1, 1);', TOUR),
    Mutation("the raymarching passes ignore the quality", "template_c.html",
             "gl.uniform1i(UU.uQual, qual);", "gl.uniform1i(UU.uQual, 1);", TOUR),
    Mutation("the harbour recess closed just behind its portal", "mesh_grounds.py",
             'a0, back, a1 = PORTAL_M, rec["depth"] - BACK_IN_M, rec["depth"] + ROOF_PAST_M',
             'a0, back, a1 = PORTAL_M, PORTAL_M + 1.0, rec["depth"] + ROOF_PAST_M', HARBOUR),
]
