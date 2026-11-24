"""Mutations of the page's shaders that the shading pass's compile-size gate (src/test_shaders.py) must catch: a
shortcut for a cell of one class that inlines `surfCol` once more."""
from tools.mutate.run import Mutation

GATE = ("test_shaders.ShaderTests.test_the_shading_pass_inlines_to_no_more_than_its_bound",)
MUTATIONS = [
    Mutation("a cell of one class shaded through a surfCol site of its own", "frag4.glsl",
             "  return classesCol(c, k, xz, pix) / soft;",
             "  return all(equal(c, c.xxxx)) ? surfCol(c.x, xz, pix) : classesCol(c, k, xz, pix) / soft;", GATE,
             rebuild=False),
]
