"""Mutations of the page that the qualification's shot tests (src/test_render_inspection.py) must catch."""
from tools.mutate.run import Mutation

TESTS = ("test_render_inspection.DisplayedShotTests",)
MUTATIONS = [
    Mutation("the canvas drawn at the internal size and stretched to the window", "template_c.html",
             "cv.width = Math.max(2, Math.round(innerWidth * dpr)); cv.height = Math.max(2, Math.round(innerHeight * dpr));",
             "cv.width = Math.max(2, Math.round(innerWidth * dpr * scale)); "
             "cv.height = Math.max(2, Math.round(innerHeight * dpr * scale));", TESTS),
]
