"""Builders and tests measure heights from the lake or the castle platform and never write those levels out as numbers,
so the scene can move to a site at another height. HOG_DATUM_SHIFT_M (assets_testing) checks this by running the asset
tests on a raised world; this test reads the code, and so also covers the rendering tests and the probe."""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assets_testing as world  # noqa: E402
from mesh_source import load_data  # noqa: E402

CHECKED = ("build_meshes.py", "mesh_*.py", "materials/*.py", "assets_testing.py", "test_*.py")
NEAR_M = 8.0          # a number this close to a reference level reads as that level written out


def float_literals(source):
    """(line, value) of each float literal in Python source, negated under a unary minus."""
    tree = ast.parse(source)
    negated = {id(n.operand) for n in ast.walk(tree) if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub)}
    return [(n.lineno, -n.value if id(n) in negated else n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, float)]


def written_levels(source, levels):
    return [(line, value, name) for line, value in float_literals(source)
            for name, level in levels.items() if abs(value - level) <= NEAR_M]


class DatumLiteralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.levels = world.levels(load_data())          # the registered site's levels, whatever the datum shift

    def test_no_reference_level_is_written_as_a_number(self):
        src = ROOT / "src"
        found = [f"{path.relative_to(src).as_posix()}:{line}: {value} reads as the {name} level"
                 for path in sorted({p for pattern in CHECKED for p in src.glob(pattern)})
                 for line, value, name in written_levels(path.read_text(encoding="utf-8"), self.levels)]
        self.assertEqual(found, [], "measure from assets_testing.levels() or the sampled ground instead")

    def test_a_level_written_out_is_found(self):
        lake = float(self.levels["lake"])
        source = (f"plant_base = {lake!r} + 0.3\n"
                  f"west = -{lake!r}\n"
                  f"elsewhere = {lake + 2 * NEAR_M!r}\n"
                  "relative = lake + 3.0\n")
        self.assertEqual(written_levels(source, self.levels), [(1, lake, "lake")])


if __name__ == "__main__":
    unittest.main()
