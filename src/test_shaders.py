"""Every program the page links compiles and links under Khronos glslang's OpenGL ES 3.00 rules, in both float-filtering
variants (MANUAL_BILERP, for GPUs without OES_texture_float_linear): a CPU check that runs before any browser.
shaders.py assembles the sources as the page does; test_render_core checks that the page compiles exactly those."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))

import assemble  # noqa: E402
import shaders  # noqa: E402
from mesh_source import load_data  # noqa: E402
from tools import glslang  # noqa: E402
from tools.inputs import Blocked  # noqa: E402


def complaints(programs, exe):
    """glslang's output for each program, by name, that does not compile and link."""
    with tempfile.TemporaryDirectory(prefix="hog-glslang-") as folder:
        def check(program):
            vs, fs = Path(folder) / f"{program.name}.vert", Path(folder) / f"{program.name}.frag"
            vs.write_text(program.vertex, encoding="utf-8", newline="\n")
            fs.write_text(program.fragment, encoding="utf-8", newline="\n")
            run = subprocess.run([str(exe), "-l", str(vs), str(fs)], capture_output=True, text=True)
            return program.name, (run.stdout + run.stderr).strip() if run.returncode else None
        with ThreadPoolExecutor(8) as pool:
            return {name: log for name, log in pool.map(check, programs) if log}


class ShaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.exe = glslang.executable()
        except Blocked as e:
            raise unittest.SkipTest(f"blocked: {e}") from e
        cls.data = load_data()

    def test_every_program_compiles_and_links_in_both_filtering_variants(self):
        for float_linear in (True, False):
            with self.subTest(float_linear=float_linear):
                self.assertEqual(complaints(shaders.programs(shaders.cpu_defs(self.data, float_linear)), self.exe), {})

    def test_the_check_rejects_a_broken_program(self):
        good = shaders.programs(shaders.cpu_defs(self.data))[0]
        broken = good._replace(fragment=good.fragment + "\nfloat typo() { return undeclared; }\n")
        self.assertIn(good.name, complaints([broken], self.exe))


def step_rule(block, t, k, vd):
    """The primary march's step rule as `block` writes it, its own names for the distance (t), the slope factor (k) and
    the tree bound (vd) put back as T, K and VD: the hit epsilon, the slope factor, the local grid's edge bound and the
    step."""
    found = {}
    for name, pattern in (("eps", rf"float eps = (min\(0\.00045 \* {t}\b[^;]*);"),
                          ("k", rf"float {k} = (mix\([^;]*);"),
                          ("edge", r"\bdE = (s\.g\.ed[^;]*);"),
                          ("step", rf"\b{t} \+= (min\([^;]*);")):
        hits = re.findall(pattern, block)
        if len(hits) != 1:
            raise AssertionError(f"{len(hits)} matches for the {name} line")
        text = re.sub(rf"\b{t}\b", "T", hits[0])
        found[name] = re.sub(rf"\b{re.escape(vd)}\b", "VD", re.sub(rf"\b{k}\b", "K", text))
    return found


class StepRuleTests(unittest.TestCase):
    def test_the_cone_pass_replays_the_marchs_own_step_rule(self):
        # the cone pre-pass replays trace()'s march along each block's centre ray to resume it there: two copies of the
        # step rule that must not drift apart
        source = assemble.fragment()
        start = source.index("Res trace(")
        march = source[start:source.index("\n  return r;\n}", start)]
        cone = source[source.index("#elif defined(CONE_PASS)"):source.index("#elif defined(SUN_PASS)")]
        self.assertEqual(step_rule(cone, "tr", "km", "vd"), step_rule(march, "t", "k", "s.vd"))


if __name__ == "__main__":
    unittest.main()
