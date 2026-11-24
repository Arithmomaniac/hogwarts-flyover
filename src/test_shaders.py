"""Every program the page links compiles and links under Khronos glslang's OpenGL ES 3.00 rules, in both float-filtering
variants (MANUAL_BILERP, for GPUs without OES_texture_float_linear) and with mesh trees (the default) as well as without
(?trees=raymarched): a CPU check that runs before any browser. shaders.py assembles the sources as the page does;
test_render_core checks that the page compiles exactly those, and test_render_trees that the raymarched dial's page
does. The shading pass's program must also stay within a size that bounds its compile time in Edge (REBUILD.md)."""
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

MAX_P3_TOKENS = 61_000      # the shading pass's inlined size (inlined_size), a compile-time budget: REBUILD.md


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


TOKEN = re.compile(r"[A-Za-z_]\w*|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fFuU]?|&&|\|\||[-+*/%<>=!]=|\+\+|--|\S")
NAME, NUMBER = re.compile(r"[A-Za-z_]\w*"), re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF]?")
COMPARE = {"<": float.__lt__, "<=": float.__le__, ">": float.__gt__, ">=": float.__ge__, "==": float.__eq__, "!=": float.__ne__}
ASSIGN = {"=", "+=", "-=", "*=", "/=", "++", "--"}


def closing(tokens, i):
    """The index of the bracket that closes tokens[i]."""
    depth = 0
    for j in range(i, len(tokens)):
        depth += (tokens[j] in ("(", "[", "{")) - (tokens[j] in (")", "]", "}"))
        if depth == 0:
            return j
    raise ValueError(f"unclosed {tokens[i]}")


def commas(tokens):
    """tokens split at their top-level commas."""
    parts, depth, start = [], 0, 0
    for j, t in enumerate(tokens):
        depth += (t in ("(", "[", "{")) - (t in (")", "]", "}"))
        if t == "," and depth == 0:
            parts.append(tokens[start:j])
            start = j + 1
    return parts + [tokens[start:]] if tokens else parts


def statement_end(tokens, i):
    """The index after the statement that starts at tokens[i]."""
    if tokens[i] == "{":
        return closing(tokens, i) + 1
    depth = 0
    for j in range(i, len(tokens)):
        depth += (tokens[j] in ("(", "[", "{")) - (tokens[j] in (")", "]", "}"))
        if tokens[j] == ";" and depth == 0:
            return j + 1
    return len(tokens)


def inlined_size(fragment, exe):
    """The tokens of the fragment shader's main() with every call written out in place, as the HLSL compiler inlines
    them. A call that passes a number for a parameter keeps only the live branch of the callee's `if (p < 2.5)` tests
    of it (the compiler folds the rest away), and a loop counts once (ANGLE keeps loops)."""
    with tempfile.TemporaryDirectory(prefix="hog-glslang-") as folder:
        path = Path(folder) / "p3.frag"
        path.write_text(fragment, encoding="utf-8", newline="\n")
        run = subprocess.run([str(exe), "-E", str(path)], capture_output=True, text=True, check=True)
    tokens = TOKEN.findall(run.stdout)
    functions, i = {}, 0
    while i < len(tokens):
        if tokens[i] in ("(", "{"):
            end = closing(tokens, i)
            if tokens[i] == "(" and i >= 2 and NAME.fullmatch(tokens[i - 2]) and tokens[end + 1] == "{":
                body = closing(tokens, end + 1)
                params = [next((t for t in reversed(p) if NAME.fullmatch(t)), None) for p in commas(tokens[i + 1:end])]
                functions.setdefault(tokens[i - 1], []).append(([] if params == ["void"] else params, tokens[end + 2:body]))
                end = body
            i = end
        i += 1

    def value(arg, env):
        if len(arg) == 1 and NUMBER.fullmatch(arg[0]):
            return float(arg[0].rstrip("fF"))
        return env.get(arg[0]) if len(arg) == 1 else None

    def decide(cond, env):
        """The condition's truth where it tests bound parameters against numbers only, else None."""
        joins = {t for t in cond if t in ("&&", "||")}
        parts = [[]]
        for t in cond:
            if t in joins:
                parts.append([])
            else:
                parts[-1].append(t)
        tests = [len(p) == 3 and p[1] in COMPARE and (value([p[0]], env), value([p[2]], env)) for p in parts]
        if len(joins) > 1 or not all(t and None not in t for t in tests):
            return None
        truths = [COMPARE[p[1]](*t) for p, t in zip(parts, tests)]
        return any(truths) if joins == {"||"} else all(truths)

    memo = {}

    def call(name, args):
        if (name, args) not in memo:
            matching = [f for f in functions[name] if len(f[0]) == len(args)] or functions[name]
            memo[name, args] = max(size(body, {p: a for p, a in zip(params, args) if a is not None})
                                   for params, body in matching)
        return memo[name, args]

    def size(tokens, env):
        env = {p: v for p, v in env.items()
               if not any(t == p and tokens[j + 1] in ASSIGN for j, t in enumerate(tokens[:-1]))}
        total, i = 0, 0
        while i < len(tokens):
            if tokens[i] == "if" and env:
                close = closing(tokens, i + 1)
                end, truth = statement_end(tokens, close + 1), decide(tokens[i + 2:close], env)
                if truth is not None and (end == len(tokens) or tokens[end] != "else"):
                    if truth:
                        total += size(tokens[close + 1:end], env)
                        if tokens[close + 1] == "return":
                            return total
                    i = end
                    continue
            if tokens[i] in functions and tokens[i + 1] == "(":
                total += call(tokens[i], tuple(value(a, env) for a in commas(tokens[i + 2:closing(tokens, i + 1)])))
            total += 1
            i += 1
        return total

    return call("main", ())


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
            for trees in (False, True):
                with self.subTest(float_linear=float_linear, trees=trees):
                    programs = shaders.programs(shaders.cpu_defs(self.data, float_linear, trees))
                    self.assertEqual(complaints(programs, self.exe), {})

    def test_the_check_rejects_a_broken_program(self):
        good = shaders.programs(shaders.cpu_defs(self.data))[0]
        broken = good._replace(fragment=good.fragment + "\nfloat typo() { return undeclared; }\n")
        self.assertIn(good.name, complaints([broken], self.exe))

    def test_the_shading_pass_inlines_to_no_more_than_its_bound(self):
        p3 = next(p for p in shaders.programs(shaders.cpu_defs(self.data)) if p.name == "p3P")
        self.assertLessEqual(inlined_size(p3.fragment, self.exe), MAX_P3_TOKENS)


def step_rule(block, t, k, vd):
    """The primary march's step rule as `block` writes it, its own names for the distance (t), the slope factor (k) and
    the tree bound (vd) put back as T, K and VD: the hit epsilon, the slope factor, the local grid's edge bound, where
    the ray leaves the tree table and what the scene evaluation gets for it (evalTop bounds the trees within the table
    by it), and the step."""
    found = {}
    for name, pattern in (("eps", rf"float eps = (min\(0\.00045 \* {t}\b[^;]*);"),
                          ("k", rf"float {k} = (mix\([^;]*);"),
                          ("edge", rf"\bdE = ({t} < tIn\b[^;]*);"),
                          ("exit", r"\btOut = ([^;]*);"),
                          ("table", r"\bevalTop\(p, [^;]*, ([^,;]*)\);"),
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
