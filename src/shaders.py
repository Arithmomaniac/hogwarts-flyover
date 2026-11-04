"""The page's WebGL programs, assembled on the CPU as the page assembles them, for the glslang check (test_shaders.py).

The page builds its programs in JavaScript (template_b.html): the raymarcher's fragment source with the #defines and the
lamp glow put in, ten programs from it chosen by a #define line or written out, and the mesh passes' template literals.
This module reads that script through assemble.py and takes the same steps. A step it does not recognise raises
ValueError, so a change in the page's shader plumbing stops the check rather than leaving it checking something else.
test_render_core.py compares the result with the sources the page hands to WebGL.
"""
import ast
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import NamedTuple

import assemble

STRING = r'"(?:[^"\\\n]|\\.)*"'
LITERAL = rf"(?:{STRING}|`[^`]*`)"
GROUND_KEYS = ()                     # defines the page samples from its terrain (none since the raymarched objects went)
ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'"}


class Program(NamedTuple):
    name: str                # the page's variable for it
    vertex: str
    fragment: str


def fl(x):
    """JavaScript's (+x).toFixed(4): the exact binary value rounded half away from zero, with no sign on zero."""
    x = float(x)
    if x < 0:
        return "-" + fl(-x)
    return str(Decimal(abs(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def js_number(x):
    """String(x) for a finite JavaScript number."""
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def js_replace(text, old, new, what):
    """String.prototype.replace with string arguments, which replaces the first occurrence: here it must be the only one,
    and the replacement may hold no '$', which the method would interpret."""
    if text.count(old) != 1:
        raise ValueError(f"{what}: {old!r} occurs {text.count(old)} times, not once")
    if "$" in new:
        raise ValueError(f"{what}: the replacement holds '$', which String.replace interprets")
    return text.replace(old, new, 1)


class Source:
    """template_b.html, the script that builds the programs, with the registry's sections filled in."""

    def __init__(self):
        self.text = assemble.part_b()

    def one(self, pattern, what):
        found = re.findall(pattern, self.text, re.S)
        if len(found) != 1:
            raise ValueError(f"{what}: {len(found)} matches in template_b.html, expected one")
        return found[0]

    def number(self, name):
        return float(self.one(rf"\b{name} = (-?\d+(?:\.\d+)?)[,;]", f"the number {name}"))


class Script(Source):
    """The script evaluated for one set of #defines, with the fragment source as the page builds it."""

    def __init__(self, defs):
        super().__init__()
        self.defs = dict(defs)
        self.used = set()
        self._check_fragment_statement()
        self.fragment = js_replace(js_replace(assemble.fragment(), "//__DEFINES__",
                                              "\n".join(f"#define {k} {v}" for k, v in self.defs.items()), "fragSrc"),
                                   "//__LAMPS__", self.literal("lampGlsl"), "fragSrc")

    def _check_fragment_statement(self):
        self.one(re.escape('const fragSrc = $("fragSrc").textContent.replace("//__DEFINES__", Object.entries(defs).map('
                           '([k, v]) => `#define ${k} ${v}`).join("\\n"))\n    .replace("//__LAMPS__", lampGlsl);'),
                 "the fragSrc statement")

    def string(self, literal):
        """The value of a JavaScript string or template literal."""
        if literal.startswith('"'):
            return re.sub(r"\\(.)", lambda m: ESCAPES[m.group(1)], literal[1:-1])
        body = literal[1:-1]
        if "\\" in body:
            raise ValueError("a template literal with escapes")
        return re.sub(r"\$\{([^}]*)\}", lambda m: self.value(m.group(1).strip()), body)

    def literal(self, name):
        self.used.add(name)
        return self.string(self.one(rf"\bconst {name} = ({LITERAL});", f"the {name} literal"))

    def arithmetic(self, expr):
        def run(node):
            if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
                return OPERATORS[type(node.op)](run(node.left), run(node.right))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -run(node.operand)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.Name):
                return self.number(node.id)
            raise ValueError(f"shaders.py cannot evaluate {expr!r}")
        return run(ast.parse(expr, mode="eval").body)

    def define(self, name):
        found = re.search(rf"#define {name} (\S+)", self.fragment)
        if found is None:
            raise ValueError(f"defOf({name!r}): no such #define in the fragment source")
        return found.group(1)

    def value(self, expr):
        """The text of one ${expr} in the page's shader literals."""
        if m := re.fullmatch(r"defs\.(\w+)", expr):
            return str(self.defs[m.group(1)])
        if m := re.fullmatch(r'defOf\("(\w+)"\)', expr):
            return self.define(m.group(1))
        if m := re.fullmatch(r"fl\((.*)\)", expr):
            return fl(self.arithmetic(m.group(1)))
        if re.fullmatch(r"[A-Za-z_]\w*", expr):
            return self.literal(expr) if re.search(rf"\bconst {expr} = `", self.text) else js_number(self.number(expr))
        raise ValueError(f"shaders.py does not know how the page evaluates ${{{expr}}}")

    def source(self, ref):
        """A shader source the page names: a literal, a derived source or a string written in place."""
        if ref.startswith('"'):
            return self.string(ref)
        if ref == "vsSrc":
            return self.string(self.one(rf"\bconst vsSrc = ({STRING});", "vsSrc"))
        if ref == "reflVs":
            text = self.literal("meshVs")
            chain = self.one(rf"\bconst reflVs = meshVs((?:\.replace\({LITERAL}, {LITERAL}\))+);", "reflVs")
            for old, new in re.findall(rf"\.replace\(({LITERAL}), ({LITERAL})\)", chain, re.S):
                text = js_replace(text, self.string(old), self.string(new), "reflVs")
            return text
        return self.literal(ref)


OPERATORS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
             ast.Div: lambda a, b: a / b}


def _items(body, pattern, what):
    """The matches of `pattern` that make up a comma-separated JavaScript list, which must hold nothing else."""
    found = list(re.finditer(pattern, body, re.S))
    rest = re.sub(pattern, "", body, flags=re.S)
    if not found or rest.replace(",", "").strip():
        raise ValueError(f"{what}: unrecognised entries {rest.strip()[:80]!r}")
    return found


def programs(defs):
    """Every program the page links, in the order it hands their sources to WebGL (vertex shader, then fragment)."""
    s = Script(defs)
    names, links = s.one(r"\[([\w, ]+)\] = \[((?:await linked\(ps\[\d+\]\),?\s*)+)\];", "the raymarcher programs")
    names = [n.strip() for n in names.split(",")]
    if [int(i) for i in re.findall(r"ps\[(\d+)\]", links)] != list(range(len(names))):
        raise ValueError("the raymarcher programs are not linked from ps in order")
    variants = _items(s.one(r"const ps = \[(.*?)\]\.map\(\(fs\) => program\(fs\)\);", "the ps list"),
                      rf"fragSrc\.replace\(({STRING}), ({STRING})\)|\b(\w+)\b", "the ps list")
    if len(variants) != len(names):
        raise ValueError(f"{len(variants)} raymarcher sources for {len(names)} programs")
    vertex = s.source("vsSrc")
    out = []
    for name, v in zip(names, variants):
        old, new, ref = v.groups()
        fragment = js_replace(s.fragment, s.string(old), s.string(new), name) if old else s.source(ref)
        out.append(Program(name, vertex, fragment))
    mesh_names, calls = s.one(r"\[(meshP[\w, ]*)\] = await Promise\.all\(\[(.*?)\]\.map\(linked\)\);", "the mesh programs")
    calls = _items(calls, rf"program\(({STRING}|\w+)(?:, ({STRING}|\w+))?\)", "the mesh programs")
    mesh_names = [n.strip() for n in mesh_names.split(",")]
    if len(calls) != len(mesh_names):
        raise ValueError(f"{len(calls)} mesh programs for {len(mesh_names)} names")
    for name, call in zip(mesh_names, calls):
        fs, vs = call.groups()
        out.append(Program(name, s.source(vs or "vsSrc"), s.source(fs)))
    shaders = set(re.findall(r"\bconst (\w+) = `#version", s.text))
    if shaders - s.used:
        raise ValueError(f"shader literals no program uses: {sorted(shaders - s.used)}")
    return out


def cpu_defs(data, float_linear=True):
    """The page's #define values (template_b.html's `defs`), from the registered world."""
    s = Source()
    meta, extras = data["meta"], data["meta"]["extras"]
    v2 = lambda a: f"vec2({fl(a[0])},{fl(a[1])})"
    poi = lambda name: next(p for p in extras["poi"] if p["name"] == name)["p"]
    brg, towers = meta["pitch"]["brg"] * math.pi / 180, meta["towers"]
    tree_n = s.number("TREE_N")
    defs = {
        "LAKE": fl(meta["lake"]), "PITCH_C": v2(meta["pitch"]["c"]), "PITCH_DIR": v2([math.sin(brg), math.cos(brg)]),
        "STATION": v2(poi("Hogsmeade station")), "GRAWP": v2(poi("Grawp's clearing")), "NTW": str(len(towers)),
        "CAS_CLS": f"vec2[6]({','.join(v2(c) for c in meta['castle']['cls'])})", "TREE_N": js_number(tree_n),
        "TREE_G0": fl(tree_n / 2),
    }
    defs.update(re.findall(r'\b([A-Z_]+): "([^"]*)"', s.one(r"TREE_G0: fl\(TREE_N / 2\),(.*?)\n  \};", "the fixed defs")))
    grids = s.one(r"const CG = \[(.*?)\];", "the cone grids")
    for key, grid in zip(("CG_FINE", "CG_MID", "CG_FAR"), re.findall(r"\{([^}]*)\}", grids)):
        g = dict((k, float(v)) for k, v in re.findall(r"(\w): (-?\d+)", grid))
        defs[key] = f"vec4({fl(g['c'])},{fl(g['o'])},{fl(g['s'])},{fl(g['n'])})"
    if not float_linear:
        defs["MANUAL_BILERP"] = "1"
    return defs
