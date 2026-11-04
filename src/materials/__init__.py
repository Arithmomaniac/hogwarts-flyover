"""The scene's (material, aux) codes in one registry: one module per family in this package, standard library only.

A code is the G-buffer pair (material, aux): material 4 is the castle, 5 the mapped building kinds, 6 boundary
walls (aux 1) and hedges (aux 0), 7 the set pieces, and 8 glazing (drawn in its own pass, never in the G-buffer).

The registry is the single source of
- the builders' constants: `code(name)` is a (material, aux) pair;
- the set pieces' branches in the shading pass (frag5.glsl), their albedos in the lake reflections (template_b.html)
  and the lamp glow of lit glazing in both, which src/assemble.py generates into the page where the sources hold
  markers (`fill`).

Each family module declares FAMILY, RANGES ((material, first aux, last aux), ...) and MATERIALS. A family may only use
codes inside its own ranges and no two families' ranges overlap, so families developed in parallel cannot collide.
Shading text is kept exactly as it reads in the page; a new set piece writes its own branch body (`p3`, the text after
the `if (k == ...)` condition) and reflection albedo (`refl`), or names a surface model (`model`) with its own values.
Every distinct body is one more branch of the shading pass's set-piece chain, and the pass compiles more slowly with
each (REBUILD.md), so set pieces that differ only in constants share a model: one branch reads each code's values from
constant arrays indexed by code.
"""
import importlib
import pkgutil
import re
from typing import NamedTuple

SET_PIECE = 7
P3_MARKER = "      //__SET_PIECES_P3__\n"
REFL_MARKER = "    //__SET_PIECES_REFL__\n"
LAMP_MARKER = "  //__LAMP_GLOW__\n"
# distinct set-piece bodies, each a branch of the shading pass. Measured: as an else-if ladder, the 58 of lane B's
# village M3 failed to start on NVIDIA (D3D11 out of memory creating the pixel shader), and any ladder past 110 fails to
# compile (ANGLE: expression too complex); 200 independent ifs started on both GPUs but took 25-29 s to link
MAX_P3_BRANCHES = 40


class Model(NamedTuple):
    """A surface model: the shading pass's statements for every set piece that names it, with `{param}` wherever a
    code's own value goes; each set piece gives every parameter a literal of its type in `values`."""
    body: str
    params: tuple               # (name, GLSL type) for each parameter
    note: str


# shade()'s defaults are spec 0. and shin 32.: a matte set piece gives those, which leave its shading unchanged
MODELS = {
    "flat": Model("alb = {alb}; spec = {spec}; shin = {shin};", (("alb", "vec3"), ("spec", "float"), ("shin", "float")),
                  "flat colours, matte or glossy"),
    "speckled": Model("alb = {alb} * ({lo} + {amp} * vnoise(p.xz * {freq} + p.y * {rise})); spec = {spec}; shin = {shin};",
                      (("alb", "vec3"), ("lo", "float"), ("amp", "float"), ("freq", "float"), ("rise", "float"),
                       ("spec", "float"), ("shin", "float")), "colours speckled by value noise"),
}


class Material(NamedTuple):
    name: str
    mat: int
    aux: int
    count: int = 1              # consecutive codes aux .. aux + count - 1 (class k, tower i)
    note: str = ""
    p3: str | None = None       # set piece: the shading pass's branch after its condition; None where hand-written
    refl: str | None = None     # set piece: the lake reflection's albedo; None falls back to the default set piece's
    refl_tail: str = ""         # what follows the reflection line's semicolon (an aligned comment)
    glow: str | None = None     # lit glazing: lampGlow's term before the dusk ramp (page template text)
    default: bool = False       # the set piece that every unregistered code is shaded as
    model: str | None = None    # set piece: the surface model (MODELS) shading it, in place of p3
    values: tuple = ()          # with a model: (parameter, literal) for each of the model's parameters

    @property
    def codes(self):
        return [(self.mat, a) for a in range(self.aux, self.aux + self.count)]

    @property
    def code(self):
        return self.mat, self.aux


def set_piece(aux, name, **fields):
    if isinstance(fields.get("values"), dict):
        fields["values"] = tuple(fields["values"].items())
    return Material(name, SET_PIECE, aux, **fields)


def discover():
    """{family: (ranges, materials)} from every module of this package."""
    found = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        module = importlib.import_module(f"{__name__}.{info.name}")
        if module.FAMILY in found:
            raise ValueError(f"two material modules declare the family {module.FAMILY}")
        found[module.FAMILY] = (tuple(module.RANGES), tuple(module.MATERIALS))
    return found


def validate(families):
    """The registered materials, checked: ranges disjoint across families, codes unique and inside their family's
    ranges, names unique, one default set piece, shading text only on set pieces, one comment per reflection group."""
    owner, out, names = {}, [], set()
    for family, (ranges, materials) in families.items():
        for mat, lo, hi in ranges:
            for aux in range(lo, hi + 1):
                if (mat, aux) in owner and owner[mat, aux] != family:
                    raise ValueError(f"families {owner[mat, aux]} and {family} both reserve material {mat}, aux {aux}")
                owner[mat, aux] = family
    used = {}
    for family, (ranges, materials) in families.items():
        for m in materials:
            if m.name in names:
                raise ValueError(f"material name {m.name!r} is registered twice")
            names.add(m.name)
            for c in m.codes:
                if owner.get(c) != family:
                    raise ValueError(f"{family} material {m.name!r} uses {c}, which is "
                                     + (f"reserved by {owner[c]}" if c in owner else "outside every family's ranges"))
                if c in used:
                    raise ValueError(f"materials {used[c]!r} and {m.name!r} both claim {c}")
                used[c] = m.name
            if m.mat != SET_PIECE and (m.p3 or m.refl or m.default or m.model):
                raise ValueError(f"{m.name!r}: shading text belongs to set pieces only")
            if m.model is not None:
                _check_model(m)
            out.append(m)
    defaults = [m for m in out if m.default]
    if len(defaults) != 1 or not (defaults[0].p3 and defaults[0].refl):
        raise ValueError("exactly one default set piece, with a shading branch and a reflection albedo, is required")
    branches = len(_groups(out, "p3"))
    if branches > MAX_P3_BRANCHES:
        raise ValueError(f"{branches} distinct set-piece shading bodies, over the cap of {MAX_P3_BRANCHES}: share a "
                         "surface model, or another set piece's body, before writing a new one (REBUILD.md)")
    for codes, _, tails in _groups(out, "refl"):
        if sum(bool(t) for t in tails) > 1:
            raise ValueError(f"set pieces {codes} share a reflection line but carry more than one comment")
    return sorted(out, key=lambda m: m.code)


_CACHE = None


def _literal_type(value):
    """The GLSL type of a model value: a float literal (with a point or an exponent), or a vector of one float literal
    or of exactly as many as it has components (GLSL refuses any other count)."""
    found = re.fullmatch(r"\s*vec([234])\((.*)\)\s*", value)
    parts = [p.strip() for p in found.group(2).split(",")] if found else [value.strip()]
    if not all(re.fullmatch(r"-?(\d+\.\d*|\.\d+|\d+(\.\d*)?[eE][-+]?\d+)", p) for p in parts):
        raise ValueError(f"{value!r} is not a float or vector literal")
    if found and len(parts) not in (1, int(found.group(1))):
        raise ValueError(f"{value!r} gives {len(parts)} components to a vec{found.group(1)}")
    return f"vec{found.group(1)}" if found else "float"


def _check_model(m):
    model = MODELS.get(m.model)
    if model is None:
        raise ValueError(f"{m.name!r}: there is no surface model {m.model!r}; the models are {sorted(MODELS)}")
    if m.p3 is not None or m.default:
        raise ValueError(f"{m.name!r} has both shading text and a model; a default set piece has no model")
    given, types = [param for param, _ in m.values], dict(model.params)
    if sorted(given) != sorted(types):
        raise ValueError(f"{m.name!r} gives the values {given}; model {m.model!r} takes {list(types)}")
    for param, value in m.values:
        try:
            kind = _literal_type(value)
        except ValueError as e:
            raise ValueError(f"{m.name!r}, {param}: {e}") from None
        if kind != types[param]:
            raise ValueError(f"{m.name!r}, {param}: {value!r} is a {kind}, not a {types[param]}")


def model_bodies(mats):
    """Each surface model's branch body: constant arrays of every parameter over its set pieces' codes, then the model's
    statements reading the arrays at `k`. A code in the arrays' span that is not the model's never reads them."""
    out = {}
    for name, model in MODELS.items():
        members = [m for m in mats if m.model == name]
        if not members:
            continue
        by_code = {a: dict(m.values) for m in members for a in range(m.aux, m.aux + m.count)}
        first, last = min(by_code), max(by_code)
        filler = by_code[first]
        decls, reads = [], {}
        for param, kind in model.params:
            array, size = f"{name.upper()}_{param.upper()}", last - first + 1
            items = ", ".join(by_code.get(a, filler)[param] for a in range(first, last + 1))
            decls.append(f"const {kind} {array}[{size}] = {kind}[{size}]({items});")
            reads[param] = f"{array}[k - {first}]"
        out[name] = f" {{ {' '.join(decls)} {model.body.format(**reads)} }}   // {model.note}"
    return out


def materials():
    global _CACHE
    if _CACHE is None:
        _CACHE = validate(discover())
    return _CACHE


def code(name):
    """The (material, aux) pair of a registered material, by name."""
    for m in materials():
        if m.name == name:
            return m.code
    raise KeyError(f"no material named {name!r}")


def registered(mat, aux):
    return any((mat, aux) in m.codes for m in materials())


def _groups(mats, field):
    """Set pieces sharing the same `field` text: [(sorted auxes, text, tails)], by smallest aux, default excluded. A
    modelled set piece's shading text is its model's body, so a model's set pieces share one branch."""
    shared = model_bodies(mats) if field == "p3" else {}
    groups = {}
    for m in mats:
        text = shared[m.model] if field == "p3" and m.model else getattr(m, field)
        if m.mat == SET_PIECE and text is not None and not m.default:
            groups.setdefault(text, []).append(m)
    return sorted(((sorted(a for m in g for a in range(m.aux, m.aux + m.count)), text, [m.refl_tail for m in g])
                   for text, g in groups.items()), key=lambda group: group[0])


def _condition(auxes):
    return " || ".join(f"k == {a}" for a in auxes)


def p3_branches(mats):
    """The shading pass's set-piece chain: an independent branch per distinct body, each marking its codes shaded, then
    the default set piece's body for every code that none shaded. An else-if ladder compiles ever more slowly with its
    depth, and past a depth that depends on its bodies does not start; independent branches start at 200 (REBUILD.md)."""
    default = next(m for m in mats if m.default)
    return ("      bool shaded = false;\n"
            + "".join(f"      if ({_condition(auxes)}) {{{body}\n        shaded = true; }}\n"
                      for auxes, body, _ in _groups(mats, "p3"))
            + f"      if (!shaded){default.p3}\n")


def refl_lines(mats):
    """The lake reflections' set-piece albedos: one line per distinct albedo, then the default's."""
    lines = [f"    if ({_condition(auxes)}) return {refl};{next((t for t in tails if t), '')}\n"
             for auxes, refl, tails in _groups(mats, "refl")]
    default = next(m for m in mats if m.default)
    return "".join(lines) + f"    return {default.refl};\n"


def lamp_lines(mats):
    """lampGlow's body: per material, the glow of each lit glazing code."""
    by_mat = {}
    for m in mats:
        if m.glow:
            by_mat.setdefault(m.mat, []).extend((a, m.glow) for a in range(m.aux, m.aux + m.count))
    return "".join(f"  if (mat == {mat}) return " + " : ".join(f"k == {a} ? {g}" for a, g in sorted(terms))
                   + " : vec3(0.);\n" for mat, terms in sorted(by_mat.items()))


GENERATORS = {P3_MARKER: p3_branches, REFL_MARKER: refl_lines, LAMP_MARKER: lamp_lines}


def fill(text, *markers):
    """`text` with each named marker line, which must occur exactly once, replaced by its generated text."""
    mats = materials()
    for marker in markers:
        found = text.count(marker)
        if found != 1:
            raise ValueError(f"marker {marker.strip()} occurs {found} times, not once")
        text = text.replace(marker, GENERATORS[marker](mats))
    return text
