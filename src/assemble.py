"""Assemble the single-file flyover: template + shader parts + packed data.

The set pieces' shading branches, their reflection albedos and the lamp glow come from the material registry
(src/materials), filled in where the sources hold its markers. The meshes' decoder is meshoptimizer's, inlined from
src/vendor after src/meshopt.py checks it against its pinned hash. Standard library only. src/shaders.py reads the
fragment source and the program-building script from here.
"""
import os, sys

import materials
import meshopt

SRC = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(SRC)


def read(*p):
    return open(os.path.join(*p), encoding="utf-8").read()


def once(text, marker, value):
    """`text` with `marker`, which must occur exactly once, replaced by `value`."""
    found = text.count(marker)
    if found != 1:
        raise ValueError(f"{marker} occurs {found} times in the page template, not once")
    return text.replace(marker, value)


def fragment():
    """The raymarcher's fragment source: the text of the page's fragSrc script."""
    frag = materials.fill("".join(read(SRC, f"frag{i}.glsl") for i in range(1, 6)), materials.P3_MARKER)
    assert "</script" not in frag
    return frag


def part_b():
    """The script that builds the programs, with the registry's reflection albedos and lamp glow filled in."""
    return materials.fill(read(SRC, "template_b.html"), materials.REFL_MARKER, materials.LAMP_MARKER)


def page(meshes_js=None):
    """The page, with build/meshes.js or the payload at `meshes_js`."""
    html = read(SRC, "template.html")
    html = once(html, "/*__FRAG__*/", fragment())
    html = once(html, "<!--__PART_B__-->", part_b() + read(SRC, "template_c.html"))
    data = read(WORK, "build", "data.js")
    assert "</script" not in data
    html = once(html, "/*__DATA__*/", data.strip())
    decoder = meshopt.vendored("meshopt_decoder.cjs")      # checked against its pinned SHA256
    assert "</script" not in decoder
    html = once(html, "/*__MESHOPT__*/", decoder.strip())
    meshes = read(meshes_js or os.path.join(WORK, "build", "meshes.js"))    # python src/build_meshes.py
    assert "</script" not in meshes
    return once(html, "/*__MESHES__*/", meshes.strip())


if __name__ == "__main__":
    args = sys.argv[1:]                              # [OUT] [--meshes PAYLOAD]
    meshes_js = args.pop(args.index("--meshes") + 1) if "--meshes" in args else None
    args = [a for a in args if a != "--meshes"]
    OUT = args[0] if args else os.path.join(WORK, "hogwarts-flyover.html")
    html = page(meshes_js)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8", newline="\n").write(html)
    print("wrote", OUT, f"{len(html) / 1e6:.2f} MB")
