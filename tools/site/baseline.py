"""The frozen baseline page re-frozen at a site, the reference for timing and parity there.

It is the renderer of the input baseline-s4-idle (main at 4c687c9, built at Scardroy), with the castle's window band,
shading ramp and fog made relative to the lake as b46e680 made the page's own, the page's colours (src/frag4.glsl
classCol) for the surface classes the site's data adds, such as the overgrown garden's, but its own for the classes it
had, the site's build/data.js in place of Scardroy's data, and the page's tour (its waypoints through the path the
camera flies) in place of the frozen one, so S6 and parity, which set both pages at the same tour times, see the same
views. The file is pinned as an input (baseline-x), never committed: like the S0 input, it is made in each worktree,
kept in build/inputs/ where tools.inputs finds it, and checked against its pin.

    python -X utf8 -m tools.site.baseline [OUT.html] [--data build/data.js] [--page hogwarts-flyover.html]
"""
import argparse
import json
import sys
from pathlib import Path

from tools import inputs
from tools.paths import BUILD, PAGE, ROOT, sha256

LEVELS = (                  # Scardroy's absolute levels in the frozen shaders, and the lake-relative ones of b46e680
    ("exp(-b * (ro.y - 150.))", "exp(-b * (ro.y - (LAKE - 3.)))"),
    ("windowMask(q, faceCoord(q, n, true), 192.,", "windowMask(q, faceCoord(q, n, true), LAKE + 39.,"),
    ("windowMask(p, wx, 192.,", "windowMask(p, wx, LAKE + 39.,"),
    ("smoothstep(188., 200., p.y)", "smoothstep(LAKE + 35., LAKE + 47., p.y)"),
)
CLASS_COLOURS = "vec3 classCol("    # the surface classes' colours: the frozen renderer colours classes 0 to
FROZEN_CLASSES = 8                  # FROZEN_CLASSES - 1, and takes the page's colours for those the site's data adds
SHADER = ROOT / "src" / "frag4.glsl"
DATA = "const DATA="
TOUR = "const WP = ["            # a page's tour: its waypoints, then the path the camera flies through them (pathAt)
SECTION = "// ----"              # the script's next section, which ends the tour
NAME = "baseline-x"


def function(text, head):
    """The text of the one GLSL function in `text` that begins with `head`, to the brace that closes it at a line's
    start."""
    if text.count(head) != 1:
        raise ValueError(f"the text holds {head!r} {text.count(head)} times, not once")
    start = text.index(head)
    return text[start: text.index("\n}\n", start) + 3]


def with_added_classes(frozen, page):
    """The frozen renderer's classCol `frozen`, whose last line colours class FROZEN_CLASSES - 1 and any higher one,
    given the lines of the page's classCol `page` for the classes it colours after that one, and nothing else of the
    page's: parity still shows a change to the colour of a class the frozen renderer had."""
    branch = f"  if (c < {FROZEN_CLASSES - 1}.5) return "
    if branch not in page:
        return frozen
    last = frozen.rindex("\n  return ") + 1
    end = frozen.index("\n", last) + 1
    added = page[page.index("\n", page.index(branch)) + 1: page.rindex("}")]
    return frozen[:last] + branch + frozen[last:end].lstrip()[len("return "):] + added + frozen[end:]


def data_span(page):
    """(start, end) of a page's one DATA statement, `const DATA={...};`."""
    if page.count(DATA) != 1:
        raise ValueError(f"the page holds {page.count(DATA)} DATA statements, not one")
    start = page.index(DATA)
    end = json.JSONDecoder().raw_decode(page, start + len(DATA))[1]
    if page[end] != ";":
        raise ValueError("the page's DATA statement does not end with its object")
    return start, end + 1


def page_data(page):
    """The DATA object a page carries."""
    start, end = data_span(page)
    return json.loads(page[start + len(DATA): end - 1])


def tour_span(page):
    """(start, end) of a page's tour: its waypoints (const WP) through the path the camera flies (pathAt), up to the
    script's next section."""
    if page.count(TOUR) != 1:
        raise ValueError(f"the page holds {page.count(TOUR)} tours, not one")
    start = page.index(TOUR)
    return start, page.index(SECTION, page.index("function pathAt(", start))


def tour_text(page):
    """The text of a page's tour."""
    start, end = tour_span(page)
    return page[start:end]


def same_data(a, b):
    """Whether the pages at paths `a` and `b` carry the same DATA object."""
    a, b = (page_data(Path(path).read_bytes().decode("utf-8")) for path in (a, b))
    return a == b


def check_baseline(baseline, page):
    """Refuse a frozen baseline that carries other data or flies another tour than `page`: S6 and the pass timer set both
    at the same tour times, and the camera follows the waypoints, the ground and the trees, so either would set
    different views side by side."""
    a, b = (Path(path).read_bytes().decode("utf-8") for path in (baseline, page))
    why = "carries other data" if page_data(a) != page_data(b) else \
        "flies another tour" if tour_text(a) != tour_text(b) else None
    if why:
        raise inputs.Blocked(f"{baseline} {why} than {page}: make {NAME} again from that page and its build/data.js "
                             "with python -X utf8 -m tools.site.baseline, and pin it")


def rebased(page, data, tour=None, shader=None):
    """The frozen page's text `page` with LEVELS made lake-relative, its classCol given the colours of the classes that
    the one in `shader` (the text of src/frag4.glsl by default) adds, its DATA statement replaced by `data`, the text of
    a build/data.js, and its tour replaced by `tour`, the text of a page's tour (tour_text), when one is given."""
    for old, new in LEVELS:
        if page.count(old) != 1:
            raise ValueError(f"the frozen page holds {old!r} {page.count(old)} times, not once")
        page = page.replace(old, new)
    shader = SHADER.read_text(encoding="utf-8") if shader is None else shader
    frozen = function(page, CLASS_COLOURS)
    page = page.replace(frozen, with_added_classes(frozen, function(shader, CLASS_COLOURS)))
    statement = data.strip()
    if not statement.startswith(DATA) or not statement.endswith(";"):
        raise ValueError("expected the text of a build/data.js")
    start, end = data_span(page)
    page = page[:start] + statement + page[end:]
    if tour is not None:
        start, end = tour_span(page)
        page = page[:start] + tour + page[end:]
    return page


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path, nargs="?", default=BUILD / "inputs" / f"{NAME}.html")
    parser.add_argument("--data", type=Path, default=BUILD / "data.js")
    parser.add_argument("--page", type=Path, default=PAGE, help="the page whose tour the baseline flies")
    args = parser.parse_args(argv)
    page = inputs.require("baseline-s4-idle").read_bytes().decode("utf-8")
    tour = tour_text(args.page.read_bytes().decode("utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(rebased(page, args.data.read_bytes().decode("ascii"), tour).encode("utf-8"))
    pin = f"size {args.out.stat().st_size} and sha256 {sha256(args.out)}"
    found = inputs.problems(NAME, args.out) if NAME in inputs.entries() else [f"{NAME} is not pinned yet"]
    if found:
        print(f"{args.out} differs from the pinned {NAME}: {found[0]}. If the change is intended, pin {NAME} in"
              f" tools/inputs.lock.json with {pin}.")
        return 1
    print(f"{args.out} holds the pinned {NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
