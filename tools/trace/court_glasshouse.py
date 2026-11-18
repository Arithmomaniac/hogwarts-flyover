"""Trace the court glasshouse on White Hound's grounds map: the small white T inside the walled north court.

Fills the white pixels connected to the component's census point, splits the fill into its wide range and its narrow
wing (the rows or columns, whichever splits the T, whose run of white is at least WIDE of the widest), and takes each
part's rectangle (the wing the longest run of narrow rows) grown half a pixel onto the ink line round it. The map comes from the White Hound cache.

    python -X utf8 -m tools.trace.court_glasshouse    # writes tools/manifest/data/greenhouses/court-glasshouse-trace.json
"""
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import Polygon

from tools import whcache
from tools.paths import BUILD, ROOT
from mesh_source import load_georef
from railway import grounds_world

OUT = ROOT / "tools" / "manifest" / "data" / "greenhouses" / "court-glasshouse-trace.json"
COURT = np.array([-49.25, 61.37])          # component 19's census point
R = 30                                     # half the search window, map pixels
WIDE = 0.7
S = 8                                      # preview magnification


def split(mask):
    """((r0, r1, c0, c1) range, (r0, r1, c0, c1) wing, misfit pixels) splitting a T's rows; the wing's columns are
    the median of its rows' runs."""
    rows = np.flatnonzero(mask.any(axis=1))
    width = mask.sum(axis=1)
    wide = width >= WIDE * width.max()
    head = rows[wide[rows]]
    narrow = rows[~wide[rows]]
    if not len(narrow):
        return None
    runs = np.split(narrow, np.flatnonzero(np.diff(narrow) > 1) + 1)
    stem = max(runs, key=len)                  # the wing; any other narrow rows are the head's ragged edge
    cols = lambda rr: (int(np.median([np.flatnonzero(mask[r])[0] for r in rr])),
                       int(np.median([np.flatnonzero(mask[r])[-1] for r in rr])) + 1)
    rng = (int(head.min()), int(head.max()) + 1, *cols(head))
    wing = (int(stem.min()), int(stem.max()) + 1, *cols(stem))
    fit = np.zeros_like(mask)
    for r0, r1, c0, c1 in (rng, wing):
        fit[r0:r1, c0:c1] = True
    misfit = int((fit != mask).sum())
    if wing[0] >= rng[1]:                      # the wing's near edge on the range's (grown half a pixel each side)
        wing = (rng[1] + 1, *wing[1:])
    else:
        wing = (wing[0], rng[0] - 1, *wing[2:])
    return rng, wing, misfit


def trace():
    img = np.asarray(Image.open(whcache.path("img/Hogwarts_map_unlabelled.gif")).convert("RGB")).astype(int)
    geo = load_georef()
    o, ex, ey = (grounds_world(np.array(p, float), geo) for p in ((0, 0), (1, 0), (0, 1)))
    seed = np.linalg.solve(np.column_stack([ex - o, ey - o]), COURT - o)
    x0, y0 = int(seed[0] - R), int(seed[1] - R)
    win = img[y0:y0 + 2 * R, x0:x0 + 2 * R]
    white = np.all(win == 255, axis=-1)
    lab, _ = ndimage.label(white)
    ys, xs = np.nonzero(white)
    near = np.argmin(np.hypot(xs - (seed[0] - x0), ys - (seed[1] - y0)))
    mask = lab == lab[ys[near], xs[near]]
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise SystemExit("the white fill reaches the window's edge")
    fits = [f for f in (split(mask), split(mask.T)) if f]
    if not fits:
        raise SystemExit("the white fill has no narrow wing")
    best = min(range(len(fits)), key=lambda k: fits[k][2])
    rng, wing, misfit = fits[best]
    corners = []
    for r0, r1, c0, c1 in (rng, wing):
        if best:                           # split along columns: rows and columns swap back
            r0, r1, c0, c1 = c0, c1, r0, r1
        ring = [(c0 - 0.5, r0 - 0.5), (c1 + 0.5, r0 - 0.5), (c1 + 0.5, r1 + 0.5), (c0 - 0.5, r1 + 0.5)]
        corners.append([[x + x0, y + y0] for x, y in ring])
    rng_px, wing_px = corners
    area = lambda c: Polygon(grounds_world(np.asarray(c, float), geo)).area
    print(f"{int(mask.sum())} white px, {misfit} misfit; range {area(rng_px):.0f} m^2, wing {area(wing_px):.0f} m^2")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"plan_px": {"range": rng_px, "wing": wing_px}}), encoding="utf-8")

    crop = Image.fromarray(win.astype(np.uint8)).resize((2 * R * S, 2 * R * S), Image.NEAREST)
    d = ImageDraw.Draw(crop)
    for ring in corners:
        d.line([((x - x0) * S, (y - y0) * S) for x, y in ring + ring[:1]], fill=(255, 0, 255), width=2)
    preview = BUILD / "review" / "court-glasshouse-trace.png"
    preview.parent.mkdir(parents=True, exist_ok=True)
    crop.save(preview)
    print("wrote", OUT.relative_to(ROOT), "and", preview.relative_to(ROOT))


if __name__ == "__main__":
    trace()
