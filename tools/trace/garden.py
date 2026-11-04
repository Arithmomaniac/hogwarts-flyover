"""Trace Hagrid's garden (4b) on White Hound's grounds map: the thin black outline west of the hut (4a).

Fills the non-ink region from a seed inside the outline, refusing a fill that reaches the window's edge (a leak through
a gap), then outlines the filled pixels and grows the outline half a pixel onto the ink line itself. The map comes from
the White Hound cache.

    python -X utf8 -m tools.trace.garden    # writes tools/manifest/data/grounds/garden-trace.json and build/review/garden-trace.png
"""
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from tools import whcache
from tools.paths import BUILD, ROOT
from mesh_source import load_georef
from railway import grounds_world

OUT = ROOT / "tools" / "manifest" / "data" / "grounds" / "garden-trace.json"
HUT = np.array([-291.55, -191.53])
R = 110                                    # half the search window, map pixels
S = 4                                      # preview magnification


def trace():
    img = np.asarray(Image.open(whcache.path("img/Hogwarts_map_unlabelled.gif")).convert("RGB")).astype(int)
    geo = load_georef()
    o, ex, ey = (grounds_world(np.array(p, float), geo) for p in ((0, 0), (1, 0), (0, 1)))
    a = np.column_stack([ex - o, ey - o])

    def to_px(w):
        return np.linalg.solve(a, np.asarray(w, float) - o)

    seed = to_px(HUT + [-30.0, -5.0])
    x0, y0 = int(seed[0] - R), int(seed[1] - R)
    win = img[y0:y0 + 2 * R, x0:x0 + 2 * R]
    ink = np.all(win == 0, axis=-1)
    for grow in (0, 1, 2):                 # close small gaps in the line only if the plain fill leaks
        wall = ndimage.binary_dilation(ink, iterations=grow) if grow else ink
        lab, _ = ndimage.label(~wall)
        region = lab == lab[int(seed[1]) - y0, int(seed[0]) - x0]
        if not (region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()):
            break
    else:
        raise SystemExit("the garden fill leaks through the window edge at every gap closing")
    if grow:
        region = ndimage.binary_dilation(region, iterations=grow) & ~ink
    ys, xs = np.nonzero(region)
    poly = unary_union([box(x, y, x + 1, y + 1) for x, y in zip(xs, ys)]).buffer(0.5, join_style=2).simplify(0.6)
    ring = np.asarray(poly.exterior.coords)[:-1] + [x0, y0]
    world = Polygon(grounds_world(ring, geo))
    print(f"closed with {grow} px gap closing; {len(ring)} vertices; area {world.area:.0f} m^2; "
          f"hut to garden centroid {np.round(np.asarray(world.centroid.coords[0]) - HUT, 1)} m")
    OUT.write_text(json.dumps({"plan_px": ring.round(1).tolist(), "gap_closing_px": grow}), encoding="utf-8")

    crop = Image.fromarray(win.astype(np.uint8)).resize((2 * R * S, 2 * R * S), Image.NEAREST)
    crop = Image.blend(crop, Image.new("RGB", crop.size, "white"), 0.4)
    d = ImageDraw.Draw(crop)
    d.line([((x - x0) * S, (y - y0) * S) for x, y in np.vstack([ring, ring[:1]])], fill=(255, 0, 255), width=3)
    hx, hy = (to_px(HUT) - [x0, y0]) * S
    d.ellipse([hx - 6, hy - 6, hx + 6, hy + 6], outline=(255, 0, 0), width=3)
    preview = BUILD / "review" / "garden-trace.png"
    preview.parent.mkdir(parents=True, exist_ok=True)
    crop.save(preview)
    print("wrote", OUT.relative_to(ROOT), "and", preview.relative_to(ROOT))


if __name__ == "__main__":
    trace()
