"""Cut the exterior mouth of White Hound's underground harbour into build/data.js (run after pack.py; idempotent).

"a wide, ivy-curtained opening which leads into a water-filled channel which penetrates the cliff-face" (harbour-mouth,
harbour-local-cutout). The heightfield cannot hold an opening under a cliff, so this step only cuts a tightly bounded,
water-filled recess into the cliff's foot; the grounds family's mesh roofs it over and closes its back, and there is no
navigable tunnel beyond.

The mouth is found from the data, so it follows the terrain wherever the scene is anchored: from White Hound's mapped
mouth (the manifest's traced harbour-mouth, map 1f), the nearest cliff foot (a dry texel within FOOT_M of the lake
that stands CLIFF_M or more above it) within SEARCH_M; the channel runs up the slope there, from the waterline into the
cliff. Inside the declared rectangle (WIDTH_M across, from APRON_M out in the lake to DEPTH_M into the cliff) the bed
is lowered to CHANNEL_M under the lake surface and becomes lake water of the surrounding lake's surface class; nothing
outside it changes. The mouth, the axis and the rectangle are recorded as meta.extras.harbour, and a second run
reuses that record, so it changes nothing.

Usage: python -X utf8 src/fix_harbour.py [data.js]
"""
import json
import sys
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage as ndi
from shapely.geometry import Polygon

from mesh_source import DATA_JS, Grid, decode, encode_height, encode_u8, load_georef, sample

MANIFEST = Path(__file__).with_name("mesh_scene.json")
SEARCH_M, FOOT_M, CLIFF_M = 60.0, 8.0, 15.0
WIDTH_M, DEPTH_M, APRON_M, CHANNEL_M = 8.0, 12.0, 2.0, 2.0
HEIGHT_Q = 0.02                       # pack.py's locBed / locW quantum


def mapped_mouth(manifest, georef):
    """World (x, z) of White Hound's mapped harbour mouth (the traced harbour-mouth, map 1f)."""
    import railway
    asset = next(a for a in manifest["assets"] if a["id"] == "harbour-mouth")
    px = np.asarray(asset["traced"][0]["plan_px"], np.float64)
    return railway.grounds_world(px, georef)[0]


def texel_world(grid):
    return grid.to_world(*np.mgrid[0:grid.ny, 0:grid.nx])


def find_mouth(meta, grid, bed, wd, mapped):
    """(waterline origin (x, z), unit (x, z) axis up into the cliff) of the cliff foot nearest the mapped mouth."""
    lake = float(meta["lake"])
    e, n = texel_world(grid)
    spacing = float(np.hypot(*np.subtract(grid.to_world(0, 1), grid.to_world(0, 0))))
    wet = wd > 0
    from_water = ndi.distance_transform_edt(~wet) * spacing
    reach = np.hypot(e - mapped[0], n - mapped[1])
    foot = (~wet) & (bed - lake >= CLIFF_M) & (from_water <= FOOT_M) & (reach <= SEARCH_M)
    if not foot.any():
        raise ValueError(f"no cliff foot within {SEARCH_M} m of the mapped harbour mouth")
    k = np.unravel_index(np.argmin(np.where(foot, reach, np.inf)), reach.shape)
    f = np.array([e[k], n[k]])
    g = lambda p: float(sample(bed, grid, np.array([p[0]]), np.array([p[1]]))[0])
    grad = np.array([g(f + (2.0, 0.0)) - g(f - (2.0, 0.0)), g(f + (0.0, 2.0)) - g(f - (0.0, 2.0))])
    u = grad / np.linalg.norm(grad) if np.linalg.norm(grad) > 1e-6 else (f - mapped) / np.linalg.norm(f - mapped)
    origin = f
    for step in np.arange(0.0, 30.0, 0.5):                 # down the slope to the waterline
        p = f - u * step
        if float(sample(wd, grid, np.array([p[0]]), np.array([p[1]]))[0]) > 0:
            origin = p
            break
    return origin, u


def recess(origin, u):
    """The declared bounds: WIDTH_M across the axis, from APRON_M out in the lake to DEPTH_M into the cliff."""
    side = np.array([-u[1], u[0]]) * WIDTH_M / 2
    a, b = origin - u * APRON_M, origin + u * DEPTH_M
    return Polygon([a - side, b - side, b + side, a + side])


def lake_class(grid, wd, surface, origin):
    """The surface class of the lake round the mouth: the commonest class of wet texels within 20 m of it."""
    e, n = texel_world(grid)
    ring = (np.hypot(e - origin[0], n - origin[1]) < 20.0) & (wd > 0)
    return int(np.bincount(surface[ring]).argmax())


def fix_harbour(meta, grid, bed, wd, surface, mapped, record=None):
    """(bed, water depth, surface class, texels changed, record) with the harbour mouth cut: inside the recorded or
    newly found bounds the bed is lowered to CHANNEL_M under the lake and becomes lake water."""
    lake = float(meta["lake"])
    if record is None:
        origin, u = find_mouth(meta, grid, bed, wd, mapped)
        bounds = recess(origin, u)
        record = {"mouth": [round(float(v), 3) for v in origin], "axis": [round(float(v), 6) for v in u],
                  "width": WIDTH_M, "depth": DEPTH_M, "apron": APRON_M,
                  "bounds": [[round(float(x), 3), round(float(z), 3)] for x, z in bounds.exterior.coords[:-1]]}
    bounds = Polygon(record["bounds"])
    e, n = texel_world(grid)
    inside = shapely.contains_xy(bounds, e, n)
    floor = np.round((lake - CHANNEL_M) / HEIGHT_Q) * HEIGHT_Q
    lowered = inside & (bed > floor + HEIGHT_Q / 2)
    new_bed = np.where(lowered, floor, bed)
    new_wd = np.where(inside, np.round((lake - new_bed) / HEIGHT_Q) * HEIGHT_Q, wd)
    water = lake_class(grid, wd, surface, np.asarray(record["mouth"]))
    recoloured = inside & (surface != water)
    new_surface = np.where(recoloured, water, surface).astype(surface.dtype)
    changed = lowered | recoloured | (inside & (np.abs(new_wd - wd) > HEIGHT_Q / 2))
    return new_bed, new_wd, new_surface, int(changed.sum()), record


def main(path=DATA_JS):
    with open(path, encoding="ascii") as f:
        text = f.read()
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    write = lambda d: "const DATA=" + json.dumps(d, separators=(",", ":")) + ";\n"
    if write(data) != text:
        raise ValueError(f"{path} is not in pack.py's serialisation; refusing to rewrite it")
    meta, tex = data["meta"], data["tex"]
    mapped = mapped_mouth(json.loads(MANIFEST.read_text(encoding="utf-8")), load_georef())
    ln = decode(tex["locN"])
    bed, wd, surface, changed, record = fix_harbour(meta, Grid(meta["local"]), decode(tex["locBed"]),
                                                    decode(tex["locW"]), ln[..., 0], mapped,
                                                    meta["extras"].get("harbour"))
    if changed or meta["extras"].get("harbour") != record:
        meta["extras"]["harbour"] = record
        ln[..., 0] = surface
        tex["locBed"], tex["locW"] = encode_height(bed, tex["locBed"]["q"]), encode_height(wd, tex["locW"]["q"])
        tex["locN"] = encode_u8(ln, tex["locN"]["delta"])
        with open(path, "w", encoding="ascii", newline="\n") as f:
            f.write(write(data))
    print(f"harbour mouth: {changed} texels changed in {path}; mouth at {record['mouth']}, axis {record['axis']}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
