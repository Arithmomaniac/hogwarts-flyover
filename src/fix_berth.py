"""Open the Durmstrang ship's berth in build/data.js into plain bay water (run after pack.py; idempotent).

The ship's ink replaced lake colour on the essay map and left a ship-shaped island. pack.py's carve_ship floods a
12 m capsule round the ship's axis, but the ink spread wider: its rim stayed as islets carrying plants, inside a
shallow band that reads as a sandbar round the hull, and its surface classes show through the water as stripes.
Within BERTH_M of the axis this caps the bed at the bay's own level (the median wet bed BAY_RING out, which the fix
never touches), easing the cap up to the lake surface over BLEND_M, sinks the islets to the bay's bed and gives the
berth's water the bay's surface class. Every rule depends only on distance and the untouched ring, so a second run
changes nothing. Mainland shore texels and the deeper trench under the hull keep their values; no other layer
changes.

Usage: python -X utf8 src/fix_berth.py [data.js]
"""
import json
import sys

import numpy as np
from scipy import ndimage as ndi

from mesh_source import DATA_JS, Grid, decode, encode_height, encode_u8

BERTH_M, BLEND_M = 16.0, 4.0
ISLET_M = BERTH_M + BLEND_M          # dry patches wholly inside this are ink, not shore
BAY_RING = (ISLET_M, ISLET_M + 10.0)
HEIGHT_Q = 0.02                      # pack.py's locBed / locW quantum


def axis_distance(meta):
    """Each local texel's distance (m) from the ship's axis."""
    grid = Grid(meta["local"])
    e, n = grid.to_world(*np.mgrid[0:grid.ny, 0:grid.nx])
    a, b = (np.asarray(meta["extras"]["ship"][k], np.float64) for k in ("a", "b"))
    ab = b - a
    h = np.clip(((e - a[0]) * ab[0] + (n - a[1]) * ab[1]) / (ab @ ab), 0.0, 1.0)
    return np.hypot(e - a[0] - h * ab[0], n - a[1] - h * ab[1])


def bay_ring(d, wd):
    return (d >= BAY_RING[0]) & (d < BAY_RING[1]) & (wd > 0)


def bay_bed(d, bed, wd):
    """The bay's bed level round the berth: the median wet bed in BAY_RING."""
    return float(np.median(bed[bay_ring(d, wd)]))


def bay_surface(d, wd, surface):
    """The bay's surface class round the berth: the commonest class of the wet texels in BAY_RING."""
    return int(np.bincount(surface[bay_ring(d, wd)]).argmax())


def islets(d, wd):
    """Dry patches lying wholly within ISLET_M of the ship's axis."""
    labels, _ = ndi.label(wd <= 0, structure=np.ones((3, 3), bool))
    reach = ndi.maximum(d, labels, index=np.arange(labels.max() + 1))
    return (wd <= 0) & (reach[labels] < ISLET_M)


def fix_berth(meta, bed, wd, surface):
    """(bed, water depth, surface class, texels changed) with the berth opened to the bay: the bed may rise no higher
    than the bay's at BERTH_M, easing up to the lake surface over BLEND_M, the ink islets sink to the bay's bed, and
    the berth's water takes the bay's surface class."""
    d = axis_distance(meta)
    bay, islet, lake = bay_bed(d, bed, wd), islets(d, wd), float(meta["lake"])
    t = np.clip((d - BERTH_M) / BLEND_M, 0.0, 1.0)
    cap = np.round(np.where(islet, bay, bay + (lake - bay) * t * t * (3.0 - 2.0 * t)) / HEIGHT_Q) * HEIGHT_Q
    lowered = ((d < ISLET_M) & (wd > 0) & (cap < bed - HEIGHT_Q / 2)) | islet
    new_bed = np.where(lowered, np.minimum(cap, bed), bed)
    new_wd = np.where(lowered, lake - new_bed, wd)
    recoloured = (d < ISLET_M) & (new_wd > 0) & (surface != bay_surface(d, wd, surface))
    new_surface = np.where(recoloured, bay_surface(d, wd, surface), surface).astype(surface.dtype)
    return new_bed, new_wd, new_surface, int((lowered | recoloured).sum())


def main(path=DATA_JS):
    with open(path, encoding="ascii") as f:
        text = f.read()
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    write = lambda d: "const DATA=" + json.dumps(d, separators=(",", ":")) + ";\n"
    if write(data) != text:
        raise ValueError(f"{path} is not in pack.py's serialisation; refusing to rewrite it")
    tex = data["tex"]
    ln = decode(tex["locN"])
    bed, wd, surface, changed = fix_berth(data["meta"], decode(tex["locBed"]), decode(tex["locW"]), ln[..., 0])
    if changed:
        ln[..., 0] = surface
        tex["locBed"], tex["locW"] = encode_height(bed, tex["locBed"]["q"]), encode_height(wd, tex["locW"]["q"])
        tex["locN"] = encode_u8(ln, tex["locN"]["delta"])
        with open(path, "w", encoding="ascii", newline="\n") as f:
            f.write(write(data))
    print(f"ship berth: {changed} texels changed in {path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
