"""Grade the railway's earthworks into build/data.js (run after pack.py and fix_berth.py; idempotent).

The essay map's railway does not fit the terrain: the station's levelled corridor sat in a trench, its junction 36 m up
the hillside 68 m away and the foot of the loop 25 m below it. The user chose to regrade the line (railway-regrade):
this clamps the local bed between railway.envelope's cut and fill surfaces, so the running lines lie on their level or
gently climbing formation, cuttings and embankments meet the ground at their side slopes, the station terrace is level
and nothing outside the earthworks changes. Trees on the earthworks are cleared, but for the pine bank the first-years'
path down to the lake cuts through, where the works only raise the ground (PS 6: "so dark either side of them that Harry
thought there must be thick trees there"); so is every tree within LINE_CLEAR_M of the open line, whatever ground it
stands on, so that no crown reaches the train. The running lines' formation takes the earth surface class and forest
floor left on a cut face, an embankment or the cleared lineside becomes scrub or meadow. Hollows that an embankment
encloses, such as a dry valley it crosses, are made up to where they would spill, with spoil from the cuttings, as
meadow. The design depends only on the manifest and meta, so a second run changes nothing. The earthworks may not reach
water.

Usage: python -X utf8 src/fix_railway.py [data.js]
"""
import json
import sys
from pathlib import Path

import numpy as np

import common
import railway
from mesh_source import CHANNELS, DATA_JS, Grid, decode, encode_height, encode_u8, load_georef
from mesh_transport import COACH_W

MANIFEST = Path(__file__).resolve().parent / "mesh_scene.json"
TOL_M = 0.05
DRY = -8.0                          # pack.py's water delta on land away from water
STREAM_NEAR = 160                   # stream channel bytes below this lie on a stream or its wet ring
NO_STREAM = 255
MEADOW, SCRUB, EARTH, FOREST = 0, 2, 3, 5
PATH_BANK_M = 8.0                   # the pines the first-years' path cuts down through stand this far either side of it
LINE_CLEAR_M = COACH_W / 2 + 1.0 + common.TREE_CROWN_M + common.TEXEL_REACH_M   # ... and no nearer a running line
                                    # than this, so no crown the renderer grows, its density read bilinearly a texel
                                    # diagonal into the cleared ground, comes within a metre of a coach's side


def window(grid, al):
    """(rows, columns) slices of the local grid that hold every point the earthworks can reach."""
    pts = np.concatenate([al.loop.xz, al.main.xz, al.path.xz, np.asarray(al.terrace.exterior.coords)])
    lo, hi = pts.min(axis=0) - railway.REACH_M, pts.max(axis=0) + railway.REACH_M
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
    rows, cols = grid.to_texel(corners[:, 0], corners[:, 1])
    return (slice(max(int(rows.min()), 0), min(int(np.ceil(rows.max())) + 1, grid.ny)),
            slice(max(int(cols.min()), 0), min(int(np.ceil(cols.max())) + 1, grid.nx)))


def surfaces(scene, meta, georef):
    al = railway.alignment(scene, meta, georef)
    grid = Grid(meta["local"])
    rows, cols = window(grid, al)
    x, z = grid.to_world(*np.mgrid[rows, cols])
    return (rows, cols), railway.envelope(al, x, z), bank(al, x, z), lineside(al, x, z)


def bank(al, x, z):
    """Ground beside the first-years' path down to the lake, clear of its own width and of the running lines, where its
    pines stand on any embankment the works raise."""
    from scipy.spatial import cKDTree

    q = np.column_stack([np.ravel(x), np.ravel(z)])
    path = cKDTree(al.path.xz).query(q)[0]
    line = cKDTree(np.concatenate([al.loop.xz, al.main.xz])).query(q)[0]
    out = (path > railway.PATH_HALF_M + 1.0) & (path < PATH_BANK_M) & (line > LINE_CLEAR_M)
    return out.reshape(np.shape(x))


def lineside(al, x, z):
    """Ground within LINE_CLEAR_M of the open line, the loop and the main line up to its tunnel, kept clear of trees."""
    from scipy.spatial import cKDTree

    lines = np.concatenate([al.loop.xz, al.main.xz[al.main.s <= al.portal_s]])
    near = cKDTree(lines).query(np.column_stack([np.ravel(x), np.ravel(z)]), distance_upper_bound=LINE_CLEAR_M)[0]
    return (near < LINE_CLEAR_M).reshape(np.shape(x))


def kept_pines(pines, cut, fill, onbed, held_down):
    """The pine bank's texels that keep their trees: on an embankment's slope, never cut down, on a formation or a level
    top (where cut and fill meet). The ground a run raises to the fill surface lies within half a height quantum of it,
    and a cut surface more than 2 * TOL_M above that surface stays above that ground by more than TOL_M, so a second run
    decides the same."""
    return pines & ~held_down & ~onbed & (cut > fill + 2 * TOL_M)


def footprint(scene, meta, georef, bed):
    """Texels whose ground the earthworks hold, on a formation or a side slope, and whose trees they clear."""
    (rows, cols), (cut, fill, onbed), pines, _ = surfaces(scene, meta, georef)
    out = np.zeros(bed.shape, bool)
    held_down, held_up = cut < bed[rows, cols] + TOL_M, fill > bed[rows, cols] - TOL_M
    out[rows, cols] = (held_down | held_up) & ~kept_pines(pines, cut, fill, onbed, held_down)
    return out


def enclosed(before, after, wet):
    """(hollow, spill level): ground that lies in a closed depression after the earthworks but did not before them,
    and the level each such hollow spills at; water drains whatever reaches it."""
    spill = common.fill(after, wet)
    return (spill - after > TOL_M) & ~(common.fill(before, wet) - before > TOL_M), spill


def graded(old, cut, fill, wet, quantum):
    """(ground, hollow): `old` clamped between the earthworks' cut and fill surfaces on the height quantum, with the
    hollows the earthworks enclose made up to the level they spill at."""
    new = np.round(railway.regrade(old, cut, fill) / quantum) * quantum
    hollow, spill = enclosed(old, new, wet)
    return np.where(hollow, np.round(spill / quantum) * quantum, new), hollow


def fix_railway(scene, meta, georef, bed, wd, trees, surface, stream, quantum=0.02):
    """(bed, water delta, trees, surface class, stream channel, texels changed) with the railway's earthworks graded
    in. Where an embankment covers a stream the stream runs through a culvert: its water and channel give way to the
    earthworks; anything else wet under them is an error. Hollows the earthworks enclose are made up to their spill
    level. The lineside is cleared of trees."""
    (rows, cols), (cut, fill, onbed), pines, clear = surfaces(scene, meta, georef)
    old = bed[rows, cols]
    new, hollow = graded(old, cut, fill, wd[rows, cols] > 0.0, quantum)
    held_down, held_up = cut < old + TOL_M, fill > old - TOL_M
    works = held_down | held_up
    wet = works & (wd[rows, cols] > DRY)
    if (wet & (stream[rows, cols] >= STREAM_NEAR)).any():
        raise ValueError("the railway's earthworks reach water that is not a stream")
    cls = surface[rows, cols].copy()
    cls[onbed] = EARTH
    flat = held_down & held_up
    bare = works & ~(kept_pines(pines, cut, fill, onbed, held_down) & ~hollow)
    cls[bare & ~onbed & (cls == FOREST)] = np.where(held_down & ~flat, SCRUB, MEADOW)[bare & ~onbed & (cls == FOREST)]
    cls[(hollow | clear) & ~works & (cls == FOREST)] = MEADOW
    tr = np.where(bare | hollow | clear, 0, trees[rows, cols]).astype(trees.dtype)
    st = np.where(works, np.maximum(stream[rows, cols], NO_STREAM), stream[rows, cols]).astype(stream.dtype)
    w = np.where(works, DRY, wd[rows, cols])
    moved = np.abs(new - old) > quantum / 2
    changed = moved | (tr != trees[rows, cols]) | (cls != surface[rows, cols]) | (st != stream[rows, cols]) | (w != wd[rows, cols])
    out = [a.copy() for a in (bed, wd, trees, surface, stream)]
    for a, v in zip(out, (np.where(moved, new, old), w, tr, cls, st)):
        a[rows, cols] = v
    return (*out, int(changed.sum()))


def main(path=DATA_JS):
    with open(path, encoding="ascii") as f:
        text = f.read()
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    write = lambda d: "const DATA=" + json.dumps(d, separators=(",", ":")) + ";\n"
    if write(data) != text:
        raise ValueError(f"{path} is not in pack.py's serialisation; refusing to rewrite it")
    tex, scene = data["tex"], json.loads(MANIFEST.read_text(encoding="utf-8"))
    la, lb, ln = decode(tex["locA"]), decode(tex["locB"]), decode(tex["locN"])
    trees_at, stream_at = CHANNELS["locA"].index("trees"), CHANNELS["locB"].index("stream")
    bed, wd, trees, surface, stream, changed = fix_railway(
        scene, data["meta"], load_georef(), decode(tex["locBed"]), decode(tex["locW"]), la[..., trees_at], ln[..., 0],
        lb[..., stream_at], tex["locBed"]["q"])
    if changed:
        la[..., trees_at], lb[..., stream_at], ln[..., 0] = trees, stream, surface
        tex["locBed"], tex["locW"] = encode_height(bed, tex["locBed"]["q"]), encode_height(wd, tex["locW"]["q"])
        tex["locA"], tex["locB"] = encode_u8(la, tex["locA"]["delta"]), encode_u8(lb, tex["locB"]["delta"])
        tex["locN"] = encode_u8(ln, tex["locN"]["delta"])
        with open(path, "w", encoding="ascii", newline="\n") as f:
            f.write(write(data))
    print(f"railway earthworks: {changed} texels changed in {path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
