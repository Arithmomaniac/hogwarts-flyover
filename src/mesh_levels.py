"""Levels of detail (build time): coarser versions of each group of an asset's pieces, and the distance from which each
may stand in for the full level (lod0).

A group is what a view picks a level for. The village and the Willow build their own levels (mesh_village.levels,
mesh_willow.levels); every other family's chunks are simplified with meshoptimizer's simplifier (meshopt.simplify),
welded per (material, aux) with their open borders locked, so abutting pieces never part, and their normals rebuilt
within CREASE_DEG. Thin parts (mesh_thin) and glazing are never simplified.

A level's switch distance is the nearest of DISTANCES from which, and from every one beyond, it passes two guards
against lod0, on cells of one output pixel of the reference view (REF_ROWS rows, FOV_DEG):
- its silhouette seen from the east, the north and above differs from lod0's by at most SHIFT_PX of mean boundary
  displacement (the cells they differ on over the cells along their outline), or XOR_CELLS cells;
- its surface lies within DEVIATION_PX of lod0's: the simplifier's own error for the levels it makes; for the lanes'
  levels, the distance within which each surface lies of the other at all but DEVIATION_TAIL of their area (sampled
  every SAMPLE_M, or more sparsely: FIRST_SAMPLES points, then up to MAX_SAMPLES where the spacing passes PRECISION of
  the answer). Flattened openings and dropped details count though they leave the silhouette alone.
The page scales the distance with its own output resolution."""
import hashlib
import math

import numpy as np
from scipy.spatial import cKDTree

import meshopt
from mesh_geometry import Mesh

REF_ROWS, FOV_DEG = 1080, 50.0
PIX_REF = 2 * math.tan(math.radians(FOV_DEG / 2)) / REF_ROWS     # radians per output pixel of the reference view
VIEWS = {"east": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), "north": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
         "above": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))}
SHIFT_PX, XOR_CELLS = 0.5, 2
DEVIATION_PX, DEVIATION_TAIL = 0.75, 0.005
SAMPLE_M, FIRST_SAMPLES, MAX_SAMPLES = 0.03, 100_000, 1_000_000
PRECISION = 0.25                                                 # sampling sparser than this share of the deviation is refined
DISTANCES = tuple(50.0 * 2 ** (k / 2) for k in range(17))       # 50 m to 12.8 km in steps of a square root of two
MAX_CELLS = 4_000_000                                            # a silhouette grid this large is too near to judge
SIMPLIFY_M = (0.15, 0.6)                                         # the simplifier's error budgets for lod1 and lod2
SAVING = 0.85                                                    # a level must keep at most this share of the one before
CREASE_DEG = 40.0
SMALL_REACH_M = 0.5                                              # a small structure's coarser piece centres within its box
LEVEL_NAMES = ("lod0", "lod1", "lod2", "shadow")


def cover(tris, cell, lo, size):
    """The cells of a `size` grid of `cell`-metre cells from `lo` whose centres the 2D triangles (t, 3, 2) cover, as a
    boolean grid: each triangle filled row by row, the span of a row's centre line it covers."""
    t = (tris - lo) / cell
    first = np.maximum(np.ceil(t[:, :, 0].min(axis=1) - 0.5), 0).astype(np.int64)
    last = np.minimum(np.floor(t[:, :, 0].max(axis=1) - 0.5).astype(np.int64), size[0] - 1)
    rows = np.maximum(last - first + 1, 0)
    k = np.repeat(np.arange(len(t)), rows)
    i = first[k] + np.arange(len(k)) - np.repeat(np.cumsum(rows) - rows, rows)
    x = i + 0.5
    y0, y1 = np.full(len(k), np.inf), np.full(len(k), -np.inf)
    for e in range(3):                      # where the row's centre line crosses each edge
        p, q = t[k, e], t[k, (e + 1) % 3]
        dx = q[:, 0] - p[:, 0]
        s = np.divide(x - p[:, 0], dx, out=np.full(len(k), -1.0), where=dx != 0)
        on = (s >= 0.0) & (s <= 1.0)
        y = p[:, 1] + np.clip(s, 0.0, 1.0) * (q[:, 1] - p[:, 1])
        y0, y1 = np.where(on, np.minimum(y0, y), y0), np.where(on, np.maximum(y1, y), y1)
    j0 = np.maximum(np.ceil(np.where(np.isfinite(y0), y0, 1.0) - 0.5), 0).astype(np.int64)
    j1 = np.minimum(np.floor(np.where(np.isfinite(y1), y1, 0.0) - 0.5).astype(np.int64), size[1] - 1)
    keep = np.isfinite(y0) & (j0 <= j1)
    width = size[1] + 1                     # a spare cell closes each row
    steps = np.bincount((i * width + j0)[keep], minlength=size[0] * width) \
        - np.bincount((i * width + j1 + 1)[keep], minlength=size[0] * width)
    return (np.cumsum(steps[:size[0] * width]) > 0).reshape(size[0], width)[:, :-1]


def silhouettes(meshes, axes, cell):
    """Each mesh's covered cells seen along the normal of the plane of `axes`, on one grid round all of them; None when
    that grid would exceed MAX_CELLS."""
    basis = np.array(axes, float).T
    uv = [m.pos[m.idx] @ basis for m in meshes]
    lo = np.min([p.reshape(-1, 2).min(axis=0) for p in uv], axis=0)
    hi = np.max([p.reshape(-1, 2).max(axis=0) for p in uv], axis=0)
    size = np.floor((hi - lo) / cell).astype(np.int64) + 1
    if size.prod() > MAX_CELLS:
        return None
    return [cover(p, cell, lo, size) for p in uv]


def alike(full, level, cell):
    """Whether `level`'s silhouette matches `full`'s on `cell`-metre cells from every view (None: too near to judge)."""
    for axes in VIEWS.values():
        grids = silhouettes([full, level], axes, cell)
        if grids is None:
            return None
        a, b = grids
        union = np.pad(a | b, 1)
        outline = union[1:-1, 1:-1] & ~(union[:-2, 1:-1] & union[2:, 1:-1] & union[1:-1, :-2] & union[1:-1, 2:])
        if (a ^ b).sum() > max(SHIFT_PX * outline.sum(), XOR_CELLS):
            return False
    return True


def samples(m, spacing, rng, budget):
    """Points on `m`'s triangles, about one per `spacing` squared of area (sparser if that would pass `budget`), and
    the spacing used."""
    tri = m.pos[m.idx]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    spacing = max(spacing, math.sqrt(area.sum() / budget))
    n = np.maximum(np.ceil(area / spacing ** 2), 1).astype(np.int64)
    k = np.repeat(np.arange(len(tri)), n)
    u, v = rng.random(len(k)), rng.random(len(k))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return tri[k, 0] + u[:, None] * (tri[k, 1] - tri[k, 0]) + v[:, None] * (tri[k, 2] - tri[k, 0]), spacing


def deviation(full, level):
    """The distance (metres) within which each surface lies of the other at all but DEVIATION_TAIL of its samples,
    sample to nearest sample: over by under a sample spacing, so the switch comes later, never earlier."""
    for budget in (FIRST_SAMPLES, MAX_SAMPLES):          # sparse first; denser only where it would blur the answer
        rng = np.random.default_rng(0)
        (a, sa), (b, sb) = samples(full, SAMPLE_M, rng, budget), samples(level, SAMPLE_M, rng, budget)
        near = np.concatenate([cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]])
        found = float(np.quantile(near, 1.0 - DEVIATION_TAIL))
        if max(sa, sb) <= PRECISION * found:
            break
    return found


def switch_distance(full, level, moved):
    """The nearest of DISTANCES from which, and from every one beyond, `level` may stand in for `full`, its surface
    having moved up to `moved` metres; None if none."""
    if not len(level.idx):
        return None
    found = None
    for d in reversed(DISTANCES):
        if moved > DEVIATION_PX * PIX_REF * d or alike(full, level, d * PIX_REF) is not True:
            break
        found = d
    return found


def smooth_normals(pos, tris):
    """Per corner of `tris` (indices into `pos`): the area-weighted mean of the normals of the faces round its vertex
    that turn from its own face by at most CREASE_DEG."""
    a, b, c = (pos[tris[:, k]] for k in range(3))
    face = np.cross(b - a, c - a)
    unit = face / np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-30)
    corners = tris.ravel()
    order = np.argsort(corners, kind="stable")
    owner = order // 3                                              # the face of each corner, by vertex
    first = np.searchsorted(corners[order], np.arange(len(pos)))
    count = np.bincount(corners, minlength=len(pos))
    out = np.zeros((len(corners), 3))
    mine = unit[np.arange(len(corners)) // 3]
    v = corners
    for s in range(int(count.max()) if len(count) else 0):
        has = count[v] > s
        g = owner[np.minimum(first[v] + s, len(order) - 1)]
        near = has & (np.einsum("ij,ij->i", unit[g], mine) >= math.cos(math.radians(CREASE_DEG)))
        out[near] += face[g[near]]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-30)


def simplified(m, error, keep):
    """`m` simplified within `error` metres, but for its triangles with any vertex in `keep` (copied unchanged), as an
    indexed mesh whose normals are rebuilt within CREASE_DEG, and the simplifier's error: how far it may have moved
    the surface (metres)."""
    held = keep[m.idx].any(axis=1)
    key = np.concatenate([m.pos, m.attr.astype(np.float64)], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    tris, reached = meshopt.simplify(inv[m.idx[~held]], uniq[:, :3], 0, error, ("LockBorder", "ErrorAbsolute"))
    tris = tris.astype(np.int64)
    if not len(tris):
        return (Mesh(m.pos[:0], m.nrm[:0], m.attr[:0], m.idx[:0]) if not held.any() else _subset(m, held)), reached
    normal = smooth_normals(uniq[:, :3], tris)
    corner = np.concatenate([tris.reshape(-1, 1), np.round(normal * 1e4).astype(np.int64)], axis=1)
    cu, ci = np.unique(corner, axis=0, return_inverse=True)
    ci = ci.ravel()
    _, first = np.unique(ci, return_index=True)          # each vertex's first corner, in the triangles' own order
    order = np.argsort(first, kind="stable")             # vertices as the triangles first use them: the codecs' locality
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    v = cu[order, 0]
    out = Mesh(uniq[v, :3], normal[first[order]], uniq[v, 3:].astype(np.uint8), rank[ci].reshape(-1, 3).astype(np.uint32))
    return (out if not held.any() else _concat(out, _subset(m, held))), reached


def _subset(m, rows):
    used, idx = np.unique(m.idx[rows].ravel(), return_inverse=True)
    return Mesh(m.pos[used], m.nrm[used], m.attr[used], idx.reshape(-1, 3).astype(np.uint32))


def _concat(a, b):
    return Mesh(np.concatenate([a.pos, b.pos]), np.concatenate([a.nrm, b.nrm]), np.concatenate([a.attr, b.attr]),
                np.concatenate([a.idx, b.idx + len(a.pos)]).astype(np.uint32))


def family_units(family, s, meshes):
    """{asset: [unit]} of a family that builds its own levels of detail (the village, the Willow), each unit a
    {level: [pieces]} (level 3: the shadow caster), whose full levels are together the asset's pieces in `meshes`: a
    piece the family's levels leave out is a unit of its own with its full level alone, which stands in at every other
    level (build_meshes.unit_groups); {} for every other family."""
    if family not in OWN_LEVELS:
        return {}
    found = OWN_LEVELS[family](s)
    key = lambda m: hashlib.sha256(m.pos.tobytes() + m.nrm.tobytes() + m.attr.tobytes() + m.idx.tobytes()).hexdigest()
    for asset, units in found.items():
        left = {}
        for m in meshes.get(asset, []):
            left.setdefault(key(m), []).append(m)
        for m in (m for u in units for m in u.get(0, [])):
            if not left.get(key(m)):
                raise ValueError(f"{asset}: its family's full level of detail holds a piece the payload does not")
            left[key(m)].pop()
        # pieces the family's levels leave out (a footprint it does not build gets the generic massing): full level only
        units += [{0: [m]} for ms in left.values() for m in ms]
    return found


def village_units(s):
    """Each built component's pieces per asset at every level (mesh_village.LEVELS) as one unit, and each small
    structure as one (small_units)."""
    import mesh_village as mv  # noqa: PLC0415 (a family module, loaded by name like the others)
    v = mv._village(s)                               # the village build_footprint built: every level reads its records
    levels = [(k, mv.LEVELS[name]) for k, name in enumerate(LEVEL_NAMES)]
    out = {}
    for cid in sorted(v.built):
        per = {k: v.component(cid, level) for k, level in levels}
        for asset in per[0]:
            out.setdefault(asset, []).append({k: per[k].get(asset, []) for k, _ in levels})
    for asset, units in small_units({k: v.small(level) for k, level in levels}).items():
        out.setdefault(asset, []).extend(units)
    return out


def small_units(small):
    """{asset: [unit]} of the village's small structures, {level: {asset: [pieces]}} (mesh_village.Village.small): each
    full piece (a lamp, a headstone, a length of the kirkyard's wall, a garden's wall) one unit, with every coarser
    piece whose box centre is nearest its own and within its box; a level that keeps no piece of it drops it."""
    out = {}
    for asset, full in small[0].items():
        lo, hi = np.array([m.pos.min(axis=0) for m in full]), np.array([m.pos.max(axis=0) for m in full])
        units = [{**{k: [] for k in small}, 0: [m]} for m in full]
        for k in sorted(small)[1:]:
            for m in small[k].get(asset, []):
                c = (m.pos.min(axis=0) + m.pos.max(axis=0)) / 2
                i = int(np.argmin(np.linalg.norm((lo + hi) / 2 - c, axis=1)))
                if np.any(c < lo[i] - SMALL_REACH_M) or np.any(c > hi[i] + SMALL_REACH_M):
                    raise ValueError(f"{asset}: a {LEVEL_NAMES[k]} piece stands within no full piece")
                units[i][k].append(m)
        out[asset] = units
    return out


def willow_units(s):
    """The Willow at every level (mesh_willow.levels), in the frame build_extras sets it in, as one unit."""
    import mesh_willow as mw  # noqa: PLC0415
    x0, z0 = s.meta["extras"]["willow"]
    y0 = float(s.ground(np.array([x0]), np.array([z0]))[0])
    ground = lambda x, z: s.ground(np.asarray(x) + x0, np.asarray(z) + z0) - y0
    shift = np.array([x0, y0, z0])
    found = mw.levels(ground, mw.gap_bearing(s.meta))
    return {"willow": [{k: [m._replace(pos=m.pos + shift) for m in found[name]] for k, name in enumerate(LEVEL_NAMES)}]}


OWN_LEVELS = {"village": village_units, "willow": willow_units}     # the families that build their own levels of detail
