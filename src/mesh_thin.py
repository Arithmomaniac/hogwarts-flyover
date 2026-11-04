"""Thin set pieces that keep a readable silhouette at any distance (build time).

A part narrower than an internal pixel is drawn only by the jittered frames whose samples happen to land on it, and the
temporal filter's neighbourhood clamp then fades it out: the goals' 0.18 m poles all but vanished from 250 m. The payload
gives each vertex of a listed part the radius of the round part it lies on, in its position's padding word, and the
page's mesh vertex shader widens the part along its normals to at least THIN_PX pixels of radius in the view drawing
it, so every frame covers it (template_b.html's meshVs; UNIT_M is its THIN_UNIT_M).

The radius comes from the surface's own normals: going round a smooth round part (a lathe's pole, a ring's tube) the
normals turn by the chord of its cross-section, so the radius is the position chord over the normal chord at its
tightest edge, exactly for a regular polygon. A flat face's normals do not turn and it gets none: widening flat faces
along their normals would part them at their edges, and a square bar seen face-on would not widen at all.

Thin casters break up the same way in the sun-shadow layers: a part narrower than a layer's texels covers their centres
only here and there along its length, so its shadow falls as a row of dots. Every long thin part of a chunk the layers
draw (widths: a stick, such as a pole, a bar or a stay, or a round tube, such as a hoop, outside a lattice of them)
leads its chunk's triangles, the narrowest first, and the payload lists their runs with their widths (narrow_first); each layer whose texels are
wider than a part draws its triangles' edges again as lines, which rasterize without a break (template_b.html's
narrowLines)."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError

from materials import code

# asset -> the materials of its thin round parts. The goals' poles and hoops (PS 8: "three golden poles with hoops on
# the end ... fifty feet high") are the pitch's only gold.
POLICY = {"pitch": ("gold",)}
UNIT_M = 1e-4                        # radii travel as u16 steps of 0.1 mm; 0 is a vertex that never widens
MAX_M = 0.5                          # a rounder surface of a listed material keeps its true size
CAST_M = 0.5                         # a narrow caster is at most this across (the coarsest layer's texels reach ~1.5 m)
LONG = 1.5                           # and at least this many times as long
ROUND_SHARE = 0.9                    # a round tube: this share of its vertices on normals turning within CAST_M across
CROWD_M = 0.2                        # a lattice: parallel parts this close, two near-layer texels (template_b.html NEAR_M)
PARALLEL = 0.95                      # parts whose axes' cosine is above this run parallel


def codes(asset):
    return [code(name) for name in POLICY.get(asset, ())]


def curvature(mesh):
    """Per vertex of `mesh` (metres): the tightest radius its normals turn at along an edge, the edge's chord over its
    normals' chord; inf where they never turn."""
    idx = mesh.idx.astype(np.int64)
    edges = np.concatenate([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]])
    turn = np.linalg.norm(mesh.nrm[edges[:, 0]] - mesh.nrm[edges[:, 1]], axis=1)
    chord = np.linalg.norm(mesh.pos[edges[:, 0]] - mesh.pos[edges[:, 1]], axis=1)
    edges, ratio = edges[turn > 1e-9], chord[turn > 1e-9] / turn[turn > 1e-9]
    tightest = np.full(len(mesh.pos), np.inf)
    np.minimum.at(tightest, edges[:, 0], ratio)
    np.minimum.at(tightest, edges[:, 1], ratio)
    return tightest


def radii(mesh, listed):
    """Per vertex of `mesh` (metres): the radius of the round part it lies on, where its (material, aux) is one of
    `listed` and that radius is at most MAX_M; 0 everywhere else."""
    out = np.zeros(len(mesh.pos))
    on = np.isin(mesh.attr[:, 0].astype(int) * 256 + mesh.attr[:, 1], [m * 256 + a for m, a in listed])
    if not on.any() or not len(mesh.idx):
        return out
    tightest = curvature(mesh)
    keep = on & (tightest <= MAX_M)
    out[keep] = tightest[keep]
    return out


def least_width(points):
    """The least width of 2D points: across each edge of their convex hull, the narrowest of its extents."""
    try:
        hull = points[ConvexHull(points).vertices]
    except QhullError:                                       # all on one line
        return 0.0
    edge = np.roll(hull, -1, axis=0) - hull
    across = np.column_stack([-edge[:, 1], edge[:, 0]]) / np.linalg.norm(edge, axis=1)[:, None]
    return float(np.ptp(hull @ across.T, axis=0).min())


def widths(mesh, weld):
    """Per triangle of `mesh`: the width (metres) of the narrow part it lies on, inf where it lies on none. A part is
    triangles joined through shared points (`weld` has a row per vertex, equal rows being one point, so a square bar's
    faces join though each has vertices of its own). It is narrow when it is long and thin: a stick, at most CAST_M
    across its principal axis and LONG times that along it, its width its least extent across; or a round tube (by
    its normals, as radii), its width twice its median radius and its length LONG times that. A part with a parallel
    one closer than CROWD_M alongside it is in a lattice (a gate's pickets) and is not narrow: lines a texel wide would
    merge the lattice's shadow into a solid one, where the layers' own coverage of its parts keeps it as dark as the
    lattice is on average."""
    out = np.full(len(mesh.idx), np.inf)
    if not len(mesh.idx):
        return out
    idx = mesh.idx.astype(np.int64)
    point = np.unique(weld, axis=0, return_inverse=True)[1].ravel()[idx]
    n = int(point.max()) + 1
    links = coo_matrix((np.ones(2 * len(idx)), (np.concatenate([point[:, 0], point[:, 1]]),
                                                np.concatenate([point[:, 1], point[:, 2]]))), shape=(n, n))
    part = connected_components(links, directed=False)[1][point[:, 0]]
    tightest = curvature(mesh)
    order = np.argsort(part, kind="stable")
    found = []                                               # each narrow part's (triangles, centre, axis, length)
    for tris in np.split(order, np.flatnonzero(np.diff(part[order])) + 1):
        vs = np.unique(idx[tris])
        if len(vs) < 3:
            continue
        q = mesh.pos[vs] - mesh.pos[vs].mean(axis=0)
        axes = np.linalg.svd(q, full_matrices=False)[2]
        extent = np.ptp(q @ axes.T, axis=0)
        width = least_width(q @ axes[1:].T) if extent[1:].max() <= CAST_M and extent[0] >= LONG * extent[1:].max() \
            else np.inf
        r = tightest[vs]
        round_ = (r > 0) & (r <= CAST_M / 2)
        if round_.mean() >= ROUND_SHARE and extent[0] >= LONG * 2 * np.median(r[round_]):
            width = min(width, 2 * float(np.median(r[round_])))
        out[tris] = width
        if np.isfinite(width):
            found.append((tris, mesh.pos[vs].mean(axis=0), axes[0], extent[0]))
    centre, axis = np.array([f[1] for f in found]).reshape(-1, 3), np.array([f[2] for f in found]).reshape(-1, 3)
    for k, (tris, c, a, length) in enumerate(found):
        along = (centre - c) @ a
        across = np.linalg.norm(centre - c - np.outer(along, a), axis=1)
        crowded = (np.abs(axis @ a) > PARALLEL) & (np.abs(along) < length / 2) & (across < CROWD_M)
        crowded[k] = False
        if crowded.any():
            out[tris] = np.inf
    return out


def narrow_first(width):
    """(triangle order, runs) from per-triangle widths (metres, inf where not narrow): the narrow triangles first, the
    narrowest first and each part's in its own order, then the rest in theirs; a run is [triangles, width in whole
    millimetres], for each width in turn."""
    mm = np.round(width * 1000)
    order = np.argsort(mm, kind="stable")
    values, counts = np.unique(mm[np.isfinite(mm)], return_counts=True)
    return order, [[int(c), int(v)] for v, c in zip(values, counts)]


def quantized(r):
    """Radii (metres) as the payload's u16 steps: a widening vertex never rounds to 0."""
    return np.where(r > 0, np.clip(np.round(r / UNIT_M), 1, 65535), 0).astype("<u2")
