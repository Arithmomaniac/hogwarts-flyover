"""Village roofs (build time): one interface over two backends.

roof(outline, angles) returns a closed roof solid over a plan outline: each edge rises inward at its own angle from
the horizontal (90 makes a gable), the slopes meeting in hips, valleys and ridges as a straight skeleton places them,
with courtyards kept. Coordinates are plan X, Y and height Z above the outline.

- "cgal" (the default, chosen by the user at gate G1): CGAL's straight-skeleton extrusion through compas_cgal's
  binding, its polygon faces triangulated by manifold3d in their own planes. compas_cgal's public extrude()
  triangulates by ear clipping, which fails on a few percent of real outlines. Where CGAL's skeleton folds a roof over
  itself (a gable beside an inside corner), roof() builds that roof with the baseline instead.
- "inrepo": the tested fallback, needing no new dependency: it builds every roof the village asks for. A rectangle
  with one pitch it roofs exactly, with a gable or a hip at each end; other rectilinear outlines with one pitch as the
  slabs of their wall grid, each gabled along its own long side and unioned with manifold3d; anything else with each
  edge rising at its own angle to a flat top (_flat_topped).
"""
import math

import numpy as np
from shapely.geometry import box
from shapely.geometry.polygon import orient

BACKENDS = ("cgal", "inrepo")
BACKEND = "cgal"


def roof(outline, angles, backend=None, cap=None):
    """(vertices (n, 3), triangles (m, 3)) of the closed roof solid over a counter-clockwise plan outline. `angles` is
    one pitch in degrees for every edge, or one per edge of the exterior and then of each hole, edge k running from
    point k to point k + 1. Where gables leave the slopes unable to meet (a lean-to between party walls), the roof is
    capped flat at `cap`, by default (cap_height) the height its slopes reach over the outline's widest inscribed
    circle."""
    outline = orient(outline, 1.0)
    rings = [np.asarray(outline.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in outline.interiors]
    sizes = [len(r) for r in rings]
    if np.ndim(angles) == 0:
        per_ring = [[float(angles)] * n for n in sizes]
    else:
        flat = [float(a) for a in angles]
        if len(flat) != sum(sizes):
            raise ValueError(f"{len(flat)} angles for {sum(sizes)} edges")
        per_ring = [flat[sum(sizes[:k]): sum(sizes[:k + 1])] for k in range(len(sizes))]
    run = {"cgal": _cgal, "inrepo": _inrepo}[backend or BACKEND]
    cap = cap_height(outline, angles) if cap is None else cap
    v, f = run(rings, per_ring, cap)
    if run is _cgal and not _stands(v, f, outline.area):
        v, f = _inrepo(rings, per_ring, cap)
    return v, f


def _stands(v, f, area):
    """Whether a closed roof solid stands on its plan once: nothing but its base faces down."""
    t = v[f]
    a = (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])
    return -a[a < 0].sum() / 2 <= area * (1 + 1e-6)


def cap_height(outline, angles):
    """The height at which roof() caps a roof over `outline` whose gables leave its slopes unable to meet: the height
    its steepest slope reaches over the outline's widest inscribed circle."""
    import shapely

    sloped = [a for a in np.atleast_1d(angles) if a < 89.999]
    if not sloped:
        raise ValueError("a roof needs at least one sloping edge")
    radius = shapely.maximum_inscribed_circle(orient(outline, 1.0), 0.01).length
    return radius * math.tan(math.radians(max(sloped))) + 1e-3


def _cgal(rings, per_ring, cap):
    import manifold3d as m3
    from compas_cgal import _straight_skeleton_2 as sk

    as3 = lambda r: np.ascontiguousarray(np.column_stack([r, np.zeros(len(r))]), np.float64)
    # CGAL's first angle belongs to the edge from the last point to the first
    speeds = [[a[-1]] + a[:-1] for a in per_ring]
    gabled = any(a >= 89.999 for ring in per_ring for a in ring)
    v, faces = sk.extrude_straight_skeleton(as3(rings[0]), [as3(r) for r in rings[1:]], speeds, True,
                                            cap if gabled else -1.0)
    v = np.asarray(v, np.float64)
    tris = []
    for face in faces:
        if len(face) == 3:
            tris.append(list(face))
            continue
        p = v[face]
        n = np.zeros(3)
        for i in range(len(p)):                      # Newell's normal
            a, b = p[i], p[(i + 1) % len(p)]
            n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
        n /= np.linalg.norm(n)
        u = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        w = np.cross(n, u)
        for t in np.asarray(m3.triangulate([np.column_stack([p @ u, p @ w])])):
            tris.append([face[k] for k in t])
    return v, np.asarray(tris, np.int64)


def _inrepo(rings, per_ring, cap):
    import manifold3d as m3

    rectilinear = all(np.abs(r - np.roll(r, 1, axis=0)).min(axis=1).max() <= 1e-6 for r in rings)
    pitch = [a for ring in per_ring for a in ring if a < 89.999]
    if not pitch:
        raise ValueError("a roof needs at least one sloping edge")
    if not rectilinear or max(pitch) - min(pitch) > 1e-6:
        return _flat_topped(rings, per_ring)
    tan = math.tan(math.radians(pitch[0]))
    pts = rings[0]
    if len(rings) == 1 and len(pts) == 4:
        return _rect(pts, per_ring[0], tan)
    from shapely.geometry import Polygon

    poly = Polygon(rings[0], rings[1:])
    xs, ys = np.unique(np.round(np.vstack(rings)[:, 0], 6)), np.unique(np.round(np.vstack(rings)[:, 1], 6))
    columns = []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        col = [poly.contains(box(x0, y0, x1, y1).centroid) for y0, y1 in zip(ys[:-1], ys[1:])]
        runs, j = [], 0
        while j < len(col):
            if col[j]:
                k = j
                while k + 1 < len(col) and col[k + 1]:
                    k += 1
                runs.append((ys[j], ys[k + 1]))
                j = k + 1
            else:
                j += 1
        columns.append((x0, x1, tuple(runs)))
    solids, i = [], 0
    while i < len(columns):
        k = i
        while k + 1 < len(columns) and columns[k + 1][2] == columns[i][2]:
            k += 1
        solids += [(columns[i][0], columns[k][1], y0, y1) for y0, y1 in columns[i][2]]
        i = k + 1
    out = m3.Manifold.batch_boolean([_gabled(*s, tan) for s in solids], m3.OpType.Add)
    mesh = out.to_mesh()
    return np.asarray(mesh.vert_properties, np.float64)[:, :3], np.asarray(mesh.tri_verts, np.int64)


def _inset(rings, per_ring, h):
    """Each ring with every edge moved in by the run of its slope up to height h (a gable's not at all), or None once
    the rings would no longer bound the same polygon: an edge vanished or turned round, a ring crossed."""
    from shapely.geometry import Polygon

    out = []
    for ring, angles in zip(rings, per_ring):
        lines = []
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            d = b - a
            n = np.array([d[1], -d[0]]) / np.linalg.norm(d)            # out of the material, for either ring
            run = 0.0 if angles[k] >= 89.999 else h / math.tan(math.radians(angles[k]))
            lines.append((a - n * run, d))
        pts = []
        for k in range(len(lines)):
            (p1, d1), (p2, d2) = lines[k - 1], lines[k]
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-12:
                pts.append(p2)
            else:
                t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
                pts.append(p1 + d1 * t)
        pts = np.asarray(pts)
        if np.any(np.einsum("ij,ij->i", np.roll(pts, -1, axis=0) - pts, np.roll(ring, -1, axis=0) - ring) <= 1e-9):
            return None
        out.append(pts)
    poly = Polygon(out[0], out[1:])
    return out if poly.is_valid and poly.area > 1e-9 else None


def _flat_topped(rings, per_ring):
    """A roof over any outline, needing no straight skeleton: each edge rises inward at its own angle (a gable's
    straight up) to the height at which the next edge would vanish, where it is capped flat. Where the skeleton would
    carry a ridge on, this roof has a flat top."""
    import manifold3d as m3

    lo, hi = 0.0, 1.0
    while _inset(rings, per_ring, hi) is not None:
        lo, hi = hi, hi * 2.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _inset(rings, per_ring, mid) is not None else (lo, mid)
    h = lo
    if h < 1e-3:
        raise ValueError("the outline is too narrow to roof")
    top = _inset(rings, per_ring, h)
    sizes = [len(r) for r in rings]
    base = np.cumsum([0] + sizes[:-1])
    n = sum(sizes)
    v = np.vstack([np.column_stack([np.vstack(rings), np.zeros(n)]), np.column_stack([np.vstack(top), np.full(n, h)])])
    faces = []
    for r, size in zip(base, sizes):
        for k in range(size):
            a, b = r + k, r + (k + 1) % size
            faces += [(a, b, n + b), (a, n + b, n + a)]
    for t in np.asarray(m3.triangulate([np.asarray(r) for r in top])):
        faces.append((n + t[0], n + t[1], n + t[2]))
    for t in np.asarray(m3.triangulate([np.asarray(r) for r in rings])):
        faces.append((t[0], t[2], t[1]))
    return v, np.asarray(faces, np.int64)


def _gabled(x0, x1, y0, y1, tan):
    """A slab's roof solid, gabled along its longer side."""
    import manifold3d as m3

    w, d = x1 - x0, y1 - y0
    if w >= d:                  # ridge along X: a profile (across, up) extruded along X
        prof = m3.CrossSection([[(0.0, 0.0), (d, 0.0), (d / 2, d / 2 * tan)]])
        return m3.Manifold.extrude(prof, w).transform(np.array([[0, 0, 1, x0], [1, 0, 0, y0], [0, 1, 0, 0]], np.float64))
    prof = m3.CrossSection([[(0.0, 0.0), (w, 0.0), (w / 2, w / 2 * tan)]])
    return m3.Manifold.extrude(prof, d).transform(np.array([[1, 0, 0, x0], [0, 0, -1, y1], [0, 1, 0, 0]], np.float64))


def _rect(pts, angles, tan):
    """A rectangle's roof: its ridge along the longer side, each end a gable (90) or a hip."""
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, d = x1 - x0, y1 - y0
    along_x = w >= d
    half = (d if along_x else w) / 2
    h = half * tan
    ends = []
    for a, b, ang in zip(pts, np.roll(pts, -1, axis=0), angles):
        across = abs(b[0] - a[0]) < 1e-9 if along_x else abs(b[1] - a[1]) < 1e-9
        if across:
            ends.append(((a[0] + b[0]) / 2 if along_x else (a[1] + b[1]) / 2, ang >= 89.999))
    lo, hi = sorted(ends)
    r0 = (lo[0] if lo[1] else lo[0] + half)
    r1 = (hi[0] if hi[1] else hi[0] - half)
    if along_x:
        verts = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0), (r0, (y0 + y1) / 2, h), (r1, (y0 + y1) / 2, h)]
    else:
        verts = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0), ((x0 + x1) / 2, r0, h), ((x0 + x1) / 2, r1, h)]
    v = np.array(verts, np.float64)
    if along_x:
        faces = [(0, 1, 5), (0, 5, 4), (2, 3, 4), (2, 4, 5), (1, 2, 5), (3, 0, 4), (0, 3, 2), (0, 2, 1)]
    else:
        faces = [(1, 2, 5), (1, 5, 4), (3, 0, 4), (3, 4, 5), (2, 3, 5), (0, 1, 4), (0, 3, 2), (0, 2, 1)]
    f = np.array(faces, np.int64)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2, h / 3])
    flip = np.einsum("ij,ij->i", n, v[f].mean(axis=1) - centre) < 0
    f[flip] = f[flip][:, [0, 2, 1]]
    area = np.linalg.norm(n, axis=1)
    return v, f[area > 1e-12]
