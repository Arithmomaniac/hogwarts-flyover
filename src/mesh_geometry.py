"""Mesh primitives shared by the asset families (build time): prisms, surfaces of revolution, heightfield roofs and
battlements, in world metres (x east, y up, z north) with outward normals."""
import math
from typing import NamedTuple

import numpy as np
import shapely
from shapely.geometry import Polygon


class Mesh(NamedTuple):
    pos: np.ndarray    # (n, 3) float64 world metres: x east, y up, z north
    nrm: np.ndarray    # (n, 3) float64 unit normals
    attr: np.ndarray   # (n, 2) uint8 G-buffer material id and aux code
    idx: np.ndarray    # (m, 3) uint32 triangles; cross(b - a, c - a) points along the face normal


def concat(meshes):
    meshes = [m for m in meshes if len(m.pos)]
    offsets = np.cumsum([0] + [len(m.pos) for m in meshes[:-1]])
    return Mesh(np.concatenate([m.pos for m in meshes]), np.concatenate([m.nrm for m in meshes]),
                np.concatenate([m.attr for m in meshes]),
                np.concatenate([m.idx + o for m, o in zip(meshes, offsets)]).astype(np.uint32))


def heights(h, x, z):
    return np.broadcast_to(np.asarray(h(x, z) if callable(h) else h, np.float64), np.shape(x)).astype(np.float64)


def faceted(tris, attr, up=False):
    """Triangles (t, 3, 3) as flat-shaded geometry; `up` turns every face to point upwards."""
    tris = np.asarray(tris, np.float64)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    if up:
        flip = n[:, 1] < 0
        tris[flip] = tris[flip][:, [0, 2, 1]]
        n[flip] = -n[flip]
    length = np.linalg.norm(n, axis=1, keepdims=True)
    keep = length[:, 0] > 1e-12
    tris, n = tris[keep], n[keep] / length[keep]
    return Mesh(tris.reshape(-1, 3), np.repeat(n, 3, axis=0), np.tile(np.asarray(attr, np.uint8), (3 * len(tris), 1)),
                np.arange(3 * len(tris), dtype=np.uint32).reshape(-1, 3))


def lid(polygon, height, attr):
    """The footprint as an upward-facing surface at `height` (metres, or a callable of x, z)."""
    import shapely

    tris = np.array([np.asarray(t.exterior.coords)[:3] for t in shapely.constrained_delaunay_triangles(polygon).geoms])
    xz, tri = np.unique(tris.reshape(-1, 2), axis=0, return_inverse=True)
    tri = tri.reshape(-1, 3)
    u, v = xz[tri[:, 1]] - xz[tri[:, 0]], xz[tri[:, 2]] - xz[tri[:, 0]]
    flip = u[:, 1] * v[:, 0] - u[:, 0] * v[:, 1] < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    pos = np.column_stack([xz[:, 0], heights(height, xz[:, 0], xz[:, 1]), xz[:, 1]])
    return Mesh(pos, np.tile([0.0, 1.0, 0.0], (len(xz), 1)), np.tile(np.asarray(attr, np.uint8), (len(pos), 1)),
                tri.astype(np.uint32))


def prism(polygon, base, top, attr, cap=True):
    """A footprint extruded from `base` to `top` (metres, or callables of x, z): outward walls and a top cap."""
    from shapely.geometry.polygon import orient

    polygon = orient(polygon, 1.0)                  # walls face out of counter-clockwise rings (shapely set ops return clockwise)
    pos, nrm, idx, n = [], [], [], 0
    for ring in (polygon.exterior, *polygon.interiors):
        a = np.asarray(ring.coords)[:-1]
        b = np.roll(a, -1, axis=0)
        d = b - a
        length = np.hypot(d[:, 0], d[:, 1])
        a, b, d, length = a[length > 1e-9], b[length > 1e-9], d[length > 1e-9], length[length > 1e-9]
        out = np.column_stack([d[:, 1], np.zeros(len(d)), -d[:, 0]]) / length[:, None]
        corners = [(a, base), (b, base), (b, top), (a, top)]
        quad = np.stack([np.column_stack([p[:, 0], heights(h, p[:, 0], p[:, 1]), p[:, 1]]) for p, h in corners], 1)
        pos.append(quad.reshape(-1, 3))
        nrm.append(np.repeat(out, 4, axis=0))
        idx.append((n + 4 * np.arange(len(quad))[:, None] + np.array([0, 3, 1, 1, 3, 2])).reshape(-1, 3))
        n += 4 * len(quad)
    if cap:
        top_face = lid(polygon, top, attr)
        pos.append(top_face.pos)
        nrm.append(top_face.nrm)
        idx.append(top_face.idx.astype(np.int64) + n)
    pos = np.concatenate(pos)
    return Mesh(pos, np.concatenate(nrm), np.tile(np.asarray(attr, np.uint8), (len(pos), 1)),
                np.concatenate(idx).astype(np.uint32))


def circle(x, z, r, segments=32):
    from shapely.geometry import Polygon

    a = 2 * np.pi * np.arange(segments) / segments
    return Polygon(np.column_stack([x + r * np.cos(a), z + r * np.sin(a)]))


def cylinder(x, z, r, base, top, attr, segments=24):
    return prism(circle(x, z, r, segments), base, top, attr)


def lathe(x, z, profile, attr, segments=32, cap=True):
    """Surface of revolution about the vertical through (x, z): `profile` is (radius, height) from bottom to top.
    Smooth around the axis, creased at every profile point; `cap` closes the top with a disc."""
    a = 2 * np.pi * np.arange(segments + 1) / segments
    ca, sa = np.cos(a), np.sin(a)
    meshes = []
    for (r0, y0), (r1, y1) in zip(profile[:-1], profile[1:]):
        nr, ny = y1 - y0, -(r1 - r0)
        length = math.hypot(nr, ny)
        if length < 1e-9:
            continue
        nr, ny = nr / length, ny / length
        ring = lambda r, y: np.column_stack([x + r * ca, np.full_like(ca, y), z + r * sa])
        pos = np.concatenate([ring(r0, y0), ring(r1, y1)])
        nrm = np.tile(np.column_stack([nr * ca, np.full_like(ca, ny), nr * sa]), (2, 1))
        k = np.arange(segments)
        idx = np.concatenate([np.column_stack([k, k + 1, k + segments + 2]), np.column_stack([k, k + segments + 2, k + segments + 1])])
        face = np.cross(pos[idx[:, 1]] - pos[idx[:, 0]], pos[idx[:, 2]] - pos[idx[:, 0]])
        flip = np.einsum("ij,ij->i", face, nrm[idx[:, 0]]) < 0
        idx[flip] = idx[flip][:, [0, 2, 1]]
        meshes.append(Mesh(pos, nrm, np.tile(np.asarray(attr, np.uint8), (len(pos), 1)), idx.astype(np.uint32)))
    r, y = profile[-1]
    if cap and r > 0:
        meshes.append(prism(circle(x, z, r, segments), y, y, attr))
    return concat(meshes)


def roof(polygon, height, attr, cell=1.0):
    """The surface height(x, z) over the polygon, flat-shaded on a `cell` grid clipped exactly to the footprint."""
    import shapely

    x0, z0, x1, z1 = polygon.bounds
    gx, gz = np.meshgrid(np.arange(math.floor(x0 / cell), math.ceil(x1 / cell)) * cell,
                         np.arange(math.floor(z0 / cell), math.ceil(z1 / cell)) * cell)
    gx, gz = gx.ravel(), gz.ravel()
    parts = shapely.intersection(shapely.box(gx, gz, gx + cell, gz + cell), polygon)
    area = shapely.area(parts)
    full = np.abs(area - cell * cell) < 1e-9
    a, b = np.column_stack([gx[full], gz[full]]), np.column_stack([gx[full] + cell, gz[full]])
    c, d = np.column_stack([gx[full] + cell, gz[full] + cell]), np.column_stack([gx[full], gz[full] + cell])
    tris = [np.stack([a, c, b], 1), np.stack([a, d, c], 1)]
    for g in parts[~full & (area > 1e-9)]:
        for p in getattr(g, "geoms", [g]):
            if p.geom_type == "Polygon" and p.area > 1e-9:
                tris.append(np.array([np.asarray(t.exterior.coords)[:3]
                                      for t in shapely.constrained_delaunay_triangles(p).geoms]))
    xz = np.concatenate([t for t in tris if len(t)])
    y = heights(height, xz[..., 0].ravel(), xz[..., 1].ravel()).reshape(xz.shape[:2])
    return faceted(np.stack([xz[..., 0], y, xz[..., 1]], -1), attr, up=True)


def crenellate(polygon, top, attr, thick=0.5, parapet=1.0, merlon=0.8, width=1.1, period=2.2):
    """A parapet of `thick` along every ring of the footprint, `parapet` above `top`, with merlons every `period`."""
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient

    inner = polygon.buffer(-thick, join_style="mitre")
    ring = polygon.difference(inner)
    parts = [p for p in getattr(ring, "geoms", [ring]) if p.geom_type == "Polygon" and p.area > 1e-6]
    meshes = [prism(p, top, lambda x, z: heights(top, x, z) + parapet, attr) for p in parts]
    for boundary in (polygon.exterior, *polygon.interiors):
        pts = np.asarray(boundary.coords)
        seg = np.diff(pts, axis=0)
        length = np.hypot(seg[:, 0], seg[:, 1])
        total = length.sum()
        for s in np.arange(period / 2, total - width / 2, period):
            i = min(np.searchsorted(np.cumsum(length), s), len(seg) - 1)
            t = seg[i] / max(length[i], 1e-9)
            p = pts[i] + seg[i] * ((s - (np.cumsum(length)[i] - length[i])) / max(length[i], 1e-9))
            inward = np.array([-t[1], t[0]])
            c = p + inward * thick / 2
            corners = [c + t * dx * width / 2 + inward * dz * thick / 2 for dx, dz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            y0 = float(heights(top, np.array([c[0]]), np.array([c[1]]))[0]) + parapet
            footprint = Polygon(corners)
            if footprint.is_valid and footprint.area > 1e-6:
                meshes.append(prism(orient(footprint, 1.0), y0, y0 + merlon, attr))
    return concat(meshes)

def panel(corners, attr):
    """A flat convex polygon (3D corners in order) as a fan of faceted triangles."""
    c = np.asarray(corners, np.float64)
    return faceted(np.stack([np.repeat(c[:1], len(c) - 2, axis=0), c[1:-1], c[2:]], 1), attr)


def bar(p0, p1, size, attr):
    """A square-section beam of side `size` from p0 to p1."""
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    axis = p1 - p0
    axis /= np.linalg.norm(axis)
    u = np.cross(axis, [0.0, 1.0, 0.0] if abs(axis[1]) < 0.9 else [1.0, 0.0, 0.0])
    u *= size / 2 / np.linalg.norm(u)
    v = np.cross(axis, u)
    v *= size / 2 / np.linalg.norm(v)
    ring = [u + v, -u + v, -u - v, u - v]
    faces = [[p0 + ring[k], p0 + ring[(k + 1) % 4], p1 + ring[(k + 1) % 4], p1 + ring[k]] for k in range(4)]
    faces += [[p1 + q for q in ring], [p0 + q for q in reversed(ring)]]
    meshes = [panel(f, attr) for f in faces]
    m = concat(meshes)
    centre = (p0 + p1) / 2                          # turn every face away from the beam's axis
    out = np.einsum("ij,ij->i", m.nrm, m.pos - centre) < 0
    tri_out = out[m.idx[:, 0]]
    idx = m.idx.copy()
    idx[tri_out] = idx[tri_out][:, [0, 2, 1]]
    nrm = m.nrm.copy()
    nrm[np.unique(idx[tri_out])] *= -1
    return Mesh(m.pos, nrm, m.attr, idx)


def fit_rectangle(polygon):
    """(centre x, centre z, length, width, unit long axis) of the footprint's minimum rotated rectangle."""
    pts = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords)[:4]
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    long_, short = (e1, e2) if np.hypot(*e1) >= np.hypot(*e2) else (e2, e1)
    c = pts.mean(axis=0)
    return c[0], c[1], float(np.hypot(*long_)), float(np.hypot(*short)), long_ / np.hypot(*long_)


def gabled(rect, base, eave, rise, wall_attr, roof_attr, overhang=0.3):
    """A rectangular building: walls from `base` to the eave height `eave`, gables and a two-slope roof whose ridge runs
    along the long axis `rise` above the eaves; the roof overhangs every wall by `overhang`."""
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient

    cx, cz, length, width, a = rect
    b = np.array([-a[1], a[0]])
    c = np.array([cx, cz])
    corner = lambda s, t: c + a * s * length / 2 + b * t * width / 2
    footprint = orient(Polygon([corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)]), 1.0)
    walls = prism(footprint, base, eave, wall_attr, cap=False)
    p3 = lambda s, t, y: np.array([*(c + a * s + b * t)[:1], y, *(c + a * s + b * t)[1:]])
    hl, hw = length / 2, width / 2
    gables = [panel([p3(s * hl, -hw, eave), p3(s * hl, hw, eave), p3(s * hl, 0.0, eave + rise)][::s], wall_attr)
              for s in (-1, 1)]
    slope = rise / hw
    e = eave - overhang * slope
    roofs = [panel([p3(-hl - overhang, t * (hw + overhang), e), p3(hl + overhang, t * (hw + overhang), e),
                    p3(hl + overhang, 0.0, eave + rise), p3(-hl - overhang, 0.0, eave + rise)][::t], roof_attr)
             for t in (-1, 1)]
    out = concat([walls, *gables, *roofs])
    centre = np.array([cx, eave, cz])                # face every panel away from the building's middle
    flip = np.einsum("ij,ij->i", out.nrm[out.idx[:, 0]], out.pos[out.idx[:, 0]] - centre) < 0
    idx = out.idx.copy()
    idx[flip] = idx[flip][:, [0, 2, 1]]
    nrm = out.nrm.copy()
    for k in np.flatnonzero(flip):
        nrm[idx[k]] = -out.nrm[out.idx[k]]
    return Mesh(out.pos, nrm, out.attr, idx)


# ---------------------------------------------------------------- triangles and local frames
def oriented(tris, normal):
    """Triangles (t, 3, 3) wound so that cross(b - a, c - a) points along `normal` (one vector or one per triangle)."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = np.einsum("ij,ij->i", n, np.broadcast_to(normal, n.shape)) < 0
    tris = tris.copy()
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return tris


def indexed(tris, normals, attr):
    """Triangles (t, 3, 3) with per-corner normals as a Mesh sharing corners that agree in position and normal."""
    corners = np.concatenate([tris.reshape(-1, 3), np.broadcast_to(normals, tris.shape).reshape(-1, 3)], axis=1)
    keys = np.round(corners * 1e4).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    pos, nrm = corners[first, :3], corners[first, 3:]
    return Mesh(pos, nrm, np.tile(np.asarray(attr, np.uint8), (len(pos), 1)),
                inverse.reshape(-1, 3).astype(np.uint32))


def frame(origin, along, y0):
    """(rotation, translation) taking a vehicle's local (x along, y up, z left) to world, rail top at y0."""
    ax = np.array([along[0], 0.0, along[1]])
    az = np.array([-along[1], 0.0, along[0]])
    return np.column_stack([ax, [0.0, 1.0, 0.0], az]), np.array([origin[0], y0, origin[1]])


def placed(mesh, rot, shift):
    return Mesh(mesh.pos @ rot.T + shift, mesh.nrm @ rot.T, mesh.attr, mesh.idx)


def box(x0, x1, y0, y1, z0, z1, attr, bottom=False):
    """An axis-aligned local box."""
    c = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], np.float64)
    faces = [(0, 1, 3, 2, (-1, 0, 0)), (4, 6, 7, 5, (1, 0, 0)), (2, 3, 7, 6, (0, 1, 0)),
             (0, 4, 5, 1, (0, -1, 0)), (0, 2, 6, 4, (0, 0, -1)), (1, 5, 7, 3, (0, 0, 1))]
    if not bottom:
        faces = [f for f in faces if f[4] != (0, -1, 0)]
    return concat([panel(c[list(f[:4])], attr) for f in faces])


def cylinder_along(axis, c, r, half, attr, segments=16, caps=True, phase=0.0):
    """A local cylinder of radius r about the x or z axis (`axis`), centred at c, `half` long either side."""
    a = phase + 2 * np.pi * np.arange(segments) / segments
    u, v = np.cos(a), np.sin(a)
    ring = lambda s: (np.column_stack([np.full(segments, c[0] + s), c[1] + r * v, c[2] + r * u]) if axis == "x"
                      else np.column_stack([c[0] + r * u, c[1] + r * v, np.full(segments, c[2] + s)]))
    radial = np.column_stack([np.zeros(segments), v, u]) if axis == "x" else np.column_stack([u, v, np.zeros(segments)])
    p0, p1 = ring(-half), ring(half)
    k = np.arange(segments)
    k1 = (k + 1) % segments
    pos = np.concatenate([p0, p1])
    idx = np.concatenate([np.column_stack([k, k1, k1 + segments]), np.column_stack([k, k1 + segments, k + segments])])
    side = Mesh(pos, np.concatenate([radial, radial]), np.tile(np.asarray(attr, np.uint8), (len(pos), 1)), idx)
    tris = oriented(side.pos[side.idx], side.nrm[side.idx].mean(axis=1))
    parts = [indexed(tris, side.nrm[side.idx], attr)]
    if caps:
        e = np.array([1.0, 0.0, 0.0]) if axis == "x" else np.array([0.0, 0.0, 1.0])
        for p, sgn in ((p0, -1.0), (p1, 1.0)):
            fan = np.stack([np.repeat(p[:1], segments - 2, axis=0), p[1:-1], p[2:]], 1)
            parts.append(indexed(oriented(fan, sgn * e), sgn * e, attr))
    return concat(parts)


def pierced(origin, u, v, outline, holes, depth, wall, fill, reveal=None):
    """A flat face spanned by unit vectors u, v from `origin`, outlined by (u, v) points, with rectangular-or-any
    (u, v) holes recessed `depth` behind it to a `fill` panel (one (material, aux) for every hole, or a list of one per
    hole); it faces along u x v."""
    fills = fill if isinstance(fill, list) else [fill] * len(holes)
    out = np.cross(u, v)
    world = lambda uv: origin + np.asarray(uv)[..., :1] * u + np.asarray(uv)[..., 1:2] * v
    face = Polygon(outline)
    hs = [Polygon(h) for h in holes]
    if hs:
        face = face.difference(shapely.union_all(hs))
    flat = np.array([np.asarray(t.exterior.coords)[:3] for t in shapely.constrained_delaunay_triangles(face).geoms])
    parts = [indexed(oriented(world(flat), out), out, wall)]
    for h, hole_fill in zip(holes, fills):
        ring = np.asarray(h, np.float64)
        if Polygon(ring).exterior.is_ccw is False:
            ring = ring[::-1]
        a, b = ring, np.roll(ring, -1, axis=0)
        d = b - a
        inward = np.column_stack([-d[:, 1], d[:, 0]]) / np.hypot(d[:, 0], d[:, 1])[:, None]
        n3 = inward[:, :1] * u + inward[:, 1:2] * v
        fa, fb = world(a), world(b)
        ba, bb = fa - out * depth, fb - out * depth
        quads = np.concatenate([np.stack([fa, fb, bb], 1), np.stack([fa, bb, ba], 1)])
        facing = np.concatenate([n3, n3])
        parts.append(indexed(oriented(quads, facing), facing[:, None, :], reveal or wall))
        pts = world(ring) - out * depth
        fan = np.stack([np.repeat(pts[:1], len(pts) - 2, axis=0), pts[1:-1], pts[2:]], 1)
        parts.append(indexed(oriented(fan, out), out, hole_fill))
    return concat(parts)
