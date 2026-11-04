"""Recessed openings (castle-norman-openings): planar walls with deep round-arched holes, the reveals lining each
hole, and the glass, door or nothing (a through-passage) closing it from behind.

Walls are planar runs between two footprint corners, facing right of their direction (outward for the
counter-clockwise rings `prism` uses). An opening's (u, y) outline is counter-clockwise seen from outside."""
import math
from typing import NamedTuple, Optional, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon

from mesh_geometry import Mesh, concat, heights, indexed, oriented

ARCH_FACET_M = 0.45        # arch heads are faceted about this finely, with an odd count so the crown is level
EDGE_M = 0.3               # least masonry between an opening and the wall's edges or another opening


class Opening(NamedTuple):
    u: float                                   # centre along the wall from its first end (m)
    sill: float                                # world height of the opening's foot (m)
    width: float
    height: float                              # foot to the crown of the arch
    depth: float                               # reveal depth into the wall
    fill: Optional[Tuple[int, int]] = None     # (material, aux) closing the back; None leaves a through-passage
    orders: Tuple[Tuple[float, float], ...] = ()   # a doorway's inner orders, outside in: (inset, further depth)


def outline(o):
    """The opening as (u, y) points, counter-clockwise seen from outside: foot, right jamb, semicircular head."""
    r = o.width / 2
    segments = max(5, 2 * round(math.pi * r / ARCH_FACET_M / 2) + 1)
    a = np.linspace(0.0, math.pi, segments + 1)
    head = np.column_stack([o.u + r * np.cos(a), o.sill + o.height - r + r * np.sin(a)])
    return np.vstack([[[o.u - r, o.sill], [o.u + r, o.sill]], head])


def facade(p0, p1, base, top, openings, wall, reveal=None, step=1.0, centre=None, cuts=()):
    """A wall from p0 to p1 (x, z) standing between `base` and `top` (metres, or callables of x, z sampled every
    `step` along it), with round-arched `openings` cut through it.

    Returns (mesh, used). Openings leaving less than EDGE_M of masonry round them, or that close to an earlier one,
    are left out of `used`. Each used opening is lined with `reveal` faces of its depth and, unless its fill is
    None, closed by a panel at that depth. `cuts` are (u, y) polygons left open with no lining (a passage lined by
    its own tube), which openings give way to. `centre` (x, z) gives the wall face radial normals, so a ring of
 facets
    shades as a round tower drum."""
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    length = float(np.hypot(*(p1 - p0)))
    t = (p1 - p0) / length
    out = np.array([t[1], 0.0, -t[0]])
    u = np.linspace(0.0, length, max(2, math.ceil(length / step) + 1))
    x, z = p0[0] + t[0] * u, p0[1] + t[1] * u
    lo = heights(base, x, z)
    hi = np.maximum(heights(top, x, z), lo + 0.01)
    shell = Polygon(np.vstack([np.column_stack([u, lo]), np.column_stack([u[::-1], hi[::-1]])]))
    inner = shell.buffer(-EDGE_M, join_style="mitre")
    holes = [Polygon(c).intersection(shell) for c in cuts]
    holes = [h for h in holes if h.geom_type == "Polygon" and h.area > 1e-6]
    used = []
    for o in openings:
        hole = Polygon(outline(o))
        if inner.contains(hole) and all(hole.distance(h) >= EDGE_M for h in holes):
            used.append(o)
            holes.append(hole)

    def world(uy):
        uy = np.asarray(uy, np.float64)
        return np.stack([p0[0] + t[0] * uy[..., 0], uy[..., 1], p0[1] + t[1] * uy[..., 0]], -1)

    face = shell.difference(shapely.union_all(holes)) if holes else shell
    flat = np.array([np.asarray(g.exterior.coords)[:3] for g in shapely.constrained_delaunay_triangles(face).geoms])
    tris = oriented(world(flat), out)
    if centre is None:
        normals = np.broadcast_to(out, tris.shape)
    else:
        radial = tris - np.array([centre[0], 0.0, centre[1]])
        radial[..., 1] = 0.0
        normals = radial / np.linalg.norm(radial, axis=-1, keepdims=True)
    parts = [indexed(tris, normals, wall)]
    back = -out

    def flat(polygon, d, attr):
        """A (u, y) polygon `d` behind the wall plane, facing out like the wall."""
        g = shapely.constrained_delaunay_triangles(polygon)
        pts = np.array([np.asarray(t.exterior.coords)[:3] for t in g.geoms])
        return indexed(oriented(world(pts) + back * d, out), out, attr)

    def lining(ring, d0, d1):
        """The reveal round an opening outline from d0 to d1 behind the wall plane, facing into the opening."""
        a, b = ring, np.roll(ring, -1, axis=0)
        d = b - a
        inward = np.column_stack([-d[:, 1], d[:, 0]]) / np.hypot(d[:, 0], d[:, 1])[:, None]
        n3 = np.column_stack([t[0] * inward[:, 0], inward[:, 1], t[1] * inward[:, 0]])
        fa, fb = world(a) + back * d0, world(b) + back * d0
        ba, bb = world(a) + back * d1, world(b) + back * d1
        quads = np.concatenate([np.stack([fa, fb, bb], 1), np.stack([fa, bb, ba], 1)])
        facing = np.concatenate([n3, n3])
        return indexed(oriented(quads, facing), facing[:, None, :], reveal or wall)

    for o in used:
        parts.append(lining(outline(o), 0.0, o.depth))
        inner, depth = o, o.depth
        for inset, further in o.orders:                 # a doorway steps in, order by order
            step = inner._replace(width=inner.width - 2 * inset, height=inner.height - inset)
            parts += [flat(Polygon(outline(inner)).difference(Polygon(outline(step))), depth, reveal or wall),
                      lining(outline(step), depth, depth + further)]
            inner, depth = step, depth + further
        if o.fill is not None:
            pts = world(outline(inner)) + back * depth
            fan = np.stack([np.repeat(pts[:1], len(pts) - 2, axis=0), pts[1:-1], pts[2:]], 1)
            parts.append(indexed(oriented(fan, out), out, o.fill))
    return concat(parts), used


def tube(near, far, attr):
    """The lining of a passage between two matching world rings (n, 3), facing in towards its axis."""
    a, b = near, np.roll(near, -1, axis=0)
    c, d = np.roll(far, -1, axis=0), far
    quads = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    centre = quads.mean(axis=1)
    p0, p1 = near.mean(axis=0), far.mean(axis=0)
    axis = (p1 - p0) / np.linalg.norm(p1 - p0)
    inward = p0 + np.outer((centre - p0) @ axis, axis) - centre
    n = np.cross(quads[:, 1] - quads[:, 0], quads[:, 2] - quads[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    n *= np.where(np.einsum("ij,ij->i", n, inward) < 0, -1.0, 1.0)[:, None]
    return indexed(oriented(quads, n), n[:, None, :], attr)
