"""Hogsmeade's plan, at build time and in two dimensions: each mapped footprint regularized into straight walls and split
into building units, and the streets those units front.

The footprints are traced from the building distance field on the 2 m local grid, so their corners arrive rounded and
a row of cottages arrives as one zig-zag blob. A footprint's walls are found by clustering the near-axis edges of its
trace in its own frame; its outline is the union of the cells of that wall-line grid which the trace mostly covers, so
a row's offsets survive and sub-metre texel noise does not. A footprint that no such grid fits keeps its trace,
straightened. Units are the grid's slabs along the footprint's long axis, thin slabs merged into a neighbour and long
rows cut into plots.

Plan coordinates: a footprint's frame turns world (x east, z north) by -theta about its origin, and plan Y is the
frame's negated second axis, so plan (X, Y) with height Z map to world (x, y, z) by a proper rotation: solids built
in plan space keep their outward winding in the world.
"""
import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from mesh_source import MAP_SIZES

AXIS_TOL_DEG = 20.0     # an edge this close to a frame axis votes for a wall line there
LINE_BANDWIDTH_M = 0.35 # wall lines are the peaks of the votes' density at this bandwidth
LINE_MIN_M = 1.0        # a peak needs this much edge length behind it
RAMP_SNAP_M = 0.6       # a diagonal edge joining two different wall lines is a step: it votes a cross wall
FILL = 0.5              # a wall-grid cell is built when the trace covers this share of it
SIMPLIFY_M = 0.2
REGULAR_IOU = 0.8       # below this the orthogonal fit is no fit: the trace is kept, straightened
IRREGULAR_M = 0.6
SNAP_M = 0.01
MIN_SLAB_M = 3.0        # thinner slabs join a neighbour
MIN_UNIT_M = 1.5        # a unit thinner than this joins any neighbour or, failing that, is dropped
HOLE_MIN_M2 = 8.0       # smaller holes in a fitted outline are notches of the trace, not courtyards
MIN_PART_M2 = 4.0       # smaller separate parts are specks of the trace (one texel is 4 m2)
ROW_MIN_M = 16.0        # a slab at least this long, and ROW_RATIO times longer than deep, is a row of plots
ROW_RATIO = 1.6
PLOT_M = (5.5, 9.5)
ROAD_NEAR_M = 1.5       # a point within this of a road texel centre is on the road
FRONT_SAMPLES_M = (2.0, 4.0, 6.0)
HIGH_STREET_M = 12.0    # a front whose outward samples come this close to the High Street's centreline fronts it
FACING_COS = 0.7        # ...if the centreline lies this nearly straight ahead of it
FRONT_MIN_M = 2.5       # a front this long holds a door and a window


@dataclass(frozen=True)
class Frame:
    """A footprint's plan frame: origin (world x, z) and the angle theta of its first axis from east."""
    ox: float
    oz: float
    theta: float

    @property
    def rot3(self):
        """Plan (X, Y, Z) vectors to world (x, y, z): a proper rotation."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.array([[c, s, 0.0], [0.0, 0.0, 1.0], [s, -c, 0.0]])

    def plan(self, xz):
        xz = np.asarray(xz, np.float64)
        c, s = math.cos(self.theta), math.sin(self.theta)
        dx, dz = xz[..., 0] - self.ox, xz[..., 1] - self.oz
        return np.stack([dx * c + dz * s, dx * s - dz * c], -1)

    def world(self, XY):
        XY = np.asarray(XY, np.float64)
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.stack([self.ox + XY[..., 0] * c + XY[..., 1] * s, self.oz + XY[..., 0] * s - XY[..., 1] * c], -1)

    def world3(self, P, y0):
        """Plan points (X, Y, Z) to world (x, y, z), with Z measured from the world height y0."""
        return np.asarray(P, np.float64) @ self.rot3.T + np.array([self.ox, y0, self.oz])

    def to_plan(self, geom):
        return orient(shapely.transform(geom, self.plan), 1.0)

    def to_world(self, geom):
        return orient(shapely.transform(geom, self.world), 1.0)


@dataclass(frozen=True)
class Footprint:
    """One part of a mapped footprint: its frame, its outline and units in plan coordinates, and how well an
    orthogonal outline fits its trace (`regular` when it does)."""
    frame: Frame
    outline: Polygon
    units: tuple
    regular: bool
    iou: float


def clean(poly):
    """The polygon without repeated or collinear vertices (a straight run of the outline is one edge)."""
    return orient(shapely.simplify(shapely.remove_repeated_points(poly, SNAP_M / 2), 1e-6), 1.0)


def parts(geom):
    if geom.is_empty:
        return []
    return [clean(g) for g in (geom.geoms if isinstance(geom, MultiPolygon) else [geom]) if g.area > 0]


def _rings(poly):
    return [np.asarray(poly.exterior.coords)] + [np.asarray(r.coords) for r in poly.interiors]


def _axis_candidates(trace):
    """Frame angles to try, in [-pi/4, pi/4): the length^2-weighted 4-theta mean of the trace's edges, and the
    minimum rotated rectangle's."""
    c = s = 0.0
    for ring in _rings(trace.simplify(0.3)):
        d = np.diff(ring, axis=0)
        w = (d ** 2).sum(1)
        th = np.arctan2(d[:, 1], d[:, 0])
        c, s = c + (w * np.cos(4 * th)).sum(), s + (w * np.sin(4 * th)).sum()
    mrr = np.asarray(trace.minimum_rotated_rectangle.exterior.coords)
    e = mrr[1] - mrr[0]
    wrap = lambda a: (a + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return [wrap(math.atan2(s, c) / 4), wrap(math.atan2(e[1], e[0]))]


def _peaks(values, weights, lo, hi):
    """Wall lines from weighted votes: the peaks of their density that carry at least LINE_MIN_M of edge, and the
    extent's ends where no peak lies within LINE_BANDWIDTH_M of them; snapped to SNAP_M."""
    v, w = np.asarray(values, np.float64), np.asarray(weights, np.float64)
    peaks = []
    if len(v):
        xs = np.arange(lo - 1.0, hi + 1.0, 0.05)
        dens = (w[None, :] * np.exp(-0.5 * ((xs[:, None] - v[None, :]) / LINE_BANDWIDTH_M) ** 2)).sum(1)
        for k in range(1, len(xs) - 1):
            if dens[k] >= dens[k - 1] and dens[k] > dens[k + 1]:
                near = np.abs(v - xs[k]) < 2 * LINE_BANDWIDTH_M
                if w[near].sum() >= LINE_MIN_M:
                    peaks.append(float(np.average(v[near], weights=w[near])))
    lines = peaks + [end for end in (lo, hi) if all(abs(end - x) > LINE_BANDWIDTH_M for x in peaks)]
    out = []
    for x in sorted(round(x / SNAP_M) * SNAP_M for x in lines):
        if not out or x - out[-1] > LINE_BANDWIDTH_M:
            out.append(x)
    return out


def _wall_lines(p):
    """Wall-line positions (X lines, Y lines) of a plan polygon: the density peaks of its near-axis edges, then the
    cross walls that its diagonal steps imply (the 2 m grid smears a cottage's offset in a row into a ramp)."""
    segs = []
    for ring in _rings(p.simplify(SIMPLIFY_M)):
        segs += list(zip(ring[:-1], ring[1:]))
    a = np.array([s[0] for s in segs])
    b = np.array([s[1] for s in segs])
    d = b - a
    length = np.hypot(d[:, 0], d[:, 1])
    ang = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))
    h, v = ang < AXIS_TOL_DEG, ang > 90 - AXIS_TOL_DEG
    mid = (a + b) / 2
    x0, y0, x1, y1 = p.bounds
    ylines = _peaks(mid[h, 1], length[h], y0, y1)
    xlines = _peaks(mid[v, 0], length[v], x0, x1)
    near = lambda val, lines: min(range(len(lines)), key=lambda k: abs(lines[k] - val))
    xs, xw, ys, yw = list(mid[v, 0]), list(length[v]), list(mid[h, 1]), list(length[h])
    for k in np.flatnonzero(~h & ~v):
        ka, kb = near(a[k, 1], ylines), near(b[k, 1], ylines)
        if ka != kb and abs(a[k, 1] - ylines[ka]) < RAMP_SNAP_M and abs(b[k, 1] - ylines[kb]) < RAMP_SNAP_M:
            xs.append(mid[k, 0])
            xw.append(max(abs(d[k, 1]), LINE_MIN_M))
        ka, kb = near(a[k, 0], xlines), near(b[k, 0], xlines)
        if ka != kb and abs(a[k, 0] - xlines[ka]) < RAMP_SNAP_M and abs(b[k, 0] - xlines[kb]) < RAMP_SNAP_M:
            ys.append(mid[k, 1])
            yw.append(max(abs(d[k, 0]), LINE_MIN_M))
    return _peaks(xs, xw, x0, x1), _peaks(ys, yw, y0, y1)


def _fit(trace, frame):
    p = frame.to_plan(trace)
    xl, yl = _wall_lines(p)
    filled = np.zeros((len(xl) - 1, len(yl) - 1), bool)
    for i in range(len(xl) - 1):
        for j in range(len(yl) - 1):
            cell = box(xl[i], yl[j], xl[i + 1], yl[j + 1])
            filled[i, j] = p.intersection(cell).area >= FILL * cell.area
    cells = [box(xl[i], yl[j], xl[i + 1], yl[j + 1]) for i, j in zip(*np.nonzero(filled))]
    shape = unary_union(cells) if cells else Polygon()
    shape = unary_union([Polygon(g.exterior, [h for h in g.interiors if Polygon(h).area >= HOLE_MIN_M2])
                         for g in parts(shape)]) if not shape.is_empty else shape
    for i in range(len(xl) - 1):                       # cells inside a filled notch are built too
        for j in range(len(yl) - 1):
            filled[i, j] = filled[i, j] or shape.covers(box(xl[i], yl[j], xl[i + 1], yl[j + 1]).centroid)
    iou = shape.intersection(p).area / shape.union(p).area if not shape.is_empty else 0.0
    return shape, iou, filled, np.array(xl), np.array(yl)


def plan_footprint(trace, seed):
    """The parts of one traced footprint (a world polygon) as Footprints, each split into units."""
    centre = trace.centroid
    best = None
    for theta in _axis_candidates(trace):
        frame = Frame(round(centre.x, 3), round(centre.y, 3), theta)
        shape, iou, filled, xl, yl = _fit(trace, frame)
        if best is None or iou > best[1]:
            best = (shape, iou, filled, xl, yl, frame)
    shape, iou, filled, xl, yl, frame = best
    if iou < REGULAR_IOU:
        straight = frame.to_plan(trace.simplify(IRREGULAR_M, preserve_topology=True))
        outline = orient(shapely.set_precision(straight, SNAP_M), 1.0)
        return [Footprint(frame, part, (part,), False, iou) for part in _kept(parts(outline))]
    return [Footprint(frame, p, tuple(split(_cells(p, filled, xl, yl), xl, yl, seed * 7 + k)), True, iou)
            for k, p in enumerate(_kept(parts(orient(shape, 1.0))))]


def _kept(found):
    """Parts worth building: specks under MIN_PART_M2 go, unless nothing bigger is left."""
    big = [p for p in found if p.area >= MIN_PART_M2]
    return big or sorted(found, key=lambda p: -p.area)[:1]


def _cells(part, filled, xl, yl):
    return filled & np.array([[part.covers(box(xl[i], yl[j], xl[i + 1], yl[j + 1]).centroid)
                               for j in range(len(yl) - 1)] for i in range(len(xl) - 1)])


def _runs(col):
    out, j = [], 0
    while j < len(col):
        if col[j]:
            k = j
            while k + 1 < len(col) and col[k + 1]:
                k += 1
            out.append((j, k + 1))
            j = k + 1
        else:
            j += 1
    return tuple(out)


def _merge(units, small, size, min_shared):
    """Units with each unit that `small` picks (thinnest first) joined to the neighbour it shares the longest wall with
    (at least `min_shared` metres), where the two make one polygon."""
    while True:
        merged = False
        for _, n in sorted((size(u), n) for n, u in enumerate(units) if small(u)):
            shared = [(units[n].boundary.intersection(u.boundary).length, m) for m, u in enumerate(units) if m != n]
            for _, m in sorted(((length, m) for length, m in shared if length > min_shared), key=lambda t: (-t[0], t[1])):
                joined = orient(shapely.set_precision(units[n].union(units[m]), SNAP_M), 1.0)
                if joined.geom_type == "Polygon":
                    units = [u for k, u in enumerate(units) if k not in (n, m)] + [joined]
                    merged = True
                    break
            if merged:
                break
        if not merged:
            return units


def split(filled, xl, yl, seed):
    """Units of a regular footprint (plan polygons): slabs of its cell grid along the long axis, slabs shorter than
    MIN_SLAB_M merged into the neighbour they share the longest wall with, slivers under MIN_UNIT_M joined to any
    neighbour or dropped, and long rows cut into plots."""
    long_x = (xl[-1] - xl[0]) >= (yl[-1] - yl[0])
    grid = filled if long_x else filled.T
    a_lines, b_lines = (xl, yl) if long_x else (yl, xl)
    rect = (lambda a0, a1, b0, b1: box(a0, b0, a1, b1)) if long_x else (lambda a0, a1, b0, b1: box(b0, a0, b1, a1))
    slabs, i = [], 0
    while i < grid.shape[0]:
        runs = _runs(grid[i])
        k = i
        while k + 1 < grid.shape[0] and _runs(grid[k + 1]) == runs:
            k += 1
        slabs += [(a_lines[i], a_lines[k + 1], b_lines[j0], b_lines[j1]) for j0, j1 in runs]
        i = k + 1
    units = [rect(*s) for s in slabs]
    along = lambda g: (g.bounds[2] - g.bounds[0]) if long_x else (g.bounds[3] - g.bounds[1])
    thinness = lambda g: min(g.bounds[2] - g.bounds[0], g.bounds[3] - g.bounds[1])
    units = _merge(units, lambda g: along(g) < MIN_SLAB_M, along, 0.5)
    units = _merge(units, lambda g: thinness(g) < MIN_UNIT_M, thinness, 0.05)
    if len(units) > 1:
        units = [u for u in units if thinness(u) >= MIN_UNIT_M]
    rng = np.random.default_rng(seed)
    out = []
    for u in sorted(units, key=lambda g: (round(g.bounds[0], 2), round(g.bounds[1], 2))):
        x0, y0, x1, y1 = u.bounds
        is_rect = abs(u.area - (x1 - x0) * (y1 - y0)) < 1e-6
        length, depth = ((x1 - x0), (y1 - y0)) if long_x else ((y1 - y0), (x1 - x0))
        if is_rect and length >= ROW_MIN_M and length > ROW_RATIO * depth:
            n = max(2, int(round(length / rng.uniform(*PLOT_M))))
            cuts = np.linspace(0.0, length, n + 1)
            cuts[1:-1] += rng.uniform(-0.15, 0.15, n - 1) * (length / n)
            cuts = np.round(cuts / SNAP_M) * SNAP_M
            start = x0 if long_x else y0
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                out.append(box(start + c0, y0, start + c1, y1) if long_x else box(x0, start + c0, x1, start + c1))
        else:
            out.append(u)
    return [clean(u) for u in out]


# ---------------------------------------------------------------- the essay map and the streets
def hogs_world(px, georef):
    """Hogsmeade-map pixels -> world (east, north) metres through the essay-map registration."""
    _, key, w, h = next(m for m in MAP_SIZES if m[0] == "hogs")
    x0, x1, y0, y1 = georef[key]
    px = np.asarray(px, np.float64)
    u, v = x0 + px[..., 0] * (x1 - x0) / w, y0 + px[..., 1] * (y1 - y0) / h
    a, b = (u - georef["U0"]) * georef["mFF"], (v - georef["V0"]) * georef["mFF"]
    r = math.radians(georef["rot_deg"])
    return np.stack([a * math.cos(r) + b * math.sin(r), a * math.sin(r) - b * math.cos(r)], -1)


class Streets:
    """Where the roads are: the road layer's texel centres (world x, z) and the High Street's centreline."""

    def __init__(self, road_xz, high_street_xz):
        from scipy.spatial import cKDTree

        self.tree = cKDTree(np.asarray(road_xz, np.float64))
        self.high = LineString(np.asarray(high_street_xz, np.float64))

    def road_share(self, pts):
        d, _ = self.tree.query(np.asarray(pts, np.float64))
        return float(np.mean(d < ROAD_NEAR_M))

    def high_street_distance(self, pts):
        return min(self.high.distance(Point(p)) for p in np.asarray(pts, np.float64))

    def faces_high_street(self, share, pts, mid, out):
        """Whether a wall at world midpoint `mid`, facing `out`, with `share` of its outward samples `pts` on the road,
        looks across the road at the High Street: its centreline within HIGH_STREET_M of the samples and ahead of it."""
        if share == 0.0 or self.high_street_distance(pts) >= HIGH_STREET_M:
            return False
        to = np.asarray(self.high.interpolate(self.high.project(Point(mid))).coords[0]) - mid
        return float(out @ to) > FACING_COS * float(np.linalg.norm(out) * np.linalg.norm(to))


def edges(poly):
    """The exterior edges (a, b) of a counter-clockwise polygon, in order: edge k runs from point k to point k + 1."""
    ring = np.asarray(orient(poly, 1.0).exterior.coords)
    return [(ring[k], ring[k + 1]) for k in range(len(ring) - 1)]


def outward(a, b):
    """The outward unit normal (x, z) of a counter-clockwise ring's edge from a to b."""
    d = (b - a) / np.linalg.norm(b - a)
    return np.array([d[1], -d[0]])


def front_edge(poly, party, streets, to_world=lambda p: p):
    """(edge index, fronts the High Street) of a unit's front. The candidates are its free edges at least FRONT_MIN_M
    long, or its longest free edge if none is. Of those facing the High Street (Streets.faces_high_street), if any, else
    of all, the front is the one with the most road frontage: the share of its outward samples on the road times its
    length, the longer on a tie. With no road beside any, it is the one facing the nearest road.
    `poly` is counter-clockwise in its own coordinates; `to_world` takes its points to world (x, z)."""
    walls = []
    for k, (a, b) in enumerate(edges(poly)):
        length = float(np.linalg.norm(b - a))
        if k in party or length < 1.0:
            continue
        n = outward(a, b)
        along = [a + (b - a) * t for t in (0.2, 0.35, 0.5, 0.65, 0.8)]
        pts = to_world(np.array([p + n * d for p in along for d in FRONT_SAMPLES_M]))
        mid = to_world((a + b) / 2)
        walls.append((k, length, streets.road_share(pts), pts, mid, to_world((a + b) / 2 + n) - mid))
    if not walls:
        return 0, False
    walls = [w for w in walls if w[1] >= FRONT_MIN_M] or [max(walls, key=lambda w: w[1])]
    high = [w for w in walls if w[1] >= FRONT_MIN_M and streets.faces_high_street(*w[2:])]
    best = max(high or walls, key=lambda w: (w[2] * w[1], w[1]))
    if best[2] > 0.0:
        return best[0], bool(high)
    centre = to_world(np.asarray(poly.centroid.coords[0]))
    _, i = streets.tree.query(centre)
    return max(walls, key=lambda w: float(w[5] @ (streets.tree.data[i] - centre)))[0], False
