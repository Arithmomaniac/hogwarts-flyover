"""Transport family (mesh-transport): the railway's track and tunnel portal, Hogsmeade station, the Hogwarts Express
and the Durmstrang ship, all static (static-exteriors).

Track: ballast, sleepers and standard-gauge rails along railway.alignment's lines, into the tunnel mouth, whose
headwall closes the junction's rock cutting (railway-regrade, railway-tunnel). Station: the 170-yard platform drawn on
the grounds map beside the coaches, walled along its back with a way out at its north-west end (the lake path) and the
building near its far end, a Plockton-style boarded building with a narrow door onto the road and a part-length canopy
(station-layout-either, station-terrace). Train: nine 64'6" scarlet coaches between two short guards' vans on the
registered coach line, and the engine and tender detached beyond them, nose first round the loop (guards-vans,
short-guards-vans, detached-engine, generic-tender-locomotive, train-scarlet-coaches). Ship: a dark three-master on the
registered hull line, bow to the shore, masts to 180 ft over the lake (ship-skeletal-book).

Vehicles are built in a local frame (x along, y up from the rail top, z to the left) and placed on the track."""
import math
from typing import NamedTuple

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

import railway
from materials import code
from mesh_geometry import (Mesh, box, concat, cylinder_along, frame, indexed, lathe, lid, oriented, panel, pierced,
                           placed, prism)
from mesh_openings import Opening, facade

FAMILY = "transport"

# G-buffer codes from the material registry (src/materials): set pieces (material 7) are shaded by the registry's
# branches in P3 and reflFs
GOLD, HULL, RIG, COPING = code("gold"), code("dark-timber"), code("rigging"), code("pale-stone")
SCARLET, IRON, ROOF, WINDOW = code("scarlet"), code("black-iron"), code("carriage-roof"), code("lamplit-glass")
DIM_WINDOW = code("dim-glass")      # "the dim light issuing from the station windows"
RAIL, SLEEPER, BALLAST, PAVING, BOARDING = (code(n) for n in ("rail", "sleeper", "ballast", "paving", "cream-boarding"))
SLATE, DOOR, VOID, PORTHOLE, SAIL, TRIM = (code(n) for n in ("slate", "green-door", "tunnel-dark", "port-glass", "sail",
                                                             "green-trim"))
STONE = code("wall-stone")          # boundary-wall stone: platform faces, the platform's back wall, the portal
MAST = HULL

GAUGE_M, RAIL_HEAD_M, RAIL_DEPTH_M = 1.435, 0.07, 0.15
SLEEPER_BASE_M, SLEEPER_TOP_M = 0.19, 0.34
RAIL_TOP_M = SLEEPER_TOP_M + RAIL_DEPTH_M
SLEEPER_LENGTH_M, SLEEPER_WIDTH_M, SLEEPER_PITCH_M = 2.6, 0.25, 0.7
BALLAST_M = ((2.3, 0.12), (1.6, 0.27))   # (half-width, top above the formation) of the bed's two steps
PIECE_M = 96.0                      # track is built in pieces this long, so its chunks cull independently
INTO_TUNNEL_M = 6.0

PLATFORM_EDGE_M, PLATFORM_BACK_M = 1.45, 12.7    # from the coaches' line: the coping's edge; the drawn back line
PLATFORM_HEIGHT_M = 0.915                        # above the rail top
RAMP_M, EXIT_GAP_M = 9.0, 6.0
BACK_WALL_M, BACK_WALL_HIGH_M = 0.45, 1.35
COPING_M = 0.6
EAVE_M, ROOF_PITCH = 3.2, 1.0       # building walls above its floor; roof rise per half-width
CANOPY_DEPTH_M, CANOPY_LOW_M, CANOPY_REACH_M = 4.2, 2.9, 7.0

COACH_M, VAN_M, BUFFERS_M = 19.66, 9.0, 0.6      # bodies, and the gap between coupled bodies
COACH_W, COACH_SIDE_M, COACH_ROOF_M = 2.82, 3.3, 3.86
BOGIE_CENTRES_M, BOGIE_WHEELBASE_M, COACH_WHEEL_M = 14.17, 2.6, 0.457
VAN_WHEELBASE_M = 5.5
ENGINE_M, TENDER_M, DRAWBAR_M = 11.4, 7.6, 0.4
PORTAL_FACE_M, PORTAL_W, PORTAL_H, PORTAL_DEPTH_M = 3.5, 4.8, 6.2, 1.2

SHIP_BEAM, SHIP_KEEL_M = 0.72, -5.0          # beam as a fraction of the registered width; keel under the lake
MASTS = ((0.22, 50.0), (0.48, 180 * 0.3048), (0.74, 44.0))   # (fraction from the bow, top over the lake)


# ---------------------------------------------------------------- local-frame geometry
def boxes(centres, along, length, width, y0, y1, attr):
    """Many level boxes at once: centres (n, 2), unit directions (n, 2) of their `length`; tops and sides only."""
    a = np.column_stack([along[:, 0], along[:, 1]])
    b = np.column_stack([-along[:, 1], along[:, 0]])
    corner = lambda i, j: centres + a * (i * length / 2) + b * (j * width / 2)
    ring = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)]
    lo = [np.column_stack([p[:, 0], np.broadcast_to(y0, len(p)), p[:, 1]]) for p in ring]
    hi = [np.column_stack([p[:, 0], np.broadcast_to(y1, len(p)), p[:, 1]]) for p in ring]
    quads = [(hi[0], hi[1], hi[2], hi[3])] + [(lo[k], lo[(k + 1) % 4], hi[(k + 1) % 4], hi[k]) for k in range(4)]
    tris = np.concatenate([np.concatenate([np.stack([q[0], q[1], q[2]], 1), np.stack([q[0], q[2], q[3]], 1)]) for q in quads])
    middle = np.concatenate([centres] * 10)
    mid_y = np.concatenate([(np.broadcast_to(y0, len(centres)) + np.broadcast_to(y1, len(centres))) / 2] * 10)
    mid3 = np.column_stack([middle[:, 0], mid_y, middle[:, 1]])
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    out = np.where(np.einsum("ij,ij->i", n, tris.mean(axis=1) - mid3) < 0, -1.0, 1.0)[:, None] * n
    tris = oriented(tris, out)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return Mesh(tris.reshape(-1, 3), np.repeat(n, 3, axis=0), np.tile(np.asarray(attr, np.uint8), (3 * len(tris), 1)),
                np.arange(3 * len(tris), dtype=np.uint32).reshape(-1, 3))


def strip(left, right, normal, attr):
    """The quad strip between two matching world polylines (n, 3), facing `normal` (n, 3) at each pair."""
    a, b, c, d = left[:-1], left[1:], right[1:], right[:-1]
    n = (normal[:-1] + normal[1:]) / 2
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    facing = np.concatenate([n, n])
    return indexed(oriented(tris, facing), facing[:, None, :], attr)


# ---------------------------------------------------------------- track
def lines(al):
    """(centreline, level, arclength) of every running line: the loop, and the main line into the tunnel."""
    keep = al.main.s <= al.portal_s + INTO_TUNNEL_M
    return [(al.loop.xz, al.loop.level, al.loop.s), (al.main.xz[keep], al.main.level[keep], al.main.s[keep])]


def tangents(xz):
    t = np.gradient(xz, axis=0)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def rails(xz, level):
    """Both rails' heads along a centreline: the inner face, top and outer face of each."""
    t = tangents(xz)
    left = np.column_stack([-t[:, 1], t[:, 0]])
    up = np.tile([0.0, 1.0, 0.0], (len(xz), 1))
    at = lambda off, h: np.column_stack([xz[:, 0] + left[:, 0] * off, level + h, xz[:, 1] + left[:, 1] * off])
    lat = lambda sgn: np.column_stack([left[:, 0] * sgn, np.zeros(len(xz)), left[:, 1] * sgn])
    parts = []
    for side in (-1.0, 1.0):
        inner, outer = side * GAUGE_M / 2, side * (GAUGE_M / 2 + RAIL_HEAD_M)
        parts += [strip(at(inner, SLEEPER_TOP_M), at(inner, RAIL_TOP_M), lat(-side), RAIL),
                  strip(at(inner, RAIL_TOP_M), at(outer, RAIL_TOP_M), up, RAIL),
                  strip(at(outer, RAIL_TOP_M), at(outer, SLEEPER_TOP_M), lat(side), RAIL)]
    return concat(parts)


def sleepers(xz, level, s, at):
    t = tangents(xz)
    c = np.column_stack([np.interp(at, s, xz[:, 0]), np.interp(at, s, xz[:, 1])])
    tt = np.column_stack([np.interp(at, s, t[:, 0]), np.interp(at, s, t[:, 1])])
    tt /= np.linalg.norm(tt, axis=1, keepdims=True)
    y = np.interp(at, s, level)
    return boxes(c, np.column_stack([-tt[:, 1], tt[:, 0]]), SLEEPER_LENGTH_M, SLEEPER_WIDTH_M, y + SLEEPER_BASE_M,
                 y + SLEEPER_TOP_M, SLEEPER)


def tiles(polygon, size=128.0):
    from shapely.geometry import box as rect

    x0, z0, x1, z1 = polygon.bounds
    out = []
    for i in range(math.floor(x0 / size), math.floor(x1 / size) + 1):
        for j in range(math.floor(z0 / size), math.floor(z1 / size) + 1):
            part = polygon.intersection(rect(i * size, j * size, (i + 1) * size, (j + 1) * size))
            out += [orient(g, 1.0) for g in getattr(part, "geoms", [part]) if g.geom_type == "Polygon" and g.area > 1e-6]
    return out


def ballast(al):
    """The ballast bed under every line in two steps, one union so the turnout's overlapping beds never fight."""
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString

    ls = lines(al)
    tree = cKDTree(np.concatenate([xz for xz, _, _ in ls]))
    levels = np.concatenate([lv for _, lv, _ in ls])
    level_at = lambda x, z: levels[tree.query(np.column_stack([np.ravel(x), np.ravel(z)]))[1]].reshape(np.shape(x))
    parts = []
    for half, top in BALLAST_M:
        bed = shapely.union_all([LineString(xz).buffer(half, cap_style="flat") for xz, _, _ in ls]).simplify(0.02)
        for piece in tiles(bed):
            parts.append(prism(piece, lambda x, z: level_at(x, z) - 0.3, lambda x, z, top=top: level_at(x, z) + top,
                               BALLAST))
    return parts


def portal(al):
    """The tunnel's stone headwall closing the cutting: an arched mouth, dark within, under a deep coping that roofs
    the gap to the rock face behind it."""
    p, t, level = al.portal
    face = p - t * PORTAL_FACE_M
    d = np.array([t[1], -t[0]])                 # along the headwall, so that it faces back down the cutting
    half = railway.FORMATION_M + railway.HEADWALL_M / railway.CUT + 1.0
    p0, p1 = face - d * half, face + d * half
    lat = lambda x, z: np.abs((np.stack([x, z], -1) - face) @ d)
    base = lambda x, z: level + railway.CUT * np.maximum(lat(x, z) - railway.FORMATION_M, 0.0) - 0.6
    top = level + railway.HEADWALL_M + 0.6
    mouth = Opening(u=half, sill=level - 0.05, width=PORTAL_W, height=PORTAL_H, depth=PORTAL_DEPTH_M, fill=VOID)
    wall, used = facade(p0, p1, base, top, [mouth], STONE, step=0.5)
    if not used:
        raise ValueError("the tunnel mouth does not fit its headwall")
    cap = orient(Polygon([p0 - t * 0.2, p1 - t * 0.2, p1 + t * (PORTAL_FACE_M + 0.6), p0 + t * (PORTAL_FACE_M + 0.6)]), 1.0)
    return concat([wall, prism(cap, top - 0.6, top + 0.35, COPING)])


def track(al):
    """Rails, sleepers and ballast on every line, in pieces about PIECE_M long, and the tunnel portal."""
    parts = []
    for xz, level, s in lines(al):
        cuts = np.linspace(0, len(s) - 1, max(2, math.ceil(s[-1] / PIECE_M) + 1)).astype(int)
        ties = np.arange(SLEEPER_PITCH_M / 2, s[-1], SLEEPER_PITCH_M)
        for i0, i1 in zip(cuts[:-1], cuts[1:]):
            parts.append(rails(xz[i0:i1 + 1], level[i0:i1 + 1]))
            mine = ties[(ties >= s[i0]) & (ties < s[i1])]
            if len(mine):
                parts.append(sleepers(xz, level, s, mine))
    return parts + ballast(al) + [portal(al)]


# ---------------------------------------------------------------- station
def station(al, ground, footprints):
    """The platform beside the coaches, its back wall and side ramp, the building on its mapped footprint (component
    21) and the canopy over part of the platform."""
    o, a, w = al.frame
    lo, hi = al.platform
    top = al.level + RAIL_TOP_M + PLATFORM_HEIGHT_M
    P = lambda s, t: o + a * s + w * t
    plan = lambda s0, s1, t0, t1: orient(Polygon([P(s0, t0), P(s1, t0), P(s1, t1), P(s0, t1)]), 1.0)
    st = lambda x, z: (np.stack([x, z], -1) - o) @ a
    tt = lambda x, z: (np.stack([x, z], -1) - o) @ w
    sunk = lambda x, z: ground(x, z) - 1.5
    parts = [prism(plan(lo, hi, PLATFORM_EDGE_M, PLATFORM_BACK_M), sunk, top, STONE, cap=False),
             prism(plan(lo, hi, PLATFORM_EDGE_M - 0.02, PLATFORM_EDGE_M + COPING_M), top - 0.06, top + 0.03, COPING),
             lid(plan(lo, hi, PLATFORM_EDGE_M + COPING_M, PLATFORM_BACK_M), top, PAVING)]
    # the way off at the near end: a ramp down from the platform's back towards the lake path's gap in the wall
    ramp = plan(lo, lo + EXIT_GAP_M, PLATFORM_BACK_M, PLATFORM_BACK_M + RAMP_M)
    fall = lambda x, z: top - (top - al.level) * np.clip((tt(x, z) - PLATFORM_BACK_M) / RAMP_M, 0.0, 1.0)
    parts += [prism(ramp, sunk, fall, STONE, cap=False), lid(ramp, fall, PAVING)]

    house = max(footprints, key=lambda g: g.area)
    ring = np.asarray(house.exterior.coords)
    b0, b1 = float(st(ring[:, 0], ring[:, 1]).min()), float(st(ring[:, 0], ring[:, 1]).max())
    f0, f1 = float(tt(ring[:, 0], ring[:, 1]).min()), float(tt(ring[:, 0], ring[:, 1]).max())
    f0 = max(f0, PLATFORM_BACK_M)
    for s0, s1 in ((lo + EXIT_GAP_M, b0), (b1, hi)):            # the back wall, broken by the building
        if s1 - s0 > 0.5:
            parts += [prism(plan(s0, s1, PLATFORM_BACK_M - BACK_WALL_M, PLATFORM_BACK_M), sunk, top + BACK_WALL_HIGH_M, STONE),
                      prism(plan(s0, s1, PLATFORM_BACK_M - BACK_WALL_M - 0.05, PLATFORM_BACK_M + 0.05),
                            top + BACK_WALL_HIGH_M, top + BACK_WALL_HIGH_M + 0.08, COPING)]
    if f0 > PLATFORM_BACK_M + 0.05:
        parts.append(prism(plan(b0, b1, PLATFORM_BACK_M, f0), sunk, top, PAVING))
    parts += building(al, ground, P, plan, (b0, b1, f0, f1), top, w)
    parts += canopy(P, plan, (b0, b1, f0), top, lo, hi)
    return parts


def building(al, ground, P, plan, extent, floor, west):
    """A single-storey boarded building on a stone plinth: windows, doors to the platform, a narrow door and steps to
    the road, a slated roof with green bargeboards and a chimney."""
    b0, b1, f0, f1 = extent
    eave = floor + EAVE_M
    sunk = lambda x, z: ground(x, z) - 1.5
    parts = [prism(plan(b0, b1, f0, f1), sunk, floor - 0.3, STONE)]
    window = lambda u: Opening(u=u, sill=floor + 0.9, width=1.0, height=1.7, depth=0.12, fill=DIM_WINDOW)
    door = lambda u, width=1.1: Opening(u=u, sill=floor, width=width, height=2.3, depth=0.1, fill=DOOR)
    ring = np.asarray(orient(Polygon([P(b0, f0), P(b1, f0), P(b1, f1), P(b0, f1)]), 1.0).exterior.coords)
    road_door = None
    for p0, p1 in zip(ring[:-1], ring[1:]):                      # counter-clockwise in world, so each wall faces out
        length = float(np.hypot(*(p1 - p0)))
        t = (p1 - p0) / length
        if (np.array([t[1], -t[0]]) @ west) < -0.9:            # facing the platform
            spots = np.linspace(0.12, 0.88, 6) * length
            kinds = (window, door, window, window, door, window)
        elif (np.array([t[1], -t[0]]) @ west) > 0.9:           # facing the road
            spots = np.linspace(0.15, 0.85, 4) * length
            kinds = (window, lambda u: door(u, 0.85), window, window)
            road_door = p0 + t * spots[1]
        else:
            spots, kinds = [length / 2], (window,)
        wall, _ = facade(p0, p1, floor - 0.3, eave, [k(u) for k, u in zip(kinds, spots)], BOARDING, reveal=TRIM, step=0.5)
        parts.append(wall)
    half = (f1 - f0) / 2
    ridge = eave + half * ROOF_PITCH
    over = 0.4
    y3 = lambda s, t, y: np.array([*P(s, t)[:1], y, *P(s, t)[1:]])
    for s in (b0, b1):                                           # gables
        tri = [y3(s, f0, eave), y3(s, f1, eave), y3(s, f0 + half, ridge)]
        g = panel(tri, BOARDING)
        out = np.array([*(P(s, 0) - P((b0 + b1) / 2, 0))[:1], 0.0, *(P(s, 0) - P((b0 + b1) / 2, 0))[1:]])
        parts.append(indexed(oriented(g.pos[g.idx], out), out, BOARDING))
        for t in (f0, f1):
            parts.append(bar_between(y3(s + np.sign(s - b0 - 0.1) * over, t + np.sign(t - f0 - 0.1) * over, eave - over * ROOF_PITCH),
                                     y3(s + np.sign(s - b0 - 0.1) * over, f0 + half, ridge + 0.05), 0.12, TRIM))
    drop = over * ROOF_PITCH
    for t_edge in (f0 - over, f1 + over):                        # the two roof slopes
        quad = [y3(b0 - over, t_edge, eave - drop), y3(b1 + over, t_edge, eave - drop),
                y3(b1 + over, f0 + half, ridge), y3(b0 - over, f0 + half, ridge)]
        g = panel(quad, SLATE)
        n = g.nrm[0] if g.nrm[0][1] > 0 else -g.nrm[0]
        parts.append(indexed(oriented(g.pos[g.idx], n), n, SLATE))
    chimney = plan(b0 + 0.25 * (b1 - b0) - 0.45, b0 + 0.25 * (b1 - b0) + 0.45, f0 + half - 0.3, f0 + half + 0.3)
    parts.append(prism(chimney, ridge - 1.2, ridge + 1.3, STONE))
    if road_door is not None:                                   # steps down from the narrow door to the road
        outward = west
        g = float(ground(*(road_door + outward * 1.5)[:, None]))
        rise = floor - g
        n = int(math.ceil(rise / 0.18))
        s_door = float((road_door - P(0, 0)) @ (P(1, 0) - P(0, 0)))
        for k in range(1, n):
            t0 = f1 + (k - 1) * 0.3
            parts.append(prism(plan(s_door - 0.7, s_door + 0.7, t0, t0 + 0.3), sunk, floor - k * rise / n, STONE))
    return parts


def bar_between(p0, p1, size, attr):
    from mesh_geometry import bar

    return bar(p0, p1, size, attr)


def canopy(P, plan, extent, floor, lo, hi):
    """A slated lean-to over the platform in front of the building and a little beyond it, on green posts."""
    b0, b1, f0 = extent
    s0, s1 = max(lo, b0 - CANOPY_REACH_M), min(hi, b1 + CANOPY_REACH_M)
    back_y, front_y = floor + EAVE_M + 0.25, floor + CANOPY_LOW_M
    t_front = f0 - CANOPY_DEPTH_M
    y3 = lambda s, t, y: np.array([*P(s, t)[:1], y, *P(s, t)[1:]])
    up = lambda q: q if q.nrm[0][1] > 0 else Mesh(q.pos, -q.nrm, q.attr, q.idx[:, [0, 2, 1]])
    roof = panel([y3(s0, t_front, front_y), y3(s1, t_front, front_y), y3(s1, f0 + 0.3, back_y), y3(s0, f0 + 0.3, back_y)], SLATE)
    under = panel([y3(s0, t_front, front_y - 0.08), y3(s1, t_front, front_y - 0.08), y3(s1, f0 + 0.3, back_y - 0.08),
                   y3(s0, f0 + 0.3, back_y - 0.08)], TRIM)
    parts = [up(roof), Mesh(under.pos, -up(under).nrm, under.attr, up(under).idx[:, [0, 2, 1]])]
    parts.append(prism(plan(s0, s1, t_front - 0.05, t_front + 0.05), front_y - 0.4, front_y + 0.02, TRIM))   # valance
    posts = np.arange(s0 + 0.3, s1 - 0.2, 3.0)
    for s in list(posts) + [s1 - 0.3]:
        parts.append(prism(plan(s - 0.08, s + 0.08, t_front + 0.4, t_front + 0.56), floor, front_y - 0.05, TRIM))
        if s < b0 or s > b1:
            t_back = PLATFORM_BACK_M - BACK_WALL_M / 2
            parts.append(prism(plan(s - 0.08, s + 0.08, t_back - 0.08, t_back + 0.08), floor, back_y - 0.1, TRIM))
    return parts


# ---------------------------------------------------------------- train
ENGINE_AXLES_M = ((1.4, 0.48), (3.3, 0.48), (5.0, 0.95), (7.15, 0.95), (9.3, 0.95))   # (from the nose, wheel radius)
TENDER_AXLES_M = (-2.2, 0.0, 2.2)
TENDER_WHEEL_M, VAN_WHEEL_M = 0.6, 0.48


class Vehicle(NamedTuple):
    kind: str           # "coach", "van", "tender" or "engine"
    s0: float           # its end towards the coaches' north-west end, round the loop
    s1: float           # its other end
    front: float        # its leading end (the engine's nose)
    axles: tuple        # arclengths of its wheelsets
    radii: tuple        # their wheel radii
    pivots: tuple       # the two arclengths its rigid body hangs between


def consist(al):
    """The train as it stands: guards' van, nine coaches and guards' van on the registered coach line, the second van
    just beyond it, and the tender and engine centred on the engine's registered segment, nose first round the loop."""
    c0, c1 = al.coaches
    spans = []
    end = c1
    for _ in range(9):
        spans.append((end - COACH_M, end))
        end -= COACH_M + BUFFERS_M
    spans.reverse()

    def coach(s0, s1):
        m, b, h = (s0 + s1) / 2, BOGIE_CENTRES_M / 2, BOGIE_WHEELBASE_M / 2
        axles = (m - b - h, m - b + h, m + b - h, m + b + h)
        return Vehicle("coach", s0, s1, s1, axles, (COACH_WHEEL_M,) * 4, (m - b, m + b))

    def van(s0, s1):
        m, h = (s0 + s1) / 2, VAN_WHEELBASE_M / 2
        return Vehicle("van", s0, s1, s1, (m - h, m + h), (VAN_WHEEL_M,) * 2, (m - h, m + h))

    first = spans[0][0] - BUFFERS_M
    out = [van(first - VAN_M, first)] + [coach(*s) for s in spans] + [van(c1 + BUFFERS_M, c1 + BUFFERS_M + VAN_M)]
    e0, e1 = al.engine
    t0 = (e0 + e1) / 2 - (TENDER_M + DRAWBAR_M + ENGINE_M) / 2
    t1 = t0 + TENDER_M
    m = (t0 + t1) / 2
    out.append(Vehicle("tender", t0, t1, t1, tuple(m + a for a in TENDER_AXLES_M), (TENDER_WHEEL_M,) * 3,
                       (m + TENDER_AXLES_M[0], m + TENDER_AXLES_M[-1])))
    g0, g1 = t1 + DRAWBAR_M, t1 + DRAWBAR_M + ENGINE_M
    out.append(Vehicle("engine", g0, g1, g1, tuple(g1 - d for d, _ in ENGINE_AXLES_M), tuple(r for _, r in ENGINE_AXLES_M),
                       (g1 - 9.3, g1 - 2.35)))
    return out


def wheelset(r, attr=IRON):
    """Two wheels on the rails' centres and their axle, in a frame whose y = 0 is the rail top."""
    rail = GAUGE_M / 2 + RAIL_HEAD_M / 2
    return concat([cylinder_along("z", (0.0, r, side * rail), r, 0.065, attr, 18, phase=-np.pi / 2)
                   for side in (-1.0, 1.0)] + [cylinder_along("z", (0.0, r, 0.0), 0.08, rail, IRON, 8)])


def buffers(x, sgn, height=1.05):
    """A pair of sprung buffers at a body end x, pointing along sgn."""
    return [part for z in (-0.87, 0.87) for part in
            (cylinder_along("x", (x + sgn * 0.13, height, z), 0.12, 0.13, IRON, 10),
             cylinder_along("x", (x + sgn * 0.28, height, z), 0.2, 0.02, IRON, 12))]


def ends(length, coupled, height=1.05):
    """Gangways, buffers and couplings at a body's two ends (x = -length/2 and +length/2)."""
    parts = []
    for sgn, joined in zip((-1.0, 1.0), coupled):
        x = sgn * length / 2
        parts.append(box(min(x, x + sgn * 0.25), max(x, x + sgn * 0.25), height, 3.15, -0.45, 0.45, IRON))
        parts += buffers(x, sgn, height)
        if joined:
            parts.append(box(min(x, x + sgn * 0.3), max(x, x + sgn * 0.3), height - 0.08, height + 0.08, -0.06, 0.06, IRON))
    return parts


def side(length, windows, y0=1.05, y1=COACH_SIDE_M):
    """Both scarlet sides of a body `length` long with (centre, width, sill, head) windows set into them."""
    parts = []
    for sgn in (1.0, -1.0):
        origin = np.array([-sgn * length / 2, y0, sgn * COACH_W / 2])
        u, v = np.array([sgn, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        holes = [[(c - w / 2 + length / 2 if sgn > 0 else length / 2 - c - w / 2, s - y0),
                  (c + w / 2 + length / 2 if sgn > 0 else length / 2 - c + w / 2, s - y0),
                  (c + w / 2 + length / 2 if sgn > 0 else length / 2 - c + w / 2, h - y0),
                  (c - w / 2 + length / 2 if sgn > 0 else length / 2 - c - w / 2, h - y0)] for c, w, s, h in windows]
        parts.append(pierced(origin, u, v, [(0, 0), (length, 0), (length, y1 - y0), (0, y1 - y0)], holes, 0.05,
                             SCARLET, WINDOW))
    return parts


def roof(length, y0=COACH_SIDE_M, y1=COACH_ROOF_M, segments=8):
    """The body's arched roof and the scarlet end panels under it."""
    a = np.linspace(0.0, np.pi, segments + 1)
    z, y = COACH_W / 2 * np.cos(a), y0 + (y1 - y0) * np.sin(a)
    parts = []
    for k in range(segments):
        p = lambda x, j: np.array([x, y[j], z[j]])
        mid = np.array([0.0, (y[k] + y[k + 1]) / 2 - (y0 - 1.0), (z[k] + z[k + 1]) / 2])
        n = mid / np.linalg.norm(mid)
        parts.append(strip(np.array([p(-length / 2, k), p(length / 2, k)]), np.array([p(-length / 2, k + 1), p(length / 2, k + 1)]),
                           np.array([n, n]), ROOF))
    for sgn in (-1.0, 1.0):
        cap = np.column_stack([np.full(segments + 1, sgn * length / 2), y, z])
        e = np.array([sgn, 0.0, 0.0])
        parts.append(indexed(oriented(np.stack([np.repeat(cap[:1], segments - 1, axis=0), cap[1:-1], cap[2:]], 1), e), e, ROOF))
        end = np.array([[sgn * length / 2, 1.05, -COACH_W / 2], [sgn * length / 2, 1.05, COACH_W / 2],
                        [sgn * length / 2, y0, COACH_W / 2], [sgn * length / 2, y0, -COACH_W / 2]])
        parts.append(indexed(oriented(np.stack([end[[0, 1, 2]], end[[0, 2, 3]]]), e), e, SCARLET))
    return parts


def coach_body(length, coupled):
    count = 8
    gaps = np.linspace(-length / 2 + 2.9, length / 2 - 2.9, count)
    windows = [(c, 1.5, 1.95, 2.8) for c in gaps] + [(sgn * (length / 2 - 1.2), 0.55, 2.05, 2.8) for sgn in (-1, 1)]
    parts = side(length, windows) + roof(length) + ends(length, coupled)
    parts.append(box(-length / 2 + 1.0, length / 2 - 1.0, 0.62, 1.05, -1.15, 1.15, IRON))        # underframe
    for x in (-BOGIE_CENTRES_M / 2, BOGIE_CENTRES_M / 2):
        parts.append(box(x - 1.7, x + 1.7, 0.36, 0.72, -1.05, 1.05, IRON))                        # bogie frames
    return concat(parts)


def van_body(length, coupled):
    windows = [(sgn * (length / 2 - 1.1), 0.6, 2.1, 2.7) for sgn in (-1, 1)]
    parts = side(length, windows) + roof(length) + ends(length, coupled)
    parts.append(box(-length / 2 + 0.5, length / 2 - 0.5, 0.62, 1.05, -1.15, 1.15, IRON))
    for x in (-VAN_WHEELBASE_M / 2, VAN_WHEELBASE_M / 2):
        parts.append(box(x - 0.6, x + 0.6, 0.4, 0.95, -1.12, 1.12, IRON))                         # axle guards
    return concat(parts)


def tender_body(length):
    parts = [box(-length / 2, length / 2, 0.62, 1.2, -1.2, 1.2, IRON),
             box(-length / 2 + 0.2, length / 2 - 0.3, 1.2, 3.0, -1.35, 1.35, SCARLET),
             box(-0.8, length / 2 - 0.5, 3.0, 3.3, -1.15, 1.15, IRON)]                            # coal
    parts += buffers(-length / 2, -1.0)                                                           # facing the coaches, uncoupled
    parts.append(box(length / 2 - 0.3, length / 2 + DRAWBAR_M + 0.3, 0.95, 1.1, -0.08, 0.08, IRON))   # drawbar to the engine
    return concat(parts)


def engine_body(length):
    """A generic scarlet 4-6-0: boiler, black smokebox and chimney, brass dome, cab, cylinders, splashers and rods."""
    nose, rear = length / 2, -length / 2
    parts = [box(rear + 0.3, nose - 0.3, 0.55, 1.25, -0.6, 0.6, IRON),                          # frames
             box(rear + 1.9, nose - 0.25, 1.25, 1.36, -1.35, 1.35, IRON),                         # running plate
             cylinder_along("x", ((rear + 3.0 + nose - 1.9) / 2, 2.55, 0.0), 0.8, (nose - 1.9 - rear - 3.0) / 2, SCARLET, 20),
             cylinder_along("x", (nose - 1.15, 2.55, 0.0), 0.86, 0.75, IRON, 20),                # smokebox
             box(rear + 2.0, rear + 3.1, 1.36, 3.25, -0.95, 0.95, SCARLET),                       # firebox
             box(nose - 0.3, nose, 0.7, 1.3, -1.35, 1.35, SCARLET),                               # buffer beam
             lathe(nose - 1.2, 0.0, [(0.3, 3.2), (0.28, 3.85), (0.36, 3.95), (0.0, 3.95)], IRON, 14),          # chimney
             lathe(0.3, 0.0, [(0.45, 3.2), (0.42, 3.5), (0.26, 3.72), (0.0, 3.78)], GOLD, 14),                # dome
             lathe(rear + 3.4, 0.0, [(0.25, 3.2), (0.2, 3.5), (0.0, 3.5)], GOLD, 10)]                        # safety valves
    cab = [(rear + 0.25, rear + 2.1)]
    for x0, x1 in cab:
        parts += [box(x0, x1, 1.36, 3.55, -1.4, -1.25, SCARLET), box(x0, x1, 1.36, 3.55, 1.25, 1.4, SCARLET),
                  box(x1 - 0.12, x1, 1.36, 3.55, -1.25, 1.25, SCARLET),
                  box(x0 - 0.1, x1 + 0.1, 3.55, 3.68, -1.45, 1.45, IRON)]
        for z in (-1.405, 1.405):
            parts.append(box(x0 + 0.5, x0 + 1.3, 2.45, 3.15, z - 0.01, z + 0.01, WINDOW))
        for z in (-0.6, 0.6):
            parts.append(box(x1 + 0.005, x1 + 0.02, 2.75, 3.2, z - 0.25, z + 0.25, WINDOW))
    for z in (-1.05, 1.05):
        parts.append(cylinder_along("x", (nose - 2.35, 0.95, z), 0.36, 0.7, IRON, 14))            # cylinders
        drivers = [nose - d for d, r in ENGINE_AXLES_M if r > 0.9]
        for x in drivers:
            parts.append(box(x - 1.0, x + 1.0, 1.36, 2.0, z - 0.3 if z > 0 else z - 0.05, z + 0.05 if z > 0 else z + 0.3, SCARLET))
        parts.append(box(min(drivers) - 0.1, max(drivers) + 0.1, 0.88, 1.02, z * 0.88 - 0.04, z * 0.88 + 0.04, IRON))   # rods
    parts += buffers(nose, 1.0)
    return concat(parts)


def train(al):
    """Every vehicle of the consist placed on the loop: rigid bodies between their pivots, wheelsets on the rails."""
    xz, s = al.loop.xz, al.loop.s
    rail_top = al.level + RAIL_TOP_M
    vehicles = consist(al)
    parts = []
    for k, v in enumerate(vehicles):
        p1, p2 = (railway.point_at(xz, p) for p in sorted(v.pivots))
        along = (p2 - p1) / np.linalg.norm(p2 - p1)
        mid = (min(v.pivots) + max(v.pivots)) / 2
        rot, shift = frame((p1 + p2) / 2, along, rail_top)
        length = v.s1 - v.s0
        body_mid = (v.s0 + v.s1) / 2 - mid
        if v.kind == "coach":
            body = coach_body(length, (True, True))
        elif v.kind == "van":
            body = van_body(length, (k != 0, k == 0))
        elif v.kind == "tender":
            body = tender_body(length)
        else:
            body = engine_body(length)
        body = Mesh(body.pos + np.array([body_mid, 0.0, 0.0]), body.nrm, body.attr, body.idx)
        parts.append(placed(body, rot, shift))
        for a, r in zip(v.axles, v.radii):
            ra, sa = frame(railway.point_at(xz, a), railway.tangent_at(xz, a), rail_top)
            parts.append(placed(wheelset(r, SCARLET if v.kind == "engine" and r > 0.9 else IRON), ra, sa))
    return parts


# ---------------------------------------------------------------- ship
def ship(meta):
    """A dark three-masted ship on the registered hull line, bow to the shore: lofted hull and deck, raised
    forecastle and quarterdeck, a lit stern gallery and a row of dim ports, tapered masts with tops, yards with
    furled sails, a bowsprit and standing rigging."""
    from mesh_geometry import bar

    a, b = (np.asarray(meta["extras"]["ship"][k], np.float64) for k in ("a", "b"))
    lake = float(meta["lake"])
    length = float(np.linalg.norm(b - a))
    u = (b - a) / length
    across = np.array([-u[1], u[0]])
    beam = SHIP_BEAM * meta["extras"]["ship"]["w"] / 2
    at = lambda s, c, y: np.array([a[0] + u[0] * s + across[0] * c, y, a[1] + u[1] * s + across[1] * c])
    f = np.linspace(0.0, 1.0, 41)
    half = beam * np.where(f < 0.4, np.sqrt(np.clip(1 - ((0.4 - f) / 0.4) ** 2, 0, 1)),
                           1 - 0.32 * np.clip((f - 0.72) / 0.28, 0, 1) ** 2)
    sheer = lake + 4.2 + 2.5 * (2 * f - 1) ** 2
    keel = lake + SHIP_KEEL_M + 1.5 * np.clip((0.08 - f) / 0.08, 0, 1)
    pf = np.array([0.0, 0.55, 0.88, 1.0, 1.0, 0.95])
    py = np.array([0.0, 0.06, 0.25, 0.5, 0.85, 1.0])
    parts = []
    for sgn in (1.0, -1.0):
        grid = np.array([[at(f[i] * length, sgn * half[i] * pf[j], keel[i] + (sheer[i] - keel[i]) * py[j])
                          for j in range(len(pf))] for i in range(len(f))])
        for j in range(len(pf) - 1):
            out = np.column_stack([across[0] * sgn * np.ones(len(f)), np.full(len(f), py[j] - 0.4), across[1] * sgn * np.ones(len(f))])
            parts.append(strip(grid[:, j], grid[:, j + 1], out, HULL))
    deck_l = np.array([at(f[i] * length, half[i] * 0.95, sheer[i]) for i in range(len(f))])
    deck_r = np.array([at(f[i] * length, -half[i] * 0.95, sheer[i]) for i in range(len(f))])
    parts.append(strip(deck_l, deck_r, np.tile([0.0, 1.0, 0.0], (len(f), 1)), HULL))
    stern = np.array([at(length, c * half[-1], keel[-1] + (sheer[-1] - keel[-1]) * y) for c, y in
                      zip(np.concatenate([pf, -pf[::-1]]), np.concatenate([py, py[::-1]]))])
    parts.append(indexed(oriented(np.stack([np.repeat(stern[:1], len(stern) - 2, axis=0), stern[1:-1], stern[2:]], 1),
                                  np.array([u[0], 0.0, u[1]])), np.array([u[0], 0.0, u[1]]), HULL))
    level = lambda x, z: np.interp((np.stack([x, z], -1) - a) @ u / length, f, sheer)
    for f0, f1, rise in ((0.03, 0.16, 2.0), (0.76, 1.0, 2.6)):                      # forecastle, quarterdeck
        ss = np.linspace(f0, f1, 8)
        hw = np.interp(ss, f, half) * 0.93
        ring = [at(x * length, w, 0)[[0, 2]] for x, w in zip(ss, hw)] + [at(x * length, -w, 0)[[0, 2]] for x, w in zip(ss[::-1], hw[::-1])]
        parts.append(prism(orient(Polygon(ring), 1.0), lambda x, z: level(x, z) - 0.5, lambda x, z, r=rise: level(x, z) + r, HULL))
    for c in np.linspace(-0.55, 0.55, 5) * half[-1]:                                # stern gallery lights
        parts.append(panel([at(length + 0.05, c - 0.4, sheer[-1] + 0.6), at(length + 0.05, c + 0.4, sheer[-1] + 0.6),
                            at(length + 0.05, c + 0.4, sheer[-1] + 1.5), at(length + 0.05, c - 0.4, sheer[-1] + 1.5)], PORTHOLE))
    for x in np.arange(0.16, 0.86, 3.0 / length):                                   # a row of ports either side
        w = float(np.interp(x, f, half)) + 0.04
        for sgn in (1.0, -1.0):
            q = [at(x * length - 0.3, sgn * w, lake + 1.3), at(x * length + 0.3, sgn * w, lake + 1.3),
                 at(x * length + 0.3, sgn * w, lake + 1.9), at(x * length - 0.3, sgn * w, lake + 1.9)]
            g = panel(q, PORTHOLE)
            n = np.array([across[0] * sgn, 0.0, across[1] * sgn])
            parts.append(indexed(oriented(g.pos[g.idx], n), n, PORTHOLE))
    tops = []
    for frac, top in MASTS:
        s = frac * length
        deck = float(np.interp(frac, f, sheer))
        h = lake + top - deck
        x, _, z = at(s, 0.0, 0.0)
        parts.append(lathe(x, z, [(0.8, deck - 1.0), (0.72, deck + 2.0), (0.52, deck + 0.55 * h), (0.34, deck + 0.85 * h),
                                  (0.2, lake + top)], MAST, 10))
        scale = {0.48: 1.0}.get(frac, 0.92 if frac < 0.48 else 0.75)
        for k, (yf, reach) in enumerate(zip((0.3, 0.5, 0.68, 0.84), (1.95, 1.6, 1.25, 0.9))):
            y = deck + yf * h
            if k in (0, 2):
                parts.append(lathe(x, z, [(0.0, y - 1.2), (2.4 * scale, y - 1.2), (2.4 * scale, y - 0.95), (0.0, y - 0.95)], MAST, 10))
            reach_m = beam * reach * scale
            parts.append(bar(at(s - 0.8, -reach_m, y), at(s - 0.8, reach_m, y), 0.32, RIG))
            parts.append(bar(at(s - 1.1, -0.85 * reach_m, y - 0.45), at(s - 1.1, 0.85 * reach_m, y - 0.45), 0.55, SAIL))
        tops.append((s, deck, deck + 0.5 * h, lake + top))
        for sgn in (1.0, -1.0):                                                      # shrouds
            for ds in (-2.4, -0.8, 0.8, 2.4):
                w = float(np.interp((s + ds) / length, f, half))
                parts.append(bar(at(s + ds, sgn * w, deck + 0.3), at(s, sgn * 0.4, deck + 0.5 * h), 0.1, RIG))
            parts.append(bar(at(s, sgn * 0.4, deck + 0.5 * h), at(s, sgn * 0.25, lake + top - 2.0), 0.1, RIG))
    sprit = (at(0.0, 0.0, sheer[0] + 0.5), at(-0.22 * length, 0.0, sheer[0] + 6.0))
    parts.append(bar(*sprit, 0.6, HULL))
    parts.append(bar(at(tops[0][0], 0.0, tops[0][3] - 1.0), sprit[1], 0.12, RIG))                       # forestay
    for (s0, _, _, y0), (s1, _, mid, _) in zip(tops[1:], tops[:-1]):
        parts.append(bar(at(s0, 0.0, y0 - 1.0), at(s1, 0.0, mid), 0.12, RIG))                           # stays
    parts.append(bar(at(tops[-1][0], 0.0, tops[-1][3] - 1.0), at(length - 1.0, 0.0, sheer[-1] + 2.6), 0.12, RIG))
    return parts


# ---------------------------------------------------------------- family hooks (build_meshes.massing)
def build_footprint(scene, asset, c, polygons):
    """The station on its footprint; the family's other footprints take the generic massing."""
    if asset == "station":
        return {"station": station(scene.line, scene.ground, polygons)}
    return None


def build_extras(scene):
    return {"track": track(scene.line), "train": train(scene.line), "ship": ship(scene.meta)}
