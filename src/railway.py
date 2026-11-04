"""Railway alignment and earthworks (build time, mesh-transport).

The manifest's track entry traces White Hound's line on the grounds map (turning-loop): the junction, the loop from it
past the platform and round, and the main line to the map's edge. Fitted here in world metres, the loop runs along
meta.train's coach segment (the registered ink); the engine stands on the curve round the foot of the loop with its
ends on its segment; and the junction is a turnout: the station side runs on
from the main line and the north side leaves it at TURNOUT_DEG. The formation follows the user's choice
(railway-regrade): the whole loop level with the station, the main line climbing 1 in 40 to a tunnel portal
(railway-tunnel) meta.train's portal_m up it from the turnout's toe, else PORTAL_M, rock cuttings at CUT and embankments
at FILL; the station terrace is level from the track to the boundary wall (station-terrace) and the lake path is cut
through the spur at 1 in 4.

`envelope` gives every point's highest and lowest allowed ground; clamping the terrain between them builds the
earthworks (fix_railway.py). The design depends only on the manifest and meta, never on the terrain it reshapes.
"""
import math
from typing import NamedTuple

import numpy as np

from mesh_source import MAP_SIZES

GRADE = 1 / 40
FORMATION_M = 5.5                   # level half-width either side of a running line (the 2 m terrain keeps it level under the ballast)
CUT, FILL = 2.0, 1 / 1.5            # side slopes: rise per metre out from the formation
PATH_HALF_M, PATH_GRADE, PATH_CUT = 1.2, 0.25, 1.0
PORTAL_M, HEADWALL_M = 25.0, 10.0   # portal along the main line from the toe (unless meta.train sets portal_m);
                                    # rock face starts this far up
TURNOUT_DEG = 6.0                   # the north side leaves the main line at about 1 in 9.5
TOE_M = 20.0                        # the turnout's toe, up the main line from the traced junction
TURNOUT_M = 12.0                    # the main line stays level over the turnout before it climbs
LEAD_M, FREE_M = 8.0, 14.0          # pinned straight leads; traced points nearer the junction than FREE_M give way
NORTH_FREE_M = 35.0                 # ... and on the north side, which must swing round to the toe
STEP_M = 0.5
TRACE_SIGMA_M, PIN_SIGMA_M, DESIGN_SIGMA_M = 0.6, 0.005, 0.05
ROOM_M = 8.0                        # traced points this near a pinned or designed stretch give way to it
FOOT_R_M = 66.0                     # the drawn foot of the loop, from the engine's nose round to the east side, is
FOOT_JOIN_M = 150.0                 # nearly a circle of this radius; the designed foot rejoins the ink this far round
REACH_M = 160.0                     # no earthwork reaches further from its feature


class Line(NamedTuple):
    xz: np.ndarray      # (n, 2) centreline every STEP_M
    s: np.ndarray       # (n,) arclength from the junction (the path: from the platform)
    level: np.ndarray   # (n,) formation level


class Feature(NamedTuple):
    xz: np.ndarray      # centreline samples (unused for a polygon)
    level: np.ndarray
    half: float
    cut: float
    fill: float         # 0: never raises the ground
    polygon: object = None    # a level area instead of a centreline
    before: tuple = None      # (point, direction): only points with (q - point) . direction <= 0


class Alignment(NamedTuple):
    toe: np.ndarray     # the turnout's toe, where the loop's two sides leave the main line
    loop: Line          # toe -> station side -> foot of the loop -> north side -> toe
    main: Line          # toe -> portal -> tunnel
    coaches: tuple      # loop arclengths of the coach segment's north-west and south-east ends
    engine: tuple       # loop arclengths of the engine segment's ends
    portal: tuple       # (point, unit tangent into the tunnel, formation level)
    frame: tuple        # (origin, along, west): the coaches' north-west end and the platform side
    platform: tuple     # (start, end) metres along the coaches from their north-west end
    terrace: object     # shapely Polygon
    path: Line
    level: float
    portal_s: float = PORTAL_M      # the portal's arclength along the main line from the toe


def grounds_world(px, georef):
    """Grounds-map pixels -> world (east, north) metres through the essay-map registration."""
    _, key, w, h = next(m for m in MAP_SIZES if m[0] == "grounds")
    x0, x1, y0, y1 = georef[key]
    px = np.asarray(px, np.float64)
    u, v = x0 + px[..., 0] * (x1 - x0) / w, y0 + px[..., 1] * (y1 - y0) / h
    a, b = (u - georef["U0"]) * georef["mFF"], (v - georef["V0"]) * georef["mFF"]
    r = math.radians(georef["rot_deg"])
    return np.stack([a * math.cos(r) + b * math.sin(r), a * math.sin(r) - b * math.cos(r)], -1)


def arclength(p):
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])


def resample(p, step=STEP_M):
    s = arclength(p)
    t = np.linspace(0.0, s[-1], max(2, int(round(s[-1] / step)) + 1))
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def point_at(p, at):
    s = arclength(p)
    return np.array([np.interp(at, s, p[:, 0]), np.interp(at, s, p[:, 1])])


def tangent_at(p, at, h=1.0):
    d = point_at(p, at + h) - point_at(p, at - h)
    return d / np.linalg.norm(d)


def project(p, q):
    """Arclength along polyline p of its closest point to q."""
    a, d = p[:-1], np.diff(p, axis=0)
    t = np.clip(np.einsum("ij,ij->i", q - a, d) / np.einsum("ij,ij->i", d, d), 0.0, 1.0)
    k = int(np.argmin(np.hypot(*(a + d * t[:, None] - q).T)))
    return float(arclength(p)[k] + t[k] * np.hypot(*d[k]))


def unit(v):
    return v / np.linalg.norm(v)


def rotate(v, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


def fitted(points, sigma):
    """A smoothing spline through ordered points (each within about its sigma), resampled every STEP_M."""
    from scipy.interpolate import splev, splprep

    u = arclength(points)
    tck, _ = splprep(points.T, w=1.0 / np.asarray(sigma), u=u / u[-1], s=float(len(points)), k=3)
    return resample(np.column_stack(splev(np.linspace(0.0, 1.0, int(u[-1] / 0.1) + 1), tck)))


def traced(scene, asset, id):
    return next(t for t in next(a for a in scene["assets"] if a["id"] == asset)["traced"] if t["id"] == id)["plan_px"]


def pinned(a, b):
    n = max(2, int(math.ceil(np.linalg.norm(b - a))) + 1)
    return a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]


def arc(p0, p1, radius, turn, step=1.0):
    """Points every `step` on the circular arc of `radius` from p0 to p1, turning left (turn = 1) or right (-1)."""
    chord = p1 - p0
    c = float(np.linalg.norm(chord))
    d = chord / c
    left = np.array([-d[1], d[0]])
    centre = (p0 + p1) / 2 + left * turn * math.sqrt(radius ** 2 - (c / 2) ** 2)
    start = math.atan2(*(p0 - centre)[::-1])
    sweep = 2 * math.asin(c / (2 * radius))
    a = start + turn * np.linspace(0.0, sweep, max(2, int(math.ceil(radius * sweep / step)) + 1))
    return centre + radius * np.column_stack([np.cos(a), np.sin(a)])


def s_curve(p0, t0, p1, t1, step=1.0):
    """Points every `step` on the cubic Bezier from p0 heading t0 to p1 heading t1 whose sharpest bend is gentlest."""
    u = np.linspace(0.0, 1.0, 400)[:, None]
    chord = float(np.linalg.norm(p1 - p0))
    best = None
    for d0 in np.linspace(0.05, 0.75, 29) * chord:
        for d1 in np.linspace(0.05, 0.75, 29) * chord:
            c = (p0, p0 + t0 * d0, p1 - t1 * d1, p1)
            v = 3 * ((1 - u) ** 2 * (c[1] - c[0]) + 2 * u * (1 - u) * (c[2] - c[1]) + u ** 2 * (c[3] - c[2]))
            a = 6 * ((1 - u) * (c[2] - 2 * c[1] + c[0]) + u * (c[3] - 2 * c[2] + c[1]))
            k = np.abs(v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]) / np.hypot(*v.T) ** 3
            if best is None or k.max() < best[0]:
                best = (k.max(), c)
    c = best[1]
    pts = (1 - u) ** 3 * c[0] + 3 * u * (1 - u) ** 2 * c[1] + 3 * u ** 2 * (1 - u) * c[2] + u ** 3 * c[3]
    return resample(pts, step)


def alignment(scene, meta, georef):
    track = traced(scene, "track", "turning-loop")
    junction = grounds_world(track["junction"], georef)
    loop = grounds_world(track["loop"], georef)
    loop[0] = loop[-1] = junction
    main = grounds_world(track["main"], georef)
    main[0] = junction
    level = float(meta["train"]["level"])
    car, eng = (np.asarray(meta["train"][k], np.float64) for k in ("carriages", "engine"))

    total = arclength(loop)[-1]
    station_dir = unit(point_at(loop, 30.0) - junction)
    main_dir = unit(point_at(main, 30.0) - junction)
    ahead = unit(station_dir - main_dir)             # through the traced junction from the main line onto the station side
    lead = np.arange(1.0, LEAD_M + 1e-9)
    sm = arclength(main)
    points = np.concatenate([junction[None], junction - ahead * lead[:, None], main[sm > FREE_M]])
    sigma = np.concatenate([np.full(1 + len(lead), PIN_SIGMA_M), np.full(int((sm > FREE_M).sum()), TRACE_SIGMA_M)])
    through = fitted(points, sigma)                  # main line from the traced junction outwards

    # The ink shows the two sides apart from the traced junction on, so the turnout's toe stands TOE_M up the main
    # line, where the north side can leave it at TURNOUT_DEG and curve gently away.
    toe = point_at(through, TOE_M)
    into = -tangent_at(through, TOE_M)
    north_dir = unit(point_at(loop, total - 30.0) - junction)
    side = 1.0 if into[0] * north_dir[1] - into[1] * north_dir[0] > 0 else -1.0
    branch = rotate(into, side * TURNOUT_DEG)
    shared = np.concatenate([toe[None], through[arclength(through) < TOE_M - 1e-6][::-1]])

    s = arclength(loop)
    c0, c1 = project(loop, car[0]), project(loop, car[1])
    tail, nose = project(loop, eng[0]), project(loop, eng[1])
    # Past the coaches the ink swings out round WH's drawing of the engine, then rounds the foot of the loop in what is
    # nearly a circle of FOOT_R_M from the engine's nose to the east side. The line there is designed: the engine
    # stands on that curve with its ends on its ink, the gentlest S joins it to the coaches' straight, and the gentlest
    # curve carries it on round the foot to the drawn east side.
    join_s = nose + FOOT_JOIN_M
    join = point_at(loop, join_s)
    join_dir = unit(point_at(loop, join_s + 10.0) - point_at(loop, join_s - 10.0))
    chord = unit(eng[1] - eng[0])
    turn = 1.0 if chord[0] * join_dir[1] - chord[1] * join_dir[0] > 0 else -1.0
    half = math.degrees(math.asin(np.linalg.norm(eng[1] - eng[0]) / (2 * FOOT_R_M)))
    tail_dir, nose_dir = rotate(chord, -turn * half), rotate(chord, turn * half)
    bridge = s_curve(car[1], unit(car[1] - car[0]), eng[0], tail_dir)[1:-1]
    standing = arc(eng[0], eng[1], FOOT_R_M, turn)
    foot = s_curve(eng[1], nose_dir, join, join_dir)[1:]
    keep = (s > FREE_M) & (s < total - NORTH_FREE_M) & ((s < c0 - ROOM_M) | (s > join_s + ROOM_M))
    items = [(-TOE_M, shared, PIN_SIGMA_M), (1.0, junction + ahead * lead[:, None], PIN_SIGMA_M),
             (c0, pinned(car[0], car[1]), PIN_SIGMA_M), (c1 + 0.5, bridge, DESIGN_SIGMA_M),
             (tail, standing, DESIGN_SIGMA_M), (nose + 0.5, foot, DESIGN_SIGMA_M),
             (total + 1.0, (toe + branch * lead[:, None])[::-1], PIN_SIGMA_M), (total + 2.0, toe[None], PIN_SIGMA_M)]
    items += [(float(t), p[None], TRACE_SIGMA_M) for t, p in zip(s[keep], loop[keep])]
    items.sort(key=lambda item: item[0])
    points = np.concatenate([pts for _, pts, _ in items])
    sigma = np.concatenate([np.full(len(pts), sg) for _, pts, sg in items])
    loop_xz = fitted(points, sigma)
    loop_s = arclength(loop_xz)
    main_xz = np.concatenate([toe[None], through[arclength(through) > TOE_M + 1e-6]])
    main_s = arclength(main_xz)

    along = unit(car[1] - car[0])
    west = np.array([-along[1], along[0]])
    if west @ (loop_xz.mean(axis=0) - car[0]) > 0:
        west = -west                                 # the platform side faces away from the loop's middle
    ends = [grounds_world(traced(scene, "station", "platform")[k], georef) for k in ("north_west", "south_east")]
    platform = tuple(float((e - car[0]) @ along) for e in ends)
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient

    terrace = orient(Polygon(grounds_world(traced(scene, "station", "station-terrace"), georef)).buffer(0), 1.0)
    lake = grounds_world(traced(scene, "station", "lake-path"), georef)
    path_xz = resample(lake)
    path_s = arclength(path_xz)
    gap = project(path_xz, lake[1])
    portal_s = float(meta["train"].get("portal_m", PORTAL_M))
    portal = (point_at(main_xz, portal_s), tangent_at(main_xz, portal_s), level + GRADE * (portal_s - TURNOUT_M))
    return Alignment(
        toe=toe,
        loop=Line(loop_xz, loop_s, np.full(len(loop_s), level)),
        main=Line(main_xz, main_s, level + GRADE * np.maximum(main_s - TURNOUT_M, 0.0)),
        coaches=tuple(project(loop_xz, p) for p in car), engine=tuple(project(loop_xz, p) for p in eng),
        portal=portal, frame=(car[0], along, west), platform=platform, terrace=terrace,
        path=Line(path_xz, path_s, level - PATH_GRADE * np.maximum(path_s - gap, 0.0)), level=level, portal_s=portal_s)


def lake_walk(al, scene, georef):
    """The first-years' whole walk from the platform to the shore (PS 6), every STEP_M: the lake path, then White
    Hound's path to the boats (1d) on through the pines and round its last bend on to the lakeside."""
    rest = resample(grounds_world(traced(scene, "station", "path-to-boats"), georef))
    return np.concatenate([al.path.xz, rest[1:]])


def features(al):
    upto = al.main.s <= al.portal_s + STEP_M
    return [Feature(al.loop.xz, al.loop.level, FORMATION_M, CUT, FILL),
            Feature(al.main.xz[upto], al.main.level[upto], FORMATION_M, CUT, FILL, before=al.portal[:2]),
            Feature(None, np.array([al.level]), 0.0, CUT, FILL, polygon=al.terrace),
            Feature(al.path.xz, al.path.level, PATH_HALF_M, PATH_CUT, 0.0)]


def envelope(al, x, z):
    """(cut, fill, bed): the earthworks' highest and lowest allowed ground at world points, and the points on a
    running line's level formation."""
    import shapely
    from scipy.spatial import cKDTree

    q = np.column_stack([np.ravel(x), np.ravel(z)]).astype(np.float64)
    cut, fill = np.full(len(q), np.inf), np.full(len(q), -np.inf)
    bed = np.zeros(len(q), bool)
    for k, f in enumerate(features(al)):
        if f.polygon is not None:
            d, level = shapely.distance(shapely.points(q), f.polygon), np.full(len(q), f.level[0])
        else:
            d, i = cKDTree(f.xz).query(q, distance_upper_bound=REACH_M)
            level = f.level[np.minimum(i, len(f.xz) - 1)]
        reach = np.maximum(d - f.half, 0.0)
        c = level + f.cut * reach
        fl = level - f.fill * reach if f.fill > 0 else np.full(len(q), -np.inf)
        out = ~np.isfinite(d) | (d > REACH_M)
        if f.before is not None:
            out |= (q - f.before[0]) @ f.before[1] > 0
        c[out], fl[out] = np.inf, -np.inf
        cut, fill = np.minimum(cut, c), np.maximum(fill, fl)
        if k < 2:
            bed |= ~out & (d <= f.half)
    p, t, level = al.portal                          # rock face over the portal, rising from the headwall's top
    a, lat = (q - p) @ t, np.abs((q - p) @ np.array([-t[1], t[0]]))
    face = level + HEADWALL_M + CUT * a + CUT * np.maximum(lat - (FORMATION_M + HEADWALL_M / CUT), 0.0)
    cut = np.where(a > 0, np.minimum(cut, face), cut)
    shape = np.shape(x)
    return cut.reshape(shape), fill.reshape(shape), bed.reshape(shape)


def regrade(natural, cut, fill):
    return np.minimum(np.maximum(natural, fill), cut)
