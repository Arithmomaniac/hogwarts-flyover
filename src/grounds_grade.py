"""White Hound's slopes kept where a site's own ground breaks them (site X): Hagrid's cabin raised on a terrace above
the gates, with the drive climbing to it; the Whomping Willow on a knoll of its own, in sight of the northern
greenhouses; and the first-years' first sight of the castle kept for the end of their walk to the lake, with
wooded rises between it and the station and the train's approach. Each is a switch in common.SITES; the book evidence
and the user's choices are recorded in tools/manifest/general.py."""
import numpy as np
import shapely
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

import castle_base
import fix_railway
import railway

TERRACE_REACH_M = 150.0         # the cabin's terrace is level this far round it ...
FALL = 1 / 3.0                  # ... then falls 1 in 3, as do the drive's embankment's sides
CREST_M = 12.0                  # half the drive embankment's level top
LAWN_M = 20.0                   # half the width of the lawn kept at the terrace's level from it to the castle's front
RAMP_START_M = 0.3              # the drive leaves the gates this far below their ground, so they keep their own
SHORE_M, SHORE_GRADE, SHORE_REACH_M = 0.35, 0.45, 160.0     # the lake's shore rule away from the castle (build_final)
OFF_ROAD_COST = 20.0            # what a texel off the essay map's roads costs the drive's path, against 1 on them
KNOLL_TOP_M, KNOLL_ROUND_M = 18.0, 6.0   # a knoll: the radius of its top, and how softly its shoulder rounds
KNOLL_FALL = 1 / 2.6            # the grade of its flanks
KNOLL_TOE_M = 8.0               # how broadly its toe blends into the ground round it
SHELF_M, SHELF_FALL = 20.0, 1 / 2.0     # the station wood's shelf rises over this width beyond the boundary wall, then
                                        # falls 1 in 2
WOOD_RIDGE = ((196.0, 206.0), (206.0, 150.0), (214.0, 100.0))   # its ridge's crest, west of the path to the boats
WOOD_RIDGE_HALF_M = 30.0                # from that crest to the ridge's toe
PATH_CLEAR_M = 6.0                      # the walk to the lake and the roads keep their own ground this far either side,
RISE_BATTER = railway.CUT               # and beyond it no rise climbs from them more steeply than the railway's rock cuttings
WOOD_M = 1.0                            # ground raised more than this is planted with dense trees
SCREEN_OUT_M, SCREEN_HALF_M = 24.0, 30.0        # the train's screen: its crest this far out from the track, its flanks
SCREEN_MAIN_M, SCREEN_LOOP_M = 165.0, 105.0     # either side, broad enough to seat the grounds' boundary wall along it;
                                                # from this far up the main line from the toe, past the junction, to
                                                # this far along the loop's station side
SCREEN_KERB_M = fix_railway.LINE_CLEAR_M        # its trees stand no nearer the track, so no crown reaches the train


def drive_line(road, E, N, start, end, margin=150.0):
    """The drive's centre line from `start` to `end` (world east, north): the cheapest 8-connected path over the grid's
    texels, where a texel off the essay map's roads (`road` below 0.3) costs OFF_ROAD_COST times one on them."""
    def texel(p):
        return np.unravel_index(int(np.argmin(np.hypot(E - p[0], N - p[1]))), E.shape)
    (a0, b0), (a1, b1) = texel(start), texel(end)
    step = float(np.hypot(E[0, 1] - E[0, 0], N[0, 1] - N[0, 0]))
    pad = int(np.ceil(margin / step))
    r0, r1 = max(min(a0, a1) - pad, 0), min(max(a0, a1) + pad + 1, E.shape[0])
    c0, c1 = max(min(b0, b1) - pad, 0), min(max(b0, b1) + pad + 1, E.shape[1])
    cost = np.where(road[r0:r1, c0:c1] > 0.3, 1.0, OFF_ROAD_COST)
    h, w = cost.shape
    ids = np.arange(h * w).reshape(h, w)
    rows, cols, weights = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):        # each texel's neighbour di rows down, dj columns across
        a = ids[:h - di, max(-dj, 0):w - max(dj, 0)]
        b = ids[di:, max(dj, 0):w - max(-dj, 0)]
        wt = np.hypot(di, dj) * step * (cost.ravel()[a.ravel()] + cost.ravel()[b.ravel()]) / 2.0
        rows += [a.ravel(), b.ravel()]; cols += [b.ravel(), a.ravel()]; weights += [wt, wt]
    graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(h * w,) * 2)
    source, target = ids[a0 - r0, b0 - c0], ids[a1 - r0, b1 - c0]
    _, pred = dijkstra(graph.tocsr(), indices=source, return_predecessors=True)
    path = [target]
    while path[-1] != source:
        path.append(pred[path[-1]])
    i, j = np.unravel_index(np.array(path[::-1]), (h, w))
    return np.column_stack([E[i + r0, j + c0], N[i + r0, j + c0]])


def _edges_blurred(F, new):
    """`new` with the 2 m band along the edges of what it changed from F blurred, so a hard max or min meets the ground
    without a crease or a smooth blend's over- or undershoot."""
    changed = np.abs(new - F) > 0.05
    edge = ndi.binary_dilation(changed, iterations=2) & ~ndi.binary_erosion(changed, iterations=2)
    return np.where(edge, ndi.gaussian_filter(new, 1.0), new)


def cabin_terrace(F, E, N, cabin, drive, rise, lake_sd, lake):
    """Hagrid's cabin's hollow raised to a terrace `rise` metres above the gates' ground (the drive's first point),
    level within TERRACE_REACH_M of the cabin, then falling 1 in 3; the drive carried up to it from the gates on an
    embankment, and on at the terrace's level until the ground round it rises higher; a lawn kept at least at that level
    from the cabin to the castle's front (the drive's last point), so the lawns fall from the castle to the cabin without
    a hollow below it; nowhere by the lake steeper than its shore rule. Only ever raises the ground."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(drive, axis=0).T))])
    t = np.arange(0.0, s[-1], 1.0)
    line = np.column_stack([np.interp(t, s, drive[:, 0]), np.interp(t, s, drive[:, 1])])
    i, j = np.unravel_index(int(np.argmin(np.hypot(E - line[0, 0], N - line[0, 1]))), E.shape)
    gates = float(F[i, j]) - RAMP_START_M
    level = gates + RAMP_START_M + rise
    reach = float(t[int(np.argmax(np.hypot(*(line - np.asarray(cabin)).T) <= TERRACE_REACH_M))])
    plateau = level - FALL * np.maximum(np.hypot(E - cabin[0], N - cabin[1]) - TERRACE_REACH_M, 0.0)
    crest = np.minimum(gates + (level - gates) * t / reach, level)
    d, k = cKDTree(line).query(np.column_stack([E.ravel(), N.ravel()]), distance_upper_bound=150.0)
    d, k = d.reshape(E.shape), np.minimum(k.reshape(E.shape), len(line) - 1)
    embankment = np.where(np.isfinite(d), crest[k] - FALL * np.maximum(d - CREST_M, 0.0), -np.inf)
    lawn = level - FALL * np.maximum(segment_distance(E, N, cabin, line[-1]) - LAWN_M, 0.0)
    shore = np.where(lake_sd < SHORE_REACH_M, lake + SHORE_M + SHORE_GRADE * np.maximum(lake_sd, 0.0), np.inf)
    raised = np.maximum(F, np.minimum(np.maximum.reduce([plateau, embankment, lawn]), shore))
    return np.clip(_edges_blurred(F, raised), F, np.maximum(F, shore)).astype(np.float32)


def knoll(F, E, N, at, rise):
    """The ground at `at` raised `rise` metres on a knoll of its own: a rounded top KNOLL_TOP_M across its middle, flanks
    falling 1 in 2.6, a toe blending smoothly into the ground round it. Only ever raises the ground."""
    r = np.hypot(E - at[0], N - at[1])
    top = float(F.flat[int(np.argmin(r))]) + rise
    profile = top - castle_base.softplus(r - KNOLL_TOP_M, KNOLL_ROUND_M) * KNOLL_FALL
    return castle_base.smooth_max(F, profile, KNOLL_TOE_M).astype(np.float32)


def line_distance(E, N, line):
    """(metres to the polyline `line` (n, 2), every point at most 1 m apart, index of its nearest point) per (E, N)."""
    d, k = cKDTree(np.asarray(line, float)).query(np.column_stack([np.ravel(E), np.ravel(N)]))
    return d.reshape(np.shape(E)), k.reshape(np.shape(E))


def resampled(line, step=0.5):
    p = np.asarray(line, float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    t = np.linspace(0.0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def room(E, N, walk, own=None):
    """Metres from each (E, N) beyond the ground that keeps its own level, PATH_CLEAR_M either side of the walk to the
    lake and of the ground in `own` (such as roads); 0 within it."""
    dist = line_distance(E, N, resampled(walk))[0]
    if own is not None and own.any():
        step = float(np.hypot(E[0, 1] - E[0, 0], N[0, 1] - N[0, 0]))
        dist = np.minimum(dist, ndi.distance_transform_edt(~own) * step)
    return np.maximum(dist - PATH_CLEAR_M, 0.0)


def station_wood(F, E, N, terrace, west, origin, level, rise, crest, land, room, ridge=WOOD_RIDGE):
    """(ground, wood) keeping PS 6's first sight of the castle for the end of the walk from the platform to the lake,
    with White Hound's "some sort of rise in the ground between the station/path and the lake". Behind the station's
    boundary wall (outside the `terrace` polygon, on the `west` side of the track through `origin`, alongside the
    terrace and fading out over SHELF_M past its ends) a shelf rises `rise` m above the terrace's `level` over SHELF_M,
    then falls 1 in 2; west of the walk a ridge's crest stands `crest` m above that level along `ridge`, on flanks
    falling as cos² to its toe WOOD_RIDGE_HALF_M away. Beside the ground that keeps its own level each rises no more
    steeply than RISE_BATTER over the `room` beyond it (grounds_grade.room), so the walk and the roads keep theirs. `wood`
    is the ground raised more than WOOD_M, for dense pines. Only ever raises `land`."""
    cap = RISE_BATTER * room
    axis = np.array([west[1], -west[0]])                        # along the track
    s = (E - origin[0]) * axis[0] + (N - origin[1]) * axis[1]
    ends = (np.asarray(terrace.exterior.coords) - np.asarray(origin)) @ axis
    fade = np.clip(1.0 - np.maximum(np.maximum(ends.min() - s, s - ends.max()), 0.0) / SHELF_M, 0.0, 1.0)
    beyond = land & ((E - origin[0]) * west[0] + (N - origin[1]) * west[1] > 0.0) & (fade > 0.0) & \
        ~shapely.contains_xy(terrace, E, N)
    out = np.zeros(E.shape)
    out[beyond] = shapely.distance(terrace, shapely.points(E[beyond], N[beyond]))
    shelf = level + rise * np.minimum(out / SHELF_M, 1.0) - SHELF_FALL * np.maximum(out - SHELF_M, 0.0)
    dr = line_distance(E, N, resampled(ridge))[0]
    bump = np.where(dr < WOOD_RIDGE_HALF_M, np.cos(np.pi / 2 * dr / WOOD_RIDGE_HALF_M) ** 2, 0.0)
    raised = np.maximum(np.where(beyond, F + np.minimum((shelf - F) * fade, cap), F),
                        np.where(land, F + np.minimum((level + crest - F) * bump, cap), F))
    new = np.maximum(F, raised)
    return new.astype(np.float32), new - F > WOOD_M


def train_screen(F, E, N, route, formation, castle, crest, keep, room):
    """(ground, wood): White Hound's "barrier – wall, trees or rising land – preventing the castle from being
    seen from the train as it approaches the station", as a wooded ridge on the castle's side of the train's last
    approach. Along `route` ((n, 2), every 0.5 m, on its `formation` levels) and on the side facing `castle`, its crest
    stands `crest` m above the formation SCREEN_OUT_M out, its flanks falling as cos² to its toes SCREEN_HALF_M either
    side; never past the route's ends, and only on `keep`. Beside the ground that keeps its own level it rises no more
    steeply than RISE_BATTER over the `room` beyond it (grounds_grade.room), so the walk to the lake keeps its ground.
    Its trees stand where it does, SCREEN_KERB_M or more from the track. Only ever raises the ground."""
    route = np.asarray(route, float)
    d, k = line_distance(E, N, route)
    t = np.gradient(route, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    p, t = route[k], t[k]
    side = np.sign(t[..., 0] * (N - p[..., 1]) - t[..., 1] * (E - p[..., 0])) == \
        np.sign(t[..., 0] * (castle[1] - p[..., 1]) - t[..., 1] * (castle[0] - p[..., 0]))
    off = np.abs(d - SCREEN_OUT_M)
    bump = np.where(off < SCREEN_HALF_M, np.cos(np.pi / 2 * off / SCREEN_HALF_M) ** 2, 0.0)
    on = keep & side & (k > 0) & (k < len(route) - 1) & (bump > 0.0)
    rise = np.minimum(np.maximum(np.asarray(formation, float)[k] + crest - F, 0.0) * bump, RISE_BATTER * room)
    new = np.where(on, F + rise, F)
    return new.astype(np.float32), on & (bump > 0.05) & (room > 0.0) & (d >= SCREEN_KERB_M)


def made_up(F, filled, may):
    """The ground with the hollows the rises enclose made up, where `may` allows, to the level they would spill at
    (`filled`), as fix_railway makes up an embankment's. Their trees stand on as drawn: a meadow cut into the wood there
    would be a clearing no visitor could read (clearings.py)."""
    return np.where(may & (filled - F > 0.05), filled, F).astype(np.float32)

def segment_distance(E, N, a, b):
    """Metres from each (E, N) to the segment from a to b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((E - a[0]) * ab[0] + (N - a[1]) * ab[1]) / float(ab @ ab), 0.0, 1.0)
    return np.hypot(E - (a[0] + t * ab[0]), N - (a[1] + t * ab[1]))

