"""How far White Hound's Forest reaches over a site's real roads, railway and buildings. Where his reasoning does not need
the Forest there, it ends short of them; where it does, corridors are cleared through it. A site's rules are its
SITES[...]["forest"] (None keeps his Forest as drawn); src/mesh_scene.json records the decisions with his words."""
import re

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import shapely

import clearings
from common import to_tm

A_ROAD_M = 12.0             # cleared each side of an A-road's centre line: carriageway, verges and fences
RAILWAY_M = 10.0            # each side of a railway's
BUILDING_M = 10.0           # round a building: its yard
HOME_WOBBLE_M, HOME_BLUR_M, HOME_SEED = 8.0, 12.0, 11       # beyond its yard a homestead's clearing wanders out by up
                                                            # to twice this (clearings.glade)
SETBACK_MIN_M, SETBACK_M, SETBACK_MAX_M = 25.0, 50.0, 90.0     # the Forest's edge short of a road it ends at
VILLAGE_MIN_M, VILLAGE_M, VILLAGE_MAX_M = 40.0, 60.0, 100.0    # and round the buildings along that road
VILLAGE_REACH_M = 150.0     # a building this near such a road belongs to the settlement along it
WOBBLE_M, WOBBLE_SPAN_M, SEED = 20.0, 150.0, 7                 # the edge wanders by about this much, over this far
EDGE_M = 24.0               # inside its new edge the Forest thickens to its drawn density over this far
STEP_M = 1.0                # lines are measured from points this far apart
A_ROAD = re.compile(r"\bA\d+\b")


def is_a_road(tags):
    return "highway" in tags and bool(A_ROAD.search(tags.get("ref") or ""))


def is_railway(tags):
    return tags.get("railway") == "rail"


def is_building(tags):
    return "building" in tags


def ends_here(tags, rules):
    """Whether the Forest ends at this road."""
    return "highway" in tags and bool(set(re.split(r"\s*;\s*", tags.get("ref") or "")) & set(rules.get("ends_at", ())))


def ways(osm, anchor, rules):
    """The OpenStreetMap ways the rules read: (tags, castle-relative E, N)."""
    out = []
    for el in osm["elements"] if rules is not None else ():
        tags, g = el.get("tags", {}), el.get("geometry")
        if el["type"] == "way" and g and (is_a_road(tags) or is_railway(tags) or is_building(tags)
                                         or ends_here(tags, rules)):
            e, n = to_tm.transform([p["lon"] for p in g], [p["lat"] for p in g])
            out.append((tags, np.asarray(e) - anchor[0], np.asarray(n) - anchor[1]))
    return out


def _points(lines):
    """(k, 2) points every STEP_M or less along the polylines."""
    pts = [np.empty((0, 2))]
    for e, n in lines:
        for k in range(len(e) - 1):
            m = max(1, int(np.ceil(np.hypot(e[k + 1] - e[k], n[k + 1] - n[k]) / STEP_M)))
            t = np.arange(m) / m
            pts.append(np.column_stack([e[k] + t * (e[k + 1] - e[k]), n[k] + t * (n[k + 1] - n[k])]))
        pts.append(np.array([[e[-1], n[-1]]]))
    return np.vstack(pts)


def _near(lines, E, N, within):
    """Where texel centres lie within `within` metres (at most) of the polylines (to STEP_M / 2)."""
    pts = _points(lines)
    if not len(pts):
        return np.zeros(E.shape, bool)
    d = cKDTree(pts).query(np.column_stack([E.ravel(), N.ravel()]), distance_upper_bound=within)[0]
    return (d < within).reshape(E.shape)


def homesteads(E, N, ways_, texel):
    """The multiplier (0 to 1) of White Hound's Forest round the buildings it covers, at a layer's texel centres E, N,
    `texel` metres apart. Each building's yard, BUILDING_M round it (its texels a diagonal further) and inside it, is
    cleared, and the clearing spreads on beyond as the homestead's ground: its edge wanders, and the Forest thickens back
    over EDGE_M (clearings.glade), so that it reads as a croft's clearing, not as a cutout round its walls."""
    reach = texel * np.sqrt(2.0) + STEP_M
    built = [(e, n) for t, e, n in ways_ if is_building(t)]
    yard = _near(built, E, N, BUILDING_M + reach)
    homes = [shapely.make_valid(shapely.Polygon(np.column_stack([e, n]))) for e, n in built if len(e) >= 4]
    if homes:
        inside = shapely.union_all(homes)
        shapely.prepare(inside)
        yard |= shapely.contains_xy(inside, E, N)
    if not yard.any():
        return np.ones(E.shape, np.float32)
    return clearings.glade(yard, texel, HOME_SEED, margin_m=0.0, wobble_m=HOME_WOBBLE_M, blur_m=HOME_BLUR_M,
                           feather_m=EDGE_M)


def ends(rules, E, N, ways_, texel):
    """The multiplier (0 to 1) of White Hound's Forest where it ends: beyond each road it ends at, seen from the castle
    (0, 0), and short of it by a setback, it is 0, and so round the buildings along that road and between them and it;
    inside that new edge it thickens over EDGE_M. Its texels are cleared a diagonal further, so that a bilinear read of
    the layer is 0 everywhere within the setbacks."""
    reach = texel * np.sqrt(2.0) + STEP_M
    out = np.ones(E.shape, np.float32)
    ends_at = [(e, n) for t, e, n in ways_ if ends_here(t, rules)]
    if ends_at:
        xy = np.column_stack([E.ravel(), N.ravel()])
        wobble = ndi.gaussian_filter(np.random.default_rng(SEED).standard_normal(E.shape), WOBBLE_SPAN_M / texel)
        wobble *= WOBBLE_M / wobble.std()
        road = cKDTree(_points(ends_at))
        d = road.query(xy, distance_upper_bound=SETBACK_MAX_M + reach + EDGE_M)[0].reshape(E.shape)
        inward = d - np.clip(SETBACK_M + wobble, SETBACK_MIN_M, SETBACK_MAX_M)       # metres inside the new edge
        settled = []
        for t, e, n in ways_:
            if is_building(t):
                dist, k = road.query(_points([(e, n)]))
                if dist.min() <= VILLAGE_REACH_M:
                    to = road.data[k[dist.argmin()]]
                    settled += [(e, n), (np.array([e.mean(), to[0]]), np.array([n.mean(), to[1]]))]
        if settled:
            d = cKDTree(_points(settled)).query(xy, distance_upper_bound=VILLAGE_MAX_M + reach + EDGE_M)[0]
            inward = np.minimum(inward, d.reshape(E.shape) - np.clip(VILLAGE_M + wobble, VILLAGE_MIN_M, VILLAGE_MAX_M))
        sides, _ = ndi.label(inward >= 0)
        castle = sides[np.unravel_index(np.argmin(E * E + N * N), E.shape)]
        if not ((sides > 0) & (sides != castle)).any():
            raise ValueError(f"the Forest ends at {', '.join(rules['ends_at'])}, which cuts nothing off from the castle")
        out = np.where(sides == castle, np.clip((inward - reach) / EDGE_M, 0.0, 1.0), 0.0).astype(np.float32)
    return out


def keep(rules, E, N, ways_, texel):
    """The multiplier (0 to 1) of White Hound's Forest density at a layer's texel centres E, N, `texel` metres apart:
    where it ends (`ends`); along the A-roads and the railway it still covers, corridors cleared, their texels a diagonal
    further so that a bilinear read of the layer is 0 within them; and round its buildings, their homesteads' clearings
    (`homesteads`)."""
    if rules is None:
        return np.ones(E.shape, np.float32)
    reach = texel * np.sqrt(2.0) + STEP_M
    corridor = np.zeros(E.shape, bool)
    for select, half in ((is_a_road, A_ROAD_M), (is_railway, RAILWAY_M)):
        corridor |= _near([(e, n) for t, e, n in ways_ if select(t)], E, N, half + reach)
    out = ends(rules, E, N, ways_, texel) * homesteads(E, N, ways_, texel)
    out[corridor] = 0.0
    return out
