"""Hogsmeade's small structures, standing on the sampled ground: the High Street's cast-iron lamps and its flagstone
footway, the kirkyard's wall and headstones, and garden walls and gates near the High Street."""
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient

import mesh_village_plan as plan

LAMP = {"every": 30.0, "out": 1.6, "end": 0.3, "clear": 0.6,
        "plinth": 0.46, "plinth_top": 0.6, "foot": 0.3, "post": (0.16, 0.08), "post_top": 3.4,
        "lantern": 0.34, "lantern_h": 0.48, "cap": 0.44, "cap_h": 0.3,
        "flare": (0.26, 0.45),          # the post's flared foot: (width, height)
        "frame": 0.035,                 # the lantern's iron corner bars' width
        "arm": (0.56, 0.35, 0.04),      # the ladder bar across the post: (length, its depth under the lantern, width)
        "finial": (0.07, 0.16)}         # a knob on the cap's apex: (its widest, its height over the apex)
KIRKYARD = {"out": 8.0, "piece": 4.0, "thick": 0.5, "high": 1.1, "foot": 0.3, "clear": 1.0}
HEADSTONE = {"wide": (0.45, 0.75), "thick": (0.1, 0.16), "high": (0.7, 1.1), "foot": 0.2, "rows": 6, "every": 2.0,
             "out": (2.5, 4.5),         # their rows' distances out from the nave's long wall
             "off": (0.35, 0.3),        # each off its place in the rows by up to this along and across them
             "turn": 12.0, "lean": 6.0,  # turned and leaning by up to these, in degrees
             "gone": 0.3}               # the share of places with no stone
GARDEN = {"near": 60.0, "reach": 15.0, "deep": 3.0, "back": 0.8, "thick": 0.4, "high": 1.0, "foot": 0.3,
          "step": 1.5, "gate": 1.4, "pier": 0.45, "pier_h": 1.3,
          "short": 0.25}                # a return stops this short of the house's front
FOOTWAY = {"wide": 2.0, "kerb": 0.12, "foot": 0.2, "into": 0.05, "under": 0.05, "door": 1.0, "every": 4.0,
           "clear": 0.01}               # a kerb face nearer a wall than this is left out: it would fight the wall's face


def high_street_fronts(v):
    """(start, end, outward normal) of every High Street building's front, in world (x, z)."""
    out = []
    for cid in sorted(v.units):
        for u in v.units[cid]:
            if u.high:
                a, b = plan.edges(u.outline)[u.front]
                f = u.fp.frame
                n = f.world((a + b) / 2 + plan.outward(a, b)) - f.world((a + b) / 2)
                out.append((f.world(a), f.world(b), n / np.linalg.norm(n)))
    return out


def lamps(v):
    """The High Street's lamps (HBP 27; White Hound: Victorian cast iron, and none up the Hog's Head's side road), one
    piece each: every LAMP['every'] metres along the street, alternately on either side, on the pavement LAMP['out']
    in front of the end of a High Street building's front nearest that spot (clear of its door), where it stands clear
    of every building."""
    line = v.streets.high
    start, end = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
    axis = (end - start) / np.linalg.norm(end - start)
    left = np.array([-axis[1], axis[0]])
    spots = []                                      # (along the street, side, spot, along its front)
    for a, b, n in high_street_fronts(v):
        for p, q in ((a, b), (b, a)):
            t = (q - p) / np.linalg.norm(q - p)
            spot = p + t * LAMP["end"] + n * LAMP["out"]
            spots.append((float((spot - start) @ axis), n @ left > 0.0, spot, t))
    out = []
    for k, target in enumerate(np.arange(LAMP["every"] / 2, float((end - start) @ axis), LAMP["every"])):
        for side in ((True, False) if k % 2 == 0 else (False, True)):
            near = sorted(((abs(s - target), i) for i, (s, sd, _, _) in enumerate(spots)
                           if sd == side and abs(s - target) < LAMP["every"] / 2))
            pick = next((spots[i] for _, i in near if v.clear_of_buildings(spots[i][2], LAMP["clear"])), None)
            if pick is not None:
                out.append(_lamp(v.scene, pick[2], pick[3]))
                break
    return out


def _lamp(scene, p, along):
    """A lamp at world point p, squared to the front it stands before (along: a unit vector along it), an ornate
    Victorian one in cast iron (White Hound): a stone plinth sunk LAMP['foot'] into the ground; a tapering post on a
    flared foot, a ladder bar across it under the lantern; a glass lantern framed in iron at its corners; an iron cap
    crowned by a finial."""
    from mesh_geometry import concat

    g = float(scene.ground(*p))
    base, post, lantern = g + LAMP["plinth_top"], g + LAMP["post_top"], g + LAMP["post_top"] + LAMP["lantern_h"]
    top = lantern + LAMP["cap_h"]
    (flare, flare_h), bar = LAMP["flare"], LAMP["frame"]
    (arm, arm_down, arm_w), (knob, knob_h) = LAMP["arm"], LAMP["finial"]
    across = np.array([-along[1], along[0]])
    corners = [p + (along * x + across * z) * (LAMP["lantern"] / 2) for x, z in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    y = post - arm_down
    return concat([_frustum(p, LAMP["plinth"], LAMP["plinth"], g - LAMP["foot"], base, "pale-stone", along, top=True),
                   _frustum(p, *LAMP["post"], base, post, "black-iron", along),
                   _frustum(p, flare, LAMP["post"][0] * 0.9, base, base + flare_h, "black-iron", along),
                   _slab(p - along * arm / 2, p + along * arm / 2, arm_w, y - arm_w / 2, y + arm_w / 2, "black-iron",
                         bottom=True),
                   _frustum(p, LAMP["lantern"], LAMP["lantern"], post, lantern, "lamp-glass", along, bottom=True)]
                  + [_frustum(c, bar, bar, post, lantern, "black-iron", along, bottom=True) for c in corners]
                  + [_frustum(p, LAMP["cap"], 0.0, lantern, top, "black-iron", along, bottom=True),
                     _frustum(p, 0.02, knob, top - 0.02, top + knob_h * 0.45, "black-iron", along),
                     _frustum(p, knob, 0.0, top + knob_h * 0.45, top + knob_h, "black-iron", along)])


def _frustum(p, w0, w1, y0, y1, mat, along=(1.0, 0.0), top=False, bottom=False):
    """A square frustum centred on world point p and squared to `along` (a unit vector in world x, z), w0 wide at height
    y0 and w1 at y1 (0: a pyramid), with its top and bottom faces if asked, every face turned outward."""
    import mesh_village as mv
    from materials import code
    from mesh_geometry import concat, panel

    t = np.asarray(along, np.float64)
    n = np.array([-t[1], t[0]])
    ring = [t * x + n * z for x, z in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    lo = [np.array([p[0] + r[0] * w0 / 2, y0, p[1] + r[1] * w0 / 2]) for r in ring]
    hi = [np.array([p[0] + r[0] * w1 / 2, y1, p[1] + r[1] * w1 / 2]) for r in ring]
    faces = [[lo[i], lo[(i + 1) % 4], hi[(i + 1) % 4]] + ([hi[i]] if w1 > 0 else []) for i in range(4)]
    if top and w1 > 0:
        faces.append(hi)
    if bottom:
        faces.append(lo)
    return mv._outward(concat([panel(f, code(mat)) for f in faces]), np.array([p[0], (y0 + y1) / 2, p[1]]))


def footway(v):
    """The High Street's flagstone footway (White Hound: "fancy shop signs hanging above the pavements"), on which its
    lamps stand: one surface running FOOTWAY['wide'] out from every High Street building's front and FOOTWAY['into']
    into its wall, cut back where any building stands, in a piece for each of the payload's tiles it crosses. Its top is
    draped FOOTWAY['kerb'] over the ground, through points at most FOOTWAY['every'] apart along its edges; those within
    FOOTWAY['door'] of the nearest High Street front, along it, stand no higher than FOOTWAY['under'] below that
    building's floor (which stands only FLOOR_OVER_M over the ground before it), so that where it meets the front it
    stays under the floor, and no door opens below it. Its kerb, its edge, runs from its top FOOTWAY['foot'] down into
    the ground. Its edges are split where they come within FOOTWAY['clear'] of a wall, and its sides that near a wall,
    which would fight the wall's face for its pixels, and those between its tiles, which nothing sees, are left out."""
    from build_meshes import TILE_M
    from materials import code

    strips, fronts, floors = [], [], []
    for cid in sorted(v.units):
        for u in v.units[cid]:
            if not u.high:
                continue
            a, b = plan.edges(u.outline)[u.front]
            f = u.fp.frame
            A, B = f.world(a), f.world(b)
            n = f.world((a + b) / 2 + plan.outward(a, b)) - f.world((a + b) / 2)
            n /= np.linalg.norm(n)
            back = 2 * FOOTWAY["into"]                          # the walls cut it back to `into` inside them
            strips.append(Polygon([A - n * back, B - n * back, B + n * FOOTWAY["wide"], A + n * FOOTWAY["wide"]]))
            fronts.append(LineString([A, B]))
            floors.append(v.eave(u) - u.style.eaves)
    if not strips:
        return []
    walls = shapely.union_all(list(v.outlines.values()))
    way = shapely.union_all(strips).difference(walls.buffer(-FOOTWAY["into"]))
    tree, floors = shapely.STRtree(fronts), np.asarray(floors)

    def top(x, z):
        x, z = np.asarray(x, np.float64), np.asarray(z, np.float64)
        points = shapely.points(x, z)
        found = tree.query_nearest(points, all_matches=False)
        cap = np.full(len(x), np.inf)
        near = shapely.distance(np.asarray(fronts, dtype=object)[found[1]], points[found[0]]) <= FOOTWAY["door"]
        cap[found[0][near]] = floors[found[1][near]] - FOOTWAY["under"]
        return np.minimum(np.asarray(v.scene.ground(x, z), np.float64) + FOOTWAY["kerb"], cap)

    def base(x, z):
        return np.minimum(np.asarray(v.scene.ground(x, z), np.float64) - FOOTWAY["foot"], top(x, z) - 0.05)

    inner = way.buffer(-0.001)
    walled = walls.buffer(FOOTWAY["clear"])
    near = walled.boundary
    hidden = lambda x, z: shapely.contains_xy(walled, x, z) | shapely.contains_xy(inner, x, z)
    x0, z0, x1, z1 = way.bounds
    pieces = []
    for i in range(math.floor(x0 / TILE_M), math.floor(x1 / TILE_M) + 1):
        for j in range(math.floor(z0 / TILE_M), math.floor(z1 / TILE_M) + 1):
            part = way.intersection(box(i * TILE_M, j * TILE_M, (i + 1) * TILE_M, (j + 1) * TILE_M))
            for g in getattr(part, "geoms", [part]):
                if g.geom_type == "Polygon" and g.area > 1e-3:
                    g = _noded(shapely.segmentize(g, FOOTWAY["every"]), near)
                    pieces.append(_draped(g, base, top, code("flagstone"), hidden))
    return pieces


def _noded(poly, lines, gap=1e-4):
    """Polygon `poly` (world x, z) with a vertex wherever one of its edges crosses `lines` more than `gap` from the
    edge's ends, so that each of its edges lies to one side of them."""
    x0, z0, x1, z1 = poly.bounds
    lines = shapely.clip_by_rect(lines, x0 - 1.0, z0 - 1.0, x1 + 1.0, z1 + 1.0)

    def ring(r):
        p = np.asarray(r.coords)
        hits = shapely.intersection(shapely.linestrings(np.stack([p[:-1], p[1:]], axis=1)), lines)
        out = []
        for a, b, hit in zip(p[:-1], p[1:], hits):
            out.append(a)
            d = b - a
            length = float(np.hypot(*d))
            s = np.unique((shapely.get_coordinates(hit) - a) @ d / length) if length else np.empty(0)
            s = s[(s > gap) & (s < length - gap)]
            out.extend(a + s[:, None] * d / length)
        return out + [p[-1]]

    return Polygon(ring(poly.exterior), [ring(r) for r in poly.interiors])


def _draped(poly, base, top, attr, hidden):
    """A slab over polygon `poly` (world x, z) from `base` to `top` (callables of x, z): its top, and its sides but
    those whose middles `hidden` (a callable of x, z) holds, every face turned outward."""
    from mesh_geometry import Mesh, concat, heights, lid

    poly = orient(poly, 1.0)                            # sides face out of counter-clockwise rings, as in prism
    out = [lid(poly, top, attr)]
    for ring in (poly.exterior, *poly.interiors):
        a = np.asarray(ring.coords)[:-1]
        b = np.roll(a, -1, axis=0)
        d = b - a
        length = np.hypot(d[:, 0], d[:, 1])
        keep = (length > 1e-9) & ~hidden((a[:, 0] + b[:, 0]) / 2, (a[:, 1] + b[:, 1]) / 2)
        if not keep.any():
            continue
        a, b, d, length = a[keep], b[keep], d[keep], length[keep]
        nrm = np.column_stack([d[:, 1], np.zeros(len(d)), -d[:, 0]]) / length[:, None]
        quad = np.stack([np.column_stack([p[:, 0], heights(h, p[:, 0], p[:, 1]), p[:, 1]])
                         for p, h in ((a, base), (b, base), (b, top), (a, top))], 1)
        idx = (4 * np.arange(len(quad))[:, None] + np.array([0, 3, 1, 1, 3, 2])).reshape(-1, 3)
        out.append(Mesh(quad.reshape(-1, 3), np.repeat(nrm, 4, axis=0),
                        np.tile(np.asarray(attr, np.uint8), (4 * len(quad), 1)), idx.astype(np.uint32)))
    return concat(out)


def kirkyard(v):
    """The kirkyard (the user's choice of Hogsmeade's small structures): its wall, then its headstones."""
    return kirkyard_wall(v) + headstones(v)


def _kirk(v):
    """(the kirk's units, its nave, its plan in world) or None."""
    kirk = [u for us in v.units.values() for u in us if u.asset == "kirk"]
    if not kirk:
        return None
    nave = max(kirk, key=lambda u: u.outline.area)
    return kirk, nave, shapely.union_all([u.fp.frame.to_world(u.outline) for u in kirk])


def kirkyard_wall(v):
    """The kirkyard's wall: low rubble KIRKYARD['out'] round the kirk, in pieces of at most KIRKYARD['piece'] that step
    with the ground, none within reach of a road or a building, so the path comes in through a gap."""
    found = _kirk(v)
    if found is None:
        return []
    _, _, world = found
    ring = np.asarray(orient(world.buffer(KIRKYARD["out"], join_style=2), 1.0).exterior.coords)
    out = []
    for p, q in zip(ring[:-1], ring[1:]):
        n = max(1, math.ceil(float(np.linalg.norm(q - p)) / KIRKYARD["piece"]))
        for i in range(n):
            a, b = p + (q - p) * i / n, p + (q - p) * (i + 1) / n
            along = a + (b - a) * np.linspace(0.0, 1.0, 5)[:, None]
            if v.streets.road_share(along) > 0.0 or not all(v.clear_of_buildings(x, KIRKYARD["clear"]) for x in along):
                continue
            g = np.asarray(v.scene.ground(along[:, 0], along[:, 1]), np.float64)
            out.append(_slab(a, b, KIRKYARD["thick"], float(g.min()) - KIRKYARD["foot"],
                             float(g.max()) + KIRKYARD["high"], "rubble"))
    return out


def headstones(v):
    """A few headstones in rows on the nave's side away from its porch, each one piece."""
    found = _kirk(v)
    if found is None:
        return []
    kirk, nave, world = found
    return _headstones(v, nave, kirk, world.buffer(KIRKYARD["out"] - 1.0, join_style=2))


def _headstones(v, nave, kirk, inside):
    """Rows of headstones on the nave's long side away from its porch, their faces to the nave's ends, clear of the
    buildings and inside the kirkyard: old stones, no two alike. Each is a slab of seeded height, width and thickness,
    off its place in the rows, turned and leaning a little, and some places have none (HEADSTONE)."""
    f = nave.fp.frame
    edges = plan.edges(nave.outline)
    k = max(range(len(edges)), key=lambda j: np.linalg.norm(edges[j][1] - edges[j][0]))
    porch = [u for u in kirk if u is not nave]
    a, b = edges[k]
    if porch and float((np.asarray(porch[0].outline.centroid.coords[0]) - (a + b) / 2) @ plan.outward(a, b)) > 0.0:
        k = (k + 2) % len(edges)                       # the opposite long wall, away from the porch
        a, b = edges[k]
    t, n, length = (b - a) / np.linalg.norm(b - a), plan.outward(a, b), float(np.linalg.norm(b - a))
    rng = np.random.default_rng(nave.cid * 13 + 5)
    out = []
    for row in range(HEADSTONE["rows"]):
        for d in HEADSTONE["out"]:
            high, wide, thick = (float(rng.uniform(*HEADSTONE[key])) for key in ("high", "wide", "thick"))
            along, out_by, turn, lean = rng.uniform(-1.0, 1.0, 4) * (*HEADSTONE["off"], HEADSTONE["turn"],
                                                                       HEADSTONE["lean"])
            gone = rng.uniform() < HEADSTONE["gone"]
            s = length / 2 + (row - (HEADSTONE["rows"] - 1) / 2) * HEADSTONE["every"] + along
            at = a + t * s + n * (d + out_by)
            p = f.world(at)
            if gone or not v.clear_of_buildings(p, 1.0) or not inside.contains(Point(p)):
                continue
            g = float(v.scene.ground(*p))
            turn = math.radians(turn)                  # its width runs out from the wall, so its faces look along the
            across = f.world(at + (n * math.cos(turn) + t * math.sin(turn)) * 0.5) - p    # nave to its ends
            side = across / np.linalg.norm(across) * wide / 2
            out.append(_slab(p - side, p + side, thick, g - HEADSTONE["foot"], g + high, "pale-stone",
                             lean=(high + HEADSTONE["foot"]) * math.tan(math.radians(lean))))
    return out


def _slab(a, b, thick, y0, y1, mat, lean=0.0, bottom=False):
    """A box along world segment a-b (x, z), `thick` across it, from height y0 to y1, its top moved `lean` across it
    so that it leans, without its bottom face unless asked."""
    import mesh_village as mv
    from materials import code
    from mesh_geometry import concat, panel

    d = (b - a) / np.linalg.norm(b - a)
    w = np.array([-d[1], d[0]]) * thick / 2
    shift = w / np.linalg.norm(w) * lean
    ring = [a - w, b - w, b + w, a + w]
    lo = [np.array([p[0], y0, p[1]]) for p in ring]
    hi = [np.array([p[0] + shift[0], y1, p[1] + shift[1]]) for p in ring]
    faces = [hi] + [[lo[i], lo[(i + 1) % 4], hi[(i + 1) % 4], hi[i]] for i in range(4)] + ([lo] if bottom else [])
    centre = np.mean(lo + hi, axis=0)
    return mv._outward(concat([panel(f, code(mat)) for f in faces]), centre)


def garden_walls(v, whole=False):
    """Garden walls and gates near the High Street (the user's choice of Hogsmeade's small structures), one piece per
    house. Before each detached house within GARDEN['near'] of the High Street's centreline, off it, that stands at
    least GARDEN['deep'] back from its road, a low rubble wall runs along the road GARDEN['back'] short of it, stepping
    with the ground, with a painted gate between stone piers before the house's door (in its front's middle bay, as
    mesh_village opens it where the way ahead is clear). From both its ends the wall returns to stop GARDEN['short']
    before the house's front corners, enclosing the garden. None stands where the road or a building comes near it.
    `whole`: in their stead a slab along each of its runs, gate and piers included, which the shadow caster stands in
    with (_garden)."""
    import mesh_village as mv

    line = v.streets.high
    out = []
    for cid in sorted(v.units):
        for u in v.units[cid]:
            if u.high or u.party or u.asset != mv.HOUSE or u.style.family == "outshot":
                continue
            a, b = plan.edges(u.outline)[u.front]
            f = u.fp.frame
            A, B = f.world(a), f.world(b)
            mid = (A + B) / 2
            if line.distance(Point(mid)) > GARDEN["near"]:
                continue
            n = f.world((a + b) / 2 + plan.outward(a, b)) - mid
            n /= np.linalg.norm(n)
            road = next((r for r in np.arange(0.5, GARDEN["reach"], 0.25) if v.streets.road_share([mid + n * r]) > 0.0),
                        None)
            if road is None or road < GARDEN["deep"]:
                continue
            length = float(np.linalg.norm(B - A))
            t = (B - A) / length
            centres, _ = mv.bays(length)
            gate = centres[len(centres) // 2]
            p0, p1 = A + n * (road - GARDEN["back"]), B + n * (road - GARDEN["back"])
            along = p0 + (p1 - p0) * np.linspace(0.0, 1.0, 9)[:, None]
            faces = np.concatenate([along + n * GARDEN["pier"] / 2, along - n * GARDEN["pier"] / 2])
            if v.streets.road_share(faces) > 0.0:
                continue
            returns = [(end + n * GARDEN["thick"] / 2, corner + n * GARDEN["short"]) for end, corner in ((p0, A), (p1, B))]
            if not all(_stands_clear(v, *run) for run in returns):
                continue
            g0, g1 = p0 + t * (gate - GARDEN["gate"] / 2), p0 + t * (gate + GARDEN["gate"] / 2)
            out.append(_garden(v.scene, p0, g0, g1, p1, t, whole, returns))
    return out


def _stands_clear(v, a, b):
    """Whether a run of garden wall from a to b (world x, z) stands off the road and GARDEN['short'] or more from every
    building."""
    d = (b - a) / np.linalg.norm(b - a)
    side = np.array([-d[1], d[0]]) * GARDEN["pier"] / 2
    along = a + (b - a) * np.linspace(0.0, 1.0, 9)[:, None]
    run = LineString([a, b]).buffer(GARDEN["thick"] / 2, cap_style=2)
    return (v.streets.road_share(np.concatenate([along + side, along - side])) == 0.0
            and min(o.distance(run) for o in v.outlines.values()) >= GARDEN["short"] - 1e-6)


def _garden(scene, p0, g0, g1, p1, t, whole=False, returns=()):
    """One garden's wall from p0 to p1 (world x, z), its gate between g0 and g1, and its returns, each a run (start,
    end) back from beside one of its ends: the wall in rubble lengths of at most GARDEN['step'] each stepping with the
    ground, a stone pier either side of the gate, and the painted gate between them. `whole`: in their stead one rubble
    slab along each run, the wall before the house and each return, from the lowest foot of any of the run's lengths
    to the highest top of any (and, before the house, of the wall over the gate's ground), so every length stands
    within one."""
    from mesh_geometry import concat

    ground = lambda *pts: np.asarray(scene.ground(np.array([p[0] for p in pts]), np.array([p[1] for p in pts])))

    def lengths(a, b):
        span = float(np.linalg.norm(b - a))
        count = math.ceil(span / GARDEN["step"]) if span >= 0.3 else 0
        out = []
        for i in range(count):
            q0, q1 = a + (b - a) * i / count, a + (b - a) * (i + 1) / count
            g = ground(q0, q1)
            out.append(_slab(q0, q1, GARDEN["thick"], float(g.min()) - GARDEN["foot"],
                             float(g.max()) + GARDEN["high"], "rubble"))
        return out

    g = ground((g0 + g1) / 2)
    runs = [(p0, p1, lengths(p0, g0) + lengths(g1, p1), [g - GARDEN["foot"], g + GARDEN["high"]])]
    runs += [(a, b, lengths(a, b), []) for a, b in returns]
    if whole:
        slabs = []
        for a, b, parts, extra in runs:
            y = np.concatenate([m.pos[:, 1] for m in parts] + extra)
            slabs.append(_slab(a, b, GARDEN["thick"], float(y.min()), float(y.max()), "rubble"))
        return concat(slabs)
    parts = [m for _, _, ms, _ in runs for m in ms]
    half = t * GARDEN["pier"] / 2
    for c in (g0, g1):
        g = float(ground(c)[0])
        parts.append(_slab(c - half, c + half, GARDEN["pier"], g - GARDEN["foot"], g + GARDEN["pier_h"], "pale-stone"))
    g = float(ground((g0 + g1) / 2)[0])
    parts.append(_slab(g0 + half, g1 - half, 0.06, g + 0.05, g + GARDEN["high"], "door-green"))
    return concat(parts)
