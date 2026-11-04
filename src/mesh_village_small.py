"""Hogsmeade's small structures, standing on the sampled ground: the High Street's cast-iron lamps, the kirkyard's
wall and headstones, and garden walls and gates near the High Street."""
import math

import numpy as np
import shapely
from shapely.geometry import Point
from shapely.geometry.polygon import orient

import mesh_village_plan as plan

LAMP = {"every": 30.0, "out": 1.6, "end": 0.3, "clear": 0.6,
        "plinth": 0.46, "plinth_top": 0.6, "foot": 0.3, "post": (0.16, 0.08), "post_top": 3.4,
        "lantern": 0.34, "lantern_h": 0.48, "cap": 0.44, "cap_h": 0.3}
KIRKYARD = {"out": 8.0, "piece": 4.0, "thick": 0.5, "high": 1.1, "foot": 0.3, "clear": 1.0}
HEADSTONE = {"wide": 0.6, "thick": 0.12, "high": (0.7, 1.1), "foot": 0.2, "rows": 6, "every": 2.0,
             "out": (2.5, 4.5)}         # their rows' distances out from the nave's long wall
GARDEN = {"near": 60.0, "reach": 15.0, "deep": 3.0, "back": 0.8, "thick": 0.4, "high": 1.0, "foot": 0.3,
          "step": 1.5, "gate": 1.4, "pier": 0.45, "pier_h": 1.3}


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
    spots = []                                      # (along the street, side, spot)
    for a, b, n in high_street_fronts(v):
        for p, q in ((a, b), (b, a)):
            spot = p + (q - p) / np.linalg.norm(q - p) * LAMP["end"] + n * LAMP["out"]
            spots.append((float((spot - start) @ axis), n @ left > 0.0, spot))
    out = []
    for k, target in enumerate(np.arange(LAMP["every"] / 2, float((end - start) @ axis), LAMP["every"])):
        for side in ((True, False) if k % 2 == 0 else (False, True)):
            near = sorted(((abs(s - target), i) for i, (s, sd, _) in enumerate(spots)
                           if sd == side and abs(s - target) < LAMP["every"] / 2))
            pick = next((spots[i][2] for _, i in near if v.clear_of_buildings(spots[i][2], LAMP["clear"])), None)
            if pick is not None:
                out.append(_lamp(v.scene, pick))
                break
    return out


def _lamp(scene, p):
    """A lamp at world point p: a stone plinth sunk LAMP['foot'] into the ground, a tapering cast-iron post, a glass
    lantern and an iron cap."""
    from mesh_geometry import concat

    g = float(scene.ground(*p))
    base, post, lantern = g + LAMP["plinth_top"], g + LAMP["post_top"], g + LAMP["post_top"] + LAMP["lantern_h"]
    return concat([_frustum(p, LAMP["plinth"], LAMP["plinth"], g - LAMP["foot"], base, "pale-stone", top=True),
                   _frustum(p, *LAMP["post"], base, post, "black-iron"),
                   _frustum(p, LAMP["lantern"], LAMP["lantern"], post, lantern, "lamp-glass", bottom=True),
                   _frustum(p, LAMP["cap"], 0.0, lantern, lantern + LAMP["cap_h"], "black-iron", bottom=True)])


def _frustum(p, w0, w1, y0, y1, mat, top=False, bottom=False):
    """A square frustum centred on world point p, w0 wide at height y0 and w1 at y1 (0: a pyramid), with its top and
    bottom faces if asked, every face turned outward."""
    import mesh_village as mv
    from materials import code
    from mesh_geometry import concat, panel

    ring = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    lo = [np.array([p[0] + x * w0 / 2, y0, p[1] + z * w0 / 2]) for x, z in ring]
    hi = [np.array([p[0] + x * w1 / 2, y1, p[1] + z * w1 / 2]) for x, z in ring]
    faces = [[lo[i], lo[(i + 1) % 4], hi[(i + 1) % 4]] + ([hi[i]] if w1 > 0 else []) for i in range(4)]
    if top and w1 > 0:
        faces.append(hi)
    if bottom:
        faces.append(lo)
    return mv._outward(concat([panel(f, code(mat)) for f in faces]), np.array([p[0], (y0 + y1) / 2, p[1]]))


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
    """Rows of headstones on the nave's long side away from its porch, their faces to the nave's ends, each a slab of
    seeded height clear of the buildings and inside the kirkyard."""
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
        s = length / 2 + (row - (HEADSTONE["rows"] - 1) / 2) * HEADSTONE["every"]
        for d in HEADSTONE["out"]:
            p = f.world(a + t * s + n * d)
            high = float(rng.uniform(*HEADSTONE["high"]))
            if not v.clear_of_buildings(p, 1.0) or not inside.contains(Point(p)):
                continue
            g = float(v.scene.ground(*p))
            across = f.world(a + t * s + n * d + n * 0.5) - p              # its width runs out from the wall, so its
            side = across / np.linalg.norm(across) * HEADSTONE["wide"] / 2  # faces look along the nave to its ends
            out.append(_slab(p - side, p + side, HEADSTONE["thick"], g - HEADSTONE["foot"], g + high, "pale-stone"))
    return out


def _slab(a, b, thick, y0, y1, mat):
    """A box along world segment a-b (x, z), `thick` across it, from height y0 to y1, without its bottom face."""
    import mesh_village as mv
    from materials import code
    from mesh_geometry import concat, panel

    d = (b - a) / np.linalg.norm(b - a)
    w = np.array([-d[1], d[0]]) * thick / 2
    ring = [a - w, b - w, b + w, a + w]
    lo = [np.array([p[0], y0, p[1]]) for p in ring]
    hi = [np.array([p[0], y1, p[1]]) for p in ring]
    faces = [hi] + [[lo[i], lo[(i + 1) % 4], hi[(i + 1) % 4], hi[i]] for i in range(4)]
    centre = np.array([(a[0] + b[0]) / 2, (y0 + y1) / 2, (a[1] + b[1]) / 2])
    return mv._outward(concat([panel(f, code(mat)) for f in faces]), centre)


def garden_walls(v, whole=False):
    """Garden walls and gates near the High Street (the user's choice of Hogsmeade's small structures), one piece per
    house. Before each detached house within GARDEN['near'] of the High Street's centreline, off it, that stands at
    least GARDEN['deep'] back from its road, a low rubble wall runs along the road GARDEN['back'] short of it, stepping
    with the ground, with a painted gate between stone piers before the house's door (in its front's middle bay, as
    mesh_village opens it where the way ahead is clear). None stands where the road comes near it. `whole`: each as one
    slab along its front, gate and piers included, which the shadow caster stands in with (_garden)."""
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
            g0, g1 = p0 + t * (gate - GARDEN["gate"] / 2), p0 + t * (gate + GARDEN["gate"] / 2)
            out.append(_garden(v.scene, p0, g0, g1, p1, t, whole))
    return out


def _garden(scene, p0, g0, g1, p1, t, whole=False):
    """One garden's wall from p0 to p1 (world x, z), its gate between g0 and g1: the wall in rubble lengths of at most
    GARDEN['step'] each stepping with the ground, a stone pier either side of the gate, and the painted gate between
    them. `whole`: one rubble slab from p0 to p1 in their stead, from the lowest foot of any length to the highest top
    of any (or of the wall over the gate's ground), so every length stands within it."""
    from mesh_geometry import concat

    ground = lambda *pts: np.asarray(scene.ground(np.array([p[0] for p in pts]), np.array([p[1] for p in pts])))
    parts = []
    for a, b in ((p0, g0), (g1, p1)):
        span = float(np.linalg.norm(b - a))
        if span < 0.3:
            continue
        n = math.ceil(span / GARDEN["step"])
        for i in range(n):
            q0, q1 = a + (b - a) * i / n, a + (b - a) * (i + 1) / n
            g = ground(q0, q1)
            parts.append(_slab(q0, q1, GARDEN["thick"], float(g.min()) - GARDEN["foot"],
                               float(g.max()) + GARDEN["high"], "rubble"))
    if whole:
        g = ground((g0 + g1) / 2)
        y = np.concatenate([m.pos[:, 1] for m in parts] + [g - GARDEN["foot"], g + GARDEN["high"]])
        return _slab(p0, p1, GARDEN["thick"], float(y.min()), float(y.max()), "rubble")
    half = t * GARDEN["pier"] / 2
    for c in (g0, g1):
        g = float(ground(c)[0])
        parts.append(_slab(c - half, c + half, GARDEN["pier"], g - GARDEN["foot"], g + GARDEN["pier_h"], "pale-stone"))
    g = float(ground((g0 + g1) / 2)[0])
    parts.append(_slab(g0 + half, g1 - half, 0.06, g + 0.05, g + GARDEN["high"], "door-green"))
    return concat(parts)
