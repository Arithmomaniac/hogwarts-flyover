"""Greenhouse groups (mesh-greenhouses): span glasshouses on the mapped glass footprints and their potting sheds.

No greenhouse photograph exists on White Hound's pages, so the design is documented inference (greenhouse-design-inferred):
a dwarf stone wall, painted posts, rafters and ridge, tinted see-through panes, and benches of potted plants
(greenhouse-glass)."""
import zlib

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

import railway
from materials import code
from mesh_geometry import bar, concat, fit_rectangle, indexed, lathe, oriented, panel, pierced, prism
from mesh_source import field_polygons

FAMILY = "greenhouses"

# G-buffer codes from the material registry (src/materials): building kinds, hedge and glazing
FRAME, FOLIAGE, GLASS = code("building-glass"), code("hedge"), code("glazing")
# fittings take plain materials: building walls would gain painted windows and their upper faces slate
WALL_STONE, BENCH = code("wall-stone"), code("dark-timber")    # boundary-wall stone; set-piece dark timber
SHED_WALL, SHED_GLASS = code("shed-stone"), code("shed-glass")
THATCH, SLATE = code("building-thatch"), code("building-slate")
POT = code("chimney-pot")                   # the village's fired clay
DWARF_M, EAVE_M, PITCH = 0.7, 2.3, 0.62     # dwarf wall and glass eaves above the floor; roof rise per half-width
WALL_M = 0.25                               # dwarf wall thickness
BAY_M, BAR_M = 1.2, 0.08
BENCH_W, BENCH_H, BENCH_TOP_M, LEG_M, LEG_GAP_M = 0.9, 0.8, 0.05, 0.05, 2.4
PLANT_M, PLANT_ROW_M, CROWN_M = 0.45, 0.2, 0.38   # along the bench, each side of its middle; a crown's mean height
RAIL_M = 0.8                                # the glazing rail above the dwarf wall
SHED_EAVE_M, SHED_PITCH = 2.6, 0.9
SINK_M = 1.5
FLOOR_GAP_M = 0.03                          # a glasshouse's floor stands this far above the highest ground inside it
# the potting sheds' openings: windows of four panes, a plank door, each recessed REVEAL_M into the wall
WINDOW_W, WINDOW_H, WINDOW_SILL_M, WINDOW_GAP_M = 0.9, 1.0, 0.9, 2.4    # sill above the floor or the ground under it
DOOR_W, DOOR_H, DOOR_MIN_M = 1.0, 2.0, 1.7
REVEAL_M, HEAD_M, PIER_M, MULLION_M = 0.18, 0.3, 0.5, 0.06   # masonry over a head; beside a window at a wall's end
STEP_M, ROOF_OVERHANG_M, SAMPLE_M = 0.4, 0.3, 0.25
SHED_REACH_M = 40.0                         # a shed's door faces the glasshouses this near it


def outline(polygons):
    shape = unary_union(polygons)
    return max(getattr(shape, "geoms", [shape]), key=lambda p: p.area)


def clear_floor(inside, ground, step=0.25):
    """A floor level FLOOR_GAP_M above the highest ground inside `inside` (sampled every `step` m and at its corners),
    so the ground never shows through it nor meets it where it can be seen."""
    import shapely

    x0, z0, x1, z1 = inside.bounds
    x, z = np.meshgrid(np.arange(x0, x1 + step, step), np.arange(z0, z1 + step, step))
    keep = shapely.contains_xy(inside, x, z)
    corners = np.asarray(inside.exterior.coords)
    xs, zs = np.concatenate([x[keep], corners[:, 0]]), np.concatenate([z[keep], corners[:, 1]])
    return float(np.max(ground(xs, zs))) + FLOOR_GAP_M


def floor_level(rect, ground):
    cx, cz, length, width, a = rect
    b = np.array([-a[1], a[0]])
    xz = np.array([[cx, cz] + a * s * length / 2 + b * t * width / 2 for s in (-1, 1) for t in (-1, 1)])
    return float(np.median(ground(xz[:, 0], xz[:, 1])))


def greenhouse(polygons, ground, open_towards=None):
    """(opaque frame, benches and plants; glazing) for one mapped glass footprint. A wing given `open_towards` (x, z)
    runs its ridge towards that point, and its end there has no dwarf wall or end frame: it opens into the range it
    joins there (court-glasshouse), whose side glazing closes it below the eave; only its gable above is glazed."""
    rect = cx, cz, length, width, a = fit_rectangle(outline(polygons))
    if open_towards is not None:
        v = np.asarray(open_towards, np.float64) - [cx, cz]
        if abs(v @ [-a[1], a[0]]) > abs(v @ a):
            a, length, width = np.array([-a[1], a[0]]), width, length
        a = a if v @ a > 0 else -a
        rect = cx, cz, length, width, a
    b = np.array([-a[1], a[0]])
    hl, hw = length / 2, width / 2
    at = lambda s, t, h: np.array([cx + a[0] * s + b[0] * t, h, cz + a[1] * s + b[1] * t])
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    plan = lambda s0, s1, t0, t1: orient(Polygon([(p[0], p[2]) for p in (at(s0, t0, 0), at(s1, t0, 0), at(s1, t1, 0),
                                                                         at(s0, t1, 0))]), 1.0)
    shut = (-hl, hl) if open_towards is None else (-hl,)    # the ends that are walled and glazed
    footprint = plan(-hl, hl, -hw, hw)
    inside = plan(-hl + WALL_M, hl - WALL_M if hl in shut else hl, -hw + WALL_M, hw - WALL_M)
    y = clear_floor(inside, ground)                 # seen through the glass: the floor, never the lawn under it
    rise = hw * PITCH
    eave, ridge, sill = y + EAVE_M, y + EAVE_M + rise, y + DWARF_M
    below = lambda top: lambda x, z: np.minimum(ground(x, z) - SINK_M, top)   # sunk, but never above its own top
    opaque = [prism(footprint.difference(inside), below(sill), sill, WALL_STONE), prism(inside, below(y), y, WALL_STONE)]   # wall ring, floor
    ss = np.linspace(-hl, hl, max(2, int(round(length / BAY_M)) + 1))
    for t in (-hw, hw):
        opaque += [bar(at(s, t, sill), at(s, t, eave), BAR_M, FRAME) for s in ss]
        opaque += [bar(at(s, t, eave), at(s, 0.0, ridge), BAR_M, FRAME) for s in ss]
        opaque.append(bar(at(-hl, t, eave), at(hl, t, eave), BAR_M, FRAME))
    for s in shut:
        opaque += [bar(at(s, t, sill), at(s, t, eave), BAR_M, FRAME) for t in np.linspace(-hw, hw, 3)]
        opaque.append(bar(at(s, 0.0, sill), at(s, 0.0, ridge), BAR_M, FRAME))
    opaque.append(bar(at(-hl, 0.0, ridge), at(hl, 0.0, ridge), BAR_M * 1.5, FRAME))
    for t in (-hw, hw):                            # the sill plate, a glazing rail and a purlin halfway up each slope
        opaque += [bar(at(-hl, t, sill), at(hl, t, sill), BAR_M, FRAME), bar(at(-hl, t, sill + RAIL_M), at(hl, t, sill + RAIL_M), BAR_M, FRAME),
                   bar(at(-hl, t / 2, (eave + ridge) / 2), at(hl, t / 2, (eave + ridge) / 2), BAR_M, FRAME)]
    opaque += [bar(at(s, -hw, sill + RAIL_M), at(s, hw, sill + RAIL_M), BAR_M, FRAME) for s in shut]
    for t in (-1, 1):                              # benches on legs along both long walls, potted plants zigzag on each
        inner, outer = t * (hw - WALL_M - 0.05 - BENCH_W), t * (hw - WALL_M - 0.05)
        top = y + BENCH_H
        opaque.append(prism(plan(-hl + 0.5, hl - 0.5, inner, outer), top - BENCH_TOP_M, top, BENCH))
        for s in np.linspace(-hl + 0.6, hl - 0.6, max(2, int(round((length - 1.2) / LEG_GAP_M)) + 1)):
            opaque += [bar(at(s, e, y), at(s, e, top - BENCH_TOP_M), LEG_M, BENCH)
                       for e in (inner + t * 0.06, outer - t * 0.06)]
        for k, s in enumerate(np.arange(-hl + 0.8, hl - 0.7, PLANT_M)):
            p = at(s, (inner + outer) / 2 + PLANT_ROW_M * (-1) ** k, top)
            h = CROWN_M * (0.7 + 0.6 * (zlib.crc32(np.round(p, 1).tobytes()) % 1000) / 1000)   # plants of all sizes
            pot = top + 0.16
            opaque.append(lathe(p[0], p[2], [(0.08, top), (0.12, pot)], POT, 5, cap=False))
            opaque.append(lathe(p[0], p[2], [(0.0, pot - 0.02), (0.16, pot + 0.3 * h), (0.13, pot + 0.75 * h), (0.0, pot + h)],
                                FOLIAGE, 5, cap=False))
    glass = [panel([at(-hl, t, sill), at(hl, t, sill), at(hl, t, eave), at(-hl, t, eave)], GLASS) for t in (-hw, hw)]
    glass += [panel([at(s, -hw, sill), at(s, hw, sill), at(s, hw, eave), at(s, 0.0, ridge), at(s, -hw, eave)], GLASS)
              for s in shut]
    if open_towards is not None:                   # the gable above the range's eave, which its side glass can't close
        glass.append(panel([at(hl, -hw, eave), at(hl, hw, eave), at(hl, 0.0, ridge)], GLASS))
    glass += [panel([at(-hl, t, eave), at(hl, t, eave), at(hl, 0.0, ridge), at(-hl, 0.0, ridge)], GLASS)
              for t in (-hw, hw)]
    return concat(opaque), concat(glass)


def slots(u0, u1):
    """Window centres spread evenly along a stretch of wall from u0 to u1, none when it is too short for one."""
    span = u1 - u0
    if span < WINDOW_W + 2 * PIER_M:
        return []
    n = max(1, int((span - 2 * PIER_M) // WINDOW_GAP_M))
    return [u0 + span * (k + 0.5) / n for k in range(n)]


def shed(polygons, ground, roof=SLATE, towards=None):
    """A stone potting shed under a pitched roof on the footprint's rectangle: a plank door in the long wall facing
    `towards` (x, z; the first long wall when None), and windows recessed into its walls. Building walls would take
    the shading pass's painted windows, which the eaves cut to slits wherever the ground falls (potting-shed-1); the
    shed's walls are plain stone and its openings real, each sill above the ground under it and each head below the
    eaves."""
    rect = cx, cz, length, width, a = fit_rectangle(outline(polygons))
    b = np.array([-a[1], a[0]])
    c = np.array([cx, cz])
    hl, hw = length / 2, width / 2
    floor = floor_level(rect, ground)
    eave, rise = floor + SHED_EAVE_M, hw * SHED_PITCH
    up = np.array([0.0, 1.0, 0.0])
    door_side = 1.0 if towards is None or np.dot(np.asarray(towards) - c, b) >= 0 else -1.0
    parts = []
    for out, centre, span, gable in ((b, b * hw, length, False), (-b, -b * hw, length, False),
                                     (a, a * hl, width, True), (-a, -a * hl, width, True)):
        n3 = np.array([out[0], 0.0, out[1]])
        u3 = np.array([out[1], 0.0, -out[0]])                     # along the wall, so that u x up faces out
        start = c + centre - u3[[0, 2]] * span / 2
        along = lambda u: start + np.outer(np.atleast_1d(u), u3[[0, 2]])
        under = lambda u0, u1: float(np.max(ground(*along(np.linspace(u0, u1, 9)).T)))
        base = min(float(np.min(ground(*along(np.arange(0.0, span + SAMPLE_M, SAMPLE_M)).T))) - SINK_M, eave)
        holes, fills = [], []
        door = not gable and np.dot(out, b) * door_side > 0
        runs = [(0.0, span / 2 - DOOR_W / 2), (span / 2 + DOOR_W / 2, span)] if door else [(0.0, span)]
        if door:
            u0, u1 = span / 2 - DOOR_W / 2, span / 2 + DOOR_W / 2
            sill = max(floor, under(u0, u1))
            top = min(sill + DOOR_H, eave - HEAD_M)
            if top - sill >= DOOR_MIN_M:
                holes.append([(u0, sill), (u1, sill), (u1, top), (u0, top)])
                fills.append(BENCH)
                if sill - under(u0 - 0.2, u1 + 0.2) > 0.05:      # a step up to a door above the ground
                    s0, s1 = start + u3[[0, 2]] * (u0 - 0.2), start + u3[[0, 2]] * (u1 + 0.2)
                    step = Polygon([s0, s1, s1 + out * STEP_M, s0 + out * STEP_M])
                    parts.append(prism(step, lambda x, z: ground(x, z) - SINK_M, sill, SHED_WALL))
        for u0, u1 in runs:
            for u in slots(u0, u1):
                w0, w1 = u - WINDOW_W / 2, u + WINDOW_W / 2
                sill = max(floor, under(w0, w1)) + WINDOW_SILL_M
                if sill + WINDOW_H > eave - HEAD_M:
                    continue
                holes.append([(w0, sill), (w1, sill), (w1, sill + WINDOW_H), (w0, sill + WINDOW_H)])
                fills.append(SHED_GLASS)
                p = lambda uu, y, d: np.array([start[0] + u3[0] * uu, y, start[1] + u3[2] * uu]) + n3 * d
                parts += [bar(p(u, sill, -REVEAL_M + 0.04), p(u, sill + WINDOW_H, -REVEAL_M + 0.04), MULLION_M, BENCH),
                          bar(p(w0, sill + WINDOW_H / 2, -REVEAL_M + 0.04), p(w1, sill + WINDOW_H / 2, -REVEAL_M + 0.04),
                              MULLION_M, BENCH)]
                s0, s1 = start + u3[[0, 2]] * (w0 - 0.08), start + u3[[0, 2]] * (w1 + 0.08)
                parts.append(prism(Polygon([s0, s1, s1 + out * 0.1, s0 + out * 0.1]), sill - 0.07, sill, SHED_WALL))
        top = [(span, eave), (span / 2, eave + rise), (0.0, eave)] if gable else [(span, eave), (0.0, eave)]
        origin = np.array([start[0], 0.0, start[1]])
        parts.append(pierced(origin, u3, up, [(0.0, base), (span, base), *top], holes, REVEAL_M, SHED_WALL, fills))
    slope, e = rise / hw, eave - ROOF_OVERHANG_M * rise / hw
    at = lambda s, t, y: np.array([cx + a[0] * s + b[0] * t, y, cz + a[1] * s + b[1] * t])
    for t in (-1.0, 1.0):
        n = np.array([t * b[0] * slope, 1.0, t * b[1] * slope])
        n /= np.linalg.norm(n)
        q = panel([at(-hl - ROOF_OVERHANG_M, t * (hw + ROOF_OVERHANG_M), e), at(hl + ROOF_OVERHANG_M, t * (hw + ROOF_OVERHANG_M), e),
                   at(hl + ROOF_OVERHANG_M, 0.0, eave + rise), at(-hl - ROOF_OVERHANG_M, 0.0, eave + rise)], roof)
        parts.append(indexed(oriented(q.pos[q.idx], n), n, roof))
    return concat(parts)


def glass_near(scene, polygons, reach=SHED_REACH_M):
    """The middle of the mapped glasshouses within `reach` of a shed, which its door faces; None when there are none."""
    c = np.asarray(outline(polygons).centroid.coords[0])
    near = [(g["x"], g["z"]) for g in scene.labelled[1] if g["kind"] == "glass" and np.hypot(g["x"] - c[0], g["z"] - c[1]) <= reach]
    return tuple(np.mean(near, axis=0)) if near else None


# ---------------------------------------------------------------- family hooks (build_meshes.massing)
def build_footprint(scene, asset, c, polygons):
    """A glass footprint as a greenhouse, any other as a potting shed under thatch or slate."""
    if c["kind"] == "glass":
        t = next((t["plan_px"] for t in scene.assets[asset].get("traced", []) if isinstance(t["plan_px"], dict)), None)
        if t is None:
            return {asset: greenhouse(polygons, scene.ground)}
        rng, wing = (Polygon(railway.grounds_world(np.asarray(t[k], np.float64), scene.georef)) for k in ("range", "wing"))
        c = rng.centroid                            # the T White Hound's map draws: a range, and a wing opening into it
        return {asset: [*greenhouse([rng], scene.ground), *greenhouse([wing], scene.ground, (c.x, c.y))]}
    return {asset: [shed(polygons, scene.ground, THATCH if c["kind"] == "thatch" else SLATE, glass_near(scene, polygons))]}


def build_castle_pieces(scene, asset, keep):
    """Round castle-grid masses re-read as potting sheds (castle-mass-sheds), one shed per piece."""
    image, classes = scene.castle_labelled[0], len(scene.meta["castle"]["cls"])
    parts = []
    for piece in sorted(p for p, a in scene.piece_owners.items() if a == asset):
        one = image == piece
        polygons = [poly for k in range(classes) for poly in field_polygons(scene.csd[..., k], scene.castle, 0.0, one)]
        parts.append(shed(polygons, scene.ground, towards=glass_near(scene, polygons)))
    return {asset: parts}
