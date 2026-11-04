"""Greenhouse groups (mesh-greenhouses): span glasshouses on the mapped glass footprints and their potting sheds.

No greenhouse photograph exists on White Hound's pages, so the design is documented inference (greenhouse-design-inferred):
a dwarf stone wall, painted posts, rafters and ridge, tinted see-through panes, and benches of potted plants
(greenhouse-glass)."""
import numpy as np
from shapely.ops import unary_union

from materials import code
from mesh_geometry import bar, concat, fit_rectangle, gabled, lathe, panel, prism
from mesh_source import field_polygons

FAMILY = "greenhouses"

# G-buffer codes from the material registry (src/materials): building kinds, hedge and glazing
STONE, FRAME, FOLIAGE, GLASS = code("building-big"), code("building-glass"), code("hedge"), code("glazing")
# fittings take plain materials: building walls would gain painted windows and their upper faces slate
WALL_STONE, BENCH = code("wall-stone"), code("dark-timber")    # boundary-wall stone; set-piece dark timber
THATCH, SLATE = code("building-thatch"), code("building-slate")
DWARF_M, EAVE_M, PITCH = 0.7, 2.3, 0.62     # dwarf wall and glass eaves above the floor; roof rise per half-width
WALL_M = 0.25                               # dwarf wall thickness
BAY_M, BAR_M = 1.2, 0.08
BENCH_W, BENCH_H, PLANT_M = 0.9, 0.8, 0.55
SHED_EAVE_M, SHED_PITCH = 2.6, 0.9
SINK_M = 1.5


def outline(polygons):
    shape = unary_union(polygons)
    return max(getattr(shape, "geoms", [shape]), key=lambda p: p.area)


def floor_level(rect, ground):
    cx, cz, length, width, a = rect
    b = np.array([-a[1], a[0]])
    xz = np.array([[cx, cz] + a * s * length / 2 + b * t * width / 2 for s in (-1, 1) for t in (-1, 1)])
    return float(np.median(ground(xz[:, 0], xz[:, 1])))


def greenhouse(polygons, ground):
    """(opaque frame, benches and plants; glazing) for one mapped glass footprint."""
    rect = cx, cz, length, width, a = fit_rectangle(outline(polygons))
    b = np.array([-a[1], a[0]])
    y = floor_level(rect, ground)
    hl, hw, rise = length / 2, width / 2, width / 2 * PITCH
    at = lambda s, t, h: np.array([cx + a[0] * s + b[0] * t, h, cz + a[1] * s + b[1] * t])
    eave, ridge, sill = y + EAVE_M, y + EAVE_M + rise, y + DWARF_M
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    plan = lambda s0, s1, t0, t1: orient(Polygon([(p[0], p[2]) for p in (at(s0, t0, 0), at(s1, t0, 0), at(s1, t1, 0),
                                                                         at(s0, t1, 0))]), 1.0)
    footprint, inside = plan(-hl, hl, -hw, hw), plan(-hl + WALL_M, hl - WALL_M, -hw + WALL_M, hw - WALL_M)
    below = lambda top: lambda x, z: np.minimum(ground(x, z) - SINK_M, top)   # sunk, but never above its own top
    opaque = [prism(footprint.difference(inside), below(sill), sill, WALL_STONE), prism(inside, below(y), y, WALL_STONE)]   # wall ring, floor
    ss = np.linspace(-hl, hl, max(2, int(round(length / BAY_M)) + 1))
    for t in (-hw, hw):
        opaque += [bar(at(s, t, sill), at(s, t, eave), BAR_M, FRAME) for s in ss]
        opaque += [bar(at(s, t, eave), at(s, 0.0, ridge), BAR_M, FRAME) for s in ss]
        opaque.append(bar(at(-hl, t, eave), at(hl, t, eave), BAR_M, FRAME))
    for s in (-hl, hl):
        opaque += [bar(at(s, t, sill), at(s, t, eave), BAR_M, FRAME) for t in np.linspace(-hw, hw, 3)]
        opaque.append(bar(at(s, 0.0, sill), at(s, 0.0, ridge), BAR_M, FRAME))
    opaque.append(bar(at(-hl, 0.0, ridge), at(hl, 0.0, ridge), BAR_M * 1.5, FRAME))
    for t in (-1, 1):                              # benches along both long walls, a row of potted plants on each
        inner, outer = t * (hw - WALL_M - 0.05 - BENCH_W), t * (hw - WALL_M - 0.05)
        bench = plan(-hl + 0.5, hl - 0.5, inner, outer)
        opaque.append(prism(bench, y, y + BENCH_H, BENCH))
        top = y + BENCH_H
        for s in np.arange(-hl + 0.8, hl - 0.7, PLANT_M):
            p = at(s, (inner + outer) / 2, top)
            opaque.append(lathe(p[0], p[2], [(0.0, top), (0.22, top + 0.12), (0.27, top + 0.3), (0.15, top + 0.46),
                                             (0.0, top + 0.52)], FOLIAGE, 5, cap=False))
    glass = [panel([at(-hl, t, sill), at(hl, t, sill), at(hl, t, eave), at(-hl, t, eave)], GLASS) for t in (-hw, hw)]
    glass += [panel([at(s, -hw, sill), at(s, hw, sill), at(s, hw, eave), at(s, 0.0, ridge), at(s, -hw, eave)], GLASS)
              for s in (-hl, hl)]
    glass += [panel([at(-hl, t, eave), at(hl, t, eave), at(hl, 0.0, ridge), at(-hl, 0.0, ridge)], GLASS)
              for t in (-hw, hw)]
    return concat(opaque), concat(glass)


def shed(polygons, ground, roof=SLATE):
    """A low stone potting shed with a pitched roof on the footprint's rectangle."""
    rect = fit_rectangle(outline(polygons))
    eave = floor_level(rect, ground) + SHED_EAVE_M
    sunk = lambda x, z: np.minimum(ground(x, z) - SINK_M, eave)   # sunk, but never above the eaves
    return gabled(rect, sunk, eave, rect[3] / 2 * SHED_PITCH, STONE, roof)


# ---------------------------------------------------------------- family hooks (build_meshes.massing)
def build_footprint(scene, asset, c, polygons):
    """A glass footprint as a greenhouse, any other as a potting shed under thatch or slate."""
    if c["kind"] == "glass":
        return {asset: greenhouse(polygons, scene.ground)}
    return {asset: [shed(polygons, scene.ground, THATCH if c["kind"] == "thatch" else SLATE)]}


def build_castle_pieces(scene, asset, keep):
    """Round castle-grid masses re-read as potting sheds (castle-mass-sheds), one shed per piece."""
    image, classes = scene.castle_labelled[0], len(scene.meta["castle"]["cls"])
    parts = []
    for piece in sorted(p for p, a in scene.piece_owners.items() if a == asset):
        one = image == piece
        parts.append(shed([poly for k in range(classes) for poly in field_polygons(scene.csd[..., k], scene.castle, 0.0, one)],
                          scene.ground))
    return {asset: parts}
