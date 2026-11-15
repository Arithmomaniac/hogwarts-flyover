"""The detailed castle exterior (mesh-castle): wings and courts from the registered mass classes with hip roofs or
battlements, deep round-arched windows (castle-norman-openings), and the named towers re-ranked to White Hound's
skyline (Astronomy highest, Ravenclaw and Gryffindor next).

Heights follow castleD: class k stands `CAS_CLS[k].x` above the castle base and roof types 1/3 rise with the distance
in from the eaves; towers come from meta.towers less the two masses the manifest reads as potting sheds. The raster
outlines are straightened into long planar walls that meet without gaps or slivers: each lower mass stops just inside the
taller masonry it abuts, where its hidden walls are left out and its roof stands up as a gable. Hip roofs follow a
straight skeleton, and walls take storeys of windows wherever they look out into open air."""
import math
import zlib
from typing import NamedTuple

import numpy as np
import shapely
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient

from materials import code
import mesh_grounds
from mesh_geometry import (Mesh, bar, circle, concat, crenellate, faceted, fit_rectangle, heights, lathe, lid, panel,
                           prism, roof)
from mesh_openings import EDGE_M, Opening, facade, tube
from mesh_openings import outline as arch
from mesh_source import CASTLE_CLASSES, field_polygons, sample

FAMILY = "castle"

ROOFS = {1: (5.5, 0.95), 3: (8.5, 1.35)}   # massD: (largest rise m, slope) per roof type
EAVE_M = 0.3                               # massD's roofs start this far up their slope
FLAT, LOW = 0, 4
PARAPET, MERLON = 1.0, 0.8
CROWN = PARAPET + MERLON                   # battlements above a flat top
CONE, ASTRONOMY, NORTH, GRYFFINDOR = 1, 3, 4, 5   # meta.towers roof types
SKYLINE_CLEARANCE_M = 6.0                  # Astronomy clears everything else in the castle by this much
RANK_CLEARANCE_M = 2.0                     # Ravenclaw and Gryffindor clear every other tower, and their drums the main buildings
MESH_TOWER_AUX = code("castle-mesh-tower")[1]          # tower i is aux 150 + i; its re-ranked top reaches P3 in ctop
CASTLE_CLASS_AUX = code("castle-class-painted")[1]     # class k as the raymarched castle tags it (aux 10 + k): painted windows
REAL_AUX = code("castle-class-openings")[1]            # class k with real openings (aux 20 + k): P3 paints none
# opening parts: a pane, a pane lit at dusk, oak doors, the masonry lining
GLASS, GLASS_LIT, OAK, REVEAL = (code(n)[1] for n in ("castle-glass", "castle-glass-lit", "castle-oak", "castle-reveal"))
GLASS_DIM = code("castle-glass-dim")[1]               # a pane lit dimly at dusk: the room's candle further from the window
GLASS_PANES = (GLASS, GLASS_LIT, GLASS_DIM)
IRON = code("castle-iron")[1]                          # the oak doors' straps and the joints of their boards
PLAIN = code("castle-plain")[1]                        # exposed plain masonry (steps): P3 paints no windows or slate on it
LIT_PERCENT, DIM_PERCENT = 40, 15          # panes lit at dusk, as P3 lights its painted windows; of them, this many dimly
TURRET_M = 15.0                            # Gryffindor's trefoil turrets above its drum (towerD)
COLUMN, OVERHANG, NECK = 0.62, 1.3, 4.0    # Astronomy/North: column and top radii (x r), corbel height (m)
DROP_M = 8.0                               # depth of the overhanging top
TURRET_SPACING_M = 45.0                    # at most one corbelled corner turret per this much block perimeter
TURRET_R, TURRET_CAP = 1.6, 2.6            # small Scottish turret radius and restrained cap height
TURRET_FOOT_M = 9.0                        # corbelled turrets hang this far below the block top
STRAIGHTEN_M = 1.0                         # traced outlines straightened within this (the map's walls bow by ~0.5 m)
ROOF_PIECE_M2 = 4000.0                     # flat tops are cut into pieces this large, well inside one chunk's vertices
PARTY_M = 0.1                              # a lower mass runs this far into the taller mass or drum it abuts, out of sight
SLIVER_M = 0.6                             # what a cut leaves narrower than twice this beside the taller mass is dropped
NT_CLEAR_M = 1.2                           # the North Tower's overhanging top clears the masses round it by this much
HOLE_M2 = 4.0                              # smaller holes in a mass are raster noise, filled
GAP_HELD = 0.15                            # straightening's gaps that two masses each hold this much of the rim of are
                                           # filled
STRAY_M2, STRAY_M = 30.0, 1.0             # a smaller mass (but a low wall) nowhere twice STRAY_M wide, meeting at most
                                           # one other, is a raster stray
ALONG = 0.3                                # a sliver lies along the taller wall for this much of its rim
CLING = 0.25                               # a small roofed mass along a taller wall for this much of its rim is fringe
NEEDLE = 2e-3                              # a triangle with less area than this times its longest edge squared is a needle
SKELETON_M2 = 30.0                         # smaller roofs take the heightfield, which CGAL's skeleton cannot always roof
END_M = 1.2                                # masonry kept clear at each end of a wall
WING_M2 = 50.0                             # smaller masses are raster slivers between wings and take no windows
CLEAR_M = 2.9                              # a window needs open air this far out from its wall,
HEADROOM_M = 1.0                           # with any roof, battlement, tower or turret there this far below its sill
NEAR_M = 0.6                               # passing this close to another mass counts as meeting it: half a window's width,
                                           # so merlons and corners beside its centre line still block it
FACET_M = 1.9                              # tower drums are faceted about this wide


class Fenestration(NamedTuple):
    width: float
    height: float                          # foot to the crown of the arch
    depth: float                           # of the reveal
    spacing: float                         # centre to centre along a wall
    storey: float                          # sill to sill; 0 for a single row
    first: float                           # lowest sill above the castle base


# inferred (castle-norman-openings): about 4 m storeys in the 8-storey, 32 m blocks; the Great Hall's tall windows sit
# high above the courts; the keep's are smaller and further apart
WINDOWS = {"range": Fenestration(1.0, 2.2, 0.7, 3.2, 4.0, 1.0), "block": Fenestration(1.0, 2.2, 0.7, 3.2, 4.0, 1.0),
           "great_hall": Fenestration(1.6, 6.5, 0.9, 5.0, 0.0, 9.0), "keep": Fenestration(0.9, 1.9, 0.9, 4.0, 4.5, 1.0)}
TOWER_WINDOW = Fenestration(0.9, 1.9, 0.6, 0.0, 4.0, 1.0)   # columns on every 2nd or 3rd facet, storeys staggered
SLIT_M = 0.2                                # a slender drum's facets take windows down to this narrow (slits)
TURRET_WINDOW = Fenestration(0.7, 1.6, 0.4, 0.0, 4.0, 2.5)  # Gryffindor's turrets: on alternate facets facing out, over
                                                            # the drum's battlements
TURRET_FACETS = 16
BOARD_M, STRAP_M = 0.32, 0.07              # the main doors' boards and their iron straps
# the main doors (castle-norman-openings): a deep doorway stepping in by two orders to oak double doors, its threshold
# raised for "a flight of stone steps"
DOOR = Opening(u=0.0, sill=0.0, width=4.4, height=6.6, depth=0.4, fill=(4, OAK), orders=((0.35, 0.35), (0.35, 0.45)))
THRESHOLD_M = 1.2                          # above the castle base
RISE_M, TREAD_M, LANDING_M = 0.17, 0.38, 1.6
STAIR_MARGIN_M = 1.0                       # the flight is this much wider than the doorway on each side
# the pend into the broomshed court (inferred size: a cart's width and a round arch), gated at the court end
PEND_WIDTH_M, PEND_HEIGHT_M = 4.0, 5.0
GATE_M = PEND_HEIGHT_M - PEND_WIDTH_M / 2 - 0.1   # gate leaves stop under the springing of the arch
# the Head's tower (inferred: round, as "a beautiful circular room" and Pomfrey's office "a slice through the tower")
HEAD_R, HEAD_RISE_M = 5.5, 40.0            # its top above the castle base: over the hospital wing, under the named towers
STAIR_TURRET_R = 1.7
HEAD_DOOR = Opening(u=0.0, sill=0.0, width=1.2, height=2.6, depth=0.6, fill=(4, OAK))
HEAD_THRESHOLD_M = 1.0                     # its garden door above the ground outside
GARDEN_DEPTH_M, GARDEN_WIDTH_M = 12.0, 14.0   # the Physick Garden (inferred size)
GARDEN_WALL_M = 1.6
WALL_STONE = code("wall-stone")            # the grounds' boundary-wall stone
HERBS = code("hedge")                      # planted beds, as the hedges
BED_L, BED_W, BED_H, PATH_M = 2.4, 1.2, 0.35, 0.8
# court-buildings-as-houses: court pieces that are no tower's base are domestic houses (inferred size and pitch)
HOUSE_MAX_M2 = 300.0
HOUSE_EAVE_M, HOUSE_PITCH = 7.0, 1.0       # two storeys; steep roofs, rising a metre per metre of half-width
HOUSE_WINDOW = Fenestration(0.8, 1.4, 0.3, 2.6, 3.4, 1.0)   # domestic windows: two storeys under the eaves
HOUSE_DOOR = Opening(u=0.0, sill=0.0, width=1.1, height=2.4, depth=0.35, fill=(4, OAK))
HOUSE_OPEN_M = 3.2                         # a house's wall takes openings where the court is open this far out from it
CHIMNEY_SPACING_M, CHIMNEY_M, CHIMNEY_INSET_M = 35.0, 2.2, 3.5   # ranges and blocks: sparing stacks well inside the roof


def dome(r):
    """North Tower onion dome profile above its overhanging top (radius, rise), with a short finial."""
    R = OVERHANG * r
    return [(R, 0.0), (1.12 * R, 0.35 * R), (1.08 * R, 0.8 * R), (0.7 * R, 1.35 * R), (0.25 * R, 1.85 * R),
            (0.12 * R, 2.2 * R), (0.12 * R, 2.7 * R), (0.0, 2.75 * R)]


def crown(kind, r):
    """How far a tower's top features rise above its `top`."""
    if kind == NORTH:
        return dome(r)[-1][1]
    if kind == GRYFFINDOR:
        return TURRET_M + CROWN
    return CROWN


def skyline(towers, named, body_top, dropped):
    """Kept towers as dicts with re-ranked tops, and a report; cones become battlements (towers-flat-battlemented)."""
    kept = [dict(i=i, x=x, z=z, r=r, top=top, kind=ASTRONOMY if rf == ASTRONOMY else NORTH if rf == NORTH else
                 GRYFFINDOR if rf == GRYFFINDOR else FLAT)
            for i, (x, z, r, top, rf) in enumerate(towers) if i not in dropped]
    by_name = {}
    for name in ("AT", "RT", "GT", "NT"):
        by_name[name] = min(kept, key=lambda t: math.dist((t["x"], t["z"]), named[name]))
    total = {t["i"]: t["top"] + crown(t["kind"], t["r"]) for t in kept}
    at, rt, gt, nt = (by_name[n]["i"] for n in ("AT", "RT", "GT", "NT"))
    others = max(total[t["i"]] for t in kept if t["i"] not in (at, rt, gt, nt))
    before = dict(total)
    total[rt] = max(total[rt], others + RANK_CLEARANCE_M)
    total[gt] = max(total[gt], others + RANK_CLEARANCE_M)
    for t in (by_name["RT"], by_name["GT"]):     # their drums, not just their crowns, clear the main buildings
        total[t["i"]] = max(total[t["i"]], body_top + RANK_CLEARANCE_M + crown(t["kind"], t["r"]))
    total[nt] = min(total[nt], min(total[rt], total[gt]) - RANK_CLEARANCE_M)
    total[at] = max(total[at], max(body_top, *(v for k, v in total.items() if k != at)) + SKYLINE_CLEARANCE_M)
    for t in kept:
        t["top"] = total[t["i"]] - crown(t["kind"], t["r"])
    names = {by_name[n]["i"]: n for n in by_name}
    report = [f"{names.get(i, '#%d' % i)}: {before[i]:.1f} -> {total[i]:.1f} m" for i in (at, rt, gt, nt)]
    return kept, report + [f"other towers <= {others:.1f} m; main buildings {body_top:.1f} m"]


def facets(t):
    """Facets round a tower's drums: about FACET_M wide on its widest drum, at least 12."""
    widest = OVERHANG * t["r"] if t["kind"] in (ASTRONOMY, NORTH) else t["r"]
    return max(12, round(2 * math.pi * widest / FACET_M))


def drums(t):
    """A tower's faceted drum sections as (radius, foot or None for the ground, head, storeys staggered)."""
    r, top = t["r"], t["top"]
    if t["kind"] in (ASTRONOMY, NORTH):
        return [(COLUMN * r, None, top - DROP_M - NECK, True), (OVERHANG * r, top - DROP_M, top, False)]
    return [(r, None, top, True)]


def tower(t, sunk, attr, openings):
    """The tower's faceted drums, pierced by `openings` {(drum, facet): [Opening]}, and its top."""
    x, z, r, top, kind = t["x"], t["z"], t["r"], t["top"], t["kind"]
    n = facets(t)
    parts = []
    for d, (radius, foot, head, _) in enumerate(drums(t)):
        pts = np.asarray(circle(x, z, radius, n).exterior.coords)[:-1]
        for f in range(n):
            parts.append(facade(pts[f], pts[(f + 1) % n], sunk if foot is None else foot, head, openings.get((d, f), []),
                                attr, reveal=(4, REVEAL), centre=(x, z))[0])
    if kind in (ASTRONOMY, NORTH):
        column, head = COLUMN * r, OVERHANG * r
        parts.append(lathe(x, z, [(column, top - DROP_M - NECK), (head, top - DROP_M)], attr, n, cap=False))
        if kind == ASTRONOMY:                       # open to the elements: a battlemented platform
            parts += [lid(circle(x, z, head, n), top, attr), crenellate(circle(x, z, head, n), top, attr)]
        else:
            parts.append(lathe(x, z, [(rr, top + dy) for rr, dy in dome(r)], attr, n, cap=False))
        return concat(parts)
    parts += [lid(circle(x, z, r, n), top, attr), crenellate(circle(x, z, r, n), top, attr)]
    if kind == GRYFFINDOR:                          # three thinner turrets in a trefoil on the wide base, windowed
        for k in range(3):
            a = k * 2 * math.pi / 3 + 0.5
            cx, cz, tr = x + math.cos(a) * r * 0.5, z + math.sin(a) * r * 0.5, r * 0.44
            away = np.array([math.cos(a), math.sin(a)])
            pts = np.asarray(circle(cx, cz, tr, TURRET_FACETS).exterior.coords)[:-1]
            for f in range(TURRET_FACETS):
                q0, q1 = pts[f], pts[(f + 1) % TURRET_FACETS]
                mid, side = (q0 + q1) / 2, math.dist(q0, q1)
                spec = TURRET_WINDOW._replace(width=min(TURRET_WINDOW.width, side - 2 * EDGE_M - 0.1))
                facing = (mid - np.array([cx, cz])) @ away / tr                 # out of the trefoil, not into its fellows
                sills = np.arange(top + TURRET_WINDOW.first, top + TURRET_M - EDGE_M - spec.height, spec.storey)
                ops = [glazed(spec, side / 2, mid[0], mid[1], s) for j, s in enumerate(sills)
                       if facing > 0.3 and f % 2 == 0 and (j + f // 2) % 2 == 0 and spec.width >= SLIT_M]
                parts.append(facade(q0, q1, top - 2.0, top + TURRET_M, ops, attr, reveal=(4, REVEAL), centre=(cx, cz))[0])
            parts += [lid(circle(cx, cz, tr, TURRET_FACETS), top + TURRET_M, attr),
                      crenellate(circle(cx, cz, tr, TURRET_FACETS), top + TURRET_M, attr)]
    return concat(parts)


def turret_sites(polygon, top):
    """(x, z, top) of small corbelled turrets at well-spaced convex corners of a block (small-scottish-turrets)."""
    outline = orient(polygon.simplify(1.5, preserve_topology=True), 1.0)
    pts = np.asarray(outline.exterior.coords)[:-1]
    if len(pts) < 3:
        return []
    budget = max(1, int(outline.exterior.length // TURRET_SPACING_M))
    corners = []
    for i, p in enumerate(pts):
        a, b = p - pts[i - 1], pts[(i + 1) % len(pts)] - p
        if min(np.hypot(*a), np.hypot(*b)) < 4.0 or a[0] * b[1] - a[1] * b[0] <= 0:   # short edge or reflex corner
            continue
        bisector = a / np.hypot(*a) - b / np.hypot(*b)
        corners.append((np.hypot(*a) + np.hypot(*b), p + bisector / max(np.hypot(*bisector), 1e-9) * 0.6))
    return [(float(c[0]), float(c[1]), float(top(np.array([c[0]]), np.array([c[1]]))[0]))
            for _, c in sorted(corners, key=lambda e: -e[0])[:budget]]


def turret(x, z, y, attr):
    return lathe(x, z, [(0.25, y - TURRET_FOOT_M), (TURRET_R, y - 6.0), (TURRET_R, y + 2.5), (0.0, y + 2.5 + TURRET_CAP)],
                 attr, 16, cap=False)


def polygons(g, least=0.05):
    return [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon" and p.area > least]


def wall_top(k, cls):
    """How far mass class k's walls (with its battlements or eaves) stand above the castle base: its rank."""
    rel, rt = cls[k]
    return rel + (EAVE_M * ROOFS[int(rt)][1] if int(rt) in ROOFS else CROWN if int(rt) == FLAT else 0.0)


def trimmed(o, inside):
    """`o` less `inside` (taller masonry less PARTY_M), less the slivers that leaves along it, narrower than twice
    SLIVER_M: a lower mass's raster fringe poking out past a taller wall. A thin neck meeting it end-on stays, as it
    joins the two."""
    o = shapely.set_precision(o, 1e-3).difference(inside)
    kept = o.buffer(-SLIVER_M, join_style="mitre").buffer(SLIVER_M, join_style="mitre").intersection(o)
    near = inside.buffer(PARTY_M + SLIVER_M / 2, join_style="mitre")
    slivers = [g for g in polygons(o.difference(kept))
               if g.boundary.intersection(near).length > ALONG * g.boundary.length]
    return o.difference(shapely.union_all(slivers)) if slivers else o


def straighten(raws, cls):
    """The traced masses as (class, counter-clockwise outline) with long straight walls, meeting without gaps or
    slivers. Each class's pieces are merged, then simplified within STRAIGHTEN_M, which moves walls in or out by as
    much. The raster's classes tile the castle, so where that opens a gap between two masses (not low walls) the
    lower of them takes it back. Taller classes come first, and each lower mass runs only PARTY_M into those before
    it, less what that leaves as slivers beside them."""
    order = sorted(range(len(cls)), key=lambda k: -wall_top(k, cls))
    merged = [(k, p) for k in order for p in polygons(shapely.union_all([q for kk, q in raws if kk == k]))]
    straight = [p.simplify(STRAIGHTEN_M, preserve_topology=True) for _, p in merged]
    gaps = shapely.union_all([p for _, p in raws]).difference(shapely.union_all(straight))
    tree = shapely.STRtree(straight)
    fill = [[] for _ in merged]
    for g in polygons(gaps):
        between = [int(i) for i in tree.query(g.buffer(0.05), predicate="intersects")
                   if CASTLE_CLASSES[merged[i][0]] != "low_wall"]
        held = sorted(((g.boundary.intersection(straight[i].buffer(0.05)).length / g.boundary.length, i) for i in between),
                      reverse=True)
        if len(held) >= 2 and held[1][0] >= GAP_HELD:      # a slit between two masses, not a strip along one wall
            fill[max(held[0][1], held[1][1])].append(g)
    out, taken, fringes = [], Polygon(), []
    for (k, _), s, f in zip(merged, straight, fill):
        inside = taken.buffer(-PARTY_M, join_style="mitre")
        o = trimmed(shapely.union_all([s, *f]), inside)
        taken = taken.union(shapely.union_all([s, *f]))
        for p in polygons(o, 1.0):
            joins = len(tree.query(p.buffer(0.2), predicate="intersects")) > 2      # itself and two it joins
            stray = (CASTLE_CLASSES[k] != "low_wall" and p.area < STRAY_M2 and p.buffer(-STRAY_M).is_empty)
            fringe = (int(cls[k][1]) in ROOFS and p.area < STRAY_M2           # a roofed fringe along a taller wall
                      and p.boundary.intersection(inside.buffer(2 * PARTY_M)).length > CLING * p.length)
            if fringe:
                fringes.append((k, p))
            elif joins or not stray:
                out.append((k, orient(p, 1.0)))
    for k, p in fringes:                # the raster's edge of the taller mass it clings to: that mass takes it
        near = [(p.boundary.intersection(q.buffer(2 * PARTY_M)).length, j) for j, (kk, q) in enumerate(out)
                if wall_top(kk, cls) > wall_top(k, cls) and q.intersects(p.buffer(2 * PARTY_M))]
        if near:
            j = max(near)[1]
            out[j] = (out[j][0], orient(max(polygons(out[j][1].union(p.buffer(PARTY_M, join_style="mitre"))),
                                            key=lambda g: g.area), 1.0))
        else:
            out.append((k, orient(p, 1.0)))
    return out


def tidy(outlines):
    """Valid counter-clockwise outlines without near-collinear corners, edges under a few centimetres or holes under
    HOLE_M2, which CGAL's skeleton cannot always take."""
    out = []
    for k, o in outlines:
        for p in polygons(shapely.make_valid(o), 1.0):
            p = p.simplify(0.05, preserve_topology=True)
            p = Polygon(p.exterior, [r for r in p.interiors if Polygon(r).area >= HOLE_M2])
            for q in polygons(shapely.make_valid(shapely.remove_repeated_points(p, 0.05)), 1.0):
                q = orient(q, 1.0)
                if q.is_valid:
                    out.append((k, q))
    return out


def sound(mesh):
    """The mesh less its needle triangles, so thin against their length that their normals are noise (quantization
    can turn them over); the cracks they leave are under a millimetre wide."""
    t = mesh.pos[mesh.idx]
    longest = np.max(np.linalg.norm(t - np.roll(t, 1, axis=1), axis=2), axis=1)
    keep = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) > NEEDLE * longest ** 2
    return mesh._replace(idx=mesh.idx[keep])


def battlements(outline, cover, top, attr, thick=0.5, parapet=1.0, merlon=0.8, width=1.1, period=2.2):
    """crenellate()'s parapet and merlons along the outline's walls, except where taller masonry `cover` stands against
    them: there they would stand proud of its face."""
    q = shapely.set_precision(outline, 1e-3)
    ring = q.difference(q.buffer(-thick, join_style="mitre"))
    band = cover.buffer(thick + 0.05, join_style="mitre") if not cover.is_empty else None
    if band is not None:
        ring = ring.difference(band)
    meshes = [prism(orient(p, 1.0), top, lambda x, z: heights(top, x, z) + parapet, attr) for p in polygons(ring, 1e-6)]
    for p0, p1 in edges(outline):
        length = math.dist(p0, p1)
        if length < width:
            continue
        t = (p1 - p0) / length
        inward = np.array([-t[1], t[0]])                    # the material lies left of every ring of a CCW outline
        for s in np.arange(period / 2, length - width / 2, period):
            c = p0 + t * s + inward * thick / 2
            foot = Polygon([c + t * dx * width / 2 + inward * dz * thick / 2 for dx, dz in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            if band is not None and foot.intersects(band):
                continue
            y0 = float(heights(top, np.array([c[0]]), np.array([c[1]]))[0]) + parapet
            meshes.append(prism(orient(foot, 1.0), y0, y0 + merlon, attr))
    return sound(concat(meshes)) if meshes else None


def overhangs(towers):
    """(clearance, foot) round each re-ranked North Tower's overhanging top: the masses standing above `foot` are cut
    back to clear it by NT_CLEAR_M, so the tower stands beside them rather than fused into their roofs."""
    out = []
    for t in towers:
        if t["kind"] == NORTH:
            n = facets(t)
            out.append((circle(t["x"], t["z"], (OVERHANG * t["r"] + NT_CLEAR_M) / math.cos(math.pi / n), n),
                        t["top"] - DROP_M - NT_CLEAR_M))
    return out


def drum_feet(towers, head, stairs):
    """(footprint just inside its facets, top) of each tower's ground drum (tower dicts as skyline() keeps them), the
    Head's tower and its stair turrets."""
    feet = []
    for t in list(towers) + [head]:
        radius, n = drums(t)[0][0], facets(t)
        feet.append((circle(t["x"], t["z"], radius - PARTY_M / math.cos(math.pi / n), n), t["top"]))
    feet += [(circle(x, z, STAIR_TURRET_R - PARTY_M / math.cos(math.pi / 16), 16), top) for x, z, top in stairs]
    return feet


def clear_of_drums(outlines, feet, clear, cls, cbase, local):
    """The masses less the tower drums standing taller than their walls and the `clear`ances round overhanging tops
    (disc, foot) whose foot they rise above, and the slivers either leaves."""
    out = []
    for k, o in outlines:
        rel, rt = cls[k]
        high = wall_top(k, cls) + ((ROOFS[int(rt)][0] - EAVE_M) * ROOFS[int(rt)][1] if int(rt) in ROOFS else 0.0)
        for disc, top, lower in [(f, t, True) for f, t in feet] + [(f, t, False) for f, t in clear]:
            if not o.intersects(disc):
                continue
            c = disc.centroid
            base = float(sample(cbase, local, np.array([c.x]), np.array([c.y]))[0])
            if (top > base + wall_top(k, cls)) if lower else (base + high > top):
                o = trimmed(o, disc)
        out += [(k, orient(p, 1.0)) for p in polygons(o, 1.0)]
    return out


def party_walls(outlines, cls, feet, cbase, local, base=None):
    """Per mass, (whether each wall of edges(outline) stands against taller masonry just outside it along its whole
    length, so is hidden in it; that masonry): a taller mass, an earlier one of the same height, or a tower drum taller
    than its walls. `base(x, z)` gives the castle base (by default sampled from cbase on the local grid)."""
    base = base or (lambda x, z: sample(cbase, local, x, z))
    tops = [wall_top(k, cls) for k, _ in outlines]
    out = []
    for i, (k, o) in enumerate(outlines):
        c = o.representative_point()
        floor = float(base(np.array([c.x]), np.array([c.y]))[0])
        cover = shapely.union_all([p for j, (_, p) in enumerate(outlines) if j != i and (tops[j], -j) > (tops[i], -i)]
                                  + [foot.buffer(PARTY_M) for foot, top in feet if top > floor + tops[i]], grid_size=1e-3)
        walls = edges(o)
        hidden = np.zeros(len(walls), bool)
        for w, (p0, p1) in enumerate(walls):            # hidden only if covered along its whole length
            out_ = np.array([p1[1] - p0[1], p0[0] - p1[0]]) / math.dist(p0, p1) * 0.05
            f = np.linspace(0.02, 0.98, max(3, int(math.dist(p0, p1) / 0.5) + 1))
            pts = p0 + (p1 - p0) * f[:, None] + out_
            hidden[w] = bool(shapely.contains_xy(cover, pts[:, 0], pts[:, 1]).all())
        out.append((hidden, cover))
    return out


def roof_pieces(polygon, limit=ROOF_PIECE_M2):
    """The footprint halved across its longer side until no piece is larger than `limit` square metres."""
    if polygon.area <= limit:
        return [polygon]
    x0, z0, x1, z1 = polygon.bounds
    if x1 - x0 >= z1 - z0:
        halves = box(x0, z0, (x0 + x1) / 2, z1), box((x0 + x1) / 2, z0, x1, z1)
    else:
        halves = box(x0, z0, x1, (z0 + z1) / 2), box(x0, (z0 + z1) / 2, x1, z1)
    out = []
    for half in halves:
        part = polygon.intersection(half)
        out += [q for p in getattr(part, "geoms", [part]) if p.geom_type == "Polygon" and p.area > 1e-6
                for q in roof_pieces(p, limit)]
    return out


def hip(outline, top, rise, slope, party=None):
    """massD's hip roof over a straight outline, rising `slope` per metre in from its eaves for at most `rise` metres;
    measured from the outline's open walls (not those `party` hides), so eaves stay level along each and meet its top."""
    walls = [w for w, p in zip(edges(outline), np.zeros(len(edges(outline)), bool) if party is None else party) if not p]
    edge = shapely.MultiLineString([list(map(tuple, w)) for w in walls]) if walls else None

    def height(x, z):
        x, z = np.asarray(x, np.float64), np.asarray(z, np.float64)
        d = shapely.distance(edge, shapely.points(x, z)) if edge is not None else np.full(np.shape(x), rise)
        return top(x, z) + np.clip(EAVE_M + d, 0.0, rise) * slope
    return height


def draped(tris, fallback):
    """The height of the upward triangles (t, 3, 3) over each point (x, z); `fallback` where none passes over it."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    tris = tris[n[:, 1] > 1e-6]
    a, v0, v1 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    den = v0[:, 0] * v1[:, 2] - v0[:, 2] * v1[:, 0]

    def height(x, z):
        x, z = np.asarray(x, np.float64), np.asarray(z, np.float64)
        px, pz = x.ravel()[:, None] - a[None, :, 0], z.ravel()[:, None] - a[None, :, 2]
        v = (px * v1[:, 2] - pz * v1[:, 0]) / den
        w = (v0[:, 0] * pz - v0[:, 2] * px) / den
        y = np.where((v >= -1e-9) & (w >= -1e-9) & (v + w <= 1 + 1e-9), a[:, 1] + v * v0[:, 1] + w * v1[:, 1], -np.inf)
        best = y.max(axis=1) if len(tris) else np.full(x.size, -np.inf)
        return np.where(np.isfinite(best), best, np.asarray(fallback(x, z), np.float64).ravel()).reshape(np.shape(x))
    return height


def hip_roof(outline, party, eave, rise, slope, attr):
    """massD's hip roof over a straight outline, as a straight skeleton places its hips, valleys and ridges: each open
    wall's slope rises `slope` per metre in from the eaves (eave(x, z), the walls' top) to at most `rise` metres in,
    where the roof is capped flat; a wall `party` hides stands straight up to the roof as a gable, inside the masonry
    hiding it. None where CGAL's skeleton folds the roof over itself."""
    import manifold3d as m3
    from compas_cgal import _straight_skeleton_2 as sk

    rings = []
    for ring in (outline.exterior, *outline.interiors):             # edges()'s walls, edge k from point k to k + 1
        c = np.asarray(ring.coords)
        rings.append(c[:-1][np.hypot(*(c[1:] - c[:-1]).T) > 1e-3])
    flags = iter(party)
    angles = [[90.0 if next(flags) else math.degrees(math.atan(slope)) for _ in r] for r in rings]
    if all(a == 90.0 for ring in angles for a in ring):
        return None
    as3 = lambda r: np.ascontiguousarray(np.column_stack([r, np.zeros(len(r))]), np.float64)
    try:            # CGAL's first angle belongs to the edge from the last point to the first
        v, faces = sk.extrude_straight_skeleton(as3(rings[0]), [as3(r) for r in rings[1:]],
                                                [[a[-1]] + a[:-1] for a in angles], True, (rise - EAVE_M) * slope)
    except RuntimeError:
        return None
    v = np.asarray(v, np.float64)
    tris = []
    for face in faces:
        p = v[face]
        if np.all(p[:, 2] < 1e-9):                     # the base
            continue
        nrm = np.zeros(3)
        for i in range(len(p)):                         # Newell's normal
            a, b = p[i], p[(i + 1) % len(p)]
            nrm += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
        nrm /= np.linalg.norm(nrm)
        u = np.cross(nrm, [0.0, 0.0, 1.0] if abs(nrm[2]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        w = np.cross(nrm, u)
        tris += [p[list(t)] for t in np.asarray(m3.triangulate([np.column_stack([p @ u, p @ w])]))]
    if not tris:
        return None
    tris = np.asarray(tris)
    plan = (tris[:, 1, 0] - tris[:, 0, 0]) * (tris[:, 2, 1] - tris[:, 0, 1]) - (tris[:, 2, 0] - tris[:, 0, 0]) * (tris[:, 1, 1] - tris[:, 0, 1])
    if -plan[plan < 0].sum() / 2 > 1e-6 * outline.area:          # a slope facing down: the roof folds over itself
        return None
    flat = tris.reshape(-1, 3)
    y = np.asarray(eave(flat[:, 0], flat[:, 1]), np.float64) + flat[:, 2]
    world = np.column_stack([flat[:, 0], y, flat[:, 1]]).reshape(-1, 3, 3)
    n = np.cross(world[:, 1] - world[:, 0], world[:, 2] - world[:, 0])
    vertical = np.abs(n[:, 1]) < 1e-6 * np.linalg.norm(n, axis=1)
    return sound(concat([faceted(world[~vertical], attr, up=True), faceted(world[vertical], attr)]))


class Mass(NamedTuple):
    k: int                                 # mass class
    outline: object                        # straightened footprint, counter-clockwise
    eave: object                           # wall top, a callable of x, z
    height: object                         # highest masonry or roof over a point inside, a callable of x, z


class Air:
    """Where a window looks out into open air. Its sight line runs from just behind its pane to CLEAR_M out from its
    wall: the part behind the wall must stay inside the window's own mass (no other wall crosses its reveal), and the
    whole line must stay above the ground and clear of any other mass or round solid (tower, turret) with a top that
    is not HEADROOM_M below the sill. `solids` are (x, z, radius, foot, top, id)."""

    def __init__(self, masses, solids, ground):
        self.masses, self.solids, self.ground = masses, solids, ground
        self.near = [m.outline.buffer(NEAR_M, join_style="mitre") for m in masses]
        for g in (*self.near, *(m.outline for m in masses)):
            shapely.prepare(g)

    def __call__(self, x, z, out, sill, head, depth, mass=None, tower=None):
        """Per window at (x, z) on its wall facing `out`; `mass` (each window's own mass, an index into `masses`, or -1)
        and `tower` (the own tower of a tower's windows) are the masonry its reveal runs through."""
        n = len(x)
        mass = np.full(n, -1) if mass is None else mass
        wall = np.column_stack([x, z])
        a, b = wall - out * (depth + 0.1)[:, None], wall + out * CLEAR_M
        ok = np.ones(n, bool)
        for s in np.linspace(0.0, 1.0, 31):
            q = a + (b - a) * s
            ok &= self.ground(q[:, 0], q[:, 1]) + 0.5 <= sill
        sight = shapely.linestrings(np.stack([a, b], 1))
        reveal = shapely.linestrings(np.stack([a, wall - out * 0.01], 1))
        beyond = shapely.linestrings(np.stack([wall + out * 0.01, b], 1))
        for j, m in enumerate(self.masses):
            mine = mass == j
            ok[mine] &= shapely.within(reveal[mine], m.outline)
            for k in np.flatnonzero(ok & shapely.intersects(sight, self.near[j])):
                line, body = (beyond[k], m.outline) if mine[k] else (sight[k], self.near[j])
                pts = shapely.get_coordinates(shapely.intersection(shapely.segmentize(line, 0.1), body))
                if len(pts) and m.height(pts[:, 0], pts[:, 1]).max() + HEADROOM_M > sill[k]:
                    ok[k] = False
        d = b - a
        for cx, cz, r, foot, top, t in self.solids:
            if t is not None and t == tower:
                continue
            s = np.clip(((cx - a[:, 0]) * d[:, 0] + (cz - a[:, 1]) * d[:, 1]) / (d * d).sum(axis=1), 0.0, 1.0)
            gap = np.hypot(a[:, 0] + d[:, 0] * s - cx, a[:, 1] + d[:, 1] * s - cz)
            ok &= ~((gap < r + NEAR_M) & (top + HEADROOM_M > sill) & (foot < head + 0.5))
        return ok


def pane(x, z, y):
    """A pane's glass code: lit at dusk (a fixed LIT_PERCENT share, DIM_PERCENT of all panes dimly) or dark, picked by
    its position."""
    share = zlib.crc32(f"{x:.1f} {z:.1f} {y:.1f}".encode()) % 100
    return GLASS_DIM if share < DIM_PERCENT else GLASS_LIT if share < LIT_PERCENT else GLASS


def glazed(spec, u, x, z, sill):
    return Opening(float(u), float(sill), spec.width, spec.height, spec.depth, (4, pane(x, z, sill)))


def wall_windows(walls, cbase, local, air):
    """The windows each wall (p0, p1, spec, eave, mass index) takes: columns `spacing` apart clear of its ends, one per
    storey up to the eaves, wherever they look out into open air."""
    found = [[] for _ in walls]
    rows = []
    for w, (p0, p1, spec, eave, mass) in enumerate(walls):
        length = math.dist(p0, p1)
        if spec is None or length < 2 * END_M + spec.width:
            continue
        t = (p1 - p0) / length
        n = int((length - 2 * END_M - spec.width) // spec.spacing) + 1
        u = length / 2 + (np.arange(n) - (n - 1) / 2) * spec.spacing
        x, z = p0[0] + t[0] * u, p0[1] + t[1] * u
        floor = sample(cbase, local, x, z) + spec.first
        highest = eave(x, z) - EDGE_M - spec.height
        storeys = np.arange(max(int(np.max((highest - floor) // spec.storey)) + 1, 1) if spec.storey else 1)
        sill = floor[:, None] + spec.storey * storeys[None, :]
        for i, j in zip(*np.nonzero(sill <= highest[:, None])):
            rows.append((w, u[i], x[i], z[i], t[1], -t[0], sill[i, j], spec.height, spec.depth, mass))
    if rows:
        w, u, x, z, ox, oz, sill, head, depth, mass = np.array(rows).T
        ok = air(x, z, np.column_stack([ox, oz]), sill, sill + head, depth, mass=mass.astype(int))
        for k in np.flatnonzero(ok):
            found[int(w[k])].append(glazed(walls[int(w[k])][2], u[k], x[k], z[k], sill[k]))
    return found


def tower_windows(t, cbase, local, air):
    """{(drum, facet): [Opening]} for a tower: columns on every 2nd or 3rd facet, storeys staggered between columns on
    the main drum, wherever they look out into open air; drums too slender for a window keep none."""
    x, z = t["x"], t["z"]
    n = facets(t)
    step = 2 if n <= 16 else 3
    floor = float(sample(cbase, local, np.array([x]), np.array([z]))[0]) + TOWER_WINDOW.first
    rows = []
    for d, (radius, foot, head, stagger) in enumerate(drums(t)):
        side = 2 * radius * math.sin(math.pi / n)
        width = min(TOWER_WINDOW.width, side - 2 * EDGE_M - 0.1)
        if width < SLIT_M:
            continue
        pts = np.asarray(circle(x, z, radius, n).exterior.coords)[:-1]
        lowest = floor if foot is None else foot + 0.5
        for f in range(0, n - step + 1, step):
            p0, p1 = pts[f], pts[(f + 1) % n]
            mid, tangent = (p0 + p1) / 2, (p1 - p0) / side
            for j in range(int((head - EDGE_M - TOWER_WINDOW.height - floor) // TOWER_WINDOW.storey) + 1):
                sill = floor + TOWER_WINDOW.storey * j
                if sill >= lowest and (not stagger or (j + f // step) % 2 == 0):
                    rows.append((d, f, mid[0], mid[1], tangent[1], -tangent[0], sill, width, side / 2))
    found = {}
    if rows:
        d, f, mx, mz, ox, oz, sill, width, u = np.array(rows).T
        ok = air(mx, mz, np.column_stack([ox, oz]), sill, sill + TOWER_WINDOW.height,
                 np.full(len(mx), TOWER_WINDOW.depth), tower=t["i"])
        for k in np.flatnonzero(ok):
            spec = TOWER_WINDOW._replace(width=float(width[k]))
            found.setdefault((int(d[k]), int(f[k])), []).append(glazed(spec, u[k], mx[k], mz[k], sill[k]))
    return found


def plan_fit(plan_px, named):
    """World (x, z) of a castle-plan pixel (u, v): the similarity transform (scale, rotation, shift) that best maps the
    plan's named features onto meta.named; also the fit's residuals (m) at those features."""
    keys = sorted(plan_px)
    p = np.array([complex(plan_px[k][0], -plan_px[k][1]) for k in keys])
    q = np.array([complex(*named[k]) for k in keys])
    a = np.conj(p - p.mean()) @ (q - q.mean()) / (np.conj(p - p.mean()) @ (p - p.mean()))
    b = q.mean() - a * p.mean()

    def world(u, v):
        w = a * complex(u, -v) + b
        return np.array([w.real, w.imag])
    return world, np.abs(a * p + b - q)


def front_door(walls, aim, cbase, local):
    """(wall index, Opening) of the main doors: on the west-facing wall nearest `aim`, as near it as the doorway fits."""
    best = None
    for w, (_, m, p0, p1) in enumerate(walls):
        length = math.dist(p0, p1)
        t = (p1 - p0) / max(length, 1e-9)
        if t[1] > -0.5 or length < DOOR.width + 2 * EDGE_M + 0.2:      # outward (t.z, -t.x) must face west
            continue
        u = float(np.clip((aim - p0) @ t, DOOR.width / 2 + EDGE_M + 0.1, length - DOOR.width / 2 - EDGE_M - 0.1))
        gap = math.dist(p0 + t * u, aim)
        if best is None or gap < best[0]:
            best = (gap, w, u, p0 + t * u)
    _, w, u, at = best
    sill = float(sample(cbase, local, np.array([at[0]]), np.array([at[1]]))[0]) + THRESHOLD_M
    return w, DOOR._replace(u=u, sill=sill)


def front_steps(p0, p1, door, ground, sunk, attr, drop=THRESHOLD_M, margin=STAIR_MARGIN_M):
    """A flight from a door's threshold down to the grass: a landing, then treads falling RISE_M each through `drop`,
    a few more while the ground keeps falling away, every step a block standing on the terrain."""
    t = (p1 - p0) / math.dist(p0, p1)
    out = np.array([t[1], -t[0]])
    c = p0 + t * door.u
    half = door.width / 2 + margin
    to_base = round(drop / RISE_M)
    steps, near, far, top = [], -EDGE_M, LANDING_M, door.sill
    for i in range(to_base + 4):
        foot = [c + t * s * half + out * d for s, d in ((-1, near), (1, near), (1, far), (-1, far))]
        steps.append(prism(orient(Polygon(foot), 1.0), sunk, top, attr))
        edge = c + out * (far + TREAD_M)
        below = top - float(ground(np.array([edge[0]]), np.array([edge[1]]))[0])
        if below <= 1.5 * RISE_M or (i >= to_base - 1 and below <= 2 * RISE_M):
            break                                   # one ordinary step, or at the castle base a short one, to the grass
        near, far, top = far, far + TREAD_M, top - RISE_M
    return steps


def door_seam(p0, p1, door, attr):
    """The dark meeting line of the double doors, just proud of them."""
    t = (p1 - p0) / math.dist(p0, p1)
    out = np.array([t[1], 0.0, -t[0]])
    inset = sum(i for i, _ in door.orders)
    depth = door.depth + sum(d for _, d in door.orders) - 0.02
    crown = door.sill + door.height - inset - 0.05
    base = np.array([p0[0], 0.0, p0[1]]) + np.array([t[0], 0.0, t[1]]) * door.u - out * depth
    side = np.array([t[0], 0.0, t[1]]) * 0.03
    corners = [base + side + [0, door.sill, 0], base - side + [0, door.sill, 0], base - side + [0, crown, 0],
               base + side + [0, crown, 0]]
    seam = panel(corners, attr)
    flip = seam.nrm[0] @ out < 0
    return Mesh(seam.pos, -seam.nrm if flip else seam.nrm, seam.attr, seam.idx[:, [0, 2, 1]] if flip else seam.idx)


def door_boards(p0, p1, door, attr):
    """The double doors' boards and ironwork, in `attr` just proud of the oak: the joints between boards BOARD_M
    wide, under the arch's head, and three straps across each leaf from its hinge nearly to the meeting line."""
    t = (p1 - p0) / math.dist(p0, p1)
    along, out = np.array([t[0], 0.0, t[1]]), np.array([t[1], 0.0, -t[0]])
    inset = sum(i for i, _ in door.orders)
    depth = door.depth + sum(d for _, d in door.orders) - 0.02
    half = door.width / 2 - inset                   # the leaves' half-width, the arch's radius
    spring = door.sill + door.height - inset - half
    at = lambda s, y, proud=0.0: np.array([p0[0], y, p0[1]]) + along * (door.u + s) - out * (depth - proud)
    parts = []
    for s in [side * k * BOARD_M for side in (-1, 1) for k in range(1, int((half - 0.05) / BOARD_M) + 1)]:
        head = spring + math.sqrt(max(half ** 2 - s ** 2, 0.0)) - 0.05     # each leaf boarded out from the meeting line
        joint = panel([at(s - 0.015, door.sill), at(s + 0.015, door.sill), at(s + 0.015, head), at(s - 0.015, head)], attr)
        flip = joint.nrm[0] @ out < 0
        parts.append(Mesh(joint.pos, -joint.nrm if flip else joint.nrm, joint.attr,
                          joint.idx[:, [0, 2, 1]] if flip else joint.idx))
    for y in (door.sill + 0.8, (door.sill + spring) / 2 + 0.4, spring - 0.1):
        for side in (-1, 1):
            parts.append(bar(at(side * (half - 0.1), y, STRAP_M / 2), at(side * 0.15, y, STRAP_M / 2), STRAP_M, attr))
    return parts


def edges(outline):
    """The outline's wall runs (p0, p1), in the order the castle builds them."""
    return [(p0, p1) for ring in (outline.exterior, *outline.interiors)
            for p0, p1 in zip(np.asarray(ring.coords)[:-1], np.asarray(ring.coords)[1:]) if math.dist(p0, p1) > 1e-3]


class Passage(NamedTuple):
    host: object                           # the mass it runs through, grown by a gate bay round the way through
    faces: tuple                           # the (p0, p1) walls it opens in: lawn side, court side
    cuts: tuple                            # their (u, y) openings
    near: np.ndarray                       # (n, 3) world ring of the arch on the lawn side
    far: np.ndarray                        # the same ring at the court side
    corridor: object                       # the way through and a little beyond, cleared of other masses


BAY_M = 1.0                                # the gate bay's masonry either side of the pend
PEND_SHIFT_M = 1.5                         # the pend runs square to the lawn wall if that keeps it this near the trace


def passage(outlines, lawn, court, ground):
    """The pend through the storeyed mass the traced line crosses furthest: square to that mass's wall facing the
    lawn where one does near the trace (else along the trace), in a shallow bay standing just proud of both sides, so
    it opens in two straight faces."""
    reach = court - lawn
    line = shapely.LineString([lawn - reach, court + reach])
    host = max((i for i, (k, _) in enumerate(outlines) if CASTLE_CLASSES[k] in WINDOWS),
               key=lambda i: line.intersection(outlines[i][1]).length)
    o = outlines[host][1]
    crossing = line.intersection(o)
    crossing = max(getattr(crossing, "geoms", [crossing]), key=lambda g: g.length)
    a = min(np.asarray(crossing.coords), key=lambda p: (p - lawn) @ reach)   # the end towards the lawn
    toward = -reach / np.linalg.norm(reach)
    faces = [(p0, p1) for p0, p1 in edges(o) if math.dist(p0, p1) >= PEND_WIDTH_M + 2 * EDGE_M + 0.4
             and shapely.LineString([p0, p1]).distance(shapely.Point(a)) < 8.0
             and np.array([p1[1] - p0[1], p0[0] - p1[0]]) @ toward > 0.85 * math.dist(p0, p1)]
    margin = PEND_WIDTH_M / 2 + EDGE_M + 0.2
    square = None
    if faces:
        p0, p1 = min(faces, key=lambda e: shapely.LineString(e).distance(shapely.Point(a)))
        t = (p1 - p0) / math.dist(p0, p1)
        on = p0 + t * float(np.clip((a - p0) @ t, margin, math.dist(p0, p1) - margin))
        if math.dist(on, a) <= PEND_SHIFT_M:
            square = t, -np.array([t[1], -t[0]]), on          # into the mass, square to the lawn face
    if square is None:                                     # no wall there faces the lawn: along the traced line
        d = -toward
        square = np.array([-d[1], d[0]]), d, a
    t, d, a = square
    half = PEND_WIDTH_M / 2 + BAY_M
    strip = shapely.Polygon([a - t * half - d * 20, a + t * half - d * 20, a + t * half + d * 60, a - t * half + d * 60])
    through = o.intersection(strip)
    through = min(getattr(through, "geoms", [through]), key=lambda g: g.distance(shapely.Point(a)))
    s = (shapely.get_coordinates(through) - a) @ d
    s0, s1 = s.min() - 0.5, s.max() + 0.5
    corner = lambda side, depth: a + t * side * half + d * depth
    bay = Polygon([corner(-1, s0), corner(1, s0), corner(1, s1), corner(-1, s1)])
    grown = orient(o.union(bay), 1.0)
    under = [a + d * s_ + t * w for s_ in np.arange(s0, s1 + 0.25, 0.5) for w in (-PEND_WIDTH_M / 2, 0.0, PEND_WIDTH_M / 2)]
    floor = max(float(ground(np.array([p[0]]), np.array([p[1]]))[0]) for p in under) + 0.25   # over the terrain's fine relief
    ring = arch(Opening(u=0.0, sill=floor, width=PEND_WIDTH_M, height=PEND_HEIGHT_M, depth=0.0))
    across = lambda depth: np.column_stack([a[0] + t[0] * ring[:, 0] + d[0] * depth, ring[:, 1],
                                            a[1] + t[1] * ring[:, 0] + d[1] * depth])
    walls, cuts = [], []
    for ends, depth in (({tuple(corner(-1, s0)), tuple(corner(1, s0))}, s0), ({tuple(corner(-1, s1)), tuple(corner(1, s1))}, s1)):
        q0, q1 = next((q0, q1) for q0, q1 in edges(grown)       # within the host's 1 mm precision grid
                      if all(min(math.dist(q, e) for e in ends) < 0.01 for q in (q0, q1)))
        tq = (q1 - q0) / math.dist(q0, q1)
        centre = a + d * depth
        walls.append((q0, q1))
        cuts.append(Polygon(np.column_stack([(centre - q0) @ tq + ring[:, 0] * (t @ tq), ring[:, 1]])))
    side = t * (half + 0.5)
    corridor = Polygon([a - side + d * (s0 - 2), a + side + d * (s0 - 2), a + side + d * (s1 + 2), a - side + d * (s1 + 2)])
    return Passage(grown, tuple(walls), tuple(cuts), across(s0), across(s1), corridor), host


def gates(p, attr):
    """Two oak leaves hung inside the court end, folded back against the walls of the way through: each boarded, with
    two ledges and a brace on its face to the passage."""
    d = p.far[:, [0, 2]].mean(axis=0) - p.near[:, [0, 2]].mean(axis=0)
    d /= np.linalg.norm(d)
    feet = p.far[:2, [0, 2]]                          # the outline starts with its two feet
    floor = p.far[0, 1]
    leaves = []
    for k in range(2):
        inward = feet[1 - k] - feet[k]
        inward /= np.linalg.norm(inward)
        hinge = feet[k] - d * 0.15 + inward * 0.05
        foot = [hinge, hinge - d * PEND_WIDTH_M / 2, hinge - d * PEND_WIDTH_M / 2 + inward * 0.1, hinge + inward * 0.1]
        leaves.append(prism(orient(Polygon(foot), 1.0), floor, floor + GATE_M, attr))
        face = lambda s, y: np.array([*(hinge - d * s + inward * 0.16)[:1], floor + y, *(hinge - d * s + inward * 0.16)[1:]])
        near, far = 0.12, PEND_WIDTH_M / 2 - 0.12
        for y in (0.45, GATE_M - 0.45):
            leaves.append(bar(face(near, y), face(far, y), 0.12, attr))
        leaves.append(bar(face(near, 0.5), face(far, GATE_M - 0.5), 0.1, attr))
    return leaves


def heads_tower(at, garden, cbase, local, ground, sunk):
    """The Head's tower at the end of the hospital wing: the round tower (kind FLAT, battlemented), the two stair
    turrets running up its side, a quarter turn either side of the garden so that neither stands on the garden door,
    and that door's external stair down. Returns (tower, turret (x, z, top) list, meshes, (door facet, door))."""
    base = float(sample(cbase, local, np.array([at[0]]), np.array([at[1]]))[0])
    head = dict(i="HO", x=float(at[0]), z=float(at[1]), r=HEAD_R, top=base + HEAD_RISE_M, kind=FLAT)
    g = (garden - at) / np.linalg.norm(garden - at)
    turrets, parts = [], []
    for turn in (math.pi / 2, -math.pi / 2):
        d = np.array([g[0] * math.cos(turn) - g[1] * math.sin(turn), g[0] * math.sin(turn) + g[1] * math.cos(turn)])
        c = at + d * (HEAD_R + 0.4)
        foot = float(sunk(np.array([c[0]]), np.array([c[1]]))[0])
        top = head["top"] + 3.0
        parts.append(lathe(c[0], c[1], [(STAIR_TURRET_R, foot), (STAIR_TURRET_R, top), (0.0, top + TURRET_CAP)],
                           (4, REAL_AUX + CASTLE_CLASSES.index("block")), 16, cap=False))
        turrets.append((float(c[0]), float(c[1]), top + TURRET_CAP))
    n = facets(head)
    pts = np.asarray(circle(at[0], at[1], HEAD_R, n).exterior.coords)[:-1]
    f = int(round(math.atan2(g[1], g[0]) / (2 * math.pi / n) - 0.5)) % n
    p0, p1 = pts[f], pts[(f + 1) % n]
    outside = (p0 + p1) / 2 + g
    sill = float(ground(np.array([outside[0]]), np.array([outside[1]]))[0]) + HEAD_THRESHOLD_M
    door = HEAD_DOOR._replace(u=math.dist(p0, p1) / 2, sill=sill)
    parts += front_steps(p0, p1, door, ground, sunk, (4, PLAIN), drop=HEAD_THRESHOLD_M, margin=0.4)
    return head, turrets, parts, (f, door)


def physick_garden(at, toward, outlines, clear, ground, sunk):
    """The Physick Garden and Headmaster's Pleasaunce: the corner below the Head's tower towards `toward`, clear of the
    castle's masses (and `clear`), walled low with a gate in its far side, its stair let through, and planted with herb
    beds in rows. Returns its meshes and its walls as masses (for the windows' open-air check)."""
    g = (toward - at) / np.linalg.norm(toward - at)
    s = np.array([-g[1], g[0]])
    near = HEAD_R + 0.3
    corner = lambda a, b: at + g * a + s * b
    box = lambda a0, a1, b0, b1: Polygon([corner(a0, b0), corner(a0, b1), corner(a1, b1), corner(a1, b0)])
    garden = box(near, near + GARDEN_DEPTH_M, -GARDEN_WIDTH_M / 2, GARDEN_WIDTH_M / 2)
    garden = garden.difference(shapely.union_all([o.buffer(0.2) for o in outlines] + [clear]))
    garden = max(getattr(garden, "geoms", [garden]), key=lambda p: p.area)
    stair = box(HEAD_R - 1.0, near + GARDEN_DEPTH_M, -1.6, 1.6)
    gate = box(near + GARDEN_DEPTH_M - 1.0, near + GARDEN_DEPTH_M + 1.0, -0.8, 0.8)
    ring = garden.difference(garden.buffer(-0.45, join_style="mitre"))
    ring = ring.difference(gate).difference(box(HEAD_R - 1.0, near + 1.0, -1.6, 1.6))
    top = lambda x, z: ground(x, z) + GARDEN_WALL_M
    walls = [orient(p, 1.0) for p in getattr(ring, "geoms", [ring]) if p.geom_type == "Polygon" and p.area > 0.05]
    parts = [prism(w, sunk, top, WALL_STONE) for w in walls]
    beds = garden.buffer(-1.2, join_style="mitre").difference(stair)
    for i in range(int(GARDEN_DEPTH_M // (BED_W + PATH_M)) + 1):
        for j in range(int(GARDEN_WIDTH_M // (BED_L + PATH_M)) + 1):
            a = near + 1.6 + i * (BED_W + PATH_M)
            b = -GARDEN_WIDTH_M / 2 + 1.6 + j * (BED_L + PATH_M)
            bed = box(a, a + BED_W, b, b + BED_L)
            if beds.contains(bed):
                parts.append(prism(orient(bed, 1.0), sunk, lambda x, z: ground(x, z) + BED_H, HERBS))
    return parts, [Mass(0, w, top, top) for w in walls]


def stack(x, z, foot, top, attr, along=(1.0, 0.0)):
    """A chimney stack from `foot` to `top` with a projecting cope, its long side along `along`."""
    a = np.asarray(along, np.float64) / np.linalg.norm(along)
    b = np.array([-a[1], a[0]])
    rect = lambda la, lb: orient(Polygon([np.array([x, z]) + a * sa * la + b * sb * lb
                                          for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]), 1.0)
    return [prism(rect(0.7, 0.45), foot, top, attr), prism(rect(0.8, 0.55), top, top + 0.2, attr)]


def chimneys(outline, height, attr):
    """Sparing stacks on a roof, evenly spread over the part of it well inside its walls: candidates on a fine grid,
    kept in order while at least CHIMNEY_SPACING_M from every stack already kept. Returns their meshes and their
    (x, z, radius, foot, top, id) solids."""
    inner = outline.buffer(-CHIMNEY_INSET_M)
    if inner.is_empty:
        return [], []
    x0, z0, x1, z1 = inner.bounds
    kept = []
    for x in np.arange(math.floor(x0 / 2.0) * 2.0 + 1.0, x1, 2.0):
        for z in np.arange(math.floor(z0 / 2.0) * 2.0 + 1.0, z1, 2.0):
            if all(math.dist((x, z), k) >= CHIMNEY_SPACING_M for k in kept) and inner.contains(shapely.Point(x, z)):
                kept.append((float(x), float(z)))
    parts, solids = [], []
    for x, z in kept:
        y = float(height(np.array([x]), np.array([z]))[0])
        parts += stack(x, z, y - 0.5, y + CHIMNEY_M, attr)
        solids.append((x, z, 1.0, y - 0.5, y + CHIMNEY_M + 0.2, None))
    return parts, solids


def house(polygon, ground, sunk, attr, outside):
    """A court building as a two-storey gabled house on its footprint's rectangle: its door and windows on the walls
    that look out on open court (`outside(x, z)` of points there), a window to each storey of an open gable end, and
    a chimney stack on the ridge at each gable."""
    cx, cz, length, width, a = fit_rectangle(polygon)
    b = np.array([-a[1], a[0]])
    c = np.array([cx, cz])
    floor = float(ground(np.array([cx]), np.array([cz]))[0])
    eave, rise = floor + HOUSE_EAVE_M, width / 2 * HOUSE_PITCH
    corner = lambda s, t: c + a * s * length / 2 + b * t * width / 2
    ring = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)]       # counter-clockwise: walls face out
    sides = [(ring[k], ring[(k + 1) % 4]) for k in range(4)]
    def mouth(p0, p1):
        """How far out from the castle's masses the court lies HOUSE_OPEN_M out from the wall p0-p1, at its least."""
        out = np.array([p1[1] - p0[1], p0[0] - p1[0]]) / math.dist(p0, p1) * HOUSE_OPEN_M
        pts = np.array([p0 + (p1 - p0) * f + out for f in (0.2, 0.5, 0.8)])
        return float(np.min(outside(pts[:, 0], pts[:, 1])))
    openness = [mouth(p0, p1) for p0, p1 in sides]
    front = max((0, 2), key=lambda k: openness[k])                           # the long side most open to the court
    parts = []
    for k, (p0, p1) in enumerate(sides):
        span = math.dist(p0, p1)
        ops = []
        if openness[k] > 0:
            if k == front:
                door_at = p0 + (p1 - p0) / 2
                ops.append(HOUSE_DOOR._replace(u=span / 2, sill=float(ground(*door_at[:, None])[0]) + 0.05))
            n = max(1, int((span - 2 * END_M) // HOUSE_WINDOW.spacing)) if k in (0, 2) else 1
            for u in span / 2 + (np.arange(n) - (n - 1) / 2) * HOUSE_WINDOW.spacing:
                at = p0 + (p1 - p0) * u / span
                for j in range(2):
                    sill = floor + HOUSE_WINDOW.first + j * HOUSE_WINDOW.storey
                    if j or k != front or abs(u - span / 2) > HOUSE_DOOR.width / 2 + HOUSE_WINDOW.width:
                        ops.append(glazed(HOUSE_WINDOW, u, at[0], at[1], sill))
        parts.append(facade(p0, p1, sunk, eave, ops, attr, reveal=(4, REVEAL))[0])
    p3 = lambda s, t, y: np.array([(c + a * s + b * t)[0], y, (c + a * s + b * t)[1]])
    hl, hw, over = length / 2, width / 2, 0.4
    e = eave - over * rise / hw
    shell = concat([panel([p3(s * hl, -hw, eave), p3(s * hl, hw, eave), p3(s * hl, 0.0, eave + rise)], attr) for s in (-1, 1)]
                   + [panel([p3(-hl - over, t * (hw + over), e), p3(hl + over, t * (hw + over), e),
                             p3(hl + over, 0.0, eave + rise), p3(-hl - over, 0.0, eave + rise)], attr) for t in (-1, 1)])
    centre = np.array([cx, eave, cz])                # face every panel away from the house's middle
    flip = np.einsum("ij,ij->i", shell.nrm[shell.idx[:, 0]], shell.pos[shell.idx[:, 0]] - centre) < 0
    idx, nrm = shell.idx.copy(), shell.nrm.copy()
    idx[flip] = idx[flip][:, [0, 2, 1]]
    for k in np.flatnonzero(flip):
        nrm[idx[k]] = -shell.nrm[shell.idx[k]]
    parts.append(Mesh(shell.pos, nrm, shell.attr, idx))
    for s in (-1, 1):                                # a stack on the ridge at each gable
        end = c + a * s * (hl - 0.8)
        parts += stack(end[0], end[1], eave + rise - 0.6, eave + rise + 1.8, (4, PLAIN), along=(-a[1], a[0]))
    return parts


def castle(data, csd, castle_grid, local, cbase, keep, ground, sunk, dropped, traced, image, regions):
    """Pieces of the castle asset and the re-ranked towers (for the payload's tower table); `image` and `regions` are
    the castle grid's piece labels and pieces."""
    meta = data["meta"]
    pieces, masses, sites = [], [], []
    named = {k: tuple(v) for k, v in meta["named"].items()}
    plan, _ = plan_fit(traced["plan-fit"]["plan_px"], named)
    texel = lambda x, z: image[tuple(int(round(float(v))) for v in castle_grid.to_texel(x, z))]
    bases = {texel(x, z) for x, z, *_ in meta["towers"]}
    houses = [r["id"] for r in regions if r["role"] == "court" and r["area"] < HOUSE_MAX_M2 and r["id"] not in bases]
    homes, stacks = [], []
    outside = lambda x, z: np.min([sample(csd[..., k], castle_grid, x, z) for k in range(len(meta["castle"]["cls"]))],
                                  axis=0)               # metres out from the castle's masses (negative inside them)
    for h in houses:                                    # court buildings as houses (court-buildings-as-houses)
        shape = shapely.union_all([p for k in range(len(meta["castle"]["cls"]))
                                   for p in field_polygons(csd[..., k], castle_grid, 0.0, image == h)])
        shape = max(getattr(shape, "geoms", [shape]), key=lambda g: g.area)
        pieces += house(shape, ground, sunk, (4, REAL_AUX + CASTLE_CLASSES.index("range")), outside)
        cx, cz, length, width, a = fit_rectangle(shape)
        ridge = float(ground(np.array([cx]), np.array([cz]))[0]) + HOUSE_EAVE_M + width / 2 * HOUSE_PITCH
        b = np.array([-a[1], a[0]])
        footprint = Polygon([np.array([cx, cz]) + a * sa * (length / 2 + 0.3) + b * sb * (width / 2 + 0.3)
                             for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))])    # as built, with its eaves
        homes.append(Mass(CASTLE_CLASSES.index("range"), orient(footprint, 1.0), lambda x, z, y=ridge: np.full(np.shape(x), y),
                          lambda x, z, y=ridge: np.full(np.shape(x), y)))
    keep = keep & ~np.isin(image, houses)
    cls = meta["castle"]["cls"]
    raws = [(k, p) for k in range(len(cls)) for p in field_polygons(csd[..., k], castle_grid, 0.0, keep)]
    ends = traced["broomshed-pend"]["plan_px"]
    lawn, court = plan(*ends["lawn"]), plan(*ends["court"])
    way = shapely.LineString([lawn - (court - lawn) * 0.5, court + (court - lawn) * 0.5]).buffer(PEND_WIDTH_M / 2 + 1.5)
    raws = [(k, p) for k, p in raws if p.area >= WING_M2 or not p.intersects(way)]   # raster slivers across the pend
    outlines = straighten(raws, cls)
    pend, host = passage(outlines, lawn, court, ground)
    outlines = [(outlines[host][0], pend.host)] + [
        (k, orient(g, 1.0)) for i, (k, o) in enumerate(outlines) if i != host
        for g in getattr(o.difference(pend.corridor), "geoms", [o.difference(pend.corridor)])
        if g.geom_type == "Polygon" and g.area > 1.0]
    head, stairs, head_parts, (door_facet, head_door) = heads_tower(
        plan(*traced["heads-tower"]["plan_px"]), plan(*traced["physick-garden"]["plan_px"]), cbase, local, ground, sunk)

    def raised(outlines, feet):
        """The masses' walls' tops, roofs, battlements, turrets and chimneys: (party walls, pieces, masses, turret sites,
        chimney solids)."""
        party = party_walls(outlines, cls, feet, cbase, local)
        built, masses, sites, stacks = [], [], [], []
        for (k, outline), (hidden, cover) in zip(outlines, party):
            rel, rt = cls[k]
            attr = (4, REAL_AUX + k)

            def top(x, z, rel=rel):
                return sample(cbase, local, x, z) + rel

            if int(rt) in ROOFS:
                rise, slope = ROOFS[int(rt)]
                eave = lambda x, z, top=top, slope=slope: top(x, z) + EAVE_M * slope
                height = hip(outline, top, rise, slope, hidden)
                skin = hip_roof(outline, hidden, eave, rise, slope, attr) if outline.area >= SKELETON_M2 else None
                if skin is None:                            # a heightfield where the skeleton folds
                    built += [roof(part, height, attr) for part in roof_pieces(outline)]
                else:
                    built.append(skin)
                    height = draped(skin.pos[skin.idx], height)
                masses.append(Mass(k, outline, eave, height))
                if CASTLE_CLASSES[k] == "block":           # the tall blocks carry the corner turrets
                    sites += turret_sites(outline, top)
                if CASTLE_CLASSES[k] in ("range", "block"):
                    parts, solids = chimneys(outline, height, (4, PLAIN))
                    built += parts
                    stacks += solids
            else:
                built += [sound(lid(part, top, attr)) for part in roof_pieces(outline)]
                if int(rt) == FLAT and (b := battlements(outline, cover, top, attr)) is not None:
                    built.append(b)
                masses.append(Mass(k, outline, top, (lambda x, z, top=top: top(x, z) + CROWN) if int(rt) == FLAT else top))
        built += [turret(x, z, y, (4, REAL_AUX + CASTLE_CLASSES.index("block"))) for x, z, y in sites]
        return party, built, masses, sites, stacks

    outlines0 = outlines
    towers = [dict(i=i, x=x, z=z, r=r, top=top, kind=rf if rf in (ASTRONOMY, NORTH) else FLAT)
              for i, (x, z, r, top, rf) in enumerate(meta["towers"]) if i not in dropped]   # provisional, as meta has them
    ranked = None
    for _ in range(3):              # cut round the drums and the North Tower's top as the skyline re-ranks them
        feet = drum_feet(towers, head, stairs)
        outlines = tidy(clear_of_drums(outlines0, feet, overhangs(towers) if ranked else [], cls, cbase, local))
        party, built, masses, sites, stacks = raised(outlines, feet)
        body_top = max(float(p.pos[:, 1].max()) for p in pieces + built)
        towers, report = skyline(meta["towers"], named, body_top, dropped)
        tops = [(t["i"], round(t["top"], 3)) for t in towers]
        if tops == ranked:
            break
        ranked = tops
    pieces += built
    clear = shapely.union_all([shapely.Point(head["x"], head["z"]).buffer(HEAD_R + 0.2)]
                              + [shapely.Point(x, z).buffer(STAIR_TURRET_R + 0.3) for x, z, _ in stairs])
    garden, garden_walls = physick_garden(np.array([head["x"], head["z"]]), plan(*traced["physick-garden"]["plan_px"]),
                                          [o for _, o in outlines], clear, ground, sunk)
    solids = [(t["x"], t["z"], (OVERHANG if t["kind"] in (ASTRONOMY, NORTH) else 1.0) * t["r"], -math.inf,
               t["top"] + crown(t["kind"], t["r"]), t["i"]) for t in towers]
    solids += [(x, z, TURRET_R, y - TURRET_FOOT_M, y + 2.5 + TURRET_CAP, None) for x, z, y in sites]
    solids += [(head["x"], head["z"], HEAD_R, -math.inf, head["top"] + CROWN, head["i"])]
    solids += [(x, z, STAIR_TURRET_R, -math.inf, top, None) for x, z, top in stairs] + stacks
    air = Air(masses + garden_walls + homes, solids, ground)
    walls = [(j, m, p0, p1) for j, m in enumerate(masses) for (p0, p1), hid in zip(edges(m.outline), party[j][0]) if not hid]
    openings = wall_windows([(p0, p1, WINDOWS.get(CASTLE_CLASSES[m.k]) if m.outline.area >= WING_M2 else None, m.eave, j)
                             for j, m, p0, p1 in walls], cbase, local, air)
    w, door = front_door(walls, plan(*traced["front-steps"]["plan_px"]), cbase, local)
    openings[w] = [door] + openings[w]                  # windows crowding the doorway give way to it
    _, _, p0, p1 = walls[w]
    pieces += front_steps(p0, p1, door, ground, sunk, (4, PLAIN)) + [door_seam(p0, p1, door, (4, GLASS))]
    pieces += door_boards(p0, p1, door, (4, IRON))
    cuts = {}                                           # the pend's openings in its host's walls
    for (q0, q1), cut in zip(pend.faces, pend.cuts):
        cuts[next(i for i, (_, _, p0, p1) in enumerate(walls) if np.allclose(p0, q0) and np.allclose(p1, q1))] = [cut]
    for i, ((_, m, p0, p1), found) in enumerate(zip(walls, openings)):
        pieces.append(facade(p0, p1, sunk, m.eave, found, (4, REAL_AUX + m.k), reveal=(4, REVEAL), cuts=cuts.get(i, ()))[0])
    pieces += [tube(pend.near, pend.far, (4, REVEAL))] + gates(pend, (4, OAK))
    head_openings = tower_windows(head, cbase, local, air)
    head_openings[(0, door_facet)] = [head_door] + head_openings.get((0, door_facet), [])
    pieces += head_parts + garden + [tower(head, sunk, (4, REAL_AUX + CASTLE_CLASSES.index("keep")), head_openings)]
    pieces += [tower(t, sunk, (4, MESH_TOWER_AUX + t["i"]), tower_windows(t, cbase, local, air)) for t in towers]
    panes = sum(len(o) for o in openings)
    return pieces, towers, report + [f"{panes} wall windows"]


# ---------------------------------------------------------------- family hooks (build_meshes.massing)
def build_castle_pieces(scene, asset, keep):
    """The castle itself from its castle-grid pieces; the family's other castle-grid assets take the generic massing.
    Records the mesh towers' tops and the skyline report for the payload."""
    if asset != "castle":
        return None
    image, pieces = scene.castle_labelled
    owned = scene.piece_owners
    dropped = set()                 # round masses re-read as another asset's building (castle-mass-sheds)
    for i, (x, z, *_) in enumerate(scene.meta["towers"]):
        row, col = (int(round(float(v))) for v in scene.castle.to_texel(x, z))
        piece = image[row, col] if 0 <= row < scene.castle.ny and 0 <= col < scene.castle.nx else 0
        if piece and owned[piece] != "castle":
            dropped.add(i)
    traced = {t["id"]: t for t in scene.assets["castle"]["traced"]}
    walk = mesh_grounds.surface(scene)      # over the harbour mouth, its roof's rock: the channel runs under the castle
    sunk = lambda x, z: scene.sunk(x, z) + (walk(x, z) - scene.ground(x, z))
    parts, towers, report = castle(scene.data, scene.csd, scene.castle, scene.local, scene.cbase, keep, walk, sunk,
                                   dropped, traced, image, pieces)
    scene.outputs["towers"], scene.outputs["report"] = towers, report
    return {"castle": parts}
