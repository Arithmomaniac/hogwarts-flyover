"""Grounds family (mesh-grounds): Hagrid's hut and its garden, as White Hound measures and draws them.

Hut (hut-plan, hut-front-gable, timber-hut, hut-weatherboard-creosoted, hut-roof-shingles, hut-stovepipe): a
weatherboarded house 35 by 21 ft on the drawn footprint (4a), its long front facing the castle and built as a gable
end, so the eaves stand 10 ft high at the sides and a 15 ft ridge runs from front to back; a 5 ft door between two
4 by 8 ft sash windows of square panes, front and back; a stovepipe through the roof over the front-left corner; a
gutter along the right eave whose downpipe fills the rain-barrel right of the front door, and another barrel at the back
on the left.

Garden (garden-fence, garden-contents): a post-and-rail fence low enough to vault along White Hound's traced outline
(4b), stopping at the house, with a gate near it and a growing tree standing in its far side; a dozen boulder-sized
pumpkins near the back door, a row of runner beans, a potato bed, a compost heap and a hen-coop.

The hut is built in a local frame: x to the right as you face the front, y up from the floor, z from the front
towards the back (a proper rotation of world x east, y up, z north).

Pitch (pitch-layout-wh, stands-full-ring, pitch-goals-equal-height, pitch-stairs-from-lawn,
pitch-changing-room-materials): on the registered ellipse (meta.pitch: centre, semi-axes a and b of the stands' outer
edge, bearing of the long axis, level of the playing surface), three staggered single-row tiers of planked decks on an
open timber frame, a vaultable barrier with a walkway behind it, four 15 ft stairs at parameter angles ±40° and ±140°
that climb from the lawn behind the stands to the top tier and down through a cut in the tiers to the walkway, a 10 ft
entrance in the middle of the castle (north) side, the 85 by 18 ft changing block interrupting the south stands with
five doors onto the pitch, a freestanding commentator's podium near the middle of the castle side, and six golden hoops
50 ft high. Rings of the stands are offsets of the outer ellipse along its inward normal, so an offset `d` is exactly
`d` metres in from the outer edge.

Gates (gates-pillars-boars, gates-closed): in the boundary wall's mapped gap (meta.extras.gates), two tall pale-stone
pillars either side of the road through it (found in the road channel), 18 ft apart, the wall built up to them from the
gap's ends; on each a winged boar of weathered stone (tools/blender/grounds_boar.py, exported to src/assets/grounds),
facing out of the grounds; and between them high, paired wrought-iron leaves, closed, with spear-topped bars under an
arched top rail.

Walls, hedges and the tomb (walls-coped, cliff-edge-safety-wall, vegetable-patch-wall-ruin, hedge-crowns, tomb-chest):
the boundary walls (3a) 2.2 m high under a pale coping; the wall along the cliff's edge (2e) a low coped safety wall on
the contour near the top of its mapped footprint; the wall at the edge of the vegetable patch (5d) the thick, low,
broken remnant of the former curtain wall; the field hedges with irregular crowns; and Dumbledore's white marble chest
tomb on two steps (meta.extras.tomb)."""
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from materials import code
from mesh_geometry import (Mesh, bar, box, concat, faceted, fit_rectangle, frame, gabled, lathe, lid, panel, pierced,
                           placed, prism)
from mesh_source import CHANNELS, decode, field_polygons, sample, sd_bytes

FAMILY = "grounds"

FT = 0.3048
# G-buffer codes from the material registry (src/materials); several are transport's or shared, reused here
TIMBER, IRON, GLASS, BARK, SASH = (code(n) for n in ("dark-timber", "black-iron", "dim-glass", "sleeper",
                                                     "cream-boarding"))
WEATHERBOARD, SHINGLES, PUMPKIN, EARTH, PLANKS = (code(n) for n in ("creosoted-weatherboard", "shingles", "pumpkin",
                                                                    "turned-earth", "weathered-planks"))
FOLIAGE = code("hedge")                   # the hedges' green
WALL_STONE = code("grounds-rubble")       # the grounds' walls, gate pillars and bridges: rubble stone in rough courses
WALL_IN = 1.05                            # boundary walls are drawn wider than scale and thinned (template_b WALL_IN)
HEDGE_LEVEL = 0.45
WALL_HEIGHT_M, HEDGE_HEIGHT_M = 2.2, 1.5

HUT_W, HUT_D = 35 * FT, 21 * FT           # frontage and depth
EAVE_M, RIDGE_M = 10 * FT, 15 * FT        # under the roof, above the floor: at the sides and along the ridge
ROOF_M, SIDE_OVER_M, GABLE_OVER_M = 0.15, 0.45, 0.3
DOOR_W, DOOR_H = 5 * FT, 11 * FT
WINDOW_W, SILL_M, HEAD_M, PANE_M = 4 * FT, 3 * FT, 11 * FT, 2 * FT
WINDOW_AT = 0.22 * HUT_W                  # window centres either side of the doors, as cabin.jpg places them
CASING_M, CASING_PROUD_M = 0.12, 0.04      # the openings' casings: width, and how far they stand proud of the boards
REVEAL_M = 0.12

POST_M, POST_PITCH_M, FENCE_M, RAILS_M = 0.12, 2.4, 1.25, (0.45, 1.05)
GATE_M, GATE_FROM_M = 1.4, 2.0            # the gate's opening, and how far along the fence from its end by the house
PUMPKINS, PUMPKIN_PITCH_M, PUMPKIN_RIB = 12, 3.0, 0.17
FENCE_CLEAR_M = 2.5                    # the fence stops this far from the hut's walls, then runs on to them

# the pitch (see the module notes); insets are metres in from the stands' registered outer edge, heights metres above
# the playing surface
GOLD = code("gold")
BOARDING, DOOR = code("changing-room-boarding"), code("changing-room-door")
TIERS = ((3.15, 4.65, 0.6), (1.65, 3.15, 1.9), (0.15, 1.65, 3.2))   # (back inset, front inset, deck top), lowest first
DECK_M, SEAT_M, SEAT_DEPTH_M, RAIL_M = 0.06, 0.45, 0.35, 1.0
SEGMENT_M = 2.5                           # the stands' bays along the outer edge, and their post spacing
BARRIER_IN_M, BARRIER_M = 7.5, 1.1        # the barrier's inset and height: low enough to vault
STAIRS_DEG, STAIR_W = (-140.0, -40.0, 40.0, 140.0), 15 * FT
STAIR_FOOT_M, RISER_M, GOING_M, LANDING_M = 5.6, 0.2, 0.3, 0.9    # the inner flight's foot, out on the walkway
ENTRANCE_W = 10 * FT
HOOP_HEIGHT, HOOP_SPACING, HOOP_ACROSS, HOOP_TUBE_M = 50 * FT, 17 * FT, 5 * FT, 0.06
HOOP_U_M = 250 * FT - 8.0                 # the hoops stand inside the scoring areas, in from the pitch's ends
BLOCK_L, BLOCK_D, BLOCK_EAVE_M, BLOCK_RISE_M = 85 * FT, 18 * FT, 2.7, 1.1
BLOCK_DOORS, DOOR_W_M, DOOR_H_M = 5, 0.9, 2.0                    # four changing rooms and Madam Hooch's office
WINDOW_SILL_M, WINDOW_OFF_M = 1.1, 1.5                         # each room's window onto the pitch, beside its door
PODIUM_U_M, PODIUM_IN_M, PODIUM_SIZE_M, PODIUM_TOP_M, PODIUM_BOARD_M = -12.0, 11.0, 2.4, 2.6, 0.6


class Hut(NamedTuple):
    centre: np.ndarray                    # (x, z)
    right: np.ndarray                     # unit (x, z): to the right as you face the front
    front: np.ndarray                     # unit (x, z): out of the front, towards the castle
    floor: float
    sink: float                           # how far the walls reach below the floor, into the lowest ground under them


def hut_layout(footprint, ground):
    """The hut on White Hound's drawn footprint: its centre and axes, with the long side facing the castle (at the world
    origin) as the front, and a floor just above the ground under it."""
    cx, cz, _, _, axis = fit_rectangle(footprint)
    centre = np.array([cx, cz])
    front = np.array([-axis[1], axis[0]])
    if front @ -centre < 0:
        front = -front
    right = np.array([-front[1], front[0]])          # facing north (0, 1), your right is east (1, 0)
    s, t = np.meshgrid(np.linspace(-0.5, 0.5, 5) * HUT_W, np.linspace(-0.5, 0.5, 4) * HUT_D)
    xz = centre + s.reshape(-1, 1) * right + t.reshape(-1, 1) * front
    g = np.asarray(ground(xz[:, 0], xz[:, 1]), np.float64)
    floor = float(np.median(g)) + 0.1
    return Hut(centre, right, front, floor, floor - float(g.min()) + 0.3)


def world(h, x, z):
    """Hut-local (x, z) to world (x, z)."""
    return h.centre + np.asarray(x)[..., None] * h.right - np.asarray(z)[..., None] * h.front


def toward(m, direction):
    """A flat panel turned to face `direction`."""
    return m if m.nrm[0] @ direction > 0 else Mesh(m.pos, -m.nrm, m.attr, m.idx[:, [0, 2, 1]])


def footprint(h, grow=0.0):
    hw, hd = HUT_W / 2 + grow, HUT_D / 2 + grow
    return Polygon([world(h, x, z) for x, z in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd))])


def hut(h, ground):
    """Hagrid's hut (see the module notes), in world coordinates."""
    hw, hd = HUT_W / 2, HUT_D / 2
    slope = (RIDGE_M - EAVE_M) / hw
    X, Y, Z = np.eye(3)
    gable = [(-hw, -h.sink), (hw, -h.sink), (hw, EAVE_M), (0.0, RIDGE_M), (-hw, EAVE_M)]
    door = [(-DOOR_W / 2, 0.0), (DOOR_W / 2, 0.0), (DOOR_W / 2, DOOR_H), (-DOOR_W / 2, DOOR_H)]
    window = lambda c: [(c - WINDOW_W / 2, SILL_M), (c + WINDOW_W / 2, SILL_M), (c + WINDOW_W / 2, HEAD_M),
                        (c - WINDOW_W / 2, HEAD_M)]
    holes, fills = [door, window(-WINDOW_AT), window(WINDOW_AT)], [PLANKS, GLASS, GLASS]
    side = [(0.0, -h.sink), (HUT_D, -h.sink), (HUT_D, EAVE_M), (0.0, EAVE_M)]
    parts = [pierced(-hd * Z, -X, Y, gable, holes, REVEAL_M, WEATHERBOARD, fills, TIMBER),       # front, facing -z
             pierced(hd * Z, X, Y, gable, holes, REVEAL_M, WEATHERBOARD, fills, TIMBER),         # back, facing +z
             pierced(hw * X + hd * Z, -Z, Y, side, [], 0.0, WEATHERBOARD, []),                   # right, facing +x
             pierced(-hw * X - hd * Z, Z, Y, side, [], 0.0, WEATHERBOARD, [])]                   # left, facing -x
    for s in (1.0, -1.0):                       # sash bars between the face and the glass: square panes, two sashes
        z0, z1 = sorted((s * (hd - 0.09), s * (hd - 0.05)))
        for c in (-WINDOW_AT, WINDOW_AT):
            parts.append(box(c - 0.03, c + 0.03, SILL_M, HEAD_M, z0, z1, SASH))
            for y in np.arange(SILL_M + PANE_M, HEAD_M - 0.01, PANE_M):
                parts.append(box(c - WINDOW_W / 2, c + WINDOW_W / 2, y - 0.03, y + 0.03, z0, z1, SASH))
    for s in (1.0, -1.0):                       # casings proud of the boards round each opening, sills, the door's ironwork
        z0, z1 = sorted((s * hd, s * (hd + CASING_PROUD_M)))
        for x0, x1, y0, y1 in [(-DOOR_W / 2, DOOR_W / 2, 0.0, DOOR_H)] + [
                (c - WINDOW_W / 2, c + WINDOW_W / 2, SILL_M, HEAD_M) for c in (-WINDOW_AT, WINDOW_AT)]:
            w = CASING_M
            parts += [box(x0 - w, x0, y0, y1 + w, z0, z1, TIMBER), box(x1, x1 + w, y0, y1 + w, z0, z1, TIMBER),
                      box(x0 - w, x1 + w, y1, y1 + w, z0, z1, TIMBER)]
            if y0 > 0.0:                        # a sill board, projecting
                zs0, zs1 = sorted((s * hd, s * (hd + 0.1)))
                parts.append(box(x0 - w - 0.04, x1 + w + 0.04, y0 - 0.06, y0, zs0, zs1, TIMBER))
        zi0, zi1 = sorted((s * (hd - REVEAL_M + 0.02), s * (hd - REVEAL_M + 0.04)))   # on the door's face, in its reveal
        for y in (0.6, DOOR_H - 0.6):
            parts.append(box(-DOOR_W / 2 + 0.05, 0.1, y - 0.05, y + 0.05, zi0, zi1, IRON))      # strap hinges
        parts.append(lathe(DOOR_W / 2 - 0.25, s * (hd - REVEAL_M + 0.06), [(0.07, 1.1), (0.07, 1.16)], IRON, 8))
    eave_y, zo = EAVE_M - SIDE_OVER_M * slope, hd + GABLE_OVER_M
    for s in (1.0, -1.0):                       # the two slopes, a slab each: top, underside, fascia, bargeboards
        x1 = s * (hw + SIDE_OVER_M)
        top = [(0.0, RIDGE_M + ROOF_M, -zo), (0.0, RIDGE_M + ROOF_M, zo), (x1, eave_y + ROOF_M, zo), (x1, eave_y + ROOF_M, -zo)]
        under = [(x, y - ROOF_M, z) for x, y, z in top]
        parts += [toward(panel(top, SHINGLES), Y), toward(panel(under, SHINGLES), -Y),
                  toward(panel([under[2], under[3], top[3], top[2]], SHINGLES), s * X)]
        for zs in (1.0, -1.0):
            edge = [p for p in (top + under) if p[2] == zs * zo]
            parts.append(toward(panel([edge[0], edge[1], edge[3], edge[2]], SHINGLES), zs * Z))
    px, pz = -hw + 1.0, -hd + 0.9               # the stovepipe over the front-left corner, with a conical cap
    py = RIDGE_M + ROOF_M - slope * abs(px)
    parts.append(lathe(px, pz, [(0.12, py - 0.3), (0.12, py + 1.5), (0.24, py + 1.5), (0.0, py + 1.75)], IRON, 14))
    gx, gy = hw + SIDE_OVER_M - 0.08, eave_y - 0.08      # gutter along the right eave, its downpipe at the front
    parts.append(box(gx - 0.06, gx + 0.06, gy - 0.05, gy + 0.05, -zo, zo, IRON))
    bx, bz = hw - 0.55, -hd - 0.55
    parts.append(bar((gx, gy, -zo + 0.1), (gx, 1.6, -zo + 0.1), 0.08, IRON))
    parts.append(bar((gx, 1.6, -zo + 0.1), (bx + 0.15, 1.15, bz), 0.08, IRON))
    for bx, bz in ((hw - 0.55, -hd - 0.55), (-hw + 0.55, hd + 0.55)):   # right of the front door; back left
        base = float(ground(*world(h, bx, bz)[:, None])[0]) - h.floor
        profile = [(0.36, base - 0.05), (0.42, base + 0.5), (0.36, base + 1.0)]
        parts.append(lathe(bx, bz, profile, TIMBER, 16))
        for k in (0.25, 0.75):
            y, r = base + k, 0.42 - 0.06 * abs(k - 0.5) / 0.5 + 0.012
            parts.append(lathe(bx, bz, [(r, y - 0.03), (r, y + 0.03)], IRON, 16, cap=False))
    rot, shift = frame(h.centre, h.right, h.floor)
    return [placed(concat(parts), rot, shift)]


# ---------------------------------------------------------------- the garden
def ellipsoid(x, y, z, rx, ry, attr, segments=12):
    a = np.linspace(0.0, math.pi, 7)
    return lathe(x, z, [(rx * math.sin(t), y - ry * math.cos(t)) for t in a], attr, segments, cap=False)


def tree(x, z, y):
    """A broad deciduous tree about 10 m tall: a tapering trunk, three limbs and a crown of leafy masses."""
    parts = [lathe(x, z, [(0.32, y - 0.4), (0.27, y + 1.5), (0.2, y + 4.5), (0.12, y + 7.0)], BARK, 12)]
    for a in (0.3, 2.4, 4.4):
        tip = (x + 2.1 * math.cos(a), y + 6.4, z + 2.1 * math.sin(a))
        parts.append(bar((x, y + 3.4, z), tip, 0.16, BARK))
        parts.append(ellipsoid(tip[0], tip[1] + 0.6, tip[2], 2.3, 1.9, FOLIAGE))
    parts.append(ellipsoid(x, y + 8.2, z, 2.8, 2.2, FOLIAGE))
    return parts


def pumpkin(x, z, y, d, phase):
    """A pumpkin `d` across and about 0.6 d high sitting on y: ten deep ribs, a dimple at the top under a short green
    stem, shaded smoothly from its own surface."""
    th = np.linspace(0.0, 2 * math.pi, 41)
    ph = np.linspace(0.0, math.pi, 11)
    T, P = np.meshgrid(th, ph)                                 # rows from the bottom up
    r = d / 2 * (1.0 - PUMPKIN_RIB * (1.0 - np.cos(10 * (T + phase))) / 2) * np.sin(P)
    ry = 0.31 * d
    dimple = 0.09 * d * np.exp(-((math.pi - P) / 0.55) ** 2)
    pos = np.stack([x + r * np.cos(T), y + ry * (1.0 - np.cos(P)) - dimple, z + r * np.sin(T)], -1).reshape(-1, 3)
    centre = np.array([x, y + ry, z])
    cols = len(th)
    a = (np.arange(len(ph) - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
    idx = np.concatenate([np.column_stack([a, a + cols + 1, a + 1]), np.column_stack([a, a + cols, a + cols + 1])])
    face = np.cross(pos[idx[:, 1]] - pos[idx[:, 0]], pos[idx[:, 2]] - pos[idx[:, 0]])
    flip = np.einsum("ij,ij->i", face, pos[idx].mean(axis=1) - centre) < 0
    idx[flip] = idx[flip][:, [0, 2, 1]]
    face[flip] = -face[flip]
    nrm = np.zeros_like(pos)                                   # its ribs shade from the faces about each vertex
    for k in range(3):
        np.add.at(nrm, idx[:, k], face)
    seam = np.arange(len(ph)) * cols                           # the seam's two columns are one
    nrm[seam] = nrm[seam + cols - 1] = nrm[seam] + nrm[seam + cols - 1]
    nrm[np.linalg.norm(nrm, axis=1) < 1e-12] = (pos - centre)[np.linalg.norm(nrm, axis=1) < 1e-12]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    body = Mesh(pos, nrm, np.tile(np.asarray(PUMPKIN, np.uint8), (len(pos), 1)), idx.astype(np.uint32))
    top = y + 2 * ry - 0.09 * d
    stem = lathe(x, z, [(0.07, top - 0.05), (0.05, top + 0.12), (0.04, top + 0.2)], FOLIAGE, 7)
    return [body, stem]


def pumpkin_leaf(c, y, heading, size):
    """A broad, five-lobed pumpkin leaf lying cupped on the ground at y, its stalk end at c (x, z)."""
    t = np.linspace(0.0, 2 * math.pi, 16, endpoint=False)
    rad = size * (0.75 + 0.25 * np.abs(np.cos(2.5 * t)))
    mid = np.asarray(c) + size * np.array([math.cos(heading), math.sin(heading)])
    rim = np.column_stack([mid[0] + rad * np.cos(t + heading), y + 0.03 + 0.12 * size * (1 + np.sin(t)) / 2,
                           mid[1] + rad * np.sin(t + heading)])
    hub = np.array([mid[0], y + 0.12 * size, mid[1]])
    tris = np.array([[hub, rim[k], rim[(k + 1) % len(rim)]] for k in range(len(rim))])
    return faceted(tris, FOLIAGE, up=True)


def oriented_box(h, c, length, width, base, height, attr, ground):
    """A box `length` along the hut's right and `width` along its front, centred on world (x, z) c, standing on the
    ground there."""
    y = float(ground(*np.asarray(c)[:, None])[0])
    rot, shift = frame(c, h.right, y + base)
    return placed(box(-length / 2, length / 2, 0.0, height, -width / 2, width / 2, attr), rot, shift)


def place(candidates, shape, fits, taken):
    """The first candidate (x, z) where `shape` (a polygon about the origin, in hut axes) fits and clears `taken`."""
    for c in candidates:
        p = affinity.translate(shape, *c)
        if fits.contains(p) and not any(p.intersects(t) for t in taken):
            return np.asarray(c), p
    raise ValueError("no room left in the garden for " + shape.wkt[:40])


def garden(outline, h, ground):
    """Hagrid's garden on White Hound's traced outline (see the module notes), in world coordinates."""
    house, rng = footprint(h), np.random.default_rng(4)
    g = lambda xz: np.asarray(ground(xz[..., 0], xz[..., 1]), np.float64)
    from shapely.ops import nearest_points
    lines = LineString(outline.exterior.coords).difference(house.buffer(FENCE_CLEAR_M))   # clear of the hut's walls
    lines = [l for l in getattr(lines, "geoms", [lines]) if l.length > 3.0]
    for k, l in enumerate(lines):                # and where a line comes up to the hut, it runs on to the wall
        pts = list(l.coords)
        for end in (0, -1):
            if house.distance(Point(*pts[end])) < FENCE_CLEAR_M + 0.5:
                wall = nearest_points(house.exterior, Point(*pts[end]))[0].coords[0]
                pts = [wall] + pts if end == 0 else pts + [wall]
        lines[k] = LineString(pts)
    front_corners = [world(h, s * HUT_W / 2, -HUT_D / 2) for s in (-1.0, 1.0)]
    ends = [(min(np.hypot(*(np.asarray(l.coords[k]) - c)) for c in front_corners), i, k)
            for i, l in enumerate(lines) for k in (0, -1)]
    _, gate_line, gate_end = min(ends)
    far = max(((l.interpolate(t).distance(Point(*h.centre)), i, t) for i, l in enumerate(lines)
               for t in np.linspace(0.0, l.length, 60)))
    parts, inside = [], lambda p, n: outline.contains(Point(*(p + 0.5 * n)))
    for i, line in enumerate(lines):
        L = line.length
        stops = set(np.linspace(0.0, L, int(math.ceil(L / POST_PITCH_M)) + 1).round(3))
        gap = None
        if i == gate_line:
            g0 = GATE_FROM_M if gate_end == 0 else L - GATE_FROM_M - GATE_M
            gap = (g0, g0 + GATE_M)
            stops = {s for s in stops if not gap[0] - 0.6 < s < gap[1] + 0.6} | set(gap)
        stops = sorted(stops)
        tree_at = None
        if i == far[1]:
            tree_at = min(stops, key=lambda s: abs(s - far[2]))
        pts = np.array([line.interpolate(s).coords[0] for s in stops])
        tops = g(pts)
        for s, p, y in zip(stops, pts, tops):
            if s == tree_at:
                parts += tree(p[0], p[1], y)
            else:
                parts.append(bar((p[0], y - 0.3, p[1]), (p[0], y + FENCE_M, p[1]), POST_M, TIMBER))
        for k in range(len(stops) - 1):
            if gap and stops[k] == gap[0] and stops[k + 1] == gap[1]:
                continue
            for r in RAILS_M:
                parts.append(bar((*pts[k][:1], tops[k] + r, pts[k][1]), (*pts[k + 1][:1], tops[k + 1] + r, pts[k + 1][1]),
                                 0.08, TIMBER))
        if gap:                                 # the gate stands open into the garden, hinged on its far post
            hinge = np.asarray(line.interpolate(gap[1]).coords[0])
            along = (np.asarray(line.interpolate(gap[0]).coords[0]) - hinge) / GATE_M
            n = np.array([-along[1], along[0]])
            swing = n if inside(hinge, n) else -n
            leaf = hinge + swing * (GATE_M - 0.1)
            y = float(g(hinge[None])[0])
            for r in (0.25, 1.0):
                parts.append(bar((hinge[0], y + r, hinge[1]), (leaf[0], y + r, leaf[1]), 0.07, TIMBER))
            parts.append(bar((leaf[0], y + 0.2, leaf[1]), (leaf[0], y + 1.1, leaf[1]), 0.07, TIMBER))
            parts.append(bar((hinge[0], y + 0.25, hinge[1]), (leaf[0], y + 1.0, leaf[1]), 0.06, TIMBER))
    fits = outline.difference(house.buffer(1.5)).buffer(-1.2)
    back_door = world(h, 0.0, HUT_D / 2)
    lattice = [back_door - h.front * (2.6 + i * PUMPKIN_PITCH_M * 0.87)
               + h.right * ((j + 0.5 * (i % 2)) * PUMPKIN_PITCH_M) + rng.uniform(-0.5, 0.5, 2)
               for i in range(8) for j in range(-6, 7)]                # offset rows, sprawling
    lattice = [p for p in lattice if fits.contains(Point(*p).buffer(1.3))]
    lattice.sort(key=lambda p: np.hypot(*(p - back_door)) + rng.uniform(0.0, 2.5))
    if len(lattice) < PUMPKINS:
        raise ValueError("no room for a dozen pumpkins behind the hut")
    patch, grown = [], []
    for q in lattice[:PUMPKINS]:
        d = rng.uniform(1.5, 2.0)                                    # "the size of a large boulder"
        y = float(g(q[None])[0])
        parts += pumpkin(q[0], q[1], y - 0.08, d, rng.uniform(0, 1))
        for k in range(5):                                           # its leaves, round its foot
            heading = rng.uniform(0, 2 * math.pi)
            foot = q + (d / 2 - 0.1) * np.array([math.cos(heading), math.sin(heading)])
            parts.append(pumpkin_leaf(foot, float(g(foot[None])[0]), heading + rng.uniform(-0.5, 0.5), rng.uniform(0.35, 0.55)))
        patch.append(Point(*q).buffer(d / 2 + rng.uniform(0.5, 1.1), 10))
        grown.append(q)
    for q in grown:                                                  # short runners curling out from each, leafed
        for k in range(3):
            heading, turn = rng.uniform(0, 2 * math.pi), rng.uniform(-0.9, 0.9)
            way = [q + 1.0 * np.array([math.cos(heading), math.sin(heading)])]
            for _ in range(4):
                heading += turn / 4
                way.append(way[-1] + 0.4 * np.array([math.cos(heading), math.sin(heading)]))
            ys = g(np.array(way)) + 0.05
            for (pa, ya), (pb, yb) in zip(zip(way[:-1], ys[:-1]), zip(way[1:], ys[1:])):
                parts.append(bar((pa[0], ya, pa[1]), (pb[0], yb, pb[1]), 0.04, FOLIAGE))
            parts.append(pumpkin_leaf(way[-1], float(ys[-1]) - 0.05, heading, rng.uniform(0.3, 0.45)))
    soil = unary_union(patch).intersection(fits.buffer(0.6))         # a lobed patch, not a cut polygon
    soil = max(getattr(soil, "geoms", [soil]), key=lambda p: p.area)
    parts.append(lid(soil, lambda x, z: np.asarray(ground(x, z)) + 0.03, EARTH))
    taken = [soil.buffer(0.8)]
    axes = lambda length, width: affinity.rotate(Polygon([(-length / 2, -width / 2), (length / 2, -width / 2),
                                                           (length / 2, width / 2), (-length / 2, width / 2)]),
                                                 math.atan2(h.right[1], h.right[0]), origin=(0, 0), use_radians=True)
    grid = [back_door - h.front * a + h.right * b for a in np.arange(2.0, 45.0, 1.0) for b in np.arange(-25.0, 25.5, 1.0)]
    by_door = sorted(grid, key=lambda p: np.hypot(*(p - back_door)))
    c, beans = place(by_door, axes(7.0, 1.4), fits, taken)            # runner beans up five wigwams of poles
    taken.append(beans.buffer(0.8))
    for k in range(5):
        p = c + h.right * (k - 2) * 1.35
        y = float(g(p[None])[0])
        parts.append(lathe(p[0], p[1], [(0.55, y - 0.05), (0.45, y + 0.6), (0.06, y + 2.3)], FOLIAGE, 10, cap=False))
        parts.append(bar((p[0], y + 2.0, p[1]), (p[0], y + 2.6, p[1]), 0.05, BARK))
    c, bed = place(by_door, axes(6.0, 3.2), fits, taken)              # a potato bed of four earthed-up ridges
    taken.append(bed.buffer(0.8))
    for k in range(4):
        mid = c + h.front * (k - 1.5) * 0.8
        tips = [mid - h.right * 2.8, mid + h.right * 2.8]
        spine = np.array([tips[0] + (tips[1] - tips[0]) * t for t in np.linspace(0.0, 1.0, 6)])
        ys = g(spine)
        for a, b, ya, yb in zip(spine[:-1], spine[1:], ys[:-1], ys[1:]):
            for sgn in (1.0, -1.0):
                o = h.front * 0.3 * sgn
                quad = [(a[0] + o[0], ya - 0.02, a[1] + o[1]), (b[0] + o[0], yb - 0.02, b[1] + o[1]),
                        (b[0], yb + 0.3, b[1]), (a[0], ya + 0.3, a[1])]
                parts.append(toward(panel(quad, EARTH), np.array([o[0], 0.6, o[1]])))
        for p_, y_ in zip(spine[1:-1], ys[1:-1]):                    # the potatoes' haulms along the ridge
            for t_ in (-0.25, 0.25):
                q_ = p_ + (tips[1] - tips[0]) / 5 * t_
                parts.append(lathe(q_[0], q_[1], [(0.3, y_ + 0.22), (0.24, y_ + 0.45), (0.0, y_ + 0.58)], FOLIAGE, 7))
    patch_far = max(np.hypot(*(np.asarray(p.centroid.coords[0]) - back_door)) for p in patch)
    beyond = [p for p in by_door if np.hypot(*(p - back_door)) > patch_far + 3.0]
    c, heap = place(beyond, Point(0, 0).buffer(1.5, 8), fits, taken)  # the compost heap, beyond the pumpkins
    taken.append(heap.buffer(0.8))
    y = float(g(c[None])[0])
    parts.append(lathe(c[0], c[1], [(1.5, y - 0.1), (1.3, y + 0.5), (0.8, y + 0.9), (0.0, y + 1.1)], EARTH, 14))
    by_house = sorted(grid, key=lambda p: house.distance(Point(*p)))
    c, coop = place(by_house, axes(2.0, 1.4), fits, taken)            # the hen-coop, under a lean-to roof
    parts.append(oriented_box(h, c, 2.0, 1.2, 0.0, 1.0, TIMBER, ground))
    y = float(g(c[None])[0])
    p3 = lambda xz, dy: (xz[0], y + dy, xz[1])
    lo, hi = c + h.front * 0.75, c - h.front * 0.75
    roof = [p3(lo - h.right * 1.1, 1.0), p3(lo + h.right * 1.1, 1.0), p3(hi + h.right * 1.1, 1.25), p3(hi - h.right * 1.1, 1.25)]
    parts.append(toward(panel(roof, SHINGLES), np.array([0.0, 1.0, 0.0])))
    return parts


# ---------------------------------------------------------------- the Quidditch pitch
class Pitch(NamedTuple):
    c: np.ndarray                         # centre (x, z)
    a: float                              # semi-axes of the stands' outer edge: along the long axis, and across
    b: float
    along: np.ndarray                     # unit (x, z) of the long axis, towards u > 0 (the west end)
    across: np.ndarray                    # unit (x, z) towards v > 0 (the south side)
    level: float                          # the playing surface

    def at(self, u, v):
        """World (x, z) of pitch coordinates (u, v)."""
        return self.c + np.asarray(u, np.float64)[..., None] * self.along + np.asarray(v, np.float64)[..., None] * self.across

    def normal(self, t):
        """Unit inward normals (u, v) of the outer edge at parameter angles t (radians)."""
        nu, nv = -np.cos(t) / self.a, -np.sin(t) / self.b
        k = np.hypot(nu, nv)
        return nu / k, nv / k

    def edge(self, t, d=0.0):
        """World (x, z) `d` metres in from the outer edge at parameter angle t (radians), along its inward normal."""
        t = np.asarray(t, np.float64)
        nu, nv = self.normal(t)
        return self.at(self.a * np.cos(t) + d * nu, self.b * np.sin(t) + d * nv)

    def inward(self, t):
        nu, nv = self.normal(np.asarray(t, np.float64))
        return np.asarray(nu)[..., None] * self.along + np.asarray(nv)[..., None] * self.across


def pitch_layout(meta):
    ax = math.radians(meta["brg"])
    return Pitch(np.asarray(meta["c"], np.float64), float(meta["a"]), float(meta["b"]),
                 np.array([math.sin(ax), math.cos(ax)]), np.array([-math.cos(ax), math.sin(ax)]), float(meta["level"]))


def bays(p):
    """Parameter angles of the stands' bay ends, evenly spaced about SEGMENT_M apart along the outer edge."""
    t = np.linspace(-math.pi, math.pi, 4001)
    xz = p.edge(t)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xz, axis=0).T))])
    return np.interp(np.linspace(0.0, s[-1], int(round(s[-1] / SEGMENT_M)) + 1), s, t)


def flipped(m):
    return Mesh(m.pos, -m.nrm, m.attr, m.idx[:, [0, 2, 1]])


def pieces_of(geometry):
    return [g for g in getattr(geometry, "geoms", [geometry]) if g.geom_type == "Polygon" and g.area > 1e-3]


def slab(geometry, top, thick, attr, base=None):
    """Flat-topped boards over a footprint: edges, a top and (unless they reach down to `base`) an underside."""
    out = []
    for poly in pieces_of(geometry):
        out.append(prism(poly, top - thick if base is None else base, top, attr))
        if base is None:
            out.append(flipped(lid(poly, top - thick, attr)))
    return out


def y3(xz, y):
    return np.array([xz[0], y, xz[1]])


def stair_frame(p, deg):
    """(origin on the outer edge, unit inward normal, rotation and shift) of the stair at parameter angle `deg`: local x
    inward, y up, z across the stair."""
    t = math.radians(deg)
    o, n = p.edge(t), p.inward(t)
    rot, shift = frame(o, n, 0.0)
    return o, n, rot, shift


def stair(p, deg, ground):
    """A 15 ft stair from the lawn behind the stands up to the top tier (a landing) and down through the tiers to the
    walkway (pitch-stairs-from-lawn), on two timber stringers each side, with handrails."""
    o, n, rot, shift = stair_frame(p, deg)
    side = np.array([-n[1], n[0]])
    gy = lambda r: float(ground(*(o + r * n)[:, None])[0])
    half, top = STAIR_W / 2, p.level + TIERS[-1][2]
    back = TIERS[-1][0]
    parts, steps = [], []
    foot = gy(STAIR_FOOT_M)
    count = max(int(math.ceil((top - foot) / RISER_M)), 1)
    rise, going = (top - foot) / count, (STAIR_FOOT_M - back) / count
    for k in range(count):                        # the inner flight, climbing outwards from the walkway
        r1 = STAIR_FOOT_M - k * going
        steps.append((r1 - going, r1, foot + (k + 1) * rise, rise))
    steps.append((back - LANDING_M, back, top, 0.05))                     # the landing at the top tier's back
    r, y = back - LANDING_M, top
    for _ in range(60):                           # the outer flight, stepping to the lawn behind the stands
        g = gy(r - GOING_M / 2)
        if abs(g - y) <= RISER_M:
            break
        y += math.copysign(RISER_M, g - y)
        steps.append((r - GOING_M, r, y, RISER_M))
        r -= GOING_M
    for r0, r1, yt, h in steps:                   # planked treads; the risers and ends dark, like the stringers
        m = box(r0, r1, yt - h, yt, -half, half, PLANKS)
        attr = m.attr.copy()
        attr[m.nrm[:, 1] < 0.5] = TIMBER
        parts.append(Mesh(m.pos, m.nrm, attr, m.idx))
    outer = steps[-1]
    for w in (-half - 0.07, half + 0.07):        # stringers and handrails
        a, b = (STAIR_FOOT_M, foot), (back, top)
        c = (outer[0], outer[2])
        for (ra, ya), (rb, yb) in ((a, b), ((back - LANDING_M, top), c)):
            if abs(ra - rb) > 0.05:
                parts.append(bar((ra, ya - 0.12, w), (rb, yb - 0.12, w), 0.12, TIMBER))
                parts.append(bar((ra, ya + 0.95, w), (rb, yb + 0.95, w), 0.06, TIMBER))
                for rr, yy in ((ra, ya), (rb, yb)):           # a newel at each end of the handrail, down to the ground
                    parts.append(bar((rr, min(gy(rr), yy) - 0.3, w), (rr, yy + 1.2, w), 0.12, TIMBER))
        for rr in (back - LANDING_M, back):       # posts under the landing
            base = gy(rr) - 0.3
            if top - 0.05 - base > 0.2:
                parts.append(bar((rr, base, w), (rr, top + 0.95, w), 0.12, TIMBER))
    return [placed(concat(parts), rot, shift)]


def stair_cut(p, deg):
    o, n, _, _ = stair_frame(p, deg)
    side = np.array([-n[1], n[0]])
    half = STAIR_W / 2 + 0.15
    return Polygon([o + r * n + w * side for r, w in ((-LANDING_M - 1.0, -half), (-LANDING_M - 1.0, half),
                                                      (STAIR_FOOT_M + 0.2, half), (STAIR_FOOT_M + 0.2, -half))])


def block_rect(p):
    centre = p.at(0.0, p.b - BLOCK_D / 2)
    return centre[0], centre[1], BLOCK_L, BLOCK_D, p.along


def changing_rooms(p, scene):
    """The 85 by 18 ft block at the middle of the south side, interrupting the stands: whitewashed boarding under a
    shingled roof, five doors onto the pitch (four changing rooms and Madam Hooch's office) each with a window beside it,
    and on the lawn side a door
    or, where the bank stands too high behind it, a window to each room. Its floor stands on the highest ground along its
    front onto the pitch."""
    front = p.at(np.linspace(-BLOCK_L / 2, BLOCK_L / 2, 11), np.full(11, p.b - BLOCK_D))
    floor = max(p.level + 0.05, float(np.max(scene.ground(front[:, 0], front[:, 1]))) + 0.02)
    parts = [gabled(block_rect(p), scene.sunk, p.level + BLOCK_EAVE_M, BLOCK_RISE_M, BOARDING, SHINGLES)]
    bay = BLOCK_L / BLOCK_DOORS
    for k in range(BLOCK_DOORS):
        u = (k - (BLOCK_DOORS - 1) / 2) * bay
        rot, shift = frame(p.at(u, p.b - BLOCK_D), -p.across, floor)          # onto the pitch
        parts.append(placed(box(0.0, 0.04, 0.0, DOOR_H_M, -DOOR_W_M / 2, DOOR_W_M / 2, DOOR), rot, shift))
        parts.append(placed(box(0.0, 0.03, WINDOW_SILL_M, WINDOW_SILL_M + 0.7,           # and a window beside it
                                WINDOW_OFF_M - 0.45, WINDOW_OFF_M + 0.45, GLASS), rot, shift))
        out = p.at(u, p.b)
        g = float(scene.ground(*out[:, None])[0])
        rot, shift = frame(out, p.across, floor)                                # onto the lawn
        if g < floor + 0.3:
            parts.append(placed(box(0.0, 0.04, g - floor, DOOR_H_M, -DOOR_W_M / 2, DOOR_W_M / 2, DOOR), rot, shift))
        else:
            sill = max(1.0, g - floor + 0.3)
            if sill + 0.7 < BLOCK_EAVE_M - 0.2:
                parts.append(placed(box(0.0, 0.03, sill, sill + 0.7, -0.45, 0.45, GLASS), rot, shift))
    return parts


def podium(p, ground):
    """The commentator's podium: a planked platform on four timber legs, railed, with a ladder at its back, standing
    clear of the barrier near the middle of the castle side."""
    v0 = -p.b * math.sqrt(1.0 - (PODIUM_U_M / p.a) ** 2) + PODIUM_IN_M
    o = p.at(PODIUM_U_M, v0)
    rot, shift = frame(o, p.along, 0.0)               # local x along the pitch, z towards the south (across)
    h, top = PODIUM_SIZE_M / 2, p.level + PODIUM_TOP_M
    g = lambda x, z: float(ground(*(o + x * p.along + z * p.across)[:, None])[0])
    parts = []
    boards = int(round(PODIUM_SIZE_M / PODIUM_BOARD_M))
    for k in range(boards):                       # boards run along the pitch
        z0 = -h + k * PODIUM_BOARD_M
        parts.append(box(-h, h, top - 0.06, top, z0 + 0.01, z0 + PODIUM_BOARD_M - 0.01, PLANKS))
    corners = [(sx * (h - 0.1), sz * (h - 0.1)) for sx in (-1, 1) for sz in (-1, 1)]
    for x, z in corners:
        parts.append(bar((x, g(x, z) - 0.3, z), (x, top + 1.0, z), 0.14, TIMBER))
    for sz in (-1, 1):                            # cross-bracing between the legs, and the rails
        z = sz * (h - 0.1)
        parts.append(bar((-h + 0.1, g(-h, z) + 0.2, z), (h - 0.1, top - 0.2, z), 0.08, TIMBER))
    for sx in (-1, 1):
        x = sx * (h - 0.1)
        parts.append(bar((x, g(x, -h) + 0.2, -h + 0.1), (x, top - 0.2, h - 0.1), 0.08, TIMBER))
    for (xa, za), (xb, zb) in (((-h, -h), (h, -h)), ((-h, h), (h, h)), ((-h, -h), (-h, h)), ((h, -h), (h, h))):
        parts.append(bar((xa, top + 1.0, za), (xb, top + 1.0, zb), 0.07, TIMBER))
    parts.append(box(-0.4, 0.4, top, top + 1.1, h - 0.45, h - 0.1, PLANKS))      # the commentator's desk, facing the pitch
    for x in (-0.25, 0.25):                       # a ladder up the back, from the stands' side
        parts.append(bar((x, g(x, -h) - 0.1, -h - 0.6), (x, top + 0.9, -h - 0.05), 0.07, TIMBER))
    for k in range(1, int(PODIUM_TOP_M / 0.3)):
        f = k * 0.3 / (PODIUM_TOP_M + 0.1)
        parts.append(bar((-0.25, g(0.0, -h) + k * 0.3, -h - 0.6 + 0.55 * f), (0.25, g(0.0, -h) + k * 0.3, -h - 0.6 + 0.55 * f),
                         0.05, TIMBER))
    return [placed(concat(parts), rot, shift)]


def ring(cx, cy, cz, side, radius, tube, attr, around=32, segments=10):
    """A smooth ring of centre-line `radius` and `tube` radius standing in the vertical plane through (cx, cz) along
    the unit (x, z) `side`."""
    e1, e2 = np.array([side[0], 0.0, side[1]]), np.array([0.0, 1.0, 0.0])
    e3 = np.cross(e1, e2)
    A, B = np.meshgrid(2 * np.pi * np.arange(around) / around, 2 * np.pi * np.arange(segments) / segments, indexing="ij")
    line = np.cos(A)[..., None] * e1 + np.sin(A)[..., None] * e2
    nrm = (np.cos(B)[..., None] * line + np.sin(B)[..., None] * e3).reshape(-1, 3)
    pos = np.array([cx, cy, cz]) + radius * line.reshape(-1, 3) + tube * nrm
    i, j = np.meshgrid(np.arange(around), np.arange(segments), indexing="ij")
    a, b = i * segments + j, ((i + 1) % around) * segments + j
    c, d = ((i + 1) % around) * segments + (j + 1) % segments, i * segments + (j + 1) % segments
    idx = np.concatenate([np.column_stack([a.ravel(), b.ravel(), c.ravel()]), np.column_stack([a.ravel(), c.ravel(), d.ravel()])])
    face = np.cross(pos[idx[:, 1]] - pos[idx[:, 0]], pos[idx[:, 2]] - pos[idx[:, 0]])
    flip = np.einsum("ij,ij->i", face, nrm[idx].sum(axis=1)) < 0
    idx[flip] = idx[flip][:, [0, 2, 1]]
    return Mesh(pos, nrm, np.tile(np.asarray(attr, np.uint8), (len(pos), 1)), idx.astype(np.uint32))


def hoops(p, ground):
    """Three golden hoops at each end, 50 ft high, 17 ft apart and 5 ft across (pitch-goals-equal-height)."""
    parts = []
    for end in (1.0, -1.0):
        for k in (-1.0, 0.0, 1.0):
            x, z = p.at(end * HOOP_U_M, k * HOOP_SPACING)
            base = float(ground(np.array([x]), np.array([z]))[0]) - 0.5
            parts.append(lathe(x, z, [(0.09, base), (0.09, p.level + HOOP_HEIGHT - HOOP_ACROSS)], GOLD, 12, cap=False))
            parts.append(ring(x, p.level + HOOP_HEIGHT - HOOP_ACROSS / 2, z, p.across, HOOP_ACROSS / 2 - HOOP_TUBE_M,
                              HOOP_TUBE_M, GOLD))
    return parts


def pitch(scene):
    """The Quidditch pitch's stands, stairs, barrier, changing rooms, podium and hoops (see the module notes)."""
    p, ground = pitch_layout(scene.meta["pitch"]), scene.ground
    entrance = Polygon([p.at(u, v) for u, v in ((-ENTRANCE_W / 2, -p.b - 8.0), (ENTRANCE_W / 2, -p.b - 8.0),
                                                (ENTRANCE_W / 2, -p.b + BARRIER_IN_M + 1.5),
                                                (-ENTRANCE_W / 2, -p.b + BARRIER_IN_M + 1.5))])
    cx, cz, length, depth, _ = block_rect(p)
    block = Polygon([p.at(u, v) for u, v in ((-length / 2, p.b - depth), (length / 2, p.b - depth), (length / 2, p.b + 2.0),
                                             (-length / 2, p.b + 2.0))])
    stands_cut = unary_union([entrance, block.buffer(0.3, join_style=2)] + [stair_cut(p, d) for d in STAIRS_DEG])
    gap = Polygon([p.at(u, v) for u, v in ((-length / 2, p.b - BARRIER_IN_M - 1.0), (length / 2, p.b - BARRIER_IN_M - 1.0),
                                           (length / 2, p.b - depth), (-length / 2, p.b - depth))])
    barrier_cut = unary_union([entrance, gap])
    clear = lambda xz, cut, margin=0.15: not shapely.contains_xy(cut.buffer(margin), *xz)
    T = bays(p)
    out = []
    for t0, t1 in zip(T[:-1], T[1:]):
        parts = []
        for d0, d1, h in TIERS:                  # the decks, a bench along the back of each, on timber bearers
            top = p.level + h
            quad = Polygon([p.edge(t0, d0), p.edge(t1, d0), p.edge(t1, d1), p.edge(t0, d1)]).difference(stands_cut)
            parts += slab(quad, top, DECK_M, PLANKS)
            s0, s1 = d0 + 0.08, d0 + 0.08 + SEAT_DEPTH_M
            seat = Polygon([p.edge(t0, s0), p.edge(t1, s0), p.edge(t1, s1), p.edge(t0, s1)]).difference(stands_cut)
            parts += slab(seat, top + SEAT_M, 0.05, PLANKS)
            mid = p.edge((t0 + t1) / 2, (s0 + s1) / 2)
            if clear(mid, stands_cut):
                parts.append(bar(y3(mid, top), y3(mid, top + SEAT_M - 0.05), 0.08, TIMBER))
            a, b = p.edge(t0, d0), p.edge(t0, d1)
            if clear(a, stands_cut) and clear(b, stands_cut):
                parts.append(bar(y3(a, top - DECK_M - 0.06), y3(b, top - DECK_M - 0.06), 0.12, TIMBER))
        for d, h in ((TIERS[2][0], TIERS[2][2] + RAIL_M + DECK_M), (TIERS[2][1], TIERS[2][2]), (TIERS[1][1], TIERS[1][2]),
                     (TIERS[0][1], TIERS[0][2])):          # posts at each bay end, the back ones carrying the rail
            xz = p.edge(t0, d)
            g = float(ground(*xz[:, None])[0])
            top = p.level + h - DECK_M
            if clear(xz, stands_cut) and top - g > 0.15:
                parts.append(bar(y3(xz, g - 0.3), y3(xz, top), 0.14, TIMBER))
        a, b = p.edge(t0, TIERS[2][0]), p.edge(t1, TIERS[2][0])
        top = p.level + TIERS[2][2]
        if clear(a, stands_cut) and clear(b, stands_cut):
            for k, y in enumerate((top + RAIL_M, top + RAIL_M / 2)):           # the back rail
                parts.append(bar(y3(a, y), y3(b, y), 0.07 if k else 0.09, TIMBER))
        f = p.edge(t0, TIERS[2][1])
        ga = float(ground(*a[:, None])[0])
        if clear(a, stands_cut) and clear(f, stands_cut) and top - ga > 1.0:   # a raking brace in each frame
            parts.append(bar(y3(a, ga + 0.1), y3(f, top - DECK_M - 0.15), 0.1, TIMBER))
        wall = Polygon([p.edge(t0, BARRIER_IN_M), p.edge(t1, BARRIER_IN_M), p.edge(t1, BARRIER_IN_M + 0.06),
                        p.edge(t0, BARRIER_IN_M + 0.06)]).difference(barrier_cut)
        parts += slab(wall, p.level + BARRIER_M, 0.0, PLANKS, base=scene.sunk)
        post = p.edge(t0, BARRIER_IN_M - 0.08)
        if clear(post, barrier_cut):
            parts.append(bar(y3(post, float(ground(*post[:, None])[0]) - 0.3), y3(post, p.level + BARRIER_M), 0.1, TIMBER))
        if parts:
            out.append(concat(parts))
    for d in STAIRS_DEG:
        out += stair(p, d, ground)
    out += changing_rooms(p, scene) + podium(p, ground) + hoops(p, ground)
    return out


# ---------------------------------------------------------------- the front gates
PALE_STONE, BOAR_STONE = code("pale-stone"), code("boar-stone")
BOAR = Path(__file__).with_name("assets") / FAMILY / "boar.npz"
GATE_OPEN_M = 18 * FT                     # between the pillars: a carriage road's width
PIER_M, PIER_H_M, PLINTH_M, CAP_M = 1.0, 4.5, 0.55, 0.3   # the pillars: square, shaft top above the ground, base, cornice
BOAR_SCALE = 0.7
GATE_H_M, GATE_RISE_M, BALUSTER_M = 3.0, 0.5, 0.13       # the leaves: height at the pillars, rise to the middle, bar pitch


def boar_statue():
    """The winged boar (tools/blender/grounds_boar.py) in its own frame: x forward, y up from its hooves, z its left."""
    z = np.load(BOAR)
    pos = z["pos"].astype(np.float64) * BOAR_SCALE
    return Mesh(pos, z["nrm"].astype(np.float64), np.tile(np.asarray(BOAR_STONE, np.uint8), (len(pos), 1)),
                z["idx"].astype(np.uint32))


def road_centre(scene, a, b):
    """How far along the gap a -> b the road through it is centred: the middle of the stretch inside the road channel,
    or of the gap where no road is mapped."""
    road = sd_bytes(scene.locA[..., CHANNELS["locA"].index("road")])
    gap = float(np.linalg.norm(b - a))
    s = np.linspace(0.0, gap, 121)
    pts = a + s[:, None] * (b - a) / gap
    inside = s[sample(road, scene.local, pts[:, 0], pts[:, 1]) > 0.0]
    return float((inside.min() + inside.max()) / 2) if len(inside) else gap / 2


def gate_line(scene, a, b):
    """Where the boundary wall, as drawn (locB.wall inside -WALL_IN), opens at the mapped gap a -> b: (start, end) of
    the opening on the wall's own centre line, which can lie a few metres off the mapped points, and the wall's
    thickness."""
    field = sd_bytes(decode(scene.tex["locB"])[..., CHANNELS["locB"].index("wall")])
    gap = float(np.linalg.norm(b - a))
    d = (b - a) / gap
    n = np.array([-d[1], d[0]])
    s = np.arange(-12.0, gap + 12.0, 0.1)
    beyond = (s < 0.0) | (s > gap)
    offsets = np.arange(-8.0, 8.01, 0.1)
    at = lambda o, s: sample(field, scene.local, *(a + np.outer(s, d) + o * n).T)
    depth = np.array([at(o, s[beyond]).mean() for o in offsets])
    o = float(offsets[np.argmin(depth)])                     # the wall's centre line runs where it is deepest
    inside = at(o, s) < -WALL_IN
    middle = int(np.argmin(np.abs(s - gap / 2)))
    if inside[middle]:
        raise ValueError("the boundary wall does not open at the mapped gates")
    i0 = middle - int(np.argmax(inside[middle::-1]))         # the wall's last texel either side of the opening
    i1 = middle + int(np.argmax(inside[middle:]))
    thick = float(np.count_nonzero(np.array([(at(q, s[beyond]) < -WALL_IN).mean() > 0.5 for q in offsets])) * 0.1)
    return a + s[i0] * d + o * n, a + s[i1] * d + o * n, max(thick, 0.5)


def gates(scene):
    """The front gates where the boundary wall opens at the mapped gap (see GatesTests): tall stone pillars either side
    of the road with a winged boar on each, facing out of the grounds; high, paired wrought-iron leaves, closed, their
    bars reaching down to the ground; and the wall run on from each pillar into the boundary wall."""
    a, b, thick = gate_line(scene, *(np.asarray(p, np.float64) for p in scene.meta["extras"]["gates"]))
    gap = float(np.linalg.norm(b - a))
    d = (b - a) / gap
    out = np.array([-d[1], d[0]])
    mid = a + d * road_centre(scene, a, b)
    if out @ mid < 0:                              # the boars and the gates' faces look away from the castle
        out = -out
    g = lambda xz: float(scene.ground(*np.asarray(xz, np.float64).reshape(-1, 2).T)[0])
    parts, half = [], GATE_OPEN_M / 2 + PIER_M / 2
    statue = boar_statue()
    for s in (-1.0, 1.0):
        c = mid + s * half * d
        corners = [c + (i * d + j * out) * PIER_M / 2 for i in (-1, 1) for j in (-1, 1)]
        foot = max(g(p) for p in corners)
        top = foot + PIER_H_M
        sunk = min(g(p) for p in corners) - 0.5
        rot, shift = frame(c, out, 0.0)
        r = PIER_M / 2
        pier = [box(-r - 0.1, r + 0.1, sunk, foot + PLINTH_M, -r - 0.1, r + 0.1, WALL_STONE),     # plinth
                box(-r, r, foot + PLINTH_M, top, -r, r, WALL_STONE),                              # shaft
                box(-r - 0.12, r + 0.12, top, top + CAP_M, -r - 0.12, r + 0.12, PALE_STONE),      # cornice
                box(-r + 0.05, r - 0.05, top + CAP_M, top + CAP_M + 0.12, -r + 0.05, r - 0.05, PALE_STONE)]
        parts.append(placed(concat(pier), rot, shift))
        parts.append(placed(statue, rot, np.array([c[0], top + CAP_M + 0.12, c[1]])))
        end = (a if s < 0 else b) + s * d * 0.3    # the wall from the pillar's side into the boundary wall
        face = c + s * d * r
        if (end - face) @ (s * d) > 0.05:
            wall = Polygon([p + q * out * thick / 2 for p, q in ((face, -1), (end, -1), (end, 1), (face, 1))])
            parts += coped(scene, [wall], WALL_HEIGHT_M)
    floor = max(g(mid + t * d) for t in np.linspace(-GATE_OPEN_M / 2, GATE_OPEN_M / 2, 9)) + 0.05
    rot, shift = frame(mid, d, floor)              # local x across the way, z towards the outside
    foot = lambda x: g(mid + x * d) - floor + 0.03                   # the ground under the leaves, raking downhill
    for s in (-1.0, 1.0):                          # each leaf from its pillar to the middle
        x0, x1 = s * (GATE_OPEN_M / 2 - 0.03), s * 0.01
        top = lambda x: GATE_H_M + GATE_RISE_M * (1.0 - abs(x) / (GATE_OPEN_M / 2)) ** 2
        leaf = [bar((x0, foot(x0), 0.0), (x0, top(x0) + 0.05, 0.0), 0.08, IRON),         # hanging stile
                bar((x1, foot(x1), 0.0), (x1, top(x1) + 0.05, 0.0), 0.07, IRON),         # meeting stile
                bar((x0, 1.1, 0.0), (x1, 1.1, 0.0), 0.05, IRON)]                          # middle rail
        xs = np.linspace(x0, x1, 9)
        for xa, xb in zip(xs[:-1], xs[1:]):        # the arched top rail and the raked bottom rail, in short lengths
            leaf.append(bar((xa, top(xa), 0.0), (xb, top(xb), 0.0), 0.06, IRON))
            leaf.append(bar((xa, foot(xa) + 0.1, 0.0), (xb, foot(xb) + 0.1, 0.0), 0.06, IRON))
        n = int(abs(x1 - x0) / BALUSTER_M)
        for k in range(1, n):                      # the bars, each finished with a spear point above the top rail
            x = x0 + (x1 - x0) * k / n
            leaf.append(bar((x, foot(x), 0.0), (x, top(x) + 0.12, 0.0), 0.028, IRON))
            leaf.append(lathe(x, 0.0, [(0.035, top(x) + 0.1), (0.0, top(x) + 0.3)], IRON, 4, cap=False))
        parts.append(placed(concat(leaf), rot, shift))
    return parts


# ---------------------------------------------------------------- walls, hedges and the tomb
COPING_M, COPING_OVER_M = 0.15, 0.06      # the coping course: depth, and overhang either side of the wall
SAFETY_WALL_M, SAFETY_THICK_M, SAFETY_INSET_M, SAFETY_PATH_M = 1.1, 0.45, 0.0, 0.8   # the cliff-edge safety wall (2e)
SAFETY_SKIRT_M = 1.0                           # its foot at most this far below the path's level, off the harbour roof
BRINK_SEARCH_M = 6.0                           # how far in from the mapped edge the cliff's brink may lie
RUIN_THICK_M, RUIN_LOW_M, RUIN_HIGH_M, RUIN_BLOCK_M = 1.0, 0.9, 1.4, 1.2   # the old curtain wall's remnant (5d)
HEDGE_CROWN_M, HEDGE_STEP_M = 0.2, 1.5    # the crown's lumps; the spacing of the rows' cross-sections
HEDGE_HALF_M, HEDGE_SQUARENESS = (0.45, 0.8), 3.2   # half-width; the loaf's squareness (2 an ellipse)
HEDGE_PROFILE_DEG = (0.0, 10.0, 35.0, 65.0, 90.0, 115.0, 145.0, 170.0, 180.0)
GATEWAY_M = 1.0                             # a gateway opens this far either side of a road or path's ink
WALL_JOIN_M = 3.5                           # breaks in a boundary wall narrower than a gateway
HEDGE_JOIN_M, HEDGE_MIN_M = 4.0, 4.0        # breaks in a row narrower than a gateway; the shortest row kept
MARBLE = code("white-marble")


def above(scene, h, reach=0.0):
    """Heights `h` above the ground; with `reach`, above the highest ground within that distance, so a thin wall's top
    stays level across it on a slope."""
    if not reach:
        return lambda x, z: scene.ground(x, z) + h
    offsets = ((0.0, 0.0), (reach, 0.0), (-reach, 0.0), (0.0, reach), (0.0, -reach))
    return lambda x, z: np.max([scene.ground(np.asarray(x) + dx, np.asarray(z) + dz) for dx, dz in offsets], axis=0) + h


def coped(scene, polygons, height, base=None, foot=None, step=None):
    """Walls on the footprints: stone up to `height` less the coping, under a coping course that overhangs them, both
    level across the wall. `base(x, z)` is the level the heights are measured from: by default the highest ground within
    0.5 m. `foot(x, z)` is where the stone starts: by default sunk into the ground, and with `step` sampled at least
    that often along the wall."""
    base = base or above(scene, 0.0, 0.5)
    foot = foot or scene.sunk
    at = lambda h: (lambda x, z: base(x, z) + h)
    out = []
    for poly in polygons:
        s = poly.simplify(0.05, preserve_topology=True)
        poly = s if s.is_valid and s.geom_type == "Polygon" and not s.is_empty else poly
        poly = poly.segmentize(step) if step else poly
        out.append(prism(poly, foot, at(height - COPING_M), WALL_STONE, cap=False))
        for cap in pieces_of(poly.buffer(COPING_OVER_M, join_style=2)):
            cap = cap.segmentize(step) if step else cap
            out.append(prism(cap, at(height - COPING_M), at(height), PALE_STONE))
            out.append(flipped(lid(cap, at(height - COPING_M), PALE_STONE)))
    return out


def ruin(scene, footprint):
    """The wall at the edge of the vegetable patch (5d), the remnant of the former curtain wall: thick and low along
    the middle of the map's broad ink, its top broken between RUIN_LOW_M and RUIN_HIGH_M above the ground."""
    cx, cz, length, _, axis = fit_rectangle(footprint)
    c = np.array([cx, cz])
    line = LineString([c - axis * (length / 2 + 1.0), c + axis * (length / 2 + 1.0)]).intersection(footprint)
    out = []
    for seg in (g for g in getattr(line, "geoms", [line]) if g.geom_type == "LineString" and g.length > 0.5):
        n = max(int(round(seg.length / RUIN_BLOCK_M)), 1)
        for k in range(n):
            a, b = seg.interpolate(k * seg.length / n), seg.interpolate((k + 1) * seg.length / n)
            jitter = (math.sin(12.9898 * k + 78.233 * cx) * 43758.5453) % 1.0
            h = RUIN_LOW_M + (RUIN_HIGH_M - RUIN_LOW_M) * jitter
            out.append(prism(LineString([a, b]).buffer(RUIN_THICK_M / 2, cap_style=2), scene.sunk, above(scene, h),
                             WALL_STONE))
    return out


def safety_wall(scene, footprint):
    """The wall along the edge of the cliff overlooking the lake (2e): a low coped wall on the cliff's brink along the
    lake-side edge of the mapped strip, wherever the ground falls away beyond it, standing SAFETY_WALL_M above the
    ground of the cliff-top path (2f) behind it."""
    from shapely.geometry.polygon import orient
    ring = orient(footprint.simplify(0.3, preserve_topology=True), 1.0).exterior
    s = np.arange(0.0, ring.length, 0.5)
    p = np.array([ring.interpolate(t).coords[0] for t in s])
    ahead = np.array([ring.interpolate((t + 3.0) % ring.length).coords[0] for t in s])     # normals over 6 m of edge
    behind = np.array([ring.interpolate((t - 3.0) % ring.length).coords[0] for t in s])
    q = ahead - behind
    n = np.column_stack([q[:, 1], -q[:, 0]]) / np.maximum(np.hypot(*q.T), 1e-9)[:, None]   # outward
    walk = surface(scene)
    g = lambda xz: walk(xz[:, 0], xz[:, 1])

    def foot(x, z, base=None):                     # on the harbour's roof where it stands over the cut, not its bed; the
        x, z = np.asarray(x, np.float64), np.asarray(z, np.float64)     # lowest within 0.5 m, so it spans the cut's bank
        level = lambda x, z: np.where(walk(x, z) > np.asarray(scene.ground(x, z)) + 0.01, walk(x, z) - 0.5,
                                      scene.sunk(x, z))
        here = level(x, z)
        if base is not None:                       # no deeper than SAFETY_SKIRT_M under its own level: no teeth down the
            here = np.maximum(here, base(x, z) - SAFETY_SKIRT_M)          # sheer face its outer side overhangs
        ring = [(0.0, 0.0)] + [(r * math.cos(t), r * math.sin(t)) for r in (0.25, 0.5)
                               for t in np.arange(8) * math.pi / 4]
        low = np.min([level(x + dx, z + dz) for dx, dz in ring], axis=0)
        rec = scene.meta["extras"].get("harbour")
        if rec is None:
            return here
        m, u = np.asarray(rec["mouth"], np.float64), np.asarray(rec["axis"], np.float64)
        rel = np.stack([x, z], -1) - m                          # only by the roof: elsewhere its foot is under it,
        a, w = rel @ u, rel @ np.array([-u[1], u[0]])           # not hung down the cliff face beyond the brink
        by_roof = (a > PORTAL_M - 2.0) & (a < rec["depth"] + ROOF_PAST_M + 2.0) & (np.abs(w) < SHELL_HALF_M + 2.0)
        return np.where(by_roof, low, here)
    drop = g(p - 1.5 * n) - g(p + 1.5 * n) > 0.3
    if drop.all():
        runs = [np.arange(len(p))]
    elif not drop.any():
        runs = []
    else:                                          # runs of the ring along the drop, walked from a point without one
        order = np.roll(np.arange(len(p)), -int(np.argmin(drop)))
        runs, cur = [], []
        for i in order:
            if drop[i]:
                cur.append(i)
            elif cur:
                runs.append(np.array(cur))
                cur = []
        if cur:
            runs.append(np.array(cur))
    out = []
    for run in runs:
        if len(run) < 8:
            continue
        on, by = brink(scene, p[run], n[run])
        if len(on) < 4:
            continue
        line, path = LineString(on).simplify(0.2), LineString(by)
        if line.length < 2.0:
            continue
        wall = line.buffer(SAFETY_THICK_M / 2, cap_style=2, join_style=2)
        base = along(scene, line, path)
        out += coped(scene, pieces_of(wall), SAFETY_WALL_M, base, lambda x, z: foot(x, z, base), step=0.5)
    return out


def brink(scene, p, n, window=9):
    """(wall line, path line) for a run of the mapped edge `p` with outward normals `n`: each point slid along its
    normal to the cliff's brink, where the steep face (falling at least 1 in 1) begins, taking the face nearest the
    mapped edge and no further than BRINK_SEARCH_M in from it or 2 m out; less the wall's half-thickness and
    SAFETY_INSET_M. The path runs SAFETY_PATH_M further in. Offsets are median-smoothed, and points that would step
    back along the edge are dropped, so the wall runs as one line."""
    from scipy.ndimage import median_filter, uniform_filter1d
    s = np.arange(-BRINK_SEARCH_M, 2.0 + 1e-9, 0.25)
    pts = p[:, None, :] + s[None, :, None] * n[:, None, :]
    g = surface(scene)(pts[..., 0], pts[..., 1])
    steep = -np.gradient(g, s, axis=1) >= 1.0
    starts = steep & ~np.concatenate([np.zeros((len(p), 1), bool), steep[:, :-1]], axis=1)
    found = starts.any(axis=1)
    near = np.where(starts, np.abs(s)[None, :], np.inf).argmin(axis=1)
    edge = np.where(found, s[near], np.nan)
    if not found.any():
        return np.zeros((0, 2)), np.zeros((0, 2))
    edge = np.interp(np.arange(len(p)), np.flatnonzero(found), edge[found])
    off = median_filter(edge, size=window, mode="nearest") - SAFETY_THICK_M / 2 - SAFETY_INSET_M
    smooth = lambda q: uniform_filter1d(q, 2 * window + 1, axis=0, mode="nearest")      # straightens corner knots
    on, by = smooth(p + off[:, None] * n), smooth(p + (off - SAFETY_PATH_M)[:, None] * n)
    along = LineString(p).project
    keep, last = [], -np.inf
    for i, q in enumerate(on):                 # no doubling back where the normals swing round a corner
        a = along(Point(*q))
        if a > last + 0.05:
            keep.append(i)
            last = a
    return on[keep], by[keep]


def along(scene, line, path, step=0.5, window=5):
    """Heights for a wall on `line`, level across it: the ground of the `path` beside it at the same fraction of their
    lengths (median-smoothed over `window` samples), interpolated to each point by its projection onto the line, so the
    stone and the coping that overhangs it rise together without steps."""
    from scipy.ndimage import median_filter
    f = np.linspace(0.0, 1.0, max(int(line.length / step), 2) + 1)
    by = np.array([path.interpolate(v, normalized=True).coords[0] for v in f])
    level = median_filter(surface(scene)(by[:, 0], by[:, 1]), size=window, mode="nearest")

    def height(x, z):
        x, z = np.asarray(x, np.float64), np.asarray(z, np.float64)
        t = shapely.line_locate_point(line, shapely.points(np.ravel(x), np.ravel(z)), normalized=True)
        return np.interp(t, f, level).reshape(np.shape(x))
    return height


def joined(polygons, join, paths=None):
    """The map's pieces of a wall or hedge joined across breaks narrower than `join` (its ink breaks up at bends and
    along thin runs), opened again wherever a road or path crosses them."""
    from shapely.ops import nearest_points
    pieces = pieces_of(unary_union(polygons).buffer(join / 2, quad_segs=4).buffer(-join / 2, quad_segs=4))
    tree, links = shapely.STRtree(pieces), []
    for i, a in enumerate(pieces):               # breaks the closing leaves where the pieces stand staggered
        for j in tree.query(a.buffer(join)):
            b = pieces[j]
            if j > i and a.distance(b) < join:
                width = float(np.clip(2 * min(a.area / a.length, b.area / b.length), 0.5, 1.4))
                links.append(LineString(nearest_points(a, b)).buffer(width / 2, quad_segs=4))
    rows = pieces_of(unary_union(pieces + links))
    if paths is not None and not paths.is_empty:                     # a gateway wherever a road or path crosses a run:
        cut = []                                                     # where its ink splits the run, not runs beside it
        for g in rows:
            for k in pieces_of(paths.intersection(g)):
                if len(pieces_of(g.difference(k.buffer(0.05)))) > 1:
                    cut.append(k.buffer(GATEWAY_M))
        rows = pieces_of(unary_union(rows).difference(unary_union(cut))) if cut else rows
    return rows


def hedge_rows(polygons, paths=None):
    """The hedges' rows (joined, across breaks narrower than HEDGE_JOIN_M), and the fragments that stand alone dropped."""
    out = []
    for g in joined(polygons, HEDGE_JOIN_M, paths):
        e = np.asarray(g.minimum_rotated_rectangle.exterior.coords)
        if max(np.hypot(*(e[1] - e[0])), np.hypot(*(e[2] - e[1]))) > HEDGE_MIN_M:
            out.append(g.simplify(0.25, preserve_topology=True))
    return out


def centrelines(row):
    """[(points (n, 2), half-widths (n,))] along a row's medial axis (the Voronoi diagram of its sides, its spurs
    pruned), each run between the axis's ends and forks, carried on to the row's own ends."""
    import networkx as nx
    from scipy.spatial import Voronoi
    edge = row.segmentize(0.4)
    pts = np.concatenate([np.asarray(r.coords)[:-1] for r in (edge.exterior, *edge.interiors)])
    vor = Voronoi(pts)
    inside = shapely.contains_xy(row, vor.vertices[:, 0], vor.vertices[:, 1])
    radius = shapely.distance(row.boundary, shapely.points(vor.vertices))
    graph = nx.Graph()
    for a, b in vor.ridge_vertices:
        if a >= 0 and b >= 0 and inside[a] and inside[b]:
            graph.add_edge(a, b, w=float(np.hypot(*(vor.vertices[a] - vor.vertices[b]))))
    if not graph.number_of_nodes():
        return []
    spur = 2.5 * float(np.median(radius[list(graph.nodes)]))
    for _ in range(30):                          # prune the spurs running out to the sides' corners
        cut = []
        for leaf in [n for n in graph.nodes if graph.degree(n) == 1]:
            path, length, here = [leaf], 0.0, leaf
            while True:
                nxt = [m for m in graph.neighbors(here) if m not in path]
                if len(nxt) != 1 or graph.degree(here) > 2 and here != leaf:
                    break
                length += graph[here][nxt[0]]["w"]
                path.append(nxt[0])
                here = nxt[0]
                if graph.degree(here) != 2:
                    break
            if graph.degree(here) > 2 and length < spur:
                cut += path[:-1]
        if not cut:
            break
        graph.remove_nodes_from(set(cut))
    from scipy.spatial import cKDTree
    parts = [list(c) for c in nx.connected_components(graph)]
    if len(parts) > 1:                           # the axis breaks where the row narrows: join the pieces up again
        links = nx.Graph()
        for i in range(len(parts)):
            tree = cKDTree(vor.vertices[parts[i]])
            for j in range(i + 1, len(parts)):
                dist, at = tree.query(vor.vertices[parts[j]])
                k = int(np.argmin(dist))
                a, b = parts[i][at[k]], parts[j][k]
                if dist[k] < 3.0 and row.buffer(0.2).contains(LineString(vor.vertices[[a, b]])):
                    links.add_edge(i, j, w=float(dist[k]), ends=(a, b))
        for i, j, data in nx.minimum_spanning_tree(links, weight="w").edges(data=True):
            graph.add_edge(*data["ends"], w=data["w"])
    lines = []
    stops = {n for n in graph.nodes if graph.degree(n) != 2}
    seen = set()
    for s0 in stops or [next(iter(graph.nodes))]:
        for nb in graph.neighbors(s0):
            if (s0, nb) in seen:
                continue
            chain = [s0, nb]
            while chain[-1] not in stops and len(chain) < 100000:
                nxt = [m for m in graph.neighbors(chain[-1]) if m != chain[-2]]
                if not nxt or nxt[0] == chain[0]:
                    break
                chain.append(nxt[0])
            seen.update({(chain[-1], chain[-2]), (chain[0], chain[1])})
            line = LineString(vor.vertices[chain]).simplify(0.3)
            if line.length < 0.5:
                continue
            xy = np.asarray(line.coords)
            for end, nb_ in ((0, 1), (-1, -2)):          # carry an end on to the row's own end
                if graph.degree(chain[end]) == 1:
                    d = xy[end] - xy[nb_]
                    d /= max(np.linalg.norm(d), 1e-9)
                    reach = float(radius[chain[end]])
                    xy = np.concatenate([xy[:1] + d * reach, xy]) if end == 0 else np.concatenate([xy, xy[-1:] + d * reach])
            line = LineString(xy)
            n = max(int(math.ceil(line.length / HEDGE_STEP_M)), 1)
            q = np.array([line.interpolate(t, normalized=True).coords[0] for t in np.linspace(0.0, 1.0, n + 1)])
            half = np.clip(shapely.distance(row.boundary, shapely.points(q)), HEDGE_HALF_M[0], HEDGE_HALF_M[1])
            lines.append((q, half))
    return lines


def hedge_crown(x, z):
    """The hedges' irregular crown: lumps of a few metres and smaller tufts, deterministic in world position."""
    x, z = np.asarray(x, np.float64), np.asarray(z, np.float64)
    return HEDGE_CROWN_M * (0.65 * np.sin(1.1 * x + 0.6 * z) * np.sin(0.8 * z - 0.9 * x + 1.3)
                            + 0.35 * np.sin(1.7 * x - 1.2 * z + 0.4) * np.sin(1.4 * z + 1.1 * x))


def hedges(scene, polygons, paths=None):
    """Field hedges in continuous rows (hedge_rows): along each row's centreline a rounded loaf of foliage, its sides all
    but upright to shoulders low on a domed top (a superellipse), trimmed to about HEDGE_HEIGHT_M under an irregular
    crown (hedge_crown) and shaded smoothly; its ends closed."""
    angles = np.radians(HEDGE_PROFILE_DEG)
    cx, cy = np.cos(angles), np.sin(angles)
    e = 2.0 / HEDGE_SQUARENESS                              # a superellipse: |x|^n + |y|^n = 1
    px = np.sign(cx) * np.abs(cx) ** e
    py = np.abs(cy) ** e
    nx_ = np.sign(px) * np.abs(px) ** (HEDGE_SQUARENESS - 1)   # its normal, before scaling by half-width and height
    ny_ = np.abs(py) ** (HEDGE_SQUARENESS - 1)
    out, built = [], []
    rows = hedge_rows(polygons, paths)
    for row in rows:
        lines = [(q, half) for q, half in centrelines(row) if len(q) > 1]
        shapes = [LineString(q) for q, _ in lines]
        for k, (q, half) in enumerate(lines):
            if shapes[k].length < HEDGE_MIN_M and not any(   # a short run only where it joins a longer one
                    shapes[j].length >= HEDGE_MIN_M and shapes[j].distance(shapes[k]) < 0.6 for j in range(len(lines))):
                continue
            t = np.gradient(q, axis=0)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            side = np.column_stack([-t[:, 1], t[:, 0]])
            ground = np.asarray(scene.ground(q[:, 0], q[:, 1]), np.float64)
            height = HEDGE_HEIGHT_M + hedge_crown(q[:, 0], q[:, 1]) + 0.3          # from 0.3 m into the ground
            xz = q[:, None, :] + (half[:, None] * px[None, :])[..., None] * side[:, None, :]
            y = ground[:, None] - 0.3 + height[:, None] * py[None, :]
            pos = np.concatenate([xz[..., :1], y[..., None], xz[..., 1:]], axis=-1)
            n2 = (nx_[None, :] / half[:, None]) [..., None] * side[:, None, :]
            nrm = np.concatenate([n2[..., :1], (ny_[None, :] / height[:, None])[..., None], n2[..., 1:]], axis=-1)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
            k, m = pos.shape[:2]
            grid = np.arange(k * m).reshape(k, m)
            a_, b_, c_, d_ = grid[:-1, :-1].ravel(), grid[1:, :-1].ravel(), grid[1:, 1:].ravel(), grid[:-1, 1:].ravel()
            idx = np.concatenate([np.column_stack([a_, b_, c_]), np.column_stack([a_, c_, d_])])
            pos, nrm = pos.reshape(-1, 3), nrm.reshape(-1, 3)
            face = np.cross(pos[idx[:, 1]] - pos[idx[:, 0]], pos[idx[:, 2]] - pos[idx[:, 0]])
            flip = np.einsum("ij,ij->i", face, nrm[idx].sum(axis=1)) < 0
            idx[flip] = idx[flip][:, [0, 2, 1]]
            out.append(Mesh(pos, nrm, np.tile(np.asarray(FOLIAGE, np.uint8), (len(pos), 1)), idx.astype(np.uint32)))
            for end, sgn in ((0, -1.0), (-1, 1.0)):            # its ends
                ring = pos.reshape(k, m, 3)[end]
                out.append(toward(panel(ring, FOLIAGE), np.array([sgn * t[end][0], 0.0, sgn * t[end][1]])))
            built.append(LineString(q).buffer(float(half.max()) + 3.0))   # a row stands in sight of these
    covered = unary_union(built) if built else Polygon()
    near_path = paths.buffer(GATEWAY_M) if paths is not None and not paths.is_empty else Polygon()
    for piece in polygons:                       # what no row carries stands as a rounded shrub, not a block or nothing
        for rest in pieces_of(piece.difference(near_path).difference(covered)):
            if rest.area > 0.1:
                c = rest.representative_point()
                r = float(np.clip(math.sqrt(rest.area / math.pi) + 0.3, 0.6, 1.2))
                y = float(np.asarray(scene.ground(np.array([c.x]), np.array([c.y])))[0])
                h = HEDGE_HEIGHT_M + float(hedge_crown(c.x, c.y))
                out.append(lathe(c.x, c.y, [(r, y - 0.3), (r, y + 0.45 * h), (0.85 * r, y + 0.75 * h),
                                            (0.5 * r, y + 0.95 * h), (0.0, y + h)], FOLIAGE, 12, cap=False))
    return out


def tomb(scene):
    """Dumbledore's white marble tomb (tomb-chest): a chest tomb under a low-ridged lid on a moulded plinth
    and two steps, its long axis east-west, standing on the highest ground under it."""
    at = np.asarray(scene.meta["extras"]["tomb"], np.float64)
    east = np.array([1.0, 0.0])
    corners = [at + np.array([sx * 1.7, sz * 1.0]) for sx in (-1, 1) for sz in (-1, 1)]
    g = [float(scene.ground(np.array([c[0]]), np.array([c[1]]))[0]) for c in corners]
    foot, sunk = max(g), min(g) - 0.4
    parts = [box(-1.7, 1.7, sunk, foot + 0.25, -1.0, 1.0, MARBLE),            # the lower step
             box(-1.5, 1.5, foot + 0.25, foot + 0.5, -0.8, 0.8, MARBLE),      # the upper step
             box(-1.28, 1.28, foot + 0.5, foot + 0.62, -0.62, 0.62, MARBLE),   # the plinth's moulding
             box(-1.2, 1.2, foot + 0.62, foot + 1.5, -0.55, 0.55, MARBLE),     # the chest
             box(-1.3, 1.3, foot + 1.5, foot + 1.62, -0.65, 0.65, MARBLE)]     # the lid's slab
    ridge, eave = foot + 1.78, foot + 1.62
    for s in (1.0, -1.0):                         # the lid's two low slopes and its gable ends
        quad = [(-1.25, eave, s * 0.6), (1.25, eave, s * 0.6), (1.25, ridge, 0.0), (-1.25, ridge, 0.0)]
        parts.append(toward(panel(quad, MARBLE), np.array([0.0, 0.6, 0.0]) + s * np.array([0.0, 0.0, 1.0])))
        parts.append(toward(panel([(s * 1.25, eave, -0.6), (s * 1.25, eave, 0.6), (s * 1.25, ridge, 0.0)], MARBLE),
                            s * np.array([1.0, 0.0, 0.0])))
    rot, shift = frame(at, east, 0.0)
    return [placed(concat(parts), rot, shift)]


# ---------------------------------------------------------------- the harbour mouth
HARBOUR_ROCK, IVY = code("harbour-rock"), code("ivy")
PORTAL_M, BACK_IN_M, ROOF_PAST_M = -0.5, 2.0, 1.0  # the portal's lip, out over the water; the back wall's in
                                                    # front of the cut's end; how far the roof runs past it
CROWN_M, SPRING_M, CHANNEL_HALF_M, SHELL_HALF_M, FLOOR_M = 2.0, 0.8, 2.25, 5.0, 2.5  # above / below the lake;
# the channel spans only the cut's wet width: fix_harbour lowers 2 m texels, so its banks ramp up inside the 8 m rectangle
IVY_TO_M, IVY_STRAND_M, IVY_PITCH_M = 0.8, 0.3, 0.2
IVY_SPREAD_M, IVY_ABOVE_M = 1.2, 3.0   # the ivy's reach beyond the opening either side, and above its crown
RELIEF_M = 0.5                                    # the roof's broken surface, in and out of the cliff's line


def vault(w):
    """The vault's underside above the lake across the channel: a low, wide elliptical arch."""
    w = np.clip(np.abs(np.asarray(w, np.float64)) / CHANNEL_HALF_M, 0.0, 1.0)
    return SPRING_M + (CROWN_M - SPRING_M) * np.sqrt(1.0 - w * w)


def harbour_frame(scene):
    """(mouth, unit axis into the cliff, unit across, local -> world (x, z), ground at local (a, w), the roof's surface
    at local (a, w)) of the harbour mouth recorded in meta.extras.harbour. The roof is the uncut cliff's surface drawn
    across the recess, broken a little in and out of its line, and never lower than the arch it covers."""
    rec, lake = scene.meta["extras"]["harbour"], float(scene.meta["lake"])
    m, u = np.asarray(rec["mouth"], np.float64), np.asarray(rec["axis"], np.float64)
    side = np.array([-u[1], u[0]])
    at = lambda a, w: m + np.asarray(a)[..., None] * u + np.asarray(w)[..., None] * side
    g = lambda a, w: np.asarray(scene.ground(*np.moveaxis(at(a, w), -1, 0)), np.float64)
    beyond = SHELL_HALF_M + 0.5                        # the uncut cliff either side, which the roof is drawn between

    def top(a, w):
        a, w = np.broadcast_arrays(np.asarray(a, np.float64), np.asarray(w, np.float64))
        f = (w + beyond) / (2 * beyond)
        cliff = (1 - f) * g(a, -beyond + 0 * w) + f * g(a, beyond + 0 * w)
        bulge = 1.0 - np.minimum((w / SHELL_HALF_M) ** 2, 1.0)  # fades to nothing where the roof meets the uncut cliff
        rough = RELIEF_M * bulge * (0.6 * np.sin(1.4 * w + 0.3 * a) + 0.4 * np.sin(2.9 * w - 0.5 * a + 1.1))   # gullies
        t = np.maximum(g(a, w), cliff) + 0.05 + rough
        return np.where(np.abs(w) < CHANNEL_HALF_M + 0.5, np.maximum(t, lake + vault(w) + 0.6), t)
    return m, u, side, at, g, top


def surface(scene):
    """The ground as people walk on it: the terrain, except over the harbour mouth, where the roof's rock carries the
    cliff top across the recess cut under it."""
    if scene.meta["extras"].get("harbour") is None:
        return scene.ground
    rec = scene.meta["extras"]["harbour"]
    m, u, side, at, g, top = harbour_frame(scene)
    a1 = rec["depth"] + ROOF_PAST_M

    def height(x, z):
        x, z = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(z, np.float64))
        rel = np.stack([x, z], -1) - m
        a, w = rel @ u, rel @ side
        roofed = (a >= PORTAL_M) & (a <= a1) & (np.abs(w) <= SHELL_HALF_M)
        out = np.asarray(scene.ground(x, z), np.float64)
        if roofed.any():
            out = np.where(roofed, top(np.where(roofed, a, PORTAL_M), np.where(roofed, w, 0.0)), out)
        return out
    return height


def harbour(scene):
    """The exterior mouth of the underground harbour (harbour-local-cutout, harbour-heads-down), over the recess that
    fix_harbour.py cut (meta.extras.harbour): a block of rock roofing the recess back to the cliff's own surface, a low,
    wide arch under it whose crown stands CROWN_M above the water, the channel's rock walls and a closed back wall
    short of the cut's end (where its bed ramps up), so there is no tunnel beyond; and a curtain of ivy hanging over the opening to IVY_TO_M
    above the water, through which the boats pass with heads bent. Local frame: x into the cliff from the waterline,
    y up, z across."""
    rec = scene.meta["extras"].get("harbour")
    if rec is None:
        return []
    lake = float(scene.meta["lake"])
    m, u, side, at, g, top = harbour_frame(scene)
    a0, back, a1 = PORTAL_M, rec["depth"] - BACK_IN_M, rec["depth"] + ROOF_PAST_M

    A, W = np.meshgrid(np.linspace(a0, a1, int(math.ceil((a1 - a0) / 0.5)) + 1),
                       np.linspace(-SHELL_HALF_M, SHELL_HALF_M, int(2 * SHELL_HALF_M / 0.5) + 1), indexing="ij")
    T = top(A, W)

    def sheet(Y, a, w):
        """Triangles of a grid of heights Y over local (a, w), in local (x, y, z)."""
        P = np.stack([a, Y, w], -1)
        q = [P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]]
        return np.concatenate([np.stack([q[0], q[1], q[2]], -2).reshape(-1, 3, 3),
                               np.stack([q[0], q[2], q[3]], -2).reshape(-1, 3, 3)])

    P = np.stack([A, T, W], -1)                                                    # the roof, as the cliff's surface,
    nrm = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))                 # shaded smoothly as the cliff is
    nrm[nrm[..., 1] < 0] *= -1
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
    grid = np.arange(P.shape[0] * P.shape[1]).reshape(P.shape[:2])
    q = [grid[:-1, :-1].ravel(), grid[1:, :-1].ravel(), grid[1:, 1:].ravel(), grid[:-1, 1:].ravel()]
    idx = np.concatenate([np.column_stack([q[0], q[1], q[2]]), np.column_stack([q[0], q[2], q[3]])])
    pos, nrm = P.reshape(-1, 3), nrm.reshape(-1, 3)
    face = np.cross(pos[idx[:, 1]] - pos[idx[:, 0]], pos[idx[:, 2]] - pos[idx[:, 0]])
    flip = face[:, 1] < 0
    idx[flip] = idx[flip][:, [0, 2, 1]]
    parts = [Mesh(pos, nrm, np.tile(np.asarray(HARBOUR_ROCK, np.uint8), (len(pos), 1)), idx.astype(np.uint32))]
    Av, Wv = np.meshgrid(np.linspace(a0, back, int(math.ceil((back - a0) / 0.5)) + 1),
                         np.linspace(-CHANNEL_HALF_M, CHANNEL_HALF_M, 17), indexing="ij")
    parts.append(flipped(faceted(sheet(lake + vault(Wv), Av, Wv), HARBOUR_ROCK, up=True)))    # the vault, facing down
    floor = lake - FLOOR_M
    for s in (1.0, -1.0):                                                          # the channel's walls
        w = s * CHANNEL_HALF_M
        quad = [(a0, floor, w), (back, floor, w), (back, lake + SPRING_M, w), (a0, lake + SPRING_M, w)]
        parts.append(toward(panel(quad, HARBOUR_ROCK), np.array([0.0, 0.0, -s])))
        a = A[:, 0]                                                                # and the block's closed sides
        sunk = g(a, s * SHELL_HALF_M + 0 * a) - 1.0
        tri = [[(a[i], sunk[i], s * SHELL_HALF_M), (a[i + 1], sunk[i + 1], s * SHELL_HALF_M),
                (a[i + 1], T[i + 1, -1 if s > 0 else 0], s * SHELL_HALF_M)] for i in range(len(a) - 1)]
        tri += [[(a[i], sunk[i], s * SHELL_HALF_M), (a[i + 1], T[i + 1, -1 if s > 0 else 0], s * SHELL_HALF_M),
                 (a[i], T[i, -1 if s > 0 else 0], s * SHELL_HALF_M)] for i in range(len(a) - 1)]
        parts.append(toward(faceted(np.array(tri), HARBOUR_ROCK), np.array([0.0, 0.0, s])))
    ws = np.linspace(-CHANNEL_HALF_M, CHANNEL_HALF_M, 17)                          # the closed back, under the vault
    parts.append(toward(panel([(back, floor, CHANNEL_HALF_M), (back, floor, -CHANNEL_HALF_M)]
                              + [(back, lake + float(vault(w)), w) for w in ws], HARBOUR_ROCK),
                        np.array([-1.0, 0.0, 0.0])))
    face = []                                                                      # the portal's face round the arch
    wf = np.linspace(-SHELL_HALF_M, SHELL_HALF_M, int(2 * SHELL_HALF_M / 0.25) + 1)
    tf = top(np.full_like(wf, a0), wf)
    low = np.where(np.abs(wf) <= CHANNEL_HALF_M, lake + vault(wf), np.minimum(g(np.full_like(wf, a0), wf) - 1.0, floor))
    for i in range(len(wf) - 1):
        face.append([(a0, low[i], wf[i]), (a0, low[i + 1], wf[i + 1]), (a0, tf[i + 1], wf[i + 1])])
        face.append([(a0, low[i], wf[i]), (a0, tf[i + 1], wf[i + 1]), (a0, tf[i], wf[i])])
    parts.append(toward(faceted(np.array(face), HARBOUR_ROCK), np.array([-1.0, 0.0, 0.0])))
    ivy = []                                    # a curtain of ivy over the opening, draping down the face round it
    ga = np.arange(a0, a1 + 1e-9, 0.25)
    for k, w in enumerate(np.arange(-CHANNEL_HALF_M - IVY_SPREAD_M + IVY_STRAND_M / 2, CHANNEL_HALF_M + IVY_SPREAD_M,
                                    IVY_PITCH_M)):
        h1, h2, h3 = (((math.sin(12.9898 * k + c) * 43758.5453) % 1.0) for c in (4.1, 9.7, 2.3))
        inside = abs(w) < CHANNEL_HALF_M - 0.1
        bottom = lake + (IVY_TO_M + 0.35 * h1 if inside else SPRING_M + 0.3 + 1.2 * h1)
        top_y = lake + CROWN_M + IVY_ABOVE_M * (0.5 + 0.5 * h3)
        face_at = top(ga, np.full_like(ga, w))
        width = IVY_STRAND_M * (0.8 + 0.6 * h2)
        ys = np.arange(top_y, bottom - 1e-9, -0.4)
        if len(ys) < 2:
            continue
        xs = []
        for y in ys:                            # on the face where it stands this high; under the lintel, hanging free
            j = np.flatnonzero(face_at >= y)
            if not len(j) or j[0] == 0:
                at_ = a0
            else:                               # between the samples either side of the height, not stepped
                i0 = j[0] - 1
                f = (y - face_at[i0]) / max(face_at[j[0]] - face_at[i0], 1e-6)
                at_ = ga[i0] + f * (ga[j[0]] - ga[i0])
            xs.append(at_ - 0.15 - 0.1 * h3)
        for (xa, ya), (xb, yb) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
            ivy.append(toward(panel([(xb, yb, w - width / 2), (xb, yb, w + width / 2), (xa, ya, w + width / 2),
                                     (xa, ya, w - width / 2)], IVY), np.array([-1.0, 0.0, 0.0])))
    rot, shift = frame(m, u, 0.0)
    return [placed(concat(parts), rot, shift), placed(concat(ivy), rot, shift)]


# ---------------------------------------------------------------- bridges
ARCH_SHADOW, RUBBLE, DECK = code("arch-shadow"), WALL_STONE, code("bridge-deck")
CROSSING_GAP_M, ROAD_DENSITY = 25.0, 0.5            # the widest road between two pieces of a stream, and road ink
DECK_HALF_M, PARAPET_THICK_M, PARAPET_M = (1.2, 5.0), 0.45, 1.05   # the deck's clear half-width range; parapets
RING_M, COVER_M, ABUT_M = 0.22, 0.3, 1.0             # the arch ring; the stone over it, under the deck; wall past it
SPAN_M, RISE = (1.2, 6.5), (0.45, 1.1)               # the arch's span and rise ranges
FACE_REACH_M = 4.0                                  # how far past the road's edge a face may stand, out to the burn
RAMP_GRADE = 7.0                                    # a raised deck's approaches: 1 in RAMP_GRADE


def road_ink(scene):
    """The road and path ink (locA's road density over ROAD_DENSITY), as world (x, z) points of its texel centres."""
    road = scene.locA[..., CHANNELS["locA"].index("road")] / 255.0
    e, n = scene.local.to_world(*np.mgrid[0:scene.local.ny, 0:scene.local.nx])
    return road, np.column_stack([e[road > ROAD_DENSITY], n[road > ROAD_DENSITY]])


def crossings(scene):
    """[(centre (x, z), unit road axis, road half-width, [the burn's end on either side of the road])] where a road
    runs between two pieces of a burn: the build keeps roads dry, so each crossing shows in the data as a burn broken
    by a road."""
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    road, ink = road_ink(scene)
    stream = sd_bytes(decode(scene.tex["locB"])[..., CHANNELS["locB"].index("stream")])
    e, n = scene.local.to_world(*np.mgrid[0:scene.local.ny, 0:scene.local.nx])
    lab, k = ndi.label(stream < 0, np.ones((3, 3), bool))
    pieces = [np.column_stack([e[lab == c], n[lab == c]]) for c in range(1, k + 1)]
    inked = cKDTree(ink)
    out = []
    for a in range(k):
        for b in range(a + 1, k):
            dist, at = cKDTree(pieces[b]).query(pieces[a])
            j = int(np.argmin(dist))
            if dist[j] > CROSSING_GAP_M:
                continue
            p, q = pieces[a][j], pieces[b][at[j]]
            line = p + np.linspace(0.0, 1.0, 60)[:, None] * (q - p)
            on = sample(road, scene.local, line[:, 0], line[:, 1]) >= ROAD_DENSITY
            if not on.any():
                continue
            c = line[on].mean(axis=0)
            near = ink[inked.query_ball_point(c, 8.0)]
            axis = np.linalg.svd(near - near.mean(axis=0))[2][0]
            across = (ink[inked.query_ball_point(c, 6.0)] - c) @ np.array([-axis[1], axis[0]])
            out.append((c, axis, float(np.percentile(np.abs(across), 90)), [p, q]))
    return out


def road_ink_polygon(scene):
    """All the road and path ink, as a polygon of its texels' squares."""
    road, _ = road_ink(scene)
    return unary_union(field_polygons(-road, scene.local, -ROAD_DENSITY))


def ink_polygon(scene, c, reach):
    """The road ink within `reach` of c as a polygon of its texels."""
    road, _ = road_ink(scene)
    i, j = scene.local.to_texel(np.array([c[0]]), np.array([c[1]]))
    span = int(reach / 1.5) + 2
    squares = []
    for r in range(max(int(i[0]) - span, 0), min(int(i[0]) + span, scene.local.ny)):
        for q in range(max(int(j[0]) - span, 0), min(int(j[0]) + span, scene.local.nx)):
            if road[r, q] > ROAD_DENSITY:
                corners = [scene.local.to_world(r + dr, q + dq) for dr, dq in ((-.5, -.5), (-.5, .5), (.5, .5), (.5, -.5))]
                squares.append(Polygon([(float(x), float(z)) for x, z in corners]))
    return unary_union(squares) if squares else Polygon()


def upright(outline, z, attr, facing):
    """A flat polygon (local (x, y), holes and notches allowed) standing in the plane z, facing +z or -z."""
    import shapely
    tris = np.array([np.asarray(t.exterior.coords)[:3] for t in shapely.constrained_delaunay_triangles(outline).geoms])
    pos = np.concatenate([tris, np.full(tris.shape[:2] + (1,), z)], axis=-1)
    return toward(faceted(pos, attr), np.array([0.0, 0.0, facing]))


def burn_run(stream, scene, at, x0, z, reach=10.0):
    """(lo, hi) along the road of the burn's water in a strip just outside a face at z, the run nearest x0; or None."""
    xs = np.arange(-reach, reach + 1e-9, 0.25)
    wet = np.zeros_like(xs, bool)
    for off in (0.75, 1.5):
        wet |= sample(stream, scene.local, *np.moveaxis(at(xs, np.full_like(xs, z + np.sign(z) * off)), -1, 0)) < 0
    if not wet.any():
        return None
    from scipy import ndimage as ndi
    lab, _ = ndi.label(wet)
    k = lab[np.flatnonzero(wet)[np.argmin(np.abs(xs[wet] - x0))]]
    run = xs[lab == k]
    return float(run.min()), float(run.max())


def arch_opening(x0, half, spring, radius, centre, floor):
    """The opening beneath a segmental arch in a bridge's face (x along the road, y up): the arch, a circle of `radius`
    about (x0, centre), from its springing at (x0 - half, spring) over to its springing at (x0 + half, spring), then
    back along the bed beneath it, floor = (xs from x0 - half to x0 + half, the bed's heights there, none above the
    springing)."""
    th = np.linspace(-math.asin(min(half / radius, 1.0)), math.asin(min(half / radius, 1.0)), 15)
    soffit = [(x0 + radius * math.sin(t), centre + radius * math.cos(t)) for t in th]
    # exactly on the springing: computed, the ends miss it by an ulp and can cross a bed lying there
    soffit[0], soffit[-1] = (x0 - half, spring), (x0 + half, spring)
    return Polygon(soffit + list(zip(floor[0][::-1], floor[1][::-1])))


def bridge(scene, c, axis, half_width, ends, stream=None):
    """One small rubble-stone bridge carrying the road over a burn: a spandrel wall at each face, where the burn leaves
    the road's bank (no further than FACE_REACH_M past the road's edge), rising into a coped parapet; in each face a
    segmental arch over the burn's water there, ringed in pale stone over the dark beneath (a skew bridge where the
    burn crosses obliquely); and between the faces a deck, humped where the road lies too low over the water for the
    arches and eased back down to it at 1 in RAMP_GRADE. The walls stop short of any path leaving the road, so each
    parapet runs unbroken. Local frame: x along the road, z across it, y up."""
    side = np.array([-axis[1], axis[0]])
    at = lambda x, z: c + np.asarray(x)[..., None] * axis + np.asarray(z)[..., None] * side
    g = lambda x, z: np.asarray(scene.ground(*np.moveaxis(at(x, z), -1, 0)), np.float64)
    road_density = scene.locA[..., CHANNELS["locA"].index("road")] / 255.0
    zs = np.arange(0.0, 8.01, 0.25)
    extent, face, arch = {}, {}, {}
    for s in (1.0, -1.0):                       # the road's edge each side: where its ink ends across it
        edge = []
        for x in (-1.5, 0.0, 1.5):
            d = sample(road_density, scene.local, *np.moveaxis(at(np.full_like(zs, x), s * zs), -1, 0))
            edge.append(zs[np.argmax(d < ROAD_DENSITY)] if (d < ROAD_DENSITY).any() else zs[-1])
        extent[s] = float(np.clip(max(edge), *DECK_HALF_M)) + 0.1
        mine = [e for e in ends if float((e - c) @ side) * s > 0]
        z0 = abs(float((mine[0] - c) @ side)) if mine else 0.0
        face[s] = float(np.clip(z0, extent[s] + PARAPET_THICK_M, extent[s] + FACE_REACH_M))
        x_end = float((mine[0] - c) @ axis) if mine else 0.0
        run = burn_run(stream, scene, at, x_end, s * face[s]) if stream is not None else None
        lo, hi = run or (x_end - 0.8, x_end + 0.8)
        span = float(np.clip(hi - lo + 0.5, *SPAN_M))
        x0, half = 0.5 * (lo + hi), span / 2
        xa = np.linspace(x0 - half, x0 + half, 9)
        bank = g(xa, np.full_like(xa, s * face[s]))
        spring = float(np.median(bank)) + 0.05             # on the banks; the opening runs down to the channel's bed
        rise = float(np.clip(0.22 * span, *RISE))
        radius = (half ** 2 + rise ** 2) / (2 * rise)
        arch[s] = (x0, half, spring, rise, radius, spring + rise - radius, (xa, np.minimum(bank - 0.15, spring)))
    xs = np.arange(-40.0, 40.01, 0.5)
    a_lo = min(a[0] - a[1] - RING_M for a in arch.values())
    a_hi = max(a[0] + a[1] + RING_M for a in arch.values())
    over = lambda x: np.maximum(np.maximum(a_lo - np.asarray(x), np.asarray(x) - a_hi), 0.0)   # beyond the arches
    road = lambda x, z: g(x, np.clip(z, -extent[-1.0], extent[1.0]))   # the road's own cross-fall, carried to the walls
    hump = max(0.0, *(a[2] + a[3] + RING_M + COVER_M - float(road(np.array([a[0]]), np.array([s * extent[s]]))[0])
                      for s, a in arch.items()))
    bump = lambda x: np.maximum(hump - over(x) / RAMP_GRADE, 0.0)
    keep = (over(xs) <= ABUT_M) | (bump(xs) > 0.02)
    i0, i1 = np.flatnonzero(keep)[[0, -1]]
    deck_x = xs[max(i0 - 1, 0): i1 + 2]
    surface_at = lambda x, z: road(np.asarray(x, np.float64), np.asarray(z, np.float64)) + bump(x)
    walled = (over(xs) <= ABUT_M) | (bump(xs) > 0.25)             # parapets over the arches and wherever it stands proud
    w0, w1 = xs[np.flatnonzero(walled)[[0, -1]]]
    other = ink_polygon(scene, c, 30.0)                 # paths and bends leaving it: the walls stop short of them
    rect = Polygon([at(x, z) for x, z in ((-40, -extent[-1.0]), (40, -extent[-1.0]), (40, extent[1.0]), (-40, extent[1.0]))])
    beyond = max(face.values()) + 1.0               # a path leaves the road: ink reaching past the faces, not the
    other = unary_union([g for g in pieces_of(other.difference(rect.buffer(0.3)))   # road's own ragged edge
                         if max(abs((np.asarray(g.exterior.coords) - c) @ side)) > beyond])
    other = other.buffer(0.5) if not other.is_empty else other
    for s in (1.0, -1.0):                           # the walls stop short of it, but never while the deck stands proud
        line = lambda x: Point(*at(x, s * (face[s] - PARAPET_THICK_M / 2)))
        for x in np.arange(a_hi, w1 + 0.25, 0.25):
            if not other.is_empty and other.intersects(line(x)):
                end = max(x - 0.5, a_hi + 0.3)
                w1 = end if bump(np.array([end]))[0] <= 0.25 else w1
                break
        for x in np.arange(a_lo, w0 - 0.25, -0.25):
            if not other.is_empty and other.intersects(line(x)):
                end = min(x + 0.5, a_lo - 0.3)
                w0 = end if bump(np.array([end]))[0] <= 0.25 else w0
                break
    wx = np.unique(np.concatenate([np.arange(w0, w1, 0.5), [w1]]))
    parts = []
    for s in (1.0, -1.0):
        x0, half, spring, rise, radius, centre, floor = arch[s]
        deck_at = lambda x: surface_at(x, np.full_like(np.asarray(x, np.float64), s * extent[s]))
        th = np.linspace(-math.asin(min(half / radius, 1.0)), math.asin(min(half / radius, 1.0)), 15)
        zo, zi = s * face[s], s * (face[s] - PARAPET_THICK_M)
        foot = np.minimum(g(wx, np.full_like(wx, zo)), g(wx, np.full_like(wx, zi))) - 0.5
        opening = arch_opening(x0, half, spring, radius, centre, floor)         # down to the bed along the face
        rail = deck_at(wx) + PARAPET_M - COPING_M
        wall = Polygon(list(zip(wx, foot)) + list(zip(wx[::-1], rail[::-1]))).buffer(0)
        parts.append(upright(wall.difference(opening), zo, RUBBLE, s))             # the face and its parapet
        back = Polygon(list(zip(wx, deck_at(wx) - 0.3)) + list(zip(wx[::-1], rail[::-1]))).buffer(0)
        parts.append(upright(back, zi, RUBBLE, -s))                               # the parapet's back, over the deck
        for x, f in ((wx[0], -1.0), (wx[-1], 1.0)):                               # the walls' ends
            y0, y1 = float(np.interp(x, wx, foot)), float(np.interp(x, wx, rail))
            parts.append(toward(panel([(x, y0, zi), (x, y0, zo), (x, y1, zo), (x, y1, zi)], RUBBLE), np.array([f, 0.0, 0.0])))
        parts += [toward(panel([(x0 + rr * math.sin(t), centre + rr * math.cos(t), zo + s * 0.02) for rr, t in
                                ((radius, ta), (radius + RING_M, ta), (radius + RING_M, tb), (radius, tb))], PALE_STONE),
                         np.array([0.0, 0.0, s])) for ta, tb in zip(th[:-1], th[1:])]   # the arch ring's voussoirs
        depth = zo - s * 0.2                                                      # the dark under the arch
        parts.append(upright(opening, depth, ARCH_SHADOW, s))
        soffit = [(x0 + radius * math.sin(t), centre + radius * math.cos(t)) for t in th]
        for (xa_, ya), (xb_, yb) in zip(soffit[:-1], soffit[1:]):
            parts.append(toward(panel([(xa_, ya, zo), (xb_, yb, zo), (xb_, yb, depth), (xa_, ya, depth)], ARCH_SHADOW),
                                np.array([x0 - 0.5 * (xa_ + xb_), centre - 0.5 * (ya + yb), 0.0])))
        for x, y, f in ((x0 - half, floor[1][0], 1.0), (x0 + half, floor[1][-1], -1.0)):   # its jambs, to the bed
            if spring - y > 0.02:
                parts.append(toward(panel([(x, y, zo), (x, spring, zo), (x, spring, depth), (x, y, depth)],
                                          ARCH_SHADOW), np.array([f, 0.0, 0.0])))
        cap = Polygon([(wx[0] - COPING_OVER_M, zi - s * COPING_OVER_M), (wx[-1] + COPING_OVER_M, zi - s * COPING_OVER_M),
                       (wx[-1] + COPING_OVER_M, zo + s * COPING_OVER_M), (wx[0] - COPING_OVER_M, zo + s * COPING_OVER_M)])
        cap = cap.segmentize(0.5)
        lift = lambda h: (lambda x, z: deck_at(np.asarray(x, np.float64)) + h + 0.0 * np.asarray(z, np.float64))
        parts.append(prism(cap, lift(PARAPET_M - COPING_M), lift(PARAPET_M), PALE_STONE))
        parts.append(flipped(lid(cap, lift(PARAPET_M - COPING_M), PALE_STONE)))
    def sheet(xs_, zs_):
        X, Z = np.meshgrid(xs_, zs_, indexing="ij")
        P = np.stack([X, surface_at(X, Z) + 0.03, Z], -1)
        q = [P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]]
        return faceted(np.concatenate([np.stack([q[0], q[1], q[2]], -2).reshape(-1, 3, 3),
                                       np.stack([q[0], q[2], q[3]], -2).reshape(-1, 3, 3)]), DECK, up=True)
    zl, zr = -extent[-1.0], extent[1.0]
    parts.append(sheet(deck_x, np.linspace(zl, zr, max(int((zr - zl) / 1.0), 1) + 1)))   # the deck over the road
    inside = np.unique(np.concatenate([[wx[0]], deck_x[(deck_x >= wx[0]) & (deck_x <= wx[-1])], [wx[-1]]]))
    parts.append(sheet(inside, np.array([zr, face[1.0] - PARAPET_THICK_M])))    # and its verges, between the walls
    parts.append(sheet(inside, np.array([-(face[-1.0] - PARAPET_THICK_M), zl])))
    for z, f in ((zl, -1.0), (zr, 1.0)):                                          # its edges, down to the ground
        ys = surface_at(deck_x, np.full_like(deck_x, z))
        gz = g(deck_x, np.full_like(deck_x, z)) - 0.3
        skirt = [panel([(xa_, ya, z), (xb_, yb, z), (xb_, gb, z), (xa_, ga, z)], RUBBLE)
                 for xa_, xb_, ya, yb, ga, gb in zip(deck_x[:-1], deck_x[1:], ys[:-1], ys[1:], gz[:-1], gz[1:])
                 if max(ya - ga, yb - gb) > 0.35]
        if skirt:
            parts.append(toward(concat(skirt), np.array([0.0, 0.0, f])))
    rot, shift = frame(c, axis, 0.0)
    return [placed(concat(parts), rot, shift)]


def bridges(scene):
    stream = sd_bytes(decode(scene.tex["locB"])[..., CHANNELS["locB"].index("stream")])
    return [p for c, axis, half, ends in crossings(scene) for p in bridge(scene, c, axis, half, ends, stream)]


# ---------------------------------------------------------------- family hooks (build_meshes.massing)
def build_footprint(scene, asset, c, polygons):
    """Hagrid's hut and its garden on the hut's footprint; the whole pitch on the first of its claimed components (the
    stands and the changing block), and nothing on the rest."""
    if asset == "hut":
        layout = hut_layout(polygons[0], scene.ground)
        return {"hut": hut(layout, scene.ground), "hut-garden": garden(scene.garden, layout, scene.ground)}
    if asset == "pitch":
        first = min(k["id"] for k in scene.assets["pitch"]["components"])
        return {"pitch": pitch(scene) if c["id"] == first else []}
    return None


def build_castle_pieces(scene, asset, keep):
    """The walls drawn on the castle grid: the vegetable-patch wall (5d) as the old curtain wall's low remnant, and the
    wall along the cliff's edge (2e) as a low safety wall; any other piece keeps the generic massing."""
    if asset != "grounds-walls":
        return None
    image = scene.castle_labelled[0]
    notes = scene.assets[asset]["castle_pieces"]
    out = []
    for piece in np.unique(image[keep]):
        mine = image == piece
        polygons = [p for k in range(len(scene.meta["castle"]["cls"]))
                    for p in field_polygons(scene.csd[..., k], scene.castle, 0.0, mine)]
        footprint = unary_union(polygons) if polygons else None
        if footprint is None or footprint.is_empty:
            continue
        c = np.asarray(footprint.centroid.coords[0])
        note = min(notes, key=lambda n: float(np.hypot(*(np.asarray(n["at"]) - c))))["note"]
        for part in pieces_of(footprint):
            if "(5d)" in note:
                out += ruin(scene, part)
            elif "(2e)" in note:
                out += safety_wall(scene, part)
        if "(5d)" not in note and "(2e)" not in note:
            out += scene.castle_classes(mine)
    return {asset: out}


def build_extras(scene):
    """The boundary walls under their coping and the field hedges with irregular crowns, as the raymarched ones stand
    (the hedge heuristic's line round the garden is the garden's fence, garden-fence, so it carries no hedge); the front
    gates; Dumbledore's tomb; the harbour mouth's shell; and the bridges where roads cross burns."""
    walls = sd_bytes(decode(scene.tex["locB"])[..., CHANNELS["locB"].index("wall")])
    hedge_field = sd_bytes(scene.locA[..., CHANNELS["locA"].index("hedge")])
    fence = LineString(scene.garden.exterior.coords).buffer(2.5)
    kept = [g for p in field_polygons(hedge_field, scene.local, HEDGE_LEVEL)
            for g in getattr(p.difference(fence), "geoms", [p.difference(fence)])
            if g.geom_type == "Polygon" and not g.is_empty]
    paths = road_ink_polygon(scene)
    runs = [g.simplify(0.1, preserve_topology=True) for g in joined(field_polygons(walls, scene.local, -WALL_IN), WALL_JOIN_M, paths)]
    return {"grounds-walls": coped(scene, runs, WALL_HEIGHT_M),
            "field-boundaries": hedges(scene, kept, paths),
            "gates": gates(scene),
            "tomb": tomb(scene),
            "harbour-mouth": harbour(scene),
            "bridges": bridges(scene)}
