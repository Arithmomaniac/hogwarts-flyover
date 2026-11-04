"""Hogsmeade's named buildings where they differ from their style families. mesh_village builds every building; these
hooks, given its unit or its _Building, return None (or False) where a building follows its family's rules.

The Hog's Head follows White Hound's plan of it: its door at the left of its front as you face it, a canted bay just
right of the front's middle and another on its left side wall near the front, its fireplaces' stack in its right
gable, its eaves along its street; its sign hangs from a rusty bracket straight over its door, the boar's severed head
painted on both faces (OotP 16).

The Shrieking Shack is two storeys of weatherboard under corrugated iron (White Hound), every window boarded but the
one that looks at the castle (PoA 14, DH 32), its garden ringed by a low fence (PoA 14).

Along a counter-clockwise plan outline, an edge's start is at the right of a viewer facing it from outside: plan
space is world (x east, z north) turned and mirrored (mesh_village_plan), so its counter-clockwise is clockwise seen
from above.
"""
import math
from dataclasses import replace

import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.polygon import orient

import mesh_village_plan as plan

HOGS_HEAD, SHACK, HONEYDUKES, BROOMSTICKS = "hogs-head", "shrieking-shack", "honeydukes", "three-broomsticks"
KIRK, SCHOOL, DISTILLERY, HALL, HANGAR = "kirk", "school", "distillery", "village-hall", "carriage-hangar"
ROLES = {SCHOOL: ("schoolroom", "schoolhouse"), HALL: ("hall", "annexe")}    # its largest unit's, and the rest's
PUBLIC = {   # the inferred designs' style records, by the family layout() gives each of their units
    "schoolroom": dict(storeys=1.0, eaves=4.2, wall="rubble", plinth="rubble"),
    "schoolhouse": dict(storeys=2.0, eaves=5.4, wall="harl-white", plinth="plinth-dark"),
    "still": dict(storeys=2.0, eaves=5.6, wall="harl-white", plinth="plinth-dark"),
    "kiln": dict(storeys=2.0, eaves=5.0, wall="harl-white", plinth="plinth-dark", pitch=55.0),
    "hall": dict(storeys=1.0, eaves=4.5, wall="harl-cream", plinth="plinth-dark"),
    "annexe": dict(storeys=1.0, eaves=3.0, wall="harl-cream", plinth="plinth-dark"),
    "hangar": dict(storeys=1.0, eaves=4.6, wall="weatherboard", plinth="rubble", pitch=30.0),
}
TALL = {"wide": 1.2, "sill": 1.0, "high": 2.4}
CARRIAGE = {"wide": 3.2, "high": 3.4, "every": 6.5}
VENT = {"wide": 1.3, "high": 0.9, "cap": 1.8, "cap_h": 0.7}
OLD_END = 0.4                   # the Three Broomsticks' very old end takes this share of its length
NAVE_SHARE = 0.5                # a slice across the kirk's footprint is nave where the footprint covers this much of it
PORCH_M = 3.6
LANCET = {"wide": 0.75, "sill": 1.1, "high": 2.9, "every": 4.8}
BUTTRESS = {"wide": 0.6, "out": 0.55, "high": 3.6, "top": 0.6, "into": 0.1}   # "top": how far its top slopes back
TURRET = {"base": (1.3, 1.1), "base_h": 1.0, "sunk": 0.6, "post": 0.18, "open": 1.3, "cap": (1.5, 1.3),
          "cap_h": 0.95, "bell": 0.4}
SKEW_M = 0.22
BAY = {"wide": 2.2, "face": 1.1, "deep": 0.6, "sill": 0.75, "head": 2.3, "top": 2.6, "pitch": 30.0,
       "into": 0.1}             # a bay runs this far into its wall, so the two make one solid
SIGN_OVER_M = 1.0               # the Hog's Head's bracket stands this far above its door's head
BOARDS_M = 0.08                 # planks nailed up this far behind a wall's face
FENCE = {"out": 4.0, "post": 3.5, "high": 1.1, "rail": 0.95, "post_w": 0.1, "rail_w": 0.07}
MULLION = {"wide": 0.09, "light": 0.8, "back": 0.02}     # no light between Honeydukes' mullions is wider than "light"
DERVISH, GLADRAGS = "dervish-and-banges", "gladrags"
POST_OFFICE, SCRIVENSHAFTS, PUDDIFOOTS = "post-office", "scrivenshafts", "puddifoots"
LOFT = {"long": 2.4, "wide": 1.5, "high": 1.3, "cap": 0.3, "cap_h": 0.8, "holes": 3, "hole": (0.24, 0.3),
        "hole_z": 0.45, "ledge": 0.12}  # holes and their ledge stand "hole_z" over the ridge
CROW = {"step": 0.5, "up": 0.25, "sunk": 0.35, "deep": 0.45, "proud": 0.07, "apex": (0.6, 0.3)}
BLIND = {"under": 2.7, "out": 1.4, "drop": 0.4, "valance": 0.2, "thick": 0.03}
CLOCK = {"at": 4.2, "out": 0.8, "r": 0.42, "face": 0.36, "half": 0.1, "sides": 12}  # its centre "at" over the floor


def restyle(named, style):
    """A named building's style record: the Hog's Head is a dark rubble inn under slate with no shopfront; the Shack
    two storeys of weatherboard on a rubble footing, under corrugated iron; the Three Broomsticks a two-storey inn,
    white-harled under slate, with its very old end (a cottage) low rubble under thatch. Scrivenshaft's is slated and
    gabled for its crow steps; Madam Puddifoot's is harled pink."""
    if named == HOGS_HEAD:
        return replace(style, wall="rubble", plinth="rubble", roof="slate", roof_material="village-slate", pitch=40.0,
                       hips=False, shopfront=None, margins=None, sign=True)
    if named == SHACK:
        return replace(style, family="shack", storeys=2.0, eaves=5.2, wall="weatherboard", plinth="rubble",
                       roof="iron", roof_material="corrugated-iron", pitch=35.0, hips=False, shopfront=None,
                       margins=None, door="door-black", sign=False)
    if named == BROOMSTICKS and style.family == "cottage":
        return replace(style, wall="rubble", plinth="rubble", roof="thatch", roof_material="village-thatch",
                       pitch=50.0, hips=False, shopfront=None, margins=None, sign=False)
    if named == BROOMSTICKS:
        return replace(style, wall="harl-white", plinth="plinth-dark", roof="slate", roof_material="village-slate",
                       pitch=40.0, hips=False, shopfront=None, sign=True)
    if named == KIRK:
        return replace(style, storeys=1.0, eaves=2.8 if style.family == "porch" else 5.2, wall="rubble",
                       plinth="rubble", roof="slate", roof_material="village-slate", pitch=50.0, hips=False,
                       shopfront=None, margins=None, door="door-black", sign=False)
    if named in (SCHOOL, DISTILLERY, HALL, HANGAR) and style.family in PUBLIC:
        return replace(style, **{"roof": "slate", "roof_material": "village-slate", "pitch": 45.0, "hips": False,
                                 "shopfront": None, "margins": None, "door": "door-green", "sign": False,
                                 **PUBLIC[style.family]})
    if named == SCRIVENSHAFTS:
        return replace(style, roof="slate", roof_material="village-slate", pitch=45.0, hips=False)
    if named == PUDDIFOOTS:
        return replace(style, wall="harl-pink", plinth="plinth-dark")
    return style


def layout(named, fp):
    """A named building's own plan of a footprint part, as (outline, family) units, else None. The school's largest
    unit is its schoolroom and the rest its schoolhouse; the village hall's largest is the hall and the rest its
    annexe; the distillery's most nearly square unit is its kiln and the rest its still house; the carriage hangar is
    its whole footprint, one shed. The kirk's nave
    ("kirk") is the footprint's body along its long axis, with the cross the map draws on it filled in: the slices
    across the long axis that the footprint covers for at least NAVE_SHARE of its length. A gabled porch ("porch")
    stands against the nave's side where the footprint steps out beyond it."""
    import shapely

    if named in ROLES:
        main, rest = ROLES[named]
        big = max(fp.units, key=lambda g: g.area)
        return [(g, main if g is big else rest) for g in fp.units]
    if named == DISTILLERY:
        def squareness(g):
            x0, y0, x1, y1 = g.bounds
            return min(x1 - x0, y1 - y0) / max(x1 - x0, y1 - y0)
        kiln = max(fp.units, key=squareness)
        return [(g, "kiln" if g is kiln else "still") for g in fp.units]
    if named == HANGAR:
        return [(fp.outline, "hangar")]
    if named != KIRK:
        return None
    o = fp.outline
    x0, y0, x1, y1 = o.bounds
    swap = (x1 - x0) > (y1 - y0)                       # work with the long axis along y
    flip = (lambda g: shapely.transform(g, lambda p: p[:, ::-1])) if swap else (lambda g: g)
    o = flip(o)
    x0, y0, x1, y1 = o.bounds
    xs = sorted({round(float(x), 3) for x, _ in o.exterior.coords})
    body = [(a, b) for a, b in zip(xs, xs[1:])
            if o.intersection(box(a, y0, b, y1)).area / (b - a) >= NAVE_SHARE * (y1 - y0)]
    lo, hi = min(a for a, _ in body), max(b for _, b in body)
    nave = box(lo, y0, hi, y1)
    out = [(nave, "kirk")]
    rest = o.difference(nave)
    rest = [g for g in getattr(rest, "geoms", [rest]) if g.area > 1.0]
    if rest:
        step = max(rest, key=lambda g: g.area).centroid
        x = hi if step.x > hi else lo - PORCH_M
        out.append((box(x, step.y - PORCH_M / 2, x + PORCH_M, step.y + PORCH_M / 2), "porch"))
    return [(orient(shapely.set_precision(flip(g), plan.SNAP_M), 1.0), family) for g, family in out]


def chimneys(b):
    """Whether a named building has chimney stacks: the kirk, the distillery's kiln and the carriage hangar have none."""
    return b.u.asset != KIRK and b.st.family not in ("kiln", "hangar")


def split(named, outline):
    """A named building's own division of its unit, as (outline, family) pieces, else None: the Three Broomsticks'
    later inn and, across its length at one end, its very old part, a cottage (White Hound's Cramond Inn)."""
    if named != BROOMSTICKS:
        return None
    x0, y0, x1, y1 = outline.bounds
    if abs(outline.area - (x1 - x0) * (y1 - y0)) > 1e-6:
        return None
    if x1 - x0 >= y1 - y0:
        cut = round(x0 + OLD_END * (x1 - x0), 2)
        old, new = box(x0, y0, cut, y1), box(cut, y0, x1, y1)
    else:
        cut = round(y0 + OLD_END * (y1 - y0), 2)
        old, new = box(x0, y0, x1, cut), box(x0, cut, x1, y1)
    return [(orient(new, 1.0), "shop"), (orient(old, 1.0), "cottage")]


def roof_angles(u):
    """Each edge's pitch where a named building's roof runs its own way, else None: the Hog's Head's eaves run along
    its front and back, its sides are gables; the kirk's porch is gabled over its door and against the nave, its
    square plan leaving the generic rule nothing to choose by."""
    rect = len(u.outline.exterior.coords) == 5 and not u.outline.interiors
    if u.asset == HOGS_HEAD and rect:
        return [u.style.pitch if (k - u.front) % 2 == 0 else 90.0 for k in range(4)]
    if u.asset == KIRK and u.style.family == "porch" and rect:
        return [90.0 if (k - u.front) % 2 == 0 else u.style.pitch for k in range(4)]
    if u.asset == DISTILLERY and u.style.family == "kiln":
        return [u.style.pitch] * len(plan.edges(u.outline))           # a pyramid, hipped against the still house too
    return None


def stack_gables(b):
    """The gables a named building's stacks may stand on, else None (any): the Hog's Head's right one as you face its
    front, where White Hound's plan puts every fireplace; the Shack's left one; any of the schoolroom's but the one its
    bell turret stands on."""
    if b.u.asset == HOGS_HEAD:
        return {(b.u.front - 1) % len(b.edges)}
    if b.u.asset == SHACK:
        return {(b.u.front + 1) % len(b.edges)}
    if b.u.asset == SCHOOL and b.st.family == "schoolroom":
        return set(range(len(b.edges))) - {_bell_gable(b)}
    return None


def stack_wall(b):
    """What a named building's stacks are built of, if not its walls: the Shack's of rubble, like its footing."""
    return "rubble" if b.u.asset == SHACK else None


def openings(b):
    """Cut a named building's openings and hang its fittings, if it has its own; whether it did."""
    if b.u.asset == HOGS_HEAD:
        _hogs_head(b)
        return True
    if b.u.asset == SHACK:
        _shack(b)
        return True
    if b.u.asset == KIRK:
        _kirk(b) if b.st.family == "kirk" else _porch(b)
        return True
    if b.st.family in ("schoolroom", "hall") and b.u.asset in (SCHOOL, HALL):
        _tall_room(b, bell=b.st.family == "schoolroom")
        return True
    if b.st.family == "kiln" and b.u.asset == DISTILLERY:
        _kiln(b)
        return True
    if b.st.family == "hangar" and b.u.asset == HANGAR:
        _hangar(b)
        return True
    return False


def finish(b):
    """Hang a named shop's own fittings on it once its openings are cut: the post office's owl loft, Scrivenshaft's
    crow steps, Gladrags' shop blind and Dervish & Banges' bracket clock."""
    if b.u.asset == POST_OFFICE:
        _owl_loft(b)
    elif b.u.asset == SCRIVENSHAFTS:
        _crow_steps(b)
    elif b.u.asset == GLADRAGS:
        _blind(b)
    elif b.u.asset == DERVISH:
        _clock(b)


def dress(b, k, s, z0, w, h, depth, mat):
    """Dress a front window just cut in facade k (centred s along it, from z0, w by h, its glass `depth` behind the
    wall's face, its surround `mat`). Honeydukes' are mullioned (PoA 10), divided into lights at most MULLION['light']
    wide by upright bars of `mat` standing in the opening from sill to head. Madam Puddifoot's shop windows (those in
    its shopfront's paint) are steamed up (OotP 25): a misted pane stands just in front of the glass."""
    if b.u.asset == PUDDIFOOTS and mat == b.st.shopfront:
        b.fittings.append(b._local_box(k, s, -depth + 0.01, -depth + 0.02, z0, z0 + h, w, "glass-steamed"))
        return
    if b.u.asset != HONEYDUKES:
        return
    n = math.ceil(w / MULLION["light"]) - 1
    for i in range(1, n + 1):
        b.fittings.append(b._local_box(k, s - w / 2 + w * i / (n + 1), -depth, -MULLION["back"], z0, z0 + h,
                                       MULLION["wide"], mat, back=True))


def _hogs_head(b):
    import mesh_village as mv

    u, st = b.u, b.st
    m = len(b.edges)
    front, left, back = u.front, (u.front + 1) % m, (u.front + 2) % m
    _, _, _, length = b._facade(front)
    zf, upper = b.z_floor, b.z_floor + mv.STOREY_M + mv.SILL_M - 0.1
    door_w, door_h = 1.0, min(2.05, b._head(front) - zf)
    door = length - 1.5                                 # the bar's door at the front's left end
    b._hole(front, door, zf, door_w, door_h, st.door, st.wall, 0.12)
    b._steps(front, door, door_w)
    _sign(b, front, door, zf + door_h + SIGN_OVER_M)
    bay = length / 2 - 0.2
    _bay(b, front, bay)
    b._window(front, bay + 0.4, upper, 0.9, 1.3)       # the sitting-room window looking down on the street
    _bay(b, left, 2.5)
    b._window(left, 2.0, upper, 0.8, 1.2)              # the guest bedroom
    _, _, _, depth = b._facade(back)
    b._hole(back, 1.2, zf, 0.9, door_h, st.door, st.wall, 0.12)     # the back room's door to the paddock
    b._steps(back, 1.2, 0.9)
    b._window(back, depth / 2 + 0.6, zf + mv.SILL_M, 0.8, 1.1)
    b._window(back, depth / 2, upper, 0.7, 1.0)


def _shack(b):
    """The Shrieking Shack's openings: a door in the middle of its front, sealed like every other way in (PoA 14), and
    windows on both storeys of every free wall, each boarded over but the ground-floor one in the middle of the wall
    that faces the castle at the world's origin (DH 32); then its garden fence."""
    import mesh_village as mv

    u, st, f = b.u, b.st, b.fp.frame
    free = [k for k in range(len(b.edges)) if k not in u.party]
    centre = f.world(np.asarray(u.outline.centroid.coords[0]))
    castle = max(free, key=lambda k: float(_facing(b, k) @ (-centre / np.linalg.norm(centre))))
    zf, upper = b.z_floor, b.z_floor + mv.STOREY_M + mv.SILL_M - 0.1
    for k in free:
        _, _, _, length = b._facade(k)
        centres, _ = mv.bays(length)
        door = len(centres) // 2 if k == u.front else None
        clear = (len(centres) // 2 + (1 if k == u.front else 0)) % len(centres) if k == castle else None
        for i, s in enumerate(centres):
            if i == door:
                b._hole(k, s, zf, 1.0, 2.0, st.door, st.wall, 0.12)
                b._steps(k, s, 1.0)
            else:
                _window(b, k, s, zf + mv.SILL_M, 0.8, 1.2, "glass-dim" if i == clear else None)
            _window(b, k, s, upper, 0.8, 1.1)
    _fence(b)


def _window(b, k, s, z0, w, h, glass=None):
    """A window of facade k at s, from z0 up: glazed in `glass`, or else boarded over with rough planks."""
    import mesh_village as mv

    h = min(h, b._head(k) - z0)
    if h < mv.MIN_OPENING_M or not b._open(k, s, w):
        return
    b._hole(k, s, z0, w, h, glass or "boarding", b.st.wall, mv.REVEAL_M if glass else BOARDS_M)


def _fence(b):
    """The Shack's garden fence (PoA 14: they "leaned on the fence, looking up at it"): posts at most FENCE['post']
    apart on a ring FENCE['out'] beyond its walls, standing in the ground and joined by a rail; none within reach of a
    road, so the path runs through it. Standing out beyond the eaves, it casts."""
    from materials import code
    from mesh_geometry import bar, concat

    ring = np.asarray(orient(b.u.outline.buffer(FENCE["out"], join_style=2), 1.0).exterior.coords)
    pts = []
    for p, q in zip(ring[:-1], ring[1:]):
        n = max(1, math.ceil(float(np.linalg.norm(q - p)) / FENCE["post"]))
        pts += [p + (q - p) * i / n for i in range(n)]
    pts = np.array(pts)
    world = b.fp.frame.world(pts)
    g = np.asarray(b.v.scene.ground(world[:, 0], world[:, 1]), np.float64) - b.y_ref
    standing = [b.v.streets.road_share([w]) == 0.0 for w in world]
    attr = code("boarding")
    pieces = []
    for i in range(len(pts)):
        if not standing[i]:
            continue
        pieces.append(bar((*pts[i], g[i] - 0.2), (*pts[i], g[i] + FENCE["high"]), FENCE["post_w"], attr))
        j = (i + 1) % len(pts)
        if standing[j]:
            pieces.append(bar((*pts[i], g[i] + FENCE["rail"]), (*pts[j], g[j] + FENCE["rail"]), FENCE["rail_w"], attr))
    if pieces:
        b.cast(concat(pieces))


def _facing(b, k):
    """The world (x, z) outward normal of facade k."""
    a, c = b.edges[k]
    mid = (a + c) / 2
    n = b.fp.frame.world(mid + plan.outward(a, c)) - b.fp.frame.world(mid)
    return n / np.linalg.norm(n)


def _kirk(b):
    """The kirk's nave, after Balquhidder (White Hound): pointed lancets along its long walls, each between
    buttresses (none where its porch stands), one lancet in its westward gable under the lantern turret on its apex,
    three in the other gable, the middle one tallest, and stone skews up its gables."""
    u, zf = b.u, b.z_floor
    lengths = [b._facade(k)[3] for k in range(len(b.edges))]
    sides = [k for k, length in enumerate(lengths) if length > max(lengths) - 0.01]
    west, east = sorted((k for k in range(len(lengths)) if k not in sides),
                        key=lambda k: -float(_facing(b, k) @ np.array([-1.0, 0.0])))
    porches = [w.outline for w in b.v.units[u.cid] if w.part == u.part and w is not u]
    for k in sides:
        a, t, n, length = b._facade(k)
        taken = [(float(min(p)), float(max(p))) for p in
                 (((np.asarray(g.exterior.coords) - a) @ t) for g in porches
                  if g.distance(LineString([a, a + t * length])) < 0.05)]
        count = max(2, int((length - 2.0) // LANCET["every"]))
        bay = (length - 2.0) / count
        for i in range(count):
            s = 1.0 + bay * (i + 0.5)
            if not any(lo - 0.5 < s + LANCET["wide"] / 2 and s - LANCET["wide"] / 2 < hi + 0.5 for lo, hi in taken):
                _lancet(b, k, s, zf + LANCET["sill"], LANCET["wide"], LANCET["high"])
        for s in [0.5] + [1.0 + bay * i for i in range(1, count)] + [length - 0.5]:
            if not any(lo - 0.35 < s + BUTTRESS["wide"] / 2 and s - BUTTRESS["wide"] / 2 < hi + 0.35 for lo, hi in taken):
                _buttress(b, k, s)
    _lancet(b, west, lengths[west] / 2, zf + 1.6, 0.7, 3.0)
    for ds, w, h in ((-1.3, 0.62, 3.0), (0.0, 0.8, 3.8), (1.3, 0.62, 3.0)):
        _lancet(b, east, lengths[east] / 2 + ds, zf + LANCET["sill"] + 0.3, w, h)
    _turret(b, west)
    _skews(b)


def _porch(b):
    """The kirk's porch: a pointed door in its front gable, up its steps, and skews on that gable."""
    k = b.u.front
    _, _, _, length = b._facade(k)
    _lancet(b, k, length / 2, b.z_floor, 1.1, 2.5, b.st.door, 0.12)
    b._steps(k, length / 2, 1.1)
    _skews(b)


def _lancet(b, k, s, z0, w, h, back="glass-dark", depth=None):
    """A pointed lancet in facade k centred s along it, from z0: w wide and h high to the point of its equilateral
    arch, its back face `back` (glass, or a door) `depth` behind the wall's face."""
    import mesh_village as mv

    depth = mv.REVEAL_M if depth is None else depth
    a, t, n, _ = b._facade(k)
    c = a + t * s
    rise = w * math.sqrt(3.0) / 2
    profile = np.array([(-w / 2, 0.0), (w / 2, 0.0), (w / 2, h - rise), (0.0, h), (-w / 2, h - rise)])
    cutter = (b.m3.Manifold.extrude(b.m3.CrossSection([profile], b.m3.FillRule.EvenOdd), depth + 0.3)
              .rotate((90.0, 0.0, 0.0))            # the profile upright; its depth now runs along -y
              .translate((0.0, depth, 0.0))        # 0.3 m outside the face to `depth` inside it, once turned along t
              .rotate((0.0, 0.0, math.degrees(math.atan2(t[1], t[0]))))
              .translate((c[0], c[1], z0)))
    b.cutters.append((cutter, mv._cut(back, b.st.wall, np.array([n[0], n[1], 0.0]))))
    b.panels.append((c, t, n, z0, [tuple(p) for p in profile], back))


def _buttress(b, k, s):
    """A buttress against facade k at s: BUTTRESS['wide'] along the wall and standing BUTTRESS['out'] from it, from below
    the ground to BUTTRESS['high'] over the floor, its top weathered back to the wall."""
    import mesh_village as mv

    a, t, n, _ = b._facade(k)
    c = a + t * s
    d, h, top, into = BUTTRESS["out"], BUTTRESS["high"], BUTTRESS["top"], BUTTRESS["into"]
    low = b.z_bottom - b.z_floor
    profile = np.array([(-into, low), (d, low), (d, h - top), (0.0, h), (-into, h)])     # (out of the wall, up)
    solid = b.m3.Manifold.extrude(b.m3.CrossSection([profile], b.m3.FillRule.EvenOdd), BUTTRESS["wide"])
    # local (out, up, along -t) to plan: a proper rotation, then the buttress's middle onto c at the floor
    m = np.array([[n[0], 0.0, -t[0], c[0] + t[0] * BUTTRESS["wide"] / 2],
                  [n[1], 0.0, -t[1], c[1] + t[1] * BUTTRESS["wide"] / 2],
                  [0.0, 1.0, 0.0, b.z_floor]])
    b.solids.append((solid.transform(m), mv._material(b.st.wall), "feature"))


def _turret(b, k):
    """The kirk's lantern bell turret on the apex of gable k (White Hound: "this traditional, lantern-like
    decoration"): a stone base astride the apex, set DEPTH_GAP_M back from the gable's face, four stone posts round an
    open stage with its bell, a pyramid cap."""
    import mesh_village as mv

    _, _, _, length = b._facade(k)
    s, z = length / 2, b.z_ridge
    bw, bd = TURRET["base"]
    z1 = z - TURRET["sunk"] + TURRET["base_h"]
    zt = z1 + TURRET["open"]
    p = TURRET["post"]
    b.feature(b._local_box(k, s, -bd, -mv.DEPTH_GAP_M, z - TURRET["sunk"], z1, bw, "pale-stone", back=True))
    for ds in (-(bw - p) / 2 + 0.05, (bw - p) / 2 - 0.05):
        for o0 in (-0.05 - p, -bd + 0.05):
            b.feature(b._local_box(k, s + ds, o0, o0 + p, z1, zt, p, "pale-stone", back=True))
    bell = TURRET["bell"]
    b.fittings.append(b._local_box(k, s, -(bd + bell) / 2, -(bd - bell) / 2, zt - 0.55, zt - 0.2, bell, "black-iron",
                                   back=True))
    b.feature(_pyramid(b, k, s, -bd / 2, *TURRET["cap"], zt, zt + TURRET["cap_h"], "pale-stone"))


def _pyramid(b, k, s, o, w, d, z0, z1, mat):
    """A pyramid on facade k's frame: its base w along the facade and d across it, centred s along and o out of it, at
    z0, its apex at z1."""
    import mesh_village as mv
    from materials import code
    from mesh_geometry import concat, panel

    a, t, n, _ = b._facade(k)
    c = a + t * s + n * o
    base = [np.array([*(c + t * x * w / 2 + n * y * d / 2), z0]) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    apex = np.array([c[0], c[1], z1])
    faces = [base] + [[base[i], base[(i + 1) % 4], apex] for i in range(4)]
    return mv._outward(concat([panel(f, code(mat)) for f in faces]), np.array([c[0], c[1], (z0 + z1) / 2]))


def _skews(b):
    """Stone skews up the rakes of a building's free gables."""
    from materials import code
    from mesh_geometry import bar

    for k, (a, c) in enumerate(b.edges):
        if b.angles[k] < 89.999 or k in b.u.party:
            continue
        n = plan.outward(a, c)
        mid = (a + c) / 2 - n * SKEW_M / 2
        for end in (a - n * SKEW_M / 2, c - n * SKEW_M / 2):
            b.fittings.append(bar((*end, b.z_eave + 0.12), (*mid, b.z_ridge + 0.12), SKEW_M, code("pale-stone")))


def _tall_room(b, bell):
    """A tall one-storey room (the school's schoolroom, the village hall): a door in the middle of its front, tall
    windows in its other bays and along its back, and on the schoolroom a bell turret on its longest free gable."""
    import mesh_village as mv

    u, zf = b.u, b.z_floor
    k = u.front
    _, _, _, length = b._facade(k)
    centres, pitch = mv.bays(length)
    for i, s in enumerate(centres):
        if i == len(centres) // 2:
            b._hole(k, s, zf, 1.2, 2.3, b.st.door, b.st.wall, 0.12)
            b._steps(k, s, 1.2)
        else:
            b._window(k, s, zf + TALL["sill"], min(TALL["wide"], pitch - 0.6), TALL["high"])
    back = b._back()
    if back is not None:
        _, _, _, depth = b._facade(back)
        centres, pitch = mv.bays(depth)
        for s in centres:
            b._window(back, s, zf + TALL["sill"], min(TALL["wide"], pitch - 0.6), TALL["high"])
    if bell and _bell_gable(b) is not None:
        _turret(b, _bell_gable(b))


def _bell_gable(b):
    """The gable a schoolroom's bell turret stands on: its longest free gable, or None."""
    gables = [k for k in range(len(b.edges)) if b.angles[k] >= 89.999 and k not in b.u.party]
    return max(gables, key=lambda k: b._facade(k)[3]) if gables else None


def _kiln(b):
    """The distillery's kiln: a door in the middle of its front with a small window beside it, and on its pyramid
    roof's peak a louvred ventilator under a slate cap, the pagoda of a Highland distillery."""
    import mesh_village as mv
    import mesh_village_roofs as roofs

    k, zf = b.u.front, b.z_floor
    a, t, n, length = b._facade(k)
    b._hole(k, length / 2, zf, 1.1, 2.1, b.st.door, b.st.wall, 0.12)
    b._steps(k, length / 2, 1.1)
    b._window(k, length / 2 + 1.6, zf + mv.SILL_M, 0.7, 1.0)
    v, _ = roofs.roof(b.u.outline, b.angles)
    peak = v[v[:, 2] >= v[:, 2].max() - 1e-6][:, :2].mean(axis=0)
    s, o = float((peak - a) @ t), float((peak - a) @ n)
    w = VENT["wide"]
    z0 = b.z_ridge - w / 2 * math.tan(math.radians(b.st.pitch)) - 0.1       # down to meet the slopes at its sides
    z1 = b.z_ridge + VENT["high"]
    b.feature(b._local_box(k, s, o - w / 2, o + w / 2, z0, z1, w, "boarding", back=True))
    b.feature(_pyramid(b, k, s, o, VENT["cap"], VENT["cap"], z1, z1 + VENT["cap_h"], "village-slate"))


def _hangar(b):
    """The carriage hangar: a row of carriage doors along its front, CARRIAGE['wide'] by CARRIAGE['high'], each
    opening FLOOR_OVER_M over the ground before it, however the ground falls along the shed, so a carriage can roll
    in."""
    import mesh_village as mv

    k, zf = b.u.front, b.z_floor
    a, t, n, length = b._facade(k)
    count = max(1, int((length - 2.0) // CARRIAGE["every"]))
    for i in range(count):
        s = 1.0 + (length - 2.0) * (i + 0.5) / count
        g = float(b.v.scene.ground(*b.fp.frame.world(a + t * s + n * 0.6))) - b.y_ref
        z0 = min(zf, g + mv.FLOOR_OVER_M)
        b._hole(k, s, z0, CARRIAGE["wide"], min(CARRIAGE["high"], b._head(k) - z0), b.st.door, b.st.wall, 0.12)


def _owl_loft(b):
    """The post office's owl loft (OotP 16: "the post office, from which owls issued at regular intervals"): a
    weatherboarded box along the middle of its longest ridge under a hipped slate cap, reaching down to meet the roof's
    slopes, with a row of flight holes over a perching ledge along each long side."""
    import mesh_village_roofs as roofs

    v, f = roofs.roof(b.u.outline, b.angles)
    top = v[:, 2] >= v[:, 2].max() - 1e-6
    p, q = max(((v[i, :2], v[j, :2]) for tri in f for i, j in zip(tri, np.roll(tri, -1)) if top[i] and top[j]),
               key=lambda e: float(np.linalg.norm(e[1] - e[0])))
    d = (q - p) / np.linalg.norm(q - p)
    k = max(range(len(b.edges)), key=lambda k: abs(float(b._facade(k)[1] @ d)))     # a facade along the ridge
    a, t, n, _ = b._facade(k)
    s, o = float(((p + q) / 2 - a) @ t), float(((p + q) / 2 - a) @ n)
    long, wide = min(LOFT["long"], float(np.linalg.norm(q - p))), LOFT["wide"]
    z0 = b.z_ridge - wide / 2 * math.tan(math.radians(b.st.pitch)) - 0.1
    z1 = b.z_ridge + LOFT["high"]
    b.feature(b._local_box(k, s, o - wide / 2, o + wide / 2, z0, z1, long, "weatherboard", back=True))
    b.feature(_pyramid(b, k, s, o, long + 2 * LOFT["cap"], wide + 2 * LOFT["cap"], z1, z1 + LOFT["cap_h"],
                               "village-slate"))
    (hole_w, hole_h), zh = LOFT["hole"], b.z_ridge + LOFT["hole_z"]
    for side in (-1.0, 1.0):
        face = o + side * wide / 2                     # a box's first `out` is the face it leaves open, against the loft
        for i in range(LOFT["holes"]):
            x = s + (i - (LOFT["holes"] - 1) / 2) * long / LOFT["holes"]
            b.fittings.append(b._local_box(k, x, face, face + side * 0.005, zh, zh + hole_h, hole_w, "margin-black"))
        b.feature(b._local_box(k, s, face, face + side * LOFT["ledge"], zh - 0.05, zh, long - 0.2, "weatherboard"))


def _crow_steps(b):
    """Scrivenshaft's crow-stepped gables (White Hound: "A sort of Mediaeval-style stationers"): up each rake of its
    free gables, stone steps about CROW['step'] wide stand on the gable wall, each from under the roof to CROW['up']
    over the roof at its upper end, and a taller stone caps the apex."""
    import mesh_village as mv

    top = mv.attic_top(b.u.outline, b.angles)
    for k in range(len(b.edges)):
        if b.angles[k] < 89.999 or k in b.u.party:
            continue
        a, t, n, length = b._facade(k)

        def rake(x):
            x = np.clip(np.atleast_1d(x), 0.05, length - 0.05)            # just inside the gable, off its corners
            return b.z_eave + top(a + np.outer(x, t) - n * 0.05)

        xs = np.linspace(0.0, length, 101)
        apex = float(xs[np.argmax(rake(xs))])
        for x0 in (mv.DEPTH_GAP_M, length - mv.DEPTH_GAP_M):        # off the corners, out of the side walls' planes
            count = max(2, round(abs(apex - x0) / CROW["step"]))
            w = (apex - x0) / count
            for i in range(count):
                lo, hi = x0 + w * i, x0 + w * (i + 1)
                b.feature(b._local_box(k, (lo + hi) / 2, -CROW["deep"], CROW["proud"],
                                               float(rake(lo)[0]) - CROW["sunk"], float(rake(hi)[0]) + CROW["up"],
                                               abs(w), "pale-stone", back=True))
        wide, over = CROW["apex"]
        z = float(rake(apex)[0])
        b.feature(b._local_box(k, apex, -CROW["deep"], CROW["proud"], z - CROW["sunk"], z + CROW["up"] + over,
                                       wide, "pale-stone", back=True))


def _blind(b):
    """Gladrags' canvas shop blind, an outfitter's: as wide as its fascia, pulled out from under it and sloping down
    BLIND['drop'] to a valance BLIND['out'] out over the pavement. Standing out beyond the eaves, it casts."""
    import mesh_village as mv

    k, zf = b.u.front, b.z_floor
    _, _, _, length = b._facade(k)
    centres, pitch = mv.bays(length)
    s, half = (centres[0] + centres[-1]) / 2, (pitch * len(centres) - 0.2) / 2
    z_in = zf + BLIND["under"]
    z_out, out, thick = z_in - BLIND["drop"], BLIND["out"], BLIND["thick"]
    b.cast(_plate(b, k, s, [(0.12, z_in), (out, z_out), (out, z_out - thick), (0.12, z_in - thick)], "shop-blind",
                  half))
    b.cast(_plate(b, k, s, [(out - thick, z_out), (out, z_out), (out, z_out - BLIND["valance"]),
                            (out - thick, z_out - BLIND["valance"])], "shop-blind", half))


def _clock(b):
    """Dervish & Banges' bracket clock, an instrument maker's sign (PoA 5: "magical instruments and stuff"): a drum
    with a white face on either side, looking up and down the street, held out over the pavement on an iron bar from
    the pier nearest the middle of its front, between its upper windows."""
    import mesh_village as mv

    k, zf = b.u.front, b.z_floor
    _, _, _, length = b._facade(k)
    centres, _ = mv.bays(length)
    m = len(centres) // 2
    s = (centres[m - 1] + centres[m]) / 2 if m else centres[0]
    o, z, r = CLOCK["out"], zf + CLOCK["at"], CLOCK["r"]
    ring = lambda rad: [(o + rad * math.cos(2 * math.pi * i / CLOCK["sides"]),
                         z + rad * math.sin(2 * math.pi * i / CLOCK["sides"])) for i in range(CLOCK["sides"])]
    b.fittings.append(b._local_box(k, s, 0.0, o - r + 0.02, z - 0.03, z + 0.03, 0.06, "black-iron"))
    b.fittings.append(_plate(b, k, s, ring(r), "black-iron", CLOCK["half"]))
    for side in (-1.0, 1.0):
        b.fittings.append(_plate(b, k, s + side * (CLOCK["half"] + 0.0025), ring(CLOCK["face"]), "clock-face"))


def _bay(b, k, s):
    """A canted bay window on facade k, centred s along it: a trapezoid of the wall's stone from below the ground to over
    its windows, standing BAY['deep'] out, under a slate roof sloping back to the wall, its face and both canted sides
    glazed with grimy glass."""
    import mesh_village as mv

    a, t, n, _ = b._facade(k)
    c = a + t * s
    wb, wf, d = BAY["wide"] / 2, BAY["face"] / 2, BAY["deep"]
    inner = c - n * BAY["into"]
    poly = orient(Polygon([inner - t * wb, inner + t * wb, c + t * wf + n * d, c - t * wf + n * d]), 1.0)
    top = b.z_floor + BAY["top"]
    b.solids.append((b._prism(poly, b.z_bottom, top), mv._material(b.st.wall), "feature"))
    sides = plan.edges(poly)
    against = [abs(float(((p + q) / 2 - c) @ n) + BAY["into"]) < 1e-6 for p, q in sides]
    angles = [90.0 if w else BAY["pitch"] for w in against]
    rise = (d + BAY["into"]) * math.tan(math.radians(BAY["pitch"]))
    b.solids.append((b._roof_mesh(poly, angles, top, cap=rise), mv._material(b.st.roof_material), "feature"))
    for (p, q), w in zip(sides, against):
        if w:
            continue
        span = float(np.linalg.norm(q - p))
        tf = (q - p) / span
        b._opening((p + q) / 2, tf, plan.outward(p, q), b.z_floor + BAY["sill"], min(0.8, span - 0.36),
                   BAY["head"] - BAY["sill"], "glass-dark", b.st.wall)


def _sign(b, k, s, z):
    """The Hog's Head's sign (OotP 16): a battered board on a rusty bracket at height z straight over the door at s
    along facade k, hanging out across the wall's face; on each face of the board a white cloth carries the boar's dark
    severed head, and its blood below it."""
    b.fittings.append(b._local_box(k, s, 0.0, 0.95, z, z + 0.05, 0.05, "rust-iron"))
    b.fittings.append(b._local_box(k, s, 0.3, 0.95, z - 0.62, z - 0.02, 0.05, "sign-board", back=True))
    for side in (-1.0, 1.0):
        b.fittings.append(b._local_box(k, s + side * 0.03, 0.34, 0.91, z - 0.58, z - 0.06, 0.01, "sign-cloth",
                                       back=True))
        at = s + side * 0.0375
        b.fittings.append(_plate(b, k, at, [(0.47, z - 0.2), (0.52, z - 0.13), (0.66, z - 0.12), (0.74, z - 0.2),
                                            (0.71, z - 0.32), (0.53, z - 0.34)], "margin-black"))     # the head
        b.fittings.append(_plate(b, k, at, [(0.72, z - 0.25), (0.86, z - 0.29), (0.86, z - 0.35), (0.7, z - 0.33)],
                                 "margin-black"))                                                        # the snout
        b.fittings.append(_plate(b, k, at, [(0.55, z - 0.36), (0.65, z - 0.36), (0.63, z - 0.5), (0.57, z - 0.5)],
                                 "margin-oxblood"))                                                      # the blood


def _plate(b, k, s, outline, mat, half=0.0025):
    """A plate 2 * half thick centred s along facade k: a convex polygon of (metres out of the wall, height) points."""
    import mesh_village as mv
    from materials import code
    from mesh_geometry import concat, panel

    a, t, n, _ = b._facade(k)
    c = a + t * s
    p = lambda x, o, z: np.array([c[0] + t[0] * x + n[0] * o, c[1] + t[1] * x + n[1] * o, z])
    ring = list(outline)
    faces = [[p(-half, o, z) for o, z in ring], [p(half, o, z) for o, z in ring[::-1]]]
    for (o0, z0), (o1, z1) in zip(ring, ring[1:] + ring[:1]):
        faces.append([p(-half, o0, z0), p(half, o0, z0), p(half, o1, z1), p(-half, o1, z1)])
    centre = p(0.0, *np.mean(np.asarray(ring), axis=0))
    return mv._outward(concat([panel(f, code(mat)) for f in faces]), centre)
