"""The village family (Hogsmeade): each mapped footprint regularized and split into buildings (mesh_village_plan), and
each building built from its style record as one manifold3d solid: walls and a plinth down to below the ground, an
attic whose vertical faces are the gables, a roof skin from the straight-skeleton roofer (mesh_village_roofs) whose
eaves and verges overhang, chimney stacks capped in pale stone, and windows, doors and shop windows cut from the solid.
Every face takes its material from the solid it came from (manifold3d keeps each face's original solid): a cut's
reveal takes its own material and its back face becomes the glass or the door. Fine fittings (chimney pots, sills,
steps, hanging signs) are separate pieces, so integration can drop them with distance.

Plan space is each footprint part's frame (mesh_village_plan.Frame): plan X, Y with height Z above the part's
reference ground, mapped to world by a proper rotation. Heights come from the sampled ground, never from fixed levels.

Every footprint is built but one whose plan is only specks (a sliver of a wing), which keeps the generic massing. So
that the Shrieking Shack stands a little way above the rest of the village (PoA 14), the manifest leaves the hamlet on
the hill to its north-west unbuilt, and two cottages are built across their lane from where the map draws them (MOVED).
The village keeps to its share of the payload by spending fine detail where the flyover, passing 50 m and more above it,
can see it: stacks are capped flush, a raised door's steps are one block, and off the High Street a stack carries one
pot and a back wall one window.

The same building records give the coarser levels of detail (LEVELS, levels()), so whatever places a building places it
at every level: lod1 lays its windows and doors flat on its walls and stands its stacks in its roofs as boxes, keeping
its features but none of its fittings; lod2 is its walls and roofs; its shadow caster its walls and attic, with what
stands out beyond its eaves and casts a shadow of its own (Gladrags' blind, the Shack's fence, the kirkyard's and the
gardens' walls). The payload carries each level that passes its guards and saves enough triangles,
and the page draws one per view.
"""
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient

import mesh_village_named as bespoke
import mesh_village_plan as plan
import mesh_village_roofs as roofs
from materials import code
from mesh_geometry import Mesh, concat, indexed, oriented
from mesh_source import CHANNELS

FAMILY = "village"
SHOPS = ("dervish-and-banges", "gladrags", "three-broomsticks", "zonkos", "post-office", "hogs-head", "scrivenshafts",
         "honeydukes", "puddifoots")
INNS = ("three-broomsticks", "hogs-head")
BUILDINGS = ("kirk", "school", "distillery", "village-hall", "carriage-hangar", "shrieking-shack")
NAMED = SHOPS + BUILDINGS
HOUSE, WING = "village-house", "village-wing"
HIGH_STREET_SPAN_M = 16.0       # road texels this far either side of the traced line find the High Street's centre
# Cottages built across their lane so that the Shrieking Shack stands above the rest of the village (the manifest
# decision "shack-above-the-village"): component -> (its decoded centre, world x and z; where that point goes; the turn
# in degrees about it). Each turns about the lane onto its far side, its front still looking across at the lane.
MOVED = {23: ((-1242.94, -643.56), (-1268.42, -661.65), 180.0),
         28: ((-1220.83, -685.26), (-1243.84, -730.85), 155.64)}

SINK_M = 1.5                    # walls reach this far below the lowest ground under them
FLOOR_OVER_M = 0.15             # the floor stands this far above the ground at the front
PLINTH_M = 0.5                  # a painted base course rises at most this far above the lowest ground under its walls
STOREY_M = 2.8
SILL_M = 0.85
EPS_M = 0.02                    # the roof skin rides this far above the attic, so their faces never coincide
THATCH = {"eave": 0.45, "verge": 0.3, "thick": 0.38}
SLATE = {"eave": 0.14, "verge": 0.06, "thick": 0.09}
REVEAL_M = 0.16                 # windows and doors stand this far behind the wall face
MARGIN_M, MARGIN_DEPTH_M = 0.12, 0.025
BAY_M = 2.7
STACK = {"across": 0.62, "along": 1.15, "spare": 0.07,      # a stack keeps `spare` inside the walls it stands on,
         "over": 0.95, "clear": 2.3}                         # and its top `over` above every roof `clear` near it
SMOOTH_DEG = 50.0               # thatch shades smoothly across edges sharper than this
SLIVER_M = 0.004                # no triangle is thinner than this: 16-bit quantization could turn it over
WELD_M = 2 * SLIVER_M           # corners this near each other are joined...
NEEDLE_M = 0.025                # ...and so are the ends of a sliver's shortest edge, if it is shorter than this
TIDY_PASSES = 12
NO_AREA = 1e-12                 # a triangle whose corners' cross product is no longer than this covers nothing
LIT, DIM = 0.25, 0.15           # share of windows lit, and dimly lit, at dusk
SOFFIT_GAP_M = 0.08             # an opening's head keeps this far under the eaves' soffit
MIN_OPENING_M = 0.6             # a window the eaves would leave lower than this is left out
DOOR_MIN_M = 1.7
CLEAR_M = 3.0                   # a window opens only where nothing of the village stands this near in front of it
DOOR_CLEAR_M = 0.6              # ...and a door where nothing stands this near
PARTY_SHARE = 0.5               # a wall standing against other buildings for this share of its length is a party wall
NEIGHBOUR_M = 1.0               # buildings this near a building stop its eaves and verges...
NEIGHBOUR_GAP_M = 0.005         # ...this far inside their walls
SKIN_RETRY_M = 0.003            # a roof skin the roofer fails on is tried again with its overhang this much wider
DORMERS = 2                     # at most this many dormers on a front
DORMER = {"cheek": 0.25, "below": 0.3, "window": 1.0, "lintel": 0.25, "tail": 0.25, "end": 0.7, "gap": 0.6,
          "bury": 0.1}                  # a dormer's back stands at least "bury" under the roof behind it
DORMER_WIDEST_M = 1.0 + 2 * DORMER["cheek"]
DEPTH_GAP_M = 0.05              # a face that would otherwise lie in the plane of a building's face stands this far off
                                # it, clear of it in a 24-bit depth buffer from the camera's near metre out to some
                                # 900 m: a coarser level's flat windows and doors before their walls, its boxes of stacks
                                # flush with a gable behind it, a turret's base behind its gable, crow steps off the
                                # gable's corners


class Level(NamedTuple):
    """What a level of detail builds of each building and of the small structures, from the same records as the full
    level, so that whatever places a building places it at every level."""
    roof: str           # "skin": the attic under its overhanging skin; "hull": the skin's outer solid (trimmed back to
                        # the gable walls where the level shows openings); "attic": the attic alone, within the walls
    openings: str       # "cut" into the walls with their reveals and painted margins, "flat" panels on them, or "none"
    stacks: str         # "joined" into the roofs, "boxes" standing in them, or "none"
    features: bool      # base courses, dormers, a named building's bays, buttresses and roof features (turrets, the
                        # kiln's ventilator, the owl loft, crow steps); the kirkyard's and the gardens' walls
    fittings: bool      # chimney pots, sills, steps, signs, fascias and the like; lamps and headstones
    casts: bool         # what stands out beyond the eaves and casts a shadow of its own, whatever else the level keeps:
                        # the fittings hung to cast (_Building.cast: Gladrags' blind, the Shack's fence), the kirkyard's
                        # wall, and the gardens' walls (where the level has no features, one slab along each garden)


LEVELS = {
    "lod0": Level("skin", "cut", "joined", True, True, True),
    "lod1": Level("hull", "flat", "boxes", True, False, False),
    "lod2": Level("hull", "none", "none", False, False, False),
    "shadow": Level("attic", "none", "none", False, False, True),
}
FULL = LEVELS["lod0"]

PALETTE = {"bright": (("harl-white", 3.0), ("harl-cream", 2.0), ("harl-ochre", 2.0), ("harl-pink", 1.5),
                      ("harl-blue", 1.5), ("harl-green", 1.0), ("harl-terracotta", 1.0)),
           "plain": (("harl-white", 5.5), ("harl-cream", 2.0)),
           "cottage": (("harl-white", 5.5), ("harl-cream", 1.0))}
DOORS = ("door-green", "door-red", "door-blue", "door-black")
GLASS = ("glass-dark", "glass-lit", "glass-dim", "glass-shop")
ROOFING = ("village-thatch", "village-thatch-new", "village-slate", "corrugated-iron")
FASCIAS = ("fascia-green", "fascia-maroon", "fascia-navy")
MARGINS = ("margin-black", "margin-oxblood")


@dataclass(frozen=True)
class Style:
    """One building's style record."""
    family: str             # shop, terrace, cottage-dormer, cottage, villa, outshot
    storeys: float          # 1, 1.5 (upper windows in dormers through the eaves) or 2
    eaves: float            # wall height from the floor to the eaves, metres
    wall: str               # material names
    plinth: str
    roof: str               # "thatch" or "slate"
    roof_material: str
    pitch: float            # degrees
    hips: bool              # free ends hipped rather than gabled
    shopfront: str | None   # fascia material of a shopfront, or None
    margins: str | None     # paint round the front's openings, or None
    door: str
    sign: bool              # a hanging sign over the pavement


def _pick(rng, weighted):
    names, weights = zip(*weighted)
    return names[int(rng.choice(len(names), p=np.asarray(weights) / sum(weights)))]


def choose_style(rng, family, named=None, previous=None):
    """A style record for a unit of `family`, varied by `rng`; `previous` (the last unit of the same row) never
    shares its wall colour and storeys with this one."""
    for _ in range(4):
        thatch = rng.random() < {"shop": 0.35, "terrace": 0.85, "cottage-dormer": 0.7, "cottage": 0.9, "villa": 0.0,
                                 "outshot": 0.0}.get(family, 0.0)
        if family == "shop":
            storeys, eaves = 2.0, 5.9 + rng.uniform(-0.3, 0.3)
            wall = "rubble" if rng.random() < 0.25 else _pick(rng, PALETTE["bright"])
        elif family == "terrace":
            storeys = 1.5 if rng.random() < 0.5 else 2.0
            eaves = (3.2 if storeys == 1.5 else 5.3) + rng.uniform(-0.2, 0.2)
            wall = "rubble" if rng.random() < 0.25 else _pick(rng, PALETTE["plain"])
        elif family == "cottage-dormer":
            storeys, eaves = 1.5, 3.1 + rng.uniform(-0.2, 0.2)
            wall = "rubble" if rng.random() < 0.5 else _pick(rng, PALETTE["plain"])
        elif family == "villa":
            storeys, eaves = 2.0, 5.6 + rng.uniform(-0.2, 0.2)
            wall = "rubble" if rng.random() < 0.3 else _pick(rng, PALETTE["bright"])
        elif family == "outshot":
            storeys, eaves = 1.0, 2.3
            wall = "rubble"
        else:
            storeys, eaves = 1.0, 2.6 + rng.uniform(-0.2, 0.2)
            wall = "rubble" if rng.random() < 0.45 else _pick(rng, PALETTE["cottage"])
        if previous is None or (wall, storeys) != (previous.wall, previous.storeys):
            break
    else:
        wall = "harl-white" if previous.wall == "rubble" else "rubble"             # still alike: change the colour
    roof = "thatch" if thatch else "slate"
    shop = family == "shop" and named not in INNS and (named is not None or rng.random() < 0.7)
    return bespoke.restyle(named, Style(
        family=family, storeys=storeys, eaves=round(eaves, 2), wall=wall,
        plinth="rubble" if wall == "rubble" else "plinth-dark", roof=roof,
        roof_material=("village-thatch" if rng.random() < 0.75 else "village-thatch-new") if thatch else "village-slate",
        pitch=round((50.0 if thatch else 40.0) + rng.uniform(-2.0, 2.0), 1),
        hips=family in ("cottage", "cottage-dormer", "villa") and rng.random() < 0.25,
        shopfront=FASCIAS[int(rng.integers(len(FASCIAS)))] if shop else None,
        margins=MARGINS[int(rng.integers(len(MARGINS)))] if family == "shop" and rng.random() < 0.5 else None,
        door=DOORS[int(rng.integers(len(DOORS)))], sign=family == "shop" and (shop or named in INNS)))


@dataclass
class Unit:
    """A building: its plan outline in its footprint part's frame, which of its edges are party walls and which is
    its front, the asset it belongs to and its style."""
    cid: int
    part: int
    k: int
    fp: plan.Footprint
    outline: Polygon
    party: frozenset
    front: int
    high: bool
    asset: str
    style: Style


# ---------------------------------------------------------------- the village, once per build
def build_footprint(scene, asset, c, polygons):
    village = _village(scene)
    if c["id"] not in village.built:
        return None
    return village.component(c["id"])


def build_extras(scene):
    """The village's structures on no mapped footprint: the High Street's lamps, the kirkyard's wall and headstones,
    and the garden walls near the High Street."""
    return _village(scene).small()


def levels(village):
    """Every level of detail of the village (LEVELS), world space: {level: {asset: pieces}}, each built footprint's and
    the small structures it keeps. The payload carries each level that passes its guards and saves enough triangles
    (mesh_levels.village_units), and the page chooses among them for each view."""
    out = {}
    for name, level in LEVELS.items():
        found = {}
        for cid in sorted(village.built):
            for asset, pieces in village.component(cid, level).items():
                found.setdefault(asset, []).extend(pieces)
        for asset, pieces in village.small(level).items():
            found.setdefault(asset, []).extend(pieces)
        out[name] = found
    return out


@lru_cache(maxsize=2)
def _village(scene):
    return Village(scene)


def _moved(frame, c, at, to, turn):
    """A footprint part's `frame` turned `turn` degrees about `at`, its component c's decoded centre, and carried with
    that point to `to` (MOVED)."""
    from build_meshes import CLAIM_TOLERANCE_M
    if math.dist(at, (c["x"], c["z"])) > CLAIM_TOLERANCE_M:
        raise ValueError(f"component {c['id']} is no longer where the village moves it from")
    t = math.radians(turn)
    dx, dz = frame.ox - at[0], frame.oz - at[1]
    return plan.Frame(to[0] + dx * math.cos(t) - dz * math.sin(t), to[1] + dx * math.sin(t) + dz * math.cos(t),
                      frame.theta + t)


class Village:
    def __init__(self, scene):
        self.scene = scene
        family = {a["id"]: a["family"] for a in scene.manifest["assets"]}
        self.components = {c["id"]: c for c in scene.labelled[1]
                           if scene.owners.get(c["id"]) and family[scene.owners[c["id"]]] == FAMILY}
        self.streets = self._streets()
        self.parts = {}
        for cid, c in sorted(self.components.items()):
            traces = scene.footprint(c)
            found = [fp for k, t in enumerate(traces) for fp in plan.plan_footprint(t, cid * 31 + k)]
            self.parts[cid] = [fp for fp in found if fp.outline.area >= plan.MIN_PART_M2] or \
                sorted(found, key=lambda fp: -fp.outline.area)[:1]                  # specks go, unless nothing else
            if cid in MOVED:
                self.parts[cid] = [replace(fp, frame=_moved(fp.frame, c, *MOVED[cid])) for fp in self.parts[cid]]
        self.outlines = {cid: shapely.union_all([fp.frame.to_world(fp.outline) for fp in fps])
                         for cid, fps in self.parts.items()}
        self._standing_cids = sorted(self.outlines)
        self._standing = [self.outlines[c] for c in self._standing_cids]
        self._standing_index = shapely.STRtree(self._standing)
        labels = self._labels()
        self.built = {cid for cid in self.outlines
                      if max((fp.outline.area for fp in self.parts[cid]), default=0.0) >= plan.MIN_PART_M2}
        self.units = {cid: self._units(cid, labels) for cid in sorted(self.built)}
        self._ridges = {}
        self._attics = {}
        self._buildings = {}
        everyone = [u for cid in sorted(self.units) for u in self.units[cid]]
        self._index = (shapely.STRtree([u.fp.frame.to_world(u.outline) for u in everyone]), everyone)

    def ridge(self, u):
        """A building's highest roof point (world height), computed once."""
        key = (u.cid, u.part, u.k)
        if key not in self._ridges:
            self._ridges[key] = ridge(self.scene, u)
        return self._ridges[key]

    def eave(self, u):
        """A building's eaves (world height): its floor's, FLOOR_OVER_M above the highest ground along its front, and
        its style's eaves above that."""
        return floor_level(self.scene, u) + FLOOR_OVER_M + u.style.eaves

    def attic(self, u):
        """attic_top for a building's own outline and pitches, computed once."""
        key = (u.cid, u.part, u.k)
        if key not in self._attics:
            self._attics[key] = attic_top(u.outline, roof_angles(u))
        return self._attics[key]

    def ridges_near(self, xz, radius):
        """The ridges (world heights) of the built buildings within `radius` of a world point."""
        tree, everyone = self._index
        hits = tree.query(Point(xz).buffer(radius))
        return [self.ridge(everyone[i]) for i in sorted(hits)
                if everyone[i].fp.frame.to_world(everyone[i].outline).distance(Point(xz)) <= radius]

    def open_ahead(self, xz, out, clear=CLEAR_M):
        """Whether nothing of the village stands within `clear` in front of a wall's point xz facing `out` (world)."""
        p, d = np.asarray(xz, np.float64), np.asarray(out, np.float64)
        ahead = LineString([p + d * 0.05, p + d * clear])
        return not any(self._standing[i].intersects(ahead) for i in self._standing_index.query(ahead))

    def clear_of_buildings(self, xz, clear):
        """Whether no building stands within `clear` of a world point."""
        near = Point(np.asarray(xz, np.float64)).buffer(clear)
        return not any(self._standing[i].intersects(near) for i in self._standing_index.query(near))

    def neighbours(self, u):
        """The footprints of the buildings within NEIGHBOUR_M of unit u (the other units of its footprint, and the
        village's other footprints), in u's plan."""
        f = u.fp.frame
        out = [w.outline if w.fp is u.fp else f.to_plan(w.fp.frame.to_world(w.outline))
               for w in self.units.get(u.cid, []) if w is not u]
        hits = self._standing_index.query(f.to_world(u.outline).buffer(NEIGHBOUR_M))
        out += [f.to_plan(self._standing[i]) for i in sorted(hits) if self._standing_cids[i] != u.cid]
        return [g for g in out if g.distance(u.outline) < NEIGHBOUR_M]

    def _streets(self):
        s = self.scene
        road = s.locA[..., CHANNELS["locA"].index("road")]
        ii, jj = np.nonzero(road > 127)
        rx, rz = s.local.to_world(ii, jj)
        pts = np.column_stack([rx, rz])
        traced = next(t for t in s.assets["street-furniture"]["traced"] if t["id"] == "high-street")["plan_px"]
        a, b = plan.hogs_world(np.asarray(traced, np.float64), s.georef)
        axis = (b - a) / np.linalg.norm(b - a)
        perp = np.array([-axis[1], axis[0]])
        line = []
        for t in np.arange(0.0, np.linalg.norm(b - a), 6.0):
            c = a + axis * t
            d = pts - c
            near = (np.abs(d @ axis) < 4.0) & (np.abs(d @ perp) < HIGH_STREET_SPAN_M)
            if near.sum() >= 4:
                line.append(c + perp * float(np.median(d[near] @ perp)))
        return plan.Streets(pts, line)

    def _labels(self):
        """Named asset -> world point of its label's pointer on the Hogsmeade map."""
        out = {}
        for asset in SHOPS:
            traced = self.scene.assets[asset].get("traced", [])
            label = next((t for t in traced if t["id"] == "label"), None)
            if label:
                out[asset] = plan.hogs_world(np.asarray(label["plan_px"], np.float64), self.scene.georef)
        return out

    def _units(self, cid, labels):
        owner = self.scene.owners[cid]
        named = owner if owner in SHOPS else None
        units = []
        families = {}
        for part, fp in enumerate(self.parts[cid]):
            planned = bespoke.layout(owner, fp)          # a named building planned its own way
            for k, (u, family) in enumerate(planned or [(u, None) for u in fp.units]):
                if family:
                    families[len(units)] = family
                units.append((part, k, fp, u))
        target = None
        if named:
            label = labels.get(named)
            pts = [fp.frame.to_world(u) for _, _, fp, u in units]
            target = min(range(len(units)), key=lambda i: pts[i].distance(Point(label)) if label is not None
                         else -pts[i].area)
        targets = {target}
        pieces = bespoke.split(named, units[target][3]) if target is not None else None
        if pieces:                                     # a named building divided its own way
            part, k, fp, _ = units[target]
            spare = max(k2 for _, k2, _, _ in units) + 1
            units[target:target + 1] = [(part, k if j == 0 else spare + j - 1, fp, poly)
                                        for j, (poly, _) in enumerate(pieces)]
            targets = set(range(target, target + len(pieces)))
            families.update({target + j: family for j, (_, family) in enumerate(pieces)})
        others = [w for c, w in self.outlines.items() if c != cid and w.distance(self.outlines[cid]) < 1.0]
        others += [fp.frame.to_world(fp.outline) for fp in self.parts[cid]]
        out, previous = [], None
        for i, (part, k, fp, u) in enumerate(units):
            own_part = fp.frame.to_world(fp.outline)
            mates = fp.frame.to_world(shapely.union_all([u2 for m, (_, _, fp2, u2) in enumerate(units)
                                                         if m != i and fp2 is fp]))
            beside = shapely.union_all([mates.buffer(0.05)] + [o.buffer(0.35) for o in others if not o.equals(own_part)])
            party = set()
            for e, (a, b) in enumerate(plan.edges(u)):
                edge = LineString(fp.frame.world(np.array([a, b])))
                if edge.intersection(beside).length >= PARTY_SHARE * edge.length:
                    party.add(e)
            front, high = plan.front_edge(u, party, self.streets, to_world=fp.frame.world)
            is_named = i in targets
            asset = owner if (is_named or owner in (WING,) or not named) else HOUSE
            if i in families:
                family = families[i]
            elif owner == WING:
                family = "outshot"
            elif high or is_named:
                family = "shop"
            elif party:
                family = "terrace"
            elif u.area > 140.0:
                family = "villa"
            else:
                family = "cottage-dormer"
            rng = np.random.default_rng([cid, part, k, 7])
            own = owner if is_named or owner in BUILDINGS else None
            style = choose_style(rng, family, own, previous if party else None)
            unit = Unit(cid, part, k, fp, u, frozenset(party), front, high, asset, style)
            if style.storeys == 1.5 and attic_window(unit, roof_angles(unit), self.opener(fp.frame, u)) is None:
                style = choose_style(rng, "cottage", None, previous if party else None)   # no room for an attic window
                unit = Unit(cid, part, k, fp, u, frozenset(party), front, high, asset, style)
            previous = style
            out.append(unit)
        return out

    def opener(self, frame, outline):
        """is_open(k, s, w=0, clear=CLEAR_M) for a plan outline in `frame`: whether the way ahead of its edge k is open
        for `clear` across an opening w wide centred s along it (and a hand's breadth either side)."""
        es = plan.edges(outline)

        def is_open(k, s, w=0.0, clear=CLEAR_M):
            a, b = es[k]
            t = (b - a) / np.linalg.norm(b - a)
            out = frame.world(a + plan.outward(a, b)) - frame.world(a)
            return all(self.open_ahead(frame.world(a + t * x), out, clear)
                       for x in (s - w / 2 - 0.1, s, s + w / 2 + 0.1))
        return is_open

    # ------------------------------------------------------------ geometry
    def component(self, cid, level=FULL):
        """{asset: [pieces]} for one built component at a level of detail."""
        pieces = {}
        for part, fp in enumerate(self.parts[cid]):
            units = [u for u in self.units[cid] if u.part == part]
            for asset in sorted({u.asset for u in units}):
                group = [u for u in units if u.asset == asset]
                shell, fittings = self._build(fp, group, level)
                pieces.setdefault(asset, []).extend([shell] + fittings)
        return pieces

    def small(self, level=FULL):
        """{asset: [pieces]} of the structures on no mapped footprint that a level of detail keeps: the lamps and the
        headstones are fittings, the kirkyard's wall and the garden walls features; both walls cast, the gardens' as
        one slab along each garden where the level keeps no features."""
        import mesh_village_small as small
        gardens = small.garden_walls(self) if level.features else small.garden_walls(self, whole=True) if level.casts \
            else []
        return {"street-furniture": small.lamps(self) if level.fittings else [],
                "kirk": (small.kirkyard_wall(self) if level.features or level.casts else [])
                + (small.headstones(self) if level.fittings else []),
                HOUSE: gardens}

    def buildings(self, fp, units):
        """(the reference ground height, the _Building of each unit) of a group of units of footprint part fp, built
        once: every level of detail is cut from the same records."""
        key = tuple((u.cid, u.part, u.k) for u in units)
        if key not in self._buildings:
            ring = np.asarray(fp.outline.exterior.coords)
            y_ref = round(float(np.median(self.scene.ground(*fp.frame.world(ring).T))), 2)
            self._buildings[key] = y_ref, [_Building(self, fp, u, y_ref) for u in units]
        return self._buildings[key]

    def _build(self, fp, units, level=FULL):
        import manifold3d as m3

        y_ref, buildings = self.buildings(fp, units)
        solids, cutters, fittings = [], [], []
        for b in buildings:
            found = b.at(level)
            solids += found[0]
            cutters += found[1]
            fittings += found[2]
        specs = {}
        tagged = []
        for solid, spec in solids:
            o = solid.as_original()
            specs[o.original_id()] = spec
            tagged.append(o)
        whole = m3.Manifold.batch_boolean(tagged, m3.OpType.Add)
        if cutters:
            cut = []
            for solid, spec in cutters:
                o = solid.as_original()
                specs[o.original_id()] = spec
                cut.append(o)
            whole = whole - m3.Manifold.batch_boolean(cut, m3.OpType.Add)
        mesh = whole.to_mesh()
        v = np.asarray(mesh.vert_properties, np.float64)[:, :3]
        f = np.asarray(mesh.tri_verts, np.int64)
        tri_id = np.repeat(np.asarray(mesh.run_original_id), np.diff(np.asarray(mesh.run_index)) // 3)
        v, f, tri_id = _tidy(v, f, tri_id)
        return _shaded(v, f, tri_id, specs, fp.frame, y_ref), [placed_world(m, fp.frame, y_ref) for m in fittings]


def _tidy(v, f, tri_id):
    """The solid's corners and triangles (indices into them) without slivers, still closed. Corners nearer each other
    than WELD_M are joined, and so are the ends of a sliver's edge shorter than NEEDLE_M, which drops needle-like
    triangles; a triangle whose corner lies almost on its longest edge goes, the triangle across that edge split at
    that corner. A sliver could turn over when the payload quantizes it, and dropping it would leave a crack. Where
    splits only trade one sliver for another (its corners and its neighbours' all but in line), the sliver's apex moves
    onto its longest side, at most SLIVER_M, unless that would turn another triangle at it over: covering nothing then,
    the sliver is left out as the payload's encoder would leave it out, and its sides meet in a line. Two slivers whose
    longest sides meet at a corner, sharing an apex in the wedge between them, would each lift it off the other's side:
    it joins that corner instead, the one place on both, and the pair goes. Of the triangles that joining brings onto
    the same corners only pairs wound opposite ways go, so no edge is left odd."""
    from scipy.spatial import cKDTree

    v = np.array(v, np.float64)
    root = np.arange(len(v))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    def join(pairs):
        for a, b in sorted(pairs):
            ra, rb = find(a), find(b)
            if ra != rb:
                root[max(ra, rb)] = min(ra, rb)

    join(map(tuple, cKDTree(v).query_pairs(WELD_M, output_type="ndarray")))

    def measure(f, tri_id):
        f = np.array([find(i) for i in range(len(v))])[f]
        keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])
        f, tri_id = f[keep], tri_id[keep]
        _, twin, count = np.unique(np.sort(f, axis=1), axis=0, return_inverse=True, return_counts=True)
        twin = twin.ravel()
        first = np.argmin(f, axis=1)
        rows = np.arange(len(f))
        up = f[rows, (first + 1) % 3] < f[rows, (first + 2) % 3]      # which way round its corners run
        keep = np.ones(len(f), bool)
        for g in np.flatnonzero(count > 1):
            same = np.flatnonzero(twin == g)
            a, b = same[up[same]], same[~up[same]]
            pairs = min(len(a), len(b))
            keep[a[:pairs]] = keep[b[:pairs]] = False    # two on the same corners, wound opposite ways, enclose nothing
        f, tri_id = f[keep], tri_id[keep]
        tris = v[f]
        lengths = np.stack([np.linalg.norm(tris[:, (i + 1) % 3] - tris[:, i], axis=1) for i in range(3)], axis=1)
        area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
        return f, tri_id, lengths, area <= SLIVER_M * lengths.max(axis=1)

    for _ in range(TIDY_PASSES):
        f, tri_id, lengths, thin = measure(f, tri_id)
        needles = np.flatnonzero(thin & (lengths.min(axis=1) < NEEDLE_M))
        if len(needles):
            ends = [(int(f[t][e]), int(f[t][(e + 1) % 3])) for t, e in zip(needles, np.argmin(lengths[needles], axis=1))]
            join(tuple(sorted(p)) for p in ends)
            f, tri_id, lengths, thin = measure(f, tri_id)
        if not thin.any():
            break
        across = {}
        for t, (i, j, k) in enumerate(f):
            across[(i, j)], across[(j, k)], across[(k, i)] = t, t, t
        gone, new_f, new_id = set(), [], []
        for t in np.flatnonzero(thin):
            e = int(np.argmax(lengths[t]))
            a, b, c = f[t][e], f[t][(e + 1) % 3], f[t][(e + 2) % 3]
            n = across.get((b, a))
            if t in gone or n is None or n in gone:
                continue
            d = next((x for x in f[n] if x != a and x != b), c)
            gone |= {t, n}
            if d != c:
                new_f += [(b, c, d), (c, a, d)]
                new_id += [tri_id[n], tri_id[n]]
        if not gone:
            break
        kept = np.array([t not in gone for t in range(len(f))])
        f = np.vstack([f[kept], np.asarray(new_f, np.int64).reshape(-1, 3)])
        tri_id = np.concatenate([tri_id[kept], np.asarray(new_id, tri_id.dtype)])
    normal = lambda p: np.cross(p[1] - p[0], p[2] - p[0])
    facing = None                                   # each triangle's normal before any apex moved, by which it was
    for _ in range(TIDY_PASSES):
        f, tri_id, lengths, thin = measure(f, tri_id)
        if not thin.any():
            break
        if facing is None:
            tris = v[f]
            facing = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            tri_id = np.column_stack([tri_id, np.arange(len(f), dtype=tri_id.dtype)])
        slivers = [int(t) for t in np.flatnonzero(thin)]
        sides = {t: [int(f[t][(int(np.argmax(lengths[t])) + i) % 3]) for i in range(3)] for t in slivers}
        moved = False
        for t in slivers:
            a, b, c = sides[t]
            around = [int(s) for s in np.flatnonzero((f == c).any(axis=1)) if s != t]

            def keeps(s, to):               # still facing as it did before any apex moved, which one with no area can't
                n0 = facing[tri_id[s, 1]]
                return np.linalg.norm(n0) > NO_AREA and n0 @ normal(np.where((f[s] == c)[:, None], to, v[f[s]])) > 0
            # another sliver at this apex whose longest side meets this one's at a corner: laid on either side, the apex
            # lifts off the other, so it joins that corner, the one place on both, and the two slivers go
            corner = next((k for s in around if sides.get(s, (None,) * 3)[2] == c for k in {a, b} & set(sides[s][:2])),
                          None)
            ab = v[b] - v[a]
            to = v[a] + ab * float((v[c] - v[a]) @ ab / (ab @ ab)) if corner is None else v[corner].copy()
            stay = around if corner is None else [s for s in around if corner not in f[s]]
            if np.any(to != v[c]) and all(keeps(s, to) for s in stay):
                v[c], moved = to, True
                if corner is not None:      # joined: the triangles round the corner are measured afresh next pass
                    root[c] = corner
                    break
        if not moved:
            break
    return v, f, tri_id if facing is None else tri_id[:, 0]


def placed_world(mesh, frame, y_ref):
    return Mesh(frame.world3(mesh.pos, y_ref), mesh.nrm @ frame.rot3.T, mesh.attr, mesh.idx)


def _shaded(v, f, tri_id, specs, frame, y_ref):
    """The solid's triangles (plan space) as a world Mesh: each face's (material, aux) from its solid's spec, judged
    on its plan normal; flat-shaded, thatch smoothed across its softer edges. A triangle with no area, which covers
    nothing and has no normal to give its corners, is left out, as the payload's encoder would leave it out."""
    tris = v[f]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = np.linalg.norm(n, axis=1)
    flat = area > NO_AREA
    tris, n, tri_id = tris[flat], n[flat] / area[flat, None], tri_id[flat]
    attr = np.array([specs[t](nn) for t, nn in zip(tri_id, n)], np.uint8).reshape(-1, 2)
    tris = frame.world3(tris.reshape(-1, 3), y_ref).reshape(-1, 3, 3)
    n = n @ frame.rot3.T
    thatch = np.isin(attr[:, 1], [code("village-thatch")[1], code("village-thatch-new")[1]]) & (attr[:, 0] == 7)
    meshes = []
    for mat in np.unique(attr, axis=0):
        sel = (attr == mat).all(axis=1)
        if thatch[sel].all():
            meshes.append(_smooth(tris[sel], n[sel], tuple(int(x) for x in mat)))
        else:
            meshes.append(indexed(tris[sel], n[sel][:, None, :], tuple(int(x) for x in mat)))
    return concat(meshes)


def _smooth(tris, n, attr):
    """Triangles sharing a corner position average their normals where they turn by less than SMOOTH_DEG."""
    corners = tris.reshape(-1, 3)
    key = np.round(corners * 1000).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    face_n = np.repeat(n, 3, axis=0)
    out = np.empty_like(face_n)
    cos = math.cos(math.radians(SMOOTH_DEG))
    order = np.argsort(inv, kind="stable")
    bounds = np.flatnonzero(np.diff(inv[order])) + 1
    for group in np.split(order, bounds):
        g = face_n[group]
        sim = (g @ g.T) > cos
        acc = sim.astype(np.float64) @ g
        out[group] = acc / np.linalg.norm(acc, axis=1, keepdims=True)
    return indexed(tris, out.reshape(-1, 3, 3), attr)


# ---------------------------------------------------------------- one building
POT_M, POT_H = 0.22, 0.5


def floor_level(scene, u):
    """A building's floor (world height): the highest ground along its front, which its door opens onto."""
    a, b = plan.edges(u.outline)[u.front]
    n = plan.outward(a, b)
    along = np.array([a + (b - a) * t + n * 0.6 for t in np.linspace(0.05, 0.95, 9)])
    return float(np.max(scene.ground(*u.fp.frame.world(along).T)))


def roof_angles(u):
    """Each edge's pitch: party walls are gables, and so are a rectangle's short ends unless it is hipped; a building
    its neighbours enclose slopes along its long sides regardless. A storey-and-a-half rectilinear front too short for
    a dormer between its ends is a gable, its attic window in the gable, unless an end is an inside corner: the roof
    beyond a gable there would have to fold back over itself. A named building may roof itself its own way
    (mesh_village_named)."""
    own = bespoke.roof_angles(u)
    if own is not None:
        return own
    poly = u.outline
    x0, y0, x1, y1 = poly.bounds
    long_x = (x1 - x0) >= (y1 - y0)
    rect = len(poly.exterior.coords) == 5 and abs(poly.area - (x1 - x0) * (y1 - y0)) < 1e-6
    edges = plan.edges(poly)
    out = []
    for k, (a, b) in enumerate(edges):
        d = (b - a) / np.linalg.norm(b - a)
        across = abs(d[0]) < 0.2 if long_x else abs(d[1]) < 0.2
        out.append(90.0 if k in u.party or (not u.style.hips and rect and across) else u.style.pitch)
    k, m = u.front, len(out)
    rectilinear = all(min(abs((b - a)[0]), abs((b - a)[1])) < 1e-6 for a, b in edges)
    inside = [float(np.dot(edges[j][1] - edges[j][0], plan.outward(*edges[j - 1]))) > 1e-9 for j in (k, (k + 1) % m)]
    if u.style.storeys == 1.5 and out[k] < 89.999 and rectilinear and not any(inside):
        a, b = edges[k]
        wd = DORMER_WIDEST_M
        ends = sum(dormer_depth(u.style.pitch, wd) if out[(k + j) % m] < 89.999 else DORMER["end"] for j in (-1, 1))
        if np.linalg.norm(b - a) < wd + ends:
            out[k] = 90.0
    if all(a >= 89.999 for a in out):
        out = [90.0 if k in u.party else u.style.pitch for k in range(len(out))]   # a terrace house: party gables
        if all(a >= 89.999 for a in out):
            out = [u.style.pitch] * len(out)                                         # enclosed: hipped all round
    return out + [u.style.pitch] * sum(len(r.coords) - 1 for r in poly.interiors)   # courtyards slope inward


def ridge(scene, u):
    """A building's highest roof point (world height): its floor, its eaves and its attic's apex."""
    v, _ = roofs.roof(u.outline, roof_angles(u))
    return floor_level(scene, u) + FLOOR_OVER_M + u.style.eaves + float(v[:, 2].max())


def dormer_depth(pitch, wd):
    """How far a dormer wd wide must run back into a roof of `pitch` for its hood's ridge to meet the slope short of
    its back."""
    rise = DORMER["window"] - DORMER["below"] + DORMER["lintel"]          # the dormer's top above the eaves
    return rise / math.tan(math.radians(pitch)) + wd / 2 + DORMER["tail"]


def bays(length):
    """(the bay centres of a front `length` long, measured from its start; the bays' pitch)."""
    n = max(1, int((length - 1.0) // BAY_M))
    if n == 1 and length >= 4.0:
        n = 2                                   # room for a window beside the door
    pitch_m = (length - 1.0) / n
    return [0.5 + pitch_m * (i + 0.5) for i in range(n)], pitch_m


def attic_top(outline, angles):
    """The height of the roof over `outline` (roofs.roof with `angles`, its eaves at 0) above plan points (n, 2), as a
    function of them; -inf off it."""
    v, f = roofs.roof(outline, angles)
    tris = v[f]
    a, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    den = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    keep = np.abs(den) > 1e-9                            # upright faces have no plan area
    a, e1, e2, den = a[keep], e1[keep], e2[keep], den[keep]

    def at(pts):
        d = np.asarray(pts, np.float64)[:, None, :2] - a[None, :, :2]
        u = (d[..., 0] * e2[:, 1] - d[..., 1] * e2[:, 0]) / den
        w = (e1[:, 0] * d[..., 1] - e1[:, 1] * d[..., 0]) / den
        z = a[:, 2] + u * e1[:, 2] + w * e2[:, 2]
        inside = (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1.0 + 1e-9)
        return np.where(inside, z, -np.inf).max(axis=1)
    return at


def attic_window(u, angles, is_open=lambda k, s, w=0.0: True):
    """How a storey-and-a-half unit lights its attic from its front, heights above its eaves: ("dormers", [where along
    the front], window width) on an eaved front, or ("gable", where, sill, width, height) in a gable front; None if
    neither fits. A High Street front takes at most DORMERS dormers, over its bays or else one over its middle; any
    other front one, over its middle or else the bay nearest it. Each is kept off a hipped end by its own depth and off
    a gable by DORMER['end'], and only where the roof behind it stands high enough to bury its back and the way ahead
    `is_open(k, s)`. A gable window keeps under its verge."""
    k = u.front
    a, b = plan.edges(u.outline)[k]
    length = float(np.linalg.norm(b - a))
    t, n = (b - a) / length, plan.outward(a, b)
    centres, pitch_m = bays(length)
    w = min(1.0, pitch_m - 0.6)
    if w < 0.45:
        return None
    top = attic_top(u.outline, angles)
    if angles[k] >= 89.999:
        dims = THATCH if u.style.roof == "thatch" else SLATE
        w = min(w, 0.9)
        margin = MARGIN_M if u.style.margins else 0.0
        mid = a + t * (length / 2) - n * 0.02
        under = float(top(np.array([mid - t * (w / 2 + margin), mid + t * (w / 2 + margin)])).min()) \
            - dims["thick"] / math.cos(math.radians(u.style.pitch)) + EPS_M
        h = min(DORMER["window"], under - SOFFIT_GAP_M - margin - DORMER["lintel"])
        if h < MIN_OPENING_M or not is_open(k, length / 2, w):
            return None
        return "gable", length / 2, DORMER["lintel"], w, h
    wd = w + 2 * DORMER["cheek"]
    dd = dormer_depth(u.style.pitch, wd)
    rise = DORMER["window"] - DORMER["below"] + DORMER["lintel"]
    apex = rise + wd / 2 * math.tan(math.radians(u.style.pitch))
    m = len(plan.edges(u.outline))
    ends = [dd if angles[(k + j) % m] < 89.999 else DORMER["end"] for j in (-1, 1)]

    def fits(s):
        if s - wd / 2 < ends[0] or s + wd / 2 > length - ends[1]:
            return False
        back = a + t * s - n * dd
        h = top(np.array([back - t * wd / 2, back + t * wd / 2, back]))
        return bool(min(h[0], h[1]) >= rise + DORMER["bury"] and h[2] >= apex + DORMER["bury"]) and is_open(k, s, wd)

    picks = [s for s in centres if fits(s)]
    if not u.high:                                     # off the High Street, one dormer, over the middle if it can be
        picks = [length / 2] if fits(length / 2) else sorted(picks, key=lambda s: abs(s - length / 2))[:1]
    elif len(picks) > DORMERS:
        picks = [picks[round(i * (len(picks) - 1) / (DORMERS - 1))] for i in range(DORMERS)]
    if any(q - p < wd + DORMER["gap"] for p, q in zip(picks, picks[1:])):
        picks = picks[:1]
    if not picks and fits(length / 2):
        picks = [length / 2]
    return ("dormers", picks, w) if picks else None


def _outward(mesh, centre):
    """The mesh with every face turned away from `centre` (a convex piece)."""
    tris = mesh.pos[mesh.idx]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    away = np.einsum("ij,ij->i", n, tris.mean(axis=1) - centre) >= 0
    n = np.where(away[:, None], n, -n)
    return indexed(oriented(tris, n), n[:, None, :], tuple(int(x) for x in mesh.attr[0]))


def _apart(solid, spec):
    """A solid standing apart from the building's shell (a stack in its roof), plan space, flat-shaded in its spec's
    (material, aux): without the faces facing straight down, which the roof it stands in buries."""
    mesh = solid.to_mesh()
    tris = np.asarray(mesh.vert_properties, np.float64)[:, :3][np.asarray(mesh.tri_verts, np.int64)]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    up = n[:, 2] > -0.99
    tris, n = tris[up], n[up]
    attr = np.array([spec(x) for x in n], np.uint8).reshape(-1, 2)
    return concat([indexed(tris[(attr == a).all(axis=1)], n[(attr == a).all(axis=1)][:, None, :],
                           tuple(int(x) for x in a)) for a in np.unique(attr, axis=0)])


def _panel(c, t, n, z0, profile, back):
    """An opening of a face shown flat: its profile ((along t, up from z0) points round a convex outline, centred on
    plan point c of a face running along t and facing n) DEPTH_GAP_M in front of the face, facing out, in its back
    face's material."""
    o = c + n * DEPTH_GAP_M
    ring = np.array([[o[0] + t[0] * x, o[1] + t[1] * x, z0 + z] for x, z in profile])
    tris = np.stack([np.repeat(ring[:1], len(ring) - 2, axis=0), ring[1:-1], ring[2:]], axis=1)
    out = np.array([n[0], n[1], 0.0])
    return indexed(oriented(tris, out), out, code(back))


def _pot(p, t, z0):
    """A square clay chimney pot standing on (p, z0), turned with its stack's long side t; no bottom."""
    from mesh_geometry import panel
    n = np.array([t[1], -t[0]])
    h = POT_M / 2
    q = lambda x, y, z: np.array([p[0] + t[0] * x + n[0] * y, p[1] + t[1] * x + n[1] * y, z])
    z1 = z0 + POT_H
    faces = [[q(-h, -h, z1), q(h, -h, z1), q(h, h, z1), q(-h, h, z1)]]
    corners = [(-h, -h), (h, -h), (h, h), (-h, h)]
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        faces.append([q(x0, y0, z0), q(x1, y1, z0), q(x1, y1, z1), q(x0, y0, z1)])
    return _outward(concat([panel(f, code("chimney-pot")) for f in faces]), q(0.0, 0.0, z0 + POT_H / 2))


def _material(name):
    m = code(name)
    return lambda n: m


def _by_normal(wall, roof):
    """Vertical faces take the wall's material, the rest the roof's (an attic: its gables and its slopes)."""
    w, r = code(wall), code(roof)
    return lambda n: w if abs(n[2]) < 0.25 else r


def _capped(wall, cap):
    """The top face takes the cap's material, every other face the wall's (a chimney stack)."""
    w, c = code(wall), code(cap)
    return lambda n: c if n[2] > 0.9 else w


def _cut(back, reveal, out):
    """A cutter's faces: the one facing out of the wall is what the opening shows (glass, a door); the rest are its
    reveals."""
    bk, rv, o = code(back), code(reveal), np.asarray(out, np.float64)
    return lambda n: bk if n @ o > 0.9 else rv


class _Building:
    """One building's records, from which every level of detail is cut (at): its solids, each tagged with its role
    ("walls", "attic", "skin", "stack" or "feature"), the cutters of its openings, its fittings (those its features
    include among them, and those that cast a shadow of their own), its openings' faces, the notches cut from its eaves
    for its dormers, and the boxes its stacks flush with a gable stand apart as."""

    def __init__(self, village, fp, u, y_ref):
        import manifold3d as m3

        self.m3 = m3
        self.v, self.fp, self.u, self.st, self.y_ref = village, fp, u, u.style, y_ref
        self.solids, self.cutters, self.fittings = [], [], []
        self.featured, self.casting, self.panels, self.notches, self.boxes = [], [], [], [], {}
        self.rng = np.random.default_rng([u.cid, u.part, u.k, 11])
        self.edges = plan.edges(u.outline)
        self._levels()
        self._walls()
        self._roof()
        self._openings()
        bespoke.finish(self)
        self._chimneys()
        if self.skin is not None:
            self.solids.append((self.skin, _material(self.st.roof_material), "skin"))

    def feature(self, mesh):
        """Hang a fitting that the levels with features keep (a named building's roof feature)."""
        self.fittings.append(mesh)
        self.featured.append(mesh)

    def cast(self, mesh):
        """Hang a fitting standing out beyond the eaves that casts a shadow of its own: the levels that keep what casts
        keep it (the shadow caster), whatever else they keep of the fittings."""
        self.fittings.append(mesh)
        self.casting.append(mesh)

    def at(self, level):
        """The building at a level of detail (Level), plan space: (the solids to join, as (solid, spec) pairs; the
        cutters to cut from them; the pieces that stand apart)."""
        solids, pieces = [], []
        courses = level.features                      # base courses go with the features...
        if not courses:                               # ...else the wall runs from its foot to its eaves
            solids.append((self._prism(self.u.outline, self.z_bottom, self.z_eave), _material(self.st.wall)))
        for solid, spec, role in self.solids:
            if role == "stack" and level.stacks == "boxes":
                pieces.append(_apart(self.boxes.get(id(solid), solid), spec))
            elif role == "skin" and level.roof == "skin":
                solids.append((solid if level.features else self.plain_skin, spec))
            elif (role == "walls" and courses or role == "attic" and level.roof != "hull"
                  or role == "feature" and level.features or role == "stack" and level.stacks == "joined"):
                solids.append((solid, spec))
        if level.roof == "hull":
            solids.append((self._hull(level.features, level.openings != "none"),
                           _by_normal(self.st.wall, self.st.roof_material)))
        if level.openings == "flat" and self.panels:
            pieces.append(concat([_panel(*p) for p in self.panels]))
        featured, casting = {id(m) for m in self.featured}, {id(m) for m in self.casting}
        pieces += [m for m in self.fittings if level.fittings or level.features and id(m) in featured
                   or level.casts and id(m) in casting]
        return solids, self.cutters if level.openings == "cut" else [], pieces

    def _hull(self, notched, trimmed):
        """The roof's skin filled in: its outer solid, stopping at the buildings round it as the skin does; trimmed back
        to the gable walls if `trimmed`, for its verges would bury the windows of a level that shows them; cut away in
        front of the dormers if `notched`. The attic where the roofer built no skin."""
        if self.plain_skin is None:
            return self.attic
        hull = self.outer
        if trimmed and self.trim is not None:
            box = hull.bounding_box()
            hull = hull ^ self._prism(self.trim, box[2] - 1.0, box[5] + 1.0)
        if self.near is not None:
            hull = hull - self.near
        for notch in self.notches if notched else ():
            hull = hull - notch
        return hull

    # levels: the floor from the ground along the front; walls to below the lowest ground
    def _levels(self):
        s, f = self.v.scene, self.fp.frame
        floor = floor_level(s, self.u)
        ring = np.asarray(self.u.outline.exterior.coords)
        lowest = float(np.min(s.ground(*f.world(ring).T)))
        self.z_floor = floor + FLOOR_OVER_M - self.y_ref
        self.z_bottom = min(lowest - SINK_M, floor - SINK_M) - self.y_ref
        self.z_plinth = min(self.z_floor - 0.05, lowest + PLINTH_M - self.y_ref)
        self.z_eave = self.z_floor + self.st.eaves

    def _prism(self, poly, z0, z1):
        m3 = self.m3
        contours = [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in poly.interiors]
        cs = m3.CrossSection(contours, m3.FillRule.EvenOdd)
        return m3.Manifold.extrude(cs, z1 - z0).translate((0.0, 0.0, z0))

    def _walls(self):
        o = self.u.outline
        if self.st.plinth == self.st.wall:                    # a plinth of the wall's own stone is the wall
            self.solids.append((self._prism(o, self.z_bottom, self.z_eave), _material(self.st.wall), "walls"))
            return
        self.solids.append((self._prism(o, self.z_bottom, self.z_plinth), _material(self.st.plinth), "walls"))
        self.solids.append((self._prism(o, self.z_plinth, self.z_eave), _material(self.st.wall), "walls"))

    # roofs: the attic over the walls, then the overhanging skin
    def _roof_mesh(self, poly, angles, z, cap=None):
        v, f = roofs.roof(poly, angles, cap=cap)
        mesh = self.m3.Mesh(vert_properties=np.ascontiguousarray(v, np.float32),
                            tri_verts=np.ascontiguousarray(f, np.uint32))
        mesh.merge()
        m = self.m3.Manifold(mesh)
        if m.status() != self.m3.Error.NoError or m.is_empty():
            raise ValueError(f"roof of component {self.u.cid} unit {self.u.k} is not a closed solid")
        return m.translate((0.0, 0.0, z))

    def _roof(self):
        """The attic over the walls, then the skin: the roof over the outline pushed out by the eaves and verges,
        lowered so its slopes run on from the attic's, riding EPS_M above it with the same cap, the thickness of its
        covering deep, and stopping at the buildings round it. The skin also covers the attic raised by half of EPS_M:
        where one roof is CGAL's and the other the in-repo roofer's (roofs.roof falls back on a roof CGAL folds over
        itself), or the two outlines' skeletons differ, their slopes need not run on from each other. The skin's outer
        solid, the outline pushed out on its eaves alone and the cut the buildings round it make are kept for the
        coarser levels' hull (_hull)."""
        st, o = self.st, self.u.outline
        dims = THATCH if st.roof == "thatch" else SLATE
        angles = roof_angles(self.u)
        self.angles = angles
        cap = roofs.cap_height(o, angles)
        attic = self._roof_mesh(o, angles, self.z_eave, cap)
        self.attic = attic
        self.solids.append((attic, _by_normal(st.wall, st.roof_material), "attic"))
        tan = math.tan(math.radians(st.pitch))
        drop = dims["thick"] / math.cos(math.radians(st.pitch))
        skin, self.soffit = None, self.z_eave + EPS_M
        self.outer = self.trim = self.near = None
        for extra in (0.0, SKIN_RETRY_M, None):       # CGAL's rare failures are numerical: nudge, then drop the overhang
            if extra is None:
                poly, lift = o, 0.0
            else:
                poly, applied = self._outset(o, angles, dims, extra)
                if not applied:
                    continue
                lift = (dims["eave"] + extra) * tan
            try:
                outer = self._roof_mesh(poly, angles, self.z_eave - lift + EPS_M, cap + lift)
            except ValueError:
                continue
            outer = outer + attic.translate((0.0, 0.0, EPS_M / 2))    # whichever roofer built each, it covers the attic
            skin = outer - outer.translate((0.0, 0.0, -drop))
            self.soffit = self.z_eave - lift + EPS_M          # the eaves' underside, flat out to the wall face
            self.outer = outer
            if extra is not None:                             # the outline pushed out on its eaves alone
                trim, applied = self._outset(o, angles, {**dims, "verge": -extra}, extra)
                self.trim = trim if applied else None
            break
        if skin is not None and not skin.is_empty():
            near = self.v.neighbours(self.u)
            if near:
                top = skin.bounding_box()
                self.near = self._prisms(shapely.union_all(near).buffer(-NEIGHBOUR_GAP_M), top[2] - 1.0, top[5] + 1.0)
                skin = skin - self.near
        self.skin = self.plain_skin = skin if skin is not None and not skin.is_empty() else None
        self.z_ridge = attic.bounding_box()[5]

    def _prisms(self, geom, z0, z1):
        """The prism of every polygon of `geom` from z0 to z1."""
        parts = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and not g.is_empty]
        return self.m3.Manifold.batch_boolean([self._prism(g, z0, z1) for g in parts], self.m3.OpType.Add) \
            if parts else self.m3.Manifold()

    def _outset(self, poly, angles, dims, extra=0.0):
        """(the outline pushed out by the eaves' overhang on sloping edges and the verges' on free gables, each `extra`
        more, each edge keeping its index; whether it could be): party gables stay put."""
        pts = np.asarray(poly.exterior.coords)[:-1]
        lines = []
        for k, (a, b) in enumerate(plan.edges(poly)):
            sloped = angles[k] < 89.999
            off = dims["eave"] + extra if sloped else 0.0 if k in self.u.party else dims["verge"] + extra
            n = plan.outward(a, b)
            lines.append((a + n * off, b - a))
        out = []
        for k in range(len(lines)):
            (p1, d1), (p2, d2) = lines[k - 1], lines[k]
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-9:
                out.append(p2)
            else:
                t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
                out.append(p1 + d1 * t)
        moved = Polygon(out)
        if poly.interiors or len(out) != len(pts) or not moved.is_valid or not moved.exterior.is_ccw \
                or not moved.buffer(1e-6).covers(poly):
            return poly, False
        return moved, True

    # openings: per bay and storey on the front; fewer behind
    def _facade(self, k):
        a, b = self.edges[k]
        t = (b - a) / np.linalg.norm(b - a)
        return a, t, plan.outward(a, b), float(np.linalg.norm(b - a))

    def _hole(self, k, s, z0, w, h, back, reveal, depth=REVEAL_M, panel=True):
        """Cut an opening of width w and height h into facade k, centred s along it from its start, from z0 up."""
        a, t, n, _ = self._facade(k)
        self._opening(a + t * s, t, n, z0, w, h, back, reveal, depth, panel)

    def _opening(self, c, t, n, z0, w, h, back, reveal, depth=REVEAL_M, panel=True):
        """Cut an opening of width w and height h, centred on plan point c of a face running along t and facing n
        (a counter-clockwise outline's edge), from z0 up: `depth` back from the face, its back face `back`. Unless
        `panel` is false (a margin's paint), a coarser level shows it as a flat panel on the face."""
        ang = math.degrees(math.atan2(t[1], t[0]))
        cutter = (self.m3.Manifold.cube((w, depth + 0.3, h), True)
                  .translate((0.0, (depth - 0.3) / 2, h / 2))
                  .rotate((0.0, 0.0, ang)).translate((c[0], c[1], z0)))
        # rotating by the face's angle turns the cube's +y to the face's left of t, which is into the solid: the
        # cube spans 0.3 m outside the face to `depth` inside it
        out3 = np.array([n[0], n[1], 0.0])
        self.cutters.append((cutter, _cut(back, reveal, out3)))
        if panel:
            self.panels.append((c, t, n, z0, [(-w / 2, 0.0), (w / 2, 0.0), (w / 2, h), (-w / 2, h)], back))

    def _glass(self):
        r = self.rng.random()
        return "glass-lit" if r < LIT else "glass-dim" if r < LIT + DIM else "glass-dark"

    def _head(self, k):
        """The highest an opening's head (and its margin) may reach on facade k: under the soffit of an eave there."""
        if self.angles[k] >= 89.999 or k in self.u.party or self.skin is None:
            return math.inf
        return self.soffit - SOFFIT_GAP_M - (MARGIN_M if self.st.margins else 0.0)

    def _openings(self):
        if bespoke.openings(self):
            return
        st, u = self.st, self.u
        k = u.front
        _, _, _, length = self._facade(k)
        reveal = st.wall
        centres, pitch_m = bays(length)
        door_bay = len(centres) // 2 if st.shopfront is None else (0 if self.rng.random() < 0.5 else len(centres) - 1)
        zf = self.z_floor
        win_w = min(1.0, pitch_m - 0.6)
        if win_w < 0.45:
            return
        win_h = 1.15 if st.family in ("cottage", "outshot") else 1.45
        door_w = min(1.0, pitch_m - 0.4)
        door_h = max(DOOR_MIN_M, min(2.05, self._head(k) - zf))
        order = sorted(range(len(centres)), key=lambda i: (abs(i - door_bay), i))
        door_bay = next((i for i in order if self._open(k, centres[i], door_w, DOOR_CLEAR_M)), None)
        for i, s in enumerate(centres):
            if i == door_bay:
                self._margin(k, s, zf, door_w, door_h)
                self._hole(k, s, zf, door_w, door_h, st.door, reveal, 0.12)
                if st.sign:
                    self._sign(k, s, door_w)
                self._steps(k, s, door_w)
            elif st.shopfront:
                h = min(2.0, self._head(k) - zf - 0.55)
                w = min(pitch_m - 0.5, 2.6)
                if h >= MIN_OPENING_M and self._open(k, s, w):
                    self._hole(k, s, zf + 0.55, w, h, "glass-shop", st.shopfront, 0.1)
                    bespoke.dress(self, k, s, zf + 0.55, w, h, 0.1, st.shopfront)
            else:
                self._window(k, s, zf + SILL_M, win_w, win_h)
        if st.shopfront:
            self._fascia(k, centres, pitch_m, zf)
        if st.storeys >= 2.0:
            for s in centres:
                self._window(k, s, zf + STOREY_M + SILL_M - 0.1, win_w, win_h)
        elif st.storeys == 1.5:
            how = attic_window(u, self.angles, self._open)
            if how is not None and how[0] == "dormers":
                for s in how[1]:
                    self._dormer(k, s, how[2])
            elif how is not None:
                _, s, z0, w, h = how
                self._window(k, s, self.z_eave + z0, w, h)
        back = self._back()
        if back is not None:
            _, _, _, blen = self._facade(back)
            h = min(1.1, self._head(back) - zf - SILL_M)
            if h >= MIN_OPENING_M:
                w = min(0.9, win_w)
                for s in ([blen / 2] if blen < 6.0 or not u.high else [blen / 3, 2 * blen / 3]):
                    if self._open(back, s, w):
                        self._hole(back, s, zf + SILL_M, w, h, self._glass(), reveal)

    def _open(self, k, s, w=0.0, clear=CLEAR_M):
        """Whether nothing of the village stands within `clear` in front of an opening w wide on facade k, s along it."""
        return self.v.opener(self.fp.frame, self.u.outline)(k, s, w, clear)

    def _dormer(self, k, s, w):
        """A wall-head dormer over bay s of facade k (see attic_window): its face flush with the wall and rising through
        the eaves, its window from just under the eaves, its roof hipped in thatch or gabled in slate and running back
        into the main roof, whose eave is cut away in front of it."""
        a, t, n, _ = self._facade(k)
        c = a + t * s
        wd = w + 2 * DORMER["cheek"]
        dd = dormer_depth(self.st.pitch, wd)
        sill, head = self.z_eave - DORMER["below"], self.z_eave - DORMER["below"] + DORMER["window"]
        top = head + DORMER["lintel"]
        p0, p1 = c - t * wd / 2, c + t * wd / 2
        rect = Polygon([p0, p1, p1 - n * dd, p0 - n * dd])
        self.solids.append((self._prism(rect, sill - 0.2, top), _material(self.st.wall), "feature"))
        front = self.st.pitch if self.st.roof == "thatch" else 90.0
        hood = self._roof_mesh(rect, [front, self.st.pitch, 90.0, self.st.pitch], top)
        self.solids.append((hood, _by_normal(self.st.wall, self.st.roof_material), "feature"))
        if self.skin is not None:
            notch = Polygon([p0 - t * 0.05, p1 + t * 0.05, p1 + t * 0.05 + n * 1.0, p0 - t * 0.05 + n * 1.0])
            notch = self._prism(orient(notch, 1.0), self.z_eave - 1.0, top + 2.0)
            self.notches.append(notch)
            self.skin = self.skin - notch
        self._hole(k, s, sill, w, head - sill, self._glass(), self.st.wall)

    def _back(self):
        a, b = self.edges[self.u.front]
        n = plan.outward(a, b)
        cands = [(float(plan.outward(*e) @ n), k) for k, e in enumerate(self.edges)
                 if k not in self.u.party and k != self.u.front and np.linalg.norm(e[1] - e[0]) > 2.0]
        cands = [c for c in cands if c[0] < -0.9]
        return min(cands)[1] if cands else None

    def _window(self, k, s, z0, w, h):
        h = min(h, self._head(k) - z0)
        if h < MIN_OPENING_M or not self._open(k, s, w):
            return
        self._margin(k, s, z0, w, h)
        self._hole(k, s, z0, w, h, self._glass(), self.st.wall)
        if k == self.u.front:
            bespoke.dress(self, k, s, z0, w, h, REVEAL_M, "pale-stone")
        if self.u.high or self.st.family in ("shop", "villa"):
            self._sill(k, s, z0, w)

    def _margin(self, k, s, z0, w, h):
        if self.st.margins:
            self._hole(k, s, z0 - MARGIN_M, w + 2 * MARGIN_M, h + 2 * MARGIN_M, self.st.margins, self.st.margins,
                       MARGIN_DEPTH_M, panel=False)

    def _local_box(self, k, s, out0, out1, z0, z1, w, mat, back=False):
        """A box on facade k: centred s along it, w wide, from out0 to out1 metres out of the wall, z0 to z1 high;
        without its face against the wall unless `back`."""
        from mesh_geometry import panel
        a, t, n, _ = self._facade(k)
        c = a + t * s
        p = lambda x, z, o: np.array([c[0] + t[0] * x + n[0] * o, c[1] + t[1] * x + n[1] * o, z])
        x0, x1 = -w / 2, w / 2
        faces = [[p(x0, z1, out0), p(x0, z1, out1), p(x1, z1, out1), p(x1, z1, out0)],        # top
                 [p(x0, z0, out1), p(x1, z0, out1), p(x1, z1, out1), p(x0, z1, out1)],        # front
                 [p(x0, z0, out0), p(x0, z0, out1), p(x0, z1, out1), p(x0, z1, out0)],        # left end
                 [p(x1, z0, out1), p(x1, z0, out0), p(x1, z1, out0), p(x1, z1, out1)],        # right end
                 [p(x0, z0, out0), p(x1, z0, out0), p(x1, z0, out1), p(x0, z0, out1)]]        # bottom
        if back:
            faces.append([p(x1, z0, out0), p(x0, z0, out0), p(x0, z1, out0), p(x1, z1, out0)])
        meshes = [panel(f, code(mat)) for f in faces]
        centre = p(0.0, (z0 + z1) / 2, (out0 + out1) / 2)
        return _outward(concat(meshes), centre)

    def _sill(self, k, s, z0, w):
        self.fittings.append(self._local_box(k, s, 0.0, 0.07, z0 - 0.07, z0, w + 0.12, "pale-stone"))

    def _steps(self, k, s, w):
        """A raised door's steps as one stone block, from below the ground to its floor and as deep as the flight
        would run: its 17 cm risers are lost from the flyover's height."""
        a, t, n, _ = self._facade(k)
        g = float(self.v.scene.ground(*self.fp.frame.world(a + t * s + n * 0.6))) - self.y_ref
        rise = self.z_floor - g
        if rise > 0.2:
            depth = 0.32 * math.ceil((rise - 0.2) / 0.17)
            self.fittings.append(self._local_box(k, s, 0.0, depth, g - 0.3, self.z_floor, w + 0.3, "pale-stone"))

    def _fascia(self, k, centres, pitch_m, zf):
        w = pitch_m * len(centres)
        s = centres[len(centres) // 2] if len(centres) % 2 else (centres[0] + centres[-1]) / 2
        box_mesh = self._local_box(k, s, 0.0, 0.12, zf + 2.7, zf + 3.15, w - 0.2, self.st.shopfront)
        self.fittings.append(box_mesh)

    def _sign(self, k, s, door_w):
        z = self.z_floor + 3.5
        side = s + door_w / 2 + 0.25
        self.fittings.append(self._local_box(k, side, 0.0, 0.95, z, z + 0.05, 0.05, "black-iron"))
        self.fittings.append(self._local_box(k, side, 0.3, 0.95, z - 0.62, z - 0.02, 0.05, "sign-board", back=True))

    def _chimneys(self):
        """Stacks on the gables: on each free gable end at least 2.5 m long but the front, flush with the gable, and
        once on each party wall shared with another building of this footprint part, straddling it, built from the
        side whose outward normal points the frame's positive way. A building with no gable stack of its own or its
        neighbour's gets one straddling its ridge, where the ridge lies deepest inside its plan. A stack's top rises
        STACK['over'] above the higher of the ridges beside it. A named building may keep its stacks to some of its
        gables (mesh_village_named)."""
        if self.st.family == "outshot" or not bespoke.chimneys(self):
            return
        allowed = bespoke.stack_gables(self)
        spots, served = [], False
        for k, (a, b) in enumerate(self.edges):
            if self.angles[k] < 89.999 or (allowed is not None and k not in allowed):
                continue
            length = float(np.linalg.norm(b - a))
            n = plan.outward(a, b)
            if k in self.u.party:
                c = self._shared(a, b)
                if c is None:
                    continue
                if not (n[0] > 0.5 or (abs(n[0]) <= 0.5 and n[1] > 0.0)):
                    served = True                             # the neighbour builds this one
                    continue
            elif length < 2.5 or k == self.u.front:
                continue
            else:
                c = (a + b) / 2 - n * STACK["across"] / 2
            spots.append((c, (b - a) / length, None if k in self.u.party else n))
        if not spots and not served:
            spot = self._ridge_spot()
            if spot is not None:
                spots.append((*spot, None))
        for c, t, flush in spots:
            self._stack(c, t, flush)

    def _ridge_spot(self):
        """(plan point, direction) for a stack straddling the ridge: of the attic's highest points and the middles of
        pairs of them, along the line between them, those far enough inside the plan for the stack to stand on it with
        STACK['spare'] round it, the deepest, nearest the plan's middle on a tie; else the plan's most inland point.
        None for a building too small to carry a stack."""
        ring = self.u.outline
        need = math.hypot(STACK["along"] / 2 + STACK["spare"], STACK["across"] / 2 + STACK["spare"])
        inland = shapely.maximum_inscribed_circle(ring, 0.01)
        if inland.length < need:
            return None
        v, _ = roofs.roof(ring, self.angles)
        top = v[v[:, 2] >= v[:, 2].max() - 1e-6][:, :2]
        top = top[np.lexsort((top[:, 1], top[:, 0]))]
        cands = [(p, np.array([1.0, 0.0])) for p in top]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                d = top[j] - top[i]
                if np.linalg.norm(d) > 1e-6:
                    cands.append(((top[i] + top[j]) / 2, d / np.linalg.norm(d)))
        middle = np.asarray(ring.centroid.coords[0])
        scored = [(round(ring.exterior.distance(Point(p)), 2), -round(float(np.linalg.norm(p - middle)), 3), i)
                  for i, (p, _) in enumerate(cands) if ring.covers(Point(p))]
        scored = [s for s in scored if s[0] >= need]
        if scored:
            return cands[max(scored)[2]]
        return np.asarray(inland.coords[0]), np.array([1.0, 0.0])

    def _stack(self, c, t, flush=None):
        """A stack centred on plan point c, its long side along t, capped flush in pale stone (a cope's few
        centimetres of overhang are lost from the flyover's height), with a pot, or on the High Street a pot or two.
        It starts 0.3 m above its eaves, or lower, 0.1 m under the lowest of the roofs below its foot (at its corners,
        its ends' middles and 2 cm either side of its middle line: a neighbour may slope down to a party wall), so that
        it rises out of every roof under it. A stack flush with a free gable (`flush`, the gable's outward normal)
        stands DEPTH_GAP_M inside it where it stands apart from the shell as a box (boxes), out of the gable's plane."""
        near = self.v.ridges_near(self.fp.frame.world(c), STACK["clear"])
        top = max([self.z_ridge] + [r - self.y_ref for r in near]) + STACK["over"]
        ang = math.degrees(math.atan2(t[1], t[0]))
        n = np.array([-t[1], t[0]])
        foot = np.array([c + t * a + n * b for a in (-STACK["along"] / 2, 0.0, STACK["along"] / 2)
                         for b in (-STACK["across"] / 2, -0.02, 0.02, STACK["across"] / 2)])
        roof = np.full(len(foot), -np.inf)
        for w in self.v.units[self.u.cid]:
            inside = shapely.contains_xy(w.outline.buffer(0.01), foot[:, 0], foot[:, 1]) if w.part == self.u.part \
                else np.zeros(len(foot), bool)
            if inside.any():
                z = self.v.attic(w)(foot) + self.v.eave(w) - self.y_ref
                roof = np.where(inside, np.maximum(roof, z), roof)
        under = roof[np.isfinite(roof)]
        z0 = min(self.z_eave + 0.3, float(under.min()) - 0.1) if len(under) else self.z_eave + 0.3
        stack = (self.m3.Manifold.cube((STACK["along"], STACK["across"], top - z0), True)
                 .translate((0.0, 0.0, (top - z0) / 2)).rotate((0.0, 0.0, ang)).translate((c[0], c[1], z0)))
        self.solids.append((stack, _capped(bespoke.stack_wall(self) or self.st.wall, "pale-stone"), "stack"))
        if flush is not None:
            self.boxes[id(stack)] = stack.translate((-flush[0] * DEPTH_GAP_M, -flush[1] * DEPTH_GAP_M, 0.0))
        second = self.rng.random() < 0.6
        pots = 2 if second and self.u.high else 1
        for j in range(pots):
            p = c + t * ((j - (pots - 1) / 2) * 0.4)
            self.fittings.append(_pot(p, t, top))

    def _shared(self, a, b):
        """The middle of the longest stretch of edge a-b that another building of this footprint part stands along, if
        a stack fits within it with STACK['spare'] and a hand's breadth to spare; else None."""
        others = [w.outline for w in self.v.units[self.u.cid] if w.part == self.u.part and w is not self.u]
        if not others:
            return None
        shared = LineString([a, b]).intersection(shapely.union_all(others).buffer(0.05))
        pieces = [g for g in getattr(shared, "geoms", [shared]) if g.geom_type == "LineString" and not g.is_empty]
        best = max(pieces, key=lambda g: g.length, default=None)
        if best is None or best.length < STACK["along"] + 2 * STACK["spare"] + 0.1:
            return None
        return (np.asarray(best.coords[0]) + np.asarray(best.coords[-1])) / 2
