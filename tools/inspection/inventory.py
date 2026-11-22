"""The qualification's inventory: every distinctly rendered or defined object of the finished page, each with the point
its close-up looks at, its owner, and the manifest decisions (with their quotations) its reviewer checks it against.

    python -X utf8 -m tools.inspection.inventory [--out build/inspection/inventory.json]

The objects, in the order the inspection shoots them:

- every manifest asset (src/mesh_scene.json), and the parts its labels and traced features name: the castle's towers,
  wings, doors and pend (its plan's labels, placed by the plan's similarity fit, mesh_castle.plan_fit), the station's
  building and platform, the train's engine and vans, the pitch's stands, hoops, podium and changing rooms, and so on;
- one representative per kind of repeated unnamed thing, of median size, and its outliers (the largest and smallest,
  and those placed or shaped unlike the rest): Hogsmeade's buildings by style family (mesh_village's plan), lamps,
  headstones, garden walls, coaches, hedges, bridges, boundary walls, greenhouses, court houses, generic towers and the
  Forest's homesteads;
- what the payload does not draw but the manifest or the build defines: the clearings (meta.clearings and
  clearings.LINEAR), the ground classes (frag4.glsl classCol), the terrain the decisions shape (the castle's mound and
  cliff, the cabin's terrace, the Willow's knoll and the like), the hills, and the footprints the manifest leaves
  unbuilt (absences: nothing may stand there);
- a tree of each of the payload kit's species and forms (a mature and a young Scots pine, a yew, an oak, a beech, a
  birch), each shot at the one nearest its seed that stands apart (tools.inspection.shoot);
- the renderer's set pieces and surfaces: lake water, its reflections by day and by night, streams, greenhouse glass,
  mesh shadows, trees, the sky and the night, levels of detail and thin parts at distance;
- a still of each of the tour's waypoints (template_c.html WP), naming the objects its caption names.

Owners are the sessions that fix what an object shows: the castle, greenhouses and transport went through phase 0 and
the first-pass reviews, so the trunk keeper takes them, as it takes the tour; grounds are lane A's, Hogsmeade lane B's,
the Willow lane C's, the mesh trees' species lane T's, the site (terrain, trees, clearings, ground) lane D's, and the
renderer integration's.
`uncovered` lists what no object covers; the CLI fails while anything is.
"""
import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from tools.inspection.frame import View
from tools.paths import BUILD, PAGE, SRC

MANIFEST = SRC / "mesh_scene.json"
PAYLOAD = BUILD / "meshes.js"
OUT_DIR = BUILD / "inspection"

OWNERS = {"castle": "trunk keeper", "greenhouses": "trunk keeper", "transport": "trunk keeper", "grounds": "lane A",
          "village": "lane B", "willow": "lane C", "trees": "lane T", "site": "lane D", "renderer": "integration",
          "tour": "trunk keeper"}
SESSIONS = frozenset(OWNERS.values())
KINDS = ("asset", "part", "sample", "outlier", "species", "absence", "surface", "clearing", "terrain", "tour")
UNNAMED = ("village-house", "village-wing")
GROUND_CLASSES = ("rough meadow", "lawn", "rough scrubland", "earth track", "vegetable beds", "forest floor",
                  "Quidditch pitch", "shingle", "overgrown garden")         # frag4.glsl classCol, by class index
FOREST_FLOOR, SHINGLE = 5, 7        # the floor under the Forest's trees, seen only at its edge; the lake's beaches
CLASS_DECISIONS = {"rough scrubland": ("scrubland-as-scrub",), "vegetable beds": ("vegetable-patch-beds",)}
DASHED_HILLS = ("hillock 4h", "howe 4i")    # where White Hound's dashed outlines, read as tree ink, stood rings of trees
FLOOR_BAND_M = (4.0, 12.0)          # the forest floor is shot this far in from the wood's edge, from the open side
SHINGLE_FROM_CASTLE_M = 300.0       # the castle's base and cliff cover the beaches at its foot
STREET_FRONTS = ("zonkos",)         # named shops whose sun side shows their back: framed from the High Street, over
STREET_FRONT_ELEVATION = 30.0       # the roofs in front
FOREST_PLACES = ("Aragog's hollow", "Grawp's clearing")                    # src/forest_places.py DRAWN
POINTS = 400                        # each payload group keeps about this many of its vertices (east, north) for picks
DAY = 15.0                          # mid-afternoon: the sun in the south-west, high enough for every view
DUSK = 21.5
DARK_HOURS = ("civil dusk", "night")    # the review sheets' named hours after dark
# decisions only a shot after dark can show: an object shot by day does not cover them
AFTER_DARK = ("lamplit-windows", "village-night-windows")
# decisions no picture can show
NOT_VISUAL = {
    "wh-authority": "the order of authority among sources",
    "live-origin-provenance": "where the cited pages were fetched",
    "performance-budget": "frame time, measured by the timing run, not seen",
    "willow-generator": "that the Willow is rebuilt byte for byte from a seed",
    "trees-generator": "that the trees' variants are built from a seed, and that the generator places none",
    "elevation-model": "which elevation model the terrain comes from; its result is the terrain the site objects show",
    "castle-plan-interiors-excluded": "rooms, corridors and statues inside the castle: nothing to see from outside",
    "static-exteriors": "that nothing animates: a still cannot show it (the motion evidence does)",
    "tour-length": "the tour's duration",
}
# castle-plan labels the manifest does not trace, at the pointer dots of castle_plan.gif (744 x 533 px)
CASTLE_PLAN_PX = {"Ch": (195, 128), "NB": (178, 202), "EW": (440, 176), "WW": (284, 415), "HW": (295, 356),
                  "FO": (365, 376)}
# where the search for a tree of a species starts, by class (conifer or not): the pines between the station and the
# castle's shore (wh-forest's "swathe of pine trees between the station and the lake"), and the Forest's edge by the hut
TREE_SEEDS = {True: (145.0, 80.0), False: (-330.9, -193.1)}
TREE_TITLES = {"scots-pine-mature": "A mature Scots pine", "scots-pine-young": "A young Scots pine", "yew": "A yew",
               "oak": "An oak", "beech": "A beech", "birch": "A birch"}


@dataclass(frozen=True)
class Obj:
    id: str
    title: str
    family: str                     # a manifest family, or "site", "renderer", "tour"
    kind: str                       # KINDS
    at: tuple                       # (east, height above its ground, north): what the close-up looks at
    size: float                     # metres across: what the close-up frames
    assets: tuple = ()              # manifest assets it shows, whose labels and decisions it answers for
    decisions: tuple = ()
    labels: tuple = ()
    views: tuple = ()               # () framed automatically, else (close-up[, context]) frame.View
    hour: object = DAY
    sample: str = ""                # a sample or outlier: what population it stands for, and why this one
    subjects: tuple = ()            # a tour still: the objects its caption names
    box: tuple = ()                 # (x0, y0, z0, x1, y1, z1), world, for the occlusion check; () a cube of `size`
    azimuth: object = None          # the side to shoot from (compass bearing to the camera), None the sun's
    elevation: float = 12.0
    note: str = ""
    owner: str = ""

    def __post_init__(self):
        if not self.owner:
            object.__setattr__(self, "owner", OWNERS[self.family])

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        out = {}
        for f in fields(cls):
            v = d.get(f.name, f.default)
            if f.name == "views":
                v = tuple(View(x[0], tuple(x[1]), tuple(x[2]), x[3], x[4]) for x in v)
            elif isinstance(v, list):
                v = tuple(v)
            out[f.name] = v
        return cls(**out)


def load(path):
    return [Obj.from_dict(r) for r in json.loads(Path(path).read_text(encoding="utf-8"))]


# ---------------------------------------------------------------- the world the inventory reads
class World:
    """What the inventory is built from: the manifest, the registered world's meta, the tour, each asset's payload
    boxes, Hogsmeade's planned buildings and small structures, the footprint groups and castle pieces with their
    owners, a point of each ground class, the ground's height, and the tree kit's variants (name, conifer, share, crown
    radius and height)."""

    def __init__(self, manifest, meta, waypoints, boxes, groups, units=(), small=None, components=None, pieces=(),
                 classes=None, corridor=None, ground=None, sides=None, kit=()):
        self.manifest, self.meta, self.waypoints = manifest, meta, waypoints
        self.boxes, self.groups = boxes, groups
        self.units, self.small = list(units), small or {}
        self.components, self.pieces = components or {}, list(pieces)
        self.classes, self.corridor, self.sides = classes or {}, corridor, sides or {}
        self._ground = ground
        self.kit = list(kit)

    def ground(self, x, z):
        return float(self._ground(x, z)) if self._ground else 0.0

    @classmethod
    def load(cls):
        import build_meshes
        import mesh_village
        from mesh_source import decode, load_data, sample, Grid
        from tools.review import tour

        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        data = load_data()
        meta = data["meta"]
        payload = build_meshes.decode_payload(PAYLOAD.read_text(encoding="utf-8"))
        found = {}
        for c in payload["chunks"]:
            if c["level"] == 0 and len(c["pos"]):
                found.setdefault(c["asset"], {}).setdefault(c["group"], []).append(c["pos"])
        boxes, groups = {}, {}
        for asset, by_group in found.items():
            for g in sorted(by_group):
                pos = np.concatenate(by_group[g])
                box = tuple(round(float(v), 2) for v in np.concatenate([pos.min(axis=0), pos.max(axis=0)]))
                mid = (np.array(box[:3]) + np.array(box[3:])) / 2
                near = pos[int(np.argmin(np.linalg.norm(pos - mid, axis=1)))]
                thinned = np.round(pos[::max(1, len(pos) // POINTS)][:, [0, 2]], 2)
                boxes.setdefault(asset, []).append(box)
                groups.setdefault(asset, []).append((box, (round(float(near[0]), 2), round(float(near[2]), 2)),
                                                     thinned))
        scene = build_meshes.Scene(data, manifest)
        village = mesh_village.Village(scene)
        local, bed = Grid(meta["local"]), decode(data["tex"]["locBed"])
        ground = lambda x, z: sample(bed, local, x, z)           # noqa: E731
        components = {}
        for c in scene.labelled[1]:
            components.setdefault(scene.owners.get(c["id"]), []).append(
                {"id": c["id"], "x": c["x"], "z": c["z"], "area": c["area"]})
        pieces = [dict(p, owner=scene.piece_owners[p["id"]]) for p in scene.castle_labelled[1]]
        classes, sides = ground_classes(data)
        kit = [{k: v[k] for k in ("name", "conifer", "share", "crown_r", "height")}
               for v in (payload.get("kit") or {}).get("variants", ())]
        return cls(manifest, meta, tour.waypoints(PAGE.read_text(encoding="utf-8", errors="replace")), boxes, groups,
                   village_units(village, scene), small_structures(village), components, pieces,
                   classes, corridor_point(data, meta), ground, sides, kit)


def village_units(village, scene):
    """Hogsmeade's planned buildings as plain records: footprint component, asset, style family, area and centre
    (world), whether it fronts the High Street, its footprint fits a square grid, its roof is hipped, how far the
    ground falls across it, and whether the village moves it."""
    import mesh_village
    out = []
    for cid, units in sorted(village.units.items()):
        for u in units:
            w = u.fp.frame.to_world(u.outline)
            ring = np.asarray(w.exterior.coords)
            heights = scene.ground(ring[:, 0], ring[:, 1])
            out.append({"cid": cid, "part": u.part, "k": u.k, "asset": u.asset, "family": u.style.family,
                        "area": round(float(w.area), 1), "x": round(float(w.centroid.x), 2),
                        "z": round(float(w.centroid.y), 2), "high": bool(u.high), "regular": bool(u.fp.regular),
                        "hips": bool(u.style.hips), "drop": round(float(np.max(heights) - np.min(heights)), 2),
                        "roof": u.style.roof_material, "moved": cid in mesh_village.MOVED})
    return out


def small_structures(village):
    """{kind: [(x, z, size)]} of Hogsmeade's structures on no footprint: lamps, headstones, the kirkyard's wall pieces
    and garden walls, each placed at its centre and sized by its extent."""
    import mesh_village_small as small

    def where(meshes):
        out = []
        for m in meshes:
            lo, hi = m.pos.min(axis=0), m.pos.max(axis=0)
            out.append((round(float((lo[0] + hi[0]) / 2), 2), round(float((lo[2] + hi[2]) / 2), 2),
                        round(float(np.hypot(hi[0] - lo[0], hi[2] - lo[2])), 2)))
        return out
    return {"lamps": where(small.lamps(village)), "headstones": where(small.headstones(village)),
            "kirkyard-wall": where(small.kirkyard_wall(village)), "garden-walls": where(small.garden_walls(village))}


def ground_classes(data):
    """({class: (x, z)}, {class: azimuth}): for each surface class of the local layer, the point deepest inside its
    largest patch of dry, treeless ground with no building on it, and the side to frame it from where the sun's side
    would hide it. The forest floor shows only under its trees' crowns, so it is taken at the edge of the Forest (its
    largest wood) nearest the castle, FLOOR_BAND_M in, and framed from the open ground; the shingle at the widest point
    of its longest beach, framed from over the lake."""
    from scipy import ndimage as ndi
    from mesh_source import CHANNELS, Grid, decode, sd_bytes
    meta, tex = data["meta"], data["tex"]
    grid = Grid(meta["local"])
    surface = decode(tex["locN"])[..., CHANNELS["locN"].index("surface")]
    a = decode(tex["locA"])
    trees, bsd = a[..., CHANNELS["locA"].index("trees")], sd_bytes(a[..., CHANNELS["locA"].index("bsd")])
    dry = decode(tex["locW"]) <= 0.0
    open_ground = dry & (trees < 26) & (bsd > 4.0)
    rows, cols = np.indices(surface.shape)
    x, z = grid.to_world(rows, cols)
    far_from_castle = np.hypot(x, z) > SHINGLE_FROM_CASTLE_M
    out, sides = {}, {}
    for c in range(len(GROUND_CLASSES)):
        if c == FOREST_FLOOR:
            wood = (surface == c) & (trees >= 26)
            if not wood.any():
                continue
            lab, n = ndi.label(wood)
            wood = lab == 1 + int(np.argmax(ndi.sum(wood, lab, range(1, n + 1))))   # the Forest, not a copse
            i, j = edge_texel(wood, open_ground, grid.texel_size()[0], FLOOR_BAND_M, grid.to_world)
            sides[c] = nearest_bearing(open_ground, i, j, grid.to_world)
        else:                                   # a beach runs into the water; the castle's base covers those at its foot
            mask = ((surface == c) & far_from_castle) if c == SHINGLE else (surface == c) & open_ground
            if not mask.any():
                continue
            lab, n = ndi.label(mask)
            big = 1 + int(np.argmax(ndi.sum(mask, lab, range(1, n + 1))))
            depth = ndi.distance_transform_edt(lab == big)
            i, j = np.unravel_index(int(np.argmax(depth)), depth.shape)
            if c == SHINGLE:
                water, m = ndi.label(~dry)
                lake = water == 1 + int(np.argmax(ndi.sum(~dry, water, range(1, m + 1))))
                sides[c] = nearest_bearing(lake, i, j, grid.to_world)
        x, z = grid.to_world(i, j)
        out[c] = (round(float(x), 1), round(float(z), 1))
    return out, sides


def nearest_bearing(mask, i, j, to_world):
    """The compass bearing (0 north, 90 east) from texel (i, j) to the nearest texel of `mask`; to_world(i, j) gives a
    texel's (east, north)."""
    from scipy import ndimage as ndi
    _, (ii, jj) = ndi.distance_transform_edt(~mask, return_indices=True)
    (x0, z0), (x1, z1) = to_world(i, j), to_world(ii[i, j], jj[i, j])
    return round(math.degrees(math.atan2(float(x1 - x0), float(z1 - z0))) % 360.0, 1)


def edge_texel(inside, outside, texel_m, band, to_world):
    """The texel of `inside` between band[0] and band[1] metres from the nearest texel of `outside`, nearest the world's
    origin (the castle)."""
    from scipy import ndimage as ndi
    depth = ndi.distance_transform_edt(~outside) * texel_m
    ii, jj = np.nonzero(inside & (depth >= band[0]) & (depth <= band[1]))
    x, z = to_world(ii, jj)
    k = int(np.argmin(np.hypot(np.asarray(x, float), np.asarray(z, float))))
    return int(ii[k]), int(jj[k])


def corridor_point(data, meta):
    """The point of the Forest's corridors (forest-corridors-a890-railway) nearest Grawp's clearing: a texel the Forest
    layer clears with dense Forest round it."""
    from scipy import ndimage as ndi
    from mesh_source import Grid, decode
    density = decode(data["tex"]["forest"])[..., 0].astype(float)
    grid = Grid(meta["forest"])
    ii, jj = np.indices(density.shape)
    E, N = grid.to_world(ii, jj)
    inner = (density < 8) & (ndi.uniform_filter((density > 100).astype(float), size=13) > 0.6)
    at = place(meta, "Grawp's clearing")
    r = np.where(inner, np.hypot(E - at[0], N - at[1]), np.inf)
    k = int(np.argmin(r))
    return (round(float(E.flat[k]), 1), round(float(N.flat[k]), 1)) if np.isfinite(r.flat[k]) else None


def place(meta, name):
    """A place by name: a clearing's centre from meta.clearings when the build records one under its name (Aragog's
    hollow as "aragogs-hollow"), else its point of interest."""
    key = slug(name.replace("'", ""))
    for c in meta.get("clearings", []):
        if c["id"] == key:
            o = np.asarray(c["outline"], float)
            return float(o[:, 0].mean()), float(o[:, 1].mean())
    return tuple(next(p["p"] for p in meta["extras"]["poi"] if p["name"] == name))


# ---------------------------------------------------------------- sampling repeated things
def spread_samples(items, what, ends=("largest", "smallest"), middle="of median size"):
    """[(item id, "sample" or "outlier", reason)] from [(id, size)]: the representative, the item of median size
    (median_low, the first such), and with three or more the largest and the smallest as outliers (`ends` and `middle`
    word the reasons)."""
    if not items:
        return []
    sizes = [s for _, s in items]
    mid = statistics.median_low(sizes)
    rep = min(range(len(items)), key=lambda k: (abs(sizes[k] - mid), k))
    out = [(items[rep][0], "sample", f"the representative of the {len(items)} {what}, {middle}")]
    if len(items) >= 3:
        for k, why in ((max(range(len(items)), key=lambda k: (sizes[k], -k)), ends[0]),
                       (min(range(len(items)), key=lambda k: (sizes[k], k)), ends[1])):
            if k != rep and items[k][0] not in [o[0] for o in out]:
                out.append((items[k][0], "outlier", f"the {why} of the {len(items)} {what}"))
    return out


def village_samples(units):
    """[(unit, "sample" or "outlier", reason)] of Hogsmeade's unnamed buildings (UNNAMED): one representative per style
    family, the building whose area is nearest its family's median (median_low, ties to the lower component), then the
    outliers not already chosen: every building the village moves, the largest and the smallest, the one on the
    steepest ground, the largest whose footprint fits no square grid, and a hipped roof when no representative has
    one."""
    pool = [u for u in units if u["asset"] in UNNAMED]
    key = lambda u: (u["cid"], u["part"], u["k"])        # noqa: E731
    by_family = {}
    for u in pool:
        by_family.setdefault(u["family"], []).append(u)
    chosen, out = set(), []
    for family, us in by_family.items():
        mid = statistics.median_low([u["area"] for u in us])
        rep = min(us, key=lambda u: (abs(u["area"] - mid), key(u)))
        chosen.add(key(rep))
        out.append((rep, "sample", f"the representative of the {len(us)} {family} buildings, of median area "
                                   f"({rep['area']:.0f} m2)"))
    n = len(pool)
    outliers = [(u, f"moved across its lane from where the map draws it (one of {n} buildings)")
                for u in sorted(pool, key=key) if u.get("moved")]
    if pool:
        outliers += [(max(pool, key=lambda u: (u["area"], key(u))), f"the largest of the {n} buildings"),
                     (min(pool, key=lambda u: (u["area"], key(u))), f"the smallest of the {n} buildings"),
                     (max(pool, key=lambda u: (u["drop"], key(u))), f"on the steepest ground of the {n} buildings")]
    irregular = [u for u in pool if not u["regular"]]
    if irregular:
        outliers.append((max(irregular, key=lambda u: (u["area"], key(u))),
                         f"the largest of the {len(irregular)} buildings whose footprint keeps its trace"))
    hipped = [u for u in pool if u["hips"]]
    if hipped and not any(u["hips"] for u, kind, _ in out):
        mid = statistics.median_low([u["area"] for u in hipped])
        outliers.append((min(hipped, key=lambda u: (abs(u["area"] - mid), key(u))),
                         f"a hipped roof, of median area of the {len(hipped)} hipped buildings"))
    for u, why in outliers:
        if key(u) not in chosen:
            chosen.add(key(u))
            out.append((u, "outlier", why))
    return out


# ---------------------------------------------------------------- helpers
def sheet_view(family, label, hour=None):
    """A review sheet's view by its label (tools/review/<family>.py), as a frame.View."""
    import importlib
    views = importlib.import_module(f"tools.review.{family}").VIEWS
    v = next(v for v in views if v[0] == label)
    view = View.from_sheet(v)
    return view._replace(hour=hour) if hour is not None else view


def union(boxes):
    b = np.array(boxes, float)
    return tuple(float(v) for v in np.concatenate([b[:, :3].min(0), b[:, 3:].max(0)]))


def spot(world, box, anchor=None, cap=None):
    """(at, size) of a subject in a box: at its anchor (east, north; the point of its geometry nearest the box's
    centre, else the centre), its target the box's middle height above the ground there, and its size the box's
    horizontal diagonal, at most `cap`."""
    x, z = anchor if anchor else ((box[0] + box[3]) / 2, (box[2] + box[5]) / 2)
    h = max(0.3, (box[1] + box[4]) / 2 - world.ground(x, z))
    size = max(2.0, math.hypot(box[3] - box[0], box[5] - box[2]))
    return (round(float(x), 2), round(h, 2), round(float(z), 2)), round(min(size, cap) if cap else size, 2)


def asset_spot(world, asset, cap=None):
    """spot() of a whole asset: its union box, anchored on its group nearest that box's centre."""
    box = union(world.boxes[asset])
    mid = ((box[0] + box[3]) / 2, (box[2] + box[5]) / 2)
    anchor = min((a for _, a, _ in world.groups[asset]), key=lambda a: math.dist(a, mid))
    return spot(world, box, anchor, cap), box


def nearest_group(world, asset, xz):
    """(box, anchor) of the asset's group with a vertex nearest the point xz (east, north), anchored on that vertex."""
    best = min(((box, pts[int(np.argmin(np.hypot(*(pts - np.asarray(xz)).T)))]) for box, _, pts in world.groups[asset]),
               key=lambda bp: math.dist(bp[1], xz))
    return best[0], (float(best[1][0]), float(best[1][1]))


def decisions_of(world, asset):
    return tuple(next(a for a in world.manifest["assets"] if a["id"] == asset).get("decisions") or ())


def labels_of(world, asset):
    return tuple(next(a for a in world.manifest["assets"] if a["id"] == asset).get("labels") or ())


def slug(text):
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-").replace("--", "-")


def at_point(world, x, z, h=0.0):
    return (round(float(x), 2), float(h), round(float(z), 2))


# ---------------------------------------------------------------- the families' objects
# the castle sheet's "Great Hall from the south-east" frames the ranges in front of the hall, and a taller range hides
# its south wall from every southern view: from across the court to its south-south-west, its long roof and its west
# gable's tall windows fill 47% of the frame's middle third (its G-buffer class, measured at ab7ab06)
GREAT_HALL = View("across the court from the south-south-west, 15:00", (-25.1, 50.0, -151.5), (-4.6, 26.0, -95.1),
                  DAY, "target")


def castle_objects(world):
    import mesh_castle
    meta = world.meta
    castle = next(a for a in world.manifest["assets"] if a["id"] == "castle")
    traced = {t["id"]: t for t in castle["traced"]}
    plan, _ = mesh_castle.plan_fit(traced["plan-fit"]["plan_px"], meta["named"])
    where = {k: tuple(v) for k, v in meta["named"].items()}
    where.update({k: tuple(float(c) for c in plan(*px)) for k, px in CASTLE_PLAN_PX.items()})
    where["HO"] = tuple(float(c) for c in plan(*traced["heads-tower"]["plan_px"]))
    where["BS"] = tuple(float(c) for c in plan(*traced["broomshed-pend"]["plan_px"]["court"]))
    (whole_at, _), whole = asset_spot(world, "castle")
    obj = lambda slug_, title, **kw: Obj(id=f"castle/{slug_}", title=title, family="castle",          # noqa: E731
                                        assets=("castle",), **kw)
    out = [obj("whole", "Hogwarts castle, whole", kind="asset", at=whole_at, size=250.0,
               decisions=("wh-skyline", "warm-masonry", "castle-colour-red-cast", "castle-speck-threshold",
                          "small-scottish-turrets"),
               views=(sheet_view("castle", "Whole castle from 700 m, 18:30"),), hour=18.5, box=whole)]
    parts = (  # slug, title, labels, where, height of its middle above the ground, size, decisions, close-up
        ("astronomy-tower", "Astronomy Tower", ("AT", "EH"), "AT", 30.0, 70.0,
         ("wh-skyline", "astronomy-beside-doors", "castle-norman-openings"),
         ("castle", "West front and Astronomy Tower, 18:30")),
        ("gryffindor-tower", "Gryffindor Tower", ("GT",), "GT", 25.0, 60.0, ("wh-skyline", "castle-norman-openings"),
         None),
        ("north-tower", "North Tower, with the chapel at its foot", ("NT", "Ch"), "NT", 30.0, 60.0,
         ("wh-skyline", "north-tower-stands-clear"), None),
        ("ravenclaw-tower", "Ravenclaw Tower", ("RT",), "RT", 30.0, 60.0, ("wh-skyline", "ravenclaw-battlements"),
         ("castle", "Ravenclaw Tower")),
        ("west-tower", "West Tower (the Owlery)", ("WT",), "WT", 25.0, 50.0, ("towers-flat-battlemented",), None),
        ("great-hall", "Great Hall", ("GH",), "GH", 15.0, 50.0, ("castle-norman-openings", "warm-masonry"),
         GREAT_HALL),
        ("main-doors", "The Entrance Hall's main doors and their steps", ("EH",), "EH", 3.0, 16.0,
         ("astronomy-beside-doors", "castle-norman-openings"), ("castle", "Main doors and their steps, 17:00")),
        ("north-battlements", "North Battlements curtain wall", ("NB",), "NB", 8.0, 40.0,
         ("mont-orgueil-roofed-wall-walks", "towers-flat-battlemented"), None),
        ("east-wing", "East wing", ("EW",), "EW", 15.0, 50.0, ("castle-norman-openings", "small-scottish-turrets"),
         None),
        ("west-wing", "West wing (the west range)", ("WW",), "WW", 15.0, 50.0,
         ("castle-norman-openings", "small-scottish-turrets"), ("castle", "West range facade, 60 m")),
        ("hospital-wing", "Hospital wing", ("HW",), "HW", 15.0, 40.0, ("castle-norman-openings",), None),
        ("heads-tower", "The Head's tower and the Physick Garden", ("HO",), "HO", 15.0, 40.0,
         ("castle-norman-openings",), ("castle", "Head's tower and Physick Garden, 15:00")),
        ("flitwicks-office", "Flitwick's office", ("FO",), "FO", 15.0, 30.0, ("castle-norman-openings",), None),
        ("broomshed-pend", "The broomshed court and its gated pend", ("BS",), "BS", 3.0, 14.0,
         ("castle-norman-openings",), ("castle", "The pend from the broomshed court, 16:00")),
        ("keep", "The keep", (), "KEEP", 20.0, 45.0, ("castle-norman-openings", "towers-flat-battlemented"), None),
    )
    for s, title, labels, key, h, size, decisions, view in parts:
        x, z = where[key]
        views = ((view if isinstance(view, View) else sheet_view(*view)),) if view else ()
        out.append(obj(s, title, kind="part", at=at_point(world, x, z, h), size=size, decisions=decisions,
                       labels=tuple(f"wh-castle-plan:{k}" for k in labels), views=views,
                       hour=views[0].hour if views else DAY))
    towers_at = [(t[0], t[1], t[2]) for t in meta["towers"]]
    court = [p for p in world.pieces if p["owner"] == "castle" and p["role"] == "court" and "keep" not in p["classes"]
             and all(math.dist((p["x"], p["z"]), (x, z)) > r + 2.0 for x, z, r in towers_at)]   # a tower's base
    pieces = {p["id"]: p for p in world.pieces}
    for pid, kind, why in spread_samples([(p["id"], p["area"]) for p in court], "court buildings"):
        p = pieces[pid]
        out.append(obj(f"court-house-{pid}", "A house-like court building", kind=kind,
                       at=at_point(world, p["x"], p["z"], 6.0), size=max(12.0, 1.6 * math.sqrt(p["area"])),
                       decisions=("court-buildings-as-houses",), sample=why, elevation=40.0))
    outside = [p for p in world.pieces if p["owner"] == "castle" and p["role"] == "outside"]
    for p in outside:
        out.append(obj(f"wall-piece-{p['id']}", "A detached castle wall piece outside the walls", kind="outlier",
                       at=at_point(world, p["x"], p["z"], 1.0), size=max(6.0, 2.0 * math.sqrt(p["area"])),
                       decisions=("castle-speck-threshold",),
                       sample=f"one of the {len(outside)} castle wall slivers and stubs outside the walls"))
    named_at = [where[k] for k in ("AT", "RT", "GT", "NT", "WT")]
    sheds = [(p["x"], p["z"]) for p in world.pieces if p["owner"] != "castle" and p["role"] == "detached"]
    generic = [(i, t) for i, t in enumerate(meta["towers"])
               if min(math.dist(t[:2], w) for w in named_at) > 3.0 and all(math.dist(t[:2], s) > 5.0 for s in sheds)]
    towers = dict(generic)
    for i, kind, why in spread_samples([(i, t[2]) for i, t in generic], "other towers (by radius)"):
        x, z, r, top, _ = towers[i]
        h = max(10.0, (top - world.ground(x, z)) / 2)
        out.append(obj(f"tower-{i}", "A flat battlemented tower", kind=kind, at=at_point(world, x, z, h),
                       size=max(20.0, 2 * h), decisions=("towers-flat-battlemented",), sample=why))
    out.append(obj("lit-windows-dusk", "The castle's lit windows at civil dusk", kind="part",
                   at=(-30.0, 25.0, -85.0), size=120.0, decisions=("lamplit-windows",),
                   views=(sheet_view("windows", "West front at civil dusk"), sheet_view("windows", "West front at night")),
                   hour="civil dusk"))
    return out


def greenhouse_objects(world):
    out = []
    for asset in ("greenhouses-north", "greenhouses-5b", "court-glasshouse"):
        (at, size), box = asset_spot(world, asset)
        out.append(Obj(id=f"greenhouses/{asset}", title=next(a["title"] for a in world.manifest["assets"]
                                                             if a["id"] == asset),
                       family="greenhouses", kind="asset", at=at, size=size, assets=(asset,),
                       decisions=decisions_of(world, asset), labels=labels_of(world, asset), box=box,
                       elevation=40.0 if asset == "court-glasshouse" else 20.0))
        comps = {c["id"]: c for c in world.components.get(asset, [])}
        if len(comps) > 1:
            for cid, kind, why in spread_samples([(c["id"], c["area"]) for c in comps.values()],
                                                 f"glasshouses of {asset}"):
                c = comps[cid]
                out.append(Obj(id=f"greenhouses/{asset}-{cid}", title="A glasshouse, its panes, benches and plants",
                               family="greenhouses", kind=kind, at=at_point(world, c["x"], c["z"], 2.0),
                               size=max(8.0, 1.4 * math.sqrt(c["area"])), assets=(asset,),
                               decisions=("greenhouse-glass", "greenhouse-design-inferred"), sample=why, elevation=20.0))
    sheds = [p for p in world.pieces if p["owner"] == "greenhouses-5b"]
    for pid, kind, why in spread_samples([(p["id"], p["area"]) for p in sheds], "potting sheds"):
        p = next(q for q in sheds if q["id"] == pid)
        out.append(Obj(id=f"greenhouses/potting-shed-{pid}", title="A potting shed by the 5b greenhouses",
                       family="greenhouses", kind=kind, at=at_point(world, p["x"], p["z"], 3.0), size=16.0,
                       assets=("greenhouses-5b",), decisions=("castle-mass-sheds",), sample=why))
    return out


def transport_objects(world):
    t = world.meta["train"]
    (ax, az), (bx, bz) = t["carriages"]
    (ex, ez), _ = t["engine"]
    along = lambda f: (ax + f * (bx - ax), az + f * (bz - az))             # noqa: E731
    obj = lambda s, title, asset, **kw: Obj(id=f"transport/{s}", title=title, family="transport",   # noqa: E731
                                            assets=(asset,), **kw)
    sv = lambda label: (sheet_view("transport", label),)                   # noqa: E731
    (ship_at, _), ship = asset_spot(world, "ship")
    return [
        obj("station-building", "Hogsmeade station: its building and canopy", "station", kind="part",
            at=(340.7, 3.0, 110.2), size=30.0, decisions=("station-layout-either",),
            labels=("wh-grounds-map:1a",), views=sv("The building and canopy from the road, 15:00")),
        obj("station-platform", "The platform, its barrier and exits", "station", kind="part", at=(282.4, 2.5, 185.7),
            size=60.0, decisions=("station-layout-either", "station-terrace"),
            views=sv("Along the platform from its far end, 16:00"), hour=16.0),
        obj("station-dusk", "The station's lamplit windows at dusk", "station", kind="part", at=(340.7, 3.0, 110.2),
            size=40.0, decisions=("lamplit-windows",), views=sv("The station at dusk, 21:30"), hour=DUSK),
        obj("track-loop", "The track's turning loop", "track", kind="asset", at=(330.0, 0.0, 150.0), size=300.0,
            decisions=("railway-curves", "railway-regrade", "detached-engine"), labels=("wh-grounds-map:1a",),
            views=sv("The loop from the south-east, 15:00")),
        obj("track-junction-tunnel", "The junction cutting and the tunnel portal", "track", kind="part",
            at=(148.0, 5.0, 287.0), size=40.0, decisions=("railway-tunnel", "railway-regrade"),
            views=sv("Junction cutting and the tunnel portal, 15:00")),
        obj("track-embankment", "The embankment and culvert at the foot of the loop", "track", kind="part",
            at=(460.0, 8.0, 18.0), size=60.0, decisions=("railway-culvert", "railway-regrade"),
            views=(sheet_view("railway", "Embankment at the foot of the loop, from the south"),)),
        obj("track-cutting", "The rock cutting on the loop's north side", "track", kind="part", at=(290.0, 0.0, 262.0),
            size=60.0, decisions=("railway-regrade",),
            views=(sheet_view("railway", "North-side rock cutting from above"),)),
        obj("track-rails", "Rails, sleepers and ballast, close", "track", kind="part",
            at=at_point(world, (bx + ex) / 2, (bz + ez) / 2, 0.3), size=8.0, decisions=("railway-curves",),
            elevation=25.0),
        obj("engine-tender", "The engine and tender, detached on the curve", "train", kind="part",
            at=(420.0, 2.0, 30.0), size=25.0, decisions=("detached-engine", "generic-tender-locomotive"),
            views=sv("Engine and tender, detached on the curve, 16:00"), hour=16.0),
        obj("coach", "A scarlet coach", "train", kind="sample", at=at_point(world, *along(0.5), 2.0), size=24.0,
            decisions=("train-scarlet-coaches",), sample="the representative of the 9 coaches: the middle one",
            views=sv("Coaches from across the track, 14:00"), hour=14.0),
        obj("guards-van-nw", "The north-west guards' van", "train", kind="outlier",
            at=at_point(world, *along(0.02), 2.0), size=14.0, decisions=("guards-vans", "short-guards-vans"),
            sample="one of the 2 guards' vans, at the coaches' north-west end"),
        obj("guards-van-se", "The south-east guards' van", "train", kind="outlier",
            at=at_point(world, *along(1.0), 2.0), size=14.0, decisions=("guards-vans", "short-guards-vans"),
            sample="one of the 2 guards' vans, in the gap at the coaches' south-east end"),
        obj("coach-dusk", "The coaches' lamplit windows at dusk", "train", kind="part",
            at=at_point(world, *along(0.5), 2.0), size=40.0, decisions=("lamplit-windows",), hour=DUSK),
        obj("ship", "The Durmstrang ship", "ship", kind="asset", at=ship_at, size=60.0,
            decisions=("ship-skeletal-book",), labels=("wh-grounds-map:8d",), box=ship,
            views=sv("The Durmstrang ship from the shore, 18:30"), hour=18.5),
        obj("ship-dusk", "The ship's misty ports at dusk", "ship", kind="part", at=ship_at, size=60.0,
            decisions=("lamplit-windows",), views=(sheet_view("windows", "The ship from the bay, 21:30"),), hour=DUSK),
    ]


GROUNDS_PARTS = (  # slug, title, asset, close-up label, context label or None, decisions, labels, size
    ("hut-front", "Hagrid's hut, its front", "hut", "Its front, close up, 17:00",
     "Hagrid's hut from the castle side, 16:00",
     ("hut-plan", "timber-hut", "hut-front-gable", "hut-weatherboard-creosoted", "hut-roof-shingles", "hut-stovepipe"),
     ("wh-grounds-map:4a",), 14.0),
    ("hut-dusk", "Hagrid's hut at dusk", "hut", "Hagrid's hut at dusk, 21:30", None, ("lamplit-windows",), (), 20.0),
    ("garden", "Hagrid's garden from above and behind", "hut-garden", "The garden from above and behind, 15:00", None,
     ("garden-contents", "hut-plan"), ("wh-grounds-map:4b",), 50.0),
    ("pumpkins", "The pumpkin patch", "hut-garden", "The pumpkin patch from the back door, 16:00", None,
     ("garden-contents",), (), 10.0),
    ("garden-fence", "The garden's fence, gate and tree", "hut-garden", "The far fence and its tree, 16:00", None,
     ("garden-fence",), (), 14.0),
    ("pitch", "The Quidditch pitch, whole", "pitch", "The pitch from above the castle approach, 14:00",
     "The pitch from the path down from the castle, 15:00", ("pitch-layout-wh", "stands-full-ring"),
     ("wh-grounds-map:4c",), 200.0),
    ("pitch-entrance", "The pitch's entrance on the castle side", "pitch", "The entrance on the castle side, 15:00",
     None, ("pitch-layout-wh",), (), 20.0),
    ("pitch-stair", "A stair from the lawn", "pitch", "A stair from the lawn, north-west, 16:00", None,
     ("pitch-stairs-from-lawn",), (), 20.0),
    ("changing-rooms", "The changing rooms", "pitch", "The changing rooms from the pitch, 16:00", None,
     ("pitch-changing-room-materials",), (), 30.0),
    ("hoops", "The west goal hoops", "pitch", "The west hoops from above midfield, 16:00", None,
     ("pitch-goals-equal-height",), (), 30.0),
    ("podium", "The commentator's podium and the north stands", "pitch",
     "The podium and the north stands from above, 16:00", None, ("pitch-podium-clear", "stands-full-ring"), (), 20.0),
    ("gates-outside", "The winged-boar gates from the road outside", "gates", "The gates from the road outside, 15:00",
     None, ("gates-pillars-boars", "gates-closed"), ("wh-grounds-map:3g", "wh-hogsmeade-map:1a"), 20.0),
    ("gates-inside", "The gates from the drive inside", "gates", "The gates from the drive inside, 17:00", None,
     ("gates-closed",), (), 20.0),
    ("boar", "A winged boar", "gates", "A winged boar close up, 14:00", None, ("gates-pillars-boars",), (), 4.0),
    ("boundary-wall", "A coped boundary wall beside the gates", "grounds-walls",
     "A coped boundary wall beside the gates, 15:00", None, ("walls-coped", "third-entrance-speculative"),
     ("wh-grounds-map:3a",), 16.0),
    ("cliff-wall", "The cliff-edge safety wall (2e)", "grounds-walls", "The cliff-edge safety wall close up, 16:00",
     "The castle from the lake: the cliff-edge wall (2e), 15:00", ("cliff-edge-safety-wall",),
     ("wh-grounds-map:2e",), 30.0),
    ("vegetable-wall", "The vegetable-patch wall (5d)", "grounds-walls", "The vegetable-patch wall (5d), 15:00", None,
     ("vegetable-patch-wall-ruin",), ("wh-grounds-map:5d",), 20.0),
    ("tomb", "Dumbledore's tomb", "tomb", "Dumbledore's tomb, 16:00", None, ("tomb-chest",), ("wh-grounds-map:2c",),
     6.0),
    ("harbour-mouth", "The cliff opening to the underground harbour", "harbour-mouth",
     "The harbour mouth close, on a boat's line, 9:30", "The harbour mouth from the lake, from the south-east, 9:00",
     ("harbour-local-cutout", "harbour-heads-down", "exclude-boats", "tour-no-harbour"),
     ("wh-grounds-map:1f", "wh-castle-plan:WG"), 14.0),
    ("village-bridge", "A village bridge over the burn", "bridges", "A village bridge over the burn, 11:00", None,
     ("bridges-stone-arch",), (), 12.0),
    ("track-bridge", "The bridge on the track east of the lake", "bridges",
     "The bridge on the track east of the lake, 10:00", None, ("bridges-stone-arch",), (), 12.0),
    ("footbridge", "The footbridge on the Forest burn", "bridges", "The footbridge on the Forest burn, 13:00", None,
     ("bridges-stone-arch",), (), 12.0),
)
# the sheet's close-up of the tomb stands 2.5 m over the waterline looking inland, at the shingle, so its reflection
# shows only from over the lake: 40 m out from the shore nearest it (bearing 335 degrees), 4 m above its ground
OVER_WATER = {"tomb": View("from over the lake, 40 m out", (95.4, 4.0, -607.4), (116.7, 1.0, -653.8), 16.0, "target")}


def grounds_objects(world):
    out = []
    for s, title, asset, close, context, decisions, labels, size in GROUNDS_PARTS:
        views = (sheet_view("grounds", close),) + ((sheet_view("grounds", context),) if context else ())
        if s in OVER_WATER:                     # its reflection first, then the sheet's own close look
            views = (OVER_WATER[s], views[0])
        out.append(Obj(id=f"grounds/{s}", title=title, family="grounds", kind="part",
                       at=views[0].target, size=size, assets=(asset,), decisions=decisions, labels=labels,
                       views=views, hour=views[0].hour))
    (hut_at, _), hut = asset_spot(world, "hut")
    front = sheet_view("grounds", "Its front, close up, 17:00")
    back = (math.degrees(math.atan2(front.eye[0] - front.target[0], front.eye[2] - front.target[2])) + 180.0) % 360
    out.append(Obj(id="grounds/hut-back", title="Hagrid's hut, its back and side", family="grounds", kind="part",
                   at=hut_at, size=16.0, assets=("hut",), box=hut, azimuth=round(back, 1),
                   decisions=("hut-plan", "hut-stovepipe", "lamplit-windows")))
    for asset, what, decisions, cap in (("field-boundaries", "hedges", ("hedge-heuristic", "hedge-crowns"), 30.0),
                                        ("grounds-walls", "lengths of wall", ("walls-coped",), 30.0),
                                        ("bridges", "bridges", ("bridges-stone-arch",), 20.0)):
        groups = world.groups[asset]
        for k, kind, why in spread_samples([(k, math.hypot(b[3] - b[0], b[5] - b[2])) for k, (b, _, _) in
                                            enumerate(groups)], f"{what} (tiles of the payload)"):
            box, anchor, _ = groups[k]
            at, size = spot(world, box, anchor, cap)
            out.append(Obj(id=f"grounds/{asset}-{k}", title=f"One of the {what}", family="grounds", kind=kind,
                           at=at, size=size, assets=(asset,), decisions=decisions, sample=why, box=box,
                           elevation=20.0))
    village = (-982.0, -1079.0)           # Dervish & Banges, at the High Street's end towards the castle
    box, anchor = nearest_group(world, "grounds-walls", village)
    at, size = spot(world, box, anchor, 30.0)
    out.append(Obj(id="grounds/hogsmeade-wall", title="The boundary wall on Hogsmeade's side (1b)", family="grounds",
                   kind="part", at=at, size=size, assets=("grounds-walls",), decisions=("walls-coped",),
                   labels=("wh-hogsmeade-map:1b",), box=box, elevation=20.0))
    return out


def willow_objects(world):
    w = lambda label: sheet_view("willow", label)                         # noqa: E731
    obj = lambda s, title, views, decisions, **kw: Obj(                    # noqa: E731
        id=f"willow/{s}", title=title, family="willow", kind="part", at=views[0].target, size=kw.pop("size", 24.0),
        assets=("willow",), decisions=decisions, views=views, hour=views[0].hour, **kw)
    (willow_at, _), box = asset_spot(world, "willow")
    return [
        obj("whole", "The Whomping Willow", (w("The Willow from the castle side, 16:00"),
                                              w("Across the lawn from the castle's side, 17:00")),
            ("willow-limbs", "willow-crown", "willow-in-leaf", "willow-branch-clearance", "willow-root-gap"),
            labels=("wh-grounds-map:5a",), box=box),
        obj("roots", "Its root gap, arching root and knot", (w("The root gap and the knot, close up, 15:00"),),
            ("willow-root-gap", "willow-root-gap-size", "willow-arching-root", "willow-knot"), size=4.0),
        obj("limbs", "Its limbs from beneath", (w("Its limbs from beneath, the north-west side, 11:00"),),
            ("willow-limbs", "willow-branch-clearance")),
        obj("crown", "Its crown from above", (w("From above, 13:00"),), ("willow-crown", "willow-in-leaf")),
        Obj(id="willow/levels", title="The Willow at a distance, at its coarser levels", family="willow", kind="part",
            at=willow_at, size=140.0, assets=("willow",), decisions=("willow-levels",), box=box),
    ]


def tree_form(name):
    """A kit variant's species and form, as its name gives them: scots-pine-mature-1 is a scots-pine-mature."""
    return name.rsplit("-", 1)[0]


def tree_objects(world):
    """A tree of each of the kit's species and forms, with every decision on the Forest's trees that a picture can
    show. Its point is its class's seed (TREE_SEEDS): the shot frames the tree it finds from there (shoot.py)."""
    decisions = tuple(d for d in decisions_of(world, "forest-trees") if d not in NOT_VISUAL)
    forms = {}
    for v in world.kit:
        forms.setdefault(tree_form(v["name"]), []).append(v)
    out = []
    for form, variants in forms.items():
        height = max(v["height"] for v in variants)
        out.append(Obj(id=f"trees/{form}", title=TREE_TITLES.get(form, form), family="trees", kind="species",
                       at=at_point(world, *TREE_SEEDS[variants[0]["conifer"]], height / 2), size=round(1.3 * height, 1),
                       assets=("forest-trees",), decisions=decisions, elevation=10.0,
                       sample=f"the page's {form} trees (the kit's {', '.join(v['name'] for v in variants)}): the tree "
                              "at the centre of the close-up, the nearest its class's seed that stands apart, where "
                              "its whole form shows"))
    return out


VILLAGE_VIEWS = {  # named building: (close-up, context) labels of tools/review/village.py
    "dervish-and-banges": ("Dervish & Banges' bracket clock, 15:00", "Down the High Street from Dervish & Banges, 15:00"),
    "gladrags": ("Gladrags' blind, 15:00", None),
    "three-broomsticks": ("The Three Broomsticks and Zonko's, 16:00", None),
    "post-office": ("The post office's owl loft and Scrivenshaft's crow steps from above the High Street, 12:00", None),
    "hogs-head": ("The Hog's Head from its side street, 15:00", None),
    "honeydukes": ("Honeydukes from across the High Street, 11:00", None),
    "puddifoots": ("Madam Puddifoot's up its side road, 16:00", None),
    "kirk": ("The kirk, its porch and kirkyard from the air, 14:00", None),
    "school": ("The school, 11:00", None),
    "distillery": ("The distillery's kiln, 10:00", None),
    "carriage-hangar": ("The carriage hangar from the air, 15:00", None),
    "shrieking-shack": ("The Shrieking Shack and its fence, 16:00", None),
}


def village_objects(world):
    import mesh_village
    out = []
    lamps = world.small.get("lamps", [])
    for asset in mesh_village.NAMED:
        (at, size), box = asset_spot(world, asset)
        close, context = VILLAGE_VIEWS.get(asset, (None, None))
        views = tuple(sheet_view("village", v) for v in (close, context) if v)
        azimuth = None
        if asset in STREET_FRONTS and lamps and not views:      # from the High Street: its nearest lamp's side
            lx, lz, _ = min(lamps, key=lambda p: math.hypot(p[0] - at[0], p[1] - at[2]))
            azimuth = round(math.degrees(math.atan2(lx - at[0], lz - at[2])) % 360.0, 1)
        out.append(Obj(id=f"village/{asset}", title=next(a["title"] for a in world.manifest["assets"]
                                                         if a["id"] == asset),
                       family="village", kind="asset", at=at, size=max(size, 12.0), assets=(asset,),
                       decisions=decisions_of(world, asset), labels=labels_of(world, asset), box=box, views=views,
                       hour=views[0].hour if views else DAY, azimuth=azimuth,
                       **({"elevation": STREET_FRONT_ELEVATION} if azimuth is not None else {})))
    house = decisions_of(world, "village-house")
    for u, kind, why in village_samples(world.units):
        out.append(Obj(id=f"village/{u['asset']}-{u['cid']}-{u['part']}-{u['k']}",
                       title=f"A {u['family']} building ({u['roof'].replace('village-', '')} roof)", family="village",
                       kind=kind, at=at_point(world, u["x"], u["z"], 3.0), size=max(10.0, 1.8 * math.sqrt(u["area"])),
                       assets=(u["asset"],), decisions=house if u["asset"] == "village-house" else
                       decisions_of(world, u["asset"]), sample=why, elevation=20.0))
    wings = world.groups.get("village-wing", [])
    for k, kind, why in spread_samples([(k, math.hypot(b[3] - b[0], b[5] - b[2])) for k, (b, _, _) in
                                        enumerate(wings)], "tiles of wings and outshots that keep the generic massing"):
        box, anchor, _ = wings[k]
        at, size = spot(world, box, anchor, 40.0)
        out.append(Obj(id=f"village/wing-massing-{k}", title="Wings and outshots kept as the generic massing",
                       family="village", kind=kind, at=at, size=max(size, 8.0), assets=("village-wing",),
                       decisions=decisions_of(world, "village-wing"), sample=why, box=box, elevation=30.0))
    lamps = world.small.get("lamps", [])
    if lamps:
        a, b = np.array(lamps[0][:2]), np.array(lamps[-1][:2])
        axis = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        along = [(k, float(np.dot(np.array(p[:2]) - a, axis))) for k, p in enumerate(lamps)]
        for k, kind, why in spread_samples(along, "High Street lamps", ends=("lamp at the far end", "lamp at the near end"),
                                           middle="midway along the street"):
            x, z, _ = lamps[k]
            out.append(Obj(id=f"village/lamp-{k}", title="A High Street lamp", family="village", kind=kind,
                           at=at_point(world, x, z, 2.0), size=5.0, assets=("street-furniture",),
                           decisions=decisions_of(world, "street-furniture"), sample=why,
                           labels=("wh-hogsmeade-map:1h",) if kind == "sample" else ()))
    headstones = world.small.get("headstones", [])
    if headstones:
        xs = np.array([h[:2] for h in headstones])
        out.append(Obj(id="village/kirkyard", title="The kirkyard: its headstones and wall", family="village",
                       kind="sample", at=at_point(world, *xs.mean(axis=0), 0.8), size=14.0, assets=("kirk",),
                       decisions=("village-ancillary", "kirk-lantern-finial"), elevation=25.0,
                       sample=f"the kirkyard's {len(headstones)} headstones and {len(world.small['kirkyard-wall'])} "
                              f"lengths of wall, seen together"))
    walls = world.small.get("garden-walls", [])
    for k, kind, why in spread_samples([(k, w[2]) for k, w in enumerate(walls)], "garden walls near the High Street"):
        x, z, length = walls[k]
        out.append(Obj(id=f"village/garden-wall-{k}", title="A garden wall near the High Street", family="village",
                       kind=kind, at=at_point(world, x, z, 0.6), size=max(6.0, length), assets=("village-house",),
                       decisions=("village-ancillary",), sample=why, elevation=20.0))
    v = lambda label, **kw: sheet_view("village", label, **kw)             # noqa: E731
    out += [
        Obj(id="village/high-street", title="The High Street from the air", family="village", kind="part",
            at=v("The High Street from the air, 13:00").target, size=200.0, assets=("village-house",),
            decisions=("village-high-street", "village-roof-mix", "village-style-families", "no-half-timbering",
                       "thatch-straw-animals-optional", "village-detail-where-seen"),
            views=(v("The High Street from the air, 13:00"),), hour=13.0),
        Obj(id="village/streets", title="The streets south and east of the High Street", family="village",
            kind="part", at=v("The streets south and east of the High Street from the air, 13:00").target, size=300.0,
            assets=("village-house",), decisions=("village-units", "village-footprints-regularized"),
            views=(v("The streets south and east of the High Street from the air, 13:00"),), hour=13.0),
        Obj(id="village/lane", title="Down a lane east of the High Street", family="village", kind="part",
            at=v("Down a lane east of the High Street, 15:00").target, size=40.0, assets=("village-house",),
            decisions=("village-detail-where-seen",), views=(v("Down a lane east of the High Street, 15:00"),)),
        Obj(id="village/dusk", title="The High Street at dusk: lamps and lit windows", family="village",
            kind="part", at=v("Down the High Street at civil dusk").target, size=120.0, assets=("street-furniture",),
            decisions=("hogsmeade-lamps-high-street-only", "lamplit-windows", "village-night-windows"),
            views=(v("Down the High Street at civil dusk", hour=DUSK),), hour=DUSK),      # at 21:30, every lamp lit
    ]
    return out


def site_objects(world):
    meta = world.meta
    obj = lambda s, title, kind, at, size, decisions=(), **kw: Obj(        # noqa: E731
        id=f"site/{s}", title=title, family="site", kind=kind, at=at, size=size, decisions=decisions, **kw)
    gview = lambda label: (sheet_view("grounds", label),)                 # noqa: E731
    out = [
        obj("setting", "The setting: the castle on its shelf above the lake, from the tour's opening", "terrain",
            (0.0, 40.0, 0.0), 1500.0, ("site-x", "scope-exterior", "exclude-gof-set-pieces"),
            views=(View("the tour's opening point", (1100.0, 300.0, -1500.0), (0.0, 40.0, 0.0), 18.5),), hour=18.5),
        obj("castle-mound", "The castle's mound and its lawns", "terrain", (-40.0, 0.0, -60.0), 260.0,
            ("castle-base-mound",), azimuth=225.0, elevation=15.0),
        obj("lake-cliff", "The castle's cliff above the lake", "terrain", (80.0, 20.0, 0.0), 120.0,
            ("castle-base-lake-cliff",), views=gview("The castle from the lake: the cliff-edge wall (2e), 15:00")),
        obj("vegetable-patch", "The vegetable patch in its hollow", "terrain",
            at_point(world, *world.classes.get(4, (-114.6, 217.4)), 0.0), 60.0,
            ("vegetable-patch-hollow", "vegetable-patch-beds"), elevation=25.0),
        obj("outflow", "The lake's outflow where it leaves the castle's base", "terrain", (-150.0, 0.0, 297.0), 50.0,
            ("outflow-under-the-base",), elevation=20.0),
        obj("cabin-terrace", "Hagrid's cabin on its terrace above the gates", "terrain", (-300.0, 2.0, -200.0), 300.0,
            ("cabin-above-the-gates",),
            views=(View("from beyond the gates", (-520.0, 25.0, -480.0), (-300.0, 2.0, -200.0), DAY),)),
        obj("drive-dip", "The drive's dip by the lake", "terrain", at_point(world, -88.0, -212.0, 0.0), 80.0,
            ("drive-dips-by-the-lake",), elevation=15.0),
        obj("willow-knoll", "The Willow on its knoll", "terrain", at_point(world, *meta["extras"]["willow"], 5.0),
            120.0, ("willow-on-its-knoll",), azimuth=90.0, elevation=10.0),
        obj("station-wood", "The castle hidden from the station by the wood", "terrain", (0.0, 30.0, 0.0), 300.0,
            ("station-wood",), views=(View("from the platform", (290.0, 2.0, 160.0), (0.0, 30.0, 0.0), DAY),)),
        obj("train-screen", "The castle hidden from the train's approach", "terrain", (0.0, 30.0, 0.0), 300.0,
            ("train-screen",), views=(View("from the tunnel portal", (150.0, 4.0, 285.0), (0.0, 30.0, 0.0), DAY),)),
        obj("high-fringe", "From the High Street up to the gates", "terrain", (-444.5, 3.0, -505.5), 400.0,
            ("hogsmeade-high-fringe",),     # from over the gardens' hedges: at 2 m the camera stood inside one
            views=(View("from above the High Street", (-800.0, 12.0, -1180.0), (-444.5, 3.0, -505.5), DAY),)),
        obj("forest-above", "The Forbidden Forest from above", "terrain", (-900.0, 0.0, 1500.0), 1200.0,
            ("forest-extent-wh-reasoning", "forest-tracks-kept"),
            views=(View("the tour's Forest waypoint", (-420.0, 110.0, 760.0), (-900.0, 0.0, 1500.0), DAY),)),
        obj("forest-edge", "The Forest's edge against the grounds", "terrain",   # by Hagrid's hut, from the lawn
            at_point(world, *world.classes.get(FOREST_FLOOR, (-330.9, -193.1)), 5.0), 80.0,
            ("forest-extent-wh-reasoning",), azimuth=world.sides.get(FOREST_FLOOR, 64.9), elevation=10.0),
        obj("sunset", "The sky, the sun and the mountains at sunset", "terrain", (-1000.0, 60.0, -300.0), 1500.0,
            ("sunset-shortfall",),
            views=(View("west at sunset", (0.0, 60.0, -300.0), (-1000.0, 60.0, -300.0), 19.75),), hour=19.75),
        obj("sirius-cave", "Sirius's cave on the mountain", "terrain",
            at_point(world, *place(meta, "Sirius's cave"), 0.0), 600.0,
            views=(View("the tour's waypoint", (-1900.0, 250.0, 350.0), (-5018.0, 0.0, 888.0), DAY),)),
        obj("beach", "The lake shore whose ink reads as glass: no greenhouse there", "absence", (255.0, 0.0, -85.0),
            140.0, ("beach-false-positives",), elevation=20.0),
        obj("hamlet", "The unbuilt hamlet on the hill north-west of the Shack", "absence", (-1072.0, 0.0, -621.0), 90.0,
            ("shack-above-the-village",), elevation=25.0),
    ]
    for name, x, z, rx, rz, height in meta.get("hills", []):
        out.append(obj(f"hill-{slug(name)}", f"Terrain: {name}", "terrain", at_point(world, x, z, height / 2),
                       max(60.0, 2.5 * max(rx, rz)), ("dashed-outlines",) if name in DASHED_HILLS else (),
                       elevation=10.0))
    homesteads = [c for c in meta.get("clearings", []) if c["decision"] == "forest-homesteads"]
    centre = lambda c: np.asarray(c["outline"], float).mean(axis=0)        # noqa: E731

    def area(c):
        x, z = np.asarray(c["outline"], float).T
        return float(abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1))) / 2)
    for k, kind, why in spread_samples([(k, area(c)) for k, c in enumerate(homesteads)], "homesteads' clearings"):
        c = homesteads[k]
        out.append(obj(f"clearing-homestead-{k}", "A homestead's clearing in the Forest", "clearing",
                       at_point(world, *centre(c), 0.0), max(80.0, 1.5 * math.sqrt(area(c))),
                       ("forest-homesteads", "clearings-plausible"), sample=why, elevation=30.0))
    for c in meta.get("clearings", []):
        if c["decision"] == "forest-homesteads":
            continue
        decisions = tuple(dict.fromkeys((c["decision"], "clearings-plausible")))
        out.append(obj(f"clearing-{c['id']}", f"The clearing: {c['id']}", "clearing",
                       at_point(world, *centre(c), 0.0), max(80.0, 1.5 * math.sqrt(area(c))), decisions,
                       elevation=25.0))
    recorded = {c["id"] for c in meta.get("clearings", [])}
    for name in FOREST_PLACES:             # until the build records them as clearings, their points of interest
        if slug(name.replace("'", "")) not in recorded:
            out.append(obj(f"clearing-{slug(name.replace(chr(39), ''))}", f"The clearing: {name}", "clearing",
                           at_point(world, *place(meta, name), 0.0), 150.0,
                           ("forest-extent-wh-reasoning", "clearings-plausible"), elevation=30.0))
    import clearings
    linear = {  # (east, north) of a point on it, size, the side to shoot from
        "the railway's earthworks, and its lineside within 10 m of the open line": ((200.0, 280.0), 150.0, 200.0),
        "the Forest's corridors along the A890 and the Kyle railway": (world.corridor or (-1948.6, 916.9), 300.0, 135.0),
        "the Forest's edge short of the A832": ((-500.0, 4200.0), 600.0, 180.0),
        "the Forest's edge round Achnasheen, along the A832": ((204.5, 4255.8), 500.0, 180.0),
    }
    for k, (what, decision) in enumerate(clearings.LINEAR.items()):
        (x, z), size, azimuth = linear[what]
        out.append(obj(f"linear-clearing-{k}", f"A clearing along a linear feature: {what}", "clearing",
                       (float(x), 0.0, float(z)), size, (decision, "clearings-plausible"), azimuth=azimuth,
                       elevation=25.0, note=what))
    for c, name in enumerate(GROUND_CLASSES):
        if c not in world.classes:
            continue
        x, z = world.classes[c]
        out.append(obj(f"ground-{slug(name)}", f"Ground class: {name}", "surface", at_point(world, x, z, 0.0), 30.0,
                       CLASS_DECISIONS.get(name, ()), azimuth=world.sides.get(c),
                       elevation=6.0 if c == FOREST_FLOOR else 20.0))       # under the edge trees' crowns
    return out


def renderer_objects(world):
    lake = world.meta["extras"]["lake_c"]
    obj = lambda s, title, at, size, views=(), decisions=(), **kw: Obj(     # noqa: E731
        id=f"renderer/{s}", title=title, family="renderer", kind="surface", at=at, size=size, views=views,
        decisions=decisions, hour=views[0].hour if views else kw.pop("hour", DAY), **kw)
    north_glass = next((o for o in greenhouse_objects(world) if o.kind == "sample"
                        and o.assets == ("greenhouses-north",)), None)
    return [
        obj("lake-water", "Lake water: its colour, ripples and sun glint", (lake[0], 0.0, lake[1]), 60.0,
            (View("low over the water", (lake[0] + 50.0, 4.0, lake[1] + 50.0), (lake[0], 0.0, lake[1]), 18.5),)),
        obj("reflection-day", "The castle reflected in the lake by day", (0.0, 30.0, 0.0), 300.0,
            (sheet_view("windows", "Castle and its reflection across the lake, night", hour=DAY),)),
        obj("reflection-night", "Lit windows shining in the lake at night", (0.0, 30.0, 0.0), 300.0,
            (sheet_view("windows", "Castle and its reflection across the lake, night"),), ("lamplit-windows",)),
        # the widest burn in the open, below Hogsmeade (the outflow's first stretch runs buried, under the castle's base)
        obj("streams", "Burns: running water in the streams", (-1135.4, 0.0, -1307.9), 30.0, elevation=17.0),
        obj("greenhouse-glass", "Greenhouse glass: plants through overlapping panes", north_glass.at if north_glass
            else (-43.5, 2.0, 260.7), 10.0, (), ("greenhouse-glass",), elevation=20.0),
        obj("shadows", "Mesh shadows on the ground in the evening", (60.0, 0.0, -60.0), 200.0,
            (View("the castle's shadow at 18:30", (60.0, 30.0, -300.0), (60.0, 0.0, -60.0), 18.5),)),
        obj("trees", "Trees close: broadleaves and pines", (-951.0, 14.0, 46.0), 40.0,
            elevation=30.0),        # over the canopy: lower, the camera stood inside a crown
        obj("night", "The whole scene at night", (0.0, 30.0, 0.0), 800.0,
            (View("from 700 m at night", (420.0, 140.0, -620.0), (0.0, 30.0, 0.0), "night"),)),
        obj("levels-distance", "Levels of detail from a distance: the village from 900 m", (-860.0, 0.0, -1100.0),
            600.0, (View("from 900 m", (-300.0, 120.0, -400.0), (-860.0, 0.0, -1100.0), DAY),), ("village-levels",)),
        obj("thin-parts", "Thin parts from 250 m: the goal hoops' poles", (-311.8, 10.0, -484.4), 40.0,
            (View("from 250 m", (-60.0, 20.0, -486.0), (-311.8, 10.0, -484.4), DAY),)),
    ]


TOUR_SUBJECTS = {
    "Hogwarts Castle": ("castle/whole",), "The Lake": ("renderer/lake-water", "renderer/reflection-day"),
    "Durmstrang ship": ("transport/ship",), "Gryffindor Tower": ("castle/gryffindor-tower",),
    "Hogsmeade station": ("transport/station-building", "transport/station-platform"),
    "North Tower": ("castle/north-tower",), "Greenhouses": ("greenhouses/greenhouses-north", "greenhouses/greenhouses-5b"),
    "Whomping Willow": ("willow/whole",), "Forbidden Forest": ("site/forest-above",),
    "Entrance Hall & Great Hall": ("castle/main-doors", "castle/great-hall"),
    "West Tower (Owlery)": ("castle/west-tower",), "Hagrid's hut": ("grounds/hut-front",),
    "Astronomy Tower": ("castle/astronomy-tower",), "Quidditch pitch": ("grounds/pitch",),
    "Winged-boar gates": ("grounds/gates-outside",), "Hogsmeade": ("village/high-street",),
    "Hogsmeade High Street": ("village/high-street", "village/three-broomsticks", "village/honeydukes"),
    "Shrieking Shack": ("village/shrieking-shack",), "Dumbledore's tomb": ("grounds/tomb",),
    "Sirius's cave": ("site/sirius-cave",),
}
TOUR_DECISIONS = {
    "Hogwarts Castle": ("tour-opening-wide",),
    "Durmstrang ship": ("tour-ship-close",), "Gryffindor Tower": ("tour-gryffindor-close",),
    "Hogsmeade station": ("tour-station-close",), "North Tower": ("tour-north-tower-close",),
    "Greenhouses": ("tour-greenhouses-close",), "Whomping Willow": ("tour-willow-close",),
    "Entrance Hall & Great Hall": ("tour-great-hall-closer",), "West Tower (Owlery)": ("tour-west-tower-close",),
    "Hagrid's hut": ("tour-hut-close",), "Astronomy Tower": ("tour-astronomy-close",),
    "Quidditch pitch": ("tour-pitch-close",), "Winged-boar gates": ("tour-gates-close",),
    "Hogsmeade": ("tour-hogsmeade-low",), "Hogsmeade High Street": ("tour-high-street",),
    "Shrieking Shack": ("tour-shack-from-above",), "Dumbledore's tomb": ("tour-tomb-close-pass",),
    "Aragog's hollow": ("tour-aragogs-hollow",), "Grawp's clearing": ("tour-grawps-clearing",),
}


def tour_objects(world, objects):
    ids = {o.id for o in objects}
    out = []
    for k, w in enumerate(world.waypoints):
        caption = w[6]
        subjects = TOUR_SUBJECTS.get(caption, ())
        if caption in FOREST_PLACES:
            subjects = (f"site/clearing-{slug(caption.replace(chr(39), ''))}",)
        decisions = TOUR_DECISIONS.get(caption, ())
        if caption is None and k > 0 and world.waypoints[k - 1][6] == "Gryffindor Tower":
            decisions = ("tour-castle-east-transition",)
        out.append(Obj(id=f"tour/{k + 1:02d}-{slug(caption or 'between')}", title=f"Tour waypoint {k + 1}: "
                       f"{caption or '(no caption)'}", family="tour", kind="tour", at=(float(w[3]), float(w[5]),
                                                                                     float(w[4])),
                       size=100.0, decisions=decisions, subjects=tuple(s for s in subjects if s in ids),
                       note="the still as the flyover shows it, at the moment it passes the waypoint"))
    return out


def build(world):
    objects = (castle_objects(world) + greenhouse_objects(world) + transport_objects(world) + grounds_objects(world)
               + willow_objects(world) + village_objects(world) + site_objects(world) + tree_objects(world)
               + renderer_objects(world))
    return objects + tour_objects(world, objects)


# ---------------------------------------------------------------- coverage
def after_dark(o):
    """Whether an object's close-up is shot at dusk or later."""
    hour = o.views[0].hour if o.views else o.hour
    return hour in DARK_HOURS if isinstance(hour, str) else hour >= DUSK


def uncovered(objects, world):
    """{category: [what no object covers]} for every category the inventory answers for."""
    manifest = world.manifest
    assets = {a for o in objects for a in o.assets}
    decisions = {d for o in objects for d in o.decisions if d not in AFTER_DARK or after_dark(o)} | set(NOT_VISUAL)
    labels = {lb for o in objects for lb in o.labels}
    ids = {o.id for o in objects}
    stills = [o for o in objects if o.kind == "tour"]
    clearing_decisions = {d for o in objects if o.kind == "clearing" for d in o.decisions}
    clearing_notes = {o.note for o in objects if o.kind == "clearing"}
    import clearings
    excluded = {c["id"]: c["decision"] for c in manifest["excluded"]["components"]}
    unbuilt = {excluded[c["id"]] for c in world.components.get(None, []) if c["id"] in excluded}
    absent = {d for o in objects if o.kind == "absence" for d in o.decisions}
    out = {
        "assets": [a["id"] for a in manifest["assets"] if a["id"] not in assets],
        "decisions": [d for d in manifest["decisions"] if d not in decisions],
        "labels": [lb for a in manifest["assets"] for lb in a.get("labels") or () if lb not in labels],
        "waypoints": [k + 1 for k, w in enumerate(world.waypoints)
                      if k >= len(stills) or (w[6] and not stills[k].subjects)],
        "clearings": [c["id"] for c in world.meta.get("clearings", []) if c["decision"] not in clearing_decisions]
        + [w for w in clearings.LINEAR if w not in clearing_notes],
        "classes": [n for n in GROUND_CLASSES if f"site/ground-{slug(n)}" not in ids],
        "footprints": [str(a) for a in world.components if a is not None and a not in assets]
        + sorted(unbuilt - absent),
    }
    return out


def counts(objects):
    """{family: {kind: n}}."""
    out = {}
    for o in objects:
        out.setdefault(o.family, {}).setdefault(o.kind, 0)
        out[o.family][o.kind] += 1
    return out


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR / "inventory.json")
    args = parser.parse_args(argv)
    from tools import locks
    with locks.heavy("inspection inventory (tools.inspection.inventory)"):
        world = World.load()
        objects = build(world)
    gaps = {k: v for k, v in uncovered(objects, world).items() if v}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([o.as_dict() for o in objects], indent=1), encoding="utf-8")
    for family, kinds in counts(objects).items():
        print(f"{family:12} {sum(kinds.values()):4}  " + ", ".join(f"{k} {n}" for k, n in kinds.items()))
    print(f"{'all':12} {len(objects):4}  wrote {args.out}")
    for k, v in gaps.items():
        print(f"UNCOVERED {k}: {v}")
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
