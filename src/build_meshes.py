"""Build build/meshes.js: indexed exterior meshes in world metres from the registered layers and src/mesh_scene.json.

Each family in the manifest builds its own assets. Module mesh_<family>.py, where it exists, may define
    build_footprint(scene, asset, component, polygons) -> {asset: [Mesh]} or None    one claimed building footprint
    build_castle_pieces(scene, asset, keep) -> {asset: [Mesh]} or None               an asset's castle-grid pieces
    build_extras(scene) -> {asset: [Mesh]}                                            assets claiming no mapped structure
where `scene` is this build's Scene. A hook that returns None, and every family without a module, gets the generic
footprint massing, the shapes the raymarched objects had. A hook may only return its own family's assets.

Usage: python -X utf8 src/build_meshes.py [output]                        the whole payload
       python -X utf8 src/build_meshes.py --family NAME --part FILE        one family's encoded part (stage S1)
       python -X utf8 src/build_meshes.py --parts DIR [output]             every family's part merged (stage S2)
"""
import base64
import hashlib
import importlib
import importlib.util
import json
import math
import sys
import zlib
from functools import cached_property
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

import meshopt
import mesh_footprint
import mesh_levels
import mesh_thin
import railway
from materials import code
from mesh_geometry import Mesh, concat, cylinder, prism  # noqa: F401  (Mesh and prism are part of this module's API)
from mesh_source import (BUILDING_KINDS, CHANNELS, Grid, building_labels, castle_class_sd, castle_piece_labels, decode,
                         field_polygons, load_data, load_georef, sample, sd_bytes)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "src" / "mesh_scene.json"
DATA_JS = ROOT / "build" / "data.js"
OUT = ROOT / "build" / "meshes.js"
FORMAT, VERSION = "hogwarts-flyover/meshes", 4
STREAMS = ("pos", "nrm", "attr", "idx")
RAW = STREAMS + ("thin",)           # decoded, each vertex also has its thin radius (mesh_thin), in its position's padding
POS_STRIDE = 8                      # the vertex codec takes whole 4-byte words: a position's 6 bytes and its thin radius
MAX_VERTS = 65535                   # chunks use 16-bit indices
TILE_M = 128.0                      # chunks never span more than one tile, so they cull independently
CLAIMS = ("components", "castle_pieces", "rule")
READY = ("built", "integrated")
CLAIM_TOLERANCE_M = 1.0
SIMPLIFY_M = 0.05
SINK_M = 1.5                        # walls reach below the local ground so sloped terrain never shows a gap
MAT_CASTLE, MAT_BUILDING, MAT_WALL = 4, 5, 6   # G-buffer material ids the shading pass shades (frag5.glsl shade())
OPAQUE_MATS = (4, 5, 6, 7)
MAT_GLASS = 8                       # glazing: blended over the shaded image in its own pass, never in the G-buffer
CASTLE_CLASS_AUX = code("castle-class-painted")[1]      # castle mass class k is aux 10 + k; the mesh shader derives roof type and top from CAS_CLS
BUILDING_WALL_LEVEL = -0.3          # the fallback massing's walls stand 0.3 m inside the footprint distance, as bldD's did
TOWER_SEGMENTS = 24
# finer sun-shadow layers: each group's footprint gets its own depth layer; long low features stay on the coarse one
SHADOW_GROUPS = (("castle", "greenhouses"), ("grounds", "transport", "willow"), ("village",))
SPRAWLING = {"grounds-walls", "field-boundaries", "bridges", "track"}
SHADOW_MARGIN_M = 10.0


# ---------------------------------------------------------------- payload
def check_meshes(meshes, scene, family=None):
    """Validate an asset -> pieces map against the manifest; with `family`, only that family's assets need geometry."""
    assets = {a["id"]: a for a in scene["assets"]}
    unknown = sorted(set(meshes) - set(assets))
    if unknown:
        raise ValueError(f"geometry for assets missing from the manifest: {unknown}")
    for id, asset in assets.items():
        if family is not None and asset["family"] != family:
            continue
        if (set(CLAIMS) & set(asset) or asset.get("status") in READY) and not meshes.get(id):
            raise ValueError(f"{id} claims mapped structures but has no geometry")
    for id, parts in meshes.items():
        if family is not None and assets[id]["family"] != family:
            raise ValueError(f"{id} belongs to the {assets[id]['family']} family, not {family}")
        for m in parts:
            n = len(m.pos)
            if m.pos.shape != (n, 3) or m.nrm.shape != (n, 3) or m.attr.shape != (n, 2) or m.idx.shape[1:] != (3,):
                raise ValueError(f"{id}: inconsistent mesh arrays")
            if n > MAX_VERTS:
                raise ValueError(f"{id}: a piece with {n} vertices exceeds one chunk")
            if not (np.isfinite(m.pos).all() and np.isfinite(m.nrm).all()):
                raise ValueError(f"{id}: non-finite vertex data")
            if len(m.idx) and (m.idx.min() < 0 or m.idx.max() >= n):
                raise ValueError(f"{id}: triangle index outside its vertices")
            pass_of(m, id)


def pass_of(mesh, asset="mesh"):
    """'glass' for glazing pieces, 'opaque' for G-buffer pieces; a piece may not mix them."""
    mats = set(np.unique(mesh.attr[:, 0]).tolist())
    if mats == {MAT_GLASS}:
        return "glass"
    if not mats <= set(OPAQUE_MATS):
        raise ValueError(f"{asset}: materials {sorted(mats)} mix glazing with opaque faces or are unknown")
    return "opaque"


def without_degenerates(m):
    """The piece without zero-area triangles (lathe poles, collapsed caps) and the vertices only they used."""
    a, b, c = (m.pos[m.idx[:, k]] for k in range(3))
    keep = np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-12
    if keep.all():
        return m
    used, idx = np.unique(m.idx[keep].ravel(), return_inverse=True)
    return Mesh(m.pos[used], m.nrm[used], m.attr[used], idx.reshape(-1, 3).astype(np.uint32))


def chunks_of(parts):
    """An asset's pieces grouped by pass and tile (by piece centre), then packed into chunks under the vertex limit."""
    tiles = {}
    for m in filter(lambda m: len(m.idx), map(without_degenerates, parts)):
        cx, _, cz = (m.pos.min(axis=0) + m.pos.max(axis=0)) / 2
        tiles.setdefault((pass_of(m), math.floor(cx / TILE_M), math.floor(cz / TILE_M)), []).append(m)
    for key in sorted(tiles):
        group = []
        for m in tiles[key]:
            if group and sum(len(g.pos) for g in group) + len(m.pos) > MAX_VERTS:
                yield key[0], concat(group)
                group = []
            group.append(m)
        yield key[0], concat(group)


def sign(v):
    return np.where(v >= 0.0, 1.0, -1.0)


def oct_encode(n):
    s = np.abs(n).sum(axis=1, keepdims=True)
    if not (s > 0).all():
        raise ValueError(f"{int((s == 0).sum())} vertices have no normal")
    p = n[:, :2] / s
    fold = n[:, 2] < 0
    p[fold] = (1.0 - np.abs(p[fold][:, ::-1])) * sign(p[fold])
    return np.round(np.clip(p, -1.0, 1.0) * 127.0).astype(np.int8)


def oct_decode(e):
    p = e.astype(np.float64) / 127.0
    z = 1.0 - np.abs(p).sum(axis=1)
    fold = z < 0
    p[fold] = (1.0 - np.abs(p[fold][:, ::-1])) * sign(p[fold])
    n = np.column_stack([p, z])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def wrap_js(payload):
    return "const MESHES = " + json.dumps(payload, separators=(",", ":")) + ";\n"


def strip_js(text):
    return text[text.index("{"): text.rindex("}") + 1]


def quantized(m, lo, hi):
    """A chunk's 16-bit positions in its box, and the chunk less the triangles that quantization collapses to no area
    and the vertices only they used. The box still bounds what remains, so every kept position is unchanged."""
    q = np.round((m.pos - lo) / (hi - lo) * 65535).astype(np.int64)
    a, b, c = (q[m.idx[:, k]] for k in range(3))
    keep = np.cross(b - a, c - a).any(axis=1)                 # exact: collinear in integers is collinear decoded
    if not keep.all():
        used, idx = np.unique(m.idx[keep].ravel(), return_inverse=True)
        m, q = Mesh(m.pos[used], m.nrm[used], m.attr[used], idx.reshape(-1, 3).astype(np.uint32)), q[used]
    return q.astype("<u2"), m


def encode_assets(meshes, units=None):
    """Per asset (sorted): its chunk records, its level-of-detail groups and raw (uncompressed) per-chunk streams, as
    the payload will hold them. An asset in `units` ({asset: [{level: [pieces]}]}: its family's own levels of detail,
    whose full levels are the asset's pieces in `meshes`) is grouped unit by unit (unit_groups). Each lod0 chunk of every
    other asset is a group whose coarser levels the simplifier makes (generated_groups). A chunk the sun-shadow layers
    draw (an opaque full level or shadow caster) leads with its narrow parts' triangles, narrowest first, and lists their
    runs as `narrow` (mesh_thin.narrow_first)."""
    out = {}
    for asset in sorted(meshes):
        chunks, groups, streams, thin = [], [], {k: [] for k in RAW}, mesh_thin.codes(asset)
        found = unit_groups(units[asset]) if units and asset in units else generated_groups(meshes[asset], thin)
        for kind, levels, switch in found:
            mine, data, box = [], [], None
            for level in sorted(levels):
                for m in levels[level]:
                    lo = np.round(np.floor(m.pos.min(axis=0) * 1000) / 1000, 3)
                    hi = np.maximum(np.round(np.ceil(m.pos.max(axis=0) * 1000) / 1000, 3), lo + 0.001)
                    q, m = quantized(m, lo, hi)
                    if not len(m.idx):
                        continue
                    runs = []
                    if kind == "opaque" and level in (0, 3):
                        order, runs = mesh_thin.narrow_first(mesh_thin.widths(m, q))
                        m = Mesh(m.pos, m.nrm, m.attr, m.idx[order])
                    mine.append({"asset": asset, **({"pass": kind} if kind != "opaque" else {}),
                                 "box": [float(v) for v in (*lo, *hi)], "verts": len(m.pos), "tris": len(m.idx),
                                 "group": len(groups), "level": level, **({"narrow": runs} if runs else {})})
                    if level == 0:
                        box = (lo, hi) if box is None else (np.minimum(box[0], lo), np.maximum(box[1], hi))
                    data.append((q.tobytes(), oct_encode(m.nrm).tobytes(), m.attr.astype(np.uint8).tobytes(),
                                 m.idx.astype("<u2").tobytes(), mesh_thin.quantized(mesh_thin.radii(m, thin)).tobytes()))
            if box is None:                              # nothing of its full level survived quantization
                continue
            groups.append({"box": [float(v) for v in (*box[0], *box[1])], "switch": switch})
            chunks += mine
            for row in data:
                for k, v in zip(("pos", "nrm", "attr", "idx", "thin"), row):
                    streams[k].append(v)
        out[asset] = {"chunks": chunks, "groups": groups, "streams": {k: b"".join(v) for k, v in streams.items()}}
    return out


def split_mesh(m):
    """`m` as meshes under the vertex limit, its triangles in order."""
    if len(m.pos) <= MAX_VERTS:
        return [m]
    out, start = [], 0
    while start < len(m.idx):
        seen, end = set(), start
        while end < len(m.idx):
            new = {int(v) for v in m.idx[end]} - seen
            if len(seen) + len(new) > MAX_VERTS:
                break
            seen |= new
            end += 1
        used, idx = np.unique(m.idx[start:end].ravel(), return_inverse=True)
        out.append(Mesh(m.pos[used], m.nrm[used], m.attr[used], idx.reshape(-1, 3).astype(np.uint32)))
        start = end
    return out


def packed_pieces(pieces):
    """Pieces packed into meshes under the vertex limit, in order."""
    group = []
    for m in pieces:
        if group and sum(len(g.pos) for g in group) + len(m.pos) > MAX_VERTS:
            yield concat(group)
            group = []
        group.append(m)
    if group:
        yield concat(group)


def unit_groups(units):
    """(pass, {level: [meshes]}, switch distances of lod1 and lod2) per group of a family's own levels of detail: each
    unit's every level joins the group of its full level's pass and tile (as chunks_of tiles pieces), so a view never
    draws a unit twice or not at all. A unit without a level at all (a piece its family's levels leave out) stands in
    at that level with its full level; one whose level is empty has nothing there. Glazing keeps its full level only. A
    coarser level stands in from its switch distance (mesh_levels), its deviation measured against the full level; one
    that never may, or saves too little, is left out. The shadow caster (level 3) has none: only the shadow layers draw
    it, and a group gets one only where a unit has its own."""
    tiles = {}
    for unit in units:
        full = [m for m in map(without_degenerates, unit.get(0, [])) if len(m.idx)]
        if not full:
            continue
        cx, _, cz = (np.min([m.pos.min(axis=0) for m in full], axis=0) + np.max([m.pos.max(axis=0) for m in full], axis=0)) / 2
        tile = (math.floor(cx / TILE_M), math.floor(cz / TILE_M))
        mine = {}
        for level, pieces in unit.items():
            for m in filter(lambda m: len(m.idx), map(without_degenerates, pieces)):
                kind = pass_of(m)
                if kind == "opaque" or level == 0:
                    mine.setdefault(kind, {}).setdefault(level, []).append(m)
        for kind, found in mine.items():
            if 0 in found:
                tiles.setdefault((kind, *tile), []).append((set(unit), found))
    for key in sorted(tiles):
        members = tiles[key]
        at = lambda level: [m for own, found in members for m in (found.get(level, []) if level in own else found[0])]
        full = concat(at(0))
        levels, switch, previous = {0: list(packed_pieces(at(0)))}, [None, None], len(full.idx)
        if key[0] == "opaque":
            for level in (1, 2):
                pieces = at(level)
                if not pieces or not any(level in own for own, _ in members):
                    continue
                coarse = concat(pieces)
                if len(coarse.idx) > mesh_levels.SAVING * previous:
                    continue
                d = mesh_levels.switch_distance(full, coarse, mesh_levels.deviation(full, coarse))
                if d is not None:
                    levels[level], switch[level - 1], previous = list(packed_pieces(pieces)), d, len(coarse.idx)
            if any(3 in own for own, _ in members):
                levels[3] = list(packed_pieces(at(3)))
        yield key[0], levels, switch


def generated_groups(pieces, thin):
    """(pass, {level: [meshes]}, switch distances of lod1 and lod2) per lod0 chunk of an asset whose family has no
    levels of its own: lod1 and lod2 simplified from the chunk (mesh_levels.simplified), the parts with a thin radius
    (mesh_thin) copied unchanged, each standing in from its switch distance; glazing keeps its full level only."""
    for kind, m in chunks_of(pieces):
        levels, switch, previous = {0: [m]}, [None, None], len(m.idx)
        if kind == "opaque":
            keep = mesh_thin.radii(m, thin) > 0
            for level, budget in enumerate(mesh_levels.SIMPLIFY_M, 1):
                coarse, moved = mesh_levels.simplified(m, budget, keep)
                coarse = without_degenerates(coarse)                # slivers it flattened: no area, so no normal
                if len(coarse.idx) > mesh_levels.SAVING * previous:
                    continue
                d = mesh_levels.switch_distance(m, coarse, moved)
                if d is not None:
                    levels[level], switch[level - 1], previous = split_mesh(coarse), d, len(coarse.idx)
        yield kind, levels, switch


def packed(streams):
    """Version-3 stream bytes from the raw ones: positions, each with its thin radius as its fourth u16, through
    meshoptimizer's vertex codec, indices through its index-sequence codec (exact: unlike the triangle codec it never
    rotates a triangle). Normals and (material, aux) pairs stay raw, because the vertex codec makes them larger."""
    wide = np.zeros((len(streams["pos"]) // 6, POS_STRIDE), np.uint8)
    wide[:, :6] = np.frombuffer(streams["pos"], np.uint8).reshape(-1, 6)
    wide[:, 6:] = np.frombuffer(streams["thin"], np.uint8).reshape(-1, 2)
    return {"pos": meshopt.encode_vertices(wide.tobytes(), POS_STRIDE), "nrm": streams["nrm"], "attr": streams["attr"],
            "idx": meshopt.encode_indices(np.frombuffer(streams["idx"], "<u2"))}


def merge(encoded, source=None, shadow=None, towers=None, footprint=None):
    """The payload text from encoded assets: chunks, their level-of-detail groups and streams in sorted asset order,
    each stream packed, then zlib-compressed and base64-encoded; `footprint` is the footprint layer's entry
    (mesh_footprint.encode). `levels` names the view the groups' switch distances are measured for
    (mesh_levels.REF_ROWS output rows, FOV_DEG)."""
    chunks, groups, streams = [], [], {k: [] for k in RAW}
    for asset in sorted(encoded):
        chunks += [{**c, "group": c["group"] + len(groups)} for c in encoded[asset]["chunks"]]
        groups += encoded[asset]["groups"]
        for k in RAW:
            streams[k].append(encoded[asset]["streams"][k])
    payload = {"format": FORMAT, "version": VERSION, **({"source": source} if source else {}),
               **({"shadow": shadow} if shadow else {}), **({"towers": towers} if towers and any(towers) else {}),
               **({"footprint": footprint} if footprint else {}),
               "levels": {"rows": mesh_levels.REF_ROWS, "fov": mesh_levels.FOV_DEG}, "groups": groups,
               "chunks": chunks,
               "streams": {k: base64.b64encode(zlib.compress(v, 9)).decode("ascii")
                           for k, v in packed({k: b"".join(v) for k, v in streams.items()}).items()}}
    return wrap_js(payload)


def encode_payload(meshes, scene, source=None, shadow=None, towers=None, footprint=None):
    """Validated, deterministic payload: per chunk 16-bit positions in its box with each vertex's thin radius
    (mesh_thin), octahedral int8 normals, (material, aux) bytes and 16-bit local indices, each stream packed (see
    packed), zlib-compressed and base64-encoded; `shadow` lists world boxes that get their own finer sun-shadow layer,
    `towers` the mesh towers' tops by meta.towers index and `footprint` the footprint layer's entry."""
    check_meshes(meshes, scene)
    return merge(encode_assets(meshes), source, shadow, towers, footprint)


def decode_streams(payload):
    """A payload's raw streams, byte for byte as the encoder quantized them. Only this version is read: the page ships
    with its own payload, so no other version ever meets this reader. The vertex codec cannot vouch for its count, so
    the raw normal and (material, aux) streams pin the vertex count before the positions decode."""
    if (payload.get("format"), payload.get("version")) != (FORMAT, VERSION):
        raise ValueError(f"not a version-{VERSION} mesh payload")
    verts, tris = sum(c["verts"] for c in payload["chunks"]), sum(c["tris"] for c in payload["chunks"])
    stored = {k: zlib.decompress(base64.b64decode(payload["streams"][k])) for k in STREAMS}
    for k in ("nrm", "attr"):
        if len(stored[k]) != 2 * verts:
            raise ValueError(f"{k} stream holds {len(stored[k])} bytes but the chunks need {2 * verts}")
    wide = np.frombuffer(meshopt.decode_vertices(stored["pos"], verts, POS_STRIDE), np.uint8).reshape(-1, POS_STRIDE)
    return {"pos": wide[:, :6].tobytes(), "nrm": stored["nrm"], "attr": stored["attr"],
            "idx": meshopt.decode_indices(stored["idx"], 3 * tris), "thin": wide[:, 6:].tobytes()}


def decode_payload(text):
    payload = json.loads(strip_js(text))
    raw = decode_streams(payload)
    pos = np.frombuffer(raw["pos"], "<u2").reshape(-1, 3)
    nrm = np.frombuffer(raw["nrm"], np.int8).reshape(-1, 2)
    attr = np.frombuffer(raw["attr"], np.uint8).reshape(-1, 2)
    idx = np.frombuffer(raw["idx"], "<u2").reshape(-1, 3)
    thin = np.frombuffer(raw["thin"], "<u2") * mesh_thin.UNIT_M
    out, v0, t0 = [], 0, 0
    for c in payload["chunks"]:
        v1, t1 = v0 + c["verts"], t0 + c["tris"]
        lo, hi = np.array(c["box"][:3]), np.array(c["box"][3:])
        local = idx[t0:t1].astype(np.uint32)
        if c["verts"] > MAX_VERTS or (len(local) and local.max() >= c["verts"]):
            raise ValueError(f"chunk of {c['asset']} indexes past its {c['verts']} vertices")
        out.append({**c, "pos": lo + pos[v0:v1] / 65535 * (hi - lo), "nrm": oct_decode(nrm[v0:v1]),
                    "attr": attr[v0:v1], "idx": local, "thin": thin[v0:v1]})
        v0, t0 = v1, t1
    payload["chunks"] = out
    return payload


# ---------------------------------------------------------------- claims
def matches(rule, c):
    return (c["region"] == rule["region"] and c["kind"] in rule["kinds"]
            and rule.get("min_area", 0) <= c["area"] <= rule.get("max_area", math.inf))


def component_owners(scene, components):
    """Component id -> owning asset id (None for excluded), from explicit claims, exclusions and rules."""
    by_id, owners = {c["id"]: c for c in components}, {}
    claims = [(a["id"], c) for a in scene["assets"] for c in a.get("components", [])]
    for owner, claim in claims + [(None, c) for c in scene["excluded"]["components"]]:
        c = by_id[claim["id"]]
        if math.dist(claim["at"], (c["x"], c["z"])) > CLAIM_TOLERANCE_M:
            raise ValueError(f"component {c['id']} is no longer where the manifest claims it")
        owners[c["id"]] = owner
    rules = [(a["id"], a["rule"]) for a in scene["assets"] if "rule" in a]
    for c in components:
        if c["id"] not in owners:
            hits = [owner for owner, rule in rules if matches(rule, c)]
            if len(hits) != 1:
                raise ValueError(f"component {c['id']} matches rules {hits}")
            owners[c["id"]] = hits[0]
    return owners


def piece_owners(scene, pieces):
    """Castle piece id -> owning asset id: whole roles first, then explicit positions."""
    owners = {}
    for a in scene["assets"]:
        for claim in a.get("castle_pieces", []):
            if "role" in claim:
                owners.update({p["id"]: a["id"] for p in pieces if p["role"] == claim["role"]})
            else:
                hits = [p for p in pieces if math.dist(claim["at"], (p["x"], p["z"])) <= CLAIM_TOLERANCE_M]
                if len(hits) != 1:
                    raise ValueError(f"{a['id']} castle claim at {claim['at']} matches {len(hits)} pieces")
                owners[hits[0]["id"]] = a["id"]
    missing = [p["id"] for p in pieces if p["id"] not in owners]
    if missing:
        raise ValueError(f"castle pieces without an owner: {missing}")
    return owners


# ---------------------------------------------------------------- the shared scene and generic massing
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def simplified(polygon):
    s = polygon.simplify(SIMPLIFY_M, preserve_topology=True)
    return s if s.is_valid and not s.is_empty and s.geom_type == "Polygon" else polygon


def split(polygon):
    """Pieces small enough for one chunk, cut along the tile grid only when needed."""
    from shapely.geometry import box

    if 4 * (len(polygon.exterior.coords) + sum(len(r.coords) for r in polygon.interiors)) < MAX_VERTS // 2:
        return [polygon]
    x0, z0, x1, z1 = polygon.bounds
    out = []
    for i in range(math.floor(x0 / TILE_M), math.floor(x1 / TILE_M) + 1):
        for j in range(math.floor(z0 / TILE_M), math.floor(z1 / TILE_M) + 1):
            part = polygon.intersection(box(i * TILE_M, j * TILE_M, (i + 1) * TILE_M, (j + 1) * TILE_M))
            out += [g for g in getattr(part, "geoms", [part]) if g.geom_type == "Polygon" and not g.is_empty]
    return out


class Scene:
    """What every family builds from, computed once per build: the decoded layers and grids, the ground, the claimed
    footprints and castle pieces with their owners, and the registered railway alignment and garden outline.
    `outputs` collects families' by-products for the payload (the castle's tower tops and skyline report)."""

    def __init__(self, data, manifest):
        self.data, self.manifest = data, manifest
        self.meta, self.tex = data["meta"], data["tex"]
        self.local, self.castle = Grid(self.meta["local"]), Grid(self.meta["castle"])
        self.bed, self.cbase = decode(self.tex["locBed"]), decode(self.tex["locCB"])
        self.assets = {a["id"]: a for a in manifest["assets"]}
        self.outputs = {}

    def ground(self, x, z):
        return sample(self.bed, self.local, x, z)

    def sunk(self, x, z):
        return self.ground(x, z) - SINK_M

    def family_of(self, asset):
        return self.assets[asset]["family"]

    @cached_property
    def georef(self):
        return load_georef()

    @cached_property
    def locA(self):
        return decode(self.tex["locA"])

    @cached_property
    def bsd(self):
        """Signed distance (m) to the mapped building footprints on the local grid."""
        return sd_bytes(self.locA[..., CHANNELS["locA"].index("bsd")])

    @cached_property
    def labelled(self):
        """(building label image, building components) on the local grid."""
        return building_labels(self.data)

    @cached_property
    def owners(self):
        return component_owners(self.manifest, self.labelled[1])

    @cached_property
    def castle_labelled(self):
        """(castle piece image, castle pieces) on the castle grid."""
        return castle_piece_labels(self.data)

    @cached_property
    def piece_owners(self):
        return piece_owners(self.manifest, self.castle_labelled[1])

    @cached_property
    def csd(self):
        return castle_class_sd(self.data)

    @cached_property
    def line(self):
        return railway.alignment(self.manifest, self.meta, self.georef)

    @cached_property
    def garden(self):
        traced = self.assets["hut-garden"]["traced"][0]["plan_px"]
        return Polygon(railway.grounds_world(np.asarray(traced, np.float64), self.georef))

    def footprint(self, c):
        """A claimed building component's outline polygons, traced as the raymarched buildings drew their walls."""
        labels = self.labelled[0]
        rows = slice(max(c["rows"][0] - 2, 0), c["rows"][1] + 3)
        cols = slice(max(c["cols"][0] - 2, 0), c["cols"][1] + 3)
        keep = labels[rows, cols] == c["id"]
        bsd = self.bsd[rows, cols]
        return (field_polygons(bsd, self.local, BUILDING_WALL_LEVEL, keep, (rows.start, cols.start))
                or field_polygons(bsd, self.local, 0.0, keep, (rows.start, cols.start)))

    def prisms(self, polygons, top, attr):
        """Footprint prisms from the sunk ground up to `top`, cut to chunk size."""
        return [prism(part, self.sunk, top, attr) for polygon in polygons for part in split(simplified(polygon))]

    def building(self, c, polygons):
        """The generic massing of a building footprint, as the raymarcher drew it (the stands' sloping top included)."""
        if c["kind"] == "stands":
            pitch, ground = self.meta["pitch"], self.ground
            axis = np.radians(pitch["brg"])

            def top(x, z):
                u = (x - pitch["c"][0]) * np.sin(axis) + (z - pitch["c"][1]) * np.cos(axis)
                v = -(x - pitch["c"][0]) * np.cos(axis) + (z - pitch["c"][1]) * np.sin(axis)
                ro = np.hypot(u / pitch["a"], v / pitch["b"])
                return ground(x, z) + 1.2 + 8.6 * smoothstep(0.83, 0.99, ro)
        else:
            rim = np.concatenate([np.asarray(p.exterior.coords) for p in polygons])
            top = float(np.median(self.ground(rim[:, 0], rim[:, 1]))) + c["eave"]
        return self.prisms(polygons, top, (MAT_BUILDING, BUILDING_KINDS.index(c["kind"])))

    def castle_classes(self, keep):
        """The generic massing of castle-grid pieces: each mass class to its height over the castle base."""
        out = []
        for k in range(len(self.meta["castle"]["cls"])):
            def top(x, z, rel=self.meta["castle"]["cls"][k][0]):
                return sample(self.cbase, self.local, x, z) + rel
            out += self.prisms(field_polygons(self.csd[..., k], self.castle, 0.0, keep), top,
                               (MAT_CASTLE, CASTLE_CLASS_AUX + k))
        return out


# ---------------------------------------------------------------- families
def family_module(name):
    """mesh_<name> when the family has a module (which must declare FAMILY = name), else None: generic massing."""
    module = f"mesh_{name}"
    if importlib.util.find_spec(module) is None:
        return None
    found = importlib.import_module(module)
    if getattr(found, "FAMILY", None) != name:
        raise ValueError(f"{module}.py is not the {name} family's module (it declares no FAMILY = {name!r})")
    return found


def massing(data, scene, families=None, s=None):
    """Asset id -> pieces for the manifest's families (all, or the named subset), plus the mesh towers' tops by
    meta.towers index and the castle's skyline report; `s`, the Scene to build them in (the levels of detail reuse the
    family builders it caches)."""
    s = s or Scene(data, scene)
    names = [f for f in scene["families"] if families is None or f in families]
    modules = {f: family_module(f) for f in names}
    meshes = {}

    def take(family, built, hook):
        for asset, parts in built.items():
            if s.family_of(asset) != family:
                raise ValueError(f"mesh_{family}.{hook} built {asset}, which belongs to {s.family_of(asset)}")
            meshes.setdefault(asset, []).extend(parts)

    for c in s.labelled[1]:                          # claimed building footprints, in component order
        owner = s.owners[c["id"]]
        if owner is None or s.family_of(owner) not in modules:
            continue
        family = s.family_of(owner)
        hook = getattr(modules[family], "build_footprint", None)
        polygons = s.footprint(c)
        built = hook(s, owner, c, polygons) if hook else None
        take(family, built if built is not None else {owner: s.building(c, polygons)}, "build_footprint")
    image = s.castle_labelled[0]
    for asset in sorted(set(s.piece_owners.values())):   # castle-grid pieces, asset by asset
        family = s.family_of(asset)
        if family not in modules:
            continue
        keep = np.isin(image, [p for p, a in s.piece_owners.items() if a == asset])
        hook = getattr(modules[family], "build_castle_pieces", None)
        built = hook(s, asset, keep) if hook else None
        take(family, built if built is not None else {asset: s.castle_classes(keep)}, "build_castle_pieces")
    for family, module in modules.items():          # assets without claimed structures
        hook = getattr(module, "build_extras", None)
        if hook:
            take(family, hook(s), "build_extras")
    tops = [0.0] * len(s.meta["towers"])
    for t in s.outputs.get("towers", []):
        tops[t["i"]] = round(float(t["top"]), 2)
    return meshes, tops, s.outputs.get("report", [])


def bounds(meshes):
    """Asset -> (lo, hi) world corners of its pieces (before chunking), for the shadow layers."""
    return {asset: (np.min([m.pos.min(axis=0) for m in parts], axis=0), np.max([m.pos.max(axis=0) for m in parts], axis=0))
            for asset, parts in meshes.items() if parts}


def shadow_regions(extents, scene):
    """World boxes (x0, y0, z0, x1, y1, z1) of each SHADOW_GROUPS footprint that has geometry, from asset extents."""
    family = {a["id"]: a["family"] for a in scene["assets"]}
    regions = []
    for group in SHADOW_GROUPS:
        boxes = [extents[asset] for asset in sorted(extents) if family[asset] in group and asset not in SPRAWLING]
        if boxes:
            lo = np.min([b[0] for b in boxes], axis=0) - SHADOW_MARGIN_M
            hi = np.max([b[1] for b in boxes], axis=0) + SHADOW_MARGIN_M
            regions.append([round(float(v), 1) for v in (*lo, *hi)])
    return regions


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def source_digests():
    return {"data": digest(DATA_JS), "manifest": digest(MANIFEST)}


def write(out, text):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="ascii", newline="\n")


# ---------------------------------------------------------------- family parts (stages S1 and S2)
def footprint_marks(encoded, local):
    """The silhouettes (mesh_footprint.silhouettes) of the encoded assets' triangles as the payload holds them, on the
    local grid (its meta)."""
    triangles = {}
    for asset, e in encoded.items():
        pos = np.frombuffer(e["streams"]["pos"], "<u2").reshape(-1, 3)
        idx = np.frombuffer(e["streams"]["idx"], "<u2").reshape(-1, 3)
        out, v0, t0 = [], 0, 0
        for c in e["chunks"]:
            lo, hi = np.array(c["box"][:3]), np.array(c["box"][3:])
            if c["level"] == 0:                          # the full level: coarser ones stand in for it from afar
                out.append((lo + pos[v0:v0 + c["verts"]] / 65535 * (hi - lo))[idx[t0:t0 + c["tris"]]])
            v0, t0 = v0 + c["verts"], t0 + c["tris"]
        triangles[asset] = np.concatenate(out) if out else np.zeros((0, 3, 3))
    return mesh_footprint.silhouettes(triangles, Grid(local))


def encode_part(family, data=None, scene=None):
    """One family's assets encoded for the payload, with their extents, the silhouettes of those the footprint layer
    marks (mesh_footprint) and the family's by-products (JSON-ready)."""
    scene = scene or json.loads(MANIFEST.read_text(encoding="utf-8"))
    data = data or load_data()
    s = Scene(data, scene) if family in mesh_levels.OWN_LEVELS else None
    meshes, tops, report = massing(data, scene, {family}, s)
    check_meshes(meshes, scene, family)
    extents = bounds(meshes)
    encoded = encode_assets(meshes, mesh_levels.family_units(family, s, meshes) if s else None)
    marked = footprint_marks(encoded, data["meta"]["local"])
    return {"family": family, "local": data["meta"]["local"],
            "assets": {a: {"chunks": e["chunks"], "groups": e["groups"],
                           "streams": {k: base64.b64encode(v).decode("ascii") for k, v in e["streams"].items()},
                           "extent": ([[float(v) for v in extents[a][0]], [float(v) for v in extents[a][1]]]
                                      if a in extents else None),
                           **({"footprint": mesh_footprint.pack(marked[a])} if a in marked else {})}
                       for a, e in encoded.items()},
            "towers": tops, "report": report}


def footprint_entry(marked, local):
    """The footprint layer's payload entry from the marked assets' silhouettes on the local grid (its meta)."""
    return mesh_footprint.encode(mesh_footprint.layer(marked, Grid(local)))


def merge_parts(parts, scene=None):
    """The payload text from every family's part: the same bytes as building all families at once."""
    scene = scene or json.loads(MANIFEST.read_text(encoding="utf-8"))
    family = {a["id"]: a["family"] for a in scene["assets"]}
    missing = sorted(set(scene["families"]) - {p["family"] for p in parts})
    if missing:
        raise ValueError(f"no part for the families {missing}")
    if any(p["local"] != parts[0]["local"] for p in parts):
        raise ValueError("the parts were built on different local grids")
    encoded, extents, towers, marked = {}, {}, None, {}
    for p in parts:
        for asset, e in p["assets"].items():
            if asset in encoded:
                raise ValueError(f"{asset} appears in two parts")
            if family.get(asset) != p["family"]:
                raise ValueError(f"the {p['family']} part holds {asset}, which is not one of that family's assets")
            encoded[asset] = {"chunks": e["chunks"], "groups": e["groups"],
                              "streams": {k: base64.b64decode(v) for k, v in e["streams"].items()}}
            if e["extent"] is not None:
                extents[asset] = (np.array(e["extent"][0]), np.array(e["extent"][1]))
            if "footprint" in e:
                marked[asset] = mesh_footprint.unpack(e["footprint"])
        if any(p["towers"] or ()):
            towers = p["towers"]
    for a in scene["assets"]:
        if (set(CLAIMS) & set(a) or a.get("status") in READY) and a["id"] not in encoded:
            raise ValueError(f"{a['id']} claims mapped structures but has no geometry")
    return merge(encoded, source_digests(), shadow_regions(extents, scene), towers,
                 footprint_entry(marked, parts[0]["local"]) if parts else None)


def clearing_report(marked, data):
    """Lines naming the tree cover (m2) that the footprint layer clears beside each marked asset, measured against the
    data's own building distances: the clearings the meshes open (mesh_footprint.cleared)."""
    grid, locA = Grid(data["meta"]["local"]), decode(data["tex"]["locA"])
    channel = lambda name: locA[..., CHANNELS["locA"].index(name)]
    trees = channel("trees") / 255.0 * (1.0 - channel("road") / 255.0)
    cleared = mesh_footprint.cleared(marked, grid, mesh_footprint.layer(marked, grid), sd_bytes(channel("bsd")), trees)
    return [f"{asset} {m2:.0f} m2" for asset, m2 in cleared.items() if m2 >= 0.5]


def main(argv):
    if "--family" in argv:
        family, out = argv[argv.index("--family") + 1], Path(argv[argv.index("--part") + 1])
        part = encode_part(family)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(part, separators=(",", ":")), encoding="ascii", newline="\n")
        print(f"wrote {out} ({len(part['assets'])} assets)")
        return
    if "--parts" in argv:
        i = argv.index("--parts")
        folder, rest = Path(argv[i + 1]), argv[:i] + argv[i + 2:]
        out = rest[0] if rest else OUT
        parts = [json.loads(p.read_text(encoding="ascii")) for p in sorted(folder.glob("*.json"))]
        text = merge_parts(parts)
        write(out, text)
        print(f"wrote {out} ({len(text) / 1e6:.2f} MB from {len(parts)} parts)")
        marked = {a: mesh_footprint.unpack(e["footprint"]) for p in parts for a, e in p["assets"].items()
                  if "footprint" in e}
        if marked:
            print("trees cleared beside the meshes:", "; ".join(clearing_report(marked, load_data())))
        return
    out = argv[0] if argv else OUT
    scene, data = json.loads(MANIFEST.read_text(encoding="utf-8")), load_data()
    s = Scene(data, scene)
    meshes, tower_tops, report = massing(data, scene, None, s)
    check_meshes(meshes, scene)
    units = {}
    for family in scene["families"]:
        units.update(mesh_levels.family_units(family, s, meshes))
    encoded = encode_assets(meshes, units)
    marked = footprint_marks(encoded, data["meta"]["local"])
    text = merge(encoded, source_digests(), shadow_regions(bounds(meshes), scene), tower_tops,
                 footprint_entry(marked, data["meta"]["local"]))
    write(out, text)
    print("skyline:", "; ".join(report))
    print("trees cleared beside the meshes:", "; ".join(clearing_report(marked, data)))
    print(f"wrote {out} ({len(text) / 1e6:.2f} MB, {sum(len(p) for p in meshes.values())} pieces)")


if __name__ == "__main__":
    main(sys.argv[1:])
