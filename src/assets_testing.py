"""Shared fixtures and probes of the asset tests (test_assets_*.py).

The registered world (build/data.js) and the built payload are read once per test run. HOG_DATUM_SHIFT_M=<metres>
raises every absolute height in the world by that much before any test reads it (src/datum_shift.py: the elevation
layers and the absolute levels in meta), and the payload is then merged from every family's part on the shifted world,
which stage S1 builds and caches keyed on the shift. A builder or a test that measures from a fixed level, rather than
from the lake or the castle platform, fails under a shift, so the scene can move to a site at another height.
"""
import copy
import functools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import build_meshes  # noqa: E402
from datum_shift import shifted  # noqa: E402
from mesh_source import Grid, decode, load_data  # noqa: E402

MANIFEST = ROOT / "src" / "mesh_scene.json"
PAYLOAD = ROOT / "build" / "meshes.js"
SHIFT_M = float(os.environ.get("HOG_DATUM_SHIFT_M") or 0.0)
FT = 0.3048


def scene():
    """The manifest, freshly parsed (tests may not share a mutable copy)."""
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def family_assets(family):
    return {a["id"] for a in scene()["assets"] if a["family"] == family}


@functools.lru_cache(maxsize=None)
def _world():
    world = load_data()
    return shifted(world, SHIFT_M) if SHIFT_M else world


def data():
    """The registered world, shifted by HOG_DATUM_SHIFT_M when it is set; each call gets its own copy to change."""
    world = _world()
    return {"tex": {name: dict(layer) for name, layer in world["tex"].items()}, "meta": copy.deepcopy(world["meta"])}


@functools.lru_cache(maxsize=None)
def levelled():
    """The decoded payload with every level of detail, shared and read-only: the checked-in one, or under a datum shift
    one merged from every family's part on the shifted world, which stage S1 builds and caches keyed on the shift
    (tools/stages.py)."""
    if SHIFT_M:
        if str(ROOT) not in sys.path:
            sys.path.insert(1, str(ROOT))
        from tools import stages  # noqa: PLC0415
        parts = [json.loads(path.read_text(encoding="ascii")) for path in stages.shifted_parts(SHIFT_M)]
        decoded = build_meshes.decode_payload(build_meshes.merge_parts(parts, scene()))
    else:
        decoded = build_meshes.decode_payload(PAYLOAD.read_text(encoding="utf-8"))
    for chunk in decoded["chunks"]:
        for value in chunk.values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
    return decoded


@functools.lru_cache(maxsize=None)
def payload():
    """levelled() with only the full level of detail's chunks: each asset's geometry as the families build it."""
    return {**levelled(), "chunks": [c for c in levelled()["chunks"] if c["level"] == 0]}


def chunks(asset):
    return [c for c in payload()["chunks"] if c["asset"] == asset]


def payload_part(asset):
    """(triangles, attrs, unit normals) of one asset in the built payload."""
    found = chunks(asset)
    if not found:
        raise AssertionError(f"no {asset} geometry in the payload")
    tris = np.concatenate([c["pos"][c["idx"]] for c in found])
    attr = np.concatenate([c["attr"][c["idx"][:, 0]] for c in found])
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return tris, attr, n / np.linalg.norm(n, axis=1, keepdims=True)


# ---------------------------------------------------------------- trimesh as an independent oracle on the decoded payload
SLIVER_M2 = 1e-3      # 16-bit quantization may turn a triangle smaller than this over; it then shades from its normals


def _trimeshes(assets):
    for asset in sorted(assets):
        for c in chunks(asset):
            yield asset, c, trimesh.Trimesh(vertices=c["pos"], faces=c["idx"], process=False, validate=False)


def turned(vertices, faces, normals):
    """Which faces, larger than a sliver, wind against the mean of their stored vertex normals (by trimesh)."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
    return (np.einsum("ij,ij->i", mesh.face_normals, normals[faces].mean(axis=1)) < 0) & (mesh.area_faces > SLIVER_M2)


def winding_faults(assets):
    """{asset: faces} of the assets' decoded triangles larger than a sliver that wind against their stored normals."""
    out = {}
    for asset, c, _ in _trimeshes(assets):
        if found := int(turned(c["pos"], c["idx"], c["nrm"]).sum()):
            out[asset] = out.get(asset, 0) + found
    return out


def collapsed_faces(assets=None):
    """{asset: faces} of the decoded triangles with no area: the encoder drops those before quantizing, not after."""
    out = {}
    for asset, _, mesh in _trimeshes(assets if assets is not None else {c["asset"] for c in payload()["chunks"]}):
        if found := int((mesh.area_faces <= 0).sum()):
            out[asset] = out.get(asset, 0) + found
    return out


def mesh_faults(family):
    """What trimesh finds wrong in a family's decoded meshes. Each family's test module pins its own as MESH_FAULTS,
    so a new fault fails its test, and so does a fix until the pin is updated."""
    assets = family_assets(family)
    return {"winding": winding_faults(assets), "collapsed": collapsed_faces(assets)}


def levels(world=None):
    """The scene's reference levels: the lake surface and the castle platform (the castle base's floor)."""
    world = world or data()
    return {"lake": world["meta"]["lake"], "platform": float(decode(world["tex"]["locCB"]).min())}


def datum():
    """The level synthetic fixtures stand on: the lake's, so they rise with a datum shift as the world does."""
    return levels()["lake"]


# ---------------------------------------------------------------- probes shared by several families' tests
GRID_DEG = 25.1   # the local and castle grids are both rotated by the essay-map registration
TEXEL_M = 2.0


def rotated_grid(n=40):
    """A map-aligned grid of n x n 2 m texels centred on the origin, rotated like the registered layers."""
    c, s, w = math.cos(math.radians(GRID_DEG)), math.sin(math.radians(GRID_DEG)), n * TEXEL_M
    return Grid({"nx": n, "ny": n, "aff": [[c / w, s / w, 0.5], [s / w, -c / w, 0.5]]})


def box_field(grid, centre, half_u, half_v):
    """Signed distance (m) at each texel centre to a rectangle aligned with the grid's own axes."""
    c, s = math.cos(math.radians(GRID_DEG)), math.sin(math.radians(GRID_DEG))
    ii, jj = np.mgrid[0:grid.ny, 0:grid.nx]
    x, z = grid.to_world(ii, jj)
    du, dv = x - centre[0], z - centre[1]
    q = np.stack([np.abs(du * c + dv * s) - half_u, np.abs(du * s - dv * c) - half_v])
    return np.linalg.norm(np.maximum(q, 0.0), axis=0) + np.minimum(q.max(axis=0), 0.0)


def triangle_areas(mesh, rows):
    a, b, c = (mesh.pos[mesh.idx[rows, k]][:, [0, 2]] for k in range(3))
    return 0.5 * np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])


def surface_heights(tris, x, z):
    """Heights at which the triangles (t, 3, 3) pass over the point (x, z); vertical triangles pass over nothing."""
    a, b, c = (tris[:, k] for k in range(3))
    v0, v1, v2 = (b - a)[:, [0, 2]], (c - a)[:, [0, 2]], np.array([x, z]) - a[:, [0, 2]]
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    with np.errstate(divide="ignore", invalid="ignore"):
        v, w = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
        over = (np.abs(den) > 1e-12) & (v >= 0) & (w >= 0) & (v + w <= 1)
        return (a[:, 1] + v * (b - a)[:, 1] + w * (c - a)[:, 1])[over]


def segment_hits(starts, directions, length, tris, cell=2.0):
    """Whether each segment start + s * direction, 0 < s < length, crosses any of the triangles (t, 3, 3)."""
    lo = np.floor(tris[:, :, [0, 2]].min(axis=1) / cell).astype(np.int64)
    hi = np.floor(tris[:, :, [0, 2]].max(axis=1) / cell).astype(np.int64)
    span = hi - lo + 1
    count = span[:, 0] * span[:, 1]
    owner = np.repeat(np.arange(len(tris)), count)
    k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    row = np.repeat(lo[:, 0], count) + k // np.repeat(span[:, 1], count)
    key = row * 1_000_000 + np.repeat(lo[:, 1], count) + k % np.repeat(span[:, 1], count)
    order = np.argsort(key, kind="stable")
    key, owner = key[order], owner[order]
    hits = np.zeros(len(starts), bool)
    for r, (s, d) in enumerate(zip(starts, directions)):
        e = s + d * length
        (i0, j0), (i1, j1) = (np.floor(v[[0, 2]] / cell).astype(np.int64) for v in (np.minimum(s, e), np.maximum(s, e)))
        near = np.unique(np.concatenate([owner[np.searchsorted(key, i * 1_000_000 + j0):
                                               np.searchsorted(key, i * 1_000_000 + j1, side="right")]
                                         for i in range(i0, i1 + 1)]))
        if not len(near):
            continue
        a, e1, e2 = tris[near, 0], tris[near, 1] - tris[near, 0], tris[near, 2] - tris[near, 0]
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-9
        inv = 1.0 / np.where(ok, det, 1.0)
        q = s - a
        u = np.einsum("ij,ij->i", q, p) * inv
        w = np.cross(q, e1)
        v = (w @ d) * inv
        t = np.einsum("ij,ij->i", e2, w) * inv
        hits[r] = np.any(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (t < length))
    return hits
