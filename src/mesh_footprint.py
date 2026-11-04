"""The footprint layer: the ground the standing meshes cover, as a signed distance on the local grid in the encoding of
the data's building-distance channel (locA's bsd: 128 + 10 * metres in a byte, -12.8 to 12.7 m). The page puts it in
place of that channel whenever it loads a payload, so the raymarcher's trees and micro-relief follow the meshes rather
than the mapped footprints: trees fade in between 0.5 and 2.5 m from a wall (frag1.glsl vegMask) and the micro-relief
between 0 and 3 m (detailH), as they did round the raymarched buildings. The footprints the manifest leaves unbuilt
(the hamlet above the village, the beach's false positives, the moved cottages' old sites) read as open ground.

POLICY says how each manifest asset marks the ground:
  BUILT      its silhouette seen from above, glazing included: nothing grows under it;
  ENCLOSURE  its convex hull: a fenced garden encloses its ground, whatever stands in it (the hut garden's hull keeps
             within 1.6 m of White Hound's traced outline, inside the clearing its fence makes anyway);
  None       nothing: the linear features (walls, hedges, the track, bridges), the train and the ship, the Willow (a
             tree among trees) and the street furniture (posts and lamps in the streets, which have no trees).

Silhouettes are rasterized at FINE cells per texel (0.5 m), and each texel holds the least distance over its own cell
rather than the distance at its centre. The page's bilinear lookup then reads below the tree line (0.5 m) at every point
under a mesh, however narrow (a gate pier, a fence rail): 0.32 m at worst over 300 random narrow features, where the
distances at the texel centres read up to 1.3 m. The price is a clearing about 1.2 m wider than the wall's own (1.4 m
at most)."""
import base64
import zlib

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, cKDTree

from mesh_source import decode, encode_u8

BUILT, ENCLOSURE = "built", "enclosure"
POLICY = {
    "castle": BUILT,
    "court-glasshouse": BUILT, "greenhouses-north": BUILT, "greenhouses-5b": BUILT,
    "station": BUILT, "track": None, "train": None, "ship": None,
    "hut": BUILT, "hut-garden": ENCLOSURE, "pitch": BUILT, "gates": BUILT, "grounds-walls": None,
    "field-boundaries": None, "tomb": BUILT, "harbour-mouth": BUILT, "bridges": None,
    "willow": None,
    "dervish-and-banges": BUILT, "gladrags": BUILT, "three-broomsticks": BUILT, "zonkos": BUILT,
    "post-office": BUILT, "hogs-head": BUILT, "scrivenshafts": BUILT, "honeydukes": BUILT, "puddifoots": BUILT,
    "kirk": BUILT, "school": BUILT, "distillery": BUILT, "village-hall": BUILT, "carriage-hangar": BUILT,
    "shrieking-shack": BUILT, "village-house": BUILT, "village-wing": BUILT, "street-furniture": None,
}
FINE = 4                       # raster cells per texel along each axis
TILE = 256 * FINE              # raster cells per side of the tiles the distances are found in
OPEN_M = 12.7                  # the encoding's largest distance: open ground
TREES_FROM_M, TREES_FULL_M = 0.5, 2.5    # frag1.glsl vegMask: trees fade in between these distances from a wall


def to_fine(grid, x, z):
    """World (east, north) -> fractional raster (row, column); raster cell r spans [r, r + 1), and texel i's cell is
    raster cells FINE * i to FINE * i + FINE - 1."""
    i, j = grid.to_texel(x, z)
    return (i + 0.5) * FINE, (j + 0.5) * FINE


def cell_m(grid):
    """The raster cell's side in metres (the grid's texels are square)."""
    return float(np.mean(grid.texel_size())) / FINE


def _fill(mask, r0, c0, rows, cols):
    """`mask` (origin r0, c0) with the cells marked that the polygons cover, the polygons' vertices given as raster
    (rows, cols), each (k, n): PIL fills the cells whose centres lie inside and draws each edge."""
    img = Image.new("1", (mask.shape[1], mask.shape[0]), 0)
    draw = ImageDraw.Draw(img)
    xs, ys = cols - c0 - 0.5, rows - r0 - 0.5       # PIL puts a cell's centre at its integer coordinates
    for k in range(len(xs)):
        draw.polygon(list(zip(xs[k].tolist(), ys[k].tolist())), fill=1, outline=1)
    return mask | np.asarray(img, bool)


def mark(tris, grid, enclosure=False):
    """(row, column, mask): the raster cells that the world triangles (t, 3, 3) pass over seen from above, as a boolean
    image whose first cell is (row, column); with `enclosure`, every cell of the triangles' convex hull."""
    rows, cols = to_fine(grid, tris[..., 0], tris[..., 2])
    r0, c0 = int(np.floor(rows.min())) - 2, int(np.floor(cols.min())) - 2
    shape = (int(np.floor(rows.max())) - r0 + 3, int(np.floor(cols.max())) - c0 + 3)
    mask = _fill(np.zeros(shape, bool), r0, c0, rows, cols)
    if enclosure:
        pts = np.stack([rows.ravel(), cols.ravel()], 1)
        ring = pts[ConvexHull(pts).vertices]
        mask = _fill(mask, r0, c0, ring[None, :, 0], ring[None, :, 1])
    return r0, c0, mask


def silhouettes(triangles, grid):
    """{asset: (row, column, mask)} of every asset that POLICY marks, from its world triangles {asset: (t, 3, 3)} as the
    payload holds them (what quantization collapses is drawn nowhere, so it marks nothing)."""
    out = {}
    for asset in sorted(triangles):
        kind = POLICY[asset]
        if kind is not None and len(triangles[asset]):
            out[asset] = mark(triangles[asset], grid, enclosure=kind == ENCLOSURE)
    return out


def pack(silhouette):
    r0, c0, mask = silhouette
    return {"origin": [int(r0), int(c0)], "shape": [int(n) for n in mask.shape],
            "bits": base64.b64encode(zlib.compress(np.packbits(mask).tobytes(), 9)).decode("ascii")}


def unpack(packed):
    (r0, c0), (h, w) = packed["origin"], packed["shape"]
    bits = np.frombuffer(zlib.decompress(base64.b64decode(packed["bits"])), np.uint8)
    return r0, c0, np.unpackbits(bits)[:h * w].reshape(h, w).astype(bool)


def _union(sils, grid):
    """(row, column, mask): every silhouette on one raster, clipped to the grid and padded so that distances out to
    OPEN_M are exact; None when nothing marked lies on the grid (bldSd reads open ground off the local layer anyway)."""
    size = (FINE * grid.ny, FINE * grid.nx)
    on = {a: s for a, s in sils.items()
          if s[0] < size[0] and s[1] < size[1] and s[0] + s[2].shape[0] > 0 and s[1] + s[2].shape[1] > 0}
    if not on:
        return None
    pad = _pad(grid)
    lo = np.min([(r0, c0) for r0, c0, _ in on.values()], axis=0) - pad
    hi = np.max([(r0 + m.shape[0], c0 + m.shape[1]) for r0, c0, m in on.values()], axis=0) + pad
    lo = np.maximum(lo // FINE * FINE, 0)                                   # whole texels, inside the grid
    hi = np.minimum(-(-hi // FINE) * FINE, size)
    marked = np.zeros(tuple(hi - lo), bool)
    for r0, c0, m in on.values():
        rows, cols = slice(max(r0 - lo[0], 0), min(r0 + m.shape[0] - lo[0], marked.shape[0])), \
            slice(max(c0 - lo[1], 0), min(c0 + m.shape[1] - lo[1], marked.shape[1]))
        marked[rows, cols] |= m[rows.start + lo[0] - r0:rows.stop + lo[0] - r0, cols.start + lo[1] - c0:cols.stop + lo[1] - c0]
    return int(lo[0]), int(lo[1]), marked


def _pad(grid):
    """Raster cells, whole texels, past OPEN_M from a texel's cell: within this every distance the layer stores is exact."""
    return FINE * (int(np.ceil(OPEN_M / (FINE * cell_m(grid)))) + 2)


def layer(sils, grid):
    """The footprint layer: (grid.ny, grid.nx) bytes of the least signed distance over each texel's cell to the marked
    raster cells (negative inside), floored to the encoding's 0.1 m steps. The distances are found a tile at a time,
    each tile with _pad cells round it, which bounds the memory; no stored distance reaches past the pad."""
    out = np.full((grid.ny, grid.nx), 128 + int(round(10 * OPEN_M)), np.uint8)
    union = _union(sils, grid)
    if union is None:
        return out
    r0, c0, marked = union
    cell, pad, (h, w) = cell_m(grid), _pad(grid), marked.shape
    for a in range(0, h, TILE):
        for b in range(0, w, TILE):
            ra, rb, ca, cb = max(a - pad, 0), min(a + TILE + pad, h), max(b - pad, 0), min(b + TILE + pad, w)
            if not marked[ra:rb, ca:cb].any():
                continue
            m = np.pad(marked[ra:rb, ca:cb], 1)          # an open ring, so every distance inside is finite
            sd = np.where(m, 0.5 - ndi.distance_transform_edt(m), ndi.distance_transform_edt(~m) - 0.5) * cell
            core = sd[1 + a - ra:1 + min(a + TILE, h) - ra, 1 + b - ca:1 + min(b + TILE, w) - ca]
            least = core.reshape(core.shape[0] // FINE, FINE, core.shape[1] // FINE, FINE).min(axis=(1, 3))
            i0, j0 = (r0 + a) // FINE, (c0 + b) // FINE
            out[i0:i0 + least.shape[0], j0:j0 + least.shape[1]] = np.clip(np.floor(128 + 10 * least), 0, 255)
    return out


def encode(footprint):
    """The layer as a payload entry, in the data's own u8 encoding (mesh_source.encode_u8, delta-coded rows)."""
    return encode_u8(footprint[..., None], [1])


def decoded(entry):
    return decode(entry)[..., 0]


def vegetation(sd):
    """frag1.glsl vegMask's fade with the distance from a wall: 0 up to TREES_FROM_M, 1 from TREES_FULL_M."""
    t = np.clip((np.asarray(sd, np.float64) - TREES_FROM_M) / (TREES_FULL_M - TREES_FROM_M), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def cleared(sils, grid, footprint, data_sd, trees):
    """{asset: m2 of tree cover} that the layer removes beside each asset, measured against the data's own building
    distances `data_sd` with the data's tree density `trees` (both texel images): each texel's loss goes to the asset
    whose silhouette's edge lies nearest its centre."""
    sd = (footprint.astype(np.float64) - 128.0) / 10.0
    loss = trees * np.maximum(vegetation(data_sd) - vegetation(sd), 0.0)
    i, j = np.nonzero(loss > 0)
    names = sorted(sils)
    if not len(i) or not names:
        return {}
    edges, owners = [], []
    for k, asset in enumerate(names):
        r0, c0, m = sils[asset]
        edge = np.argwhere(m & ~ndi.binary_erosion(m)) + (r0, c0)
        edges.append(edge + 0.5)
        owners.append(np.full(len(edge), k))
    _, nearest = cKDTree(np.concatenate(edges)).query(np.stack([FINE * i + FINE / 2, FINE * j + FINE / 2], 1))
    area = float(np.prod(grid.texel_size()))
    out = {}
    for k, w in zip(np.concatenate(owners)[nearest], loss[i, j]):
        out[names[k]] = out.get(names[k], 0.0) + float(w) * area
    return dict(sorted(out.items()))
