"""Read the scene layers embedded in build/data.js exactly as the page decodes them (build-time only)."""
import base64
import json
import os
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_JS = os.path.join(ROOT, "build", "data.js")
GEOREF_JSON = os.path.join(ROOT, "georef.json")

# packed u8 channels, in the order build_final.py / pack.py write them
CHANNELS = {
    "locA": ("road", "trees", "hedge", "bsd"),
    "locB": ("wall", "stream", "far"),
    "locN": ("surface", "bcode"),
    "casA": ("low_wall", "wall", "range", "block"),
    "casB": ("great_hall", "keep", "tower1", "tower2"),
}
BUILDING_KINDS = ("thatch", "slate", "glass", "hut", "shack", "stands", "big")
# essay map image sizes (px) for the georef rectangles; the build prefers the Hogsmeade map where both exist
MAP_SIZES = (("hogs", "hogs_rect", 1735, 2261), ("grounds", "grounds_rect", 2000, 2348))


def load_data(path=DATA_JS):
    with open(path, encoding="ascii") as f:
        s = f.read()
    return json.loads(s[s.index("{"): s.rindex("}") + 1])


def load_georef():
    with open(GEOREF_JSON, encoding="utf-8") as f:
        return json.load(f)


def _raw(e):
    return np.frombuffer(zlib.decompress(base64.b64decode(e["data"])), np.uint8)


def decode(e):
    """A packed layer as rows x columns (x channels); u8 layers stay bytes, h16 layers become metres."""
    w, h = e["w"], e["h"]
    if e["enc"] == "u8":
        planes = _raw(e).reshape(e["ch"], h, w)
        out = np.empty((h, w, e["ch"]), np.uint8)
        for c, d in enumerate(e["delta"]):
            out[..., c] = (np.cumsum(planes[c], axis=1, dtype=np.uint32) & 255) if d else planes[c]
        return out
    if e["enc"] == "h16":
        raw = _raw(e).reshape(2, h, w).astype(np.int64)
        z = raw[0] | (raw[1] << 8)
        r = np.where(z & 1, -((z + 1) >> 1), z >> 1)
        return r.cumsum(0).cumsum(1) * e["q"]
    raise ValueError(f"undecodable layer encoding {e['enc']!r}")


def encode_height(h, q):
    """An h16 layer as pack.py's enc_height writes it: planar-predicted residuals, zigzag, lo/hi byte planes, zlib."""
    v = np.round(np.asarray(h, np.float64) / q).astype(np.int64)
    pred = np.zeros_like(v)
    pred[1:, 1:] = v[:-1, 1:] + v[1:, :-1] - v[:-1, :-1]
    pred[0, 1:] = v[0, :-1]
    pred[1:, 0] = v[:-1, 0]
    r = v - pred
    z = np.where(r >= 0, 2 * r, -2 * r - 1)
    if z.max() >= 65536:
        raise ValueError("height residuals overflow 16 bits")
    z = z.astype(np.uint16)
    raw = (z & 255).astype(np.uint8).tobytes() + (z >> 8).astype(np.uint8).tobytes()
    return {"enc": "h16", "w": int(h.shape[1]), "h": int(h.shape[0]), "q": q,
            "data": base64.b64encode(zlib.compress(raw, 9)).decode("ascii")}


def encode_u8(a, delta):
    """A u8 layer as pack.py's enc_u8 writes it with explicit per-channel horizontal-delta flags."""
    planes = []
    for c, use in enumerate(delta):
        p = a[..., c].astype(np.uint8)
        if use:
            d = p.astype(np.int16)
            d[:, 1:] = p[:, 1:].astype(np.int16) - p[:, :-1].astype(np.int16)
            p = (d & 255).astype(np.uint8)
        planes.append(p)
    return {"enc": "u8", "w": int(a.shape[1]), "h": int(a.shape[0]), "ch": int(a.shape[2]), "delta": list(delta),
            "data": base64.b64encode(zlib.compress(b"".join(p.tobytes() for p in planes), 9)).decode("ascii")}


def sd_bytes(b, offset=128.0):
    """Signed distance in metres from a u8 channel (texel byte / 10 - 12.8, as the shaders read it)."""
    return (b.astype(np.float32) - offset) / 10.0


class Grid:
    """A map-aligned texel grid placed in world (east, north) metres by its affine (x, z, 1) -> (u, v)."""

    def __init__(self, g):
        self.nx, self.ny = g["nx"], g["ny"]
        self.aff = np.array(g["aff"], np.float64)
        self.inv = np.linalg.inv(self.aff[:, :2])

    def to_world(self, i, j):
        """Texel row i, column j (fractional allowed; centres at integer values) -> (east, north)."""
        u = (np.asarray(j, np.float64) + 0.5) / self.nx - self.aff[0, 2]
        v = (np.asarray(i, np.float64) + 0.5) / self.ny - self.aff[1, 2]
        return self.inv[0, 0] * u + self.inv[0, 1] * v, self.inv[1, 0] * u + self.inv[1, 1] * v

    def to_texel(self, x, z):
        """(east, north) -> fractional (row, column) with texel centres at integer values."""
        x, z = np.asarray(x, np.float64), np.asarray(z, np.float64)
        u = self.aff[0, 0] * x + self.aff[0, 1] * z + self.aff[0, 2]
        v = self.aff[1, 0] * x + self.aff[1, 1] * z + self.aff[1, 2]
        return v * self.ny - 0.5, u * self.nx - 0.5

    def texel_size(self):
        return 1.0 / np.hypot(*self.aff[0, :2]) / self.nx, 1.0 / np.hypot(*self.aff[1, :2]) / self.ny


def map_region(x, z, georef=None):
    """The essay map ('hogs' or 'grounds') whose ink supplied world (east, north), or 'outside'."""
    g = georef or load_georef()
    rot = np.deg2rad(g["rot_deg"])
    u = g["U0"] + (x * np.cos(rot) + z * np.sin(rot)) / g["mFF"]
    v = g["V0"] + (x * np.sin(rot) - z * np.cos(rot)) / g["mFF"]
    for name, key, w, h in MAP_SIZES:
        x0, x1, y0, y1 = g[key]
        px, py = (u - x0) * w / (x1 - x0), (v - y0) * h / (y1 - y0)
        if 0 <= px < w - 1 and 0 <= py < h - 1:
            return name
    return "outside"


def sample(field, grid, x, z):
    """Bilinear samples of a texel-centre field at world (east, north), clamped at the edges like the shaders."""
    from scipy import ndimage as ndi

    i, j = grid.to_texel(x, z)
    return ndi.map_coordinates(np.asarray(field, np.float64), [np.ravel(i), np.ravel(j)], order=1,
                               mode="nearest").reshape(np.shape(i))


# texels outside `keep` sit at least this far above the contour level, so a neighbour touching diagonally
# cannot join the polygon across the shared corner
KEEP_MARGIN = 0.5


def field_polygons(field, grid, level, keep=None, origin=(0, 0)):
    """World polygons (holes kept) where a texel-centre field lies below `level`, traced as bilinear sampling
    draws it; `keep` limits the inside to selected texels so touching neighbours stay separate, and `origin` is
    the (row, column) of a cropped field within the grid."""
    from contourpy import FillType, contour_generator
    from shapely import make_valid
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient

    f = np.asarray(field, np.float64)
    if keep is not None:
        f = np.where(keep, f, np.maximum(f, level + KEEP_MARGIN))
    f = np.pad(f, 1, constant_values=max(float(f.max()), level) + 1.0)   # close rings that reach the grid edge
    points, offsets = contour_generator(z=f, fill_type=FillType.OuterOffset).filled(float(f.min()) - 1.0, level)
    polygons = []
    for pts, offs in zip(points, offsets):
        rings = [pts[a:b] for a, b in zip(offs[:-1], offs[1:])]
        world = [np.column_stack(grid.to_world(r[:, 1] - 1.0 + origin[0], r[:, 0] - 1.0 + origin[1])) for r in rings]
        polygon = Polygon(world[0], world[1:])
        # a saddle can pinch a ring to a point; splitting it there keeps each part a valid polygon
        parts = [polygon] if polygon.is_valid else getattr(make_valid(polygon), "geoms", [make_valid(polygon)])
        polygons += [orient(p, 1.0) for p in parts if p.geom_type == "Polygon" and p.area > 0]
    return polygons


EIGHT = np.ones((3, 3), bool)
# casB's last two channels are nearest-tower indices, not masks
CASTLE_CLASSES = CHANNELS["casA"] + CHANNELS["casB"][:2]
# one- and two-texel specks are resampling noise; the smallest real piece is a four-texel wall stub
CASTLE_SPECK_M2 = 0.75
HULL_CLOSING_ITERATIONS = 8


def _centre(grid, ii, jj):
    return tuple(round(float(c), 2) for c in grid.to_world(ii.mean(), jj.mean()))


def building_components(data=None):
    """Mapped building footprints: 4-connected texels with bsd < 0, numbered in scipy label order.

    `touching` lists the footprints that meet this one, including diagonally."""
    return building_labels(data)[1]


def building_labels(data=None):
    """(label image on the local grid, building_components list); label k is component id k."""
    from scipy import ndimage as ndi

    data = data or load_data()
    tex, meta = data["tex"], data["meta"]
    bsd = sd_bytes(decode(tex["locA"])[..., CHANNELS["locA"].index("bsd")])
    bcode = decode(tex["locN"])[..., CHANNELS["locN"].index("bcode")]
    grid = Grid(meta["local"])
    texel_area = float(np.prod(grid.texel_size()))
    georef = load_georef()
    labels, _ = ndi.label(bsd < 0)
    out = []
    for k, sl in enumerate(ndi.find_objects(labels), 1):
        ii, jj = np.nonzero(labels[sl] == k)
        ii, jj = ii + sl[0].start, jj + sl[1].start
        x, z = _centre(grid, ii, jj)
        code = int(np.bincount(bcode[ii, jj]).argmax())
        near = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
        ring = labels[near][ndi.binary_dilation(labels[near] == k, EIGHT)]
        out.append({
            "id": k, "x": x, "z": z, "area": round(len(ii) * texel_area, 1),
            "kind": BUILDING_KINDS[code >> 4], "eave": 2.5 + 0.5 * (code & 15),
            "region": map_region(x, z, georef),
            "rows": (int(ii.min()), int(ii.max())), "cols": (int(jj.min()), int(jj.max())),
            "touching": sorted(int(t) for t in set(np.unique(ring)) - {0, k}),
        })
    return labels, out


def castle_hull(masses):
    """The main castle body's outline with its courts filled: the largest 8-connected piece of the
    non-low classes, closed across court-wide gaps and hole-filled."""
    from scipy import ndimage as ndi

    labels, _ = ndi.label(masses[..., 1:].any(-1), EIGHT)
    main = labels == np.bincount(labels.ravel())[1:].argmax() + 1
    return ndi.binary_fill_holes(ndi.binary_closing(main, EIGHT, iterations=HULL_CLOSING_ITERATIONS))


def castle_pieces(data=None):
    """Castle-grid mass pieces by role: 'main' body, 'court' pieces inside its hull, 'detached' pieces wholly
    outside it, and 'outside' remnants of hull-touching pieces (cliff walls and wall slivers)."""
    return castle_piece_labels(data)[1]


def castle_piece_labels(data=None):
    """(piece-id image on the castle grid, castle_pieces list); speck texels stay 0."""
    from scipy import ndimage as ndi

    data = data or load_data()
    meta = data["meta"]
    masses = castle_masses(data)
    grid = Grid(meta["castle"])
    texel_area = float(np.prod(grid.texel_size()))
    hull = castle_hull(masses)
    labels, _ = ndi.label(masses.any(-1), EIGHT)
    main = np.bincount(labels[hull].ravel())[1:].argmax() + 1
    pieces, outside = [], np.zeros_like(hull)
    painted = np.zeros(labels.shape, np.int32)

    def add(role, mask, sl):
        ii, jj = np.nonzero(mask)
        if len(ii) * texel_area < CASTLE_SPECK_M2:
            return
        ii, jj = ii + sl[0].start, jj + sl[1].start
        x, z = _centre(grid, ii, jj)
        present = masses[ii, jj].any(0)
        pieces.append({"id": len(pieces) + 1, "x": x, "z": z, "area": round(len(ii) * texel_area, 1),
                       "role": role, "classes": [c for c, p in zip(CASTLE_CLASSES, present) if p]})
        painted[ii, jj] = len(pieces)

    for k, sl in enumerate(ndi.find_objects(labels), 1):
        mask = labels[sl] == k
        inside = hull[sl] & mask
        if not inside.any():
            add("detached", mask, sl)
            continue
        add("main" if k == main else "court", inside, sl)
        outside[sl] |= mask & ~inside
    out_labels, _ = ndi.label(outside, EIGHT)
    for k, sl in enumerate(ndi.find_objects(out_labels), 1):
        add("outside", out_labels[sl] == k, sl)
    return painted, pieces


def castle_masses(data):
    """Per-texel membership of the six castle mass classes (castle grid rows x columns x CASTLE_CLASSES)."""
    return castle_class_bytes(data) < 128


def castle_class_sd(data):
    """Signed distance (m) to each castle mass class as the shaders read it: q8, grown 10 cm (byte / 10 - 12.9)."""
    return castle_class_bytes(data).astype(np.float64) / 10.0 - 12.9


def castle_class_bytes(data):
    tex = data["tex"]
    return np.concatenate([decode(tex["casA"]), decode(tex["casB"])[..., :2]], axis=-1)
