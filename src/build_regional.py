"""Regional LOD layers in a north-up frame centred on the castle (x = east, y = north, metres):
   M: +-5 km @ 10 m, A: +-12 km @ 30 m, B: +-36 km @ 120 m.
   Each height level stores bed height and water-surface height (-1000 = no water).
   Rows are stored south -> north (row 0 = minimum y).
   Also the Sentinel-2 cloudless colour of each level (S_*) and the plantation density of the M and A levels (SF_*)."""
from common import *
from PIL import ImageDraw
from scipy.spatial import cKDTree
import math

LEVELS = {"M": (5000.0, 10.0), "A": (12000.0, 30.0), "B": (36000.0, 120.0)}
NOWATER = -1000.0
RIVER_M = 12.0              # a river's width where OpenStreetMap gives none
RIVER_SINK_M = 0.55         # its water stands this far below the valley floor along it,
RIVER_GRADE, RIVER_DEEP_M = 0.1, 1.05   # its bed falls from its edges at this slope to this deep, and its banks rise
                                        # so across the texel beyond: a bilinear read of the water's depth, as the
                                        # renderer's, then crosses nought where the water's edge is
RIVER_BLEND_M = 30.0        # beyond that texel its banks give way to the ground as mapped over this far
RIVER_MIN_TEXELS = 1.2      # no river is drawn narrower than this many texels: a bilinear read of the depth found no
                            # water wherever a narrower one's line ran between the texels' centres


def grid(R, st):
    n = int(round(2 * R / st)) + 1
    c = -R + st * np.arange(n)                    # texel centres
    X, Y = np.meshgrid(c, c)                        # row 0 = south
    return X, Y, n


def assemble_rings(ways):
    """Stitch open way segments (lists of (lon,lat)) into closed rings."""
    segs = [list(w) for w in ways if len(w) >= 2]
    rings = []
    while segs:
        cur = segs.pop(0)
        changed = True
        while cur[0] != cur[-1] and changed:
            changed = False
            for i, s in enumerate(segs):
                if s[0] == cur[-1]: cur += s[1:]; segs.pop(i); changed = True; break
                if s[-1] == cur[-1]: cur += s[::-1][1:]; segs.pop(i); changed = True; break
                if s[-1] == cur[0]: cur = s[:-1] + cur; segs.pop(i); changed = True; break
                if s[0] == cur[0]: cur = s[::-1][:-1] + cur; segs.pop(i); changed = True; break
        if len(cur) >= 4: rings.append(cur)
    return rings


def water_raster(R=36000.0, st=10.0):
    """Rasterise OSM water polygons into a 10 m north-up grid (row 0 = NORTH for PIL), returns bool array south-up."""
    d = json.load(open(P("osm_water_wide.json")))
    n = int(round(2 * R / st))
    img = Image.new("L", (n, n), 0); dr = ImageDraw.Draw(img)

    def topx(coords):
        lon = np.array([c[0] for c in coords]); lat = np.array([c[1] for c in coords])
        e, nn = to_tm.transform(lon, lat)
        x = (e - CASTLE_E + R) / st; y = (R - (nn - CASTLE_N)) / st
        return list(zip(x, y))

    outers, inners = [], []
    for el in d["elements"]:
        t = el.get("tags", {})
        if el["type"] == "way" and (t.get("natural") == "water" or t.get("waterway") == "riverbank"):
            g = el.get("geometry") or []
            co = [(p["lon"], p["lat"]) for p in g]
            if len(co) >= 4 and co[0] == co[-1]: outers.append(co)
        elif el["type"] == "relation" and t.get("natural") == "water":
            ow, iw = [], []
            for m in el.get("members", []):
                if m.get("type") != "way" or not m.get("geometry"): continue
                co = [(p["lon"], p["lat"]) for p in m["geometry"]]
                (iw if m.get("role") == "inner" else ow).append(co)
            outers += assemble_rings(ow); inners += assemble_rings(iw)
    for r in outers: dr.polygon(topx(r), fill=255)
    for r in inners: dr.polygon(topx(r), fill=0)
    a = np.asarray(img) > 127
    print("water polygons:", len(outers), "outer,", len(inners), "inner; water fraction", a.mean())
    return a[::-1].copy()          # south-up


def osm_rivers(osm=None, anchor=(CASTLE_E, CASTLE_N)):
    """OpenStreetMap's rivers (the build's water extract unless `osm` is given): (points (k, 2) relative to `anchor`,
    half width), RIVER_M wide where it gives no width."""
    d = osm if osm is not None else json.load(open(P("osm_water_wide.json")))
    out = []
    for el in d["elements"]:
        t = el.get("tags", {})
        if el["type"] != "way" or t.get("waterway") != "river": continue
        g = el.get("geometry") or []
        e, nn = to_tm.transform([p["lon"] for p in g], [p["lat"] for p in g])
        w = float(t.get("width", "12").split()[0]) if t.get("width", "").replace('.', '', 1).isdigit() else RIVER_M
        out.append((np.column_stack([np.array(e) - anchor[0], np.array(nn) - anchor[1]]), w / 2))
    return out


def carve_rivers(bed, surf, R, st, rivers, wet, ground):
    """A level's `bed` and water surface `surf` (its texel centres st apart from -R, rows running north) with `rivers`
    carved into them, but not where it already has water (`wet`). Each river's water stands RIVER_SINK_M below the
    valley floor along its line: the lowest ground across the line, out a texel beyond its half width either side, as
    the elevation model has it (`ground`), before the level's water bodies are levelled. A water body's bed, levelled
    below its water, had sunk the ground beside a loch into pits; its water, one level for the whole body, stands above
    the ground at the lower end of a river's stretch mapped as one and below it at the upper; and the ground smoothed
    over a texel stood metres above a narrow valley's bottom. The river is level across, out to its half width but at
    least RIVER_MIN_TEXELS across; its bed falls from there at RIVER_GRADE to RIVER_DEEP_M, and its banks rise so across
    the texel beyond, then give way to the ground as mapped over RIVER_BLEND_M. Where a water body's level, spread over
    the shore beside it, stands below the river's water, the river's stands: below its bed, that level drew the river
    dry beside the upper end of a stretch mapped as water. Measured from the line itself, its water's edge runs smooth
    along it: carved round a raster of the line, a texel wide, it ran in a trench stepped wherever the raster stepped."""
    if not rivers:
        return bed, surf
    pts, half, across = [], [], []
    for xy, hw in rivers:
        step = np.diff(xy, axis=0)
        for start, d, m in zip(xy[:-1], step, np.maximum(np.ceil(np.hypot(*step.T)), 1).astype(int)):
            pts.append(start + (np.arange(m) / m)[:, None] * d)          # points 1 m apart or less
            half.append(np.full(m, hw))
            across.append(np.tile(np.array([-d[1], d[0]]) / max(float(np.hypot(*d)), 1e-9), (m, 1)))
        pts.append(xy[-1:])
        half.append([hw])
        across.append(across[-1][-1:] if len(xy) > 1 else np.zeros((1, 2)))
    pts, half, across = (np.vstack(pts), np.maximum(np.concatenate(half), RIVER_MIN_TEXELS * st / 2.0),
                         np.vstack(across))
    at = pts + np.linspace(-1.0, 1.0, 17)[:, None, None] * ((half + st)[:, None] * across)    # across each point
    low = map_coordinates(ground, [(at[..., 1] + R) / st, (at[..., 0] + R) / st], order=1, mode="nearest").min(axis=0)
    water = low - RIVER_SINK_M
    X, Y, _ = grid(R, st)
    reach = float(half.max()) + st + RIVER_BLEND_M
    d, k = cKDTree(pts).query(np.column_stack([X.ravel(), Y.ravel()]), distance_upper_bound=reach)
    near = (d < reach).reshape(bed.shape) & ~wet
    k = np.minimum(k, len(pts) - 1).reshape(bed.shape)
    beyond = np.where(near, d.reshape(bed.shape) - half[k], 0.0)         # metres out from the water's edge
    level = water[k]
    banks = level + np.clip(RIVER_GRADE * beyond, -RIVER_DEEP_M, RIVER_GRADE * st)
    t = np.clip((beyond - st) / RIVER_BLEND_M, 0.0, 1.0)
    carved = banks + (bed - banks) * t * t * (3.0 - 2.0 * t)
    return (np.where(near, carved, bed).astype(bed.dtype),
            np.where(near & (beyond <= st), np.maximum(surf, level), surf).astype(surf.dtype))


def main():
    mos = tm_dem()
    water10 = water_raster()                       # south-up, 7200x7200, cell centres at -R+5+10*i
    np.save(P("build", "water10.npy"), water10)
    lab, nl = ndi.label(water10)
    print("loch components", nl)
    # loch levels from the DEM (median over eroded interior)
    Rw, stw = 36000.0, 10.0
    nw = water10.shape[0]
    cw = -Rw + stw * (np.arange(nw) + 0.5)
    level = np.full(nl + 1, np.nan, np.float32)
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs):
        if sl is None: continue
        m = lab[sl] == i + 1
        mi = ndi.binary_erosion(m, iterations=2)
        if mi.sum() < 4: mi = m
        yy, xx = np.nonzero(mi)
        if len(yy) > 4000:
            k = np.random.default_rng(i).choice(len(yy), 4000, replace=False); yy, xx = yy[k], xx[k]
        X = cw[xx + sl[1].start]; Y = cw[yy + sl[0].start]
        h = dem_sample(mos, X + CASTLE_E, Y + CASTLE_N)
        level[i + 1] = float(np.median(h))
    np.save(P("build", "loch_level.npy"), level)
    dist_in = ndi.distance_transform_edt(water10) * stw            # distance to shore inside water (m)
    lvl10 = np.where(water10, level[lab], np.nan)

    for name, (R, st) in LEVELS.items():
        X, Y, n = grid(R, st)
        if name == "B":
            src = ndi.gaussian_filter(mos, (st / 31.0 * 0.5, st / 25.0 * 0.5))
            bed = dem_sample(src, X + CASTLE_E, Y + CASTLE_N, order=1)
        else:
            bed = dem_sample(mos, X + CASTLE_E, Y + CASTLE_N, order=3 if name == "M" else 1)
        bed = bed.astype(np.float32)
        ground = bed                                # the elevation model's, before the water bodies are levelled
        # sample water info from the 10 m raster (box-filtered to this level's cell size)
        k = max(1, int(round(st / stw)))
        ix = np.clip(((X + Rw) / stw).astype(int), 0, nw - 1); iy = np.clip(((Y + Rw) / stw).astype(int), 0, nw - 1)
        if k > 1:
            frac = ndi.uniform_filter(water10.astype(np.float32), size=k)[iy, ix]
            dmean = ndi.uniform_filter(dist_in, size=k)[iy, ix]
            lab_c = ndi.maximum_filter(lab, size=k)[iy, ix]
        else:
            frac = water10[iy, ix].astype(np.float32); dmean = dist_in[iy, ix]; lab_c = lab[iy, ix]
        is_w = frac >= 0.5
        lv = np.where(lab_c > 0, level[lab_c], np.nan)
        depth = np.minimum(35.0, 0.6 + 0.12 * dmean + 0.25 * st)
        bed = np.where(is_w & np.isfinite(lv), lv - depth, bed)
        # sea / tidal water: very low ground
        sea = bed < 0.8
        surf = np.full(bed.shape, NOWATER, np.float32)
        surf[is_w & np.isfinite(lv)] = lv[is_w & np.isfinite(lv)]
        surf[sea] = 0.0
        # extend the flat level 2 texels outward over ground that is at/above it, so bilinear shores stay flat
        has = surf > NOWATER + 1
        for _ in range(2):
            idx = ndi.distance_transform_edt(~has, return_distances=False, return_indices=True)
            near = surf[idx[0], idx[1]]
            grow = ndi.binary_dilation(has) & ~has & (bed >= near - 0.05)
            surf = np.where(grow, near, surf); has = has | grow
        if name == "M":
            bed, surf = carve_rivers(bed, surf, R, st, osm_rivers(), is_w, ground)
        np.save(P("build", f"R_{name}_bed.npy"), bed.astype(np.float32))
        np.save(P("build", f"R_{name}_surf.npy"), surf.astype(np.float32))
        print(f"level {name}: {n}x{n} @ {st} m, bed {bed.min():.1f}..{bed.max():.1f}, water texels {(surf > bed).mean() * 100:.2f}%")

    # Sentinel-2 cloudless colour for each level (2x2 supersampled)
    Image.MAX_IMAGE_PIXELS = None
    meta = json.load(open(P("s2tiles", "mosaic.json")))
    Ms = np.asarray(Image.open(P("s2tiles", "mosaic.png"))).astype(np.float32)
    zn = 2 ** meta["z"]
    for name, (R, st) in LEVELS.items():
        if name == "B":
            continue
        X, Y, n = grid(R, st)
        acc = 0
        for dx in (-st / 4, st / 4):
            for dy in (-st / 4, st / 4):
                lon, lat = to_ll.transform(X + CASTLE_E + dx, Y + CASTLE_N + dy)
                xt = (np.asarray(lon) + 180) / 360 * zn; yt = (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * zn
                px = (xt - meta["x0"]) * 256 - 0.5; py = (yt - meta["y0"]) * 256 - 0.5
                acc = acc + np.stack([map_coordinates(Ms[..., c], [py, px], order=1, mode='nearest') for c in range(3)], -1)
        np.save(P("build", f"S_{name}.npy"), np.clip(acc / 4, 0, 255).astype(np.uint8))
        print("satellite", name, n)
    np.save(P("build", "S_B.npy"), satellite_B((CASTLE_E, CASTLE_N), P("s2tiles")))
    for name in PLANTATIONS:
        layers = [np.load(P("build", f"{kind}.npy")) for kind in (f"S_{name}", f"R_{name}_bed", f"R_{name}_surf")]
        np.save(P("build", f"SF_{name}.npy"), plantations(name, *layers))


def satellite_B(anchor, tiles, z=11, R=36500.0):
    """The 120 m layer's Sentinel-2 cloudless colour round the anchor, 2 x 2 supersampled from the zoom-11 tiles in
    the folder `tiles`."""
    E0, N0 = anchor
    n = 2 ** z
    lons, lats = to_ll.transform([E0 - R, E0 + R, E0 - R, E0 + R], [N0 - R, N0 - R, N0 + R, N0 + R])
    xs = [(lon + 180) / 360 * n for lon in lons]
    ys = [(1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n for lat in lats]
    x0, x1, y0, y1 = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
    Mz = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256, 3), np.float32)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            tile = Image.open(os.path.join(tiles, f"{z}_{x}_{y}.jpg")).convert("RGB")
            Mz[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = np.asarray(tile)
    X, Y, _ = grid(*LEVELS["B"])
    acc = 0
    for dx in (-30, 30):
        for dy in (-30, 30):
            lon, lat = to_ll.transform(X + E0 + dx, Y + N0 + dy)
            xt = (np.asarray(lon) + 180) / 360 * n; yt = (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n
            px = (xt - x0) * 256 - 0.5; py = (yt - y0) * 256 - 0.5
            acc = acc + np.stack([map_coordinates(Mz[..., c], [py, px], order=1, mode='nearest') for c in range(3)], -1)
    return np.clip(acc / 4, 0, 255).astype(np.uint8)


PLANTATIONS = {"M": (2, 0.7), "A": (1, 0.6)}     # level: (texels kept dry beside water, smoothing sigma in texels)


def plantations(name, S, bed, surf):
    """A level's forest density from its Sentinel-2 colour S: dark green ground away from its water (surf above bed)."""
    dry_px, sigma = PLANTATIONS[name]
    S = S.astype(np.float32) / 255.0
    lum = 0.3 * S[..., 0] + 0.59 * S[..., 1] + 0.11 * S[..., 2]
    gr = S[..., 1] - S[..., 0]
    dens = np.clip((0.165 - lum) / 0.035, 0, 1) * np.clip((gr + 0.005) / 0.025, 0, 1)
    dens = np.where(ndi.binary_dilation(surf > bed, iterations=dry_px), 0, dens)
    dens = ndi.gaussian_filter(ndi.median_filter(dens, size=3), sigma)
    return np.clip(dens * 255, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    main()
