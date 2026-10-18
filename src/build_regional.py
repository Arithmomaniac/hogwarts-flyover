"""Regional LOD layers in a north-up frame centred on the castle (x = east, y = north, metres):
   M: +-5 km @ 10 m, A: +-12 km @ 30 m, B: +-36 km @ 120 m.
   Each height level stores bed height and water-surface height (-1000 = no water).
   Rows are stored south -> north (row 0 = minimum y)."""
from common import *
from PIL import ImageDraw
import math

LEVELS = {"M": (5000.0, 10.0), "A": (12000.0, 30.0), "B": (36000.0, 120.0)}
NOWATER = -1000.0


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


def river_raster(R, st):
    d = json.load(open(P("osm_water_wide.json")))
    n = int(round(2 * R / st)) + 1
    img = Image.new("L", (n, n), 0); dr = ImageDraw.Draw(img)
    for el in d["elements"]:
        t = el.get("tags", {})
        if el["type"] != "way" or t.get("waterway") != "river": continue
        g = el.get("geometry") or []
        e, nn = to_tm.transform([p["lon"] for p in g], [p["lat"] for p in g])
        x = (np.array(e) - CASTLE_E + R) / st + 0.0; y = (R - (np.array(nn) - CASTLE_N)) / st
        w = float(t.get("width", "12").split()[0]) if t.get("width", "").replace('.', '', 1).isdigit() else 12.0
        dr.line(list(zip(x, y)), fill=255, width=max(1, int(round(w / st))))
    return (np.asarray(img) > 127)[::-1].copy()


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
            riv = river_raster(R, st)
            riv &= ~is_w
            rd = ndi.distance_transform_edt(~riv) * st
            carve = np.clip(1 - rd / 12.0, 0, 1)
            smooth = ndi.gaussian_filter(bed, 1.0)
            bed = np.where(carve > 0, smooth - 1.6 * carve, bed)
            rs = smooth - 0.55
            wmask = (rd <= 2 * st) & (surf <= NOWATER + 1)
            surf = np.where(wmask, rs, surf)
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


if __name__ == "__main__":
    main()
