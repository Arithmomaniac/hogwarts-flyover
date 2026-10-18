"""Castle layer (0.5 m, map-aligned) from the essay's grounds-map castle + castle plan registration."""
from common import *
from features import local_grid, NX, NY

CX0, CY0, CW, CH = 1030, 560, 500, 620       # grounds-map px window holding the castle
CRES = 0.5                                     # metres per castle texel
PG = 0.72


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def local_thickness(mask, rmax):
    th = np.zeros(mask.shape, np.float32)
    for r in range(1, rmax + 1):
        op = ndi.binary_opening(mask, structure=disk(r))
        th[op] = r
        if not op.any(): break
    return th


def main():
    mass = np.load(P("build", "castle_mass_g.npy"))
    mass = ndi.binary_closing(mass, structure=disk(1))
    named = json.load(open(P("build", "castle_named.json")))
    # world -> castle-window native px
    def w2c(e, n):
        U, V = world_to_ff(e, n); u, v = ff_to_g(U, V); return u - CX0, v - CY0
    sd = np.where(mass, -ndi.distance_transform_edt(mass), ndi.distance_transform_edt(~mass)) * PG
    th = local_thickness(mass, 30) * PG                      # half-width (m) of the largest inscribed disk
    # --- tower detection: convex round knobs ---
    dt = ndi.distance_transform_edt(mass) * PG
    towers = []
    yy, xx = np.mgrid[0:mass.shape[0], 0:mass.shape[1]]
    peaks = (dt == ndi.maximum_filter(dt, size=9)) & (dt >= 3.7)
    for (py, px) in zip(*np.nonzero(peaks)):
        r = dt[py, px]
        rp = r / PG
        d2 = (xx - px) ** 2 + (yy - py) ** 2
        ann = (d2 > (rp + 2.0) ** 2) & (d2 < (rp + 6.0) ** 2)
        frac = mass[ann].mean() if ann.any() else 1
        # a tower must bulge clearly beyond the wall it sits on
        ring = (d2 > (rp + 3.0) ** 2) & (d2 < (rp + 9.0) ** 2) & mass
        wall_half = np.median(dt[ring]) if ring.any() else 0.0
        if frac < 0.36 and r <= 9.0 and (wall_half < 0.72 * r or frac < 0.12):
            towers.append((px, py, r, frac))
    # non-max suppression
    towers.sort(key=lambda t: -t[2]); kept = []
    for t in towers:
        if all((t[0] - k[0]) ** 2 + (t[1] - k[1]) ** 2 > (max(t[2], k[2]) / PG * 1.6) ** 2 for k in kept): kept.append(t)
    print("towers detected:", len(kept))
    # --- per-pixel eave height + roof type ---
    eave = np.zeros(mass.shape, np.float32); rtype = np.zeros(mass.shape, np.uint8)
    low = mass & (th < 1.6)
    wall = mass & (th >= 1.6) & (th < 3.3); rng = mass & (th >= 3.3) & (th < 7.0); blk = mass & (th >= 7.0)
    eave[low] = 2.2; rtype[low] = 4
    eave[wall] = 13.0; rtype[wall] = 0
    eave[rng] = 24.0; rtype[rng] = 1
    eave[blk] = 32.0; rtype[blk] = 1
    # Great Hall: the E-W range through the plan's GH point (steep roof, lower eaves)
    gx, gy = w2c(*named["GH"])
    print("GH in castle window px", gx, gy)
    hall = mass & (th >= 3.3) & (xx > gx - 34) & (xx < gx + 62) & (yy > gy - 32) & (yy < gy + 12)
    eave[hall] = 21.0; rtype[hall] = 3
    # Keep: square with four corner towers -> flat battlements
    kx, ky = w2c(*named["KEEP"])
    keep = mass & (((xx - kx) ** 2 + (yy - ky) ** 2) < (22 / PG) ** 2)
    eave[keep] = 27.0; rtype[keep] = 0
    rng_ = np.random.default_rng(7)
    special = {k: w2c(*named[k]) for k in ("AT", "NT", "GT", "WT", "RT")}
    tower_list = []
    for (px, py, r, frac) in kept:
        if any((px - sx) ** 2 + (py - sy) ** 2 < (12 / PG) ** 2 for sx, sy in special.values()):
            continue
        rp = r / PG + 0.8
        m = mass & (((xx - px) ** 2 + (yy - py) ** 2) <= rp ** 2)
        inkeep = (px - kx) ** 2 + (py - ky) ** 2 < (22 / PG) ** 2
        if inkeep:
            eave[m] = 33.0; rtype[m] = 0
        else:
            h = float(rng_.uniform(21, 29)); conical = rng_.uniform() < 0.72
            eave[m] = h; rtype[m] = 2 if conical else 0
            if not conical: eave[m] += 3
        tower_list.append((float(px), float(py), float(r)))
    # --- resample to 0.5 m texels ---
    nxc = int(round(CW * PG / CRES)); nyc = int(round(CH * PG / CRES))
    cu = (np.arange(nxc) + 0.5) * CRES / PG - 0.5; cv = (np.arange(nyc) + 0.5) * CRES / PG - 0.5
    CU, CV = np.meshgrid(cu, cv)
    sd_c = map_coordinates(sd.astype(np.float32), [CV, CU], order=1, mode='nearest')
    # spread heights/types outward so edge texels have valid values
    idx = ndi.distance_transform_edt(~mass, return_distances=False, return_indices=True)
    eave_s = eave[idx[0], idx[1]]; rtype_s = rtype[idx[0], idx[1]]
    ri = np.clip(np.round(CV).astype(int), 0, CH - 1); ci = np.clip(np.round(CU).astype(int), 0, CW - 1)
    eave_c = eave_s[ri, ci]; rtype_c = rtype_s[ri, ci]
    np.savez_compressed(P("build", "castle_layer.npz"), sd=sd_c.astype(np.float32), eave=eave_c.astype(np.float32), rtype=rtype_c.astype(np.uint8))
    meta = {"CX0": CX0, "CY0": CY0, "CW": CW, "CH": CH, "CRES": CRES, "nx": nxc, "ny": nyc,
            "towers_world": [[float(v) for v in ff_to_world(*g_to_ff(px + CX0, py + CY0))] + [r] for (px, py, r) in tower_list]}
    json.dump(meta, open(P("build", "castle_meta.json"), "w"), indent=1)
    print("castle texels", nxc, nyc)
    # castle zone in the local grid
    UU, VV = local_grid(); u, v = ff_to_g(UU, VV)
    zone = ndi.binary_dilation(mass, structure=disk(int(8 / PG)))
    zl = map_coordinates(zone.astype(np.float32), [v - CY0, u - CX0], order=0, mode='constant', cval=0) > 0.5
    np.save(P("build", "C_zone_local.npy"), zl)
    # preview
    col = np.zeros(mass.shape + (3,), np.float32) + 255
    pal = {0: [150, 150, 160], 1: [90, 90, 120], 2: [40, 60, 130], 3: [120, 60, 60], 4: [190, 190, 140]}
    for k, c in pal.items(): col[mass & (rtype == k)] = c
    col[mass] *= (0.5 + 0.5 * np.clip(eave[mass] / 34, 0, 1))[:, None]
    for (px, py, r) in tower_list:
        col[((xx - px) ** 2 + (yy - py) ** 2 < 4) ] = [255, 255, 0]
    for k, (sx, sy) in special.items():
        col[((xx - sx) ** 2 + (yy - sy) ** 2 < 25)] = [255, 0, 0]
    save_rgb(np.kron(col, np.ones((2, 2, 1))), "castle_layer.png")


if __name__ == "__main__":
    main()
