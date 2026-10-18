"""Terraform the local 2 m height field: real Copernicus DEM + the essay's landforms."""
from common import *
from features import local_grid, NX, NY, FX0, FY0, RES
from build_local import LAKE_LEVEL

L = LAKE_LEVEL


def h2w(u, v):
    U, V = h_to_ff(u, v); return ff_to_world(U, V)


def g2w(u, v):
    U, V = g_to_ff(u, v); return ff_to_world(U, V)


def mound(E, N, ce, cn, ra, rb, h, ang=0.0, power=2.0):
    """Smooth dome: height h at centre, 0 at the ellipse edge."""
    c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    x = (E - ce) * c - (N - cn) * s; y = (E - ce) * s + (N - cn) * c
    r = np.sqrt((x / ra) ** 2 + (y / rb) ** 2)
    t = np.clip(1 - r, 0, 1)
    return h * (t * t * (3 - 2 * t)) ** (1.0 / power * 2.0)


def main():
    UU, VV = local_grid()
    E, N = ff_to_world(UU, VV)
    D = np.load(P("build", "local_dem_2m.npy")).astype(np.float32)
    lake_sd = np.load(P("build", "L_lake_sd.npy")); water_sd = np.load(P("build", "L_water_sd.npy"))
    river = np.load(P("build", "L_river.npy"))
    lake_sd = np.where(lake_sd >= 199.0, 500.0, lake_sd)
    road = np.load(P("build", "L_road.npy")); bsd = np.load(P("build", "L_bsd.npy")); eave = np.load(P("build", "L_eave.npy"))
    F0 = ndi.gaussian_filter(D, 3.0)
    F = F0.copy()

    # ---- hillocks from the dashed outlines on the essay maps ----
    hills = []
    for nm, (u, v), ra, rb, h in [("4f", (975, 968), 58, 40, 5.0), ("4g", (883, 1022), 31, 20, 3.5), ("4f2", (878, 1120), 52, 26, 3.5),
                                  ("4h", (1135, 535), 24, 42, 4.0), ("4i howe", (1559, 212), 70, 66, 7.0)]:
        e, n = g2w(u, v); hills.append((nm, float(e), float(n), ra, rb, h))
    for nm, (u, v), ra, rb, h in [("shack base", (606, 1012), 118, 112, 9.0), ("shack top", (612, 962), 26, 22, 5.0),
                                  ("fold", (1037, 598), 190, 140, 6.0)]:
        e, n = h2w(u, v); hills.append((nm, float(e), float(n), ra, rb, h))
    for nm, e, n, ra, rb, h in hills:
        F += mound(E, N, e, n, ra, rb, h, ang=-25.1)
        print(f"hill {nm:12s} E{e:7.1f} N{n:7.1f} r({ra},{rb}) h{h}")

    # ---- castle platform: smooth base surface under and around the castle ----
    cz = np.load(P("build", "C_zone_local.npy"))          # castle mass dilated (bool) resampled to local grid
    B = ndi.gaussian_filter(F0, 12.0)
    B = np.maximum(B, L + 27.0)
    w = ndi.gaussian_filter(cz.astype(np.float32), 6.0)
    w = np.clip(w * 1.6, 0, 1)
    F = F * (1 - w) + B * w
    np.save(P("build", "T_castle_base.npy"), B.astype(np.float32))

    # ---- Quidditch pitch: flatten an ellipse (centre/axis measured from the stands ring) ----
    pc_e, pc_n, pa, pb, pbrg = -243.6, -485.9, 101.6, 54.6, 271.25
    t = np.deg2rad(90 - pbrg)
    x = (E - pc_e) * np.cos(t) + (N - pc_n) * np.sin(t); y = -(E - pc_e) * np.sin(t) + (N - pc_n) * np.cos(t)
    r = np.sqrt((x / (pa + 12)) ** 2 + (y / (pb + 12)) ** 2)
    inside = r < 1.0
    level = float(np.median(F[inside]))
    wp = np.clip((1.35 - r) / 0.35, 0, 1); wp = wp * wp * (3 - 2 * wp)
    F = F * (1 - wp) + level * wp
    print("pitch level", level)

    # ---- building pads (Hogsmeade, Hagrid's hut, glasshouses) ----
    bmask = bsd < 1.0
    lab, n = ndi.label(bmask)
    if n:
        means = ndi.mean(F, lab, range(1, n + 1))
        idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        near = lab[idx[0], idx[1]]
        padh = np.where(near > 0, np.concatenate([[0], means])[near], F)
        wb = np.clip((3.0 - bsd) / 3.0, 0, 1)
        F = F * (1 - wb) + padh * wb

    # ---- roads: smooth beds ----
    Fs = ndi.gaussian_filter(F, 4.0)
    wr = np.clip(ndi.gaussian_filter(road, 1.5) * 1.5, 0, 1)
    F = F * (1 - wr) + (Fs - 0.15) * wr

    # ---- lake: bed + banks + castle cliff (only near the shore) ----
    s = lake_sd
    depth = 0.8 + 11.5 * (1 - np.exp(-np.maximum(-s, 0) / 50.0))
    near_castle = ndi.gaussian_filter(cz.astype(np.float32), 10.0) > 0.02
    kmax = np.where(near_castle, 3.0, 0.45)
    land = s >= 0
    channel = (water_sd < 0) | river
    sp = np.maximum(s, 0)
    lo = L + 0.35 + 0.05 * sp
    hi = L + 0.35 + kmax * sp
    raise_w = np.clip(1 - sp / 25.0, 0, 1) * (~channel)
    Fl = np.minimum(F, hi)
    Fl = np.maximum(Fl, lo * raise_w + Fl * (1 - raise_w))
    keep_w = np.clip((sp - 50.0) / 110.0, 0, 1); keep_w = keep_w * keep_w * (3 - 2 * keep_w)
    Fl = Fl * (1 - keep_w) + F * keep_w
    F = np.where(land, Fl, L - depth)

    # ---- rivers / streams: shallow channels (water surface == terrain surface) ----
    st = (water_sd < 0) & land & (lake_sd > 2.0)
    Fst = ndi.grey_erosion(ndi.gaussian_filter(F, 2.0), size=5)
    F = np.where(st, np.minimum(F, Fst - 0.25), F)

    np.save(P("build", "T_local_height.npy"), F.astype(np.float32))
    print("local height range", F.min(), F.max())
    # hillshade preview
    gy, gx = np.gradient(F, RES)
    sl = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
    hs = np.clip(np.sin(np.deg2rad(40)) * np.cos(sl) + np.cos(np.deg2rad(40)) * np.sin(sl) * np.cos(np.deg2rad(315 + 25.1) - asp), 0, 1)
    img = np.stack([hs] * 3, -1) * 220 + 25
    img[lake_sd < 0] = img[lake_sd < 0] * 0.5 + np.array([30, 80, 160]) * 0.5
    c = (np.floor(F / 5) != np.floor(np.roll(F, 1, 0) / 5)) | (np.floor(F / 5) != np.floor(np.roll(F, 1, 1) / 5))
    img[c] = img[c] * 0.6 + np.array([200, 120, 0]) * 0.4
    save_rgb(img, "terrain_local.png")
    json.dump({"pitch_level": level, "hills": hills}, open(P("build", "terrain_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
