"""Pack all layers into a compact JS data blob (base64) for the single-file HTML."""
from common import *
import zlib, base64, io

OUT = P("build", "data.js")


def b64(b): return base64.b64encode(b).decode("ascii")


def enc_height(h, q, name):
    """Planar-predicted int residuals, zigzag -> uint16 split into lo/hi byte planes, zlib."""
    v = np.round(np.asarray(h, np.float64) / q).astype(np.int64)
    pred = np.zeros_like(v)
    pred[1:, 1:] = v[:-1, 1:] + v[1:, :-1] - v[:-1, :-1]
    pred[0, 1:] = v[0, :-1]; pred[1:, 0] = v[:-1, 0]
    r = v - pred
    z = np.where(r >= 0, 2 * r, -2 * r - 1)
    assert z.max() < 65536, (name, int(z.max()))
    z = z.astype(np.uint16)
    raw = (z & 255).astype(np.uint8).tobytes() + (z >> 8).astype(np.uint8).tobytes()
    c = zlib.compress(raw, 9)
    return {"enc": "h16", "w": int(h.shape[1]), "h": int(h.shape[0]), "q": q, "data": b64(c)}, len(c)


def enc_u8(a, name, delta=None):
    """Planar uint8 channels; per-channel optional horizontal delta; zlib over the planar stack."""
    a = np.asarray(a, np.uint8)
    if a.ndim == 2: a = a[..., None]
    planes = []; flags = []
    for c in range(a.shape[2]):
        p = a[..., c]
        d = p.astype(np.int16).copy(); d[:, 1:] = p[:, 1:].astype(np.int16) - p[:, :-1].astype(np.int16)
        d = (d & 255).astype(np.uint8)
        use = delta[c] if delta is not None else (len(zlib.compress(d.tobytes(), 6)) < len(zlib.compress(p.tobytes(), 6)))
        planes.append(d if use else p); flags.append(1 if use else 0)
    c = zlib.compress(b"".join(pl.tobytes() for pl in planes), 9)
    return {"enc": "u8", "w": int(a.shape[1]), "h": int(a.shape[0]), "ch": int(a.shape[2]), "delta": flags, "data": b64(c)}, len(c)


def enc_jpg(rgb, q=84):
    im = Image.fromarray(np.asarray(rgb, np.uint8))
    b = io.BytesIO(); im.save(b, "JPEG", quality=q, optimize=True, subsampling=2)
    return {"enc": "jpg", "w": im.size[0], "h": im.size[1], "data": b64(b.getvalue())}, len(b.getvalue())


def deshade(S, bed, st, az=152.0, el=45.0, amb=0.35):
    """Remove the Sentinel-2 acquisition-time hill shading so the shader can relight the terrain."""
    lin = (S.astype(np.float32) / 255.0) ** 2.2
    b = ndi.gaussian_filter(bed.astype(np.float32), 0.8)
    gy, gx = np.gradient(b, st)                       # rows run south -> north
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    a, e = np.radians(az), np.radians(el)
    s = np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    L = np.clip(n @ s, 0, None)
    shade = amb + (1 - amb) * L / np.sin(e)
    alb = lin / np.clip(shade, 0.45, 1.6)[..., None]
    return (np.clip(alb, 0, 1) ** (1 / 2.2) * 255).round().astype(np.uint8)


def clean_water(b, s, minsize):
    """Drop tiny water fragments (stream slivers that alias into sparkles) and their shore rings."""
    w = (s > -999.0) & (s - b > 0)
    lab, n = ndi.label(w)
    sz = np.bincount(lab.ravel())
    keep = sz >= minsize; keep[0] = False
    K = keep[lab]
    drop = w & ~K
    b = np.where(drop, np.maximum(b, s), b)
    s = np.where(ndi.binary_dilation(K, iterations=2), s, -9999.0)
    print(f"  water: kept {K.sum()} texels, dropped {drop.sum()} in {int((~keep[1:]).sum())} fragments")
    return b, s


def local_EN(meta):
    """World (E, N) of every local 2 m texel centre."""
    loc = meta["local"]; au, av = np.array(loc["aff"][0]), np.array(loc["aff"][1]); nx, ny = loc["nx"], loc["ny"]
    Mi = np.linalg.inv(np.array([[au[0], au[1]], [av[0], av[1]]]))
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    uu = (jj + 0.5) / nx - au[2]; vv = (ii + 0.5) / ny - av[2]
    return Mi[0, 0] * uu + Mi[0, 1] * vv, Mi[1, 0] * uu + Mi[1, 1] * vv


def carve_ship(meta, bed, wd):
    """The ship's ink replaced lake colour on the essay map, leaving a ship-shaped island: flood its berth."""
    E, N = local_EN(meta)
    s = meta["extras"]["ship"]; A, B = np.array(s["a"]), np.array(s["b"]); ab = B - A
    h = np.clip(((E - A[0]) * ab[0] + (N - A[1]) * ab[1]) / (ab @ ab), 0, 1)
    d = np.hypot(E - A[0] - h * ab[0], N - A[1] - h * ab[1])
    lake = meta["lake"]; m = d < 12
    bed = np.where(m, np.minimum(bed, lake - 1.5 - 4.0 * (12 - d) / 12), bed)
    wd = np.where(m, lake - bed, wd)
    print(f"  ship berth: flooded {int(m.sum())} texels")
    return bed, wd


def far_field(meta, la, l2):
    """Horizontal distance (m, 0..255) from each local 2 m texel to the nearest hard object; lets low rays take long steps."""
    E, N = local_EN(meta)
    sdq = lambda a: (a.astype(np.float32) - 128) / 10
    occ = (sdq(la[..., 3]) < 0.5) | (sdq(la[..., 2]) < 0.6) | (sdq(l2[..., 0]) < 0.5)
    ca = meta["castle"]; cau, cav = ca["aff"]
    cu = cau[0] * E + cau[1] * N + cau[2]; cv = cav[0] * E + cav[1] * N + cav[2]
    csd = np.load(P("build", "final_Csd.npy")).astype(np.float32)
    vals = ndi.map_coordinates(csd, [(cv * ca["ny"] - 0.5).ravel(), (cu * ca["nx"] - 0.5).ravel()], order=1, mode="nearest").reshape(E.shape)
    occ |= (cu > 0) & (cu < 1) & (cv > 0) & (cv < 1) & (vals < 133)
    for t in meta["towers"]:
        occ |= np.hypot(E - t[0], N - t[1]) < t[2] * 1.5

    def disc(c, r):
        return np.hypot(E - c[0], N - c[1]) < r

    def capsule(a, b, r):
        a, b = np.asarray(a, float), np.asarray(b, float); ab = b - a
        h = np.clip(((E - a[0]) * ab[0] + (N - a[1]) * ab[1]) / (ab @ ab), 0, 1)
        return np.hypot(E - a[0] - h * ab[0], N - a[1] - h * ab[1]) < r

    X = meta["extras"]; pc = np.array(meta["pitch"]["c"]); br = np.radians(meta["pitch"]["brg"])
    dv = np.array([np.sin(br), np.cos(br)])
    occ |= disc(pc + 61 * dv, 9) | disc(pc - 61 * dv, 9)
    tr = meta["train"]
    occ |= capsule(*tr["carriages"], 3.5) | capsule(*tr["engine"], 3.5) | capsule(X["ship"]["a"], X["ship"]["b"], 13)
    for g in X["gates"]:
        occ |= disc(g, 4.5)
    occ |= disc(X["tomb"], 3.5) | disc(X["willow"], 12.5)
    d = ndi.distance_transform_edt(~occ) * 2.0
    print(f"  far field: {occ.mean() * 100:.1f}% occupied, median free distance {np.median(d):.0f} m")
    return np.clip(np.floor(d), 0, 255).astype(np.uint8)


def main():
    meta = json.load(open(P("build", "final_meta.json")))
    tex = {}; sizes = {}

    def add(name, pair):
        tex[name], sizes[name] = pair

    # ---- local (2 m, map-aligned) ----
    bed = np.load(P("build", "final_bed.npy")); wd = np.load(P("build", "final_wdelta.npy"))
    bed, wd = carve_ship(meta, bed, wd)
    wd = np.where(wd < -1.0, -8.0, wd)
    add("locBed", enc_height(bed, 0.02, "locBed"))
    add("locW", enc_height(wd, 0.02, "locW"))
    q16 = lambda a: (np.round(np.asarray(a, np.float32) / 17.0) * 17).astype(np.uint8)
    la = np.load(P("build", "final_L4.npy")).copy(); la[..., 0] = q16(la[..., 0]); la[..., 1] = q16(la[..., 1])
    add("locA", enc_u8(la, "locA"))
    l2 = np.load(P("build", "final_L2.npy"))
    add("locB", enc_u8(np.dstack([l2, far_field(meta, la, l2)]), "locB"))
    add("locN", enc_u8(np.load(P("build", "final_LN.npy")), "locN", delta=[0, 0]))

    # ---- regional LODs (north-up, rows south->north) ----
    for name, q in (("M", 0.1), ("A", 0.25), ("B", 0.5)):
        b = np.load(P("build", f"R_{name}_bed.npy")); s = np.load(P("build", f"R_{name}_surf.npy"))
        b, s = clean_water(b, s, {"M": 30, "A": 8, "B": 4}[name])
        w = np.where(s > -999.0, s - b, -8.0)
        w = np.where(w < -1.0, -8.0, w)
        add(f"{name}Bed", enc_height(b, q, name + "Bed"))
        add(f"{name}W", enc_height(w, 0.02, name + "W"))
        st = {"M": 10.0, "A": 30.0, "B": 120.0}[name]
        S = deshade(np.load(P("build", f"S_{name}.npy")), b, st)
        np.save(P("build", f"Sd_{name}.npy"), S)
        add(f"{name}Sat", enc_jpg(S, 84 if name != "B" else 86))
        if name in ("M", "A"):
            add(f"{name}For", enc_u8(q16(np.load(P("build", f"SF_{name}.npy"))), name + "For"))

    # ---- castle (0.5 m) ----
    csd = np.load(P("build", "final_Csd.npy")); cn = np.load(P("build", "final_CN.npy"))
    ca = meta["castle"]; au, av = np.array(ca["aff"][0]), np.array(ca["aff"][1])
    nxc, nyc = ca["nx"], ca["ny"]
    # invert the affine: (u, v) -> (e, n)
    M2 = np.array([[au[0], au[1]], [av[0], av[1]]]); Mi = np.linalg.inv(M2)
    jj, ii = np.meshgrid(np.arange(nxc), np.arange(nyc))
    uu = (jj + 0.5) / nxc - au[2]; vv = (ii + 0.5) / nyc - av[2]
    E = Mi[0, 0] * uu + Mi[0, 1] * vv; N = Mi[1, 0] * uu + Mi[1, 1] * vv
    tw = np.array(meta["towers"], np.float64)
    d = np.hypot(E[..., None] - tw[:, 0], N[..., None] - tw[:, 1]) - tw[:, 2]
    o = np.argsort(d, axis=-1)
    t1 = o[..., 0].astype(np.uint8); t2 = o[..., 1].astype(np.uint8)
    add("cSd", enc_u8(csd, "cSd"))
    add("cN", enc_u8(np.stack([cn[..., 0], cn[..., 1], t1, t2], -1), "cN", delta=[0, 0, 0, 0]))

    # ---- essay Forest map layer (8 m) ----
    fo = np.load(P("build", "final_F2.npy")).copy()
    fo[..., 0] = q16(ndi.gaussian_filter(fo[..., 0].astype(np.float32), 0.6)); fo[..., 1] = q16(fo[..., 1])
    add("forest", enc_u8(fo, "forest"))

    total = sum(sizes.values())
    for k, v in sizes.items(): print(f"{k:10s} {v / 1024:8.1f} KB")
    print(f"TOTAL binary {total / 1024:.0f} KB  (base64 ~{total * 4 / 3 / 1024:.0f} KB)")
    js = "const DATA=" + json.dumps({"tex": tex, "meta": meta}, separators=(",", ":")) + ";\n"
    open(OUT, "w", encoding="ascii").write(js)
    print("wrote", OUT, len(js) // 1024, "KB")


if __name__ == "__main__":
    main()
