"""Final consolidated build of all essay-derived local layers (map-aligned frame) + castle + forest layers.
Outputs into build/final_*.npy and build/final_meta.json"""
from common import *
from PIL import ImageDraw
import shapely
import castle_base

import clearings
import forest_extent
import forest_places
import grounds_grade
import railway
from mesh_source import load_georef

# ------------------------------------------------------------------ frames
FX0, FX1, FY0, FY1 = 935.0, 1705.0, 2580.0, 3295.0       # local frame in Forest-map px (covers grounds + Hogsmeade maps)
RES = LOCAL_RES_M
NX = int(round((FX1 - FX0) * MFF / RES)); NY = int(round((FY1 - FY0) * MFF / RES))
PG = 0.72                                                  # m / px of grounds + Hogsmeade maps
LAKE = SITE["lake"]
CX0, CY0, CW, CH = 1030, 560, 500, 620                     # castle window (grounds px)
CRES = 0.5
FRES = 8.0                                                 # forest layer resolution (m)
# castle mass classes (relative top in m, roof type): low wall, wall, range, block, Great Hall, keep
CAS_CLS = [(2.2, 4), (13.0, 0), (24.0, 1), (32.0, 1), (21.0, 3), (27.0, 0)]
PITCH_FLAT_M = 3.0                                         # the pitch's level ground runs past its stands (> TEXEL_REACH_M)
PITCH_BATTER = 0.4                                         # then cut or filled back to the ground at 1 in 2.5
WILLOW_LINE_M = 4.0                                        # each greenhouse's line to the Willow is kept clear this far
                                                           # either side; the clearing's margin covers its crown's breadth
SHACK_VIEW_DEG = 2.0                                       # the Shack's window sees the castle's breadth and this either
SHACK_WINDOW_M = 3.0                                       # side, from anywhere this near its line to the castle
PLANTED = 9                                                # the planted woods' floor, a class of its own while it blends

DECOR_G = [(990, 2170, 1900, 2300), (1580, 1840, 1960, 2180), (740, 2270, 900, 2330)]
DECOR_H = [(1260, 2150, 1660, 2240), (1290, 1070, 1650, 1430), (1430, 1460, 1590, 1520)]


def local_texel_ff(i, j):
    return FX0 + (np.asarray(j) + 0.5) * RES / MFF, FY0 + (np.asarray(i) + 0.5) * RES / MFF


def local_grid():
    jj, ii = np.meshgrid(np.arange(NX), np.arange(NY))
    return local_texel_ff(ii, jj)


def sdf(mask, px_m):
    if not mask.any(): return np.full(mask.shape, 1e3, np.float32)
    out = ndi.distance_transform_edt(~mask) * px_m
    ins = ndi.distance_transform_edt(mask) * px_m
    return np.where(mask, -ins + 0.5 * px_m, out - 0.5 * px_m).astype(np.float32)


def sample(img, u, v, order=1, cval=0.0):
    return map_coordinates(np.asarray(img, np.float32), [v, u], order=order, mode='constant', cval=cval)


def blank(mask, boxes):
    m = mask.copy()
    for x0, y0, x1, y1 in boxes: m[y0:y1, x0:x1] = False
    return m


def drop_small(m, min_px):
    lab, n = ndi.label(m)
    if not n: return m
    sz = ndi.sum(m, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.nonzero(sz >= min_px)[0])


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def mound(E, N, ce, cn, ra, rb, h, ang=0.0):
    c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    x = (E - ce) * c - (N - cn) * s; y = (E - ce) * s + (N - cn) * c
    t = np.clip(1 - np.sqrt((x / ra) ** 2 + (y / rb) ** 2), 0, 1)
    return h * t * t * (3 - 2 * t)


def channel_sd(lines, shape, step=0.1):
    """Signed distance (m) from every texel centre to channels along texel polylines (px, py, width in m), negative
    inside: exact to within step / 2 texels, so a channel narrower than a texel stays continuous at any angle."""
    from scipy.spatial import cKDTree

    ii, jj = np.mgrid[0:shape[0], 0:shape[1]]
    centres = np.stack([ii.ravel(), jj.ravel()], 1).astype(np.float64)
    sd = np.full(centres.shape[0], 50.0)
    for px, py, width in lines:
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
        t = np.arange(0.0, s[-1] + step, step)
        d, _ = cKDTree(np.stack([np.interp(t, s, py), np.interp(t, s, px)], 1)).query(centres, distance_upper_bound=25.0)
        sd = np.minimum(sd, d * RES - width / 2.0)
    return sd.reshape(shape).astype(np.float32)


def downhill(lines, ground):
    """A level that never rises along any of the lines (texel polylines, each in its downstream order): the running
    minimum of `ground` along each line (inf where it is not ground to follow), taken by every texel from its nearest
    line texel."""
    level = np.full(ground.shape, np.inf, ground.dtype)
    for px, py in lines:
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
        t = np.arange(0.0, s[-1], 0.5)
        j = np.round(np.interp(t, s, px)).astype(int); i = np.round(np.interp(t, s, py)).astype(int)
        inside = (i >= 0) & (i < ground.shape[0]) & (j >= 0) & (j < ground.shape[1])
        h = np.full(len(t), np.inf); h[inside] = ground[i[inside], j[inside]]
        np.minimum.at(level, (i[inside], j[inside]), np.minimum.accumulate(h)[inside])
    near = ndi.distance_transform_edt(~np.isfinite(level), return_distances=False, return_indices=True)
    return level[near[0], near[1]]


WILLOW_G = (1060, 492)          # the Whomping Willow (5a) on the grounds map


def g2w(u, v): return ff_to_world(*g_to_ff(u, v))
def h2w(u, v): return ff_to_world(*h_to_ff(u, v))


def thin_lines(blk, keep_out):
    """Long thin black ink lines (field boundaries / hedges)."""
    cand = blk & ~keep_out
    lab, n = ndi.label(cand, structure=np.ones((3, 3), bool))
    if not n: return np.zeros_like(blk)
    objs = ndi.find_objects(lab); sz = ndi.sum(cand, lab, range(1, n + 1))
    keep = np.zeros(n + 1, bool)
    for i, s in enumerate(objs):
        h = s[0].stop - s[0].start; w = s[1].stop - s[1].start; L = max(h, w)
        if L >= 45 and sz[i] / L < 3.2 and sz[i] / (h * w) < 0.25: keep[i + 1] = True
    return keep[lab]


def wall_chain_polygon(wall):
    """Greedy-chain the boundary-wall pixels into a polyline and close it (grounds native px)."""
    ys, xs = np.nonzero(wall)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    # thin to ~6 px spacing
    key = (pts // 6).astype(np.int32); _, idx = np.unique(key[:, 0] * 10000 + key[:, 1], return_index=True)
    pts = pts[idx]
    start = int(np.argmin(pts[:, 0] + 0.001 * pts[:, 1]))          # left-most point
    used = np.zeros(len(pts), bool); chain = [start]; used[start] = True
    while True:
        cur = pts[chain[-1]]
        d = np.hypot(*(pts - cur).T); d[used] = 1e9
        k = int(np.argmin(d))
        if d[k] > 90: break
        chain.append(k); used[k] = True
    return pts[chain]


def gates_gap(wall):
    """The winged-boar gates (3g): the widest gap in the boundary wall's chain near their label, as world (east, north)
    ends."""
    chain = wall_chain_polygon(wall)
    label = np.array([416.0, 1248.0])
    gaps = [(np.linalg.norm(b - a), a, b) for a, b in zip(chain[:-1], chain[1:]) if np.linalg.norm((a + b) / 2 - label) < 140]
    _, a, b = max(gaps, key=lambda gap: gap[0])
    return np.array(g2w(*a), float), np.array(g2w(*b), float)


def extract_train(red):
    """Red ink on the grounds map = the Hogwarts Express drawn to scale (carriages at the platform, engine on the loop)."""
    lab, n = ndi.label(red, structure=np.ones((3, 3)))
    sz = ndi.sum(red, lab, range(1, n + 1))
    comps = []
    for i in np.argsort(sz)[::-1][:4]:
        ys, xs = np.nonzero(lab == i + 1)
        if len(xs) >= 60: comps.append((xs, ys))
    plat = [c for c in comps if c[1].mean() < 1000]; loco = [c for c in comps if c[1].mean() >= 1000]

    def axis(xs, ys):
        cx, cy = xs.mean(), ys.mean(); ev, evc = np.linalg.eigh(np.cov(np.stack([xs - cx, ys - cy]))); d = evc[:, 1]
        pr = (xs - cx) * d[0] + (ys - cy) * d[1]
        a = (cx + d[0] * pr.min(), cy + d[1] * pr.min()); b = (cx + d[0] * pr.max(), cy + d[1] * pr.max())
        return [float(v) for v in g2w(*a)], [float(v) for v in g2w(*b)]
    t = {"carriages": axis(np.concatenate([c[0] for c in plat]), np.concatenate([c[1] for c in plat]))}
    t["engine"] = axis(*loco[0])
    # orient so that carriages run [platform far end -> engine end]
    ca, cb = map(np.array, t["carriages"]); ea, eb = map(np.array, t["engine"])
    if np.linalg.norm(ca - ea) < np.linalg.norm(cb - ea): t["carriages"] = [cb.tolist(), ca.tolist()]
    if np.linalg.norm(eb - np.array(t["carriages"][1])) < np.linalg.norm(ea - np.array(t["carriages"][1])): t["engine"] = [eb.tolist(), ea.tolist()]
    return t


def main():
    cg = np.load(P("build", "cls_g.npy")); ch = np.load(P("build", "cls_h.npy")); cff = np.load(P("build", "cls_ff.npy"))
    mg = dict(np.load(P("build", "native_g.npz"))); mh = dict(np.load(P("build", "native_h.npz")))
    for k in ("tree_ink", "wall", "build", "earth", "white", "wood", "water", "red"):
        mg[k] = blank(mg[k], DECOR_G); mh[k] = blank(mh[k], DECOR_H)
    blk_g = blank(cg == C_BLACK, DECOR_G); blk_h = blank(ch == C_BLACK, DECOR_H)

    FU, FV = local_grid()
    E, N = ff_to_world(FU, FV)
    gu, gv = ff_to_g(FU, FV); hu, hv = ff_to_h(FU, FV)
    inG = (gu >= 0) & (gu < G_W - 1) & (gv >= 0) & (gv < G_H - 1)
    inH = (hu >= 0) & (hu < H_W - 1) & (hv >= 0) & (hv < H_H - 1)
    useH = inH; useG = inG & ~inH; useF = ~(inG | inH)
    print("local grid", NX, NY)

    # ---------------- castle mass (grounds window) ----------------
    cmass = np.load(P("build", "castle_mass_g.npy"))
    cm_full = np.zeros((G_H, G_W), bool); cm_full[CY0:CY0 + CH, CX0:CX0 + CW] = cmass
    castle_zone = ndi.binary_dilation(cm_full, iterations=8)

    # ---------------- water ----------------
    wg = mg["water"].copy(); wh = mh["water"].copy()
    lab, n = ndi.label(wg); sizes = ndi.sum(wg, lab, range(1, n + 1)); lake = lab == 1 + int(np.argmax(sizes))
    holes = ndi.binary_fill_holes(lake) & ~lake
    hl, hn = ndi.label(holes)
    greenish = mg["tree_ink"] | (cg == C_LAWN)
    island = np.zeros_like(lake)
    for i in range(1, hn + 1):
        reg = hl == i
        if reg.sum() > 400 and (greenish & reg).sum() > 0.3 * reg.sum(): island |= reg
    lake_full = ndi.binary_closing(ndi.binary_fill_holes(lake) & ~island, iterations=2) & ~island
    wg = drop_small(ndi.binary_opening((wg & ~lake), iterations=1), 80) | lake_full
    wh = drop_small(ndi.binary_opening(wh, iterations=1), 80)
    sd_lake = sdf(lake_full, PG)
    lake_sd = np.where(inG, sample(sd_lake, gu, gv, cval=500), 500.0)
    stream = np.where(useH, sample(sdf(wh, PG), hu, hv, cval=50), np.where(useG, sample(sdf(wg & ~lake_full, PG), gu, gv, cval=50), 50.0))
    # the site's real watercourses (OSM) through the frame, as texel lines in their downstream order with widths (m)
    osm = json.load(open(P("osm.json")))
    riv = Image.new("L", (NX, NY), 0); dr = ImageDraw.Draw(riv); lines = []
    for el in osm["elements"]:
        t = el.get("tags", {}); g = el.get("geometry")
        if el["type"] != "way" or not g or t.get("waterway") not in SITE["carve"]: continue
        xy = np.array(to_tm.transform([p["lon"] for p in g], [p["lat"] for p in g])).T
        U, V = world_to_ff(xy[:, 0] - CASTLE_E, xy[:, 1] - CASTLE_N)
        px = (U - FX0) * MFF / RES - 0.5; py = (V - FY0) * MFF / RES - 0.5
        if px.max() < -50 or py.max() < -50 or px.min() > NX + 50 or py.min() > NY + 50: continue
        width = SITE["carve"][t["waterway"]]
        dr.line(list(zip(px, py)), fill=255, width=round(width / RES)); lines.append((px, py, width))
    river = np.asarray(riv) > 0
    river_sd = channel_sd(lines, river.shape) if SITE["graded"] else sdf(river, RES) + 1.0
    if SITE["graded"]:
        river = river_sd < 0
    if not SITE["carve_in_maps"]:       # inside White Hound's maps, the maps' own streams stand for the real ones
        river &= useF; river_sd = np.where(useF, river_sd, 50.0)
    stream = np.minimum(stream, river_sd)
    stream = np.where(lake_sd < 3.0, 50.0, stream)

    # ---------------- boundary wall, inside-grounds polygon ----------------
    chain = wall_chain_polygon(mg["wall"])
    print("wall chain points", len(chain))
    poly = Image.new("L", (G_W, G_H), 0); ImageDraw.Draw(poly).polygon([tuple(p) for p in chain], fill=255)
    inside_g = (np.asarray(poly) > 0) | cm_full
    inside = np.where(inG, sample(inside_g.astype(np.float32), gu, gv, order=0), 0.0) > 0.5
    wall_sd = np.where(useH, sample(sdf(mh["wall"], PG), hu, hv, cval=50), np.where(useG, sample(sdf(mg["wall"] & ~castle_zone, PG), gu, gv, cval=50), 50.0))

    # ---------------- hedges / field dykes (thin ink lines) ----------------
    def outline_of(m, blk):
        nongreen = m["water"] | m["build"] | m["earth"] | m["wood"] | m["white"] | m["red"]
        return blk & ndi.binary_dilation(nongreen, iterations=2)
    keep_g = ndi.binary_dilation(mg["wall"], iterations=2) | outline_of(mg, blk_g) | castle_zone
    keep_h = ndi.binary_dilation(mh["wall"], iterations=2) | outline_of(mh, blk_h)
    hedge_g = thin_lines(blk_g, keep_g); hedge_h = thin_lines(blk_h, keep_h)
    hedge_sd = np.where(useH, sample(sdf(hedge_h, PG), hu, hv, cval=50), np.where(useG, sample(sdf(hedge_g, PG), gu, gv, cval=50), 50.0))
    print("hedge px g/h", hedge_g.sum(), hedge_h.sum())

    # ---------------- buildings with kinds ----------------
    KIND_THATCH, KIND_SLATE, KIND_GLASS, KIND_HUT, KIND_SHACK, KIND_STANDS, KIND_BIG, KIND_STATION = range(8)
    lake_near = ndi.binary_dilation(lake_full | mg["water"], iterations=4)
    white_g = mg["white"] & ~lake_near
    pitch_box = np.zeros_like(lake_full); pitch_box[1240:1450, 530:830] = True
    stands = mg["wood"] & pitch_box
    bmask_g = ((mg["build"] | white_g) & ~castle_zone) | stands
    bmask_h = mh["build"] | (mh["white"] & ~ndi.binary_dilation(mh["water"], iterations=4))
    shack_w = h2w(365 * 1735 / 1041, 607 * 1735 / 1041)
    hut_w = (-291.5, -191.9)
    rng = np.random.default_rng(11)

    def attrs(bm, white, wood, to_w):
        bm = ndi.binary_opening(bm, iterations=1)
        lab, n = ndi.label(bm)
        objs = ndi.find_objects(lab)
        area = ndi.sum(bm, lab, range(1, n + 1)) * PG * PG
        wf = ndi.sum(white, lab, range(1, n + 1)) * PG * PG / np.maximum(area, 1)
        woodf = ndi.sum(wood, lab, range(1, n + 1)) * PG * PG / np.maximum(area, 1)
        code = np.zeros(n + 1, np.uint8); keep = np.zeros(n + 1, bool)
        for i in range(n):
            a = area[i]
            if a < 10: continue
            s = objs[i]; cy = (s[0].start + s[0].stop) / 2; cx = (s[1].start + s[1].stop) / 2
            e, nn = to_w(cx, cy); e = float(e); nn = float(nn)
            if woodf[i] > 0.5: kind, eave = KIND_STANDS, 9.0
            elif wf[i] > 0.5: kind, eave = KIND_GLASS, 3.0
            elif np.hypot(e - hut_w[0], nn - hut_w[1]) < 12: kind, eave = KIND_HUT, 3.0
            elif np.hypot(e - shack_w[0], nn - shack_w[1]) < 25: kind, eave = KIND_SHACK, 6.0
            elif a > 400: kind, eave = KIND_BIG, 8.0
            elif a > 160: kind, eave = KIND_SLATE, 6.5
            else:
                kind = KIND_THATCH if rng.uniform() < 0.72 else KIND_SLATE
                eave = 4.0 if a < 60 else 5.0
            ecode = int(np.clip(round((eave - 2.5) / 0.5), 0, 15))
            code[i + 1] = kind * 16 + ecode; keep[i + 1] = True
        return keep[lab], code[lab]

    kb_g, code_g = attrs(bmask_g, white_g, stands, g2w)
    kb_h, code_h = attrs(bmask_h, mh["white"], mh["wood"], h2w)
    bsd = np.where(useH, sample(sdf(kb_h, PG), hu, hv, cval=50), np.where(useG, sample(sdf(kb_g, PG), gu, gv, cval=50), 50.0))

    def spread(vals, keep):
        idx = ndi.distance_transform_edt(~keep, return_distances=False, return_indices=True)
        return vals[idx[0], idx[1]]
    bcode = np.where(useH, sample(spread(code_h, kb_h), hu, hv, order=0), np.where(useG, sample(spread(code_g, kb_g), gu, gv, order=0), 0)).astype(np.uint8)
    bcode = np.where(bsd < 8.0, bcode, 0).astype(np.uint8)

    # ---------------- roads ----------------
    road = np.where(useH, sample(ndi.gaussian_filter(mh["earth"].astype(np.float32), 0.7), hu, hv),
                    np.where(useG, sample(ndi.gaussian_filter((mg["earth"] & ~castle_zone).astype(np.float32), 0.7), gu, gv), 0.0))
    road = np.clip(road * 1.25, 0, 1)

    # ---------------- pitch frame: centre, the stands' outer semi-axes and the long axis's bearing ----------------
    pc_e, pc_n, pa, pb, pbrg = -243.6, -485.9, 101.6, 54.6, 271.25
    t = np.deg2rad(90 - pbrg)
    px_ = (E - pc_e) * np.cos(t) + (N - pc_n) * np.sin(t); py_ = -(E - pc_e) * np.sin(t) + (N - pc_n) * np.cos(t)
    rpitch = np.sqrt((px_ / (pa - 9)) ** 2 + (py_ / (pb - 9)) ** 2)
    turn = np.linspace(-np.pi, np.pi, 4000)
    stands_edge = shapely.Polygon(np.column_stack([pa * np.cos(turn), pb * np.sin(turn)]))
    near_p = (np.abs(px_) < pa + 150.0) & (np.abs(py_) < pb + 150.0)   # as far as the pitch's earthworks may reach
    beyond = np.full(E.shape, np.inf)                         # metres outside the stands' outer edge, 0 within it
    beyond[near_p] = shapely.distance(stands_edge, shapely.points(px_[near_p], py_[near_p]))

    # ---------------- trees ----------------
    def tree_density(m):
        d = ndi.gaussian_filter(m["tree_ink"].astype(np.float32), 3.0)
        t = np.clip((d - 0.10) / 0.22, 0, 1)
        t = np.maximum(t, ndi.gaussian_filter(m["under"].astype(np.float32), 2.0) * 0.6)
        excl = ndi.binary_dilation(m["earth"] | m["build"] | m["water"] | m["white"] | m["wood"] | m["wall"] | m["red"], iterations=3)
        return np.where(excl, 0, t)
    tg = tree_density(mg); th = tree_density(mh)
    tg = np.where(castle_zone, 0, tg)
    trees = np.where(useH, sample(th, hu, hv), np.where(useG, sample(tg, gu, gv), 0.0))
    tf = np.clip(ndi.gaussian_filter((cff == C_DENSE).astype(np.float32), 1.0) + 0.55 * ndi.gaussian_filter((cff == C_UNDER).astype(np.float32), 1.0), 0, 1)
    tf_l = sample(tf, FU, FV)
    dG = np.where(inG, np.minimum.reduce([gu, G_W - gu, gv, G_H - gv]) * PG, 0.0)
    dH = np.where(inH, np.minimum.reduce([hu, H_W - hu, hv, H_H - hv]) * PG, 0.0)
    wdet = np.clip(np.maximum(dG, dH) / 45.0, 0, 1)
    trees = np.where(useF, tf_l, trees * wdet + tf_l * (1 - wdet))
    drawn = trees.copy()        # the maps' trees: below, each rule that keeps them off or clears them is accounted for
    trees = np.where((stream < 1.0) | (lake_sd < 2.0) | (bsd < 2.0) | (wall_sd < 1.5), 0, trees)
    # nor on the pitch, whose markings read as tree ink on the essay map, nor where a crown could reach its stands
    trees = np.where(beyond < TREE_CROWN_M + TEXEL_REACH_M, 0, trees)
    accounted = trees < drawn

    # ---------------- surface class ----------------
    # 0 meadow, 1 lawn (inside grounds), 2 scrub, 3 earth/paving, 4 cultivated, 5 forest floor, 6 pitch, 7 shingle,
    # 8 an overgrown garden (clearings.OVERGROWN)
    def surf_class(c, m):
        s = np.zeros(c.shape, np.uint8)
        s[c == C_SCRUB] = 2; s[c == C_UNDER] = 4; s[m["earth"]] = 3
        return s
    sc = np.where(useH, sample(surf_class(ch, mh), hu, hv, order=0), np.where(useG, sample(surf_class(cg, mg), gu, gv, order=0), 0)).astype(np.uint8)
    ffc = np.zeros(cff.shape, np.uint8); ffc[cff == C_SCRUB] = 2; ffc[(cff == C_DENSE) | (cff == C_UNDER)] = 5
    sc = np.where(useF, sample(ffc, FU, FV, order=0), sc).astype(np.uint8)
    sc[(sc == 0) & inside] = 1
    sc[(trees > 0.55) & (sc <= 2)] = 5
    sc[rpitch < 1.0] = 6
    near_lake = (lake_sd > 0) & (lake_sd < 3.5) & (sc != 3)
    sc[near_lake] = 7

    # ---------------- terrain ----------------
    mos = tm_dem()
    D = dem_sample(mos, E + CASTLE_E, N + CASTLE_N, order=3).astype(np.float32)
    F0 = ndi.gaussian_filter(D, 3.0); F = F0.copy()
    hills = []
    for nm, (u, v), ra, rb, hh in [("hillock 4f", (975, 968), 58, 40, 5.0), ("hillock 4g", (883, 1022), 31, 20, 3.5),
                                   ("hillock 4f2", (878, 1120), 52, 26, 3.5), ("hillock 4h", (1135, 535), 24, 42, 4.0),
                                   ("howe 4i", (1559, 212), 70, 66, 7.0)]:
        e, nn = g2w(u, v); hills.append((nm, float(e), float(nn), ra, rb, hh))
    for nm, (u, v), ra, rb, hh in [("Shrieking Shack hill", (606, 1012), 118, 112, 9.0), ("Shack knoll", (612, 962), 26, 22, 5.0),
                                   ("fold of ground", (1037, 598), 190, 140, 6.0)]:
        e, nn = h2w(u, v); hills.append((nm, float(e), float(nn), ra, rb, hh))
    for nm, e, nn, ra, rb, hh in hills: F += mound(E, N, e, nn, ra, rb, hh, ang=-25.1)
    cz_l = np.where(inG, sample(castle_zone.astype(np.float32), gu, gv, order=0), 0.0) > 0.5
    Bc = np.maximum(ndi.gaussian_filter(F0, 12.0), LAKE + 36.0)       # essay: cliff over the lake ~120 ft
    F_open = F
    garden_hollow = np.zeros(F.shape, bool)
    if SITE["base"]:           # the castle on its own little hill or mound, or a crag, over the cliff above the lake
        cm_l = np.where(inG, sample(cm_full.astype(np.float32), gu, gv, order=0), 0.0) > 0.5
        toward = sum(np.array(p) / np.hypot(*p) for p in (hut_w, (pc_e, pc_n)))
        patch = ndi.gaussian_filter((sc == 4).astype(np.float32), 3.0) > 0.15     # the patch's hatching, as an area
        glass = (bcode // 16 == KIND_GLASS) & (bsd < 0.5)
        garden = ndi.binary_fill_holes(patch) | glass                                     # and its greenhouses
        dry = (bsd < 4.0) | (road > 0.3)               # buildings, their pads and the paths stand clear of the flood
        F, cliff, garden_hollow = castle_base.shape(SITE["base"], F, E, N, cm_l, lake_sd, Bc, LAKE,
                                                    toward / np.hypot(*toward), garden, dry, glass)
    else:
        wc = np.clip(ndi.gaussian_filter(cz_l.astype(np.float32), 6.0) * 1.6, 0, 1)
        F = F * (1 - wc) + Bc * wc
        cliff = np.zeros(F.shape, bool)
    barred = cz_l | (lake_sd < 2.0) | (bsd < 2.0) | (wall_sd < 1.5) | (beyond < TREE_CROWN_M + TEXEL_REACH_M)
    if SITE["graded"]:          # where the castle's base buries the lake's outflow, it runs under the base, not over it
        buried = (F - F_open > 1.0) & (F > LAKE - 0.3)
        course = buried & (stream < 3.0)        # with the margin its ink kept clear of trees
        stream = np.where(buried, np.maximum(stream, 50.0), stream)
        # and leaves no treeless line over it: its course is wooded as the ground round it, where trees may grow at all
        round_it = ndi.gaussian_filter(np.where(course, 0.0, trees), 3.0) / \
            np.maximum(ndi.gaussian_filter((~course).astype(np.float32), 3.0), 1e-6)
        trees = np.where(course & ~barred, round_it, trees)
        sc[(trees > 0.55) & (sc <= 2)] = 5
        accounted |= course
    # no patch of ground ends at a line (clearings.blend): the maps' scrub fades into the meadow and lawn round it, the
    # woods' floor into the meadow, lawn and scrub under their thinning edge (only under the canopy: beyond it, tufts of
    # the floor read as specks on open ground), and the lawns into the meadow beyond the grounds, but along the wall
    along_wall = wall_sd < 3.0
    sc = clearings.blend(sc, clearings.SCRUB, RES, 31, (0, 1), along_wall)
    sc = clearings.blend(sc, 5, RES, 32, (0, 1, clearings.SCRUB), along_wall, may=trees > 0.05)
    sc = clearings.blend(sc, 1, RES, 33, (0,), along_wall)
    glades = []                 # the clearings recorded in meta.clearings (clearings.py)
    # where a clearing's scattered trees may stand: where the renderer grows them (frag1.glsl treeDens: off the water,
    # which reaches a texel past a stream's ink and over the lake's shallows, the roads and the buildings), and off the
    # vegetable patch and its hollow
    stand = ~barred & (stream >= 2.5) & (lake_sd >= 5.0) & (road < 0.05) & (bsd >= 2.5) & ~garden_hollow & \
        ~ndi.binary_dilation(sc == 4, iterations=2)
    if SITE["willow_lawn"]:     # the Willow "stood alone": a lawn opens to it from the northern greenhouses, clear of trees
        willow = np.array(g2w(*WILLOW_G), float)
        scene = json.load(open(P("src", "mesh_scene.json"), encoding="utf-8"))
        north = next(a for a in scene["assets"] if a["id"] == "greenhouses-north")["components"]
        core = np.hypot(E - willow[0], N - willow[1]) < SITE["willow_lawn"]
        for c in north:         # each one's line to the Willow, kept clear either side
            core |= grounds_grade.segment_distance(E, N, c["at"], willow) < WILLOW_LINE_M
        lawn = clearings.glade(core, RES, 9)
        glades += clearings.record("willow-lawn", "willow-lawn", "local", lawn, E, N, trees * (1.0 - lawn) > 0.05)
        trees = np.maximum(trees * lawn, clearings.scattered(lawn, core, E, N, RES, 4, 9, stand))
        turf = clearings.ecotone(lawn, RES, 109) & (sc == 5)   # the woods' floor, giving way to lawn as they thin out;
        sc[turf] = np.where(inside, 1, 0)[turf]                 # the vegetable patch keeps its beds
        accounted |= lawn < 1.0
    if SITE["shack_garden"]:    # DH 32's one unblocked window looks over the Shack's "dank overgrown garden" (PoA 14)
        shack = (bcode // 16 == KIND_SHACK) & (bsd < 0.0)
        if not shack.any():
            raise RuntimeError("shack_garden is set, but no Shrieking Shack stands on the map to keep a view from")
        at = np.array([E[shack].mean(), N[shack].mean()])
        toward = -at / np.hypot(*at)
        reach = float(((E[shack] - at[0]) * toward[0] + (N[shack] - at[1]) * toward[1]).max()) + SITE["shack_garden"]
        turn = lambda e, n: np.angle(((e - at[0]) + 1j * (n - at[1])) / complex(*toward))     # bearing off `toward`
        ys, xs = np.nonzero(cm_full)
        castle, off = turn(*g2w(xs, ys)), turn(E, N)
        margin = np.radians(SHACK_VIEW_DEG)
        core = (np.hypot(E - at[0], N - at[1]) < reach) & (off >= castle.min() - margin) & (off <= castle.max() + margin)
        core |= grounds_grade.segment_distance(E, N, at, at + reach * toward) < SHACK_WINDOW_M
        overgrown = clearings.glade(core, RES, 14)
        glades += clearings.record("shack-garden", "shack-garden", "local", overgrown, E, N,
                                   trees * (1.0 - overgrown) > 0.05)
        trees = np.maximum(trees * overgrown, clearings.scattered(overgrown, core, E, N, RES, 3, 14, stand))
        rank = clearings.ecotone(overgrown, RES, 114) & np.isin(sc, (0, 2, 4, 5))  # rank grass, nettles and bramble,
        sc[rank] = clearings.OVERGROWN                                              # giving way as the trees close in
        accounted |= overgrown < 1.0
    if SITE["cabin_terrace"]:   # White Hound: "The front gates are lower down than Hagrid's cabin."
        drive = grounds_grade.drive_line(road, E, N, np.mean(gates_gap(mg["wall"]), axis=0),
                                         json.load(open(P("build", "castle_named.json")))["EH"])
        F = grounds_grade.cabin_terrace(F, E, N, hut_w, drive, SITE["cabin_terrace"], lake_sd, LAKE)
    if SITE["willow_knoll"]:    # "From the greenhouses it is possible to see at least the upper part of the Whomping Willow"
        F = grounds_grade.knoll(F, E, N, g2w(*WILLOW_G), SITE["willow_knoll"])
    inside_p = rpitch < 1.25
    level = float(np.median(F[inside_p]))
    # graded to the playing level from past the stands' outer edge inwards, so no deck meets the ground, then battered
    reach = PITCH_BATTER * np.maximum(beyond - PITCH_FLAT_M, 0.0)
    if np.any((np.abs(F - level) > reach)[near_p & ~ndi.binary_erosion(near_p)]):
        raise RuntimeError("the pitch's batters run past the frame its earthworks are computed in")
    F = np.clip(F, level - reach, level + reach)
    pitch_flat = beyond <= PITCH_FLAT_M                       # road cambers stop at it
    bm = bsd < 1.0
    lab, n = ndi.label(bm)
    if n:
        means = ndi.mean(F, lab, range(1, n + 1))
        idx = ndi.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        near = lab[idx[0], idx[1]]
        padh = np.where(near > 0, np.concatenate([[0], means])[near], F)
        wb = np.clip((3.0 - bsd) / 3.0, 0, 1)
        F = F * (1 - wb) + padh * wb
    wr = np.where(pitch_flat, 0.0, np.clip(ndi.gaussian_filter(road, 1.5) * 1.5, 0, 1))
    F = F * (1 - wr) + (ndi.gaussian_filter(F, 4.0) - 0.12) * wr
    # station: level the platform + track corridor under the Hogwarts Express (red ink on the grounds map)
    train = extract_train(mg["red"])
    seg = [train["carriages"][0], train["carriages"][1], train["engine"][1]]
    dmin = np.full(E.shape, 1e9)
    for (a, b) in zip(seg[:-1], seg[1:]):
        a = np.array(a); b = np.array(b); ab = b - a
        tt = np.clip(((E - a[0]) * ab[0] + (N - a[1]) * ab[1]) / (ab @ ab), 0, 1)
        dmin = np.minimum(dmin, np.hypot(E - (a[0] + tt * ab[0]), N - (a[1] + tt * ab[1])))
    corridor = dmin < 7.0
    plat_level = float(np.median(F[corridor]))
    wpl = np.clip((22.0 - dmin) / 15.0, 0, 1); wpl = wpl * wpl * (3 - 2 * wpl)
    F = F * (1 - wpl) + plat_level * wpl
    train["level"] = plat_level
    if SITE["portal_m"]:        # where the hill first rises over the main line (railway.alignment)
        train["portal_m"] = SITE["portal_m"]
    print("platform level", plat_level)
    s = lake_sd
    depth = 0.8 + 11.5 * (1 - np.exp(-np.maximum(-s, 0) / 50.0))
    near_castle = ndi.gaussian_filter(cz_l.astype(np.float32), 10.0) > 0.02
    kmax = np.where(cliff, castle_base.FACE_GRADE, np.where(near_castle, 3.0, 0.45))
    land = s >= 0
    channel = (stream < 0) | river
    sp = np.maximum(s, 0)
    lo = LAKE + 0.35 + 0.05 * sp; hi = LAKE + 0.35 + kmax * sp
    raise_w = np.clip(1 - sp / 25.0, 0, 1) * (~channel)
    Fl = np.minimum(F, hi); Fl = np.maximum(Fl, lo * raise_w + Fl * (1 - raise_w))
    keep_w = np.clip((sp - 50.0) / 110.0, 0, 1); keep_w = keep_w * keep_w * (3 - 2 * keep_w)
    Fl = Fl * (1 - keep_w) + F * keep_w
    if SITE["bank"]:            # a perched lake: banked up level for a crest's width, then sloping down to the ground
        crest, grade = SITE["bank"]
        Fl = np.maximum(Fl, LAKE + 0.35 + 0.05 * np.minimum(sp, crest) - grade * np.maximum(sp - crest, 0.0))
    F = np.where(land, Fl, LAKE - depth)
    if SITE["fill_pits"]:       # hollows the earthworks enclose fill to where they would spill, into the lake if need be
        filled, unnatural = fill(F, ~land), fill(F0, ~land) - F0 <= 0.05
        hollow = land & (filled - F > 0.05) & unnatural & ~garden_hollow   # but the patch floods
        F = np.where(hollow, filled, F).astype(np.float32)
    if SITE["station_wood"] or SITE["train_screen"]:
        # PS 6: "Yeh'll get yer firs' sight o' Hogwarts in a sec ... jus' round this bend here." Wooded rises keep it there
        scene, georef = json.load(open(P("src", "mesh_scene.json"), encoding="utf-8")), load_georef()
        al = railway.alignment(scene, {"train": train}, georef)
        terrace = shapely.contains_xy(al.terrace, E, N)
        room = grounds_grade.room(E, N, railway.lake_walk(al, scene, georef), (road > 0.3) & ~terrace)
        rises, wood = [F], np.zeros(F.shape, bool)     # the walk to the lake and the roads keep their own level
        if SITE["station_wood"]:    # from the platform and the walk to the lake
            raised, planted = grounds_grade.station_wood(F, E, N, al.terrace, al.frame[2], al.frame[0], al.level,
                                                         *SITE["station_wood"], land, room)
            rises.append(raised); wood |= planted
        if SITE["train_screen"]:    # and from the train's last approach, from the portal to the platform
            on_main, on_loop = al.main.s <= grounds_grade.SCREEN_MAIN_M, al.loop.s <= grounds_grade.SCREEN_LOOP_M
            route = np.concatenate([al.main.xz[on_main][::-1], al.loop.xz[on_loop][1:]])
            formation = np.concatenate([al.main.level[on_main][::-1], al.loop.level[on_loop][1:]])
            raised, screen = grounds_grade.train_screen(F, E, N, route, formation, (0.0, 0.0), SITE["train_screen"],
                                                        land & ~terrace & (bsd >= 2.0), room)
            rises.append(raised); wood |= screen
        F = np.maximum.reduce(rises)    # each from the ground before them, so neither climbs on the other's batter
        trees = np.where(wood & ~barred & (stream >= 1.0), 1.0, trees)      # dense trees, pines near the station
        # whose floor gives way to the ground round them in tufts (clearings.blend, as a class of its own for the
        # moment, so the floor round them is not blended twice). They stand dense to a line, beyond which there is no
        # canopy for the floor to spread under, so the band lies within it: grass under their outer trees, and along the
        # grass where the boundary wall runs through them, which keeps its trees off; measured past a wood they run on
        # into, whose floor is theirs
        woods = (trees > 0.55) & ((sc <= 2) | (sc == 5))
        sc[wood & woods] = PLANTED
        sc = clearings.blend(sc, PLANTED, RES, 34, (0, 1, 2), inset_m=clearings.ECOTONE_M / 2, within=woods & ~wood)
        sc[sc == PLANTED] = 5
        if SITE["fill_pits"]:   # the hollows the rises enclose are made up too, keeping their trees
            F = grounds_grade.made_up(F, fill(F, ~land), land & unnatural & ~garden_hollow)
    lost = clearings.unaccounted(drawn, trees, accounted)
    if lost.any():
        raise RuntimeError(f"trees removed where no rule or recorded clearing accounts for them: {lost.sum() * RES * RES:.0f}"
                           f" m² round ({E[lost].mean():.0f}, {N[lost].mean():.0f})")
    st_m = (stream < 0) & land & (lake_sd > 2.0)
    Fst = ndi.grey_erosion(ndi.gaussian_filter(F, 2.0), size=5)
    if SITE["graded"]:      # the carved watercourses never rise along their course
        beside = ndi.binary_dilation(river, iterations=2)
        Fst = np.where(beside, np.minimum(Fst, downhill([l[:2] for l in lines], np.where(river & land, Fst, np.inf))), Fst)
    bed = np.where(st_m, np.minimum(F, Fst - 0.3), F)
    surf = np.full(bed.shape, np.nan, np.float32)
    surf[~land] = LAKE
    ring = land & (lake_sd < 5.0) & (bed >= LAKE - 0.05)
    surf[ring] = LAKE
    sring = ndi.binary_dilation(st_m, iterations=1) & np.isnan(surf)
    surf[sring] = (Fst - 0.12)[sring]
    wdelta = np.where(np.isnan(surf), -2.0, surf - bed).astype(np.float32)
    print("pitch level", level, "bed range", bed.min(), bed.max())

    # ---------------- castle layer ----------------
    named = json.load(open(P("build", "castle_named.json")))
    cm = ndi.binary_closing(cmass, structure=disk(1))
    def local_thickness(mask, rmax):
        th_ = np.zeros(mask.shape, np.float32)
        for r in range(1, rmax + 1):
            op = ndi.binary_opening(mask, structure=disk(r))
            th_[op] = r
            if not op.any(): break
        return th_
    th_c = local_thickness(cm, 30) * PG
    dt = ndi.distance_transform_edt(cm) * PG
    yy, xx = np.mgrid[0:cm.shape[0], 0:cm.shape[1]]
    peaks = (dt == ndi.maximum_filter(dt, size=9)) & (dt >= 3.7)
    towers = []
    for (py, px) in zip(*np.nonzero(peaks)):
        r = dt[py, px]; rp = r / PG; d2 = (xx - px) ** 2 + (yy - py) ** 2
        ann = (d2 > (rp + 2.0) ** 2) & (d2 < (rp + 6.0) ** 2); frac = cm[ann].mean()
        ringm = (d2 > (rp + 3.0) ** 2) & (d2 < (rp + 9.0) ** 2) & cm
        wall_half = np.median(dt[ringm]) if ringm.any() else 0.0
        if frac < 0.36 and r <= 9.0 and (wall_half < 0.72 * r or frac < 0.12): towers.append((px, py, r))
    towers.sort(key=lambda t_: -t_[2]); kept = []
    for t_ in towers:
        if all((t_[0] - k[0]) ** 2 + (t_[1] - k[1]) ** 2 > (max(t_[2], k[2]) / PG * 1.6) ** 2 for k in kept): kept.append(t_)
    def w2c(e, nn):
        U, V = world_to_ff(e, nn); u, v = ff_to_g(U, V); return float(u - CX0), float(v - CY0)
    special = {k: w2c(*named[k]) for k in ("AT", "NT", "GT", "WT", "RT")}
    kx, ky = w2c(*named["KEEP"]); gx_, gy_ = w2c(*named["GH"])
    rel = np.zeros(cm.shape, np.float32); rtype = np.zeros(cm.shape, np.uint8)
    low = cm & (th_c < 1.6); wallm = cm & (th_c >= 1.6) & (th_c < 3.3); rngm = cm & (th_c >= 3.3) & (th_c < 7.0); blkm = cm & (th_c >= 7.0)
    hall = cm & (th_c >= 3.3) & (xx > gx_ - 34) & (xx < gx_ + 62) & (yy > gy_ - 32) & (yy < gy_ + 12)
    keep = cm & (((xx - kx) ** 2 + (yy - ky) ** 2) < (22 / PG) ** 2)
    for k, m_ in enumerate((low, wallm, rngm, blkm, hall, keep)):   # later classes override earlier ones
        rel[m_], rtype[m_] = CAS_CLS[k]
    # castle base height (absolute) from the platform field
    cu_n = xx + CX0; cv_n = yy + CY0
    U, V = g_to_ff(cu_n, cv_n)
    li = (V - FY0) * MFF / RES - 0.5; lj = (U - FX0) * MFF / RES - 0.5
    base_c = map_coordinates(Bc, [li, lj], order=1, mode='nearest')
    tower_list = []
    trng = np.random.default_rng(7)
    rel0 = rel.copy(); rtype0 = rtype.copy()
    discs = np.zeros(cm.shape, bool)

    def surround(px, py, rp):
        d2_ = (xx - px) ** 2 + (yy - py) ** 2
        ringm_ = cm & (d2_ > (rp + 1.5) ** 2) & (d2_ < (rp + 7.0) ** 2)
        return float(np.median(rel0[ringm_])) if ringm_.any() else 13.0

    for (px, py, r) in kept:
        if any((px - sx) ** 2 + (py - sy) ** 2 < (12 / PG) ** 2 for sx, sy in special.values()): continue
        inkeep = (px - kx) ** 2 + (py - ky) ** 2 < (24 / PG) ** 2
        b0 = float(base_c[int(py), int(px)])
        rp = r / PG + 0.8
        sur = surround(px, py, rp)
        if inkeep: top, roof = 33.0, 0
        else:
            conical = trng.uniform() < 0.72
            if sur >= 24.0: top = sur + float(trng.uniform(8, 14))       # essay: "eight storeys plus towers"
            else: top = max(sur + 9.0, float(trng.uniform(22, 30)))
            roof = 1 if conical else 0
        tower_list.append([px, py, r, b0 + top, roof])
        discs |= cm & (((xx - px) ** 2 + (yy - py) ** 2) <= rp ** 2)
    # canon-derived special towers (castle page): AT & NT thin columns with overhanging tops, GT trefoil, WT battlements, RT tall cone
    spec_def = {"AT": (3.6, 58.0, 3), "NT": (3.6, 54.0, 4), "GT": (7.0, 30.0, 5), "WT": (4.6, 41.0, 0), "RT": (5.2, 45.0, 1)}
    for k, (sx, sy) in special.items():
        r, top, roof = spec_def[k]
        b0 = float(base_c[int(round(sy)), int(round(sx))])
        tower_list.append([sx, sy, r, b0 + top, roof])
        discs |= cm & (((xx - sx) ** 2 + (yy - sy) ** 2) <= (r / PG + 1.0) ** 2)
    # under the analytic towers the extruded mass takes the surrounding wall's height/type
    rest = cm & ~discs
    idx_r = ndi.distance_transform_edt(~rest, return_distances=False, return_indices=True)
    rel = np.where(discs, rel0[idx_r[0], idx_r[1]], rel0)
    rtype = np.where(discs, rtype0[idx_r[0], idx_r[1]], rtype0).astype(np.uint8)
    # absolute tops (the shader rebuilds them from the base height and CAS_CLS)
    top_abs = base_c + rel
    sd_c = np.where(cm, -ndi.distance_transform_edt(cm), ndi.distance_transform_edt(~cm)) * PG
    nxc = int(round(CW * PG / CRES)); nyc = int(round(CH * PG / CRES))
    cu = (np.arange(nxc) + 0.5) * CRES / PG - 0.5; cv = (np.arange(nyc) + 0.5) * CRES / PG - 0.5
    CU, CV = np.meshgrid(cu, cv)
    c_sd = map_coordinates(sd_c.astype(np.float32), [CV, CU], order=1, mode='nearest')
    # footprint signed distance per mass class: the shader extrudes each class separately and takes the union.
    # Each class is clipped by the one shared outline, so there's no notch where two classes meet on an outer wall.
    ccls = np.argmin(np.abs(rel[..., None] - np.array([c[0] for c in CAS_CLS], np.float32)), -1)
    sd_out = sdf(cm, PG)
    c_cls = np.stack([map_coordinates(np.maximum(sd_out, sdf((ccls == k) | ~cm, PG)), [CV, CU], order=1, mode='nearest')
                      for k in range(len(CAS_CLS))], -1)
    towers_world = []
    for (px, py, r, topz, roof) in tower_list:
        e, nn = ff_to_world(*g_to_ff(px + CX0, py + CY0))
        towers_world.append([round(float(e), 2), round(float(nn), 2), round(float(r), 2), round(float(topz), 2), int(roof)])
    print("towers", len(towers_world), "castle tex", nxc, nyc, "top range", top_abs[cm].min(), top_abs[cm].max())
    print("train", train)

    # ---------------- forest layer over the whole Forest map (map-aligned, 8 m) ----------------
    nfx = int(round(FF_W * MFF / FRES)); nfy = int(round(FF_H * MFF / FRES))
    fu = (np.arange(nfx) + 0.5) * FRES / MFF; fv = (np.arange(nfy) + 0.5) * FRES / MFF
    FUU, FVV = np.meshgrid(fu, fv)
    dense = ndi.gaussian_filter((cff == C_DENSE).astype(np.float32), 1.2)
    semi = ndi.gaussian_filter((cff == C_UNDER).astype(np.float32), 1.2)
    # its woods are its green: its black ink draws paths, tracks, rivers' banks, lettering and symbols, never trees
    fden = np.clip(dense + 0.5 * semi, 0, 1)
    f_den = sample(fden, FUU - 0.5, FVV - 0.5)
    ffc2 = np.zeros(cff.shape, np.float32); ffc2[cff == C_LAWN] = 1.0
    ffc2 = ndi.gaussian_filter(ndi.median_filter(ffc2, size=5), 1.0)
    f_field = sample(ffc2, FUU - 0.5, FVV - 0.5)
    # treeline: fade forest density above ~520 m
    Ef, Nf = ff_to_world(FUU, FVV)
    hf = dem_sample(mos, Ef + CASTLE_E, Nf + CASTLE_N)
    f_den = f_den * np.clip((640.0 - hf) / 140.0, 0.0, 1.0)
    rules = SITE.get("forest")
    ways = forest_extent.ways(osm, (CASTLE_E, CASTLE_N), rules)
    f_keep = forest_extent.keep(rules, Ef, Nf, ways, FRES)
    if rules is not None:       # round the buildings the Forest covers, their homesteads' clearings
        f_home = forest_extent.homesteads(Ef, Nf, ways, FRES)
        standing = f_den * forest_extent.ends(rules, Ef, Nf, ways, FRES)
        glades += clearings.record("homestead", "forest-homesteads", "forest", f_home, Ef, Nf,
                                   standing * (1 - f_home) > 0.05)
    if SITE["aragog_hollow"]:   # moved, so the bare patch his map draws round his crater takes the Forest round it
        patch = forest_places.crater(f_den, Ef, Nf, ff_to_world(*forest_places.DRAWN["Aragog's hollow"]))
        f_den, f_field = forest_places.restored(f_den, f_field, patch)
        print(f"aragog's crater as drawn: {patch.sum() * FRES ** 2 / 1e4:.2f} ha given the Forest round it")
        f_hollow = forest_places.hollow_clearing(Ef, Nf, SITE["aragog_hollow"], FRES)     # and the hollow is cleared
        glades += clearings.record("aragogs-hollow", "aragogs-hollow", "forest", f_hollow, Ef, Nf,
                                   f_den * f_keep * (1 - f_hollow) > 0.05)
        f_keep = f_keep * f_hollow
    f_den = f_den * f_keep
    if SITE["grawp_clearing"]:  # cleared but for a few small pines
        f_grawp = forest_places.grawp_clearing(Ef, Nf, SITE["grawp_clearing"], FRES)
        glades += clearings.record("grawps-clearing", "grawps-clearing", "forest", f_grawp, Ef, Nf,
                                   f_den - forest_places.with_pines(f_den, f_grawp) > 0.05)
        f_den = forest_places.with_pines(f_den, f_grawp)
    print("forest layer", nfx, nfy, "forest elev p50/p90", np.percentile(hf[fden[(FVV.astype(int)).clip(0, FF_H - 1), (FUU.astype(int)).clip(0, FF_W - 1)] > 0.5], [50, 90]))

    # ---------------- save ----------------
    np.save(P("build", "final_bed.npy"), bed.astype(np.float32))
    np.save(P("build", "final_wdelta.npy"), wdelta)
    def q8sdf(sd, k=10.0): return np.clip(np.round(128 + sd * k), 0, 255).astype(np.uint8)
    u8 = lambda a: np.clip(np.round(a * 255), 0, 255).astype(np.uint8)
    np.save(P("build", "final_L4.npy"), np.stack([u8(road), u8(trees), q8sdf(hedge_sd), q8sdf(bsd)], -1))     # LINEAR
    np.save(P("build", "final_L2.npy"), np.stack([q8sdf(wall_sd), q8sdf(stream)], -1))                        # LINEAR
    np.save(P("build", "final_LN.npy"), np.stack([sc, bcode], -1).astype(np.uint8))                           # NEAREST
    np.save(P("build", "final_Csd.npy"), q8sdf(c_sd))                                                         # LINEAR
    np.save(P("build", "final_Ccls.npy"), q8sdf(c_cls))                                                       # LINEAR, one channel per CAS_CLS
    np.save(P("build", "final_cbase.npy"), Bc.astype(np.float32))                                             # castle base height, local grid
    np.save(P("build", "final_F2.npy"), np.stack([u8(f_den), u8(f_field)], -1))                               # LINEAR
    np.save(P("build", "final_Fclear.npy"), f_keep == 0)                       # cleared by the site's forest rules

    def affine_from(fn_world, nx, ny):
        """world (e,n) -> uv affine, fitted on texel centres; fn_world(i, j) gives world coords of texel centre."""
        pts = [(0, 0), (0, nx - 1), (ny - 1, 0), (ny - 1, nx - 1)]
        A = []; bu = []; bv = []
        for i, j in pts:
            e, nn = fn_world(i, j); A.append([float(e), float(nn), 1.0]); bu.append((j + 0.5) / nx); bv.append((i + 0.5) / ny)
        A = np.array(A); cu_ = np.linalg.lstsq(A, np.array(bu), rcond=None)[0]; cv_ = np.linalg.lstsq(A, np.array(bv), rcond=None)[0]
        return [list(map(float, cu_)), list(map(float, cv_))]
    meta = {
        "lake": LAKE, "pitch": {"c": [pc_e, pc_n], "a": pa, "b": pb, "brg": pbrg, "level": level},
        "local": {"nx": NX, "ny": NY, "aff": affine_from(lambda i, j: ff_to_world(*local_texel_ff(i, j)), NX, NY)},
        "castle": {"nx": nxc, "ny": nyc, "aff": affine_from(lambda i, j: g2w(CX0 + (j + 0.5) * CRES / PG - 0.5, CY0 + (i + 0.5) * CRES / PG - 0.5), nxc, nyc),
                   "cls": [list(c) for c in CAS_CLS]},
        "forest": {"nx": nfx, "ny": nfy, "aff": affine_from(lambda i, j: ff_to_world((j + 0.5) * FRES / MFF, (i + 0.5) * FRES / MFF), nfx, nfy)},
        "towers": towers_world, "train": train, "hills": hills,
        "named": named, "clearings": glades,
    }
    json.dump(meta, open(P("build", "final_meta.json"), "w"), indent=1)
    # previews
    vis = np.zeros((NY, NX, 3), np.float32) + [120, 170, 90]
    vis[sc == 2] = [150, 140, 80]; vis[sc == 5] = [60, 100, 50]; vis[sc == 4] = [100, 120, 70]; vis[sc == 1] = [110, 200, 90]
    vis[sc == 6] = [70, 220, 70]; vis[sc == 7] = [200, 190, 150]
    vis = vis * (1 - 0.6 * trees[..., None]) + np.array([20, 70, 30]) * 0.6 * trees[..., None]
    vis[road > 0.5] = [200, 180, 130]
    vis[wdelta > 0] = [60, 120, 220]
    vis[hedge_sd < 0.6] = [90, 70, 50]
    vis[wall_sd < 0.8] = [60, 60, 60]
    vis[bsd < 0] = [170, 70, 40]
    save_rgb(vis, "final_local.png")


if __name__ == "__main__":
    main()
