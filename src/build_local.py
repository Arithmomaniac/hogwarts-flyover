"""Build local (2 m, map-aligned) layers: water SDF, buildings SDF + heights, roads, trees, surface class,
and the terraformed height field around the grounds and Hogsmeade."""
from common import *
from features import local_grid, NX, NY, FX0, FY0, RES

LAKE_LEVEL = 153.0

# Decorations (title, compass, scale bar) to blank in native map pixels: (x0, y0, x1, y1)
DECOR_G = [(990, 2170, 1900, 2300), (1580, 1840, 1960, 2180), (740, 2270, 900, 2330)]
DECOR_H = [(1260, 2150, 1660, 2240), (1290, 1070, 1650, 1430), (1430, 1460, 1590, 1520)]


def blank(mask, boxes):
    for x0, y0, x1, y1 in boxes: mask[y0:y1, x0:x1] = False
    return mask


def sdf(mask, px_m):
    """Signed distance (metres), negative inside."""
    if not mask.any(): return np.full(mask.shape, 1e3, np.float32)
    out = ndi.distance_transform_edt(~mask) * px_m
    ins = ndi.distance_transform_edt(mask) * px_m
    return np.where(mask, -ins + 0.5 * px_m, out - 0.5 * px_m).astype(np.float32)


def sample(img, u, v, order=1, cval=0.0):
    return map_coordinates(img.astype(np.float32), [v, u], order=order, mode='constant', cval=cval)


def main():
    UU, VV = local_grid()
    E, N = ff_to_world(UU, VV)
    mg = dict(np.load(P("build", "native_g.npz"))); mh = dict(np.load(P("build", "native_h.npz")))
    for k in ("tree_ink", "wall", "build", "earth", "white", "wood", "water"):
        blank(mg[k], DECOR_G); blank(mh[k], DECOR_H)
    gu, gv = ff_to_g(UU, VV); hu, hv = ff_to_h(UU, VV)
    inG = (gu >= 0) & (gu < G_W - 1) & (gv >= 0) & (gv < G_H - 1)
    inH = (hu >= 0) & (hu < H_W - 1) & (hv >= 0) & (hv < H_H - 1)
    useH = inH
    useG = inG & ~inH
    useF = ~(inG | inH)
    pg = 0.72

    # ---------- castle region (handled by castle layer) ----------
    cmass = np.load(P("build", "castle_mass_g.npy"))      # sub-image of grounds at (560:1180, 1030:1530)
    cm_full = np.zeros((G_H, G_W), bool); cm_full[560:1180, 1030:1530] = cmass
    castle_zone = ndi.binary_dilation(cm_full, iterations=8)

    # ---------- water ----------
    wg = mg["water"].copy(); wh = mh["water"].copy()
    lab, n = ndi.label(wg); sizes = ndi.sum(wg, lab, range(1, n + 1)); lake_id = 1 + int(np.argmax(sizes))
    lake = lab == lake_id
    holes = ndi.binary_fill_holes(lake) & ~lake
    hl, hn = ndi.label(holes)
    green = mg["tree_ink"] | (np.load(P("build", "cls_g.npy")) == C_LAWN)
    island = np.zeros_like(lake)
    for i in range(1, hn + 1):
        reg = hl == i
        if reg.sum() > 400 and (green & reg).sum() > 0.3 * reg.sum(): island |= reg
    lake_full = ndi.binary_fill_holes(lake) & ~island
    lake_full = ndi.binary_closing(lake_full, iterations=2) & ~island
    wg = (wg & ~lake) | lake_full
    wg = ndi.binary_opening(wg, iterations=1) | lake_full
    wh = ndi.binary_opening(wh, iterations=1)
    def drop_small(m, min_px):
        lab_, n_ = ndi.label(m)
        if not n_: return m
        sz = ndi.sum(m, lab_, range(1, n_ + 1))
        return np.isin(lab_, 1 + np.nonzero(sz >= min_px)[0])
    wg = drop_small(wg, 80); wh = drop_small(wh, 80)
    sd_wg = sdf(wg, pg); sd_wh = sdf(wh, pg)
    sd_lake = sdf(lake_full, pg)
    # forest-map water (2.874 m/px)
    cff = np.load(P("build", "cls_ff.npy")); wf = drop_small(cff == C_WATER, 6)
    sd_wf = sdf(wf, MFF)
    water_sd = np.where(useH, sample(sd_wh, hu, hv, cval=50), np.where(useG, sample(sd_wg, gu, gv, cval=50), sample(sd_wf, UU, VV, cval=50)))
    lake_sd = np.where(inG, sample(sd_lake, gu, gv, cval=200), 200.0)
    # real River Meig (OSM) where outside the essay maps' own water features
    osm = json.load(open(P("osm.json")))
    from PIL import ImageDraw
    riv = Image.new("L", (NX, NY), 0); dr = ImageDraw.Draw(riv)
    for el in osm["elements"]:
        t = el.get("tags", {}); g = el.get("geometry")
        if el["type"] != "way" or not g or t.get("waterway") != "river": continue
        xy = np.array(to_tm.transform([p["lon"] for p in g], [p["lat"] for p in g])).T
        U, V = world_to_ff(xy[:, 0] - CASTLE_E, xy[:, 1] - CASTLE_N)
        px = (U - FX0) * MFF / RES; py = (V - FY0) * MFF / RES
        if px.max() < -50 or py.max() < -50 or px.min() > NX + 50 or py.min() > NY + 50: continue
        dr.line(list(zip(px, py)), fill=255, width=6)
    river = np.asarray(riv) > 0
    river_sd = sdf(river, RES)
    water_sd = np.minimum(water_sd, river_sd + 1.0)
    np.save(P("build", "L_water_sd.npy"), water_sd.astype(np.float32))
    np.save(P("build", "L_lake_sd.npy"), lake_sd.astype(np.float32))
    np.save(P("build", "L_river.npy"), river)

    # ---------- buildings (outside castle) ----------
    bg = (mg["build"] | mg["white"]) & ~castle_zone
    bh_ = mh["build"] | mh["white"]
    def comps_heights(b, white):
        b = ndi.binary_opening(b, iterations=1)
        lab, n = ndi.label(b)
        area = ndi.sum(b, lab, range(1, n + 1)) * pg * pg
        wfrac = ndi.sum(white, lab, range(1, n + 1)) * pg * pg / np.maximum(area, 1)
        eave = np.zeros(n + 1, np.float32); glass = np.zeros(n + 1, bool)
        for i in range(n):
            a = area[i]
            if a < 12: eave[i + 1] = 0
            elif wfrac[i] > 0.5: eave[i + 1] = 3.2; glass[i + 1] = True
            elif a < 70: eave[i + 1] = 4.2
            elif a < 160: eave[i + 1] = 5.2
            elif a < 400: eave[i + 1] = 6.5
            else: eave[i + 1] = 8.0
        keep = eave[lab] > 0
        return keep, eave[lab], glass[lab], lab
    kg, eg, glg, labg = comps_heights(bg, mg["white"])
    kh, eh, glh, labh = comps_heights(bh_, mh["white"])
    bsd_g = sdf(kg, pg); bsd_h = sdf(kh, pg)
    bsd = np.where(useH, sample(bsd_h, hu, hv, cval=50), np.where(useG, sample(bsd_g, gu, gv, cval=50), 50.0))
    # eave height: nearest from the nearest building pixel (dilate heights outward so edges are defined)
    def spread(vals, keep):
        idx = ndi.distance_transform_edt(~keep, return_distances=False, return_indices=True)
        return vals[idx[0], idx[1]]
    eg2 = spread(eg, kg); eh2 = spread(eh, kh)
    eave = np.where(useH, sample(eh2, hu, hv, order=0), np.where(useG, sample(eg2, gu, gv, order=0), 0.0))
    glass = np.where(useH, sample(spread(glh.astype(np.float32), kh), hu, hv, order=0), np.where(useG, sample(spread(glg.astype(np.float32), kg), gu, gv, order=0), 0.0)) > 0.5
    eave = np.where(bsd < 6.0, eave, 0.0)
    np.save(P("build", "L_bsd.npy"), bsd.astype(np.float32)); np.save(P("build", "L_eave.npy"), eave.astype(np.float32)); np.save(P("build", "L_glass.npy"), glass)

    # ---------- roads / paving ----------
    road_g = mg["earth"].astype(np.float32); road_h = mh["earth"].astype(np.float32)
    road = np.where(useH, sample(ndi.gaussian_filter(road_h, 0.8), hu, hv), np.where(useG, sample(ndi.gaussian_filter(road_g, 0.8), gu, gv), 0.0))
    road_f = sample(ndi.gaussian_filter(((cff == C_EARTH) | (cff == C_UNDER) & False).astype(np.float32), 0.5), UU, VV)
    road = np.where(useF, road_f, road)
    np.save(P("build", "L_road.npy"), np.clip(road, 0, 1).astype(np.float32))

    # ---------- walls ----------
    wall_g = sdf(mg["wall"], pg); wall_h = sdf(mh["wall"], pg)
    wall_sd = np.where(useH, sample(wall_h, hu, hv, cval=50), np.where(useG, sample(wall_g, gu, gv, cval=50), 50.0))
    np.save(P("build", "L_wall_sd.npy"), wall_sd.astype(np.float32))

    # ---------- trees ----------
    def tree_density(m):
        ink = m["tree_ink"].astype(np.float32)
        d = ndi.gaussian_filter(ink, 3.0)
        dense = (m["under"]).astype(np.float32)
        t = np.clip((d - 0.10) / 0.22, 0, 1)
        t = np.maximum(t, ndi.gaussian_filter(dense, 2.0) * 0.6)
        excl = ndi.binary_dilation(m["earth"] | m["build"] | m["water"] | m["white"] | m["wood"] | m["wall"], iterations=3)
        return np.where(excl, 0, t)
    tg = tree_density(mg); th = tree_density(mh)
    trees = np.where(useH, sample(th, hu, hv), np.where(useG, sample(tg, gu, gv), 0.0))
    dense_f = ndi.gaussian_filter((cff == C_DENSE).astype(np.float32), 1.0)
    semi_f = ndi.gaussian_filter((cff == C_UNDER).astype(np.float32), 1.0)
    tf = np.clip(dense_f * 1.0 + semi_f * 0.55, 0, 1)
    tf_l = sample(tf, UU, VV)
    # feather detailed-map trees into the Forest-map trees near the inset rectangle edges
    dG = np.where(inG, np.minimum.reduce([gu, G_W - gu, gv, G_H - gv]) * pg, 0.0)
    dH = np.where(inH, np.minimum.reduce([hu, H_W - hu, hv, H_H - hv]) * pg, 0.0)
    wdet = np.clip(np.maximum(dG, dH) / 45.0, 0, 1)
    trees = np.where(useF, tf_l, trees * wdet + tf_l * (1 - wdet))
    trees = np.where((water_sd < 1.5) | (bsd < 1.5), 0, trees)
    np.save(P("build", "L_trees.npy"), np.clip(trees, 0, 1).astype(np.float32))

    # ---------- surface class (nearest) ----------
    cg = np.load(P("build", "cls_g.npy")); chh = np.load(P("build", "cls_h.npy"))
    def surf(c, m):
        s = np.full(c.shape, 1, np.uint8)            # 1 rough grass / meadow
        s[c == C_SCRUB] = 2                          # scrub / heather hillside
        s[c == C_UNDER] = 5                          # undergrowth / cultivated hatching
        s[m["earth"]] = 3
        return s
    sg = surf(cg, mg); sh = surf(chh, mh)
    sf = np.full(cff.shape, 2, np.uint8); sf[cff == C_LAWN] = 1; sf[(cff == C_DENSE) | (cff == C_UNDER)] = 6; sf[cff == C_EARTH] = 3
    surfc = np.where(useH, sample(sh, hu, hv, order=0), np.where(useG, sample(sg, gu, gv, order=0), sample(sf, UU, VV, order=0))).astype(np.uint8)
    np.save(P("build", "L_surf.npy"), surfc)

    # ---------- inside-grounds mask (enclosed by the boundary wall; Forest acts as the northern barrier) ----------
    wl = ndi.binary_dilation(wall_sd < 2.5, iterations=11) | (ndi.gaussian_filter(trees, 4.0) > 0.55)
    lab, n = ndi.label(~wl)
    c_iy, c_ix = np.unravel_index(np.argmin((E + 120) ** 2 + (N + 120) ** 2), E.shape)
    inside = lab == lab[c_iy, c_ix]
    inside = ndi.binary_dilation(inside, iterations=11) & ~(wall_sd < 1.0)
    print("inside-grounds fraction of local grid:", inside.mean())
    np.save(P("build", "L_inside.npy"), inside)

    # preview
    vis = np.zeros((NY, NX, 3), np.float32) + [120, 170, 90]
    vis[surfc == 2] = [150, 140, 80]; vis[surfc == 6] = [60, 100, 50]; vis[surfc == 5] = [90, 120, 60]
    vis[inside] = vis[inside] * 0.8 + np.array([120, 220, 90]) * 0.2
    vis = vis * (1 - 0.6 * trees[..., None]) + np.array([20, 70, 30]) * 0.6 * trees[..., None]
    vis[road > 0.5] = [200, 180, 130]
    vis[water_sd < 0] = [60, 120, 220]
    vis[wall_sd < 0.8] = [80, 80, 80]
    vis[bsd < 0] = [170, 70, 40]; vis[(bsd < 0) & glass] = [230, 240, 255]
    save_rgb(vis, "local_layers.png")


if __name__ == "__main__":
    main()
