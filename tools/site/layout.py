"""White Hound's registered layout in castle-relative metres and the earthworks model that places it on real terrain,
ported from the location review (2026-10-31).

The layout comes from the original build's traced map layers (the pinned original intermediates) through the project's
registration (georef.json: 25.1 degree compass rotation, fixed map scales). Placing it with its castle at a TM anchor,
the model engineers five features against the terrain's bed (existing water treated as 10 m below its surface):
  lake      cut the map outline's basin (the project's profile, 0.8-12.3 m deep) below lake level L
  rim       fill the 5-45 m ring outside the shore to L + 1 m so the lake is held (a dam where the ground is low)
  castle    fill the castle footprint up to the platform P = L + 120 ft; cut anything over P + 10 m
  pitch     level the stands' ellipse at its median height, clamped to L + 2 .. P - 3
  station   level the station terrace at its median height, clamped to L + 10 .. L + 60
L minimises the lake, rim and castle volumes (convex and piecewise linear in L; bisection on its slope).
"""
import json
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi

from tools.inputs import require
from tools.site.sun import CLIFF
from common import G_W, GX0, GY0, HX0, HY0, SGX, SGY, SHX, SHY, world_to_ff

RES = 5.0                                   # mask raster (m)
XR, ZR = (-1800.0, 1000.0), (-1800.0, 1100.0)
C_LAWN, C_DENSE, C_UNDER, C_BUILD = 0, 2, 3, 5
MASKS = ("lake", "castle", "pitch", "station", "village", "lawn", "forest")
POINTS = ("gates", "hagrid", "aragog", "f6c")


def basin_depth(d):
    """The project's lake basin profile: 0.8 + 11.5 (1 - exp(-d/50)) m at inward distance d (REBUILD.md)."""
    return 0.8 + 11.5 * (1 - np.exp(-d / 50.0))


def _from_map(mask, X, Z, to_px):
    U, V = world_to_ff(X, Z)
    u, v = to_px(U, V)
    u, v = np.round(u).astype(int), np.round(v).astype(int)
    ok = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
    out = np.zeros(X.shape, bool)
    out[ok] = mask[v[ok], u[ok]]
    return out


@lru_cache(maxsize=1)
def template():
    """The layout: feature masks on a 5 m castle-relative grid, sample points at 25 m and 10 m spacing, and the
    essays' marker points (feature_offsets.json)."""
    root = require("original-intermediates")
    meta = json.loads((root / "build" / "final_meta.json").read_text(encoding="utf-8"))
    offs = json.loads((root / "feature_offsets.json").read_text(encoding="utf-8"))
    xs, zs = np.arange(XR[0], XR[1] + 1e-6, RES), np.arange(ZR[0], ZR[1] + 1e-6, RES)
    X, Z = np.meshgrid(xs, zs)                                      # row 0 = south
    grounds = lambda U, V: ((U - GX0) * SGX, (V - GY0) * SGY)       # noqa: E731
    hogs = lambda U, V: ((U - HX0) * SHX, (V - HY0) * SHY)          # noqa: E731

    lake = ndi.binary_opening(_from_map(np.load(root / "lake_mask_grounds.npy"), X, Z, grounds), iterations=1)
    lab, n = ndi.label(lake)
    lake = ndi.binary_fill_holes(lab == 1 + int(np.argmax(ndi.sum(lake, lab, range(1, n + 1)))))  # the main body
    din = ndi.distance_transform_edt(lake) * RES
    dout = ndi.distance_transform_edt(~lake) * RES

    cm = np.load(root / "build" / "castle_mass_g.npy")              # the grounds window from (1030, 560)
    castle = ndi.binary_closing(_from_map(cm, X, Z, lambda U, V: ((U - GX0) * SGX - 1030, (V - GY0) * SGY - 560)),
                                iterations=2)
    cls_g = np.load(root / "build" / "cls_g.npy")
    forest = _from_map((cls_g == C_DENSE) | (cls_g == C_UNDER), X, Z, grounds)
    lawn_g = _from_map(cls_g == C_LAWN, X, Z, grounds)
    bld_h = _from_map(np.load(root / "build" / "cls_h.npy") == C_BUILD, X, Z, hogs)
    village = ndi.binary_closing(bld_h, iterations=4) | ndi.binary_dilation(bld_h, iterations=3)
    vlab, vn = ndi.label(village)
    village = np.isin(vlab, 1 + np.nonzero(ndi.sum(village, vlab, range(1, vn + 1)) > 400)[0])   # clusters over 1 ha

    pc = np.array(meta["pitch"]["c"], float)
    brg = np.radians(271.25)
    ux, uz = np.sin(brg), np.cos(brg)
    dx, dz = X - pc[0], Z - pc[1]
    pitch = ((dx * ux + dz * uz) / 101.6) ** 2 + ((-dx * uz + dz * ux) / 54.6) ** 2 <= 1
    ta, tb = np.array([246.4, 234.0]), np.array([375.8, 92.4])       # the coaches along the platform (meta.train)
    seg = tb - ta
    sl = np.hypot(*seg)
    t = ((X - ta[0]) * seg[0] + (Z - ta[1]) * seg[1]) / sl
    station = (t > -20) & (t < sl + 20) & (np.abs((X - ta[0]) * seg[1] - (Z - ta[1]) * seg[0]) / sl < 30)

    cls_ff = np.load(root / "build" / "cls_ff.npy")
    U, V = world_to_ff(X, Z)
    ui, vi = np.round(U).astype(int), np.round(V).astype(int)
    ok = (ui >= 0) & (ui < cls_ff.shape[1]) & (vi >= 0) & (vi < cls_ff.shape[0])
    forest[ok] |= np.isin(cls_ff[vi[ok], ui[ok]], (C_DENSE, C_UNDER))
    r = np.hypot(X, Z)
    bearing = np.mod(np.degrees(np.arctan2(X, Z)), 360)
    lawn = lawn_g & ~castle & ~lake & ~pitch & (r > 60) & (r < 700) & (bearing > 180) & (bearing < 330)
    forest_near = forest & (r > 250) & (r < 900)

    out = {}
    for step in (25, 10):
        k = round(step / RES)

        def pts(mask):
            sub = mask[::k, ::k]
            return np.stack([X[::k, ::k][sub], Z[::k, ::k][sub]], 1)

        li = pts(lake)
        out[f"lake_{step}"] = li
        out[f"lake_d_{step}"] = basin_depth(ndi.map_coordinates(din, [(li[:, 1] - ZR[0]) / RES, (li[:, 0] - XR[0]) / RES],
                                                                order=1))
        out[f"rim_{step}"] = pts((dout > 5) & (dout <= 45))
        for name, mask in (("castle", castle), ("pitch", pitch), ("station", station), ("village", village),
                           ("lawn", lawn), ("forest", forest_near)):
            out[f"{name}_{step}"] = pts(mask)
    for name in ("gates", "hagrid", "front", "station", "willow", "greenh", "aragog", "f6c", "cave", "stile", "mtn1",
                 "mtn2", "mtn3", "face", "db", "honeyd", "shack"):
        out["pt_" + name] = np.array(offs[name], float)
    out["village_c"] = np.array([np.mean(X[village]), np.mean(Z[village])])
    out["masks"] = np.stack([lake, castle, pitch, station, village, lawn, forest]).astype(np.uint8)
    out["mask_extent"] = np.array([XR[0], XR[1], ZR[0], ZR[1], RES])
    return out


def _sample(grid, anchors, offsets):
    return grid.sample(anchors[:, 0:1] + offsets[None, :, 0], anchors[:, 1:2] + offsets[None, :, 1])


def bed_of(dem, water):
    """The ground with existing water replaced by a bed 10 m below its surface."""
    from tools.site.terrain import Grid

    return Grid((dem.a - 10.0 * (water.a > 0.5)).astype(np.float32), dem.E0, dem.N0, dem.step)


def slope_of(dem):
    from tools.site.terrain import Grid

    gy, gx = np.gradient(np.nan_to_num(dem.a.astype(np.float64)), dem.step)
    return Grid(np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32), dem.E0, dem.N0, dem.step)


def polar(grid, anchors, centre, b0, b1, r0, r1, nb=15, nr=16):
    """Heights on a fan of bearings b0..b1 and distances r0..r1 around anchor + centre."""
    b = np.radians(np.linspace(b0, b1, nb))
    r = np.linspace(r0, r1, nr)
    dE, dN = (np.sin(b)[:, None] * r[None, :]).ravel(), (np.cos(b)[:, None] * r[None, :]).ravel()
    return grid.sample(anchors[:, 0][:, None] + centre[0] + dE[None, :], anchors[:, 1][:, None] + centre[1] + dN[None, :])


def evaluate(dem, water, anchors, step=25):
    """Earthworks, the best lake level, the grounds' up-and-down ordering and landforms for TM anchors."""
    A = np.atleast_2d(np.asarray(anchors, float))
    tpl, s, a = template(), f"_{step}", float(step) ** 2
    bed, slope = bed_of(dem, water), slope_of(dem)
    zl, dl = _sample(bed, A, tpl["lake" + s]), tpl["lake_d" + s][None, :]
    zr, zc = _sample(bed, A, tpl["rim" + s]), _sample(bed, A, tpl["castle" + s])
    zp, zs = _sample(bed, A, tpl["pitch" + s]), _sample(bed, A, tpl["station" + s])
    zv, sv = _sample(dem, A, tpl["village" + s]), _sample(slope, A, tpl["village" + s])
    zw, zf = _sample(dem, A, tpl["lawn" + s]), _sample(dem, A, tpl["forest" + s])
    pt = {k: dem.sample(A[:, 0] + tpl["pt_" + k][0], A[:, 1] + tpl["pt_" + k][1]) for k in POINTS}

    # the lake level: the zero of dV/dL = -#lake cut + #rim fill + #castle fill - #castle cut (counts monotone in L)
    lo = np.minimum(np.nanmin(zl + dl, 1), np.nanmin(zr - 1, 1)) - 50
    hi = np.maximum(np.nanmax(zl + dl, 1), np.nanmax(zc - CLIFF, 1)) + 50
    for _ in range(32):
        m = 0.5 * (lo + hi)
        mc = m[:, None]
        g = (-np.sum(zl + dl > mc, 1) + np.sum(zr - 1 < mc, 1) + np.sum(zc - CLIFF < mc, 1)
             - np.sum(zc - CLIFF - 10 > mc, 1))
        lo, hi = np.where(g < 0, m, lo), np.where(g < 0, hi, m)
    Lv = 0.5 * (lo + hi)
    volumes = _volumes(Lv, zl, dl, zr, zc, zp, zs, a, A)
    return volumes | _ordering(Lv, volumes["Ls"], zv, sv, zw, zf, pt, dem, A, tpl)


def earthworks_at(dem, water, anchor, level, step=10):
    """The five features' earthworks and the ordering with the lake held at `level` instead of its best level."""
    A = np.atleast_2d(np.asarray(anchor, float))
    tpl, s, a = template(), f"_{step}", float(step) ** 2
    bed = bed_of(dem, water)
    zv = _sample(dem, A, tpl["village" + s])
    pt = {k: dem.sample(A[:, 0] + tpl["pt_" + k][0], A[:, 1] + tpl["pt_" + k][1]) for k in POINTS}
    Lv = np.full(len(A), float(level))
    volumes = _volumes(Lv, _sample(bed, A, tpl["lake" + s]), tpl["lake_d" + s][None, :], _sample(bed, A, tpl["rim" + s]),
                       _sample(bed, A, tpl["castle" + s]), _sample(bed, A, tpl["pitch" + s]),
                       _sample(bed, A, tpl["station" + s]), a, A)
    return volumes | _ordering(Lv, volumes["Ls"], zv, _sample(slope_of(dem), A, tpl["village" + s]),
                               _sample(dem, A, tpl["lawn" + s]), _sample(dem, A, tpl["forest" + s]), pt, dem, A, tpl)


def _volumes(Lv, zl, dl, zr, zc, zp, zs, a, A):
    Lc, P = Lv[:, None], Lv + CLIFF
    Pc = P[:, None]
    cut_l, fill_r = np.maximum(0, zl - (Lc - dl)), np.maximum(0, Lc + 1 - zr)
    fill_c, cut_c = np.maximum(0, Pc - zc), np.maximum(0, zc - (Pc + 10))
    Lp, Ls = np.clip(np.nanmedian(zp, 1), Lv + 2, P - 3), np.clip(np.nanmedian(zs, 1), Lv + 10, Lv + 60)
    dp, ds = zp - Lp[:, None], zs - Ls[:, None]
    out = {"E": A[:, 0], "N": A[:, 1], "L": Lv, "P": P, "Lp": Lp, "Ls": Ls,
           "V_lake": a * np.nansum(cut_l, 1), "V_rim": a * np.nansum(fill_r, 1),
           "V_castle": a * np.nansum(fill_c + cut_c, 1), "V_pitch": a * np.nansum(np.abs(dp), 1),
           "V_station": a * np.nansum(np.abs(ds), 1),
           "max_cut": np.nanmax(np.concatenate([cut_l, cut_c, np.maximum(dp, 0), np.maximum(ds, 0)], 1), 1),
           "max_fill": np.nanmax(np.concatenate([fill_r, fill_c, np.maximum(-dp, 0), np.maximum(-ds, 0)], 1), 1),
           "lake_relief": np.nanmax(zl, 1) - np.nanmin(zl, 1)}
    out["V_total"] = sum(out[k] for k in ("V_lake", "V_rim", "V_castle", "V_pitch", "V_station"))
    return out


def _ordering(Lv, Ls, zv, sv, zw, zf, pt, dem, A, tpl):
    """Metres by which the essays' up-and-down relationships fail (gates < Hagrid's < castle; village < gates; Forest
    floor at least 20 ft below the castle; Aragog's hollow below it; front lawns and station not above it), and the
    landforms: relief W-NNW of the village and E-NE of the castle, and the castle's height over the valley floor."""
    P = Lv + CLIFF
    zv_med = np.nanmedian(zv, 1)
    o = {"o_gates": np.maximum(0, pt["gates"] - (P - 10)),
         "o_hagrid": np.maximum(0, pt["hagrid"] - (P - 3)) + np.maximum(0, pt["gates"] - pt["hagrid"]),
         "o_village": np.maximum(0, zv_med - (pt["gates"] - 5)),
         "o_forest": np.maximum(0, np.nanmedian(zf, 1) - (P - 6)),
         "o_aragog": np.maximum(0, pt["aragog"] - (P - 10)),
         "o_lawn": np.maximum(0, np.nanpercentile(zw, 90, axis=1) - (P + 3)),
         "o_station": np.maximum(0, Ls - (P + 5))}
    return {**o, "order": sum(o.values()), "village_z": zv_med, "village_slope": np.nanmean(sv, 1),
            "village_steep": np.nanmean(sv > 15, 1),
            "nw_relief": np.nanmax(polar(dem, A, tpl["village_c"], 260, 345, 1500, 6000), 1) - zv_med,
            "ne_relief": np.nanmax(polar(dem, A, (0, 0), 10, 100, 1000, 6000), 1) - P,
            "shelf": P - np.nanpercentile(polar(dem, A, (0, 0), 0, 350, 300, 2500, nb=36, nr=10), 5, axis=1)}
