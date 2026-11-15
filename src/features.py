"""Extract feature masks from the essay's grounds + Hogsmeade maps (native 0.72 m/px) and
resample everything into the local 2 m map-aligned grid."""
import sys
from common import *

FX0, FX1, FY0, FY1 = 900.0, 1740.0, 2560.0, 3320.0   # local frame extent in Forest-map px
RES = 2.0
NX = int(round((FX1 - FX0) * MFF / RES)); NY = int(round((FY1 - FY0) * MFF / RES))
PX_M2 = 0.72 ** 2                                    # the essay maps' pixel, m^2
# White Hound outlines landforms (the hillocks 4f, 4g and 4h, the howe 4i) and some paths with dashed lines: short solid
# strokes two or three pixels thick, each no more than DASH_PX, most of whose pixels lie in such thick parts; his tree
# symbols are spiky stars of 80 px or more in one-pixel lines, and his woods scribbles of one-pixel strokes
DASH_PX, DASH_SOLID = 40, 0.6
PLOT_M2 = 1000.0                                     # his hatching (the vegetable patch, 5c) in plots this big or more;
                                                     # smaller specks of its colour fringe other ink
PLOT_GAP_PX = 3                                      # the hatching's strokes and the paler strokes across it lie closer


def local_grid():
    uu = FX0 + (np.arange(NX) + 0.5) * RES / MFF
    vv = FY0 + (np.arange(NY) + 0.5) * RES / MFF
    return np.meshgrid(uu, vv)


def dashes(ink):
    """The strokes of black `ink` that are the dashes of a dashed line (DASH_PX, DASH_SOLID), not parts of a tree."""
    lab, n = ndi.label(ink, structure=np.ones((3, 3), bool))
    size = np.bincount(lab.ravel(), minlength=n + 1)
    thick = np.bincount(lab[ndi.binary_opening(ink, structure=np.ones((2, 2), bool))], minlength=n + 1)
    dash = (size <= DASH_PX) & (thick >= DASH_SOLID * size)
    dash[0] = False
    return dash[lab]


def plots(hatching):
    """The hatched plots: the hatching closed over the gaps between its strokes, holes filled, PLOT_M2 or bigger."""
    area = ndi.binary_fill_holes(ndi.binary_closing(hatching, structure=ndi.iterate_structure(
        ndi.generate_binary_structure(2, 1), PLOT_GAP_PX)))
    lab, n = ndi.label(area)
    big = np.bincount(lab.ravel(), minlength=n + 1) * PX_M2 >= PLOT_M2
    big[0] = False
    return big[lab]


def native_masks(cls):
    """Per-pixel boolean masks for a 0.72 m/px essay map."""
    m = {}
    m["water"] = cls == C_WATER
    m["build"] = cls == C_BUILD
    m["earth"] = cls == C_EARTH
    m["wood"] = cls == C_WOOD
    m["white"] = cls == C_WHITE
    m["red"] = cls == C_RED
    m["scrub"] = cls == C_SCRUB
    m["under"] = (cls == C_UNDER) | (cls == C_DENSE)
    m["beds"] = plots(cls == C_UNDER)
    m["lawn"] = cls == C_LAWN
    blk = cls == C_BLACK
    # thick black strokes = boundary walls (drawn wider than scale)
    thick = ndi.binary_opening(blk, structure=np.ones((5, 5), bool))
    lab, n = ndi.label(thick)
    if n:
        sl = ndi.find_objects(lab)
        sizes = ndi.sum(thick, lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool)
        for i, s in enumerate(sl):
            h = s[0].stop - s[0].start; w = s[1].stop - s[1].start
            fill = sizes[i] / max(1, h * w)
            if max(h, w) > 120 and fill < 0.35: keep[i + 1] = True
        wall = keep[lab]
        wall = ndi.binary_dilation(wall, iterations=2) & blk
    else:
        wall = np.zeros_like(blk)
    m["wall"] = wall
    # outlines: black adjacent to non-green classes
    nongreen = m["water"] | m["build"] | m["earth"] | m["wood"] | m["white"] | m["red"]
    outline = blk & ndi.binary_dilation(nongreen, iterations=2)
    m["tree_ink"] = blk & ~wall & ~outline & ~dashes(blk & ~wall & ~outline)
    return m


def resample_mask(mask_native, to_native_fn, UU, VV, sigma_px=0.0):
    """Area-ish resample of a native mask into the local grid (fraction 0..1)."""
    f = mask_native.astype(np.float32)
    if sigma_px > 0: f = ndi.gaussian_filter(f, sigma_px)
    u, v = to_native_fn(UU, VV)
    inside = (u >= 0) & (u < f.shape[1] - 1) & (v >= 0) & (v < f.shape[0] - 1)
    out = map_coordinates(f, [v, u], order=1, mode='constant', cval=0.0)
    return out, inside


if __name__ == "__main__":
    UU, VV = local_grid()
    print("local grid", NX, NY)
    cg = np.load(P("build", "cls_g.npy")); ch = np.load(P("build", "cls_h.npy"))
    mg = native_masks(cg); mh = native_masks(ch)
    np.savez_compressed(P("build", "native_g.npz"), **mg)
    np.savez_compressed(P("build", "native_h.npz"), **mh)
    for nm, m in (("g", mg), ("h", mh)):
        vis = np.zeros(m["lawn"].shape + (3,), np.uint8) + 230
        vis[m["tree_ink"]] = [0, 120, 0]; vis[m["wall"]] = [255, 0, 0]; vis[m["water"]] = [60, 140, 255]
        vis[m["build"]] = [150, 60, 20]; vis[m["earth"]] = [200, 170, 110]; vis[m["wood"]] = [110, 90, 60]
        vis[m["white"]] = [255, 255, 255]; vis[m["red"]] = [255, 0, 255]
        s = 1400 / max(vis.shape[:2])
        Image.fromarray(vis).resize((int(vis.shape[1] * s), int(vis.shape[0] * s)), Image.NEAREST).save(P("view", f"native_{nm}.png"))
    print("done")
