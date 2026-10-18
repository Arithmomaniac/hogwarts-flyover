"""Extract feature masks from the essay's grounds + Hogsmeade maps (native 0.72 m/px) and
resample everything into the local 2 m map-aligned grid."""
import sys
from common import *

FX0, FX1, FY0, FY1 = 900.0, 1740.0, 2560.0, 3320.0   # local frame extent in Forest-map px
RES = 2.0
NX = int(round((FX1 - FX0) * MFF / RES)); NY = int(round((FY1 - FY0) * MFF / RES))


def local_grid():
    uu = FX0 + (np.arange(NX) + 0.5) * RES / MFF
    vv = FY0 + (np.arange(NY) + 0.5) * RES / MFF
    return np.meshgrid(uu, vv)


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
    m["tree_ink"] = blk & ~wall & ~outline
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
