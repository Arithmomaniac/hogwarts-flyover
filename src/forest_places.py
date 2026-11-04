"""Aragog's hollow and Grawp's clearing: the two places in the Forbidden Forest that White Hound's Forest map draws (6g,
4f; DRAWN), and where a site moves them (its SITES[...] "aragog_hollow" and "grawp_clearing", world metres from the
castle). A site that moves neither keeps them as drawn: his crater a patch of the map bare of trees, Grawp's spot among
the pines frag1.glsl grows round it. Where a site moves them, src/mesh_scene.json records why, with the books' words
and his:

- the hollow is cut into the 10 m regional ground (pack.py) within RIM_M of its centre, the floor DEPTH_M below the
  lowest ground on its rim and level, its sides rising at SIDE_DEG to meet the ground at the rim (`hollow`). Its
  clearing takes the Forest off it, out to beyond its rim (`hollow_clearing`), and the patch his map draws bare round
  his crater takes the Forest round that patch (`crater`, `restored`);
- Grawp's clearing is cleared of all but a few small pines (`grawp_clearing`, `with_pines`).
Each clearing's edge wanders and the wood thickens back beyond it (clearings.glade), and the build records its outline
in meta.clearings."""
import numpy as np
from scipy import ndimage as ndi

import clearings
from common import ff_to_world

DRAWN = {"Aragog's hollow": (1560, 2134), "Grawp's clearing": (940, 2186)}     # his Forest map's pixels for 6g and 4f
SITE_KEYS = {"Aragog's hollow": "aragog_hollow", "Grawp's clearing": "grawp_clearing"}
RIM_M, DEPTH_M, SIDE_DEG = 50.0, 10.0, 45.0     # the hollow's rim round its centre, its floor's depth below the lowest
                                                # ground on that rim, and its sides' slope
HOLLOW_WOBBLE_M, HOLLOW_BLUR_M, HOLLOW_FEATHER_M, HOLLOW_SEED = 6.0, 12.0, 16.0, 13    # its clearing's edge
GRAWP_CORE_M = 25.0                             # Grawp's clearing holds only small pines this far round its centre,
GRAWP_WOBBLE_M, GRAWP_BLUR_M, GRAWP_FEATHER_M, GRAWP_SEED = 6.0, 24.0, 16.0, 17        # and as far again as its edge
PINES = 0.2                                     # wanders; the density of the pines standing there
CRATER_REACH_M, BARE = 90.0, 0.15               # his crater's bare patch: under this density, all within this reach;
CRATER_EDGE_M = 12.0                            # and its edge, which the layer's sampling of his map blurs, this far out


def where(site, name):
    """Where a site puts one of the Forest's places: (east, north) metres from the castle, the site's own position for
    it if it moves it, else where his map draws it."""
    moved = site.get(SITE_KEYS[name])
    return tuple(float(v) for v in (moved if moved else ff_to_world(*DRAWN[name])))


def hollow(F, E, N, at, ground, rim_m=RIM_M, depth_m=DEPTH_M, side_deg=SIDE_DEG):
    """Heights F at world points E, N with the hollow cut into them round `at`. `ground` gives the ground's height at
    world points (east, north arrays); read on the rim, rim_m round `at`, every half degree, its lowest less depth_m is
    the floor. Within the rim each point lies on the floor or, where higher, on a side rising at side_deg from the floor
    to meet the ground at the rim on its own bearing from `at`. The hollow only cuts, and nothing beyond its rim
    changes."""
    a = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    floor = float(np.min(ground(at[0] + rim_m * np.cos(a), at[1] + rim_m * np.sin(a)))) - depth_m
    de, dn = np.asarray(E, float) - at[0], np.asarray(N, float) - at[1]
    r = np.hypot(de, dn)
    inside = r <= rim_m
    bearing = np.arctan2(dn[inside], de[inside])
    rim = np.asarray(ground(at[0] + rim_m * np.cos(bearing), at[1] + rim_m * np.sin(bearing)), float)
    out = np.array(F, copy=True)
    out[inside] = np.minimum(out[inside], np.maximum(floor, rim - np.tan(np.radians(side_deg)) * (rim_m - r[inside])))
    return out


def crater(density, E, N, at, reach_m=CRATER_REACH_M, bare=BARE, edge_m=CRATER_EDGE_M):
    """The texels of the patch his map draws bare round his crater, at a layer's texel centres E, N: those of density
    under `bare` joined to the texel nearest `at`, and those within edge_m of them, where the layer's sampling of his
    map thins the wood and tints its ground as his crater's lawn. A spot that is not bare, or bare ground reaching
    further than reach_m from it, such as open country, is refused (ValueError)."""
    k = np.unravel_index(np.argmin((E - at[0]) ** 2 + (N - at[1]) ** 2), E.shape)
    labels, _ = ndi.label(density < bare)
    if not labels[k]:
        raise ValueError(f"the Forest is not bare at {at}")
    patch = labels == labels[k]
    far = float(np.hypot(E[patch] - at[0], N[patch] - at[1]).max())
    if far > reach_m:
        raise ValueError(f"the bare ground at {at} reaches {far:.0f} m from it, further than a crater's {reach_m:.0f} m")
    texel = float(np.hypot(E[0, 1] - E[0, 0], N[0, 1] - N[0, 0]))
    return ndi.distance_transform_edt(~patch) * texel <= edge_m


def restored(density, field, patch, ring=3, wooded=0.3):
    """The Forest layer's density and field share with a crater's patch given the Forest round it: no texel of it is
    thinner than the median density of the texels within `ring` texels round it that hold `wooded` or more, and none
    keeps a field share."""
    near = ndi.binary_dilation(patch, iterations=ring) & ~patch
    wood = density[near][density[near] >= wooded]
    if not len(wood):
        raise ValueError("no Forest round the patch to give it")
    d, f = np.array(density, copy=True), np.array(field, copy=True)
    d[patch], f[patch] = np.maximum(d[patch], np.median(wood)), 0.0
    return d, f


def hollow_clearing(E, N, at, texel):
    """The multiplier (0 to 1) of the Forest round Aragog's hollow at `at`, at a layer's texel centres E, N, `texel`
    metres apart: 0 out to its rim and a texel's diagonal beyond, so that a bilinear read of the layer grows no tree in
    the hollow, and as far again as its edge wanders, the wood thickening back beyond (clearings.glade)."""
    core = np.hypot(E - at[0], N - at[1]) <= RIM_M + texel * np.sqrt(2.0)
    return clearings.glade(core, texel, HOLLOW_SEED, margin_m=0.0, wobble_m=HOLLOW_WOBBLE_M, blur_m=HOLLOW_BLUR_M,
                           feather_m=HOLLOW_FEATHER_M)


def grawp_clearing(E, N, at, texel):
    """The multiplier (0 to 1) of the Forest round Grawp's clearing at `at`, at a layer's texel centres E, N, `texel`
    metres apart: 0 within GRAWP_CORE_M of it and as far again as its edge wanders, the wood thickening back beyond
    (clearings.glade). Only its small pines stand where it is 0 (`with_pines`)."""
    core = np.hypot(E - at[0], N - at[1]) <= GRAWP_CORE_M
    return clearings.glade(core, texel, GRAWP_SEED, margin_m=0.0, wobble_m=GRAWP_WOBBLE_M, blur_m=GRAWP_BLUR_M,
                           feather_m=GRAWP_FEATHER_M)


def with_pines(density, m, pines=PINES):
    """The Forest's density with Grawp's clearing (multiplier m) cut but for its small pines: where m is 0, `pines`
    stands, or less where less was drawn; the density drawn returns as m rises to 1, and nowhere rises above it.
    frag1.glsl grows mostly pines round the clearing, and grows trees smaller where the density is low."""
    return density - (density - np.minimum(density, pines)) * (1.0 - np.asarray(m, float))
