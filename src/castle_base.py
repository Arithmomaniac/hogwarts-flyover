"""The castle's base, where the site's ground lies far below the castle platform (at X, 54 to 77 m below it).

White Hound: "The castle stands on top of its own little hill or mound ... It is surrounded by lawns, which generally
slope downwards towards the perimeter of the grounds (at least in the direction of the front gates) ... There is a
substantial skirt of lawn in front of the castle" (grounds map 4d "castle mound", 4e "sloping lawn"); "most of the
slopes, except the drop straight down the cliff from castle to lake, are shallow enough to be walked up or down". The
cliff over the lake has "the lower part of the cliff either vertical or sloping back, and the upper part sloping back
towards the base of the castle, too steep to build on but shallow enough to allow for a foot-path along the face of
the cliff; possibly with a safety-wall between the path and the drop" (2e, 2f): "an irregular rock ... with earth piled
against it on the side away from the water".

Two bases share that lake front and the skirt of lawn: the "mound", whose lawns fall away on every landward side, gently
towards Hagrid's cabin, the pitch and the gates, more steeply elsewhere; and the "crag", whose rock faces and scree
stand on the landward sides away from those, with the same lawns falling towards them. Both only ever add ground, and
meet the natural ground in a rounded toe. Everything is on the local grid (TEXEL_M texels), in metres.
"""
import numpy as np
from scipy import ndimage as ndi

TEXEL_M = 2.0
SKIRT_M, SKIRT_FALL_M = 30.0, 2.0      # the skirt of lawn round the castle, and how far it falls across its width
SHOULDER_M = 10.0                      # the rounding where the skirt turns down into the slope
TOE_M = 10.0                           # the rounding where the slope meets the natural ground
LAWN_GRADE = {"toward": 1 / 4.0, "away": 1 / 2.6}   # the mound's lawns: towards the cabin, pitch and gates, and elsewhere
TOWARD_HALF_DEG = {"mound": 75.0, "crag": 40.0}     # the bearings round the cabin, pitch and gates that count as "toward"
PATH_GRADE = 0.4                       # the cliff's upper part, the foot-path's slope (2f): 22 degrees
FACE_GRADE = 6.5                       # the cliff's lower part, below the lip (2e): 81 degrees
SHORE_GRADE = 0.45                     # elsewhere the lake's shores rise no faster than this (build_final's lake shaping)
CLIFF_BLEND_M = 20.0                   # where the castle's cliff gives way to lawns falling to the lake
FRONT_WALL_M = 12.0                    # thin castle walls this near the lake are the cliff-edge wall (2e), not the castle
CRAG_RIM_M = 12.0                      # the crag: lawn along its rim, then a rock face ...
CRAG_FACE_GRADE = 3.5                  # ... at 74 degrees,
CRAG_FACE_M = (24.0, 44.0)             # ... between these heights, varying round the crag,
CRAG_SCREE_GRADE = 0.7                 # ... then scree at 35 degrees down to the ground
CRAG_WOBBLE_M, CRAG_NOISE_M = 7.0, 45.0   # how far the face wanders in and out, and the scale of its wandering
CRAG_CLOSE_M = 30.0                    # the crag's top bridges gaps between the castle's parts narrower than this
SEED = 4                               # the crag's irregularity: one seed, so the build is reproducible
GARDEN_GAP_M = 25.0                    # the vegetable patch: the bank down to it from the platform takes this much room,
GARDEN_REACH_M = 100.0                 # ... and its level ground reaches no further from the platform than this;
GARDEN_BANK_GRADE = 0.45               # ... the banks round it, cut or filled, can be walked up or down;
GARDEN_HOLLOW_M, GARDEN_RIM_M = 0.25, 8.0  # ... its own little hollow: a flood to splash across, reached so far in


def softplus(x, r):
    return r * np.logaddexp(0.0, x / r)


def smooth_max(a, b, k):
    """max(a, b), the corner where they cross filled by a fillet about k across: as a mound's toe meets the ground."""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.maximum(a, b) + h * h * k / 4.0


def smooth_min(a, b, k):
    return -smooth_max(-a, -b, k)


def noise(shape, scale_m, seed):
    """Smooth noise of unit spread, its features about `scale_m` across."""
    field = ndi.gaussian_filter(np.random.default_rng(seed).standard_normal(shape), scale_m / TEXEL_M / 2.0)
    return field / field.std()


def front_walls(castle, lake_sd):
    """The castle's thin walls along the lake: the cliff-edge wall (2e), which stands on the cliff, not on the platform."""
    thin = castle & ~ndi.binary_opening(castle, structure=np.ones((3, 3), bool))
    return thin & (lake_sd < FRONT_WALL_M)


def lawn_fall(d, grade):
    """How far below the platform the lawn lies `d` metres out from the castle: the skirt, then the slope."""
    skirt = SKIRT_FALL_M / SKIRT_M
    return skirt * d + (grade - skirt) * softplus(d - SKIRT_M, SHOULDER_M)


def crag_fall(d, height, rim):
    """The crag: a lawn rim, a rock face falling `height`, then scree."""
    skirt = SKIRT_FALL_M / SKIRT_M
    face_end = rim + height / CRAG_FACE_GRADE
    return (skirt * d + (CRAG_FACE_GRADE - skirt) * softplus(d - rim, 1.5)
            - (CRAG_FACE_GRADE - CRAG_SCREE_GRADE) * softplus(d - face_end, 2.5))


def shape(kind, ground, E, N, castle, lake_sd, platform, lake, toward, garden=None, dry=None, glass=None):
    """The ground with the castle's base `kind` ("mound" or "crag") built on it, where the lake's shore may rise as the
    cliff's face, and the vegetable patch's hollow: (ground, cliff, hollow). `castle`: the castle's mass; `lake_sd`:
    metres from the lake's shore, negative in it; `platform`: the castle's base level (a number or the base field);
    `toward`: unit (east, north) from the castle towards Hagrid's cabin and the pitch; `garden`: White Hound's vegetable
    patch and greenhouses, whose ground between GARDEN_GAP_M and GARDEN_REACH_M from the platform is laid level, a
    little hollow that floods in heavy rain, but for `dry` ground in it (the greenhouses, their pads and the paths);
    `glass`: the greenhouses' footprints, those in the patch standing wholly on its level ground."""
    if kind not in ("mound", "crag"):
        raise ValueError(f"unknown castle base {kind!r}")
    # the platform carries the castle's main body and its courts; its detached pieces (the vegetable patch's old wall,
    # 5d) and the cliff-edge wall stand on the ground round it
    pieces, _ = ndi.label(castle, structure=np.ones((3, 3), bool))
    main = pieces == np.bincount(pieces[pieces > 0]).argmax()
    body = main & ~front_walls(castle, lake_sd)
    flat = ndi.binary_fill_holes(ndi.binary_dilation(body, iterations=2))     # the platform, its courts included
    d = ndi.distance_transform_edt(~flat) * TEXEL_M
    bearing = np.degrees(np.arccos(np.clip((E * toward[0] + N * toward[1]) / np.maximum(np.hypot(E, N), 1e-6), -1, 1)))
    t = np.clip((TOWARD_HALF_DEG[kind] + 20.0 - bearing) / 40.0, 0.0, 1.0)
    toward_w = t * t * (3 - 2 * t)                                                 # 1 within the half-angle, 0 beyond
    lawn = platform - lawn_fall(d, LAWN_GRADE["toward"] * toward_w + LAWN_GRADE["away"] * (1 - toward_w))
    if kind == "mound":
        top = lawn
    else:
        r = int(round(CRAG_CLOSE_M / 2 / TEXEL_M))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        rock = ndi.binary_closing(np.pad(flat, r), structure=xx * xx + yy * yy <= r * r)[r:-r, r:-r] | flat
        wobble = noise(ground.shape, CRAG_NOISE_M, SEED) * CRAG_WOBBLE_M
        height = np.interp(noise(ground.shape, CRAG_NOISE_M * 2, SEED + 1), (-2.0, 2.0), CRAG_FACE_M)
        dr = ndi.distance_transform_edt(~rock) * TEXEL_M
        crag = platform - crag_fall(np.maximum(dr + wobble, 0.0), height, CRAG_RIM_M)
        top = lawn * toward_w + crag * (1 - toward_w)
    # by the lake: where the castle stands too near it for a walkable slope down to the water, the cliff (the foot-path's
    # slope down from the castle, then the face); elsewhere the lawns round down to the shore, no steeper than it allows
    sp = np.maximum(lake_sd, 0.0)
    room = (platform - lake - 0.35) / SHORE_GRADE
    c = np.clip((room + CLIFF_BLEND_M - (d + sp)) / CLIFF_BLEND_M, 0.0, 1.0)
    c = c * c * (3 - 2 * c)
    lawns = smooth_min(top, lake + 0.35 + SHORE_GRADE * sp, TOE_M)
    cliffs = np.minimum(platform - PATH_GRADE * d, lake + 0.35 + FACE_GRADE * sp)
    top = lawns * (1 - c) + cliffs * c
    # the vegetable patch: level, below the bank from the platform, in its own little hollow
    terrace = np.zeros(ground.shape, bool)
    if garden is not None:
        terrace = garden & (d >= GARDEN_GAP_M) & (d <= GARDEN_REACH_M) & (lake_sd > 0)
    if glass is not None:           # the patch's greenhouses, wherever it reaches, stand wholly on its level ground
        parts, _ = ndi.label(garden, structure=np.ones((3, 3), bool))
        patch = np.isin(parts, np.unique(parts[terrace & (parts > 0)]))
        houses, _ = ndi.label(glass, structure=np.ones((3, 3), bool))
        reached = np.unique(houses[glass & patch])
        terrace |= np.isin(houses, reached[reached > 0]) & (d >= GARDEN_GAP_M) & (lake_sd > 0)
    hollow = terrace & ~dry if dry is not None else terrace
    if terrace.any():
        highest = float(np.max(np.broadcast_to(platform, ground.shape)[flat]))
        level = max(float(np.median(top[terrace])), highest - GARDEN_BANK_GRADE * GARDEN_GAP_M)
        bank = ndi.distance_transform_edt(~terrace) * TEXEL_M * GARDEN_BANK_GRADE
        top = np.clip(top, level - bank, level + bank)
        inside = ndi.distance_transform_edt(hollow) * TEXEL_M          # a rim two texels wide: no diagonal leaks
        top = top - GARDEN_HOLLOW_M * np.clip((inside - 2 * TEXEL_M) / (GARDEN_RIM_M - 2 * TEXEL_M), 0.0, 1.0)
    out = smooth_max(ground, top, TOE_M)
    return np.where(flat, platform, out).astype(np.float32), c > 0, hollow
