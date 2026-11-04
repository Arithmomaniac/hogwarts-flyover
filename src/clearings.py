"""Clearings in the trees that read as places a visitor would recognise, not as cutouts made for a line of sight. The
user: "make sure that the clearings, wherever they are, are large enough to be plausible and not look suspiciously like
cutouts created just for line of sight purposes".

A clearing (`glade`) keeps its `core` clear, such as the ground under a line of sight. It spreads beyond the core by a
margin that no crown the renderer grows can reach across (SAFE_M), and as far again as its edge wanders (smooth noise,
up to 2 * WOBBLE_M). Beyond that edge the wood thickens back to its drawn density over FEATHER_M. A few trees stand
scattered inside it, clear of the core (`scattered`). Its own ground, a lawn or an overgrown garden, gives way to the
wood's across that same thinning edge in clumps (`ecotone`), so that the ground does not end at a line either; nor does
any patch of ground, such as scrub or the woods' floor (`blend`), but where a wall or a path explains its line. Every
cleared region either follows a linear feature (LINEAR) or is
such a clearing: at least MIN_M2 in area and MIN_WIDE_M across, with no straight stretch of its edge LONG_M long and no
two STRAIGHT_M long that run parallel and face each other across it (`straight_edges`, `parallel_edges`). The build
records each clearing's outline in meta.clearings (`record`), with the manifest decision that gives its reason, and
src/test_site.py checks them against the packed tree layers."""
import contourpy
import numpy as np
import shapely
from scipy import ndimage as ndi

from common import TEXEL_REACH_M, TREE_CROWN_M

SAFE_M = TREE_CROWN_M + TEXEL_REACH_M   # no crown the renderer grows reaches this far into cleared ground
MIN_M2, MIN_WIDE_M = 2500.0, 30.0       # a clearing that follows no linear feature is at least this big and this wide;
STRAIGHT_M, STRAIGHT_TOL_M = 40.0, 1.0  # a stretch of its edge is straight where every run this long keeps within this
LONG_M = 60.0                           # of a straight line; none is this long
WOBBLE_M, WOBBLE_BLUR_M = 6.0, 6.0      # its edge wanders out by up to twice this, in bays and spurs some tens of metres
                                        # long: smooth noise, white noise blurred over this far
FEATHER_M = 12.0                        # beyond its edge the wood thickens back over this far
CELL_M = 7.0                            # the renderer's tree cells (frag1.glsl treeSD): a tree stands within 1.75 m of
TREE_M = 5.5                            # a cell's centre, so density 1 this far round the centre grows it, on ground the
                                        # renderer lets trees grow on
SPACING_M = 20.0                        # scattered trees stand at least this far apart
ECOTONE_M, CLUMP_M = 16.0, 1.0          # across this width its ground and the ground round it interleave, in tufts this
                                        # small (smooth noise's), so that the edge fades rather than wavers;
BLEND_M = 5.0                           # the share of its ground falls from nine tenths to a tenth over this far or more,
INTERLEAVE = 0.2                        # and along at least this share of the lines across its edge the two interleave
SCRUB = 2                               # the surface class of the maps' rough scrubland (frag4.glsl classCol)
OVERGROWN = 8                           # the surface class of an overgrown garden's ground
LINEAR = {      # the clearings that follow a linear feature, and the manifest decision that gives each one's reason
    "the railway's earthworks, and its lineside within 10 m of the open line": "railway-regrade",
    "the Forest's corridors along the A890 and the Kyle railway": "forest-corridors-a890-railway",
    "the Forest's edge short of the A832": "forest-ends-at-a832",
    "the Forest's edge round Achnasheen, along the A832": "forest-clear-of-achnasheen",
}


def noise(shape, sigma, seed, clip=True):
    """Smooth noise over a grid: white noise blurred over `sigma` texels, with a finer octave blurred over a third of
    that at half its weight, scaled to a spread of 2/3 and clipped to [-1, 1] unless `clip` is False."""
    rng = np.random.default_rng(seed)
    coarse, fine = (ndi.gaussian_filter(rng.standard_normal(shape), s) for s in (sigma, sigma / 3.0))
    n = coarse / coarse.std() + 0.5 * fine / fine.std()
    n = n / (1.5 * n.std())
    return np.clip(n, -1.0, 1.0) if clip else n


def window(mask, pad):
    """Slices round `mask`'s texels, `pad` texels wider, within its grid."""
    rows, cols = np.nonzero(mask)
    return (slice(max(int(rows.min()) - pad, 0), int(rows.max()) + pad + 1),
            slice(max(int(cols.min()) - pad, 0), int(cols.max()) + pad + 1))


def glade(core, texel, seed, margin_m=SAFE_M, wobble_m=WOBBLE_M, blur_m=WOBBLE_BLUR_M, feather_m=FEATHER_M):
    """The tree multiplier (0 to 1) of a clearing round `core`, a grid of texels `texel` metres apart: 0 within margin_m
    of the core and as far again as its edge wanders (0 to 2 * wobble_m, by noise blurred over blur_m), then rising
    smoothly to 1 over feather_m."""
    if not core.any():
        raise ValueError("a clearing needs a core to keep clear")
    out = np.ones(core.shape, np.float32)
    w = window(core, int(np.ceil((margin_m + 2.0 * wobble_m + feather_m) / texel)) + 2)
    d = ndi.distance_transform_edt(~core[w]) * texel
    edge = margin_m + wobble_m * (1.0 + noise(d.shape, blur_m / texel, seed))
    t = np.clip((d - edge) / feather_m, 0.0, 1.0)
    out[w] = t * t * (3.0 - 2.0 * t)
    return out


def ecotone(m, texel, seed, width_m=ECOTONE_M, clump_m=CLUMP_M, inset_m=0.0):
    """Where a clearing whose multiplier is `m` has its own ground. Across its outline (where m crosses a half) the
    ground gives way to the wood's in clumps about clump_m across: each texel's chance falls evenly with its distance
    across the outline, from 1 at width_m / 2 inside it to 0 at width_m / 2 outside (the chance that smooth noise,
    evened out by its ranks, exceeds it), or over a band centred inset_m inside the outline."""
    inside = m < 0.5
    out = inside.copy()
    w = window(m < 1.0, int(np.ceil(width_m / texel)) + 2)
    sd = np.where(inside[w], 0.5 * texel - ndi.distance_transform_edt(inside[w]) * texel,
                  ndi.distance_transform_edt(~inside[w]) * texel - 0.5 * texel)
    n = noise(sd.shape, clump_m / texel, seed, clip=False)        # unclipped, so that no two texels tie in rank
    rank = np.empty(n.size)
    rank[np.argsort(n, axis=None)] = (np.arange(n.size) + 0.5) / n.size
    out[w] = rank.reshape(sd.shape) < 0.5 - (sd + inset_m) / width_m
    return out


def blend(sc, cls, texel, seed, soft, barrier=None, may=None, width_m=ECOTONE_M, inset_m=0.0, within=None):
    """The surface classes `sc` with the edges of their class-`cls` patches blended into the ground of the `soft`
    classes round them over width_m, as a clearing's own ground is (`ecotone`): a patch takes in soft ground beyond its
    edge in tufts, where `may` lets it (such as under the canopy, for the woods' floor), and gives up its own to the soft
    ground across its edge. Inset by inset_m, the band of tufts lies that much further in: by half width_m, a patch takes
    in nothing beyond its edge, as the floor of a wood whose trees end at a line does. Its edge runs past the ground
    `within`, such as a wood a planting runs on into, which keeps its own. It changes no other ground, such as a path,
    and nothing within `barrier`, such as a wall, whose line the edge keeps."""
    patch = sc == cls
    if not patch.any():
        return sc
    whole = patch if within is None else patch | within
    own = ecotone(np.where(whole, 0.0, 1.0), texel, seed, width_m, inset_m=inset_m)
    kept = np.zeros(sc.shape, bool) if barrier is None else barrier
    open_ground = np.isin(sc, soft) & ~kept
    across = ndi.distance_transform_edt(whole, return_distances=False, return_indices=True)     # each patch texel's
    nearest = ndi.distance_transform_edt(~patch, return_distances=False, return_indices=True)   # nearest other, and back
    out = sc.copy()
    lost = patch & ~kept & ~own & open_ground[across[0], across[1]]
    out[lost] = sc[across[0], across[1]][lost]
    gained = ~whole & own & open_ground & ~kept[nearest[0], nearest[1]]
    out[gained if may is None else gained & may] = cls
    return out


def to_texel(E, N):
    """The inverse of a grid's affine texel-to-world map, given its texel centres' world coordinates E, N."""
    e0, n0 = float(E[0, 0]), float(N[0, 0])
    m = np.array([[E[1, 0] - e0, E[0, 1] - e0], [N[1, 0] - n0, N[0, 1] - n0]], float)
    inv = np.linalg.inv(m)
    return lambda e, n: inv @ np.vstack([np.ravel(e) - e0, np.ravel(n) - n0])


def scattered(m, core, E, N, texel, count, seed, may=None):
    """Tree density (0 or 1): `count` trees or fewer standing scattered in the clearing whose multiplier is `m` round
    `core`, at least SPACING_M apart. Each is density 1 within TREE_M of a renderer cell's centre, where that cell's tree
    stands, and that cell is one where m is 0. Its disc of full density lies SAFE_M or more from the core, so that no
    crown it grows reaches it, and wholly on ground where trees may stand (`may`, such as off roads, water and
    buildings, where the renderer grows them)."""
    out = np.zeros(m.shape, np.float32)
    w = window(m < 1.0, 2)
    d = ndi.distance_transform_edt(~core[w]) * texel
    room = np.full(d.shape, np.inf) if may is None else ndi.distance_transform_edt(may[w]) * texel
    reach = TREE_M + texel * np.sqrt(0.5)       # the farthest a texel of the disc lies from the texel nearest its centre
    e, n = E[w], N[w]
    cx = np.arange(np.floor(e.min() / CELL_M), np.ceil(e.max() / CELL_M)) + 0.5
    cz = np.arange(np.floor(n.min() / CELL_M), np.ceil(n.max() / CELL_M)) + 0.5
    ce, cn = (a.ravel() * CELL_M for a in np.meshgrid(cx, cz))
    r, c = np.rint(to_texel(E, N)(ce, cn)).astype(int)
    r, c = r - w[0].start, c - w[1].start
    ok = (r >= 0) & (c >= 0) & (r < d.shape[0]) & (c < d.shape[1])
    ce, cn, r, c = ce[ok], cn[ok], r[ok], c[ok]
    ok = (m[w][r, c] == 0.0) & (d[r, c] >= SAFE_M + reach) & (room[r, c] > reach)
    picked = []
    for k in np.random.default_rng(seed).permutation(np.nonzero(ok)[0]):
        if len(picked) == count:
            break
        if all(np.hypot(ce[k] - a, cn[k] - b) >= SPACING_M for a, b in picked):
            picked.append((ce[k], cn[k]))
    for a, b in picked:
        out[w] = np.maximum(out[w], (np.hypot(e - a, n - b) <= TREE_M).astype(np.float32))
    return out


def outlines(m, E, N, level=0.5):
    """The clearing's parts, each a polygon in world (east, north) coordinates bounded where its multiplier `m` crosses
    `level` (traced across texels, as smooth as the multiplier), with the islands of wood it leaves standing as holes.
    A clearing that the grid's edge cuts off has no closed outline and is left out: the renderer fades the Forest out
    towards its layer's edges (frag1.glsl vegDensity), and no local clearing comes near the local layer's."""
    w = window(m < level, 2)
    lines = contourpy.contour_generator(z=np.asarray(m[w], np.float64)).lines(level)
    rows0, cols0 = w[0].start, w[1].start
    e0, n0 = float(E[0, 0]), float(N[0, 0])
    de = (float(E[1, 0] - e0), float(E[0, 1] - e0))
    dn = (float(N[1, 0] - n0), float(N[0, 1] - n0))
    rings = []
    for xy in lines:
        if len(xy) < 4 or not np.allclose(xy[0], xy[-1]):
            continue
        rr, cc = xy[:, 1] + rows0, xy[:, 0] + cols0
        rings.append(np.column_stack([e0 + rr * de[0] + cc * de[1], n0 + rr * dn[0] + cc * dn[1]]))
    shells = [shapely.Polygon(r) for r in rings]
    depth = [sum(p.contains(q) for p in shells if p is not q) for q in shells]   # even: a part's edge; odd: an island's
    return [shapely.Polygon(rings[i], [rings[j] for j in range(len(rings))
                                       if depth[j] == depth[i] + 1 and shells[i].contains(shells[j])])
            for i in range(len(rings)) if depth[i] % 2 == 0]


def polygon(record):
    """A meta.clearings record's ground: its outline, less its islands."""
    return shapely.Polygon(record["outline"], record.get("islands", []))


def record(name, decision, layer, m, E, N, removed):
    """meta.clearings' records of a clearing whose multiplier is `m`: one per part round ground where it removed trees
    (`removed`), each {"id", "decision", "layer", "outline": [[east, north], ...]} and, if it leaves islands of wood
    standing, their "islands", to 0.1 m."""
    out = []
    gone = np.column_stack([E[removed], N[removed]])
    ring = lambda r: np.round(np.asarray(r.coords), 1).tolist()
    for part in outlines(m, E, N):
        if len(gone) and shapely.contains_xy(part, gone[:, 0], gone[:, 1]).any():
            out.append({"id": name, "decision": decision, "layer": layer, "outline": ring(part.exterior)}
                       | ({"islands": [ring(h) for h in part.interiors]} if part.interiors else {}))
    return out


def _resample(ring, step):
    """A closed outline's points every `step` metres along it."""
    ring = np.asarray(ring, float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(ring, axis=0).T))])
    t = np.arange(0.0, float(s[-1]), step)
    return np.column_stack([np.interp(t, s, ring[:, 0]), np.interp(t, s, ring[:, 1])])


def straight_edges(ring, length=STRAIGHT_M, tol=STRAIGHT_TOL_M, step=1.0):
    """The straight stretches of a closed outline: each stretch along which every run `length` long keeps within `tol`
    of the line between its ends, as (its length in metres, its start, its end)."""
    pts = _resample(ring, step)
    k, n = int(round(length / step)), len(pts)
    if n <= k:
        return []
    straight = np.zeros(n, bool)
    for i in range(n):
        run = pts[(i + np.arange(k + 1)) % n]
        chord = run[-1] - run[0]
        norm = float(np.hypot(*chord))
        off = np.abs((run[:, 0] - run[0, 0]) * chord[1] - (run[:, 1] - run[0, 1]) * chord[0]) / max(norm, 1e-9)
        straight[i] = norm > 0.0 and off.max() < tol
    if straight.all():
        return [(n * step, pts[0], pts[0])]
    out = []
    first = int(np.argmin(straight))           # start from a point that begins no straight run
    i = 0
    while i < n:
        j = (first + i) % n
        if straight[j]:
            m = i
            while m < n and straight[(first + m) % n]:
                m += 1
            out.append(((m - i - 1 + k) * step, pts[j], pts[(first + m - 1 + k) % n]))
            i = m
        else:
            i += 1
    return out


def parallel_edges(ring, length=STRAIGHT_M, tol=STRAIGHT_TOL_M, angle_deg=10.0):
    """Pairs of straight stretches of a closed outline, each `length` or longer, that run parallel (within angle_deg)
    and face each other across the clearing over half that length or more: the signature of a strip cut along a line.
    Each pair as its two stretches' starts."""
    poly = shapely.Polygon(ring)
    edges = straight_edges(ring, length, tol)
    out = []
    for a in range(len(edges)):
        for b in range(a + 1, len(edges)):
            (_, p0, p1), (_, q0, q1) = edges[a], edges[b]
            u = (p1 - p0) / np.hypot(*(p1 - p0))
            v = (q1 - q0) / np.hypot(*(q1 - q0))
            if abs(float(u @ v)) < np.cos(np.radians(angle_deg)):
                continue
            t = sorted((float((q0 - p0) @ u), float((q1 - p0) @ u)))
            overlap = min(float(np.hypot(*(p1 - p0))), t[1]) - max(0.0, t[0])
            middle = ((p0 + p1) / 2 + (q0 + q1) / 2) / 2
            if overlap >= length / 2 and shapely.contains_xy(poly, *middle):
                out.append((tuple(np.round(p0, 1)), tuple(np.round(q0, 1))))
    return out


def unaccounted(drawn, final, accounted, floor=0.05):
    """Texels whose trees the build removed (their drawn density fell by more than `floor`) where no rule that may
    remove them (`accounted`) applies."""
    return (drawn - final > floor) & ~accounted
