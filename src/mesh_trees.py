"""The tree species kit (mesh-trees): seeded meshes of the Forest's species, for the instanced mesh trees.

The kit places nothing: the renderer keeps the Forest's own placement rule and fits a variant to each placement, scaling
it so that its crown radius and height meet the placement's. Each species is drawn from White Hound's Forest (trees-
species): Scots pines, mature and young, and yews; oaks, beeches, and the site's birches. A species is a crown envelope
(how wide the crown is at each height, swelling and parting unevenly round the tree), a trunk that forks into limbs or
runs up as one leader, and leaf clumps in the envelope's outer shell; branches run from the limbs or the leader out to
the clumps. The full level draws the clumps as leaf cards, which the renderer cuts to leaf shapes, round solid cores in
the dense species; the middle level, fewer and larger cards on the trunk and limbs; the coarsest, a trunk and two solid
lumps (or a cone, for a young pine). Every variant is built in tree-local metres, its base at the origin and +y up, and
fitted so that no vertex of any level lies beyond its recorded crown radius or height. One seed fixes each variant."""
import math
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw

from materials.trees import COLOURS
from mesh_willow import Y, blob, faces_out, grow, perpendicular, rotate, tube, unit, vertex_normals

FAMILY = "trees"
BARK, FOLIAGE, CARD = 0, 1, 2
BUDGETS = ((1500, 1100), (300, 220), (60, 45))   # (triangles, vertices) at each level, the renderer's contract
SINK = 0.025               # the trunk starts this fraction of the tree's height below the origin, so it meets a slope
FLARE, FLARE_H = 0.35, 0.04   # the trunk's foot swells by this much, decaying over this fraction of the height
SEGMENTS = ((8, 5, 4), (5, 4), (4,))   # sides of the trunk, the limbs and the branches, at each level
LEAF_TILE = 64.0           # the page multiplies a leaf card's uv by this (template_b.html treeMeshVs), so a card's uv
                           # can span a part of the unit square: as many leaves as its size in metres holds
LEAF_GRID, LEAF_CELLS = 3.5, 1.24   # the page's leaf cells per unit of uv, and a leaf's length in cells (frag5.glsl leafMask)
CARD_FILL = 0.48           # the share of a leaf card the renderer's leaf mask keeps (lane T: frag5.glsl leafMask)
LOW_SLACK = 0.1            # a coarser level's crown starts within this fraction of the height of the full level's
CARD_SIZES = np.linspace(0.9, 3.3, 17)    # the middle level's card sizes over its groups' reach, tried in turn
LUMP_SCALES = [(sx, sy) for sx in np.linspace(0.25, 1.1, 18) for sy in np.linspace(0.7, 1.0, 7)]   # the coarsest
                           # level's lumps (or cone) across and up the full level's span from their base, tried
HOLD = 0.88                # a coarser level keeps at least this fraction of the full level's top
COVER_BAND = 1.15          # and covers within this factor of the full level's leaves, as wide as it can
FIT = 0.95                 # and needs shrinking by no more than this to fit within the full level
LEAN_IN = (0.25, 0.85)     # a leaf normal's least cosine to its own surface's normal: on a card, on a solid lump's
                           # coarse facets


class TreeMesh(NamedTuple):
    pos: np.ndarray    # (n, 3) float64 tree-local metres: the base at the origin, +y up
    nrm: np.ndarray    # (n, 3) unit normals; leaves are lit as one mass, by the crown's own normals
    attr: np.ndarray   # (n, 2) uint8 (part, aux): part BARK, FOLIAGE or CARD; aux, on bark, how far it has turned to its
                       # upper colour (0 to 255), and on leaves how deep inside the crown they lie (0 outside, 255 at
                       # its heart)
    idx: np.ndarray    # (m, 3) uint32 triangles; cross(b - a, c - a) points along their vertices' normals
    uv: np.ndarray     # (n, 2) float32: each leaf card's corners over [0, 1]^2; 0 elsewhere


class Variant(NamedTuple):
    name: str
    species: str
    conifer: bool
    share: float       # its weight within its class (conifers or broadleaves); a class's shares sum to 1
    crown_r: float     # metres: no vertex of any level lies further from the trunk's axis
    height: float      # metres: none lies higher
    crown_base: float  # metres: where its leaves start
    trunk_r: float     # metres: the trunk's radius at the ground
    levels: tuple      # three TreeMesh, the full level first


class Species(NamedTuple):
    name: str
    form: str              # "" or which of a species' forms (a Scots pine's "mature" or "young")
    conifer: bool
    share: float           # the form's weight within its class, shared evenly among its variants
    height: float          # metres, nominal: the renderer fits every variant to its placement
    ratios: tuple          # each variant's crown radius over its height
    seeds: tuple
    base: tuple            # the crown's base over the height, (lo, hi)
    widest: float          # where the crown is widest, over its depth from its base
    cone: bool             # it narrows straight to its top (a young pine)
    trunk_r: float         # the trunk's radius over the height
    fork: float            # where the trunk splits into limbs, over the height; 0: one leader runs to the top
    limbs: tuple           # (lo, hi) limbs at the fork
    clumps: int            # leaf clumps at the full level
    clump_r: float         # a leaf clump's radius over the crown's
    shell: float           # clumps lie between this fraction of the crown's radius and its edge
    lobes: float           # how unevenly the outline swells and parts round the tree
    flat: float            # 0: leaf cards face every way; 1: they lie flat (a pine's pads)
    core: float            # solid cores, over a clump's radius, in the largest third of the clumps (0: none)
    upper: tuple           # the bark turns to its upper colour between these fractions of the height
    rise: tuple            # limbs' and branches' angle over the horizontal, (lo, hi) degrees
    leaf: float = 0.1      # metres: a leaf's length (a pine's or yew's, a spray of needles), as the page cuts the cards
    card: float = 1.0      # the full level's cards' size over its clumps': cut to leaves this size, they can fill the crown


NO_TURN = (1.0, 1.0)
SPECIES = (
    # "the swathe of pine trees between the station and the lake": tall, with a high, flat-topped, broken crown on
    # branches that leave the stem level, their needles in flat pads
    Species("scots-pine", "mature", True, 0.55, 18.0, (0.23, 0.23), (11, 12), (0.5, 0.62), 0.55, False, 0.022, 0.0,
            (0, 0), 20, 0.38, 0.3, 0.4, 0.8, 0.0, (0.3, 0.55), (0.0, 25.0), 0.1, 1.2),
    # "an area of small pine trees": conical, branched almost to the ground
    Species("scots-pine", "young", True, 0.25, 9.0, (0.23,), (21,), (0.1, 0.16), 0.0, True, 0.02, 0.0, (0, 0), 28,
            0.21, 0.45, 0.15, 0.55, 0.0, (0.3, 0.6), (-5.0, 15.0), 0.09, 2.4),
    # "two elderly yews": a short, thick, many-stemmed bole under a dense, dark crown down almost to the ground
    Species("yew", "", True, 0.2, 12.0, (0.22, 0.24), (31, 32), (0.08, 0.14), 0.4, False, 0.05, 0.1, (2, 3), 26,
            0.38, 0.25, 0.12, 0.2, 0.8, NO_TURN, (60.0, 80.0), 0.07),
    # "oak trees and a carpet of fallen leaves": a low fork into spreading limbs and a broad, uneven crown
    Species("oak", "", False, 0.4, 16.0, (0.48, 0.54, 0.6), (41, 42, 43), (0.22, 0.3), 0.4, False, 0.03, 0.28, (4, 6),
            30, 0.24, 0.45, 0.3, 0.2, 0.45, NO_TURN, (15.0, 45.0), 0.1, 1.2),
    # "a towering beech tree": a tall, smooth bole, steep limbs, a dense dome
    Species("beech", "", False, 0.4, 20.0, (0.33, 0.39, 0.45), (51, 52, 53), (0.3, 0.38), 0.45, False, 0.025, 0.38,
            (3, 5), 30, 0.24, 0.5, 0.2, 0.35, 0.5, NO_TURN, (45.0, 70.0), 0.08, 1.25),
    # the site's Highland birch: a slender white stem, dark at its foot, under a narrow, light, open crown
    Species("birch", "", False, 0.2, 15.0, (0.2, 0.27), (61, 62), (0.25, 0.35), 0.35, False, 0.016, 0.0, (0, 0), 34,
            0.2, 0.3, 0.3, 0.0, 0.0, (0.05, 0.22), (35.0, 60.0), 0.05, 1.7),
)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def envelope(sp, t):
    """The crown's radius at fraction `t` of its depth up from its base, over its widest."""
    t = np.clip(t, 0.0, 1.0)
    if sp.cone:
        return (1 - t) * (0.75 + 0.25 * smoothstep(t / 0.15))
    m = sp.widest
    u = np.where(t < m, (t - m) / m, (t - m) / (1 - m))
    return np.sqrt(np.clip(1 - u * u, 0.0, 1.0))


def clumps(rng, sp, height, reach, base_y):
    """Leaf clumps' centres and radii in the envelope's outer shell, with one at the top to keep the crown's height."""
    depth, phase = height - base_y, rng.uniform(0, 2 * np.pi, 2)
    centres, radii = [], []
    while len(centres) < sp.clumps - 1:
        t, a = rng.uniform(0.03, 0.97), rng.uniform(0, 2 * np.pi)
        e = float(envelope(sp, t))
        if rng.random() > e:                       # as many clumps as the crown is wide at that height
            continue
        lobe = 1 + sp.lobes * (0.6 * math.sin(3 * a + phase[0] + 2 * t) + 0.4 * math.sin(5 * a + phase[1] - 3 * t))
        r = sp.clump_r * reach * rng.uniform(0.8, 1.2) * (0.6 + 0.4 * e if sp.cone else 1.0)
        rho = rng.uniform(sp.shell, 1.0) * max(reach * e * lobe - 0.7 * r, 0.0)
        y = min(max(base_y + t * depth, base_y + r), height - 0.8 * r)
        centres.append((rho * math.cos(a), y, rho * math.sin(a)))
        radii.append(r)
    r = sp.clump_r * reach * (0.5 if sp.cone else 0.8)
    centres.append((0.0, height - 0.8 * r, 0.0))
    radii.append(r)
    return np.array(centres), np.array(radii)


class Wood(NamedTuple):
    pts: np.ndarray
    r: np.ndarray
    kind: int                  # 0 trunk, 1 limb, 2 branch
    clump: int = -1            # the clump a branch or a limb feeds, if one


def angled(d, rise):
    """`d` turned up or down to an angle over the horizontal within `rise` (degrees)."""
    flat = np.array([d[0], 0.0, d[2]])
    flat = unit(flat) if np.linalg.norm(flat) > 1e-9 else np.array([1.0, 0.0, 0.0])
    a = np.clip(math.degrees(math.asin(np.clip(d[1], -1, 1))), *rise)
    return unit(flat * math.cos(math.radians(a)) + Y * math.sin(math.radians(a)))


def skeleton(rng, sp, height, base_y, centres, radii):
    """The trunk, its limbs, and a branch out to every clump that no limb reaches."""
    sink, r0 = SINK * height, sp.trunk_r * height
    top = sp.fork * height if sp.fork else 0.94 * height
    lean = unit(Y + np.array([rng.normal(0, 0.03), 0.0, rng.normal(0, 0.03)]))
    pts, r, _, _ = grow(rng, np.array([0.0, -sink, 0.0]), lean, top + sink, r0, r0 * (0.7 if sp.fork else 0.12),
                        (height / 5, height / 3.5), (2.0, 6.0), 0.6, spacing=(top + sink) / 7)
    pts, r = ringed(pts, r, foot(sp, height))
    r = r * (1 + FLARE * np.exp(-(pts[:, 1] + sink) / (FLARE_H * height)))
    wood = [Wood(pts, r, 0)]
    holders = [wood[0]]
    if sp.fork:
        fork, bearing = pts[-1], np.arctan2(centres[:, 2], centres[:, 0])
        k = int(rng.integers(sp.limbs[0], sp.limbs[1] + 1))
        sector = ((bearing - rng.uniform(0, 2 * np.pi)) % (2 * np.pi) * k / (2 * np.pi)).astype(int)
        for s in range(k):
            mine = np.flatnonzero(sector == s)
            if not len(mine):
                continue
            target = np.average(centres[mine], axis=0, weights=radii[mine] ** 2)
            d = angled(unit(target - fork), sp.rise)
            length = max(0.8 * np.linalg.norm(target - fork), 1.0)
            rl = r[-1] * min(0.8, 1.6 / math.sqrt(k))
            lp, lr, _, _ = grow(rng, fork, d, length, rl, 0.25 * rl, (length / 3, length / 2), (8.0, 18.0), 0.15,
                                spacing=length / 4)
            wood.append(Wood(lp, lr, 1))
        holders = wood[1:] or holders
    near = np.concatenate([h.pts for h in holders])
    near_r = np.concatenate([h.r for h in holders])
    for c, (centre, cr) in enumerate(zip(centres, radii)):
        if sp.fork:
            j = int(np.argmin(np.linalg.norm(near - centre, axis=1)))
        else:                                       # off the leader, at the height a branch at its angle leaves it
            out = math.hypot(centre[0], centre[2])
            rise = math.radians(float(np.clip(rng.uniform(*sp.rise), -80, 80)))
            j = int(np.argmin(np.abs(near[:, 1] - (centre[1] - out * math.tan(rise)))))
            j = min(j, len(near) - 2)
        start = near[j]
        gap = np.linalg.norm(centre - start)
        if gap < 0.8 * cr:
            continue
        rb = min(0.5 * near_r[j], 0.012 * height)
        bp, br, _, _ = grow(rng, start, unit(centre - start), gap, rb, 0.2 * rb, (gap / 2, gap / 1.5), (6.0, 14.0),
                            0.1, spacing=gap / 2)
        wood.append(Wood(bp, br, 2, c))
    return wood


def foot(sp, height):
    """Where the bark starts, is halfway and ends turning: trunk rings there keep the page's interpolation of the turn
    off the foot."""
    if sp.upper[1] > sp.upper[0]:  # rings
        return height * np.array([sp.upper[0], sum(sp.upper) / 2, sp.upper[1]])
    return np.zeros(0)


def ringed(pts, r, ys):
    """A rising polyline (and its radii) with points added at heights ys along it."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    ys = [y for y in ys if pts[0, 1] < y < pts[-1, 1] and np.min(np.abs(pts[:, 1] - y)) > 1e-3]
    at = np.sort(np.concatenate([s, np.interp(ys, pts[:, 1], s)]))
    return np.stack([np.interp(at, s, pts[:, k]) for k in range(3)], 1), np.interp(at, s, r)


def coarse(w, every):
    keep = np.unique(np.append(np.arange(0, len(w.pts), every), len(w.pts) - 1))
    return w._replace(pts=w.pts[keep], r=w.r[keep])


def turned(pos, sp, height):
    """Bark attributes: how far up the stem it has turned to its upper colour."""
    turn = (np.round(255 * smoothstep((pos[:, 1] / height - sp.upper[0]) / (sp.upper[1] - sp.upper[0])))
            if sp.upper[1] > sp.upper[0] else np.zeros(len(pos)))
    return np.stack([np.full(len(pos), BARK), turn], 1).astype(np.uint8)


def bark(wood, segments, sp, height, cap=True):
    parts = []
    for w in wood:
        m = tube(w.pts, w.r, (BARK, 0), segments[w.kind], cap=cap and w.kind == 0)
        if w.kind == 2:            # a thin branch's long slivers: each ring's radial normals, steady through its kinks
            m = m._replace(nrm=unit_rows(m.pos - np.repeat(w.pts, segments[2], axis=0)))
        parts.append(with_uv(m._replace(attr=turned(m.pos, sp, height))))
    return parts


class Crown(NamedTuple):
    centre: np.ndarray
    scale: np.ndarray          # the crown's half-extents: its reach, half its depth, its reach

    def outward(self, pos):
        return unit_rows((pos - self.centre) / self.scale ** 2)

    def normals(self, pos, local, lean):
        """The crown's own normal at each point, mostly, and a little of the surface's own (`local`, unit), so the leaves
        light as a mass; held at least `lean` to the surface's outer side, so each face still faces along its normals."""
        return leaning(unit_rows(0.7 * self.outward(pos) + 0.3 * local), local, lean)

    def depth(self, pos):
        d = np.linalg.norm((pos - self.centre) / self.scale, axis=1)
        return np.round(255 * np.clip(1 - d, 0, 1))


def unit_rows(v):
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def leaning(n, local, lean):
    """Unit normals `n` held at least `lean` (a cosine) towards `local`."""
    k = np.einsum("ij,ij->i", n, local)
    return unit_rows(n + np.maximum(lean - k, 0.0)[:, None] * local)


def held(m):
    """The leaves' normals held to their faces again after a fit has squeezed the tree: each vertex's normal at least
    LEAN_IN towards the faces round it (a card's own facing; a lump's local surface)."""
    nrm = m.nrm.copy()
    for part, lean in ((CARD, LEAN_IN[0]), (FOLIAGE, LEAN_IN[1])):
        tris = m.idx[m.attr[m.idx[:, 0], 0] == part]
        if not len(tris):
            continue
        local = vertex_normals(m.pos, tris, m.nrm)
        mine = np.unique(tris)
        nrm[mine] = leaning(nrm[mine], local[mine], lean)
    return m._replace(nrm=nrm)


def cards(rng, sp, crown, centres, radii, counts, size=1.0, floor=None, bound=None):
    """Leaf cards: square quads in each clump, each wound to face along its own normal. A card that would hang below
    `floor` is lifted to it, and one that would reach past `bound` (reach, top) is drawn in towards the trunk or down,
    so larger cards fill the crown without moving its edges."""
    pos, nrm, uv, idx = [], [], [], []
    corner = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    for centre, cr, n in zip(centres, radii, counts):
        for _ in range(int(n)):
            o = centre + np.clip(rng.normal(0, 0.35, 3), -0.6, 0.6) * cr
            d = unit(rng.normal(size=3))
            d = unit((1 - sp.flat) * d + sp.flat * Y * (1.0 if d[1] >= 0 else -1.0))
            d = d if d @ crown.outward(o[None])[0] >= 0 else -d       # its face turned out of the crown
            u = rotate(perpendicular(d), d, rng.uniform(0, 2 * np.pi))
            v = np.cross(d, u)
            half = 0.75 * cr * size * rng.uniform(0.9, 1.2)
            span = min(2 * half * LEAF_CELLS / (LEAF_GRID * LEAF_TILE * sp.leaf), 1.0)   # its leaves sp.leaf long
            at = rng.uniform(0, 1 - span, 2) if span < 1 else np.zeros(2)
            quad = o + half * (corner[:, :1] * u + corner[:, 1:] * v)
            if bound is not None:
                out = math.hypot(o[0], o[2])
                over = float(np.hypot(quad[:, 0], quad[:, 2]).max()) - bound[0]
                if over > 0 and out > 0:
                    quad[:, [0, 2]] -= min(over, out) * np.array([o[0], o[2]]) / out
                quad[:, 1] -= max(float(quad[:, 1].max()) - bound[1], 0.0)
            if floor is not None:
                quad[:, 1] += max(floor - quad[:, 1].min(), 0.0)
            n4 = crown.normals(quad, np.tile(d, (4, 1)), LEAN_IN[0])
            tri = [0, 1, 2, 0, 2, 3] if np.cross(quad[1] - quad[0], quad[2] - quad[0]) @ d > 0 else [0, 2, 1, 0, 3, 2]
            idx.append(np.array(tri).reshape(2, 3) + 4 * len(pos))
            pos.append(quad)
            nrm.append(n4)
            uv.append(at + span * (corner + 1) / 2)
    if not pos:
        return None
    pos = np.concatenate(pos)
    attr = np.stack([np.full(len(pos), CARD), crown.depth(pos)], 1).astype(np.uint8)
    return TreeMesh(pos, np.concatenate(nrm), attr, np.concatenate(idx).astype(np.uint32),
                    np.concatenate(uv).astype(np.float32))


def solid(m, crown, centre):
    """A solid lump of leaves, wound outwards and lit as part of the crown."""
    idx, own = faces_out(m.pos, m.idx, centre)
    nrm = crown.normals(m.pos, own, LEAN_IN[1])
    attr = np.stack([np.full(len(m.pos), FOLIAGE), crown.depth(m.pos)], 1).astype(np.uint8)
    return TreeMesh(m.pos, nrm, attr, idx.astype(np.uint32), np.zeros((len(m.pos), 2), np.float32))


def with_uv(m):
    return TreeMesh(m.pos, m.nrm, m.attr, m.idx, np.zeros((len(m.pos), 2), np.float32))


def joined(parts):
    parts = [p for p in parts if p is not None and len(p.pos)]
    offsets = np.cumsum([0] + [len(p.pos) for p in parts[:-1]])
    return TreeMesh(*(np.concatenate([getattr(p, f) for p in parts]) for f in ("pos", "nrm", "attr")),
                    np.concatenate([p.idx + o for p, o in zip(parts, offsets)]).astype(np.uint32),
                    np.concatenate([p.uv for p in parts]))


def groups(pts, weight, k, rng):
    """k clusters of the points (seeded k-means from a farthest-point start): each point's cluster, and the centres."""
    k = min(k, len(pts))
    chosen = [int(rng.integers(len(pts)))]
    for _ in range(1, k):
        chosen.append(int(np.argmax(np.min(np.linalg.norm(pts[:, None] - pts[chosen][None], axis=2), axis=1))))
    centre = pts[chosen].copy()
    for _ in range(12):
        label = np.argmin(np.linalg.norm(pts[:, None] - centre[None], axis=2), axis=1)
        for g in range(k):
            if np.any(label == g):
                centre[g] = np.average(pts[label == g], axis=0, weights=weight[label == g])
    return label, centre


def shares(total, weight):
    """`total` split in proportion to `weight`, in whole numbers that sum to it."""
    raw = total * weight / weight.sum()
    out = np.floor(raw).astype(int)
    out[np.argsort(out - raw)[:total - out.sum()]] += 1
    return out


def full_level(rng, sp, wood, crown, centres, radii, height):
    tris, verts = BUDGETS[0]
    parts = bark(wood, SEGMENTS[0], sp, height)
    if sp.core:
        big = np.argsort(-radii)[:len(radii) // 3]
        parts += [solid(blob(centres[c], radii[c] * sp.core, (FOLIAGE, 0), rng, lumps=0.12), crown, centres[c])
                  for c in big]
    used_v, used_t = sum(len(p.pos) for p in parts), sum(len(p.idx) for p in parts)
    n = min((verts - used_v) // 4, (tris - used_t) // 2)
    return joined(parts + [cards(rng, sp, crown, centres, radii, shares(n, radii ** 2), size=sp.card)])


def middle_level(rng, sp, wood, crown, centres, radii, height, size, floor, bound):
    tris, verts = BUDGETS[1]
    held = [coarse(w, 2) for w in wood if w.kind < 2]
    held[0] = held[0]._replace(**dict(zip(("pts", "r"), ringed(held[0].pts, held[0].r, foot(sp, height)))))
    parts = bark(held, SEGMENTS[1], sp, height)
    used_v, used_t = sum(len(p.pos) for p in parts), sum(len(p.idx) for p in parts)
    k = max(1, min((verts - used_v) // 8, (tris - used_t) // 4))
    label, centre = groups(centres, radii ** 2, k, rng)
    reach = np.array([math.sqrt(np.mean(radii[label == g] ** 2)) + 0.6 * math.sqrt(np.mean(
        np.sum((centres[label == g] - centre[g]) ** 2, axis=1))) if np.any(label == g) else 0.0 for g in range(len(centre))])
    keep = reach > 0
    return joined(parts + [cards(rng, sp, crown, centre[keep], reach[keep], np.full(int(keep.sum()), 2), size=size,
                                  floor=floor, bound=bound)])


def cone(crown, base_y, height, reach, sides=7):
    """A cone over a shallow underside, each with its own rim, so neither's normals turn against the other's faces."""
    a = 2 * np.pi * np.arange(sides) / sides
    rim = np.stack([reach * np.cos(a), np.full(sides, base_y), reach * np.sin(a)], 1)
    pos = np.vstack([rim, [[0.0, height, 0.0]], rim, [[0.0, base_y + 0.15 * (height - base_y), 0.0]]])
    i = np.arange(sides)
    idx = np.vstack([np.stack([i, np.full(sides, sides), (i + 1) % sides], 1),
                     np.stack([sides + 1 + i, sides + 1 + (i + 1) % sides, np.full(sides, 2 * sides + 1)], 1)])
    m = TreeMesh(pos, np.zeros_like(pos), np.zeros((len(pos), 2), np.uint8), idx, np.zeros((len(pos), 2), np.float32))
    return solid(m, crown, np.array([0.0, base_y + 0.35 * (height - base_y), 0.0]))


def coarse_level(rng, sp, wood, crown, centres, radii, height, base_y, reach, full, scale):
    """A trunk and two solid lumps of leaves (one, for a narrow crown), or a cone for a young pine; the trunk runs up
    into the lowest lump, so none floats over it."""
    lumps, tops = [], []
    sx, sy = scale
    if sp.cone:
        lumps.append(cone(crown, base_y, base_y + sy * (height - base_y), sx * reach))
        tops.append(base_y + 0.15 * (height - base_y))
    else:
        label, centre = groups(centres, radii ** 2, 2 if reach > 0.35 * height else 1, rng)
        leaf = full.pos[full.attr[:, 0] != BARK]       # each lump spans the full level's leaves nearest its clumps
        nearest = label[np.argmin(np.linalg.norm(leaf[:, None] - centres[None], axis=2), axis=1)]
        for g in range(len(centre)):
            mine = leaf[nearest == g]
            if not len(mine):
                continue
            lo, hi = mine.min(0), mine.max(0)
            mid, half = (lo + hi) / 2, (hi - lo) / 2 * np.array([sx, 1.0, sx])
            m = blob(mid, half, (FOLIAGE, 0), rng, lumps=0.12)
            # its own lowest and highest points set from the same base, whatever its height
            y0, y1 = lo[1] + 0.05 * (hi - lo)[1], lo[1] + (0.05 + 0.9 * sy) * (hi - lo)[1]
            y = m.pos[:, 1]
            m = m._replace(pos=np.column_stack([m.pos[:, 0], y0 + (y - y.min()) / (y.max() - y.min()) * (y1 - y0),
                                                m.pos[:, 2]]))
            mid[1] = (y0 + y1) / 2
            lumps.append(solid(m, crown, mid))
            tops.append(mid[1])
    trunk = wood[0]
    j = max(int(np.argmin(np.abs(trunk.pts[:, 1] - min(tops)))), 1)
    top = trunk.pts[j].copy()
    top[1] = max(top[1], min(tops))                    # past a fork, on up into the lump
    stem = trunk._replace(pts=np.stack([trunk.pts[0], top]), r=trunk.r[[0, j]])
    return joined(bark([stem], SEGMENTS[2], sp, height, cap=False) + lumps)


def extent(m):
    return float(np.hypot(m.pos[:, 0], m.pos[:, 2]).max()), float(m.pos[:, 1].max())


def scaled(m, k, ky):
    scale, inverse = np.array([k, ky, k]), np.array([1 / k, 1 / ky, 1 / k])
    return held(m._replace(pos=m.pos * scale, nrm=unit_rows(m.nrm * inverse)))


def within(m, full):
    """The level shrunk to the full level's reach and height where it overshoots them."""
    (crown, height), (reach, top) = extent(full), extent(m)
    return scaled(m, min(1.0, crown / reach), min(1.0, height / top)) if reach > crown or top > height else m


def fitted(levels, ratio):
    """The levels squeezed or stretched across the trunk's axis so that the full level's crown radius is `ratio` of its
    height; a coarser level that reaches further or higher is shrunk to the full level's extent first."""
    crown, height = extent(levels[0])
    s = ratio * height / crown
    return [scaled(within(m, levels[0]), s, 1.0) for m in levels], s


def coverage(m, turns=(0.0, math.pi / 2), cells=256):
    """The leaves' silhouette seen from the side, in square metres, averaged over `turns` about the trunk and sampled at
    the centres of a grid `cells` rows up the tree: solid leaves cover their outline, and each leaf card (its two
    triangles taken as one quad) CARD_FILL of its own, independently of the others."""
    tris = m.idx[m.attr[m.idx[:, 0], 0] != BARK]
    card = m.attr[tris[:, 0], 0] == CARD
    pieces = [[t] for t in tris[~card]]
    firsts = tris[card].min(axis=1)                       # a card's two triangles share its first corner
    pieces += [list(tris[card][firsts == f]) for f in np.unique(firsts)]
    is_card = [False] * int((~card).sum()) + [True] * (len(pieces) - int((~card).sum()))
    per_m = cells / max(float(m.pos[:, 1].max()), 1.0)
    out = []
    for t in turns:
        u = (math.cos(t) * m.pos[:, 0] + math.sin(t) * m.pos[:, 2]) * per_m
        v = m.pos[:, 1] * per_m
        u0, v0 = math.floor(u.min()), math.floor(v.min())
        solid = np.zeros((math.ceil(v.max()) - v0 + 1, math.ceil(u.max()) - u0 + 1), bool)
        count = np.zeros(solid.shape, np.int16)
        for piece, c in zip(pieces, is_card):
            corners = np.unique(np.concatenate(piece))
            i0, i1 = math.floor(u[corners].min()) - u0, math.ceil(u[corners].max()) - u0
            j0, j1 = math.floor(v[corners].min()) - v0, math.ceil(v[corners].max()) - v0
            pu, pv = np.meshgrid(np.arange(i0, i1) + u0 + 0.5, np.arange(j0, j1) + v0 + 0.5)
            inside = np.zeros(pu.shape, bool)
            for a, b, cc in piece:
                d = (v[b] - v[cc]) * (u[a] - u[cc]) + (u[cc] - u[b]) * (v[a] - v[cc])
                if abs(d) < 1e-12:
                    continue
                l1 = ((v[b] - v[cc]) * (pu - u[cc]) + (u[cc] - u[b]) * (pv - v[cc])) / d
                l2 = ((v[cc] - v[a]) * (pu - u[cc]) + (u[a] - u[cc]) * (pv - v[cc])) / d
                inside |= (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
            if c:
                count[j0:j1, i0:i1] += inside
            else:
                solid[j0:j1, i0:i1] |= inside
        out.append(np.where(solid, 1.0, 1 - (1 - CARD_FILL) ** count).sum() / per_m ** 2)
    return float(np.mean(out))


def matched(build, choices, full):
    """Of the levels `build` makes for each of `choices`, the one whose leaves' silhouette is nearest the full level's,
    so a tree neither thins nor thickens as it changes level: among those that keep its crown's base and top, the
    widest within COVER_BAND of it, or else the nearest. Cover comes before reach: solid lumps the full crown's size
    cover twice an airy crown's sparse cards (a birch's, a young pine's tiers), so theirs are narrower."""
    target, low = coverage(full), lowest_leaf(full)
    (reach, height), top = extent(full), highest_leaf(full)
    built = [build(c) for c in choices]
    # a level that overshoots the full one is shrunk to it, trunk and all: allow only a slight shrink
    levels = [within(m, full) for m in built if extent(m)[0] <= reach / FIT and extent(m)[1] <= height / FIT]
    shaped = [m for m in levels if abs(lowest_leaf(m) - low) <= LOW_SLACK * height and highest_leaf(m) >= HOLD * top]
    if not shaped:
        raise ValueError("no candidate level keeps the crown's base and top within the full level's")
    off = [abs(math.log(coverage(m) / target)) for m in shaped]
    near = [m for m, o in zip(shaped, off) if o <= math.log(COVER_BAND)]
    return max(near, key=leaf_reach) if near else shaped[int(np.argmin(off))]


def lowest_leaf(m):
    return float(m.pos[m.attr[:, 0] != BARK, 1].min())


def highest_leaf(m):
    return float(m.pos[m.attr[:, 0] != BARK, 1].max())


def leaf_reach(m):
    leaf = m.pos[m.attr[:, 0] != BARK]
    return float(np.hypot(leaf[:, 0], leaf[:, 2]).max())


def variant(sp, k, seed, ratio):
    rng = np.random.default_rng(seed)
    height = sp.height * rng.uniform(0.92, 1.08)
    reach = ratio * height
    base_y = height * rng.uniform(*sp.base)
    centres, radii = clumps(rng, sp, height, reach, base_y)
    wood = skeleton(rng, sp, height, base_y, centres, radii)
    crown = Crown(np.array([0.0, (base_y + height) / 2, 0.0]), np.array([reach, (height - base_y) / 2, reach]))
    full = full_level(rng, sp, wood, crown, centres, radii, height)
    middle = matched(lambda size: middle_level(np.random.default_rng([seed, 1]), sp, wood, crown, centres, radii, height,
                                               size, lowest_leaf(full), extent(full)), CARD_SIZES, full)
    coarsest = matched(lambda scale: coarse_level(np.random.default_rng([seed, 2]), sp, wood, crown, centres, radii,
                                                  height, base_y, reach, full, scale), LUMP_SCALES, full)
    levels, s = fitted([full, middle, coarsest], ratio)
    leaves = levels[0].pos[levels[0].attr[:, 0] != BARK]
    trunk = wood[0]
    r_ground = float(np.interp(0.0, trunk.pts[:, 1], trunk.r))
    return Variant(f"{sp.name}{'-' + sp.form if sp.form else ''}-{k + 1}", sp.name, sp.conifer,
                   sp.share / len(sp.seeds),
                   max(float(np.hypot(m.pos[:, 0], m.pos[:, 2]).max()) for m in levels),
                   max(float(m.pos[:, 1].max()) for m in levels), float(leaves[:, 1].min()), r_ground * s,
                   tuple(levels))


@lru_cache(maxsize=1)
def _kit():
    return tuple(variant(sp, k, seed, ratio) for sp in SPECIES for k, (seed, ratio) in enumerate(zip(sp.seeds, sp.ratios)))


def kit():
    """Every variant of every species, conifers first; no placements."""
    return list(_kit())


def kit_colours():
    """{species: {"bark", "bark_upper", "foliage": linear rgb, "autumn_share": float}}."""
    return {name: dict(c) for name, c in COLOURS.items()}
