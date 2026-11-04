"""The Whomping Willow (mesh-willow): a static tree from a seeded build-time branch generator.

The skeleton follows White Hound and JKR's sketch: a short thick trunk that splits low into six or seven thick, sinuous
limbs, each bent at knuckles (a kink with a bulge); unevenly spread finer branches and whippy twigs whose tips curl in
like clenched fingers; flared buttress roots, one arching out of the ground, framing a gap with a tunnel mouth that
faces the lawn between the castle doors and Hagrid's hut (willow-root-gap); a knot low on the trunk; and lower branches
that stay clear of the ground. It is in leaf, as the scene is: grey-green clumps on the twigs, light enough to show the
limbs. Everything is built in a local frame on the trunk's base (x east, y up, z north) over the ground relative to it,
then moved to the mapped place. One seed fixes the tree, so the payload rebuilds byte for byte."""
import math
from typing import NamedTuple

import numpy as np

from materials import code
from mesh_geometry import Mesh, concat

FAMILY = "willow"
BARK, KNOT, HOLE, LEAVES = (code(n) for n in ("willow-bark", "willow-knot", "willow-root-hole", "willow-leaves"))
SEED = 1971                         # "They planted the Whomping Willow the same year that I arrived at Hogwarts."

TRUNK_R, FORK_R, FORK_M, SINK_M = 0.95, 0.72, 2.4, 1.0
FLARE, FLARE_M = 0.6, 0.45          # the trunk's base flares by this much, decaying over this height
RIDGES, RIDGE_DEPTH = 11, 0.07      # bark ridges round the trunk
TRUNK_SEGMENTS = 64
LEAN = 0.08                         # radians
LIMBS = 7
LIMB_R, LIMB_LEN, LIMB_RISE = (0.32, 0.44), (7.0, 9.5), (0.35, 0.9)
JOINT_M, KINK_DEG, TROPISM = (1.2, 1.8), (24.0, 38.0), 0.14
KNUCKLE = 1.35                      # a knuckle's bulge over the limb's radius
BRANCHES, BRANCH_LEN = (2, 6), (1.8, 4.2)
TWIGS, TWIG_LEN, TWIG_R = (2, 5), (0.9, 2.0), 0.015
BRANCH_STEP_M, TWIG_STEP_M = 0.2, 0.24  # sample spacing along branches and twigs
CROWN_M = 11.0                      # no growth reaches further from the trunk's axis
ROOTS, ROOT_LEN, ROOT_R = 7, (2.0, 2.4), 0.36
ROOT_START_M = 0.45                 # roots start deep in the trunk, so their open ends stay hidden
ROOT_ZONE_M = 3.0                   # roots stay inside this radius; everything beyond it clears the ground
GAP_HALF_DEG, RECESS = 65.0, 0.88   # the hollow's half-width round the trunk, and how deep it undercuts the trunk
ROOT_GAP_DEG = 105.0                # the roots leave the front clear for the hollow and the knot beside it
MOUTH_W, MOUTH_H, MOUTH_M = 0.75, 0.55, 0.3   # the tunnel mouth's width and height, its centre over the ground
MOUTH_RIM_M, MOUTH_DIVE = 0.5, 1.0  # the mouth's rim from the trunk's axis; the tunnel's fall per metre inwards
KNOT_M, KNOT_DEG = 0.4, 75.0        # the knot's height over the ground and its bearing from the gap
CLUMP_M = (0.45, 0.85)              # leaf clump radius
MASS_M, MASS_LUMPS = (1.0, 2.2), 0.22   # leaf mass radius round a branch's twig ends, and how lumpy it is
Y = np.array([0.0, 1.0, 0.0])


class Branch(NamedTuple):
    pts: np.ndarray                 # (k, 3) centreline, local metres
    r: np.ndarray                   # (k,) radius
    level: int                      # 0 trunk, 1 limb, 2 branch, 3 twig, -1 root


def unit(v):
    return v / np.linalg.norm(v)


def rotate(v, axis, angle):
    axis = unit(axis)
    return v * math.cos(angle) + np.cross(axis, v) * math.sin(angle) + axis * (axis @ v) * (1 - math.cos(angle))


def perpendicular(d):
    return unit(np.cross(d, Y if abs(d[1]) < 0.95 else np.array([1.0, 0.0, 0.0])))


def spline(ctrl, spacing):
    """Catmull-Rom samples through the control points about `spacing` apart, and each control point's sample index."""
    c = np.concatenate([ctrl[:1] * 2 - ctrl[1:2], ctrl, ctrl[-1:] * 2 - ctrl[-2:-1]])
    out, at = [], []
    for i in range(1, len(c) - 2):
        p0, p1, p2, p3 = c[i - 1], c[i], c[i + 1], c[i + 2]
        n = max(1, int(math.ceil(np.linalg.norm(p2 - p1) / spacing)))
        at.append(len(out))
        for t in np.arange(n) / n:
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    at.append(len(out))
    out.append(ctrl[-1])
    return np.array(out), at


def spacing_of(r):
    return float(np.clip(0.5 * r + 0.08, 0.1, 0.3))


def grow(rng, start, d, length, r0, r1, joint, kink, tropism, curl=0.0, spacing=None):
    """A sinuous centreline: straight runs between joints, kinked alternately from side to side, drawn up by tropism;
    `curl` bends the last run in. Returns (samples, radii, joint sample indices)."""
    ctrl, side, sign, s = [start], perpendicular(d), 1.0 if rng.random() < 0.5 else -1.0, 0.0
    while s < length - 1e-6:
        run = min(rng.uniform(*joint), length - s)
        if length - s - run < 0.4 * joint[0]:
            run = length - s
        ctrl.append(ctrl[-1] + d * run)
        s += run
        twist = math.radians(rng.uniform(-35.0, 35.0))
        axis = rotate(np.cross(d, side), d, twist)
        d = unit(rotate(d, axis, sign * math.radians(rng.uniform(*kink))) + tropism * Y)
        side, sign = perpendicular(d), -sign
        if curl and length - s < joint[1]:
            d = unit(rotate(d, np.cross(d, Y) if abs(d[1]) < 0.95 else side, -curl))
    pts, at = spline(np.array(ctrl), spacing or spacing_of(r0))
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    r = r0 + (r1 - r0) * arc / arc[-1]
    return pts, r, at[1:-1], arc


def knuckled(r, arc, joints, bulge):
    out = r.copy()
    for j in joints:
        w = max(0.8 * r[j], 2 * spacing_of(r[j]))
        out *= 1 + (bulge - 1) * np.exp(-((arc - arc[j]) / w) ** 2)
    return out


def gap_cut(theta, y, gap):
    """How deeply the root gap undercuts the trunk (0 to 1) at bearing `theta` and height y over its base."""
    off = np.abs((theta - gap + np.pi) % (2 * np.pi) - np.pi) / math.radians(GAP_HALF_DEG)
    return smoothstep(np.clip((1 - off) / 0.3, 0, 1)) * smoothstep(np.clip((0.95 - y) / 0.5, 0, 1))


def trunk_radius(theta, y, gap):
    """The trunk's radius at bearing `theta` (radians from +x towards +z) and height y over its base: flared, ridged and
    undercut by the root gap."""
    base = TRUNK_R + (FORK_R - TRUNK_R) * np.clip(y / FORK_M, 0, 1)
    r = base * (1 + FLARE * np.exp(-np.maximum(y, 0) / FLARE_M)) * (1 + RIDGE_DEPTH * np.sin(RIDGES * theta + 2.0 * y))
    return r * (1 - RECESS * gap_cut(theta, y, gap))


def smoothstep(x):
    return x * x * (3 - 2 * x)


def tree(ground, gap=-math.pi / 2, seed=SEED):
    """The Willow's branches, from the trunk out, over `ground(x, z)` (heights relative to the trunk's base); the root
    gap faces bearing `gap` (radians from +x towards +z)."""
    rng = np.random.default_rng(seed)
    lean = rotate(Y, perpendicular(np.array([math.cos(gap + 2.1), 0.0, math.sin(gap + 2.1)])), LEAN)
    ys = np.linspace(-SINK_M, FORK_M, 36)
    axis = ys[:, None] * lean / lean[1]
    out = [Branch(axis, TRUNK_R + (FORK_R - TRUNK_R) * np.clip(ys / FORK_M, 0, 1), 0)]
    top = axis[-1]
    # the limbs, spread unevenly round the trunk: their bearings bunch towards one side
    bearings = np.sort((gap + np.pi * 0.9 + np.cumsum(rng.uniform(0.55, 1.35, LIMBS))) % (2 * np.pi))
    for k, b in enumerate(bearings):
        heavy = 0.5 + 0.5 * math.cos(b - gap - np.pi * 0.7)          # the side away from the gap grows more
        out_dir = np.array([math.cos(b), 0.0, math.sin(b)])
        d = unit(out_dir + rng.uniform(*LIMB_RISE) * Y)
        start = top + out_dir * 0.35 * FORK_R - 0.3 * Y
        r0 = rng.uniform(*LIMB_R)
        pts, r, joints, arc = grow(rng, start, d, rng.uniform(*LIMB_LEN) * (0.85 + 0.25 * heavy), r0, 0.07,
                                   JOINT_M, KINK_DEG, TROPISM)
        limb = pruned(Branch(pts, knuckled(r, arc, joints, KNUCKLE), 1))
        out.append(limb)
        out += offshoots(rng, limb, rng.integers(BRANCHES[0], BRANCHES[1] + 1) + round(2 * heavy))
    out += roots(rng, ground, gap)
    return out


def arc_of(pts):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])


def offshoots(rng, parent, count):
    """Branches along a limb, and twigs along each branch: shorter, thinner, more upright, the twigs whippy and curled."""
    out, arc = [], arc_of(parent.pts)
    for _ in range(count):
        level = parent.level + 1
        k = min(int(np.searchsorted(arc, arc[-1] * rng.uniform(0.3, 0.92))), len(arc) - 2)
        t = unit(parent.pts[k + 1] - parent.pts[k])
        d = rotate(t, rotate(perpendicular(t), t, rng.uniform(0, 2 * np.pi)), math.radians(rng.uniform(30, 58)))
        d = unit(d + 0.35 * Y)
        if d[1] < 0:                                   # nothing grows down towards the ground
            d = unit(d * np.array([1.0, -0.5, 1.0]))
        if level == 2:
            length, r0, r1 = rng.uniform(*BRANCH_LEN), parent.r[k] * rng.uniform(0.45, 0.6), 0.03
            pts, r, joints, a = grow(rng, parent.pts[k], d, length, r0, r1, (0.6, 1.0), (14.0, 28.0), 0.18,
                                     spacing=BRANCH_STEP_M)
            b = pruned(Branch(pts, knuckled(r, a, joints, 1.18), level))
            out.append(b)
            out += offshoots(rng, b, rng.integers(TWIGS[0], TWIGS[1] + 1))
        else:
            length = rng.uniform(*TWIG_LEN)
            r0 = max(parent.r[k] * 0.55, TWIG_R + 0.008)
            pts, r, _, _ = grow(rng, parent.pts[k], d, length, r0, TWIG_R, (0.35, 0.6), (10.0, 22.0), 0.25,
                                curl=-math.radians(30.0), spacing=TWIG_STEP_M)
            out.append(pruned(Branch(pts, r, level)))
    return out


def pruned(b):
    """The branch cut where it would reach beyond the crown (its first two samples always stay)."""
    far = np.flatnonzero(np.hypot(b.pts[:, 0], b.pts[:, 2]) > CROWN_M - CLUMP_M[1])
    if not len(far):
        return b
    keep = max(int(far[0]), 2)
    return Branch(b.pts[:keep], b.r[:keep], b.level)


def roots(rng, ground, gap):
    """Buttress roots following the ground, leaving the gap clear; the one beside the gap arches out of the ground."""
    free = 2 * math.radians(ROOT_GAP_DEG)
    bearings = gap + free / 2 + np.linspace(0, 2 * np.pi - free, ROOTS) + rng.uniform(-0.12, 0.12, ROOTS)
    out = []
    for k, b in enumerate(bearings):
        u = np.array([math.cos(b), 0.0, math.sin(b)])
        length = ROOT_LEN[1] if k == 0 else rng.uniform(*ROOT_LEN)
        t = np.linspace(0.0, 1.0, 28)
        rho = ROOT_START_M + t * length
        rr = ROOT_R * (1 - 0.8 * t) + 0.04
        x, z = rho * u[0], rho * u[2]
        if k == 0:                                     # Ron "hooked [his leg] around a root"
            lift = 0.3 + 0.25 * np.sin(np.pi * (t - 0.15) / 0.75) - 0.7 * smoothstep(np.clip((t - 0.62) / 0.38, 0, 1))
            y = ground(x, z) + rr * 0.2 + 0.05 + lift
        else:
            y = ground(x, z) + rr * 0.35 - 0.6 * t ** 3 + 0.35 * np.clip(1 - t / 0.6, 0, 1) ** 2
        out.append(Branch(np.stack([x, y, z], axis=1), rr, -1))
    return out


# ---------------------------------------------------------------- meshing
def segments_of(r):
    return int(np.clip(round(r * 28) + 4, 5, 16))


def tube(pts, r, attr, segments, radius=None, cap=True):
    """A swept tube with outward normals; `radius(theta, i)` overrides r per ring vertex (theta about the frame)."""
    n = len(pts)
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nx = np.empty_like(pts)
    nx[0] = perpendicular(t[0])
    for i in range(1, n):
        v = nx[i - 1] - (nx[i - 1] @ t[i]) * t[i]
        nx[i] = v / np.linalg.norm(v)
    bx = np.cross(t, nx)
    a = 2 * np.pi * np.arange(segments) / segments
    radial = np.cos(a)[None, :, None] * nx[:, None] + np.sin(a)[None, :, None] * bx[:, None]
    rr = radius(radial, np.arange(n)) if radius else np.broadcast_to(r[:, None], (n, segments))
    pos = pts[:, None] + rr[..., None] * radial
    i, j = np.meshgrid(np.arange(n - 1), np.arange(segments), indexing="ij")
    a0, b0 = i * segments + j, i * segments + (j + 1) % segments
    c0, d0 = a0 + segments, b0 + segments
    idx = np.concatenate([np.stack([a0, b0, c0], -1).reshape(-1, 3), np.stack([b0, d0, c0], -1).reshape(-1, 3)])
    pos, fallback = pos.reshape(-1, 3), radial.reshape(-1, 3)
    if cap:
        tip = len(pos)
        pos = np.vstack([pos, pts[-1] + t[-1] * r[-1]])
        fallback = np.vstack([fallback, t[-1]])
        last = (n - 1) * segments
        idx = np.vstack([idx, np.stack([last + np.arange(segments), last + (np.arange(segments) + 1) % segments,
                                        np.full(segments, tip)], -1)])
    return Mesh(pos, vertex_normals(pos, idx, fallback), np.tile(np.asarray(attr, np.uint8), (len(pos), 1)),
                idx.astype(np.uint32))


def vertex_normals(pos, idx, fallback):
    """Area-weighted normals of the faces round each vertex, so undercuts, bulges and tapers shade as they are built
    and every face agrees with its vertices; `fallback` where no face has area."""
    tris = pos[idx]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm = np.zeros_like(pos)
    for k in range(3):
        np.add.at(nrm, idx[:, k], fn)
    length = np.linalg.norm(nrm, axis=1, keepdims=True)
    return np.where(length > 1e-12, nrm / np.maximum(length, 1e-300), fallback)


def icosahedron(subdivisions=0):
    g = (1 + 5 ** 0.5) / 2
    v = [unit(np.array(p, float)) for p in ([-1, g, 0], [1, g, 0], [-1, -g, 0], [1, -g, 0], [0, -1, g], [0, 1, g],
                                            [0, -1, -g], [0, 1, -g], [g, 0, -1], [g, 0, 1], [-g, 0, -1], [-g, 0, 1])]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7),
         (9, 8, 1)]
    for _ in range(subdivisions):
        mid, nf = {}, []
        def m(a, b):
            if (a, b) not in mid:
                mid[a, b] = mid[b, a] = len(v)
                v.append(unit(v[a] + v[b]))
            return mid[a, b]
        for a, b, c in f:
            ab, bc, ca = m(a, b), m(b, c), m(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f, np.uint32)


def lumpy(v, rng, amp, waves=5, freq=2.6):
    """Unit directions pushed in and out by a few smooth random waves: a lumpy, rounded surface."""
    k = rng.normal(size=(waves, 3))
    k *= freq / np.linalg.norm(k, axis=1, keepdims=True)
    phase = rng.uniform(0, 2 * np.pi, waves)
    return v * (1 + amp * np.sin(v @ k.T + phase).mean(axis=1, keepdims=True))


def faces_out(pos, idx, centre):
    """Triangles wound to face away from `centre`, with area-weighted vertex normals from those faces."""
    tris = pos[idx]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = np.einsum("ij,ij->i", fn, tris.mean(axis=1) - centre) < 0
    idx = idx.copy()
    idx[flip] = idx[flip][:, [0, 2, 1]]
    return idx, vertex_normals(pos, idx, (pos - centre) / np.linalg.norm(pos - centre, axis=1, keepdims=True))


def blob(centre, radii, attr, rng, subdivisions=0, lumps=0.15, outward=None):
    """A lumpy ellipsoid; with `outward`, its normals lean that way (a canopy lit as one mass)."""
    v, f = icosahedron(subdivisions)
    yaw = rng.uniform(0, 2 * np.pi)
    rot = np.array([[math.cos(yaw), 0, -math.sin(yaw)], [0, 1, 0], [math.sin(yaw), 0, math.cos(yaw)]])
    pos = centre + (lumpy(v, rng, lumps) * radii) @ rot.T
    f, nrm = faces_out(pos, f, centre)
    if outward is not None:
        nrm = 0.7 * nrm + 0.3 * unit(outward)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return Mesh(pos, nrm, np.tile(np.asarray(attr, np.uint8), (len(pos), 1)), f)


def coarse(b, every):
    """The branch with every `every`-th sample kept, and always its last."""
    keep = np.unique(np.append(np.arange(0, len(b.pts), every), len(b.pts) - 1))
    return Branch(b.pts[keep], b.r[keep], b.level)


def trunk_mesh(trunk, gap, segments=TRUNK_SEGMENTS):
    def radius(radial, rows):
        theta = np.arctan2(radial[..., 2], radial[..., 0])
        return trunk_radius(theta, trunk.pts[rows, 1][:, None], gap)
    body = tube(trunk.pts, trunk.r, BARK, segments, radius, cap=False)
    # what is left of the trunk deep in the hollow is the dark earth beyond the tunnel's mouth
    lean = trunk.pts[-1][[0, 2]] / trunk.pts[-1][1]
    off = body.pos[:, [0, 2]] - body.pos[:, 1:2] * lean
    deep = (gap_cut(np.arctan2(off[:, 1], off[:, 0]), body.pos[:, 1], gap) > 0.9) & (body.pos[:, 1] < 0.5)
    body = body._replace(attr=np.where(deep[:, None], np.asarray(HOLE, np.uint8), body.attr))
    top = trunk.pts[-1]
    dome = tube(np.array([top, top + 0.25 * Y, top + 0.45 * Y]), np.array([FORK_R, 0.6 * FORK_R, 0.15]), BARK,
                segments)
    return concat([body, dome])


def mouth(ground, gap):
    """The tunnel: an open, dark, tapering tube diving under the trunk from the hollow at its base, seen from inside."""
    u = np.array([math.cos(gap), 0.0, math.sin(gap)])
    rim = MOUTH_RIM_M * u
    rim[1] = float(ground(rim[:1], rim[2:])[0]) + MOUTH_M
    d = unit(-u - MOUTH_DIVE * Y)
    side = unit(np.cross(Y, u))
    up = np.cross(d, side)
    a = 2 * np.pi * np.arange(16) / 16
    ellipse = np.cos(a)[:, None] * side * MOUTH_W / 2 + np.sin(a)[:, None] * up * MOUTH_H / 2

    def sleeve(depth, scale):
        """Rings of the mouth's ellipse along the dive, scaled, closed at the far end."""
        pts = rim + np.linspace(0.0, depth, 9)[:, None] * d
        pos = (pts[:, None] + ellipse[None] * scale[:, None, None]).reshape(-1, 3)
        i, j = np.meshgrid(np.arange(len(pts) - 1), np.arange(16), indexing="ij")
        a0, b0 = i * 16 + j, i * 16 + (j + 1) % 16
        idx = np.concatenate([np.stack([a0, a0 + 16, b0], -1).reshape(-1, 3),
                              np.stack([b0, a0 + 16, b0 + 16], -1).reshape(-1, 3)])
        end, last = len(pos), (len(pts) - 1) * 16
        pos = np.vstack([pos, pts[-1]])
        idx = np.vstack([idx, np.stack([last + (np.arange(16) + 1) % 16, last + np.arange(16), np.full(16, end)], -1)])
        return pos, idx.astype(np.uint32)

    axis = lambda p: rim + np.clip((p - rim) @ d, 0, None)[:, None] * d
    pos, idx = sleeve(1.3, np.linspace(1.0, 0.55, 9))
    nrm = axis(pos) - pos                              # seen from inside: normals towards the tunnel's axis
    nrm[-1] = -d
    return oriented_by(Mesh(pos, nrm / np.linalg.norm(nrm, axis=1, keepdims=True),
                            np.tile(np.asarray(HOLE, np.uint8), (len(pos), 1)), idx))


def oriented_by(m):
    """The mesh with each triangle wound to face the way its vertices' normals point."""
    tris = m.pos[m.idx]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = np.einsum("ij,ij->i", fn, m.nrm[m.idx].mean(axis=1)) < 0
    idx = m.idx.copy()
    idx[flip] = idx[flip][:, [0, 2, 1]]
    return m._replace(idx=idx)


def knot(ground, gap, seed):
    b = gap + math.radians(KNOT_DEG)
    u = np.array([math.cos(b), 0.0, math.sin(b)])
    y = float(ground(u[:1] * TRUNK_R, u[2:] * TRUNK_R)[0]) + KNOT_M
    r = float(trunk_radius(np.array([b]), np.array([y]), gap)[0])
    return blob(u * r + y * Y, (0.2, 0.15, 0.2), KNOT, np.random.default_rng([seed, 0]), subdivisions=1, lumps=0.1)


def foliage(branches, seed, subdivisions=2, clumps=True, grow=1.0):
    """Leaves: a lumpy mass round each branch's twig ends (grown by `grow` where coarser levels drop the clumps), and
    small clumps on twigs that reach out of it, their normals leaning out from the crown's middle so the canopy lights
    as one. Each mass and clump has its own seeded lumps, so every level puts them in the same places."""
    groups, tips = [], [b.pts[-1] for b in branches if b.level >= 2]
    centre = np.mean(tips, axis=0)
    for b in branches:
        if b.level == 2:
            groups.append([b])
        elif b.level == 3 and groups:
            groups[-1].append(b)
    out = []
    for k, group in enumerate(groups):
        ends = np.array([b.pts[-1] for b in group])
        mid = ends.mean(axis=0) + 0.2 * Y
        r = float(np.clip(0.8 * np.linalg.norm(ends - mid, axis=1).max() + 0.6, *MASS_M))
        r = max(min(r * grow, (CROWN_M + 0.3 - math.hypot(mid[0], mid[2])) / (1 + MASS_LUMPS)), MASS_M[0] / 2)
        out.append(blob(mid, (r, 0.75 * r, r), LEAVES, np.random.default_rng([seed, 1, k]), subdivisions=subdivisions,
                        lumps=MASS_LUMPS, outward=mid - centre))
        if clumps:
            for j, p in enumerate(ends[np.linalg.norm(ends - mid, axis=1) > 0.8 * r]):
                rng = np.random.default_rng([seed, 2, k, j])
                s = rng.uniform(*CLUMP_M)
                out.append(blob(p, (s, 0.8 * s, s), LEAVES, rng, subdivisions=1, outward=p - centre))
    return out


class Level(NamedTuple):
    depth: int          # the finest branch level drawn
    every: int          # keep every n-th sample along the wood
    segments: float     # ring segments as a fraction of the full level's (at least 4)
    trunk: int          # the trunk's ring segments
    leaves: int         # the leaf masses' subdivisions
    clumps: bool        # the small clumps on the twigs
    grow: float         # the leaf masses' growth, standing in for dropped twigs and clumps
    roots: bool
    details: bool       # the knot and the tunnel mouth


LEVELS = {
    "lod0": Level(3, 1, 1.0, TRUNK_SEGMENTS, 2, True, 1.0, True, True),
    "lod1": Level(2, 2, 0.5, 24, 1, False, 1.12, True, True),
    "lod2": Level(1, 3, 0.0, 10, 0, False, 1.2, True, False),
    "shadow": Level(2, 3, 0.0, 10, 1, False, 1.12, False, False),
}


def meshes(branches, ground, gap, seed=SEED, level=LEVELS["lod0"]):
    """The Willow's pieces in its local frame at one level of detail: trunk and roots with the knot and mouth, each
    limb's wood, the leaves."""
    seg = lambda r: max(4, round(segments_of(r) * level.segments))
    trunk = next(b for b in branches if b.level == 0)
    base = [trunk_mesh(coarse(trunk, level.every), gap, level.trunk)]
    if level.details:
        base += [mouth(ground, gap), knot(ground, gap, seed)]
    if level.roots:
        base += [tube(c.pts, c.r, BARK, seg(c.r[0])) for c in (coarse(b, level.every) for b in branches if b.level == -1)]
    wood, piece = [], []
    for b in branches:
        if b.level == 1 and piece:
            wood.append(concat(piece))
            piece = []
        if 1 <= b.level <= level.depth:
            c = coarse(b, level.every)
            piece.append(tube(c.pts, c.r, BARK, seg(c.r[0])))
    wood.append(concat(piece))
    leaves = foliage(branches, seed, level.leaves, level.clumps, level.grow)
    half = len(leaves) // 2
    return [concat(base)] + wood + [concat(leaves[:half]), concat(leaves[half:])]


def levels(ground, gap=-math.pi / 2, seed=SEED):
    """Every level of detail and the shadow caster, in the local frame. The payload carries every level
    (mesh_levels.willow_units), and the page selects among them per view."""
    branches = tree(ground, gap, seed)
    return {name: meshes(branches, ground, gap, seed, level) for name, level in LEVELS.items()}


def gap_bearing(meta):
    """willow-root-gap: towards the middle of the lawn between the castle's front doors and Hagrid's hut (radians
    from +x towards +z)."""
    at = np.array(meta["extras"]["willow"], float)
    named = {p["name"]: np.array(p["p"], float) for p in meta["extras"]["poi"]}
    v = (named["Entrance Hall & Great Hall"] + named["Hagrid's hut"]) / 2 - at
    return math.atan2(v[1], v[0])


# ---------------------------------------------------------------- family hook (build_meshes.massing)
def build_extras(scene):
    x0, z0 = scene.meta["extras"]["willow"]
    y0 = float(scene.ground(np.array([x0]), np.array([z0]))[0])
    ground = lambda x, z: scene.ground(np.asarray(x) + x0, np.asarray(z) + z0) - y0
    gap = gap_bearing(scene.meta)
    shift = np.array([x0, y0, z0])
    return {"willow": [m._replace(pos=m.pos + shift) for m in meshes(tree(ground, gap), ground, gap)]}
