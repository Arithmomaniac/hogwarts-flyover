"""The ground's look at the site as the page renders it: the cliffs' rock shows no regular pattern."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mesh_source import Grid, decode, load_data  # noqa: E402
from render_testing import FRAME_TIMEOUT_S, MAT_GROUND, RenderTestCase  # noqa: E402

DBG_NO_SUN_SHADOWS, DBG_NO_TREES, DBG_NO_CLOUD_SHADOWS, DBG_CLEAN_EDGES = 1, 8, 1024, 2048
# the cliff: the largest stretch of ground near the castle (CLIFF_NEAR_M) rising more than CLIFF_RISE per metre, seen
# square-on from ROCK_BACK_M in front of its middle at mid-height. Its rock (the ground's normal up to ROCK_NY) is laid
# out along and up the face in ROCK_CELL_M cells, the broad shading divided out (over ROCK_TREND_M), and over the
# largest rectangle its cells cover nearly whole, in its spectrum smoothed over three cells by three, no period of
# ROCK_PERIODS_M stands more than ROCK_PEAK times the median of the nine by nine round it. The rock's strata, 3.3 m
# apart, measured 9.9 times; rock textured on the face's own planes, 3.7
CLIFF_RISE, CLIFF_NEAR_M = 2.5, 450.0
ROCK_BACK_M = 45.0
ROCK_NY = 0.35
ROCK_CELL_M, ROCK_TREND_M = 0.25, 3.0
ROCK_FILL = 0.97
ROCK_PERIODS_M = (1.5, 8.0)
ROCK_PEAK = 6.0
# the beds' edge (frag4.glsl classMix): the straightest stretch of it on open ground (BEDS_OPEN_TEXELS from trees, roads
# and water) running slantwise to the texels, BEDS_SLANT degrees off them, seen from BEDS_VIEW_M above it, facing across
# it, at noon with the classes' noise and the shadows off. Across it the ground's colour turns from the beds' to the
# ground's beside them; traced every BEDS_STEP_M along the stretch, within BEDS_HALF_M either side of its middle, the line
# where it turns strays from its own running mean over BEDS_SPAN_M, about one of the texels' steps along it, by no more
# than BEDS_TEETH_M, root mean square. Taken where the beds' weight among the four texels round a point passed a half,
# it traced the texels' steps (tour 08): 0.155 m; by their B-spline share, 0.084 m (AMD)
BEDS_OPEN_TEXELS = 3
BEDS_SLANT = (15.0, 40.0)
BEDS_VIEW_M = 25.0
BEDS_STEP_M, BEDS_HALF_M, BEDS_SPAN_M = 0.25, 12.0, 3.0
BEDS_TEETH_M = 0.12
# Fails since wave 1: the castle's cliff drops through single texels slantwise to the grid, and its normal, taken from a
# short step, followed the bilinear cells' staircase, so its rock showed streaks every 3.4 m (24.5 times the median round
# them at b7cabbc, AMD). The steep-face normal (frag3.glsl ST_GWIDE) took the streaks out, but a period of 2.04 m along
# the face still stands 6.2 times the median round it (AMD): 8.4 with the rock's albedo flat and 6.1 with the wide normal
# unblended, so the face's shading still follows its cells somewhere. Lane D's, open.
EXPECTED_FAILURES = ("GroundLookTests.test_the_cliffs_rock_repeats_at_no_period",)


def cliff():
    """((east, north) of the cliff's middle, its outward horizontal normal, (foot, top) heights) from the local layer."""
    data = load_data()
    grid = Grid(data["meta"]["local"])
    bed = decode(data["tex"]["locBed"]).astype(float)
    texel = float(np.mean(grid.texel_size()))
    gi, gj = np.gradient(bed, texel)
    rise = np.hypot(gi, gj)
    ii, jj = np.mgrid[0:grid.ny, 0:grid.nx]
    steep = (rise > CLIFF_RISE) & (np.hypot(*grid.to_world(ii, jj)) < CLIFF_NEAR_M)
    lab, n = ndi.label(steep, structure=np.ones((3, 3), bool))
    rows, cols = np.nonzero(lab == 1 + int(np.argmax(ndi.sum(steep, lab, range(1, n + 1)))))
    k = len(rows) // 2                                          # its middle texel, in row-major order
    i, j = int(rows[k]), int(cols[k])
    down = -np.array([gi[i, j], gj[i, j]]) / rise[i, j]          # downhill, in texels
    at = np.array(grid.to_world(i, j), float)
    normal = np.array(grid.to_world(i + down[0], j + down[1]), float) - at
    near = bed[max(i - 4, 0):i + 5, max(j - 4, 0):j + 5]
    return at, normal / np.linalg.norm(normal), (float(near.min()), float(near.max()))


BEDS, SOFT = 4, (0, 1, 2, 5, 8)        # build_final.py's surface classes: the beds, and the soft ground (not hardClass)


def beds_edge():
    """((east, north) of the middle of the straightest stretch of the beds' edge on open ground running BEDS_SLANT
    degrees off the texels, the unit vector along it, and the one across it pointing out of the beds)."""
    from scipy.spatial import cKDTree

    from mesh_source import CHANNELS

    data = load_data()
    grid = Grid(data["meta"]["local"])
    cls = decode(data["tex"]["locN"])[..., CHANNELS["locN"].index("surface")].astype(int)
    la = decode(data["tex"]["locA"])
    busy = (la[..., CHANNELS["locA"].index("trees")] > 10) | (la[..., CHANNELS["locA"].index("road")] > 10)
    busy |= decode(data["tex"]["locW"]) > 0
    edge = (cls == BEDS) & ndi.binary_dilation(np.isin(cls, SOFT), ndi.generate_binary_structure(2, 1))
    pts = np.argwhere(edge & ~ndi.binary_dilation(busy, iterations=BEDS_OPEN_TEXELS)).astype(float)
    tree, best = cKDTree(pts), None
    for p in pts:
        near = pts[tree.query_ball_point(p, 8.0)]
        if len(near) < 12:
            continue
        _, spread, axes = np.linalg.svd(near - near.mean(axis=0), full_matrices=False)
        slant = math.degrees(math.atan2(abs(axes[0][0]), abs(axes[0][1])))
        if BEDS_SLANT[0] < min(slant, 90.0 - slant) < BEDS_SLANT[1] and (best is None or spread[1] / spread[0] < best[0]):
            best = (spread[1] / spread[0], near.mean(axis=0), axes[0])
    (i, j), (di, dj) = best[1], best[2]
    at = np.array(grid.to_world(i, j), float)
    along = np.array(grid.to_world(i + di, j + dj), float) - at
    along /= np.linalg.norm(along)
    out = np.array([along[1], -along[0]])
    r, c = (int(round(float(v))) for v in grid.to_texel(*(at + 3.0 * out)))
    return at, along, (-out if cls[r, c] == BEDS else out)


def edge_line(s, t, chroma):
    """For ground pixels at `s` along the beds' edge and `t` across it (out of the beds; metres), of log chromaticity
    `chroma`: for each BEDS_STEP_M along it, the offset across it of the line between the beds' colour and the ground's
    beside them, which best parts the pixels nearer each, or NaN where too few pixels lie."""
    mine, theirs = (np.median(chroma[(np.abs(s) < BEDS_HALF_M) & (k * t > 3.0) & (k * t < 5.0)], axis=0) for k in (-1, 1))
    beds = (chroma - theirs) @ (mine - theirs) / float((mine - theirs) @ (mine - theirs)) > 0.5
    line = []
    for lo in np.arange(-BEDS_HALF_M, BEDS_HALF_M, BEDS_STEP_M):
        m = (s >= lo) & (s < lo + BEDS_STEP_M) & (np.abs(t) < 4.0)
        if m.sum() < 20:
            line.append(np.nan)
            continue
        k = np.argsort(t[m])
        ts, inside = t[m][k], beds[m][k]
        split = int(np.argmax(np.cumsum(inside) - np.cumsum(~inside)))   # the beds before it, the ground after
        line.append(0.5 * (ts[split] + ts[min(split + 1, len(ts) - 1)]))
    return np.array(line)


def teeth(line):
    """The edge line's offsets from its own running mean over BEDS_SPAN_M, along each unbroken run of it."""
    span = int(round(BEDS_SPAN_M / BEDS_STEP_M)) | 1
    runs = [r[np.isfinite(r)] for r in np.split(line, np.nonzero(~np.isfinite(line))[0])]
    return np.concatenate([r[span // 2:span // 2 + len(r) - span + 1] - np.convolve(r, np.ones(span) / span, "valid")
                           for r in runs if len(r) > 2 * span])


def largest_rectangle(has, fill=ROCK_FILL, step=4):
    """Slices of the largest rectangle, its sides multiples of `step` cells, at least `fill` of whose cells are set."""
    cs = np.pad(np.cumsum(np.cumsum(has, 0), 1), ((1, 0), (1, 0)))
    best, area = None, 0
    for h in range(8 * step, has.shape[0] + 1, step):
        for w in range(8 * step, has.shape[1] + 1, step):
            if h * w <= area:
                continue
            r, c = np.mgrid[0:has.shape[0] - h + 1:step, 0:has.shape[1] - w + 1:step]
            covered = cs[r + h, c + w] - cs[r, c + w] - cs[r + h, c] + cs[r, c]
            if (covered >= fill * h * w).any():
                k = np.argmax(covered >= fill * h * w)
                best, area = (slice(r.flat[k], r.flat[k] + h), slice(c.flat[k], c.flat[k] + w)), h * w
    return best


class GroundLookTests(RenderTestCase):
    QUERY = "?t=60&pause=1&scale=0.5&hud=0"

    async def test_the_cliffs_rock_repeats_at_no_period(self):
        # the rock under the castle read as a plaid of light and dark squares (tours 3, 5 and 6): regular strata up
        # the face across streaks on the world's grid
        at, normal, (foot, top) = cliff()
        eye = at + normal * ROCK_BACK_M
        frames = await self.page.evaluate("([p, y, d]) => { __hog.tod(12.5); __hog.dbg(d); __hog.camera(p, y, 0); "
                                          "return __hog.frames; }",
                                          [[float(eye[0]), (foot + top) / 2, float(eye[1])],
                                           math.atan2(-normal[0], -normal[1]),
                                           DBG_NO_SUN_SHADOWS | DBG_NO_TREES | DBG_NO_CLOUD_SHADOWS])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 90}", timeout=FRAME_TIMEOUT_S * 1000 * 4)
        view, probed, colours = await self.page.evaluate("""() => { const v = __hog.probe(0, 0), [w, h] = v.res;
            return [v, __hog.probeRect(0, 0, w, h), Array.from(__hog.probeColorRect(0, 0, w, h))]; }""")
        w, h = view["res"]
        g = np.array(probed, float).reshape(h, w, 6)
        rgb = np.array(colours, float).reshape(h, w, 4)[..., :3]
        yy, xx = np.mgrid[0:h, 0:w]
        rays = np.stack([(2 * (xx + 0.5 + view["jit"][0]) - w) / h * view["tanH"],
                         (2 * (yy + 0.5 + view["jit"][1]) - h) / h * view["tanH"], np.ones((h, w))], -1)
        rays = rays @ np.array(view["rot"]).reshape(3, 3)
        p = np.array(view["pos"]) + rays / np.linalg.norm(rays, axis=-1, keepdims=True) * g[..., 5:6]
        rock = (g[..., 0] == MAT_GROUND) & (g[..., 2] < ROCK_NY)
        self.assertGreater(int(rock.sum()), 20000, "too little rock in view")
        along = p[..., 0] * -normal[1] + p[..., 2] * normal[0]
        s = np.floor((along[rock] - along[rock].min()) / ROCK_CELL_M).astype(int)
        u = np.floor((p[..., 1][rock] - p[..., 1][rock].min()) / ROCK_CELL_M).astype(int)
        lum = np.log(np.maximum(rgb[rock] @ [0.2126, 0.7152, 0.0722], 1e-4))
        total, count = np.zeros((u.max() + 1, s.max() + 1)), np.zeros((u.max() + 1, s.max() + 1))
        np.add.at(total, (u, s), lum)
        np.add.at(count, (u, s), 1)
        rows, cols = largest_rectangle(count > 0)
        a, has = (total / np.maximum(count, 1))[rows, cols], count[rows, cols] > 0
        self.assertGreaterEqual(min(a.shape) * ROCK_CELL_M, 16.0, "too little of the face in view")
        a = np.where(has, a, a[has].mean())
        a = a - ndi.gaussian_filter(a, ROCK_TREND_M / ROCK_CELL_M)
        a = (a - a.mean()) * np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
        power = ndi.uniform_filter(np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2, 3)
        fu, fs = (np.fft.fftshift(np.fft.fftfreq(n, ROCK_CELL_M)) for n in a.shape)
        period = 1.0 / np.maximum(np.hypot(*np.meshgrid(fu, fs, indexing="ij")), 1e-9)
        band = (period >= ROCK_PERIODS_M[0]) & (period <= ROCK_PERIODS_M[1])
        peak = (power / np.maximum(ndi.median_filter(power, 9), 1e-30))[band]
        self.assertLess(float(peak.max()), ROCK_PEAK)

    async def beds_edge_line(self):
        """The beds' edge (beds_edge) as the page draws it: for each BEDS_STEP_M along it, the offset (m) across it of
        the line between the beds' colour and the ground's beside them, or NaN where the frame leaves too little."""
        at, along, out = beds_edge()
        top = await self.ground(float(at[0]), float(at[1]))
        # facing across the edge, so that it runs along the frame's width (±17.8 m), tipped 4° from straight down
        frames = await self.page.evaluate("([p, y, d]) => { __hog.tod(12.5); __hog.dbg(d); __hog.camera(p, y, -1.5); "
                                          "return __hog.frames; }",
                                          [[float(at[0]), top + BEDS_VIEW_M, float(at[1])], math.atan2(out[0], out[1]),
                                           DBG_CLEAN_EDGES | DBG_NO_SUN_SHADOWS | DBG_NO_CLOUD_SHADOWS])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 8}", timeout=FRAME_TIMEOUT_S * 1000)
        view, probed, colours = await self.page.evaluate("""() => { const v = __hog.probe(0, 0), [w, h] = v.res;
            return [v, __hog.probeRect(0, 0, w, h), Array.from(__hog.probeColorRect(0, 0, w, h))]; }""")
        w, h = view["res"]
        g = np.array(probed, float).reshape(h, w, 6)
        rgb = np.array(colours, float).reshape(h, w, 4)[..., :3]
        yy, xx = np.mgrid[0:h, 0:w]
        rays = np.stack([(2 * (xx + 0.5 + view["jit"][0]) - w) / h * view["tanH"],
                         (2 * (yy + 0.5 + view["jit"][1]) - h) / h * view["tanH"], np.ones((h, w))], -1)
        rays = rays @ np.array(view["rot"]).reshape(3, 3)
        p = np.array(view["pos"]) + rays / np.linalg.norm(rays, axis=-1, keepdims=True) * g[..., 5:6]
        ground = g[..., 0] == MAT_GROUND
        s, t = ((p[..., [0, 2]][ground] - at) @ np.column_stack([along, out])).T
        chroma = np.log(np.maximum(rgb[ground][:, [0, 2]], 1e-4)) - np.log(np.maximum(rgb[ground][:, [1]], 1e-4))
        return edge_line(s, t, chroma)

    async def test_the_beds_edge_runs_smooth_not_along_the_texels_steps(self):
        line = await self.beds_edge_line()
        self.assertGreater(int(np.isfinite(line).sum()), 0.9 * len(line), "too little of the beds' edge in view")
        self.assertLess(float(np.sqrt(np.mean(teeth(line) ** 2))), BEDS_TEETH_M)


if __name__ == "__main__":
    unittest.main()
