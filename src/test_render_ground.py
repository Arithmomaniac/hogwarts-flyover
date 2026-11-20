"""The ground's look at the site as the page renders it: the cliffs' rock shows no regular pattern, and the steps its
texels make of a cliff cast no shadow on it."""
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
# ROCK_PERIODS_M stands more than ROCK_PEAK times the median of the nine by nine round it, in any frame of the camera's
# jitter cycle, at ROCK_HOUR, when the sun rakes the face and its bumps' shade shows any lattice they follow; nor more
# than ROCK_SKYLIT_PEAK times at ROCK_SKYLIT_HOUR, when the sun is behind the face and the sky and the ground's bounce
# light it (with the sun's shadows off, only the few bumps turned to it catch the sun), so that its colour shows. The
# rock's strata, 3.3 m apart, measured 9.9 times at the old site; at X they read 3.87 in the bumps' shade at 12:30,
# against 3.72 without them, and 53 to 60 at 17:00, where the rock's own beds and joints read 5.1. Rock textured on
# the face's own planes, 3.7; its bumps on one lattice of value noise, 6.45 at 2 m along the face (09ac6e8's worst
# frame), on two, 3.73
CLIFF_RISE, CLIFF_NEAR_M = 2.5, 450.0
ROCK_BACK_M = 45.0
ROCK_NY = 0.35
ROCK_CELL_M, ROCK_TREND_M = 0.25, 3.0
ROCK_FILL = 0.97
ROCK_PERIODS_M = (1.5, 8.0)
ROCK_HOUR, ROCK_PEAK = 12.5, 6.0
ROCK_SKYLIT_HOUR, ROCK_SKYLIT_PEAK = 17.0, 10.0
# the cliff in its own shade: seen as for its rock but with the sun's shadows on, at 12:30, when the sun rakes the face
# from the south-south-east, no more than ROCK_SHADED of its rock lies in shadow (a sun term under a half) in any frame
# of the jitter cycle. Marched from the face itself, the sun's rays met the steps its texels make of it, which shaded one
# another in wedges with hatched edges: 17% of the rock (09ac6e8, both GPUs); marched from 1.5 m out, 0.1%
ROCK_SHADED = 0.05
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


def rock_spectrum(total, count):
    """For rock laid out in ROCK_CELL_M cells (each cell's summed log luminance and its pixel count): the shape of the
    largest rectangle its cells cover nearly whole, and over that, the period (m) of each frequency, its power over the
    median of the nine by nine round it, and which periods the test judges (ROCK_PERIODS_M)."""
    rows, cols = largest_rectangle(count > 0)
    a, has = (total / np.maximum(count, 1))[rows, cols], count[rows, cols] > 0
    a = np.where(has, a, a[has].mean())
    a = a - ndi.gaussian_filter(a, ROCK_TREND_M / ROCK_CELL_M)
    a = (a - a.mean()) * np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    power = ndi.uniform_filter(np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2, 3)
    fu, fs = (np.fft.fftshift(np.fft.fftfreq(n, ROCK_CELL_M)) for n in a.shape)
    period = 1.0 / np.maximum(np.hypot(*np.meshgrid(fu, fs, indexing="ij")), 1e-9)
    band = (period >= ROCK_PERIODS_M[0]) & (period <= ROCK_PERIODS_M[1])
    return a.shape, power / np.maximum(ndi.median_filter(power, 9), 1e-30), period, band


# In-page JS for one frame: the rock in view (`ground` whose normal's y is under `ny`) laid out in cells `cell` m wide
# along the face (outward normal (nx, nz)) and up it, from its least: [rows, columns, each cell's summed log luminance
# and its pixel count (row by row), the rock's pixels, those of them with a sun term under a half]
CLIFF_CELLS = """([nx, nz, ny, cell, ground]) => {
  const v = __hog.probe(0, 0), [w, h] = v.res, R = v.rot, g = __hog.probeRect(0, 0, w, h), c = __hog.probeColorRect(0, 0, w, h);
  const along = [], up = [], lum = [];
  let shaded = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, q = g[i];
    if (q[0] !== ground || !(q[2] < ny)) continue;
    const rx = (2 * (x + 0.5 + v.jit[0]) - w) / h * v.tanH, ry = (2 * (y + 0.5 + v.jit[1]) - h) / h * v.tanH;
    const dx = rx * R[0] + ry * R[3] + R[6], dy = rx * R[1] + ry * R[4] + R[7], dz = rx * R[2] + ry * R[5] + R[8];
    const s = q[5] / Math.hypot(dx, dy, dz);
    along.push((v.pos[0] + dx * s) * -nz + (v.pos[2] + dz * s) * nx);
    up.push(v.pos[1] + dy * s);
    lum.push(Math.log(Math.max(0.2126 * c[4 * i] + 0.7152 * c[4 * i + 1] + 0.0722 * c[4 * i + 2], 1e-4)));
    shaded += __hog.probeSun(x, y) < 0.5 ? 1 : 0;
  }
  let a0 = Infinity, u0 = Infinity, nr = 0, nc = 0;
  for (let k = 0; k < along.length; k++) { a0 = Math.min(a0, along[k]); u0 = Math.min(u0, up[k]); }
  const col = along.map((a) => Math.floor((a - a0) / cell)), row = up.map((u) => Math.floor((u - u0) / cell));
  for (let k = 0; k < along.length; k++) { nc = Math.max(nc, col[k] + 1); nr = Math.max(nr, row[k] + 1); }
  const total = new Float64Array(nr * nc), count = new Float64Array(nr * nc);
  for (let k = 0; k < along.length; k++) { total[row[k] * nc + col[k]] += lum[k]; count[row[k] * nc + col[k]] += 1; }
  return [nr, nc, Array.from(total), Array.from(count), along.length, shaded]; }"""


class GroundLookTests(RenderTestCase):
    QUERY = "?t=60&pause=1&scale=0.5&hud=0"

    async def cliff_frames(self, dbg, hour=ROCK_HOUR):
        """The cliff (cliff()) seen square on from ROCK_BACK_M at its mid-height at `hour` with debug bits `dbg`, over a
        jitter cycle: for each frame, its rock laid out in ROCK_CELL_M cells along and up the face (each cell's summed
        log luminance, and its pixel count), and the share of the rock in shadow (a sun term under a half)."""
        at, normal, (foot, top) = cliff()
        eye = at + normal * ROCK_BACK_M
        frames = await self.page.evaluate("([p, y, d, h]) => { __hog.tod(h); __hog.dbg(d); __hog.camera(p, y, 0); "
                                          "return __hog.frames; }",
                                          [[float(eye[0]), (foot + top) / 2, float(eye[1])],
                                           math.atan2(-normal[0], -normal[1]), dbg, hour])
        await self.page.wait_for_function(f"__hog.frames >= {frames + 90}", timeout=FRAME_TIMEOUT_S * 1000 * 4)
        found = []
        for rows, cols, total, count, rock, shaded in await self.each_frame(
                CLIFF_CELLS, [float(normal[0]), float(normal[1]), ROCK_NY, ROCK_CELL_M, MAT_GROUND]):
            self.assertGreater(rock, 20000, "too little rock in view")
            found.append((np.array(total, float).reshape(rows, cols), np.array(count, float).reshape(rows, cols),
                          shaded / rock))
        return found

    async def test_the_cliffs_rock_repeats_at_no_period(self):
        # the rock under the castle read as a plaid of light and dark squares (tours 3, 5 and 6): regular strata up
        # the face across streaks on the world's grid. Judged in every frame of the jitter cycle: one frame's peak
        # stood under the limit on one GPU and over it on the other
        peaks = {}
        for hour, limit in ((ROCK_HOUR, ROCK_PEAK), (ROCK_SKYLIT_HOUR, ROCK_SKYLIT_PEAK)):
            peaks[hour] = []
            for total, count, _ in await self.cliff_frames(DBG_NO_SUN_SHADOWS | DBG_NO_TREES | DBG_NO_CLOUD_SHADOWS,
                                                           hour):
                shape, ratio, _, band = rock_spectrum(total, count)
                self.assertGreaterEqual(min(shape) * ROCK_CELL_M, 16.0, "too little of the face in view")
                peaks[hour].append(round(float(ratio[band].max()), 2))
            self.assertLess(max(peaks[hour]), limit, peaks)

    async def test_the_cliffs_texel_steps_cast_no_shadow_on_its_face(self):
        # at 12:30 the cliff's rock showed dark wedges, hatched at their edges, where its texels' steps shaded one another
        shaded = [round(s, 3) for _, _, s in await self.cliff_frames(DBG_NO_TREES | DBG_NO_CLOUD_SHADOWS)]
        self.assertLess(max(shaded), ROCK_SHADED, shaded)

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
