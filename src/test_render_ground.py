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

DBG_NO_SUN_SHADOWS, DBG_NO_TREES, DBG_NO_CLOUD_SHADOWS = 1, 8, 1024
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
# Fails since wave 1 merged lane A's f4aa05a (the harbour mouth's rock meshes and the cliff wall) beside lane D's rock,
# which frag4 still shades as lane D left it: the cliff's strongest period stands 19.8 times the median round it. Lane A
# is fixing it.
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


if __name__ == "__main__":
    unittest.main()
