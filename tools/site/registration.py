"""Registration of the candidate elevation models against GLO-30 and against mapped summits.

1. Offsets: cross-correlation of smoothed slope images on a 5 m grid, refined iteratively, tile by tile around the
   site, gives how far each model's landforms lie from GLO-30's, to about a metre. Tiles with little relief, or much
   canopy (GLO-30 standing over FABDEM), are skipped; the median over tiles is the model's offset.
2. Summits: every peak or hill mapped in OpenStreetMap with a height (mostly Ordnance Survey spot heights) is matched to
   each model's highest point within 100 m: the median offset of those maxima from the mapped positions, and of their
   heights from the mapped heights (ODN).
3. Vertical: the median difference from GLO-30 on open, gently sloping ground, where canopy cannot explain it.

    python -X utf8 -m tools.site.registration [SITE]     # default x; prints the results as JSON
"""
import json
import sys

import numpy as np
from scipy import ndimage as ndi

from tools.inputs import require
from tools.site import terrain as T
from common import SITES

TILE_M, STEP = 2000.0, 5.0


def _peak(r, i, j):
    """Sub-cell position of a correlation peak at (i, j): a parabola through it and its neighbours on each axis."""
    ny, nx = r.shape

    def vertex(m, c, p):
        d = m - 2 * c + p
        return 0.0 if d == 0 else 0.5 * (m - p) / d

    return (i + vertex(r[(i - 1) % ny, j], r[i, j], r[(i + 1) % ny, j]),
            j + vertex(r[i, (j - 1) % nx], r[i, j], r[i, (j + 1) % nx]))


def _correlation_peak(reference, moving, step, sigma):
    """(rows, columns) of the cross-correlation peak of the two grids' smoothed gradients (both components, Hann
    tapered). Gradients are linear in the heights, so a symmetric blur of one grid cannot move the peak; a slope
    magnitude would, wherever a ridge is steeper on one side."""
    window = np.outer(np.hanning(reference.shape[0]), np.hanning(reference.shape[1]))

    def spectra(a):
        gy, gx = np.gradient(ndi.gaussian_filter(a, sigma), step)
        return [np.fft.fft2((g - g.mean()) * window) for g in (gx, gy)]

    r = np.fft.ifft2(sum(np.conj(a) * b for a, b in zip(spectra(reference), spectra(moving)))).real
    i, j = _peak(r, *np.unravel_index(int(np.argmax(r)), r.shape))
    ny, nx = r.shape
    return (i - ny if i > ny / 2 else i), (j - nx if j > nx / 2 else j)


def offset(reference, moving, step, sigma_m=20.0, iterations=4):
    """(dE, dN) in metres by which `moving`'s features lie from `reference`'s: two grids on the same cells, rows from
    the south. Gradients smoothed over sigma_m are cross-correlated, then `moving` is shifted back by the estimate and
    the remainder measured again, so the window's pull towards zero shift fades. Plain rather than phase-only
    correlation, so that two models' different fine texture (posts 50 m apart against 30 m pixels) cannot set the peak."""
    reference = np.nan_to_num(reference, nan=float(np.nanmean(reference)))
    moving = np.nan_to_num(moving, nan=float(np.nanmean(moving)))
    total, shifted = np.zeros(2), moving
    for _ in range(iterations):
        total += _correlation_peak(reference, shifted, step, sigma_m / step)
        shifted = ndi.shift(moving, -total, order=1, mode="nearest")
    return float(total[1] * step), float(total[0] * step)


def tile_offsets(anchor, model, reach=16000.0, spacing=4000.0):
    """The model's offset from GLO-30 on each usable tile within `reach` of the anchor: [(E, N, dE, dN), ...]."""
    out = []
    n = int(TILE_M / STEP)
    for cE in np.arange(-reach + spacing / 2, reach, spacing):
        for cN in np.arange(-reach + spacing / 2, reach, spacing):
            E, N = np.meshgrid(anchor[0] + cE + STEP * (np.arange(n) - n / 2), anchor[1] + cN + STEP * (np.arange(n) - n / 2))
            glo, other = T.heights("glo30", E, N), T.heights(model, E, N)
            canopy = glo - T.heights("fabdem", E, N)
            if np.isnan(other).mean() > 0.01 or np.nanstd(glo) < 30 or np.nanpercentile(canopy, 90) > 3:
                continue
            out.append((float(anchor[0] + cE), float(anchor[1] + cN), *offset(glo, other, STEP)))
    return out


def summits(anchor):
    """OpenStreetMap peaks and hills with heights near the site: [(name, E, N, ele)]."""
    data = json.loads((require("site-osm-x") / "osm.json").read_text(encoding="utf-8"))
    out = []
    for el in data["elements"]:
        t = el.get("tags", {})
        if el["type"] == "node" and t.get("natural") in ("peak", "hill") and t.get("ele"):
            try:
                ele = float(t["ele"].split()[0].rstrip("m"))
            except ValueError:
                continue
            E, N = T.tm(el["lon"], el["lat"])
            out.append((t.get("name", ""), float(E), float(N), ele))
    return out


def summit_errors(model, peaks, radius=100.0, step=5.0):
    """For each summit, (dE, dN, dz): where the model's highest point within `radius` lies from the mapped summit, and
    how much higher it is than the mapped height."""
    out = []
    t = np.arange(-radius, radius + step / 2, step)
    dx, dy = np.meshgrid(t, t)
    disc = np.hypot(dx, dy) <= radius
    for _, E, N, ele in peaks:
        z = T.heights(model, E + dx, N + dy)
        if np.isnan(z[disc]).any():
            continue
        k = np.nanargmax(np.where(disc, z, -np.inf))
        out.append((float(dx.ravel()[k]), float(dy.ravel()[k]), float(z.ravel()[k] - ele)))
    return out


def vertical(anchor, model, reach=20000.0, n=200000, seed=11):
    """Median of the model minus GLO-30 on open, gently sloping ground (slope under 5 degrees, no canopy)."""
    rng = np.random.default_rng(seed)
    E, N = anchor[0] + rng.uniform(-reach, reach, n), anchor[1] + rng.uniform(-reach, reach, n)
    glo, other = T.heights("glo30", E, N), T.heights(model, E, N)
    gx = (T.heights("glo30", E + STEP, N) - T.heights("glo30", E - STEP, N)) / (2 * STEP)
    gy = (T.heights("glo30", E, N + STEP) - T.heights("glo30", E, N - STEP)) / (2 * STEP)
    flat = np.degrees(np.arctan(np.hypot(gx, gy))) < 5
    bare = np.abs(glo - T.heights("fabdem", E, N)) < 0.5
    ok = flat & bare & np.isfinite(other) & (glo > 5)
    return {"median": float(np.median(other[ok] - glo[ok])), "p10": float(np.percentile(other[ok] - glo[ok], 10)),
            "p90": float(np.percentile(other[ok] - glo[ok], 90)), "n": int(ok.sum())}


def report(site="x"):
    anchor = SITES[site]["anchor"]
    peaks = summits(anchor)
    out = {"site": site, "summits": len(peaks), "models": {}}
    for model in T.MODELS:
        m = {}
        if model != "glo30":
            tiles = np.array(tile_offsets(anchor, model))
            m["offset_m"] = {"dE": float(np.median(tiles[:, 2])), "dN": float(np.median(tiles[:, 3])),
                             "spread_m": float(np.median(np.hypot(tiles[:, 2] - np.median(tiles[:, 2]),
                                                                  tiles[:, 3] - np.median(tiles[:, 3])))),
                             "tiles": int(len(tiles))}
            m["vertical_m"] = vertical(anchor, model)
        errs = np.array(summit_errors(model, peaks))
        m["summits"] = {"n": int(len(errs)), "dE": float(np.median(errs[:, 0])), "dN": float(np.median(errs[:, 1])),
                        "dz": float(np.median(errs[:, 2])), "dz_abs": float(np.median(np.abs(errs[:, 2])))}
        out["models"][model] = m
    return out


if __name__ == "__main__":
    print(json.dumps(report(*sys.argv[1:2]), indent=1))
