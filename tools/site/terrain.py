"""Terrain frames and elevation models for the site lane.

The project's TM frame is stated on WGS84 here, so transformations to other datums are real ones; British National Grid
goes through the pinned OSTN15 grid. A north-up TM grid holds any model resampled (bilinear) onto its cell centres.

Models, each read from its pinned source:
  glo30    Copernicus GLO-30, a surface model (canopy and buildings included): today's source
  fabdem   FABDEM V1-2, GLO-30 with forests and buildings removed
  gedtm30  GEDTM30 v1.2, a bare-earth ensemble model
  os50     OS Terrain 50, Ordnance Survey's 50 m bare-earth grid
GLO-30, FABDEM and GEDTM30 put pixel centres on whole arc-seconds (their GDAL origins sit half a pixel outside the
degree corner). OS Terrain 50's heights are posts 50 m apart at the centres of the ASCII grid's cells, 25 m in from
whole multiples of 50 m. Heights are EGM2008 geoid heights, except OS Terrain 50's, which are Ordnance Datum Newlyn.
"""
import io
import json
import zipfile
from functools import lru_cache

import numpy as np
from pyproj import Proj, Transformer
from scipy import ndimage as ndi

from tools.inputs import require
from common import TM

TM_WGS84 = TM + " +towgs84=0,0,0"
MODELS = ("glo30", "fabdem", "gedtm30", "os50")
ARCSEC = 1.0 / 3600.0


@lru_cache(maxsize=1)
def _geographic():
    return (Transformer.from_crs(TM_WGS84, "EPSG:4326", always_xy=True),
            Transformer.from_crs("EPSG:4326", TM_WGS84, always_xy=True))


@lru_cache(maxsize=1)
def _national_grid():
    grid = require("site-ostn15") / "uk_os_OSTN15_NTv2_OSGBtoETRS.tif"
    return Transformer.from_crs(TM_WGS84, "+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 +x_0=400000 +y_0=-100000 "
                                f"+ellps=airy +units=m +nadgrids={grid.as_posix()}", always_xy=True)


def _apply(transformer, a, b):
    """transformer.transform on scalars or arrays; size-1 arrays go through as floats (pyproj's scalar path would
    otherwise convert them itself, which NumPy deprecates)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 1 and b.size == 1 and (a.ndim or b.ndim):
        x, y = transformer.transform(float(a.ravel()[0]), float(b.ravel()[0]))
        shape = np.broadcast(a, b).shape
        return np.full(shape, x), np.full(shape, y)
    return transformer.transform(a, b)


def ll(E, N):
    return _apply(_geographic()[0], E, N)


def tm(lon, lat):
    return _apply(_geographic()[1], lon, lat)


def bng(E, N):
    """British National Grid (OSGB36) eastings and northings of TM points, through OSTN15."""
    return _apply(_national_grid(), E, N)


def tm_proj():
    return Proj(TM_WGS84)


class Grid:
    """A north-up TM raster: cell centres at (E0 + step * j, N0 + step * i), row 0 at the south."""

    def __init__(self, a, E0, N0, step):
        self.a, self.E0, self.N0, self.step = a, float(E0), float(N0), float(step)
        self.ny, self.nx = a.shape

    def coords(self):
        return self.E0 + self.step * np.arange(self.nx), self.N0 + self.step * np.arange(self.ny)

    def sample(self, E, N, order=1):
        """Values at TM points (bilinear by default); NaN off the grid."""
        i = (np.asarray(N, float) - self.N0) / self.step
        j = (np.asarray(E, float) - self.E0) / self.step
        out = ndi.map_coordinates(self.a, [np.ravel(i), np.ravel(j)], order=order, mode="constant", cval=np.nan,
                                  prefilter=False)
        return out.reshape(np.shape(i))


def square(centre, half, step):
    """(E0, N0, n): a square of cell centres on whole multiples of `step` covering centre ± half."""
    E0 = step * np.floor((centre[0] - half) / step)
    N0 = step * np.floor((centre[1] - half) / step)
    return E0, N0, int(np.ceil(2 * half / step)) + 2


class Lattice(tuple):
    """(heights, centre longitude of column 0, centre latitude of row 0, degrees per column, degrees per row): a
    geographic raster with rows running south from its northern edge."""


def _lattice(a, transform):
    """A Lattice from a GDAL affine (area convention: its origin is the outer corner of pixel (0, 0))."""
    return Lattice((a, transform[2] + transform[0] / 2, transform[5] + transform[4] / 2, transform[0], -transform[4]))


def _mosaic(tiles):
    """West-to-east tiles of one latitude band as one Lattice, checking they meet on the same lattice."""
    a = np.concatenate([t[0] for t in tiles], axis=1)
    first = tiles[0]
    for k, t in enumerate(tiles[1:], 1):
        expected = first[1] + first[3] * sum(u[0].shape[1] for u in tiles[:k])
        if abs(t[1] - expected) > first[3] * 1e-3 or abs(t[2] - first[2]) > first[4] * 1e-3:
            raise ValueError("tiles do not share a lattice")
    return Lattice((a, first[1], first[2], first[3], first[4]))


def _read_tif(path):
    import rasterio

    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        return _lattice(a, list(ds.transform)[:6])


@lru_cache(maxsize=1)
def _glo30():
    root = require("site-glo30")
    return _mosaic([_read_tif(root / f"Copernicus_DSM_COG_10_N57_00_W00{w}_00_DEM.tif") for w in (6, 5)])


@lru_cache(maxsize=1)
def _fabdem():
    root = require("site-fabdem")
    return _mosaic([_read_tif(root / f"N57W00{w}_FABDEM_V1-2.tif") for w in (6, 5)])


@lru_cache(maxsize=1)
def _gedtm30():
    root = require("site-gedtm30")
    grid = json.loads((root / "gedtm30_n57_w006_w004.json").read_text(encoding="utf-8"))
    return _lattice(np.load(root / "gedtm30_n57_w006_w004.npy"), grid["transform"])


OS50_SQUARES = {"ng": (100000, 800000), "nh": (200000, 800000)}


@lru_cache(maxsize=1)
def _os50():
    """(heights, E of column 0, N of row 0): the NG and NH squares' posts, 50 m apart, rows from the south. Each post
    stands at its ESRI cell centre, 25 m east and north of the header's xllcorner and yllcorner: read there, OS Terrain
    50's landforms line up with GLO-30's (the registration study; on the header corners they lie 25 m south-west)."""
    archive = zipfile.ZipFile(require("site-os50") / "terr50_gagg_gb.zip")
    a = np.full((2000, 4000), np.nan, np.float32)
    for name in archive.namelist():
        parts = name.lower().split("/")
        if len(parts) != 3 or parts[1] not in OS50_SQUARES or not parts[2].endswith(".zip"):
            continue
        inner = zipfile.ZipFile(io.BytesIO(archive.read(name)))
        asc = next(n for n in inner.namelist() if n.lower().endswith(".asc"))
        lines = inner.read(asc).decode("ascii").splitlines()
        head = {k.lower(): float(v) for k, v in (line.split() for line in lines[:5])}
        posts = np.array([line.split() for line in lines[5:]], np.float32)[::-1]
        i0 = int(round((head["yllcorner"] - 800000) / 50))
        j0 = int(round((head["xllcorner"] - 100000) / 50))
        a[i0:i0 + posts.shape[0], j0:j0 + posts.shape[1]] = posts
    return a, 100025.0, 800025.0


def _sample_lattice(lat_, lon, lat):
    """Bilinear heights at longitudes and latitudes. Each pixel covers half a pixel round its centre, so points out to
    the outer edges of the edge pixels take their values (a mosaic's last row may land a rounding error past them);
    NaN beyond."""
    a, lon0, lat0, dlon, dlat = lat_
    rows = (lat0 - np.asarray(lat)) / dlat
    cols = (np.asarray(lon) - lon0) / dlon
    ny, nx = a.shape
    inside = (rows >= -0.5) & (rows <= ny - 0.5) & (cols >= -0.5) & (cols <= nx - 0.5)
    out = ndi.map_coordinates(a, [np.clip(rows, 0, ny - 1), np.clip(cols, 0, nx - 1)], order=1, mode="nearest",
                              prefilter=False)
    return np.where(inside, out, np.nan)


def heights(model, E, N):
    """The model's heights (m) at TM points; NaN outside its coverage."""
    E, N = np.asarray(E, float), np.asarray(N, float)
    shape = np.broadcast(E, N).shape
    E, N = np.broadcast_to(E, shape).ravel(), np.broadcast_to(N, shape).ravel()
    if model == "os50":
        a, x0, y0 = _os50()
        x, y = bng(E, N)
        out = ndi.map_coordinates(a, [(np.asarray(y) - y0) / 50.0, (np.asarray(x) - x0) / 50.0], order=1,
                                  mode="constant", cval=np.nan, prefilter=False)
    else:
        lon, lat = ll(E, N)
        out = _sample_lattice({"glo30": _glo30, "fabdem": _fabdem, "gedtm30": _gedtm30}[model](), lon, lat)
    return out.reshape(shape)


def heights_ll(model, lon, lat):
    """The model's heights (m) at longitudes and latitudes (WGS84), broadcast together; NaN outside its coverage."""
    lon, lat = np.broadcast_arrays(np.asarray(lon, float), np.asarray(lat, float))
    if model == "os50":
        return heights(model, *tm(lon, lat))
    lattice = {"glo30": _glo30, "fabdem": _fabdem, "gedtm30": _gedtm30}[model]()
    return _sample_lattice(lattice, lon.ravel(), lat.ravel()).reshape(lon.shape)


def model_grid(model, E0, N0, step, nx, ny, rows=256):
    """The model on a TM grid of cell centres."""
    E = E0 + step * np.arange(nx)
    out = np.empty((ny, nx), np.float32)
    for r in range(0, ny, rows):
        N = N0 + step * np.arange(r, min(r + rows, ny))
        out[r:r + len(N)] = heights(model, E[None, :], N[:, None])
    return Grid(out, E0, N0, step)


def osm_fraction(grid, source, file, keep, sub=3):
    """The fraction of each grid cell inside the OpenStreetMap polygons and multipolygons of a pinned extract whose
    tags `keep` accepts, supersampled sub x sub."""
    from PIL import Image, ImageDraw
    from build_regional import assemble_rings

    data = json.loads((require(source) / file).read_text(encoding="utf-8"))
    n_x, n_y = grid.nx * sub, grid.ny * sub
    image = Image.new("L", (n_x, n_y), 0)
    draw = ImageDraw.Draw(image)
    s = grid.step / sub

    def pixels(ring):
        e, n = tm([p[0] for p in ring], [p[1] for p in ring])
        x = (np.asarray(e) - (grid.E0 - grid.step / 2)) / s
        y = n_y - (np.asarray(n) - (grid.N0 - grid.step / 2)) / s       # PIL rows run from the north
        return list(zip(x, y))

    outers, inners = [], []
    for el in data["elements"]:
        if not keep(el.get("tags", {})):
            continue
        if el["type"] == "way":
            ring = [(p["lon"], p["lat"]) for p in el.get("geometry") or []]
            if len(ring) >= 4 and ring[0] == ring[-1]:
                outers.append(ring)
        elif el["type"] == "relation":
            parts = {"outer": [], "inner": []}
            for m in el.get("members", []):
                if m.get("type") == "way" and m.get("geometry"):
                    parts["inner" if m.get("role") == "inner" else "outer"].append(
                        [(p["lon"], p["lat"]) for p in m["geometry"]])
            outers += assemble_rings(parts["outer"])
            inners += assemble_rings(parts["inner"])
    for ring in outers:
        draw.polygon(pixels(ring), fill=1)
    for ring in inners:
        draw.polygon(pixels(ring), fill=0)
    fine = np.asarray(image, np.float32)[::-1]
    return Grid(fine.reshape(grid.ny, sub, grid.nx, sub).mean(axis=(1, 3)), grid.E0, grid.N0, grid.step)


def water_fraction(grid, source="site-osm-water-x"):
    """The fraction of each grid cell under OpenStreetMap water bodies and riverbanks."""
    return osm_fraction(grid, source, "osm_water_wide.json",
                        lambda t: t.get("natural") == "water" or t.get("waterway") == "riverbank")


def woodland_fraction(grid, source="site-osm-x"):
    """The fraction of each grid cell under OpenStreetMap forest and woodland."""
    return osm_fraction(grid, source, "osm.json", lambda t: t.get("landuse") == "forest" or t.get("natural") == "wood")
