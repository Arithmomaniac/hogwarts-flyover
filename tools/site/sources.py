"""Pinned geographic sources for moving the scene: elevation models, the OSTN15 grid, OpenStreetMap extracts and
Sentinel-2 cloudless tiles.

Each source is a `site-*` tree input in tools/inputs.lock.json whose files are pinned by size and SHA256 in
tools/site/locks/<name>.json. `fetch` fills cache_dir("sources", name) from the pinned original intermediates where
they hold the same bytes, otherwise from the publisher, checks every file against the pins and records the folder for
this machine, so a stage gets it from tools.inputs.require(name). OpenStreetMap extracts are queried from Overpass at a
pinned date and stored canonically (elements sorted, formatting fixed), so a later fetch reproduces the same bytes. A
copy recorded with `python -X utf8 -m tools.inputs set NAME FOLDER` that verifies is used without fetching.

    python -X utf8 -m tools.site.sources fetch [NAME ...]   # fetch what is missing, verify it and record its place
    python -X utf8 -m tools.site.sources list               # every source with its licence and state
    python -X utf8 -m tools.site.sources pin [NAME ...]     # (maintenance) fetch afresh, pin it and record the recipe;
                                                            # without names, not the sources replaced at each release
"""
import dataclasses
import hashlib
import io
import json
import math
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import inputs
from tools.paths import cache_dir, sha256
from common import SITES, to_ll

AGENT = {"User-Agent": "hogwarts-flyover data build (pinned downloads, low rate)"}
OVERPASS = "https://overpass-api.de/api/interpreter"
OSM_DATE = "2026-10-31T00:00:00Z"
X = SITES["x"]["anchor"]


@dataclass(frozen=True)
class Http:
    """Whole files, ((relative path, URL), ...); `seed` is an (input, folder) that may already hold the same bytes."""
    files: tuple
    seed: tuple = ()


@dataclass(frozen=True)
class ZipMembers:
    """Members of a remote zip archive, read through HTTP ranges instead of downloading the whole archive."""
    url: str
    members: tuple


@dataclass(frozen=True)
class Window:
    """A window of a remote cloud-optimised GeoTIFF: heights in metres (<stem>.npy, NaN where empty) and their grid
    (<stem>.json); `scale` turns stored values into metres."""
    url: str
    bounds: tuple           # west, south, east, north (degrees)
    stem: str
    scale: float = 1.0


@dataclass(frozen=True)
class Overpass:
    """An OpenStreetMap extract as it stood at `date`, stored canonically in `file`."""
    body: str
    date: str
    file: str


@dataclass(frozen=True)
class Tiles:
    """Web-map tiles over ((z, x0, x1, y0, y1), ...) inclusive, stored flat as tiles/{z}_{x}_{y}.jpg."""
    url: str                # with {z}, {x} and {y}
    ranges: tuple
    seed: tuple = ()


@dataclass(frozen=True)
class Source:
    recipe: object
    licence: str
    attribution: str
    current_release_only: bool = False      # the publisher replaces the file with each release


def bounds(anchor, half_m, n=41):
    """(west, south, east, north) in degrees of the TM square anchor ± half_m, its edges sampled densely."""
    t = np.linspace(-half_m, half_m, n)
    e = np.concatenate([t, t, np.full(n, -half_m), np.full(n, half_m)]) + anchor[0]
    nn = np.concatenate([np.full(n, -half_m), np.full(n, half_m), t, t]) + anchor[1]
    lon, lat = to_ll.transform(e, nn)
    return float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))


def tile_xy(lon, lat, z):
    """Fractional web-map tile coordinates of a point at zoom z."""
    n = 2 ** z
    return (lon + 180.0) / 360.0 * n, (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n


def tile_range(b, z):
    """(z, x0, x1, y0, y1): the zoom-z tiles, inclusive, that cover bounds b."""
    x0, y0 = tile_xy(b[0], b[3], z)
    x1, y1 = tile_xy(b[2], b[1], z)
    return z, math.floor(x0), math.floor(x1), math.floor(y0), math.floor(y1)


def bbox(b, digits=4):
    """Overpass's (south,west,north,east) for bounds b, rounded outwards."""
    s = 10 ** digits
    return (f"{math.floor(b[1] * s) / s:.{digits}f},{math.floor(b[0] * s) / s:.{digits}f},"
            f"{math.ceil(b[3] * s) / s:.{digits}f},{math.ceil(b[2] * s) / s:.{digits}f}")


def local_query(anchor):
    """The original build's local extract (peaks, places, water, watercourses, roads, railways, buildings, woods),
    centred on the anchor; hills join the peaks because some named summits are tagged as hills."""
    b = bbox(bounds(anchor, 13000))
    return "".join(f"{q}({b});" for q in (
        'node["natural"~"^(peak|hill)$"]', 'node["place"]', 'way["natural"="water"]', 'relation["natural"="water"]',
        'way["waterway"~"river|stream"]', 'way["highway"]', 'way["railway"]', 'way["building"]',
        'way["landuse"="forest"]', 'way["natural"="wood"]'))


def water_query(anchor):
    """The original build's wide water extract: water bodies, riverbanks and coastline over ±40 km for the ±36 km
    regional layer, and watercourses over ±13 km."""
    wide, near = bbox(bounds(anchor, 40000)), bbox(bounds(anchor, 13000))
    return (f'way["natural"="water"]({wide});relation["natural"="water"]({wide});way["waterway"="riverbank"]({wide});'
            f'way["natural"="coastline"]({wide});way["waterway"~"^(river|stream)$"]({near});')


def _glo30(tile):
    name = f"Copernicus_DSM_COG_10_{tile}_DEM"
    return f"{name}.tif", f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


OSM_LICENCE, OSM_CREDIT = "ODbL 1.0", "© OpenStreetMap contributors"
SOURCES = {
    "site-glo30": Source(
        Http(tuple(_glo30(t) for t in ("N57_00_W006_00", "N57_00_W005_00")), seed=("original-intermediates", "dem")),
        "Copernicus DEM licence: free use with attribution",
        "Copernicus GLO-30 DEM © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under "
        "COPERNICUS by the European Union and ESA"),
    "site-os50": Source(
        # OS's 2026-07 release; the API serves only the current release, so after OS's next one a kept copy is needed
        Http((("terr50_gagg_gb.zip", "https://api.os.uk/downloads/v1/products/Terrain50/downloads?area=GB&"
               "format=ASCII+Grid+and+GML+%28Grid%29&redirect"),)),
        "Open Government Licence v3", "Contains OS data © Crown copyright and database right 2026",
        current_release_only=True),
    "site-ostn15": Source(
        Http((("uk_os_OSTN15_NTv2_OSGBtoETRS.tif", "https://cdn.proj.org/uk_os_OSTN15_NTv2_OSGBtoETRS.tif"),)),
        "Open Government Licence v3 (OSTN15, via PROJ-data)", "OSTN15 © Crown copyright and database right"),
    "site-fabdem": Source(
        ZipMembers("https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/N50W010-N60E000_FABDEM_V1-2.zip",
                   ("N57W006_FABDEM_V1-2.tif", "N57W005_FABDEM_V1-2.tif")),
        "CC BY-NC-SA 4.0", "FABDEM V1-2, Hawker et al. 2022, University of Bristol"),
    "site-gedtm30": Source(
        # the file's 0.1 scale tag does not describe its values, which are metres (median 0.999 of GLO-30's here)
        Window("https://s3.opengeohub.org/global/dtm/v1.2/gedtm_rf_m_30m_s_20060101_20151231_go_epsg.4326.3855_v1.2.tif",
               (-6.0, 57.0, -4.0, 58.0), "gedtm30_n57_w006_w004", scale=1.0),
        "CC BY 4.0 (v1.2.0 says: use for testing purposes only)", "GEDTM30 v1.2, Ho et al. 2025, OpenGeoHub"),
    "site-osm-x": Source(Overpass(local_query(X), OSM_DATE, "osm.json"), OSM_LICENCE, OSM_CREDIT),
    "site-osm-water-x": Source(Overpass(water_query(X), OSM_DATE, "osm_water_wide.json"), OSM_LICENCE, OSM_CREDIT),
    "site-s2-x": Source(
        Tiles("https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg",
              (tile_range(bounds(X, 12100), 14), tile_range(bounds(X, 36200), 11)), seed=("original-intermediates", "s2tiles")),
        "CC BY-NC-SA 4.0",
        "Sentinel-2 cloudless 2023 by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2023)"),
}


def canonical_osm(raw):
    """Overpass JSON as stable bytes: only the elements, sorted by type and id, with fixed formatting, so the header's
    database timestamp and the server's key order never change the pin. Overpass reports a failed or timed-out query
    as a remark beside whatever elements it had, which is refused rather than kept."""
    payload = json.loads(raw)
    if "remark" in payload:
        raise ValueError(f"Overpass reported a problem: {payload['remark']}")
    elements = sorted(payload["elements"], key=lambda e: (e["type"], e["id"]))
    return (json.dumps({"elements": elements}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _download(url, data=None, headers=None, timeout=300):
    request = urllib.request.Request(url, data=data, headers={**AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class _Ranged(io.RawIOBase):
    """A remote file read through HTTP range requests, seekable, for zipfile."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        head = urllib.request.Request(url, headers=AGENT, method="HEAD")
        with urllib.request.urlopen(head, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = _download(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        self.pos += len(data)
        return data


def _pins(name):
    try:
        return {rel: tuple(v) for rel, v in inputs.tree_manifest(name).items()}
    except FileNotFoundError:
        return {}


def _holds(path, pin):
    """True when the file at path matches its pin (any existing file when there is no pin, which only happens while
    pinning into an empty folder)."""
    if not path.is_file():
        return False
    return pin is None or (path.stat().st_size == pin[0] and sha256(path) == pin[1])


def _seed(seed, rel, target, pin):
    """Copy rel from the seed input's folder when it holds the pinned bytes; whether it did. Never without a pin."""
    if not seed or pin is None:
        return False
    root = inputs.locate(seed[0])
    candidate = root / seed[1] / Path(rel).name if root is not None else None
    if candidate is None or not _holds(candidate, pin):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate, target)
    return True


def _fill(target, pin, produce):
    """Write produce()'s bytes to target unless it already holds the pinned bytes."""
    if _holds(target, pin):
        return "cached"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(produce())
    return "fetched"


def _http(recipe, pins, out, seeding):
    import pooch

    for rel, url in recipe.files:
        target, pin = out / rel, pins.get(rel)
        if _holds(target, pin) or (seeding and _seed(recipe.seed, rel, target, pin)):
            continue
        pooch.retrieve(url, known_hash=f"sha256:{pin[1]}" if pin else None, fname=rel, path=out, progressbar=False,
                       downloader=pooch.HTTPDownloader(headers=AGENT, progressbar=False))


def _zip_members(recipe, pins, out, seeding):
    archive = None
    for member in recipe.members:
        if _holds(out / member, pins.get(member)):
            continue
        archive = archive or zipfile.ZipFile(_Ranged(recipe.url))
        (out / member).write_bytes(archive.read(member))        # zipfile checks each member's CRC


def _window(recipe, pins, out, seeding):
    import rasterio
    from rasterio.windows import Window as RasterWindow

    npy, meta = out / f"{recipe.stem}.npy", out / f"{recipe.stem}.json"
    if _holds(npy, pins.get(npy.name)) and _holds(meta, pins.get(meta.name)):
        return
    with rasterio.open("/vsicurl/" + recipe.url) as ds:
        west, south, east, north = recipe.bounds
        c0, r0 = ~ds.transform * (west, north)
        c1, r1 = ~ds.transform * (east, south)
        window = RasterWindow(round(c0), round(r0), round(c1 - c0), round(r1 - r0))
        raw = ds.read(1, window=window, masked=True)
        heights = (raw.astype(np.float64) * recipe.scale).filled(np.nan).astype(np.float32)
        transform = ds.window_transform(window)
        grid = {"source": recipe.url, "crs": ds.crs.to_string(), "transform": list(transform)[:6],
                "shape": list(heights.shape), "pixel": ds.tags().get("AREA_OR_POINT", "Area"),
                "vertical": ds.tags().get("VERTICAL_DATUM", ""), "units": "m", "scale": recipe.scale,
                "scale_tag": ds.scales[0]}
    with open(npy, "wb") as f:
        np.save(f, heights)
    meta.write_text(json.dumps(grid, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _overpass(recipe, pins, out, seeding):
    query = f'[out:json][timeout:900][date:"{recipe.date}"];({recipe.body});out geom;'
    _fill(out / recipe.file, pins.get(recipe.file),
          lambda: canonical_osm(_download(OVERPASS, data=urllib.parse.urlencode({"data": query}).encode(), timeout=1000)))


def tile_names(recipe):
    return [f"tiles/{z}_{x}_{y}.jpg" for z, x0, x1, y0, y1 in recipe.ranges
            for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def _tiles(recipe, pins, out, seeding):
    def one(rel):
        target, pin = out / rel, pins.get(rel)
        if _holds(target, pin) or (seeding and _seed(recipe.seed, rel, target, pin)):
            return
        z, x, y = (int(v) for v in Path(rel).stem.split("_"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_download(recipe.url.format(z=z, x=x, y=y), timeout=60))

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, tile_names(recipe)))


FETCH = {Http: _http, ZipMembers: _zip_members, Window: _window, Overpass: _overpass, Tiles: _tiles}
LOCKS = Path(__file__).with_name("locks")


def folder(name):
    return cache_dir("sources", name)


def recipe_record(recipe):
    """The recipe as plain JSON data: what a source's pins were made from. Seeds are left out: they only save
    downloads, since a seed is used only when it already holds the pinned bytes."""
    fields = {k: v for k, v in dataclasses.asdict(recipe).items() if k != "seed"}
    return json.loads(json.dumps({"kind": type(recipe).__name__, **fields}))


def write_recipe_record(name, recipe):
    (LOCKS / f"{name}.recipe.json").write_text(json.dumps(recipe_record(recipe), indent=1, sort_keys=True) + "\n",
                                               encoding="utf-8", newline="\n")


def recipe_changed(name):
    path = LOCKS / f"{name}.recipe.json"
    return not path.exists() or json.loads(path.read_text(encoding="utf-8")) != recipe_record(SOURCES[name].recipe)


def _digest(pins):
    return hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()[:12]


def _pin(name, recipe):
    """Fetch the recipe afresh into an empty folder, with no seeds and nothing reused from the cache, pin what arrived
    and record the recipe beside the pins; then put it in the cache. Before the pins change, a cached copy that held the
    old pins is set aside as <name>.pinned-<digest of those pins>, and it is kept if they do change, since its publisher
    may no longer serve those bytes. Whatever fails before the new pins are written leaves the old pins and copy."""
    target, old = folder(name), _pins(name)
    old_intact = bool(old) and not inputs.problems(name, target)
    staging = cache_dir("sources", name + ".pinning")
    shutil.rmtree(staging)
    staging.mkdir()
    FETCH[type(recipe)](recipe, {}, staging, False)
    aside = target.with_name(f"{name}.pinned-{_digest(old)}")
    if old_intact and not aside.exists():
        target.rename(aside)
    else:
        shutil.rmtree(target)               # nothing pinned, or those pinned bytes are set aside already
    inputs.write_tree(name, staging)
    write_recipe_record(name, recipe)
    staging.rename(target)
    if _pins(name) == old:
        if aside.exists():
            shutil.rmtree(aside)
    elif old_intact:
        print(f"{name}: the pins changed; the copy pinned before is kept in {aside}")


def fetch(name, pin=False):
    """This machine's verified copy of the source; the folder.

    A copy recorded for this machine that verifies is used as it is (a kept copy is recorded with
    `python -X utf8 -m tools.inputs set NAME FOLDER`). Otherwise the cache folder is filled, checked against the pins and
    recorded. A fetch refuses a recipe that has changed since it was pinned; `pin` fetches the recipe afresh and
    rewrites the pins (see _pin)."""
    recipe = SOURCES[name].recipe
    if pin:
        _pin(name, recipe)
    elif recipe_changed(name):
        raise inputs.Blocked(f"source {name}'s recipe has changed since it was pinned; "
                             f"run `python -X utf8 -m tools.site.sources pin {name}`")
    else:
        kept = inputs.locate(name)
        if kept is not None and not inputs.problems(name, kept):
            return kept
        FETCH[type(recipe)](recipe, _pins(name), folder(name), True)
    found = inputs.problems(name, folder(name))
    if found:
        raise inputs.Blocked(f"source {name} does not match its pins: " + "; ".join(found[:5]))
    inputs.main(["set", name, str(folder(name))])
    return folder(name)


def state(name):
    path = inputs.locate(name)
    if path is None:
        return "missing"
    return "ok" if not inputs.problems(name, path) else "MISMATCH"


def main(argv):
    if argv[:1] == ["list"]:
        for name, s in SOURCES.items():
            print(f"{name:18} {state(name):9} {s.licence}; {s.attribution}")
        return 0
    if argv[:1] in (["fetch"], ["pin"]):
        pin = argv[0] == "pin"
        names = argv[1:] or list(SOURCES)
        if pin and not argv[1:]:
            left = [n for n in names if SOURCES[n].current_release_only]
            names = [n for n in names if n not in left]
            print(f"left as pinned: {', '.join(left)} (the publisher serves only its current release; pin by name)")
        failed = 0
        for name in names:
            try:
                fetch(name, pin=pin)
            except (inputs.Blocked, OSError, ValueError) as e:
                failed += 1
                print(f"{name}: not fetched: {e}\n  {inputs.spec(name)['obtain']}", file=sys.stderr)
        return 1 if failed else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
