"""Public LiDAR coverage around a castle site, from the Scottish Remote Sensing Portal's catalogue.

The catalogue's product search returns every DTM product whose footprint intersects a box; footprints can overstate
coverage (the library report found a Phase 2 product whose raster lay kilometres away), so each product's GeoTIFF
header is read by range request and its real extent compared with the scene's layers.

    python -X utf8 -m tools.site.lidar [SITE]      # default x; prints the products and their coverage as JSON
"""
import json
import sys
import urllib.request

import numpy as np

from tools.site.terrain import bng, ll
from common import SITES

API = "https://api.remotesensing.data.gov.scot"
LAYERS = {"castle": 300.0, "local": 1200.0, "M": 5000.0, "A": 12000.0}


def _post(path, body):
    request = urllib.request.Request(API + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _box_wkt(anchor, half):
    corners = [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]
    lon, lat = ll(np.array([anchor[0] + c[0] for c in corners]), np.array([anchor[1] + c[1] for c in corners]))
    return "POLYGON((" + ",".join(f"{x:.5f} {y:.5f}" for x, y in zip(lon, lat)) + "))"


def dtm_collections():
    with urllib.request.urlopen(API + "/search/collection/scotland-gov/*", timeout=120) as response:
        return [c["name"] for c in json.load(response)["result"] if "/dtm" in c["name"]]


def products(anchor, half):
    out, offset = [], 0
    while True:
        page = _post("/search/product", {"collections": dtm_collections(), "footprint": _box_wkt(anchor, half),
                                         "spatialop": "intersects", "offset": offset, "limit": 50})["result"]
        out += page
        if len(page) < 50:
            return out
        offset += 50


def _url(product):
    data = product.get("data", {}).get("product", {})
    if "http" in data:
        return data["http"]["url"]
    s3 = data.get("s3")
    return f"https://{s3['bucket']}.s3.{s3['region']}.amazonaws.com/{s3['key']}" if s3 else None


def raster_extent(url):
    """(west, south, east, north) of a GeoTIFF in British National Grid, read from its header by range request."""
    import rasterio

    with rasterio.open("/vsicurl/" + url) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 27700:
            raise ValueError(f"{url} is not in British National Grid ({ds.crs})")
        return tuple(ds.bounds)


def coverage(anchor, extent):
    """The share of each layer's square (TM, castle-centred) that falls inside a BNG rectangle."""
    out = {}
    for layer, half in LAYERS.items():
        t = np.linspace(-half, half, 41)
        E, N = np.meshgrid(anchor[0] + t, anchor[1] + t)
        x, y = bng(E, N)
        inside = (x >= extent[0]) & (x <= extent[2]) & (y >= extent[1]) & (y <= extent[3])
        out[layer] = round(float(inside.mean()), 3)
    return out


def main(argv):
    site = argv[0] if argv else "x"
    anchor = SITES[site]["anchor"]
    report = []
    for p in products(anchor, LAYERS["A"]):
        url = _url(p)
        entry = {"name": p["name"], "collection": p["collectionName"], "url": url}
        try:
            entry["extent_bng"] = raster_extent(url)
            entry["coverage"] = coverage(anchor, entry["extent_bng"])
        except Exception as e:                      # an unreadable product is reported, not hidden
            entry["error"] = str(e)
        report.append(entry)
    print(json.dumps({"site": site, "anchor": anchor, "products": report}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
