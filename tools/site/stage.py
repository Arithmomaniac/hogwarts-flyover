"""Stage the data pipeline's inputs for a castle site into the worktree's untracked working paths, from pinned inputs
only, so src/build_*.py read them where they always have (build/, dem/, s2tiles/ and the root extracts).

    python -X utf8 -m tools.site.stage SITE [--dem MODEL]     # SITE: scardroy or x; MODEL: the site's "dem" by default

Site-independent inputs (the essay maps' class rasters, the castle mass and its named points, the grounds' lake mask)
come from the pinned original intermediates. Scardroy keeps the original build's own geographic inputs, so it rebuilds
the original scene; its data differs from the original's only by the fixes made since for every site (for example the
pitch graded and kept clear of trees, the Forest map's ink no longer read as trees, and the page's place note and
elevation-model credit), so it no longer matches the pinned data-before-railway byte for byte. Site X's come from the
site-* sources: the DEM mosaic resampled from the chosen model onto GLO-30's lattice (1" north-south by 1.5" east-west
from 58 N, 6 W, the lattice common.dem_sample reads), the two OpenStreetMap extracts, and the Sentinel-2 mosaic
assembled from its zoom-14 tiles beside its zoom-11 tiles. build/stage.json records what was staged, and common.py
builds the site it names.
"""
import argparse
import json
import shutil
import sys

import numpy as np
from PIL import Image

from tools.inputs import require
from tools.paths import BUILD, ROOT, sha256, write_atomic
from tools.site import sources as SRC, terrain as T

INDEPENDENT = ("build/cls_g.npy", "build/cls_h.npy", "build/cls_ff.npy", "build/castle_mass_g.npy",
               "build/castle_named.json", "lake_mask_grounds.npy")
ORIGINAL_GEOGRAPHY = ("dem/mosaic.npy", "osm.json", "osm_water_wide.json", "s2tiles/mosaic.png", "s2tiles/mosaic.json")
LATTICE = (3600, 4800)            # GLO-30's N57 band, W006 and W005 side by side
STAMP = BUILD / "stage.json"


def _copy(source, rel):
    target = ROOT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return rel


def dem_mosaic(model):
    """The model's heights on GLO-30's lattice of the N57 band from 6 W to 4 W: rows from 58 N southwards every 1",
    columns from 6 W eastwards every 1.5". A height missing anywhere is refused here: the build's spline sampling
    would spread it over the whole local bed."""
    if model == "glo30":
        import rasterio

        bands = []
        for w in (6, 5):
            with rasterio.open(require("site-glo30") / f"Copernicus_DSM_COG_10_N57_00_W00{w}_00_DEM.tif") as ds:
                bands.append(ds.read(1).astype(np.float32))
        out = np.concatenate(bands, axis=1)
    else:
        lon = -6.0 + np.arange(LATTICE[1]) / 2400.0
        out = np.empty(LATTICE, np.float32)
        for r0 in range(0, LATTICE[0], 300):
            lat = 58.0 - np.arange(r0, min(r0 + 300, LATTICE[0])) / 3600.0
            out[r0:r0 + len(lat)] = T.heights_ll(model, lon[None, :], lat[:, None])
    missing = np.count_nonzero(~np.isfinite(out))
    if missing:
        raise ValueError(f"{model}: {missing} heights missing on the lattice")
    return out


def s2_mosaic(recipe):
    """(RGB mosaic, its metadata) of the source's zoom-14 tiles, as the original build stored s2tiles/mosaic.png."""
    root = require("site-s2-x") / "tiles"
    z, x0, x1, y0, y1 = next(r for r in recipe.ranges if r[0] == 14)
    image = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256, 3), np.uint8)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            tile = np.asarray(Image.open(root / f"{z}_{x}_{y}.jpg").convert("RGB"))
            image[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = tile
    return image, {"z": z, "x0": x0, "y0": y0, "W": image.shape[1], "H": image.shape[0]}


def stage(site, dem=None):
    """Put the site's inputs in place; the files staged. A record of the stage in progress replaces the stamp first,
    and the stamp is written last, so a stage that fails part-way leaves that record: the build steps refuse to run on
    the mixture (common.site_name), and staging again replaces it."""
    dem = dem or SRC.SITES[site]["dem"]
    write_atomic(STAMP, (json.dumps({"staging": site, "dem": dem}) + "\n").encode("utf-8"))
    original = require("original-intermediates")
    staged = [_copy(original / rel, rel) for rel in INDEPENDENT]
    shutil.rmtree(ROOT / "s2tiles", ignore_errors=True)
    shutil.rmtree(ROOT / "dem", ignore_errors=True)
    if site == "scardroy":
        if dem != "glo30":
            raise ValueError("Scardroy rebuilds from its original GLO-30 mosaic only")
        staged += [_copy(original / rel, rel) for rel in ORIGINAL_GEOGRAPHY]
        staged += [_copy(p, f"s2tiles/{p.name}") for p in sorted((original / "s2tiles").glob("11_*.jpg"))]
    elif site == "x":
        (ROOT / "dem").mkdir()
        np.save(ROOT / "dem" / "mosaic.npy", dem_mosaic(dem))
        staged.append("dem/mosaic.npy")
        staged.append(_copy(require("site-osm-x") / "osm.json", "osm.json"))
        staged.append(_copy(require("site-osm-water-x") / "osm_water_wide.json", "osm_water_wide.json"))
        recipe = SRC.SOURCES["site-s2-x"].recipe
        image, meta = s2_mosaic(recipe)
        (ROOT / "s2tiles").mkdir()
        Image.fromarray(image).save(ROOT / "s2tiles" / "mosaic.png")
        (ROOT / "s2tiles" / "mosaic.json").write_text(json.dumps(meta), encoding="utf-8")
        staged += ["s2tiles/mosaic.png", "s2tiles/mosaic.json"]
        staged += [_copy(p, f"s2tiles/{p.name}") for p in sorted((require("site-s2-x") / "tiles").glob("11_*.jpg"))]
    else:
        raise ValueError(f"unknown site {site!r}")
    BUILD.mkdir(exist_ok=True)
    record = {"site": site, "dem": dem, "files": {rel: sha256(ROOT / rel) for rel in staged}}
    write_atomic(STAMP, (json.dumps(record, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    return staged


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site", choices=sorted(SRC.SITES))
    parser.add_argument("--dem", choices=list(T.MODELS), help="the elevation model (default: the site's)")
    args = parser.parse_args(argv)
    staged = stage(args.site, args.dem)
    print(f"staged {len(staged)} inputs for {args.site}; record in {STAMP.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
