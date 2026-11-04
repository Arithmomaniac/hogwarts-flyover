"""The elevation-model comparison for a castle site's terrain (gate a): two sheets and a table, CPU only.

    python -X utf8 -m tools.site.compare OUT_DIR [SITE]

OUT_DIR/terrain-maps.png      hillshades of the grounds, each model's difference from GLO-30 over the 5 km layer, imagery
OUT_DIR/terrain-sections.png  four sections through the layout with the design levels
OUT_DIR/terrain-comparison.json and .md  the numbers: registration, canopy, earthworks and sun with the lake at the site's
                              level (the review's model), resolution, datum and licence
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from tools import locks  # noqa: E402
from tools.inputs import require  # noqa: E402
from tools.site import layout as LY, registration as RG, sources as SRC, sun as SU, terrain as T  # noqa: E402
from common import SITES  # noqa: E402

NAMES = {"glo30": "Copernicus GLO-30", "fabdem": "FABDEM V1-2", "gedtm30": "GEDTM30 v1.2", "os50": "OS Terrain 50"}
FACTS = {
    "glo30": ("surface model: canopy and buildings included", "1″ × 1.5″ (31 × 25 m)", "EGM2008",
              "TanDEM-X 2011–2015", "site-glo30"),
    "fabdem": ("GLO-30 with forests and buildings removed by machine learning", "1″ (31 × 17 m)", "EGM2008",
               "GLO-30 2021_1; released 2022", "site-fabdem"),
    "gedtm30": ("bare-earth ensemble, machine learning on ICESat-2 and GEDI", "1″ (31 × 17 m)", "EGM2008",
                "inputs 2006–2015; v1.2.0 2026", "site-gedtm30"),
    "os50": ("bare-earth grid from Ordnance Survey mapping", "50 m posts", "Ordnance Datum Newlyn",
             "release 2026-07 (files dated 2026-05-29)", "site-os50"),
}
MASK_COLOURS = {"lake": "cyan", "castle": "white", "pitch": "gold", "station": "violet", "village": "tomato",
                "forest": "lime"}
# castle-relative sections (E, N metres): name, start, end
SECTIONS = [("South to north: lake, castle, greenhouses", (121, -1000), (60, 700)),
            ("Castle to the Forest, down Allt Mhartuin (north-west)", (0, 0), (-1300, 1600)),
            ("Castle to Hogsmeade (south-south-west)", (0, 0), (-800, -1400)),
            ("Station to the main line's map edge and on (north-north-east)", (309, 117), (420, 1500))]


def hillshade(a, step, az=315.0, alt=45.0):
    """Lambertian shading under a sun at azimuth az (clockwise from north) and altitude alt; rows run south to north."""
    gy, gx = np.gradient(a, step)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    a_r, e_r = np.radians(az), np.radians(alt)
    return np.clip(n @ np.array([np.sin(a_r) * np.cos(e_r), np.cos(a_r) * np.cos(e_r), np.sin(e_r)]), 0, 1)


def grid_of(model, anchor, half, step):
    E0, N0, n = T.square(anchor, half, step)
    return T.model_grid(model, E0, N0, step, n, n)


def extent(g, anchor):
    E, N = g.coords()
    return [E[0] - anchor[0] - g.step / 2, E[-1] - anchor[0] + g.step / 2, N[0] - anchor[1] - g.step / 2,
            N[-1] - anchor[1] + g.step / 2]


def outlines(ax, tpl):
    m, e = tpl["masks"], tpl["mask_extent"]
    xs, zs = np.arange(e[0], e[1] + 1e-6, e[4]), np.arange(e[2], e[3] + 1e-6, e[4])
    for k, name in enumerate(LY.MASKS):
        if name in MASK_COLOURS:
            mask = ndi.gaussian_filter(m[k].astype(float), 6.0) if name == "forest" else m[k].astype(float)
            ax.contour(xs, zs, mask, levels=[0.5], colors=MASK_COLOURS[name], linewidths=0.8 if name == "forest" else 1.2)


def imagery(anchor, half, px):
    """The Sentinel-2 cloudless z14 tiles over anchor ± half, resampled to px metres (row 0 north)."""
    root = require("site-s2-x") / "tiles"
    z, x0, x1, y0, y1 = SRC.SOURCES["site-s2-x"].recipe.ranges[0]
    mosaic = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256, 3), np.float32)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            tile = np.asarray(Image.open(root / f"{z}_{x}_{y}.jpg").convert("RGB"), np.float32) / 255.0
            mosaic[(y - y0) * 256:(y - y0 + 1) * 256, (x - x0) * 256:(x - x0 + 1) * 256] = tile
    t = np.arange(-half, half, px) + px / 2
    X, Z = np.meshgrid(t, t[::-1])
    lon, lat = T.ll(anchor[0] + X, anchor[1] + Z)
    n = 2 ** z
    tx = (np.asarray(lon) + 180) / 360 * n
    ty = (1 - np.arcsinh(np.tan(np.radians(np.asarray(lat)))) / math.pi) / 2 * n
    return np.stack([ndi.map_coordinates(mosaic[..., c], [(ty - y0) * 256 - 0.5, (tx - x0) * 256 - 0.5], order=1)
                     for c in range(3)], -1)


def canopy(near, models):
    """Over OpenStreetMap's woodland in the 5 km layer: its share of the layer, and how far each model stands below
    GLO-30 there (GLO-30 is a surface model, so this is how much canopy each one removes)."""
    glo = near["glo30"].a
    mask = T.woodland_fraction(near["glo30"]).a > 0.5
    out = {"share_of_layer": float(mask.mean())}
    for m in models:
        d = glo[mask] - near[m].a[mask]
        out[m] = {"median_below_glo30_m": float(np.nanmedian(d)), "p90_below_glo30_m": float(np.nanpercentile(d, 90))}
    return out


def numbers(site, wide, near, registration):
    anchor, level = SITES[site]["anchor"], SITES[site]["lake"]
    water = T.water_fraction(wide["glo30"])
    out = {"site": site, "anchor": anchor, "lake": level, "models": {}, "canopy": canopy(near, list(wide)[1:])}
    for m, g in wide.items():
        w = LY.earthworks_at(g, water, [anchor], level)
        s = SU.site_metrics(g, np.array([anchor[0]]), np.array([anchor[1]]), np.array([level]))
        kind, posts, datum, vintage, source = FACTS[m]
        castle = LY._sample(LY.bed_of(g, water), np.atleast_2d(anchor), LY.template()["castle_10"])[0]
        out["models"][m] = {
            "name": NAMES[m], "kind": kind, "posts": posts, "datum": datum, "vintage": vintage,
            "licence": SRC.SOURCES[source].licence, "attribution": SRC.SOURCES[source].attribution,
            "registration": registration["models"][m],
            "castle_ground_m": {"min": float(np.nanmin(castle)), "median": float(np.nanmedian(castle))},
            "earthworks": {k: float(w[k][0]) for k in ("P", "Ls", "V_lake", "V_rim", "V_castle", "V_pitch", "V_station",
                                                       "V_total", "max_cut", "max_fill", "lake_relief")},
            "ordering_m": {k: float(w[k][0]) for k in ("o_gates", "o_hagrid", "o_village", "o_forest", "o_aragog",
                                                       "o_lawn", "o_station")},
            "sun": {k: float(s[k][0]) for k in ("set_0606", "set_0621", "rise_0331", "rise_0504", "rise_0622",
                                                "north_max", "south_max")},
        }
    return out


def maps(site, near, out_path):
    anchor = SITES[site]["anchor"]
    tpl = LY.template()
    local = {m: grid_of(m, anchor, 1600.0, 10.0) for m in near}
    fig, axs = plt.subplots(2, 4, figsize=(24, 12.5), dpi=90)
    for ax, (m, g) in zip(axs[0], local.items()):
        e = extent(g, anchor)
        ax.imshow(hillshade(g.a, g.step), origin="lower", extent=e, cmap="gray", vmin=0, vmax=1)
        ax.contour(*np.meshgrid(*[c - a for c, a in zip(g.coords(), anchor)]), g.a, levels=np.arange(200, 420, 10),
                   colors="k", linewidths=0.3, alpha=0.5)
        outlines(ax, tpl)
        ax.set_xlim(-1600, 1600)
        ax.set_ylim(-1600, 1600)
        ax.set_title(f"{NAMES[m]}: hillshade, 10 m contours", fontsize=11)
    woods = T.woodland_fraction(near["glo30"])
    for ax, m in zip(axs[1][:3], ("fabdem", "gedtm30", "os50")):
        d = near[m].a - near["glo30"].a
        g = near["glo30"]
        im = ax.imshow(d, origin="lower", extent=extent(g, anchor), cmap="RdBu", vmin=-10, vmax=10)
        ax.contour(*np.meshgrid(*[c - a for c, a in zip(g.coords(), anchor)]), woods.a, levels=[0.5], colors="darkgreen",
                   linewidths=0.8)
        outlines(ax, tpl)
        ax.set_xlim(-5000, 5000)
        ax.set_ylim(-5000, 5000)
        ax.set_title(f"{NAMES[m]} minus GLO-30 (m)", fontsize=11)
    fig.colorbar(im, ax=axs[1][:3], shrink=0.7, label="m (red: model below GLO-30; dark green: mapped woodland)")
    axs[1][3].imshow(np.clip(imagery(anchor, 5000.0, 20.0) * 1.2, 0, 1), extent=[-5000, 5000, -5000, 5000])
    outlines(axs[1][3], tpl)
    axs[1][3].set_title("Sentinel-2 cloudless 2023 (EOX), 5 km layer", fontsize=11)
    for ax in axs.ravel():
        ax.set_xlabel("m east of the castle", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.text(0.01, 0.005, "Layout: lake cyan, castle white, pitch gold, station violet, Hogsmeade tomato, Forest lime. "
             "Differences are bilinear on a 25 m grid; elevation sources as credited in terrain-comparison.md.",
             fontsize=9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def sections(site, wide, out_path):
    anchor, level = SITES[site]["anchor"], SITES[site]["lake"]
    fig, axs = plt.subplots(len(SECTIONS), 1, figsize=(16, 4.2 * len(SECTIONS)), dpi=90)
    colours = {"glo30": "k", "fabdem": "tab:green", "gedtm30": "tab:orange", "os50": "tab:blue"}
    for ax, (title, a, b) in zip(axs, SECTIONS):
        s = np.linspace(0, 1, 600)
        E, N = a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s
        dist = s * math.hypot(b[0] - a[0], b[1] - a[1])
        for m, g in wide.items():
            ax.plot(dist, g.sample(anchor[0] + E, anchor[1] + N), color=colours[m], lw=1.2, label=NAMES[m])
        ax.axhline(level, color="cyan", ls="--", lw=1, label=f"lake {level:.0f} m")
        ax.axhline(level + SU.CLIFF, color="0.4", ls=":", lw=1, label=f"castle platform {level + SU.CLIFF:.1f} m")
        tpl = LY.template()
        m, e = tpl["masks"], tpl["mask_extent"]
        j = np.round((E - e[0]) / e[4]).astype(int)
        i = np.round((N - e[2]) / e[4]).astype(int)
        ok = (i >= 0) & (i < m.shape[1]) & (j >= 0) & (j < m.shape[2])
        for k, name in enumerate(LY.MASKS):
            if name in ("lake", "castle", "station", "village", "pitch"):
                inside = np.zeros(len(s), bool)
                inside[ok] = m[k][i[ok], j[ok]] > 0
                if inside.any():
                    ax.fill_between(dist, 0, 1, where=inside, transform=ax.get_xaxis_transform(),
                                    color="dimgray" if name == "castle" else MASK_COLOURS[name], alpha=0.15)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("m")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, ncol=3)
    axs[-1].set_xlabel("m along the section (shaded: lake, castle, station, pitch, village)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def table(nums):
    m = nums["models"]
    cols = list(m)
    rows = [
        ("What it is", lambda r: r["kind"]),
        ("Grid", lambda r: r["posts"]),
        ("Heights", lambda r: r["datum"]),
        ("Vintage", lambda r: r["vintage"]),
        ("Licence", lambda r: r["licence"]),
        ("Offset from GLO-30 (E, N)", lambda r: "—" if "offset_m" not in r["registration"] else
         f"{r['registration']['offset_m']['dE']:+.1f}, {r['registration']['offset_m']['dN']:+.1f} m"),
        ("Height minus GLO-30 on open level ground", lambda r: "—" if "vertical_m" not in r["registration"] else
         f"{r['registration']['vertical_m']['median']:+.1f} m"),
        ("Summits: median height error against mapped heights", lambda r: f"{r['registration']['summits']['dz']:+.1f} m"),
        ("Ground under the castle (lowest, median)", lambda r: f"{r['castle_ground_m']['min']:.0f}, "
                                                             f"{r['castle_ground_m']['median']:.0f} m"),
        ("Castle mound at the most (platform minus ground)", lambda r: f"{r['earthworks']['max_fill']:.0f} m"),
        ("Earthworks: lake + rim + castle + pitch + station", lambda r: f"{r['earthworks']['V_total'] / 1e6:.2f} Mm³ "
         f"({r['earthworks']['V_lake'] / 1e6:.2f} + {r['earthworks']['V_rim'] / 1e6:.2f} + "
         f"{r['earthworks']['V_castle'] / 1e6:.2f} + {r['earthworks']['V_pitch'] / 1e6:.2f} + "
         f"{r['earthworks']['V_station'] / 1e6:.2f})"),
        ("Gates above Hagrid's hut (White Hound: below)", lambda r: f"{r['ordering_m']['o_hagrid']:.0f} m"),
        ("Sunsets 6 Jun 1994 / 21 Jun 1996 (min early)", lambda r: f"{r['sun']['set_0606']:.1f} / {r['sun']['set_0621']:.1f}"),
        ("Sunrises 31 Mar / 4 May 1998 (min late)", lambda r: f"{r['sun']['rise_0331']:.1f} / {r['sun']['rise_0504']:.1f}"),
        ("Skyline due north / south (degrees)", lambda r: f"{r['sun']['north_max']:.1f} / {r['sun']['south_max']:.1f}"),
    ]
    lines = ["| Measure | " + " | ".join(m[c]["name"] for c in cols) + " |", "| --- |" + " --- |" * len(cols)]
    for label, f in rows:
        lines.append(f"| {label} | " + " | ".join(f(m[c]) for c in cols) + " |")
    c = nums["canopy"]
    lines.append(f"| Canopy: on OpenStreetMap's woodland ({c['share_of_layer'] * 100:.1f}% of the 5 km layer), each model's "
                 "median and 90th percentile below GLO-30 | — | "
                 + " | ".join(f"{c[k]['median_below_glo30_m']:.1f}, {c[k]['p90_below_glo30_m']:.1f} m"
                              for k in cols[1:]) + " |")
    return "\n".join(lines) + "\n"


def main(argv):
    out = Path(argv[0])
    site = argv[1] if len(argv) > 1 else "x"
    out.mkdir(parents=True, exist_ok=True)
    anchor = SITES[site]["anchor"]
    with locks.heavy(f"terrain comparison at {site} (tools.site.compare)"):
        wide = {m: grid_of(m, anchor, 33000.0, 25.0) for m in T.MODELS}
        near = {m: grid_of(m, anchor, 5000.0, 25.0) for m in T.MODELS}
        nums = numbers(site, wide, near, RG.report(site))
        (out / "terrain-comparison.json").write_text(json.dumps(nums, indent=1, ensure_ascii=False), encoding="utf-8")
        (out / "terrain-comparison.md").write_text(table(nums), encoding="utf-8")
        maps(site, near, out / "terrain-maps.png")
        sections(site, wide, out / "terrain-sections.png")
    print(table(nums))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
