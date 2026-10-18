# Hogwarts flyover

A single HTML file that renders a raymarched 3D flyover of the Hogwarts grounds over their purported real location. All rendering happens in one inline GLSL fragment shader (WebGL2), fed by packed terrain, imagery and layout data inlined into the same file.

The castle, lake, grounds, Hogsmeade and the Forbidden Forest are laid out using only White Hound's essay *The Map of Hogwarts* and its maps, georeferenced onto real geographic data. The castle sits in the glen of the River Meig at Scardroy, Strathconon, Ross-shire (57.5223°N 4.9907°W). That is the nearest ground that fits the essay's descriptions of the horizon; the essay's own reference point, 57°30′N 5°00′W, lies 2.5 km SSW and doesn't fit them.

## View it

Open `hogwarts-flyover.html` in a browser with WebGL2 (tested in Microsoft Edge on Windows). The shader takes about 10–20 s to compile on first load. The camera then flies a captioned loop of about 7 minutes.

Controls: drag to look, **F** free flight (**WASD**, **E**/**C** up and down, **Shift** faster), **Space** pause, **←**/**→** scrub, **↑**/**↓** time of day, **Q** quality, **L** detail, **H** hide the HUD.

URL parameters: `t` (start time in seconds), `tod` (hour of day, BST; default 18.5), `q` (quality 0–2), `scale` (fixed render scale instead of adaptive), `pause=1`, `detail=0`, `hud=0`.

## Edit and rebuild

`hogwarts-flyover.html` is generated, so edit the sources in `src/` and rebuild it with Python 3 (standard library only):

```
python src/assemble.py
```

| Path | Contents |
|---|---|
| `src/frag1.glsl` – `src/frag5.glsl` | The fragment shader in five parts: (1) uniforms, noise, terrain, water and trees; (2) castle, buildings, walls, hedges and set-pieces; (3) the single loop that runs every ray query; (4) surface colours, sky and fog; (5) shading and `main` |
| `src/template.html` | Page markup, CSS, HUD, data decoders and a CPU copy of the terrain sampler |
| `src/template_b.html` | WebGL setup: textures, shader `#define`s from the layout metadata, shader compilation, temporal upsampling and render targets |
| `src/template_c.html` | Sun position on 1 September 1991, lighting, camera path, captions, controls and render loop |
| `src/assemble.py` | Inlines the shader and `build/data.js` into the templates |
| `build/data.js` | Packed data layers (terrain and water at several resolutions, Sentinel-2 imagery, layers traced from the essay's maps, castle and forest) plus layout metadata |
| `src/shoot.py`, `src/bench.py` | Headless Edge checks: screenshots at given flyover times, and frame times per debug flag. They need `playwright` and write to `view/` |

## Data pipeline (reference only)

`build/data.js` was produced by the Python scripts in `src/`, which use numpy, scipy, Pillow and pyproj. They can't be rerun from this repository alone. Their inputs, about 500 MB of downloaded sources and intermediates, are not included. Several steps also ran as one-off commands that were never saved as scripts: the downloads, the DEM and Sentinel-2 mosaics (`dem/mosaic.npy`, `s2tiles/mosaic.png`), classifying the colours of the essay's maps (`build/cls_*.npy`), extracting the castle footprint and its named points (`build/castle_mass_g.npy`, `build/castle_named.json`), the lake mask from the grounds map (`lake_mask_grounds.npy`), the 120 m imagery layer (`build/S_B.npy`) and the plantation masks (`build/SF_M.npy`, `build/SF_A.npy`).

The saved scripts run in this order: `features.py`, then `build_final.py`, then `build_extras.py`; `build_regional.py` is independent of those three; `pack.py` runs last and writes `build/data.js`. `common.py` holds shared constants and coordinate transforms and reads `georef.json`, which records the measured origin, rotation and scale of the essay's maps. `build_extras.py` also reads `feature_offsets.json`. `build_local.py`, `build_terrain.py` and `build_castle.py` are earlier iterations that `build_final.py` replaced.

## Sources and credits

- Layout: White Hound, *The Map of Hogwarts* (essay and maps), [archived copy](https://web.archive.org/web/20261008141630/http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts.htm).
- Terrain: Copernicus GLO-30 DEM © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA.
- Imagery: Sentinel-2 cloudless 2023 by EOX IT Services GmbH (contains modified Copernicus Sentinel data 2023), licensed CC BY-NC-SA 4.0. `build/data.js` and the generated page contain imagery derived from it, so those terms apply to them.
- Lochs and rivers: © OpenStreetMap contributors, ODbL.
- This is an unofficial, non-commercial fan visualisation. Harry Potter © J.K. Rowling / Warner Bros.
