> Created/edited by GitHub Copilot; pending human review.

# Rebuilding and reconstructing the Hogwarts flyover

This guide records the scene's construction so a rewrite can preserve its reasoning without preserving its original renderer. For the live site, controls and source credits, see the [README](README.md).

The scene visualizes **White Hound's interpretation**, not a definitive location or architectural plan. Fictional geography comes only from [*The Map of Hogwarts* and related pages](https://web.archive.org/web/20261008141630/http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts.htm); geographic context comes from elevation, satellite imagery and OpenStreetMap data. It does not use the films' castle or theme-park layout.

This guide describes the accepted WebGL version based on commit [`9153cc9`](https://github.com/Arithmomaniac/hogwarts-flyover/commit/9153cc939f10411d596a7fe44429381d20cc5306), not the local experimental WebGPU work. Assembling the existing page is supported; regenerating its source data is not yet self-contained.

## Contents

- [Assembling the existing page](#assembling-the-existing-page)
- [Evidence and interpretation](#evidence-and-interpretation)
- [Location and coordinate systems](#location-and-coordinate-systems)
- [Terrain, water and imagery](#terrain-water-and-imagery)
- [Castle and other buildings](#castle-and-other-buildings)
- [Grounds, village and forest](#grounds-village-and-forest)
- [Rendering and camera](#rendering-and-camera)
- [Reconstructing the scene](#reconstructing-the-scene)
- [Reproducibility limits](#reproducibility-limits)
- [Sources and credits](README.md#sources-and-credits)

## Assembling the existing page

The generated page can be rebuilt from a **clean clone of public main**, using Python 3's standard library:

```powershell
python src\assemble.py
```

This assembles the tracked GLSL, templates and already-packed [`build/data.js`](build/data.js). It does **not** regenerate the geographic/layout data. A local experimental checkout may have a different assembler; the public source and deployed page use WebGL.

GitHub Pages publishes only this HTML, renamed to `index.html` in a temporary hosting artifact. The [deployment workflow](.github/workflows/pages.yml) does not expose the source directory, data-preparation inputs or local experiment reports as website files.

## Evidence and interpretation

Keep three kinds of information separate when rewriting:

| Kind | Examples | How to use it |
| --- | --- | --- |
| Source interpretation | White Hound's grounds maps, castle plan, relative positions and textual arguments | Preserve the source relationship and its uncertainty; do not label it definitive canon |
| Geographic measurements | Copernicus heights, real watercourses, satellite colors | Provide regional context; fictional grounds deliberately modify the local landscape |
| Implementation choices | Exact Scardroy anchor, tower heights in metres, smoothing, lake level, seeded roofs and trees | Revisit freely; these are modeling parameters, not independently established facts |

The essay itself describes its castle plan as one working possibility. Its [location](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/location.htm), [setting](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/setting.htm), [grounds](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/grounds.htm), [castle](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/castle.htm), [Hogsmeade](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/Hogsmeade.htm), [forest](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/forest.htm) and [journeys](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/journeys.htm) sections explain different parts of the interpretation. These are original-site links; use the archived index above to locate archived copies if the originals are unavailable.

Map assets used in preparation include `FF_map_unlabelled.gif`, `Hogwarts_map_unlabelled.gif`, `Hogsmeade_map_unlabelled.gif`, their labelled counterparts, and `castle_plan.gif`. They are source-site assets, **not files bundled in this repository**. Labels help identify landmarks; unlabelled maps reduce text contamination during mask extraction. Some remaining decorative areas were explicitly excluded in the scripts.

## Location and coordinate systems

### A chosen anchor, not a discovered castle site

The essay uses **57 degrees 30 minutes north, 5 degrees west** as a standard reference for calculations such as sunset times. That is not a precise, uniquely established castle position. This implementation instead places the castle near **57.5223 N, 4.9907 W**, in the River Meig glen near Scardroy, Strathconon, Ross-shire.

The implemented offset from the reference point is 560 m east and 2,480 m north in the projection below, about 2.54 km away. It was a placement choice intended to fit the fictional grounds into nearby mountain scenery. There is no preserved exhaustive location search establishing it as the nearest, only or best fit. A rewrite should treat the anchor as adjustable.

The original page HUD called this the "nearest ground" fitting the horizons. That stronger wording was not a validated geographic conclusion; the current About dialog describes the anchor as a modeling choice instead.

### Geographic and world frames

[`src/common.py`](src/common.py) defines a WGS84 transverse-Mercator frame with this PROJ definition:

```text
+proj=tmerc +lat_0=57.5 +lon_0=-5 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m
```

`pyproj.Transformer(..., always_xy=True)` receives **longitude, latitude**, not latitude, longitude. Project a geographic point into `(E_tm, N_tm)`, then subtract the chosen castle anchor:

```text
E = E_tm - 560
N = N_tm - 2480
```

World coordinates are metres east/north of the castle. In the shader, a point is `(x, y, z) = (E, elevation, N)`: positive `z` is north, not south. Elevation uses the numeric DEM heights with local edits; no separate vertical-datum conversion was recorded.

### Registering the essay maps

The 2,900 by 3,440 pixel Forest map is the common image frame. Pixel `U` increases rightward and `V` downward. [`georef.json`](georef.json) and `common.py` record the registration:

| Parameter | Value |
| --- | --- |
| Castle centroid in Forest-map pixels `(U0, V0)` | `(1510.0404553608678, 2810.5158763811805)` |
| Scale `s` | `2.8738285714285716` metres per Forest-map pixel |
| Rotation `a` | `25.1` degrees |
| Grounds source size | `2000 x 2348` pixels |
| Grounds rectangle in Forest frame `(left, right, top, bottom)` | `(1190.5, 1693.5, 2591.5, 3180.5)` |
| Hogsmeade source size | `1735 x 2261` pixels |
| Hogsmeade rectangle in Forest frame | `(949.5, 1385.5, 2719.5, 3283.5)` |

The actual image-to-world transform is:

```text
du = (U - U0) * s
dv = (V - V0) * s
E = du*cos(a) + dv*sin(a)
N = du*sin(a) - dv*cos(a)
```

The inverse is `U = U0 + (E*cos(a) + N*sin(a))/s`, `V = V0 + (E*sin(a) - N*cos(a))/s`. Angles passed to trigonometric functions must be in radians.

For grounds pixels `(u, v)`, use the stored **`(left, right, top, bottom)`** rectangle in this order:

```text
U = 1190.5 + u * (1693.5 - 1190.5) / 2000
V = 2591.5 + v * (3180.5 - 2591.5) / 2348
```

Apply the same construction using the Hogsmeade rectangle and image size. `common.py` recomputes separate horizontal/vertical scales from the rectangles; the stored `sgx`/`sgy` fields are not the authoritative calculations.

Two useful checks: `(U0, V0)` must map to `(0, 0)`, and the willow's grounds pixel `(1060, 492)` maps to approximately **`(-254.34, 184.24)` metres**. The nominal `0.72 m/pixel` used by native-map mask geometry and the rectangle-derived registration are slightly different approximations; preserve that distinction when reproducing this build.

The local raster follows map-image axes, not north-up. The regional rasters are north-up in world coordinates but stored with row zero at the **south** edge. Grid sampling uses texel centers; `build_final.py` fits world-to-UV affines, and `build_regional.py`/`frag1.glsl` account for regional half-texel extents. Reversing an array twice or omitting the half-texel offset produces misplaced features.

## Terrain, water and imagery

[`build_final.py`](src/build_final.py) constructs the detailed region; [`build_regional.py`](src/build_regional.py) constructs geographic context. Their important choices are:

| Layer | Extent/resolution | Construction |
| --- | --- | --- |
| Local grounds/village | Forest pixels `U=935..1705`, `V=2580..3295`; 2 m cells | DEM plus map-derived masks, water, edited ground and buildings |
| Castle footprint | Grounds window `(1030, 560)`, size `500 x 620` source pixels; 0.5 m output cells | Six footprint-distance layers and nearby tower indices |
| Forest-map layer | Entire Forest map; 8 m cells | Density and field fraction from map colors |
| Regional M | Castle-relative `+/-5 km`; 10 m grid | Resampled geographic heights, water and imagery |
| Regional A | `+/-12 km`; 30 m grid | Same, coarser |
| Regional B | `+/-36 km`; 120 m grid | Distant mountains and water |

The elevation source is Copernicus **GLO-30**, a digital surface model, not a surveyed bare-earth model. A 2 m or 10 m output grid does not turn it into measured terrain at that resolution. The missing preparation produced `dem/mosaic.npy`; `dem_sample` transforms each world point back to longitude/latitude before sampling that mosaic.

The local builder smooths the DEM, adds selected hillocks and the Shack's hill, and raises the castle platform to at least `153 + 36 = 189 m`. That models the essay's roughly 120-foot lakeside cliff; its exact platform profile is an implementation choice. The pitch, building pads and station corridor are flattened; roads receive additional smoothing.

The fictional lake outline comes from the grounds map, not the outline of an existing real loch. Its chosen surface level is **153 m**. A signed-distance field controls shoreline shape and a basin of roughly `0.8 + 11.5*(1-exp(-inward_distance/50))` metres depth. Shore gradients differ near the castle. Stream masks combine essay-map water with the real River Meig from OSM. Regional OSM water polygons are levelled using sampled DEM heights. The ship's berth is explicitly flooded so its map ink does not become a ship-shaped island.

The local layer blends into regional terrain over the first **80 m** inside its edge; regional levels also blend near their boundaries. The resulting landscape is intentionally a composite of fictional landforms and real surroundings, not an untouched geographic reconstruction.

Sentinel-2 cloudless 2023 imagery supplies regional ground color. [`pack.py`](src/pack.py) approximately removes acquisition-time hill shading using terrain normals and chosen illumination parameters, allowing the shader to relight the terrain. The imagery is not geometry, and this heuristic is not a physically complete recovery of surface reflectance.

## Castle and other buildings

### From map ink to footprints

[`common.py`](src/common.py) classifies the maps' indexed palette into lawn, scrub, dense/understory vegetation, water, building, earth, wood, white, black, red and other. [`features.py`](src/features.py) separates thick black boundary-wall strokes, outlines and probable tree ink. [`build_final.py`](src/build_final.py) excludes decorative rectangles and combines the maps: **Hogsmeade takes precedence where maps overlap**, then grounds, then Forest-map fallback.

Castle extraction was partly manual/one-shot: the build expects an untracked `castle_mass_g.npy` footprint and `castle_named.json` annotations. They are not recoverable just by rerunning the checked-in scripts. The final packed data retains resulting positions, footprints and metadata, but not a complete record of the extraction decisions.

### Castle masses and towers

The builder estimates footprint thickness with morphological opening and distance transforms. It separates low walls, walls, ranges, blocks, the Great Hall and keep. These are the **chosen top heights above the local base**, before additional roof shapes:

| Class | Height | Roof treatment |
| --- | --- | --- |
| Low wall | 2.2 m | Low/flat treatment |
| Wall | 13 m | Battlements |
| Range | 24 m | Sloped roof |
| Block | 32 m | Sloped roof |
| Great Hall | 21 m | Taller sloped roof |
| Keep | 27 m | Battlements |

These dimensions approximate the essay's tall, multi-storey castle; the exact metre values are not quoted source measurements. Each class has its own signed-distance footprint, clipped to a common outline. [`frag2.glsl`](src/frag2.glsl) unions the separately extruded classes so height changes become walls rather than interpolation ramps.

Candidate round towers are found from distance-field peaks, filtered for shape and separation, and supplemented by named towers. Non-special tower variation uses NumPy seed `7`. Named tower parameters are:

| Tower | Center `(E, N)` m | Radius / top above base | Shape |
| --- | --- | --- | --- |
| Astronomy (`AT`) | `(-25.69, -91.08)` | `3.6 / 58 m` | Thin shaft, overhanging open top |
| North (`NT`) | `(64.47, 180.79)` | `3.6 / 54 m` | Thin shaft, overhanging onion-dome top |
| Gryffindor (`GT`) | `(-2.24, -166.25)` | `7 / 30 m` | Broad base with three upper turrets |
| West/Owlery (`WT`) | `(-107.04, -9.09)` | `4.6 / 41 m` | Flat battlemented top |
| Ravenclaw (`RT`) | `(-100.75, 143.30)` | `5.2 / 45 m` | Conical top |

The essay motivates the distinctive tower forms; positions are registered annotations and exact radii/heights are modeling choices. These top values are inputs to procedural shapes, not necessarily the highest roof points. The packed castle textures also store the two nearby tower indices used to reduce shader lookups.

### Village and small buildings

Connected map components supply ordinary building footprints. Components below 10 square metres are dropped. White/glass, wood/stands and proximity to the hut/Shack select special kinds; larger footprints become larger/slate-roofed buildings. Remaining small buildings choose thatch versus slate with seed `11` and probability `0.72` for thatch.

Chosen eaves are 3 m for greenhouses and the hut, 6 m for the Shack, 8 m for buildings over 400 square metres, 6.5 m above 160 square metres, otherwise 4 or 5 m. The shader adds roofs according to kind and inward footprint distance. This is a massing model, not a reconstruction of individual interiors or surveyed elevations.

## Grounds, village and forest

Coordinates below are castle-relative metres. [`feature_offsets.json`](feature_offsets.json) contains earlier map-derived markers; it is **not** the final authority for every feature. The final builder refines some positions, and the packed metadata in `build/data.js` records the actual scene values. [`build_extras.py`](src/build_extras.py) adds named points and tour annotations.

| Feature | Placement and source relationship | Implementation/approximation |
| --- | --- | --- |
| Lake | Map mask; centroid about `(121.17, -417.25)` | Fictional basin and selected level; not a real-loch substitution |
| Hagrid's hut | `(-291.5, -191.9)`, at the mapped forest edge | Refined hut position; generic hut geometry |
| Pitch | Center `(-243.6, -485.9)`, bearing `271.25` degrees | Outer ellipse semi-axes `101.6, 54.6 m` include the modeled stadium; do not confuse them with playing-field dimensions |
| Station and train | Carriage/engine endpoints extracted from red map components | Principal-axis fit, platform/track corridor flattened to a sampled level |
| Hogsmeade | Registered village map, southwest of castle | Footprints from ink; heights/roofs assigned by the rules above |
| Shrieking Shack | Near `(-877, -883.5)`; separate mapped hill | Footprint classification and added hill/knoll |
| Greenhouses | Mapped white components; marker about `(42.46, 237.29)` | Glass building kind; low selected eaves |
| Gates | Wall-gap endpoints around `(-447.67, -498.61)` and `(-441.25, -512.32)` | Gap chosen near the map label; procedural gateposts/boars |
| Willow | Grounds pixel `(1060, 492)` -> `(-254.34, 184.24)` | Individually placed procedural tree |
| Tomb | Grounds pixel `(1033, 1763)` -> `(116.67, -653.79)` | White lakeside set-piece |
| Durmstrang ship | Hull axis about `(-54.85, -214.94)` to `(-63.35, -308.85)` | Map principal-axis extraction, procedural hull/masts and flooded berth |
| Forest clearings/cave | Registered Forest-map markers | Map-derived locations plus procedural vegetation/landform treatment |

Roads use earth-colored masks; walls use thick black strokes with classification/thinning heuristics; hedges use long thin ink components after excluding object outlines. Those are interpretations of drawing conventions, not literal physical widths of the pen strokes.

The Forbidden Forest's **density** comes from map vegetation colors and tree ink. Individual tree positions, sizes and shapes are procedural and deterministic, not individually sourced. Ground/water/building masks exclude trees. Density blends near map boundaries and fades with elevation between roughly 500 and 640 m; conifer fractions combine selected defaults with regional plantation masks. Near trees use 3D distance-field shapes on a 7 m cell grid; distant vegetation becomes a canopy representation. A rewrite can choose meshes instead without changing the map-derived forest footprint.

The scene includes features from different books, including the ship and tomb, while the lighting calculation uses 1 September 1991. It is therefore **not a historically consistent snapshot of that date**; the date controls the sun, not the existence of every object.

## Rendering and camera

The only runtime delivery is the generated HTML. Its templates decode compressed embedded arrays, create textures, compile GLSL, tabulate tree data and run temporal reconstruction/presentation passes. There are no downloaded models or scene-library dependencies.

| File | Responsibility |
| --- | --- |
| [`frag1.glsl`](src/frag1.glsl) | Uniforms, coordinate sampling, noise, terrain, water and vegetation |
| [`frag2.glsl`](src/frag2.glsl) | Castle, buildings, walls, hedges and set-pieces |
| [`frag3.glsl`](src/frag3.glsl) | Combined ray queries for terrain, water, vegetation and objects |
| [`frag4.glsl`](src/frag4.glsl) | Surface colors, satellite albedo, windows, sky and fog |
| [`frag5.glsl`](src/frag5.glsl) | Lighting, shading and main scene entry point |
| [`template.html`](src/template.html) | Markup, controls, embedded-data decoders and CPU terrain sampling |
| [`template_b.html`](src/template_b.html) | WebGL resources, shader definitions, tree prepass and temporal passes |
| [`template_c.html`](src/template_c.html) | Solar direction, camera path, captions, interaction and render loop |
| [`assemble.py`](src/assemble.py) | Concatenation/inlining into one page |

Terrain is height-field sampled; hard objects use conservative distance estimates. The combined marcher takes bounded steps, refines hits and shades by material. Shadows, lake reflections, fog, procedural detail and temporal accumulation are renderer choices rather than evidence about the place.

The camera path and captions are authored in `template_c.html`, not derived routes verified against the essay. The solar calculation uses 1 September 1991, BST and the selected location. Adaptive resolution, quality and detail controls trade rendering cost against appearance. The raymarcher is deliberately self-contained, but none of these techniques is a requirement for a more intentional rewrite.

## Reconstructing the scene

1. **Separate evidence from design parameters.** Start with a manifest of the essay pages/maps and geographic datasets. Record which constraints each source supports, and keep chosen heights, anchor, seeds and terrain edits visibly separate.
2. **Establish frames before geometry.** Select an anchor, define metre-based east/north coordinates, register the maps, and verify the centroid/willow examples above. Keep geographic coordinates, map pixels and world coordinates distinct.
3. **Prepare geographic context.** Mosaic/sample the elevation model, rasterize real water polygons/rivers, and prepare satellite colors. Keep provenance, source resolution and vertical-height assumptions with each layer.
4. **Extract fictional layout.** Classify map colors, remove annotations, identify footprints/lines/vegetation, and define overlap precedence. Store named landmarks and manual corrections explicitly rather than burying them in extraction commands.
5. **Construct the local landscape.** Blend the lake, castle cliff/platform, hillocks and flat functional areas into the real terrain. Review shoreline, drainage and seam behavior before adding buildings.
6. **Construct architecture and vegetation.** Use the castle plan and named tower forms, then parameterize uncertain dimensions. Generate village buildings and trees from explicit deterministic rules. Preserve map-derived placement independently of your geometry technique.
7. **Render and validate the composition.** Check castle/lake/forest relationships, hut and station placement, village orientation, tower silhouettes and terrain seams. Compare high and low views, day/night and the castle/tree sequence near 60-65 tour seconds. These checks do not prove the chosen real-world location.
8. **Make the next pipeline reproducible.** Save download manifests, classifications, annotations, transforms, intermediate schemas and build commands. The original project did not fully preserve this boundary; a rewrite should.

The saved final pipeline is `features.py` -> `build_final.py` -> `build_extras.py`; `build_regional.py` prepares regional layers independently; `pack.py` consumes all final layers; `assemble.py` creates the HTML. This order assumes their missing source inputs already exist. `build_local.py`, `build_terrain.py` and `build_castle.py` are earlier iterations, not extra required final stages.

`pack.py` quantizes height layers, predicts neighboring values, zigzag-encodes residuals and compresses byte planes with zlib. It packs masks/classes similarly and imagery as JPEG. The browser reverses these encodings. Those are storage choices: a rewrite can keep ordinary arrays or explicit geometry instead, provided it preserves coordinates, units and categorical sampling.

## Reproducibility limits

**Supported:** rebuild the existing standalone page from tracked sources and `build/data.js`, without GIS dependencies or original downloads. This was checked byte-for-byte against the accepted HTML.

**Not currently self-contained:** regenerate `build/data.js` from original maps and geography. Data scripts need NumPy, SciPy, Pillow and pyproj plus absent source files/intermediates. About 500 MB of original downloaded/prepared data was kept outside Git during development. Missing preparation includes:

- Download manifests and DEM/Sentinel-2 mosaic commands (`dem/mosaic.npy`, `s2tiles/mosaic.png`).
- Map palette classifications (`build/cls_*.npy`) and manual castle footprint/named-point extraction.
- The grounds lake mask (`lake_mask_grounds.npy`).
- The coarse imagery layer (`build/S_B.npy`) and regional plantation masks (`build/SF_M.npy`, `build/SF_A.npy`).

The repository's `georef.json`, `feature_offsets.json`, saved scripts and packed metadata preserve substantial reconstruction information, but not every one-shot decision. The registration values are recorded measurements/choices, not a reproducible fit procedure with residuals. No complete location-search proof, uncertainty model or automated source-acquisition pipeline is claimed.

The legacy `shoot.py`/`bench.py` provide developer diagnostics with Playwright/Edge; they are not required to view, assemble or deploy the site. Performance observations on one GPU should not be treated as guarantees on another.
