> Created/edited by GitHub Copilot; pending human review.

# Rebuilding and reconstructing the Hogwarts flyover

This guide records the scene's construction so a rewrite can preserve its reasoning without preserving its original renderer. For the live site, controls and source credits, see the [README](README.md).

The scene visualizes **White Hound's interpretation**, not a definitive location or architectural plan. Fictional geography comes only from [*The Map of Hogwarts* and related pages](https://web.archive.org/web/20261008141630/http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts.htm); geographic context comes from elevation, satellite imagery and OpenStreetMap data. It does not use the films' castle or theme-park layout.

This guide describes the accepted WebGL version based on commit [`9153cc9`](https://github.com/Arithmomaniac/hogwarts-flyover/commit/9153cc939f10411d596a7fe44429381d20cc5306), not the local experimental WebGPU work. Assembling the existing page is supported; regenerating its source data is not yet self-contained.

## Contents

- [Assembling the existing page](#assembling-the-existing-page)
- [Developing the meshes: environment, stages and lanes](#developing-the-meshes-environment-stages-and-lanes)
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

This assembles the tracked GLSL, templates and already-packed [`build/data.js`](build/data.js) and [`build/meshes.js`](build/meshes.js). It does **not** regenerate the geographic/layout data. A local experimental checkout may have a different assembler; the public source and deployed page use WebGL.

GitHub Pages publishes only this HTML, renamed to `index.html` in a temporary hosting artifact. The [deployment workflow](.github/workflows/pages.yml) does not expose the source directory, data-preparation inputs or local experiment reports as website files.

## Developing the meshes: environment, stages and lanes

This section describes the current tooling for the exterior meshes: the pinned environment, the inputs kept outside Git, the pipeline stages and their ledger, the test suites, the machine-wide locks, Blender for the families modelled in it, and the recipes for lane milestones and trunk merges. Lanes are parallel branches that each add or revise one asset family; the trunk is the branch they start from and merge back into.

### Environment

The tools run in a virtual environment per worktree on Python 3.10. It is built from [`requirements.lock.txt`](requirements.lock.txt), which pins every package by hash for Windows x86-64 and is compiled from [`requirements.in`](requirements.in); no `uv.lock` is kept. Install only through the machine's package proxies, never from the public registries:

```powershell
uv venv .venv --python 3.10 --python-preference only-system
uv pip sync requirements.lock.txt --python .venv\Scripts\python.exe
.venv\Scripts\python.exe -X utf8 -m tools.glslang      # Khronos glslang 16.6.0 from its GitHub release, checked by SHA256
.venv\Scripts\python.exe -X utf8 -m tools.blender.fetch   # Blender 5.2.3 LTS from blender.org, checked by SHA256; only for families modelled in Blender
.venv\Scripts\python.exe -X utf8 -m tools.site.sources fetch   # the site's geographic sources, checked against their pins
.venv\Scripts\python.exe -X utf8 -m tools.site.build x         # site X's data, and S0's input data-before-railway-x
```

To change a pin, edit `requirements.in` and recompile with `uv pip compile requirements.in --python-platform x86_64-pc-windows-msvc --python-version 3.10 --generate-hashes -o requirements.lock.txt`. Run everything from the repository root with `.venv\Scripts\python.exe -X utf8` (doit is `.venv\Scripts\doit.exe`).

### Vendored code

[`src/vendor/meshoptimizer`](src/vendor/meshoptimizer) holds `meshopt_decoder.cjs`, `meshopt_encoder.js`, `meshopt_simplifier.js` and `LICENSE.md` from the meshoptimizer 1.3.0 npm package (MIT), byte for byte. [`src/meshopt.py`](src/meshopt.py) pins their SHA256 and refuses a file that differs. It runs the files' embedded WebAssembly under `wasmtime`, so the build needs no Node. The simplifier runs at build time only, to make levels of detail (below); the page ships the decoder alone. To refresh them, download the package tarball through the npm proxy (`https://packagefeedproxy.microsoft.io/npm/meshoptimizer` lists each version's tarball and its sha512 integrity), check the sha512, copy the four files unchanged and update `meshopt.PINNED`.

### Mesh payload format

`build/meshes.js` defines `MESHES`, format `hogwarts-flyover/meshes` **version 4**. Its chunk records give each chunk's asset, box, vertex and triangle counts, pass, level-of-detail group and level, and its narrow parts' runs (below). Four streams run through the chunks in order:

| Stream | Raw content | Stored as (then zlib, then base64) |
| --- | --- | --- |
| `pos` | 3 × u16 per vertex, within the chunk's box | meshoptimizer vertex codec, as 8-byte vertices whose fourth u16 is the vertex's thin radius (below) |
| `nrm` | 2 × int8 octahedral normal per vertex | raw: the vertex codec makes it larger |
| `attr` | (material, aux) bytes per vertex | raw, for the same reason |
| `idx` | 3 × u16 chunk-local indices per triangle | meshoptimizer index-sequence codec. It never rotates a triangle, which the triangle codec does, so every buffer decodes exactly as the encoder wrote it |

`build_meshes.decode_streams` and the page's `meshStreams` both decode with meshoptimizer's own decoder module and give back the raw streams byte for byte, the thin radii as a fifth, `thin`. The vertex codec cannot vouch for its vertex count, so the raw `nrm` and `attr` lengths pin it before the positions decode. A payload of another version is rejected rather than read: the page ships with its own payload, so no other version ever reaches its reader. S1's family parts keep the raw streams, and the codecs run only when S2 merges them.

Thin set pieces keep a readable silhouette at any distance (`src/mesh_thin.py`). A part narrower than an internal pixel is drawn only by the jittered frames whose samples land on it, and the temporal filter's neighbourhood clamp then fades it: the goals' 0.18 m poles all but vanished from 250 m. `mesh_thin.POLICY` lists the (asset, material) pairs whose round parts widen: the pitch's gold, the goals' poles and hoops. Each vertex of such a part carries the radius of the part it lies on, measured from its own normals, in u16 steps of 0.1 mm (0 for every other vertex). The mesh vertex shader widens the part along its normals to at least `THIN_PX` (0.55) pixels of radius in the view drawing it: the depth pre-pass and the G-buffer pass with the same value, so their positions stay invariant, and the half-resolution mirror in its own pixels. A 10-sided tube then still covers a pixel centre in every row. Only round parts can widen this way: flat faces pulled apart along their normals would open at their edges, and a square bar seen face-on would not widen at all, so making a flat-faced part round (the ship's square rigging, say) is a change to its asset. Debug bit 4096 keeps thin parts at their true size.

Thin casters' sun shadows stay whole too. The quarter-resolution shadow pass marches each 4 × 4 block's shadow from its centre pixel, so a goal pole's shadow, a line a pixel or two wide from its close pass, would show only where it crossed a block's centre, as a row of dots. The shading pass therefore looks the meshes' shadow layers up for each pixel itself, from that pixel's own shadow start. A part narrower than a layer's texels also covers their centres only here and there along its length. So each chunk the shadow layers draw (an opaque full level or shadow caster) leads with its narrow parts' triangles, narrowest first, and its `narrow` runs give their counts and widths in whole millimetres (`mesh_thin.widths` and `narrow_first`). A part is narrow when it is long and thin. It is either a stick at most 0.5 m across its principal axis and 1.5 times that long (a pole, a bar, a rail, a stay, a fence board, a bracket), its width its least extent across, or a round tube found from its normals (a hoop, a twig), its width twice its median radius. A part with a parallel one alongside it closer than 0.2 m (two near-layer texels) is in a lattice, as the gates' pickets 0.135 m apart are, and is left out: lines a texel wide would merge the lattice's shadow into a solid one, while the layers' own coverage of its parts keeps it about as dark as the lattice is on average. A layer whose texels are wider than a part draws the part's triangle edges again as lines (`narrowLines` in [`src/template_b.html`](src/template_b.html)), which rasterize without a break. In the near layer (9 cm texels) these are the glazing bars, fences, railings, the gates' frames, brackets and twigs; in the static layers (0.22–1.31 m) every narrow part, the goals' poles and hoops, the lamp posts and the ship's stays included. A static layer's line casts a soft shadow about a texel wide and at most half dark, wider than the part's own, and parallel parts close together in it can merge theirs ([Known limitations](#known-limitations)); the near layer's are as fine as its texels. Toggled within one page, looking the layers up for each pixel left a frame's GPU time unchanged within 0.5% on both GPUs. The lines cost at most 0.02% of a frame as the tour flies, and of a frame in which the near layer is drawn again at most 0.44% on AMD and 0.27% on NVIDIA, at the greenhouses' glazing.

Levels of detail ([`src/mesh_levels.py`](src/mesh_levels.py)) cut the triangles the mesh passes draw, which bound their cost. The payload's `groups` give each group's box (its full level's) and the switch distances of its levels 1 and 2 (null where it has none), and its `levels` name the reference view those are measured for: 1080 output rows at a 50° field of view. A chunk's `level` is 0 (the full level), 1 or 2 (coarser) or 3 (the shadow caster), and every group has a full level. The village and the Willow use their lanes' own levels (`mesh_village.levels`, `mesh_willow.levels`, shadow caster included). Each village component's asset, each small structure (a lamp, a headstone, a length of the kirkyard's wall, a garden's wall) and the Willow is a unit, and all of a unit's levels join the group of its full level's pass and tile, so a view never draws a unit twice or drops it at a switch, and a group spans at most a tile and one structure. A small structure's coarser pieces are those whose box centres lie nearest its full piece's and within its box. A piece the lanes' levels leave out, such as the generic massing of a footprint the village doesn't build, stands in with its full level at every level its group has, the shadow caster included; a lane's own empty level still drops a piece. Every other family's opaque chunks are groups whose levels 1 and 2 come from meshoptimizer's simplifier, within 0.15 m and 0.6 m. It welds them per (material, aux) and locks their open borders, so abutting pieces never part; normals are rebuilt within 40°, and thin parts are copied unchanged. Glazing has its full level only. A coarser level is kept only with at most 85% of the triangles of the level before it, and only from its switch distance: the nearest of 50 m × √2ᵏ, up to 12.8 km, from which, and from every one beyond, it passes two guards against the full level, on cells of one reference pixel. Its silhouette from the east, the north and above may differ by at most half a pixel of mean boundary displacement. Its surface may lie at most 0.75 pixel from the full level's: by the simplifier's own error for the levels it makes, and for the lanes' levels by a sampled two-sided distance at the 99.5th percentile, which catches flattened openings and dropped fittings that leave the silhouette alone.

The page scales each switch distance by its view's output rows and field of view, never by the internal scale, so adaptive scaling cannot change a level. Each view keeps its own pick per group: the main view, which the depth pre-pass and the G-buffer pass share, and the mirror at half resolution. A group goes coarser only 1.2 times past a switch distance and finer again once nearer than it, so a camera waiting at one never flickers. The shadow layers draw each group's caster where it has one, otherwise its full level. A coarser level may flatten a slab thinner than its budget, such as a wall's coping, into twin faces at one depth, facing opposite ways; the opaque passes cull no back faces, so both mesh fragment shaders turn each normal to the side their view sees, as the glazing pass does, and the twins shade as the slab's near side. Everything that reads the payload's geometry as the families built it reads level 0: the footprint marks, and the family tests through `assets_testing.payload` (`assets_testing.levelled` returns every level). The levels take the payload from 4.2 to 7.8 MB, because the simplifier's irregular positions compress to 5–9 bytes a triangle against the full levels' 3–5.

The payload also carries `footprint`, the footprint layer (`src/mesh_footprint.py`): a u8 layer on the local grid in `DATA.tex`'s own encoding, holding each texel's signed distance (128 + 10 × metres) to the ground the standing meshes cover. While a payload is loaded, the page reads it in place of the data's building distances (`locA`'s alpha) and draws its tree table again, so trees and micro-relief follow the meshes. A payload without one, such as a test's diagnostic payload, brings the data's own distances back. `mesh_footprint.POLICY` says how each manifest asset marks the ground: by its silhouette seen from above (`BUILT`), by its convex hull (`ENCLOSURE`, the hut garden), or not at all (linear features, the train and the ship, the Willow and the street furniture). Silhouettes come from the triangles exactly as the payload holds them. S1 records each asset's silhouette in its part, at 0.5 m; S2 merges them and stores in each texel the least distance over the texel's own cell. That way the page's bilinear lookup reads below the 0.5 m where trees begin at every point under a mesh, however narrow. S2's log names the tree cover the layer clears beside each asset.

### Inputs and shared state

State shared by every worktree on the machine lives outside Git, in `%LOCALAPPDATA%\hogwarts-flyover` (`HOG_STATE_DIR` overrides it):

| Path | Holds |
| --- | --- |
| `inputs.json` | This machine's copies of the external inputs |
| `cache\whitehound\` | White Hound's pages and images. The repository is public, so they are fetched (`python -X utf8 -m tools.whcache fetch`) rather than tracked |
| `cache\tools\` | glslang, and Blender for the families modelled in it |
| `cache\parts\` | The family payload parts that stage S1 builds, by key |
| `cache\verified.json` | Hashes of verified files, by path, size and modification time |
| `ledger.jsonl` | Every stage verdict |
| `locks\` | The GPU and quiet locks, and the heavy work now running |

[`tools/inputs.lock.json`](tools/inputs.lock.json) pins the external inputs by SHA256. `python -X utf8 -m tools.inputs status` shows where each one is and whether it verifies, and `python -X utf8 -m tools.inputs set NAME PATH` records this machine's copy. A stage whose input is missing or changed is **blocked**.

| Input | What it is | Needed by |
| --- | --- | --- |
| `data-before-railway-x` | Site X's `build/data.js` from before the railway regrade | S0 |
| `data-before-railway` | Scardroy's `build/data.js` from before the railway regrade, from the original build | No stage since the move to X |
| `baseline-s4-idle` | The frozen baseline page at Scardroy, from which `baseline-x` is made | `tools.site.baseline`, the site tests |
| `baseline-x` | The frozen baseline page re-frozen at site X: its renderer, with the castle's and the fog's levels made relative to the lake and the page's colours for the surface classes X's data adds, X's `build/data.js` and the page's tour | S6 and `passes` (timing); parity at X |
| `whitehound-mirror` | A local copy of White Hound's site; the text extracts and image index exist nowhere else | The manifest |
| `original-intermediates` | The original build's intermediates | `tools.site.build`, for the layers that do not depend on the site |
| `site-*` | The geographic sources for site X: elevation models, the OSTN15 grid, OpenStreetMap extracts and Sentinel-2 tiles | `tools.site.build` |

S0 starts from `data-before-railway-x`. Unlike the other inputs, it is built in each worktree rather than shared: `python -X utf8 -m tools.site.build x` stages X's pinned sources, runs the data pipeline and writes the file to the worktree's `build/inputs/`, where `tools.inputs` finds it, then checks it against its pin. It takes a minute or two once `python -X utf8 -m tools.site.sources fetch` has fetched the sources. Until it has run, S0 reports blocked. Don't record the file in `inputs.json`: that file is shared by every worktree, and X's pin changes whenever its data does.

`baseline-x` is made the same way, not fetched. `python -X utf8 -m tools.site.baseline` takes `baseline-s4-idle`, makes its castle's window band, shading ramp and fog relative to the lake (as b46e680 made the page's own), gives its `classCol` the page's colours (from `src/frag4.glsl`) for the surface classes it lacks, the overgrown garden X's data adds, keeping its own for the rest, swaps in the worktree's `build/data.js` and the page's tour (its waypoints through the path the camera flies), writes the result to `build/inputs/` and checks it against its pin. S6 and `passes` set the two pages at the same tour times, so they are blocked while the baseline carries other data or flies another tour than the page. A change to the tour also moves the probe's windows (`mesh_probe.WINDOWS`, tour start times): `python -X utf8 -m tools.timing.windows OLD_PAGE` prints them for the worktree's page, each where the new tour passes its old place. Run it while `WINDOWS` still holds `OLD_PAGE`'s times; it fails, printing no windows, when the new tour passes more than 10 m from a window's place. Then move that window by hand to the heaviest view of its subject in the same stretch of the tour, measured with the framing test's grid, and say how in the comment above `WINDOWS`. `python -X utf8 -m tools.site.parity OUT_DIR` then compares the page, meshes off, with it at five tour times and four suns, under the GPU lock (`--gpu amd` or `nvidia`), and writes an image of each view that differs; since the meshes replaced the raymarched objects, the baseline's objects, with their shadows and reflections, differ by design. Make and pin it again whenever X's data or the tour changes.

### Stages

[`dodo.py`](dodo.py) runs the pipeline stages with doit; [`tools/stages.py`](tools/stages.py) defines them.

| Task | Stage | What it does | Writes |
| --- | --- | --- | --- |
| `manifest` | | Generates the asset manifest from [`tools/manifest`](tools/manifest) and checks its quotations | `src/mesh_scene.json` |
| `s0` | S0 terrain | Applies `fix_berth.py`, `fix_harbour.py` and `fix_railway.py` to the pinned pre-railway `data.js` | `build/data.js` |
| `s1` | S1 family meshes | Builds each family's payload part, in parallel | Parts in the shared cache |
| `s2` | S2 page | Merges the parts, then assembles the page | `build/meshes.js`, `hogwarts-flyover.html` |
| `stale` | Gate | Checks that HEAD's generated files equal a rebuild of HEAD's sources; fails while any other file differs from HEAD or is untracked | |
| `uncached` | Trunk gate | Rebuilds the payload from the sources without the part cache and checks it equals S2's, which would catch a part key that misses an input; about 3.5 min, so only the trunk keeper runs it, at merges | Logs |
| `s3:SUITE` | S3 CPU gates | Asset tests, also on a world raised and lowered by 140 m (site X raises the lake 131.0 m and the castle platform 131.6 m); page checks (material registry, assembly, glslang); site checks; tool tests | Logs |
| `s4:SUITE:GPU` | S4 GPU correctness | Rendering and UI tests | Logs |
| `smoke:GPU` | S4 subset | The rendering tests tagged `@smoke` | Logs |
| `s5:FAMILY` | S5 review sheets | A labelled sheet of the family's views, rendered on NVIDIA | `build/review/FAMILY-PAGEHASH.png` |
| `s5:tour` | S5 review sheets | The tour's contact sheet: a still at each waypoint, as the flyover passes it, with the meshes on (`tools/review/tour.py`) | `build/review/tour-PAGEHASH.png` |
| `s6:GPU`, `passes:GPU` | S6 timing | The frame-time probe against the frozen baseline, or GPU time per pass, alone on the machine | `build/logs/timing-GPU-PAGEHASH.json` |
| `ledger` | | The latest verdict of each stage in this worktree | |

```powershell
.venv\Scripts\doit.exe s2               # rebuild what is out of date: about 2.5 min from scratch (mostly the village's level-of-detail guards), 1 to 2 s when nothing changed
.venv\Scripts\doit.exe -n 3 s3          # the CPU gates (the default task): about 1 min
.venv\Scripts\doit.exe stale            # the staleness gate, on a committed tree
.venv\Scripts\doit.exe -n 4 s4          # rendering and UI tests, one worker per suite: about 12 min on a quiet machine
.venv\Scripts\doit.exe smoke:nvidia     # the smoke subset
.venv\Scripts\doit.exe ledger
```

A stage's key hashes what it reads:

- its code: the import closure of its scripts, plus the modules loaded by name (material families, family builders, manifest families);
- its data files, and the pinned inputs by their verified hashes;
- for a family's S1 part, also every file under `src/assets/FAMILY/`, so committing a new export rebuilds that family's part and the page, and the staleness gate checks them; for a part on the asset tests' shifted world, also the shift and the code that applies it;
- the locked package versions and the Python version;
- for browser stages, also the page, the GPU adapter and driver, the Edge version, `HOG_GPU` and Edge's launch flags.

A stage is skipped when its key has a passing verdict in the ledger and the files it wrote still hold what that verdict recorded; a part built in one worktree serves every other. Editing a test file reruns S4, for example, but editing this guide does not. S0's key comes from the pinned pre-railway input, never from `build/data.js`, which `fix_railway.py` rewrites. Logs go to `build/logs`, named after the page hash or the stage key.

A verdict is **pass**, **fail**, **error** or **blocked**. Blocked means the stage could not run, for example because an input, glslang, a GPU or the White Hound files are missing; it is never a pass. Only passes are reused. Fail, error and blocked verdicts are recorded and run again next time.

### Test suites

`python -X utf8 -m tools.testrun SUITE [--gpu amd|nvidia] [--shift METRES] [-k PATTERN]` runs one suite with the same locks and logs as the stages:

| Suite | Tests | Runs on |
| --- | --- | --- |
| `assets` | `src/test_assets_*.py`, with shared fixtures in `src/assets_testing.py`, and trimesh as an independent oracle on the decoded payload (winding against stored normals, triangles that quantization collapses) | CPU |
| `uncached` | `src/test_uncached_build.py`: the payload rebuilt without the part cache equals S2's | CPU |
| `checks` | The material registry, the assembly, and glslang on every program in both float-filtering variants | CPU |
| `site` | `src/test_site.py`: the site's sources and pins, staging and building, the elevation models, and the packed site's earthworks. It needs the site's fetched sources (`python -X utf8 -m tools.site.sources fetch`) | CPU |
| `tools` | The runner, the locks and the stages (`tools/test_*.py`) | CPU |
| `render` | `src/test_render_*.py`: one browser and page per class, reset between tests | `--gpu` |
| `smoke` | The `@smoke` rendering tests | `--gpu` |
| `ui`, `probe` | `src/test_ui.py`, `src/test_mesh_probe.py` | `--gpu` |

Each test ends as pass, fail, error, expected-fail, skipped or blocked, and its time is logged. A known failure is listed by its exact `Class.method` ID in its module's `EXPECTED_FAILURES`. None is listed today: the last, `test_assets_grounds.PitchTests.test_the_decks_stand_clear_of_the_ground`, has passed since the site data build (`src/build_final.py`, run by `tools.site.build`) graded the ground under the south-east stand at site X; S0 starts from its output. Each family's test module also pins the faults trimesh finds in its decoded meshes, as `MESH_FAULTS` checked against `assets_testing.mesh_faults(FAMILY)`, so a new fault fails and so does a fix until the pin is updated. Today every pin is empty: no face winds against its stored normals, and the encoder drops every triangle that is degenerate before or after its 16-bit quantization. The gate compares exact IDs, not counts: a listed test fails the gate if it passes, is skipped or does not run, and a class setup error is an error even when the class's tests are all listed. With `-k` or the smoke tag, only the selected tests' listings apply. `HOG_GPU=amd|nvidia` pins Edge's adapter; when the page reports a different one, the tests are blocked.

A suite run without `-k` resumes where an earlier one stopped. Each class that passes (every test a pass, a skip or a listed failure, its class fixtures clean) is recorded as soon as its `tearDownClass` has run, or, in a module with a `tearDownModule` or module cleanups, once those have run clean, with its tests' outcomes and times and its part of the transcript, under the suite's input key: the key of its S3 or S4 stage, which also tells the GPUs apart (`build/logs/testrun/`). A run under the same key runs only the classes without a record, so a pause or a crash loses only the class in progress (in a module with a module-level teardown, the module's classes run so far), and its log and report still cover every test, the recorded classes' parts first. A class that fails, errors or is blocked is never recorded and always runs again, as does one whose tests all skip; any changed input changes the key and reruns every class. A `-k` run neither skips nor records classes.

Measure heights from the lake or the castle platform (`assets_testing.levels()`), from `assets_testing.datum()` for synthetic fixtures, or from the sampled ground, never as absolute levels. `test_assets_datum` fails on any float literal within 8 m of the registered lake or platform level. `HOG_DATUM_SHIFT_M=METRES` (or `--shift`) reruns the asset tests on a raised world, which `src/datum_shift.py` raises. Its payload is merged from family parts built on that world (`tools/shifted_part.py`), which S1 caches keyed on the shift, so a rerun reuses every part whose sources have not changed (`s1:FAMILY+140m` in the ledger). A part that fails to build on the raised world makes the run an error.

A still view stops rendering once it has accumulated 96 frames (`IDLE_N` in `template_c.html`), while `__hog.frames` keeps counting. A rendering test that needs several fresh frames, for example one per jitter phase, should gather them inside the page with `RenderTestCase.each_frame` (or `NEXT_FRAME` after `REARM`, from `src/render_testing.py`). It should not wait on `__hog.frames` across round trips: on a busy machine those can outlast the 96 frames, and the test then probes one frame repeatedly or waits for a frame that never comes.

### Locks and timing

- A browser suite holds its GPU's machine-wide lock, so a second run on the same GPU waits. AMD and NVIDIA correctness suites may run at the same time. Run S4 with one doit worker per suite (`doit -n 4 s4`): with fewer, the workers can all be waiting on one GPU's lock, held by another worktree or by this run's other suite, while the other GPU sits idle.
- Builds (the site's data build, `tools.site.build`, included), suites, mutation checks (`tools.mutate.run`, `tools.site.mutate`), terrain comparisons (`tools.site.compare`) and Blender runs register as heavy work while they run; a browser suite or tool registers once it holds its GPU. `tools.mutate.run` registers each mutation on its own, so a timing run can start between two. A mutation whose tests drive a browser (a GPU suite's tests: rendering, UI, the probe) is judged on the GPU that `HOG_GPU` names, holding that GPU's lock through its copy, rebuild and tests; without a valid `HOG_GPU`, or when its tests skip as blocked because Edge renders on another GPU, it reports BLOCKED. Timing (`s6`, `passes`) holds the quiet lock, which waits until no heavy work runs and keeps new heavy work waiting until it ends.
- Register any other heavy command by running it as `python -X utf8 -m tools.locks heavy -- COMMAND ...`. Check `status` first: unregistered work is invisible to the quiet lock, and a timing run would measure it.
- Run a browser script of your own (a diagnosis, a look check) as `python -X utf8 -m tools.locks gpu amd|nvidia -- COMMAND ...`. It takes that GPU's lock and then registers as heavy work, as the next point describes, and sets `HOG_GPU` for the command, so it waits for a suite on that GPU and the suite waits for it, instead of the two slowing each other until frame waits time out. A script that starts Edge itself must pass that GPU's flags (`mesh_probe.GPU_ARGS`). A tool that takes the same GPU's lock itself (`tools.testrun`, `tools.review.sheet`, `tools.timing.windows`, `tools.site.parity`) runs under it, holding the lock through its parent.
- Every GPU tool takes its GPU's lock before it registers as heavy work (`locks.gpu_work`), so waiting for a GPU holds up no timing run. If the quiet lock stays held for more than 2 s once the GPU comes free (a timing run holds it for minutes, a registration for a moment), the tool lets the GPU go, waits for the timing run to end and starts again. It never holds a GPU while it waits for a timing run: work that registered and then waited for that GPU (a worktree that has not merged this order yet, or a GPU tool inside registered work) would wait for it, and the timing run for that work. Don't run a GPU tool inside registered work (under `heavy --`, or from a script inside `locks.heavy`): a timing run would wait for that work while it waits for a GPU. Don't run one under `gpu --` for the other GPU either: two such commands, crossed, would wait for each other's GPU.
- Work started inside registered work never waits for the quiet lock: that would deadlock with a timing run that holds the lock while it waits for the enclosing work. One registration covers all of a process's threads (doit runs stages in threads) until the last of their blocks ends. A child process registers its own entry at once, so it stays counted if the enclosing work ends first.
- Run frame-time probes only in a declared quiet window, one GPU at a time. `src/mesh_probe.py` and `tools.timing.passes` refuse to run without the quiet lock's token. Run them through doit, or by hand as `python -X utf8 -m tools.locks quiet -- .venv\Scripts\python.exe -X utf8 src\mesh_probe.py --gpu amd --baseline PAGE --out JSON`.
- On the AMD iGPU a second live WebGL context runs slower than the first, whatever its page: 15–20% when the probe loaded the baseline twice; 8.5–20.7% by window in integration's per-pass A/A test, P1 included; and in the trunk's A/A test of November 2026, 11–20% by window with both copies in one browser and 13–34% with each in its own browser, so a second browser does not avoid it; on NVIDIA the later copy ran 0–1.4% faster. One page alive at a time, one tab navigating between the copies, came out even (0.96–1.03), but the block after each navigation ran about 4% slower, and at one window both navigated copies ran about 20% above a freshly opened page. A second A/A test set a page's later visits against its first, freshly opened one: through navigation they ran 1.115 times as slow, through a new tab 1.23, and through a fresh browser for every visit 1.025, so anything that keeps the browser's GPU process leaves a residue. The probe can therefore open a fresh browser for every visit to a page (its fresh layouts, on trial for AMD); by default it keeps both pages open in each browser session (pages). Both alternate the two load orders (`mesh_probe.LOAD_ORDERS`) between sessions, as does `passes --mode pages`, and combine them by geometric mean, in which an order effect cancels only when both pages share it. Never compare two pages timed in one load order.
- S6 runs each GPU's layout (`mesh_probe.LAYOUTS`, both `pages` for now) with the probe's defaults: 16 rounds in four browser sessions, two per load order, each session running every order of the probe's Williams square once. The fresh layouts (`--layout fresh` or `fresh-window`) default to 32 rounds in eight sessions of four rounds, four sessions per load order: each round visits the two pages in its session's order, each visit in a fresh browser that times every window (`fresh`) or one (`fresh-window`) after a warm-up block, the candidate's configurations in the order the round's row of the Williams square gives them. Two A/A sessions of four rounds visit the candidate twice; their second visits over their first, pooled over the windows, check for a gross error in fresh visits themselves, as the control does for the measurement: every window is inconclusive (`absolute`) only when the A/A lies more than 2% from 1 throughout its interval. The bounds already hold any effect of visiting a page second that moves both pages the same way, so the verdict does not need the A/A to show the visits alike. In a validation run on AMD at T17 (16 pair and 8 A/A rounds, 44 min) the `fresh` layout's order effects (0.94–1.03) were consistent with 1 but too noisy, ±13–22%, to show its bounds meet: with a window's two pages a minute apart its paired ratios scattered 17–25% a round, against 4–9% with both pages open, so the bounds' intervals came out ±15–29% and the A/A at 1.056 [0.890–1.208]. The probe gives each ratio a 90% bootstrap interval, drawing n − 1 of each load order's n sessions again and then their rounds. Opening a page second changes its frame time by a factor of its own, slower on AMD and up to 1.4% faster on NVIDIA. So long as both pages move the same way, a ratio between them lies between its values in the sessions that open the baseline second and in those that open the candidate second, however unequally the two pages are moved. Each window passes when both of these bounds' intervals for integrated/baseline lie within the plan's 1.10, fails when both lie above it, and is otherwise inconclusive. A check also leaves a window inconclusive: fewer than two sessions per load order, or a session that skips an order; or a control (the integrated scene timed against itself) beyond 2% throughout its interval. The order effect, the square root of the bound from the sessions that open the candidate second over the other, shows how far apart the two pages' factors put the bounds: within 1.4% of 1 on NVIDIA, so its bounds nearly meet. With both pages open on AMD it was 1.02–1.34 by window and launch, which left most AMD windows inconclusive however many rounds ran; the fresh layouts keep a single page alive, which should bring their bounds together, though the first validation run was too noisy to show it. A run passes only when every window does, and otherwise exits with status 1. Report an inconclusive or failing window as it is: the plan rules out measuring again until a window passes, so another run needs a new reason, such as a focused repair.
- `python -X utf8 -m tools.timing.qualify` runs the whole frame-time qualification, unattended, in one hold of the quiet lock, for a window when the machine is left alone (other sessions and apps paused). NVIDIA runs first, on two pages, so its verdicts stand even if the window ends early. After two minutes to cool, AMD runs a trial of the `fresh-window` layout over castle, trees-65 and greenhouses: 16 pair rounds in four sessions and two A/A sessions. From the trial's paired ratios the driver estimates their scatter between rounds and between sessions, and projects, by simulating the probe's own bootstrap, how wide each bound's interval comes out at that design: about 0.95 times the scatter between rounds either way, when the sessions agree, so the trial passes only when the paired ratios scatter by about 5% a round or less. If that projection is within ±5% and the trial's A/A raises no flag, AMD runs `fresh-window` at the same design over every window. Greenhouses, the window nearest the budget, is then measured again in a run of its own, whose rounds join the first run's for its verdict, unless the two runs would take more than two hours at the trial's pace (`--max-amd-hours`): then greenhouses is judged on the first run alone. Otherwise, as when the trial measured nothing, or if the first run alone would take more than two hours, AMD runs on two pages, judged by its bounds. A step that other processes' load blocks runs again after five minutes, up to three times; a step that measured never runs again. Each step's log and probe report go to a folder of the run's own, `build/logs/qualify-PAGEHASH-STARTED/`, with `summary.json` and `summary.txt`, so no run overwrites another's; `--dry-run` prints the steps. Exit status 0 means every window passed on both GPUs, 1 that one failed or stayed inconclusive (or a step crashed), 3 that load blocked NVIDIA's or AMD's run through every attempt; the trial only chooses AMD's layout.
- The probe also measures the load of processes the lock cannot see, from one snapshot of every process per sample. Security and OS processes (Defender, Global Secure Access, System, svchost) work on the probe's own browser, so they have their own cap (`--max-induced-cores`, 5); every other process outside the probe counts against 2 cores (`--max-other-cores`). The probe waits for both to fall within their caps before it starts, and measures a window again while either is exceeded. A run that cannot start, or keeps a window over a cap through every attempt, is blocked (exit status 3), never a pass. The caps are absolute because the frozen baseline runs interleaved with the candidate in every window, so a steady background within them slows both alike. The induced cap stays at 5 for qualification, low enough that a Defender scan or update should still exceed it, though that has not been tested: the first quiet window measured about 3.5 cores (Defender 2.1, Global Secure Access 0.9, System and svchost 0.5), before the probe's browser stopped writing caches, and on a quiet NVIDIA run in November 2026 the probe's windows induced 0.7–2.4 cores. The qualification's report (`tools.timing.qualify`) logs each judged window's induced and other load over its sessions. Each window's two loads are printed next to its ratios. The probe's browser reuses one profile in the machine cache and writes no disk or shader cache, so the security processes have little to scan.
- `python -X utf8 -m tools.locks status` shows who holds each lock.

### Blender

Families modelled in Blender use Blender 5.2.3 LTS. [`tools/blender/fetch.py`](tools/blender/fetch.py) downloads the official Windows build from blender.org once, checks it against its published SHA256, and unpacks it into `cache\tools\`, where every worktree finds it (about 1.3 GB with its archive). `fetch.run(SCRIPT, ARGS...)` runs a script in it headless, always with `--background --factory-startup --offline-mode --python-exit-code 1`, from the repository root, on an empty user profile and without the user's `BLENDER_*` variables. The script reads its own arguments after `--` in `sys.argv`. It runs in Blender's own Python 3.13 and its bundled packages, not in this environment.

No stage runs Blender. A lane runs its generator by hand and commits what it writes to `src/assets/FAMILY/`, which stage S1 reads. `fetch.run` and the fetch register as heavy work, so a timing run waits for them. The `tools` suite runs Blender only when it is in the cache, and its key records that, so fetching Blender reruns the suite.

### Adding a family

A lane adds a family by adding its own files; it edits no shared file:

| File | Holds |
| --- | --- |
| `tools/manifest/families/FAMILY.py` (and `tools/manifest/data/FAMILY/`) | The family's assets, decisions and image notes, appended to the manifest, and through an optional `excluded(ctx)` the mapped footprints it leaves unbuilt, each with its decision |
| `src/materials/FAMILY.py` | Its material codes, inside its reserved ranges |
| `src/mesh_FAMILY.py` | `FAMILY`, and the hooks `build_footprint`, `build_castle_pieces` and `build_extras`, which return only the family's own assets |
| `src/assets/FAMILY/` | Optional source assets that `src/mesh_FAMILY.py` reads, such as JSON, OBJ or NPZ exports from Blender, in place of a generated Python module. They are part of the family's S1 key, and Git stores them byte for byte |
| `tools/blender/FAMILY_*.py` | Optional generator scripts that write those assets in Blender, run with `tools.blender.fetch.run` (see [Blender](#blender)) |
| `src/test_assets_FAMILY.py` | Its asset tests, its pinned `MESH_FAULTS` with the test that checks them (every family with geometry needs one), and any `EXPECTED_FAILURES` |
| `src/test_render_FAMILY.py` | Its rendering tests, with `@smoke` on the cheapest |
| `tools/review/FAMILY.py` | The views and columns of its review sheet |

A family module finds its source assets at `Path(__file__).with_name("assets") / FAMILY`. Keep their coordinates local to each asset, and place the assets from the sampled ground, the lake or the castle platform, never at absolute heights: the asset gates on a world raised and lowered by 140 m, and the move to site X, depend on it.

Every new asset also needs a footprint policy in `src/mesh_footprint.py` (see [Mesh payload format](#mesh-payload-format)), a shared file the trunk keeper edits: S1 fails on an asset without one, and `test_assets_integration` checks that the table names every manifest asset. Leave an asset off the layer only for a reason a visitor would accept: a strip cleared along a wall or a track would read as a cutout.

Reserved material codes (material, aux), from `src/materials`:

| Family | Codes |
| --- | --- |
| Shared | (5, 0–15), (6, 0–15), (7, 1–9), (8, 0–15) |
| Castle | (4, 0–255) |
| Transport | (7, 10–25), (7, 100–119) |
| Grounds | (7, 26–59) |
| Village | (7, 60–99) |
| Willow | (7, 120–139) |
| Greenhouses | (7, 140–149) |

Reuse a set piece's shading body before writing a new one. Every distinct `p3` body becomes another branch of the set-piece chain in the shading pass (`materials.p3_branches`), and that program's compile time dominates the page's load time: the village's 25 new bodies took it from 4.5 s to 12.3 s on NVIDIA. So `materials.validate` refuses more than 40 distinct bodies (`MAX_P3_BRANCHES`), as a compile-time budget. A set piece that names a surface model, `model="flat"` or `model="speckled"` (`materials.MODELS`), with its own literal `values` shares that model's single branch with every set piece of every family that names it; the branch reads each code's values from constant arrays. A set piece whose `p3` text is identical to another's shares its branch too.

The chain is a run of independent branches: each sets `shaded` for its own codes, and the default set piece's body runs under `!shaded`. Through T6 it was an else-if ladder, and past a depth that depends on its bodies a ladder stops the page from starting at all. With the village's bodies before lane B gave its harls and glosses one branch each, 55 branches started on NVIDIA and 58 ran out of D3D11 memory creating the pixel shader; the trunk's bodies, padded with copies of themselves, stopped compiling past 110 branches on both GPUs (ANGLE: "Expression too complex"). Independent branches with the `shaded` flag still start at 200 on both GPUs, so the cap, not a GPU, now limits the chain. At T6's 31 bodies the change rendered the same pixels, and cost no frame time measurable on either GPU. Live frame time comes first: a chain form that costs frame time to load faster is out.

These chain forms were measured in November 2026, with the chain padded by copies of the registry's bodies; the measuring scripts are not in the repository. Link times were taken on NVIDIA under load, so compare them only within their column. Frame cost is GPU time per frame over the frame-time probe's set-piece windows, against the ladder, beside an A/A control.

| Chain form | NVIDIA link at 58 / 111 branches | Starts at | Frame cost | Status |
| --- | --- | --- | --- | --- |
| Else-if ladder | 23 s / fails (25 s at 110) | 110 on both GPUs; with the village's earlier bodies, 55 but not 58 on NVIDIA | Baseline | Replaced after T6 |
| `switch (k)`, a case label per code | 24 / 40 s | 111 on NVIDIA; fails at 155 | Not measured | Not pursued: slow to link, and its own cliff |
| Independent ifs in `do { } while (false)`, each ending in `break` | 25 / 46 s | At least 111 | Not measured | Not pursued: links as slowly as the ladder |
| A constant table from code to branch, then `switch` on the branch | 17 / 32 s | At least 111 | Not measured | Not pursued: slow to link |
| The same table, then independent ifs on the branch | 9 / 18 s | 96 on AMD (context lost at 97); fails at 200 on NVIDIA | Not measured | Not pursued: its own cliff |
| Independent ifs, the default under the negation of every code | 11 s / not measured | At least 200 on both GPUs | Shading pass 3–4% slower on NVIDIA (about 0.01 ms a frame), 1–5% on AMD | Rejected: frame cost |
| Independent ifs with the `shaded` flag | 9 / 15 s | At least 200 on both GPUs | None measurable: frame 1.000–1.006 of the ladder's on NVIDIA (A/A control 1.001–1.006), 0.92–1.05 on AMD (A/A 0.99–1.05) | Current |

### Lane milestone recipe

1. Merge the trunk into the lane: `git merge TRUNK`. Generated files are never merged by hand (`-merge` in [`.gitattributes`](.gitattributes)); on a conflict either side will do until the next step.
2. Rebuild: `doit s2`, and commit the rebuilt files with their sources.
3. Run the CPU gates: `doit -n 3 s3`, then `doit stale` on the committed tree.
4. Run the NVIDIA smoke subset, under the GPU lock: `doit smoke:nvidia`.
5. Make the family's review sheet: `doit s5:FAMILY`, and show `build/review/FAMILY-PAGEHASH.png` to the user.
6. Run the code review: the `code-review` agent at reasoning effort `max` on the lane's diff. Fix the findings it confirms.
7. Report the verdicts with their log paths (`doit ledger`).

### Trunk merges

The trunk keeper merges a lane that has already merged the trunk, rebuilds (`doit s2`), and runs `doit -n 3 s3` and, once the merge is committed, `doit stale` and `doit uncached`. It then runs `doit -n 4 s4` for the rendering and UI tests on both GPUs at once, and reports the verdicts with their log paths. The lanes check their own families, so the trunk keeper also looks at the integrated scene: it renders the tour's contact sheet with the previous merge's page beside the new one (`python -X utf8 -m tools.review.sheet tour --before PREVIOUS_PAGE`; `doit s5:tour` makes the trunk page's own), looks at each pair, and reports only breakages: missing, floating or misplaced objects, holes, and broken lighting. Frame-time probes (`doit s6:amd`, then `doit s6:nvidia`) run only in a quiet window that the orchestrator declares.

## Evidence and interpretation

Keep three kinds of information separate when rewriting:

| Kind | Examples | How to use it |
| --- | --- | --- |
| Source interpretation | White Hound's grounds maps, castle plan, relative positions and textual arguments | Preserve the source relationship and its uncertainty; do not label it definitive canon |
| Geographic measurements | Copernicus heights, real watercourses, satellite colors | Provide regional context; fictional grounds deliberately modify the local landscape |
| Implementation choices | Exact site anchor, tower heights in metres, smoothing, lake level, seeded roofs and trees | Revisit freely; these are modeling parameters, not independently established facts |

The essay itself describes its castle plan as one working possibility. Its [location](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/location.htm), [setting](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/setting.htm), [grounds](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/grounds.htm), [castle](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/castle.htm), [Hogsmeade](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/Hogsmeade.htm), [forest](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/forest.htm) and [journeys](http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts/journeys.htm) sections explain different parts of the interpretation. These are original-site links; use the archived index above to locate archived copies if the originals are unavailable.

Map assets used in preparation include `FF_map_unlabelled.gif`, `Hogwarts_map_unlabelled.gif`, `Hogsmeade_map_unlabelled.gif`, their labelled counterparts, and `castle_plan.gif`. They are source-site assets, **not files bundled in this repository**. Labels help identify landmarks; unlabelled maps reduce text contamination during mask extraction. Some remaining decorative areas were explicitly excluded in the scripts.

## Location and coordinate systems

### A chosen anchor, not a discovered castle site

The essay uses **57 degrees 30 minutes north, 5 degrees west** as "my standard reference point for purposes of calculating sunset times etc." That is not a precise, uniquely established castle position, and the essay names glens rather than the point as likely settings: Hogwarts "is probably close to either Glen Carron (best fit for the railway-line) or Glen Affric (best fit for the snow-covered mountain)", it might lie "somewhere at the end of Glenuaig", and Google Maps' placing of Hogsmeade at Glun Odhar in Strathconon Forest is "a perfectly feasible spot". This implementation places the castle at site X, **57.5420 N, 5.0760 W** (NH 1599 5437), on the shelf south of Loch Gowan at the head of Glen Carron, Ross-shire. The original build placed it near **57.5223 N, 4.9907 W**, in the River Meig glen near Scardroy, Strathconon. `SITES` in [`src/common.py`](src/common.py) keeps both, and the page names the site from its data.

Site X lies 4,550 m west and 4,675 m north of the reference point in the projection below, about 6.5 km away. Scardroy lay 560 m east and 2,480 m north, about 2.54 km away; it was a placement choice intended to fit the fictional grounds into nearby mountain scenery. The original build's location searches survive as `search1.npz` to `search4.npz` in the pinned input `original-intermediates` ([`tools/inputs.lock.json`](tools/inputs.lock.json)), but they do not establish Scardroy as the best fit:

- The horizon-aware search (`search3.npz`: anchors within 5 km of the reference point on a 60 m grid, scored on earthworks, White Hound's horizon angles, the grounds' up-and-down ordering and distance) ranked a north-west glen site at TM (−2480, 4060) first. The grid point nearest Scardroy came 229th of 21,804.
- Scardroy ranked first only in the last search (`search4.npz`: within 3.5 km on a 20 m grid), which scored earthworks, ordering, relief and distance but no horizons. The north-west glen site lay outside its range.

Scardroy was therefore adopted as the nearest good local fit, not as the best fit found. A rewrite should treat the anchor as adjustable.

The move to X followed a later location review, kept outside the repository. With its own model, on bare earth (OS Terrain 50) and at each site's best lake level, it scored X 0.685, the best of the four sites it checked, against 0.630 and 0.623 for two others and 0.572 for Scardroy. Against White Hound's horizons and the grounds' up-and-down ordering, it measured at X:

- sunsets on his two dated evenings (6 June 1994, 21 June 1996) 23 and 24 minutes early, against his "around thirty minutes early" and "about forty minutes" (54 and 56 at Scardroy);
- sunrises (31 March, 4 May 1998) 44 and 46 minutes late, against his "about forty-five minutes";
- the skyline due north at 4.3 degrees (14.7 at Scardroy);
- the Forest's floor 45 m below the castle and Aragog's hollow 152 m below it (54 m and 291 m above it at Scardroy);
- Hogsmeade on 2-degree ground, 4 m below the gates (23 m above them, on 17-degree slopes, at Scardroy).

It put about 65% confidence on X fitting White Hound better than Scardroy, and about 40% on X being the single best site. The user chose X, with the lake at 284 m and the castle on a mound. X's sunsets thus come earlier than a level horizon allows, but less early than White Hound infers: the mountains north-west of X are lower than his reasoning needs. That is recorded, not fixed (the `sunset-shortfall` decision), because his figures are inferences from the books' timelines, which give no clock time for either sunset.

The original page HUD called this the "nearest ground" fitting the horizons. That stronger wording was not a validated geographic conclusion; the current About dialog describes the anchor as a modeling choice instead.

### Geographic and world frames

[`src/common.py`](src/common.py) defines a WGS84 transverse-Mercator frame with this PROJ definition:

```text
+proj=tmerc +lat_0=57.5 +lon_0=-5 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m
```

`pyproj.Transformer(..., always_xy=True)` receives **longitude, latitude**, not latitude, longitude. Project a geographic point into `(E_tm, N_tm)`, then subtract the site's castle anchor (its `anchor` in `SITES`):

```text
E = E_tm + 4550    # site X; at Scardroy, E_tm - 560
N = N_tm - 4675    # site X; at Scardroy, N_tm - 2480
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

The elevation model is each site's `dem` in `SITES` ([`src/common.py`](src/common.py)), one line to change. Site X uses **FABDEM V1-2**, the Copernicus GLO-30 surface model with forests and buildings removed, so the renderer's trees and the meshes' buildings do not stand on canopy and roofs baked into the ground ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); Hawker et al. 2022, University of Bristol, [doi:10.5523/bris.s5hqmjcdj8yo2ibzi9b4ew3sn](https://doi.org/10.5523/bris.s5hqmjcdj8yo2ibzi9b4ew3sn)). The page credits whichever model the data was built from (`extras.dem`). The original build, at Scardroy, used Copernicus **GLO-30** itself, a digital surface model. Neither is a surveyed bare-earth model: a 2 m or 10 m output grid does not turn either into measured terrain at that resolution. [`tools.site.stage`](tools/site/stage.py) resamples the model onto GLO-30's lattice as `dem/mosaic.npy`; `dem_sample` transforms each world point back to longitude/latitude before sampling that mosaic.

The Forest-map layer is White Hound's Forest as drawn, except where a site's `forest` rules in `SITES` say otherwise ([`src/forest_extent.py`](src/forest_extent.py)). At site X his Forest, drawn over the real valley floor, ends 25 to 90 m short of the A832 and round Achnasheen. Everything there lies north of Aragog's nest, where his essay calls the Forest's content a "for instance", and the Forest keeps his scale, about three miles south to north. Where his reasoning needs Forest over a real feature, corridors are cleared instead: 12 m each side of the A890 and 10 m each side of the railway. Each building it still covers stands in its homestead's clearing rather than a cutout round its walls: its yard, 10 m round it, is clear, and the clearing spreads on beyond it, its edge wandering and the Forest thinning out over 24 m (six such homesteads clear standing Forest, 0.55 to 1.8 ha each). Minor roads and tracks stay under the Forest. [`pack.py`](src/pack.py) keeps the cleared texels clear of its smoothing. The decisions, with his words, are in [`mesh_scene.json`](src/mesh_scene.json).

At site X his Aragog's hollow (6g) lay 37 m from the Kyle railway and 77 m from Loch Gowan, where a hollow would have its floor 6 m below the loch and the railway across its rim, and his Grawp's clearing (4f) lay on a river's floodplain, 49 m from its water; neither was modelled beyond his drawing, the crater as a patch bare of trees and Grawp's spot as the pines `frag1.glsl` grows round it. The site's `aragog_hollow` and `grawp_clearing` in `SITES` move them into dry Forest nearby ([`forest_places.py`](src/forest_places.py)). `pack.py` cuts the hollow into the 10 m regional ground, once the imagery's hill shading has been taken out against the natural ground: 100 m across at its rim, its level floor 10 m below the lowest ground on its rim and 11 m above the loch, and its sides rising at 45 degrees (0.095 Mm³). Its clearing, 1.69 ha, takes the Forest off it and out beyond its rim, and his crater's patch, 1.58 ha with the edge the layer's sampling blurs, takes the Forest round it. Grawp's clearing, 0.36 ha, keeps six small pines. Both are recorded clearings, each with its decision.

The local builder smooths the DEM, adds selected hillocks and the Shack's hill, and raises the castle platform to at least 36 m above the lake: `153 + 36 = 189 m` at Scardroy, `284 + 36 = 320 m` at X, where the castle's base carries it. That models the essay's roughly 120-foot lakeside cliff; its exact platform profile is an implementation choice. The pitch, building pads and station corridor are flattened; roads receive additional smoothing.

At site X the ground lies 54 to 80 m below the platform (`284 + 36 = 320 m`), so [`castle_base.py`](src/castle_base.py) builds the castle's own base, the site's `base` in `SITES` (`"mound"`; `"crag"` is the one-line alternative). Lawns fall from a skirt round the castle, 1 in 4 towards Hagrid's cabin and the pitch and 1 in 2.6 elsewhere. Towards the lake the base drops as the essay's cliff: a foot-path slope below the walls, then a near-vertical face to the water, with the 2e wall at its lip. The vegetable patch and greenhouses lie on a level terrace below the castle, in a 0.25 m hollow that floods in heavy rain, as the books have it; the greenhouses, their pads and the paths stand dry. Where the base buries the lake's outflow, the outflow runs under it and emerges at its foot. The decisions, with White Hound's words, are in [`mesh_scene.json`](src/mesh_scene.json).

At site X, Hagrid's cabin also stood in a hollow 24 m below the gates and 5 m below the lake, against White Hound's "The front gates are lower down than Hagrid's cabin". The site's `cabin_terrace` in `SITES` therefore raises the hollow ([`grounds_grade.py`](src/grounds_grade.py)). A terrace stands 4 m above the gates' own ground: level within 150 m of the cabin, then falling 1 in 3. The drive climbs to it from the gates on an embankment and runs on at its level until the ground rises higher, and a lawn 40 m wide stays at that level as far as the castle mound, so the lawns fall from the castle's front to the cabin, as the books walk them, with no hollow below it. The drive's line is the cheapest path along the essay map's roads from the gap in the boundary wall to the Entrance Hall. That comes to 2.7 Mm³ of fill, with nothing cut. Where the drive passes 22 to 31 m from the lake, between the cabin and the castle, it still dips to 9.5 m below the gates, where the shore keeps its walkable bank. The user kept that dip: the gates stay "the bottom of the drive" as its far end from the castle, and the dip hides them until the cabin's terrace, where they "came into view" (HBP 25). The Whomping Willow stood in the same low ground, 10 m below the lake, its crown hidden from every greenhouse. The site's `willow_knoll` raises it on a knoll of its own, 24 m high with flanks of 1 in 2.6 (0.5 Mm³), so that from the northern greenhouses its crown clears the ground while a person at its foot stays hidden, as White Hound has it. Its `willow_lawn` opens a lawn from those greenhouses to it, 2.4 ha, with no tree over any of their lines to its crown nor within 20 m of it, where the light woods his map draws beside them stood over three of those lines. Round the greenhouses the lawn takes in the vegetable patch, whose beds and hollow stay as they were. From each greenhouse the trees now hide none of the crown that the ground leaves in sight.

Beyond the grounds, the trees White Hound's Hogsmeade map draws on the Shrieking Shack's hill stood in front of its one unblocked window, glazed on its ground floor, so that only 38 of 150 points on the castle's upper parts showed past them. In DH 32 "The single unblocked window revealed distant bursts of light where the castle stood", so the site's `shack_garden` in `SITES` keeps the window's view of the castle's breadth clear of trees for 60 m beyond the Shack's wall, inside its "dank overgrown garden" (PoA 14): 0.54 ha of rank grass, nettles and bramble, a surface class of its own that `frag4.glsl` paints in dull greens and browns with clumps of shade, running 91 m from the Shack down towards the castle. 103 of the 150 points now show. The rest of the hill keeps its trees.

Neither clearing is cut as a straight strip ([`clearings.py`](src/clearings.py)). The user asked that clearings, wherever they are, be "large enough to be plausible and not look suspiciously like cutouts created just for line of sight purposes". So each spreads beyond what its sight lines need by a margin no crown the renderer grows can reach across, and by as much again as its edge wanders. Beyond that edge the trees thin out over 12 m, and a few stand scattered inside it, clear of the lines, on ground where the renderer grows them (off the water, the roads and paths, and the buildings) and off the vegetable patch. Its own ground does not end at a line either: it gives way to the ground round it in tufts a texel or so across, each texel's chance of it falling evenly from 8 m inside the clearing's outline to 8 m outside (`ecotone`), so that the two interleave rather than meet at a wavering line. Every cleared region either follows a feature that explains it (the railway, a road, the Forest's edge along the A832) or is such a place, as the homesteads in the Forest are too: at least 0.25 ha and 30 m across, with no straight stretch of edge 60 m long and no two parallel ones facing each other. `build_final.py` records each in `meta.clearings` with the decision that gives its reason, and refuses to build if trees are removed anywhere in the local layer that no rule or recorded clearing accounts for. The Forest layer loses trees only where it ends and to its corridors and homesteads, and the railway's works clear them only along the line. `test_site.py` checks each record's size and shape, that the packed tree layers are cleared inside it but for its scattered trees, that each of those grows where it stands, that along each 40 m of its edge where the ground's colour changes it changes over 5 m or more, that the garden's colours are dull greens and browns, and the sight lines through the two cleared for them.

White Hound's maps draw rough scrubland, which their legend calls "Rough scrubland, not suitable for farming". In the 2 m layer that covers his grounds and Hogsmeade maps, [`frag4.glsl`](src/frag4.glsl) draws it as scrub, shrubby ground of olive green and bracken brown, not heather in bloom (the `scrubland-as-scrub` decision): nothing places heath or moor on the ground his maps draw, since his legend has no such ground and his setting page calls the land round Hogsmeade's fields moorland only in general. Beyond that layer, over most of the Forest map's scrubland, the ground keeps the Sentinel-2 imagery's colours, tinted green where the Forest map draws fields. Like a clearing's ground, every patch of ground (the scrub, the woods' floor and the lawns) gives way to the ground round it in tufts across 16 m of its edge (`clearings.blend`), but where a wall, a path, the railway's works or the shore explains its line. The woods' floor lies only under their canopy, beyond which its tufts would read as specks on open ground: it takes in the scrub, lawn and meadow round it only under the trees' thinning edge, and scrub gives none of its ground up to it. The station wood and the train's screen stand dense to a line, so their floor gives way within it instead, across the 16 m inside their edge and towards the grass where the boundary wall runs through them, though not towards a wood they run on into: grass under their outer trees. `test_site.py` checks that along each 40 m of every patch's edge, the two grounds interleave along at least a fifth of the lines across it, which a line, straight or wavering, would not; that no forest floor lies off the canopy but on stream beds and in the clearings' own fading edges; and that under a wood's dense canopy the floor gives way only within 8 m of its edge, or 16 m of a line where its trees end.

The essay map's railway does not fit that terrain on its own: the flattened station corridor sat in a trench below its road, the junction 36 m up the hillside 68 m away and the foot of the turning loop 25 m below it. [`fix_railway.py`](src/fix_railway.py) therefore regrades a bounded railway corridor designed by [`railway.py`](src/railway.py) from the traced line in [`mesh_scene.json`](src/mesh_scene.json): the loop level with the station, the main line climbing 1 in 40 into a tunnel, rock cuttings at 2:1 and embankments at 1 in 1.5, a level station terrace from the track to the boundary wall, and the lake path cut through the spur. It clamps the bed between the earthworks' cut and fill surfaces, so nothing outside them changes; it clears trees off the earthworks and sets their surface classes, clears every tree within 10 m of the open line (the loop, and the main line up to its tunnel). The renderer reads the tree density bilinearly, so it can stand a tree up to a texel diagonal inside cleared ground, and a crown spreads up to 4.9 m; at 10 m no crown comes within a metre of a coach's side. A stream the works reach gives way to them (at Scardroy under the embankment at the loop's foot, in a culvert; at X where the cutting there meets a burn). No other water may lie under the works. The portal stands `PORTAL_M` (25 m) up the main line from the turnout's toe unless the site sets `portal_m`, which `build_final.py` passes on in `meta.train`. At X it stands 180 m up, at the first point where the hillside stands the 10 m headwall and a 2 m margin over the formation across the portal's width. The line climbs to it on an embankment up to 2 m high, then in a rock cutting up to 13 m deep. The embankment crosses the dry valley by the turnout, which the train's screen (below) closes, so the valley is made up first, by up to 5.9 m. Hollows the railway's own embankments enclose are made up to where they would spill, as meadow, with spoil from the cuttings (12 m³ at X).

At site X the station terrace stands 8.4 m above the lake and 7.6 m above the flat ground the lake's rim fills west of it. White Hound's pines there grew barely above the eye, so the castle stood in full view from the platform and from the top of the lake path. That broke PS 6, "Yeh'll get yer firs' sight o' Hogwarts in a sec … jus' round this bend here". The site's `station_wood` in `SITES` applies White Hound's own remedy, "some sort of rise in the ground between the station/path and the lake" ([`grounds_grade.py`](src/grounds_grade.py)). Behind the boundary wall, alongside the terrace and easing out past its ends, a shelf rises 3 m above the terrace over 20 m, then falls 1 in 2. West of his path to the boats a ridge's crest stands 8.4 m above the terrace. Both carry dense pines: 0.10 Mm³ of fill and 1.25 ha of pinewood. The station's road keeps its own ground, and so does the walk to the lake: the lake path, then his path to the boats (1d) on through the pines, both traced in [`mesh_scene.json`](src/mesh_scene.json). Each keeps its ground 6 m either side as it was, and beyond that no rise climbs from it more steeply than the railway's rock cuttings, 2:1. Through the trees at most a glimpse of the castle shows (5 of 150 points on its upper parts, from anywhere on the platform or the walk). Where the walk rounds its last bend and opens on to the lake, 73 of them show. Beside the path's steep stretch, where the terrace's embankment spreads, [`fix_railway.py`](src/fix_railway.py) keeps White Hound's pines: "It was so dark either side of them that Harry thought there must be thick trees there" (PS 6).

The train's own approach had the same fault at X. Its tunnel portal stands 180 m up the main line, where the hill first covers the line, so from the 225 m of open line and loop below it the castle stood in full view. At Scardroy the train left its tunnel at the junction and saw nothing. White Hound reasons that "there must be a barrier – wall, trees or rising land – preventing the castle from being seen from the train as it approaches the station", and the user ruled that the train "is supposed to only show a hogwarts coming into view at a specific point". The site's `train_screen` in `SITES` builds a wooded ridge on the castle's side of the line ([`grounds_grade.py`](src/grounds_grade.py)). It runs the 270 m from the cutting's mouth past the junction to the platform's end. Its crest stands 10 m above the formation, 24 m out from the track, with flanks reaching 30 m either side. White Hound's boundary wall of the grounds (3a) runs along it, mostly 15 to 33 m from the track, so the ridge is broad enough for the wall to stand on gently sloping ground, not on a steep bank. Like the station wood it climbs from the walk to the lake no more steeply than 2:1. Both are shaped from the ground as it was before either, the higher standing where they meet, so neither climbs on the other's slope. It is planted with dense trees, none nearer the track than that same 10 m: 0.12 Mm³ of fill and 0.89 ha of wood. The hollows the two rises enclose are made up to where they would spill, keeping their trees, so they leave no small meadows in the wood: 0.010 Mm³ at X, most of it in that dry valley. From a seated or a standing passenger's eyes at the carriages' windows, 2.25 m and 2.7 m above the rail top, at most 2 of the 150 points on the castle's upper parts now show. The 8 m crest first costed left 9 and 11. Two alternatives were costed and set aside. A covered way extending the tunnel to the junction, as at Scardroy, still left the junction in view. No route within 1 in 40 and 60 m curves is hidden by the real ground.

The fictional lake outline comes from the grounds map, not the outline of an existing real loch. Its surface level is the site's `lake` in `SITES`: **284 m** at X, 153 m at Scardroy. A signed-distance field controls shoreline shape and a basin of roughly `0.8 + 11.5*(1-exp(-inward_distance/50))` metres depth. Shore gradients differ near the castle. Stream masks combine essay-map water with the site's real watercourses from OpenStreetMap, its `carve` in `SITES`: at X, Allt Mhartuin and its burns, carved only outside White Hound's maps; at Scardroy, the River Meig. Regional OSM water polygons are levelled using sampled DEM heights. The ship's berth is explicitly flooded so its map ink does not become a ship-shaped island: [`pack.py`](src/pack.py) floods a 12 m capsule round the hull's axis, then [`fix_berth.py`](src/fix_berth.py) opens the whole berth to the surrounding bay's depth and surface class, removing the ink's leftover islets, shallow rim and colour stripes. The underground harbour's exterior mouth ("a wide, ivy-curtained opening which leads into a water-filled channel which penetrates the cliff-face") is cut by [`fix_harbour.py`](src/fix_harbour.py): from White Hound's mapped mouth (grounds map 1f) it finds the nearest cliff foot in the data, a dry texel within 8 m of water standing 15 m or more above the lake, and lowers the bed inside a declared rectangle, 8 m wide from 2 m out in the lake to 12 m into the cliff, to 2 m under the lake as lake water. It records the mouth, axis and rectangle as `meta.extras.harbour`, which a rerun reuses, and changes nothing outside the rectangle; the grounds family's mesh roofs the recess over and closes its back. The lake's shingle (surface class 7, from the map's colours) comes in 2 m texels on a grid turned 25 degrees, which drew a beach as a staircase. The page therefore blurs the shingle into a smooth field when it loads and mixes it in by that field, with the classes' usual noise on its edge. Within the water's reach the ground normal takes a 2 m step, so a bank that drops through a single texel shades as one slope rather than cell by cell. Faint facets remain on the bank in oblique views; the tour's view at the tomb looks across the bank, a thin strip beyond the tomb, rather than along it.

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

These dimensions approximate the essay's tall, multi-storey castle; the exact metre values are not quoted source measurements. Each class has its own signed-distance footprint, clipped to a common outline. The raymarcher once unioned the separately extruded classes, so height changes became walls rather than interpolation ramps; the castle's meshes (`src/mesh_castle.py`) now stand in its place in every pass.

Candidate round towers are found from distance-field peaks, filtered for shape and separation, and supplemented by named towers. Non-special tower variation uses NumPy seed `7`. Named tower parameters are:

| Tower | Center `(E, N)` m | Radius / top above base | Shape |
| --- | --- | --- | --- |
| Astronomy (`AT`) | `(-25.69, -91.08)` | `3.6 / 58 m` | Thin shaft, overhanging open top |
| North (`NT`) | `(64.47, 180.79)` | `3.6 / 54 m` | Thin shaft, overhanging onion-dome top |
| Gryffindor (`GT`) | `(-2.24, -166.25)` | `7 / 30 m` | Broad base with three upper turrets |
| West/Owlery (`WT`) | `(-107.04, -9.09)` | `4.6 / 41 m` | Flat battlemented top |
| Ravenclaw (`RT`) | `(-100.75, 143.30)` | `5.2 / 45 m` | Conical top |

The essay motivates the distinctive tower forms; positions are registered annotations and exact radii/heights are modeling choices. These top values are inputs to procedural shapes, not necessarily the highest roof points. The packed castle textures (`casA`, `casB`) also stored the two nearby tower indices the raymarched castle used to reduce shader lookups; `DATA` still carries them, but the page no longer decodes them.

### Village and small buildings

Connected map components supply ordinary building footprints. Components below 10 square metres are dropped. White/glass, wood/stands and proximity to the hut/Shack select special kinds; larger footprints become larger/slate-roofed buildings. Remaining small buildings choose thatch versus slate with seed `11` and probability `0.72` for thatch.

Chosen eaves are 3 m for greenhouses and the hut, 6 m for the Shack, 8 m for buildings over 400 square metres, 6.5 m above 160 square metres, otherwise 4 or 5 m. The raymarched buildings added roofs according to kind and inward footprint distance; the meshes now carry every roof, and `build_meshes`' fallback massing stands a footprint as a prism up to its eave (a sloping top for the stands). This is a massing model, not a reconstruction of individual interiors or surveyed elevations.

### Hogsmeade's buildings (the village mesh family)

Status: current for every village footprint (milestones M2 and M3), except one whose plan is only specks (a sliver of a wing), which keeps the generic massing. The named buildings have their own designs and the small structures are built (M3), and so are the coarser levels of detail and shadow casters (M3b). M4 fits the village to site X (below), where by the user's choice the Shrieking Shack stands above the rest of the village: a hamlet north-west of it is left unbuilt and two cottages are moved.

[`src/mesh_village.py`](src/mesh_village.py) builds Hogsmeade's buildings as meshes from their mapped footprints, in four steps:

1. **Plan.** [`src/mesh_village_plan.py`](src/mesh_village_plan.py) straightens each traced footprint. It finds the wall lines from the trace, rebuilds the outline on them and fills holes under 8 m². If the result overlaps the trace by less than 0.8 (intersection over union), it keeps the trace, straightened. It splits each outline into buildings ("units") at the offsets in a row, and cuts rows over 16 m long into plots 5.5–9.5 m wide. A unit's front is the free wall at least 2.5 m long with the most road in front of it, or one looking across the road at the High Street if it has one. A unit fronting the High Street (the line traced in [`tools/manifest/data/village/traced.json`](tools/manifest/data/village/traced.json)) is a shop, and each named shop is the unit its map label points to.
2. **Style.** Each unit gets a style record from its family (shop, terrace, storey-and-a-half cottage, cottage, villa or outshot) and a seeded random choice: storeys and eaves, harl colour or rubble, thatch or slate, pitch, hipped or gabled ends, shopfront and door. Neighbours in a row never share both colour and height. The White Hound essay sets the defaults: most cottages are thatched, and the Hog's Head is "probably tiled not thatched". The manifest records each choice as a decision with its quote ([`tools/manifest/families/village.py`](tools/manifest/families/village.py)).
3. **Roofs.** [`src/mesh_village_roofs.py`](src/mesh_village_roofs.py) is the one roofing interface, `roof(outline, angles)`, with one pitch per edge and 90° for a gable. Its backend is CGAL's straight skeleton through `compas_cgal` 0.10.0, which the user chose at gate G1. The in-repo backend stays behind the same interface as the tested fallback: it roofs any outline, rectangles and other rectilinear plans with one pitch exactly, and anything else with each edge rising at its own angle to a flat top. CGAL's skeleton can fold a roof over itself, as it did for L-shaped cottages gabled beside the inside corner of the L; `roof()` checks that nothing but a roof's base faces down and otherwise builds that roof with the in-repo backend, and no short attic front is made a gable beside an inside corner or on a plan that is not square-cornered.
4. **Solids.** Each building is one manifold3d solid: walls reaching 1.5 m below the lowest ground under them, with a dark-painted base course (on harled walls) at most 0.5 m above that ground; an attic whose upright faces are the gables; and a roof skin whose eaves and verges overhang. Openings are cut out of the solid: windows, doors and shop windows on the front, a window at the back (two on a long back on the High Street), and on a storey-and-a-half front, dormers through the eaves (up to two on the High Street, one elsewhere) or a window in the gable. An opening's head stays under the eaves, and a window opens only where no village building stands within 3 m in front of it. Chimney stacks stand on gable ends and on shared party walls, or straddle the ridge of a building with no gable, their pale stone tops at least 0.6 m above any roof within 2.3 m. Every face takes its material from the solid it came from. The joined solid is tidied so that 16-bit quantization cannot turn a triangle over. Corners within 8 mm are joined, and slivers (triangles under 4 mm thick) are split away. A sliver that splitting cannot remove has its apex moved onto its longest side, and is left out with its sides meeting in a line. Where two slivers share an apex in the narrow wedge between their longest sides, each move would lift the apex off the other's side, so the apex joins the corner where those sides meet, and both slivers go (one such pair at site X, in a terrace's party gable). The payload moves a corner at most its chunk's diagonal over 65535 off the side facing it, which is under 4 mm for every village chunk. The shell stays closed. Fine fittings (chimney pots, sills, steps, fascias and hanging signs) are separate pieces, so integration can drop them with distance. No piece lies in the plane of one of its building's faces, where the two would fight in the depth buffer: a turret's base stands 5 cm behind its gable and crow steps 5 cm in from the gable's corners (`DEPTH_GAP_M`), and a sill starts at the wall's face.

[`src/mesh_village_named.py`](src/mesh_village_named.py) gives the named buildings their own designs through hooks that `mesh_village` calls at each step: a style (`restyle`), a plan of units (`layout`, `split`), roof pitches (`roof_angles`), openings and fittings (`openings`, `dress`, `finish`) and chimneys (`chimneys`, `stack_gables`, `stack_wall`).

- The Hog's Head follows White Hound's plan of it.
- The Shrieking Shack is weatherboard under corrugated iron, its windows boarded but the one facing the castle.
- The Three Broomsticks is a later inn with a very old end.
- The kirk is a rubble nave with a porch, lancets, buttresses and a lantern bell turret.
- The school, village hall, distillery and carriage hangar, which White Hound places but does not draw, follow Highland types.
- The other shops each have one feature drawn from the books or White Hound: Honeydukes' mullions, the post office's owl loft, Scrivenshaft's crow steps, Madam Puddifoot's steamed windows, Gladrags' blind and Dervish & Banges' bracket clock.

[`src/mesh_village_small.py`](src/mesh_village_small.py) builds what stands on no mapped footprint (`build_extras`): the High Street's lamps, the kirkyard's wall and headstones, and garden walls and gates near the High Street. Each design is a manifest decision with its quotations.

The same building records give two coarser levels of detail and a shadow caster (`LEVELS` and `levels(village)` in `mesh_village.py`; the manifest decision `village-levels`). Each building is built once, and every level is cut from that record (`_Building.at`), so whatever places a building places it at every level. Every level that passes its guards and saves enough triangles enters the payload, and the page draws one per view (see the levels of detail above), so some groups, the kirk's and the lamps' among them, ship fewer levels. The user ranks live frame time above page load, so the levels are designed for frame time, and none is cut to fit the payload's room.

- **lod1** lays every window and door flat on its wall, 5 cm proud of it (`DEPTH_GAP_M`), and stands the chimney stacks in the roofs as boxes; a stack flush with a gable stands 5 cm inside it. Its roofs are each skin filled in: the skin's outer solid, stopping at the buildings round it as the skin does, notched in front of the dormers, and trimmed back to the gable walls so that windows in the gables show. It keeps the base courses, the dormers, the named buildings' bays, buttresses and roof features (turrets, the kiln's ventilator, the owl loft and crow steps) and the kirkyard's and gardens' walls. It drops every other fitting, the painted margins round openings and the openings' reveals.
- **lod2** is each building's walls under its filled-in skin, verges included. It keeps the eaves because the flyover sees the village's far roofs from above: without them lod2 would take 21k triangles rather than 32k, but its roofs would cover a sixth less seen from above (intersection over union 0.84).
- **The shadow caster** is each building's walls and attic, within its walls: the cheapest level. It also keeps what stands out beyond the eaves and casts a shadow of its own (`Level.casts`): Gladrags' blind and the Shack's fence as built (hung with `_Building.cast`), the kirkyard's wall, and each garden's wall and gate as one slab along them. Seen from above it leaves no patch of more than 1 m² of the full village farther than 0.75 m from it; the eaves and verges it leaves out reach 0.45 m. The page draws its mesh shadow layers again only when the sun or the payload changes (`meshShadows` in [`src/template_b.html`](src/template_b.html)), as when the time of day is dragged, and the village layer's texels are 0.59–0.91 m as the sun moves (its box spans 1.2–1.9 km of the sun's view at 2048 texels), wider than the eaves' shadow, which the near layer draws round the camera (`nearShadow`).

Across the whole village at site X, with the small structures each level keeps (packed sizes are the village's chunks of that level packed and compressed alone):

| Level | Triangles | Share of lod0 | Packed |
| --- | --- | --- | --- |
| lod0 | 170,548 | 1 | 1.54 MB |
| lod1 | 67,947 | 0.40 | 0.64 MB |
| lod2 | 31,770 | 0.19 | 0.31 MB |
| shadow | 21,303 | 0.12 | 0.23 MB |

The village's full level may add about 1.5 MB to the payload, its named buildings and small structures included. Its flat-shaded triangles cost about 9 bytes each in payload format v2, so that share, not a triangle count, is what limits it. Built with all its detail, the village would have added about 1.6 MB (225k triangles). Since the flyover passes 50 m and more above the village, fine detail now goes where it can be seen (the manifest decision `village-detail-where-seen`): stacks are capped flush rather than under overhanging copes, a raised door's steps are one stone block, and off the High Street a stack carries one pot, a back wall one window and a storey-and-a-half front one dormer. With its named buildings and small structures (M3), the village is 177k triangles. Its own share of the payload, its chunks packed and compressed alone, was 1.57 MB, 1.29 MB more than at T4, which left about 0.2 MB. At site X (M4) it is 171k triangles and 1.54 MB. Since the payload carries every level of detail, the village's share is 2.68 MB; packed alone, as above, its full level takes 1.54 MB, levels 1 and 2 0.64 and 0.31 MB, and its shadow caster 0.23 MB. The tests hold the village to 200k triangles, and each named building to under 4,000.

`compas_cgal` and the packages it brings are build-time dependencies only: they are installed from [`requirements.lock.txt`](requirements.lock.txt), never vendored, and none of their code ships in the page. Their licences:

| Package | Licence |
| --- | --- |
| `compas_cgal` 0.10.0 | LGPL-3.0 |
| CGAL's 2D straight skeleton, compiled into the `compas_cgal` wheel | GPL-3.0-or-later |
| `compas` 2.15.1 | MIT |
| `networkx` 3.4.2 | BSD-3-Clause |
| `jsonschema` 4.26.0, `jsonschema-specifications` 2025.9.1, `referencing` 0.37.0 and `rpds-py` 0.30.0 | MIT |
| `watchdog` 6.0.0 | Apache-2.0 |

Heights come from the sampled ground, never from fixed levels: the floor stands 0.15 m above the highest ground along the front, with a stone block of steps where the ground falls away from the door.

At site X (M4) the village fits its ground with no change to these rules. Under its buildings the ground slopes 0.2° at the median and at most 6.1°; every pane of glass stands at least 0.56 m above the ground 0.9 m in front of it, and the carriage hangar's doors open onto the ground along a front that rises 1 m. The quoted book evidence holds at X: the Hog's Head, along its side street off the High Street, faces 134°, and its front takes the morning sun on 7 October 1995 (OotP 16); the High Street's shops keep their order (OotP 16); and Madam Puddifoot's stands up a side road off it (OotP 25).

The Shrieking Shack "stood a little way above the rest of the village" (PoA 14). At X its floor stood above 812 of the village's 819 other buildings, but a farm hamlet on the hill to its north-west stood 13–15 m higher, and two cottages on the lane to the stile less than a metre lower. The user chose option 5b (the manifest decision `shack-above-the-village`):

- The manifest leaves the hamlet's four footprints unbuilt: they are excluded components, through the village family's `excluded(ctx)`. The hamlet's track stays.
- The village builds cottages 23 and 28 across their lane (`MOVED` in `mesh_village.py`), turned to face it, 32.5 m and 50 m from where the map draws them. Cottage 28 moves 5 m further than costed: the raymarcher grows trees from the site's tree layer, which keeps them off mapped footprints only, so the costed site had trees at its walls.

The Shack's floor now stands 4.1 m above every other, and `test_assets_village` holds it at least 2 m above. Since the meshes replaced the raymarched buildings (mesh integration), nothing stands on the hamlet's footprints or on the two cottages' mapped sites.

The family's material codes are (7, 60–95), in [`src/materials/village.py`](src/materials/village.py). Every family's set pieces share the page's one shading shader, a branch for each distinct shading body, and `materials.validate` refuses more than 40 bodies (`MAX_P3_BRANCHES`; see [Adding a family](#adding-a-family) for the limits behind the cap). The village's seven harls and eleven gloss paints therefore share one branch each, reading their colour from a constant array by code.

| File | Checks |
| --- | --- |
| [`src/test_assets_village.py`](src/test_assets_village.py) | Plans, roofs, the tidy and the built village: which way buildings face, doors at floor level with steps where raised, walls below ground and base courses at their foot, windows recessed, unobstructed and clear of the ground before them, roof skins above their attics, dormers running into their roofs, stacks clearing the roofs and standing on walls, closed shells with no sliver, a deterministic build, each named building's design and small structure, and the Shrieking Shack's floor at least 2 m above every other, with the cottages moved for it across their lane, square to it and clear of trees. `VillageLodTests` checks the coarser levels against the full one: each level's share of the triangles; each building group's silhouette from the east, the north and above (the shadow caster's along a low sun) on 0.25 m cells; what each level takes from the building records; lod1's flat openings, stacks and roof features without fittings; lod2 as walls and roofs alone, and the shadow caster as those and what casts beyond the eaves, covering every part of the full village seen from above; and, at lod0 and lod1, no piece in the plane of one of its building's faces. The village is built once per test process (`shared()`), and every class reads it. `VillageBuildTests` checks every footprint and holds the village to its triangle budget. With `VILLAGE_SAMPLE=high-street`, which the mutation runs set, it builds only the footprints within 14 m of the High Street's centreline, the named buildings and the houses with garden walls |
| [`src/test_render_village.py`](src/test_render_village.py) | The materials, and windows glowing at dusk, on the GPU |
| [`tools/mutate/village.py`](tools/mutate/village.py) | The mutations those tests must catch: `python -X utf8 -m tools.mutate.run village` |
| [`tools/review/village.py`](tools/review/village.py) | The review sheet's views, the named buildings close up among them: `doit s5:village` |

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

The Forbidden Forest's **density** comes from map vegetation colors and tree ink. Individual tree positions, sizes and shapes are procedural and deterministic, not individually sourced. Ground/water/building masks exclude trees; the building mask is the meshes' footprint layer, so no tree stands on ground a standing mesh covers, and trees fade in over the 2 m beyond the layer's 0.5 m, which reads about 1.2 m short of a wall's true distance. Density blends near map boundaries and fades with elevation between roughly 500 and 640 m; conifer fractions combine selected defaults with regional plantation masks. Near trees use 3D distance-field shapes on a 7 m cell grid; distant vegetation becomes a canopy representation. A rewrite can choose meshes instead without changing the map-derived forest footprint.

The scene includes features from different books, including the ship and tomb, while the lighting calculation uses 1 September 1991. It is therefore **not a historically consistent snapshot of that date**; the date controls the sun, not the existence of every object.

## Rendering and camera

The only runtime delivery is the generated HTML. Its templates decode compressed embedded arrays, create textures, compile GLSL, tabulate tree data and run temporal reconstruction/presentation passes. There are no downloaded models or scene-library dependencies.

| File | Responsibility |
| --- | --- |
| [`frag1.glsl`](src/frag1.glsl) | Uniforms, coordinate sampling, noise, terrain, water and vegetation |
| [`frag2.glsl`](src/frag2.glsl) | The pitch's frame, for the ground's markings. It held the raymarched castle, buildings, walls, hedges and set pieces until the exterior meshes replaced them in every pass |
| [`frag3.glsl`](src/frag3.glsl) | Combined ray queries for terrain, water and vegetation |
| [`frag4.glsl`](src/frag4.glsl) | Surface colors, satellite albedo, windows, sky and fog |
| [`frag5.glsl`](src/frag5.glsl) | Lighting, shading and main scene entry point |
| [`template.html`](src/template.html) | Markup, controls, embedded-data decoders and CPU terrain sampling |
| [`template_b.html`](src/template_b.html) | WebGL resources, shader definitions, the tree table (drawn again whenever a payload changes the footprint layer) and temporal passes |
| [`template_c.html`](src/template_c.html) | Solar direction, camera path, captions, interaction and render loop |
| [`assemble.py`](src/assemble.py) | Concatenation/inlining into one page |

Terrain is height-field sampled and the near trees are distance fields; the combined marcher takes bounded steps, refines hits and shades by material. The buildings and set pieces are the exterior meshes (`build/meshes.js`): a depth pre-pass ends the march at the nearest mesh, each mesh writes the same G-buffer record as the raymarched surfaces, and the meshes cast their own sun shadows (cached layers, which the shading pass looks up for each pixel) and lake reflections (a mirrored pass). The mirrored pass draws only the meshes whose mirror image could show through the lake's water: the rays to a mesh's mirrored box must cross the water's plane within reach of the box round the lake's water (found at load). The reach is a quarter of the crossing's distance, because a rippled reflection looks the image up displaced by up to 0.18 of the water's distance. Debug bit 8192 mirrors every mesh the camera's frustum holds. Nor is the mirrored pass drawn while none of the lake's water shows, as when the castle hides it: after the G-buffer pass an occlusion query asks whether any pixel is the water the shading pass mirrors the meshes in, and is answered a frame or more later. A "none" holds for the state it was asked in and while the camera stays within 10 m and 2° of where it asked, so a jump, a turn or a change of state draws the mirror until a new answer comes; water that comes into view as the camera moves on smoothly lacks the meshes' reflections for the frames an answer takes. Debug bit 32768 draws the mirror whatever the query found. The meshes' shadows also have a near layer (`nearShadow` in [`src/template_b.html`](src/template_b.html)): 192 m of light-space footprint round the camera's ground point, snapped to 32 m cells, at about 9 cm texels, drawing every group's full level, eaves included, and drawn again only when that cell, the sun or the payload changes. It costs 0.0–0.3% of a frame's GPU time on both GPUs. Debug bit 16384 leaves it out. The meshes are on by default; `?meshes=0` shows the landscape alone, as a diagnostic. Shadows, lake reflections, fog, procedural detail and temporal accumulation are renderer choices rather than evidence about the place.

The camera path and captions are authored in `template_c.html`, not derived routes verified against the essay. The tour lasts about 574 s, bringing the camera close wherever the meshes have detail to show. `src/test_render_tour.py` checks that each caption's subject is the visible surface over at least 0.25% of the frame at its waypoint, as the flyover shows it, so trees and terrain in front count against it. The Whomping Willow, a tree among trees, must reach 1%. Aragog's hollow and Grawp's clearing are measured as their ground within their clearings' recorded outlines (`meta.clearings`), which, being dry ground in the Forest, must also show more than any water in the frame. Sirius's cave, which the scene does not model, must be in shot: its spot lies in the frame, a tenth of the frame in from each edge, and nothing nearer hides it. The path's clearance follows the ground and the trees, not the meshes, so the same test also keeps the camera out of every building's box (assets spread over more than 400 m, such as walls and bridges, excepted). When a waypoint moves, record why in the manifest (`tour_decisions` in `tools/manifest/general.py`). Then move the probe's windows and make and pin `baseline-x` again, as [Inputs and shared state](#inputs-and-shared-state) describes. The solar calculation uses 1 September 1991, BST and the selected location. Adaptive resolution, quality and detail controls trade rendering cost against appearance. The raymarcher is deliberately self-contained, but none of these techniques is a requirement for a more intentional rewrite.

### Known limitations

Two behaviours of the meshes' sun shadows are known and accepted. Both appear only where a static shadow layer shades the ground, past the near layer. That layer's 192 m footprint, snapped to 32 m cells, reaches 73–119 m from the camera's ground point across the sun's bearing, and farther along it (1.7 times as far with the sun 37° up). The sizes below were measured on the shipped payload with the sun more than 9° up; pixel counts are at the default internal scale of 0.5 and 1080 output rows.

- **Thin casters' far shadows are wide and faint.** A static layer draws each narrow part as lines one texel wide (see [Mesh payload format](#mesh-payload-format)). The shading pass's four filtered taps spread such a line into a band that is half dark across one texel and fades out over one more texel on either side. The layers' texels vary with the sun: 0.22–0.27 m over the castle and greenhouses, 0.45–0.62 m over the grounds, the transport and the Willow, 0.59–0.91 m over the village and 0.86–1.31 m elsewhere. So a street lamp's 0.16 m post casts a band 0.59–0.91 m wide at half dark, not a dark line 0.16 m wide: from 150 m, two to four internal pixels at half dark instead of under one at full dark. Without the lines, such a shadow breaks into dots or vanishes. Debug bit 16384 leaves out the near layer and shows these bands close up.
- **Parallel thin parts close together merge their far shadows.** The lattice rule leaves out of the lines only parts within 0.2 m of a parallel one, two near-layer texels. In a static layer, parallel narrow parts closer than about two of its texels still merge their bands into one soft band, darker than their shadows are on average; the near layer shows them apart. In the shipped payload these are 261 of the pitch's 2,472 narrow parts (mostly its stands' rails, with braces and posts, 0.25–0.95 m apart), 104 of the hut garden's 185 (its fence rails, 0.6–1.0 m apart), 30 of the ship's 46 (its shrouds, 0.65–0.8 m apart) and 53 of the Willow's 145 (0.2–1.1 m apart). In the village they are 156 of the houses' 1,328 (chimney pots on shared stacks, fascias and sills, 0.4–1.6 m apart) and a few on the named shops.

## Reconstructing the scene

1. **Separate evidence from design parameters.** Start with a manifest of the essay pages/maps and geographic datasets. Record which constraints each source supports, and keep chosen heights, anchor, seeds and terrain edits visibly separate.
2. **Establish frames before geometry.** Select an anchor, define metre-based east/north coordinates, register the maps, and verify the centroid/willow examples above. Keep geographic coordinates, map pixels and world coordinates distinct.
3. **Prepare geographic context.** Mosaic/sample the elevation model, rasterize real water polygons/rivers, and prepare satellite colors. Keep provenance, source resolution and vertical-height assumptions with each layer.
4. **Extract fictional layout.** Classify map colors, remove annotations, identify footprints/lines/vegetation, and define overlap precedence. Store named landmarks and manual corrections explicitly rather than burying them in extraction commands.
5. **Construct the local landscape.** Blend the lake, castle cliff/platform, hillocks and flat functional areas into the real terrain. Review shoreline, drainage and seam behavior before adding buildings.
6. **Construct architecture and vegetation.** Use the castle plan and named tower forms, then parameterize uncertain dimensions. Generate village buildings and trees from explicit deterministic rules. Preserve map-derived placement independently of your geometry technique.
7. **Render and validate the composition.** Check castle/lake/forest relationships, hut and station placement, village orientation, tower silhouettes and terrain seams. Compare high and low views, day/night and the castle/tree sequence near 60-65 tour seconds. These checks do not prove the chosen real-world location.
8. **Make the next pipeline reproducible.** Save download manifests, classifications, annotations, transforms, intermediate schemas and build commands. The original project did not fully preserve this boundary; a rewrite should.

The saved final pipeline is `features.py` -> `build_final.py` -> `build_extras.py`; `build_regional.py` prepares regional layers independently; `pack.py` consumes all final layers; [`fix_berth.py`](src/fix_berth.py) then rewrites the local height and surface layers round the ship's berth in `build/data.js` (it is idempotent and touches no other layer); [`fix_harbour.py`](src/fix_harbour.py) then cuts the harbour mouth into the local height, water and surface layers (idempotent, inside its declared rectangle only); [`fix_railway.py`](src/fix_railway.py) then grades the railway's earthworks into the local height, water, tree, stream and surface layers (also idempotent); `assemble.py` creates the HTML. This order assumes their missing source inputs already exist. `build_local.py`, `build_terrain.py` and `build_castle.py` are earlier iterations, not extra required final stages.

`pack.py` quantizes height layers, predicts neighboring values, zigzag-encodes residuals and compresses byte planes with zlib. It packs masks/classes similarly and imagery as JPEG. The browser reverses these encodings. Those are storage choices: a rewrite can keep ordinary arrays or explicit geometry instead, provided it preserves coordinates, units and categorical sampling.

## Reproducibility limits

**Supported:** rebuild the existing standalone page from tracked sources and `build/data.js`, without GIS dependencies or original downloads. This was checked byte-for-byte against the accepted HTML.

**Supported since the move to X:** regenerate site X's `build/data.js` from pinned sources. `python -X utf8 -m tools.site.sources fetch` fetches the geographic sources (elevation models, the OSTN15 grid, OpenStreetMap extracts and Sentinel-2 tiles), each pinned by size and SHA-256 in [`tools/inputs.lock.json`](tools/inputs.lock.json). `python -X utf8 -m tools.site.build x` stages them, runs the data pipeline and checks the result against its own pin, `data-before-railway-x` (see [Inputs and shared state](#inputs-and-shared-state)). The staging reproduces the original build's elevation mosaic, coarse imagery and plantation masks from the same sources, and `test_site` checks that it does.

**Not self-contained:** the layers that do not depend on the site come from the original build's intermediates, pinned as `original-intermediates`, rather than from the maps. They include:

- the map palette classifications (`build/cls_*.npy`) and the manual castle footprint and named-point extraction;
- the grounds lake mask (`lake_mask_grounds.npy`).

The repository's `georef.json`, `feature_offsets.json`, saved scripts and packed metadata preserve substantial reconstruction information, but not every one-shot decision. The registration values are recorded measurements/choices, not a reproducible fit procedure with residuals. No complete location-search proof, uncertainty model or automated source-acquisition pipeline is claimed.

The legacy `shoot.py`/`bench.py` provide developer diagnostics with Playwright/Edge; they are not required to view, assemble or deploy the site. Performance observations on one GPU should not be treated as guarantees on another.
