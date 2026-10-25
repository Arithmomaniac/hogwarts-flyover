> Created/edited by GitHub Copilot; pending human review.

# Hogwarts flyover

[**Fly over Hogwarts**](https://arithmomaniac.github.io/hogwarts-flyover/)

A standalone, offline-capable HTML page containing a WebGL2/GLSL raymarched landscape: Hogwarts, its grounds, Hogsmeade and the Forbidden Forest, placed within real Scottish topography. The captioned camera tour lasts about seven minutes; you can also explore freely.

This is a visualization of **White Hound's interpretation**, not a definitive location or architectural plan. Fictional geography comes only from [*The Map of Hogwarts* and related pages](https://web.archive.org/web/20261008141630/http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts.htm); geographic context comes from elevation, satellite imagery and OpenStreetMap data. It does not use the films' castle or theme-park layout.

The published renderer is the accepted WebGL version based on commit [`9153cc9`](https://github.com/Arithmomaniac/hogwarts-flyover/commit/9153cc939f10411d596a7fe44429381d20cc5306). Experimental WebGPU and optimization work is not part of the deployed site.

## View it

Use the [live site](https://arithmomaniac.github.io/hogwarts-flyover/), or download [`hogwarts-flyover.html`](hogwarts-flyover.html) and open it locally in a browser with hardware WebGL2. It has no runtime asset downloads or server dependency. Microsoft Edge on Windows was the development browser. Initial shader compilation can take noticeable time; performance depends on your GPU and resolution.

The overlay keeps the current landmark, tour progress, playback buttons and time-of-day slider visible. **About** explains the location and links to the repository/rebuild guide; **Controls** lists shortcuts; **Stats** reveals camera coordinates and rendering diagnostics; **Credits** expands the compact source attribution. Dialogs pause the tour and restore its previous playback state when closed with Close or Escape.

| Control | Action |
| --- | --- |
| Drag | Look around |
| **F** | Toggle free flight |
| **WASD**, **E/C**, **Shift** | Move, rise/descend, move faster in free flight |
| Playback buttons / **Space** | Pause/resume |
| **Left/Right arrows** | Scrub the tour |
| **Up/Down arrows** | Change time of day |
| **Q**, **L**, **H** | Cycle quality, toggle detail, hide/show HUD |

URL parameters are `t` (tour seconds), `tod` (hour, BST; default `18.5`), `q` (quality `0` to `2`), `scale` (fixed render scale instead of adaptive resolution), `pause=1`, `detail=0` and `hud=0`. For example, [pause near the castle/tree sequence](https://arithmomaniac.github.io/hogwarts-flyover/?t=60&pause=1).

Click the scene to use keyboard shortcuts; inputs, buttons and dialogs keep their normal keyboard behavior. To run the overlay's browser checks with Python, Playwright and installed Microsoft Edge: `python -m unittest discover -s src -p test_ui.py`.

### Frame pacing and GPU selection

**Stats** shows animation-loop FPS and the median, 95th-percentile (p95), and maximum frame gaps over the last 120 intervals. These use uncapped elapsed time, not the capped movement step. They measure animation-loop cadence, not GPU execution time or confirmed display presentation. The GPU line identifies the selected adapter when the browser exposes it; otherwise it reports that adapter details are unavailable.

For jerky panning, check the longer frame gaps and the selected GPU, not just average FPS. The page already requests WebGL's `high-performance` power preference, but [that is only a hint](https://registry.khronos.org/webgl/specs/latest/1.0/#5.2) and the browser may still select integrated graphics.

On Windows with multiple GPUs, the following launches a local copy in a separate Edge profile using Chromium's [`--force-high-performance-gpu` switch](https://chromium.googlesource.com/chromium/src/+/refs/heads/main/gpu/config/gpu_switches.cc). Run it from the repository root:

```powershell
$edge = Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe'
$profile = Join-Path $env:TEMP ("hogwarts-flyover-gpu-" + [guid]::NewGuid())
$page = (Resolve-Path .\hogwarts-flyover.html).Path
& $edge "--user-data-dir=$profile" --force-high-performance-gpu $page
```

The separate profile starts a new browser instance rather than reusing one launched without the switch. It leaves the normal Edge profile and Windows settings unchanged; its profile files remain in the temporary directory after closing. Verify the intended adapter in **Stats**. In local tests this selected the NVIDIA RTX adapter without changing resolution, shader quality, or scene detail; selection and performance still depend on the machine and browser.

## Rebuild or rewrite

See [REBUILD.md](REBUILD.md) for assembly instructions, source evidence, location reasoning, coordinate transforms, terrain and building construction, feature placement, rendering logic, and a reconstruction sequence.

The existing page can be assembled from tracked sources and packed data. Regenerating that data from the original maps and geography is not currently self-contained; the guide lists the missing preparation steps.

## Sources and credits

- **Fictional layout:** White Hound, [*The Map of Hogwarts*](https://web.archive.org/web/20261008141630/http://members.madasafish.com/~cj_whitehound/Fanfic/map_of_Hogwarts.htm), including related essay sections and maps. Interpretation remains the author's; this repository does not relicense those source works.
- **Elevation:** [Copernicus DEM GLO-30](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM), copyright DLR e.V. 2010-2014 and Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA.
- **Imagery:** Sentinel-2 cloudless 2023 by [EOX IT Services GmbH](https://s2maps.eu/), containing modified Copernicus Sentinel data 2023, under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Embedded imagery was modified by mosaicking/resampling, approximate de-shading and JPEG encoding; its applicable attribution, non-commercial and share-alike terms remain relevant to the derived imagery in `build/data.js` and the page.
- **Real lochs and rivers:** [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), database under [ODbL](https://opendatacommons.org/licenses/odbl/1-0/).
- **Fictional universe:** an unofficial, non-commercial fan visualization. Harry Potter copyright J.K. Rowling / Warner Bros.; no affiliation or endorsement is claimed.

These credits are not a blanket license for the repository or permission to relicense third-party material. Public source availability does not erase the individual source terms.
