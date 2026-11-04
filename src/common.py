import os, json
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from pyproj import Transformer

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(WORK, *a)

# Essay reference point (location.htm / setting.htm): 57deg30'N 5deg00'W
REF_LAT, REF_LON = 57.5, -5.0
TM = "+proj=tmerc +lat_0=57.5 +lon_0=-5 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
to_ll = Transformer.from_crs(TM, "EPSG:4326", always_xy=True)
to_tm = Transformer.from_crs("EPSG:4326", TM, always_xy=True)

# Castle sites:
# - anchor: the castle in the TM frame (metres east and north of the reference point);
# - lake: the lake's surface level (m);
# - dem: the elevation model tools.site.stage prepares;
# - carve: the OpenStreetMap waterways carved into the local terrain, with their channels' widths (m);
# - carve_in_maps: whether they are carved inside White Hound's grounds and Hogsmeade maps too, or only outside them,
#   where the maps' own streams do not stand for them;
# - graded: whether the carved channels follow their lines exactly and never rise along their course;
# - fill_pits: whether hollows that the earthworks enclose are filled to where they would spill;
# - bank: for a lake perched above its surroundings, the (crest width in m, outer slope) of the bank that holds it;
# - base: the castle's base, built by castle_base.py ("mound" or "crag"), or None for the original build's plinth;
# - cabin_terrace: Hagrid's cabin's hollow raised to a terrace this many metres above the gates' ground, with the drive
#   climbing to it (grounds_grade.cabin_terrace), or None for the ground as it lies;
# - willow_knoll: the Whomping Willow on a knoll this many metres above its ground (grounds_grade.knoll), or None;
# - willow_lawn: the radius (m) round the Willow kept clear of other trees, in a lawn opened to it from the northern
#   greenhouses (clearings.glade), or None;
# - shack_garden: how far (m) beyond the Shrieking Shack's castle-facing wall its one unblocked window's view of the castle
#   is kept clear of trees, in its overgrown garden (clearings.glade), or None;
# - aragog_hollow, grawp_clearing: where (east, north, m from the castle) Aragog's hollow is cut into the ground and
#   cleared of trees, and where Grawp's clearing is cleared but for a few small pines (forest_places.py), or None for
#   the places White Hound's Forest map draws, left as drawn;
# - portal_m: where the main line enters its tunnel, metres up it from the turnout's toe (meta.train.portal_m), or None for
#   railway.PORTAL_M;
# - station_wood: (rise, crest) of the wooded shelf behind the station's boundary wall and of the wooded ridge west of
#   the path to the boats, metres above the station terrace (grounds_grade.station_wood), or None;
# - train_screen: the crest of a wooded ridge on the castle's side of the train's last approach, metres above the
#   formation (grounds_grade.train_screen), or None;
# - names: real places recorded in the extras, each with its spellings;
# - place: where the castle is, as the page's location note says it;
# - glen: the glen the page's title, loading screen and heading name the site by;
# - forest: where White Hound's Forest stops short of the site's real roads, railway and buildings, and the corridors
#   cleared through it elsewhere (forest_extent.keep); absent, his Forest stands as drawn.
# Scardroy is the original build's site. X is the location review's best fit: the shelf south of Loch Gowan at the head
# of Glen Carron (57.5420 N 5.0760 W, NH 1599 5437), where Allt Mhartuin and its burns are carved outside White Hound's
# maps.
SITES = {
    "scardroy": {"anchor": (560.0, 2480.0), "lake": 153.0, "dem": "glo30", "carve": {"river": 12.0},
                 "carve_in_maps": True, "graded": False, "fill_pits": False, "bank": None, "base": None,
                 "cabin_terrace": None, "willow_knoll": None, "willow_lawn": None, "shack_garden": None,
                 "aragog_hollow": None, "grawp_clearing": None,
                 "portal_m": None, "station_wood": None, "train_screen": None,
                 "names": (("Loch Beannacharain",), ("Scardroy Lodge",), ("Creag Coire na Feòla",), ("Bac an Eich",),
                           ("Sgùrr a' Ghlas Leathaid", "Sgurr a' Ghlas Leathaid"), ("Meallan nan Uan",),
                           ("Maoile Lunndaidh",), ("Sgùrr na Fearstaig", "Sgurr na Fearstaig")),
                 "place": "near Scardroy in the River Meig glen", "glen": "Strathconon"},
    "x": {"anchor": (-4550.0, 4675.0), "lake": 284.0, "dem": "fabdem", "carve": {"river": 4.0, "stream": 2.5},
          "carve_in_maps": False, "graded": True, "fill_pits": True, "bank": (12.0, 0.4), "base": "mound",
          "cabin_terrace": 4.1, "willow_knoll": 24.0, "willow_lawn": 20.0, "shack_garden": 60.0,
          "aragog_hollow": (-581.0, 1739.0), "grawp_clearing": (-1700.0, 650.0),
          "portal_m": 180.0, "station_wood": (3.0, 8.4), "train_screen": 10.0,
          "names": (("Loch Gowan",), ("Loch Carn Mhartuin",), ("Loch Coir' a' Bhuic",), ("Allt Mhartuin",),
                    ("Achnasheen",), ("Cnoc an t-Sìthein", "Cnoc an t-Sithein"), ("Fionn Bheinn",),
                    ("Càrn Gorm", "Carn Gorm"), ("Mòruisg", "Moruisg")),
          "place": "south of Loch Gowan at the head of Glen Carron", "glen": "Glen Carron",
          "forest": {"ends_at": ("A832",)}},
}
STAMP = P("build", "stage.json")


def site_name(stamp=STAMP):
    """The castle site to build: the one whose inputs tools.site.stage put in place (build/stage.json), else the site
    the committed build/data.js shows. A stamp naming no site defined here (one left by another branch, say) is refused,
    and so is the record of a stage that did not finish, whose inputs may mix two sites'."""
    if not os.path.exists(stamp):
        return "x"
    try:
        with open(stamp, encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, ValueError):
        record = None
    if isinstance(record, dict) and "staging" in record:
        raise RuntimeError(f"{stamp} records a stage of {record['staging']} that did not finish; stage it again with "
                           f"`python -X utf8 -m tools.site.stage {record['staging']}`")
    site = record.get("site") if isinstance(record, dict) else None
    if site not in SITES:
        raise RuntimeError(f"{stamp} names no site defined in src/common.py ({', '.join(SITES)}); delete it, then stage "
                           f"one with `python -X utf8 -m tools.site.stage SITE`")
    return site


def site_dem(stamp=STAMP):
    """The elevation model the staged inputs come from (another than the site's own when staged with --dem, to compare
    models), else the site's own."""
    site = site_name(stamp)
    if not os.path.exists(stamp):
        return SITES[site]["dem"]
    with open(stamp, encoding="utf-8") as f:
        return json.load(f)["dem"]

def compass16(e, n):
    """The 16-point compass direction (from grid north) of a vector e east, n north."""
    points = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
    return points[int(np.floor(np.degrees(np.arctan2(e, n)) % 360 / 22.5 + 0.5)) % 16]


def fill(z, outlets=None):
    """z with every closed depression raised to the level it spills at: a priority flood (8-connected) from the edges
    and from any outlet cells, such as a lake, which drain whatever reaches them."""
    import heapq

    z = np.asarray(z, np.float64)
    ny, nx = z.shape
    out, done = z.copy(), np.zeros(z.shape, bool) if outlets is None else np.array(outlets, bool)
    done[[0, -1], :] = True
    done[:, [0, -1]] = True
    heap = [(z[i, j], i, j) for i, j in zip(*np.nonzero(done))]
    heapq.heapify(heap)
    steps = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    while heap:
        h, i, j = heapq.heappop(heap)
        for di, dj in steps:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and not done[a, b]:
                done[a, b] = True
                out[a, b] = max(z[a, b], h)
                heapq.heappush(heap, (out[a, b], a, b))
    return out


def __getattr__(name):
    """SITE and the castle's anchor resolve when first used, from the staged site. A build step's `from common import *`
    uses them, so it refuses an unknown or unfinished stage; the site tools import only the constants here, so they
    can still stage again."""
    if name not in ("SITE", "CASTLE_E", "CASTLE_N"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    site = SITES[site_name()]
    globals().update(SITE=site, CASTLE_E=site["anchor"][0], CASTLE_N=site["anchor"][1])
    return globals()[name]


GEO = json.load(open(P("georef.json")))
U0, V0, MFF = GEO["U0"], GEO["V0"], GEO["mFF"]          # castle centroid in Forest-map px, metres per FF px
ROT = np.deg2rad(GEO["rot_deg"])                        # map north is rotated ROT clockwise from image-up
GX0, GX1, GY0, GY1 = GEO["grounds_rect"]
HX0, HX1, HY0, HY1 = GEO["hogs_rect"]
FF_W, FF_H = 2900, 3440
G_W, G_H = 2000, 2348
H_W, H_H = 1735, 2261
SGX, SGY = G_W / (GX1 - GX0), G_H / (GY1 - GY0)
SHX, SHY = H_W / (HX1 - HX0), H_H / (HY1 - HY0)
LOCAL_RES_M = 2.0                                       # the local layer's texels
TREE_CROWN_M = 4.9                                      # the widest crown the raymarcher grows (frag1.glsl treeH)
TEXEL_REACH_M = LOCAL_RES_M * np.sqrt(2.0)              # a bilinear sample reads texel centres up to a diagonal away


def ff_to_world(U, V):
    """Forest-map pixel -> world metres (east, north) relative to castle."""
    du, dv = (np.asarray(U, float) - U0) * MFF, (np.asarray(V, float) - V0) * MFF
    return du * np.cos(ROT) + dv * np.sin(ROT), du * np.sin(ROT) - dv * np.cos(ROT)


def world_to_ff(E, N):
    E, N = np.asarray(E, float), np.asarray(N, float)
    du = E * np.cos(ROT) + N * np.sin(ROT)
    dv = E * np.sin(ROT) - N * np.cos(ROT)
    return U0 + du / MFF, V0 + dv / MFF


def g_to_ff(u, v): return GX0 + np.asarray(u, float) / SGX, GY0 + np.asarray(v, float) / SGY
def h_to_ff(u, v): return HX0 + np.asarray(u, float) / SHX, HY0 + np.asarray(v, float) / SHY
def ff_to_g(U, V): return (np.asarray(U, float) - GX0) * SGX, (np.asarray(V, float) - GY0) * SGY
def ff_to_h(U, V): return (np.asarray(U, float) - HX0) * SHX, (np.asarray(V, float) - HY0) * SHY


def load_idx(name):
    im = Image.open(P("img", name))
    pal = np.array(im.getpalette()[:768], dtype=np.int32).reshape(-1, 3)
    return np.asarray(im), pal


# ---- colour classes for the essay maps (3-3-2 style palette) ----
C_LAWN, C_SCRUB, C_DENSE, C_UNDER, C_WATER, C_BUILD, C_EARTH, C_WOOD, C_WHITE, C_BLACK, C_RED, C_OTHER = range(12)
CLASS_NAMES = ["lawn", "scrub", "dense", "under", "water", "build", "earth", "wood", "white", "black", "red", "other"]


def classify_palette(pal):
    cls = np.full(len(pal), C_OTHER, np.int32)
    for i, (r, g, b) in enumerate(pal):
        if r == 0 and g == 0 and b == 0: c = C_BLACK
        elif r >= 219 and g >= 219 and b >= 170: c = C_WHITE
        elif r == 255 and g < 100 and b < 100: c = C_RED
        elif b >= 170 and g >= 146 and r <= 36: c = C_WATER
        elif r <= 36 and g >= 219 and b <= 85: c = C_LAWN
        elif r == 0 and g <= 146 and b <= 85: c = C_DENSE
        elif r == 0 and g == 182 and b <= 85: c = C_UNDER
        elif r == 36 and g <= 109 and b <= 85: c = C_UNDER
        elif r == 73 and g <= 73: c = C_UNDER
        elif r in (36, 73, 109) and g >= 146 and b <= 85: c = C_SCRUB
        elif r in (146, 182) and g <= 73: c = C_BUILD
        elif r == 109 and g == 109: c = C_WOOD
        elif r in (146, 182) and g in (109, 146): c = C_EARTH
        else: c = C_OTHER
        cls[i] = c
    return cls


def tm_dem():
    return np.load(P("dem", "mosaic.npy"))


def dem_sample(mos, E_tm, N_tm, order=1):
    E_tm = np.asarray(E_tm, float); N_tm = np.asarray(N_tm, float)
    scalar = E_tm.ndim == 0
    lon, lat = to_ll.transform(np.atleast_1d(E_tm), np.atleast_1d(N_tm))
    out = map_coordinates(mos, [(58.0 - np.asarray(lat)) * 3600.0, (np.asarray(lon) + 6.0) * 2400.0], order=order, mode='nearest')
    return float(out[0]) if scalar else out.reshape(E_tm.shape)


def save_rgb(arr, name):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(P("view", name))


CLASS_COLORS = np.array([
    [120, 220, 90],   # lawn
    [190, 200, 60],   # scrub
    [10, 70, 20],     # dense
    [60, 130, 60],    # under
    [40, 120, 255],   # water
    [170, 60, 30],    # build
    [210, 180, 120],  # earth
    [120, 100, 70],   # wood
    [255, 255, 255],  # white
    [0, 0, 0],        # black
    [255, 0, 0],      # red
    [255, 0, 255],    # other
], np.uint8)

# what a build step's `from common import *` takes: every public name above, and the site's, resolved on first use
__all__ = [name for name in globals() if not name.startswith("_")] + ["SITE", "CASTLE_E", "CASTLE_N"]
