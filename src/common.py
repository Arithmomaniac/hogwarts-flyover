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

# Chosen castle anchor in the TM frame (metres east/north of the reference point)
CASTLE_E, CASTLE_N = 560.0, 2480.0

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
