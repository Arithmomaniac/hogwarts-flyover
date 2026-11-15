"""Framing for the qualification's inspection: a close-up and a context camera for each object, and how much of the
object a rendered view shows and how much something in front of it hides.

A view is a camera and a target (tools.review.sheet's convention: east, height, north), with the hour it is shot at.
`rel` says what its heights are above: "own", the ground under each point, as the review sheets' views are written;
"target", the ground under the target for both, which keeps a framed camera's elevation angle wherever the ground
under it lies. Azimuths are compass bearings from the subject to the camera (0 north, 90 east), as the page gives the
sun's.

The page draws 1080 rows at a 50 degree vertical field of view (template_c.html FOV), so a subject `size` metres across
fills CLOSE_FILL of the frame's height from close_distance(size). The shot tool renders each candidate camera
(candidates: the sun behind the camera first, so the subject's lit side faces it, then every other side) and probes
its G-buffer on the framing test's grid (src/test_render_tour.py); `score` turns those samples into the share of the
rays through the subject's box that end on the subject (visible) and the share that end in front of it (occluded), and
`choose` takes the first candidate that shows the subject clearly.
"""
import math
from typing import NamedTuple

import numpy as np

FOV_DEG = 50.0                      # the page's vertical field of view (template_c.html FOV)
TAN_H = math.tan(math.radians(FOV_DEG / 2))
CLOSE_FILL = 0.6                    # a close-up's subject spans this share of the frame's height
CONTEXT_X = 3.0                     # a context view stands this many times as far back,
CONTEXT_MIN_M = 120.0               # and at least this far
CLOSE_MIN_M = 6.0                   # no close-up stands nearer than this
ELEVATIONS = (12.0, 30.0)           # degrees above the target: a close-up's, a context view's
MIN_VISIBLE = 0.1                   # a view shows its subject when this share of the rays through its box end on it
MAX_OCCLUDED = 0.15                 # ...and at most this share end on something in front of it
MARGIN_M = 1.0                      # a hit this far outside the box still counts as on it
LANDSCAPE = (0, 1, 2, 3)            # G-buffer materials that are no object: sky, ground, water, trees


class View(NamedTuple):
    label: str
    eye: tuple                      # (east, height, north)
    target: tuple
    hour: object                    # BST hours, or "civil dusk" / "night" (tools.review.sheet.hours_by_name)
    rel: str = "own"                # heights above the ground under each point ("own") or under the target ("target")

    @classmethod
    def from_sheet(cls, v):
        """A view in tools.review.sheet's VIEWS form: (label, x, height, z, target x, target height, target z, hour)."""
        label, x, y, z, tx, ty, tz, hour = v
        return cls(label, (x, y, z), (tx, ty, tz), hour, "own")


def close_distance(size, fill=CLOSE_FILL):
    """The distance at which a sphere `size` metres across spans `fill` of the frame's height."""
    return max(CLOSE_MIN_M, (size / 2) / math.tan(math.radians(fill * FOV_DEG / 2)))


def context_distance(size):
    return max(CONTEXT_MIN_M, CONTEXT_X * close_distance(size))


def candidates(sun_azimuth, elevation=ELEVATIONS[0]):
    """(azimuth, elevation) of the cameras to try, best first: from the sun's side, then turning away from it by 45
    degrees at a time, alternately either way, to the far side."""
    turns = (0, -45, 45, -90, 90, -135, 135, 180)
    return [((sun_azimuth + t) % 360.0, elevation) for t in turns]


def eye(target, azimuth, elevation, distance):
    """The camera point `distance` from `target` (east, height, north) on the compass bearing `azimuth`, `elevation`
    degrees above it."""
    a, e = math.radians(azimuth), math.radians(elevation)
    return (target[0] + distance * math.cos(e) * math.sin(a), target[1] + distance * math.sin(e),
            target[2] + distance * math.cos(e) * math.cos(a))


def aim(eye_point, target):
    """The page's (yaw, pitch) for a camera at `eye_point` looking at `target` (__hog.camera)."""
    d = [target[i] - eye_point[i] for i in range(3)]
    return math.atan2(d[0], d[2]), math.atan2(d[1], math.hypot(d[0], d[2]))


def framed(label, target, size, azimuth, hour, context=False, elevation=None):
    """A close-up (or context) view of a subject `size` across at `target` (east, height above its ground, north) from
    the compass bearing `azimuth`, `elevation` degrees above it (ELEVATIONS by default)."""
    distance = context_distance(size) if context else close_distance(size)
    if elevation is None:
        elevation = ELEVATIONS[1] if context else ELEVATIONS[0]
    return View(label, tuple(round(v, 2) for v in eye(target, azimuth, elevation, distance)), tuple(target), hour,
                "target")


def bearing(view):
    """The compass bearing from a view's target to its camera."""
    return math.degrees(math.atan2(view.eye[0] - view.target[0], view.eye[2] - view.target[2])) % 360.0


def rays(view, xy):
    """Unit world directions of the rays through internal pixels xy ((n, 2), x right, y up from the bottom row) of a
    probed view ({res, rot, tanH, jit}), as render_testing.ray gives them."""
    w, h = view["res"]
    u = (2 * (xy[:, 0] + 0.5 + view["jit"][0]) - w) / h * view["tanH"]
    v = (2 * (xy[:, 1] + 0.5 + view["jit"][1]) - h) / h * view["tanH"]
    d = np.stack([u, v, np.ones_like(u)], axis=1) @ np.array(view["rot"], float).reshape(3, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def ray_box(origins, dirs, lo, hi):
    """(entry, exit) distances of rays into an axis-aligned box (slabs); both inf where a ray misses it. A ray that
    starts inside enters at 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t0, t1 = (lo - origins) * inv, (hi - origins) * inv
    near = np.nanmax(np.where(np.isnan(t0), -np.inf, np.minimum(t0, t1)), axis=1)
    far = np.nanmin(np.where(np.isnan(t1), np.inf, np.maximum(t0, t1)), axis=1)
    hit = far >= np.maximum(near, 0.0)
    return np.where(hit, np.maximum(near, 0.0), np.inf), np.where(hit, far, np.inf)


def score(view, samples, box, margin=MARGIN_M):
    """(visible, occluded) shares of a probed view's rays through a subject's box (x0, y0, z0, x1, y1, z1): rays that end
    on an object inside the box, and rays that end on anything in front of the box. `samples` holds rows (x, y, hit
    distance, material) on the internal pixel grid; rays that pass through the box and end beyond it count as
    neither. A hit within `margin` of the box counts as on it."""
    samples = np.asarray(samples, float)
    lo, hi = np.array(box[:3], float), np.array(box[3:], float)
    d = rays(view, samples[:, :2])
    enter, leave = ray_box(np.repeat(np.array(view["pos"], float)[None], len(d), axis=0), d, lo, hi)
    through = np.isfinite(enter)
    if not through.any():
        return 0.0, 0.0
    t, mat = samples[:, 2], samples[:, 3].astype(int)
    front = through & (t < enter - margin)
    on = through & (t >= enter - margin) & (t <= leave + margin) & ~np.isin(mat, LANDSCAPE)
    n = float(through.sum())
    return float(on.sum()) / n, float(front.sum()) / n


def choose(scores, min_visible=MIN_VISIBLE, max_occluded=MAX_OCCLUDED):
    """The index of the first (visible, occluded) score that shows its subject clearly, else the one hiding least of
    it (by visible less occluded share)."""
    for k, (visible, occluded) in enumerate(scores):
        if visible >= min_visible and occluded <= max_occluded:
            return k
    return max(range(len(scores)), key=lambda k: (scores[k][0] - scores[k][1], -k))
