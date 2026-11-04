"""Sun and horizon tests from White Hound's setting essay, ported from the location review (2026-10-31).

Solar positions use the NOAA algorithm with Saemundsson refraction and the upper limb; a delay is the visible event over
the real horizon minus the official event on a level sea horizon at the same coordinates. Horizons are traced every
degree of true azimuth out to 30 km, with earth curvature, refraction (k = 0.13) and the TM grid's meridian convergence.

Viewpoints: the castle's front steps (anchor + (-93, -80) m, eye 2 m above the platform, lake + 120 ft) for the sunsets,
sunrises and skylines; the Head's office (the anchor, eye lake + 100 yd) for the midsummer sunrise.
"""
import numpy as np

from tools.site import terrain as T

R_EARTH, K_REFR = 6371000.0, 0.13
FT, YD = 0.3048, 0.9144
CLIFF = 120 * FT                    # grounds essay: the cliff is about 120 ft above the lake
OFFICE = 100 * YD                   # setting essay: Dumbledore's office about 100 yd above the lake
SEMI = 0.267                        # solar semidiameter, degrees
FRONT = np.array([-93.0, -80.0])
EVENTS = {
    "set_0606": ("1994-06-06", "set", "front"),      # "So the sun in fact set half an hour early"
    "set_0621": ("1996-06-21", "set", "front"),      # about 40-45 minutes early
    "rise_0331": ("1998-03-31", "rise", "front"),    # "retarded by about forty-five minutes"
    "rise_0504": ("1998-05-04", "rise", "front"),    # about 45 minutes late
    "rise_0622": ("1996-06-22", "rise", "office"),   # seen rising over mountains from Dumbledore's office
}
AZ = np.arange(0.0, 360.0, 1.0)
DIST = np.concatenate([np.arange(150, 2000, 30.0), np.arange(2000, 8000, 60.0), np.arange(8000, 30001, 200.0)])


def sun_altaz(t_utc, lat, lon):
    """Geometric solar elevation and azimuth (degrees, azimuth clockwise from true north) at UTC datetimes."""
    t = np.asarray(t_utc, dtype="datetime64[s]")
    jd = (t - np.datetime64("1970-01-01T00:00:00")).astype(float) / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    L0 = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360)
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    Mr = np.radians(M)
    C = np.sin(Mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + np.sin(2 * Mr) * (0.019993 - 0.000101 * jc) \
        + np.sin(3 * Mr) * 0.000289
    omega = 125.04 - 1934.136 * jc
    app_long = L0 + C - 0.00569 - 0.00478 * np.sin(np.radians(omega))
    eps = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60 \
        + 0.00256 * np.cos(np.radians(omega))
    dec = np.degrees(np.arcsin(np.sin(np.radians(eps)) * np.sin(np.radians(app_long))))
    y = np.tan(np.radians(eps / 2)) ** 2
    L0r = np.radians(L0)
    eqt = 4 * np.degrees(y * np.sin(2 * L0r) - 2 * e * np.sin(Mr) + 4 * e * y * np.sin(Mr) * np.cos(2 * L0r)
                         - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * Mr))
    minutes = (jd + 0.5 - np.floor(jd + 0.5)) * 1440.0
    tst = np.mod(minutes + eqt + 4 * lon, 1440)
    ha = np.where(tst / 4 < 0, tst / 4 + 180, tst / 4 - 180)
    latr, decr, har = np.radians(lat), np.radians(dec), np.radians(ha)
    cz = np.clip(np.sin(latr) * np.sin(decr) + np.cos(latr) * np.cos(decr) * np.cos(har), -1, 1)
    zen = np.arccos(cz)
    caz = np.clip((np.sin(latr) * cz - np.sin(decr)) / (np.cos(latr) * np.sin(zen)), -1, 1)
    azd = np.degrees(np.arccos(caz))
    az = np.where(ha > 0, np.mod(azd + 180, 360), np.mod(540 - azd, 360))
    return 90 - np.degrees(zen), az


def refraction(h):
    """Atmospheric refraction (degrees) to add to a true altitude h (Saemundsson). The formula is only meaningful near
    the horizon: it spikes where the sun passes 5.11 degrees down in civil twilight, and those spikes would read as
    sunrises or sunsets, so altitudes below -2 degrees take the value at -2 (no visible event comes from there)."""
    h = np.maximum(np.asarray(h, float), -2.0)
    return 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11))) / 60.0


def _crossing(t, f, kind):
    pos = f > 0
    idx = np.nonzero(~pos[:-1] & pos[1:])[0] if kind == "rise" else np.nonzero(pos[:-1] & ~pos[1:])[0]
    if len(idx) == 0:
        return None
    k = idx[0] if kind == "rise" else idx[-1]
    return t[k] + np.timedelta64(int(round(f[k] / (f[k] - f[k + 1]) * 30)), "s")


def rise_set(day, lat, lon, kind, hor_az=None, hor_alt=None):
    """UTC times of sunrise or sunset on `day` ('YYYY-MM-DD'): (official on a sea horizon, visible over the given
    horizon profile, the sun's azimuth then)."""
    t0 = np.datetime64(day + "T00:00:00") + (np.arange(0, 1440 * 2) * 30).astype("timedelta64[s]")
    h, az = sun_altaz(t0, lat, lon)
    upper = h + refraction(h) + SEMI
    off = _crossing(t0, upper, kind)
    if hor_az is None:
        return off, None, None
    loc = _crossing(t0, upper - np.interp(np.mod(az, 360), hor_az, hor_alt, period=360), kind)
    laz = None if loc is None else float(sun_altaz(np.array([loc]), lat, lon)[1][0])
    return off, loc, laz


def convergence(E, N):
    """Meridian convergence (degrees) of the TM frame: grid bearing = true bearing - convergence."""
    lon, lat = T.ll(E, N)
    return np.asarray(T.tm_proj().get_factors(lon, lat).meridian_convergence)


def horizon(grid, E, N, eye, dist=DIST, az=AZ):
    """Apparent horizon altitudes (degrees) at true azimuths for viewpoints (E, N, eye height): (altitude [n, naz],
    distance of the occluding sample [n, naz])."""
    E, N, eye = np.atleast_1d(E), np.atleast_1d(N), np.atleast_1d(eye)
    gam = convergence(E, N)
    out = np.empty((len(E), len(az)), np.float32)
    dout = np.empty((len(E), len(az)), np.float32)
    drop = dist ** 2 * (1 - K_REFR) / (2 * R_EARTH)
    for s in range(len(E)):
        b = np.radians(az - gam[s])
        z = grid.sample(E[s] + np.sin(b)[:, None] * dist[None, :], N[s] + np.cos(b)[:, None] * dist[None, :])
        ang = np.degrees(np.arctan2(z - eye[s] - drop[None, :], dist[None, :]))
        ang = np.where(np.isnan(ang), -90, ang)
        k = np.argmax(ang, axis=1)
        out[s] = ang[np.arange(len(az)), k]
        dout[s] = dist[k]
    return out, dout


def _minutes(a, b):
    return (a - b).astype("timedelta64[s]").astype(float) / 60.0


def site_metrics(grid, E, N, Lv, keep_profiles=False):
    """Sun delays (minutes: sunsets early, sunrises late) and skyline heights (degrees) for anchors (TM) and lake
    levels, over the terrain `grid`."""
    E, N, Lv = map(np.atleast_1d, (E, N, Lv))
    hf, df = horizon(grid, E + FRONT[0], N + FRONT[1], Lv + CLIFF + 2.0)
    ho, _ = horizon(grid, E, N, Lv + OFFICE)
    lon, lat = T.ll(E, N)
    out = {k: np.full(len(E), np.nan) for k in EVENTS}
    for s in range(len(E)):
        for k, (day, kind, view) in EVENTS.items():
            off, loc, _ = rise_set(day, lat[s], lon[s], kind, AZ, hf[s] if view == "front" else ho[s])
            out[k][s] = 600.0 if loc is None else (_minutes(off, loc) if kind == "set" else _minutes(loc, off))
    north = (AZ >= 330) | (AZ <= 20)
    out["north_max"] = hf[:, north].max(1)
    out["south_max"] = hf[:, (AZ >= 135) & (AZ <= 200)].max(1)
    out["nw_face"] = hf[:, (AZ >= 305) & (AZ <= 320)].max(1)
    if keep_profiles:
        out.update(profile_front=hf, profile_dist=df, profile_office=ho)
    return out
