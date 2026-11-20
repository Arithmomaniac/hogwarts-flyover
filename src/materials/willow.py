"""Whomping Willow set pieces (material 7, aux 120-139): the mesh Willow's own codes (shared codes 8-9 were its
raymarched predecessor's, retired with it)."""
import math

from materials import set_piece

FAMILY = "willow"
RANGES = ((7, 120, 139),)
# the bark's codes, each naming the direction its furrows run in (x east, y up, z north): 120 for the trunk, 124-131
# for lying wood every 22.5 degrees from east towards north (either way along it), 132-139 for wood rising at 35 degrees
# every 45 degrees from east towards north. The builder gives every other piece of wood the code nearest its own direction
LYING, RISING = 124, 132
RISE = 0.7                                  # tan(35 degrees)
# the trunk's axis in the world, the bark's furrows wrap round: through its base at (x, height over the lake, z), leaning
# (dx, dz) metres per metre up. The shader takes the base's height from the lake's (LAKE), so the axis rises with the
# world; test_assets_willow checks it against the built tree, so an edit to the ground under the Willow fails there
TRUNK_AXIS = (-254.3407, 13.4888, 184.2352, -0.05386, -0.05938)


def _axis(aux):
    if aux == 120:
        return (0.0, 1.0, 0.0)
    a, y = ((aux - LYING) * math.pi / 8, 0.0) if aux < RISING else ((aux - RISING) * math.pi / 4, RISE)
    s = math.hypot(1.0, y)
    return (round(math.cos(a) / s, 6), round(y / s, 6), round(math.sin(a) / s, 6))


BARK_AXES = {aux: _axis(aux) for aux in (120, *range(LYING, RISING + 8))}
_AXIS_LIST = ", ".join(f"vec3({x!r}, {y!r}, {z!r})" for x, y, z in BARK_AXES.values())
# Old willow bark close up: long grey-brown ridges, a centimetre or two wide, that run along the wood, part and meet again
# in a diamond-like net round deep, dark furrows, each a spindle a few centimetres wide and ten or more times as long.
# Cells of a jittered grid stretched along the wood are the furrows, their borders the ridges: a first pass finds the
# nearest cell's centre, a second the nearest border round it (the distance to the border is then continuous). The
# cells' space wraps round the wood: across it, a circle of ROUND metres at the bearing of the trunk's axis (on the trunk)
# or of the surface's own normal (on every other piece of wood); along it, the code's direction. The furrows then follow
# the wood however it swells, flares, bends or leans from its code, where a space fixed in the world cuts its long cells
# crosswise into polygons wherever the surface tilts from the code's axis. Each pixel compares its cell coordinates with
# its neighbours' (in the G-buffer), and the pattern fades to its mean (FAR_H: what it averages to over the wood) before
# a pixel grows as wide as a ridge, as on twigs and where the wood faces along its own axis, and where the cells grow
# several times the trunk's (wood seen edge-on to its axis). The ridges share one colour (a shade per cell would jump
# wherever the nearest border changes, on the lit crests) and catch the light on their flanks (the normal tilts across
# each ridge). Broad value noise varies the wood, grey lichen patches the ridges, and moss greens the faces that look up.
CELL_PER_M, STRETCH = 14.0, 0.05        # cells per metre across the wood; their length along it is 1/STRETCH times that
ROUND = {120: 1.0, "wood": 0.36}        # the circle's radius round the trunk, and round every other piece of wood (a limb's)
# on the trunk the circle swells and shrinks a little, in helical lobes (depth as a fraction, lobes round, turns per metre
# up, phase), so the cells are cut at varying depths: their furrows taper, lean and interlace, as a trunk's ridges do
WOBBLE = ((0.06, 11, 0.9, 0.0), (0.045, 17, -1.3, 1.3))
# close up, the ridges show a grain of streaks a couple of centimetres wide and a few tens long, along the furrows; it
# varies their albedo by up to GRAIN either way about its mean, and fades out as a pixel grows from GRAIN_FADE[1] to [0]
GRAIN, GRAIN_FADE = 0.45, (0.016, 0.008)
RIDGE_E, RIDGE_W = 0.06, 0.12           # the ridge's crest each side of the border, and its flank down into the furrow, in cells
FAR_H = 0.54
FADE = (0.42, 0.168)                    # cells per pixel: the pattern fades out between these
COARSE = (0.1, 0.3)                     # and where it moves this fraction of the trunk's cells per pixel, or less
# each centre keeps within JITTER_AMP / 2 of its cell's middle, so no two plates touch unless their cells are neighbours
# and the 3x3x3 search round the nearest centre always finds the border between them
JITTER_AMP = 0.7
_JITTER = ("vec3 hq = fract(q * vec3(0.1031, 0.103, 0.0973)); hq += dot(hq, hq.yxz + 33.33); "
           f"vec3 r = o + 0.5 + {JITTER_AMP!r} * (fract((hq.xxy + hq.yxx) * hq.zyx) - 0.5) - f;")
_X0, _Y0, _Z0, _LX, _LZ = TRUNK_AXIS
_WOBBLE = "".join(f" + {a!r} * sin({n}. * th + {k!r} * hy + {ph!r})" for a, n, k, ph in WOBBLE)
_WOBBLE_T = "".join(f" + {round(a * n, 6)!r} * cos({n}. * th + {k!r} * hy + {ph!r})" for a, n, k, ph in WOBBLE)   # its slopes
_WOBBLE_Y = "".join(f" + {round(a * k, 6)!r} * cos({n}. * th + {k!r} * hy + {ph!r})" for a, n, k, ph in WOBBLE)
BARK_P3 = (" {                             // the Willow's bark: long ridges in a net round deep furrows along the wood\n"
           f"        const vec3 AX[{len(BARK_AXES)}] = vec3[{len(BARK_AXES)}]({_AXIS_LIST});\n"
           f"        vec3 d = AX[k == 120 ? 0 : k - {LYING - 1}], e1 = normalize(cross(d, abs(d.y) > 0.9 ? vec3(1., 0., 0.) : vec3(0., 1., 0.))), e2 = cross(d, e1);\n"
           f"        float R = k == 120 ? {ROUND[120]!r} : {ROUND['wood']!r}, rate = 0., h = {FAR_H!r};\n"
           "        vec3 S = vec3(0.), a = e1, W = vec3(1., 0., 0.); int got = 0; float grain = 1.;\n"
           "        for (int j = ZERO; j < 5; j++) {                  // this pixel, then its four neighbours\n"
           "          vec3 q = p, m = n;\n"
           "          if (j > 0) {\n"
           "#ifdef SPLIT_P3\n"
           "            ivec2 ip = ivec2(gl_FragCoord.xy) + ivec2(j == 1 ? 1 : j == 2 ? -1 : 0, j == 3 ? 1 : j == 4 ? -1 : 0);\n"
           "            vec4 b0 = texelFetch(tG0, ip, 0);\n"
           "            if ((int(b0.y) & 7) != 7 || b0.z < 119.5 || b0.z > 139.5 || abs(b0.x - t) > 0.02 * t + 0.05) continue;\n"
           "            vec2 uv = (2. * (vec2(ip) + 0.5 + uJitter) - uRes) / uRes.y;\n"
           "            q = ro + normalize(uCamRot * vec3(uv * uTanHalf, 1.)) * b0.x; m = texelFetch(tG1, ip, 0).xyz;\n"
           "#else\n"
           "            break;\n"
           "#endif\n"
           "          }\n"
           f"          vec3 o = k == 120 ? q - vec3({_X0!r}, LAKE + {_Y0!r}, {_Z0!r}) - vec3({_LX!r}, 1., {_LZ!r}) * (q.y - LAKE - {_Y0!r}) : m - d * dot(m, d);\n"
           "          o /= max(length(o), 1e-4);                         // the bearing round the wood\n"
           "          float wb = 1., wt = 0., wy = 0.;                 // the trunk's circle swells and shrinks\n"
           f"          if (k == 120) {{ float th = atan(o.z, o.x), hy = q.y - LAKE - {_Y0!r}; wb = 1.{_WOBBLE}; wt = 0.{_WOBBLE_T}; wy = 0.{_WOBBLE_Y}; }}\n"
           "          // along the wood from the trunk's base, so the pattern keeps its place on the wood at any datum\n"
           f"          vec3 sj = vec3(R * wb * dot(o, e1), dot(q - vec3({_X0!r}, LAKE + {_Y0!r}, {_Z0!r}), d) * {STRETCH!r}, R * wb * dot(o, e2)) * {CELL_PER_M!r};\n"
           "          if (j == 0) { S = sj; a = o; W = vec3(wb, wt, wy); } else { rate = max(rate, length(sj - S)); got++; }\n"
           "        }\n"
           f"        if (got == 0) rate = pix * {CELL_PER_M!r};\n"
           f"        float aa = smoothstep({FADE[0]!r}, {FADE[1]!r}, rate) * smoothstep({COARSE[0]!r}, {COARSE[1]!r}, rate / (pix * {CELL_PER_M!r}));\n"
           "        if (aa > 0.) {\n"
           "          vec3 c = floor(S), f = S - c, m1 = vec3(9.), mo = vec3(0.);\n"
           "          for (int i = ZERO; i < 27; i++) {                // the nearest furrow's centre\n"
           f"            vec3 o = vec3(float(i % 3), float((i / 3) % 3), float(i / 9)) - 1., q = c + o; {_JITTER}\n"
           "            if (dot(r, r) < dot(m1, m1)) { m1 = r; mo = o; }\n"
           "          }\n"
           "          float e = 9.; vec3 u = vec3(1., 0., 0.);\n"
           "          for (int i = ZERO; i < 27; i++) {                // the nearest border round it\n"
           f"            vec3 o = mo + vec3(float(i % 3), float((i / 3) % 3), float(i / 9)) - 1., q = c + o; {_JITTER}\n"
           "            vec3 b = r - m1;\n"
           "            if (dot(b, b) > 1e-6) { float eb = dot(0.5 * (m1 + r), normalize(b)); if (eb < e) { e = eb; u = normalize(b); } }\n"
           "          }\n"
           f"          float w = clamp((e - {RIDGE_E!r}) / {RIDGE_W!r}, 0., 1.);   // down the ridge's flank\n"
           f"          h = mix({FAR_H!r}, 1. - w * w * (3. - 2. * w), aa);\n"
           "          // the border's normal in the wood's own directions: round it, and along it (with the swelling's slopes)\n"
           "          float ur = u.x * dot(a, e1) + u.z * dot(a, e2), ua = W.x * (u.z * dot(a, e1) - u.x * dot(a, e2)) - W.y * ur;\n"
           f"          vec3 g = (ua * cross(d, a) + ({STRETCH!r} * u.y + R * W.z * ur) * d) * {CELL_PER_M * 6 / RIDGE_W * 0.006!r} * w * (1. - w) * aa;\n"
           "          n = normalize(n - (g - n * dot(n, g)));\n"
           f"          float fine = aa * smoothstep({GRAIN_FADE[0]!r}, {GRAIN_FADE[1]!r}, pix);   // close up, the ridges' grain\n"
           f"          grain = 1. + {GRAIN!r} * fine * (vnoise(vec2(S.x * 3. + S.z * 1.7, S.y * 2.5)) + vnoise(vec2(S.z * 3. - S.x * 1.7, S.y * 2.5 + 17.)) - 1.);\n"
           "        }\n"
           "        vec3 plate = vec3(0.238, 0.204, 0.161) * grain;\n"
           "        alb = mix(vec3(0.028, 0.023, 0.018), plate, h) * (0.85 + 0.3 * vnoise(p.xz * 0.6 + p.y * 0.45));\n"
           "        alb = mix(alb, vec3(0.11, 0.115, 0.09), 0.35 * h * smoothstep(0.62, 0.85, vnoise(p.xz * 1.4 + p.y * 1.1)));\n"
           "        alb = mix(alb, vec3(0.05, 0.07, 0.03), 0.45 * smoothstep(0.35, 0.9, n.y));\n"
           "      }")
BARK_REFL = "vec3(0.172, 0.148, 0.117)"
MATERIALS = [
    set_piece(120, "willow-bark",
              note="the Willow's bark on its trunk: grey-brown plates split by deep furrows",
              p3=BARK_P3, refl=BARK_REFL, refl_tail="                     // the Willow's bark"),
    set_piece(LYING, "willow-bark-lying", count=8,
              note="the Willow's bark on lying wood, its furrows along one of eight bearings",
              p3=BARK_P3, refl=BARK_REFL),
    set_piece(RISING, "willow-bark-rising", count=8,
              note="the Willow's bark on wood rising at 35 degrees, its furrows along one of eight bearings",
              p3=BARK_P3, refl=BARK_REFL),
    set_piece(121, "willow-knot",
              note="the knot on the Willow's trunk that stills it",
              model="speckled", values={"alb": "vec3(0.034, 0.029, 0.024)", "lo": "0.75", "amp": "0.4", "freq": "9.",
                                        "rise": "9.", "spec": "0.", "shin": "32."},
              refl="vec3(0.034, 0.029, 0.024)",
              refl_tail="                     // its knot"),
    set_piece(122, "willow-root-hole",
              note="the earthy tunnel mouth in the Willow's roots",
              model="flat", values={"alb": "vec3(0.012, 0.009, 0.007)", "spec": "0.", "shin": "32."},
              refl="vec3(0.012, 0.009, 0.007)",
              refl_tail="                  // the tunnel mouth"),
    set_piece(123, "willow-leaves",
              note="the Willow's leaves: narrow, grey-green, silvery underneath",
              p3=(" {                             // the Willow's leaves: grey-green sprays, silvery where they turn down\n"
                  "        alb = vec3(0.05, 0.072, 0.03) * (0.75 + 0.5 * vnoise(p.xz * 1.7 + p.y * 2.1));\n"
                  "        alb = mix(alb, vec3(0.09, 0.1, 0.08), 0.35 * smoothstep(0., -0.6, n.y));\n"
                  "        n = normalize(n + (vec3(vnoise(p.xz * 3.), vnoise(p.zy * 3.), vnoise(p.xy * 3.)) - 0.5) * 0.9); wrap = 0.35;\n"
                  "      }"),
              refl="vec3(0.05, 0.072, 0.03)",
              refl_tail="                      // its leaves"),
]
