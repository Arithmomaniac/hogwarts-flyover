"""Whomping Willow set pieces (material 7, aux 120-139): the mesh Willow's own codes (shared codes 8-9 were its
raymarched predecessor's, retired with it)."""
import math

from materials import set_piece

FAMILY = "willow"
RANGES = ((7, 120, 139),)
# the bark's codes, each naming the direction its furrows run in (x east, y up, z north): 120 for upright wood, 124-131
# for lying wood every 22.5 degrees from east towards north (either way along it), 132-139 for wood rising at 35 degrees
# every 45 degrees from east towards north. The builder gives each piece of wood the code nearest its own direction
LYING, RISING = 124, 132
RISE = 0.7                                  # tan(35 degrees)


def _axis(aux):
    if aux == 120:
        return (0.0, 1.0, 0.0)
    a, y = ((aux - LYING) * math.pi / 8, 0.0) if aux < RISING else ((aux - RISING) * math.pi / 4, RISE)
    s = math.hypot(1.0, y)
    return (round(math.cos(a) / s, 6), round(y / s, 6), round(math.sin(a) / s, 6))


BARK_AXES = {aux: _axis(aux) for aux in (120, *range(LYING, RISING + 8))}
_AXIS_LIST = ", ".join(f"vec3({x!r}, {y!r}, {z!r})" for x, y, z in BARK_AXES.values())
# Old willow bark close up: long grey-brown plates, a few centimetres wide and ten or more times as long, in a net of deep
# furrows that run along the wood, part and meet again round each plate (a diamond-like network, with no islands).
# Cells of a jittered grid stretched along the wood's axis are the plates, their borders the furrows: a first pass finds
# the nearest cell's centre, a second the nearest border round it (the distance to the border is then continuous). Each
# plate takes its own shade and catches the light at its edges (the normal tilts across each furrow). The pattern fades
# to its mean (FAR_H, FAR_SHADE: what it averages to over the wood) before a pixel grows as wide as a plate, and is not
# computed past that. Grey lichen patches the plates; moss greens the faces that look up.
CELL_PER_M, STRETCH = 26.0, 0.08        # cells per metre across the wood; their length along it is 1/STRETCH times that
FURROW_E, FURROW_W = 0.03, 0.2          # the furrow's floor and the ramp up to the plate, in cells from the border
SHADE_LO, SHADE_AMP = 0.86, 0.24        # each plate's shade, uniform over [lo, lo + amp]
FAR_H, FAR_SHADE = 0.435, SHADE_LO + SHADE_AMP / 2
# each centre keeps within JITTER_AMP / 2 of its cell's middle, so no two plates touch unless their cells are neighbours
# and the 3x3x3 search round the nearest centre always finds the border between them
JITTER_AMP = 0.7
_JITTER = ("vec3 hq = fract(q * vec3(0.1031, 0.103, 0.0973)); hq += dot(hq, hq.yxz + 33.33); "
           f"vec3 r = o + 0.5 + {JITTER_AMP!r} * (fract((hq.xxy + hq.yxx) * hq.zyx) - 0.5) - f;")
BARK_P3 = (" {                             // the Willow's bark: long plates in a net of deep furrows along the wood\n"
           f"        const vec3 AX[{len(BARK_AXES)}] = vec3[{len(BARK_AXES)}]({_AXIS_LIST});\n"
           f"        vec3 d = AX[k == 120 ? 0 : k - {LYING - 1}], e1 = normalize(cross(d, abs(d.y) > 0.9 ? vec3(1., 0., 0.) : vec3(0., 1., 0.))), e2 = cross(d, e1);\n"
           f"        float aa = smoothstep(0.03, 0.012, pix), h = {FAR_H!r}, shade = {FAR_SHADE!r};\n"
           "        if (aa > 0.) {\n"
           f"          vec3 S = vec3(dot(p, e1), dot(p, d) * {STRETCH!r}, dot(p, e2)) * {CELL_PER_M!r}, c = floor(S), f = S - c, m1 = vec3(9.), id = c, mo = vec3(0.);\n"
           "          for (int i = ZERO; i < 27; i++) {                // the nearest plate's centre\n"
           f"            vec3 o = vec3(float(i % 3), float((i / 3) % 3), float(i / 9)) - 1., q = c + o; {_JITTER}\n"
           "            if (dot(r, r) < dot(m1, m1)) { m1 = r; id = q; mo = o; }\n"
           "          }\n"
           "          float e = 9.; vec3 u = vec3(1., 0., 0.);\n"
           "          for (int i = ZERO; i < 27; i++) {                // the nearest border round it\n"
           f"            vec3 o = mo + vec3(float(i % 3), float((i / 3) % 3), float(i / 9)) - 1., q = c + o; {_JITTER}\n"
           "            vec3 b = r - m1;\n"
           "            if (dot(b, b) > 1e-6) { float eb = dot(0.5 * (m1 + r), normalize(b)); if (eb < e) { e = eb; u = normalize(b); } }\n"
           "          }\n"
           f"          float w = clamp((e - {FURROW_E!r}) / {FURROW_W!r}, 0., 1.);   // across the furrow's wall\n"
           f"          h = mix({FAR_H!r}, w * w * (3. - 2. * w), aa);\n"
           f"          shade = mix({FAR_SHADE!r}, {SHADE_LO!r} + {SHADE_AMP!r} * fract(sin(dot(id, vec3(12.9898, 78.233, 37.719))) * 43758.5453), aa);\n"
           f"          vec3 g = -(u.x * e1 + {STRETCH!r} * u.y * d + u.z * e2) * {CELL_PER_M * 6 / FURROW_W * 0.006!r} * w * (1. - w) * aa;\n"
           "          n = normalize(n - (g - n * dot(n, g)));\n"
           "        }\n"
           "        vec3 plate = vec3(0.289, 0.249, 0.196) * shade;\n"
           "        alb = mix(vec3(0.028, 0.023, 0.018), plate, h) * (0.85 + 0.3 * vnoise(p.xz * 0.6 + p.y * 0.45));\n"
           "        alb = mix(alb, vec3(0.11, 0.115, 0.09), 0.35 * h * smoothstep(0.62, 0.85, vnoise(p.xz * 1.4 + p.y * 1.1)));\n"
           "        alb = mix(alb, vec3(0.05, 0.07, 0.03), 0.45 * smoothstep(0.35, 0.9, n.y));\n"
           "      }")
BARK_REFL = "vec3(0.17, 0.147, 0.116)"
MATERIALS = [
    set_piece(120, "willow-bark",
              note="the Willow's bark on upright wood: grey-brown plates split by deep furrows",
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
