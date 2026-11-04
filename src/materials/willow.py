"""Whomping Willow set pieces (material 7, aux 120-139): the mesh Willow's own codes (shared codes 8-9 were its
raymarched predecessor's, retired with it)."""
from materials import set_piece

FAMILY = "willow"
RANGES = ((7, 120, 139),)
MATERIALS = [
    set_piece(120, "willow-bark",
              note="the Willow's bark: grey-brown, ridged, greening on its upper faces",
              p3=(" {                             // the Willow's bark: grey-brown, ridged, greening where rain and lichen sit\n"
                  "        alb = vec3(0.085, 0.072, 0.058) * (0.7 + 0.45 * vnoise(p.xz * 4. + p.y * 1.3)) * (0.82 + 0.25 * vnoise(p.xz * 13. + p.y * 3.));\n"
                  "        alb = mix(alb, vec3(0.06, 0.075, 0.045), 0.45 * smoothstep(0.3, 0.9, n.y));\n"
                  "      }"),
              refl="vec3(0.08, 0.068, 0.055)",
              refl_tail="                     // the Willow's bark"),
    set_piece(121, "willow-knot",
              note="the knot on the Willow's trunk that stills it",
              model="speckled", values={"alb": "vec3(0.04, 0.03, 0.022)", "lo": "0.8", "amp": "0.3", "freq": "9.",
                                        "rise": "9.", "spec": "0.", "shin": "32."},
              refl="vec3(0.04, 0.03, 0.022)",
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
