"""Grounds set pieces (material 7, aux 26-59): the hut and garden, then the pitch, gates, walls, tomb, harbour and
bridges."""
from materials import set_piece
from materials.village import MATERIALS as VILLAGE

RUBBLE = next(m for m in VILLAGE if m.name == "rubble")

FAMILY = "grounds"
RANGES = ((7, 26, 59),)
MATERIALS = [
    set_piece(26, "creosoted-weatherboard",
              note="Hagrid's hut: creosoted weatherboards",
              p3=(" {                             // Hagrid's hut: creosoted weatherboards, each a little uneven\n"
                  "        vec2 board = vec2(floor(p.y / 0.2), floor(faceCoord(p, n, false) / 3.1));\n"
                  "        alb = vec3(0.07, 0.05, 0.034) * (0.8 + 0.4 * hash12(board + 17.3)) * (0.78 + 0.22 * smoothstep(0., 0.3, fract(p.y / 0.2)));\n"
                  "      }"),
              refl="vec3(0.07, 0.05, 0.034)",
              refl_tail="                     // the hut's creosoted boards"),
    set_piece(27, "shingles",
              note="weathered timber shingles",
              p3=(" {                             // weathered timber shingles, laid in courses\n"
                  "        float c = floor(p.y / 0.11);\n"
                  "        alb = vec3(0.075, 0.068, 0.058) * (0.78 + 0.4 * hash12(vec2(c, floor(faceCoord(p, n, false) / 0.28 + 0.5 * c))));\n"
                  "      }"),
              refl="vec3(0.075, 0.068, 0.058)",
              refl_tail="                   // its shingles"),
    set_piece(28, "pumpkin",
              note="pumpkins",
              model="speckled", values={"alb": "vec3(0.55, 0.2, 0.025)", "lo": "0.85", "amp": "0.25", "freq": "2.3",
                                        "rise": "1.", "spec": "0.25", "shin": "30."},
              refl="vec3(0.55, 0.2, 0.025)",
              refl_tail="                      // pumpkins"),
    set_piece(29, "turned-earth",
              note="turned earth",
              model="speckled", values={"alb": "vec3(0.06, 0.045, 0.032)", "lo": "0.8", "amp": "0.4", "freq": "1.7",
                                        "rise": "0.", "spec": "0.", "shin": "32."},
              refl="vec3(0.06, 0.045, 0.032)",
              refl_tail="                    // turned earth"),
    set_piece(30, "weathered-planks",
              note="the hut's doors",
              p3=(" {                             // the hut's doors: weathered, unpainted upright planks\n"
                  "        float plank = floor(faceCoord(p, n, false) / 0.18);\n"
                  "        alb = vec3(0.16, 0.13, 0.09) * (0.82 + 0.3 * hash12(vec2(plank, 5.1))) * (0.85 + 0.15 * step(0.08, fract(faceCoord(p, n, false) / 0.18)));\n"
                  "      }"),
              refl="vec3(0.16, 0.13, 0.09)",
              refl_tail="                      // weathered door planks"),
    set_piece(31, "changing-room-boarding",
              note="the changing rooms: whitewashed weatherboards",
              p3=(" {                             // the changing rooms: whitewashed weatherboards, weathering unevenly\n"
                  "        float board = floor(p.y / 0.18);\n"
                  "        alb = vec3(0.6, 0.58, 0.53) * (0.9 + 0.12 * hash12(vec2(board, floor(faceCoord(p, n, false) / 2.7)))) * (0.8 + 0.2 * smoothstep(0., 0.25, fract(p.y / 0.18)));\n"
                  "      }"),
              refl="vec3(0.6, 0.58, 0.53)",
              refl_tail="                       // whitewashed boarding"),
    set_piece(32, "changing-room-door",
              note="the changing rooms' doors: boarded and painted dark green",
              p3=(" {                             // the changing rooms' doors: upright boards painted dark green\n"
                  "        float plank = floor(faceCoord(p, n, false) / 0.15);\n"
                  "        alb = vec3(0.035, 0.085, 0.045) * (0.88 + 0.2 * hash12(vec2(plank, 9.7))) * (0.85 + 0.15 * step(0.08, fract(faceCoord(p, n, false) / 0.15)));\n"
                  "        spec = 0.2; shin = 25.;\n"
                  "      }"),
              refl="vec3(0.035, 0.085, 0.045)",
              refl_tail="                    // green doors"),
    set_piece(33, "boar-stone",
              note="the winged boars on the gate pillars: weathered, lichened stone",
              p3=(" {                             // the boars' weathered stone, lichened where the rain lies\n"
                  "        float w = vnoise(p.xz * 6.1 + p.y * 4.3) * 0.65 + vnoise(p.xz * 17. - p.y * 13.) * 0.35;\n"
                  "        alb = vec3(0.19, 0.185, 0.165) * (0.55 + 0.7 * w) * (0.7 + 0.3 * smoothstep(-0.2, 0.6, n.y));\n"
                  "        alb = mix(alb, vec3(0.15, 0.16, 0.08), 0.6 * smoothstep(0.5, 0.75, vnoise(p.xz * 2.3 - p.y * 1.7)) * smoothstep(0., 0.5, n.y));\n"
                  "      }"),
              refl="vec3(0.17, 0.168, 0.15)",
              refl_tail="                    // the boars' stone"),
    set_piece(34, "harbour-rock",
              note="the harbour mouth's rock: the terrain's cliff rock (frag4.glsl rockAlb), so it matches the cliff "
                   "around it",
              p3=" { alb = rockAlb(p, n); spec = 0.05; shin = 20.; }",
              refl="vec3(0.1, 0.094, 0.084)",
              refl_tail="                   // the harbour mouth's rock"),
    set_piece(35, "ivy",
              note="ivy: dark glossy leaves, curtaining the harbour mouth",
              model="speckled", values={"alb": "vec3(0.022, 0.045, 0.014)", "lo": "0.6", "amp": "0.8", "freq": "9.",
                                        "rise": "7.", "spec": "0.3", "shin": "24."},
              refl="vec3(0.022, 0.045, 0.014)",
              refl_tail="                  // ivy"),
    set_piece(36, "arch-shadow",
              note="the dark under a bridge's arch",
              model="flat", values={"alb": "vec3(0.006, 0.006, 0.005)", "spec": "0.", "shin": "32."},
              refl="vec3(0.006, 0.006, 0.005)",
              refl_tail="           // the dark under an arch"),
    set_piece(37, "grounds-rubble",
              note="the grounds' walls, gate pillars and bridges: the village's rubble stone, in rough courses",
              p3=RUBBLE.p3, refl=RUBBLE.refl),         # the same body as the village's, so the same shading branch
    set_piece(38, "bridge-deck",
              note="a raised bridge deck's gravel, as the tracks'",
              model="speckled", values={"alb": "vec3(0.17, 0.155, 0.13)", "lo": "0.8", "amp": "0.35", "freq": "1.9",
                                        "rise": "0.", "spec": "0.", "shin": "32."},
              refl="vec3(0.17, 0.155, 0.13)",
              refl_tail="                // a bridge's deck"),
]
