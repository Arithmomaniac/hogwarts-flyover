"""Transport set pieces: the train, track, station and ship (material 7, aux 10-25 and 100-119)."""
from materials import set_piece

FAMILY = "transport"
RANGES = ((7, 10, 25), (7, 100, 119))
# shared by 13, 14, 23
LIT_GLASS_P3 = (" {       // glass lit at dusk: the coaches' lamps, the station's dim ones; the ship's misty ports\n"
                "        alb = vec3(0.012); spec = 0.6; shin = 400.;\n"
                "        float fres = 0.04 + 0.96 * pow(1. - max(dot(n, -rd), 0.), 5.);\n"
                "        emi += fres * mix(uHorCol, uSkyCol, clamp(reflect(rd, n).y * 2. + 0.5, 0., 1.)) + lampGlow(7, sid) * uDusk;\n"
                "      }")
GREEN_PAINT = {"alb": "vec3(0.03, 0.075, 0.042)", "spec": "0.3", "shin": "40."}
MATERIALS = [
    set_piece(10, "scarlet",
              note="scarlet livery",
              model="flat", values={"alb": "vec3(0.33, 0.018, 0.012)", "spec": "0.35", "shin": "50."},
              refl="vec3(0.33, 0.018, 0.012)"),
    set_piece(11, "black-iron",
              note="black iron",
              model="flat", values={"alb": "vec3(0.022, 0.021, 0.02)", "spec": "0.25", "shin": "40."},
              refl="vec3(0.025, 0.022, 0.02)"),
    set_piece(12, "carriage-roof",
              note="carriage roofs",
              model="flat", values={"alb": "vec3(0.05, 0.05, 0.055)", "spec": "0.", "shin": "32."},
              refl="vec3(0.05, 0.05, 0.058)"),
    set_piece(13, "lamplit-glass",
              note="glass lit at dusk by the coaches' lamps",
              p3=LIT_GLASS_P3,
              refl="vec3(0.012)",
              glow="${defs.LAMP} * 0.2"),
    set_piece(14, "dim-glass",
              note="glass lit dimly at dusk: the station's oil lamps, the hut's windows",
              p3=LIT_GLASS_P3,
              refl="vec3(0.012)",
              glow="${defs.DIMLAMP} * 0.1"),
    set_piece(15, "rail",
              note="rails: polished heads, rusty sides",
              p3=(" {                             // rails: polished heads, rusty sides\n"
                  "        bool head = n.y > 0.7;\n"
                  "        alb = head ? vec3(0.3, 0.3, 0.31) : vec3(0.075, 0.045, 0.028); spec = head ? 0.7 : 0.05; shin = 90.;\n"
                  "      }"),
              refl="vec3(0.12, 0.1, 0.09)"),
    set_piece(16, "sleeper",
              note="sleepers; dark wood such as bark",
              model="flat", values={"alb": "vec3(0.05, 0.04, 0.03)", "spec": "0.", "shin": "32."},
              refl="vec3(0.025, 0.022, 0.02)"),
    set_piece(17, "ballast",
              note="track ballast",
              model="speckled", values={"alb": "vec3(0.15, 0.145, 0.135)", "lo": "0.75", "amp": "0.5", "freq": "3.1",
                                        "rise": "0.", "spec": "0.", "shin": "32."},
              refl="vec3(0.15, 0.145, 0.135)"),
    set_piece(18, "paving",
              note="platform paving",
              model="speckled", values={"alb": "vec3(0.11, 0.105, 0.1)", "lo": "0.85", "amp": "0.3", "freq": "0.9",
                                        "rise": "0.", "spec": "0.", "shin": "32."},
              refl="vec3(0.11, 0.105, 0.1)"),
    set_piece(19, "cream-boarding",
              note="cream-painted boarding and sashes",
              p3=" alb = vec3(0.5, 0.46, 0.37) * (0.94 + 0.06 * step(0.12, fract(faceCoord(p, n, true) / 0.16)));   // boarding",
              refl="vec3(0.5, 0.46, 0.37)"),
    set_piece(20, "slate",
              note="slate roofs",
              model="speckled", values={"alb": "vec3(0.045, 0.05, 0.06)", "lo": "0.85", "amp": "0.3", "freq": "0.6",
                                        "rise": "0.", "spec": "0.12", "shin": "24."},
              refl="vec3(0.05, 0.05, 0.058)"),
    set_piece(21, "green-door",
              note="green-painted doors",
              model="flat", values=GREEN_PAINT,
              refl="vec3(0.03, 0.075, 0.042)"),
    set_piece(22, "tunnel-dark",
              note="the tunnel's darkness",
              model="flat", values={"alb": "vec3(0.002)", "spec": "0.", "shin": "32."},
              refl="vec3(0.012)"),
    set_piece(23, "port-glass",
              note="the ship's misty ports",
              p3=LIT_GLASS_P3,
              refl="vec3(0.012)",
              glow="vec3(0.42, 0.5, 0.44) * 0.025"),
    set_piece(24, "sail",
              note="furled sails; also every unregistered set-piece code",
              p3=" alb = vec3(0.16, 0.14, 0.11);                                                          // furled sails",
              refl="vec3(0.16, 0.14, 0.11)",
              default=True),
    set_piece(25, "green-trim",
              note="green-painted trim",
              model="flat", values=GREEN_PAINT,
              refl="vec3(0.03, 0.075, 0.042)"),
]
