"""Hogsmeade set pieces (material 7, aux 60-99): the village's roofs, walls, trim, glazing and fittings. 96-99 are still
free."""
from materials import set_piece

FAMILY = "village"
RANGES = ((7, 60, 99),)


def glazing(pane_w, pane_h, note):
    """Glass with a dim room behind it and painted glazing bars every pane_w by pane_h metres, faded out once a bar
    is under a pixel; the sky shows in it at grazing angles, and lamplight at dusk where the code glows."""
    return (f" {{       // {note}: a dim room behind small panes; sky at grazing angles; lamplight at dusk\n"
            f"        vec2 g = vec2(fract(faceCoord(p, n, true) / {pane_w:g}), fract(p.y / {pane_h:g}));\n"
            f"        float bar = max(step(g.x, 0.1), step(g.y, 0.1)) * smoothstep(0.05, 0.015, pix);\n"
            f"        alb = mix(vec3(0.03, 0.028, 0.025), vec3(0.5, 0.49, 0.46), bar); spec = 0.6 * (1. - bar); shin = 400.;\n"
            f"        float fres = 0.04 + 0.96 * pow(1. - max(dot(n, -rd), 0.), 5.);\n"
            f"        emi += (1. - bar) * (fres * mix(uHorCol, uSkyCol, clamp(reflect(rd, n).y * 2. + 0.5, 0., 1.)) + lampGlow(7, sid) * uDusk);\n"
            f"      }}")


WINDOW_P3 = glazing(0.27, 0.31, "sash windows")
SHOP_P3 = glazing(0.62, 1.15, "shop windows")


def shared(name, table, body, note):
    """One shading branch for every code of `table` ({aux: (name, rgb, note)}): `body` reads the code's colour from
    the constant array `name`, so the page's shader holds one branch for them all, not one each, and every family's
    distinct bodies together stay within materials.MAX_P3_BRANCHES."""
    first, last = min(table), max(table)
    rgbs = ", ".join("vec3(" + ", ".join(f"{v:g}" for v in table.get(a, ("", (0.0, 0.0, 0.0)))[1]) + ")"
                     for a in range(first, last + 1))
    size = last - first + 1
    return f" {{ const vec3 {name}[{size}] = vec3[{size}]({rgbs}); {body.format(rgb=f'{name}[k - {first}]')} }}   // {note}"


HARLS = {63: ("harl-white", (0.56, 0.55, 0.51), "white harl"), 64: ("harl-cream", (0.55, 0.49, 0.36), "cream harl"),
         65: ("harl-ochre", (0.56, 0.42, 0.17), "ochre-yellow harl"), 66: ("harl-pink", (0.55, 0.35, 0.35), "pink harl"),
         67: ("harl-blue", (0.24, 0.35, 0.47), "blue harl"), 68: ("harl-green", (0.3, 0.42, 0.3), "green harl"),
         69: ("harl-terracotta", (0.44, 0.19, 0.11), "terracotta harl")}
GLOSSES = {71: ("margin-black", (0.025, 0.025, 0.028), "black-painted margins"),
           72: ("margin-oxblood", (0.16, 0.035, 0.03), "oxblood-painted margins"),
           73: ("door-green", (0.03, 0.1, 0.05), "green door"), 74: ("door-red", (0.26, 0.03, 0.025), "red door"),
           75: ("door-blue", (0.03, 0.07, 0.17), "blue door"), 76: ("door-black", (0.02, 0.02, 0.022), "black door"),
           81: ("fascia-green", (0.035, 0.1, 0.06), "green fascia and shopfront"),
           82: ("fascia-maroon", (0.17, 0.035, 0.045), "maroon fascia and shopfront"),
           83: ("fascia-navy", (0.03, 0.045, 0.12), "navy fascia and shopfront"),
           91: ("sign-cloth", (0.6, 0.58, 0.53), "the white cloth painted on the Hog's Head's sign"),
           95: ("clock-face", (0.62, 0.6, 0.55), "the white enamel faces of Dervish & Banges' bracket clock")}
HARL_P3 = shared("HARL", HARLS, "alb = {rgb} * (0.9 + 0.14 * vnoise(p.xz * 2.3 + p.y * 1.7));",
                 "harled walls, limewashed or painted: a soft speckle in each code's colour")
GLOSS_P3 = shared("GLOSS", GLOSSES, "alb = {rgb}; spec = 0.3; shin = 40.;", "gloss paint and enamel in each code's colour")


def harl(aux):
    """Harled (roughcast) wall, limewashed or painted: a soft speckle."""
    name, rgb, note = HARLS[aux]
    return set_piece(aux, name, note=note, p3=HARL_P3, refl="vec3(" + ", ".join(f"{v:g}" for v in rgb) + ")")


def gloss(aux):
    name, rgb, note = GLOSSES[aux]
    return set_piece(aux, name, note=note, p3=GLOSS_P3, refl="vec3(" + ", ".join(f"{v:g}" for v in rgb) + ")")


MATERIALS = [
    set_piece(60, "village-thatch",
              note="weathered thatch: grey-brown straw laid in courses down the slope",
              p3=(" {                             // weathered thatch: grey-brown straw streaked down the slope, patchy with age\n"
                  "        float u = faceCoord(p, n, false);\n"
                  "        float straw = vnoise(vec2(u * 9., p.y * 1.3)) * 0.6 + vnoise(vec2(u * 2.3, p.y * 0.4)) * 0.4;\n"
                  "        alb = vec3(0.118, 0.1, 0.078) * (0.62 + 0.55 * straw) * (0.85 + 0.2 * vnoise(p.xz * 0.27)); spec = 0.02; shin = 8.;\n"
                  "      }"),
              refl="vec3(0.118, 0.1, 0.078)"),
    set_piece(61, "village-thatch-new",
              note="newer thatch: golden-brown straw",
              p3=(" {                             // newer thatch: golden-brown straw streaked down the slope\n"
                  "        float u = faceCoord(p, n, false);\n"
                  "        float straw = vnoise(vec2(u * 9., p.y * 1.3)) * 0.6 + vnoise(vec2(u * 2.3, p.y * 0.4)) * 0.4;\n"
                  "        alb = vec3(0.24, 0.185, 0.11) * (0.68 + 0.5 * straw) * (0.88 + 0.14 * vnoise(p.xz * 0.27)); spec = 0.03; shin = 10.;\n"
                  "      }"),
              refl="vec3(0.24, 0.185, 0.11)"),
    set_piece(62, "village-slate",
              note="Scottish slate, laid in courses",
              p3=(" { alb = vec3(0.05, 0.056, 0.066) * (0.8 + 0.34 * hash12(vec2(floor(p.y / 0.17), floor(faceCoord(p, n, true) / 0.28))));"
                  " spec = 0.18; shin = 30.; }   // slate in courses"),
              refl="vec3(0.05, 0.056, 0.066)"),
    harl(63),
    harl(64),
    harl(65),
    harl(66),
    harl(67),
    harl(68),
    harl(69),
    set_piece(70, "rubble",
              note="rubble stone in rough courses",
              p3=(" {                             // rubble stone in rough courses\n"
                  "        float row = floor(p.y / 0.31);\n"
                  "        alb = vec3(0.2, 0.185, 0.16) * (0.7 + 0.5 * hash12(vec2(row, floor(faceCoord(p, n, true) / 0.47 + 0.5 * row))));\n"
                  "      }"),
              refl="vec3(0.2, 0.185, 0.16)"),
    gloss(71),
    gloss(72),
    gloss(73),
    gloss(74),
    gloss(75),
    gloss(76),
    set_piece(77, "glass-dark", note="window glass, unlit", p3=WINDOW_P3, refl="vec3(0.03, 0.028, 0.025)"),
    set_piece(78, "glass-lit", note="window glass lit at dusk", p3=WINDOW_P3, refl="vec3(0.03, 0.028, 0.025)",
              glow="${defs.DIMLAMP} * 0.2"),
    set_piece(79, "glass-dim", note="window glass dimly lit at dusk", p3=WINDOW_P3, refl="vec3(0.03, 0.028, 0.025)",
              glow="${defs.DIMLAMP} * 0.1"),
    set_piece(80, "glass-shop", note="shop windows, brightly lit at dusk", p3=SHOP_P3, refl="vec3(0.03, 0.028, 0.025)",
              glow="${defs.LAMP} * 0.25"),
    gloss(81),
    gloss(82),
    gloss(83),
    set_piece(84, "chimney-pot", note="clay chimney pots",
              p3=" alb = vec3(0.36, 0.16, 0.09) * (0.85 + 0.25 * vnoise(p.xz * 9.));   // clay chimney pots",
              refl="vec3(0.36, 0.16, 0.09)"),
    set_piece(85, "sign-board", note="painted hanging signs",
              p3=" alb = vec3(0.3, 0.25, 0.16) * (0.8 + 0.3 * hash12(floor(p.xz * 2.)));   // painted hanging signs",
              refl="vec3(0.3, 0.25, 0.16)"),
    set_piece(86, "plinth-dark", note="a dark-painted base course",
              p3=" alb = vec3(0.06, 0.06, 0.065) * (0.85 + 0.25 * vnoise(p.xz * 3.));   // dark-painted base course",
              refl="vec3(0.06, 0.06, 0.065)"),
    set_piece(87, "corrugated-iron",
              note="corrugated iron, rusting",
              p3=(" {                             // corrugated iron, rusting in patches\n"
                  "        float rib = mix(0.5, 0.5 + 0.5 * sin(faceCoord(p, n, true) * 82.7), smoothstep(0.05, 0.015, pix));\n"
                  "        alb = mix(vec3(0.2, 0.2, 0.2), vec3(0.24, 0.1, 0.045), smoothstep(0.35, 0.8, vnoise(p.xz * 0.9 + p.y * 0.6)))"
                  " * (0.78 + 0.32 * rib); spec = 0.22; shin = 30.;\n"
                  "      }"),
              refl="vec3(0.21, 0.15, 0.12)"),
    set_piece(88, "weatherboard",
              note="weathered, unpainted weatherboarding",
              p3=(" {                             // weathered, unpainted weatherboards, each a little uneven\n"
                  "        vec2 board = vec2(floor(p.y / 0.19), floor(faceCoord(p, n, false) / 2.9));\n"
                  "        alb = vec3(0.2, 0.175, 0.14) * (0.8 + 0.4 * hash12(board + 3.7)) * (0.76 + 0.24 * smoothstep(0., 0.3, fract(p.y / 0.19)));\n"
                  "      }"),
              refl="vec3(0.2, 0.175, 0.14)"),
    set_piece(89, "boarding",
              note="rough planks nailed over windows, and a garden fence",
              p3=(" {                             // rough grey planks, nailed up across windows\n"
                  "        float plank = floor(faceCoord(p, n, false) / 0.17);\n"
                  "        alb = vec3(0.15, 0.135, 0.115) * (0.78 + 0.36 * hash12(vec2(plank, floor(p.y / 1.7)))) * (0.82 + 0.18 * step(0.08, fract(faceCoord(p, n, false) / 0.17)));\n"
                  "      }"),
              refl="vec3(0.15, 0.135, 0.115)"),
    set_piece(90, "lamp-glass",
              note="the streetlamps' lanterns, alight at dusk",
              p3=(" { alb = vec3(0.42, 0.38, 0.27); spec = 0.5; shin = 200.; emi += lampGlow(7, sid) * uDusk; }"
                  "   // the streetlamps' lanterns, alight at dusk"),
              refl="vec3(0.42, 0.38, 0.27)",
              glow="${defs.LAMP} * 0.6"),
    gloss(91),
    set_piece(92, "rust-iron", note="rusty wrought iron",
              p3=(" { alb = vec3(0.2, 0.085, 0.04) * (0.7 + 0.5 * vnoise(p.xz * 13. + p.y * 7.)); spec = 0.08;"
                  " shin = 12.; }   // rusty wrought iron"),
              refl="vec3(0.2, 0.085, 0.04)"),
    set_piece(93, "glass-steamed",
              note="a teashop's windows steamed up from inside, lamplit at dusk",
              p3=(" { alb = vec3(0.5, 0.49, 0.47) * (0.88 + 0.16 * vnoise(p.xz * 11. + p.y * 9.)); spec = 0.12;"
                  " shin = 18.; emi += lampGlow(7, sid) * uDusk; }   // a teashop's windows, steamed up from inside"),
              refl="vec3(0.5, 0.49, 0.47)",
              glow="${defs.LAMP} * 0.2"),
    set_piece(94, "shop-blind",
              note="a shop's canvas blind in broad stripes",
              p3=(" {                             // a shop's canvas blind, striped down its slope\n"
                  "        float band = mix(0.5, step(0.5, fract(faceCoord(p, n, true) / 0.56)), smoothstep(0.2, 0.07, pix));\n"
                  "        alb = mix(vec3(0.06, 0.16, 0.1), vec3(0.6, 0.57, 0.48), band); spec = 0.05; shin = 8.;\n"
                  "      }"),
              refl="vec3(0.33, 0.37, 0.29)"),
    gloss(95),
]
