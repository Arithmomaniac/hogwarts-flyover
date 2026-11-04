"""Castle codes (material 4). Their shading is hand-written in frag5.glsl's castle branch and template_b.html's
reflection albedo; the registry records them so no other family reuses them, and supplies the lamp glow of lit panes."""
from materials import Material

FAMILY = "castle"
RANGES = ((4, 0, 255),)
MATERIALS = [
    Material("castle-roof-type", 4, 0, count=10,
             note="roof type 0-9: what the mesh shader writes for a class face; never in the payload"),
    Material("castle-class-painted", 4, 10, count=6,
             note="castle mass class k with painted windows (10 + k): the fallback massing's code"),
    Material("castle-class-openings", 4, 20, count=6, note="castle mass class k with real openings (20 + k): no painted windows"),
    Material("castle-glass", 4, 40, note="a window pane, dark at dusk"),
    Material("castle-glass-lit", 4, 41, note="a pane lit at dusk by candlelight", glow="${defs.CANDLE} * 0.2"),
    Material("castle-oak", 4, 42, note="oak doors and gates"),
    Material("castle-reveal", 4, 43, note="the masonry lining an opening, recessed"),
    Material("castle-plain", 4, 44, note="exposed plain masonry: steps, chimney stacks"),
    Material("castle-glass-dim", 4, 45, note="a pane lit dimly at dusk", glow="${defs.CANDLE} * 0.1"),
    Material("castle-mesh-tower", 4, 150, count=50, note="mesh tower i (150 + i): its re-ranked top reaches P3 in ctop"),
]
