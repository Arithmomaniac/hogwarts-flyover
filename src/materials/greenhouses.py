"""Greenhouse set pieces (material 7, aux 140-149): the potting sheds' stone and window glass."""
from materials import set_piece
from materials.transport import LIT_GLASS_P3

FAMILY = "greenhouses"
RANGES = ((7, 140, 149),)
MATERIALS = [
    set_piece(140, "shed-stone",
              note="the potting sheds' rubble stone",
              model="speckled", values={"alb": "vec3(0.2, 0.185, 0.155)", "lo": "0.75", "amp": "0.45", "freq": "1.7",
                                        "rise": "2.1", "spec": "0.", "shin": "32."},
              refl="vec3(0.2, 0.185, 0.155)"),
    set_piece(141, "shed-glass",
              note="the potting sheds' window panes, unlit at dusk",
              p3=LIT_GLASS_P3,
              refl="vec3(0.012)"),
]