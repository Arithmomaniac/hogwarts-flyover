"""Codes every family shares: the building kinds of build_meshes' fallback massing (material 5), boundary walls and hedges
(6), the glazing pass (8), and the set-piece codes 1-7 (7), whose shading is hand-written in frag5.glsl. Codes 2, 3, 8 and
9 drew the raymarched train and Willow, which the meshes replaced; they are left unregistered."""
from materials import Material, set_piece

FAMILY = "shared"
RANGES = ((5, 0, 15), (6, 0, 15), (7, 1, 9), (8, 0, 15))
MATERIALS = [
    # the building kinds of the mapped footprints, in mesh_source.BUILDING_KINDS order (aux = kind index)
    Material("building-thatch", 5, 0, note="thatched building"),
    Material("building-slate", 5, 1, note="slate-roofed building"),
    Material("building-glass", 5, 2, note="greenhouse glass, also the greenhouses' frames"),
    Material("building-hut", 5, 3, note="the hut kind, for a footprint the fallback massing builds"),
    Material("building-shack", 5, 4, note="the Shrieking Shack"),
    Material("building-stands", 5, 5, note="the pitch's stands"),
    Material("building-big", 5, 6, note="large building; also the greenhouses' dwarf walls"),
    Material("hedge", 6, 0, note="hedges; also leaves and planted beds"),
    Material("wall-stone", 6, 1, note="boundary-wall stone; also platform faces and garden walls"),
    Material("glazing", 8, 0, note="see-through glazing, blended over the shaded image in its own pass"),
    set_piece(1, "gold",
              note="the goal hoops' gold, brass fittings",
              refl="vec3(0.55, 0.38, 0.09)"),
    set_piece(4, "dark-timber",
              note="dark timber: hulls, gates, benches, fences, barrels",
              refl="vec3(0.045, 0.035, 0.025)"),
    set_piece(5, "rigging",
              note="rigging",
              refl="vec3(0.03, 0.025, 0.02)"),
    set_piece(6, "pale-stone",
              note="pale stone: the gate piers, coping",
              refl="vec3(0.21, 0.2, 0.185)"),
    set_piece(7, "white-marble",
              note="Dumbledore's white marble tomb",
              refl="vec3(0.62, 0.61, 0.58)"),
]
