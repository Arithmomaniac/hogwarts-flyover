"""Castle assets, decisions and reference-image notes for the mesh scene manifest."""
from tools.manifest.common import C, CP, GM, asset, choice, decision, labels, note, q

FAMILY = "castle"
BUILT = {"castle"}                      # detailed geometry in the payload
CENSUS = {}                             # component id -> census position (x, z) from the provenance research
CASTLE_IMAGES = ["castle_Chepstow_above.jpg", "castle_Chepstow_river.jpg", "castle_Chepstow_water.jpg",
                 "castle_Chepstow_cliff.jpg", "castle_Caldicot_front.jpg", "castle_Caldicot_inside.jpg",
                 "castle_Mont_Orgueil.jpg", "castle_Alnwick.jpg", "castle_Craigievar_dark.jpg",
                 "castle_Craigievar_pale.jpg", "castle_Borthwick.jpg", "castle_Stirling.jpg", "castle_Edinburgh_dusk.jpg",
                 "castle_Edinburgh.jpg", "castle_Edinburgh_loch.jpg", "castle_Eilean_Donan.jpg", "castle_Duncraig.jpg",
                 "blockdiag_with_broomshed.jpg"]


def assets(ctx):
    cl = ctx.claims
    return [
        asset("castle", "castle", "Hogwarts castle", [
            q(C, "the three highest Towers – Ravenclaw, Astronomy and Gryffindor"),
            q(C, "both the North Tower and the Astronomy Tower have a long thin column topped by a wider, overhanging top"),
            q(C, "In the case of the Astronomy Tower this top is open to the elements"),
            q(C, "the North Tower is covered by some sort of onion dome"),
            q(C, "I have constructed Gryffindor Tower with a wide circular base"),
            q(C, "topped with three thinner turrets arranged in a trefoil shape"),
            q(C, "There are stairs leading up the front and side of the Head's tower, inside turrets."),
            q(C, "leads down to an enclosed corner of garden called the Physick Garden and Headmaster's Pleasaunce"),
            q(C, "containing (probably among many other things) the broomshed"),
            q(C, "that area of courtyard is grassed"),
            q(C, "the courtyard at Hogwarts is a complex shape, with secluded areas and corners you can walk round"),
            q(C, "at least fifteen windows on the top floor to the right of the tower"),
            note(CP, "registered plan of the named towers, wings, halls and broomshed court"),
            note("data", "castle-grid mass classes and the registered tower list")],
            labels=labels(CP, "AT", "GT", "NT", "RT", "WT", "NB", "GH", "EH", "EW", "WW", "HW", "HO", "FO", "Ch", "BS"),
            images=CASTLE_IMAGES,
            inputs=["casA.low_wall", "casA.wall", "casA.range", "casA.block", "casB.great_hall", "casB.keep",
                    "casB.tower1", "casB.tower2", "meta.castle", "meta.towers", "meta.named"],
            castle_pieces=[{"role": "main"}, {"role": "court"}, cl.piece(-58.89, 157.49, "wall sliver A"),
                           cl.piece(-12.73, 169.05, "wall sliver B"), cl.piece(61.91, 182.17, "wall stub C")],
            traced=[{"id": "plan-fit", "frame": CP, "feature": "similarity fit of the plan's named towers to meta.named",
                     "plan_px": {"NT": [188, 130], "RT": [178, 358], "WT": [368, 427], "AT": [510, 355],
                                 "GT": [618, 357], "GH": [532, 315], "EH": [487, 337], "KEEP": [257, 225]}},
                    {"id": "heads-tower", "frame": CP, "feature": "Head's tower (HO) at the end of the hospital wing, with"
                                                                    " front and side stair turrets", "plan_px": [316, 376]},
                    {"id": "physick-garden", "frame": CP, "feature": "enclosed garden corner below the Head's tower"
                                                                       " (inferred: the corner that the west block, the"
                                                                       " curtain wall and the hospital wing enclose beside"
                                                                       " the tower)", "plan_px": [303, 385]},
                    {"id": "broomshed-pend", "frame": CP, "feature": "pend from the front lawn into the broomshed court"
                                                                       " (BS), gated: the plan's dotted passage",
                     "plan_px": {"lawn": [370, 393], "court": [352, 393]}},
                    {"id": "front-steps", "frame": CP, "feature": "main doors at the south end of the Entrance Hall's"
                                                                     " front gap (483-492 px), nearest the Astronomy"
                                                                     " Tower, with steps up to them",
                     "plan_px": [490, 350]},
                    {"id": "court-buildings", "frame": GM, "feature": "fused court buildings, map-sweep ids 117, 56,"
                                                                       " 124, 77 and 75"}],
            parts=["connected wings", "court voids and court houses", "named towers", "roofs and parapets",
                   "battlements", "corbelled turrets", "recessed windows and doors", "entrances and stairs",
                   "North Battlements curtain wall"],
            materials=["warm brown-grey masonry", "slate roofs"],
            inferred=["arch forms", "roof pitches", "turret placement", "slate roofs", "window sizes, storeys and spacing",
                      "door and pend sizes", "the pend's gate bay", "the Head's tower's round form and height",
                      "the Physick Garden's place and layout", "court house sizes", "chimney stacks"],
            decisions=["wh-skyline", "towers-flat-battlemented", "warm-masonry", "castle-colour-red-cast",
                       "ravenclaw-battlements", "small-scottish-turrets", "court-buildings-as-houses",
                       "castle-norman-openings", "mont-orgueil-roofed-wall-walks", "castle-speck-threshold",
                       "castle-plan-interiors-excluded", "astronomy-beside-doors", "lamplit-windows"]),
    ]


def decisions(ctx):
    choices, defaults, offered = ctx.choices, ctx.defaults, ctx.offered
    return {
        "wh-skyline": decision(
            "Astronomy Tower is highest in complete world-space height, with Ravenclaw and Gryffindor next.",
            q(C, "the three highest Towers – Ravenclaw, Astronomy and Gryffindor"),
            q(C, "the Astronomy Tower, the highest of the castle")),
        "towers-flat-battlemented": decision(
            "Towers default to flat battlemented tops; conical roofs are avoided.",
            q(C, "French castle towers also had conical roofs, but this was never common in England where they usually"
                 " had flat tops."),
            q(C, "the West Tower definitely has a flat top and battlements, in the standard English/Welsh style"),
            images=["castle_Orford.jpg", "castle_Alnwick.jpg"]),
        "warm-masonry": decision(
            "Castle masonry is warm brown-grey: (0.26, 0.228, 0.19) in the shading pass, chosen against the earlier"
            " (0.25, 0.235, 0.21).", defaults, offered, choice(choices["masonry_colour"]),
            q(C, "Hogwarts, too, is quite a warm brown-grey, probably with a red cast in some lights"),
            images=["castle_Eilean_Donan.jpg", "castle_Duncraig.jpg"]),
        "astronomy-beside-doors": decision(
            "The Astronomy Tower keeps its plan position at the corner beside the main doors; its overhanging top is"
            " not stretched over them, which are seen nearly straight down from its parapet instead.",
            q(C, "The Astronomy Tower is so close to the main doors that it has an overhang which is actually above"
                 " them."),
            note(CP, "the plan draws the tower about 15 m along the front from the traced door point"),
            q(C, "you can see the front door, past the Astronomy tower"),
            choice(choices["astronomy_doors"])),
        "castle-colour-red-cast": decision(
            "The masonry may take a red cast in warm light, without becoming pink.",
            q(C, "probably with a red cast in some lights"), images=["castle_Caerlaverock.jpg"]),
        "ravenclaw-battlements": decision(
            "Ravenclaw Tower has battlements.", defaults, offered, inferred=True),
        "small-scottish-turrets": decision(
            "Small projecting Scottish-style corbelled turrets are used selectively.", defaults, offered,
            q(C, "might have begun as a plain block like Borthwick which was magically enhanced when the"
                 " Craigievar-style twiddly little turrets came into fashion"),
            images=["castle_Borthwick.jpg", "castle_Craigievar_dark.jpg", "castle_Craigievar_pale.jpg"]),
        "court-buildings-as-houses": decision(
            "Castle court pieces are domestic, house-like buildings with pitched roofs and chimneys against the curtain"
            " walls.", q(C, "domestic, house-like buildings and even gardens within a more stark curtain wall"),
            images=["castle_Caldicot_inside.jpg"]),
        "castle-norman-openings": decision(
            "Windows and doors are deep, Norman-style round-arched openings.",
            q(C, "The architectural style would be essentially Norman, as at Chepstow, literally towering above the"
                 " water."), images=["castle_Chepstow_above.jpg", "castle_Caldicot_front.jpg"], inferred=True),
        "mont-orgueil-roofed-wall-walks": decision(
            "Some wall-walks may be roofed, as a lacework of walls, courts and small buildings over uneven ground.",
            q(C, "set in a lacework of criss-crossing walls and courtyards and smaller buildings, draped over uneven"
                 " ground"), images=["castle_Mont_Orgueil.jpg"], inferred=True),
        "castle-plan-interiors-excluded": decision(
            "Castle-plan labels for rooms, corridors, statues and stairs are interiors and are not modelled.",
            choice(choices["detail"])),
        "castle-speck-threshold": decision(
            "Castle-grid pieces under 0.75 m² are resampling noise, not structures.",
            note("data", "the smallest real piece is a four-texel wall stub")),
    }


CASTLE_PARTIAL_COLOUR = ["warm brown-grey stone colour"]
CHEPSTOW_AVOID = ["ruin damage and roofless shells", "ivy"]
IMAGE_NOTES = {
    "castle_Orford.jpg": ("comparative", [], ["conical tower tops, which WH notes were rare in British castles"]),
    "castle_Hogwarts_WB.jpg": ("negative", [], ["the film castle's clusters of tall spires"]),
    "castle_Temple_de_Paris.jpg": ("negative", [], ["French conical-roofed towers"]),
    "castle_St-Fargeau.jpg": ("negative", [], ["French chateau conical roofs"]),
    "castle_Craigievar_dark.jpg": ("partial", ["small corbelled turrets with restrained caps"], ["pink harl"]),
    "castle_Balmoral.jpg": ("comparative", [], ["Victorian Baronial mansion massing"]),
    "castle_Neuschwanstein.jpg": ("negative", [], ["Bavarian Gothic-revival romanticism", "candle-snuffer spires"]),
    "castle_Disneyland.jpg": ("negative", [], ["fairy-tale spires"]),
    "castle_Alnwick.jpg": ("partial", ["warm heavy masonry", "flat battlemented towers", "crenellated curtain walls"],
                           ["copying Alnwick's specific plan"]),
    "castle_Neuschwanstein_winter.jpg": ("negative", [], ["Bavarian Gothic-revival romanticism",
                                                          "candle-snuffer spires"]),
    "castle_Torosay.jpg": ("comparative", [], ["Victorian Baronial house with conical-roofed towers"]),
    "castle_Coch.jpg": ("comparative", [], ["squat rebuilt towers with conical roofs"]),
    "St_Pancras.jpg": ("negative", [], ["the Victorian-Gothic railway-station look"]),
    "castle_Chepstow_above.jpg": ("endorsed", ["long multi-ward plan along a cliff", "walls rising from the rock",
                                               "drum and rectangular battlemented towers"], CHEPSTOW_AVOID),
    "St_Mungo_plaque.jpg": ("context", [], []),
    "castle_Craigievar_pale.jpg": ("partial", ["small corbelled turrets with restrained caps"], ["pink harl"]),
    "castle_Borthwick.jpg": ("partial", ["plain block massing for the oldest core"], ["a lone twin-block tower house"]),
    "castle_Stirling.jpg": ("partial", ["pitched-roof buildings on a crag inside low perimeter walls"],
                            ["copying Stirling's specific buildings"]),
    "castle_Bothwell.jpg": ("comparative", ["toy-fort court of curtain walls and towers"], ["ruin damage"]),
    "castle_Edinburgh_dusk.jpg": ("partial", ["village-like sprawl", "cliff-edge battlements"], ["night lighting"]),
    "castle_Cardoness.jpg": ("comparative", [], ["a lone tower house"]),
    "castle_Caldicot_front.jpg": ("endorsed", ["deep arched entrance", "square gatehouse massing"], []),
    "castle_Chepstow_river.jpg": ("endorsed", ["walls rising from the water's edge", "water entrance under the castle"],
                                  CHEPSTOW_AVOID),
    "castle_Edinburgh_loch.jpg": ("partial", ["layered rock-top massing above water"],
                                  ["geography inferred from a painting"]),
    "castle_Chepstow_water.jpg": ("endorsed", ["long low silhouette above the water", "battlemented towers"],
                                  CHEPSTOW_AVOID + ["sunset colour cast"]),
    "castle_Chepstow_cliff.jpg": ("endorsed", ["castle on a low cliff over water", "walls following the cliff edge"],
                                  CHEPSTOW_AVOID),
    "castle_Edinburgh.jpg": ("partial", ["tall ranges with regular windows", "crow-stepped gables and dormers",
                                         "low zig-zag battery walls"],
                             ["copying the specific palace or Half Moon Battery"]),
    "castle_Caerlaverock.jpg": ("comparative", [], ["definite pink stone", "moat", "ruin damage"]),
    "castle_Eilean_Donan.jpg": ("partial", CASTLE_PARTIAL_COLOUR, ["its silhouette"]),
    "castle_Duncraig.jpg": ("partial", CASTLE_PARTIAL_COLOUR, ["its silhouette"]),
    "castle_Caldicot_inside.jpg": ("endorsed", ["pitched-roof stone court houses with chimneys against the curtain wall",
                                                "grassed court"], ["half-timbered restoration"]),
    "castle_Mont_Orgueil.jpg": ("endorsed", ["lacework of criss-crossing walls and courts over uneven ground",
                                             "roofed wall-walks", "keep forming part of the perimeter"],
                                ["ruin damage"]),
    "blockdiag_with_broomshed.jpg": ("schematic", ["pend into the broomshed court"], []),
}
