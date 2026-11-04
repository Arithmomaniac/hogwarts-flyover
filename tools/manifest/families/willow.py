"""Whomping Willow asset, decisions and reference-image notes for the mesh scene manifest."""
from tools.manifest.common import G, GM, asset, choice, decision, labels, note, q

FAMILY = "willow"
BUILT = {"willow"}
CENSUS = {}


def assets(ctx):
    return [
        asset("willow", "willow", "Whomping Willow", [
            q(G, "thick, gnarled branches which flail like tentacles"), q(G, "probably start quite near the ground"),
            q(G, "It can clench its twigs like knuckles."),
            q(G, "as thick and gnarled as an ancient oak, but as flexible and whippy as a normal willow"),
            q(G, "The Whomping Willow stands alone in the grounds."),
            q(G, "the limbs of the oak resemble JKR's own sketch of the Whomping Willow to a remarkable degree")],
            labels=labels(GM, "5a"),
            images=["JKR_WW.gif", "St_Pierre_tree.jpg", "St_Pierre_tree2.jpg", "Coxley_Wick_willows.jpg",
                    "Golders_Green_willow.jpg"],
            inputs=["meta.extras.willow", "meta.extras.poi"],
            dimensions={"fork_height_m": 2.4, "trunk_radius_m": 0.95, "crown_reach_m": 11.0, "mouth_width_m": 0.75,
                        "mouth_height_m": 0.55, "knot_height_m": 0.4, "branch_clearance_m": 1.0},
            parts=["short thick trunk", "six or seven low sinuous limbs", "knuckles", "flared roots with a gap",
                   "finer asymmetric growth", "an arching root", "the knot", "leaves"],
            materials=["grey-brown ridged bark", "dark knot", "earthy tunnel mouth", "grey-green willow leaves"],
            decisions=["willow-root-gap", "willow-limbs", "willow-root-gap-size", "willow-arching-root", "willow-knot",
                       "willow-branch-clearance", "willow-crown", "willow-in-leaf", "willow-generator", "willow-levels"]),
    ]


def decisions(ctx):
    return {
        "willow-root-gap": decision(
            "The Willow's root gap faces the castle lawn and the hut.",
            q(G, "dragging Ron backwards into a large gap in the roots"),
            note(G, "orientation inferred from the approach routes"), images=["JKR_WW.gif"], inferred=True),
        "willow-limbs": decision(
            "A short, thick trunk splits about 2.4 m up into seven thick limbs, spread unevenly round it, each sinuous "
            "and bent at knuckles that bulge; finer branches and whippy twigs grow unevenly from them, their tips "
            "curling.",
            q(G, "The Willow has a thick trunk, and thick, gnarled branches which flail like tentacles"),
            q(G, "It can clench its twigs like knuckles."),
            q(G, "as thick and gnarled as an ancient oak, but as flexible and whippy as a normal willow"),
            images=["JKR_WW.gif", "St_Pierre_tree.jpg", "Golders_Green_willow.jpg"], inferred=True),
        "willow-root-gap-size": decision(
            "The tunnel mouth is 0.75 m wide and 0.55 m high, set into the trunk's undercut base between two "
            "buttress roots, and dives under the trunk.",
            q(G, "wide enough to admit a large dog dragging a tallish fourteen-year-old boy"),
            q(G, "small enough to be a tight squeeze for a young adult"),
            inferred=True),
        "willow-arching-root": decision(
            "The root beside the gap arches out of the ground.",
            q(G, "The roots are so gnarled that at least one evidently curls right out of the ground"),
            inferred=True),
        "willow-knot": decision(
            "The knot sits 0.4 m up the trunk, a little round from the gap.",
            q(G, "On the lower part of the trunk, at a height accessible to a large domestic cat or an agile rat"),
            inferred=True),
        "willow-branch-clearance": decision(
            "Beyond its roots no branch or leaf comes within 1 m of the ground.",
            q(G, "Nor do the lower branches hang right down to the ground."),
            q(G, "The Willow's lower branches are presumably not all that long"), inferred=True),
        "willow-crown": decision(
            "The crown reaches at most 11 m from the trunk and stands about 11 m high, inside the clearing the "
            "raymarched Willow keeps.",
            q(G, "The Whomping Willow stands alone in the grounds."),
            q(G, "From the greenhouses it is possible to see at least the upper part of the Whomping Willow"),
            inferred=True),
        "willow-in-leaf": decision(
            "The Willow is in leaf, as the rest of the scene is: grey-green leaf masses on the branches, light enough "
            "to show the knuckled limbs (the user accepted this cover). The winter reference photographs inform only "
            "its limbs.",
            choice("willow_foliage: Keep it as it is: light cover with the limbs showing, closest to the reference "
                   "trees"),
            note("user", "approved plan: do not turn winter reference photography into an unrequested season change"),
            images=["St_Pierre_tree.jpg", "St_Pierre_tree2.jpg", "Golders_Green_willow.jpg"]),
        "willow-generator": decision(
            "A seeded build-time branch generator (src/mesh_willow.py) builds the Willow, so the payload rebuilds "
            "byte for byte; it stays static.",
            note("user", "approved plan: generate distinctive branch structure rather than adding more triangles to "
                         "the existing blob; keep the Willow static"),
            q(G, "They planted the Whomping Willow the same year that I arrived at Hogwarts.")),
        "willow-levels": decision(
            "The same tree is also built at two coarser levels of detail (about a fifth and a twentieth of the full "
            "tree's triangles) and as a coarse shadow caster. The payload carries every level, and the page chooses "
            "among them for each view.",
            note("user", "approved plan: a build-time branch generator with levels of detail and a shadow mesh")),
    }


IMAGE_NOTES = {
    "JKR_WW.gif": ("model", ["short thick trunk splitting low into six or seven sinuous limbs", "flared roots"], []),
    "St_Pierre_tree.jpg": ("endorsed", ["low, wide, knuckled limbs"], ["a season change"]),
    "St_Pierre_tree2.jpg": ("endorsed", ["reclining trunk with limbs close to the ground"], ["a season change"]),
    "Coxley_Wick_willows.jpg": ("endorsed", ["very gnarled willow trunks"], ["a season change"]),
    "Golders_Green_willow.jpg": ("endorsed", ["gnarled upright trunk with asymmetric finer growth"],
                                 ["a season change"]),
}
