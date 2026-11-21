"""The Forest's tree species kit: its asset, decisions and quotations for the mesh scene manifest. The kit places
nothing; the renderer places its variants by the Forest's own rule (common.TREE_*, frag1.glsl treeDens and treeH)."""
from tools.manifest.common import L, asset, decision, note, q

FAMILY = "trees"
BUILT = set()
CENSUS = {}
IMAGE_NOTES = {}
F = "wh-forest"


def assets(ctx):
    return [
        asset("forest-trees", "trees", "The Forest's trees", [
            q(F, "most areas of the Forest are primarily deciduous"),
            q(F, "except for the swathe of pine trees between the station and the lake"),
            q(F, "Its trees where it meets the Hogwarts grounds are mostly well-grown and dark")],
            parts=["Scots pines, mature and young", "yews", "oaks", "beeches", "birches",
                   "three levels of detail per variant"],
            materials=["each species' bark, turning up the stem on pines and birches", "each species' leaves"],
            decisions=["trees-species", "trees-birch", "trees-form", "trees-in-leaf", "trees-colours",
                       "trees-generator", "trees-reach"]),
    ]


def decisions(ctx):
    return {
        "trees-species": decision(
            "The Forest's trees are Scots pines, mature (as in the swathe of pines between the station and the lake) "
            "and young (as in Grawp's stand of small pines); yews; oaks; and beeches, with birches as a third, lesser "
            "broadleaf (trees-birch).",
            q(F, "most areas of the Forest are primarily deciduous"),
            q(F, "except for the swathe of pine trees between the station and the lake"),
            q(F, "The wood at this point is apparently deciduous (the beech tree certainly is) and very dense."),
            q(F, "they encounter oak trees and a carpet of fallen leaves (not needles)"),
            q(F, "we get our first glimpse of conifers, in the form of two elderly yews"),
            q(F, "to come to an area of small pine trees")),
        "trees-birch": decision(
            "Birches, the commonest broadleaf of the site's Highland woods, are a fifth of the broadleaves. White "
            "Hound names oaks and beeches, and remarks that the books never mention birch; the site fills that gap.",
            q(F, "even where deciduous woods occur they are primarily silver birch: one type of deciduous tree which "
                 "has never been mentioned in relation to Hogwarts"),
            q(L, "Deciduous ones are usually heavily biased towards birch, alder and ash."),
            note("user", "a landscape choice that contradicts no quoted book evidence: the books name no tree against "
                         "it, so under the user's rule this departure from White Hound's conclusion needs no decision"),
            inferred=True),
        "trees-form": decision(
            "The trees are well grown and dark. Their trunks show under crowns that part unevenly, so light comes "
            "through them; a mature pine's crown is high and flat-topped on a bare stem, while the yews and the young "
            "pines keep their leaves almost to the ground.",
            q(F, "Its trees where it meets the Hogwarts grounds are mostly well-grown and dark"),
            q(F, "Towards the edge, however, they are wider-set, so that there is a visible difference in the amount of "
                 "light getting through them"),
            q(F, "yews do grow quite close to the ground"), inferred=True),
        "trees-in-leaf": decision(
            "Every tree is in leaf, as the scene is; no broadleaf has turned.",
            q(L, "we see that trees come into leaf very early at Hogwarts")),
        "trees-colours": decision(
            "Each species' colours are the linear albedo of its sunlit summer bark and leaves: dark greens, darkest "
            "on the yews; a pine's stem turning orange-red, and a birch's white above its dark foot.",
            q(F, "Its trees where it meets the Hogwarts grounds are mostly well-grown and dark"), inferred=True),
        "trees-generator": decision(
            "A seeded build-time generator (src/mesh_trees.py) builds two or three variants of each species, each at "
            "three levels of detail within the renderer's budgets, with its crown radius and height recorded. It "
            "places none: the renderer fits a variant to each of the Forest's placements, so every clearance holds.",
            note("user", "user's suggestion: a higher order helper for generating and placing trees, each of them a "
                         "mesh")),
        "trees-reach": decision(
            "The page draws the Forest's trees as meshes by default, each where the raymarched tree stood: its "
            "species variant to 70 m from the camera (200 m for a tree that stands apart, with at most 12 others "
            "within 25 m, as on a lawn), its drawn crown beyond to 2.6 km, and the raymarched trees' canopy sheet "
            "farther. ?trees=raymarched keeps the raymarched trees for comparison.",
            note("user", "chose the mesh trees as the page's default from the review sheets and frame times; this "
                         "changes how scope-exterior's ordinary trees are drawn, not where they stand")),
    }
