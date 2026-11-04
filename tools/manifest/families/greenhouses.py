"""Greenhouse assets, decisions and reference-image notes for the mesh scene manifest."""
from tools.manifest.common import FOOTPRINT, G, GM, asset, choice, decision, labels, note, q

FAMILY = "greenhouses"
BUILT = {"court-glasshouse", "greenhouses-north", "greenhouses-5b"}
CENSUS = {4: (-56.13, 245.01), 19: (-49.25, 61.37)}
GREENHOUSES_NORTH = (1, 2, 3, 5, 6, 8)
GREENHOUSES_5B = (9, 10, 12, 13, 14, 16, 17)


def assets(ctx):
    cl = ctx.claims
    return [
        asset("court-glasshouse", "greenhouses", "Small glasshouse in the castle's walled north court", [
            choice("court: glasshouse"), note(GM, "small T-shaped white building inside the walled north court")],
            inputs=FOOTPRINT, components=cl.comp(19, kinds=("glass",)),
            decisions=["court-glasshouse", "greenhouse-glass", "greenhouse-design-inferred"]),
        asset("greenhouses-north", "greenhouses", "Northern greenhouses and potting shed", [
            q(G, "There are at least three greenhouses, and are probably more, possibly plus potting sheds etc.."),
            note(GM, "greenhouse group (5b)")],
            labels=labels(GM, "5b"), inputs=FOOTPRINT,
            components=cl.comp(*GREENHOUSES_NORTH, kinds=("glass",)) + cl.comp(4, kinds=("thatch",)),
            parts=["thin frames", "tinted panes", "ridge roofs", "benches", "potted plants", "potting shed"],
            decisions=["greenhouse-glass", "greenhouse-design-inferred"]),
        asset("greenhouses-5b", "greenhouses", "Main greenhouse group and potting sheds", [
            q(G, 'We know that there are at least three greenhouses, because there is a "Greenhouse Three".'),
            note(GM, "greenhouse group (5b)")],
            labels=labels(GM, "5b"), inputs=FOOTPRINT + ["casA.low_wall", "casA.wall", "casA.range"],
            components=cl.comp(*GREENHOUSES_5B, kinds=("glass",)),
            castle_pieces=[cl.piece(28.62, 233.58, "potting shed"), cl.piece(45.74, 216.86, "potting shed")],
            parts=["thin frames", "tinted panes", "ridge roofs", "benches", "potted plants", "potting sheds"],
            decisions=["greenhouse-glass", "greenhouse-design-inferred", "castle-mass-sheds"]),
    ]


def decisions(ctx):
    choices = ctx.choices
    return {
        "greenhouse-glass": decision(
            "Greenhouse panes are see-through and tinted, with visible framing, benches and plants and no elaborate"
            " refraction.", choice(choices["greenhouse_glass"])),
        "court-glasshouse": decision(
            "The small white T-shaped building in the castle's walled north court is a small glasshouse, following the"
            " map's white-means-greenhouse convention.",
            choice(choices["court"]), note(GM, "white fill marks the greenhouses (5b)")),
        "castle-mass-sheds": decision(
            "The two detached castle-grid masses beside the 5b greenhouses are their potting sheds.",
            q(G, "There are at least three greenhouses, and are probably more, possibly plus potting sheds etc..")),
        "greenhouse-design-inferred": decision(
            "No greenhouse photograph exists in the inspected pages, so the glazing, frames, benches and plants are"
            " documented inference within the approved glass behaviour.",
            choice(choices["greenhouse_glass"]),
            note("book", "Chamber of Secrets ch. 6 places larger, more dangerous plants in Greenhouse Three"),
            inferred=True),
    }


IMAGE_NOTES = {}
