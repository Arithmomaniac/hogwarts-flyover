"""Grounds assets (hut and garden, pitch, gates, walls, hedges, tomb, harbour mouth, bridges), decisions and reference-image notes for the mesh scene manifest."""
import json
from pathlib import Path

from tools.manifest.common import CP, FOOTPRINT, G, GM, H, HM, asset, choice, decision, ft, labels, note, q

FAMILY = "grounds"
BUILT = {"hut", "hut-garden", "pitch", "gates", "grounds-walls", "field-boundaries", "tomb", "harbour-mouth", "bridges"}
CENSUS = {22: (-291.55, -191.53), 66: (-243.6, -535.64)}
STANDS = (59, 61, 62, 64, 67, 68, 69, 71)
GARDEN = Path(__file__).resolve().parents[1] / "data" / "grounds" / "garden-trace.json"
# look choices White Hound leaves open, which the user delegated to the lane without reviewing them
DELEGATED = {"source": "user", "quote": "I only want substantive issues, not visual reviews."}


def assets(ctx):
    cl = ctx.claims
    garden = json.loads(GARDEN.read_text(encoding="utf-8"))
    return [
        asset("hut", "grounds", "Hagrid's hut", [
            q(G, "It faces towards the castle"), q(G, "The floor-size of the house is around 35ft by 21ft."),
            q(G, "probably gabled, with a high ridge along the middle"),
            q(G, "The height of the roof varies from about 10ft up to probably about 15ft"),
            q(G, "So the low end(s) must be at the side."), q(G, "The house has a chimney, to one side."),
            q(G, "both probably about 5ft wide, and two windows each at front and back"),
            q(G, "The windows are divided into square panes"),
            q(G, "there is a rain-barrel by the right-hand window"),
            q(G, "But the barrel has to remain right of the front door."),
            q(G, "There is probably another barrel at the back, on the left"),
            q(G, "light spilling from Hagrid's window")],
            labels=labels(GM, "4a"), images=["cabin.jpg", "cabin_front.jpg"], inputs=FOOTPRINT,
            components=cl.comp(22, kinds=("hut",)),
            dimensions={"frontage_m": ft(35), "depth_m": ft(21), "eave_height_m": ft(10), "ridge_height_m": ft(15),
                        "door_width_m": ft(5), "window_width_m": ft(4), "window_sill_m": ft(3), "window_top_m": ft(11)},
            parts=["weatherboard walls", "gabled roof with a front-to-back ridge", "stovepipe chimney",
                   "front and back doors", "two sash windows front and back", "rain-barrels right of the front door and"
                   " back left", "gutter and downpipe"],
            materials=["creosoted weatherboard", "timber shingles"],
            decisions=["hut-plan", "timber-hut", "hut-front-gable", "hut-weatherboard-creosoted", "hut-roof-shingles",
                       "hut-stovepipe", "lamplit-windows"]),
        asset("hut-garden", "grounds", "Hagrid's garden and living fence", [
            q(G, "Behind the house there is a vegetable garden"), q(G, "must be at least 40ft by 60ft"),
            q(G, "a fence which has at least one growing tree incorporated into it"),
            q(G, "this fence is either incomplete or it has a gate in it")],
            labels=labels(GM, "4b"), images=["Beauxbatons.jpg"],
            traced=[{"id": "hut-garden", "frame": GM, "feature": "garden outline around the hut (4b)",
                     "plan_px": garden["plan_px"]}],
            dimensions={"min_width_m": ft(40), "min_length_m": ft(60)},
            parts=["post-and-rail fence with a gate", "growing tree in the fence line", "a dozen boulder-sized pumpkins",
                   "runner beans", "potato bed", "compost heap", "hen-coop"],
            decisions=["hut-plan", "garden-fence", "garden-contents"]),
        asset("pitch", "grounds", "Quidditch pitch, stands and changing rooms", [
            q(G, "The goalposts are golden hoops 50ft high, about 17ft apart and having hoops about 5ft across."),
            q(G, 'There is probably an "offside" strip about 50ft wide around the pitch proper.'),
            q(G, "tiered seating which is wooden and hollow"), q(G, "at least three tiers of seats, but probably only three"),
            q(G, "There is a barrier between the playing area and the stands, low enough to vault over."),
            q(G, "There is a wooden podium for the match commentator, probably placed midway along one of the long"
                 " sides."),
            q(G, "There are four changing rooms, one per house")],
            labels=labels(GM, "4c"), images=["Quidditch_pitch.jpg", "changing_rooms.jpg", "pitch_with_stands.jpg"],
            inputs=FOOTPRINT + ["meta.pitch"],
            components=cl.comp(*STANDS, kinds=("stands",)) + cl.comp(66, kinds=("slate",)),
            dimensions={"pitch_length_m": ft(500), "pitch_width_m": ft(180), "offside_strip_m": ft(50),
                        "goal_height_m": ft(50), "goal_spacing_m": ft(17), "hoop_diameter_m": ft(5), "tier_count": 3,
                        "changing_block_length_m": ft(85), "changing_block_depth_m": ft(18), "changing_room_count": 4,
                        "stair_count": 4, "stair_width_m": ft(15), "entrance_width_m": ft(10)},
            parts=["three-tier hollow timber stands", "barrier", "walkway", "stairs from the lawn", "commentator's podium",
                   "changing-room block", "goal hoops"],
            materials=["weathered planks on a dark timber frame", "gold hoops", "whitewashed boarding",
                       "dark green doors", "timber shingles"],
            decisions=["stands-full-ring", "pitch-goals-equal-height", "pitch-layout-wh", "pitch-stairs-from-lawn",
                       "pitch-changing-room-materials", "pitch-podium-clear"]),
        asset("gates", "grounds", "Winged-boar gates", [
            q(G, "The front gates are set into a wall and flanked on either side by tall stone pillars bearing figures"
                 " of winged boars."),
            q(G, "high, paired gates of wrought iron")],
            labels=labels(GM, "3g") + labels(HM, "1a"), inputs=["meta.extras.gates", "locA.road"],
            dimensions={"opening_m": ft(18), "pillar_height_m": 4.5, "pillar_width_m": 1.0, "gate_height_m": 3.0},
            parts=["paired gates", "tall stone pillars", "winged-boar figures", "wall up to the pillars"],
            materials=["wrought iron", "pale stone", "weathered stone (the boars)"],
            decisions=["gates-pillars-boars", "gates-closed"]),
        asset("grounds-walls", "grounds", "Boundary, cliff-edge and vegetable-patch walls", [
            q(G, "There is a low wall along part of the edge of the vegetable patch"),
            note(GM, "cliff-edge wall (2e), boundary wall (3a) and vegetable-patch wall (5d)"),
            note(HM, "boundary wall of the grounds (1b)")],
            labels=labels(GM, "2e", "3a", "5d") + labels(HM, "1b"), inputs=["locB.wall", "casA.low_wall"],
            castle_pieces=[cl.piece(-170.68, 120.53, "vegetable-patch wall (5d)"),
                           cl.piece(-176.97, 71.14, "vegetable-patch wall (5d)"),
                           cl.piece(85.75, 68.99, "cliff-edge wall (2e)"), cl.piece(94.81, -93.56, "cliff-edge wall (2e)"),
                           cl.piece(77.66, -169.16, "cliff-edge wall (2e)")],
            dimensions={"boundary_wall_height_m": 2.2, "safety_wall_height_m": 1.1, "ruin_min_height_m": 0.9,
                        "ruin_max_height_m": 1.4},
            parts=["coped boundary walls", "low safety wall along the cliff's edge", "the old curtain wall's low remnant"],
            materials=["rubble stone", "pale stone coping"],
            decisions=["walls-coped", "cliff-edge-safety-wall", "vegetable-patch-wall-ruin"]),
        asset("field-boundaries", "grounds", "Field hedges", [note("data", "decoded hedge lines")],
              inputs=["locA.hedge"], parts=["trimmed hedges with irregular crowns"],
              decisions=["hedge-heuristic", "hedge-crowns"]),
        asset("tomb", "grounds", "Dumbledore's tomb", [
            q(G, "The white marble tomb"), q(G, "so tall and/or so close to the water that it reflects in the lake")],
              labels=labels(GM, "2c"), inputs=["meta.extras.tomb"], materials=["white marble"],
              dimensions={"length_m": 3.4, "width_m": 2.0, "height_m": 1.78}, parts=["two steps", "plinth", "chest",
                                                                                     "ridged lid"],
              decisions=["tomb-chest"]),
        asset("harbour-mouth", "grounds", "Cliff opening to the underground harbour", [
            q(G, "a wide, ivy-curtained opening which leads into a water-filled channel which penetrates the"
                 " cliff-face")],
            labels=labels(GM, "1f") + labels(CP, "WG"), inputs=["meta.lake", "meta.extras.harbour"],
            traced=[{"id": "harbour-mouth", "frame": GM, "feature": "tunnel mouth (1f) near map px (1424, 944)",
                     "plan_px": [[1424.0, 944.0]]}],
            dimensions={"opening_width_m": 4.5, "crown_above_water_m": 2.0, "ivy_to_above_water_m": 1.0,
                        "recess_depth_m": 10.0},
            parts=["recessed cliff opening", "rock roof and closed back", "ivy curtain"],
            materials=["dark cliff rock", "ivy"], decisions=["harbour-local-cutout", "harbour-heads-down"]),
        asset("bridges", "grounds", "Bridges at mapped stream crossings", [
            note(GM, "crossings where mapped tracks meet streams"), note(HM, "crossings of the village watercourse")],
            inputs=["locB.stream", "locA.road"],
            traced=[{"id": "grounds-bridges", "frame": GM, "feature": "track and path crossings of mapped streams"},
                    {"id": "village-bridges", "frame": HM, "feature": "road crossings of the village watercourse"}],
            parts=["coped stone parapets", "arched openings over the water"], materials=["rubble stone", "pale stone"],
            inferred=["bridge construction"], decisions=["bridges-stone-arch"]),
    ]


def decisions(ctx):
    defaults, offered = ctx.defaults, ctx.offered
    return {
        "hedge-heuristic": decision(
            "Field boundaries follow the decoded hedge channel as low hedges.",
            note("data", "locA hedge channel"), inferred=True),
        "stands-full-ring": decision(
            "Stands run all the way round the pitch, interrupted only by the changing-room block and entrances.",
            q(G, "JK's map shows them running all the way round"),
            q(G, "The block of changing rooms probably interrupts the stands at that point")),
        "pitch-goals-equal-height": decision(
            "All six goal hoops stand 50 ft high.",
            q(G, "The goalposts are golden hoops 50ft high, about 17ft apart and having hoops about 5ft across.")),
        "pitch-layout-wh": decision(
            "The pitch follows White Hound's plan: three tiers, barrier, stairs, podium and changing block.",
            note(G, "pitch_with_stands, Quidditch_pitch and changing_rooms drawings"),
            images=["pitch_with_stands.jpg", "Quidditch_pitch.jpg", "changing_rooms.jpg"]),
        "pitch-stairs-from-lawn": decision(
            "Each of the four 15 ft stairs climbs from the lawn behind the stands to a landing at the top tier, and"
            " continues down through a 15 ft cut in the tiers to the walkway behind the barrier (user: lawn, as White"
            " Hound writes).",
            q(G, "The seating is accessed by stairs which are entered from the lawn, and which are widely spaced."),
            q(G, "the exit from the stair opens onto the lawn, not inside the stadium itself"),
            q(G, "four 15ft-wide stairs cutting their way through the stands"),
            images=["pitch_with_stands.jpg"]),
        "pitch-changing-room-materials": decision(
            "The changing block is clad in whitewashed weatherboards under a shingled roof, with boarded doors painted"
            " dark green, a window onto the pitch beside each room's door, and on the lawn side a window instead of a"
            " door where the bank stands above the floor (at X the ground is graded, so every room has the door White"
            " Hound draws onto the lawn). White Hound names no materials."
            " Decided by lane A under the user's delegation of look choices; not reviewed by the user.",
            DELEGATED, q(G, "lighted, with benches to sit on and at least one window per changing room"),
            q(G, "Ron strode across to the window and stared out into the rain"),
            q(G, "each with a captain's office attached, and opening directly onto the pitch"),
            images=["changing_rooms.jpg"], inferred=True),
        "pitch-podium-clear": decision(
            "The podium stands on four legs inside the barrier, near the middle of the castle side and beside the"
            " entrance rather than over it, well clear of the stands."
            " Decided by lane A under the user's delegation of look choices; not reviewed by the user.",
            DELEGATED,
            q(G, "There is a wooden podium for the match commentator, probably placed midway along one of the long"
                 " sides."),
            q(G, "It is probably not built into the stands, or right up close to them"),
            images=["pitch_with_stands.jpg"], inferred=True),
        "hut-plan": decision(
            "The hut is a broad timber house about 35 by 21 ft, facing the castle, gabled with low eaves at the sides.",
            q(G, "The floor-size of the house is around 35ft by 21ft."), q(G, "So the low end(s) must be at the side."),
            images=["cabin.jpg", "cabin_front.jpg"]),
        "timber-hut": decision(
            "The hut is broad and timber-clad, matching White Hound's drawing and dimensions.", defaults, offered,
            images=["cabin.jpg", "cabin_front.jpg"]),
        "hut-front-gable": decision(
            "The 35 ft front is a gable end: the ridge runs front to back, so the low eaves are at the sides, as White"
            " Hound's front elevation draws it. The doors are 11 ft high, level with the window heads he draws.",
            q(G, "So the low end(s) must be at the side."),
            q(G, "they must extend from about 3ft above ground to about 11ft, and are probably sash windows, about 4ft"
                 " wide"),
            images=["cabin_front.jpg", "cabin.jpg"], inferred=True),
        "hut-weatherboard-creosoted": decision(
            "The walls are overlapping horizontal weatherboards, creosoted dark and slightly irregular; White Hound"
            " allows creosote or whitewash.",
            q(G, "it is likely that the walls are weatherboarded (clad with overlapping horizontal planks)"),
            q(G, "Weatherboarding might be either creosoted or whitewashed"), images=["cabin_front.jpg"], inferred=True),
        "hut-roof-shingles": decision(
            "The roof is covered with weathered timber shingles; White Hound does not say what covers it.",
            q(G, "probably gabled, with a high ridge along the middle"), images=["cabin_front.jpg"], inferred=True),
        "hut-stovepipe": decision(
            "The chimney is a stovepipe through the roof over the front-left corner, as White Hound draws it.",
            q(G, "The chimney may be over a corner: if so, it is one of the corners at the front of the house."),
            images=["cabin_front.jpg"]),
        "garden-fence": decision(
            "A post-and-rail fence low enough to vault follows White Hound's traced garden outline (4b) from the back"
            " corners of the house, with a gate beside the house and a growing tree standing in its far side. The"
            " map's thin line round the garden is this fence, so no hedge is planted along it (hedge-heuristic had read"
            " it as one).",
            q(G, "a fence which has at least one growing tree incorporated into it"),
            q(G, "this fence is either incomplete or it has a gate in it"),
            q(G, "vaulted the fence into the pumpkin patch"), q(G, "tethered to a tree behind Hagrid's Pumpkin patch"),
            inferred=True),
        "garden-contents": decision(
            "The garden holds a dozen boulder-sized pumpkins near the back door, a row of runner beans, a potato bed, a"
            " compost heap and a hen-coop. Aragog's grave, dug for one night, is left out.",
            q(G, "a dozen of the largest pumpkins Harry had ever seen"), q(G, "Each was the size of a large boulder."),
            q(G, "a stand of runner beans, a potato bed, an enormous grave and a compost heap big enough for Hagrid to"
                 " collapse onto"),
            q(G, "There is also probably a hen-coop"),
            q(G, "Seen from Hagrid's back door, the pumpkin patch does not even extend as far as the farther limit of"
                 " the garden"),
            inferred=True),
        "harbour-local-cutout": decision(
            "Only a tightly bounded exterior recess with a closed-back shell is cut into the cliff; there is no"
            " boathouse or navigable tunnel. src/fix_harbour.py cuts the water-filled recess, 8 m wide and 12 m into"
            " the cliff at the cliff foot nearest the mapped mouth (1f), and the mesh roofs it with rock back to the"
            " cliff's surface, spans the cut's 4.5 m of open water with a low arch and closes it 10 m in.",
            q(G, "a wide, ivy-curtained opening which leads into a water-filled channel which penetrates the"
                 " cliff-face")),
        "harbour-heads-down": decision(
            "The opening is wide and low: a wide elliptical arch whose crown stands 2 m above the water, curtained by"
            " ivy hanging to 1 m above it, so the boats pass in under the ivy with heads bent.",
            q(G, "yelled Hagrid as the first boats reached the cliff"), q(G, "they all bent their heads"),
            q(G, "a curtain of ivy which hid a wide opening in the cliff face"), inferred=True),
        "bridges-stone-arch": decision(
            "Wherever a road or path runs between two pieces of a burn in the data, a small rubble-stone bridge carries"
            " it over: coped parapets 1 m high either side of the road, and an arch ringed in pale stone over the"
            " water's opening in each face. White Hound gives no construction; the stone arch is the Highland"
            " vernacular, decided by lane A under the user's delegation of look choices.",
            DELEGATED, q(H, "it's quite possible that some of the streets include bridges over a water-course"),
            q(H, "it probably does pass under the High Street rather than alongside it"), inferred=True),
        "gates-pillars-boars": decision(
            "Two tall pale-stone pillars stand either side of the road where the boundary wall, as drawn, opens at the"
            " mapped gap, 18 ft apart, with the wall run on from them into the boundary wall; each carries a winged"
            " boar of weathered stone, wings raised, facing out of the grounds. The leaves' bars follow the falling"
            " road down to the ground. White Hound gives no width, stone or facing."
            "Decided by lane A under the user's delegation of look choices; not reviewed by the user.",
            DELEGATED,
            q(G, "The front gates are set into a wall and flanked on either side by tall stone pillars bearing figures"
                 " of winged boars."),
            q(G, "They walked between the tall stone pillars topped with winged boars"), inferred=True),
        "gates-closed": decision(
            "The paired wrought-iron gates stand closed, as Harry finds them at night."
            "Decided by lane A under the user's delegation of look choices; not reviewed by the user.",
            DELEGATED,
            q(G, "high, paired gates of wrought iron"), q(G, "he found them chained shut"), inferred=True),
        "walls-coped": decision(
            "The boundary walls stand 2.2 m high, as the raymarched walls do, in rubble stone under a pale stone coping"
            " that overhangs them a little. White Hound gives no height or construction."
            "Decided by lane A under the user's delegation of look choices; not reviewed by the user.",
            DELEGATED,
            note(GM, "boundary wall (3a)"), inferred=True),
        "cliff-edge-safety-wall": decision(
            "The wall along the edge of the cliff overlooking the lake is a low (1.1 m) coped safety wall on the cliff's"
            " brink along the mapped footprint's lake-side edge, between the cliff-top path and the drop; the generic massing had"
            " built the footprint as a masonry block from the cliff's foot to the castle platform. Visible from the"
            " lake; reversible.",
            q(GM, "wall along edge of cliff overlooking lake"), q(GM, "path along edge of cliff, behind safety wall"),
            q(G, "may well have its own lower safety-wall along the edge")),
        "vegetable-patch-wall-ruin": decision(
            "The wall at the edge of the vegetable patch is the low remnant of the former curtain wall: 1 m thick along"
            " the middle of the map's broad ink, its top broken between 0.9 and 1.4 m. White Hound says it is low and"
            " perhaps a ruin, not how ruined."
            "Decided by lane A under the user's delegation of look choices; not reviewed by the user.",
            DELEGATED,
            q(G, "There is a low wall along part of the edge of the vegetable patch"),
            q(GM, "part of former curtain wall which once ran from the North Tower to Gryffindor Tower"), inferred=True),
        "hedge-crowns": decision(
            "The field hedges are trimmed to about 1.5 m with irregular crowns rather than flat tops, for a more"
            " natural look."
            "Decided by lane A under the user's delegation of look choices; not reviewed by the user.",
            DELEGATED, note("data", "locA hedge channel"), inferred=True),
        "tomb-chest": decision(
            "Dumbledore's tomb is a white marble chest tomb under a low-ridged lid, on a moulded plinth and two steps,"
            " 3.4 by 2 m and 1.8 m high, its long axis east-west. White Hound gives no form, only that it is white"
            " marble and tall or close enough to the water to reflect in it."
            "Decided by lane A under the user's delegation of look choices; not reviewed by the user.",
            DELEGATED,
            q(G, "The white marble tomb"), q(G, "so tall and/or so close to the water that it reflects in the lake"),
            inferred=True),
    }


IMAGE_NOTES = {
    "Loch_na_Tuadh.jpg": ("context", [], []),
    "Glenhead.jpg": ("context", [], []),
    "Scots_Pines.jpg": ("context", [], []),
    "mer_village.jpg": ("context", [], []),
    "Beauxbatons.jpg": ("schematic", ["hut at the north-east corner of a rectangular garden extending south"], []),
    "cabin.jpg": ("model", ["34 x 22 ft plan", "central front door with a window each side", "back door",
                            "chimney at the left end"], []),
    "cabin_front.jpg": ("model", ["broad low timber frontage", "barrel right of the front door"], []),
    "Quidditch_pitch_17thC.jpg": ("historical", [], ["historical pitch forms"]),
    "Quidditch_basket.jpg": ("historical", [], ["basket goals"]),
    "Quidditch_pitch.jpg": ("model", ["pointed-oval pitch with three equal-height hoops at each end"], []),
    "changing_rooms.jpg": ("model", ["single-storey block about 85 x 18 ft", "four rooms plus offices",
                                     "doors on both long sides"], []),
    "pitch_with_stands.jpg": ("model", ["three tiers of stands", "barrier", "four diagonal stairs",
                                        "north entrance gap", "commentator's podium",
                                        "changing block interrupting the south stands"], []),
}
