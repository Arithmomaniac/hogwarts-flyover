"""Hogsmeade assets, decisions and reference-image notes for the mesh scene manifest."""
import json
from pathlib import Path

from tools.manifest.common import FOOTPRINT, H, HM, asset, choice, decision, ft, labels, note, q

FAMILY = "village"
BUILT = set()
TRACED = Path(__file__).resolve().parents[1] / "data" / "village" / "traced.json"
CENSUS = {42: (-1313.48, -862.47), 52: (-1220.52, -899.78), 74: (-861.69, -914.59), 86: (-973.33, -1085.66),
          108: (-1114.83, -1229.71), 110: (-922.91, -1153.89), 125: (-866.62, -1159.66), 136: (-825.71, -1175.94),
          140: (-821.1, -1194.64), 143: (-763.25, -1172.53), 154: (-789.0, -1217.17), 183: (-754.68, -1258.73),
          316: (-540.6, -1282.7), 411: (-473.26, -1352.85), 517: (-454.74, -1438.39), 25: (-1086.51, -616.27),
          26: (-1061.96, -609.16), 29: (-1079.3, -629.85), 30: (-1060.15, -631.76)}
HAMLET = (25, 26, 29, 30)       # the farm hamlet on the hill north-west of the Shack, left unbuilt
WING_PARENTS = {84: 81, 95: 96, 225: 221, 237: 239, 248: 238, 276: 283, 283: 276, 286: 273, 292: 285, 308: 296,
                310: 313, 404: 407, 454: 438, 472: 447}
SHOPS = (("dervish-and-banges", 86, "2a"), ("gladrags", 110, "2b"), ("three-broomsticks", 125, "2c"),
         ("zonkos", 136, "2d"), ("post-office", 140, "2e"), ("hogs-head", 143, "2f"), ("scrivenshafts", 154, "2g"),
         ("honeydukes", 183, "2h"), ("puddifoots", 316, "2i"))
VILLAGE_BUILDINGS = (("kirk", 42, "3a"), ("school", 52, "3b"), ("distillery", 108, "3c"), ("village-hall", 411, "3d"),
                     ("carriage-hangar", 517, "3e"), ("shrieking-shack", 74, "1f"))
OWN_DECISIONS = {"hogs-head": ["hogs-head-slate", "hogs-head-plan"], "honeydukes": ["honeydukes-mullions"],
                 "three-broomsticks": ["three-broomsticks-two-parts"], "post-office": ["post-office-owl-loft"],
                 "scrivenshafts": ["scrivenshafts-crow-steps"], "puddifoots": ["puddifoots-steamed-windows"],
                 "gladrags": ["gladrags-shop-blind"], "zonkos": ["zonkos-open"],
                 "dervish-and-banges": ["dervish-and-banges-clock"]}
SHOP_PARTS = {"honeydukes": ["shopfront", "hanging sign", "mullioned windows"],
              "three-broomsticks": ["inn front", "hanging sign", "the very old end, low under thatch"],
              "post-office": ["shopfront", "hanging sign", "owl loft"],
              "scrivenshafts": ["shopfront", "hanging sign", "crow-stepped gables"],
              "puddifoots": ["shopfront", "hanging sign", "steamed windows"],
              "gladrags": ["shopfront", "hanging sign", "shop blind"],
              "dervish-and-banges": ["shopfront", "hanging sign", "bracket clock"]}
PUBLIC_PARTS = {"school": ["schoolroom", "tall windows", "bell turret", "schoolhouse"],
                "distillery": ["still house", "kiln", "pagoda ventilator"],
                "village-hall": ["hall", "tall windows", "annexe"], "carriage-hangar": ["long shed", "carriage doors"]}
VILLAGE_IMAGES = ["Craigellachie.jpg", "Comrie.jpg", "Dunkeld.jpg", "Fortingall.jpg", "Tobermory.jpg", "Auchindrain.jpg",
                  "Kinrossie.jpg", "Camserney.jpg", "Fortingall2.jpg", "Torrlaoighseach.jpg", "Christmas_Card.jpg",
                  "straw_owl.jpg", "Gearranan.jpg", "Lochinver.jpg", "WB_Hogsmeade.jpg", "Strasbourg.jpg",
                  "German_Advent_Calendar.jpg"]


def assets(ctx):
    cl, hog_key = ctx.claims, ctx.hog_key
    traced = json.loads(TRACED.read_text(encoding="utf-8"))
    out = []
    for id, cid, label in SHOPS:
        title = hog_key[label]
        sources = [q(HM, title), q(H, "fancy shop signs hanging above the pavements")]
        extra = {}
        if id == "hogs-head":
            sources += [q(H, "The Hog's Head has bay windows."),
                        q(H, "hanging from a rusty bracket, there is a battered wooden sign painted with the severed"
                             " head of a wild boar"),
                        q(H, "the Hog's Head is probably tiled not thatched")]
            extra = {"images": ["Hogs_Head.gif"], "dimensions": {"width_m": ft(24), "depth_m": ft(32),
                                                                 "storey_count": 2},
                     "parts": ["bay windows", "boar's-head sign on a rusty bracket", "side bay"]}
        else:
            extra = {"images": ["St_Mary_Street.jpg", "Craigellachie.jpg", "Comrie.jpg"],
                     "parts": SHOP_PARTS.get(id, ["shopfront", "hanging sign"])}
        out.append(asset(id, "village", title, sources, labels=labels(HM, label), inputs=FOOTPRINT,
                         components=cl.comp(cid, kinds=("thatch", "slate")),
                         traced=[{"id": "label", "frame": HM, "feature": f"the pointer of the map's label {label}: the"
                                                                         " building it names within its row",
                                  "plan_px": traced["labels"][label]}],
                         decisions=["hogsmeade-hanging-signs", "village-roof-mix", "village-units",
                                    "village-shop-labels"] + OWN_DECISIONS.get(id, []),
                         **extra))
    for id, cid, label in VILLAGE_BUILDINGS:
        title = hog_key[label]
        sources, extra = [q(HM, title)], {}
        if id == "kirk":
            sources.append(q(H, "something like the modern kirk at Balquhidder but with this traditional, lantern-like"
                                " decoration"))
            extra = {"images": ["Balquhidder_kirk.jpg", "Balquhidder_kirk_old.jpg"],
                     "decisions": ["kirk-lantern-finial", "village-ancillary"],
                     "parts": ["rubble nave", "gabled porch", "lantern bell turret", "kirkyard wall", "headstones"]}
        elif id == "shrieking-shack":
            sources += [q(H, "a timber frame clad with weatherboarding"),
                        q(H, "possible that it has a corrugated-iron roof")]
            extra = {"images": ["Shrieking_Shack_plan.gif"], "dimensions": {"storey_count": 2},
                     "parts": ["weatherboard walls", "boarded windows", "low fence"],
                     "decisions": ["shack-character", "shack-above-the-village"]}
        else:
            sources.append(q(H, "the presence of a church and primary school, of a few large buildings amongst the"
                                " smaller ones"))
            extra = {"decisions": ["village-public-buildings"], "parts": PUBLIC_PARTS[id]}
        kinds = ("big",) if id in ("school", "carriage-hangar") else ("shack",) if id == "shrieking-shack" else (
            "thatch", "slate")
        out.append(asset(id, "village", title, sources, labels=labels(HM, label), inputs=FOOTPRINT,
                         components=cl.comp(cid, kinds=kinds), **extra))
    out += [
        asset("village-house", "village", "Hogsmeade houses and shops", [
            q(H, "a proper High Street with houses and shops on both sides of it"), q(H, "smallish two-storey buildings"),
            q(H, "enough variation in shape and style to allow for alleys between them"),
            q(H, "somewhat like Dunkeld but with many of the buildings thatched"),
            q(H, "painted in bright colours like Tobermory"), q(H, "probably sixteenth to eighteenth century in style"),
            q(H, "so long as the houses and shops in the main part of the village are mostly small and thatched")],
            images=VILLAGE_IMAGES, inputs=FOOTPRINT,
            rule={"region": "hogs", "kinds": ["thatch", "slate"], "min_area": 20},
            parts=["walls", "thatched or slate roofs", "gable chimneys", "dormers", "doors and windows",
                   "garden walls and gates near the High Street"],
            materials=["harl", "whitewash", "rubble stone", "thatch", "slate"],
            decisions=["village-roof-mix", "no-half-timbering", "thatch-straw-animals-optional",
                       "village-footprints-regularized", "village-units", "village-style-families",
                       "village-detail-where-seen", "village-levels", "village-ancillary", "shack-above-the-village"]),
        asset("village-wing", "village", "Wings and outshots of village buildings", [
            note("data", "footprints of 16 m² or less that touch a larger footprint")],
            inputs=FOOTPRINT, rule={"region": "hogs", "kinds": ["thatch", "slate"], "max_area": 16},
            parents={str(k): v for k, v in sorted(WING_PARENTS.items())}, decisions=["village-wings-join-parents"]),
        asset("street-furniture", "village", "High Street lamps", [
            q(H, "There are at least some streetlamps, which will probably be ornate Victorian cast-iron jobs rather"
                 " than modern."), q(H, "In DH the side-road to the Hog's Head is not lit")],
            labels=labels(HM, "1h"), images=["Brechin_lamp.jpg"], parts=["cast-iron lamps on stone plinths"],
            traced=[{"id": "high-street", "frame": HM, "feature": "the High Street (1h): two points on its line, one"
                                                                  " at each end; the road layer gives its centreline"
                                                                  " between them",
                     "plan_px": traced["high-street"]}],
            decisions=["hogsmeade-lamps-high-street-only", "village-high-street", "village-ancillary"]),
    ]
    return out


def excluded(ctx):
    """The mapped footprints the village leaves unbuilt, each with its decision."""
    return [{**c, "decision": "shack-above-the-village"} for c in ctx.claims.comp(*HAMLET, kinds=("thatch", "slate"))]


def decisions(ctx):
    village_negatives = ["WB_Hogsmeade.jpg", "Strasbourg.jpg", "German_Advent_Calendar.jpg"]
    return {
        "kirk-lantern-finial": decision(
            "The kirk follows the modern kirk at Balquhidder: a rubble nave under a steep slate roof with stone skews,"
            " filling its mapped footprint where the map draws its cross, with pointed lancets between buttresses along"
            " its long walls, three in its eastward gable (the middle one tallest) and one in its westward gable, and a"
            " gabled porch with a pointed door where the footprint steps out from the nave. On the westward gable's"
            " apex stands the old kirk's open lantern turret: a stone base, four posts round its bell, a pyramid cap."
            " It has no chimney.",
            q(H, "something like the modern kirk at Balquhidder but with this traditional, lantern-like decoration"),
            images=["Balquhidder_kirk.jpg", "Balquhidder_kirk_old.jpg"]),
        "hogsmeade-lamps-high-street-only": decision(
            "Ornate Victorian cast-iron lamps stand on the High Street only, about every 30 m, alternately on either"
            " side: each on the pavement 1.6 m in front of the end of a building's front, clear of its door, a glass"
            " lantern on a tapering iron post over a stone plinth. The Hog's Head side road is unlit.",
            q(H, "the darkness complete but for a few streetlamps and lit upper windows"),
            q(H, "There are at least some streetlamps, which will probably be ornate Victorian cast-iron jobs rather"
                 " than modern."), q(H, "In DH the side-road to the Hog's Head is not lit"),
            images=["Brechin_lamp.jpg"]),
        "hogsmeade-hanging-signs": decision(
            "Shops have signs hanging over the pavement.", q(H, "fancy shop signs hanging above the pavements"),
            images=["St_Mary_Street.jpg"]),
        "thatch-straw-animals-optional": decision(
            "A few thatched roofs may carry straw animals.",
            q(H, "decorative straw animals woven into the thatch"), images=["straw_owl.jpg"]),
        "no-half-timbering": decision(
            "No half-timbering, jetties, steep concave roofs or crooked chimneys.",
            q(H, "not massive, towering, dark multi-storey buildings with tiled roofs"),
            images=village_negatives, inferred=True),
        "village-roof-mix": decision(
            "Roofs follow each building's style family and street, not the decoded random mix: the main village's"
            " cottages are mostly thatched, about two High Street shops in three are slated as in Dunkeld and Comrie,"
            " and the Hog's Head is slated. The user chose this mix; the decoded mix was a random draw (REBUILD: seed"
            " 11, thatch probability 0.72).",
            q(H, "most of the cottages at least being thatched"),
            q(H, "it's unclear whether the shops are also little and thatched"),
            q(H, "the Hog's Head is probably tiled not thatched"), choice(ctx.choices["village_roofs"]),
            inferred=True),
        "village-ancillary": decision(
            "Hogsmeade's small structures are the High Street's lamps, the shops' hanging signs and the Hog's Head's"
            " boar sign, the Shack's fence, a kirkyard wall with a few headstones, and garden walls and gates near the"
            " High Street.",
            choice(ctx.choices["village_ancillary"]), q(H, "fancy shop signs hanging above the pavements"),
            inferred=True),
        "village-wings-join-parents": decision(
            "Village footprints of 16 m² or less are wings or outshots of the building they touch.",
            note("data", "no village footprint lies between 16 and 20 m²")),
        "hogs-head-slate": decision(
            "The Hog's Head is slated, not thatched.",
            q(H, "The presence of bay windows and the general state of delapidation suggest to me that the Hog's Head"
                 " is probably tiled not thatched.")),
        "hogs-head-plan": decision(
            "The Hog's Head is built to White Hound's plan of it: its door at the left of its front as you face it, a"
            " canted bay just right of the front's middle and another on its left side wall near the front, both glazed"
            " in grimy glass; its eaves run along its street and its sides are gables, its one stack rising in the right"
            " gable, where his plan puts every fireplace (flush in the gable like the village's other gable stacks,"
            " rather than on the outside breasts he draws); its sign hangs from a rusty bracket straight over the door,"
            " each face of the board a white cloth painted with the boar's severed head and its blood; and it is built"
            " of dark rubble.",
            q(H, "The bay windows were so encrusted with grime that very little daylight could permeate the room"),
            q(H, "A battered wooden sign hung from a rusty bracket over the door"),
            q(H, "with a picture on it of a wild boar's severed head, leaking blood on to the white cloth around it"),
            q(H, "There is one bay window on the side-wall of the bar to the left of the door, and one at front middle"),
            q(H, "that wall is most likely the one nearest the castle"), q(H, "the general state of delapidation"),
            images=["Hogs_Head.gif"]),
        "shack-character": decision(
            "The Shrieking Shack is two storeys of unpainted weatherboard on a rubble footing under corrugated iron,"
            " with one rubble gable stack. Its door is shut and every window is boarded over but one, on the ground"
            " floor of the wall that faces the castle; a low fence rings its garden a few metres out, open where the"
            " path comes up to it.",
            q(H, "even in daylight was slightly creepy, with its boarded windows and dank overgrown garden"),
            q(H, "all the windows boarded except for one"),
            q(H, "The single unblocked window revealed distant bursts of light where the castle stood"),
            q(H, "as they leaned on the fence, looking up at it"), q(H, "all the entrances are sealed shut"),
            q(H, "a timber frame clad with weatherboarding"), q(H, "possible that it has a corrugated-iron roof"),
            images=["Shrieking_Shack_plan.gif"]),
        "shack-above-the-village": decision(
            "The Shrieking Shack stands a little way above the rest of the village: its floor stands at least 2 m above"
            " every other building's. At site X a hamlet of seven farm buildings on the hill to its north-west, four"
            " mapped footprints at the end of a track off the lane to the stile, stood 13 to 15 m higher, and two"
            " cottages on that lane less than a metre lower. The hamlet is not built, though its track is kept, and the"
            " two cottages are built across their lane from where the map draws them, each turned to face it: cottage"
            " 23 is carried 32.5 m and cottage 28 50 m, 5 m further down the lane than the 45 m costed, so that no tree"
            " of the site's tree layer stands within 2.5 m of its walls. The lane to the stile keeps its cottages,"
            " fewer and in larger gardens than the village's, to its end.",
            q(H, "It stood a little way above the rest of the village"),
            q(H, "The cottages were fewer here, and their gardens larger"),
            q(H, "the incorporation of groups of farm cottages"),
            q(H, 'are just my "for example" suggestions as to what might be there'),
            q(H, "You can vary them as you please and still stay within canon"),
            note("data", "at site X the Shack's floor stood above 812 of the village's 819 other buildings; the"
                         " hamlet's seven stood 13.1 to 14.9 m above it, and cottages 28 and 23 0.4 and 0.6 m below it."
                         " Without the hamlet (1,532 triangles) and with the cottages across their lane, the highest"
                         " other floor is cottage 23's, 4.1 m below the Shack's. Within 40 m of the lane to the stile"
                         " stand 23 footprints, 1.3 to 5.3 every 100 m with median gaps of 10 to 31 m, against 6 m in"
                         " the village proper"),
            choice(ctx.choices["shack_above"]),
            note("user", "chose option 5b in an ask_user form shortly after 02:30 on 2 November 2026")),
        "honeydukes-mullions": decision(
            "Honeydukes' windows are mullioned: every window in its front, the shop's and those of the owners' rooms"
            " over it, is divided into lights at most 0.8 m wide by upright bars, painted like its shopfront in the"
            " shop windows and of stone above.",
            q(H, "nodding through the mullioned windows at the thick, swirling snow"),
            q(H, "It has mullioned windows (that is, divided by a narrow vertical bar)"),
            q(H, "The owners live over the shop.")),
        "three-broomsticks-two-parts": decision(
            "The Three Broomsticks is in two parts, both fronting the High Street: a two-storey inn, white-harled under"
            " slate, with Rosmerta's rooms over the bar looking onto the street; and at one end, across two-fifths of"
            " its length, its very old part, low rubble under thatch.",
            q(H, "The pub called the Three Broomsticks is probably the oldest building in the village."),
            q(H, "one end of which is very old, although the rest is late 18th C"),
            q(H, "in a few minutes were entering the tiny inn"),
            q(H, "I saw you Apparate as I was pulling my bedroom curtains!"), inferred=True),
        "post-office-owl-loft": decision(
            "The post office carries an owl loft along the middle of its ridge: a weatherboarded box under a hipped"
            " slate cap, with a row of flight holes over a perching ledge along each long side, for the owls that fly"
            " from it.",
            q(H, "the post office, from which owls issued at regular intervals"),
            q(H, "The owls sat hooting softly down at him, at least three hundred of them"), inferred=True),
        "scrivenshafts-crow-steps": decision(
            "Scrivenshaft's is slated, and its gables are crow-stepped in plain pale stone up both rakes to a taller"
            " stone at the apex: the coping of Scotland's sixteenth- and seventeenth-century burgh houses, in keeping"
            " with the village's period and the quill shop's old-fashioned trade, and plain, unlike the ornate gables"
            " of the negative references.",
            q(H, "A sort of Mediaeval-style stationers"), q(H, "probably sixteenth to eighteenth century in style"),
            q(H, "a handsome display of pheasant feather quills in the window"), inferred=True),
        "puddifoots-steamed-windows": decision(
            "Madam Puddifoot's is harled pink, and its shop windows are steamed up from inside, misted panes that glow"
            " with lamplight at dusk.",
            q(H, "a small teashop that Harry had never noticed before"),
            q(H, "It was a cramped, steamy little place where everything seemed to have been decorated with frills or"
                 " bows."), q(H, "Harry was reminded unpleasantly of Umbridge's office."), inferred=True),
        "gladrags-shop-blind": decision(
            "Gladrags, an outfitter with branches in London and Paris, pulls a striped canvas blind out over its"
            " shopfront from under its fascia, sloping down to a valance over the pavement.",
            q(H, "London, Paris, Hogsmeade"), inferred=True),
        "zonkos-open": decision(
            "Zonko's is open, its shopfront glazed: the flyover shows the Hogsmeade of PoA to OotP, not that of HBP,"
            " where it is boarded up.",
            q(H, "Zonko's Joke Shop had been boarded up"), q(H, "Then they visited Zonko's, which was so packed with"
                                                                " students")),
        "dervish-and-banges-clock": decision(
            "Dervish & Banges, which sells magical instruments, hangs an instrument maker's bracket clock out over the"
            " pavement from the pier nearest the middle of its front: a drum with a white face on either side, on an"
            " iron bar.",
            q(H, "They sell that sort of thing in Dervish and Banges, magical instruments and stuff"),
            q(H, "Dervish and Banges, the wizarding equipment shop"), inferred=True),
        "village-public-buildings": decision(
            "White Hound places the village's public buildings but draws none, so they follow Highland types. The"
            " school is a tall one-storey schoolroom of rubble under slate, its door in the middle of its front, tall"
            " windows front and back and a bell turret on its free gable, beside a two-storey white-harled schoolhouse."
            " The village hall is a tall one-storey harled hall, its door between tall windows, with a low annexe. The"
            " distillery is white-harled under slate: a still house beside a square kiln under a steep pyramid roof"
            " crowned by a louvred ventilator, the pagoda of a Highland distillery; the kiln has no chimney of its own."
            " The carriage hangar is one long weatherboarded shed under slate, kept for the carriages, its long front a"
            " row of carriage doors.",
            q(H, "the existence and position of a village hall, a hangar to store the Thestral-drawn carriages, a whisky"
                 " distillery"),
            q(H, "probably some sort of garage where the Thestral-drawn carriages are kept"),
            q(H, "Indeed it is very likely that there is a distillery in or near the village"),
            q(H, "the presence of a church and primary school, of a few large buildings amongst the smaller ones"),
            inferred=True),
        "village-footprints-regularized": decision(
            "Each footprint traced on the 2 m grid is straightened: its walls are the clustered near-square edges of"
            " its trace, its outline the wall-grid cells the trace mostly covers; a footprint no such grid fits (under"
            " IoU 0.8) keeps its trace, straightened at 0.6 m.",
            note("data", "locA.bsd is sampled every 2 m, so mapped corners arrive rounded; median IoU of the fits"
                         " 0.89"), inferred=True),
        "village-units": decision(
            "A footprint drawn as one block is a row of separate buildings: it is split at the offsets of its walls,"
            " slabs under 3 m join a neighbour, and rows of 16 m or more are cut into plots of 5.5 to 9.5 m, each"
            " with its own height, roof and front.",
            q(H, "with enough variation in shape and style to allow for alleys between them, rather than the"
                 " buildings all being one solid strip"), inferred=True),
        "village-style-families": decision(
            "Unnamed buildings take a style family by where they stand: High Street shop-houses (two storeys,"
            " brightly painted harl or rubble, shopfronts and hanging signs); terraces of two or one-and-a-half"
            " storeys under thatch; one-and-a-half-storey cottages; villas on large plots; outshots. No individual"
            " lore is invented for them.",
            q(H, "somewhat like Dunkeld but with many of the buildings thatched"),
            q(H, "Hogsmeade is probably painted in bright colours like Tobermory"),
            q(H, "smallish two-storey buildings"), choice(ctx.choices["village_look"]),
            images=["Dunkeld.jpg", "Tobermory.jpg", "Comrie.jpg", "Craigellachie.jpg", "Fortingall.jpg",
                    "Fortingall2.jpg", "Camserney.jpg", "Auchindrain.jpg", "Torrlaoighseach.jpg"], inferred=True),
        "village-high-street": decision(
            "The High Street is the road layer's centreline between two points traced at its ends on the labelled"
            " map; buildings whose fronts face it are the High Street's.",
            q(H, "the High Street forming a straight central spine probably not more than about 600 yards long,"
                 " sloping downwards from the Hogwarts end"), note("data", "the traced centreline is 555 m long")),
        "village-shop-labels": decision(
            "A named shop is the building its map label points to; the rest of its row are houses.",
            note("data", "every label pointer lies inside its census footprint")),
        "village-detail-where-seen": decision(
            "Every footprint is built, and the village keeps to its share of the payload by spending fine detail where"
            " the flyover sees it: chimney stacks are capped flush in pale stone rather than under overhanging copes, a"
            " raised door's steps are one stone block, and off the High Street a stack carries one pot, a back wall one"
            " window and a storey-and-a-half front one dormer. The camera passes 50 m and more above the village,"
            " where a cope's overhang or a step's riser spans a pixel or two.",
            note("data", "built with all its detail the village came to 225k triangles, about 1.6 MB of payload at 8"
                         " bytes a triangle, over the 1.5 MB it may add before its named buildings; trimmed, 174k"),
            inferred=True),
        "village-levels": decision(
            "The village's building records also give two coarser levels of detail and a shadow caster, so whatever"
            " places a building places it at every level. lod1 lays every window and door flat on its wall, stands the"
            " chimney stacks in the roofs as boxes and fills each roof's skin in, trimmed back to its gables so their"
            " windows show; it keeps base courses, dormers, the named buildings' bays, buttresses and roof features and"
            " the kirkyard's and gardens' walls, but no fittings, painted margins or reveals. lod2 is each building's"
            " walls under its filled-in skin, which keeps the eaves that the flyover sees from above; the shadow caster,"
            " which the page draws again only when the sun moves, is its walls and attic, the cheapest, with what stands"
            " out beyond the eaves and casts a shadow of its own: Gladrags' blind and the Shack's fence as built, the"
            " kirkyard's wall, and each garden's wall and gate as one slab along them. Seen from above it leaves no"
            " patch of more than a square metre of the full village farther than 0.75 m from it. They are"
            " designed for frame time, none cut to fit the payload's room. The payload carries each level that passes"
            " its guards and saves enough triangles, and the page chooses among them for each view.",
            note("user", "approved plan: levels of detail and the triangle budget are among the village's design"
                         " categories"),
            note("user", "live frame time ranks above page load"),
            note("data", "at site X, of the full village's 171k triangles lod1 takes 68k, lod2 32k and the shadow caster"
                         " 21k; packed alone, they take 0.64, 0.31 and 0.23 MB. Without its eaves lod2 would take"
                         " about 21k, but its roofs seen from above would cover a sixth less (intersection over union"
                         " 0.84)"),
            note("data", "integration found that the walls and attic alone cast no shadow for the kirkyard's wall, the"
                         " blind or the fence: seen from above, they left the kirkyard's wall (56 m²), the blind (9 m²),"
                         " the fence (4.5 m²) and the gardens' walls (up to 3.6 m² each) farther than 0.75 m from the"
                         " caster; with them, no patch of more than 0.3 m²")),
    }


VILLAGE_NEGATIVE = ["steep, high, concave roofs", "crooked chimneys", "half-timbering", "jettied upper floors",
                    "ornate gables"]
IMAGE_NOTES = {
    "Gearranan.jpg": ("partial", ["low rubble-stone black house with soft thatch, for the village fringes only"],
                      ["black houses in the village centre"]),
    "Killiecrankie.jpg": ("context", [], []),
    "Torrlaoighseach.jpg": ("partial", ["one-and-a-half-storey croft with dormers"], []),
    "farm_near_Kilbride.jpg": ("context", [], []),
    "WB_Hogsmeade.jpg": ("negative", [], VILLAGE_NEGATIVE),
    "Strasbourg.jpg": ("negative", [], VILLAGE_NEGATIVE),
    "Lochinver.jpg": ("comparative", [], ["houses strung along one side of a road only"]),
    "German_Advent_Calendar.jpg": ("negative", [], VILLAGE_NEGATIVE),
    "Christmas_Card.jpg": ("partial", ["low British cottages with soft thatch"], ["chocolate-box colouring"]),
    "St_Mary_Street.jpg": ("partial", ["fascia shopfronts", "hanging shop signs"], []),
    "Craigellachie.jpg": ("endorsed", ["two-storey street terraces", "harl and stone fronts", "slate with gable stacks",
                                       "dormers and shopfronts"], []),
    "Dunkeld.jpg": ("endorsed", ["varied fronts along a rising high street"], []),
    "Comrie.jpg": ("endorsed", ["two-storey street terraces", "harl and stone fronts", "slate with gable stacks",
                                "dormers and shopfronts"], []),
    "Auchindrain.jpg": ("endorsed", ["tiny whitewashed thatched cottage", "deep soft thatch"], []),
    "Kinrossie.jpg": ("endorsed", ["long low rubble-stone building under deep thatch"], []),
    "Fortingall.jpg": ("endorsed", ["terraced two-storey white thatched houses with dormers", "red trim"], []),
    "Camserney.jpg": ("endorsed", ["low stone farm cottages", "eyebrow dormers in thatch"], []),
    "Fortingall2.jpg": ("endorsed", ["plain conjoined cottages", "simple gable chimneys"], []),
    "straw_owl.jpg": ("partial", ["optional straw animals on thatch"], []),
    "Tobermory.jpg": ("endorsed", ["brightly painted fronts"], []),
    "Balquhidder_kirk_old.jpg": ("partial", ["lantern-like finial over the door"], ["ruin damage"]),
    "Brechin_lamp.jpg": ("endorsed", ["ornate Victorian cast-iron lamp on a stone plinth"], []),
    "Balquhidder_kirk.jpg": ("endorsed", ["rubble stone", "steep slate roof", "lancet windows", "buttresses",
                                          "side porch"], []),
    "Hogs_Head.gif": ("model", ["two storeys, 24 x 32 ft", "front-left door", "central bow window", "side bay",
                                "sign at the front left"], []),
    "Shrieking_Shack_plan.gif": ("model", ["at least two storeys", "weatherboard cladding", "boarded windows",
                                           "low fence"], []),
}
