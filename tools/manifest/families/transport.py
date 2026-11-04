"""Station, track, train and ship assets, decisions and reference-image notes for the mesh scene manifest."""
import json
from pathlib import Path

from tools.manifest.common import C, FOOTPRINT, FT, G, GM, L, asset, choice, decision, ft, labels, note, q

FAMILY = "transport"
BUILT = {"station", "track", "train", "ship"}
CENSUS = {21: (338.62, 107.7)}
TRACK = Path(__file__).resolve().parents[1] / "data" / "transport" / "track-traced.json"


def assets(ctx):
    cl = ctx.claims
    track = json.loads(TRACK.read_text(encoding="utf-8"))
    return [
        asset("station", "transport", "Hogsmeade station", [
            q(G, "It has a narrow platform about 170 yards long, with some sort of barrier along the back."),
            q(G, "there is a building of some sort, with windows, and a narrow door which lets onto the road outside"),
            q(G, "the station isn't all under a canopy"),
            q(G, "a small Highland station close to the probable location of Hogsmeade")],
            labels=labels(GM, "1a"), images=["station.jpg", "station2.jpg", "Plockton_station.jpg"],
            inputs=FOOTPRINT + ["meta.train"], components=cl.comp(21, kinds=("thatch",)),
            traced=[{"id": "platform", "frame": GM, "feature": "the platform's ends on the coaches' line (the drawn"
                                                                " platform is 155.4 m, 170 yards, long)",
                     "plan_px": track["platform"]},
                    {"id": "station-terrace", "frame": GM, "feature": "level terrace from the track to the boundary"
                                                                       " wall's outer face along the platform",
                     "plan_px": track["station-terrace"]},
                    {"id": "lake-path", "frame": GM, "feature": "the lake path from the platform's north-west end"
                                                                 " through the gap between the wall's two ends",
                     "plan_px": track["lake-path"]},
                    {"id": "path-to-boats", "frame": GM, "feature": "the path to the boats (1d) on from the lake path's"
                                                                     " corner, south through the pines (1c) and round"
                                                                     " its last bend on to the shore",
                     "plan_px": track["path-to-boats"]}],
            dimensions={"platform_length_m": round(170 * 3 * FT, 1)},
            parts=["two-room station building", "long narrow platform", "barrier along the back of the platform",
                   "part-length canopy", "exits at both ends", "station terrace"],
            materials=["painted timber boarding"], inferred=["canopy length", "building cladding", "terrace extent"],
            decisions=["station-layout-either", "station-terrace", "lamplit-windows"]),
        asset("track", "transport", "Railway track and turning loop", [
            q(G, "The railway line goes around a sharp bend just before it reaches the station."),
            q(G, "the track has a loop in it like the eye of a needle, with the two sides of the loop diverging just"
                 " before the station"),
            q(G, "parked halfway through another turning loop, and it then completes the turn to get back onto the"
                 " main track")],
            labels=labels(GM, "1a"), inputs=["meta.train", "locBed"],
            traced=[{"id": "turning-loop", "frame": GM, "feature": "station track and its turning loop (1a): the"
                                                                    " junction, the loop from it past the platform and"
                                                                    " round, and the main line to the map's edge",
                     "plan_px": track["turning-loop"]},
                    {"id": "platform-barrier", "frame": GM, "feature": "barrier along the back of the platform"}],
            parts=["rails and sleepers", "ballast", "turning loop", "points", "rock cuttings", "embankments",
                   "tunnel portal", "culvert"],
            inferred=["turnout geometry and its toe 20 m up the main line", "curves eased to at least 3 chains: the"
                      " engine on a 66 m curve, the S-curve from the coaches onto it, and the foot of the loop",
                      "formation grades", "earthwork slopes", "tunnel portal"],
            decisions=["detached-engine", "railway-regrade", "railway-tunnel", "railway-culvert", "railway-curves"]),
        asset("train", "transport", "Hogwarts Express (static, engine detached)", [
            q(G, "at least nine carriages at about 64'6\" each"),
            q(G, "The engine then uncouples from the train (taking its tender if any with it)"),
            q(G, "This would mean there are probably two guards' vans, one either end of the carriages"),
            q(L, "Just imagine the train to be scarlet."), choice("guards_vans: two-vans")],
            images=["Lochalsh_train.jpg"], inputs=["meta.train"],
            dimensions={"coach_length_m": ft(64.5), "coach_count": 9, "guards_van_count": 2},
            parts=["tender locomotive", "tender", "passenger coaches", "guards' vans", "bogies and wheels",
                   "couplings between coupled vehicles only"],
            materials=["scarlet livery"], inferred=["locomotive design", "guards' van length"],
            decisions=["guards-vans", "short-guards-vans", "detached-engine", "generic-tender-locomotive",
                       "train-scarlet-coaches", "lamplit-windows"]),
        asset("ship", "transport", "Durmstrang ship (static)", [
            q(C, "Moody can see the topmost sail of the Durmstrang ship"), q(C, "a mast can be very high"),
            q(C, "Say the ship is 180ft tall or 60 yards."),
            q(C, "we don't want the Durmstrang ship to go much over that"),
            note("book", "the ship's appearance is paraphrased in the ship-skeletal-book decision")],
            labels=labels(GM, "8d"), inputs=["meta.extras.ship"], dimensions={"mast_height_m": ft(180)},
            parts=["hull", "deck", "masts", "yards", "rigging"], decisions=["ship-skeletal-book"]),
    ]


def decisions(ctx):
    user, choices, defaults, offered = ctx.user, ctx.choices, ctx.defaults, ctx.offered
    train_gap_m = ctx.train_gap_m
    return {
        "guards-vans": decision(
            "The train has two guards' vans, one at each end of the nine coaches; the second van stands in the existing"
            " gap at the south-east end.",
            choice(choices["guards_vans"]),
            q(G, "This would mean there are probably two guards' vans, one either end of the carriages")),
        "detached-engine": decision(
            "The engine and tender stand detached south-east of the coaches, as the turning-loop practice allows; no"
            " coupling is stretched across the gap and the engine is not moved.",
            q(G, "The engine then uncouples from the train (taking its tender if any with it)"),
            note("data", f"meta.train places the engine {train_gap_m:.1f} m beyond the coaches")),
        "short-guards-vans": decision(
            "The guards' vans are short four-wheeled brake vans, so the nine 64'6\" coaches and the north-west van fill"
            " the coaches' registered 191.8 m.",
            q(G, "at least nine carriages at about 64'6\" each"),
            note("data", "meta.train's coach segment is 191.8 m; nine coaches and their buffers take 181.7 m"),
            inferred=True),
        "railway-regrade": decision(
            "The railway runs on a graded formation: the turning loop level with the station and the main line climbing"
            " 1 in 40 from the junction, in rock cuttings and on embankments. A post-pack step regrades the terrain"
            " inside the line's earthworks only and clears their trees, and every tree within 10 m of the open line,"
            " so that no crown the renderer grows comes within a metre of a coach's side.",
            choice(choices["railway_profile"]),
            q(G, "probably more trees, or a spur of hillside"),
            note("data", "the existing heightfield put the junction 36 m above the station 68 m away and the foot of"
                         " the loop 25 m below it")),
        "railway-tunnel": decision(
            "The main line enters a tunnel 180 m up it from the turnout's toe, at the first point where the hillside"
            " stands a headwall and a 2 m margin over the formation across the portal's width. Before it the line"
            " climbs 1 in 40 on an embankment up to 2 m high, across a dry valley made up where the train's screen"
            " closes it, then in a rock cutting up to 13 m deep. The hill stands"
            " for the spur that hides the departing train, which leaves heading as White Hound's line arrives: from the"
            " north, or slightly west of north.",
            q(G, "which prevents the train from being seen from the station once it has rounded the bend"),
            q("wh-setting", "On my re-worked, canon-consistent map it's coming from the north, or slightly west of north."),
            choice(choices["railway_profile"]), inferred=True),
        "railway-culvert": decision(
            "Where the cutting round the foot of the loop meets the burn beside it, the burn's water gives way to the"
            " earthworks; no other water is touched.",
            choice(choices["railway_profile"]),
            {"source": "user", "quote": "You can change the water layers, that's fine."},
            note("data", "the burn runs 10 m from the loop's centreline there, about 10 m above the formation on the"
                         " cutting's side; 2 texels (8 m\u00b2) of it are dried"),
            inferred=True),
        "railway-curves": decision(
            "No curve is tighter than about 3 chains (60 m; the tightest is 61 m), which a tender engine can take at"
            " walking pace. White Hound's ink kinks to about 40 m round his drawing of the engine, so the line there is"
            " designed: the engine stands on the curve round the foot of the loop (radius 66 m, the drawn foot's own)"
            " with both ends on its registered segment, the gentlest S joins it to the coaches' straight, and the foot"
            " rejoins his drawn east side.",
            {"source": "user", "quote": "if the loop is plausibly tight, there's nothing that says the train can't go"
                                        " slow, right? I'll leave that up to you."},
            note(GM, "the loop is drawn up to 6 m away from the built line where it swings round the drawn engine, and"
                     " within 2.4 m elsewhere"),
            inferred=True),
        "station-terrace": decision(
            "The station stands on a level terrace from the track to the boundary wall along the platform: the spur that"
            " rose up to 20 m over its north-west half is cut back and the ground that fell 3-6 m below its south-east"
            " half is made up, so the platform, its back wall, the road outside and the building share one level. The"
            " lake path leaves through the gap between the wall's two ends in a short cutting through the spur.",
            q(G, "a narrow door which lets onto the road outside"),
            q(G, "a way through this barrier, which leads to an entrance in the Hogwarts boundary wall"),
            choice(choices["railway_profile"]), images=["station.jpg", "station2.jpg"], inferred=True),
        "generic-tender-locomotive": decision(
            "A generic scarlet tender locomotive, not the film engine.", defaults, offered,
            q(L, "Just imagine the train to be scarlet."), images=["Lochalsh_train.jpg"], inferred=True),
        "train-scarlet-coaches": decision(
            "Coaches are scarlet.", q(L, "Just imagine the train to be scarlet."), images=["Lochalsh_train.jpg"]),
        "station-layout-either": decision(
            "White Hound offers two station sketches; either is acceptable, so the build follows whichever fits the"
            " registered track, platform and road.", note(G, "station and station2 sketches are alternatives"),
            images=["station.jpg", "station2.jpg"]),
        "ship-skeletal-book": decision(
            "The ship is a real hull, deck, masts and rigging with a skeletal, wreck-like character.",
            note("book", "Goblet of Fire ch. 15 describes the ship's skeletal, resurrected-wreck look; paraphrase only"),
            q(C, "Moody can see the topmost sail of the Durmstrang ship")),
    }


IMAGE_NOTES = {
    "Plockton_station.jpg": ("endorsed", ["small low station building", "vertical timber boarding", "simple canopy"],
                             []),
    "station.jpg": ("model", ["two-room building at the platform end", "long platform between track and wall",
                              "exits at both ends"], []),
    "station2.jpg": ("model", ["two-room building at the platform end", "long platform between track and wall",
                               "exits at both ends"], []),
    "Lochalsh_train.jpg": ("endorsed", ["scarlet coaches"], []),
}
