"""Scene-wide parts of the mesh scene manifest: the cited sources and frame, the exclusions, the reference-image
records, the decisions that span families and the notes on images no family uses."""
import re

from tools.manifest import whsrc as w
from tools.manifest.common import CP, G, GM, H, HM, L, choice, decision, note, q

TITLES = {"wh-index": "The Map of Hogwarts", "wh-castle": "The Map of Hogwarts: Hogwarts Castle",
          "wh-grounds": "The Map of Hogwarts: the castle grounds", "wh-hogsmeade": "The Map of Hogwarts: Hogsmeade",
          "wh-location": "The Map of Hogwarts: location", "wh-forest": "The Map of Hogwarts: the Forbidden Forest",
          "wh-setting": "The Map of Hogwarts: setting"}
BEACH_GLASS = (31, 32, 33, 34, 36, 37, 38, 39, 43, 44, 45, 46)
CASTLE_INTERIORS = ("3CR", "CC", "DC3", "DC4", "DO", "GtS", "HBW", "HoM", "L", "McGO", "McGQ", "MM", "MoE", "PB",
                    "RoR", "SC", "SR", "SS", "T?", "TC", "TR", "TS")
KEPT_TERRAIN = "terrain landform; the existing heightfield is kept"
KEPT_PLANTING = "vegetation; ordinary trees and planting are kept"
KEPT_PATH = "track, road or path surface; the existing ground shading is kept"
KEPT_WATER = "water surface; the existing water rendering is kept"
# White Hound's own line drawings and maps carry no caption credit
WH_DRAWN = re.compile(r"blockdiag_.*|station2?\.jpg|mer_village\.jpg|Beauxbatons\.jpg|cabin(_front)?\.jpg|"
                      r"Quidditch_.*|changing_rooms\.jpg|pitch_with_stands\.jpg|Hogs_Head\.gif|"
                      r"Shrieking_Shack_plan\.gif|train-route_.*")
SPECIAL_CREDITS = {
    "castle_Edinburgh_loch.jpg": "painting by Alexander Nasmyth",
    "JKR_WW.gif": "JK Rowling's sketch, as shown by White Hound",
    "castle_Cardoness.jpg": "Undiscovered Scotland",
    "WB_Hogsmeade.jpg": "Warner Bros. film still, as shown by White Hound",
    "Lochalsh_train.jpg": "Ingrid Deschryver of Lochcarron, from BBC website – Your Pictures of Scotland: 9-16 May 2014",
    "Loch_na_Tuadh.jpg": "adapted from Streetmap.co.uk",
}
UNAVAILABLE_REASON = "unavailable (404 since 2016)"


def page(sid):
    url = w.BASE + w.PAGES[sid][0]
    return {"kind": "page", "author": "White Hound", "title": TITLES[sid], "url": url, "archive": w.WAYBACK + url}


def map_source(title, sid, key_page, image, px, key):
    return {"kind": "map", "author": "White Hound", "title": title, "page": sid,
            "url": w.BASE + "map_of_Hogwarts/" + key_page, "image": w.BASE + "map_of_Hogwarts/artwork/" + image,
            "px": px, "key": key}


def sources():
    out = {sid: page(sid) for sid in TITLES}
    out["wh-grounds-map"] = map_source("Labelled map of the castle grounds", "wh-grounds", "grounds_map.htm",
                                       "Hogwarts_map_labelled.gif", [2000, 2348], w.map_key("grounds"))
    out["wh-hogsmeade-map"] = map_source("Labelled map of Hogsmeade", "wh-hogsmeade", "Hogsmeade_map.htm",
                                         "Hogsmeade_map_labelled.gif", [1735, 2261], w.map_key("Hogsmeade"))
    out["wh-castle-plan"] = map_source("Plan of Hogwarts castle", "wh-castle", "castle.htm", "castle_plan.gif",
                                       [744, 533], w.castle_plan_key())
    out["data"] = {"kind": "data", "title": "Registered scene layers and metadata", "path": "build/data.js"}
    out["user"] = {"kind": "user", "title": "Project owner's recorded decisions"}
    out["book"] = {"kind": "book", "title": "Harry Potter novels (secondary; paraphrased notes only)"}
    return out


def frame():
    return {
        "units": "metres", "axes": {"x": "east", "y": "up", "z": "north"}, "authority": "build/data.js",
        "registrations": {
            "wh-grounds-map": {"method": "georef", "georef": "georef.json", "rect": "grounds_rect"},
            "wh-hogsmeade-map": {"method": "georef", "georef": "georef.json", "rect": "hogs_rect"},
            "wh-castle-plan": {"method": "similarity fit to the meta.named tower centres", "status": "planned"},
        },
    }


def excluded(cl):
    rows = [(GM, "1e", {"decision": "exclude-boats"})]
    rows += [(GM, k, {"decision": "exclude-gof-set-pieces"}) for k in ("8a", "8b", "8c", "8e", "8f")]
    rows.append((GM, "3c", {"decision": "third-entrance-speculative"}))
    rows += [(GM, k, {"reason": KEPT_TERRAIN, "decision": "scope-exterior"})
             for k in ("1b", "2g", "4d", "4e", "4f", "4g", "4h", "4i")]
    rows += [(GM, k, {"reason": KEPT_PLANTING, "decision": "scope-exterior"})
             for k in ("1c", "5e", "5f", "6a", "6b", "6c", "7a", "7b", "7c", "7d", "7e")]
    rows += [(GM, k, {"reason": KEPT_PATH}) for k in ("1d", "2f", "3b", "3e", "3f", "3h")]
    rows += [(GM, k, {"reason": KEPT_WATER}) for k in ("2a", "6d", "6e")]
    rows += [(GM, "2b", {"reason": "underwater village; not visible from the flyover"}),
             (GM, "2d", {"reason": "creature, not a structure"}), (GM, "3d", {"reason": "area label, not a structure"}),
             (GM, "5c", {"reason": "planted ground surface, not a structure"})]
    rows += [(HM, k, {"reason": KEPT_PATH}) for k in ("1c", "1d", "1e", "1g", "1i", "1j", "1k", "1l")]
    rows += [(HM, "3f", {"reason": "open park ground"})] + [(HM, k, {"reason": KEPT_WATER}) for k in ("4a", "4b")]
    rows += [(CP, k, {"decision": "castle-plan-interiors-excluded"}) for k in CASTLE_INTERIORS]
    beach = [{**c, "decision": "beach-false-positives"} for c in cl.comp(*BEACH_GLASS, kinds=("glass",))]
    return {"labels": [{"label": f"{src}:{k}", **why} for src, k, why in rows], "components": beach}


def stance_of(name, notes):
    if name in notes:
        return notes[name]
    if name.startswith("blockdiag_"):
        return "schematic", [], []
    raise KeyError(f"no stance recorded for {name}")


def credit(name, caption):
    if name in SPECIAL_CREDITS:
        return SPECIAL_CREDITS[name]
    if WH_DRAWN.fullmatch(name):
        return "White Hound"
    for mark in ("©", "courtesy of"):
        if mark in caption:
            return caption[caption.index(mark):].strip()
    hits = list(re.finditer(r"\bfrom\b", caption))
    if hits:
        return caption[hits[-1].start():].strip()
    raise ValueError(f"no credit in the caption of {name}: {caption!r}")


def build_images(notes):
    """Manifest images (name -> record) and the excluded unavailable images, from the inspected page index; `notes`
    holds every family's stances."""
    images, excluded_images = {}, {}
    for entry in w.image_index():
        name, sid = entry["name"], w.INDEX_PAGES[entry["page"]]
        if entry.get("unavailable"):
            excluded_images.setdefault(name, {"name": name, "pages": [], "reason": UNAVAILABLE_REASON})["pages"].append(sid)
            continue
        if not entry.get("inspected"):
            raise ValueError(f"{name} was never inspected")
        if name in images:
            images[name]["pages"].append(sid)
            continue
        cap, alt = w.clean_caption(entry.get("cap")), w.clean_caption(entry.get("alt"))
        caption = cap if cap and not re.fullmatch(r"[a-z]\)", cap) else alt
        stance, carry, avoid = stance_of(name, notes)
        images[name] = {"pages": [sid], "src": w.BASE + "map_of_Hogwarts/" + entry["src"], "caption": caption,
                        "credit": credit(name, cap), "stance": stance, "carry": carry, "avoid": avoid}
    unknown = set(notes) - set(images)
    if unknown:
        raise ValueError(f"stances for images not in the index: {sorted(unknown)}")
    return images, list(excluded_images.values())


def decisions(ctx):
    user, choices = ctx.user, ctx.choices
    return {
        "wh-authority": decision(
            "White Hound's essay, maps and drawings set the layout and interpretation; compatible book-based analysis may"
            " only fill gaps, and any departure from a White Hound conclusion needs the user's approval.",
            {"source": "user", "quote": user["quotes"][0]}),
        "live-origin-provenance": decision(
            "The original White Hound site was live again on 2026-10-29; the fetched pages and maps are hash-identical to"
            " the cited Wayback capture, so either copy is authoritative.",
            note("wh-index", "fetched live on 2026-10-29 and compared byte-for-byte with the Wayback capture")),
        "scope-exterior": decision(
            "Model every building and named exterior object; ordinary trees and terrain stay as they are.",
            choice(choices["scope"])),
        "static-exteriors": decision(
            "Detailed exteriors only: the train, ship and Willow are static, with no explorable interiors or new"
            " animations.", choice(choices["detail"])),
        "performance-budget": decision(
            "At the default half-resolution internal scale, frame time may rise by at most 10% on each GPU for clearly"
            " better detail.", choice(choices["performance"])),
        "exclude-gof-set-pieces": decision(
            "The temporary Triwizard set pieces (paddocks, Beauxbatons caravan, swimming start and grandstands) are not"
            " modelled; the scene keeps its existing multi-book composite.", choice(choices["gof"])),
        "exclude-boats": decision(
            "The first-years' boats are not modelled; they normally sit in the underground harbour, whose mouth is"
            " modelled.", choice(choices["boats"])),
        "lamplit-windows": decision(
            "Lit windows glow with lamplight, not white: candlelight (about 2200 K) in the castle, the coaches' lamps"
            " (about 2700 K) and dimmer oil lamps (about 2400 K) at the station. Some rooms are lit only dimly, the glow"
            " stays short of the tone curve's white shoulder at the night exposure, and lit windows shine in the lake.",
            {"source": "user", "quote": "what's the realistic color for Windows at Dusk? Is there anything more you"
                                        " could do about it?"},
            q(G, "because the station has lighted windows even when the platform itself is dark"),
            inferred=True),
        "beach-false-positives": decision(
            "Twelve lake-shore footprints decoded as glass are beach and shingle, not greenhouses.",
            note(GM, "the shore ink shares the greenhouse white; the real greenhouses are at 5b")),
        "third-entrance-speculative": decision(
            "The possible third entrance is speculative and is not modelled.",
            q(G, "We do not know where the third entrance is.")),
    } | (forest_decisions(choices) | base_decisions() | grounds_decisions(choices) | site_decisions(choices)
         | tour_decisions())


F = "wh-forest"
NEST_IS_FOR_INSTANCE = "everything which lies north and north-west of Aragog's nest is just a \"for instance\""
FOREST_DOESNT_GO_THERE = ("preferably you should just say the forest doesn't go there but if that doesn't match the"
                          " reasoning then yeah you should clear")
CLEARINGS_PLAUSIBLE = ("the clearings, wherever they are, are large enough to be plausible and not look suspiciously like cutouts"
                       " created just for line of sight purposes")


def forest_decisions(choices):
    """How far White Hound's Forest reaches over site X's real roads, railway and buildings (src/forest_extent.py and
    SITES["x"]["forest"] in src/common.py), and where its places stand (src/forest_places.py)."""
    return {
        "forest-extent-wh-reasoning": decision(
            "The Forbidden Forest keeps what White Hound's reasoning from the books needs: its edge against the grounds,"
            " the near Forest along both paths out past Aragog's nest, and its overall scale. Beyond the nest, where"
            " its content is his 'for instance', it ends short of the real roads and village; only where his reasoning"
            " needs Forest over a real feature is a corridor cleared through it.",
            {"source": "user", "quote": "not all of the things mentioned by Whitehound are actually necessary for"
                                        " compliance with the book"},
            {"source": "user", "quote": "clear as much as you need in order to actually match the reasoning he"
                                        " provides"},
            {"source": "user", "quote": FOREST_DOESNT_GO_THERE},
            q(F, "until you are about one and a third miles in a straightish line from the edge of the Forest"),
            q(F, NEST_IS_FOR_INSTANCE)),
        "forest-ends-at-a832": decision(
            "The Forest ends 25 to 90 m short of the A832, and nothing beyond the road, seen from the castle, is Forest:"
            " the 5.5 km of the A832 it covered, and its reaches north of the road onto Fionn Bheinn's lower slopes,"
            " lie outside it. All of that lies north of Aragog's nest, and the Forest keeps White Hound's scale: 3.2"
            " miles south to north (4.8 as drawn at this site) and 2.9 miles south-east to north-west.",
            q(F, "Apart from the overall scale of the forest, and the position of the mountains"),
            q(F, NEST_IS_FOR_INSTANCE),
            q(F, "It is probably about four miles across, at least in the south/north direction."),
            q(F, "the Forbidden Forest needs to be three or four miles across"),
            q(G, "The Forbidden Forest therefore is unlikely to be more than about four miles across"),
            {"source": "user", "quote": FOREST_DOESNT_GO_THERE}),
        "forest-clear-of-achnasheen": decision(
            "Achnasheen, on the A832, stands outside the Forest: its edge runs 40 to 100 m round every building within"
            " 150 m of the road, with no Forest between them and the road (2.7 km of building outlines it covered).",
            q(F, NEST_IS_FOR_INSTANCE),
            {"source": "user", "quote": FOREST_DOESNT_GO_THERE}),
        "forest-corridors-a890-railway": decision(
            "The A890 and the Kyle railway cross the Forest by Aragog's nest and Grawp's clearing: the railway 178 m from"
            " the nest's hollow and 247 m from the clearing, the A890 541 m and 723 m from them (White Hound draws the"
            " nest, his 6g, 37 m from the railway and 400 m from the A890). The Forest must reach on past the nest for"
            " the nest to lie in its heart, so there it is not pulled back but cleared: 12 m each side of the A890's"
            " centre line and 10 m each side of the railway's (5.0 km of the A890 and 7.8 km of the railway lay under"
            " it).",
            q(F, "The Acromantula nest was said to be in the heart of the Forest"),
            q(F, "in the sense that it is in the dense inner area as opposed to the outskirts"),
            {"source": "user", "quote": FOREST_DOESNT_GO_THERE}),
        "forest-homesteads": decision(
            "Each building the Forest still covers stands in its homestead's clearing, not in a cutout round its walls:"
            " its yard, 10 m round it, is clear, and the clearing spreads on beyond as the homestead's ground, its edge"
            " wandering and the Forest thinning out over 24 m beyond it. Six clear Forest that stands, each 0.55 to"
            " 1.8 ha.",
            q(F, NEST_IS_FOR_INSTANCE),
            {"source": "user", "quote": FOREST_DOESNT_GO_THERE},
            {"source": "user", "quote": CLEARINGS_PLAUSIBLE}),
        "forest-tracks-kept": decision(
            "Minor roads and tracks keep the Forest over them where it stands (9.7 km of tracks, 0.7 km of service"
            " roads and 0.2 km of unclassified road): White Hound's own path is the one clear way into the deep Forest"
            " near the school.",
            q(F, "This path is probably the only significant clear path penetrating the deep Forest anywhere near the"
                 " school.")),
        "aragogs-hollow": decision(
            "Aragog's hollow lies 140 m south-east of White Hound's 6g. At this site his crater lay 37 m from the Kyle"
            " railway and 77 m from Loch Gowan, where a hollow would have its floor 6 m below the loch and the railway"
            " across its rim. The hollow is cut into the hillside there (0.095 Mm\u00b3), 100 m across at its rim, a"
            " vast hollow with a steep slope down into it, as CoS 15 has it: its floor is level, 10 m below the lowest"
            " ground on its rim (9 m below where it would spill, in the 10 m ground), and its sides rise at 45 degrees,"
            " 10 m high on its downhill side and 25 m on its uphill side. Its floor stands 11 m above Loch Gowan, whose"
            " shore lies 212 m from its centre. It lies 1.21 miles from Hagrid's hut, 18 degrees west of north from the"
            " castle, in the Forest's dense heart: 178 m from the railway and 541 m from the A890, with the Forest 0.82"
            " dense from 30 to 100 m beyond its rim. It is cleared of trees out beyond its rim, 1.69 ha, its edge"
            " wandering and the Forest thickening back over 16 m beyond it; no tree stands in it, and DH 34 finds it"
            " still a clearing. His crater's bare patch, with the edge the Forest layer's sampling of his map blurs"
            " (1.58 ha), holds Forest as dense as the Forest round it. Aragog's web, whose remnants stand there in DH"
            " 34, is left out.",
            q(F, "they had reached the rim of a vast hollow, a hollow which had been cleared of trees"),
            q(F, "made its way down the steep slope, towards a misty domed web in the very centre of the hollow"),
            q(F, "stepped out into a clearing that Harry knew had been the place where the monstrous Aragog had once"
                 " lived"),
            q(F, "The remnants of his vast web were there still"),
            q(F, "steep-sided, cleared of trees and at least fifty yards across and probably bigger"),
            q(F, "The hollow should be fairly close to a water supply"),
            q(F, "but not so close that it floods"),
            q(F, "the lair cannot be much more than a mile and a half into the woods, and really should be less than"
                 " that"),
            q(F, "It makes sense that the Acromantula nest would tend to be to the right and north because the forest"
                 " slopes down in that direction"),
            q(F, "in the sense that it is in the dense inner area as opposed to the outskirts"),
            {"source": "user", "quote": CLEARINGS_PLAUSIBLE},
            note("user", "orchestrated choice: after the trunk's tour framed lochs at waypoints 11 and 12, the"
                         " orchestrator had White Hound's 6g moved into dry Forest about (-581, 1739) and the hollow"
                         " carved there, about 100 m across with its floor about 10 m below its lowest rim, and his"
                         " drawn patch given back to the Forest; the web left out"),
            choice(choices["book_evidence_rule"])),
        "grawps-clearing": decision(
            "Grawp's clearing lies 613 m east-south-east of White Hound's 4f, which at this site lay on a river's"
            " floodplain, inside a meander 49 m from its water, in unbroken Forest. It stands in dry Forest on a"
            " 5-degree slope, 292 m from the nearest water, 1.02 miles from Hagrid's hut and west-north-west of the"
            " Entrance Hall (293 degrees), broadly in front of the castle, whose front faces somewhat north of west: 247"
            " m from the railway and 723 m from the A890. It is an apparently artificial clearing, the trees ripped up"
            " round where Grawp is tied (OotP 30), of 0.36 ha, its edge wandering and the Forest thickening back over"
            " 16 m beyond it. In it six small pines stand, 9.8 to 14.3 m tall, as small as the renderer grows trees,"
            " against twenty-foot pines (OotP 31), and the renderer makes 85% of the trees within 260 m of it pines."
            " The ropes, and the heaps of trunks and boughs round it, are left out.",
            q(F, "Trees had been ripped up at the roots all around the mound, so that it stood on a bare patch of"
                 " ground"),
            q(F, "The trees to which the other ends of the ropes around Grawp's wrists and ankles were attached creaked"
                 " ominously"),
            q(F, "seized a bird's nest from the upper branches of a towering pine"),
            q(F, "enjoys ripping up twenty-foot pine trees"),
            q(F, "Here they come to an apparently artificial clearing where Grawp is shackled, surrounded by small pine"
                 " trees."),
            q(F, "They must be small ones"),
            q(F, "About 600 yards after leaving the path, there is an area of pine-trees, surrounding an artificial"
                 " clearing where Grawp was confined."),
            q(F, "a substantial chunk of the Forest must be broadly in front of the castle"),
            q(F, "which faces somewhat north of west"),
            {"source": "user", "quote": CLEARINGS_PLAUSIBLE},
            note("user", "orchestrated choice: after the trunk's tour framed a lake at waypoint 12, the orchestrator"
                         " had White Hound's 4f moved into dry Forest about (-1700, 650) and an artificial clearing of"
                         " 0.3 to 0.4 ha cleared there, with a few small pines standing"),
            choice(choices["book_evidence_rule"])),
    }



def base_decisions():
    """The castle's base at site X (src/castle_base.py, SITES["x"]["base"] in src/common.py)."""
    return {
        "castle-base-mound": decision(
            "Where the ground lies 54 to 80 m below the castle's platform, the castle stands on its own mound: a skirt of"
            " lawn round it, then lawns falling 1 in 4 towards Hagrid's cabin and the pitch and 1 in 2.6 elsewhere, all"
            " walkable. A crag with rock faces is the one-line alternative (SITES[\"x\"][\"base\"]).",
            q(G, "The castle stands on top of its own little hill or mound"), q(GM, "castle mound"),
            q(GM, "sloping lawn"), q(G, "There is a substantial skirt of lawn in front of the castle.")),
        "castle-base-lake-cliff": decision(
            "Between the castle and the lake the base drops 36 m (about 118 ft) as White Hound's cliff: a foot-path slope"
            " (22 degrees) below the castle's walls, then a near-vertical face (81 degrees) to the water, with the 2e"
            " wall at its lip.",
            q(G, "the lower part of the cliff either vertical or sloping back, and the upper part sloping back towards"
                 " the base of the castle"),
            q(G, "the cliff needs to be no more than about 120ft"), q(GM, "wall along edge of cliff overlooking lake")),
        "vegetable-patch-hollow": decision(
            "The vegetable patch between the castle and its greenhouses lies on a level terrace about 11 m below the"
            " castle's platform, in its own little hollow, 0.25 m deep, that floods in heavy rain; the greenhouses,"
            " their pads and the paths stand dry on it, and a walkable bank (1 in 2.2) rises from it to the castle walls.",
            q(G, "The vegetable patch must be either quite low-lying relative to the rest of the grounds, or in its own"
                 " little hollow"),
            q(G, "splashed across the flooded vegetable patch to double Herbology"),
            q(G, "The approach from the north is either a slope up to the castle walls or, if a drop, then a very low"
                 " one.")),
        "outflow-under-the-base": decision(
            "Where the castle's base buries the lake's outflow it runs under the base and comes out at its foot, never"
            " perched above the lake.",
            q(F, "from the lake north and down through the forest"),
            q(F, "west through an underground channel which leads towards the village")),
    }


def site_decisions(choices):
    """The move to site X: where it is, its terrain, and its sunsets against White Hound's."""
    return {
        "site-x": decision(
            "The castle stands at site X, on the shelf south of Loch Gowan at the head of Glen Carron (57.5420 N 5.0760 W,"
            " NH 1599 5437), 6.5 km north-west of White Hound's reference point, with the lake at 284 m and the castle's"
            " platform 36 m above it. The original build stood near Scardroy in Strathconon, with the lake at 153 m. A"
            " location review scored X the best of the four sites it checked on bare earth (0.685, against 0.572 for"
            " Scardroy) and measured White Hound's ordering there: the Forest's floor 45 m and Aragog's hollow 152 m"
            " below the castle, and Hogsmeade 4 m below the gates on 2-degree ground. It put about 65% confidence on X"
            " fitting him better than Scardroy.",
            q(L, "is probably close to either Glen Carron (best fit for the railway-line) or Glen Affric (best fit for"
                 " the snow-covered mountain)"),
            q(L, "my standard reference point for purposes of calculating sunset times etc."),
            choice(choices["site"])),
        "elevation-model": decision(
            "Site X's terrain is FABDEM V1-2: Copernicus GLO-30 with forests and buildings removed, so the renderer's"
            " trees and the meshes' buildings do not stand on canopy and roofs baked into the ground. The four models"
            " compared (GLO-30, FABDEM, GEDTM30 and OS Terrain 50) agree within about 1 m on what the layout depends on."
            " It is one line of the site's configuration (SITES[\"x\"][\"dem\"]), and the page credits it under"
            " CC BY-NC-SA 4.0.",
            note("user", "orchestrated choice: with the user away, the orchestrator chose FABDEM V1-2 at gate (a)")),
        "sunset-shortfall": decision(
            "At site X the sun sets 23 and 24 minutes early on White Hound's two dated evenings (6 June 1994 and 21 June"
            " 1996), against his around thirty and about forty minutes, though it rises 44 and 46 minutes late (31 March"
            " and 4 May 1998), as his about forty-five minutes has it: the mountains north-west of X are lower than his"
            " reasoning needs. It is recorded, not fixed. His figures are inferences from the books' timelines, which give"
            " no clock time for either sunset, so no quoted book evidence is contradicted.",
            q(G, "we know that the sun set around thirty minutes early due to the presence of a mountain or mountains"
                 " which obscured the true horizon"),
            q("wh-setting", "there are mountains between Hogwarts and the midsummer sunset which brought the suunset"
                            " forwards by about forty minutes"),
            q("wh-setting", "the sunrise at Hogsmeade/Hogwarts in early to mid April is retarded by about forty-five"
                            " minutes"),
            choice(choices["book_evidence_rule"])),
    }


def tour_decisions():
    """Where the tour's waypoints (WP in src/template_c.html) stand to show what their captions name.
    src/test_render_tour.py holds each named subject to a share of the frame, measured as the flyover shows it, and
    keeps the camera out of every building, since the path's clearance follows the ground and the trees, not the
    meshes. The showcase tour brought the camera close wherever the meshes have detail to show; distances are from
    each waypoint's camera to what it names, and "old" is the tour before it. Heights are the camera's above the
    ground or water as the tour flies it, which that clearance can raise above the height a waypoint writes."""
    delegated = note("user", "decided by the orchestrator under the user's delegation of look choices")
    return {
        "tour-ship-close": decision(
            "The tour's Durmstrang ship waypoint stands 157 m south-east of the ship, about 22 m above the lake, where"
            " the ship fills about 3.7% of the frame with the castle's cliff behind it. The old waypoint was 394 m"
            " off.", delegated),
        "tour-gryffindor-close": decision(
            "The tour's Gryffindor Tower waypoint stands 105 m south-south-east of the tower, 34 m above the lake, high"
            " enough to see it over the trees on the shore, where the castle fills about 17% of the frame. The old"
            " waypoint was 170 m off.", delegated),
        "tour-castle-east-transition": decision(
            "After Gryffindor Tower the tour turns east over the lake, 40 m up and south-east of the castle, looking"
            " back at it, before it flies north to the station; the old turn stood about 70 m up east of the castle."
            " The path's clearance follows the ground and the trees, not the meshes, and a candidate that turned"
            " closer to the castle took the camera through it.", delegated),
        "tour-station-close": decision(
            "The tour's Hogsmeade station waypoint stands 123 m east-north-east of the station, 30 m up, where it fills"
            " about 8% of the frame, the train along its platform and the castle above. The old waypoint was 168 m"
            " off and about 47 m up.", delegated),
        "tour-north-tower-close": decision(
            "The tour's North Tower waypoint stands 118 m north-east of the tower and looks at its middle rather than"
            " above it, so its onion dome stands at the centre of the castle's north front, over the greenhouses;"
            " the castle fills about 16% of the frame. The old waypoint was 357 m off.", delegated),
        "tour-greenhouses-close": decision(
            "The tour's Greenhouses waypoint stands 88 m north-east of the northern greenhouses, about 32 m up, where"
            " they fill about 3.4% of the frame, and the tour then passes over their glass toward the Willow. The old"
            " waypoint was 133 m off and about 54 m up.", delegated),
        "tour-willow-close": decision(
            "The tour's Whomping Willow waypoint stands 45 m north-north-east of the tree, where it fills about 3% of"
            " the frame, over the 1% the framing test asks of a tree among trees. The old waypoint was 65 m off.",
            delegated),
        "tour-aragogs-hollow": decision(
            "The tour's Aragog's hollow waypoint stands 228 m north-west of the hollow, about 120 m up, looking"
            " south-east down into it at 30 degrees, so its cleared bowl is the frame's centre: the hollow's ground"
            " fills about 8.6% of the frame and no water shows. Coming in, the camera passes just beyond its rim, 60 m"
            " from its centre, looking steeply down into it. The framing test holds the hollow's ground, within its"
            " clearing's recorded outline, to its share of the frame and to more than any water in it. The old"
            " waypoint looked at White Hound's 6g, which at this site lay by Loch Gowan, and framed the loch.",
            note("user", "the user withdrew the framing exemption for Aragog's hollow, Grawp's clearing and Sirius's"
                         " cave: they are named places in the books, dry ground in the Forest, so their frames must"
                         " show them"),
            note("data", "lane D's camera for the hollow, at its new place, framed no water and the cleared bowl at"
                         " its centre"),
            delegated),
        "tour-grawps-clearing": decision(
            "The tour's Grawp's clearing waypoint stands 113 m north-east of the clearing, about 109 m up, looking down"
            " at 42 degrees, steeply enough to see past the pines at its edge, so the clearing and the six small"
            " pines standing in it are the frame's centre: its ground fills about 5.2% of the frame and no water"
            " shows; the camera then passes over the clearing, looking down into it, as it turns west for Sirius's"
            " cave. From the south its ground filled 4.1% of the frame; from the south-west 5.7%, but off-centre, with"
            " the dark pines filling the rest of the frame. The framing test holds it as it holds Aragog's hollow. The"
            " old waypoint looked at White Hound's 4f, on a river's floodplain at this site, and framed water.",
            note("data", "lane D found that below about a 30-degree pitch the pines at the clearing's near edge hide"
                         " its centre"),
            delegated),
        "tour-great-hall-closer": decision(
            "The tour's Entrance Hall & Great Hall waypoint stands 312 m west of the front doors, about 65 m up, where"
            " the castle's west front fills about 12% of the frame as the tour comes in from the forest. The old"
            " waypoint was 1,032 m off and 140 m up.", delegated),
        "tour-west-tower-close": decision(
            "The tour's West Tower (Owlery) waypoint stands 154 m west of the tower, where the castle fills about 19%"
            " of the frame. The old waypoint was 325 m off.", delegated),
        "tour-hut-close": decision(
            "The tour's Hagrid's hut waypoint stands 50 m west-south-west of the hut, where the hut fills about 1.7% of"
            " the frame beside its pumpkin patch. The old waypoint was 74 m off.", delegated),
        "tour-astronomy-close": decision(
            "The tour's Astronomy Tower waypoint stands 180 m south-west of the tower, where the castle fills about 24%"
            " of the frame with the tower at its centre beside the front doors. The old waypoint was 387 m off.",
            delegated),
        "tour-pitch-close": decision(
            "The tour's Quidditch pitch pass shows its goals as hoops: three golden poles, fifty feet high, with hoops"
            " on the end. It comes in over the pitch's east end, where both goals show, and drops past the nearer"
            " one: its first waypoint stands on the pitch's long axis 25 m east of the nearer hoops, about 27 m up,"
            " where their rings, 1.52 m across, are about 42 pixels wide in a 720-row frame, seen against the pitch's"
            " grass. A second waypoint sweeps the camera down the pitch to 24 m short of the far goal, whose rings"
            " show at about 46 pixels against the stands behind them, before the tour turns to the gates. The path's"
            " clearance keeps the camera at least 26 m above open ground, above the hoops' 15 m, so the rings are"
            " seen from above rather than against the sky. The pitch fills about 7-11% of the frame. The old waypoint"
            " stood north-east of the pitch, about 30 m up, with the nearer hoops 66 m off, thin gold lines with small"
            " rings, and the far ones barely showing.",
            note("user", "the user asked where the Quidditch hoops are; the orchestrator asked for a pass that drops"
                         " past the near goal at about 20-30 m and sweeps down the pitch to the far goal, so that a"
                         " set of hoops reads clearly as hoops"),
            delegated),
        "tour-gates-close": decision(
            "The tour's winged-boar gates waypoint stands 60 m south-south-west of the gates, where they fill about"
            " 0.8% of the frame. The old waypoint was 84 m off.", delegated),
        "tour-hogsmeade-low": decision(
            "The tour's Hogsmeade waypoint stands at the village's south-east end, 40 m up, looking up the High Street,"
            " where the village fills about 19% of the frame. The old waypoint stood 70 m up and looked past the High"
            " Street into the village's middle.", delegated),
        "tour-high-street": decision(
            "A new waypoint, Hogsmeade High Street, flies the High Street about 30 m up: over Honeydukes and the bridge"
            " by Scrivenshaft's, then past the Hog's Head's side street, looking ahead to the Three Broomsticks, before"
            " the tour turns north to the Shack. The shops fill about 4% of the frame.", delegated),
        "tour-shack-from-above": decision(
            "The tour's Shrieking Shack waypoint stands 68 m north-east of the Shack and about 69 m up, where it fills"
            " about 1.2% of the frame, its roof and north-east wall among the trees of its garden. The old waypoint"
            " stood there about 97 m up.", delegated),
        "tour-tomb-close-pass": decision(
            "The tour's Dumbledore's tomb waypoint stands about 23 m south-south-east of the tomb and about 28 m up,"
            " looking north-north-west over it, where it fills about 0.46% of the frame on its lawn. The lake's bank,"
            " about 10 m beyond the tomb to its west and north, crosses the upper part of the frame as a thin strip: the"
            " view looks across the bank, whose faint facets show in oblique views, rather than along it. The tour comes"
            " to it over land from the south, then turns east of it to look across the lake at the castle and climbs to"
            " its closing view as before. The showcase tour's waypoint stood 20 m south of the tomb, about 38 m up,"
            " looking north over it along the bank.", delegated),
        "tour-no-harbour": decision(
            "The cliff opening to the underground harbour is not a waypoint: it faces east, in shadow through the"
            " afternoon and evening, and from 56 m its frame was dark cliff, the 4.5 m opening unreadable.",
            delegated),
        "tour-length": decision(
            "The tour lasts about 574 s, under the ten minutes allowed for it: the camera slows when it flies low and"
            " near what it looks at, so the closer waypoints lengthen it. Its speed formula stays as it was.",
            delegated),
    }


def grounds_decisions(choices):
    """Where site X's own ground breaks White Hound's slopes, the evidence kept (src/grounds_grade.py, SITES["x"])."""
    return {
        "cabin-above-the-gates": decision(
            "Hagrid's cabin stands on a terrace 4.0 m above the gates' ground, raised over the hollow it stood in at this"
            " site (2.7 Mm\u00b3 of fill, up to 35 m deep): level within 150 m of the cabin, falling 1 in 3 beyond, with"
            " the drive climbing from the gates to it on an embankment, and a lawn 40 m wide kept at its level as far as"
            " the castle mound, so the lawns fall from the castle's front to the cabin with no hollow below it. The gates"
            " keep their own ground; the cabin stands 23.7 m above the lake; and White Hound's three sight lines clear"
            " the ground (cabin to castle doors by 1.8 m, gates to cabin by 0.5 m, doors to gates by 1.4 m).",
            q(G, "The front gates are lower down than Hagrid's cabin."),
            q(G, "as they went down the sloping lawns to Hagrid's hut"),
            q(G, "down the sloping lawn towards Hagrid's small wooden cabin"),
            q(G, "and up through the grounds to Hagrid's cabin"),
            q(G, "Harry does scramble up from the lake to get to Hagrid's house"),
            q(G, "There is a clear line of sight from Hagrid's cabin to the castle doors"),
            q(G, "from at or near the front gates to Hagrid's cabin and from the castle doors to the front gates"),
            choice(choices["gates_cabin"])),
        "drive-dips-by-the-lake": decision(
            "Between Hagrid's cabin and the castle the drive passes 22 to 31 m from the lake and dips there to 294 m,"
            " 9.5 m below the gates, where the shore keeps its walkable bank. The gates are still at the bottom of the"
            " drive in its sense of the drive's far end from the castle (as the bottom of a garden), 16 m below the"
            " castle's front; and walking down from the castle the dip hides them until the cabin's terrace, 449 m out,"
            " where they come into view with the drive falling 4 m to them.",
            q(G, "They made their way down the drive in the gathering twilight."),
            q(G, "as the gates at the bottom of the drive came into view"),
            q(G, "The slope of the grounds is probably quite uneven and lumpy"),
            choice(choices["drive_dip"])),
        "willow-on-its-knoll": decision(
            "The Whomping Willow stands on a knoll of its own, 24 m above the hollow it stood in at this site, where its"
            " crown was hidden from every greenhouse (0.5 Mm\u00b3 of fill, flanks of 1 in 2.6). It stands 13.5 m above"
            " the lake; the top of its crown, 11.5 m up, clears the ground by 0.7 to 1.7 m from each of the seven"
            " northern greenhouses, while a person at its foot stays hidden from every greenhouse.",
            q(G, "From the greenhouses it is possible to see at least the upper part of the Whomping Willow"),
            q(G, "yet not possible to see a person standing at its base"),
            q(G, "that obstruction is probably a bulge in the ground rather than trees"),
            choice(choices["willow_site"])),
        "willow-lawn": decision(
            "A lawn opens from the northern greenhouses to the Willow, 2.4 ha (328 by 118 m), so that it stands alone on"
            " its knoll: no tree stands over any of the seven greenhouses' lines to its crown, nor within 20 m of it. The"
            " light woods White Hound's map draws beside those greenhouses lay over the lines from three of them. The"
            " lawn's edge wanders, the woods thin out over 12 m beyond it, where the lawn gives way in tufts to such"
            " leaf litter as lies under them, and four trees stand scattered on it, clear of"
            " the lines, the paths and the vegetable patch, which it takes in round the greenhouses with its beds and"
            " hollow as they were. From each greenhouse the trees now hide none of the points on the Willow's top quarter"
            " (8.6 to 11.5 m above its knoll) that the ground leaves in sight: 60, 9, 22, 7, 8, 6 and 12 of 60 show."
            " Through the 20 m sight line it replaces they hid all 60 from one greenhouse, all 7 from another and 4 of 6"
            " from a third.",
            q(G, "so you wouldn't expect there to be other trees near it to obscure it from view"),
            q(G, "Harry spotted the Whomping Willow in the distance"),
            {"source": "user", "quote": CLEARINGS_PLAUSIBLE},
            choice(choices["willow_site"])),
        "station-wood": decision(
            "From the platform and all the way down the walk to the lake the castle stays hidden, but for glimpses through"
            " the pines, until the path rounds its last bend and opens on to the lake. At this site the station terrace"
            " stands 7.6 m above the flat ground the lake's rim fills west of it, where White Hound's pines grew barely"
            " above the eye: the castle stood in full view from the platform (81 to 132 of 150 points on its upper parts)"
            " and from the top of the lake path (51). A wooded shelf behind the boundary wall, rising 3 m above the"
            " terrace over 20 m, and a wooded ridge west of his path to the boats, its crest 8.4 m above the terrace, both"
            " of dense pines (0.10 Mm\u00b3 of fill, 1.25 ha), leave at most 5 of the 150 in sight from the platform and 2"
            " from the walk; round the bend 73 show.",
            q(G, "Yeh'll get yer firs' sight o' Hogwarts in a sec"),
            q(G, "The narrow path had opened suddenly on to the edge of a great black lake."),
            q(G, "there is some sort of rise in the ground between the station/path and the lake"),
            q(G, "we know the castle cannot be seen from either station or path, until the path turns the final bend and"
                 " opens out onto the lakeside"),
            q(G, "There could be a wall or a bank running through the trees, and they pass through a gap in it in the dark,"
                 " without noticing."),
            choice(choices["book_evidence_rule"])),
        "train-screen": decision(
            "From the train's last approach, from the tunnel's portal down the main line, through the junction and along"
            " the loop to the platform's end, the castle stays out of sight but for glimpses through the trees, so the"
            " first-years' first sight of it is still where the path to the boats opens on to the lake. At this site the"
            " portal stands 180 m up the line, and from the 225 m of open line and loop below it the castle stood in full"
            " view (64 to 80 of 150 points on its upper parts). A ridge on the castle's side of the line, its crest 10 m"
            " above the formation 24 m out and its flanks 30 m either side, planted with dense trees no nearer the track"
            " than 10 m along the 270 m from the cutting's mouth to the platform's end (0.12 Mm\u00b3 of fill, 0.89 ha),"
            " leaves at most 2 in sight from a seated or a standing passenger's eyes at the windows. White Hound's"
            " boundary wall of the grounds (3a) runs along the ridge, mostly 15 to 33 m from the track, on its broad"
            " back."
            " Costed against it: the tunnel extended to the junction as a covered way, as at Scardroy (155 m of box"
            " tunnel under 0.045 Mm\u00b3 of fill), still left the junction in view and needed the ridge's last 110 m"
            " too (0.022 Mm\u00b3); and no route within 1 in 40 and 60 m curves is hidden by the real ground, from"
            " which the castle shows everywhere within 350 m north of the junction.",
            q(G, "Yeh'll get yer firs' sight o' Hogwarts in a sec"),
            q(G, "Since this is the first glimpse the students get of the castle there must be a barrier \u2013 wall,"
                 " trees or rising land \u2013"),
            q(G, "preventing the castle from being seen from the train as it approaches the station"),
            {"source": "user", "quote": "The problem here is that the train is supposed to only show a hogwarts coming"
                                        " into view at a specific point."},
            note("user", "orchestrated choice: after the user's ruling, the orchestrator chose the ridge over the covered"
                         " way, since no route the ground hides exists, and reported it to the user"),
            note("data", "the ridge was costed with an 8 m crest, which left 9 of the 150 in sight from a seated"
                         " passenger's eyes and 11 from a standing one's (2.25 and 2.7 m above the rail top, the"
                         " carriages' windows standing 1.95 to 2.8 m above it); lane D raised the crest to 10 m")),
        "hogsmeade-high-fringe": decision(
            "Hogsmeade's High Street stands below the gates: its buildings' median ground 2.1 m below them, Honeydukes"
            " 3.6 m and the Three Broomsticks 2.1 m, so the way from the village to the gates goes up, as does the lane"
            " from the station (by 11 m). Its upper fringe stands above them: Dervish & Banges by 1.2 m, the distillery by"
            " 2.4 m, the church by 11 m, the school by 14.5 m, and the Shrieking Shack, above the rest of the village, by"
            " 24 m. White Hound's conclusion that even the village's highest point lies below the gates is not met; the"
            " books' own evidence is.",
            q(G, "We are told at least twice that the trio go up from Hogsmeade to the Hogwarts gates"),
            q(G, "They trudged up the dark, deserted lane, following the freshly made carriage tracks."),
            q(H, "It stood a little way above the rest of the village"),
            q(G, "even the highest point is still lower than the front gates"),
            choice(choices["book_evidence_rule"])),
        "shack-garden": decision(
            "From the Shrieking Shack's one unblocked window, glazed in the middle of its ground floor's castle-facing"
            " wall, the castle stands in view over the Shack's dank, overgrown garden. The garden is 0.54 ha of rank"
            " grass, nettles and bramble, in dull greens and browns, reaching 25 m behind the Shack and 91 m from it down"
            " towards the castle, kept clear of trees over the window's view of the castle's breadth for 60 m beyond the"
            " wall. Its edge wanders; across 16 m of it, where the trees on the hill thin out, its rank growth gives way"
            " to the hill's meadow in tufts; and three trees stand scattered in it, off its tracks. 103 of 150"
            " points on the castle's upper parts show from the window, as through the straight strip it replaces; under"
            " the trees White Hound's map draws on the hill, 38 showed. The ground hides none of them; the rest are"
            " hidden by trees further off.",
            q(H, "The single unblocked window revealed distant bursts of light where the castle stood"),
            q(H, "one of which has been unboarded and commands a view of the castle"),
            q(H, "with its boarded windows and dank overgrown garden"),
            q(H, "There must be trees either in the garden or nearby"),
            {"source": "user", "quote": CLEARINGS_PLAUSIBLE},
            note("user", "orchestrated choice: the orchestrator had the view cleared as X's default under the user's"
                         " rule on book evidence, then shaped as the Shack's garden under the user's rule on clearings,"
                         " and had its ground made rank growth fading into the meadow after the trunk's tour showed it"
                         " as a sharp-edged purple patch"),
            choice(choices["book_evidence_rule"])),
        "clearings-plausible": decision(
            "Every clearing in the trees either follows a feature that explains it or reads as a place. The railway"
            " clears its earthworks and its lineside, within 10 m of the open line; the Forest has corridors along the"
            " A890 and the Kyle railway, and ends short of the A832 and round Achnasheen. The places are the Willow's"
            " lawn, the Shack's garden, the homesteads round the buildings the Forest covers, Aragog's hollow and Grawp's"
            " clearing. Each is at least 0.25 ha in area and 30 m across, with no straight stretch of edge 60 m long and"
            " no two that run parallel and face each other across it, and the wood thins out towards it. In the two"
            " cleared for lines of sight a few trees stand, and their ground does not end at a line either: it gives way"
            " to the ground round it in tufts across 16 m of that edge, and no patch of ground ends at a line"
            " (scrubland-as-scrub). In Grawp's clearing a few small pines stand, and Aragog's hollow is bare, as the"
            " books have them. The ground of the homesteads and of those two places is the real ground's, under Forest"
            " that thins out over 24 m or 16 m. The hollows the station's wooded rises enclose are made up keeping their"
            " trees, not as meadow. Otherwise trees are only kept off streams, the lake's edge, walls, buildings' plots"
            " and the ground within a crown's reach of the pitch's stands.",
            {"source": "user", "quote": CLEARINGS_PLAUSIBLE}),
        "scrubland-as-scrub": decision(
            "White Hound's three maps draw rough scrubland, as their legend names it. In the 2 m layer that covers his"
            " grounds and Hogsmeade maps the scene draws it as scrub: shrubby ground of olive green and bracken brown,"
            " not heather in bloom. Nothing places heath or moor on the ground his maps draw: his legend has no such"
            " ground, and he calls the land round Hogsmeade's fields moorland only in general. Beyond that layer, over"
            " most of the Forest map's scrubland, the ground keeps the Sentinel-2 imagery's colours, tinted green where"
            " the Forest map draws fields."
            " Every patch of ground, the scrub, the woods' floor and the lawns as well as the clearings' ground, gives"
            " way to the ground round it in tufts across 16 m of its edge rather than ending at a line, but where a wall,"
            " a path, the railway's works or the shore explains its line. The woods' floor lies only under their canopy,"
            " beyond which its tufts would read as specks on open ground; the station wood and the train's screen stand"
            " dense to a line, so their floor gives way within it, to grass under their outer trees.",
            q(H, "Rough scrubland, not suitable for farming."),
            q(H, "scrubby patch of ground which rose to meet the rocky foot of the mountain"),
            q("wh-setting", "a fairly fertile valley containing a patchwork of fields surrounded by unmodified moorland"),
            {"source": "user", "quote": CLEARINGS_PLAUSIBLE},
            note("user", "orchestrated choice: after the trunk's tour showed the maps' scrub as heather in bloom with"
                         " hard edges, the orchestrator had it drawn as scrub, heather kept only where a source says"
                         " heath or moor, and every patch of ground blended as the clearings' ground is")),
    }

IMAGE_NOTES = {
    "train-route_through_England.jpg": ("context", [], []),
    "train-route_through_Scotland.jpg": ("context", [], []),
    "Glenuaig_Lodge.jpg": ("context", [], []),
    "train-route_via_Glenfinnan.jpg": ("context", [], []),
}
