"""tools.inspection: the qualification's inventory of every distinctly rendered or defined object, the framing of each
one's close-up and context views, and the packets its reviewers get."""
import copy
import json
import math
import unittest

import numpy as np

from tools.inspection import contact, frame, inventory, motion, review, shoot


def unit(cid, family, area, k=0, asset="village-house", regular=True, hips=False, drop=0.5, x=0.0, z=0.0):
    return {"cid": cid, "part": 0, "k": k, "asset": asset, "family": family, "area": area, "x": x, "z": z,
            "high": family == "shop", "regular": regular, "hips": hips, "drop": drop, "roof": "village-thatch",
            "moved": cid in (23, 28)}


class CoverageTests(unittest.TestCase):
    """The inventory of the real scene leaves nothing the manifest, the tour or the renderer defines without an
    object."""

    @classmethod
    def setUpClass(cls):
        cls.world = inventory.World.load()
        cls.objects = inventory.build(cls.world)
        cls.gaps = inventory.uncovered(cls.objects, cls.world)
        cls.by_id = {o.id: o for o in cls.objects}

    def test_every_manifest_asset_has_an_object(self):
        self.assertEqual(self.gaps["assets"], [])

    def test_every_decision_is_on_an_object_or_named_as_not_visual(self):
        self.assertEqual(self.gaps["decisions"], [])
        for name, reason in inventory.NOT_VISUAL.items():
            with self.subTest(decision=name):
                self.assertIn(name, self.world.manifest["decisions"])
                self.assertTrue(reason)
                self.assertFalse(any(name in o.decisions for o in self.objects), "a decision is either seen or not")

    def test_a_decision_added_to_the_street_furniture_reaches_its_lamps(self):
        """The High Street's lamps take the street furniture's decisions from the manifest, so one that a lane adds
        to the asset is covered once it merges (high-street-footway, lane B), and never named before it does."""
        world = copy.copy(self.world)
        world.manifest = copy.deepcopy(self.world.manifest)
        furniture = next(a for a in world.manifest["assets"] if a["id"] == "street-furniture")
        furniture["decisions"] = [*(furniture.get("decisions") or ()), "a-new-street-decision"]
        lamps = [o for o in inventory.village_objects(world) if o.id.startswith("village/lamp-")]
        self.assertTrue(lamps)
        for o in lamps:
            with self.subTest(object=o.id):
                self.assertIn("a-new-street-decision", o.decisions)

    def test_a_decision_only_the_dark_shows_is_covered_only_by_an_object_shot_after_dark(self):
        """village-night-windows rides on the village's houses, shot at 15:00 when no window is lit: the High Street
        at dusk covers it, and without that shot it is uncovered."""
        by_day = [o for o in self.objects if o.id != "village/dusk"]
        self.assertTrue(any("village-night-windows" in o.decisions for o in by_day))
        self.assertIn("village-night-windows", inventory.uncovered(by_day, self.world)["decisions"])
        self.assertIn("village-night-windows", self.by_id["village/dusk"].decisions)

    def test_every_label_of_a_built_asset_has_an_object(self):
        self.assertEqual(self.gaps["labels"], [])

    def test_every_waypoint_has_a_still_and_every_caption_its_subjects(self):
        self.assertEqual(self.gaps["waypoints"], [])
        stills = [o for o in self.objects if o.kind == "tour"]
        self.assertEqual(len(stills), len(self.world.waypoints))
        for o in stills:
            for s in o.subjects:
                self.assertIn(s, self.by_id, f"{o.id} names {s}")

    def test_every_recorded_clearing_and_linear_clearing_has_an_object(self):
        self.assertEqual(self.gaps["clearings"], [])

    def test_every_ground_class_has_an_object(self):
        self.assertEqual(self.gaps["classes"], [])

    def test_every_footprint_group_and_excluded_group_has_an_object(self):
        self.assertEqual(self.gaps["footprints"], [])

    def test_every_style_family_of_the_village_has_a_representative(self):
        families = {u["family"] for u in self.world.units if u["asset"] in inventory.UNNAMED}
        samples = [o.sample for o in self.objects if o.kind == "sample" and set(o.assets) & set(inventory.UNNAMED)]
        self.assertTrue(families)
        self.assertEqual({f for f in families if any(f" {f} buildings" in s for s in samples)}, families)

    def test_ids_are_unique_and_every_owner_is_a_session(self):
        ids = [o.id for o in self.objects]
        self.assertEqual(len(ids), len(set(ids)))
        for o in self.objects:
            with self.subTest(object=o.id):
                self.assertIn(o.owner, inventory.SESSIONS)
                self.assertIn(o.kind, inventory.KINDS)

    def test_every_object_has_a_place_a_size_and_known_decisions(self):
        decisions = self.world.manifest["decisions"]
        for o in self.objects:
            with self.subTest(object=o.id):
                self.assertEqual(len(o.at), 3)
                self.assertTrue(all(math.isfinite(v) for v in o.at))
                self.assertGreater(o.size, 0.0)
                for d in o.decisions:
                    self.assertIn(d, decisions)

    def test_samples_and_outliers_say_what_they_stand_for(self):
        for o in self.objects:
            if o.kind in ("sample", "outlier"):
                with self.subTest(object=o.id):
                    self.assertRegex(o.sample, r"\d+")      # the size of the population it is drawn from

    def test_the_inventory_round_trips_through_json(self):
        rows = json.loads(json.dumps([o.as_dict() for o in self.objects]))
        self.assertEqual([inventory.Obj.from_dict(r) for r in rows], self.objects)


class SamplingTests(unittest.TestCase):
    """Hogsmeade's unnamed buildings: one representative per style family, of median size, and the outliers."""

    UNITS = [unit(1, "shop", 40.0), unit(2, "shop", 50.0), unit(3, "shop", 60.0),
             unit(4, "terrace", 60.0), unit(5, "terrace", 70.0), unit(6, "terrace", 200.0),
             unit(7, "cottage", 30.0, hips=True), unit(8, "cottage-dormer", 45.0), unit(9, "cottage-dormer", 48.0),
             unit(23, "cottage-dormer", 47.0), unit(10, "villa", 120.0), unit(11, "outshot", 12.0, asset="village-wing"),
             unit(12, "terrace", 65.0, regular=False), unit(13, "terrace", 66.0, drop=4.0),
             unit(14, "kirk", 400.0, asset="kirk")]

    def picks(self, units=None):
        return {(u["cid"], kind): reason for u, kind, reason in inventory.village_samples(units or self.UNITS)}

    def test_one_representative_per_style_family_of_median_size(self):
        reps = {cid for cid, kind in self.picks() if kind == "sample"}
        self.assertEqual(reps, {2, 13, 7, 23, 10, 11})

    def test_the_moved_largest_smallest_steepest_irregular_and_hipped_units_are_outliers(self):
        outliers = {cid for cid, kind in self.picks() if kind == "outlier"}
        # 23 moved, but it already represents its family; 6 the largest, 11 the smallest (already a representative),
        # 13 the steepest (already one), 12 irregular; 7 the only hipped roof already represents the cottages
        self.assertEqual(outliers, {6, 12})

    def test_named_buildings_are_not_sampled(self):
        self.assertFalse(any(cid == 14 for cid, _ in self.picks()))

    def test_each_pick_names_its_population(self):
        picks = self.picks()
        self.assertIn("of the 5 terrace buildings", picks[13, "sample"])
        self.assertIn("largest", picks[6, "outlier"])

    def test_a_moved_unit_that_does_not_represent_its_family_is_an_outlier(self):
        units = [u for u in self.UNITS if u["cid"] != 8] + [unit(8, "cottage-dormer", 47.0)]
        picks = self.picks(units)
        self.assertIn((23, "outlier"), picks)
        self.assertIn("moved", picks[23, "outlier"])

    def test_spread_samples_take_the_median_and_the_extremes(self):
        items = [("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 10.0), ("e", 0.5)]
        picked = inventory.spread_samples(items, "hedges")
        self.assertEqual([(i, k) for i, k, _ in picked], [("b", "sample"), ("d", "outlier"), ("e", "outlier")])
        self.assertIn("of the 5 hedges", picked[0][2])
        self.assertEqual([(i, k) for i, k, _ in inventory.spread_samples(items[:2], "x")], [("a", "sample")])

    def to_world(self, i, j):
        return 2.0 * j - 10.0, 2.0 * i - 10.0           # 2 m texels, east by column and north by row, (5, 5) at 0, 0

    def test_a_ground_class_is_framed_from_its_open_side(self):
        east, north = np.zeros((10, 10), bool), np.zeros((10, 10), bool)
        east[:, 9], north[9, :] = True, True
        self.assertAlmostEqual(inventory.nearest_bearing(east, 5, 2, self.to_world), 90.0)
        self.assertAlmostEqual(inventory.nearest_bearing(north, 2, 5, self.to_world), 0.0)

    def test_the_forest_floor_is_taken_at_the_woods_edge_nearest_the_castle(self):
        wood = np.zeros((10, 10), bool)
        wood[:, :6] = True                              # the wood west of column 6, open ground east of it
        i, j = inventory.edge_texel(wood, ~wood, 2.0, (4.0, 6.0), self.to_world)
        self.assertEqual((i, j), (5, 4))                # 2 texels (4 m) in from the edge, level with the castle


class FrameTests(unittest.TestCase):
    """Close-up and context cameras, and how much of a subject a rendered view shows or hides."""

    def test_the_close_up_frames_the_subject_at_its_share_of_the_height(self):
        d = frame.close_distance(10.0)
        self.assertAlmostEqual(math.degrees(2 * math.atan(5.0 / d)) / frame.FOV_DEG, frame.CLOSE_FILL, places=6)

    def test_the_context_stands_further_back(self):
        self.assertGreaterEqual(frame.context_distance(10.0), frame.CONTEXT_MIN_M)
        self.assertAlmostEqual(frame.context_distance(200.0), frame.CONTEXT_X * frame.close_distance(200.0))

    def test_the_first_candidate_looks_from_the_sun_side(self):
        # the sun in the south-west (azimuth 225): the camera stands to the south-west of the subject, looking north-east
        az, el = frame.candidates(225.0)[0]
        self.assertEqual(az, 225.0)
        eye = frame.eye((0.0, 0.0, 0.0), az, el, 100.0)
        self.assertLess(eye[0], 0.0)
        self.assertLess(eye[2], 0.0)
        self.assertGreater(eye[1], 0.0)

    def test_the_candidates_cover_every_side_once(self):
        azimuths = sorted({az % 360 for az, _ in frame.candidates(10.0)})
        self.assertEqual(azimuths, sorted((10.0 + 45 * k) % 360 for k in range(8)))

    def test_yaw_and_pitch_point_the_camera_at_the_target(self):
        yaw, pitch = frame.aim((0.0, 10.0, 0.0), (10.0, 0.0, 10.0))
        self.assertAlmostEqual(yaw, math.atan2(10.0, 10.0))
        self.assertAlmostEqual(pitch, -math.atan2(10.0, math.hypot(10.0, 10.0)))

    def view(self, w=64, h=36):
        # a camera at the origin looking north (+z), no jitter; rot maps camera (right, up, forward) to world
        return {"res": [w, h], "pos": [0.0, 0.0, 0.0], "rot": [1, 0, 0, 0, 1, 0, 0, 0, 1], "tanH": 0.5,
                "jit": [0.0, 0.0]}

    def samples(self, view, hit):
        w, h = view["res"]
        rows = []
        for y in range(2, h, 4):
            for x in range(2, w, 4):
                t, mat = hit(frame.rays(view, np.array([[x, y]], float))[0])
                rows.append([x, y, t, mat])
        return np.array(rows, float)

    def test_a_subject_in_plain_view_is_visible_and_unoccluded(self):
        view, box = self.view(), (-2.0, -2.0, 18.0, 2.0, 2.0, 22.0)
        enter = lambda d: frame.ray_box(np.zeros((1, 3)), d[None], np.array(box[:3]), np.array(box[3:]))[0][0]

        def hit(d):
            t = enter(d)
            return (t, 7) if np.isfinite(t) else (500.0, 0)
        visible, occluded = frame.score(view, self.samples(view, hit), box)
        self.assertGreater(visible, 0.0)
        self.assertEqual(occluded, 0.0)

    def test_something_in_front_of_the_subject_occludes_it(self):
        view, box = self.view(), (-2.0, -2.0, 18.0, 2.0, 2.0, 22.0)
        enter = lambda d: frame.ray_box(np.zeros((1, 3)), d[None], np.array(box[:3]), np.array(box[3:]))[0][0]

        def hit(d):
            t = enter(d)
            if not np.isfinite(t):
                return 500.0, 0
            return (t * 0.5, 3) if d[0] < 0 else (t, 7)     # a tree hides the subject's left half
        visible, occluded = frame.score(view, self.samples(view, hit), box)
        self.assertAlmostEqual(occluded, 0.5, delta=0.1)
        self.assertAlmostEqual(visible, 0.5, delta=0.1)

    def test_the_first_candidate_that_shows_its_subject_is_chosen(self):
        self.assertEqual(frame.choose([(0.05, 0.6), (0.2, 0.05), (0.3, 0.0)]), 1)
        self.assertEqual(frame.choose([(0.05, 0.6), (0.01, 0.4)]), 1)   # none passes: the least hidden share wins


class ShootTests(unittest.TestCase):
    """The capture tool's geometry: where a view's camera and target stand, the context view it derives, the box it
    scores, and the files it writes."""

    def ground(self, x, z):
        return 100.0 + 0.1 * x                          # a slope rising to the east

    def test_own_heights_stand_above_the_ground_under_each_point(self):
        v = frame.View("v", (10.0, 5.0, 0.0), (-20.0, 2.0, 0.0), 15.0, "own")
        eye, target = shoot.resolve(v, self.ground, lake=(0.0, 0.0))
        self.assertEqual(eye, (10.0, 106.0, 0.0))
        self.assertEqual(target, (-20.0, 100.0, 0.0))

    def test_target_heights_stand_above_the_targets_ground_and_the_camera_clears_its_own(self):
        v = frame.View("v", (10.0, 1.0, 0.0), (-20.0, 2.0, 0.0), 15.0, "target")
        eye, target = shoot.resolve(v, self.ground, lake=(0.0, 0.0))
        self.assertEqual(target, (-20.0, 100.0, 0.0))
        self.assertEqual(eye, (10.0, 101.0 + shoot.CLEAR_M, 0.0))      # 99 would stand under the ground at x = 10

    def test_night_is_when_the_sun_is_well_down_not_when_every_lamp_is_lit(self):
        import asyncio

        class Page:                                     # the sun sets at 20:00 and sinks 6 degrees an hour
            async def evaluate(self, script, h):
                el = 6.0 * (20.0 - h)
                return {"el": el, "dusk": min(1.0, max(0.0, -el / 2.5)), "az": 300.0}
        named = asyncio.run(shoot.hours_by_name(Page()))
        self.assertEqual(named["civil dusk"], 20.1)     # the review sheets' civil dusk: the sun just under
        self.assertEqual(named["night"], 22.05)         # 12 degrees down; every lamp is lit from 20:27

    def test_only_names_exact_ids_or_prefixes(self):
        objs = [inventory.Obj(id=i, title=i, family=i.split("/")[0], kind="part", at=(0.0, 0.0, 0.0), size=5.0)
                for i in ("castle/court-house-1", "castle/court-house-12", "grounds/hut", "grounds/hut-front")]
        ids = lambda only: [o.id for o in shoot.chosen(objs, only)]          # noqa: E731
        self.assertEqual(ids(["castle/court-house-1", "grounds/hut"]), ["castle/court-house-1", "grounds/hut"])
        self.assertEqual(ids(["grounds/"]), ["grounds/hut", "grounds/hut-front"])
        self.assertEqual(ids(["castle/court-house-*"]), ["castle/court-house-1", "castle/court-house-12"])
        self.assertEqual(ids(None), [o.id for o in objs])

    def test_a_camera_with_no_place_stands_over_the_lakes_centre(self):
        v = frame.View("v", (None, 6.0, None), (0.0, 30.0, 0.0), 15.0, "own")
        eye, _ = shoot.resolve(v, self.ground, lake=(120.0, -400.0))
        self.assertEqual(eye, (120.0, 118.0, -400.0))

    def obj(self, views=(), size=10.0, box=()):
        return inventory.Obj(id="grounds/x", title="x", family="grounds", kind="part", at=(0.0, 2.0, 0.0), size=size,
                             views=views, box=box)

    def test_the_context_looks_from_the_close_ups_side_from_further_back(self):
        close = frame.View("c", (10.0, 3.0, 10.0), (0.0, 2.0, 0.0), 16.0, "target")
        context = shoot.context_view(self.obj((close,)), close, self.ground, lake=(0.0, 0.0))
        self.assertAlmostEqual(frame.bearing(context), 45.0, places=4)
        self.assertGreater(math.dist(context.eye, context.target), frame.CONTEXT_MIN_M - 0.05)   # to the centimetre
        self.assertEqual(context.hour, 16.0)

    def test_a_given_second_view_is_the_context(self):
        close = frame.View("c", (10.0, 3.0, 10.0), (0.0, 2.0, 0.0), 16.0, "own")
        second = frame.View("d", (50.0, 9.0, 50.0), (0.0, 2.0, 0.0), 21.5, "own")
        self.assertEqual(shoot.context_view(self.obj((close, second)), close, self.ground, (0.0, 0.0)), second)

    def test_the_scored_box_is_the_objects_own_or_a_cube_of_its_size_round_its_target(self):
        self.assertEqual(shoot.subject_box(self.obj(box=(1, 2, 3, 4, 5, 6)), self.ground), (1, 2, 3, 4, 5, 6))
        # the cube stops just under the ground: a ray into it below ground would hit the ground in front of the box
        self.assertEqual(shoot.subject_box(self.obj(size=10.0), self.ground), (-5.0, 99.5, -5.0, 5.0, 107.0, 5.0))
        hung = inventory.Obj(id="village/sign", title="s", family="village", kind="part", at=(0.0, 5.0, 0.0), size=2.0)
        self.assertEqual(shoot.subject_box(hung, self.ground), (-1.0, 104.0, -1.0, 1.0, 106.0, 1.0))

    def test_shot_files_are_named_for_their_object_and_role(self):
        self.assertEqual(shoot.file_name("castle/astronomy-tower", "close"), "castle__astronomy-tower-close.png")
        self.assertEqual(shoot.record_name("tour/01-hogwarts-castle"), "tour__01-hogwarts-castle.json")

    BASELINE = ("let free = false, fp = [0, 0, 0], fyaw = 0, fpitch = 0;\n  function lighting(hBST) { return 1; }\n"
                "  window.__hog = {\n    pause(p) { paused = p; }, dbg(n) { dbg = n; }, path: pathAt, ground,\n  };\n")

    def test_the_baseline_gets_the_camera_and_light_hooks_the_shots_need(self):
        hooked = shoot.hooked_baseline(self.BASELINE)
        self.assertIn("path: pathAt, ground, " + shoot.BASELINE_HOOKS, hooked)
        self.assertEqual(hooked.count("camera(pos, yaw, pitch)"), 1)

    def test_a_baseline_without_the_anchor_or_the_free_camera_is_refused(self):
        for text in (self.BASELINE.replace("path: pathAt, ground,", "path: pathAt,"),
                     self.BASELINE.replace("fyaw", "yaw0"), self.BASELINE + self.BASELINE):
            with self.subTest(text=text[:30]), self.assertRaises(ValueError):
                shoot.hooked_baseline(text)


class ContactTests(unittest.TestCase):
    """The asset contact sheet: per family, each object's baseline close-up, close-up and context side by side, with
    the decisions it answers for."""

    def obj(self, oid, kind="part", family="grounds", decisions=("hut-plan",)):
        return inventory.Obj(id=oid, title=oid.split("/")[1], family=family, kind=kind, at=(0.0, 0.0, 0.0), size=5.0,
                             decisions=decisions)

    def test_each_object_gets_a_row_of_baseline_close_up_and_context(self):
        objs = [self.obj("grounds/a"), self.obj("grounds/b", kind="terrain", family="grounds")]
        records = {"grounds/a": {"shots": [{"file": "a-close.png", "role": "close"}, {"file": "a-ctx.png",
                                                                                       "role": "context"},
                                            {"file": "a-base.png", "role": "baseline"}]},
                   "grounds/b": {"shots": [{"file": "b-close.png", "role": "close"}, {"file": "b-ctx.png",
                                                                                       "role": "context"}]}}
        rows = contact.rows(objs, records)
        self.assertEqual([r.cells for r in rows], [("a-base.png", "a-close.png", "a-ctx.png"),
                                                   (None, "b-close.png", "b-ctx.png")])
        self.assertEqual(rows[0].decisions, ("hut-plan",))

    def test_objects_without_shots_and_tour_stills_are_left_out(self):
        objs = [self.obj("grounds/a"), self.obj("tour/01-x", kind="tour", family="tour")]
        self.assertEqual(contact.rows(objs, {"tour/01-x": {"shots": [{"file": "t.png", "role": "tour"}]}}), [])

    def test_the_sheet_places_each_cell_in_its_column(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, colour in (("base.png", (255, 0, 0)), ("close.png", (0, 255, 0)), ("ctx.png", (0, 0, 255))):
                Image.new("RGB", (64, 36), colour).save(d / name)
            row = contact.Row("grounds/a", "a", "", ("base.png", "close.png", "ctx.png"), ("hut-plan",))
            out = contact.compose(d / "sheet.png", "grounds", [row], d, {"hut-plan": "The hut faces the castle."})
            img = Image.open(out).convert("RGB")
            for column, colour in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
                x, y = contact.cell_origin(0, column)
                self.assertEqual(img.getpixel((x + contact.THUMB[0] // 2, y + contact.THUMB[1] // 2)), colour)


class MotionTests(unittest.TestCase):
    """Continuous-motion sequences: one explicitly rendered frame per step along the tour, and the frames whose change
    from the one before stands out (pops)."""

    def test_a_sequence_runs_at_its_frame_rate_after_a_preroll(self):
        times = motion.schedule(100.0, seconds=1.0, fps=10, preroll=3)
        self.assertEqual(len(times), 13)
        self.assertAlmostEqual(times[0], 99.7)
        self.assertAlmostEqual(times[3], 100.0)
        self.assertAlmostEqual(times[-1], 100.9)

    PAGE = ("  let simT = 0;\n  function render(pos, f) {\n    return 1;\n  }\n\n  function loop(now) {\n    draw();\n  }\n"
            "\n  window.__hog = {\n    ready: false,\n  };\n")

    def test_the_page_copy_stops_its_loop_and_renders_on_request(self):
        text = motion.instrumented(self.PAGE)
        self.assertIn("function loop(now) {\n    if (motionStop) return;\n", text)
        self.assertLess(text.index("window.__motion"), text.index("window.__hog = {"))

    def test_a_page_without_the_anchors_is_refused(self):
        for text in (self.PAGE.replace("function render(pos, f) {", "function draw(pos, f) {"),
                     self.PAGE.replace("function loop(now) {", "function tick(now) {"), self.PAGE + self.PAGE):
            with self.subTest(text=text[:20]), self.assertRaises(ValueError):
                motion.instrumented(text)

    def test_a_change_far_larger_than_its_neighbours_is_a_pop(self):
        series = [1.0, 1.1, 0.9, 1.0, 6.0, 1.0, 1.05, 0.95]
        self.assertEqual(motion.pops(series), [4])
        self.assertEqual(motion.pops([1.0] * 8), [])

    def test_a_pop_is_recorded_as_the_frame_that_jumps(self):
        # changes[k] runs from frame k to frame k + 1, so the fifth change shows in frame 5
        self.assertEqual(motion.pop_frames([1.0, 1.1, 0.9, 1.0, 6.0, 1.0, 1.05, 0.95]), [5])

    WPS = [[0, 0, 0, 0, 0, 0, "Castle"], [0, 0, 0, 0, 0, 0, "Pitch"], [0, 0, 0, 0, 0, 0, None],
           [0, 0, 0, 0, 0, 0, "Pitch"]]

    def test_a_sequence_starts_at_its_own_pass_of_a_repeated_caption(self):
        times = [10.0, 20.0, 30.0, 40.0]
        self.assertEqual(motion.pass_time(self.WPS, times, "Castle"), 10.0)
        self.assertEqual(motion.pass_time(self.WPS, times, "Pitch", 1), 20.0)
        self.assertEqual(motion.pass_time(self.WPS, times, "Pitch", 2), 40.0)

    def test_a_repeated_caption_without_its_pass_or_a_missing_one_is_refused(self):
        times = [10.0, 20.0, 30.0, 40.0]
        for caption, occurrence in (("Pitch", None), ("Pitch", 3), ("Lake", None)):
            with self.subTest(caption=caption, occurrence=occurrence), self.assertRaises(ValueError):
                motion.pass_time(self.WPS, times, caption, occurrence)

    def test_every_sequence_names_a_pass_the_page_flies(self):
        from tools.paths import PAGE
        from tools.review import tour
        wps = tour.waypoints(PAGE.read_text(encoding="utf-8", errors="replace"))
        for name, (caption, occurrence, _, _) in motion.SEQUENCES.items():
            with self.subTest(sequence=name):
                motion.pass_time(wps, list(range(len(wps))), caption, occurrence)

    def test_the_change_between_frames_is_their_mean_absolute_difference(self):
        a = np.zeros((4, 4), np.uint8)
        b = a.copy()
        b[:2] = 100                                          # half the pixels change by 100
        self.assertEqual(motion.changes([a, b, b]), [50.0, 0.0])


class ReviewTests(unittest.TestCase):
    """Reviewer batches, packets and the collation of their verdicts."""

    def objects(self, n, family="village"):
        return [inventory.Obj(id=f"{family}/o{k}", title=f"Object {k}", family=family, kind="part",
                              at=(0.0, 0.0, 0.0), size=5.0, decisions=("hut-plan",)) for k in range(n)]

    def test_batches_hold_about_eight_to_twelve_objects_each_once(self):
        for n in (8, 12, 13, 25, 31, 97):
            objs = self.objects(n)
            got = review.batches(objs)
            with self.subTest(n=n):
                self.assertEqual(sorted(o.id for b in got for o in b), sorted(o.id for o in objs))
                self.assertTrue(all(review.LEAST <= len(b) <= review.MOST for b in got), [len(b) for b in got])

    def test_only_shot_objects_are_reviewed(self):
        objs = self.objects(3)
        images = {objs[0].id: ["a-close.png"], objs[2].id: ["c-close.png"]}
        self.assertEqual([o.id for o in review.reviewable(objs, images)], [objs[0].id, objs[2].id])

    def test_a_batch_keeps_to_one_owner_where_it_can(self):
        objs = self.objects(10, "village") + self.objects(10, "grounds")
        for b in review.batches(objs):
            self.assertEqual(len({o.owner for o in b}), 1)

    def test_tour_stills_are_batched_apart_from_objects(self):
        stills = [inventory.Obj(id=f"tour/{k:02d}-x", title="t", family="tour", kind="tour", at=(0.0, 0.0, 0.0),
                                size=100.0) for k in range(1, 6)]
        objs = self.objects(9, "castle") + stills          # the castle and the tour share an owner, the trunk keeper
        for b in review.batches(objs):
            self.assertEqual(len({o.kind == "tour" for o in b}), 1)

    def test_the_packet_quotes_every_decision_verbatim_and_lists_the_checklist(self):
        manifest = {"decisions": {"hut-plan": {"text": "The hut faces the castle.", "basis": [
            {"source": "wh-grounds", "quote": "It faces towards the castle"},
            {"source": "user", "choice": "hut: broad"}]}}}
        objs = self.objects(2)
        text = review.packet(objs, manifest, {o.id: [f"{o.id}-close.png", f"{o.id}-context.png"] for o in objs})
        self.assertIn('"It faces towards the castle"', text)
        self.assertIn("The hut faces the castle.", text)
        for item in review.CHECKLIST:
            self.assertIn(item, text)
        for o in objs:
            self.assertIn(f"{o.id}-close.png", text)

    def test_the_packet_gives_each_image_its_full_path(self):
        from pathlib import Path
        manifest = {"decisions": {"hut-plan": {"text": "t", "basis": []}}}
        objs = self.objects(1)
        text = review.packet(objs, manifest, {objs[0].id: ["a-close.png"]}, folder=Path("C:/shots"))
        self.assertIn(str(Path("C:/shots") / "a-close.png"), text)

    def test_a_blank_frame_is_one_of_almost_one_colour(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            crown, dusk = Path(d) / "crown.png", Path(d) / "dusk.png"
            Image.new("RGB", (192, 108), (8, 15, 9)).save(crown)
            Image.linear_gradient("L").point(lambda v: 5 + v * 10 // 255).resize((192, 108)).save(dusk)
            self.assertTrue(review.blank(crown))
            self.assertFalse(review.blank(dusk))           # dark, as a dusk context is, but it shows something

    def test_the_packet_marks_a_blank_baseline(self):
        manifest = {"decisions": {"hut-plan": {"text": "t", "basis": []}}}
        objs = self.objects(1)
        text = review.packet(objs, manifest, {objs[0].id: ["a-close.png", "a-baseline.png"]}, blank={"a-baseline.png"})
        self.assertIn(f"a-baseline.png {review.BLANK_NOTE}", text)
        self.assertNotIn(f"a-close.png {review.BLANK_NOTE}", text)

    def test_any_other_blank_shot_must_be_framed_again(self):
        roles = {"a-close.png": "close", "a-context.png": "context", "a-baseline.png": "baseline"}
        self.assertEqual(review.unusable({"a-baseline.png"}, roles), [])
        self.assertEqual(review.unusable({"a-context.png", "a-close.png", "a-baseline.png"}, roles),
                         ["a-close.png", "a-context.png"])

    def verdict(self, oid, verdict="defect", severity="major", owner="lane B", image="a-close.png"):
        return {"object": oid, "verdict": verdict, "severity": severity, "checks": ["geometry"],
                "evidence": [{"image": image, "region": "left third"}], "finding": "a gap", "owner": owner}

    def test_collation_groups_defects_by_owner_most_severe_first(self):
        objs = {o.id: o for o in self.objects(3)}
        images = {oid: ["a-close.png"] for oid in objs}
        found = review.collate([self.verdict("village/o0", severity="minor"), self.verdict("village/o1"),
                                self.verdict("village/o2", verdict="pass", severity="none")], objs, images)
        self.assertEqual(list(found), ["lane B"])
        self.assertEqual([f["object"] for f in found["lane B"]], ["village/o1", "village/o0"])

    def test_evidence_may_cite_an_image_by_the_full_path_its_packet_gives(self):
        objs = {o.id: o for o in self.objects(1)}
        found = review.collate([self.verdict("village/o0", image="C:\\shots\\a-close.png")], objs,
                               {"village/o0": ["a-close.png"]})
        self.assertEqual([v["object"] for v in found["lane B"]], ["village/o0"])

    def test_evidence_may_cite_an_images_review_copy(self):
        objs = {o.id: o for o in self.objects(1)}
        found = review.collate([self.verdict("village/o0", image="C:\\packets\\images\\a-close.jpg")], objs,
                               {"village/o0": ["a-close.png"]})
        self.assertEqual([v["object"] for v in found["lane B"]], ["village/o0"])

    def test_a_review_copy_is_a_jpeg_no_larger_than_a_model_reads(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "a-close.png"
            Image.fromarray(np.random.default_rng(1).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)).save(src)
            name = review.review_copy(src, d)
            with Image.open(Path(d) / name) as im:
                self.assertEqual((name, im.format, max(im.size)), ("a-close.jpg", "JPEG", review.REVIEW_EDGE))
            self.assertLess((Path(d) / name).stat().st_size, src.stat().st_size / 2)

    def test_collation_refuses_a_verdict_it_cannot_route(self):
        objs = {o.id: o for o in self.objects(1)}
        images = {"village/o0": ["a-close.png"]}
        for bad in (self.verdict("nobody"), self.verdict("village/o0", severity="huge"),
                    self.verdict("village/o0", owner="someone"), self.verdict("village/o0", image="other.png"),
                    {**self.verdict("village/o0"), "evidence": []}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                review.collate([bad], objs, images)

    def test_every_object_needs_a_verdict(self):
        objs = {o.id: o for o in self.objects(2)}
        with self.assertRaises(ValueError):
            review.collate([self.verdict("village/o0")], objs, {k: ["a-close.png"] for k in objs}, complete=True)

    def test_a_later_verdict_for_an_object_replaces_an_earlier_one(self):
        objs = {o.id: o for o in self.objects(1)}
        images = {"village/o0": ["a-close.png"]}
        passed = self.verdict("village/o0", verdict="pass", severity="none")
        self.assertEqual(review.collate([self.verdict("village/o0"), passed], objs, images), {})
        self.assertEqual([v["object"] for v in review.collate([passed, self.verdict("village/o0")], objs,
                                                              images)["lane B"]], ["village/o0"])

    def test_a_rerun_takes_only_the_named_objects_in_smaller_batches(self):
        objs = self.objects(11) + self.objects(2, "grounds")
        got = review.batches(shoot.chosen(objs, ["village/o1", "village/o2", "village/o3", "village/o4", "village/o5",
                                                 "grounds/"]), most=4)
        self.assertEqual(sorted(o.id for b in got for o in b),
                         ["grounds/o0", "grounds/o1"] + [f"village/o{k}" for k in range(1, 6)])
        self.assertTrue(all(len(b) <= 4 for b in got), [len(b) for b in got])

    def test_an_answers_verdicts_are_read_with_their_defects_however_the_json_is_laid_out(self):
        verdicts = [self.verdict("village/o0"), self.verdict("village/o1", verdict="pass", severity="none")]
        for text in (json.dumps(verdicts),
                     "Here are my verdicts.\n```json\n" + json.dumps(verdicts, indent=2) + "\n```\n",
                     'An example: [{"object": "x"}] is not the answer.\n' + json.dumps(verdicts)):
            with self.subTest(text=text[:40]):
                self.assertEqual(review.parse(text), verdicts)

    def test_an_answer_without_verdicts_is_refused(self):
        for text in ("no verdicts", '{"object": "x"}', "[1, 2]"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                review.parse(text)


if __name__ == "__main__":
    unittest.main()
