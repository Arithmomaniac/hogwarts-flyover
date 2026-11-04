"""The frame-time probe must time real, GPU-completed renders at the settings it reports (plan: measurement integrity)."""
import contextlib
import io
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mesh_probe  # noqa: E402

GPU = os.environ.get("HOG_GPU", "").lower()


class OrderTests(unittest.TestCase):
    def test_each_configuration_takes_every_position_and_follows_every_other_once(self):
        for n in (2, 4, 6):
            orders = mesh_probe.balanced_orders(n)
            self.assertEqual(len(orders), n)
            for position in range(n):
                self.assertEqual(sorted(order[position] for order in orders), list(range(n)))
            follows = {(order[i], order[i + 1]) for order in orders for i in range(n - 1)}
            self.assertEqual(len(follows), n * (n - 1))

    def test_each_session_runs_every_order_each_two_rounds_an_order_and_its_mirror(self):
        for n in (4, 6):
            orders = mesh_probe.balanced_orders(n)
            for session in range(4):
                with self.subTest(n=n, session=session):
                    rows = [mesh_probe.session_order(session, k, n) for k in range(n)]
                    self.assertEqual(sorted(rows), list(range(n)))
                    for k in range(0, n, 2):
                        self.assertEqual(orders[rows[k]][::-1], orders[rows[k + 1]])


class InstrumentTests(unittest.TestCase):
    def test_a_page_without_the_adapter_anchors_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            page = Path(folder) / "page.html"
            page.write_text("<script>(async () => {\n  function render(pos, f) {\n  }\n})();</script>", encoding="utf-8")
            with self.assertRaises(ValueError):
                mesh_probe.instrument(page, Path(folder) / "probe.html")


class SummaryTests(unittest.TestCase):
    @staticmethod
    def blocks(ratio, rounds=16, sessions=4, noise=lambda r: 1.0, second=(1.0, 1.0), control=1.0):
        """One window's blocks: in each round the candidate's configurations take `ratio` times the baseline's 10 ms,
        times that round's noise, and the control `control` times the integrated scene; whichever page a session opened
        second runs at its own factor (`second`: the baseline's and the candidate's), slower on the AMD iGPU."""
        out = []
        for r in range(rounds):
            session = r % sessions
            first = mesh_probe.LOAD_ORDERS[session % len(mesh_probe.LOAD_ORDERS)][0]
            for config, (page, _, _) in mesh_probe.CONFIGS.items():
                ms = 10.0 * (ratio * noise(r) if page == "candidate" else 1.0) * (control if config == "control" else 1.0)
                ms *= 1.0 if page == first else second[page == "candidate"]
                out.append({"window": "castle", "round": r, "session": session, "config": config, "frameMs": ms})
        return out

    def test_without_an_order_effect_the_bounds_meet_and_a_window_is_judged_against_the_budget(self):
        def wobble(r):                                  # half of each session's rounds 5% low, half 5% high
            return (0.95, 1.05)[r // 4 % 2]
        summary = mesh_probe.summarize(self.blocks(1.05))["castle"]
        judged = summary["integrated/baseline"]
        for value in (judged["estimate"], *judged["band"], judged["upper"]["estimate"], judged["lower"]["estimate"]):
            self.assertAlmostEqual(value, 1.05)
        self.assertAlmostEqual(summary["integrated/landscape"]["estimate"], 1.0)
        self.assertAlmostEqual(summary["orderEffect"]["estimate"], 1.0)
        for ratio, noise, verdict in ((1.05, lambda r: 1.0, "pass"), (1.15, lambda r: 1.0, "fail"),
                                      (mesh_probe.BUDGET, wobble, "inconclusive")):
            with self.subTest(ratio=ratio, verdict=verdict):
                summary = mesh_probe.summarize(self.blocks(ratio, noise=noise))["castle"]
                self.assertEqual((summary["flags"], summary["verdict"]), ([], verdict))

    def test_the_page_opened_second_bounds_the_ratio_however_unequally_the_pages_are_slowed(self):
        # 1.12, over the budget, averages to 1.072 over the two load orders when the baseline loses 20% opened second
        # and the candidate 10%; the sessions that open the candidate second still show it over the budget
        summary = mesh_probe.summarize(self.blocks(1.12, second=(1.2, 1.1)))["castle"]
        judged = summary["integrated/baseline"]
        self.assertAlmostEqual(judged["estimate"], 1.12 * math.sqrt(1.1 / 1.2))
        self.assertAlmostEqual(judged["upper"]["estimate"], 1.12 * 1.1)
        self.assertAlmostEqual(judged["lower"]["estimate"], 1.12 / 1.2)
        self.assertAlmostEqual(summary["orderEffect"]["estimate"], math.sqrt(1.2 * 1.1))
        self.assertEqual((summary["flags"], summary["verdict"]), ([], "inconclusive"))
        for seed in range(20):                         # whatever the noise, never a pass
            shake = random.Random(seed)
            noise = [math.exp(shake.gauss(0.0, 0.05)) for _ in range(16)]
            with self.subTest(seed=seed):
                summary = mesh_probe.summarize(self.blocks(1.12, noise=noise.__getitem__, second=(1.2, 1.1)))["castle"]
                self.assertNotEqual(summary["verdict"], "pass")

    def test_a_slowdown_both_pages_share_still_leaves_the_bounds_apart(self):
        # it cancels in the estimate, but the probe cannot tell a shared slowdown from unequal ones
        summary = mesh_probe.summarize(self.blocks(1.05, second=(1.3, 1.3)))["castle"]
        self.assertAlmostEqual(summary["integrated/baseline"]["estimate"], 1.05)
        self.assertAlmostEqual(summary["orderEffect"]["estimate"], 1.3)
        self.assertEqual(summary["verdict"], "inconclusive")

    def test_a_small_order_effect_still_lets_a_window_pass_or_fail(self):
        for ratio, verdict in ((1.05, "pass"), (1.2, "fail")):
            with self.subTest(ratio=ratio):
                summary = mesh_probe.summarize(self.blocks(ratio, second=(1.01, 1.02)))["castle"]
                self.assertEqual(summary["verdict"], verdict)

    def test_pages_that_run_faster_opened_second_cross_the_bounds_without_passing_a_window(self):
        # on NVIDIA the later copy ran up to 1.4% faster: the sessions that open the candidate second then show it
        # under the budget though it is over
        summary = mesh_probe.summarize(self.blocks(1.115, second=(0.986, 0.986)))["castle"]
        judged = summary["integrated/baseline"]
        self.assertLess(judged["upper"]["estimate"], mesh_probe.BUDGET)
        self.assertGreater(judged["lower"]["estimate"], mesh_probe.BUDGET)
        self.assertEqual(summary["verdict"], "inconclusive")

    def test_the_interval_holds_the_variation_between_browser_launches(self):
        launch = (0.95, 1.05, 1.05, 0.95)              # each session's own offset, the same in all its rounds
        judged = mesh_probe.summarize(self.blocks(1.05, noise=lambda r: launch[r % 4]))["castle"]["integrated/baseline"]
        for bounded in (judged, judged["upper"], judged["lower"]):
            low, high = bounded["band"]
            self.assertLess(low, 1.05)
            self.assertGreater(high, 1.05)

    def test_each_draw_takes_all_but_one_of_a_load_orders_sessions_and_all_their_rounds(self):
        session = [{"baseline": 10.0}] * 3
        groups = [[session, session], [session, session]]
        drawn = mesh_probe.interval(groups, lambda g: sum(len(s) for o in g for s in o), random.Random(0))
        self.assertEqual(drawn, (6, 6))                 # one session of two per load order, three rounds each

    def test_a_control_that_does_not_come_out_even_leaves_a_window_inconclusive(self):
        summary = mesh_probe.summarize(self.blocks(1.05, control=1.05))["castle"]
        self.assertEqual((summary["flags"], summary["verdict"]), (["control"], "inconclusive"))

    def test_a_run_without_two_sessions_per_load_order_each_running_every_order_is_inconclusive(self):
        for rounds, sessions in ((8, 2), (8, 4)):
            with self.subTest(rounds=rounds, sessions=sessions):
                summary = mesh_probe.summarize(self.blocks(1.05, rounds, sessions))["castle"]
                self.assertAlmostEqual(summary["integrated/baseline"]["estimate"], 1.05)
                self.assertIsNone(summary["integrated/baseline"]["band"])
                self.assertIsNone(summary["integrated/baseline"]["upper"]["band"])
                self.assertEqual((summary["flags"], summary["verdict"]), (["design"], "inconclusive"))


class FreshSummaryTests(unittest.TestCase):
    """The fresh layout's A/A: rounds that visit the candidate twice, each visit in a fresh browser, check for a gross
    error in fresh visits themselves, which nothing else would catch: a run counts unless the two visits come out more
    than TOLERANCE apart throughout the A/A's interval."""

    @staticmethod
    def aa_blocks(factor, sessions=2, rounds=4, windows=("castle",), noise=lambda r: 1.0):
        """A/A blocks: each round's second visit takes `factor` times the first's 10 ms, times that round's noise."""
        out = []
        for s in range(sessions):
            for k in range(rounds):
                r = 100 + s * rounds + k
                for v in range(2):
                    for window in windows:
                        out.append({"window": window, "round": r, "session": 100 + s, "kind": "aa", "visit": v,
                                    "config": "integrated", "frameMs": 10.0 * (factor * noise(r) if v else 1.0)})
        return out

    def test_aa_rounds_stay_out_of_the_ratios_and_a_second_visit_off_by_more_than_the_tolerance_flags_the_run(self):
        pairs = SummaryTests.blocks(1.05)
        for factor, flags, verdict in ((1.0, [], "pass"), (1.05, ["absolute"], "inconclusive"),
                                       (1 / 1.05, ["absolute"], "inconclusive")):
            with self.subTest(factor=factor):
                summary = mesh_probe.summarize(pairs + self.aa_blocks(factor))["castle"]
                self.assertAlmostEqual(summary["integrated/baseline"]["estimate"], 1.05)
                self.assertAlmostEqual(summary["absolute"]["estimate"], factor)
                self.assertEqual((summary["flags"], summary["verdict"]), (flags, verdict))

    def test_an_aa_interval_reaching_within_the_tolerance_raises_no_flag(self):
        # half the A/A rounds 5% fast and half 5% slow about 1, or 2% either way about 1.03: too noisy to show the
        # visits within the tolerance, or to show them beyond it
        for factor, spread in ((1.0, 0.05), (1.03, 0.02)):
            with self.subTest(factor=factor):
                def noise(r, spread=spread):
                    return (1 - spread, 1 + spread)[r % 2]
                aa = self.aa_blocks(factor, noise=noise)
                summary = mesh_probe.summarize(SummaryTests.blocks(1.05) + aa)["castle"]
                low, high = summary["absolute"]["band"]
                self.assertFalse(1 / mesh_probe.TOLERANCE <= low and high <= mesh_probe.TOLERANCE)
                self.assertTrue(low <= mesh_probe.TOLERANCE and 1 / mesh_probe.TOLERANCE <= high)
                self.assertEqual((summary["flags"], summary["verdict"]), ([], "pass"))

    def test_the_aa_is_pooled_over_the_windows(self):
        blocks = SummaryTests.blocks(1.05) + [{**b, "window": "trees-65"} for b in SummaryTests.blocks(1.05)]
        aa = self.aa_blocks(1.0, windows=("castle",)) + self.aa_blocks(1.04, windows=("trees-65",))
        summary = mesh_probe.summarize(blocks + aa, fresh=True)
        for window in ("castle", "trees-65"):
            self.assertAlmostEqual(summary[window]["absolute"]["estimate"], math.sqrt(1.04))

    def test_a_fresh_run_with_fewer_than_two_aa_sessions_is_inconclusive(self):
        for aa in (self.aa_blocks(1.0, sessions=1), []):
            with self.subTest(aa_sessions=len({b["session"] for b in aa})):
                summary = mesh_probe.summarize(SummaryTests.blocks(1.05) + aa, fresh=True)["castle"]
                self.assertIsNone(summary["absolute"]["band"])
                self.assertEqual((summary["flags"], summary["verdict"]), (["design"], "inconclusive"))

    def test_a_fresh_plan_needs_aa_sessions_and_no_more_than_its_pair_sessions(self):
        self.assertEqual(mesh_probe.fresh_plan(4, 2), [("pair", 0), ("pair", 1), ("aa", None)] * 2)
        for aa in (0, 5):
            with self.subTest(aa=aa), self.assertRaises(ValueError):
                mesh_probe.fresh_plan(4, aa)

    def test_the_two_page_layout_has_no_aa(self):
        summary = mesh_probe.summarize(SummaryTests.blocks(1.05))["castle"]
        self.assertNotIn("absolute", summary)
        self.assertEqual(summary["verdict"], "pass")


class FreshLayoutTests(unittest.IsolatedAsyncioTestCase):
    """The fresh layout opens one browser at a time, a fresh one for every visit: on the AMD iGPU a second live page runs
    slower, and a page switched to in a browser that held another runs slower than a freshly opened one."""

    @staticmethod
    def fake_visits(second=1.0):
        """A visit factory recording each visit's role, how many browsers were alive at once and every timed block. The
        candidate's frame time is `second` times as long on the second visit of a round (every round visits twice)."""
        state = {"alive": 0, "most": 0, "opened": [], "calls": []}

        @contextlib.asynccontextmanager
        async def visit(role):
            state["alive"] += 1
            state["most"] = max(state["most"], state["alive"])
            state["opened"].append(role)
            index = len(state["opened"]) - 1

            async def time_block(settings, frames):
                state["calls"].append((index, frames[0]["t"], settings))
                ms = 10.0 * (1.05 if role == "candidate" and settings["meshes"] else 1.0)
                ms *= second if role == "candidate" and index % 2 else 1.0
                return {"batches": [ms * mesh_probe.IN_FLIGHT] * ((mesh_probe.PREROLL + mesh_probe.COUNT)
                                                                   // mesh_probe.IN_FLIGHT),
                        "drainMs": 0.0, "retries": 0,
                        "snapshot": {"output": [mesh_probe.VIEWPORT["width"], mesh_probe.VIEWPORT["height"]]}}
            try:
                yield time_block, {"page": role, "seconds": 1.0, "adapter": "fake", "compileMs": 0.0}
            finally:
                state["alive"] -= 1
        return visit, state

    async def fresh_run(self, rounds=8, sessions=2, aa_sessions=2, windows=("castle", "trees-65"), split=False,
                        second=1.0):
        visit, state = self.fake_visits(second)
        idle = lambda: (0.0, 0.0, 0.0, {})                                         # noqa: E731
        with contextlib.redirect_stdout(io.StringIO()):
            blocks, other, loads = await mesh_probe.run_fresh(list(windows), rounds, sessions, aa_sessions, visit,
                                                              sample=idle, split=split)
        return blocks, other, loads, state

    async def test_each_visit_runs_alone_in_a_fresh_browser(self):
        blocks, _, loads, state = await self.fresh_run()
        self.assertEqual(state["most"], 1)
        self.assertEqual(len(state["opened"]), 8 * 2 + 2 * 4 * 2)           # pair rounds' visits, then A/A rounds'
        self.assertEqual(len(loads), len(state["opened"]))

    async def test_pair_sessions_alternate_the_load_orders_and_aa_sessions_visit_the_candidate_twice(self):
        _, _, _, state = await self.fresh_run()
        pair = [list(mesh_probe.LOAD_ORDERS[o]) * 4 for o in range(2)]
        self.assertEqual(state["opened"], pair[0] + ["candidate"] * 8 + pair[1] + ["candidate"] * 8)

    async def test_each_window_warms_up_first_with_its_first_timed_configurations_settings(self):
        blocks, _, loads, state = await self.fresh_run()
        windows, pair_rounds, aa_visits = 2, 8, 16
        self.assertEqual(len(blocks), windows * (pair_rounds * (1 + 3) + aa_visits * 1))   # baseline 1, candidate 3
        starts = {mesh_probe.schedule(mesh_probe.WINDOWS[w], w in mesh_probe.PANNED)[0]["t"]: w
                  for w in ("castle", "trees-65")}
        calls = {}
        for index, t, settings in state["calls"]:
            calls.setdefault((index, starts[t]), []).append(settings)
        for (index, window), made in calls.items():
            load = loads[index]
            timed = sorted((b for b in blocks if (b["session"], b["round"], b["visit"], b["window"])
                            == (load["session"], load["round"], load["visit"], window)), key=lambda b: b["position"])
            expected = [{**mesh_probe.SETTINGS, "dbg": mesh_probe.CONFIGS[b["config"]][2],
                         "meshes": mesh_probe.CONFIGS[b["config"]][1]} for b in timed]
            self.assertEqual(made, expected[:1] + expected, (index, window))

    async def test_sessions_are_grouped_by_their_load_order_not_their_place_in_the_plan(self):
        # the A/A sessions shift the pair sessions' indices (pair, pair, A/A, pair, pair, A/A), so a session's parity no
        # longer gives its load order; here the candidate runs 10% slower as a round's second visit
        blocks, other, _, _ = await self.fresh_run(rounds=16, sessions=4, second=1.1)
        judged = mesh_probe.summarize(blocks, other, fresh=True)["castle"]["integrated/baseline"]
        self.assertAlmostEqual(judged["upper"]["estimate"], 1.05 * 1.1)
        self.assertAlmostEqual(judged["lower"]["estimate"], 1.05)

    async def test_split_visits_time_one_window_each_so_a_windows_two_visits_follow_each_other(self):
        windows = ("castle", "trees-65")
        blocks, _, loads, state = await self.fresh_run(split=True, windows=windows)
        starts = {mesh_probe.schedule(mesh_probe.WINDOWS[w], w in mesh_probe.PANNED)[0]["t"]: w for w in windows}
        timed = {}
        for index, t, _ in state["calls"]:
            timed.setdefault(index, set()).add(starts[t])
        self.assertEqual(len(state["opened"]), 2 * len(windows) * (8 + 8))     # pair and A/A rounds, a visit per window
        self.assertTrue(all(len(w) == 1 for w in timed.values()))
        sequence = [next(iter(timed[i])) for i in range(len(state["opened"]))]
        self.assertEqual(sequence[:8], ["castle", "castle", "trees-65", "trees-65"] * 2)
        self.assertEqual([load["windows"] for load in loads[:2]], [["castle"], ["castle"]])
        self.assertEqual(len(blocks), len(windows) * (8 * (1 + 3) + 16 * 1))

    async def test_blocks_carry_their_session_order_kind_and_visit_and_summarize(self):
        blocks, other, _, _ = await self.fresh_run(rounds=16, sessions=4)      # two sessions per load order
        for b in blocks:
            if b["kind"] == "pair":
                self.assertEqual(mesh_probe.LOAD_ORDERS[b["order"]][b["visit"]],
                                 mesh_probe.CONFIGS[b["config"]][0])
            else:
                self.assertEqual((b["config"], b["order"]), ("integrated", None))
        self.assertEqual(len({(b["session"], b["round"]) for b in blocks}), 16 + 8)     # pair rounds, A/A rounds
        summary = mesh_probe.summarize(blocks, other, fresh=True)
        for window in ("castle", "trees-65"):
            self.assertAlmostEqual(summary[window]["integrated/baseline"]["estimate"], 1.05)
            self.assertAlmostEqual(summary[window]["absolute"]["estimate"], 1.0)
            self.assertEqual((summary[window]["flags"], summary[window]["verdict"]), ([], "pass"))


class ProbePageTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocks_are_completed_renders_at_the_requested_settings_with_the_loop_stopped(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "probe.html"
            mesh_probe.instrument(ROOT / "hogwarts-flyover.html", target)
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(channel="msedge", headless=True, chromium_sandbox=True,
                                                           args=mesh_probe.GPU_ARGS.get(GPU, []))
                context = await browser.new_context(viewport={"width": 640, "height": 360}, offline=True)
                page = await context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                await mesh_probe.open_page(page, target, meshes=True)
                loop = await page.evaluate("__hog.frames")
                frames = mesh_probe.schedule(60.5, pan=True)[:5]
                # full internal resolution: frames heavy enough that timing only their submission would show
                settings = {"q": 2, "detail": 0, "scale": 1.0, "tod": 12.0, "dbg": 2, "meshes": True}
                block = await mesh_probe.block(page, settings, frames)
                await page.wait_for_timeout(300)
                self.assertEqual(await page.evaluate("__hog.frames"), loop)      # the animation loop stays stopped
                self.assertEqual(block["rendered"], 5)                           # each frame a real render() call
                self.assertEqual(len(block["batches"]), 3)                       # two in flight, then the last one
                self.assertTrue(all(ms > 0 for ms in block["batches"]))
                shot = block["snapshot"]
                self.assertEqual({k: shot[k] for k in settings}, settings)       # read back from the shaders' uniforms
                self.assertEqual(shot["internal"], [640, 360])
                self.assertLess(block["drainMs"], min(block["batches"]))       # the GPU was idle when the block ended
                self.assertAlmostEqual(shot["t"], frames[-1]["t"], places=3)
                self.assertGreater(shot["drawn"], 0)
                await browser.close()
                self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
