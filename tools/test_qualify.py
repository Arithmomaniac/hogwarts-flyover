"""tools.timing.qualify: the unattended qualification measures NVIDIA first, then trials AMD's fresh-window layout and
runs AMD on the layout the trial supports, in one hold of the quiet lock; it reruns a step only when other processes'
load blocked it, never one that measured."""
import contextlib
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from tools import locks
from tools.paths import SRC  # noqa: F401  (puts src/ on the path)
from tools.timing import qualify
import mesh_probe as mp

CONFIGS = ("baseline", "landscape", "integrated", "control")


def fresh_report(windows=qualify.TRIAL_WINDOWS, sessions=4, aa_sessions=2, ratio=0.9, sigma_round=0.0,
                 sigma_session=0.0, slow=None, seed=0, summary=True, induced=lambda session: 0.5):
    """A fresh-window run's report as mesh_probe writes it: baseline frames of 10 ms and the candidate's `ratio` times
    that, scattered in log terms by `sigma_round` between rounds and `sigma_session` between sessions; A/A rounds whose
    two visits differ by that scatter alone; the candidate's visits in round `slow` (session, round) twice as slow; and
    the security and OS processes' cores `induced(session)`, the other processes' 0.5."""
    rng, blocks, loads, other = random.Random(seed), [], [], []
    per = qualify.ROUNDS_PER_SESSION
    for session, (kind, o) in enumerate(mp.fresh_plan(sessions, aa_sessions)):
        offset = rng.gauss(0.0, sigma_session)
        for k in range(per):
            r = session * per + k
            for window in windows:
                roles = ("candidate", "candidate") if kind == "aa" else mp.LOAD_ORDERS[o]
                for v, role in enumerate(roles):
                    candidate = 10.0 * ratio * math.exp(offset + rng.gauss(0.0, sigma_round))
                    candidate *= 2.0 if (session, r) == slow else 1.0
                    configs = ({"integrated": candidate} if kind == "aa" else {"baseline": 10.0} if role == "baseline"
                               else {"landscape": candidate, "integrated": candidate, "control": candidate})
                    blocks += [{"window": window, "round": r, "session": session, "order": o, "kind": kind, "visit": v,
                                "position": position, "config": c, "frameMs": ms}
                               for position, (c, ms) in enumerate(configs.items())]
                    loads.append({"session": session, "round": r, "visit": v, "windows": [window], "page": role})
                    other.append({"window": window, "session": session, "attempt": 0, "otherCores": 0.5,
                                  "inducedCores": induced(session), "retaken": False})
    return {"layout": "fresh-window", "blocks": blocks, "loads": loads, "otherLoad": {"windows": other},
            "summary": mp.summarize(blocks, other, fresh=True) if summary else None}


def pages_report(windows=("castle", "greenhouses"), ratio=0.9):
    """A two-page run's report at the probe's default design, the candidate's frames `ratio` times the baseline's."""
    blocks = [{"window": w, "round": r, "session": r % mp.SESSIONS, "config": c,
               "frameMs": 10.0 * (1.0 if c == "baseline" else ratio)}
              for w in windows for r in range(mp.ROUNDS) for c in CONFIGS]
    return {"layout": "pages", "blocks": blocks, "loads": [], "otherLoad": {"windows": []},
            "summary": mp.summarize(blocks)}


def measured(name, result, status=0, seconds=600.0):
    return {"name": name, "status": status, "attempts": 1, "seconds": seconds, "log": f"{name}.log", "result": result}


class ScatterTests(unittest.TestCase):
    def test_the_scatter_between_rounds_and_between_sessions_comes_back(self):
        stats = qualify.scatter(fresh_report(sessions=32, sigma_round=0.06, sigma_session=0.03, summary=False))
        self.assertAlmostEqual(stats["round"], 0.06, delta=0.006)
        self.assertAlmostEqual(stats["session"], 0.03, delta=0.012)
        self.assertEqual(stats["roundsPerSession"], qualify.ROUNDS_PER_SESSION)

    def test_one_slow_visit_barely_moves_the_scatter_between_rounds(self):
        calm = qualify.scatter(fresh_report(sessions=8, sigma_round=0.05, summary=False))["round"]
        slowed = qualify.scatter(fresh_report(sessions=8, sigma_round=0.05, slow=(0, 1), summary=False))["round"]
        self.assertAlmostEqual(slowed, calm, delta=0.1 * calm)


class HalfwidthTests(unittest.TestCase):
    def test_a_bounds_interval_narrows_with_sessions_and_widens_with_scatter(self):
        widths = [qualify.halfwidth(0.06, 0.0, n, replicates=12) for n in (2, 4, 8)]
        self.assertGreater(widths[0], widths[1])
        self.assertGreater(widths[1], widths[2])
        self.assertGreater(qualify.halfwidth(0.06, 0.03, 4, replicates=12), widths[1])

    def test_it_projects_the_intervals_the_probe_gives_a_fresh_runs_bounds(self):
        # each window of a run is an independent replicate of its two bounds
        report = fresh_report(windows=tuple(mp.WINDOWS), sigma_round=0.06)
        realised = sorted(math.log(b["band"][1] / b["band"][0]) / 2 for s in report["summary"].values()
                          for b in (s["integrated/baseline"]["upper"], s["integrated/baseline"]["lower"]))
        projected = qualify.halfwidth(0.06, 0.0, qualify.TRIAL["sessions"] // 2)
        self.assertAlmostEqual(realised[len(realised) // 2], projected, delta=0.3 * projected)


class DesignTests(unittest.TestCase):
    STATS = {"round": 0.05, "session": 0.0}

    def design(self, trial_width=0.04, aa_flagged=False, visit_s=14.0, max_hours=qualify.MAX_AMD_HOURS):
        return qualify.amd_design(self.STATS, aa_flagged, visit_s, max_hours, projection=lambda *args: trial_width)

    def test_a_trial_within_the_limit_runs_fresh_window_at_its_design_and_the_focus_window_again(self):
        design = self.design()
        self.assertEqual((design["layout"], [r["name"] for r in design["runs"]]), ("fresh-window", ["fresh", "focus"]))
        fresh, focus = (r["args"] for r in design["runs"])
        self.assertEqual((fresh["rounds"], fresh["sessions"], fresh["aa_sessions"]), (16, 4, 2))      # the floor
        self.assertEqual((fresh["windows"], focus["windows"]), (tuple(mp.WINDOWS), qualify.FOCUS))
        self.assertEqual({k: v for k, v in focus.items() if k != "windows"},
                         {k: v for k, v in fresh.items() if k != "windows"})
        self.assertAlmostEqual(design["projectedS"], (16 + 8) * (len(mp.WINDOWS) + 1) * 2 * 14.0)

    def test_a_noisy_trial_or_a_flagged_aa_sends_amd_to_two_pages(self):
        for kwargs in ({"trial_width": qualify.LIMIT * 1.2}, {"aa_flagged": True}):
            with self.subTest(**kwargs):
                design = self.design(**kwargs)
                self.assertEqual((design["layout"], design["runs"]), ("pages", [{"name": "pages", "args": {}}]))
                self.assertEqual(len(design["reasons"]), 1)

    def test_runs_that_would_outlast_the_cap_drop_the_focus_run_and_then_fresh_window(self):
        first = (16 + 8) * len(mp.WINDOWS) * 2                        # fresh browsers the first run opens
        cap_s = qualify.MAX_AMD_HOURS * 3600
        design = self.design(visit_s=cap_s / (first + 8))              # the first run fits, the focus run not
        self.assertEqual((design["layout"], [r["name"] for r in design["runs"]]), ("fresh-window", ["fresh"]))
        design = self.design(visit_s=cap_s / (first - 8))
        self.assertEqual(design["layout"], "pages")


class StepTests(unittest.TestCase):
    def run_step(self, statuses, writes=None, stale=False):
        """step() over a fake probe that exits with `statuses` in turn and writes its report on each attempt `writes`
        marks (every attempt by default: the probe writes one before it exits 3 too), in a folder that already holds an
        older run's report when `stale`."""
        calls, sleeps, writes = [], [], writes or [True] * len(statuses)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "probe.json"
            if stale:
                out.write_text(json.dumps({"summary": "older"}), encoding="utf-8")

            def runner(*argv, log, env):
                calls.append(env)
                if writes[len(calls) - 1]:
                    out.write_text(json.dumps({"summary": len(calls)}), encoding="utf-8")
                return statuses[len(calls) - 1]
            result = qualify.step("x", ["src/mesh_probe.py", "--out", out], "token", Path(folder), runner=runner,
                                  sleep=sleeps.append)
        return result, calls, sleeps

    def test_a_blocked_step_runs_again_after_a_pause_and_a_step_that_measured_never_does(self):
        result, calls, sleeps = self.run_step([3, 3, 1])
        self.assertEqual((result["status"], result["attempts"], result["result"]), (1, 3, {"summary": 3}))
        self.assertEqual((sleeps, calls), ([qualify.BLOCKED_WAIT_S] * 2, [{locks.TOKEN: "token"}] * 3))
        result, calls, sleeps = self.run_step([1, 0, 0])
        self.assertEqual((result["attempts"], len(calls), sleeps), (1, 1, []))

    def test_a_step_blocked_every_time_or_that_wrote_no_report_of_its_own_has_none(self):
        result, _, _ = self.run_step([3, 3, 3])
        self.assertEqual((result["status"], result["attempts"], result["result"]), (3, 3, None))
        # a crash exits 1 too, without a report: neither an older run's report nor a blocked attempt's is its own
        for statuses, writes, stale in (([1], [False], True), ([3, 1], [True, False], False)):
            with self.subTest(statuses=statuses, stale=stale):
                result, _, _ = self.run_step(statuses, writes, stale)
                self.assertEqual((result["status"], result["result"]), (1, None))


class QualifyTests(unittest.TestCase):
    def qualify(self, reports, projection=lambda *args: 0.04, statuses=None):
        events = []

        @contextlib.contextmanager
        def quiet(label, say):
            events.append("lock")
            yield "token"
            events.append("unlock")

        def run(name, argv, token, folder):
            events.append((name, tuple(map(str, argv[argv.index("--out") + 2:]))))
            self.assertEqual(token, "token")
            return measured(name, reports.get(name), status=(statuses or {}).get(name, 0))
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(qualify, "git_head", return_value="head"), \
                mock.patch.object(qualify.inputs, "spec", return_value={"sha256": "baseline"}):
            summary = qualify.qualify("baseline.html", Path(folder), qualify.MAX_AMD_HOURS, quiet=quiet, run=run,
                                      sleep=lambda s: events.append(f"sleep {s}"), projection=projection,
                                      say=lambda message: None)
        return summary, events

    def test_nvidia_runs_first_then_the_amd_trial_then_amd_all_in_one_hold_of_the_quiet_lock(self):
        reports = {"nvidia": pages_report(), "trial-amd": fresh_report(sigma_round=0.02),
                   "amd-fresh": fresh_report(windows=("castle", "greenhouses")),
                   "amd-focus": fresh_report(windows=qualify.FOCUS)}
        summary, events = self.qualify(reports)
        trial = ("--layout", "fresh-window", "--windows", ",".join(qualify.TRIAL_WINDOWS), "--rounds", "16",
                 "--sessions", "4", "--aa-sessions", "2")
        self.assertEqual(events[:4], ["lock", ("nvidia", ()), f"sleep {qualify.COOL_S}", ("trial-amd", trial)])
        self.assertEqual([e[0] for e in events[4:6]], ["amd-fresh", "amd-focus"])
        self.assertEqual(events[4][1][3], ",".join(mp.WINDOWS))
        self.assertEqual(events[5][1][3], ",".join(qualify.FOCUS))
        self.assertEqual(events[-1], "unlock")
        self.assertEqual((summary["design"]["layout"], summary["outcome"]), ("fresh-window", "pass"))
        self.assertAlmostEqual(summary["trial"]["visitS"], 600.0 / len(reports["trial-amd"]["loads"]))

    def test_a_noisy_trial_runs_amd_on_two_pages(self):
        reports = {"nvidia": pages_report(), "trial-amd": fresh_report(), "amd-pages": pages_report()}
        summary, events = self.qualify(reports, projection=lambda *args: qualify.LIMIT * 2)
        self.assertEqual(events[4], ("amd-pages", ()))
        self.assertEqual(summary["design"]["layout"], "pages")

    def test_a_trial_that_measured_nothing_runs_amd_on_two_pages_and_leaves_the_outcome_to_the_verdicts(self):
        for status in (qualify.BLOCKED, 1):
            with self.subTest(status=status):
                reports = {"nvidia": pages_report(), "amd-pages": pages_report()}
                summary, events = self.qualify(reports, statuses={qualify.TRIAL_STEP: status})
                self.assertEqual(events[4], ("amd-pages", ()))
                self.assertEqual((summary["design"]["layout"], summary["trial"]), ("pages", None))
                self.assertEqual(summary["outcome"], "pass")
                self.assertIn("trial measured nothing", qualify.report(summary))

    def test_the_focus_window_is_judged_on_both_runs_rounds_together(self):
        first, focus = fresh_report(windows=("castle", "greenhouses")), fresh_report(windows=qualify.FOCUS, seed=1)
        judged = qualify.verdicts([measured("amd-fresh", first), measured("amd-focus", focus)])
        self.assertEqual(judged["castle"], first["summary"]["castle"])
        sessions = judged["greenhouses"]["integrated/baseline"]["sessions"]
        self.assertEqual(len(sessions), 2 * len(first["summary"]["greenhouses"]["integrated/baseline"]["sessions"]))

    def test_the_outcome_is_blocked_then_an_error_then_each_windows_verdict(self):
        passed = {"amd": {"castle": {"verdict": "pass"}}}
        self.assertEqual(qualify.outcome([measured("a", None, status=3)], passed), "blocked")
        self.assertEqual(qualify.outcome([measured("a", None, status=1)], passed), "error")
        self.assertEqual(qualify.outcome([measured("a", {})], passed), "pass")
        self.assertEqual(qualify.outcome([measured("a", {})], {"amd": {"castle": {"verdict": "inconclusive"}}}),
                         "fail")


class ReportTests(unittest.TestCase):
    def test_each_judged_windows_line_logs_its_induced_and_other_load(self):
        first = fresh_report(windows=("castle", "greenhouses"), induced=lambda session: 0.7 + 0.1 * session)
        focus = fresh_report(windows=qualify.FOCUS, seed=1, induced=lambda session: 2.4)
        summary = {"page": "page", "head": "head", "outcome": "pass", "steps": [], "trial": None, "design": None,
                   "verdicts": {"amd": qualify.verdicts([measured("amd-fresh", first), measured("amd-focus", focus)])}}
        lines = {window: next(line for line in qualify.report(summary).splitlines() if f" {window} " in line)
                 for window in ("castle", "greenhouses")}
        self.assertIn("induced 0.7-1.2 cores, other 0.5-0.5", lines["castle"])
        self.assertIn("induced 0.7-2.4 cores", lines["greenhouses"])          # the focus window's, over both runs


if __name__ == "__main__":
    unittest.main()
