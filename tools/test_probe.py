"""src/mesh_probe.py's load accounting. Two absolute caps hold, since the frozen baseline runs interleaved with the
candidate: one on other processes' busy cores, and a looser one on security and OS processes (Defender, Global Secure
Access, System, svchost), whose work follows the probe's own browser. The probe waits for the machine to settle under
both before it starts, measures a window again while either is exceeded, and reports a run blocked, never passed, if
the machine never settles or keeps a window over a cap through every attempt."""
import asyncio
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import psutil

from tools.paths import SRC  # noqa: F401  (puts src/ on the path)
import mesh_probe

CAPS = (2.0, 4.0)
DEFENDER = 4242                          # a pid standing for an induced process


def sampler(*loads):
    """A cpu_seconds() stand-in for successive 10 s measurements over which other processes keep `loads` cores busy
    (a number, or (other, induced) cores), while the probe's own processes use 2 cores."""
    samples, t, busy, mine, induced = [], 0.0, 0.0, 0.0, 0.0
    for load in loads:
        other, more = load if isinstance(load, tuple) else (load, 0.0)
        samples.append((t, busy, mine, {DEFENDER: induced}))
        t, busy, mine, induced = t + 10.0, busy + 10.0 * (other + more) + 20.0, mine + 20.0, induced + 10.0 * more
        samples.append((t, busy, mine, {DEFENDER: induced}))
    samples = iter(samples)
    return lambda: next(samples)


def probe(*loads, caps=CAPS):
    """mesh_probe.measured on one window whose successive attempts see `loads`; the blocks it keeps and its load
    records."""
    attempts = []

    async def measure():
        attempts.append(len(attempts))
        return [f"attempt {len(attempts)}"]
    return asyncio.run(mesh_probe.measured("castle", 0, measure, caps, sample=sampler(*loads)))


def result(*cores, caps=CAPS, ms=None):
    """A run's result whose castle window kept, in each session, an attempt that saw `cores` ((other, induced) each), and
    whose rounds took `ms` ({configuration: frame ms}; 10 ms each by default, which passes), summarized as the probe
    does."""
    blocks = [{"window": "castle", "round": r, "session": r % mesh_probe.SESSIONS, "config": c,
               "frameMs": (ms or {}).get(c, 10.0)} for r in range(mesh_probe.ROUNDS) for c in mesh_probe.CONFIGS]
    windows = [{"window": "castle", "session": session, "attempt": 0, "otherCores": other, "inducedCores": induced,
                "retaken": False, **({"contended": True} if other > caps[0] or induced > caps[1] else {})}
               for session, (other, induced) in enumerate(cores)]
    return {"summary": mesh_probe.summarize(blocks, windows),
            "otherLoad": {"capCores": caps[0], "inducedCapCores": caps[1], "windows": windows}}


class LoadTests(unittest.TestCase):
    def test_a_window_measured_under_other_load_is_measured_again(self):
        blocks, loads = probe(3.0, 0.5)
        self.assertEqual(blocks, ["attempt 2"])
        self.assertEqual([(round(x["otherCores"], 2), x["retaken"]) for x in loads], [(3.0, True), (0.5, False)])
        self.assertNotIn("contended", loads[-1])

    def test_a_steady_load_over_the_cap_contends_every_attempt(self):
        # a margin over the load when the run started would pass this, if the run started loaded
        blocks, loads = probe(*[2.5] * mesh_probe.LOAD_ATTEMPTS)
        self.assertEqual(blocks, [f"attempt {mesh_probe.LOAD_ATTEMPTS}"])
        self.assertEqual([x["retaken"] for x in loads], [True] * (mesh_probe.LOAD_ATTEMPTS - 1) + [False])
        self.assertTrue(loads[-1]["contended"])

    def test_induced_load_within_its_own_cap_does_not_count_against_the_other_cap(self):
        blocks, loads = probe((1.0, 3.5))
        self.assertEqual(blocks, ["attempt 1"])
        self.assertEqual([(round(x["otherCores"], 2), round(x["inducedCores"], 2)) for x in loads], [(1.0, 3.5)])
        self.assertNotIn("contended", loads[-1])

    def test_a_window_over_the_induced_cap_is_measured_again_and_contends_if_it_stays_over(self):
        blocks, loads = probe((0.5, 5.0), (0.5, 1.0))
        self.assertEqual((blocks, [x["retaken"] for x in loads]), (["attempt 2"], [True, False]))
        blocks, loads = probe(*[(0.5, 5.0)] * mesh_probe.LOAD_ATTEMPTS)    # a Defender scan still contends
        self.assertTrue(loads[-1]["contended"])

    def test_the_probes_own_work_does_not_count_as_other_load(self):
        self.assertEqual(mesh_probe.loads((0.0, 100.0, 50.0, {}), (10.0, 130.0, 75.0, {})), (0.5, 0.0))

    def test_induced_processes_count_apart_and_one_that_ends_counts_nothing(self):
        before, after = (0.0, 100.0, 50.0, {4: 10.0, 7: 5.0}), (10.0, 160.0, 75.0, {4: 20.0, 9: 99.0})
        other, induced = mesh_probe.loads(before, after)
        self.assertAlmostEqual(induced, 1.0)                        # pid 7 ended and pid 9 began: neither counts
        self.assertAlmostEqual(other, 2.5)

    def test_one_snapshot_finds_every_process_as_psutil_does(self):
        procs, me = mesh_probe.process_times(), psutil.Process()
        parent, _, name, cpu = procs[os.getpid()]
        self.assertEqual((parent, name), (me.ppid(), me.name().lower()))
        self.assertAlmostEqual(cpu, sum(me.cpu_times()[:2]), delta=0.5)
        self.assertEqual(procs[4][2], "system")
        self.assertIn("system", mesh_probe.INDUCED)

    def test_the_kernels_own_processes_count_as_induced_by_the_names_the_snapshot_reads(self):
        # the kernel names some of them apart from psutil and Task Manager: "Memory Compression", not "MemCompression"
        procs = mesh_probe.process_times()
        kernel = {p.pid: p.info["name"] for p in psutil.process_iter(["name"])
                  if p.info["name"] in ("System", "Registry", "MemCompression")}
        self.assertIn(4, kernel)
        for pid, name in kernel.items():
            self.assertIn(procs[pid][2], mesh_probe.INDUCED, name)

    def test_security_and_os_processes_count_as_induced(self):
        for name in ("MsMpEng.exe", "MsSense.exe", "GlobalSecureAccessClient.exe", "System", "svchost.exe"):
            self.assertIn(name.lower(), mesh_probe.INDUCED)
        for name in ("dotnet.exe", "python.exe", "msedge.exe", "copilot.exe"):
            self.assertNotIn(name.lower(), mesh_probe.INDUCED)

    def test_the_browser_gives_the_security_processes_little_to_scan(self):
        # every launch reuses one profile, with no caches written and no background traffic to inspect
        options = mesh_probe.launch_options("amd", Path("profile"))
        self.assertEqual(options["user_data_dir"], Path("profile"))
        for flag in ("--disable-gpu-shader-disk-cache", "--disk-cache-size=1", "--disable-background-networking",
                     *mesh_probe.GPU_ARGS["amd"]):
            self.assertIn(flag, options["args"])
        self.assertTrue(options["offline"])

    def test_a_run_waits_for_the_machine_to_settle_under_both_caps_before_it_starts(self):
        load, settled = asyncio.run(mesh_probe.starting_load(CAPS, attempts=3, seconds=0,
                                                             sample=sampler(3.0, (1.0, 5.0), 1.5)))
        self.assertEqual(([round(c, 2) for c in load], settled), ([1.5, 0.0], True))

    def test_a_run_whose_machine_stays_busy_does_not_start(self):
        load, settled = asyncio.run(mesh_probe.starting_load(CAPS, attempts=2, seconds=0,
                                                             sample=sampler(3.0, (0.5, 4.5))))
        self.assertEqual(([round(c, 2) for c in load], settled), ([0.5, 4.5], False))


class ExitTests(unittest.TestCase):
    """The stage runner reads exit status 3 as blocked: never a pass, and never reused. Status 1 is a fail: a window over
    the budget, or one the run cannot judge."""

    def main(self, outcome, *extra):
        """mesh_probe.main with its run replaced by `outcome`, a result or an exception to raise, and `extra` arguments;
        its exit status and whether it wrote its report."""
        async def run(*args, **kwargs):
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "probe.json"
            argv = ["mesh_probe.py", "--gpu", "amd", "--baseline", "baseline.html", "--out", str(out), *extra]
            with mock.patch.object(mesh_probe, "run", run), mock.patch("tools.locks.require_quiet"), \
                    mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
                return mesh_probe.main(), out.is_file()

    def test_a_run_within_both_caps_passes(self):
        self.assertEqual(self.main(result((0.4, 3.0), (1.9, 0.5))), (0, True))

    def test_a_window_over_either_cap_through_every_attempt_blocks_the_run(self):
        self.assertEqual(self.main(result((0.4, 3.0), (2.6, 0.5))), (3, True))      # its measurements are kept
        self.assertEqual(self.main(result((0.4, 4.5), (1.0, 0.5))), (3, True))

    def test_a_machine_that_never_settles_blocks_the_run(self):
        self.assertEqual(self.main(mesh_probe.Loaded("other processes kept 3.00 cores busy")), (3, False))

    def test_a_window_over_the_budget_or_that_cannot_be_judged_fails_the_run(self):
        over = {"integrated": 11.5, "control": 11.5}
        self.assertEqual(self.main(result((0.4, 3.0), ms=over)), (1, True))
        self.assertEqual(self.main(result((0.4, 3.0), ms={"control": 10.5})), (1, True))     # the control is off

    def run_args(self, *extra):
        """The arguments mesh_probe.main gives run with `extra` arguments, its run replaced by one that passes."""
        calls = []

        async def run(*args, **kwargs):
            calls.append((args, kwargs))
            return result((0.4, 3.0))
        with tempfile.TemporaryDirectory() as folder:
            argv = ["mesh_probe.py", "--gpu", "amd", "--baseline", "baseline.html", "--out", str(Path(folder) / "p.json"),
                    *extra]
            with mock.patch.object(mesh_probe, "run", run), mock.patch("tools.locks.require_quiet"), \
                    mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(mesh_probe.main(), 0)
        return calls[0]

    def test_each_layout_runs_its_own_default_design(self):
        # both GPUs keep both pages open by default; AMD's fresh layouts, a fresh browser per visit, are on trial
        fresh = (mesh_probe.FRESH_ROUNDS, mesh_probe.FRESH_SESSIONS)
        for extra, layout, design in ((("--gpu", "amd"), "pages", (mesh_probe.ROUNDS, mesh_probe.SESSIONS)),
                                      (("--gpu", "nvidia"), "pages", (mesh_probe.ROUNDS, mesh_probe.SESSIONS)),
                                      (("--layout", "fresh"), "fresh", fresh),
                                      (("--layout", "fresh-window"), "fresh-window", fresh)):
            with self.subTest(extra=extra):
                args, kwargs = self.run_args(*extra)
                self.assertEqual((kwargs["layout"], (args[4], args[7])), (layout, design))

    def test_a_fresh_run_needs_two_aa_sessions_and_no_more_than_its_pair_sessions(self):
        self.assertEqual(self.run_args("--layout", "fresh", "--aa-sessions", "8")[1]["aa_sessions"], 8)
        for aa in ("1", "9"):
            with self.subTest(aa=aa), self.assertRaises(SystemExit):
                self.run_args("--layout", "fresh-window", "--aa-sessions", aa)

    def test_a_run_too_thin_to_judge_is_refused_before_it_starts(self):
        for rounds, sessions in ((8, 2), (8, 4), (12, 4)):
            with self.subTest(rounds=rounds, sessions=sessions), self.assertRaises(SystemExit):
                self.main(result((0.4, 3.0)), "--rounds", str(rounds), "--sessions", str(sessions))


if __name__ == "__main__":
    unittest.main()
