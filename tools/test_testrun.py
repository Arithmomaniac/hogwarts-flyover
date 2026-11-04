"""tools.testrun's gate: outcomes per exact test ID, so matching failure counts cannot hide a swap, and a test that could
not run is blocked, never passed. A run stopped part-way resumes at the class it was in."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import uuid

from tools import testrun

PITCH = [f"test_assets_grounds.PitchTests.test_{i}" for i in range(7)]
CASTLE = "test_assets_castle.FacadeTests.test_windows"


def records(**outcomes):
    return {test_id: {"outcome": outcome, "detail": "", "seconds": 0.0} for test_id, outcome in outcomes.items()}


class GateTests(unittest.TestCase):
    def gate(self, outcomes, expected=PITCH):
        found = records(**outcomes)
        return testrun.gate(list(found), found, [], set(expected))

    def test_the_listed_failures_failing_pass_the_gate(self):
        final, problems, verdict = self.gate({**{t: "fail" for t in PITCH[:5]}, **{t: "error" for t in PITCH[5:]}, CASTLE: "pass"})
        self.assertEqual(verdict, "pass")
        self.assertEqual({final[t]["outcome"] for t in PITCH}, {"expected-fail"})
        self.assertEqual(problems, [])

    def test_fixing_a_pitch_test_while_breaking_a_castle_test_fails_though_the_counts_match(self):
        outcomes = {PITCH[0]: "pass", **{t: "fail" for t in PITCH[1:]}, CASTLE: "fail"}
        final, _, verdict = self.gate(outcomes)
        self.assertEqual(sum(o != "pass" for o in outcomes.values()), len(PITCH))      # seven failures, as before
        self.assertEqual(verdict, "fail")
        self.assertEqual(final[PITCH[0]]["outcome"], "fail")                          # listed but passed
        self.assertEqual(final[CASTLE]["outcome"], "fail")

    def test_a_selected_listed_test_that_did_not_run_fails_the_gate(self):
        ids = [*PITCH, CASTLE]
        found = records(**{t: "fail" for t in PITCH[1:]}, **{CASTLE: "pass"})        # PITCH[0] has no record
        final, _, verdict = testrun.gate(ids, found, [], set(PITCH))
        self.assertEqual((final[PITCH[0]]["outcome"], verdict), ("error", "error"))
        self.assertIn("did not run", final[PITCH[0]]["detail"])

    def test_a_listing_with_no_such_test_fails_the_gate(self):
        _, problems, verdict = self.gate({t: "fail" for t in PITCH[1:]})                # PITCH[0] is not in the suite
        self.assertEqual(verdict, "fail")
        self.assertEqual(len(problems), 1)
        self.assertIn("no such test", problems[0])

    def test_filtered_out_listed_tests_are_not_missing_but_a_listed_test_that_does_not_exist_is(self):
        found = records(**{CASTLE: "pass"})
        _, problems, verdict = testrun.gate([CASTLE], found, [], set(PITCH), every={CASTLE, *PITCH})   # -k castle
        self.assertEqual((problems, verdict), ([], "pass"))
        _, problems, verdict = testrun.gate([CASTLE], found, [], {*PITCH, "test_assets_grounds.PitchTests.gone"},
                                            every={CASTLE, *PITCH})
        self.assertEqual(verdict, "fail")
        self.assertIn("PitchTests.gone", problems[0])

    def test_a_listed_test_that_was_skipped_fails_the_gate(self):
        final, _, verdict = self.gate({PITCH[0]: "skipped", **{t: "fail" for t in PITCH[1:]}})
        self.assertEqual((final[PITCH[0]]["outcome"], verdict), ("fail", "fail"))

    def run_classes(self, *classes):
        tests = [cls(name) for cls in classes for name in unittest.TestLoader().getTestCaseNames(cls)]
        result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=testrun.Recorder).run(unittest.TestSuite(tests))
        return [t.id() for t in tests], result

    def test_a_setup_error_in_a_class_of_listed_tests_fails_the_gate(self):
        class Pitch(unittest.TestCase):              # the pitch gone from the payload: setUpClass cannot find it
            @classmethod
            def setUpClass(cls):
                raise AssertionError("no pitch geometry in the payload")

            def test_a(self):
                pass

            def test_b(self):
                pass

        ids, result = self.run_classes(Pitch)
        final, _, verdict = testrun.gate(ids, result.records, result.fixtures, set(ids))
        self.assertEqual(verdict, "error")
        self.assertEqual({final[i]["outcome"] for i in ids}, {"error"})

    def test_a_skipped_subtest_keeps_its_test_from_passing(self):
        class Gpus(unittest.TestCase):
            def test_each_gpu(self):
                for gpu in ("amd", "nvidia"):
                    with self.subTest(gpu=gpu):
                        self.skipTest(f"blocked: no {gpu} adapter")

        ids, result = self.run_classes(Gpus)
        final, _, verdict = testrun.gate(ids, result.records, result.fixtures, set())
        self.assertEqual((final[ids[0]]["outcome"], verdict), ("blocked", "blocked"))

    def test_a_class_skipped_as_blocked_reports_each_test_blocked_and_never_passes(self):
        class NoGpu(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise unittest.SkipTest("blocked: HOG_GPU=amd, but Edge renders on 'NVIDIA'")

            def test_a(self):
                pass

            def test_b(self):
                pass

        tests = [NoGpu("test_a"), NoGpu("test_b")]
        ids = [t.id() for t in tests]
        result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=testrun.Recorder).run(unittest.TestSuite(tests))
        final, _, verdict = testrun.gate(ids, result.records, result.fixtures, set())
        self.assertEqual(verdict, "blocked")
        self.assertEqual({final[i]["outcome"] for i in ids}, {"blocked"})

    def test_a_class_whose_setup_fails_reports_each_test_as_an_error(self):
        class Broken(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise RuntimeError("the page failed to start")

            def test_a(self):
                pass

        ids = [Broken("test_a").id()]
        result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=testrun.Recorder).run(
            unittest.TestSuite([Broken("test_a")]))
        final, _, verdict = testrun.gate(ids, result.records, result.fixtures, set())
        self.assertEqual((final[ids[0]]["outcome"], verdict), ("error", "error"))


class LoadTests(unittest.TestCase):
    def test_a_module_that_raises_on_import_is_an_error_not_a_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "test_raises_on_import.py").write_text("raise ValueError('two families reserve one code')\n",
                                                               encoding="utf-8")
            sys.path.insert(0, folder)
            try:
                with mock.patch.dict(testrun.SUITES, {"broken": {"modules": ("test_raises_on_import",)}}):
                    suite, broken, _ = testrun.load("broken")
            finally:
                sys.path.remove(folder)
                sys.modules.pop("test_raises_on_import", None)
        self.assertEqual(len(broken), 1)
        ids = [t.id() for t in testrun.tests_in(suite)]
        result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=testrun.Recorder).run(suite)
        final, _, verdict = testrun.gate(ids, result.records, result.fixtures, set())
        self.assertEqual(verdict, "error")
        self.assertIn("two families reserve one code", next(iter(final.values()))["detail"])

    def test_an_import_failure_keeps_the_smoke_filter_and_stays_selected(self):
        # beside a module that fails to import, the smoke tag (and -k) still leaves out a real module's untagged tests,
        # and the import failure itself stays selected
        with mock.patch.dict(testrun.SUITES, {"broken": {"modules": ("no_such_test_module", "test_materials"),
                                                         "smoke": True}}):
            suite, broken, everything = testrun.load("broken")
        selected = list(testrun.tests_in(suite))
        self.assertEqual(selected, broken)
        self.assertGreater(len(everything), len(selected))
        with mock.patch.dict(testrun.SUITES, {"broken": {"modules": ("no_such_test_module", "test_materials")}}):
            suite, broken, _ = testrun.load("broken", pattern="nothing_matches_this")
        self.assertEqual(list(testrun.tests_in(suite)), broken)

    def test_a_shift_is_never_inherited_from_the_shell(self):
        with mock.patch.dict(os.environ, {"HOG_DATUM_SHIFT_M": "140", "HOG_GPU": "amd"}):
            testrun.set_environment(None, None)
            self.assertNotIn("HOG_DATUM_SHIFT_M", os.environ)
            self.assertEqual(os.environ["HOG_GPU"], "")
            testrun.set_environment("nvidia", -140.0)
            self.assertEqual((os.environ["HOG_DATUM_SHIFT_M"], os.environ["HOG_GPU"]), ("-140.0", "nvidia"))


# a suite of four classes whose tests note that they ran; B can stop the whole run once, as a pause does, or fail once,
# and D's tearDownClass can fail once
RESUMABLE = '''import unittest
from pathlib import Path

TRACE, ONCE, TEARDOWN = Path({trace!r}), Path({once!r}), Path({teardown!r})


def note(test):
    with TRACE.open("a", encoding="utf-8") as f:
        f.write(test.id().split(".", 1)[1] + "\\n")


class A(unittest.TestCase):
    def test_one(self):
        note(self)

    def test_two(self):
        note(self)


class B(unittest.TestCase):
    def test_one(self):
        note(self)
        if ONCE.exists():
            what = ONCE.read_text(encoding="utf-8")
            ONCE.unlink()
            if what == "stop":
                raise KeyboardInterrupt
            self.fail("B fails once")


class C(unittest.TestCase):
    def test_one(self):
        note(self)


class D(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        if TEARDOWN.exists():
            TEARDOWN.unlink()
            raise RuntimeError("D's teardown fails once")

    def test_one(self):
        note(self)
'''
EVERY = ["A.test_one", "A.test_two", "B.test_one", "C.test_one", "D.test_one"]
# a second module, whose tearDownModule can fail once
TORN_DOWN = '''import unittest
from pathlib import Path

TRACE, MODULE_TEARDOWN = Path({trace!r}), Path({flag!r})


def note(test):
    with TRACE.open("a", encoding="utf-8") as f:
        f.write(test.id().split(".", 1)[1] + "\\n")


def tearDownModule():
    if MODULE_TEARDOWN.exists():
        MODULE_TEARDOWN.unlink()
        raise RuntimeError("the module's teardown fails once")


class E(unittest.TestCase):
    def test_one(self):
        note(self)


class F(unittest.TestCase):
    def test_one(self):
        note(self)
'''


class ResumeTests(unittest.TestCase):
    """A run stopped part-way loses only the class it was in: each class that passes is recorded under the suite's
    input key once it ends, and a run under the same key runs only the classes without a record, its log and report
    still covering every test."""

    def setUp(self):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        self.module, self.tests = f"test_resume_{uuid.uuid4().hex[:8]}", folder / "tests"
        self.tests.mkdir()
        self.trace, self.once, self.teardown = folder / "trace.txt", folder / "once.txt", folder / "teardown.txt"
        self.source = self.tests / f"{self.module}.py"
        self.source.write_text(RESUMABLE.format(trace=str(self.trace), once=str(self.once),
                                                teardown=str(self.teardown)), encoding="utf-8")
        for patch in (mock.patch.dict(testrun.SUITES, {"resume": {"patterns": ("test_resume_*.py",),
                                                                  "dir": str(self.tests)}}),
                      mock.patch.object(testrun, "LOGS", folder / "logs"),
                      mock.patch.object(testrun.locks, "heavy", lambda *args, **kwargs: contextlib.nullcontext())):
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(sys.modules.pop, self.module, None)
        self.addCleanup(lambda: str(self.tests) in sys.path and sys.path.remove(str(self.tests)))

    def ran(self):
        found = self.trace.read_text(encoding="utf-8").split() if self.trace.exists() else []
        self.trace.unlink(missing_ok=True)
        return found

    def run_suite(self, pattern=None):
        """The tests that ran, the verdict and the report of one run of the suite."""
        with contextlib.redirect_stderr(io.StringIO()):
            verdict, report = testrun.run("resume", pattern=pattern)
        return self.ran(), verdict, json.loads(Path(report).read_text(encoding="utf-8"))

    def test_a_rerun_after_an_interruption_runs_only_the_unfinished_classes(self):
        self.once.write_text("stop", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            testrun.run("resume")
        self.assertEqual(self.ran(), ["A.test_one", "A.test_two", "B.test_one"])
        ran, verdict, report = self.run_suite()
        self.assertEqual((ran, verdict), (["B.test_one", "C.test_one", "D.test_one"], "pass"))
        self.assertEqual({t.split(".", 1)[1]: r["outcome"] for t, r in report["tests"].items()},
                         dict.fromkeys(EVERY, "pass"))
        self.assertEqual(report["resumed"]["classes"], [f"{self.module}.A"])
        log = Path(report["log"]).read_text(encoding="utf-8")
        for test in EVERY:                           # A's part of the stopped run's transcript, and the table
            cls, method = test.split(".")
            self.assertRegex(log, rf"{method} \({re.escape(self.module)}\.{cls}(\.{method})?\) \.\.\. ok")
            self.assertRegex(log, rf"\npass +[0-9.]+  {re.escape(f'{self.module}.{test}')}\n")

    def test_a_changed_input_reruns_every_class(self):
        self.assertEqual(self.run_suite()[:2], (EVERY, "pass"))
        ran, verdict, report = self.run_suite()
        self.assertEqual((ran, verdict, len(report["tests"])), ([], "pass", len(EVERY)))    # every class recorded
        with self.source.open("a", encoding="utf-8") as f:
            f.write("# an input of the suite changes, and with it the suite's key\n")
        self.assertEqual(self.run_suite()[:2], (EVERY, "pass"))

    def test_a_failed_class_never_counts(self):
        self.once.write_text("fail", encoding="utf-8")
        self.teardown.write_text("fail", encoding="utf-8")
        ran, verdict, _ = self.run_suite()
        self.assertEqual((ran, verdict), (EVERY, "error"))           # B fails and D's teardown errors
        ran, verdict, report = self.run_suite()
        self.assertEqual((ran, verdict), (["B.test_one", "D.test_one"], "pass"))
        self.assertEqual(report["resumed"]["classes"], [f"{self.module}.A", f"{self.module}.C"])

    def test_a_modules_classes_count_only_once_its_module_teardown_has_run_clean(self):
        flag = self.tests.parent / "module-teardown.txt"
        (self.tests / f"{self.module}_m.py").write_text(TORN_DOWN.format(trace=str(self.trace), flag=str(flag)),
                                                        encoding="utf-8")
        self.addCleanup(sys.modules.pop, f"{self.module}_m", None)
        flag.write_text("fail", encoding="utf-8")
        ran, verdict, _ = self.run_suite()
        self.assertEqual((ran, verdict), ([*EVERY, "E.test_one", "F.test_one"], "error"))
        ran, verdict, _ = self.run_suite()
        self.assertEqual((ran, verdict), (["E.test_one", "F.test_one"], "pass"))
        self.assertEqual(self.run_suite()[:2], ([], "pass"))

    def test_a_filtered_run_neither_skips_recorded_classes_nor_records_its_own(self):
        self.once.write_text("stop", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            testrun.run("resume")                    # A recorded
        self.ran()
        self.assertEqual(self.run_suite(pattern="*.A.*")[:2], (["A.test_one", "A.test_two"], "pass"))
        self.assertEqual(self.run_suite(pattern="*.C.*")[:2], (["C.test_one"], "pass"))
        self.assertEqual(self.run_suite()[0], ["B.test_one", "C.test_one", "D.test_one"])


if __name__ == "__main__":
    unittest.main()
