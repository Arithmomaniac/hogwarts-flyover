"""tools.stages: what a stage's key covers, when the ledger lets a verdict be reused, the staleness gate, and the stages
that must be blocked rather than passed."""
import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from tools import stages
from tools.inputs import Blocked
from tools.paths import ROOT


@contextlib.contextmanager
def state():
    """A private state folder (ledger, hash cache, locks), so tests never touch the machine's."""
    with tempfile.TemporaryDirectory(prefix="hog-state-") as folder, \
            mock.patch.dict(os.environ, {"HOG_STATE_DIR": folder}):
        yield Path(folder)


@contextlib.contextmanager
def sandbox():
    """A copy of the sources (and stand-ins for the generated files) for stages to read in place of the worktree."""
    with tempfile.TemporaryDirectory(prefix="hog-sandbox-") as folder, state():
        root = Path(folder)
        for tree in ("src", "tools"):
            shutil.copytree(ROOT / tree, root / tree, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("requirements.lock.txt", "README.md", *stages.ROOT_DATA):
            shutil.copy2(ROOT / name, root / name)
        (root / "build").mkdir()
        for name in ("build/data.js", "build/meshes.js", "hogwarts-flyover.html"):
            (root / name).write_text(name, encoding="utf-8")
        with mock.patch.object(stages, "ROOT", root), mock.patch.object(stages, "SRC", root / "src"), \
                mock.patch.object(stages, "LOGS", root / "build" / "logs"), \
                mock.patch.object(stages, "browser", lambda gpu: {"gpu": gpu}):
            stages._closure.cache_clear()
            try:
                yield root
            finally:
                stages._closure.cache_clear()


def counting(verdicts):
    """A stage whose input is `inputs` and whose runs return the given verdicts in turn; and the list of its runs."""
    runs, inputs = [], {"value": 1}
    stage = stages.Stage("demo", lambda: dict(inputs), lambda key: (runs.append(key), (next(verdicts), None, {}))[1])
    return stage, runs, inputs


class KeyTests(unittest.TestCase):
    def test_a_test_edit_reruns_s4_but_a_readme_edit_does_not(self):
        with sandbox() as root:
            key = lambda: stages.key_of(stages.suite_inputs("render", "nvidia"))
            before = key()
            (root / "README.md").write_text("edited\n", encoding="utf-8")
            self.assertEqual(key(), before)
            test = root / "src" / "test_render_castle.py"
            test.write_text(test.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
            self.assertNotEqual(key(), before)

    def test_the_tool_tests_rerun_when_a_script_they_run_changes(self):
        with sandbox() as root:
            before = stages.key_of(stages.suite_inputs("tools"))
            probe = root / "src" / "mesh_probe.py"           # test_locks runs it as a subprocess
            probe.write_text(probe.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
            self.assertNotEqual(stages.key_of(stages.suite_inputs("tools")), before)

    def test_the_site_tests_rerun_when_a_file_they_read_by_path_changes(self):
        with sandbox() as root:
            for name in ("tools/site/review_at_x.json", "tools/site/locks/site-glo30.json", "tools/inputs.lock.json",
                         "tools/original-intermediates.lock.json", "tools/stages.py"):
                with self.subTest(name=name):
                    before = stages.key_of(stages.suite_inputs("site"))
                    path = root / name
                    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                    self.assertNotEqual(stages.key_of(stages.suite_inputs("site")), before)

    def test_fetching_blender_reruns_the_tool_tests_that_skipped_without_it(self):
        with sandbox():
            before = stages.key_of(stages.suite_inputs("tools"))
            with mock.patch("tools.blender.fetch.cached", return_value=Path("blender.exe")):
                self.assertNotEqual(stages.key_of(stages.suite_inputs("tools")), before)

    def test_the_asset_tests_rerun_when_the_registration_changes(self):
        with sandbox() as root:
            before = stages.key_of(stages.suite_inputs("assets"))
            georef = root / "georef.json"
            georef.write_text(georef.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            self.assertNotEqual(stages.key_of(stages.suite_inputs("assets")), before)

    def test_the_uncached_rebuild_reruns_when_any_source_changes(self):
        # it runs build_meshes.py as a subprocess, which loads the family builders by name: no import reaches them
        with sandbox() as root:
            before = stages.key_of(stages.suite_inputs("uncached"))
            builder = root / "src" / "mesh_village.py"
            builder.write_text(builder.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
            self.assertNotEqual(stages.key_of(stages.suite_inputs("uncached")), before)

    def test_a_shifted_part_is_keyed_on_its_shift_and_its_sources(self):
        # the asset tests' raised and lowered worlds reuse their parts until the shift or a source the parts read changes
        with sandbox() as root:
            key = lambda shift: stages.key_of(stages.part_inputs("willow", shift))
            before = {shift: key(shift) for shift in (None, 140, -140)}
            self.assertEqual(len(set(before.values())), 3)
            builder = root / "src" / "mesh_willow.py"
            builder.write_text(builder.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
            edited = {shift: key(shift) for shift in before}
            for shift in before:
                with self.subTest(shift=shift):
                    self.assertNotEqual(edited[shift], before[shift])
            raising = root / "src" / "datum_shift.py"              # how a world is raised: only the shifted parts read it
            raising.write_text(raising.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
            self.assertEqual(key(None), edited[None])
            self.assertNotEqual(key(140), edited[140])

    def test_the_shifted_asset_tests_rerun_when_their_parts_would_be_built_differently(self):
        with sandbox() as root:
            key = lambda shift: stages.key_of(stages.suite_inputs("assets", shift=shift))
            before = {shift: key(shift) for shift in (None, 140)}
            builder = root / "tools" / "shifted_part.py"
            builder.write_text(builder.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
            self.assertEqual(key(None), before[None])
            self.assertNotEqual(key(140), before[140])

    def test_a_family_asset_rebuilds_that_family_and_the_page_but_no_other_family(self):
        with sandbox() as root:
            parts = lambda: {f: stages.key_of(stages.part_inputs(f)) for f in ("willow", "castle")}
            page = lambda: stages.key_of(stages.s2_stage().inputs())    # the staleness gate rebuilds through S2
            before, page_before = parts(), page()
            export = root / "src" / "assets" / "willow" / "stage-key-probe.json"    # beside any exports a lane commits
            export.parent.mkdir(parents=True, exist_ok=True)
            export.write_text('{"limbs": 3}\n', encoding="utf-8")
            after = parts()
            self.assertNotEqual(after["willow"], before["willow"])
            self.assertEqual(after["castle"], before["castle"])
            self.assertNotEqual(page(), page_before)

    def test_family_assets_are_stored_byte_for_byte(self):
        # S1 keys hash the files' bytes, so Git must not convert their line endings in any worktree
        found = subprocess.run(["git", "check-attr", "text", "--", "src/assets/willow/tree.json"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout
        self.assertEqual(found.strip(), "src/assets/willow/tree.json: text: unset")

    def test_a_shared_helper_edit_reruns_every_suite_that_imports_it(self):
        with sandbox() as root:
            before = {s: stages.key_of(stages.suite_inputs(s, "nvidia")) for s in ("render", "smoke")}
            helper = root / "src" / "render_testing.py"
            helper.write_text(helper.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
            for suite, key in before.items():
                with self.subTest(suite=suite):
                    self.assertNotEqual(stages.key_of(stages.suite_inputs(suite, "nvidia")), key)

    def test_an_absent_gpu_blocks_a_browser_stage(self):
        with mock.patch.object(stages, "machine", return_value={"adapters": [("NVIDIA RTX 2000", "32.0")], "edge": "156"}):
            self.assertEqual(stages.browser("nvidia")["adapter"], ("NVIDIA RTX 2000", "32.0"))
            with self.assertRaises(Blocked):
                stages.browser("amd")


class LedgerTests(unittest.TestCase):
    def test_a_pass_is_reused_until_an_input_changes(self):
        with state():
            stage, runs, inputs = counting(iter(["pass", "pass"]))
            self.assertTrue(stages.execute(stage))
            self.assertTrue(stages.execute(stage))
            self.assertEqual(len(runs), 1)
            inputs["value"] = 2
            self.assertTrue(stages.execute(stage))
            self.assertEqual(len(runs), 2)

    def test_fail_error_and_blocked_verdicts_are_never_reused(self):
        with state():
            stage, runs, _ = counting(iter(["fail", "error", "blocked", "pass"]))
            self.assertEqual([stages.execute(stage) for _ in range(5)], [False, False, False, True, True])
            self.assertEqual(len(runs), 4)

    def test_a_pass_is_reused_while_its_outputs_hold_and_rerun_once_they_change(self):
        with sandbox() as root:
            runs = []
            stage = stages.Stage("demo", lambda: {},
                                 lambda key: (runs.append(key), ("pass", None, stages.files("build/meshes.js")))[1])
            stages.execute(stage)
            stages.execute(stage)
            self.assertEqual(len(runs), 1)
            (root / "build" / "meshes.js").write_text("edited by hand", encoding="utf-8")
            stages.execute(stage)
            self.assertEqual(len(runs), 2)

    def test_s0_is_blocked_without_the_pinned_input(self):
        with state() as folder, mock.patch.object(stages.inputs, "BUILD", folder), \
                mock.patch.dict(os.environ, {"HOG_INPUT_DATA_BEFORE_RAILWAY_X": str(folder / "missing.js")}):
            self.assertFalse(stages.execute(stages.s0_stage()))
            (entry,) = stages.history("s0")
            self.assertEqual(entry["verdict"], "blocked")
            self.assertIn("data-before-railway", entry["note"])


class StalenessTests(unittest.TestCase):
    def test_a_source_change_committed_without_its_rebuild_fails_the_gate(self):
        with tempfile.TemporaryDirectory(prefix="hog-stale-") as folder, state() as logs:
            root = Path(folder)
            git = lambda *args: subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
            git("init", "-q")
            git("config", "user.email", "test@example.com")
            git("config", "user.name", "test")
            (root / "src").mkdir()
            (root / "src" / "mesh_hut.py").write_text("DOOR_FT = 5\n", encoding="utf-8")
            for name in stages.GENERATED:
                (root / name).parent.mkdir(exist_ok=True)
                (root / name).write_text("built with a 5 ft door\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-qm", "built")
            with mock.patch.object(stages, "ROOT", root), mock.patch.object(stages, "LOGS", logs):   # outside the repository
                self.assertTrue(stages.execute(stages.stale_stage()))
                (root / "src" / "mesh_hut.py").write_text("DOOR_FT = 4\n", encoding="utf-8")
                git("commit", "-qam", "a 4 ft door, without rebuilding")
                (root / "build" / "meshes.js").write_text("built with a 4 ft door\n", encoding="utf-8")   # what s1 and s2 write
                self.assertFalse(stages.execute(stages.stale_stage()))
                git("commit", "-qam", "the rebuilt payload")
                self.assertTrue(stages.execute(stages.stale_stage()))
                (root / "src" / "mesh_willow.py").write_text("FAMILY = 'willow'\n", encoding="utf-8")   # loaded by name
                self.assertFalse(stages.execute(stages.stale_stage()))                               # but not committed

    def test_a_tree_git_cannot_read_is_an_error_not_a_pass(self):
        with tempfile.TemporaryDirectory(prefix="hog-stale-") as folder, state() as logs, \
                mock.patch.dict(os.environ, {"GIT_DIR": str(Path(folder) / "no-such-git-dir")}), \
                mock.patch.object(stages, "ROOT", Path(folder)), mock.patch.object(stages, "LOGS", logs):
            self.assertFalse(stages.execute(stages.stale_stage()))
            (entry,) = stages.history("stale")
            self.assertEqual(entry["verdict"], "error")


def timed_page(lake=284.0, look_height=40):
    """A page as the timing checks read it: its DATA object and its tour, from the waypoints through pathAt."""
    return (f'<script>const DATA={{"meta": {{"lake": {lake}}}}};\n'
            f'  const WP = [\n    [1100, -1500, 300, 0, 0, {look_height}, "Hogwarts Castle"],\n  ];\n'
            '  function pathAt(t) { return t; }\n\n  // ---------------------------------------------------------------- HUD\n'
            '</script>')


class TimingTests(unittest.TestCase):
    def test_timing_is_blocked_while_the_baseline_carries_other_data_or_flies_another_tour(self):
        # S6 sets the two pages at the same tour times; the camera follows the waypoints, the ground and the trees, so a
        # baseline with other data or another tour shows other views
        with sandbox() as root:
            page, baseline = root / "hogwarts-flyover.html", root / "build" / "baseline-x.html"
            page.write_text(timed_page(), encoding="utf-8")
            stage = stages.timing_stage("s6:amd", "amd", lambda: [])
            with mock.patch.object(stages.inputs, "require", return_value=baseline), \
                    mock.patch.object(stages.inputs, "spec", return_value={"sha256": "575ad974"}):
                for other, why in ((timed_page(lake=153.0), "other data"), (timed_page(look_height=60), "another tour")):
                    with self.subTest(baseline=why):
                        baseline.write_text(other, encoding="utf-8")
                        with self.assertRaisesRegex(Blocked, f"{why}.*tools.site.baseline"):
                            stage.inputs()
                baseline.write_text(page.read_text(encoding="utf-8"), encoding="utf-8")
                self.assertEqual(stage.inputs()["baseline"], "575ad974")

    def test_the_candidate_pass_timer_needs_no_baseline(self):
        # it times the page's landscape against its integrated scene, within the page alone
        with sandbox() as root:
            (root / "hogwarts-flyover.html").write_text(timed_page(), encoding="utf-8")
            stage = stages.timing_stage("passes_candidate:amd", "amd", lambda: [], baseline=False)
            with mock.patch.object(stages.inputs, "require", side_effect=Blocked("no baseline-x here")):
                self.assertNotIn("baseline", stage.inputs())

    def test_a_hand_run_of_the_pass_timer_is_blocked_while_the_baseline_carries_other_data(self):
        from tools.timing import passes
        with tempfile.TemporaryDirectory(prefix="hog-passes-") as folder:
            page, baseline, out = Path(folder) / "page.html", Path(folder) / "baseline-x.html", Path(folder) / "out.json"
            page.write_text(timed_page(), encoding="utf-8")
            baseline.write_text(timed_page(lake=153.0), encoding="utf-8")
            argv = ["--gpu", "amd", "--mode", "pages", "--candidate", str(page), "--out", str(out)]
            with mock.patch("tools.locks.require_quiet"), mock.patch.object(passes.inputs, "require", return_value=baseline), \
                    mock.patch.object(passes, "time_pages", mock.AsyncMock(return_value={})) as timed:
                self.assertEqual(passes.main(argv), 3)
                timed.assert_not_called()
                baseline.write_text(page.read_text(encoding="utf-8"), encoding="utf-8")
                self.assertEqual(passes.main(argv), 0)
                timed.assert_called_once_with("amd", baseline, page)


if __name__ == "__main__":
    unittest.main()
