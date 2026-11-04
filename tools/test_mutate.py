"""tools.mutate.run: a mutation is judged only by tests that pass without it. A module's listed expected failures fail
with or without the mutation, so they neither invalidate the unmutated run nor count as catching the mutation."""
import contextlib
import io
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import locks
from tools.mutate import run
from tools.paths import ROOT
from tools.test_stages import state

MODULE = """
import unittest

DOOR_FT = 5
EXPECTED_FAILURES = ("ShedTests.test_the_roof_keeps_the_rain_out",)


class ShedTests(unittest.TestCase):
    def test_the_door_is_5_ft(self):
        self.assertEqual(DOOR_FT, 5)

    def test_the_roof_keeps_the_rain_out(self):
        self.fail("the roof is not built yet")
"""
LISTED = "test_shed.ShedTests.test_the_roof_keeps_the_rain_out"


def mutation(name, new, tests=("test_shed.ShedTests",)):
    return run.Mutation(name, "test_shed.py", "DOOR_FT = 5", new, tests, rebuild=False)


def verdicts(*mutations, inputs=(), module=MODULE, files=None):
    """main's verdict for each mutation of a stand-in worktree holding test_shed.py (and `files`, {path: text} from its
    root), its module needing `inputs`."""
    with tempfile.TemporaryDirectory(prefix="hog-mutate-test-") as folder, state():
        root = Path(folder)
        (root / "src").mkdir()
        (root / "src" / "test_shed.py").write_text(textwrap.dedent(module), encoding="utf-8")
        for rel, text in (files or {}).items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(textwrap.dedent(text), encoding="utf-8")
        with mock.patch.object(run, "ROOT", root), mock.patch.object(run, "COPIED", ()), \
                mock.patch.object(run.importlib, "import_module",
                                  return_value=SimpleNamespace(MUTATIONS=list(mutations), INPUTS=inputs)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            run.main(["shed"])
    return dict(line.removeprefix("shed: ").split(": ", 1) for line in out.getvalue().splitlines())


class ExpectedFailureTests(unittest.TestCase):
    def test_tests_skipped_as_blocked_make_a_mutation_blocked_not_survived(self):
        # a page test skips as blocked when Edge renders on another GPU than HOG_GPU names: counted as passing, its
        # mutation would look as if it had survived
        module = MODULE.replace("class ShedTests(unittest.TestCase):\n",
                                "class ShedTests(unittest.TestCase):\n"
                                "    @classmethod\n"
                                "    def setUpClass(cls):\n"
                                "        raise unittest.SkipTest('blocked: HOG_GPU=amd, but Edge renders on another')\n\n")
        found = verdicts(mutation("a 4 ft door", "DOOR_FT = 4"), module=module)
        self.assertTrue(found["a 4 ft door"].startswith("BLOCKED"), found)

    def test_a_listed_failure_neither_invalidates_nor_catches_a_mutation(self):
        found = verdicts(mutation("a 4 ft door", "DOOR_FT = 4"),
                         mutation("a change only the roof test sees", "DOOR_FT = 5  # the roof's turn"))
        self.assertEqual(found, {"a 4 ft door": "caught", "a change only the roof test sees": "SURVIVED"})

    def test_a_mutation_named_only_for_listed_failures_is_invalid(self):
        found = verdicts(mutation("a 4 ft door", "DOOR_FT = 4", tests=(LISTED,)))
        self.assertEqual(found["a 4 ft door"], run.ONLY_LISTED)


class InputTests(unittest.TestCase):
    def test_a_module_whose_inputs_are_missing_reports_its_mutations_blocked(self):
        missing = run.inputs.Blocked("input data-before-railway-x is not available")
        with mock.patch.object(run.inputs, "require", side_effect=missing):
            found = verdicts(mutation("a 4 ft door", "DOOR_FT = 4"), inputs=("data-before-railway-x",))
        self.assertTrue(found["a 4 ft door"].startswith("BLOCKED (input data-before-railway-x"), found)


class ToolImportTests(unittest.TestCase):
    def test_a_test_that_imports_the_repositorys_tools_judges_its_mutations(self):
        # test_shaders and test_site import tools: without tools/ in the copy they cannot even load there
        module = """
        import sys
        import unittest
        from pathlib import Path

        sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
        from tools.shed_plan import ROOF_PITCH_DEG  # noqa: E402

        DOOR_FT = 5


        class ShedTests(unittest.TestCase):
            def test_the_door_is_5_ft(self):
                self.assertEqual(DOOR_FT, 5)
                self.assertEqual(ROOF_PITCH_DEG, 40)
        """
        found = verdicts(mutation("a 4 ft door", "DOOR_FT = 4"), module=module,
                         files={"tools/__init__.py": "", "tools/shed_plan.py": "ROOF_PITCH_DEG = 40\n"})
        self.assertEqual(found, {"a 4 ft door": "caught"})


class HeavyWorkTests(unittest.TestCase):
    def judged(self, *mutations, gpu=None):
        """[(GPU lock held: amd, nvidia; heavy work registered)] in each test run of `mutations`, and the verdicts with
        what was registered as each was printed."""
        seen, printed = [], []

        def record(m, mutate, kept=()):
            seen.append((tuple((locks.folder() / f"gpu-{g}.json").is_file() for g in locks.GPUS),
                         len(locks.heavy_work())))
            return "fail" if mutate else "pass"
        with state(), mock.patch.dict(run.os.environ, {"HOG_GPU": gpu or ""}), \
                mock.patch.object(run, "run", side_effect=record), mock.patch.object(run, "check_anchors"), \
                mock.patch.object(run, "print", create=True,
                                  side_effect=lambda line, **kw: printed.append((line, len(locks.heavy_work())))), \
                mock.patch.object(run.importlib, "import_module", return_value=SimpleNamespace(MUTATIONS=list(mutations))):
            run.main(["shed"])
        return seen, printed

    def test_a_mutation_run_counts_as_heavy_work(self):
        # its builds and test runs load the machine, so a timing run must wait for it
        seen, _ = self.judged(mutation("a 4 ft door", "DOOR_FT = 4"))
        self.assertEqual(seen, [((False, False), 1), ((False, False), 1)])

    def test_each_mutation_registers_on_its_own(self):
        # so a timing run can start between two mutations instead of waiting for the whole run
        _, printed = self.judged(mutation("a 4 ft door", "DOOR_FT = 4"), mutation("a 3 ft door", "DOOR_FT = 3"))
        self.assertEqual(printed, [("shed: a 4 ft door: caught", 0), ("shed: a 3 ft door: caught", 0)])

    def test_a_mutation_whose_tests_drive_a_browser_holds_its_gpu(self):
        # its tests run Edge on that GPU, so a suite on it must wait for them rather than overlap them
        seen, _ = self.judged(mutation("a 4 ft door seen in the page", "DOOR_FT = 4", tests=("test_render_shed.ShedTests",)),
                              mutation("a 3 ft door", "DOOR_FT = 3"), gpu="nvidia")
        self.assertEqual(seen, [((False, True), 1), ((False, True), 1), ((False, False), 1), ((False, False), 1)])

    def test_a_mutation_whose_tests_drive_a_browser_is_blocked_without_a_gpu(self):
        seen, printed = self.judged(mutation("a 4 ft door seen in the page", "DOOR_FT = 4",
                                             tests=("test_mesh_probe.ShedTests",)))
        self.assertEqual(seen, [])
        self.assertTrue(printed[0][0].startswith("shed: a 4 ft door seen in the page: BLOCKED"), printed)


class RebuildTests(unittest.TestCase):
    @staticmethod
    def steps(file):
        """The scripts a rebuilding mutation of src/`file` runs in its copy before its tests."""
        ran = []

        def python(*args, cwd):
            ran.append(args[0])
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory(prefix="hog-mutate-test-") as folder:
            root = Path(folder)
            (root / "src").mkdir()
            (root / "src" / file).write_text("DOOR_FT = 5\n", encoding="utf-8")
            with mock.patch.object(run, "ROOT", root), mock.patch.object(run, "COPIED", ()), \
                    mock.patch.object(run, "python", side_effect=python):
                run.run(run.Mutation("a 4 ft door", file, "DOOR_FT = 5", "DOOR_FT = 4", ("test_shed.ShedTests",)), mutate=True)
        return ran[:-1]                                     # the last runs the tests

    def test_a_page_template_or_shader_mutation_rebuilds_the_page_alone(self):
        # the payload's build never reads them, and it takes minutes
        for file in ("template_c.html", "frag5.glsl"):
            with self.subTest(file=file):
                self.assertEqual(self.steps(file), ["src/assemble.py"])

    def test_any_other_mutation_rebuilds_the_payload_and_the_page(self):
        self.assertEqual(self.steps("mesh_levels.py"), ["src/build_meshes.py", "src/assemble.py"])


class SiteMutationTests(unittest.TestCase):
    def test_every_site_mutation_finds_its_text_exactly_once(self):
        # tools.site.mutate checks each mutation's text only when it reaches it, hours into a full run
        from tools.site import mutate
        for m in mutate.MUTATIONS:
            with self.subTest(mutation=m.name):
                self.assertEqual((ROOT / m.file).read_text(encoding="utf-8").count(m.old), 1, m.file)


class FamilyMutationTests(unittest.TestCase):
    def test_every_recorded_mutation_finds_its_text_exactly_once(self):
        # tools.mutate.run checks them before a run; this sees a stale one when the code it anchors on is edited
        for path in sorted((ROOT / "tools" / "mutate").glob("*.py")):
            if path.stem in ("__init__", "run", "runner"):
                continue
            for m in run.importlib.import_module(f"tools.mutate.{path.stem}").MUTATIONS:
                with self.subTest(module=path.stem, mutation=m.name):
                    self.assertEqual((ROOT / "src" / m.file).read_text(encoding="utf-8").count(m.old), 1, m.file)


if __name__ == "__main__":
    unittest.main()
