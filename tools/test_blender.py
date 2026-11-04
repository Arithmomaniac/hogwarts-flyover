"""tools.blender.fetch: the pinned Blender build, fetched once into the machine cache, and scripts run in it headless."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

from filelock import FileLock, Timeout

from tools import locks
from tools.blender import fetch
from tools.inputs import Blocked
from tools.paths import ROOT
from tools.test_stages import state

SCRIPT = Path("tools/blender/willow_tree.py")       # a lane's generator, by the naming rule; never run here


def unpacking(folder="blender-x"):
    """A stand-in for pooch.retrieve that unpacks an empty blender.exe; and, call by call, whether the build's cache
    folder still existed when it started."""
    seen = []

    def retrieve(*args, **kwargs):
        seen.append(fetch.home().exists())
        exe = fetch.home() / folder / "blender.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"")
    return retrieve, seen


class FetchTests(unittest.TestCase):
    def test_an_empty_cache_is_blocked_without_a_download(self):
        with state(), mock.patch("pooch.retrieve") as retrieve:
            with self.assertRaises(Blocked):
                fetch.executable(fetch=False)
        retrieve.assert_not_called()

    def test_the_download_must_match_the_pinned_hash(self):
        with state(), mock.patch("pooch.retrieve", side_effect=ValueError("SHA256 hash does not match")) as retrieve:
            with self.assertRaises(Blocked):
                fetch.executable()
        self.assertEqual(retrieve.call_args.kwargs["known_hash"], f"sha256:{fetch.SHA256}")

    def test_an_interrupted_unpacking_is_not_taken_for_the_build_and_is_unpacked_again(self):
        retrieve, seen = unpacking()
        with state():
            partial = fetch.home() / "blender-x" / "blender.exe"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"")
            self.assertIsNone(fetch.cached())
            with mock.patch("pooch.retrieve", side_effect=retrieve), \
                    mock.patch.object(fetch, "version", return_value=f"Blender {fetch.VERSION} LTS"):
                exe = fetch.executable()
            self.assertEqual(seen, [False])
            self.assertEqual((exe, fetch.cached()), (partial, partial))

    def test_a_build_that_reports_another_version_is_blocked_and_not_marked(self):
        retrieve, _ = unpacking()
        with state(), mock.patch("pooch.retrieve", side_effect=retrieve), \
                mock.patch.object(fetch, "version", return_value="Blender 5.2.2 LTS"):
            with self.assertRaises(Blocked):
                fetch.executable()
            self.assertIsNone(fetch.cached())

    def test_fetching_counts_as_heavy_work(self):
        retrieve, _ = unpacking()
        seen = []

        def recording(*args, **kwargs):
            seen.append(len(locks.heavy_work()))
            retrieve(*args, **kwargs)
        with state(), mock.patch("pooch.retrieve", side_effect=recording), \
                mock.patch.object(fetch, "version", return_value=f"Blender {fetch.VERSION} LTS"):
            fetch.executable()
        self.assertEqual(seen, [1])

    def test_the_fetch_lock_stays_free_while_the_fetch_waits_for_a_timing_run(self):
        # registered work that needs Blender may wait for the fetch lock, and a timing run waits for that work: holding
        # the fetch lock while waiting for the timing run would deadlock all three
        retrieve, _ = unpacking()
        hold = "import sys\nfrom tools import locks\nwith locks.held('quiet', 'a timing run'):\n" \
               "    print('held', flush=True)\n    sys.stdin.read()"
        with state():
            holder = subprocess.Popen([sys.executable, "-c", hold], env={**os.environ, "PYTHONPATH": str(ROOT)},
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            self.assertEqual(holder.stdout.readline().strip(), "held")
            found, free = [], []
            fetch_lock = FileLock(str(locks.folder() / "blender-fetch.lock"))
            with mock.patch("pooch.retrieve", side_effect=retrieve), \
                    mock.patch.object(fetch, "version", return_value=f"Blender {fetch.VERSION} LTS"):
                fetching = threading.Thread(target=lambda: found.append(fetch.executable()))
                fetching.start()
                for _ in range(30):
                    try:
                        fetch_lock.acquire(timeout=0)
                        fetch_lock.release()
                        free.append(True)
                    except Timeout:
                        free.append(False)
                    time.sleep(0.1)
                holder.stdin.close()
                holder.wait()
                fetching.join(30)
        self.assertTrue(all(free), free)
        self.assertEqual(len(found), 1)


class RunTests(unittest.TestCase):
    def test_the_fixed_arguments_precede_the_script_and_its_own_follow_the_separator(self):
        self.assertEqual(fetch.command(SCRIPT, "src/assets/willow/tree.npz", 3, exe="blender.exe"),
                         ["blender.exe", "--background", "--factory-startup", "--offline-mode", "--python-exit-code",
                          "1", "--python", str(SCRIPT), "--", "src/assets/willow/tree.npz", "3"])

    def test_a_script_runs_from_the_root_on_an_empty_profile_without_the_users_blender_variables(self):
        calls = []

        def started(command, **kwargs):
            calls.append((command, kwargs, sorted(Path(kwargs["env"]["BLENDER_USER_RESOURCES"]).iterdir())))
        with state(), mock.patch.dict(os.environ, {"BLENDER_USER_SCRIPTS": "elsewhere"}), \
                mock.patch.object(fetch.subprocess, "run", side_effect=started):
            fetch.run(SCRIPT, "tree.npz", exe="blender.exe")
        (command, kwargs, profile), = calls
        self.assertEqual(command, fetch.command(SCRIPT, "tree.npz", exe="blender.exe"))
        self.assertEqual((kwargs["cwd"], kwargs["check"]), (ROOT, True))
        self.assertEqual(profile, [])
        self.assertNotIn("BLENDER_USER_SCRIPTS", kwargs["env"])

    def test_a_script_run_counts_as_heavy_work(self):
        seen = []
        with state(), mock.patch.object(fetch.subprocess, "run", side_effect=lambda *a, **k: seen.append(len(locks.heavy_work()))):
            fetch.run(SCRIPT, exe="blender.exe")
        self.assertEqual(seen, [1])


class HeadlessTests(unittest.TestCase):
    """Blender itself, once `python -X utf8 -m tools.blender.fetch` has put it in the machine cache. The tools suite's
    stage key records whether it has, so fetching it reruns these."""

    @classmethod
    def setUpClass(cls):
        cls.exe = fetch.cached()
        if cls.exe is None:
            raise unittest.SkipTest(f"Blender {fetch.VERSION} is not in the machine cache")

    def run_script(self, body, *args):
        with tempfile.TemporaryDirectory(prefix="hog-blender-test-") as folder:
            script, out = Path(folder) / "probe.py", Path(folder) / "out.json"
            script.write_text(textwrap.dedent(body), encoding="utf-8")
            done = fetch.run(script, out, *args, exe=self.exe, check=False, capture_output=True, text=True,
                             timeout=600)
            return done, json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None

    def test_a_script_runs_headless_and_offline_in_the_pinned_version_with_its_arguments(self):
        done, seen = self.run_script("""
            import json, os, sys
            import bpy
            out, *args = sys.argv[sys.argv.index("--") + 1:]
            json.dump({"version": list(bpy.app.version), "background": bpy.app.background,
                       "online": bpy.app.online_access, "args": args, "cwd": os.getcwd(),
                       "user": bpy.utils.resource_path("USER"), "profile": os.environ["BLENDER_USER_RESOURCES"]},
                      open(out, "w", encoding="utf-8"))
        """, "src/assets/willow/tree.npz", 3)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(seen["version"], [int(n) for n in fetch.VERSION.split(".")])
        self.assertEqual((seen["background"], seen["online"]), (True, False))
        self.assertEqual(seen["args"], ["src/assets/willow/tree.npz", "3"])
        self.assertEqual(Path(seen["cwd"]), ROOT)
        self.assertEqual(Path(seen["user"]), Path(seen["profile"]))

    def test_an_exception_in_the_script_exits_1(self):
        done, _ = self.run_script("""
            raise RuntimeError("the generator failed")
        """)
        self.assertEqual(done.returncode, 1)


if __name__ == "__main__":
    unittest.main()
