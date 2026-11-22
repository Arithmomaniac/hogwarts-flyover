"""tools.locks: a second run on one GPU waits for the first, the two GPUs do not wait for each other, a timing run
waits for heavy work to end, a timing script refuses to run without the quiet lock's token, and every wait is logged."""
import contextlib
from datetime import datetime
import io
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import psutil

from tools import locks
from tools.paths import ROOT, SRC

HOLD = """
import sys, threading
from filelock import FileLock
from tools import locks
kind, seconds = sys.argv[1], float(sys.argv[2])
holds = {"heavy": lambda: locks.heavy("a build"), "quiet": lambda: locks.held("quiet", "a timing run"),
         "log": lambda: FileLock(str(locks.waits_file()) + ".lock")}
released = threading.Event()
threading.Thread(target=lambda: (sys.stdin.read(), released.set()), daemon=True).start()
with holds.get(kind, lambda: locks.gpu(kind, "the first run"))():
    print("held", flush=True)
    released.wait(seconds)
"""


def gone(*paths, seconds=10.0):
    """Whether every one of `paths` has been deleted within `seconds`."""
    deadline = time.monotonic() + seconds
    while any(path.exists() for path in paths):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


class LockTests(unittest.TestCase):
    def setUp(self):
        state = tempfile.TemporaryDirectory(prefix="hog-locks-")
        self.addCleanup(state.cleanup)
        self.env = {**os.environ, "HOG_STATE_DIR": state.name, "PYTHONPATH": str(ROOT)}
        patch = mock.patch.dict(os.environ, {"HOG_STATE_DIR": state.name})
        patch.start()
        self.addCleanup(patch.stop)

    def hold(self, kind, seconds):
        """Another process holding a lock for at most `seconds`; returns once it holds it. The test's end releases it, and
        waits for it to exit (the venv's python.exe is a launcher, whose interpreter child holds the lock files)."""
        holder = subprocess.Popen([sys.executable, "-c", HOLD, kind, str(seconds)], env=self.env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)
        self.addCleanup(holder.wait)
        self.addCleanup(holder.stdin.close)                    # cleanups run last first: release, then wait
        self.assertEqual(holder.stdout.readline().strip(), "held")
        holder.stdout.close()
        return lambda: (holder.stdin.close(), holder.wait())

    def gpu_command(self, gpu, code):
        """`tools.locks gpu GPU -- python -c CODE` started in the background."""
        cli = subprocess.Popen([sys.executable, "-X", "utf8", "-m", "tools.locks", "gpu", gpu, "--", sys.executable, "-c",
                                code], cwd=ROOT, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: cli.poll() is None and cli.kill())
        return cli

    def says(self, process, words, seconds=30.0):
        """Waits until `process` says something with `words` in it on stderr."""
        lines = queue.Queue()
        threading.Thread(target=lambda: [lines.put(line) for line in process.stderr], daemon=True).start()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            with contextlib.suppress(queue.Empty):
                if words in lines.get(timeout=0.5):
                    return
        self.fail(f"it never said {words!r}")

    def test_a_second_run_on_the_same_gpu_waits_for_the_first(self):
        self.hold("nvidia", 2.0)
        said, started = [], time.monotonic()
        with locks.gpu("nvidia", "the second run", say=said.append):
            waited = time.monotonic() - started
        self.assertGreater(waited, 1.0)
        self.assertIn("the first run", said[0])

    def test_the_two_gpus_do_not_wait_for_each_other(self):
        self.hold("nvidia", 2.0)
        started = time.monotonic()
        with locks.gpu("amd", "an AMD run", say=self.fail):
            self.assertLess(time.monotonic() - started, 1.0)

    def test_a_timing_run_waits_for_heavy_work_to_end(self):
        self.hold("heavy", 2.0)
        said, started = [], time.monotonic()
        with locks.quiet("timing", say=said.append, poll_s=0.2):
            waited = time.monotonic() - started
        self.assertGreater(waited, 1.0)
        self.assertIn("a build", said[0])

    def test_a_timing_run_without_the_quiet_lock_is_refused(self):
        for command in ([str(SRC / "mesh_probe.py"), "--gpu", "nvidia", "--baseline", "b.html", "--out", "p.json"],
                        ["-m", "tools.timing.passes", "--gpu", "nvidia"]):
            with self.subTest(command=command[0]):
                run = subprocess.run([sys.executable, "-X", "utf8", *command], cwd=ROOT, env=self.env, capture_output=True,
                                     text=True)
                self.assertEqual(run.returncode, 3, run.stderr)
                self.assertIn("quiet lock", run.stderr)

    def test_the_quiet_lock_hands_its_command_the_token(self):
        check = "from tools import locks; locks.require_quiet('a timing script'); print('timing')"
        with locks.quiet("timing") as token:
            run = subprocess.run([sys.executable, "-c", check], env={**self.env, locks.TOKEN: token}, capture_output=True,
                                 text=True)
        self.assertEqual((run.returncode, run.stdout.strip()), (0, "timing"))
        stale = subprocess.run([sys.executable, "-c", check], env={**self.env, locks.TOKEN: token}, capture_output=True)
        self.assertEqual(stale.returncode, 3)                      # the token ends with the lock

    def test_the_heavy_command_registers_its_command_while_it_runs(self):
        check = "import sys; from tools import locks; print(*(h['label'] for h in locks.heavy_work()), sep='\\n'); sys.exit(7)"
        run = subprocess.run([sys.executable, "-X", "utf8", "-m", "tools.locks", "heavy", "--", sys.executable, "-c", check],
                             cwd=ROOT, env=self.env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 7, run.stderr)
        self.assertIn(sys.executable, run.stdout)
        self.assertEqual(locks.heavy_work(), [])

    def test_the_gpu_command_holds_its_gpus_lock_and_registers_its_command_while_it_runs(self):
        # a one-off browser script run through it queues behind a suite on that GPU instead of slowing it, and a timing
        # run waits for it
        check = ("import json, sys; from tools import locks\n"
                 "held = {g: (locks.read(locks.folder() / f'gpu-{g}.json') or {}).get('label') for g in locks.GPUS}\n"
                 "print(json.dumps({'held': held, 'heavy': [h['label'] for h in locks.heavy_work()]})); sys.exit(7)")
        run = subprocess.run([sys.executable, "-X", "utf8", "-m", "tools.locks", "gpu", "amd", "--", sys.executable, "-c",
                              check], cwd=ROOT, env=self.env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 7, run.stderr)
        seen = json.loads(run.stdout)
        self.assertIn(sys.executable, seen["held"]["amd"])
        self.assertIsNone(seen["held"]["nvidia"])
        self.assertEqual(len(seen["heavy"]), 1)
        self.assertIn(sys.executable, seen["heavy"][0])
        self.assertEqual(locks.heavy_work(), [])
        self.assertIsNone(locks.read(locks.folder() / "gpu-amd.json"))

    def test_the_gpu_command_refuses_a_gpu_it_does_not_know_without_running_the_command(self):
        run = subprocess.run([sys.executable, "-X", "utf8", "-m", "tools.locks", "gpu", "intel", "--", sys.executable, "-c",
                              "import sys; sys.exit(7)"], cwd=ROOT, env=self.env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 2, run.stderr)

    def test_the_gpu_command_waits_for_its_gpu_without_registering_heavy_work(self):
        # registered work would hold a quiet window up for the whole wait
        release = self.hold("nvidia", 30.0)
        cli = self.gpu_command("nvidia", "import sys; from tools import locks; print(len(locks.heavy_work())); sys.exit(7)")
        self.says(cli, "waiting for the gpu-nvidia lock")
        self.assertEqual(locks.heavy_work(), [])
        release()
        self.assertEqual(cli.wait(timeout=60), 7)
        self.assertEqual(cli.stdout.read().strip(), "1")

    def test_the_gpu_command_lets_its_gpu_go_while_a_timing_run_holds_the_quiet_lock(self):
        # holding the GPU while it waits for the timing run could deadlock with a worktree whose tools register first and
        # then wait for that GPU; registering would hold the timing run up. Once the timing run ends it takes the GPU
        # again
        release = self.hold("quiet", 30.0)
        cli = self.gpu_command("amd", "import sys; from tools import locks\n"
                                      "holder = locks.read(locks.folder() / 'gpu-amd.json')\n"
                                      "print(len(locks.heavy_work()), locks.ancestor(holder)); sys.exit(7)")
        self.says(cli, "waiting for the quiet lock")
        self.assertIsNone(locks.read(locks.folder() / "gpu-amd.json"))
        self.assertEqual(locks.heavy_work(), [])
        release()
        self.assertEqual(cli.wait(timeout=60), 7)
        self.assertEqual(cli.stdout.read().split(), ["1", "True"])

    def test_a_moment_on_the_quiet_lock_does_not_cost_a_tool_its_gpu(self):
        # every registration holds the quiet lock for a moment: letting a GPU go for that would hand it to the next in
        # line
        self.hold("quiet", 1.0)
        said = []
        with locks.gpu_work("amd", "a tool", say=said.append):
            self.assertEqual(len(locks.heavy_work()), 1)
        self.assertEqual([s for s in said if "lets the amd GPU go" in s], [])

    def test_a_tool_that_takes_the_same_gpu_runs_under_the_gpu_command(self):
        # rather than waiting for ever for the lock its own parent holds; it learns the GPU from HOG_GPU
        tool = ("import os, sys; from tools import locks\n"
                "with locks.gpu_work(os.environ['HOG_GPU'], 'a tool that takes the lock itself'):\n"
                "    print(os.environ['HOG_GPU'], len(locks.heavy_work())); sys.exit(7)")
        cli = self.gpu_command("amd", tool)
        self.assertEqual(cli.wait(timeout=60), 7, cli.stderr.read())
        self.assertEqual(cli.stdout.read().split(), ["amd", "2"])

    def test_the_site_tools_register_their_work_while_it_runs(self):
        # the site's data build and its mutation and terrain checks run for minutes: a timing run must wait for them
        from tools.site import build, compare, mutate

        class Stop(Exception):
            pass

        def first_step(*args, **kwargs):
            seen.append([h["pid"] for h in locks.heavy_work()])
            raise Stop
        out = tempfile.TemporaryDirectory(prefix="hog-compare-")
        self.addCleanup(out.cleanup)
        for tool, step, argv in ((build, "build", ["x"]), (mutate, "check", [mutate.MUTATIONS[0].name]),
                                 (compare, "grid_of", [out.name])):
            seen = []
            with self.subTest(tool=tool.__name__), mock.patch.object(tool, step, first_step), self.assertRaises(Stop):
                tool.main(argv)
            self.assertEqual(seen, [[os.getpid()]], tool.__name__)
            self.assertEqual(locks.heavy_work(), [])

    def test_work_inside_registered_work_does_not_wait_for_the_quiet_lock(self):
        # waiting would deadlock: a timing run holds the quiet lock while it waits for the enclosing work. Here another
        # process holds it for 30 s, so any such wait would take that long. A child process registers its own entry.
        child = ("import time; from tools import locks; started = time.monotonic()\n"
                 "with locks.heavy('a build inside it'):\n"
                 "    print(len(locks.heavy_work()), time.monotonic() - started < 1.0)")
        with locks.heavy("a suite"):
            self.hold("quiet", 30.0)
            started = time.monotonic()
            with locks.heavy("a tool the suite runs"):
                self.assertEqual(len(locks.heavy_work()), 1)
            self.assertLess(time.monotonic() - started, 1.0)
            run = subprocess.run([sys.executable, "-c", child], env={**self.env, **os.environ}, capture_output=True,
                                 text=True, timeout=60)
        self.assertEqual(run.stdout.split(), ["2", "True"], run.stderr)

    def test_a_child_started_inside_registered_work_stays_registered_after_it(self):
        # doit's threads share one environment, so a suite one thread starts can inherit another thread's
        # registration, which may end first
        child = "import sys\nfrom tools import locks\nwith locks.heavy('a suite'):\n    print('in', flush=True)\n" \
                "    sys.stdin.read()"
        with locks.heavy("a sibling stage"):
            self.hold("quiet", 30.0)
            suite = subprocess.Popen([sys.executable, "-c", child], env={**self.env, **os.environ}, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)
            self.addCleanup(suite.wait)
            self.addCleanup(suite.stdin.close)
            started = time.monotonic()
            self.assertEqual(suite.stdout.readline().strip(), "in")
            self.assertLess(time.monotonic() - started, 20.0)        # it did not wait out the 30 s quiet lock
        self.assertEqual(len(locks.heavy_work()), 1)                  # the sibling has ended; the suite still counts

    def test_one_registration_covers_a_process_threads_until_the_last_ends(self):
        # doit runs stages in threads, and S1 builds four families at once: each build must stay registered
        first_in, first_out, second_in = threading.Event(), threading.Event(), threading.Event()
        seen = []

        def first():
            with locks.heavy("a family"):
                first_in.set()
                second_in.wait(10)
            first_out.set()

        def second():
            first_in.wait(10)
            with locks.heavy("another family"):
                second_in.set()
                first_out.wait(10)
                seen.append(len(locks.heavy_work()))         # the first family's block has ended; this one runs on
        threads = [threading.Thread(target=f) for f in (first, second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        self.assertEqual(seen, [1])
        self.assertEqual(locks.heavy_work(), [])

    def test_a_timing_run_sees_a_child_that_registered_while_it_looked(self):
        # the child writes its entry after the poll lists the folder, and the parent removes its own before the poll
        # reads it: that poll finds nothing running
        place = locks.folder("heavy")
        parent, child = place / "parent.json", place / "child.json"
        parent.write_text(json.dumps(locks.me("a suite")), encoding="utf-8")
        read = locks.read

        def racing(path):
            if path == parent and parent.exists():
                child.write_text(json.dumps(locks.me("a build inside it")), encoding="utf-8")
                parent.unlink()
            return read(path)
        said = []

        def say(message):
            said.append(message)
            child.unlink()                                  # the child's work ends once the timing run has seen it
        with mock.patch.object(locks, "read", racing), locks.quiet("timing", say=say, poll_s=0):
            pass
        self.assertIn("a build inside it", " ".join(said))

    def test_an_entry_being_read_as_its_work_ends_is_still_removed(self):
        # Windows refuses to delete a file anyone has open, and every timing run's poll reads each entry
        with locks.heavy("a build"):
            entry = Path(os.environ[locks.HEAVY])
            reader = open(entry, encoding="utf-8")
            closer = threading.Timer(0.5, reader.close)
            closer.start()
            self.addCleanup(closer.join)
        self.assertFalse(entry.exists())

    @unittest.skipUnless(os.name == "nt", "only Windows refuses to delete a file that is open")
    def test_an_entry_left_behind_is_removed_once_its_reader_lets_go(self):
        # until then other processes count the ended work as running: a timing run elsewhere waits on it while holding
        # the quiet lock, which this process's next heavy work waits for, so nothing else this process does may be
        # needed to remove it
        with mock.patch.object(locks, "REMOVE_ATTEMPTS", 1):
            with locks.heavy("a stage", say=lambda message: None):
                entry = Path(os.environ[locks.HEAVY])
                reader = open(entry, encoding="utf-8")
        self.addCleanup(reader.close)
        self.assertTrue(entry.exists())
        reader.close()
        self.assertTrue(gone(entry))

    def test_this_process_never_waits_on_its_own_ended_work(self):
        # a doit run that ends an S3 stage whose entry Windows would not let it remove, then starts S6
        with mock.patch.object(locks, "REMOVE_ATTEMPTS", 1):
            with locks.heavy("a stage", say=lambda message: None):
                reader = open(os.environ[locks.HEAVY], encoding="utf-8")
                self.addCleanup(reader.close)
            self.assertEqual(locks.heavy_work(), [])
            with locks.quiet("timing", say=self.fail, poll_s=0):
                pass

    def test_a_child_that_cannot_withdraw_its_entry_stops_instead_of_waiting_on_it(self):
        # the enclosing work ends between the child's write and its check, so the child must withdraw its entry before
        # it waits for the quiet lock. If Windows will not let it (a reader holds the entry open), a timing run waiting
        # on the entry and the child waiting on the timing run would wait for ever, so the child stops, and the entry is
        # removed once the reader lets go
        place, outer, readers, entries = locks.folder("heavy"), psutil.Process().parent(), [], []
        parent = place / "parent.json"
        parent.write_text(json.dumps({**locks.me("a suite"), "pid": outer.pid, "created": outer.create_time()}),
                          encoding="utf-8")
        self.addCleanup(lambda: [reader.close() for reader in readers])
        write = locks.write_atomic

        def written(path, data):
            write(path, data)
            entries.append(path)
            readers.append(open(path, encoding="utf-8"))
            parent.unlink()
        with mock.patch.dict(os.environ, {locks.HEAVY: str(parent)}), mock.patch.object(locks, "write_atomic", written), \
                mock.patch.object(locks, "REMOVE_ATTEMPTS", 2):
            with self.assertRaises(RuntimeError), locks.heavy("a build inside it"):
                pass
        for reader in readers:
            reader.close()
        self.assertTrue(gone(*entries))

    def test_a_wait_is_logged_once_the_lock_is_released(self):
        # the log measures how long work queues for each lock. A timing run's wait for heavy work to end is its longest,
        # so it is logged as a wait for "heavy work"
        hush = lambda message: None
        for kind, lock, holder, take in (
                ("nvidia", "gpu-nvidia", "the first run", lambda: locks.gpu("nvidia", "a second run", hush)),
                ("heavy", "heavy work", "a build", lambda: locks.quiet("a second run", hush, poll_s=0.2))):
            with self.subTest(lock=lock):
                locks.waits_file().unlink(missing_ok=True)
                self.hold(kind, 2.0)
                began = time.time()
                with take():
                    time.sleep(0.5)
                ended = time.time()
                records = [json.loads(line) for line in locks.waits_file().read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(records), 1, records)
                times = {key: records[0].pop(key) for key in ("start", "waited_s", "held_s")}
                self.assertEqual(records[0], {"lock": lock, "host": socket.gethostname(), "label": "a second run",
                                              "worktree": str(ROOT), "holder": holder, "holder_worktree": str(ROOT)})
                self.assertGreater(times["waited_s"], 1.0)
                self.assertGreaterEqual(times["held_s"], 0.5)
                self.assertLessEqual(began, times["start"] + 0.001)
                self.assertLessEqual(sum(times.values()), ended + 0.01)    # it began, waited and held within the block

    def test_a_lock_taken_at_once_logs_nothing(self):
        with locks.gpu_work("amd", "a tool", say=self.fail):
            pass
        with locks.quiet("timing", say=self.fail, poll_s=0):
            pass
        self.assertFalse(locks.waits_file().exists())

    def test_a_wait_that_cannot_be_logged_still_takes_and_frees_the_lock(self):
        # every worktree shares the log: neither a broken log nor another process appending to it may hold a lock up
        def wait_for_nvidia():
            self.hold("nvidia", 1.0)
            said = []
            with locks.gpu("nvidia", "a second run", say=said.append):
                left = time.monotonic()
            self.assertLess(time.monotonic() - left, 5.0)
            self.assertIn("could not log the wait for the gpu-nvidia lock", said[-1])
            with locks.gpu("nvidia", "a third run", say=self.fail):
                pass
        with mock.patch.object(locks, "LOG_PATIENCE_S", 0.5):
            with self.subTest(case="a folder in the log's place"):
                locks.waits_file().mkdir()
                wait_for_nvidia()
                locks.waits_file().rmdir()
            with self.subTest(case="another process appending to the log"):
                self.hold("log", 30.0)
                wait_for_nvidia()

    def test_appends_from_many_processes_at_once_stay_whole_lines(self):
        # every worktree appends to the one log, and Windows appends by seeking to the end and then writing, so two
        # appends at once could overwrite each other. Each writer's records have a length of their own
        writers, records = 8, 200
        append = ("import sys\nfrom tools import locks\nlocks.LOG_PATIENCE_S = 60.0\nn = int(sys.argv[1])\n"
                  "print('ready', flush=True)\nsys.stdin.read()\n"
                  f"for i in range({records}):\n"
                  "    locks.log_wait('gpu-amd', str(n) * (500 + 300 * n), [], 0.0, float(i), 0.0,\n"
                  "                   say=lambda message: print(message, file=sys.stderr))\n")
        processes = []
        for n in range(writers):
            process = subprocess.Popen([sys.executable, "-c", append, str(n)], env=self.env, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, text=True)
            self.addCleanup(lambda process=process: process.poll() is None and process.kill())
            processes.append(process)
        for process in processes:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            process.stdout.close()
        for process in processes:
            process.stdin.close()                                 # they all start appending at once
        self.assertEqual([process.wait(timeout=180) for process in processes], [0] * writers)
        lines, whole = locks.waits_file().read_text(encoding="utf-8").splitlines(), []
        for line in lines:
            with contextlib.suppress(ValueError):
                whole.append(json.loads(line))
        self.assertEqual(len(whole), len(lines), "some lines are not whole records")
        self.assertCountEqual([(len(record["label"]), record["waited_s"]) for record in whole],
                              [(500 + 300 * n, float(i)) for n in range(writers) for i in range(records)])

    def test_the_waits_command_sums_up_each_locks_waits(self):
        now = time.time()
        logged = [*({"lock": "gpu-amd", "start": now - 60, "waited_s": waited} for waited in (4, 1, 10, 3, 2)),
                  {"lock": "gpu-amd", "start": now - 7200, "waited_s": 1000.0},             # before --since
                  {"lock": "heavy work", "start": now - 60, "waited_s": 30.0}]
        locks.waits_file().write_text("".join(json.dumps(record) + "\n" for record in logged) + '{"lock": "gpu-am\n',
                                      encoding="utf-8")
        since = datetime.fromtimestamp(now - 3600).isoformat(timespec="seconds")
        run = subprocess.run([sys.executable, "-X", "utf8", "-m", "tools.locks", "waits", "--since", since], cwd=ROOT,
                             env=self.env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        rows = {row[0]: row[1:] for row in (line.rsplit(None, 5) for line in run.stdout.splitlines()) if len(row) == 6}
        self.assertEqual(rows["gpu-amd"], ["5", "20.0", "3.0", "7.6", "10.0"])     # waits, total, p50, p90, max
        self.assertEqual(rows["heavy work"], ["1", "30.0", "30.0", "30.0", "30.0"])
        self.assertIn("skipped 1 unreadable line", run.stdout)


class GpuOrderTests(unittest.TestCase):
    """Every GPU tool takes its GPU's lock before it registers heavy work (locks.gpu_work): one that registered first and
    then waited for a GPU could close a cycle with a timing run and that GPU's holder."""

    def order(self, call):
        seen, real_gpu, real_heavy = [], locks.gpu, locks.heavy

        @contextlib.contextmanager
        def gpu(name, label, say=print):
            seen.append(f"gpu {name}")
            with real_gpu(name, label, say):
                yield

        @contextlib.contextmanager
        def heavy(label, say=print, patience=None):
            seen.append("heavy")
            with real_heavy(label, say, patience):
                yield
        with tempfile.TemporaryDirectory(prefix="hog-locks-") as state, \
                mock.patch.dict(os.environ, {"HOG_STATE_DIR": state}), mock.patch.object(locks, "gpu", gpu), \
                mock.patch.object(locks, "heavy", heavy), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            call()
        return seen

    def test_each_gpu_tool_takes_its_gpu_before_it_registers_heavy_work(self):
        from tools import testrun
        from tools.mutate import run as mutation_run
        from tools.review import sheet
        from tools.site import parity
        from tools.timing import windows
        folder = tempfile.TemporaryDirectory(prefix="hog-order-")
        self.addCleanup(folder.cleanup)
        page, line = Path(folder.name) / "page.html", np.stack([np.arange(0.0, 400.0, 1.0)] + [np.zeros(400)] * 2, 1)
        page.write_text("a page", encoding="utf-8")

        def suite():
            with mock.patch.dict(os.environ), mock.patch.object(testrun, "LOGS", Path(folder.name)), \
                    mock.patch.object(testrun, "load", return_value=(unittest.TestSuite(), [], [])):
                testrun.run("render", gpu="amd", pattern="none")

        def review_sheet():
            with mock.patch.object(sheet, "make", mock.AsyncMock(return_value=page)):
                sheet.main(["tour", "--gpu", "amd"])

        def probe_windows():
            with mock.patch.object(windows, "tracks", mock.AsyncMock(return_value=((400.0, line), (400.0, line)))), \
                    mock.patch.object(windows, "WINDOWS", {"a": 10.0}):
                windows.main([str(page), str(page), "--gpu", "amd"])

        def mutation_check():
            m = mutation_run.Mutation("a door seen in the page", "x.py", "a", "b", ("test_render_x.Tests",), rebuild=False)
            with mock.patch.dict(os.environ, {"HOG_GPU": "amd"}), mock.patch.object(mutation_run, "check_anchors"), \
                    mock.patch.object(mutation_run, "run", return_value="pass"), \
                    mock.patch.object(mutation_run.importlib, "import_module", return_value=SimpleNamespace(MUTATIONS=[m])):
                mutation_run.main(["x"])

        def site_parity():
            with mock.patch.object(parity, "shots", mock.AsyncMock(return_value=[])):
                parity.main([str(Path(folder.name) / "out"), "--baseline", str(page), "--candidate", str(page),
                             "--gpu", "amd"])
        for name, call in (("tools.locks gpu", lambda: locks.main(["gpu", "amd", "--", sys.executable, "-c", "pass"])),
                           ("tools.testrun", suite), ("tools.review.sheet", review_sheet),
                           ("tools.timing.windows", probe_windows), ("tools.site.parity", site_parity),
                           ("tools.mutate.run", mutation_check)):
            with self.subTest(tool=name):
                self.assertEqual(self.order(call), ["gpu amd", "heavy"])


if __name__ == "__main__":
    unittest.main()
