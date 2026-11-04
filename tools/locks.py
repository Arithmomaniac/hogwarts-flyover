"""Machine-wide locks, shared by every worktree through the state folder: one per GPU, and the quiet lock for timing.

A browser suite holds its adapter's lock, so a second run on the same GPU waits rather than overlapping it; the two GPUs'
correctness suites may run at the same time. Heavy work (builds, test suites, mutation checks, Blender) registers while
it runs. A timing run holds the quiet lock: it waits until no heavy work is registered and keeps new heavy work waiting
until it ends, and the timing scripts refuse to start without the token that holding the lock hands to the command it
runs. Work started inside registered work never waits for the quiet lock, which would deadlock with a timing run that
holds it while waiting for the enclosing work: a process's threads share one registration, and a child process
registers its own entry at once, so it stays counted if the enclosing work ends first.

    python -X utf8 -m tools.locks status
    python -X utf8 -m tools.locks quiet -- COMMAND ...      # run a timing command alone on the machine
    python -X utf8 -m tools.locks heavy -- COMMAND ...      # run an ad-hoc heavy command, which timing runs wait for
    python -X utf8 -m tools.locks gpu amd|nvidia -- COMMAND ...    # the same on one GPU: its lock first, then heavy work,
                                                                   # with HOG_GPU set, so a browser script queues behind
                                                                   # a suite on that GPU instead of slowing it
"""
import contextlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
import uuid

import psutil
from filelock import FileLock, Timeout

from tools.paths import ROOT, state_dir, write_atomic

GPUS = ("amd", "nvidia")
TOKEN = "HOG_QUIET_TOKEN"
HEAVY = "HOG_HEAVY"                  # the registration enclosing this process's work, which its children inherit
REMOVE_ATTEMPTS = 50                 # 0.1 s apart: Windows refuses to delete a file while another process reads it
QUIET_PATIENCE_S = 2.0               # a GPU tool lets its GPU go only if the quiet lock stays held this long: every
                                     # registration holds it for a moment, a timing run for minutes
_leftover = set()                    # this process's ended registrations that Windows has not yet let it remove
_leftover_lock = threading.Lock()


def folder(*parts):
    path = state_dir().joinpath("locks", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def me(label):
    p = psutil.Process()
    return {"pid": p.pid, "created": p.create_time(), "label": label, "worktree": str(ROOT), "since": time.time()}


def alive(holder):
    try:
        return abs(psutil.Process(holder["pid"]).create_time() - holder["created"]) < 1e-3
    except (psutil.Error, KeyError, TypeError):
        return False


def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def remove(path):
    """Delete `path`, retried for a few seconds while Windows refuses because another process has it open (every poll
    of heavy_work reads each entry); True once it is gone."""
    for attempt in range(REMOVE_ATTEMPTS):
        if attempt:
            time.sleep(0.1)
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError:
            pass
    return False


def leave(entry):
    """Hand this process's ended heavy-work `entry`, which remove() could not delete, to a thread that tries again once
    a second for as long as the process runs. Until it is gone other processes count the work as running, and a timing
    run waiting for it holds the quiet lock that this process's next heavy work waits for, so its removal must not
    depend on anything else the process does. This process's own polls ignore it meanwhile. Entry names are never
    reused, so the thread cannot delete a later registration."""
    with _leftover_lock:
        if not _leftover:
            threading.Thread(target=_sweep, name="heavy-work sweeper", daemon=True).start()
        _leftover.add(entry)


def _sweep():
    while True:
        time.sleep(1.0)
        with _leftover_lock:
            for entry in list(_leftover):
                with contextlib.suppress(OSError):
                    entry.unlink(missing_ok=True)
                    _leftover.discard(entry)
            if not _leftover:
                return


def describe(holder):
    if not holder:
        return "an unknown process"
    return f"{holder['label']} (pid {holder['pid']}, {holder['worktree']}, for {time.time() - holder['since']:.0f} s)"


class Busy(Exception):
    """A lock that held() gave up on after its `patience`."""


@contextlib.contextmanager
def held(name, label, say=print, patience=None):
    """Hold lock `name` for the whole block, saying who holds it while waiting; with a `patience` in seconds, raises
    Busy if the lock is not free within it."""
    lock, info = FileLock(str(folder() / f"{name}.lock")), folder() / f"{name}.json"
    try:
        lock.acquire(timeout=0)
    except Timeout:
        if patience is not None:
            try:
                lock.acquire(timeout=patience)
            except Timeout:
                raise Busy(name) from None
        else:
            say(f"waiting for the {name} lock, held by {describe(read(info))}")
            lock.acquire()
    try:
        write_atomic(info, json.dumps(me(label)).encode("utf-8"))
        yield
    finally:
        remove(info)
        lock.release()


@contextlib.contextmanager
def gpu(name, label, say=print):
    """The lock of one GPU (amd or nvidia). A process whose ancestor holds it works under the ancestor's hold, so a tool
    that takes the lock itself runs under `tools.locks gpu GPU -- COMMAND` instead of waiting for its own parent."""
    if name not in GPUS:
        raise ValueError(f"unknown GPU {name!r}; the GPUs are {GPUS}")
    if ancestor(read(folder() / f"gpu-{name}.json")):
        yield
        return
    with held(f"gpu-{name}", label, say):
        yield


def ancestor(holder):
    """Whether lock holder `holder` (a lock's info) is a live ancestor of this process."""
    if not holder:
        return False
    try:
        return any(p.pid == holder.get("pid") and abs(p.create_time() - holder.get("created", 0)) < 1e-3
                   for p in psutil.Process().parents())
    except psutil.Error:
        return False


@contextlib.contextmanager
def gpu_work(name, label, say=print):
    """Browser work on GPU `name`: its lock, then a heavy-work registration. Waiting for the GPU registers nothing, so it
    never holds up a timing run. And it never holds the GPU while it waits for a timing run: if one holds the quiet lock
    when the GPU comes free, it lets the GPU go, waits for the timing run to end and starts again. Holding the GPU there
    could close a cycle with the timing run and work that registered first and then waited for this GPU (a worktree's
    older tools, or a GPU tool run inside registered work), since the timing run waits for that work."""
    while True:
        with contextlib.ExitStack() as hold:
            hold.enter_context(gpu(name, label, say))
            try:
                hold.enter_context(heavy(label, say, patience=QUIET_PATIENCE_S))
            except Busy:
                holder = describe(read(folder() / "quiet.json"))
                say(f"{holder} holds the quiet lock: {label} lets the {name} GPU go until it ends")
            else:
                yield
                return
        with held("quiet", f"waiting to start {label}", say):
            pass


def heavy_work():
    """The registered heavy work that is still running; stale registrations are removed. A child of registered work
    writes its entry before it checks that its parent's is still there (see covered), so one listing can miss the
    child's entry and then find the parent's gone: the folder is listed again until a listing shows no entry that the
    one before it did not."""
    place, seen = folder("heavy"), None
    while True:
        listed = set(place.glob("*.json"))
        if seen is not None and listed <= seen:
            return []
        if running := [holder for holder in map(running_holder, sorted(listed)) if holder]:
            return running
        seen = listed


def running_holder(path):
    """The holder of heavy-work entry `path` while its work runs, else None; a stale entry is removed."""
    if path in _leftover:                            # this process's own ended work, which leave() is removing
        return None
    holder = read(path)
    if holder is not None and alive(holder):
        return holder
    try:
        stale = holder is not None or time.time() - path.stat().st_mtime > 60
    except FileNotFoundError:                        # its work ended while we looked
        return None
    if stale:
        with contextlib.suppress(OSError):
            path.unlink()
    return None


def covered(place, entry, label):
    """Writes `entry` without waiting for the quiet lock while an ancestor process's registered work, in `place`, still
    runs: a timing run may then be waiting for that work, and waiting for it in turn would deadlock. The entry is
    written before the ancestor's is checked again, so heavy work is never left unregistered between the two. False,
    with nothing written, if no ancestor's work covers it."""
    outer = os.environ.get(HEAVY)
    if not outer or Path(outer).parent != place:
        return False
    holder = read(Path(outer))
    if holder is None or holder.get("pid") == os.getpid() or not alive(holder):
        return False
    write_atomic(entry, json.dumps(me(label)).encode("utf-8"))
    if Path(outer).is_file():
        return True
    if not remove(entry):
        leave(entry)
        raise RuntimeError(f"could not withdraw {entry} once the work enclosing {label} had ended: a timing run that "
                           "waits for it would wait for ever")
    return False


_registered = threading.Lock()
_open = {}                           # this process's registration per heavy folder: [open blocks, entry, outer HOG_HEAVY]


@contextlib.contextmanager
def heavy(label, say=print, patience=None):
    """Register heavy work for the block; while a timing run holds the quiet lock this waits for it to end, or, with a
    `patience` in seconds, raises Busy once that has passed. One registration covers all of a process's open blocks, in
    every thread, until the last of them ends. Work in a child of registered work registers its own entry at once (see
    covered), so it stays counted if its parent's ends first."""
    place = folder("heavy")
    with _registered:
        mine = _open.get(place)
        if mine is None:
            entry = place / f"{os.getpid()}-{uuid.uuid4().hex}.json"
            if not covered(place, entry, label):
                with held("quiet", f"registering {label}", say, patience):
                    write_atomic(entry, json.dumps(me(label)).encode("utf-8"))
            mine = _open[place] = [0, entry, os.environ.get(HEAVY)]
            os.environ[HEAVY] = str(entry)
        mine[0] += 1
    try:
        yield
    finally:
        with _registered:
            mine[0] -= 1
            if not mine[0]:
                del _open[place]
                if mine[2] is None:
                    os.environ.pop(HEAVY, None)
                else:
                    os.environ[HEAVY] = mine[2]
                if not remove(mine[1]):
                    leave(mine[1])
                    say(f"could not remove {mine[1]} yet: other processes count this process's heavy work as running "
                        "until a background thread manages to")


@contextlib.contextmanager
def quiet(label, say=print, poll_s=5.0):
    """The quiet lock: no heavy work runs on the machine during the block. Yields the token timing scripts check."""
    with held("quiet", label, say):
        while busy := heavy_work():
            say("waiting for heavy work to finish: " + "; ".join(describe(h) for h in busy))
            time.sleep(poll_s)
        token = secrets.token_hex(16)
        write_atomic(folder() / "quiet-token.json", json.dumps({**me(label), "token": token}).encode("utf-8"))
        try:
            yield token
        finally:
            remove(folder() / "quiet-token.json")


def require_quiet(what):
    """Refuse a timing run that does not hold the quiet lock: exit status 3, which the stage runner reads as blocked."""
    holder = read(folder() / "quiet-token.json")
    if not (holder and alive(holder) and secrets.compare_digest(holder.get("token", ""), os.environ.get(TOKEN, ""))):
        print(f"blocked: {what} times frames, so it needs the machine to itself: run it under the quiet lock, as "
              f"`python -X utf8 -m tools.locks quiet -- python -X utf8 {' '.join(sys.argv)}`, or through doit",
              file=sys.stderr)
        sys.exit(3)


def status():
    for name in ("quiet", *(f"gpu-{g}" for g in GPUS)):
        holder = read(folder() / f"{name}.json")
        print(f"{name:12s} {'held by ' + describe(holder) if holder and alive(holder) else 'free'}")
    work = heavy_work()
    print(f"heavy work   {'; '.join(describe(h) for h in work) if work else 'none'}")


def main(argv):
    if argv[:1] == ["status"]:
        status()
        return 0
    say = lambda m: print(m, file=sys.stderr)
    if argv[:1] == ["gpu"] and argv[1:2] and argv[1] in GPUS and argv[2:3] == ["--"] and argv[3:]:
        command = argv[3:]
        with gpu_work(argv[1], " ".join(command), say=say):
            return subprocess.run(command, env={**os.environ, "HOG_GPU": argv[1]}).returncode
    if argv[:1] in (["quiet"], ["heavy"]) and "--" in argv and argv.index("--") + 1 < len(argv):
        command = argv[argv.index("--") + 1:]
        if argv[0] == "heavy":
            with heavy(" ".join(command), say=say):
                return subprocess.run(command).returncode
        with quiet(" ".join(command), say=say) as token:
            return subprocess.run(command, env={**os.environ, TOKEN: token}).returncode
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
