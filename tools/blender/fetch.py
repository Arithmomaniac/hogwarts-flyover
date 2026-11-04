"""Blender 5.2 LTS, for the families whose source assets are modelled in Blender and committed as exports under
src/assets/FAMILY/.

The Python environment cannot install Blender, so the official Windows x64 build is downloaded from blender.org once,
checked against its published SHA256 and unpacked into the machine cache (outside Git), where every worktree finds it:

    python -X utf8 -m tools.blender.fetch          # fetch if needed (about 400 MB) and print the path and version

A family's generator, tools/blender/FAMILY_*.py, runs in Blender's own Python through run(), always with the same
arguments: headless, factory settings, offline, on an empty user profile, from the repository root, and with exit code
1 on a Python exception. The script reads its own arguments after "--" in sys.argv:

    from tools.blender import fetch
    fetch.run("tools/blender/willow_tree.py", "src/assets/willow/tree.npz")

run() and the fetch register as heavy work (tools.locks), so a timing run waits for them to end; inside other
registered work, such as a test suite, they are already counted.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

from tools.inputs import Blocked
from tools.locks import heavy, held
from tools.paths import ROOT, cache_dir

VERSION = "5.2.3"
ARCHIVE = f"blender-{VERSION}-windows-x64.zip"
URL = f"https://download.blender.org/release/Blender{VERSION.rsplit('.', 1)[0]}/{ARCHIVE}"
SHA256 = "723f08a4401502561eb763cdeacf8745b31e5f3f154d287d3830d97cf75dc2fb"
ARGS = ("--background", "--factory-startup", "--offline-mode", "--python-exit-code", "1")
MARKER = "unpacked.txt"         # written last, holding blender.exe's path, once the build reports the pinned version


def home():
    return cache_dir("tools") / f"blender-{VERSION}"


def cached():
    """Path of the unpacked build's blender.exe, or None; never downloads."""
    marker = home() / MARKER
    if not marker.is_file():
        return None
    exe = home() / marker.read_text(encoding="utf-8").strip()
    return exe if exe.is_file() else None


def executable(fetch=True):
    """Path of blender.exe, downloading and unpacking the pinned build when `fetch` and it is not cached; Blocked
    otherwise."""
    exe = cached()
    if exe:
        return exe
    if not fetch:
        raise Blocked(f"Blender {VERSION} is not in the cache; run `python -X utf8 -m tools.blender.fetch`")
    with heavy(f"fetching Blender {VERSION}"), held("blender-fetch", f"fetching Blender {VERSION}"):
        return cached() or _fetch()             # one download for every worktree, registered before its lock is taken


def _fetch():
    import pooch

    target = home()
    shutil.rmtree(target, ignore_errors=True)          # what an interrupted unpacking left
    try:
        pooch.retrieve(URL, known_hash=f"sha256:{SHA256}", fname=ARCHIVE, path=cache_dir("tools"),
                       processor=pooch.Unzip(extract_dir=target.name), progressbar=False)
    except Exception as e:                             # network or hash failure: Blender cannot run
        raise Blocked(f"could not fetch Blender {VERSION}: {e}") from e
    found = sorted(target.rglob("blender.exe"))
    if not found:
        raise Blocked(f"the Blender {VERSION} archive holds no blender.exe")
    reported = version(found[0])
    if not re.fullmatch(rf"Blender {re.escape(VERSION)}( LTS)?", reported):
        raise Blocked(f"the unpacked build reports {reported!r}, not Blender {VERSION}")
    (target / MARKER).write_text(found[0].relative_to(target).as_posix(), encoding="utf-8")
    return found[0]


def version(exe=None):
    out = subprocess.run([str(exe or executable()), "--version"], capture_output=True, text=True, errors="replace",
                         check=True).stdout
    return next((line.strip() for line in out.splitlines() if re.match(r"Blender \d", line)), out.strip())


def command(script, *args, exe=None):
    return [str(exe or executable()), *ARGS, "--python", str(script), "--", *map(str, args)]


def run(script, *args, exe=None, check=True, **kwargs):
    """Run a script headless in Blender with the fixed arguments; other keyword arguments go to subprocess.run. The
    user's Blender settings, add-ons and BLENDER_* variables are left out: the profile is a new, empty folder."""
    with heavy(f"Blender {VERSION}: {script}"), \
            tempfile.TemporaryDirectory(prefix="hog-blender-", ignore_cleanup_errors=True) as profile:
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith("BLENDER_")}
        env["BLENDER_USER_RESOURCES"] = profile
        return subprocess.run(command(script, *args, exe=exe), cwd=ROOT, env=env, check=check, **kwargs)


if __name__ == "__main__":
    try:
        exe = executable()
    except Blocked as e:
        print(f"blocked: {e}", file=sys.stderr)
        sys.exit(3)
    print(exe)
    print(version(exe))
    sys.exit(0)
