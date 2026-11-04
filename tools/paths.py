"""Repository paths, and the machine-wide state folder that every worktree of this repository shares.

The state folder (locks, the stage ledger, caches and this machine's input locations) is derived at run time, so no
absolute path is committed: `HOG_STATE_DIR` overrides the per-user default from platformdirs."""
import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

import platformdirs

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BUILD = ROOT / "build"
LOGS = BUILD / "logs"
PAGE = ROOT / "hogwarts-flyover.html"
APP = "hogwarts-flyover"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def state_dir():
    path = Path(os.environ.get("HOG_STATE_DIR") or platformdirs.user_state_dir(APP, appauthor=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(*parts):
    path = state_dir().joinpath("cache", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def page_tag(page=PAGE):
    """The short page hash that names logs and review sheets."""
    return sha256(page)[:12]


REPLACE_ATTEMPTS = 50   # 0.1 s apart


def replace(source, target):
    """os.replace, retried for a few seconds while Windows refuses it because another process has the target open."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.1)


def write_atomic(path, data):
    """Write bytes so that readers see the old file or the new one, never a partial one. Windows refuses to replace a
    file while another process has it open, so the swap is retried for a few seconds before giving up."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
