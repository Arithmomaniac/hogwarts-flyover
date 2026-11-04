"""A local cache of White Hound's pages, text extracts, map images and reference photographs, kept out of Git.

tools/whcache.lock.json pins every file by SHA256. Each file comes from the first source that yields the pinned bytes:
the cache itself, the configured mirror (input `whitehound-mirror`), White Hound's live site, then the Wayback
capture of 2026-10-08. The text extracts and the image index have no web source (their extraction was not preserved),
so a machine without the mirror cannot rebuild the manifest; the runner then reports blocked.

    python -X utf8 -m tools.whcache fetch [--references]   # fill the cache (reference photographs on request)
    python -X utf8 -m tools.whcache lock                   # (maintenance) rewrite the lock from the mirror
"""
import hashlib
import http.client
import json
import sys
import time
import urllib.request
from pathlib import Path

from tools import inputs
from tools.inputs import Blocked, HashCache
from tools.paths import cache_dir, sha256, write_atomic

LOCK = Path(__file__).with_name("whcache.lock.json")
LIVE = "http://members.madasafish.com/~cj_whitehound/"
WAYBACK = "https://web.archive.org/web/20261008141630id_/http://members.madasafish.com/~cj_whitehound/"
CORE = ("raw", "txt", "img", "legend", "whref-index.json")      # what the manifest tools read
MIRROR = {"raw": "hogwarts/raw", "txt": "hogwarts/txt", "img": "hogwarts/img", "legend": "hogwarts/legend",
          "whref": "whref"}


def root():
    return cache_dir("whitehound")


def entries():
    return json.loads(LOCK.read_text(encoding="utf-8"))["files"]


def is_core(rel):
    return rel.split("/", 1)[0] in CORE


def mirror_path(mirror, rel):
    top, _, rest = rel.partition("/")
    return mirror / (MIRROR[top] + "/" + rest if rest else top)


def _download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "hogwarts-flyover cache (low rate)"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _fill(rel, spec, mirror):
    """Write the pinned bytes of one file into the cache (atomically) from the first source that has them; the source,
    or None."""
    target = root() / rel
    if mirror is not None:
        candidate = mirror_path(mirror, rel)
        if candidate.is_file() and candidate.stat().st_size == spec["size"]:
            data = candidate.read_bytes()
            if hashlib.sha256(data).hexdigest() == spec["sha256"]:
                write_atomic(target, data)
                return "mirror"
    for base in (LIVE, WAYBACK) if spec.get("url") else ():
        try:
            data = _download(base + spec["url"])
        except (OSError, http.client.HTTPException):
            continue
        if hashlib.sha256(data).hexdigest() == spec["sha256"]:
            write_atomic(target, data)
            time.sleep(0.5)                    # a gentle rate for a personal site
            return "live" if base == LIVE else "wayback"
    return None


def _present(rel, spec, hashes):
    f = root() / rel
    return f.is_file() and f.stat().st_size == spec["size"] and hashes.sha256(f) == spec["sha256"]


def fetch(references=False, quiet=False):
    """Fill the cache; the relative paths that are still missing."""
    mirror = inputs.locate("whitehound-mirror")
    missing = []
    with HashCache() as hashes:
        for rel, spec in entries().items():
            if not (references or is_core(rel)) or _present(rel, spec, hashes):
                continue
            source = _fill(rel, spec, mirror)
            if not quiet:
                print(f"{rel}: {source or 'UNAVAILABLE'}")
            if source is None:
                missing.append(rel)
    return missing


def path(rel):
    """The cached file, verified against its pinned hash and fetched when absent or changed; Blocked when no source
    has the pinned bytes."""
    spec = entries().get(rel)
    if spec is None:
        raise KeyError(f"{rel} is not in tools/whcache.lock.json")
    with HashCache() as hashes:
        if _present(rel, spec, hashes):
            return root() / rel
    if _fill(rel, spec, inputs.locate("whitehound-mirror")) is None:
        raise Blocked(f"White Hound file {rel} is not cached and no source has it; configure the whitehound-mirror "
                      f"input (python -X utf8 -m tools.inputs set whitehound-mirror PATH)")
    return root() / rel


def url_of(rel, index):
    top, _, name = rel.partition("/")
    if top == "raw":
        page = name.replace("map_of_Hogwarts__", "map_of_Hogwarts/", 1)
        return "Fanfic/" + page
    if top in ("img", "legend"):
        return "Fanfic/map_of_Hogwarts/artwork/" + name
    if top == "whref" and name in index:
        src = index[name]
        return "Fanfic/" + src[3:] if src.startswith("../") else "Fanfic/map_of_Hogwarts/" + src
    return None


def write_lock():
    mirror = inputs.require("whitehound-mirror")
    index = {e["name"]: e["src"] for e in json.loads((mirror / "whref-index.json").read_text(encoding="utf-8"))}
    files = {}
    for top, sub in MIRROR.items():
        for f in sorted(p for p in (mirror / sub).iterdir() if p.is_file()):
            rel = f"{top}/{f.name}"
            files[rel] = {"size": f.stat().st_size, "sha256": sha256(f), "url": url_of(rel, index)}
    f = mirror / "whref-index.json"
    files["whref-index.json"] = {"size": f.stat().st_size, "sha256": sha256(f), "url": None}
    LOCK.write_text(json.dumps({"files": files}, indent=1) + "\n", encoding="utf-8", newline="\n")
    return len(files)


def main(argv):
    if argv[:1] == ["fetch"]:
        missing = fetch(references="--references" in argv)
        core = [m for m in missing if is_core(m)]
        print(f"cache {root()}: {len(missing)} files unavailable ({len(core)} needed by the manifest tools)")
        return 3 if core else 0                  # 3: blocked, as the stage runner reads it
    if argv[:1] == ["lock"]:
        print(f"{write_lock()} files pinned in {LOCK.name}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
