"""External inputs that stay out of Git, pinned by hash in tools/inputs.lock.json and located per machine.

A stage that needs an input calls `require(name)`. The path comes from the environment variable HOG_INPUT_<NAME>,
then from this machine's <state>/inputs.json, then from build/inputs/<file>. A missing input or a hash mismatch raises
Blocked, which the stage runner reports as blocked: never a pass and never cached.

    python -X utf8 -m tools.inputs status             # where each input is and whether it verifies
    python -X utf8 -m tools.inputs set NAME PATH      # record this machine's copy of NAME
    python -X utf8 -m tools.inputs tree NAME DIR      # (maintenance) rewrite a tree input's file manifest from DIR
"""
import json
import os
import sys
from pathlib import Path

from filelock import FileLock

from tools.paths import BUILD, sha256, state_dir, write_atomic

LOCK = Path(__file__).with_name("inputs.lock.json")


class Blocked(RuntimeError):
    """A prerequisite is unavailable, so the stage cannot run; that is never a pass."""


def entries():
    return json.loads(LOCK.read_text(encoding="utf-8"))


def spec(name):
    try:
        return entries()[name]
    except KeyError:
        raise ValueError(f"unknown input {name!r}; the inputs are {sorted(entries())}") from None


def machine_file():
    return state_dir() / "inputs.json"


def machine():
    path = machine_file()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def env_name(name):
    return "HOG_INPUT_" + name.upper().replace("-", "_")


def locate(name):
    """This machine's copy of the input, or None; existence only, not verified."""
    s = spec(name)
    for candidate in (os.environ.get(env_name(name)), machine().get(name), BUILD / "inputs" / s.get("file", name)):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def _cache_file():
    return state_dir() / "cache" / "verified.json"


class HashCache:
    """sha256 of files, remembered per path, size and modification time in <state>/cache/verified.json, which every
    worktree shares: read once, merged and written once under a lock, replaced atomically. An unreadable cache counts
    as empty, since it only saves hashing."""

    def __enter__(self):
        self.known, self.found = _read_json(_cache_file()), {}
        return self

    def sha256(self, path):
        st = path.stat()
        key = str(path.resolve())
        hit = self.found.get(key) or self.known.get(key)
        if hit and hit["size"] == st.st_size and hit["mtime_ns"] == st.st_mtime_ns:
            return hit["sha256"]
        self.found[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256(path)}
        return self.found[key]["sha256"]

    def __exit__(self, *exc):
        if self.found:
            target = _cache_file()
            try:
                with FileLock(str(target) + ".lock"):
                    merged = {**_read_json(target), **self.found}
                    write_atomic(target, json.dumps(merged, indent=0, sort_keys=True).encode("utf-8"))
            except OSError as e:
                print(f"note: the hash cache was not saved ({e}); the next check hashes again", file=sys.stderr)
        return False


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def tree_manifest(name):
    return json.loads((LOCK.parent / spec(name)["manifest"]).read_text(encoding="utf-8"))


def problems(name, path):
    """Why the copy at `path` does not match the lock (empty when it does)."""
    s = spec(name)
    if s["kind"] == "dir":
        return [] if path.is_dir() else [f"{path} is not a folder"]
    with HashCache() as hashes:
        if s["kind"] == "file":
            if not path.is_file():
                return [f"{path} is not a file"]
            if path.stat().st_size != s["size"]:
                return [f"{path} has {path.stat().st_size} bytes, the lock expects {s['size']}"]
            digest = hashes.sha256(path)
            return [] if digest == s["sha256"] else [f"{path} hashes to {digest[:12]}…, the lock expects {s['sha256'][:12]}…"]
        out = []
        for rel, (size, digest) in tree_manifest(name).items():
            f = path / rel
            if not f.is_file():
                out.append(f"missing {rel}")
            elif f.stat().st_size != size or hashes.sha256(f) != digest:
                out.append(f"changed {rel}")
        return out


def require(name):
    """The verified path of an input; Blocked when it is missing or differs from the lock."""
    s = spec(name)
    path = locate(name)
    if path is None:
        raise Blocked(f"input {name} is not available ({s['about']}). {s['obtain']} Then record it with "
                      f"`python -X utf8 -m tools.inputs set {name} PATH` or set {env_name(name)}.")
    found = problems(name, path)
    if found:
        raise Blocked(f"input {name} at {path} does not match the lock: " + "; ".join(found[:5])
                      + (f" (and {len(found) - 5} more)" if len(found) > 5 else ""))
    return path


def write_tree(name, directory):
    s = spec(name)
    root = Path(directory)
    files = {}
    for sub in s["include"]:
        target = root / sub
        paths = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
        for p in paths:
            files[p.relative_to(root).as_posix()] = [p.stat().st_size, sha256(p)]
    (LOCK.parent / s["manifest"]).write_text(json.dumps(dict(sorted(files.items())), indent=1) + "\n",
                                              encoding="utf-8", newline="\n")
    return len(files)


def main(argv):
    if not argv or argv[0] == "status":
        status = 0
        for name, s in entries().items():
            path = locate(name)
            if path is None:
                print(f"{name:26} missing   ({s['about']})")
                status = 1
                continue
            found = problems(name, path)
            print(f"{name:26} {'ok' if not found else 'MISMATCH':9} {path}" + ("" if not found else f"  {found[0]}"))
            status |= bool(found)
        return status
    if argv[0] == "set" and len(argv) == 3:
        name, path = argv[1], Path(argv[2]).resolve()
        spec(name)
        with FileLock(str(machine_file()) + ".lock"):
            config = machine()
            config[name] = str(path)
            write_atomic(machine_file(), (json.dumps(config, indent=1, sort_keys=True) + "\n").encode("utf-8"))
        found = problems(name, path) if path.exists() else [f"{path} does not exist"]
        print(f"{name} -> {path}: " + ("verified" if not found else "; ".join(found[:3])))
        return 1 if found else 0
    if argv[0] == "tree" and len(argv) == 3:
        print(f"{write_tree(argv[1], argv[2])} files written to the {argv[1]} manifest")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
