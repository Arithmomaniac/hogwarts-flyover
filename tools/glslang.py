"""Khronos glslang, the reference GLSL compiler, for the CPU shader check.

glslang has no PyPI package, so the official Windows release is downloaded from GitHub once, checked against its
published SHA256 and unpacked into the machine cache (outside Git):

    python -X utf8 -m tools.glslang          # fetch if needed and print the executable's path and version
"""
import subprocess
import sys

from tools.inputs import Blocked
from tools.paths import cache_dir

VERSION = "16.6.0"
URL = f"https://github.com/KhronosGroup/glslang/releases/download/{VERSION}/glslang-{VERSION}-windows-x86_64-release.zip"
SHA256 = "82bf434e69b9bb4829de7e2b4bc2c5e7a7861e53d66cf75e5cc70f5f694a8d9b"


def executable(fetch=True):
    """Path of glslang.exe, downloading the pinned release when `fetch` and it is not cached; Blocked otherwise."""
    folder = cache_dir("tools", f"glslang-{VERSION}")
    found = sorted(folder.rglob("glslang.exe"))
    if found:
        return found[0]
    if not fetch:
        raise Blocked(f"glslang {VERSION} is not in the cache; run `python -X utf8 -m tools.glslang`")
    import pooch

    try:
        pooch.retrieve(URL, known_hash=f"sha256:{SHA256}", fname=f"glslang-{VERSION}.zip", path=cache_dir("tools"),
                       processor=pooch.Unzip(extract_dir=folder.name), progressbar=False)
    except Exception as e:                     # network or hash failure: the check cannot run
        raise Blocked(f"could not fetch glslang {VERSION}: {e}") from e
    found = sorted(folder.rglob("glslang.exe"))
    if not found:
        raise Blocked(f"the glslang {VERSION} archive holds no glslang.exe")
    return found[0]


def version(exe=None):
    out = subprocess.run([str(exe or executable()), "--version"], capture_output=True, text=True, check=True).stdout
    return out.splitlines()[0].strip()


if __name__ == "__main__":
    exe = executable()
    print(exe)
    print(version(exe))
    sys.exit(0)
