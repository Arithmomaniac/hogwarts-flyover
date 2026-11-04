"""One family's payload part, built on the registered world raised by SHIFT metres: stage S1's part for the asset tests'
shifted runs, which reuse it until the shift or a source it reads changes (tools/stages.py, s1_stage).

    python -X utf8 -m tools.shifted_part FAMILY SHIFT OUT
"""
import json
from pathlib import Path
import sys

from tools.paths import SRC  # noqa: F401  (puts src/ on the path)
import build_meshes  # noqa: E402
from datum_shift import shifted  # noqa: E402
from mesh_source import load_data  # noqa: E402


def main(argv):
    family, shift, out = argv[0], float(argv[1]), Path(argv[2])
    part = build_meshes.encode_part(family, data=shifted(load_data(), shift))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(part, separators=(",", ":")), encoding="ascii", newline="\n")
    print(f"wrote {out} ({len(part['assets'])} assets, on the world raised {shift:+g} m)")


if __name__ == "__main__":
    main(sys.argv[1:])
