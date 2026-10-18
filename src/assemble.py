"""Assemble the single-file flyover: template + shader parts + packed data."""
import os, sys

SRC = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(SRC)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, "hogwarts-flyover.html")


def read(*p):
    return open(os.path.join(*p), encoding="utf-8").read()


frag = "".join(read(SRC, f"frag{i}.glsl") for i in range(1, 6))
assert "</script" not in frag
html = read(SRC, "template.html")
html = html.replace("/*__FRAG__*/", frag, 1)
html = html.replace("<!--__PART_B__-->", read(SRC, "template_b.html") + read(SRC, "template_c.html"), 1)
data = read(WORK, "build", "data.js")
assert "</script" not in data
html = html.replace("/*__DATA__*/", data.strip(), 1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8", newline="\n").write(html)
print("wrote", OUT, f"{len(html) / 1e6:.2f} MB")
