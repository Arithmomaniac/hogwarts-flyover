"""The asset contact sheet the plan's qualification asks for: for every family, each inventory object's close-up on the
baseline page (baseline-x, its low-detail version), its close-up and its context view on the finished page, side by
side, with the manifest decisions it answers for, so the sheet compares the scene with both the baseline and the source
and design criteria.

    python -X utf8 -m tools.inspection.contact [--shots DIR] [--out DIR]

It reads the shot tool's records (build/inspection/shots/*.json) and writes build/inspection/contact-<family>.png. An
object the shot tool has not shot is left out, as are the tour's stills, which have the tour sheet (tools/review/tour.py).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont

from tools.inspection import inventory
from tools.inspection.review import shot_records

THUMB = (640, 360)
PAD, LABEL_H, HEAD_H, NOTE_W = 12, 30, 44, 560
COLUMNS = ("Baseline (baseline-x)", "Close-up", "Context")
ROLES = ("baseline", "close", "context")
BACKGROUND, INK, FAINT = (24, 24, 28), (235, 235, 235), (170, 170, 170)


class Row(NamedTuple):
    object: str
    title: str
    sample: str
    cells: tuple                # a file name or None under each of COLUMNS
    decisions: tuple


def rows(objects, records):
    """A Row for each shot object (not the tour's stills), in inventory order."""
    out = []
    for o in objects:
        record = records.get(o.id)
        if o.kind == "tour" or not record:
            continue
        files = {s["role"]: s["file"] for s in record["shots"]}
        out.append(Row(o.id, o.title, o.sample, tuple(files.get(role) for role in ROLES), tuple(o.decisions)))
    return out


def cell_origin(row, column):
    """The top-left pixel of a row's thumbnail under a column."""
    return PAD + column * (THUMB[0] + PAD), HEAD_H + row * (THUMB[1] + LABEL_H + PAD) + LABEL_H


def compose(out, family, sheet_rows, folder, decisions):
    """Draw one family's sheet to `out`: its rows' thumbnails (from `folder`) under COLUMNS, each row labelled with
    its object, and beside it the decisions it answers for (decisions: {name: text})."""
    font = ImageFont.truetype("arial.ttf", 20)
    small = ImageFont.truetype("arial.ttf", 15)
    width = PAD + len(COLUMNS) * (THUMB[0] + PAD) + NOTE_W + PAD
    img = Image.new("RGB", (width, HEAD_H + len(sheet_rows) * (THUMB[1] + LABEL_H + PAD) + PAD), BACKGROUND)
    draw = ImageDraw.Draw(img)
    for c, title in enumerate(COLUMNS):
        draw.text((cell_origin(0, c)[0], 10), f"{family}: {title}", fill=INK, font=font)
    for r, row in enumerate(sheet_rows):
        x0, y0 = cell_origin(r, 0)
        draw.text((x0, y0 - LABEL_H + 4), f"{row.object}: {row.title}" + (f"  ({row.sample})" if row.sample else ""),
                  fill=INK, font=font)
        for c, name in enumerate(row.cells):
            x, y = cell_origin(r, c)
            if name is None:
                draw.text((x + 8, y + 8), "(no baseline shot)" if c == 0 else "(not shot)", fill=FAINT, font=font)
                continue
            img.paste(Image.open(Path(folder) / name).convert("RGB").resize(THUMB, Image.LANCZOS), (x, y))
        x, y = cell_origin(r, len(COLUMNS))
        for name in row.decisions:
            for line in _wrap(f"{name}: {decisions.get(name, '')}", small, NOTE_W - 8, draw):
                if y > cell_origin(r, 0)[1] + THUMB[1] - 18:
                    break
                draw.text((x, y), line, fill=FAINT, font=small)
                y += 18
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def _wrap(text, font, width, draw):
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if draw.textlength(trial, font=font) > width and line:
            lines.append(line)
            line = w
        else:
            line = trial
    return lines + ([line] if line else [])


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots", type=Path, default=inventory.OUT_DIR / "shots")
    parser.add_argument("--out", type=Path, default=inventory.OUT_DIR)
    args = parser.parse_args(argv)
    objects = inventory.load(args.shots.parent / "inventory.json")
    manifest = json.loads(inventory.MANIFEST.read_text(encoding="utf-8"))
    texts = {k: d["text"] for k, d in manifest["decisions"].items()}
    all_rows = rows(objects, shot_records(args.shots))
    for family in dict.fromkeys(o.family for o in objects if o.kind != "tour"):
        family_rows = [r for r in all_rows if r.object.split("/")[0] == family]
        if family_rows:
            print("wrote", compose(args.out / f"contact-{family}.png", family, family_rows, args.shots, texts))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
