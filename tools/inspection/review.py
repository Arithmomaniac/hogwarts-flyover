"""The inspection's reviewers: fresh-context agents, each given one batch of objects with only those objects' images,
manifest decisions and quotations, and the defect checklist; and the collation of their verdicts into findings for
each owning session.

    python -X utf8 -m tools.inspection.review packets [--only ID,FAMILY/,PREFIX*,...] [--most N] [--with ASSET,...]
                                                      [--out DIR]
    python -X utf8 -m tools.inspection.review collate VERDICTS ... [--partial] [--out FILE]

Both take --shots DIR and --inventory FILE. A batch holds about LEAST to MOST objects, of one owner where it can, so a
reviewer sees related objects together; `--only` and `--most` make packets for a re-review of some objects in smaller
batches. A packet is the reviewer's whole brief: it names each object, its kind and owner, its images (close-up, context
and, for a payload object, the baseline page's close-up), its decisions with each quotation verbatim, and the checklist;
it asks for one JSON verdict per object. `packets` gives the reviewer JPEG copies of the images (review_copy), and
refuses a blank shot (one colour: its camera stood inside a solid) other than the baseline page's, which it marks.
`collate` reads the verdict files in the order given, an object's later verdict replacing its earlier one, and refuses
a verdict it cannot route: an unknown object, severity or owner, or a defect without evidence in that object's own
images (cited by name, full path or review copy); unless `--partial`, every object needs a verdict.
"""
import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

from tools.inspection import inventory, shoot

LEAST, MOST = 6, 12                 # objects per reviewer ("about 8-12"; an even split of 13 gives 7 and 6)
SEVERITIES = ("blocker", "major", "minor", "cosmetic")
VERDICTS = ("pass", "defect")
BLANK_STD = 1.0                     # grey levels, at an eighth of the size: a frame this flat shows nothing
BLANK_NOTE = "(blank: its camera stands inside a solid on that page; judge without it)"
# every request a reviewer makes re-sends every image it has opened, so it is given JPEG copies no larger than a vision
# model reads: a few hundred KB each, not the shots' megabytes, which timed out the requests of a 30-image batch
REVIEW_EDGE, REVIEW_QUALITY = 1568, 90
CHECKS = ("evidence", "geometry", "shading", "lod", "plausibility", "tour distance", "baseline")
CHECKLIST = (
    "evidence: anything that directly contradicts the quoted book evidence (a White Hound conclusion may give way; "
    "the books' quoted words may not)",
    "geometry: holes, gaps, floating or buried parts, z-fighting, clipping, inverted faces, wrong scale",
    "shading: texture, shading and shadow faults; light leaks",
    "lod: level-of-detail or culling faults visible in the shot (pops, missing pieces, a coarse stand-in up close)",
    "plausibility: photorealism and plausibility, including the clearings rule: a clearing must be large enough to be "
    "plausible and not look like a cutout made for a line of sight",
    "tour distance: whether the object reads at the distance the tour shows it",
    "baseline: whether the object is no better than the baseline's low-detail version (the baseline image, where given)",
)
# what `packets --with ASSET` adds to the checklist, for a round about one asset in every view that shows it
WITH_CHECKS = {
    "forest-trees": "trees, wherever they show (evidence, geometry, shading, lod): each species' form and colours as "
                    "its decisions give them; trunks meeting the ground; crowns clear of walls, roofs, roads and "
                    "water; no seam or pop where the species meshes give way to the drawn crowns, or those to the "
                    "canopy sheet; the trees' shadows and reflections",
}


def with_assets(objects, manifest, assets):
    """The objects, each also answering for the decisions of `assets` that a picture shows (after its own), as a round
    about those assets reviews every view of them."""
    extra = [d for a in manifest["assets"] if a["id"] in assets for d in a.get("decisions") or ()
             if d not in inventory.NOT_VISUAL]
    return [dataclasses.replace(o, decisions=tuple(dict.fromkeys((*o.decisions, *extra)))) for o in objects]


def reviewable(objects, images):
    """The objects that have shots to review, in inventory order: a fix round re-shoots only the objects it touched."""
    return [o for o in objects if images.get(o.id)]


def blank(path):
    """Whether an image is one colour, near enough: its camera stood inside a solid (a crown, a hedge, a wall)."""
    from PIL import Image, ImageStat
    with Image.open(path) as im:
        return ImageStat.Stat(im.convert("L").reduce(8)).stddev[0] < BLANK_STD


def unusable(blanks, roles):
    """The blank shots ({file: role}) to frame again before review: all but the baseline page's, whose solids are not
    the page's, so a camera framed on the page may stand inside one there."""
    return sorted(name for name in blanks if roles.get(name) != "baseline")


def review_copy(src, folder):
    """A reviewer's copy of a shot in `folder`: a JPEG of the same stem, its longest edge REVIEW_EDGE."""
    from PIL import Image
    out = Path(folder) / (Path(src).stem + ".jpg")
    with Image.open(src) as im:
        copy = im.convert("RGB")
        copy.thumbnail((REVIEW_EDGE, REVIEW_EDGE), Image.LANCZOS)
        copy.save(out, quality=REVIEW_QUALITY)
    return out.name


def shot_name(cited):
    """The shot an evidence entry cites, by its file name, its full path, or its review copy's (the same stem)."""
    path = Path(str(cited))
    return path.stem + ".png" if path.name else ""


def batches(objects, least=LEAST, most=MOST):
    """The objects in reviewer batches of `least` to `most`: each owner's objects split evenly, in inventory order, its
    tour stills apart from its objects (a still answers for the tour's distance, not for one object), and a group too
    small to stand alone packed with other groups' leftovers."""
    by_owner = {}
    for o in objects:
        by_owner.setdefault((o.owner, o.kind == "tour"), []).append(o)
    out, left = [], []
    for group in by_owner.values():
        if len(group) < least:
            left.extend(group)
            continue
        k = math.ceil(len(group) / most)
        out.extend(_even(group, k))
    if left:
        k = max(1, math.ceil(len(left) / most))
        out.extend(_even(left, k))
    return out


def _even(items, k):
    size, extra = divmod(len(items), k)
    out, at = [], 0
    for i in range(k):
        n = size + (i < extra)
        out.append(items[at:at + n])
        at += n
    return out


def quoted(manifest, decision):
    """(its text, [each basis entry as the reviewer reads it]) of a manifest decision: a quotation verbatim, in double
    quotes, with its source; a user's choice or a data note as written."""
    d = manifest["decisions"][decision]
    basis = []
    for b in d.get("basis", []):
        if "quote" in b:
            basis.append(f'{b["source"]}: "{b["quote"]}"')
        else:
            basis.append(f'{b["source"]} ({"choice" if "choice" in b else "note"}): {b.get("choice") or b.get("note")}')
    return d["text"], basis


def packet(batch, manifest, images, folder=None, blank=(), checklist=CHECKLIST):
    """The brief for one reviewer: its objects, their images (images[object id]: file names, in `folder` when given;
    those in `blank` marked so), decisions and quotations, the checklist, and the verdict format."""
    lines = [
        "# Visual inspection batch",
        "",
        "You are reviewing rendered images of objects in an offline WebGL flyover of Hogwarts, drawn from White Hound's "
        "interpretation of the books. Judge only what the images show, against the decisions and quotations given "
        "for each object. Each object has a close-up and a context view (the final displayed 1920x1080 frames, at "
        "internal scale 0.5, as 1568x882 JPEG copies) and, where given, the baseline page's close-up for comparison.",
        "",
        "## Checklist (apply every item to every object)",
        "",
        *[f"- {c}" for c in checklist],
        "",
        "## Objects",
    ]
    for o in batch:
        lines += ["", f"### {o.id}: {o.title}", "",
                  f"- Kind: {o.kind}; family: {o.family}; owner: {o.owner}"]
        if o.sample:
            lines.append(f"- Sampled: {o.sample}")
        if o.note:
            lines.append(f"- Note: {o.note}")
        lines.append("- Images: " + ", ".join((str(Path(folder) / name) if folder else name) +
                                              (f" {BLANK_NOTE}" if name in blank else "")
                                              for name in images.get(o.id, [])))
        for name in o.decisions:
            text, basis = quoted(manifest, name)
            lines.append(f"- Decision `{name}`: {text}")
            lines += [f"  - {b}" for b in basis]
    lines += [
        "",
        "## Verdicts",
        "",
        "Give each object one verdict, in this JSON format (your brief says where to put them):",
        "",
        '`{"object": ID, "verdict": "pass" or "defect", "severity": "none" for a pass, else one of '
        + ", ".join(f'"{s}"' for s in SEVERITIES) + ', "checks": the checklist items it fails (' +
        ", ".join(f'"{c}"' for c in CHECKS) + '), "evidence": [{"image": the image\'s file name or path, "region": '
        'where in it}], "finding": what is wrong, in one or two sentences, "owner": the session to fix it (one of '
        + ", ".join(f'"{s}"' for s in sorted(inventory.SESSIONS)) + ')}`',
        "",
        "Severity: blocker, wrong or broken at the tour's distance; major, plainly visible close up; minor, visible "
        "only on inspection; cosmetic, a taste call. Suggest the owner given for the object unless the fault lies "
        "elsewhere (shadows, reflections, glass and levels of detail are integration's; the mesh trees' species, "
        "drawn crowns and reach are lane T's, and a species' own mesh lane C's; where trees stand, terrain, "
        "clearings and ground are lane D's).",
    ]
    return "\n".join(lines) + "\n"


def collate(verdicts, objects, images, complete=False):
    """{owner: [defect verdicts, most severe first]} from reviewers' verdicts; objects: {id: Obj}; images: {id: [file
    names]}. An object's later verdict replaces its earlier one (a re-review, after a re-shoot or a fix). A verdict
    that cannot be routed raises ValueError, as does, with `complete`, an object without one."""
    latest = {}
    for v in verdicts:
        oid = v.get("object")
        if oid not in objects:
            raise ValueError(f"a verdict for an unknown object: {oid}")
        latest[oid] = v
    found = {}
    for oid, v in latest.items():
        if v.get("verdict") not in VERDICTS:
            raise ValueError(f"{oid}: verdict {v.get('verdict')!r}")
        if v["verdict"] == "pass":
            continue
        if v.get("severity") not in SEVERITIES:
            raise ValueError(f"{oid}: severity {v.get('severity')!r}")
        if v.get("owner") not in inventory.SESSIONS:
            raise ValueError(f"{oid}: owner {v.get('owner')!r}")
        evidence = v.get("evidence") or []
        cited = [shot_name(e.get("image", "")) for e in evidence]       # a full path, or a review copy's
        if not evidence or any(name not in images.get(oid, ()) for name in cited):
            raise ValueError(f"{oid}: a defect needs evidence in its own images {images.get(oid)}")
        found.setdefault(v["owner"], []).append(v)
    if complete and set(objects) - set(latest):
        raise ValueError(f"no verdict for {sorted(set(objects) - set(latest))}")
    rank = {s: k for k, s in enumerate(SEVERITIES)}
    return {owner: sorted(vs, key=lambda v: (rank[v["severity"]], v["object"])) for owner, vs in sorted(found.items())}


def parse(text):
    """The verdicts in a reviewer's answer: its last JSON array of objects that each name an "object", however it is
    laid out (compact, indented or in a code fence) and whatever precedes it."""
    decoder, found, k = json.JSONDecoder(), None, 0
    while (k := text.find("[", k)) >= 0:
        try:
            value, end = decoder.raw_decode(text, k)
        except ValueError:
            k += 1
            continue
        if isinstance(value, list) and value and all(isinstance(v, dict) and "object" in v for v in value):
            found, k = value, end
        else:
            k += 1
    if found is None:
        raise ValueError("no verdicts in the answer")
    return found


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("packets")
    p.add_argument("--out", type=Path, default=inventory.OUT_DIR / "packets")
    p.add_argument("--only", type=lambda s: [x for x in s.split(",") if x])
    p.add_argument("--most", type=int, default=MOST)
    p.add_argument("--with", dest="assets", type=lambda s: [x for x in s.split(",") if x], default=[],
                   help="assets whose decisions a picture shows every object also answers for, with their checks")
    c = sub.add_parser("collate")
    c.add_argument("verdicts", type=Path, nargs="+")
    c.add_argument("--out", type=Path, default=inventory.OUT_DIR / "findings.json")
    c.add_argument("--partial", action="store_true")
    for s in (p, c):
        s.add_argument("--shots", type=Path, default=inventory.OUT_DIR / "shots")
        s.add_argument("--inventory", type=Path, default=inventory.OUT_DIR / "inventory.json")
    args = parser.parse_args(argv)
    manifest = json.loads(inventory.MANIFEST.read_text(encoding="utf-8"))
    images = shot_images(args.shots)
    objects = reviewable(inventory.load(args.inventory), images)
    if args.cmd == "packets":
        roles = {s["file"]: s["role"] for r in shot_records(args.shots).values() for s in r["shots"]}
        blanks = {name for name in roles if blank(args.shots / name)}
        if bad := unusable(blanks, roles):
            raise SystemExit(f"blank shots, to frame again before review: {bad}")
        args.out.mkdir(parents=True, exist_ok=True)
        copies = args.out / "images"
        copies.mkdir(exist_ok=True)
        blank_copies = {Path(name).stem + ".jpg" for name in blanks}
        checklist = CHECKLIST + tuple(WITH_CHECKS[a] for a in args.assets if a in WITH_CHECKS)
        chosen = with_assets(shoot.chosen(objects, args.only), manifest, args.assets)
        for k, batch in enumerate(batches(chosen, most=args.most), 1):
            given = {o.id: [review_copy(args.shots / name, copies) for name in images[o.id]] for o in batch}
            (args.out / f"batch-{k:02d}.md").write_text(packet(batch, manifest, given, copies.resolve(), blank_copies,
                                                               checklist), encoding="utf-8")
            print(f"batch {k:02d}: {len(batch)} objects, owners {sorted({o.owner for o in batch})}")
        if blanks:
            print(f"blank baseline shots, marked in their packets: {sorted(blanks)}")
        return 0
    verdicts = [v for path in args.verdicts for v in parse(path.read_text(encoding="utf-8"))]
    found = collate(verdicts, {o.id: o for o in objects}, images, complete=not args.partial)
    for owner, vs in found.items():
        print(f"{owner}: {len(vs)} defects")
        for v in vs:
            print(f"  [{v['severity']}] {v['object']}: {v['finding']} ({', '.join(e['image'] for e in v['evidence'])})")
    args.out.write_text(json.dumps(found, indent=1), encoding="utf-8")
    return 0


def shot_records(shots):
    """{object id: record} of the shot tool's records (shots/*.json)."""
    out = {}
    for path in sorted(Path(shots).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        out[record["object"]] = record
    return out


def shot_images(shots):
    """{object id: [image file names]} from the shot tool's records (shots/*.json)."""
    return {oid: [s["file"] for s in record["shots"]] for oid, record in shot_records(shots).items()}


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
