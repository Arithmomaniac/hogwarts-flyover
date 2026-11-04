"""Generate src/mesh_scene.json, the source-labelled exterior asset manifest, from the per-family sources in
tools/manifest/families (one module per family: its assets, decisions, image notes, built assets and census, and
optionally, through excluded(ctx), the mapped footprints it leaves unbuilt).

    python -X utf8 -m tools.manifest.gen            # rewrite src/mesh_scene.json
    python -X utf8 -m tools.manifest.gen --check    # exit 1 unless a fresh generation equals the committed file
    python -X utf8 -m tools.manifest.gen --out PATH # write elsewhere

Rebuild the meshes afterwards: the payload stores the manifest's digest. Check the quotations with tools.manifest.verify.
"""
import importlib
import json
import math
import pkgutil
import sys
from pathlib import Path

from tools.paths import ROOT
from tools.manifest import families as family_package
from tools.manifest import general
from tools.manifest import whsrc as w
from tools.manifest.common import Claims, Context, choice, note
from mesh_source import building_components, castle_pieces, load_data

OUT = ROOT / "src" / "mesh_scene.json"
WIDTH = 118
# The phase-0 families keep their places: in the manifest's list of families and, in another order, in its list of
# assets. A new family follows them in both lists, by name.
FAMILY_ORDER = ["castle", "greenhouses", "transport", "village", "grounds", "willow"]
ASSET_ORDER = ["castle", "greenhouses", "transport", "grounds", "willow", "village"]
# The phase-0 decisions keep their places; new decisions follow, scene-wide ones first, then family by family.
DECISION_ORDER = [
    "wh-authority", "live-origin-provenance", "scope-exterior", "static-exteriors", "performance-budget",
    "greenhouse-glass", "exclude-gof-set-pieces", "exclude-boats", "court-glasshouse", "guards-vans",
    "detached-engine", "short-guards-vans", "railway-regrade", "railway-tunnel", "railway-culvert", "railway-curves",
    "lamplit-windows", "station-terrace", "generic-tender-locomotive", "train-scarlet-coaches",
    "station-layout-either", "ship-skeletal-book", "wh-skyline", "towers-flat-battlemented", "warm-masonry",
    "astronomy-beside-doors", "castle-colour-red-cast", "ravenclaw-battlements", "small-scottish-turrets",
    "court-buildings-as-houses", "castle-norman-openings", "mont-orgueil-roofed-wall-walks", "castle-mass-sheds",
    "castle-plan-interiors-excluded", "castle-speck-threshold", "beach-false-positives",
    "third-entrance-speculative", "hedge-heuristic", "stands-full-ring", "pitch-goals-equal-height",
    "pitch-layout-wh", "hut-plan", "timber-hut", "hut-front-gable", "hut-weatherboard-creosoted",
    "hut-roof-shingles", "hut-stovepipe", "garden-fence", "garden-contents", "willow-root-gap",
    "harbour-local-cutout", "kirk-lantern-finial", "hogsmeade-lamps-high-street-only", "hogsmeade-hanging-signs",
    "thatch-straw-animals-optional", "no-half-timbering", "village-roof-mix", "village-wings-join-parents",
    "greenhouse-design-inferred"]


def families():
    """Family name -> its manifest module, from every module in tools/manifest/families."""
    found = {}
    for info in pkgutil.iter_modules(family_package.__path__):
        module = importlib.import_module(f"{family_package.__name__}.{info.name}")
        if module.FAMILY in found:
            raise ValueError(f"two manifest modules declare the family {module.FAMILY}")
        found[module.FAMILY] = module
    return found


def ordered(names, legacy):
    return [n for n in legacy if n in names] + sorted(set(names) - set(legacy))


def merged(parts, what):
    out = {}
    for part in parts:
        clash = out.keys() & part.keys()
        if clash:
            raise ValueError(f"{what} defined twice: {sorted(clash)}")
        out.update(part)
    return out


def generate(data=None):
    data = data or load_data()
    mods = families()
    census = merged((m.CENSUS for m in mods.values()), "census positions")
    claims = Claims(building_components(data), castle_pieces(data), census)
    train = data["meta"]["train"]
    gap = min(math.dist(c, e) for c in train["carriages"] for e in train["engine"])
    sources = general.sources()
    user = w.user_record()
    choices = {c.split(":", 1)[0]: c for c in user["choices"]}
    ctx = Context(claims, user, choices, choice(choices["visual_choices"]),
                  note("user", "approved form defaults: " + user["visual_defaults_offered"]), gap,
                  sources["wh-hogsmeade-map"]["key"])
    images, unavailable = general.build_images(merged([general.IMAGE_NOTES] + [m.IMAGE_NOTES for m in mods.values()],
                                                      "image notes"))
    assets = []
    for name in ordered(mods, ASSET_ORDER):
        for record in mods[name].assets(ctx):
            if record["family"] != name:
                raise ValueError(f"{record['id']} is listed by the {name} module but belongs to {record['family']}")
            if record["id"] in mods[name].BUILT:
                record["status"] = "built"
            assets.append(record)
    found = merged([general.decisions(ctx)] + [mods[n].decisions(ctx) for n in ordered(mods, ASSET_ORDER)],
                   "decisions")
    decisions = {k: found[k] for k in DECISION_ORDER if k in found}
    decisions.update({k: v for k, v in found.items() if k not in decisions})
    excluded = general.excluded(claims)
    excluded["components"] += [c for n in ordered(mods, ASSET_ORDER) if hasattr(mods[n], "excluded")
                               for c in mods[n].excluded(ctx)]
    return {"schema": "hogwarts-flyover/mesh-scene", "version": 1, "frame": general.frame(), "sources": sources,
            "images": images, "decisions": decisions, "families": ordered(mods, FAMILY_ORDER), "assets": assets,
            "excluded": {**excluded, "images": unavailable}}


def fmt(obj, indent=0, col=0):
    """JSON with each container on one line when it fits, so claims and coordinate pairs stay compact."""
    one = json.dumps(obj, ensure_ascii=False)
    if not isinstance(obj, (dict, list)) or not obj or col + len(one) + 1 <= WIDTH:
        return one
    pad, inner = " " * indent, " " * (indent + 2)
    if isinstance(obj, list):
        return "[\n" + ",\n".join(inner + fmt(v, indent + 2, indent + 2) for v in obj) + f"\n{pad}]"
    rows = []
    for k, v in obj.items():
        key = json.dumps(k, ensure_ascii=False) + ": "
        rows.append(inner + key + fmt(v, indent + 2, indent + 2 + len(key)))
    return "{\n" + ",\n".join(rows) + f"\n{pad}}}"


def text(scene):
    return fmt(scene) + "\n"


def main(argv):
    scene = generate()
    out = text(scene)
    if "--check" in argv:
        same = OUT.read_bytes() == out.encode("utf-8")
        print(f"{OUT.relative_to(ROOT)} {'matches' if same else 'DIFFERS FROM'} a fresh generation")
        return 0 if same else 1
    target = Path(argv[argv.index("--out") + 1]) if "--out" in argv else OUT
    target.write_text(out, encoding="utf-8", newline="\n")
    print(f"wrote {target} ({len(scene['assets'])} assets, {len(scene['images'])} images, "
          f"{len(scene['decisions'])} decisions)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
