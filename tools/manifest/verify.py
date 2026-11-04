"""Check every manifest quotation and choice against White Hound's text, the parsed map keys and the recorded user
answers, and the manifest's images against the inspected image index.

    python -X utf8 -m tools.manifest.verify      (exit status 1 on any failure)
"""
import json
import sys

from tools.paths import ROOT
from tools.manifest import whsrc as w

MANIFEST = ROOT / "src" / "mesh_scene.json"
DEFAULTS_PREFIX = "approved form defaults: "


def page_corpora():
    """Normalised page text plus that page's image captions/alt text, per page source."""
    corpora = {sid: [w.norm(w.page_text(sid))] for sid in w.PAGES}
    for entry in w.image_index():
        corpora[w.INDEX_PAGES[entry["page"]]] += [w.clean_caption(entry.get("cap")), w.clean_caption(entry.get("alt"))]
    return corpora


def locate(quote, text):
    """Longest leading run of the quote's words found in the text, to point at a near miss."""
    words = quote.split()
    for n in range(len(words), 0, -1):
        at = text.find(" ".join(words[:n]))
        if at >= 0:
            return n, text[max(0, at - 40): at + len(quote) + 60]
    return 0, ""


def failures(scene):
    """(number of quotations and choices checked, failure messages)."""
    corpora, user, out = page_corpora(), w.user_record(), []
    keys = {"wh-grounds-map": w.map_key("grounds"), "wh-hogsmeade-map": w.map_key("Hogsmeade"),
            "wh-castle-plan": w.castle_plan_key()}
    for sid, key in keys.items():
        if scene["sources"][sid]["key"] != key:
            out.append(f"{sid}: manifest key differs from the parsed source key")
    evidence = [(a["id"], e) for a in scene["assets"] for e in a["sources"]]
    evidence += [(d, e) for d, v in scene["decisions"].items() for e in v["basis"]]
    checked = 0
    for owner, entry in evidence:
        source = entry["source"]
        if "quote" in entry:
            checked += 1
            quote = w.norm(entry["quote"])
            if source in corpora:
                if not any(quote in text for text in corpora[source]):
                    n, near = locate(quote, corpora[source][0])
                    out.append(f"{owner}: {source} lacks {quote!r}; first {n} words match near: {near!r}")
            elif source in keys:
                if not any(quote in text for text in keys[source].values()):
                    out.append(f"{owner}: {source} key lacks {quote!r}")
            elif source == "user":
                if entry["quote"] not in user["quotes"]:
                    out.append(f"{owner}: unrecorded user quote {entry['quote']!r}")
            else:
                out.append(f"{owner}: quotes from {source} are not allowed")
        if "choice" in entry:
            checked += 1
            if source != "user" or entry["choice"] not in user["choices"]:
                out.append(f"{owner}: unrecorded user choice {entry['choice']!r}")
        if source == "user" and entry.get("note", "").startswith(DEFAULTS_PREFIX):
            if entry["note"][len(DEFAULTS_PREFIX):] != user["visual_defaults_offered"]:
                out.append(f"{owner}: approved-defaults note differs from the recorded form text")
    index = {}
    for entry in w.image_index():
        index.setdefault(entry["name"], set()).add(w.INDEX_PAGES[entry["page"]])
    images = scene["images"]
    unavailable = {e["name"]: set(e["pages"]) for e in scene["excluded"]["images"]}
    if set(images) & set(unavailable):
        out.append("an image is both used and excluded")
    if set(index) != set(images) | set(unavailable):
        out.append(f"image index mismatch: {sorted(set(index) ^ (set(images) | set(unavailable)))}")
    for name, pages in index.items():
        recorded = set(images[name]["pages"]) if name in images else unavailable.get(name)
        if recorded != pages:
            out.append(f"{name}: pages {recorded} != index pages {pages}")
    return checked, out, len(index), len(keys)


def main():
    checked, found, images, keys = failures(json.loads(MANIFEST.read_text(encoding="utf-8")))
    print(f"checked {checked} quotes/choices, {images} indexed images, {keys} map keys")
    for f in found:
        print("FAIL", f)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
