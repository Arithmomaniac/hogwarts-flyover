"""The book register's checker and viewer. canon.json at the repository root is the register: one row per assertion about
how Hogwarts and its grounds should look, each with locators into White Hound's quoted book passages and prose, and the
`accounted` section that disposes of every statement on his pages. It holds locators only, never book text.

    python -X utf8 -m tools.canon check [FILE ...]     check canon.json (or the given register files)
    python -X utf8 -m tools.canon show ROW [--file F]  print a row's passages from the cache, with a Wayback link for each
    python -X utf8 -m tools.canon show-item ITEM       print an accounted item's words, context and disposition
    python -X utf8 -m tools.canon reanchor [FILE]      after the pinned extracts change, find each locator's span again by its hash

`check` works like tools.manifest.verify. It reads every page through tools.whcache (which verifies the pinned SHA-256),
re-hashes each locator's span, confirms each quote's closing citation, checks that every id resolves, that every statement
on the pages is accounted for, and runs the no-text scan: it fails when 6 consecutive normalised words of any string field
also occur in a cached White Hound page (phrases in tools/canon-allowlist.txt, one per line, are exempt; the file is
optional). It never prints the words it matched. Exit status 1 on any failure, 3 when the cache is unavailable.
Nothing `show` or `show-item` prints is written to a tracked file.
"""
import argparse
import collections
import hashlib
import json
import re
import sys
import urllib.parse
from pathlib import Path

from tools import whcache
from tools.inputs import Blocked
from tools.manifest import whsrc
from tools.paths import BUILD, ROOT, SRC, write_atomic

CANON = ROOT / "canon.json"
ALLOWLIST = Path(__file__).with_name("canon-allowlist.txt")
USER_DIR = Path(__file__).with_name("manifest") / "user"
PAGE_TAG = "720edefc4121"
WAYBACK = "https://web.archive.org/web/20261008141630/http://members.madasafish.com/~cj_whitehound/"

# page id -> the txt extract's name in the cache
PAGES = {
    "wh-index": "map_of_Hogwarts.htm",
    "wh-castle": "map_of_Hogwarts__castle.htm",
    "wh-grounds": "map_of_Hogwarts__grounds.htm",
    "wh-hogsmeade": "map_of_Hogwarts__Hogsmeade.htm",
    "wh-location": "map_of_Hogwarts__location.htm",
    "wh-forest": "map_of_Hogwarts__forest.htm",
    "wh-setting": "map_of_Hogwarts__setting.htm",
    "wh-introduction": "map_of_Hogwarts__introduction.htm",
    "wh-journeys": "map_of_Hogwarts__journeys.htm",
    "wh-castle-routes": "map_of_Hogwarts__castle_routes.htm",
    "wh-location-essay": "Location_Location.htm",
}
INDEX_PAGE_OF = {"castle": "wh-castle", "grounds": "wh-grounds", "Hogsmeade": "wh-hogsmeade", "location": "wh-location"}
CITE = re.compile(r"\[\s*(PS|CoS|PoA|GoF|OotP|HBP|DH|QTtA|FB|BtB)\s+ch\.\s*#?\s*(\d+)(?:/\d+)?\s*[;,]*\s*(?:p[./]\s*([^\]]*))?\]")
# non-binding sources White Hound cites: tier 3 Rowling's formal writing (the Famous Wizard cards), tier 4 her informal statements
SOURCE_CITES = (
    (3, "famous-wizard-card", re.compile(r"\[\s*Famous Wizards?(?:\s+Cards?|\s+n[\u00b0o]\s*\d+)\s*\]")),
    (4, "aol-chat", re.compile(r"\[\s*America Online chat transcript[^\]]*\]")),
)
QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})

DATE = re.compile(r"\d{4}-\d{2}-\d{2}$")
KEBAB = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*$")
EVIDENCE = re.compile(r"(astra|opus):[\w.-]+$")
LOG_SOURCE = re.compile(r"(decision_log#|pending_user#|limitation:)\d+$")
READ_BY = ("book", "wh", "ours", "user")
ROLES = ("asserts", "quotes")
FAILURES = ("shows", "omits")
STATUSES = ("draft", "agreed", "split")
KINDS = ("still", "los", "distance", "bearing", "height", "route", "material", "presence", "count")
ABOVE = ("lake", "platform", "ground")
HEIGHT_KEYS = ("height", "eye_h", "h", "from_h", "to_h", "target_h")
ABSOLUTE_KEYS = ("y", "abs", "level", "absolute", "elevation_abs", "z")
STILL_KEYS = ("anchor", "bearing", "elevation", "distance", "hour", "look_for")
ALLOWED_EXCLUSIONS = {"excluded:interior", "excluded:event", "excluded:off-scene", "excluded:no-claim", "excluded:meta",
                      "excluded:manifest"}
PASSAGE_KINDS = ("passage", "passage-repeat", "decision-quote-in-passage", "asset-quote-in-passage")
CITE_KINDS = ("passage", "passage-repeat", "cite-only")
NO_TEXT_SKIP = {"cite", "page", "sha", "id", "evidence", "tests", "image", "by", "anchor", "kind", "status", "verified_on",
                "basis", "source", "date", "object", "pagenum", "role", "disposition", "page_tag", "file", "map", "label",
                "passage"}
GRAM = 6
MIN_PROSE_WORDS = 8          # a page line of this many words or more must lie inside some accounted item
_cache = {}


# ---------------------------------------------------------------------------------------------------- the pages
def page_lines(page):
    """The page's cached extract, as a list of lines (line n is index n-1)."""
    if page not in _cache:
        _cache[page] = whcache.path(f"txt/{PAGES[page]}.txt").read_text(encoding="utf-8").split("\n")
    return _cache[page]


def norm_words(text):
    """The normalised words: whitespace collapsed, curly quotes straight and [cut] dropped."""
    return text.replace("[cut]", " ").translate(QUOTES).split()


def sha12(words):
    return hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest()[:12]


def source_cite(line):
    """(tier, kind, match) of the first non-binding source citation on the line, or None."""
    for tier, kind, rx in SOURCE_CITES:
        m = rx.search(line)
        if m:
            return tier, kind, m
    return None


def span_words(page, lines, cite=None):
    """Words of lines [first, last] of a page. A locator with a printed citation stops where that citation starts on its
    last line."""
    first, last = lines
    rows = page_lines(page)[first - 1:last]
    if cite is not None:
        at = rows[-1].find(cite)
        if at < 0:
            raise ValueError(f"{page}:{last} does not contain {cite}")
        rows[-1] = rows[-1][:at]
    return norm_words(" ".join(rows))


def tokens(text):
    """Lower-case alphanumeric tokens; White Hound's bracketed insertions and [cut] are skipped."""
    text = re.sub(r"\[[^\]]*\]", " ", text.translate(QUOTES))
    return re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))


def page_url(page):
    return WAYBACK + whcache.entries()[f"raw/{PAGES[page]}"]["url"]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, doc):
    write_atomic(path, (json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))


# ---------------------------------------------------------------------------------------------------- what ids resolve to
def ctx():
    """Everything the id checks resolve against: the manifest, the recorded user answers and the page's named places."""
    if "scene" not in _cache:
        scene = load(SRC / "mesh_scene.json")
        users = {}
        for f in USER_DIR.glob("*.json"):
            d = load(f)
            users[f.stem] = {x.split(":")[0] for x in d.get("choices", [])} | {f"quote{i}" for i in range(len(d.get("quotes", [])))}
        data = (BUILD / "data.js").read_text(encoding="utf-8")
        meta = json.loads(data[data.index("{"):data.rindex("}") + 1])["meta"]
        places = {f"named:{k}" for k in meta["named"]} | {f"extra:{k}" for k in meta["extras"]}
        places |= {f"poi:{p['name']}" for p in meta["extras"]["poi"]}
        places |= {f"clearing:{x['id']}" for x in meta["clearings"]} | {f"hill:{h[0]}" for h in meta["hills"]}
        _cache.update(scene=scene, decisions=set(scene["decisions"]), assets={a["id"] for a in scene["assets"]},
                      users=users, places=places)
    return _cache


def grams():
    """Every GRAM consecutive tokens of the cached pages, and the allowlisted phrases."""
    if "grams" not in _cache:
        seen = set()
        for page in PAGES:
            toks = tokens("\n".join(page_lines(page)).replace("[", " ").replace("]", " "))
            for i in range(len(toks) - GRAM + 1):
                seen.add(" ".join(toks[i:i + GRAM]))
        allow = ALLOWLIST.read_text(encoding="utf-8").splitlines() if ALLOWLIST.exists() else []
        _cache["grams"] = seen
        _cache["allow"] = " | ".join(" ".join(tokens(x)) for x in allow if x.strip())
    return _cache["grams"]


class Report:
    def __init__(self):
        self.fails, self.counts = [], collections.Counter()

    def fail(self, where, msg):
        self.fails.append(f"FAIL {where}: {msg}")


def resolve_locator(loc, where, rep, quote):
    """Re-hash a locator's span against the cache. `quote` locators also name the citation closing their passage."""
    rep.counts["locators"] += 1
    need = ("page", "lines", "span", "sha") + (("cite",) if quote else ())
    missing = [k for k in need if k not in loc]
    if missing:
        return rep.fail(where, f"locator lacks {missing}")
    if loc["page"] not in PAGES:
        return rep.fail(where, f"unknown page {loc['page']}")
    rows = page_lines(loc["page"])
    a, b = loc["lines"]
    if not (isinstance(a, int) and isinstance(b, int) and 1 <= a <= b <= len(rows)):
        return rep.fail(where, f"lines {loc['lines']} are outside {loc['page']}")
    try:
        words = span_words(loc["page"], loc["lines"], loc["cite"] if quote else None)
    except ValueError as e:
        return rep.fail(where, str(e))
    if quote:
        sc = source_cite(rows[b - 1])
        if sc:
            return rep.fail(where, f"a tier {sc[0]} source (ruling canon-tiers) cannot be a quote locator; cite it in a reading's "
                                   f"basis (source:{loc['page']}:{b}) or an alternative")
        m = CITE.search(rows[b - 1])
        if not m or m.group(0) != loc["cite"]:
            return rep.fail(where, f"{loc['cite']} is not the citation closing line {b} of {loc['page']}")
    i, j = loc["span"]
    if not (0 <= i <= j < len(words)):
        return rep.fail(where, f"span {loc['span']} is outside the {len(words)} words of lines {loc['lines']}")
    if sha12(words[i:j + 1]) != loc["sha"]:
        return rep.fail(where, f"span hash differs on {loc['page']}:{a}-{b}; run reanchor")


def scan_text(value, where, rep, path=""):
    """The no-text scan over every string field."""
    if isinstance(value, str):
        toks = tokens(value)
        hits = [i for i in range(len(toks) - GRAM + 1) if " ".join(toks[i:i + GRAM]) in grams()]
        hits = [i for i in hits if " ".join(toks[i:i + GRAM]) not in _cache["allow"]]
        if hits:
            rep.fail(where, f"no-text scan: {GRAM}+ words of {path or 'a field'} occur in a White Hound page "
                            f"(word offsets {hits[:4]})")
    elif isinstance(value, dict):
        for k, v in value.items():
            if k not in NO_TEXT_SKIP:
                scan_text(v, where, rep, f"{path}.{k}" if path else k)
    elif isinstance(value, list):
        for n, v in enumerate(value):
            scan_text(v, where, rep, f"{path}[{n}]")


# ---------------------------------------------------------------------------------------------------- rows and rulings
def check_ref(ref, kind, pool, where, rep):
    if ref not in pool:
        rep.fail(where, f"{kind} id {ref!r} does not exist")


def check_basis(basis, rulings, where, rep):
    x = ctx()
    for b in basis:
        kind, _, rest = b.partition(":")
        if kind == "decision":
            check_ref(rest, "decision", x["decisions"], where, rep)
        elif kind == "user":
            f, _, key = rest.partition("#")
            if key not in x["users"].get(f, ()):
                rep.fail(where, f"user basis {b!r} is not a recorded choice or quote")
        elif kind == "ruling":
            if rest not in rulings:
                rep.fail(where, f"ruling {rest!r} is not in the rulings section")
        elif kind == "source":
            page, _, line = rest.rpartition(":")
            ok = page in PAGES and line.isdigit() and 1 <= int(line) <= len(page_lines(page)) \
                and source_cite(page_lines(page)[int(line) - 1]) is not None
            if not ok:
                rep.fail(where, f"source basis {b!r} must be source:<page>:<line> of a tier 3 or 4 citation "
                                f"(Famous Wizard card, AOL chat)")
        else:
            rep.fail(where, f"basis {b!r} must be decision:, user:, ruling: or source:")


def check_check(spec, inventory, where, rep):
    x = ctx()
    if spec.get("kind") not in KINDS:
        return rep.fail(where, f"check kind {spec.get('kind')!r}")
    for k in ABSOLUTE_KEYS:
        if k in spec:
            rep.fail(where, f"check has absolute field {k!r}; give heights above lake, platform or ground")
    for k in HEIGHT_KEYS:
        if k in spec and spec.get("above") not in ABOVE:
            rep.fail(where, f"check height {k!r} needs `above` in {ABOVE}")
    if spec["kind"] == "still":
        for k in STILL_KEYS:
            if k not in spec:
                rep.fail(where, f"still lacks {k}")
    anchors = [spec[k] for k in ("anchor", "from", "to") if isinstance(spec.get(k), str)]
    if not anchors:
        rep.fail(where, "check names no anchor")
    for a in anchors:
        if a not in inventory and a not in x["assets"] and a not in x["places"]:
            rep.fail(where, f"anchor {a!r} is not an inventory id, asset id or place")


def check_row(row, rulings, inventory, ids, rep):
    x = ctx()
    where = row.get("id", "?")
    rep.counts["rows"] += 1
    for k in ("id", "assertion", "reading", "alternatives", "quote", "wh", "object", "check", "holds", "failure",
              "note", "waiver", "verified_on", "evidence", "tests", "status"):
        if k not in row:
            return rep.fail(where, f"row lacks {k}")
    if not KEBAB.match(row["id"]) or row["id"] in ids:
        rep.fail(where, "id is not kebab-case or is repeated")
    ids.add(row["id"])
    if not row["assertion"] or len(row["assertion"].split()) > 30:
        rep.fail(where, "assertion is empty or longer than 30 words")
    r = row["reading"]
    if r.get("by") not in READ_BY:
        rep.fail(where, f"reading.by must be one of {READ_BY}")
    check_basis(r.get("basis", []), rulings, f"{where} reading", rep)
    if "differs_from_wh" not in r:
        rep.fail(where, "reading lacks differs_from_wh")
    for n, alt in enumerate(row["alternatives"]):
        w = f"{where} alternatives[{n}]"
        base_ok = alt.get("reading") and alt.get("why") and alt.get("by") in READ_BY
        if row["status"] == "split" and row["holds"] is None:
            if alt.get("unmeasured") is True and alt.get("followed") is None and base_ok:
                rep.counts["unmeasured alternatives"] += 1
            else:
                measured = isinstance(alt.get("followed"), bool) and base_ok
                reading_ok = alt.get("followed") is None and base_ok and alt.get("holds") in (True, False)
                if not (measured or reading_ok):
                    rep.fail(w, "a reading on a split row needs reading, by, why and holds, and followed null; "
                                "an alternative measured on it needs followed true or false")
        elif alt.get("followed") is None and alt.get("unmeasured") is True and base_ok:
            rep.counts["unmeasured alternatives"] += 1
        elif not isinstance(alt.get("followed"), bool) or not base_ok:
            rep.fail(w, "an alternative needs reading, by, followed (true/false) and why; followed may be null only with "
                        "unmeasured true (a reading awaiting measurement)")
        check_basis(alt.get("basis", []), rulings, w, rep)
    if not row["quote"]:
        rep.fail(where, "a row needs at least one quote locator")
    for n, loc in enumerate(row["quote"]):
        resolve_locator(loc, f"{where} quote[{n}]", rep, True)
    for n, loc in enumerate(row["wh"]):
        resolve_locator(loc, f"{where} wh[{n}]", rep, False)
        if loc.get("role") not in ROLES:
            rep.fail(f"{where} wh[{n}]", f"role must be one of {ROLES}")
    o = row["object"]
    pools = {"inventory": inventory, "assets": x["assets"], "decisions": x["decisions"], "places": x["places"]}
    for kind, pool in pools.items():
        for ref in o.get(kind, []):
            check_ref(ref, kind, pool, f"{where} object", rep)
    if not row["check"]:
        rep.fail(where, "a row needs at least one check")
    for n, spec in enumerate(row["check"]):
        check_check(spec, inventory, f"{where} check[{n}]", rep)
    if row["holds"] not in (True, False, None):
        rep.fail(where, "holds must be true, false or null")
    if row["holds"] is False and row["failure"] not in FAILURES:
        rep.fail(where, f"a row that does not hold needs failure in {FAILURES}")
    if row["holds"] is not False and row["failure"] is not None:
        rep.fail(where, "failure is set only when holds is false")
    if row["holds"] is None and row["status"] not in ("split", "draft"):
        rep.fail(where, "holds is empty only on a split row or a judge's unmeasured draft row")
    if row["status"] not in STATUSES:
        rep.fail(where, f"status must be one of {STATUSES}")
    if row["waiver"] is not None and (row["holds"] is not False or not isinstance(row["waiver"], dict)
                                      or row["waiver"].get("by") != "user"):
        rep.fail(where, "a waiver is set only by the user, on a row that does not hold")
    if row["verified_on"] != PAGE_TAG:
        rep.fail(where, f"verified_on must be the page hash {PAGE_TAG}")
    for e in row["evidence"]:
        if not EVIDENCE.match(e):
            rep.fail(where, f"evidence {e!r} must be astra:<id> or opus:<id>, an audit row")
    for t in row["tests"]:
        f = t.split(":")[0]
        if not (SRC / f).is_file():
            rep.fail(where, f"test file {f} does not exist")
    if row["status"] == "split" and not row["alternatives"] and not row.get("split"):
        rep.fail(where, "a split row lists the readings in alternatives or the positions in `split`")


def check_rulings(rulings, rep):
    x = ctx()
    for rid, r in rulings.items():
        w = f"ruling {rid}"
        rep.counts["rulings"] += 1
        if not KEBAB.match(rid):
            rep.fail(w, "id is not kebab-case")
        if (r.get("date") is not None and not DATE.match(r["date"])) or r.get("by") not in ("user", "orchestrator") \
                or not r.get("summary"):
            rep.fail(w, "needs by (user or orchestrator), summary and, where known, date (YYYY-MM-DD)")
        src = r.get("source", "")
        if src.startswith("decision_log#") and not r.get("date"):
            rep.fail(w, "a decision_log ruling carries its date")
        ok = (LOG_SOURCE.match(src)                     # rows of the orchestrator's records, which the repository does not hold
              or (src.startswith("decision:") and src[9:] in x["decisions"])
              or (src.startswith("user:") and src[5:].partition("#")[2] in x["users"].get(src[5:].partition("#")[0], ()))
              or (src.startswith("file:") and (ROOT / src[5:].partition("#")[0]).is_file())
              or src.startswith("brief:"))
        if not ok:
            rep.fail(w, f"source {src!r} does not resolve")


# ---------------------------------------------------------------------------------------------------- accounted
def disposition_targets(a):
    """The dispositions of an item: its own, or each part's when split, and any `also`."""
    own = [p["disposition"] for p in a.get("parts") or []] if a["disposition"] == "split" else [a["disposition"]]
    return own + list(a.get("also", []))


def check_accounted(doc, rep):
    """Each item carries its locator, which re-hashes against the cache, and a disposition that resolves; every statement on
    the pages (each citation, map key, image, manifest book note and line of 8 words or more) lies inside an item."""
    x = ctx()
    acc = doc.get("accounted", {})
    rows = {r["id"] for r in doc["rows"]}
    rulings = set(doc.get("rulings", {}))
    keys = {"wh-grounds-map": whsrc.map_key("grounds"), "wh-hogsmeade-map": whsrc.map_key("Hogsmeade"),
            "wh-castle-plan": whsrc.castle_plan_key()}
    index = {(e["name"], INDEX_PAGE_OF[e["page"]]): e for e in whsrc.image_index()}

    def resolves(d, iid):
        k, _, rest = d.partition(":")
        return (k == "row" and rest in rows) \
            or (k == "same-as" and rest in acc and rest != iid) \
            or (k == "decision" and rest in x["decisions"]) \
            or (k == "asset" and rest in x["assets"]) \
            or (k == "ruling" and rest in rulings) \
            or d in ALLOWED_EXCLUSIONS \
            or (d.startswith("excluded:user-choice:") and d.split(":")[2] in x["decisions"])

    covered, cited, mapped, imaged, book_notes = collections.defaultdict(set), set(), set(), set(), set()
    for iid, it in acc.items():
        rep.counts["accounted"] += 1
        loc, kind = it.get("locator") or {}, it.get("kind")
        if kind in PASSAGE_KINDS:
            resolve_locator(loc, iid, rep, True)
        elif kind == "cite-only":
            rows_ = page_lines(loc["page"]) if loc.get("page") in PAGES else []
            if not rows_ or loc["cite"] not in rows_[loc["lines"][0] - 1]:
                rep.fail(iid, "cite-only reference not on its line")
        elif kind in ("wh-prose", "wh-tail", "wh-lead-in"):
            resolve_locator(loc, iid, rep, False)
        elif kind in ("tier3-source", "tier4-source"):
            rep.counts["locators"] += 1
            sc = source_cite(page_lines(loc["page"])[loc["lines"][1] - 1]) if loc.get("page") in PAGES else None
            if not sc or sc[2].group(0) != loc["cite"] or kind != f"tier{sc[0]}-source":
                rep.fail(iid, f"the {kind} citation is not the one closing line {loc['lines'][1]} of {loc['page']}")
            elif sha12(span_words(loc["page"], loc["lines"], loc["cite"])[loc["span"][0]:loc["span"][1] + 1]) != loc["sha"]:
                rep.fail(iid, "tier source hash differs from the text before its citation")
        elif kind == "map-key":
            text = keys.get(loc.get("map"), {}).get(loc.get("label"))
            if text is None or sha12(norm_words(text)) != loc["sha"]:
                rep.fail(iid, "map key item missing or its hash differs")
            mapped.add((loc.get("map"), loc.get("label")))
        elif kind == "image":
            e = index.get((loc.get("image"), loc.get("page")))
            words = norm_words(" ".join([whsrc.clean_caption(e.get("cap")), whsrc.clean_caption(e.get("alt"))])) if e else None
            if e is None or sha12(words) != loc["sha"]:
                rep.fail(iid, "image entry missing or its caption hash differs")
            imaged.add((loc.get("image"), loc.get("page")))
        elif kind == "test-line":
            lines = (SRC / loc["file"]).read_text(encoding="utf-8").split("\n") if (SRC / loc["file"]).is_file() else []
            line = lines[loc["line"] - 1] if 0 < loc["line"] <= len(lines) else ""
            if not any(m.group(1) == loc["book"] and int(m.group(2)) == loc["chapter"]
                       for m in re.finditer(r"\b(PS|CoS|PoA|GoF|OotP|HBP|DH)\s*(?:ch\.?\s*)?#?(\d{1,2})\b", line)):
                rep.fail(iid, "test line no longer cites that book and chapter")
        elif kind == "manifest-book-note":
            if not any(b["source"] == "book" for b in x["scene"]["decisions"].get(it.get("decision"), {}).get("basis", [])):
                rep.fail(iid, "decision has no book note")
            book_notes.add(it["decision"])
        else:
            rep.fail(iid, f"unknown item kind {kind}")
        if loc.get("page") in PAGES and "lines" in loc:
            covered[loc["page"]].update(range(loc["lines"][0], loc["lines"][1] + 1))
            if kind in CITE_KINDS:
                cited.add((loc["page"], loc["lines"][1]))

        if not it.get("disposition") or (it["disposition"] == "split" and not it.get("parts")):
            rep.fail(iid, "no disposition, or a split item without parts")
            continue
        for d in disposition_targets(it):
            if not resolves(d, iid):
                rep.fail(iid, f"disposition {d!r} is not allowed or does not resolve")
        rep.counts["disposed " + it["disposition"].split(":")[0]] += 1

    # nothing unaccounted: derive what the pages hold, and compare
    want = {(p, n) for p in PAGES for n, line in enumerate(page_lines(p), 1) if CITE.search(line) and not source_cite(line)}
    for p, n in sorted(want - cited):
        rep.fail(f"{p}:{n}", "unaccounted: no item for the passage closing on this line")
    for m, label in sorted({(m, k) for m, v in keys.items() for k in v} - mapped):
        rep.fail(f"{m}:{label}", "unaccounted: no item for this map key")
    for name, page in sorted(set(index) - imaged):
        rep.fail(f"{page}:{name}", "unaccounted: no item for this image")
    for d in sorted({d for d, v in x["scene"]["decisions"].items() if any(b["source"] == "book" for b in v["basis"])} - book_notes):
        rep.fail(d, "unaccounted: no item for this decision's book note")
    for p in PAGES:
        for n, line in enumerate(page_lines(p), 1):
            if len(norm_words(line)) >= MIN_PROSE_WORDS and n not in covered[p]:
                rep.fail(f"{p}:{n}", "unaccounted: no item covers this line")
    check_row_locators(acc, rows_by_id(doc), rep)


def rows_by_id(doc):
    return {r["id"]: r for r in doc["rows"]}


def check_row_locators(acc, rows, rep):
    """An item (or a part of one) disposed `row:X`, or listed in `also`, needs a locator on row X. A passage: any occurrence
    of its text, in quote or wh, on overlapping lines of the same page. A White Hound item: a wh locator with role asserts (an
    alternative-source link carries alternative_basis_only) on overlapping lines."""
    by_base = collections.defaultdict(list)       # a passage and its repeats (P0010, P0010#2, ...) are occurrences of one text
    for iid, it in acc.items():
        if it.get("kind") in ("passage", "passage-repeat"):
            by_base[iid.split("#")[0]].append(it["locator"])

    def overlap(a, b):
        return a["page"] == b["page"] and a["lines"][0] <= b["lines"][1] and b["lines"][0] <= a["lines"][1]

    missing = 0
    for iid, it in acc.items():
        loc = it.get("locator") or {}
        if "lines" not in loc or "page" not in loc or not it.get("disposition"):
            continue
        for rid in sorted({d[4:] for d in disposition_targets(it) if d.startswith("row:")}):
            row = rows.get(rid)
            if row is None:
                continue
            if it["kind"] in PASSAGE_KINDS:
                occ = by_base.get(it.get("passage") or iid.split("#")[0], [loc]) + [loc]
                ok = any(overlap(x, o) for x in row["quote"] + row["wh"] for o in occ)
            elif it["kind"].startswith("wh-"):
                ok = any(overlap(x, loc) and x.get("role") == "asserts" for x in row["wh"])
            else:
                continue
            if not ok:
                missing += 1
                rep.fail(iid, f"disposed row:{rid}, but that row has no locator for it")
    rep.counts["row dispositions without a locator on the row"] = missing


# ---------------------------------------------------------------------------------------------------- the register
def check_doc(doc, name="canon.json"):
    """The report for a register document."""
    rep, ids = Report(), set()
    if doc.get("page_tag") != PAGE_TAG:
        rep.fail(name, f"page_tag must be {PAGE_TAG}")
    inventory = set(doc.get("inventory", []))
    rulings = doc.get("rulings", {})
    check_rulings(rulings, rep)
    for row in doc["rows"]:
        check_row(row, rulings, inventory, ids, rep)
    for e in doc.get("excluded", []):
        where = f"excluded {e.get('id')}"
        rep.counts["excluded"] += 1
        if e.get("disposition") not in ALLOWED_EXCLUSIONS or e.get("id") in ids:
            rep.fail(where, "needs an allowed exclusion and an id not used by a row")
        if e.get("decision") not in ctx()["decisions"]:
            rep.fail(where, f"decision {e.get('decision')!r} does not exist")
        for n, loc in enumerate(e.get("quote", [])):
            resolve_locator(loc, f"{where} quote[{n}]", rep, True)
    check_accounted(doc, rep)
    scan_text({k: v for k, v in doc.items() if k != "inventory"}, name, rep)
    return rep


def cmd_check(files):
    bad = 0
    for p in [Path(f) for f in files] or [CANON]:
        grams()
        rep = check_doc(load(p), p.name)
        print(f"{p.name}: " + ", ".join(f"{v} {k}" for k, v in sorted(rep.counts.items()))
              + (" - OK" if not rep.fails else f" - {len(rep.fails)} FAILURES"))
        for f in rep.fails[:60]:
            print("  " + f)
        if len(rep.fails) > 60:
            print(f"  ... and {len(rep.fails) - 60} more")
        bad += len(rep.fails)
    print("canon:", "OK" if not bad else f"{bad} failures")
    return 1 if bad else 0


# ---------------------------------------------------------------------------------------------------- looking things up
def fragment_link(page, words):
    a = " ".join(words[:4])
    z = " ".join(words[-4:])
    frag = urllib.parse.quote(a, safe="") + ("" if len(words) <= 8 else "," + urllib.parse.quote(z, safe=""))
    return f"{page_url(page)}#:~:text={frag}"


def cmd_show(rid, file=None):
    doc = load(file or CANON)
    row = next((r for r in doc["rows"] if r["id"] == rid), None)
    if row is None:
        raise SystemExit(f"{rid}: no such row")
    print(f"{row['id']} [{row['status']}] holds={row['holds']} failure={row['failure']}\n  {row['assertion']}")
    for key in ("quote", "wh"):
        for loc in row[key]:
            words = span_words(loc["page"], loc["lines"], loc.get("cite"))[loc["span"][0]:loc["span"][1] + 1]
            print(f"\n{key} {loc['page']}:{loc['lines'][0]}-{loc['lines'][1]} {loc.get('cite', loc.get('role', ''))}")
            print("  " + " ".join(words))
            print("  " + fragment_link(loc["page"], words))
    return 0


def cmd_show_item(iid, file=None):
    """An accounted item's words from the pinned cache with a line of context either side, and its disposition."""
    acc = load(file or CANON).get("accounted", {})
    if iid not in acc:
        raise SystemExit(f"{iid}: not an accounted item")
    it = acc[iid]
    loc = it["locator"]
    where = f"{loc['page']}:{loc['lines'][0]}-{loc['lines'][1]}" if "lines" in loc else str(loc)
    print(f"{iid} [{it['kind']}] {where} {loc.get('cite', '')}".rstrip())
    print(f"disposition: {it['disposition']} (by {it['by']})" + (f" | also {it['also']}" if it.get("also") else ""))
    if it.get("reason"):
        print(f"reason: {it['reason']}")
    for k, p in enumerate(it.get("parts", [])):
        print(f"  part {k} sentences {p['sentences']}: {p['disposition']} - {p.get('reason', '')}")
    words = []
    if "span" in loc:
        words = span_words(loc["page"], loc["lines"], loc.get("cite"))[loc["span"][0]:loc["span"][1] + 1]
    elif it["kind"] == "map-key":
        words = norm_words(whsrc.map_key("grounds" if "grounds" in loc["map"] else "Hogsmeade")[loc["label"]]) \
            if loc["map"] != "wh-castle-plan" else norm_words(whsrc.castle_plan_key()[loc["label"]])
    elif it["kind"] == "test-line":
        print("\nTEST src/%s:%d: %s" % (loc["file"], loc["line"],
                                        (SRC / loc["file"]).read_text(encoding="utf-8").split("\n")[loc["line"] - 1].strip()[:200]))
    if words:
        print("\nITEM: " + " ".join(words))
    if "lines" in loc and loc.get("page") in PAGES:
        rows = page_lines(loc["page"])
        a, b = loc["lines"]
        for side, rng in (("before", range(a - 1, 0, -1)), ("after", range(b + 1, len(rows) + 1))):
            for n in rng:
                if rows[n - 1].strip():
                    text = " ".join(rows[n - 1].split())
                    print(f"context-{side} ({n}): " + (text[-200:] if side == "before" else text[:200]))
                    break
        if words:
            print("link: " + fragment_link(loc["page"], words[:12] if len(words) > 12 else words))
    return 0


def cmd_reanchor(file):
    """Find each locator's span again by its hash after the pinned extracts change."""
    path = Path(file)
    doc = load(path)
    fixed = lost = 0

    def candidates(loc):
        rows = page_lines(loc["page"])
        for n, line in enumerate(rows, 1):
            if (loc["cite"] in line) if "cite" in loc else CITE.search(line):
                first = n
                while first > 1 and rows[first - 2].strip() and not CITE.search(rows[first - 2]):
                    first -= 1
                yield [first, n]
        if "cite" not in loc:
            for n in range(1, len(rows) + 1):
                yield [n, n]

    def visit(loc):
        nonlocal fixed, lost
        rep = Report()
        resolve_locator(loc, "x", rep, "cite" in loc)
        if not rep.fails:
            return
        size = loc["span"][1] - loc["span"][0] + 1
        for lines in candidates(loc):
            words = span_words(loc["page"], lines, loc.get("cite"))
            for i in range(0, len(words) - size + 1):
                if sha12(words[i:i + size]) == loc["sha"]:
                    loc["lines"], loc["span"] = lines, [i, i + size - 1]
                    fixed += 1
                    return
        lost += 1
        print("lost:", loc["page"], loc["lines"], loc["sha"])

    def walk(v):
        if isinstance(v, dict):
            if {"page", "lines", "span", "sha"} <= set(v):
                visit(v)
            else:
                for x in v.values():
                    walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(doc)
    dump(path, doc)
    print(f"reanchored {fixed}, lost {lost}")
    return 1 if lost else 0


def main(argv):
    p = argparse.ArgumentParser(prog="python -X utf8 -m tools.canon", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check")
    s.add_argument("files", nargs="*")
    s = sub.add_parser("show")
    s.add_argument("row")
    s.add_argument("--file")
    s = sub.add_parser("show-item")
    s.add_argument("item")
    s.add_argument("--file")
    s = sub.add_parser("reanchor")
    s.add_argument("file", nargs="?", default=str(CANON))
    ns = p.parse_args(argv)
    try:
        if ns.cmd == "check":
            return cmd_check(ns.files)
        if ns.cmd == "show":
            return cmd_show(ns.row, ns.file)
        if ns.cmd == "show-item":
            return cmd_show_item(ns.item, ns.file)
        return cmd_reanchor(ns.file)
    except Blocked as e:
        print(f"blocked: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
