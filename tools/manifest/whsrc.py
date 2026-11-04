"""Readers for White Hound's pages, map keys and image index (from the git-ignored cache), and the recorded user answers.

Shared by the manifest generator and the quotation checker. White Hound's text never enters Git: tools/whcache.py
fills the cache, pinned file by file."""
import html
import json
import re
from pathlib import Path

from tools import whcache

USER = Path(__file__).resolve().parent / "user"

BASE = "http://members.madasafish.com/~cj_whitehound/Fanfic/"
WAYBACK = "https://web.archive.org/web/20261008141630/"
PAGES = {
    "wh-index": ("map_of_Hogwarts.htm", "map_of_Hogwarts.htm"),
    "wh-castle": ("map_of_Hogwarts/castle.htm", "map_of_Hogwarts__castle.htm"),
    "wh-grounds": ("map_of_Hogwarts/grounds.htm", "map_of_Hogwarts__grounds.htm"),
    "wh-hogsmeade": ("map_of_Hogwarts/Hogsmeade.htm", "map_of_Hogwarts__Hogsmeade.htm"),
    "wh-location": ("map_of_Hogwarts/location.htm", "map_of_Hogwarts__location.htm"),
    "wh-forest": ("map_of_Hogwarts/forest.htm", "map_of_Hogwarts__forest.htm"),
    "wh-setting": ("map_of_Hogwarts/setting.htm", "map_of_Hogwarts__setting.htm"),
}
INDEX_PAGES = {"castle": "wh-castle", "grounds": "wh-grounds", "Hogsmeade": "wh-hogsmeade", "location": "wh-location"}
KEY_RE = re.compile(r"<strong>\((\w+)\)</strong></td>\s*<td[^>]*>(.*?)</td>", re.S)
# castle-plan key: castle txt lines 925..1075, label then meaning two lines later, every fourth line
PLAN_FIRST, PLAN_LAST = 925, 1075


def norm(s):
    return " ".join(s.split())


def strip_markup(s):
    return norm(html.unescape(re.sub(r"<[^>]+>", " ", s)))


def raw_page(name):
    data = whcache.path(f"raw/{name}").read_bytes()
    try:
        return data.decode("cp1252")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def page_text(source):
    return whcache.path(f"txt/{PAGES[source][1]}.txt").read_text(encoding="utf-8")


def map_key(page):
    """Verbatim key of the grounds or Hogsmeade labelled map: label -> text."""
    found = KEY_RE.findall(raw_page(f"map_of_Hogwarts__{page}_map.htm"))
    key = {label: strip_markup(body) for label, body in found}
    if len(key) != len(found):
        raise ValueError(f"duplicate labels in the {page} map key")
    return key


def castle_plan_key():
    lines = page_text("wh-castle").split("\n")
    key = {}
    for n in range(PLAN_FIRST, PLAN_LAST, 4):
        label, meaning = lines[n - 1].strip(), lines[n + 1].strip()
        if not label or not meaning or label in key:
            raise ValueError(f"castle-plan key line {n} is not a label/meaning pair")
        key[label] = meaning
    return key


def clean_caption(s):
    return norm((s or "").replace("\\'", "'")).rstrip(">").strip()


def image_index():
    return json.loads(whcache.path("whref-index.json").read_text(encoding="utf-8"))


def user_record():
    """The user's recorded quotes and choices: user/general.json first, then any family's own file by name."""
    files = [USER / "general.json"] + sorted(p for p in USER.glob("*.json") if p.name != "general.json")
    record = {"quotes": [], "choices": []}
    for f in files:
        part = json.loads(f.read_text(encoding="utf-8"))
        record["quotes"] += part.get("quotes", [])
        record["choices"] += part.get("choices", [])
        if "visual_defaults_offered" in part:
            record["visual_defaults_offered"] = part["visual_defaults_offered"]
    return record
