"""Shared vocabulary of the manifest sources: evidence records, asset records, footprint claims and the context that
every family's assets and decisions are built from."""
import math
from typing import NamedTuple

C, G, H, L = "wh-castle", "wh-grounds", "wh-hogsmeade", "wh-location"
GM, HM, CP = "wh-grounds-map", "wh-hogsmeade-map", "wh-castle-plan"

FT = 0.3048
FOOTPRINT = ["locA.bsd", "locN.bcode"]
ORDER = ("id", "family", "title", "status", "labels", "sources", "images", "inputs", "components", "castle_pieces",
         "rule", "parents", "traced", "dimensions", "parts", "materials", "inferred", "decisions")


def q(source, quote):
    return {"source": source, "quote": quote}


def note(source, text):
    return {"source": source, "note": text}


def choice(text):
    return {"source": "user", "choice": text}


def decision(text, *basis, images=None, inferred=False):
    out = {"text": text, "basis": list(basis)}
    if images:
        out["images"] = images
    if inferred:
        out["inferred"] = True
    return out


def ft(feet):
    return round(feet * FT, 2)


def labels(source, *keys):
    return [f"{source}:{k}" for k in keys]


def asset(id, family, title, sources, **fields):
    record = {"id": id, "family": family, "title": title, "status": "planned", "sources": sources, **fields}
    return {k: record[k] for k in ORDER if record.get(k)}


class Claims:
    """Footprint and castle-piece claims located from the live decode (`at` = decoded centre).

    `census` holds positions recorded by the provenance research (merged from every family's CENSUS); a component that
    has drifted from its census position fails loudly rather than being re-assigned silently."""

    def __init__(self, components, pieces, census):
        self.components = {c["id"]: c for c in components}
        self.pieces = pieces
        self.census = census

    def comp(self, *ids, kinds):
        out = []
        for i in ids:
            c = self.components[i]
            if c["kind"] not in kinds:
                raise ValueError(f"component {i} is {c['kind']}, expected one of {kinds}")
            if i in self.census and math.dist(self.census[i], (c["x"], c["z"])) > 1.0:
                raise ValueError(f"component {i} moved from its census position")
            out.append({"id": i, "at": [c["x"], c["z"]]})
        return out

    def piece(self, x, z, what):
        hits = [p for p in self.pieces if math.dist((x, z), (p["x"], p["z"])) <= 1.0]
        if len(hits) != 1 or hits[0]["role"] in ("main", "court"):
            raise ValueError(f"no unique detached/outside castle piece at {(x, z)}")
        return {"at": [hits[0]["x"], hits[0]["z"]], "note": what}


class Context(NamedTuple):
    """What the families' assets and decisions are built from."""
    claims: Claims
    user: dict                 # the recorded user answers (whsrc.user_record)
    choices: dict              # choice key -> the recorded "key: answer" text
    defaults: dict             # the approved form defaults, as a choice record
    offered: dict              # the text of the defaults offered, as a user note
    train_gap_m: float         # the registered gap between the coaches and the engine
    hog_key: dict              # the Hogsmeade map's key, label -> text
