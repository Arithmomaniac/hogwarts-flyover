"""The mesh scene manifest must account for every mapped structure, map label and reference image exactly once."""

import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assets_testing as world  # noqa: E402
from mesh_source import CHANNELS, building_components, castle_pieces  # noqa: E402


TOP_KEYS = {"schema", "version", "frame", "sources", "images", "decisions", "families", "assets", "excluded"}


ASSET_KEYS = {"id", "family", "title", "status", "labels", "sources", "images", "inputs", "components",
              "castle_pieces", "rule", "parents", "traced", "dimensions", "parts", "materials", "inferred",
              "decisions"}


REQUIRED_ASSET_KEYS = {"id", "family", "title", "status", "sources"}


STATUSES = {"planned", "built", "integrated"}


STANCES = {"model", "endorsed", "partial", "comparative", "negative", "historical", "schematic", "context"}


GUIDING = {"model", "endorsed", "partial"}


EVIDENCE = {"quote", "choice", "note"}


MAX_QUOTE_WORDS = 25


CLAIM_TOLERANCE_M = 1.0


DIMENSION_SUFFIXES = ("_m", "_count", "_deg")


# whole-mass castle pieces are claimed by role; detached and outside remnants need an explicit position
ROLE_CLAIMS = {"main", "court"}


REQUIRED_INPUTS = {
    "casA.low_wall", "casA.wall", "casA.range", "casA.block", "casB.great_hall", "casB.keep",
    "locA.bsd", "locN.bcode", "locA.hedge", "locB.wall",
    "meta.towers", "meta.named", "meta.train", "meta.pitch",
    "meta.extras.gates", "meta.extras.tomb", "meta.extras.ship", "meta.extras.willow",
}


def near(at, item):
    return math.dist(at, (item["x"], item["z"])) <= CLAIM_TOLERANCE_M


def matches(rule, component):
    return (component["region"] == rule["region"] and component["kind"] in rule["kinds"]
            and rule.get("min_area", 0) <= component["area"] <= rule.get("max_area", math.inf))


class MeshSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = world.scene()
        cls.data = world.data()
        cls.components = {c["id"]: c for c in building_components(cls.data)}
        cls.pieces = castle_pieces(cls.data)
        cls.assets = cls.scene["assets"]

    def explicit_claims(self):
        claims = [(a["id"], c) for a in self.assets for c in a.get("components", [])]
        return claims + [("excluded", c) for c in self.scene["excluded"]["components"]]

    def test_schema(self):
        self.assertEqual(set(self.scene), TOP_KEYS)
        self.assertEqual((self.scene["schema"], self.scene["version"]), ("hogwarts-flyover/mesh-scene", 1))
        ids = [a["id"] for a in self.assets]
        self.assertEqual(len(ids), len(set(ids)))
        families = self.scene["families"]
        self.assertEqual(len(families), len(set(families)))
        self.assertEqual({a["family"] for a in self.assets}, set(families))
        for asset in self.assets:
            with self.subTest(asset=asset["id"]):
                self.assertLessEqual(set(asset), ASSET_KEYS)
                self.assertLessEqual(REQUIRED_ASSET_KEYS, set(asset))
                self.assertIn(asset["status"], STATUSES)
                self.assertTrue(asset["sources"])
                for key, value in asset.get("dimensions", {}).items():
                    self.assertTrue(key.endswith(DIMENSION_SUFFIXES), key)
                    self.assertGreater(value, 0, key)

    def test_building_components_are_claimed_once(self):
        claims = self.explicit_claims()
        ids = [c["id"] for _, c in claims]
        self.assertEqual(len(ids), len(set(ids)))
        for owner, claim in claims:
            with self.subTest(owner=owner, component=claim["id"]):
                self.assertIn(claim["id"], self.components)
                self.assertTrue(near(claim["at"], self.components[claim["id"]]))
        rules = [(a["id"], a["rule"]) for a in self.assets if "rule" in a]
        for component in self.components.values():
            if component["id"] not in ids:
                with self.subTest(component=component["id"]):
                    self.assertEqual(len([owner for owner, rule in rules if matches(rule, component)]), 1)

    def test_village_wings_join_touching_parents(self):
        claimed = {c["id"] for _, c in self.explicit_claims()}
        winged = [a for a in self.assets if "parents" in a]
        self.assertTrue(winged)
        for asset in winged:
            wings = {c["id"] for c in self.components.values()
                     if c["id"] not in claimed and matches(asset["rule"], c)}
            parents = {int(wing): parent for wing, parent in asset["parents"].items()}
            self.assertEqual(set(parents), wings)
            for wing, parent in parents.items():
                with self.subTest(wing=wing):
                    self.assertIn(parent, self.components[wing]["touching"])

    def test_castle_pieces_are_claimed_once(self):
        family = {a["id"]: a["family"] for a in self.assets}
        roles = [(a["id"], p["role"]) for a in self.assets for p in a.get("castle_pieces", []) if "role" in p]
        self.assertEqual(sorted(role for _, role in roles), sorted(ROLE_CLAIMS))
        self.assertEqual({family[owner] for owner, _ in roles}, {"castle"})
        spots = [(a["id"], p["at"]) for a in self.assets for p in a.get("castle_pieces", []) if "at" in p]
        for piece in self.pieces:
            with self.subTest(piece=piece["id"], role=piece["role"]):
                hits = [owner for owner, at in spots if near(at, piece)]
                self.assertEqual(len(hits), 0 if piece["role"] in ROLE_CLAIMS else 1)
        for owner, at in spots:
            with self.subTest(owner=owner, at=at):
                self.assertEqual(sum(near(at, piece) for piece in self.pieces), 1)

    def test_map_labels_are_dispositioned(self):
        maps = {k: v for k, v in self.scene["sources"].items() if v["kind"] == "map"}
        self.assertTrue(maps)
        claimed = {label for a in self.assets for label in a.get("labels", [])}
        excluded = [e["label"] for e in self.scene["excluded"]["labels"]]
        self.assertEqual(len(excluded), len(set(excluded)))
        self.assertFalse(claimed & set(excluded))
        self.assertEqual(claimed | set(excluded), {f"{m}:{label}" for m, v in maps.items() for label in v["key"]})

    def test_references_resolve(self):
        sources, decisions, images = self.scene["sources"], self.scene["decisions"], self.scene["images"]
        pages = {k for k, v in sources.items() if v["kind"] == "page"}
        maps = {k for k, v in sources.items() if v["kind"] == "map"}
        evidence = [(a["id"], e) for a in self.assets for e in a["sources"]]
        evidence += [(d, e) for d, v in decisions.items() for e in v["basis"]]
        for owner, entry in evidence:
            with self.subTest(owner=owner, source=entry.get("source")):
                self.assertIn(entry["source"], sources)
                self.assertTrue(EVIDENCE & set(entry))
                if "quote" in entry:
                    self.assertTrue(1 <= len(entry["quote"].split()) <= MAX_QUOTE_WORDS)
        for asset in self.assets:
            with self.subTest(asset=asset["id"]):
                self.assertLessEqual(set(asset.get("decisions", [])), set(decisions))
                self.assertLessEqual(set(asset.get("images", [])), set(images))
                for trace in asset.get("traced", []):
                    self.assertIn(trace["frame"], maps)
        for name, decision in decisions.items():
            with self.subTest(decision=name):
                self.assertTrue(decision["text"] and decision["basis"])
                self.assertLessEqual(set(decision.get("images", [])), set(images))
        excluded = self.scene["excluded"]
        for entry in excluded["labels"] + excluded["components"]:
            with self.subTest(excluded=entry.get("label", entry.get("id"))):
                self.assertTrue(entry.get("reason") or entry.get("decision") in decisions)
                if "decision" in entry:
                    self.assertIn(entry["decision"], decisions)
        for name, image in images.items():
            with self.subTest(image=name):
                self.assertTrue(image["pages"])
                self.assertLessEqual(set(image["pages"]), pages)

    def test_inputs_resolve(self):
        used = {name for a in self.assets for name in a.get("inputs", [])}
        for name in used:
            with self.subTest(input=name):
                head, _, rest = name.partition(".")
                if head == "meta":
                    node = self.data["meta"]
                    for part in rest.split("."):
                        self.assertIn(part, node)
                        node = node[part]
                elif rest:
                    self.assertIn(rest, CHANNELS.get(head, ()))
                else:
                    self.assertIn(head, self.data["tex"])
        self.assertLessEqual(REQUIRED_INPUTS, used)

    def test_images_have_stances(self):
        images = self.scene["images"]
        used = {name for a in self.assets for name in a.get("images", [])}
        for name, image in images.items():
            with self.subTest(image=name):
                self.assertIn(image["stance"], STANCES)
                if image["stance"] in GUIDING:
                    self.assertTrue(image["carry"])
                    self.assertIn(name, used)
                if image["stance"] == "negative":
                    self.assertTrue(image["avoid"])
        excluded = self.scene["excluded"]["images"]
        self.assertFalse({e["name"] for e in excluded} & set(images))
        self.assertTrue(all(e["reason"] for e in excluded))


if __name__ == "__main__":
    unittest.main()
