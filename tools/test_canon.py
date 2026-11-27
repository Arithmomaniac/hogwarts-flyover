"""tools.canon: the checker passes on canon.json, and fails on each kind of break it exists to catch. Like the manifest
checker, it reads White Hound's pages from the pinned cache, so these tests skip without it."""
import contextlib
import copy
import io
import unittest

from tools import canon
from tools.inputs import Blocked


def failures(doc, text):
    return [f for f in canon.check_doc(doc).fails if text in f]


class CanonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            canon.grams()
        except Blocked as e:
            raise unittest.SkipTest(f"blocked: {e}") from e
        cls.doc = canon.load(canon.CANON)

    def setUp(self):
        self.doc = copy.deepcopy(self.__class__.doc)

    def test_the_register_passes(self):
        self.assertEqual(canon.check_doc(self.doc).fails, [])

    def test_a_tampered_span_hash_fails(self):
        self.doc["rows"][0]["quote"][0]["sha"] = "0" * 12
        self.assertTrue(failures(self.doc, "span hash differs"))

    def test_an_unaccounted_item_fails(self):
        passage = next(k for k, it in self.doc["accounted"].items() if it["kind"] == "passage")
        del self.doc["accounted"][passage]
        self.assertTrue(failures(self.doc, "unaccounted: no item for the passage"))

    def test_six_words_of_a_page_in_a_string_field_fail(self):
        line = next(l for l in canon.page_lines("wh-grounds") if len(l.split()) > 12 and "[" not in l)
        self.doc["rows"][0]["assertion"] = " ".join(line.split()[:7])
        self.assertTrue(failures(self.doc, "no-text scan"))

    def test_a_row_disposition_without_a_locator_on_the_row_fails(self):
        item = next(it for it in self.doc["accounted"].values()
                    if it["kind"] == "passage" and it["disposition"].startswith("row:"))
        loc = item["locator"]
        elsewhere = next(r["id"] for r in self.doc["rows"]
                         if not any(x["page"] == loc["page"] and x["lines"][0] <= loc["lines"][1] and loc["lines"][0] <= x["lines"][1]
                                    for x in r["quote"] + r["wh"]))
        item["disposition"] = f"row:{elsewhere}"
        self.assertTrue(failures(self.doc, "has no locator for it"))

    def test_a_waiver_needs_the_user(self):
        row = next(r for r in self.doc["rows"] if r["holds"] is False)
        row["waiver"] = {"by": "orchestrator", "reason": "x"}
        self.assertTrue(failures(self.doc, "a waiver is set only by the user"))
        row["waiver"] = {"by": "user", "reason": "x"}
        self.assertFalse(failures(self.doc, "a waiver is set only by the user"))

    def test_an_unknown_decision_id_fails(self):
        row = self.doc["rows"][0]
        row["object"]["decisions"] = ["no-such-decision"]
        row["reading"]["basis"] = ["decision:no-such-decision"]
        self.assertEqual(len(failures(self.doc, "no-such-decision")), 2)

    def test_show_prints_from_the_cache_with_a_wayback_link_and_writes_nothing(self):
        before = canon.CANON.read_bytes()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            canon.cmd_show(self.doc["rows"][0]["id"])
        self.assertIn(canon.WAYBACK, out.getvalue())
        self.assertEqual(canon.CANON.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
