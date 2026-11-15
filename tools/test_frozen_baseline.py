"""The plan's baseline-assembly acceptance case, at the site: the frozen page re-frozen here (baseline-x, made by
tools.site.baseline) comes back exactly from the sources it is made of, and stale sources fail its comparison with the
pin."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from tools import inputs
from tools.paths import BUILD, PAGE
from tools.site import baseline


class FrozenBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            inputs.require("baseline-s4-idle")
        except inputs.Blocked as e:                     # the frozen page is a pinned input, fetched into build/inputs
            raise unittest.SkipTest(str(e))

    def build(self, folder, *argv):
        with contextlib.redirect_stdout(io.StringIO()):
            return baseline.main([str(Path(folder) / "baseline-x.html"), *argv])

    def test_the_sources_reproduce_the_pinned_frozen_page(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.build(d), 0)

    def test_stale_sources_fail_the_comparison(self):
        data = (BUILD / "data.js").read_bytes().decode("ascii")
        page = PAGE.read_bytes().decode("utf-8")
        tour = baseline.tour_text(page)
        k = next(k for k, ch in enumerate(tour) if ch.isdigit())
        stale_tour = tour[:k] + str((int(tour[k]) + 1) % 10) + tour[k + 1:]          # a waypoint moved
        k = next(k for k in range(len(data) // 2, len(data)) if data[k].isdigit())
        stale_data = data[:k] + str((int(data[k]) + 1) % 10) + data[k + 1:]          # one value changed
        with tempfile.TemporaryDirectory() as d:
            copies = {"page.html": page, "data.js": data, "stale.html": page.replace(tour, stale_tour, 1),
                      "stale.js": stale_data}
            # byte for byte, as the tool reads them: on Windows, write_text's CRLF alone would fail the pin
            for name, text in copies.items():
                (Path(d) / name).write_bytes(text.encode("utf-8"))
            # the unchanged copies come back to the pin, so each stale copy fails it for its own change
            for stale, (page_copy, data_copy, want) in {"none": ("page.html", "data.js", 0),
                                                        "tour": ("stale.html", "data.js", 1),
                                                        "data": ("page.html", "stale.js", 1)}.items():
                with self.subTest(stale=stale):
                    argv = ("--page", str(Path(d) / page_copy), "--data", str(Path(d) / data_copy))
                    self.assertEqual(self.build(d, *argv), want)


if __name__ == "__main__":
    unittest.main()
