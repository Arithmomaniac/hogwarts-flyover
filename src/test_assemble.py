"""The modular sources must reproduce the standalone deliverable."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assemble  # noqa: E402
import meshopt  # noqa: E402

class AssemblyTests(unittest.TestCase):
    def test_sources_reproduce_standalone_page(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "flyover.html"
            subprocess.run(
                [sys.executable, str(ROOT / "src" / "assemble.py"), str(output)],
                check=True, capture_output=True,
            )
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(),
                hashlib.sha256((ROOT / "hogwarts-flyover.html").read_bytes()).hexdigest(),
            )
            page = output.read_text(encoding="utf-8")
            self.assertNotIn("/*__", page)
            self.assertNotIn("<!--__", page)

    def test_the_page_inlines_the_pinned_meshoptimizer_decoder(self):
        decoder = (meshopt.VENDOR / "meshopt_decoder.cjs").read_text(encoding="utf-8").strip()
        self.assertIn(f"<script>{decoder}</script>", (ROOT / "hogwarts-flyover.html").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            tampered = Path(folder) / "meshopt_decoder.cjs"
            tampered.write_text(decoder.replace("Malformed buffer data", "Malformed buffer"), encoding="utf-8")
            with mock.patch.object(meshopt, "VENDOR", Path(folder)), self.assertRaises(ValueError):
                assemble.page()


if __name__ == "__main__":
    unittest.main()
