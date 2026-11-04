"""The payload rebuilt from the sources alone, without the S1 part cache, equals the one the stages built. The staleness
gate rebuilds through the cache, so a part key that missed an input would let it pass on a stale part; this catches
that. It takes about a minute and a half, so the trunk keeper runs it at merges (`doit uncached`) and lanes' S3 gates
leave it out."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class UncachedBuildTests(unittest.TestCase):
    def test_checked_in_payload_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "meshes.js"
            subprocess.run([sys.executable, str(ROOT / "src" / "build_meshes.py"), str(output)],
                           check=True, capture_output=True)
            self.assertEqual(output.read_bytes(), (ROOT / "build" / "meshes.js").read_bytes())


if __name__ == "__main__":
    unittest.main()
