"""tools.review.sheet: a sheet renders in Edge for minutes, so a run holds its GPU's lock and counts as heavy work."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest import mock

from tools import locks, stages
from tools.review import sheet
from tools.test_stages import sandbox, state


def held(gpu):
    return (locks.folder() / f"gpu-{gpu}.json").is_file()


class SheetLockTests(unittest.TestCase):
    def test_a_hand_run_holds_its_gpus_lock_and_counts_as_heavy_work(self):
        # so a suite on the same GPU waits for it, and so does a timing run
        seen = []

        async def make(*args):
            seen.append((held("amd"), held("nvidia"), len(locks.heavy_work())))
            return Path("sheet.png")
        with state(), mock.patch.object(sheet, "make", side_effect=make), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sheet.main(["tour", "--gpu", "amd"]), 0)
            self.assertFalse(held("amd"))
        self.assertEqual(seen, [(True, False, 1)])

    def test_the_stage_leaves_both_locks_to_the_sheet_it_runs(self):
        # the sheet takes its GPU and then registers itself: a stage holding the GPU would leave its sheet waiting for
        # ever, and one registering around it would hold up a timing run while the sheet waits for the GPU
        seen = []

        def child(*args, log):
            seen.append((held("nvidia"), len(locks.heavy_work())))
            return 0
        with sandbox(), mock.patch.object(stages, "python", side_effect=child):
            verdict, _, _ = stages.review_stage("tour").run("0" * 64)
        self.assertEqual((verdict, seen), ("pass", [(False, 0)]))


if __name__ == "__main__":
    unittest.main()
