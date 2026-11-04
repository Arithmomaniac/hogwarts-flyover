"""The pass timer's page mode times the baseline and the candidate in both load orders. On the AMD iGPU a second live
page runs slower whatever it is (8.5-20.7% by window in integration's A/A test), so one page timed against itself must
come out even: the two orders' geometric mean cancels a slowdown both pages share."""
import contextlib
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.paths import PAGE  # noqa: E402
from tools.timing import passes  # noqa: E402

SECOND = 1.2                    # how much slower a browser runs the page it opened second
PASS_MS = {"p1": 4.0, "shade": 2.0, "mesh": 1.0}


class FakePage:
    """Answers the probe's adapter and the pass timer, every time SECOND times longer if its browser opened it second."""

    def __init__(self, second):
        self.factor, self.settings = SECOND if second else 1.0, None

    async def bring_to_front(self):
        pass

    async def evaluate(self, expression, arg=None):
        if "__probe.settings" in expression:
            self.settings = arg
        elif "__probe.block" in expression:
            frames, in_flight = arg
            return {"batches": [10.0 * self.factor] * (len(frames) // in_flight), "drainMs": 0.0,
                    "rendered": len(frames), "loopFrames": 0, "snapshot": {**self.settings, "visible": "visible"}}
        elif expression == "__pass.collect()":
            return {"frames": 16, "disjoint": False, "ms": {k: v * self.factor for k, v in PASS_MS.items()}, "cpu": {}}
        return None


def fake_sessions(opened):
    @contextlib.asynccontextmanager
    async def session(gpu, pages):
        opened.append(tuple(role for role, _ in pages))
        yield {role: FakePage(second=k > 0) for k, (role, _) in enumerate(pages)}
    return session


class PagesTests(unittest.IsolatedAsyncioTestCase):
    async def ratios(self, opened):
        """The candidate's times over the baseline's, meshes off in both, for one page timed as both: each pass's and
        the wall time, in every window."""
        out = await passes.time_pages("amd", PAGE, PAGE, session=fake_sessions(opened))
        found = []
        for window in passes.PAGE_WINDOWS:
            candidate, baseline = out[f"{window}|landscape"], out[f"{window}|baseline"]
            found += [candidate["ms"][k] / baseline["ms"][k] for k in PASS_MS]
            found.append(candidate["wallMs"] / baseline["wallMs"])
        return found

    async def test_one_page_timed_against_itself_comes_out_even(self):
        with mock.patch.object(passes.mp, "LOAD_ORDERS", (("baseline", "candidate"),)):
            one_order = await self.ratios([])
        for ratio in one_order:          # the fake browser slows the page it opened second, as the AMD iGPU does
            self.assertAlmostEqual(ratio, SECOND)
        opened = []
        ratios = await self.ratios(opened)
        for ratio in ratios:
            self.assertAlmostEqual(ratio, 1.0)
        self.assertEqual(sorted(opened), sorted(passes.mp.LOAD_ORDERS))


if __name__ == "__main__":
    unittest.main()
