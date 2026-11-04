"""tools.timing.windows: the probe's windows keep their places on a changed tour."""
import contextlib
import io
import unittest
from unittest import mock

import numpy as np

from tools.test_stages import state
from tools.timing import windows


def path(xs, zs=None):
    return np.stack([xs, np.zeros_like(xs), np.zeros_like(xs) if zs is None else zs], axis=1)


class RemapTests(unittest.TestCase):
    def test_a_window_moves_to_where_the_new_tour_passes_its_point(self):
        # the new tour lingers 10 s at the start, then flies the same line: every later window starts 10 s later
        old = path(np.arange(0.0, 400.0, 1.0))
        new = path(np.concatenate([np.zeros(10), np.arange(0.0, 400.0, 1.0)]))
        moved = windows.remap({"a": 50.0, "b": 300.0}, old, 400.0, new, 410.0, step=1.0)
        self.assertEqual(moved, {"a": (60.0, 0.0), "b": (310.0, 0.0)})

    def test_a_tour_that_passes_a_place_twice_keeps_each_windows_pass(self):
        # out and back along one line: x = 50 is passed at 50 s and again at 349 s
        line = path(np.concatenate([np.arange(0.0, 200.0, 1.0), np.arange(199.0, -1.0, -1.0)]))
        moved = windows.remap({"out": 50.0, "back": 349.0}, line, 400.0, line, 400.0, step=1.0)
        self.assertEqual(moved, {"out": (50.0, 0.0), "back": (349.0, 0.0)})

    def test_a_window_whose_place_the_new_tour_does_not_pass_fails_the_run(self):
        # as when WINDOWS already hold the new tour's times: a window that far from its place would time another view.
        # The new tour flies the old line for 200 s, then a parallel one 100 m off.
        t = np.arange(0.0, 400.0, windows.tour.STEP_S)
        old, new = path(t), path(t, np.where(t < 200.0, 0.0, 100.0))

        def run(probe_windows):
            tracks = mock.AsyncMock(return_value=((400.0, old), (400.0, new)))
            with state(), mock.patch.object(windows, "tracks", tracks), \
                    mock.patch.object(windows, "WINDOWS", probe_windows), contextlib.redirect_stdout(io.StringIO()):
                return windows.main(["old.html", "new.html"])
        self.assertEqual(run({"early": 50.0, "mid": 150.0}), 0)
        self.assertEqual(run({"early": 50.0, "late": 300.0}), 1)


if __name__ == "__main__":
    unittest.main()
