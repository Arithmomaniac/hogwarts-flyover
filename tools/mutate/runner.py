"""Runs unittest names as `python -m unittest` does from the current folder, leaving out the tests that their modules
list in EXPECTED_FAILURES: those fail with or without a mutation, so they say nothing about one. Exit status 0 when
every other test passes, 1 when one fails or errors, NO_TESTS when the names hold no other test, and BLOCKED when none
fails but one was skipped as blocked (a page test whose Edge renders on another GPU than HOG_GPU names), which judges
nothing.

tools.mutate.run starts it in a copy of src/. It imports nothing from the repository, because tools.paths would put the
worktree's own src/ ahead of the copy's mutated one."""
import os
import sys
import unittest

NO_TESTS = 5
BLOCKED = 6


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


def listed(test):
    module = type(test).__module__
    return test.id().removeprefix(f"{module}.") in getattr(sys.modules.get(module), "EXPECTED_FAILURES", ())


def main(names):
    sys.path[0] = os.getcwd()          # as for `python -m unittest`: the copy's src/, not this script's folder
    kept = [t for t in flatten(unittest.defaultTestLoader.loadTestsFromNames(names)) if not listed(t)]
    if not kept:
        return NO_TESTS
    result = unittest.TextTestRunner().run(unittest.TestSuite(kept))
    if not result.wasSuccessful():
        return 1
    return BLOCKED if any(reason.startswith("blocked:") for _, reason in result.skipped) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
