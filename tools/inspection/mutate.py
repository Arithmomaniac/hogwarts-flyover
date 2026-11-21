"""Mutations of the inspection's tools that tools/test_inspection.py must catch, and of the frozen baseline's
comparison that tools/test_frozen_baseline.py must catch.

    python -X utf8 -m tools.inspection.mutate [NAME-SUBSTRING]

Each mutation runs in a temporary copy of the worktree (tools.site.mutate.check), whose tests must first pass
unmutated, so a failure that is not the mutation's reports INVALID rather than caught. Exit status 1 when any mutation
survives or is invalid.
"""
import sys

from tools import locks
from tools.site.mutate import Mutation, check

T = "tools.test_inspection."
FROZEN = "tools.test_frozen_baseline.FrozenBaselineTests"
BASELINE = "tools/site/baseline.py"
INVENTORY, FRAME, SHOOT, REVIEW, CONTACT, MOTION = (f"tools/inspection/{m}.py" for m in
                                                    ("inventory", "frame", "shoot", "review", "contact", "motion"))
MUTATIONS = [
    Mutation("decisions shown by no picture counted as uncovered", INVENTORY,
             "decisions = {d for o in objects for d in o.decisions if d not in AFTER_DARK or after_dark(o)} | "
             "set(NOT_VISUAL)",
             "decisions = {d for o in objects for d in o.decisions if d not in AFTER_DARK or after_dark(o)}",
             (T + "CoverageTests.test_every_decision_is_on_an_object_or_named_as_not_visual",)),
    Mutation("a shot by day covers what only the dark shows", INVENTORY,
             "    return hour in DARK_HOURS if isinstance(hour, str) else hour >= DUSK\n", "    return True\n",
             (T + "CoverageTests.test_a_decision_only_the_dark_shows_is_covered_only_by_an_object_shot_after_dark",)),
    Mutation("the unbuilt hamlet's absence left out", INVENTORY,
             'obj("hamlet", "The unbuilt hamlet on the hill north-west of the Shack", "absence",',
             'obj("hamlet", "The unbuilt hamlet on the hill north-west of the Shack", "terrain",',
             (T + "CoverageTests.test_every_footprint_group_and_excluded_group_has_an_object",)),
    Mutation("a captioned waypoint without its subjects", INVENTORY, '"Dumbledore\'s tomb": ("grounds/tomb",),', "",
             (T + "CoverageTests.test_every_waypoint_has_a_still_and_every_caption_its_subjects",)),
    Mutation("a tree kit's variant read as its species, not its form", INVENTORY,
             'return name.rsplit("-", 1)[0]', 'return name.split("-")[0]',
             (T + "CoverageTests.test_each_form_of_the_tree_kit_has_a_tree_with_the_forest_trees_decisions",)),
    Mutation("the trees take only the decisions their kit has today", INVENTORY,
             'decisions = tuple(d for d in decisions_of(world, "forest-trees") if d not in NOT_VISUAL)',
             'decisions = ("trees-species", "trees-birch", "trees-form", "trees-in-leaf", "trees-colours", '
             '"trees-reach")',
             (T + "CoverageTests.test_each_form_of_the_tree_kit_has_a_tree_with_the_forest_trees_decisions",)),
    Mutation("the largest building represents its family", INVENTORY,
             'rep = min(us, key=lambda u: (abs(u["area"] - mid), key(u)))',
             'rep = max(us, key=lambda u: (u["area"], key(u)))', (T + "SamplingTests",)),
    Mutation("what stands in front of the subject not counted as hiding it", FRAME,
             "front = through & (t < enter - margin)", "front = through & (t < enter - 100 * margin)",
             (T + "FrameTests",)),
    Mutation("the scored cube reaches under the ground", SHOOT, "max(y - s, g - 0.5)", "y - s", (T + "ShootTests",)),
    Mutation("a defect routed on another object's image", REVIEW,
             "if not evidence or any(name not in images.get(oid, ()) for name in cited):",
             "if not evidence:", (T + "ReviewTests",)),
    Mutation("the contact sheet's columns in the wrong order", CONTACT,
             "tuple(files.get(role) for role in ROLES)", "tuple(files.get(role) for role in reversed(ROLES))",
             (T + "ContactTests",)),
    Mutation("a thumbnail pasted under another column", CONTACT, "            x, y = cell_origin(r, c)\n",
             "            x, y = cell_origin(r, len(row.cells) - 1 - c)\n", (T + "ContactTests",)),
    Mutation("pops measured against the largest change", MOTION,
             "    limit = factor * max(statistics.median(series), floor)\n",
             "    limit = factor * max(max(series), floor)\n", (T + "MotionTests",)),
    Mutation("the page's loop keeps running between motion frames", MOTION,
             'LOOP + "    if (motionStop) return;\\n"', "LOOP", (T + "MotionTests",)),
    Mutation("a repeated caption taken at whichever pass", MOTION, "    if occurrence is None and len(hits) > 1:\n",
             "    if False:\n", (T + "MotionTests",)),
    # the acceptance case the qualification adds a test for: tools/test_frozen_baseline.py
    Mutation("a stale frozen baseline accepted", BASELINE, "    if found:\n", "    if False:\n", (FROZEN,)),
    Mutation("the frozen page re-frozen without its lake-relative levels", BASELINE,
             "        page = page.replace(old, new)\n", "        page = page\n", (FROZEN,)),
]


def main(argv):
    clean, bad = {}, 0
    with locks.heavy("inspection mutation checks (tools.inspection.mutate)"):
        for m in MUTATIONS:
            if argv and not any(a in m.name for a in argv):
                continue
            if m.tests not in clean:
                clean[m.tests] = check(m, mutate=False)
            verdict = ("caught" if check(m) == "fail" else "SURVIVED") if clean[m.tests] == "pass" else \
                "INVALID (its tests fail without the mutation)"
            bad += verdict != "caught"
            print(f"{m.name}: {verdict}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
