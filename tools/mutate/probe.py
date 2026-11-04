"""Mutations of the frame-time probe that its tests must catch."""
from tools.mutate.run import Mutation

TESTS = ("test_mesh_probe",)
MUTATIONS = [
    Mutation("loop keeps running", "mesh_probe.py", 'LOOP + "    if (probeStop) return;\\n"', "LOOP", TESTS, rebuild=False),
    Mutation("plain rotations", "mesh_probe.py", "[[(v + r) % n for v in first]", "[[(v + r) % n for v in range(n)]",
             TESTS, rebuild=False),
    Mutation("quality not applied", "mesh_probe.py", "qual = s.q; ", "", TESTS, rebuild=False),
    Mutation("no completion wait", "mesh_probe.py", "        await probeDone();\n        batches.push",
             "        batches.push", TESTS, rebuild=False),
    Mutation("anchor check skipped", "mesh_probe.py", "if any(text.count(anchor) != 1",
             "if False and any(text.count(anchor) != 1", TESTS, rebuild=False),
]
