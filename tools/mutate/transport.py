"""Mutations of the transport builders that the transport tests must catch."""
from tools.mutate.run import Mutation

TRANSPORT = "test_assets_transport.TransportTests."
MUTATIONS = [
    Mutation("coupling stretched across the gap", "mesh_transport.py",
             "            parts.append(placed(wheelset(r, SCARLET if v.kind == \"engine\" and r > 0.9 else IRON), ra, sa))\n"
             "    return parts\n",
             "            parts.append(placed(wheelset(r, SCARLET if v.kind == \"engine\" and r > 0.9 else IRON), ra, sa))\n"
             "    from mesh_geometry import bar\n"
             "    q0, q1 = (railway.point_at(xz, p) for p in (vehicles[10].s1, vehicles[11].s0))\n"
             "    parts.append(bar([q0[0], rail_top + 1.0, q0[1]], [q1[0], rail_top + 1.0, q1[1]], 0.12, IRON))\n"
             "    return parts\n",
             (TRANSPORT + "test_couplings_join_only_the_coupled_vehicles",)),
    Mutation("engine facing the coaches", "mesh_transport.py",
             "    parts += buffers(nose, 1.0)\n    return concat(parts)\n\n\ndef train(al):",
             "    parts += buffers(nose, 1.0)\n    m = concat(parts)\n"
             "    return Mesh(m.pos * [-1.0, 1.0, 1.0], m.nrm * [-1.0, 1.0, 1.0], m.attr, m.idx[:, [0, 2, 1]])\n\n\n"
             "def train(al):",
             (TRANSPORT + "test_the_engine_leads_away_from_the_coaches",)),
    Mutation("canopy over the whole platform", "mesh_transport.py",
             "CANOPY_DEPTH_M, CANOPY_LOW_M, CANOPY_REACH_M = 4.2, 2.9, 7.0",
             "CANOPY_DEPTH_M, CANOPY_LOW_M, CANOPY_REACH_M = 4.2, 2.9, 200.0",
             (TRANSPORT + "test_the_canopy_covers_only_part_of_the_platform",)),
    Mutation("bowsprit into the shore", "mesh_transport.py",
             "        if ground is None or clear(ground, [sprit, (foretop, sprit[1])], SPRIT_CLEAR_M):",
             "        if True:",
             (TRANSPORT + "test_nothing_of_the_ship_above_the_water_runs_into_the_shore",)),
    Mutation("a barge's square stern", "mesh_transport.py",
             "STERN_TAPER = 0.5 ", "STERN_TAPER = 0.2 ",
             (TRANSPORT + "test_the_hull_narrows_aft_to_a_transom",)),
    Mutation("blank gables", "mesh_transport.py",
             "        if GABLE_SILL_M + GABLE_WINDOW_H > rise * (1 - GABLE_WINDOW_W / 2 / half) - 0.2:",
             "        if True:",
             (TRANSPORT + "test_each_gable_has_a_lit_window_over_the_canopy",)),
    Mutation("the loop in a sheer rock cut", "railway.py",
             "Feature(al.loop.xz, al.loop.level, FORMATION_M, LOOP_CUT, FILL)",
             "Feature(al.loop.xz, al.loop.level, FORMATION_M, CUT, FILL)",
             ("test_assets_transport.RailwayTests.test_cuttings_and_embankments_meet_the_ground_at_their_slopes",),
             rebuild=False),
    Mutation("the formation laid bare", "fix_railway.py",
             "    cls[onbed & (cls == FOREST)] = MEADOW\n", "    cls[onbed] = EARTH\n",
             ("test_assets_transport.RailwayTests.test_the_formation_beside_the_ballast_is_grass_not_bare_earth",),
             rebuild=False),
]
