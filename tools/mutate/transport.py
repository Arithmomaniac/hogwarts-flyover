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
]
