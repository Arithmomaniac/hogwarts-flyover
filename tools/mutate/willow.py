"""Mutations of the Willow's generator that its asset tests must catch."""
from tools.mutate.run import Mutation

SKELETON = ("test_assets_willow.WillowSkeletonTests",)
BUILT = ("test_assets_willow.WillowTests",)
LODS = ("test_assets_willow.WillowLodTests",)
MUTATIONS = [
    Mutation("five limbs", "mesh_willow.py", "LIMBS = 7", "LIMBS = 5", SKELETON, rebuild=False),
    Mutation("no knuckles", "mesh_willow.py", "KNUCKLE = 1.35", "KNUCKLE = 1.0", SKELETON, rebuild=False),
    Mutation("unseeded", "mesh_willow.py", "    rng = np.random.default_rng(seed)\n",
             "    rng = np.random.default_rng()\n", SKELETON, rebuild=False),
    Mutation("gap mirrored north for south", "mesh_willow.py", "    return math.atan2(v[1], v[0])",
             "    return math.atan2(-v[1], v[0])", BUILT),
    Mutation("drooping limbs", "mesh_willow.py", "LIMB_RISE = (0.32, 0.44), (7.0, 9.5), (0.35, 0.9)",
             "LIMB_RISE = (0.32, 0.44), (7.0, 9.5), (-0.3, 0.1)", BUILT),
    Mutation("knot out of a cat's reach", "mesh_willow.py", "KNOT_M, KNOT_DEG = 0.4, 75.0", "KNOT_M, KNOT_DEG = 1.3, 75.0",
             BUILT),
    Mutation("knot behind a root", "mesh_willow.py", "KNOT_M, KNOT_DEG = 0.4, 75.0", "KNOT_M, KNOT_DEG = 0.4, 110.0",
             BUILT),
    Mutation("knot in plain bark", "mesh_willow.py", "(0.2, 0.15, 0.2), KNOT, np.random",
             "(0.2, 0.15, 0.2), BARK, np.random", BUILT),
    Mutation("gap too narrow for a dog and a boy", "mesh_willow.py", "GAP_HALF_DEG, RECESS = 65.0,",
             "GAP_HALF_DEG, RECESS = 35.0,", BUILT),
    Mutation("no arching root", "mesh_willow.py", "        if k == 0:", "        if k == -1:", BUILT),
    Mutation("root ends out of the trunk", "mesh_willow.py", "ROOT_START_M = 0.45", "ROOT_START_M = 1.0", SKELETON,
             rebuild=False),
    Mutation("lod1 at full detail", "mesh_willow.py", '"lod1": Level(2, 2, 0.5,', '"lod1": Level(3, 1, 1.0,', LODS,
             rebuild=False),
    Mutation("lod2 crown shrunk", "mesh_willow.py", "Level(1, 3, 0.0, 10, 0, False, 1.2,",
             "Level(1, 3, 0.0, 10, 0, False, 0.6,", LODS, rebuild=False),
    Mutation("shadow crown shrunk", "mesh_willow.py", "Level(2, 3, 0.0, 10, 1, False, 1.12, False, False)",
             "Level(2, 3, 0.0, 10, 1, False, 0.5, False, False)", LODS, rebuild=False),
]
