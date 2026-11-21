"""The tree species kit's colours (mesh_trees.py). The trees draw as the tree record, material 3, not as set pieces, so
the family reserves no codes; the renderer uploads these colours as uniforms and varies them per instance.

Linear albedo, each the sRGB of a typical sunlit surface in summer converted to linear. A tree's bark mixes from `bark`
at its foot to `bark_upper` over its stem by the bark vertices' aux (mesh_trees.py): a Scots pine's grey-brown foot
turns orange-red up the stem, and a birch's dark, fissured foot white. Every tree is in leaf, as the scene is, so no
broadleaf turns: `autumn_share` is 0."""

FAMILY = "trees"
RANGES = ()
MATERIALS = ()


def _linear(r, g, b):
    def one(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return tuple(round(one(float(c)), 4) for c in (r, g, b))


COLOURS = {
    "scots-pine": {"bark": _linear(92, 78, 66), "bark_upper": _linear(165, 96, 58), "foliage": _linear(50, 68, 48),
                   "autumn_share": 0.0},
    "yew": {"bark": _linear(104, 62, 46), "bark_upper": _linear(118, 66, 46), "foliage": _linear(30, 48, 28),
            "autumn_share": 0.0},
    "oak": {"bark": _linear(84, 77, 68), "bark_upper": _linear(92, 84, 72), "foliage": _linear(58, 80, 36),
            "autumn_share": 0.0},
    "beech": {"bark": _linear(128, 126, 118), "bark_upper": _linear(140, 137, 126), "foliage": _linear(66, 92, 36),
              "autumn_share": 0.0},
    "birch": {"bark": _linear(66, 62, 58), "bark_upper": _linear(206, 202, 190), "foliage": _linear(86, 108, 48),
              "autumn_share": 0.0},
}
