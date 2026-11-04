"""Railway earthworks review views (meshes off: the regraded terrain alone); compare with --before."""
from tools.review.sheet import LANDSCAPE

COLUMNS = [("terrain (meshes off)", LANDSCAPE)]
VIEWS = [
    ("Overview from the south-east, 15:00", 700, 260, -260, 330, 0, 150, 15.0),
    ("Junction cutting and the tunnel portal, from the coaches' end", 238, 14, 238, 150, 8, 287, 15.0),
    ("North-side rock cutting from above", 300, 70, 345, 290, 0, 262, 15.0),
    ("Embankment at the foot of the loop, from the south", 470, 18, -95, 460, 8, 18, 15.0),
    ("Station terrace from beyond the boundary wall", 195, 35, 125, 300, 0, 175, 15.0),
    ("The loop from the castle side (west), 18:30", -60, 120, 120, 330, 0, 150, 18.5),
]
