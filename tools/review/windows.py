"""Lamplight review views: lit windows at civil dusk and at night, their reflection, and the eased loop."""
from tools.review.sheet import MESHES

COLUMNS = [("lamplight", MESHES)]
VIEWS = [
    ("Station at dusk, 21:30", 332.0, 8.0, 132.0, 340.7, 3.0, 110.2, 21.5),
    ("The ship from the bay, 21:30", -67.4, 10.0, -353.7, -59.0, 12.0, -262.0, 21.5),
    ("West front at civil dusk", -230.0, 45.0, -40.0, -30.0, 25.0, -85.0, "civil dusk"),
    ("West front at night", -230.0, 45.0, -40.0, -30.0, 25.0, -85.0, "night"),
    ("Castle and its reflection across the lake, night", None, 6.0, None, 0.0, 30.0, 0.0, "night"),
    ("Engine on the eased curve, 16:00", 455.0, 5.0, 15.0, 420.0, 2.0, 30.0, 16.0),
    ("The loop from the south-east, 15:00", 650.0, 220.0, -150.0, 330.0, 0.0, 150.0, 15.0),
]
