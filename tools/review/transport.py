"""Station, train and ship review views; compare passes with --before."""
from tools.review.sheet import MESHES

COLUMNS = [("transport", MESHES)]
VIEWS = [
    ("Along the platform from its far end, 16:00", 366.0, 3.1, 92.7, 282.4, 2.5, 185.7, 16.0),
    ("The building and canopy from the road, 15:00", 318.1, 6.0, 96.3, 340.7, 3.0, 110.2, 15.0),
    ("Engine and tender, detached on the curve, 16:00", 455.0, 5.0, 15.0, 420.0, 2.0, 30.0, 16.0),
    ("Coaches from across the track, 14:00", 337.7, 4.0, 154.9, 327.4, 2.0, 145.4, 14.0),
    ("Junction cutting and the tunnel portal, 15:00", 226.0, 6.0, 252.0, 148.0, 5.0, 287.0, 15.0),
    ("The Durmstrang ship from the shore, 18:30", -20.0, 8.0, -175.0, -59.0, 15.0, -262.0, 18.5),
    ("The ship at dusk, 21:30", -20.0, 8.0, -175.0, -59.0, 15.0, -262.0, 21.5),
    ("The station at dusk, 21:30", 332.0, 8.0, 132.0, 340.7, 3.0, 110.2, 21.5),
    ("The loop from the south-east, 15:00", 650.0, 220.0, -150.0, 330.0, 0.0, 150.0, 15.0),
]
