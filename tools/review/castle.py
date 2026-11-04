"""Castle review views; compare passes with --before."""
from tools.review.sheet import MESHES

COLUMNS = [("castle", MESHES)]
VIEWS = [
    ("West front and Astronomy Tower, 18:30", -230, 45, -40, -30, 25, -85, 18.5),
    ("West range facade, 60 m", -165, 22, 30, -105, 18, 20, 16.0),
    ("Ravenclaw Tower", -185, 55, 190, -100, 40, 143, 16.0),
    ("Great Hall from the south-east", 95, 60, -185, 7, 20, -98, 15.0),
    ("West front at dusk, 21:00", -230, 45, -40, -30, 25, -85, 21.0),
    ("Whole castle from 700 m, 18:30", 420, 140, -620, 0, 30, 0, 18.5),
    ("Main doors and their steps, 17:00", -44, 9, -84, -26, 1.5, -77, 17.0),
    ("The pend from the broomshed court, 16:00", -93, 4, 16, -88, 3, 3, 16.0),
    ("Head's tower and Physick Garden, 15:00", -112, 32, 74, -85, 8, 45, 15.0),
]
