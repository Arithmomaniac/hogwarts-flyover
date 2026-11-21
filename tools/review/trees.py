"""Mesh trees review views: the raymarched trees (?trees=raymarched) beside the mesh trees as the page ships them (the
payload kit's species near the camera, farther for trees that stand apart, a lawn's, with the sparse dial on as the page
opens; the drawn crowns beyond, to 2.6 km, and the raymarched trees' canopy sheet farther), and the drawn crowns alone
(?treedetail=0, no species meshes). The tour's views are its waypoints' cameras, numbered as the inspection numbers them
(tour/NN)."""
from tools.review.sheet import MESHES

COLUMNS = [("Raymarched trees (trees=raymarched)", MESHES + "&trees=raymarched"),
           ("Mesh trees: species near, drawn crowns to 2.6 km, the sheet beyond", MESHES + "&trees=mesh"),
           ("Mesh trees: drawn crowns only (treedetail=0)", MESHES + "&trees=mesh&treedetail=0")]
VIEWS = [
    ("Tour 4: Gryffindor Tower, 15:00", 40.0, 55.0, -262.0, -2.0, 30.0, -166.0, 15.0),
    ("Tour 8: the greenhouses, 15:00", 30.0, 26.0, 305.0, -20.0, 2.0, 245.0, 15.0),
    ("Tour 10: the Forbidden Forest (site/forest-above), 15:00", -420.0, 110.0, 760.0, -900.0, 0.0, 1500.0, 15.0),
    ("Tour 10 again, 18:30", -420.0, 110.0, 760.0, -900.0, 0.0, 1500.0, 18.5),
    ("Tour 11: Aragog's hollow, 15:00", -760.0, 120.0, 1880.0, -581.0, 0.0, 1739.0, 15.0),
    ("Tour 12: Grawp's clearing, 15:00", -1620.0, 110.0, 730.0, -1700.0, 0.0, 650.0, 15.0),
    ("Tour 13: Sirius's cave, by the tree table's edge, 15:00", -1900.0, 250.0, 350.0, -5018.0, 0.0, 888.0, 15.0),
    ("Tour 14: the Entrance Hall and Great Hall, 15:00", -330.0, 60.0, -68.0, -18.0, 40.0, -72.0, 15.0),
    ("Tour 15: the West Tower, 15:00", -260.0, 100.0, 10.0, -107.0, 35.0, -9.0, 15.0),
    ("Tour 16: Hagrid's hut and the Forest's edge, 15:00", -340.0, 26.0, -205.0, -291.5, 3.0, -191.8, 15.0),
    ("The station wood from the platform, 15:00", 290.0, 2.0, 160.0, 0.0, 30.0, 0.0, 15.0),
    ("The castle and its reflection across the lake, 15:00", None, 6.0, None, 0.0, 30.0, 0.0, 15.0),
    ("Trees close, from the south-west (renderer/trees), 15:00", -997.0, 51.0, 0.0, -951.0, 14.0, 46.0, 15.0),
]
