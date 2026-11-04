"""Village review views: the High Street, the streets beyond it, and the named buildings close up; compare passes with
--before."""
from tools.review.sheet import MESHES

COLUMNS = [("village", MESHES)]
VIEWS = [
    ("Down the High Street from Dervish & Banges, 15:00", -975.0, 2.5, -1063.0, -875.0, 4.0, -1155.0, 15.0),
    ("The Three Broomsticks and Zonko's, 16:00", -875.3, 2.0, -1173.8, -858.0, 4.0, -1168.0, 16.0),
    ("Honeydukes from across the High Street, 11:00", -748.3, 2.0, -1246.7, -753.0, 4.0, -1255.6, 11.0),
    ("The High Street from the air, 13:00", -960.0, 70.0, -1110.0, -860.0, 0.0, -1175.0, 13.0),
    ("The Hog's Head from its side street, 15:00", -752.6, 2.5, -1182.8, -759.8, 3.5, -1175.9, 15.0),
    ("The streets south and east of the High Street from the air, 13:00", -780.0, 90.0, -1250.0, -520.0, 0.0,
     -1450.0, 13.0),
    ("Down a lane east of the High Street, 15:00", -580.1, 2.5, -1367.8, -581.0, 4.0, -1400.0, 15.0),
    ("The post office's owl loft and Scrivenshaft's crow steps from above the High Street, 12:00", -825.5, 28.0,
     -1238.1, -804.6, 9.0, -1207.6, 12.0),
    ("Gladrags' blind, 15:00", -909.8, 1.8, -1134.5, -908.1, 2.6, -1146.5, 15.0),
    ("Dervish & Banges' bracket clock, 15:00", -983.6, 2.0, -1066.5, -980.4, 4.2, -1076.0, 15.0),
    ("Madam Puddifoot's up its side road, 16:00", -524.4, 1.8, -1280.0, -534.7, 2.2, -1280.4, 16.0),
    ("The kirk, its porch and kirkyard from the air, 14:00", -1292.4, 14.0, -838.0, -1314.7, 4.0, -861.5, 14.0),
    ("The school, 11:00", -1245.3, 3.0, -920.4, -1224.0, 5.0, -903.5, 11.0),
    ("The distillery's kiln, 10:00", -1091.2, 2.5, -1239.1, -1113.3, 8.0, -1234.0, 10.0),
    ("The carriage hangar from the air, 15:00", -488.7, 30.0, -1408.7, -462.4, 4.0, -1431.8, 15.0),
    ("The Shrieking Shack and its fence, 16:00", -864.2, 1.8, -936.6, -862.8, 3.5, -917.4, 16.0),
    ("Down the High Street at civil dusk", -975.0, 2.5, -1063.0, -875.0, 4.0, -1155.0, "civil dusk"),
]
