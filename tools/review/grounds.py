"""Grounds review views: Hagrid's hut and garden, the Quidditch pitch and the front gates; compare passes with
--before."""
from tools.review.sheet import MESHES

COLUMNS = [("grounds", MESHES)]
VIEWS = [
    ("Hagrid's hut from the castle side, 16:00", -265.4, 4.0, -180.0, -291.5, 2.5, -191.8, 16.0),
    ("Its front, close up, 17:00", -282.0, 2.0, -181.1, -291.5, 2.6, -191.8, 17.0),
    ("The garden from above and behind, 15:00", -321.8, 35.0, -226.5, -308.0, 0.0, -205.1, 15.0),
    ("The pumpkin patch from the back door, 16:00", -295.7, 2.5, -194.9, -300.5, 0.5, -198.4, 16.0),
    ("The far fence and its tree, 16:00", -318.0, 4.0, -201.0, -329.6, 3.0, -208.0, 16.0),
    ("Hagrid's hut at dusk, 21:30", -265.4, 4.0, -180.0, -291.5, 2.5, -191.8, 21.5),
    ("The pitch from above the castle approach, 14:00", -180.0, 90.0, -380.0, -243.6, 0.0, -485.9, 14.0),
    ("The pitch from the path down from the castle, 15:00", -240.0, 12.0, -395.0, -243.6, 2.0, -485.9, 15.0),
    ("The entrance on the castle side, 15:00", -242.0, 1.7, -418.0, -242.6, 1.5, -440.0, 15.0),
    ("A stair from the lawn, north-west, 16:00", -333.2, 3.0, -443.1, -320.6, 2.5, -449.1, 16.0),
    ("The changing rooms from the pitch, 16:00", -244.25, 2.0, -515.9, -244.6, 1.8, -532.0, 16.0),
    ("The west hoops from above midfield, 16:00", -272.0, 22.0, -470.0, -311.8, 10.0, -484.4, 16.0),
    ("The podium and the north stands from above, 16:00", -229.0, 9.0, -460.0, -230.7, 2.5, -443.0, 16.0),
    ("The gates from the road outside, 15:00", -457.1, 3.0, -512.8, -440.85, 3.5, -505.17, 15.0),
    ("The gates from the drive inside, 17:00", -427.3, 2.5, -498.8, -440.85, 3.0, -505.17, 17.0),
    ("A winged boar close up, 14:00", -442.3, 6.5, -511.6, -439.5, 5.8, -508.1, 14.0),
    ("The castle from the lake: the cliff-edge wall (2e), 15:00", None, 25.0, None, 80.0, 20.0, 0.0, 15.0),
    ("The cliff-edge safety wall close up, 16:00", 125.0, 8.0, 30.0, 88.0, 2.0, 60.0, 16.0),
    ("The vegetable-patch wall (5d), 15:00", -155.0, 2.5, 100.0, -171.0, 1.0, 121.0, 15.0),
    ("A coped boundary wall beside the gates, 15:00", -425.0, 2.5, -490.0, -438.0, 2.0, -500.0, 15.0),
    ("Dumbledore's tomb, 16:00", 106.0, 2.5, -645.0, 116.7, 1.0, -653.8, 16.0),
    ("The harbour mouth from the lake, from the south-east, 9:00", 151.3, 8.0, -14.3, 99.8, 4.0, -7.0, 9.0),
    ("The harbour mouth from the lake, from the north-east, 10:00", 141.2, 8.0, 24.4, 99.8, 4.0, -7.0, 10.0),
    ("The harbour mouth close, on a boat's line, 9:30", 118.2, 6.0, -2.2, 99.8, 4.0, -7.0, 9.5),
    ("A village bridge over the burn, 11:00", -722.7, 2.0, -1182.8, -709.1, 0.5, -1188.7, 11.0),
    ("The bridge on the track east of the lake, 10:00", 388.3, 2.5, 0.6, 376.9, 0.5, -9.0, 10.0),
    ("The footbridge on the Forest burn, 13:00", -241.2, 2.0, 310.6, -255.7, 0.5, 313.6, 13.0),
]
