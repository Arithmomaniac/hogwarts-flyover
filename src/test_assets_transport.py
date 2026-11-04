"""The transport family: the regraded railway, the track, train, station and ship, and the ship's berth."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import assets_testing as world  # noqa: E402
import fix_berth  # noqa: E402
import fix_railway  # noqa: E402
import mesh_transport  # noqa: E402
import railway  # noqa: E402
from mesh_source import CHANNELS, Grid, decode, load_georef, sample  # noqa: E402

# the faults trimesh finds in this family's decoded meshes today (assets_testing.mesh_faults), pinned: a new fault
# fails, and so does a fix until this is updated
MESH_FAULTS = {"winding": {}, "collapsed": {}}
EXPECTED_FAILURES = ()


class ShipBerthTests(unittest.TestCase):
    """The essay map's ship ink left a ship-shaped island in the lake; the berth must be open bay water."""

    @classmethod
    def setUpClass(cls):
        data = world.data()
        cls.meta = data["meta"]
        cls.bed, cls.wd = decode(data["tex"]["locBed"]), decode(data["tex"]["locW"])
        cls.surface = decode(data["tex"]["locN"])[..., 0]

    def test_berth_is_open_water_like_the_bay(self):
        d = fix_berth.axis_distance(self.meta)
        self.assertFalse(fix_berth.islets(d, self.wd).any())           # no islets of ink
        berth = (d < fix_berth.BERTH_M) & (self.wd > 0)
        self.assertLessEqual(self.bed[berth].max(), fix_berth.bay_bed(d, self.bed, self.wd) + 0.02)   # no shallow band
        self.assertEqual(set(self.surface[berth].tolist()), {fix_berth.bay_surface(d, self.wd, self.surface)})  # no ink stripes

    def test_fixing_is_idempotent(self):
        *_, changed = fix_berth.fix_berth(self.meta, self.bed, self.wd, self.surface)
        self.assertEqual(changed, 0)


class RailwayTests(unittest.TestCase):
    """White Hound's line, fitted to the registered train and graded as the user chose (railway-regrade)."""

    @classmethod
    def setUpClass(cls):
        data = world.data()
        cls.meta = data["meta"]
        cls.scene = world.scene()
        cls.georef = load_georef()
        cls.al = railway.alignment(cls.scene, cls.meta, cls.georef)
        cls.grid = Grid(cls.meta["local"])
        cls.bed = decode(data["tex"]["locBed"])
        cls.wd = decode(data["tex"]["locW"])
        cls.trees = decode(data["tex"]["locA"])[..., CHANNELS["locA"].index("trees")]
        cls.stream = decode(data["tex"]["locB"])[..., CHANNELS["locB"].index("stream")]
        cls.surface = decode(data["tex"]["locN"])[..., 0]

    def ground(self, xz):
        xz = np.asarray(xz, np.float64)
        return sample(self.bed, self.grid, xz[..., 0], xz[..., 1])

    def test_the_loop_runs_through_the_registered_train(self):
        loop = self.al.loop.xz
        off = lambda p: float(np.hypot(*(railway.point_at(loop, railway.project(loop, p)) - p)))
        a, b = np.asarray(self.meta["train"]["carriages"], np.float64)
        for p in a + (b - a) * np.linspace(0.0, 1.0, 21)[:, None]:
            with self.subTest(segment="carriages", at=p.round(1).tolist()):
                self.assertLess(off(p), 0.05)                                      # the coaches' straight
        a, b = np.asarray(self.meta["train"]["engine"], np.float64)
        for p in (a, b):
            with self.subTest(segment="engine end", at=p.round(1).tolist()):
                self.assertLess(off(p), 0.15)                                      # the engine's ends on its ink,
        self.assertLess(off((a + b) / 2), 1.2)                                     # standing on the curve as a chord
        self.assertLess(np.hypot(*(loop[[0, -1]] - self.al.toe).T).max(), 0.05)    # the loop closes at the toe
        self.assertLess(self.al.coaches[1], self.al.engine[0])                     # the engine stands on beyond the coaches

    def test_the_sides_of_the_loop_leave_the_main_line_at_a_turnout(self):
        loop, main = self.al.loop.xz, self.al.main.xz
        out = lambda a, b: railway.unit(b - a)
        angle = lambda u, v: math.degrees(math.acos(float(np.clip(u @ v, -1.0, 1.0))))
        station, north, up = out(loop[0], loop[8]), out(loop[-1], loop[-9]), out(main[0], main[8])
        self.assertLess(angle(station, -up), 1.5)                                  # straight on from the main line
        self.assertTrue(2.0 < angle(station, north) < 10.0)                        # a turnout, not a crossing
        junction = railway.grounds_world(railway.traced(self.scene, "track", "turning-loop")["junction"], self.georef)
        half = len(loop) // 2
        self.assertGreater(np.hypot(*(loop[half:] - junction).T).min(), 2.0)       # apart where the ink shows them apart

    def test_curves_are_no_tighter_than_three_chains_and_keep_to_the_drawn_line(self):
        def radius(line):
            v, a = np.gradient(line, axis=0), np.gradient(np.gradient(line, axis=0), axis=0)
            return np.hypot(*v.T) ** 3 / np.maximum(np.abs(v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]), 1e-12)
        three_chains = 3 * 20.1168                                                 # what a tender engine might take at walking pace
        self.assertGreater(radius(self.al.loop.xz)[3:-3].min(), 0.97 * three_chains)
        self.assertGreater(radius(self.al.main.xz[self.al.main.s < 300])[3:-3].min(), 70.0)   # "a sharp bend"
        drawn = railway.grounds_world(railway.traced(self.scene, "track", "turning-loop")["loop"], self.georef)
        off = np.array([np.hypot(*(self.al.loop.xz - p).T).min() for p in drawn])
        s = railway.arclength(drawn)
        car, eng = (np.asarray(self.meta["train"][k], np.float64) for k in ("carriages", "engine"))
        swing = (s > railway.project(drawn, car[1])) & (s < railway.project(drawn, eng[0]))
        self.assertLess(off[~swing].max(), 3.0)                                    # on White Hound's ink,
        self.assertLess(off[swing].max(), 7.0)                                     # bar its swing round his drawn engine
        self.assertLess(np.median(off), 1.5)

    def test_the_ground_is_graded_under_every_running_line(self):
        level = self.meta["train"]["level"]
        self.assertTrue(np.allclose(self.ground(self.al.loop.xz), level, atol=0.05))
        upto = self.al.main.s < self.al.portal_s - 4.0
        ground = self.ground(self.al.main.xz[upto])
        rise = railway.GRADE * np.maximum(self.al.main.s[upto] - railway.TURNOUT_M, 0.0)
        self.assertTrue(np.allclose(ground, level + rise, atol=0.05))
        across = self.al.loop.xz[::40]
        normal = np.gradient(self.al.loop.xz, axis=0)[::40][:, ::-1] * [1.0, -1.0]
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        for offset in (-2.5, 2.5):                                                 # level under the ballast too
            self.assertTrue(np.allclose(self.ground(across + offset * normal), level, atol=0.05))

    def test_the_station_terrace_is_level_to_the_boundary_wall(self):
        import shapely

        x0, z0, x1, z1 = self.al.terrace.bounds
        gx, gz = np.meshgrid(np.arange(x0, x1, 2.0), np.arange(z0, z1, 2.0))
        inside = shapely.contains_xy(self.al.terrace.buffer(-2.0), gx, gz)        # a texel clear of the side slopes
        points = np.column_stack([gx[inside], gz[inside]])
        path = np.array([np.hypot(*(self.al.path.xz - p).T).min() for p in points]) < railway.PATH_HALF_M + 4.0
        heights = self.ground(points[~path]) - self.meta["train"]["level"]
        self.assertGreater(len(heights), 1000)
        self.assertTrue(np.allclose(heights, 0.0, atol=0.05))                      # cut back on the spur, made up below it

    def test_a_rock_face_rises_over_the_portal(self):
        p, t, level = self.al.portal
        self.assertAlmostEqual(float(self.ground(p - 4.0 * t)), level - 4.0 * railway.GRADE, delta=0.1)   # the cutting floor
        self.assertGreater(float(self.ground(p + 3.0 * t)), level + railway.HEADWALL_M)                 # hill over the tunnel

    def test_regrading_is_idempotent_and_clears_the_earthworks_of_trees(self):
        *_, changed = fix_railway.fix_railway(self.scene, self.meta, self.georef, self.bed, self.wd, self.trees,
                                              self.surface, self.stream)
        self.assertEqual(changed, 0)
        works = fix_railway.footprint(self.scene, self.meta, self.georef, self.bed)
        self.assertGreater(works.sum(), 5000)
        self.assertEqual(int(self.trees[works].max()), 0)
        self.assertLessEqual(float(self.wd[works].max()), fix_railway.DRY)          # streams run under in culverts
        self.assertEqual(int(self.stream[works].min()), fix_railway.NO_STREAM)

    def test_cuttings_and_embankments_meet_the_ground_at_their_slopes(self):
        level, loop = self.al.level, self.al.loop
        at = loop.s[-1] / 2
        t = railway.tangent_at(loop.xz, at)
        side = (railway.point_at(loop.xz, at) + np.array([-t[1], t[0]]) * (railway.FORMATION_M + 2.0))[None]
        cut, fill, _ = railway.envelope(self.al, side[:, 0], side[:, 1])
        self.assertAlmostEqual(float(railway.regrade(np.array([level + 30.0]), cut, fill)[0]),
                               level + 2.0 * railway.CUT, delta=0.3)               # a rock cutting, 2 m out
        self.assertAlmostEqual(float(railway.regrade(np.array([level - 30.0]), cut, fill)[0]),
                               level - 2.0 * railway.FILL, delta=0.3)              # an embankment, 2 m out


class TransportTests(unittest.TestCase):
    """The station, train and ship as White Hound places them: nine 64'6" coaches between two guards' vans on the
    registered line, the engine and tender detached beyond them (detached-engine), a 170-yard platform with a barrier
    along its back, a building with a narrow door onto the road, a part-length canopy, and the 180 ft ship."""

    @classmethod
    def setUpClass(cls):
        data = world.data()
        cls.meta = data["meta"]
        cls.scene = world.scene()
        cls.al = railway.alignment(cls.scene, cls.meta, load_georef())
        cls.consist = mesh_transport.consist(cls.al)
        chunks = world.payload()["chunks"]
        cls.parts = {}
        for asset in ("track", "station", "train", "ship"):
            mine = [c for c in chunks if c["asset"] == asset]
            if not mine:
                continue
            tris = np.concatenate([c["pos"][c["idx"]] for c in mine])
            attr = np.concatenate([c["attr"][c["idx"][:, 0]] for c in mine])
            n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            cls.parts[asset] = (tris, attr, n / np.linalg.norm(n, axis=1, keepdims=True))
        cls.rail_top = cls.al.level + mesh_transport.RAIL_TOP_M
        from scipy.spatial import cKDTree
        cls.loop_tree = cKDTree(cls.al.loop.xz)

    def along(self, points):
        """Arclength round the loop of each point's nearest centreline sample."""
        return self.al.loop.s[self.loop_tree.query(np.asarray(points)[..., [0, 2]].reshape(-1, 2))[1]]

    def part(self, asset, attr=None):
        self.assertIn(asset, self.parts, f"no {asset} geometry in the payload")
        tris, a, n = self.parts[asset]
        keep = np.all(a == attr, axis=1) if attr is not None else np.ones(len(a), bool)
        return tris[keep], n[keep]

    def frame(self, points):
        origin, along, west = self.al.frame
        return (points[..., [0, 2]] - origin) @ along, (points[..., [0, 2]] - origin) @ west

    def test_rails_run_at_standard_gauge_round_the_loop_and_into_the_tunnel(self):
        rails, _ = self.part("track", mesh_transport.RAIL)
        pts = rails.reshape(-1, 3)
        stations = [(self.al.loop, s) for s in np.arange(40.0, self.al.loop.s[-1] - 40.0, 37.0)]
        stations += [(self.al.main, s) for s in (5.0, self.al.portal_s, self.al.portal_s + 5.0)]
        for line, s in stations:
            p, t = railway.point_at(line.xz, s), railway.tangent_at(line.xz, s)
            near = pts[np.abs((pts[:, [0, 2]] - p) @ t) < 0.4]
            lateral = (near[:, [0, 2]] - p) @ np.array([-t[1], t[0]])
            near, lateral = near[np.abs(lateral) < 1.2], lateral[np.abs(lateral) < 1.2]
            with self.subTest(at=round(float(s)), line=len(line.s)):
                self.assertTrue((lateral > 0).any() and (lateral < 0).any())
                gauge = lateral[lateral > 0].min() - lateral[lateral < 0].max()
                self.assertAlmostEqual(gauge, mesh_transport.GAUGE_M, delta=0.02)
                level = float(np.interp(s, line.s, line.level))
                self.assertAlmostEqual(float(near[:, 1].max()), level + mesh_transport.RAIL_TOP_M, delta=0.02)

    def test_sleepers_lie_on_ballast_under_the_rails(self):
        sleepers, _ = self.part("track", mesh_transport.SLEEPER)
        ballast, n = self.part("track", mesh_transport.BALLAST)
        self.assertGreater(len(sleepers) / 10, self.al.loop.s[-1] / 0.8)          # boxes of ten triangles, ~0.7 m apart
        top = ballast[n[:, 1] > 0.99][:, :, 1].max()
        self.assertLess(top, sleepers[:, :, 1].max())
        self.assertGreater(len(np.unique(np.round(self.along(sleepers.mean(axis=1))))), 0.9 * self.al.loop.s[-1])

    def test_nine_coaches_stand_between_two_guards_vans_in_the_registered_train(self):
        kinds = [v.kind for v in self.consist]
        self.assertEqual(kinds, ["van"] + ["coach"] * 9 + ["van", "tender", "engine"])
        coaches = [v for v in self.consist if v.kind == "coach"]
        for v in coaches:
            self.assertAlmostEqual(v.s1 - v.s0, mesh_transport.COACH_M, delta=0.01)          # 64'6"
        lo, hi = self.al.coaches
        self.assertGreaterEqual(self.consist[0].s0, lo - 0.01)                               # the north-west van in the ink
        self.assertLessEqual(coaches[-1].s1, hi + 0.01)
        self.assertGreater(self.consist[10].s0, hi)                                           # the second van in the gap

    def test_couplings_join_only_the_coupled_vehicles(self):
        tris, _ = self.part("train")
        s = self.along(tris.reshape(-1, 3)).reshape(-1, 3)
        van, tender = self.consist[10], self.consist[11]
        self.assertGreater(tender.s0 - van.s1, 40.0)                                         # the engine stands well off
        self.assertFalse(((s > van.s1 + 1.5) & (s < tender.s0 - 1.5)).any())                 # nothing stands in the gap
        self.assertFalse(((s.min(axis=1) < van.s1 + 1.5) & (s.max(axis=1) > tender.s0 - 1.5)).any())   # or spans it
        for a, b in zip(self.consist[:10], self.consist[1:11]):
            self.assertLess(b.s0 - a.s1, 1.0)                                                 # buffer to buffer
        self.assertLess(self.consist[12].s0 - tender.s1, 1.0)

    def test_every_wheel_stands_on_a_rail(self):
        tris, _ = self.part("train")
        pts = tris.reshape(-1, 3)
        self.assertGreater(pts[:, 1].min(), self.rail_top - 0.02)                            # nothing sinks into the rails
        for v in self.consist:
            for s in v.axles:
                p, t = railway.point_at(self.al.loop.xz, s), railway.tangent_at(self.al.loop.xz, s)
                near = pts[(np.abs((pts[:, [0, 2]] - p) @ t) < 0.3)
                           & (np.abs(np.abs((pts[:, [0, 2]] - p) @ np.array([-t[1], t[0]])) - 0.75) < 0.1)]
                with self.subTest(vehicle=v.kind, axle=round(float(s), 1)):
                    self.assertAlmostEqual(float(near[:, 1].min()), self.rail_top, delta=0.03)

    def test_the_engine_leads_away_from_the_coaches(self):
        engine = self.consist[12]
        self.assertGreater(engine.front, engine.s0 + (engine.s1 - engine.s0) / 2)            # nose to the south-east
        tris, _ = self.part("train")
        pts = tris.reshape(-1, 3)
        s = self.along(pts)
        on = (s > engine.s0) & (s < engine.s1)
        tallest = s[on][np.argmax(pts[on, 1])]
        self.assertGreater(tallest, (engine.s0 + engine.s1) / 2)                             # the chimney at the front

    def test_the_platform_runs_170_yards_beside_the_coaches(self):
        top, n = self.part("station", mesh_transport.PAVING)
        up = top[n[:, 1] > 0.99]
        s, t = self.frame(up.reshape(-1, 3))
        self.assertAlmostEqual(float(s.max() - s.min()), 170 * 0.9144, delta=0.5)
        self.assertGreater(t.min(), mesh_transport.PLATFORM_EDGE_M - 0.01)                   # clear of the coaches
        self.assertAlmostEqual(float(up[:, :, 1].mean()), self.rail_top + mesh_transport.PLATFORM_HEIGHT_M, delta=0.02)
        tris, _ = self.part("station")
        s, t = self.frame(tris.reshape(-1, 3))
        above = tris.reshape(-1, 3)[:, 1] > self.rail_top + 0.1
        self.assertGreater(t[above].min(), 1.40)                                             # nothing fouls the train

    def test_a_wall_backs_the_platform_with_ways_out_at_both_ends(self):
        wall, _ = self.part("station", mesh_transport.STONE)
        pts = wall.reshape(-1, 3)
        s, t = self.frame(pts)
        back = (np.abs(t - mesh_transport.PLATFORM_BACK_M) < 0.6) & (pts[:, 1] > self.rail_top + mesh_transport.PLATFORM_HEIGHT_M + 1.0)
        lo, hi = self.al.platform
        self.assertGreater(s[back].max() - s[back].min(), 0.7 * (hi - lo))                  # along the back
        self.assertGreater(s[back].min() - lo, 3.0)                                          # the way out to the lake path
        self.assertGreater(s[back].max(), hi - 2.0)                                          # closed to the far end
        house, _ = self.part("station", mesh_transport.BOARDING)
        hs, ht = self.frame(house.reshape(-1, 3))
        self.assertTrue(lo < hs.min() and hs.max() < hi)                                     # the building beside the platform
        self.assertGreater(ht.min(), mesh_transport.PLATFORM_BACK_M - 1.0)                   # behind its back, onto the road

    def test_the_building_has_windows_and_a_narrow_door_onto_the_road(self):
        glass, _ = self.part("station", mesh_transport.DIM_WINDOW)                 # "the dim light issuing from the station windows"
        door, n = self.part("station", mesh_transport.DOOR)
        self.assertGreater(len(glass), 8)
        self.assertEqual(len(self.part("station", mesh_transport.WINDOW)[0]), 0)
        self.assertGreater(len(self.part("train", mesh_transport.WINDOW)[0]), 100)  # the coaches' brighter lamps
        west = self.al.frame[2]
        road = (n[:, [0, 2]] @ west) > 0.9
        self.assertTrue(road.any())                                                          # a door facing the road
        width = self.frame(door[road].reshape(-1, 3))[0]
        self.assertLess(float(width.max() - width.min()), 1.3)                               # a narrow one

    def test_the_canopy_covers_only_part_of_the_platform(self):
        roof, n = self.part("station", mesh_transport.SLATE)
        s, t = self.frame(roof.reshape(-1, 3))
        over = t < mesh_transport.PLATFORM_BACK_M - 1.0
        covered = s[over].max() - s[over].min()
        lo, hi = self.al.platform
        self.assertTrue(0.12 * (hi - lo) < covered < 0.5 * (hi - lo), covered)                # "isn't all under a canopy"

    def test_the_tunnel_mouth_is_dark_and_faces_the_cutting(self):
        void, n = self.part("track", mesh_transport.VOID)
        p, t, level = self.al.portal
        centre = void.reshape(-1, 3).mean(axis=0)
        self.assertLess(float(np.hypot(*(centre[[0, 2]] - p))), 6.0)
        self.assertGreater(float(-(n[:, [0, 2]] @ t).mean()), 0.9)                          # facing back down the cutting
        self.assertGreater(float(void[:, :, 1].max() - level), 4.5)                          # room for the engine

    def test_the_ship_has_three_masts_and_its_mainmast_is_180_feet(self):
        tris, _ = self.part("ship")
        pts = tris.reshape(-1, 3)
        self.assertAlmostEqual(float(pts[:, 1].max() - self.meta["lake"]), 180 * 0.3048, delta=1.0)
        a, b = (np.asarray(self.meta["extras"]["ship"][k], np.float64) for k in ("a", "b"))
        axis = (b - a) / np.linalg.norm(b - a)
        side = np.abs((pts[:, [0, 2]] - a) @ np.array([-axis[1], axis[0]]))
        self.assertLess(float(np.percentile(side, 99)), self.meta["extras"]["ship"]["w"] / 2 + 6.0)   # rigging to the hull's sides
        masts, _ = self.part("ship", mesh_transport.MAST)
        tall = masts[masts[:, :, 1].max(axis=1) > self.meta["lake"] + 30.0].mean(axis=1)
        along = np.sort((tall[:, [0, 2]] - a) @ axis)
        self.assertEqual(int((np.diff(along) > 8.0).sum()) + 1, 3)                  # three masts, well apart


class TransportMeshOracleTests(unittest.TestCase):
    """trimesh, as an independent oracle on the decoded payload."""

    def test_the_decoded_meshes_have_only_the_known_faults(self):
        self.assertEqual(world.mesh_faults("transport"), MESH_FAULTS)


if __name__ == "__main__":
    unittest.main()
