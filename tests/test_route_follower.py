import json
import math
from pathlib import Path
import unittest
from unittest.mock import Mock

from src.navigation.osm_world import OSMWorld, RestrictedZone, _segment_distance
from src.navigation.route_follower import (
    LocalMapFrame, VehicleState, RouteFollower, gate_velocity, shortcut_route,
)
from tools.fly_dubai_route import Link

WORLD = Path(__file__).resolve().parents[1]/'worlds/dubai_osm'


class TestRouteFollower(unittest.TestCase):
    def setUp(self):
        self.state = VehicleState((0., 0., -6.), (0., 0., 0.), 10.)
        self.world = Mock()
        self.world.segment_violation.return_value = None

    def test_map_axes_and_roundtrip(self):
        frame = LocalMapFrame(25.142895, 55.258255)
        latitude, longitude = frame.to_geographic(-360., -400.)
        self.assertLess(latitude, frame.latitude)
        self.assertLess(longitude, frame.longitude)
        for actual, expected in zip(frame.to_map(latitude, longitude), (-360., -400.)):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_command_acceleration_is_vector_limited(self):
        follower = RouteFollower([(0., 0.), (100., 100.)], 6.)
        previous = (0., 0., 0.)
        for _ in range(100):
            command = follower.propose(self.state, .05)
            self.assertLessEqual(math.dist(command, previous), .050000001)
            self.assertLessEqual(math.hypot(*command[:2]), 4.000001)
            previous = command

    def test_corner_requires_low_measured_speed(self):
        follower = RouteFollower([(0., 0.), (10., 0.), (10., 10.)], 6.)
        follower.propose(VehicleState((10., 0., -6.), (1., 0., 0.), 10.), .05)
        self.assertEqual(follower.index, 1)
        follower.propose(VehicleState((10., 0., -6.), (0., 0., 0.), 10.), .05)
        self.assertEqual(follower.index, 2)

    def test_done_at_stopped_destination(self):
        follower = RouteFollower([(-10., 0.), (0., 0.)], 6.)
        self.assertEqual(follower.propose(self.state, .05), (0., 0., 0.))
        self.assertTrue(follower.done)

    def test_delayed_controller_rejected(self):
        with self.assertRaises(ValueError):
            RouteFollower([(0., 0.), (10., 0.)], 6.).propose(self.state, .3)

    def test_fresh_safe_command(self):
        self.assertIsNone(gate_velocity(self.world, self.state, (1., 0., 0.), 10.1, 0.))
        self.assertEqual(self.world.segment_violation.call_count, 2)

    def test_stale_and_future_sample_rejected(self):
        for now in (10.31, 9.99):
            self.assertEqual(gate_velocity(self.world, self.state, (0., 0., 0.), now, 0.), 'STALE_TELEMETRY')

    def test_tracking_and_speed_limits(self):
        self.assertEqual(gate_velocity(self.world, self.state, (0., 0., 0.), 10.1, 2.1), 'TRACKING_ERROR')
        self.assertEqual(gate_velocity(self.world, self.state, (4., 4., 0.), 10.1, 0.), 'VELOCITY_LIMIT')

    def test_nonfinite_rejected(self):
        self.assertEqual(gate_velocity(self.world, self.state, (math.nan, 0., 0.), 10.1, 0.), 'NONFINITE_STATE')

    def test_stopping_corridor_rejected(self):
        self.world.segment_violation.return_value = 'STATIC_BUILDING:test'
        self.assertEqual(gate_velocity(self.world, self.state, (1., 0., 0.), 10.1, 0.), 'STATIC_BUILDING:test')

    def test_thin_barrier_cannot_be_skipped(self):
        zone = RestrictedZone('thin', ((.4,-1.),(.5,-1.),(.5,1.),(.4,1.)))
        world = OSMWorld.from_directory(WORLD, [zone])
        world.buildings = []
        self.assertEqual(world.segment_violation((-10.,0.,-6.),(10.,0.,-6.),0.), 'RESTRICTED_ZONE:thin')
        self.assertIsNone(world.segment_violation((-10.,3.,-6.),(10.,3.,-6.),1.))

    def test_segment_distances(self):
        self.assertEqual(_segment_distance((0,0),(2,0),(1,-1),(1,1)), 0.)
        self.assertEqual(_segment_distance((0,0),(2,0),(1,0),(3,0)), 0.)
        self.assertEqual(_segment_distance((0,0),(2,0),(3,0),(4,0)), 1.)
        self.assertEqual(_segment_distance((0,0),(0,0),(0,2),(0,3)), 2.)

    def test_dubai_shortcuts_remain_clear(self):
        scenario = json.loads((WORLD/'smoke_scenario.json').read_text())
        zones = [RestrictedZone(z['id'], tuple(map(tuple,z['polygon_ned']))) for z in scenario['restricted_zones']]
        world = OSMWorld.from_directory(WORLD, zones)
        route = world.plan(tuple(scenario['start_ned']), tuple(scenario['goal_ned']), 6., 8., 8.)
        path = shortcut_route(world, route.path_ned, 6., 8.)
        self.assertLess(len(path), len(route.path_ned))
        for a,b in zip(path,path[1:]):
            self.assertIsNone(world.segment_violation((*a,-6.),(*b,-6.),8.))


class TestTelemetryReplay(unittest.TestCase):
    def make_link(self, samples):
        link = Link.__new__(Link)
        link.connection = Mock()
        link.connection.recv_match.side_effect = [*samples, None]
        link.messages, link.times = {}, {}
        link.sysid, link.compid = 1, 1
        link.heartbeat_at = 0.
        link.last_position_boot_ms = 100
        return link

    def message(self, boot):
        m = Mock(time_boot_ms=boot)
        m.get_type.return_value = 'LOCAL_POSITION_NED'
        m.get_srcSystem.return_value = 1
        m.get_srcComponent.return_value = 1
        return m

    def test_duplicate_does_not_refresh(self):
        link = self.make_link([self.message(100)])
        link.pump()
        self.assertNotIn('LOCAL_POSITION_NED', link.times)

    def test_backward_clock_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'backwards'):
            self.make_link([self.message(99)]).pump()

    def test_fresh_sample_accepted(self):
        link = self.make_link([self.message(101)])
        link.pump()
        self.assertIn('LOCAL_POSITION_NED', link.times)


if __name__ == '__main__':
    unittest.main()
