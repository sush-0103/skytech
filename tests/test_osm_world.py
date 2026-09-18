import json
from pathlib import Path
import unittest

from src.navigation.osm_world import OSMWorld, RestrictedZone
from src.interface.safety_supervisor import CandidateSetpoint, DeterministicSafetySupervisor
import time


WORLD = Path(__file__).resolve().parents[1]/'worlds'/'dubai_osm'


class TestOSMWorld(unittest.TestCase):
    def setUp(self):
        scenario = json.loads((WORLD/'smoke_scenario.json').read_text(encoding='utf-8'))
        self.scenario = scenario
        self.zone = RestrictedZone('DEMO_RESTRICTED_BASE',
                                   tuple(map(tuple, scenario['restricted_zones'][0]['polygon_ned'])))
        self.world = OSMWorld.from_directory(WORLD, [self.zone])

    def test_coordinate_bounds_and_restricted_zone(self):
        self.assertAlmostEqual(self.world.north_bounds[0], -407.0981920844361)
        self.assertAlmostEqual(self.world.east_bounds[1], 511.7275860413864)
        self.assertEqual(self.world.violation(0, 0, -30), 'RESTRICTED_ZONE:DEMO_RESTRICTED_BASE')

    def test_building_blocks_low_altitude_but_can_be_overflown(self):
        building = self.world.buildings[0]
        point = building.rings_ned[0][0]
        self.assertTrue(self.world.violation(*point, -(building.height_m-1)).startswith('STATIC_BUILDING:'))
        self.assertIsNone(self.world.violation(*point, -(building.height_m+4), horizontal_clearance_m=0))

    def test_smoke_route_avoids_all_static_constraints(self):
        s = self.scenario
        route = self.world.plan(tuple(s['start_ned']), tuple(s['goal_ned']), s['altitude_m_agl'],
                                s['grid_resolution_m'], s['horizontal_clearance_m'])
        self.assertGreater(len(route.path_ned), 2)
        for north, east in route.path_ned:
            self.assertIsNone(self.world.violation(north, east, -route.altitude_m_agl,
                                                   s['horizontal_clearance_m']))

    def test_supervisor_rejects_restricted_world_setpoint(self):
        supervisor = DeterministicSafetySupervisor(static_world=self.world)
        now = time.time()
        decision = supervisor.evaluate_setpoint(
            CandidateSetpoint(timestamp=now, x=0, y=0, z=-30), now=now)
        self.assertFalse(decision.accepted)
        self.assertIn('RESTRICTED_ZONE:DEMO_RESTRICTED_BASE', decision.rejection_reason)

    def test_supervisor_rejects_jump_across_restricted_zone(self):
        supervisor = DeterministicSafetySupervisor(static_world=self.world)
        now = time.time()
        first = supervisor.evaluate_setpoint(
            CandidateSetpoint(timestamp=now, x=-100, y=0, z=-30), now=now)
        self.assertTrue(first.accepted)
        crossed = supervisor.evaluate_setpoint(
            CandidateSetpoint(timestamp=now+.01, x=100, y=0, z=-30), now=now+.01)
        self.assertFalse(crossed.accepted)
        self.assertIn('Command segment intersects RESTRICTED_ZONE', crossed.rejection_reason)


if __name__ == '__main__':
    unittest.main()
