"""Run the deterministic start-to-goal Dubai world routing smoke test."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.navigation.osm_world import OSMWorld, RestrictedZone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=Path, default=ROOT/'worlds'/'dubai_osm')
    args = parser.parse_args()
    scenario = json.loads((args.world/'smoke_scenario.json').read_text(encoding='utf-8'))
    zones = [RestrictedZone(z['id'], tuple(map(tuple, z['polygon_ned'])), z['reason'])
             for z in scenario['restricted_zones']]
    world = OSMWorld.from_directory(args.world, zones)
    start, goal = tuple(scenario['start_ned']), tuple(scenario['goal_ned'])
    direct_mid = ((start[0]+goal[0])/2, (start[1]+goal[1])/2)
    direct_violation = world.violation(*direct_mid, -scenario['altitude_m_agl'],
                                       scenario['horizontal_clearance_m'])
    if not direct_violation:
        raise RuntimeError('scenario does not force the direct route through a constraint')
    route = world.plan(start, goal, scenario['altitude_m_agl'],
                       scenario['grid_resolution_m'], scenario['horizontal_clearance_m'])
    for point in route.path_ned:
        violation = world.violation(*point, -route.altitude_m_agl,
                                    scenario['horizontal_clearance_m'])
        if violation:
            raise RuntimeError(f'planned waypoint violates {violation}: {point}')
    result = {
        'scenario_id': scenario['scenario_id'], 'status': 'PASS',
        'direct_route_blocked_by': direct_violation,
        'route_waypoints_ned': route.path_ned, 'route_length_m': round(route.length_m, 2),
        'altitude_m_agl': route.altitude_m_agl, 'expanded_nodes': route.expanded_nodes,
        'resolution_m': route.resolution_m,
        'claim_limit': 'Offline geometric planner test; no SITL vehicle was launched'
    }
    (args.world/'smoke_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
