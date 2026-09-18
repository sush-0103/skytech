import unittest
from src.navigation.reactive_avoidance import evaluate_escape_manifolds, ReactiveAvoidanceDecision


class ReactiveAvoidanceTests(unittest.TestCase):
    def test_obstacle_on_left_evades_right(self):
        # Building is on the left (rel_y = -8.0m)
        decision = evaluate_escape_manifolds(
            obstacle_distance_m=18.0,
            obstacle_rel_y=-8.0,
            obstacle_height_m=35.0,
            uav_speed_mps=4.0
        )
        self.assertTrue(decision.obstacle_detected)
        self.assertEqual(decision.optimal_path, "RIGHT")
        self.assertEqual(decision.optimal_primitive, "BANK_RIGHT_EVADE")
        self.assertEqual(decision.threat_level, "CAUTION")
        self.assertGreater(decision.clearance_margin_m, 5.0)

    def test_obstacle_on_right_evades_left(self):
        # Building is on the right (rel_y = +8.0m)
        decision = evaluate_escape_manifolds(
            obstacle_distance_m=14.0,
            obstacle_rel_y=8.0,
            obstacle_height_m=40.0,
            uav_speed_mps=4.0
        )
        self.assertTrue(decision.obstacle_detected)
        self.assertEqual(decision.optimal_path, "LEFT")
        self.assertEqual(decision.optimal_primitive, "LATERAL_EVADE_WEST")
        self.assertEqual(decision.threat_level, "CRITICAL")
        self.assertGreater(decision.clearance_margin_m, 5.0)

    def test_low_building_or_blocked_sides_climbs_top(self):
        # Low building (8m height, UAV at 6m) with both lateral sides blocked
        decision = evaluate_escape_manifolds(
            obstacle_distance_m=22.0,
            obstacle_rel_y=0.0,
            obstacle_height_m=8.0,
            uav_speed_mps=3.5,
            uav_alt_agl_m=6.0,
            left_blocked=True,
            right_blocked=True
        )
        self.assertTrue(decision.obstacle_detected)
        self.assertEqual(decision.optimal_path, "TOP")
        self.assertEqual(decision.optimal_primitive, "ASCEND_RAPID_CLEAR")
        self.assertGreater(decision.clearance_margin_m, 5.0)

    def test_bounding_box_within_screen_bounds(self):
        decision = evaluate_escape_manifolds(
            obstacle_distance_m=12.0,
            obstacle_rel_y=-4.0,
            obstacle_height_m=25.0
        )
        bbox = decision.bbox
        self.assertGreaterEqual(bbox["left"], 0.0)
        self.assertLessEqual(bbox["left"] + bbox["width"], 100.0)
        self.assertGreaterEqual(bbox["top"], 0.0)
        self.assertLessEqual(bbox["top"] + bbox["height"], 100.0)

    def test_to_dict_structure(self):
        decision = evaluate_escape_manifolds(
            obstacle_distance_m=15.0,
            obstacle_rel_y=0.0,
            obstacle_height_m=20.0
        )
        d = decision.to_dict()
        self.assertIn("candidate_paths", d)
        self.assertEqual(len(d["candidate_paths"]), 3)
        self.assertIn("optimal_action", d)
        self.assertIn("threat_level", d)


if __name__ == '__main__':
    unittest.main()
