import unittest
from src.navigation.power_battery_manager import (
    calculate_battery_reachability,
    filter_air_traffic_by_radius,
    compute_hardware_compute_metrics,
    SAFE_LANDING_SITES
)


class TelemetryParametersTests(unittest.TestCase):
    def test_battery_safe_nominal(self):
        # UAV near start [-360, -400], dest [300, 400], battery 85%
        res = calculate_battery_reachability(
            current_pos=(-360.0, -400.0, -6.0),
            dest_pos=(300.0, 400.0, -6.0),
            battery_pct=85.0,
            speed_mps=4.0
        )
        self.assertTrue(res.is_safe_to_dest)
        self.assertFalse(res.emergency_divert_active)
        self.assertGreater(res.margin_m, 0.0)
        self.assertIn("NOMINAL", res.action_decision)

    def test_battery_critical_triggers_emergency_divert(self):
        # Battery dropped to 12% (insufficient range)
        res = calculate_battery_reachability(
            current_pos=(-200.0, -150.0, -6.0),
            dest_pos=(300.0, 400.0, -6.0),
            battery_pct=12.0,
            speed_mps=4.0
        )
        self.assertFalse(res.is_safe_to_dest)
        self.assertTrue(res.emergency_divert_active)
        self.assertIsNotNone(res.nearest_safe_site)
        self.assertIn("EMERGENCY DIVERT", res.action_decision)
        # Confirms nearest safe site is one of the designated sites
        self.assertTrue(any(s["id"] == res.nearest_safe_site["id"] for s in SAFE_LANDING_SITES))

    def test_air_traffic_filter_by_radius_and_level(self):
        sample_flights = [
            {"callsign": "IGO1477", "x": 100.0, "y": 200.0},
            {"callsign": "AIC883", "x": 3000.0, "y": 4000.0}
        ]
        # Radius 1.0 km, VERY_LOW density
        res_low = filter_air_traffic_by_radius(sample_flights, (0.0, 0.0), radius_km=1.0, traffic_level="VERY_LOW")
        self.assertEqual(res_low["monitored_flights_count"], 1)

        # Radius 5.0 km, HIGH density
        res_high = filter_air_traffic_by_radius(sample_flights, (0.0, 0.0), radius_km=5.0, traffic_level="HIGH")
        self.assertEqual(res_high["monitored_flights_count"], 7)
        self.assertLessEqual(res_high["conflict_risk_pct"], 50.0)

    def test_hardware_compute_metrics_scaling(self):
        m1 = compute_hardware_compute_metrics(uav_speed_mps=2.0)
        m2 = compute_hardware_compute_metrics(uav_speed_mps=6.0)
        # Higher speed increases graph expansion rate and CPU load
        self.assertGreater(m2["graph_node_expansion_rate"], m1["graph_node_expansion_rate"])
        self.assertGreater(m2["cpu_path_processing_pct"], m1["cpu_path_processing_pct"])
        self.assertIn("ram_path_cache_mb", m1)



if __name__ == '__main__':
    unittest.main()
