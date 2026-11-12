from datetime import datetime, timezone
import unittest

from src.health_guidance import (
    WAQI_AQI_CATEGORIES,
    build_health_guidance,
)


class HealthGuidanceTests(unittest.TestCase):
    def _observation(self, **updates):
        observation = {
            "available": True,
            "aqi": 50,
            "freshness": "FRESH",
            "observed_at": datetime(2026, 10, 26, 10, tzinfo=timezone.utc),
        }
        observation.update(updates)
        return observation

    def test_documented_us_epa_categories_and_boundaries(self):
        expectations = {
            0: "Good",
            50: "Good",
            51: "Moderate",
            100: "Moderate",
            101: "Unhealthy for Sensitive Groups",
            150: "Unhealthy for Sensitive Groups",
            151: "Unhealthy",
            200: "Unhealthy",
            201: "Very Unhealthy",
            300: "Very Unhealthy",
            301: "Hazardous",
        }
        self.assertEqual(len(WAQI_AQI_CATEGORIES), 6)
        for aqi, category in expectations.items():
            with self.subTest(aqi=aqi):
                guidance = build_health_guidance(self._observation(aqi=aqi))
                self.assertEqual(guidance["category"], category)
                self.assertEqual(guidance["scale"], "WAQI-reported US EPA 2016 AQI")
                self.assertTrue(guidance["effects"])
                self.assertTrue(guidance["precautions"])

    def test_missing_stale_or_invalid_observations_have_no_current_guidance(self):
        cases = (
            {"available": False},
            {"freshness": "STALE"},
            {"freshness": "DELAYED"},
            {"aqi": None},
            {"aqi": float("nan")},
            {"aqi": float("inf")},
            {"aqi": -1},
            {"observed_at": None},
        )
        for updates in cases:
            with self.subTest(updates=updates):
                self.assertIsNone(
                    build_health_guidance(self._observation(**updates))
                )


if __name__ == "__main__":
    unittest.main()
