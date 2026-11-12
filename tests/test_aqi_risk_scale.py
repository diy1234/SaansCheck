from datetime import datetime, timezone
import unittest

from src.aqi_risk_scale import (
    INDIAN_CPCB_SCALE,
    US_AQI_SCALE,
    aqi_category,
    current_risk_observation,
)
from src.waqi_data import AQI_SCALE_LABEL, SOURCE_LABEL


class AqiRiskScaleTests(unittest.TestCase):
    def test_cpcb_category_boundaries(self):
        boundaries = (
            (50, "Good"),
            (51, "Satisfactory"),
            (100, "Satisfactory"),
            (101, "Moderate"),
            (150, "Moderate"),
            (151, "Moderate"),
            (200, "Moderate"),
            (201, "Poor"),
            (300, "Poor"),
            (301, "Very Poor"),
            (400, "Very Poor"),
            (401, "Severe"),
            (500, "Severe"),
        )
        for value, expected in boundaries:
            with self.subTest(value=value):
                self.assertEqual(aqi_category(INDIAN_CPCB_SCALE, value), expected)

    def test_us_aqi_category_boundaries_are_not_cpcb_categories(self):
        boundaries = (
            (50, "Good"),
            (51, "Moderate"),
            (100, "Moderate"),
            (101, "Unhealthy for Sensitive Groups"),
            (150, "Unhealthy for Sensitive Groups"),
            (151, "Unhealthy"),
            (200, "Unhealthy"),
            (201, "Very Unhealthy"),
            (300, "Very Unhealthy"),
            (301, "Hazardous"),
            (400, "Hazardous"),
            (401, "Hazardous"),
            (500, "Hazardous"),
        )
        for value, expected in boundaries:
            with self.subTest(value=value):
                self.assertEqual(aqi_category(US_AQI_SCALE, value), expected)
        self.assertEqual(aqi_category(INDIAN_CPCB_SCALE, 51), "Satisfactory")
        self.assertEqual(aqi_category(US_AQI_SCALE, 51), "Moderate")
        self.assertEqual(aqi_category(INDIAN_CPCB_SCALE, 151), "Moderate")
        self.assertEqual(aqi_category(US_AQI_SCALE, 151), "Unhealthy")

    def test_values_outside_the_legend_are_not_assigned_a_category(self):
        self.assertIsNone(aqi_category(INDIAN_CPCB_SCALE, 501))
        self.assertIsNone(aqi_category(US_AQI_SCALE, 501))
        self.assertIsNone(aqi_category(US_AQI_SCALE, -1))
        self.assertIsNone(aqi_category(US_AQI_SCALE, float("nan")))

    def test_only_fresh_scale_matched_observation_can_be_current(self):
        waqi = {
            "available": True,
            "source": SOURCE_LABEL,
            "aqi_label": AQI_SCALE_LABEL,
            "freshness": "FRESH",
            "aqi": 151,
            "station_name": "Anand Vihar, Delhi, India",
            "observed_at": datetime.now(timezone.utc),
        }
        current = current_risk_observation(None, waqi)
        self.assertEqual(current["scale"], US_AQI_SCALE)
        self.assertEqual(current["category"], "Unhealthy")

        for freshness in ("DELAYED", "STALE", None):
            stale_or_unknown = {**waqi, "freshness": freshness}
            with self.subTest(freshness=freshness):
                self.assertIsNone(current_risk_observation(None, stale_or_unknown))

        wrong_scale = {**waqi, "aqi_label": "Indian National AQI"}
        self.assertIsNone(current_risk_observation(None, wrong_scale))

    def test_verified_fresh_cpcb_observation_takes_priority_on_indian_scale(self):
        cpcb = {
            "available": True,
            "aqi": 151,
            "category": "Moderate",
            "station": "R.K. Puram, Delhi, India",
            "location": "Delhi",
            "observed_at": datetime.now(timezone.utc),
            "freshness": "FRESH",
            "scale": "Indian National AQI",
            "source": "data.gov.in · CPCB-sourced resource",
        }
        waqi = {
            "available": True,
            "source": SOURCE_LABEL,
            "aqi_label": AQI_SCALE_LABEL,
            "freshness": "FRESH",
            "aqi": 151,
            "station_name": "Anand Vihar, Delhi, India",
            "observed_at": datetime.now(timezone.utc),
        }
        current = current_risk_observation(cpcb, waqi)
        self.assertEqual(current["scale"], INDIAN_CPCB_SCALE)
        self.assertEqual(current["category"], "Moderate")
        self.assertEqual(current["station"], "R.K. Puram, Delhi, India")

    def test_cpcb_mismatch_stale_or_invalid_value_does_not_create_marker(self):
        cpcb = {
            "available": True,
            "aqi": 180,
            "category": "Moderate",
            "station": "R.K. Puram, Delhi, India",
            "location": "Delhi",
            "observed_at": datetime.now(timezone.utc),
            "freshness": "STALE",
            "scale": "Indian National AQI",
            "source": "data.gov.in · CPCB-sourced resource",
        }
        self.assertIsNone(current_risk_observation(cpcb, None))
        cpcb["freshness"] = "FRESH"
        cpcb["category"] = "Poor"
        self.assertIsNone(current_risk_observation(cpcb, None))

    def test_us_value_above_500_is_reported_beyond_range_without_pointer_value(self):
        waqi = {
            "available": True,
            "source": SOURCE_LABEL,
            "aqi_label": AQI_SCALE_LABEL,
            "freshness": "FRESH",
            "aqi": 520,
            "station_name": "Anand Vihar, Delhi, India",
            "observed_at": datetime.now(timezone.utc),
        }
        current = current_risk_observation(None, waqi)
        self.assertEqual(current["scale"], US_AQI_SCALE)
        self.assertTrue(current["beyond_range"])
        self.assertIsNone(current["category"])


if __name__ == "__main__":
    unittest.main()
