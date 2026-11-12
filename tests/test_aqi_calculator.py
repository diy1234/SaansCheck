"""Focused validation tests for the standalone CPCB-style AQI calculator."""

import math
import unittest

from src.aqi_calculator import PollutantReading, calculate_aqi


def reading(value, pollutant, *, unit=None, period=None, valid_hours=24):
    """Build a valid default reading for one pollutant."""
    units = unit or ("mg/m³" if pollutant == "CO" else "µg/m³")
    periods = period or (8 if pollutant in {"CO", "O3"} else 24)
    return PollutantReading(value, units, periods, valid_hours)


class CPCBAQICalculatorTests(unittest.TestCase):
    def test_known_breakpoint_interpolation(self):
        result = calculate_aqi(
            {
                "PM2.5": reading(31, "PM2.5"),
                "PM10": reading(100, "PM10"),
                "NO2": reading(40, "NO2"),
            }
        )
        self.assertEqual(result.overall_aqi, 100)
        self.assertEqual(result.category, "Satisfactory")
        self.assertEqual(result.dominant_pollutant, "PM10")
        self.assertAlmostEqual(result.pollutant_sub_indices["PM2.5"].sub_index, 50 + 50 / 30)
        self.assertEqual(result.pollutant_sub_indices["NO2"].sub_index, 50)
        self.assertEqual(result.status, "valid_with_warnings")

    def test_ambiguous_duplicate_pollutants_are_excluded(self):
        result = calculate_aqi(
            {
                "PM2.5": reading(20, "PM2.5"),
                "pm25": reading(25, "PM2.5"),
                "PM10": reading(30, "PM10"),
                "NO2": reading(30, "NO2"),
                "SO2": reading(20, "SO2"),
            }
        )
        self.assertEqual(
            result.pollutant_sub_indices["PM2.5"].status,
            "duplicate_readings",
        )
        self.assertEqual(result.overall_aqi, 37.5)
        self.assertEqual(result.status, "valid_with_warnings")

    def test_overall_requires_three_pollutants_and_particulate(self):
        two_pollutants = calculate_aqi(
            {"PM2.5": reading(20, "PM2.5"), "NO2": reading(30, "NO2")}
        )
        no_particulate = calculate_aqi(
            {
                "NO2": reading(30, "NO2"),
                "SO2": reading(30, "SO2"),
                "CO": reading(0.5, "CO"),
            }
        )
        self.assertEqual(two_pollutants.status, "insufficient_pollutants")
        self.assertIsNone(two_pollutants.overall_aqi)
        self.assertIsNone(no_particulate.overall_aqi)
        self.assertTrue(
            any("PM2.5 or PM10" in text for text in no_particulate.validation_messages)
        )

    def test_missing_and_insufficient_averaging_are_reported(self):
        result = calculate_aqi(
            {
                "PM2.5": reading(20, "PM2.5", valid_hours=15),
                "PM10": None,
                "NO2": reading(30, "NO2"),
            }
        )
        self.assertEqual(
            result.pollutant_sub_indices["PM2.5"].status,
            "insufficient_or_invalid_data",
        )
        self.assertEqual(result.pollutant_sub_indices["PM10"].status, "missing")
        self.assertIsNone(result.overall_aqi)

    def test_invalid_value_unit_and_period_are_not_substituted(self):
        result = calculate_aqi(
            {
                "PM2.5": reading(-1, "PM2.5"),
                "PM10": reading(20, "PM10", unit="ppm"),
                "NO2": reading(30, "NO2", period=8),
                "SO2": reading(float("nan"), "SO2"),
            }
        )
        self.assertEqual(result.pollutant_sub_indices["PM2.5"].status, "invalid_value")
        self.assertEqual(
            result.pollutant_sub_indices["PM10"].status,
            "unknown_or_incompatible_unit",
        )
        self.assertEqual(
            result.pollutant_sub_indices["NO2"].status,
            "invalid_averaging_period",
        )
        self.assertEqual(result.pollutant_sub_indices["SO2"].status, "invalid_value")
        self.assertIsNone(result.overall_aqi)

    def test_values_above_final_breakpoint_have_no_invented_numeric_index(self):
        result = calculate_aqi(
            {
                "PM2.5": reading(251, "PM2.5"),
                "PM10": reading(100, "PM10"),
                "NO2": reading(40, "NO2"),
            }
        )
        self.assertEqual(
            result.pollutant_sub_indices["PM2.5"].status,
            "severe_numeric_rule_unspecified",
        )
        self.assertEqual(result.pollutant_sub_indices["PM2.5"].category, "Severe")
        self.assertIsNone(result.overall_aqi)
        self.assertEqual(result.category, "Severe")
        self.assertEqual(result.dominant_pollutant, "PM2.5")

    def test_ozone_does_not_apply_starred_one_hour_cutoff_to_eight_hour_input(self):
        result = calculate_aqi(
            {
                "PM2.5": reading(20, "PM2.5"),
                "PM10": reading(30, "PM10"),
                "NO2": reading(30, "NO2"),
                "O3": reading(300, "O3", period=8),
            }
        )
        ozone = result.pollutant_sub_indices["O3"]
        self.assertEqual(ozone.status, "averaging_breakpoint_mismatch")
        self.assertIsNone(ozone.sub_index)
        self.assertIsNone(ozone.category)
        self.assertEqual(result.status, "unresolved_numeric_sub_index")
        self.assertIsNone(result.overall_aqi)
        self.assertIsNone(result.category)

    def test_all_six_pollutants_and_normalized_units(self):
        result = calculate_aqi(
            {
                "pm25": reading(20, "PM2.5", unit="ug/m3"),
                "PM10": reading(25, "PM10", unit="µg/m³"),
                "NO2": reading(20, "NO2", unit="μg/m³"),
                "SO2": reading(20, "SO2"),
                "CO": reading(0.5, "CO"),
                "O3": reading(30, "O3", period=8),
            }
        )
        self.assertEqual(result.status, "valid")
        self.assertEqual(len(result.pollutant_sub_indices), 6)
        self.assertIsNotNone(result.overall_aqi)

    def test_infinite_values_are_rejected(self):
        result = calculate_aqi(
            {
                "PM2.5": reading(math.inf, "PM2.5"),
                "PM10": reading(20, "PM10"),
                "NO2": reading(30, "NO2"),
            }
        )
        self.assertEqual(result.pollutant_sub_indices["PM2.5"].status, "invalid_value")
        self.assertIsNone(result.overall_aqi)


if __name__ == "__main__":
    unittest.main(verbosity=2)
