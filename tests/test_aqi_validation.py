"""Focused tests for the research-only AQI validation workflow."""

import unittest

import pandas as pd

from src.aqi_validation import build_daily_comparisons


def hourly_rows(start, hours, location="Test"):
    timestamps = pd.date_range(start, periods=hours, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "location_name": location,
            "aqi": 80.0,
            "pm25": 20.0,
            "pm10": 25.0,
            "no2": 20.0,
            "so2": 20.0,
            "co": 0.5,
            "o3": 30.0,
        }
    )


class AQIValidationTests(unittest.TestCase):
    def test_complete_day_uses_daily_mean_and_complete_eight_hour_averages(self):
        raw = hourly_rows("2023-02-01 00:00", 24)
        raw.loc[raw.index[-1], "o3"] = 70.0

        result = build_daily_comparisons(raw).iloc[0]

        self.assertEqual(result["comparison_status"], "comparable")
        self.assertEqual(result["source_aqi_valid_hours"], 24)
        self.assertEqual(result["pm25_daily_or_max_8h_concentration"], 20.0)
        self.assertEqual(result["co_valid_8h_window_count"], 17)
        self.assertEqual(result["o3_valid_8h_window_count"], 17)
        self.assertEqual(result["o3_daily_or_max_8h_concentration"], 35.0)
        self.assertEqual(result["original_dataset_aqi"], 80.0)
        self.assertIsNotNone(result["cpcb_style_calculated_aqi"])

    def test_incomplete_day_is_retained_as_excluded_not_calculated(self):
        raw = hourly_rows("2023-02-01 00:00", 23)

        result = build_daily_comparisons(raw).iloc[0]

        self.assertEqual(result["comparison_status"], "excluded")
        self.assertIsNone(result["original_dataset_aqi"])
        self.assertIsNone(result["cpcb_style_calculated_aqi"])
        self.assertIn("24 consecutive hourly timestamps", result["exclusion_reason"])

    def test_nonconsecutive_observations_do_not_form_eight_hour_window(self):
        raw = hourly_rows("2023-02-01 00:00", 24)
        raw.loc[[1, 4, 7, 10, 13, 16, 19, 22], ["co", "o3"]] = None

        result = build_daily_comparisons(raw).iloc[0]

        self.assertEqual(result["comparison_status"], "comparable")
        self.assertEqual(result["co_valid_hour_count"], 16)
        self.assertEqual(result["co_valid_8h_window_count"], 0)
        self.assertEqual(result["co_sub_index_status"], "missing")
        self.assertIsNone(result["co_daily_or_max_8h_concentration"])
        self.assertIn(
            "no complete consecutive 8-hour window",
            result["pollutant_validation_notes"],
        )

    def test_duplicate_location_timestamp_is_rejected(self):
        raw = hourly_rows("2023-02-01 00:00", 24)
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "Duplicate location/timestamp"):
            build_daily_comparisons(raw)

    def test_unresolved_eight_hour_ozone_withholds_overall_aqi(self):
        raw = hourly_rows("2023-02-01 00:00", 24)
        raw["o3"] = 300.0

        result = build_daily_comparisons(raw).iloc[0]

        self.assertEqual(result["comparison_status"], "excluded")
        self.assertEqual(
            result["o3_sub_index_status"], "averaging_breakpoint_mismatch"
        )
        self.assertEqual(result["calculator_status"], "unresolved_numeric_sub_index")
        self.assertIsNone(result["cpcb_style_calculated_aqi"])
        self.assertIn("8-hour sub-index unresolved", result["exclusion_reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
