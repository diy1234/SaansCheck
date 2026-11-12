import unittest

import numpy as np
import pandas as pd

from src.dashboard_insights import (
    available_history_years,
    build_key_findings,
    build_location_leaderboard,
    build_month_calendar,
    summarize_walk_forward_results,
)


class DashboardInsightsTests(unittest.TestCase):
    def setUp(self):
        self.daily = pd.DataFrame({
            "date": [
                "2023-01-01", "2023-01-02", "2023-01-03",
                "2023-01-01", "2023-01-02", "2022-12-31",
                "invalid", "2023-01-04", "2023-01-01",
            ],
            "location_name": [
                "Delhi", "Delhi", "Delhi", "Noida", "Noida", "Delhi",
                "Delhi", "Noida", "NoData",
            ],
            "aqi": [80, 120, np.nan, 70, 90, 60, 100, 110, np.nan],
            "aqi_category": [
                "Satisfactory", "Moderate", None, "Satisfactory",
                "Satisfactory", "Satisfactory", "Moderate", "Moderate", None,
            ],
        })

    def test_key_findings_respect_location_year_and_missing_aqi(self):
        findings = build_key_findings(self.daily, "Delhi", year=2023)
        self.assertEqual(findings["count"], 2)
        self.assertEqual(findings["minimum_aqi"], 80)
        self.assertEqual(findings["maximum_aqi"], 120)
        self.assertEqual(findings["latest_aqi"], 120)
        self.assertEqual(findings["trend_change"], 40)
        self.assertEqual(findings["latest_date"], pd.Timestamp("2023-01-02"))

    def test_key_findings_handle_empty_and_single_observation_periods(self):
        empty = build_key_findings(self.daily, "Unknown", year=2023)
        self.assertEqual(empty["count"], 0)
        single = build_key_findings(self.daily, "Delhi", year=2022)
        self.assertEqual(single["count"], 1)
        self.assertIsNone(single["trend_change"])

    def test_year_choices_ignore_invalid_dates(self):
        self.assertEqual(available_history_years(self.daily), [2022, 2023])

    def test_location_leaderboard_supports_mean_max_counts_and_coverage(self):
        mean = build_location_leaderboard(self.daily, year=2023, metric="mean")
        self.assertEqual(mean.iloc[0]["location_name"], "Delhi")
        self.assertAlmostEqual(mean.iloc[0]["Average_AQI"], 100)
        noida = mean.loc[mean.location_name.eq("Noida")].iloc[0]
        self.assertEqual(int(noida["Days_Recorded"]), 3)
        self.assertEqual(noida["Coverage_Start"], pd.Timestamp("2023-01-01"))
        self.assertEqual(noida["Coverage_End"], pd.Timestamp("2023-01-04"))
        self.assertEqual(int(mean.loc[mean.location_name.eq("Delhi"), "Days_Recorded"].iloc[0]), 2)
        no_data = mean.loc[mean.location_name.eq("NoData")].iloc[0]
        self.assertEqual(int(no_data["Days_Recorded"]), 0)
        self.assertTrue(pd.isna(no_data["Average_AQI"]))

        maximum = build_location_leaderboard(self.daily, year=2023, metric="max")
        self.assertEqual(maximum.iloc[0]["location_name"], "Delhi")
        self.assertEqual(maximum.iloc[0]["Maximum_AQI"], 120)

    def test_location_leaderboard_rejects_unsupported_metric(self):
        with self.assertRaises(ValueError):
            build_location_leaderboard(self.daily, metric="median")

    def test_calendar_distinguishes_observed_missing_and_unobserved_days(self):
        result = build_month_calendar(self.daily, "Delhi", 2023, 1)
        self.assertEqual(result["values"][1], 80)
        self.assertIn(1, result["observed_days"])
        self.assertNotIn(5, result["observed_days"])
        self.assertIsNone(result["values"][3])
        self.assertEqual(len(result["weeks"]), 6)
        self.assertTrue(all(len(week) == 7 for week in result["weeks"]))
        self.assertEqual(len(result["records"]), 3)
        self.assertTrue(pd.isna(result["records"].iloc[2]["aqi"]))

    def test_calendar_keeps_an_observed_day_with_missing_aqi_as_missing(self):
        result = build_month_calendar(self.daily, "Delhi", 2023, 1)
        self.assertIn(3, result["observed_days"])
        self.assertIsNone(result["values"][3])

    def test_calendar_empty_month_has_no_invented_values(self):
        result = build_month_calendar(self.daily, "Delhi", 2023, 2)
        self.assertEqual(result["values"], {})
        self.assertEqual(result["observed_days"], set())
        self.assertTrue(result["records"].empty)

    def test_calendar_rejects_duplicate_location_dates(self):
        duplicate = pd.concat([self.daily, self.daily.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            build_month_calendar(duplicate, "Delhi", 2023, 1)

    def test_walk_forward_summary_uses_pooled_random_forest_windows(self):
        rows = []
        for window, base in (("fold-1", 6.0), ("fold-2", 7.0), ("fold-3", 8.0)):
            for horizon in ("Day+1", "Day+2", "Day+3"):
                rows.append({
                    "model": "Random Forest",
                    "location": "ALL",
                    "horizon": horizon,
                    "validation_window": window,
                    "validation_start": "2021-01-01",
                    "validation_end": "2021-12-31",
                    "mae": base,
                    "rmse": base + 1,
                })
        rows.append({
            **rows[0],
            "model": "Linear Regression",
            "mae": 999,
            "rmse": 999,
        })
        summary = summarize_walk_forward_results(pd.DataFrame(rows))
        self.assertEqual(summary["Day+1"]["window_count"], 3)
        self.assertEqual(summary["Day+1"]["mae"], 7)
        self.assertEqual(summary["Day+1"]["rmse"], 8)
        self.assertEqual(summary["Day+1"]["validation_start"], pd.Timestamp("2021-01-01"))

    def test_walk_forward_summary_omits_incomplete_windows(self):
        rows = pd.DataFrame([{
            "model": "Random Forest",
            "location": "ALL",
            "horizon": "Day+1",
            "validation_window": "fold-1",
            "validation_start": "2021-01-01",
            "validation_end": "2021-12-31",
            "mae": 6,
            "rmse": 8,
        }])
        self.assertEqual(summarize_walk_forward_results(rows), {})

    def test_repository_walk_forward_mae_values_are_verified_from_pooled_rows(self):
        from pathlib import Path

        results_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "walk_forward_results.csv"
        results = pd.read_csv(results_path)
        summary = summarize_walk_forward_results(results)
        self.assertAlmostEqual(summary["Day+1"]["mae"], 6.82845, places=5)
        self.assertAlmostEqual(summary["Day+2"]["mae"], 6.802084, places=5)
        self.assertAlmostEqual(summary["Day+3"]["mae"], 6.895022, places=5)


if __name__ == "__main__":
    unittest.main()
