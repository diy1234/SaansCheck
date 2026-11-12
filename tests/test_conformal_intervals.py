from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.forecast_features import FORECAST_FEATURE_COLUMNS
from src.conformal_intervals import (
    CALIBRATION_END,
    CALIBRATION_START,
    EVALUATION_END,
    EVALUATION_START,
    NOMINAL_COVERAGE,
    TRAIN_TARGET_END,
    conformal_quantile,
    interval_metrics,
    prediction_interval,
    validate_chronological_intervals,
)
from src.forecast_features import FORECAST_STATIONS


class MeanRegressor:
    def __init__(self, **kwargs):
        self.parameters = kwargs

    def fit(self, x, y):
        self.value = float(np.mean(y))
        self.feature_names_in_ = np.asarray(FORECAST_FEATURE_COLUMNS)
        return self

    def predict(self, x):
        return np.full(len(x), self.value, dtype=float)


def _synthetic_dataset() -> pd.DataFrame:
    dates = pd.date_range("2019-01-04", "2023-12-31", freq="D")
    rows = []
    for index, location in enumerate(FORECAST_STATIONS):
        aqi = 80 + index * 8 + np.arange(len(dates), dtype=float) * 0.02
        group = pd.DataFrame({
            "date": dates,
            "location_name": location,
            "aqi": aqi,
            "aqi_category": np.where(aqi < 100, "Moderate", "Poor"),
            "pm25": aqi * 0.5,
            "pm10": aqi * 0.7,
            "no2": aqi * 0.1,
            "so2": aqi * 0.03,
            "co": aqi * 0.01,
            "o3": aqi * 0.08,
        })
        group["AQI_lag_1"] = group.aqi.shift(1)
        group["AQI_lag_2"] = group.aqi.shift(2)
        group["AQI_lag_3"] = group.aqi.shift(3)
        group["AQI_rolling_mean_3"] = group.aqi.shift(1).rolling(3, min_periods=1).mean()
        group["AQI_rolling_mean_7"] = group.aqi.shift(1).rolling(7, min_periods=1).mean()
        group["AQI_rolling_std_7"] = group.aqi.shift(1).rolling(7, min_periods=2).std()
        group["month"] = group.date.dt.month
        group["day"] = group.date.dt.day
        group["day_of_week"] = group.date.dt.dayofweek
        group["is_weekend"] = (group.day_of_week >= 5).astype(int)
        for station in FORECAST_STATIONS:
            group[f"loc_{station}"] = int(location == station)
        for step in (1, 2, 3):
            group[f"AQI_t+{step}"] = group.aqi.shift(-step)
        rows.append(group.iloc[3:])
    return pd.concat(rows, ignore_index=True)


class ConformalIntervalTests(unittest.TestCase):
    def test_finite_sample_quantile_uses_conformal_rank(self):
        self.assertEqual(conformal_quantile([1, 2, 3, 4], coverage=0.6), 3.0)
        self.assertTrue(np.isinf(conformal_quantile([1, 2, 3, 4], coverage=0.9)))

    def test_quantile_rejects_invalid_calibration_scores(self):
        for scores in ([], [1, np.nan], [-1, 2]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                conformal_quantile(scores)
        with self.assertRaises(ValueError):
            conformal_quantile([1, 2], coverage=1.0)

    def test_prediction_intervals_are_finite_and_ordered(self):
        lower, upper = prediction_interval([10, 20], 2.5)
        np.testing.assert_allclose(lower, [7.5, 17.5])
        np.testing.assert_allclose(upper, [12.5, 22.5])
        self.assertTrue(np.isfinite(lower).all() and np.isfinite(upper).all())
        self.assertTrue((lower <= upper).all())
        with self.assertRaises(ValueError):
            prediction_interval([1], np.inf)

    def test_empirical_coverage_and_average_width(self):
        metrics = interval_metrics(
            actual=[0, 2, 5, 10],
            lower=[0, 1, 4, 9],
            upper=[1, 3, 6, 11],
        )
        self.assertEqual(metrics["n_evaluation"], 4)
        self.assertEqual(metrics["empirical_coverage"], 1.0)
        self.assertEqual(metrics["average_interval_width"], 1.75)

    def test_interval_metrics_reject_reversed_or_misaligned_bounds(self):
        with self.assertRaises(ValueError):
            interval_metrics([1], [2], [1])
        with self.assertRaises(ValueError):
            interval_metrics([1, 2], [0], [2])

    def test_chronological_split_is_separate_and_horizons_are_aligned(self):
        data = _synthetic_dataset()
        with patch(
            "src.conformal_intervals.RandomForestRegressor",
            MeanRegressor,
        ):
            results = validate_chronological_intervals(data)

        self.assertEqual(set(results["horizon"]), {"Day+1", "Day+2", "Day+3"})
        self.assertEqual(
            set(results["location"]),
            {"ALL", *FORECAST_STATIONS},
        )
        self.assertTrue(
            results["train_target_end"].eq(TRAIN_TARGET_END.date().isoformat()).all()
        )
        self.assertTrue(
            results["calibration_target_start"].eq(CALIBRATION_START.date().isoformat()).all()
        )
        self.assertTrue(
            results["calibration_target_end"].eq(CALIBRATION_END.date().isoformat()).all()
        )
        self.assertTrue(
            results["evaluation_target_start"].eq(EVALUATION_START.date().isoformat()).all()
        )
        self.assertTrue(
            results["evaluation_target_end"].eq(EVALUATION_END.date().isoformat()).all()
        )
        self.assertTrue(results["nominal_coverage"].eq(NOMINAL_COVERAGE).all())
        self.assertTrue(results["n_calibration"].gt(0).all())
        self.assertTrue(results["n_evaluation"].gt(0).all())
        self.assertTrue(
            np.isfinite(
                results.loc[results.location.ne("ALL"), "conformal_radius"]
            ).all()
        )
        self.assertTrue(np.isfinite(results["empirical_coverage"]).all())
        self.assertTrue(np.isfinite(results["average_interval_width"]).all())
        self.assertTrue(results["average_interval_width"].ge(0).all())
        for horizon, group in results.groupby("horizon"):
            overall = group.loc[group.location.eq("ALL")].iloc[0]
            station_rows = group.loc[group.location.ne("ALL")]
            self.assertEqual(
                int(station_rows["n_evaluation"].sum()),
                int(overall["n_evaluation"]),
                msg=horizon,
            )


if __name__ == "__main__":
    unittest.main()
