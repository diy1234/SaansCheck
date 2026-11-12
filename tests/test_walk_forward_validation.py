from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.forecast_features import FORECAST_FEATURE_COLUMNS, FORECAST_STATIONS
from src.walk_forward_validation import (
    HORIZONS,
    VALIDATION_WINDOWS,
    _validate_dataset,
    calculate_metrics,
    evaluate_walk_forward,
    mae_improvement_percent,
    persistence_predictions,
    skill_score,
    split_window,
)


class MeanRegressor:
    def __init__(self, **kwargs):
        self.parameters = kwargs

    def fit(self, x, y):
        self.value = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.value, dtype=float)


def _synthetic_dataset() -> pd.DataFrame:
    dates = pd.date_range("2019-01-04", "2023-12-31", freq="D")
    all_rows = []
    for location_index, location in enumerate(FORECAST_STATIONS):
        aqi = 80 + location_index * 8 + np.arange(len(dates), dtype=float) * 0.02
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
        all_rows.append(group.iloc[3:])
    return pd.concat(all_rows, ignore_index=True)


class WalkForwardValidationTests(unittest.TestCase):
    def test_validation_windows_are_strictly_chronological(self):
        for window in VALIDATION_WINDOWS:
            self.assertGreater(window.validation_start, window.train_cutoff)
            self.assertEqual(window.validation_start.year, window.validation_year)
            self.assertEqual(window.train_cutoff.year, window.train_through_year)

        frame = _validate_dataset(_synthetic_dataset())
        for window in VALIDATION_WINDOWS:
            train, validation = split_window(frame, window, horizon=3)
            self.assertLessEqual(
                train["_target_date_3"].max(),
                window.train_cutoff,
            )
            self.assertGreater(
                validation["_target_date_3"].min(),
                window.train_cutoff,
            )

    def test_validation_design_uses_calendar_windows_not_shuffle(self):
        self.assertEqual(
            [(window.train_through_year, window.validation_year) for window in VALIDATION_WINDOWS],
            [(2020, 2021), (2021, 2022), (2022, 2023)],
        )
        self.assertEqual(
            [window.validation_year for window in VALIDATION_WINDOWS],
            sorted(window.validation_year for window in VALIDATION_WINDOWS),
        )

    def test_all_three_horizons_and_five_locations_are_evaluated(self):
        with patch("src.walk_forward_validation.LinearRegression", MeanRegressor), patch(
            "src.walk_forward_validation.RandomForestRegressor", MeanRegressor
        ):
            results, _ = evaluate_walk_forward(_synthetic_dataset())
        self.assertEqual(set(results["horizon"]), {label for label, _, _ in HORIZONS})
        location_rows = results.loc[~results.location.eq("ALL")]
        self.assertEqual(set(location_rows.location), set(FORECAST_STATIONS))
        self.assertEqual(set(results.model), {"Linear Regression", "Random Forest", "Persistence"})

    def test_persistence_uses_only_observed_origin_aqi(self):
        sample = pd.DataFrame({"aqi": [101.0, 102.0], "AQI_t+1": [999.0, 888.0]})
        first = persistence_predictions(sample)
        sample["AQI_t+1"] = [-1.0, -2.0]
        second = persistence_predictions(sample)
        np.testing.assert_array_equal(first, [101.0, 102.0])
        np.testing.assert_array_equal(first, second)

    def test_future_target_columns_do_not_enter_predictors(self):
        self.assertEqual(len(FORECAST_FEATURE_COLUMNS), 22)
        self.assertFalse(any(column.startswith("AQI_t+") for column in FORECAST_FEATURE_COLUMNS))
        frame = _validate_dataset(_synthetic_dataset())
        train, _ = split_window(frame, VALIDATION_WINDOWS[0], horizon=1)
        x_train = train[FORECAST_FEATURE_COLUMNS]
        self.assertFalse(set(column for _, column, _ in HORIZONS).intersection(x_train.columns))

    def test_results_have_finite_metrics_and_no_infinite_values(self):
        with patch("src.walk_forward_validation.LinearRegression", MeanRegressor), patch(
            "src.walk_forward_validation.RandomForestRegressor", MeanRegressor
        ):
            results, _ = evaluate_walk_forward(_synthetic_dataset())
        metric_cols = ["mae", "rmse", "r2", "persistence_mae",
                       "skill_vs_persistence", "mae_improvement_vs_persistence"]
        self.assertTrue(np.isfinite(results[metric_cols].to_numpy(dtype=float)).all())

    def test_skill_and_improvement_formulas(self):
        self.assertAlmostEqual(skill_score(6.0, 10.0), 0.4)
        self.assertAlmostEqual(mae_improvement_percent(10.0, 6.0), 40.0)
        with self.assertRaises(ValueError):
            skill_score(1.0, 0.0)

    def test_metrics_are_calculated_without_accuracy_relabeling(self):
        metrics = calculate_metrics([1, 2, 3], [1, 2, 4])
        self.assertAlmostEqual(metrics["mae"], 1 / 3)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(1 / 3))
        self.assertIn("r2", metrics)
        self.assertNotIn("accuracy", metrics)

    def test_production_model_artifacts_are_not_overwritten(self):
        root = Path(__file__).resolve().parents[1]
        model_paths = sorted((root / "models").glob("*.joblib"))
        self.assertEqual(len(model_paths), 6)

        def hashes():
            return {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in model_paths
            }

        before = hashes()
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.walk_forward_validation.LinearRegression", MeanRegressor
        ), patch("src.walk_forward_validation.RandomForestRegressor", MeanRegressor):
            from src.walk_forward_validation import run_walk_forward_validation

            data_path = Path(temp_dir) / "ml_ready_dataset.csv"
            output_path = Path(temp_dir) / "output"
            _synthetic_dataset().to_csv(data_path, index=False)
            run_walk_forward_validation(data_path, output_path)
        self.assertEqual(before, hashes())


if __name__ == "__main__":
    unittest.main()
