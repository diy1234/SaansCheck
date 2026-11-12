from datetime import timedelta
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.forecast_features import (
    FORECAST_FEATURE_COLUMNS,
    build_direct_forecasts,
    build_forecast_feature_row,
)
from src.data_loader import load_trained_models


def _history(station="Delhi", final_aqi=100.0):
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    values = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 80.0, 90.0, final_aqi]
    rows = []
    for date, aqi in zip(dates, values):
        rows.append({
            "date": date,
            "location_name": station,
            "aqi": aqi,
            "pm25": aqi / 2,
            "pm10": aqi / 3,
            "no2": aqi / 4,
            "so2": aqi / 5,
            "co": aqi / 6,
            "o3": aqi / 7,
        })
    return pd.DataFrame(rows)


class CapturingModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.n_features_in_ = 22
        self.feature_names_in_ = np.array(FORECAST_FEATURE_COLUMNS)
        self.inputs = []

    def predict(self, features):
        self.inputs.append(features.copy())
        return np.array([self.prediction])


class ForecastFeatureTests(unittest.TestCase):
    def test_rolling_mean_3_excludes_current_and_future_aqi(self):
        features, _ = build_forecast_feature_row(_history(), "Delhi")
        self.assertEqual(features.loc[0, "aqi"], 100.0)
        self.assertEqual(features.loc[0, "AQI_rolling_mean_3"], (90 + 80 + 90) / 3)
        changed_current, _ = build_forecast_feature_row(
            _history(final_aqi=9999.0), "Delhi"
        )
        self.assertEqual(
            features.loc[0, "AQI_rolling_mean_3"],
            changed_current.loc[0, "AQI_rolling_mean_3"],
        )

    def test_rolling_mean_7_uses_seven_prior_observations(self):
        features, _ = build_forecast_feature_row(_history(), "Delhi")
        expected = np.mean([50, 60, 70, 80, 90, 80, 90])
        self.assertAlmostEqual(features.loc[0, "AQI_rolling_mean_7"], expected)

    def test_rolling_std_7_matches_training_sample_std_convention(self):
        features, _ = build_forecast_feature_row(_history(), "Delhi")
        prior_seven = [50, 60, 70, 80, 90, 80, 90]
        self.assertAlmostEqual(
            features.loc[0, "AQI_rolling_std_7"],
            float(pd.Series(prior_seven).std(ddof=1)),
        )
        self.assertNotAlmostEqual(
            features.loc[0, "AQI_rolling_std_7"],
            float(pd.Series(prior_seven).std(ddof=0)),
        )

    def test_lags_are_aligned_to_observed_history(self):
        features, _ = build_forecast_feature_row(_history(), "Delhi")
        self.assertEqual(features.loc[0, "AQI_lag_1"], 90.0)
        self.assertEqual(features.loc[0, "AQI_lag_2"], 80.0)
        self.assertEqual(features.loc[0, "AQI_lag_3"], 90.0)

    def test_location_one_hot_is_correct(self):
        for station in ("Delhi", "Faridabad", "Ghaziabad", "Gurgaon", "Noida"):
            features, _ = build_forecast_feature_row(_history(station), station)
            expected = [int(value == station) for value in
                        ("Delhi", "Faridabad", "Ghaziabad", "Gurgaon", "Noida")]
            actual = features.loc[0, [
                "loc_Delhi", "loc_Faridabad", "loc_Ghaziabad",
                "loc_Gurgaon", "loc_Noida",
            ]].astype(int).tolist()
            self.assertEqual(actual, expected)

    def test_feature_order_matches_saved_training_schema(self):
        expected = [
            "aqi", "AQI_lag_1", "AQI_lag_2", "AQI_lag_3",
            "AQI_rolling_mean_3", "AQI_rolling_mean_7", "AQI_rolling_std_7",
            "pm25", "pm10", "no2", "so2", "co", "o3",
            "month", "day", "day_of_week", "is_weekend",
            "loc_Delhi", "loc_Faridabad", "loc_Ghaziabad", "loc_Gurgaon", "loc_Noida",
        ]
        self.assertEqual(FORECAST_FEATURE_COLUMNS, expected)
        features, _ = build_forecast_feature_row(_history(), "Delhi")
        self.assertEqual(features.columns.tolist(), expected)
        self.assertEqual(features.shape, (1, 22))

    def test_each_direct_horizon_uses_observed_origin_not_previous_predictions(self):
        models = {
            f"rf_day{step}": CapturingModel(100.0 + step * 10)
            for step in (1, 2, 3)
        }
        history = _history()
        forecasts = build_direct_forecasts(history, "Delhi", models)

        self.assertEqual([row["horizon"] for row in forecasts],
                         ["Day +1", "Day +2", "Day +3"])
        for step in (1, 2, 3):
            model_input = models[f"rf_day{step}"].inputs[0]
            self.assertEqual(model_input.loc[0, "aqi"], 100.0)
            self.assertEqual(model_input.loc[0, "AQI_lag_1"], 90.0)
            self.assertEqual(model_input.loc[0, "AQI_rolling_mean_3"], (90 + 80 + 90) / 3)
            self.assertEqual(model_input.shape[1], 22)

        origin_date = history["date"].max()
        self.assertEqual(
            [row["date"] for row in forecasts],
            [origin_date + timedelta(days=step) for step in (1, 2, 3)],
        )

    def test_model_feature_order_mismatch_is_rejected(self):
        models = {
            f"rf_day{step}": CapturingModel(100.0)
            for step in (1, 2, 3)
        }
        models["rf_day2"].feature_names_in_ = np.array(list(reversed(FORECAST_FEATURE_COLUMNS)))
        with self.assertRaisesRegex(ValueError, "feature order"):
            build_direct_forecasts(_history(), "Delhi", models)

    def test_missing_history_is_not_silently_substituted(self):
        with self.assertRaisesRegex(ValueError, "four observed AQI records"):
            build_forecast_feature_row(_history().head(3), "Delhi")

    def test_features_match_training_dataset_rows(self):
        project_root = Path(__file__).resolve().parents[1]
        daily = pd.read_csv(
            project_root / "data" / "processed" / "daily_air_quality.csv",
            parse_dates=["date"],
        )
        ml_ready = pd.read_csv(
            project_root / "data" / "processed" / "ml_ready_dataset.csv",
            parse_dates=["date"],
        )
        expected = ml_ready.loc[
            (ml_ready["location_name"] == "Delhi")
            & (ml_ready["date"] == pd.Timestamp("2023-12-31"))
        ].iloc[0]
        history = daily.loc[
            (daily["location_name"] == "Delhi")
            & (daily["date"] <= pd.Timestamp("2023-12-31"))
        ]
        actual, _ = build_forecast_feature_row(history, "Delhi")
        for feature in FORECAST_FEATURE_COLUMNS:
            self.assertAlmostEqual(
                float(actual.loc[0, feature]),
                float(expected[feature]),
                places=10,
                msg=f"Training feature mismatch: {feature}",
            )

    def test_all_saved_models_load_and_historical_forecast_runs(self):
        project_root = Path(__file__).resolve().parents[1]
        models = load_trained_models(str(project_root / "models"))
        self.assertEqual(
            set(models),
            {"lr_day1", "rf_day1", "lr_day2", "rf_day2", "lr_day3", "rf_day3"},
        )
        for model in models.values():
            self.assertEqual(model.n_features_in_, 22)

        daily = pd.read_csv(
            project_root / "data" / "processed" / "daily_air_quality.csv",
            parse_dates=["date"],
        )
        forecasts = build_direct_forecasts(daily, "Delhi", models)
        self.assertEqual(len(forecasts), 3)
        self.assertEqual(
            [row["horizon"] for row in forecasts],
            ["Day +1", "Day +2", "Day +3"],
        )
        self.assertTrue(all(np.isfinite(row["predicted_aqi"]) for row in forecasts))


if __name__ == "__main__":
    unittest.main()
