from datetime import datetime, timezone
import unittest

import numpy as np
import pandas as pd

from src.forecast_features import FORECAST_FEATURE_COLUMNS
from src.live_forecast import build_live_forecast_status


class CapturingModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.n_features_in_ = 22
        self.feature_names_in_ = np.array(FORECAST_FEATURE_COLUMNS)

    def predict(self, features):
        return np.array([self.prediction])


def _compatible_history(station_name="Delhi"):
    dates = pd.date_range("2026-10-21", periods=7, freq="D")
    rows = []
    for idx, date in enumerate(dates):
        aqi = 50.0 + idx * 2
        rows.append({
            "date": date,
            "location_name": station_name,
            "aqi": aqi,
            "pm25": aqi / 2,
            "pm10": aqi / 3,
            "no2": aqi / 4,
            "so2": aqi / 5,
            "co": aqi / 6,
            "o3": aqi / 7,
        })
    history = pd.DataFrame(rows)
    history.attrs["aqi_scale_verified"] = True
    return history


def _observation(**overrides):
    result = {
        "requested_location": "Delhi",
        "available": True,
        "error": None,
        "source": "World Air Quality Index (WAQI)",
        "aqi": 88,
        "station_id": "2556",
        "station_name": "R.K. Puram, Delhi, India",
        "observed_at": datetime(2026, 10, 26, 19, 0, tzinfo=timezone.utc),
        "freshness": "FRESH",
    }
    result.update(overrides)
    return result


class LiveForecastStatusTests(unittest.TestCase):
    def test_targets_follow_station_observation_local_calendar_date(self):
        status = build_live_forecast_status(
            _observation(),
            expected_station_id="2556",
            now=datetime(2026, 10, 26, 20, 0, tzinfo=timezone.utc),
        )

        self.assertTrue(status["available"])
        self.assertTrue(status["observation_is_today"])
        self.assertEqual(status["forecast_origin"].strftime("%Y-%m-%d %H:%M %Z"),
                         "2026-10-27 00:30 IST")
        self.assertEqual(
            {key: value.strftime("%Y-%m-%d") for key, value in status["target_dates"].items()},
            {
                "today": "2026-10-27",
                "tomorrow": "2026-10-28",
                "day_after": "2026-10-29",
            },
        )
        self.assertIsNone(status["predictions"]["tomorrow"])
        self.assertIsNone(status["predictions"]["day_after"])
        self.assertIsNone(status["today_estimate"])

    def test_older_source_date_is_not_mislabeled_as_todays_observation(self):
        status = build_live_forecast_status(
            _observation(),
            expected_station_id="2556",
            now=datetime(2026, 10, 27, 20, 0, tzinfo=timezone.utc),
        )

        self.assertFalse(status["observation_is_today"])
        self.assertIsNotNone(status["observed_aqi"])

    def test_stale_observation_has_origin_dates_but_no_predictions(self):
        status = build_live_forecast_status(
            _observation(freshness="STALE"),
            expected_station_id="2556",
        )

        self.assertEqual(status["target_dates"]["tomorrow"].strftime("%Y-%m-%d"),
                         "2026-10-28")
        self.assertIn("not fresh", status["reason"])
        self.assertIsNone(status["predictions"]["tomorrow"])

    def test_compatible_history_generates_real_predicted_values(self):
        prediction_map = {1: 101.5, 2: 103.25, 3: 104.75}
        models = {f"rf_day{step}": CapturingModel(prediction_map[step]) for step in (1, 2, 3)}
        history = _compatible_history()
        status = build_live_forecast_status(
            _observation(),
            expected_station_id="2556",
            candidate_daily_history=history,
            models_dict=models,
        )

        self.assertTrue(status["available"])
        self.assertEqual(status["target_dates"]["tomorrow"].strftime("%Y-%m-%d"), "2026-10-28")
        self.assertEqual(status["predictions"]["tomorrow"], 101.5)
        self.assertEqual(status["predictions"]["day_after"], 103.25)
        self.assertIn("Compatible station-matched historical daily data", status["compatibility_reason"])

    def test_missing_or_unverified_history_cannot_produce_forecast(self):
        history = _compatible_history()
        history.attrs["aqi_scale_verified"] = False
        models = {f"rf_day{step}": CapturingModel(99.0 + step) for step in (1, 2, 3)}
        status = build_live_forecast_status(
            _observation(),
            expected_station_id="2556",
            candidate_daily_history=history,
            models_dict=models,
        )

        self.assertIsNone(status["predictions"]["tomorrow"])
        self.assertIn("explicitly verified", status["compatibility_reason"])

    def test_station_id_and_city_mismatch_are_rejected(self):
        wrong_id = build_live_forecast_status(
            _observation(), expected_station_id="999"
        )
        wrong_city = build_live_forecast_status(
            _observation(station_name="Sector 62, Noida, India"),
            expected_station_id="2556",
        )

        self.assertIn("station ID", wrong_id["reason"])
        self.assertIsNone(wrong_id["forecast_origin"])
        self.assertIn("does not match the selected city", wrong_city["reason"])
        self.assertIsNone(wrong_city["forecast_origin"])

    def test_api_error_invalid_aqi_and_missing_timestamp_have_no_forecast_origin(self):
        cases = (
            _observation(available=False, error="WAQI request timed out."),
            _observation(aqi=None),
            _observation(observed_at=None),
        )
        for observation in cases:
            with self.subTest(observation=observation):
                status = build_live_forecast_status(
                    observation, expected_station_id="2556"
                )
                self.assertFalse(status["available"])
                self.assertIsNone(status["forecast_origin"])
                self.assertIsNone(status["predictions"]["tomorrow"])

    def test_latest_reading_is_not_enough_to_claim_a_model_forecast(self):
        observation = _observation()
        observation["pollutants"] = {
            "PM2.5": {"value": 30.0, "unit": "µg/m³"}
        }
        status = build_live_forecast_status(
            observation, expected_station_id="2556"
        )

        self.assertIn("station-matched", status["compatibility_reason"])
        self.assertEqual(status["predictions"], {"tomorrow": None, "day_after": None})
        self.assertIsNone(status["today_estimate"])

    def test_missing_pollutants_never_get_filled_with_historical_or_example_values(self):
        status = build_live_forecast_status(
            _observation(pollutants={
                "PM2.5": {"value": None, "unit": "Unit not specified by source"},
                "PM10": {"value": None, "unit": "Unit not specified by source"},
            }),
            expected_station_id="2556",
        )

        self.assertEqual(status["predictions"], {"tomorrow": None, "day_after": None})
        self.assertIn("AQI scale", status["compatibility_reason"])

    def test_duplicate_or_missing_history_cannot_be_silently_filled(self):
        observation = _observation()
        observation["candidate_daily_history"] = [
            {"station_id": "2556", "date": "2026-10-24", "aqi": 75},
            {"station_id": "2556", "date": "2026-10-24", "aqi": 76},
            {"station_id": "2556", "date": "2026-10-26", "aqi": 83},
        ]
        status = build_live_forecast_status(
            observation, expected_station_id="2556"
        )

        self.assertEqual(status["predictions"], {"tomorrow": None, "day_after": None})
        self.assertIn("WAQI-authorized, station-matched daily history",
                      status["reason"])


if __name__ == "__main__":
    unittest.main()
