from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

import requests

from src import cpcb_national_aqi as cpcb


NOW = datetime.now(timezone.utc)
FIELD_MAP = {
    "location": "city",
    "station": "station",
    "observed_at": "last_update",
    "aqi": "overall_aqi",
}


def _record(**updates):
    result = {
        "city": "Delhi",
        "station": "R.K. Puram, Delhi, India",
        "last_update": NOW.isoformat(),
        "overall_aqi": 180,
    }
    result.update(updates)
    return result


class CpcbNationalAqiTests(unittest.TestCase):
    def setUp(self):
        cpcb.load_cpcb_national_aqi.clear()
        cpcb._request_records.clear()

    def _load(self, records, station="R.K. Puram, Delhi, India"):
        with (
            patch.object(cpcb, "_api_key", return_value="not-displayed"),
            patch.object(
                cpcb,
                "_schema_configuration",
                return_value=("records", FIELD_MAP),
            ),
            patch.object(
                cpcb,
                "_request_records",
                return_value={"payload": {"records": records}},
            ),
        ):
            return cpcb.load_cpcb_national_aqi("Delhi", station)

    def test_api_key_uses_requested_name_and_legacy_name_without_output(self):
        with patch.object(
            cpcb,
            "_setting",
            side_effect=lambda name: {
                "CPCB_OGD_API_KEY": "configured",
                "CPCB_API_KEY": "legacy",
            }.get(name),
        ):
            self.assertEqual(cpcb._api_key(), "configured")
        with patch.object(
            cpcb,
            "_setting",
            side_effect=lambda name: {"CPCB_API_KEY": "legacy"}.get(name),
        ):
            self.assertEqual(cpcb._api_key(), "legacy")

    def test_missing_key_fails_closed(self):
        with patch.object(cpcb, "_api_key", return_value=None):
            result = cpcb.load_cpcb_national_aqi("Delhi", "R.K. Puram")
        self.assertFalse(result["available"])
        self.assertIsNone(result["aqi"])
        self.assertIn("CPCB_OGD_API_KEY", result["error"])

    def test_unverified_schema_does_not_make_an_api_request(self):
        with (
            patch.object(cpcb, "_api_key", return_value="not-displayed"),
            patch.object(cpcb, "_schema_configuration", return_value=None),
            patch.object(cpcb, "_request_records") as request,
        ):
            result = cpcb.load_cpcb_national_aqi("Delhi", "R.K. Puram")
        self.assertFalse(result["available"])
        self.assertIsNone(result["aqi"])
        self.assertIn("schema", result["error"])
        request.assert_not_called()

    def test_valid_direct_overall_aqi_is_reported_with_cpcb_category(self):
        result = self._load([_record()])
        self.assertTrue(result["available"])
        self.assertEqual(result["aqi"], 180)
        self.assertEqual(result["category"], "Moderate")
        self.assertEqual(result["scale"], "Indian National AQI")
        self.assertEqual(result["station"], "R.K. Puram, Delhi, India")

    def test_missing_overall_aqi_is_not_replaced_by_pollutant_measurement(self):
        result = self._load([
            _record(overall_aqi=None, pollutant_avg=156, pollutant_id="PM2.5")
        ])
        self.assertFalse(result["available"])
        self.assertIsNone(result["aqi"])

    def test_station_mismatch_is_not_presented_as_selected_station(self):
        result = self._load([_record(station="Anand Vihar, Delhi, India")])
        self.assertFalse(result["available"])
        self.assertIsNone(result["aqi"])
        self.assertIn("exact station", result["error"])

    def test_stale_record_does_not_display_aqi(self):
        result = self._load([
            _record(last_update=(NOW - timedelta(hours=7)).isoformat())
        ])
        self.assertFalse(result["available"])
        self.assertIsNone(result["aqi"])
        self.assertIn("stale", result["error"])

    def test_cpcb_category_thresholds(self):
        expectations = {
            0: "Good",
            50: "Good",
            51: "Satisfactory",
            100: "Satisfactory",
            101: "Moderate",
            200: "Moderate",
            201: "Poor",
            300: "Poor",
            301: "Very Poor",
            400: "Very Poor",
            401: "Severe",
            500: "Severe",
        }
        for value, expected in expectations.items():
            with self.subTest(value=value):
                self.assertEqual(cpcb.cpcb_aqi_category(value), expected)
        for value in (None, -1, 501, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                self.assertIsNone(cpcb.cpcb_aqi_category(value))

    def test_schema_refuses_pollutant_average_as_overall_aqi(self):
        with (
            patch.object(cpcb, "_setting", side_effect={
                "CPCB_OGD_RECORDS_PATH": "records",
                "CPCB_OGD_FIELD_MAP": {
                    "location": "city",
                    "station": "station",
                    "observed_at": "last_update",
                    "aqi": "pollutant_avg",
                },
                "CPCB_OGD_AQI_IS_OFFICIAL_NATIONAL": "true",
            }.get),
        ):
            self.assertIsNone(cpcb._schema_configuration())

    def test_rate_limit_and_timeout_responses_are_sanitized(self):
        with patch.object(
            cpcb.requests,
            "get",
            return_value=type("Response", (), {"status_code": 429, "ok": False})(),
        ):
            limited = cpcb._request_records("secret")
        self.assertIn("rate limit", limited["error"])
        self.assertNotIn("secret", limited["error"])

        cpcb._request_records.clear()
        with patch.object(cpcb.requests, "get", side_effect=requests.Timeout):
            timed_out = cpcb._request_records("secret")
        self.assertIn("timed out", timed_out["error"])
        self.assertNotIn("secret", timed_out["error"])

    def test_access_denial_and_invalid_json_are_sanitized(self):
        forbidden = type(
            "Response",
            (),
            {"status_code": 403, "ok": False},
        )()
        with patch.object(cpcb.requests, "get", return_value=forbidden):
            denied = cpcb._request_records("secret")
        self.assertIn("rejected", denied["error"])
        self.assertNotIn("secret", denied["error"])

        invalid_json = type(
            "Response",
            (),
            {
                "status_code": 200,
                "ok": True,
                "json": lambda self: (_ for _ in ()).throw(ValueError()),
            },
        )()
        cpcb._request_records.clear()
        with patch.object(cpcb.requests, "get", return_value=invalid_json):
            invalid = cpcb._request_records("different-secret")
        self.assertIn("not valid JSON", invalid["error"])
        self.assertNotIn("different-secret", invalid["error"])

    def test_cpcb_guidance_and_alert_use_fresh_indian_scale_observation(self):
        observation = self._load([_record()])
        self.assertEqual(
            cpcb.build_cpcb_health_guidance(observation)["scale"],
            "Indian National AQI",
        )
        self.assertEqual(
            cpcb.build_cpcb_observed_alert(observation)["category"],
            "Moderate",
        )
        observation["freshness"] = "DELAYED"
        self.assertIsNone(cpcb.build_cpcb_health_guidance(observation))
        self.assertIsNone(cpcb.build_cpcb_observed_alert(observation))


if __name__ == "__main__":
    unittest.main()
