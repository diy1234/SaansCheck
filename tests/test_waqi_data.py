from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
import unittest

import requests

from src.waqi_data import (
    AQI_UNAVAILABLE,
    CACHE_TTL_SECONDS,
    SEARCH_URL,
    WaqiRequestError,
    _fetch_json,
    _freshness,
    _parse_timestamp,
    _pollutant_readings,
    load_city_stations,
    load_nearby_station_observation,
    load_station_observation,
    merge_station_search_results,
    refresh_observation_freshness,
)


NOW = datetime(2026, 10, 26, 10, tzinfo=timezone.utc)


def _response(payload, status_code=200, ok=True):
    response = Mock()
    response.ok = ok
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _feed_payload(
    *,
    name="R.K. Puram, Delhi, Delhi, India",
    station_id=2556,
    aqi="160",
    time_data=None,
):
    data = {
        "idx": station_id,
        "aqi": aqi,
        "city": {"name": name},
        "iaqi": {"pm25": {"v": 18.7, "unit": "µg/m³"}},
        "attributions": [{"name": "Delhi Pollution Control Committee"}],
    }
    if time_data is not None:
        data["time"] = time_data
    return {"status": "ok", "data": data}


class WaqiDataTests(unittest.TestCase):
    def setUp(self):
        _fetch_json.clear()

    def tearDown(self):
        _fetch_json.clear()

    def _mock_token(self):
        return patch("src.waqi_data._load_token", return_value="test-token")

    def test_freshness_thresholds_never_use_live_label(self):
        now = datetime(2026, 10, 8, 12, tzinfo=timezone.utc)
        self.assertEqual(_freshness(now - timedelta(minutes=60), now)[0], "FRESH")
        self.assertEqual(_freshness(now - timedelta(minutes=61), now)[0], "DELAYED")
        self.assertEqual(_freshness(now - timedelta(hours=6), now)[0], "DELAYED")
        self.assertEqual(_freshness(now - timedelta(hours=6, minutes=1), now)[0], "STALE")
        self.assertIsNone(_freshness(now + timedelta(minutes=1), now))

    def test_retained_observation_freshness_is_recalculated_from_source_time(self):
        observation = {
            "available": True,
            "aqi": 120,
            "aqi_display": "120",
            "observed_at": NOW - timedelta(hours=7),
            "freshness": "FRESH",
            "age_minutes": 1,
        }
        refreshed = refresh_observation_freshness(observation, now=NOW)
        self.assertEqual(refreshed["freshness"], "STALE")
        self.assertEqual(refreshed["age_minutes"], 420)

        future = refresh_observation_freshness(
            observation,
            now=NOW - timedelta(hours=8),
        )
        self.assertFalse(future["available"])
        self.assertIsNone(future["aqi"])
        self.assertIn("future", future["error"])

    def test_refresh_keeps_freshness_in_sync_with_zoned_source_timestamp(self):
        observed_at = datetime.fromisoformat("2026-10-26T15:30:00+05:30")
        observation = {
            "available": True,
            "aqi": 120,
            "aqi_display": "120",
            "observed_at": observed_at,
            "freshness": "STALE",
            "age_minutes": 999,
        }

        refreshed = refresh_observation_freshness(
            observation,
            now=datetime(2026, 10, 26, 10, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(refreshed["freshness"], "FRESH")
        self.assertEqual(refreshed["age_minutes"], 30)
        self.assertEqual(refreshed["observed_at"], observed_at)
        self.assertEqual(refreshed["aqi"], 120)
        self.assertEqual(observation["freshness"], "STALE")
        self.assertEqual(observation["age_minutes"], 999)

    def test_refresh_marks_missing_or_invalid_timestamp_unverified(self):
        observation = {
            "available": True,
            "aqi": 120,
            "aqi_display": "120",
            "observed_at": NOW,
            "freshness": "FRESH",
            "age_minutes": 1,
        }

        for invalid_timestamp in (None, "not-a-timestamp", datetime(2026, 10, 26, 10)):
            with self.subTest(observed_at=invalid_timestamp):
                observation["observed_at"] = invalid_timestamp
                refreshed = refresh_observation_freshness(observation, now=NOW)

                self.assertTrue(refreshed["available"])
                self.assertEqual(refreshed["aqi"], 120)
                self.assertIsNone(refreshed["freshness"])
                self.assertIsNone(refreshed["age_minutes"])

    def test_parse_accepts_zoned_iso_and_waqi_epoch_fields(self):
        self.assertIsNone(_parse_timestamp({"iso": "2026-10-08T12:00:00"}))
        timestamp = _parse_timestamp({"iso": "2026-10-08T12:00:00+05:30"})
        self.assertEqual(timestamp.tzinfo, timezone.utc)
        station_time = _parse_timestamp({
            "s": "2026-10-08 12:00:00",
            "tz": "+05:30",
        })
        self.assertEqual(
            station_time,
            datetime(2026, 10, 8, 6, 30, tzinfo=timezone.utc),
        )
        self.assertIsNone(_parse_timestamp({"iso": "not a timestamp"}))
        self.assertIsNone(_parse_timestamp({"vtime": 0}))
        self.assertIsNotNone(_parse_timestamp({"s": "1791460800"}))
        self.assertIsNotNone(_parse_timestamp({"vtime": 1791460800}))

    def test_pollutants_preserve_values_and_only_source_units(self):
        readings, warnings = _pollutant_readings({
            "pm25": {"v": 12.5},
            "pm10": {"v": "bad"},
        })
        self.assertEqual(readings["PM2.5"]["value"], 12.5)
        self.assertEqual(readings["PM10"]["value"], None)
        self.assertEqual(readings["PM2.5"]["unit"], "Unit not specified by source")
        self.assertIsNone(readings["NO2"]["value"])
        self.assertTrue(warnings)

    def test_city_search_keeps_only_exact_city_station_matches(self):
        payload = {
            "status": "ok",
            "data": [
                {"uid": 1, "station": {"name": "Delhi Station, Delhi, India"}},
                {"uid": 2, "station": {"name": "Sector 62, Noida, India"}},
                {"uid": 3, "station": {"name": "Indirapuram, Ghaziabad, India"}},
                {"uid": 2, "station": {"name": "Sector 62, Noida, India"}},
            ],
        }
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response(payload),
        ) as get:
            result = load_city_stations("Noida")

        self.assertTrue(result["api_available"])
        self.assertEqual(
            result["stations"],
            [{
                "station_id": "2",
                "station_name": "Sector 62, Noida, India",
                "geo": [],
            }],
        )
        self.assertEqual(get.call_args.args[0], SEARCH_URL)
        self.assertEqual(get.call_args.kwargs["params"]["keyword"], "Noida")
        self.assertEqual(get.call_args.kwargs["params"]["token"], "test-token")

    def test_city_aliases_match_gurgaon_and_gurugram(self):
        payload = {
            "status": "ok",
            "data": [
                {"uid": 1, "station": {"name": "Sector 51, Gurugram, India"}},
                {"uid": 2, "station": {"name": "Gurgaon Central, Gurgaon, India"}},
            ],
        }
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response(payload),
        ):
            result = load_city_stations("Gurgaon")
        self.assertEqual(
            [station["station_id"] for station in result["stations"]],
            ["2", "1"],
        )

    def test_locality_search_uses_waqi_query_and_keeps_only_exact_city_locality_matches(self):
        searches = {
            "Rohini": [
                {"uid": 11, "station": {"name": "Rohini, Delhi, India"}},
                {"uid": 11, "station": {"name": "Rohini, Delhi, India"}},
                {"uid": 12, "station": {"name": "Rohini, Gurugram, India"}},
                {"uid": 13, "station": {"name": "Delhi Station, Delhi, India"}},
            ],
            "Dwarka": [
                {"uid": 21, "station": {"name": "Dwarka Sector 10, Delhi, India"}},
            ],
            "Janakpuri": [
                {"uid": 31, "station": {"name": "Janakpuri, Delhi, India"}},
            ],
            "Anand Vihar": [
                {"uid": 41, "station": {"name": "Anand Vihar, Delhi, India"}},
            ],
        }

        def response_for_query(url, params, **kwargs):
            return _response({"status": "ok", "data": searches[params["keyword"]]})

        for locality, expected_id in (
            ("Rohini", "11"),
            ("Dwarka", "21"),
            ("Janakpuri", "31"),
            ("Anand Vihar", "41"),
        ):
            _fetch_json.clear()
            with self.subTest(locality=locality), self._mock_token(), patch(
                "src.waqi_data.requests.get",
                side_effect=response_for_query,
            ) as get:
                result = load_city_stations("Delhi", query=locality)

            self.assertEqual(get.call_args.kwargs["params"]["keyword"], locality)
            self.assertEqual(
                [station["station_id"] for station in result["stations"]],
                [expected_id],
            )
            self.assertEqual(result["station_count"], 1)

    def test_city_search_accepts_new_delhi_alias_and_reports_filter_counts(self):
        payload = {
            "status": "ok",
            "data": [
                {"uid": 1, "station": {"name": "R.K. Puram, New Delhi, India"}},
                {"uid": 2, "station": {"name": "Delhi Station, Delhi, India"}},
                {"uid": 3, "station": {"name": "Noida Station, Noida, India"}},
                {"uid": 2, "station": {"name": "Delhi Station, Delhi, India"}},
            ],
        }
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response(payload),
        ):
            result = load_city_stations("Delhi")

        self.assertEqual(result["raw_result_count"], 4)
        self.assertEqual(result["city_match_count"], 3)
        self.assertEqual(result["station_count"], 2)
        self.assertEqual(
            [station["station_id"] for station in result["stations"]],
            ["2", "1"],
        )

    def test_locality_without_exact_station_returns_no_invented_station(self):
        payload = {
            "status": "ok",
            "data": [
                {"uid": 1, "station": {"name": "Delhi Station, Delhi, India"}},
                {"uid": 2, "station": {"name": "Rohini, Gurugram, India"}},
            ],
        }
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response(payload),
        ):
            result = load_city_stations("Delhi", query="Janakpuri")

        self.assertTrue(result["api_available"])
        self.assertEqual(result["raw_result_count"], 2)
        self.assertEqual(result["stations"], [])
        self.assertEqual(result["station_count"], 0)
        self.assertIn("no matching stations", result["error"])

    def test_merging_city_and_locality_results_deduplicates_only_by_station_id(self):
        city_result = {
            "requested_location": "Delhi",
            "raw_result_count": 2,
            "stations": [
                {"station_id": "1", "station_name": "Same Station, Delhi, India", "geo": []},
                {"station_id": "2", "station_name": "Similar Name, Delhi, India", "geo": []},
            ],
        }
        locality_result = {
            "raw_result_count": 2,
            "stations": [
                {"station_id": "1", "station_name": "Same Station, Delhi, India", "geo": []},
                {"station_id": "3", "station_name": "Same Station, Delhi, India", "geo": []},
            ],
        }

        merged = merge_station_search_results(city_result, locality_result)

        self.assertEqual(
            [station["station_id"] for station in merged["stations"]],
            ["1", "2", "3"],
        )
        self.assertEqual(merged["station_count"], 3)
        self.assertEqual(merged["raw_result_count"], 4)

    def test_all_supported_city_names_require_an_exact_city_segment(self):
        station_names = {
            "Delhi": "Minto Road, Delhi, India",
            "Faridabad": "Sector 11, Faridabad, India",
            "Ghaziabad": "Sanjay Nagar, Ghaziabad, India",
            "Gurgaon": "Sector 51, Gurugram, India",
            "Noida": "Sector 62, Noida, India",
        }
        for index, (location, name) in enumerate(station_names.items(), start=1):
            with self.subTest(location=location):
                _fetch_json.clear()
                other_city = next(
                    other for other in station_names if other != location
                )
                payload = {
                    "status": "ok",
                    "data": [
                        {
                            "uid": index + 100,
                            "station": {
                                "name": f"Wrong City, {other_city}, India"
                            },
                        },
                        {"uid": index, "station": {"name": name}},
                    ],
                }
                with self._mock_token(), patch(
                    "src.waqi_data.requests.get",
                    return_value=_response(payload),
                ):
                    result = load_city_stations(location)
                self.assertEqual(
                    [station["station_name"] for station in result["stations"]],
                    [name],
                )

    def test_successful_station_feed_extracts_direct_aqi_and_nested_values(self):
        payload = _feed_payload(
            time_data={"iso": "2026-10-26T15:00:00+05:30"},
        )
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response(payload),
        ) as get:
            result = load_station_observation("Delhi", "2556", now=NOW)

        self.assertTrue(result["available"])
        self.assertEqual(result["aqi"], 160)
        self.assertEqual(result["aqi_source_value"], "160")
        self.assertEqual(result["aqi_display"], "160")
        self.assertEqual(result["station_id"], "2556")
        self.assertEqual(result["station_name"], "R.K. Puram, Delhi, Delhi, India")
        self.assertEqual(result["pollutants"]["PM2.5"]["value"], 18.7)
        self.assertEqual(
            result["observed_at"],
            datetime(2026, 10, 26, 9, 30, tzinfo=timezone.utc),
        )
        self.assertEqual(result["freshness"], "FRESH")
        self.assertIsNotNone(result["fetched_at"])
        self.assertFalse(result["cached"])
        self.assertIn("US EPA 2016", result["aqi_label"])
        self.assertEqual(
            get.call_args.args[0],
            "https://api.waqi.info/feed/@2556/",
        )

    def test_nearby_feed_verifies_station_distance_and_returns_live_observation(self):
        payload = _feed_payload(
            time_data={"iso": "2026-10-26T15:00:00+05:30"},
        )
        payload["data"]["city"]["geo"] = [28.6139, 77.2090]
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response(payload),
        ) as get:
            result = load_nearby_station_observation(
                28.6139, 77.2090, now=NOW
            )

        self.assertTrue(result["available"])
        self.assertEqual(result["station_id"], "2556")
        self.assertEqual(result["station_name"], "R.K. Puram, Delhi, Delhi, India")
        self.assertEqual(result["aqi"], 160)
        self.assertEqual(result["distance_km"], 0)
        self.assertEqual(result["freshness"], "FRESH")
        self.assertNotIn("latitude", result)
        self.assertNotIn("longitude", result)
        self.assertIn("/feed/geo:28.613900;77.209000/", get.call_args.args[0])

    def test_invalid_or_too_distant_location_never_selects_a_station(self):
        invalid = load_nearby_station_observation(float("nan"), 77.2)
        self.assertFalse(invalid["available"])
        self.assertIn("invalid coordinates", invalid["error"])

        payload = _feed_payload()
        payload["data"]["city"]["geo"] = [29.0, 77.2]
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response(payload),
        ):
            distant = load_nearby_station_observation(28.6, 77.2, now=NOW)
        self.assertFalse(distant["available"])
        self.assertIsNone(distant["aqi"])
        self.assertIn("could not be verified within", distant["error"])

    def test_nearby_missing_aqi_or_stale_timestamp_is_not_current(self):
        missing_aqi = _feed_payload(
            aqi="-",
            time_data={"iso": "2026-10-26T15:00:00+05:30"},
        )
        missing_aqi["data"]["city"]["geo"] = [28.6, 77.2]
        stale = _feed_payload(
            time_data={"iso": "2026-10-25T10:00:00+05:30"},
        )
        stale["data"]["city"]["geo"] = [28.6, 77.2]

        for payload, expected in (
            (missing_aqi, None),
            (stale, "STALE"),
        ):
            with self.subTest(expected=expected), self._mock_token(), patch(
                "src.waqi_data.requests.get",
                return_value=_response(payload),
            ):
                result = load_nearby_station_observation(
                    28.6, 77.2, now=NOW
                )
            self.assertTrue(result["available"])
            self.assertEqual(result["aqi"], 160 if expected else None)
            self.assertEqual(result["freshness"], expected or "FRESH")
            if expected is None:
                self.assertTrue(any("valid station AQI" in warning for warning in result["warnings"]))

    def test_missing_aqi_is_not_calculated_or_replaced(self):
        payload = _feed_payload(
            aqi="-",
            time_data={"iso": "2026-10-26T15:00:00+05:30"},
        )
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response(payload),
        ):
            result = load_station_observation("Delhi", "2556", now=NOW)

        self.assertTrue(result["available"])
        self.assertIsNone(result["aqi"])
        self.assertEqual(result["aqi_display"], AQI_UNAVAILABLE)
        self.assertEqual(result["pollutants"]["PM2.5"]["value"], 18.7)
        self.assertTrue(any("valid station AQI" in warning for warning in result["warnings"]))

    def test_missing_timestamp_preserves_reported_aqi_and_marks_time_unavailable(self):
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response(_feed_payload()),
        ):
            result = load_station_observation("Delhi", "2556", now=NOW)

        self.assertTrue(result["available"])
        self.assertEqual(result["aqi"], 160)
        self.assertIsNone(result["observed_at"])
        self.assertIsNone(result["freshness"])
        self.assertTrue(any("timestamp" in warning for warning in result["warnings"]))

    def test_malformed_search_and_feed_structures_fail_without_values(self):
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response({"status": "ok", "data": {"unexpected": []}}),
        ):
            search = load_city_stations("Delhi")
        self.assertTrue(search["api_available"])
        self.assertFalse(search["stations"])
        self.assertIn("unexpected response", search["error"])

        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response({"status": "ok", "data": []}),
        ):
            feed = load_station_observation("Delhi", "2556", now=NOW)
        self.assertFalse(feed["available"])
        self.assertIsNone(feed["aqi"])
        self.assertIn("unexpected response", feed["error"])

    def test_location_mismatch_does_not_return_another_city_aqi(self):
        payload = _feed_payload(
            name="R.K. Puram, Delhi, Delhi, India",
            time_data={"iso": "2026-10-26T15:00:00+05:30"},
        )
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response(payload),
        ):
            result = load_station_observation("Noida", "2556", now=NOW)

        self.assertFalse(result["available"])
        self.assertIsNone(result["aqi"])
        self.assertIn("did not match Noida", result["error"])
        self.assertIn("R.K. Puram", result["station_name"])

    def test_missing_secret_and_invalid_location_are_unavailable(self):
        with patch("src.waqi_data._load_token", return_value=None):
            result = load_city_stations("Delhi")
        self.assertFalse(result["api_available"])
        self.assertIn("not configured", result["error"])

        result = load_city_stations("Unknown")
        self.assertFalse(result["api_available"])
        self.assertIn("not configured", result["error"])

    def test_timeout_http_invalid_token_and_rate_limit_are_safe_failures(self):
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            side_effect=requests.Timeout,
        ):
            timed_out = load_city_stations("Delhi")
        self.assertFalse(timed_out["api_available"])
        self.assertIn("timed out", timed_out["error"])

        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response({"status": "error", "data": "Invalid key secret"}, 200),
        ):
            invalid_token = load_city_stations("Delhi")
        self.assertIn("rejected", invalid_token["error"])
        self.assertNotIn("secret", invalid_token["error"])

        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response({"status": "error", "data": "Over quota"}, 200),
        ):
            rate_limited = load_city_stations("Delhi")
        self.assertIn("quota or rate limit", rate_limited["error"])

        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response({}, 503, ok=False),
        ):
            http_error = load_city_stations("Delhi")
        self.assertIn("HTTP 503", http_error["error"])

    def test_malformed_json_and_network_errors_are_unavailable(self):
        bad_json = _response({})
        bad_json.json.side_effect = ValueError("malformed")
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=bad_json,
        ):
            malformed = load_city_stations("Delhi")
        self.assertIn("not valid JSON", malformed["error"])

        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            side_effect=requests.ConnectionError,
        ):
            unavailable = load_city_stations("Delhi")
        self.assertIn("unavailable", unavailable["error"])

    def test_cache_is_location_specific_and_only_successes_are_cached(self):
        payload = {
            "status": "ok",
            "data": [{
                "uid": 2,
                "station": {"name": "Sector 62, Noida, India"},
            }],
        }
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            return_value=_response(payload),
        ) as get:
            load_city_stations("Delhi")
            load_city_stations("Delhi")
            load_city_stations("Noida")
        self.assertEqual(get.call_count, 2)
        self.assertEqual(CACHE_TTL_SECONDS, 120)

        _fetch_json.clear()
        with self._mock_token(), patch(
            "src.waqi_data.requests.get",
            side_effect=[
                requests.Timeout(),
                _response(payload),
            ],
        ) as get:
            first = load_city_stations("Delhi")
            second = load_city_stations("Delhi")
        self.assertFalse(first["api_available"])
        self.assertTrue(second["api_available"])
        self.assertEqual(get.call_count, 2)


if __name__ == "__main__":
    unittest.main()
