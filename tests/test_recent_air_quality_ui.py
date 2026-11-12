from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _station_search(location: str) -> dict:
    records = {
        "Delhi": [
            {"station_id": "101", "station_name": "R.K. Puram, Delhi, India", "geo": []},
            {"station_id": "102", "station_name": "Pusa, Delhi, India", "geo": []},
        ],
        "Noida": [
            {"station_id": "201", "station_name": "Sector 62, Noida, India", "geo": []},
        ],
    }
    return {
        "requested_location": location,
        "api_available": True,
        "status_code": 200,
        "stations": records.get(location, []),
        "error": None if records.get(location) else f"No WAQI stations for {location}.",
        "source": "World Air Quality Index (WAQI)",
    }


def _observation(location: str, station_id: str) -> dict:
    retrieved_at = datetime.now(timezone.utc)
    return {
        "requested_location": location,
        "api_available": True,
        "status_code": 200,
        "available": True,
        "error": None,
        "source": "World Air Quality Index (WAQI)",
        "aqi_label": "WAQI-reported AQI · US EPA 2016 scale",
        "aqi_scale_url": "https://aqicn.org/scale/",
        "aqi": 87,
        "aqi_display": "87",
        "station_name": (
            "R.K. Puram, Delhi, India"
            if location == "Delhi"
            else "Sector 62, Noida, India"
        ),
        "station_id": station_id,
        "attributions": [],
        "observed_at": retrieved_at - timedelta(minutes=5),
        "fetched_at": retrieved_at,
        "cached": False,
        "age_minutes": 5.0,
        "freshness": "FRESH",
        "pollutants": {
            name: {"value": None, "unit": "Unit not specified by source"}
            for name in ("PM2.5", "PM10", "NO2", "SO2", "CO", "O3")
        },
        "warnings": [],
    }


class RecentAirQualityUiTests(unittest.TestCase):
    def test_manual_locality_search_merges_real_city_and_query_station_ids(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["workspace_mode"] = "Recent Air Quality"
        app.session_state["active_page"] = "Recent Overview"
        app.session_state["recent_location_mode"] = "Manual city/station"
        app.session_state["recent_station_query"] = "Rohini"
        app.session_state["recent_station_query_applied"] = "Rohini"

        def search(location, query=None):
            if not query:
                return _station_search(location)
            return {
                "requested_location": location,
                "query": query,
                "api_available": True,
                "status_code": 200,
                "raw_result_count": 2,
                "city_match_count": 2,
                "station_count": 1,
                "stations": [
                    {
                        "station_id": "103",
                        "station_name": "Rohini Sector 16, Delhi, India",
                        "geo": [28.74, 77.11],
                    }
                ],
                "error": None,
                "source": "World Air Quality Index (WAQI)",
            }

        with (
            patch("src.waqi_data.load_city_stations", side_effect=search),
            patch(
                "src.waqi_data.load_station_observation",
                side_effect=_observation,
            ),
            patch(
                "src.cpcb_national_aqi.load_cpcb_national_aqi",
                return_value={"available": False, "error": "CPCB unavailable."},
            ),
        ):
            app.run()

        self.assertFalse(app.exception)
        visible = "\n".join(
            str(element.value)
            for element in (*app.markdown, *app.caption)
        )
        self.assertIn("City:", visible)
        self.assertEqual(
            app.text_input(key="recent_station_query").label,
            "Search locality or station",
        )
        self.assertIn("2 records", visible)
        self.assertIn("unique station IDs", visible)
        self.assertEqual(len(app.selectbox(key="recent_station_uid").options), 3)
        self.assertEqual(app.session_state["recent_station_query_applied"], "Rohini")

    def test_missing_locality_match_is_reported_without_creating_a_station(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["workspace_mode"] = "Recent Air Quality"
        app.session_state["active_page"] = "Recent Overview"
        app.session_state["recent_location_mode"] = "Manual city/station"
        app.session_state["recent_station_query"] = "No Such Locality"
        app.session_state["recent_station_query_applied"] = "No Such Locality"

        def search(location, query=None):
            if not query:
                return _station_search(location)
            return {
                "requested_location": location,
                "query": query,
                "api_available": True,
                "status_code": 200,
                "raw_result_count": 1,
                "city_match_count": 1,
                "station_count": 0,
                "stations": [],
                "error": f"WAQI returned no matching stations for {query} in {location}.",
                "source": "World Air Quality Index (WAQI)",
            }

        with (
            patch("src.waqi_data.load_city_stations", side_effect=search),
            patch(
                "src.cpcb_national_aqi.load_cpcb_national_aqi",
                return_value={"available": False, "error": "CPCB unavailable."},
            ),
        ):
            app.run()

        self.assertFalse(app.exception)
        visible = "\n".join(
            str(element.value)
            for element in (*app.markdown, *app.info, *app.caption)
        )
        self.assertIn("No exact monitoring station found for this locality.", visible)
        self.assertIn("Other available stations in the selected city", visible)
        self.assertEqual(len(app.selectbox(key="recent_station_uid").options), 2)

    def test_risk_scale_shows_fresh_waqi_marker_only_on_us_scale(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["workspace_mode"] = "Recent Air Quality"
        app.session_state["active_page"] = "Recent Overview"
        app.session_state["recent_location_mode"] = "Manual city/station"

        with (
            patch("src.waqi_data.load_city_stations", side_effect=_station_search),
            patch(
                "src.waqi_data.load_station_observation",
                side_effect=_observation,
            ),
            patch(
                "src.cpcb_national_aqi.load_cpcb_national_aqi",
                return_value={"available": False, "error": "CPCB unavailable."},
            ),
        ):
            app.run()
            app.selectbox(key="recent_station_uid").set_value("101").run()

            visible = "\n".join(str(element.value) for element in app.markdown)
            self.assertIn("Air Quality Index", visible)
            self.assertIn("Legend:", visible)
            self.assertIn("WAQI", visible)
            self.assertIn("US AQI", visible)
            self.assertIn('<span class="sc-risk-marker"', visible)
            self.assertIn("Current: Moderate", visible)
            self.assertIn("AQI 87", visible)

            app.radio(key="recent_aqi_risk_scale").set_value(
                "Indian National AQI — CPCB"
            ).run()
            visible = "\n".join(str(element.value) for element in app.markdown)

        self.assertFalse(app.exception)
        self.assertIn("Indian National AQI", visible)
        self.assertIn("Legend:", visible)
        self.assertNotIn('<span class="sc-risk-marker"', visible)
        self.assertIn("CPCB National AQI unavailable", visible)

    def test_risk_scale_suppresses_stale_waqi_marker(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["workspace_mode"] = "Recent Air Quality"
        app.session_state["active_page"] = "Recent Overview"
        app.session_state["recent_location_mode"] = "Manual city/station"
        stale = _observation("Delhi", "101")
        stale["freshness"] = "STALE"
        stale["observed_at"] = datetime.now(timezone.utc) - timedelta(hours=12)

        with (
            patch("src.waqi_data.load_city_stations", side_effect=_station_search),
            patch("src.waqi_data.load_station_observation", return_value=stale),
            patch(
                "src.cpcb_national_aqi.load_cpcb_national_aqi",
                return_value={"available": False, "error": "CPCB unavailable."},
            ),
        ):
            app.run()
            app.selectbox(key="recent_station_uid").set_value("101").run()
            app.radio(key="recent_aqi_risk_scale").set_value(
                "WAQI — US AQI"
            ).run()

        visible = "\n".join(str(element.value) for element in app.markdown)
        self.assertFalse(app.exception)
        self.assertNotIn('<span class="sc-risk-marker"', visible)
        self.assertIn("WAQI observation is stale", visible)

    def test_cpcb_is_primary_unavailable_state_and_waqi_remains_labeled_alternative(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["workspace_mode"] = "Recent Air Quality"
        app.session_state["active_page"] = "Recent Overview"
        app.session_state["recent_location_mode"] = "Manual city/station"

        with (
            patch("src.waqi_data.load_city_stations", side_effect=_station_search),
            patch(
                "src.waqi_data.load_station_observation",
                side_effect=_observation,
            ),
            patch(
                "src.cpcb_national_aqi.load_cpcb_national_aqi",
                return_value={
                    "available": False,
                    "error": "CPCB test source unavailable.",
                },
            ),
        ):
            app.run()
            app.selectbox(key="recent_station_uid").set_value("101").run()

        self.assertFalse(app.exception)
        visible = "\n".join(str(element.value) for element in app.markdown)
        self.assertIn("Indian National AQI — CPCB", visible)
        self.assertIn("CPCB National AQI unavailable", visible)
        self.assertIn("https://airquality.cpcb.gov.in/AQI/", visible)
        self.assertIn("WAQI — US AQI (alternative source)", visible)
        self.assertIn("87", visible)

    def test_verified_cpcb_observation_is_separate_from_waqi_scale_and_alert(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["workspace_mode"] = "Recent Air Quality"
        app.session_state["active_page"] = "Recent Overview"
        app.session_state["recent_location_mode"] = "Manual city/station"
        cpcb_reading = {
            "available": True,
            "aqi": 180.0,
            "aqi_display": "180",
            "category": "Moderate",
            "station": "R.K. Puram, Delhi, India",
            "location": "Delhi",
            "observed_at": datetime.now(timezone.utc) - timedelta(minutes=5),
            "freshness": "FRESH",
            "source": "data.gov.in · CPCB-sourced resource",
            "scale": "Indian National AQI",
            "error": None,
        }

        with (
            patch("src.waqi_data.load_city_stations", side_effect=_station_search),
            patch(
                "src.waqi_data.load_station_observation",
                side_effect=_observation,
            ),
            patch(
                "src.cpcb_national_aqi.load_cpcb_national_aqi",
                return_value=cpcb_reading,
            ),
        ):
            app.run()
            app.selectbox(key="recent_station_uid").set_value("101").run()
            visible = "\n".join(str(element.value) for element in app.markdown)
            self.assertIn("Current: Moderate", visible)
            self.assertIn("AQI 180", visible)
            self.assertIn('<span class="sc-risk-marker"', visible)

            app.radio(key="recent_aqi_risk_scale").set_value(
                "WAQI — US AQI"
            ).run()
            visible = "\n".join(str(element.value) for element in app.markdown)

        self.assertFalse(app.exception)
        warnings = "\n".join(str(element.value) for element in app.warning)
        self.assertNotIn('<span class="sc-risk-marker"', visible)
        self.assertIn("Fresh Indian National AQI", visible)
        self.assertIn("CPCB category: Moderate", visible)
        self.assertIn("CPCB National AQI Alert", warnings)
        self.assertIn("CPCB National AQI health guidance", visible)
        self.assertIn("WAQI — US AQI health guidance", visible)
        self.assertIn("AQI: 180 · Moderate", visible)
        self.assertIn("WAQI — US AQI (alternative source)", visible)

    def test_comparison_shows_waqi_and_unconnected_aqi_in_for_selected_station(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["workspace_mode"] = "Recent Air Quality"
        app.session_state["active_page"] = "Recent Overview"
        app.session_state["recent_location_mode"] = "Manual city/station"

        with (
            patch("src.waqi_data.load_city_stations", side_effect=_station_search),
            patch(
                "src.waqi_data.load_station_observation",
                side_effect=_observation,
            ),
        ):
            app.run()
            app.selectbox(key="recent_station_uid").set_value("101").run()

        self.assertFalse(app.exception)
        comparison = app.expander[0]
        visible = "\n".join(
            str(element.value)
            for element in (*comparison.markdown, *comparison.caption)
        )
        self.assertEqual(comparison.label, "Compare Air Quality Sources")
        self.assertIn("WAQI — US AQI", visible)
        self.assertIn("AQI: 87", visible)
        self.assertIn("AQI.in — AQI-US", visible)
        self.assertIn("AQI.in live data not connected", visible)
        self.assertIn("R.K. Puram, Delhi, India", visible)
        self.assertIn("Open AQI.in to find this station", visible)
        self.assertIn(
            "AQI values can differ because providers may use different "
            "measurements, aggregation methods, reporting times or station mappings.",
            "\n".join(str(element.value) for element in comparison.markdown),
        )
        self.assertNotIn("302", visible)
        self.assertNotIn("anand-vihar", visible)

    def test_aqi_in_link_is_not_kept_on_another_selected_station(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["workspace_mode"] = "Recent Air Quality"
        app.session_state["active_page"] = "Recent Overview"
        app.session_state["recent_location_mode"] = "Manual city/station"

        with (
            patch("src.waqi_data.load_city_stations", side_effect=_station_search),
            patch(
                "src.waqi_data.load_station_observation",
                side_effect=_observation,
            ),
        ):
            app.run()
            app.selectbox(key="recent_station_uid").set_value("101").run()
            app.selectbox(key="navbar_city_search").set_value("Noida").run()

        self.assertFalse(app.exception)
        visible = "\n".join(
            str(element.value)
            for element in (
                *app.expander[0].markdown,
                *app.expander[0].caption,
            )
        )
        self.assertIn("Sector 62, Noida, India", visible)
        self.assertIn("https://www.aqi.in/", visible)
        self.assertNotIn("anand-vihar", visible)
        self.assertNotIn("Anand Vihar", visible)

    def test_fresh_visit_defaults_to_recent_overview_and_requests_browser_location(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        with (
            patch("src.browser_location.request_browser_location", return_value=None) as browser_location,
            patch("src.waqi_data.load_city_stations", side_effect=_station_search),
        ):
            app.run()

        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["workspace_mode"], "Recent Air Quality")
        self.assertEqual(app.session_state["active_page"], "Recent Overview")
        self.assertEqual(app.session_state["recent_location_mode"], "Automatic nearby")
        self.assertTrue(browser_location.call_args.kwargs["enabled"])

    def test_granted_location_loads_verified_nearby_observation(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        nearby = _observation("Nearby area", "2556")
        nearby["distance_km"] = 3.4
        nearby["station_name"] = "R.K. Puram, Delhi, Delhi, India"
        lookup = {
            "status": "granted",
            "latitude": 28.6,
            "longitude": 77.2,
        }
        with (
            patch(
                "src.browser_location.request_browser_location",
                return_value=lookup,
            ),
            patch("src.waqi_data.load_city_stations", side_effect=_station_search),
            patch(
                "src.waqi_data.load_nearby_station_observation",
                return_value=nearby,
            ) as nearby_lookup,
        ):
            app.run()

        self.assertFalse(app.exception)
        nearby_lookup.assert_called_once_with(28.6, 77.2)
        self.assertEqual(app.session_state["browser_location_status"], "granted")
        self.assertNotIn("latitude", app.session_state["nearby_station_observation"])
        self.assertIn(
            "R.K. Puram, Delhi, Delhi, India",
            "\n".join(str(element.value) for element in app.markdown),
        )
        captions = "\n".join(str(element.value) for element in app.caption)
        self.assertIn("3.4 km away", captions)
        self.assertIn("Moderate", "\n".join(str(element.value) for element in app.markdown))

    def test_denied_location_keeps_manual_city_and_station_selection_available(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        feed_requests: list[tuple[str, str]] = []

        def load_observation(location: str, station_id: str) -> dict:
            feed_requests.append((location, station_id))
            return _observation(location, station_id)

        with (
            patch(
                "src.browser_location.request_browser_location",
                return_value={"status": "denied"},
            ) as browser_location,
            patch("src.waqi_data.load_city_stations", side_effect=_station_search),
            patch(
                "src.waqi_data.load_station_observation",
                side_effect=load_observation,
            ),
        ):
            app.run()
            self.assertFalse(app.exception)
            self.assertIn(
                "permission was denied",
                "\n".join(str(element.value) for element in app.info).lower()
                + "\n"
                + "\n".join(str(element.value) for element in app.warning).lower(),
            )

            app.radio(key="recent_location_mode").set_value(
                "Manual city/station"
            ).run()
            self.assertFalse(app.exception)
            app.selectbox(key="recent_station_uid").set_value("101").run()
            self.assertFalse(app.exception)

        self.assertEqual(feed_requests, [("Delhi", "101")])
        self.assertFalse(browser_location.call_args_list[-1].kwargs["enabled"])
        self.assertIn("R.K. Puram, Delhi, India", "\n".join(
            str(element.value) for element in app.markdown
        ))

    def test_location_change_uses_matching_station_and_no_historical_fallback(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["workspace_mode"] = "Recent Air Quality"
        app.session_state["active_page"] = "Recent Overview"
        app.session_state["recent_location_mode"] = "Manual city/station"
        feed_requests: list[tuple[str, str]] = []
        loaded_observations: list[dict] = []

        def load_observation(location: str, station_id: str) -> dict:
            feed_requests.append((location, station_id))
            result = _observation(location, station_id)
            loaded_observations.append(result)
            return result

        with (
            patch(
                "src.waqi_data.load_city_stations",
                side_effect=_station_search,
            ),
            patch(
                "src.waqi_data.load_station_observation",
                side_effect=load_observation,
            ),
        ):
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(feed_requests, [])
            station_picker = app.selectbox(key="recent_station_uid")
            self.assertEqual(
                station_picker.options,
                ["R.K. Puram, Delhi, India", "Pusa, Delhi, India"],
            )
            self.assertIsNone(station_picker.value)
            self.assertIn(
                "Choose one of the 2 matching WAQI stations",
                "\n".join(str(element.value) for element in app.markdown),
            )

            app.selectbox(key="recent_station_uid").set_value("101").run()
            self.assertFalse(app.exception)
            self.assertEqual(feed_requests[-1], ("Delhi", "101"))
            visible = "\n".join(str(element.value) for element in app.markdown)
            self.assertIn("R.K. Puram, Delhi, India", visible)
            self.assertIn("87", visible)
            expected_observation_time = loaded_observations[-1]["observed_at"].astimezone(
                ZoneInfo("Asia/Kolkata")
            ).isoformat(sep=" ", timespec="minutes")
            self.assertIn(expected_observation_time, visible)

            app.selectbox(key="navbar_city_search").set_value("Noida").run()
            self.assertFalse(app.exception)
            self.assertEqual(feed_requests[-1], ("Noida", "201"))
            self.assertEqual(app.session_state["recent_station_uid"], "201")
            self.assertIn(
                "Sector 62, Noida, India",
                "\n".join(str(element.value) for element in app.markdown),
            )

        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["workspace_mode"] = "Recent Air Quality"
        app.session_state["active_page"] = "Recent Overview"
        app.session_state["recent_location_mode"] = "Manual city/station"
        with patch(
            "src.waqi_data.load_city_stations",
            return_value={
                "requested_location": "Delhi",
                "api_available": False,
                "status_code": None,
                "stations": [],
                "error": "WAQI request timed out.",
                "source": "World Air Quality Index (WAQI)",
            },
        ), patch("src.waqi_data.load_station_observation") as load_feed:
            app.run()

        self.assertFalse(app.exception)
        self.assertFalse(load_feed.called)
        visible = "\n".join(str(element.value) for element in app.markdown)
        self.assertIn("Fresh Delhi data unavailable", visible)
        self.assertIn("WAQI request timed out.", visible)
        self.assertNotIn("Highest observed historical AQI", visible)
        self.assertNotIn("AQI source", visible)

    def test_stale_cached_observation_is_explicitly_labeled(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["workspace_mode"] = "Recent Air Quality"
        app.session_state["active_page"] = "Recent Overview"
        stale = _observation("Delhi", "101")
        stale["freshness"] = "STALE"
        stale["age_minutes"] = 12 * 60
        stale["observed_at"] = datetime.now(timezone.utc) - timedelta(hours=12)
        stale["cached"] = True
        stale["fetched_at"] = datetime(2026, 10, 26, 9, 30, tzinfo=timezone.utc)

        with patch(
            "src.waqi_data.load_city_stations",
            side_effect=_station_search,
        ), patch(
            "src.waqi_data.load_station_observation",
            return_value=stale,
        ):
            app.run()
            app.selectbox(key="recent_station_uid").set_value("101").run()

        self.assertFalse(app.exception)
        visible = "\n".join(str(element.value) for element in app.markdown)
        info = "\n".join(str(element.value) for element in app.info)
        warnings = "\n".join(str(element.value) for element in app.warning)
        self.assertIn("CACHED · STALE", visible)
        self.assertIn("short-term cache", info)
        self.assertIn("Current conditions are not confirmed", warnings)

    def test_recent_forecast_keeps_predictions_unavailable_and_alert_is_dismissible(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["workspace_mode"] = "Recent Air Quality"
        app.session_state["active_page"] = "Forecast"
        app.session_state["recent_location_mode"] = "Manual city/station"
        high = _observation("Delhi", "101")
        high["aqi"] = 151

        with patch(
            "src.waqi_data.load_city_stations",
            side_effect=_station_search,
        ), patch(
            "src.waqi_data.load_station_observation",
            return_value=high,
        ), patch("src.forecast_features.build_direct_forecasts") as historical_model:
            app.run()
            app.selectbox(key="recent_station_uid").set_value("101").run()

            self.assertFalse(app.exception)
            visible = "\n".join(str(element.value) for element in app.markdown)
            self.assertIn("Station Forecast", visible)
            self.assertIn("observed AQI", visible)
            self.assertIn("estimated AQI", visible)
            self.assertIn("predicted AQI", visible)
            self.assertIn("Forecast unavailable", visible)
            self.assertIn("High AQI Alert", visible)
            self.assertTrue(
                any(button.key == "dismiss_waqi_aqi_alert" for button in app.button)
            )
            self.assertFalse(historical_model.called)
            event_key = app.session_state["saanscheck_aqi_alert_event_key"]

            app.segmented_control(key="appearance_theme").set_value("Light").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["appearance_theme"], "Light")
            self.assertEqual(
                app.session_state["saanscheck_aqi_alert_event_key"],
                event_key,
            )
            next(
                button for button in app.button
                if button.key == "dismiss_waqi_aqi_alert"
            ).click().run()
            self.assertFalse(app.exception)
            self.assertEqual(
                app.session_state["saanscheck_aqi_alert_dismissed_key"],
                event_key,
            )
            self.assertFalse(
                any(button.key == "dismiss_waqi_aqi_alert" for button in app.button)
            )

    def test_station_change_updates_alert_event_and_stale_station_does_not_alert(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=60)
        app.session_state["workspace_mode"] = "Recent Air Quality"
        app.session_state["active_page"] = "Recent Overview"
        app.session_state["recent_location_mode"] = "Manual city/station"

        def load_observation(location: str, station_id: str) -> dict:
            result = _observation(location, station_id)
            result["aqi"] = 175
            if location == "Noida":
                result["freshness"] = "STALE"
                result["observed_at"] = datetime.now(timezone.utc) - timedelta(hours=12)
            return result

        with patch(
            "src.waqi_data.load_city_stations",
            side_effect=_station_search,
        ), patch(
            "src.waqi_data.load_station_observation",
            side_effect=load_observation,
        ):
            app.run()
            app.selectbox(key="recent_station_uid").set_value("101").run()
            first_event = app.session_state["saanscheck_aqi_alert_event_key"]
            self.assertIsNotNone(first_event)

            app.selectbox(key="navbar_city_search").set_value("Noida").run()
            self.assertEqual(app.session_state["recent_station_uid"], "201")
            self.assertIsNone(app.session_state["saanscheck_aqi_alert_event_key"])
            self.assertFalse(
                any(button.key == "dismiss_waqi_aqi_alert" for button in app.button)
            )


if __name__ == "__main__":
    unittest.main()
