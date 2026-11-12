from datetime import datetime, timedelta, timezone
import unittest

from src.aqi_alerts import (
    HIGH_AQI_PRECAUTION,
    HIGH_AQI_SEVERITY,
    HIGH_AQI_THRESHOLD,
    build_observed_aqi_alert,
    cloud_alert_component_html,
    dismiss_alert_session_state,
    update_alert_session_state,
)
from src.theme import theme_css


OBSERVED_AT = datetime(2026, 10, 26, 10, tzinfo=timezone.utc)


def _observation(**overrides):
    result = {
        "available": True,
        "aqi": 100,
        "freshness": "FRESH",
        "station_id": "2556",
        "station_name": "R.K. Puram, Delhi, India",
        "observed_at": OBSERVED_AT,
    }
    result.update(overrides)
    return result


class ObservedAqiAlertTests(unittest.TestCase):
    def test_alert_starts_at_documented_us_epa_sensitive_group_threshold(self):
        self.assertEqual(HIGH_AQI_THRESHOLD, 101)
        self.assertIsNone(build_observed_aqi_alert(_observation(aqi=100)))

        alert = build_observed_aqi_alert(_observation(aqi=101))
        self.assertEqual(alert["severity"], HIGH_AQI_SEVERITY)
        self.assertEqual(alert["aqi"], 101)
        self.assertIn("Sensitive groups", alert["precaution"])
        self.assertEqual(alert["precaution"], HIGH_AQI_PRECAUTION)
        self.assertEqual(
            build_observed_aqi_alert(_observation(aqi=151))["severity"],
            "Unhealthy",
        )
        self.assertEqual(
            build_observed_aqi_alert(_observation(aqi=201))["severity"],
            "Very Unhealthy",
        )
        self.assertEqual(
            build_observed_aqi_alert(_observation(aqi=301))["severity"],
            "Hazardous",
        )

    def test_missing_stale_invalid_and_failed_observations_do_not_alert(self):
        for changes in (
            {"available": False},
            {"aqi": None},
            {"freshness": "STALE", "aqi": 250},
            {"freshness": "DELAYED", "aqi": 250},
            {"observed_at": None},
        ):
            with self.subTest(changes=changes):
                observation = _observation(aqi=151)
                observation.update(changes)
                self.assertIsNone(build_observed_aqi_alert(observation))

    def test_station_switch_is_a_new_event_and_same_observation_is_suppressed(self):
        first = build_observed_aqi_alert(_observation(aqi=151))
        state = {}
        active, is_new = update_alert_session_state(state, first)
        self.assertEqual(active, first)
        self.assertTrue(is_new)

        active, is_new = update_alert_session_state(
            state, build_observed_aqi_alert(_observation(aqi=151))
        )
        self.assertFalse(is_new)
        self.assertEqual(active, first)

        second = build_observed_aqi_alert(_observation(
            aqi=151,
            station_id="999",
            station_name="Sector 62, Noida, India",
        ))
        active, is_new = update_alert_session_state(state, second)
        self.assertTrue(is_new)
        self.assertEqual(active["station_id"], "999")

    def test_new_observation_and_changed_risk_level_start_a_new_event(self):
        state = {}
        first = build_observed_aqi_alert(_observation(aqi=151))
        update_alert_session_state(state, first)

        newer = build_observed_aqi_alert(_observation(
            aqi=205, observed_at=OBSERVED_AT + timedelta(hours=1)
        ))
        active, is_new = update_alert_session_state(state, newer)
        self.assertTrue(is_new)
        self.assertEqual(active["aqi"], 205)
        self.assertEqual(active["severity"], "Very Unhealthy")

    def test_dismissal_persists_until_event_changes_or_is_no_longer_relevant(self):
        state = {}
        alert = build_observed_aqi_alert(_observation(aqi=151))
        update_alert_session_state(state, alert)
        dismiss_alert_session_state(state)

        active, is_new = update_alert_session_state(state, alert)
        self.assertIsNone(active)
        self.assertFalse(is_new)

        changed = build_observed_aqi_alert(_observation(
            aqi=151, observed_at=OBSERVED_AT + timedelta(minutes=15)
        ))
        active, is_new = update_alert_session_state(state, changed)
        self.assertTrue(is_new)
        self.assertIsNotNone(active)

        active, _ = update_alert_session_state(state, None)
        self.assertIsNone(active)
        self.assertIsNone(state["saanscheck_aqi_alert_event_key"])

    def test_cloud_timer_theme_and_reduced_motion_configuration(self):
        alert = build_observed_aqi_alert(_observation(aqi=151))
        component = cloud_alert_component_html(alert)
        self.assertIn("setTimeout", component)
        self.assertIn("}, 5000)", component)
        self.assertIn("High AQI Alert", component)
        self.assertIn("sc-waqi-cloud-motion 5s", component)

        for theme in ("Dark", "Light"):
            css = theme_css(theme)
            self.assertIn(".st-key-waqi_alert_dock", css)
            self.assertIn("prefers-reduced-motion: reduce", css)
            self.assertIn("@media (max-width: 640px)", css)


if __name__ == "__main__":
    unittest.main()
