from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _run_page(page: str, station: str = "Delhi") -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=45)
    app.session_state["workspace_mode"] = "Historical Intelligence"
    app.session_state["active_page"] = page
    app.run()
    app.selectbox(key="historical_station").set_value(station).run()
    return app


def _markdown_text(app: AppTest) -> str:
    return "\n".join(str(element.value) for element in app.markdown)


def _caption_text(app: AppTest) -> str:
    return "\n".join(str(element.value) for element in app.caption)


class DashboardCopyTests(unittest.TestCase):
    def test_overview_does_not_assert_historical_pollutant_units(self):
        app = _run_page("Overview")
        self.assertFalse(app.exception)
        self.assertIn("Historical source units are not verified", _caption_text(app))
        self.assertIn("Unverified", "\n".join(str(item.value) for item in app.dataframe))

    def test_forecast_discloses_historical_origin_and_target_dates(self):
        app = _run_page("AQI Forecast")
        self.assertFalse(app.exception)
        warning_text = "\n".join(str(item.value) for item in app.warning)
        self.assertIn("not a current forecast", warning_text)
        self.assertIn("December 31, 2023", warning_text)
        self.assertIn("January 01–January 03, 2024", warning_text)
        self.assertIn("dataset-defined", _caption_text(app))
        self.assertIn("Forecast origin: 2023-12-31", _markdown_text(app))

    def test_health_advisor_dates_and_qualifies_historical_guidance(self):
        app = _run_page("Health Advisor")
        self.assertFalse(app.exception)
        markdown = _markdown_text(app)
        self.assertIn("Historical AQI (dataset-defined)", markdown)
        self.assertIn("Historical date:", markdown)
        self.assertIn("2023-12-31", markdown)
        warning_text = "\n".join(str(item.value) for item in app.warning)
        self.assertIn("not a live assessment", warning_text)
        self.assertIn("does not provide medical diagnosis or treatment", warning_text)

    def test_pollution_analytics_does_not_assert_unverified_units(self):
        app = _run_page("Pollution Analytics")
        self.assertFalse(app.exception)
        markdown = _markdown_text(app)
        self.assertIn("Historical source unit unverified", markdown)
        self.assertIn("unit unverified", markdown)
        self.assertIn("not directly comparable", _caption_text(app))

    def test_model_insights_identifies_selected_station_and_chart_dates(self):
        app = _run_page("Model Insights", station="Delhi")
        self.assertFalse(app.exception)
        captions = "\n".join(str(item.value) for item in app.caption)
        self.assertIn("Delhi", captions)
        self.assertIn("2023-11-", captions)
        self.assertIn("to 2023-12-30", captions)

    def test_model_insights_reports_empty_station_predictions(self):
        empty_predictions = pd.DataFrame(columns=[
            "date",
            "location_name",
            "actual_AQI_t+1",
            "pred_rf_AQI_t+1",
            "actual_AQI_t+2",
            "pred_rf_AQI_t+2",
            "actual_AQI_t+3",
            "pred_rf_AQI_t+3",
        ])
        with patch(
            "src.data_loader.load_test_predictions",
            return_value=empty_predictions,
        ):
            app = _run_page("Model Insights", station="Delhi")
        self.assertFalse(app.exception)
        self.assertIn(
            "No actual and predicted values are available for Delhi.",
            "\n".join(str(item.value) for item in app.info),
        )

    def test_model_insights_renders_derived_category_evaluation(self):
        app = _run_page("Model Insights", station="Delhi")
        self.assertFalse(app.exception)
        text = "\n".join(
            [
                _markdown_text(app),
                _caption_text(app),
                "\n".join(str(item.value) for item in app.warning),
                "\n".join(str(item.value) for item in app.subheader),
                "\n".join(str(item.value) for item in app.dataframe),
            ]
        )
        self.assertIn("Historical AQI Category Classification Evaluation", text)
        self.assertIn("Macro Precision", "\n".join(str(item.label) for item in app.metric))
        self.assertIn("2023-01-01", text)
        self.assertIn("2023-12-30", text)
        self.assertIn("AQI source", text)
        self.assertIn("boundary overlap", text)
        self.assertIn("Actual support", text)
        self.assertTrue(any(item.key == "category_evaluation_model" for item in app.selectbox))
        self.assertTrue(any(item.key == "category_evaluation_horizon" for item in app.selectbox))


if __name__ == "__main__":
    unittest.main()
