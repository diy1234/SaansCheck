from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _run_page(page: str) -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["workspace_mode"] = "Historical Intelligence"
    app.session_state["active_page"] = page
    app.run()
    return app


def _run_recent_page(page: str = "Recent Overview") -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.session_state["workspace_mode"] = "Recent Air Quality"
    app.session_state["active_page"] = page
    app.run()
    return app


def _visible_text(app: AppTest) -> str:
    return "\n".join(str(element.value) for element in app.markdown)


def _headings(app: AppTest) -> str:
    return "\n".join(
        str(element.value)
        for element in [*app.title, *app.header, *app.subheader]
    )


class DashboardFeatureTests(unittest.TestCase):
    def test_theme_switch_preserves_location_and_survives_widget_reruns(self):
        app = _run_page("Overview")
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["appearance_theme"], "Dark")

        app.selectbox(key="historical_station").set_value("Noida").run()
        app.segmented_control(key="appearance_theme").set_value("Light").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["appearance_theme"], "Light")
        self.assertEqual(app.session_state["historical_station"], "Noida")

        app.radio(key="workspace_mode").set_value("Recent Air Quality").run()
        app.selectbox(key="navbar_city_search").set_value("Faridabad").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["appearance_theme"], "Light")
        self.assertEqual(app.session_state["historical_station"], "Faridabad")
        self.assertEqual(app.session_state["workspace_mode"], "Recent Air Quality")

    def test_topbar_city_search_shows_supported_cities_and_aliases(self):
        app = _run_recent_page()
        self.assertFalse(app.exception)
        city_search = next(
            item for item in app.selectbox if item.key == "navbar_city_search"
        )
        self.assertEqual(
            list(city_search.options),
            [
                "Delhi / New Delhi",
                "Noida",
                "Ghaziabad",
                "Gurugram / Gurgaon",
                "Faridabad",
            ],
        )
        self.assertIsNone(city_search.value)

    def test_topbar_city_search_updates_dashboard_and_opens_recent_overview(self):
        app = _run_recent_page()
        app.selectbox(key="navbar_city_search").set_value("Faridabad").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["workspace_mode"], "Recent Air Quality")
        self.assertEqual(app.session_state["active_page"], "Recent Overview")
        self.assertEqual(app.session_state["primary_nav"], "Overview")
        self.assertEqual(app.session_state["historical_station"], "Faridabad")
        self.assertEqual(app.session_state["location_bar_station"], "Faridabad")

    def test_topbar_city_search_resolves_delhi_and_gurugram_aliases(self):
        for label, canonical in (
            ("Delhi / New Delhi", "Delhi"),
            ("Noida", "Noida"),
            ("Ghaziabad", "Ghaziabad"),
            ("Gurugram / Gurgaon", "Gurgaon"),
            ("Faridabad", "Faridabad"),
        ):
            with self.subTest(label=label):
                app = _run_recent_page()
                app.selectbox(key="navbar_city_search").set_value(label).run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["location_bar_station"], canonical)
                self.assertEqual(app.session_state["historical_station"], canonical)

    def test_city_search_and_home_nav_are_not_rendered_in_historical_workspace(self):
        app = _run_page("Overview")
        self.assertFalse(app.exception)
        widget_keys = {item.key for item in app.selectbox}
        button_keys = {item.key for item in app.button}
        self.assertNotIn("navbar_city_search", widget_keys)
        self.assertNotIn("navbar_home", button_keys)
        self.assertNotIn("navbar_brand_home", button_keys)
        self.assertFalse(any(item.label == "Primary navigation" for item in app.segmented_control))

    def test_historical_location_updates_shared_selected_city(self):
        app = _run_page("Overview")
        app.selectbox(key="historical_station").set_value("Noida").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["location_bar_station"], "Noida")

    def test_brand_and_home_buttons_return_to_recent_overview(self):
        for page, button_key in (
            ("Recent Overview", "navbar_home"),
            ("Recent Overview", "navbar_brand_home"),
            ("Pollutant Trends", "navbar_home"),
        ):
            with self.subTest(page=page, button=button_key):
                app = _run_recent_page(page)
                app.session_state["location_bar_station"] = "Noida"
                app.session_state["historical_station"] = "Noida"
                app.run()
                app.button(key=button_key).click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["workspace_mode"], "Recent Air Quality")
                self.assertEqual(app.session_state["active_page"], "Recent Overview")
                self.assertEqual(app.session_state["primary_nav"], "Overview")
                self.assertEqual(app.session_state["location_bar_station"], "Noida")

    def test_overview_renders_dynamic_key_findings(self):
        app = _run_page("Overview")
        self.assertFalse(app.exception)
        self.assertIn("Key Findings", _headings(app))
        self.assertIn("Highest observed historical AQI", _visible_text(app))
        self.assertTrue(any(item.key == "overview_findings_year" for item in app.selectbox))

    def test_calendar_page_renders_historical_calendar_controls(self):
        app = _run_page("Daily AQI Calendar")
        self.assertFalse(app.exception)
        self.assertIn("Daily Historical AQI Calendar", _visible_text(app))
        self.assertTrue(any(item.key == "calendar_year" for item in app.selectbox))
        self.assertTrue(any(item.key == "calendar_month" for item in app.selectbox))
        self.assertFalse(app.info)

    def test_location_page_renders_period_aware_leaderboard(self):
        app = _run_page("Location Intelligence")
        self.assertFalse(app.exception)
        self.assertIn("Leaderboard metric", "\n".join(str(item.label) for item in app.selectbox))
        compare_widget = next(
            item for item in app.multiselect if item.key == "location_comparison_locations"
        )
        self.assertIn(app.session_state["historical_station"], compare_widget.value)
        self.assertIn(app.session_state["historical_station"], compare_widget.options)
        dataframe_columns = {
            str(column)
            for element in app.dataframe
            for column in element.value.columns
        }
        self.assertIn("Valid AQI days", dataframe_columns)
        self.assertIn("Coverage start", dataframe_columns)
        styles = "\n".join(str(element.value) for element in app.markdown)
        self.assertIn(".sc-location-rank-card .kpi-meta strong", styles)
        self.assertIn(".st-key-location_comparison_locations [data-tag]", styles)

    def test_forecast_displays_verified_walk_forward_reliability(self):
        app = _run_page("AQI Forecast")
        self.assertFalse(app.exception)
        self.assertIn("Forecast Error & Reliability", _headings(app))
        self.assertIn("6.828", _visible_text(app))
        self.assertIn("not an accuracy percentage", "\n".join(str(item.value) for item in app.caption))
        forecast_markup = _visible_text(app)
        self.assertIn(
            "Evaluation: 2023 historical test origins; temporal target-boundary overlap is a known limitation.",
            forecast_markup,
        )
        self.assertNotIn("Validation: Unseen 2023 test data", forecast_markup)
        self.assertIn("Random Forest Regressor", forecast_markup)
        stylesheet = next(
            str(element.value)
            for element in app.markdown
            if ".sc-forecast-card .kpi-meta strong" in str(element.value)
        )
        self.assertIn(".sc-forecast-card .kpi-meta strong", stylesheet)
        self.assertIn(".sc-forecast-model-name", stylesheet)
        self.assertIn(".sc-forecast-model-detail", stylesheet)

    def test_learn_page_explains_source_separation_and_limitations(self):
        app = _run_page("Learn About AQI")
        self.assertFalse(app.exception)
        text = _visible_text(app)
        self.assertIn("Learn About AQI", text)
        self.assertIn("not been verified as CPCB", text)
        self.assertIn("pollutant units are not verified", text)
        self.assertIn("not medical advice", "\n".join(str(item.value) for item in app.info))

    def test_intro_renders_in_sidebar_once_per_session(self):
        app = _run_page("Overview")
        self.assertTrue(app.session_state["saanscheck_intro_v3_seen"])
        visible_text = _visible_text(app)
        self.assertIn('<div class="sc-sidebar-intro"', visible_text)
        self.assertIn("Delhi-NCR Air Quality Intelligence", visible_text)
        self.assertNotIn("sc-intro-overlay", visible_text)
        self.assertLess(
            visible_text.index('<div class="sc-sidebar-brand"'),
            visible_text.index('<div class="sc-sidebar-intro"'),
        )
        self.assertLess(
            visible_text.index('<div class="sc-sidebar-intro"'),
            visible_text.index("DATA WORKSPACE"),
        )
        stylesheet = next(
            str(element.value)
            for element in app.markdown
            if str(element.value).startswith("<style>")
        )
        self.assertIn("sc-sidebar-symbol 5s", stylesheet)
        self.assertIn("sc-sidebar-firework 5s", stylesheet)
        self.assertIn("sc-sidebar-glow 5s", stylesheet)
        self.assertIn("sc-sidebar-copy 5s", stylesheet)
        self.assertIn("prefers-reduced-motion: reduce", stylesheet)

    def test_fullscreen_startup_intro_precedes_dashboard_and_is_session_gated(self):
        app = _run_page("Overview")
        self.assertTrue(app.session_state["saanscheck_startup_intro_v1_seen"])
        visible_text = _visible_text(app)
        self.assertIn('id="sc-startup-intro"', visible_text)
        self.assertIn("SaansCheck emblem", visible_text)
        self.assertIn("Delhi-NCR Air Quality Intelligence", visible_text)
        self.assertLess(
            visible_text.index('id="sc-startup-intro"'),
            visible_text.index('<div class="sc-sidebar-brand"'),
        )
        stylesheet = next(
            str(element.value)
            for element in app.markdown
            if str(element.value).startswith("<style>")
        )
        self.assertIn("sc-startup-lifecycle 4s", stylesheet)
        self.assertIn("sc-startup-scroll-release 4s", stylesheet)
        self.assertIn("z-index: 2147483647", stylesheet)
        self.assertIn("prefers-reduced-motion: reduce", stylesheet)
        self.assertNotIn('<span class="sc-topnav-marker">', visible_text)
        app.selectbox(key="historical_station").set_value("Faridabad").run()
        self.assertFalse(app.exception)
        self.assertNotIn('id="sc-startup-intro"', _visible_text(app))

    def test_intro_does_not_reappear_on_later_widget_rerun(self):
        app = _run_page("Overview")
        station = app.selectbox(key="historical_station")
        station.set_value("Faridabad").run()
        self.assertFalse(app.exception)
        self.assertNotIn('<div class="sc-sidebar-intro"', _visible_text(app))
        self.assertEqual(app.session_state["historical_station"], "Faridabad")


if __name__ == "__main__":
    unittest.main()
