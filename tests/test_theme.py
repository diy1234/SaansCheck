from __future__ import annotations

import unittest

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.theme import apply_chart_theme, chart_palette, theme_css


class ThemeTests(unittest.TestCase):
    def test_dark_and_light_themes_have_distinct_accessible_palettes(self):
        dark = chart_palette("Dark")
        light = chart_palette("Light")

        self.assertEqual(dark["background"], "#0B1220")
        self.assertEqual(dark["text"], "#F1F5F9")
        self.assertEqual(light["background"], "#F5F8FA")
        self.assertEqual(light["text"], "#17232D")
        self.assertNotEqual(dark["surface"], light["surface"])

    def test_theme_styles_include_system_and_reduced_motion_support(self):
        dark_styles = theme_css("Dark")
        light_styles = theme_css("Light")

        self.assertIn("color-scheme: dark", dark_styles)
        self.assertIn("color-scheme: light", light_styles)
        self.assertIn("prefers-reduced-motion: reduce", dark_styles)
        self.assertIn('[data-testid="stSidebarContent"]', dark_styles)
        self.assertIn(".sc-topnav-marker", dark_styles)
        self.assertIn('[data-testid="stSelectbox"]', dark_styles)
        self.assertIn("#0B1220", dark_styles)
        self.assertIn("#F5F8FA", light_styles)
        for styles in (dark_styles, light_styles):
            self.assertIn(".sc-historical-aqi-value", styles)
            self.assertIn(".insight-card .value", styles)
            self.assertIn("color: var(--sc-text) !important", styles)

    def test_chart_theme_sets_surface_and_text_colors(self):
        figure, axis = plt.subplots()
        try:
            apply_chart_theme(figure, "Dark")
            palette = chart_palette("Dark")
            self.assertEqual(figure.get_facecolor()[:3], (11 / 255, 18 / 255, 32 / 255))
            self.assertEqual(axis.get_facecolor()[:3], (17 / 255, 28 / 255, 46 / 255))
            self.assertEqual(axis.title.get_color(), palette["text"])
        finally:
            plt.close(figure)

    def test_unknown_theme_is_rejected(self):
        with self.assertRaises(ValueError):
            theme_css("System")


if __name__ == "__main__":
    unittest.main()
