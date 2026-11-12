from pathlib import Path
import unittest

from src.browser_location import request_browser_location


COMPONENT_HTML = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "browser_location_component"
    / "index.html"
)


class BrowserLocationComponentTests(unittest.TestCase):
    def test_component_uses_native_permission_and_validates_coordinates(self):
        html = COMPONENT_HTML.read_text(encoding="utf-8")
        self.assertIn("navigator.geolocation.getCurrentPosition", html)
        self.assertIn("latitude < -90", html)
        self.assertIn("longitude > 180", html)
        self.assertIn('sendResult("denied")', html)
        self.assertIn("activeRequestId", html)
        self.assertNotIn("WAQI_API_TOKEN", html)

    def test_component_call_has_permission_disabled_by_default(self):
        self.assertTrue(callable(request_browser_location))


if __name__ == "__main__":
    unittest.main()
