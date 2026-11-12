"""Browser geolocation component for one-time approximate-area lookups."""

from pathlib import Path

import streamlit.components.v1 as components


_component = components.declare_component(
    "saanscheck_browser_location",
    path=str(Path(__file__).with_name("browser_location_component")),
)


def request_browser_location(
    request_id: str,
    enabled: bool,
    clear_result: bool = False,
) -> dict | None:
    """Ask the browser for location only when the active request is enabled."""
    return _component(
        request_id=request_id,
        enabled=enabled,
        clear_result=clear_result,
        default=None,
    )
