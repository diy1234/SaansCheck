"""AQI category definitions and current-observation validation for the UI scale."""

from __future__ import annotations

import math
from datetime import datetime
from numbers import Real
from typing import Any, Mapping

from src.waqi_data import AQI_SCALE_LABEL, SOURCE_LABEL


INDIAN_CPCB_SCALE = "Indian National AQI — CPCB"
US_AQI_SCALE = "WAQI — US AQI"

CPCB_CATEGORIES = (
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (500, "Severe"),
)
US_AQI_CATEGORIES = (
    (50, "Good"),
    (100, "Moderate"),
    (150, "Unhealthy for Sensitive Groups"),
    (200, "Unhealthy"),
    (300, "Very Unhealthy"),
    (500, "Hazardous"),
)


def _numeric_aqi(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        return None
    return numeric


def aqi_category(scale: str, value: Any) -> str | None:
    """Return a category using only the explicitly selected AQI scale."""
    numeric = _numeric_aqi(value)
    if numeric is None:
        return None
    if scale == INDIAN_CPCB_SCALE:
        categories = CPCB_CATEGORIES
    elif scale == US_AQI_SCALE:
        categories = US_AQI_CATEGORIES
    else:
        return None
    for upper_bound, category in categories:
        if numeric <= upper_bound:
            return category
    return None


def _valid_timestamp(observation: Mapping[str, Any]) -> bool:
    timestamp = observation.get("observed_at")
    return (
        isinstance(timestamp, datetime)
        and timestamp.tzinfo is not None
        and timestamp.utcoffset() is not None
    )


def _valid_cpcb_observation(observation: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(observation, Mapping):
        return None
    value = _numeric_aqi(observation.get("aqi"))
    category = aqi_category(INDIAN_CPCB_SCALE, value)
    if (
        observation.get("available") is not True
        or observation.get("scale") != "Indian National AQI"
        or observation.get("freshness") != "FRESH"
        or category is None
        or observation.get("category") != category
        or not observation.get("station")
        or not observation.get("location")
        or not _valid_timestamp(observation)
    ):
        return None
    return {
        "scale": INDIAN_CPCB_SCALE,
        "aqi": value,
        "category": category,
        "station": str(observation["station"]),
        "source": str(observation.get("source") or "CPCB"),
    }


def _valid_waqi_observation(observation: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(observation, Mapping):
        return None
    value = _numeric_aqi(observation.get("aqi"))
    category = aqi_category(US_AQI_SCALE, value)
    if (
        observation.get("available") is not True
        or observation.get("source") != SOURCE_LABEL
        or observation.get("aqi_label") != AQI_SCALE_LABEL
        or observation.get("freshness") != "FRESH"
        or (category is None and (value is None or value <= 500))
        or not observation.get("station_name")
        or not _valid_timestamp(observation)
    ):
        return None
    return {
        "scale": US_AQI_SCALE,
        "aqi": value,
        "category": category,
        "beyond_range": value > 500,
        "station": str(observation["station_name"]),
        "source": SOURCE_LABEL,
    }


def current_risk_observation(
    cpcb_observation: Mapping[str, Any] | None,
    waqi_observation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Prefer a fresh verified CPCB value; otherwise use fresh WAQI on its own scale."""
    return (
        _valid_cpcb_observation(cpcb_observation)
        or _valid_waqi_observation(waqi_observation)
    )
