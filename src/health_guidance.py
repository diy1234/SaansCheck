"""Public-health guidance for WAQI-reported US EPA 2016 AQI values."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any


WAQI_AQI_SCALE_URL = "https://aqicn.org/api/"
WAQI_AQI_CATEGORIES = (
    (0, 50, "Good", "Air quality is satisfactory, with little or no risk from air pollution.",
     "Enjoy outdoor activities."),
    (51, 100, "Moderate",
     "Air quality is acceptable, though unusually sensitive people may experience effects.",
     "Sensitive people, including active children and people with respiratory disease, should limit prolonged outdoor exertion."),
    (101, 150, "Unhealthy for Sensitive Groups",
     "Sensitive groups may experience health effects; the general public is less likely to be affected.",
     "Sensitive people should limit prolonged outdoor exertion."),
    (151, 200, "Unhealthy",
     "Everyone may begin to experience health effects; sensitive groups may experience more serious effects.",
     "Sensitive people should avoid prolonged outdoor exertion; others should limit it."),
    (201, 300, "Very Unhealthy",
     "Health warnings apply to the entire population, which is more likely to be affected.",
     "Sensitive people should avoid outdoor exertion; others should limit it."),
    (301, math.inf, "Hazardous",
     "Everyone may experience more serious health effects.",
     "Everyone should avoid outdoor exertion."),
)


def build_health_guidance(observation: dict[str, Any]) -> dict[str, str] | None:
    """Return scale-specific guidance only for a valid, fresh observation."""
    aqi = observation.get("aqi")
    observed_at = observation.get("observed_at")
    if (
        not observation.get("available")
        or observation.get("freshness") != "FRESH"
        or isinstance(aqi, bool)
        or not isinstance(aqi, (int, float))
        or not math.isfinite(float(aqi))
        or float(aqi) < 0
        or not isinstance(observed_at, datetime)
        or observed_at.tzinfo is None
    ):
        return None

    value = float(aqi)
    for lower, upper, category, effects, precautions in WAQI_AQI_CATEGORIES:
        if lower <= value <= upper:
            return {
                "category": category,
                "effects": effects,
                "precautions": precautions,
                "scale": "WAQI-reported US EPA 2016 AQI",
                "scale_url": WAQI_AQI_SCALE_URL,
            }
    return None
