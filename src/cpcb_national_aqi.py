"""Fail-closed access to provider-reported CPCB National AQI from OGD."""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import requests
import streamlit as st


RESOURCE_URL = (
    "https://api.data.gov.in/resource/"
    "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
)
SOURCE_LABEL = "data.gov.in · CPCB-sourced resource"
REQUEST_TIMEOUT = (3, 5)
FRESH_MAX_AGE = timedelta(hours=1)
DELAYED_MAX_AGE = timedelta(hours=6)
CPCB_CATEGORIES = (
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (500, "Severe"),
)
REQUIRED_FIELDS = ("location", "station", "observed_at", "aqi")
LOCATION_NAMES = {
    "delhi": "delhi",
    "new delhi": "delhi",
    "nct of delhi": "delhi",
    "national capital territory of delhi": "delhi",
    "faridabad": "faridabad",
    "ghaziabad": "ghaziabad",
    "gurgaon": "gurgaon",
    "gurugram": "gurgaon",
    "noida": "noida",
}
DIRECT_AQI_FIELD_NAMES = frozenset(
    {"aqi", "overall_aqi", "national_aqi", "national_aqi_value"}
)


def cpcb_aqi_category(value: Any) -> str | None:
    """Return CPCB's category for a finite overall National AQI in [0, 500]."""
    if isinstance(value, bool):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value) or not 0 <= numeric_value <= 500:
        return None
    for upper_bound, category in CPCB_CATEGORIES:
        if numeric_value <= upper_bound:
            return category
    return None


def _unavailable(message: str) -> dict[str, Any]:
    return {
        "available": False,
        "aqi": None,
        "category": None,
        "station": None,
        "location": None,
        "observed_at": None,
        "freshness": None,
        "source": SOURCE_LABEL,
        "scale": "Indian National AQI",
        "error": message,
    }


def _setting(name: str) -> Any:
    value = os.getenv(name)
    if value is not None:
        return value
    try:
        return st.secrets.get(name)
    except (KeyError, FileNotFoundError):
        return None


def _api_key() -> str | None:
    value = _setting("CPCB_OGD_API_KEY")
    if value is None:
        value = _setting("CPCB_API_KEY")
    return str(value).strip() if value is not None and str(value).strip() else None


def _schema_configuration() -> tuple[str, dict[str, str]] | None:
    records_path = _setting("CPCB_OGD_RECORDS_PATH")
    raw_field_map = _setting("CPCB_OGD_FIELD_MAP")
    confirmed_overall_aqi = _setting("CPCB_OGD_AQI_IS_OFFICIAL_NATIONAL")
    if not records_path or not raw_field_map:
        return None
    if str(confirmed_overall_aqi).strip().casefold() not in {"true", "1", "yes"}:
        return None
    try:
        field_map = (
            dict(raw_field_map)
            if isinstance(raw_field_map, Mapping)
            else json.loads(str(raw_field_map))
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(field_map, dict):
        return None
    normalized_map = {
        str(key): str(value).strip()
        for key, value in field_map.items()
        if str(value).strip()
    }
    if any(not normalized_map.get(name) for name in REQUIRED_FIELDS):
        return None
    if normalized_map["aqi"].casefold() not in DIRECT_AQI_FIELD_NAMES:
        return None
    return str(records_path).strip(), normalized_map


@st.cache_data(ttl=120, show_spinner=False)
def _request_records(api_key: str) -> dict[str, Any]:
    try:
        response = requests.get(
            RESOURCE_URL,
            params={"api-key": api_key, "format": "json", "limit": "1000"},
            headers={"Accept": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        return {"error": "The data.gov.in CPCB request timed out."}
    except requests.RequestException:
        return {"error": "The data.gov.in CPCB source is unavailable."}

    if response.status_code == 429:
        return {"error": "The data.gov.in API rate limit was reached; try again later."}
    if response.status_code in {401, 403}:
        return {"error": "data.gov.in rejected the configured CPCB API key or access."}
    if not response.ok:
        return {"error": f"The data.gov.in CPCB request failed (HTTP {response.status_code})."}
    try:
        payload = response.json()
    except ValueError:
        return {"error": "The data.gov.in CPCB response was not valid JSON."}
    return {"payload": payload}


def _records_at_path(payload: Any, path: str) -> list[dict[str, Any]] | None:
    value = payload
    if path != "$":
        for component in path.split("."):
            if not isinstance(value, Mapping) or component not in value:
                return None
            value = value[component]
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        return None
    return [dict(row) for row in value]


def _normalize_location(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())
    return LOCATION_NAMES.get(normalized)


def _normalize_station(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())


def _parse_timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    return parsed.astimezone(timezone.utc)


def _freshness(timestamp: datetime, now: datetime) -> tuple[str, float] | None:
    age = now.astimezone(timezone.utc) - timestamp
    if age < timedelta(0):
        return None
    if age <= FRESH_MAX_AGE:
        return "FRESH", age.total_seconds() / 60
    if age <= DELAYED_MAX_AGE:
        return "DELAYED", age.total_seconds() / 60
    return "STALE", age.total_seconds() / 60


def build_cpcb_health_guidance(observation: Mapping[str, Any]) -> dict[str, str] | None:
    """Return Indian-scale guidance only for a valid, fresh CPCB observation."""
    category = cpcb_aqi_category(observation.get("aqi"))
    observed_at = observation.get("observed_at")
    if (
        not observation.get("available")
        or observation.get("freshness") != "FRESH"
        or category is None
        or not isinstance(observed_at, datetime)
        or observed_at.tzinfo is None
    ):
        return None
    guidance = {
        "Good": (
            "Air quality is considered good, with minimal expected health impact.",
            "Enjoy usual outdoor activities.",
        ),
        "Satisfactory": (
            "Sensitive people may experience minor breathing discomfort.",
            "People with respiratory sensitivity may limit prolonged outdoor exertion.",
        ),
        "Moderate": (
            "People with lung or heart disease, children, and older adults may experience breathing discomfort.",
            "Sensitive people should reduce prolonged outdoor exertion.",
        ),
        "Poor": (
            "Prolonged exposure may cause breathing discomfort for many people.",
            "Reduce prolonged outdoor exertion, especially if sensitive to air pollution.",
        ),
        "Very Poor": (
            "Prolonged exposure may cause respiratory illness.",
            "Avoid prolonged outdoor exertion and follow local public-health advice.",
        ),
        "Severe": (
            "Health effects may occur even in healthy people, with more serious effects for sensitive groups.",
            "Avoid outdoor exertion when practical and follow local public-health advice.",
        ),
    }
    effects, precautions = guidance[category]
    return {
        "category": category,
        "effects": effects,
        "precautions": precautions,
        "scale": "Indian National AQI",
        "scale_url": "https://airquality.cpcb.gov.in/AQI/",
    }


def build_cpcb_observed_alert(observation: Mapping[str, Any]) -> dict[str, str] | None:
    """Create a CPCB-scale alert without passing values to WAQI alert rules."""
    aqi = observation.get("aqi")
    observed_at = observation.get("observed_at")
    category = cpcb_aqi_category(aqi)
    if (
        not observation.get("available")
        or observation.get("freshness") != "FRESH"
        or not isinstance(observed_at, datetime)
        or observed_at.tzinfo is None
        or category in {None, "Good", "Satisfactory"}
    ):
        return None
    return {
        "category": category,
        "station": str(observation["station"]),
        "aqi": str(observation["aqi_display"]),
    }


@st.cache_data(ttl=60, show_spinner=False)
def load_cpcb_national_aqi(
    selected_location: str,
    selected_station: str | None,
) -> dict[str, Any]:
    """Load an exact-station, fresh provider-reported AQI only when configured."""
    location = _normalize_location(selected_location)
    if location is None:
        return _unavailable(
            "Choose a supported city manually; nearby browser location is not a CPCB station mapping."
        )
    if not selected_station or not str(selected_station).strip():
        return _unavailable("Select a monitoring station before matching CPCB data.")
    api_key = _api_key()
    if not api_key:
        return _unavailable("CPCB_OGD_API_KEY is not configured.")
    schema = _schema_configuration()
    if schema is None:
        return _unavailable(
            "The OGD response schema and direct overall CPCB AQI field are not verified/configured."
        )

    records_path, field_map = schema
    request_result = _request_records(api_key)
    if request_result.get("error"):
        return _unavailable(request_result["error"])
    records = _records_at_path(request_result.get("payload"), records_path)
    if records is None:
        return _unavailable("The OGD response did not match the configured records schema.")

    source_station = _normalize_station(selected_station)
    matching = [
        row for row in records
        if _normalize_location(row.get(field_map["location"])) == location
        and _normalize_station(row.get(field_map["station"])) == source_station
    ]
    if not matching:
        return _unavailable(
            "No CPCB record matched the selected city and exact station."
        )

    now = datetime.now(timezone.utc)
    candidates = []
    for row in matching:
        timestamp = _parse_timestamp(row.get(field_map["observed_at"]))
        category = cpcb_aqi_category(row.get(field_map["aqi"]))
        if timestamp is None or category is None:
            continue
        freshness = _freshness(timestamp, now)
        if freshness is not None:
            candidates.append((timestamp, category, freshness, row[field_map["aqi"]]))
    if not candidates:
        return _unavailable("The matched CPCB record has no valid AQI or timestamp.")

    timestamp, category, (freshness_label, age_minutes), raw_aqi = max(
        candidates, key=lambda candidate: candidate[0]
    )
    if freshness_label == "STALE":
        return _unavailable("The matched CPCB AQI observation is stale.")
    return {
        "available": True,
        "aqi": float(raw_aqi),
        "aqi_display": str(raw_aqi).strip(),
        "category": category,
        "station": str(selected_station).strip(),
        "location": selected_location,
        "observed_at": timestamp,
        "freshness": freshness_label,
        "age_minutes": age_minutes,
        "source": SOURCE_LABEL,
        "scale": "Indian National AQI",
        "error": None,
    }
