"""Read-only XKDR registry and recent-measurement access for Recent mode."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st


BASE_URL = "https://airquality.xkdr.org/v1"
REQUEST_TIMEOUT = (5, 20)


def _api_key() -> str | None:
    """Read the key only from Streamlit secrets; never fall back to literals."""
    try:
        value = st.secrets["XKDR_API_KEY"]
    except Exception:
        return None
    return str(value).strip() or None


def _safe_error(payload: Any, status_code: int, api_key: str) -> str:
    """Return a short API error label without serializing response contents."""
    if isinstance(payload, dict):
        value = payload.get("detail") or payload.get("error")
        if isinstance(value, str) and value.strip():
            value = value.replace(api_key, "[REDACTED]")
            return value[:200]
    return f"HTTP {status_code}"


def _get_json(path: str, api_key: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Perform one bounded authenticated request and retain no raw body on failure."""
    try:
        response = requests.get(
            f"{BASE_URL}/{path.lstrip('/')}",
            params=params,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout:
        return {"status": None, "payload": None, "error": "Request timed out"}
    except requests.RequestException:
        return {"status": None, "payload": None, "error": "Source unavailable"}

    try:
        payload = response.json()
    except ValueError:
        return {
            "status": response.status_code,
            "payload": None,
            "error": "Response was not valid JSON",
        }

    if not response.ok:
        return {
            "status": response.status_code,
            "payload": None,
            "error": _safe_error(payload, response.status_code, api_key),
        }
    return {"status": response.status_code, "payload": payload, "error": None}


def _records(payload: Any) -> list[dict[str, Any]]:
    """Extract object records from the documented data envelope, if present."""
    if not isinstance(payload, dict):
        return []
    rows = payload.get("data")
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, dict)]
    if isinstance(rows, dict):
        for key in ("records", "items", "stations", "parameters"):
            nested = rows.get(key)
            if isinstance(nested, list):
                return [dict(row) for row in nested if isinstance(row, dict)]
    return []


@st.cache_data(ttl=900, show_spinner=False)
def load_xkdr_registry() -> dict[str, Any]:
    """Load the XKDR metadata, parameter catalog, and CPCB station registry."""
    api_key = _api_key()
    if not api_key:
        return {
            "configured": False,
            "meta": None,
            "parameters": [],
            "stations": [],
            "statuses": {},
        }

    meta_result = _get_json("meta", api_key)
    parameter_result = _get_json("parameters", api_key)
    station_result = _get_json("stations", api_key, {"source": "cpcb_caaqm"})
    meta_payload = meta_result["payload"]

    return {
        "configured": True,
        "meta": meta_payload if isinstance(meta_payload, dict) else None,
        "parameters": _records(parameter_result["payload"]),
        "stations": _records(station_result["payload"]),
        "statuses": {
            "meta": {"status": meta_result["status"], "error": meta_result["error"]},
            "parameters": {"status": parameter_result["status"], "error": parameter_result["error"]},
            "stations": {"status": station_result["status"], "error": station_result["error"]},
        },
    }


@st.cache_data(ttl=900, show_spinner=False)
def load_recent_measurements(city: str, parameter: str | None = None) -> dict[str, Any]:
    """Fetch raw recent CPCB readings without aggregation or AQI derivation."""
    api_key = _api_key()
    if not api_key:
        return {"status": None, "error": "XKDR_API_KEY is not configured", "records": []}

    end_date = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    start_date = end_date - timedelta(days=30)
    params: dict[str, Any] = {
        "source": "cpcb_caaqm",
        "city": city,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "limit": "1000",
        "format": "json",
    }
    if parameter:
        params["parameter"] = parameter

    result = _get_json("measurements", api_key, params)
    return {
        "status": result["status"],
        "error": result["error"],
        "records": _records(result["payload"]),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }


def records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert actual API records to a frame without imputing or renaming values."""
    return pd.DataFrame.from_records(records)


def registry_locations(stations: list[dict[str, Any]]) -> list[str]:
    """Return city names actually present in the fetched CPCB station registry."""
    cities = {
        str(station.get("city_name", "")).strip()
        for station in stations
        if str(station.get("city_name", "")).strip()
    }
    return sorted(cities)