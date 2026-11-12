"""Readiness and date alignment for WAQI-backed forecast displays."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.forecast_features import build_direct_forecasts


LOCAL_TIMEZONE = "Asia/Kolkata"
LOCATION_ALIASES = {
    "Delhi": {"delhi"},
    "Faridabad": {"faridabad"},
    "Ghaziabad": {"ghaziabad"},
    "Gurgaon": {"gurgaon", "gurugram"},
    "Gurugram": {"gurgaon", "gurugram"},
    "Gurgaon / Gurugram": {"gurgaon", "gurugram"},
    "Noida": {"noida"},
}
FORECAST_UNAVAILABLE_REASON = (
    "Forecast unavailable: no WAQI-authorized, station-matched daily history "
    "and model validated on compatible AQI scale, pollutant units, and "
    "averaging periods are available."
)
TODAY_ESTIMATE_UNAVAILABLE_REASON = (
    "No separately validated same-day estimation method is available."
)
REQUIRED_DAILY_HISTORY_COLUMNS = {
    "date",
    "location_name",
    "aqi",
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
}


def _has_verified_history_scale(daily_history: pd.DataFrame | None) -> bool:
    if daily_history is None:
        return False
    attrs = getattr(daily_history, "attrs", {})
    return bool(
        attrs.get("aqi_scale_verified")
        or attrs.get("aqi_definition_verified")
        or attrs.get("scale_verified")
    )


def validate_compatible_daily_history(
    daily_history: pd.DataFrame | None,
    station_name: str,
    observation: dict[str, Any] | None = None,
    expected_station_id: str | None = None,
    local_timezone: str = LOCAL_TIMEZONE,
) -> tuple[bool, str]:
    """Reject daily history unless it is explicitly verified and compatible."""
    if daily_history is None or not isinstance(daily_history, pd.DataFrame) or daily_history.empty:
        return False, (
            "Forecast unavailable because the selected station does not have "
            "a compatible daily history for the model contract."
        )

    if not _has_verified_history_scale(daily_history):
        return False, (
            "Forecast unavailable because the historical AQI scale and definition "
            "are not explicitly verified as compatible with the live WAQI observation."
        )

    missing = sorted(REQUIRED_DAILY_HISTORY_COLUMNS.difference(daily_history.columns))
    if missing:
        return False, (
            "Forecast unavailable because the daily history is missing required model columns: "
            f"{missing}."
        )

    requested_location = observation.get("requested_location") if observation else None
    location_names = {""}
    if station_name:
        location_names.update(part.strip().casefold() for part in station_name.split(",") if part.strip())
    if requested_location:
        location_names.add(requested_location.casefold())
    location_names.discard("")

    station_history = daily_history.loc[
        daily_history["location_name"].astype(str).str.casefold().isin(location_names)
    ].copy()
    if station_history.empty:
        return False, (
            f"Forecast unavailable because there are no compatible records for {station_name} "
            "in the supplied daily history."
        )

    station_history["date"] = pd.to_datetime(station_history["date"], errors="coerce")
    if station_history["date"].isna().any():
        return False, (
            "Forecast unavailable because the supplied daily history contains invalid dates."
        )
    if len(station_history) < 4:
        return False, (
            "Forecast unavailable because at least four daily rows are required to build "
            "the required AQI lags and rolling statistics."
        )

    if expected_station_id is not None and "station_id" in station_history.columns:
        unique_ids = station_history["station_id"].dropna().astype(str).unique()
        if unique_ids.size and expected_station_id not in unique_ids:
            return False, (
                "Forecast unavailable because the supplied daily history does not match "
                "the selected WAQI station ID."
            )

    latest_date = pd.Timestamp(station_history["date"].max()).tz_localize("UTC").tz_convert(
        ZoneInfo(local_timezone)
    ).normalize()
    if observation and isinstance(observation.get("observed_at"), datetime):
        observed_local = pd.Timestamp(observation["observed_at"]).tz_convert(
            ZoneInfo(local_timezone)
        ).normalize()
        if latest_date != observed_local:
            return False, (
                "Forecast unavailable because the latest compatible daily-history date "
                "does not match the selected station observation date."
            )

    return True, "Compatible daily history validated for live forecasting."


def _build_predictions_from_history(
    daily_history: pd.DataFrame,
    station_name: str,
    models_dict: dict[str, Any],
) -> dict[str, float]:
    forecast_rows = build_direct_forecasts(daily_history, station_name, models_dict)
    results: dict[str, float] = {
        "tomorrow": float(forecast_rows[0]["predicted_aqi"]),
        "day_after": float(forecast_rows[1]["predicted_aqi"]),
    }
    if len(forecast_rows) > 2:
        results["day_after_2"] = float(forecast_rows[2]["predicted_aqi"])
    return results


def build_live_forecast_status(
    observation: dict[str, Any],
    local_timezone: str = LOCAL_TIMEZONE,
    expected_station_id: str | None = None,
    now: datetime | None = None,
    candidate_daily_history: pd.DataFrame | None = None,
    models_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a genuine observation and target dates, never unsupported predictions."""
    result: dict[str, Any] = {
        "available": False,
        "station_id": None,
        "station_name": None,
        "observed_aqi": None,
        "observed_at": None,
        "freshness": observation.get("freshness"),
        "source": observation.get("source"),
        "forecast_origin": None,
        "target_dates": {"today": None, "tomorrow": None, "day_after": None},
        "observation_is_today": False,
        "today_estimate": None,
        "predictions": {"tomorrow": None, "day_after": None},
        "reason": FORECAST_UNAVAILABLE_REASON,
        "compatibility_reason": FORECAST_UNAVAILABLE_REASON,
        "today_estimate_reason": TODAY_ESTIMATE_UNAVAILABLE_REASON,
    }
    if not observation.get("available"):
        result["reason"] = observation.get("error") or (
            "Forecast unavailable because the selected station observation could not be retrieved."
        )
        return result

    station_id = observation.get("station_id")
    station_name = observation.get("station_name")
    aqi = observation.get("aqi")
    observed_at = observation.get("observed_at")
    if (
        station_id is None
        or not isinstance(station_name, str)
        or not station_name.strip()
        or isinstance(aqi, bool)
        or not isinstance(aqi, (int, float))
        or not math.isfinite(float(aqi))
        or float(aqi) < 0
        or not isinstance(observed_at, datetime)
        or observed_at.tzinfo is None
    ):
        result["reason"] = (
            "Forecast unavailable because the selected station observation "
            "is missing a valid AQI, station identity, or timezone-aware timestamp."
        )
        return result

    if expected_station_id is not None and str(station_id) != str(expected_station_id):
        result["reason"] = (
            "Forecast unavailable because the observation station ID does not "
            "match the selected WAQI station."
        )
        return result
    requested_location = observation.get("requested_location")
    aliases = LOCATION_ALIASES.get(requested_location)
    segments = {
        " ".join(
            "".join(character.casefold() if character.isalnum() else " " for character in segment).split()
        )
        for segment in station_name.split(",")
    }
    if aliases is None or not segments.intersection(aliases):
        result["reason"] = (
            "Forecast unavailable because the returned WAQI station name does "
            "not match the selected city."
        )
        return result

    result["station_id"] = str(station_id)
    result["station_name"] = station_name
    result["observed_aqi"] = float(aqi)
    result["observed_at"] = observed_at
    local_origin = pd.Timestamp(observed_at).tz_convert(ZoneInfo(local_timezone))
    origin_date = local_origin.normalize()
    result["available"] = True
    result["forecast_origin"] = local_origin
    current_time = now or datetime.now(ZoneInfo(local_timezone))
    if current_time.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local_today = pd.Timestamp(current_time).tz_convert(
        ZoneInfo(local_timezone)
    ).normalize()
    result["observation_is_today"] = origin_date == local_today
    result["target_dates"] = {
        "today": origin_date,
        "tomorrow": origin_date + pd.Timedelta(days=1),
        "day_after": origin_date + pd.Timedelta(days=2),
    }
    if observation.get("freshness") != "FRESH":
        result["reason"] = (
            "Forecast unavailable because the selected station observation is "
            "not fresh. A fresh WAQI observation is required."
        )
        return result

    if candidate_daily_history is None or models_dict is None:
        result["compatibility_reason"] = (
            "Forecast unavailable because no station-matched, scale-verified daily history "
            "and trained model set were provided for the selected station; the AQI scale "
            "and daily-history contract remain incompatible."
        )
        return result

    forecast_station = observation.get("requested_location") or station_name
    if forecast_station not in {"Delhi", "Faridabad", "Ghaziabad", "Gurgaon", "Noida"}:
        forecast_station = station_name

    valid, validation_reason = validate_compatible_daily_history(
        candidate_daily_history,
        forecast_station,
        observation,
        expected_station_id,
        local_timezone,
    )
    if not valid:
        result["reason"] = validation_reason
        result["compatibility_reason"] = validation_reason
        return result

    try:
        result["predictions"] = _build_predictions_from_history(
            candidate_daily_history,
            forecast_station,
            models_dict,
        )
        result["compatibility_reason"] = (
            "Compatible station-matched historical daily data and model contract were validated; "
            "the displayed forecast values are authentic model outputs from the selected station history."
        )
        result["reason"] = (
            "Forecast values are produced only from verified, station-matched daily history and "
            "do not mix WAQI observations with historical AQI features without a verified scale match."
        )
        return result
    except (ValueError, KeyError, TypeError) as error:
        result["reason"] = (
            "Forecast unavailable because the supplied daily history could not satisfy the "
            f"historical model contract: {error}"
        )
        result["compatibility_reason"] = result["reason"]
        return result
