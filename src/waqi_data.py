"""Read-only WAQI station search and observation access."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from numbers import Real
from typing import Any

import requests
import streamlit as st


API_BASE_URL = "https://api.waqi.info"
SEARCH_URL = f"{API_BASE_URL}/search/"
REQUEST_TIMEOUT = (5, 20)
CACHE_TTL_SECONDS = 120
POLLUTANT_FIELDS = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co",
    "O3": "o3",
}
LOCATION_ALIASES = {
    "Delhi": (
        "delhi",
        "new delhi",
        "nct of delhi",
        "national capital territory of delhi",
    ),
    "Faridabad": ("faridabad",),
    "Ghaziabad": ("ghaziabad",),
    "Gurgaon": ("gurgaon", "gurugram"),
    "Gurugram": ("gurgaon", "gurugram"),
    "Gurgaon / Gurugram": ("gurgaon", "gurugram"),
    "Noida": ("noida",),
}
SOURCE_LABEL = "World Air Quality Index (WAQI)"
AQI_SCALE_LABEL = "WAQI-reported AQI · US EPA 2016 scale"
AQI_SCALE_URL = "https://aqicn.org/scale/"
UNIT_UNSPECIFIED = "Unit not specified by source"
AQI_UNAVAILABLE = "Not provided by source"
MAX_AUTOMATIC_STATION_DISTANCE_KM = 25.0


class WaqiRequestError(Exception):
    """A sanitized WAQI request or response failure."""


def _load_token() -> str | None:
    """Read the WAQI token only from Streamlit secrets."""
    try:
        token = st.secrets["WAQI_API_TOKEN"]
    except (KeyError, FileNotFoundError):
        return None
    if not isinstance(token, str):
        return None
    return token.strip() or None


def _api_error_message(message: Any, status_code: int | None = None) -> str:
    """Map provider errors to safe messages without reflecting response text."""
    normalized = str(message or "").casefold()
    if status_code in (401, 403) or any(
        marker in normalized for marker in ("invalid token", "invalid key", "unauthorized", "access token")
    ):
        return "WAQI rejected the configured API token. Check WAQI_API_TOKEN."
    if status_code == 429 or any(
        marker in normalized for marker in ("quota", "rate limit", "over limit", "too many")
    ):
        return "WAQI API quota or rate limit was reached. Try again later."
    if status_code is not None and not 200 <= status_code < 300:
        return f"WAQI request failed with HTTP {status_code}."
    return "WAQI API returned an unsuccessful response."


def _request_json(
    url: str,
    params: tuple[tuple[str, str], ...],
    token: str,
) -> dict[str, Any]:
    """Fetch and validate one WAQI response without retaining request coordinates."""
    request_params = dict(params)
    request_params["token"] = token
    try:
        response = requests.get(
            url,
            params=request_params,
            headers={"Accept": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.Timeout as error:
        raise WaqiRequestError("WAQI request timed out.") from error
    except requests.RequestException as error:
        raise WaqiRequestError("WAQI source is unavailable.") from error

    if not response.ok:
        raise WaqiRequestError(_api_error_message(None, response.status_code))

    try:
        payload = response.json()
    except ValueError as error:
        raise WaqiRequestError("WAQI response was not valid JSON.") from error

    if not isinstance(payload, dict) or payload.get("status") != "ok":
        message = payload.get("data") if isinstance(payload, dict) else None
        raise WaqiRequestError(_api_error_message(message, response.status_code))

    return {
        "payload": payload,
        "status_code": response.status_code,
        "fetched_at": datetime.now(timezone.utc),
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_json(
    url: str,
    params: tuple[tuple[str, str], ...],
    token: str,
) -> dict[str, Any]:
    """Cache successful, location-keyed WAQI responses for two minutes."""
    return _request_json(url, params, token)


def _valid_coordinate(value: Any, lower: float, upper: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
        and lower <= float(value) <= upper
    )


def _station_coordinates(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    latitude, longitude = value[0], value[1]
    if not _valid_coordinate(latitude, -90, 90) or not _valid_coordinate(longitude, -180, 180):
        return None
    return float(latitude), float(longitude)


def _distance_km(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    latitude_a, longitude_a = map(math.radians, first)
    latitude_b, longitude_b = map(math.radians, second)
    latitude_delta = latitude_b - latitude_a
    longitude_delta = longitude_b - longitude_a
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_a) * math.cos(latitude_b)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(math.sqrt(min(haversine, 1.0)))


def _numeric_value(value: Any) -> float | None:
    """Convert a finite source-provided numeric value without inventing one."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_timestamp(time_data: Any) -> datetime | None:
    """Parse a timezone-aware WAQI timestamp or its Unix-epoch equivalent."""
    if not isinstance(time_data, dict):
        return None

    zone = None
    offset = time_data.get("tz")
    if isinstance(offset, str) and offset.strip():
        try:
            zone = datetime.fromisoformat(
                f"2000-01-01T00:00:00{offset.strip()}"
            ).tzinfo
        except ValueError:
            zone = None

    iso_value = time_data.get("iso")
    if isinstance(iso_value, str) and iso_value.strip():
        try:
            parsed = datetime.fromisoformat(iso_value.strip().replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None and zone is not None:
                parsed = parsed.replace(tzinfo=zone)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc)

    station_time = time_data.get("s")
    if isinstance(station_time, str) and station_time.strip():
        try:
            parsed = datetime.fromisoformat(station_time.strip().replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None and zone is not None:
                parsed = parsed.replace(tzinfo=zone)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc)

    for field in ("s", "v", "vtime"):
        epoch_value = _numeric_value(time_data.get(field))
        if epoch_value is None or epoch_value <= 0:
            continue
        try:
            return datetime.fromtimestamp(epoch_value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            continue
    return None


def _freshness(observed_at: datetime, now: datetime) -> tuple[str, float] | None:
    """Classify timestamp age without describing cached observations as live."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    age_minutes = (now.astimezone(timezone.utc) - observed_at).total_seconds() / 60
    if age_minutes < 0:
        return None
    if age_minutes <= 60:
        status = "FRESH"
    elif age_minutes <= 6 * 60:
        status = "DELAYED"
    else:
        status = "STALE"
    return status, age_minutes


def refresh_observation_freshness(
    observation: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Re-evaluate source timestamp age before displaying a retained observation."""
    refreshed = observation.copy()
    observed_at = refreshed.get("observed_at")
    if not isinstance(observed_at, datetime) or observed_at.tzinfo is None:
        refreshed["freshness"] = None
        refreshed["age_minutes"] = None
        return refreshed

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    freshness = _freshness(observed_at, current_time)
    if freshness is None:
        refreshed["available"] = False
        refreshed["aqi"] = None
        refreshed["aqi_display"] = AQI_UNAVAILABLE
        refreshed["freshness"] = None
        refreshed["age_minutes"] = None
        refreshed["error"] = "WAQI station observation timestamp is in the future."
        return refreshed

    refreshed["freshness"], refreshed["age_minutes"] = freshness
    return refreshed


def _normalize_city_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()


def _station_matches_location(station_name: str, requested_location: str) -> bool:
    """Require the requested city to appear as an exact comma-delimited segment."""
    aliases = LOCATION_ALIASES.get(requested_location)
    if aliases is None:
        return False
    segments = {
        _normalize_city_name(segment)
        for segment in station_name.split(",")
        if segment.strip()
    }
    return bool(segments.intersection(aliases))


def _normalize_search_text(value: str) -> str:
    """Normalize punctuation and spacing without losing locality-name tokens."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def _station_matches_search(station_name: str, query: str, location: str) -> bool:
    """Match a submitted locality against a station-name segment, not its city."""
    normalized_query = _normalize_search_text(query)
    if not normalized_query:
        return True
    aliases = set(LOCATION_ALIASES.get(location, ()))
    if normalized_query in aliases:
        return True
    query_compact = normalized_query.replace(" ", "")
    segments = (
        _normalize_search_text(segment)
        for segment in station_name.split(",")
        if segment.strip()
    )
    return any(
        segment not in aliases
        and (
            normalized_query in segment
            or query_compact in segment.replace(" ", "")
        )
        for segment in segments
    )


def merge_station_search_results(
    city_search: dict[str, Any],
    locality_search: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge city and locality results by WAQI UID without truncating matches."""
    stations_by_id: dict[str, dict[str, Any]] = {}
    for search in (city_search, locality_search):
        if not isinstance(search, dict):
            continue
        for station in search.get("stations", []):
            if not isinstance(station, dict):
                continue
            station_id = station.get("station_id")
            if isinstance(station_id, str) and station_id.isdigit():
                stations_by_id.setdefault(station_id, station)

    merged = dict(city_search)
    merged["stations"] = list(stations_by_id.values())
    merged["station_count"] = len(merged["stations"])
    merged["locality_search"] = locality_search
    merged["raw_result_count"] = int(city_search.get("raw_result_count", 0)) + int(
        locality_search.get("raw_result_count", 0)
        if isinstance(locality_search, dict)
        else 0
    )
    return merged


def _result_template(location: str) -> dict[str, Any]:
    return {
        "requested_location": location,
        "api_available": False,
        "status_code": None,
        "available": False,
        "error": None,
        "source": SOURCE_LABEL,
        "aqi_label": AQI_SCALE_LABEL,
        "aqi_scale_url": AQI_SCALE_URL,
        "aqi": None,
        "aqi_source_value": None,
        "aqi_display": AQI_UNAVAILABLE,
        "station_name": None,
        "station_id": None,
        "attributions": [],
        "observed_at": None,
        "fetched_at": None,
        "cached": False,
        "age_minutes": None,
        "freshness": None,
        "pollutants": {
            label: {"value": None, "unit": UNIT_UNSPECIFIED}
            for label in POLLUTANT_FIELDS
        },
        "warnings": [],
    }


def _request_context() -> tuple[str | None, str | None]:
    token = _load_token()
    if token is None:
        return None, "WAQI_API_TOKEN is not configured in Streamlit secrets."
    return token, None


def load_city_stations(location: str, query: str | None = None) -> dict[str, Any]:
    """Search WAQI for a city or locality and retain verified city station IDs."""
    result: dict[str, Any] = {
        "requested_location": location,
        "query": query or location,
        "api_available": False,
        "status_code": None,
        "stations": [],
        "raw_result_count": 0,
        "city_match_count": 0,
        "station_count": 0,
        "error": None,
        "source": SOURCE_LABEL,
    }
    if location not in LOCATION_ALIASES:
        result["error"] = f"Recent air-quality lookup is not configured for {location}."
        return result

    token, error = _request_context()
    if token is None:
        result["error"] = error
        return result

    try:
        fetched = _fetch_json(
            SEARCH_URL,
            (("keyword", query.strip() if query and query.strip() else location),),
            token,
        )
    except WaqiRequestError as request_error:
        result["error"] = str(request_error)
        return result

    result["api_available"] = True
    result["status_code"] = fetched["status_code"]
    payload = fetched["payload"]
    data = payload.get("data")
    if not isinstance(data, list):
        result["error"] = "WAQI station search returned an unexpected response structure."
        return result

    result["raw_result_count"] = len(data)
    stations: dict[str, dict[str, Any]] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        station = entry.get("station")
        if not isinstance(station, dict):
            continue
        name = station.get("name")
        uid = entry.get("uid")
        if not isinstance(name, str) or not name.strip() or isinstance(uid, bool):
            continue
        try:
            station_id = int(uid)
        except (TypeError, ValueError, OverflowError):
            continue
        if station_id <= 0 or not _station_matches_location(name, location):
            continue
        result["city_match_count"] += 1
        if query and query.strip() and not _station_matches_search(
            name,
            query.strip(),
            location,
        ):
            continue
        stations[str(station_id)] = {
            "station_id": str(station_id),
            "station_name": name.strip(),
            "geo": station.get("geo") if isinstance(station.get("geo"), list) else [],
        }

    result["station_count"] = len(stations)
    result["stations"] = sorted(
        stations.values(),
        key=lambda item: (item["station_name"].casefold(), item["station_id"]),
    )
    if not result["stations"]:
        search_area = query.strip() if query and query.strip() else location
        result["error"] = f"WAQI returned no matching stations for {search_area} in {location}."
    return result


def _attributions(data: dict[str, Any]) -> list[dict[str, str | None]]:
    values = data.get("attributions")
    if not isinstance(values, list):
        return []
    return [
        {
            "name": item["name"].strip(),
            "url": item.get("url") if isinstance(item.get("url"), str) else None,
        }
        for item in values
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and item["name"].strip()
    ]


def _pollutant_readings(iaqi: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Extract known pollutants and only units explicitly included by WAQI."""
    readings: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    source_values = iaqi if isinstance(iaqi, dict) else {}

    for label, source_field in POLLUTANT_FIELDS.items():
        entry = source_values.get(source_field)
        if not isinstance(entry, dict):
            readings[label] = {"value": None, "unit": UNIT_UNSPECIFIED}
            warnings.append(f"{label} not provided by source")
            continue

        value = _numeric_value(entry.get("v"))
        if value is not None and value < 0:
            value = None
        if value is None:
            warnings.append(f"{label} value is missing, negative, or non-numeric")
        unit = entry.get("unit")
        unit = str(unit).strip() if isinstance(unit, str) and unit.strip() else UNIT_UNSPECIFIED
        readings[label] = {"value": value, "unit": unit}

    return readings, warnings


def load_station_observation(
    location: str,
    station_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch a selected WAQI station and preserve only its reported values."""
    result = _result_template(location)
    if location not in LOCATION_ALIASES:
        result["error"] = f"Recent air-quality lookup is not configured for {location}."
        return result
    if not str(station_id).isdigit() or int(station_id) <= 0:
        result["error"] = "The selected WAQI station identifier is invalid."
        return result

    token, error = _request_context()
    if token is None:
        result["error"] = error
        return result

    feed_url = f"{API_BASE_URL}/feed/@{station_id}/"
    try:
        fetched = _fetch_json(feed_url, (), token)
    except WaqiRequestError as request_error:
        result["error"] = str(request_error)
        return result

    result["api_available"] = True
    result["status_code"] = fetched["status_code"]
    result["fetched_at"] = fetched["fetched_at"]
    result["cached"] = (
        datetime.now(timezone.utc) - fetched["fetched_at"]
    ).total_seconds() > 0.1
    payload = fetched["payload"]
    data = payload.get("data")
    if not isinstance(data, dict):
        result["error"] = "WAQI station feed returned an unexpected response structure."
        return result

    city = data.get("city")
    station_name = city.get("name") if isinstance(city, dict) else None
    result["station_name"] = station_name if isinstance(station_name, str) else None
    actual_id = data.get("idx")
    result["station_id"] = str(actual_id) if actual_id is not None else str(station_id)
    if not isinstance(station_name, str) or not station_name.strip():
        result["error"] = "WAQI station feed did not identify its station name."
        return result
    if not _station_matches_location(station_name, location):
        result["error"] = (
            f"Selected station did not match {location}; WAQI identified it as "
            f"{station_name}. No AQI is shown for the requested location."
        )
        return result
    if str(result["station_id"]) != str(station_id):
        result["error"] = (
            f"WAQI returned station ID {result['station_id']} instead of the "
            f"selected station ID {station_id}."
        )
        return result

    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    observed_at = _parse_timestamp(data.get("time"))
    result["observed_at"] = observed_at
    if observed_at is None:
        result["warnings"].append("WAQI did not provide a usable observation timestamp.")
    else:
        freshness = _freshness(observed_at, now_utc)
        if freshness is None:
            result["error"] = "WAQI station observation timestamp is in the future."
            return result
        result["freshness"], result["age_minutes"] = freshness

    result["pollutants"], result["warnings"] = _pollutant_readings(data.get("iaqi"))
    if observed_at is None:
        result["warnings"].append("WAQI did not provide a usable observation timestamp.")

    aqi_source_value = data.get("aqi")
    result["aqi_source_value"] = aqi_source_value
    aqi = _numeric_value(aqi_source_value)
    if aqi is not None and aqi >= 0:
        result["aqi"] = aqi
        result["aqi_display"] = str(aqi_source_value).strip()
    else:
        result["warnings"].append("WAQI did not provide a valid station AQI value.")

    result["attributions"] = _attributions(data)
    result["available"] = True
    return result


def load_nearby_station_observation(
    latitude: Any,
    longitude: Any,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Use WAQI's geolocalized feed and verify station identity and distance."""
    result = _result_template("Nearby area")
    if (
        not _valid_coordinate(latitude, -90, 90)
        or not _valid_coordinate(longitude, -180, 180)
    ):
        result["error"] = "Browser location returned invalid coordinates."
        return result

    token, error = _request_context()
    if token is None:
        result["error"] = error
        return result

    feed_url = (
        f"{API_BASE_URL}/feed/geo:{float(latitude):.6f};"
        f"{float(longitude):.6f}/"
    )
    try:
        fetched = _request_json(feed_url, (), token)
    except WaqiRequestError as request_error:
        result["error"] = str(request_error)
        return result

    result["api_available"] = True
    result["status_code"] = fetched["status_code"]
    result["fetched_at"] = fetched["fetched_at"]
    payload = fetched["payload"]
    data = payload.get("data")
    if not isinstance(data, dict):
        result["error"] = "WAQI nearby lookup returned an unexpected response structure."
        return result

    city = data.get("city")
    station_name = city.get("name") if isinstance(city, dict) else None
    station_id = data.get("idx")
    try:
        parsed_station_id = int(station_id)
    except (TypeError, ValueError, OverflowError):
        parsed_station_id = 0
    station_coords = _station_coordinates(city.get("geo") if isinstance(city, dict) else None)
    if (
        not isinstance(station_name, str)
        or not station_name.strip()
        or isinstance(station_id, bool)
        or parsed_station_id <= 0
        or station_coords is None
    ):
        result["error"] = (
            "WAQI nearby lookup did not provide a verifiable station name, "
            "station ID, and coordinates."
        )
        return result

    distance = _distance_km(
        (float(latitude), float(longitude)),
        station_coords,
    )
    if not math.isfinite(distance) or distance > MAX_AUTOMATIC_STATION_DISTANCE_KM:
        result["error"] = (
            "WAQI's nearest station could not be verified within "
            f"{MAX_AUTOMATIC_STATION_DISTANCE_KM:g} km. Choose a city and station manually."
        )
        return result

    result["station_id"] = str(parsed_station_id)
    result["station_name"] = station_name.strip()
    result["distance_km"] = round(distance, 1)
    result["fetched_at"] = fetched["fetched_at"]
    observed_at = _parse_timestamp(data.get("time"))
    result["observed_at"] = observed_at
    if observed_at is None:
        result["warnings"].append("WAQI did not provide a usable observation timestamp.")
    else:
        now_utc = now or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        freshness = _freshness(observed_at, now_utc)
        if freshness is None:
            result["error"] = "WAQI station observation timestamp is in the future."
            return result
        result["freshness"], result["age_minutes"] = freshness

    result["pollutants"], result["warnings"] = _pollutant_readings(data.get("iaqi"))
    if observed_at is None:
        result["warnings"].append("WAQI did not provide a usable observation timestamp.")

    aqi_source_value = data.get("aqi")
    result["aqi_source_value"] = aqi_source_value
    aqi = _numeric_value(aqi_source_value)
    if aqi is not None and aqi >= 0:
        result["aqi"] = aqi
        result["aqi_display"] = str(aqi_source_value).strip()
    else:
        result["warnings"].append("WAQI did not provide a valid station AQI value.")

    result["attributions"] = _attributions(data)
    result["available"] = True
    return result
