"""CPCB/OGD recent-data adapter and persistent observation buffer."""

from __future__ import annotations

import json
import os
import socket
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_RESOURCE_URL = (
    "https://api.data.gov.in/resource/"
    "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
)
DEFAULT_BUFFER_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "live" / "recent_air_quality.csv"
)

POLLUTANT_FIELDS = ("aqi", "pm25", "pm10", "no2", "so2", "co", "o3")
REQUIRED_SOURCE_FIELDS = (
    "observation_time",
    "location_name",
    "station_name",
    *POLLUTANT_FIELDS,
)
BUFFER_COLUMNS = (
    "observation_time",
    "location_name",
    "station_name",
    "station_id",
    *POLLUTANT_FIELDS,
    "aqi_unit",
    "pm25_unit",
    "pm10_unit",
    "no2_unit",
    "so2_unit",
    "co_unit",
    "o3_unit",
    "source",
    "fetched_at",
)

LOCATION_ALIASES = {
    "delhi": "Delhi",
    "national capital territory of delhi": "Delhi",
    "nct of delhi": "Delhi",
    "faridabad": "Faridabad",
    "ghaziabad": "Ghaziabad",
    "gurgaon": "Gurgaon",
    "gurugram": "Gurgaon",
    "noida": "Noida",
}


class LiveDataError(RuntimeError):
    """Base error for live-source and buffer failures."""


class LiveDataConfigurationError(LiveDataError):
    """Raised when required source configuration is absent or invalid."""


class LiveDataSourceError(LiveDataError):
    """Raised when the remote source cannot return a usable response."""


class LiveDataSchemaError(LiveDataError):
    """Raised when the configured source schema does not match its response."""


@dataclass(frozen=True)
class CPCBConfig:
    """Explicit CPCB/OGD endpoint and schema configuration."""

    api_key: str
    resource_url: str = DEFAULT_RESOURCE_URL
    records_path: str = ""
    field_map: Mapping[str, str] = field(default_factory=dict)
    source_timezone: str = "Asia/Kolkata"
    source_units: Mapping[str, str] = field(default_factory=dict)
    target_units: Mapping[str, str] = field(default_factory=dict)
    request_params: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 15.0


def _setting(name: str, default: Any = None) -> Any:
    """Read a setting from the environment or Streamlit secrets."""
    value = os.getenv(name)
    if value is not None:
        return value

    try:
        import streamlit as st

        secrets = st.secrets
        nested = secrets.get("cpcb", {})
        if isinstance(nested, Mapping):
            secret_value = nested.get(name.removeprefix("CPCB_").lower())
            if secret_value is not None:
                return secret_value
        return secrets.get(name, default)
    except Exception:
        return default


def _json_setting(name: str, default: dict[str, Any]) -> dict[str, Any]:
    """Parse a JSON object setting without exposing secret values in errors."""
    value = _setting(name)
    if value is None or value == "":
        return default
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise LiveDataConfigurationError(f"{name} must contain a JSON object.") from exc
    if not isinstance(parsed, dict):
        raise LiveDataConfigurationError(f"{name} must contain a JSON object.")
    return parsed


def load_cpcb_config() -> CPCBConfig:
    """Load credentials and the explicitly configured API response schema."""
    api_key = _setting("CPCB_OGD_API_KEY") or _setting("CPCB_API_KEY")
    if not api_key:
        raise LiveDataConfigurationError(
            "Missing CPCB OGD API key. Set CPCB_OGD_API_KEY in the environment "
            "or Streamlit secrets."
        )

    records_path = str(_setting("CPCB_RECORDS_PATH", "")).strip()
    field_map = _json_setting("CPCB_FIELD_MAP", {})
    missing_map = [name for name in REQUIRED_SOURCE_FIELDS if not field_map.get(name)]
    if not records_path:
        raise LiveDataConfigurationError(
            "CPCB_RECORDS_PATH is required; configure the response path after "
            "confirming the OGD response schema. Use '$' for a top-level record list."
        )
    if missing_map:
        raise LiveDataConfigurationError(
            "CPCB_FIELD_MAP must map these canonical fields to source field names: "
            f"{missing_map}. Confirm the OGD schema before configuring it."
        )

    try:
        timeout_seconds = float(_setting("CPCB_TIMEOUT_SECONDS", 15))
    except (TypeError, ValueError) as exc:
        raise LiveDataConfigurationError("CPCB_TIMEOUT_SECONDS must be numeric.") from exc
    if timeout_seconds <= 0:
        raise LiveDataConfigurationError("CPCB_TIMEOUT_SECONDS must be greater than zero.")

    request_params = _json_setting("CPCB_REQUEST_PARAMS", {})
    return CPCBConfig(
        api_key=str(api_key),
        resource_url=str(_setting("CPCB_API_URL", DEFAULT_RESOURCE_URL)),
        records_path=records_path,
        field_map={str(key): str(value) for key, value in field_map.items()},
        source_timezone=str(_setting("CPCB_SOURCE_TIMEZONE", "Asia/Kolkata")),
        source_units=_json_setting("CPCB_SOURCE_UNITS", {}),
        target_units=_json_setting("CPCB_TARGET_UNITS", {}),
        request_params={str(key): str(value) for key, value in request_params.items()},
        timeout_seconds=timeout_seconds,
    )


def fetch_cpcb_response(config: CPCBConfig | None = None) -> Any:
    """Fetch raw JSON from the configured OGD endpoint; no schema is presumed."""
    config = config or load_cpcb_config()
    query = {"api-key": config.api_key, "format": "json", "limit": "1000"}
    query.update(config.request_params)
    parts = urlsplit(config.resource_url)
    merged_query = dict(parse_qsl(parts.query, keep_blank_values=True))
    merged_query.update(query)
    url = urlunsplit(parts._replace(query=urlencode(merged_query)))
    request = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        raise LiveDataSourceError(
            f"OGD request failed with HTTP {exc.code}; check endpoint, key, and access."
        ) from exc
    except (URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise LiveDataSourceError(
            "CPCB/OGD is unavailable or the request timed out. No live observations were loaded."
        ) from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise LiveDataSourceError("CPCB/OGD returned a non-JSON or malformed response.") from exc


def normalize_timestamps(
    values: pd.Series,
    source_timezone: str = "Asia/Kolkata",
) -> pd.Series:
    """Convert timestamps to UTC; naive source timestamps use the configured zone."""
    normalized = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                normalized.append(pd.NaT)
                continue
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(
                    source_timezone,
                    ambiguous="NaT",
                    nonexistent="NaT",
                )
            normalized.append(timestamp.tz_convert("UTC"))
        except (TypeError, ValueError, KeyError, OverflowError):
            normalized.append(pd.NaT)
    return pd.Series(pd.to_datetime(normalized, errors="coerce", utc=True), index=values.index)


def normalize_location_name(value: Any) -> str:
    """Map known CPCB city labels to project locations; reject unknown cities."""
    if value is None or pd.isna(value):
        raise LiveDataSchemaError("A record is missing its configured city/location value.")
    normalized = " ".join(str(value).strip().lower().replace("-", " ").split())
    location = LOCATION_ALIASES.get(normalized)
    if location is None:
        raise LiveDataSchemaError(
            f"Unknown CPCB city/location {value!r}; no project location mapping exists."
        )
    return location


def normalize_station_location_names(observations: pd.DataFrame) -> pd.DataFrame:
    """Normalize project city labels while preserving source station names verbatim."""
    result = observations.copy()
    validate_required_columns(result, ("location_name", "station_name"))
    result["location_name"] = result["location_name"].map(normalize_location_name)
    result["station_name"] = result["station_name"].astype("string").str.strip()
    missing_station = result["station_name"].isna() | result["station_name"].eq("")
    if missing_station.any():
        raise LiveDataSchemaError(
            f"{int(missing_station.sum())} record(s) have an empty station name."
        )
    if "station_id" not in result.columns:
        result["station_id"] = ""
    result["station_id"] = result["station_id"].fillna("").astype(str).str.strip()
    return result


def validate_required_columns(
    observations: pd.DataFrame,
    required_columns: tuple[str, ...] = BUFFER_COLUMNS,
) -> None:
    """Fail with a clear schema error when required columns are absent."""
    missing = [column for column in required_columns if column not in observations.columns]
    if missing:
        raise LiveDataSchemaError(f"Missing required columns: {missing}.")


def validate_pollutant_fields(observations: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with numeric AQI and pollutant values; never impute readings."""
    validate_required_columns(observations, POLLUTANT_FIELDS)
    result = observations.copy()
    for column in POLLUTANT_FIELDS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result.loc[~result[column].map(lambda value: pd.notna(value) and abs(value) != float("inf")), column] = pd.NA

    incomplete = result[list(POLLUTANT_FIELDS)].isna().any(axis=1)
    if incomplete.any():
        warnings.warn(
            f"Skipped {int(incomplete.sum())} observation(s) with missing or invalid "
            "AQI/pollutant values; no values were filled in.",
            RuntimeWarning,
            stacklevel=2,
        )
        result = result.loc[~incomplete].copy()
    return result


def _canonical_unit(unit: Any) -> str:
    """Normalize supported mass-concentration unit spellings."""
    text = str(unit).strip().lower()
    text = text.replace("\u00b5", "u").replace("\u03bc", "u").replace("\u00b3", "3")
    text = text.replace(" ", "").replace("^", "")
    aliases = {
        "ug/m3": "ug/m3",
        "ugm-3": "ug/m3",
        "mg/m3": "mg/m3",
        "mgm-3": "mg/m3",
        "index": "index",
    }
    if text not in aliases:
        raise LiveDataConfigurationError(
            f"Unsupported configured unit {unit!r}; no conversion will be guessed."
        )
    return aliases[text]


def convert_pollutant_units(
    observations: pd.DataFrame,
    source_units: Mapping[str, str],
    target_units: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Convert only explicitly configured units; support mg/m3 and ug/m3 scaling."""
    result = observations.copy()
    targets = target_units or {}
    for field_name in POLLUTANT_FIELDS:
        source_label = source_units.get(field_name)
        result[f"{field_name}_unit"] = str(source_label) if source_label else "unknown"
        if field_name not in targets:
            continue
        if not source_label:
            raise LiveDataConfigurationError(
                f"Cannot convert {field_name}: its source unit is not explicitly configured."
            )
        source_unit = _canonical_unit(source_label)
        target_unit = _canonical_unit(targets[field_name])
        if field_name == "aqi" and (source_unit != "index" or target_unit != "index"):
            raise LiveDataConfigurationError("AQI is an index and cannot use concentration units.")
        if source_unit == target_unit:
            continue
        if field_name == "aqi":
            raise LiveDataConfigurationError("AQI index units cannot be converted.")
        if {source_unit, target_unit} == {"mg/m3", "ug/m3"}:
            factor = 1000.0 if source_unit == "mg/m3" else 0.001
            result[field_name] = pd.to_numeric(result[field_name], errors="coerce") * factor
            result[f"{field_name}_unit"] = target_unit
            continue
        raise LiveDataConfigurationError(
            f"No supported conversion from {source_unit} to {target_unit} for {field_name}."
        )
    return result


def _extract_records(payload: Any, records_path: str) -> list[dict[str, Any]]:
    """Read records only from the explicitly configured JSON path."""
    if not records_path:
        raise LiveDataConfigurationError("A CPCB_RECORDS_PATH must be configured explicitly.")
    records: Any = payload
    if records_path != "$":
        for part in records_path.split("."):
            if not isinstance(records, Mapping) or part not in records:
                raise LiveDataSchemaError(
                    f"Configured records path {records_path!r} was not present in the response."
                )
            records = records[part]
    if not isinstance(records, list):
        raise LiveDataSchemaError(
            f"Configured records path {records_path!r} must point to a list of records."
        )
    if not records:
        raise LiveDataSourceError("CPCB/OGD returned an empty observation list.")
    if not all(isinstance(record, Mapping) for record in records):
        raise LiveDataSchemaError("The configured record list contains non-object entries.")
    return [dict(record) for record in records]


def _deduplicate_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Remove identical duplicate keys and reject conflicting measurements."""
    key_columns = ["observation_time", "location_name", "station_name", "station_id"]
    value_columns = list(POLLUTANT_FIELDS)
    duplicates = observations.loc[observations.duplicated(key_columns, keep=False)]
    for _, group in duplicates.groupby(key_columns, dropna=False, sort=False):
        if len(group[value_columns].drop_duplicates()) > 1:
            raise LiveDataSchemaError(
                "Conflicting readings share the same timestamp, location, and station; "
                "the buffer was not updated."
            )
    return observations.drop_duplicates(key_columns, keep="last").reset_index(drop=True)


def prepare_recent_observations(
    payload: Any,
    config: CPCBConfig,
) -> pd.DataFrame:
    """Map validated wide-format source records into normalized buffer rows."""
    missing_map = [name for name in REQUIRED_SOURCE_FIELDS if not config.field_map.get(name)]
    if missing_map:
        raise LiveDataConfigurationError(f"CPCB field mapping is incomplete: {missing_map}.")

    records = _extract_records(payload, config.records_path)
    raw = pd.DataFrame.from_records(records)
    source_columns = {config.field_map[name] for name in REQUIRED_SOURCE_FIELDS}
    missing_columns = sorted(source_columns.difference(raw.columns))
    if missing_columns:
        raise LiveDataSchemaError(
            f"CPCB response is missing configured source fields: {missing_columns}. "
            "Confirm the OGD schema and update CPCB_FIELD_MAP."
        )

    rename_map = {source_name: canonical for canonical, source_name in config.field_map.items()}
    result = raw.rename(columns=rename_map).copy()
    if "station_id" not in result.columns:
        result["station_id"] = ""
    result["observation_time"] = normalize_timestamps(
        result["observation_time"], config.source_timezone
    )
    bad_timestamps = result["observation_time"].isna()
    if bad_timestamps.any():
        warnings.warn(
            f"Skipped {int(bad_timestamps.sum())} observation(s) with malformed timestamps.",
            RuntimeWarning,
            stacklevel=2,
        )
        result = result.loc[~bad_timestamps].copy()
    if result.empty:
        raise LiveDataSourceError("No records had a valid timestamp; no observations were saved.")

    result = normalize_station_location_names(result)
    result = convert_pollutant_units(result, config.source_units, config.target_units)
    result = validate_pollutant_fields(result)
    if result.empty:
        raise LiveDataSourceError("No records contained all required numeric AQI/pollutant values.")

    result["source"] = "CPCB/OGD"
    result["fetched_at"] = pd.Timestamp.now(tz="UTC")
    result = result.reindex(columns=BUFFER_COLUMNS)
    return _deduplicate_observations(result)


def prepare_cpcb_recent_data(config: CPCBConfig | None = None) -> pd.DataFrame:
    """Fetch, schema-check, and normalize one CPCB/OGD response."""
    config = config or load_cpcb_config()
    return prepare_recent_observations(fetch_cpcb_response(config), config)


def _empty_buffer() -> pd.DataFrame:
    """Return an empty buffer with the canonical schema."""
    return pd.DataFrame(columns=BUFFER_COLUMNS)


def load_recent_data_buffer(
    path: str | Path = DEFAULT_BUFFER_PATH,
) -> pd.DataFrame:
    """Load and validate the persistent CSV buffer, returning empty if absent."""
    buffer_path = Path(path)
    if not buffer_path.exists():
        return _empty_buffer()
    try:
        result = pd.read_csv(buffer_path)
    except pd.errors.EmptyDataError:
        return _empty_buffer()
    validate_required_columns(result)
    if result.empty:
        return _empty_buffer()

    result["observation_time"] = pd.to_datetime(
        result["observation_time"], errors="coerce", utc=True
    )
    result["fetched_at"] = pd.to_datetime(result["fetched_at"], errors="coerce", utc=True)
    if result[["observation_time", "fetched_at"]].isna().any().any():
        raise LiveDataSchemaError("The saved buffer contains malformed timestamps.")
    result = normalize_station_location_names(result)
    result = validate_pollutant_fields(result)
    if result.empty:
        return _empty_buffer()
    return _deduplicate_observations(result.reindex(columns=BUFFER_COLUMNS))


def save_recent_data_buffer(
    observations: pd.DataFrame,
    path: str | Path = DEFAULT_BUFFER_PATH,
) -> pd.DataFrame:
    """Merge validated observations into the persistent CSV without inventing values."""
    validate_required_columns(observations)
    incoming = observations.reindex(columns=BUFFER_COLUMNS).copy()
    incoming["observation_time"] = pd.to_datetime(
        incoming["observation_time"], errors="coerce", utc=True
    )
    incoming["fetched_at"] = pd.to_datetime(incoming["fetched_at"], errors="coerce", utc=True)
    if incoming[["observation_time", "fetched_at"]].isna().any().any():
        raise LiveDataSchemaError("Cannot save observations with malformed timestamps.")
    incoming = normalize_station_location_names(incoming)
    incoming = validate_pollutant_fields(incoming)
    if incoming.empty:
        raise LiveDataSourceError("No complete observations are available to save.")

    buffer_path = Path(path)
    existing = load_recent_data_buffer(buffer_path)
    combined = _deduplicate_observations(pd.concat([existing, incoming], ignore_index=True))
    combined = combined.sort_values(
        ["location_name", "station_name", "observation_time"]
    ).reset_index(drop=True)

    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = buffer_path.with_suffix(buffer_path.suffix + ".tmp")
    try:
        combined.to_csv(temporary_path, index=False)
        os.replace(temporary_path, buffer_path)
    except OSError as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise LiveDataError(f"Could not persist the recent-data buffer: {exc}") from exc
    return combined