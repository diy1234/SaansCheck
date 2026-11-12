"""Standalone CPCB-style National AQI calculation from averaged concentrations.

Official references
-------------------
Central Pollution Control Board (CPCB), *National Air Quality Index*,
report CUPS/82/2014-15:
https://cpcb.gov.in/displaypdf.php?id=bmF0aW9uYWwtYWlyLXF1YWxpdHktaW5kZXgvRklOQUwtUkVQT1JUX0FRSV8ucGRm

CPCB, *How is AQI calculated?*:
https://cpcb.gov.in/displaypdf.php?id=bmF0aW9uYWwtYWlyLXF1YWxpdHktaW5kZXgvSG93X0FRSV9DYWxjdWxhdGVkLnBkZg==

Method notes
------------
* Supply concentrations that have already been averaged using the CPCB
  averaging interval: 24 hours for PM2.5, PM10, NO2 and SO2; 8 hours for CO
  and O3. This module does not aggregate timestamped observations.
* CPCB requires at least 16 hours of data for an individual pollutant
  sub-index, and at least three pollutant sub-indices, including PM2.5 or
  PM10, for an overall AQI.
* Concentration breakpoints below are in the units printed by CPCB. They
  describe successive ranges, with each upper limit included in that range:
  PM2.5: 30, 60, 90, 120, 250 µg/m³;
  PM10: 50, 100, 250, 350, 430 µg/m³;
  NO2: 40, 80, 180, 280, 400 µg/m³;
  SO2: 40, 80, 380, 800, 1600 µg/m³;
  CO: 1, 2, 10, 17, 34 mg/m³;
  O3: 50, 100, 168, 208 µg/m³ for unstarred 8-hour bands.
  The corresponding AQI interval endpoints are 0, 50, 100, 200, 300, 400.
* A sub-index is linearly interpolated within the applicable concentration
  interval using CPCB's formula:
  Ip = ((Ihi - Ilo) / (Bhi - Blo)) * (Cp - Blo) + Ilo.
  CPCB's note says to subtract one from Ilo when Ilo is greater than 50.
  The cited CPCB documents do not specify nearest-integer rounding, so this
  implementation preserves the interpolated sub-index as a float. For
  category assignment, the published integer upper cutoffs are applied as
  inclusive numeric cutoffs; CPCB does not state a separate rule for
  fractional values around category boundaries.
* Overall AQI is the maximum of the available valid pollutant sub-indices.
  CPCB's concentration tables classify concentrations above the final
  breakpoint as Severe, but the cited calculation note does not specify
  whether to cap the numeric sub-index at 500 or extrapolate. To avoid
  inventing a rule, this module marks such a sub-index unresolved and does
  not report an overall numeric AQI if any provided concentration is beyond
  its applicable table breakpoint.
* CPCB's O3 table marks the terminal 748 and 748+ breakpoints as 1-hour
  monitoring values. Since the calculation note requires 8-hour O3, this
  implementation does not apply those starred cutoffs to an 8-hour input.
  O3 concentrations above 208 µg/m³ therefore have an unresolved sub-index.
  * CPCB's O3 table marks the terminal 748 and 748+ breakpoints as 1-hour
  monitoring values. Since the calculation note requires 8-hour O3, this
  implementation does not apply those starred cutoffs to an 8-hour input.
  O3 concentrations above 208 µg/m³ therefore have an unresolved sub-index.
* AQI categories: Good 0-50; Satisfactory 51-100; Moderate 101-200;
  Poor 201-300; Very Poor 301-400; Severe 401-500. A numeric value above
  500 has no category defined by this table.

This is a standalone implementation of the CPCB interpolation/breakpoint
method. It is not connected to Streamlit and does not change SaansCheck's
existing dataset, AQI values, models, or forecasting pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class PollutantReading:
    """A concentration already averaged for the CPCB pollutant time window.

    ``valid_hours`` is the number of valid hourly observations available in
    the reporting period. CPCB requires at least 16 for an individual
    sub-index. The value itself must already be the CPCB-averaged
    concentration; this class does not calculate a time average.
    """

    value: Optional[float]
    unit: Optional[str]
    averaging_period_hours: Optional[int]
    valid_hours: Optional[int]


@dataclass(frozen=True)
class PollutantSubIndex:
    """Validation and calculation result for one pollutant."""

    concentration: Optional[float]
    unit: Optional[str]
    sub_index: Optional[float]
    category: Optional[str]
    status: str
    message: Optional[str] = None


@dataclass(frozen=True)
class AQIResult:
    """Structured overall AQI result and pollutant-level diagnostics."""

    overall_aqi: Optional[float]
    category: Optional[str]
    pollutant_sub_indices: Mapping[str, PollutantSubIndex]
    dominant_pollutant: Optional[str]
    status: str
    validation_messages: Tuple[str, ...]


# (canonical unit, required averaging duration in hours, concentration upper
# breakpoints). The severe concentration range above the last listed value is
# intentionally not interpolated because the cited CPCB calculation note does
# not define a numeric cap or extrapolation rule for that range.
_POLLUTANT_SPEC = {
    "PM2.5": ("µg/m³", 24, (30.0, 60.0, 90.0, 120.0, 250.0)),
    "PM10": ("µg/m³", 24, (50.0, 100.0, 250.0, 350.0, 430.0)),
    "NO2": ("µg/m³", 24, (40.0, 80.0, 180.0, 280.0, 400.0)),
    "SO2": ("µg/m³", 24, (40.0, 80.0, 380.0, 800.0, 1600.0)),
    "CO": ("mg/m³", 8, (1.0, 2.0, 10.0, 17.0, 34.0)),
    "O3": ("µg/m³", 8, (50.0, 100.0, 168.0, 208.0)),
}
_AQI_BREAKPOINTS = (0, 50, 100, 200, 300, 400)
_PARTICULATES = frozenset(("PM2.5", "PM10"))
_CATEGORY_BREAKPOINTS = (
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (500, "Severe"),
)
_POLLUTANT_ALIASES = {
    "PM25": "PM2.5",
    "PM2.5": "PM2.5",
    "PM10": "PM10",
    "NO2": "NO2",
    "SO2": "SO2",
    "CO": "CO",
    "O3": "O3",
}


def _normalize_unit(unit: str) -> str:
    """Normalize common spellings without converting between units."""
    normalized = unit.strip().lower()
    normalized = normalized.replace("μ", "µ").replace("³", "3")
    normalized = normalized.replace(" ", "")
    return normalized


def _canonical_pollutant(name: str) -> Optional[str]:
    """Return a supported pollutant key, accepting case-insensitive names."""
    return _POLLUTANT_ALIASES.get(name.strip().upper())


def _category_for_aqi(value: float) -> Optional[str]:
    """Map an in-range numeric AQI to its CPCB category."""
    for upper, category in _CATEGORY_BREAKPOINTS:
        if value <= upper:
            return category
    return None


def _calculate_sub_index(
    concentration: float,
    concentration_breakpoints: Tuple[float, ...],
) -> Optional[float]:
    """Interpolate a CPCB sub-index within known concentration bands."""
    if concentration_breakpoints[-1] < concentration:
        return None

    concentration_lower = 0.0
    for band_index, concentration_upper in enumerate(concentration_breakpoints):
        if concentration <= concentration_upper:
            index_lower = _AQI_BREAKPOINTS[band_index]
            index_upper = _AQI_BREAKPOINTS[band_index + 1]
            if index_lower > 50:
                index_lower -= 1
            interpolated = (
                (index_upper - index_lower)
                / (concentration_upper - concentration_lower)
                * (concentration - concentration_lower)
                + index_lower
            )
            return interpolated
        concentration_lower = concentration_upper
    return None


def _invalid_reading(
    reading: Optional[PollutantReading],
    status: str,
    message: str,
) -> PollutantSubIndex:
    """Create a non-calculated pollutant result while preserving input facts."""
    return PollutantSubIndex(
        concentration=reading.value if reading is not None else None,
        unit=reading.unit if reading is not None else None,
        sub_index=None,
        category=None,
        status=status,
        message=message,
    )


def calculate_aqi(
    readings: Mapping[str, Optional[PollutantReading]],
) -> AQIResult:
    """Calculate a CPCB-style AQI from available, validated pollutant readings.

    Args:
        readings: Mapping of pollutant names to already-averaged readings.
            Supported pollutant names are PM2.5, PM10, NO2, SO2, CO and O3.
            Each non-missing reading must specify its value, explicit unit,
            averaging duration (24 hours for particulate matter, NO2 and SO2;
            8 hours for CO and O3), and available valid hourly data count.
            Unknown pollutant names are reported as validation warnings.

    Returns:
        An :class:`AQIResult` with one diagnostic result for each supported
        pollutant, a dominant pollutant when the numeric AQI is calculable,
        and explicit overall validation status. Missing or invalid pollutant
        values are never imputed.

    Notes:
        Overall AQI requires at least three valid pollutant sub-indices,
        including PM2.5 or PM10. Values above a terminal breakpoint whose
        numeric calculation is not specified by the cited CPCB documents are
        reported as unresolved rather than being capped or extrapolated.
    """
    canonical_readings = {}
    duplicate_pollutants = set()
    validation_messages = []

    for input_name, reading in readings.items():
        canonical_name = _canonical_pollutant(input_name)
        if canonical_name is None:
            validation_messages.append(f"Unsupported pollutant name: {input_name!r}.")
            continue
        if canonical_name in canonical_readings:
            duplicate_pollutants.add(canonical_name)
            validation_messages.append(
                f"Duplicate readings supplied for {canonical_name}; "
                "the pollutant is excluded because the input is ambiguous."
            )
            continue
        canonical_readings[canonical_name] = reading

    results = {}
    for pollutant, (expected_unit, expected_period, breakpoints) in _POLLUTANT_SPEC.items():
        if pollutant in duplicate_pollutants:
            results[pollutant] = _invalid_reading(
                None,
                "duplicate_readings",
                "Multiple readings were supplied; no value was selected.",
            )
            continue
        if pollutant not in canonical_readings or canonical_readings[pollutant] is None:
            results[pollutant] = _invalid_reading(
                None, "missing", "No pollutant reading was supplied."
            )
            continue

        reading = canonical_readings[pollutant]
        if not isinstance(reading, PollutantReading):
            results[pollutant] = _invalid_reading(
                None, "invalid_reading", "Reading must be a PollutantReading instance."
            )
            validation_messages.append(f"{pollutant}: invalid reading object.")
            continue

        value = reading.value
        if value is None:
            results[pollutant] = _invalid_reading(
                reading, "missing_value", "Concentration value is missing."
            )
            validation_messages.append(f"{pollutant}: concentration value is missing.")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            results[pollutant] = _invalid_reading(
                reading, "invalid_value", "Concentration must be a finite number."
            )
            validation_messages.append(f"{pollutant}: concentration is not finite numeric data.")
            continue
        value = float(value)
        if value < 0:
            results[pollutant] = _invalid_reading(
                reading, "invalid_value", "Negative concentrations are not valid."
            )
            validation_messages.append(f"{pollutant}: negative concentration is invalid.")
            continue

        if not isinstance(reading.unit, str) or not reading.unit.strip():
            results[pollutant] = _invalid_reading(
                reading, "unknown_unit", "An explicit supported unit is required."
            )
            validation_messages.append(f"{pollutant}: unit is missing or unknown.")
            continue
        normalized_unit = _normalize_unit(reading.unit)
        supported_units = {
            "µg/m3": "µg/m³",
            "ug/m3": "µg/m³",
            "mg/m3": "mg/m³",
        }
        if supported_units.get(normalized_unit) != expected_unit:
            results[pollutant] = _invalid_reading(
                reading,
                "unknown_or_incompatible_unit",
                f"Expected {expected_unit}; received {reading.unit!r}.",
            )
            validation_messages.append(
                f"{pollutant}: unit {reading.unit!r} is not the required {expected_unit}."
            )
            continue

        if (
            isinstance(reading.averaging_period_hours, bool)
            or not isinstance(reading.averaging_period_hours, int)
            or reading.averaging_period_hours != expected_period
        ):
            results[pollutant] = _invalid_reading(
                reading,
                "invalid_averaging_period",
                f"Expected a {expected_period}-hour averaged concentration.",
            )
            validation_messages.append(
                f"{pollutant}: expected {expected_period}-hour averaging."
            )
            continue
        if (
            isinstance(reading.valid_hours, bool)
            or not isinstance(reading.valid_hours, int)
            or reading.valid_hours < 16
            or reading.valid_hours > 24
        ):
            results[pollutant] = _invalid_reading(
                reading,
                "insufficient_or_invalid_data",
                "CPCB requires at least 16 valid hourly observations; valid_hours must be 16-24.",
            )
            validation_messages.append(
                f"{pollutant}: valid hourly observations must be between 16 and 24."
            )
            continue

        sub_index = _calculate_sub_index(value, breakpoints)
        if pollutant == "O3" and sub_index is None:
            results[pollutant] = PollutantSubIndex(
                concentration=value,
                unit=expected_unit,
                sub_index=None,
                category=None,
                status="averaging_breakpoint_mismatch",
                message=(
                    "CPCB Table 3.8 marks O3's terminal 748/748+ cutoffs as "
                    "1-hour values, while the calculation note requires 8-hour "
                    "O3. No higher 8-hour breakpoint is established here."
                ),
            )
            validation_messages.append(
                "O3: above 208 µg/m³, CPCB's cited table only provides starred "
                "1-hour terminal cutoffs; 8-hour sub-index unresolved."
            )
            continue
        if sub_index is None:
            results[pollutant] = PollutantSubIndex(
                concentration=value,
                unit=expected_unit,
                sub_index=None,
                category="Severe",
                status="severe_numeric_rule_unspecified",
                message=(
                    "Concentration exceeds the final CPCB breakpoint. The source "
                    "classifies this range as Severe but does not specify a numeric "
                    "cap or extrapolation rule."
                ),
            )
            validation_messages.append(
                f"{pollutant}: above the final breakpoint; numeric sub-index unresolved."
            )
            continue

        results[pollutant] = PollutantSubIndex(
            concentration=value,
            unit=expected_unit,
            sub_index=sub_index,
            category=_category_for_aqi(sub_index),
            status="valid",
        )

    valid = {
        pollutant: result.sub_index
        for pollutant, result in results.items()
        if result.sub_index is not None
    }
    unresolved = [
        pollutant
        for pollutant, result in results.items()
        if result.status in {
            "severe_numeric_rule_unspecified",
            "averaging_breakpoint_mismatch",
        }
    ]
    available_pollutants = set(valid) | set(unresolved)
    particulate_available = bool(available_pollutants & _PARTICULATES)

    if len(available_pollutants) < 3 or not particulate_available:
        overall_aqi = None
        category = None
        dominant_pollutant = None
        status = "insufficient_pollutants"
        if len(available_pollutants) < 3:
            validation_messages.append(
                "At least 3 valid pollutant sub-indices are required; "
                f"{len(available_pollutants)} eligible pollutant readings available."
            )
        if not particulate_available:
            validation_messages.append(
                "At least one valid PM2.5 or PM10 sub-index is required."
            )
    elif unresolved:
        overall_aqi = None
        unresolved_statuses = {
            results[pollutant].status for pollutant in unresolved
        }
        category = (
            "Severe"
            if unresolved_statuses == {"severe_numeric_rule_unspecified"}
            else None
        )
        dominant_pollutant = (
            unresolved[0]
            if len(unresolved) == 1
            and results[unresolved[0]].status == "severe_numeric_rule_unspecified"
            else None
        )
        status = "unresolved_numeric_sub_index"
        validation_messages.append(
            "Overall AQI is withheld because CPCB's numeric rule above the "
            "final concentration breakpoint is unspecified."
        )
    else:
        dominant_pollutant = max(valid, key=valid.__getitem__)
        overall_aqi = valid[dominant_pollutant]
        category = _category_for_aqi(overall_aqi)
        has_warnings = bool(validation_messages) or any(
            result.status != "valid" for result in results.values()
        )
        status = "valid_with_warnings" if has_warnings else "valid"

    return AQIResult(
        overall_aqi=overall_aqi,
        category=category,
        pollutant_sub_indices=results,
        dominant_pollutant=dominant_pollutant,
        status=status,
        validation_messages=tuple(validation_messages),
    )
