"""Research-only validation of supplied AQI against CPCB-style calculations.

This module does not modify source or processed datasets, production code, or
forecasting artifacts. It writes a separate location-day comparison, report,
and visualization under ``data/processed``.

The raw CSV does not declare pollutant units. This experiment uses the unit
convention displayed by the existing application (µg/m³ except CO in mg/m³);
all conclusions are conditional on that unverified source-unit assumption.

Daily comparison protocol:
* The supplied AQI is the arithmetic mean of the available hourly AQI values
  for each location and complete calendar day, matching ``src/preprocess.py``.
* PM2.5, PM10, NO2, and SO2 use their calendar-day arithmetic means.
* CO and O3 use the maximum complete trailing 8-hour arithmetic mean ending
  during that calendar day. Windows require eight consecutive hourly samples.
* A daily pollutant average requires at least 16 valid non-negative hourly
  observations, following the calculator's CPCB data-count validation.
* The calculator receives each pollutant's daily valid-hour count. This
  satisfies its 16-hour reporting-period check; it does not describe the
  number of hourly samples in an 8-hour window.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.aqi_calculator import PollutantReading, calculate_aqi


RAW_PATH = Path("data/raw/delhi_ncr_air_quality.csv.csv")
RESULTS_PATH = Path("data/processed/aqi_validation_results.csv")
REPORT_PATH = Path("data/processed/aqi_validation_report.txt")
FIGURE_PATH = Path("data/processed/aqi_validation_comparison.png")

POLLUTANT_COLUMNS = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co",
    "O3": "o3",
}
UNITS = {
    "PM2.5": "µg/m³",
    "PM10": "µg/m³",
    "NO2": "µg/m³",
    "SO2": "µg/m³",
    "CO": "mg/m³",
    "O3": "µg/m³",
}
PERIODS = {
    "PM2.5": 24,
    "PM10": 24,
    "NO2": 24,
    "SO2": 24,
    "CO": 8,
    "O3": 8,
}
AQI_CATEGORIES = (
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (500, "Severe"),
)
SEASONS = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Pre-monsoon",
    4: "Pre-monsoon",
    5: "Pre-monsoon",
    6: "Monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Post-monsoon",
    11: "Post-monsoon",
}


def _category(value: object) -> Optional[str]:
    """Return the CPCB category for a finite AQI value."""
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        return None
    for upper, label in AQI_CATEGORIES:
        if numeric <= upper:
            return label
    return None


def _pollutant_reason(
    pollutant: str,
    count: int,
    valid_windows: int,
    negative_count: int,
    missing_count: int,
) -> Optional[str]:
    """Explain why a pollutant could not provide its required daily value."""
    notes = []
    if negative_count:
        notes.append(f"{negative_count} negative hourly value(s) excluded")
    if missing_count:
        notes.append(f"{missing_count} missing/invalid hourly value(s)")
    if count < 16:
        notes.append(f"only {count} valid hourly value(s); at least 16 required")
    if PERIODS[pollutant] == 8 and valid_windows == 0:
        notes.append("no complete consecutive 8-hour window ending in the day")
    return f"{pollutant}: " + "; ".join(notes) if notes else None


def _season_for_month(month: int) -> str:
    return SEASONS[month]


def build_daily_comparisons(raw: pd.DataFrame) -> pd.DataFrame:
    """Build location-day CPCB-style comparisons and explicit exclusions.

    ``raw`` must contain hourly timestamp, location, source AQI, and six
    pollutant concentration columns. Incomplete location-days are retained
    with an exclusion status and no calculated AQI.
    """
    required = {"timestamp", "location_name", "aqi", *POLLUTANT_COLUMNS.values()}
    missing_columns = sorted(required - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Raw data is missing required columns: {missing_columns}")

    data = raw.loc[:, sorted(required)].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    if data["location_name"].isna().any():
        raise ValueError("Raw data contains missing location_name values.")
    if data.duplicated(["location_name", "timestamp"]).any():
        raise ValueError("Duplicate location/timestamp observations are ambiguous.")

    numeric_columns = ["aqi", *POLLUTANT_COLUMNS.values()]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    rows: List[dict] = []
    for location, location_data in data.groupby("location_name", sort=True):
        location_data = location_data.sort_values("timestamp").set_index("timestamp")
        observed_timestamps = location_data.index
        expected_hours = pd.date_range(
            location_data.index.min().floor("h"),
            location_data.index.max().floor("h"),
            freq="h",
        )
        hourly = location_data.reindex(expected_hours)
        hourly["location_name"] = location
        hourly.index.name = "timestamp"

        pollutant_clean = {}
        pollutant_rolling_8h = {}
        for pollutant, column in POLLUTANT_COLUMNS.items():
            values = hourly[column]
            clean_values = values.where(values >= 0)
            pollutant_clean[pollutant] = clean_values
            if PERIODS[pollutant] == 8:
                pollutant_rolling_8h[pollutant] = clean_values.rolling(
                    window=8, min_periods=8
                ).mean()

        first_day = hourly.index.min().normalize()
        last_day = hourly.index.max().normalize()
        for day in pd.date_range(first_day, last_day, freq="D"):
            day_end = day + pd.Timedelta(hours=23)
            day_hours = hourly.loc[day:day_end]
            expected_day_hours = pd.date_range(day, day_end, freq="h")
            complete_day = expected_day_hours.isin(observed_timestamps).all()
            original_values = day_hours["aqi"].dropna()
            original_aqi = (
                float(original_values.mean())
                if complete_day and len(original_values) >= 16
                else None
            )

            readings: Dict[str, PollutantReading] = {}
            pollutant_reasons: List[str] = []
            daily_concentrations: Dict[str, Optional[float]] = {}
            daily_valid_hours: Dict[str, int] = {}
            daily_window_counts: Dict[str, int] = {}
            daily_pollutant_notes: List[str] = []

            for pollutant, column in POLLUTANT_COLUMNS.items():
                values = day_hours.index.to_series().map(
                    pollutant_clean[pollutant]
                )
                negative_count = int((day_hours[column] < 0).sum())
                valid_count = int(values.notna().sum())
                daily_valid_hours[pollutant] = valid_count
                daily_window_counts[pollutant] = 0
                concentration: Optional[float] = None
                if PERIODS[pollutant] == 24 and valid_count >= 16:
                    concentration = float(values.mean())
                elif PERIODS[pollutant] == 8:
                    rolling = pollutant_rolling_8h[pollutant].loc[day:day_end]
                    valid_windows = int(rolling.notna().sum())
                    daily_window_counts[pollutant] = valid_windows
                    if valid_count >= 16 and valid_windows:
                        concentration = float(rolling.max())

                daily_concentrations[pollutant] = concentration
                reason = _pollutant_reason(
                    pollutant,
                    valid_count,
                    daily_window_counts[pollutant],
                    negative_count,
                    len(day_hours) - valid_count,
                )
                if reason:
                    pollutant_reasons.append(reason)
                    daily_pollutant_notes.append(reason)
                if concentration is not None:
                    readings[pollutant] = PollutantReading(
                        value=concentration,
                        unit=UNITS[pollutant],
                        averaging_period_hours=PERIODS[pollutant],
                        valid_hours=valid_count,
                    )

            if not complete_day:
                result = None
                exclusion_reasons = ["fewer than 24 consecutive hourly timestamps"]
                original_aqi = None
            elif original_aqi is None:
                result = None
                exclusion_reasons = ["fewer than 16 valid hourly source AQI values"]
            else:
                result = calculate_aqi(readings)
                exclusion_reasons = list(pollutant_reasons)
                if result.overall_aqi is None:
                    exclusion_reasons.extend(result.validation_messages)
                if result.overall_aqi is not None and result.status not in {
                    "valid",
                    "valid_with_warnings",
                }:
                    exclusion_reasons.append(
                        f"calculator returned non-comparable status {result.status}"
                    )

            comparable = (
                complete_day
                and original_aqi is not None
                and result is not None
                and result.overall_aqi is not None
                and result.status in {"valid", "valid_with_warnings"}
            )
            calculated_aqi = (
                float(result.overall_aqi) if comparable and result else None
            )
            difference = (
                calculated_aqi - original_aqi
                if calculated_aqi is not None and original_aqi is not None
                else None
            )

            row = {
                "location_name": location,
                "date": day.date().isoformat(),
                "year": int(day.year),
                "month": int(day.month),
                "season": _season_for_month(int(day.month)),
                "is_complete_24_hour_day": complete_day,
                "source_aqi_valid_hours": int(len(original_values)),
                "original_dataset_aqi": original_aqi,
                "original_dataset_aqi_category": _category(original_aqi),
                "cpcb_style_calculated_aqi": calculated_aqi,
                "cpcb_style_category": result.category if comparable and result else None,
                "difference_calculated_minus_original": difference,
                "absolute_difference": abs(difference) if difference is not None else None,
                "dominant_pollutant": (
                    result.dominant_pollutant if comparable and result else None
                ),
                "calculator_status": result.status if result else "not_calculated",
                "comparison_status": "comparable" if comparable else "excluded",
                "exclusion_reason": (
                    "" if comparable else "; ".join(dict.fromkeys(exclusion_reasons))
                ),
                "pollutant_validation_notes": "; ".join(daily_pollutant_notes),
                "assumed_units": "µg/m³ except CO in mg/m³; raw CSV has no unit metadata",
                "cpcb_hourly_averaging_window_method": (
                    "calendar-day means for 24-hour pollutants; daily maximum "
                    "complete trailing 8-hour mean for CO/O3"
                ),
            }
            for pollutant in POLLUTANT_COLUMNS:
                key = pollutant.lower().replace(".", "")
                row[f"{key}_daily_or_max_8h_concentration"] = daily_concentrations[
                    pollutant
                ]
                row[f"{key}_valid_hour_count"] = daily_valid_hours[pollutant]
                row[f"{key}_valid_8h_window_count"] = daily_window_counts[pollutant]
                sub_index = (
                    result.pollutant_sub_indices[pollutant]
                    if result is not None
                    else None
                )
                row[f"{key}_sub_index"] = (
                    sub_index.sub_index if sub_index is not None else None
                )
                row[f"{key}_sub_index_status"] = (
                    sub_index.status if sub_index is not None else "not_calculated"
                )
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["location_name", "date"]).reset_index(
        drop=True
    )


def _metric_summary(frame: pd.DataFrame) -> dict:
    """Calculate requested metrics for rows with both AQI values."""
    comparable = frame.loc[
        frame["comparison_status"] == "comparable",
        [
            "original_dataset_aqi",
            "cpcb_style_calculated_aqi",
            "difference_calculated_minus_original",
            "absolute_difference",
            "original_dataset_aqi_category",
            "cpcb_style_category",
        ],
    ].dropna()
    n = len(comparable)
    if n == 0:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "pearson_r": None,
            "mean_difference": None,
            "median_difference": None,
            "maximum_absolute_difference": None,
            "within_5_percent": None,
            "within_10_percent": None,
            "category_agreement_percent": None,
        }

    differences = comparable["difference_calculated_minus_original"]
    pearson = None
    if comparable["original_dataset_aqi"].nunique() > 1 and comparable[
        "cpcb_style_calculated_aqi"
    ].nunique() > 1:
        pearson = float(
            comparable["original_dataset_aqi"].corr(
                comparable["cpcb_style_calculated_aqi"], method="pearson"
            )
        )
    return {
        "n": n,
        "mae": float(comparable["absolute_difference"].mean()),
        "rmse": float(np.sqrt(np.square(differences).mean())),
        "pearson_r": pearson,
        "mean_difference": float(differences.mean()),
        "median_difference": float(differences.median()),
        "maximum_absolute_difference": float(comparable["absolute_difference"].max()),
        "within_5_percent": float(
            (comparable["absolute_difference"] <= 5).mean() * 100
        ),
        "within_10_percent": float(
            (comparable["absolute_difference"] <= 10).mean() * 100
        ),
        "category_agreement_percent": float(
            (
                comparable["original_dataset_aqi_category"]
                == comparable["cpcb_style_category"]
            ).mean()
            * 100
        ),
    }


def _grouped_metrics(
    frame: pd.DataFrame, column: str, ordered_values: Optional[Sequence[object]] = None
) -> List[Tuple[str, dict]]:
    valid = frame.loc[frame["comparison_status"] == "comparable"]
    grouped = []
    groups = list(valid.groupby(column, sort=True, dropna=False))
    if ordered_values is not None:
        group_map = {str(key): group for key, group in groups}
        groups = [
            (value, group_map[str(value)])
            for value in ordered_values
            if str(value) in group_map
        ]
    for label, group in groups:
        grouped.append((str(label), _metric_summary(group)))
    return grouped


def _format_metrics(metrics: Mapping[str, object]) -> str:
    """Format one metric record for the plain-text report."""
    if metrics["n"] == 0:
        return "n=0; no comparable observations"
    return (
        f"n={metrics['n']}; MAE={metrics['mae']:.3f}; "
        f"RMSE={metrics['rmse']:.3f}; Pearson r="
        f"{metrics['pearson_r']:.4f}" if metrics["pearson_r"] is not None else
        f"n={metrics['n']}; MAE={metrics['mae']:.3f}; "
        f"RMSE={metrics['rmse']:.3f}; Pearson r=not defined"
    ) + (
        f"; mean difference={metrics['mean_difference']:.3f}; "
        f"median difference={metrics['median_difference']:.3f}; "
        f"max |difference|={metrics['maximum_absolute_difference']:.3f}; "
        f"within ±5={metrics['within_5_percent']:.2f}%; "
        f"within ±10={metrics['within_10_percent']:.2f}%; "
        f"category agreement={metrics['category_agreement_percent']:.2f}%"
    )


def _make_report(frame: pd.DataFrame, raw: pd.DataFrame) -> str:
    comparable = frame.loc[frame["comparison_status"] == "comparable"].copy()
    overall = _metric_summary(frame)
    exclusions = frame.loc[frame["comparison_status"] == "excluded"]
    exclusion_counts = (
        exclusions["exclusion_reason"].replace("", "unspecified").value_counts()
    )
    dominant_counts = comparable["dominant_pollutant"].value_counts()
    unresolved_count = int(
        (frame["calculator_status"] == "unresolved_numeric_sub_index").sum()
    )
    o3_unresolved_count = int(
        (frame["o3_sub_index_status"] == "averaging_breakpoint_mismatch").sum()
    )
    dominant_breakdown = _grouped_metrics(comparable, "dominant_pollutant")
    by_location = _grouped_metrics(frame, "location_name")
    by_year = _grouped_metrics(frame, "year")
    by_month = _grouped_metrics(frame, "month", list(range(1, 13)))
    season_order = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]
    by_season = _grouped_metrics(frame, "season", season_order)
    raw_correlations = raw[["aqi", *POLLUTANT_COLUMNS.values()]].corr()["aqi"].drop(
        "aqi"
    )
    raw_correlation_text = ", ".join(
        f"{pollutant} r={raw_correlations[column]:.4f}"
        for pollutant, column in POLLUTANT_COLUMNS.items()
    )
    raw_timestamped = raw.copy()
    raw_timestamped["timestamp"] = pd.to_datetime(
        raw_timestamped["timestamp"], errors="raise"
    )
    coverage_lines = []
    for location, group in raw_timestamped.groupby("location_name", sort=True):
        timestamps = group["timestamp"].sort_values()
        nonhourly_steps = int(
            timestamps.diff().dropna().ne(pd.Timedelta(hours=1)).sum()
        )
        coverage_lines.append(
            f"- {location}: {len(group):,} observations, "
            f"{timestamps.iloc[0]} through {timestamps.iloc[-1]}, "
            f"non-hourly steps={nonhourly_steps}."
        )
    numeric_values = raw[["aqi", *POLLUTANT_COLUMNS.values()]].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_numeric_count = int(numeric_values.isna().sum().sum())

    strong = (
        overall["within_10_percent"] is not None
        and overall["within_10_percent"] >= 80
        and overall["category_agreement_percent"] >= 80
    )
    partial = (
        overall["within_10_percent"] is not None
        and overall["within_10_percent"] >= 50
        and overall["category_agreement_percent"] >= 50
    )
    classification = (
        "STRONG AGREEMENT"
        if strong
        else "PARTIAL AGREEMENT"
        if partial
        else "POOR AGREEMENT"
    )

    def lines_for_groups(title: str, groups: Iterable[Tuple[str, dict]]) -> List[str]:
        output = [title]
        for label, metrics in groups:
            output.append(f"  {label}: {_format_metrics(metrics)}")
        if len(output) == 1:
            output.append("  No comparable observations.")
        return output

    lines = [
        "SAANSCHECK RESEARCH VALIDATION: SOURCE AQI VS CPCB-STYLE AQI",
        "=" * 76,
        "",
        "Scope and interpretation",
        f"- Raw rows inspected: {len(raw):,}.",
        f"- Candidate location-days: {len(frame):,}; comparable: {overall['n']:,}; "
        f"excluded: {len(exclusions):,}.",
        f"- Hourly coverage: {raw['location_name'].nunique()} locations; "
        f"{invalid_numeric_count:,} missing/non-numeric source AQI or pollutant "
        "values after numeric parsing.",
        *coverage_lines,
        "- Unit limitation: the raw CSV has no unit metadata. The experiment "
        "assumes the units displayed by the existing application (µg/m³ except "
        "CO in mg/m³). All calculated values and conclusions are conditional "
        "on this unverified assumption.",
        "- This is a CPCB-style calculation using the project's standalone "
        "calculator and cited breakpoint rules. It does not establish that the "
        "source dataset is official CPCB data.",
        "- Comparison grain: location-day. Original AQI is the arithmetic "
        "mean of that day's supplied hourly AQI, matching project preprocessing.",
        "- PM2.5/PM10/NO2/SO2 use the daily 24-hour mean. CO/O3 use the maximum "
        "complete trailing 8-hour mean ending within the calendar day; no "
        "single-hour O3/CO value is used as an 8-hour average.",
        "- A pollutant daily value requires at least 16 valid non-negative "
        "hourly observations. Eight-hour means additionally require eight "
        "consecutive hourly observations. Calculator-unresolved cases, "
        "including O3 above its validated 8-hour range, are excluded.",
        "- Category mapping uses CPCB AQI bands; unrounded float AQI values "
        "are retained, following the calculator.",
        "- Same-hour raw AQI/pollutant Pearson correlations: "
        f"{raw_correlation_text}. This shows association (strongest with "
        "particulate concentrations here), not the formula or provenance used "
        "to produce the supplied AQI.",
        "",
        "A. Do the dataset AQI and CPCB-style AQI agree?",
        f"- {classification}. Overall: {_format_metrics(overall)}.",
        "- Classification rubric (study-specific, not a CPCB standard): "
        "STRONG requires both at least 80% within ±10 AQI points and at least "
        "80% category agreement; PARTIAL requires both measures at least 50%; "
        "otherwise POOR.",
        "",
        "B. If they disagree, how large is the disagreement?",
        f"- Mean absolute difference (MAE): {overall['mae']:.3f} AQI points."
        if overall["n"]
        else "- Not calculable: no comparable observations.",
        f"- Mean signed difference (calculated minus source): "
        f"{overall['mean_difference']:.3f}; median: "
        f"{overall['median_difference']:.3f}; maximum absolute difference: "
        f"{overall['maximum_absolute_difference']:.3f}."
        if overall["n"]
        else "",
        f"- Within ±5: {overall['within_5_percent']:.2f}%; within ±10: "
        f"{overall['within_10_percent']:.2f}%; category agreement: "
        f"{overall['category_agreement_percent']:.2f}%."
        if overall["n"]
        else "",
        "",
        "C. Is disagreement concentrated in particular pollutants?",
        "- Dominant pollutant is the largest valid calculated sub-index. "
        "This is an association/breakdown, not proof that a pollutant caused "
        "the source-vs-calculated difference.",
    ]
    if len(dominant_counts):
        for pollutant in ("PM2.5", "PM10", "NO2", "SO2", "CO", "O3"):
            count = int(dominant_counts.get(pollutant, 0))
            share = count / overall["n"] * 100 if overall["n"] else 0
            lines.append(f"- {pollutant}: dominant on {count:,} days ({share:.2f}%).")
        lines.extend(lines_for_groups("  Metrics by calculated dominant pollutant:", dominant_breakdown))
    else:
        lines.append("- No calculated dominant pollutants among comparable days.")
    lines.extend(
        [
            "",
            "D. Is disagreement different by location?",
            *lines_for_groups("Metrics by location:", by_location),
            "",
            "E. Is disagreement different by season/year?",
            *lines_for_groups("Metrics by year:", by_year),
            *lines_for_groups("Metrics by calendar month:", by_month),
            *lines_for_groups("Metrics by season:", by_season),
            "",
            "F. Implication for ML forecasting",
            "- Existing forecasting targets use the source AQI and should be "
            "interpreted as forecasts of that source-defined target, not as "
            "validated CPCB AQI forecasts.",
            "- If agreement is poor or partial, the existing target embeds a "
            "different AQI semantic from the CPCB-style concentration-based "
            "index. Do not silently relabel it as CPCB AQI. Any future switch "
            "to a CPCB-derived target should be a separately reviewed dataset "
            "and model experiment; this validation did not retrain models.",
            "- Even high agreement would remain conditional on undocumented "
            "source provenance/units, this dataset's physical-consistency "
            "issues, and the chosen daily 8-hour maximum convention; that "
            "calendar-day summary choice is not independently established here "
            "as a mandatory CPCB daily aggregation rule.",
            "",
            "Daily aggregation assessment",
            "- The current 24-hour arithmetic mean of source AQI is reproducible "
            "and was used for an apples-to-apples comparison with the existing "
            "project target. It is not mathematically equivalent in general to "
            "calculating AQI from daily pollutant means because interpolation "
            "and the maximum-pollutant rule are nonlinear.",
            "- For CPCB-style daily comparison, 24-hour pollutant means and "
            "within-day maximum complete 8-hour CO/O3 averages are more "
            "methodologically aligned than calculating from a single hourly "
            "pollutant record. The source AQI's own averaging/provenance is "
            "undocumented.",
            "",
            "Exclusions",
            f"- Excluded candidate location-days: {len(exclusions):,}.",
            f"- Calculator unresolved numeric/averaging cases: {unresolved_count:,}; "
            f"O3 8-hour breakpoint-mismatch cases: {o3_unresolved_count:,}. "
            "These cases have no CPCB-style AQI value in the comparison output.",
        ]
    )
    if len(exclusion_counts):
        lines.extend(
            f"- {reason}: {int(count):,}" for reason, count in exclusion_counts.items()
        )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "Files",
            f"- Comparison table: {RESULTS_PATH.as_posix()}",
            f"- This report: {REPORT_PATH.as_posix()}",
            f"- Visualization: {FIGURE_PATH.as_posix()}",
            "",
            "No source AQI, raw/processed dataset, model, training logic, "
            "forecasting code, or production UI was modified.",
            "",
        ]
    )
    return "\n".join(lines)


def _save_figure(frame: pd.DataFrame, path: Path) -> None:
    """Save a compact observed-vs-calculated comparison plot."""
    comparable = frame.loc[frame["comparison_status"] == "comparable"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    palette = plt.get_cmap("Set2")
    locations = sorted(comparable["location_name"].unique())
    for index, location in enumerate(locations):
        subset = comparable.loc[comparable["location_name"] == location]
        axes[0].scatter(
            subset["original_dataset_aqi"],
            subset["cpcb_style_calculated_aqi"],
            s=10,
            alpha=0.36,
            color=palette(index % 8),
            label=location,
            edgecolors="none",
        )
    if not comparable.empty:
        limits = [
            min(
                comparable["original_dataset_aqi"].min(),
                comparable["cpcb_style_calculated_aqi"].min(),
            ),
            max(
                comparable["original_dataset_aqi"].max(),
                comparable["cpcb_style_calculated_aqi"].max(),
            ),
        ]
        axes[0].plot(limits, limits, color="#53636F", linewidth=1.2, linestyle="--")
        axes[0].set_xlim(limits)
        axes[0].set_ylim(limits)
    axes[0].set_title("Daily source AQI vs CPCB-style AQI")
    axes[0].set_xlabel("Original dataset AQI (daily mean)")
    axes[0].set_ylabel("CPCB-style calculated AQI")
    if locations:
        axes[0].legend(frameon=False, fontsize=8)

    if not comparable.empty:
        location_mae = (
            comparable.groupby("location_name")["absolute_difference"]
            .mean()
            .sort_values()
        )
        axes[1].bar(
            location_mae.index,
            location_mae.values,
            color="#0F8B8D",
            edgecolor="none",
        )
    axes[1].set_title("Mean absolute difference by location")
    axes[1].set_xlabel("Location")
    axes[1].set_ylabel("MAE (AQI points)")
    axes[1].tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.grid(axis="y", color="#DDE5E9", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SaansCheck AQI validation — location-day comparisons", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def run_validation(
    raw_path: Path = RAW_PATH,
    results_path: Path = RESULTS_PATH,
    report_path: Path = REPORT_PATH,
    figure_path: Path = FIGURE_PATH,
) -> pd.DataFrame:
    """Run the research experiment and write its separate output artifacts."""
    raw = pd.read_csv(raw_path)
    frame = build_daily_comparisons(raw)
    frame.to_csv(results_path, index=False)
    report = _make_report(frame, raw)
    report_path.write_text(report, encoding="utf-8")
    _save_figure(frame, figure_path)
    return frame


if __name__ == "__main__":
    result_frame = run_validation()
    counts = result_frame["comparison_status"].value_counts().to_dict()
    print(f"Wrote {RESULTS_PATH}: {len(result_frame):,} candidate location-days")
    print(f"Comparison statuses: {counts}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {FIGURE_PATH}")
