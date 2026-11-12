"""Pure data helpers for historical dashboard summaries and visualizations."""

from __future__ import annotations

import calendar
from typing import Any

import numpy as np
import pandas as pd


def _prepare_history(
    daily_df: pd.DataFrame,
    location: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    required = {"date", "location_name", "aqi"}
    missing = required.difference(daily_df.columns)
    if missing:
        raise ValueError(f"Historical data is missing required columns: {sorted(missing)}")

    frame = daily_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["aqi"] = pd.to_numeric(frame["aqi"], errors="coerce")
    frame = frame.loc[
        frame["date"].notna()
        & frame["location_name"].notna()
        & np.isfinite(frame["aqi"])
    ].copy()
    if location is not None:
        frame = frame.loc[frame["location_name"].eq(location)]
    if year is not None:
        frame = frame.loc[frame["date"].dt.year.eq(int(year))]
    return frame.sort_values(["date", "location_name"]).reset_index(drop=True)


def available_history_years(daily_df: pd.DataFrame) -> list[int]:
    """Return years represented by parseable historical dates."""
    if "date" not in daily_df.columns:
        raise ValueError("Historical data is missing the required date column.")
    dates = pd.to_datetime(daily_df["date"], errors="coerce").dropna()
    return sorted(int(year) for year in dates.dt.year.unique())


def build_key_findings(
    daily_df: pd.DataFrame,
    location: str,
    year: int | None = None,
) -> dict[str, Any]:
    """Summarize actual AQI observations for one location and optional year."""
    frame = _prepare_history(daily_df, location=location, year=year)
    if frame.empty:
        return {"count": 0, "location": location, "year": year}

    earliest = frame.iloc[0]
    latest = frame.iloc[-1]
    minimum = frame.loc[frame["aqi"].idxmin()]
    maximum = frame.loc[frame["aqi"].idxmax()]
    trend_change = (
        float(latest["aqi"] - earliest["aqi"])
        if frame["date"].nunique() > 1
        else None
    )
    categories = (
        frame["aqi_category"].dropna().astype(str).value_counts()
        if "aqi_category" in frame.columns
        else pd.Series(dtype="int64")
    )
    return {
        "count": int(frame["aqi"].count()),
        "location": location,
        "year": year,
        "minimum_aqi": float(minimum["aqi"]),
        "minimum_date": minimum["date"],
        "maximum_aqi": float(maximum["aqi"]),
        "maximum_date": maximum["date"],
        "latest_aqi": float(latest["aqi"]),
        "latest_date": latest["date"],
        "first_aqi": float(earliest["aqi"]),
        "first_date": earliest["date"],
        "trend_change": trend_change,
        "most_common_category": str(categories.index[0]) if not categories.empty else None,
    }


def build_location_leaderboard(
    daily_df: pd.DataFrame,
    year: int | None = None,
    metric: str = "mean",
) -> pd.DataFrame:
    """Rank available dataset locations by mean or maximum observed AQI."""
    if metric not in {"mean", "max"}:
        raise ValueError("metric must be either 'mean' or 'max'.")

    filtered = _prepare_history(daily_df, year=year)
    metric_column = "Average_AQI" if metric == "mean" else "Maximum_AQI"
    rows: list[dict[str, Any]] = []

    locations = sorted(daily_df["location_name"].dropna().astype(str).unique())
    for location in locations:
        records = filtered.loc[filtered["location_name"].eq(location)]
        values = records["aqi"]
        rows.append({
            "location_name": location,
            metric_column: float(values.mean() if metric == "mean" else values.max())
            if not values.empty else np.nan,
            "Average_AQI": float(values.mean()) if not values.empty else np.nan,
            "Median_AQI": float(values.median()) if not values.empty else np.nan,
            "Maximum_AQI": float(values.max()) if not values.empty else np.nan,
            "Minimum_AQI": float(values.min()) if not values.empty else np.nan,
            "Days_Recorded": int(values.count()),
            "Coverage_Start": records["date"].min() if not records.empty else pd.NaT,
            "Coverage_End": records["date"].max() if not records.empty else pd.NaT,
        })

    result = pd.DataFrame(rows)
    if result.empty:
        result["Rank"] = pd.Series(dtype="int64")
        return result
    result["Rank"] = result[metric_column].rank(method="dense", ascending=False)
    result = result.sort_values(
        [metric_column, "location_name"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    return result


def build_month_calendar(
    daily_df: pd.DataFrame,
    location: str,
    year: int,
    month: int,
) -> dict[str, Any]:
    """Create calendar positions and source-observed AQI values for one month."""
    weeks = calendar.monthcalendar(int(year), int(month))
    required = {"date", "location_name", "aqi"}
    missing = required.difference(daily_df.columns)
    if missing:
        raise ValueError(f"Historical data is missing required columns: {sorted(missing)}")
    frame = daily_df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["aqi"] = pd.to_numeric(frame["aqi"], errors="coerce")
    frame = frame.loc[
        frame["date"].notna()
        & frame["location_name"].eq(location)
        & frame["date"].dt.year.eq(int(year))
        & frame["date"].dt.month.eq(int(month))
    ].copy()
    frame["day"] = frame["date"].dt.day.astype(int)
    if frame["day"].duplicated().any():
        raise ValueError("Historical calendar data contains duplicate location/date rows.")

    observed_days = set(frame["day"].tolist())
    values = {
        int(row.day): float(row.aqi) if np.isfinite(row.aqi) else None
        for row in frame[["day", "aqi"]].itertuples(index=False)
    }
    return {
        "weeks": weeks,
        "values": values,
        "observed_days": observed_days,
        "records": frame[["date", "aqi"]].sort_values("date").reset_index(drop=True),
    }


def summarize_walk_forward_results(results_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Summarize verified pooled Random Forest metrics across three time windows."""
    required = {
        "model", "location", "horizon", "validation_window",
        "validation_start", "validation_end", "mae", "rmse",
    }
    missing = required.difference(results_df.columns)
    if missing:
        raise ValueError(f"Walk-forward results are missing columns: {sorted(missing)}")

    pooled = results_df.loc[
        results_df["model"].astype(str).eq("Random Forest")
        & results_df["location"].astype(str).eq("ALL")
    ].copy()
    for metric in ("mae", "rmse"):
        pooled[metric] = pd.to_numeric(pooled[metric], errors="coerce")
    pooled = pooled.loc[np.isfinite(pooled["mae"]) & np.isfinite(pooled["rmse"])]

    summaries: dict[str, dict[str, Any]] = {}
    for horizon, rows in pooled.groupby("horizon", sort=False):
        windows = rows["validation_window"].dropna().astype(str).unique()
        if len(windows) != 3:
            continue
        starts = pd.to_datetime(rows["validation_start"], errors="coerce").dropna()
        ends = pd.to_datetime(rows["validation_end"], errors="coerce").dropna()
        if starts.empty or ends.empty:
            continue
        summaries[str(horizon)] = {
            "mae": float(rows["mae"].mean()),
            "rmse": float(rows["rmse"].mean()),
            "window_count": int(len(windows)),
            "validation_start": starts.min(),
            "validation_end": ends.max(),
        }
    return summaries
