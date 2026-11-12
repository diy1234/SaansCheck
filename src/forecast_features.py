"""Training-contract feature construction for historical AQI forecasts."""

from __future__ import annotations

from typing import Any

import pandas as pd


FORECAST_FEATURE_COLUMNS = [
    "aqi", "AQI_lag_1", "AQI_lag_2", "AQI_lag_3",
    "AQI_rolling_mean_3", "AQI_rolling_mean_7", "AQI_rolling_std_7",
    "pm25", "pm10", "no2", "so2", "co", "o3",
    "month", "day", "day_of_week", "is_weekend",
    "loc_Delhi", "loc_Faridabad", "loc_Ghaziabad", "loc_Gurgaon", "loc_Noida",
]

FORECAST_STATIONS = ["Delhi", "Faridabad", "Ghaziabad", "Gurgaon", "Noida"]
POLLUTANT_COLUMNS = ["pm25", "pm10", "no2", "so2", "co", "o3"]


def build_forecast_feature_row(
    daily_df: pd.DataFrame,
    station_name: str,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build the observed-origin feature row using the training definitions.

    The input row is the latest observed day t, used by each direct model to
    predict AQI_t+1, AQI_t+2, or AQI_t+3. Rolling features are computed from
    prior observations only, with the same shift, min_periods, and pandas
    sample-standard-deviation convention as preprocessing.
    """
    if station_name not in FORECAST_STATIONS:
        raise ValueError(f"Unknown forecast station: {station_name}")

    required = {"date", "location_name", "aqi", *POLLUTANT_COLUMNS}
    missing = sorted(required.difference(daily_df.columns))
    if missing:
        raise ValueError(f"Historical forecast data is missing columns: {missing}")

    station_daily = daily_df.loc[
        daily_df["location_name"] == station_name
    ].copy()
    if station_daily.empty:
        raise ValueError(f"No records available for station: {station_name}")

    station_daily["date"] = pd.to_datetime(station_daily["date"], errors="coerce")
    if station_daily["date"].isna().any():
        raise ValueError(f"Historical dates are invalid for station: {station_name}")
    station_daily = station_daily.sort_values("date").reset_index(drop=True)
    aqi = pd.to_numeric(station_daily["aqi"], errors="coerce")
    if aqi.isna().any():
        raise ValueError(f"Historical AQI contains missing or invalid values for {station_name}")
    if len(aqi) < 4:
        raise ValueError(
            f"At least four observed AQI records are required for {station_name} "
            "to construct the three lag features."
        )

    latest = station_daily.iloc[-1]
    origin_date = pd.Timestamp(latest["date"])
    past_only = aqi.shift(1)

    feature_row: dict[str, Any] = {
        "aqi": float(aqi.iloc[-1]),
        "AQI_lag_1": float(aqi.iloc[-2]),
        "AQI_lag_2": float(aqi.iloc[-3]),
        "AQI_lag_3": float(aqi.iloc[-4]),
        "AQI_rolling_mean_3": float(past_only.rolling(window=3, min_periods=1).mean().iloc[-1]),
        "AQI_rolling_mean_7": float(past_only.rolling(window=7, min_periods=1).mean().iloc[-1]),
        "AQI_rolling_std_7": float(past_only.rolling(window=7, min_periods=2).std().iloc[-1]),
        "month": int(origin_date.month),
        "day": int(origin_date.day),
        "day_of_week": int(origin_date.dayofweek),
        "is_weekend": int(origin_date.dayofweek >= 5),
    }

    for pollutant in POLLUTANT_COLUMNS:
        value = pd.to_numeric(pd.Series([latest[pollutant]]), errors="coerce").iloc[0]
        if pd.isna(value):
            raise ValueError(
                f"Latest observed {pollutant} value is missing or invalid for {station_name}"
            )
        feature_row[pollutant] = float(value)

    for location in FORECAST_STATIONS:
        feature_row[f"loc_{location}"] = int(location == station_name)

    ordered = pd.DataFrame([feature_row], columns=FORECAST_FEATURE_COLUMNS)
    if ordered.shape != (1, 22):
        raise AssertionError(f"Expected one row with 22 predictors, got {ordered.shape}")
    return ordered, origin_date


def build_direct_forecasts(
    daily_df: pd.DataFrame,
    station_name: str,
    models_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    """Predict each horizon from the same observed origin feature row.

    These are separately trained direct-horizon models, not recursive models.
    No predicted or future actual AQI is fed into another horizon's features.
    """
    features, origin_date = build_forecast_feature_row(daily_df, station_name)
    forecasts: list[dict[str, Any]] = []

    for step in (1, 2, 3):
        model = models_dict[f"rf_day{step}"]
        model_feature_names = getattr(model, "feature_names_in_", None)
        if model_feature_names is not None and list(model_feature_names) != FORECAST_FEATURE_COLUMNS:
            raise ValueError(
                f"rf_day{step} feature order does not match the training contract."
            )
        feature_count = getattr(model, "n_features_in_", None)
        if feature_count is not None and feature_count != len(FORECAST_FEATURE_COLUMNS):
            raise ValueError(
                f"rf_day{step} expects {feature_count} features; "
                f"{len(FORECAST_FEATURE_COLUMNS)} are required."
            )

        prediction = float(model.predict(features)[0])
        forecasts.append({
            "horizon": f"Day +{step}",
            "date": origin_date + pd.Timedelta(days=step),
            "predicted_aqi": max(prediction, 0.0),
        })

    return forecasts
