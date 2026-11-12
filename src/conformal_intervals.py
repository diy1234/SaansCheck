"""Chronological split-conformal interval validation for historical AQI models."""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.forecast_features import FORECAST_FEATURE_COLUMNS, FORECAST_STATIONS
from src.walk_forward_validation import HORIZONS, RF_PARAMETERS, _validate_dataset


NOMINAL_COVERAGE = 0.90
TRAIN_TARGET_END = pd.Timestamp("2020-12-31")
CALIBRATION_START = pd.Timestamp("2021-01-01")
CALIBRATION_END = pd.Timestamp("2021-12-31")
EVALUATION_START = pd.Timestamp("2022-01-01")
EVALUATION_END = pd.Timestamp("2023-12-31")


def conformal_quantile(
    absolute_residuals: Any,
    coverage: float = NOMINAL_COVERAGE,
) -> float:
    """Return the finite-sample split-conformal absolute-residual quantile.

    The one-indexed rank is ceil((n + 1) * coverage), as used by split
    conformal prediction under exchangeability.
    """
    scores = np.asarray(absolute_residuals, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Calibration residuals must be a non-empty one-dimensional array.")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Calibration residuals must be finite and non-negative.")
    if not np.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("Coverage must be finite and strictly between zero and one.")

    rank = ceil((scores.size + 1) * coverage)
    if rank > scores.size:
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def prediction_interval(
    predictions: Any,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct symmetric finite prediction bounds without clipping AQI values."""
    point = np.asarray(predictions, dtype=float)
    if point.ndim != 1 or not np.isfinite(point).all():
        raise ValueError("Predictions must be a finite one-dimensional array.")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Interval radius must be finite and non-negative.")
    lower = point - radius
    upper = point + radius
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Prediction interval bounds must be finite.")
    return lower, upper


def interval_metrics(actual: Any, lower: Any, upper: Any) -> dict[str, float | int]:
    """Calculate empirical inclusive coverage and mean interval width."""
    observed = np.asarray(actual, dtype=float)
    lower_bounds = np.asarray(lower, dtype=float)
    upper_bounds = np.asarray(upper, dtype=float)
    if (
        observed.ndim != 1
        or observed.size == 0
        or observed.shape != lower_bounds.shape
        or observed.shape != upper_bounds.shape
    ):
        raise ValueError("Actual values and bounds must be non-empty arrays of equal length.")
    if not (
        np.isfinite(observed).all()
        and np.isfinite(lower_bounds).all()
        and np.isfinite(upper_bounds).all()
    ):
        raise ValueError("Actual values and interval bounds must be finite.")
    if (lower_bounds > upper_bounds).any():
        raise ValueError("Every lower interval bound must be at or below its upper bound.")

    return {
        "n_evaluation": int(observed.size),
        "empirical_coverage": float(
            np.mean((observed >= lower_bounds) & (observed <= upper_bounds))
        ),
        "average_interval_width": float(np.mean(upper_bounds - lower_bounds)),
    }


def validate_chronological_intervals(
    dataset: pd.DataFrame,
    model_factory: Any | None = None,
) -> pd.DataFrame:
    """Evaluate station- and horizon-specific split-conformal intervals.

    A validation-only model is fit on target dates through 2020. Absolute
    residuals from 2021 form calibration scores, and 2022-2023 targets are
    reserved for evaluation. No estimator or artifact is saved.
    """
    frame = _validate_dataset(dataset)
    estimator_factory = model_factory or RandomForestRegressor
    results: list[dict[str, Any]] = []

    for horizon_label, target_column, step in HORIZONS:
        target_date_column = f"_target_date_{step}"
        target_dates = frame[target_date_column]
        target_values = pd.to_numeric(frame[target_column], errors="coerce")

        train_mask = target_values.notna() & target_dates.le(TRAIN_TARGET_END)
        calibration_mask = (
            target_values.notna()
            & target_dates.ge(CALIBRATION_START)
            & target_dates.le(CALIBRATION_END)
        )
        evaluation_mask = (
            target_values.notna()
            & target_dates.ge(EVALUATION_START)
            & target_dates.le(EVALUATION_END)
        )
        train = frame.loc[train_mask]
        calibration = frame.loc[calibration_mask]
        evaluation = frame.loc[evaluation_mask]
        if train.empty or calibration.empty or evaluation.empty:
            raise ValueError(f"Empty chronological partition for {horizon_label}.")
        if not (
            train[target_date_column].max() < calibration[target_date_column].min()
            and calibration[target_date_column].max() < evaluation[target_date_column].min()
        ):
            raise AssertionError("Training, calibration, and evaluation targets overlap.")
        if target_column in FORECAST_FEATURE_COLUMNS:
            raise AssertionError("A future target is present in the predictor contract.")

        model = estimator_factory(**RF_PARAMETERS)
        model.fit(
            train[FORECAST_FEATURE_COLUMNS],
            target_values.loc[train.index],
        )
        calibration_predictions = np.asarray(
            model.predict(calibration[FORECAST_FEATURE_COLUMNS]),
            dtype=float,
        )
        evaluation_predictions = np.asarray(
            model.predict(evaluation[FORECAST_FEATURE_COLUMNS]),
            dtype=float,
        )
        calibration_actual = target_values.loc[calibration.index].to_numpy(dtype=float)
        evaluation_actual = target_values.loc[evaluation.index].to_numpy(dtype=float)
        if not (
            np.isfinite(calibration_predictions).all()
            and np.isfinite(evaluation_predictions).all()
            and np.isfinite(calibration_actual).all()
            and np.isfinite(evaluation_actual).all()
        ):
            raise ValueError(f"Non-finite values encountered for {horizon_label}.")

        per_location: dict[str, dict[str, Any]] = {}
        for location in FORECAST_STATIONS:
            calibration_rows = calibration["location_name"].eq(location).to_numpy()
            evaluation_rows = evaluation["location_name"].eq(location).to_numpy()
            if not calibration_rows.any() or not evaluation_rows.any():
                raise ValueError(f"Missing calibration or evaluation rows for {location}.")
            residuals = np.abs(
                calibration_actual[calibration_rows]
                - calibration_predictions[calibration_rows]
            )
            radius = conformal_quantile(residuals, NOMINAL_COVERAGE)
            lower, upper = prediction_interval(
                evaluation_predictions[evaluation_rows],
                radius,
            )
            metrics = interval_metrics(
                evaluation_actual[evaluation_rows],
                lower,
                upper,
            )
            per_location[location] = {
                "radius": radius,
                "n_calibration": int(calibration_rows.sum()),
                "metrics": metrics,
            }
            results.append({
                "horizon": horizon_label,
                "location": location,
                "nominal_coverage": NOMINAL_COVERAGE,
                "calibration_method": "Split conformal; absolute residual; location-specific",
                "train_target_end": TRAIN_TARGET_END.date().isoformat(),
                "calibration_target_start": CALIBRATION_START.date().isoformat(),
                "calibration_target_end": CALIBRATION_END.date().isoformat(),
                "n_calibration": int(calibration_rows.sum()),
                "evaluation_target_start": EVALUATION_START.date().isoformat(),
                "evaluation_target_end": EVALUATION_END.date().isoformat(),
                "n_evaluation": metrics["n_evaluation"],
                "conformal_radius": radius,
                "empirical_coverage": metrics["empirical_coverage"],
                "average_interval_width": metrics["average_interval_width"],
            })

        aggregate_actual = []
        aggregate_lower = []
        aggregate_upper = []
        for location in FORECAST_STATIONS:
            location_rows = evaluation["location_name"].eq(location).to_numpy()
            lower, upper = prediction_interval(
                evaluation_predictions[location_rows],
                per_location[location]["radius"],
            )
            aggregate_actual.extend(evaluation_actual[location_rows])
            aggregate_lower.extend(lower)
            aggregate_upper.extend(upper)
        aggregate_metrics = interval_metrics(
            aggregate_actual,
            aggregate_lower,
            aggregate_upper,
        )
        results.append({
            "horizon": horizon_label,
            "location": "ALL",
            "nominal_coverage": NOMINAL_COVERAGE,
            "calibration_method": "Split conformal; absolute residual; location-specific",
            "train_target_end": TRAIN_TARGET_END.date().isoformat(),
            "calibration_target_start": CALIBRATION_START.date().isoformat(),
            "calibration_target_end": CALIBRATION_END.date().isoformat(),
            "n_calibration": int(len(calibration)),
            "evaluation_target_start": EVALUATION_START.date().isoformat(),
            "evaluation_target_end": EVALUATION_END.date().isoformat(),
            "n_evaluation": aggregate_metrics["n_evaluation"],
            "conformal_radius": np.nan,
            "empirical_coverage": aggregate_metrics["empirical_coverage"],
            "average_interval_width": aggregate_metrics["average_interval_width"],
        })

    return pd.DataFrame(results)
