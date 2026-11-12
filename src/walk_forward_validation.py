"""Chronological walk-forward evaluation for SaansCheck AQI forecasts.

Temporary validation estimators are fitted in memory. This module never loads,
rewrites, or saves the production model artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.forecast_features import FORECAST_FEATURE_COLUMNS, FORECAST_STATIONS


HORIZONS = (
    ("Day+1", "AQI_t+1", 1),
    ("Day+2", "AQI_t+2", 2),
    ("Day+3", "AQI_t+3", 3),
)
AQI_CATEGORIES = ("Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe")
CATEGORY_MIN_SAMPLES = 30
RF_PARAMETERS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": -1,
}


@dataclass(frozen=True)
class ValidationWindow:
    """An expanding-window fold scored over one complete target calendar year."""

    name: str
    train_through_year: int
    validation_year: int

    @property
    def train_cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(self.train_through_year, 12, 31)

    @property
    def validation_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.validation_year, 1, 1)

    @property
    def validation_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.validation_year, 12, 31)


VALIDATION_WINDOWS = (
    ValidationWindow("2019-2020_to_2021", 2020, 2021),
    ValidationWindow("2019-2021_to_2022", 2021, 2022),
    ValidationWindow("2019-2022_to_2023", 2022, 2023),
)

RESULT_COLUMNS = [
    "validation_window",
    "train_start",
    "train_end",
    "train_origin_start",
    "train_origin_end",
    "validation_start",
    "validation_end",
    "model",
    "horizon",
    "location",
    "n_samples",
    "mae",
    "rmse",
    "r2",
    "persistence_mae",
    "skill_vs_persistence",
    "mae_improvement_vs_persistence",
    "mae_improvement_vs_linear_regression",
]


def _validate_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, chronological alignment, and feature leakage boundaries."""
    required = {
        "date", "location_name", "aqi", "aqi_category",
        *FORECAST_FEATURE_COLUMNS,
        *(target for _, target, _ in HORIZONS),
    }
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"ML-ready dataset is missing required columns: {missing}")

    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("ML-ready dataset contains invalid dates.")
    if frame.duplicated(["location_name", "date"]).any():
        raise ValueError("ML-ready dataset contains duplicate location/date rows.")
    if not set(FORECAST_STATIONS).issubset(set(frame["location_name"].unique())):
        raise ValueError("ML-ready dataset does not contain all five forecast locations.")

    for location, group in frame.groupby("location_name", sort=False):
        if not group["date"].is_monotonic_increasing:
            raise ValueError(f"Dates are not chronological for location {location}.")

    target_columns = {target for _, target, _ in HORIZONS}
    if target_columns.intersection(FORECAST_FEATURE_COLUMNS):
        raise AssertionError("A future target is included in the predictor schema.")
    if not set(FORECAST_FEATURE_COLUMNS).issubset(frame.columns):
        raise AssertionError("The training predictor schema is incomplete.")

    numeric_features = frame[FORECAST_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        raise ValueError("Predictor features contain missing or non-finite values.")

    dummy_columns = [f"loc_{location}" for location in FORECAST_STATIONS]
    if not numeric_features[dummy_columns].sum(axis=1).eq(1).all():
        raise ValueError("Location indicators are not one-hot encoded.")
    for location in FORECAST_STATIONS:
        expected = frame["location_name"].eq(location).astype(int)
        if not numeric_features[f"loc_{location}"].eq(expected).all():
            raise ValueError(f"Location indicator does not match {location} rows.")

    sorted_frame = frame.sort_values(["location_name", "date"]).copy()
    grouped = sorted_frame.groupby("location_name", sort=False)
    for horizon, (_, target, step) in enumerate(HORIZONS, 1):
        aligned = grouped["aqi"].shift(-step)
        actual_target = pd.to_numeric(sorted_frame[target], errors="coerce")
        valid = aligned.notna()
        if not np.allclose(
            aligned[valid].to_numpy(dtype=float),
            actual_target[valid].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
        ):
            raise ValueError(f"{target} does not align with the {step}-day grouped AQI shift.")
        if actual_target[~valid].notna().any():
            raise ValueError(f"{target} has values where its future AQI is unavailable.")
        target_date = sorted_frame["date"] + pd.to_timedelta(step, unit="D")
        category_target = grouped["aqi_category"].shift(-step)
        sorted_frame[f"_target_date_{horizon}"] = target_date
        sorted_frame[f"_target_category_{horizon}"] = category_target

    # Check stored lag and shifted rolling features after the dataset's initial
    # rows, where their history is fully represented in this ML-ready file.
    for _, group in sorted_frame.groupby("location_name", sort=False):
        aq = pd.to_numeric(group["aqi"], errors="coerce")
        expected = {
            "AQI_lag_1": aq.shift(1),
            "AQI_lag_2": aq.shift(2),
            "AQI_lag_3": aq.shift(3),
            "AQI_rolling_mean_3": aq.shift(1).rolling(3, min_periods=1).mean(),
            "AQI_rolling_mean_7": aq.shift(1).rolling(7, min_periods=1).mean(),
            "AQI_rolling_std_7": aq.shift(1).rolling(7, min_periods=2).std(),
        }
        check_rows = np.arange(len(group)) >= 10
        for column, calculated in expected.items():
            stored = pd.to_numeric(group[column], errors="coerce")
            if not np.allclose(
                stored.to_numpy(dtype=float)[check_rows],
                calculated.to_numpy(dtype=float)[check_rows],
                rtol=1e-8,
                atol=1e-8,
                equal_nan=True,
            ):
                raise ValueError(f"{column} is inconsistent with its backward-looking definition.")

    return sorted_frame.reset_index(drop=True)


def split_window(
    frame: pd.DataFrame,
    window: ValidationWindow,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by target dates, excluding labels that cross the training cutoff."""
    target_date_col = f"_target_date_{horizon}"
    if target_date_col not in frame:
        raise ValueError(f"Validated target dates are missing for horizon {horizon}.")
    target_col = f"AQI_t+{horizon}"
    target_dates = frame[target_date_col]
    targets = pd.to_numeric(frame[target_col], errors="coerce")

    train_mask = targets.notna() & target_dates.le(window.train_cutoff)
    validation_mask = (
        targets.notna()
        & target_dates.ge(window.validation_start)
        & target_dates.le(window.validation_end)
    )
    train = frame.loc[train_mask].copy()
    validation = frame.loc[validation_mask].copy()

    if train.empty or validation.empty:
        raise ValueError(f"Empty training or validation partition for {window.name}, Day+{horizon}.")
    if train[target_date_col].max() > window.train_cutoff:
        raise AssertionError("A training target crosses the training cutoff.")
    if validation[target_date_col].min() <= window.train_cutoff:
        raise AssertionError("A validation target is not strictly after the training cutoff.")
    if validation[target_date_col].min() < window.validation_start:
        raise AssertionError("A validation target precedes its validation window.")
    if validation[target_date_col].max() > window.validation_end:
        raise AssertionError("A validation target exceeds its validation window.")
    if set(train[target_date_col]).intersection(set(validation[target_date_col])):
        raise AssertionError("Training and validation target dates overlap.")
    return train, validation


def calculate_metrics(actual: Any, predicted: Any) -> dict[str, float]:
    """Return finite MAE, RMSE and R² for a non-empty numeric sample."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("Metrics require non-empty actual and predicted arrays of equal size.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metrics cannot be calculated from non-finite values.")
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if not np.isfinite(list(metrics.values())).all():
        raise ValueError("Calculated metric is non-finite.")
    return metrics


def persistence_predictions(validation: pd.DataFrame) -> np.ndarray:
    """Predict every horizon with observed AQI at its forecast origin."""
    observed_origin_aqi = pd.to_numeric(validation["aqi"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(observed_origin_aqi).all():
        raise ValueError("Persistence origin AQI contains missing or non-finite values.")
    return observed_origin_aqi.copy()


def skill_score(model_mae: float, persistence_mae: float) -> float:
    """Compute persistence-relative skill; positive is better than persistence."""
    if not np.isfinite(model_mae) or not np.isfinite(persistence_mae) or persistence_mae <= 0:
        raise ValueError("Skill score requires finite MAEs and positive persistence MAE.")
    return float(1.0 - model_mae / persistence_mae)


def mae_improvement_percent(baseline_mae: float, model_mae: float) -> float:
    """Compute percentage MAE reduction relative to the named baseline."""
    if not np.isfinite(baseline_mae) or not np.isfinite(model_mae) or baseline_mae <= 0:
        raise ValueError("MAE improvement requires finite values and positive baseline MAE.")
    return float((baseline_mae - model_mae) / baseline_mae * 100.0)


def _fit_models(x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """Fit fresh validation-only estimators with production hyperparameters."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(**RF_PARAMETERS),
    }
    for model in models.values():
        model.fit(x_train, y_train)
    return models


def evaluate_walk_forward(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate in-memory models and return aggregate/location and category results."""
    frame = _validate_dataset(dataset)
    year_set = set(frame["date"].dt.year.unique())
    for window in VALIDATION_WINDOWS:
        if window.train_through_year not in year_set or window.validation_year not in year_set:
            raise ValueError(f"Dataset does not cover the years required for {window.name}.")

    results: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    target_models = ("Linear Regression", "Random Forest", "Persistence")

    for window in VALIDATION_WINDOWS:
        for horizon_label, target_col, step in HORIZONS:
            train, validation = split_window(frame, window, step)
            x_train = train[FORECAST_FEATURE_COLUMNS]
            x_validation = validation[FORECAST_FEATURE_COLUMNS]
            y_train = pd.to_numeric(train[target_col], errors="raise")
            y_validation = pd.to_numeric(validation[target_col], errors="raise").to_numpy(dtype=float)

            if set(x_train.columns) != set(FORECAST_FEATURE_COLUMNS):
                raise AssertionError("Unexpected predictor columns entered model training.")
            if target_col in x_train.columns or any(
                future_target in x_train.columns
                for _, future_target, _ in HORIZONS
            ):
                raise AssertionError("A future target entered the training feature matrix.")
            if not train[f"_target_date_{step}"].le(window.train_cutoff).all():
                raise AssertionError("Validation-period target information entered training.")

            models = _fit_models(x_train, y_train)
            predictions = {
                model_name: np.asarray(model.predict(x_validation), dtype=float)
                for model_name, model in models.items()
            }
            predictions["Persistence"] = persistence_predictions(validation)
            if not np.isfinite(y_validation).all():
                raise ValueError(f"Non-finite target values in {window.name}, {horizon_label}.")
            for model_name, values in predictions.items():
                if values.shape != y_validation.shape or not np.isfinite(values).all():
                    raise ValueError(f"Invalid predictions for {model_name}, {window.name}, {horizon_label}.")

            prediction_frame = validation[
                ["location_name", "date", f"_target_date_{step}", f"_target_category_{step}"]
            ].copy()
            prediction_frame["actual"] = y_validation
            for model_name, values in predictions.items():
                prediction_frame[model_name] = values

            persistence_by_location = {
                location: calculate_metrics(
                    group["actual"].to_numpy(),
                    group["Persistence"].to_numpy(),
                )["mae"]
                for location, group in prediction_frame.groupby("location_name")
            }
            persistence_by_location["ALL"] = calculate_metrics(
                prediction_frame["actual"], prediction_frame["Persistence"]
            )["mae"]
            lr_mae_by_location = {
                location: calculate_metrics(group["actual"], group["Linear Regression"])["mae"]
                for location, group in prediction_frame.groupby("location_name")
            }
            lr_mae_by_location["ALL"] = calculate_metrics(
                prediction_frame["actual"], prediction_frame["Linear Regression"]
            )["mae"]

            group_frames = [("ALL", prediction_frame)]
            group_frames.extend(
                (location, group)
                for location, group in prediction_frame.groupby("location_name", sort=True)
            )
            train_target_start = train[f"_target_date_{step}"].min()
            train_target_end = train[f"_target_date_{step}"].max()
            train_origin_start = train["date"].min()
            train_origin_end = train["date"].max()

            for location, group in group_frames:
                baseline_mae = persistence_by_location[location]
                lr_mae = lr_mae_by_location[location]
                for model_name in target_models:
                    metrics = calculate_metrics(group["actual"], group[model_name])
                    skill = (
                        skill_score(metrics["mae"], baseline_mae)
                        if model_name != "Persistence" else 0.0
                    )
                    improvement_vs_persistence = (
                        mae_improvement_percent(baseline_mae, metrics["mae"])
                        if model_name != "Persistence" else 0.0
                    )
                    improvement_vs_lr = mae_improvement_percent(lr_mae, metrics["mae"])
                    results.append({
                        "validation_window": window.name,
                        "train_start": train_target_start.date().isoformat(),
                        "train_end": train_target_end.date().isoformat(),
                        "train_origin_start": train_origin_start.date().isoformat(),
                        "train_origin_end": train_origin_end.date().isoformat(),
                        "validation_start": window.validation_start.date().isoformat(),
                        "validation_end": window.validation_end.date().isoformat(),
                        "model": model_name,
                        "horizon": horizon_label,
                        "location": location,
                        "n_samples": int(len(group)),
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                        "persistence_mae": baseline_mae,
                        "skill_vs_persistence": skill,
                        "mae_improvement_vs_persistence": improvement_vs_persistence,
                        "mae_improvement_vs_linear_regression": improvement_vs_lr,
                    })

            category_col = f"_target_category_{step}"
            rf_errors = prediction_frame["Random Forest"] - prediction_frame["actual"]
            category_frame = prediction_frame.assign(
                absolute_error=rf_errors.abs(),
                squared_error=rf_errors.pow(2),
            )
            for category, group in category_frame.groupby(category_col, dropna=False):
                category_name = "Uncategorized" if pd.isna(category) else str(category)
                count = int(len(group))
                category_rows.append({
                    "validation_window": window.name,
                    "validation_year": window.validation_year,
                    "horizon": horizon_label,
                    "actual_aqi_category": category_name,
                    "n_samples": count,
                    "mae": float(group["absolute_error"].mean()),
                    "rmse": float(np.sqrt(group["squared_error"].mean())),
                    "included_in_interpretation": count >= CATEGORY_MIN_SAMPLES,
                })

    results_df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    category_df = pd.DataFrame(category_rows)
    numeric_metrics = results_df[
        ["mae", "rmse", "r2", "persistence_mae", "skill_vs_persistence",
         "mae_improvement_vs_persistence"]
    ]
    if not np.isfinite(numeric_metrics.to_numpy(dtype=float)).all():
        raise ValueError("Walk-forward results contain non-finite core metrics.")
    return results_df, category_df


def build_skill_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize pooled-location metrics and mean skill across validation windows."""
    aggregate = results.loc[results["location"].eq("ALL")].copy()
    columns = [
        "validation_window", "model", "horizon", "n_samples", "mae", "rmse", "r2",
        "persistence_mae", "skill_vs_persistence", "mae_improvement_vs_persistence",
        "mae_improvement_vs_linear_regression",
    ]
    return aggregate[columns].sort_values(
        ["horizon", "validation_window", "model"]
    ).reset_index(drop=True)


def _save_plots(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_location = results.loc[results["location"].eq("ALL")]
    horizon_order = [label for label, _, _ in HORIZONS]
    model_order = ["Linear Regression", "Random Forest", "Persistence"]
    colors = {
        "Linear Regression": "#627D98",
        "Random Forest": "#0F8B8D",
        "Persistence": "#2E9D68",
    }
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    x = np.arange(len(horizon_order))
    width = 0.24
    for model_index, model_name in enumerate(model_order):
        group = all_location.loc[all_location["model"].eq(model_name)]
        means = group.groupby("horizon")["mae"].mean().reindex(horizon_order)
        deviations = group.groupby("horizon")["mae"].std(ddof=1).reindex(horizon_order).fillna(0)
        ax.bar(x + (model_index - 1) * width, means, width, yerr=deviations,
               capsize=3, label=model_name, color=colors[model_name])
    ax.set(title="Walk-forward MAE by model and horizon",
           ylabel="MAE (dataset-defined AQI points)", xlabel="Forecast horizon",
           xticks=x, xticklabels=horizon_order)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    path = output_dir / "mae_by_model_and_horizon.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    rf = all_location.loc[all_location["model"].eq("Random Forest")]
    for horizon in horizon_order:
        group = rf.loc[rf["horizon"].eq(horizon)].sort_values("validation_window")
        ax.plot(group["validation_window"], group["mae"], marker="o", linewidth=2,
                label=horizon, color={"Day+1": "#0F8B8D", "Day+2": "#2E9D68", "Day+3": "#627D98"}[horizon])
    ax.set(title="Random Forest MAE across chronological validation windows",
           ylabel="MAE (dataset-defined AQI points)", xlabel="Validation window")
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    path = output_dir / "random_forest_mae_by_window.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    rf = all_location.loc[all_location["model"].eq("Random Forest")]
    for horizon in horizon_order:
        group = rf.loc[rf["horizon"].eq(horizon)].sort_values("validation_window")
        ax.plot(group["validation_window"], group["mae_improvement_vs_persistence"],
                marker="o", linewidth=2, label=horizon,
                color={"Day+1": "#0F8B8D", "Day+2": "#2E9D68", "Day+3": "#627D98"}[horizon])
    ax.axhline(0, color="#64748B", linewidth=1)
    ax.set(title="Random Forest MAE improvement versus persistence",
           ylabel="MAE improvement (%)", xlabel="Validation window")
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    path = output_dir / "rf_improvement_vs_persistence.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    outputs.append(path)

    rf_locations = results.loc[
        results["model"].eq("Random Forest") & ~results["location"].eq("ALL")
    ]
    location_mean = rf_locations.groupby(["location", "horizon"])["mae"].mean().unstack("horizon")
    location_mean = location_mean.reindex(index=FORECAST_STATIONS, columns=horizon_order)
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    location_mean.plot(kind="bar", ax=ax, color=["#0F8B8D", "#2E9D68", "#627D98"])
    ax.set(title="Random Forest MAE by location",
           ylabel="MAE (dataset-defined AQI points)", xlabel="Location")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="Horizon", frameon=False)
    path = output_dir / "rf_mae_by_location.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    outputs.append(path)
    return outputs


def _write_report(
    results: pd.DataFrame,
    skill_summary: pd.DataFrame,
    category_results: pd.DataFrame,
    output_path: Path,
    plot_paths: list[Path],
) -> None:
    all_location = results.loc[results["location"].eq("ALL")]
    rf = all_location.loc[all_location["model"].eq("Random Forest")]
    lines = [
        "SAANSCHECK WALK-FORWARD VALIDATION AND MODEL SKILL ANALYSIS",
        "=" * 66,
        "",
        "1. METHODOLOGY",
        "Expanding chronological validation; no shuffling or random split.",
        "Each temporary model is fit in memory using the existing 22 predictors.",
        "Linear Regression and Random Forest are fit separately for each horizon.",
        "Random Forest settings match src/train_models.py: " + repr(RF_PARAMETERS),
        "Persistence predicts the observed origin-day aqi for each target horizon.",
        "The primary comparisons pool observations across locations. Location rows",
        "are also emitted separately in the machine-readable results.",
        "",
        "2. VALIDATION WINDOWS AND LABEL BOUNDARIES",
        "Window | training target dates allowed | validation target dates",
        "------- | ----------------------------- | ------------------------",
    ]
    for window in VALIDATION_WINDOWS:
        lines.append(
            f"{window.name} | 2019-01-05 through {window.train_cutoff.date()} "
            f"(per horizon) | {window.validation_start.date()} through {window.validation_end.date()}"
        )
    lines.extend([
        "",
        "The training mask is based on target dates, not just feature-row years.",
        "Rows whose future target crosses the year-end cutoff are excluded from",
        "training. Validation samples are selected by their actual target date;",
        "thus the first validation outcomes may use observed origin rows from the",
        "last few days of the previous year, which are available at forecast time.",
        "No validation target contributes to model fitting or tuning.",
        "",
        "3. WINDOW-LEVEL POOLED MODEL METRICS",
        "Each cell is MAE / RMSE / R². R² measures explained variance, not accuracy.",
    ])
    for window in VALIDATION_WINDOWS:
        lines.append(f"\n{window.name}")
        subset = all_location.loc[all_location["validation_window"].eq(window.name)]
        pivot = subset.pivot(index="horizon", columns="model", values=["mae", "rmse", "r2"])
        lines.append("Horizon | Linear Regression | Random Forest | Persistence")
        for horizon, _, _ in HORIZONS:
            values = []
            for model_name in ("Linear Regression", "Random Forest", "Persistence"):
                values.append(
                    f"{pivot.loc[horizon, ('mae', model_name)]:.3f} / "
                    f"{pivot.loc[horizon, ('rmse', model_name)]:.3f} / "
                    f"{pivot.loc[horizon, ('r2', model_name)]:.4f}"
                )
            lines.append(f"{horizon} | " + " | ".join(values))

    lines.extend([
        "",
        "4. RF IMPROVEMENT AND PERSISTENCE-RELATIVE SKILL",
        "Improvement (%) = (baseline MAE - RF MAE) / baseline MAE * 100.",
        "Skill = 1 - (model MAE / persistence MAE); positive means better than persistence.",
        "Window | Horizon | RF MAE | LR MAE | Persistence MAE | RF vs LR (%) | RF vs persistence (%) | RF skill",
    ])
    indexed = {
        (row.validation_window, row.horizon, row.model): row
        for row in all_location.itertuples(index=False)
    }
    for window in VALIDATION_WINDOWS:
        for horizon, _, _ in HORIZONS:
            r = indexed[(window.name, horizon, "Random Forest")]
            l = indexed[(window.name, horizon, "Linear Regression")]
            p = indexed[(window.name, horizon, "Persistence")]
            lines.append(
                f"{window.name} | {horizon} | {r.mae:.3f} | {l.mae:.3f} | {p.mae:.3f} | "
                f"{r.mae_improvement_vs_linear_regression:.2f} | "
                f"{r.mae_improvement_vs_persistence:.2f} | {r.skill_vs_persistence:.4f}"
            )

    lines.extend(["", "5. HORIZON COMPARISON (MEAN MAE ACROSS WINDOWS)"])
    horizon_mean = all_location.groupby(["model", "horizon"])["mae"].mean().unstack("horizon")
    lines.append("Model | Day+1 MAE | Day+2 MAE | Day+3 MAE | Day+3 minus Day+1")
    for model_name in ("Linear Regression", "Random Forest", "Persistence"):
        row = horizon_mean.loc[model_name]
        lines.append(
            f"{model_name} | {row['Day+1']:.3f} | {row['Day+2']:.3f} | "
            f"{row['Day+3']:.3f} | {row['Day+3'] - row['Day+1']:+.3f}"
        )
    lines.append(
        "Interpretation: increases/decreases above describe these observed mean MAEs; "
        "they are not assumed to be monotonic."
    )

    lines.extend([
        "",
        "6. RANDOM FOREST PERFORMANCE STABILITY ACROSS WINDOWS",
        "Values are mean and sample standard deviation (ddof=1) across the three windows.",
        "Horizon | MAE mean | MAE SD | RMSE mean | RMSE SD | R² mean | R² SD",
    ])
    for horizon, _, _ in HORIZONS:
        subset = rf.loc[rf["horizon"].eq(horizon)]
        lines.append(
            f"{horizon} | {subset.mae.mean():.3f} | {subset.mae.std(ddof=1):.3f} | "
            f"{subset.rmse.mean():.3f} | {subset.rmse.std(ddof=1):.3f} | "
            f"{subset.r2.mean():.4f} | {subset.r2.std(ddof=1):.4f}"
        )

    lines.extend([
        "",
        "7. LOCATION-WISE RANDOM FOREST PERFORMANCE",
        "Mean MAE across validation windows; location labels are dataset locations,",
        "not claims that each row represents a physical monitoring station.",
        "Location | Day+1 | Day+2 | Day+3",
    ])
    rf_location = results.loc[
        results["model"].eq("Random Forest") & ~results["location"].eq("ALL")
    ]
    location_pivot = rf_location.groupby(["location", "horizon"])["mae"].mean().unstack()
    for location in FORECAST_STATIONS:
        row = location_pivot.loc[location]
        lines.append(
            f"{location} | {row['Day+1']:.3f} | {row['Day+2']:.3f} | {row['Day+3']:.3f}"
        )

    lines.extend([
        "",
        "8. ACTUAL-AQI-CATEGORY ERROR ANALYSIS (RF ONLY)",
        f"Categories are the existing dataset-defined labels; metrics are reported only "
        f"when n >= {CATEGORY_MIN_SAMPLES}. These are not identified as CPCB categories.",
        "Window | Horizon | Dataset category | n | MAE | RMSE | Included",
    ])
    for row in category_results.sort_values(
        ["validation_window", "horizon", "actual_aqi_category"]
    ).itertuples(index=False):
        lines.append(
            f"{row.validation_window} | {row.horizon} | {row.actual_aqi_category} | "
            f"{row.n_samples} | {row.mae:.3f} | {row.rmse:.3f} | "
            f"{'yes' if row.included_in_interpretation else 'no (small n)'}"
        )

    lines.extend([
        "",
        "9. LEAKAGE AND IMPLEMENTATION CHECKS",
        "- Rows are checked for chronological date ordering within each location.",
        "- No random split or shuffle is used; validation windows are calendar-based.",
        "- The 22 predictors are explicit; no future target column is included in X.",
        "- AQI_t+1/+2/+3 are asserted equal to within-location forward AQI shifts.",
        "- Stored lags and shifted rolling features are checked against backward-only",
        "  computations after the dataset's initial history rows.",
        "- Location one-hot indicators are checked against location_name.",
        "- Training target dates are capped at the year-end before validation.",
        "- Validation observations are selected by target date, with no overlap of",
        "  training and validation target dates.",
        "- Persistence predictions are copied from observed origin-day aqi only.",
        "- Temporary models are fitted in memory; this script does not serialize models.",
        "",
        "10. EXISTING 2023 SPLIT BOUNDARY CONCERN",
        "The existing training script selects training rows by origin year <= 2022.",
        "For direct horizons, origins on the last 1, 2, or 3 days of 2022 can have",
        "targets in 2023. Those target labels cross the nominal 2023 test boundary;",
        "the existing split therefore is not a perfectly isolated target-date split.",
        "This experiment excludes such rows from each temporary training fold. No",
        "production artifact, historical dataset, or existing result file is changed.",
        "The 2023 window is scored only after fitting on labels ending by 2022-12-31.",
        "",
        "11. LIMITATIONS",
        "- Five locations from one dataset and three annual validation periods limit",
        "  geographic and temporal generalization claims.",
        "- The AQI remains dataset-defined; category labels and AQI provenance are not",
        "  converted or represented as verified CPCB AQI.",
        "- Annual folds provide only three observations for window-to-window stability;",
        "  standard deviations are descriptive, not inferential confidence intervals.",
        "- The 2023 period is both the prior project holdout and the final walk-forward",
        "  validation year; it was not used for fitting or hyperparameter selection here.",
        "- This compares historical inputs, not live-data forecasting performance.",
        "",
        "12. OUTPUT PLOTS",
    ])
    lines.extend(f"- {path.as_posix()}" for path in plot_paths)
    lines.extend(["", "13. CONCLUSION"])
    for horizon, _, _ in HORIZONS:
        grouped = all_location.loc[all_location["horizon"].eq(horizon)]
        wins = int((grouped.loc[grouped["model"].eq("Random Forest"), "mae"].to_numpy()
                    < grouped.loc[grouped["model"].eq("Linear Regression"), "mae"].to_numpy()).sum())
        rf_skill = grouped.loc[grouped["model"].eq("Random Forest"), "skill_vs_persistence"]
        lines.append(
            f"- {horizon}: RF MAE is lower than LR in {wins}/3 windows; "
            f"RF persistence-relative skill ranges {rf_skill.min():.4f} to {rf_skill.max():.4f}."
        )
    lines.extend([
        "Interpret the window-specific tables above rather than relying on one aggregate",
        "score. Positive persistence skill means lower MAE than the baseline; it is not",
        "a percentage-accuracy claim. R² is explained variance, not accuracy.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_walk_forward_validation(
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run validation and write results, report, summary, and research plots."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = Path(dataset_path or project_root / "data" / "processed" / "ml_ready_dataset.csv")
    output_dir = Path(output_dir or project_root / "data" / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(dataset_path)
    results, categories = evaluate_walk_forward(dataset)
    summary = build_skill_summary(results)
    plots_dir = output_dir / "plots"
    plot_paths = _save_plots(results, plots_dir)

    results_path = output_dir / "walk_forward_results.csv"
    skill_path = output_dir / "model_skill_summary.csv"
    category_path = output_dir / "walk_forward_category_results.csv"
    report_path = output_dir / "walk_forward_validation_report.txt"
    results.to_csv(results_path, index=False, float_format="%.10g")
    summary.to_csv(skill_path, index=False, float_format="%.10g")
    categories.to_csv(category_path, index=False, float_format="%.10g")
    _write_report(results, summary, categories, report_path, plot_paths)
    return results, summary, categories


if __name__ == "__main__":
    walk_forward_results, _, _ = run_walk_forward_validation()
    print(
        walk_forward_results.loc[
            walk_forward_results["location"].eq("ALL")
        ].to_string(index=False)
    )
