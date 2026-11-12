"""Category metrics derived from held-out historical regression predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocess import get_cpcb_aqi_category


HISTORICAL_CATEGORY_ORDER = (
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe",
)
MODEL_PREDICTION_COLUMNS = {
    "Linear Regression": "lr",
    "Random Forest": "rf",
}
HORIZON_DAYS = {"Day+1": 1, "Day+2": 2, "Day+3": 3}


def evaluate_regression_categories(
    predictions: pd.DataFrame,
    model: str,
    horizon: str,
) -> dict:
    """Evaluate categories obtained by applying preprocessing thresholds to AQI values."""
    if model not in MODEL_PREDICTION_COLUMNS:
        raise ValueError(f"Unsupported regression model: {model!r}")
    if horizon not in HORIZON_DAYS:
        raise ValueError(f"Unsupported forecast horizon: {horizon!r}")

    step = HORIZON_DAYS[horizon]
    target_suffix = f"AQI_t+{step}"
    actual_column = f"actual_{target_suffix}"
    prediction_column = (
        f"pred_{MODEL_PREDICTION_COLUMNS[model]}_{target_suffix}"
    )
    required = {"date", actual_column, prediction_column}
    missing_columns = required.difference(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Test predictions are missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    actual = pd.to_numeric(predictions[actual_column], errors="coerce")
    predicted = pd.to_numeric(predictions[prediction_column], errors="coerce")
    valid = (
        actual.notna()
        & predicted.notna()
        & np.isfinite(actual)
        & np.isfinite(predicted)
    )
    scored = predictions.loc[valid, ["date"]].copy()
    scored["actual"] = actual.loc[valid].to_numpy()
    scored["predicted"] = predicted.loc[valid].to_numpy()
    if scored.empty:
        raise ValueError(
            f"No finite actual/predicted values are available for {model} {horizon}."
        )

    y_true = scored["actual"].map(get_cpcb_aqi_category)
    y_pred = scored["predicted"].map(get_cpcb_aqi_category)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(HISTORICAL_CATEGORY_ORDER),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(HISTORICAL_CATEGORY_ORDER),
    )
    report_table = pd.DataFrame.from_dict(
        {
            category: report[category]
            for category in HISTORICAL_CATEGORY_ORDER
        },
        orient="index",
    ).rename_axis("Category")
    report_table["Actual support"] = report_table["support"].astype(int)
    report_table["Predicted count"] = (
        pd.Series(y_pred).value_counts().reindex(HISTORICAL_CATEGORY_ORDER, fill_value=0)
        .astype(int)
        .to_numpy()
    )
    report_table = report_table[
        ["precision", "recall", "f1-score", "Actual support", "Predicted count"]
    ].rename(
        columns={
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1-score",
        }
    )

    scored["date"] = pd.to_datetime(scored["date"], errors="coerce")
    origin_start = scored["date"].min()
    origin_end = scored["date"].max()
    return {
        "model": model,
        "horizon": horizon,
        "sample_count": int(len(scored)),
        "excluded_count": int(len(predictions) - len(scored)),
        "excluded_actual_count": int((actual.isna() | ~np.isfinite(actual)).sum()),
        "excluded_prediction_count": int(
            (predicted.isna() | ~np.isfinite(predicted)).sum()
        ),
        "categories": HISTORICAL_CATEGORY_ORDER,
        "report": report,
        "per_class": report_table,
        "confusion_matrix": matrix,
        "origin_start": origin_start,
        "origin_end": origin_end,
        "target_start": origin_start + pd.Timedelta(days=step),
        "target_end": origin_end + pd.Timedelta(days=step),
        "actual_distribution": (
            pd.Series(y_true)
            .value_counts()
            .reindex(HISTORICAL_CATEGORY_ORDER, fill_value=0)
            .astype(int)
        ),
        "predicted_distribution": (
            pd.Series(y_pred)
            .value_counts()
            .reindex(HISTORICAL_CATEGORY_ORDER, fill_value=0)
            .astype(int)
        ),
    }
