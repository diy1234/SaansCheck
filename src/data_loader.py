"""
===============================================================================
SaansCheck: A Personalized Air Quality Risk Advisor and AQI Forecasting System
Phase 5A: Reusable Data & Model Loader Module
===============================================================================

This module provides safe, cached, and validated data loading functions for the
SaansCheck Streamlit dashboard.

Guiding Principles:
  1. Real data only - No synthetic/demo placeholders.
  2. Graceful error handling - Helpful UI messages if an artifact is missing.
  3. Strict schema validation - Ensures columns, dates, and models match expectations.
  4. Streamlit caching - Fast dashboard response times with @st.cache_data and @st.cache_resource.
===============================================================================
"""

import os
import joblib
import pandas as pd
import streamlit as st


# =============================================================================
# 1. CORE DATA LOADING FUNCTIONS (WITH STREAMLIT CACHING)
# =============================================================================

@st.cache_data(show_spinner="Loading daily air quality records...")
def load_daily_air_quality(filepath: str = "data/processed/daily_air_quality.csv") -> pd.DataFrame:
    """
    Safely load and validate the cleaned daily air quality dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing processed dataset: '{filepath}'. Run 'python src/preprocess.py' first.")

    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"Dataset '{filepath}' is empty.")

    # Validate essential columns
    required_cols = ["date", "location_name", "aqi", "aqi_category", "pm25", "pm10", "no2", "so2", "co", "o3"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset '{filepath}' is missing required columns: {missing}")

    # Parse and validate dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isnull().any():
        raise ValueError(f"Dataset '{filepath}' contains unparseable dates.")

    return df


@st.cache_data(show_spinner="Loading ML-ready dataset...")
def load_ml_ready_dataset(filepath: str = "data/processed/ml_ready_dataset.csv") -> pd.DataFrame:
    """
    Safely load and validate the feature-engineered ML dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing ML dataset: '{filepath}'. Run 'python src/features.py' first.")

    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"Dataset '{filepath}' is empty.")

    required_targets = ["AQI_t+1", "AQI_t+2", "AQI_t+3"]
    missing = [t for t in required_targets if t not in df.columns]
    if missing:
        raise ValueError(f"Dataset '{filepath}' is missing target columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading model evaluation metrics...")
def load_model_results(filepath: str = "data/processed/model_results.csv") -> pd.DataFrame:
    """
    Safely load the empirical model benchmarking results.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing metrics file: '{filepath}'. Run 'python src/train_models.py' first.")

    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"Metrics file '{filepath}' is empty.")

    expected_cols = ["horizon", "target", "model", "train_rows", "test_rows", "MAE", "RMSE", "R2"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Metrics file '{filepath}' is missing expected columns: {missing}")

    return df


@st.cache_data(show_spinner="Loading test predictions...")
def load_test_predictions(filepath: str = "data/processed/test_predictions.csv") -> pd.DataFrame:
    """
    Safely load test predictions containing ground-truth vs model outputs for 2023.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing test predictions file: '{filepath}'. Run 'python src/train_models.py' first.")

    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"Test predictions file '{filepath}' is empty.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner="Loading feature importance rankings...")
def load_feature_importance(filepath: str = "data/processed/feature_importance.csv") -> pd.DataFrame:
    """
    Safely load Random Forest feature importances.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing feature importance file: '{filepath}'. Run 'python src/train_models.py' first.")

    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError(f"Feature importance file '{filepath}' is empty.")

    expected_cols = ["feature", "importance_day1", "importance_day2", "importance_day3", "mean_importance"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature importance file '{filepath}' is missing columns: {missing}")

    return df


# =============================================================================
# 2. MODEL ARTIFACT LOADING FUNCTIONS
# =============================================================================

@st.cache_resource(show_spinner="Loading trained ML models...")
def load_trained_models(models_dir: str = "models") -> dict:
    """
    Load all six serialized scikit-learn models using joblib.
    Returns a dictionary of loaded model objects.
    """
    model_filenames = {
        "lr_day1": "linear_regression_day1.joblib",
        "rf_day1": "random_forest_day1.joblib",
        "lr_day2": "linear_regression_day2.joblib",
        "rf_day2": "random_forest_day2.joblib",
        "lr_day3": "linear_regression_day3.joblib",
        "rf_day3": "random_forest_day3.joblib"
    }

    loaded_models = {}
    for key, filename in model_filenames.items():
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Serialized model '{filename}' not found in '{models_dir}'. Run 'python src/train_models.py' first.")
        loaded_models[key] = joblib.load(path)

    return loaded_models


# =============================================================================
# 3. END-TO-END VALIDATION AUDIT
# =============================================================================

def validate_all_system_artifacts() -> dict:
    """
    Non-cached diagnostic function to verify that all datasets, models,
    and reports exist and conform to required specifications.
    """
    status = {
        "all_valid": True,
        "datasets": {},
        "models": {},
        "errors": []
    }

    # Data files to verify
    data_files = {
        "daily_air_quality": ("data/processed/daily_air_quality.csv", ["date", "location_name", "aqi"]),
        "ml_ready_dataset": ("data/processed/ml_ready_dataset.csv", ["date", "location_name", "AQI_t+1", "AQI_t+2", "AQI_t+3"]),
        "model_results": ("data/processed/model_results.csv", ["horizon", "model", "MAE", "RMSE", "R2"]),
        "test_predictions": ("data/processed/test_predictions.csv", ["date", "location_name", "aqi"]),
        "feature_importance": ("data/processed/feature_importance.csv", ["feature", "mean_importance"])
    }

    for name, (path, req_cols) in data_files.items():
        if not os.path.exists(path):
            status["all_valid"] = False
            status["errors"].append(f"Missing file: {path}")
            status["datasets"][name] = {"exists": False}
            continue

        try:
            df = pd.read_csv(path)
            missing = [c for c in req_cols if c not in df.columns]
            if missing:
                status["all_valid"] = False
                status["errors"].append(f"File {path} missing columns: {missing}")
                status["datasets"][name] = {"exists": True, "valid": False, "error": f"Missing cols: {missing}"}
            else:
                status["datasets"][name] = {
                    "exists": True,
                    "valid": True,
                    "rows": len(df),
                    "cols": len(df.columns),
                    "size_kb": round(os.path.getsize(path) / 1024, 1)
                }
        except Exception as e:
            status["all_valid"] = False
            status["errors"].append(f"Error reading {path}: {str(e)}")
            status["datasets"][name] = {"exists": True, "valid": False, "error": str(e)}

    # Model files to verify
    model_files = [
        "linear_regression_day1.joblib",
        "random_forest_day1.joblib",
        "linear_regression_day2.joblib",
        "random_forest_day2.joblib",
        "linear_regression_day3.joblib",
        "random_forest_day3.joblib"
    ]

    for mf in model_files:
        path = os.path.join("models", mf)
        if not os.path.exists(path):
            status["all_valid"] = False
            status["errors"].append(f"Missing model: {path}")
            status["models"][mf] = {"exists": False}
        else:
            size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
            status["models"][mf] = {"exists": True, "size_mb": size_mb}

    return status
