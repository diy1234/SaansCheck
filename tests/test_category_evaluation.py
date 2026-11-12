import unittest

import numpy as np
import pandas as pd

from src.category_evaluation import (
    HISTORICAL_CATEGORY_ORDER,
    evaluate_regression_categories,
)


def _predictions():
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5, freq="D"),
            "actual_AQI_t+1": [50, 51, 101, 201, np.nan],
            "pred_lr_AQI_t+1": [51, 51, 201, 201, 120],
            "pred_rf_AQI_t+1": [50, 51, 101, 201, 120],
            "actual_AQI_t+2": [50, 51, np.nan, 201, 100],
            "pred_lr_AQI_t+2": [51, 51, 120, np.inf, 100],
            "pred_rf_AQI_t+2": [50, 101, 120, 200, 101],
            "actual_AQI_t+3": [50, 51, 101, 201, 301],
            "pred_lr_AQI_t+3": [50, 51, 101, 201, 301],
            "pred_rf_AQI_t+3": [51, 51, 201, 201, 301],
        }
    )


class CategoryEvaluationTests(unittest.TestCase):
    def test_confusion_matrix_order_and_known_metrics(self):
        result = evaluate_regression_categories(
            _predictions(), "Linear Regression", "Day+1"
        )

        self.assertEqual(result["categories"], HISTORICAL_CATEGORY_ORDER)
        self.assertEqual(result["confusion_matrix"].shape, (6, 6))
        self.assertEqual(
            result["confusion_matrix"].tolist(),
            [
                [0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        )
        self.assertAlmostEqual(result["report"]["macro avg"]["precision"], 1 / 6)
        self.assertAlmostEqual(result["report"]["macro avg"]["recall"], 1 / 3)
        self.assertAlmostEqual(result["report"]["macro avg"]["f1-score"], 2 / 9)
        self.assertAlmostEqual(result["report"]["weighted avg"]["f1-score"], 1 / 3)
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["excluded_count"], 1)

    def test_missing_and_non_finite_pairs_are_excluded(self):
        result = evaluate_regression_categories(
            _predictions(), "Linear Regression", "Day+2"
        )

        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["excluded_count"], 2)
        self.assertEqual(result["excluded_actual_count"], 1)
        self.assertEqual(result["excluded_prediction_count"], 1)
        self.assertEqual(int(result["per_class"]["Actual support"].sum()), 3)

    def test_preprocessing_threshold_boundaries_are_applied_consistently(self):
        boundaries = [50, 51, 100, 101, 150, 151, 200, 201, 300, 301, 400, 401, 500]
        predictions = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=len(boundaries)),
                "actual_AQI_t+1": boundaries,
                "pred_lr_AQI_t+1": boundaries,
            }
        )
        result = evaluate_regression_categories(
            predictions, "Linear Regression", "Day+1"
        )

        self.assertEqual(
            result["confusion_matrix"].diagonal().tolist(),
            [1, 2, 4, 2, 2, 2],
        )
        self.assertEqual(int(result["confusion_matrix"].sum()), len(boundaries))

    def test_all_six_classes_are_reported_including_zero_support(self):
        result = evaluate_regression_categories(
            _predictions(), "Linear Regression", "Day+1"
        )

        self.assertEqual(list(result["per_class"].index), list(HISTORICAL_CATEGORY_ORDER))
        self.assertEqual(result["per_class"].loc["Good", "Actual support"], 1)
        self.assertEqual(result["per_class"].loc["Very Poor", "Actual support"], 0)
        self.assertEqual(result["per_class"].loc["Severe", "Predicted count"], 0)
        self.assertEqual(result["report"]["Severe"]["f1-score"], 0)

    def test_model_and_horizon_use_their_own_predictions(self):
        predictions = _predictions()
        lr_day1 = evaluate_regression_categories(
            predictions, "Linear Regression", "Day+1"
        )
        rf_day1 = evaluate_regression_categories(
            predictions, "Random Forest", "Day+1"
        )
        lr_day3 = evaluate_regression_categories(
            predictions, "Linear Regression", "Day+3"
        )

        self.assertFalse(
            np.array_equal(
                lr_day1["confusion_matrix"], rf_day1["confusion_matrix"]
            )
        )
        self.assertFalse(
            np.array_equal(
                lr_day1["confusion_matrix"], lr_day3["confusion_matrix"]
            )
        )
        self.assertEqual(lr_day1["target_start"], pd.Timestamp("2023-01-02"))
        self.assertEqual(lr_day3["target_start"], pd.Timestamp("2023-01-04"))

    def test_invalid_selection_or_missing_artifact_columns_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_regression_categories(_predictions(), "Classifier", "Day+1")
        with self.assertRaises(ValueError):
            evaluate_regression_categories(_predictions(), "Linear Regression", "Day+4")
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            evaluate_regression_categories(pd.DataFrame(), "Linear Regression", "Day+1")


if __name__ == "__main__":
    unittest.main()
