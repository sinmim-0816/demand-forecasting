import unittest

import pandas as pd

from src.evaluation.group_bias import (
    calculate_metrics,
    classify_direction,
    consistently_weak_crops,
    summarize_groups,
)


class GroupBiasEvaluationTests(unittest.TestCase):
    def test_signed_error_direction(self):
        metrics = calculate_metrics([100, 120], [80, 100])
        self.assertEqual(metrics["mean_signed_error"], 20.0)
        self.assertEqual(classify_direction(metrics["normalized_bias"]), "underprediction")

    def test_group_summary_flags_high_wape_group(self):
        data = pd.DataFrame(
            {
                "crop": ["A"] * 4 + ["B"] * 4,
                "actual": [100] * 8,
                "prediction": [98, 102, 99, 101, 50, 55, 60, 65],
            }
        )
        overall = calculate_metrics(data["actual"], data["prediction"])["wape"]
        report = summarize_groups(
            data,
            ["crop"],
            "actual",
            "prediction",
            overall,
            min_group_rows=4,
        ).set_index("crop")
        self.assertFalse(bool(report.loc["A", "weak_group"]))
        self.assertTrue(bool(report.loc["B", "weak_group"]))

    def test_consistently_weak_crop_requires_repeated_months(self):
        crop_report = pd.DataFrame(
            {
                "crop": ["A", "B"],
                "rows": [30, 30],
                "wape": [0.30, 0.10],
                "normalized_bias": [0.20, 0.01],
                "bias_direction": ["underprediction", "balanced"],
            }
        )
        crop_month_report = pd.DataFrame(
            {
                "crop": ["A", "A", "A", "B", "B", "B"],
                "target_month_label": ["2025-10", "2025-11", "2025-12"] * 2,
                "wape": [0.30, 0.28, 0.12, 0.10, 0.11, 0.09],
            }
        )
        result = consistently_weak_crops(
            crop_report, crop_month_report, overall_wape=0.20
        ).set_index("crop")
        self.assertTrue(bool(result.loc["A", "consistently_weak"]))
        self.assertFalse(bool(result.loc["B", "consistently_weak"]))


if __name__ == "__main__":
    unittest.main()
