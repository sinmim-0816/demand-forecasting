from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


KEY_COLUMNS = [
    "forecast_origin_month",
    "target_month",
    "crop",
    "location",
    "forecast_horizon",
]


@dataclass(frozen=True)
class EvaluationConfig:
    domain: str
    target_column: str
    prediction_column: str
    prediction_filename: str
    unit: str


CONFIGS = {
    "demand": EvaluationConfig(
        domain="demand",
        target_column="future_demand_kg",
        prediction_column="xgboost_tuned_prediction",
        prediction_filename="xgboost_tuned_test_predictions.csv",
        unit="kg",
    ),
    "price": EvaluationConfig(
        domain="price",
        target_column="future_weighted_avg_procurement_price_rm_per_kg",
        prediction_column="xgboost_prediction",
        prediction_filename="xgboost_test_predictions.csv",
        unit="RM/kg",
    ),
}


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    signed_error = actual - predicted
    absolute_actual_total = np.abs(actual).sum()
    actual_total = actual.sum()
    squared_total = np.square(actual - actual.mean()).sum()
    return {
        "rows": int(len(actual)),
        "mean_actual": float(actual.mean()),
        "mean_prediction": float(predicted.mean()),
        "mae": float(np.abs(signed_error).mean()),
        "rmse": float(np.sqrt(np.square(signed_error).mean())),
        "wape": (
            float(np.abs(signed_error).sum() / absolute_actual_total)
            if absolute_actual_total
            else np.nan
        ),
        "mean_signed_error": float(signed_error.mean()),
        "median_signed_error": float(np.median(signed_error)),
        "normalized_bias": (
            float(signed_error.sum() / actual_total) if actual_total else np.nan
        ),
        "r2": (
            float(1 - np.square(signed_error).sum() / squared_total)
            if squared_total
            else np.nan
        ),
    }


def classify_direction(normalized_bias, threshold=0.05):
    if pd.isna(normalized_bias) or abs(normalized_bias) < threshold:
        return "balanced"
    return "underprediction" if normalized_bias > 0 else "overprediction"


def summarize_groups(
    data,
    group_columns,
    target_column,
    prediction_column,
    overall_wape,
    min_group_rows=12,
):
    records = []
    group_key = group_columns[0] if len(group_columns) == 1 else group_columns
    for values, group in data.groupby(group_key, dropna=False, observed=True):
        values = values if isinstance(values, tuple) else (values,)
        record = dict(zip(group_columns, values))
        record.update(calculate_metrics(group[target_column], group[prediction_column]))
        records.append(record)

    report = pd.DataFrame(records)
    if report.empty:
        return report

    eligible = report.loc[report["rows"] >= min_group_rows, "wape"].dropna()
    dimension_threshold = float(eligible.quantile(0.75)) if not eligible.empty else np.nan
    weak_threshold = max(overall_wape, dimension_threshold) if not np.isnan(
        dimension_threshold
    ) else overall_wape
    report["weak_threshold_wape"] = weak_threshold
    report["weak_group"] = (
        (report["rows"] >= min_group_rows)
        & (report["wape"] >= weak_threshold)
        & (report["wape"] > overall_wape)
    )
    report["bias_direction"] = report["normalized_bias"].map(classify_direction)
    return report.sort_values(["weak_group", "wape"], ascending=[False, False])


def consistently_weak_crops(crop_report, crop_month_report, overall_wape):
    monthly = crop_month_report.assign(
        worse_than_overall=crop_month_report["wape"] > overall_wape
    )
    consistency = (
        monthly.groupby("crop", observed=True)
        .agg(
            target_months_evaluated=("target_month_label", "nunique"),
            months_worse_than_overall=("worse_than_overall", "sum"),
        )
        .reset_index()
    )
    consistency["proportion_months_worse"] = (
        consistency["months_worse_than_overall"]
        / consistency["target_months_evaluated"]
    )
    consistency = consistency.merge(
        crop_report[["crop", "rows", "wape", "normalized_bias", "bias_direction"]],
        on="crop",
        how="left",
        validate="one_to_one",
    ).rename(columns={"rows": "total_rows", "wape": "overall_crop_wape"})
    consistency["consistently_weak"] = (
        (consistency["target_months_evaluated"] >= 2)
        & (consistency["proportion_months_worse"] >= (2 / 3))
        & (consistency["overall_crop_wape"] > overall_wape)
    )
    return consistency.sort_values(
        ["consistently_weak", "overall_crop_wape"], ascending=[False, False]
    )
