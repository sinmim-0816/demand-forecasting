import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Project files

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEY_COLUMNS = [
    "forecast_origin_month",
    "target_month",
    "crop",
    "location",
    "forecast_horizon",
]
TARGET_COLUMN = "future_demand_kg"
PREDICTION_FILES = {
    "original": ("xgboost_test_predictions.csv", "xgboost_prediction"),
    "tuned": ("xgboost_tuned_test_predictions.csv", "xgboost_tuned_prediction"),
}


# Compute comparable metrics for one group

def group_metrics(group, prediction_column):
    actual = group[TARGET_COLUMN].to_numpy(dtype=float)
    predicted = group[prediction_column].to_numpy(dtype=float)
    error = actual - predicted
    denominator = np.abs(actual).sum()
    squared_total = np.square(actual - actual.mean()).sum()
    return {
        "rows": len(group),
        "mae_kg": float(np.abs(error).mean()),
        "rmse_kg": float(np.sqrt(np.square(error).mean())),
        "wape": float(np.abs(error).sum() / denominator) if denominator else np.nan,
        "r2": float(1 - np.square(error).sum() / squared_total)
        if squared_total
        else np.nan,
        "mean_error_kg": float(error.mean()),
    }


# Match the exact same test rows before comparing model errors

def load_predictions(project_root):
    prediction_dir = project_root / "artifacts" / "predictions"
    tables = []
    for variant, (filename, prediction_column) in PREDICTION_FILES.items():
        path = prediction_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {variant} test predictions: {path}")
        data = pd.read_csv(path, parse_dates=KEY_COLUMNS[:2])
        required = KEY_COLUMNS + [TARGET_COLUMN, prediction_column]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        if data.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{path} contains duplicate test row keys.")
        tables.append(data[required].rename(columns={prediction_column: variant}))

    joined = tables[0].merge(
        tables[1], on=KEY_COLUMNS, how="outer", validate="one_to_one",
        suffixes=("_original", "_tuned"), indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Original and tuned predictions do not cover the same test rows.")
    joined = joined.drop(columns="_merge")
    original_target = joined[f"{TARGET_COLUMN}_original"]
    tuned_target = joined[f"{TARGET_COLUMN}_tuned"]
    if not np.allclose(original_target, tuned_target, rtol=0, atol=1e-8):
        raise ValueError("Original and tuned test targets differ; comparison is invalid.")
    return joined.rename(columns={f"{TARGET_COLUMN}_original": TARGET_COLUMN}).drop(
        columns=f"{TARGET_COLUMN}_tuned"
    )


# Summarize overall, then separately by crop, location, and forecast horizon

def build_report(joined):
    records = []
    dimensions = [("overall", None)] + [
        (column, column) for column in ("crop", "location", "forecast_horizon")
    ]
    for dimension_name, column in dimensions:
        groups = [("all", joined)] if column is None else joined.groupby(column)
        for value, group in groups:
            original = group_metrics(group, "original")
            tuned = group_metrics(group, "tuned")
            records.append(
                {
                    "dimension": dimension_name,
                    "value": value,
                    "rows": len(group),
                    "original_mae_kg": original["mae_kg"],
                    "tuned_mae_kg": tuned["mae_kg"],
                    "mae_improvement_kg": original["mae_kg"] - tuned["mae_kg"],
                    "original_wape": original["wape"],
                    "tuned_wape": tuned["wape"],
                    "wape_improvement_points": 100
                    * (original["wape"] - tuned["wape"]),
                    "original_rmse_kg": original["rmse_kg"],
                    "tuned_rmse_kg": tuned["rmse_kg"],
                    "original_r2": original["r2"],
                    "tuned_r2": tuned["r2"],
                    "original_mean_error_kg": original["mean_error_kg"],
                    "tuned_mean_error_kg": tuned["mean_error_kg"],
                }
            )
    return pd.DataFrame(records)


# Command-line interface

def main():
    parser = argparse.ArgumentParser(
        description="Compare original and tuned XGBoost test errors by group."
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()

    joined = load_predictions(project_root)
    report = build_report(joined)
    output_path = project_root / "artifacts" / "reports" / "xgboost_error_breakdown.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    print("Overall test comparison:")
    print(report.loc[report["dimension"] == "overall"].to_string(index=False))
    for dimension in ("crop", "location", "forecast_horizon"):
        subset = report.loc[report["dimension"] == dimension]
        worst = subset.sort_values("tuned_wape", ascending=False).head(5)
        print(f"\nHighest tuned WAPE by {dimension}:")
        print(worst[["value", "rows", "tuned_wape", "wape_improvement_points"]].to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
