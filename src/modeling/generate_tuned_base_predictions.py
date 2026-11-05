"""Generate validation and test predictions for tuned stacking inputs."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "demand": {"folder": "demand", "target": "future_demand_kg", "dataset": "horizon_training_dataset.csv"},
    "price": {"folder": "price", "target": "future_weighted_avg_procurement_price_rm_per_kg", "dataset": "price_horizon_training_dataset.csv"},
    "supply": {"folder": "supply", "target": "future_production_kg", "dataset": "supply_horizon_training_dataset.csv"},
}


def load_splits(task, variant):
    config = CONFIG[task]
    directory = PROJECT_ROOT / "data" / "processed" / config["folder"] / variant / "model_splits"
    paths = {name: directory / f"{name}.csv" for name in ("validation", "test")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model splits:\n" + "\n".join(missing))
    return {
        name: pd.read_csv(path, parse_dates=["forecast_origin_month", "target_month"])
        for name, path in paths.items()
    }


def make_prediction_table(frame, predictions, target, model_name, split):
    columns = [
        "forecast_origin_month", "target_month", "crop", "location",
        "forecast_horizon", target,
    ]
    output = frame[columns].copy()
    output[f"{model_name}_prediction"] = np.maximum(np.asarray(predictions, dtype=float), 0)
    output["split"] = split
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=CONFIG, required=True)
    parser.add_argument("--variant", choices=("core", "full"), required=True)
    parser.add_argument("--xgb-model", type=Path)
    parser.add_argument("--etr-model", type=Path)
    args = parser.parse_args()

    config = CONFIG[args.task]
    root = PROJECT_ROOT / "artifacts" / config["folder"] / args.variant
    xgb_path = args.xgb_model or (root / "models" / "global_xgboost_tuned_pipeline.joblib")
    etr_path = args.etr_model or (root / "models" / "global_etr_tuned_pipeline.joblib")
    if not xgb_path.exists():
        raise FileNotFoundError(f"Tuned XGBoost pipeline not found: {xgb_path}")
    if not etr_path.exists():
        raise FileNotFoundError(f"Tuned ETR pipeline not found: {etr_path}")

    splits = load_splits(args.task, args.variant)
    xgb = joblib.load(xgb_path)
    etr = joblib.load(etr_path)
    target = config["target"]
    excluded = {"forecast_origin_month", "target_month", target}
    feature_columns = [column for column in splits["test"].columns if column not in excluded]

    for split_name, frame in splits.items():
        xgb_output = make_prediction_table(
            frame, xgb.predict(frame[feature_columns]), target, "xgboost_tuned", split_name
        )
        etr_output = make_prediction_table(
            frame, etr.predict(frame[feature_columns]), target, "etr_tuned", split_name
        )
        output = xgb_output.merge(
            etr_output[
                ["forecast_origin_month", "target_month", "crop", "location", "forecast_horizon", "etr_tuned_prediction"]
            ],
            on=["forecast_origin_month", "target_month", "crop", "location", "forecast_horizon"],
            validate="one_to_one",
        )
        output_dir = root / "predictions" / "tuned_base"
        output_dir.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_dir / f"tuned_base_{split_name}_predictions.csv", index=False)
        print(f"Saved: {output_dir / f'tuned_base_{split_name}_predictions.csv'}")


if __name__ == "__main__":
    main()
