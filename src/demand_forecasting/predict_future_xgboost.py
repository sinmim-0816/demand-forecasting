"""Generate H1-H12 future demand forecasts with the tuned XGBoost pipeline."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTION_DIR = PROJECT_ROOT / "src" / "prediction"
if str(PREDICTION_DIR) not in sys.path:
    sys.path.insert(0, str(PREDICTION_DIR))

from model_variant_selector import select_demand_variant
REQUIRED_FUTURE_CONTEXT = [
    "target_public_holiday_count",
    "target_cny",
    "target_hari_raya",
    "target_gawai",
    "target_christmas",
    "target_major_festival",
    "target_month_normal_temperature",
    "target_month_normal_precipitation",
    "target_month_normal_solar_radiation",
]


def variant_paths(variant):
    artifact_root = PROJECT_ROOT / "artifacts" / "demand" / variant
    tuned_model = artifact_root / "models" / "global_xgboost_tuned_pipeline.joblib"
    tuned_metadata = artifact_root / "reports" / "xgboost_tuning_metadata.json"
    if tuned_model.exists() and tuned_metadata.exists():
        model_file, metadata_file, model_name = tuned_model, tuned_metadata, "tuned_xgboost"
    else:
        model_file = artifact_root / "models" / "global_xgboost_pipeline.joblib"
        metadata_file = artifact_root / "reports" / "xgboost_training_metadata.json"
        model_name = "xgboost"
    return {
        "model": model_file,
        "metadata": metadata_file,
        "features": PROJECT_ROOT / "data" / "processed" / "demand" / variant / "future_horizon_features.csv",
        "output": artifact_root / "predictions" / f"{model_name}_future_forecast_h1_h12.csv",
        "model_name": model_name,
    }


def require_files(paths):
    for path in (paths["model"], paths["metadata"], paths["features"]):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")


def resolve_variant(requested_variant):
    if requested_variant != "auto":
        return requested_variant, []
    core_paths = variant_paths("core")
    full_paths = variant_paths("full")
    require_files(core_paths)
    try:
        require_files(full_paths)
    except FileNotFoundError:
        return "core", ["Full demand artifacts are unavailable; core model selected."]
    core_metadata = json.loads(core_paths["metadata"].read_text(encoding="utf-8"))
    full_metadata = json.loads(full_paths["metadata"].read_text(encoding="utf-8"))
    full_features = pd.read_csv(full_paths["features"])
    decision = select_demand_variant(full_features, core_metadata, full_metadata)
    return decision.variant, decision.warnings


def main(variant="auto"):
    variant, warnings = resolve_variant(variant)
    paths = variant_paths(variant)
    require_files(paths)
    model_file = paths["model"]
    metadata_file = paths["metadata"]
    future_feature_file = paths["features"]
    output_file = paths["output"]
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]

    try:
        pipeline = joblib.load(model_file)
    except (AttributeError, ModuleNotFoundError, ValueError) as error:
        raise RuntimeError(
            "Could not load the tuned demand pipeline. Install the versions in "
            "requirements.txt and rerun this script."
        ) from error

    future = pd.read_csv(
        future_feature_file,
        parse_dates=["forecast_origin_month", "target_month"],
    )
    missing_features = sorted(set(feature_columns) - set(future.columns))
    if missing_features:
        raise ValueError(f"Future features are missing columns: {missing_features}")

    missing_context = future[REQUIRED_FUTURE_CONTEXT].isna().sum()
    missing_context = missing_context[missing_context > 0]
    if not missing_context.empty:
        raise ValueError(
            "Future calendar or weather-normal context is incomplete: "
            f"{missing_context.to_dict()}"
        )

    expected_rows = future[["crop", "location"]].drop_duplicates().shape[0] * 12
    if len(future) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} crop/location/H1-H12 rows, found {len(future)}."
        )

    predicted = pipeline.predict(future[feature_columns])
    future["predicted_demand_kg"] = np.maximum(predicted, 0.0)
    future["forecast_source"] = f"{paths['model_name']}_demand_{variant}"
    future["model_variant"] = variant

    output = future[
        [
            "forecast_origin_month",
            "target_month",
            "forecast_horizon",
            "crop",
            "location",
            "predicted_demand_kg",
            "forecast_source",
            "model_variant",
        ]
    ].copy()
    output["predicted_demand_kg"] = output["predicted_demand_kg"].round(4)
    output = output.sort_values(
        ["target_month", "crop", "location"]
    ).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)
    stable_output_file = (
        PROJECT_ROOT
        / "artifacts"
        / "demand"
        / "predictions"
        / "xgboost_tuned_future_forecast_h1_h12.csv"
    )
    stable_output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(stable_output_file, index=False)

    print("FUTURE DEMAND PREDICTION COMPLETE")
    print(f"Rows: {len(output):,}")
    print(
        f"Target range: {output['target_month'].min().date()} to "
        f"{output['target_month'].max().date()}"
    )
    print(f"Saved: {output_file}")
    print(f"Published for crop recommendation: {stable_output_file}")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["auto", "core", "full"], default="auto")
    main(parser.parse_args().variant)

