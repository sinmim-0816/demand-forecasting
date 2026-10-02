"""Generate leakage-safe price forecasts with per-row full/core fallback."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_DIR = PROJECT_ROOT / "src" / "prediction"
if str(PREDICTION_DIR) not in sys.path:
    sys.path.insert(0, str(PREDICTION_DIR))

from model_variant_selector import select_price_variants


KEY_COLUMNS = [
    "forecast_origin_month",
    "target_month",
    "forecast_horizon",
    "crop",
    "location",
]
TARGET_COLUMN = "future_weighted_avg_procurement_price_rm_per_kg"
PREDICTION_COLUMN = "predicted_procurement_price_rm_per_kg"


def variant_paths(variant):
    artifact_root = PROJECT_ROOT / "artifacts" / "price" / variant
    return {
        "model": artifact_root / "models" / "global_xgboost_pipeline.joblib",
        "native_model": artifact_root / "models" / "global_xgboost_model.json",
        "metadata": artifact_root / "reports" / "xgboost_training_metadata.json",
        "features": PROJECT_ROOT / "data" / "processed" / "price" / variant / "future_price_horizon_features.csv",
        "training": PROJECT_ROOT / "data" / "processed" / "price" / variant / "price_horizon_training_dataset.csv",
    }


def rebuild_pipeline_from_native(variant, paths, metadata):
    """Recover across incompatible joblib/XGBoost versions using portable JSON."""

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBRegressor

    for key in ("native_model", "training"):
        if not paths[key].exists():
            raise FileNotFoundError(
                f"Cannot rebuild {variant} pipeline; required file is missing: {paths[key]}"
            )

    training = pd.read_csv(paths["training"], parse_dates=["target_month"])
    training_end = pd.Timestamp(metadata["training_data_end"])
    fit_data = training[training["target_month"] <= training_end].copy()
    feature_columns = metadata["feature_columns"]
    categorical_features = metadata["categorical_features"]
    numeric_features = [
        column for column in feature_columns if column not in categorical_features
    ]
    if fit_data.empty:
        raise ValueError("No rows are available to reconstruct the saved preprocessor.")

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(
        [
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.fit(fit_data[feature_columns])

    model = XGBRegressor()
    model.load_model(paths["native_model"])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def load_variant(variant):
    paths = variant_paths(variant)
    for key in ("model", "metadata", "features"):
        path = paths[key]
        if not path.exists():
            raise FileNotFoundError(f"Required {variant} price file not found: {path}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    features = pd.read_csv(
        paths["features"],
        parse_dates=["forecast_origin_month", "target_month"],
    )
    missing = sorted(set(metadata["feature_columns"]) - set(features.columns))
    if missing:
        raise ValueError(f"{variant} future features are missing: {missing}")
    try:
        pipeline = joblib.load(paths["model"])
    except Exception as error:
        print(
            f"WARNING: Could not deserialize the {variant} joblib pipeline "
            f"({type(error).__name__}). Rebuilding it from the native XGBoost model."
        )
        try:
            pipeline = rebuild_pipeline_from_native(variant, paths, metadata)
        except Exception as rebuild_error:
            raise RuntimeError(
                f"Could not load or reconstruct the {variant} price pipeline. "
                "Use the training package versions or retrain locally."
            ) from rebuild_error
    return pipeline, metadata, features


def align_templates(core, full):
    core_keys = core[KEY_COLUMNS].sort_values(KEY_COLUMNS).reset_index(drop=True)
    full_keys = full[KEY_COLUMNS].sort_values(KEY_COLUMNS).reset_index(drop=True)
    if not core_keys.equals(full_keys):
        raise ValueError("Core and full future price templates do not contain identical forecast keys.")


def predict_auto():
    core_pipeline, core_metadata, core = load_variant("core")
    full_pipeline, full_metadata, full = load_variant("full")
    align_templates(core, full)

    variants = select_price_variants(full, core_metadata, full_metadata)
    output = full[KEY_COLUMNS].copy()
    output["model_variant"] = variants
    output[PREDICTION_COLUMN] = np.nan

    for variant, pipeline, metadata, features in (
        ("core", core_pipeline, core_metadata, core),
        ("full", full_pipeline, full_metadata, full),
    ):
        mask = variants.eq(variant)
        if mask.any():
            values = pipeline.predict(features.loc[mask, metadata["feature_columns"]])
            output.loc[mask, PREDICTION_COLUMN] = np.maximum(values, 0.0)

    output["forecast_source"] = "xgboost_price_auto_full_core"
    return output


def predict_one(variant):
    pipeline, metadata, features = load_variant(variant)
    output = features[KEY_COLUMNS].copy()
    values = pipeline.predict(features[metadata["feature_columns"]])
    output[PREDICTION_COLUMN] = np.maximum(values, 0.0)
    output["model_variant"] = variant
    output["forecast_source"] = f"xgboost_price_{variant}"
    return output


def main(variant="auto"):
    output = predict_auto() if variant == "auto" else predict_one(variant)
    output[PREDICTION_COLUMN] = output[PREDICTION_COLUMN].round(4)
    output = output.sort_values(["target_month", "crop", "location"]).reset_index(drop=True)
    output_file = (
        PROJECT_ROOT
        / "artifacts"
        / "price"
        / "predictions"
        / "xgboost_future_price_forecast_h1_h12.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)

    print("FUTURE PRICE PREDICTION COMPLETE")
    print(output["model_variant"].value_counts().to_string())
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["auto", "core", "full"], default="auto")
    main(parser.parse_args().variant)
