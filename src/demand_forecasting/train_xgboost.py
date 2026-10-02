import argparse
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_model_data import (
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    select_model_columns,
    split_xy,
)


# Project paths

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# XGBoost settings

RANDOM_STATE = 42
XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "mae",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# Load chronological data splits

def load_splits(project_root, variant="core"):
    split_dir = project_root / "data" / "processed" / "demand" / variant / "model_splits"
    split_files = {
        "train": split_dir / "train.csv",
        "validation": split_dir / "validation.csv",
        "test": split_dir / "test.csv",
    }

    missing_files = [str(path) for path in split_files.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Model splits are missing:\n"
            + "\n".join(missing_files)
            + "\nRun src/modeling/prepare_model_data.py first."
        )

    splits = {
        name: pd.read_csv(
            path,
            parse_dates=["forecast_origin_month", "target_month"],
        )
        for name, path in split_files.items()
    }

    if not (
        splits["train"]["target_month"].max()
        < splits["validation"]["target_month"].min()
        <= splits["validation"]["target_month"].max()
        < splits["test"]["target_month"].min()
    ):
        raise ValueError("Chronological model splits overlap or are out of order.")

    return splits


# Build a complete preprocessing and XGBoost pipeline

def build_xgboost_pipeline(numeric_features, device="cpu"):
    try:
        from sklearn.pipeline import Pipeline
        from xgboost import XGBRegressor
    except ImportError as error:
        raise ImportError(
            "Training requires scikit-learn, xgboost, and joblib. "
            "Install the packages in requirements.txt first."
        ) from error

    params = XGBOOST_PARAMS.copy()
    params["device"] = device

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features)),
            ("model", XGBRegressor(**params)),
        ]
    )


# Calculate regression metrics

def calculate_metrics(actual, predicted):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual).sum()

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wape": float(np.abs(actual - predicted).sum() / denominator)
        if denominator > 0
        else None,
        "r2": float(r2_score(actual, predicted)),
    }


# Create prediction and metric tables

def create_prediction_table(source_data, predictions, split_name):
    output = source_data[
        [
            "forecast_origin_month",
            "target_month",
            "crop",
            "location",
            "forecast_horizon",
            TARGET_COLUMN,
        ]
    ].copy()
    output["xgboost_prediction"] = predictions
    output["split"] = split_name
    output["absolute_error"] = (
        output[TARGET_COLUMN] - output["xgboost_prediction"]
    ).abs()
    return output


def create_metrics_table(validation_predictions, test_predictions):
    records = []

    for split_name, predictions in (
        ("validation", validation_predictions),
        ("test", test_predictions),
    ):
        overall = calculate_metrics(
            predictions[TARGET_COLUMN], predictions["xgboost_prediction"]
        )
        records.append(
            {
                "model": "xgboost",
                "split": split_name,
                "forecast_horizon": "overall",
                "rows": len(predictions),
                **overall,
            }
        )

        for horizon, group in predictions.groupby("forecast_horizon"):
            horizon_metrics = calculate_metrics(
                group[TARGET_COLUMN], group["xgboost_prediction"]
            )
            records.append(
                {
                    "model": "xgboost",
                    "split": split_name,
                    "forecast_horizon": int(horizon),
                    "rows": len(group),
                    **horizon_metrics,
                }
            )

    return pd.DataFrame(records)


# Save local artifacts and optionally copy them to Google Drive

def save_artifacts(
    project_root,
    final_pipeline,
    validation_predictions,
    test_predictions,
    metrics,
    feature_columns,
    device,
    variant,
    drive_dir=None,
):
    import joblib
    import sklearn
    import xgboost

    artifact_root = project_root / "artifacts" / "demand" / variant
    model_dir = artifact_root / "models"
    prediction_dir = artifact_root / "predictions"
    report_dir = artifact_root / "reports"

    for directory in (model_dir, prediction_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pipeline_file = model_dir / "global_xgboost_pipeline.joblib"
    native_model_file = model_dir / "global_xgboost_model.json"
    validation_file = prediction_dir / "xgboost_validation_predictions.csv"
    test_file = prediction_dir / "xgboost_test_predictions.csv"
    metrics_file = report_dir / "xgboost_metrics.csv"
    metadata_file = report_dir / "xgboost_training_metadata.json"

    joblib.dump(final_pipeline, pipeline_file)
    final_pipeline.named_steps["model"].save_model(native_model_file)
    validation_predictions.to_csv(validation_file, index=False)
    test_predictions.to_csv(test_file, index=False)
    metrics.to_csv(metrics_file, index=False)

    metadata = {
        "model": "global_direct_multi_horizon_xgboost",
        "model_version": f"{variant}_v1",
        "model_variant": variant,
        "requires_air_quality": variant == "full",
        "requires_fama": False,
        "air_quality_used": variant == "full",
        "stock_used": False,
        "target": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "categorical_features": CATEGORICAL_FEATURES,
        "xgboost_parameters": {
            **XGBOOST_PARAMS,
            "device": device,
        },
        "final_fit_data": "train + validation",
        "validation_prediction_model": "train only",
        "test_prediction_model": "train + validation",
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    with metadata_file.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)

    local_files = [
        pipeline_file,
        native_model_file,
        validation_file,
        test_file,
        metrics_file,
        metadata_file,
    ]

    drive_files = []
    if drive_dir is not None:
        drive_dir = Path(drive_dir).expanduser().resolve()
        destination_map = {
            pipeline_file: drive_dir / "models" / pipeline_file.name,
            native_model_file: drive_dir / "models" / native_model_file.name,
            validation_file: drive_dir / "predictions" / validation_file.name,
            test_file: drive_dir / "predictions" / test_file.name,
            metrics_file: drive_dir / "reports" / metrics_file.name,
            metadata_file: drive_dir / "reports" / metadata_file.name,
        }

        for source, destination in destination_map.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            drive_files.append(destination)

    return local_files, drive_files


# Train, validate, refit, and test

def run_training(project_root=DEFAULT_PROJECT_ROOT, drive_dir=None, device="cpu", variant="core"):
    project_root = Path(project_root).expanduser().resolve()
    splits = load_splits(project_root, variant)

    feature_columns, numeric_features = select_model_columns(splits["train"])
    x_train, y_train = split_xy(splits["train"], feature_columns)
    x_validation, y_validation = split_xy(splits["validation"], feature_columns)
    x_test, _ = split_xy(splits["test"], feature_columns)

    print("Training validation-stage XGBoost pipeline on train data...")
    validation_pipeline = build_xgboost_pipeline(numeric_features, device=device)
    validation_pipeline.fit(x_train, y_train)
    validation_values = validation_pipeline.predict(x_validation)
    validation_predictions = create_prediction_table(
        splits["validation"], validation_values, "validation"
    )

    print("Refitting final XGBoost pipeline on train + validation data...")
    train_validation = pd.concat(
        [splits["train"], splits["validation"]], ignore_index=True
    ).sort_values("target_month")
    x_train_validation, y_train_validation = split_xy(
        train_validation, feature_columns
    )

    final_pipeline = build_xgboost_pipeline(numeric_features, device=device)
    final_pipeline.fit(x_train_validation, y_train_validation)
    test_values = final_pipeline.predict(x_test)
    test_predictions = create_prediction_table(splits["test"], test_values, "test")

    metrics = create_metrics_table(validation_predictions, test_predictions)
    local_files, drive_files = save_artifacts(
        project_root=project_root,
        final_pipeline=final_pipeline,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        metrics=metrics,
        feature_columns=feature_columns,
        device=device,
        variant=variant,
        drive_dir=drive_dir,
    )

    return {
        "pipeline": final_pipeline,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "metrics": metrics,
        "local_files": local_files,
        "drive_files": drive_files,
    }


# Command-line interface

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and save the global H1-H12 XGBoost pipeline."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Project root containing data/processed/demand/model_splits.",
    )
    parser.add_argument(
        "--variant", choices=["core", "full"], default="core",
        help="Model variant to train (default: core).",
    )
    parser.add_argument(
        "--drive-dir",
        type=Path,
        default=None,
        help="Optional Google Drive artifact directory.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="XGBoost training device.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_training(
        project_root=args.project_root,
        drive_dir=args.drive_dir,
        device=args.device,
        variant=args.variant,
    )

    print("\nXGBOOST TRAINING COMPLETE")
    print("\nOverall metrics:")
    print(
        results["metrics"]
        .query("forecast_horizon == 'overall'")
        .to_string(index=False)
    )
    print("\nLocal artifacts:")
    for path in results["local_files"]:
        print(f"  {path}")
    if results["drive_files"]:
        print("\nGoogle Drive artifacts:")
        for path in results["drive_files"]:
            print(f"  {path}")


if __name__ == "__main__":
    main()
