import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from prepare_model_data import CATEGORICAL_FEATURES, TARGET_COLUMN, select_model_columns, split_xy
from train_xgboost import (
    XGBOOST_PARAMS,
    build_xgboost_pipeline,
    calculate_metrics,
    create_prediction_table,
    load_splits,
)


# Paths and reproducible search settings

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42
SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.02, 0.03, 0.05],
    "min_child_weight": [3, 5, 8],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "n_estimators": [300, 500, 700],
}


# Keep all rows for the same target month together in each fold

def make_rolling_folds(data, validation_months=3, folds=3):
    months = sorted(data["target_month"].unique())
    required = validation_months * (folds + 1)
    if len(months) < required:
        raise ValueError(
            f"Need at least {required} distinct pre-test target months for "
            f"{folds} folds of {validation_months} months; found {len(months)}."
        )

    result = []
    for fold_number in range(folds):
        end = len(months) - validation_months * (folds - fold_number - 1)
        start = end - validation_months
        validation_start = months[start]
        validation_end = months[end - 1]
        training = data[data["target_month"] < validation_start]
        validation = data[data["target_month"].between(validation_start, validation_end)]
        if training.empty or validation.empty:
            raise ValueError("A rolling fold has no training or validation rows.")
        result.append((training, validation))
    return result


# Fit the same full preprocessing pipeline used by regular XGBoost training

def fit_pipeline(training, feature_columns, numeric_features, params, device):
    pipeline = build_xgboost_pipeline(numeric_features, device=device)
    pipeline.set_params(**{f"model__{key}": value for key, value in params.items()})
    x_training, y_training = split_xy(training, feature_columns)
    pipeline.fit(x_training, y_training)
    return pipeline


# Search by average fold WAPE; also report pooled errors for readability

def evaluate_candidate(folds, feature_columns, numeric_features, params, device):
    fold_rows = []
    predictions = []
    for fold_number, (training, validation) in enumerate(folds, start=1):
        pipeline = fit_pipeline(training, feature_columns, numeric_features, params, device)
        x_validation, y_validation = split_xy(validation, feature_columns)
        predicted = pipeline.predict(x_validation)
        metrics = calculate_metrics(y_validation, predicted)
        fold_rows.append(
            {
                "fold": fold_number,
                "train_end": training["target_month"].max().strftime("%Y-%m-%d"),
                "validation_start": validation["target_month"].min().strftime("%Y-%m-%d"),
                "validation_end": validation["target_month"].max().strftime("%Y-%m-%d"),
                "rows": len(validation),
                **metrics,
            }
        )
        predictions.append((y_validation.to_numpy(), predicted))

    actual = np.concatenate([item[0] for item in predictions])
    predicted = np.concatenate([item[1] for item in predictions])
    pooled = calculate_metrics(actual, predicted)
    return fold_rows, pooled, float(np.mean([row["wape"] for row in fold_rows]))


# Tune without inspecting the test set, then refit for future prediction

def run_tuning(
    project_root=DEFAULT_PROJECT_ROOT,
    trials=12,
    folds=3,
    validation_months=3,
    device="cpu",
    evaluate_test=False,
):
    project_root = Path(project_root).expanduser().resolve()
    splits = load_splits(project_root)
    pretest = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    pretest = pretest.sort_values("target_month").reset_index(drop=True)
    feature_columns, numeric_features = select_model_columns(pretest)
    rolling_folds = make_rolling_folds(pretest, validation_months, folds)

    candidates = [("current", {})]
    candidates.extend(
        (f"trial_{index:02d}", params)
        for index, params in enumerate(
            ParameterSampler(SEARCH_SPACE, n_iter=trials, random_state=RANDOM_STATE),
            start=1,
        )
    )

    summaries = []
    fold_records = []
    for candidate_name, params in candidates:
        print(f"Evaluating {candidate_name} ({len(rolling_folds)} rolling folds)...", flush=True)
        fold_rows, pooled, mean_fold_wape = evaluate_candidate(
            rolling_folds, feature_columns, numeric_features, params, device
        )
        summaries.append(
            {
                "candidate": candidate_name,
                **{key: params.get(key, XGBOOST_PARAMS[key]) for key in SEARCH_SPACE},
                "mean_fold_wape": mean_fold_wape,
                "pooled_mae": pooled["mae"],
                "pooled_rmse": pooled["rmse"],
                "pooled_wape": pooled["wape"],
                "pooled_r2": pooled["r2"],
            }
        )
        fold_records.extend({"candidate": candidate_name, **row} for row in fold_rows)

    leaderboard = pd.DataFrame(summaries).sort_values(
        ["mean_fold_wape", "pooled_mae", "candidate"]
    ).reset_index(drop=True)
    best_name = leaderboard.iloc[0]["candidate"]
    best_params = dict(next(params for name, params in candidates if name == best_name))
    print(f"Selected {best_name}; refitting on all pre-test months...", flush=True)
    final_pipeline = fit_pipeline(
        pretest, feature_columns, numeric_features, best_params, device
    )

    model_dir = project_root / "artifacts" / "demand" / "models"
    report_dir = project_root / "artifacts" / "demand" / "reports"
    prediction_dir = project_root / "artifacts" / "demand" / "predictions"
    for directory in (model_dir, report_dir, prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_file = model_dir / "global_xgboost_tuned_pipeline.joblib"
    native_file = model_dir / "global_xgboost_tuned_model.json"
    leaderboard_file = report_dir / "xgboost_tuning_leaderboard.csv"
    folds_file = report_dir / "xgboost_tuning_folds.csv"
    metadata_file = report_dir / "xgboost_tuning_metadata.json"
    joblib.dump(final_pipeline, model_file)
    final_pipeline.named_steps["model"].save_model(native_file)
    leaderboard.to_csv(leaderboard_file, index=False)
    pd.DataFrame(fold_records).to_csv(folds_file, index=False)

    metadata = {
        "selected_candidate": best_name,
        "selected_parameters": {**XGBOOST_PARAMS, **best_params, "device": device},
        "selection_metric": "mean_fold_wape",
        "folds": folds,
        "validation_months_per_fold": validation_months,
        "training_data": "train + validation splits only",
        "test_evaluated": evaluate_test,
        "target_column": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "categorical_features": CATEGORICAL_FEATURES,
        "python_version": platform.python_version(),
    }

    test_metrics = None
    test_file = None
    if evaluate_test:
        x_test, y_test = split_xy(splits["test"], feature_columns)
        test_values = final_pipeline.predict(x_test)
        test_metrics = calculate_metrics(y_test, test_values)
        test_predictions = create_prediction_table(splits["test"], test_values, "test")
        test_predictions = test_predictions.rename(
            columns={"xgboost_prediction": "xgboost_tuned_prediction"}
        )
        test_predictions["absolute_error"] = (
            test_predictions[TARGET_COLUMN] - test_predictions["xgboost_tuned_prediction"]
        ).abs()
        test_file = prediction_dir / "xgboost_tuned_test_predictions.csv"
        test_predictions.to_csv(test_file, index=False)
        metadata["test_metrics"] = test_metrics

    with metadata_file.open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)

    return {
        "leaderboard": leaderboard,
        "fold_results": pd.DataFrame(fold_records),
        "selected_candidate": best_name,
        "test_metrics": test_metrics,
        "files": [model_file, native_file, leaderboard_file, folds_file, metadata_file]
        + ([test_file] if test_file else []),
    }


# Command-line interface

def parse_args():
    parser = argparse.ArgumentParser(description="Tune the global H1-H12 XGBoost model.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--validation-months", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the held-out test split once after tuning; do not use it to select settings.",
    )
    args = parser.parse_args()
    if args.trials < 1 or args.folds < 1 or args.validation_months < 1:
        parser.error("--trials, --folds, and --validation-months must be positive.")
    return args


def main():
    args = parse_args()
    result = run_tuning(
        project_root=args.project_root,
        trials=args.trials,
        folds=args.folds,
        validation_months=args.validation_months,
        device=args.device,
        evaluate_test=args.evaluate_test,
    )
    print("\nXGBOOST TUNING COMPLETE")
    print(result["leaderboard"].head(5).to_string(index=False))
    print(f"\nSelected: {result['selected_candidate']}")
    if result["test_metrics"] is not None:
        print(f"Test metrics: {result['test_metrics']}")
    for path in result["files"]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
