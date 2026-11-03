"""Fine-tune the price XGBoost model using chronological validation."""

import argparse
import importlib.util
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42
TARGET_COLUMN = "future_weighted_avg_procurement_price_rm_per_kg"
CATEGORICAL_FEATURES = ["crop", "location", "origin_season", "target_season"]
SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.02, 0.03, 0.05],
    "min_child_weight": [3, 5, 8],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "n_estimators": [300, 500, 700],
}
BASE_PARAMS = {
    "objective": "reg:squarederror",
    "reg_alpha": 0.05,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "mae",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def price_prepare_module():
    source = Path(__file__).with_name("prepare_model_data.py")
    spec = importlib.util.spec_from_file_location("price_prepare_model_data", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_splits(variant):
    directory = PROJECT_ROOT / "data" / "processed" / "price" / variant / "model_splits"
    paths = {name: directory / f"{name}.csv" for name in ("train", "validation", "test")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing price splits:\n" + "\n".join(missing))
    return {name: pd.read_csv(path, parse_dates=["forecast_origin_month", "target_month"])
            for name, path in paths.items()}


def make_folds(data, folds, validation_months):
    months = sorted(data.target_month.unique())
    required = folds * validation_months + validation_months
    if len(months) < required:
        raise ValueError(f"Need at least {required} target months; found {len(months)}.")
    result = []
    for number in range(folds):
        end = len(months) - validation_months * (folds - number - 1)
        start = end - validation_months
        valid = data[data.target_month.between(months[start], months[end - 1])]
        train = data[data.target_month < months[start]]
        result.append((train, valid))
    return result


def score(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wape": float(np.abs(error).sum() / np.abs(actual).sum()),
        "r2": float(r2_score(actual, predicted)),
    }


def build_pipeline(numeric_features, params, device):
    from xgboost import XGBRegressor

    module = price_prepare_module()
    model_params = {**BASE_PARAMS, **params, "device": device}
    return Pipeline([
        ("preprocessor", module.build_preprocessor(numeric_features)),
        ("model", XGBRegressor(**model_params)),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("core", "full"), default="core")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--validation-months", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    splits = load_splits(args.variant)
    pretest = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    pretest = pretest.sort_values("target_month").reset_index(drop=True)
    excluded = {"forecast_origin_month", "target_month", TARGET_COLUMN}
    features = [column for column in pretest if column not in excluded]
    numeric = [column for column in features if column not in CATEGORICAL_FEATURES]
    folds = make_folds(pretest, args.folds, args.validation_months)
    candidates = [("current", {})]
    candidates += [(f"trial_{i:02d}", params) for i, params in enumerate(
        ParameterSampler(SEARCH_SPACE, n_iter=args.trials, random_state=RANDOM_STATE), 1
    )]

    summaries, fold_records = [], []
    for name, params in candidates:
        print(f"Evaluating {name} ({len(folds)} chronological folds)...", flush=True)
        fold_scores = []
        for number, (train, valid) in enumerate(folds, 1):
            pipeline = build_pipeline(numeric, params, args.device)
            pipeline.fit(train[features], train[TARGET_COLUMN])
            prediction = pipeline.predict(valid[features])
            fold_score = score(valid[TARGET_COLUMN], prediction)
            fold_scores.append(fold_score)
            fold_records.append({"candidate": name, "fold": number, **fold_score})
        summaries.append({
            "candidate": name, **params,
            "mean_fold_wape": float(np.mean([row["wape"] for row in fold_scores])),
            "pooled_mae": float(np.mean([row["mae"] for row in fold_scores])),
            "pooled_rmse": float(np.mean([row["rmse"] for row in fold_scores])),
            "pooled_wape": float(np.mean([row["wape"] for row in fold_scores])),
            "pooled_r2": float(np.mean([row["r2"] for row in fold_scores])),
        })

    leaderboard = pd.DataFrame(summaries).sort_values(
        ["mean_fold_wape", "pooled_mae", "candidate"]
    ).reset_index(drop=True)
    selected = leaderboard.iloc[0]["candidate"]
    selected_params = next(params for name, params in candidates if name == selected)
    final_pipeline = build_pipeline(numeric, selected_params, args.device)
    final_pipeline.fit(pretest[features], pretest[TARGET_COLUMN])
    test_prediction = np.maximum(final_pipeline.predict(splits["test"][features]), 0)
    test_metrics = score(splits["test"][TARGET_COLUMN], test_prediction)

    root = PROJECT_ROOT / "artifacts" / "price" / args.variant
    for folder in (root / "models", root / "reports", root / "predictions"):
        folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, root / "models" / "global_xgboost_tuned_pipeline.joblib")
    final_pipeline.named_steps["model"].save_model(
        root / "models" / "global_xgboost_tuned_model.json"
    )
    leaderboard.to_csv(root / "reports" / "xgboost_tuning_leaderboard.csv", index=False)
    pd.DataFrame(fold_records).to_csv(root / "reports" / "xgboost_tuning_folds.csv", index=False)
    output = splits["test"][["forecast_origin_month", "target_month", "crop", "location", "forecast_horizon", TARGET_COLUMN]].copy()
    output["xgboost_tuned_prediction"] = test_prediction
    output.to_csv(root / "predictions" / "xgboost_tuned_test_predictions.csv", index=False)
    metadata = {
        "model": "xgboost", "model_variant": args.variant,
        "selected_candidate": selected, "selected_parameters": {**BASE_PARAMS, **selected_params},
        "test_metrics": test_metrics, "feature_columns": features,
        "training_data": "train + validation", "folds": args.folds,
        "validation_months_per_fold": args.validation_months,
        "python_version": platform.python_version(),
    }
    (root / "reports" / "xgboost_tuning_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nPRICE XGBOOST TUNING COMPLETE")
    print(leaderboard.head(5).to_string(index=False))
    print(f"Selected: {selected}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
