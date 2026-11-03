"""Tune Extra Trees for demand, price, or supply horizon datasets.

The test period is never used to select parameters. Candidates are selected
using rolling chronological folds built from train + validation rows, then the
selected pipeline is refit on all pre-test rows.
"""

import argparse
import importlib.util
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42

TASKS = {
    "demand": {
        "target": "future_demand_kg",
        "dataset": "horizon_training_dataset.csv",
        "folder": "demand",
    },
    "price": {
        "target": "future_weighted_avg_procurement_price_rm_per_kg",
        "dataset": "price_horizon_training_dataset.csv",
        "folder": "price",
    },
    "supply": {
        "target": "future_production_kg",
        "dataset": "supply_horizon_training_dataset.csv",
        "folder": "supply",
    },
}

CATEGORICAL_FEATURES = ["crop", "location", "origin_season", "target_season"]
SEARCH_SPACE = {
    "n_estimators": [300, 500, 700],
    "max_depth": [None, 12, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [0.6, 0.8, 1.0],
}


def load_preprocessor(task, numeric_features):
    """Reuse the task's production preprocessing contract."""
    source = PROJECT_ROOT / "src" / f"{task}_forecasting" / "prepare_model_data.py"
    spec = importlib.util.spec_from_file_location(f"{task}_prepare_model_data", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "build_preprocessor"):
        return module.build_preprocessor(numeric_features)
    # Supply uses the same contract but keeps its preprocessing implementation
    # in the Colab notebook, so provide the equivalent production-safe encoder.
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer([
        ("categorical", categorical, CATEGORICAL_FEATURES),
        ("numeric", numeric, numeric_features),
    ], remainder="drop", verbose_feature_names_out=False)


def load_splits(task, variant):
    folder = PROJECT_ROOT / "data" / "processed" / TASKS[task]["folder"] / variant / "model_splits"
    paths = {name: folder / f"{name}.csv" for name in ("train", "validation", "test")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing chronological splits:\n" + "\n".join(missing))
    return {
        name: pd.read_csv(path, parse_dates=["forecast_origin_month", "target_month"])
        for name, path in paths.items()
    }


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual).sum()
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wape": float(np.abs(actual - predicted).sum() / denominator),
        "r2": float(r2_score(actual, predicted)),
    }


def rolling_folds(data, validation_months, folds):
    months = sorted(data["target_month"].unique())
    required = validation_months * (folds + 1)
    if len(months) < required:
        raise ValueError(f"Need {required} target months, found {len(months)}.")
    result = []
    for index in range(folds):
        end = len(months) - validation_months * (folds - index - 1)
        start = end - validation_months
        start_month, end_month = months[start], months[end - 1]
        train = data[data.target_month < start_month]
        valid = data[data.target_month.between(start_month, end_month)]
        result.append((train, valid))
    return result


def make_pipeline(task, numeric_features, params):
    model = ExtraTreesRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        criterion="squared_error",
        **params,
    )
    return Pipeline(
        [("preprocessor", load_preprocessor(task, numeric_features)), ("model", model)]
    )


def run(args):
    config = TASKS[args.task]
    target = config["target"]
    splits = load_splits(args.task, args.variant)
    pretest = pd.concat([splits["train"], splits["validation"]], ignore_index=True)
    pretest = pretest.sort_values("target_month").reset_index(drop=True)
    excluded = {"forecast_origin_month", "target_month", target}
    feature_columns = [column for column in pretest.columns if column not in excluded]
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    folds = rolling_folds(pretest, args.validation_months, args.folds)
    candidates = [("current", {})]
    candidates += [(f"trial_{i:02d}", params) for i, params in enumerate(
        ParameterSampler(SEARCH_SPACE, n_iter=args.trials, random_state=RANDOM_STATE), 1
    )]

    leaderboard, fold_rows = [], []
    for name, params in candidates:
        print(f"Evaluating {name} ({len(folds)} chronological folds)...", flush=True)
        predictions, fold_metrics = [], []
        for fold_number, (train, valid) in enumerate(folds, 1):
            pipeline = make_pipeline(args.task, numeric, params)
            pipeline.fit(train[feature_columns], train[target])
            predicted = pipeline.predict(valid[feature_columns])
            actual = valid[target].to_numpy()
            fold_metric = metrics(actual, predicted)
            fold_metrics.append(fold_metric)
            predictions.extend(zip(actual, predicted))
            fold_rows.append({
                "candidate": name, "fold": fold_number,
                "rows": len(valid), **fold_metric,
            })
        actual = np.array([row[0] for row in predictions])
        predicted = np.array([row[1] for row in predictions])
        pooled = metrics(actual, predicted)
        leaderboard.append({
            "candidate": name, **params,
            "mean_fold_wape": float(np.mean([row["wape"] for row in fold_metrics])),
            "pooled_mae": pooled["mae"], "pooled_rmse": pooled["rmse"],
            "pooled_wape": pooled["wape"], "pooled_r2": pooled["r2"],
        })

    leaderboard = pd.DataFrame(leaderboard).sort_values(
        ["mean_fold_wape", "pooled_mae", "candidate"]
    ).reset_index(drop=True)
    selected = leaderboard.iloc[0]["candidate"]
    selected_params = next(params for name, params in candidates if name == selected)
    final_pipeline = make_pipeline(args.task, numeric, selected_params)
    final_pipeline.fit(pretest[feature_columns], pretest[target])
    test = splits["test"]
    test_prediction = np.maximum(final_pipeline.predict(test[feature_columns]), 0)
    test_metrics = metrics(test[target], test_prediction)

    root = PROJECT_ROOT / "artifacts" / config["folder"] / args.variant
    for directory in (root / "models", root / "reports", root / "predictions"):
        directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, root / "models" / "global_etr_tuned_pipeline.joblib")
    leaderboard.to_csv(root / "reports" / "etr_tuning_leaderboard.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(root / "reports" / "etr_tuning_folds.csv", index=False)
    output = test[["forecast_origin_month", "target_month", "crop", "location", "forecast_horizon", target]].copy()
    output["etr_tuned_prediction"] = test_prediction
    output.to_csv(root / "predictions" / "etr_tuned_test_predictions.csv", index=False)
    metadata = {
        "task": args.task, "model": "extra_trees", "model_variant": args.variant,
        "selected_candidate": selected, "selected_parameters": selected_params,
        "test_metrics": test_metrics, "feature_columns": feature_columns,
        "training_data": "train + validation", "folds": args.folds,
        "validation_months_per_fold": args.validation_months,
        "python_version": platform.python_version(),
    }
    (root / "reports" / "etr_tuning_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\n{args.task.upper()} ETR TUNING COMPLETE ({args.variant})")
    print(leaderboard.head(5).to_string(index=False))
    print(f"Selected: {selected}")
    print(f"Test metrics: {test_metrics}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--variant", choices=("core", "full"), default="core")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--validation-months", type=int, default=3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
