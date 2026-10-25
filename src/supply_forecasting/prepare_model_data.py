import argparse
import json
from pathlib import Path

import pandas as pd


# Model-data contract

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_MONTHS = 3
TEST_MONTHS = 3
TARGET_COLUMN = "future_production_kg"
VALID_VARIANTS = ("core", "full")
CATEGORICAL_FEATURES = ["crop", "location", "origin_season", "target_season"]
IDENTIFIER_COLUMNS = ["forecast_origin_month", "target_month"]
EXCLUDED_FEATURES = [*IDENTIFIER_COLUMNS, TARGET_COLUMN]
AIR_QUALITY_PREFIXES = ("origin_avg_api", "origin_max_api", "origin_moderate", "origin_unhealthy", "origin_api_observation", "api_lag_", "rolling_avg_api_", "rolling_unhealthy_")


def variant_paths(variant):
    processed = PROJECT_ROOT / "data" / "processed" / "supply" / variant
    reports = PROJECT_ROOT / "artifacts" / "supply" / variant / "reports"
    return processed / "supply_horizon_training_dataset.csv", processed / "model_splits", reports


def load_horizon_data(variant):
    input_file, _, _ = variant_paths(variant)
    if not input_file.exists():
        raise FileNotFoundError(
            f"Supply horizon dataset cannot be found:\n{input_file}\n"
            "Run the supply pipeline through the horizons stage first."
        )
    data = pd.read_csv(input_file, parse_dates=IDENTIFIER_COLUMNS)
    required = [*IDENTIFIER_COLUMNS, *CATEGORICAL_FEATURES, "forecast_horizon", TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Supply horizon data is missing: " + ", ".join(missing))
    if data[required].isna().any().any():
        raise ValueError("Supply training data contains missing required values.")
    if data.duplicated(["forecast_origin_month", "crop", "location", "forecast_horizon"]).any():
        raise ValueError("Supply training data contains duplicate horizon rows.")
    if variant == "core":
        unexpected = [column for column in data if column.startswith(AIR_QUALITY_PREFIXES)]
        if unexpected:
            raise ValueError("Core supply data unexpectedly contains air-quality fields.")
    return data.sort_values(["target_month", "forecast_origin_month", "crop", "location"]).reset_index(drop=True)


def select_model_columns(data):
    feature_columns = [column for column in data if column not in EXCLUDED_FEATURES]
    numeric_features = [column for column in feature_columns if column not in CATEGORICAL_FEATURES]
    unexpected = [
        column for column in numeric_features
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if unexpected:
        raise ValueError("Supply features need an encoding policy: " + ", ".join(unexpected))
    return feature_columns, numeric_features


def create_chronological_split(data):
    final_month = data["target_month"].max().to_period("M").to_timestamp()
    test_start = final_month - pd.DateOffset(months=TEST_MONTHS - 1)
    validation_start = test_start - pd.DateOffset(months=VALIDATION_MONTHS)
    train = data[data["target_month"] < validation_start].copy()
    validation = data[data["target_month"].between(validation_start, test_start - pd.DateOffset(months=1))].copy()
    test = data[data["target_month"] >= test_start].copy()
    splits = {"train": train, "validation": validation, "test": test}
    for name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"The supply {name} split is empty.")
        available = set(frame["forecast_horizon"].astype(int).unique())
        if available != set(range(1, 13)):
            raise ValueError(f"The supply {name} split does not contain every H1-H12 horizon.")
    if not (train["target_month"].max() < validation["target_month"].min() <= validation["target_month"].max() < test["target_month"].min()):
        raise ValueError("Supply chronological split boundaries overlap.")
    boundaries = {
        "train_start": train["target_month"].min().strftime("%Y-%m-%d"),
        "train_end": train["target_month"].max().strftime("%Y-%m-%d"),
        "validation_start": validation["target_month"].min().strftime("%Y-%m-%d"),
        "validation_end": validation["target_month"].max().strftime("%Y-%m-%d"),
        "test_start": test["target_month"].min().strftime("%Y-%m-%d"),
        "test_end": test["target_month"].max().strftime("%Y-%m-%d"),
    }
    return train, validation, test, boundaries


def save_outputs(train, validation, test, feature_columns, numeric_features, boundaries, variant):
    _, split_dir, report_dir = variant_paths(variant)
    split_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame.to_csv(split_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    metadata = {
        "target_column": TARGET_COLUMN,
        "model_variant": variant,
        "requires_air_quality": variant == "full",
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": numeric_features,
        "feature_columns": feature_columns,
        "validation_months": VALIDATION_MONTHS,
        "test_months": TEST_MONTHS,
        "boundaries": boundaries,
        "row_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
    }
    metadata_file = report_dir / "model_data_split.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved splits: {split_dir}")
    print(f"Saved: {metadata_file}")


def main(variant="core"):
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Variant must be one of: {', '.join(VALID_VARIANTS)}")
    print(f"Preparing supply chronological splits ({variant} variant)...")
    data = load_horizon_data(variant)
    feature_columns, numeric_features = select_model_columns(data)
    train, validation, test, boundaries = create_chronological_split(data)
    save_outputs(train, validation, test, feature_columns, numeric_features, boundaries, variant)
    print(f"Features: {len(feature_columns)}")
    for name, frame in (("Train", train), ("Validation", validation), ("Test", test)):
        print(f"{name} rows: {len(frame):,}")
    print("SUPPLY MODEL DATA PREPARATION COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VALID_VARIANTS, default="core")
    main(parser.parse_args().variant)
