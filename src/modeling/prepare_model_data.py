import json
from pathlib import Path

import pandas as pd


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "horizon_training_dataset.csv"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "model_splits"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports"


# Split settings

VALIDATION_MONTHS = 3
TEST_MONTHS = 3
TARGET_COLUMN = "future_demand_kg"


# Feature settings

CATEGORICAL_FEATURES = [
    "crop",
    "location",
    "origin_season",
    "target_season",
]

IDENTIFIER_COLUMNS = [
    "forecast_origin_month",
    "target_month",
]

EXCLUDED_FEATURES = [
    *IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
]


# Load and validate the horizon dataset

def load_horizon_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Horizon training dataset cannot be found:\n{INPUT_FILE}\n"
            "Run the preprocessing pipeline through the horizons stage first."
        )

    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=["forecast_origin_month", "target_month"],
    )

    required_columns = [
        *IDENTIFIER_COLUMNS,
        *CATEGORICAL_FEATURES,
        "forecast_horizon",
        TARGET_COLUMN,
    ]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Horizon dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    if data[IDENTIFIER_COLUMNS].isna().any().any():
        raise ValueError("Forecast-origin or target-month values are missing.")

    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Training data contains rows without a valid future target.")

    if data.duplicated(
        ["forecast_origin_month", "crop", "location", "forecast_horizon"]
    ).any():
        raise ValueError("Training data contains duplicate horizon samples.")

    return data.sort_values(
        ["target_month", "forecast_origin_month", "crop", "location"]
    ).reset_index(drop=True)


# Select model inputs and target

def select_model_columns(data):
    feature_columns = [
        column for column in data.columns if column not in EXCLUDED_FEATURES
    ]
    numeric_features = [
        column for column in feature_columns if column not in CATEGORICAL_FEATURES
    ]

    unexpected_text_columns = [
        column
        for column in numeric_features
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if unexpected_text_columns:
        raise ValueError(
            "The following features need an explicit encoding policy: "
            + ", ".join(unexpected_text_columns)
        )

    return feature_columns, numeric_features


# Create chronological train, validation, and test sets

def create_chronological_split(data):
    final_target_month = data["target_month"].max().to_period("M").to_timestamp()
    test_start = final_target_month - pd.DateOffset(months=TEST_MONTHS - 1)
    validation_start = test_start - pd.DateOffset(months=VALIDATION_MONTHS)

    train = data[data["target_month"] < validation_start].copy()
    validation = data[
        data["target_month"].between(
            validation_start,
            test_start - pd.DateOffset(months=1),
        )
    ].copy()
    test = data[data["target_month"] >= test_start].copy()

    splits = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    for split_name, split_data in splits.items():
        if split_data.empty:
            raise ValueError(f"The {split_name} split is empty.")

        available_horizons = set(split_data["forecast_horizon"].unique())
        expected_horizons = set(range(1, 13))
        if available_horizons != expected_horizons:
            missing_horizons = sorted(expected_horizons - available_horizons)
            raise ValueError(
                f"The {split_name} split does not contain all H1-H12 horizons. "
                f"Missing: {missing_horizons}"
            )

    if not (
        train["target_month"].max()
        < validation["target_month"].min()
        <= validation["target_month"].max()
        < test["target_month"].min()
    ):
        raise ValueError("Chronological split boundaries overlap.")

    boundaries = {
        "train_start": train["target_month"].min().strftime("%Y-%m-%d"),
        "train_end": train["target_month"].max().strftime("%Y-%m-%d"),
        "validation_start": validation["target_month"].min().strftime("%Y-%m-%d"),
        "validation_end": validation["target_month"].max().strftime("%Y-%m-%d"),
        "test_start": test["target_month"].min().strftime("%Y-%m-%d"),
        "test_end": test["target_month"].max().strftime("%Y-%m-%d"),
    }

    return train, validation, test, boundaries


# Define reusable categorical and numeric preprocessing

def build_preprocessor(numeric_features):
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as error:
        raise ImportError(
            "scikit-learn is required to build the model preprocessor. "
            "Install the packages in requirements.txt before model training."
        ) from error

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# Separate predictors and labels

def split_xy(data, feature_columns):
    features = data[feature_columns].copy()
    target = data[TARGET_COLUMN].copy()
    return features, target


# Save reusable raw splits and split metadata

def save_outputs(train, validation, test, feature_columns, numeric_features, boundaries):
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(SPLIT_DIR / "train.csv", index=False)
    validation.to_csv(SPLIT_DIR / "validation.csv", index=False)
    test.to_csv(SPLIT_DIR / "test.csv", index=False)

    metadata = {
        "target_column": TARGET_COLUMN,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": numeric_features,
        "feature_columns": feature_columns,
        "validation_months": VALIDATION_MONTHS,
        "test_months": TEST_MONTHS,
        "boundaries": boundaries,
        "row_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
    }

    metadata_file = REPORT_DIR / "model_data_split.json"
    with metadata_file.open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)

    print(f"Saved: {SPLIT_DIR / 'train.csv'}")
    print(f"Saved: {SPLIT_DIR / 'validation.csv'}")
    print(f"Saved: {SPLIT_DIR / 'test.csv'}")
    print(f"Saved: {metadata_file}")


# Run model-data preparation

def main():
    print("Preparing model-ready chronological datasets...")

    data = load_horizon_data()
    feature_columns, numeric_features = select_model_columns(data)
    train, validation, test, boundaries = create_chronological_split(data)

    save_outputs(
        train,
        validation,
        test,
        feature_columns,
        numeric_features,
        boundaries,
    )

    print(f"Features: {len(feature_columns)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Train rows: {len(train):,} ({boundaries['train_start']} to {boundaries['train_end']})")
    print(
        "Validation rows: "
        f"{len(validation):,} "
        f"({boundaries['validation_start']} to {boundaries['validation_end']})"
    )
    print(f"Test rows: {len(test):,} ({boundaries['test_start']} to {boundaries['test_end']})")
    print("MODEL DATA PREPARATION COMPLETE")


if __name__ == "__main__":
    main()
