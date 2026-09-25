"""Generate H1-H12 future demand forecasts with the tuned XGBoost pipeline."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "demand"
    / "models"
    / "global_xgboost_tuned_pipeline.joblib"
)
METADATA_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "demand"
    / "reports"
    / "xgboost_tuning_metadata.json"
)
FUTURE_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demand"
    / "future_horizon_features.csv"
)
OUTPUT_FILE = (
    PROJECT_ROOT
    / "artifacts"
    / "demand"
    / "predictions"
    / "xgboost_tuned_future_forecast_h1_h12.csv"
)

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


def require_files():
    for path in (MODEL_FILE, METADATA_FILE, FUTURE_FEATURE_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")


def main():
    require_files()
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]

    try:
        pipeline = joblib.load(MODEL_FILE)
    except (AttributeError, ModuleNotFoundError, ValueError) as error:
        raise RuntimeError(
            "Could not load the tuned demand pipeline. Install the versions in "
            "requirements.txt and rerun this script."
        ) from error

    future = pd.read_csv(
        FUTURE_FEATURE_FILE,
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
    future["forecast_source"] = "tuned_xgboost_demand"

    output = future[
        [
            "forecast_origin_month",
            "target_month",
            "forecast_horizon",
            "crop",
            "location",
            "predicted_demand_kg",
            "forecast_source",
        ]
    ].copy()
    output["predicted_demand_kg"] = output["predicted_demand_kg"].round(4)
    output = output.sort_values(
        ["target_month", "crop", "location"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print("FUTURE DEMAND PREDICTION COMPLETE")
    print(f"Rows: {len(output):,}")
    print(
        f"Target range: {output['target_month'].min().date()} to "
        f"{output['target_month'].max().date()}"
    )
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

