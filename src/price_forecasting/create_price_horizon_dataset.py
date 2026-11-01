import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = PROJECT_ROOT / "data" / "processed" / "shared"
CONTEXTUAL_FILE = SHARED_DIR / "monthly_demand_contextual.csv"
HOLIDAY_FILE = PROJECT_ROOT / "data" / "raw" / "contextual" / "Contextual_Data.xlsx"

# Horizon and target settings

GROUP_COLUMNS = ["crop", "location"]
HORIZONS = range(1, 13)
PRICE_COLUMN = "weighted_avg_procurement_price_rm_per_kg"
TARGET_COLUMN = "future_weighted_avg_procurement_price_rm_per_kg"
TARGET_HOLIDAY_COLUMNS = [
    "public_holiday_count",
    "cny",
    "hari_raya",
    "gawai",
    "christmas",
    "major_festival",
]
SOLAR_COLUMN = "avg_daily_shortwave_radiation_mj_m2"
ALWAYS_REMOVED_MODEL_COLUMNS = [
    "supermarket_order_count",
    "trader_order_count",
    "total_stock_available_kg",
    "avg_stock_coverage_ratio",
    "stock_to_demand_ratio",
    "stock_available_lag_1",
    "stock_to_demand_ratio_lag_1",
]
AIR_QUALITY_COLUMNS = [
    "avg_api",
    "max_api",
    "moderate_or_worse_days",
    "unhealthy_days",
    "api_observation_days",
]
FAMA_FEATURE_PREFIX = "fama_"
VALID_VARIANTS = ("core", "full")
HOLIDAY_RENAME_MAP = {
    "Month": "month",
    "Public Holiday Count": "public_holiday_count",
    "CNY": "cny",
    "Hari Raya": "hari_raya",
    "Gawai": "gawai",
    "Christmas": "christmas",
    "Major Festival": "major_festival",
}


# Load feature, target-source, and future-known calendar data

def require_columns(data, required, dataset_name):
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


def variant_paths(variant):
    price_dir = PROJECT_ROOT / "data" / "processed" / "price" / variant
    return {
        "features": price_dir / "price_features.csv",
        "training_csv": price_dir / "price_horizon_training_dataset.csv",
        "training_parquet": price_dir / "price_horizon_training_dataset.parquet",
        "training_excel": price_dir / "price_horizon_training_dataset.xlsx",
        "prediction_template": price_dir / "future_price_horizon_features.csv",
    }


def load_inputs(variant):
    paths = variant_paths(variant)
    features_file = paths["features"]
    if not features_file.exists():
        raise FileNotFoundError(
            f"Price feature dataset cannot be found:\n{features_file}\n"
            "Run src/price_forecasting/create_price_features.py first."
        )
    if not CONTEXTUAL_FILE.exists():
        raise FileNotFoundError(
            f"Shared monthly contextual dataset cannot be found:\n{CONTEXTUAL_FILE}"
        )
    if not HOLIDAY_FILE.exists():
        raise FileNotFoundError(f"Holiday workbook cannot be found:\n{HOLIDAY_FILE}")

    features = pd.read_csv(features_file, parse_dates=["month"])
    contextual = pd.read_csv(CONTEXTUAL_FILE, parse_dates=["month"])
    holiday_calendar = pd.read_excel(
        HOLIDAY_FILE,
        sheet_name="Holidays",
        engine="openpyxl",
    )

    require_columns(
        features,
        ["month", *GROUP_COLUMNS, PRICE_COLUMN, "price_lag_12"],
        "Price feature dataset",
    )
    require_columns(
        contextual,
        [
            "month",
            *GROUP_COLUMNS,
            PRICE_COLUMN,
            "avg_temperature_c",
            "total_precipitation_mm",
            SOLAR_COLUMN,
            *TARGET_HOLIDAY_COLUMNS,
        ],
        "Shared monthly contextual dataset",
    )
    require_columns(holiday_calendar, list(HOLIDAY_RENAME_MAP), "Holiday calendar")

    removed_columns = list(ALWAYS_REMOVED_MODEL_COLUMNS)
    if variant == "core":
        removed_columns.extend(AIR_QUALITY_COLUMNS)
        removed_columns.extend(
            column for column in features if column.startswith(FAMA_FEATURE_PREFIX)
        )
    unexpected_removed_columns = [column for column in removed_columns if column in features]
    if unexpected_removed_columns:
        raise ValueError(
            "Removed fields are still present in the price feature dataset: "
            + ", ".join(unexpected_removed_columns)
            + ". Rerun create_price_features.py before creating horizons."
        )

    if features.duplicated(["month", *GROUP_COLUMNS]).any():
        raise ValueError("Price feature data contains duplicate crop/location/month rows.")
    if contextual.duplicated(["month", *GROUP_COLUMNS]).any():
        raise ValueError("Shared monthly data contains duplicate crop/location/month rows.")

    holiday_calendar = holiday_calendar[list(HOLIDAY_RENAME_MAP)].rename(
        columns=HOLIDAY_RENAME_MAP
    )
    holiday_calendar["month"] = pd.to_datetime(
        holiday_calendar["month"], errors="coerce"
    )
    if holiday_calendar["month"].isna().any():
        raise ValueError("Holiday calendar contains invalid month values.")
    holiday_calendar["month"] = (
        holiday_calendar["month"].dt.to_period("M").dt.to_timestamp()
    )
    if holiday_calendar.duplicated(["month"]).any():
        raise ValueError("Holiday calendar contains duplicate month rows.")

    return features, contextual, holiday_calendar


# Create target-month calendar fields known at the forecast origin

def target_season(month_number):
    if month_number in {11, 12, 1, 2, 3}:
        return "northeast_monsoon"
    if month_number in {6, 7, 8, 9}:
        return "southwest_monsoon"
    return "inter_monsoon"


def expand_horizons(features):
    horizon_table = pd.DataFrame({"forecast_horizon": list(HORIZONS)})
    expanded = features.merge(horizon_table, how="cross")

    origin_rename_map = {
        "month": "forecast_origin_month",
        PRICE_COLUMN: "origin_weighted_avg_procurement_price_rm_per_kg",
        "demand_kg": "origin_demand_kg",
        "order_count": "origin_order_count",
        "avg_temperature_c": "origin_avg_temperature_c",
        "avg_min_temperature_c": "origin_avg_min_temperature_c",
        "avg_max_temperature_c": "origin_avg_max_temperature_c",
        "total_precipitation_mm": "origin_total_precipitation_mm",
        SOLAR_COLUMN: f"origin_{SOLAR_COLUMN}",
        "weather_observation_days": "origin_weather_observation_days",
        "year": "origin_year",
        "month_number": "origin_month_number",
        "quarter": "origin_quarter",
        "season": "origin_season",
        "avg_api": "origin_avg_api",
        "max_api": "origin_max_api",
        "moderate_or_worse_days": "origin_moderate_or_worse_days",
        "unhealthy_days": "origin_unhealthy_days",
        "api_observation_days": "origin_api_observation_days",
        **{
            column: f"origin_{column}"
            for column in TARGET_HOLIDAY_COLUMNS
            if column in features.columns
        },
    }
    expanded = expanded.rename(
        columns={
            source: target
            for source, target in origin_rename_map.items()
            if source in expanded.columns
        }
    )

    expanded["target_month"] = (
        expanded["forecast_origin_month"].dt.to_period("M")
        + expanded["forecast_horizon"]
    ).dt.to_timestamp()
    expanded["target_year"] = expanded["target_month"].dt.year
    expanded["target_month_number"] = expanded["target_month"].dt.month
    expanded["target_quarter"] = expanded["target_month"].dt.quarter
    expanded["target_season"] = expanded["target_month_number"].map(target_season)
    expanded["_horizon_row_id"] = np.arange(len(expanded))
    return expanded


# Join only the future price target, never actual target-month predictors

def add_future_price(expanded, contextual):
    future_price = contextual[["month", *GROUP_COLUMNS, PRICE_COLUMN]].rename(
        columns={"month": "target_month", PRICE_COLUMN: TARGET_COLUMN}
    )
    if future_price.duplicated(["target_month", *GROUP_COLUMNS]).any():
        raise ValueError("Shared monthly data has duplicate future-price keys.")

    return expanded.merge(
        future_price,
        on=["target_month", *GROUP_COLUMNS],
        how="left",
        validate="many_to_one",
    )


# Add holidays because their dates are known before the target month

def add_target_holidays(expanded, holiday_calendar):
    holidays = holiday_calendar.rename(
        columns={
            "month": "target_month",
            **{
                column: f"target_{column}" for column in TARGET_HOLIDAY_COLUMNS
            },
        }
    )
    return expanded.merge(
        holidays,
        on="target_month",
        how="left",
        validate="many_to_one",
    )


# Target weather uses historical normals available by each origin month

def add_leakage_safe_weather_normals(expanded, contextual):
    weather = contextual[
        [
            "month",
            "location",
            "avg_temperature_c",
            "total_precipitation_mm",
            SOLAR_COLUMN,
        ]
    ].drop_duplicates()
    if weather.duplicated(["month", "location"]).any():
        raise ValueError("Shared monthly data has conflicting weather rows.")

    weather = weather.rename(columns={"month": "weather_month"})
    weather["weather_calendar_month"] = weather["weather_month"].dt.month
    candidates = expanded[
        [
            "_horizon_row_id",
            "forecast_origin_month",
            "location",
            "target_month_number",
        ]
    ].merge(
        weather,
        left_on=["location", "target_month_number"],
        right_on=["location", "weather_calendar_month"],
        how="left",
    )
    candidates = candidates[
        candidates["weather_month"] <= candidates["forecast_origin_month"]
    ]
    normals = candidates.groupby("_horizon_row_id", as_index=False).agg(
        target_month_normal_temperature=("avg_temperature_c", "mean"),
        target_month_normal_precipitation=("total_precipitation_mm", "mean"),
        target_month_normal_solar_radiation=(SOLAR_COLUMN, "mean"),
        target_month_weather_history_years=("weather_month", "count"),
    )
    return expanded.merge(
        normals,
        on="_horizon_row_id",
        how="left",
        validate="one_to_one",
    )


# Build training rows and the latest-origin future template

def build_horizon_dataset(features, contextual, holiday_calendar):
    expanded = expand_horizons(features)
    expanded = add_future_price(expanded, contextual)
    expanded = add_target_holidays(expanded, holiday_calendar)
    expanded = add_leakage_safe_weather_normals(expanded, contextual)
    expanded["target_month_normal_temperature"] = expanded[
        "target_month_normal_temperature"
    ].round(2)
    expanded["target_month_normal_precipitation"] = expanded[
        "target_month_normal_precipitation"
    ].round(2)
    expanded["target_month_normal_solar_radiation"] = expanded[
        "target_month_normal_solar_radiation"
    ].round(2)
    return expanded.drop(columns=["_horizon_row_id"]).sort_values(
        [*GROUP_COLUMNS, "forecast_origin_month", "forecast_horizon"]
    ).reset_index(drop=True)


def create_training_dataset(horizon_data):
    training = horizon_data[horizon_data[TARGET_COLUMN].notna()].copy()
    required_context = [
        "target_month_normal_temperature",
        "target_month_normal_precipitation",
        "target_month_normal_solar_radiation",
        *[f"target_{column}" for column in TARGET_HOLIDAY_COLUMNS],
    ]
    if training[required_context].isna().any().any():
        missing = training[required_context].isna().sum()
        missing = missing[missing > 0]
        raise ValueError(
            "Historical price rows have missing target context: "
            + ", ".join(f"{key}={value}" for key, value in missing.items())
        )
    return training


def create_prediction_template(horizon_data):
    latest_origin = horizon_data["forecast_origin_month"].max()
    return horizon_data[
        horizon_data["forecast_origin_month"].eq(latest_origin)
    ].reset_index(drop=True)


# Save price artifacts without mixing them with demand artifacts

def save_outputs(training, prediction_template, variant):
    paths = variant_paths(variant)
    paths["training_csv"].parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(paths["training_csv"], index=False)
    prediction_template.to_csv(paths["prediction_template"], index=False)

    excel_saved = True
    try:
        if len(training) <= 100_000:
            training.to_excel(paths["training_excel"], index=False)
        elif paths["training_excel"].exists():
            paths["training_excel"].unlink()
            print("Skipped Excel export because the horizon table exceeds 100,000 rows.")
    except OSError as error:
        excel_saved = False
        print(f"Excel skipped because the output file is unavailable: {error}")

    parquet_saved = True
    try:
        training.to_parquet(paths["training_parquet"], index=False)
    except ImportError:
        parquet_saved = False
        print("Parquet skipped: install pyarrow to enable Parquet output.")

    print(f"Price training rows: {len(training):,}")
    print(f"Future price prediction-template rows: {len(prediction_template):,}")
    print(f"Saved: {paths['training_csv']}")
    if excel_saved:
        if paths["training_excel"].exists():
            print(f"Saved: {paths['training_excel']}")
    print(f"Saved: {paths['prediction_template']}")
    if parquet_saved:
        print(f"Saved: {paths['training_parquet']}")


def main(variant="core"):
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Variant must be one of: {', '.join(VALID_VARIANTS)}")
    print(f"Creating price H1-H12 dataset ({variant} variant)...")
    features, contextual, holiday_calendar = load_inputs(variant)
    horizon_data = build_horizon_dataset(features, contextual, holiday_calendar)
    training = create_training_dataset(horizon_data)
    prediction_template = create_prediction_template(horizon_data)

    if training.empty:
        raise ValueError("No price training rows contain known future targets.")
    if set(training["forecast_horizon"].unique()) != set(HORIZONS):
        raise ValueError("Price training data does not contain every H1-H12 horizon.")
    if prediction_template[TARGET_COLUMN].notna().any():
        raise ValueError("Future price template unexpectedly contains actual future prices.")

    save_outputs(training, prediction_template, variant)

    missing_future_holidays = prediction_template[
        [f"target_{column}" for column in TARGET_HOLIDAY_COLUMNS]
    ].isna().any(axis=1).sum()
    if missing_future_holidays:
        print(
            "NOTE: Future price rows need holiday data for "
            f"{missing_future_holidays} rows before prediction."
        )
    else:
        print("Future price target-month holiday features are complete.")
    print("PRICE HORIZON DATASET CREATION COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VALID_VARIANTS, default="core")
    main(parser.parse_args().variant)
