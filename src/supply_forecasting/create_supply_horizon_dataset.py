import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths and horizon settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPLY_DIR = PROJECT_ROOT / "data" / "processed" / "supply"
MONTHLY_SUPPLY_FILE = SUPPLY_DIR / "shared" / "monthly_supply.csv"
DEFAULT_CONTEXTUAL_FILE = PROJECT_ROOT / "data" / "raw" / "contextual" / "Contextual_Data.xlsx"
SUPPLY_CONTEXTUAL_FILE = PROJECT_ROOT / "data" / "raw" / "supply" / "Contextual_Data_10Years_Synthetic.xlsx"
HORIZONS = range(1, 13)
VALID_VARIANTS = ("core", "full")
SOLAR_COLUMN = "avg_daily_shortwave_radiation_mj_m2"
TARGET_HOLIDAY_COLUMNS = [
    "public_holiday_count",
    "cny",
    "hari_raya",
    "gawai",
    "christmas",
    "major_festival",
]

WEATHER_RENAME = {
    "Month": "month",
    "Location/Region": "location",
    "Avg Temperature (°C)": "avg_temperature_c",
    "Total Precipitation (mm)": "total_precipitation_mm",
    "Avg Daily Shortwave Radiation (MJ/m²)": SOLAR_COLUMN,
}
HOLIDAY_RENAME = {
    "Month": "month",
    "Public Holiday Count": "public_holiday_count",
    "CNY": "cny",
    "Hari Raya": "hari_raya",
    "Gawai": "gawai",
    "Christmas": "christmas",
    "Major Festival": "major_festival",
}


def variant_paths(variant):
    directory = SUPPLY_DIR / variant
    return {
        "features": directory / "supply_features.csv",
        "training_csv": directory / "supply_horizon_training_dataset.csv",
        "training_parquet": directory / "supply_horizon_training_dataset.parquet",
        "training_excel": directory / "supply_horizon_training_dataset.xlsx",
        "future": directory / "future_supply_horizon_features.csv",
    }


def require_columns(data, columns, label):
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def load_inputs(variant):
    paths = variant_paths(variant)
    if not paths["features"].exists():
        raise FileNotFoundError(
            f"Supply features cannot be found:\n{paths['features']}\n"
            "Run create_supply_features.py first."
        )
    contextual_file = SUPPLY_CONTEXTUAL_FILE if SUPPLY_CONTEXTUAL_FILE.exists() else DEFAULT_CONTEXTUAL_FILE
    if not MONTHLY_SUPPLY_FILE.exists() or not contextual_file.exists():
        raise FileNotFoundError("Monthly supply or contextual workbook is unavailable.")

    features = pd.read_csv(paths["features"], parse_dates=["month"])
    supply = pd.read_csv(MONTHLY_SUPPLY_FILE, parse_dates=["month"])
    weather = pd.read_excel(contextual_file, sheet_name="Weather", engine="openpyxl")
    holidays = pd.read_excel(contextual_file, sheet_name="Holidays", engine="openpyxl")
    require_columns(features, ["month", "crop", "location", "production_kg"], "Supply features")
    require_columns(supply, ["month", "crop", "location", "production_kg"], "Monthly supply")
    require_columns(weather, WEATHER_RENAME, "Weather")
    require_columns(holidays, HOLIDAY_RENAME, "Holidays")

    weather = weather[list(WEATHER_RENAME)].rename(columns=WEATHER_RENAME)
    holidays = holidays[list(HOLIDAY_RENAME)].rename(columns=HOLIDAY_RENAME)
    for frame in (weather, holidays):
        frame["month"] = pd.to_datetime(frame["month"], errors="coerce")
        if frame["month"].isna().any():
            raise ValueError("Contextual workbook contains invalid month values.")
        frame["month"] = frame["month"].dt.to_period("M").dt.to_timestamp()
    weather["location"] = weather["location"].astype("string").str.strip().str.title()
    if weather.duplicated(["month", "location"]).any():
        raise ValueError("Weather contains duplicate month/location rows.")
    if holidays.duplicated(["month"]).any():
        raise ValueError("Holiday calendar contains duplicate months.")
    return features, supply, weather, holidays


def season_from_month(month_number):
    if month_number in {11, 12, 1, 2, 3}:
        return "northeast_monsoon"
    if month_number in {6, 7, 8, 9}:
        return "southwest_monsoon"
    return "inter_monsoon"


def expand_horizons(features):
    expanded = features.merge(
        pd.DataFrame({"forecast_horizon": list(HORIZONS)}), how="cross"
    )
    rename = {
        "month": "forecast_origin_month",
        "production_kg": "origin_production_kg",
        "year": "origin_year",
        "month_number": "origin_month_number",
        "quarter": "origin_quarter",
        "season": "origin_season",
        "avg_temperature_c": "origin_avg_temperature_c",
        "avg_min_temperature_c": "origin_avg_min_temperature_c",
        "avg_max_temperature_c": "origin_avg_max_temperature_c",
        "total_precipitation_mm": "origin_total_precipitation_mm",
        SOLAR_COLUMN: f"origin_{SOLAR_COLUMN}",
        "weather_observation_days": "origin_weather_observation_days",
        "avg_api": "origin_avg_api",
        "max_api": "origin_max_api",
        "moderate_or_worse_days": "origin_moderate_or_worse_days",
        "unhealthy_days": "origin_unhealthy_days",
        "api_observation_days": "origin_api_observation_days",
        **{column: f"origin_{column}" for column in TARGET_HOLIDAY_COLUMNS},
    }
    expanded = expanded.rename(
        columns={key: value for key, value in rename.items() if key in expanded.columns}
    )
    expanded["target_month"] = (
        expanded["forecast_origin_month"].dt.to_period("M")
        + expanded["forecast_horizon"]
    ).dt.to_timestamp()
    expanded["target_year"] = expanded["target_month"].dt.year
    expanded["target_month_number"] = expanded["target_month"].dt.month
    expanded["target_quarter"] = expanded["target_month"].dt.quarter
    expanded["target_season"] = expanded["target_month_number"].map(season_from_month)
    expanded["_row_id"] = np.arange(len(expanded))
    return expanded


def add_target_production(expanded, monthly_supply):
    targets = monthly_supply[["month", "crop", "location", "production_kg"]].rename(
        columns={"month": "target_month", "production_kg": "future_production_kg"}
    )
    if targets.duplicated(["target_month", "crop", "location"]).any():
        raise ValueError("Monthly supply contains duplicate target keys.")
    return expanded.merge(
        targets,
        on=["target_month", "crop", "location"],
        how="left",
        validate="many_to_one",
    )


def add_target_holidays(expanded, holidays):
    target_holidays = holidays.rename(
        columns={
            "month": "target_month",
            **{column: f"target_{column}" for column in TARGET_HOLIDAY_COLUMNS},
        }
    )
    return expanded.merge(target_holidays, on="target_month", how="left", validate="many_to_one")


def add_weather_normals(expanded, weather):
    history = weather.rename(columns={"month": "weather_month"}).copy()
    history["weather_calendar_month"] = history["weather_month"].dt.month
    candidates = expanded[
        ["_row_id", "forecast_origin_month", "location", "target_month_number"]
    ].merge(
        history,
        left_on=["location", "target_month_number"],
        right_on=["location", "weather_calendar_month"],
        how="left",
    )
    candidates = candidates[
        candidates["weather_month"] <= candidates["forecast_origin_month"]
    ]
    normals = candidates.groupby("_row_id", as_index=False).agg(
        target_month_normal_temperature=("avg_temperature_c", "mean"),
        target_month_normal_precipitation=("total_precipitation_mm", "mean"),
        target_month_normal_solar_radiation=(SOLAR_COLUMN, "mean"),
        target_month_weather_history_years=("weather_month", "count"),
    )
    return expanded.merge(normals, on="_row_id", how="left", validate="one_to_one")


def build_horizon_dataset(features, supply, weather, holidays):
    expanded = expand_horizons(features)
    expanded = add_target_production(expanded, supply)
    expanded = add_target_holidays(expanded, holidays)
    expanded = add_weather_normals(expanded, weather)
    for column in [
        "target_month_normal_temperature",
        "target_month_normal_precipitation",
        "target_month_normal_solar_radiation",
    ]:
        expanded[column] = expanded[column].round(2)
    return expanded.drop(columns=["_row_id"]).sort_values(
        ["crop", "location", "forecast_origin_month", "forecast_horizon"]
    ).reset_index(drop=True)


def create_training_dataset(data):
    training = data[data["future_production_kg"].notna()].copy()
    required = [
        "target_month_normal_temperature",
        "target_month_normal_precipitation",
        "target_month_normal_solar_radiation",
        *[f"target_{column}" for column in TARGET_HOLIDAY_COLUMNS],
    ]
    if training.empty:
        raise ValueError("No known future production targets were created.")
    missing = training[required].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(
            "Historical supply targets have incomplete contextual data: "
            + ", ".join(f"{key}={value}" for key, value in missing.items())
        )
    return training


def save_outputs(training, future, variant):
    paths = variant_paths(variant)
    paths["training_csv"].parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(paths["training_csv"], index=False, date_format="%Y-%m-%d")
    training.to_parquet(paths["training_parquet"], index=False)
    # Large 10-year, multi-crop horizon tables can exceed Excel's practical
    # worksheet limit. CSV and Parquet remain the authoritative training files.
    if len(training) <= 100_000:
        training.to_excel(paths["training_excel"], index=False)
    elif paths["training_excel"].exists():
        paths["training_excel"].unlink()
        print("Skipped Excel export because the horizon table exceeds 100,000 rows.")
    future.to_csv(paths["future"], index=False, date_format="%Y-%m-%d")
    print(f"Training rows: {len(training):,}")
    print(f"Future prediction-template rows: {len(future):,}")
    for key in ("training_csv", "training_parquet", "training_excel", "future"):
        if paths[key].exists():
            print(f"Saved: {paths[key]}")


def main(variant="core"):
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Variant must be one of: {', '.join(VALID_VARIANTS)}")
    print(f"Creating supply H1-H12 dataset ({variant} variant)...")
    features, supply, weather, holidays = load_inputs(variant)
    horizon = build_horizon_dataset(features, supply, weather, holidays)
    training = create_training_dataset(horizon)
    latest_origin = horizon["forecast_origin_month"].max()
    future = horizon[horizon["forecast_origin_month"].eq(latest_origin)].copy()
    save_outputs(training, future, variant)
    missing_holidays = future[
        [f"target_{column}" for column in TARGET_HOLIDAY_COLUMNS]
    ].isna().any(axis=1).sum()
    if missing_holidays:
        print(f"NOTE: {missing_holidays:,} future rows still need target holiday coverage.")
    print("SUPPLY HORIZON DATASET CREATION COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VALID_VARIANTS, default="core")
    main(parser.parse_args().variant)
