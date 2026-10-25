import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Project paths and feature settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPLY_DIR = PROJECT_ROOT / "data" / "processed" / "supply"
MONTHLY_SUPPLY_FILE = SUPPLY_DIR / "shared" / "monthly_supply.csv"
CONTEXTUAL_FILE = PROJECT_ROOT / "data" / "raw" / "contextual" / "Contextual_Data.xlsx"

GROUP_COLUMNS = ["crop", "location"]
PRODUCTION_LAGS = [1, 3, 6, 12]
VALID_VARIANTS = ("core", "full")

WEATHER_COLUMNS = {
    "Month": "month",
    "Location/Region": "location",
    "Avg Temperature (°C)": "avg_temperature_c",
    "Avg Min Temperature (°C)": "avg_min_temperature_c",
    "Avg Max Temperature (°C)": "avg_max_temperature_c",
    "Total Precipitation (mm)": "total_precipitation_mm",
    "Avg Daily Shortwave Radiation (MJ/m²)": "avg_daily_shortwave_radiation_mj_m2",
    "Observation Days": "weather_observation_days",
}
AIR_QUALITY_COLUMNS = {
    "Month": "month",
    "Location/Region": "location",
    "Avg API": "avg_api",
    "Max API": "max_api",
    "Moderate or Worse Days": "moderate_or_worse_days",
    "Unhealthy Days": "unhealthy_days",
    "API Observation Days": "api_observation_days",
}
HOLIDAY_COLUMNS = {
    "Month": "month",
    "Public Holiday Count": "public_holiday_count",
    "CNY": "cny",
    "Hari Raya": "hari_raya",
    "Gawai": "gawai",
    "Christmas": "christmas",
    "Major Festival": "major_festival",
}


def output_paths(variant):
    output_dir = SUPPLY_DIR / variant
    return (
        output_dir / "supply_features.csv",
        output_dir / "supply_features.parquet",
        output_dir / "supply_features.xlsx",
    )


def require_columns(data, columns, label):
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def load_inputs(variant):
    if not MONTHLY_SUPPLY_FILE.exists():
        raise FileNotFoundError(
            f"Monthly supply data cannot be found:\n{MONTHLY_SUPPLY_FILE}\n"
            "Run prepare_supply_data.py first."
        )
    if not CONTEXTUAL_FILE.exists():
        raise FileNotFoundError(f"Contextual workbook cannot be found:\n{CONTEXTUAL_FILE}")

    supply = pd.read_csv(MONTHLY_SUPPLY_FILE, parse_dates=["month"])
    weather = pd.read_excel(CONTEXTUAL_FILE, sheet_name="Weather", engine="openpyxl")
    holidays = pd.read_excel(CONTEXTUAL_FILE, sheet_name="Holidays", engine="openpyxl")
    air_quality = None
    if variant == "full":
        air_quality = pd.read_excel(
            CONTEXTUAL_FILE, sheet_name="Air_Quality", engine="openpyxl"
        )

    require_columns(supply, ["month", "crop", "location", "production_kg"], "Supply")
    require_columns(weather, WEATHER_COLUMNS, "Weather")
    require_columns(holidays, HOLIDAY_COLUMNS, "Holidays")
    if air_quality is not None:
        require_columns(air_quality, AIR_QUALITY_COLUMNS, "Air quality")
    return supply, weather, holidays, air_quality


def normalize_context(data, rename_map, label):
    result = data[list(rename_map)].rename(columns=rename_map).copy()
    result["month"] = pd.to_datetime(result["month"], errors="coerce")
    if result["month"].isna().any():
        raise ValueError(f"{label} contains invalid month values.")
    result["month"] = result["month"].dt.to_period("M").dt.to_timestamp()
    if "location" in result:
        result["location"] = result["location"].astype("string").str.strip().str.title()
        keys = ["month", "location"]
    else:
        keys = ["month"]
    if result.duplicated(keys).any():
        raise ValueError(f"{label} contains duplicate {'/'.join(keys)} rows.")
    return result


def merge_context(supply, weather, holidays, air_quality=None):
    supply = supply.copy()
    supply["location"] = supply["location"].astype("string").str.strip().str.title()
    weather = normalize_context(weather, WEATHER_COLUMNS, "Weather")
    holidays = normalize_context(holidays, HOLIDAY_COLUMNS, "Holidays")
    merged = supply.merge(weather, on=["month", "location"], how="left", validate="many_to_one")
    merged = merged.merge(holidays, on="month", how="left", validate="many_to_one")
    required = [
        *[value for value in WEATHER_COLUMNS.values() if value not in {"month", "location"}],
        *[value for value in HOLIDAY_COLUMNS.values() if value != "month"],
    ]
    if air_quality is not None:
        air_quality = normalize_context(air_quality, AIR_QUALITY_COLUMNS, "Air quality")
        merged = merged.merge(
            air_quality, on=["month", "location"], how="left", validate="many_to_one"
        )
        required.extend(
            value for value in AIR_QUALITY_COLUMNS.values() if value not in {"month", "location"}
        )

    missing = merged[required].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        details = ", ".join(f"{key}={value}" for key, value in missing.items())
        raise ValueError("Supply context is incomplete after merging: " + details)
    return merged.sort_values([*GROUP_COLUMNS, "month"]).reset_index(drop=True)


def season_from_month(month_number):
    if month_number in {11, 12, 1, 2, 3}:
        return "northeast_monsoon"
    if month_number in {6, 7, 8, 9}:
        return "southwest_monsoon"
    return "inter_monsoon"


def create_features(data, variant):
    featured = data.copy()
    featured["year"] = featured["month"].dt.year
    featured["month_number"] = featured["month"].dt.month
    featured["quarter"] = featured["month"].dt.quarter
    featured["season"] = featured["month_number"].map(season_from_month)

    groups = featured.groupby(GROUP_COLUMNS, sort=False)["production_kg"]
    for lag in PRODUCTION_LAGS:
        featured[f"production_lag_{lag}"] = groups.shift(lag)
    prior = groups.shift(1)
    keys = [featured[column] for column in GROUP_COLUMNS]
    featured["rolling_production_mean_3"] = prior.groupby(keys).transform(
        lambda values: values.rolling(3).mean()
    )
    featured["rolling_production_mean_6"] = prior.groupby(keys).transform(
        lambda values: values.rolling(6).mean()
    )
    featured["production_change_1m"] = featured["production_lag_1"] - groups.shift(2)
    featured["production_pct_change_1m"] = np.where(
        groups.shift(2) != 0,
        featured["production_change_1m"] / groups.shift(2),
        np.nan,
    )

    if variant == "full":
        aq_groups = featured.groupby(GROUP_COLUMNS, sort=False)["avg_api"]
        featured["api_lag_1"] = aq_groups.shift(1)
        featured["rolling_avg_api_3m"] = aq_groups.shift(1).groupby(keys).transform(
            lambda values: values.rolling(3).mean()
        )
        unhealthy_groups = featured.groupby(GROUP_COLUMNS, sort=False)["unhealthy_days"]
        featured["rolling_unhealthy_days_3m"] = unhealthy_groups.shift(1).groupby(
            keys
        ).transform(lambda values: values.rolling(3).sum())
    return featured.replace([np.inf, -np.inf], np.nan)


def keep_model_ready_rows(featured, variant):
    required = [
        *[f"production_lag_{lag}" for lag in PRODUCTION_LAGS],
        "rolling_production_mean_3",
        "rolling_production_mean_6",
        "production_change_1m",
        "avg_temperature_c",
        "total_precipitation_mm",
        "avg_daily_shortwave_radiation_mj_m2",
    ]
    if variant == "full":
        required.extend(["avg_api", "api_lag_1", "rolling_avg_api_3m"])
    return featured.dropna(subset=required).copy(), required


def save_outputs(data, variant):
    output_csv, output_parquet, output_excel = output_paths(variant)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_csv, index=False, date_format="%Y-%m-%d")
    data.to_parquet(output_parquet, index=False)
    data.to_excel(output_excel, index=False)
    print(f"Supply feature rows: {len(data):,}")
    print(f"Saved: {output_csv}")
    print(f"Saved: {output_parquet}")
    print(f"Saved: {output_excel}")


def main(variant="core"):
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Variant must be one of: {', '.join(VALID_VARIANTS)}")
    print(f"Creating leakage-safe supply features ({variant} variant)...")
    supply, weather, holidays, air_quality = load_inputs(variant)
    contextual = merge_context(supply, weather, holidays, air_quality)
    featured = create_features(contextual, variant)
    model_ready, required = keep_model_ready_rows(featured, variant)
    print(f"Rows removed because 12-month history is unavailable: {len(featured) - len(model_ready):,}")
    if model_ready.empty:
        raise ValueError("No supply rows remain after applying history requirements.")
    if model_ready[required].isna().any().any():
        raise ValueError("Supply feature output still contains missing required history.")
    save_outputs(model_ready, variant)
    print("SUPPLY FEATURE CREATION COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VALID_VARIANTS, default="core")
    main(parser.parse_args().variant)
