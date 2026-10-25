from pathlib import Path
import sys

import pandas as pd


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from pipeline_config import get_supply_settings

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "supply" / "shared"
OUTPUT_CSV = OUTPUT_DIR / "monthly_supply.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "monthly_supply.parquet"
OUTPUT_EXCEL = OUTPUT_DIR / "monthly_supply.xlsx"


# Load the configured raw supply file without assuming a specific file format

def load_raw_supply(input_file, sheet_name=None):
    if not input_file.exists():
        raise FileNotFoundError(
            f"Supply history cannot be found:\n{input_file}\n"
            "Add the real or synthetic file described in docs/SUPPLY_FORECASTING.md, "
            "or update supply_data.file in pipeline_config.json."
        )

    suffix = input_file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_file)
    if suffix == ".parquet":
        return pd.read_parquet(input_file)
    if suffix in {".xlsx", ".xls"}:
        engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
        return pd.read_excel(input_file, sheet_name=sheet_name or 0, engine=engine)
    raise ValueError("Supply data must be CSV, Parquet, XLSX, or XLS.")


# Standardize raw records and aggregate them to one crop-location-month row

def prepare_monthly_supply(raw, settings):
    source_columns = {
        settings["date_column"]: "month",
        settings["crop_column"]: "crop",
        settings["location_column"]: "location",
        settings["production_column"]: "production_kg",
    }
    missing = [column for column in source_columns if column not in raw.columns]
    if missing:
        raise ValueError("Supply data is missing columns: " + ", ".join(missing))

    supply = raw[list(source_columns)].rename(columns=source_columns).copy()
    supply["month"] = pd.to_datetime(supply["month"], errors="coerce")
    supply["production_kg"] = pd.to_numeric(
        supply["production_kg"], errors="coerce"
    )
    supply["crop"] = supply["crop"].astype("string").str.strip()
    supply["location"] = (
        supply["location"].astype("string").str.strip().str.title()
    )

    if supply["month"].isna().any():
        raise ValueError("Supply data contains missing or invalid dates.")
    if supply[["crop", "location"]].isna().any().any():
        raise ValueError("Supply data contains missing crop or location values.")
    if supply[["crop", "location"]].eq("").any().any():
        raise ValueError("Supply data contains blank crop or location values.")
    if supply["production_kg"].isna().any():
        raise ValueError("Supply data contains missing or invalid production values.")
    if (supply["production_kg"] < 0).any():
        raise ValueError("Production quantity cannot be negative.")

    supply["month"] = supply["month"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        supply.groupby(["month", "crop", "location"], as_index=False)
        .agg(
            production_kg=("production_kg", "sum"),
            source_record_count=("production_kg", "size"),
        )
        .sort_values(["crop", "location", "month"])
        .reset_index(drop=True)
    )
    return monthly


# Missing months must not be silently interpreted as zero production

def validate_monthly_series(monthly):
    if monthly.empty:
        raise ValueError("No monthly supply rows remain after preparation.")
    if monthly.duplicated(["month", "crop", "location"]).any():
        raise ValueError("Monthly supply contains duplicate crop/location/month rows.")

    incomplete = []
    for (crop, location), group in monthly.groupby(
        ["crop", "location"], sort=False
    ):
        actual = pd.DatetimeIndex(group["month"]).sort_values()
        expected = pd.date_range(actual.min(), actual.max(), freq="MS")
        missing = expected.difference(actual)
        if len(missing):
            incomplete.append(f"{crop}/{location}: {len(missing)} missing month(s)")
    if incomplete:
        raise ValueError(
            "Supply lags require consecutive monthly records. Missing months must be "
            "confirmed and added as real zero-production rows or supplied as observations: "
            + "; ".join(incomplete[:10])
        )


def save_outputs(monthly):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_CSV, index=False, date_format="%Y-%m-%d")
    monthly.to_excel(OUTPUT_EXCEL, index=False)
    monthly.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Monthly supply rows: {len(monthly):,}")
    print(f"Crops: {monthly['crop'].nunique():,}")
    print(f"Locations: {monthly['location'].nunique():,}")
    print(f"Coverage: {monthly['month'].min():%Y-%m} to {monthly['month'].max():%Y-%m}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_EXCEL}")
    print(f"Saved: {OUTPUT_PARQUET}")


def main():
    print("Preparing monthly crop supply data...")
    input_file, settings = get_supply_settings()
    raw = load_raw_supply(input_file, settings["sheet"])
    monthly = prepare_monthly_supply(raw, settings)
    validate_monthly_series(monthly)
    save_outputs(monthly)
    print("MONTHLY SUPPLY PREPARATION COMPLETE")


if __name__ == "__main__":
    main()
