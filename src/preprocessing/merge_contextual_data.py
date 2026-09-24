from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "shared"
MONTHLY_DEMAND_FILE = PROCESSED_DIR / "monthly_demand.csv"
CONTEXTUAL_FILE = PROJECT_ROOT / "data" / "raw" / "contextual" / "Contextual_Data.xlsx"

OUTPUT_CSV = PROCESSED_DIR / "monthly_demand_contextual.csv"
OUTPUT_PARQUET = PROCESSED_DIR / "monthly_demand_contextual.parquet"
OUTPUT_EXCEL = PROCESSED_DIR / "monthly_demand_contextual.xlsx"

WEATHER_RENAME_MAP = {
    "Month": "month",
    "Location/Region": "location",
    "Avg Temperature (°C)": "avg_temperature_c",
    "Avg Min Temperature (°C)": "avg_min_temperature_c",
    "Avg Max Temperature (°C)": "avg_max_temperature_c",
    "Total Precipitation (mm)": "total_precipitation_mm",
    "Observation Days": "weather_observation_days",
}

HOLIDAY_RENAME_MAP = {
    "Month": "month",
    "Public Holiday Count": "public_holiday_count",
    "CNY": "cny",
    "Hari Raya": "hari_raya",
    "Gawai": "gawai",
    "Christmas": "christmas",
    "Major Festival": "major_festival",
}


def require_columns(df, required, dataset_name):
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


def load_inputs():
    if not MONTHLY_DEMAND_FILE.exists():
        raise FileNotFoundError(
            f"Monthly demand file cannot be found:\n{MONTHLY_DEMAND_FILE}"
        )
    if not CONTEXTUAL_FILE.exists():
        raise FileNotFoundError(
            f"Contextual workbook cannot be found:\n{CONTEXTUAL_FILE}"
        )

    demand = pd.read_csv(MONTHLY_DEMAND_FILE)
    weather = pd.read_excel(CONTEXTUAL_FILE, sheet_name="Weather", engine="openpyxl")
    holidays = pd.read_excel(
        CONTEXTUAL_FILE, sheet_name="Holidays", engine="openpyxl"
    )

    require_columns(demand, ["month", "crop", "location", "demand_kg"], "Demand")
    require_columns(weather, WEATHER_RENAME_MAP, "Weather")
    require_columns(holidays, HOLIDAY_RENAME_MAP, "Holidays")

    return demand, weather, holidays


def prepare_inputs(demand, weather, holidays):
    demand = demand.copy()
    weather = weather[list(WEATHER_RENAME_MAP)].rename(columns=WEATHER_RENAME_MAP)
    holidays = holidays[list(HOLIDAY_RENAME_MAP)].rename(columns=HOLIDAY_RENAME_MAP)

    demand["month"] = pd.to_datetime(demand["month"], errors="coerce")
    weather["month"] = pd.to_datetime(weather["month"], errors="coerce")
    holidays["month"] = pd.to_datetime(holidays["month"], errors="coerce")

    for name, df in (("Demand", demand), ("Weather", weather), ("Holidays", holidays)):
        if df["month"].isna().any():
            raise ValueError(f"{name} contains invalid month values.")
        df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()

    demand["location"] = demand["location"].astype("string").str.strip().str.title()
    weather["location"] = weather["location"].astype("string").str.strip().str.title()

    if demand.duplicated(["month", "crop", "location"]).any():
        raise ValueError("Demand has duplicate Month + Crop + Location rows.")
    if weather.duplicated(["month", "location"]).any():
        raise ValueError("Weather has duplicate Month + Location rows.")
    if holidays.duplicated(["month"]).any():
        raise ValueError("Holidays has duplicate Month rows.")

    return demand, weather, holidays


def merge_contextual_data(demand, weather, holidays):
    merged = demand.merge(
        weather,
        on=["month", "location"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        holidays,
        on="month",
        how="left",
        validate="many_to_one",
    )

    contextual_columns = [
        *[column for column in WEATHER_RENAME_MAP.values() if column not in {"month", "location"}],
        *[column for column in HOLIDAY_RENAME_MAP.values() if column != "month"],
    ]
    missing_counts = merged[contextual_columns].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        details = ", ".join(
            f"{column}={count}" for column, count in missing_counts.items()
        )
        raise ValueError(
            "Contextual data is incomplete after the left joins: " + details
        )

    merged = merged.sort_values(["crop", "location", "month"]).reset_index(drop=True)
    return merged


def save_outputs(merged):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    merged.to_excel(OUTPUT_EXCEL, index=False)

    parquet_saved = True
    try:
        merged.to_parquet(OUTPUT_PARQUET, index=False)
    except ImportError:
        parquet_saved = False
        print("Parquet skipped: install pyarrow to enable Parquet output.")

    print(f"Merged rows: {len(merged):,}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_EXCEL}")
    if parquet_saved:
        print(f"Saved: {OUTPUT_PARQUET}")


def main():
    print("Merging monthly demand with Weather and Holidays...")
    demand, weather, holidays = load_inputs()
    demand, weather, holidays = prepare_inputs(demand, weather, holidays)
    merged = merge_contextual_data(demand, weather, holidays)
    save_outputs(merged)
    print("CONTEXTUAL MERGE COMPLETE")


if __name__ == "__main__":
    main()
