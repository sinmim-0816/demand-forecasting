from pathlib import Path

import pandas as pd

from pipeline_config import (
    get_air_quality_settings,
    get_client_date_range,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONTEXTUAL_FILE = (
    PROJECT_ROOT / "data" / "raw" / "contextual" / "Contextual_Data.xlsx"
)
MONTHLY_DEMAND_FILE = (
    PROJECT_ROOT / "data" / "processed" / "shared" / "monthly_demand.csv"
)

MODERATE_API_THRESHOLD = 51
UNHEALTHY_API_THRESHOLD = 101

SOURCE_COLUMNS = [
    "Station ID",
    "Station Location",
    "Date",
    "API",
]

OUTPUT_COLUMNS = [
    "Month",
    "Location/Region",
    "Avg API",
    "Max API",
    "Moderate or Worse Days",
    "Unhealthy Days",
    "API Observation Days",
    "Source",
]


def get_required_locations():
    if not MONTHLY_DEMAND_FILE.exists():
        raise FileNotFoundError(
            f"Monthly demand file cannot be found:\n{MONTHLY_DEMAND_FILE}\n"
            "Run prepare_monthly_data.py before preparing air quality."
        )

    demand = pd.read_csv(MONTHLY_DEMAND_FILE, usecols=["location"])
    locations = (
        demand["location"]
        .astype("string")
        .str.strip()
        .str.title()
        .dropna()
        .unique()
        .tolist()
    )
    if not locations:
        raise ValueError("Monthly demand data contains no valid locations.")
    return sorted(locations)


def load_daily_air_quality():
    input_file, sheet_name, header_row = get_air_quality_settings()
    if not input_file.exists():
        raise FileNotFoundError(
            f"Configured API workbook cannot be found:\n{input_file}\n"
            "Update air_quality_data.file in pipeline_config.json."
        )

    daily = pd.read_excel(
        input_file,
        sheet_name=sheet_name,
        header=header_row - 1,
        engine="openpyxl",
    )
    missing = [column for column in SOURCE_COLUMNS if column not in daily.columns]
    if missing:
        raise ValueError(
            "API workbook is missing required columns: " + ", ".join(missing)
        )

    daily = daily[SOURCE_COLUMNS].copy()
    daily["date"] = pd.to_datetime(daily["Date"], errors="coerce")
    daily["api"] = pd.to_numeric(daily["API"], errors="coerce")
    daily["location"] = (
        daily["Station Location"]
        .astype("string")
        .str.split(",", n=1)
        .str[0]
        .str.strip()
        .str.title()
    )

    required_locations = get_required_locations()
    daily = daily[daily["location"].isin(required_locations)].copy()

    start_date, end_date = get_client_date_range()
    daily = daily[daily["date"].between(start_date, end_date)].copy()

    if daily.empty:
        raise ValueError(
            "No API observations match the client date range and locations."
        )
    if daily["date"].isna().any():
        raise ValueError("Selected API rows contain invalid dates.")
    if daily["api"].isna().any():
        raise ValueError("Selected API rows contain missing or invalid API values.")
    if (daily["api"] < 0).any():
        raise ValueError("Selected API rows contain negative API values.")
    if daily.duplicated(["location", "date"]).any():
        raise ValueError(
            "API data contains duplicate daily observations for a location. "
            "Confirm that the exact city station, not a second nearby station, is used."
        )

    available_locations = set(daily["location"].unique())
    missing_locations = sorted(set(required_locations) - available_locations)
    if missing_locations:
        raise ValueError(
            "API workbook has no matching observations for: "
            + ", ".join(missing_locations)
        )

    expected_dates = pd.date_range(start_date, end_date, freq="D")
    incomplete = []
    for location in required_locations:
        actual_dates = pd.DatetimeIndex(
            daily.loc[daily["location"].eq(location), "date"]
        ).sort_values()
        if not actual_dates.equals(expected_dates):
            missing_count = len(expected_dates.difference(actual_dates))
            extra_count = len(actual_dates.difference(expected_dates))
            incomplete.append(
                f"{location}: missing={missing_count}, extra={extra_count}"
            )

    if incomplete:
        raise ValueError(
            "API observations do not completely cover the client period: "
            + "; ".join(incomplete)
        )

    return daily, input_file


def create_monthly_air_quality(daily, source_file):
    prepared = daily.copy()
    prepared["month"] = prepared["date"].dt.to_period("M").dt.to_timestamp()
    prepared["moderate_or_worse"] = (
        prepared["api"] >= MODERATE_API_THRESHOLD
    ).astype(int)
    prepared["unhealthy"] = (
        prepared["api"] >= UNHEALTHY_API_THRESHOLD
    ).astype(int)

    monthly = prepared.groupby(["month", "location"], as_index=False).agg(
        avg_api=("api", "mean"),
        max_api=("api", "max"),
        moderate_or_worse_days=("moderate_or_worse", "sum"),
        unhealthy_days=("unhealthy", "sum"),
        api_observation_days=("date", "count"),
    )
    monthly["avg_api"] = monthly["avg_api"].round(2)
    monthly["max_api"] = monthly["max_api"].round(2)
    monthly = monthly.rename(
        columns={
            "month": "Month",
            "location": "Location/Region",
            "avg_api": "Avg API",
            "max_api": "Max API",
            "moderate_or_worse_days": "Moderate or Worse Days",
            "unhealthy_days": "Unhealthy Days",
            "api_observation_days": "API Observation Days",
        }
    )
    monthly["Source"] = f"Official API source workbook: {source_file.name}"
    return monthly[OUTPUT_COLUMNS].sort_values(
        ["Month", "Location/Region"]
    ).reset_index(drop=True)


def validate_monthly_air_quality(monthly):
    start_date, end_date = get_client_date_range()
    required_locations = get_required_locations()
    expected_months = pd.period_range(start=start_date, end=end_date, freq="M")
    expected_rows = len(expected_months) * len(required_locations)

    if len(monthly) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} monthly API rows but found {len(monthly)}."
        )
    if monthly.duplicated(["Month", "Location/Region"]).any():
        raise ValueError("Monthly API output contains duplicate month/location rows.")
    if monthly[OUTPUT_COLUMNS[:-1]].isna().any().any():
        raise ValueError("Monthly API output contains missing values.")


def save_air_quality_sheet(monthly):
    if not CONTEXTUAL_FILE.exists():
        raise FileNotFoundError(
            f"Contextual workbook cannot be found:\n{CONTEXTUAL_FILE}\n"
            "Run prepare_weather_data.py first."
        )

    try:
        with pd.ExcelWriter(
            CONTEXTUAL_FILE,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="replace",
        ) as writer:
            monthly.to_excel(writer, sheet_name="Air_Quality", index=False)
    except PermissionError:
        print("Close Contextual_Data.xlsx in Microsoft Excel and run again.")
        raise


def main():
    print("Preparing monthly official API haze features...")
    daily, source_file = load_daily_air_quality()
    monthly = create_monthly_air_quality(daily, source_file)
    validate_monthly_air_quality(monthly)
    save_air_quality_sheet(monthly)

    print(f"Daily API rows used: {len(daily):,}")
    print(f"Monthly API rows: {len(monthly):,}")
    print(f"Saved: {CONTEXTUAL_FILE} -> Air_Quality")
    print("AIR QUALITY PREPARATION COMPLETE")


if __name__ == "__main__":
    main()
