from pathlib import Path

import holidays
import pandas as pd

from pipeline_config import get_client_date_range


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONTEXTUAL_DIR = PROJECT_ROOT / "data" / "raw" / "contextual"
CONTEXTUAL_FILE = CONTEXTUAL_DIR / "Contextual_Data.xlsx"


# Holiday settings

COUNTRY_CODE = "MY"
SARAWAK_SUBDIVISION = "SWK"
FORECAST_HORIZON_MONTHS = 12


# Exceptional officially announced changes

# Add dates here only when an official holiday is missing from the installed
# holidays package. Format: "YYYY-MM-DD": "Holiday name".
HOLIDAY_ADDITIONS = {}

# Add dates here only when an entry from the installed holidays package has
# been officially cancelled for Sarawak. Format: "YYYY-MM-DD".
HOLIDAY_REMOVALS = set()


# Derive historical and forecast calendar coverage

def get_required_date_range():
    client_start, client_end = get_client_date_range()
    calendar_start = client_start.to_period("M").to_timestamp()
    calendar_end = (
        client_end.to_period("M") + FORECAST_HORIZON_MONTHS
    ).to_timestamp()
    return calendar_start, calendar_end


# Classify the major festivals used as model features

def classify_festival(holiday_name):
    normalized_name = str(holiday_name).casefold()

    if "chinese new year" in normalized_name:
        return "CNY"
    if "hari raya aidilfitri" in normalized_name or "eid al-fitr" in normalized_name:
        return "Hari Raya"
    if "gawai" in normalized_name or "dayak festival" in normalized_name:
        return "Gawai"
    if "christmas" in normalized_name:
        return "Christmas"
    return None


# Generate Sarawak holidays with the open-source holidays package

def create_holiday_dataframe(start_date, end_date):
    required_years = list(range(start_date.year, end_date.year + 1))
    sarawak_calendar = holidays.country_holidays(
        country=COUNTRY_CODE,
        subdiv=SARAWAK_SUBDIVISION,
        years=required_years,
        observed=True,
        language="en_US",
    )

    holiday_records = [
        {"Date": pd.Timestamp(holiday_date), "Holiday Name": holiday_name}
        for holiday_date, holiday_name in sorted(sarawak_calendar.items())
    ]
    generated = pd.DataFrame(holiday_records)

    if generated.empty:
        raise ValueError(
            "The holidays package returned no Malaysian/Sarawak holidays for "
            f"{required_years}."
        )

    additions = pd.DataFrame(
        [
            {"Date": pd.Timestamp(date), "Holiday Name": name}
            for date, name in HOLIDAY_ADDITIONS.items()
        ]
    )
    if not additions.empty:
        generated = pd.concat([generated, additions], ignore_index=True)

    removal_dates = {pd.Timestamp(date).normalize() for date in HOLIDAY_REMOVALS}
    generated["Date"] = pd.to_datetime(generated["Date"]).dt.normalize()
    if removal_dates:
        generated = generated[~generated["Date"].isin(removal_dates)]

    generated = generated[
        generated["Date"].between(start_date, end_date + pd.offsets.MonthEnd(0))
    ].copy()
    generated = (
        generated.groupby("Date", as_index=False)["Holiday Name"]
        .agg(lambda names: " / ".join(sorted(set(map(str, names)))))
        .sort_values("Date")
        .reset_index(drop=True)
    )

    generated["Festival"] = generated["Holiday Name"].map(classify_festival)
    generated["Month"] = generated["Date"].dt.to_period("M").dt.to_timestamp()
    return generated


# Create every required historical and future month

def create_monthly_calendar(start_date, end_date):
    return pd.DataFrame(
        {"Month": pd.date_range(start=start_date, end=end_date, freq="MS")}
    )


# Aggregate holiday dates into monthly model features

def create_monthly_holiday_features(holiday_dates, start_date, end_date):
    monthly = create_monthly_calendar(start_date, end_date)

    holiday_counts = holiday_dates.groupby("Month")["Date"].nunique()
    monthly["Public Holiday Count"] = (
        monthly["Month"].map(holiday_counts).fillna(0).astype(int)
    )

    festival_columns = ["CNY", "Hari Raya", "Gawai", "Christmas"]
    for festival in festival_columns:
        festival_months = holiday_dates.loc[
            holiday_dates["Festival"].eq(festival), "Month"
        ].drop_duplicates()
        monthly[festival] = monthly["Month"].isin(festival_months).astype(int)

    monthly["Major Festival"] = monthly[festival_columns].max(axis=1).astype(int)
    monthly["Source"] = "Python holidays package - Malaysia/Sarawak (SWK)"
    return monthly


# Validate generated dates and monthly features

def validate_holidays(holiday_dates, monthly, start_date, end_date):
    expected_rows = len(pd.period_range(start=start_date, end=end_date, freq="M"))

    print("\nHOLIDAY DATA QUALITY CHECK")
    print(f"Calendar range: {start_date.date()} to {end_date.date()}")
    print(f"Generated holiday dates: {len(holiday_dates):,}")
    print(f"Expected monthly rows: {expected_rows}")
    print(f"Actual monthly rows: {len(monthly)}")
    print(f"Duplicate months: {monthly.duplicated(subset=['Month']).sum()}")

    if len(monthly) != expected_rows:
        raise ValueError("Monthly holiday calendar does not cover the required range.")
    if monthly.duplicated(subset=["Month"]).any():
        raise ValueError("Monthly holiday calendar contains duplicate months.")
    if monthly.isna().any().any():
        raise ValueError("Monthly holiday calendar contains missing values.")

    future_start = end_date - pd.DateOffset(months=FORECAST_HORIZON_MONTHS - 1)
    future_holidays = holiday_dates[holiday_dates["Date"] >= future_start]
    if future_holidays.empty:
        raise ValueError(
            "No holidays were generated for the future forecast period. "
            "Update the holidays package before continuing."
        )

    print("\nPublic holiday days by year:")
    print(
        monthly.assign(Year=monthly["Month"].dt.year)
        .groupby("Year")["Public Holiday Count"]
        .sum()
    )


# Replace the Holidays worksheet

def save_holiday_sheet(monthly):
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
            monthly.to_excel(writer, sheet_name="Holidays", index=False)
    except PermissionError:
        print("Close Contextual_Data.xlsx in Microsoft Excel and run again.")
        raise

    print(f"Holidays worksheet saved: {CONTEXTUAL_FILE}")


# Run holiday preparation

def main():
    start_date, end_date = get_required_date_range()
    holiday_dates = create_holiday_dataframe(start_date, end_date)
    monthly = create_monthly_holiday_features(
        holiday_dates,
        start_date,
        end_date,
    )
    validate_holidays(holiday_dates, monthly, start_date, end_date)
    save_holiday_sheet(monthly)
    print("HOLIDAY PREPARATION COMPLETE")


if __name__ == "__main__":
    main()
