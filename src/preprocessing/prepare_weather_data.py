import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from pipeline_config import get_client_date_range

# 1. Projects Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONTEXTUAL_DIR = PROJECT_ROOT / "data" / "raw" / "contextual"
CONTEXTUAL_DIR.mkdir(parents=True, exist_ok=True)
CONTEXTUAL_FILE = CONTEXTUAL_DIR / "Contextual_Data.xlsx"

# 2. OPEN-METEO HISTORICAL API
API_URL = "https://archive-api.open-meteo.com/" "v1/archive"

# 3. LOCATIONS
LOCATIONS = {
    "Kuching": {
        "lat": 1.5535,
        "lon": 110.3593,
    },
    "Sibu": {
        "lat": 2.2873,
        "lon": 111.8305,
    },
    "Miri": {
        "lat": 4.3995,
        "lon": 113.9914,
    },
}

# 5. CREATE CONTEXTUAL WORKBOOK IF NEEDED
def create_contextual_workbook_if_needed():

    if CONTEXTUAL_FILE.exists():
        return

    print("Creating Contextual_Data.xlsx...")

    weather_columns = [
        "Month",
        "Location/Region",
        "Avg Temperature (°C)",
        "Avg Min Temperature (°C)",
        "Avg Max Temperature (°C)",
        "Total Precipitation (mm)",
        "Avg Daily Shortwave Radiation (MJ/m²)",
        "Observation Days",
        "Source",
    ]

    air_quality_columns = [
        "Month",
        "Location/Region",
        "Avg API",
        "Max API",
        "Moderate or Worse Days",
        "Unhealthy Days",
        "API Observation Days",
        "Source",
    ]

    holiday_columns = [
        "Month",
        "Public Holiday Count",
        "CNY",
        "Hari Raya",
        "Gawai",
        "Christmas",
        "Source",
    ]

    with pd.ExcelWriter(CONTEXTUAL_FILE, engine="openpyxl") as writer:

        pd.DataFrame(columns=weather_columns).to_excel(
            writer, sheet_name="Weather", index=False
        )

        pd.DataFrame(columns=air_quality_columns).to_excel(
            writer, sheet_name="Air_Quality", index=False
        )

        pd.DataFrame(columns=holiday_columns).to_excel(
            writer, sheet_name="Holidays", index=False
        )

    print(f"Workbook created:\n{CONTEXTUAL_FILE}")

# 6. DOWNLOAD ONE LOCATION
def fetch_weather_for_location(location, latitude, longitude, start_date, end_date):

    print()
    print("=" * 60)

    print(f"Downloading weather for {location}")

    print("=" * 60)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        # Daily variables needed for the FYP
        "daily": (
            "temperature_2m_mean,"
            "temperature_2m_min,"
            "temperature_2m_max,"
            "precipitation_sum,"
            "shortwave_radiation_sum"
        ),
        # Malaysian local time
        "timezone": "Asia/Kuching",
        # Keep one consistent historical weather source.
        "models": "era5",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }

    request_url = f"{API_URL}?{urlencode(params)}"
    try:
        with urlopen(request_url, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Open-Meteo HTTP {error.code} for {location}: {details}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not reach Open-Meteo for {location}: {error.reason}"
        ) from error

    if "daily" not in data:

        raise RuntimeError(f"No daily weather data returned " f"for {location}.")

    daily = data["daily"]

    # CONVERT JSON -> DATAFRAME
    weather = pd.DataFrame(
        {
            "date": daily["time"],
            "daily_mean_temperature_c": daily["temperature_2m_mean"],
            "daily_min_temperature_c": daily["temperature_2m_min"],
            "daily_max_temperature_c": daily["temperature_2m_max"],
            "daily_precipitation_mm": daily["precipitation_sum"],
            "daily_shortwave_radiation_mj_m2": daily[
                "shortwave_radiation_sum"
            ],
        }
    )

    weather["date"] = pd.to_datetime(weather["date"])

    weather["location"] = location

    print(f"Daily rows downloaded: " f"{len(weather):,}")

    print(f"First date: " f"{weather['date'].min().date()}")

    print(f"Last date: " f"{weather['date'].max().date()}")

    return weather

# 7. DOWNLOAD ALL THREE LOCATIONS
def download_all_weather(start_date, end_date):

    all_weather = []

    for location, coordinates in LOCATIONS.items():

        weather = fetch_weather_for_location(
            location=location,
            latitude=coordinates["lat"],
            longitude=coordinates["lon"],
            start_date=start_date,
            end_date=end_date,
        )

        all_weather.append(weather)

    combined = pd.concat(all_weather, ignore_index=True)

    return combined

# 8. DAILY -> MONTHLY AGGREGATION
def create_monthly_weather(daily_weather):

    print()
    print("=" * 60)
    print("CREATING MONTHLY WEATHER DATA")
    print("=" * 60)

    weather = daily_weather.copy()

    # --------------------------------------------------------
    # Month = first day of each month
    # --------------------------------------------------------

    weather["month"] = weather["date"].dt.to_period("M").dt.to_timestamp()

    # --------------------------------------------------------
    # Aggregate each Month + Location
    # --------------------------------------------------------

    monthly = weather.groupby(["month", "location"], as_index=False).agg(
        avg_temperature_c=("daily_mean_temperature_c", "mean"),
        avg_min_temperature_c=("daily_min_temperature_c", "mean"),
        avg_max_temperature_c=("daily_max_temperature_c", "mean"),
        total_precipitation_mm=("daily_precipitation_mm", "sum"),
        avg_daily_shortwave_radiation_mj_m2=(
            "daily_shortwave_radiation_mj_m2",
            "mean",
        ),
        observation_days=("date", "count"),
    )

    # ========================================================
    # ROUND NUMERIC VALUES
    # ========================================================

    monthly["avg_temperature_c"] = monthly["avg_temperature_c"].round(2)

    monthly["avg_min_temperature_c"] = monthly["avg_min_temperature_c"].round(2)

    monthly["avg_max_temperature_c"] = monthly["avg_max_temperature_c"].round(2)

    monthly["total_precipitation_mm"] = monthly["total_precipitation_mm"].round(2)

    monthly["avg_daily_shortwave_radiation_mj_m2"] = monthly[
        "avg_daily_shortwave_radiation_mj_m2"
    ].round(2)

    # ========================================================
    # RENAME FINAL COLUMNS
    # ========================================================

    monthly = monthly.rename(
        columns={
            "month": "Month",
            "location": "Location/Region",
            "avg_temperature_c": "Avg Temperature (°C)",
            "avg_min_temperature_c": "Avg Min Temperature (°C)",
            "avg_max_temperature_c": "Avg Max Temperature (°C)",
            "total_precipitation_mm": "Total Precipitation (mm)",
            "avg_daily_shortwave_radiation_mj_m2": (
                "Avg Daily Shortwave Radiation (MJ/m²)"
            ),
            "observation_days": "Observation Days",
        }
    )

    monthly["Source"] = "Open-Meteo Historical Weather API - ERA5"

    monthly = monthly.sort_values(["Month", "Location/Region"]).reset_index(drop=True)

    return monthly

# 9. VALIDATE DATA
def validate_weather(monthly_weather, start_date, end_date):

    print()
    print("=" * 60)
    print("WEATHER DATA QUALITY CHECK")
    print("=" * 60)

    expected_months = len(
        pd.period_range(start=start_date, end=end_date, freq="M")
    )
    expected_rows = expected_months * len(LOCATIONS)

    actual_rows = len(monthly_weather)

    print(f"Expected monthly rows: " f"{expected_rows}")

    print(f"Actual monthly rows: " f"{actual_rows}")

    # --------------------------------------------------------
    # Duplicate Month + Location
    # --------------------------------------------------------

    duplicates = monthly_weather.duplicated(subset=["Month", "Location/Region"]).sum()

    print(f"Duplicate rows: " f"{duplicates}")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print()
    print("Missing values:")

    print(monthly_weather.isna().sum())

    # --------------------------------------------------------
    # Check observation days
    # --------------------------------------------------------

    incomplete = monthly_weather[monthly_weather["Observation Days"] < 28]

    print()
    print("Months with fewer than " f"28 observations: {len(incomplete)}")

    if expected_rows != actual_rows:

        print()
        print("WARNING: Monthly row count " "does not match expectation.")


# 10. SAVE INTO WEATHER WORKSHEET
def save_weather_sheet(monthly_weather):

    create_contextual_workbook_if_needed()

    # --------------------------------------------------------
    # Replace ONLY Weather sheet.
    #
    # Air_Quality, Holidays and Crop_Production remain.
    # --------------------------------------------------------

    with pd.ExcelWriter(
        CONTEXTUAL_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:

        monthly_weather.to_excel(writer, sheet_name="Weather", index=False)

    print()
    print("=" * 60)

    print("Weather worksheet saved successfully.")

    print("=" * 60)

    print(f"\nWorkbook:\n" f"{CONTEXTUAL_FILE}")

# 11. MAIN
def main():

    create_contextual_workbook_if_needed()

    start_date, end_date = get_client_date_range()

    print(
        "Weather date range derived from client data: "
        f"{start_date.date()} to {end_date.date()}"
    )

    daily_weather = download_all_weather(start_date, end_date)

    monthly_weather = create_monthly_weather(daily_weather)

    validate_weather(monthly_weather, start_date, end_date)

    save_weather_sheet(monthly_weather)

    print()
    print("=" * 60)

    print("OPEN-METEO WEATHER " "PREPARATION COMPLETE")

    print("=" * 60)


if __name__ == "__main__":
    main()
