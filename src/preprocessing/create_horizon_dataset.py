from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_FILE = PROCESSED_DIR / "model_features.csv"
CONTEXTUAL_FILE = PROCESSED_DIR / "monthly_demand_contextual.csv"
HOLIDAY_FILE = PROJECT_ROOT / "data" / "raw" / "contextual" / "Contextual_Data.xlsx"

TRAINING_CSV = PROCESSED_DIR / "horizon_training_dataset.csv"
TRAINING_PARQUET = PROCESSED_DIR / "horizon_training_dataset.parquet"
TRAINING_EXCEL = PROCESSED_DIR / "horizon_training_dataset.xlsx"
PREDICTION_TEMPLATE_CSV = PROCESSED_DIR / "future_horizon_features.csv"

GROUP_COLUMNS = ["crop", "location"]
HORIZONS = range(1, 13)
TARGET_HOLIDAY_COLUMNS = [
    "public_holiday_count",
    "cny",
    "hari_raya",
    "gawai",
    "christmas",
    "major_festival",
]
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
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset cannot be found:\n{FEATURES_FILE}\n"
            "Run src/preprocessing/create_features.py first."
        )
    if not CONTEXTUAL_FILE.exists():
        raise FileNotFoundError(
            f"Contextual dataset cannot be found:\n{CONTEXTUAL_FILE}\n"
            "Run src/preprocessing/merge_contextual_data.py first."
        )
    if not HOLIDAY_FILE.exists():
        raise FileNotFoundError(
            f"Holiday workbook cannot be found:\n{HOLIDAY_FILE}\n"
            "Run src/preprocessing/prepare_holidays_data.py first."
        )

    features = pd.read_csv(FEATURES_FILE, parse_dates=["month"])
    contextual = pd.read_csv(CONTEXTUAL_FILE, parse_dates=["month"])
    holiday_calendar = pd.read_excel(
        HOLIDAY_FILE,
        sheet_name="Holidays",
        engine="openpyxl",
    )

    require_columns(
        features,
        ["month", "crop", "location", "demand_kg"],
        "Feature dataset",
    )
    require_columns(
        contextual,
        [
            "month",
            "crop",
            "location",
            "demand_kg",
            "avg_temperature_c",
            "total_precipitation_mm",
            *TARGET_HOLIDAY_COLUMNS,
        ],
        "Contextual dataset",
    )
    require_columns(holiday_calendar, HOLIDAY_RENAME_MAP, "Holiday calendar")

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
        raise ValueError("Holiday calendar contains duplicate months.")

    return features, contextual, holiday_calendar


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
        "demand_kg": "origin_demand_kg",
        "weighted_avg_procurement_price_rm_per_kg": "origin_weighted_avg_procurement_price_rm_per_kg",
        "avg_temperature_c": "origin_avg_temperature_c",
        "avg_min_temperature_c": "origin_avg_min_temperature_c",
        "avg_max_temperature_c": "origin_avg_max_temperature_c",
        "total_precipitation_mm": "origin_total_precipitation_mm",
        "weather_observation_days": "origin_weather_observation_days",
        "year": "origin_year",
        "month_number": "origin_month_number",
        "quarter": "origin_quarter",
        "season": "origin_season",
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


def add_future_demand(expanded, contextual):
    future_demand = contextual[["month", "crop", "location", "demand_kg"]].rename(
        columns={"month": "target_month", "demand_kg": "future_demand_kg"}
    )
    if future_demand.duplicated(["target_month", "crop", "location"]).any():
        raise ValueError("Contextual data has duplicate target-demand keys.")

    return expanded.merge(
        future_demand,
        on=["target_month", "crop", "location"],
        how="left",
        validate="many_to_one",
    )


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


def add_leakage_safe_weather_normals(expanded, contextual):
    """Add target-month climatology using weather available by each origin month."""

    weather = contextual[
        [
            "month",
            "location",
            "avg_temperature_c",
            "total_precipitation_mm",
        ]
    ].drop_duplicates()
    if weather.duplicated(["month", "location"]).any():
        raise ValueError("Contextual data has conflicting weather rows for a month/location.")

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

    # Future observations are excluded so validation cannot see weather that was
    # unknown at the historical forecast origin.
    candidates = candidates[
        candidates["weather_month"] <= candidates["forecast_origin_month"]
    ]
    normals = candidates.groupby("_horizon_row_id", as_index=False).agg(
        target_month_normal_temperature=("avg_temperature_c", "mean"),
        target_month_normal_precipitation=("total_precipitation_mm", "mean"),
        target_month_weather_history_years=("weather_month", "count"),
    )

    expanded = expanded.merge(
        normals,
        on="_horizon_row_id",
        how="left",
        validate="one_to_one",
    )
    return expanded


def build_horizon_dataset(features, contextual, holiday_calendar):
    expanded = expand_horizons(features)
    expanded = add_future_demand(expanded, contextual)
    expanded = add_target_holidays(expanded, holiday_calendar)
    expanded = add_leakage_safe_weather_normals(expanded, contextual)

    expanded["target_month_normal_temperature"] = expanded[
        "target_month_normal_temperature"
    ].round(2)
    expanded["target_month_normal_precipitation"] = expanded[
        "target_month_normal_precipitation"
    ].round(2)

    expanded = expanded.drop(columns=["_horizon_row_id"])
    expanded = expanded.sort_values(
        ["crop", "location", "forecast_origin_month", "forecast_horizon"]
    ).reset_index(drop=True)
    return expanded


def create_training_dataset(horizon_data):
    training = horizon_data[horizon_data["future_demand_kg"].notna()].copy()
    required = [
        "target_month_normal_temperature",
        "target_month_normal_precipitation",
        *[f"target_{column}" for column in TARGET_HOLIDAY_COLUMNS],
    ]
    if training[required].isna().any().any():
        missing = training[required].isna().sum()
        missing = missing[missing > 0]
        raise ValueError(
            "Historical training rows have missing target context: "
            + ", ".join(f"{key}={value}" for key, value in missing.items())
        )
    return training


def create_prediction_template(horizon_data):
    latest_origin = horizon_data["forecast_origin_month"].max()
    return horizon_data[
        horizon_data["forecast_origin_month"].eq(latest_origin)
    ].reset_index(drop=True)


def save_outputs(training, prediction_template):
    training.to_csv(TRAINING_CSV, index=False)
    training.to_excel(TRAINING_EXCEL, index=False)
    prediction_template.to_csv(PREDICTION_TEMPLATE_CSV, index=False)

    parquet_saved = True
    try:
        training.to_parquet(TRAINING_PARQUET, index=False)
    except ImportError:
        parquet_saved = False
        print("Parquet skipped: install pyarrow to enable Parquet output.")

    print(f"Training rows: {len(training):,}")
    print(f"Future prediction-template rows: {len(prediction_template):,}")
    print(f"Saved: {TRAINING_CSV}")
    print(f"Saved: {TRAINING_EXCEL}")
    print(f"Saved: {PREDICTION_TEMPLATE_CSV}")
    if parquet_saved:
        print(f"Saved: {TRAINING_PARQUET}")


def main():
    print("Creating global direct multi-horizon H1-H12 dataset...")
    features, contextual, holiday_calendar = load_inputs()
    horizon_data = build_horizon_dataset(features, contextual, holiday_calendar)
    training = create_training_dataset(horizon_data)
    prediction_template = create_prediction_template(horizon_data)
    save_outputs(training, prediction_template)

    missing_future_holidays = prediction_template[
        [f"target_{column}" for column in TARGET_HOLIDAY_COLUMNS]
    ].isna().any(axis=1).sum()
    if missing_future_holidays:
        print(
            "NOTE: Future prediction rows need holiday data for "
            f"{missing_future_holidays} rows before production prediction."
        )
    else:
        print("Future target-month holiday features are complete.")
    print("HORIZON DATASET CREATION COMPLETE")


if __name__ == "__main__":
    main()
