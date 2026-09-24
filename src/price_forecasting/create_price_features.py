from pathlib import Path

import numpy as np
import pandas as pd


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = PROJECT_ROOT / "data" / "processed" / "shared"
PRICE_DIR = PROJECT_ROOT / "data" / "processed" / "price"
INPUT_FILE = SHARED_DIR / "monthly_demand_contextual.csv"
OUTPUT_CSV = PRICE_DIR / "price_features.csv"
OUTPUT_PARQUET = PRICE_DIR / "price_features.parquet"
OUTPUT_EXCEL = PRICE_DIR / "price_features.xlsx"


# Feature settings

GROUP_COLUMNS = ["crop", "location"]
PRICE_COLUMN = "weighted_avg_procurement_price_rm_per_kg"
PRICE_LAGS = [1, 3, 6, 12]
DEMAND_LAGS = [1, 3]


# Load and validate the shared monthly table

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Shared monthly contextual dataset cannot be found:\n{INPUT_FILE}\n"
            "Run scripts/run_preprocessing_pipeline.py first."
        )

    data = pd.read_csv(INPUT_FILE, parse_dates=["month"])
    required = [
        "month",
        "crop",
        "location",
        PRICE_COLUMN,
        "demand_kg",
        "order_count",
        "total_stock_available_kg",
        "stock_to_demand_ratio",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Shared monthly data is missing: " + ", ".join(missing))

    if data["month"].isna().any():
        raise ValueError("Shared monthly data contains invalid month values.")
    if data[PRICE_COLUMN].isna().any():
        raise ValueError("Monthly procurement price contains missing values.")
    if (data[PRICE_COLUMN] <= 0).any():
        raise ValueError("Monthly procurement price must be greater than zero.")
    if data.duplicated(["month", *GROUP_COLUMNS]).any():
        raise ValueError("Shared monthly data contains duplicate crop/location/month rows.")

    return data.sort_values([*GROUP_COLUMNS, "month"]).reset_index(drop=True)


# Lagging by row is safe only when every series has consecutive months

def validate_complete_monthly_series(data):
    invalid_series = []
    for keys, group in data.groupby(GROUP_COLUMNS, sort=False):
        expected = pd.date_range(group["month"].min(), group["month"].max(), freq="MS")
        actual = pd.DatetimeIndex(group["month"])
        if not actual.equals(expected):
            invalid_series.append(" / ".join(map(str, keys)))

    if invalid_series:
        raise ValueError(
            "Price lags require complete consecutive monthly series. Invalid series: "
            + ", ".join(invalid_series[:5])
        )


# Use Sarawak monsoon periods as readable categorical context

def season_from_month(month_number):
    if month_number in {11, 12, 1, 2, 3}:
        return "northeast_monsoon"
    if month_number in {6, 7, 8, 9}:
        return "southwest_monsoon"
    return "inter_monsoon"


# Create features using only the current origin month or earlier months

def create_features(data):
    featured = data.copy()
    featured["year"] = featured["month"].dt.year
    featured["month_number"] = featured["month"].dt.month
    featured["quarter"] = featured["month"].dt.quarter
    featured["season"] = featured["month_number"].map(season_from_month)

    price_groups = featured.groupby(GROUP_COLUMNS, sort=False)[PRICE_COLUMN]
    for lag in PRICE_LAGS:
        featured[f"price_lag_{lag}"] = price_groups.shift(lag)

    prior_price = price_groups.shift(1)
    price_history_groups = [featured[column] for column in GROUP_COLUMNS]
    featured["rolling_price_mean_3"] = prior_price.groupby(
        price_history_groups
    ).transform(lambda values: values.rolling(3).mean())
    featured["rolling_price_mean_6"] = prior_price.groupby(
        price_history_groups
    ).transform(lambda values: values.rolling(6).mean())
    featured["price_change_1m"] = featured["price_lag_1"] - price_groups.shift(2)
    featured["price_pct_change_1m"] = np.where(
        price_groups.shift(2) != 0,
        featured["price_change_1m"] / price_groups.shift(2),
        np.nan,
    )

    demand_groups = featured.groupby(GROUP_COLUMNS, sort=False)["demand_kg"]
    for lag in DEMAND_LAGS:
        featured[f"demand_lag_{lag}"] = demand_groups.shift(lag)
    prior_demand = demand_groups.shift(1)
    featured["rolling_demand_mean_3"] = prior_demand.groupby(
        price_history_groups
    ).transform(lambda values: values.rolling(3).mean())

    featured["order_count_lag_1"] = featured.groupby(
        GROUP_COLUMNS, sort=False
    )["order_count"].shift(1)
    featured["stock_available_lag_1"] = featured.groupby(
        GROUP_COLUMNS, sort=False
    )["total_stock_available_kg"].shift(1)
    featured["stock_to_demand_ratio_lag_1"] = featured.groupby(
        GROUP_COLUMNS, sort=False
    )["stock_to_demand_ratio"].shift(1)

    return featured


# Remove only rows that lack the minimum 12-month history

def keep_model_ready_rows(featured):
    required_history = [
        "price_lag_1",
        "price_lag_3",
        "price_lag_6",
        "price_lag_12",
        "rolling_price_mean_3",
        "rolling_price_mean_6",
        "price_change_1m",
        "price_pct_change_1m",
        "demand_lag_1",
        "demand_lag_3",
        "rolling_demand_mean_3",
        "order_count_lag_1",
        "stock_available_lag_1",
        "stock_to_demand_ratio_lag_1",
    ]
    featured = featured.replace([np.inf, -np.inf], np.nan)
    return featured.dropna(subset=required_history).copy(), required_history


# Save price features separately from demand features

def save_outputs(model_ready):
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    model_ready.to_csv(OUTPUT_CSV, index=False)
    model_ready.to_excel(OUTPUT_EXCEL, index=False)

    parquet_saved = True
    try:
        model_ready.to_parquet(OUTPUT_PARQUET, index=False)
    except ImportError:
        parquet_saved = False
        print("Parquet skipped: install pyarrow to enable Parquet output.")

    print(f"Price-feature rows: {len(model_ready):,}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_EXCEL}")
    if parquet_saved:
        print(f"Saved: {OUTPUT_PARQUET}")


def main():
    print("Creating leakage-safe platform-only price features...")
    data = load_data()
    validate_complete_monthly_series(data)
    featured = create_features(data)
    model_ready, required_history = keep_model_ready_rows(featured)

    removed = len(featured) - len(model_ready)
    print(f"Rows removed because 12-month history is unavailable: {removed:,}")
    if model_ready.empty:
        raise ValueError("No price-feature rows remain after applying history rules.")
    if model_ready[required_history].isna().any().any():
        raise ValueError("Price feature output still contains missing required history.")

    save_outputs(model_ready)
    print("PRICE FEATURE CREATION COMPLETE")


if __name__ == "__main__":
    main()
