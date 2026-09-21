from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_FILE = PROCESSED_DIR / "monthly_demand_contextual.csv"

OUTPUT_CSV = PROCESSED_DIR / "model_features.csv"
OUTPUT_PARQUET = PROCESSED_DIR / "model_features.parquet"
OUTPUT_EXCEL = PROCESSED_DIR / "model_features.xlsx"

GROUP_COLUMNS = ["crop", "location"]
PRICE_COLUMN = "weighted_avg_procurement_price_rm_per_kg"
DEMAND_LAGS = [1, 3, 6, 12]
PRICE_LAGS = [1, 3]


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Merged contextual dataset cannot be found:\n{INPUT_FILE}\n"
            "Run src/preprocessing/merge_contextual_data.py first."
        )

    df = pd.read_csv(INPUT_FILE)
    required = ["month", "crop", "location", "demand_kg", PRICE_COLUMN]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    if df["month"].isna().any():
        raise ValueError("Input contains invalid month values.")

    return df.sort_values(GROUP_COLUMNS + ["month"]).reset_index(drop=True)


def validate_complete_monthly_series(df):
    problems = []
    for keys, group in df.groupby(GROUP_COLUMNS, sort=False):
        expected = pd.date_range(group["month"].min(), group["month"].max(), freq="MS")
        actual = pd.DatetimeIndex(group["month"])
        if actual.duplicated().any() or not actual.equals(expected):
            problems.append(" / ".join(map(str, keys)))

    if problems:
        preview = ", ".join(problems[:5])
        raise ValueError(
            "Lag features require complete, unique monthly series. "
            f"Invalid series include: {preview}"
        )


def monsoon_from_month(month):
    if month in {11, 12, 1, 2, 3}:
        return "northeast_monsoon"
    if month in {6, 7, 8, 9}:
        return "southwest_monsoon"
    return "inter_monsoon"


def create_features(df):
    featured = df.copy()
    featured["year"] = featured["month"].dt.year
    featured["month_number"] = featured["month"].dt.month
    featured["quarter"] = featured["month"].dt.quarter
    featured["season"] = featured["month_number"].map(monsoon_from_month)

    demand_groups = featured.groupby(GROUP_COLUMNS, sort=False)["demand_kg"]
    for lag in DEMAND_LAGS:
        featured[f"demand_lag_{lag}"] = demand_groups.shift(lag)

    prior_demand = demand_groups.shift(1)
    featured["rolling_demand_mean_3"] = prior_demand.groupby(
        [featured[column] for column in GROUP_COLUMNS]
    ).transform(lambda values: values.rolling(3).mean())
    featured["rolling_demand_mean_6"] = prior_demand.groupby(
        [featured[column] for column in GROUP_COLUMNS]
    ).transform(lambda values: values.rolling(6).mean())
    featured["rolling_demand_std_3"] = prior_demand.groupby(
        [featured[column] for column in GROUP_COLUMNS]
    ).transform(lambda values: values.rolling(3).std())

    price_groups = featured.groupby(GROUP_COLUMNS, sort=False)[PRICE_COLUMN]
    for lag in PRICE_LAGS:
        featured[f"price_lag_{lag}"] = price_groups.shift(lag)

    prior_price = price_groups.shift(1)
    featured["rolling_price_mean_3"] = prior_price.groupby(
        [featured[column] for column in GROUP_COLUMNS]
    ).transform(lambda values: values.rolling(3).mean())

    return featured


def keep_model_ready_rows(featured):
    required_history = [
        "demand_lag_1",
        "demand_lag_3",
        "demand_lag_6",
        "demand_lag_12",
        "rolling_demand_mean_3",
        "rolling_demand_mean_6",
        "rolling_demand_std_3",
        "price_lag_1",
        "price_lag_3",
        "rolling_price_mean_3",
    ]

    # Infinite values cannot be consumed safely by the modelling stage.
    featured = featured.replace([np.inf, -np.inf], np.nan)
    model_ready = featured.dropna(subset=required_history).copy()
    return model_ready, required_history


def save_outputs(model_ready):
    model_ready.to_csv(OUTPUT_CSV, index=False)
    model_ready.to_excel(OUTPUT_EXCEL, index=False)

    parquet_saved = True
    try:
        model_ready.to_parquet(OUTPUT_PARQUET, index=False)
    except ImportError:
        parquet_saved = False
        print("Parquet skipped: install pyarrow to enable Parquet output.")

    print(f"Model-ready rows: {len(model_ready):,}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_EXCEL}")
    if parquet_saved:
        print(f"Saved: {OUTPUT_PARQUET}")


def main():
    print("Creating leakage-safe time-series features...")
    df = load_data()
    validate_complete_monthly_series(df)
    featured = create_features(df)
    model_ready, required_history = keep_model_ready_rows(featured)

    removed = len(featured) - len(model_ready)
    print(f"Rows removed because required history is unavailable: {removed:,}")
    if model_ready.empty:
        raise ValueError(
            "No model-ready rows remain. At least 13 complete months per series are needed."
        )

    save_outputs(model_ready)
    print("FEATURE CREATION COMPLETE")


if __name__ == "__main__":
    main()
