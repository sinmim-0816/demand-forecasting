from pathlib import Path
import pandas as pd
import numpy as np

from pipeline_config import get_client_settings

# 1. Project Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_FILE, SHEET_NAME, CLIENT_DATE_COLUMN = get_client_settings()

# Create processed file to store preprocessing dataset if not exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 2. Expected Client Columns
REQUIRED_COLUMNS = [
    "Crop/Product",
    "Requested Quantity",
    "Unit",
    "Buyer Type",
    "Location/Region",
    "Procurement Date",
    "Price",
    "Order",
    "Order Status",
    "Stock Available",
]

# 3. Internal Standard Column Names
COLUMN_RENAME_MAP = {
    "Crop/Product": "crop",
    "Requested Quantity": "requested_quantity",
    "Unit": "unit",
    "Buyer Type": "buyer_type",
    "Location/Region": "location",
    "Procurement Date": "procurement_date",
    "Price": "price_rm",
    "Order": "order_id",
    "Order Status": "order_status",
    "Stock Available": "stock_available",
}


# 4. Helper Functions
def clean_text(series):
    """
    Remove leading/trailing spaces and repeated spaces.
    """
    return series.astype("string").str.strip().str.replace(r"\s+", "", regex=True)


def standardize_unit(unit):
    """
    Convert common unit names to a standard representation.
    """
    if pd.isna(unit):
        return np.nan

    unit = str(unit).strip().lower()

    mapping = {
        "kg": "kg",
        "kgs": "kg",
        "kilogram": "kg",
        "kilograms": "kg",
        "g": "g",
        "gram": "g",
        "grams": "g",
        "tonne": "tonne",
        "tonnes": "tonne",
        "metric tonne": "tonne",
        "metric tonnes": "tonne",
        "t": "tonne",
    }

    return mapping.get(unit, unit)


def quantity_to_kg(quantity, unit):
    """
    Convert quantity into kilograms.
    """
    if pd.isna(quantity) or pd.isna(unit):
        return np.nan

    if unit == "kg":
        return quantity

    if unit == "g":
        return quantity / 1000

    if unit == "tonne":
        return quantity * 1000

    # Unknown unit
    return np.nan


def weighted_average(values, weights):
    """
    Calculate weighted average safely.
    """
    mask = values.notna() & weights.notna() & (weights > 0)

    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return np.nan

    return np.average(values, weights=weights)


# 5. Load raw client data
def load_data():
    print("\n============================================")
    print("Loading procurement dataset")
    print("============================================")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"\nInput file cannot be found:\n{INPUT_FILE}\n")

    # Only read columns A:J.
    # This avoids the synthetic notes/reference tables on the right.
    df = pd.read_excel(
        INPUT_FILE,
        sheet_name=SHEET_NAME,
        usecols="A:J",
        engine="openpyxl",
    )

    print(f"Raw rows loaded: {len(df):,}")

    return df


# 6. Validate Client Schema
def validate_schema(df):
    print("\n============================================")
    print("Validating client dataset structure")
    print("============================================")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n" + "\n".join(missing_columns)
        )

    print("All required columns are present.")


# 7. Clean Transaction Data
def clean_transactions(df):
    print("\n============================================")
    print("Cleaning transaction data")
    print("============================================")

    df = df.copy()

    # --------------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(how="all")

    print(f"Completely empty rows removed: " f"{before - len(df):,}")

    # --------------------------------------------------------
    # Rename columns to ML-friendly names
    # --------------------------------------------------------

    df = df.rename(columns=COLUMN_RENAME_MAP)

    # --------------------------------------------------------
    # Clean text fields
    # --------------------------------------------------------

    text_columns = [
        "crop",
        "unit",
        "buyer_type",
        "location",
        "order_id",
        "order_status",
    ]

    for col in text_columns:
        df[col] = clean_text(df[col])

    # Standard casing
    df["crop"] = df["crop"].str.title()
    df["buyer_type"] = df["buyer_type"].str.title()
    df["location"] = df["location"].str.title()
    df["order_status"] = df["order_status"].str.title()

    # Order IDs usually easier to read uppercase
    df["order_id"] = df["order_id"].str.upper()

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    df["procurement_date"] = pd.to_datetime(df["procurement_date"], errors="coerce")

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    df["requested_quantity"] = pd.to_numeric(df["requested_quantity"], errors="coerce")

    df["price_rm"] = pd.to_numeric(df["price_rm"], errors="coerce")

    df["stock_available"] = pd.to_numeric(df["stock_available"], errors="coerce")

    # --------------------------------------------------------
    # Standardize units
    # --------------------------------------------------------

    df["unit"] = df["unit"].apply(standardize_unit)

    # --------------------------------------------------------
    # Convert quantities to kilograms
    # --------------------------------------------------------

    df["requested_quantity_kg"] = df.apply(
        lambda row: quantity_to_kg(row["requested_quantity"], row["unit"]), axis=1
    )

    df["stock_available_kg"] = df.apply(
        lambda row: quantity_to_kg(row["stock_available"], row["unit"]), axis=1
    )

    # --------------------------------------------------------
    # Remove exact duplicate records
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    df = df.drop_duplicates()

    print(f"Exact duplicate rows removed: " f"{duplicate_count:,}")

    # --------------------------------------------------------
    # Identify invalid rows
    # --------------------------------------------------------

    invalid_mask = (
        df["crop"].isna()
        | df["location"].isna()
        | df["procurement_date"].isna()
        | df["requested_quantity_kg"].isna()
        | (df["requested_quantity_kg"] <= 0)
    )

    invalid_rows = df.loc[invalid_mask].copy()

    if len(invalid_rows) > 0:
        invalid_output = PROCESSED_DIR / "invalid_transactions.csv"

        invalid_rows.to_csv(invalid_output, index=False)

        print(f"Invalid rows found: {len(invalid_rows):,}")

        print(f"Saved invalid rows to: " f"{invalid_output}")

    # Remove invalid rows from modelling data
    df = df.loc[~invalid_mask].copy()

    # --------------------------------------------------------
    # Handle invalid price values
    # --------------------------------------------------------

    df.loc[df["price_rm"] <= 0, "price_rm"] = np.nan

    # --------------------------------------------------------
    # Handle impossible negative stock
    # --------------------------------------------------------

    df.loc[df["stock_available_kg"] < 0, "stock_available_kg"] = np.nan

    # --------------------------------------------------------
    # Create stock coverage ratio
    # --------------------------------------------------------

    df["stock_coverage_ratio"] = df["stock_available_kg"] / df["requested_quantity_kg"]

    # --------------------------------------------------------
    # Create month field
    # --------------------------------------------------------

    df["month"] = df["procurement_date"].dt.to_period("M").dt.to_timestamp()

    df["year"] = df["month"].dt.year
    df["month_number"] = df["month"].dt.month

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "crop",
            "location",
            "procurement_date",
            "order_id",
        ]
    ).reset_index(drop=True)

    print(f"Valid cleaned transactions: " f"{len(df):,}")

    return df


# 8. Create Monthly Demand Dataset
def aggregate_monthly(df):
    print("\n============================================")
    print("Creating monthly forecasting dataset")
    print("============================================")

    working = df.copy()

    # --------------------------------------------------------
    # PRICE CALCULATION POLICY
    #
    # Cancelled requests still represent expressed demand,
    # so requested quantity remains in the demand target.
    #
    # However, a cancelled order may not represent an actual
    # transaction price.
    #
    # Therefore price aggregation excludes Cancelled orders.
    # --------------------------------------------------------

    price_valid = working["order_status"].str.lower().ne("cancelled")

    working["price_weight_quantity"] = np.where(
        price_valid, working["requested_quantity_kg"], 0
    )

    working["price_weight_value"] = np.where(
        price_valid, working["price_rm"] * working["requested_quantity_kg"], 0
    )

    # --------------------------------------------------------
    # Base monthly aggregation
    # --------------------------------------------------------

    monthly = working.groupby(
        [
            "month",
            "crop",
            "location",
        ],
        as_index=False,
    ).agg(
        # Main demand target
        demand_kg=("requested_quantity_kg", "sum"),
        # Number of unique procurement orders
        order_count=("order_id", "nunique"),
        # Stock information
        total_stock_available_kg=("stock_available_kg", "sum"),
        avg_stock_coverage_ratio=("stock_coverage_ratio", "mean"),
        # Used for weighted procurement price
        total_price_weight_value=("price_weight_value", "sum"),
        total_price_weight_quantity=("price_weight_quantity", "sum"),
    )

    # --------------------------------------------------------
    # Weighted procurement price
    # --------------------------------------------------------

    monthly["weighted_avg_procurement_price_rm_per_kg"] = np.where(
        monthly["total_price_weight_quantity"] > 0,
        monthly["total_price_weight_value"] / monthly["total_price_weight_quantity"],
        np.nan,
    )

    # Intermediate fields no longer needed
    monthly = monthly.drop(
        columns=[
            "total_price_weight_value",
            "total_price_weight_quantity",
        ]
    )

    # --------------------------------------------------------
    # Order status counts
    # --------------------------------------------------------

    status_counts = working.pivot_table(
        index=[
            "month",
            "crop",
            "location",
        ],
        columns="order_status",
        values="order_id",
        aggfunc="nunique",
        fill_value=0,
    ).reset_index()

    status_counts.columns.name = None

    # Convert names such as:
    # Completed -> completed_orders
    rename_status = {}

    for col in status_counts.columns:
        if col not in ["month", "crop", "location"]:

            clean_name = str(col).lower().replace(" ", "_").replace("-", "_")

            rename_status[col] = f"{clean_name}_orders"

    status_counts = status_counts.rename(columns=rename_status)

    monthly = monthly.merge(
        status_counts,
        on=[
            "month",
            "crop",
            "location",
        ],
        how="left",
    )

    # --------------------------------------------------------
    # Buyer-type order counts
    # --------------------------------------------------------

    buyer_counts = working.pivot_table(
        index=[
            "month",
            "crop",
            "location",
        ],
        columns="buyer_type",
        values="order_id",
        aggfunc="nunique",
        fill_value=0,
    ).reset_index()

    buyer_counts.columns.name = None

    rename_buyer = {}

    for col in buyer_counts.columns:
        if col not in ["month", "crop", "location"]:

            clean_name = str(col).lower().replace(" ", "_").replace("-", "_")

            rename_buyer[col] = f"{clean_name}_order_count"

    buyer_counts = buyer_counts.rename(columns=rename_buyer)

    monthly = monthly.merge(
        buyer_counts,
        on=[
            "month",
            "crop",
            "location",
        ],
        how="left",
    )

    # --------------------------------------------------------
    # Add year/month fields
    # --------------------------------------------------------

    monthly["year"] = monthly["month"].dt.year

    monthly["month_number"] = monthly["month"].dt.month

    # --------------------------------------------------------
    # Monthly stock-to-demand ratio
    # --------------------------------------------------------

    monthly["stock_to_demand_ratio"] = np.where(
        monthly["demand_kg"] > 0,
        monthly["total_stock_available_kg"] / monthly["demand_kg"],
        np.nan,
    )

    # --------------------------------------------------------
    # Round useful numeric values
    # --------------------------------------------------------

    monthly["weighted_avg_procurement_price_rm_per_kg"] = monthly[
        "weighted_avg_procurement_price_rm_per_kg"
    ].round(2)

    monthly["avg_stock_coverage_ratio"] = monthly["avg_stock_coverage_ratio"].round(4)

    monthly["stock_to_demand_ratio"] = monthly["stock_to_demand_ratio"].round(4)

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    monthly = monthly.sort_values(
        [
            "crop",
            "location",
            "month",
        ]
    ).reset_index(drop=True)

    print(f"Monthly rows created: " f"{len(monthly):,}")

    return monthly


# 9. Check Monthly time-series completeness
def check_missing_months(monthly):
    print("\n============================================")
    print("Checking missing months")
    print("============================================")

    records = []

    for (crop, location), group in monthly.groupby(["crop", "location"]):

        start_month = group["month"].min()
        end_month = group["month"].max()

        expected_months = pd.date_range(start=start_month, end=end_month, freq="MS")

        actual_months = set(group["month"])

        for month in expected_months:

            if month not in actual_months:

                records.append(
                    {
                        "crop": crop,
                        "location": location,
                        "missing_month": month,
                    }
                )

    missing_months = pd.DataFrame(records)

    if missing_months.empty:

        print("No missing months found " "within existing crop-location series.")

    else:

        output = PROCESSED_DIR / "missing_months.csv"

        missing_months.to_csv(output, index=False)

        print(f"Missing crop-location months: " f"{len(missing_months):,}")

        print(f"Saved to: {output}")

    return missing_months


# 10. Create Data Quality Report
def create_quality_report(raw_df, clean_df, monthly_df, missing_months):

    print("\n============================================")
    print("Creating data quality report")
    print("============================================")

    report = {
        "raw_transaction_rows": len(raw_df),
        "clean_transaction_rows": len(clean_df),
        "monthly_rows": len(monthly_df),
        "unique_crops": clean_df["crop"].nunique(),
        "unique_locations": clean_df["location"].nunique(),
        "unique_buyer_types": clean_df["buyer_type"].nunique(),
        "unique_orders": clean_df["order_id"].nunique(),
        "date_start": clean_df["procurement_date"].min(),
        "date_end": clean_df["procurement_date"].max(),
        "missing_month_count": len(missing_months),
    }

    report_df = pd.DataFrame(report.items(), columns=["metric", "value"])

    report_file = PROCESSED_DIR / "data_quality_report.csv"

    report_df.to_csv(report_file, index=False)

    print(report_df.to_string(index=False))

    return report_df


# 11. Save Processed Data
def save_outputs(cleaned_transactions, monthly_data):

    print("\n============================================")
    print("Saving processed datasets")
    print("============================================")

    # --------------------------------------------------------
    # Cleaned transaction data
    # --------------------------------------------------------

    cleaned_csv = PROCESSED_DIR / "cleaned_procurement_transactions.csv"

    cleaned_parquet = PROCESSED_DIR / "cleaned_procurement_transactions.parquet"

    cleaned_transactions.to_csv(cleaned_csv, index=False)

    cleaned_transactions.to_parquet(cleaned_parquet, index=False)

    # --------------------------------------------------------
    # Monthly modelling data
    # --------------------------------------------------------

    monthly_csv = PROCESSED_DIR / "monthly_demand.csv"

    monthly_parquet = PROCESSED_DIR / "monthly_demand.parquet"

    monthly_excel = PROCESSED_DIR / "monthly_demand.xlsx"

    monthly_data.to_csv(monthly_csv, index=False)

    monthly_data.to_parquet(monthly_parquet, index=False)

    # Excel copy is useful for human inspection/reporting.
    monthly_data.to_excel(monthly_excel, index=False)

    print("\nSaved:")

    print(f"  {cleaned_csv}")

    print(f"  {cleaned_parquet}")

    print(f"  {monthly_csv}")

    print(f"  {monthly_parquet}")

    print(f"  {monthly_excel}")


# 12. Main Program
def main():

    print("\n")
    print("=" * 60)
    print("AGRICULTURAL DEMAND FORECASTING")
    print("CLIENT DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Read raw Excel data
    raw_df = load_data()

    # 2. Check that expected columns exist
    validate_schema(raw_df)

    # 3. Clean transactions
    cleaned_df = clean_transactions(raw_df)

    # 4. Aggregate monthly
    monthly_df = aggregate_monthly(cleaned_df)

    # 5. Check time-series continuity
    missing_months = check_missing_months(monthly_df)

    # 6. Generate data-quality summary
    create_quality_report(raw_df, cleaned_df, monthly_df, missing_months)

    # 7. Save final processed files
    save_outputs(cleaned_df, monthly_df)

    print("\n")
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

