from pathlib import Path
import sys

import pandas as pd


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "preprocessing"
if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from pipeline_config import get_fama_price_settings

OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "market_prices"
OUTPUT_CSV = OUTPUT_DIR / "fama_ladang_prices.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "fama_ladang_prices.parquet"
COVERAGE_CSV = OUTPUT_DIR / "fama_ladang_coverage.csv"


# Workbook parsing

def normalize_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().upper().split())


def find_market_section(sheet, market_level):
    first_column = sheet.iloc[:, 0].map(normalize_text)
    section_label = f"PERINGKAT: {market_level}"
    matches = first_column[first_column.eq(section_label)].index.tolist()
    if len(matches) != 1:
        raise ValueError(
            f"Expected one '{section_label}' section, found {len(matches)}."
        )

    section_start = matches[0]
    later_sections = first_column[
        (first_column.index > section_start)
        & first_column.str.startswith("PERINGKAT:")
    ].index.tolist()
    section_end = later_sections[0] if later_sections else len(sheet)
    header_row = section_start + 1

    if normalize_text(sheet.iat[header_row, 0]) != "KOMODITI":
        raise ValueError("The commodity header was not found below the market section.")

    return header_row, section_end


def extract_sheet_prices(sheet, sheet_name, market_level, source_file):
    header_row, section_end = find_market_section(sheet, market_level)
    category = normalize_text(sheet_name.split("|")[0])
    month_columns = []

    for column_index in range(3, sheet.shape[1]):
        month = pd.to_datetime(sheet.iat[header_row, column_index], errors="coerce")
        if pd.notna(month):
            month_columns.append(
                (column_index, month.to_period("M").to_timestamp())
            )

    if not month_columns:
        raise ValueError(f"No monthly price columns found in sheet '{sheet_name}'.")

    records = []
    for row_index in range(header_row + 1, section_end):
        commodity = normalize_text(sheet.iat[row_index, 0])
        if not commodity or commodity.startswith("SUMBER"):
            continue

        grade = normalize_text(sheet.iat[row_index, 1])
        unit = normalize_text(sheet.iat[row_index, 2])
        if not grade or not unit:
            continue

        for column_index, month in month_columns:
            raw_price = sheet.iat[row_index, column_index]
            price = pd.to_numeric(raw_price, errors="coerce")
            records.append(
                {
                    "month": month,
                    "category": category,
                    "fama_commodity": commodity,
                    "grade": grade,
                    "unit": unit,
                    "market_level": market_level,
                    "fama_ladang_price_rm_per_unit": price,
                    "price_available": int(pd.notna(price)),
                    "source_sheet": sheet_name,
                    "source_file": source_file.name,
                }
            )

    return pd.DataFrame.from_records(records)


# Load and validate all FAMA Ladang series

def load_fama_workbook(input_file, market_level):
    if not input_file.exists():
        raise FileNotFoundError(f"FAMA workbook cannot be found:\n{input_file}")

    try:
        workbook = pd.read_excel(
            input_file,
            sheet_name=None,
            header=None,
            engine="xlrd",
        )
    except ImportError as error:
        raise ImportError(
            "Reading the legacy FAMA .xls workbook requires xlrd. "
            "Install the packages in requirements.txt."
        ) from error

    if not workbook:
        raise ValueError("The FAMA workbook contains no worksheets.")

    extracted = [
        extract_sheet_prices(sheet, sheet_name, market_level, input_file)
        for sheet_name, sheet in workbook.items()
    ]
    prices = pd.concat(extracted, ignore_index=True)
    return prices.sort_values(
        ["category", "fama_commodity", "grade", "unit", "month"]
    ).reset_index(drop=True)


def validate_prices(prices, market_level):
    key_columns = ["month", "category", "fama_commodity", "grade", "unit"]
    if prices.empty:
        raise ValueError("No FAMA market-price rows were extracted.")
    if prices["month"].isna().any():
        raise ValueError("FAMA output contains invalid month values.")
    if not prices["market_level"].eq(market_level).all():
        raise ValueError("FAMA output contains an unexpected market level.")
    if prices.duplicated(key_columns).any():
        duplicates = prices.loc[prices.duplicated(key_columns, keep=False), key_columns]
        raise ValueError(
            "FAMA output contains duplicate commodity-month records: "
            + repr(duplicates.head(3).to_dict(orient="records"))
        )

    invalid_prices = prices["fama_ladang_price_rm_per_unit"].notna() & (
        prices["fama_ladang_price_rm_per_unit"] <= 0
    )
    if invalid_prices.any():
        raise ValueError("FAMA output contains a zero or negative available price.")


def create_coverage_report(prices):
    group_columns = ["category", "fama_commodity", "grade", "unit"]
    return (
        prices.groupby(group_columns, as_index=False)
        .agg(
            first_month=("month", "min"),
            last_month=("month", "max"),
            expected_months=("month", "size"),
            available_months=("price_available", "sum"),
        )
        .assign(
            coverage_ratio=lambda data: (
                data["available_months"] / data["expected_months"]
            ).round(4)
        )
        .sort_values(["category", "fama_commodity", "grade", "unit"])
        .reset_index(drop=True)
    )


# Save normalized machine-readable outputs

def save_outputs(prices, coverage):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUTPUT_CSV, index=False, date_format="%Y-%m-%d")
    coverage.to_csv(COVERAGE_CSV, index=False, date_format="%Y-%m-%d")

    parquet_saved = True
    try:
        prices.to_parquet(OUTPUT_PARQUET, index=False)
    except ImportError:
        parquet_saved = False
        print("Parquet skipped: install pyarrow to enable Parquet output.")

    print(f"Normalized FAMA rows: {len(prices):,}")
    print(f"FAMA commodity series: {len(coverage):,}")
    print(
        "Available Ladang prices: "
        f"{prices['price_available'].sum():,} / {len(prices):,}"
    )
    print(f"Coverage: {prices['month'].min():%Y-%m} to {prices['month'].max():%Y-%m}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {COVERAGE_CSV}")
    if parquet_saved:
        print(f"Saved: {OUTPUT_PARQUET}")


def main():
    print("Preparing normalized FAMA Ladang price data...")
    input_file, market_level = get_fama_price_settings()
    prices = load_fama_workbook(input_file, market_level)
    validate_prices(prices, market_level)
    coverage = create_coverage_report(prices)
    save_outputs(prices, coverage)
    print("FAMA PRICE PREPARATION COMPLETE")


if __name__ == "__main__":
    main()
