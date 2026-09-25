import argparse
import importlib
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_DIR = PROJECT_ROOT / "src" / "preprocessing"
DEMAND_DIR = PROJECT_ROOT / "src" / "demand_forecasting"
for source_dir in (PREPROCESSING_DIR, DEMAND_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from pipeline_config import get_client_date_range, get_client_settings


STAGES = [
    ("monthly", "Prepare monthly demand", "prepare_monthly_data"),
    ("weather", "Download and prepare weather", "prepare_weather_data"),
    (
        "air-quality",
        "Prepare official API haze features",
        "prepare_air_quality_data",
    ),
    ("holidays", "Prepare holiday features", "prepare_holidays_data"),
    ("merge", "Merge contextual data", "merge_contextual_data"),
    ("features", "Create time-series features", "create_features"),
    ("horizons", "Create H1-H12 horizon dataset", "create_horizon_dataset"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the demand preprocessing pipeline.")
    parser.add_argument(
        "--start-at",
        choices=[stage[0] for stage in STAGES],
        default="monthly",
        help="Resume from this stage (default: monthly).",
    )
    return parser.parse_args()


def preflight():
    input_file, sheet_name, date_column = get_client_settings()
    start_date, end_date = get_client_date_range()
    print("=" * 60)
    print("DEMAND FORECASTING PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Client file: {input_file}")
    print(f"Sheet: {sheet_name}")
    print(f"Date column: {date_column}")
    print(f"Client date range: {start_date.date()} to {end_date.date()}")


def main():
    args = parse_args()
    preflight()
    start_index = next(
        index for index, stage in enumerate(STAGES) if stage[0] == args.start_at
    )
    pipeline_start = time.perf_counter()

    for position, (key, label, module_name) in enumerate(
        STAGES[start_index:], start=start_index + 1
    ):
        print("\n" + "=" * 60)
        print(f"STAGE {position}/{len(STAGES)}: {label}")
        print("=" * 60)
        stage_start = time.perf_counter()
        try:
            stage_module = importlib.import_module(module_name)
            stage_module.main()
        except Exception as error:
            print(f"\nPIPELINE STOPPED AT STAGE '{key}': {error}")
            raise
        print(f"Stage completed in {time.perf_counter() - stage_start:.1f} seconds.")

    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE COMPLETE")
    print(f"Total time: {time.perf_counter() - pipeline_start:.1f} seconds.")
    print(
        "Training dataset: "
        f"{PROJECT_ROOT / 'data' / 'processed' / 'demand' / 'horizon_training_dataset.csv'}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
