import argparse
import importlib
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRICE_DIR = PROJECT_ROOT / "src" / "price_forecasting"
if str(PRICE_DIR) not in sys.path:
    sys.path.insert(0, str(PRICE_DIR))


STAGES = [
    ("fama", "Normalize FAMA Ladang price data", "prepare_fama_price_data"),
    ("features", "Create leakage-safe price features", "create_price_features"),
    (
        "horizons",
        "Create price H1-H12 horizon datasets",
        "create_price_horizon_dataset",
    ),
    ("splits", "Create chronological price model splits", "prepare_model_data"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the price preprocessing pipeline.")
    parser.add_argument(
        "--start-at",
        choices=[stage[0] for stage in STAGES],
        default="fama",
        help="Resume from this stage (default: fama).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_index = next(
        index for index, stage in enumerate(STAGES) if stage[0] == args.start_at
    )
    pipeline_start = time.perf_counter()

    print("PRICE FORECASTING PREPROCESSING PIPELINE")
    for position, (key, label, module_name) in enumerate(
        STAGES[start_index:], start=start_index + 1
    ):
        print(f"\nSTAGE {position}/{len(STAGES)}: {label}")
        stage_start = time.perf_counter()
        try:
            stage_module = importlib.import_module(module_name)
            stage_module.main()
        except Exception as error:
            print(f"\nPIPELINE STOPPED AT STAGE '{key}': {error}")
            raise
        print(f"Stage completed in {time.perf_counter() - stage_start:.1f} seconds.")

    print("\nPRICE PREPROCESSING PIPELINE COMPLETE")
    print(f"Total time: {time.perf_counter() - pipeline_start:.1f} seconds.")
    print(
        "Training dataset: "
        f"{PROJECT_ROOT / 'data' / 'processed' / 'price' / 'price_horizon_training_dataset.csv'}"
    )


if __name__ == "__main__":
    main()
