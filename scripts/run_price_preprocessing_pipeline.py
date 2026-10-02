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
VARIANT_STAGES = {"features", "horizons", "splits"}
VALID_VARIANTS = ("core", "full", "both")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the price preprocessing pipeline.")
    parser.add_argument(
        "--start-at",
        choices=[stage[0] for stage in STAGES],
        default="fama",
        help="Resume from this stage (default: fama).",
    )
    parser.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default="both",
        help="Build core, full, or both price datasets (default: both).",
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
            if key in VARIANT_STAGES:
                variants = ("core", "full") if args.variant == "both" else (args.variant,)
                for variant in variants:
                    print(f"Building {variant} variant...")
                    stage_module.main(variant=variant)
            elif key == "fama" and args.variant == "core":
                print("Skipped: the core price variant does not use FAMA.")
            else:
                stage_module.main()
        except Exception as error:
            print(f"\nPIPELINE STOPPED AT STAGE '{key}': {error}")
            raise
        print(f"Stage completed in {time.perf_counter() - stage_start:.1f} seconds.")

    print("\nPRICE PREPROCESSING PIPELINE COMPLETE")
    print(f"Total time: {time.perf_counter() - pipeline_start:.1f} seconds.")
    print(
        "Variant datasets: "
        f"{PROJECT_ROOT / 'data' / 'processed' / 'price' / '<core|full>'}"
    )


if __name__ == "__main__":
    main()
