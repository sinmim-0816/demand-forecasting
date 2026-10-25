import argparse
import importlib
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPLY_DIR = PROJECT_ROOT / "src" / "supply_forecasting"
if str(SUPPLY_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPLY_DIR))

STAGES = [
    ("monthly", "Prepare monthly crop production", "prepare_supply_data"),
    ("features", "Create leakage-safe supply features", "create_supply_features"),
    ("horizons", "Create supply H1-H12 datasets", "create_supply_horizon_dataset"),
    ("splits", "Create chronological supply splits", "prepare_model_data"),
]
VARIANT_STAGES = {"features", "horizons", "splits"}
VALID_VARIANTS = ("core", "full", "both")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the supply preprocessing pipeline.")
    parser.add_argument("--start-at", choices=[stage[0] for stage in STAGES], default="monthly")
    parser.add_argument("--variant", choices=VALID_VARIANTS, default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    start_index = next(index for index, stage in enumerate(STAGES) if stage[0] == args.start_at)
    pipeline_start = time.perf_counter()
    print("SUPPLY FORECASTING PREPROCESSING PIPELINE")
    for position, (key, label, module_name) in enumerate(STAGES[start_index:], start=start_index + 1):
        print(f"\nSTAGE {position}/{len(STAGES)}: {label}")
        stage_start = time.perf_counter()
        try:
            module = importlib.import_module(module_name)
            if key in VARIANT_STAGES:
                variants = ("core", "full") if args.variant == "both" else (args.variant,)
                for variant in variants:
                    print(f"Building {variant} variant...")
                    module.main(variant=variant)
            else:
                module.main()
        except Exception as error:
            print(f"\nPIPELINE STOPPED AT STAGE '{key}': {error}")
            raise
        print(f"Stage completed in {time.perf_counter() - stage_start:.1f} seconds.")
    print("\nSUPPLY PREPROCESSING PIPELINE COMPLETE")
    print(f"Total time: {time.perf_counter() - pipeline_start:.1f} seconds.")


if __name__ == "__main__":
    main()
