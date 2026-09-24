"""Compatibility entry point for the relocated demand XGBoost trainer."""

from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    target_dir = Path(__file__).resolve().parents[1] / "demand_forecasting"
    sys.path.insert(0, str(target_dir))
    runpy.run_path(
        target_dir / "train_xgboost.py",
        run_name="__main__",
    )
