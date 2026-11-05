"""Load tuned base-model predictions for stacking notebooks."""

from pathlib import Path

import pandas as pd

META_FEATURES = ["xgboost_tuned_prediction", "etr_tuned_prediction"]
KEY_COLUMNS = [
    "forecast_origin_month",
    "target_month",
    "crop",
    "location",
    "forecast_horizon",
]


def load_tuned_stacking_inputs(
    drive_project_dir,
    task,
    variant,
):
    """Return validation/test frames and meta features for a tuned stacker."""
    input_dir = (
        Path(drive_project_dir)
        / "artifacts"
        / task
        / variant
        / "predictions"
        / "tuned_base"
    )
    validation_file = input_dir / "tuned_base_validation_predictions.csv"
    test_file = input_dir / "tuned_base_test_predictions.csv"
    missing = [str(path) for path in (validation_file, test_file) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Upload tuned base prediction files before running stacking:\n"
            + "\n".join(missing)
        )

    validation = pd.read_csv(validation_file)
    test = pd.read_csv(test_file)
    required = [*KEY_COLUMNS, *META_FEATURES]
    for name, frame in (("validation", validation), ("test", test)):
        missing_columns = [column for column in required if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"{name} tuned stacking input is missing columns: "
                + ", ".join(missing_columns)
            )
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} tuned stacking input contains duplicate horizon rows.")

    return validation, test, META_FEATURES
