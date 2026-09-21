import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = PROJECT_ROOT / "pipeline_config.json"

def load_config():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Pipeline configuration cannot be found:\n{CONFIG_FILE}")

    with CONFIG_FILE.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    try:
        client = config["client_data"]
        required_values = [
            client["file"],
            client["sheet"],
            client["date_column"],
        ]
    except KeyError as error:
        raise ValueError(f"Missing pipeline configuration setting: {error}") from error

    if any(not value for value in required_values):
        raise ValueError("Pipeline configuration values cannot be empty.")

    return config


def get_client_settings():
    client = load_config()["client_data"]
    input_file = Path(client["file"])
    if not input_file.is_absolute():
        input_file = PROJECT_ROOT / input_file
    return input_file, client["sheet"], client["date_column"]


def get_client_date_range():
    input_file, sheet_name, date_column = get_client_settings()

    if not input_file.exists():
        raise FileNotFoundError(f"Client dataset cannot be found:\n{input_file}")

    try:
        dates = pd.read_excel(
            input_file,
            sheet_name=sheet_name,
            usecols=[date_column],
            engine="openpyxl",
        )[date_column]
    except ValueError as error:
        raise ValueError(
            f"Column '{date_column}' was not found in sheet '{sheet_name}' "
            f"of {input_file}."
        ) from error

    dates = pd.to_datetime(dates, errors="coerce").dropna()
    if dates.empty:
        raise ValueError(f"No valid dates were found in the '{date_column}' column.")

    return dates.min().normalize(), dates.max().normalize()

