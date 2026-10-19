from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from group_bias import (
    CONFIGS,
    KEY_COLUMNS,
    calculate_metrics,
    consistently_weak_crops,
    summarize_groups,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = {
    "crop": ["crop"],
    "target_month": ["target_month_label"],
    "target_season": ["target_season"],
    "forecast_horizon": ["forecast_horizon"],
    "location": ["location"],
    "crop_and_season": ["crop", "target_season"],
    "crop_and_target_month": ["crop", "target_month_label"],
}


def load_evaluation_data(project_root, domain, variant):
    config = CONFIGS[domain]
    split_path = (
        project_root
        / "data"
        / "processed"
        / domain
        / variant
        / "model_splits"
        / "test.csv"
    )
    prediction_path = (
        project_root
        / "artifacts"
        / domain
        / variant
        / "predictions"
        / config.prediction_filename
    )
    for path in (split_path, prediction_path):
        if not path.exists():
            raise FileNotFoundError(f"Required evaluation file is missing: {path}")

    split = pd.read_csv(
        split_path, parse_dates=["forecast_origin_month", "target_month"]
    )
    predictions = pd.read_csv(
        prediction_path, parse_dates=["forecast_origin_month", "target_month"]
    )
    required_split = KEY_COLUMNS + [config.target_column, "target_season"]
    required_predictions = KEY_COLUMNS + [
        config.target_column,
        config.prediction_column,
    ]
    for path, data, required in (
        (split_path, split, required_split),
        (prediction_path, predictions, required_predictions),
    ):
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        if data.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{path} contains duplicate evaluation keys.")

    joined = predictions[required_predictions].merge(
        split[required_split],
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_prediction_file", "_split"),
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        counts = joined["_merge"].value_counts().to_dict()
        raise ValueError(f"Prediction and test rows are not aligned: {counts}")

    prediction_target = joined[f"{config.target_column}_prediction_file"]
    split_target = joined[f"{config.target_column}_split"]
    if not np.allclose(prediction_target, split_target, rtol=0, atol=1e-8):
        raise ValueError("Prediction and split files contain different actual targets.")

    joined[config.target_column] = split_target
    joined = joined.drop(
        columns=[
            f"{config.target_column}_prediction_file",
            f"{config.target_column}_split",
            "_merge",
        ]
    )
    joined["target_month_label"] = joined["target_month"].dt.strftime("%Y-%m")
    joined["target_month_number"] = joined["target_month"].dt.month
    return joined.sort_values(KEY_COLUMNS).reset_index(drop=True)


def build_limitations(data, min_group_rows):
    target_months = sorted(data["target_month_label"].unique())
    target_seasons = sorted(data["target_season"].dropna().astype(str).unique())
    return [
        (
            f"The held-out test set covers {len(target_months)} target months "
            f"({target_months[0]} to {target_months[-1]}), not a complete annual cycle. "
            "Month and season findings are diagnostic rather than conclusive."
        ),
        (
            "The represented target seasons are: "
            + ", ".join(target_seasons)
            + ". Unrepresented seasons cannot be evaluated from this test set."
        ),
        (
            f"Groups with fewer than {min_group_rows} test rows are reported but are "
            "not flagged as weak because their estimates are less stable."
        ),
        (
            "Group error differences identify where a model is weaker; they do not "
            "establish that crop identity, season, or horizon caused the error."
        ),
        (
            "This is an offline held-out evaluation. Production bias and drift require "
            "stored forecasts to be matched with actual outcomes after target months close."
        ),
    ]


def write_markdown_summary(path, summary, weak_groups, consistent):
    lines = [
        f"# {summary['domain'].title()} forecast group evaluation",
        "",
        f"- Variant: `{summary['variant']}`",
        f"- Model: `{summary['model']}`",
        f"- Test rows: {summary['overall_metrics']['rows']}",
        f"- Overall WAPE: {summary['overall_metrics']['wape'] * 100:.2f}%",
        f"- Target months: {', '.join(summary['coverage']['target_months'])}",
        "",
        "## Highest-error eligible groups",
        "",
    ]
    if weak_groups.empty:
        lines.append("No groups met the weak-group rule.")
    else:
        for row in weak_groups.head(15).itertuples(index=False):
            lines.append(
                f"- {row.dimension}: {row.group_value} — WAPE "
                f"{row.wape * 100:.2f}%, {row.bias_direction}."
            )
    lines.extend(["", "## Consistently weak crops", ""])
    consistent_rows = consistent.loc[consistent["consistently_weak"]]
    if consistent_rows.empty:
        lines.append("No crop met the consistency rule across the available target months.")
    else:
        for row in consistent_rows.itertuples(index=False):
            lines.append(
                f"- {row.crop}: WAPE {row.overall_crop_wape * 100:.2f}%; "
                f"worse than overall in {row.months_worse_than_overall}/"
                f"{row.target_months_evaluated} target months; {row.bias_direction}."
            )
    lines.extend(["", "## Data limitations", ""])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_domain(project_root, domain, variant, min_group_rows):
    config = CONFIGS[domain]
    data = load_evaluation_data(project_root, domain, variant)
    overall = calculate_metrics(data[config.target_column], data[config.prediction_column])
    reports = {}
    for name, columns in DIMENSIONS.items():
        reports[name] = summarize_groups(
            data,
            columns,
            config.target_column,
            config.prediction_column,
            overall["wape"],
            min_group_rows=min_group_rows,
        )

    consistent = consistently_weak_crops(
        reports["crop"], reports["crop_and_target_month"], overall["wape"]
    )
    weak_records = []
    for name, columns in DIMENSIONS.items():
        if name == "crop_and_target_month":
            continue
        weak = reports[name].loc[reports[name]["weak_group"]].copy()
        if weak.empty:
            continue
        weak.insert(0, "dimension", name)
        weak.insert(
            1,
            "group_value",
            weak[columns].astype(str).agg(" | ".join, axis=1),
        )
        weak_records.append(
            weak[
                [
                    "dimension",
                    "group_value",
                    "rows",
                    "wape",
                    "mae",
                    "rmse",
                    "normalized_bias",
                    "bias_direction",
                ]
            ]
        )
    weak_groups = (
        pd.concat(weak_records, ignore_index=True).sort_values("wape", ascending=False)
        if weak_records
        else pd.DataFrame(
            columns=[
                "dimension",
                "group_value",
                "rows",
                "wape",
                "mae",
                "rmse",
                "normalized_bias",
                "bias_direction",
            ]
        )
    )

    output_dir = (
        project_root
        / "artifacts"
        / domain
        / variant
        / "reports"
        / "group_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        report.to_csv(output_dir / f"error_by_{name}.csv", index=False)
    consistent.to_csv(output_dir / "consistently_weak_crops.csv", index=False)
    weak_groups.to_csv(output_dir / "weak_groups.csv", index=False)

    limitations = build_limitations(data, min_group_rows)
    summary = {
        "domain": domain,
        "variant": variant,
        "model": (
            "tuned_xgboost" if domain == "demand" else "xgboost"
        ),
        "target_column": config.target_column,
        "prediction_column": config.prediction_column,
        "unit": config.unit,
        "signed_error_definition": "actual - predicted",
        "positive_bias_means": "underprediction",
        "weak_group_rule": (
            "rows >= minimum and WAPE >= dimension 75th percentile and WAPE > overall"
        ),
        "consistent_crop_rule": (
            "crop overall WAPE > model overall WAPE and crop is worse than overall "
            "in at least two target months and at least two-thirds of evaluated months"
        ),
        "minimum_group_rows": min_group_rows,
        "overall_metrics": overall,
        "coverage": {
            "target_months": sorted(data["target_month_label"].unique().tolist()),
            "target_seasons": sorted(
                data["target_season"].dropna().astype(str).unique().tolist()
            ),
            "forecast_horizons": sorted(
                data["forecast_horizon"].astype(int).unique().tolist()
            ),
            "crops": int(data["crop"].nunique()),
            "locations": int(data["location"].nunique()),
        },
        "weak_group_count": int(len(weak_groups)),
        "consistently_weak_crops": consistent.loc[
            consistent["consistently_weak"], "crop"
        ].tolist(),
        "limitations": limitations,
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown_summary(
        output_dir / "evaluation_summary.md", summary, weak_groups, consistent
    )
    return summary, weak_groups, output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate forecast errors across crops, months, seasons, and horizons."
    )
    parser.add_argument(
        "--domain", choices=["demand", "price", "both"], default="both"
    )
    parser.add_argument("--variant", choices=["core", "full"], default="core")
    parser.add_argument("--min-group-rows", type=int, default=12)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_group_rows < 1:
        raise ValueError("--min-group-rows must be at least 1.")
    domains = CONFIGS if args.domain == "both" else [args.domain]
    for domain in domains:
        summary, weak_groups, output_dir = evaluate_domain(
            args.project_root.resolve(), domain, args.variant, args.min_group_rows
        )
        overall = summary["overall_metrics"]
        print(f"\n{domain.upper()} {args.variant.upper()} GROUP EVALUATION COMPLETE")
        print(
            f"Rows: {overall['rows']:,} | WAPE: {overall['wape'] * 100:.2f}% | "
            f"R2: {overall['r2']:.3f}"
        )
        print(
            "Consistently weak crops: "
            + (", ".join(summary["consistently_weak_crops"]) or "none")
        )
        if not weak_groups.empty:
            print("Highest-error flagged groups:")
            print(
                weak_groups.head(8)[
                    ["dimension", "group_value", "rows", "wape", "bias_direction"]
                ].to_string(index=False)
            )
        print(f"Saved reports: {output_dir}")


if __name__ == "__main__":
    main()
