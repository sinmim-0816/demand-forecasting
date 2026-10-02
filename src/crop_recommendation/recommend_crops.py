"""Generate ranked crop market-opportunity recommendations."""

import csv
import json
from datetime import datetime, timezone

try:
    from .config import (
        ACTIVE_PLANTINGS_FILE,
        CROP_DEFAULTS_FILE,
        DEMAND_FORECAST_FILE,
        DEMAND_WEIGHT,
        FARMER_INPUT_FILE,
        OUTPUT_FILE,
        PRICE_FORECAST_FILE,
        PRICE_WEIGHT,
        RECOMMENDATION_ARTIFACT_DIR,
    )
    from .scoring import score_candidates
    from .validators import validate_recommendation_inputs
except ImportError:
    from config import (
        ACTIVE_PLANTINGS_FILE,
        CROP_DEFAULTS_FILE,
        DEMAND_FORECAST_FILE,
        DEMAND_WEIGHT,
        FARMER_INPUT_FILE,
        OUTPUT_FILE,
        PRICE_FORECAST_FILE,
        PRICE_WEIGHT,
        RECOMMENDATION_ARTIFACT_DIR,
    )
    from scoring import score_candidates
    from validators import validate_recommendation_inputs


SUPPORTED_LOCATIONS = {"Kuching", "Miri", "Sibu"}
FORECAST_KEY_FIELDS = ("crop", "location", "target_month")


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def load_supported_crops(path):
    if not path.exists():
        raise FileNotFoundError(f"Crop defaults file not found: {path}")

    with path.open(encoding="utf-8-sig", newline="") as input_file:
        crops = [row["crop"].strip() for row in csv.DictReader(input_file)]

    crops = [crop for crop in crops if crop]
    if not crops:
        raise ValueError("crop_defaults.csv does not contain any crops.")
    if len(crops) != len(set(crops)):
        raise ValueError("crop_defaults.csv contains duplicate crop names.")
    return crops


def normalize_target_month(value, file_label, row_number):
    text = str(value).strip()
    for date_format in ("%Y-%m", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, date_format).strftime("%Y-%m")
        except ValueError:
            continue
    raise ValueError(
        f"{file_label} row {row_number} target_month must use YYYY-MM or YYYY-MM-DD."
    )


def load_forecasts(path, prediction_field):
    if not path.exists():
        raise FileNotFoundError(f"Forecast file not found: {path}")

    required_fields = {*FORECAST_KEY_FIELDS, prediction_field}
    forecasts = {}

    with path.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing = sorted(required_fields - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}.")

        for row_number, row in enumerate(reader, start=2):
            crop = row["crop"].strip()
            location = row["location"].strip()
            target_month = normalize_target_month(
                row["target_month"], path.name, row_number
            )
            key = (crop, location, target_month)
            if key in forecasts:
                raise ValueError(
                    f"{path.name} contains a duplicate forecast for {key}."
                )
            try:
                prediction = float(row[prediction_field])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path.name} row {row_number} has an invalid {prediction_field}."
                ) from error
            if prediction < 0:
                raise ValueError(
                    f"{path.name} row {row_number} has a negative {prediction_field}."
                )

            cleaned = dict(row)
            cleaned.update(
                {
                    "crop": crop,
                    "location": location,
                    "target_month": target_month,
                    prediction_field: prediction,
                }
            )
            forecasts[key] = cleaned

    if not forecasts:
        raise ValueError(f"{path.name} does not contain any forecast rows.")
    return forecasts


def month_number(month_text):
    parsed = datetime.strptime(month_text, "%Y-%m")
    return parsed.year * 12 + parsed.month - 1


def create_candidates(farmer, supported_crops):
    demand = load_forecasts(DEMAND_FORECAST_FILE, "predicted_demand_kg")
    price = load_forecasts(
        PRICE_FORECAST_FILE, "predicted_procurement_price_rm_per_kg"
    )

    demand_keys = set(demand)
    price_keys = set(price)
    if demand_keys != price_keys:
        missing_price = len(demand_keys - price_keys)
        missing_demand = len(price_keys - demand_keys)
        raise ValueError(
            "Demand and price forecast rows do not match. "
            f"Missing price rows: {missing_price}; missing demand rows: {missing_demand}."
        )

    start_month = month_number(farmer["planting_month"])
    end_month = start_month + farmer["planning_horizon_months"] - 1
    allowed_crops = set(supported_crops)
    candidates = []

    for key in sorted(demand_keys):
        crop, location, target_month = key
        target_number = month_number(target_month)
        if location != farmer["location"]:
            continue
        if crop not in allowed_crops:
            continue
        if not start_month <= target_number <= end_month:
            continue

        demand_row = demand[key]
        price_row = price[key]
        candidates.append(
            {
                "crop": crop,
                "location": location,
                "target_month": target_month,
                "predicted_demand_kg": demand_row["predicted_demand_kg"],
                "predicted_price_rm_per_kg": price_row[
                    "predicted_procurement_price_rm_per_kg"
                ],
                "demand_forecast_source": demand_row.get(
                    "forecast_source", "unspecified"
                ),
                "demand_model_variant": demand_row.get(
                    "model_variant", "unspecified"
                ),
                "price_forecast_source": price_row.get(
                    "forecast_source", "xgboost_price"
                ),
                "price_model_variant": price_row.get(
                    "model_variant", "unspecified"
                ),
                "forecast_confidence": "not_assessed",
            }
        )

    if not candidates:
        raise ValueError(
            "No matching demand and price forecasts were found for the farmer's "
            "location and planning period."
        )
    return candidates


def add_explanations(recommendations):
    for item in recommendations:
        item["explanation"] = (
            f"Relative demand score {item['normalized_demand_score']:.2f} and "
            f"relative price score {item['normalized_price_score']:.2f} produced "
            f"a market-opportunity score of {item['opportunity_score']:.3f}."
        )
    return recommendations


def generate_recommendations():
    farmer_input = load_json(FARMER_INPUT_FILE)
    active_plantings = load_json(ACTIVE_PLANTINGS_FILE)
    supported_crops = load_supported_crops(CROP_DEFAULTS_FILE)

    validated = validate_recommendation_inputs(
        farmer_input,
        active_plantings,
        supported_locations=SUPPORTED_LOCATIONS,
        supported_crops=supported_crops,
    )
    farmer = validated["farmer_input"]
    candidates = create_candidates(farmer, supported_crops)
    recommendations = add_explanations(score_candidates(candidates))

    output = {
        "status": "success",
        "recommendation_type": "market_opportunity",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "farmer_input": farmer,
        "land_availability": validated["land_availability"],
        "scoring": {
            "normalization": "min_max_within_current_candidate_set",
            "demand_weight": DEMAND_WEIGHT,
            "price_weight": PRICE_WEIGHT,
            "confidence_in_score": False,
        },
        "limitations": [
            "Forecast confidence is not yet estimated for individual candidate rows.",
            "This ranks market opportunity and does not assess agronomic suitability.",
            "Crop growth duration is not yet used to convert planting month into harvest month.",
        ],
        "recommendations": recommendations,
    }

    RECOMMENDATION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
    return output


def main():
    output = generate_recommendations()
    best = output["recommendations"][0]
    land = output["land_availability"]

    print("CROP MARKET-OPPORTUNITY RECOMMENDATION COMPLETE")
    print(f"Location: {output['farmer_input']['location']}")
    print(f"Remaining land: {land['remaining_area_hectares']:.2f} ha")
    print(f"Candidates scored: {len(output['recommendations'])}")
    print(
        f"Top candidate: {best['crop']} for {best['target_month']} "
        f"(score {best['opportunity_score']:.3f})"
    )
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

