from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RECOMMENDATION_DATA_DIR = PROJECT_ROOT / "data" / "recommendation"
RECOMMENDATION_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "recommendation"

FARMER_INPUT_FILE = RECOMMENDATION_DATA_DIR / "sample_farmer_input.json"
ACTIVE_PLANTINGS_FILE = RECOMMENDATION_DATA_DIR / "sample_active_plantings.json"
CROP_DEFAULTS_FILE = RECOMMENDATION_DATA_DIR / "crop_defaults.csv"
OUTPUT_FILE = RECOMMENDATION_ARTIFACT_DIR / "recommendations.json"

# Simulated forecast inputs used before API and database integration
DEMAND_FORECAST_FILE = RECOMMENDATION_DATA_DIR / "simulated_demand_predictions.csv"
PRICE_FORECAST_FILE = RECOMMENDATION_DATA_DIR / "simulated_price_predictions.csv"

# Neutral market-opportunity weights
# Confidence is reported separately and does not change market attractiveness.
DEMAND_WEIGHT = 0.50
PRICE_WEIGHT = 0.50

