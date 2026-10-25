# Supply Forecasting Preprocessing

This pipeline prepares data for direct H1-H12 monthly crop-production forecasting. It does not generate synthetic production values and it does not train a model yet.

## Raw data contract

Place the future real or synthetic file at:

```text
data/raw/supply/supply_history.xlsx
```

The default worksheet is `Supply Data` and the required columns are:

| Column | Meaning |
|---|---|
| `Month` | Observation date or month. Any date in the month is normalized to its first day. |
| `Crop` | Crop name using the same naming convention as the other forecasts. |
| `Location` | Location matching the contextual weather locations, such as Kuching, Sibu, or Miri. |
| `Production (kg)` | Observed local production in kilograms. Must be zero or positive. |

Multiple records for the same crop, location, and month are summed. Missing months are not automatically converted to zero because a missing observation is different from confirmed zero production. Change names or paths under `supply_data` in `pipeline_config.json` when necessary.

## Pipeline

```text
Raw production records
    -> monthly crop-location production
    -> historical weather, solar, calendar, and optional AQ merge
    -> production lags and rolling features
    -> direct H1-H12 training rows and future feature rows
    -> chronological train, validation, and test splits
```

Core and Full use the same target:

| Variant | Context |
|---|---|
| `core` | Production history, weather, solar radiation, and calendar |
| `full` | Core context plus air quality |

The primary target is `future_production_kg`. Target-month temperature, rainfall, and solar inputs are historical normals calculated using observations available no later than each forecast origin. Actual future weather is never copied into training features.

## Commands

Run both variants:

```powershell
python scripts\run_supply_preprocessing_pipeline.py
```

Run only Core or Full:

```powershell
python scripts\run_supply_preprocessing_pipeline.py --variant core
python scripts\run_supply_preprocessing_pipeline.py --variant full
```

Resume from an existing stage:

```powershell
python scripts\run_supply_preprocessing_pipeline.py --start-at features --variant both
python scripts\run_supply_preprocessing_pipeline.py --start-at horizons --variant both
python scripts\run_supply_preprocessing_pipeline.py --start-at splits --variant both
```

Until the configured supply file exists, the first command intentionally stops with a message describing the required input. This protects the pipeline from silently inventing production values.

## Outputs

```text
data/processed/supply/shared/monthly_supply.*
data/processed/supply/{core,full}/supply_features.*
data/processed/supply/{core,full}/supply_horizon_training_dataset.*
data/processed/supply/{core,full}/future_supply_horizon_features.csv
data/processed/supply/{core,full}/model_splits/{train,validation,test}.csv
artifacts/supply/{core,full}/reports/model_data_split.json
```

## Minimum history

The feature stage needs at least 12 consecutive prior months for every crop-location series. In practice, several years are preferable because the direct H1-H12 transformation and chronological validation/test windows also consume historical coverage.

## Current boundary

This implementation completes preprocessing only. Model training, tuning, evaluation, inference, and CropWise integration should begin after the production dataset has been reviewed for units, coverage, missing months, and crop/location definitions.
