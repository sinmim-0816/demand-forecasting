# Demand Forecasting

The demand workflow predicts monthly crop demand for horizons H1 through H12 using one global model per algorithm. The crop, location, and forecast horizon are model inputs.

Commands are documented in [RUN_GUIDE.md](RUN_GUIDE.md). Folder placement is documented in [ARCHITECTURE.md](ARCHITECTURE.md).

## Pipeline

```text
Client transactions
    -> monthly aggregation
    -> weather and solar radiation
    -> API haze indicators
    -> holidays
    -> contextual merge
    -> demand/price lag and rolling features
    -> H1-H12 transformation
    -> chronological split
    -> XGBoost / Extra Trees / stacking comparison
    -> selected demand model
```

## Preprocessing stages

### Monthly aggregation

`src/preprocessing/prepare_monthly_data.py` reads the configured client workbook and creates one monthly row per crop and location. The start and end dates are inferred from the client data rather than hardcoded.

Main values include demand, weighted procurement price, order count, completed/cancelled order counts, and stock context.

### Weather and solar radiation

`src/preprocessing/prepare_weather_data.py` prepares monthly temperature, rainfall, and shortwave solar-radiation context. Weather coverage follows the client dataset automatically.

### Air quality

`src/preprocessing/prepare_air_quality_data.py` creates monthly Sarawak API features such as average API, maximum API, moderate-or-worse days, unhealthy days, and observation count.

### Holidays

`src/preprocessing/prepare_holidays_data.py` creates public-holiday and major-festival indicators. Future prediction requires verified calendar coverage for every target month.

### Contextual merge

`src/preprocessing/merge_contextual_data.py` combines the monthly platform table with weather, solar radiation, air quality, and holidays.

### Historical features

`src/demand_forecasting/create_features.py` creates historical demand, price, stock, and rolling features. Lag values stay anchored to the forecast-origin month.

`supermarket_order_count` and `trader_order_count` remain available in the shared monthly table for reporting, but are deliberately removed from the demand model contract because CropWise cannot observe a reliable order type.

### H1-H12 transformation

`src/demand_forecasting/create_horizon_dataset.py` expands each forecast origin into 12 rows and adds:

- `forecast_horizon`
- `target_month`, year, month number, quarter, and season
- target holiday/festival indicators
- leakage-safe target-month weather normals
- `future_demand_kg` for historical training rows

Historical rows without a known future target are excluded from training. Latest-origin rows are retained in `future_horizon_features.csv` for real prediction.

## Model input rules

- Historical demand, price, stock, weather, solar, and haze features come from the origin month or earlier.
- `future_demand_kg` is the target and never an input feature.
- Actual target-month price, rainfall, temperature, solar radiation, or API cannot be used.
- Target calendar and holiday values are allowed because they are known in advance.
- Future weather context uses historical monthly normals, not unknown actual observations.

## Chronological evaluation

`src/demand_forecasting/prepare_model_data.py` creates non-overlapping train, validation, and test periods based on `target_month`. Random row splitting is not appropriate because it would allow later time periods to influence earlier evaluations.

Metrics:

| Metric | Meaning |
|---|---|
| MAE | Average absolute prediction error in kilograms. Lower is better. |
| RMSE | Similar to MAE but penalizes large errors more strongly. Lower is better. |
| WAPE | Total absolute error divided by total actual demand. Lower is better. |
| R² | Fraction of target variation explained by the model. Higher is better. |

## Models

- XGBoost is the selected primary model after tuning.
- Extra Trees provides a different tree-based comparison.
- Ridge stacking combines their predictions but is retained only if it improves the untouched test period.
- The original and tuned XGBoost artifacts should both be preserved for reproducibility.

## Important outputs

```text
data/processed/shared/monthly_demand_contextual.csv
data/processed/demand/model_features.csv
data/processed/demand/horizon_training_dataset.csv
data/processed/demand/future_horizon_features.csv
data/processed/demand/model_splits/
artifacts/demand/models/
artifacts/demand/predictions/
artifacts/demand/reports/
```

## Updating with new client data

Replace the configured client workbook, confirm `pipeline_config.json`, rerun the demand pipeline, recreate splits, and retrain the models. Date coverage is inferred automatically, but future holiday coverage still needs to exist for prediction target months.
