# Agricultural Demand, Price, and Crop Recommendation

This repository contains three connected but separate components:

1. H1-H12 crop demand forecasting
2. H1-H12 procurement-price forecasting
3. Rule-based crop market-opportunity recommendation

The detailed documentation is separated by responsibility so that model instructions, architecture, and commands do not become mixed together.

## Documentation

| Document | Contents |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | Repository structure, data flow, ownership boundaries, and software integration. |
| [Run guide](docs/RUN_GUIDE.md) | Authoritative preprocessing, training, prediction, and recommendation commands. |
| [Demand forecasting](docs/DEMAND_FORECASTING.md) | Demand stages, features, leakage rules, models, evaluation, and outputs. |
| [Price forecasting](docs/PRICE_FORECASTING.md) | FAMA preprocessing, price features, horizons, models, and outputs. |
| [Crop recommendation](docs/CROP_RECOMMENDATION.md) | Inputs, validation, land calculation, scoring, limitations, and API migration. |

## Quick setup

Python 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Update `pipeline_config.json` when an input workbook, worksheet, date column, air-quality source, or FAMA workbook changes.

## Quick start

Demand preprocessing:

```powershell
python scripts\run_preprocessing_pipeline.py
```

Price preprocessing:

```powershell
python scripts\run_price_preprocessing_pipeline.py
```

Crop recommendation after future demand and price predictions exist:

```powershell
python src\crop_recommendation\recommend_crops.py
```

For training notebooks, tuning commands, prediction commands, and resume options, use [docs/RUN_GUIDE.md](docs/RUN_GUIDE.md).

## Evaluation status and prediction intervals

The demand and price models currently produce point forecasts. They use a
chronological train/validation/test design rather than a random split:

```text
Training data             -> fit candidate models
Validation/rolling folds  -> compare models and select settings
Training + validation     -> fit the final model
Held-out test data        -> evaluate final point-forecast performance
```

This establishes held-out performance for metrics such as MAE, RMSE, WAPE,
and R-squared. It does not establish prediction intervals. The current API and
forecast files therefore must not describe their outputs as 80% or 95%
uncertainty ranges.

Prediction intervals are planned future work. A valid implementation must:

1. Reserve calibration observations that were not used to fit or tune the
   corresponding model, or use a carefully designed rolling-origin
   calibration procedure after the model settings are fixed.
2. Calculate lower and upper bounds, preferably separately for H1-H12 when
   enough calibration observations are available.
3. Evaluate the bounds on unseen chronological test data.
4. Report empirical coverage and average interval width before exposing the
   intervals through the API or frontend.

For example, a nominal 80% interval should contain approximately 80% of the
unseen test targets. The held-out test set must not be used both to choose the
interval width and to report its final coverage.

## Main outputs

```text
data/processed/demand/{core,full}/  Demand features, horizons, future rows, splits
data/processed/price/{core,full}/   Price features, horizons, future rows, splits
artifacts/demand/{core,full}/       Demand models, predictions, reports
artifacts/price/{core,full}/        Price models, predictions, reports
artifacts/recommendation/           Ranked crop recommendations
```

Raw inputs are preserved under `data/raw`. Generated data belongs under `data/processed` or `data/external`; trained models and reports belong under `artifacts`.
