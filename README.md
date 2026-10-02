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

## Main outputs

```text
data/processed/demand/{core,full}/  Demand features, horizons, future rows, splits
data/processed/price/{core,full}/   Price features, horizons, future rows, splits
artifacts/demand/{core,full}/       Demand models, predictions, reports
artifacts/price/{core,full}/        Price models, predictions, reports
artifacts/recommendation/           Ranked crop recommendations
```

Raw inputs are preserved under `data/raw`. Generated data belongs under `data/processed` or `data/external`; trained models and reports belong under `artifacts`.
