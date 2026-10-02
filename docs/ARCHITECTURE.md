# Project Architecture

This document explains where data, source code, notebooks, and generated artifacts belong. Execution commands are kept separately in [RUN_GUIDE.md](RUN_GUIDE.md).

## System flow

```text
Shared raw procurement and contextual data
                    |
                    v
       Shared monthly contextual table
              /                 \
             v                   v
   Demand forecasting      Price forecasting
   H1-H12 demand rows      H1-H12 price rows
             \                   /
              v                 v
          Saved model predictions
                    |
                    v
       Crop recommendation scoring
```

Demand and price forecasting share cleaned source data, but they have different targets, feature tables, splits, models, reports, and predictions. Crop recommendation consumes their prediction outputs; it does not train another forecasting model.

## Repository structure

```text
demand-forecasting/
├── README.md
├── pipeline_config.json
├── requirements.txt
├── docs/
│   ├── ARCHITECTURE.md
│   ├── RUN_GUIDE.md
│   ├── DEMAND_FORECASTING.md
│   ├── PRICE_FORECASTING.md
│   └── CROP_RECOMMENDATION.md
├── data/
│   ├── raw/
│   │   ├── Dataset_5Years_15Crops_Synthetic.xlsx
│   │   ├── contextual/
│   │   └── external/market_prices/
│   ├── external/market_prices/
│   ├── processed/
│   │   ├── shared/
│   │   ├── demand/{core,full}/
│   │   └── price/{core,full}/
│   └── recommendation/
├── scripts/
│   ├── run_preprocessing_pipeline.py
│   └── run_price_preprocessing_pipeline.py
├── src/
│   ├── preprocessing/
│   ├── demand_forecasting/
│   ├── price_forecasting/
│   ├── crop_recommendation/
│   ├── evaluation/
│   ├── prediction/
│   └── modeling/
├── notebooks/
│   ├── demand_forecasting/
│   └── price_forecasting/
├── artifacts/
│   ├── demand/{core,full}/
│   ├── price/{core,full}/
│   └── recommendation/
└── tests/
```

## Folder responsibilities

| Folder | Responsibility |
|---|---|
| `data/raw` | Original client and contextual inputs. Preprocessing must not overwrite them. |
| `data/raw/external/market_prices` | Preserved original FAMA workbooks. |
| `data/external/market_prices` | Normalized FAMA tables, coverage reports, and reviewed mappings. |
| `data/processed/shared` | Reusable monthly demand, price, weather, solar, haze, and holiday context. |
| `data/processed/demand/{core,full}` | Separate demand feature contracts, horizon rows, future rows, and splits. |
| `data/processed/price/{core,full}` | Separate price feature contracts, horizon rows, future rows, and splits. |
| `data/recommendation` | Temporary JSON/CSV inputs used before API and database integration. |
| `src/preprocessing` | Shared cleaning and contextual-data preparation. |
| `src/demand_forecasting` | Demand-specific feature engineering, splitting, tuning, and prediction. |
| `src/price_forecasting` | FAMA processing and price-specific feature engineering and splitting. |
| `src/crop_recommendation` | Validation, land calculation, normalization, scoring, and ranking. |
| `notebooks` | Colab/Jupyter training and prediction workflows. |
| `artifacts/demand` | Demand models, predictions, and reports. |
| `artifacts/price` | Price models, predictions, and reports. |
| `artifacts/recommendation` | Ranked recommendation JSON output. |

## Data ownership rules

- Raw files are immutable inputs.
- Generated tables belong under `data/processed` or `data/external`.
- Trained models and metrics belong under `artifacts`, never under `src`.
- Demand and price artifacts must not share filenames or directories.
- Notebooks may orchestrate experiments, but reusable transformation logic belongs under `src`.
- The future application should call the forecasting and recommendation logic through the separate Python AI service.

## Leakage boundary

For a forecast made at an origin month:

- Historical price, demand, weather, solar, and FAMA values must come from the origin month or earlier.
- Both variants exclude the unreliable order-level stock feature family.
- Demand full uses AQ; demand core excludes AQ.
- Price full uses AQ plus FAMA; price core excludes both.
- FAMA prices are lagged by at least one month.
- Target-month calendar and holiday fields are allowed because they are known beforehand.
- Target weather uses historical normals, not actual future observations.
- Actual target-month demand and price are labels during training only.

## Future software integration

```text
Next.js / Expo clients
        |
        v
NestJS business API and Supabase
        |
        v
Python FastAPI AI service
        |
        +-- demand full/core selector
        +-- price full/core selector per crop
        +-- recommendation engine
```

The NestJS API should own authentication, farms, planting records, and authorization. The Python service should own model loading, forecasting, and recommendation calculations.

## Runtime model selection

```text
Forecast request
    -> validate core history
    -> validate optional AQ and FAMA context
    -> full context ready? use full
    -> otherwise use core and return a warning
```

Demand selects one variant for the request. Price selection is per crop because FAMA coverage differs by crop. The reusable readiness logic is in `src/prediction/model_variant_selector.py`.
