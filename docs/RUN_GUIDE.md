# Run Guide

This document is the single command reference for preprocessing, training, prediction, and crop recommendation. Run commands from the project root in PowerShell.

## First-time environment setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Update `pipeline_config.json` when the client workbook path, worksheet, date column, air-quality workbook, or FAMA workbook changes.

## Demand workflow

### 1. Run complete demand preprocessing

```powershell
python scripts\run_preprocessing_pipeline.py
```

Resume from a later stage when earlier outputs are already current:

```powershell
python scripts\run_preprocessing_pipeline.py --start-at weather
python scripts\run_preprocessing_pipeline.py --start-at air-quality
python scripts\run_preprocessing_pipeline.py --start-at holidays
python scripts\run_preprocessing_pipeline.py --start-at merge
python scripts\run_preprocessing_pipeline.py --start-at features
python scripts\run_preprocessing_pipeline.py --start-at horizons
```

### 2. Create chronological model splits

```powershell
python src\demand_forecasting\prepare_model_data.py
```

### 3. Train demand models

Run these notebooks in Google Colab:

```text
notebooks/demand_forecasting/train_xgboost_demand.ipynb
notebooks/demand_forecasting/train_etr_demand.ipynb
notebooks/demand_forecasting/train_stacking_ensemble_demand.ipynb
```

The Python training entry points remain available for local use:

```powershell
python src\demand_forecasting\train_xgboost.py
python src\demand_forecasting\train_etr.py
```

### 4. Tune demand XGBoost

```powershell
python src\demand_forecasting\tune_xgboost.py --trials 12 --folds 3 --validation-months 3
```

### 5. Analyze the tuned model

```powershell
python src\demand_forecasting\analyze_xgboost_errors.py
```

### 6. Generate future demand predictions

Preferred reproducible command:

```powershell
python src\demand_forecasting\predict_future_xgboost.py
```

The local notebooks are also available:

```text
notebooks/demand_forecasting/predict_xgboost.ipynb
notebooks/demand_forecasting/predict_etr.ipynb
```

## Price workflow

The shared demand preprocessing must have produced `data/processed/shared/monthly_demand_contextual.csv` before running this workflow.

### 1. Run complete price preprocessing

```powershell
python scripts\run_price_preprocessing_pipeline.py
```

Resume from a later price stage:

```powershell
python scripts\run_price_preprocessing_pipeline.py --start-at features
python scripts\run_price_preprocessing_pipeline.py --start-at horizons
python scripts\run_price_preprocessing_pipeline.py --start-at splits
```

Individual price stages are:

```powershell
python src\price_forecasting\prepare_fama_price_data.py
python src\price_forecasting\create_price_features.py
python src\price_forecasting\create_price_horizon_dataset.py
python src\price_forecasting\prepare_model_data.py
```

### 2. Train price models

Run these notebooks in Google Colab:

```text
notebooks/price_forecasting/train_xgboost.ipynb
notebooks/price_forecasting/train_etr.ipynb
notebooks/price_forecasting/train_stacking_ensemble.ipynb
```

### 3. Generate future price predictions

Run locally with the same package versions used for training:

```text
notebooks/price_forecasting/predict_xgboost.ipynb
```

## Crop recommendation workflow

The engine requires current future demand and future price prediction CSV files.

```powershell
python src\crop_recommendation\recommend_crops.py
```

Output:

```text
artifacts/recommendation/recommendations.json
```

Before running, update these temporary inputs when needed:

```text
data/recommendation/sample_farmer_input.json
data/recommendation/sample_active_plantings.json
data/recommendation/crop_defaults.csv
```

## Recommended full execution order

```text
1. Run demand preprocessing
2. Prepare demand splits
3. Retrain/select demand model
4. Generate future demand predictions
5. Run price preprocessing
6. Retrain/select price model
7. Generate future price predictions
8. Run crop recommendation
```

Do not rerun model training merely because a prediction file is regenerated. Retrain when historical data, features, mappings, preprocessing logic, or model settings change.
