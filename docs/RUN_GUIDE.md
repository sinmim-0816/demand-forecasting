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

The default builds both `core` and `full`. To build only one contract:

```powershell
python scripts\run_preprocessing_pipeline.py --start-at features --variant core
python scripts\run_preprocessing_pipeline.py --start-at features --variant full
```

Resume from a later stage when earlier outputs are already current:

```powershell
python scripts\run_preprocessing_pipeline.py --start-at weather
python scripts\run_preprocessing_pipeline.py --start-at air-quality
python scripts\run_preprocessing_pipeline.py --start-at holidays
python scripts\run_preprocessing_pipeline.py --start-at merge
python scripts\run_preprocessing_pipeline.py --start-at features
python scripts\run_preprocessing_pipeline.py --start-at horizons
python scripts\run_preprocessing_pipeline.py --start-at splits
```

The runner now creates chronological splits automatically. Individual execution is still available:

```powershell
python src\demand_forecasting\prepare_model_data.py --variant core
python src\demand_forecasting\prepare_model_data.py --variant full
```

### 3. Train demand models

Run these notebooks in Google Colab:

```text
notebooks/demand_forecasting/train_xgboost.ipynb
notebooks/demand_forecasting/train_etr_demand.ipynb
notebooks/demand_forecasting/train_stacking_ensemble_demand.ipynb
```

Each notebook trains both variants in one run. Upload the two horizon files as:

```text
MyDrive/demand-forecasting/Dataset/demand/core/horizon_training_dataset.csv
MyDrive/demand-forecasting/Dataset/demand/full/horizon_training_dataset.csv
```

The Python training entry points remain available for local use:

```powershell
python src\demand_forecasting\train_xgboost.py --variant core
python src\demand_forecasting\train_xgboost.py --variant full
python src\demand_forecasting\train_etr.py --variant core
python src\demand_forecasting\train_etr.py --variant full
```

### 4. Tune demand XGBoost

```powershell
python src\demand_forecasting\tune_xgboost.py --variant core --trials 12 --folds 3 --validation-months 3
python src\demand_forecasting\tune_xgboost.py --variant full --trials 12 --folds 3 --validation-months 3
```

### 5. Analyze the tuned model

```powershell
python src\demand_forecasting\analyze_xgboost_errors.py
```

### 6. Generate future demand predictions

Preferred reproducible command:

```powershell
python src\demand_forecasting\predict_future_xgboost.py --variant core
python src\demand_forecasting\predict_future_xgboost.py --variant full
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

The default builds both variants. Core skips FAMA; full requires verified FAMA and AQ context.

Resume from a later price stage:

```powershell
python scripts\run_price_preprocessing_pipeline.py --start-at features
python scripts\run_price_preprocessing_pipeline.py --start-at horizons
python scripts\run_price_preprocessing_pipeline.py --start-at splits
```

Individual price stages are:

```powershell
python src\price_forecasting\prepare_fama_price_data.py
python src\price_forecasting\create_price_features.py --variant core
python src\price_forecasting\create_price_features.py --variant full
python src\price_forecasting\create_price_horizon_dataset.py --variant core
python src\price_forecasting\create_price_horizon_dataset.py --variant full
python src\price_forecasting\prepare_model_data.py --variant core
python src\price_forecasting\prepare_model_data.py --variant full
```

### 2. Train price models

Run these notebooks in Google Colab:

```text
notebooks/price_forecasting/train_xgboost.ipynb
notebooks/price_forecasting/train_etr.ipynb
notebooks/price_forecasting/train_stacking_ensemble.ipynb
```

Each notebook trains both variants. Upload:

```text
MyDrive/demand-forecasting/Dataset/price/core/price_horizon_training_dataset.csv
MyDrive/demand-forecasting/Dataset/price/full/price_horizon_training_dataset.csv
```

### 3. Generate future price predictions

After copying both trained variant artifacts into the local artifact folders, run:

```powershell
python src\price_forecasting\predict_future_xgboost.py --variant auto
```

The auto mode uses full only for rows with complete AQ and confirmed FAMA context; all other crops use core. The local notebook remains available:

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
1. Run demand preprocessing for core and full (splits are included)
2. Train and compare both demand variants
3. Generate future demand predictions with the readiness-selected variant
4. Run price preprocessing for core and full
5. Train and compare both price variants
6. Generate price predictions per crop with the readiness-selected variant
7. Run crop recommendation
```

Do not rerun model training merely because a prediction file is regenerated. Retrain when historical data, features, mappings, preprocessing logic, or model settings change.

## Variant artifact contract

Training saves each variant independently:

```text
artifacts/<demand|price>/<core|full>/models/
artifacts/<demand|price>/<core|full>/predictions/
artifacts/<demand|price>/<core|full>/reports/
```

Do not copy a core pipeline into a full directory or rename one variant as the other. The metadata feature list and readiness flags must match the saved pipeline.
