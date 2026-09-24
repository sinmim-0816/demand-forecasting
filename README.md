# Agricultural Demand and Price Forecasting

This project converts shared procurement and contextual data into separate leakage-safe demand and price forecasting workflows. Demand forecasting is implemented; price forecasting has dedicated folders ready for its own feature, model, prediction, and report outputs.

The final dataset supports one global model per algorithm, such as Extra Trees or XGBoost, that predicts horizons H1 through H12 using `forecast_horizon` as an input feature.

## Pipeline overview

```text
Raw client Excel workbook
        |
        v
1. Clean and aggregate monthly demand
        |
        +--------------------+
        |                    |
        v                    v
2. Prepare weather     3. Prepare holidays
        |                    |
        +---------+----------+
                  |
                  v
4. Merge contextual data
                  |
                  v
5. Create lag and rolling features
                  |
                  v
6. Expand into H1-H12 horizon rows
                  |
                  v
horizon_training_dataset.csv
```

## Project structure

```text
demand-forecasting/
├── pipeline_config.json
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── Dataset_3Years_15Crops_Synthetic.xlsx
│   │   └── contextual/
│   │       └── Contextual_Data.xlsx
│   ├── processed/
│   │   ├── shared/             # Cleaned monthly/contextual tables
│   │   ├── demand/             # Demand features, horizons, and splits
│   │   └── price/              # Price features, horizons, and splits
│   └── external/
│       └── market_prices/      # Optional FAMA/wholesale inputs
├── scripts/
│   └── run_preprocessing_pipeline.py
├── src/
│   ├── preprocessing/
│   │   ├── pipeline_config.py
│   │   ├── prepare_monthly_data.py
│   │   ├── prepare_weather_data.py
│   │   ├── prepare_holidays_data.py
│   │   └── merge_contextual_data.py
│   ├── demand_forecasting/
│   │   ├── create_features.py
│   │   ├── create_horizon_dataset.py
│   │   ├── prepare_model_data.py
│   │   ├── train_etr.py
│   │   ├── train_xgboost.py
│   │   ├── tune_xgboost.py
│   │   └── analyze_xgboost_errors.py
│   ├── price_forecasting/      # Separate price workflow (to implement)
│   ├── modeling/               # Compatibility entry points for old commands
│   ├── evaluation/
│   └── prediction/
├── notebooks/
│   ├── demand_forecasting/
│   │   ├── train_etr.ipynb
│   │   ├── train_xgboost.ipynb
│   │   ├── train_stacking_ensemble.ipynb
│   │   ├── predict_xgboost.ipynb
│   │   └── predict_etr.ipynb
│   └── price_forecasting/      # Future price notebooks
├── artifacts/
│   ├── demand/
│   │   ├── models/
│   │   ├── predictions/
│   │   └── reports/
│   └── price/
│       ├── models/
│       ├── predictions/
│       └── reports/
└── tests/
```

### Folder responsibilities

| Folder | Purpose |
|---|---|
| `data/raw` | Original client and contextual source data. These files are inputs and should not be overwritten by modelling code. |
| `data/processed/shared` | Reusable cleaned monthly demand, price, weather, and holiday context. |
| `data/processed/demand` | Demand-specific features, H1–H12 rows, future features, and splits. |
| `data/processed/price` | Price-specific features, H1–H12 rows, future features, and splits. |
| `data/external/market_prices` | Optional external market-price sources such as FAMA data. |
| `src/preprocessing` | Shared cleaning, aggregation, weather, holiday, and contextual merging. |
| `src/demand_forecasting` | Demand feature engineering, splitting, training, tuning, and evaluation. |
| `src/price_forecasting` | Price feature engineering, splitting, training, tuning, and evaluation. |
| `src/modeling` | Compatibility wrappers that preserve the previous demand commands. |
| `src/evaluation` | Shared metrics and model-comparison logic. |
| `src/prediction` | Loading selected saved models and generating future forecasts. |
| `scripts` | Simple commands that coordinate complete pipelines. |
| `notebooks` | Colab/Jupyter exploration and experiments. Final reusable logic should be moved into `src`. |
| `artifacts/demand` | Existing demand models, predictions, tuning results, and reports. |
| `artifacts/price` | Price models, predictions, and reports, kept separate from demand. |
| `tests` | Automated checks for preprocessing, features, splitting, and prediction. |

Keep source code out of `data` and do not save trained models under `src`.

## Requirements and installation

Python 3.10 or newer is recommended.

From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`pyarrow` is required for Parquet output. Most later stages can still produce CSV and Excel files without it, but the monthly aggregation stage expects it to be installed.

## Configuration

Edit `pipeline_config.json` when the client workbook name, worksheet, or date-column name changes:

```json
{
  "client_data": {
    "file": "data/raw/Dataset_5Years_15Crops_Synthetic.xlsx",
    "sheet": "Procurement Data",
    "date_column": "Procurement Date"
  }
}
```

Relative file paths are resolved from the project root. The weather date range is derived from the valid client procurement dates. The holiday calendar covers the same historical period plus the next 12 forecast months automatically.

The configured five-year workbook covers January 2021–December 2025. Its added 2021–2022 transactions are generated synthetic examples based on the original 2023–2025 synthetic workbook; they are not independently observed client history. Do not present a higher score from this expanded workbook as proof that more real data improved the model. The original three-year workbook remains unchanged under `data/raw`.

## Demand forecasting — quick run guide

Run these commands from the project root after activating `.venv`. They are shown in the order normally required.

### Complete demand preprocessing

```powershell
python scripts\run_preprocessing_pipeline.py
```

This runs raw-data cleaning, monthly aggregation, weather, holidays, contextual merging, demand features, and H1–H12 transformation. Its final outputs are:

```text
data/processed/demand/horizon_training_dataset.csv
data/processed/demand/future_horizon_features.csv
```

### Prepare chronological model splits

```powershell
python src\demand_forecasting\prepare_model_data.py
```

This must be rerun after the horizon training dataset changes.

### Train the original XGBoost model

```powershell
python src\demand_forecasting\train_xgboost.py
```

For CUDA-enabled XGBoost:

```powershell
python src\demand_forecasting\train_xgboost.py --device cuda
```

### Tune and train the selected XGBoost model

```powershell
python src\demand_forecasting\tune_xgboost.py --trials 12 --folds 3 --validation-months 3
```

Use `--evaluate-test` only for the final held-out comparison:

```powershell
python src\demand_forecasting\tune_xgboost.py --trials 12 --folds 3 --validation-months 3 --evaluate-test
```

The current selected production candidate is `artifacts/demand/models/global_xgboost_tuned_pipeline.joblib`. The original XGBoost pipeline remains available for comparison.

### Train Extra Trees

```powershell
python src\demand_forecasting\train_etr.py
```

### Run the stacking experiment

Open and run the following Colab notebook after XGBoost and Extra Trees predictions exist:

```text
notebooks/demand_forecasting/train_stacking_ensemble.ipynb
```

The current stacker is an experimental comparison and is not the selected deployable model.

### Compare original and tuned XGBoost

```powershell
python src\demand_forecasting\analyze_xgboost_errors.py
```

### Generate future demand predictions

Run one of these notebooks locally:

```text
notebooks/demand_forecasting/predict_xgboost.ipynb
notebooks/demand_forecasting/predict_etr.ipynb
```

`predict_xgboost.ipynb` uses the tuned model by default. Set `MODEL_VARIANT = "original"` inside the notebook to use the retained original model.

### Full demand workflow summary

```text
run_preprocessing_pipeline.py
        ↓
prepare_model_data.py
        ↓
train_xgboost.py ──→ tune_xgboost.py
        └──────────→ train_etr.py
                            ↓
              optional stacking notebook
                            ↓
        compare metrics and select model
                            ↓
              prediction notebook
```

The preprocessing runner does not automatically retrain models. This separation prevents new input data from silently replacing validated model artifacts.

## Demand preprocessing — complete pipeline

Run this command from the project root:

```powershell
python scripts\run_preprocessing_pipeline.py
```

The runner executes all six stages in order and stops immediately if a stage fails validation.

To resume from a particular stage:

```powershell
python scripts\run_preprocessing_pipeline.py --start-at weather
python scripts\run_preprocessing_pipeline.py --start-at holidays
python scripts\run_preprocessing_pipeline.py --start-at merge
python scripts\run_preprocessing_pipeline.py --start-at features
python scripts\run_preprocessing_pipeline.py --start-at horizons
```

Starting partway through assumes all required outputs from earlier stages already exist and are current.

## Demand preprocessing — individual stages

Each stage can also be executed directly from the project root:

```powershell
python src\preprocessing\prepare_monthly_data.py
python src\preprocessing\prepare_weather_data.py
python src\preprocessing\prepare_holidays_data.py
python src\preprocessing\merge_contextual_data.py
python src\demand_forecasting\create_features.py
python src\demand_forecasting\create_horizon_dataset.py
```

## Stage 1 — Prepare monthly demand

Script: `src/preprocessing/prepare_monthly_data.py`

Input:

```text
Configured client Excel workbook
Configured procurement worksheet
```

The stage:

- validates the expected client columns;
- removes empty and duplicate records;
- standardizes text, units, quantities, prices, and dates;
- converts quantities to kilograms;
- separates invalid transactions for inspection;
- aggregates transactions by `month + crop + location`;
- calculates demand, order, stock, status, buyer-type, and weighted-price measures;
- checks for missing months within each crop/location series.

Primary outputs:

```text
data/processed/shared/cleaned_procurement_transactions.csv
data/processed/shared/cleaned_procurement_transactions.parquet
data/processed/shared/monthly_demand.csv
data/processed/shared/monthly_demand.parquet
data/processed/shared/monthly_demand.xlsx
data/processed/shared/data_quality_report.csv
```

`demand_kg` is the forecasting target before horizon expansion.

## Stage 2 — Prepare weather

Script: `src/preprocessing/prepare_weather_data.py`

The stage reads the client workbook to determine its minimum and maximum procurement dates. No weather start or end date needs to be edited manually.

It downloads daily ERA5 weather through the Open-Meteo Historical Weather API for Kuching, Sibu, and Miri, then aggregates it by `month + location`.

Weather variables include:

```text
Average temperature
Average minimum temperature
Average maximum temperature
Total precipitation
Observation-day count
```

Output:

```text
data/raw/contextual/Contextual_Data.xlsx → Weather sheet
```

This stage requires an internet connection.

## Stage 3 — Prepare holidays

Script: `src/preprocessing/prepare_holidays_data.py`

The stage uses the open-source `holidays` Python package to generate Malaysian public holidays for the Sarawak subdivision (`MY-SWK`). It covers the client data period plus the next 12 forecast months automatically, so no start date, end date, or yearly holiday list normally needs to be edited.

It creates these monthly features:

```text
Public holiday count
CNY
Hari Raya
Gawai
Christmas
Major festival
```

Output:

```text
data/raw/contextual/Contextual_Data.xlsx → Holidays sheet
```

Keep the `holidays` dependency current when preparing forecasts for a newly announced year. If the government announces an exceptional one-off change not yet included by the package, use `HOLIDAY_ADDITIONS` or `HOLIDAY_REMOVALS` near the top of `prepare_holidays_data.py` and document the official source.

## Stage 4 — Merge contextual data

Script: `src/preprocessing/merge_contextual_data.py`

Merge relationships:

```text
monthly_demand
    LEFT JOIN Weather
    ON month + location

    LEFT JOIN Holidays
    ON month
```

The stage validates required columns, unique merge keys, valid month values, and contextual completeness. It fails instead of silently producing training rows with missing weather or holidays.

Outputs:

```text
data/processed/shared/monthly_demand_contextual.csv
data/processed/shared/monthly_demand_contextual.parquet
data/processed/shared/monthly_demand_contextual.xlsx
```

## Stage 5 — Create historical features

Script: `src/demand_forecasting/create_features.py`

Features are created independently within every `crop + location` series.

Calendar features:

```text
year
month_number
quarter
season
```

Historical-demand features:

```text
demand_lag_1
demand_lag_3
demand_lag_6
demand_lag_12
rolling_demand_mean_3
rolling_demand_mean_6
rolling_demand_std_3
```

Historical-price features:

```text
price_lag_1
price_lag_3
rolling_price_mean_3
```

All rolling features shift by one month before calculating the statistic. Therefore, the current target value is never included in its own historical feature.

The stage checks that each crop/location series contains complete, unique monthly observations. Rows without the required 12-month history are removed from the model-ready feature table.

Outputs:

```text
data/processed/demand/model_features.csv
data/processed/demand/model_features.parquet
data/processed/demand/model_features.xlsx
```

For the current dataset, 1,620 monthly rows become 1,080 feature rows after the required history is applied.

## Stage 6 — Create the H1–H12 horizon dataset

Script: `src/demand_forecasting/create_horizon_dataset.py`

This stage converts each eligible forecast origin into as many as twelve supervised-learning rows:

| Forecast origin | Horizon | Target month | Target |
|---|---:|---|---:|
| January 2024 | 1 | February 2024 | February demand |
| January 2024 | 2 | March 2024 | March demand |
| January 2024 | 12 | January 2025 | January 2025 demand |

It adds:

```text
forecast_horizon
forecast_origin_month
target_month
target_year
target_month_number
target_quarter
target_season
target_public_holiday_count
target_cny
target_hari_raya
target_gawai
target_christmas
target_major_festival
future_demand_kg
```

### Target-month weather patterns

The stage also creates:

```text
target_month_normal_temperature
target_month_normal_precipitation
target_month_weather_history_years
```

These are historical weather patterns for the target calendar month and location. For example, a July Kuching target uses previous Kuching July observations.

The calculation is leakage-safe: for a historical forecast made in January 2024, only weather observed on or before January 2024 may contribute to its target-month normal. Weather occurring after the forecast origin is excluded.

The training table does not use actual target-month price, temperature, or rainfall because those values would not be known when making a real future forecast.

Outputs:

```text
data/processed/demand/horizon_training_dataset.csv
data/processed/demand/horizon_training_dataset.parquet
data/processed/demand/horizon_training_dataset.xlsx
data/processed/demand/future_horizon_features.csv
```

`horizon_training_dataset` contains rows whose future demand is historically known. This is the main input for chronological model splitting and training.

`future_horizon_features.csv` contains H1–H12 rows for the latest forecast origin. `future_demand_kg` is empty because it is what the trained model must predict.

For the current dataset, the stage produces 9,450 historical training rows and 540 latest-origin future rows.

## Data-leakage rules

The following rules must remain true when the pipeline is changed:

1. Demand and price lags are calculated within the same crop/location series.
2. Rolling features use `shift(1)` before rolling calculations.
3. `future_demand_kg` comes only from the target month and is used only as the label.
4. Actual target-month price, temperature, and rainfall are not model inputs.
5. Target-month calendar and verified holiday information may be used because they are known beforehand.
6. Target weather normals use only weather available by the forecast-origin date.
7. Train, validation, and test sets must be separated chronologically, never through random row splitting.

## Updating the pipeline for new client data

1. Put the new workbook under `data/raw`, or replace the existing workbook.
2. Update `pipeline_config.json` if its filename, worksheet, or date column differs.
3. Update project dependencies so the latest supported holiday years are available.
4. Close generated Excel files if they are open in Microsoft Excel.
5. Run the complete pipeline.

```powershell
python scripts\run_preprocessing_pipeline.py
```

Weather dates, holiday months, lag features, and horizon targets update from the new client date range automatically.

## Important future-prediction note

The latest-origin prediction template targets the next 12 months. Its target holiday features come from the complete generated Sarawak calendar rather than the historical demand table. The horizon script validates these fields and warns if the installed `holidays` package cannot supply the required period.

## Demand modelling reference

Preprocessing and multi-horizon transformation are complete. The next stage begins with `data/processed/demand/horizon_training_dataset.csv`.

Implemented modelling flow:

```text
horizon_training_dataset.csv
        |
        v
Chronological train/validation/test split
        |
        +--------------------+
        |                    |
        v                    v
Global ETR             Global XGBoost
        |                    |
        +----------+---------+
                   |
                   v
          Optional Ridge stacker
                   |
                   v
       Compare ETR, XGBoost, ensemble
                   |
                   v
          Select and save final model
                   |
                   v
       Predict future H1-H12 demand
```

Demand modules and future shared helpers:

```text
src/demand_forecasting/prepare_model_data.py
src/demand_forecasting/train_etr.py
src/demand_forecasting/train_xgboost.py
src/demand_forecasting/tune_xgboost.py
src/demand_forecasting/analyze_xgboost_errors.py
src/evaluation/metrics.py
src/evaluation/evaluate_models.py
src/prediction/predict_future.py
scripts/run_training_pipeline.py
scripts/run_prediction.py
```

The demand modelling implementation:

1. creates chronological or rolling-origin train/validation/test splits;
2. fits categorical encoding using training data only;
3. trains one global Extra Trees model;
4. trains and tunes one global XGBoost model;
5. generates leakage-safe base-model predictions for experimental Ridge stacking;
6. compares ETR, XGBoost, and the ensemble overall and for every horizon;
7. saves the complete fitted pipelines and evaluation results.

A small seasonal-naive calculation may be included inside evaluation as a sanity check, but it does not require its own script.

Categorical values such as `crop`, `location`, `origin_season`, and `target_season` should remain readable in the stored dataset. Encode them inside each fitted scikit-learn pipeline with `OneHotEncoder(handle_unknown="ignore")`. Saving the complete pipeline preserves the fitted encoder, feature order, and model together.

The model should treat `crop`, `location`, and `forecast_horizon` as predictors and `future_demand_kg` as the target.

Do not perform a random row split. Related horizons and adjacent months can otherwise leak future information into training. Split by time and fit all encoders, imputers, and models using training data only.

### Prepare chronological model data

Run:

```powershell
python src\demand_forecasting\prepare_model_data.py
```

This script selects 52 model features, defines four categorical features, and creates dynamic target-month splits. With the current dataset, the last three target months form the test set, the preceding three target months form validation, and all earlier target months form training:

```text
Train:      February 2022 through June 2025   19,170 rows
Validation: July 2025 through September 2025  1,620 rows
Test:       October 2025 through December 2025 1,620 rows
```

Every split is validated to contain all H1–H12 horizons. Outputs are saved to `data/processed/demand/model_splits`, and split metadata is saved to `artifacts/demand/reports/model_data_split.json`.

The module exposes `build_preprocessor(numeric_features)` for the training scripts. It uses median imputation for numeric columns and most-frequent imputation plus one-hot encoding for categorical columns. The training scripts must place this preprocessor and their regressor in one scikit-learn `Pipeline` so it is fitted using training data only.

### Train global XGBoost

The reusable training module is `src/demand_forecasting/train_xgboost.py`. It performs two fits:

1. fit on training data and create honest validation predictions;
2. refit a new final pipeline on training plus validation data and create untouched test predictions.

Run locally with CPU:

```powershell
python src\demand_forecasting\train_xgboost.py
```

Use a supported NVIDIA GPU with:

```powershell
python src\demand_forecasting\train_xgboost.py --device cuda
```

Optionally copy every generated artifact to another folder:

```powershell
python src\demand_forecasting\train_xgboost.py --drive-dir "G:\My Drive\demand-forecasting\artifacts\demand"
```

The standalone Jupyter/Colab workflow is available in `notebooks/demand_forecasting/train_xgboost.ipynb`. It does not import the local project scripts. Upload the horizon dataset to `MyDrive/demand-forecasting/Dataset/horizon_training_dataset.csv`. The notebook performs splitting, preprocessing, training, and evaluation internally, saves artifacts under `/content/xgboost_artifacts`, and copies them to `MyDrive/demand-forecasting/artifacts`.

XGBoost outputs:

```text
artifacts/demand/models/global_xgboost_pipeline.joblib
artifacts/demand/models/global_xgboost_model.json
artifacts/demand/predictions/xgboost_validation_predictions.csv
artifacts/demand/predictions/xgboost_test_predictions.csv
artifacts/demand/reports/xgboost_metrics.csv
artifacts/demand/reports/xgboost_training_metadata.json
```

`global_xgboost_pipeline.joblib` is the main reusable artifact because it contains both the fitted categorical/numeric preprocessor and the fitted XGBoost regressor. The native JSON file contains only the XGBoost model and does not contain the scikit-learn preprocessing steps.

### Fine-tune XGBoost

Run the separate tuner after `prepare_model_data.py` has created `data/processed/demand/model_splits/{train,validation,test}.csv`:

```powershell
python src\demand_forecasting\tune_xgboost.py --trials 12 --folds 3 --validation-months 3
```

It compares the current XGBoost settings with 12 reproducible sampled alternatives. Each alternative is trained and checked on three rolling, chronological three-month windows from the train and validation period. It selects the lowest average fold WAPE, then refits on all pre-test data. The test split is not used for tuning or evaluated by default. Add `--evaluate-test` only for a final held-out comparison after choosing the tuning design; do not repeatedly change settings based on test results.

Outputs use separate names and do not overwrite the current model: `artifacts/demand/models/global_xgboost_tuned_pipeline.joblib`, `global_xgboost_tuned_model.json`, `artifacts/demand/reports/xgboost_tuning_leaderboard.csv`, `xgboost_tuning_folds.csv`, and `xgboost_tuning_metadata.json`. With `--evaluate-test`, the script also writes `artifacts/demand/predictions/xgboost_tuned_test_predictions.csv` and test metrics in the metadata. For Colab, upload or mount the repository with these three split CSVs and run the same command from the repository root after installing `requirements.txt`.

### Train global Extra Trees

The reusable training module is `src/demand_forecasting/train_etr.py`. It follows the same chronological contract as XGBoost:

1. fit on training data and create honest validation predictions;
2. refit a new final pipeline on training plus validation data and create untouched test predictions.

Run locally with:

```powershell
python src\demand_forecasting\train_etr.py
```

Optionally copy every generated artifact to another folder:

```powershell
python src\demand_forecasting\train_etr.py --drive-dir "G:\My Drive\demand-forecasting\artifacts\demand"
```

The standalone Jupyter/Colab workflow is available in `notebooks/demand_forecasting/train_etr.ipynb`. Upload only `horizon_training_dataset.csv` to `MyDrive/demand-forecasting/Dataset/`. The notebook recreates the same chronological split and preprocessing policy as the XGBoost notebook, trains and evaluates Extra Trees, saves artifacts under `/content/etr_artifacts`, and copies them to `MyDrive/demand-forecasting/artifacts`. Extra Trees uses CPU parallelism through `n_jobs=-1`; selecting a GPU runtime does not accelerate this scikit-learn estimator.

Extra Trees outputs:

```text
artifacts/demand/models/global_etr_pipeline.joblib
artifacts/demand/predictions/etr_validation_predictions.csv
artifacts/demand/predictions/etr_test_predictions.csv
artifacts/demand/reports/etr_metrics.csv
artifacts/demand/reports/etr_training_metadata.json
```

`global_etr_pipeline.joblib` contains the fitted categorical/numeric preprocessing and the fitted `ExtraTreesRegressor` together. Its validation and test prediction tables use the same identifying columns as the XGBoost prediction tables, which allows them to be joined safely for the later Ridge ensemble.

### Compare XGBoost and Extra Trees with a Ridge stacker

Run `notebooks/demand_forecasting/train_stacking_ensemble.ipynb` in Colab after both base-model notebooks. It reads the four validation/test prediction CSVs from `MyDrive/demand-forecasting/artifacts/demand/predictions`, matches rows by forecast origin, target month, crop, location, and horizon, trains Ridge on validation predictions, and compares XGBoost, Extra Trees, and stacking on test predictions. It also plots actual versus predicted values, absolute errors, and MAE by horizon. No training dataset upload is needed for this experiment.

The notebook saves an experimental Ridge model, comparison predictions, metrics, and metadata to `/content/stacking_artifacts` and copies them to the matching `artifacts/demand/models`, `artifacts/demand/predictions`, and `artifacts/demand/reports` folders on Drive. Its validation stacking score is in-sample. The base models behind the validation predictions were trained on training data only, while the test predictions came from base models refitted on training plus validation. Therefore, treat this as a comparison experiment, not a deployable end-to-end ensemble; a production stack needs time-ordered out-of-fold base predictions and a later holdout.

### Predict future demand with XGBoost

Use `notebooks/demand_forecasting/predict_xgboost.ipynb` after training. Run Jupyter from the project root or the `notebooks/demand_forecasting` folder. It defaults to `MODEL_VARIANT = "tuned"` and loads `artifacts/demand/models/global_xgboost_tuned_pipeline.joblib`. Change that variable to `"original"` to use the retained `global_xgboost_pipeline.joblib`. The notebook reads `data/processed/demand/future_horizon_features.csv` locally, validates all expected features and future context, predicts every crop/location/H1-H12 row, and saves variant-specific full and date-filtered forecasts under `artifacts/demand/predictions`.

To compare errors on the same held-out test rows, run `python src/demand_forecasting/analyze_xgboost_errors.py` with your Python 3 environment. It writes `artifacts/demand/reports/xgboost_error_breakdown.csv` with overall, crop, location, and horizon metrics. Positive improvement values mean the tuned model did better; negative values identify groups where it did worse.

The prediction notebook intentionally stops when target holiday or weather-pattern values are missing. Regenerate `future_horizon_features.csv` through the preprocessing pipeline whenever the source client data changes; missing future context must not be silently interpreted as zero or replaced by a training median.

### Predict future demand with Extra Trees

Use `notebooks/demand_forecasting/predict_etr.ipynb` after Extra Trees training. It follows the same local prediction workflow and validation rules as the XGBoost notebook, loads `artifacts/demand/models/global_etr_pipeline.joblib`, reads `data/processed/demand/future_horizon_features.csv`, and saves full plus date-filtered forecasts under `artifacts/demand/predictions`.

```text
artifacts/demand/predictions/etr_future_forecast_h1_h12.csv
artifacts/demand/predictions/etr_forecast_2026_10_to_2026_11.csv
```

## Price forecasting — separate workflow

Price forecasting shares the original procurement, calendar, weather, crop, and location sources with demand forecasting. It does not reuse the final demand feature table, target, splits, models, or artifact filenames.

```text
Shared raw and contextual data
        ↓
Create origin-based price and demand lag features
        ↓
Create price H1–H12 rows
        ↓
Chronological price train/validation/test split
        ↓
Price XGBoost + Price Extra Trees
        ↓
Optional time-ordered stacking
        ↓
Compare and select the price model
        ↓
Predict future price in RM/kg
```

### Price folders

```text
data/processed/price/          # Price features, horizons, future rows, splits
data/external/market_prices/  # Optional FAMA or wholesale market prices
src/price_forecasting/        # Price-specific source code
artifacts/price/models/       # Fitted price pipelines
artifacts/price/predictions/  # Validation, test, and future price predictions
artifacts/price/reports/      # Metrics, tuning results, and metadata
```

### Price stage 1 — create historical features

The platform-only price feature stage is implemented and does not require FAMA data:

```powershell
python src\price_forecasting\create_price_features.py
```

It reads `data/processed/shared/monthly_demand_contextual.csv` and creates leakage-safe price, demand, order, and stock history. The first 12 months of each crop/location series are removed because a valid `price_lag_12` is unavailable. Current output:

```text
data/processed/price/price_features.csv
data/processed/price/price_features.xlsx
data/processed/price/price_features.parquet  # when pyarrow is installed
```

With the current five-year dataset, the stage produces 2,160 rows covering January 2022 through December 2025 for all 45 crop/location series.

Created features include `price_lag_1`, `price_lag_3`, `price_lag_6`, `price_lag_12`, three- and six-month rolling price means, one-month absolute and percentage price changes, lagged demand, rolling demand, lagged order count, lagged stock, and lagged stock-to-demand ratio.

### Price stage 2 — create H1–H12 datasets

The price multi-horizon stage is implemented:

```powershell
python src\price_forecasting\create_price_horizon_dataset.py
```

It expands every eligible forecast-origin row into horizons 1 through 12. The training target is `future_weighted_avg_procurement_price_rm_per_kg`. Price, demand, order, and stock features remain anchored to the forecast-origin month. Only known target-month calendar fields and historical target-month weather normals are added for the future month.

The script creates:

```text
data/processed/price/price_horizon_training_dataset.csv
data/processed/price/price_horizon_training_dataset.xlsx
data/processed/price/price_horizon_training_dataset.parquet  # when pyarrow is installed
data/processed/price/future_price_horizon_features.csv
```

With the current data, the training table contains 22,410 labelled rows. The future template contains 540 unlabelled rows: 45 crop/location series multiplied by 12 horizons, using December 2025 as the forecast origin and January through December 2026 as target months.

The future template deliberately leaves the future price target empty. It must be passed to a fitted model only after the later model-data preparation and training stages are complete.

### Price stage 3 — prepare chronological model splits

The price model-data preparation stage is implemented:

```powershell
python src\price_forecasting\prepare_model_data.py
```

It excludes the two date identifiers and `future_weighted_avg_procurement_price_rm_per_kg` from the model inputs, defines `crop`, `location`, `origin_season`, and `target_season` as categorical features, and defines the remaining inputs as numeric features. The reusable preprocessor applies most-frequent imputation and one-hot encoding to categorical values, and median imputation to numeric values. Each fitted model pipeline must fit this preprocessor on its training rows only.

The split is based on `target_month`, not random rows:

```text
Train:      19,170 rows, February 2022 through June 2025
Validation:  1,620 rows, July 2025 through September 2025
Test:        1,620 rows, October 2025 through December 2025
```

All three splits contain H1–H12 and have no overlapping target months. Outputs are saved to:

```text
data/processed/price/model_splits/train.csv
data/processed/price/model_splits/validation.csv
data/processed/price/model_splits/test.csv
artifacts/price/reports/model_data_split.json
```

### Price stage 4 — train XGBoost in Colab

Open and run `notebooks/price_forecasting/train_xgboost.ipynb` in Google Colab. Upload this one training file to Google Drive first:

```text
MyDrive/demand-forecasting/Dataset/price_horizon_training_dataset.csv
```

The notebook is standalone and recreates the same target-month chronological split internally, so the three local split CSVs do not need to be uploaded. It performs categorical encoding and numeric imputation inside the model pipeline, trains a validation model on training rows, refits a final model on training plus validation rows, and evaluates the untouched test period.

It reports MAE, RMSE, WAPE, and R² overall and for every H1–H12 horizon. It also displays actual-versus-predicted prices, the absolute-error distribution, and MAE by horizon.

Artifacts are saved temporarily under `/content/price_xgboost_artifacts` and copied to:

```text
MyDrive/demand-forecasting/artifacts/price/models/global_xgboost_pipeline.joblib
MyDrive/demand-forecasting/artifacts/price/models/global_xgboost_model.json
MyDrive/demand-forecasting/artifacts/price/predictions/xgboost_validation_predictions.csv
MyDrive/demand-forecasting/artifacts/price/predictions/xgboost_test_predictions.csv
MyDrive/demand-forecasting/artifacts/price/reports/xgboost_metrics.csv
MyDrive/demand-forecasting/artifacts/price/reports/xgboost_training_metadata.json
```

The `.joblib` pipeline is the artifact to use for prediction because it contains both preprocessing and XGBoost. The `.json` file contains only the native XGBoost regressor.

### Price stage 5 — train Extra Trees in Colab

Open and run `notebooks/price_forecasting/train_etr.ipynb` in Google Colab. It reuses the same uploaded file as XGBoost:

```text
MyDrive/demand-forecasting/Dataset/price_horizon_training_dataset.csv
```

The notebook uses the same target-month split, feature columns, categorical preprocessing, metrics, prediction keys, and graphs as price XGBoost. Extra Trees trains on CPU with `n_jobs=-1`; selecting a GPU runtime does not accelerate it.

It copies these artifacts to Google Drive:

```text
MyDrive/demand-forecasting/artifacts/price/models/global_etr_pipeline.joblib
MyDrive/demand-forecasting/artifacts/price/predictions/etr_validation_predictions.csv
MyDrive/demand-forecasting/artifacts/price/predictions/etr_test_predictions.csv
MyDrive/demand-forecasting/artifacts/price/reports/etr_metrics.csv
MyDrive/demand-forecasting/artifacts/price/reports/etr_training_metadata.json
```

Extra Trees produces one `.joblib` model artifact rather than a second native JSON file. The joblib file contains the fitted preprocessor and `ExtraTreesRegressor` together.

### Price stage 6 — compare stacking in Colab

After both base-model notebooks finish, open and run `notebooks/price_forecasting/train_stacking_ensemble.ipynb`. It reads these four files directly from Google Drive:

```text
MyDrive/demand-forecasting/artifacts/price/predictions/xgboost_validation_predictions.csv
MyDrive/demand-forecasting/artifacts/price/predictions/xgboost_test_predictions.csv
MyDrive/demand-forecasting/artifacts/price/predictions/etr_validation_predictions.csv
MyDrive/demand-forecasting/artifacts/price/predictions/etr_test_predictions.csv
```

No training dataset or future-feature dataset is needed for this experiment. The notebook matches rows by forecast origin, target month, crop, location, and horizon; verifies that the actual prices and split labels agree; fits Ridge to the two validation prediction columns; and compares XGBoost, Extra Trees, and stacking overall and by horizon.

It saves:

```text
MyDrive/demand-forecasting/artifacts/price/models/experimental_ridge_stacker.joblib
MyDrive/demand-forecasting/artifacts/price/predictions/stacking_validation_comparison.csv
MyDrive/demand-forecasting/artifacts/price/predictions/stacking_test_comparison.csv
MyDrive/demand-forecasting/artifacts/price/reports/stacking_metrics.csv
MyDrive/demand-forecasting/artifacts/price/reports/stacking_training_metadata.json
```

This Ridge stack is an exploratory comparison, not a standalone deployable pipeline. Its validation metric is in-sample, and its two base models were trained on different data amounts for validation and test prediction. A production stack would require time-ordered out-of-fold base predictions and both fitted base pipelines during future forecasting.

### Remaining planned price scripts

The following scripts have not been implemented yet and must not be run until their files are created:

```text
src/price_forecasting/tune_xgboost.py
src/price_forecasting/analyze_errors.py
src/price_forecasting/predict.py
```

The intended complete execution order will be:

```powershell
python src\price_forecasting\create_price_features.py
python src\price_forecasting\create_price_horizon_dataset.py
python src\price_forecasting\prepare_model_data.py
# Run notebooks/price_forecasting/train_xgboost.ipynb in Google Colab
python src\price_forecasting\tune_xgboost.py --trials 12 --folds 3 --validation-months 3
# Run notebooks/price_forecasting/train_etr.ipynb in Google Colab
# Run notebooks/price_forecasting/train_stacking_ensemble.ipynb in Google Colab
python src\price_forecasting\analyze_errors.py
python src\price_forecasting\predict.py
```

### Planned price target and inputs

The price target should be one clearly defined monthly value, such as:

```text
future_weighted_avg_procurement_price_rm_per_kg
```

Recommended origin-known inputs include:

- price lags for 1, 3, 6, and 12 months;
- rolling price means for 3 and 6 months;
- one-month price change;
- lagged demand, rolling demand, order count, and stock information;
- crop, location, and forecast horizon;
- target-month calendar and festival indicators;
- historical weather and target-month climatological normals;
- lagged external FAMA or wholesale prices when reliable mappings exist.

Actual target-month internal price, demand, stock, orders, rainfall, and temperature must not be used as input features. The tuned demand-model parameters must also not be assumed optimal for price forecasting; price XGBoost and Extra Trees require their own chronological evaluation and tuning.
