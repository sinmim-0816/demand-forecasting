# Price Forecasting

The price workflow predicts future weighted average procurement price in RM/kg. This represents the marketplace/farm procurement price, not retail consumer price.

Commands are documented in [RUN_GUIDE.md](RUN_GUIDE.md). Folder placement is documented in [ARCHITECTURE.md](ARCHITECTURE.md).

## Pipeline

```text
Shared monthly platform data + original FAMA workbook
    -> normalize FAMA Ladang prices
    -> platform and FAMA historical features
    -> price H1-H12 transformation
    -> chronological split
    -> XGBoost / Extra Trees / stacking comparison
    -> selected price model
```

## FAMA preprocessing

`src/price_forecasting/prepare_fama_price_data.py` reads the legacy FAMA `.xls` workbook and extracts only `PERINGKAT: LADANG` from every sheet. It preserves all commodities in a normalized table for future expansion.

Outputs:

```text
data/external/market_prices/fama_ladang_prices.csv
data/external/market_prices/fama_ladang_prices.parquet
data/external/market_prices/fama_ladang_coverage.csv
data/external/market_prices/fama_crop_mapping.csv
```

The mapping file controls which FAMA series may enter the model. Only confirmed kilogram-based mappings are enabled. Ambiguous names, insufficient history, unavailable Ladang series, and incompatible units are explicitly disabled instead of guessed.

Current confirmed mappings are Bittergourd, Cucumber, Eggplant, Kangkung, Longbean, Okra, Pumpkin, and Spinach.

## Historical price features

`src/price_forecasting/create_price_features.py` creates:

- Platform price lags 1, 3, 6, and 12
- Rolling price means for 3 and 6 months
- One-month absolute and percentage price change
- Demand lags and rolling demand
- Lagged order count, stock, and stock-to-demand ratio
- FAMA Ladang price lags 1, 3, 6, and 12
- FAMA rolling means and price changes
- FAMA mapping and availability indicators

FAMA is contextual data. The current-month FAMA value is not exposed to the model; the minimum FAMA lag is one month.

`supermarket_order_count` and `trader_order_count` remain available in the shared monthly table for reporting, but are deliberately removed from the price model contract because CropWise cannot observe a reliable order type.

The first 12 months of each crop/location series are removed because `price_lag_12` does not yet exist.

## H1-H12 transformation

`src/price_forecasting/create_price_horizon_dataset.py` expands every eligible forecast origin into H1-H12 rows.

Target:

```text
future_weighted_avg_procurement_price_rm_per_kg
```

Origin price, demand, stock, FAMA, and contextual history remain anchored to the forecast origin. Target calendar and holiday fields and historical target-month weather normals may be added because they do not reveal the actual future price.

## Chronological evaluation

`src/price_forecasting/prepare_model_data.py` splits by target month into train, validation, and test periods. All three algorithms must use the same split files for a fair comparison.

Use MAE, RMSE, WAPE, and R². Select the final model using the untouched test period rather than training performance.

## Models

Training notebooks:

```text
notebooks/price_forecasting/train_xgboost.ipynb
notebooks/price_forecasting/train_etr.ipynb
notebooks/price_forecasting/train_stacking_ensemble.ipynb
```

The stacking model is not automatically better. It should be selected only when it improves test performance over both standalone models.

## Future prediction

The preprocessing pipeline creates:

```text
data/processed/price/future_price_horizon_features.csv
```

Run `notebooks/price_forecasting/predict_xgboost.ipynb` locally with the exact scikit-learn and XGBoost versions used for training. The resulting future price forecast is also an input to crop recommendation.

## Important outputs

```text
data/processed/price/price_features.csv
data/processed/price/price_horizon_training_dataset.csv
data/processed/price/future_price_horizon_features.csv
data/processed/price/model_splits/
artifacts/price/models/
artifacts/price/predictions/
artifacts/price/reports/
```

Whenever FAMA mappings, feature engineering, or historical data change, rerun preprocessing and retrain all price models before comparing them again.
