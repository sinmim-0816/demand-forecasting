# Demand forecast group evaluation

- Variant: `full`
- Model: `tuned_xgboost`
- Test rows: 1620
- Overall WAPE: 19.21%
- Target months: 2025-10, 2025-11, 2025-12

## Highest-error eligible groups

- crop_and_season: Longbean | inter_monsoon — WAPE 59.88%, overprediction.
- crop_and_season: Bittergourd | inter_monsoon — WAPE 44.78%, overprediction.
- crop: Bittergourd — WAPE 41.08%, overprediction.
- crop_and_season: Bittergourd | northeast_monsoon — WAPE 39.24%, overprediction.
- crop_and_season: Lettuce | inter_monsoon — WAPE 37.22%, overprediction.
- crop_and_season: Okra | northeast_monsoon — WAPE 31.07%, overprediction.
- crop_and_season: Pumpkin | northeast_monsoon — WAPE 28.36%, underprediction.
- crop: Lettuce — WAPE 27.89%, balanced.
- crop: Longbean — WAPE 26.50%, overprediction.
- crop_and_season: Sweetcorn | inter_monsoon — WAPE 26.45%, underprediction.
- crop_and_season: Kangkung | northeast_monsoon — WAPE 25.47%, balanced.
- crop: Pumpkin — WAPE 25.12%, underprediction.
- location: Miri — WAPE 20.91%, underprediction.
- forecast_horizon: 2 — WAPE 19.58%, balanced.
- forecast_horizon: 4 — WAPE 19.47%, balanced.

## Consistently weak crops

- Bittergourd: WAPE 41.08%; worse than overall in 3/3 target months; overprediction.
- Lettuce: WAPE 27.89%; worse than overall in 2/3 target months; balanced.
- Longbean: WAPE 26.50%; worse than overall in 2/3 target months; overprediction.
- Pumpkin: WAPE 25.12%; worse than overall in 2/3 target months; underprediction.
- Okra: WAPE 24.63%; worse than overall in 2/3 target months; overprediction.
- Eggplant: WAPE 20.30%; worse than overall in 2/3 target months; underprediction.

## Data limitations

- The held-out test set covers 3 target months (2025-10 to 2025-12), not a complete annual cycle. Month and season findings are diagnostic rather than conclusive.
- The represented target seasons are: inter_monsoon, northeast_monsoon. Unrepresented seasons cannot be evaluated from this test set.
- Groups with fewer than 12 test rows are reported but are not flagged as weak because their estimates are less stable.
- Group error differences identify where a model is weaker; they do not establish that crop identity, season, or horizon caused the error.
- This is an offline held-out evaluation. Production bias and drift require stored forecasts to be matched with actual outcomes after target months close.
