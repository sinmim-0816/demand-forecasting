# Price forecast group evaluation

- Variant: `full`
- Model: `xgboost`
- Test rows: 1620
- Overall WAPE: 3.25%
- Target months: 2025-10, 2025-11, 2025-12

## Highest-error eligible groups

- crop_and_season: Sweetcorn | northeast_monsoon — WAPE 5.64%, balanced.
- crop: Sweetcorn — WAPE 5.04%, balanced.
- crop_and_season: Bittergourd | inter_monsoon — WAPE 4.48%, balanced.
- crop_and_season: Choysum | inter_monsoon — WAPE 4.21%, balanced.
- crop_and_season: Tomato | northeast_monsoon — WAPE 3.94%, balanced.
- target_month: 2025-12 — WAPE 3.86%, balanced.
- crop: Bittergourd — WAPE 3.86%, balanced.
- crop_and_season: Longbean | northeast_monsoon — WAPE 3.85%, balanced.
- location: Kuching — WAPE 3.77%, balanced.
- crop_and_season: Sweetcorn | inter_monsoon — WAPE 3.71%, balanced.
- crop_and_season: Redchilli | northeast_monsoon — WAPE 3.69%, balanced.
- crop: Tomato — WAPE 3.56%, balanced.
- crop: Longbean — WAPE 3.56%, balanced.
- crop_and_season: Bittergourd | northeast_monsoon — WAPE 3.55%, balanced.
- target_season: northeast_monsoon — WAPE 3.43%, balanced.

## Consistently weak crops

- Sweetcorn: WAPE 5.04%; worse than overall in 3/3 target months; balanced.
- Bittergourd: WAPE 3.86%; worse than overall in 2/3 target months; balanced.
- Pumpkin: WAPE 3.38%; worse than overall in 2/3 target months; balanced.
- Choysum: WAPE 3.32%; worse than overall in 2/3 target months; balanced.

## Data limitations

- The held-out test set covers 3 target months (2025-10 to 2025-12), not a complete annual cycle. Month and season findings are diagnostic rather than conclusive.
- The represented target seasons are: inter_monsoon, northeast_monsoon. Unrepresented seasons cannot be evaluated from this test set.
- Groups with fewer than 12 test rows are reported but are not flagged as weak because their estimates are less stable.
- Group error differences identify where a model is weaker; they do not establish that crop identity, season, or horizon caused the error.
- This is an offline held-out evaluation. Production bias and drift require stored forecasts to be matched with actual outcomes after target months close.
