# Price forecast group evaluation

- Variant: `core`
- Model: `xgboost`
- Test rows: 1620
- Overall WAPE: 3.25%
- Target months: 2025-10, 2025-11, 2025-12

## Highest-error eligible groups

- crop_and_season: Sweetcorn | northeast_monsoon — WAPE 5.81%, balanced.
- crop: Sweetcorn — WAPE 5.11%, balanced.
- crop_and_season: Bittergourd | inter_monsoon — WAPE 4.41%, balanced.
- crop_and_season: Choysum | inter_monsoon — WAPE 4.38%, balanced.
- crop_and_season: Longbean | northeast_monsoon — WAPE 4.03%, balanced.
- crop_and_season: Tomato | northeast_monsoon — WAPE 3.98%, balanced.
- target_month: 2025-12 — WAPE 3.82%, balanced.
- location: Kuching — WAPE 3.72%, balanced.
- crop: Tomato — WAPE 3.67%, balanced.
- crop: Bittergourd — WAPE 3.67%, balanced.
- crop: Longbean — WAPE 3.65%, balanced.
- crop_and_season: Redchilli | northeast_monsoon — WAPE 3.61%, balanced.
- crop_and_season: Sweetcorn | inter_monsoon — WAPE 3.55%, balanced.
- crop_and_season: Pumpkin | inter_monsoon — WAPE 3.53%, balanced.
- target_season: northeast_monsoon — WAPE 3.40%, balanced.

## Consistently weak crops

- Sweetcorn: WAPE 5.11%; worse than overall in 3/3 target months; balanced.
- Bittergourd: WAPE 3.67%; worse than overall in 2/3 target months; balanced.
- Choysum: WAPE 3.40%; worse than overall in 2/3 target months; balanced.

## Data limitations

- The held-out test set covers 3 target months (2025-10 to 2025-12), not a complete annual cycle. Month and season findings are diagnostic rather than conclusive.
- The represented target seasons are: inter_monsoon, northeast_monsoon. Unrepresented seasons cannot be evaluated from this test set.
- Groups with fewer than 12 test rows are reported but are not flagged as weak because their estimates are less stable.
- Group error differences identify where a model is weaker; they do not establish that crop identity, season, or horizon caused the error.
- This is an offline held-out evaluation. Production bias and drift require stored forecasts to be matched with actual outcomes after target months close.
