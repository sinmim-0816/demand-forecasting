import unittest

import pandas as pd

from src.supply_forecasting.create_supply_features import (
    create_features,
    keep_model_ready_rows,
)
from src.supply_forecasting.create_supply_horizon_dataset import (
    add_target_production,
    expand_horizons,
)
from src.supply_forecasting.prepare_supply_data import (
    prepare_monthly_supply,
    validate_monthly_series,
)


class SupplyPreprocessingTests(unittest.TestCase):
    def test_raw_records_aggregate_to_monthly_production(self):
        raw = pd.DataFrame(
            {
                "Date": ["2024-01-05", "2024-01-20", "2024-02-01"],
                "Commodity": ["Tomato", "Tomato", "Tomato"],
                "District": ["kuching", "Kuching", "Kuching"],
                "Quantity": [100, 50, 120],
            }
        )
        settings = {
            "date_column": "Date",
            "crop_column": "Commodity",
            "location_column": "District",
            "production_column": "Quantity",
        }
        monthly = prepare_monthly_supply(raw, settings)
        validate_monthly_series(monthly)
        self.assertEqual(monthly["production_kg"].tolist(), [150, 120])
        self.assertEqual(monthly["source_record_count"].tolist(), [2, 1])

    def test_missing_month_is_rejected_instead_of_filled_with_zero(self):
        monthly = pd.DataFrame(
            {
                "month": pd.to_datetime(["2024-01-01", "2024-03-01"]),
                "crop": ["Tomato", "Tomato"],
                "location": ["Kuching", "Kuching"],
                "production_kg": [100, 120],
            }
        )
        with self.assertRaisesRegex(ValueError, "consecutive monthly records"):
            validate_monthly_series(monthly)

    def test_supply_features_use_only_prior_production_history(self):
        months = pd.date_range("2023-01-01", periods=15, freq="MS")
        data = pd.DataFrame(
            {
                "month": months,
                "crop": "Tomato",
                "location": "Kuching",
                "production_kg": range(100, 115),
                "avg_temperature_c": 27.0,
                "total_precipitation_mm": 200.0,
                "avg_daily_shortwave_radiation_mj_m2": 16.0,
            }
        )
        featured = create_features(data, "core")
        ready, _ = keep_model_ready_rows(featured, "core")
        first = ready.iloc[0]
        self.assertEqual(first["production_kg"], 112)
        self.assertEqual(first["production_lag_1"], 111)
        self.assertEqual(first["production_lag_12"], 100)

    def test_horizon_target_comes_from_target_month(self):
        features = pd.DataFrame(
            {
                "month": pd.to_datetime(["2024-01-01"]),
                "crop": ["Tomato"],
                "location": ["Kuching"],
                "production_kg": [100],
                "year": [2024],
                "month_number": [1],
                "quarter": [1],
                "season": ["northeast_monsoon"],
            }
        )
        supply = pd.DataFrame(
            {
                "month": pd.to_datetime(["2024-02-01", "2024-03-01"]),
                "crop": ["Tomato", "Tomato"],
                "location": ["Kuching", "Kuching"],
                "production_kg": [130, 140],
            }
        )
        expanded = add_target_production(expand_horizons(features), supply)
        self.assertEqual(expanded.loc[expanded["forecast_horizon"].eq(1), "future_production_kg"].iloc[0], 130)
        self.assertEqual(expanded.loc[expanded["forecast_horizon"].eq(2), "future_production_kg"].iloc[0], 140)
        self.assertTrue(expanded.loc[expanded["forecast_horizon"].eq(3), "future_production_kg"].isna().all())


if __name__ == "__main__":
    unittest.main()
