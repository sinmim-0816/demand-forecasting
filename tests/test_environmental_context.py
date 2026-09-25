import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "preprocessing"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "demand_forecasting"))

from create_horizon_dataset import add_leakage_safe_weather_normals
from prepare_air_quality_data import create_monthly_air_quality
from prepare_weather_data import create_monthly_weather


class EnvironmentalContextTests(unittest.TestCase):
    def test_monthly_api_threshold_counts(self):
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "location": ["Kuching", "Kuching"],
                "api": [50, 101],
            }
        )

        monthly = create_monthly_air_quality(daily, Path("official_api.xlsx"))

        self.assertEqual(len(monthly), 1)
        self.assertEqual(monthly.loc[0, "Avg API"], 75.5)
        self.assertEqual(monthly.loc[0, "Max API"], 101)
        self.assertEqual(monthly.loc[0, "Moderate or Worse Days"], 1)
        self.assertEqual(monthly.loc[0, "Unhealthy Days"], 1)
        self.assertEqual(monthly.loc[0, "API Observation Days"], 2)

    def test_monthly_weather_includes_average_daily_solar_radiation(self):
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "location": ["Kuching", "Kuching"],
                "daily_mean_temperature_c": [27.0, 29.0],
                "daily_min_temperature_c": [23.0, 24.0],
                "daily_max_temperature_c": [31.0, 32.0],
                "daily_precipitation_mm": [5.0, 7.0],
                "daily_shortwave_radiation_mj_m2": [18.0, 22.0],
            }
        )

        monthly = create_monthly_weather(daily)

        self.assertEqual(
            monthly.loc[0, "Avg Daily Shortwave Radiation (MJ/m²)"],
            20.0,
        )
        self.assertEqual(monthly.loc[0, "Total Precipitation (mm)"], 12.0)

    def test_target_solar_normal_excludes_weather_after_origin(self):
        expanded = pd.DataFrame(
            {
                "_horizon_row_id": [0],
                "forecast_origin_month": pd.to_datetime(["2022-01-01"]),
                "location": ["Kuching"],
                "target_month_number": [2],
            }
        )
        contextual = pd.DataFrame(
            {
                "month": pd.to_datetime(["2021-02-01", "2022-02-01"]),
                "location": ["Kuching", "Kuching"],
                "avg_temperature_c": [27.0, 29.0],
                "total_precipitation_mm": [100.0, 200.0],
                "avg_daily_shortwave_radiation_mj_m2": [18.0, 24.0],
            }
        )

        result = add_leakage_safe_weather_normals(expanded, contextual)

        self.assertEqual(result.loc[0, "target_month_normal_temperature"], 27.0)
        self.assertEqual(
            result.loc[0, "target_month_normal_precipitation"], 100.0
        )
        self.assertEqual(
            result.loc[0, "target_month_normal_solar_radiation"], 18.0
        )
        self.assertEqual(result.loc[0, "target_month_weather_history_years"], 1)


if __name__ == "__main__":
    unittest.main()
