import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "prediction"))

from model_variant_selector import select_demand_variant, select_price_variants


class ModelVariantSelectorTests(unittest.TestCase):
    def test_demand_falls_back_when_air_quality_is_missing(self):
        core = {"feature_columns": ["demand_lag_1"]}
        full = {"feature_columns": ["demand_lag_1", "origin_avg_api"]}
        features = pd.DataFrame({"demand_lag_1": [100.0], "origin_avg_api": [None]})

        decision = select_demand_variant(features, core, full)

        self.assertEqual(decision.variant, "core")
        self.assertEqual(decision.context_status["air_quality"], "insufficient")

    def test_demand_uses_full_when_air_quality_is_complete(self):
        core = {"feature_columns": ["demand_lag_1"]}
        full = {"feature_columns": ["demand_lag_1", "origin_avg_api"]}
        features = pd.DataFrame({"demand_lag_1": [100.0], "origin_avg_api": [55.0]})

        self.assertEqual(select_demand_variant(features, core, full).variant, "full")

    def test_price_selection_is_per_row(self):
        core = {"feature_columns": ["price_lag_1"]}
        full = {
            "feature_columns": [
                "price_lag_1",
                "origin_avg_api",
                "fama_ladang_price_lag_1",
                "fama_ladang_mapped",
                "fama_ladang_lag_1_available",
            ]
        }
        features = pd.DataFrame(
            {
                "price_lag_1": [4.0, 5.0],
                "origin_avg_api": [50.0, 50.0],
                "fama_ladang_price_lag_1": [3.5, None],
                "fama_ladang_mapped": [1, 0],
                "fama_ladang_lag_1_available": [1, 0],
            }
        )

        variants = select_price_variants(features, core, full)

        self.assertEqual(variants.tolist(), ["full", "core"])


if __name__ == "__main__":
    unittest.main()
