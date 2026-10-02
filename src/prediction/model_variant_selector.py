"""Select full or core forecasting models from explicit readiness evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class VariantDecision:
    variant: str
    context_status: dict[str, str]
    warnings: list[str]


def _missing_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return sorted(column for column in columns if column not in frame.columns)


def _incomplete_columns(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    available = [column for column in columns if column in frame.columns]
    return sorted(column for column in available if frame[column].isna().any())


def full_only_features(core_metadata: dict, full_metadata: dict) -> list[str]:
    core_features = set(core_metadata.get("feature_columns", []))
    full_features = list(full_metadata.get("feature_columns", []))
    return [column for column in full_features if column not in core_features]


def select_demand_variant(
    full_features: pd.DataFrame,
    core_metadata: dict,
    full_metadata: dict,
) -> VariantDecision:
    """Use full demand only when every additional AQ input is reproducible."""

    contextual_features = full_only_features(core_metadata, full_metadata)
    missing = _missing_columns(full_features, contextual_features)
    incomplete = _incomplete_columns(full_features, contextual_features)
    if not contextual_features or missing or incomplete:
        details = sorted(set(missing + incomplete))
        warning = "Air-quality context is unavailable or incomplete; core demand model selected."
        if details:
            warning += " Fields: " + ", ".join(details)
        return VariantDecision(
            variant="core",
            context_status={"air_quality": "insufficient"},
            warnings=[warning],
        )

    return VariantDecision(
        variant="full",
        context_status={"air_quality": "ready"},
        warnings=[],
    )


def select_price_variants(
    full_features: pd.DataFrame,
    core_metadata: dict,
    full_metadata: dict,
) -> pd.Series:
    """Select price full/core per row so unmapped crops always use core."""

    contextual_features = full_only_features(core_metadata, full_metadata)
    missing = _missing_columns(full_features, contextual_features)
    if not contextual_features or missing:
        return pd.Series("core", index=full_features.index, name="model_variant")

    ready = full_features[contextual_features].notna().all(axis=1)
    if "fama_ladang_mapped" in full_features:
        ready &= full_features["fama_ladang_mapped"].eq(1)
    else:
        ready &= False
    if "fama_ladang_lag_1_available" in full_features:
        ready &= full_features["fama_ladang_lag_1_available"].eq(1)
    else:
        ready &= False

    return pd.Series(
        ready.map({True: "full", False: "core"}),
        index=full_features.index,
        name="model_variant",
    )
