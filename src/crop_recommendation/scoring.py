"""Normalize forecasts and calculate crop market-opportunity scores."""

from math import isfinite
from numbers import Real

try:
    from .config import DEMAND_WEIGHT, PRICE_WEIGHT
except ImportError:
    from config import DEMAND_WEIGHT, PRICE_WEIGHT


DEMAND_FIELD = "predicted_demand_kg"
PRICE_FIELD = "predicted_price_rm_per_kg"

REQUIRED_CANDIDATE_FIELDS = {
    "crop",
    "location",
    "target_month",
    DEMAND_FIELD,
    PRICE_FIELD,
}


class ScoringError(ValueError):
    """Raised when forecast candidates cannot be scored safely."""


def _validate_weights(demand_weight, price_weight):
    for name, value in {
        "demand_weight": demand_weight,
        "price_weight": price_weight,
    }.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ScoringError(f"{name} must be a number.")
        if not isfinite(float(value)) or value < 0:
            raise ScoringError(f"{name} must be a finite non-negative number.")

    if abs(float(demand_weight) + float(price_weight) - 1.0) > 1e-9:
        raise ScoringError("Demand and price weights must add up to 1.0.")


def _validate_metric(value, field, index):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ScoringError(f"candidates[{index}].{field} must be a number.")

    cleaned = float(value)
    if not isfinite(cleaned):
        raise ScoringError(f"candidates[{index}].{field} must be finite.")
    if cleaned < 0:
        raise ScoringError(f"candidates[{index}].{field} cannot be negative.")
    return cleaned


def validate_candidates(candidates):
    """Validate forecast candidates and return normalized record copies."""
    if not isinstance(candidates, list) or not candidates:
        raise ScoringError("candidates must be a non-empty list.")

    cleaned_candidates = []
    seen = set()

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ScoringError(f"candidates[{index}] must be an object.")

        missing = sorted(REQUIRED_CANDIDATE_FIELDS - set(candidate))
        if missing:
            raise ScoringError(
                f"candidates[{index}] is missing fields: {', '.join(missing)}."
            )

        cleaned = dict(candidate)
        for field in ("crop", "location", "target_month"):
            value = candidate[field]
            if not isinstance(value, str) or not value.strip():
                raise ScoringError(
                    f"candidates[{index}].{field} must be a non-empty string."
                )
            cleaned[field] = value.strip()

        cleaned[DEMAND_FIELD] = _validate_metric(
            candidate[DEMAND_FIELD], DEMAND_FIELD, index
        )
        cleaned[PRICE_FIELD] = _validate_metric(
            candidate[PRICE_FIELD], PRICE_FIELD, index
        )

        identity = (
            cleaned["crop"],
            cleaned["location"],
            cleaned["target_month"],
        )
        if identity in seen:
            raise ScoringError(
                "Duplicate crop/location/target_month candidate: "
                f"{identity[0]}, {identity[1]}, {identity[2]}."
            )
        seen.add(identity)
        cleaned_candidates.append(cleaned)

    return cleaned_candidates


def min_max_normalize(values):
    """Scale numeric values to 0–1 using the current candidate range."""
    if not values:
        raise ScoringError("Cannot normalize an empty value list.")

    minimum = min(values)
    maximum = max(values)
    spread = maximum - minimum

    if abs(spread) <= 1e-12:
        return [0.5 for _ in values]

    return [(value - minimum) / spread for value in values]


def score_candidates(
    candidates,
    demand_weight=DEMAND_WEIGHT,
    price_weight=PRICE_WEIGHT,
):
    """Return candidates ranked by the configured 50/50 opportunity score.

    Normalization is relative to all candidates supplied for one farmer
    request. Forecast confidence may be present in candidate records, but it is
    copied to the output and is not included in the opportunity score.
    """
    _validate_weights(demand_weight, price_weight)
    cleaned = validate_candidates(candidates)

    demand_scores = min_max_normalize(
        [candidate[DEMAND_FIELD] for candidate in cleaned]
    )
    price_scores = min_max_normalize(
        [candidate[PRICE_FIELD] for candidate in cleaned]
    )

    scored = []
    for candidate, demand_score, price_score in zip(
        cleaned, demand_scores, price_scores
    ):
        opportunity_score = (
            demand_score * float(demand_weight)
            + price_score * float(price_weight)
        )
        result = dict(candidate)
        result["normalized_demand_score"] = round(demand_score, 6)
        result["normalized_price_score"] = round(price_score, 6)
        result["opportunity_score"] = round(opportunity_score, 6)
        scored.append(result)

    scored.sort(
        key=lambda item: (
            -item["opportunity_score"],
            item["crop"],
            item["target_month"],
        )
    )

    previous_score = None
    current_rank = 0
    for position, candidate in enumerate(scored, start=1):
        if previous_score is None or candidate["opportunity_score"] != previous_score:
            current_rank = position
        candidate["rank"] = current_rank
        previous_score = candidate["opportunity_score"]

    return scored

