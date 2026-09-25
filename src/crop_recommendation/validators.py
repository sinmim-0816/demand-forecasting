"""Validation rules and land-availability calculations."""

from calendar import monthrange
from datetime import date, datetime
from numbers import Real


REQUIRED_FARMER_FIELDS = {
    "farmer_id",
    "farm_id",
    "location",
    "total_area_hectares",
    "requested_area_hectares",
    "planting_month",
    "planning_horizon_months",
}

REQUIRED_PLANTING_FIELDS = {
    "farm_id",
    "crop",
    "allocated_area_hectares",
    "planting_date",
    "expected_harvest_date",
    "status",
}

ALLOWED_PLANTING_STATUSES = {
    "active",
    "planned",
    "harvested",
    "cancelled",
}

LAND_BLOCKING_STATUSES = {"active", "planned"}


class InputValidationError(ValueError):
    """Raised when recommendation input is missing or invalid."""


def _require_object(value, label):
    if not isinstance(value, dict):
        raise InputValidationError(f"{label} must be a JSON object.")


def _require_fields(value, required_fields, label):
    missing = sorted(required_fields - set(value))
    if missing:
        raise InputValidationError(
            f"{label} is missing required fields: {', '.join(missing)}."
        )


def _clean_text(value, field, label):
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{label}.{field} must be a non-empty string.")
    return value.strip()


def _positive_number(value, field, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(f"{label}.{field} must be a number.")
    cleaned = float(value)
    if cleaned <= 0:
        raise InputValidationError(f"{label}.{field} must be greater than zero.")
    return cleaned


def _parse_date(value, field, label):
    text = _clean_text(value, field, label)
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise InputValidationError(
            f"{label}.{field} must use YYYY-MM-DD format."
        ) from error


def _parse_month(value, field, label):
    text = _clean_text(value, field, label)
    try:
        return datetime.strptime(text, "%Y-%m").date().replace(day=1)
    except ValueError as error:
        raise InputValidationError(
            f"{label}.{field} must use YYYY-MM format."
        ) from error


def validate_farmer_input(farmer_input, supported_locations=None):
    """Validate and normalize one farmer recommendation request."""
    label = "farmer_input"
    _require_object(farmer_input, label)
    _require_fields(farmer_input, REQUIRED_FARMER_FIELDS, label)

    cleaned = dict(farmer_input)
    cleaned["farmer_id"] = _clean_text(farmer_input["farmer_id"], "farmer_id", label)
    cleaned["farm_id"] = _clean_text(farmer_input["farm_id"], "farm_id", label)
    cleaned["location"] = _clean_text(farmer_input["location"], "location", label)

    if supported_locations is not None:
        allowed = set(supported_locations)
        if cleaned["location"] not in allowed:
            raise InputValidationError(
                f"Unsupported location '{cleaned['location']}'. "
                f"Supported locations: {', '.join(sorted(allowed))}."
            )

    cleaned["total_area_hectares"] = _positive_number(
        farmer_input["total_area_hectares"], "total_area_hectares", label
    )
    cleaned["requested_area_hectares"] = _positive_number(
        farmer_input["requested_area_hectares"], "requested_area_hectares", label
    )
    if cleaned["requested_area_hectares"] > cleaned["total_area_hectares"]:
        raise InputValidationError(
            "farmer_input.requested_area_hectares cannot exceed total_area_hectares."
        )

    planting_month = _parse_month(
        farmer_input["planting_month"], "planting_month", label
    )
    cleaned["planting_month"] = planting_month.isoformat()[:7]

    horizon = farmer_input["planning_horizon_months"]
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise InputValidationError(
            "farmer_input.planning_horizon_months must be a whole number."
        )
    if not 1 <= horizon <= 12:
        raise InputValidationError(
            "farmer_input.planning_horizon_months must be between 1 and 12."
        )
    cleaned["planning_horizon_months"] = horizon

    return cleaned


def validate_active_plantings(active_plantings, supported_crops=None):
    """Validate and normalize planting records."""
    if not isinstance(active_plantings, list):
        raise InputValidationError("active_plantings must be a JSON array.")

    allowed_crops = set(supported_crops) if supported_crops is not None else None
    cleaned_records = []

    for index, planting in enumerate(active_plantings):
        label = f"active_plantings[{index}]"
        _require_object(planting, label)
        _require_fields(planting, REQUIRED_PLANTING_FIELDS, label)

        cleaned = dict(planting)
        cleaned["farm_id"] = _clean_text(planting["farm_id"], "farm_id", label)
        cleaned["crop"] = _clean_text(planting["crop"], "crop", label)
        if allowed_crops is not None and cleaned["crop"] not in allowed_crops:
            raise InputValidationError(
                f"{label}.crop '{cleaned['crop']}' is not supported."
            )

        cleaned["allocated_area_hectares"] = _positive_number(
            planting["allocated_area_hectares"],
            "allocated_area_hectares",
            label,
        )
        planting_date = _parse_date(planting["planting_date"], "planting_date", label)
        harvest_date = _parse_date(
            planting["expected_harvest_date"], "expected_harvest_date", label
        )
        if harvest_date < planting_date:
            raise InputValidationError(
                f"{label}.expected_harvest_date cannot be before planting_date."
            )
        cleaned["planting_date"] = planting_date.isoformat()
        cleaned["expected_harvest_date"] = harvest_date.isoformat()

        status = _clean_text(planting["status"], "status", label).lower()
        if status not in ALLOWED_PLANTING_STATUSES:
            raise InputValidationError(
                f"{label}.status must be one of: "
                f"{', '.join(sorted(ALLOWED_PLANTING_STATUSES))}."
            )
        cleaned["status"] = status
        cleaned_records.append(cleaned)

    return cleaned_records


def calculate_land_availability(farmer_input, active_plantings):
    """Calculate land occupied during the requested planting month."""
    planting_month = _parse_month(
        farmer_input["planting_month"], "planting_month", "farmer_input"
    )
    month_end = planting_month.replace(
        day=monthrange(planting_month.year, planting_month.month)[1]
    )

    blocking_records = []
    allocated_area = 0.0
    for planting in active_plantings:
        if planting["farm_id"] != farmer_input["farm_id"]:
            continue
        if planting["status"] not in LAND_BLOCKING_STATUSES:
            continue

        start = date.fromisoformat(planting["planting_date"])
        end = date.fromisoformat(planting["expected_harvest_date"])
        overlaps_requested_month = start <= month_end and end >= planting_month
        if not overlaps_requested_month:
            continue

        allocated_area += planting["allocated_area_hectares"]
        blocking_records.append(planting)

    total_area = farmer_input["total_area_hectares"]
    if allocated_area > total_area + 1e-9:
        raise InputValidationError(
            f"Allocated area ({allocated_area:.2f} ha) exceeds the farm's total "
            f"area ({total_area:.2f} ha)."
        )

    remaining_area = max(0.0, total_area - allocated_area)
    requested_area = farmer_input["requested_area_hectares"]

    return {
        "total_area_hectares": round(total_area, 4),
        "allocated_area_hectares": round(allocated_area, 4),
        "remaining_area_hectares": round(remaining_area, 4),
        "requested_area_hectares": round(requested_area, 4),
        "has_sufficient_land": requested_area <= remaining_area + 1e-9,
        "blocking_planting_count": len(blocking_records),
    }


def validate_recommendation_inputs(
    farmer_input,
    active_plantings,
    supported_locations=None,
    supported_crops=None,
):
    """Validate a complete request and ensure sufficient land is available."""
    farmer = validate_farmer_input(farmer_input, supported_locations)
    plantings = validate_active_plantings(active_plantings, supported_crops)
    land = calculate_land_availability(farmer, plantings)

    if not land["has_sufficient_land"]:
        raise InputValidationError(
            f"Requested area ({land['requested_area_hectares']:.2f} ha) exceeds "
            f"remaining area ({land['remaining_area_hectares']:.2f} ha) for "
            f"{farmer['planting_month']}."
        )

    return {
        "farmer_input": farmer,
        "active_plantings": plantings,
        "land_availability": land,
    }

