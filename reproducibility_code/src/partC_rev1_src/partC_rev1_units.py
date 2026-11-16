"""Explicit flow unit conversions for partC_rev1."""

from __future__ import annotations

from typing import Any


def _partC_rev1_unit_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("unit_conversion", {})


def _partC_rev1_payload(config: dict[str, Any]) -> float:
    return float(_partC_rev1_unit_cfg(config).get("average_payload_tons_per_truck", 18.0))


def _partC_rev1_days(config: dict[str, Any]) -> float:
    return float(_partC_rev1_unit_cfg(config).get("annual_operating_days", 365.0))


def _partC_rev1_ktons_factor(config: dict[str, Any]) -> float:
    return float(_partC_rev1_unit_cfg(config).get("annual_ktons_to_annual_tons_factor", 1000.0))


def partC_rev1_annual_ktons_to_annual_tons(x, config: dict[str, Any]):
    return x * _partC_rev1_ktons_factor(config)


def partC_rev1_annual_ktons_to_daily_truck_trips(x, config: dict[str, Any]):
    return partC_rev1_annual_ktons_to_annual_tons(x, config) / _partC_rev1_payload(config) / _partC_rev1_days(config)


def partC_rev1_daily_truck_trips_to_annual_ktons(x, config: dict[str, Any]):
    return x * _partC_rev1_payload(config) * _partC_rev1_days(config) / _partC_rev1_ktons_factor(config)


def partC_rev1_convert_flow_units(x, from_unit: str, to_unit: str, config: dict[str, Any]):
    from_unit = str(from_unit)
    to_unit = str(to_unit)
    if from_unit == to_unit:
        return x

    unit_cfg = _partC_rev1_unit_cfg(config)
    if not bool(unit_cfg.get("allow_unit_conversion", True)):
        raise ValueError(f"Unit conversion disabled: {from_unit} -> {to_unit}")

    supported = {"annual_ktons", "annual_tons", "daily_truck_trips"}
    if from_unit not in supported or to_unit not in supported:
        if bool(unit_cfg.get("strict_no_unit_mixing", True)):
            raise ValueError(f"Unsupported flow unit conversion: {from_unit} -> {to_unit}")
        return x

    if from_unit == "annual_ktons":
        annual_ktons = x
    elif from_unit == "annual_tons":
        annual_ktons = x / _partC_rev1_ktons_factor(config)
    elif from_unit == "daily_truck_trips":
        annual_ktons = partC_rev1_daily_truck_trips_to_annual_ktons(x, config)
    else:  # pragma: no cover - guarded by supported set.
        raise ValueError(f"Unsupported source flow unit: {from_unit}")

    if to_unit == "annual_ktons":
        return annual_ktons
    if to_unit == "annual_tons":
        return partC_rev1_annual_ktons_to_annual_tons(annual_ktons, config)
    if to_unit == "daily_truck_trips":
        return partC_rev1_annual_ktons_to_daily_truck_trips(annual_ktons, config)
    raise ValueError(f"Unsupported target flow unit: {to_unit}")
