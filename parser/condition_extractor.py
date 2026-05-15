"""Rule-based extraction of implicit physical conditions."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .unit_normalizer import normalize_quantity


PHRASE_RULES = [
    ("initial_rest", ["from rest", "starts from rest", "released from rest", "initially at rest"]),
    ("frictionless", ["frictionless", "smooth surface", "neglect friction", "without friction"]),
    ("no_air_resistance", ["neglect air resistance", "ignore air resistance"]),
    ("constant_speed", ["constant speed", "uniform speed"]),
    ("constant_acceleration", ["constant acceleration", "uniform acceleration", "uniformly accelerated", "accelerates uniformly"]),
    ("ideal_gas", ["ideal gas"]),
    ("thermal_equilibrium", ["thermal equilibrium"]),
    ("massless_string", ["massless string"]),
    ("ideal_pulley", ["light pulley"]),
    ("ideal_ammeter", ["ideal ammeter"]),
    ("ideal_voltmeter", ["ideal voltmeter"]),
    ("ideal_battery", ["ideal battery"]),
    ("series_circuit", ["series circuit", "connected in series"]),
    ("parallel_circuit", ["parallel circuit", "connected in parallel"]),
    ("toward_each_other", ["toward each other", "towards each other", "move toward", "move towards", "opposite directions"]),
    ("same_direction_chasing", ["same direction", "catch up", "catches up", "chasing"]),
    ("downstream_upstream", ["downstream", "upstream", "current speed"]),
    ("right_angle", ["right angle", "right-angled", "perpendicular", "90 degrees", "90°"]),
    ("equilateral_triangle", ["equilateral triangle"]),
    ("square_center", ["square", "center of the square", "centre of the square", "intersection point of the square"]),
    ("parallel_plate_capacitor", ["parallel plate capacitor", "parallel-plate capacitor", "parallel plate", "parallel-plate"]),
    ("circular_plate", ["circular plate", "circular plates", "disk", "circle"]),
    ("battery_connected", ["battery remains connected", "while connected to the battery", "remains connected to the battery", "remains connected to a battery"]),
    ("battery_disconnected", ["battery is disconnected", "after disconnecting the battery", "disconnected from the battery", "isolated capacitor", "then disconnected", "and disconnected", "disconnected"]),
]


def extract_conditions(
    problem_text: str,
    known_quantities: Dict[str, Dict[str, object]],
) -> Tuple[List[str], Dict[str, Dict[str, object]]]:
    """Extract condition labels and add deterministic implied quantities."""
    lowered = problem_text.lower()
    conditions: List[str] = []
    additions: Dict[str, Dict[str, object]] = {}
    for condition, phrases in PHRASE_RULES:
        if any(phrase in lowered for phrase in phrases) and condition not in conditions:
            conditions.append(condition)

    if "initial_rest" in conditions and "v_0" not in known_quantities:
        normalized_value, normalized_unit = normalize_quantity(0.0, "m/s")
        additions["v_0"] = {
            "value": 0.0,
            "unit_symbol": "m/s",
            "unit_name": "meter per second",
            "dimension": "velocity",
            "source_text": "from rest",
            "normalized_value": normalized_value,
            "normalized_unit_symbol": normalized_unit,
        }

    if "constant_speed" in conditions and "a" not in known_quantities:
        normalized_value, normalized_unit = normalize_quantity(0.0, "m/s^2")
        additions["a"] = {
            "value": 0.0,
            "unit_symbol": "m/s^2",
            "unit_name": "meter per second squared",
            "dimension": "acceleration",
            "source_text": "constant speed",
            "normalized_value": normalized_value,
            "normalized_unit_symbol": normalized_unit,
        }

    return conditions, additions
