"""Rule-based extraction of implicit physical conditions."""

from __future__ import annotations

import re
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
    ("resonance", ["at resonance", "in resonance", "case of resonance", "electrical resonance", "currently in resonance", "resonance occurs", "resonant frequency"]),
    ("toward_each_other", ["toward each other", "towards each other", "move toward", "move towards", "opposite directions"]),
    ("same_direction_chasing", ["same direction", "catch up", "catches up", "chasing"]),
    ("downstream_upstream", ["downstream", "upstream", "current speed"]),
    ("right_angle", ["right angle", "right-angle", "right-angled", "right isosceles", "isosceles right", "perpendicular", "90 degrees", "90°"]),
    ("perpendicular_bisector", ["perpendicular bisector"]),
    ("midpoint", ["midpoint", "mid point"]),
    ("midpoint_between_two_charges", ["midpoint between the two charges"]),
    ("line_connecting", ["line connecting", "line segment connecting", "line joining"]),
    ("collinear", ["straight line", "same straight line", "collinear", "collinear points"]),
    ("equidistant_on_line", ["equidistant from the two charges and located on the line", "equidistant from the two charges", "equidistant from a and b"]),
    ("opposite_sides_target_charge", ["opposite sides of q", "opposite sides of the charge", "on opposite sides of q"]),
    ("outside_segment", ["outside the segment", "outside the line segment", "to the left of", "to the right of"]),
    ("equilateral_triangle", ["equilateral triangle"]),
    ("triangle_center", ["center of the triangle", "centre of the triangle", "center of triangle", "centre of triangle"]),
    ("square_center", ["square", "center of the square", "centre of the square", "intersection point of the square"]),
    ("square_center_point", ["center of the square", "centre of the square", "center of square", "centre of square", "center of the square", "intersection point of the diagonals", "intersection of the diagonals", "intersection point of the square's diagonals", "square's diagonals", "center of the square"]),
    ("four_identical_square_charges", ["four identical charges", "four equal charges"]),
    ("square_three_charges_fourth_vertex", ["three vertices of a square", "three consecutive vertices of a square", "fourth vertex of the square", "field at the fourth vertex"]),
    ("parallel_plate_capacitor", ["parallel plate capacitor", "parallel-plate capacitor", "parallel plate", "parallel-plate"]),
    ("parallel_insulating_plates", ["parallel insulating plates", "wide, parallel insulating plates", "surface charge densities"]),
    ("infinite_metal_plate", ["infinitely large, flat metal plate", "infinite metal plate", "flat metal plate is uniformly charged"]),
    ("infinite_line_charge", ["infinitely long straight wire", "infinite straight wire", "linear charge density"]),
    ("charged_circular_ring", ["thin circular ring", "circular ring"]),
    ("finite_charged_rod", ["thin, non-conducting rod", "non-conducting rod", "rod of length"]),
    ("electric_equilibrium", ["is in equilibrium", "in equilibrium"]),
    ("field_towards_charge", ["points towards the charge", "points toward the charge", "directed towards the charge", "directed toward the charge"]),
    ("circular_plate", ["circular plate", "circular plates", "circular capacitor plates", "flat circular capacitor plates", "disk", "circle"]),
    ("plate_distance_doubled", ["distance between them doubles", "distance between them doubled", "distance between its plates is doubled", "distance between the plates is doubled", "plate separation is doubled", "plate separation is then doubled", "plate separation was doubled", "plates are moved apart so that the distance between them doubles"]),
    ("capacitor_short_circuit", ["short-circuited", "short circuited", "short-circuiting", "short circuiting", "plates are short-circuited", "plates are short circuited"]),
    ("capacitor_charge_sharing", ["charge is equally shared", "distributed equally among", "connected with another uncharged", "connected in series with another", "connected to another uncharged"]),
    ("uncharged_capacitor", ["uncharged capacitor", "uncharged capacitors"]),
    ("battery_connected", ["battery remains connected", "while connected to the battery", "remains connected to the battery", "remains connected to a battery", "remains connected to the voltage source", "connected to the voltage source", "still connected to the source", "connected to the source"]),
    ("battery_disconnected", ["battery is disconnected", "after disconnecting the battery", "disconnected from the battery", "isolated capacitor", "then disconnected", "and disconnected", "disconnected"]),
    ("frequency_doubled", ["frequency doubles", "frequency is doubled", "frequency f is doubled", "frequency doubled", "frequency increases to double", "frequency increases by a factor of 2", "frequency is increased by 2 times", "frequency is increased by a factor of 2"]),
    ("frequency_tripled", ["frequency is tripled", "frequency tripled", "frequency triples", "frequency is increased by 3 times", "frequency is increased by a factor of 3"]),
    ("frequency_quadrupled", ["frequency is increased by 4 times", "frequency increases by 4 times", "frequency is increased by a factor of 4", "frequency quadrupled", "frequency is quadrupled", "frequency quadruples"]),
    ("frequency_sextupled", ["frequency is increased by 6 times", "frequency increases by 6 times", "frequency is increased by a factor of 6"]),
    ("current_halved", ["current is halved", "current becomes half", "current decreases to 1/2", "current decreases to one half", "current is reduced to half", "decreases to 1/2", "decreases to one half", "reduced to half"]),
    ("ab_quadrature_circuit", ["circuit ab", "segment am", "section am", "uam", "u_am", "u am"]),
    ("target_segment_mb", ["across segment mb", "across section mb", "voltage across mb", "voltage across segment mb", "effective voltage across segment mb", "rms voltage across segment mb"]),
    ("target_mb_power", ["power consumed by mb", "power consumed by the mb segment", "power consumed by segment mb", "power consumed by the mb", "power consumed by mb segment"]),
    ("target_power_abs_error", ["absolute error of the power"]),
    ("target_total_resistance_abs_error", ["absolute error of the total resistance"]),
    ("lc_capacitor_max_charge", ["capacitor is maximally charged", "capacitor is fully charged", "charge on the capacitor is maximum", "capacitor has maximum charge"]),
    ("unknown_parallel_capacitor_charge_branch", ["one of the two capacitors has a charge", "one of the capacitors has a charge"]),
    ("lc_current_maximum", ["current in an lc circuit is at its maximum", "current reaches maximum", "current is maximum", "maximum current"]),
    ("half_least_count_uncertainty", ["instrument has a least count"]),
    ("maximum_possible_value", ["maximum possible"]),
    ("field_angle_given", ["electric fields they produce", "electric fields produced by", "fields they produce", "fields produced by these charges"]),
    ("zero_electric_field_point", ["net electric field is zero", "net electric field at point m is zero", "electric field vector is zero", "resultant electric field strength", "net electric field due to", "field is zero"]),
    ("point_charge_field_line", ["same electric field line generated by a positive point charge", "same electric field line generated by a point charge", "same electric field line"]),
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

    if (
        "square" in lowered
        and "positive charges" in lowered
        and "negative charges" in lowered
        and ("a and c" in lowered or "a, c" in lowered)
        and ("b and d" in lowered or "b, d" in lowered)
        and "square_opposite_equal_charges" not in conditions
    ):
        conditions.append("square_opposite_equal_charges")
    if (
        ("q1 = q2" in lowered or "q1=q2" in lowered or "two identical charges" in lowered or "two identical point charges" in lowered or "two equal charges" in lowered or "both equal" in lowered or "both equal to" in lowered)
        and "two_equal_like_charges" not in conditions
    ):
        conditions.append("two_equal_like_charges")
    if (
        re.search(r"r-?c\s+(?:section|combination).*c-?l(?!r)\s+(?:section|combination).*both", lowered)
        or re.search(r"c-?l(?!r)\s+(?:section|combination).*r-?c\s+(?:section|combination).*both", lowered)
    ) and "rlc_equal_rc_cl_section_voltage" not in conditions:
        conditions.append("rlc_equal_rc_cl_section_voltage")
    if (
        ("q1 = -q2" in lowered or "q1=-q2" in lowered or "q1 = −q2" in lowered or "q1=−q2" in lowered)
        and "two_equal_opposite_charges" not in conditions
    ):
        conditions.append("two_equal_opposite_charges")
    if "equidistant_on_line" in conditions and "midpoint" not in conditions:
        conditions.append("midpoint")
    if re.search(r"frequency\s*(?:\([^)]*\)|[a-z])?\s*(?:is\s+)?tripled", lowered) and "frequency_tripled" not in conditions:
        conditions.append("frequency_tripled")
    if re.search(r"frequency\s*(?:\([^)]*\)|[a-z])?\s*(?:is\s+)?increased\s+by\s+3\s+times", lowered) and "frequency_tripled" not in conditions:
        conditions.append("frequency_tripled")
    if (
        "outside_segment" in conditions
        and ("to the right of a" in lowered or "right of a" in lowered)
        and "outside_right_of_first_endpoint" not in conditions
    ):
        conditions.append("outside_right_of_first_endpoint")
    if (
        "inductive reactance" in lowered
        and re.search(r"\bwhat\s+is\b[^?]*(?:at|when\s+f\s*=|when\s+the\s+frequency)[^?]*\d+\s*hz", lowered)
        and "resonant frequency" not in lowered[lowered.rfind("what"): lowered.find("?") if "?" in lowered else len(lowered)]
        and "initial frequency" not in lowered[lowered.rfind("what"): lowered.find("?") if "?" in lowered else len(lowered)]
        and "off_frequency_reactance_target" not in conditions
    ):
        conditions.append("off_frequency_reactance_target")
    if (
        "outside_segment" in conditions
        and ("to the left of a" in lowered or "left of a" in lowered)
        and "outside_left_of_first_endpoint" not in conditions
    ):
        conditions.append("outside_left_of_first_endpoint")
    if ("voltage across capacitor c1" in lowered or "voltage across c1" in lowered or "across capacitor c₁" in lowered or "across c₁" in lowered) and "target_voltage_c1" not in conditions:
        conditions.append("target_voltage_c1")
    if ("voltage across capacitor c2" in lowered or "voltage across c2" in lowered or "across capacitor c₂" in lowered or "across c₂" in lowered) and "target_voltage_c2" not in conditions:
        conditions.append("target_voltage_c2")

    if re.search(r"distance\s+from\s+a\b|distance\s+am\b|\bcalculate\s+am\b", lowered) and "target_distance_from_A" not in conditions:
        conditions.append("target_distance_from_A")
    if re.search(r"distance\s+from\s+b\b|distance\s+(?:from\s+)?(?:point\s+)?m\s+to\s+b\b|distance\s+bm\b|\bcalculate\s+bm\b", lowered) and "target_distance_from_B" not in conditions:
        conditions.append("target_distance_from_B")
    if re.search(r"(?:among|with)\s+3\s+identical\s+capacitors|\bamong\s+three\s+identical\s+capacitors", lowered) and "share_count_3" not in conditions:
        conditions.append("share_count_3")
    if (
        ("capacitor_charge_sharing" in conditions or "uncharged_capacitor" in conditions)
        and "share_count_3" not in conditions
        and "share_count_2" not in conditions
    ):
        conditions.append("share_count_2")
    g_match = re.search(r"(?:take|given)\s+g\s*=\s*(\d+(?:\.\d+)?)", lowered)
    if g_match and "g" not in known_quantities:
        g_value = float(g_match.group(1))
        normalized_value, normalized_unit = normalize_quantity(g_value, "m/s^2")
        additions["g"] = {
            "value": g_value,
            "unit_symbol": "m/s^2",
            "unit_name": "meter per second squared",
            "dimension": "acceleration",
            "source_text": g_match.group(0),
            "normalized_value": normalized_value,
            "normalized_unit_symbol": normalized_unit,
        }

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
