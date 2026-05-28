import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser.main import parse_problem
from parser.rule_extractor import extract_quantities, extract_relations
from parser.target_detector import detect_target
from scripts.analyze_stage0_failures import _clean_display_text


def outputs(plan):
    seen = []
    for step in plan:
        seen.extend((step.get("output_var") or {}).keys())
    return seen


def test_capacitance_parallel_plate_area_template():
    parsed = parse_problem("A parallel plate capacitor has plate area 33.2 cm^2 and separation 2 mm. Find its capacitance.")
    assert parsed["unknown_quantity"] == "C_cap"
    assert parsed["known_quantities"]["A"]["dimension"] == "area"
    assert parsed["known_quantities"]["d"]["dimension"] == "length"
    assert "C_cap" in outputs(parsed["step_plan"])
    assert any("epsilon_0" in str(step.get("output_var")) for step in parsed["step_plan"])


def test_capacitance_circular_plate_radius_template():
    parsed = parse_problem("A circular parallel plate capacitor has radius 60 cm and plate separation 2 mm. Find the capacitance.")
    assert parsed["unknown_quantity"] == "C_cap"
    assert any(q["dimension"] == "length" and name in {"r", "R_radius"} for name, q in parsed["known_quantities"].items())
    assert "A" in outputs(parsed["step_plan"])
    assert "C_cap" in outputs(parsed["step_plan"])


def test_coulomb_scalar_template():
    parsed = parse_problem("Two charges q1 = 6 nC and q2 = 3 nC are separated by 4 cm. Determine the electric force between them.")
    assert parsed["unknown_quantity"] == "F_e"
    assert {"q1", "q2"}.issubset(parsed["known_quantities"])
    assert any("Coulomb" in step["goal"] for step in parsed["step_plan"])


def test_coulomb_equilateral_vector_template():
    parsed = parse_problem("Three charges form an equilateral triangle of side 4 cm. Find the net electric force acting on q3.")
    assert parsed["unknown_quantity"] in {"F_on_q3", "F_net"}
    assert "equilateral_triangle" in parsed["conditions"]
    assert any("Combine pairwise electric forces" in step["goal"] for step in parsed["step_plan"])


def test_quantity_extraction_stage05_patterns():
    area = extract_quantities("The plate area is 33.2 cm².")
    assert area["A"]["dimension"] == "area"
    assert area["A"]["normalized_value"] == pytest.approx(33.2e-4)

    charge = extract_quantities("The charge is 2 × 10^-6 C.")
    assert next(q for q in charge.values() if q["dimension"] == "charge")["normalized_value"] == pytest.approx(2e-6)

    angle = extract_quantities("The two forces make an angle of 60°.")
    assert angle["theta"]["dimension"] == "angle"

    relations = extract_relations("q1 = 4q2.")
    assert any(r["left"] == "q1" and r["right"] == "q2" and r["factor"] == 4.0 for r in relations)

    time = extract_quantities("After another 20 minutes, find the distance.")
    assert time["t"]["dimension"] == "time"
    assert not any(q["dimension"] == "frequency" for q in time.values())


def test_target_detection_stage05_patterns():
    assert detect_target("How many times greater is the first charge than the second?") == ("ratio", None)
    assert detect_target("Find the direction of the resultant force.") == ("theta", "rad")
    assert detect_target("Find q.") == ("q", "C")


def test_summary_target_counts_array_loads(tmp_path):
    summary = {"target_counts": [{"target": "v", "count": 10}, {"target": "V", "count": 20}]}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["target_counts"][0]["target"] == "v"


def test_analysis_display_encoding_cleanup():
    text = _clean_display_text("60Â° and 2 Î¼C and 3 Ã— 10^-6")
    assert "60°" in text
    assert "2 μC" in text
    assert "3 × 10^-6" in text


def test_next_round_capacitor_energy_inverse_templates():
    energy = parse_problem("A capacitor has a charge of 40 μC and a voltage of 8 V. Calculate the energy stored in the capacitor.")
    assert energy["unknown_quantity"] == "U_cap"
    assert energy["metadata"]["verifier_status"] == "PASS"
    assert "capacitor_energy_charge_voltage" in energy["metadata"]["used_template_names"]

    capacitance = parse_problem("A capacitor has a charge of 60 μC and a voltage of 12 V. Calculate its capacitance.")
    assert capacitance["unknown_quantity"] == "C_cap"
    assert capacitance["metadata"]["verifier_status"] == "PASS"
    assert "capacitance_definition" in capacitance["metadata"]["used_template_names"]


def test_next_round_inductor_energy_inverse_templates():
    inductance = parse_problem("An inductor has a magnetic field energy of 0.45 mJ, and the current through it is 0.3 A. Calculate the inductance.")
    assert inductance["unknown_quantity"] == "L_ind"
    assert inductance["metadata"]["verifier_status"] == "PASS"
    assert "inductance_from_energy_current" in inductance["metadata"]["used_template_names"]

    energy = parse_problem("An inductor has an inductance L = 0.1 H, and a current of 4 A flows through it. Calculate the stored magnetic energy.")
    assert energy["unknown_quantity"] == "U_B"
    assert energy["metadata"]["verifier_status"] == "PASS"


def test_next_round_force_and_field_symbolic_templates():
    resultant = parse_problem("Two electric forces, each with a magnitude of 5 N, act at an angle of 60° to each other. What is the resultant force?")
    assert resultant["unknown_quantity"] == "F_net"
    assert resultant["metadata"]["verifier_status"] == "PASS"
    assert "force_resultant_equal_angle" in resultant["metadata"]["used_template_names"]

    field = parse_problem("Three identical charges Q are fixed at the three vertices of an equilateral triangle with side a. What is the magnitude of the electric field intensity at the center of the triangle?")
    assert field["unknown_quantity"] == "E"
    assert field["metadata"]["verifier_status"] == "PASS"
    assert "electric_field_equilateral_center" in field["metadata"]["used_template_names"]
