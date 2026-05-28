import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser.rule_extractor import extract_quantities, extract_relations


def test_extract_mechanics_quantities():
    quantities = extract_quantities("A 2 kg block moves at 5 m/s for 3 s.")
    assert quantities["m_object"]["dimension"] == "mass"
    assert quantities["v"]["unit_symbol"] == "m/s"
    assert quantities["t"]["value"] == 3


def test_extract_two_charges_and_distance():
    quantities = extract_quantities("Two charges q1 = 6 nC and q2 = 3 nC are separated by 4 cm.")
    assert quantities["q1"]["normalized_value"] == pytest.approx(6e-9)
    assert quantities["q2"]["normalized_value"] == pytest.approx(3e-9)
    assert quantities["d"]["normalized_value"] == pytest.approx(0.04)


def test_extract_resistor_and_voltage():
    quantities = extract_quantities("A 10 Ω resistor is connected to a 5 V battery.")
    assert quantities["R"]["dimension"] == "resistance"
    assert quantities["V"]["dimension"] == "voltage"


def test_extract_symbolic_ratio_relations():
    relations = extract_relations("R1 = 2R2, q1 = 4q2, and m2 = 3m1.")
    assert any(r["left"] == "R1" and r["right"] == "R2" and r["factor"] == 2.0 for r in relations)
    assert any(r["left"] == "q1" and r["right"] == "q2" and r["factor"] == 4.0 for r in relations)
    assert any(r["left"] == "m2" and r["right"] == "m1" and r["factor"] == 3.0 for r in relations)


def test_extract_word_ratio_relations():
    relations = extract_relations("The charge is half the maximum charge. The electric energy is 1/4 of the total energy.")
    assert any(r["left"] == "q" and r["right"] == "Q_max" and r["factor"] == 0.5 for r in relations)
    assert any(r["left"] == "U_E" and r["right"] == "U_total" and r["factor"] == 0.25 for r in relations)


def test_extract_uncertainty_quantities():
    quantities = extract_quantities("U = 6.0 ± 0.1 V, I = 2.0 +/- 0.05 A, and R = 10 plus or minus 0.5 Ω.")
    assert quantities["V"]["value"] == 6.0
    assert quantities["delta_V"]["value"] == 0.1
    assert quantities["I"]["value"] == 2.0
    assert quantities["delta_I"]["value"] == 0.05
    assert quantities["R"]["value"] == 10
    assert quantities["delta_R"]["value"] == 0.5


def test_stage05_area_quantity():
    quantities = extract_quantities("The plate area is 33.2 cm².")
    assert quantities["A"]["dimension"] == "area"
    assert quantities["A"]["normalized_value"] == pytest.approx(33.2e-4)


def test_stage05_scientific_notation_charge():
    quantities = extract_quantities("The charge is 2 × 10^-6 C.")
    charge = next(q for q in quantities.values() if q["dimension"] == "charge")
    assert charge["normalized_value"] == pytest.approx(2e-6)


def test_stage05_angle_degree():
    quantities = extract_quantities("The two forces make an angle of 60°.")
    assert quantities["theta"]["dimension"] == "angle"
    assert quantities["theta"]["normalized_unit_symbol"] == "rad"


def test_stage05_symbolic_ratio_without_space():
    relations = extract_relations("q1 = 4q2.")
    assert any(r["left"] == "q1" and r["right"] == "q2" and r["factor"] == 4.0 for r in relations)


def test_stage05_minutes_are_time_not_frequency():
    quantities = extract_quantities("After another 20 minutes, find the distance.")
    assert quantities["t"]["dimension"] == "time"
    assert not any(q["dimension"] == "frequency" for q in quantities.values())
