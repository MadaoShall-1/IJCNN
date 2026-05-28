import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser.main import parse_problem
from parser.rule_extractor import extract_relations
from parser.target_detector import detect_target


def test_trig_assignment_function_extraction():
    relations = extract_relations("The current varies according to I = 3sin(50*pi*t). Calculate the maximum magnetic field energy.")
    function = next(relation for relation in relations if relation["type"] == "function")
    assert function["function_name"] == "I"
    assert function["dimension"] == "current"
    assert "sin" in function["expression"]


def test_fraction_relation_extraction():
    relations = extract_relations("The electric field energy is 3/4 of the total energy.")
    fraction = next(relation for relation in relations if relation["type"] == "percentage")
    assert fraction["quantity"] == "electric_energy_fraction"
    assert fraction["value"] == pytest.approx(0.75)


def test_remaining_target_phrases():
    assert detect_target("Calculate the percent relative uncertainty.") == ("percent_error", None)
    assert detect_target("What is the efficiency of the circuit?") == ("efficiency", None)
    assert detect_target("Calculate WC.") == ("U_E", "J")
    assert detect_target("Calculate the total flux linkage.") == ("Phi_link", "Wb")
    assert detect_target("Find C'.") == ("C_after", "F")
    assert detect_target("Calculate the power of each lamp.") == ("P_each", "W")


def test_percentage_relation_hook_for_efficiency():
    parsed = parse_problem("The efficiency is 60%. What is the efficiency of the circuit?")
    assert parsed["unknown_quantity"] == "efficiency"
    assert parsed["metadata"]["verifier_status"] == "PASS"
    assert "percentage_relation" in parsed["metadata"]["used_template_names"]
