import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser.condition_extractor import extract_conditions
from parser.main import parse_problem
from parser.parse_verifier import verify_parse
from parser.rule_extractor import extract_quantities, extract_relations
from parser.target_detector import detect_target


def test_function_current_extraction():
    text = "The current is I(t)=2e^{-t/0.5} A. Find the charge passing through the wire."
    relations = extract_relations(text)
    function = next(relation for relation in relations if relation["type"] == "function")
    assert function["function_name"] == "I"
    assert function["independent_var"] == "t"
    assert function["dimension"] == "current"
    assert function["unit_symbol"] == "A"
    assert "exp" in function["expression"]
    assert detect_target(text)[0] in {"q", "Q"}


def test_function_voltage_extraction():
    relations = extract_relations("The voltage is U(t)=12sin(100*pi*t) V. Determine the maximum voltage.")
    function = next(relation for relation in relations if relation["type"] == "function")
    assert function["function_name"] == "U"
    assert function["dimension"] == "voltage"
    assert function["unit_symbol"] == "V"


def test_fraction_percentage_and_ratio_extraction():
    relations = extract_relations("The charge q1 is half of q2. The efficiency is 25%.")
    assert any(relation["type"] == "ratio" and relation["left"] == "q1" and relation["right"] == "q2" and relation["factor"] == 0.5 for relation in relations)
    percentage = next(relation for relation in relations if relation["type"] == "percentage")
    assert percentage["value"] == pytest.approx(0.25)
    assert percentage["raw_percent"] == pytest.approx(25.0)
    assert detect_target("How many times greater is q1 than q2?") == ("ratio", None)


def test_uncertainty_relations_and_quantities():
    quantities = extract_quantities("I = 2.0 ± 0.1 A.")
    relations = extract_relations("I = 2.0 ± 0.1 A.")
    assert quantities["I"]["value"] == pytest.approx(2.0)
    assert quantities["delta_I"]["value"] == pytest.approx(0.1)
    uncertainty = next(relation for relation in relations if relation["type"] == "uncertainty")
    assert uncertainty["quantity"] == "I"
    assert uncertainty["uncertainty"] == pytest.approx(0.1)

    voltage_relations = extract_relations("U = 5.0 +/- 0.2 V.")
    assert next(relation for relation in voltage_relations if relation["type"] == "uncertainty")["quantity"] == "V"

    relative = extract_relations("The relative uncertainty is 5%.")
    assert any(relation["type"] == "uncertainty" and relation["relative_uncertainty"] == pytest.approx(0.05) for relation in relative)


def test_equation_and_motion_relations():
    ratio = extract_relations("q1 = 4q2")
    assert any(relation["type"] == "ratio" and relation["factor"] == 4.0 for relation in ratio)

    equations = extract_relations("I1 + I2 = I")
    assert any(relation["type"] == "equation" and relation["equation"] == "I1 + I2 = I" for relation in equations)

    conditions, _ = extract_conditions("The two vehicles move toward each other and meet after 2 h.", {})
    assert "toward_each_other" in conditions
    motion = extract_relations("The two vehicles move toward each other and meet after 2 h.")
    assert any(relation["type"] == "equation" and "v1 + v2" in relation["equation"] for relation in motion)


def test_label_sensitive_disambiguation():
    radius = extract_quantities("R = 60 cm is the radius of a circular plate.")
    assert any(name in radius and radius[name]["dimension"] == "length" for name in ("r", "R_radius"))
    assert not any(quantity["dimension"] == "resistance" for quantity in radius.values())

    resistance = extract_quantities("R = 60 Ω.")
    assert resistance["R"]["dimension"] == "resistance"

    function = next(relation for relation in extract_relations("U(t)=10cos(100t) V.") if relation["type"] == "function")
    assert function["dimension"] == "voltage"

    energy = extract_quantities("The energy U is 10 J.")
    assert any(quantity["dimension"] == "energy" for quantity in energy.values())


def test_verifier_covers_numbers_inside_function_relation():
    parse_object = {
        "problem_text": "The current is I(t)=2e^{-t/0.5} A. Find the charge.",
        "known_quantities": {},
        "relations": extract_relations("The current is I(t)=2e^{-t/0.5} A. Find the charge."),
        "unknown_quantity": "Q",
        "step_plan": [],
        "domain_confidence": 0.9,
        "plan_confidence": 0.0,
        "metadata": {},
    }
    result = verify_parse(parse_object)
    assert not any(error.error_type == "missing_quantity" for error in result.errors)


def test_relation_driven_template_hooks():
    current = parse_problem("The current is I(t)=2e^{-t/0.5} A. Find the charge passing through the wire.")
    assert "function_current_integration" in current["metadata"]["used_template_names"]

    equation = parse_problem("Two vehicles move toward each other and meet after 2 h. Find the distance.")
    assert "equation_system_setup" in equation["metadata"]["used_template_names"]
