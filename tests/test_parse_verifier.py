import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser.parse_verifier import verify_parse


def base_parse():
    return {
        "problem_text": "A 5 V battery is connected to a 10 Ω resistor. Find the current.",
        "known_quantities": {
            "V": {"value": 5.0, "dimension": "voltage"},
            "R": {"value": 10.0, "dimension": "resistance"},
        },
        "unknown_quantity": "I",
        "step_plan": [
            {"step_id": "step_1", "type": "formula_application", "input_var": {"V": "V", "R": "R"}, "output_var": {"I": "I = V / R"}},
            {"step_id": "step_2", "type": "conclusion", "input_var": {"I": "I"}, "output_var": {"I": "I"}},
        ],
        "domain_confidence": 0.8,
        "plan_confidence": 0.8,
        "metadata": {},
    }


def test_missing_target_fails():
    data = base_parse()
    data["unknown_quantity"] = None
    assert any(error.error_type == "missing_target" for error in verify_parse(data).errors)


def test_invalid_dependency_fails():
    data = base_parse()
    data["step_plan"][0]["input_var"]["X"] = "X"
    assert any(error.error_type == "invalid_dependency" for error in verify_parse(data).errors)


def test_wrong_unit_dimension_fails():
    data = base_parse()
    data["known_quantities"]["I"] = {"value": 2.0, "dimension": "mass"}
    assert any(error.error_type == "wrong_unit" for error in verify_parse(data).errors)


def test_final_step_must_output_unknown():
    data = base_parse()
    data["step_plan"][-1]["output_var"] = {"P": "P"}
    assert any(error.error_type == "invalid_final_step" for error in verify_parse(data).errors)


def test_relation_factor_counts_as_numeric_coverage():
    data = base_parse()
    data["problem_text"] = "R1 = 2R2. Find the current."
    data["relations"] = [{"type": "ratio", "left": "R1", "right": "R2", "factor": 2.0, "source_text": "R1 = 2R2"}]
    result = verify_parse(data)
    assert not any(error.error_type == "missing_quantity" and "2" in error.description for error in result.errors)


def test_conceptual_relation_target_can_use_lightweight_plan():
    data = base_parse()
    data["problem_text"] = "What is the relationship between E1 and E2?"
    data["known_quantities"] = {}
    data["unknown_quantity"] = "relation_E"
    data["step_plan"] = [
        {"step_id": "step_1", "type": "formula_application", "input_var": {}, "output_var": {"relation_E": "extracted_relationship"}},
        {"step_id": "step_2", "type": "conclusion", "input_var": {"relation_E": "relation_E"}, "output_var": {"relation_E": "relation_E"}},
    ]
    data["plan_confidence"] = 0.8
    assert verify_parse(data).status == "PASS"
