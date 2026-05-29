"""Regression checks for deterministic Type 2 stages 2 through 6."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parser.schemas import ProblemParseObject
from type2.schemas import DiagnosisObject, FormulaEntry, FormulaSet, StepObject, TraceObject
from type2.stage2 import DeterministicSolveTrace, init_vso, map_formula_vars_to_vso
from type2.stage4 import diagnose_trace
from type2.stage5 import repair_trace
from type2.stage6 import build_response, extract_final_answer


def ohms_law_entry() -> FormulaEntry:
    return FormulaEntry(
        id="CKT-001",
        topic="circuits",
        subtopic="ohms_law",
        target_quantities=["V", "I", "R"],
        canonical_quantity_names=[
            "electric_potential",
            "electric_current",
            "resistance",
        ],
        text="Ohm's Law.",
        formula="V = I * R",
        sympy_expr="Eq(V, I * R)",
        tool_dispatch="sympy",
        variables={
            "V": {"symbol": "V", "name": "voltage", "unit_symbol": "V", "unit_name": "volts"},
            "I": {"symbol": "I", "name": "current", "unit_symbol": "A", "unit_name": "amperes"},
            "R": {"symbol": "R", "name": "resistance", "unit_symbol": "ohm", "unit_name": "ohms"},
        },
        premise_text="Ohm's Law: V = IR",
        fol_axiom="forall V I R: V = I * R",
    )


def power_vi_entry() -> FormulaEntry:
    return FormulaEntry(
        id="CKT-004",
        topic="circuits",
        subtopic="power_vi",
        target_quantities=["P", "V", "I"],
        canonical_quantity_names=[
            "power",
            "electric_potential",
            "electric_current",
        ],
        text="Electric power.",
        formula="P = V * I",
        sympy_expr="Eq(P, V * I)",
        tool_dispatch="sympy",
        variables={
            "P": {"symbol": "P", "name": "power", "unit_symbol": "W", "unit_name": "watts"},
            "V": {"symbol": "V", "name": "voltage", "unit_symbol": "V", "unit_name": "volts"},
            "I": {"symbol": "I", "name": "current", "unit_symbol": "A", "unit_name": "amperes"},
        },
        premise_text="Electric power: P = VI",
        fol_axiom="forall P V I: P = V * I",
    )


class DeterministicSolveTraceTests(unittest.TestCase):
    def test_solves_ohms_law_current_without_llm(self) -> None:
        parse_obj = ProblemParseObject(
            problem_text="A 10 ohm resistor is connected to a 5 V battery. Find current.",
            domains=["electricity"],
            sub_domains=["ohms_law"],
            known_quantities={
                "R": {"value": 10.0, "unit_symbol": "ohm", "unit_name": "ohm"},
                "V": {"value": 5.0, "unit_symbol": "V", "unit_name": "volt"},
            },
            step_plan=[
                {
                    "step_id": "step_1",
                    "goal": "Apply Ohm's law for current.",
                    "type": "formula_application",
                    "input_var": {"V": "V", "R": "R"},
                    "output_var": {"I": "I = V / R"},
                },
                {
                    "step_id": "step_2",
                    "goal": "Report final current.",
                    "type": "conclusion",
                    "input_var": {"I": "I"},
                    "output_var": {"I": "I"},
                },
            ],
        )
        formula_set = FormulaSet({"step_1": ohms_law_entry()}, 1.0, 0)

        trace = DeterministicSolveTrace().forward(parse_obj, formula_set, "ohms")

        self.assertEqual(trace.trace_status, "PASS")
        self.assertEqual(trace.steps[0].status, "OK")
        self.assertAlmostEqual(trace.vso["I"]["value"], 0.5)
        self.assertEqual(trace.final_answer, "0.5 A")

    def test_preserves_parser_output_alias_for_downstream_steps(self) -> None:
        parse_obj = ProblemParseObject(
            problem_text="A circuit has 5 V and 0.5 A. Find total resistance.",
            domains=["electricity"],
            sub_domains=["ohms_law"],
            known_quantities={
                "V": {"value": 5.0, "unit_symbol": "V", "unit_name": "volt"},
                "I": {"value": 0.5, "unit_symbol": "A", "unit_name": "ampere"},
            },
            step_plan=[
                {
                    "step_id": "step_1",
                    "goal": "Compute total resistance.",
                    "type": "formula_application",
                    "input_var": {"V": "V", "I": "I"},
                    "output_var": {"R_total": "R_total = V / I"},
                },
                {
                    "step_id": "step_2",
                    "goal": "Report final resistance.",
                    "type": "conclusion",
                    "input_var": {"R_total": "R_total"},
                    "output_var": {"R_total": "R_total"},
                },
            ],
        )
        formula_set = FormulaSet({"step_1": ohms_law_entry()}, 1.0, 0)

        trace = DeterministicSolveTrace().forward(parse_obj, formula_set, "resistance")

        self.assertEqual(trace.trace_status, "PASS")
        self.assertAlmostEqual(trace.vso["R_total"]["value"], 10.0)
        self.assertAlmostEqual(trace.vso["R"]["value"], 10.0)
        self.assertEqual(trace.final_answer, "10 ohm")

    def test_canonical_mapping_does_not_use_constants_as_problem_values(self) -> None:
        parse_obj = ProblemParseObject(problem_text="Find the electric field.")
        vso = init_vso(parse_obj)
        entry = FormulaEntry(
            id="EM-TEST",
            topic="electromagnetism",
            subtopic="electric_field",
            target_quantities=["E", "q"],
            canonical_quantity_names=["electric_field", "electric_charge"],
            text="Electric field relation.",
            formula="F = q * E",
            sympy_expr="Eq(F, q * E)",
            tool_dispatch="sympy",
            variables={
                "F": {"symbol": "F", "name": "force", "unit_symbol": "N", "unit_name": "newtons"},
                "q": {"symbol": "q", "name": "charge", "unit_symbol": "C", "unit_name": "coulombs"},
                "E": {"symbol": "E", "name": "electric field", "unit_symbol": "N/C", "unit_name": "newtons per coulomb"},
            },
        )

        mapped = map_formula_vars_to_vso(entry, vso)

        self.assertNotIn("E", mapped)
        self.assertNotIn("q", mapped)

    def test_repair_preserves_initial_known_quantities_when_fws_is_first(self) -> None:
        parse_obj = ProblemParseObject(
            problem_text="A 10 ohm resistor is connected to a 5 V battery. Find current.",
            domains=["electricity"],
            sub_domains=["ohms_law"],
            known_quantities={
                "R": {"value": 10.0, "unit_symbol": "ohm", "unit_name": "ohm"},
                "V": {"value": 5.0, "unit_symbol": "V", "unit_name": "volt"},
            },
            step_plan=[
                {
                    "step_id": "step_1",
                    "goal": "Apply Ohm's law for current.",
                    "type": "formula_application",
                    "input_var": {"V": "V", "R": "R"},
                    "output_var": {"I": "I = V / R"},
                },
                {
                    "step_id": "step_2",
                    "goal": "Report final current.",
                    "type": "conclusion",
                    "input_var": {"I": "I"},
                    "output_var": {"I": "I"},
                },
            ],
        )
        failed_trace = TraceObject(problem_id="repair", formula_path_index=0)
        failed_trace.steps = [
            StepObject(
                step_id="step_1",
                goal="Apply Ohm's law for current.",
                type="formula_application",
                formula_ids=["CKT-004"],
                status="WRONG",
            )
        ]
        diagnosis = DiagnosisObject(global_error_type="E1", fws_index=0)
        wrong_set = FormulaSet({"step_1": power_vi_entry()}, 0.6, 0)
        right_set = FormulaSet({"step_1": ohms_law_entry()}, 1.0, 1)

        repaired = repair_trace(
            trace=failed_trace,
            formula_set=wrong_set,
            parse_obj=parse_obj,
            diagnosis=diagnosis,
            solver=DeterministicSolveTrace(),
            all_formula_sets=[wrong_set, right_set],
        )

        self.assertEqual(repaired.trace_status, "REPAIRED")
        self.assertEqual(repaired.final_answer, "0.5 A")

    def test_stage4_diagnoses_missing_formula_as_formula_selection_error(self) -> None:
        trace = TraceObject(problem_id="diag", formula_path_index=0)
        trace.steps = [
            StepObject(
                step_id="step_1",
                goal="Compute current.",
                type="formula_application",
                status="WRONG",
                verifier_notes="No usable formula.",
            )
        ]
        formula_set = FormulaSet({"step_1": None}, 0.0, 0)

        diagnosis = diagnose_trace(trace, formula_set)

        self.assertEqual(diagnosis.fws_index, 0)
        self.assertEqual(diagnosis.global_error_type, "E1")
        self.assertIn("Replace the formula", diagnosis.repair_hint)

    def test_stage4_detects_chain_propagation_from_prior_output(self) -> None:
        trace = TraceObject(problem_id="diag-chain", formula_path_index=0)
        trace.steps = [
            StepObject(
                step_id="step_1",
                goal="Compute intermediate current.",
                type="formula_application",
                output_var={"I": 0.5},
                intermediate_answer="0.5 A",
                status="OK",
            ),
            StepObject(
                step_id="step_2",
                goal="Compute power.",
                type="formula_application",
                input_var={"I": {"value": 0.5}},
                intermediate_answer="bad value",
                status="WRONG",
            ),
        ]
        formula_set = FormulaSet(
            {"step_1": ohms_law_entry(), "step_2": power_vi_entry()},
            1.0,
            0,
        )

        diagnosis = diagnose_trace(trace, formula_set)

        self.assertEqual(diagnosis.fws_index, 1)
        self.assertEqual(diagnosis.global_error_type, "E5")
        self.assertIn("Roll back", diagnosis.repair_hint)

    def test_stage6_builds_response_with_answer_and_supporting_fields(self) -> None:
        parse_obj = ProblemParseObject(
            problem_text="A 10 ohm resistor is connected to a 5 V battery. Find current.",
            domains=["electricity"],
            sub_domains=["ohms_law"],
            known_quantities={
                "R": {"value": 10.0, "unit_symbol": "ohm", "unit_name": "ohm"},
                "V": {"value": 5.0, "unit_symbol": "V", "unit_name": "volt"},
            },
            step_plan=[
                {
                    "step_id": "step_1",
                    "goal": "Apply Ohm's law for current.",
                    "type": "formula_application",
                    "input_var": {"V": "V", "R": "R"},
                    "output_var": {"I": "I = V / R"},
                },
                {
                    "step_id": "step_2",
                    "goal": "Report final current.",
                    "type": "conclusion",
                    "input_var": {"I": "I"},
                    "output_var": {"I": "I"},
                },
            ],
        )
        formula_set = FormulaSet({"step_1": ohms_law_entry()}, 1.0, 0)
        trace = DeterministicSolveTrace().forward(parse_obj, formula_set, "stage6")
        diagnosis = diagnose_trace(trace, formula_set)

        response = build_response(trace, parse_obj, formula_set, diagnosis)

        self.assertEqual(extract_final_answer(trace), "0.5 A")
        self.assertEqual(response["answer"], "0.5 A")
        self.assertEqual(response["trace_status"], "PASS")
        self.assertGreater(response["confidence"], 0.9)
        self.assertEqual(response["premises"], ["Ohm's Law: V = IR"])
        self.assertEqual(response["fol_axioms"], ["forall V I R: V = I * R"])
        self.assertEqual(response["diagnosis"]["fws_index"], None)
        self.assertEqual(len(response["steps"]), 2)


if __name__ == "__main__":
    unittest.main()
