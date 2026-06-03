"""Regression checks for Type 2 Stage 1 formula retrieval coverage."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parser.schemas import ProblemParseObject
from type2.stage1 import (
    FormulaRetriever,
    canonicalize_variable,
    canonicalize_variable_with_quantity,
    infer_domains,
    load_library,
)


class Stage1RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = FormulaRetriever()

    def _best_ids(self, parse_obj: ProblemParseObject) -> dict[str, str | None]:
        formula_sets = self.retriever.retrieve(parse_obj, beam_n=3)
        self.assertTrue(formula_sets)
        best = formula_sets[0]
        return {
            step_id: (entry.id if entry else None)
            for step_id, entry in best.formulas.items()
        }

    def test_library_contains_new_stage1_coverage_formulas(self) -> None:
        formula_ids = {entry.id for entry in load_library()}
        self.assertIn("CKT-016", formula_ids)
        self.assertIn("CKT-017", formula_ids)
        self.assertIn("CKT-019", formula_ids)
        self.assertIn("CKT-021", formula_ids)
        self.assertIn("CKT-022", formula_ids)
        self.assertIn("CKT-023", formula_ids)
        self.assertIn("CKT-024", formula_ids)
        self.assertIn("MEA-001", formula_ids)
        self.assertIn("MEA-002", formula_ids)
        self.assertIn("ELS-009", formula_ids)

    def test_canonicalizes_new_variable_families(self) -> None:
        self.assertEqual(canonicalize_variable("L_ind"), "inductance")
        self.assertEqual(canonicalize_variable("omega"), "angular_frequency")
        self.assertEqual(canonicalize_variable("X_L"), "reactance")
        self.assertEqual(canonicalize_variable("average reading"), "mean_value")
        self.assertEqual(canonicalize_variable("absolute error"), "absolute_error")
        self.assertEqual(canonicalize_variable("f"), "frequency")
        self.assertEqual(canonicalize_variable("F"), "force")
        self.assertEqual(canonicalize_variable("wavelength"), "wavelength")

    def test_unit_aware_canonicalization_overrides_ambiguous_names(self) -> None:
        self.assertEqual(
            canonicalize_variable_with_quantity("E", {"unit_symbol": "J"}),
            "energy",
        )
        self.assertEqual(
            canonicalize_variable_with_quantity("U", {"unit_symbol": "V"}),
            "electric_potential",
        )
        self.assertEqual(
            canonicalize_variable_with_quantity("L", {"unit_symbol": "cm"}),
            "displacement",
        )
        self.assertEqual(
            canonicalize_variable_with_quantity("C", {"unit_symbol": "nC"}),
            "electric_charge",
        )
        self.assertEqual(
            canonicalize_variable_with_quantity("T", {"unit_symbol": "ms"}),
            "time",
        )

    def test_infers_domain_when_stage0_reports_unknown(self) -> None:
        parse_obj = ProblemParseObject(
            problem_text="A capacitor in an LC resonant circuit has frequency 350 Hz.",
            domains=["unknown"],
            sub_domains=[],
            known_quantities={"f": {"value": 350, "unit_symbol": "Hz"}},
            unknown_quantity="C_cap",
            unknown_unit="pF",
        )
        domains, sub_domains = infer_domains(parse_obj)
        self.assertIn("circuits", domains)
        self.assertIn("lc_resonance", sub_domains)

    def test_retrieves_lc_resonance_capacitance(self) -> None:
        parse_obj = ProblemParseObject(
            problem_text="An LC circuit has L = 0.07 H and f = 350 Hz. Find capacitance.",
            domains=["electricity"],
            sub_domains=["lc_resonance"],
            known_quantities={
                "L": {"value": 0.07, "unit_symbol": "H", "unit_name": "henry"},
                "f": {"value": 350, "unit_symbol": "Hz", "unit_name": "hertz"},
            },
            unknown_quantity="C_cap",
            step_plan=[
                {
                    "step_id": "step_1",
                    "goal": "Solve LC resonance condition for capacitance.",
                    "type": "formula_application",
                    "template_name": "lc_resonance_capacitance",
                    "input_var": {"L": "L", "f": "f"},
                    "output_var": {"C_cap": "C_cap"},
                }
            ],
        )

        self.assertEqual(self._best_ids(parse_obj)["step_1"], "CKT-017")

    def test_retrieves_inductor_energy(self) -> None:
        parse_obj = ProblemParseObject(
            problem_text="Find the energy stored by an inductor.",
            domains=["electricity"],
            sub_domains=["inductor_energy"],
            known_quantities={
                "L_ind": {"value": 0.2, "unit_symbol": "H", "unit_name": "henry"},
                "I": {"value": 4, "unit_symbol": "A", "unit_name": "ampere"},
            },
            unknown_quantity="U_B",
            step_plan=[
                {
                    "step_id": "step_1",
                    "goal": "Use inductor energy formula.",
                    "type": "formula_application",
                    "template_name": "inductor_energy",
                    "input_var": {"L_ind": "L_ind", "I": "I"},
                    "output_var": {"U_B": "U_B"},
                }
            ],
        )

        self.assertEqual(self._best_ids(parse_obj)["step_1"], "CKT-019")

    def test_retrieves_ac_inductive_reactance_chain(self) -> None:
        parse_obj = ProblemParseObject(
            problem_text="Find inductive reactance from frequency and inductance.",
            domains=["electricity"],
            sub_domains=["ac_inductive_reactance"],
            known_quantities={
                "f": {"value": 60, "unit_symbol": "Hz", "unit_name": "hertz"},
                "L_ind": {"value": 0.5, "unit_symbol": "H", "unit_name": "henry"},
            },
            unknown_quantity="X_L",
            step_plan=[
                {
                    "step_id": "step_1",
                    "goal": "Compute angular frequency.",
                    "type": "formula_application",
                    "template_name": "angular_frequency",
                    "input_var": {"f": "f"},
                    "output_var": {"omega": "omega"},
                },
                {
                    "step_id": "step_2",
                    "goal": "Compute inductive reactance.",
                    "type": "formula_application",
                    "template_name": "ac_inductive_reactance",
                    "input_var": {"omega": "omega", "L_ind": "L_ind"},
                    "output_var": {"X_L": "X_L"},
                },
            ],
        )

        self.assertEqual(
            self._best_ids(parse_obj),
            {"step_1": "CKT-021", "step_2": "CKT-022"},
        )

    def test_retrieves_capacitor_charge(self) -> None:
        parse_obj = ProblemParseObject(
            problem_text="Find the charge on a capacitor from capacitance and voltage.",
            domains=["electricity"],
            sub_domains=["capacitor_charge"],
            known_quantities={
                "C_cap": {"value": 12, "unit_symbol": "F", "unit_name": "farad"},
                "V": {"value": 5, "unit_symbol": "V", "unit_name": "volt"},
            },
            unknown_quantity="Q",
            step_plan=[
                {
                    "step_id": "step_1",
                    "goal": "Use capacitor charge relation.",
                    "type": "formula_application",
                    "template_name": "capacitor_charge",
                    "input_var": {"C_cap": "C_cap", "V": "V"},
                    "output_var": {"Q": "Q"},
                }
            ],
        )

        self.assertEqual(self._best_ids(parse_obj)["step_1"], "CKT-016")

    def test_retrieves_measurement_absolute_error(self) -> None:
        parse_obj = ProblemParseObject(
            problem_text="Find the absolute error from a least count of 0.1 cm.",
            domains=["unknown"],
            sub_domains=[],
            known_quantities={
                "least_count": {
                    "value": 0.1,
                    "unit_symbol": "cm",
                    "normalized_value": 0.001,
                    "normalized_unit_symbol": "m",
                },
            },
            unknown_quantity="abs_error",
            unknown_unit="cm",
            step_plan=[
                {
                    "step_id": "step_1",
                    "goal": "Use least count as absolute error.",
                    "type": "formula_application",
                    "template_name": "measurement_absolute_error",
                    "input_var": {"least_count": "least_count"},
                    "output_var": {"abs_error": "abs_error"},
                }
            ],
        )

        self.assertEqual(self._best_ids(parse_obj)["step_1"], "MEA-001")

    def test_retrieves_electric_field_point_charge(self) -> None:
        parse_obj = ProblemParseObject(
            problem_text="Find the electric field due to a point charge.",
            domains=["unknown"],
            sub_domains=[],
            known_quantities={
                "q": {"value": 2e-9, "unit_symbol": "C"},
                "r": {"value": 0.2, "unit_symbol": "m"},
            },
            unknown_quantity="E",
            unknown_unit="N/C",
            step_plan=[
                {
                    "step_id": "step_1",
                    "goal": "Use point charge electric field.",
                    "type": "formula_application",
                    "template_name": "electric_field_point_charge",
                    "input_var": {"q": "q", "r": "r"},
                    "output_var": {"E": "E"},
                }
            ],
        )

        self.assertEqual(self._best_ids(parse_obj)["step_1"], "ELS-002")


if __name__ == "__main__":
    unittest.main()
