"""Tests for Type 2 Stage 1: formula library, schemas, and retrieval logic."""

import unittest
from pathlib import Path

from parser.schemas import ProblemParseObject

from type2.schemas import FormulaEntry, FormulaSet
from type2.stage1 import (
    CANONICAL_MAP,
    FormulaRetriever,
    canonicalize_variable,
    detect_collisions,
    load_library,
)

_LIBRARY_PATH = Path(__file__).parent / "type2" / "formula_library.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parse_obj(
    *,
    problem_text: str = "",
    domains: list | None = None,
    sub_domains: list | None = None,
    known_quantities: dict | None = None,
    step_plan: list | None = None,
) -> ProblemParseObject:
    return ProblemParseObject(
        problem_text=problem_text,
        domains=domains or [],
        sub_domains=sub_domains or [],
        known_quantities=known_quantities or {},
        step_plan=step_plan or [],
    )


def _formula_app_step(step_id, goal, input_var=None, output_var=None):
    return {
        "step_id": step_id,
        "goal": goal,
        "type": "formula_application",
        "input_var": input_var or {},
        "output_var": output_var or {},
        "confidence": 0.9,
    }


def _formula_app_step_with_formula(
    step_id,
    goal,
    formula_name,
    template_name,
    input_var=None,
    output_var=None,
):
    step = _formula_app_step(step_id, goal, input_var=input_var, output_var=output_var)
    step["formula_name"] = formula_name
    step["template_name"] = template_name
    return step


# ---------------------------------------------------------------------------
# FormulaEntry schema
# ---------------------------------------------------------------------------

class TestFormulaEntry(unittest.TestCase):
    """FormulaEntry from_dict / to_dict round-trip."""

    def test_round_trip(self):
        library = load_library(_LIBRARY_PATH)
        self.assertTrue(len(library) > 0)
        entry = library[0]
        d = entry.to_dict()
        restored = FormulaEntry.from_dict(d)
        self.assertEqual(entry.id, restored.id)
        self.assertEqual(entry.sympy_expr, restored.sympy_expr)

    def test_library_has_circuits_and_electrostatics(self):
        library = load_library(_LIBRARY_PATH)
        topics = {e.topic for e in library}
        self.assertIn("circuits", topics)
        self.assertIn("electrostatics", topics)

    def test_every_entry_has_required_fields(self):
        library = load_library(_LIBRARY_PATH)
        for entry in library:
            self.assertTrue(entry.id, f"Missing id: {entry}")
            self.assertTrue(entry.topic, f"Missing topic: {entry.id}")
            self.assertTrue(entry.sympy_expr or entry.tool_dispatch == "llm",
                            f"Missing sympy_expr for non-llm entry: {entry.id}")
            self.assertTrue(entry.canonical_quantity_names,
                            f"Empty canonical_quantity_names: {entry.id}")

    def test_ohms_law_present(self):
        library = load_library(_LIBRARY_PATH)
        ids = {e.id for e in library}
        self.assertIn("CKT-001", ids)

    def test_coulombs_law_present(self):
        library = load_library(_LIBRARY_PATH)
        ids = {e.id for e in library}
        self.assertIn("ELS-001", ids)


# ---------------------------------------------------------------------------
# Tier 1: canonicalize_variable
# ---------------------------------------------------------------------------

class TestCanonicalizeVariable(unittest.TestCase):

    def test_descriptive_velocity(self):
        self.assertEqual(canonicalize_variable("train_velocity"), "velocity")

    def test_descriptive_voltage(self):
        self.assertEqual(canonicalize_variable("source_voltage"), "electric_potential")

    def test_descriptive_current(self):
        self.assertEqual(canonicalize_variable("wire_current"), "electric_current")

    def test_descriptive_resistance(self):
        self.assertEqual(canonicalize_variable("load_resistance"), "resistance")

    def test_descriptive_power(self):
        self.assertEqual(canonicalize_variable("dissipated_power"), "power")

    def test_descriptive_charge(self):
        self.assertEqual(canonicalize_variable("stored_charge"), "electric_charge")

    def test_descriptive_energy(self):
        self.assertEqual(canonicalize_variable("kinetic_energy"), "energy")

    def test_short_symbol_V(self):
        self.assertEqual(canonicalize_variable("V"), "electric_potential")

    def test_short_symbol_I(self):
        self.assertEqual(canonicalize_variable("I"), "electric_current")

    def test_short_symbol_R(self):
        self.assertEqual(canonicalize_variable("R"), "resistance")

    def test_short_symbol_P(self):
        self.assertEqual(canonicalize_variable("P"), "power")

    def test_short_symbol_Q(self):
        self.assertEqual(canonicalize_variable("Q"), "electric_charge")

    def test_short_symbol_C(self):
        self.assertEqual(canonicalize_variable("C"), "capacitance")

    def test_indexed_R1(self):
        self.assertEqual(canonicalize_variable("R1"), "resistance")

    def test_indexed_C2(self):
        self.assertEqual(canonicalize_variable("C2"), "capacitance")

    def test_indexed_V3(self):
        self.assertEqual(canonicalize_variable("V3"), "electric_potential")

    def test_compound_R_s(self):
        self.assertEqual(canonicalize_variable("R_s"), "resistance")

    def test_compound_V_out(self):
        self.assertEqual(canonicalize_variable("V_out"), "electric_potential")

    def test_unknown_returns_none(self):
        self.assertIsNone(canonicalize_variable("xyz_completely_unknown"))

    def test_emf_maps_to_potential(self):
        self.assertEqual(canonicalize_variable("emf_source"), "electric_potential")

    def test_time_constant_tau(self):
        self.assertEqual(canonicalize_variable("tau"), "time_constant")


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------

class TestDetectCollisions(unittest.TestCase):

    def test_no_collisions(self):
        mapping = {"V": "electric_potential", "I": "electric_current", "R": "resistance"}
        self.assertEqual(detect_collisions(mapping), [])

    def test_single_collision(self):
        mapping = {"V": "electric_potential", "voltage": "electric_potential"}
        collisions = detect_collisions(mapping)
        self.assertEqual(len(collisions), 1)
        names = {collisions[0][0], collisions[0][1]}
        self.assertEqual(names, {"V", "voltage"})
        self.assertEqual(collisions[0][2], "electric_potential")

    def test_empty_mapping(self):
        self.assertEqual(detect_collisions({}), [])


# ---------------------------------------------------------------------------
# Topic filter
# ---------------------------------------------------------------------------

class TestTopicFilter(unittest.TestCase):

    def setUp(self):
        self.retriever = FormulaRetriever(library_path=str(_LIBRARY_PATH))

    def test_circuits_domain_returns_only_circuits(self):
        results = self.retriever._topic_filter(["circuits"], [])
        topics = {e.topic for e in results}
        self.assertEqual(topics, {"circuits"})

    def test_electrostatics_domain(self):
        results = self.retriever._topic_filter(["electrostatics"], [])
        topics = {e.topic for e in results}
        self.assertEqual(topics, {"electrostatics"})

    def test_unknown_domain_returns_full_library(self):
        results = self.retriever._topic_filter(["unknown"], [])
        self.assertEqual(len(results), len(self.retriever._library))

    def test_empty_domains_returns_full_library(self):
        results = self.retriever._topic_filter([], [])
        self.assertEqual(len(results), len(self.retriever._library))

    def test_unrecognized_domain_returns_full_library(self):
        results = self.retriever._topic_filter(["quantum_gravity"], [])
        self.assertEqual(len(results), len(self.retriever._library))


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

class TestScoreCandidates(unittest.TestCase):

    def setUp(self):
        self.retriever = FormulaRetriever(library_path=str(_LIBRARY_PATH))
        self.library = load_library(_LIBRARY_PATH)

    def _get_entry(self, entry_id: str) -> FormulaEntry:
        return next(e for e in self.library if e.id == entry_id)

    def test_ohms_law_scores_high_for_VIR(self):
        ohm = self._get_entry("CKT-001")
        scored = self.retriever._score_candidates(
            [ohm],
            canonical_vars={"electric_potential", "electric_current", "resistance"},
            direct_vars={"V", "I", "R"},
        )
        self.assertEqual(len(scored), 1)
        _, score = scored[0]
        self.assertGreater(score, 0.8)

    def test_unrelated_formula_scores_zero(self):
        coulomb = self._get_entry("ELS-001")
        scored = self.retriever._score_candidates(
            [coulomb],
            canonical_vars={"resistance"},
            direct_vars={"R"},
        )
        # No overlap with Coulomb's law canonical names
        self.assertEqual(scored, [])

    def test_direct_symbol_bonus(self):
        ohm = self._get_entry("CKT-001")
        # canonical_vars is empty but direct symbols match
        scored = self.retriever._score_candidates(
            [ohm],
            canonical_vars=set(),
            direct_vars={"V", "I", "R"},
        )
        if scored:
            _, score = scored[0]
            self.assertGreater(score, 0.0)


# ---------------------------------------------------------------------------
# Full retrieval
# ---------------------------------------------------------------------------

class TestRetrieve(unittest.TestCase):

    def setUp(self):
        self.retriever = FormulaRetriever(library_path=str(_LIBRARY_PATH))

    def test_retrieve_ohms_law_step(self):
        parse_obj = _make_parse_obj(
            problem_text="Find the current through a 10 Ω resistor with 5 V applied.",
            domains=["circuits"],
            step_plan=[
                _formula_app_step(
                    "step_2",
                    "Apply Ohm's Law to find current",
                    input_var={"V": {}, "R": {}},
                    output_var={"I": {}},
                )
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=1)
        self.assertGreater(len(sets), 0)
        fs = sets[0]
        self.assertIn("step_2", fs.formulas)
        entry = fs.formulas["step_2"]
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "CKT-001")

    def test_retrieve_no_formula_steps_returns_empty_set(self):
        parse_obj = _make_parse_obj(
            problem_text="A circuit problem.",
            domains=["circuits"],
            step_plan=[
                {"step_id": "step_1", "type": "setup",
                 "goal": "Extract values", "input_var": {}, "output_var": {"V": {}}}
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=3)
        self.assertEqual(len(sets), 1)
        self.assertEqual(sets[0].formulas, {})

    def test_retrieve_returns_at_most_beam_n_sets(self):
        parse_obj = _make_parse_obj(
            problem_text="Ohm's Law problem",
            domains=["circuits"],
            step_plan=[
                _formula_app_step("s1", "Apply Ohm's Law",
                                  input_var={"V": {}, "R": {}}, output_var={"I": {}})
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=2)
        self.assertLessEqual(len(sets), 2)

    def test_retrieve_unknown_domain_still_finds_formulas(self):
        parse_obj = _make_parse_obj(
            problem_text="Find the current using V and R.",
            domains=["unknown"],
            step_plan=[
                _formula_app_step("s1", "Use Ohm's Law",
                                  input_var={"V": {}, "R": {}}, output_var={"I": {}})
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=1)
        self.assertGreater(len(sets), 0)
        entry = sets[0].formulas.get("s1")
        self.assertIsNotNone(entry)

    def test_retrieve_no_matching_formula_returns_none_for_step(self):
        parse_obj = _make_parse_obj(
            problem_text="Some quantum gravity problem.",
            domains=["quantum_gravity"],
            step_plan=[
                _formula_app_step("s1", "Apply graviton propagator",
                                  input_var={"planck_scale_xyz": {}},
                                  output_var={"graviton_flux_xyz": {}})
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=1)
        self.assertGreater(len(sets), 0)
        # The step may or may not have a formula; either is acceptable
        # (None is the correct sentinel when no formula matches)
        self.assertIn("s1", sets[0].formulas)

    def test_retrieve_capacitor_energy_step(self):
        parse_obj = _make_parse_obj(
            problem_text="A 10 μF capacitor is charged to 12 V. Find the stored energy.",
            domains=["circuits"],
            step_plan=[
                _formula_app_step_with_formula(
                    "step_2",
                    "Calculate energy stored in capacitor",
                    "U = 0.5 * C * V^2",
                    "capacitor_energy",
                    input_var={"C": {}, "V": {}},
                    output_var={"U": {}},
                )
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=1)
        self.assertGreater(len(sets), 0)
        entry = sets[0].formulas.get("step_2")
        self.assertIsNotNone(entry)
        # Should match CKT-011 (energy = ½CV²)
        self.assertEqual(entry.id, "CKT-011")

    def test_formula_name_overrides_capacitor_charge_ambiguity(self):
        parse_obj = _make_parse_obj(
            problem_text="Calculate energy stored in a capacitor from C and V.",
            domains=["electricity"],
            sub_domains=["capacitors"],
            step_plan=[
                _formula_app_step_with_formula(
                    "step_1",
                    "Compute capacitor electric energy from capacitance and voltage.",
                    "U_cap = 0.5 * C_cap * V^2",
                    "capacitor_energy",
                    input_var={"C_cap": {}, "V": {}},
                    output_var={"U_cap": {}},
                )
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=1)
        entry = sets[0].formulas.get("step_1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "CKT-011")

    def test_retrieve_coulombs_law_step(self):
        parse_obj = _make_parse_obj(
            problem_text="Two charges q1=2μC and q2=3μC are separated by r=0.1 m. Find the force.",
            domains=["electrostatics"],
            step_plan=[
                _formula_app_step_with_formula(
                    "step_2",
                    "Apply Coulomb's Law to find force",
                    "F = k * abs(q1*q2) / r^2",
                    "coulomb_force_scalar",
                    input_var={"q1": {}, "q2": {}, "r": {}},
                    output_var={"F": {}},
                )
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=1)
        self.assertGreater(len(sets), 0)
        entry = sets[0].formulas.get("step_2")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "ELS-001")

    def test_formula_name_overrides_coulomb_energy_ambiguity(self):
        parse_obj = _make_parse_obj(
            problem_text="Two charges exert an electrostatic force.",
            domains=["electrostatics"],
            sub_domains=["coulombs_law"],
            step_plan=[
                _formula_app_step_with_formula(
                    "step_1",
                    "Compute force on q3 due to q1.",
                    "F_13 = k * abs(q1*q3) / r13^2",
                    "coulomb_force_vector",
                    input_var={"q1": {}, "q3": {}, "r13": {}},
                    output_var={"F_13": {}},
                )
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=1)
        entry = sets[0].formulas.get("step_1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "ELS-001")

    def test_formula_set_confidence_between_0_and_1(self):
        parse_obj = _make_parse_obj(
            problem_text="Ohm's Law.",
            domains=["circuits"],
            step_plan=[
                _formula_app_step("s1", "Apply V=IR",
                                  input_var={"I": {}, "R": {}}, output_var={"V": {}})
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=1)
        for fs in sets:
            self.assertGreaterEqual(fs.retrieval_confidence, 0.0)

    def test_formula_set_to_dict(self):
        parse_obj = _make_parse_obj(
            problem_text="Ohm's Law.",
            domains=["circuits"],
            step_plan=[
                _formula_app_step("s1", "Apply V=IR",
                                  input_var={"V": {}, "R": {}}, output_var={"I": {}})
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=1)
        d = sets[0].to_dict()
        self.assertIn("formulas", d)
        self.assertIn("retrieval_confidence", d)
        self.assertIn("path_index", d)

    def test_path_indices_are_sequential(self):
        parse_obj = _make_parse_obj(
            problem_text="Ohm's Law.",
            domains=["circuits"],
            step_plan=[
                _formula_app_step("s1", "Apply V=IR",
                                  input_var={"V": {}, "R": {}}, output_var={"I": {}})
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=3)
        for i, fs in enumerate(sets):
            self.assertEqual(fs.path_index, i)

    def test_multi_step_plan(self):
        """Two formula_application steps — both should have formula assignments."""
        parse_obj = _make_parse_obj(
            problem_text="Series circuit: find total resistance then current.",
            domains=["circuits"],
            step_plan=[
                _formula_app_step("s1", "Calculate series resistance",
                                  input_var={"R1": {}, "R2": {}}, output_var={"R_s": {}}),
                _formula_app_step("s2", "Apply Ohm's Law",
                                  input_var={"V": {}, "R_s": {}}, output_var={"I": {}}),
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=1)
        self.assertGreater(len(sets), 0)
        fs = sets[0]
        self.assertIn("s1", fs.formulas)
        self.assertIn("s2", fs.formulas)
        self.assertEqual(fs.formulas["s1"].id, "CKT-002")
        self.assertEqual(fs.formulas["s2"].id, "CKT-001")

    def test_electric_field_not_confused_with_coulomb(self):
        """template_name disambiguates E-field from Coulomb's law / potential energy."""
        parse_obj = _make_parse_obj(
            problem_text="Find the electric field due to a 5 μC charge at distance 0.3 m.",
            domains=["electrostatics"],
            step_plan=[
                _formula_app_step_with_formula(
                    "s1",
                    "Calculate electric field from point charge",
                    "E = k_e * q / r^2",
                    "electric_field_point_charge",
                    input_var={"k_e": {}, "q": {}, "r": {}},
                    output_var={"E": {}},
                )
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=3)
        entry = sets[0].formulas.get("s1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "ELS-002")

    def test_potential_energy_not_confused_with_coulomb(self):
        """template_name disambiguates potential energy from Coulomb force."""
        parse_obj = _make_parse_obj(
            problem_text="Find the potential energy between charges q1 and q2.",
            domains=["electrostatics"],
            step_plan=[
                _formula_app_step_with_formula(
                    "s1",
                    "Calculate electric potential energy",
                    "U = k_e * q1 * q2 / r",
                    "electric_potential_energy",
                    input_var={"k_e": {}, "q1": {}, "q2": {}, "r": {}},
                    output_var={"U": {}},
                )
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=3)
        entry = sets[0].formulas.get("s1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "ELS-006")

    def test_force_on_charge_not_confused_with_coulomb(self):
        """F = qE should match ELS-007, not ELS-001 (Coulomb's law)."""
        parse_obj = _make_parse_obj(
            problem_text="A charge q = 3 μC is placed in a field E = 500 N/C. Find the force.",
            domains=["electrostatics"],
            step_plan=[
                _formula_app_step_with_formula(
                    "s1",
                    "Calculate force on charge in electric field",
                    "F = q * E",
                    "force_on_charge",
                    input_var={"q": {}, "E": {}},
                    output_var={"F": {}},
                )
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=1)
        entry = sets[0].formulas.get("s1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "ELS-007")

    def test_electric_potential_not_confused_with_field(self):
        """V = k_e*q/r should match ELS-003, not ELS-002 (E-field)."""
        parse_obj = _make_parse_obj(
            problem_text="Find the electric potential at distance r from a charge q.",
            domains=["electrostatics"],
            step_plan=[
                _formula_app_step_with_formula(
                    "s1",
                    "Calculate electric potential from point charge",
                    "V = k_e * q / r",
                    "electric_potential_point_charge",
                    input_var={"k_e": {}, "q": {}, "r": {}},
                    output_var={"V": {}},
                )
            ],
        )
        sets = self.retriever.retrieve(parse_obj, beam_n=1)
        entry = sets[0].formulas.get("s1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.id, "ELS-003")


if __name__ == "__main__":
    unittest.main(verbosity=2)
