import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser.condition_extractor import extract_conditions


def test_rest_condition_adds_initial_velocity():
    conditions, additions = extract_conditions("A car starts from rest.", {})
    assert "initial_rest" in conditions
    assert additions["v_0"]["value"] == 0


def test_frictionless_condition():
    conditions, _ = extract_conditions("A block slides on a frictionless surface.", {})
    assert "frictionless" in conditions


def test_parallel_condition():
    conditions, _ = extract_conditions("The capacitors are connected in parallel.", {})
    assert "parallel_circuit" in conditions
