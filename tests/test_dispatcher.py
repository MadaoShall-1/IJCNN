"""Tests for the top-level dispatcher and output normalizer."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
_TYPE2 = str(Path(__file__).resolve().parent.parent / "type2")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _TYPE2 not in sys.path:
    sys.path.insert(0, _TYPE2)

from dispatcher import dispatch
from output_normalizer import (
    normalize_type1_output,
    normalize_type2_output,
)

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"query_id", "answer", "unit", "explanation", "premises_used", "reasoning"}


def _assert_valid_response(result_list):
    assert isinstance(result_list, list), "response must be a list"
    assert len(result_list) == 1, "response must have exactly one element"
    item = result_list[0]
    for key in REQUIRED_KEYS:
        assert key in item, f"missing key: {key}"
    assert isinstance(item["premises_used"], list)


# ---------------------------------------------------------------------------
# Output normalizer unit tests
# ---------------------------------------------------------------------------

class TestNormalizeType1:
    def test_exact_option_match(self):
        raw = {"answer": "No", "explanation": "some reason"}
        result = normalize_type1_output("T1_001", raw, ["Yes", "No", "Uncertain"])
        assert result["answer"] == "No"
        assert result["unit"] == ""
        assert result["premises_used"] == []

    def test_case_insensitive_match(self):
        raw = {"answer": "yes", "explanation": "reason"}
        result = normalize_type1_output("T1_002", raw, ["Yes", "No", "Uncertain"])
        assert result["answer"] == "Yes"

    def test_alias_true_to_yes(self):
        raw = {"answer": "true", "explanation": "reason"}
        result = normalize_type1_output("T1_003", raw, ["Yes", "No", "Uncertain"])
        assert result["answer"] == "Yes"

    def test_unknown_fallback_to_uncertain(self):
        raw = {"answer": "unknown", "explanation": "reason"}
        result = normalize_type1_output("T1_004", raw, ["Yes", "No", "Uncertain"])
        assert result["answer"] == "Uncertain"

    def test_invalid_answer_fallback(self):
        raw = {"answer": "banana", "explanation": "reason"}
        result = normalize_type1_output("T1_005", raw, ["Yes", "No", "Uncertain"])
        assert result["answer"] == "Uncertain"

    def test_no_options_passthrough(self):
        raw = {"answer": "42 degrees", "explanation": "reason"}
        result = normalize_type1_output("T1_006", raw, [])
        assert result["answer"] == "42 degrees"
        assert result["unit"] == ""

    def test_empty_explanation_fallback(self):
        raw = {"answer": "Yes", "explanation": ""}
        result = normalize_type1_output("T1_007", raw, ["Yes", "No"])
        assert result["explanation"] != ""


class TestNormalizeType2:
    def test_splits_answer_and_unit(self):
        raw = {"answer": "5 A", "explanation": "Ohm's law"}
        result = normalize_type2_output("T2_001", raw)
        assert result["answer"] == "5"
        assert result["unit"] == "A"
        assert result["premises_used"] == []

    def test_unicode_unit_normalisation(self):
        raw = {"answer": "10", "unit": "Ω", "explanation": "reason"}
        result = normalize_type2_output("T2_002", raw)
        assert result["unit"] == "ohm"

    def test_micro_unit_normalisation(self):
        raw = {"answer": "4.7", "unit": "μF", "explanation": "reason"}
        result = normalize_type2_output("T2_003", raw)
        assert result["unit"] == "uF"


# ---------------------------------------------------------------------------
# Dispatcher integration tests
# ---------------------------------------------------------------------------

class TestDispatchType1:
    def test_type1_choice_question(self):
        mock_raw = {
            "answer": "No",
            "explanation": "Student A has 118 credits, below 120.",
            "premises_used": [0, 1],
        }
        with patch("dispatcher._solve_type1", return_value=mock_raw):
            result = dispatch({
                "query_id": "T1_0001",
                "type": "type1",
                "query": "Is Student A eligible for graduation?",
                "premises": [
                    "A student with at least 120 credits is eligible.",
                    "Student A has 118 credits.",
                ],
                "options": ["Yes", "No", "Uncertain"],
            })
        _assert_valid_response(result)
        assert result[0]["query_id"] == "T1_0001"
        assert result[0]["answer"] in ["Yes", "No", "Uncertain"]
        assert result[0]["unit"] == ""
        assert isinstance(result[0]["premises_used"], list)
        assert result[0]["explanation"]

    def test_type1_free_form(self):
        mock_raw = {
            "answer": "The capital of France is Paris.",
            "explanation": "Based on premises.",
        }
        with patch("dispatcher._solve_type1", return_value=mock_raw):
            result = dispatch({
                "query_id": "T1_0002",
                "type": "type1",
                "query": "What is the capital of France?",
                "premises": ["France is a country in Europe."],
                "options": [],
            })
        _assert_valid_response(result)
        assert result[0]["unit"] == ""
        assert result[0]["answer"]
        assert isinstance(result[0]["premises_used"], list)
        assert result[0]["explanation"]


class TestDispatchType2:
    def test_type2_physics(self):
        mock_raw = {
            "answer": "5",
            "unit": "A",
            "chain_of_thought": "1/R = 1/4 + 1/6 = 5/12, R = 2.4, I = 12/2.4 = 5",
            "confidence": 0.95,
        }
        with patch("dispatcher._solve_type2", return_value=mock_raw):
            result = dispatch({
                "query_id": "T2_0001",
                "type": "type2",
                "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12V battery. Find the total current.",
                "premises": [],
                "options": [],
            })
        _assert_valid_response(result)
        assert result[0]["query_id"] == "T2_0001"
        assert result[0]["answer"]
        assert result[0]["premises_used"] == []
        assert result[0]["explanation"]
        assert result[0]["unit"].isascii()


class TestDispatchExceptions:
    def test_type1_solver_exception(self):
        with patch("dispatcher._solve_type1", side_effect=RuntimeError("boom")):
            result = dispatch({
                "query_id": "T1_ERR",
                "type": "type1",
                "query": "test?",
                "premises": [],
                "options": ["Yes", "No", "Uncertain"],
            })
        _assert_valid_response(result)
        assert result[0]["query_id"] == "T1_ERR"
        assert result[0]["unit"] == ""

    def test_type2_solver_exception(self):
        with patch("dispatcher._solve_type2", side_effect=RuntimeError("boom")):
            result = dispatch({
                "query_id": "T2_ERR",
                "type": "type2",
                "query": "test?",
                "premises": [],
                "options": [],
            })
        _assert_valid_response(result)
        assert result[0]["query_id"] == "T2_ERR"
        assert result[0]["premises_used"] == []

    def test_unknown_type_fallback(self):
        result = dispatch({
            "query_id": "UNK",
            "type": "type99",
            "query": "test?",
        })
        _assert_valid_response(result)
        assert result[0]["query_id"] == "UNK"
