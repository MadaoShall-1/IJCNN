"""Tests for the /predict FastAPI endpoint (HTTP-level)."""

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

try:
    from fastapi.testclient import TestClient
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

REQUIRED_KEYS = {"query_id", "answer", "unit", "explanation", "premises_used", "reasoning"}


def _assert_valid_response(resp_json):
    assert isinstance(resp_json, list)
    assert len(resp_json) == 1
    for key in REQUIRED_KEYS:
        assert key in resp_json[0], f"missing key: {key}"


@pytest.fixture
def client():
    if not _FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not installed")
    with patch("api._load_models"):
        from api import app
        with TestClient(app) as c:
            yield c


class TestPredictEndpoint:
    def test_type1_choice(self, client):
        mock_raw = {
            "answer": "No",
            "explanation": "Student A has only 118 credits.",
            "premises_used": [0, 1],
        }
        with patch("dispatcher._solve_type1", return_value=mock_raw):
            resp = client.post("/predict", json={
                "query_id": "T1_0001",
                "type": "type1",
                "query": "Is Student A eligible for graduation?",
                "premises": [
                    "A student with at least 120 credits is eligible.",
                    "Student A has 118 credits.",
                ],
                "options": ["Yes", "No", "Uncertain"],
            })
        assert resp.status_code == 200
        data = resp.json()
        _assert_valid_response(data)
        assert data[0]["query_id"] == "T1_0001"
        assert data[0]["answer"] in ["Yes", "No", "Uncertain"]
        assert data[0]["unit"] == ""

    def test_type1_free_form(self, client):
        mock_raw = {"answer": "Paris", "explanation": "Paris is the capital."}
        with patch("dispatcher._solve_type1", return_value=mock_raw):
            resp = client.post("/predict", json={
                "query_id": "T1_0002",
                "type": "type1",
                "query": "What is the capital of France?",
                "premises": ["France is a European country."],
                "options": [],
            })
        assert resp.status_code == 200
        data = resp.json()
        _assert_valid_response(data)
        assert data[0]["unit"] == ""
        assert data[0]["answer"] == "Paris"

    def test_type2_physics(self, client):
        mock_raw = {
            "answer": "5",
            "unit": "A",
            "chain_of_thought": "I = V / R_total",
            "confidence": 0.9,
        }
        with patch("dispatcher._solve_type2", return_value=mock_raw):
            resp = client.post("/predict", json={
                "query_id": "T2_0001",
                "type": "type2",
                "query": "Two resistors R1=4ohm R2=6ohm in parallel across 12V. Find total current.",
                "premises": [],
                "options": [],
            })
        assert resp.status_code == 200
        data = resp.json()
        _assert_valid_response(data)
        assert data[0]["query_id"] == "T2_0001"
        assert data[0]["premises_used"] == []
        assert data[0]["unit"].isascii()

    def test_type1_solver_crash(self, client):
        with patch("dispatcher._solve_type1", side_effect=RuntimeError("crash")):
            resp = client.post("/predict", json={
                "query_id": "T1_CRASH",
                "type": "type1",
                "query": "test?",
                "premises": [],
                "options": ["Yes", "No", "Uncertain"],
            })
        assert resp.status_code == 200
        data = resp.json()
        _assert_valid_response(data)
        assert data[0]["query_id"] == "T1_CRASH"

    def test_type2_solver_crash(self, client):
        with patch("dispatcher._solve_type2", side_effect=RuntimeError("crash")):
            resp = client.post("/predict", json={
                "query_id": "T2_CRASH",
                "type": "type2",
                "query": "test?",
                "premises": [],
                "options": [],
            })
        assert resp.status_code == 200
        data = resp.json()
        _assert_valid_response(data)
        assert data[0]["premises_used"] == []

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
