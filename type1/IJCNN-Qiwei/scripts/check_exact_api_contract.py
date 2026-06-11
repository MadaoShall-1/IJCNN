#!/usr/bin/env python3
"""Local contract check for the unified root /predict output shape.

Type 1 is checked in-process via Type1Predictor. Type 2 now lives in the
sibling type2/ repo and is served only by the root-level api.py, so this
script checks it over HTTP when the root API is running (skipped otherwise).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ijcnn_qiwei.type1_predictor import Type1Predictor

REQUIRED_KEYS = ("query_id", "answer", "unit", "explanation", "premises_used", "reasoning")
ROOT_PREDICT_URL = "http://127.0.0.1:8080/predict"


def _check_row(payload: dict, result: list) -> None:
    assert isinstance(result, list) and len(result) == 1
    row = result[0]
    for key in REQUIRED_KEYS:
        assert key in row, key
    assert row["query_id"] == payload["query_id"]
    assert row["explanation"]
    if payload["options"]:
        assert row["answer"] in payload["options"]
    if payload["type"] == "type2":
        assert row["premises_used"] == []
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    type1_payload = {
        "query_id": "T1_0001",
        "type": "type1",
        "query": "Is Student A eligible for graduation?",
        "premises": [
            "A student with at least 120 credits is eligible for graduation.",
            "Student A has completed 118 credits.",
        ],
        "options": ["Yes", "No", "Uncertain"],
    }
    _check_row(type1_payload, Type1Predictor().predict_payload(type1_payload))

    type2_payload = {
        "query_id": "T2_0001",
        "type": "type2",
        "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12V battery. Find the total current.",
        "premises": [],
        "options": [],
    }
    request = urllib.request.Request(
        ROOT_PREDICT_URL,
        data=json.dumps(type2_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            _check_row(type2_payload, json.loads(response.read().decode("utf-8")))
    except urllib.error.URLError:
        print("type2 check skipped: root /predict is not running at", ROOT_PREDICT_URL)


if __name__ == "__main__":
    main()
