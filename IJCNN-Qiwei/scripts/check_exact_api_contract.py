#!/usr/bin/env python3
"""Local contract check for EXACT 2026 /predict output shape."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ijcnn_qiwei.exact_api import ExactPredictor


def main() -> None:
    predictor = ExactPredictor()
    examples = [
        {
            "query_id": "T1_0001",
            "type": "type1",
            "query": "Is Student A eligible for graduation?",
            "premises": [
                "A student with at least 120 credits is eligible for graduation.",
                "Student A has completed 118 credits.",
            ],
            "options": ["Yes", "No", "Uncertain"],
        },
        {
            "query_id": "T2_0001",
            "type": "type2",
            "query": "Two resistors R1 = 4 ohm and R2 = 6 ohm are in parallel across a 12V battery. Find the total current.",
            "premises": [],
            "options": [],
        },
    ]
    for payload in examples:
        result = predictor.predict_payload(payload)
        assert isinstance(result, list) and len(result) == 1
        row = result[0]
        for key in ("query_id", "answer", "unit", "explanation", "premises_used", "reasoning"):
            assert key in row, key
        assert row["query_id"] == payload["query_id"]
        assert row["explanation"]
        if payload["options"]:
            assert row["answer"] in payload["options"]
        if payload["type"] == "type2":
            assert row["premises_used"] == []
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
