#!/usr/bin/env python3
"""Local Type1-only contract check for EXACT 2026 /predict output shape."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ijcnn_qiwei.type1_predictor import Type1Predictor as ExactPredictor


REQUIRED_KEYS = {
    "query_id",
    "answer",
    "unit",
    "explanation",
    "premises_used",
    "reasoning",
}


def main() -> None:
    predictor = ExactPredictor()
    examples = [
        {
            "query_id": "T1_YNU_0001",
            "type": "type1",
            "query": "Is Student A eligible for graduation?",
            "premises": [
                "A student with at least 120 credits is eligible for graduation.",
                "Student A has completed 118 credits.",
            ],
            "options": ["Yes", "No", "Uncertain"],
        },
        {
            "query_id": "T1_MCQ_0001",
            "type": "type1",
            "query": "Which option is supported by the premises?",
            "premises": [
                "The library is open on weekdays.",
                "Today is Wednesday.",
            ],
            "options": [
                "The library is open today.",
                "The library is closed today.",
                "The library only opens on weekends.",
            ],
        },
        {
            "query_id": "T1_FREE_0001",
            "type": "type1",
            "query": "How many credits has Student A completed?",
            "premises": ["Student A has completed 118 credits."],
            "options": [],
        },
    ]

    for payload in examples:
        result = predictor.predict_payload(payload)
        assert isinstance(result, list) and len(result) == 1
        row = result[0]
        missing = REQUIRED_KEYS - set(row)
        assert not missing, f"missing key(s): {sorted(missing)}"
        assert row["query_id"] == payload["query_id"]
        assert row["unit"] == ""
        assert row["explanation"]
        assert isinstance(row["premises_used"], list)
        if payload["options"]:
            assert row["answer"] in payload["options"]
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
