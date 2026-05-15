"""Robust JSONL loading and normalization for Stage 0 refinement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def infer_error_type(error: Dict[str, Any]) -> str:
    explicit = error.get("error_type")
    if explicit:
        return str(explicit)
    text = " ".join(str(error.get(key, "")) for key in ("description", "repair_hint", "message", "error")).lower()
    if "mismatch" in text and "target" in text:
        return "target_mismatch"
    if "target" in text:
        return "missing_target"
    if "quantity" in text or "numeric" in text:
        return "missing_quantity"
    if "final step" in text or "conclusion" in text or "step_plan" in text:
        return "invalid_final_step"
    if "confidence" in text:
        return "low_confidence"
    if "dependency" in text:
        return "invalid_dependency"
    return "unknown"


def normalize_error(error: Any) -> Dict[str, str]:
    data = as_dict(error)
    description = str(data.get("description") or data.get("message") or data.get("error") or "")
    repair_hint = str(data.get("repair_hint") or data.get("hint") or "")
    return {
        "error_type": infer_error_type(data),
        "description": description,
        "repair_hint": repair_hint,
    }


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_errors(record: Dict[str, Any], parse: Dict[str, Any]) -> List[Dict[str, str]]:
    metadata = as_dict(parse.get("metadata"))
    record_metadata = as_dict(record.get("metadata"))
    verifier_result = as_dict(record.get("verifier_result"))
    raw_errors = (
        metadata.get("verifier_errors")
        or record_metadata.get("verifier_errors")
        or verifier_result.get("errors")
        or []
    )
    return [normalize_error(error) for error in as_list(raw_errors)]


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    parse = as_dict(record.get("parse") or record.get("parse_object"))
    metadata = as_dict(parse.get("metadata"))
    problem_text = str(parse.get("problem_text") or record.get("question") or record.get("problem_text") or "")
    errors = extract_errors(record, parse)
    status = str(metadata.get("verifier_status") or record.get("status") or ("FAIL" if errors else "UNKNOWN"))
    return {
        "dataset_id": record.get("dataset_id") or record.get("id"),
        "row_index": _coerce_int(record.get("row_index") or record.get("index")),
        "problem_text": problem_text,
        "answer": record.get("answer"),
        "unit": record.get("unit"),
        "parse": parse,
        "status": status,
        "errors": errors,
        "question_type": str(parse.get("question_type") or "unknown"),
        "known_quantities": as_dict(parse.get("known_quantities")),
        "conditions": [str(item) for item in as_list(parse.get("conditions"))],
        "relations": [as_dict(item) for item in as_list(parse.get("relations"))],
        "unknown_quantity": parse.get("unknown_quantity"),
        "unknown_unit": parse.get("unknown_unit"),
        "step_plan": [as_dict(item) for item in as_list(parse.get("step_plan"))],
        "metadata": metadata,
    }


def load_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    malformed = 0
    if not path.exists():
        return records, malformed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(raw, dict):
                malformed += 1
                continue
            records.append(normalize_record(raw))
    return records, malformed


def dump_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sort_key(record: Dict[str, Any]) -> Tuple[str, int]:
    return (str(record.get("dataset_id") or ""), int(record.get("row_index") or 0))

