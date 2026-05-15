"""Codex patch prompt and devset helpers for Stage 0 refinement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .record_loader import dump_jsonl, sort_key


def build_codex_prompt(suggestions: List[Dict[str, Any]]) -> str:
    title = " + ".join(suggestion["title"] for suggestion in suggestions)
    lines = [
        f"Implement Stage 0 refinement: {title}",
        "",
        "Context:",
        "We have a deterministic-first Stage 0 physics parser. The parser is static at inference time; this task is to implement a human-reviewed refinement patch from data-driven failure analysis.",
        "",
        "Evidence:",
    ]
    for suggestion in suggestions:
        evidence = suggestion.get("evidence", {})
        lines.extend(
            [
                f"- {suggestion['suggestion_id']} [{suggestion['priority']}] {suggestion['title']}",
                f"  Category: {suggestion['category']}",
                f"  Count: {evidence.get('count')}",
                f"  Cluster ids: {evidence.get('cluster_ids')}",
                f"  Dominant errors: {evidence.get('dominant_error_types')}",
            ]
        )
        for text in evidence.get("example_texts", [])[:3]:
            lines.append(f"  Example: {text}")
    lines.extend(["", "Required changes:"])
    for suggestion in suggestions:
        lines.append(f"- {suggestion['suggestion_id']}: {suggestion['proposed_change']}")
        lines.append(f"  Files to modify: {', '.join(suggestion.get('files_to_modify', []))}")
    lines.extend(["", "Tests:"])
    for suggestion in suggestions:
        for test in suggestion.get("tests_to_add", []):
            lines.append(f"- {test}")
    lines.extend(
        [
            "",
            "Validation:",
            "pytest",
            "python scripts/run_stage0_on_dataset.py --output-dir outputs/stage0_next",
            "",
            "Design constraints:",
            "- Do not weaken verifier globally.",
            "- Do not create conclusion-only PASS cases.",
            "- Keep outputs JSON-serializable.",
            "- Preserve source_text in evidence and parse relations.",
            "- Prefer deterministic rules and focused tests.",
            "- Do not use proprietary APIs.",
        ]
    )
    return "\n".join(lines)


def select_suggestions(data: Dict[str, Any], suggestion_id: str | None = None, top_n: int | None = None) -> List[Dict[str, Any]]:
    suggestions = list(data.get("suggestions", []))
    if suggestion_id:
        selected = [suggestion for suggestion in suggestions if suggestion.get("suggestion_id") == suggestion_id]
        if not selected:
            raise ValueError(f"Suggestion id {suggestion_id} not found")
        return selected
    if top_n:
        return suggestions[:top_n]
    return suggestions[:1]


def write_devset(path: Path, records: Iterable[Dict[str, Any]], limit: int) -> int:
    selected = sorted(records, key=sort_key)[:limit]
    dump_jsonl(path, selected)
    return len(selected)


def build_devsets(
    output_dir: Path,
    failures: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    suggestions: Dict[str, Any],
    clusters: List[Dict[str, Any]],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cluster_examples: Dict[str, List[str]] = {
        cluster["cluster_id"]: [str(example.get("dataset_id")) for example in cluster.get("examples", [])]
        for cluster in clusters
    }
    by_id = {str(record.get("dataset_id")): record for record in failures + results}
    manifest: Dict[str, Any] = {"devsets": {}}

    for suggestion in suggestions.get("suggestions", []):
        sid = suggestion["suggestion_id"]
        ids: List[str] = []
        for cid in suggestion.get("evidence", {}).get("cluster_ids", []):
            ids.extend(cluster_examples.get(cid, []))
        records = [by_id[item_id] for item_id in ids if item_id in by_id]
        path = output_dir / f"suggestion_{sid}.jsonl"
        manifest["devsets"][path.name] = write_devset(path, records, 50)

    p0_records: List[Dict[str, Any]] = []
    for suggestion in suggestions.get("suggestions", []):
        if suggestion.get("priority") == "P0":
            ids = suggestion.get("evidence", {}).get("example_ids", [])
            p0_records.extend(by_id[item_id] for item_id in ids if item_id in by_id)
    manifest["devsets"]["top_p0_mixed.jsonl"] = write_devset(output_dir / "top_p0_mixed.jsonl", p0_records, 100)

    def has_error(record: Dict[str, Any], error_type: str) -> bool:
        return any(error.get("error_type") == error_type for error in record.get("errors", []))

    manifest["devsets"]["missing_target_100.jsonl"] = write_devset(output_dir / "missing_target_100.jsonl", [r for r in failures if has_error(r, "missing_target")], 100)
    manifest["devsets"]["missing_quantity_100.jsonl"] = write_devset(output_dir / "missing_quantity_100.jsonl", [r for r in failures if has_error(r, "missing_quantity")], 100)
    manifest["devsets"]["invalid_final_step_100.jsonl"] = write_devset(output_dir / "invalid_final_step_100.jsonl", [r for r in failures if has_error(r, "invalid_final_step")], 100)
    manifest["devsets"]["regression_pass_100.jsonl"] = write_devset(output_dir / "regression_pass_100.jsonl", [r for r in results if r.get("status") == "PASS"], 100)
    manifest["devsets"]["regression_relation_100.jsonl"] = write_devset(output_dir / "regression_relation_100.jsonl", [r for r in results if r.get("relations")], 100)
    manifest["devsets"]["regression_template_100.jsonl"] = write_devset(output_dir / "regression_template_100.jsonl", [r for r in results if (r.get("metadata") or {}).get("used_template_fallback")], 100)

    (output_dir / "devset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest

