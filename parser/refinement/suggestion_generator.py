"""Generate human-reviewable rule/template suggestions from clusters."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List


def _priority(count: int, risk: str) -> str:
    if count >= 50 and risk == "low":
        return "P0"
    if count >= 20:
        return "P1"
    return "P2"


def _example_ids(cluster: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for example in cluster.get("examples", [])[:5]:
        ids.append(str(example.get("dataset_id") or example.get("row_index") or "unknown"))
    return ids


def _example_texts(cluster: Dict[str, Any]) -> List[str]:
    return [str(example.get("problem_text", "")) for example in cluster.get("examples", [])[:3]]


def _suggestion(
    index: int,
    cluster: Dict[str, Any],
    category: str,
    title: str,
    proposed_change: str,
    files_to_modify: List[str],
    tests_to_add: List[str],
    risk: str,
) -> Dict[str, Any]:
    count = int(cluster.get("count", 0))
    return {
        "suggestion_id": f"S{index:03d}",
        "priority": _priority(count, risk),
        "category": category,
        "title": title,
        "evidence": {
            "cluster_ids": [cluster["cluster_id"]],
            "count": count,
            "dominant_error_types": cluster.get("error_type_counts", {}),
            "example_ids": _example_ids(cluster),
            "example_texts": _example_texts(cluster),
        },
        "proposed_change": proposed_change,
        "files_to_modify": files_to_modify,
        "tests_to_add": tests_to_add,
        "risk": risk,
        "regression_checks": [
            "pytest",
            "Run targeted suggestion devset",
            "Run missing_target/missing_quantity/invalid_final_step devsets",
            "Run full Stage 0 dataset and compare summary metrics",
        ],
    }


def _cluster_suggestion(cluster: Dict[str, Any], index: int) -> Dict[str, Any]:
    primary_error = str(cluster.get("primary_error_type", "unknown"))
    phrases = cluster.get("top_question_phrases", {})
    patterns = cluster.get("top_missing_number_patterns", {})
    relation_types = cluster.get("relation_type_tuple", "")
    template_candidates = cluster.get("template_candidates", {})
    target = str(cluster.get("target_quantity", "null"))

    if primary_error == "missing_target" and phrases:
        phrase = next(iter(phrases))
        return _suggestion(
            index,
            cluster,
            "target_detector",
            f"Map repeated target phrase '{phrase}'",
            f"Add or refine target_detector.py mapping for phrase '{phrase}', choosing the target symbol conservatively from examples.",
            ["parser/target_detector.py"],
            [f"Add target detector tests for phrase '{phrase}' and representative examples."],
            "low",
        )

    if primary_error == "missing_quantity" and patterns:
        pattern = next(iter(patterns))
        category = "relation_extractor" if pattern in {"function", "equation", "uncertainty", "ratio", "percentage"} else "quantity_extractor"
        files = ["parser/rule_extractor.py"]
        if pattern in {"angle_degree", "scientific_notation"}:
            files.append("parser/unit_normalizer.py")
        return _suggestion(
            index,
            cluster,
            category,
            f"Extract repeated {pattern} pattern",
            f"Add deterministic extraction/normalization for {pattern} occurrences and preserve source_text in relations or known_quantities.",
            files,
            [f"Add rule_extractor tests covering {pattern} examples from this cluster."],
            "low" if pattern not in {"equation", "function"} else "medium",
        )

    if primary_error == "invalid_final_step" and template_candidates:
        candidate = next(iter(template_candidates))
        return _suggestion(
            index,
            cluster,
            "template_fallback",
            f"Add executable template hook for {candidate}",
            f"Quantities/relations are present but step_plan is empty. Add a real formula_application/calculation template for {candidate}; do not create conclusion-only PASS traces.",
            ["parser/template_fallback.py"],
            [f"Add end-to-end parser test for {candidate} and verify final step is a conclusion after an executable step."],
            "medium",
        )

    if "function" in relation_types or "equation" in relation_types:
        return _suggestion(
            index,
            cluster,
            "template_fallback",
            f"Consume extracted relation types for target {target}",
            "Add a conservative symbolic setup template that consumes existing function/equation relations without solving them in Stage 0.",
            ["parser/template_fallback.py"],
            ["Add relation-driven template tests with extracted function/equation relations."],
            "medium",
        )

    return _suggestion(
        index,
        cluster,
        "verifier",
        f"Review residual cluster for target {target}",
        "Inspect examples to decide whether verifier coverage, target detection, extraction, or template coverage is the true blocker.",
        ["parser/parse_verifier.py"],
        ["Add a regression test after human diagnosis."],
        "high",
    )


def generate_suggestions(clusters: List[Dict[str, Any]], total_failures: int) -> Dict[str, Any]:
    suggestions: List[Dict[str, Any]] = []
    for index, cluster in enumerate(clusters, start=1):
        suggestions.append(_cluster_suggestion(cluster, index))
    top_errors: Counter[str] = Counter()
    top_question_types: Counter[str] = Counter()
    for cluster in clusters:
        top_errors.update(cluster.get("error_type_counts", {}))
        top_question_types[str(cluster.get("question_type", "unknown"))] += int(cluster.get("count", 0))
    return {
        "summary": {
            "total_failures": total_failures,
            "total_clusters": len(clusters),
            "top_error_types": dict(top_errors.most_common()),
            "top_question_types": dict(top_question_types.most_common()),
        },
        "suggestions": suggestions,
    }


def suggestions_markdown(data: Dict[str, Any]) -> str:
    lines = ["# Rule / Template Suggestions", ""]
    for suggestion in data.get("suggestions", []):
        evidence = suggestion.get("evidence", {})
        lines.extend(
            [
                f"## {suggestion['suggestion_id']} [{suggestion['priority']}] {suggestion['title']}",
                "",
                f"- Category: {suggestion['category']}",
                f"- Count: {evidence.get('count', 0)}",
                f"- Risk: {suggestion['risk']}",
                f"- Why it matters: dominant errors {evidence.get('dominant_error_types', {})}",
                f"- Proposed change: {suggestion['proposed_change']}",
                f"- Files: {', '.join(suggestion.get('files_to_modify', []))}",
                f"- Tests: {'; '.join(suggestion.get('tests_to_add', []))}",
                "",
            ]
        )
    return "\n".join(lines)


def clusters_markdown(clusters: List[Dict[str, Any]]) -> str:
    lines = ["# Top Failure Clusters", ""]
    for rank, cluster in enumerate(clusters, start=1):
        lines.extend(
            [
                f"## {rank}. {cluster['count']} records",
                "",
                f"- Cluster: `{cluster['cluster_id']}`",
                f"- Errors: {cluster.get('error_type_counts', {})}",
                f"- Target: {cluster.get('target_quantity')} ({cluster.get('target_dim')})",
                f"- Known dimensions: `{cluster.get('known_dim_tuple')}`",
                f"- Relation types: `{cluster.get('relation_type_tuple')}`",
                f"- Template used: {cluster.get('template_used')}",
                "",
                "Examples:",
            ]
        )
        for example in cluster.get("examples", [])[:5]:
            lines.append(f"- {example.get('dataset_id')} #{example.get('row_index')}: {example.get('problem_text')}")
        lines.append("")
    return "\n".join(lines)

