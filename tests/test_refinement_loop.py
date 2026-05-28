import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parser.refinement.failure_clusterer import cluster_failures, cluster_key, infer_target_dimension, mine_patterns
from parser.refinement.prompt_builder import build_codex_prompt, build_devsets
from parser.refinement.record_loader import load_jsonl, normalize_record
from parser.refinement.suggestion_generator import generate_suggestions


def record(text="Find the current.", target="I", errors=None, known=None, relations=None, status="FAIL"):
    return normalize_record(
        {
            "dataset_id": "A1",
            "row_index": 1,
            "question": text,
            "parse": {
                "problem_text": text,
                "question_type": "numeric_calc",
                "domains": ["electricity"],
                "sub_domains": ["circuits"],
                "known_quantities": known or {"V": {"dimension": "voltage"}, "R": {"dimension": "resistance"}},
                "conditions": [],
                "relations": relations or [],
                "unknown_quantity": target,
                "unknown_unit": "A",
                "step_plan": [],
                "metadata": {
                    "verifier_status": status,
                    "verifier_errors": errors or [{"description": "No step_plan was produced."}],
                    "used_template_fallback": False,
                },
            },
        }
    )


def test_record_normalization_variants(tmp_path):
    raw = {
        "id": "X",
        "parse_object": {
            "problem_text": "Find voltage.",
            "metadata": {"verifier_errors": [{"description": "target missing"}]},
        },
    }
    normalized = normalize_record(raw)
    assert normalized["dataset_id"] == "X"
    assert normalized["errors"][0]["error_type"] == "missing_target"

    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(raw) + "\nnot-json\n", encoding="utf-8")
    loaded, malformed = load_jsonl(path)
    assert len(loaded) == 1
    assert malformed == 1


def test_cluster_key_generation():
    item = record(relations=[{"type": "equation"}, {"type": "function"}])
    fields, cid = cluster_key(item)
    assert fields["known_dim_tuple"] == "resistance|voltage"
    assert fields["target_dim"] == "current"
    assert fields["relation_type_tuple"] == "equation|function"
    assert cid == cluster_key(item)[1]
    assert infer_target_dimension("F_net") == "force"


def test_pattern_mining():
    text = "I(t)=2e-6 A, angle 60°, q1 = 4q2, 1/2, 25%, x +/- 1, downstream."
    patterns = mine_patterns(text)
    assert patterns["function"]
    assert patterns["angle_degree"]
    assert patterns["scientific_notation"]
    assert patterns["fraction_numeric"]
    assert patterns["percentage"]
    assert patterns["uncertainty"]
    assert patterns["ratio"]


def test_suggestion_generation_categories():
    missing_target = record("Calculate the mystery power.", target=None, errors=[{"error_type": "missing_target"}])
    missing_quantity = record("The angle is 60°.", errors=[{"error_type": "missing_quantity"}], known={})
    empty_step = record("A circuit has 5 V and 2 ohm. Find current.", errors=[{"error_type": "invalid_final_step"}])
    relation = record("I(t)=2 A. Find charge.", target="Q", errors=[{"error_type": "invalid_final_step"}], relations=[{"type": "function"}], known={})
    clusters = cluster_failures([missing_target, missing_quantity, empty_step, relation], top_k=10)
    suggestions = generate_suggestions(clusters, total_failures=4)["suggestions"]
    categories = {suggestion["category"] for suggestion in suggestions}
    assert "target_detector" in categories
    assert "quantity_extractor" in categories
    assert "template_fallback" in categories


def test_prompt_generation():
    suggestion = {
        "suggestion_id": "S001",
        "priority": "P0",
        "category": "target_detector",
        "title": "Map target phrase",
        "evidence": {"count": 3, "cluster_ids": ["c"], "dominant_error_types": {}, "example_texts": ["Find q."]},
        "proposed_change": "Add mapping.",
        "files_to_modify": ["parser/target_detector.py"],
        "tests_to_add": ["Add target test."],
    }
    prompt = build_codex_prompt([suggestion])
    assert "Find q." in prompt
    assert "parser/target_detector.py" in prompt
    assert "Do not weaken verifier globally" in prompt


def test_devset_builder_deterministic(tmp_path):
    fail = record("Find current.", errors=[{"error_type": "missing_target"}])
    fail["dataset_id"] = "B"
    fail["row_index"] = 2
    passed = record("Find voltage.", status="PASS", errors=[])
    passed["dataset_id"] = "A"
    passed["row_index"] = 1
    clusters = cluster_failures([fail], top_k=10)
    suggestions = generate_suggestions(clusters, total_failures=1)
    manifest = build_devsets(tmp_path, [fail], [passed, fail], suggestions, clusters)
    assert "regression_pass_100.jsonl" in manifest["devsets"]
    suggestion_files = [name for name in manifest["devsets"] if name.startswith("suggestion_")]
    assert suggestion_files
    assert (tmp_path / suggestion_files[0]).exists()

