"""Run the full API pipeline over a CSV dataset and write JSONL artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api  # noqa: E402
import type1.pipeline as type1_pipeline  # noqa: E402
from config import SolverConfig  # noqa: E402
from router import detect_query_type  # noqa: E402


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _maybe_json(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] not in "[{\"" and stripped.lower() not in {"true", "false", "null"}:
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _row_to_payload(row: Dict[str, str], id_field: str, text_field: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            continue
        payload[key] = _maybe_json(value)

    if id_field in row and row[id_field]:
        payload["id"] = row[id_field]
    if text_field in row and row[text_field]:
        payload["question"] = row[text_field]
    return payload


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _print_progress(current: int, total: int, start_time: float, counts: Counter[str]) -> None:
    elapsed = time.monotonic() - start_time
    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = (total - current) / rate if rate > 0 else 0.0
    pct = 100.0 * current / total if total else 100.0
    line = (
        f"  [{current:>5}/{total}] {pct:5.1f}% | "
        f"PASS={counts.get('PASS', 0)} FAIL={counts.get('FAIL', 0)} "
        f"ERROR={counts.get('ERROR', 0)} | "
        f"elapsed {_format_eta(elapsed)} | ETA {_format_eta(remaining)} | "
        f"{rate:.2f} rows/s"
    )
    print("\r" + line + " " * 6, end="", flush=True)


def _apply_runtime_overrides(
    cfg: SolverConfig,
    beam_n: int | None,
    step_retry_limit: int | None,
    trace_budget: int | None,
) -> SolverConfig:
    if beam_n is not None:
        cfg.beam_n = beam_n
    if step_retry_limit is not None:
        cfg.step_retry_limit = step_retry_limit
    if trace_budget is not None:
        cfg.trace_budget = trace_budget
    return cfg


def _result_status(result: Dict[str, Any]) -> str:
    if result.get("error"):
        return "ERROR"
    if str(result.get("trace_status", "PASS")).upper() == "FAIL":
        return "FAIL"
    return "PASS"


def _final_answer_check(result: Dict[str, Any]) -> Dict[str, Any]:
    raw = result.get("final_answer_check") or result.get("answer_level_verification") or {}
    verdict = raw.get("verdict") or raw.get("final_answer_verdict") or result.get("final_answer_verdict") or "UNKNOWN"
    return {
        "verdict": verdict,
        "error_type": raw.get("error_type") or raw.get("final_answer_error_type") or result.get("final_answer_error_type"),
        "repair_attempted": bool(raw.get("repair_attempted") or raw.get("numeric_repair_attempted") or result.get("numeric_repair_attempted")),
        "repair_accepted": bool(raw.get("repair_accepted") or raw.get("numeric_repair_accepted") or result.get("numeric_repair_accepted")),
        "notes": raw.get("notes") or raw.get("repair_hint") or result.get("repair_hint"),
    }


def _configure_api(cfg: SolverConfig, load_type1: bool = True) -> None:
    api._config = cfg
    api._load_models(cfg, load_type1=load_type1)


def _run_one_payload(
    payload: Dict[str, Any],
    query_type: str,
    cfg: SolverConfig,
    t_start: float,
) -> Dict[str, Any]:
    if query_type == "type1":
        return type1_pipeline.run(payload=payload, config=cfg, solver=api._type1_solver)
    return api._run_type2(payload, cfg, t_start)


def run_api_dataset(
    dataset_path: Path,
    output_dir: Path,
    text_field: str,
    id_field: str,
    limit: int | None,
    start_row: int,
    resume: bool,
    force_query_type: str | None,
    use_stage0_cache: bool | None,
    type2_solver_mode: str,
    beam_n: int | None,
    step_retry_limit: int | None,
    trace_budget: int | None,
) -> Dict[str, Any]:
    rows = _load_csv(dataset_path)
    if start_row < 1:
        raise ValueError("--start-row must be >= 1")
    if start_row > 1:
        rows = rows[start_row - 1 :]
    if limit is not None:
        rows = rows[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "api_results.jsonl"
    failures_path = output_dir / "api_failures.jsonl"
    summary_path = output_dir / "api_summary.json"

    cfg = SolverConfig()
    if type2_solver_mode == "deterministic":
        cfg.dspy_model = ""
        cfg.dspy_api_base = ""
    elif type2_solver_mode == "dspy" and not cfg.dspy_model:
        raise ValueError("--type2-solver-mode dspy requires DSPY_MODEL to be set")
    elif type2_solver_mode == "hybrid" and not cfg.dspy_model:
        raise ValueError("--type2-solver-mode hybrid requires DSPY_MODEL to be set")
    _apply_runtime_overrides(cfg, beam_n, step_retry_limit, trace_budget)

    deterministic_cfg: SolverConfig | None = None
    llm_cfg: SolverConfig | None = None
    loaded_mode: str | None = None
    if type2_solver_mode == "hybrid":
        deterministic_cfg = _apply_runtime_overrides(
            SolverConfig(),
            beam_n,
            step_retry_limit,
            trace_budget,
        )
        deterministic_cfg.dspy_model = ""
        deterministic_cfg.dspy_api_base = ""
        llm_cfg = cfg
        _configure_api(deterministic_cfg, load_type1=force_query_type != "type2")
        loaded_mode = "deterministic"
    else:
        _configure_api(cfg, load_type1=force_query_type != "type2")
        loaded_mode = type2_solver_mode

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    query_type_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    completed_rows: set[int] = set()

    if resume and results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_index = record.get("row_index")
                if isinstance(row_index, int):
                    completed_rows.add(row_index)

    start_time = time.monotonic()
    total = len(rows)
    run_total = total - len([i for i in completed_rows if start_row <= i < start_row + total])
    print(f"\nStarting full API pipeline run on {total} rows...")
    if resume and completed_rows:
        print(f"Resuming: {len(completed_rows)} existing rows will be skipped.")
    print()

    file_mode = "a" if resume else "w"
    results_handle = results_path.open(file_mode, encoding="utf-8")
    failures_handle = failures_path.open(file_mode, encoding="utf-8")

    try:
        processed = 0
        for index, row in enumerate(rows, start=start_row):
            if resume and index in completed_rows:
                continue
            processed += 1
            t_start = time.monotonic()
            item_id = row.get(id_field) or str(index)
            payload = _row_to_payload(row, id_field=id_field, text_field=text_field)
            if force_query_type:
                payload["query_type"] = force_query_type
            if use_stage0_cache is not None:
                payload["use_stage0_cache"] = use_stage0_cache

            query_type = detect_query_type(payload)
            query_type_counts[query_type] += 1

            try:
                if type2_solver_mode == "hybrid" and query_type == "type2":
                    assert deterministic_cfg is not None
                    assert llm_cfg is not None
                    if loaded_mode != "deterministic":
                        _configure_api(deterministic_cfg, load_type1=False)
                        loaded_mode = "deterministic"
                    deterministic_result = _run_one_payload(
                        payload=payload,
                        query_type=query_type,
                        cfg=deterministic_cfg,
                        t_start=t_start,
                    )
                    deterministic_status = _result_status(deterministic_result)
                    if deterministic_status == "PASS":
                        result = deterministic_result
                        result["hybrid_source"] = "deterministic"
                    else:
                        print(
                            f"\n  row {index} ({item_id}) deterministic={deterministic_status}; "
                            "calling LLM fallback...",
                            flush=True,
                        )
                        if loaded_mode != "dspy":
                            _configure_api(llm_cfg, load_type1=False)
                            loaded_mode = "dspy"
                        # FWS-style: inject deterministic trace context into
                        # the payload so the LLM can do local repair instead
                        # of solving from scratch.
                        repair_payload = dict(payload)
                        det_steps = deterministic_result.get("steps") or []
                        det_answer = deterministic_result.get("answer", "")
                        correct_prefix = [
                            s for s in det_steps
                            if str(s.get("status", "")).upper() in ("OK", "PASS", "REPAIRED")
                        ]
                        first_wrong = next(
                            (s for s in det_steps if str(s.get("status", "")).upper() not in ("OK", "PASS", "REPAIRED")),
                            None,
                        )
                        if correct_prefix:
                            prefix_summary = "; ".join(
                                f"{s.get('step_id')}: {s.get('goal', '')[:60]} -> {s.get('intermediate_answer', '')}"
                                for s in correct_prefix[-3:]
                            )
                            repair_context = (
                                "\n\n[LOCAL REPAIR CONTEXT]\n"
                                "The deterministic solver produced a valid prefix. "
                                "Preserve these correct intermediate results:\n"
                                f"{prefix_summary}\n"
                            )
                            if first_wrong:
                                repair_context += (
                                    f"First wrong step: {first_wrong.get('step_id')} "
                                    f"({first_wrong.get('goal', '')})\n"
                                    "Repair only this step. Do not change the quantities above.\n"
                                )
                            if det_answer:
                                repair_context += (
                                    f"Deterministic answer (may be a valid intermediate): {det_answer}\n"
                                )
                            existing_q = str(repair_payload.get("question", ""))
                            repair_payload["question"] = existing_q + repair_context
                        result = _run_one_payload(
                            payload=repair_payload,
                            query_type=query_type,
                            cfg=llm_cfg,
                            t_start=t_start,
                        )
                        result["hybrid_source"] = "llm_fallback"
                        result["hybrid_deterministic_status"] = deterministic_status
                        result["hybrid_deterministic_trace_status"] = deterministic_result.get("trace_status")
                        result["hybrid_deterministic_answer"] = deterministic_result.get("answer")
                        result["hybrid_deterministic_error"] = deterministic_result.get("error")
                        if _result_status(result) != "PASS" and deterministic_result.get("answer"):
                            result["hybrid_preserved_deterministic_answer"] = deterministic_result.get("answer")
                            if not result.get("answer"):
                                result["answer"] = deterministic_result.get("answer")
                else:
                    result = _run_one_payload(
                        payload=payload,
                        query_type=query_type,
                        cfg=cfg,
                        t_start=t_start,
                    )
                result["query_type"] = query_type
                result["latency_seconds"] = round(time.monotonic() - t_start, 3)
            except Exception as exc:  # noqa: BLE001
                result = {
                    "problem_id": item_id,
                    "query_type": query_type,
                    "answer": "",
                    "confidence": 0.0,
                    "error": str(exc),
                    "latency_seconds": round(time.monotonic() - t_start, 3),
                }

            status = _result_status(result)
            if status == "ERROR":
                error_counts[str(result.get("error"))] += 1
            status_counts[status] += 1

            record = {
                "dataset_id": item_id,
                "row_index": index,
                "payload": payload,
                "result": result,
                "status": status,
                "final_answer_check": _final_answer_check(result),
            }
            results.append(record)
            results_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            results_handle.flush()

            if status != "PASS":
                failures.append(record)
                failures_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                failures_handle.flush()

            if processed % 5 == 0 or status != "PASS" or processed == run_total:
                _print_progress(processed, run_total, start_time, status_counts)
    finally:
        results_handle.close()
        failures_handle.close()

    print()
    elapsed = time.monotonic() - start_time
    print(f"Run complete in {_format_eta(elapsed)}.")

    summary: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "total": run_total,
        "start_row": start_row,
        "resume": resume,
        "status_counts": dict(status_counts),
        "query_type_counts": dict(query_type_counts),
        "type2_solver_mode": type2_solver_mode if type2_solver_mode == "hybrid" else api._type2_solver_mode,
        "dspy_model": cfg.dspy_model,
        "dspy_api_base": cfg.dspy_api_base,
        "dspy_max_tokens": cfg.dspy_max_tokens,
        "beam_n": cfg.beam_n,
        "step_retry_limit": cfg.step_retry_limit,
        "trace_budget": cfg.trace_budget,
        "error_counts": dict(error_counts.most_common(20)),
        "pass_rate": round(status_counts.get("PASS", 0) / total, 4) if total else 0.0,
        "elapsed_seconds": round(elapsed, 3),
        "outputs": {
            "results": str(results_path),
            "failures": str(failures_path),
            "summary": str(summary_path),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full API pipeline over a CSV dataset.")
    parser.add_argument("--dataset", default="Dataset/Physics_Problems_Text_Only.csv")
    parser.add_argument("--output-dir", default="outputs/api_dataset")
    parser.add_argument("--text-field", default="question")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--query-type", choices=["type1", "type2"], default=None)
    parser.add_argument("--use-stage0-cache", action="store_true", default=None)
    parser.add_argument("--no-stage0-cache", action="store_false", dest="use_stage0_cache")
    parser.add_argument("--beam-n", type=int, default=None)
    parser.add_argument("--step-retry-limit", type=int, default=None)
    parser.add_argument("--trace-budget", type=int, default=None)
    parser.add_argument(
        "--type2-solver-mode",
        choices=["auto", "deterministic", "dspy", "hybrid"],
        default="auto",
        help=(
            "auto uses DSPy when DSPY_MODEL is set; deterministic disables the "
            "Type 2 LM path; dspy requires DSPY_MODEL; hybrid runs deterministic "
            "first and calls DSPy only for deterministic failures."
        ),
    )
    args = parser.parse_args()

    summary = run_api_dataset(
        dataset_path=_resolve_path(args.dataset),
        output_dir=_resolve_path(args.output_dir),
        text_field=args.text_field,
        id_field=args.id_field,
        limit=args.limit,
        start_row=args.start_row,
        resume=args.resume,
        force_query_type=args.query_type,
        use_stage0_cache=args.use_stage0_cache,
        type2_solver_mode=args.type2_solver_mode,
        beam_n=args.beam_n,
        step_retry_limit=args.step_retry_limit,
        trace_budget=args.trace_budget,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
