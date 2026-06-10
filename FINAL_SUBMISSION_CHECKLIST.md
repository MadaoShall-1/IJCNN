# Final Submission Checklist

## Required Checks

- [ ] Code compiles
- [ ] `/predict` is publicly reachable during the registered grading slot
- [ ] `/predict` accepts the unified input fields: `query_id`, `type`, `query`, `premises`, `options`
- [ ] `/predict` routes both `type1` and `type2` through the same endpoint
- [ ] `/predict` returns a JSON list with one result object, even for a single query
- [ ] Output always includes `query_id`, `answer`, `unit`, `explanation`, `premises_used`, `reasoning`
- [ ] Response `query_id` exactly matches request `query_id`
- [ ] Explanation is non-empty and not null
- [ ] Type 1 choice answers exactly match one of the provided `options`
- [ ] Type 1 `premises_used` contains 0-based indices from the request `premises`
- [ ] Type 2 `answer` contains only the numerical value
- [ ] Type 2 `unit` is ASCII (`ohm`, `uF`, `nC`, `V/m`, etc.)
- [ ] Type 2 `premises_used` is `[]`
- [ ] Each request responds within 60 seconds; evaluation has no retries
- [ ] vLLM `/v1/models` endpoint is publicly reachable during the grading slot
- [ ] Deterministic run completes
- [ ] Hybrid run completes
- [ ] Full dataset run completes
- [ ] `api_results.jsonl` generated
- [ ] `api_summary.json` generated
- [ ] `error_analysis` generated
- [ ] `false_pass_report.md` generated
- [ ] README contains run commands
- [ ] No hardcoded absolute local paths
- [ ] No GPT/Claude/Gemini dependency
- [ ] Model is open-source and <=8B
- [ ] Final output format matches EXACT 2026 Submission Guide
- [ ] Solver main logic frozen

## Submission Package

- [ ] ZIP filename is `<team_name>.zip`
- [ ] ZIP contains `solution.pdf`
- [ ] ZIP contains `source_code.zip`
- [ ] ZIP contains `urls.txt`
- [ ] ZIP contains `notation_mapping.csv`
- [ ] `urls.txt` lists the prediction URL and every vLLM `/v1/models` URL
- [ ] `solution.pdf` is one page
- [ ] `solution.pdf` lists datasets used, source/origin, sample count, and sample entries
- [ ] `solution.pdf` summarizes the pipeline approach
- [ ] `solution.pdf` lists every LLM and parameter count
- [ ] `solution.pdf` proves the active loaded/running LLM total stays within the 8B-class limit
- [ ] `notation_mapping.csv` is filled in for any notation that differs from the canonical form
- [ ] Registered a one-hour grading slot
- [ ] Submission is completed before June 12, 2026

## Reproduction Commands

```powershell
python scripts\final_submission_check.py
```

Run syntax checks directly:

```powershell
python scripts\final_submission_check.py --run-syntax
```

Run deterministic test:

```powershell
python scripts\run_api_on_dataset.py `
  --dataset Dataset\Physics_Problems_Text_Only_test.csv `
  --output-dir outputs\final_test_deterministic `
  --query-type type2 `
  --type2-solver-mode deterministic
```

Run hybrid test:

```powershell
$env:DSPY_MODEL="openai/qwen3-8b-awq"
$env:DSPY_API_BASE="http://localhost:8002/v1"

python scripts\run_api_on_dataset.py `
  --dataset Dataset\Physics_Problems_Text_Only_test.csv `
  --output-dir outputs\final_test_hybrid `
  --query-type type2 `
  --type2-solver-mode hybrid
```

Run full hybrid:

```powershell
python scripts\run_api_on_dataset.py `
  --dataset Dataset\Physics_Problems_Text_Only.csv `
  --output-dir outputs\final_full_hybrid `
  --query-type type2 `
  --type2-solver-mode hybrid
```

Analyze full run:

```powershell
python scripts\analyze_api_results.py `
  --results outputs\final_full_hybrid\api_results.jsonl `
  --output-dir outputs\final_full_hybrid\error_analysis `
  --sample-limit 50
```
