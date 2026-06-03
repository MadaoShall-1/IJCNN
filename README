# Type2 Physics Candidate Pipeline — Setup, Current Architecture, and Next Improvements

## 1. Project Status Summary

This document summarizes the current Type2 physics QA pipeline after the migration from a single parser-driven architecture to a candidate-based architecture.

The current pipeline is already functional end-to-end:

```text
raw physics problem
→ Stage-0 parser
→ adapter
→ candidate generator
→ candidate verifier
→ rule-based ranker
→ candidate pipeline wrapper
→ numeric executor
→ answer sanity checker
→ final answer
```

The pipeline has completed the following major stages:

| Stage | Status | Main Outcome |
|---|---:|---|
| Step 1 Adapter | Done | Converts Stage-0 parser output into `Type2WorldModelInput`. |
| Step 2 Candidate Generator | Done | Produces multiple candidate step plans instead of relying on one parser plan. |
| Step 2.1 Candidate Gating | Done | Reduces cross-domain noise and irrelevant candidates. |
| Step 3 Candidate Verifier | Done | Scores each candidate with deterministic features. |
| Step 3.1 Verifier Calibration | Done | Reduces score saturation and improves penalties. |
| Step 4 Rule-based Ranker | Done | Selects the best candidate using verifier score plus domain rules. |
| Step 5 End-to-End Pipeline | Done | Connects adapter, generator, verifier, ranker, and final answer builder. |
| Step 6 Numeric Executor | Done | Executes selected symbolic step plans into numeric answers. |
| Step 6.1 Executor Alias Calibration | Done | Improves variable aliases, chained equality, and target writeback. |
| Step 6.2 Role-aware Coulomb Executor | Done | Adds target/source point reasoning for Coulomb force scenes. |
| Step 6.3 Geometry-aware Composition | Done | Adds law-of-cosines composition for safe vector magnitudes. |
| Step 6.4 Full Dataset Validation | Done | Adds full-dataset diagnostics and failure sampling. |
| Step 6.5 ROI Formula Dispatch Expansion | Done | Adds high-frequency formulas such as electric field, Ohm/power, parallel plate, LC/RLC. |
| Step 7 Sanity Checker | Done | Adds risk-aware final answer gating. |
| Step 7.1 Sanity Calibration | Done | Reduces over-conservative HIGH-risk classification. |
| Step 8 Failure Clustering | Done | Groups remaining symbolic/fail rows into actionable clusters. |
| Step 9 Low-risk Executor Expansion | Done | Adds LC/RLC, magnetic/flux/EMF, capacitor inverse coverage. |
| Step 10 LLM Fallback Canonicalization | Implemented, but live integration bug remains | Executor backstop works, but canonicalized candidate path did not fire in the real LLM run. |

Current best deterministic non-LLM full-dataset result after Step 9:

```text
evaluated: 1352
numeric_answer_count: 990
symbolic_trace_count: 326
failed_count: 1
unsupported_dispatch_count: 68
```

Current real LLM Step 10 run result:

```text
evaluated: 1352
numeric_answer_count: 996
symbolic_trace_count: 321
failed_count: 0
unsupported_dispatch_count: 117
llm_fallback_selected_count: 70
llm_fallback_numeric_count: 9
llm_fallback_symbolic_count: 61
llm_fallback_canonicalized_candidate_count: 0
```

Interpretation:

- Step 9 is verified and stable.
- Step 10 partially works through executor-side loose formula aliasing.
- Step 10 candidate-level canonicalization is implemented but did not fire in the live LLM run.
- The next priority is to diagnose and fix the canonicalized-candidate integration path, not to add Transformer code or training.

---

## 2. Hard Constraints

Do not violate the following constraints:

1. Do not add Transformer code yet.
2. Do not train anything yet.
3. Do not make network calls.
4. Do not use unsafe `eval`.
5. Do not execute arbitrary LLM-generated expressions.
6. Do not change `parse_problem` behavior unless the change is strictly backward compatible.
7. Do not add cache inside parser, generator, executor, or the core pipeline.
8. If caching is needed, implement it only as an optional, off-by-default validation-layer wrapper.
9. The deterministic non-LLM path must remain stable:
   - Full-dataset validation without `--use-llm-fallback` should still produce:
     ```text
     numeric_answer_count = 990
     failed_count = 1
     ```
   - There should be zero `answer_type` diffs against the Step 9 baseline.
10. Treat any deterministic regression as a blocker.

---

## 3. Environment Setup

### 3.1 Repository Layout Assumption

The current Windows layout is assumed to be:

```text
E:\Education System\
├── Dataset\
│   └── Physics_Problems_Text_Only.csv
├── parser\
│   ├── main.py
│   ├── condition_extractor.py
│   ├── target_detector.py
│   ├── llm_fallback.py
│   └── ...
├── pipeline\
│   ├── type2_adapter.py
│   ├── type2_candidate_generator.py
│   ├── type2_candidate_verifier.py
│   ├── type2_candidate_ranker.py
│   ├── type2_candidate_pipeline.py
│   ├── type2_numeric_executor.py
│   ├── type2_answer_sanity_checker.py
│   ├── type2_llm_canonicalizer.py
│   └── ...
├── validate\
│   ├── validate_type2_full_dataset.py
│   └── ...
└── outputs\
```

The `parser/` package uses relative imports, while most `type2_*` modules import each other as top-level modules. Therefore both the root directory and the `pipeline` directory must be on `PYTHONPATH`.

### 3.2 PowerShell Setup

```powershell
cd "E:\Education System"
$env:PYTHONPATH="E:\Education System;E:\Education System\pipeline"
```

### 3.3 Basic Full Dataset Validation

```powershell
py validate\validate_type2_full_dataset.py `
  --input "Dataset\Physics_Problems_Text_Only.csv" `
  --output outputs\type2_full_dataset_validation.json `
  --save-jsonl outputs\type2_full_dataset_rows.jsonl `
  --max-candidates 8 `
  --compact
```

### 3.4 Strict Sanity Mode

```powershell
py validate\validate_type2_full_dataset.py `
  --input "Dataset\Physics_Problems_Text_Only.csv" `
  --output outputs\type2_full_dataset_validation_strict.json `
  --save-jsonl outputs\type2_full_dataset_rows_strict.jsonl `
  --max-candidates 8 `
  --compact `
  --downgrade-high-risk-numeric
```

### 3.5 LLM Fallback Mode

```powershell
py validate\validate_type2_full_dataset.py `
  --input "Dataset\Physics_Problems_Text_Only.csv" `
  --output outputs\type2_full_dataset_validation_llm.json `
  --save-jsonl outputs\type2_full_dataset_rows_llm.jsonl `
  --max-candidates 8 `
  --compact `
  --use-llm-fallback
```

### 3.6 LLM Fallback Model

The LLM fallback uses local Qwen3-8B GGUF through `llama-cpp-python`.

Expected model path:

```text
~/.cache/qwen3/Qwen3-8B-Q4_K_M.gguf
```

or set:

```powershell
$env:QWEN3_GGUF_PATH="path\to\Qwen3-8B-Q4_K_M.gguf"
```

If the model or `llama-cpp-python` is missing, the system silently runs in mock mode. In mock mode, no real LLM repair is applied and no meaningful `llm_fallback` behavior should be expected.

---

## 4. Current Architecture

## 4.1 Stage-0 Parser

Main entry:

```python
from parser.main import parse_problem
```

The Stage-0 parser extracts:

- domain / subdomain
- target
- target unit
- known quantities
- conditions
- initial symbolic step plan
- parser warnings and errors

Important parser-side modules include:

```text
parser/main.py
parser/condition_extractor.py
parser/target_detector.py
parser/question_type_classifier.py
parser/unit_normalizer.py
parser/llm_fallback.py
```

The parser is still a deterministic/static parser with optional LLM fallback. The LLM fallback is not the main reasoning engine. It is used only as a repair mechanism when parser output is incomplete.

---

## 4.2 Adapter

File:

```text
type2_adapter.py
```

Purpose:

```text
Stage-0 parser output
→ Type2WorldModelInput
```

The adapter normalizes parser output into a stable object used by the candidate pipeline. It keeps:

- problem text
- target
- target unit
- known quantities
- conditions
- original parser step plan
- diagnostics

The adapter should not perform heavy reasoning. Its role is structural conversion and lightweight normalization.

---

## 4.3 Candidate Generator

File:

```text
type2_candidate_generator.py
```

Purpose:

```text
one parser step_plan
→ multiple candidate step_plans
```

Candidate types include:

- legacy parser candidate
- deterministic formula variants
- geometry-specific candidates
- non-numeric boolean candidate
- skeleton placeholder
- raw LLM fallback candidate
- intended Step 10 candidate: `llm_fallback_canonicalized`

The generator should preserve the original parser candidate and add alternatives rather than replacing the deterministic path.

Current important issue:

`llm_fallback_canonicalized` candidates are expected after Step 10, but the live run shows:

```text
llm_fallback_canonicalized_candidate_count: 0
```

This strongly suggests the updated generator hook was not loaded, or it was not appending canonicalized candidates.

---

## 4.4 Candidate Verifier

File:

```text
type2_candidate_verifier.py
```

Purpose:

```text
candidate step_plan
→ deterministic feature vector
→ verifier status and score
```

The verifier evaluates whether the candidate:

- matches the target
- has required inputs
- matches the domain
- has reasonable formula structure
- avoids obvious scalar/vector mistakes
- contains unsupported or symbolic-only steps

Verifier output is used by the ranker. It is not a neural model.

---

## 4.5 Rule-based Ranker

File:

```text
type2_candidate_ranker.py
```

Purpose:

```text
verified candidates
→ selected candidate
```

The ranker combines verifier scores and rule-based adjustments.

Important behavior:

- boost geometry-specific candidates when geometry is explicit
- reject scalar Coulomb in multi-charge vector settings
- penalize skeleton placeholders
- avoid over-selecting raw LLM fallback
- small boost for clean `llm_fallback_canonicalized` candidates

Step 10 added:

```text
+0.015 boost for source == "llm_fallback_canonicalized"
```

only when the verifier profile is clean and status is PASS.

---

## 4.6 Candidate Pipeline Wrapper

File:

```text
type2_candidate_pipeline.py
```

Purpose:

```text
parse_and_adapt
→ generate_step_plan_candidates
→ verify_step_plan_candidates
→ rank_verified_candidates
→ execute selected step_plan
→ sanity check
→ final answer
```

This file is the main runtime orchestration layer.

It should remain stable and should not contain heavy formula-specific logic.

---

## 4.7 Numeric Executor

File:

```text
type2_numeric_executor.py
```

Purpose:

```text
selected step_plan
+ known quantities
+ target
→ numeric answer / symbolic fallback
```

The executor is deterministic.

It does not use unsafe `eval`.

Its core dispatch function is:

```text
_execute_formula_step
```

The executor supports:

- capacitor formulas
- Coulomb scalar formulas
- Coulomb vector / role-aware scene extraction
- electric field scalar formulas
- Ohm and power formulas
- parallel plate capacitance
- LC / RLC formulas
- inductor energy
- magnetic field / flux / EMF formulas
- measurement and uncertainty formulas
- target writeback aliases
- executor-side loose LLM formula aliases

Important Step 9 additions:

- `mu_0 = 4*pi*1e-7`
- LC/RLC formula dispatches
- magnetic / flux / EMF dispatches
- capacitor inverse dispatches
- function amplitude parsing
- current delta parsing
- target writeback alias expansion

Important Step 10 additions:

- `_LOOSE_FORMULA_ALIASES`
- `_normalize_loose_formula`
- defensive mapping of unambiguous loose formula names
- `U_R = I_rms * R`
- `U_C = I_rms * X_C`

---

## 4.8 Answer Sanity Checker

File:

```text
type2_answer_sanity_checker.py
```

Purpose:

```text
numeric answer
→ LOW / MEDIUM / HIGH / CRITICAL risk
→ accept or downgrade
```

The sanity checker evaluates:

- missing numeric value
- NaN / inf / extreme values
- missing units
- target-unit mismatch
- negative magnitudes
- parser failure
- verifier warning or failure
- low rank margin
- vector geometry warnings
- unexpected zero
- skeleton numeric outputs

After Step 7.1, the sanity checker separates hard risk from soft risk.

Current calibrated risk behavior:

- LOW: accept
- MEDIUM: accept with reduced confidence
- HIGH: accept in non-strict mode, downgrade in strict mode
- CRITICAL: downgrade

---

## 4.9 LLM Fallback Canonicalizer

File:

```text
type2_llm_canonicalizer.py
```

Purpose:

```text
loose LLM fallback step_plan
→ canonical executor-supported step_plan
```

Example mappings:

| LLM fallback formula | Canonical formula |
|---|---|
| `electric_force` | `F = k * abs(q1*q2) / r^2` |
| `energy_in_capacitor` | `U_cap = 0.5 * C_cap * V^2` |
| `magnetic_field_solenoid` | `B = mu_0 * n_turns_per_meter * I` |
| `current_rms` | `I_rms = V / Z` |
| `impedance_series` | `Z = sqrt(R^2 + (X_L - X_C)^2)` |
| `resonance_condition` | `f_res = 1 / (2*pi*sqrt(L_ind*C_cap))` |

The canonicalizer is deterministic and uses safety guards:

- do not collapse multi-charge force problems into scalar Coulomb
- do not force ambiguous vector geometry numeric
- do not execute arbitrary formulas
- do not overwrite deterministic candidates
- only add a canonicalized candidate when the mapping is safe

Current live bug:

The canonicalizer is implemented, but the live LLM run produced:

```text
llm_fallback_canonicalized_candidate_count: 0
canonicalization_status_counts: {}
canonicalization_mapping_counts: {}
```

This means the candidate-level canonicalization path did not run or was not loaded.

---

## 5. Validation Results

## 5.1 Step 6.5 Full Dataset Baseline

After high-frequency formula dispatch expansion:

```text
evaluated: 1352
numeric_answer_count: 857
numeric_rate: 63.39%
symbolic_trace_count: 459
failed_count: 1
```

Major improvements:

- electric field point charge: nearly fully covered
- parallel plate capacitance: fully covered
- LC/RLC and inductor formulas: large improvement

---

## 5.2 Step 7.1 Sanity-Calibrated Result

Non-strict:

```text
numeric_answer_count: 854
symbolic_trace_count: 462
failed_count: 1

LOW: 672
MEDIUM: 167
HIGH: 15
CRITICAL: 3
```

Strict:

```text
numeric_answer_count: 839
symbolic_trace_count: 477
failed_count: 1
```

Main conclusion:

The sanity checker is usable and should be frozen unless a clear bug appears.

---

## 5.3 Step 8 Failure Clustering

Step 8 identified the top remaining clusters:

| Cluster | Count | Priority | Risk | Decision |
|---|---:|---|---|---|
| LC / RLC unresolved | 98 | P0 | Low | Fixed in Step 9 |
| Magnetic / flux / EMF unsupported | 32 | P0 | Low | Mostly fixed in Step 9 |
| Capacitor inverse aliases | 35 | P1 | Medium | Fixed in Step 9 |
| Coulomb geometry unresolved | 69 | P1 | High | Deferred |
| Skeleton / parser low confidence | 140 | P0 | High | Deferred |
| Measurement arrays unavailable | 22 | P2 | High | Deferred |
| Non-numeric conceptual / boolean | 68 | P2 | Medium | Deferred |

---

## 5.4 Step 9 Result

Step 9 was the biggest recent improvement after Step 6.5.

| Metric | Before Step 9 | After Step 9 |
|---|---:|---:|
| Non-strict numeric | 854 | 990 |
| Strict numeric | 839 | 974 |
| Symbolic trace | 462 | 326 |
| Failed | 1 | 1 |
| Unsupported dispatch | 189 | 68 |
| Sanity CRITICAL | 3 | 3 |
| Sanity HIGH | 15 | 16 |

Step 9 fixed:

- LC/RLC unresolved cluster: 98 → 31
- magnetic / flux / EMF cluster: 32 → 7
- capacitor inverse cluster: 35 → 0

The deterministic non-LLM path must stay at this level.

---

## 5.5 Step 10 Live LLM Result

With real Qwen3 LLM fallback enabled:

```text
evaluated: 1352
numeric_answer_count: 996
symbolic_trace_count: 321
failed_count: 0
unsupported_dispatch_count: 117
llm_fallback_selected_count: 70
llm_fallback_numeric_count: 9
llm_fallback_symbolic_count: 61
llm_fallback_canonicalized_candidate_count: 0
llm_fallback_canonicalized_selected_count: 0
llm_fallback_canonicalized_numeric_count: 0
```

Interpretation:

- LLM fallback improves parser status.
- Executor-side loose formula aliasing helps slightly.
- Candidate-level canonicalization did not fire.
- Step 10 is only partially effective until the generator/canonicalizer integration bug is fixed.

---

## 6. Known Open Bug

## 6.1 Bug: Canonicalized Candidate Path Not Firing

Live Step 10 result:

```text
llm_fallback_canonicalized_candidate_count: 0
canonicalization_status_counts: {}
canonicalization_mapping_counts: {}
```

Expected:

```text
llm_fallback_canonicalized_candidate_count > 0
canonicalization_status_counts non-empty
canonicalization_mapping_counts non-empty
```

Evidence:

- New executor is loaded because executor-side formula aliases reduced unsupported dispatch.
- New validator is loaded because Step 10 diagnostic keys exist.
- Candidate-level canonicalization did not fire.
- The run did not crash, so it likely did not import the updated generator hook.

Most likely cause:

```text
The pipeline is importing a stale type2_candidate_generator.py from another path.
```

## 6.2 First Diagnostic Command

Run this on the user's Windows machine:

```powershell
cd "E:\Education System"
$env:PYTHONPATH="E:\Education System;E:\Education System\pipeline"
py -c "import type2_candidate_generator as g; print('hook present:', hasattr(g,'_add_llm_fallback_canonicalized_variant')); print('loaded from:', g.__file__)"
```

Expected:

```text
hook present: True
loaded from: E:\Education System\pipeline\type2_candidate_generator.py
```

If the result is:

```text
hook present: False
```

then the loaded generator is stale. Replace the file at the printed path with the updated `type2_candidate_generator.py`.

## 6.3 Secondary Causes After Deployment Is Correct

Even after the generator is correctly loaded, canonicalized numeric gain may still be limited by:

1. Sparse `known_quantities`
   - Example: parallel-plate problems may have radius but not area.
   - The canonicalizer currently requires `A` and `d`.
   - It should derive `A = pi*r^2` when radius exists.

2. Deliberately unmapped risky geometry formulas
   - `vector_addition`
   - `electric_field_superposition`
   - `electric_field_zero`
   - `electric_field_and_voltage`

3. Counter blind spot
   - Current counters may count only selected canonicalized candidates.
   - Need to distinguish:
     ```text
     generated canonicalized candidates
     selected canonicalized candidates
     numeric canonicalized candidates
     symbolic canonicalized candidates
     ```

---

## 7. Recommended Next Improvements

## 7.1 Step 10.1 — Fix LLM Canonicalizer Integration

Priority: P0

Goal:

```text
Ensure llm_fallback_canonicalized candidates are actually generated, visible, selected when appropriate, and counted.
```

Tasks:

1. Confirm `type2_candidate_generator.py` import path.
2. Confirm `_add_llm_fallback_canonicalized_variant` exists.
3. Confirm `generate_step_plan_candidates` calls the hook.
4. Confirm the hook appends a new candidate to the candidate list.
5. Confirm candidate source is exactly:
   ```text
   llm_fallback_canonicalized
   ```
6. Confirm verifier and ranker accept this source.
7. Add generated-candidate counters from `generation_summary`.
8. Re-run LLM full dataset validation.

Success metrics:

```text
llm_fallback_canonicalized_candidate_count > 0
llm_fallback_canonicalized_selected_count > 0
llm_fallback_canonicalized_numeric_count > 0
canonicalization_status_counts not empty
canonicalization_mapping_counts not empty
```

Regression gate:

```text
non-LLM numeric_answer_count must remain 990
non-LLM failed_count must remain 1
0 answer_type diffs vs Step 9 baseline
```

---

## 7.2 Step 10.2 — Widen Canonicalizer Input Resolution

Priority: P1

Only do this after Step 10.1 confirms canonicalized candidates are generated.

Improve canonicalizer resolution:

1. Derive area from radius:
   ```text
   A = pi * r^2
   ```
2. Support circular plate phrasing:
   ```text
   radius = ...
   plate radius = ...
   circular plates
   ```
3. Add energy aliases:
   ```text
   energy
   stored_energy
   electric_energy
   energy_in_capacitor
   U_E
   U_cap
   ```
4. Add voltage aliases:
   ```text
   V
   U
   U_C
   voltage
   peak_voltage
   V_max
   U_max
   ```
5. Add charge aliases:
   ```text
   Q
   q
   Q_max
   charge
   charge_on_capacitor
   ```
6. Add safe mappings for high-frequency unsupported formulas only if deterministic:
   - `electric_field_between_parallel_plates`
   - `electric_field_between_parallel_charged_sheets`
   - `charge_on_capacitor`
   - `capacitance_formula`
   - `energy_stored_in_capacitor`
   - `ohms_law`

Do not map ambiguous multi-charge vector geometry into numeric formulas.

---

## 7.3 Step 11 — End-to-End Accuracy Evaluation

Priority: P0/P1

Coverage is now high, but coverage is not the same as correctness.

Need to implement or run:

```text
predicted answer vs gold answer
```

Metrics:

1. numeric tolerance accuracy
2. unit accuracy
3. exact string accuracy for symbolic / conceptual outputs
4. template-level accuracy
5. risk-level accuracy
6. LOW/MEDIUM/HIGH sanity accuracy
7. failure-type accuracy

Recommended output:

```text
overall_accuracy
numeric_accuracy
unit_accuracy
safe_numeric_accuracy
risky_numeric_accuracy
accuracy_by_template
accuracy_by_target
accuracy_by_domain
wrong_but_high_confidence_count
correct_but_symbolic_count
```

This should happen before adding Transformer or doing more parser rewrites.

---

## 7.4 Step 12 — Remaining High-Risk Coverage

Priority: P1/P2

After accuracy evaluation, consider targeted fixes for:

### A. Coulomb geometry unresolved

Current remaining issues:

- midpoint cases
- collinear opposite-side cases
- square center / symmetric cancellation
- target point inference
- source-target distance inference
- symbolic variables such as `a`, `r`, `h`

Do not force numeric if geometry is ambiguous.

### B. Skeleton / parser low confidence

This may require parser-level changes:

```text
target_detector.py
condition_extractor.py
template_fallback.py
type2_candidate_generator.py
```

Risk is high because parser changes can destabilize current numeric coverage.

### C. Measurement arrays

Current limitation:

- executor can compute mean/error only if measurement arrays exist
- parser often does not extract arrays
- fixing requires extraction changes

### D. Conceptual / boolean handler

Current numeric executor is not designed for conceptual relation questions.

Possible future module:

```text
type2_symbolic_executor.py
```

This can handle:

- relation questions
- yes/no questions
- proportional reasoning
- qualitative direction questions
- conceptual formula comparisons

---

## 7.5 Optional Validation-Layer LLM Cache

Priority: P3

Only add if reruns are too slow.

Constraints:

- off by default
- validation layer only
- no cache inside parser/generator/executor/core pipeline
- must not change `parse_problem` default behavior
- only safe if Qwen3 inference is deterministic

Suggested implementation:

```text
llm_cache.py
--llm-cache outputs/llm_cache.sqlite
```

Cache key:

```text
(problem_text, missing_fields, prompt_version)
```

Cache value:

```text
raw LLM JSON response
```

Before implementing cache, confirm:

- temperature = 0
- fixed seed or deterministic decoding
- same prompt gives same response

---

## 8. File Inventory

| File | Status | Purpose |
|---|---|---|
| `type2_adapter.py` | existing | Converts parser output into world model input. |
| `type2_candidate_generator.py` | edited | Generates candidate step plans; should include canonicalizer hook. |
| `type2_candidate_verifier.py` | existing | Scores candidate validity. |
| `type2_candidate_ranker.py` | edited | Selects candidate; includes small canonicalized boost. |
| `type2_candidate_pipeline.py` | existing | End-to-end orchestration. |
| `type2_numeric_executor.py` | heavily edited | Formula dispatch, numeric execution, target writeback, loose formula alias fallback. |
| `type2_answer_sanity_checker.py` | existing | Risk-aware final answer checker. |
| `type2_llm_canonicalizer.py` | new | Maps loose LLM formulas to canonical step plans. |
| `validate_type2_full_dataset.py` | edited | Full-dataset validation and diagnostics. |
| `analyze_type2_remaining_failures.py` | existing | Failure clustering report tool. |
| `parser/llm_fallback.py` | unchanged | Stage-0 LLM fallback. |
| `parser/condition_extractor.py` | unchanged | Quantity/condition extraction. |
| `parser/target_detector.py` | unchanged | Target extraction. |

---

## 9. Recommended Commands for Next Agent

### 9.1 Check Generator Deployment

```powershell
cd "E:\Education System"
$env:PYTHONPATH="E:\Education System;E:\Education System\pipeline"
py -c "import type2_candidate_generator as g; print('hook present:', hasattr(g,'_add_llm_fallback_canonicalized_variant')); print('loaded from:', g.__file__)"
```

### 9.2 Run Deterministic Regression

```powershell
py validate\validate_type2_full_dataset.py `
  --input "Dataset\Physics_Problems_Text_Only.csv" `
  --output outputs\type2_full_dataset_validation_regression.json `
  --save-jsonl outputs\type2_full_dataset_rows_regression.jsonl `
  --max-candidates 8 `
  --compact
```

Expected:

```text
numeric_answer_count: 990
failed_count: 1
```

### 9.3 Run LLM Canonicalized Validation

```powershell
py validate\validate_type2_full_dataset.py `
  --input "Dataset\Physics_Problems_Text_Only.csv" `
  --output outputs\type2_full_dataset_validation_step10_fixed.json `
  --save-jsonl outputs\type2_full_dataset_rows_step10_fixed.jsonl `
  --max-candidates 8 `
  --compact `
  --use-llm-fallback
```

Expected after Step 10.1 fix:

```text
llm_fallback_canonicalized_candidate_count > 0
canonicalization_status_counts not empty
canonicalization_mapping_counts not empty
```

### 9.4 Re-run Failure Clustering

```powershell
py validate\analyze_type2_remaining_failures.py `
  --summary-json outputs\type2_full_dataset_validation_step10_fixed.json `
  --rows-jsonl outputs\type2_full_dataset_rows_step10_fixed.jsonl `
  --output-json outputs\type2_remaining_failure_clusters_step10_fixed.json `
  --output-md outputs\type2_remaining_failure_clusters_step10_fixed.md `
  --top-k 30 `
  --sample-per-cluster 10
```

---

## 10. Current Development Strategy

The project has moved past basic pipeline construction.

Current phase:

```text
coverage → correctness → robustness → final API
```

Recommended order:

1. Fix Step 10.1 canonicalized candidate integration.
2. Run deterministic regression.
3. Run real LLM validation.
4. Implement end-to-end accuracy evaluation.
5. Only then decide whether to:
   - improve Coulomb geometry
   - repair skeleton/parser failures
   - add symbolic/conceptual executor
   - add more formula dispatch
   - introduce Transformer/learning-based components

Do not move to Transformer until answer-level accuracy is measured.

---

## 11. One-Sentence Handoff

The Type2 pipeline is already end-to-end functional and reaches 990/1352 numeric answers in deterministic mode; Step 9 is stable, while Step 10’s executor-side loose formula aliasing works partially, but the `llm_fallback_canonicalized` candidate path currently does not fire in the real LLM run and should be debugged first before adding new modeling components.
