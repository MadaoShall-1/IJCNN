# Education System — Optimization Log

## Project Status Overview

### Completed

1. **Stage 0 Parser** — success rate **96%** on eval cohort
   - Deterministic-first physics problem parser (`parser/main.py`)
   - Covers: question type classification, domain classification, condition extraction, quantity extraction, relation extraction, target detection, step plan proposal, verification
   - Template coverage: AC off-resonance, resonance design, EM/solenoid, capacitor energy/merge, mechanics (free-fall, braking, Newton's 2nd law), measurement error, thermodynamics basics, least-count error, AB-circuit quadrature
   - Concept classifier handles symbolic/boolean/numeric routing with strong/weak override hierarchy

2. **Stage 1 Formula Retrieval** — verification complete
   - Formula library matching and retrieval (`stage1/`)
   - Evaluation results saved in `outputs/`

3. **No-vLLM Pipeline** — end-to-end run successful
   - Full pipeline using RAG (retrieval-augmented generation) without vLLM inference
   - Covers: parsing → formula retrieval → solution generation
   - Validated on dataset end-to-end

---

### TODO — Improvements & Testing Needed

1. **Full Pipeline with vLLM**
   - Integrate vLLM-based inference into the pipeline
   - Test end-to-end: parsing → formula retrieval → vLLM reasoning → answer generation
   - Benchmark latency and accuracy vs no-vLLM baseline
   - Validate on full eval cohort

2. **Stage 1 Suspicious Cases Bug Fix**
   - Fix `low_formula_name_overlap` issues in `stage1_suspicious.jsonl` (e.g., vector sum steps matched to wrong formulas like `V=IR` or `E=k_e*q/r²`)
   - Improve canonicalization and scoring for multi-charge force/vector-sum steps
   - Add missing formula entries for vector addition, capacitor charge (`Q=CV`), and force resultant templates

3. **Stage 2 & 3 Code Correctness Verification**
   - Stage 2 (`type2/stage2.py`): verify intermediate computation logic, edge case handling
   - Stage 3: verify solution derivation and answer formatting
   - Unit tests and integration tests for stage 2/3 modules
   - Regression testing against known-good outputs

---

## Architecture Summary

```
Input (physics problem text)
  → Stage 0: Parser (deterministic, regex/template)
    → question_type_classifier → domain_classifier → condition_extractor
    → rule_extractor → target_detector → template_fallback → parse_verifier
  → Stage 1: Formula Retrieval (RAG-based)
    → formula_library.json matching → step plan enrichment
  → Stage 2: Computation (needs verification)
  → Stage 3: Solution Generation (needs verification)
```

Pipeline entry points:
- No-vLLM: `router.py` → `api.py` (RAG path)
- With vLLM: TBD integration

## Key Files

| Module | Path | Status |
|--------|------|--------|
| Parser pipeline | `parser/main.py` | Done |
| Question type classifier | `parser/question_type_classifier.py` | Done |
| Domain classifier | `parser/domain_classifier.py` | Done |
| Condition extractor | `parser/condition_extractor.py` | Done |
| Rule extractor | `parser/rule_extractor.py` | Done |
| Target detector | `parser/target_detector.py` | Done |
| Template fallback | `parser/template_fallback.py` | Done |
| Parse verifier | `parser/parse_verifier.py` | Done |
| Stage 1 formula retrieval | `stage1/stage1.py` | Done |
| Formula library | `stage1/formula_library.json`, `type2/formula_library.json` | Done |
| Type 1 pipeline (logic) | `type1/pipeline.py` | Done |
| Type 2 stages | `type2/stage1.py` ~ `type2/stage6.py` | Stage 2/3 needs verification |
| Router | `router.py` | Done |
| API | `api.py` | Done |
| Config | `config.py` | Done |

## Parser Optimization History (Stage 0)

Eval cohort: 1,354 problems (excluding QA-prefixed out-of-distribution rows).

| Phase | PASS | PNN | FAIL | Rate |
|-------|-----:|----:|-----:|-----:|
| Baseline | 978 | 71 | 305 | 77.5% |
| Phase 1 (AC off-resonance) | 1001 | 71 | 282 | 79.2% |
| Phase 1b (omega-factor, cap merge, measurement) | 1033 | 71 | 250 | 81.5% |
| Phase 2 (energy diff, least-count, AB-circuit) | 1064 | 71 | 219 | 83.8% |
| Phase 3 (concept classifier) | 1064 | 76 | 214 | 84.2% |
| **Current (with further tuning)** | — | — | — | **~96%** |

## Critical Invariants

1. PASS requires verifier confirmation — never bypass
2. Skeleton fallback: `template_name="skeleton_placeholder"`, `confidence=0.30` — do not change
3. Numeric problems require at least one real calculation step
4. PASS_NON_NUMERIC only for `boolean_check` / `symbolic_derivation` question types
5. No external dependencies — pure Python stdlib only
