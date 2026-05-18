# Physics Problem Solver — System Design Document
**Version:** 0.8  
**Status:** Draft  
**Framework:** Agnostic (reference implementations may use DSPy or LangChain)

---

## 1. Purpose and Scope

This document defines the architecture for an automated solver targeting the competition's
unified test set, which combines Dataset Type 1 (logic-based educational queries) and
Dataset Type 2 (physics calculation problems). The API endpoint must handle both types.
The official training set contains 808 Type 1 questions and 1,354 valid Type 2 problems
(after filtering the 401 QA-prefixed annotation errors per Q19), giving an approximate
37%/63% split. The pipeline is optimised for Type 2 while maintaining a fully functional
Type 1 path.

The document is **framework-agnostic**. Where DSPy or LangChain are mentioned, they are
illustrative references only. Each stage is defined by its concrete **inputs**, **outputs**, and
**acceptance criteria** — not by implementation choice.

**Model constraint — open-source:** All models used in this pipeline must be open-source
with publicly downloadable weights. Permissive research licenses (Llama Community License,
Gemma Terms of Use, Qwen License, DeepSeek License, Mistral Research License) are
accepted alongside standard OSI licenses (Apache-2.0, MIT). No proprietary API may be
relied upon as a required component at inference time.

**Model constraint — parameter budget (Q1–Q3):** At any moment during inference, only
one LLM (≤ 8B-class nominal label) may be actively loaded in GPU memory. The 8B-class
label is interpreted by nominal release name, not exact parameter count — a model
labelled "8B" is eligible even if actual parameters are slightly above 8.0B. MoE models
are evaluated on total parameters, not active parameters. Sequential use of different
≤ 8B models across pipeline stages is allowed; simultaneous parallel inference from two
models is not.

**Model constraint — no third-party inference APIs (Q5):** The LLM component must be
self-hosted via **vLLM** (or a compatible OpenAI-style serving framework). Third-party
inference APIs (Together AI, Groq, Fireworks, HuggingFace Inference API, etc.) are
not permitted. Tools, solvers, and retrieval modules are not LLMs and may be hosted
anywhere.

**Recommended model configuration — LoRA adapter strategy:**
One base model (initial candidate: **Qwen3-8B**) is loaded once and remains resident
in GPU memory for the full request lifecycle. Role specialisation is achieved by
hot-swapping LoRA adapter weights between pipeline stages. A LoRA adapter for a 7–8B
model is typically 10–100 MB; swapping adapters takes sub-millisecond time when
pre-loaded, imposing no meaningful latency cost. A single small embedding model
(≤ 100M parameters, e.g., `bge-small-en`) handles formula library similarity search.

| Role | Adapter name | Used by |
|---|---|---|
| Physics solver | `physics-solver` | Stage 2+3 generation |
| Step verifier / annotator | `verifier` | Stage 3 LLM verification, Stage 4 diagnosis |
| Logic reasoner | `logic-reasoner` | Type 1 path, Z3 formalization |
| Response assembler | `response-assembler` | Stage 6 explanation generation |

All adapters are declared at vLLM startup and pre-loaded so no disk I/O occurs during
inference. The base model is identical for all adapters; GPU memory contains only one
set of base weights at any time.

**Knowledge distillation (Q6):** Larger teacher models (e.g., GPT-4, Claude) may be
used during training to generate annotated traces and adapter training data, provided:
the final deployed model is within the 8B-class limit; the teacher is not called at
inference time; and the teacher model and process are fully disclosed in the Data
Disclosure Document.

Note: The master's thesis motivating this design used Qwen2.5-14B-Instruct and
Gemma-3-12b-it. Both exceed the parameter budget and cannot be used directly. Performance
baselines from the thesis should be treated as upper-bound aspirational targets, not
expected results at 8B scale.

---

## 2. Design Principles

1. **Every reasoning step is explicit and addressable.** Free-form prose output is not
   acceptable as a pipeline artifact. Each step must be a structured object that downstream
   stages can read, verify, and repair without re-parsing natural language.

2. **Errors are caught as early as possible.** Verification is performed per-step during
   generation, not in a separate batch pass afterward. This prevents wasted computation on
   steps that are already known to be downstream of an error.

3. **Repair is targeted, not wholesale.** When an error is detected, the correct prefix of the
   solution is preserved and regeneration begins at the earliest identified wrong step. Full
   regeneration from scratch is a fallback of last resort, attempted only after targeted repair
   has been exhausted.

4. **Variable state is explicit.** Every step declares the variables it consumes and the
   variables it produces. A shared Variable State Object is the single source of truth for
   intermediate values. Steps do not rely on implicit context to find prior results.

5. **The verifier and the solver are independent roles.** The component that generates an
   answer must not be the sole component that checks it. Verification uses a separate call,
   a separate model, or a deterministic tool (e.g., SymPy) — never the generator's own
   confidence alone.

6. **Multiple solution paths exist in parallel.** Stage 1 retrieves top-N candidate formula
   sets, not just one. If the primary solution path fails to produce a correct answer after
   all repair attempts, the pipeline retries from Stage 2 using the next-ranked formula set.
   This is the beam-search fallback.

7. **Outcomes are framework-agnostic.** The interfaces between stages are defined as JSON
   schemas. Any framework (DSPy, LangChain, raw API calls) that produces a conforming
   JSON object satisfies the contract.

8. **Deterministic tools are preferred over LLM calls wherever the output space is finite
   or mechanically verifiable.** Domain classification uses a keyword lookup table before
   falling back to the LLM. Variable canonicalization uses a regex table before embedding
   search before the LLM. FWS identification for checkable steps uses the SymPy verifier
   result directly. COT assembly is pure string formatting. Error types E4 and E5 are
   detected programmatically. Confidence scores are derived from verifier outcomes, not
   LLM self-report. The LLM is reserved for tasks that genuinely require natural language
   understanding or novel content generation.

---

## 2.1 MVP Build Plan

This section defines the minimum viable pipeline that can produce scorable answers,
and separates it from enhancement work. Build and validate Phase 0 before adding
anything from Phase 1 or Phase 2.

### Phase 0 — Minimum Viable Submission
*Goal: able to return a valid `answer` + `explanation` for every request within 60 seconds.*

| Component | What to build | Reference |
|---|---|---|
| vLLM serving | Base model only (no LoRA adapters yet). Confirm `/v1/models` responds. | §9.1 |
| Router | Read `query_type` from payload; fallback heuristic on `premises-NL` | §3.1 |
| Stage 0 | quantulum3 + Pint quantity extraction; keyword-only domain classification; LLM step plan generation (no structural validation yet) | §5 Stage 0 |
| Formula library | Circuits/electrostatics entries only (Ohm's law, Kirchhoff's laws, series/parallel resistance, power, capacitance, Coulomb's law). `sympy_expr` required; `fol_axiom`/`premise_text` optional at this phase. | §4.3, §11 OQ#7 |
| Stage 1 | Regex canonicalization (Tier 1 only); embedding retrieval against the formula library; single formula path (`beam_n = 1`) | §5 Stage 1 |
| Stage 2+3 | `physics-solver` base model generation; SymPy verification only (no Wolfram Alpha yet); one retry per step | §5 Stage 2+3 |
| Type 1 path | LLM-only reasoning for all question types (no Z3/Prover9 yet) | §3.1 |
| Stage 6 | `answer` extraction; `explanation` via base model. Skip all optional fields (`fol`, `cot`, `premises`, `confidence`). | §5 Stage 6 |
| Hard timeout | Emit best available answer at 55s. No tier fallbacks needed yet. | §9.2 |

**Phase 0 acceptance gate:** Submit 20 training set problems. All return valid JSON with
`answer` and `explanation`. No requests timeout.

---

### Phase 1 — Core Quality
*Goal: meaningful improvement on P1 (correctness) and P2 (explanation quality).*

| Component | What to build | Adds |
|---|---|---|
| LoRA adapters: `physics-solver` + `verifier` | Train and hot-swap these two adapters. They provide the most direct quality lift. | Role specialization for generation and verification |
| Z3 for Type 1 Yes/No/Uncertain | LLM formalizes → Z3 executes → deterministic result | Replaces poorly calibrated LLM confidence on ~37% of test set |
| `type1_verify` pass | Second `verifier` adapter review for MCQ/open-ended Type 1 | Independent check on Type 1 answers |
| Stage 4+5: FWS repair | Identify first wrong step; repair from that point | Recovers traces that would otherwise FAIL |
| Beam search (N = 2) | Two candidate formula paths; retry on failure | Reduces formula retrieval misses |
| `cot` + `confidence` outputs | String-formatted COT; geometric mean confidence | Required for P3 scoring |
| Step plan structural validation | Catch bad LLM-generated plans before Stage 1 | Prevents silent pipeline corruption |
| Timeout Tier 1 (12s fallback) | Drop beam search + repair if elapsed > 12s | Protect against slow hard cases |

**Phase 1 acceptance gate:** Run full training set. P1 score meaningfully above Phase 0
baseline. No requests timeout.

---

### Phase 2 — Enhancement
*Build if time permits after Phase 1 is validated. None of these are required to score.*

| Component | Benefit | Notes |
|---|---|---|
| LoRA adapters: `logic-reasoner` + `response-assembler` | Specialized Type 1 reasoning; higher quality explanations | Train after Phase 1 adapters are validated |
| Prover9 | Better pure-predicate Type 1 reasoning | Two-pass execution required; add only after Z3 is stable |
| Wolfram Alpha API | Numeric verification for `tool_dispatch: "wolfram"` entries | Requires API key; session cache; offline fallback corpus |
| Wikipedia API fallback | Formula retrieval for out-of-library formulas | Build offline corpus first; live API is secondary |
| Embedding canonicalization (Tier 2) | Better variable name matching beyond regex | Requires `bge-small-en` embedding model |
| LLM canonicalization (Tier 3) | Catch edge cases missed by regex + embedding | One extra LLM call; only on miss |
| Collision detection in canonicalization | Prevent silent variable overwriting | Audit circuits/electrostatics terms specifically |
| `fol` + `premises` outputs | FOL axioms and premise listing | Requires `fol_axiom`/`premise_text` in formula library |
| Timeout Tier 2 (35s fallback) | Skip optional fields on slow requests | Fine-tune thresholds after measuring real latency |
| Offline Wolfram/Wikipedia cache | Day-of reliability insurance | Build from training set queries |
| Formula library: secondary domains | Mechanics, thermodynamics, optics, modern physics coverage | Only after circuits/electrostatics are fully tested |

---

## 3. High-Level Pipeline

```
API Request
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  Router: Read query_type field from request payload      │
│  query_type = "type1" → Type 1 path                     │
│  query_type = "type2" → Type 2 path                     │
│  (No inference needed — type is provided by organizers)  │
└──────────────────────────────────────────────────────────┘
        │                          │
        │ Type 1                   │ Type 2
        ▼                          ▼
┌───────────────────┐   ┌──────────────────────────────────────────┐
│  Type 1 Path      │   │  Stage 0: Problem Parsing                │
│  (Section 3.1)    │   │  (hybrid programmatic + LLM extraction)  │
└───────────────────┘   └──────────────────────────────────────────┘
        │                          │
        │                          ▼
        │               ┌──────────────────────────────────────────┐
        │               │  Stage 1: Formula Retrieval              │
        │               │  → Top-N ranked formula sets             │
        │               └──────────────────────────────────────────┘
        │                          │
        │               ┌── Candidate Path Loop ──────────────────┐
        │               │  (set #1 first; advance on exhaustion)  │
        │               ▼                                          │
        │      ┌──────────────────────────────────────────────┐   │
        │      │  Stage 2+3: Step-by-Step Generation &        │   │
        │      │  Calibration (generate → verify → repair)    │   │
        │      └──────────────────────────────────────────────┘   │
        │               │  FAIL after retry limit                  │
        │               ▼                                          │
        │      ┌──────────────────────────────────────────────┐   │
        │      │  Stage 4: Error Structuring & Diagnosis      │   │
        │      └──────────────────────────────────────────────┘   │
        │               │                                          │
        │               ▼                                          │
        │      ┌──────────────────────────────────────────────┐   │
        │      │  Stage 5: FWS-Centered Repair                │   │
        │      └──────────────────────────────────────────────┘   │
        │               │  FAIL: advance to next path ────────── ┘
        │               │  PASS
        │               ▼
        └──────────────►┌──────────────────────────────────────┐
                        │  Stage 6: Response Assembly           │
                        │  answer, explanation, fol, cot,       │
                        │  premises, confidence                  │
                        └──────────────────────────────────────┘
                                       │
                                       ▼
                                  API Response
```

**Loop termination (Type 2):** The pipeline exits with the best available trace when
either (a) a PASS is achieved on any candidate path, or (b) all candidate paths are
exhausted.

### 3.1 Type 1 Path

Type 1 queries supply their own premises in the request payload. The physics-specific
stages (formula retrieval, VSO, unit extraction) are not needed.

**Router fallback — missing or unrecognized `query_type` (#13):**
The test format is not locked until kick-off. If `query_type` is absent or not one of
`["type1", "type2"]`, apply this heuristic before routing. The check covers both the
official field name `"premises-NL"` and the alternative `"premises"` in case the field
name changes between dataset releases:
```python
if payload.get("premises-NL") or payload.get("premises"):
    query_type = "type1"   # premises field present → logic problem
else:
    query_type = "type2"   # default to physics solver
```
This ensures the pipeline never crashes on a missing field.

**Answer format detection (programmatic):**
Detect the expected answer format from the question text using pattern matching — no LLM
call required for this step:
- MCQ: question contains labelled options (A), (B), (C) … or "which of the following"
- Yes/No/Uncertain: question contains "is it true", "does X imply Y", "can we conclude"
- Open-ended: all other forms

**Routing within Type 1 — Z3 vs Prover9 decision (#17):**

Use the following rule to decide which solver to invoke for Yes/No/Uncertain questions:

| Condition | Solver |
|---|---|
| Premises contain numeric values, inequalities, or thresholds | **Z3** |
| Premises are purely symbolic predicate statements (all X are Y, A is a B) | **Prover9** |
| Cannot determine (ambiguous) | **Z3** (default) |

Both Z3 and Prover9 are fully implemented in this pipeline (see Section 9.3 for
infrastructure). The routing decision is made programmatically by checking whether any
premise string contains a digit or comparison operator (`<`, `>`, `≤`, `≥`, `=`).

**Process:**
1. Detect answer format programmatically (see above).
2. Premises are already structured in the payload — no extraction needed. Echo them
   directly to the `premises` field of the API response.
3. **For Yes/No/Uncertain questions — formal solver path (preferred):**

   **Step 3a — Z3 sub-path:**
   i.  Load the `logic-reasoner` LoRA adapter.
   ii. LLM formalises premises and the conclusion-to-test as Z3 Python API code.
   iii. **Pre-execution structural validation (#5):** Before running Z3, parse the
        generated code and check:
        - All premise variables referenced in the conclusion are declared in the code.
        - The conclusion is encoded as the negation of what is to be proved (standard
          refutation form).
        - No undefined symbols (check that every identifier used in `solver.add()`
          was previously declared via `declare_const` or equivalent).
        If validation fails, fall back immediately to the LLM reasoning path (Step 4)
        rather than executing broken Z3 code. Log the failure reason.
   iv. Execute Z3. Map result deterministically:
       - Negated conclusion is `UNSAT` → answer = `"Yes"` (conclusion must hold)
       - Negated conclusion is `SAT` and conclusion is `SAT` → answer = `"Uncertain"`
       - Conclusion itself is `UNSAT` → answer = `"No"`
       - Z3 returns `UNKNOWN` → fall back to LLM answer
   v.  The Z3 code generated by the LLM **is** the `fol` output — no separate FOL
       generation step needed. It is formally verified and directly human-auditable.

   **Step 3b — Prover9 sub-path:**
   i.  Load the `logic-reasoner` LoRA adapter.
   ii. LLM formalises premises as Prover9 assumptions, emitting two separate goal
       strings: one with the conclusion as the goal (`goal_positive`), and one with the
       negated conclusion as the goal (`goal_negative`).
   iii. **Pre-execution structural validation:** Check that both generated inputs contain
        at least one `formulas(assumptions)` block and one `formulas(goals)` block.
        If either is malformed, fall back to the LLM reasoning path.
   iv. **Two-pass execution** (Prover9 can only prove, not disprove; two runs required):

       ```
       Pass 1: goal = conclusion
         - Proof found       → answer = "Yes"  (conclusion is provable from premises)
         - No proof / timeout → proceed to Pass 2

       Pass 2: goal = negation of conclusion
         - Proof found       → answer = "No"   (negation is provable → conclusion is false)
         - No proof / timeout → answer = "Uncertain"
       ```

       Each pass has a wall-clock timeout of 5 seconds. If either pass times out,
       escalate immediately to Z3 (if not already attempted) or the LLM reasoning path
       rather than treating the timeout as "Uncertain" — a timeout is an absence of
       evidence, not evidence of uncertainty.
   v.  The Prover9 input string (both passes) is the `fol` output for this path.

4. **For MCQ and open-ended questions — LLM path:**
   a. Load the `logic-reasoner` LoRA adapter.
   b. LLM reasons from premises to a conclusion via COT inference steps.
   c. LLM generates FOL representations of premises as a separate output.
   d. **Type 1 verifier pass** (controlled by `SolverConfig.type1_verify`, default
      `true`): Load the `verifier` adapter and run a second independent review of the
      LLM's conclusion. If the verifier disagrees with the generator's conclusion, set
      `confidence = 0.4` and note the disagreement in `explanation`. Disable via
      `SolverConfig` only if latency testing shows Type 1 requests are hitting Tier 1
      timeout (≥ 12s elapsed).
5. Pass to Stage 6 for response assembly.

**Stage 6 fields for Type 1:**
- `answer`: Z3/Prover9 result mapping or LLM conclusion
- `explanation`: LLM prose summary (uses `response-assembler` adapter)
- `fol`: Z3 code or Prover9 input for Yes/No/Uncertain; LLM-generated FOL for MCQ/open-ended
- `cot`: inference steps (string-formatted from LLM reasoning trace)
- `premises`: echoed directly from request payload
- `confidence`: 1.0 if Z3/Prover9 solved; verifier-adjusted if `type1_verify=true`;
  LLM self-report otherwise

The Type 1 path shares the same base model as the Type 2 path, differing only in which
LoRA adapter is loaded. Both paths converge at Stage 6.

---

## 3.2 SolverConfig — Runtime Parameters

All scoring-sensitive and compute-sensitive decisions are controlled by a single
`SolverConfig` object. This allows behavior to be tuned after deployment without
changing pipeline code — critical for adapting to the scoring model once it is
confirmed at or after the kick-off workshop.

```json
{
  "confidence_threshold": {
    "mcq":        0.0,
    "yes_no":     0.0,
    "numerical":  0.0,
    "open_ended": 0.0
  },
  "abstain_behavior":        "best_effort",
  "beam_n":                  3,
  "step_retry_limit":        3,
  "trace_budget":            10,
  "repair_budget":           3,
  "type1_enabled":           true,
  "type2_enabled":           true,
  "type1_use_z3":            true,
  "type1_verify":            true,
  "generate_fol":            true,
  "generate_cot":            true,
  "generate_premises":       true,
  "generate_confidence":     true,
  "latency_budget_seconds":  55,
  "timeout_tier1_seconds":   12,
  "timeout_tier2_seconds":   35,
  "seed":                    42,
  "adapter_physics_solver":  "adapters/physics-solver",
  "adapter_verifier":        "adapters/verifier",
  "adapter_logic_reasoner":  "adapters/logic-reasoner",
  "adapter_response_assembler": "adapters/response-assembler"
}
```

**Confidence threshold fields:**
Each answer format has an independent threshold. If the pipeline's computed `confidence`
for a given query is below the threshold for that query's format, the `abstain_behavior`
rule applies.

- Default `0.0` = always emit an answer. **Confirmed correct** — the competition scoring
  model (Q21) is P1 + P2 + P3 with no stated penalty for wrong answers. There is no
  benefit to abstaining.
- If a future rule change introduces penalty scoring, raise thresholds. Optimal threshold
  under penalty scoring: MCQ (k choices, penalty = p × point): answer only if
  P(correct) > p/(p+1). Numerical/open-ended: set threshold 0.6–0.85.

**`abstain_behavior`:**
- `"best_effort"` (default): always emit the best available answer. Correct for the
  current no-penalty scoring regime.
- `"blank"`: emit no `answer` field. Use only if penalty scoring is confirmed.

**Compute / latency controls:**
- Hard latency cap per request is **60 seconds** (Q13). A timeout counts as a failed
  answer. `latency_budget_seconds` defaults to **55** to leave 5 seconds of margin.
- If wall time exceeds `latency_budget_seconds` mid-pipeline, the pipeline falls back
  to emitting the best available partial answer immediately.
- `beam_n`, `step_retry_limit`, `trace_budget`, `repair_budget`: reduce these to trade
  accuracy for lower latency under time pressure.
- `generate_fol / cot / premises / confidence`: set to `false` to skip optional field
  generation when latency is the binding constraint.
- `type1_use_z3`: set to `false` to bypass Z3 and use LLM-only reasoning for Type 1
  (useful if Z3 formalization quality is poor on a given problem distribution).
- `type1_verify`: enables a second `verifier` adapter pass over Type 1 MCQ and
  open-ended answers. Adds approximately one LLM call to the Type 1 path. Default
  `true` — the `verifier` adapter is already loaded and the cost is one call covering
  ~37% of the test set. Disable only if latency testing shows Type 1 requests are
  consistently hitting Tier 1 fallback (≥ 12s elapsed).

**Timeout ladder (#1):** The pipeline enforces three automatic fallback tiers based on
elapsed wall time. Tier thresholds are controlled by `timeout_tier1_seconds` and
`timeout_tier2_seconds`:

| Elapsed time | Action |
|---|---|
| < `timeout_tier1_seconds` (default 12s) | Full pipeline: beam search active, repair loops active |
| ≥ `timeout_tier1_seconds` | **Tier 1 fallback:** disable beam search (use top formula set only, `beam_n=1`); disable repair loops; continue with single-path solve |
| ≥ `timeout_tier2_seconds` (default 35s) | **Tier 2 fallback:** additionally skip optional field generation (`fol`, `cot`, `premises`); emit answer and explanation only |
| ≥ `latency_budget_seconds` (default 55s) | **Hard stop:** emit best available `final_answer` immediately, regardless of trace status |

Tier thresholds should be validated against worst-case LLM call counts. Estimate:
beam_n × trace_budget × avg_llm_call_ms should be well under `timeout_tier1_seconds`
for the expected fast path; the tiers handle pathological problems.

**`seed`:** Passed as `extra_body={"seed": seed}` in each vLLM API request body,
ensuring that the same input always produces the same LLM output. This is required
for live-demonstration reproducibility on Public Test Day (the committee may submit
the same question multiple times). The global `--seed` flag on the vLLM process sets
a server-level default; the per-request field overrides it and is preferred because it
survives server restarts. All non-LLM components (SymPy, Z3, quantulum3, Pint,
scipy.constants, embedding similarity, Wolfram Alpha, Wikipedia API) are already
deterministic and do not require a seed.

**Adapter paths:** Each `adapter_*` field is a local filesystem path passed to vLLM at
startup. All adapters are pre-loaded; the pipeline selects which to activate per stage
via the vLLM `lora_request` parameter at inference time.

---

## 4. Core Data Structures

All pipeline artifacts are JSON. Downstream stages may add fields but must not remove fields
defined in prior stages.

### 4.1 Step Object

Each reasoning step in the trace is one Step Object.

```json
{
  "step_id":              "step_3",
  "goal":                 "Calculate kinetic energy at the bottom of the ramp",
  "type":                 "ENUM[calculation | formula_application | unit_conversion | setup | conclusion]",
  "formula_ids":          ["ENE-002"],
  "input_var": {
    "m":  { "value": 2.0, "unit_symbol": "kg",  "unit_name": "kilograms" },
    "v":  { "value": 5.0, "unit_symbol": "m/s", "unit_name": "meters per second" }
  },
  "output_var": {
    "KE": { "value": 25.0, "unit_symbol": "J", "unit_name": "joules" }
  },
  "step_input":           "KE = 0.5 * m * v^2 = 0.5 * 2.0 * 5.0^2",
  "intermediate_answer":  "25.0 J",
  "thought":              "Applying kinetic energy formula with known mass and velocity.",
  "confidence":           0.95,
  "checkable":            true,
  "status":               "ENUM[OK | WRONG | UNCERTAIN | REPAIRED | null]",
  "verifier_notes":       "",
  "evaluator_response":   [],
  "cot_consistent":       "ENUM[CONSISTENT | INCONSISTENT | NOT_CHECKED | null]"
}
```

**Field rules:**
- `step_id` is sequential (`step_1`, `step_2`, …) with no gaps.
- `input_var` must reference variables that exist in the VSO at the time this step executes.
- `output_var` defines variables written to the VSO after this step completes.
- `intermediate_answer` contains the result only — no reasoning prose.
- `checkable` is set by Stage 3 classification (see Section 5.3). Once set, it persists
  across all subsequent loops and does not need to be re-evaluated.
- `status` is null until populated by Stage 3.
- `cot_consistent` is null until populated by Stage 4 (see Section 5.4).
- Both `unit_symbol` and `unit_name` are required for all dimensional quantities. This
  prevents ambiguity from overloaded symbols (e.g., γ in thermodynamics vs. relativity).

**Step types that do not require a formula:**
- `setup`: extracts given values from the problem text into the VSO. No formula needed.
- `conclusion`: reads the target variable from the VSO and states the final answer. No formula needed.
- `unit_conversion`: may reference a conversion factor rather than a physics formula.

### 4.2 Variable State Object (VSO)

The VSO is a flat dictionary that accumulates all named values produced by completed steps.
It is the single source of truth for all intermediate values throughout the pipeline.

```json
{
  "m":  { "value": 2.0,   "unit_symbol": "kg",    "unit_name": "kilograms",
          "defined_at": "step_1", "updated_at": "step_1" },
  "g":  { "value": 9.81,  "unit_symbol": "m/s^2", "unit_name": "meters per second squared",
          "defined_at": "step_1", "updated_at": "step_1" },
  "h":  { "value": 3.0,   "unit_symbol": "m",     "unit_name": "meters",
          "defined_at": "step_2", "updated_at": "step_2" },
  "PE": { "value": 58.86, "unit_symbol": "J",     "unit_name": "joules",
          "defined_at": "step_3", "updated_at": "step_3" },
  "v":  { "value": 4.2,   "unit_symbol": "m/s",   "unit_name": "meters per second",
          "defined_at": "step_4", "updated_at": "step_5" }
}
```

**`defined_at` vs `updated_at`:**
- `defined_at`: the step that first introduced this variable name to the VSO.
- `updated_at`: the most recent step that wrote a new value to this key.
- These are the same for most variables. They differ when a variable is refined or
  corrected by a later step (e.g., a unit conversion updates `v`'s value in-place).
- Both fields are useful for repair: `defined_at` tells us the origin; `updated_at` tells us
  the most recent authoritative source. When the VSO is rewound for Stage 5 repair, both
  fields are rolled back to their state at FWS entry.

**Variable naming conventions:**
- Variable names are short symbolic identifiers: `m`, `v_0`, `F_net`, `KE`, `T_1`.
- Each entry also carries `unit_name` (long-form English) alongside `unit_symbol`.
- If the same physical quantity type appears multiple times (e.g., two objects' velocities),
  they are distinguished by subscript: `v_train`, `v_bus` (problem-context names) or
  `v_1`, `v_2` (generic). The canonical physical quantity type (e.g., "velocity") is
  recorded separately in the Step Object's `goal` field.

**VSO initialization:** The VSO is created at Stage 0 with all known quantities from the
problem. Constants with universally known values (g = 9.81 m/s², c, h_planck, etc.) are
pre-populated from a constants table, not re-derived each problem.

**During repair (Stage 5):**
- A VSO snapshot is saved at every step boundary during Stage 2+3.
- Stage 5 restores the snapshot at FWS entry before regenerating from that point.
- This prevents a repaired step from reading stale values produced by the discarded
  wrong steps.

### 4.3 Formula Library Entry

The formula library is a curated JSON corpus used for Stage 1 retrieval.

```json
{
  "id":                "CKT-001",
  "topic":             "circuits",
  "subtopic":          "ohms_law",
  "target_quantities": ["V", "I", "R"],
  "canonical_quantity_names": ["voltage", "electric current", "resistance"],
  "text":              "Voltage equals current multiplied by resistance (Ohm's Law).",
  "formula":           "V = I * R",
  "sympy_expr":        "Eq(V, I * R)",
  "tool_dispatch":     "sympy",
  "variables": {
    "V": { "symbol": "V", "name": "voltage",          "unit_symbol": "V",   "unit_name": "volts"  },
    "I": { "symbol": "I", "name": "electric current", "unit_symbol": "A",   "unit_name": "amperes" },
    "R": { "symbol": "R", "name": "resistance",       "unit_symbol": "Ω",   "unit_name": "ohms"   }
  },
  "conditions": [
    "Valid for ohmic (linear) resistors at constant temperature.",
    "Current direction follows conventional current flow (positive to negative externally)."
  ],
  "fol_axiom":   "∀V ∀I ∀R (HasVoltage(V) ∧ HasCurrent(I) ∧ HasResistance(R) → OhmsLaw(V = I * R))",
  "premise_text": "Ohm's Law: V = IR"
}
```

**New fields vs. v0.1:**

- `canonical_quantity_names`: plain-English names for the physical quantities. Used by
  Stage 1 retrieval to normalize problem-specific variable names (e.g., "train_velocity"
  → canonical "speed" / "velocity") before searching the library. This prevents a search
  for "train_velocity" from missing the formula for "v".

- `sympy_expr`: a SymPy-compatible string representation of the formula, used directly
  by the Stage 3 verifier. If present, Stage 3 does not need to parse the `formula` string.

- `tool_dispatch`: indicates which tool can evaluate this formula.
  - `"sympy"`: use SymPy for symbolic/numeric evaluation.
  - `"scipy_<module>"`: dispatch to a specific SciPy function (e.g., `"scipy_integrate"`).
  - `"python_math"`: use Python's `math` module.
  - `"wolfram"`: dispatch to the Wolfram Alpha API for evaluation or constant lookup.
    Use when SymPy cannot solve the expression (transcendental equations, special
    functions). Tool calls are logged and surfaced in `step_input` so they appear in
    the COT output as required by Q7.
  - `"llm"`: no deterministic tool available; fall back to LLM verifier (`verifier` adapter).
  This field means the formula library is also a **tool dispatch table**. A formula entry
  directly tells Stage 3 which tool to use for verification, eliminating per-step guessing.

- `fol_axiom`: a first-order logic string expressing the physical law in domain-predicate
  form. Used by Stage 6 to assemble the `fol` field of the API response. The axiom states
  the general law; Stage 6 instantiates it with the specific objects and values from the
  trace (e.g., replacing generic `m` with `block_A`).

- `premise_text`: a concise, human-readable statement of the law suitable for inclusion
  in the API response `premises` list (e.g., `"Kinetic energy: KE = ½mv²"`). This avoids
  generating premises from scratch in Stage 6; they are pulled from the library at retrieval
  time and carried through the trace.

**Existing formula library sources (to be evaluated):**
- Physics formula libraries in LaTeX (e.g., physics textbook corpora) can be parsed to
  bootstrap the library. NIST CODATA provides constants. Hand-curation is still required
  to add `sympy_expr` and `tool_dispatch` fields.
- The library should be versioned alongside the code. Pre-competition curation of
  mechanics, thermodynamics, electromagnetism, and optics is the minimum target.

### 4.4 Problem Parse Object

Output of Stage 0. Represents the structured interpretation of the raw problem text.

```json
{
  "problem_text":    "A 2 kg block slides down a frictionless ramp from height 3 m...",
  "domains":         ["mechanics"],
  "sub_domains":     ["energy_conservation"],
  "domain_confidence": 0.91,
  "known_quantities": {
    "m": { "value": 2.0, "unit_symbol": "kg", "unit_name": "kilograms" },
    "h": { "value": 3.0, "unit_symbol": "m",  "unit_name": "meters"    }
  },
  "unknown_quantity": "v_final",
  "unknown_unit":     "m/s",
  "step_plan": [
    {
      "step_id":     "step_1",
      "goal":        "Identify known values and target quantity",
      "type":        "setup",
      "input_var":   {},
      "output_var":  { "m": {}, "h": {}, "g": {} },
      "confidence":  0.98
    },
    {
      "step_id":     "step_2",
      "goal":        "Apply conservation of energy: PE_initial = KE_final",
      "type":        "formula_application",
      "input_var":   { "m": {}, "h": {}, "g": {} },
      "output_var":  { "v_final": {} },
      "confidence":  0.91
    }
  ],
  "plan_confidence": 0.94
}
```

**Domain handling:**
- `domains` is a list, not a single string. A problem that involves both thermodynamics
  and mechanics (e.g., a piston problem) will have both listed. Stage 1 retrieval runs
  against all listed domains.
- If the parser cannot confidently classify the domain, it sets `domain_confidence` below
  a threshold (< 0.6) and `domains` to `["unknown"]`. In this case, Stage 1 skips the
  topic-filter step and runs quantity-match and embedding search across the full library.
  This is the graceful degradation for unseen domains.
- Domain classification drives formula library filtering in Stage 1 but does not restrict
  the solver. If a problem involves physics not seen in the training distribution, the
  pipeline will still attempt to retrieve relevant formulas by quantity matching.

### 4.5 Trace Object

The full solution trace produced by Stage 2+3 and modified by Stages 4–5.

```json
{
  "problem_id":         "p_001",
  "formula_path_index": 0,
  "steps":              [ /* ordered list of Step Objects */ ],
  "vso":                { /* Variable State Object, fully populated */ },
  "vso_snapshots":      { "after_step_1": {}, "after_step_2": {}, "..." : {} },
  "final_answer":       "7.67 m/s",
  "final_unit":         "m/s",
  "trace_status":       "ENUM[PASS | FAIL | REPAIRED]"
}
```

- `formula_path_index`: which candidate formula set from Stage 1 was used (0 = top-ranked).
- `vso_snapshots`: VSO state saved after each step, enabling Stage 5 rollback without
  re-running prior steps.

### 4.8 LoRA Adapter Registry

The four adapters share the same base model weights. The registry maps logical role names
to local adapter paths. All entries are provided to vLLM at startup via `--lora-modules`.

| Adapter | Path | Stages that load it | Training signal |
|---|---|---|---|
| `physics-solver` | `adapters/physics-solver` | Stage 2+3 generation, Stage 5 repair | GPT-4 annotated Type 2 solution traces |
| `verifier` | `adapters/verifier` | Stage 3 LLM verification, Stage 4 FWS diagnosis | GPT-4 step-level correctness judgments on training traces |
| `logic-reasoner` | `adapters/logic-reasoner` | Type 1 path — Z3 formalization, MCQ/open-ended reasoning | GPT-4 annotated Type 1 traces + natural-language-to-Z3 pairs |
| `response-assembler` | `adapters/response-assembler` | Stage 6.2 explanation generation | GPT-4 generated explanation examples over completed traces |

**Adapter swapping mechanics:**
- vLLM loads all adapters at startup from paths declared in `SolverConfig`.
- Per-inference-call adapter selection uses the `lora_request` parameter in the
  OpenAI-compatible vLLM API.
- Swap cost is sub-millisecond when adapters are pre-loaded (no disk I/O at request time).
- The base model weights remain resident throughout; GPU memory usage does not change
  between adapter activations.
- The `verifier` adapter is the only one that may be activated mid-step (interleaved with
  the `physics-solver` adapter). The pipeline alternates: generate with `physics-solver`,
  verify with `verifier`, then advance.

**Adapter training notes:**
- Train each adapter independently on its role-specific dataset.
- Use LoRA rank r=16, target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`.
- Training data for all adapters is generated by teacher models (Q6 allowance); disclose
  teacher model in Data Disclosure Document.
- Adapters for the same base model checkpoint are interchangeable at the vLLM layer.

---

### 4.6 Diagnosis Object

Output of Stage 4. Attached to a Trace Object when `trace_status` is `FAIL`.

```json
{
  "global_error_type": "ENUM[E1 | E2 | E3 | E4 | E5 | E6]",
  "fws_index":         2,
  "fws_error_type":    "ENUM[E1 | E2 | E3 | E4 | E5 | E6]",
  "fws_description":   "Model applied formula for inelastic collision instead of elastic.",
  "repair_hint":       "Verify that the collision type assumption matches the problem statement.",
  "step_labels": {
    "step_1": "OK",
    "step_2": "WRONG",
    "step_3": "OK"
  },
  "cot_issues": [
    {
      "step_id":     "step_3",
      "description": "Reasoning states energy is conserved but prior step introduced friction."
    }
  ]
}
```

### 4.7 API Response Object

The final JSON object returned by the pipeline for every query. Matches the competition
submission format exactly.

```json
{
  // ── Required ────────────────────────────────────────────────────────────────
  "answer":      "7.67 m/s",
  "explanation": "Using conservation of energy, the block's initial potential energy
                  (mgh = 58.86 J) converts entirely to kinetic energy at the bottom.
                  Solving KE = ½mv² for v gives v = √(2gh) ≈ 7.67 m/s.",

  // ── Optional (encouraged — contribute to reasoning-depth score) ─────────────
  "fol": [
    "∀m ∀h ∀g (HasMass(m) ∧ HasHeight(h) ∧ HasGravity(g) → PotentialEnergy(m * g * h))",
    "∀m ∀v (HasMass(m) ∧ HasSpeed(v) → KineticEnergy(0.5 * m * v**2))",
    "∀PE ∀KE (Frictionless → PE = KE)"
  ],
  "cot": [
    "Step 1: Extract known quantities: mass m = 2.0 kg, height h = 3.0 m, g = 9.81 m/s².",
    "Step 2: Compute initial potential energy: PE = mgh = 2.0 × 9.81 × 3.0 = 58.86 J.",
    "Step 3: Apply energy conservation (frictionless): KE_final = PE_initial = 58.86 J.",
    "Step 4: Solve for final speed: v = √(2 × KE / m) = √(2 × 58.86 / 2.0) ≈ 7.67 m/s."
  ],
  "premises": [
    "Conservation of mechanical energy: PE_initial + KE_initial = PE_final + KE_final",
    "Gravitational potential energy: PE = mgh",
    "Kinetic energy: KE = ½mv²"
  ],
  "confidence": 0.92
}
```

**Field responsibilities (which stage produces each field):**

| Field | Required | Produced by | Source |
|---|---|---|---|
| `answer` | Yes | Stage 6 | `Trace.final_answer` |
| `explanation` | Yes | Stage 6 | LLM call over completed trace |
| `fol` | No | Stage 6 | `fol_axiom` fields from retrieved formula entries, instantiated with trace variables |
| `cot` | No | Stage 6 | `Step.thought` array (CONSISTENT steps only) |
| `premises` | No | Stage 6 | `premise_text` fields from retrieved formula entries, deduplicated |
| `confidence` | No | Stage 6 | Aggregated from per-step `Step.confidence` values |

**Answer format note:** The competition example shows `"answer": "B"` (a letter, implying
multiple choice), while earlier competition context describes numerical physics answers.
The answer format for Dataset Type 2 is to be confirmed at the kick-off workshop (see Open
Question #11). The pipeline should be capable of producing both: a raw numeric value with
unit for numerical problems, or a letter selection for MCQ problems.

---

## 5. Stage Specifications

### Stage 0: Problem Parsing

**Purpose:** Convert the raw problem text into a structured Problem Parse Object. Identify
known quantities, the target unknown, the physics domain(s), and a step-by-step plan.

**Input:**
- Raw problem text (string)

#### Stage 0.1 — Quantity Extraction (Hybrid: Programmatic → LLM Fallback)

Quantity extraction is split into two passes. The programmatic pass runs first and is
deterministic; the LLM pass handles only what the programmatic pass could not resolve.

---

**Pass 1 — Programmatic extraction**

Two libraries form the programmatic stack:

**`quantulum3`** (`pip install quantulum3`)
> Extracts (value, unit) pairs from natural language using NLP entity recognition.
> Returns a list of `Quantity` objects with a value, a `Unit` (with name and entity type),
> a confidence score, and the character span in the source text.

```python
from quantulum3 import parser as q3

quants = q3.parse("A 2 kg block slides down a 3 m ramp; g = 9.81 m/s².")
# → [Quantity(2.0, 'kilogram'), Quantity(3.0, 'metre'), Quantity(9.81, 'metre / second²')]
```

**`Pint`** (`pip install pint`)
> After quantulum3 extracts a raw unit string, Pint validates and normalises it to a
> canonical form. Pint also provides both the short symbol and the long English name,
> which are required by the VSO and Step Object schemas.

```python
from pint import UnitRegistry
ureg = UnitRegistry()

q = ureg.Quantity(2.0, "kilogram")
q.units            # → <Unit('kilogram')>
str(q.units)       # → 'kilogram'        (unit_name)
f"{q.units:~P}"    # → 'kg'              (unit_symbol, compact format)
```

**Programmatic pass workflow per sentence:**
1. Run `quantulum3.parser.parse(sentence)`.
2. For each returned `Quantity`, pass the raw unit string to `Pint` for validation and
   normalisation. If Pint raises `UndefinedUnitError`, mark this quantity as
   **unresolved** and queue it for the LLM pass.
3. For each successfully validated quantity, record:
   `{ value, unit_symbol (Pint compact), unit_name (Pint long), source_span, q3_confidence }`
4. Mark quantities where `q3_confidence < 0.75` as **low-confidence** — queue for LLM
   verification even if Pint accepted the unit.

**Known limitations of the programmatic pass (always require LLM handling):**
- **Variable name assignment.** `quantulum3` returns `(2.0, 'kilogram')` but not
  `m = 2.0 kg`. Naming is inherently contextual and is always handled in Pass 2.
- **Implied zero quantities.** "starts from rest" → `v_0 = 0 m/s` is never extracted
  programmatically.
- **Ambiguous or compound units.** `"eV"`, `"parsec"`, `"rpm"`, `"dBm"` may be outside
  quantulum3's entity dictionary and fail Pint validation.
- **Quantities stated as fractions or expressions.** `"½ mv²"` in the problem text is not
  a stated given; it is a formula reference and should not be extracted as a quantity.
- **The target unknown.** The question ask (`"find the velocity"`) is not a quantity
  extraction target; it is identified in Pass 2.

---

**Pass 2 — LLM extraction and enrichment**

The LLM receives:
- The original problem text
- The list of successfully extracted quantities from Pass 1 (so it does not re-extract
  what is already known)
- The list of unresolved and low-confidence spans flagged in Pass 1

The LLM is asked to:
1. Assign a symbolic variable name to each programmatically extracted quantity
   (e.g., `m` for mass, `v_0` for initial velocity).
2. Extract any quantities the programmatic pass missed (implied zeros, non-standard units,
   quantities stated in words rather than numerals like "one kilogram").
3. Identify the target unknown quantity and its expected unit.
4. Confirm or correct any low-confidence programmatic extractions.

The LLM output is merged with Pass 1 results. Pass 1 values take precedence for value and
unit wherever both agree; LLM values are used where Pass 1 was absent or flagged.

**Structured output format for the LLM pass:**
```json
{
  "known_quantities": {
    "m":   { "value": 2.0,  "unit_symbol": "kg",    "unit_name": "kilograms",
             "source": "programmatic" },
    "h":   { "value": 3.0,  "unit_symbol": "m",     "unit_name": "metres",
             "source": "programmatic" },
    "v_0": { "value": 0.0,  "unit_symbol": "m/s",   "unit_name": "metres per second",
             "source": "llm_implied" }
  },
  "unknown_quantity": "v_final",
  "unknown_unit_symbol": "m/s",
  "unknown_unit_name": "metres per second"
}
```

The `source` field (`"programmatic"`, `"llm_explicit"`, `"llm_implied"`) is recorded for
debugging and data quality analysis but is not required by downstream stages.

---

**Process (remainder of Stage 0):**
1. Run the hybrid quantity extraction sub-process (Stage 0.1) to populate `known_quantities`
   and `unknown_quantity`.
2. **Domain classification — keyword-first (programmatic), LLM fallback only.**
   Run a keyword/phrase lookup table over the problem text:
   ```python
   DOMAIN_KEYWORDS = {
       # Circuits and electrostatics are listed first — primary Type 2 domain (#2/#3).
       # Formula library curation should prioritize these subtopics before others.
       "circuits":         ["resistor", "capacitor", "inductor", "circuit", "series",
                            "parallel", "impedance", "Kirchhoff", "KVL", "KCL",
                            "ohm", "ammeter", "voltmeter", "node", "loop", "branch",
                            "RC", "RL", "RLC", "charge", "discharge", "time constant"],
       "electrostatics":   ["charge", "Coulomb", "electric field", "electric potential",
                            "dielectric", "permittivity", "capacitance", "Gauss",
                            "flux", "field line", "point charge", "dipole", "shield"],
       "mechanics":        ["velocity", "acceleration", "force", "momentum", "torque",
                            "friction", "gravity", "Newton", "kinetic", "potential"],
       "thermodynamics":   ["temperature", "entropy", "heat", "pressure", "Carnot",
                            "isothermal", "adiabatic", "Boltzmann", "ideal gas"],
       "electromagnetism": ["voltage", "current", "resistance", "magnetic", "Ohm",
                            "flux", "inductance", "Faraday", "Lenz", "solenoid"],
       "optics":           ["wavelength", "frequency", "refraction", "reflection",
                            "lens", "mirror", "diffraction", "photon", "Snell"],
       "modern_physics":   ["quantum", "relativity", "photon", "electron", "nuclear",
                            "radioactive", "Planck", "de Broglie", "Heisenberg"],
   }
   ```
   - Score each domain by keyword hit count. Assign `domain_confidence` proportional to
     the top score normalised against all hits.
   - If `domain_confidence` ≥ 0.6: assign matching domains programmatically. No LLM call.
   - If `domain_confidence` < 0.6: invoke the LLM to classify the domain. Set
     `domains = ["unknown"]` if the LLM is also uncertain.
3. Generate the step plan via LLM: ordered Step Objects with `goal`, `type`, `input_var`
   (keys only, values empty), and `output_var` (keys only). Values are not filled at this stage.

   **Step plan structural validation (#10):** After the LLM returns the plan, validate
   it programmatically before proceeding. Do not rely on the LLM to produce a valid plan.
   Check all three conditions:
   ```python
   VALID_STEP_TYPES = {"calculation", "formula_application", "unit_conversion",
                       "setup", "conclusion"}

   def validate_step_plan(steps, known_quantities):
       available = set(known_quantities.keys())
       for i, step in enumerate(steps):
           # (a) Step type must be from the known enum
           assert step["type"] in VALID_STEP_TYPES, f"Unknown step type: {step['type']}"
           # (b) Every input_var key must resolve to a known quantity or a prior step's output
           for var in step["input_var"]:
               assert var in available, f"Step {i}: input '{var}' not yet defined"
           # (c) Register this step's outputs so later steps can reference them
           available.update(step["output_var"].keys())
       # (d) No dependency cycles — guaranteed if (b) passes in order, since each step
       #     can only reference variables defined in strictly earlier steps.
   ```
   If validation fails, re-request the step plan with the specific violation described.
   After 3 failed attempts, activate **direct-LLM fallback mode**: skip Stages 1–5
   entirely and invoke the `physics-solver` adapter directly with the full problem text,
   asking it to produce an answer, explanation, and confidence in a single pass. This
   produces a lower-quality but non-empty response — preferable to a hard crash, which
   counts the same as a timeout (zero points). Set `confidence = 0.2` and
   `trace_status = FAIL` to signal that the structured pipeline was bypassed.

4. Initialize the VSO with all `known_quantities` from Stage 0.1 and any universally known
   constants required by the plan. **Constants are sourced from `scipy.constants`**, not a
   hand-curated table:
   ```python
   import scipy.constants as const
   PHYSICS_CONSTANTS = {
       "g":        {"value": const.g,  "unit_symbol": "m/s^2", "unit_name": "metres per second squared"},
       "c":        {"value": const.c,  "unit_symbol": "m/s",   "unit_name": "metres per second"},
       "h":        {"value": const.h,  "unit_symbol": "J·s",   "unit_name": "joule seconds"},
       "e":        {"value": const.e,  "unit_symbol": "C",     "unit_name": "coulombs"},
       "m_e":      {"value": const.m_e,"unit_symbol": "kg",    "unit_name": "kilograms"},
       "N_A":      {"value": const.N_A,"unit_symbol": "mol^-1","unit_name": "per mole"},
       "k_B":      {"value": const.k,  "unit_symbol": "J/K",   "unit_name": "joules per kelvin"},
   }
   ```
   Constants are pre-loaded at service startup, not re-fetched per query. They appear in
   the VSO with `defined_at: "constants_table"`.

**Acceptance Criteria:**
- All quantities mentioned numerically in the problem text appear in `known_quantities`.
- All implied-zero quantities identified by the LLM pass appear in `known_quantities`.
- `unknown_quantity` is non-empty.
- The step plan is non-empty; the last step's `output_var` contains `unknown_quantity`.
- Every step's `input_var` keys reference either a quantity in `known_quantities` / the
  constants table, or the `output_var` of an earlier step in the plan.
- Every quantity has both `unit_symbol` and `unit_name`.

**Retry Behavior:**
- If a required component is missing, re-request up to 3 times with the specific gap identified.
- If after 3 retries the plan is still incomplete, activate direct-LLM fallback mode
  (see step plan structural validation above). Do not crash the request.

**Output:** Problem Parse Object

---

### Stage 1: Formula & Premise Retrieval

**Purpose:** For each `formula_application` step in the plan, retrieve one or more candidate
formula entries. Return top-N ranked formula sets that define N alternative solution paths.

**Input:**
- Problem Parse Object

**Process:**
1. **Topic filter:** For each `formula_application` step, select formula library entries
   whose `topic` or `subtopic` matches any entry in `domains` / `sub_domains`.
   - If `domains` is `["unknown"]`, skip this filter and proceed to quantity match.

2. **Variable canonicalization — regex-first, embedding second, LLM last.**
   Map problem-specific variable names to canonical physical quantity names using a
   three-tier cascade. No LLM call unless the first two tiers fail:

   **Tier 1 — Regex lookup table (deterministic):**
   ```python
   CANONICAL_MAP = [
       (r".*(velocity|speed).*",      "velocity"),
       (r".*(mass).*",                  "mass"),
       (r".*(weight).*",               "force"),   # weight = mg, unit: N not kg
       (r".*(height|altitude).*",     "displacement"),
       (r".*(distance|displacement|position).*", "displacement"),
       (r".*(force|thrust).*",        "force"),
       (r".*(temperature|thermal).*", "temperature"),
       (r".*(pressure).*",            "pressure"),
       (r".*(energy|work).*",         "energy"),
       (r".*(current|ampere).*",      "electric_current"),
       (r".*(voltage|potential).*",   "electric_potential"),
       (r".*(resistance).*",          "resistance"),
       (r".*(frequency|wavelength).*","frequency"),
       (r".*(angle|theta).*",         "angle"),
       (r".*(time|duration).*",       "time"),
   ]
   ```

   **Tier 2 — Embedding similarity:** If no regex matches, compute cosine similarity
   between the variable name embedding and `canonical_quantity_names` embeddings in
   the formula library. Accept if similarity > 0.8.

   **Tier 3 — LLM:** Only invoked if Tiers 1 and 2 both fail to canonicalize a name.

   **Collision detection (#9):** After all variable names in the step plan are
   canonicalized, check whether any two *different* problem variables resolved to the
   same canonical name:
   ```python
   canonical_to_original = {}
   for orig_name, canonical in canonical_map.items():
       if canonical in canonical_to_original:
           # Two different problem variables → same canonical: escalate to LLM
           escalate_to_llm(orig_name, canonical_to_original[canonical], canonical)
       else:
           canonical_to_original[canonical] = orig_name
   ```
   Common collision cases to watch for in circuits/electrostatics:
   - `"voltage"` vs `"EMF"` vs `"potential difference"` — all currently map to
     `electric_potential` but may be distinct quantities in the problem.
   - `"height"` vs `"altitude"` vs `"displacement"` — may be distinct positions.
   Note: `"mass"` and `"weight"` have been separated in `CANONICAL_MAP` ("mass" →
   `mass`; "weight" → `force`) so they no longer collide at the regex tier.
   When a collision is detected, the LLM pass receives both original names and returns
   distinct canonical names for each. This prevents silent variable overwriting in the VSO.

3. **Quantity match:** Score remaining candidates by overlap between their
   `canonical_quantity_names` and the canonicalized variable names in the step plan.

4. **Embedding fallback:** If fewer than 2 candidates score > 0.8, perform embedding
   similarity search over formula `text` fields using the problem text as query.

5. **Wikipedia API fallback:** If the embedding search also returns no candidates above
   threshold (rare — indicates a formula outside the library), retrieve a Wikipedia
   article for the step's `goal` text using the following cache-first order:
   ```
   1. Check request-scoped session cache (dict keyed by query string)
   2. Check offline pre-fetched corpus (wikipedia_cache.json, built from training set)
   3. Live Wikipedia API call (only if both caches miss)
   4. On live call success: store result in session cache for the rest of this request
   ```
   Extract formula references from the article text and attempt a second-pass quantity
   match. This is a best-effort fallback; a miss at this stage falls through to Step 6.
   On any live API failure (timeout, HTTP error), proceed directly to Step 6 without
   raising an exception.

   **Wolfram Alpha and Wikipedia reliability policy (#6):** Both are live external
   services and can fail during a 1–2 day evaluation window (rate limits, outages,
   latency spikes). Apply the following rules:
   - **Session cache:** Cache all Wolfram Alpha and Wikipedia responses in a request-scoped
     dict keyed by query string. If the same query is issued twice in one pipeline run,
     return the cached result immediately without a second network call.
   - **Wolfram Alpha scope restriction:** Only invoke Wolfram Alpha for formula library
     entries explicitly tagged `tool_dispatch: "wolfram"`. Do not use it as a general
     fallback for arbitrary expressions. This limits exposure to rate limits.
   - **Offline fallback corpus:** Maintain a local pre-fetched cache file
     (`wolfram_cache.json`, `wikipedia_cache.json`) containing responses for the most
     common queries expected from the training set. If a live API call fails (timeout,
     HTTP error, rate limit), check the offline cache before falling back to SymPy-only
     verification. Build and disclose this corpus in the Data Disclosure Document.
   - **Failure handling:** If both live API and offline cache miss, log the failure and
     continue with SymPy-only or LLM-only verification for that step. Do not raise an
     exception.

6. **Path construction:** Group retrieved formulas by step into N candidate formula sets,
   where each set covers all `formula_application` steps with a different formula option
   for ambiguous steps. Return up to N = 3 sets, ranked by aggregate score.

7. **Steps with no formula match:** If no formula is retrieved for a `formula_application`
   step, log a warning but do not block. Stage 2 will attempt to solve the step using the
   LLM's own knowledge and available tools. The absence of a retrieved formula is noted
   in the step's `formula_ids` as an empty list.

**Acceptance Criteria:**
- At least one formula set is returned (even if incomplete).
- Each returned set includes a `retrieval_confidence` score.

**Output:** List of ranked formula sets (each set is a dict mapping `step_id` → Formula Entry)

---

### Stage 2+3: Step-by-Step Generation and Calibration

These two stages are merged into a single interleaved loop. Verification is performed
immediately after each step is generated, not in a separate batch pass. This prevents
wasted computation on steps already known to be downstream of an error.

**Purpose:** Execute the step plan one step at a time. After each step is generated,
immediately verify it before proceeding to the next.

**Input:**
- Problem Parse Object
- One candidate formula set from Stage 1 (the current path being attempted)
- Initialized VSO with snapshots dict

**Process (per step):**
1. Retrieve `input_var` values from the current VSO.
2. Provide to the solver: the step `goal`, `type`, relevant formula (if any), and
   `input_var` values.
3. Solver returns: `thought`, `step_input`, `intermediate_answer`, `output_var` values.
4. **Immediate verification (Stage 3 logic):**
   a. Classify the step as `checkable` or `not-checkable` (see rules below).
      This classification is stored on the Step Object and **persists** — it is not
      re-evaluated on subsequent loops.
   b. If `checkable`, run the verifier against the `step_input` and `input_var` values.
   c. Assign: `CORRECT`, `INCORRECT`, or `UNKNOWN`.
   d. If `INCORRECT`:
      - Log the error. Increment the step's local retry counter.
      - If retry counter < step retry limit (default: 3): re-invoke the solver for this
        step only, passing the verifier's correction as additional context. Go to step 3.
      - If retry counter = step retry limit: mark step as `WRONG`. Do not proceed to
        the next step. The trace is marked `FAIL` and passed to Stage 4.
5. If `CORRECT` or `UNKNOWN` or `not-checkable`:
   a. Write `output_var` values to the VSO.
   b. Save a VSO snapshot keyed by `step_id`.
   c. Append the completed Step Object to the trace.
   d. Proceed to the next step.
6. After all steps complete, set `final_answer` = last step's `intermediate_answer`.
   If ground truth is available, check `final_answer`; mark trace `PASS` or `FAIL`.

**Checkable classification rules:**
- `checkable = true` if: step type is `calculation`, `formula_application`, or
  `unit_conversion` AND `formula_ids` is non-empty AND all `input_var` values are present
  in the VSO.
- `checkable = false` for: `setup`, `conclusion`, and any step where `formula_ids` is empty.
- Once classified, the `checkable` flag is stored on the Step Object and not re-evaluated.

**Step confidence — derived from verifier outcome, not LLM self-report:**
LLM-reported confidence values are poorly calibrated and must not be used directly.
After verification, overwrite `step.confidence` with a verifier-derived value:

| Verifier result | Assigned confidence |
|---|---|
| SymPy or Wolfram: CORRECT | 1.0 |
| SymPy or Wolfram: CORRECT (after repair) | 0.6 |
| LLM verifier (`verifier` adapter): CORRECT | 0.8 |
| LLM verifier: UNCERTAIN | 0.5 |
| Not checkable (`setup` / `conclusion`) | 0.9 |
| Not checkable (`unit_conversion` with no formula) | 0.85 |

**Tool call visibility (Q7 requirement):**
Any tool invoked during a step (SymPy, Wolfram Alpha, Z3, Python execution) must be
recorded in `step_input` so it appears in the COT output. Format:
```
"step_input": "KE = 0.5 * m * v**2 = 0.5 * 2.0 * 5.0**2  [SymPy verified: 25.0 J ✓]"
```
Tool calls are not hidden in internal logs — they are part of the reasoning record that
evaluators inspect for P3 (reasoning depth) scoring.

**Retry budget:**
- Each step has its own retry counter (default limit: 3 attempts).
- The total budget across the whole trace is 10 solver invocations. If this budget is
  exhausted before the trace completes, mark the trace `FAIL` and proceed to Stage 4.
  This prevents a pathological problem from consuming unlimited compute.

**COT consistency check (Stage 4 integration point):**
When the `final_answer` is correct but the trace has low-confidence steps or UNCERTAIN
labels, the `cot_consistent` field on each step should be evaluated (see Stage 4). A
correct answer with inconsistent COT reasoning is still a valid competition answer but
should be flagged to avoid using it as training data.

**Formatting Rules (strict):**
- `intermediate_answer` contains result value and unit only — no reasoning prose.
- `step_input` contains the expression or operation, plus any tool call annotations.
- The last step's `intermediate_answer` must exactly match `final_answer`.

**Output:** Trace Object (steps labeled with verifier status; trace_status = PASS or FAIL)

---

### Stage 4: Error Structuring and Diagnosis

**Purpose:** For traces that remain `FAIL` after Stage 2+3, identify the nature of the
failure, pinpoint the First Wrong Step (FWS), and flag any COT reasoning inconsistencies
even in steps whose computed answers are correct.

**This stage is only invoked when `trace_status` is `FAIL`.**

**Input:**
- Trace Object (after Stage 2+3)
- Original problem text

**Process:**
1. **FWS identification — programmatic first, LLM second.**
   Do not invoke the LLM to find the FWS for steps that were already verified by a
   deterministic tool:
   - **Checkable steps**: The FWS is the first step where the Stage 3 verifier returned
     `INCORRECT` (SymPy or Wolfram Alpha). This is already recorded on the Step Object.
     No LLM call is needed — scan `step.status` for the first `WRONG` entry.
   - **Non-checkable steps** (`setup`, `conclusion`, formula-free): The LLM (`verifier`
     adapter) judges each step. Apply the propagation rule: a step that is locally valid
     given inherited incorrect inputs stays `OK`; it becomes `WRONG` only if it introduces
     a new independent error.
   - Set `fws_index` to the index of the earliest `WRONG` step found by either method.
     If no step can be stably labelled `WRONG`, set `fws_index` to null.

2. **Error type classification — programmatic for E4 and E5, LLM for E1/E2/E3/E6.**
   - **E4 (Algebraic/Numerical Computation Error)**: detected programmatically when
     SymPy/Wolfram confirms the formula is structurally correct (`sympy_expr` validates)
     but the numeric value in `intermediate_answer` does not match the tool's result.
     No LLM required.
   - **E5 (Final Answer Extraction Error)**: detected programmatically when all
     intermediate steps pass verification but `Trace.final_answer` does not match the
     last step's `intermediate_answer` (string/numeric comparison with unit normalisation).
     No LLM required.
   - **E1, E2, E3, E6**: require semantic understanding of the problem and are classified
     by the LLM (`verifier` adapter).
   - Assign global `error_type` = the type of the FWS error.

3. **Propagation rule (algorithmic)**: traverse steps in order. If a step's `input_var`
   values were inherited from a `WRONG` step and the step's own computation is locally
   valid given those inputs, mark it `OK` (not `WRONG`). This is a dependency-graph
   traversal, not an LLM task.

4. Produce `fws_description` and `repair_hint` via the LLM (`verifier` adapter), using
   the FWS step and error type as context.

5. **COT consistency check** (applies to both PASS and FAIL traces with low-confidence steps):
   For each step, verify that the `thought` field is logically consistent with the step's
   `input_var`, `step_input`, and `intermediate_answer`. A step can have the correct numeric
   answer while containing flawed reasoning (e.g., correctly computing 25 J but stating an
   incorrect physical interpretation). Label each step's `cot_consistent` field accordingly.
   - Steps flagged `INCONSISTENT` are noted in `cot_issues` in the Diagnosis Object.
   - This check is particularly valuable for training data curation: traces with PASS
     answers but INCONSISTENT COT reasoning should not be used as positive training examples.

**Error Taxonomy (6 categories):**

| Code | Name | Description |
|------|------|-------------|
| `E1` | Problem Misinterpretation | Model solves for wrong quantity or ignores a stated constraint. |
| `E2` | Formula / Physical Law Error | Model understands the goal but selects or sets up the wrong formula. |
| `E3` | Conceptual / Logical Inference Error | Setup is correct but model applies a theorem, principle, or case split incorrectly. |
| `E4` | Algebraic / Numerical Computation Error | Method is correct; a local arithmetic or algebraic step produces the wrong value. |
| `E5` | Final Answer Extraction Error | Reasoning is correct but the final value is mis-extracted, mis-converted, or mis-formatted. |
| `E6` | Incomplete Reasoning | Solution path is abandoned before reaching the target quantity. |

**Acceptance Criteria:**
- Every residual failure receives exactly one global error type.
- Every step receives exactly one label (`OK`, `WRONG`, `UNCERTAIN`).
- Every step receives a `cot_consistent` label if the COT check was run.
- The FWS is the earliest `WRONG` step; no earlier `WRONG` step is skipped.

**Output:** Diagnosis Object, attached to the Trace Object.

---

### Stage 5: First-Wrong-Step Repair

**Purpose:** Repair the trace by preserving all correct steps before the FWS and
regenerating from the FWS onward, guided by the diagnosis.

**This stage is only invoked when Stage 4 produces a valid `fws_index`.**

**Input:**
- Trace Object (with diagnosis attached)
- VSO snapshot at FWS entry

**Process:**
1. Extract the **stable correct prefix**: all steps with index < `fws_index`.
2. Restore the VSO to the snapshot at FWS entry.
3. Present to the repair agent:
   - The problem text
   - The stable correct prefix (read-only context)
   - The VSO at FWS entry
   - `fws_index` ("the solution first becomes wrong at step N")
   - `fws_description` and `repair_hint`
   - `global_error_type`
   - Target answer is **not** provided (Answer-Free mode for production)
4. Repair agent regenerates the FWS step and all subsequent steps.
5. Append the new partial trace to the stable prefix.
6. The repaired trace re-enters Stage 2+3 (starting at the repaired step) for verification.

**Repair Conditions (two modes):**

| Mode | Information Provided | When Used |
|------|----------------------|-----------|
| `Answer-Free` | Stable prefix + FWS location + error type + description + hint | Production |
| `Answer-Aware` | Answer-Free info + target answer | Evaluation / analysis only |

**Acceptance Criteria for the repaired step:**
- Repaired step's `output_var` keys match the original FWS's `output_var` keys.
- Repaired step's `intermediate_answer` differs from the original.
- Repaired continuation reaches a `final_answer`.

**Repair budget:** 3 FWS repair attempts per candidate path. If exhausted, the pipeline
advances to the next candidate path (next formula set from Stage 1). If all candidate paths
are exhausted, the best-confidence trace is forwarded to Stage 6.

**Adapter usage in Stage 5:** The repair agent loads the `physics-solver` adapter (same
as generation). The verification of the repaired steps re-enters Stage 2+3 and loads the
`verifier` adapter for the LLM verification sub-step. This adapter alternation within a
single query is the practical realisation of the student-student paradigm: the generator
role and the verifier role use separately trained adapter weights on the same base model,
providing role independence without loading a second model into memory.

**Output:** Repaired Trace Object (re-enters Stage 2+3 at the repaired step)

---

### Stage 6: Response Assembly

**Purpose:** Convert the accepted Trace Object into the competition API response format
(Section 4.7). Produce all required fields and as many optional fields as possible, since
optional fields contribute directly to the reasoning-depth score.

**Input:**
- Accepted Trace Object (`trace_status = PASS`, `REPAIRED`, or best available)
- Retrieved formula entries used during Stage 1 (carried through the trace via `formula_ids`)

**Sub-processes (run in parallel where independent):**

#### 6.1 — `answer` (Required)

Extract `Trace.final_answer`. Apply competition-specific formatting:
- Numeric answers: round to the significant figures implied by the problem's given data.
  Include unit in the string (e.g., `"7.67 m/s"`).
- MCQ answers (if Dataset Type 2 uses multiple choice): extract the selected letter from
  the conclusion step's `intermediate_answer`.
- If `trace_status = FAIL`, still emit the best available `final_answer` with a note in
  the `explanation`.

**Answer normalization (#7):** To prevent compounding rounding errors and unit
formatting inconsistencies from silently costing P1 points:
- Throughout the VSO and all Step Objects, store numeric quantities as full-precision
  floats with unit stored separately (`value: float`, `unit_symbol: str`). Never store
  a pre-rounded string as an intermediate value.
- Round only at this stage (6.1), not during computation. Use Python's `round()` with
  significant-figure logic based on the input data's least-precise value.
- Accept both Unicode (Ω, μ, °) and LaTeX (`\Omega`, `\mu`, `^\circ`) unit notation
  equivalents as correct matches, per Q20. The answer string emitted should prefer
  Unicode for readability unless the problem text uses LaTeX throughout.

**Optional field suppression on FAIL (#8):** When `trace_status = FAIL` (i.e., the
pipeline could not produce a verified answer), suppress or soften the optional fields
rather than emitting trace-backed content that may contradict the answer:

| `trace_status` | `fol` | `cot` | `premises` |
|---|---|---|---|
| `PASS` or `REPAIRED` | Emit normally | Emit normally | Emit normally |
| `FAIL` | Omit (set to `[]`) | Omit (set to `[]`) | Emit (always safe — echoed from input) |

A confidently wrong explanation or COT chain can hurt P2/P3 human review scores even
when P1 is already lost. Only emit `fol` and `cot` when they are genuinely backed by
a verified trace. The `explanation` field (required) should note the verification
failure plainly when `trace_status = FAIL`.

#### 6.2 — `explanation` (Required)

Generate a concise prose paragraph (3–6 sentences) that summarises the solution.

**Input to the LLM call:**
- Problem text
- Step plan goals (the `goal` field of each step)
- `final_answer`
- `global_error_type` from diagnosis if the trace was repaired (optional, for context)

**Acceptance criteria:**
- States the physical principle(s) applied.
- States the intermediate values that were key to the solution.
- States the final answer and its unit.
- Does not reproduce the full step-by-step arithmetic (that belongs in `cot`).
- Must not mention internal pipeline concepts (Step Object, VSO, etc.).

#### 6.3 — `fol` (Optional)

Assemble a list of first-order logic statements that formally represent the reasoning chain.

**For Type 2 (physics):**
1. Collect the `fol_axiom` strings from every Formula Library entry referenced in the
   trace (via `step.formula_ids`). Deduplicate.
2. Append constraint assertions from the problem setup (e.g., `"Frictionless(ramp)"`)
   derived from Stage 0 extraction.
3. Optionally instantiate axioms with problem-specific object names via the
   `response-assembler` adapter (controlled by `SolverConfig.generate_fol`).
4. Return ordered: domain axioms first, then constraint assertions.
**Fallback:** If the formula library has no `fol_axiom` for a formula, the
`response-assembler` adapter generates an FOL statement from the formula's `text` field.
Prefer curated `fol_axiom` entries.

**For Type 1 (logic):**
The Z3 code generated during the Type 1 reasoning path **is** the FOL output — it is
formally verified and directly auditable. No additional generation step is needed.
For MCQ/open-ended Type 1 problems where Z3 was not used, the `logic-reasoner` adapter
generates FOL representations of the premises.

#### 6.4 — `cot` (Optional) — **No LLM call. Pure string formatting.**

Produce an ordered array of step-description strings by formatting existing Step Object
fields. No additional LLM inference is required.

```python
cot = []
for i, step in enumerate(trace.steps):
    if step.cot_consistent == "INCONSISTENT":
        # Replace with a placeholder — silent removal creates unexplained gaps (#11)
        cot.append(
            f"Step {i+1}: {step.goal} "
            f"[verification note: reasoning flagged inconsistent — result carried forward]"
        )
    else:
        cot.append(
            f"Step {i+1}: {step.goal} — {step.thought} Result: {step.intermediate_answer}"
        )
```

Steps flagged `INCONSISTENT` are replaced with a visible placeholder rather than
silently removed. Visible gaps are better than unexplained jumps for P3 evaluators.
Tool call annotations embedded in `step.thought` or `step_input` appear naturally in
the output, satisfying Q7's visibility requirement. Return as a JSON array (not a
concatenated string).

#### 6.5 — `premises` (Optional) — **No LLM call. Collect and deduplicate.**

```python
premises = list(dict.fromkeys(                          # deduplicate, preserve order
    entry.premise_text
    for step in trace.steps
    for entry in formula_library.lookup(step.formula_ids)
))
premises += [c for c in stage0_constraints if c not in premises]
```

Laws appear first (from formula library), then problem-stated constraints (from Stage 0
extraction, e.g., `"Surface is frictionless"`, `"Collision is perfectly elastic"`).

#### 6.6 — `confidence` (Optional) — **Derived from verifier outcomes, not LLM self-report.**

Step confidence values were written by the verifier (Stage 2+3), not by the generator.
Aggregate using geometric mean — this penalises a single very-low-confidence step more
heavily than arithmetic mean:

```python
import math
vals = [s.confidence for s in trace.steps if s.status != "WRONG"]
# Guard: if every step is WRONG (total failure), vals is empty → return minimum
if not vals:
    confidence = 0.1   # consistent with hard-stop confidence in the timeout ladder
else:
    # Floor at 1e-6 to prevent math.log(0) crash on zero-confidence steps (#16)
    vals = [max(v, 1e-6) for v in vals]
    confidence = round(math.exp(sum(math.log(v) for v in vals) / len(vals)), 2)
    if any(s.status == "WRONG" for s in trace.steps):
        confidence = min(confidence, 0.5)
    if trace.trace_status == "FAIL":
        confidence = min(confidence, 0.3)
```

**Output:** API Response Object (schema in Section 4.7)

---

## 6. Variable State Management

See Section 4.2 for the full VSO schema. Key behaviors:

- **Persistence:** A variable set in Step 1 is available at Step 8 without Step 7 needing
  to re-declare it. All steps look up `input_var` from the VSO by key.
- **Update tracking:** `defined_at` records origin; `updated_at` records most recent write.
  These differ when a variable is refined (e.g., a velocity updated after unit conversion).
- **Rollback:** VSO snapshots are saved at every step boundary. Stage 5 restores the snapshot
  at FWS entry. This is the mechanism that allows repair to start from a clean state without
  re-running prior steps.
- **Constants:** Universally known physical constants are pre-loaded from `scipy.constants`
  at service startup (not re-fetched per query). They appear in the VSO with
  `defined_at: "constants_table"`. Using `scipy.constants` eliminates a hand-curated
  table and ensures CODATA-authoritative values.

---

## 7. Alternative Solution Paths (Beam Search)

Top-N formula sets from Stage 1 define N independent solution paths. This is the
beam-search fallback when the primary path fails.

**Path priority:**
1. Path 0 (highest retrieval confidence) is attempted first.
2. If Path 0 exhausts its repair budget without a PASS, Path 1 is attempted from Stage 2.
3. Continue until a PASS is achieved or all paths are exhausted.

**Path independence:** Each path uses its own Trace Object and VSO. The stable prefix from
a failed path is not carried over to a new path (the new path starts fresh from Step 1).
This is because a different formula set may require a different problem setup.

**When all paths fail:** Return the trace with the highest `final_answer` confidence across
all paths, with `trace_status = FAIL`. Log all attempted paths for debugging.

---

## 8. Data Handling and Training

### 8.1 Official Dataset

The official training data released on May 9, 2026:
- **Type 1:** 411 records, 808 questions (logic-based educational queries)
- **Type 2:** 1,755 records — **filter out the 401 QA-prefixed samples** (annotation
  errors confirmed by organizers, Q19). Effective Type 2 training set: **1,354 problems**.

Filter implementation: `df = df[~df['id'].str.startswith('QA')]`

Using the official dataset for training is not mandatory (Q16). It defines the format
and question types; external data, synthetic data, and augmented datasets are all allowed.

### 8.2 Dataset Split

- Split problems by `sub_domain` first, then by difficulty if labeled.
- Within each group: 70% train, 15% validation, 15% test.
- Hold the test split out of all training and prompt optimization decisions.
- Maintain separate splits for Type 1 and Type 2 to ensure both are represented in
  the validation set when evaluating combined performance.

### 8.3 Knowledge Distillation and Synthetic Training Data

**Allowed (Q6, Q10):** Larger teacher models (GPT-4, Claude, etc.) may be used to
generate training data and annotate traces. The teacher must not be called at inference
time. All teacher models and synthetic data must be disclosed in the Data Disclosure
Document.

**Recommended distillation pipeline:**

1. **Physics-solver adapter training data:**
   Run all 1,354 Type 2 training problems through a teacher model (e.g., GPT-4).
   Generate complete annotated Trace Objects: step-by-step solutions with `thought`,
   `step_input`, `intermediate_answer`, formula IDs, and final answer. Filter to traces
   where `final_answer` is correct. Use these as positive training examples.

2. **Verifier adapter training data:**
   Run training traces (including deliberately introduced errors) through the teacher
   model to annotate step-level correctness judgments (`OK` / `WRONG` / `UNCERTAIN`),
   FWS labels, and error type classifications (E1–E6). The verifier adapter learns to
   replicate the teacher's judgment at 8B scale.

3. **Logic-reasoner adapter training data:**
   Run all 808 Type 1 training questions through the teacher model. Generate annotated
   inference traces and, for Yes/No/Uncertain questions, generate the corresponding Z3
   Python code alongside the natural language reasoning. Both forms are training signal.

4. **Response-assembler adapter training data:**
   Generate `explanation` prose for each completed trace using the teacher model.
   These are the targets for the `response-assembler` adapter's fine-tuning.

**Training format:** All adapter training examples are formatted as full structured
objects (Trace Objects or Stage 6 Response Objects), not question-answer pairs only.

**Training data quality gates:**
- Only include Type 2 traces where `final_answer` is verified correct.
- Only include traces where `cot_consistent = CONSISTENT` across all steps.
- Do not use traces where the final answer is correct but COT reasoning is flagged
  `INCONSISTENT` — these are misleading training signals.
- Include corrected error traces (original wrong trace → repaired trace) for the
  verifier adapter; research shows this improves error detection (An et al., 2023).

### 8.4 LoRA Adapter Training

Train each adapter independently using HuggingFace PEFT:
- Base model: the chosen 8B-class model (initial: Qwen3-8B)
- LoRA rank: r=16, alpha=32
- Target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- Training split: 70% of role-specific training data
- Validation split: 15% (use to select best checkpoint)
- Save adapter weights to local disk. Load from local path at inference — do not
  re-download. All four adapters must be present at vLLM startup.

### 8.5 Prompt Optimization (Optional, DSPy teams)

If using DSPy: define each stage as a `dspy.Signature`, optimize with `dspy.MIPROv2`
or `dspy.BootstrapFewShot` against correctness on the validation split. Save compiled
programs to disk and load at inference.

### 8.6 Data Disclosure Document (Required for Submission)

The competition requires a Data Disclosure Document submitted alongside the one-page
solution description (Q11, Q23). This document must describe:
- Every external dataset used (name, source URL, size, purpose)
- Any synthetic data generated by closed-source models (which model, how many samples, for which adapter)
- Any crawled or scraped data (source URLs, volume, preprocessing steps)
- The formula library corpus and its sources
- The knowledge base built for Wikipedia API / Wolfram Alpha fallback retrieval
- Teacher models used for distillation and which adapters they generated data for

Submissions without this document, or with undisclosed data sources discovered later,
are grounds for disqualification.

---

## 9. Deployment Requirements

### 9.1 LLM Serving

The LLM component must be served via **vLLM** (or a compatible OpenAI-style serving
framework) per Q5 and Q14. This is required — not optional — for all teams.

```bash
vllm serve Qwen/Qwen3-8B \
  --enable-lora \
  --lora-modules \
    physics-solver=./adapters/physics-solver \
    verifier=./adapters/verifier \
    logic-reasoner=./adapters/logic-reasoner \
    response-assembler=./adapters/response-assembler \
  --max-lora-rank 16 \
  --seed 42 \
  --port 8000
```

`--seed 42` sets a server-level default for all requests. Each pipeline call also
passes `"seed": 42` in the request body (from `SolverConfig.seed`) so outputs are
reproducible even after server restarts.

The committee may query the `/v1/models` endpoint at any time to confirm the loaded
model name and inspect GPU memory usage (Q14). Only one base model may be resident in
GPU memory at any time.

### 9.2 Latency Budget

- **Hard cap: 60 seconds per request** (Q13). A timeout counts as a failed answer.
- `SolverConfig.latency_budget_seconds` defaults to **55** to preserve a 5-second margin.
- Tools (SymPy, Wolfram Alpha, Z3, Prover9, Wikipedia API) do not count against the LLM
  budget but do count against wall time. Account for network latency on Wolfram Alpha /
  Wikipedia calls (typical: 200–800 ms; worst-case: 2–3 s). Prover9 runs locally with
  a 5-second timeout.

**Three-tier timeout ladder** (thresholds set in `SolverConfig`):

| Elapsed wall time | Tier | Action |
|---|---|---|
| < 12s (`timeout_tier1_seconds`) | Full pipeline | Beam search active; repair loops active |
| 12s – 35s (`timeout_tier2_seconds`) | Tier 1 fallback | Set `beam_n=1`; disable repair loops; single-path solve only |
| 35s – 55s (`latency_budget_seconds`) | Tier 2 fallback | Additionally skip optional fields (`fol`, `cot`, `premises`); emit `answer` + `explanation` only |
| ≥ 55s | Hard stop | Emit best available `final_answer` immediately; set `confidence = 0.1` |

**Budget validation (do before freezing defaults):** Estimate worst-case LLM call count
as `beam_n × (trace_budget + repair_budget)` calls. Multiply by your measured
per-call latency (typically 1–4 s for an 8B model on A100). Confirm the product is
well under 12s for a clean solve, leaving Tier 1/2 as genuine fallbacks for hard cases.

### 9.3 Infrastructure

- No restriction on GPU type, cloud provider, or geographic region (Q13, Q25).
- Recommended: cloud VM with a single A100 or equivalent (sufficient for one 8B model).
- For local machines: the endpoint must be publicly accessible (ngrok, reverse proxy,
  or port forwarding) and remain online for the full evaluation window.
- Tools, solvers, and retrieval modules (SymPy, Z3, Prover9, Wolfram Alpha API,
  Wikipedia API, quantulum3, Pint, scipy) may be hosted anywhere — they are not LLMs
  and are not subject to the parameter limit or self-hosting requirement (Q5).

**Tool / solver installation requirements:**

| Tool | Install | Notes |
|---|---|---|
| SymPy | `pip install sympy` | Pure Python; no system deps |
| Z3 | `pip install z3-solver` | Includes Python bindings |
| Prover9 | System package or binary download | Invoked via subprocess; set 5s wall-clock timeout via `--max-seconds 5` flag |
| quantulum3 | `pip install quantulum3` | Requires spaCy model download |
| Pint | `pip install pint` | Pure Python |
| scipy | `pip install scipy` | For `scipy.constants` |
| bge-small-en (embedding) | HuggingFace model download | ≤ 100M params; for formula similarity search |

Prover9 is a standalone binary (not a Python package). The pipeline invokes it via
`subprocess.run(["prover9"], input=prover9_input_string, timeout=5, ...)` and parses
stdout for `THEOREM PROVED` or `SEARCH FAILED`. The binary must be on the system PATH
of the inference server.

### 9.4 Evaluation Timeline

| Phase | Dates | Action Required |
|---|---|---|
| Phase 1 evaluation | Jun 1–2, 2026 | API endpoint online for full window |
| Model refinement | Jun 3–4, 2026 | Improve based on Phase 1 feedback |
| Phase 2 evaluation | Jun 5–7, 2026 | API endpoint online for full window |
| Top 10 announcement | Jun 10, 2026 | — |
| Public Test Day | Jun 15, 2026 | Live system demo; P3 (reasoning depth) evaluated in-person |

For the Public Test Day, finalists run their systems live before the Challenge Chairs,
who may inspect the deployment environment, GPU memory usage, and model loading. The
`response-assembler` adapter's COT and FOL outputs will be evaluated live under P3.

### 9.5 Committee Compliance Checklist (#12)

The committee will inspect the live deployment on Public Test Day. Proactively document
the following in the one-page solution description submitted alongside the system:

| Item | Value / Confirmation |
|---|---|
| Base model name | Qwen/Qwen3-8B (or updated final selection) |
| Base model parameter count | ~8B (exact count from HuggingFace model card) |
| Adapters loaded | `physics-solver`, `verifier`, `logic-reasoner`, `response-assembler` |
| Adapter sizes (approx.) | 10–100 MB each (confirm after training) |
| Serving framework | vLLM |
| vLLM startup command | (reproduce exact command from Section 9.1) |
| GPU memory at inference | One base model only; adapters are weight deltas in the same memory footprint |
| Simultaneous models | Never — one base model resident at all times; adapter swap is in-place |
| External tool calls | SymPy, Z3, Prover9, Wolfram Alpha (tagged entries only), Wikipedia API |
| Training data sources | Disclosed in Data Disclosure Document (Section 8.4) |

This table should be ready before Phase 1. The committee may query `/v1/models` and
inspect `nvidia-smi` output at any time during the evaluation window (Q14).

### 9.6 Dataset Issue Reporting (Bonus Points)

Teams that report verified dataset annotation issues (incorrect labels, ambiguous
questions, formatting bugs) during Phase 1 or Phase 2 receive a bonus on the final
score (Q22). Report via the `#dataset-issue-report` Discord channel with the record ID,
issue type, and justification. Worth reviewing both datasets before Phase 1.

---

## 10. Inter-Stage Contracts Summary

| Stage | Output Artifact | Key Fields Produced | Adapter Used |
|---|---|---|---|
| Router | — | `query_type` field read from payload | — |
| Type 1 | Type 1 Response | COT inference steps, Z3 code (FOL), answer | `logic-reasoner` |
| 0 | Problem Parse Object | `domains` (keyword-classified), `known_quantities` (hybrid extracted), `step_plan`, VSO (scipy.constants pre-loaded) | LLM fallback only |
| 1 | Ranked formula sets (N paths) | `formula_path_index`, `formula_ids[]`, `retrieval_confidence`, `fol_axiom[]`, `premise_text[]` | Regex → embedding → LLM fallback |
| 2+3 | Trace Object | `steps[].status`, `steps[].confidence` (verifier-derived), `steps[].checkable`, `step_input` (with tool annotations), `vso_snapshots`, `trace_status` | `physics-solver` (generate), `verifier` (LLM verify) |
| 4 | Diagnosis Object | `fws_index` (programmatic for checkable; LLM for non-checkable), `global_error_type` (E4/E5 programmatic; E1/2/3/6 LLM), `step_labels`, `cot_issues` | `verifier` |
| 5 | Repaired Trace Object | updated `steps[]`, updated `vso`, re-enters Stage 2+3 | `physics-solver` |
| 6 | API Response Object | `answer` ✱ (extracted), `explanation` ✱ (LLM), `fol` (library + Z3), `cot` (string-formatted), `premises` (collected), `confidence` (geometric mean) | `response-assembler` (explanation only) |

✱ = Required field. All other Stage 6 fields are produced without LLM calls except `explanation`.

Each artifact is a JSON schema. A stage is correct if its output conforms to the schema,
regardless of the framework used to produce it.

---

## 11. Open Questions / TBDs

| # | Question | Notes |
|---|---|---|
| 1 | Which 8B-class base model performs best across both query types? | Initial candidate: **Qwen3-8B** (hybrid thinking mode, strong math + reasoning). Evaluate against Qwen2.5-7B-Instruct and DeepSeek-R1-Distill-Qwen-7B on held-out validation. Final decision after empirical testing. |
| 2 | Does SymPy + Wolfram Alpha cover enough of the formula library to make `tool_dispatch: "llm"` rare? | Measure `"llm"` dispatch rate on training data. High rates indicate formula library curation gaps or formula types that need Wolfram Alpha entries added. |
| 3 | Total trace retry budget (default 10) — tune on validation set. | Lower values reduce latency. `SolverConfig.trace_budget` controls at runtime without code changes. |
| 4 | What happens when `fws_index` is null after Stage 4? | `fws_index = null` is an explicit path-advance trigger: Stage 5 is skipped entirely for this path (nothing to repair), and the pipeline immediately advances to the next candidate path from Stage 1. Stage 5's repair budget is not consumed. If all paths return `fws_index = null`, the pipeline has exhausted all structured options — forward the best-confidence trace to Stage 6 as-is, with `trace_status = FAIL`. |
| 5 | Scoring model weights — **partially resolved, revisit required** | Q21 confirms P1 (correctness) + P2 (explanation quality) + P3 (reasoning depth). No penalty for wrong answers confirmed → `confidence_threshold = 0.0` is correct. **However**, exact weights per component have not been published yet. Two conditional actions when weights drop: (a) If P2 is heavily weighted: consider also suppressing `fol`/`cot` when `trace_status = REPAIRED` with `confidence < 0.5` — a repair that barely passed may still produce a misleading explanation. (b) If P1 is heavily weighted and a wrong answer with confident optional fields can hurt P2 scoring, tighten the FAIL suppression table in Stage 6. |
| 6 | Unit tolerance for verifier matching (e.g., 7.67 vs 7.672). | Suggest ±0.1% relative tolerance as default. Q20 confirms both Unicode and LaTeX notation are normalised. |
| 7 | Formula library scope and curation ownership. | **Priority order:** circuits/electrostatics first (primary Type 2 domain — Ohm's law, Kirchhoff's laws, series/parallel resistance, power, capacitance, energy storage, Coulomb's law, electric field), then mechanics, thermodynamics, electromagnetism, optics, modern physics as secondary coverage. Both `fol_axiom` and `premise_text` fields required per entry (see Section 4.3 for schema). **Owner must be assigned before Stage 1 testing can begin — this is the critical path dependency for the entire pipeline.** |
| 8 | ~~Variable canonicalization: LLM or rule-based?~~ **RESOLVED** | Three-tier cascade: regex lookup table → embedding similarity → LLM fallback. Documented in Stage 1. |
| 9 | ~~Student-student paradigm under budget?~~ **RESOLVED** | Implemented via LoRA adapter alternation: `physics-solver` adapter generates, `verifier` adapter verifies. Same base model, no budget issue. |
| 10 | How many candidate paths (N) should Stage 1 return? | Default N=3 in `SolverConfig.beam_n`. Tune on validation. Set to 1 if latency is the binding constraint under 55s budget. |
| 11 | Z3 formalization quality on Type 1 training data — what is the failure rate? | If the `logic-reasoner` adapter frequently generates invalid Z3 code, `SolverConfig.type1_use_z3 = false` bypasses Z3. Measure on Type 1 validation split. |
| 12 | FOL curation: who adds `fol_axiom` and `premise_text` to the formula library? | Assign owner before build starts. Can be bootstrapped from existing LaTeX formula corpora; requires manual review. |
| 13 | Wolfram Alpha API rate limits and latency under evaluation load. | Evaluate typical response time. If >2s average, use only for `tool_dispatch: "wolfram"` entries (not as a general fallback). Cache repeated queries within a session. |
| 14 | Dataset issue bonus: are there annotation errors worth reporting in Phase 1? | Review both Type 1 and Type 2 datasets systematically before Jun 1. Each verified report earns a leaderboard bonus (Q22). |

---

*End of document — v0.8*
