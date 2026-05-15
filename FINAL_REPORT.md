# Stage 0 Parser Optimization — Final Report

## Headline numbers

|                      | Baseline | Final  | Delta  |
|----------------------|---------:|-------:|-------:|
| **Eval cohort (1354)** PASS+PNN rate | **77.5%** | **84.2%** | **+6.7 pp** |
| Eval cohort PASS     |      978 |   1064 |    +86 |
| Eval cohort PNN      |       71 |     76 |     +5 |
| Eval cohort FAIL     |      305 |    214 |    -91 |
| Eval cohort low_confidence | 305 |   214  |    -91 |
| Eval cohort missing_quantity | 20 |   20  |      0 |
| Eval cohort missing_target |    6 |     6 |      0 |
| **Full set (1755)** PASS+PNN rate | **69.8%** | **75.3%** | **+5.5 pp** |
| Full set PASS        |     1117 |   1207 |    +90 |
| Full set PNN         |      108 |    114 |     +6 |
| Full set FAIL        |      530 |    434 |    -96 |

The eval-cohort target was 85%. We landed 0.8 pp short. The next-best
remaining clusters require either SHM expression parsing or symbolic
algebra, both flagged out-of-scope by the brief.

## Templates added per phase

### Phase 1 — AC off-resonance (`_ac_detailed_templates`)
Single new matcher function with five branches, registered between
`_ac_supplemental_templates` and `_ac_templates`:
  * `ac_power_factor_from_R_Z`        — 2-resistance positional (R, Z)
  * `ac_impedance_RLC`                — 3-resistance positional X_L, X_C, R
  * `ac_I_rms_from_V_Z`               — V/Z current with 3R+V or 1R+V
  * `ac_U_R_from_I_R`, `ac_U_L_from_I_X_L`, `ac_U_C_from_I_X_C`
  * `ac_omega_from_f`, `ac_X_L_from_L_omega`, `ac_X_C_from_C_omega`
  * `ac_avg_power_VI_cos`, `ac_phase_angle_tan`
  * Branch C: full L+C+f+R+V chain (omega → reactances → Z → derived).
  * Branch E: extends `resonance_R_from_Z` / `impedance_at_resonance`
    to fire on `ac_circuit` / `rlc_circuit` hint with 1 lone Ω value.

Phase 1 delta (eval): PASS +23.

### Phase 1b — small follow-ups
  * Branch F of `_ac_detailed_templates`: target ∈ {X, X_C, k} with 2R
    + omega → `rlc_omega_factor_for_resonance`
  * `_capacitor_merge_templates`: V_after = (C1*V1 + C2*V2)/(C1+C2)
  * `_measurement_set_templates`: N same-base values → mean +
    abs/rel/percent error chain
  * `_parallel_plate_and_geometry_templates`: broadened `epsilon_r`
    gating to accept `dielectric_capacitor` sub-domain hint.

Phase 1b delta (eval): PASS +32. Cumulative since baseline: +55.

### Phase 2 — energy diff, least-count, off-resonance, AB-circuit
  * Extended `_capacitor_energy_templates` to accept target `U_C` as
    a voltage synonym.
  * `_lc_energy_diff_templates` — target ∈ {U_B, U_E, U_after} with 2
    energies → `target = E_total - E_other`.
  * `_least_count_percent_error_templates` — base + base2 positional
    pair (p_pressure, temperature, F, m, d, h, V, I, R).
  * `_resonance_off_frequency_templates` — 4-step X_L derivation from
    R + (f, f2) + (I, I2).
  * `_ab_circuit_quadrature_templates` — R_total = V²/P then
    other_R = R_total - given_R (LCω²=1 + quadrature simplification).

Phase 2 delta (eval): PASS +31. Cumulative since baseline: +86.

### Phase 3 — concept classifier patterns (`question_type_classifier.py`)
5 narrow SYMBOLIC_PATTERNS added (after rejecting a broader initial set
that caused 3 PASS→PNN regressions):
  * `when_X_is_fraction_of_total` — fraction + 'of (the) total'
  * `where_is_entirely_stored` — anchored on entirely/fully/completely
  * `what_happens_when` — what happens in/when (variant)
  * `what_fraction_is` — what fraction is/of/are
  * `what_appears_in` — appears/develops/forms in

Also widened existing `when_will_X_be` word-window from {0,4} to {0,8}.

Phase 3 delta (eval): PASS 0, PNN +5. Cumulative since baseline:
PASS +86, PNN +5.

## Per-phase eval-cohort progression

| Phase | PASS | PNN | FAIL | Rate  |
|-------|-----:|----:|-----:|------:|
| Baseline       |  978 |  71 |  305 | 77.47% |
| Phase 1        | 1001 |  71 |  282 | 79.17% |
| Phase 1b       | 1033 |  71 |  250 | 81.54% |
| Phase 2        | 1054 |  71 |  229 | 83.09% |
| Phase 2b       | 1064 |  71 |  219 | 83.83% |
| Phase 3 (initial — REJECTED) | 1061 | 82 | 211 | 84.42% |
| Phase 3 (narrowed — ACCEPTED) | 1064 | 76 | 214 | 84.19% |

Rejected attempt: 3 numeric problems (NL372, CH028, CH274)
reclassified to PNN, violating the regression budget (PASS drop > 2).
Patterns narrowed and re-tested.

## Eval-cohort clusters intentionally skipped

| Cluster | Count | Reason |
|---------|------:|--------|
| U_B + L_ind only (and L_ind+t) | 9 | Current expression "I(t) = 2sin(100πt)" requires SHM amplitude extraction — out of scope per deterministic-first. |
| U_R / V_rms + 2R + V (no actual R) | 9 | Resistance R is genuinely unknown in problem text; answer must stay symbolic. Stage 0 cannot produce a numeric plan without inventing a value. |
| I + (I, I2) | 5 | Same known shape covers three different physics: uncertainty (I_max = I + delta_I), parallel-circuit subtraction, ratio — no disambiguator in known set. |
| U_E + C_cap only | 4 | Under-specified; needs at least Q or V; target-detector mismatch suspected. |
| power_factor + empty known | 4 | No quantities extracted; either symbolic ("at resonance cos φ = 1") or extraction failure. |
| V_rms + (R1, R2, V_rms, theta) | 4 | AB-circuit voltage-divider with target/known collision on V_rms; would require unwrapping the segment-voltage semantics. Below ROI. |

## Distribution-shift note

The full-set rate moved from 69.8% to 75.3% (+5.5 pp), while the eval
cohort moved from 77.5% to 84.2% (+6.7 pp). The QA-prefixed subset
gained essentially nothing (44% → 45.6%, +1.2 pp from 6 incidental
PASSes, mostly via the broadened resonance-hint Branch E of
`_ac_detailed_templates`). This confirms the brief's premise that
QA-prefixed problems are out-of-distribution and that fitting them
risks regressions on the eval cohort. None of the templates added were
tuned to QA phrasings.

## Next-direction suggestions (to push beyond 84.2%)

The deterministic-first ceiling has been approached. To break 85%
would require leaving the regex/lookup-table paradigm:

1. **SHM expression parser** (~9 cases). Extract amplitude / omega /
   phase from `f(t) = A·sin/cos(ω·t + φ)` formulas. Even a small AST
   evaluator would handle these. Estimated +9 PASS.

2. **Symbolic-R routing** (~9 cases). When `_ac_detailed_templates`
   Branch B fires with only 2 R values (X_L, X_C) and no third R, and
   target is U_R / V_rms, route the problem to `symbolic_derivation`
   via a question-type override. Cleaner than emitting a misleading
   numeric plan. Would need a new STRONG_SYMBOLIC pattern matching
   "the resistor R" (R as bare letter, no value).

3. **Target-detector refinement**:
   * U_E + C_cap problems often have target=U_E but ask for V — fix
     to detect "voltage across the capacitor" → target=V.
   * U_B + L_ind problems with SHM expressions often have target=U_B
     but truly ask for I — would need expression context awareness.

4. **AB-circuit segment voltages** (~4 cases). Extend
   `_ab_circuit_quadrature_templates` to target the segment voltage
   when V_rms appears in BOTH known and target. Formula:
   `U_segment = V * sqrt(R_segment / (R1+R2))`.

5. **`I + I2` disambiguator**. Inspect the problem text for tokens
   like "±", "uncertainty", "parallel", "total" to choose between
   `I_max = I + I2`, `I_diff = I - I2`, and ratio formulas. This is
   borderline-regex but feasible.

## Boundary cases requiring breaking deterministic-first

| Boundary | Affected cases (eval) | Notes |
|---|---:|---|
| Expression parsing (sin/cos with time) | ~9 | Stage 0 charter: don't evaluate trig. Stage 1 / LLM fallback. |
| Symbolic algebra (R unknown letter) | ~12 | Stage 0 charter: don't carry symbols past parse. Stage 1+. |
| Multi-part questions (a/b/c sub-prompts) | unmeasured | Only first sub-part parsed by design. |

## Files modified
  * `parser/template_fallback.py` — added `_ac_detailed_templates`,
    `_lc_energy_diff_templates`, `_least_count_percent_error_templates`,
    `_resonance_off_frequency_templates`, `_capacitor_merge_templates`,
    `_measurement_set_templates`, `_ab_circuit_quadrature_templates`.
    Extended `_capacitor_energy_templates` target set, broadened
    `_parallel_plate_and_geometry_templates` epsilon_r gating.
    Updated matcher dispatch tuple in `propose_step_plan`.
  * `parser/question_type_classifier.py` — added 5 SYMBOLIC_PATTERNS,
    widened `when_will_X_be` word-window.

## Files created
  * `scripts/eval_cohort_summary.py` — splits results.jsonl into
    eval cohort (non-QA) / QA subset / full set blocks.
  * `outputs/stage0_phase{1,1b,2,2b,3,3b}/` — per-phase snapshots with
    eval-cohort summaries.
