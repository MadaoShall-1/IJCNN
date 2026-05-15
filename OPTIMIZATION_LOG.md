# Stage 0 Parser Optimization Log

## Evaluation cohort — important

The source CSV `Physics_Problems_Text_Only.csv` contains 1,755 rows but
**only 1,354 form the evaluation cohort**. The 401 rows with `QA`-prefixed
IDs (e.g. `QA8000164`) lack ground-truth `answer` and `unit` fields and
come from a different source whose phrasing conventions the parser was
never tuned for. They are out-of-distribution.

**All quality metrics and regression budgets must be computed on the
eval cohort only.** Full-set numbers are diagnostic — they show
distribution-shift behavior, but are not the optimization target.

To split: a row is in the eval cohort iff `not row['id'].startswith('QA')`.

| Subset | Size | Latest Pass+PNN rate |
|---|---|---|
| **Eval cohort** (non-QA) | **1,354** | **77.5%** ← optimization target |
| QA-prefixed | 401 | 44% (out-of-distribution, do not fit) |
| Full set | 1,755 | 69.8% (diagnostic only) |

## Architecture summary
Deterministic-first physics problem parser. Pipeline (in `parser/main.py`):

  parse_problem(text):
    classify_question_type    → numeric_calc | boolean_check | symbolic_derivation
    domain_classify           → domains + sub_domains
    extract_conditions        → list of physical-setup conditions
    extract_quantities        → {name: {value, unit, dimension, normalized_value}}
    extract_relations         → ratio/function/equation/uncertainty relations
    detect_target             → unknown_quantity (symbol + unit)
    propose_step_plan         → dispatches to first matching template in
                                template_fallback.propose_step_plan
    apply_verifier            → PASS / PASS_NON_NUMERIC / FAIL
    skeleton fallback         → if no template matched but target exists,
                                emit a single-step skeleton placeholder
                                with confidence 0.30 (below PASS threshold)

PASS criteria (parse_verifier.py):
  - question_type == numeric_calc → must have target, known quantities,
    a step plan with at least one non-skeleton step, all numeric tokens
    covered by extracted quantities, confidence ≥ 0.5
  - question_type in {boolean_check, symbolic_derivation} → goes through
    _verify_non_numeric, requires known quantities or recognized domain;
    no numeric step plan required → status PASS_NON_NUMERIC

Skeleton fallback contract (do not change):
  template_name == "skeleton_placeholder"
  confidence == 0.30
  metadata.used_skeleton_fallback == True

## Progression (eval cohort, 1354 problems)

| Stage | PASS | PNN | FAIL | Pass+PNN rate | low_conf | missing_q | missing_t | invalid_final |
|-------|------|-----|------|---------------|----------|-----------|-----------|---------------|
| Original baseline       | ~770 | ~30 | ~554 | ~59% | 489 | 237 | 71 | 410 |
| 修改 5 (skeleton fix)   | ~770 | ~30 | ~554 | ~59% | 489 | 237 | 71 |   0 |
| 修改 1 (target detector)| ~795 | ~30 | ~529 | ~61% | 472 | 237 | 17 |   0 |
| 修改 2 (units + verifier)| ~820 | ~30 | ~504 | ~63% | 472 |  28 | 17 |   0 |
| 修改 3a (resonance tmpl)| ~895 | ~30 | ~429 | ~68% | 396 |  28 | 17 |   0 |
| 修改 3b (EM/AC/measure) | ~950 | ~30 | ~374 | ~72% | 343 |  22 | 17 |   0 |
| 修改 A  (concept patterns)| ~945 | ~71 | ~338 | ~75% | 305 |  22 | 13 |   0 |
| **修改 B (mech + classifier v2)** | **978** | **71** | **305** | **77.5%** | **305** | **20** | **6** | **0** |

(Numbers prior to 修改 B are estimates; only 修改 B was measured exactly
on the eval cohort. Next runs should compute eval-cohort numbers
directly.)

For diagnostic reference, full-set (1755) progression:

| Stage | Full-set Pass+PNN rate |
|-------|------------------------|
| Original baseline | 53.4% |
| 修改 B (current)  | 69.8% |

The 7.7-point gap between eval cohort (77.5%) and full set (69.8%) is the
distribution-shift signal from QA-prefixed rows.

## Key design decisions made

### 修改 5 — Skeleton fallback design
Original behavior: when target identified but no template matched, parser
emitted no step plan → verifier reported invalid_final_step. Fixed by
emitting a "skeleton_placeholder" step with confidence 0.30 so verifier
sees a real plan but parse still FAILs on low_confidence. Critical
contract — do not change.

### 修改 1 — target_detector cue hierarchy
Split CUES into PRIMARY_CUES (strong) and SECONDARY_CUES (weak). Added
SOFT_CUE_PATTERNS with 60-char back-window. Added 25 new target phrases.
Added how-adverb specials: "how long"→t/s, "how far"→d/m,
"how high"→h/m, "how fast"→v/m/s. Fixed "what is R" misdetection.

### 修改 2 — Major bug fix in _parse_number
The pow10 regex was greedy and matched plain integer '1000' as
coefficient='', exponent='00' → 10^0 = 1.0. This corrupted numeric
extractions. Fixed by requiring '^' present. This single fix eliminated
~30 missing_quantity errors.

### 修改 2 — Unit dictionary expansion
Added 30+ units: kW/mW/MW, kJ/MJ/eV, specific heat (J/kg.K), latent heat
(J/kg, kJ/kg), mass (ton, tonne, mg), volume (L, liter, mL, m^3),
pressure (Pa, kPa, MPa, atm, bar, mmHg), count (turn, turns), turn
density (turns/m, turn/m, turns/cm). Added 5 new dimensions to
SI_UNIT_BY_DIMENSION.

### 修改 2 — Verifier ignore_spans system
Structural ignore for clock times (7:30 AM), LCω²=1 resonance, scientific
notation literals (4.10^-10), sequence labels (Car 1, Vehicle 2),
hyphenated compounds (2-hour), trig function arguments, μ₀ constant
declarations.

### 修改 3a — Resonance design templates (6)
  lc_resonance_capacitance:        C = 1/(4π²f²L)
  lc_resonance_inductance:         L = 1/(4π²f²C)
  resonance_R_from_Z:              R = Z at resonance
  impedance_at_resonance:          Z = R at resonance
  rlc_resonance_check:             compute f_res, compare to f
  rlc_omega_factor_for_resonance:  ω_new = ω₀ * sqrt(X_C/X_L)

### 修改 3a — sub_domains → conditions bridge
main.py's _apply_template merges sub_domains into conditions before
passing to propose_step_plan. Lets templates gate on sub_domain hints.

### 修改 3b — EM / AC supplemental / parallel plate / percent error (14 templates)
  solenoid_field_full, solenoid_field_from_density
  magnetic_flux_BA, flux_linkage_BAN, flux_linkage_from_per_turn
  emf_from_di_dt, emf_from_flux_change
  rms_current_from_resistance_at_resonance
  resonance_UL_calc, resonance_UC_calc (3-step plans)
  epsilon_r_from_capacitance, turn_density, parallel_R
  percent_error_from_least_count, percent_error_pm_uncertainty
  absolute_error_from_actual_measured, absolute_error_direct

Added mu_0 to CONSTANTS in parse_verifier.py and template_fallback.py.

Matcher registration order in propose_step_plan (do not reorder without
testing — order encodes specificity):
  relation_driven → capacitor_energy → inductor_energy → force_resultant
  → field_geometry → capacitance → coulomb_force → circuit → mechanics
  → mechanics_extended → dielectric → measurement → percent_error → lc
  → resonance_design → ac_supplemental → ac → parallel_plate
  → electromagnetism → basic

### 修改 A — question_type_classifier expansion (v1)
Added 13 SYMBOLIC patterns: how_does_X_change, how_qualitative,
what_happens_to, where_is, when_will_X_be, what_form_of, graph_shape,
formula_for, which_quantity, if_X_changes, describe_explain, units_of,
why_is.
Added 3 NUMERIC_OVERRIDE patterns: round_to_decimal, what_percentage_of,
by_what_factor.

### 修改 B — Mechanics extended (14 templates)
Free-fall: free_fall_v_from_h, free_fall_t_from_h, free_fall_h_from_t
Braking: kinematics_a_from_d_t_rest_endpoint, kinematics_v0_from_braking
Newton's 2nd law: newton_F_from_m_a, newton_a_from_F_m, newton_m_from_F_a
From-rest acceleration: kinematics_v_from_rest, kinematics_d_from_rest
Energy relation: kinematics_v_from_a_d, kinematics_a_from_v_v0_d
Average velocity from endpoints: v_avg_from_endpoints
Relative motion extension: relative_motion_v_from_d_v2_t

Also extended domain_classifier mechanics/kinematics keywords (+24 new
phrases like falls/brake/decelerat/stops/drop/object/motorbike/etc.)

### 修改 B — question_type_classifier v2
Added 13 more SYMBOLIC patterns: unit_of_X, what_does_X_depend_on,
in_what_form, when_X_max_min, characteristics_of_X, how_to_calculate,
is_there_a_formula, multiple_choice_options, find_expression,
state_the_formula, what_kind_of_field, what_does_this_indicate,
instructional_opener.

Added STRONG_SYMBOLIC_OVERRIDE_PATTERNS (6 patterns) for high-confidence
non-numeric phrases: strong_how_to, strong_unit_of, strong_formula_for,
strong_in_what_form, strong_where_is_stored, strong_what_depend_on.

Added STRONG_NUMERIC_OVERRIDE_PATTERNS (4 patterns) for definitive
numeric signals.

Rewrote the decision logic into a strong-vs-weak override hierarchy:
strong numeric > strong symbolic > both bool+sym > weak numeric vs ≥2
non-numeric signals (non-numeric wins) > weak numeric + 1 non-numeric
(numeric wins conservatively) > single-class default.

## Open work — known opportunities (in eval cohort)

* AC off-resonance circuits — largest remaining cluster (~25–40 problems).
  Templates need impedance Z computed first, then derived quantities
  (U_R, U_C, U_L, I, cos_phi). Mechanical work.

* Thermodynamics (~15 problems). Q = mcΔT, Q = mL.

* Optics (~9 problems). Thin lens, mirror formulas.

* Gas laws (~5 problems). pV = nRT, combined gas law.

* Mechanical energy (~20 problems). E_mech = KE + PE.

* Remaining missing_quantity (20) and missing_target (6) — scattered
  edge cases, mostly below the ROI threshold for individual fixes.

## Out of scope — do not pursue

* Simple harmonic motion expression parsing: extracting A, ω, φ from
  `x = 4cos(4πt + π/3)` requires symbolic-math evaluation. This is the
  deterministic-first boundary. ~15 problems in eval cohort affected.

* QA-prefixed subset improvement: out-of-distribution by construction.
  Fitting it would risk regressions on the eval cohort.

* Multi-part questions (a/b/c): only first sub-part parsed.

## Files in the parser

  parser/main.py                   — pipeline orchestrator
  parser/question_type_classifier.py — numeric vs boolean vs symbolic gate
  parser/domain_classifier.py      — domain + sub_domain tagging
  parser/condition_extractor.py    — extract physical-setup conditions
  parser/rule_extractor.py         — extract quantities + relations
  parser/unit_normalizer.py        — unit lookup and SI conversion
  parser/target_detector.py        — detect unknown quantity
  parser/template_fallback.py      — formula templates (propose_step_plan)
  parser/parse_verifier.py         — verifier gate
  parser/schemas.py                — dataclasses
  parser/error_logger.py           — failure logging

## 2026-05-15 — Phase 1 — AC off-resonance templates

Change: Added `_ac_detailed_templates` to `parser/template_fallback.py`,
registered after `_ac_supplemental_templates` and before `_ac_templates`.
Five branches:
  A. power_factor / cos_phi with 2 resistance values → cos_phi = R/R2
     (positional convention since extractor stores XL, XC, R, Z all as
     resistance-dim names R, R2, R3 by order of appearance).
  B. 3 resistance values + V → Z = sqrt(R3² + (R - R2)²), then I_rms,
     U_R, U_L, U_C, cos_phi, P_avg, tan_phi depending on target.
  C. Full proper RLC: L_ind + C_cap + f + R + V known → omega = 2πf,
     X_L = ωL, X_C = 1/(ωC), Z = sqrt(R² + (X_L-X_C)²), then I, U_R,
     U_L, U_C, cos_phi, P_avg, tan_phi.
  D. I_rms with R + V only and ac_circuit hint → I = V/R (assumes
     given Ω value is impedance).
  E. R or Z target with 1 resistance value + ac/rlc hint → R = Z
     (extends the existing resonance_R_from_Z hint set).

Cluster targeted: power_factor + 2R (9), I/I_rms + 3R + V (3), R + 1R
+ ac (7), U_R + 3R + V (~3), V_R/V_C/V_L + 3R + V (small), plus
incidental.

Eval-cohort result: PASS=1001, PNN=71, FAIL=282,
  low_conf=282, missing_q=20, missing_t=6
Eval-cohort delta vs previous: PASS +23, PNN 0, FAIL -23
  Rate 77.5% → 79.2%

Full-set result (diagnostic): PASS=1140, PNN=108, FAIL=507
  Full-set rate 69.8% → 71.1%

Regression check: pass. PASS up, PNN unchanged, all error types
unchanged or down. No new missing_q / missing_t / invalid_final_step.

Notes: The clean 23-point delta on eval cohort vs 23 on full set means
the QA subset didn't benefit at all from these AC templates — which is
expected (QA phrasings diverge from eval cohort). Symbolic-only U_R /
V_rms problems with 2R+V (no actual R) were intentionally skipped
(~12 problems): they require keeping R symbolic, out of Stage 0 scope.

## 2026-05-15 — Phase 1b — Omega-factor + capacitor merge + measurement set + epsilon_r broaden

Change: Four small additions to `parser/template_fallback.py`:
  1. New `_ac_detailed_templates` Branch F: target ∈ {X, X_C, k} with 2
     resistance values + omega in known → `target = sqrt(R2 / R)`
     (rlc_omega_factor_for_resonance). Catches the "by what factor must
     ω be changed for resonance" family that detects targets as X /
     X_C / k rather than omega.
  2. `_capacitor_merge_templates`: two capacitors + two voltages →
     `V_after = (C1*V1 + C2*V2)/(C1+C2)` (parallel like-terminal
     connection). Registered after `_percent_error_templates`.
  3. `_measurement_set_templates`: N same-base measurements (m+m2+m3
     etc.) → mean, abs_error, rel_error, percent_error chain.
     Registered after `_measurement_templates`.
  4. `_parallel_plate_and_geometry_templates` epsilon_r: broadened
     gating to include 'dielectric_capacitor' sub_domain hint.

Clusters targeted (eval cohort):
  X/X_C/k + omega+2R (~10), capacitor merge V/U_C (~6), mass abs_error
  (~3 plus other bases), epsilon_r with C+A+d (~3).

Eval-cohort result: PASS=1033, PNN=71, FAIL=250,
  low_conf=250, missing_q=20, missing_t=6
Eval-cohort delta vs Phase 1: PASS +32, PNN 0, FAIL -32
  Cumulative since baseline: PASS +55. Rate 77.5% → 81.5%

Full-set result (diagnostic): PASS=1172, PNN=108, FAIL=475
  Full-set rate 71.1% → ~73%

Regression check: pass. All error types non-increasing.

## 2026-05-15 — Phase 2 — Energy diff, least-count error, off-resonance X_L, AB-circuit quadrature

Change: Five additions to `parser/template_fallback.py`:
  1. `_capacitor_energy_templates`: extended target set {V, V_after} to
     include `U_C` so "instantaneous voltage across the capacitor" with
     given C + stored energy is captured.
  2. `_lc_energy_diff_templates`: target ∈ {U_B, U_E, U_after} with two
     extracted energies E_energy + E_energy2 → `target = E_total - E_other`.
  3. `_least_count_percent_error_templates`: positional pair pattern
     (base + base2) for least-count + measured-value, supports
     p_pressure, temperature, F, m, d, h, V, I, R bases. Produces
     rel_error / percent_error / abs_error plans.
  4. `_resonance_off_frequency_templates`: target ∈ {X_L, X_C} with R +
     two frequencies + two currents → 4-step derivation through
     V = I*R, Z_2 = V/I_2, k_ratio = f_2/f_0,
     X_L = sqrt(Z_2² - R²) / (k_ratio - 1/k_ratio).
  5. `_ab_circuit_quadrature_templates`: target ∈ {R, R1, R2} with the
     other-R + theta + P + V_rms/V → `R_total = V²/P`, `target = R_total - other`.
     Derived from LCω²=1 + quadrature collapsing reactances.

Eval-cohort result: PASS=1064, PNN=71, FAIL=219,
  low_conf=219, missing_q=20, missing_t=6
Eval-cohort delta vs Phase 1b: PASS +31, PNN 0, FAIL -31
  Cumulative since baseline: PASS +86. Rate 77.5% → 83.8%

Full-set result (diagnostic): PASS=1207, PNN=108, FAIL=440
  Full-set rate ~73% → 75.0%

Regression check: pass. All error types non-increasing.

## 2026-05-15 — Phase 3 — Concept-classifier expansion (LC + qualitative)

Change: Added 5 narrow SYMBOLIC patterns to
`parser/question_type_classifier.py`. Initial broader set (8 patterns)
caused 3 PASS→PNN regressions (NL372, CH028, CH274 — real numeric
problems with values, swept up by overly broad "what is X in LC" /
"resonant frequency definition" / "if X is modified" patterns).
Narrowed to:
  * `when_X_is_fraction_of_total` — tight fraction + 'of (the) total'
  * `where_is_entirely_stored` — requires entirely/fully/completely
    modifier, not bare 'stored'
  * `what_happens_when` — what happens in/when (drop the prior 'to')
  * `what_fraction_is` — opens "what fraction is/of"
  * `what_appears_in` — what appears/develops/forms in

Also broadened `when_will_X_be` word-window from {0,4} to {0,8} so
"When will the magnetic field energy in a coil be zero?" classifies.

Eval-cohort result: PASS=1064, PNN=76, FAIL=214,
  low_conf=214, missing_q=20, missing_t=6
Eval-cohort delta vs Phase 2: PASS 0, PNN +5, FAIL -5
  Cumulative since baseline: PASS +86, PNN +5. Rate 77.5% → 84.2%

Full-set result (diagnostic): PASS=1207, PNN=114, FAIL=434
  Full-set rate 75.0% → 75.3%

Regression check: pass. Zero PASS→non-PASS transitions confirmed by
diff against Phase 2 results.

Notes: The initial 8-pattern set leaked 3 numeric PASS into PNN. After
narrowing to anchor-phrase versions, no regressions. The reclassified
problems are genuinely conceptual LC questions: "When will magnetic
field energy be zero?", "What happens when current disconnected?",
"What fraction is the magnetic energy?", etc.

## Stopping conditions reached

Phase 3 hit 84.2% eval-cohort pass+PNN, short of the 85% target. The
next-size eval-cohort clusters (counts 4-6) are all either:
  * SHM expression parsing: "I(t) = 2sin(100πt)" — out of scope per
    deterministic-first boundary (6 cases U_B + L_ind, 3 cases U_B +
    L_ind + t).
  * Symbolic R unknown: U_R / V_rms with 2R+V where the actual R has
    no numeric value (6 + 3 cases). Problems are inherently symbolic.
  * I + I2 ambiguity: uncertainty vs parallel-current vs ratio (5
    cases). Same shape, different physics; no disambiguator available.
  * U_E + C_cap only / power_factor + empty / U_B + L_ind + t only:
    under-specified known sets (4 + 4 + 3 cases).

No further deterministic templates can be added without target-detector
or rule-extractor work, which exceeds Stage 0 template-fallback scope.

## Critical invariants — do not break

1. PASS requires verifier confirmation. Never bypass the verifier.
2. Skeleton fallback uses template_name "skeleton_placeholder" and
   confidence 0.30. Never change these values.
3. Numeric problems require at least one real (non-skeleton) calculation
   step before the conclusion step.
4. PASS_NON_NUMERIC only fires when question_type ∈
   {boolean_check, symbolic_derivation}. Don't sweep numeric problems
   into this path to silence their errors.
5. No new dependencies. Pure Python stdlib only.
6. Regression budget — on EVAL COHORT only:
     PASS may not drop > 2
     PASS_NON_NUMERIC may not drop at all
     missing_q / missing_t / invalid_final_step may not increase > 2
   Full-set regression is informational, not a block.