# Question Classification Rules

This directory contains rule-based classifications for `Physics_Problems_Text_Only.xlsx`.
The rules are implemented in `classify_questions.py` and are intended for dataset analysis, sampling, and benchmark slicing.

## Run

```powershell
python classify_questions.py --data Physics_Problems_Text_Only.xlsx
```

Optional arguments:

- `--sheet`: Excel sheet name. Defaults to the first sheet.
- `--output-dir`: output directory. Defaults to `outputs/question_classification`.
- `--no-plots`: skip PNG plot generation.

## Output Files

- `classified_questions.csv`: one row per source question, with classification columns.
- `classification_report.md`: human-readable summary.
- `topic_summary.csv`: row counts by `domain` and `topic`.
- `subtopic_summary.csv`: row counts by `domain`, `topic`, and `subtopic`.
- `prefix_topic_summary.csv`: cross-tab of original ID prefix and topic.
- `task_type_summary.csv`: row counts by task type and answer mode.
- `low_confidence_review.csv`: rows assigned low-confidence labels.
- `plots/`: generated topic/task distribution charts.

## Classification Fields

`classified_questions.csv` keeps the original dataset columns and adds:

- `prefix`: alphabetic prefix parsed from `id`, such as `LD`, `TD`, or `QA`.
- `domain`: broad physics area, such as `Electricity and magnetism`, `Mechanics`, or `Optics`.
- `topic`: mid-level concept group, such as `Electrostatics` or `Capacitors`.
- `subtopic`: more specific rule match, such as `Electric field from point charges`.
- `confidence`: `high`, `medium`, or `low`.
- `matched_rules`: short description of the rule that fired.
- `task_type`: expected problem interaction type.
- `answer_mode`: answer format inferred from `answer` and `unit`.
- `question_text`: stripped question text for easier review.

## Rule Priority

Rules are order-dependent. The first matching topic rule wins. This matters for questions that contain overlapping keywords, for example capacitor questions that include `Q = ...`.

Current topic priority:

1. Missing question text
2. AC circuits and resonance
3. Capacitors
4. Electromagnetic induction
5. Electrical energy efficiency
6. Electric force vector composition
7. Oscillations and waves
8. Magnetism and inductors
9. Point-charge electrostatics
10. Measurement and uncertainty
11. Basic circuits
12. Geometric optics
13. Wave optics
14. Motion
15. Mechanical energy
16. Heat and temperature
17. Unclassified

## Topic Rules

### AC Circuits And Resonance

Matched by keywords such as `rlc`, `resonance`, `resonant`, `ac circuit`, or `angular frequency`.

Subtopics:

- `Resonance yes/no judgment`: asks whether resonance occurs.
- `RLC impedance/resistance`: mentions resistance or impedance.
- `RLC capacitance`: asks for capacitance in an RLC context.
- `RLC inductance`: asks for inductance in an RLC context.
- `RLC frequency`: asks for frequency or angular frequency.
- `RLC resonance calculation`: fallback for other resonance problems.

### Capacitors

Matched by keywords such as `capacitor`, `capacitance`, `parallel-plate`, `dielectric`, `permittivity`, or `electric field energy`.

Subtopics:

- `Parallel-plate capacitor and dielectric`
- `Capacitor energy`
- `Capacitor charge`
- `Capacitor voltage`
- `Capacitance calculation`
- `Capacitor calculation`

This rule intentionally runs before point-charge electrostatics, because capacitor questions often use `Q`.

### Electromagnetic Induction

Matched by `induced electromotive force`, `emf`, or `electromotive force`.

Subtopic:

- `Induced EMF`

### Electric Force Vector Composition

Matched by electric-force/resultant-force wording without necessarily naming multiple point charges.

Subtopic:

- `Electric force vector composition`

### Oscillations And Waves

Matched by `lc circuit`, `simple harmonic`, `oscillation`, `oscillations`, `spring pendulum`, `amplitude`, `initial phase`, or sound pitch wording.

Subtopics:

- `LC oscillation`
- `Sound frequency`
- `Simple harmonic motion`
- `Oscillation calculation`

### Magnetism And Inductors

Matched by `solenoid`, `magnetic field`, `magnetic flux`, `magnetic field energy`, `inductor`, or `inductance`.

Subtopics:

- `Solenoid magnetic field`
- `Solenoid turn density`
- `Magnetic flux`
- `Inductor energy`
- `Inductance`
- `Magnetism calculation`

### Electrostatics

Matched by point-charge keywords such as `point charge`, `electric charge`, `charges`, `q1`, `q2`, `coulomb`, `test charge`, or a `q = ...` pattern.

Subtopics:

- `Electric field from point charges`
- `Coulomb force`
- `Electric potential`
- `Point-charge electrostatics`

### Measurement And Uncertainty

Matched by `ammeter`, `voltmeter`, `least count`, `absolute error`, `relative error`, `uncertainty`, or `measurement`.

Subtopics:

- `Absolute error and bounds`
- `Relative error`
- `Measurement uncertainty`

### Basic Circuits

Matched by `resistor`, `resistance`, `current`, `voltage`, `power`, `lamp`, `bulb`, `parallel circuit`, or `series circuit`.

Subtopics:

- `Electrical power`
- `Circuit qualitative behavior`
- `Ohm's law`
- `Basic circuit calculation`
- `Electrical energy efficiency`

### Optics

Geometric optics is matched by `lens`, `focal length`, `image`, `object distance`, `mirror`, or `principal axis`.

Wave optics is matched by `young`, `double-slit`, `interference`, `monochromatic`, `wavelength`, `refractive index`, or `light ray`.

Subtopics:

- `Lens/image calculation`
- `Interference`
- `Refraction`
- `Wave optics`

### Mechanics

Motion is matched by `car`, `motorbike`, `motorboat`, `airplane`, `travels`, `speed`, `velocity`, `distance`, `time`, `downstream`, or `upstream`.

Mechanical energy is matched by `mass`, `height`, `potential energy`, `kinetic energy`, `mechanical energy`, `dropped`, `gravity`, or `work`.

Subtopics:

- `Relative motion in current`
- `Meeting-point motion`
- `Kinematics`
- `Mechanical energy`

### Thermodynamics

Matched by `temperature`, `heat`, `thermal`, `specific heat`, or `celsius`.

Subtopic:

- `Thermal calculation`

## Task Type Rules

`task_type` is separate from physics topic.

- `missing_question`: empty question text.
- `unlabeled_question`: missing `answer` or `unit`.
- `yes_no_judgment`: boolean answer, or question starts with `does`, `do`, `will`, `can`, `should`, or contains `whether` / `determine if`.
- `qualitative_explanation`: text answer.
- `multi_part_calculation`: answer or unit contains `;`.
- `ratio_or_percent_calculation`: question mentions `ratio`, `percentage`, `relative error`, or `efficiency`.
- `single_numeric_calculation`: fallback for numeric/symbolic/scientific answers.

## Answer Mode Rules

`answer_mode` is inferred from `answer` and `unit`.

- `unlabeled`: missing answer or unit.
- `boolean`: answer is `yes`, `no`, `true`, or `false`.
- `multi_value`: answer or unit contains `;`.
- `symbolic`: answer contains `sqrt` or `\sqrt`.
- `scientific_notation`: answer uses scientific notation such as `x 10`, `* 10`, `× 10`, or `e-3`.
- `numeric`: plain integer or decimal.
- `text`: fallback for other non-empty answers.

## Current Classification Summary

The latest generated run classifies 1755 rows into 14 topics. The largest groups are:

- `Electrostatics`: 499 rows
- `AC circuits and resonance`: 357 rows
- `Capacitors`: 332 rows
- `Motion`: 198 rows
- `Magnetism and inductors`: 128 rows

`low_confidence_review.csv` is currently empty except for the header, meaning all rows are assigned to a concrete rule or known data-quality category.

## Maintenance Notes

When adding or changing rules:

- Keep more specific rules above broader rules.
- Check `low_confidence_review.csv` after every run.
- Compare `prefix_topic_summary.csv` before and after changes to catch accidental broad reclassification.
- Re-run `python -m py_compile classify_questions.py` after edits.
