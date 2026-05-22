# Question Classification

- Dataset: `Physics_Problems_Text_Only.xlsx`
- Rows classified: 1755
- Domains: 7
- Topics: 14
- Subtopics: 44
- Low-confidence rows needing manual review: 0

## Topic Summary

| domain                    | topic                       |   rows |   share |
|:--------------------------|:----------------------------|-------:|--------:|
| Electricity and magnetism | Electrostatics              |    499 |  0.2843 |
| Electricity and magnetism | AC circuits and resonance   |    357 |  0.2034 |
| Electricity and magnetism | Capacitors                  |    332 |  0.1892 |
| Mechanics                 | Motion                      |    198 |  0.1128 |
| Electricity and magnetism | Magnetism and inductors     |    128 |  0.0729 |
| Electricity and magnetism | Basic circuits              |     73 |  0.0416 |
| Measurement               | Measurement and uncertainty |     65 |  0.037  |
| Waves and oscillations    | Oscillations and waves      |     32 |  0.0182 |
| Optics                    | Geometric optics            |     26 |  0.0148 |
| Electricity and magnetism | Electromagnetic induction   |     22 |  0.0125 |
| Mechanics                 | Energy                      |     11 |  0.0063 |
| Thermodynamics            | Heat and temperature        |      6 |  0.0034 |
| Optics                    | Wave optics                 |      5 |  0.0028 |
| Data quality              | Missing question            |      1 |  0.0006 |

## Task Type Summary

| task_type                    | answer_mode         |   rows |   share |
|:-----------------------------|:--------------------|-------:|--------:|
| single_numeric_calculation   | numeric             |    894 |  0.5094 |
| unlabeled_question           | unlabeled           |    414 |  0.2359 |
| single_numeric_calculation   | scientific_notation |    220 |  0.1254 |
| qualitative_explanation      | text                |    101 |  0.0575 |
| ratio_or_percent_calculation | numeric             |     71 |  0.0405 |
| multi_part_calculation       | multi_value         |     25 |  0.0142 |
| yes_no_judgment              | boolean             |     21 |  0.012  |
| single_numeric_calculation   | symbolic            |      6 |  0.0034 |
| ratio_or_percent_calculation | scientific_notation |      2 |  0.0011 |
| missing_question             | unlabeled           |      1 |  0.0006 |

## Top Subtopics

| domain                    | topic                       | subtopic                                |   rows |   share |
|:--------------------------|:----------------------------|:----------------------------------------|-------:|--------:|
| Electricity and magnetism | Electrostatics              | Electric force vector composition       |    240 |  0.1368 |
| Electricity and magnetism | Electrostatics              | Electric field from point charges       |    204 |  0.1162 |
| Electricity and magnetism | Capacitors                  | Capacitor energy                        |    159 |  0.0906 |
| Electricity and magnetism | AC circuits and resonance   | RLC impedance/resistance                |    150 |  0.0855 |
| Electricity and magnetism | Capacitors                  | Parallel-plate capacitor and dielectric |    143 |  0.0815 |
| Electricity and magnetism | AC circuits and resonance   | RLC capacitance                         |    111 |  0.0632 |
| Mechanics                 | Motion                      | Kinematics                              |    108 |  0.0615 |
| Mechanics                 | Motion                      | Meeting-point motion                    |     86 |  0.049  |
| Electricity and magnetism | Magnetism and inductors     | Inductor energy                         |     63 |  0.0359 |
| Electricity and magnetism | Electrostatics              | Coulomb force                           |     51 |  0.0291 |
| Electricity and magnetism | AC circuits and resonance   | RLC frequency                           |     44 |  0.0251 |
| Electricity and magnetism | Basic circuits              | Ohm's law                               |     43 |  0.0245 |
| Measurement               | Measurement and uncertainty | Absolute error and bounds               |     42 |  0.0239 |
| Electricity and magnetism | Magnetism and inductors     | Solenoid magnetic field                 |     41 |  0.0234 |
| Electricity and magnetism | AC circuits and resonance   | Resonance yes/no judgment               |     36 |  0.0205 |
| Optics                    | Geometric optics            | Lens/image calculation                  |     26 |  0.0148 |
| Electricity and magnetism | Electromagnetic induction   | Induced EMF                             |     22 |  0.0125 |
| Electricity and magnetism | Capacitors                  | Capacitor charge                        |     21 |  0.012  |
| Waves and oscillations    | Oscillations and waves      | Simple harmonic motion                  |     20 |  0.0114 |
| Electricity and magnetism | Basic circuits              | Electrical power                        |     16 |  0.0091 |
| Electricity and magnetism | Basic circuits              | Circuit qualitative behavior            |     13 |  0.0074 |
| Measurement               | Measurement and uncertainty | Measurement uncertainty                 |     12 |  0.0068 |
| Mechanics                 | Energy                      | Mechanical energy                       |     11 |  0.0063 |
| Electricity and magnetism | Magnetism and inductors     | Inductance                              |     11 |  0.0063 |
| Measurement               | Measurement and uncertainty | Relative error                          |     11 |  0.0063 |
| Electricity and magnetism | AC circuits and resonance   | RLC inductance                          |      9 |  0.0051 |
| Waves and oscillations    | Oscillations and waves      | LC oscillation                          |      8 |  0.0046 |
| Electricity and magnetism | AC circuits and resonance   | RLC resonance calculation               |      7 |  0.004  |
| Thermodynamics            | Heat and temperature        | Thermal calculation                     |      6 |  0.0034 |
| Electricity and magnetism | Magnetism and inductors     | Solenoid turn density                   |      6 |  0.0034 |

## Prefix x Topic

| prefix   |   AC circuits and resonance |   Basic circuits |   Capacitors |   Electromagnetic induction |   Electrostatics |   Energy |   Geometric optics |   Heat and temperature |   Magnetism and inductors |   Measurement and uncertainty |   Missing question |   Motion |   Oscillations and waves |   Wave optics |
|:---------|----------------------------:|-----------------:|-------------:|----------------------------:|-----------------:|---------:|-------------------:|-----------------------:|--------------------------:|------------------------------:|-------------------:|---------:|-------------------------:|--------------:|
| CH       |                         274 |               11 |            5 |                           0 |                0 |        0 |                  0 |                      0 |                         0 |                             0 |                  0 |        0 |                        0 |             0 |
| CHLT     |                          20 |                0 |            0 |                           0 |                0 |        0 |                  0 |                      0 |                         0 |                             0 |                  0 |        0 |                        0 |             0 |
| DDT      |                          29 |                1 |           16 |                          20 |                0 |        0 |                  0 |                      0 |                        61 |                             0 |                  0 |        0 |                        3 |             0 |
| DT       |                           0 |                0 |            9 |                           0 |               58 |        0 |                  0 |                      0 |                         0 |                             0 |                  0 |        1 |                        0 |             0 |
| LD       |                           0 |                0 |            1 |                           0 |              398 |        0 |                  0 |                      0 |                         0 |                             0 |                  0 |        0 |                        0 |             0 |
| NL       |                           1 |                1 |          119 |                           0 |                0 |        0 |                  0 |                      0 |                        64 |                             0 |                  0 |        0 |                        5 |             0 |
| QA       |                          33 |               42 |            5 |                           2 |               43 |       11 |                 26 |                      6 |                         3 |                             3 |                  1 |      197 |                       24 |             5 |
| TD       |                           0 |                0 |          177 |                           0 |                0 |        0 |                  0 |                      0 |                         0 |                             0 |                  0 |        0 |                        0 |             0 |
| THCB     |                           0 |               18 |            0 |                           0 |                0 |        0 |                  0 |                      0 |                         0 |                            62 |                  0 |        0 |                        0 |             0 |

## Low-Confidence Examples

No low-confidence examples.

## Generated Files

- `classified_questions.csv`
- `topic_summary.csv`
- `subtopic_summary.csv`
- `prefix_topic_summary.csv`
- `task_type_summary.csv`
- `low_confidence_review.csv`

## Plots

- `plots/topics.png`
- `plots/domain_by_prefix.png`
- `plots/task_types.png`
