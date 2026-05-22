# Physics Dataset Analysis

- Dataset: `Physics_Problems_Text_Only.xlsx`
- Rows: 1755
- Columns: id, question, cot, answer, unit
- Fully labeled rows: 1340 (76.4%)
- Duplicate IDs: 0
- Duplicate non-empty question rows: 4
- Duplicate non-empty question groups: 2
- QA rows without gold answers: 401

## Prefix Summary

| prefix   |   rows |   labeled_rows |   missing_question |   missing_cot |   missing_answer |   missing_unit |   unique_units |   avg_question_words |   median_question_words |   avg_cot_words |   median_cot_words |   labeled_rate |
|:---------|-------:|---------------:|-------------------:|--------------:|-----------------:|---------------:|---------------:|---------------------:|------------------------:|----------------:|-------------------:|---------------:|
| QA       |    401 |              0 |                  1 |             0 |              401 |            401 |              0 |               54.481 |                    54   |         103.339 |               92   |          0     |
| LD       |    399 |            399 |                  0 |             2 |                0 |              0 |              9 |               47.218 |                    47   |         144.825 |              133   |          1     |
| CH       |    290 |            286 |                  0 |             0 |                0 |              4 |             11 |               28.779 |                    23.5 |          95.876 |               77.5 |          0.986 |
| NL       |    190 |            190 |                  0 |             0 |                0 |              0 |             13 |               24.8   |                    24   |         100.579 |               88   |          1     |
| TD       |    177 |            171 |                  0 |             0 |                0 |              6 |             17 |               31.932 |                    31   |          68.277 |               61   |          0.966 |
| DDT      |    130 |            127 |                  0 |             0 |                0 |              3 |             18 |               21.285 |                    22   |         145.838 |              133.5 |          0.977 |
| THCB     |     80 |             80 |                  0 |             0 |                0 |              0 |             15 |               21.475 |                    21   |          51.65  |               46.5 |          1     |
| DT       |     68 |             67 |                  0 |             0 |                0 |              1 |              9 |               51.103 |                    51   |         120.706 |              109   |          0.985 |
| CHLT     |     20 |             20 |                  0 |             0 |                0 |              0 |              1 |               28.1   |                    28   |         153.65  |              149.5 |          1     |

## Top Units

| unit      |   rows |   share |
|:----------|-------:|--------:|
| <missing> |    415 |  0.2365 |
| N         |    248 |  0.1413 |
| V/m       |    173 |  0.0986 |
| -         |    126 |  0.0718 |
| V         |     76 |  0.0433 |
| J         |     70 |  0.0399 |
| Ω         |     68 |  0.0387 |
| A         |     63 |  0.0359 |
| W         |     61 |  0.0348 |
| μF        |     41 |  0.0234 |
| pF        |     40 |  0.0228 |
| nC        |     40 |  0.0228 |
| %         |     39 |  0.0222 |
| mJ        |     37 |  0.0211 |
| nJ        |     35 |  0.0199 |
| Hz        |     30 |  0.0171 |
| H         |     29 |  0.0165 |
| mH        |     16 |  0.0091 |
| N/C       |     15 |  0.0085 |
| µF        |     14 |  0.008  |
| cm        |     10 |  0.0057 |
| μJ        |     10 |  0.0057 |
| C         |     10 |  0.0057 |
| T         |     10 |  0.0057 |
| Wb        |     10 |  0.0057 |
| μC        |      8 |  0.0046 |
| cm; %     |      7 |  0.004  |
| turns/m   |      7 |  0.004  |
| J/m³      |      4 |  0.0023 |
| g; g      |      3 |  0.0017 |

## Answer Format Summary

| answer_format       |   rows |
|:--------------------|-------:|
| plain_number        |    974 |
| missing             |    401 |
| scientific_notation |    222 |
| text_or_formula     |     75 |
| multi_value         |     28 |
| boolean             |     21 |
| other_numeric       |     20 |
| symbolic_sqrt       |      7 |
| fraction_or_ratio   |      7 |

## Frequent Question Keywords

| keyword     |   count |
|:------------|--------:|
| electric    |     785 |
| placed      |     556 |
| capacitor   |     511 |
| charges     |     492 |
| field       |     483 |
| point       |     462 |
| circuit     |     426 |
| voltage     |     369 |
| charge      |     347 |
| force       |     337 |
| energy      |     332 |
| distance    |     320 |
| speed       |     318 |
| three       |     264 |
| capacitance |     254 |
| current     |     254 |
| points      |     239 |
| acting      |     221 |
| car         |     219 |
| that        |     211 |
| each        |     196 |
| net         |     193 |
| time        |     193 |
| series      |     187 |
| air         |     178 |

## Quality Flags

| flag               |   rows |
|:-------------------|-------:|
| missing_unit       |    415 |
| missing_answer     |    401 |
| unit_dash_variant  |     42 |
| unit_micro_variant |     14 |
| duplicate_question |      4 |
| missing_cot        |      2 |
| missing_question   |      1 |

## Generated Files

- `prefix_summary.csv`
- `unit_summary.csv`
- `answer_format_summary.csv`
- `length_summary.csv`
- `quality_flags.csv`
- `sample_by_prefix.csv`
- `keyword_summary.json`

## Plots

- `plots/rows_by_prefix.png`
- `plots/missing_by_prefix.png`
- `plots/top_units.png`
- `plots/question_words_by_prefix.png`
