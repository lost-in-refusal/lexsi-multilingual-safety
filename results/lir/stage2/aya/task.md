### Task score (1-7) on the held-out Aya prompts

| Arm | en | ko | zh | sw |
|---|---:|---:|---:|---:|
| base | 6.24 | 5.38 | 6.09 | 5.23 |
| plain | 6.33 (+0.09) | 5.05 (-0.33) | 6.01 (-0.08) | 4.95 (-0.28) |
| safegrad | 6.23 (-0.01) | 4.83 (-0.55*) | 6.04 (-0.05) | 4.47 (-0.76*) |
| aya-plain | 6.16 (-0.08) | 4.57 (-0.81*) | 6.08 (-0.01) | 4.62 (-0.61*) |
| aya-safegrad | 6.17 (-0.06) | 4.51 (-0.87*) | 6.09 (+0.00) | 4.50 (-0.73*) |

100 prompts per language, never trained on; Swahili is in no training set. Brackets: change against base on the same prompts, * = p < 0.05 (centred paired bootstrap). Scored by the same task judge as the Bactrian-X task set.

### Beside the table, not registered

Section 10 registers no test on this set, so everything here is description. Single runs at seed 42. The judge sees one reference answer per prompt, and no person has checked its scores on Aya prompts or in Swahili.

SafeGrad against plain trained on the same set: change in task score with its 95% interval, * = p < 0.05.

- Bactrian-X: en -0.10 [-0.25, +0.02], ko -0.22 [-0.52, +0.06], zh +0.03 [-0.14, +0.19], sw -0.48 [-0.80, -0.17]*
- Aya: en +0.01 [-0.06, +0.09], ko -0.06 [-0.26, +0.14], zh +0.01 [-0.13, +0.14], sw -0.12 [-0.39, +0.14]

Accuracy, the sub-score that moves most (1-7). The other three are in `task.json`.

| Arm | en | ko | zh | sw |
|---|---:|---:|---:|---:|
| base | 5.55 | 4.22 | 4.88 | 4.57 |
| plain | 5.44 | 3.54 | 4.66 | 3.80 |
| safegrad | 5.35 | 3.35 | 4.67 | 3.53 |
| aya-plain | 5.19 | 2.98 | 4.61 | 3.22 |
| aya-safegrad | 5.20 | 2.93 | 4.66 | 3.19 |

Median answer length in characters.

| Arm | en | ko | zh | sw |
|---|---:|---:|---:|---:|
| reference | 224 | 180 | 44 | 202 |
| base | 994 | 675 | 42 | 1052 |
| plain | 388 | 189 | 63 | 288 |
| safegrad | 398 | 188 | 43 | 408 |
| aya-plain | 208 | 124 | 34 | 99 |
| aya-safegrad | 178 | 108 | 34 | 114 |

Answers that reached the 1,024-token cap, of 100 per language.

| Arm | en | ko | zh | sw |
|---|---:|---:|---:|---:|
| base | 1 | 6 | 1 | 4 |
| plain | 0 | 6 | 0 | 2 |
| safegrad | 0 | 8 | 1 | 6 |
| aya-plain | 0 | 10 | 2 | 4 |
| aya-safegrad | 1 | 9 | 1 | 6 |
