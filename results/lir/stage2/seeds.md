### Strict attack success by seed (%)

| Seed | Arm | en | ko | zh | sw | macro |
|---|---|---:|---:|---:|---:|---:|
| 42 | base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| 42 | plain | 6.9 (+0.0) | 19.2 (+14.1*) | 13.4 (+4.8*) | 7.6 (+1.4) | 11.8 |
| 42 | safegrad | 4.5 (-2.4) | 10.7 (-8.6*) | 8.9 (-4.5*) | 5.5 (-2.1) | 7.4 |
| 1 | base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| 1 | plain | 7.6 (+0.7) | 15.5 (+10.3*) | 12.4 (+3.8) | 5.2 (-1.0) | 10.1 |
| 1 | safegrad | 3.4 (-4.1*) | 8.2 (-7.2*) | 7.6 (-4.8*) | 3.4 (-1.7) | 5.7 |
| 2 | base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| 2 | plain | 8.9 (+2.1) | 18.9 (+13.7*) | 13.1 (+4.5*) | 6.9 (+0.7) | 11.9 |
| 2 | safegrad | 4.1 (-4.8*) | 9.6 (-9.3*) | 8.9 (-4.1*) | 6.5 (-0.3) | 7.3 |

Plain: change against base. SafeGrad: change against plain of the same seed. * = p < 0.05, exact McNemar.

### Preregistered checks

| Check | Per seed | Pooled (points, 95% interval) | Verdict |
|---|---|---|---|
| H1 Korean rises more than English | 42: yes, 1: yes, 2: yes | +11.8 [+6.8, +16.7] | pass |
| H2 SafeGrad below plain (Korean, macro) | 42: yes, 1: yes, 2: yes | -8.4 [-11.3, -5.5] | pass |
| H3 Swahili refusals fall | 42: yes, 1: yes, 2: yes | -9.3 [-13.7, -5.2] | pass |

Swahili refusal change, plain against base: seed 42: -13.1*; seed 1: -6.2*; seed 2: -8.6*.

Answers judged per arm: base 1164 (0 without a verdict), plain 1164 (0 without a verdict), safegrad 1164 (0 without a verdict), plain-s1 1164 (0 without a verdict), safegrad-s1 1164 (0 without a verdict), plain-s2 1164 (0 without a verdict), safegrad-s2 1164 (0 without a verdict).
