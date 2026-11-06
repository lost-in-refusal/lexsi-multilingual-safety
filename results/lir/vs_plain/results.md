
### Attack success, strict (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| plain | 6.9 | 19.2 | 13.4 | 7.6 | 11.8 |
| safegrad | 4.5 (-2.4) | 10.7 (-8.6*) | 8.9 (-4.5*) | 5.5 (-2.1) | 7.4 |

### Attack success, loose (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| plain | 4.5 | 14.1 | 10.0 | 1.7 | 7.6 |
| safegrad | 3.1 (-1.4) | 6.2 (-7.9*) | 5.5 (-4.5*) | 1.4 (-0.3) | 4.0 |

### Refusal rate on attack prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| plain | 66.7 | 47.4 | 58.1 | 37.8 | 52.5 |
| safegrad | 75.9 (+9.3*) | 57.0 (+9.6*) | 67.4 (+9.3*) | 56.4 (+18.6*) | 64.2 |

Parentheses: change vs the reference arm on the same prompts; * = p < 0.05 (exact McNemar for rates, centred paired bootstrap for task score). macro = equal-weight mean over languages.
