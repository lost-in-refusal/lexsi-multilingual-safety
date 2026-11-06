
### Attack success, strict (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| plain | 6.9 (+0.0) | 19.2 (+14.1*) | 13.4 (+4.8*) | 7.6 (+1.4) | 11.8 |
| plain-hannah | 13.4 (+6.5*) | 21.6 (+16.5*) | 13.4 (+4.8*) | 11.7 (+5.5*) | 15.0 |
| safegrad | 4.5 (-2.4) | 10.7 (+5.5*) | 8.9 (+0.3) | 5.5 (-0.7) | 7.4 |

### Attack success, loose (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 4.1 | 2.7 | 4.5 | 1.4 | 3.2 |
| plain | 4.5 (+0.3) | 14.1 (+11.3*) | 10.0 (+5.5*) | 1.7 (+0.3) | 7.6 |
| plain-hannah | 10.7 (+6.5*) | 16.8 (+14.1*) | 10.3 (+5.8*) | 3.4 (+2.1) | 10.3 |
| safegrad | 3.1 (-1.0) | 6.2 (+3.4*) | 5.5 (+1.0) | 1.4 (+0.0) | 4.0 |

### Refusal rate on attack prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 71.1 | 59.8 | 66.3 | 50.9 | 62.0 |
| plain | 66.7 (-4.5) | 47.4 (-12.4*) | 58.1 (-8.2*) | 37.8 (-13.1*) | 52.5 |
| plain-hannah | 58.8 (-12.4*) | 48.1 (-11.7*) | 59.8 (-6.5*) | 35.4 (-15.5*) | 50.5 |
| safegrad | 75.9 (+4.8*) | 57.0 (-2.7) | 67.4 (+1.0) | 56.4 (+5.5*) | 64.2 |

### Over-refusal on XSTest safe prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 9.6 | 7.6 | 8.8 | 7.6 | 8.4 |
| plain | 7.2 (-2.4) | 5.6 (-2.0) | 5.6 (-3.2) | 2.8 (-4.8*) | 5.3 |
| plain-hannah | 7.2 (-2.4) | 5.6 (-2.0) | 4.8 (-4.0*) | 4.0 (-3.6*) | 5.4 |
| safegrad | 7.6 (-2.0) | 6.8 (-0.8) | 5.6 (-3.2) | 5.6 (-2.0) | 6.4 |

### Refusal on task prompts (%)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| base | 0.0 | 0.0 | 0.0 | 0.0 |
| plain | 0.0 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 0.0 |
| plain-hannah | – | – | – | – |
| safegrad | 0.0 (+0.0) | 0.0 (+0.0) | 1.0 (+1.0) | 0.3 |

### Task score (1-7)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| base | 6.61 | 6.10 | 6.25 | 6.32 |
| plain | 6.36 (-0.25*) | 5.59 (-0.51*) | 5.86 (-0.39*) | 5.94 |
| plain-hannah | – | – | – | – |
| safegrad | 6.33 (-0.28*) | 5.40 (-0.70*) | 5.72 (-0.53*) | 5.82 |

Parentheses: change vs the reference arm on the same prompts; * = p < 0.05 (exact McNemar for rates, centred paired bootstrap for task score). macro = equal-weight mean over languages.
