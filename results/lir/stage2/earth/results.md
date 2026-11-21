
### Attack success, strict (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| earth-base | 7.9 | 10.7 | 11.0 | 7.6 | 9.3 |
| earth-plain | 8.9 (+1.0) | 21.3 (+10.7*) | 13.4 (+2.4) | 6.5 (-1.0) | 12.5 |
| earth-safegrad | 5.8 (-2.1) | 16.5 (+5.8*) | 8.9 (-2.1) | 5.5 (-2.1) | 9.2 |

### Attack success, loose (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| earth-base | 5.8 | 6.9 | 4.8 | 3.1 | 5.2 |
| earth-plain | 7.2 (+1.4) | 16.2 (+9.3*) | 9.3 (+4.5*) | 2.7 (-0.3) | 8.8 |
| earth-safegrad | 3.8 (-2.1) | 9.3 (+2.4) | 5.5 (+0.7) | 2.7 (-0.3) | 5.3 |

### Refusal rate on attack prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| earth-base | 67.7 | 53.3 | 59.8 | 43.3 | 56.0 |
| earth-plain | 64.9 (-2.7) | 47.8 (-5.5*) | 58.8 (-1.0) | 36.4 (-6.9*) | 52.0 |
| earth-safegrad | 71.5 (+3.8*) | 55.7 (+2.4) | 63.9 (+4.1*) | 45.0 (+1.7) | 59.0 |

### Over-refusal on XSTest safe prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| earth-base | 9.2 | 8.0 | 6.4 | 6.8 | 7.6 |
| earth-plain | 6.0 (-3.2*) | 4.8 (-3.2) | 4.8 (-1.6) | 4.0 (-2.8*) | 4.9 |
| earth-safegrad | 9.6 (+0.4) | 8.4 (+0.4) | 6.0 (-0.4) | 4.8 (-2.0) | 7.2 |

### Refusal on task prompts (%)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| earth-base | 0.0 | 0.0 | 0.0 | 0.0 |
| earth-plain | 0.0 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 0.0 |
| earth-safegrad | 0.0 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 0.0 |

### Task score (1-7)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| earth-base | 6.63 | 6.05 | 6.29 | 6.33 |
| earth-plain | 6.42 (-0.21*) | 5.34 (-0.72*) | 5.89 (-0.40*) | 5.88 |
| earth-safegrad | 6.44 (-0.19) | 5.39 (-0.67*) | 5.82 (-0.47*) | 5.88 |

Parentheses: change vs the reference arm on the same prompts; * = p < 0.05 (exact McNemar for rates, centred paired bootstrap for task score). macro = equal-weight mean over languages.
