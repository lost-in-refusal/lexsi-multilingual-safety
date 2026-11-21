
### Attack success, strict (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| aya-plain | 6.9 (+0.0) | 15.1 (+10.0*) | 11.7 (+3.1) | 6.9 (+0.7) | 10.1 |
| aya-safegrad | 5.8 (-1.0) | 8.6 (+3.4*) | 7.9 (-0.7) | 8.2 (+2.1) | 7.6 |

### Attack success, loose (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 4.1 | 2.7 | 4.5 | 1.4 | 3.2 |
| aya-plain | 5.5 (+1.4) | 10.3 (+7.6*) | 7.6 (+3.1) | 2.1 (+0.7) | 6.4 |
| aya-safegrad | 4.1 (+0.0) | 5.5 (+2.7*) | 4.5 (+0.0) | 1.4 (+0.0) | 3.9 |

### Refusal rate on attack prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 71.1 | 59.8 | 66.3 | 50.9 | 62.0 |
| aya-plain | 70.4 (-0.7) | 55.7 (-4.1) | 64.3 (-2.1) | 47.4 (-3.4) | 59.5 |
| aya-safegrad | 75.3 (+4.1*) | 59.5 (-0.3) | 68.0 (+1.7) | 52.2 (+1.4) | 63.7 |

### Over-refusal on XSTest safe prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 9.6 | 7.6 | 8.8 | 7.6 | 8.4 |
| aya-plain | 8.8 (-0.8) | 9.2 (+1.6) | 8.0 (-0.8) | 6.0 (-1.6) | 8.0 |
| aya-safegrad | 11.2 (+1.6) | 10.0 (+2.4) | 10.4 (+1.6) | 7.6 (+0.0) | 9.8 |

### Refusal on task prompts (%)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| base | 0.0 | 0.0 | 0.0 | 0.0 |
| aya-plain | 0.0 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 0.0 |
| aya-safegrad | 0.0 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 0.0 |

### Task score (1-7)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| base | 6.61 | 6.10 | 6.25 | 6.32 |
| aya-plain | 6.41 (-0.20*) | 5.84 (-0.26) | 5.79 (-0.46*) | 6.02 |
| aya-safegrad | 6.42 (-0.20*) | 5.82 (-0.28*) | 5.85 (-0.40*) | 6.03 |

Parentheses: change vs the reference arm on the same prompts; * = p < 0.05 (exact McNemar for rates, centred paired bootstrap for task score). macro = equal-weight mean over languages.
