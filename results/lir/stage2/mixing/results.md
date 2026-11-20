
### Attack success, strict (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| safegrad | 4.5 | 10.7 | 8.9 | 5.5 | 7.4 |
| plain-mix1 | 3.8 (-0.7) | 9.3 (-1.4) | 6.5 (-2.4) | 4.8 (-0.7) | 6.1 |
| plain-mix5 | 3.4 (-1.0) | 8.9 (-1.7) | 7.9 (-1.0) | 4.8 (-0.7) | 6.3 |

### Attack success, loose (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| safegrad | 3.1 | 6.2 | 5.5 | 1.4 | 4.0 |
| plain-mix1 | 2.4 (-0.7) | 5.8 (-0.3) | 3.1 (-2.4*) | 0.7 (-0.7) | 3.0 |
| plain-mix5 | 2.7 (-0.3) | 4.1 (-2.1) | 4.5 (-1.0) | 1.0 (-0.3) | 3.1 |

### Refusal rate on attack prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| safegrad | 75.9 | 57.0 | 67.4 | 56.4 | 64.2 |
| plain-mix1 | 82.5 (+6.5*) | 60.8 (+3.8*) | 72.5 (+5.2*) | 56.7 (+0.3) | 68.1 |
| plain-mix5 | 84.2 (+8.2*) | 62.2 (+5.2*) | 73.2 (+5.8*) | 56.0 (-0.3) | 68.9 |

### Over-refusal on XSTest safe prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| safegrad | 7.6 | 6.8 | 5.6 | 5.6 | 6.4 |
| plain-mix1 | 14.0 (+6.4*) | 7.2 (+0.4) | 8.8 (+3.2*) | 7.6 (+2.0) | 9.4 |
| plain-mix5 | 13.2 (+5.6*) | 8.0 (+1.2) | 9.2 (+3.6*) | 6.4 (+0.8) | 9.2 |

### Refusal on task prompts (%)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| safegrad | 0.0 | 0.0 | 1.0 | 0.3 |
| plain-mix1 | 0.0 (+0.0) | 0.0 (+0.0) | 0.0 (-1.0) | 0.0 |
| plain-mix5 | 1.0 (+1.0) | 0.0 (+0.0) | 0.0 (-1.0) | 0.3 |

### Task score (1-7)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| safegrad | 6.33 | 5.40 | 5.72 | 5.82 |
| plain-mix1 | 6.37 (+0.04) | 5.61 (+0.21) | 5.86 (+0.14) | 5.95 |
| plain-mix5 | 6.33 (+0.00) | 5.64 (+0.25*) | 5.84 (+0.12) | 5.94 |

Parentheses: change vs the reference arm on the same prompts; * = p < 0.05 (exact McNemar for rates, centred paired bootstrap for task score). macro = equal-weight mean over languages.
