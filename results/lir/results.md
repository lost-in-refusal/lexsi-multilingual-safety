
### Attack success, strict (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| plain | 6.9 (+0.0) | 19.2 (+14.1*) | 13.4 (+4.8*) | 7.6 (+1.4) | 11.8 |
| safegrad | 4.5 (-2.4) | 10.7 (+5.5*) | 8.9 (+0.3) | 5.5 (-0.7) | 7.4 |

### Attack success, loose (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 4.1 | 2.7 | 4.5 | 1.4 | 3.2 |
| plain | 4.5 (+0.3) | 14.1 (+11.3*) | 10.0 (+5.5*) | 1.7 (+0.3) | 7.6 |
| safegrad | 3.1 (-1.0) | 6.2 (+3.4*) | 5.5 (+1.0) | 1.4 (+0.0) | 4.0 |

### Refusal rate on attack prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 71.1 | 59.8 | 66.3 | 50.9 | 62.0 |
| plain | 66.7 (-4.5) | 47.4 (-12.4*) | 58.1 (-8.2*) | 37.8 (-13.1*) | 52.5 |
| safegrad | 75.9 (+4.8*) | 57.0 (-2.7) | 67.4 (+1.0) | 56.4 (+5.5*) | 64.2 |

### Over-refusal on XSTest safe prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 9.6 | 7.6 | 8.8 | 7.6 | 8.4 |
| plain | – | – | – | – | – |
| safegrad | – | – | – | – | – |

### Refusal on task prompts (%)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| base | 0.0 | 0.0 | 0.0 | 0.0 |
| plain | – | – | – | – |
| safegrad | – | – | – | – |

Parentheses: change vs the reference arm on the same prompts; * = p < 0.05 (exact McNemar for rates, centred paired bootstrap for task score). macro = equal-weight mean over languages.
