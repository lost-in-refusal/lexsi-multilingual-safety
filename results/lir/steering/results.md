
### Attack success, strict (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| plain | 6.9 | 19.2 | 13.4 | 7.6 | 11.8 |
| plain+steer-en-L17-c1 | 0.0 (-2.5 from 2.5) | 1.2 (-20.0* from 21.2) | 2.5 (-10.0* from 12.5) | 0.0 (-6.2 from 6.2) | 0.9 |
| plain+steer-native-L17-c1 | 0.0 (-2.5 from 2.5) | 10.0 (-11.2* from 21.2) | 2.5 (-10.0* from 12.5) | 5.0 (-1.2 from 6.2) | 4.4 |

### Attack success, loose (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| plain | 4.5 | 14.1 | 10.0 | 1.7 | 7.6 |
| plain+steer-en-L17-c1 | 0.0 (-2.5 from 2.5) | 0.0 (-12.5* from 12.5) | 2.5 (-7.5 from 10.0) | 0.0 (-1.2 from 1.2) | 0.6 |
| plain+steer-native-L17-c1 | 0.0 (-2.5 from 2.5) | 2.5 (-10.0* from 12.5) | 2.5 (-7.5 from 10.0) | 1.2 (+0.0 from 1.2) | 1.6 |

### Refusal rate on attack prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| plain | 66.7 | 47.4 | 58.1 | 37.8 | 52.5 |
| plain+steer-en-L17-c1 | 86.2 (+13.8* from 72.5) | 72.5 (+23.8* from 48.8) | 85.0 (+31.2* from 53.8) | 66.2 (+32.5* from 33.8) | 77.5 |
| plain+steer-native-L17-c1 | 86.2 (+13.8* from 72.5) | 67.5 (+18.8* from 48.8) | 78.8 (+25.0* from 53.8) | 55.0 (+21.2* from 33.8) | 71.9 |

### Over-refusal on XSTest safe prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| plain | 7.2 | 5.6 | 5.6 | 2.8 | 5.3 |
| plain+steer-en-L17-c1 | 37.5 (+26.2* from 11.2) | 23.8 (+17.5* from 6.2) | 20.0 (+12.5* from 7.5) | 20.0 (+16.2* from 3.8) | 25.3 |
| plain+steer-native-L17-c1 | 37.5 (+26.2* from 11.2) | 15.0 (+8.8* from 6.2) | 16.2 (+8.8* from 7.5) | 7.5 (+3.8 from 3.8) | 19.1 |

### Refusal on task prompts (%)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| plain | 0.0 | 0.0 | 0.0 | 0.0 |
| plain+steer-en-L17-c1 | – | – | – | – |
| plain+steer-native-L17-c1 | – | – | – | – |

### Task score (1-7)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| plain | 6.36 | 5.59 | 5.86 | 5.94 |
| plain+steer-en-L17-c1 | – | – | – | – |
| plain+steer-native-L17-c1 | – | – | – | – |

Parentheses: change vs the reference arm on the same prompts; * = p < 0.05 (exact McNemar for rates, centred paired bootstrap for task score). macro = equal-weight mean over languages.
