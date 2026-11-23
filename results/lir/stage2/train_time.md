### Training time on the laptop (minutes)

| Run | Seed | Rows | Optimizer steps | Minutes |
|---|---:|---:|---:|---:|
| plain | 42 | 1800 | 171 | 75.7 |
| safegrad | 42 | 1800 | 171 | 89.7 |
| plain-s1 | 1 | 1800 | 171 | 82.2 |
| safegrad-s1 | 1 | 1800 | 171 | 87.7 |
| plain-s2 | 2 | 1800 | 171 | 75.7 |
| safegrad-s2 | 2 | 1800 | 171 | 87.8 |
| plain-mix1 | 42 | 1847 | 174 | 75.6 |
| plain-mix5 | 42 | 2035 | 192 | 82.2 |
| earth-plain | 42 | 1800 | 171 | 74.4 |
| earth-safegrad | 42 | 1800 | 171 | 88.3 |

SafeGrad against plain at the same seed: seed 42: +14.0 min (+18.5%); seed 1: +5.5 min (+6.7%); seed 2: +12.1 min (+16.0%).
Mean over the seeds: +10.5 min (+13.7%).
