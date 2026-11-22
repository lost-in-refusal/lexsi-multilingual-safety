### The seed 42 repair vector added to the Aya plain adapters

The vector is the SafeGrad adapter minus the plain adapter, both trained on Bactrian-X at seed 42. It is added at full size to the plain adapters trained on the Aya set, with no training and nothing tuned. Judged on the attack set with answers capped at 256 tokens. Section 12b of the preregistration fixed the rule before the runs. T1 is decided on seeds 1 and 2. The seed 42 adapter starts from the same LoRA initialisation as the vector's two source adapters, so its row is shown and decides nothing.

Strict attack success (%):

| Seed | Arm | en | ko | zh | sw | average |
|---|---|---:|---:|---:|---:|---:|
| 1 | base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| 1 | plain | 6.2 | 14.1 | 12.0 | 8.6 | 10.2 |
| 1 | safegrad | 5.5 | 11.0 | 8.6 | 7.2 | 8.1 |
| 1 | plain + vector | 10.0 (+3.8*) | 10.7 (-3.4) | 12.0 (+0.0) | 7.2 (-1.4) | 10.0 |
| 2 | base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| 2 | plain | 5.8 | 14.4 | 10.0 | 8.2 | 9.6 |
| 2 | safegrad | 5.5 | 9.6 | 8.2 | 8.6 | 8.0 |
| 2 | plain + vector | 8.9 (+3.1*) | 8.9 (-5.5*) | 10.3 (+0.3) | 5.2 (-3.1) | 8.3 |
| 42 | base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| 42 | plain | 6.9 | 15.1 | 11.7 | 6.9 | 10.1 |
| 42 | safegrad | 5.8 | 8.6 | 7.9 | 8.2 | 7.6 |
| 42 | plain + vector | 8.2 (+1.4) | 9.6 (-5.5*) | 12.0 (+0.3) | 7.2 (+0.3) | 9.3 |

Plain and SafeGrad are the adapters trained on Aya at that seed. Parentheses: change against plain of the same seed. * = p < 0.05, exact McNemar.

| T1 | Seed 1 | Seed 2 | Pooled Korean change (points, 95% interval) | Verdict |
|---|---|---|---|---|
| Below plain on Korean and on the four-language average | yes (-3.4, p = 0.143) | yes (-5.5, p = 0.009) | -4.5 [-8.2, -0.9] | pass |

**T1 passes: added to adapters trained on a second dataset at other seeds, the seed 42 repair vector lowered Korean strict attack success with no training. Over-refusal and task score of the edited models were not measured.**

#### Beside the rule, deciding nothing

- Seed 1, Korean: the edit gives 111% of the drop that training with SafeGrad on Aya gave; edited against SafeGrad -0.3 points (p = 1.000); edited against base +5.5 points (p = 0.005).
- Seed 2, Korean: the edit gives 114% of the drop that training with SafeGrad on Aya gave; edited against SafeGrad -0.7 points (p = 0.845); edited against base +3.8 points (p = 0.052).
- Seed 42, Korean: the edit gives 84% of the drop that training with SafeGrad on Aya gave; edited against SafeGrad +1.0 points (p = 0.701); edited against base +4.5 points (p = 0.024).

Refusal rate on attack prompts (%), with the change against base:

| Seed | Arm | en | ko | zh | sw |
|---|---|---:|---:|---:|---:|
| 1 | base | 71.1 | 59.8 | 66.3 | 50.9 |
| 1 | plain | 72.2 | 57.0 | 64.6 | 47.8 |
| 1 | plain + vector | 64.3 (-6.9*) | 58.4 (-1.4) | 60.5 (-5.8*) | 44.0 (-6.9*) |
| 2 | base | 71.1 | 59.8 | 66.3 | 50.9 |
| 2 | plain | 71.5 | 56.0 | 65.6 | 49.5 |
| 2 | plain + vector | 65.3 (-5.8*) | 58.8 (-1.0) | 61.5 (-4.8*) | 43.6 (-7.2*) |
| 42 | base | 71.1 | 59.8 | 66.3 | 50.9 |
| 42 | plain | 70.4 | 55.7 | 64.3 | 47.4 |
| 42 | plain + vector | 66.0 (-5.2*) | 60.1 (+0.3) | 61.9 (-4.5*) | 43.6 (-7.2*) |

Not measured: XSTest and the task sets were not run on the edited adapters, so the over-refusal and the task cost of the edit on Aya are unknown. On Bactrian-X the same edit raised English over-refusal to about 12%.

Edited adapters, checked before this was written: `aya-plain-s1-vec` is `aya-plain-s1` plus `safegrad` minus `plain`; `aya-plain-s2-vec` is `aya-plain-s2` plus `safegrad` minus `plain`; `aya-plain-vec` is `aya-plain` plus `safegrad` minus `plain`. Each record names its three source adapters by hash, and the hashes match the files.
