### Tiny Aya Earth at seeds 42, 1 and 2: E1 on three seeds

Plain LoRA and SafeGrad on Tiny Aya Earth, judged on the attack set. Base Earth is one run. Section 11c of the preregistration fixed the rule before seeds 1 and 2 were run. E1 is read in its own terms, on the answers capped at 256 tokens: no comparison with English and no four-language average.

Strict attack success (%), answers capped at 256 tokens:

| Seed | Arm | en | ko | zh | sw |
|---|---|---:|---:|---:|---:|
| 42 | base | 7.9 | 10.7 | 11.0 | 7.6 |
| 42 | plain | 8.9 (+1.0) | 21.3 (+10.7*) | 13.4 (+2.4) | 6.5 (-1.0) |
| 42 | safegrad | 5.8 (-3.1*) | 16.5 (-4.8*) | 8.9 (-4.5*) | 5.5 (-1.0) |
| 1 | base | 7.9 | 10.7 | 11.0 | 7.6 |
| 1 | plain | 7.9 (+0.0) | 19.2 (+8.6*) | 12.7 (+1.7) | 7.6 (+0.0) |
| 1 | safegrad | 5.5 (-2.4) | 13.4 (-5.8*) | 9.3 (-3.4*) | 6.5 (-1.0) |
| 2 | base | 7.9 | 10.7 | 11.0 | 7.6 |
| 2 | plain | 7.9 (+0.0) | 21.6 (+11.0*) | 15.5 (+4.5*) | 8.6 (+1.0) |
| 2 | safegrad | 5.2 (-2.7*) | 13.7 (-7.9*) | 10.0 (-5.5*) | 5.8 (-2.7) |

Plain: change against base Earth. SafeGrad: change against plain of the same seed. * = p < 0.05, exact McNemar.

| E1 on three seeds | Per seed | Pooled Korean change (points, 95% interval) | Verdict |
|---|---|---|---|
| Drift: Korean rises significantly over base Earth | 42: yes, 1: yes, 2: yes | +10.1 [+6.4, +14.0] | pass |
| Repair: SafeGrad below plain of the same seed on Korean | 42: yes, 1: yes, 2: yes | -6.2 [-9.2, -3.4] | pass |

**Drift and repair both pass: E1 holds on a second model in each of three seeds.**

#### Shown with no pass rule

SafeGrad against base Earth on Korean: seed 42 +5.8 points (p = 0.008); seed 1 +2.7 points (p = 0.200); seed 2 +3.1 points (p = 0.136).

Swahili refusal rate under plain against base, Earth and then Tiny Aya Global at the same seed: seed 42 -6.9 (p = 0.006) and -13.1; seed 1 -4.8 (p = 0.038) and -6.2; seed 2 -3.8 (p = 0.135) and -8.6. E2 and row C keep their Nov 20 readings.

The second reading, on answers generated again at 1,024 tokens, is not available yet for: earth-plain-s1, earth-safegrad-s1, earth-plain-s2, earth-safegrad-s2.

Run records, checked before this was written: `earth-plain` seed 42, 1,800 rows of training file `0abf09c4`, commit `45a83d8`; `earth-safegrad` seed 42, 1,800 rows of training file `0abf09c4`, 44 refusal examples, commit `b0cd6ef`; `earth-plain-s1` seed 1, 1,800 rows of training file `0abf09c4`, commit `5743aef`; `earth-safegrad-s1` seed 1, 1,800 rows of training file `0abf09c4`, 44 refusal examples, commit `5743aef`; `earth-plain-s2` seed 2, 1,800 rows of training file `0abf09c4`, commit `5743aef`; `earth-safegrad-s2` seed 2, 1,800 rows of training file `0abf09c4`, 44 refusal examples, commit `a551f97`.

Not run at seeds 1 and 2: XSTest and the task set.
