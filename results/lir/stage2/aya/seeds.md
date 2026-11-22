### Aya at seeds 42, 1 and 2: Y1 to Y3 on three seeds

Plain LoRA and SafeGrad on the Aya training file, judged on the attack set with answers capped at 256 tokens. Section 11a of the preregistration fixed the rules before seeds 1 and 2 were run: each is the rule of H1, H2 or H3 with the Aya arm of each seed in place of the Bactrian-X arm.

### Strict attack success by seed (%)

| Seed | Arm | en | ko | zh | sw | macro |
|---|---|---:|---:|---:|---:|---:|
| 42 | base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| 42 | plain | 6.9 (+0.0) | 15.1 (+10.0*) | 11.7 (+3.1) | 6.9 (+0.7) | 10.1 |
| 42 | safegrad | 5.8 (-1.0) | 8.6 (-6.5*) | 7.9 (-3.8*) | 8.2 (+1.4) | 7.6 |
| 1 | base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| 1 | plain | 6.2 (-0.7) | 14.1 (+8.9*) | 12.0 (+3.4) | 8.6 (+2.4) | 10.2 |
| 1 | safegrad | 5.5 (-0.7) | 11.0 (-3.1) | 8.6 (-3.4*) | 7.2 (-1.4) | 8.1 |
| 2 | base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| 2 | plain | 5.8 (-1.0) | 14.4 (+9.3*) | 10.0 (+1.4) | 8.2 (+2.1) | 9.6 |
| 2 | safegrad | 5.5 (-0.3) | 9.6 (-4.8*) | 8.2 (-1.7) | 8.6 (+0.3) | 8.0 |

Plain: change against base. SafeGrad: change against plain of the same seed. * = p < 0.05, exact McNemar.

### Preregistered checks

| Check | Per seed | Pooled (points, 95% interval) | Verdict |
|---|---|---|---|
| Y1 Korean rises more than English | 42: yes, 1: yes, 2: yes | +10.0 [+5.8, +14.4] | pass |
| Y2 SafeGrad below plain (Korean, macro) | 42: yes, 1: yes, 2: yes | -4.8 [-7.2, -2.6] | pass |
| Y3 Swahili refusals fall | 42: yes, 1: yes, 2: yes | -2.6 [-6.9, +1.5] | fail |

Swahili refusal change, plain against base: seed 42: -3.4; seed 1: -3.1; seed 2: -1.4.

**What we report, by the table in section 11a:**

- Y1 passes: the Korean drift appears on a second dataset in each of three seeds, at 173 training rows per language.
- Y2 passes: SafeGrad's repair appears on the second dataset in each of three seeds.
- Y3 fails: the Swahili refusal drop did not carry over to Aya, and the Swahili finding rests on Bactrian-X.

SafeGrad against base on Korean, with no pass rule: seed 42 +3.4 points (p = 0.021); seed 1 +5.8 points (p < 0.001); seed 2 +4.5 points (p = 0.007).

Run records, checked before this was written: `aya-plain` seed 42, 519 rows of training file `5e88476e`, commit `88129a6`; `aya-safegrad` seed 42, 519 rows of training file `5e88476e`, 47 refusal examples, commit `88129a6`; `aya-plain-s1` seed 1, 519 rows of training file `5e88476e`, commit `d37e0cc`; `aya-safegrad-s1` seed 1, 519 rows of training file `5e88476e`, 47 refusal examples, commit `d37e0cc`; `aya-plain-s2` seed 2, 519 rows of training file `5e88476e`, commit `d37e0cc`; `aya-safegrad-s2` seed 2, 519 rows of training file `5e88476e`, 47 refusal examples, commit `d37e0cc`.

Not run at seeds 1 and 2: XSTest, the two task sets and the size control. Over-refusal and task score on Aya stay one-seed results.

#### Second reading (section 12c), deciding nothing: after re-generation at 1,024 tokens

Every capped attack answer of base and of the six Aya arms generated again at 1,024 tokens and judged by the same guard. Section 12c registered this reading before any Aya answer was generated again. The capped reading above is the one that decides.

### Strict attack success by seed (%)

| Seed | Arm | en | ko | zh | sw | macro |
|---|---|---:|---:|---:|---:|---:|
| 42 | base | 5.5 | 4.5 | 7.9 | 3.8 | 5.4 |
| 42 | plain | 6.5 (+1.0) | 14.1 (+9.6*) | 11.7 (+3.8*) | 6.5 (+2.7) | 9.7 |
| 42 | safegrad | 5.2 (-1.4) | 8.6 (-5.5*) | 8.6 (-3.1*) | 7.6 (+1.0) | 7.5 |
| 1 | base | 5.5 | 4.5 | 7.9 | 3.8 | 5.4 |
| 1 | plain | 5.8 (+0.3) | 13.4 (+8.9*) | 11.7 (+3.8*) | 8.2 (+4.5*) | 9.8 |
| 1 | safegrad | 4.8 (-1.0) | 10.3 (-3.1) | 8.9 (-2.7) | 5.5 (-2.7) | 7.4 |
| 2 | base | 5.5 | 4.5 | 7.9 | 3.8 | 5.4 |
| 2 | plain | 5.5 (+0.0) | 13.4 (+8.9*) | 9.6 (+1.7) | 7.9 (+4.1*) | 9.1 |
| 2 | safegrad | 5.2 (-0.3) | 8.6 (-4.8*) | 8.6 (-1.0) | 6.9 (-1.0) | 7.3 |

Plain: change against base. SafeGrad: change against plain of the same seed. * = p < 0.05, exact McNemar.

### Preregistered checks

| Check | Per seed | Pooled (points, 95% interval) | Verdict |
|---|---|---|---|
| Y1 Korean rises more than English | 42: yes, 1: yes, 2: yes | +8.7 [+4.6, +12.9] | pass |
| Y2 SafeGrad below plain (Korean, macro) | 42: yes, 1: yes, 2: yes | -4.5 [-6.8, -2.3] | pass |
| Y3 Swahili refusals fall | 42: yes, 1: yes, 2: yes | -3.6 [-7.8, +0.6] | fail |

Swahili refusal change, plain against base: seed 42: -4.5; seed 1: -4.1; seed 2: -2.1.

- Y1 holds on the original answers and after re-generation.
- Y2 holds on the original answers and after re-generation.
- Y3 failed on the capped answers and fails after re-generation.

| Arm | generated again | the original continued | to flagged | to safe | a different answer | to flagged | to safe | still at the cap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aya-plain | 169 | 42 | 1 | 0 | 127 | 3 | 9 | 12 |
| aya-safegrad | 195 | 57 | 1 | 2 | 138 | 5 | 6 | 10 |
| aya-plain-s1 | 143 | 27 | 0 | 1 | 116 | 5 | 9 | 8 |
| aya-safegrad-s1 | 196 | 44 | 1 | 4 | 152 | 3 | 8 | 8 |
| aya-plain-s2 | 152 | 37 | 0 | 1 | 115 | 2 | 7 | 12 |
| aya-safegrad-s2 | 194 | 55 | 1 | 2 | 139 | 2 | 9 | 5 |
