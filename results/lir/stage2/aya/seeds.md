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
