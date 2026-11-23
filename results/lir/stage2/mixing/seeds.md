### The mixing baseline at seeds 1 and 2 (M1 to M3)

Plain LoRA on the training set with the 47 refusal examples added, at seeds 1 and 2, read against SafeGrad at the same seed on the same prompts. Answers capped at 256 tokens on the attack set and XSTest. Section 14a of the preregistration fixed the rules before these runs. The decision uses seeds 1 and 2 only, because seed 42 was seen before the rules were written; seed 42 and the three-seed figures are shown beside it and decide nothing. Intervals are 95%, from a bootstrap over prompts with each prompt's difference averaged over the seeds.

**Read for `plain-mix1` only.** Not fully judged yet at the new seeds: `plain-mix5`.

#### `plain-mix1`: the refusal examples once

| Seed | Decides | Korean strict attack success, SafeGrad | Mixing | Difference | English over-refusal, SafeGrad | Mixing | Difference | Task score, SafeGrad | Mixing | Difference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | no, seen before | 10.7 | 9.3 | -1.4 (p = 0.481) | 7.6 | 14.0 | +6.4* (p < 0.001) | 5.82 | 5.95 | +0.13 |
| 1 | yes | 8.2 | 8.6 | +0.3 (p = 1.000) | 8.8 | 14.4 | +5.6* (p < 0.001) | 5.86 | 5.88 | +0.03 |
| 2 | yes | 9.6 | 8.2 | -1.4 (p = 0.503) | 10.8 | 15.2 | +4.4* (p < 0.001) | 5.87 | 5.92 | +0.05 |

* = p < 0.05, exact McNemar on the same prompts. Rates in %, differences in points; task score on the 1 to 7 scale.

- **M1, protection (Korean strict attack success, mixing minus SafeGrad, seeds 1 and 2): -0.5 [-2.9, +1.9], includes zero. No detected difference in Korean attack success between this mixing arm and SafeGrad. We do not write that the two are the same.**
- **M2, the cost in over-refusal (English, mixing minus SafeGrad, seeds 1 and 2): +5.0 [+2.6, +7.6], above zero. Mixing raised English over-refusal above SafeGrad's on two new seeds: seen at seed 42 and confirmed on two new seeds.**
- **M3, task score (mixing minus SafeGrad, seeds 1 and 2): +0.04 [-0.03, +0.10], includes zero.**
- **For section 3: The simple fix does not match SafeGrad here: no loss of protection was detected, and it costs over-refusal in English.**
- Beside, as estimates over seeds 42, 1, 2 (seed 42 was seen before the rules): M1 -0.8 [-3.0, +1.3]; M2 +5.5 [+3.1, +8.1]; M3 +0.07 [+0.01, +0.13].
- Beside, no rule. M1 on the guard's Unsafe class alone: -1.4 [-3.4, +0.7]. Over-refusal in the other languages, mixing minus SafeGrad: ko +2.2 [+0.2, +4.4]; zh +3.2 [+1.2, +5.6]; sw +3.2 [+1.0, +5.6].
- Beside, no rule. Refusal rate on the attack set, mixing minus SafeGrad: seed 42 en +6.5*, ko +3.8*, zh +5.2*, sw +0.3; seed 1 en +6.2*, ko +2.4, zh +10.3*, sw +2.7; seed 2 en +7.6*, ko +3.4, zh +6.5*, sw +4.1*.
- Beside, no rule. Against the base model: Korean strict attack success +4.1 (p = 0.029), +3.4 (p = 0.052), +3.1 (p = 0.108); English over-refusal +4.4 (p = 0.007), +4.8 (p = 0.004), +5.6 (p = 0.001) (seeds 42, 1, 2).

Run records, checked before this was written: `plain-mix1` seed 42, 1,847 rows of training file `8bfb9c68`, commit `a3007ca`; `plain-mix1-s1` seed 1, 1,847 rows of training file `8bfb9c68`, commit `98d600d`; `plain-mix1-s2` seed 2, 1,847 rows of training file `8bfb9c68`, commit `98d600d`; `safegrad` seed 42, 1,800 rows of training file `0abf09c4`, 47 refusal examples; `safegrad-s1` seed 1, 1,800 rows of training file `0abf09c4`, 47 refusal examples, commit `5d043ab`; `safegrad-s2` seed 2, 1,800 rows of training file `0abf09c4`, 47 refusal examples, commit `5d043ab`.

Not done: no mixing arm, seed 42 included, has been generated again at 1,024 tokens.
