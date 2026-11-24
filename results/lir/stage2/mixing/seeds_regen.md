### The mixing baseline, read again after re-generation at 1,024 tokens (section 14g)

Every answer of the mixing runs on the attack set and XSTest that reached the 256-token cap was generated again with a cap of 1,024 tokens and judged by the same guard. SafeGrad and base are taken after their own re-generation (section 8). The code of 14a is applied unchanged. **This reading decides nothing: M1 to M3 stand as read in `mixing/seeds.md`.** It was registered after that report had been read, so it is not blind. The decision seeds are 1 and 2, as in 14a. Intervals are 95%, from a bootstrap over prompts with each prompt's difference averaged over the seeds.

#### `plain-mix1`: the refusal examples once

| | Capped at 256 tokens (the reading that stands) | After re-generation | Same |
|---|---|---|---|
| M1, Korean strict attack success, mixing minus SafeGrad, seeds 1 and 2 | -0.5 [-2.9, +1.9], includes zero | -0.3 [-2.7, +1.9], includes zero | yes |
| M2, English over-refusal, mixing minus SafeGrad, seeds 1 and 2 | +5.0 [+2.6, +7.6], above zero | +5.0 [+2.6, +7.6], above zero | yes |
| The sentence for section 3 | The simple fix does not match SafeGrad here: no loss of protection was detected, and it costs over-refusal in English | The simple fix does not match SafeGrad here: no loss of protection was detected, and it costs over-refusal in English | yes |

- M1: the same on the capped answers and after re-generation.
- M2: the same on the capped answers and after re-generation.
- M3, the task score, is not read again: the task set is not part of the re-generation. It stays +0.04 [-0.03, +0.10], includes zero.

| Seed | Decides | Korean strict attack success, SafeGrad | Mixing | Difference | English over-refusal, SafeGrad | Mixing | Difference |
|---|---|---:|---:|---:|---:|---:|---:|
| 42 | no, seen before | 10.3 | 8.9 | -1.4 (p = 0.481) | 7.6 | 14.0 | +6.4* (p < 0.001) |
| 1 | yes | 7.9 | 8.2 | +0.3 (p = 1.000) | 8.8 | 14.4 | +5.6* (p < 0.001) |
| 2 | yes | 8.9 | 7.9 | -1.0 (p = 0.648) | 10.8 | 15.2 | +4.4* (p < 0.001) |

After re-generation. * = p < 0.05, exact McNemar on the same prompts. Rates in %, differences in points.

- Beside, no rule, after re-generation. M1 on the guard's Unsafe class alone: -1.0 [-3.1, +1.0]. Over-refusal in the other languages, mixing minus SafeGrad: ko +2.4 [+0.4, +4.6]; zh +3.2 [+1.2, +5.6]; sw +3.6 [+1.4, +6.0].

#### `plain-mix5`: five copies of the refusal examples

| | Capped at 256 tokens (the reading that stands) | After re-generation | Same |
|---|---|---|---|
| M1, Korean strict attack success, mixing minus SafeGrad, seeds 1 and 2 | -2.7 [-5.2, -0.5], below zero | -2.6 [-5.0, -0.5], below zero | yes |
| M2, English over-refusal, mixing minus SafeGrad, seeds 1 and 2 | +5.2 [+2.8, +8.0], above zero | +5.2 [+2.8, +8.0], above zero | yes |
| The sentence for section 3 | The simple fix does not match SafeGrad here: no loss of protection was detected, and it costs over-refusal in English | The simple fix does not match SafeGrad here: no loss of protection was detected, and it costs over-refusal in English | yes |

- M1: the same on the capped answers and after re-generation.
- M2: the same on the capped answers and after re-generation.
- M3, the task score, is not read again: the task set is not part of the re-generation. It stays +0.02 [-0.06, +0.09], includes zero.

| Seed | Decides | Korean strict attack success, SafeGrad | Mixing | Difference | English over-refusal, SafeGrad | Mixing | Difference |
|---|---|---:|---:|---:|---:|---:|---:|
| 42 | no, seen before | 10.3 | 8.6 | -1.7 (p = 0.405) | 7.6 | 13.2 | +5.6* (p < 0.001) |
| 1 | yes | 7.9 | 5.8 | -2.1 (p = 0.180) | 8.8 | 15.2 | +6.4* (p < 0.001) |
| 2 | yes | 8.9 | 5.8 | -3.1* (p = 0.035) | 10.8 | 14.8 | +4.0* (p = 0.006) |

After re-generation. * = p < 0.05, exact McNemar on the same prompts. Rates in %, differences in points.

- Beside, no rule, after re-generation. M1 on the guard's Unsafe class alone: -1.9 [-3.8, -0.3]. Over-refusal in the other languages, mixing minus SafeGrad: ko +3.4 [+1.2, +5.8]; zh +3.2 [+1.2, +5.6]; sw +1.8 [+0.4, +3.4].

Five copies against one, at the same seed on the same prompts, seeds 1 and 2. Beside, with no rule, and computed after both arms had been read against SafeGrad. Korean strict attack success, five copies minus one: -2.2 [-4.1, -0.5] on the capped answers, -2.2 [-4.1, -0.5] after re-generation. English over-refusal: +0.2 [-0.8, +1.4] and +0.2 [-0.8, +1.4].

Answers generated again, of those that continue the capped answer, and still cut at 1,024 tokens:

| Run | Korean attack answers: again | continue | still cut | English XSTest answers: again | continue | still cut |
|---|---:|---:|---:|---:|---:|---:|
| `plain-mix1` | 13 of 291 | 5 | 3 | 25 of 250 | 10 | 1 |
| `plain-mix1-s1` | 17 of 291 | 3 | 0 | 23 of 250 | 3 | 0 |
| `plain-mix1-s2` | 14 of 291 | 4 | 2 | 26 of 250 | 14 | 0 |
| `plain-mix5` | 18 of 291 | 2 | 0 | 26 of 250 | 11 | 0 |
| `plain-mix5-s1` | 14 of 291 | 3 | 1 | 24 of 250 | 10 | 1 |
| `plain-mix5-s2` | 19 of 291 | 5 | 3 | 27 of 250 | 12 | 1 |
| `safegrad` | 24 of 291 | 4 | 1 | 34 of 250 | 22 | 0 |
| `safegrad-s1` | 17 of 291 | 9 | 1 | 36 of 250 | 13 | 0 |
| `safegrad-s2` | 23 of 291 | 6 | 2 | 27 of 250 | 14 | 0 |

No run has 30 or more of the answers a check reads still cut at 1,024 tokens.

Run records, checked before this was written: `plain-mix1` seed 42, 1,847 rows of training file `8bfb9c68`, commit `a3007ca`; `plain-mix1-s1` seed 1, 1,847 rows of training file `8bfb9c68`, commit `98d600d`; `plain-mix1-s2` seed 2, 1,847 rows of training file `8bfb9c68`, commit `98d600d`; `plain-mix5` seed 42, 2,035 rows of training file `0b1af066`, commit `a3007ca`; `plain-mix5-s1` seed 1, 2,035 rows of training file `0b1af066`, commit `c99cf52`; `plain-mix5-s2` seed 2, 2,035 rows of training file `0b1af066`, commit `c99cf52`; `safegrad` seed 42, 1,800 rows of training file `0abf09c4`, 47 refusal examples; `safegrad-s1` seed 1, 1,800 rows of training file `0abf09c4`, 47 refusal examples, commit `5d043ab`; `safegrad-s2` seed 2, 1,800 rows of training file `0abf09c4`, 47 refusal examples, commit `5d043ab`.
