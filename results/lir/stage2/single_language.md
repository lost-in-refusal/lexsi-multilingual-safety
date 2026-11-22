### One training language at a time (seed 42)

Plain LoRA on the 600 English rows, the 600 Korean rows or the 600 Chinese rows of the Bactrian-X training set: the same 600 items, each in one language. Judged on the attack set with answers capped at 256 tokens. Section 12a of the preregistration fixed both rules before the runs. One seed, so each row is one run.

Strict attack success (%), with the change against base:

| Trained on | en | ko | zh | sw | average |
|---|---:|---:|---:|---:|---:|
| nothing (base) | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| English only | 7.6 (+0.7) | 13.1 (+7.9*) | 11.0 (+2.4) | 8.2 (+2.1) | 10.0 |
| Korean only | 7.6 (+0.7) | 17.2 (+12.0*) | 8.6 (+0.0) | 5.5 (-0.7) | 9.7 |
| Chinese only | 8.2 (+1.4) | 16.5 (+11.3*) | 12.7 (+4.1*) | 6.9 (+0.7) | 11.1 |

* = p < 0.05, exact McNemar on the same prompts.

**L1, does Korean drift** (K1's table; the bar is 6.36 points):

- English only: Korean +7.9 points (p < 0.001), outcome row 1. Training on English alone is enough for a Korean drift of this size.
- Korean only: Korean +12.0 points (p < 0.001), outcome row 1. Training on Korean alone is enough for a Korean drift of this size.
- Chinese only: Korean +11.3 points (p < 0.001), outcome row 1. Training on Chinese alone is enough for a Korean drift of this size.

**L2, which language drifts most** (Korean rise minus the rise in the language trained on, in points, with a 95% interval over the 291 prompt families):

- English only: Korean +7.9 against English +0.7; gap +7.2 [+3.1, +11.7], above zero.
- Chinese only: Korean +11.3 against Chinese +4.1; gap +7.2 [+2.1, +12.7], above zero.

**By the table in section 12a (row 1): Trained on English alone or on Chinese alone, the model drifted more in Korean than in the language it was trained on. That supports reading H1 as a weakness of Korean.**

The Korean-only run is read by L1 only: it cannot tell a weak language from a trained one.

#### Shown with no pass rule

| Trained on | Swahili refusal change | Korean against the mixed run, seed 42 | Korean against the mixed run cut to 519 rows |
|---|---:|---:|---:|
| English only | -5.2 (p = 0.040) | -6.2 (p = 0.008) | -4.5 (p = 0.041) |
| Korean only | -6.2 (p = 0.020) | -2.1 (p = 0.405) | -0.3 (p = 1.000) |
| Chinese only | -7.2 (p = 0.003) | -2.7 (p = 0.256) | -1.0 (p = 0.690) |

Each single-language run has 600 rows where the mixed run has 1,800, so the mixed run cut to 519 rows is the nearest comparison. No test compares the three runs with each other.

Run records, checked before this was written: `plain-only-en` seed 42, 600 rows of training file `8e11f97e`, commit `39312df`; `plain-only-ko` seed 42, 600 rows of training file `de63e7fe`, commit `39312df`; `plain-only-zh` seed 42, 600 rows of training file `8a487c1c`, commit `39312df`.
