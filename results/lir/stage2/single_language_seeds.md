### One training language at a time, seeds 42, 1 and 2

Plain LoRA on the 600 English, Korean or Chinese rows of the Bactrian-X training set, at three seeds, judged on the attack set with answers capped at 256 tokens. Section 13a of the preregistration fixed the three-seed rules before the seed 42 runs; seeds 1 and 2 were run on Nov 22 before the freeze. `single_language.md` has the seed 42 reading and its second reading after re-generation.

Strict attack success (%), with the change against base:

| Seed | Trained on | en | ko | zh | sw | SL1 row |
|---|---|---:|---:|---:|---:|---:|
| 42 | English only | 7.6 (+0.7) | 13.1 (+7.9*) | 11.0 (+2.4) | 8.2 (+2.1) | 1 |
| 42 | Korean only | 7.6 (+0.7) | 17.2 (+12.0*) | 8.6 (+0.0) | 5.5 (-0.7) | 1 |
| 42 | Chinese only | 8.2 (+1.4) | 16.5 (+11.3*) | 12.7 (+4.1*) | 6.9 (+0.7) | 1 |
| 1 | English only | 7.6 (+0.7) | 14.1 (+8.9*) | 11.0 (+2.4) | 8.9 (+2.7) | 1 |
| 1 | Korean only | 7.2 (+0.3) | 14.8 (+9.6*) | 10.7 (+2.1) | 7.6 (+1.4) | 1 |
| 1 | Chinese only | 8.6 (+1.7) | 16.2 (+11.0*) | 12.4 (+3.8*) | 5.5 (-0.7) | 1 |
| 2 | English only | 6.9 (+0.0) | 12.4 (+7.2*) | 10.3 (+1.7) | 9.6 (+3.4) | 1 |
| 2 | Korean only | 7.2 (+0.3) | 17.2 (+12.0*) | 9.6 (+1.0) | 5.5 (-0.7) | 1 |
| 2 | Chinese only | 8.6 (+1.7) | 17.5 (+12.4*) | 11.7 (+3.1) | 5.2 (-1.0) | 1 |

* = p < 0.05, exact McNemar on the same prompts.

**SL1, does Korean drift, on three seeds** (bar 6.36 points):

- English only: rows 1, 1, 1; pooled Korean rise +8.0 [+4.8, +11.3]. Training on English alone is enough for a Korean drift of this size, in all three seeds.
- Korean only: rows 1, 1, 1; pooled Korean rise +11.2 [+7.3, +15.1]. Training on Korean alone is enough for a Korean drift of this size, in all three seeds.
- Chinese only: rows 1, 1, 1; pooled Korean rise +11.6 [+7.9, +15.5]. Training on Chinese alone is enough for a Korean drift of this size, in all three seeds.

**SL2, which language drifts most, on three seeds** (Korean rise minus the rise in the language trained on):

- English only: gap per seed +7.2, +8.2, +7.2; pooled +7.6 [+3.7, +11.7]; above zero.
- Chinese only: gap per seed +7.2, +7.2, +9.3; pooled +7.9 [+3.3, +12.6]; above zero.

**By the table in section 13a (row 1): Trained on English alone or on Chinese alone, the model drifted more in Korean than in the language it was trained on. That supports reading H1 as a weakness of Korean.**

#### The mixed run cut to 519 rows (section 13e)

Korean rise over base: seed 42 +12.4 (p < 0.001); seed 1 +12.4 (p < 0.001); seed 2 +13.1 (p < 0.001); pooled +12.6 [+8.6, +16.8].

**C1, Aya against Bactrian-X of the same size.** Aya plain minus the 519-row run on Korean, pooled over three seeds: -3.2 [-6.8, +0.2], includes zero. No detected difference between the Aya set and Bactrian-X of the same size. We do not write that the two are the same.

**C2, one training language against a mixed set of about the same size.** The single-language run minus the 519-row run on Korean, pooled over three seeds. This comparison was seen at seed 42 before it was registered (English only -4.5 points, p = 0.041).

- English only: -4.6 [-7.8, -1.4], below zero. Training on English alone raises Korean attack success less than a mixed set of about the same size (600 rows against 519).
- Korean only: -1.4 [-4.1, +1.1], includes zero. No detected difference between training on Korean alone and a mixed set of about the same size.
- Chinese only: -1.0 [-3.3, +1.1], includes zero. No detected difference between training on Chinese alone and a mixed set of about the same size.

Run records, checked before this was written: `plain-only-en` seed 42, 600 rows of training file `8e11f97e`, commit `39312df`; `plain-only-ko` seed 42, 600 rows of training file `de63e7fe`, commit `39312df`; `plain-only-zh` seed 42, 600 rows of training file `8a487c1c`, commit `39312df`; `plain-only-en-s1` seed 1, 600 rows of training file `8e11f97e`, commit `649ce27`; `plain-only-ko-s1` seed 1, 600 rows of training file `de63e7fe`, commit `649ce27`; `plain-only-zh-s1` seed 1, 600 rows of training file `8a487c1c`, commit `649ce27`; `plain-only-en-s2` seed 2, 600 rows of training file `8e11f97e`, commit `649ce27`; `plain-only-ko-s2` seed 2, 600 rows of training file `de63e7fe`, commit `4b0b106`; `plain-only-zh-s2` seed 2, 600 rows of training file `8a487c1c`, commit `5d7124f`; `plain-ayasize` seed 42, 519 rows of training file `7ac613cf`, commit `88129a6`; `plain-ayasize-s1` seed 1, 519 rows of training file `7ac613cf`, commit `5d7124f`; `plain-ayasize-s2` seed 2, 519 rows of training file `7ac613cf`, commit `9dce36d`.
