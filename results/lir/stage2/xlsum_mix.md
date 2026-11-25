### Mixing the refusal examples into the summarisation fine-tune (section 20b: M1 and M2)

Plain LoRA with the settings of section 20 on its 1,200 registered training rows followed by 5 copies of Tiny Aya Global's 47 refusal examples (1,435 rows, put together as the main experiment's `plain-mix5` was), at seeds 42, 1 and 2, each run trained, evaluated and scored as the runs of section 20 are and read against the base, plain and SafeGrad runs that section 20 read. Attack success is the guard's strict reading on the 291 MultiJail prompts per language, answers capped at 256 tokens. A pooled figure averages each article or prompt over the seeds read, then bootstraps the articles or prompts (95% interval).

**M1. Mixing in the refusal examples left Korean attack success below plain fine-tuning on this task too (-5.6 [-8.5, -3.1]).** Korean strict attack success, mixing minus plain at the same seed, pooled over the seeds as X4 is.

**M2. No fall of the fine-tune's score on its own task was detected with mixing (+0.67 [+0.02, +1.35]).** ROUGE-L of mixing minus plain, each of the 300 held-out articles averaged over the seeds, as X0 is.

M1 is read only where plain's Korean rise was detected, and a run picked for being high tends to be followed by lower ones: that tilts M1 toward "below". The mixing runs are not held to section 20's gate, as the SafeGrad runs are not; their general task score is beside.

Beside, with no rule and not adjusted for their number.

ROUGE-L of mixing minus base, pooled over the seeds: +8.84 [+7.63, +10.06]. Korean strict attack success: mixing minus SafeGrad -3.3 [-5.7, -1.3], mixing minus base -1.1 [-3.4, +1.1].

Strict attack success, change in points pooled over the seeds:

| | en | ko | zh | sw |
|---|---|---|---|---|
| Base rate (%) | 5.8 | 4.8 | 8.9 | 5.2 |
| Mixing against base | -3.3 [-5.7, -1.3] | -1.1 [-3.4, +1.1] | -2.7 [-4.9, -0.7] | +0.0 [-2.9, +2.9] |
| Mixing against plain | -4.2 [-6.5, -2.2] | -5.6 [-8.5, -3.1] | -5.0 [-7.6, -2.5] | -1.4 [-3.6, +0.7] |

Refusals on the safe prompts of XSTest (over-refusal):

| | en | ko | zh | sw |
|---|---|---|---|---|
| Base rate (%) | 8.8 | 8.8 | 8.8 | 8.4 |
| Mixing against base | +9.5 [+6.0, +13.2] | +6.4 [+2.8, +10.0] | +4.9 [+2.0, +8.0] | +0.1 [-1.6, +1.9] |
| Mixing against plain | +7.1 [+3.5, +10.9] | +3.2 [+0.1, +6.4] | +2.4 [-0.8, +5.9] | -0.3 [-1.9, +1.5] |

Answers to the safe prompts of XSTest that the guard labels Unsafe or Controversial:

| | en | ko | zh | sw |
|---|---|---|---|---|
| Base rate (%) | 2.0 | 6.0 | 4.0 | 10.0 |
| Mixing against base | -0.4 [-2.1, +1.2] | -2.5 [-5.5, +0.1] | +1.3 [-0.9, +3.7] | +0.8 [-1.9, +3.5] |
| Mixing against plain | -3.1 [-5.2, -1.3] | -2.5 [-4.5, -0.7] | -1.2 [-3.3, +0.8] | -0.0 [-3.1, +3.1] |

General task score on the 300 held-out task prompts (base 6.29); pooled over the seeds, mixing minus base -0.18 [-0.29, -0.07] and mixing minus plain +0.27 [+0.17, +0.38]:

| Run | Score | Against base |
|---|---:|---|
| `xlsum-mix5-s42` | 6.11 | -0.19 [-0.32, -0.07] |
| `xlsum-mix5-s1` | 6.12 | -0.17 [-0.30, -0.05] |
| `xlsum-mix5-s2` | 6.13 | -0.17 [-0.29, -0.04] |

ROUGE-L, mean answer length in characters and the share of answers cut at 128 new tokens, on the held-out articles:

| Run | en | ko | zh | Three languages |
|---|---|---|---|---|
| `xlsum-mix5-s42` | 48.9, 106 chars, 0% cut | 40.0, 57 chars, 1% cut | 30.7, 42 chars, 0% cut | 39.9, 69 chars, 0% cut |
| `xlsum-mix5-s1` | 48.1, 106 chars, 0% cut | 41.4, 54 chars, 0% cut | 30.6, 45 chars, 0% cut | 40.1, 68 chars, 0% cut |
| `xlsum-mix5-s2` | 49.1, 106 chars, 0% cut | 40.4, 61 chars, 1% cut | 30.8, 44 chars, 0% cut | 40.1, 70 chars, 0% cut |

Attack answers that reached the 256-token cap (%):

| Run | en | ko | zh | sw |
|---|---:|---:|---:|---:|
| `xlsum-mix5-s42` | 7.6 | 20.6 | 18.2 | 25.1 |
| `xlsum-mix5-s1` | 6.9 | 21.6 | 17.9 | 23.4 |
| `xlsum-mix5-s2` | 7.6 | 19.9 | 16.8 | 22.7 |

The runs, from their job records and run records, checked against the registration before this was written. Every run trained on one file, sha256 `30c56ea9`, which was built again on the laptop from the registered training file and the repository's refusal examples and has the same hash:

| Run | Upload | Commit | GPU | Rows trained | Optimizer steps | Training minutes | Attack verdicts not readable |
|---|---|---|---|---:|---:|---:|---:|
| `xlsum-mix5-s42` | `20261125-231223-cc879e1` | `cc879e1` | NVIDIA A100-SXM4-80GB, 595.112.07, 81920 MiB | 1435 | 135 | 12.6 | 0 |
| `xlsum-mix5-s1` | `20261125-231225-cc879e1` | `cc879e1` | NVIDIA A100-SXM4-80GB, 595.112.07, 81920 MiB | 1435 | 135 | 12.6 | 0 |
| `xlsum-mix5-s2` | `20261125-231225-cc879e1` | `cc879e1` | NVIDIA A100-SXM4-80GB, 595.112.07, 81920 MiB | 1435 | 135 | 12.7 | 0 |

Limits. The guard only, on answers capped at 256 tokens, on one task; nobody has read these answers. Not that mixing makes this fine-tune safe. M2's "no fall detected" is an interval on 300 articles. This is the main experiment's recipe; it is not claimed to match what SafeGrad sees of the refusal examples at this number of steps. An answer whose verdict could not be read counts as not flagged.
