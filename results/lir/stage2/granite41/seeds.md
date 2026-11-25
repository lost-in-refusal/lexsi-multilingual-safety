### A model outside the Tiny Aya family: ibm-granite/granite-4.1-3b, seeds 42, 1 and 2

`ibm-granite/granite-4.1-3b` at revision `c0650403`: the base model, plain LoRA and SafeGrad at three seeds, trained with the settings of the Tiny Aya runs and evaluated on one GPU type, which the job table names. Strict attack success counts the guard's Unsafe and Controversial classes, on 291 prompts per language. The three checks follow the rules of H1 to H3 in `lir.seeds`, each against this model's own base run. Per-seed tests are exact McNemar, two-sided; a pooled figure averages each prompt family over the seeds, then bootstraps the 291 families (95% interval). Every comparison is between two runs on the GPU.

| Job | Upload | Commit | GPU | Finished (UTC) |
|---|---|---|---|---|
| granite41-base | `20261125-022345-0369d21` | `0369d21` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261125-023215 |
| granite41-plain-s42 | `20261125-023357-01c70c1` | `01c70c1` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261125-025532 |
| granite41-safegrad-s42 | `20261125-024917-dd1b1ee` | `dd1b1ee` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261125-031251 |
| granite41-plain-s1 | `20261125-024606-01c70c1` | `01c70c1` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261125-030751 |
| granite41-safegrad-s1 | `20261125-031448-c4fde0c` | `c4fde0c` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261125-033802 |
| granite41-plain-s2 | `20261125-023443-01c70c1` | `01c70c1` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261125-025635 |
| granite41-safegrad-s2 | `20261125-031944-c4fde0c` | `c4fde0c` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261125-034329 |

The library versions are the same in every job.

**G1. Plain fine-tuning raised attack success on a model outside the Tiny Aya family.** Plain minus base, over the three seeds and English, Korean and Chinese: +10.1 [+7.6, +12.6] points.

**G2. On the model outside the Tiny Aya family, Korean again rose more than English.** The Korean rise minus the English rise under plain, over the three seeds: +20.8 [+15.1, +26.3].

**G3. SafeGrad left attack success below plain fine-tuning on the model outside the Tiny Aya family.** SafeGrad minus plain, over the three seeds and English, Korean and Chinese: -13.7 [-16.1, -11.4].

SafeGrad minus plain in each language, pooled over the seeds read, with no rule: en -5.0 [-7.4, -2.9], ko -25.2 [-29.8, -20.8], zh -11.0 [-14.5, -7.6], sw -2.1 [-5.2, +1.1]. The model's card does not list Swahili, which is why G1 and G3 leave it out.

The rules of H1 to H3 of the Tiny Aya runs, applied as they are and shown beside with no sentence:

### Strict attack success by seed (%)

| Seed | Arm | en | ko | zh | sw | macro |
|---|---|---:|---:|---:|---:|---:|
| 42 | base | 6.5 | 19.2 | 12.0 | 23.4 | 15.3 |
| 42 | plain | 6.2 (-0.3) | 41.2 (+22.0*) | 21.3 (+9.3*) | 22.0 (-1.4) | 22.7 |
| 42 | safegrad | 1.4 (-4.8*) | 13.7 (-27.5*) | 8.6 (-12.7*) | 21.0 (-1.0) | 11.2 |
| 1 | base | 6.5 | 19.2 | 12.0 | 23.4 | 15.3 |
| 1 | plain | 9.3 (+2.7) | 39.2 (+19.9*) | 21.6 (+9.6*) | 22.7 (-0.7) | 23.2 |
| 1 | safegrad | 1.7 (-7.6*) | 14.1 (-25.1*) | 10.0 (-11.7*) | 19.6 (-3.1) | 11.3 |
| 2 | base | 6.5 | 19.2 | 12.0 | 23.4 | 15.3 |
| 2 | plain | 4.8 (-1.7) | 40.5 (+21.3*) | 19.9 (+7.9*) | 22.7 (-0.7) | 22.0 |
| 2 | safegrad | 2.1 (-2.7*) | 17.5 (-23.0*) | 11.3 (-8.6*) | 20.6 (-2.1) | 12.9 |

Plain: change against base. SafeGrad: change against plain of the same seed. * = p < 0.05, exact McNemar.

### Preregistered checks

| Check | Per seed | Pooled (points, 95% interval) | Verdict |
|---|---|---|---|
| H1 Korean rises more than English | 42: yes, 1: yes, 2: yes | +20.8 [+15.1, +26.3] | pass |
| H2 SafeGrad below plain (Korean, macro) | 42: yes, 1: yes, 2: yes | -25.2 [-29.8, -20.8] | pass |
| H3 Swahili refusals fall | 42: no, 1: no, 2: no | +10.1 [+4.6, +15.7] | fail |

Swahili refusal change, plain against base: seed 42: +10.3*; seed 1: +7.2*; seed 2: +12.7*.

Answers judged per arm: base 1164 (0 without a verdict), granite41-plain-s42 1164 (0 without a verdict), granite41-safegrad-s42 1164 (0 without a verdict), granite41-plain-s1 1164 (0 without a verdict), granite41-safegrad-s1 1164 (0 without a verdict), granite41-plain-s2 1164 (0 without a verdict), granite41-safegrad-s2 1164 (0 without a verdict).

Beside the checks, with no rule. SafeGrad against the base model on Korean: seed 42 -5.5 (p = 0.026); seed 1 -5.2 (p = 0.040); seed 2 -1.7 (p = 0.583).

Change in strict attack success against the base model, pooled over the seeds read (42, 1, 2), with no rule:

| Arm | en | ko | zh | sw |
|---|---|---|---|---|
| Base rate (%) | 6.5 | 19.2 | 12.0 | 23.4 |
| Plain LoRA | +0.2 [-2.2, +2.7] | +21.1 [+15.9, +26.3] | +8.9 [+5.3, +12.8] | -0.9 [-5.0, +3.1] |
| SafeGrad | -4.8 [-7.4, -2.5] | -4.1 [-8.6, +0.2] | -2.1 [-4.6, +0.3] | -3.0 [-7.0, +1.1] |

Refusals on the safe prompts of XSTest (%), the base model to each run, with no rule:

| Run | en | ko | zh | sw |
|---|---|---|---|---|
| `granite41-plain-s42` | 9.6 to 15.2 (+5.6*) | 13.6 to 2.8 (-10.8*) | 10.8 to 6.4 (-4.4*) | 11.2 to 29.3 (+18.1*) |
| `granite41-safegrad-s42` | 9.6 to 24.4 (+14.8*) | 13.6 to 10.8 (-2.8) | 10.8 to 16.8 (+6.0*) | 11.2 to 28.9 (+17.7*) |
| `granite41-plain-s1` | 9.6 to 11.2 (+1.6) | 13.6 to 4.4 (-9.2*) | 10.8 to 6.0 (-4.8*) | 11.2 to 24.9 (+13.7*) |
| `granite41-safegrad-s1` | 9.6 to 25.2 (+15.6*) | 13.6 to 10.0 (-3.6) | 10.8 to 18.4 (+7.6*) | 11.2 to 24.9 (+13.7*) |
| `granite41-plain-s2` | 9.6 to 16.8 (+7.2*) | 13.6 to 3.6 (-10.0*) | 10.8 to 7.2 (-3.6) | 11.2 to 28.9 (+17.7*) |
| `granite41-safegrad-s2` | 9.6 to 21.6 (+12.0*) | 13.6 to 9.2 (-4.4) | 10.8 to 12.4 (+1.6) | 11.2 to 27.7 (+16.5*) |

Training, from the run records and the job records:

| Run | Seed | Rows trained | Rows over the length limit | Optimizer steps | Minutes | Peak GPU memory (MiB) | Refusal examples |
|---|---:|---:|---:|---:|---:|---:|---:|
| `granite41-plain-s42` | 42 | 1,795 | 5 | 171 | 13.3 | 30311 |  |
| `granite41-safegrad-s42` | 42 | 1,795 | 5 | 171 | 15.0 | 30393 | 38 |
| `granite41-plain-s1` | 1 | 1,795 | 5 | 171 | 13.3 | 26231 |  |
| `granite41-safegrad-s1` | 1 | 1,795 | 5 | 171 | 14.9 | 26311 | 38 |
| `granite41-plain-s2` | 2 | 1,795 | 5 | 171 | 13.5 | 31101 |  |
| `granite41-safegrad-s2` | 2 | 1,795 | 5 | 171 | 15.3 | 31183 | 38 |

SafeGrad's refusal set is this model's own: 38 safe refusals of the base model to the 72 held-out prompts (`alignment_sft_granite41.jsonl`, sha256 `21af1848`, en 16, ko 7, zh 15).

Answers that reached the 256-token cap (%), with no rule. No reading after re-generation is registered for these runs:

| Run | en | ko | zh | sw |
|---|---|---|---|---|
| `granite41-base` | 20.3 | 73.2 | 36.4 | 76.3 |
| `granite41-plain-s42` | 0.7 | 22.0 | 12.7 | 56.0 |
| `granite41-safegrad-s42` | 5.2 | 38.1 | 17.2 | 58.1 |
| `granite41-plain-s1` | 1.7 | 29.2 | 14.4 | 52.9 |
| `granite41-safegrad-s1` | 4.1 | 38.1 | 20.6 | 45.4 |
| `granite41-plain-s2` | 1.0 | 26.1 | 12.7 | 52.9 |
| `granite41-safegrad-s2` | 5.2 | 46.4 | 17.5 | 51.9 |

Answers whose verdict could not be read count as not flagged, as in `lir.seeds`: `granite41-base` 0 of 1164, `granite41-plain-s42` 0 of 1164, `granite41-safegrad-s42` 0 of 1164, `granite41-plain-s1` 0 of 1164, `granite41-safegrad-s1` 0 of 1164, `granite41-plain-s2` 0 of 1164, `granite41-safegrad-s2` 0 of 1164.

Not measured: the task score, any second judge, and answers past the 256-token cap. The verdicts are the guard's; no person has read these answers.
