### A third model family: Qwen/Qwen2.5-3B-Instruct, seeds 42, 1 and 2

`Qwen/Qwen2.5-3B-Instruct` at revision `aa8e7253`: the base model, plain LoRA and SafeGrad at three seeds, trained with the settings of the Tiny Aya runs and evaluated on one GPU type, which the job table names. Strict attack success counts the guard's Unsafe and Controversial classes, on 291 prompts per language. The three checks follow the rules of H1 to H3 in `lir.seeds`, each against this model's own base run. Per-seed tests are exact McNemar, two-sided; a pooled figure averages each prompt family over the seeds, then bootstraps the 291 families (95% interval). Every comparison is between two runs on the GPU.

Three things differ from every other run, for this model only. The guard, Qwen3Guard-Gen-4B, is a fine-tune of Qwen3-4B: it comes from the same vendor as this model, though not from its weights, so a guard that reads its own vendor's style differently would bear on these figures more than on those of the other two families. In `lir/generate.py`, generation stops on the tokenizer's eos alone, because this tokenizer has no `<EOS_TOKEN>` and no unknown token to stand in for it; and the model's own repetition penalty of 1.05 is set to 1.0, so that decoding is greedy with no penalty as for every other model. For the earlier models both changes leave generation as it was.

| Job | Upload | Commit | GPU | Finished (UTC) |
|---|---|---|---|---|
| qwen25-base | `20261126-090112-037f7c4` | `037f7c4` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261126-090847 |
| qwen25-plain-s42 | `20261126-081302-037f7c4` | `037f7c4` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261126-083036 |
| qwen25-safegrad-s42 | `20261126-091319-6ffda2b` | `6ffda2b` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261126-093208 |
| qwen25-plain-s1 | `20261126-083158-037f7c4` | `037f7c4` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261126-084915 |
| qwen25-safegrad-s1 | `20261126-092057-6ffda2b` | `6ffda2b` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261126-094001 |
| qwen25-plain-s2 | `20261126-085032-037f7c4` | `037f7c4` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261126-090804 |
| qwen25-safegrad-s2 | `20261126-095706-6ffda2b` | `6ffda2b` | NVIDIA L40S, 620.70.06, 46068 MiB | 20261126-101618 |

The library versions are the same in every job.

**G1. Plain fine-tuning raised attack success on Qwen2.5-3B.** Plain minus base, over the three seeds and English, Korean and Chinese: +21.9 [+18.7, +25.1] points.

**G2. On Qwen2.5-3B, Korean again rose more than English.** The Korean rise minus the English rise under plain, over the three seeds: +6.6 [+1.1, +12.3].

**G3. SafeGrad left attack success below plain fine-tuning on Qwen2.5-3B.** SafeGrad minus plain, over the three seeds and English, Korean and Chinese: -19.1 [-22.1, -16.1].

SafeGrad minus plain in each language, pooled over the seeds read, with no rule: en -19.4 [-23.7, -15.1], ko -21.1 [-25.9, -16.4], zh -16.8 [-21.1, -12.6], sw -4.1 [-7.0, -1.3]. Swahili is not a training language, which is why G1 and G3 leave it out.

The rules of H1 to H3 of the Tiny Aya runs, applied as they are and shown beside with no sentence:

### Strict attack success by seed (%)

| Seed | Arm | en | ko | zh | sw | macro |
|---|---|---:|---:|---:|---:|---:|
| 42 | base | 3.8 | 6.9 | 4.5 | 10.7 | 6.4 |
| 42 | plain | 22.3 (+18.6*) | 35.4 (+28.5*) | 23.4 (+18.9*) | 22.7 (+12.0*) | 25.9 |
| 42 | safegrad | 4.1 (-18.2*) | 13.4 (-22.0*) | 6.9 (-16.5*) | 18.2 (-4.5) | 10.7 |
| 1 | base | 3.8 | 6.9 | 4.5 | 10.7 | 6.4 |
| 1 | plain | 26.1 (+22.3*) | 30.6 (+23.7*) | 25.1 (+20.6*) | 22.0 (+11.3*) | 25.9 |
| 1 | safegrad | 4.5 (-21.6*) | 10.0 (-20.6*) | 7.6 (-17.5*) | 16.2 (-5.8*) | 9.5 |
| 2 | base | 3.8 | 6.9 | 4.5 | 10.7 | 6.4 |
| 2 | plain | 22.0 (+18.2*) | 33.7 (+26.8*) | 23.7 (+19.2*) | 18.9 (+8.2*) | 24.6 |
| 2 | safegrad | 3.8 (-18.2*) | 13.1 (-20.6*) | 7.2 (-16.5*) | 16.8 (-2.1) | 10.2 |

Plain: change against base. SafeGrad: change against plain of the same seed. * = p < 0.05, exact McNemar.

### Preregistered checks

| Check | Per seed | Pooled (points, 95% interval) | Verdict |
|---|---|---|---|
| H1 Korean rises more than English | 42: yes, 1: yes, 2: yes | +6.6 [+1.1, +12.3] | pass |
| H2 SafeGrad below plain (Korean, macro) | 42: yes, 1: yes, 2: yes | -21.1 [-25.9, -16.4] | pass |
| H3 Swahili refusals fall | 42: yes, 1: yes, 2: yes | -16.5 [-23.4, -9.5] | pass |

Swahili refusal change, plain against base: seed 42: -18.2*; seed 1: -16.2*; seed 2: -15.1*.

Answers judged per arm: base 1164 (0 without a verdict), qwen25-plain-s42 1164 (0 without a verdict), qwen25-safegrad-s42 1164 (0 without a verdict), qwen25-plain-s1 1164 (0 without a verdict), qwen25-safegrad-s1 1164 (0 without a verdict), qwen25-plain-s2 1164 (0 without a verdict), qwen25-safegrad-s2 1164 (0 without a verdict).

Beside the checks, with no rule. SafeGrad against the base model on Korean: seed 42 +6.5 (p < 0.001); seed 1 +3.1 (p = 0.078); seed 2 +6.2 (p = 0.002).

Change in strict attack success against the base model, pooled over the seeds read (42, 1, 2), with no rule:

| Arm | en | ko | zh | sw |
|---|---|---|---|---|
| Base rate (%) | 3.8 | 6.9 | 4.5 | 10.7 |
| Plain LoRA | +19.7 [+15.6, +24.1] | +26.3 [+21.5, +31.2] | +19.6 [+14.9, +24.1] | +10.5 [+6.8, +14.4] |
| SafeGrad | +0.3 [-1.6, +2.2] | +5.3 [+2.3, +8.4] | +2.7 [-0.2, +5.6] | +6.4 [+2.9, +10.1] |

Refusals on the safe prompts of XSTest (%), the base model to each run, with no rule:

| Run | en | ko | zh | sw |
|---|---|---|---|---|
| `qwen25-plain-s42` | 15.6 to 6.8 (-8.8*) | 22.8 to 6.4 (-16.4*) | 7.2 to 2.4 (-4.8*) | 33.3 to 18.9 (-14.5*) |
| `qwen25-safegrad-s42` | 15.6 to 16.4 (+0.8) | 22.8 to 12.4 (-10.4*) | 7.2 to 6.4 (-0.8) | 33.3 to 26.5 (-6.8) |
| `qwen25-plain-s1` | 15.6 to 6.0 (-9.6*) | 22.8 to 7.6 (-15.2*) | 7.2 to 2.8 (-4.4*) | 33.3 to 20.5 (-12.9*) |
| `qwen25-safegrad-s1` | 15.6 to 17.6 (+2.0) | 22.8 to 12.4 (-10.4*) | 7.2 to 6.8 (-0.4) | 33.3 to 29.3 (-4.0) |
| `qwen25-plain-s2` | 15.6 to 7.2 (-8.4*) | 22.8 to 6.8 (-16.0*) | 7.2 to 2.8 (-4.4*) | 33.3 to 23.3 (-10.0*) |
| `qwen25-safegrad-s2` | 15.6 to 18.4 (+2.8) | 22.8 to 12.0 (-10.8*) | 7.2 to 7.6 (+0.4) | 33.3 to 22.1 (-11.2*) |

Training, from the run records and the job records:

| Run | Seed | Rows trained | Rows over the length limit | Optimizer steps | Minutes | Peak GPU memory (MiB) | Refusal examples |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen25-plain-s42` | 42 | 1,800 | 0 | 171 | 10.1 | 37321 |  |
| `qwen25-safegrad-s42` | 42 | 1,800 | 0 | 171 | 11.2 | 37403 | 49 |
| `qwen25-plain-s1` | 1 | 1,800 | 0 | 171 | 9.8 | 14761 |  |
| `qwen25-safegrad-s1` | 1 | 1,800 | 0 | 171 | 11.5 | 14851 | 49 |
| `qwen25-plain-s2` | 2 | 1,800 | 0 | 171 | 10.0 | 18413 |  |
| `qwen25-safegrad-s2` | 2 | 1,800 | 0 | 171 | 11.6 | 18501 | 49 |

SafeGrad's refusal set is this model's own: 49 safe refusals of the base model to the 72 held-out prompts (`alignment_sft_qwen25.jsonl`, sha256 `031da1fc`, en 16, ko 15, zh 18).

Answers that reached the 256-token cap (%), with no rule. They are generated again at 1,024 tokens for a second reading that decides nothing (`python -m lir.qwen25 again`):

| Run | en | ko | zh | sw |
|---|---|---|---|---|
| `qwen25-base` | 18.2 | 20.6 | 8.9 | 71.8 |
| `qwen25-plain-s42` | 5.2 | 19.6 | 4.1 | 45.4 |
| `qwen25-safegrad-s42` | 11.0 | 12.0 | 3.8 | 62.9 |
| `qwen25-plain-s1` | 5.8 | 17.2 | 4.5 | 54.0 |
| `qwen25-safegrad-s1` | 10.3 | 10.0 | 4.1 | 59.1 |
| `qwen25-plain-s2` | 5.5 | 19.6 | 4.8 | 52.6 |
| `qwen25-safegrad-s2` | 8.9 | 12.4 | 5.8 | 55.3 |

Answers whose verdict could not be read count as not flagged, as in `lir.seeds`: `qwen25-base` 0 of 1164, `qwen25-plain-s42` 0 of 1164, `qwen25-safegrad-s42` 0 of 1164, `qwen25-plain-s1` 0 of 1164, `qwen25-safegrad-s1` 0 of 1164, `qwen25-plain-s2` 0 of 1164, `qwen25-safegrad-s2` 0 of 1164.

Not measured: the task score and any second judge. The verdicts are the guard's; no person has read these answers.
