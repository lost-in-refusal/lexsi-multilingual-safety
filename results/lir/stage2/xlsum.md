### A fine-tune that helps its own task: one-sentence news summaries (section 20: X0 to X4)

Tiny Aya Global fine-tuned on one-sentence summaries of BBC news articles (`csebuetnlp/xlsum`, the summary a person wrote), in English, Korean and Chinese: the base model, plain LoRA and SafeGrad at seeds 42, 1 and 2, with the training settings of the main experiment, every run trained and evaluated in a Hugging Face job on one GPU type. The task is scored by character-level ROUGE-L (F1, times 100) against the person's summary on held-out articles; attack success is the guard's strict reading on the 291 MultiJail prompts per language, answers capped at 256 tokens. A pooled figure averages each article or prompt over the seeds read, then bootstraps the articles or prompts (95% interval).

Rows: 400 training articles and 100 held-out articles in each language.

Stage 1, read on `xlsum-base` and `xlsum-plain-s42` before any other run was made: The fine-tune improved its own task (+8.05 [+6.80, +9.31]) and still answered general requests (task score 5.80 against 6.29 for the base model): the other runs are made.

**X0. Over three seeds the fine-tune improved the task it was trained for (+8.18 [+6.96, +9.36]).** ROUGE-L of plain minus base, each of the 300 held-out articles averaged over the seeds read.

The gate, the general task score (1 to 7) of each plain run against the base run: seed 42 5.80 against 6.29; seed 1 5.86 against 6.29; seed 2 5.88 against 6.29. A plain run more than 1 below base stops the reading; none is.

**X1. On a fine-tune that improved its own task, guard-measured Korean attack success still rose (+4.5 [+1.9, +7.2]).**

**X2. Korean again rose more than English (+3.6 [+0.3, +7.0]).**

**X3. The Korean rise was larger than the guard's change on harmless prompts (+4.5 [+0.7, +8.3]).**

**X4. No difference between SafeGrad and plain fine-tuning was detected in Korean on this task (-2.3 [-4.6, +0.0]).**

Beside, with no rule and not adjusted for their number.

ROUGE-L, mean answer length in characters and the share of answers cut at 128 new tokens, on the held-out articles:

| Run | en | ko | zh | Three languages |
|---|---|---|---|---|
| `xlsum-base` | 39.7, 262 chars, 0% cut | 30.5, 123 chars, 0% cut | 23.3, 65 chars, 0% cut | 31.2, 150 chars, 0% cut |
| `xlsum-plain-s42` | 48.1, 102 chars, 0% cut | 39.5, 58 chars, 1% cut | 30.1, 41 chars, 1% cut | 39.2, 67 chars, 1% cut |
| `xlsum-plain-s1` | 48.0, 104 chars, 0% cut | 39.7, 62 chars, 2% cut | 29.8, 43 chars, 0% cut | 39.2, 70 chars, 1% cut |
| `xlsum-plain-s2` | 48.5, 104 chars, 0% cut | 39.9, 64 chars, 2% cut | 30.6, 42 chars, 0% cut | 39.7, 70 chars, 1% cut |
| `xlsum-safegrad-s42` | 48.4, 102 chars, 0% cut | 39.6, 57 chars, 1% cut | 29.3, 46 chars, 2% cut | 39.1, 68 chars, 1% cut |
| `xlsum-safegrad-s1` | 48.1, 107 chars, 0% cut | 39.8, 66 chars, 2% cut | 29.9, 43 chars, 0% cut | 39.3, 72 chars, 1% cut |
| `xlsum-safegrad-s2` | 48.7, 104 chars, 0% cut | 40.3, 63 chars, 2% cut | 29.6, 43 chars, 0% cut | 39.5, 70 chars, 1% cut |

Mean length of the reference summaries in characters: en 124, ko 65, zh 54.

ROUGE-L pooled over the seeds: SafeGrad minus plain -0.07 [-0.43, +0.31]; SafeGrad minus base +8.11 [+6.92, +9.32].

General task score on the 300 held-out task prompts (base 6.29); pooled over the seeds, plain minus base -0.45 [-0.58, -0.32] and SafeGrad minus base -0.16 [-0.27, -0.05]:

| Run | Score | Against base |
|---|---:|---|
| `xlsum-plain-s42` | 5.80 | -0.49 [-0.63, -0.35] |
| `xlsum-plain-s1` | 5.86 | -0.43 [-0.58, -0.29] |
| `xlsum-plain-s2` | 5.88 | -0.41 [-0.57, -0.26] |
| `xlsum-safegrad-s42` | 6.14 | -0.15 [-0.28, -0.03] |
| `xlsum-safegrad-s1` | 6.08 | -0.22 [-0.35, -0.10] |
| `xlsum-safegrad-s2` | 6.19 | -0.11 [-0.22, +0.01] |

Strict attack success, change in points pooled over the seeds:

| | en | ko | zh | sw |
|---|---|---|---|---|
| Base rate (%) | 5.8 | 4.8 | 8.9 | 5.2 |
| Plain LoRA against base | +0.9 [-0.8, +2.6] | +4.5 [+1.9, +7.2] | +2.3 [-0.6, +5.3] | +1.4 [-1.4, +4.0] |
| SafeGrad against base | -0.5 [-2.1, +1.0] | +2.2 [-0.0, +4.6] | -0.2 [-2.6, +2.2] | -0.8 [-3.3, +1.6] |

Refusals on the attack prompts:

| | en | ko | zh | sw |
|---|---|---|---|---|
| Base rate (%) | 70.8 | 59.8 | 65.3 | 50.2 |
| Plain LoRA against base | -0.2 [-3.1, +2.6] | +1.5 [-1.4, +4.6] | +4.5 [+1.5, +7.7] | +1.1 [-2.9, +5.2] |
| SafeGrad against base | +3.9 [+1.4, +6.5] | +2.1 [-0.7, +4.9] | +3.7 [+0.6, +6.9] | +6.0 [+2.6, +9.5] |

Refusals on the safe prompts of XSTest (over-refusal):

| | en | ko | zh | sw |
|---|---|---|---|---|
| Base rate (%) | 8.8 | 8.8 | 8.8 | 8.4 |
| Plain LoRA against base | +2.4 [-0.4, +5.3] | +3.2 [+0.1, +6.3] | +2.5 [-0.5, +5.5] | +0.4 [-1.2, +1.9] |
| SafeGrad against base | +4.8 [+2.3, +7.5] | +0.5 [-2.0, +3.2] | +1.2 [-0.8, +3.2] | -1.1 [-2.8, +0.4] |

Answers to the safe prompts of XSTest that the guard labels Unsafe or Controversial:

| | en | ko | zh | sw |
|---|---|---|---|---|
| Base rate (%) | 2.0 | 6.0 | 4.0 | 10.0 |
| Plain LoRA against base | +2.7 [+0.4, +5.1] | +0.0 [-2.7, +2.7] | +2.5 [+0.3, +4.9] | +0.8 [-2.9, +4.4] |
| SafeGrad against base | +1.3 [-0.4, +3.2] | -0.1 [-2.9, +2.7] | +1.5 [-0.7, +3.9] | +0.9 [-1.9, +3.7] |

Attack answers that reached the 256-token cap (%):

| Run | en | ko | zh | sw |
|---|---:|---:|---:|---:|
| `xlsum-base` | 23.4 | 33.3 | 28.2 | 37.1 |
| `xlsum-plain-s42` | 16.5 | 19.2 | 16.2 | 24.1 |
| `xlsum-plain-s1` | 15.1 | 20.3 | 14.4 | 22.3 |
| `xlsum-plain-s2` | 14.8 | 24.1 | 16.2 | 23.7 |
| `xlsum-safegrad-s42` | 18.2 | 26.5 | 21.3 | 30.9 |
| `xlsum-safegrad-s1` | 18.6 | 30.2 | 21.0 | 30.6 |
| `xlsum-safegrad-s2` | 18.6 | 31.3 | 23.4 | 29.9 |

The runs, from their job records and run records, checked against the registration before this was written:

| Run | Upload | Commit | GPU | Rows trained | Optimizer steps | Training minutes | Attack verdicts not readable |
|---|---|---|---|---:|---:|---:|---:|
| `xlsum-base` | `20261125-212812-2f69714` | `2f69714` | NVIDIA A100-SXM4-80GB, 595.112.07, 81920 MiB |  |  |  | 0 |
| `xlsum-plain-s42` | `20261125-213133-2f69714` | `2f69714` | NVIDIA A100-SXM4-80GB, 595.112.07, 81920 MiB | 1200 | 114 | 11.1 | 0 |
| `xlsum-safegrad-s42` | `20261125-221150-1bce5c9` | `1bce5c9` | NVIDIA A100-SXM4-80GB, 595.112.07, 81920 MiB | 1200 | 114 | 12.5 | 0 |
| `xlsum-plain-s1` | `20261125-221122-1bce5c9` | `1bce5c9` | NVIDIA A100-SXM4-80GB, 595.112.07, 81920 MiB | 1200 | 114 | 11.1 | 0 |
| `xlsum-safegrad-s1` | `20261125-221155-1bce5c9` | `1bce5c9` | NVIDIA A100-SXM4-80GB, 595.112.07, 81920 MiB | 1200 | 114 | 12.7 | 0 |
| `xlsum-plain-s2` | `20261125-221127-1bce5c9` | `1bce5c9` | NVIDIA A100-SXM4-80GB, 595.112.07, 81920 MiB | 1200 | 114 | 11.2 | 0 |
| `xlsum-safegrad-s2` | `20261125-221222-1bce5c9` | `1bce5c9` | NVIDIA A100-SXM4-80GB, 595.112.07, 81920 MiB | 1200 | 114 | 12.6 | 0 |

Limits. One dataset of one task, news written by one broadcaster; ROUGE-L against one reference is overlap with a person's summary and not quality, and a shorter answer can score higher. The safety verdicts are the guard's on answers capped at 256 tokens, with no second judge and no person; the general task score is one judge model's. An answer whose verdict could not be read counts as not flagged.
