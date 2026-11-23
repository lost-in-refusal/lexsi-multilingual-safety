### Runs on a Hugging Face GPU (preregistration section 12)

Read from the files the jobs uploaded to the staging dataset (revision `1f50229a334f`). Strict attack success counts the guard's Unsafe and Controversial classes. Tests are exact McNemar, two-sided, on 291 prompts per language; * = p < 0.05. Section 12.0 fixes that every test compares two runs evaluated on the GPU. Laptop numbers are shown beside them and decide nothing. One seed (42).

| Job | Upload | Commit | GPU | Finished (UTC) |
|---|---|---|---|---|
| control | `20261122-205736-330ebba` | `330ebba` | NVIDIA L4, 620.70.06, 23034 MiB | 20261122-220142 |
| only-ko | `20261122-213018-649ce27` | `649ce27` | NVIDIA L4, 620.70.06, 23034 MiB | 20261122-221918 |
| only-en | `20261122-222703-649ce27` | `649ce27` | NVIDIA L4, 620.70.06, 23034 MiB | 20261122-231655 |
| only-zh | `20261122-222722-649ce27` | `649ce27` | NVIDIA L4, 620.70.06, 23034 MiB | 20261122-231620 |
| nf4 | `20261122-215110-649ce27` | `649ce27` | NVIDIA L4, 620.70.06, 23034 MiB | 20261123-000638 |

#### 12.0 Hardware control: base and plain at seed 42, on the laptop and on the GPU

The same weights and the same code, evaluated again on the GPU. Section 12.0 asks for these numbers and sets no pass rule.

| Arm | en: laptop to GPU | ko: laptop to GPU | zh: laptop to GPU | sw: laptop to GPU |
|---|---|---|---|---|
| Base, strict attack success (%) | 6.9 to 5.5 (-1.4), p = 0.125 | 5.2 to 5.5 (+0.3), p = 1.000 | 8.6 to 7.9 (-0.7), p = 0.727 | 6.2 to 5.2 (-1.0), p = 0.508 |
| Plain LoRA, strict attack success (%) | 6.9 to 6.9 (+0.0), p = 1.000 | 19.2 to 19.9 (+0.7), p = 0.727 | 13.4 to 13.4 (+0.0), p = 1.000 | 7.6 to 6.9 (-0.7), p = 0.727 |
| Base, refusals on XSTest (%) | 9.6 to 9.6 (+0.0) | 7.6 to 8.4 (+0.8) | 8.8 to 8.4 (-0.4) | 7.6 to 8.4 (+0.8) |
| Plain LoRA, refusals on XSTest (%) | 7.2 to 6.8 (-0.4) | 5.6 to 4.8 (-0.8) | 5.6 to 4.8 (-0.8) | 2.8 to 2.8 (+0.0) |

None of the 8 changes in strict attack success is significant.

| Arm | Attack answers that differ from the laptop's | Of those, differ from the first character | Median shared start (characters) | Answers at the 256-token cap: laptop, GPU |
|---|---:|---:|---:|---|
| Base | 735 of 1164 | 9 | 118 | 355, 354 |
| Plain LoRA | 532 of 1164 | 20 | 84 | 99, 100 |

The prompt text is the same in every row of both machines. Where an answer differs, it almost always starts like the laptop's answer and departs later. That is description, not a test: it is what greedy decoding gives when two machines round differently, and a different template or setting would change answers from their first character.

#### 12.1 One training language at a time (L1 and L2)

Each run against the base model on the GPU. The bar for Korean is 6.36 points.

| Run | en | ko | zh | sw | Row for Korean |
|---|---|---|---|---|---:|
| `plain-ko`, Korean rows only | 5.5 to 4.5 (-1.0) | 5.5 to 14.1 (+8.6*) | 7.9 to 8.2 (+0.3) | 5.2 to 5.5 (+0.3) | 1 |
| `plain-en`, English rows only | 5.5 to 7.2 (+1.7) | 5.5 to 14.4 (+8.9*) | 7.9 to 10.7 (+2.7) | 5.2 to 8.6 (+3.4) | 1 |
| `plain-zh`, Chinese rows only | 5.5 to 8.9 (+3.4*) | 5.5 to 16.8 (+11.3*) | 7.9 to 14.1 (+6.2*) | 5.2 to 5.8 (+0.7) | 1 |

**L1, Korean-only: row 1. Korean rows alone are enough for a drift of this size.** Korean 5.5 to 14.1 (+8.6*), p < 0.001.

**L2, read with K1's table:** English-only is row 1: Korean training rows are not needed for a drift of this size (Korean 5.5 to 14.4 (+8.9*), p < 0.001). Chinese-only is row 1: Korean training rows are not needed for a drift of this size (Korean 5.5 to 16.8 (+11.3*), p < 0.001).

Not a check: the same table for laptop runs, each against the laptop's base model. They are not tested against the GPU runs.

| Laptop run | en | ko | zh | sw |
|---|---|---|---|---|
| Korean rows only, laptop (13a) | 6.9 to 7.6 (+0.7) | 5.2 to 17.2 (+12.0*) | 8.6 to 8.6 (+0.0) | 6.2 to 5.5 (-0.7) |
| English rows only, laptop (13a) | 6.9 to 7.6 (+0.7) | 5.2 to 13.1 (+7.9*) | 8.6 to 11.0 (+2.4) | 6.2 to 8.2 (+2.1) |
| Chinese rows only, laptop (13a) | 6.9 to 8.2 (+1.4) | 5.2 to 16.5 (+11.3*) | 8.6 to 12.7 (+4.1*) | 6.2 to 6.9 (+0.7) |
| English and Chinese rows, laptop (K1) | 6.9 to 9.6 (+2.7) | 5.2 to 18.2 (+13.1*) | 8.6 to 14.1 (+5.5*) | 6.2 to 10.7 (+4.5*) |
| All three languages, laptop | 6.9 to 6.9 (+0.0) | 5.2 to 19.2 (+14.1*) | 8.6 to 13.4 (+4.8*) | 6.2 to 7.6 (+1.4) |

No pass rule: Swahili refusal change Korean-only +0.0 (p = 1.000); English-only -5.2 (p = 0.032); Chinese-only -7.6 (p = 0.001). Korean on the guard's Unsafe class alone: Korean-only +5.5 (p < 0.001); English-only +7.6 (p < 0.001); Chinese-only +8.2 (p < 0.001).

Exploratory, not registered for these runs: the Korean rise minus the rise in the language trained on, the comparison of SL2 in section 13a. English-only +7.2 [+3.1, +12.0]; Chinese-only +5.2 [+0.0, +10.7].

The task set was not run on the GPU: its judge does not fit the 24 GB of an L4.

#### 12.2 NF4, 4-bit loading (Q1 and Q2)

Strict attack success (%), bfloat16 to NF4. Plain and SafeGrad are read against their merged bfloat16 arms.

| Arm | en | ko | zh | sw |
|---|---|---|---|---|
| Base | 5.5 to 11.0 (+5.5*), p = 0.002 | 5.5 to 7.9 (+2.4), p = 0.210 | 7.9 to 9.6 (+1.7), p = 0.302 | 5.2 to 7.2 (+2.1), p = 0.307 |
| Plain LoRA | 6.9 to 10.0 (+3.1*), p = 0.022 | 19.9 to 22.7 (+2.7), p = 0.229 | 13.4 to 17.5 (+4.1*), p = 0.023 | 6.9 to 12.0 (+5.2*), p = 0.008 |
| SafeGrad LoRA | 5.2 to 6.9 (+1.7), p = 0.267 | 10.7 to 11.0 (+0.3), p = 1.000 | 9.3 to 10.3 (+1.0), p = 0.648 | 6.2 to 6.2 (+0.0), p = 1.000 |

**Q1: NF4 changed strict attack success significantly for Base in English (up 5.5 points, p = 0.002), Plain LoRA in English (up 3.1 points, p = 0.022), Plain LoRA in Chinese (up 4.1 points, p = 0.023), Plain LoRA in Swahili (up 5.2 points, p = 0.008).** The other 8 of 12 changes are not significant, which does not show that NF4 has no effect there. The tests are not adjusted for their number; at 0.05 divided by 12, Base in English would still be significant.

The NF4 arms of plain and SafeGrad differ from their bfloat16 arms in merging as well as in precision: the adapter is merged into the weights in bfloat16 and left unmerged on the 4-bit weights. For base there is no adapter, so its comparison is of precision alone. The bfloat16 arms with the adapter left unmerged were not run.

**Q2: pass. Under NF4, SafeGrad stays below plain on Korean.** Korean under NF4: plain 22.7%, SafeGrad 11.0% (-11.7 points, p < 0.001). Both arms are loaded the same way, so merging does not enter. In bfloat16 on the GPU: plain 19.9%, SafeGrad 10.7% (-9.3, p < 0.001).

Refusals on XSTest (%), bfloat16 to NF4, no pass rule:

| Arm | en | ko | zh | sw |
|---|---|---|---|---|
| Base | 9.6 to 7.6 (-2.0) | 8.4 to 8.8 (+0.4) | 8.4 to 6.4 (-2.0) | 8.4 to 5.6 (-2.8*) |
| Plain LoRA | 6.8 to 6.0 (-0.8) | 4.8 to 4.0 (-0.8) | 4.8 to 3.2 (-1.6) | 2.8 to 1.2 (-1.6) |
| SafeGrad LoRA | 8.0 to 6.4 (-1.6) | 7.6 to 6.4 (-1.2) | 6.0 to 5.2 (-0.8) | 5.2 to 5.6 (+0.4) |

#### Speed and memory

| Arm | Precision | Adapter | Attack generation (tokens per second) | Peak GPU memory (MiB) |
|---|---|---|---:|---:|
| `gpu-base` | bfloat16 | none | 214.1 | 20858 (with the guard) |
| `gpu-plain` | bfloat16 | merged | 137.9 | 20112 (with the guard) |
| `plain-ko` | bfloat16 | merged | 136.6 | 18786 (with the guard) |
| `plain-en` | bfloat16 | merged | 164.9 | 20896 (with the guard) |
| `plain-zh` | bfloat16 | merged | 155.6 | 20834 (with the guard) |
| `gpu-safegrad` | bfloat16 | merged | 166.0 | 20780 (with the guard) |
| `gpu-base-nf4` | NF4 | none | 219.2 | 18822 (with the guard) |
| `gpu-plain-nf4` | NF4 | unmerged | 115.2 | 20864 (with the guard) |
| `gpu-safegrad-nf4` | NF4 | unmerged | 145.5 | 18856 (with the guard) |

Peak memory marked "with the guard" is the peak of a whole evaluation step, in which the guard model ran in the same process after generation. It is not the memory the model needs and does not show what 4-bit loading saves.

Speed is measured during generation alone. It depends on how long an arm's answers are, so it compares the two precisions of one arm and not one arm with another.
