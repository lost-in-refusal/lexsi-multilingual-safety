### Analyses of stored answers that the outside reviews asked for (section 18)

No model was run and no answer was generated: every figure is computed from answers, guard verdicts, a person's marks, task scores and run records that were already stored. Section 18 of the preregistration fixed each quantity before it was computed, and all nine are reported. **Only 18b and 18c have a sentence, and all of it is exploratory**: it was prompted by reviews of results we knew. The strict reading counts the guard's Unsafe and Controversial classes. Intervals are 95%, from a bootstrap over prompt families with each family's value averaged over the seeds (5,000 draws, the draws of `lir.seeds.pooled`).

#### 18a. The guard against the person, by run

The guard's strict reading (Unsafe or Controversial) against the person's mark, for the base answers and the plain answers apart. K2 and K3 are base and plain at seed 42 of Tiny Aya; 16d is the base run and plain at seed 42 of the model outside the Tiny Aya family. Unsure marks are counted beside and enter no share.

| Reading | Run | Answers | Marked unsafe: guard flagged | guard passed | Marked safe: guard flagged | guard passed | Unsure: flagged, passed | Of marked unsafe, the guard flagged | Of marked safe, the guard passed |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| K2 (Tiny Aya, Korean, 100 prompts) | base | 100 | 1 | 0 | 1 | 96 | 0, 2 | 1 of 1 (100%) | 96 of 97 (99%) |
| K2 (Tiny Aya, Korean, 100 prompts) | plain | 100 | 7 | 0 | 6 | 83 | 4, 0 | 7 of 7 (100%) | 83 of 89 (93%) |
| K3 (Tiny Aya, Korean, 161 prompts) | base | 161 | 4 | 6 | 3 | 145 | 0, 3 | 4 of 10 (40%) | 145 of 148 (98%) |
| K3 (Tiny Aya, Korean, 161 prompts) | plain | 161 | 10 | 3 | 9 | 126 | 4, 9 | 10 of 13 (77%) | 126 of 135 (93%) |
| 16d (Granite 4.1, Korean, 150 prompts) | base | 150 | 26 | 57 | 5 | 62 | 0, 0 | 26 of 83 (31%) | 62 of 67 (93%) |
| 16d (Granite 4.1, Korean, 150 prompts) | plain | 150 | 43 | 41 | 23 | 43 | 0, 0 | 43 of 84 (51%) | 43 of 66 (65%) |
| 120 labels (Tiny Aya, four languages, 30 answers each) | base | 36 | 20 | 3 | 5 | 8 | 0, 0 | 20 of 23 (87%) | 8 of 13 (62%) |
| 120 labels (Tiny Aya, four languages, 30 answers each) | plain | 84 | 31 | 7 | 3 | 40 | 1, 2 | 31 of 38 (82%) | 40 of 43 (93%) |

- K2, the guard's strict count of base against plain. On the 94 prompts the person's test used (both answers marked yes or no): 2 against 11, where the person marked 1 against 5 unsafe. On all 100 prompts of the sheet: 2 against 17.
- K3, the guard's strict count of base against plain. On the 145 prompts the person's test used (both answers marked yes or no): 6 against 18, where the person marked 5 against 11 unsafe. On all 161 prompts of the sheet: 7 against 23.

The 120 labels are the four languages together. Half of each language's items were picked because the guard had flagged them, so their two shares are not those of a random answer; K2 and 16d are random draws of prompts, and K3 is every Korean prompt that K2 and the label pack had left unread. The 120 labels are read against the verdict the key stored when the pack was drawn, and none differs from the verdict files as they are now.

#### 18b. The guard on harmless prompts

XSTest's safe prompts (250 a language, 249 in Swahili): the share of answers the guard labels Unsafe or Controversial, the fine-tuned runs minus the base model on the same prompts, over seeds 42, 1 and 2. Each cell: the base model's share to the fine-tuned runs' mean share (%), then the difference in points with its 95% interval.

| | en | ko | zh | sw |
|---|---|---|---|---|
| Tiny Aya, plain minus base (the registered quantity) | 1.6 to 5.3, +3.7 [+1.2, +6.4] | 5.6 to 11.1, +5.5 [+2.1, +9.1] | 4.8 to 12.3, +7.5 [+4.0, +11.2] | 10.4 to 13.0, +2.5 [-1.5, +6.4] |
| Tiny Aya, SafeGrad minus base | 1.6 to 3.6, +2.0 [-0.4, +4.5] | 5.6 to 8.9, +3.3 [+0.4, +6.4] | 4.8 to 10.7, +5.9 [+2.8, +9.3] | 10.4 to 10.6, +0.1 [-3.7, +4.0] |
| Tiny Aya, plain minus base, among the answers the guard calls no refusal | 1.8 to 5.7, +3.9 [+1.2, +6.8] | 6.1 to 11.6, +5.5 [+2.0, +9.3] | 5.3 to 13.0, +7.7 [+4.1, +11.6] | 11.3 to 13.5, +2.1 [-2.1, +6.3] |
| Tiny Aya, plain minus base, the guard's Unsafe class alone | 0.0 to 2.5, +2.5 [+0.8, +4.5] | 1.6 to 4.7, +3.1 [+1.1, +5.2] | 1.6 to 2.8, +1.2 [-0.5, +2.9] | 2.0 to 2.9, +0.9 [-0.7, +2.5] |
| The model outside the Tiny Aya family, plain minus base | 2.0 to 4.8, +2.8 [+0.9, +4.9] | 4.0 to 17.1, +13.1 [+9.6, +16.9] | 5.2 to 13.5, +8.3 [+4.5, +12.3] | 7.2 to 7.0, -0.3 [-3.5, +2.9] |

**The guard also flagged more of the fine-tuned model's Korean answers to harmless prompts, so a part of the Korean rise on the attack prompts, up to that size, may be the guard's reaction to the fine-tuned answers and not harm.** Korean, plain minus base, on the safe prompts: +5.5 [+2.1, +9.1] points, above zero.

Beside, with no rule. The Korean change minus the English change under plain: +1.7 [-2.3, +6.0]. The Korean rise on the attack prompts is +12.7 [+8.9, +16.7]; with the Korean change on the safe prompts taken off it, +7.2 [+2.0, +12.4], the attack families and the safe-prompt families each resampled on their own.

Among the answers the guard calls no refusal, the two shares are over different prompts: base has 226 such answers in en, 231 such answers in ko, 228 such answers in zh, 230 such answers in sw; the three plain runs together have 702 in en, 708 in ko, 710 in zh, 721 in sw. The figure section 18 names as seen comes back: at seed 42 over the four languages the guard flags 56 of base's 999 answers and 107 of plain's 999.

#### 18c. Answer length and the cap

The Korean attack answers of base and of plain at seeds 42, 1 and 2, split by whether the answer reached the 256-token cap and into three groups of equal size by its length in tokens. Inside a group: strict attack success of the plain answers, the three seeds put together, minus that of the base answers, with a 95% interval from a bootstrap over the 291 prompt families.

| Group | Tokens | Of them at the cap | Base answers | flagged (%) | Plain answers, three seeds | flagged (%) | Plain minus base (points) | Interval |
|---|---|---:|---:|---:|---:|---:|---|---|
| Complete answers | 10 to 255 | 0 | 193 | 1.6 | 811 | 14.7 | +13.1 [+9.4, +17.2] | above zero |
| Answers that reached the cap | 256 to 256 | 160 | 98 | 12.2 | 62 | 59.7 | +47.4 [+31.6, +64.7] | above zero |
| Shortest third | 10 to 62 | 0 | 48 | 2.1 | 340 | 7.4 | +5.3 [+0.5, +10.0] | above zero |
| Middle third | 62 to 112 | 0 | 60 | 1.7 | 328 | 17.4 | +15.7 [+8.8, +22.6] | above zero |
| Longest third | 112 to 256 | 160 | 183 | 7.1 | 205 | 36.1 | +29.0 [+20.3, +38.5] | above zero |

**The Korean rise is there among answers of the same length and among answers that were not cut.**

The thirds are cut over the 1,164 answers together (291 of base, counted once, and 873 of plain). The first cut falls among the answers of 62 tokens (12 of them below it, 1 above), the second among the answers of 112 tokens (1 of them below it, 1 above); answers of one length that a cut falls among are ordered by a hash of run and prompt. A group is defined by the answer, which the fine-tune changes, so this compares different prompts in the two runs and is a description, not a control; the re-generation of sections 9 and 13 is the registered answer to the cap.

#### 18d. The guard's Unsafe class alone on the model outside the Tiny Aya family

G1 to G3 by the code of section 16 with only the guard's Unsafe class counted, over seeds 42, 1 and 2, in points with 95% intervals. No sentence is attached; G1 to G3 stand as section 16 read them.

- G1, plain minus base, over English, Korean and Chinese: +8.9 [+6.7, +11.1], above zero. With Controversial counted too, as section 16 read it: +10.1 [+7.6, +12.6].
- G2, the Korean rise minus the English rise under plain: +18.6 [+13.6, +23.6], above zero. With Controversial counted too, as section 16 read it: +20.8 [+15.1, +26.3].
- G3, SafeGrad minus plain, over English, Korean and Chinese: -11.6 [-13.8, -9.4], below zero. With Controversial counted too, as section 16 read it: -13.7 [-16.1, -11.4].

| Run | Korean, Unsafe class (%) | English, Unsafe class (%) |
|---|---:|---:|
| `granite41-base` | 10.0 | 5.8 |
| `granite41-plain-s42` | 28.2 | 5.8 |
| `granite41-safegrad-s42` | 6.5 | 1.0 |
| `granite41-plain-s1` | 29.6 | 8.6 |
| `granite41-safegrad-s1` | 8.6 | 1.7 |
| `granite41-plain-s2` | 29.2 | 4.5 |
| `granite41-safegrad-s2` | 10.3 | 1.0 |

#### 18e. The repair comparison in full

Plain LoRA, SafeGrad and the two mixing arms at seeds 42, 1 and 2 of Tiny Aya, each run as rates. Strict attack success on MultiJail and refusals on XSTest's safe prompts in %, answers capped at 256 tokens; the task score is the mean over the 300 held-out prompts on the 1 to 7 scale.

| Method | Seed | Attack success: en | ko | zh | sw | Over-refusal: en | ko | zh | sw | Task score | Optimizer steps | Training minutes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base model (`base`) |  | 6.9 | 5.2 | 8.6 | 6.2 | 9.6 | 7.6 | 8.8 | 7.6 | 6.32 |  |  |
| Plain LoRA (`plain`) | 42, seen | 6.9 | 19.2 | 13.4 | 7.6 | 7.2 | 5.6 | 5.6 | 2.8 | 5.94 | 171 | 75.7 |
| Plain LoRA (`plain-s1`) | 1 | 7.6 | 15.5 | 12.4 | 5.2 | 6.0 | 5.6 | 4.8 | 4.0 | 5.85 | 171 | 82.2 |
| Plain LoRA (`plain-s2`) | 2 | 8.9 | 18.9 | 13.1 | 6.9 | 6.0 | 5.6 | 5.6 | 3.6 | 5.88 | 171 | 75.7 |
| SafeGrad (`safegrad`) | 42, seen | 4.5 | 10.7 | 8.9 | 5.5 | 7.6 | 6.8 | 5.6 | 5.6 | 5.82 | 171 | 89.7 |
| SafeGrad (`safegrad-s1`) | 1 | 3.4 | 8.2 | 7.6 | 3.4 | 8.8 | 7.2 | 6.4 | 6.0 | 5.86 | 171 | 87.7 |
| SafeGrad (`safegrad-s2`) | 2 | 4.1 | 9.6 | 8.9 | 6.5 | 10.8 | 7.2 | 6.4 | 6.0 | 5.87 | 171 | 87.8 |
| Mixing, the refusal examples once (`plain-mix1`) | 42, seen | 3.8 | 9.3 | 6.5 | 4.8 | 14.0 | 7.2 | 8.8 | 7.6 | 5.95 | 174 | 75.6 |
| Mixing, the refusal examples once (`plain-mix1-s1`) | 1 | 3.4 | 8.6 | 6.2 | 3.8 | 14.4 | 9.6 | 10.4 | 10.0 | 5.88 | 174 | 85.9 |
| Mixing, the refusal examples once (`plain-mix1-s2`) | 2 | 3.4 | 8.2 | 6.5 | 5.2 | 15.2 | 9.2 | 8.8 | 8.4 | 5.92 | 174 | 76.0 |
| Mixing, five copies (`plain-mix5`) | 42, seen | 3.4 | 8.9 | 7.9 | 4.8 | 13.2 | 8.0 | 9.2 | 6.4 | 5.94 | 192 | 82.2 |
| Mixing, five copies (`plain-mix5-s1`) | 1 | 2.7 | 6.2 | 5.8 | 6.2 | 15.2 | 11.2 | 9.2 | 8.8 | 5.92 | 192 | 82.7 |
| Mixing, five copies (`plain-mix5-s2`) | 2 | 3.4 | 6.2 | 6.2 | 4.5 | 14.8 | 10.0 | 9.6 | 6.8 | 5.84 | 192 | 82.2 |

Seed 42, marked seen, was known before the mixing rules of section 14a were written; seeds 1 and 2 of the mixing arms were run after them. Training minutes are on the laptop and depend on what else it was doing. The base model is shown for reference.

#### 18f. Korean against the other languages

Strict attack success under plain against base, per prompt family, pooled over seeds 42, 1 and 2, in points with 95% intervals.

- MultiJail: the Korean rise +12.7 [+8.9, +16.7], the Chinese rise +4.4 [+1.1, +7.7], the Korean rise minus the Chinese rise +8.4 [+3.7, +13.1], above zero.
- The second benchmark, the unsafe prompts of XSTest (14i): the Korean rise +10.5 [+5.8, +15.3], the Chinese rise +11.8 [+7.7, +16.3], the Korean rise minus the Chinese rise -1.3 [-7.2, +4.5], includes zero.

Quoted from `more_languages.json`, not recomputed. U3 of section 13, in its registered words: As the guard measures it, no further language rose more than Korean, and the rise in some of them could not be told apart from Korean's. "Korean is hit hardest" is then written only as: hardest among the four languages of the main evaluation, and those languages are named as not distinguishable from Korean. Korean's rise minus each further language's: it +13.5 [+8.9, +18.4], vi +4.6 [-0.0, +9.3], ar +10.9 [+5.5, +16.3], th +1.0 [-3.8, +6.0], bn +11.1 [+5.5, +17.1], jv +3.3 [-2.3, +8.7].

#### 18g. Task score and answer length by language

The 300 held-out task prompts, 100 a language, answered at up to 1,024 tokens. The fine-tuned columns are the mean over seeds 42, 1 and 2 (for lengths, the mean of the three medians). Task score on the 1 to 7 scale.

| Method | Language | Prompts | Task score: base | fine-tuned | Median characters: base | fine-tuned | Median tokens: base | fine-tuned |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Plain LoRA | en | 100 | 6.61 | 6.37 | 1619 | 521 | 340 | 108 |
| Plain LoRA | ko | 100 | 6.10 | 5.51 | 928 | 276 | 441 | 124 |
| Plain LoRA | zh | 100 | 6.25 | 5.79 | 761 | 190 | 472 | 123 |
| SafeGrad | en | 100 | 6.61 | 6.31 | 1619 | 508 | 340 | 116 |
| SafeGrad | ko | 100 | 6.10 | 5.49 | 928 | 291 | 441 | 132 |
| SafeGrad | zh | 100 | 6.25 | 5.74 | 761 | 186 | 472 | 126 |

Scores and characters are computed by `lir.tasklen` and are the figures of `task_length.md`; the tokens are added here. A prompt is read where base and all three runs of the method have a score.

#### 18h. Drift against the task score over the five training settings

The table of section 14h with the default run, read from `sensitivity.json`: plain LoRA at seed 42 with the learning rate or the number of epochs halved or doubled. Rises are in strict attack success against base, in points, with the exact McNemar p; the task score against base has a 95% interval over the 300 prompts.

| Run | Learning rate | Epochs | Korean rise | English rise | Task score | Task score against base | Training loss |
|---|---:|---:|---|---|---:|---|---:|
| our settings (`plain`) | 0.0002 | 3 | +14.1 (p < 0.001) | +0.0 (p = 1.000) | 5.94 | -0.38 [-0.52, -0.25] | 1.425 |
| learning rate halved (`plain-lr-half`) | 0.0001 | 3 | +13.1 (p < 0.001) | +2.1 (p = 0.238) | 5.71 | -0.61 [-0.77, -0.44] | 1.503 |
| learning rate doubled (`plain-lr-double`) | 0.0004 | 3 | +16.5 (p < 0.001) | +3.1 (p = 0.064) | 5.94 | -0.39 [-0.52, -0.25] | 1.324 |
| half as many epochs (`plain-ep-half`) | 0.0002 | 1.5 | +21.0 (p < 0.001) | +4.8 (p = 0.004) | 5.67 | -0.66 [-0.82, -0.48] | 1.551 |
| twice as many epochs (`plain-ep-double`) | 0.0002 | 6 | +11.3 (p < 0.001) | +1.4 (p = 0.481) | 5.97 | -0.35 [-0.48, -0.23] | 1.171 |

A description of five runs at one seed. No trend is tested.

#### 18i. The rank test of 15a on the 40 new plans alone

The random plans the gate had not shown (seeds 6 to 84), on perplexity. p is the number at or below the circuit plan plus one, over 41.

- With 7, 15 and 14 layers at 8, 4 and 2 bits: the circuit plan's perplexity is 15.92; 0 of the 40 new plans are at or below it (p = 0.024). The new plans run from 18.52 to 84.66. 15a's own count, with the gate's three: 0 of 43 (p = 0.023).
- With 3, 27 and 6 layers at 8, 4 and 2 bits: the circuit plan's perplexity is 7.81; 0 of the 40 new plans are at or below it (p = 0.024). The new plans run from 8.19 to 12.61. 15a's own count, with the gate's three: 0 of 43 (p = 0.023).

15a's own p counts the gate's three plans and is not replaced. Perplexity only: this says nothing about safety.

#### Readings chosen where section 18 leaves a choice

Each was chosen before any figure above was computed.

- 18a. An answer's verdict is found by run and uid. The 120 labels are read by run over the four languages together. The guard's two counts of base against plain are given for K2 and K3, the sheets section 18 names for them.
- 18b. Each item listed beside changes one thing in the registered quantity: the language, SafeGrad in place of plain (SafeGrad minus base), the answers counted, the class counted, or the model. Among the answers the guard calls no refusal, the share is taken over the three plain runs' such answers together and over base's, and their difference is resampled over prompt families; it is not a difference on the same prompts.
- 18c. Reached the cap is the stored `truncated` flag. Equal thirds are cut by rank of the token count over all Korean answers of base and plain, base counted once, with ties ordered by a hash. Inside a group the three plain seeds' answers are put together, and the groups are held fixed while the 291 families are resampled.
- 18d. The code of section 16 is run on copies of the verdicts with Controversial written as Safe.
- 18e. Over-refusal is the guard's refusal label on XSTest's safe prompts. The task score is the mean over a run's scored answers.
- 18g. Answer length is given in characters, the unit of section 13h, and in tokens.
- 18h. The table is read from the stored report of section 14h and nothing is computed again.
- 18i. Both tier sizes that have all 40 new plans measured.

Checked before anything was written: the table by run of each sheet gives the counts its scoring command stored; the figure of 18b that section 18 names as seen; the group figure of 18c with every answer in one group is the pooled Korean rise; G1 to G3 with both classes are those of `granite41/seeds.md`; the count over all 43 plans is that of `quantization_ranks.md`.
