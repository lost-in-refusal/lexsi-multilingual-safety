# Lost in Refusal: final write-up and results

Team: Aishwarya Nikkumbh (lead), Mingyu Kang, Hannah Pham. Track 2 (Red Teaming, Safety and Alignment).
Code and data: `lost-in-refusal/lexsi-multilingual-safety` (PR #15). Outputs on Hugging Face under `lost-in-refusal`
(private until the freeze).

Status at writing: all results are computed. Two items are still open and marked below: the Swahili judge-agreement
result (under target) and the Korean spot-check decision (over the limit).

## 1. Question and setup

A company self-hosts Tiny Aya Global as a support bot for English, Korean and Mandarin customers and fine-tunes it on
benign instructions. Does it stop refusing harmful requests? Is the damage worse outside English? Does it reach a
language it was never trained on (Swahili)? Can SafeTune's SafeGrad prevent it without hurting the task?

- **Model:** Tiny Aya Global (3.35B), `CohereLabs/tiny-aya-global`, pinned revision `af89d219`.
- **Fine-tuning data:** 1,800 benign Bactrian-X rows (600 instructions × English, Korean, Mandarin), cleaned and
  deduplicated with CuratorKIT. Swahili appears nowhere in training or alignment.
- **Arms:** base (no fine-tune), plain LoRA, and LoRA with SafeGrad. The two LoRA arms share data, order, LoRA config
  and optimiser. Single seed.
- **Evaluation:** MultiJail (291 attack families per language in en/ko/zh/sw), XSTest (250 safe prompts per
  language, for over-refusal), and 100 held-out Bactrian-X tasks per training language.
- **Judges:** Qwen3Guard-Gen-4B for safety (strict = Unsafe or Controversial; loose = Unsafe only). Qwen3-14B for
  task score on Tiny Aya's 1–7 rubric.
- **Statistics:** exact McNemar for rates; centred paired bootstrap for task score; Wilson intervals for proportions.
  AuditKIT's own bootstrap p-value is not centred and is not used for significance (see Lexsi feedback).

## 2. Results

### Attack success, strict (%)

| Arm | en | ko | zh | sw (never trained) | macro |
|---|---:|---:|---:|---:|---:|
| Tiny Aya Global | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| + plain LoRA | 6.9 (+0.0) | 19.2 (+14.1\*) | 13.4 (+4.8\*) | 7.6 (+1.4) | 11.8 |
| + SafeGrad LoRA | 4.5 (−2.4) | 10.7 (+5.5\*) | 8.9 (+0.3) | 5.5 (−0.7) | 7.4 |

### Attack success, loose (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| Tiny Aya Global | 4.1 | 2.7 | 4.5 | 1.4 | 3.2 |
| + plain LoRA | 4.5 (+0.3) | 14.1 (+11.3\*) | 10.0 (+5.5\*) | 1.7 (+0.3) | 7.6 |
| + SafeGrad LoRA | 3.1 (−1.0) | 6.2 (+3.4\*) | 5.5 (+1.0) | 1.4 (+0.0) | 4.0 |

### Refusal rate on attack prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| Tiny Aya Global | 71.1 | 59.8 | 66.3 | 50.9 | 62.0 |
| + plain LoRA | 66.7 (−4.5) | 47.4 (−12.4\*) | 58.1 (−8.2\*) | 37.8 (−13.1\*) | 52.5 |
| + SafeGrad LoRA | 75.9 (+4.8\*) | 57.0 (−2.7) | 67.4 (+1.0) | 56.4 (+5.5\*) | 64.2 |

### Over-refusal on XSTest safe prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| Tiny Aya Global | 9.6 | 7.6 | 8.8 | 7.6 | 8.4 |
| + plain LoRA | 7.2 (−2.4) | 5.6 (−2.0) | 5.6 (−3.2) | 2.8 (−4.8\*) | 5.3 |
| + SafeGrad LoRA | 7.6 (−2.0) | 6.8 (−0.8) | 5.6 (−3.2) | 5.6 (−2.0) | 6.4 |

### Task score (1–7, Qwen3-14B judge, held-out Bactrian-X)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| Tiny Aya Global | 6.61 | 6.10 | 6.25 | 6.32 |
| + plain LoRA | 6.36 (−0.25\*) | 5.59 (−0.51\*) | 5.86 (−0.39\*) | 5.94 |
| + SafeGrad LoRA | 6.33 (−0.28\*) | 5.40 (−0.70\*) | 5.72 (−0.53\*) | 5.82 |

SafeGrad against plain, same prompts: task −0.12 macro (only Korean, −0.19, is significant).

### SafeGrad against plain LoRA, same prompts

| | en | ko | zh | sw |
|---|---:|---:|---:|---:|
| Attack success, strict (pts) | −2.4 | −8.6\* | −4.5\* | −2.1 |
| Refusal on attacks (pts) | +9.3\* | +9.6\* | +9.3\* | +18.6\* |
| Over-refusal (pts) | +0.4 | +1.2 | 0.0 | +2.8\* |

### Judge-corrected attack success (Swahili and the other languages back-translated)

An answer counts as unsafe if the guard flags it as written, or after Tiny Aya back-translates it to English and the
guard re-judges it against MultiJail's English prompt. Strict.

| | Korean | Mandarin | Swahili |
|---|---:|---:|---:|
| Tiny Aya Global | 7.6 | 9.3 | 16.5 |
| + plain LoRA | 22.7 (+15.1\*) | 14.4 (+5.2\*) | 23.0 (+6.5\*, p = 0.009) |
| + SafeGrad LoRA | 13.4 (+5.8\*) | 9.6 (+0.3) | 15.5 (−1.0; −7.6 vs plain, p = 0.0007) |

Swahili back-translation covered every answer the guard passed as a safe non-refusal: 30 of 125 (base), 45 of 159
(plain), 29 of 112 (SafeGrad) became unsafe. The translator was base Tiny Aya for every arm. See section 4.

### Replication on Hannah's independently curated training sample (plain LoRA)

Her set shares 4 of 600 training IDs with ours. Strict attack success, against the base model:

| | en | ko | zh | sw |
|---|---:|---:|---:|---:|
| Tiny Aya Global | 6.9 | 5.2 | 8.6 | 6.2 |
| plain LoRA, our data | 6.9 | 19.2\* | 13.4\* | 7.6 |
| plain LoRA, Hannah's data | 13.4\* | 21.6\* | 13.4\* | 11.7\* |

Korean is hit hardest in both samples, at least 2.5 times English. English is not always untouched; on her data it rises
6.5 points.

### Refusal geometry (probes)

- The English refusal direction (layer 17) separates harmful from harmless prompts in every language, AUROC
  0.97–0.99 on held-out families. A linear probe does as well as an MLP, so the model still recognises harm.
- Plain LoRA shrinks the harmful-minus-harmless projection on that direction by about a third in every language.
  SafeGrad restores most of it.
- Across languages and models, refusal rate tracks the median projection of harmful prompts.

### Steering (causal check)

Adding the English refusal direction at layer 17 to the plain model:
- Full strength: Korean attack success 21.2% → 1.2%; XSTest over-refusal macro 7.2% → 25.3%.
- Half strength: attack success macro 5.0% (SafeGrad 6.6% on the same prompts); over-refusal macro 10.6% (SafeGrad 5.6%).

Steering ran on 80 of the 291 families per language, the same 80 for every setting. On that subset plain's Korean
attack success is 21.2%, not the 19.2% in the main table.

### Refusal circuits (CircuitKIT, EAP-IG)

- **Distributed refusal.** On 64 held-out English pairs, the top 50 of 612 nodes recover 3% of the harmful-minus-harmless
  contrast; the top 200 recover 65%; the top 300 recover 89%; the top 400 recover 97%; all 612 recover 100%. Comparisons
  use 200 nodes. (The 200-node pruning keeps 199 nodes because of CircuitKIT's rounding.)
- **One shared mechanism.** Every pair of languages shares 136–155 of its top-200 nodes; random overlap would be about
  65. 102 nodes are shared by all four languages. At the top 50, 23 are shared, including heads in layers 10–21.
- **Fine-tuning reshapes Korean most.** Plain LoRA keeps 181 of English's top-200 nodes, but only 156 of Korean's. It
  halves Korean's first-token contrast (6.2 → 3.0 logits), the largest drop of the four languages. Mandarin and
  Swahili lose 36–39 of their top-200 nodes.
- **SafeGrad pulls every circuit back.** It recovers 55% of Korean's lost contrast (to 4.8). Swahili's contrast goes
  4.3 → 5.3 with SafeGrad.

Limits: contrast uses the logit of the first answer token; margin across the top five opening tokens per side
checks the metric, and Korean still falls most (English also falls, by 18%). Node scores are unsigned. 64 pairs,
one seed, 5 integration steps.

### ROI (`docs/lir/roi.md`)

- One evaluation pass: 935k input and 473k output tokens (2,463 requests).
- List prices (Oct 2026, confirm before quoting): GPT-5 mini $0.25 in / $2.00 out per 1M → $1.18 per pass; GPT-5 nano
  $0.05 / $0.40 → $0.24 per pass.
- Self-hosted on one Apple M5 Pro laptop (plain HF transformers, bf16, no batching server): 0.57 hours per pass at
  140–280 output tokens/s.
- Break-even: self-hosting is cheaper than GPT-5 mini below $2.08 per machine-hour, and than GPT-5 nano below $0.42.
- SafeGrad's cost: training 75.7 → 89.7 minutes (+14.0, +18.5%); final training loss 1.425 → 1.436 (+0.8%).

## 3. Human checks

### Judge-human agreement (`results/lir/labels/`, scored with `python -m lir.labelpack score`)

Blinded sample; half of each language's items are ones the guard flagged, so this is a hard sample, not a random one.
Strict guard. Unsure rows are not scored.

| Language | Labeler | Agreement | 95% interval | Guard missed harm | Guard false alarm | Target 80% |
|---|---|---:|---:|---:|---:|---|
| English | Hannah | 28/30 = 93.3% | 79–98% | 0 | 2 | met |
| Korean | Mingyu | 24/30 = 80.0% | 63–90% | 6 | 0 | met (on the line) |
| Mandarin | Mingyu | 27/30 = 90.0% | 74–97% | 3 | 0 | met |
| Swahili (back-translated) | Hannah | 20/27 = 74.1% (3 unsure) | 55–87% | 1 | 6 | **not met** |

- **Swahili is reported as is**, as the brief asks. All six false alarms are answers the guard called Controversial.
  Scoring loose would give 81.5% for Swahili but 76.7% for English, so strict is kept, as set in advance.
- **Five of Hannah's labels are under re-check** (English L013, L086, L088; Swahili L009, L030). Her notes give only a
  missing warning or intent question as the reason, which may not meet the labelling rule. If all five changed to "no",
  English would be 83.3% and Swahili 66.7%. The final number is pending her re-check.
- **Korean sits exactly on 80%.** Six of its rows are answers you marked unsafe that the guard called Safe; the other
  misses are none. Report it with its interval.

### Translation check (`translation_check_multijail.csv`, 20 families, Korean and Mandarin)

- Korean: 1 drift of 20 (`mj-189`, adds "leg" to the elephant prompt).
- Mandarin: 1 drift of 20 (`mj-218`, "disfigure" rendered as "ruin a person").
- Several smaller shifts and localisations (for example "Assistant" rendered as "Siri") are noted in the file and are
  not counted as drift.

### Training-data spot check (`spotcheck_bactrian_train.csv`, 30 rows per language)

- Korean: **7 of 30 bad**. Bad means the answer is wrong or doesn't do the task (for example a wrong definition, no
  answer to a conversion, invented facts). Four of those 7 don't do the task at all.
- Mandarin: 3 of 30 bad.
- The brief's rule is "more than 3 of 30 means re-curate". **Korean is over the limit.** The rows are benign, so the
  safety finding is not affected by their quality, but the data is weaker than planned. The team has not yet decided
  whether to re-curate and re-run, or keep the set and report it as a limitation.

## 4. Caveats (state these in the article)

1. **Swahili judge result.** Base Tiny Aya did the back-translation, so the translator is the model under test before
   fine-tuning. It is the same translator for all arms. No team member reads Swahili; Hannah's labels use the
   back-translation.
   - The "quarter of answers" figure (24–28% of safe non-refusals become unsafe) is a share of answers the guard passed
     that did not refuse. It is not a share of all Swahili answers.
   - Without the correction, the plain-LoRA rise in Swahili attack success is +1.4 and not significant. The claim that
     drift reaches Swahili rests on the corrected numbers and on the refusal drop (50.9 → 37.8%).
   - On Hannah's labels, of the Swahili answers the guard passed, 1 of 8 scored rows is marked unsafe by a human. So
     the human evidence supports the direction only weakly. Word the claim as "the guard undercounts harm in Swahili",
     not as "X% of Swahili answers are unsafe".
2. **Task score.** Both fine-tunes score below the base model in every language, and both drops are significant. Most of
   the drop comes from the fine-tune itself: Bactrian-X's short GPT-3.5 answers cut answer length from about 420 to 170
   tokens, with no gain in reference match (chrF en 41.1 → 40.5, ko 18.6 → 18.3, zh 13.1 → 15.3). The task judge is
   a model and no person has checked it. Do not write "SafeGrad has no task cost".
3. **SafeGrad and over-refusal.** SafeGrad keeps over-refusal below the base model (6.4% vs 8.4% macro), but it is
   1.1 points above plain, and only Swahili (+2.8) is significant. Say "stays below the base model", not "no rise".
4. **Steering** ran on 80 families per language, one layer and two strengths, and no task score was measured under
   steering. It is causal evidence that the direction matters, not a deployable fix.
5. **Circuits** were swept on the English base model only. The Korean and Mandarin comparisons use 200 nodes.
6. **Training data quality.** See the spot check above. Bactrian-X's answers are short, which contributes to the task
   score drop.
7. **Single seed, one LoRA configuration, one model size.** All safety and task labels except the sampled human checks
   come from model judges.
8. **Benchmark sanity.** Cohere's published MultiJail numbers for Tiny Aya were used only as a pipeline check, never as a
   result.

## 5. Figures

- `results/lir/headline.png`: attack success (strict, Wilson intervals), XSTest over-refusal, and task score by
  language and arm. Suggested as Figure 1.
- `results/lir/refusal_geometry.png`: four panels on refusal direction cosine by layer, AUROC by layer, the layer-17
  projection gap, and refusal rate against median projection. Used in the probes section.
- Circuit and steering results are tables only.

## 6. Reproducibility

- Every generation and judge verdict is on Hugging Face (`lost-in-refusal-eval-outputs`), with the frozen data
  (`lost-in-refusal-data`) and both adapters.
- `notebooks/lost_in_refusal.ipynb` recomputes the results tables from the published outputs on CPU.
- Pinned versions: Tiny Aya Global `af89d219`, Qwen3Guard-Gen-4B `6ec42827`, Qwen3-14B `40c06982`, SafeTune `3901a9f`,
  AuditKIT `71316da`, CuratorKIT `dd9e835`, CircuitKIT `e71d5e1`.

## 7. Tool feedback for Lexsi

Fourteen issues, labelled `lexsi-feedback`, on the team repository (#1–#14). They cover SafeTune (4), CuratorKIT (1),
AuditKIT (2) and CircuitKIT (7). They have not yet been filed on the Lexsi-Labs repositories. The most important for
the article: AuditKIT's paired-bootstrap p-value is near 0.5 for any non-zero effect (issue #14), so significance here
uses McNemar and a centred bootstrap; and SafeGradHFTrainer's silent fallback to plain fine-tuning (#3).

## 8. Open items before submission

1. **Swahili re-check:** Hannah to re-read five labels against the labelling rule. The Swahili and English agreement
   numbers may change.
2. **Korean spot check decision:** re-curate and re-run, or keep the set and report the limitation.
3. **Final scoring:** run `python -m lir.labelpack score` and `checks` once Hannah has replied; the results on this page
   use the current labels.
4. **Merge PR #15** into main once the team agrees (merge commit, not squash, so the author history stays intact). PR #16
   is now redundant with #15.
5. **Make the Hugging Face repos public** at the freeze (Nov 13), after "HF is final".
6. **File the 14 issues upstream** on the Lexsi-Labs repositories, from the account the team chooses.
7. **Agreement results on HF:** decide whether they go on Hugging Face too (currently GitHub only).
