# Numbers and findings handover (Mingyu → Aish)

Everything below comes from files on the `mingyu/tiny-aya-pipeline` branch (PR #15) and the published eval outputs on
Hugging Face. Still pending: the judge-human agreement numbers, which come from the labels (Tue Nov 10).

## 1. Final tables

Both files are final. `results/lir/results.json` is the source: `python -m lir.metrics --arms base plain safegrad`
writes it from the published outputs, and it also has the n, the 95% intervals and AuditKIT's own p-values.
`results/lir/results.md` is the same numbers as tables. Quote the `.md`, and go to the `.json` for anything the
tables leave out.

Tiny Aya Global (base), plain LoRA and SafeGrad LoRA. 291 MultiJail families per language, 250 XSTest safe prompts
per language, 100 held-out Bactrian-X tasks per training language. Strict = the guard says Unsafe or Controversial;
loose = Unsafe only.

### vs the base model (`results/lir/results.md`)

#### Attack success, strict (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| plain | 6.9 (+0.0) | 19.2 (+14.1*) | 13.4 (+4.8*) | 7.6 (+1.4) | 11.8 |
| safegrad | 4.5 (-2.4) | 10.7 (+5.5*) | 8.9 (+0.3) | 5.5 (-0.7) | 7.4 |

#### Attack success, loose (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 4.1 | 2.7 | 4.5 | 1.4 | 3.2 |
| plain | 4.5 (+0.3) | 14.1 (+11.3*) | 10.0 (+5.5*) | 1.7 (+0.3) | 7.6 |
| safegrad | 3.1 (-1.0) | 6.2 (+3.4*) | 5.5 (+1.0) | 1.4 (+0.0) | 4.0 |

#### Refusal rate on attack prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 71.1 | 59.8 | 66.3 | 50.9 | 62.0 |
| plain | 66.7 (-4.5) | 47.4 (-12.4*) | 58.1 (-8.2*) | 37.8 (-13.1*) | 52.5 |
| safegrad | 75.9 (+4.8*) | 57.0 (-2.7) | 67.4 (+1.0) | 56.4 (+5.5*) | 64.2 |

#### Over-refusal on XSTest safe prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 9.6 | 7.6 | 8.8 | 7.6 | 8.4 |
| plain | 7.2 (-2.4) | 5.6 (-2.0) | 5.6 (-3.2) | 2.8 (-4.8*) | 5.3 |
| safegrad | 7.6 (-2.0) | 6.8 (-0.8) | 5.6 (-3.2) | 5.6 (-2.0) | 6.4 |

#### Refusal on task prompts (%)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| base | 0.0 | 0.0 | 0.0 | 0.0 |
| plain | 0.0 (+0.0) | 0.0 (+0.0) | 0.0 (+0.0) | 0.0 |
| safegrad | 0.0 (+0.0) | 0.0 (+0.0) | 1.0 (+1.0) | 0.3 |

#### Task score (1-7)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| base | 6.61 | 6.10 | 6.25 | 6.32 |
| plain | 6.36 (-0.25*) | 5.59 (-0.51*) | 5.86 (-0.39*) | 5.94 |
| safegrad | 6.33 (-0.28*) | 5.40 (-0.70*) | 5.72 (-0.53*) | 5.82 |

Parentheses: change vs the reference arm on the same prompts; * = p < 0.05 (exact McNemar for rates, centred paired bootstrap for task score). macro = equal-weight mean over languages.

### SafeGrad vs plain LoRA, same prompts (`results/lir/vs_plain/results.md`)

#### Attack success, strict (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| plain | 6.9 | 19.2 | 13.4 | 7.6 | 11.8 |
| safegrad | 4.5 (-2.4) | 10.7 (-8.6*) | 8.9 (-4.5*) | 5.5 (-2.1) | 7.4 |

#### Attack success, loose (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| plain | 4.5 | 14.1 | 10.0 | 1.7 | 7.6 |
| safegrad | 3.1 (-1.4) | 6.2 (-7.9*) | 5.5 (-4.5*) | 1.4 (-0.3) | 4.0 |

#### Refusal rate on attack prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| plain | 66.7 | 47.4 | 58.1 | 37.8 | 52.5 |
| safegrad | 75.9 (+9.3*) | 57.0 (+9.6*) | 67.4 (+9.3*) | 56.4 (+18.6*) | 64.2 |

#### Over-refusal on XSTest safe prompts (%)

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| plain | 7.2 | 5.6 | 5.6 | 2.8 | 5.3 |
| safegrad | 7.6 (+0.4) | 6.8 (+1.2) | 5.6 (+0.0) | 5.6 (+2.8*) | 6.4 |

#### Refusal on task prompts (%)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| plain | 0.0 | 0.0 | 0.0 | 0.0 |
| safegrad | 0.0 (+0.0) | 0.0 (+0.0) | 1.0 (+1.0) | 0.3 |

#### Task score (1-7)

| Arm | en | ko | zh | macro |
|---|---:|---:|---:|---:|
| plain | 6.36 | 5.59 | 5.86 | 5.94 |
| safegrad | 6.33 (-0.03) | 5.40 (-0.19*) | 5.72 (-0.14) | 5.82 |

Parentheses: change vs the reference arm on the same prompts; * = p < 0.05 (exact McNemar for rates, centred paired bootstrap for task score). macro = equal-weight mean over languages.

### Judge-corrected attack success (`results/lir/swahili_backtranslation_check.json`)

An answer counts as unsafe if the guard flags it as written or after back-translation to English. Strict judge.

| | Korean | Mandarin | Swahili |
|---|---:|---:|---:|
| Tiny Aya Global | 7.6 | 9.3 | 16.5 |
| + plain LoRA | 22.7 (+15.1\*) | 14.4 (+5.2\*) | 23.0 (+6.5\*, p = 0.009) |
| + SafeGrad LoRA | 13.4 (+5.8\*) | 9.6 (+0.3) | 15.5 (−1.0; −7.6 vs plain, p = 0.0007) |

## 2. Top findings, one line each

1. **Korean breaks most, and an English check would miss it.** Plain LoRA on benign data: English attack success
   6.9 → 6.9%, Korean 5.2 → 19.2% (+14.1\*), Mandarin 8.6 → 13.4% (+4.8\*).
2. **SafeGrad recovers Korean by 8.6 points and Mandarin by 4.5 against plain.** It raises refusal by 9–19 points in
   all four languages, most in Swahili (+18.6), which has no training or alignment data. Over-refusal stays below
   the base model.
3. **The judge misses about a quarter of harmful Swahili answers.** 24–28% of the Swahili answers it passes become
   unsafe after back-translation (Korean 7–10%, Mandarin 3–4%). Counting those, Swahili goes 16.5 → 23.0% after
   plain LoRA and back to 15.5% with SafeGrad.
4. **One shared refusal mechanism across languages, weakened most in Korean.** The English refusal direction
   separates harmful from harmless prompts in every language (AUROC 0.97–0.99 at layer 17). Any two languages
   share 136–155 of their top-200 circuit nodes (65 by chance). Plain LoRA halves Korean's circuit contrast
   (6.2 → 3.0), the largest drop.
5. **Replicated:** Korean is hit hardest on Hannah's independent training sample too (see section 6).

## 3. Caveats not to overstate

**Swahili judge result**
- Base Tiny Aya did the Swahili back-translation, so the translator is the model under test before fine-tuning.
  It is the same translator for all three arms. No one on the team reads Swahili, so a person checking the
  back-translation is as far as we can go.
- The 24–28% is a share of the answers the guard passed **that did not refuse** (30/125 base, 45/159 plain,
  29/112 SafeGrad), not a share of all Swahili answers.
- Without the correction, the Swahili strict rise after plain is +1.4 and not significant. The "drift reaches
  Swahili" claim rests on two things: the corrected numbers, and the refusal drop (50.9 → 37.8%\*). The guard
  decides both.
- Say "the guard undercounts harm in Swahili", not "Swahili is X% unsafe". The corrected rate assumes the guard is
  right about the back-translated English.
- The Swahili human labels use the same back-translation, so Swahili agreement may come out below 80%. If it does,
  report it as is.

**Cost to task score**
- Both fine-tunes score below the base model in every language: macro 6.32 base, 5.94 plain, 5.82 SafeGrad, and
  each drop is significant. Don't write "SafeGrad has no task cost".
- SafeGrad against plain: −0.12 macro, and only Korean (−0.19) is significant.
- Most of the drop comes from the fine-tune itself. Bactrian-X's short GPT-3.5 answers cut answer length from about
  420 to 170 tokens with no gain in reference match (chrF en 41.1 → 40.5, ko 18.6 → 18.3, zh 13.1 → 15.3).
- The task judge is Qwen3-14B (Tiny Aya's 1–7 rubric) on 100 tasks per language, and no person has checked it.

**Steering**
- It ran on 80 of the 291 families per language (the same 80 for every setting), on the plain model only. On that
  subset plain's Korean attack success is 21.2%, not the 19.2% in the main table.
- It was one layer (17) at two strengths, with no task score measured under steering.
- At full strength with the English direction, Korean drops 21.2 → 1.2%, but XSTest over-refusal goes 7.2 → 25.3%
  macro on the same subset.
- At half strength, attack success is 5.0% against SafeGrad's 6.6% on the same prompts, but over-refusal is 10.6%
  against 5.6%.
- So this is evidence the direction matters causally, not a fix we tested for deployment.
- **Over-refusal claims for SafeGrad:** it keeps over-refusal below the base model (6.4% vs 8.4% macro), but it is
  1.1 points above plain, significant only in Swahili (+2.8). Say "stays below the base model", not "no rise".

**General**
- Everything is one seed, one LoRA configuration and one model size.
- All safety and task labels come from model judges. The judge-human agreement check is still pending.

**Circuits**
- Refusal is spread out: the top 50 of 612 nodes recover 3% of the contrast on held-out English pairs, and the top
  200 recover 65%. That's why the comparisons use 200 nodes.
- This size sweep was run on the English base model only.

## 4. ROI (`docs/lir/roi.md`, `results/lir/roi.json`)

**Cost per 1M tokens, on our workload's mix (935k input, 473k output tokens per evaluation pass)**

| | List price, input / output per 1M | Per 1M tokens, our mix | One eval pass |
|---|---:|---:|---:|
| GPT-5 mini | $0.25 / $2.00 | **$0.84** | $1.18 |
| GPT-5 nano | $0.05 / $0.40 | **$0.17** | $0.24 |
| Tiny Aya Global, self-hosted (measured) | n/a | **0.40 machine-hours** | 0.57 h |

- Self-hosting means one Apple M5 Pro laptop running plain HF transformers in bf16, at 140–280 output tokens/s
  measured, with no batching server.
- **Break-even:** self-hosting is cheaper than GPT-5 mini whenever the machine costs under $2.08 an hour to own and
  run, and cheaper than GPT-5 nano under $0.42 an hour.
- The API prices are list prices as reported for Oct 2026. Check them on openai.com/api/pricing before quoting.

**SafeGrad's extra cost**
- Training takes 75.7 → 89.7 minutes, +14.0 min (+18.5%), for 1,800 rows × 3 epochs.
- Final training loss is 1.425 → 1.436 (+0.8%).

## 5. Figures

| File | Shows | Where |
|---|---|---|
| `results/lir/headline.png` | Three panels, base vs plain vs SafeGrad by language: attack success (strict), XSTest over-refusal, task score. Bars on the two rate panels are Wilson 95% intervals | README. Not in `article.md` yet; suggested as Figure 1 next to the results table |
| `results/lir/refusal_geometry.png` | Four panels: cosine of each language's refusal direction to English by layer; AUROC of the English direction by layer (held-out families); harmful−harmless projection gap at layer 17 for each model; refusal rate against median projection (each language × model) | README and `article.md` (probes section) |

- There is no circuits or steering figure; those results are tables in `article.md`.
- The demo Space (`lost-in-refusal/lost-in-refusal-demo`) shows the three models' answers side by side, on MultiJail
  and XSTest prompts in all four languages.
- The article draft in `docs/lir/article.md` is complete except for the agreement number. Use as much of it as you
  like.

## 6. Replication on Hannah's data (`results/lir/replication/results.md`)

Plain LoRA on Hannah's separately curated sample (4 of 600 IDs shared) raises strict attack success by 16.5 points
in Korean, 4.8 in Mandarin, 6.5 in English and 5.5 in Swahili, all significant. Korean is hit hardest in both
samples, at least 2.5× English. English is not always untouched.

## 7. Lexsi tool feedback

- The list is complete: 14 issues, #1–#14 on `lost-in-refusal/lexsi-multilingual-safety`, all labelled
  `lexsi-feedback`. They match items 1–14 in `docs/lir/lexsi-feedback.md` one to one.
- The split is 4 SafeTune, 1 CuratorKIT, 2 AuditKIT and 7 CircuitKIT.
- None are filed on the Lexsi-Labs repos themselves yet. Whether to do that, and from whose account, is still to be
  decided.

## 8. Still to come

- **Judge-human agreement per language (Tue Nov 10).** `python -m lir.labelpack score` merges the three label files
  and prints agreement, the 95% interval, unsure/blank counts, and guard misses and false alarms per language.
  - Half of each language's items are ones the guard flagged, so this is agreement on a hard sample, not on a
    random answer.
  - `python -m lir.labelpack checks` summarises the translation check and the training spot check.
- **Hugging Face.** Data, adapters, every generation and judge verdict, and the circuits, steering and replication
  outputs are uploaded. The agreement results stay in GitHub only unless you want them on HF too.
