# Results paragraphs for the paper (D15, draft for Aish to edit in)

Written Nov 22. One paragraph per deliverable, D1 to D8, D11, D12 and D18, each with its confidence label and the file its
numbers come from. Every number is from `results/lir/stage2/`; the status page there
(`results/lir/stage2/README.md`) says what each result may and may not be used for. Intervals are 95%. Attack
success is strict (guard says Unsafe or Controversial) unless a line says otherwise.

Labels follow the brief: High = three seeds, preregistered, significant. Medium = consistent direction, limited
seeds or sample. Low = single seed, small sample, or not significant.

## D1. Korean drift. Confidence: High

Plain LoRA fine-tuning on 1,800 harmless Bactrian-X rows raised Korean attack success from 5.2% to 19.2%, 15.5% and
18.9% at seeds 42, 1 and 2 (+14.1, +10.3 and +13.7 points, each significant by an exact McNemar test). English did
not move (+0.9 points on average, interval -1.8 to +3.8). The pooled Korean-minus-English gap is +11.8 points
(+6.8 to +16.7), and the preregistered check passes in every seed. No harmful example was in the training data.

Source: `seeds.md`, `drift_by_language.json`.

## D2. SafeGrad repair. Confidence: High for the direction

Training the same adapter with SafeGrad left Korean attack success 8.6, 7.2 and 9.3 points below plain fine-tuning
at the three seeds (pooled -8.4 points, -11.3 to -5.5), and the check passes in every seed. The repair is partial:
SafeGrad stays 5.5, 3.1 and 4.5 points above the base model, so it takes back roughly 60 to 70% of the rise. On
the second dataset (D7) it takes back about half, and on the second model (D8) between half and 60% depending on
whether capped answers are completed. SafeGrad added 13.7% to training time on average.

Source: `seeds.md`, `train_time.md`, `aya/seeds.md`, `earth/seeds.md`.

## D3. Swahili spillover. Confidence: High for refusals, Medium for corrected attack success

Swahili appears nowhere in training, yet the refusal rate on Swahili attack prompts fell 13.1, 6.2 and 8.6 points
under plain fine-tuning (pooled -9.3 points, -13.7 to -5.2); the check passes in every seed. The guard's own Swahili
attack success did not rise (+1.4, -1.0 and +0.7 points, none significant), and the guard missed our 80% agreement
bar in Swahili. When the answers the guard passed are translated to English with NLLB-200 and judged again,
attack success is 17.9% for base against 29.2%, 25.1% and 25.4% for plain; we report that as a direction, not as
a share of unsafe answers. Two limits belong with it. Refusals fall in all four languages under plain, so Swahili
does not fall more than the trained languages; what the result shows is that the fall reaches a language with no
training rows. And on the second dataset the Swahili refusal drop did not carry over (-2.6 points, -6.9 to +1.5),
so this finding rests on Bactrian-X.

Source: `seeds.md`, `swahili_nllb_seeds.json`, `aya/seeds.md`.

## D4. Truncation audit. Confidence: High

Attack and XSTest answers were capped at 256 new tokens, and base answers reach the cap more often than
fine-tuned ones. We generated every capped answer of base, plain and SafeGrad again at 1,024 tokens, at all three
seeds, and judged them again. The three main checks pass before and after: the Korean rise averages +12.7 points
before and +13.2 after, SafeGrad against plain is -8.4 and -8.6, and the Swahili refusal change is -9.3 and -10.0.
The same second reading was made for the runs with no Korean rows, for Tiny Aya Earth on three seeds, for the Aya
runs and for the single-language runs, each registered before its answers were generated again, and each came out
the same. The headline does not depend on the 256-token cap.

Source: `truncation/report.md`, `truncation/k1.md`, `earth/seeds.md`, `aya/seeds.md`, `single_language.md`.

## D5. Harm categories. Confidence: Medium (descriptive, no test between categories)

The Korean rise is not confined to one kind of harm. Averaged over three seeds, Korean attack success rose in all
eight MultiJail categories that have 20 or more prompts, by 3.7 to 21.0 points, on the capped answers and on the
re-generated ones, and SafeGrad was below plain in all eight. A prompt with several tags counts in each, and no
test compares one category with another, so we do not rank them.

Source: `harm_categories.md`, `harm_categories_regenerated.md`.

## D6. Training language and drift. Confidence: High (three seeds, preregistered, significant in every run)

Korean drifts whether or not Korean is in the training data. With English and Chinese rows only, Korean attack
success still rose 13.1, 13.4 and 13.1 points over three seeds, no different from training with Korean rows
(+0.5 points, -2.3 to +3.3), also after capped answers were completed. Trained on one language at a time (600
rows, three seeds), Korean rose 7.9, 8.9 and 7.2 points with English rows only, 12.0, 9.6 and 12.0 with Korean rows only and 11.3, 11.0 and 12.4 with Chinese
rows only, each significant and above the bar. In the two runs that can tell a weak language from a trained one, Korean rose more
than the language the model was trained on in every seed: pooled, by 7.6 points more than English in the English-only run (+3.7 to
+11.7) and 7.9 more than Chinese in the Chinese-only run (+3.3 to +12.6); at seed 42 both also hold after re-generation. This
supports reading D1 as a weakness of Korean in this model. The training language still changes the size of the rise: English rows alone raised Korean 4.6 points less than a 519-row mixed set (-7.8 to -1.4), while Korean rows alone and Chinese rows alone were not detectably different from it. That comparison was registered after its seed 42 value had been seen. No other test ranks the three runs. Counting only the guard's Unsafe class, Korean still rises more than the trained language on three seeds in both runs (+5.7 points, +2.3 to +9.6, and +4.1, +0.2 to +8.0); read at seed 42 alone, the Chinese-only run does not show it (+3.4, -0.7 to +7.9).

The three single-language runs were repeated on a cloud GPU (section 12.1, Hannah's jobs), each against the base model evaluated on that GPU: Korean rose 8.6 points with Korean rows only, 8.9 with English rows only and 11.3 with Chinese rows only, each significant and above the bar. That check was registered before the laptop runs but run after their results were known.

Note for Aish: the laptop runs are section 13a (checks SL1 and SL2, and C2 in 13e) and the GPU runs are your section 12.1 (L1 and L2). Seeds 1 and 2 of the laptop runs were read on capped answers only.

Source: `noko.md`, `truncation/k1.md`, `single_language_seeds.md`, `single_language.md`, `gpu.md`.

## D7. Second dataset: Aya. Confidence: High for drift and repair; the Swahili part fails

On 519 rows of the Cohere Aya Dataset (173 per language, written or edited by people) in place of Bactrian-X,
Korean attack success rose 10.0, 8.9 and 9.3 points at the three seeds (each significant) while English moved by
0.0, -0.7 and -1.0; the pooled gap is +10.0 points (+5.8 to +14.4). SafeGrad was 6.5, 3.1 and 4.8 points below
plain on Korean (pooled -4.8, -7.2 to -2.6); at seed 1 the difference is not significant alone. Both checks also
pass after capped answers were completed. SafeGrad took back about half of the rise here and stayed above base
in every seed. The Swahili refusal drop did not carry over (D3). Two limits: Cohere built both the model and
this dataset, so the model may have seen these rows, and the set is 29% of the size of the Bactrian-X one.

Source: `aya/seeds.md`, `aya/checks.md`, `docs/lir/cards/aya.md`.

## D8. Second model: Tiny Aya Earth. Confidence: High for drift and repair (three seeds); the Swahili forecast was not confirmed

On Tiny Aya Earth, with the same data and settings, plain fine-tuning raised Korean attack success from 10.7% to
21.3%, 19.2% and 21.6% (+10.7, +8.6 and +11.0 points, each significant; pooled +10.1, +6.4 to +14.0), and
SafeGrad was 4.8, 5.8 and 7.9 points below plain (pooled -6.2, -9.2 to -3.4, significant in each seed). Both
hold after capped answers were completed. SafeGrad again stays above the base model. We had forecast a larger
Swahili refusal drop on Earth than on Global and found a smaller one in every seed (6.9, 4.8 and 3.8 points
against 13.1, 6.2 and 8.6), so that forecast is not confirmed. Tiny Aya Fire and Water were not tested, and
Swahili is the only African language we evaluated.

Source: `earth/seeds.md`, `earth/checks.md`.

## D18. Hardware control. Confidence: Medium (one seed; a description with no pass rule)

Every number above comes from one laptop. To check that the results do not depend on it, the base model and the seed 42 plain adapter were evaluated again on a cloud GPU (one NVIDIA L4) with the same code. Strict attack success moved by at most 1.4 points in any language, and none of the eight differences was significant (Korean: base 5.2% and 5.5%, plain 19.2% and 19.9%). The answers themselves were not the same text: 63% of the base model's and 46% of the plain model's differ between the two machines, almost always after an identical start. The prompts, the batches and the library versions were the same on both machines, so what differs is the hardware (Apple MPS against NVIDIA CUDA, both in bfloat16). Rates carry over between machines; single answers do not.

Source: `gpu.md`.

## D11. 4-bit loading (NF4). Confidence: Medium for the repair (one seed, preregistered, significant); Low for what NF4 does to the fine-tuned models

Loaded in 4 bits (bitsandbytes NF4) on the GPU, SafeGrad stayed below plain fine-tuning on Korean: 11.0% against 22.7% strict attack success (-11.7 points, p < 0.001), which is the preregistered check. Quantization itself was not neutral. For the base model, English attack success rose from 5.5% to 11.0% (p = 0.002), the one change that stays significant after correcting for twelve tests; its Korean, Chinese and Swahili rates rose by 1.7 to 2.4 points, not significantly. For the plain fine-tune, English, Chinese and Swahili rose significantly (by 3.1 to 5.2 points), but there the 4-bit model also leaves its adapter unmerged where the bfloat16 model merges it, so the change cannot be put down to precision alone, and the control that would separate the two has not been run. SafeGrad moved by 1.7 points or less in every language. Memory use was not measured in a way that isolates the model, so we report no saving.

Source: `gpu.md`.

## D12. CircuitKIT across node counts. Confidence: Medium (holds at three of five sizes; seed 42; no intervals)

The stage 1 circuit comparison was made at 200 of 612 nodes. Repeated at 50, 100, 200, 300 and 400 nodes, the
finding that Korean keeps the fewest of the base model's top refusal nodes after plain fine-tuning holds at 100,
200 and 300 nodes and not at 50 or 400. At 200 nodes the top nodes recover 65% of the harmful-minus-harmless
contrast in English, 91% in Korean, 26% in Chinese and 40% in Swahili; at 50 and 100 nodes it is under a quarter
everywhere. We name the sizes where it fails and do not claim it holds across node counts.

Source: `circuit_nodes.md`.

## D9. Costs: task score. Confidence: High that there is a cost; the cause is not established

Both fine-tunes score below the base model on the 300 held-out task prompts in every seed: averaged over three seeds, English goes from 6.61 to 6.37 under plain and 6.31 under SafeGrad, Korean from 6.10 to 5.51 and 5.49, Chinese from 6.25 to 5.79 and 5.74 (1 to 7 scale, Qwen3-14B judge). SafeGrad adds little to the cost of fine-tuning itself. The drop is not explained by shorter answers. With length groups fixed in advance, answers about as long as the base model's still score lower (plain -0.39 points, -0.81 to -0.05; SafeGrad -0.67, -1.10 to -0.30), and the drop is the same size for answers under half the length. It sits in the judge's accuracy score (-0.6 in English, -1.2 in Korean, -1.1 in Chinese) and not in fluency. The fine-tuned models answer at the length of the references they were trained on, a third of the base model's length.

Source: `results.md`, `task_length.md`.

## D14. Judge validity. Confidence: a limitation, with one robustness table

Every safety number comes from one automatic guard. Against 120 answers labelled blind by the team it agreed 28/30 in English, 24/30 in Korean and 27/30 in Chinese (Cohen's kappa 0.87, 0.60 and 0.80) and missed our 80% bar in Swahili (20/27, kappa 0.49); half of each sample is answers the guard flagged, so it is a hard sample. Two other guards we tried as second judges both missed the bar, and a blind human reading of 200 Korean answers found the rise in the expected direction but not significantly (5 against 1 on 94 prompts, p = 0.22). The guard has two flagged classes, Unsafe and Controversial, and our two human checks disagree about the second: labellers marked 18 of 24 Controversial answers unsafe in the first check and 0 of 4 in the second. We therefore computed every check again counting Unsafe only, with the procedure fixed in advance. Every three-seed check still passes (Korean-minus-English gap +8.8 points, +4.7 to +13.1; SafeGrad against plain -6.6, -9.2 to -4.4); the Chinese-only training comparison does not when it is read at seed 42 alone, and does on three seeds (+4.1 points, +0.2 to +8.0).

Source: `second_judge_check.md`, `ko_drift.md`, `unsafe_only.md`.

## What changed since the brief's table (Sat 9:45 PM)

- D8 is no longer a single run: Earth holds on three seeds, on both readings.
- D7 holds after re-generation as well.
- D6: the three single-language runs are on three seeds on the laptop (row 1 in all nine runs), with a second reading after re-generation at seed 42, and were repeated on a GPU at seed 42.
- D11 and D18 are new: the hardware control and 4-bit loading ran on a GPU on Nov 22 (Hannah's jobs).
- Not in the brief: the seed 42 repair vector, added with no training to the Aya adapters, lowered Korean attack
  success by about 5 points on both readings (registered before it ran), but it made answers about twice as long,
  and its over-refusal and task score were not measured. If it is used at all, it belongs in the appendix with
  those limits (`vector_aya.md`).
- The Korean rise judged by a person (5 plain against 1 base answer unsafe on 94 prompts, p = 0.22) still does
  not pass, and both second judges missed the 80% bar. The size of every rise above is the size the guard measures.
