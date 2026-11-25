# Results paragraphs for the paper (D15, draft for Aish to edit in)

Written Nov 22. One paragraph per deliverable, D1 to D8, D11, D12 and D18, each with its confidence label and the file its
numbers come from. Every number is from `results/lir/stage2/`; the status page there
(`results/lir/stage2/README.md`) says what each result may and may not be used for. Intervals are 95%. A pooled
interval is a bootstrap over prompt families with each family averaged over the seeds: it covers the sampling of
prompts for the seeds we ran, not the spread between training seeds, which the per-seed values show. Attack
success is strict (guard says Unsafe or Controversial) unless a line says otherwise.

Labels follow the brief: High = three seeds, preregistered, significant. Medium = consistent direction, limited
seeds or sample. Low = single seed, small sample, or not significant.

## D1. Korean drift. Confidence: High

Plain LoRA fine-tuning on 1,800 harmless Bactrian-X rows raised Korean attack success from 5.2% to 19.2%, 15.5% and
18.9% at seeds 42, 1 and 2 (+14.1, +10.3 and +13.7 points, each significant by an exact McNemar test). English did
not move (+0.9 points on average, interval -1.8 to +3.8). The pooled Korean-minus-English gap is +11.8 points
(+6.8 to +16.7), and the preregistered check passes in every seed. No harmful example was in the training data.

Where Korean stands among more languages (section 14b, read Nov 23; Medium, because no person has checked the guard
in the six added languages). The same adapters answered the 291 attack families in six further languages that have
no training rows. As the guard measures it, Korean rose more than Italian, Arabic and Bengali, and its rise could not
be told apart from the rise in Thai (+11.7 points, +8.1 to +15.3), Javanese (+9.4, +5.4 to +13.5) or Vietnamese
(+8.1, +5.0 to +11.6). No language rose more than Korean. So "hardest in Korean" holds among the four languages of
the main evaluation (English, Korean, Chinese, Swahili), with Vietnamese, Thai and Javanese named beside it as not
distinguishable from Korean. It is not a statement about the ten languages, and it is not a finding that the three
rose as much as Korean. H1, Korean against English, is unchanged.

Read again after re-generation (section 14e, read Nov 23; registered after the first reading, so not blind; it
decides nothing). Every answer in the six languages that reached the 256-token cap was generated again at 1,024
tokens. Korean's rise minus each language's stayed on the same side of zero: larger than Italian, Arabic and
Bengali, not distinguishable from Vietnamese, Thai and Javanese, and none larger than Korean. Some figures
with no rule crossed zero, three of them in Bengali and one in Arabic (the status page lists them). The one this
paragraph touches: in Bengali the guard's attack success under plain is then above base (+5.6 points, +1.7 to +9.3,
against +1.6, -2.5 to +5.6, on the capped answers). We do not write that as a Bengali drift: the reading is not
blind, no person has checked the guard in Bengali, most re-generated answers are a different answer to the same
prompt, and Korean's rise is larger than Bengali's on both readings (+7.6 points, +2.4 to +12.9, after
re-generation).

Source: `seeds.md`, `drift_by_language.json`, `more_languages.md`, `more_languages_regen.md`. Figure: `ten_languages.png` (the drift figure over all ten languages).

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

Beyond Swahili (section 14b, read Nov 23; Medium: both averages were preregistered before any answer in these
languages existed and their intervals exclude zero, but the refusal label is the guard's and no person has checked
it in these languages). On the 291 attack families in Italian, Vietnamese, Arabic,
Thai, Bengali and Javanese, none of which has a training row, the refusal rate under plain fine-tuning was 4.9
points below base on average over the six languages and three seeds (-7.0 to -2.9), and SafeGrad kept 6.2 points
more refusals than plain (+4.4 to +8.0). Both checks are averages. By language the fall is clear in Vietnamese
(-16.0 points), Thai (-7.4) and Italian (-4.9) and was not detected in Javanese, Arabic or Bengali, so the sentence
is "on average over six more languages", never "in every untrained language". Limits: no person has checked the
guard in these languages; base answers reach the 256-token cap far more often than fine-tuned answers (23% to 66%
by language against 4% to 15%), and when every capped answer was generated again at 1,024 tokens (section 14e, a
second reading that is not blind and decides nothing) both averages came out the same (-4.7, -6.7 to -2.7, and
+6.3, +4.5 to +8.0); over-refusal and task score were not measured there;
and these are the Bactrian-X adapters only.

Source: `seeds.md`, `swahili_nllb_seeds.json`, `aya/seeds.md`, `more_languages.md`, `more_languages_regen.md`.

## D4. Truncation audit. Confidence: High

Attack and XSTest answers were capped at 256 new tokens, and base answers reach the cap more often than
fine-tuned ones. We generated every capped answer of base, plain and SafeGrad again at 1,024 tokens, at all three
seeds, and judged them again. The three main checks pass before and after: the Korean rise averages +12.7 points
before and +13.2 after, SafeGrad against plain is -8.4 and -8.6, and the Swahili refusal change is -9.3 and -10.0.
The same second reading was made for the runs with no Korean rows, for Tiny Aya Earth on three seeds, for the Aya
runs, for the single-language runs, for the six further untrained languages and for the mixing runs, each registered
before its answers
were generated again, and each came out the same (among figures with no rule, some crossed zero in the six
languages, most of them in Bengali, and three crossed a threshold in the mixing runs: see D1 and D10). The headline does not depend on the 256-token cap.

Source: `truncation/report.md`, `truncation/k1.md`, `earth/seeds.md`, `aya/seeds.md`, `single_language.md`,
`more_languages_regen.md`, `mixing/seeds_regen.md`.

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

## D13. The same fine-tune through AlignTune. Confidence: Medium (one run, one seed; preregistered, significant)

Run through Lexsi's AlignTune instead of our own training code, the same plain fine-tune drifted the same way. Korean
attack success rose from 5.2% to 18.6% (+13.4 points, p < 0.001), above the registered bar of 6.36 points; the plain run
from our own trainer at the same seed reached 19.2% (difference -0.7 points, p = 0.815). Chinese rose 5.2 points
(p = 0.014), English and Swahili did not move, and the Swahili refusal rate fell 10.7 points (p < 0.001), as in our own
run. Over-refusal and task score were not detectably different from our plain run (task score 5.88 against 5.94,
-0.06 points, -0.15 to +0.04). The Korean drift is not specific to our training code.

Limits to keep with it: one run at one seed. No difference from our plain run was detected in any language, but that is
not a test that the two training stacks are equivalent: they differ in library versions, in the optimizer
implementation and in batch order and adapter initialisation, and no second run through AlignTune measures its own
spread. Before training, all 1,800 rows were checked to have the token IDs of our own trainer. That needed one step of
ours: AlignTune's data path hands the trainer plain strings, which would train with no chat template, so each row is
wrapped as chat messages first. The run was made on the laptop. The size of the rise is the size the guard measures.

Note for Aish: this is the run of your section 12.3. Hannah's attempts on the GPU stopped before training, Mingyu took
it over, and it was registered again as 14d before it was queued, with the three sentences for T1 fixed there. You can
amend that entry. It ran under the deadline extension.

Source: `aligntune.md`.

## Refusal direction across seeds and training languages (not in the brief's list). Confidence: High for the Korean-English comparison (three seeds, preregistered, inside a margin fixed in advance); Medium for what it means

Stage 1 found a direction in the base model's activations that separates harmful from harmless prompts in all four languages. Across three seeds, plain fine-tuning moved Korean and English attack prompts the same distance along it, as predicted: the difference is 0.08 units (-0.08 to +0.25), against a shift of about 3.5 units and a margin of 1 unit fixed in advance. Korean's larger drift is therefore not a larger movement along this direction. The single-language runs sharpen this. Trained on English alone, Korean moved less far than English (by 1.41 units, 1.28 to 1.54), and Korean was still the language whose attack success rose; trained on Chinese alone, Korean and Chinese moved the same distance to within the margin. SafeGrad moved every language about 60% as far as plain. This concerns one direction at one layer, found in English on the base model, and no judge is involved. It does not show why Korean loses more for the same movement, and refusal rates do not follow the shift language by language, so the projection is not a readout of a fine-tuned model's refusals.

Source: `refusal_direction.md`; stage 1's `results/lir/probe.json` for the direction.

## Learning rate and epochs (section 14h; not in the brief's list). Confidence: Medium (one run and one seed per setting; preregistered; significant in all four runs)

The drift does not depend on our learning rate or number of epochs. We repeated the plain fine-tune at seed 42 with the learning rate halved and doubled, and with half and twice the epochs. Korean strict attack success rose from 5.2% to 18.2%, 21.6%, 26.1% and 16.5% (+13.1, +16.5, +21.0 and +11.3 points, each p < 0.001), each time by at least the registered bar, and Korean rose more than English at all four settings (Korean rise minus English rise +11.0 [+5.5, +16.5], +13.4 [+8.2, +18.9], +16.2 [+10.7, +21.6] and +10.0 [+4.8, +15.1]). A milder setting did not remove the task-score cost: halving the learning rate or the number of epochs left the task score below the base model's (-0.61 [-0.77, -0.44] and -0.66 [-0.82, -0.48]), and so did the two stronger settings (-0.39 [-0.52, -0.25] and -0.35 [-0.48, -0.23]).

- Limits to keep with it: one run and one seed per setting; no SafeGrad run at these settings; the LoRA rank was not varied; capped answers, nothing generated again. No setting is written as a fix, and no trend is written: we expected the weaker runs to drift less, and the run with half the epochs rose most. No test compares the four runs, and the three plain seeds with our settings rose 14.1, 10.3 and 13.7 points.
- The rule for the task score (S3) was added to the entry after the first of the four runs had started training and before any of them was evaluated.
- Source: `results/lir/stage2/sensitivity.md` and `sensitivity.json`; the dated note under section 14h of the preregistration.

## A second safety benchmark (section 14i; not in the brief's list). Confidence: Medium (three seeds, preregistered, all four checks pass; machine-translated prompts, and a guard no person has checked on them)

The drift and both repairs also show on a safety benchmark from another source. On the 200 unsafe prompts of XSTest, machine-translated into Korean, Chinese and Swahili, plain fine-tuning raised Korean strict attack success by +10.5 [+5.8, +15.3] points over base (6.0% for base; 21.0%, 14.5% and 14.0% at the three seeds), and Korean rose more than English (+8.7 [+3.5, +13.8]). Both repairs kept Korean below plain fine-tuning on this benchmark, which their refusal examples do not come from: SafeGrad by -7.2 [-10.7, -3.8], and mixing in five copies of the refusal examples by -8.5 [-12.2, -4.8].

- Limits to keep with it: the prompts are machine translations that no person has checked, and no person has checked the guard on these answers; 200 prompts; 35.5% of the base model's Korean answers reach the 256-token cap, against 3.5% to 6.5% of the plain runs'.
- After every capped answer was generated again at 1,024 tokens (section 14j: registered after the first reading, so not blind, and deciding nothing), all four checks came out the same: X1 +12.2 [+7.7, +16.8], X2 +10.3 [+5.3, +15.3], X3 -7.0 [-10.3, -3.8], X4 -8.5 [-12.2, -5.0]. The capped reading is the one with rules.
- X3 is a pooled result: SafeGrad against plain is -12.5 (p < 0.001), -4.5 (p = 0.093) and -4.5 (p = 0.064) at seeds 42, 1 and 2. Not "at every seed".
- Not written, by the entry: "generalises"; sizes compared with MultiJail; a ranking of the two repairs (five copies minus SafeGrad -1.3 [-3.8, +1.2], with no rule); anything about Chinese or Swahili beyond the table. The table's Chinese row rose +11.8 [+7.7, +16.3] under plain, and no rule covers it.
- Source: `results/lir/stage2/second_benchmark.md` and `second_benchmark.json`; `second_benchmark_regen.md` for the second reading; the dated notes under sections 14i and 14j of the preregistration.

## 4-bit loading on three seeds (section 14f; replaces the one-seed NF4 sentences). Confidence: High for P2 (three seeds, preregistered, significant in each); Medium for P1 (three seeds, preregistered; seed 42 under NF4 was known)

With the adapter loaded the same way in both precisions, loading the base weights in 4 bits (NF4) raised strict attack
success on both fine-tuned models: +2.9 points [+1.7, +4.2] for plain and +1.9 [+0.8, +3.0] for SafeGrad, over three
seeds and four languages (the base model, one run: +2.9 [+1.5, +4.6]). The repair survives it: under NF4, SafeGrad
stayed below plain on Korean in all three seeds (-11.7, -7.2, -10.7; pooled -9.9 [-13.3, -6.5]), and no smaller repair
than in bfloat16 was detected (-0.6 [-4.4, +3.3]). Merging the adapter into the weights made no detected difference at
seed 42, so the caveat of the Nov 22 reading, that the 4-bit arms also differed in merging, no longer applies
(`nf4_seeds.md`).

Limits to state with it: bitsandbytes NF4 with the adapter in bfloat16 on top (not an adapter merged and then
quantized, and not GGUF); one GPU type; guard verdicts on answers capped at 256 tokens; no task score.

## A model outside the Tiny Aya family (section 16; the first item of "with more credits"). Confidence: Medium (three seeds, preregistered, all three checks clear; one model; capped answers with no second reading)

The same training code and data on IBM Granite 4.1 3B, a model from another family, gave the same pattern. Plain
fine-tuning raised strict attack success by +10.1 points [+7.6, +12.6] over three seeds and English, Korean and Chinese
(G1). Korean again rose more than English: the Korean rise minus the English rise is +20.8 [+15.1, +26.3] (G2); Korean
went from 19.2% to about 40% while English did not move (+0.2 [-2.2, +2.7]). SafeGrad left attack success below plain
by -13.7 [-16.1, -11.4] (G3), and in Korean returned it to the base model's level (-4.1 [-8.6, +0.2] against base)
(`granite41/seeds.md`).

What has to be said with it. One model of one other family: this is not "holds across models", and no test compares
its sizes with Tiny Aya's. Granite's template has no system prompt, five Korean training rows were dropped as too long,
its card does not list Swahili (left out of the checks), and the runs were on a cloud GPU. The base model's answers
reached the 256-token cap far more often than the fine-tuned models' (Korean 73% against 22 to 38%), so every capped
answer was generated again at 1,024 tokens (section 16a): the three checks come out the same (+11.5 [+9.2, +13.9],
+24.2 [+18.9, +29.3], -13.9 [-16.3, -11.5]), and after it SafeGrad's Korean rate is level with the base model's (-0.5
[-4.2, +3.2]). A cost to state with the repair: on this model SafeGrad refuses more safe English prompts (9.6% to
between 21.6% and 25.2%; it did not on Tiny Aya). Guard verdicts only.

Two follow-ups on the same model. Trained with no Korean row at all (section 16b), Korean attack success still rose,
by +19.5 points [+14.3, +24.7] against base over three seeds, as K1 found for Tiny Aya: the Korean rise does not need
Korean training data on either model (`granite41/noko.md`). On the task score (section 16c), plain fine-tuning cost
0.47 points [0.35, 0.61] on the 1 to 7 scale, and no difference between SafeGrad and plain was detected (-0.07
[-0.15, +0.00], an interval that reaches zero by two thousandths), so the repair is not shown to be bought with a
weaker model (`granite41/task.md`).

A person's reading did not confirm it (section 16d). One author read 300 Korean answers of this model (150 prompts
drawn at random, base and plain at seed 42, arm hidden) and marked 83 base answers and 84 plain answers unsafe (exact
McNemar p = 1.00); on the same prompts the guard counts 31 and 66. The registered sentence: "On the model outside the
Tiny Aya family, a person's reading did not confirm the Korean rise: the difference did not reach significance." He
agreed with the guard on 174 of the 300 answers (58%) and marked 98 answers unsafe that the guard called Safe. So on
this model the Korean rise is the guard's alone, and this sentence goes wherever the Korean rise on Granite is stated,
in the main text and not only in the limitations. One labeller, not blind; how the marks came back is on record under
16d (`granite41/person.md`).

## Circuit-guided mixed-precision quantization (section 15, D21). Confidence: a registered null (one run per arm, preregistered; no protection claimed)

Keeping the layers the refusal circuit ranks highest at 8 bits and its lowest-ranked at 2 bits, at 4.0 bits on average,
did not protect safety, and we claim no protection. Uniform 4-bit quantization of the base model did not raise attack
success detectably (+0.2 points [-1.2, +1.5] over four languages), so there was no rise for a mixed plan to prevent and
the registered comparisons of the circuit plan with uniform 4-bit and with random plans were not read. The circuit plan
scored 1.31 points [1.10, 1.52] below uniform 4-bit on the task (4.78 against 6.09, on a 1 to 7 scale; the unquantized
model 6.35), and every random plan of the same shape scored lower still (3.69 to 4.15) (`quantization.md`).

What can be said beside it, with no rule: against random plans of the same shape the circuit plan had lower attack
success (-5.8 [-7.6, -3.9]) and a higher task score, and it had the lowest perplexity of four plans at every tier size
in the gate. On perplexity the rank test of section 15a points the same way (R1): none of 43
random plans of the same shape reached the circuit plan's perplexity, at the size of the arms (7.81 against 8.19 to
12.61; p = 0.023) or at the larger one, and the circuit's ranking turned upside down gave 17.75, worse than every
random plan. The registered sentence: "At the same 4.0 bits, the model kept a lower perplexity under the circuit plan
than under random plans of the same shape." So the ranking carries information about which layers bear fewer bits.
The limits registered with it: the test was added after the gate's table had been read, which is why the gate's three
plans are counted among the 43; no ranking that knows nothing of refusal was compared, so it does not show that the
refusal circuit in particular matters; perplexity is not safety, and R1 does not stand in for the two comparisons that
were not read
(`quantization_ranks.md`). On Global-MMLU-Lite, a benchmark with no judge
(section 15b), the picture is different: the circuit plan scored 14.1 points [10.1, 18.2] below uniform 4-bit (37.1%
against 51.2%; unquantized 54.5%), which agrees with the task score, and it was below two of the three random plans
(42.7% and 45.2%; the third 33.8%). So the circuit plan is better than random plans on perplexity and not on that benchmark, and uniform 4-bit is better than it on every measure taken
(`quantization_mmlu.md`).

Limits to state with it: the quantized weights were evaluated written out, so nothing about speed or memory; one
quantizer (optimum-quanto, no calibration) and one tier size, chosen by a gate on perplexity; the base model only.

## D9. Costs: task score. Confidence: High that there is a cost; the cause is not established

Both fine-tunes score below the base model on the 300 held-out task prompts in every seed: averaged over three seeds, English goes from 6.61 to 6.37 under plain and 6.31 under SafeGrad, Korean from 6.10 to 5.51 and 5.49, Chinese from 6.25 to 5.79 and 5.74 (1 to 7 scale, Qwen3-14B judge). SafeGrad adds little to the cost of fine-tuning itself. The drop is not explained by shorter answers. With length groups fixed in advance, answers about as long as the base model's still score lower (plain -0.39 points, -0.81 to -0.05; SafeGrad -0.67, -1.10 to -0.30), and the drop is the same size for answers under half the length. It sits in the judge's accuracy score (-0.6 in English, -1.2 in Korean, -1.1 in Chinese) and not in fluency. The fine-tuned models answer at the length of the references they were trained on, a third of the base model's length.

Source: `results.md`, `task_length.md`.

## D10. The simple fix: refusal examples mixed into the training data. Confidence: Medium (preregistered and significant; two new seeds decide, because seed 42 was seen before the rules)

We compared SafeGrad with the simplest alternative: adding the same 47 refusal examples to the training data, once (141 views over training) or as five copies (705 views, close to the 684 SafeGrad uses). Each mixing run is read against SafeGrad at the same seed on the same prompts, and seeds 1 and 2 decide. With five copies, the simple fix protected Korean better than SafeGrad: Korean strict attack success was 2.7 points lower (-5.2 to -0.5; 6.2% against 8.2% at seed 1 and 6.2% against 9.6% at seed 2). With the examples added once, no difference was detected (-0.5 points, -2.9 to +1.9). Both arms raised over-refusal of safe English prompts above SafeGrad's, by 5.2 points (+2.8 to +8.0) and 5.0 points (+2.6 to +7.6), while SafeGrad's own English over-refusal is not distinguishable from the base model's in any seed. No difference in task score was detected (+0.02 and +0.04 on the 1 to 7 scale; both intervals include zero). So SafeGrad's advantage over the simple fix is that it over-refuses less than either mixing arm. It is not better protection.

Read again after re-generation (section 14g, read Nov 23; registered after the first reading, so not blind; it decides nothing). Every answer of the six mixing runs that reached the 256-token cap was generated again at 1,024 tokens. Both checks came out the same for both arms: with five copies Korean attack success was 2.6 points below SafeGrad's (-5.0 to -0.5), with the examples once no difference was detected (-0.3 points, -2.7 to +1.9), and English over-refusal stayed 5.2 and 5.0 points above SafeGrad's. Beside it, with no rule and computed after both arms had been read: five copies left Korean attack success 2.2 points below one copy (-4.1 to -0.5, on both readings), with no detected difference in English over-refusal (+0.2, -0.8 to +1.4). That is an estimate, not a registered test of the two arms. Three other figures with no rule crossed zero or p = 0.05 between the two readings; the status page lists them, and no sentence here uses them.

Not to be written from this: that SafeGrad protects better than mixing; "holds on three seeds"; that more copies protect more (the two arms were not tested against each other); that mixing brings Korean back to the base model's level. The checks were decided on answers capped at 256 tokens. One guard.

Source: `mixing/seeds.md`, `mixing/seeds_regen.md`; `mixing/results.md` for seed 42; sections 14a and 14g of the preregistration.

## D14. Judge validity. Confidence: a limitation, with one robustness table

Every safety number comes from one automatic guard. Against 120 answers labelled blind by the team it agreed 28/30 in English, 24/30 in Korean and 27/30 in Chinese (Cohen's kappa 0.87, 0.60 and 0.80) and missed our 80% bar in Swahili (20/27, kappa 0.49); half of each sample is answers the guard flagged, so it is a hard sample. Two other guards we tried as second judges both missed the bar, and a human reading of 200 Korean answers with the arm hidden found the rise in the expected direction but not significantly (5 against 1 on 94 prompts, p = 0.22). A second reading by the same person, on the 161 Korean prompts not read before, did not reach significance either (11 against 5 on 145 prompts, p = 0.070). On the same prompts the guard's strict count is 17 against 2 in the first reading and 23 against 7 in the second, so the rise a person marked is smaller than the rise the guard marked. Of the answers the guard called Unsafe, the person marked 8 of 13 unsafe in the first reading and 12 of 21 in the second. The guard has two flagged classes, Unsafe and Controversial, and our two human checks disagree about the second: labellers marked 18 of 24 Controversial answers unsafe in the first check and 0 of 4 in the second. We therefore computed every check again counting Unsafe only, with the procedure fixed in advance. Every three-seed check still passes (Korean-minus-English gap +8.8 points, +4.7 to +13.1; SafeGrad against plain -6.6, -9.2 to -4.4); the Chinese-only training comparison does not when it is read at seed 42 alone, and does on three seeds (+4.1 points, +0.2 to +8.0).

Added Nov 25, the guard on harmless prompts (section 18b, exploratory, prompted by the outside reviews). On XSTest's safe prompts the guard labels 5.6% of the base model's Korean answers Unsafe or Controversial and 11.1% of the plain runs' (+5.5 points [+2.1, +9.1]); in English 1.6% and 5.3% (+3.7 [+1.2, +6.4]). The registered sentence: "The guard also flagged more of the fine-tuned model's Korean answers to harmless prompts, so a part of the Korean rise on the attack prompts, up to that size, may be the guard's reaction to the fine-tuned answers and not harm." Two things to write with it. The Korean rise with that change taken off is +7.2 [+2.0, +12.4], still above zero. And the change on harmless prompts is about the same in Korean and English (difference +1.7 [-2.3, +6.0]), so the Korean rise against the English rise, which is H1, is not weakened by it. On the model outside the Tiny Aya family the guard's change on harmless prompts is +13.1 [+9.6, +16.9] in Korean against +21.1 on attack prompts (net +8.0 [+1.4, +14.5], added after the tables). By run, the guard flags more of the plain answers a person called safe: 7% against 1 to 2% for base on Tiny Aya, 35% against 7% on the other model (18a). These safe prompts are written to look unsafe, and nobody has read the answers, so "may be" is all the sentence supports. Source: `review_tables.md`.

Added Nov 25, a third and last judge (section 17). After three outside reviews we tried one more: `openai/gpt-oss-safeguard-20b`, a safety model that applies a policy the user writes, given the rule our labellers were given and nothing else. It agreed with our labels on 26 of 30 English answers and on 22 of 30 in Korean and in Chinese (22 of 27 in Swahili), so it missed the bar in the two languages the headline is about and scored no run. In Korean and Chinese it flagged nothing we called safe and passed 8 answers we called unsafe. Three second judges have now failed the bar; every size is the first guard's. Source: `third_judge_check.md`.

Why the two second judges missed the bar. PolyGuard-Ministral agreed with our labels on 11, 18 and 18 of 30 answers in English, Korean and Chinese and on 17 of 27 in Swahili. In 113 of the 120 rows its three verdicts are the same: when it calls the request harmful it calls the answer harmful too, whatever the answer says, so it gives nearly the same verdict for every arm. Granite Guardian 3.3 8B agreed on 20, 20 and 22 of 30 and on 22 of 27. It is too lenient for our labels: in English, Korean and Chinese it passed 24 answers we marked unsafe and flagged 4 we marked safe, and its set-up was checked on made-up pairs first. Neither was used to test a hypothesis, and the team stopped at two judges, as recorded in the preregistration.

Source: `second_judge_check.md`, `ko_drift.md`, `ko_rest.md`, `unsafe_only.md`. How the K3 sheet came back is on record in section 14c of the preregistration.

Added Nov 25, the third reading (section 16d). The same person read 300 Korean answers of the model outside the Tiny Aya family, Granite 4.1 3B, and marked 83 base answers against 84 plain answers unsafe on 150 prompts (p = 1.00), where the guard counts 31 against 66. There he agreed with the guard on 58% of the answers (93% in K3) and marked 56% of all answers unsafe (7% in K3). None of the three readings confirmed the Korean rise at the registered level; the two on Tiny Aya pointed the guard's way, and the one on Granite shows no difference. Write it that way. Source: `granite41/person.md`; how the marks came back is on record in section 16d.

## What changed since the brief's table (Sat 9:45 PM)

- New in D1 and D3 (Nov 23, under the deadline extension): six more languages the model was never fine-tuned on (section 14b). It changes one sentence. "Korean is hit hardest" now holds only among the four languages of the main evaluation: as the guard measures it, the rise in Thai, Javanese and Vietnamese could not be told apart from Korean's, and no language rose more. Any sentence that ranks Korean first, the title and the abstract included, needs that scope and those three names. It also widens D3: refusals fell on average over six more untrained languages, and SafeGrad kept more of them.

- D1, D3 and D4 (Nov 23, 9:15 PM): the six languages were read again after every capped answer was generated again at 1,024 tokens (section 14e; not blind; it decides nothing). U1 to U3 came out the same and no sentence changes. Some figures with no rule crossed zero, three in Bengali and one in Arabic; the one the text touches is Bengali's attack success under plain, above base after re-generation (+5.6 points, +1.7 to +9.3). A sentence that says no answer in the six languages was generated again is out of date, and "no change was detected in Bengali" now needs "on the capped answers".

- D10 and D4 (Nov 23, 10:49 PM): the mixing runs were read again after every capped answer was generated again at 1,024 tokens (section 14g; not blind; it decides nothing). M1 and M2 came out the same for both arms and no sentence changes. New beside it, with no rule: five copies against one, -2.2 points in Korean (-4.1 to -0.5). A sentence that says no mixing arm was generated again is out of date.

- D13 is new (Nov 23, under the deadline extension): the plain fine-tune run through AlignTune, on the laptop (section 14d). Outcome row 1: Korean rose 13.4 points over base (p < 0.001; bar 6.36), against 14.1 for plain from our own trainer at the same seed. One run, one seed. A sentence that says the AlignTune run was not made is out of date.

- D10 is new (Nov 23, under the deadline extension): the mixing baseline on two new seeds. It changes one argument. With five copies of the refusal examples, mixing protected Korean better than SafeGrad, so SafeGrad's advantage over the simple fix is in over-refusal only. Any sentence that puts SafeGrad ahead of mixing on protection has to go.

- D8 is no longer a single run: Earth holds on three seeds, on both readings.
- D7 holds after re-generation as well.
- D6: the three single-language runs are on three seeds on the laptop (row 1 in all nine runs), with a second reading after re-generation at seed 42, and were repeated on a GPU at seed 42.
- D11 and D18 are new: the hardware control and 4-bit loading ran on a GPU on Nov 22 (Hannah's jobs).
- Not in the brief: the refusal-direction check on three seeds and on the single-language runs (R1 holds as predicted).
- Not in the brief: the seed 42 repair vector, added with no training to the Aya adapters, lowered Korean attack
  success by about 5 points on both readings (registered before it ran), but it made answers about twice as long,
  and its over-refusal and task score were not measured. If it is used at all, it belongs in the appendix with
  those limits (`vector_aya.md`).
- The Korean rise judged by a person does not pass in either of two readings (K2: 5 plain against 1 base answer unsafe on
  94 prompts, p = 0.22; K3, Nov 23, on the 161 prompts not read before: 11 against 5 on 145 prompts, p = 0.070), and
  both second judges missed the 80% bar. The size of every rise above is the size the guard measures.
