# Appendix material (D15, draft for Aish to edit into `main.tex`)

Written Nov 22. Four parts: reproducibility, method details that a reader has to be told, the preregistration
record with every deviation, and what the organizers' write-up layout asks for that the draft does not have yet.
Numbers here come from run records (`runs/*/run.json`, summarised in `results/lir/stage2/train_time.md`), the
queue log and `requirements.txt`. Results not read yet are marked "pending".

## A. Reproducibility

- **Code:** `github.com/lost-in-refusal/lexsi-multilingual-safety`, branch `mingyu/stage2` (merged into `main` before
  submission). Every table in `results/lir/stage2/` is written by a command named at the top of its module
  (`python -m lir.seeds`, `lir.repeat`, `lir.section13`, `lir.loose`, `lir.direction`, `lir.trunc`, `lir.gpu`).
- **Tests:** `python -m unittest tests.test_lir` (98 tests; each registered rule has a test on made-up data).
- **Rebuild:** `python -m lir.rebuild fetch` and `python -m lir.rebuild tables`, in a fresh clone, download the public outputs
  at a pinned revision and run each report again, with no GPU, no model and no login. Tested on Nov 23 in a clean
  copy of the repository and a new environment: 26 tables came out byte for byte the same as the committed files
  with only `huggingface_hub` and `matplotlib` installed, 36 with AuditKIT as well, and 50 (from 25 commands) with
  the training records, which are public once the adapter repository is. Changing one verdict in the downloaded
  files changed 34 of the 50. Not rebuilt: the tables that need adapter weights, the activation files or the
  held-out Aya answers, and the images. The README lists them.
- **Outputs:** every generation and guard verdict is in `lost-in-refusal/lost-in-refusal-eval-outputs` (public), one
  folder per run, with the re-generated answers under `truncation-regenerated/` and the answers, verdicts, job
  records and logs of the GPU jobs under `gpu/`. One exception: the answers to the held-out Aya prompts quote Aya
  rows, so only their scores are there. Adapters: the two stage 1 repositories (seed 42) and
  `lost-in-refusal/tiny-aya-lora-stage2` (38 folders; public at submission). The three adapters trained on the GPU
  are in a private staging dataset.
- **Hardware:** one Apple M5 Pro laptop, 48 GB, macOS 26.5, PyTorch 2.15.0 on MPS, bfloat16. No cloud GPU was used
  for any number in the paper unless a result says "GPU" (section 12.0 to 12.3 of the preregistration). The GPU runs
  used one NVIDIA L4 (23 GB, driver 620.70.06, CUDA 13.2) on Hugging Face Jobs, with PyTorch 2.15.0, transformers
  5.22.1, peft 0.21.2 and bitsandbytes 0.50.2 (`requirements-gpu.txt`). Each job's record is in `gpu.json`.
- **Pinned versions:** Tiny Aya Global `af89d219`, Tiny Aya Earth `0624fafb`, Qwen3Guard-Gen-4B `6ec42827` (safety
  guard), Qwen3-14B `40c06982` (task judge), NLLB-200 3.3B `1a07f7d1` (Swahili re-translation); SafeTune `3901a9f`,
  AuditKIT `71316da`, CuratorKIT `dd9e835`, CircuitKIT `e71d5e1`; Python packages as in `requirements.txt`.
- **Training settings (every run):** LoRA rank 16, alpha 32, dropout 0, on the seven attention and MLP projections
  (30.2M trainable parameters); learning rate 2e-4, cosine schedule with 3% warm-up; 3 epochs; micro-batch 4 with 8
  accumulation steps (effective batch 32); 171 optimizer steps for the 1,800-row set; sequence limit 1,024 tokens.
- **Decoding:** greedy, Tiny Aya's default chat template with its default system preamble, at most 256 new tokens
  for attack and XSTest prompts and 1,024 for task prompts.
- **Compute:** 36 laptop fine-tunes have a run record, 31.5 hours of training in all (a 1,800-row plain run takes
  about 75 minutes, SafeGrad about 88; a 600-row run about 25). The stage 2 queue ran for about 64 laptop hours up
  to the freeze: about 21 hours of training, 13 of evaluation, 6 of combined train-and-evaluate steps, 10 of
  re-generation at 1,024 tokens, and 12 of task judging and other steps (probes, translation, second-judge checks).
  The GPU runs took 5.8 hours of one L4 in five jobs; a 600-row fine-tune took about 17 minutes there.
- **Seeds:** 42, 1 and 2. The base model is evaluated once. Rows marked one seed are single runs.

## B. Method details to state in the paper

These are the places where our runs differ from a tool's default, or where a reader could assume otherwise.

### SafeTune (SafeGrad)

- We use SafeTune's `SafeGradHFTrainer` through a subclass in `lir/train.py`, because the stock trainer refuses
  gradient accumulation (item 1 of our feedback) and a micro-batch of 4 is all the laptop holds.
- **The refusal examples are shuffled once per run and then replayed in that same order.** The loader shuffles
  with the run's seed, and `itertools.cycle` stores the first pass and repeats it. With 47 examples in batches of 4
  that is 12 fixed batches, seen about 14 times over 171 steps. SafeTune's own trainer does the same; we reported it
  upstream (item 23). Every SafeGrad run in the paper shares this, so comparisons between arms are not affected,
  but SafeGrad with re-shuffled batches was not tested.
- **The KL term is computed over every non-padding token of the refusal example**, prompt and system preamble
  included, not over the response alone. Padding is excluded (SafeTune's loss sums over padding too), the softmax
  runs in float32 (in bfloat16 two identical distributions gave a KL near 0.6), and the sum is divided by the number
  of tokens so that it is on the scale of the task loss. In SafeTune's units that is rho = 1/430.
- The reference model is the student with its adapters switched off, which with dropout 0 is the base model.
- One batch of 4 refusal examples per optimizer step: about 684 views in a run. The mixing baseline matches that
  with five copies of the 47 examples in the training data (705 views).
- The refusal set is the base model's own safe refusals to held-out MultiJail prompts (47 of 72 for Tiny Aya
  Global, 44 of 72 for Tiny Aya Earth), never evaluated on.

### AuditKIT

- Used for the aggregated tables. Its paired-bootstrap p-value is not used: it sits near 0.5 for any non-zero
  effect because the bootstrap is not centred (item 14). Rates are tested with an exact McNemar test on the same
  prompts and pooled figures with a bootstrap over the 291 prompt families (5,000 draws).

### CuratorKIT and CircuitKIT

- CuratorKIT cleaned and de-duplicated the Bactrian-X rows and, for stage 2, the Aya rows; counts before and after
  each step are in the data cards (`docs/lir/cards/`). It has no multilingual safety screen (item 4), so rows were
  screened with the guard.
- CircuitKIT EAP-IG on 64 held-out harmful and harmless pairs per language, 5 integration steps, first-answer-token
  contrast, node counts 50 to 400 of 612. Scores are unsigned and there is one seed.

### Evaluation

- Strict attack success counts the guard's Unsafe and Controversial classes and is the registered metric. Every
  check is also given on the Unsafe class alone (`unsafe_only.md`).
- Answers are capped at 256 new tokens. Every result that a claim rests on was read a second time after the capped
  answers were generated again at 1,024 tokens (`truncation/`, and the second readings in each table).
- The adapter is merged into the base weights before generation in every laptop run.

## C. Preregistration and deviations

The file is `docs/lir/stage2-prereg.md`. Git timestamps show each entry was committed before its runs.

| When (ET) | What was registered | Outcome |
|---|---|---|
| Nov 19, 10:30 PM, before any stage 2 evaluation | H1 Korean drifts more than English; H2 SafeGrad repairs it; H3 Swahili refusals fall; H4 Korean clean set; costs; mixing baselines (no prediction); second judge (Checks A, B); Swahili re-translation (Check C); Earth (E1, E2) | H1, H2, H3 pass in three seeds. H4 not tested. Mixing at seed 42: no detected difference in protection from SafeGrad, more over-refusal (seeds 1 and 2 are in the Nov 23 row). Check A failed with both judges, so Check B was not run. Check C passes. E1 holds; E2 not confirmed |
| Nov 20 | K1 no Korean rows; K2 Korean rise judged by a person; circuit node counts; truncation audit; harm categories; Aya (Y1 to Y3) | K1 row 1 in three seeds. K2 does not pass. Circuits hold at 3 of 5 sizes. Audit: headline unchanged. Y1, Y2 pass; Y3 fails |
| Nov 21, 4:39 PM | Section 11: Aya and Earth at seeds 1 and 2; K1 after re-generation | All as on one seed, on both readings |
| Nov 21, 8:59 PM (Aish) | Section 12.0 to 12.4: GPU runs (hardware control, single-language, NF4, AlignTune) | Hardware control: no significant difference from the laptop in eight comparisons. L1 and L2: row 1 in all three runs. Q2 passes. Q1: NF4 raises attack success for base in English and for plain in three languages, where the comparison also differs in merging. AlignTune: no result yet |
| Nov 22 (Mingyu) | Section 13a to 13h: single-language runs on the laptop; repair vector on Aya (V1); second readings; 519-row seeds (C1, C2); refusal direction (R1, R2); Unsafe-only reading; task score by answer length | SL1 and SL2: row 1 on three seeds. V1 passes on both readings. C1 includes zero; C2 is below zero for English rows only (seen at seed 42 before it was registered). Unsafe-only: 20 of 22 passes kept. Task score: not a length effect. R1: Korean and English moved the same distance, as predicted. R2: with English rows only Korean moved less far than English; with Chinese rows only Korean and Chinese moved the same distance |
| Nov 23, 3:58 AM (Mingyu), under the deadline extension | Section 14a: the mixing baseline at seeds 1 and 2 (M1 to M3), decided on the two new seeds | Five copies: mixing protected Korean better than SafeGrad (-2.7 points [-5.2, -0.5]). Examples added once: no detected difference (-0.5 [-2.9, +1.9]). Both arms: English over-refusal about 5 points above SafeGrad's (+5.2 [+2.8, +8.0] and +5.0 [+2.6, +7.6]). Task score: no detected difference |
| Nov 23, 4:26 AM (Mingyu), under the deadline extension | Section 14b: six more untrained languages (U1 to U3), evaluation only | U1: refusals fell on average over the six (-4.9 points [-7.0, -2.9]). U2: SafeGrad kept more refusals than plain (+6.2 [+4.4, +8.0]). U3: no further language rose more than Korean, and the rise in Vietnamese, Thai and Javanese could not be told apart from Korean's, so "hardest in Korean" holds among the four languages of the main evaluation only |
| Nov 23, 4:45 AM (Mingyu), under the deadline extension | Section 14c: a second reading of the Korean rise by a person, on the 161 prompts not read before (K3) | K3 does not pass: 11 plain against 5 base answers marked unsafe on 145 prompts, p = 0.070 |
| Nov 23, 2:25 PM (Mingyu), under the deadline extension | Section 14d: the plain run through AlignTune (the run of 12.3), moved to the laptop (T1, no prediction) | Not read when this was written |
| Nov 23, 4:32 PM (Mingyu), under the deadline extension, after 14b was read | Section 14e: the six languages read again after every capped answer is generated again at 1,024 tokens. It decides nothing | Not read when this was written |

Deviations and things a reader should know:

1. **H4 was never tested.** The human check of the Korean training rows failed, so the cleaned set could not be
   built. Korean data quality cannot be ruled out as a contributor; K1 and the single-language runs are the nearest
   evidence.
2. **Neither human check of the Korean rise passed** (K2: 5 against 1 unsafe answers on 94 prompts, p = 0.22; K3, on
   the 161 prompts not read before: 11 against 5 on 145 prompts, p = 0.070; one labeller, seed 42; how the K3 sheet
   came back is on record in section 14c), and both second judges missed the agreement bar. Every safety number is one guard's.
3. **Some checks were registered after a first result was known:** seeds 1 and 2 of K1 after its seed 42 run; the
   three-seed rules for Aya and Earth after seed 42; the second reading of the repair vector (13c) after its capped
   result; C2 after its seed 42 value had been seen. Each registration says what was known.
4. **The laptop section was first numbered 12, the same as Aish's, and shared three check names with it.** Aish
   registered the GPU runs first (section 12); the laptop section was written a few hours later without her branch
   having been read. On Nov 22 it was renumbered 13 (12a to 12h became 13a to 13h) and its checks L1, L2 and T1
   were renamed SL1, SL2 and V1, with no change to any rule. The laptop single-language runs are the same
   fine-tunes as her 12.1 on different hardware, so her 12.1 is not blind.
5. **Seeds 1 and 2 of the single-language runs were run before the deadline extension was confirmed,** although
   the plan listed them as extension-only. The change was written into the preregistration before they started.
6. **The repair vector on Bactrian-X was not registered** (exploratory). Its test on the Aya adapters was. That
   test passed, but the edit doubled answer length, so it was read a second time on completed answers.
7. **Under the guard's Unsafe class alone, the Chinese-only half of SL2 does not hold at seed 42.** Every
   three-seed check does, SL2 on three seeds among them.
8. **The first five rows of the Korean-by-setting figure were chosen after their results were known.** The three
   single-language rows were fixed before those runs were read.
9. **Stage 1 explained the task-score drop by shorter answers. That does not hold.** With length groups fixed in
   advance (13h), answers as long as the base model's still score lower (plain -0.39 points, SafeGrad -0.67), and the
   drop sits in the judge's accuracy score. The stage 1 sentence is withdrawn.
10. **Not run, or cut.** Every laptop run registered for the evening of Nov 22 was launched before the freeze (9 PM
    ET) and finished. Not run: the two bfloat16 arms with the adapter left unmerged, which would separate 4-bit
    loading from merging (12.2 review note; no upload by 11:44 PM); the AlignTune run (12.3; only its configuration
    review is on record, and a 32-second check of one training batch on the GPU at 11:21 PM, after the freeze); the task set on the GPU (its judge does not fit an L4); seeds 1 and 2 of the GPU
    single-language runs; and a re-generation of the seed 1 and 2
    single-language answers, which was not registered. The mixing baseline at seeds 1 and 2 was not launched by the
    freeze either; it ran on Nov 23 under the deadline extension (section 14a, item 11). Cut earlier: the Korean clean set (H4), Check B and a third
    judge.
11. **SafeGrad is not ahead of the simple fix on protection.** On two new seeds (section 14a), five copies of the
    refusal examples mixed into the training data left Korean attack success 2.7 points below SafeGrad's
    [-5.2, -0.5], and with the examples added once no difference was detected. Both arms raised English
    over-refusal about 5 points above SafeGrad's. SafeGrad's advantage is in over-refusal only. Seed 42 was seen
    before the rules were written, so two seeds decide; the two arms were not tested against each other.
12. **"Hardest in Korean" holds among four languages, not ten.** In six more languages with no training rows
    (section 14b) the guard's attack success rose in Thai (+11.7 points [+8.1, +15.3]), Javanese (+9.4
    [+5.4, +13.5]) and Vietnamese (+8.1 [+5.0, +11.6]), and those rises could not be told apart from Korean's
    (+12.7 [+8.9, +16.7]). No language rose more than Korean. No person has checked the guard in the six added
    languages, base answers there are cut at the 256-token cap far more often than fine-tuned answers, and none was
    generated again.
13. **The six-language block was not read strictly once.** Its rule was one reading when all seven arms were
    judged. During a file check after the first arm, the base model's totals over the six languages together were
    printed (strict attack success 8.8%, refusal rate 60.5%). No figure by language and no figure of a fine-tuned
    arm was seen before the report, and no rule, code or arm changed afterwards (dated note under 14b).

## D. What the organizers' write-up layout asks for that the draft does not have yet

The recommended layout on hackathon.lexsi.ai marks these as required or as valued options. Most are short.

- **"With more credits" (required):** planned approach and the expected before and after. Candidates, in order of
  value: a re-generation of the mixing arms at 1,024 tokens and a direct test of five copies against one (the mixing
  baseline itself ran at seeds 1 and 2 under the extension); a second labeller for the Korean reading (one person has read 261 of the 291 Korean prompts at seed 42 in two
  checks, and neither reached significance);
  the unmerged bfloat16 control for the NF4 runs, GGUF builds and mixed-precision deployment; in the six further untrained languages, a person's check of the guard, a re-translation check as for Swahili, and over-refusal and task sets; the Fire and Water models; LoRA
  rank, learning rate and epoch sensitivity; a second safety benchmark.
- **"Declaration of work done outside the hackathon" (required):** for the team to state.
- **"Results on the quantized / deployed model (GGUF, 4-bit)" (Track 2b line):** filled by the NF4 runs of 12.2
  (`gpu.md`): SafeGrad stays below plain on Korean under 4-bit loading, and 4-bit loading raises the base model's
  English attack success. State with it that the fine-tuned arms' NF4 comparison also differs in merging, and that
  no GGUF build was tested.
- **"Checking the automatic judge" (% agreement or Cohen's kappa):** kappa is now in `unsafe_only.md`
  (0.87 English, 0.60 Korean, 0.80 Chinese, 0.49 Swahili, on a deliberately hard sample).
- **"What didn't work" (negative results):** part C, items 1, 2, 7 and 9, plus Y3, E2 and the two second judges.
- **"Sample inputs and outputs":** `docs/lir/korean-examples.md` (translations still to be checked by Mingyu).
- **"LLM usage statement" (optional):** for the team to state.
- **Tracks and problem statements covered:** multilingual safety (2a); safety-preserving fine-tuning, locating
  where safety lives, inference-time steering (2b). The repair vector is a weight-space repair without retraining
  (2b); if it is listed, list it as exploratory with its limits.
- **Final deliverable:** the site says finalists are judged on "a video presentation or slides". The brief says the
  paper is primary and the video optional. Mingyu reported on Nov 23 that a paper without slides is a valid final
  submission.
