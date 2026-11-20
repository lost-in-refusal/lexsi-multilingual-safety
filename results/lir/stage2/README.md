# Stage 2 results so far

Updated Nov 20, 2:25 PM ET. Every number comes from our own runs, and the file named on each line has the detail. What each check means is fixed in `docs/lir/stage2-prereg.md`.

| Check | Result | Seeds | Main numbers | Detail |
|---|---|---:|---|---|
| H1 Korean drifts more than English | Pass in every seed | 3 | Korean strict attack success rises 14.1, 10.3 and 13.7 points. Pooled Korean-minus-English gap +11.8 [+6.8, +16.7] | `seeds.md` |
| H2 SafeGrad repairs it | Pass in every seed | 3 | Korean, SafeGrad against plain: -8.6, -7.2 and -9.3 points. Pooled -8.4 [-11.3, -5.5] | `seeds.md` |
| H3 Swahili refusals fall | Pass in every seed | 3 | -13.1, -6.2 and -8.6 points. Pooled -9.3 [-13.7, -5.2] | `seeds.md` |
| Costs (no pass rule) | Reported | 3 | Over-refusal and task score per arm; SafeGrad adds 13.7% training time on average | `results.md`, `train_time.md` |
| H4 Korean clean set | Not tested | - | The human check failed, so the set could not be cleaned | section 2 of the preregistration |
| Mixing baselines (no prediction) | Same protection as SafeGrad, more over-refusal | 1 | See below | `mixing/results.md` |
| Check A second judge against our labels | Failed with both judges | - | See below | `second_judge_check.md` |
| Check B headline under a second judge | Not run | - | Allowed only for a judge that passes Check A | `second_judge_check.md` |
| Check C Swahili with an independent translator | Pass | 1 | Corrected attack success 17.9% base, 29.2% plain, 17.2% SafeGrad. Plain over base +11.3 points [+6.5, +16.5] | `swahili_nllb.json` |
| Earth E1 and E2 | Running | 1 | | `earth/` |
| K1 Korean with no Korean training rows (added Nov 20) | Queued | 1 | | `noko.md` |
| K2 Korean rise judged by a person (added Nov 20) | Sheet out, waiting for labels | 1 | 100 prompts, base and plain | `ko_drift.md` |
| Repair vector (exploratory) | Queued | 1 | | `vector.md` |

Intervals are 95%. Results marked 1 seed are single runs.

## What to say carefully

**H2.** SafeGrad does not return Korean to base. It stays 5.5, 3.1 and 4.5 points above base, so it repairs roughly 60 to 70% of the rise.

**H3.** The Swahili refusal rate falls, but raw Swahili attack success does not rise (+1.4, -1.0 and +0.7 points, none significant). The guard missed our agreement bar in Swahili, which is why the corrected figure exists.

**H4.** Mingyu marked 12 of 30 rows of the filtered set bad on a stricter reading than stage 1, and 6 of 30 counting only wrong or off-task answers. The limit is 3. Raising the cut-off cannot work: it would flag 346 of the 600 Korean training rows with 108 spare rows to replace them. So we cannot rule out that Korean data quality contributes to the Korean drift.

**Mixing.** Seed 42, single runs. Adding the 47 refusal examples to the training data once or five times:
- Korean strict attack success 9.3% and 8.9%, against SafeGrad's 10.7%. Neither difference is significant.
- All four languages together, against SafeGrad on the same 1,164 prompts: -1.3 points (p = 0.053) and -1.1 points (p = 0.12). Not significant, so do not write that mixing beats SafeGrad.
- Task score 5.95 and 5.94, against 5.82.
- Over-refusal of safe prompts is higher than SafeGrad's: English 14.0% and 13.2% against 7.6%, Chinese 8.8% and 9.2% against 5.6%, all four significant. English is also above base (9.6%).
- So the registered sentence "the simple fix matches SafeGrad" does not apply. Mixing buys the same protection and pays for it in over-refusal; SafeGrad does not.

**Second judge.** PolyGuard-Ministral and Granite Guardian 3.3 8B both missed the 80% bar against our 120 labels. Our guard is the only one of the three that reaches it in English, Korean and Chinese (28/30, 24/30, 27/30). The labels were sampled using our guard's verdicts, so this holds for that sample only. The headline rests on our guard and the labels.

**Check C.** Report it as a direction: corrected Swahili attack success stays higher under plain than base with a translator that is not the model under test. Do not write "29% of Swahili answers are unsafe". What a read of the translations found:
- 35 of the 131 answers flagged after translation are also tagged as refusals by the guard (11 of 34 base, 15 of 63 plain, 9 of 34 SafeGrad). Many of them repeat the request and then say something harmless.
- The translator adds sentences after bare heading lines, almost only in base answers (about 9 of 125). That can only raise base, which works against the result.
- A spot read found word-level errors; one to three of plain's 63 flags rest on one, at most about 1 point.
- Counting only the guard's "Unsafe" label, 22, 38 and 19 answers are flagged after translation (base, plain, SafeGrad). Stage 1 had 14, 24 and 16 with the Tiny Aya translator.

**Swahili for all three seeds (no pass rule).** The same re-judging as Check C, in `swahili_nllb_seeds.json`. Base is 17.9%.
- Plain: 29.2%, 25.1% and 25.4% at seeds 42, 1 and 2. That is +11.3, +7.2 and +7.6 points over base, each significant.
- SafeGrad: 17.2%, 17.5% and 17.5%, level with base and 12.0, 7.6 and 7.9 points below plain of the same seed, each significant.
- The cautions under Check C apply to these numbers too. The answers of seeds 1 and 2 were translated after a change to how the translator batches its input (long pieces go in smaller batches, and output length is capped by input length); the seed 42 translations are reused as they were.

**Base Earth.** Evaluated Nov 20, before the Saturday gate. It refuses fewer Swahili attack prompts than base Global (43.3% against 50.9%), which is row C of the Earth outcome table. Its refusal set keeps 44 of the 72 held-out prompts; Global's kept 47.

**Still to come.** K1, K2, the Earth training runs, and the repair-vector evaluations.
