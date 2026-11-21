# Stage 2 results so far

Updated Nov 21, 4:45 AM ET. Every number comes from our own runs, and the file named on each line has the detail. What each check means is fixed in `docs/lir/stage2-prereg.md`.

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
| Earth E1 and E2 | E1 holds. E2 not confirmed | 1 | Korean strict attack success 10.7% base, 21.3% plain, 16.5% SafeGrad. Swahili refusals fall 6.9 points, against 13.1 on Global | `earth/checks.md` |
| K1 Korean with no Korean training rows (added Nov 20) | Outcome row 1 in all three seeds: Korean rows are not needed for a drift of this size | 3 of 3 | Korean strict attack success 18.2%, 18.6% and 18.2% with no Korean rows in training (seeds 42, 1, 2), 19.2%, 15.5% and 18.9% with them, 5.2% base | `noko.md` |
| K2 Korean rise judged by a person (added Nov 20) | Does not pass | 1 | 5 plain answers against 1 base answer marked unsafe on 94 prompts, p = 0.22 | `ko_drift.md` |
| Repair vector (exploratory, not registered) | The seed 42 edit lowers Korean attack success at seeds 1 and 2 with no training | 2 | Korean 15.5% to 10.3% and 18.9% to 11.0%. SafeGrad trained at those seeds reaches 8.2% and 9.6% | `vector.md` |
| Circuit comparison at five node counts (added Nov 20) | Holds at 100, 200 and 300 nodes, not at 50 or 400 | 1 | Korean keeps the fewest base nodes under plain at three of five sizes. At 200 nodes the top nodes recover 65% of the contrast in English, 91% in Korean, 26% in Chinese, 40% in Swahili | `circuit_nodes.md` |
| Truncation audit (added Nov 20) | Seed 42: the headline does not depend on the 256-token cap. Seeds 1 and 2 and Earth still running | 1 of 3 | Korean rise +14.1 points on capped answers, +14.8 after every capped answer was generated again at 1,024 tokens. SafeGrad against plain -8.6 and -8.9 | `truncation/report.md` |
| Harm categories (added Nov 20) | Korean rises in all 8 categories with 20 or more prompts | 3 | Rises from +3.7 to +21.0 points | `harm_categories.md`, `harm_categories.png` |
| Aya as a second training dataset (added Nov 20) | Registered and queued | 1 | About 190 rows per language, because Aya has 309 usable Korean rows | section 10 of the preregistration |

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

**Earth.** Base, plain and SafeGrad on Tiny Aya Earth at seed 42, one run each, with the same data and settings as Global. SafeGrad used Earth's own refusal set, which keeps 44 of the 72 held-out prompts (Global's kept 47). No person has read Earth's answers, so every number here is the guard's.
- E1 holds. Korean strict attack success is 10.7% for base, 21.3% for plain (+10.7 points, p < 0.001) and 16.5% for SafeGrad (-4.8 against plain, p = 0.024).
- SafeGrad is below plain and still above base, by 5.8 points (p = 0.008). It takes back 4.8 of the 10.7 points, under half; on Global it took back roughly 60 to 70%. Do not write that SafeGrad repairs most of the rise on Earth. `earth/results.md` shows SafeGrad's change against base and `earth/checks.md` shows it against plain, so the two files print different numbers in that row.
- E2 is not confirmed. We forecast a larger Swahili refusal drop than Global's and this run shows a smaller one: Earth's rate falls from 43.3% to 36.4% (-6.9 points, p = 0.006), and Global's fell 13.1 points. The registered sentence is "the spillover into an untrained language appears on both models".
- Do not write "refuted": that row is for no significant drop. Do not write that Earth drifts less in Swahili either. It is one run per model, and the interval for the difference, +6.2 points [+0.7, +12.0], counts variation over prompts, not over seeds, and barely clears zero.
- Row C applies: base Earth refuses fewer Swahili attack prompts than base Global (43.3% against 50.9%), and the table says to lead with whether SafeGrad repairs it. The tests for that were added after the runs and are not registered. SafeGrad brings Earth's Swahili refusal rate to 45.0%: 8.6 points above plain (p < 0.001), level with Earth's own base (+1.7, p = 0.50), and still 5.8 points below base Global (p = 0.008). So SafeGrad takes back what fine-tuning cost. It does not close the gap Earth starts with.
- Do not cite Earth's Swahili attack success (7.6%, 6.5% and 5.5%). The guard missed our agreement bar in Swahili, and Check C's translation step was not run on Earth.
- Counting only the guard's Unsafe class (not registered): Korean 6.9% base, 16.2% plain, 9.3% SafeGrad. This reading makes the repair look larger, so give it beside the strict numbers and not in their place. K2 was a reading of Global's answers and says nothing about Earth's.
- Other languages: plain moves English by +1.0 and Chinese by +2.4 points, neither significant. SafeGrad against plain is -3.1 in English (p = 0.012) and -4.5 in Chinese (p < 0.001).
- Costs follow Global's pattern. Task score falls under both arms: Korean 6.05 for base, 5.34 for plain and 5.39 for SafeGrad (Global: 6.10, 5.59 and 5.40). Over-refusal of safe prompts averages 7.6% for base, 4.9% for plain and 7.2% for SafeGrad.
- Both runs trained at clean commits, and no training or evaluation code changed between the three Earth arms. Plain took 74 minutes and SafeGrad 88.
- E1 has not been through the truncation check yet. Half of base Earth's Korean attack answers reach the 256-token cap, against 6% for Earth plain and 18% for Earth SafeGrad, and the repair margin is thin (p = 0.024). The Earth arms are queued for re-generation at 1,024 tokens after the Global seeds, and E1 will be read again. Until then, treat E1 as measured on capped answers.

**K1.** Plain fine-tuning on English and Chinese rows only (1,747 rows) at seeds 42, 1 and 2, judged on the attack set. No row of the training file contains a Korean character. Seeds 1 and 2 were registered after the seed 42 result and before they ran.
- Korean strict attack success is 18.2%, 18.6% and 18.2% (seeds 42, 1, 2) against 5.2% for base: +13.1, +13.4 and +13.1 points, each p < 0.001, against a bar of 6.36. That is outcome row 1 in every seed. Pooled over the seeds the rise is +13.2 points [+9.4, +17.2].
- Against plain of the same seed on the usual set, which has 600 Korean rows: -1.0, +3.1 and -0.7 points (p = 0.74, 0.15, 0.87). Pooled: +0.5 points [-2.3, +3.3]. Taking Korean out of training made no detected difference to the Korean drift. That is not proof of no difference; the interval allows about 2 points less to 3 points more.
- The registered sentence, now a three-seed result, is "Korean training rows are not needed for a drift of this size". Write that, not "Korean data quality plays no role". English and Chinese each got 46% more rows here, so drift spreading from them may be larger than in the original runs, and no outcome row rules out that data quality contributed there.
- What it shows: Korean drifts whether or not it is in the training data, in all three seeds. H4 stays untested; K1 is the nearest evidence we have on the question behind it.
- After K2, the size of this rise is the guard's size. A person counted a smaller Korean rise than the guard on the usual plain run, and nobody has read these runs' answers.
- The truncation audit does not re-generate these arms, so every K1 figure is on answers capped at 256 tokens. The cap count covers seed 42 so far: 10% of that arm's Korean answers reach the cap against 34% for base, and 38 of its 53 Korean strict flags are on complete answers. Seeds 1 and 2 are counted when the audit's count is run again.
- Other languages against base (seeds 42, 1, 2): English +2.7, +3.1 and +4.5 points (significant at seed 2 only); Chinese +5.5, +6.5 and +10.0 (significant in each); Swahili strict attack success +4.5, +3.4 and +5.8 (significant at seeds 42 and 2); Swahili refusals -14.8, -10.0 and -12.7 points (each p < 0.001).
- Seen after the runs, not registered: against plain of the same seed, Swahili strict attack success is higher with no Korean rows (+3.1, +4.5, +5.2 points; p = 0.064, 0.002, 0.001) and Chinese is higher at seed 2 (+5.5 points, p = 0.002). That fits the extra English and Chinese rows, but the tests are not corrected for their number and the guard missed our agreement bar in Swahili. Do not build on it.
- Counting only the guard's Unsafe class (not part of the rule): Korean 13.4%, 13.1% and 12.0% against 2.7% for base, and 14.1%, 11.0% and 13.1% for plain on the usual set.

**K2.** Mingyu read 200 Korean answers (100 prompts, base and plain at seed 42, arm hidden) and marked 8 unsafe, 186 safe and 6 unsure. Every label is his own reading.
- The registered test fails: 1 base answer against 5 plain answers on the 94 prompts with two clear labels, p = 0.22. The two side readings: unsure counted as safe, 1 against 7 (p = 0.070); unsure counted as unsafe, 3 against 11 (p = 0.021).
- So do not write that a person confirmed the Korean rise. H1 stays a claim measured by the guard. The direction is the same in all three readings.
- It did not fail for lack of prompts. On the same 100 prompts the guard's strict count is 2 for base and 17 for plain, a rise of 15 answers, which would pass easily. Mingyu's rise is 4 answers on 94 prompts (6 on 100 with unsure counted as safe). K2 says the rise a person sees is smaller than the guard's.
- Of the guard's 19 flags he marked 8 unsafe, 7 safe and 4 unsure. Of the 15 flags with a clear label he confirmed 8: 8 of 11 in the guard's Unsafe class and 0 of 4 in its Controversial class. The guard passed no answer he marked unsafe. Agreement over all answers is 187 of 194, but 179 of those are answers both called safe.
- Already in `results.md` as "loose" (the guard's Unsafe class only, all 291 Korean prompts): 2.7% for base; 14.1%, 11.0% and 13.1% for plain at seeds 42, 1 and 2; 6.2%, 5.5% and 6.5% for SafeGrad. Pointing to it here is a choice made after K2, so it is not a registered check. It shows the Korean rise and the repair without the Controversial class.
- One labeller, who knows the expected direction and can often guess the arm from the answer's length.
- The first scoring matched both answers to a prompt to one guard verdict. That is fixed and did not touch the test. The joins for the 120-label set were checked for the same error: every id appears once there, so Check A stands.

**Circuit node counts.** The stage 1 comparison at 200 nodes, repeated at 50, 100, 200, 300 and 400 of the 612 nodes (seed 42). The sizes 50, 200 and 300 were already known; 100 and 400 are new.
- "Under plain, Korean keeps the fewest base nodes" holds at 100, 200 and 300. It fails at 50, where Korean and Chinese both keep 39, and at 400, where Korean keeps 344 and Chinese and Swahili 340.
- "SafeGrad keeps at least as many base nodes as plain in every language" holds at 100, 200, 300 and 400. It fails at 50, where English keeps 45 under SafeGrad and 46 under plain.
- So do not write "at every circuit size". Write the sizes. A sentence that is safe: "At 100, 200 and 300 nodes Korean keeps the fewest of the base model's top nodes after plain fine-tuning; at 50 it ties with Chinese, and at 400 Chinese and Swahili keep slightly fewer."
- Patching test, now run in all four languages on the base model: the share of the harmful-minus-harmless contrast that the top nodes recover on 64 held-out pairs, at 50 / 100 / 200 / 300 / 400 nodes. English 3 / 18 / 65 / 89 / 97%. Korean 0 / 24 / 91 / 100 / 100%. Chinese 0 / 6 / 26 / 60 / 88%. Swahili 8 / 7 / 40 / 100 / 100%. No pass rule.
- The tool clips the share to 0% to 100%. Before clipping, Korean is 111% at 300 and Swahili 116% at 400: the top nodes alone score above the whole model there. The table gives every unclipped figure.
- How to weigh the sizes: at 50 nodes the top set recovers 8% or less in every language, so the two failures at 50 are at a size that holds almost none of the mechanism. At 100 it is under a quarter, so the two passes at 100 carry the same discount as the failures at 50. The sizes that hold most of the contrast are 200, 300 and 400, and the Korean finding holds at two of those three. The failure of the Korean finding at 400 is different. That size recovers 88% to 100%, so it is a real exception; what limits it is that two random sets would already share 261 of 400 nodes.
- One node count is not the same share of the mechanism in each language: at 200 nodes it is 91% in Korean and 26% in Chinese. So the Chinese column of the 200-node comparison rests on a quarter of the contrast. Do not write "Korean refusal is more concentrated"; each language has its own pair of first tokens, and that may be what differs.
- Limits of the patching test: base model only, seed 42, the same 64 held-out pairs in every language, one token pair per language. The shares have no interval. Swahili goes from 8% at 50 nodes to 7% at 100, which shows that a difference of a few points is noise.

**Truncation.** Every attack and XSTest answer of base, plain and SafeGrad at seed 42 that reached the 256-token cap was generated again with a cap of 1,024 tokens and judged again by the same guard. Seeds 1 and 2 are running now and the Earth arms follow.
- Result at seed 42: outcome row 1, "the headline does not depend on the 256-token cap". Korean strict attack success before and after: base 5.2% and 4.5%, plain 19.2% and 19.2%, SafeGrad 10.7% and 10.3%. The Korean rise is +14.1 points before and +14.8 after. The change is +0.7 points [+0.0, +1.7], unchanged within noise. H1, H2 and H3 pass at this seed both ways.
- The worry was that base's cut answers would turn unsafe once they finished, which would shrink the rise. They did not. Base falls a little in every language: English 6.9% to 5.5%, Korean 5.2% to 4.5%, Chinese 8.6% to 7.9%, Swahili 6.2% to 3.8%.
- The repair is unchanged: SafeGrad against plain on Korean -8.6 points before and -8.9 after. Swahili refusals under plain fall 13.1 and 13.4 points. XSTest over-refusal moves by under one point everywhere.
- A sentence that is safe: "Generating every capped answer again at 1,024 tokens left the Korean rise at seed 42 unchanged (+14.1 and +14.8 points)." Say seed 42 until seeds 1 and 2 are in. The three-seed numbers in `seeds.md` are still on capped answers; this audit says the cap did not produce them at seed 42.
- Caveat to keep with it: most new answers are not the old answer continued. The batch an answer is generated in depends on the cap, and in about three quarters of cases the model gave a different answer to the same prompt (for base, 93 of 355 start with the original). So "after" mixes a cut answer finishing with a different answer to the same prompt.
- Both parts point the same way. Where the new answer is the old one continued (148 attack answers over the three arms), no verdict went from safe to flagged and 4 went from flagged to safe. Where it came out different (445), 7 went to flagged and 19 to safe. On the prompts where every arm's answer was complete or continued (210 of the 291 Korean prompts) the Korean rise is +11.4 points before and after. These two counts of verdict changes were made after the run.
- Share of attack answers that reach the 256-token cap at seed 42: base 24% English, 34% Korean, 29% Chinese, 36% Swahili; plain 2%, 8%, 5%, 19%; SafeGrad 9%, 8%, 9%, 21%. XSTest is higher for every arm (base 41 to 52%). At 1,024 tokens, 6, 9 and 8 attack answers of base, plain and SafeGrad still reach the cap.
- The comparison of verdicts on cut and complete answers in `truncation/counts.md` is a description, not a test. Refusals are short and finish, so the two groups differ in more than the cap.
- Not generated again: the mixing, K1 and repair-vector arms. They are counted only.

**Harm categories.** Strict attack success by MultiJail category, averaged over the three seeds. A prompt with several tags counts in each.
- Korean rises under plain in all eight categories with 20 or more prompts, from +3.7 points (Theft) to +21.0 (Discrimination & injustice). The interval is above zero in six of the eight. SafeGrad is below plain in all eight.
- This is a description. Do not write that one category drifts more than another: cells are small and no test was run between categories. The ten categories with fewer than 20 prompts are listed in the table and not interpreted.
- The figure shows the change against base under plain and under SafeGrad, for Korean and for the four languages together.

**Aya.** Not run yet. The Aya Dataset has 361 Korean rows, 309 after the filters, so the plan's 600 rows per language cannot be met. The registered rule: 100 rows per language held out as the Aya task set, and the same number of training rows in English, Korean and Chinese, about 190 each. A plain run on Bactrian-X cut to the same size is the control, so a weaker drift on Aya can be told apart from a smaller fine-tune. Tiny Aya's model card names no training data, and Cohere built both the model and the dataset, so the model may have seen these rows.

**Repair vector (exploratory, not registered).** The difference between the seed 42 SafeGrad and plain adapters, added to the plain adapters of seeds 1 and 2. No training, and nothing was tuned on the test prompts. Single runs.
- Korean strict attack success falls from 15.5% to 10.3% at seed 1 (-5.2 points, p = 0.008) and from 18.9% to 11.0% at seed 2 (-7.9, p < 0.001). SafeGrad trained at those seeds reaches 8.2% and 9.6%, so the edit gives 71% and 85% of the drop that training gave.
- Edited against SafeGrad: +2.1 points (p = 0.15) and +1.4 (p = 0.39). That is no detected difference, not proof that the two are equal. Do not write that the edit matches SafeGrad.
- Like SafeGrad, it does not return Korean to base: it stays 5.2 and 5.8 points above base, both significant.
- It carries SafeGrad's other effects too. The Swahili refusal rate goes from 44.7% to 56.7% and from 42.3% to 57.0%, above base (50.9%). English over-refusal of safe prompts is 12.0% and 11.6%, against 9.6% for base and 6.0% for plain; the difference from base is not significant. Task score is level with plain (5.81 and 5.82 against 5.85 and 5.88).
- What it suggests: the repair SafeGrad learned at one seed is a change to the weights that carries over to a model trained at another seed. That is one model, one dataset, one vector and two seeds, and it was added after we had seen the seed results. Treat it as a lead for future work, not as a finding to rest a claim on.

**On Hugging Face.** Uploaded Nov 20, 9:42 PM ET to `lost-in-refusal/lost-in-refusal-eval-outputs` (public): the outputs of seeds 1 and 2, the two mixing runs, the run with no Korean rows at seed 42 and the three Earth arms, the Swahili re-translations, and the tables in this folder. Every generation row has its `truncated` flag. The seed 42 outputs were already there. The two repair-vector arms and their tables were added at 10:30 PM, and K1 at seeds 1 and 2 with the three-seed K1 table on Nov 21 at 1:30 AM. Not uploaded yet: the re-generated answers and the Aya runs, which are still running or queued.

**Still to come, in queue order.** Truncation re-generation at seeds 1 and 2; Aya screening, two Aya runs and the size control; the Aya task set for five checkpoints; the Earth arms generated again at 1,024 tokens. The queue should be through by about noon Saturday.
