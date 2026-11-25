### The model outside the Tiny Aya family trained with no Korean row (section 16b: N1)

Plain LoRA on `ibm-granite/granite-4.1-3b` at seeds 42, 1 and 2 on `bactrian_train_noko.jsonl`, the training file of K1 (section 6): English and Chinese rows and no Korean row. Trained and evaluated as the runs of section 16, on the same GPU type, and read against the base run of section 16. Strict attack success on 291 prompts per language; a pooled figure averages each prompt family over the seeds, then bootstraps the families (95% interval).

**N1. On the model outside the Tiny Aya family, Korean attack success rose without a Korean training row.** Korean, the run with no Korean row minus base: +19.5 [+14.3, +24.7] points.

With no rule. Each language against base, seeds pooled: en +3.2 [+0.3, +6.3], ko +19.5 [+14.3, +24.7], zh +8.9 [+5.0, +13.1], sw -0.0 [-4.0, +4.2]. Korean against the usual plain run of the same seed: -1.6 [-5.0, +1.8].

| Run | en | ko | zh | sw |
|---|---|---|---|---|
| `granite41-noko-s42` | 6.5 to 9.3 (+2.7) | 19.2 to 38.5 (+19.2*) | 12.0 to 21.3 (+9.3*) | 23.4 to 21.6 (-1.7) |
| `granite41-noko-s1` | 6.5 to 9.3 (+2.7) | 19.2 to 38.1 (+18.9*) | 12.0 to 20.3 (+8.2*) | 23.4 to 26.1 (+2.7) |
| `granite41-noko-s2` | 6.5 to 10.7 (+4.1*) | 19.2 to 39.5 (+20.3*) | 12.0 to 21.3 (+9.3*) | 23.4 to 22.3 (-1.0) |

Rows trained: `granite41-noko-s42` 1,748 (0 over the length limit), `granite41-noko-s1` 1,748 (0 over the length limit), `granite41-noko-s2` 1,748 (0 over the length limit).

The same limits as section 16: one model, answers capped at 256 tokens, the guard's verdicts.
