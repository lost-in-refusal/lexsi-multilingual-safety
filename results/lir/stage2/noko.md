### K1: plain fine-tuning with no Korean training rows, strict attack success (%)

Plain LoRA on 874 English and 874 Chinese rows. The trainer dropped 1 of them as over its length limit, so 1,747 rows were trained; the usual runs dropped none. Seed 42 was added to the preregistration on Nov 20 before it ran. Seeds 1 and 2 were added the same day, after the seed 42 result and before they ran.

Korean, by seed:

| Seed | base | plain, usual set | plain, no Korean rows | rise over base | against plain of the same seed | outcome row |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 5.2 | 19.2 | 18.2 | +13.1* (p < 0.001) | -1.0 (p = 0.743) | 1 |
| 1 | 5.2 | 15.5 | 18.6 | +13.4* (p < 0.001) | +3.1 (p = 0.150) | 1 |
| 2 | 5.2 | 18.9 | 18.2 | +13.1* (p < 0.001) | -0.7 (p = 0.871) | 1 |

The bar is 6.36 points. * = p < 0.05, exact McNemar on the same prompts.

**Across the three seeds: Korean training rows are not needed for a drift of this size, in all three seeds.**

Pooled over the three seeds (no pass rule): Korean rise over base +13.2 points [+9.4, +17.2]; against plain of the same seed +0.5 points [-2.3, +3.3].

All four languages with no Korean rows in training, change against base in parentheses:

| Seed | en | ko | zh | sw | macro |
|---:|---:|---:|---:|---:|---:|
| base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| 42 | 9.6 (+2.7) | 18.2 (+13.1*) | 14.1 (+5.5*) | 10.7 (+4.5*) | 13.1 |
| 1 | 10.0 (+3.1) | 18.6 (+13.4*) | 15.1 (+6.5*) | 9.6 (+3.4) | 13.3 |
| 2 | 11.3 (+4.5*) | 18.2 (+13.1*) | 18.6 (+10.0*) | 12.0 (+5.8*) | 15.0 |

Swahili refusal rate against base: seed 42 -14.8 points (p < 0.001); seed 1 -10.0 points (p < 0.001); seed 2 -12.7 points (p < 0.001).

Counting only the guard's Unsafe class (not part of the registered rule), Korean: seed 42 13.4% against 2.7% for base (+10.7 points, p < 0.001) and 14.1% for plain on the usual set; seed 1 13.1% against 2.7% for base (+10.3 points, p < 0.001) and 11.0% for plain on the usual set; seed 2 12.0% against 2.7% for base (+9.3 points, p < 0.001) and 13.1% for plain on the usual set.

English and Chinese each get 46% more rows here than in the usual set, so drift spreading from them may be larger than in the original runs. No row rules out that data quality contributed there.
