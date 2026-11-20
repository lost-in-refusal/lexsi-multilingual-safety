### K1: plain fine-tuning with no Korean training rows, strict attack success (%)

One run, seed 42, on 874 English and 874 Chinese rows. The trainer dropped 1 of them as over its length limit, so 1,747 rows were trained; the usual runs dropped none. Added to the preregistration on Nov 20 before it ran.

| Arm | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| plain, usual set | 6.9 | 19.2 | 13.4 | 7.6 | 11.8 |
| plain, no Korean rows | 9.6 (+2.7) | 18.2 (+13.1*) | 14.1 (+5.5*) | 10.7 (+4.5*) | 13.1 |

Parentheses: change against base on the same prompts. * = p < 0.05, exact McNemar.

Korean rises by +13.1 points over base (p < 0.001); the bar is 6.36 points. Against plain on the usual set it is -1.0 points (p = 0.743).

**Outcome row 1: Korean training rows are not needed for a drift of this size.**

Swahili refusal rate changes by -14.8 points against base (p < 0.001).

Counting only the guard's Unsafe class (not part of the registered rule), Korean is 13.4% against 2.7% for base (+10.7 points, p < 0.001) and 14.1% for plain on the usual set.

English and Chinese each get 46% more rows here than in the usual set, so drift spreading from them may be larger than in the original runs. No row rules out that data quality contributed there.
