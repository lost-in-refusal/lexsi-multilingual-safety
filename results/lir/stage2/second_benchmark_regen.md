### The second safety benchmark, read again after re-generation at 1,024 tokens (section 14j)

Every answer of the ten arms to the unsafe XSTest prompts that reached the 256-token cap was generated again with a cap of 1,024 tokens and judged by the same guard. The code of 14i is applied unchanged. **This reading decides nothing: X1 to X4 stand as read in `second_benchmark.md`.** It was registered after that report had been read, so it is not blind. No person has checked the guard or the translations on these prompts, and completing answers does not test the guard.

| | Capped at 256 tokens (the reading that stands) | After re-generation | Same side |
|---|---|---|---|
| X1, Korean strict attack success, plain minus base | +10.5 [+5.8, +15.3], above zero | +12.2 [+7.7, +16.8], above zero | yes |
| X2, Korean rise minus English rise | +8.7 [+3.5, +13.8], above zero | +10.3 [+5.3, +15.3], above zero | yes |
| X3, Korean strict attack success, SafeGrad minus plain | -7.2 [-10.7, -3.8], below zero | -7.0 [-10.3, -3.8], below zero | yes |
| X4, Korean strict attack success, five copies minus plain | -8.5 [-12.2, -4.8], below zero | -8.5 [-12.2, -5.0], below zero | yes |

- X1: the same on the capped answers and after re-generation.
- X2: the same on the capped answers and after re-generation.
- X3: the same on the capped answers and after re-generation.
- X4: the same on the capped answers and after re-generation.
- Beside, no rule. Korean strict attack success, five copies minus SafeGrad: -1.3 [-3.8, +1.2] capped, -1.5 [-3.8, +0.7] after re-generation.

| Language | Base strict attack success, capped | after | Plain minus base, capped | after | SafeGrad minus plain, capped | after | Five copies minus plain, capped | after | Refusals, plain minus base, after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| English | 1.0 | 1.0 | +1.8 [-0.7, +4.3] | +1.8 [-0.7, +4.3] | -2.2 [-4.7, +0.0] | -2.0 [-4.5, +0.3] | -2.5 [-4.7, -0.8] | -2.5 [-4.7, -0.8] | -3.8 [-7.8, +0.2] |
| Korean | 6.0 | 4.0 | +10.5 [+5.8, +15.3] | +12.2 [+7.7, +16.8] | -7.2 [-10.7, -3.8] | -7.0 [-10.3, -3.8] | -8.5 [-12.2, -4.8] | -8.5 [-12.2, -5.0] | -5.5 [-10.2, -1.0] |
| Chinese | 3.5 | 3.5 | +11.8 [+7.7, +16.3] | +11.8 [+7.7, +16.3] | -5.2 [-8.8, -1.8] | -5.2 [-8.7, -2.0] | -8.8 [-12.5, -5.5] | -8.8 [-12.5, -5.5] | -8.8 [-13.3, -4.5] |
| Swahili | 5.0 | 3.5 | +0.5 [-2.5, +3.3] | +1.3 [-1.0, +3.7] | -1.2 [-3.7, +1.3] | -1.0 [-3.3, +1.3] | -3.2 [-5.7, -1.0] | -2.7 [-4.8, -0.7] | -10.8 [-15.3, -6.2] |

Rates in %, differences in points, pooled over the seeds. Only the Korean cells that X1, X3 and X4 read have a rule in 14i; the table is shown beside, and its cells are not adjusted for their number.

Per seed after re-generation, Korean strict attack success (exact McNemar on the same prompts): seed 42: plain 20.5% (+16.5 against base, p < 0.001), SafeGrad 8.5% (-12.0 against plain, p < 0.001), five copies 8.5% (-12.0 against plain, p < 0.001); seed 1: plain 14.5% (+10.5 against base, p < 0.001), SafeGrad 9.5% (-5.0 against plain, p = 0.052), five copies 7.0% (-7.5 against plain, p = 0.003); seed 2: plain 13.5% (+9.5 against base, p < 0.001), SafeGrad 9.5% (-4.0 against plain, p = 0.077), five copies 7.5% (-6.0 against plain, p = 0.008).

Answers generated again, of which the original continued, and still cut at 1,024 tokens: `base` en 19/10/0, ko 71/50/1, zh 46/11/1, sw 49/30/0; `plain` en 2/2/0, ko 13/6/0, zh 2/1/1, sw 35/16/1; `safegrad` en 11/4/0, ko 18/7/0, zh 8/4/1, sw 35/19/0; `plain-s1` en 0/0/0, ko 9/1/1, zh 4/3/2, sw 25/5/0; `safegrad-s1` en 8/4/0, ko 20/10/1, zh 8/1/0, sw 27/16/0; `plain-s2` en 2/0/0, ko 7/2/0, zh 2/1/1, sw 30/9/1; `safegrad-s2` en 9/2/0, ko 19/9/0, zh 14/5/0, sw 34/21/0; `plain-mix5` en 3/0/0, ko 13/11/0, zh 5/2/1, sw 19/12/0; `plain-mix5-s1` en 0/0/0, ko 9/5/0, zh 7/4/2, sw 27/18/0; `plain-mix5-s2` en 3/1/0, ko 15/5/0, zh 7/2/2, sw 17/11/0.
