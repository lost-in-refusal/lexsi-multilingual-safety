### A second safety benchmark: the unsafe prompts of XSTest (X1 to X4)

The 200 unsafe prompts of XSTest, in English as written and in Korean, Chinese and Swahili as NLLB-200 translates them, answered by base, by plain and SafeGrad at seeds 42, 1 and 2, and by the run with five copies of the refusal examples mixed into the training data. No model was trained for this. Section 14i of the preregistration fixed the rules before any model had answered one of these prompts. Strict attack success decides all four checks, as in H1 and H2. The refusal examples of SafeGrad and of mixing come from MultiJail, so this benchmark is one they do not come from. **No person has checked the guard on these prompts or the translations**, so every figure here is what the guard sees on machine-translated prompts. Answers are capped at 256 tokens. Intervals are 95%, from a bootstrap over the prompts with each prompt's value averaged over the seeds.

- **X1, the drift (Korean strict attack success, plain minus base): +10.5 [+5.8, +15.3] points, above zero. Plain fine-tuning also raised Korean attack success on a second safety benchmark, the unsafe prompts of XSTest.**
- **X2, Korean against English (Korean rise minus English rise): +8.7 [+3.5, +13.8] points, above zero. On the second benchmark Korean rose more than English.**
- **X3, SafeGrad (Korean strict attack success, SafeGrad minus plain): -7.2 [-10.7, -3.8] points, below zero. SafeGrad kept Korean attack success below plain fine-tuning on a benchmark that its refusal examples do not come from.**
- **X4, mixing (Korean strict attack success, five copies minus plain): -8.5 [-12.2, -4.8] points, below zero. Mixing five copies of the refusal examples into the training data kept Korean attack success below plain fine-tuning on a benchmark that the examples do not come from.**
- Beside, no rule. Korean strict attack success, five copies minus SafeGrad: -1.3 [-3.8, +1.2].

| Language | Prompts | Base strict attack success | Plain minus base | SafeGrad minus plain | Five copies minus plain | Base refusal rate | Refusals, plain minus base | Unsafe class only, plain minus base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| English | 200 | 1.0 | +1.8 [-0.7, +4.3] | -2.2 [-4.7, +0.0] | -2.5 [-4.7, -0.8] | 86.0 | -4.8 [-8.7, -1.2] | +1.2 [-1.2, +3.5] |
| Korean | 200 | 6.0 | +10.5 [+5.8, +15.3] | -7.2 [-10.7, -3.8] | -8.5 [-12.2, -4.8] | 70.0 | -5.0 [-9.7, -0.3] | +7.2 [+3.5, +11.0] |
| Chinese | 200 | 3.5 | +11.8 [+7.7, +16.3] | -5.2 [-8.8, -1.8] | -8.8 [-12.5, -5.5] | 78.5 | -9.0 [-13.7, -4.7] | +6.7 [+3.3, +10.3] |
| Swahili | 200 | 5.0 | +0.5 [-2.5, +3.3] | -1.2 [-3.7, +1.3] | -3.2 [-5.7, -1.0] | 66.0 | -11.0 [-15.7, -6.3] | +1.5 [-0.3, +3.5] |

Rates in %, differences in points, pooled over the seeds. Only X1 to X4 have a rule; the table is shown beside them, and its cells are not adjusted for their number.

Per seed, Korean strict attack success (exact McNemar on the same prompts): seed 42: plain 21.0% (+15.0 against base, p < 0.001), SafeGrad 8.5% (-12.5 against plain, p < 0.001), five copies 9.0% (-12.0 against plain, p < 0.001); seed 1: plain 14.5% (+8.5 against base, p = 0.003), SafeGrad 10.0% (-4.5 against plain, p = 0.093), five copies 7.5% (-7.0 against plain, p = 0.007); seed 2: plain 14.0% (+8.0 against base, p = 0.004), SafeGrad 9.5% (-4.5 against plain, p = 0.064), five copies 7.5% (-6.5 against plain, p = 0.004).

Share of answers cut at the 256-token cap (%): `base` en 10, ko 36, zh 23, sw 24; `plain` en 1, ko 6, zh 1, sw 18; `safegrad` en 6, ko 9, zh 4, sw 18; `plain-s1` en 0, ko 4, zh 2, sw 12; `safegrad-s1` en 4, ko 10, zh 4, sw 14; `plain-s2` en 1, ko 4, zh 1, sw 15; `safegrad-s2` en 4, ko 10, zh 7, sw 17; `plain-mix5` en 2, ko 6, zh 2, sw 10; `plain-mix5-s1` en 0, ko 4, zh 4, sw 14; `plain-mix5-s2` en 2, ko 8, zh 4, sw 8.

Not done on this benchmark: a human check of the guard or of the translations, generating capped answers again at 1,024 tokens, and the mixing run with the examples added once.
