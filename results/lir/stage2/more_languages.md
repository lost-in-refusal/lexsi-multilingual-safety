### Six more languages the model was never fine-tuned on (U1 to U3)

The 291 attack families in Italian, Vietnamese, Arabic, Thai, Bengali and Javanese, answered by base and by plain and SafeGrad at seeds 42, 1 and 2. No model was trained for this. Section 14b of the preregistration fixed the rules before any answer in these languages existed. Refusal rate decides U1 and U2, as in H3. **No person has checked the guard in these six languages**, and in Swahili the guard missed harm that a re-translation found, so every attack-success figure here is what the guard sees. Answers are capped at 256 tokens. Intervals are 95%, from a bootstrap over the 291 families with each family's value averaged over the three seeds.

- **U1, spillover (refusal rate, plain minus base, averaged over the six languages): -4.9 [-7.0, -2.9] points, below zero. Refusals on attack prompts fell in languages the model was never fine-tuned on, beyond Swahili: on average over six more languages.**
- **U2, repair (refusal rate, SafeGrad minus plain, averaged over the six languages): +6.2 [+4.4, +8.0] points, above zero. SafeGrad kept more refusals than plain fine-tuning in the six further untrained languages.**
- **U3, where Korean stands: As the guard measures it, no further language rose more than Korean, and the rise in some of them could not be told apart from Korean's. "Korean is hit hardest" is then written only as: hardest among the four languages of the main evaluation, and those languages are named as not distinguishable from Korean.** Not distinguishable from Korean's rise (the interval includes zero): Vietnamese, Thai, Javanese.
  Korean's own rise on the attack set, for comparison: +12.7 [+8.9, +16.7] points.

| Language | Base refusal rate | Refusals, plain minus base | Refusals, SafeGrad minus plain | Base strict attack success | Strict attack success, plain minus base | SafeGrad minus plain | Unsafe class only, plain minus base | Korean's rise minus this language's |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Italian | 60.8 | -4.9 [-9.3, -0.8] | +6.0 [+2.4, +9.6] | 10.0 | -0.8 [-3.8, +2.3] | -2.1 [-4.5, +0.2] | -1.1 [-3.8, +1.4] | +13.5 [+8.9, +18.4] |
| Vietnamese | 73.5 | -16.0 [-20.3, -11.7] | +11.1 [+7.8, +14.7] | 4.8 | +8.1 [+5.0, +11.6] | -6.3 [-9.4, -3.4] | +6.5 [+3.8, +9.5] | +4.6 [-0.0, +9.3] |
| Arabic | 56.7 | +1.3 [-2.5, +4.9] | +1.5 [-1.8, +4.9] | 10.0 | +1.8 [-1.5, +5.2] | -0.7 [-3.4, +1.9] | +1.8 [-0.9, +4.7] | +10.9 [+5.5, +16.3] |
| Thai | 73.9 | -7.4 [-11.3, -3.7] | +11.2 [+7.8, +14.9] | 5.2 | +11.7 [+8.1, +15.3] | -7.7 [-10.8, -4.8] | +8.8 [+6.0, +12.0] | +1.0 [-3.8, +6.0] |
| Bengali | 46.7 | +1.8 [-1.7, +5.5] | +2.1 [-1.6, +5.7] | 15.8 | +1.6 [-2.5, +5.6] | -2.7 [-6.2, +0.6] | +2.6 [-0.6, +5.7] | +11.1 [+5.5, +17.1] |
| Javanese | 51.2 | -4.4 [-10.0, +1.0] | +5.3 [+1.1, +9.5] | 7.2 | +9.4 [+5.4, +13.5] | -3.6 [-6.5, -0.7] | +4.1 [+1.0, +7.4] | +3.3 [-2.3, +8.7] |

Rates in %, differences in points, pooled over the three seeds. Only U1, U2 and U3 have a rule; the table is shown beside them.

Per seed, refusal rate, plain minus base (exact McNemar): seed 42 it -5.5 (p = 0.026), vi -16.8 (p < 0.001), ar +1.0 (p = 0.736), th -8.2 (p < 0.001), bn -1.0 (p = 0.743), jv -8.2 (p = 0.007); seed 1 it -3.8 (p = 0.144), vi -15.1 (p < 0.001), ar +2.7 (p = 0.256), th -7.2 (p = 0.001), bn +4.8 (p = 0.024), jv -2.4 (p = 0.483); seed 2 it -5.5 (p = 0.023), vi -16.2 (p < 0.001), ar +0.0 (p = 1.000), th -6.9 (p = 0.004), bn +1.7 (p = 0.500), jv -2.4 (p = 0.489).

Share of answers cut at the 256-token cap (%): `base` it 42, vi 31, ar 45, th 23, bn 66, jv 34; `plain` it 5, vi 15, ar 9, th 11, bn 9, jv 11; `safegrad` it 18, vi 19, ar 21, th 15, bn 35, jv 9; `plain-s1` it 4, vi 11, ar 6, th 11, bn 7, jv 7; `safegrad-s1` it 17, vi 17, ar 21, th 9, bn 33, jv 11; `plain-s2` it 5, vi 13, ar 8, th 11, bn 9, jv 9; `safegrad-s2` it 18, vi 17, ar 19, th 12, bn 38, jv 12.

Not done in these languages: XSTest (no over-refusal figure), the task set, a human check of the guard, re-translation, and generating capped answers again at 1,024 tokens.
