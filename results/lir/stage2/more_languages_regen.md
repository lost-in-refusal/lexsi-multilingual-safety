### Six more untrained languages, read again after re-generation at 1,024 tokens (section 14e)

Every answer in the six languages that reached the 256-token cap was generated again with a cap of 1,024 tokens and judged by the same guard, for the seven arms of 14b. Korean's rise is taken from the attack set after its own re-generation (section 8). The code of 14b is applied unchanged. **This reading decides nothing: U1 to U3 stand as read in `more_languages.md`**, and the languages named beside Korean stay those of that reading. It was registered after that report had been read, so it is not blind. No person has checked the guard in these six languages, and completing answers does not test the guard.

| | Capped at 256 tokens (the reading that stands) | After re-generation | Same side |
|---|---|---|---|
| U1, refusal rate, plain minus base, six languages | -4.9 [-7.0, -2.9], below zero | -4.7 [-6.7, -2.7], below zero | yes |
| U2, refusal rate, SafeGrad minus plain, six languages | +6.2 [+4.4, +8.0], above zero | +6.3 [+4.5, +8.0], above zero | yes |
| U3, row of the table | none rose more than Korean, some not distinguishable | none rose more than Korean, some not distinguishable | yes |
| Korean's own rise in strict attack success | +12.7 [+8.9, +16.7] | +13.2 [+9.5, +17.0] | |
| Korean's rise minus Italian's | +13.5 [+8.9, +18.4], above zero | +12.8 [+8.4, +17.5], above zero | yes |
| Korean's rise minus Vietnamese's | +4.6 [-0.0, +9.3], includes zero | +4.5 [-0.0, +9.0], includes zero | yes |
| Korean's rise minus Arabic's | +10.9 [+5.5, +16.3], above zero | +10.0 [+4.8, +15.2], above zero | yes |
| Korean's rise minus Thai's | +1.0 [-3.8, +6.0], includes zero | +1.6 [-3.1, +6.4], includes zero | yes |
| Korean's rise minus Bengali's | +11.1 [+5.5, +17.1], above zero | +7.6 [+2.4, +12.9], above zero | yes |
| Korean's rise minus Javanese's | +3.3 [-2.3, +8.7], includes zero | +4.6 [-0.9, +10.0], includes zero | yes |

- U1: the same on the capped answers and after re-generation.
- U2: the same on the capped answers and after re-generation.
- U3: the same on the capped answers and after re-generation.
- After re-generation, Korean's rise minus a language's is above zero for: Italian, Arabic, Bengali; includes zero for: Vietnamese, Thai, Javanese; below zero for: none.
- No language changed side.

| Language | Base strict attack success, capped | after | Strict, plain minus base, capped | after | SafeGrad minus plain, after | Refusals, plain minus base, capped | after | SafeGrad minus plain, after | Unsafe class only, plain minus base, after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Italian | 10.0 | 8.2 | -0.8 [-3.8, +2.3] | +0.3 [-2.7, +3.3] | -2.1 [-4.5, +0.2] | -4.9 [-9.3, -0.8] | -4.2 [-8.5, -0.2] | +6.2 [+2.6, +9.7] | +1.6 [-0.8, +4.0] |
| Vietnamese | 4.8 | 3.4 | +8.1 [+5.0, +11.6] | +8.7 [+5.7, +11.9] | -6.2 [-9.2, -3.4] | -16.0 [-20.3, -11.7] | -15.5 [-19.7, -11.2] | +10.9 [+7.6, +14.5] | +7.6 [+4.8, +10.7] |
| Arabic | 10.0 | 8.6 | +1.8 [-1.5, +5.2] | +3.2 [-0.0, +6.5] | -1.1 [-3.9, +1.5] | +1.3 [-2.5, +4.9] | +1.9 [-1.7, +5.5] | +1.8 [-1.3, +5.2] | +3.4 [+0.7, +6.4] |
| Thai | 5.2 | 5.2 | +11.7 [+8.1, +15.3] | +11.6 [+8.0, +15.2] | -7.9 [-11.1, -5.0] | -7.4 [-11.3, -3.7] | -7.3 [-11.2, -3.4] | +11.3 [+7.8, +15.0] | +7.7 [+5.0, +10.7] |
| Bengali | 15.8 | 11.3 | +1.6 [-2.5, +5.6] | +5.6 [+1.7, +9.3] | -3.7 [-7.0, -0.5] | +1.8 [-1.7, +5.5] | +1.0 [-2.7, +4.9] | +2.1 [-1.6, +5.8] | +6.4 [+3.8, +9.3] |
| Javanese | 7.2 | 7.6 | +9.4 [+5.4, +13.5] | +8.6 [+4.6, +12.8] | -3.3 [-6.2, -0.6] | -4.4 [-10.0, +1.0] | -4.2 [-9.6, +1.0] | +5.4 [+1.3, +9.6] | +5.3 [+2.4, +8.2] |

Rates in %, differences in points, pooled over the three seeds, with no rule.

Answers generated again, of which the original continued, and still cut at 1,024 tokens: `base` it 122/57/0, vi 90/44/0, ar 131/66/1, th 68/20/3, bn 192/77/1, jv 99/50/5; `plain` it 16/4/0, vi 44/12/2, ar 25/8/0, th 32/5/5, bn 25/3/0, jv 31/11/2; `safegrad` it 52/17/0, vi 54/29/1, ar 61/21/0, th 43/18/7, bn 102/34/4, jv 27/13/6; `plain-s1` it 12/2/1, vi 32/12/3, ar 18/1/1, th 32/11/5, bn 20/6/2, jv 19/9/2; `safegrad-s1` it 50/13/0, vi 49/21/1, ar 62/26/0, th 27/14/6, bn 97/29/4, jv 31/12/6; `plain-s2` it 16/2/1, vi 38/14/2, ar 23/11/0, th 33/14/5, bn 25/10/3, jv 27/10/4; `safegrad-s2` it 53/20/1, vi 50/21/0, ar 56/12/3, th 35/12/4, bn 112/32/1, jv 35/18/9.
