### The model outside the Tiny Aya family, read again after re-generation (section 16a)

Every attack answer of the seven runs of section 16 that reached the 256-token cap, generated again at 1,024 tokens by the model that wrote it and judged by the same guard. The other answers and their verdicts are those of section 16. It decides nothing: G1 to G3 stand as section 16 read them, and this reading is given beside.

**G1, plain minus base: the same on the capped answers and after re-generation.** On the capped answers +10.1 [+7.6, +12.6]; after re-generation +11.5 [+9.2, +13.9].

**G2, the Korean rise minus the English rise: the same on the capped answers and after re-generation.** On the capped answers +20.8 [+15.1, +26.3]; after re-generation +24.2 [+18.9, +29.3].

**G3, SafeGrad minus plain: the same on the capped answers and after re-generation.** On the capped answers -13.7 [-16.1, -11.4]; after re-generation -13.9 [-16.3, -11.5].

After re-generation, with no rule. Base rate (%): en 6.5, ko 14.4, zh 10.7, sw 21.0. Plain against base: en +0.2 [-2.2, +2.7], ko +24.4 [+19.6, +29.3], zh +9.9 [+6.2, +13.4], sw +1.7 [-2.3, +5.5]. SafeGrad against base: en -5.0 [-7.7, -2.6], ko -0.5 [-4.2, +3.2], zh -1.6 [-4.2, +0.8], sw +1.1 [-3.0, +5.2].

| Run | Generated again | en | ko | zh | sw | Still cut at 1,024 tokens: en, ko, zh, sw |
|---|---:|---:|---:|---:|---:|---|
| `granite41-base` | 600 | 59 | 213 | 106 | 222 | 0, 10, 1, 106 |
| `granite41-plain-s42` | 266 | 2 | 64 | 37 | 163 | 0, 9, 1, 104 |
| `granite41-safegrad-s42` | 345 | 15 | 111 | 50 | 169 | 0, 10, 1, 115 |
| `granite41-plain-s1` | 286 | 5 | 85 | 42 | 154 | 0, 16, 4, 86 |
| `granite41-safegrad-s1` | 315 | 12 | 111 | 60 | 132 | 0, 8, 1, 78 |
| `granite41-plain-s2` | 270 | 3 | 76 | 37 | 154 | 0, 12, 3, 113 |
| `granite41-safegrad-s2` | 352 | 15 | 135 | 51 | 151 | 0, 14, 1, 101 |

Batches are formed from the prompts that are generated again, so an answer generated again is often another answer from its first character and not a continuation. The verdicts are the guard's; no person has read these answers.
