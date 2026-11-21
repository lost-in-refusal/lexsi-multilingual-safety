### Truncation audit: seed 42 before and after re-generation at 1024 tokens

Strict attack success (%), read at one seed.

| | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base, before | 6.9 | 5.2 | 8.6 | 6.2 | 6.7 |
| plain, before | 6.9 | 19.2 | 13.4 | 7.6 | 11.8 |
| safegrad, before | 4.5 | 10.7 | 8.9 | 5.5 | 7.4 |
| base, after | 5.5 | 4.5 | 7.9 | 3.8 | 5.4 |
| plain, after | 6.9 | 19.2 | 13.4 | 7.9 | 11.9 |
| safegrad, after | 4.1 | 10.3 | 8.9 | 5.5 | 7.2 |

- Before: Korean rise +14.1 points (p = 5.81e-09), English rise +0.0; gap +14.1 points [+8.6, +19.6]. SafeGrad against plain on Korean -8.6 points [-12.7, -4.5]. Swahili refusals -13.1 points [-18.2, -8.2]. H1 pass, H2 pass, H3 pass.
- After: Korean rise +14.8 points (p = 2.42e-10), English rise +1.4; gap +13.4 points [+7.9, +19.2]. SafeGrad against plain on Korean -8.9 points [-13.1, -4.8]. Swahili refusals -13.4 points [-18.6, -8.6]. H1 pass, H2 pass, H3 pass.
- Size of the Korean rise, after minus before: +0.7 points [+0.0, +1.7]: unchanged within noise.
- SafeGrad against plain on Korean, after minus before: -0.3 points [-1.0, +0.0]: the repair is unchanged within noise.

Rows of the outcome table that apply:

- H1 passes and the Korean rise has not shrunk: the headline does not depend on the 256-token cap

### Second reading, deciding nothing

Only prompts where every arm's answer was complete or its re-generation starts with the original answer: 231 en, 210 ko, 214 zh, 184 sw of 291.

| | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base, before | 3.5 | 1.9 | 3.7 | 2.7 | 3.0 |
| plain, before | 4.3 | 13.3 | 7.9 | 3.3 | 7.2 |
| safegrad, before | 1.7 | 7.1 | 4.7 | 2.2 | 3.9 |
| base, after | 3.5 | 1.9 | 3.3 | 1.6 | 2.6 |
| plain, after | 4.3 | 13.3 | 7.9 | 3.3 | 7.2 |
| safegrad, after | 1.7 | 7.1 | 4.7 | 1.6 | 3.8 |

- Before: Korean rise +11.4 points (p = 3.03e-06), English rise +0.9; gap +11.2 points [+5.6, +16.9]. SafeGrad against plain on Korean -6.2 points [-10.5, -1.9]. Swahili refusals -14.1 points [-21.2, -7.1]. H1 pass, H2 pass, H3 pass.
- After: Korean rise +11.4 points (p = 3.03e-06), English rise +0.9; gap +11.2 points [+5.6, +16.9]. SafeGrad against plain on Korean -6.2 points [-10.5, -1.9]. Swahili refusals -14.7 points [-21.2, -8.2]. H1 pass, H2 pass, H3 pass.
- Size of the Korean rise, after minus before: +0.0 points [+0.0, +0.0]: unchanged within noise.
- SafeGrad against plain on Korean, after minus before: +0.0 points [+0.0, +0.0]: the repair is unchanged within noise.

### Three seeds, by the rules of section 1

- Before: H1 pass (gap +11.8 points [+6.8, +16.7]), H2 pass (-8.4 points [-11.3, -5.5]), H3 pass (-9.3 points [-13.7, -5.2]).
- After: H1 pass (gap +10.9 points [+6.0, +15.9]), H2 pass (-8.6 points [-11.5, -5.7]), H3 pass (-10.0 points [-14.2, -5.8]).
- Seed 42: Korean rise after minus before +0.7 points [+0.0, +1.7]: unchanged within noise.
- Seed 1: Korean rise after minus before +0.3 points [-0.7, +1.4]: unchanged within noise.
- Seed 2: Korean rise after minus before +0.3 points [-0.7, +1.4]: unchanged within noise.

### Tiny Aya Earth, seed 42

| | en | ko | zh | sw | macro |
|---|---:|---:|---:|---:|---:|
| base, before | 7.9 | 10.7 | 11.0 | 7.6 | 9.3 |
| plain, before | 8.9 | 21.3 | 13.4 | 6.5 | 12.5 |
| safegrad, before | 5.8 | 16.5 | 8.9 | 5.5 | 9.2 |
| base, after | 7.2 | 7.9 | 10.0 | 4.5 | 7.4 |
| plain, after | 8.9 | 21.0 | 12.7 | 5.8 | 12.1 |
| safegrad, after | 5.8 | 15.5 | 9.3 | 6.2 | 9.2 |

- Before: Korean rise +10.7 points (p = 5.54e-06), English rise +1.0; gap +9.6 points [+4.5, +14.8]. SafeGrad against plain on Korean -4.8 points [-8.9, -1.0]. Swahili refusals -6.9 points [-11.3, -2.4]. H1 pass, H2 pass, H3 pass.
- After: Korean rise +13.1 points (p = 1.37e-08), English rise +1.7; gap +11.3 points [+6.2, +16.8]. SafeGrad against plain on Korean -5.5 points [-9.6, -1.7]. Swahili refusals -8.2 points [-13.1, -3.8]. H1 pass, H2 pass, H3 pass.
- Size of the Korean rise, after minus before: +2.4 points [+0.3, +4.5]: larger.
- SafeGrad against plain on Korean, after minus before: -0.7 points [-2.1, +0.7]: the repair is unchanged within noise.
- E1 before: holds.
- E1 after: holds.

### What was generated again

| Arm | Set | Answers | Start with the original | Still cut at the new cap |
|---|---|---:|---:|---:|
| base | attack | 355 | 93 | 6 |
| base | xstest | 484 | 269 | 10 |
| plain | attack | 99 | 32 | 9 |
| plain | xstest | 153 | 60 | 15 |
| safegrad | attack | 139 | 23 | 8 |
| safegrad | xstest | 160 | 97 | 9 |
| plain-s1 | attack | 81 | 22 | 8 |
| plain-s1 | xstest | 124 | 32 | 7 |
| safegrad-s1 | attack | 111 | 41 | 3 |
| safegrad-s1 | xstest | 170 | 61 | 12 |
| plain-s2 | attack | 94 | 24 | 11 |
| plain-s2 | xstest | 131 | 44 | 4 |
| safegrad-s2 | attack | 121 | 46 | 5 |
| safegrad-s2 | xstest | 165 | 84 | 11 |
| earth-base | attack | 426 | 150 | 7 |
| earth-base | xstest | 557 | 294 | 22 |
| earth-plain | attack | 95 | 30 | 12 |
| earth-plain | xstest | 139 | 56 | 14 |
| earth-safegrad | attack | 183 | 66 | 8 |
| earth-safegrad | xstest | 170 | 66 | 12 |

### Over-refusal on XSTest (%), before -> after

| Arm | en | ko | zh | sw |
|---|---|---|---|---|
| base | 9.6 -> 8.8 | 7.6 -> 7.6 | 8.8 -> 8.0 | 7.6 -> 7.6 |
| plain | 7.2 -> 7.2 | 5.6 -> 5.6 | 5.6 -> 5.6 | 2.8 -> 3.2 |
| safegrad | 7.6 -> 7.6 | 6.8 -> 6.8 | 5.6 -> 5.6 | 5.6 -> 5.6 |
| plain-s1 | 6.0 -> 6.0 | 5.6 -> 5.2 | 4.8 -> 4.8 | 4.0 -> 4.0 |
| safegrad-s1 | 8.8 -> 8.8 | 7.2 -> 7.2 | 6.4 -> 6.4 | 6.0 -> 6.0 |
| plain-s2 | 6.0 -> 6.0 | 5.6 -> 5.2 | 5.6 -> 5.6 | 3.6 -> 3.6 |
| safegrad-s2 | 10.8 -> 10.8 | 7.2 -> 7.2 | 6.4 -> 6.4 | 6.0 -> 6.0 |
| earth-base | 9.2 -> 9.2 | 8.0 -> 8.0 | 6.4 -> 6.4 | 6.8 -> 6.4 |
| earth-plain | 6.0 -> 6.0 | 4.8 -> 4.8 | 4.8 -> 5.2 | 4.0 -> 4.8 |
| earth-safegrad | 9.6 -> 9.6 | 8.4 -> 8.4 | 6.0 -> 6.0 | 4.8 -> 4.8 |
