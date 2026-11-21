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

### What was generated again

| Arm | Set | Answers | Start with the original | Still cut at the new cap |
|---|---|---:|---:|---:|
| base | attack | 355 | 93 | 6 |
| base | xstest | 484 | 269 | 10 |
| plain | attack | 99 | 32 | 9 |
| plain | xstest | 153 | 60 | 15 |
| safegrad | attack | 139 | 23 | 8 |
| safegrad | xstest | 160 | 97 | 9 |

### Over-refusal on XSTest (%), before -> after

| Arm | en | ko | zh | sw |
|---|---|---|---|---|
| base | 9.6 -> 8.8 | 7.6 -> 7.6 | 8.8 -> 8.0 | 7.6 -> 7.6 |
| plain | 7.2 -> 7.2 | 5.6 -> 5.6 | 5.6 -> 5.6 | 2.8 -> 3.2 |
| safegrad | 7.6 -> 7.6 | 6.8 -> 6.8 | 5.6 -> 5.6 | 5.6 -> 5.6 |
