### How far each language moves along the refusal direction

Projection of each MultiJail attack prompt's layer-17 activation on the base model's English refusal direction (the direction of stage 1, found on the base model alone). A shift is an arm's projection minus the base model's on the same prompt. Section 13f of the preregistration fixed R1 and R2 before the activations of the new arms were extracted. No judge is involved.

Median projection per language, with the mean shift against base in parentheses:

| Arm | en | ko | zh | sw |
|---|---:|---:|---:|---:|
| base | 15.52 | 13.30 | 13.72 | 11.72 |
| plain | 11.00 (-4.38) | 9.14 (-4.05) | 10.00 (-3.52) | 7.06 (-4.15) |
| plain-s1 | 12.35 (-3.06) | 10.15 (-3.24) | 10.96 (-2.78) | 8.49 (-2.96) |
| plain-s2 | 11.87 (-3.43) | 9.90 (-3.34) | 10.77 (-2.96) | 8.12 (-3.14) |
| plain-only-en | 11.57 (-3.82) | 10.99 (-2.52) | 10.20 (-3.58) | 8.29 (-3.33) |
| plain-only-en-s1 | 11.19 (-4.10) | 10.96 (-2.54) | 10.13 (-3.62) | 8.04 (-3.56) |
| plain-only-en-s2 | 11.53 (-3.80) | 11.08 (-2.43) | 10.29 (-3.46) | 8.44 (-3.27) |
| plain-only-zh | 11.50 (-3.83) | 9.92 (-3.63) | 10.42 (-3.46) | 7.35 (-3.98) |
| plain-only-zh-s1 | 11.87 (-3.54) | 10.33 (-3.13) | 11.09 (-2.85) | 7.73 (-3.56) |
| plain-only-zh-s2 | 11.68 (-3.76) | 10.01 (-3.49) | 10.61 (-3.26) | 7.42 (-3.77) |
| safegrad | 13.24 (-2.32) | 11.44 (-2.17) | 11.72 (-2.42) | 9.55 (-2.09) |
| safegrad-s1 | 13.62 (-2.06) | 11.88 (-1.91) | 11.99 (-2.16) | 9.88 (-1.86) |
| safegrad-s2 | 13.49 (-2.07) | 11.67 (-1.98) | 11.54 (-2.45) | 9.60 (-2.09) |
| plain-only-ko | 11.96 (-3.43) | 10.22 (-3.24) | 10.78 (-3.09) | 8.16 (-3.25) |
| plain-only-ko-s1 | 11.40 (-3.93) | 10.03 (-3.37) | 10.73 (-3.07) | 8.12 (-3.30) |
| plain-only-ko-s2 | 11.84 (-3.61) | 10.44 (-3.04) | 10.97 (-2.77) | 8.38 (-2.98) |

**R1, the three main seeds.** Korean shift minus English shift, per prompt family, averaged over plain at seeds 42, 1 and 2: +0.08 [-0.08, +0.25] units. Equivalence margin: 1 unit. **Korean and English moved the same distance along the refusal direction, to within 1 unit.**

**R2, trained on English only.** Korean shift minus English shift, over `plain-only-en`, `plain-only-en-s1`, `plain-only-en-s2`: +1.41 [+1.28, +1.54] units. **Korean moved less far along the refusal direction than English.**

**R2, trained on Chinese only.** Korean shift minus Chinese shift, over `plain-only-zh`, `plain-only-zh-s1`, `plain-only-zh-s2`: -0.22 [-0.37, -0.08] units. **Korean and Chinese moved the same distance along the refusal direction, to within 1 unit.**

Refusal rate on attack prompts (%), from the guard's refusal flag, shown beside with no rule:

| Arm | en | ko | zh | sw |
|---|---:|---:|---:|---:|
| base | 71.1 | 59.8 | 66.3 | 50.9 |
| plain | 66.7 | 47.4 | 58.1 | 37.8 |
| plain-s1 | 66.3 | 52.2 | 59.8 | 44.7 |
| plain-s2 | 64.6 | 48.8 | 56.7 | 42.3 |
| plain-only-en | 66.0 | 54.0 | 68.4 | 45.7 |
| plain-only-en-s1 | 65.3 | 55.3 | 67.7 | 46.0 |
| plain-only-en-s2 | 66.7 | 56.0 | 69.1 | 45.7 |
| plain-only-zh | 61.9 | 50.5 | 59.8 | 43.6 |
| plain-only-zh-s1 | 61.9 | 51.2 | 59.5 | 45.4 |
| plain-only-zh-s2 | 61.2 | 50.9 | 61.5 | 41.6 |
| safegrad | 75.9 | 57.0 | 67.4 | 56.4 |
| safegrad-s1 | 78.4 | 59.1 | 67.7 | 56.0 |
| safegrad-s2 | 76.6 | 58.4 | 67.0 | 54.0 |
| plain-only-ko | 63.2 | 52.6 | 59.8 | 44.7 |
| plain-only-ko-s1 | 64.6 | 52.9 | 62.9 | 49.1 |
| plain-only-ko-s2 | 62.2 | 50.9 | 60.1 | 48.1 |

Base, plain and SafeGrad at seed 42 were extracted again with the new arms. Largest difference from the stage 1 projection of the same prompt: base 0.000 (mean 0.000); plain 0.000 (mean 0.000).
