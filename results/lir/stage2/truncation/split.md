### Re-generated answers: the original continued, or a different answer

Counted after the seed 42 re-generation had been read. A verdict change is the guard's strict flag (Unsafe or Controversial) appearing or going away.

| Arm | Set | Continued | to flagged | to safe | Different | to flagged | to safe |
|---|---|---:|---:|---:|---:|---:|---:|
| base | attack | 93 | 0 | 3 | 262 | 1 | 13 |
| base | xstest | 269 | 3 | 4 | 215 | 3 | 3 |
| plain | attack | 32 | 0 | 0 | 67 | 3 | 2 |
| plain | xstest | 60 | 1 | 2 | 93 | 2 | 3 |
| safegrad | attack | 23 | 0 | 1 | 116 | 3 | 4 |
| safegrad | xstest | 97 | 0 | 1 | 63 | 1 | 0 |
| plain-s1 | attack | 22 | 0 | 1 | 59 | 2 | 3 |
| plain-s1 | xstest | 32 | 1 | 0 | 92 | 1 | 3 |
| safegrad-s1 | attack | 41 | 0 | 0 | 70 | 1 | 4 |
| safegrad-s1 | xstest | 61 | 0 | 2 | 109 | 0 | 0 |
| plain-s2 | attack | 24 | 0 | 1 | 70 | 3 | 0 |
| plain-s2 | xstest | 44 | 1 | 1 | 87 | 2 | 2 |
| safegrad-s2 | attack | 46 | 0 | 1 | 75 | 1 | 9 |
| safegrad-s2 | xstest | 84 | 1 | 4 | 81 | 0 | 1 |
| earth-base | attack | 150 | 1 | 4 | 276 | 3 | 22 |
| earth-base | xstest | 294 | 1 | 5 | 263 | 3 | 5 |
| earth-plain | attack | 30 | 0 | 2 | 65 | 1 | 4 |
| earth-plain | xstest | 56 | 0 | 1 | 83 | 0 | 6 |
| earth-safegrad | attack | 66 | 0 | 1 | 117 | 4 | 3 |
| earth-safegrad | xstest | 66 | 0 | 2 | 104 | 4 | 6 |

### Tiny Aya Earth, E1 on prompts where every arm's answer was complete or continued

Prompts: 234 en, 182 ko, 220 zh, 192 sw of 291.

| | en | ko | zh | sw |
|---|---:|---:|---:|---:|
| base, before | 3.0 | 4.9 | 5.9 | 3.6 |
| plain, before | 5.1 | 21.4 | 6.4 | 3.1 |
| safegrad, before | 3.0 | 14.3 | 4.1 | 2.6 |
| base, after | 3.4 | 4.4 | 5.0 | 3.1 |
| plain, after | 5.1 | 21.4 | 5.9 | 2.6 |
| safegrad, after | 3.0 | 13.7 | 4.1 | 2.6 |

- Before: Korean rise +16.5 points (p = 6.94e-08), SafeGrad against plain -7.1 (p = 0.0072). E1 holds.
- After: Korean rise +17.0 points (p = 3.67e-08), SafeGrad against plain -7.7 (p = 0.00434). E1 holds.

### Tiny Aya Earth, the readings that were not registered, before and after

These are the readings under "Added after the runs were read" in `earth/checks.md`, repeated on the re-generated answers. They were added to this file after the Earth re-generation had been read. They are not registered and decide nothing. Changes in points with the exact McNemar p-value.

- Korean strict attack success, SafeGrad against base: +5.8 (p = 0.008) before, +7.6 (p < 0.001) after.
- Korean counting only the guard's Unsafe class, base, plain and SafeGrad: 6.9%, 16.2%, 9.3% before, 3.4%, 16.2%, 8.2% after. Plain against base +9.3 (p < 0.001) before, +12.7 (p < 0.001) after. SafeGrad against plain -6.9 (p < 0.001) before, -7.9 (p < 0.001) after. SafeGrad against base +2.4 (p = 0.189) before, +4.8 (p = 0.004) after.
- Row C. SafeGrad's Swahili refusal rate on Earth: 45.0% before, 44.3% after. Against Earth plain +8.6 (p < 0.001) before, +8.6 (p < 0.001) after. Against Earth base +1.7 (p = 0.500) before, +0.3 (p = 1.000) after. Against Global base -5.8 (p = 0.008) before, -6.9 (p < 0.001) after. Earth base against Global base -7.6 (p < 0.001) before, -7.2 (p < 0.001) after.

Korean attack answers that reached the 256-token cap, and the strict flags on them.

| Arm | Korean answers | Reached the cap | Flagged among those, before | after | Flagged among the complete |
|---|---:|---:|---:|---:|---:|
| earth-base | 291 | 146 | 29 | 21 | 2 |
| earth-plain | 291 | 17 | 9 | 8 | 53 |
| earth-safegrad | 291 | 52 | 16 | 13 | 32 |
