### Qwen2.5-3B, read again after re-generation

Every attack answer of the seven runs that reached the 256-token cap, generated again at 1,024 tokens by the model that wrote it and judged by the same guard. The other answers and their verdicts are those of the first reading. It decides nothing: G1 to G3 stand as first read, and this reading is given beside.

**G1, plain minus base: the same on the capped answers and after re-generation.** On the capped answers +21.9 [+18.7, +25.1]; after re-generation +22.5 [+19.2, +25.8].

**G2, the Korean rise minus the English rise: the same on the capped answers and after re-generation.** On the capped answers +6.6 [+1.1, +12.3]; after re-generation +6.6 [+1.0, +12.3].

**G3, SafeGrad minus plain: the same on the capped answers and after re-generation.** On the capped answers -19.1 [-22.1, -16.1]; after re-generation -19.2 [-22.3, -16.2].

After re-generation, with no rule. Base rate (%): en 3.1, ko 5.5, zh 4.1, sw 9.3. Plain against base: en +20.4 [+16.3, +24.6], ko +27.0 [+22.3, +32.0], zh +19.9 [+15.2, +24.5], sw +14.4 [+10.5, +18.6]. SafeGrad against base: en +0.9 [-0.9, +2.7], ko +6.1 [+3.2, +9.2], zh +2.6 [-0.2, +5.4], sw +8.2 [+4.7, +11.9].

| Run | Generated again | en | ko | zh | sw | Still cut at 1,024 tokens: en, ko, zh, sw |
|---|---:|---:|---:|---:|---:|---|
| `qwen25-base` | 348 | 53 | 60 | 26 | 209 | 1, 2, 1, 141 |
| `qwen25-plain-s42` | 216 | 15 | 57 | 12 | 132 | 1, 9, 2, 63 |
| `qwen25-safegrad-s42` | 261 | 32 | 35 | 11 | 183 | 2, 5, 2, 123 |
| `qwen25-plain-s1` | 237 | 17 | 50 | 13 | 157 | 2, 8, 2, 91 |
| `qwen25-safegrad-s1` | 243 | 30 | 29 | 12 | 172 | 1, 6, 2, 132 |
| `qwen25-plain-s2` | 240 | 16 | 57 | 14 | 153 | 1, 16, 3, 82 |
| `qwen25-safegrad-s2` | 240 | 26 | 36 | 17 | 161 | 0, 7, 1, 110 |

Batches are formed from the prompts that are generated again, so an answer generated again is often another answer from its first character and not a continuation. The verdicts are the guard's; no person has read these answers.
