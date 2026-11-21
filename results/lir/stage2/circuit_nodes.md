### Refusal circuit at five node counts (seed 42)

How many of the base model's top nodes are still in the top set, plain / SafeGrad, out of 612 nodes.

| Nodes | en | ko | zh | sw | By chance | N1 | N2 |
|---:|---|---|---|---|---:|---|---|
| 50 | 46 / 45 | 39 / 43 | 39 / 44 | 42 / 45 | 4 | no | no |
| 100 | 89 / 94 | 74 / 80 | 80 / 87 | 83 / 89 | 16 | yes | yes |
| 200 | 181 / 185 | 156 / 170 | 164 / 175 | 161 / 178 | 65 | yes | yes |
| 300 | 265 / 280 | 240 / 260 | 250 / 260 | 253 / 272 | 147 | yes | yes |
| 400 | 357 / 370 | 344 / 349 | 340 / 345 | 340 / 361 | 261 | no | yes |

- N1, under plain Korean keeps the fewest base nodes (a tie does not count): fails at 50, 400.
- N2, SafeGrad keeps at least as many base nodes as plain in every language: fails at 50.

### Share of the contrast the top nodes recover (base model, 64 held-out pairs)

| Nodes | en | ko | zh | sw |
|---:|---:|---:|---:|---:|
| 50 | 3% | 0% | 0% | 8% |
| 100 | 18% | 24% | 6% | 7% |
| 200 | 65% | 91% | 26% | 40% |
| 300 | 89% | 100% | 60% | 100% |
| 400 | 97% | 100% | 88% | 100% |

CircuitKIT's rounding: 200 keeps 199 nodes.

CircuitKIT reports the share clipped to the range 0% to 100%. Before clipping: ko -3% at 50; ko 111% at 300; ko 110% at 400; zh -1% at 50; sw 107% at 300; sw 116% at 400. The share is (top nodes - corrupted run) / (whole model - corrupted run): over 100% means the top nodes alone score above the whole model, below 0% means they score below the corrupted run.

No pass rule. A circuit that recovers little of the contrast holds little of the mechanism, so the comparison above says less at that size.
