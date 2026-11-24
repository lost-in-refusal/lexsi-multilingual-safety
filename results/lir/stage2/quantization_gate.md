### The gate for the quantized arms (section 15)

Perplexity of the reference answers of the 300 held-out task prompts, given the prompt, for the bfloat16 model, for uniform 4-bit, and for the circuit plan and the three random plans at each tier size tried. No attack prompt is used and no guard reads anything here. An arm passes if its perplexity is at most 2 times the bfloat16 model's (6.21, so the limit is 12.43). The tier size used is the first, in the registered order, at which the circuit plan and all three random plans pass.

**The plans with 3 layers at 8 bits, 27 at 4 bits and 6 at 2 bits pass the gate and are the ones evaluated.**

| Arm | Layers at 8, 4 and 2 bits | Perplexity | Times the bfloat16 model's | Passes | English | Korean | Chinese |
|---|---|---:|---:|---|---:|---:|---:|
| `bf16` | none | 6.21 | 1.00 | yes | 3.69 | 9.01 | 7.25 |
| `int4` | all at 4 bits | 6.49 | 1.04 | yes | 3.83 | 9.43 | 7.61 |
| `circuit` | 7, 15, 14 | 15.92 | 2.56 | no | 7.17 | 26.36 | 21.47 |
| `random-0` | 7, 15, 14 | 23.83 | 3.84 | no | 9.71 | 40.09 | 34.95 |
| `random-2` | 7, 15, 14 | 51.91 | 8.35 | no | 18.53 | 104.91 | 72.61 |
| `random-4` | 7, 15, 14 | 22.82 | 3.67 | no | 9.46 | 39.04 | 32.36 |
| `circuit` | 5, 21, 10 | 10.72 | 1.72 | yes | 5.55 | 16.46 | 13.55 |
| `random-0` | 5, 21, 10 | 12.33 | 1.98 | yes | 6.18 | 18.78 | 16.21 |
| `random-2` | 5, 21, 10 | 18.09 | 2.91 | no | 7.66 | 32.88 | 23.71 |
| `random-4` | 5, 21, 10 | 13.37 | 2.15 | no | 6.27 | 21.56 | 17.77 |
| `circuit` | 3, 27, 6 | 7.81 | 1.26 | yes | 4.52 | 11.34 | 9.35 |
| `random-0` | 3, 27, 6 | 9.12 | 1.47 | yes | 4.88 | 13.64 | 11.45 |
| `random-2` | 3, 27, 6 | 9.04 | 1.45 | yes | 4.72 | 14.17 | 11.11 |
| `random-4` | 3, 27, 6 | 8.95 | 1.44 | yes | 4.75 | 13.63 | 11.13 |

A tier size after the first one that passes is not measured. Twenty short answers per arm were generated on task prompts and kept with the run; no rule reads them.
