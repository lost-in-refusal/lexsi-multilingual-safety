### Circuit-guided mixed-precision quantization of the base model (section 15: Q0 to Q3)

Tiny Aya Global with no adapter, quantized through CircuitKIT with optimum-quanto, weights only and with no calibration: uniform 4-bit, the circuit plan (3 layers at 8 bits, 27 at 4 bits and 6 at 2 bits, the tier size the gate chose), and three random plans of the same shape, all at 4.0 bits on average over the quantized weights, beside the bfloat16 model run through the same command. The plans come from the stage 1 circuit scores, fixed before any run. **What was evaluated is the quantized weights written out as ordinary numbers: the same function, with no memory saved. Sizes are arithmetic.** Strict attack success over the four languages decides Q0 to Q2: each of the 291 prompt families counts with its mean over English, Korean, Chinese and Swahili, and intervals are 95% from a bootstrap over the families. One guard judges everything, and no person has checked it on these answers. Answers are capped at 256 tokens.

**No protection by the circuit plan is claimed.**

- **Q0, the damage (uniform 4-bit minus bfloat16): +0.2 [-1.2, +1.5] points, includes zero. Uniform 4-bit quantization did not raise attack success on the base model detectably.**
- **Q1 and Q2: Uniform 4-bit quantization did not raise attack success detectably, so there was no rise for a mixed plan to prevent: Q1 and Q2 are not read, and their figures are shown with no rule.** The circuit plan minus uniform 4-bit: +3.0 [+1.1, +5.0]. The circuit plan minus the random plans: -5.8 [-7.6, -3.9].
- **Q3, the task score (the circuit plan minus uniform 4-bit, 300 task prompts): -1.31 [-1.52, -1.10], below zero. The circuit plan scored below uniform 4-bit on the task, so a lower attack success under it cannot be told apart from a weaker model.**

| Language | `bf16` | `int4` | `circuit` | `random-0` | `random-2` | `random-4` | Uniform 4-bit minus bfloat16 | Circuit minus uniform 4-bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| English | 7.2 | 6.2 | 8.9 | 9.3 | 14.4 | 7.2 | -1.0 (p = 0.508) | +2.7 (p = 0.169) |
| Korean | 5.2 | 7.9 | 11.7 | 27.8 | 28.9 | 9.6 | +2.7 (p = 0.057) | +3.8 (p = 0.080) |
| Chinese | 9.3 | 8.2 | 13.4 | 26.1 | 22.7 | 10.3 | -1.0 (p = 0.648) | +5.2 (p = 0.017) |
| Swahili | 5.5 | 5.5 | 5.8 | 16.8 | 7.9 | 7.6 | +0.0 (p = 1.000) | +0.3 (p = 1.000) |

Strict attack success in %, differences in points with exact McNemar tests on the same prompts. Only Q0 to Q3 have a rule. The rows by language are shown beside them, not adjusted for their number, and no sentence is built on one language.

Against the bfloat16 model, four languages together, with no rule. Strict attack success: `int4` +0.2 [-1.2, +1.5]; `circuit` +3.2 [+1.0, +5.3]; `random-0` +13.2 [+10.7, +15.8]; `random-2` +11.7 [+9.4, +14.0]; `random-4` +1.9 [+0.1, +3.7]. Refusal rate on the attack prompts: `int4` +1.3 [-0.4, +3.1]; `circuit` +4.7 [+1.6, +8.0]; `random-0` +2.6 [-1.0, +6.4]; `random-2` -8.5 [-11.4, -5.5]; `random-4` +6.8 [+4.1, +9.5]. The guard's Unsafe class alone: `int4` +0.9 [-0.2, +2.1]; `circuit` +2.9 [+1.5, +4.5]; `random-0` +8.2 [+6.2, +10.2]; `random-2` +9.3 [+7.2, +11.3]; `random-4` +1.6 [+0.2, +3.1].

Each mixed plan against uniform 4-bit, strict attack success, four languages together, with no rule: `circuit` +3.0 [+1.1, +5.0]; `random-0` +13.1 [+10.7, +15.6]; `random-2` +11.5 [+9.3, +13.7]; `random-4` +1.7 [+0.0, +3.4].

Task score (1 to 7, 300 held-out task prompts, one judge model): `bf16` 6.35; `int4` 6.09; `circuit` 4.78; `random-0` 3.69; `random-2` 4.00; `random-4` 4.15. Against the bfloat16 model: `int4` -0.25 [-0.38, -0.14]; `circuit` -1.57 [-1.78, -1.36]; `random-0` -2.66 [-2.91, -2.42]; `random-2` -2.35 [-2.58, -2.11]; `random-4` -2.20 [-2.44, -1.95].

Over-refusal on the safe XSTest prompts (%): English `bf16` 9.6, `int4` 8.4, `circuit` 18.8, `random-0` 17.2, `random-2` 12.4, `random-4` 12.4; Korean `bf16` 7.6, `int4` 10.0, `circuit` 28.4, `random-0` 49.6, `random-2` 25.6, `random-4` 29.2; Chinese `bf16` 8.8, `int4` 8.4, `circuit` 20.8, `random-0` 27.2, `random-2` 15.6, `random-4` 21.2; Swahili `bf16` 7.6, `int4` 8.4, `circuit` 10.0, `random-0` 16.9, `random-2` 12.4, `random-4` 12.0.

Share of attack answers cut at the 256-token cap (%): `bf16` 30, `int4` 31, `circuit` 25, `random-0` 48, `random-2` 51, `random-4` 33.

The gate (perplexity, times the bfloat16 model's): `bf16` 1.00, `int4` 1.04, `circuit@7-15-14` 2.56, `random-0@7-15-14` 3.84, `random-2@7-15-14` 8.35, `random-4@7-15-14` 3.67, `circuit@5-21-10` 1.72, `random-0@5-21-10` 1.98, `random-2@5-21-10` 2.91, `random-4@5-21-10` 2.15, `circuit@3-27-6` 1.26, `random-0@3-27-6` 1.47, `random-2@3-27-6` 1.45, `random-4@3-27-6` 1.44. The plans with 3 layers at 8 bits, 27 at 4 bits and 6 at 2 bits pass the gate and are the ones evaluated.

Size by arithmetic, not measured: 2,812,280,832 weights are quantized and 536,946,688 are not. At 16 bits each the model is 6.70 GB; with the quantized weights at 4.0 bits on average it is 2.48 GB, for uniform 4-bit and for every mixed plan alike. That leaves out the scale each row carries and, in the 2- and 4-bit layers, the scale and shift each group of weights carries; with them the average is 4.00 bits for `int4`, 4.04 bits for `circuit`, 4.04 bits for `random-0`, 4.04 bits for `random-2`, 4.04 bits for `random-4`.

Not done: Global-MMLU-Lite and IFEval; a person's check of the guard on these answers; generating capped answers again at 1,024 tokens; any fine-tuned model; a packed model's speed or memory.
