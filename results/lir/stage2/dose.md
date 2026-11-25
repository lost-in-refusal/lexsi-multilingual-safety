### How much of the fine-tune's update it takes, and a random update of the same size (section 19: D1, D2)

No training. The seed 42 plain adapter with its update multiplied by 0.25, 0.5 and 0.75, and two random adapters of the same shape whose update is as large as plain's in every module. Each answered the attack set and the task set on the laptop, as the base model (strength 0) and the plain run (strength 1) did. One adapter, one seed. Rises are in points of strict attack success against base, with 95% intervals from a bootstrap over the 291 prompt families; the task score is on the 1 to 7 scale (base 6.32), with intervals over the 300 prompts.

| Arm | Korean rise | English rise | Korean minus English | Chinese rise | Swahili rise | Korean refusals | Task score | Task against base | Korean answers at the cap (of 291) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| update times 0.25 | +3.8 [+1.0, +6.5] | -1.4 [-3.8, +1.0] | +5.2 [+1.4, +9.3] | -1.0 [-3.4, +1.0] | -0.7 [-3.4, +2.1] | -2.1 [-5.2, +1.0] | 6.24 | -0.08 [-0.20, +0.02] | 52 |
| update times 0.5 | +5.5 [+2.1, +8.9] | -1.4 [-4.1, +1.0] | +6.9 [+2.7, +11.3] | +0.7 [-2.4, +3.8] | -0.3 [-3.8, +2.7] | -5.2 [-9.3, -1.0] | 6.08 | -0.24 [-0.35, -0.12] | 33 |
| update times 0.75 | +9.3 [+5.2, +13.4] | +1.0 [-1.4, +3.8] | +8.2 [+2.7, +13.7] | +1.4 [-1.7, +4.5] | +1.4 [-2.1, +4.8] | -6.2 [-10.3, -2.1] | 6.00 | -0.32 [-0.44, -0.19] | 22 |
| the plain run (update times 1) | +14.1 [+9.6, +18.9] | +0.0 [-2.7, +2.7] | +14.1 [+8.6, +19.6] | +4.8 [+1.4, +8.6] | +1.4 [-2.1, +4.8] | -12.4 [-16.5, -8.2] | 5.94 | -0.38 [-0.52, -0.25] | 24 |
| random update, draw 1 | +0.3 [-1.0, +1.7] | -0.3 [-1.7, +0.7] | +0.7 [-1.4, +2.7] | +0.3 [-1.7, +2.4] | -0.7 [-2.7, +1.4] | -1.0 [-3.4, +1.4] | 6.34 | +0.02 [-0.03, +0.08] | 103 |
| random update, draw 2 | -0.3 [-2.1, +1.4] | -1.4 [-3.1, +0.3] | +1.0 [-1.7, +3.8] | -2.1 [-4.1, -0.3] | -1.4 [-3.4, +0.7] | -0.7 [-2.7, +1.4] | 6.37 | +0.04 [-0.02, +0.12] | 97 |

- **D1. At 0.25 of the fine-tune's update Korean attack success had already risen (+3.8 [+1.0, +6.5]) with no detected fall in the task score (-0.08 [-0.20, +0.02]): the Korean rise does not need the damage to the task.**
- **D2. A random update of the same size did not raise Korean attack success (+0.3 [-1.0, +1.7] and -0.3 [-2.1, +1.4], against +14.1 [+9.6, +18.9] for the fine-tune's update): the rise is not explained by the size of the change alone.**

Scaling the finished update is not an earlier point of training, and a random update matched in size module by module is one meaning of "the same size". Guard verdicts only.

Update norms (Frobenius, over all modules): `plain-x025` 1.510; `plain-x050` 3.020; `plain-x075` 4.531; `random-r1` 6.041; `random-r2` 6.041; the plain adapter 6.041.
