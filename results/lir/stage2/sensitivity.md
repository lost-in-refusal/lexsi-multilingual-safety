### The plain fine-tune with the learning rate and the number of epochs halved and doubled (S1, S2)

Plain LoRA at seed 42 on the usual 1,800 rows, with one training setting changed in each of four runs and everything else as in the plain run. Section 14h of the preregistration fixed the reading before the runs. One run per setting, one seed. Answers capped at 256 tokens on the attack set and XSTest.

| Run | Learning rate | Epochs | Optimizer steps | Training loss | Korean strict attack success | Rise over base | S1 row | Korean rise minus English rise | Task score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| our settings (`plain`) | 0.0002 | 3 | 171 | 1.425 | 19.2 | +14.1 (p < 0.001) |  | +14.1 [+8.6, +19.6] | 5.94 |
| learning rate halved (`plain-lr-half`) | 0.0001 | 3 | 171 | 1.503 | 18.2 | +13.1 (p < 0.001) | 1 | +11.0 [+5.5, +16.5] | 5.71 |
| learning rate doubled (`plain-lr-double`) | 0.0004 | 3 | 171 | 1.324 | 21.6 | +16.5 (p < 0.001) | 1 | +13.4 [+8.2, +18.9] | 5.94 |
| half as many epochs (`plain-ep-half`) | 0.0002 | 1.5 | 86 | 1.551 | 26.1 | +21.0 (p < 0.001) | 1 | +16.2 [+10.7, +21.6] | 5.67 |
| twice as many epochs (`plain-ep-double`) | 0.0002 | 6 | 342 | 1.171 | 16.5 | +11.3 (p < 0.001) | 1 | +10.0 [+4.8, +15.1] | 5.97 |

Rates in %, differences in points; base is 5.2% in Korean. The bar is 6.36 points, half the average Korean rise of the three plain seeds. Intervals are 95%, from a bootstrap over the 291 prompt families. The first row is the plain run with our settings, shown for comparison; it is not one of the four runs.

- **S1. Rows of the K1 table, in the order above: 1, 1, 1, 1. With the learning rate and the number of epochs each halved and each doubled, plain fine-tuning raised Korean attack success significantly and by at least the bar in all four runs: a drift of this size does not depend on our choice of these two settings.**
  - Learning rate halved: plain fine-tuning raised Korean attack success significantly and by at least the bar (+13.1 points, p < 0.001).
  - Learning rate doubled: plain fine-tuning raised Korean attack success significantly and by at least the bar (+16.5 points, p < 0.001).
  - Half as many epochs: plain fine-tuning raised Korean attack success significantly and by at least the bar (+21.0 points, p < 0.001).
  - Twice as many epochs: plain fine-tuning raised Korean attack success significantly and by at least the bar (+11.3 points, p < 0.001).
- **S2. Korean rose more than English at all four settings.**
  - Learning rate halved: +11.0 [+5.5, +16.5]; Korean rose more than English at this setting.
  - Learning rate doubled: +13.4 [+8.2, +18.9]; Korean rose more than English at this setting.
  - Half as many epochs: +16.2 [+10.7, +21.6]; Korean rose more than English at this setting.
  - Twice as many epochs: +10.0 [+4.8, +15.1]; Korean rose more than English at this setting.
- Beside, no rule. Against the plain run with our settings at the same seed, strict attack success: learning rate halved en +2.1 (p = 0.031), ko -1.0 (p = 0.736), zh -3.1 (p = 0.049), sw -0.7 (p = 0.824); learning rate doubled en +3.1 (p = 0.012), ko +2.4 (p = 0.296), zh +2.4 (p = 0.092), sw -0.3 (p = 1.000); half as many epochs en +4.8 (p < 0.001), ko +6.9 (p = 0.003), zh +0.3 (p = 1.000), sw +3.4 (p = 0.041); twice as many epochs en +1.4 (p = 0.344), ko -2.7 (p = 0.243), zh -1.7 (p = 0.359), sw -1.0 (p = 0.664). The three plain seeds with our settings rose 14.1, 10.3, 13.7 points in Korean, so a difference of a few points from the plain run is within what the seed alone changes.
- Beside, no rule. Against base in the other languages: learning rate halved en +2.1 (p = 0.238), zh +1.7 (p = 0.424), sw +0.7 (p = 0.832); learning rate doubled en +3.1 (p = 0.064), zh +7.2 (p < 0.001), sw +1.0 (p = 0.690); half as many epochs en +4.8 (p = 0.004), zh +5.2 (p = 0.011), sw +4.8 (p = 0.020); twice as many epochs en +1.4 (p = 0.481), zh +3.1 (p = 0.093), sw +0.3 (p = 1.000).
- Beside, no rule. Swahili refusal rate against base: learning rate halved -7.9 (p = 0.001); learning rate doubled -10.7 (p < 0.001); half as many epochs -18.9 (p < 0.001); twice as many epochs -7.6 (p = 0.001). Korean on the guard's Unsafe class alone: learning rate halved 12.7%; learning rate doubled 14.4%; half as many epochs 20.3%; twice as many epochs 10.0% (base 2.7%, plain 14.1%).
- Beside, no rule. Over-refusal of safe prompts (XSTest), against base: learning rate halved en 6.4% (-3.2, p = 0.039), ko 6.0% (-1.6, p = 0.424), zh 5.6% (-3.2, p = 0.057), sw 3.6% (-4.0, p = 0.002); learning rate doubled en 6.8% (-2.8, p = 0.118), ko 4.0% (-3.6, p = 0.012), zh 3.6% (-5.2, p = 0.002), sw 3.2% (-4.4, p < 0.001); half as many epochs en 5.6% (-4.0, p = 0.006), ko 5.6% (-2.0, p = 0.302), zh 4.0% (-4.8, p = 0.002), sw 2.8% (-4.8, p < 0.001); twice as many epochs en 7.2% (-2.4, p = 0.070), ko 7.2% (-0.4, p = 1.000), zh 5.2% (-3.6, p = 0.064), sw 4.4% (-3.2, p = 0.057).
- Beside, no rule. Task score (1 to 7): 6.32 for base and 5.94 for plain with our settings. learning rate halved 5.71 (against base -0.61 [-0.77, -0.44], against plain -0.22 [-0.36, -0.09]); learning rate doubled 5.94 (against base -0.39 [-0.52, -0.25], against plain -0.00 [-0.10, +0.10]); half as many epochs 5.67 (against base -0.66 [-0.82, -0.48], against plain -0.27 [-0.41, -0.13]); twice as many epochs 5.97 (against base -0.35 [-0.48, -0.23], against plain +0.03 [-0.10, +0.16]).

A run that drifts less may also have learned the task less. The training loss and the task score are in the table for that reason, and a smaller drift at a weaker setting is not written as a fix.

Run records: `plain-lr-half` 171 steps in 76 minutes, commit `3b67c86`; `plain-lr-double` 171 steps in 75 minutes, commit `a105fdd`; `plain-ep-half` 86 steps in 39 minutes, commit `a105fdd`; `plain-ep-double` 342 steps in 150 minutes, commit `a105fdd`. Each record was checked to differ from the plain run's in the learning rate or the number of epochs only.
