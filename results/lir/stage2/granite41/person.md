### The Korean rise on the model outside the Tiny Aya family, judged by a person (section 16d)

Labeller: mingyu. 150 of the 291 Korean attack prompts, drawn with a fixed seed, with the answers of the base run and of plain at seed 42 of section 16 (capped at 256 tokens, as the guard read them), arm hidden. Not blind: the two models' answers can differ in style, and the labeller knew the direction we expect and the guard's result. Section 16d of the preregistration fixed the test before the sheet was made.

On the 150 prompts where both answers got a yes or a no: 83 base answers and 84 plain answers were marked unsafe. Unsafe only under plain: 16; only under base: 15; exact McNemar p = 1.000.

**The check does not pass. On the model outside the Tiny Aya family, a person's reading did not confirm the Korean rise: the difference did not reach significance.**

With every unsure counted as unsafe: 83 against 84 (p = 1.000). With every unsure counted as safe: 83 against 84 (p = 1.000).

The guard (strict) agrees with these labels on 174 of 300 answers; it passed 98 the labeller marked unsafe and flagged 28 the labeller marked safe. On these prompts the guard's strict count is 31 for base and 66 for plain.

By the guard's three classes: of the 63 answers it called Unsafe, the labeller marked 42 unsafe, 21 safe and 0 unsure; of the 34 answers it called Controversial, the labeller marked 27 unsafe, 7 safe and 0 unsure; of the 203 answers it called Safe, the labeller marked 98 unsafe, 105 safe and 0 unsure.

One labeller, an author. This covers the Korean rise at seed 42 on this model only: not Tiny Aya, where K2 and K3 did not reach significance and stand as they were read, not the comparison with English, and not SafeGrad's repair.
