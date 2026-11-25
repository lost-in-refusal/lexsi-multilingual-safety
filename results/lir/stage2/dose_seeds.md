### The lowest strengths of the update at three seeds, with the guard on harmless prompts (section 19b: S1, S2)

No training. The plain adapters of seeds 42, 1 and 2 with their update multiplied by 0.25 and 0.5, each on the attack set, XSTest's safe prompts and the task set on the laptop. Seed 42's attack and task results were known when this was registered (section 19). Rises are against base in points of the guard's strict reading, pooled over the three seeds with 95% intervals from a bootstrap over prompts; the task score is on the 1 to 7 scale.

| Strength | Korean rise | by seed (1, 2, 42) | English rise | Korean minus English | Task against base | Korean, safe prompts | English, safe prompts | Korean rise minus safe-prompt change |
|---|---|---|---|---|---|---|---|---|
| 0.25 | +4.0 [+1.5, +6.6] | +3.8, +4.5, +3.8 | -1.7 [-4.1, +0.6] | +5.7 [+2.3, +9.3] | -0.08 [-0.18, +0.02] | +0.8 [-1.7, +3.3] | +0.5 [-1.2, +2.1] | +3.2 [-0.2, +6.9] |
| 0.5 | +4.5 [+1.5, +7.4] | +3.8, +4.1, +5.5 | -2.2 [-4.7, +0.1] | +6.6 [+2.6, +10.7] | -0.25 [-0.36, -0.15] | +1.6 [-0.8, +4.1] | +2.8 [+0.7, +5.1] | +2.9 [-1.0, +6.8] |

- **S1. Over three seeds, at a quarter of the fine-tune's update Korean attack success had risen (+4.0 [+1.5, +6.6]) with no detected fall in the task score (-0.08 [-0.18, +0.02]).**
- **S2. At a quarter of the update, the Korean rise on attack prompts could not be told apart from the guard's change on harmless prompts (+3.2 [-0.2, +6.9]).**

Scaling the finished update is not an earlier point of training. Guard verdicts only; nobody has read these answers.
