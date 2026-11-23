### Answer length and the task score

Task score (1 to 7, Qwen3-14B judge) on the 300 held-out Bactrian-X prompts, for plain and SafeGrad at seeds 42, 1 and 2 against the base model. Each fine-tuned answer is grouped by its length in characters relative to the base model's answer to the same prompt: about the same (80% to 125%), shorter, under half, or longer. Section 12h of the preregistration fixed the groups and the reading before this was computed. The groups are formed after generation, so this locates the drop; it does not show what causes it.

Task score minus the base model's on the same prompt, by length group (95% interval over prompts):

| Method | About the same length | Shorter (50% to 80%) | Under half | Longer |
|---|---|---|---|---|
| Plain LoRA | -0.39 [-0.81, -0.05], 84 answers | -0.42 [-0.62, -0.21], 176 answers | -0.36 [-0.50, -0.21], 611 answers | -2.46 [-3.57, -1.35], 29 answers |
| SafeGrad LoRA | -0.67 [-1.10, -0.30], 83 answers | -0.45 [-0.67, -0.23], 184 answers | -0.41 [-0.53, -0.27], 602 answers | -2.36 [-3.36, -1.38], 31 answers |

**Plain LoRA: The fine-tuned models score lower even where their answers are about as long as the base model's, so the drop is not only a matter of length.**

**SafeGrad LoRA: The fine-tuned models score lower even where their answers are about as long as the base model's, so the drop is not only a matter of length.**

#### Description, with no rule

| Method | Language | Task score, base to fine-tuned | Median characters: base, fine-tuned, reference | Accuracy | Instruction following | Coherence | Fluency |
|---|---|---|---|---:|---:|---:|---:|
| Plain LoRA | en | 6.61 to 6.37 | 1619, 521, 549 | -0.59 | -0.18 | -0.16 | -0.03 |
| Plain LoRA | ko | 6.10 to 5.51 | 928, 276, 298 | -1.21 | -0.36 | -0.59 | -0.21 |
| Plain LoRA | zh | 6.25 to 5.79 | 761, 190, 238 | -1.09 | -0.36 | -0.37 | -0.03 |
| SafeGrad LoRA | en | 6.61 to 6.31 | 1619, 508, 549 | -0.72 | -0.23 | -0.20 | -0.07 |
| SafeGrad LoRA | ko | 6.10 to 5.49 | 928, 291, 298 | -1.22 | -0.43 | -0.65 | -0.14 |
| SafeGrad LoRA | zh | 6.25 to 5.74 | 761, 186, 238 | -1.07 | -0.40 | -0.46 | -0.10 |

Sub-score columns are the change against the base model, averaged over the three seeds.

Within the base model's own answers, the rank correlation between answer length and task score is +0.06 in en, +0.18 in ko, +0.15 in zh.
