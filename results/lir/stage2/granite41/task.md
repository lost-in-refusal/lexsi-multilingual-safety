### The task score on the model outside the Tiny Aya family (section 16c: T1 and T2)

The 300 held-out task prompts answered by the base run and the six fine-tuned runs of section 16 of `ibm-granite/granite-4.1-3b`, at up to 1,024 new tokens, and scored from 1 to 7 by the task judge used everywhere (one judge model). The adapters are the ones section 16 evaluated. A difference averages each prompt over the three seeds, then bootstraps the 300 prompts (95% interval).

**T1. Plain fine-tuning lowered the task score on the model outside the Tiny Aya family.** Plain minus base: -0.47 [-0.61, -0.35].

**T2. No difference in task score between SafeGrad and plain fine-tuning was detected on the model outside the Tiny Aya family.** SafeGrad minus plain: -0.07 [-0.15, +0.00].

With no rule: SafeGrad minus base -0.55 [-0.68, -0.42]. Mean score: `granite41-base` 6.34, `granite41-plain-s42` 5.82, `granite41-safegrad-s42` 5.77, `granite41-plain-s1` 5.88, `granite41-safegrad-s1` 5.78, `granite41-plain-s2` 5.89, `granite41-safegrad-s2` 5.82.

One judge model scores the answers; no person has read them. The score can move with the length of an answer (section 13h), so it is not a clean measure of quality.
