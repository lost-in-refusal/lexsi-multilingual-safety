### 4-bit loading on three seeds, with the adapter loaded the same way in both precisions (section 14f: P1 and P2)

Plain and SafeGrad at seeds 42, 1 and 2, each evaluated on one NVIDIA L4 in bfloat16 with the adapter left unmerged and under NF4 (bitsandbytes 4-bit, double quantization, compute in bfloat16), where the adapter is unmerged too. So the two arms of a comparison differ in the precision of the base weights and in nothing else that was set. Strict attack success counts the guard's Unsafe and Controversial classes, on 291 prompts per language. A pooled figure averages each prompt family over the runs it names, then bootstraps the 291 families (95% interval). Per-seed tests are exact McNemar, two-sided; * = p < 0.05. Every comparison is between two runs on the GPU. The seed 42 NF4 arms are those of section 12.2, read there on Nov 22.

| Job | Upload | Commit | GPU | Finished (UTC) |
|---|---|---|---|---|
| control | `20261122-205736-330ebba` | `330ebba` | NVIDIA L4, 620.70.06, 23034 MiB | 20261122-220142 |
| nf4 | `20261122-215110-649ce27` | `649ce27` | NVIDIA L4, 620.70.06, 23034 MiB | 20261123-000638 |
| nf4-s1 | `20261125-010715-553e4b6` | `553e4b6` | NVIDIA L4, 620.70.06, 23034 MiB | 20261125-021838 |
| nf4-s2 | `20261125-011929-553e4b6` | `553e4b6` | NVIDIA L4, 620.70.06, 23034 MiB | 20261125-023104 |
| unmerged | `20261125-013527-553e4b6` | `553e4b6` | NVIDIA L4, 620.70.06, 23034 MiB | 20261125-024542 |
| unmerged-s1 | `20261125-014809-553e4b6` | `553e4b6` | NVIDIA L4, 620.70.06, 23034 MiB | 20261125-025816 |
| unmerged-s2 | `20261125-010715-553e4b6` | `553e4b6` | NVIDIA L4, 620.70.06, 23034 MiB | 20261125-021656 |

The library versions are the same in every job.

**P1, plain fine-tuned. With the adapter loaded the same way in both, 4-bit loading raised attack success on the plain fine-tuned model.** NF4 minus bfloat16, over the seeds (42, 1, 2) and the four languages: +2.9 [+1.7, +4.2] points. Beside, seeds 1 and 2 alone, which were not seen before this section: +2.5 [+1.2, +3.8].

**P1, SafeGrad. With the adapter loaded the same way in both, 4-bit loading raised attack success on the SafeGrad model.** NF4 minus bfloat16, over the seeds (42, 1, 2) and the four languages: +1.9 [+0.8, +3.0] points. Beside, seeds 1 and 2 alone, which were not seen before this section: +2.3 [+1.0, +3.6].

Strict attack success (%), bfloat16 with the adapter unmerged to NF4. The pooled rows and the per-seed tests carry no rule and are not adjusted for their number.

| Model | en | ko | zh | sw |
|---|---|---|---|---|
| Plain fine-tuned, seeds pooled | +1.1 [-0.7, +3.0] | +3.2 [+0.1, +6.4] | +2.5 [+0.7, +4.5] | +4.8 [+2.4, +7.3] |
| Plain fine-tuned, seed 42 | 6.5 to 10.0 (+3.4*), p = 0.013 | 20.6 to 22.7 (+2.1), p = 0.405 | 12.0 to 17.5 (+5.5*), p = 0.002 | 7.9 to 12.0 (+4.1*), p = 0.036 |
| Plain fine-tuned, seed 1 | 7.6 to 7.2 (-0.3), p = 1.000 | 15.5 to 19.6 (+4.1), p = 0.081 | 12.0 to 13.4 (+1.4), p = 0.424 | 5.5 to 10.7 (+5.2*), p = 0.003 |
| Plain fine-tuned, seed 2 | 9.3 to 9.6 (+0.3), p = 1.000 | 19.6 to 23.0 (+3.4), p = 0.154 | 14.1 to 14.8 (+0.7), p = 0.774 | 6.9 to 12.0 (+5.2*), p = 0.006 |
| Safegrad, seeds pooled | +2.6 [+0.7, +4.7] | +2.6 [+0.3, +4.9] | +1.1 [-0.8, +3.2] | +1.1 [-1.3, +3.7] |
| Safegrad, seed 42 | 4.8 to 6.9 (+2.1), p = 0.180 | 9.6 to 11.0 (+1.4), p = 0.481 | 9.6 to 10.3 (+0.7), p = 0.804 | 5.8 to 6.2 (+0.3), p = 1.000 |
| Safegrad, seed 1 | 3.4 to 7.2 (+3.8*), p = 0.007 | 8.9 to 12.4 (+3.4*), p = 0.041 | 7.9 to 10.0 (+2.1), p = 0.180 | 3.8 to 7.2 (+3.4), p = 0.052 |
| Safegrad, seed 2 | 4.1 to 6.2 (+2.1), p = 0.109 | 9.3 to 12.4 (+3.1), p = 0.064 | 8.6 to 9.3 (+0.7), p = 0.824 | 6.5 to 6.2 (-0.3), p = 1.000 |

The seed 42 rows are the reading the review note of Nov 22 under 12.2 describes: NF4 against the unmerged bfloat16 arms.

**P2: pass. Under 4-bit loading, SafeGrad stayed below plain on Korean in all three seeds.** Korean, SafeGrad minus plain under NF4: seed 42 -11.7 (p < 0.001); seed 1 -7.2 (p < 0.001); seed 2 -10.7 (p < 0.001); pooled -9.9 [-13.3, -6.5]. Four-language average under NF4, plain and SafeGrad: seed 42 15.5 and 8.6; seed 1 12.7 and 9.2; seed 2 14.9 and 8.5. The seed 42 figure was known before this section was registered. Beside, Korean pooled over seeds 1 and 2 alone: -8.9 [-12.4, -5.7].

Beside, with no rule. The same in bfloat16 with the adapter unmerged: seed 42 -11.0 (p < 0.001); seed 1 -6.5 (p < 0.001); seed 2 -10.3 (p < 0.001); pooled -9.3 [-12.3, -6.4]. The Korean difference under NF4 minus the Korean difference in bfloat16: -0.6 [-4.4, +3.3]; above zero would be a smaller repair under NF4.

Merging, at seed 42, with no rule (bfloat16, adapter unmerged minus adapter merged): plain fine-tuned en -0.3 (p = 1.000), ko +0.7 (p = 0.727), zh -1.4 (p = 0.125), sw +1.0 (p = 0.508), four languages +0.0 [-0.8, +0.8]; SafeGrad en -0.3 (p = 1.000), ko -1.0 (p = 0.508), zh +0.3 (p = 1.000), sw -0.3 (p = 1.000), four languages -0.3 [-1.2, +0.4].

The base model has no adapter. Its change under NF4 was read in 12.2 on Nov 22 and is repeated here as description: en 5.5 to 11.0 (+5.5*), ko 5.5 to 7.9 (+2.4), zh 7.9 to 9.6 (+1.7), sw 5.2 to 7.2 (+2.1); four languages +2.9 [+1.5, +4.6].

Refusals on the safe prompts of XSTest (%), NF4 minus bfloat16 with the adapter unmerged, seeds pooled, no rule:

| Model | en | ko | zh | sw | Four languages |
|---|---|---|---|---|---|
| Plain fine-tuned | +0.4 [-1.1, +2.0] | -0.9 [-2.3, +0.3] | -1.2 [-2.8, +0.1] | -1.3 [-2.8, -0.3] | -0.8 [-1.4, -0.1] |
| Safegrad | -1.5 [-4.0, +0.9] | +0.3 [-1.3, +2.0] | -0.3 [-2.1, +1.6] | -0.3 [-1.9, +1.5] | -0.4 [-1.4, +0.5] |

Speed and memory of the generation step, from the job records, as description. A step named generate holds the model alone; an earlier step that also ran the guard is marked.

| Arm | Precision | Adapter | Attack generation (tokens per second) | Peak GPU memory (MiB) |
|---|---|---|---:|---:|
| `gpu-plain-nf4` | NF4 | unmerged | 115.2 | 20864 (with the guard) |
| `gpu-safegrad-nf4` | NF4 | unmerged | 145.5 | 18856 (with the guard) |
| `gpu-plain-s1-nf4` | NF4 | unmerged | 108.7 | 17878 |
| `gpu-safegrad-s1-nf4` | NF4 | unmerged | 141.6 | 17878 |
| `gpu-plain-s2-nf4` | NF4 | unmerged | 113.0 | 17878 |
| `gpu-safegrad-s2-nf4` | NF4 | unmerged | 146.2 | 17878 |
| `gpu-plain-unmerged` | bfloat16 | unmerged | 121.3 | 19962 |
| `gpu-safegrad-unmerged` | bfloat16 | unmerged | 141.8 | 19962 |
| `gpu-plain-s1-unmerged` | bfloat16 | unmerged | 113.9 | 19962 |
| `gpu-safegrad-s1-unmerged` | bfloat16 | unmerged | 141.7 | 19962 |
| `gpu-plain-s2-unmerged` | bfloat16 | unmerged | 119.0 | 19962 |
| `gpu-safegrad-s2-unmerged` | bfloat16 | unmerged | 142.9 | 19962 |

Not measured: an adapter merged into the weights and quantized afterwards, any other 4-bit format (GGUF among them), another GPU type, and the task score. The verdicts are the guard's; no person has read these answers.
