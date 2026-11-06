# Lexsi tool feedback

Bugs and gaps we hit while chaining CuratorKIT, SafeTune, AuditKIT and CircuitKIT on Tiny Aya Global. Each item is also a GitHub issue labelled `lexsi-feedback` in the team repo. Versions: SafeTune `3901a9f` (0.1.12), AuditKIT `71316da`, CuratorKIT `dd9e835`, CircuitKIT `e71d5e1`.

## SafeTune

1. **`SafeGradHFTrainer` refuses gradient accumulation.** Since 0.1.12 it raises if `gradient_accumulation_steps > 1`, because surgery runs per micro-batch. On a single laptop or Colab GPU, an effective batch of 32 with 1k-token sequences and a 262k vocabulary cannot fit as one micro-batch, so the trainer is effectively unusable at paper-scale batch sizes. Workaround in `lir/train.py`: subclass, accumulate the task gradient, and run `_alignment_gradient` + `_apply_global_surgery` once on the last micro-batch of each optimizer step (`accelerator.sync_gradients`). Suggest shipping that as an option.

2. **`safegrad_kl_alignment_loss` has no attention mask.** The KL is summed over every position, including right-padding, so padded alignment batches add KL terms on pad tokens and the loss scale depends on padding length. We pass only unpadded positions (`logits[attention_mask.bool()]`) and divide by batch size. Suggest an optional `attention_mask` / `labels` argument.

3. **Silent fallback to plain fine-tuning.** `SafeGradHFTrainer._alignment_gradient` catches every exception, logs a warning and skips surgery for that step. A shape or OOM error in the alignment pass therefore turns a SafeGrad run into ordinary fine-tuning with no error. We count surgery steps and assert one per optimizer step. Suggest a strict mode that raises, plus a counter in the trainer state.

## CuratorKIT

4. **No multilingual safety screening.** `ToxicityGate` uses Detoxify, whose multilingual model covers en/fr/es/it/pt/tr/ru but not Korean, Chinese or Swahili. We screened Bactrian-X with Qwen3Guard-Gen-4B outside CuratorKIT. A pluggable guard-model gate (prompt + response moderation) would fit multilingual post-training data.

## AuditKIT

5. **Paired bootstrap returns a p-value but no interval.** `RunComparison.significance()` reports `delta` and `p_value`; a 95% CI on the delta is what a results table needs. We compute it separately in `lir/metrics.py`.

## CircuitKIT

6. **Dependency cap.** CircuitKIT requires `huggingface-hub<2` and `transformer-lens==3.8.0`; a default install of the other three tools pulls hub 2.x, so one environment for all four needs a manual downgrade. We used a second venv (`.venv-circuit`).

_Further items are added as the analysis runs._
