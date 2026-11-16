# Lexsi tool feedback

Bugs and gaps we hit while chaining CuratorKIT, SafeTune, AuditKIT and CircuitKIT on Tiny Aya Global. Each item is also a GitHub issue labelled `lexsi-feedback` in the team repo. Versions: SafeTune `3901a9f` (0.1.12), AuditKIT `71316da`, CuratorKIT `dd9e835`, CircuitKIT `e71d5e1`.

## SafeTune

1. **`SafeGradHFTrainer` refuses gradient accumulation.** Since 0.1.12 it raises if `gradient_accumulation_steps > 1`, because surgery runs per micro-batch. On a single laptop or Colab GPU, an effective batch of 32 with 1k-token sequences and a 262k vocabulary cannot fit as one micro-batch, so the trainer is effectively unusable at paper-scale batch sizes. Workaround in `lir/train.py`: subclass, accumulate the task gradient, and run `_alignment_gradient` + `_apply_global_surgery` once on the last micro-batch of each optimizer step (`accelerator.sync_gradients`). Suggest shipping that as an option.

2. **`safegrad_kl_alignment_loss` has no attention mask.** The KL is summed over every position, including right-padding, so padded alignment batches add KL terms on pad tokens and the loss scale depends on padding length. We pass only unpadded positions (`logits[attention_mask.bool()]`) and divide by batch size. Suggest an optional `attention_mask` / `labels` argument.

3. **Silent fallback to plain fine-tuning.** `SafeGradHFTrainer._alignment_gradient` catches every exception, logs a warning and skips surgery for that step. A shape or OOM error in the alignment pass therefore turns a SafeGrad run into ordinary fine-tuning with no error. We count surgery steps and assert one per optimizer step. Suggest a strict mode that raises, plus a counter in the trainer state.

13. **`safegrad_kl_alignment_loss` is computed in the logits' dtype.** `log_softmax`/`softmax` run in whatever dtype the logits have. With a bf16 model (the usual case for SafeGrad on one GPU), the KL between two *identical* distributions comes out around 0.6 per sequence (Tiny Aya tokenizer, 262k vocabulary, ~400-token sequences) instead of ~1e-6 in fp32, so at the start of training the alignment gradient is rounding noise. We upcast to fp32 and evaluate the KL in 512-position chunks to bound memory. Suggest `.float()` inside the loss (and a chunked path for large vocabularies).

## CuratorKIT

4. **No multilingual safety screening.** `ToxicityGate` uses Detoxify, whose multilingual model covers en/fr/es/it/pt/tr/ru but not Korean, Chinese or Swahili. We screened Bactrian-X with Qwen3Guard-Gen-4B outside CuratorKIT. A pluggable guard-model gate (prompt + response moderation) would fit multilingual post-training data.

## AuditKIT

5. **Paired bootstrap returns a p-value but no interval.** `RunComparison.significance()` reports `delta` and `p_value`; a 95% CI on the delta is what a results table needs. We compute it separately in `lir/metrics.py`.

14. **The paired-bootstrap p-value is ~0.5 for any non-zero effect.** `_bootstrap.paired_bootstrap` resamples the raw differences and counts resamples with `|mean| >= |observed|`. The differences are not centred under the null, so the resampled means are centred on the observed effect and about half of them exceed it, whatever its size. On our data: Korean attack success rose 14.1 points over 291 paired prompts (95% CI 9.6–18.9, exact McNemar p = 6e-9) and Mandarin 4.8 points (CI 1.4–8.6, p = 0.016); `RunComparison.significance()` returned p = 0.53 for both. Fix: resample `d - mean(d)` (or report the bootstrap CI and check whether it contains 0). We report McNemar for rates and a centred bootstrap for scores, and keep AuditKIT's value in `results.json` for transparency.

## CircuitKIT

6. **Dependency cap.** CircuitKIT requires `huggingface-hub<2` and `transformer-lens==3.8.0`; a default install of the other three tools pulls hub 2.x, so one environment for all four needs a manual downgrade. We used a second venv (`.venv-circuit`).

7. **MLP node attribution multiplies mismatched quantities.** `backends/eap/attribute_node.py` (around line 170) records the MLP node's activation difference at `node.out_hook` (`hook_mlp_out`) but its gradient at `node.in_hook` (`hook_mlp_in`). Attention heads correctly use the same hook for both. On a small Cohere2 model the default node-level EAP-IG MLP score was -8.3e-05, while Δout·∂L/∂out is +4.7e-04; Δout·∂L/∂in reproduces CircuitKIT exactly. With `mlp_hook="post_act"` the scores match the correct product. The same pattern appears on the edge path in `eap_utils.py` (around lines 468/473; not verified). Present since the initial commit, so it affects default node-level runs on every model. We run with `mlp_hook="post_act"`.

8. **Attention mask ignores the padding side actually used.** `backends/eap/eap_utils.py` `tokenize_plus` (around line 175) tokenises with `padding_side=effective_padding_side` but then calls `get_attention_mask(model.tokenizer, tokens, True)`, which reads the tokenizer's own padding side. With NormalizedTaskSpec's left padding and a tokenizer that defaults to right padding (Tiny Aya), the mask is all ones, so pad tokens are attended and positions shift: max |Δlogit| 0.105 between an example alone and the same example in a padded batch, 0.0 once the tokenizer is set to left padding. We force `tokenizer.padding_side = "left"`.

9. **Circuit size off by one, and silently capped.** `api._compute_n_topn` / `ck.faithfulness` keep `int(N * (1 - sparsity))` nodes, which truncates (for N = 612 and k = 20/40/60/64/80/100, k − 1 nodes are kept). `applications/pruning/node_pruner.get_nodes_to_prune` caps head pruning at 50% per layer, so above 0.5 sparsity the evaluated circuit is much larger than requested, with no warning.

10. **NormalizedTaskSpec and chat models.** `data/normalized_task.py` has no `chat_template_mode`/templated flag on the EAP path, so pre-templated chat prompts get a second BOS; its CSV cache is keyed only by `name_len_align_seed` under a cwd-relative `./cache/normalized` and silently reused after the data changes; and `data_params.num_examples` is ignored.

11. **SafetyPromptAdapter defaults do not work with EAP.** The default `answer_contrastive` mode makes clean and corrupt prompts identical, which NormalizedTaskSpec rejects; `harmful_vs_benign` picks controls with `rng.choice` and no length matching, and left padding then shifts the whole 366-token system preamble between clean and corrupt. We pair each harmful prompt with a same-language XSTest prompt of identical templated length.

12. **Small API gaps.** `quick.load_model` never selects MPS while `discover_circuit(_model=...)` does; `build_discovery_config` is documented as public but not exported from `circuitkit`; node scores are returned as absolute values, so whether a node pushes towards refusal or compliance is lost; the memory patch that drops the `[batch, pos, heads, d_head, d_model]` attention-result tensor is opt-in (`CIRCUITKIT_TL_MEMORY_PATCH=1`) although it gave identical scores here and cut peak memory by about 30%.
