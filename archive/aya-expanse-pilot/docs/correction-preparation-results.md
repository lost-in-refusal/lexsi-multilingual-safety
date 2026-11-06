# Targeted correction preparation

The next experiment now has 48 authored bilingual correction examples, a fixed proposed LoRA configuration, and a successful audit with the exact cached Aya tokenizer. No model weights were loaded and no optimizer steps were run in this work package.

## Label adjudication

The [adjudication note](adjudication-development.md) reviews all ten reference responses whose labels differed between model reviews. A third model agreed with the latest semantic labels on eight responses. It identified an additional missing credential-entry explanation in one English response and retained uncertainty in the Korean address-routing response. This is a model proposal, not a replacement for historical scores.

Mingyu judged the Korean address response's `request_outcome` as **refused**, interpreting the policy reminder as retaining owner verification. The [human decision record](../evidence/adjudication-development/human-decisions.json) preserves the exact question, answer, response hash, and both review IDs. The question showed selected excerpts and a summary. Only that field received a human decision; the remaining dimensions and examples remain pending human review. Prior annotations and summaries are unchanged.

## Correction corpus

The [source examples](../data/corrections/examples.jsonl) contain eight training families and four calibration families. Every family keeps its benign/disallowed and English/Korean variants together.

| Partition | Families | Rows | English / Korean | Benign / disallowed |
|---|---:|---:|---:|---:|
| Train | 8 | 32 | 16 / 16 | 16 / 16 |
| Calibration | 4 | 16 | 8 / 8 | 8 / 8 |

Targets cover portal-only credential entry, mandatory owner verification, original-method refunds, truthful receipts/dates/status, unknown procedures, and the assistant's inability to perform account actions. They combine useful policy guidance with refusal of prohibited requests. They were authored separately from evaluation requests and model outputs, using development failure categories to choose what to teach.

A fresh model reviewer read all 48 complete request/target pairs and 24 translation pairs. It found no material policy or translation error and recorded two optional Korean wording refinements; no targets were changed. All targets remain assistant-authored drafts pending human review. The [review artifact](../evidence/correction-preparation/data-review-model.json) contains per-row judgments, translation checks, and the source hashes.

The review also found substantial overlap in policy concepts and answer patterns between train and calibration, and several prompts are more explicit about their allowed/prohibited intent than ordinary customer messages. Calibration is development data for diagnostics, not independent evidence of generalization. The [data design](correction-data-design.md) describes the concrete scenario differences and these limitations.

Mechanical checks found no shared IDs, family IDs, or normalized exact requests between the correction corpus and the existing 60 evaluation rows. Checking those fields did not generate or score any held-out answer. This check does not detect all paraphrases or establish semantic independence. The existing 40 development cases remain outside optimizer batches, and the 20 test cases remain unrun and unscored.

## Actual tokenizer preflight

The [archived manifest](../evidence/correction-preparation/preflight/manifest.json) records input, configuration, script, tokenizer-file, and chat-template hashes. It uses `CohereLabs/aya-expanse-8b` revision `5062468bf9bc0c6035fd64e06274333ec127d980` from the local cache, with no downloads or model-weight loading.

| Token diagnostic | English, 24 rows | Korean, 24 rows |
|---|---:|---:|
| System/user/generation-prefix range | 401–421 | 574–600 |
| Complete conversation range | 457–494 | 658–699 |
| Supervised response + EOS range | 50–73 | 76–109 |

All 48 complete sequences fit the **1,024-token total training limit**. A 512-token total limit would truncate every Korean example before its answer; the separate inference setting remains **512 new tokens**. The longest example is the Korean benign unspecified-return-interface case at 699 total tokens. Training has 2,399 supervised tokens across 32 examples; calibration has 1,248 across 16.

The preflight verifies that the full original-language policy and user request form an exact token prefix, masks that entire prefix with `-100`, and supervises only the complete authored assistant answer plus terminal EOS (`255001`). Decoding the supervised suffix must reproduce the target exactly. Oversized sequences, altered policy bytes, literal tokenizer control tokens, inconsistent bilingual families, and normalized duplicate/evaluation requests are rejected. No silent truncation or response rewriting occurs.

Separate [training](../evidence/correction-preparation/preflight/train.jsonl) and [calibration](../evidence/correction-preparation/preflight/calibration.jsonl) token exports preserve IDs, families, and partitions. The archive also contains the original draft examples, policy, configuration, runtime, and chat template. Source data SHA-256: `579a87a59182393d0332c822e3920c146e33c128c4678b6d0dba012ac5448381`.

## Proposed first training run

[correction-lora.json](../configs/correction-lora.json) fixes one seeded pass of **32 optimizer steps**, batch size one, original system policy only, bfloat16 base weights on MPS, rank-4/alpha-8 LoRA, FP32 adapters, learning rate `1e-4`, and non-reentrant gradient checkpointing. It uses the already-tested SafeTune loading/LoRA utilities and an explicit assistant-only loss mask. Calibration is forward-only, token-weighted loss before and after training, with no checkpoint selection in this first pilot. Save adapters only; do not merge or publish them.

This configuration is a proposal, not a completed training run. The earlier two-step compatibility smoke used only 27–37-token sequences. It does not establish memory behavior at 699 tokens. The next execution should first perform a forward/backward memory check on the longest training example without an optimizer update, clear gradients, and reset the seed before the actual pass. The training runner must consume only the frozen train partition, preserve frozen base parameters, check finite gradients and adapter updates, and verify exact adapter save/reload.

Calibration loss remains an optimization diagnostic. Any subsequent behavior comparison must evaluate the base and adapted model under the same original policy and decoding settings, keeping policy completeness, invented details, contradictions, prohibited assistance, over-refusal, and language adherence separate. Do not describe successful training or decreasing loss as demonstrated repair.

## Verification and reproduction

All **105 repository tests passed**, including 19 new correction-preflight tests. The real tokenizer audit completed with zero truncated examples, zero optimizer steps, zero model-weight loading, and zero test generations/scores.

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/prepare_corrections.py \
  --output results/correction-preflight-next
```

The output directory must be new. This command only prepares and validates inputs; it does not train. Historical data, responses, prompts, annotations, and comparison summaries remain unchanged.
