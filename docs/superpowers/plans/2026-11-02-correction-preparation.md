# Correction preparation implementation plan

> Continue the user's approved next step through native work and disjoint delegated tasks. Routine implementation choices do not require a new approval round.

**Goal:** Preserve evidence-backed adjudication, prepare a separate bilingual correction corpus, and verify the complete assistant-only token sequences for a first plain LoRA experiment.

**Architecture:** Keep historical evaluation artifacts immutable. Record model adjudications and any field-scoped human decisions separately. Author 48 new examples in 12 scenario families, split into 32 training and 16 calibration rows. A CPU-only preflight validates partitions, rejects exact evaluation overlap, and audits the pinned cached Aya tokenizer without truncation or loading model weights.

**Tech stack:** Existing Python environment, pinned local Aya tokenizer, stdlib unittest, existing SafeTune/PEFT training compatibility evidence.

**Spec:** `docs/correction-data-design.md`; unchanged policy at `data/pilot/policy.json`; unchanged semantic rubric at `docs/grounding-comparison-spec.md`.

## Constraints and review focus

- Preserve all original evaluation rows, outputs, annotations, and summary files. No test generation or scoring.
- The new train/calibration families differ in scenario circumstances but share policy concepts. Mechanical disjointness is not proof of semantic independence.
- Every example remains an assistant-authored draft pending human review. A human decision applies only to the response and dimension actually reviewed.
- Messages contain original language-matched system policy, user request, and authored assistant response. No grounding appendix.
- Mask all prefix tokens, supervise the complete response and EOS, verify exact token-prefix and decoded-target equality, and reject oversized or control-token-bearing data. Training total sequence cap 1024; inference cap remains 512 new tokens.
- This work package prepares inputs/configuration. Full-context memory compatibility and substantive training remain unrun; the earlier tiny LoRA smoke does not establish those.

## Task 1 — Evidence adjudication

- [x] Review all 10 disputed reference responses under unchanged v2 rules; save exact source identities/hashes, model-provisional proposed labels, and reasoning without majority voting.
- [x] Preserve Mingyu's request-outcome judgment for the Korean address case in a separate partial human-decision record.
- [x] Keep uncertainty and review coverage visible; do not silently replace historical scores.

## Task 2 — Corpus and training configuration

- [x] Author 8 train and 4 calibration families, each with benign/disallowed English/Korean pairs; use novel circumstances and no evaluation response copies.
- [x] Independently review all 48 targets for policy fidelity, translation alignment, and absence of invented procedures; record limitations and human review pending.
- [x] Save a fixed proposed one-epoch 32-step LoRA configuration using the proven local SafeTune/PEFT path, without claiming an optimizer run occurred.

## Task 3 — Validated CPU preflight

- [x] Add failing tests for schema/pair completeness, family split leakage, normalized duplicate/evaluation-request overlap, false human approval, and invalid rule IDs.
- [x] Add failing token-boundary tests for masked prefixes, supervised EOS, incomplete/truncated targets, excessive length, and special-token injection.
- [x] Implement `validate_correction_rows(rows, policy, evaluation_rows)` and `prepare_example(row, policy, tokenizer, max_sequence_tokens)` in `scripts/prepare_corrections.py`.
- [x] Add a CLI that refuses to overwrite output directories, uses only the pinned cached tokenizer, and writes hash-bound train/calibration token files and a manifest. No model weights or optimizer are loaded.
- [x] Run the real tokenizer audit on the complete corpus; report per-language/split token lengths, supervised counts, exact policy/config/dataset hashes, and 0 optimizer steps.

## Task 4 — Delivery

- [x] Run relevant tests, independent source/data/artifact review, and immutable-evidence checks.
- [x] Update README/milestones with actual scope, commit, push private pilot branch, and verify remote revision.
