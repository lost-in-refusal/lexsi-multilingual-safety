# Measured Aya model improvement

> The user established an active goal to improve the model by a significant metric. Continue through actual training and behavior measurement; compatibility checks or lower loss alone do not achieve it.

**Goal:** Obtain a substantial, verified improvement in correct policy-grounded English/Korean answers without degrading policy-safety boundaries or legitimate assistance.

**Architecture:** Train a small unmerged LoRA adapter using the frozen correction train partition, compare it with the original base-policy baseline through the same CuratorKIT/SafeTune/AuditKIT generation pipeline, and select on development data before a frozen held-out confirmation. Preserve all historical artifacts. The precise practical-versus-statistical acceptance standard is being clarified with Mingyu; do not claim either standard before its evidence exists.

**Inputs:** `configs/correction-lora.json`, `evidence/correction-preparation/preflight/`, immutable base policy/data/rubric, original baseline `evidence/pilot-development/`.

## Work sequence

- [x] Implement a hash-bound training runner and focused integrity/schedule/calibration tests.
- [x] Validate the longest 699-token training example with forward/backward and no update; reset gradients/RNG; stop on nonfinite/OOM failures.
- [x] Train exactly one seeded 32-step pass on training rows only; measure calibration loss before/after as a diagnostic.
- [x] Save an adapter without merging, verify frozen base samples and finite updates, reload adapter tensors exactly, and verify calibration replay.
- [x] Add adapter-aware development generation and comparison provenance without changing legacy behavior.
- [x] Freeze an explicit primary success metric and safety/coverage guardrails before seeing candidate outputs.
- [ ] Generate 40 fresh adapted development outputs under the original system policy, unchanged decoding and exact model revision.
- [ ] Freshly review baseline/candidate outputs with variant names withheld; quantify clean-answer improvement and all adverse changes with uncertainty and family dependence visible.
- [ ] If development is insufficient, use only training/calibration/development evidence to revise the approach and rerun; keep the final test sealed.
- [ ] Freeze the selected adapter, scoring and confirmation procedure before opening held-out cases. Confirm the actual improvement against the selected acceptance standard.
- [ ] Audit evidence before completing the goal. Model-only review remains identified as provisional; do not substitute loss, memorized targets, or software tests for behavior improvement.

## Invariants

One cached MPS bfloat16 base model at a time; existing memory fraction0.8; no model/publication/adapter merge. Original model revision, policy and all previous artifacts stay fixed. New training targets remain explicitly assistant-authored drafts, with no invented human approval. Existing40development rows and20test rows never enter optimizer batches. Preserve failed runs and distinguish execution failure from behavioral failure.
