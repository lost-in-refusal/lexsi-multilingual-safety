# Measured Aya model improvement

> The user established an active goal to improve the model by a significant metric. Continue through actual training and behavior measurement; compatibility checks or lower loss alone do not achieve it.

**Goal:** Obtain a substantial, verified improvement in correct policy-grounded English/Korean answers without degrading policy-safety boundaries or legitimate assistance.

**Architecture:** Train a small unmerged LoRA adapter using the frozen correction train partition, compare it with the original base-policy baseline through the same CuratorKIT/SafeTune/AuditKIT generation pipeline, and select on development data before a frozen held-out confirmation. Preserve all historical artifacts. Mingyu selected a practical gain of at least 20 percentage points on held-out clean answers without worse safety or over-refusal; the frozen protocol also imposes bilingual coverage and nonregression gates.

**Inputs:** `configs/correction-lora.json`, `evidence/correction-preparation/preflight/`, immutable base policy/data/rubric, original baseline `evidence/pilot-development/`.

## Work sequence

- [x] Implement a hash-bound training runner and focused integrity/schedule/calibration tests.
- [x] Validate the longest 699-token training example with forward/backward and no update; reset gradients/RNG; stop on nonfinite/OOM failures.
- [x] Train exactly one seeded 32-step pass on training rows only; measure calibration loss before/after as a diagnostic.
- [x] Save an adapter without merging, verify frozen base samples and finite updates, reload adapter tensors exactly, and verify calibration replay.
- [x] Add adapter-aware development generation and comparison provenance without changing legacy behavior.
- [x] Freeze an explicit primary success metric and safety/coverage guardrails before seeing candidate outputs.
- [x] Generate 40 fresh adapted development outputs under the original system policy, unchanged decoding and exact model revision.
- [x] Freshly review baseline/candidate outputs with variant names withheld; quantify clean-answer improvement and all adverse changes with uncertainty and family dependence visible.
- [ ] If development is insufficient, use only training/calibration/development evidence to revise the approach and rerun; keep the final test sealed.
- [ ] Freeze the selected adapter, scoring and confirmation procedure before opening held-out cases. Confirm the actual improvement against the selected acceptance standard.
- [ ] Audit evidence before completing the goal. Model-only review remains identified as provisional; do not substitute loss, memorized targets, or software tests for behavior improvement.

## Invariants

One cached MPS bfloat16 base model at a time; existing memory fraction0.8; no model/publication/adapter merge. Original model revision, policy and all previous artifacts stay fixed. New training targets remain explicitly assistant-authored drafts, with no invented human approval. Existing40development rows and20test rows never enter optimizer batches. Preserve failed runs and distinguish execution failure from behavioral failure.

## Second experiment decision

The first adapter failed the selection gate: 4/40 to 11/40 clean answers (+17.5pp), no English clean-answer gain, and required facts declined. See `docs/plain-lora-development.md` and immutable v1 evidence. Do not open the held-out set.

The next authorized iteration uses separate v2 data/configuration: 80 training rows across 24 families (64 paired benign/disallowed rows and 16 utility rows), the unchanged 16 diagnostic calibration rows, two epochs/160 updates at learning rate 0.00005, all other settings unchanged. Train from the original base, not the first adapter. New targets must pass independent policy/translation review and exact tokenizer preflight. Acceptance gates and test set stay fixed. The old preflight source hash is preserved in its archive; changing the preparation implementation requires a newly frozen preflight for new runs.

## Third experiment decision

V2 completed 160 steps and the original-policy development run. Its fresh paired review shows 18/40 confirmed clean answers versus 3/40 baseline (+37.5pp), and 100/120 required points versus 92/120 baseline with one unknown. All gates pass except conservative policy-contradiction nonregression: candidate four confirmed plus three unknown versus baseline five confirmed. Three field-scoped human questions are pending, without altering model annotations or opening test data.

Prepare v3 independently while those judgments are pending. The hypothesis is that repeated unsolicited caveats about unspecified procedures encouraged false denials of known rules. Use a separate versioned corpus/config, retaining 80 training/16 calibration rows, 24/4 families, two epochs/160 updates at 0.00005, and every other v2 setting. Remove irrelevant absence-of-policy clauses; retain explicit uncertainty only for genuinely unspecified details asked about, and make relevant applications of known rules direct and complete. Independent policy/translation review and a new source-bound tokenizer preflight precede training from the original base. Preserve both earlier candidates and results.

Complete the v3 development comparison before selecting the confirmation candidate. Only candidates that pass the unchanged gate are eligible. Among passing candidates, prefer the higher confirmed-clean fraction, breaking a tie by fewer confirmed policy contradictions. Do not pick favorable annotations or use calibration loss for selection. Any unresolved label uncertainty remains conservative until genuine field-scoped human adjudication; final held-out confirmation still requires two fresh independent model reviews and conservative consensus. Never generate test outputs before candidate and procedure are frozen.
