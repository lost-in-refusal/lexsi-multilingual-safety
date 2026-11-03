# Known-fact adapter v3

V3 revises the correction data to reduce unsolicited claims that policy details are unspecified. This is a development-informed hypothesis. Its behavior must pass the unchanged selection gate before any held-out confirmation.

## Intervention and execution

The model, original policy, optimizer, adapter configuration, and decoding settings stay fixed relative to v2. V3 changes 14 training requests and 50 targets, keeps 30 target strings unchanged, and retains the same 80 training examples across 24 families. It uses two complete epochs (160 updates) at learning rate 0.00005 from the original Aya Expanse 8B base. Earlier adapters are not loaded for training.

The training partition's 80 rows and 40 bilingual pairs passed independent model policy/translation review. The full 96-row corpus review remains **changes_required** because six inherited calibration targets omit a required verification reminder; four also have minor opportunities to clarify conditional timing. The 16 calibration rows remain byte-identical to v1/v2, never enter gradients or checkpoint/candidate selection, and serve only as a limited loss diagnostic. The original review findings and coordinator scope decision are preserved separately. This is not human approval.

All 96 complete sequences passed tokenizer and loss-mask checks without truncation. The longest training example was 818 tokens; its forward/backward memory preflight performed zero updates and left the adapter unchanged. All 160 updates completed in 400.083 seconds. All 448 adapter tensors changed; base parameters stayed frozen, and sampled values remained unchanged. Saved adapter tensors reloaded exactly, with all calibration replay losses matching tolerance. The highest synchronized MPS driver sample was 29.519 GiB, not a continuous peak measurement.

Calibration loss was 3.493739 before and 1.341292 after, reproduced after reload. This loss includes the known calibration limitations and is not a correctness, safety, or selection metric. An independent execution audit verified actual step IDs, calibration exclusion, source/data/adapter hashes, and reload checks. All 183 software tests passed before the run.

## Development evaluation

The original-policy development run produced 40 nonempty EOS-complete answers, zero token-cap hits, and zero held-out answers. It generated 3150 tokens in 256.070 seconds. The validated comparison pack pairs these answers with the same original base outputs; fresh English and Korean reviewers see shuffled items with variant names withheld. Semantic review is complete; the unchanged selection gate fails as detailed below.

Training source was frozen at `e80218b8beecaa0bdd10fc9860de77126b244729`, and generation source at `cf1810b9ee6cac7a797c1ebcfb055f39082f87af`. Metadata and immutable snapshots are in `evidence/plain-lora-v3-training/`, `evidence/correction-preparation-v3/`, and `evidence/pilot-lora-v3-development/`. Adapter weights remain local and unmerged at the path in `artifact-location.json`.

The selection procedure and all thresholds remain fixed. Only candidates that pass every gate are eligible; among passing candidates, prefer more confirmed-clean answers and then fewer confirmed contradictions. The three earlier v2 human wording questions remain pending and do not change any v3 labels. No held-out case has been generated or scored.

## Development result

V3 produced **21/40 confirmed clean answers (52.5%) versus 3/40 (7.5%) for the base**, a conservative gain of **45 percentage points**. English is 12/20 versus 3/20; Korean is 9/20 versus 0/20. The paired primary comparison has 18 improvements, zero regressions, and 22 unchanged cases. Required facts are 100/120 confirmed with two unclear, versus 91/120 with three unclear for the base.

This result does **not** pass the frozen selection gate:

- V3 has one definite policy contradiction and three unclear claims, versus three definite and three unclear baseline claims overall. Conservative comparison treats the candidate's four possible contradictions against the base's three confirmed contradictions.
- English has one definite candidate contradiction and one unclear claim, versus zero definite and three unclear baseline claims. Baseline English coverage for contradiction and complete required-fact fulfillment is 85%, below the 90% floor.
- One legitimate Korean request has an unclear outcome, while the base has no over-refusals or unclear legitimate outcomes. Its possible over-refusal fails the no-regression bound.

The definite candidate contradiction is a return-window answer that first states the correct 30-day window and later says the deadline is 20 days. The correct initial statement does not cancel the wrong deadline. No prohibited assistance is scored as provided in either variant; all outputs use the requested language. These are provisional model labels, not human-validated results or a general safety conclusion.

Raw annotations, component summaries, paired changes, and exact failed checks are preserved in `evidence/lora-v3-development-comparison/`. The baseline-consistency report records changes in independent judgments of identical base outputs; it does not choose favorable labels. No candidate has yet cleared all required gates on the existing unadjudicated evidence, and no held-out output has been generated.
