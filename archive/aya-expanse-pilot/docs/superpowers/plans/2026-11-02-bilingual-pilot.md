# Bilingual pilot implementation plan

> **For agentic workers:** Use native execution with delegated, disjoint dataset and validation tasks. The user has approved continuing the proposed pilot; no additional approval round is needed for these implementation choices.

**Goal:** Produce a bilingual support-policy pilot and an auditable Aya 8B development baseline.

**Architecture:** Original paired scenarios feed deterministic CuratorKIT curation, one SafeTune-loaded Aya model, and AuditKIT response records. Semantic scoring is a separate annotation step with explicit coverage and reviewer provenance.

**Tech Stack:** Python 3.12, pinned SafeTune/AuditKIT/CuratorKIT, PyTorch MPS, stdlib unittest.

**Spec:** `docs/pilot-spec.md`

## Global constraints

- Use the pinned Aya Expanse 8B revision; generation and judging stay local. Version the original fictional prompts in the authorized private GitHub workspace; do not publish a dataset to Hugging Face in this work package.
- English/Korean; 60 draft rows, development 40 and candidate test 20.
- One model in memory; batch 1; greedy decoding; 512 new tokens after the interrupted 128-token development calibration; record cap hits.
- No scenario family crosses splits; pending human review stays explicit.
- Unreviewed/unclear outcomes are not safe outcomes and do not enter decisive denominators.

## Review focus

- Duplicate IDs or translated families leaking across splits must fail validation.
- Partial annotation coverage must never appear as zero harmful compliance.
- Long/shared policy prefixes must not cause curation to collapse distinct scenarios.
- AuditKIT cache reuse must not be reported as fresh model generation.
- Interrupted runs must preserve per-case outputs and a non-passing status.

## Task 1 Data and scoring contract

- [x] Create original paired data and bilingual fictional policy under `data/pilot/`.
- [x] Write tests for ID collisions, split leakage, missing translations, invalid outcomes, and partial review denominators.
- [x] Implement `validate_rows(rows)` and `summarize_annotations(rows, annotations)` in `scripts/pilot_validation.py`.
- [x] Run `python -m unittest discover -s tests -v` and inspect the dataset counts.

## Task 2 Real curation and baseline

- [x] Pin the inspected CuratorKIT source and install only the compatible core environment.
- [x] Add `scripts/run_pilot_baseline.py` consuming the spec/data and validation functions.
- [x] Write curation hashes/removal records; generate development responses with per-case checkpoints, exact generation metadata, and new run identity.
- [x] Save AuditKIT output plus an annotation template; use coverage-aware aggregation for any provisional labels.
- [x] Run the real development baseline; require all cases recorded, no execution errors, and test partition unrun.

## Task 3 Review and handoff

- [x] Review English/Korean equivalence and label boundaries without claiming human validation.
- [x] Document actual counts, cap hits, metric coverage and remaining human-review work in README.
- [x] Run tests and inspect artifacts; preserve the unmodified baseline and source data.
- Branch delivery: `pilot-benchmark`; the commit and remote verification are recorded in the task handoff.
