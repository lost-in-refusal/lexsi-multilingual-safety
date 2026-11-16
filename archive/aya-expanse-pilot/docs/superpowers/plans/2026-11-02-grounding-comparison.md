# Grounding comparison implementation plan

> Execution continues the user's approved pilot through native work and delegated, disjoint tasks. Routine implementation choices do not require another approval round.

**Goal:** Run one policy-grounding prompt control and compare its 40 development answers with the preserved baseline under one revised rubric.

**Architecture:** Append a bilingual instruction to the unchanged base policy, reuse the existing CuratorKIT/SafeTune/AuditKIT runner, then review both runs through a shuffled pack with variant names withheld. Integrity checks bind the comparison to matching data/model/settings and exact response bytes.

**Tech stack:** Existing Python environment, Aya Expanse 8B on MPS, stdlib unittest.

**Spec:** `docs/grounding-comparison-spec.md`

## Constraints and review focus

- No weights, source scenarios, base policy, decoding parameters or held-out test cases change.
- Batch 1, greedy decoding, seed 42 and 512 new tokens for both runs; record cap hits and EOS.
- Preserve baseline artifacts and its original rubric summary; re-review both sets under v2.
- Freeze appendix and v2 rubric before candidate generation. Development findings motivated them; this is exploratory work.
- Missing/unclear judgments never count as safe/correct. Report coverage, model provenance, and paired denominators.
- Required facts, unsupported details and explicit policy contradictions are separate dimensions.

## Task 1 — Prompt and runner

- [x] Write the bilingual appendix and v2 rubric with no per-case answers.
- [x] Add tests for exact base-policy preservation, language coverage and invalid appendix schema.
- [x] Add `build_effective_prompts(policy, intervention)` and optional `--intervention`/`--rubric` arguments to `scripts/run_pilot_baseline.py`.
- [x] Save exact appendix, rubric and effective prompts with hashes; verify actual model input against them.
- [x] Run curation-only validation, inspect the prompt, then generate 40 candidate development responses.

## Task 2 — Comparison evidence

- [x] Add `scripts/grounding_comparison.py` with pack/summarize commands and integrity/partial-coverage tests.
- [x] Require matching source data, model, generation settings, sample IDs and AuditKIT outputs; reject mixed runs.
- [x] Create 80 shuffled review items plus a separate variant mapping; record all source hashes.
- [x] Review complete responses using the frozen v2 rubric and preserve evidence for adverse labels.
- [x] Summarize each variant/language/category and matched-case changes, with explicit unknowns.

## Task 3 — Handoff

- [x] Inspect results and avoid interpreting development/model-judge findings as validated safety improvement.
- [x] Save the candidate run, annotations, comparison and concise findings; keep test generation at zero.
- [x] Run appropriate tests and review artifacts.
- Branch delivery: `pilot-benchmark`; commit and remote verification are recorded in the task handoff.
