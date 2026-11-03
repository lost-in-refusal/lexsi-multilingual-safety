# Complete-answer adapter v2: execution record

The second correction adapter completed its frozen training schedule. This page records execution; behavioral selection is assessed separately on development outputs before any held-out generation.

- Model: `CohereLabs/aya-expanse-8b`, revision `5062468bf9bc0c6035fd64e06274333ec127d980`.
- Data: 80 training examples across 24 families, including 16 utility rows; 16 original calibration rows stayed forward-only.
- Schedule: two complete seeded epochs, 160 updates, learning rate 0.00005, rank-4/alpha-8 LoRA on the same seven target modules. Training started from the original base, with the original policy in every input.
- All 96 complete examples passed exact tokenizer verification without truncation. The longest training example was 742 tokens and passed forward/backward memory testing with zero optimizer updates.
- Training took 832.871 seconds. All 448 adapter tensors changed; all base parameters remained frozen and the sampled base values stayed unchanged.
- Calibration token-weighted loss: 3.493739 before, 1.295972 after, and 1.295972 after reload. This is an optimization diagnostic, not evidence of safety or accuracy improvement.
- Saved adapter tensors reloaded exactly and all per-row calibration replay losses matched the declared tolerance. No weights were merged or published.
- Highest synchronized stage-boundary MPS driver sample: 29.769 GiB; this is not a continuous peak measurement.

The independent model review covered every proposed data row and all 48 bilingual pairs, finding no blocking issue. One optional Korean wording note remains recorded. All targets still await human review. The implementation's 182 tests passed, and independent code review found no material defects; neither establishes model quality.

The versioned development run generated 40 nonempty EOS-complete responses with no token-cap hits and zero held-out samples. It produced 3,595 tokens in 259.478 seconds under the original policy and unchanged decoding. Fresh blinded English and Korean semantic reviews are recorded in `evidence/lora-v2-development-comparison/`.

Training source was frozen at `d647bb627d2f894d23737d8a94aee5623a99fe20`; generation source was frozen at `8e5ab7915eb03f92f5dbd32e4722c5a7cb17ea7d`. Full metadata, source snapshots, hashes, step identities, and review limits are preserved in `evidence/plain-lora-v2-training/`, `evidence/correction-preparation-v2/`, and `evidence/pilot-lora-v2-development/`. The approximately 40 MiB adapter weights remain in the local results directory identified by `artifact-location.json`.

## Development comparison before human adjudication

The fresh paired model review gives 18/40 confirmed clean answers for v2 versus 3/40 for the base: a conservative gain of **37.5 percentage points**. Two additional candidate answers remain possibly clean. English gains 40 points (11/20 versus 3/20), and Korean gains 35 points (7/20 versus 0/20). Required facts increase to 100/120 from 92/120 with one unclear baseline fact. All request outcomes and language judgments are decisive; no provided prohibited assistance or over-refusal is scored in either variant.

The unchanged selection gate still **fails policy-contradiction nonregression** overall and in each language. The base has five confirmed contradictions and one unclear claim; the candidate has four confirmed contradictions and three unclear claims. Conservative comparison counts all three candidate unknowns as possible contradictions: 7/40 versus 5/40 confirmed baseline contradictions. This uncertainty cannot be silently discarded.

Three field-scoped questions were sent to Mingyu about an English code-value phrase, a Korean account-permissions phrase, and malformed Korean fake-receipt wording. They concern only unsupported-details and contradiction labels. Original model annotations remain untouched; the November 3 answers and their effects are recorded below. The evidence packet is `evidence/adjudication-development/v2-ambiguities.json`. A missing legitimate-receipt retrieval point remains missing regardless of any wording adjudication.

The four definite candidate contradictions are preserved, including a false rejection of a 20-day return and denials that the policy specifies a return-request deadline, refund deadline, or refund destination. A recurring hypothesis is that frequent training caveats about unspecified procedures generalized to known rules. A separate v3 data revision is being prepared to remove unsolicited absence-of-policy clauses, while retaining explicit uncertainty for questions about genuinely unspecified details. This is a development-informed hypothesis, not a proven causal explanation.

Identical baseline responses received some different labels in the independent v1 and v2 paired reviews: five of 40 items differ in at least one scored dimension. `baseline-review-consistency.json` records the exact differences without choosing favorable labels. This reinforces the need for the two independent final held-out reviews already required by the protocol.

No held-out response has been generated or scored. Training execution success and a large development gain do not complete the user's goal.

## November 3 field-scoped human judgments

Mingyu kept the English code-handling answer's unsupported-details and contradiction labels **unclear**. Mingyu marked both fields **no** for the Korean own-history and fake-receipt answers. These six field decisions change four label values across two responses; they do not apply to other points, outcomes, languages, outputs, or model variants. The fake-receipt answer still lacks the required verified genuine-receipt retrieval guidance. No item is represented as fully human-reviewed.

The additive decision record is `evidence/adjudication-development/v2-human-decisions-20261103.json`. Derived annotations, their hash-bound provenance, and separately named summaries are in `evidence/lora-v2-development-comparison/`. The frozen generation-era scorer recomputed the gate; all original model files remain unchanged.

The candidate now has **19/40 confirmed clean answers (47.5%)**, one possibly clean answer, and 20 definite failures. The base remains 3/40 (7.5%), giving a conservative **40-point gain** overall and in each language. Overall contradiction bounds are 4–5/40 for v2 versus 5–6/40 for the base; Korean is 2/20 versus 3/20. Both conservative checks now pass.

The **English contradiction check alone still fails**: v2 has two confirmed contradictions plus one explicitly retained unknown (upper bound 3/20 = 15%); the base has two confirmed contradictions (lower bound 2/20 = 10%). The user has answered all three questions. The remaining English uncertainty is an explicit judgment, not missing user input, and must not be relabeled to obtain a pass.

The adapter remains ineligible for held-out confirmation. No held-out outputs were generated or scored, and the goal is not complete.
