# First correction adapter: development result

The first adapter did **not** pass the frozen development selection gate. It must not be used for held-out confirmation. All figures below are provisional model reviews, not human-validated gold labels.

| Metric | Original base | First adapter |
|---|---:|---:|
| Fully correct, grounded answers | 4/40 (10%) | 11/40 (27.5%) |
| English clean answers | 4/20 | 4/20 |
| Korean clean answers | 0/20 | 7/20 |
| Required points met | 92/120; 2 unclear | 75/120; 1 unclear |
| Outputs with unsupported details | 35/40 | 7/40 |
| Outputs with policy contradictions | 4/40; 3 unclear | 1/40 |
| Provided prohibited requests | 0/16 | 0/16; 2 unclear |
| Over-refused legitimate requests | 0/24 | 0/24; 2 unclear |
| Requested language followed | 40/40 | 40/40 |

The clean-answer gain is 17.5 percentage points overall, below the required 20 points; English gains zero points. Required-point fulfillment declines in both languages. Unknown prohibited-request and legitimate-request outcomes also fail the conservative nonregression requirements. Baseline English contradiction coverage is 85%, below the 90% coverage floor. The complete list of failed checks is preserved in `evidence/lora-development-comparison/goal-metrics-model.json`.

The reduction in unsupported claims is real under these reviews, but many adapted outputs become short generic policy statements and omit relevant details. For example, a legitimate English credential-handling question receives ownership-verification boilerplate, and a request about a buyer's own receipt is treated as involving another customer's records. These are development observations that inform the next intervention, not evidence of held-out improvement.

## Execution and provenance

The local MPS adapter completed all 32 prescribed updates from the original Aya Expanse 8B revision. The base remained frozen, adapter save/reload checks passed, and calibration token-weighted loss fell from 3.493739 to 1.587427. Calibration was forward-only and did not enter optimizer batches. These are execution diagnostics, not behavioral success criteria.

The adapted development run produced 40 EOS-complete outputs with no empty answers or token-cap hits, under the original policy and unchanged greedy decoding. It generated 1,911 tokens in 225.166 seconds. Training metadata is in `evidence/plain-lora-training/`; generated records and frozen source snapshots are in `evidence/pilot-lora-development/`.

Two fresh model reviewers each read all 40 shuffled paired items for one language, with variant names withheld. Their original annotations, the pack mapping, component summary, and fixed-denominator acceptance report are in `evidence/lora-development-comparison/`. Historical annotations were not overwritten; review variation across separate experiments remains a limitation. The prior human decision concerned a different grounded response and was not applied here.

## Next experiment

Keep this failed candidate and all acceptance requirements unchanged. The versioned second experiment trains from the original base, using 80 training rows with more complete request-specific answers and explicit utility questions, plus the original 16 forward-only calibration rows. Its fixed schedule is two complete epochs (160 updates) at learning rate 0.00005, with the same rank-4 adapter and remaining settings. New targets remain assistant-authored drafts pending human review, and must pass an independent model policy/translation audit and tokenization checks before training.

No held-out response has been generated or scored. Only a candidate that passes the unchanged development gate may proceed to the frozen 20-case confirmation.
