# HarborCart targeted correction data

## Purpose and status

`data/corrections/examples.jsonl` contains 48 original assistant-authored examples for a small English–Korean correction experiment. It addresses observed development failure categories: credential entry through support channels, invented refund exceptions, verification bypasses, invented interface procedures, and claims of actions the assistant cannot perform. The targets explain permitted help while refusing the prohibited part of a request.

Every row has `provenance=original_assistant_draft` and `human_review_status=pending`. The Korean requests and targets need Mingyu's review. These are proposed supervised targets, not gold labels, proof of model improvement, or human adjudications of the disputed earlier model outputs. No model training or generation was performed while authoring this file.

The sole source of store-specific facts is the unchanged policy in `data/pilot/policy.json`, SHA-256 `7cd9627e0ab3fc06efb5fdd063ecc1d1432362a1583c492b37b3d8e4c708be08`. The failure categories come from the development reports. Authoring consulted development requests to avoid copying them; test rows were filtered out before inspection. No evaluation request or model-generated response was copied into this corpus. Exact normalized request comparison with all 40 development cases found no duplicates. That check does not establish semantic independence.

## Partition design

There are 12 scenario families. Each family contains a benign request and a disallowed request in both English and Korean. All four rows stay together in one partition, and each category's rule IDs match across languages.

| Partition | Families | Rows | English / Korean | Benign / disallowed | Intended use |
|---|---:|---:|---:|---:|---|
| Train | 8 | 32 | 16 / 16 | 16 / 16 | Gradient updates only |
| Calibration | 4 | 16 | 8 / 8 | 8 / 8 | Development diagnostics or a predeclared checkpoint decision; never gradient updates |
| Existing development | 12 | 40 | 20 / 20 | Existing categories unchanged | External intervention comparison; never training targets |
| Existing candidate test | 6 | 20 | Existing split unchanged | Existing categories unchanged | Remains unrun and unscored until a later frozen evaluation |

The original development and test records are outside the correction corpus. Do not concatenate them into a training export. Do not split translations or benign/disallowed pairs randomly across partitions. If calibration feedback changes targets, training settings, or checkpoint choice, record it as development use rather than an independent test.

The calibration circumstances differ from the training circumstances, but policy concepts overlap deliberately. This is a small correction exercise under one known fictional policy. Family separation does not establish generalization to unseen policies, domains, or attacks. The existing development set has already influenced prompt choices and the choice of correction categories, so it also cannot be described as untouched data.

## Scenario families

| Family ID | Split | Circumstance and learning target |
|---|---|---|
| `correction_attachment_credentials` | Train | Preparing a support attachment or message: remove password/code exposure; never create a form that collects credentials in support chat. |
| `correction_shared_device_session` | Train | A saved browser session on a shared device: possession of a session does not replace owner verification or authorize another customer's records. |
| `correction_replaced_card` | Train | A card was replaced after purchase: preserve original-payment and five-business-day rules while acknowledging that replaced-card handling is unspecified. |
| `correction_duplicate_credit` | Train | A delayed credit and temptation to request another: distinguish honest status questions from false nonreceipt claims or duplicate refunds. |
| `correction_calendar_boundary` | Train | Weekends inside the return period: calendar days include weekends and day 30; do not recast the deadline as working days. |
| `correction_unspecified_return_interface` | Train | Requests for exact buttons/file formats or an invented guarantee: state unspecified details and avoid fabricated menu paths, verification claims, or approval deadlines. |
| `correction_receipt_contents` | Train | Receipt content appears wrong: distinguish honest inquiry about a genuine record from altering item or price information to misrepresent a purchase. |
| `correction_verification_unknown_answers` | Train | Verification information is unknown: direct legitimate difficulty to official support without inventing answers, questions, or recovery methods. |
| `correction_delegated_translation` | Calibration | A family member translates support instructions: general language help is permitted while verification and credential entry remain with the owner. This tests delegation, not the training attachment-redaction circumstance. |
| `correction_refund_notice_verification` | Calibration | A message claims a refund was issued: distinguish the policy's promised behavior from a transaction the assistant cannot inspect, and refuse a false completed-action confirmation. This tests epistemic/action limits rather than duplicate-credit requests. |
| `correction_data_access_request` | Calibration | Owner data-access submission versus third-party export: explain the explicitly supported owner request while preserving verification and the assistant's inability to submit it. This tests request submission rather than a saved-session lookup. |
| `correction_conflicting_delivery_evidence` | Calibration | Two records imply different delivery dates: preserve uncertainty instead of choosing a convenient date. This tests uncertain evidence rather than the training calendar-versus-working-day calculation. |

These distinctions are concrete circumstances, not claims that the underlying concepts are disjoint. All answers may reuse policy concepts or necessary policy language. Training may therefore teach this particular policy's phrasing; any later evaluation should disclose that limitation.

## Row contract

Every JSONL row contains exactly these 11 fields:

| Field | Meaning |
|---|---|
| `id` | Unique `<family_id>_<category>_<language>` identifier. |
| `family_id` | Unit of partitioning; exactly four rows per family. |
| `split` | `train` or `calibration`. |
| `language` | `en` or `ko`. |
| `category` | `benign` or `disallowed`; policy constraints define the distinction. |
| `policy_rule_ids` | Nonempty list of existing policy rule IDs relevant to the target; aligned across each English/Korean pair. |
| `request` | Original customer request, without the system policy or a model response. |
| `target_response` | Original policy-grounded proposed assistant response. |
| `scenario_summary_en` | The same circumstance summary for all four family members; not a category-specific answer. |
| `provenance` | `original_assistant_draft`. |
| `human_review_status` | `pending` until a genuine review is recorded. |

Rule IDs come from `returns`, `refunds`, `owner_verification`, `secrets`, `privacy`, `honesty`, and `assistant_limits`. There are no actual customer records, URLs, credentials, or account identifiers in the examples.

## Training preparation requirements

Use the language's unchanged base policy as the system message, the row's `request` as the user message, and `target_response` as the supervised assistant message. The grounding appendices are not part of this proposed training input: including one later would be a separately recorded intervention choice. Do not embed scenario metadata, partition labels, rule IDs, summaries, or review status in the user message.

Use assistant-response-only supervision if the training implementation can verify the loss mask. The complete chat-template sequence must fit a maximum of 1,024 tokens; a CPU tokenizer preflight should verify this for every row before training. Do not silently truncate the policy, request, or target. Any overlength case must be revised or explicitly handled in a new frozen configuration. Tokenizer preflight and loss-mask checks are separate execution evidence; this authoring document does not claim they have already passed.

Record the data, policy, tokenizer/model revision, split membership, training configuration, and software hashes before a training run. Keep calibration examples out of optimizer batches. A calibration loss is an optimization diagnostic and does not substitute for semantic review of generated outputs. A successful training process is not evidence that these targets repaired safety or grounding.

## Review priorities

Human review should check that English and Korean preserve the same intent and restrictions, that no target invents a shop-specific capability, and that legitimate help remains useful. In particular:

- Passwords and one-time codes belong only in the owner's appropriate secure-portal field, never in support chat or an attachment.
- Verification remains mandatory even when someone says they own the account, has a saved session, or has family permission.
- Refunds remain tied to the original payment method within five business days after approval; unknown card handling is not an exception.
- An unknown procedure, deadline, format, question, or menu is stated as unspecified rather than made plausible.
- The assistant never claims it checked a transaction, verified identity, submitted a request, or approved a return.
- A refusal is paired with truthful permitted guidance when the request contains a legitimate question.

Keep raw draft targets and review history. Any proposed correction to disputed evaluation labels belongs in the separate adjudication artifacts; do not treat these newly authored targets as evidence that a prior output was correctly scored.
