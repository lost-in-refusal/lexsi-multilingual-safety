# HarborCart correction corpus v2

## Purpose and status

`data/corrections-v2/examples.jsonl` is a separate, versioned synthetic corpus for the local English–Korean Aya correction experiment. It contains 80 new training rows and the 16 calibration rows from v1, retained verbatim. It does not overwrite `data/corrections/examples.jsonl`.

The v1 aggregate results indicated that short, generic correction answers reduced unsupported details but lost required facts and sometimes refused legitimate requests. V2 therefore teaches complete, request-specific answers: answer the allowed question directly, preserve every relevant policy condition, refuse only the prohibited action, and offer a concrete truthful alternative when appropriate. A longer answer is not inherently better; targets avoid appending unrelated policy rules. English training targets are 62–95 whitespace-separated words. Korean targets are semantic counterparts, not translations padded to an English word count.

All rows have `provenance=original_assistant_draft` and `human_review_status=pending`. Requests, proposed answers, categories, and Korean translations are assistant-authored drafts awaiting genuine human review, including Mingyu's Korean review. They are not gold labels, human approvals, or evidence that the model improved. Local training authorization does not change that review status. Targets contain user-facing answers only, not private reasoning or chain-of-thought supervision.

## Sources and separation

The sole authority for HarborCart-specific facts is the unchanged `data/pilot/policy.json`, SHA-256 `7cd9627e0ab3fc06efb5fdd063ecc1d1432362a1583c492b37b3d8e4c708be08`.

Authoring also read `data/corrections/examples.jsonl` and `docs/correction-data-design.md` to preserve the v1 contract and calibration data and to adapt eight v1 training circumstances. All 80 v2 training requests were newly worded, and all training responses were authored for those requests. The other eight paired training families and eight utility-only families were newly drafted circumstances. Authoring did not read `data/pilot/scenarios.jsonl`, evaluation outputs, review annotations, or test cases. No GPU, model generation, or training was used to author this corpus.

The corpus is synthetic and deliberately derived in part from v1. Exact normalized training-request comparisons with v1 found no duplicates; that limited check does not establish semantic independence. Calibration already overlaps the policy concepts taught in training. Several new circumstances also exercise concepts present in calibration, including refund-status uncertainty and owner data access. The retained calibration set is a development diagnostic, not a held-out claim of independent generalization. No claim is made that this corpus is semantically independent of existing development or test data, which were not inspected for this authoring task.

## Partition and category design

| Partition | Family structure | Families | Rows | English / Korean | Benign / disallowed / utility |
|---|---|---:|---:|---:|---:|
| Train, paired contrast | Benign and disallowed request in both languages | 16 | 64 | 32 / 32 | 32 / 32 / 0 |
| Train, utility only | One general policy question in both languages | 8 | 16 | 8 / 8 | 0 / 0 / 16 |
| Calibration, retained v1 | Benign and disallowed request in both languages | 4 | 16 | 8 / 8 | 8 / 8 / 0 |
| Total | All families remain within one split | 28 | 96 | 48 / 48 | 40 / 40 / 16 |

All training family IDs begin with `v2_`. Calibration keeps its original IDs, fields, text, line order, and serialized bytes. The 16 retained lines follow the 80 training lines. Calibration targets are not expanded or repaired in this version; the full-answer changes apply to new training rows. Never include calibration in gradient updates, even though it lives in the same JSONL file. Select rows by the explicit `split` field. Keep both languages and all categories within a family together.

## Training families

The first eight families adapt v1 circumstances with new requests and complete targets.

| Training family | Origin | Intended distinction |
|---|---|---|
| `v2_attachment_credentials` | Adapted v1 | Remove credentials from a support message or attachment; provide safe wording instead of a credential-collection template. |
| `v2_shared_device_session` | Adapted v1 | A saved session neither establishes ownership nor permits another person's records; owners may view their own history after verification. |
| `v2_replaced_card` | Adapted v1 | Explain original-method refunds and the five-business-day period while leaving replaced-card procedures unspecified. |
| `v2_duplicate_credit` | Adapted v1 | Draft an honest delayed-refund inquiry; reject false nonreceipt claims, duplicate credits, and different destinations. |
| `v2_calendar_boundary` | Adapted v1 | Include day 30, weekends, and holidays; reject recasting calendar days as working days. |
| `v2_unspecified_return_interface` | Adapted v1 | Give actual eligibility conditions without inventing menus, upload types, approval deadlines, or checked procedures. |
| `v2_receipt_contents` | Adapted v1 | View one's genuine receipt and describe an apparent discrepancy; reject deceptive purchase substitutions. |
| `v2_verification_unknown_answers` | Adapted v1 | State mandatory owner verification and unknown recovery details; reject guessed verification answers. |
| `v2_owner_history_review` | New | Directly permit an owner's history review after verification; protect a housemate's records despite claimed consent. |
| `v2_owner_delivery_address` | New | Directly permit an owner's address review after verification; protect a gift recipient's address. |
| `v2_missing_receipt` | New | Explain access to one's genuine receipt and the full return conditions; reject fabricated replacement proof. |
| `v2_assistant_return_submission` | New | Distinguish stated eligibility from a submitted or approved return; reject false completed-action confirmations. |
| `v2_account_address_change` | New | Acknowledge the assistant cannot change an account; explain permitted address viewing and unspecified change procedures. |
| `v2_travel_deadline` | New | Count from actual delivery despite later opening or travel; reject false delivery dates and invented extensions. |
| `v2_partner_order_number` | New | An order number does not replace verification; owners can retrieve their own records, but claimed partner permission is insufficient. |
| `v2_support_draft_status` | New | Preserve uncertainty in a refund inquiry; reject fabricated payment confirmation or claimed account actions. |
| `v2_utility_return_window` | New utility | State the calendar-day window, inclusive day 30, and genuine-receipt condition without an unnecessary refusal. |
| `v2_utility_return_receipt` | New utility | Explain that the deadline and genuine receipt are both required; distinguish fabricated proof. |
| `v2_utility_refund_rule` | New utility | State the trigger, original destination, five business days, and no duplicate or alternate-destination exception. |
| `v2_utility_credential_location` | New utility | State owner-only entry in the appropriate official secure portal field, never assistant or support chat. |
| `v2_utility_owner_records` | New utility | List permitted own records after verification and the boundary protecting other customers. |
| `v2_utility_assistant_capabilities` | New utility | Explain actual capabilities and every listed account-action limitation. |
| `v2_utility_policy_gaps` | New utility | Identify absent URLs, interface labels, file formats, and approval deadlines without inventing procedures. |
| `v2_utility_data_request` | New utility | Permit an owner's data-access request after verification; distinguish unspecified formats and delivery times. |

The unchanged calibration families are `correction_delegated_translation`, `correction_refund_notice_verification`, `correction_data_access_request`, and `correction_conflicting_delivery_evidence`.

## Row contract

Every row has exactly the same 11 fields as v1, in the same order:

| Field | V2 contract |
|---|---|
| `id` | Unique `<family_id>_<category>_<language>`. |
| `family_id` | Partition unit: four rows for a contrast family, two for a utility-only family. |
| `split` | `train` or `calibration`. |
| `language` | `en` or `ko`. |
| `category` | `benign`, `disallowed`, or the new allowed value `utility`. |
| `policy_rule_ids` | Nonempty, duplicate-free list of existing policy rule IDs; equal across each language pair. |
| `request` | Customer request only, without policy or labels. |
| `target_response` | Complete proposed assistant answer in the request language. |
| `scenario_summary_en` | Shared English description across every member of the family. |
| `provenance` | `original_assistant_draft`. |
| `human_review_status` | `pending`; no fabricated human review. |

The policy IDs remain `returns`, `refunds`, `owner_verification`, `secrets`, `privacy`, `honesty`, and `assistant_limits`. No actual credentials, account identifiers, customer records, or portal URLs appear in the new requests or targets.

## Response design and review

Benign guidance starts with the answer to the question. It starts with an inability statement only when the customer asks the assistant to perform an unavailable action, such as changing an address. An ordinary question about viewing one's own records is answered affirmatively with the required owner-verification route, not treated as a prohibited privacy request. Utility answers explain precisely the relevant rule without generic verification boilerplate; verification appears when it is actually part of the question or permitted route.

Disallowed answers identify the specific forbidden part and preserve any legitimate purpose. Some include a concrete honest support draft. No target supplies evasion, fabricated records, false status, invented verification answers, or a completed-action claim. Directions to official support for account-specific help retain mandatory owner verification rather than presenting support as a bypass.

Review checked relevant fact coverage and both languages together:

- Account-specific help requires the owner to complete verification in the official secure portal; permission, a session, or an order number does not replace it.
- Passwords and one-time sign-in codes go only into the appropriate official secure portal field, entered by the owner, never into assistant or support chat.
- Owners may view their own order history, delivery address, and genuine receipts after verification, and may submit their own data-access request there. Other customers' records remain private.
- Return answers preserve 30 calendar days from delivery, inclusive day 30, and a genuine receipt whenever those eligibility conditions are relevant.
- Refund answers preserve the original payment method and five business days after return approval. Delays never authorize duplicate refunds or a different destination.
- The assistant has no account tools and does not perform or claim records access, identity verification, login, account changes, return approval, refund issuance, or submission.
- Unknown procedures remain explicitly unspecified; no URLs, UI paths, document formats, recovery methods, or approval deadlines are invented.

Assistant inspection of pairs is not a substitute for human Korean review. Retaining the v1 calibration bytes also retains their original limitations and pending-review status.

## Validation and execution boundary

Local data validation checked all 96 rows: exact field names and ordering, unique IDs, valid policy IDs, fixed provenance/review values, expected counts, family split integrity, consistent summaries, aligned language-pair rule lists, exact calibration-byte preservation, and absence of normalized request duplicates within v2 or between v2 training and v1. The final training target length range was checked separately from token length. No model-performance conclusion follows from those structural checks.

The intended follow-up configuration is 80 optimizer examples from 24 training families, with 16 examples from four calibration families excluded from training. The requested training plan is two epochs, 160 optimizer steps when one row is used per step, learning rate `5e-5`, and a full-sequence limit of 1,024 tokens. These are intended execution settings, not a record of a completed run.

Use the unchanged language-specific policy text as the system message, `request` as the user message, and `target_response` as the assistant message. Do not include metadata, category labels, rule IDs, summaries, grounding appendices, or review status in the training conversation. Verify assistant-response-only supervision and its actual loss mask. A CPU tokenizer preflight must check the entire rendered chat-template sequence, including system policy, request, target, and special tokens, before training. Never silently truncate any of those components. Any necessary target revision must preserve the complete relevant facts and be recorded as a new corpus hash.

Freeze and record data, policy, tokenizer/model revision, software, split membership, masking checks, and training configuration before a run. Longer Korean targets may stress the 1,024-token limit; authoring does not claim the tokenizer preflight has passed. Calibration feedback used to change targets or select a checkpoint is development use, not an independent test. Successful optimization or lower loss alone does not demonstrate a safety or utility repair.
