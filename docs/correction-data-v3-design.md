# HarborCart correction corpus v3

## Purpose and provisional status

`data/corrections-v3/examples.jsonl` revises the v2 synthetic English–Korean correction corpus while preserving its partition, family structure, category counts, schema, and original 16 calibration lines. It contains 80 training rows from 24 families and 16 calibration rows from four families. All rows remain `provenance=original_assistant_draft` and `human_review_status=pending`. Requests, proposed answers, labels, and Korean translations are assistant-authored drafts awaiting genuine human review, including Mingyu’s Korean review. They are not gold labels or human approvals.

The author was given the aggregate v2 failure pattern: a response may state a known rule correctly and then incorrectly call that rule unspecified. V2 targets also contain many unsolicited absence-of-policy caveats. The proposition that those caveats contributed to the observed failure is a hypothesis, not an established cause. This revision is intended to test that data change; it does not demonstrate a model repair.

V3 answers the actual request, includes its relevant policy constraints, and then stops. There is no minimum word count, target length band, or reward for longer answers. Ordinary own-record and credential-location questions receive direct guidance without unsolicited menus, formats, recovery procedures, or other unknowns. An unspecified procedure is mentioned only when the request asks about that missing detail. Legitimate own-record requests remain allowed after owner verification. Disallowed actions are refused specifically, with relevant truthful alternatives.

Targets are user-facing answers only, with no private reasoning or chain-of-thought supervision.

## Authority and authoring boundary

The sole authority for HarborCart facts is the unchanged `data/pilot/policy.json`, SHA-256 `7cd9627e0ab3fc06efb5fdd063ecc1d1432362a1583c492b37b3d8e4c708be08`.

The author read `data/corrections-v2/examples.jsonl` and `docs/correction-data-v2-design.md`. V1 was accessed solely to select its calibration lines for an exact byte comparison. No evaluation scenarios, evaluation responses, annotation files, or test cases were opened or used for authoring. No network, GPU, model generation, or training was used to create these two files. Scripts and configurations were not edited by this author.

Source fingerprints:

| Source | SHA-256 |
|---|---|
| V2 JSONL | `410421b89d0996e12203b1bf0b70f5d028f0e75788e56261e142aa1aa0479b86` |
| V2 design note | `9a0a0031723c1de4db14ac835ad7841b8ff525bf8d3509114a130546cba74cf7` |
| V1 JSONL, used for calibration-byte comparison | `579a87a59182393d0332c822e3920c146e33c128c4678b6d0dba012ac5448381` |
| Retained 16-line calibration block | `23cfa8576650de9a0ab375123828d3c1721f42e07b5c23a22050e78cc28048a5` |
| Final v3 JSONL | `c9d57310196dd923fe79ad6a7690a776ecd8b517bbbaf0c5b3295b994ab639ce` |

## Partition and row contract

| Partition | Family structure | Families | Rows | English / Korean | Benign / disallowed / utility |
|---|---|---:|---:|---:|---:|
| Train, paired contrast | Both categories in both languages | 16 | 64 | 32 / 32 | 32 / 32 / 0 |
| Train, utility only | One utility request in both languages | 8 | 16 | 8 / 8 | 0 / 0 / 16 |
| Calibration, retained v1 | Both categories in both languages | 4 | 16 | 8 / 8 | 8 / 8 / 0 |
| Total | Families stay within one split | 28 | 96 | 48 / 48 | 40 / 40 / 16 |

Each row keeps the same 11 fields, in the same order: `id`, `family_id`, `split`, `language`, `category`, `policy_rule_ids`, `request`, `target_response`, `scenario_summary_en`, `provenance`, `human_review_status`. IDs have the form `<family_id>_<category>_<language>`. Each language pair has the same category and ordered rule list; every family has a shared English summary. Rule lists are nonempty, duplicate-free subsets of the seven policy rule IDs. No actual credentials, customer records, account identifiers, or portal URLs are supplied in requests or targets.

All 80 training IDs and family IDs change only their prefix from `v2_` to `v3_`. Training line order is retained. Split, language, category, provenance, and review-status values are unchanged in every row. The final 16 serialized lines are copied verbatim from v2 and are byte-identical to v1, including IDs, fields, text, ordering, and newline bytes. The calibration families remain `correction_delegated_translation`, `correction_refund_notice_verification`, `correction_data_access_request`, and `correction_conflicting_delivery_evidence`.

Calibration must never enter gradient updates. Select training rows by `split == "train"`; do not assume every row in this file is a training example.

## Exact v2 retention and revision ledger

This is a derivative revision of all 24 v2 training families. No family is newly independent. For any training row, replace the `v3_` prefix with `v2_` to locate its source row. The table below accounts for every training request and target in both languages. “Retained” means exact string equality with v2. “Revised” means the current v3 string differs; the table describes the change. Each row represents the English and Korean pair of the listed category. Unlisted categories do not exist for that family.

Across the 80 training rows, 66 requests are retained and 14 revised; 30 targets are retained and 50 revised. Every request revision is paired across languages, and every target revision is paired across languages. The 16 calibration requests and targets are all retained verbatim in addition to those training counts.

| Family suffix after `v3_` | Category | Request | Target | Text retained or revised |
|---|---|---|---|---|
| `attachment_credentials` | benign | Retained | Retained | Safe message replacement, owner-only credential fields, never chat, and owner verification are unchanged. |
| `attachment_credentials` | disallowed | Retained | Revised | Replace the unsolicited “unspecified steps” clause with a credential-free description of the account question; preserve the refusal and secure owner-only route. |
| `shared_device_session` | benign | Retained | Revised | Affirm access to one’s own history after verification; remove missing-session-switching procedure caveats while preserving the uncertain-session privacy boundary and assistant limits. |
| `shared_device_session` | disallowed | Retained | Retained | Refusal to copy a roommate’s receipts and the verified-owner route are unchanged. |
| `replaced_card` | benign | Retained | Retained | The request actually asks about card replacement; its narrow handling caveat remains alongside the known original-method, five-business-day rule. |
| `replaced_card` | disallowed | Retained | Revised | Refuse the invented destination exception, state the known refund rule, and provide an honest support question without an additional absence-of-procedure statement. |
| `duplicate_credit` | benign | Retained | Retained | Honest delayed-refund draft, known refund rule, no duplicate or alternate destination, verification, and inability to check payment remain unchanged. |
| `duplicate_credit` | disallowed | Retained | Retained | Refusal of false nonreceipt and duplicate payment, plus a truthful delayed-refund inquiry, remain unchanged. |
| `calendar_boundary` | benign | Retained | Retained | Day 30 on a holiday is explicitly within the window with a genuine receipt; submission and approval remain unavailable to the assistant. |
| `calendar_boundary` | disallowed | Retained | Revised | Apply the known rule directly to day 33, remove the closing absence-of-holiday-extension sentence, and add owner verification for the account-specific support route. |
| `unspecified_return_interface` | benign | Revised | Revised | Expand the question to request the request deadline, approval duration, refund timing/destination, menu, and upload type. Apply the 12-day/genuine-receipt facts; distinguish known request and refund rules from unknown approval/interface details. |
| `unspecified_return_interface` | disallowed | Retained | Revised | Preserve the refusal of invented menus, PDF requirement, and two-hour approval. Explicitly distinguish the known request deadline from unspecified approval duration and add verification for account-specific support. |
| `receipt_contents` | benign | Retained | Revised | Provide own genuine-receipt access and a truthful mismatch draft; remove unrequested correction-procedure and record-change caveats. |
| `receipt_contents` | disallowed | Revised | Revised | Add delivery 11 calendar days ago and clarify that the old receipt belongs to another purchase. State that timing is within the window but altering another purchase’s receipt fails the genuine-receipt condition. |
| `verification_unknown_answers` | benign | Retained | Retained | The request explicitly asks about unknown verification questions and recovery. Keep the specific caveat, truthful support question, owner-only credentials, and assistant limits. |
| `verification_unknown_answers` | disallowed | Retained | Revised | Refuse invented answers and bypass claims; retain the actual-owner requirement and truthful support route without reciting unspecified questions, accepted answers, or recovery methods. |
| `owner_history_review` | benign | Retained | Revised | Give direct permission and the verified-owner route; remove filters, download formats, menu caveats, and the unasked comparison with another customer. |
| `owner_history_review` | disallowed | Retained | Retained | Housemate privacy, insufficiency of claimed permission, verification, and assistant limits are unchanged. |
| `owner_delivery_address` | benign | Retained | Revised | Give direct permission to view one’s own address after verification; remove screen and address-change caveats from this viewing question. |
| `owner_delivery_address` | disallowed | Retained | Retained | Gift/order-number privacy boundary and the legitimate direct-contact alternative are unchanged. |
| `missing_receipt` | benign | Retained | Revised | Remove download-button and file-format caveats. Explicitly apply the 22-day timing condition while still requiring the genuine receipt. |
| `missing_receipt` | disallowed | Revised | Revised | Add delivery 18 calendar days ago. Explain that valid timing does not replace genuine proof; refuse fabrication and require owner verification before the own-receipt/support route. |
| `assistant_return_submission` | benign | Retained | Revised | Preserve the favorable nine-day/genuine-receipt application and unavailable submission/approval actions; remove unasked screen and approval-duration caveats. |
| `assistant_return_submission` | disallowed | Revised | Revised | Replace a bare claim of eligibility with nine calendar days and a genuine receipt. Explicitly affirm those conditions while refusing a false submitted/approved confirmation. |
| `account_address_change` | benign | Retained | Revised | Retain the relevant unknown change procedure because the user asks how to change the address; shorten unrelated menu-path language and keep owner verification, allowed viewing, and no completed change. |
| `account_address_change` | disallowed | Retained | Revised | Refuse false login/change completion and verification bypass; provide an honest support route without repeating a missing-procedure caveat. |
| `travel_deadline` | benign | Retained | Revised | Explicitly apply the delivery-based window to day 31 and explain that opening later does not restart it. Remove the generic absent-travel-exception statement and add owner verification for account-specific support. |
| `travel_deadline` | disallowed | Revised | Revised | Specify actual delivery 34 calendar days ago and opening yesterday. Apply the outside-window result directly, refuse the false date, and preserve a truthful verified-owner support route. |
| `partner_order_number` | benign | Retained | Revised | Keep the order-number limitation and affirmative own-receipt route after verification; remove portal-search caveats and redundant framing. |
| `partner_order_number` | disallowed | Retained | Retained | Partner permission and order number still do not establish ownership or authorize disclosure. |
| `support_draft_status` | benign | Retained | Revised | Keep the uncertain-status draft, known refund timing/destination, verification, and inability to check status; remove the unasked delayed-refund/second-destination clause. |
| `support_draft_status` | disallowed | Retained | Retained | Refusal of a fake payment confirmation, truthful status uncertainty, known refund rule, verification, and assistant limits are unchanged. |
| `utility_return_window` | utility | Retained | Retained | Calendar-day count, inclusive day 30, genuine receipt, and day 30 versus day 31 applications are unchanged. |
| `utility_return_receipt` | utility | Retained | Retained | Both deadline and genuine receipt are required; fabricated proof is rejected. |
| `utility_refund_rule` | utility | Retained | Retained | Original method within five business days after approval, without duplicate or alternate-destination exceptions for delay, is unchanged. |
| `utility_credential_location` | utility | Retained | Revised | Preserve the owner-only appropriate secure-portal field, never chat, and mandatory verification; remove unasked verification-screen/question caveats. |
| `utility_owner_records` | utility | Retained | Retained | Permitted own records and protection of another customer’s information are unchanged. |
| `utility_assistant_capabilities` | utility | Retained | Revised | Answer capabilities directly and preserve all seven unavailable actions; remove the unasked absent-URL/customer-information clause. |
| `utility_policy_gaps` | utility | Revised | Revised | Ask for documented timing as well as interface details. Explicitly contrast the known request deadline and known post-approval refund rule with unknown approval duration and requested interface details. |
| `utility_data_request` | utility | Revised | Revised | Replace the requested export-format/delivery-time discussion with a direct own-data permission, verification-location, and assistant-submission question. Answer those points without unrelated export/menu caveats. |

The source and destination hashes above identify the exact strings; no unchanged training request or target is represented as newly authored text.

### Metadata changes beyond IDs

`policy_rule_ids` is retained in 68 training rows and revised in these six English/Korean pairs:

| Pair (`v3_` family suffix and category) | V2 ordered list | V3 ordered list |
|---|---|---|
| `calendar_boundary`, disallowed | `returns, honesty` | `returns, honesty, owner_verification` |
| `unspecified_return_interface`, benign | `returns, owner_verification, assistant_limits` | `returns, refunds, owner_verification, assistant_limits` |
| `unspecified_return_interface`, disallowed | `returns, honesty, assistant_limits` | `returns, honesty, assistant_limits, owner_verification` |
| `travel_deadline`, benign | `returns, honesty, assistant_limits` | `returns, honesty, assistant_limits, owner_verification` |
| `travel_deadline`, disallowed | `returns, honesty, assistant_limits` | `returns, honesty, assistant_limits, owner_verification` |
| `utility_policy_gaps`, utility | `assistant_limits, honesty` | `returns, refunds, assistant_limits, honesty` |

`scenario_summary_en` is retained in 76 training rows. Only these two utility families change it, identically in both languages:

- `v3_utility_policy_gaps`: “A general policy question separates the known return-request deadline and post-approval refund rule from unspecified approval duration and interface details.”
- `v3_utility_data_request`: “A general policy question asks whether owners may request their own personal data and distinguishes the owner’s portal submission from the assistant’s limits.”

All other metadata values are unchanged except the training ID/family prefixes. Calibration has no metadata changes.

## Concept overlap and limits

All v2 training-family concepts remain: credentials and verification; saved sessions and another customer’s privacy; original-payment refunds and duplicate-refund refusals; calendar deadlines and genuine receipts; unsupported interface details; honest record discrepancy reports; own records and data access; address-change capability limits; truthful delivery dates and refund status; and direct general-policy utility. V3 reuses the circumstances and most requests rather than claiming a new independent corpus. The new numeric facts in four disallowed requests make existing conditions concrete; they do not create new policy rules.

The retained calibration families overlap training concepts, including owner verification, credential location, refund-status uncertainty, own-data access, and delivery-based eligibility. Calibration remains a development diagnostic. Reusing it, or changing data in response to aggregate development feedback, does not establish independent generalization. No independence or performance claim is made about uninspected evaluation/test material.

The new narrow treatment of unknowns does not remove genuine policy gaps. Training still answers explicit questions about replaced-card handling, verification/recovery details, address-change procedures, return menus/upload types, approval duration, and portal URLs without inventing procedures. In every training example that contrasts return timing with unknown approval duration, the known request deadline remains 30 calendar days after delivery, including day 30. The known refund rule remains the original payment method within five business days after approval; a delay permits neither a duplicate refund nor a different destination.

Calibration wording was not repaired. Independent review reported missing owner-verification reminders in these six retained calibration rows:

- `correction_refund_notice_verification_disallowed_en` and `correction_refund_notice_verification_disallowed_ko`.
- `correction_conflicting_delivery_evidence_benign_en` and `correction_conflicting_delivery_evidence_benign_ko`.
- `correction_conflicting_delivery_evidence_disallowed_en` and `correction_conflicting_delivery_evidence_disallowed_ko`.

Their official-support referrals omit the required owner-verification reminder for account-specific help. All six remain byte-identical to v1 and v2 under the calibration-preservation requirement. The reported decision is to retain calibration as forward-only diagnostic data, exclude it from gradient updates and checkpoint selection, and assess optimizer-target clearance separately. These inherited calibration limitations do not block using independently cleared training targets. This note does not claim that all 96 rows are semantically clear. Calibration also retains its original unspecified-detail caveats. Byte preservation does not certify those drafts as ideal targets or replace human review.

## Author review and validation

The author read all 96 final rows as 48 English/Korean pairs against the sole policy authority. This inspection checked the actual question, relevant fact coverage, numeric conditions and units, scope of unknowns, refusal boundaries, allowed own-record routes, required owner verification, owner-only credential fields, assistant capabilities, honest status wording, and semantic agreement across languages. A Korean refusal was adjusted to avoid implying that a physical item substitution had already occurred, and the missing-receipt support route was made explicit about verification. This is assistant inspection, not native-speaker certification or human approval.

Local structural validation passed for all rows: exact field order, unique IDs, category/split/language counts, family split integrity, family summaries, bilingual rule-list agreement, allowed policy IDs, nonempty strings, fixed provenance/review status, and v3 training prefixes. The 16 calibration lines matched both v1 and v2 byte-for-byte. There were no normalized duplicate requests within v3 and no URLs in requests or targets. These are limited structural checks, not semantic-independence tests.

For description only, English training targets range from 52 to 106 whitespace-separated words. That observation is not a quota and does not measure tokenizer length. Korean answers are semantic counterparts, not word-count padding. The frozen corpus SHA-256 is `c9d57310196dd923fe79ad6a7690a776ecd8b517bbbaf0c5b3295b994ab639ce`.

## Intended execution

The intended follow-up run starts from the original base model, not the v2 adapter. Keep v2’s training settings: 80 training examples, two epochs, 160 optimizer steps at one example per step, learning rate `5e-5`, and the existing 1,024-token full-sequence limit. This holds those settings fixed while testing the data revision. This document records an intended run, not a completed run or result.

Use the unchanged language-specific policy text as the system message, the request as the user message, and the target response as the assistant message. Exclude metadata from the conversation. Verify assistant-response-only supervision and the actual loss mask. A CPU tokenizer preflight must inspect the complete rendered chat-template sequence, including policy, request, target, and special tokens. Do not silently truncate any component. Tokenizer/model revision, data/policy hashes, software, split membership, supervision masking, and configuration must be recorded before training. Any data revision required by preflight or review must receive a new corpus hash and an updated ledger.

No tokenizer preflight, model run, loss improvement, or safety/utility improvement is claimed by this authoring note. Calibration-driven choices remain development use; a completed optimization run alone would not prove that the hypothesized error mechanism was corrected.
