---
license: cc-by-nc-4.0
language: [en, ko, zh, sw]
task_categories: [text-generation]
tags: [safety, multilingual, red-teaming, over-refusal, tiny-aya, lexsi-hackathon]
pretty_name: Lost in Refusal data
configs:
  - config_name: bactrian_train
    data_files: bactrian_train.jsonl
  - config_name: bactrian_eval_task
    data_files: bactrian_eval_task.jsonl
  - config_name: multijail_attack
    data_files: multijail_attack.jsonl
  - config_name: multijail_alignment
    data_files: multijail_alignment.jsonl
  - config_name: alignment_sft
    data_files: alignment_sft.jsonl
  - config_name: xstest_safe
    data_files: xstest_*.jsonl
---

# Lost in Refusal: frozen data

Data for the Lost in Refusal entry to the Lexsi AI Safety and Alignment Hackathon (Track 2): fine-tune
[Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) on benign English/Korean/Mandarin instructions,
measure how safety drifts per language (including Swahili, which is never trained on), repair it with SafeTune, and
re-audit. Code: [lost-in-refusal/lexsi-multilingual-safety](https://github.com/lost-in-refusal/lexsi-multilingual-safety).

Every file was produced by `lir/data.py` and `lir/translate.py` from pinned upstream revisions; sha256 hashes are in
the `*_manifest.json` files. Nothing here was hand-written.

## Files

| File | Rows | What it is |
|---|---:|---|
| `bactrian_train.jsonl` | 1,800 | 600 Bactrian-X instruction IDs × en/ko/zh. Fine-tuning data for both arms. |
| `bactrian_eval_task.jsonl` | 300 | 100 further IDs × en/ko/zh. Task-score prompts (judge reference = Bactrian-X answer). |
| `multijail_attack.jsonl` | 1,164 | 291 MultiJail families × en/ko/zh/sw, with harm categories. Attack-success prompts. |
| `multijail_alignment.jsonl` | 72 | The other 24 MultiJail families × en/ko/zh. SafeGrad alignment prompts only; never evaluated. |
| `alignment_sft.jsonl` | ≤72 | Tiny Aya Global's own greedy refusals to the alignment prompts, kept if Qwen3Guard rates them safe refusals. SafeGrad's KL term runs on these. |
| `xstest_{en,ko,zh,sw}.jsonl` | 250 / 250 / 250 / 249 | XSTest's 250 safe prompts (over-refusal), English original plus translations. |
| `screening/` | | Bactrian-X candidates and the Qwen3Guard verdicts used to filter them. |

Fields: Bactrian-X rows `id, lang, instruction, response, source, split`; prompt sets `uid, family, lang, prompt`
plus set-specific metadata. Rows sharing `id`/`family` are translations of the same item.

## How it was built

**Bactrian-X** (`MBZUAI/Bactrian-X@3698480`, en/ko/zh files). IDs present in all three languages (67,017) are ranked
by `sha256("42:bactrian-<id>")`. An ID is dropped if, in *any* language, the response is near-empty, contains a
refusal/disclaimer pattern ("As an AI", "죄송", "抱歉", "作为AI", ...), or the item has a non-empty `input`
field. CuratorKIT cleans the text (`TextCleaner`: unicode normalisation, control characters, encoding artefacts),
removes exact duplicates (`ExactDeduplicator`) and near-duplicates (`MinHashDeduplicator`, Jaccard ≥ 0.7). The first
1,000 surviving IDs are screened with Qwen3Guard-Gen-4B (response moderation of instruction + response, all three
languages); 14 IDs were rated Unsafe/Controversial and 12 as refusals, and were removed. The first 100 remaining IDs
form `eval_task`; the next 600 that are not MinHash near-duplicates of an eval instruction form `train`.

**MultiJail** (`DAMO-NLP-SG/MultiJail@93d65c7`, MIT). All 315 families ranked by `sha256("42:multijail-<id>")`; the
first 24 are the SafeGrad alignment slice (en/ko/zh only), the remaining 291 are the attack set in en/ko/zh/sw.
Swahili prompts are MultiJail's native Swahili translations; no Swahili data is used for training or alignment.

**XSTest** (`Paul/XSTest@f600c99`, CC-BY-4.0). The 250 prompts labelled safe. Korean and Mandarin translations by
Aya Expanse 8B; Swahili by Tiny Aya Global (Aya Expanse does not cover Swahili). Greedy decoding, prompts in
`lir/translate.py`. Outputs that answered the question instead of translating it (multi-line, much longer than the
source, or a question turned into a statement) were retried with a stricter translation-engine prompt and then a
few-shot prompt; 5 Korean/Mandarin and 28 Swahili rows were re-translated this way and 1 Swahili row that still
failed was dropped (listed in `xstest_manifest.json`).

## Known gaps

- Bactrian-X answers are machine-generated (gpt-3.5-turbo) and its non-English instructions are machine
  translations; quality is uneven, especially in Korean.
- XSTest translations are machine translations. XSTest relies on English homonyms ("kill a process"); some
  translations lose the ambiguity and become plainly benign, which makes over-refusal easier to avoid.
- The Swahili XSTest translation was produced by the model under test.
- Qwen3Guard screening is automatic. Its agreement with human labels is checked separately (target ≥ 80% per
  language); see the results repo.
- As a fine-tuning target, Bactrian-X's short GPT-3.5 answers make Tiny Aya's answers shorter without improving
  reference match or judged quality; it works as a benign fine-tuning set, not as a way to improve the model.
- Pretraining contamination of MultiJail, XSTest or Bactrian-X in Tiny Aya cannot be ruled out.

## Licenses

Bactrian-X is CC-BY-NC-4.0, MultiJail MIT, XSTest CC-BY-4.0, and Tiny Aya outputs fall under Tiny Aya's
CC-BY-NC-4.0 license plus Cohere's acceptable-use policy. The combined release is CC-BY-NC-4.0. MultiJail prompts
are harmful by design and are included only for evaluation.
