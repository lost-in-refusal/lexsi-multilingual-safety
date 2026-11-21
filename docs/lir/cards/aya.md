---
license: apache-2.0
language: [en, ko, zh, sw]
task_categories: [text-generation]
tags: [safety, multilingual, fine-tuning, tiny-aya, lexsi-hackathon]
pretty_name: Lost in Refusal, Aya training set
---

# Lost in Refusal: the Aya training set

A second harmless fine-tuning set for the Lost in Refusal entry to the Lexsi AI Safety and Alignment Hackathon
(Track 2). The first set comes from Bactrian-X, which is machine-translated. This one comes from the
[Cohere Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset), whose prompts and answers were written
or edited by people in each language. It is used to ask whether the Korean safety drift we measured after harmless
fine-tuning of [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) also appears on a second dataset.
Code: [lost-in-refusal/lexsi-multilingual-safety](https://github.com/lost-in-refusal/lexsi-multilingual-safety),
module `lir/aya.py`. The plan for these runs is section 10 of `docs/lir/stage2-prereg.md`.

## Files

| File | Rows | What it is |
|---|---:|---|
| `aya_train.jsonl` | 519 | 173 rows each in English, Korean and Simplified Chinese. Fine-tuning data for the two Aya runs. |
| `aya_eval_task.jsonl` | 400 | 100 held-out rows each in English, Korean, Chinese and Swahili. Task-score prompts; the Aya answer is the judge's reference. Never trained on. |
| `bactrian_train_ayasize.jsonl` | 519 | The first 173 IDs of our Bactrian-X training set in the same three languages. The size control. Derived from Bactrian-X, which is CC BY-NC 4.0. |
| `aya_candidates.jsonl` | 2,585 | Every row that reached the guard screen. |
| `aya_guard.jsonl` | 2,585 | The guard's verdict on each candidate. |
| `aya_manifest.json`, `aya_candidates_manifest.json` | | Counts, drop reasons and sha256 hashes. |

Fields: `uid, id, lang, instruction, response, source, split, annotation_type`. Aya is not parallel: every row
exists in one language only, so rows of different languages are different prompts. `annotation_type` is Aya's own
label for a row (`original-annotations` or `re-annotations`). Swahili appears only in the held-out task set.

## How it was built

Source: `CohereLabs/aya_dataset` at revision `f9ea04583f02a8f86404ff6c58bf75fe637df8a2`, train split, Apache 2.0.
Chinese is the rows Aya labels Simplified Chinese. Each language goes through the same steps as our Bactrian-X set.

1. Text filters. A row is dropped when the prompt is empty, the answer is near-empty (under 20 characters in
   English and Swahili, under 8 in Korean and Chinese), or the answer contains a refusal or disclaimer pattern.
   Swahili has no such pattern list, as in our first set.
2. Prompts that also appear in our Bactrian-X files are dropped. The match is exact after lower-casing and
   collapsing spaces; near-duplicates of Bactrian-X prompts are not searched for. No row was dropped by this step.
3. CuratorKIT cleans the text (`TextCleaner`) and removes exact duplicates (`ExactDeduplicator`) and
   near-duplicates (`MinHashDeduplicator`, Jaccard 0.7), per language.
4. Rows over the trainer's 1,024-token limit are dropped.
5. Rows are ranked by `sha256("42:aya-<row>")`. The first 1,000 per language go to the guard screen, or all of
   them when a language has fewer.
6. Guard screen. Qwen3Guard-Gen-4B reads each prompt with its answer. A row is kept when the verdict is Safe and
   not a refusal.
7. The first 100 kept rows per language are the held-out task set.
8. From the rest, rows that are MinHash near-duplicates of a held-out prompt are removed. Training then uses the
   same number of rows in English, Korean and Chinese, set by the language with the fewest left. That is Korean.

| | English | Korean | Chinese | Swahili |
|---|---:|---:|---:|---:|
| Rows in the Aya train split | 3,944 | 361 | 3,038 | 366 |
| After steps 1 to 4 | 2,849 | 309 | 2,130 | 276 |
| Screened by the guard | 1,000 | 309 | 1,000 | 276 |
| Dropped by the guard as not safe | 7 | 2 | 10 | 16 |
| Dropped by the guard as a refusal | 19 | 3 | 12 | 4 |
| Kept by the guard | 974 | 304 | 978 | 256 |
| Held out as the task set | 100 | 100 | 100 | 100 |
| Removed as near-duplicates of a held-out prompt | 56 | 31 | 37 | not used |
| Training rows | 173 | 173 | 173 | 0 |

In the training rows, Aya labels 97, 98 and 142 as `original-annotations` and 76, 75 and 31 as `re-annotations`
(English, Korean, Chinese).

## Limits

- The set is small. Aya has 361 Korean rows in all, so the Korean training rows are most of what Aya offers, and
  the set is 29% of the size of our Bactrian-X set (173 rows per language against 600).
- The size control matches the number of rows, not the amount of text. Aya answers are shorter: the median answer
  has 298, 160 and 43 characters in English, Korean and Chinese, against 602, 320 and 212 in the size control.
- Cohere built both Tiny Aya and the Aya Dataset, and the model card names no training data. The model may have
  seen these rows.
- No person has checked the rows by hand. The guard screen is automatic.
- Topics differ between languages, because the rows are not translations of one another.

## Hashes

| File | sha256 |
|---|---|
| `aya_train.jsonl` | `5e88476e8ba789237d4d4e606de2025a4ae72be497a7ffbfcf70f1f5f58d38d9` |
| `aya_eval_task.jsonl` | `4e0fcc782270d6facb9db075eb16f283fa0d8bbc4641a119aec7e5c194dc1d8b` |
| `bactrian_train_ayasize.jsonl` | `7ac613cf327d6df9a0589ed3c1b29af51f4a2c2388c88fe2cd9ff81d1f0b5fb2` |
