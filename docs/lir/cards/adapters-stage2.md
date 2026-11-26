---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: peft
language: [en, ko, zh]
tags: [lora, safety-drift, multilingual, lost-in-refusal]
---

# Lost in Refusal: stage 2 LoRA adapters

> From the paper *Lost in Refusal: Uneven Guard-Measured Safety Drift After Benign Multilingual Fine-Tuning, and How
> Much of It Is the Guard* (Mingyu Kang, Aishwarya Nikkumbh, Hannah Pham, 2026). Code, preregistration and every
> table: [GitHub](https://github.com/lost-in-refusal/lexsi-multilingual-safety). Everything in one place: [the collection](https://huggingface.co/collections/lost-in-refusal/lost-in-refusal-6b07c29a5b461cd1af8df254). The final model:
> [tiny-aya-global-bactrian-lora-mix5](https://huggingface.co/lost-in-refusal/tiny-aya-global-bactrian-lora-mix5).

## The study in brief

- **What was done.** Tiny Aya Global was fine-tuned with LoRA on 1,800 harmless instruction rows in English, Korean
  and Chinese, at three seeds, under a preregistration. Safety was then measured in ten languages by one automatic
  safety classifier, Qwen3Guard-Gen-4B.
- **What the guard shows.**
  - The share of Korean answers to MultiJail attack prompts labelled unsafe rose from 5.2% to 17.9% (+12.7 points,
    95% interval [+8.9, +16.7]).
  - English showed no detectable change (+0.9 [−1.8, +3.8]).
  - No Korean training rows were needed for the rise.
- **How much of it is the guard.**
  - The guard also flagged more of the fine-tuned model's Korean answers to harmless prompts (+5.5 [+2.1, +9.1]).
  - No second automatic judge reached the agreement bar in Korean.
  - One author's readings did not confirm the rise, and a second reading came back by a route the registration had
    ruled out (paper, section 3.3).
  - Every safety number in these repositories is the guard's label, not shown harm.
- **Not only a damaged model.** The rise appeared again on a second fine-tune, one-sentence news summarisation, which
  improved the task it was trained for (+4.5 [+1.9, +7.2]). The general task score fell there too.
- **Repairs.** Two repairs lowered guard-measured Korean attack success: mixing 47 of the base model's own refusals
  into the training data, and SafeGrad gradient surgery. Mixing made the model refuse more harmless requests. The
  two are points on a trade-off.
- **Scope.** One model family, Tiny Aya. A second model, IBM Granite 4.1 3B, is unconfirmed. Single-turn, direct
  requests only.


The adapters behind the stage 2 results of Lost in Refusal, one folder per run. Each is a LoRA fine-tune of
[Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (revision `af89d219`) or
[Tiny Aya Earth](https://huggingface.co/CohereLabs/tiny-aya-earth) (revision `0624fafb`) on benign instruction
data. They exist to measure how much refusal behaviour benign fine-tuning removes in each language, and how much of
it SafeTune's SafeGrad keeps.

**Two base models.** The six `earth-*` folders are adapters of Tiny Aya Earth. Every other folder is an adapter of Tiny Aya Global. The base-model field of this page can name only one model for the whole repository, so it names Global; the table below gives the base model of each folder.

**Do not deploy these adapters.** They are research artifacts whose purpose is to show safety drift.

The seed 42 runs on Tiny Aya Global are in their own repositories from stage 1:
[plain](https://huggingface.co/lost-in-refusal/tiny-aya-global-bactrian-lora-plain) and
[SafeGrad](https://huggingface.co/lost-in-refusal/tiny-aya-global-bactrian-lora-safegrad).

## Folders

| Folder | Base model | Training data | Rows | Seed | Method |
|---|---|---|---:|---|---|
| `plain-s1`, `plain-s2` | Global | `bactrian_train.jsonl`: Bactrian-X, 600 rows each in English, Korean and Chinese | 1,800 | 1, 2 | Plain LoRA |
| `safegrad-s1`, `safegrad-s2` | Global | The same file | 1,800 | 1, 2 | SafeGrad, with 47 refusals by the base model as its alignment set |
| `plain-mix1` | Global | `bactrian_train_mix1.jsonl`: the 1,800 rows and the 47 refusal examples once | 1,847 | 42 | Plain LoRA |
| `plain-mix5` | Global | `bactrian_train_mix5.jsonl`: the 1,800 rows and the 47 refusal examples five times | 2,035 | 42 | Plain LoRA |
| `plain-noko`, `plain-noko-s1`, `plain-noko-s2` | Global | `bactrian_train_noko.jsonl`: English and Chinese rows only, no Korean | 1,747 | 42, 1, 2 | Plain LoRA |
| `plain-ayasize` | Global | `bactrian_train_ayasize.jsonl`: the first 173 rows per language of the Bactrian-X set | 519 | 42 | Plain LoRA |
| `aya-plain` | Global | `aya_train.jsonl`: Cohere Aya Dataset, 173 rows each in English, Korean and Chinese | 519 | 42 | Plain LoRA |
| `aya-safegrad` | Global | The same file | 519 | 42 | SafeGrad, with the same 47 refusals |
| `earth-plain` | Earth | `bactrian_train.jsonl` | 1,800 | 42 | Plain LoRA |
| `earth-safegrad` | Earth | The same file | 1,800 | 42 | SafeGrad, with 44 refusals by Earth itself as its alignment set |
| `plain-s1-vec`, `plain-s2-vec` | Global | None | | | No training. The seed 42 SafeGrad adapter minus the seed 42 plain adapter, added to the plain adapter of seed 1 or 2. Rank 48. Exploratory |
| `aya-plain-s1`, `aya-plain-s2` | Global | `aya_train.jsonl` | 519 | 1, 2 | Plain LoRA |
| `aya-safegrad-s1`, `aya-safegrad-s2` | Global | The same file | 519 | 1, 2 | SafeGrad, with the same 47 refusals |
| `earth-plain-s1`, `earth-plain-s2` | Earth | `bactrian_train.jsonl` | 1,800 | 1, 2 | Plain LoRA |
| `earth-safegrad-s1`, `earth-safegrad-s2` | Earth | The same file | 1,800 | 1, 2 | SafeGrad, with Earth's own 44 refusals |
| `plain-only-en`, `plain-only-ko`, `plain-only-zh` | Global | `bactrian_train_only_en.jsonl`, `_ko`, `_zh`: the 600 rows of one language from the 1,800-row file, the same 600 items in each | 600 | 42 | Plain LoRA |
| `plain-only-en-s1`, `plain-only-ko-s1`, `plain-only-zh-s1`, `plain-only-en-s2`, `plain-only-ko-s2`, `plain-only-zh-s2` | Global | The same three single-language files | 600 | 1 and 2 | Plain LoRA |
| `plain-ayasize-s1`, `plain-ayasize-s2` | Global | `bactrian_train_ayasize.jsonl`, as for `plain-ayasize` | 519 | 1 and 2 | Plain LoRA |
| `aya-plain-s1-vec`, `aya-plain-s2-vec`, `aya-plain-vec` | Global | None | | | No training. The same seed 42 difference (from the Bactrian-X adapters), added to the Aya plain adapter of seed 1, 2 or 42. Rank 48. Registered before it was run. These models write answers about twice as long as the plain ones |
| `plain-mix1-s1`, `plain-mix1-s2` | Global | `bactrian_train_mix1.jsonl`, as for `plain-mix1` | 1,847 | 1, 2 | Plain LoRA |
| `plain-mix5-s1`, `plain-mix5-s2` | Global | `bactrian_train_mix5.jsonl`, as for `plain-mix5` | 2,035 | 1, 2 | Plain LoRA |
| `plain-lr-half`, `plain-lr-double` | Global | `bactrian_train.jsonl` | 1,800 | 42 | Plain LoRA with the peak learning rate halved (1e-4) and doubled (4e-4) |
| `plain-ep-half`, `plain-ep-double` | Global | `bactrian_train.jsonl` | 1,800 | 42 | Plain LoRA for half the epochs (1.5) and twice the epochs (6) |
| `plain-aligntune` | Global | `bactrian_train.jsonl` | 1,800 | 42 | Plain LoRA with the usual settings, trained through Lexsi's AlignTune and its TRL backend in place of our training script |

The file with no Korean rows has 1,748 rows. One was over the length limit and was dropped, so those runs trained
on 1,747.

The settings of the stage 1 adapters are: LoRA r=16, alpha 32, on all attention and MLP projections; 3 epochs; peak
learning rate 2e-4, cosine with 3% warmup; effective batch 32. Every trained run uses them, with these exceptions.
`plain-lr-half` and `plain-lr-double` have a peak learning rate of 1e-4 and 4e-4, and `plain-ep-half` and
`plain-ep-double` train for 1.5 and 6 epochs; each changes that one setting and nothing else. `plain-aligntune` has
the same settings and data and was not trained by our script (`lir/train.py`): it went through AlignTune
0.1.36.dev39 and `trl.SFTTrainer` (trl 1.7.1, torch 2.13). Its `run.json` lists the settings passed on and what
differs from our script. All runs were trained on an Apple M5 Pro with PyTorch MPS.

## What is in a folder

- `adapter_config.json` and `adapter_model.safetensors`, in PEFT format.
- `run.json`: base model revision, training file and its sha256, seed, code commit, loss and runtime. Paths are
  relative to the code repository.
- `train_log.jsonl`: every logged step.
- `surgery_log.jsonl` (SafeGrad runs): for each step, the KL to the base model, the cosine between the task and
  alignment gradients, and whether the gradient was projected.

The five `-vec` folders have the config, the weights and a `run.json` that names the three source adapters by hash.
The tokenizer is the base model's and is not repeated here.

The training files are in `data/lir/` of the
[code repository](https://github.com/lost-in-refusal/lexsi-multilingual-safety). The Aya training file is there and
not on Hugging Face.

## Adapters that are not here

Twelve adapters were trained on a cloud GPU, in Hugging Face Jobs, and not on the laptop. They exist and are not
released yet, here or anywhere else:

- `plain-ko`, `plain-en`, `plain-zh`: plain LoRA on one training language on Tiny Aya Global. The folders
  `plain-only-ko`, `plain-only-en` and `plain-only-zh` here are separate laptop runs of the same recipe.
- Nine adapters of IBM Granite 4.1 3B (`ibm-granite/granite-4.1-3b`), a model outside the Tiny Aya family: plain
  LoRA, SafeGrad, and plain LoRA with no Korean row, each at seeds 42, 1 and 2.

Their answers, verdicts, run records and training logs are in the evaluation outputs under `gpu/`.

## Results

Each folder name is also an arm name in the
[evaluation outputs](https://huggingface.co/datasets/lost-in-refusal/lost-in-refusal-eval-outputs), which has that
arm's answers and judge verdicts on the sets it was run on. The card there says which sets each arm has and what is
not published. The tables are under `stage2/`. The
[status page](https://github.com/lost-in-refusal/lexsi-multilingual-safety/blob/main/results/lir/stage2/README.md)
says which claims each run supports and which it does not, and the
[preregistration](https://github.com/lost-in-refusal/lexsi-multilingual-safety/blob/main/docs/lir/stage2-prereg.md)
has the tests that were fixed before the runs.

## Limits

- A row with one seed is one run.
- Cohere built both Tiny Aya and the Aya Dataset, and the model card names no training data, so the model may have
  seen the Aya rows before.
- The two `plain-s*-vec` adapters were not registered. They are an exploratory check and nothing was tuned on the test prompts. The three `aya-plain*-vec` adapters were registered before they were run; their over-refusal and task score were not measured.
- Safety was scored by an automatic guard. The status page says what people checked by hand.

## Use

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", revision="af89d219b53ed9b13b8a4645f9c8028973510324")
model = PeftModel.from_pretrained(base, "lost-in-refusal/tiny-aya-lora-stage2", subfolder="plain-s1")
```

For the six Earth folders the base model is `CohereLabs/tiny-aya-earth` at revision
`0624fafbe200f930b89566fc4ec4da314ba48d89`.
