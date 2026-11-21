---
license: cc-by-nc-4.0
base_model: [CohereLabs/tiny-aya-global, CohereLabs/tiny-aya-earth]
library_name: peft
language: [en, ko, zh]
tags: [lora, safety-drift, multilingual, lexsi-hackathon, lost-in-refusal]
---

# Lost in Refusal: stage 2 LoRA adapters

The adapters behind the stage 2 results of Lost in Refusal, one folder per run. Each is a LoRA fine-tune of
[Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (revision `af89d219`) or
[Tiny Aya Earth](https://huggingface.co/CohereLabs/tiny-aya-earth) (revision `0624fafb`) on benign instruction
data. They exist to measure how much refusal behaviour benign fine-tuning removes in each language, and how much of
it SafeTune's SafeGrad keeps.

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

The file with no Korean rows has 1,748 rows. One was over the length limit and was dropped, so those runs trained
on 1,747.

Every trained run uses the settings of the stage 1 adapters: LoRA r=16, alpha 32, on all attention and MLP
projections; 3 epochs; peak learning rate 2e-4, cosine with 3% warmup; effective batch 32. Trained on an Apple M5
Pro with PyTorch MPS.

## What is in a folder

- `adapter_config.json` and `adapter_model.safetensors`, in PEFT format.
- `run.json`: base model revision, training file and its sha256, seed, code commit, loss and runtime. Paths are
  relative to the code repository.
- `train_log.jsonl`: every logged step.
- `surgery_log.jsonl` (SafeGrad runs): for each step, the KL to the base model, the cosine between the task and
  alignment gradients, and whether the gradient was projected.

The two `-vec` folders have the config, the weights and a `run.json` that names the three source adapters by hash.
The tokenizer is the base model's and is not repeated here.

The training files are in `data/lir/` of the
[code repository](https://github.com/lost-in-refusal/lexsi-multilingual-safety). The Aya training file is there and
not on Hugging Face.

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
- The `-vec` adapters were not registered. They are an exploratory check and nothing was tuned on the test prompts.
- Safety was scored by an automatic guard. The status page says what people checked by hand.

## Use

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", revision="af89d219b53ed9b13b8a4645f9c8028973510324")
model = PeftModel.from_pretrained(base, "lost-in-refusal/tiny-aya-lora-stage2", subfolder="plain-s1")
```

For the two Earth folders the base model is `CohereLabs/tiny-aya-earth` at revision
`0624fafbe200f930b89566fc4ec4da314ba48d89`.
