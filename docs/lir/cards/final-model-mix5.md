---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: peft
language: [en, ko, zh]
tags: [lora, safety, multilingual, refusal, fine-tuning, lost-in-refusal]
datasets: [lost-in-refusal/lost-in-refusal-data]
---

# Tiny Aya Global + Bactrian-X LoRA, with the model's own refusals mixed in

This is the final model of **Lost in Refusal: Uneven Guard-Measured Safety Drift After Benign Multilingual
Fine-Tuning, and How Much of It Is the Guard**. It is a LoRA adapter for
[Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) trained on 1,800 harmless instruction rows
(600 Bactrian-X instructions, each in English, Korean and Chinese) with five copies of 47 refusal examples added to
the training data. The refusals are the base model's own answers to held-out MultiJail prompts (18 English, 13
Korean, 16 Chinese). This is the cheap repair the paper studies: it needs no change to the training code.

## Why it exists

Fine-tuning Tiny Aya Global on the same harmless rows *without* the refusal examples (plain LoRA) changed safety
unevenly across languages. Measured by one automatic safety classifier, Qwen3Guard-Gen-4B, the share of Korean
answers to MultiJail attack prompts labelled unsafe rose from 5.2% to 17.9% over three seeds. English showed no
detectable change. Mixing the refusal examples into the data brought guard-measured Korean attack success most of
the way back: 8.9% at seed 42 and 6.2% at seeds 1 and 2, against 5.2% for the base model. The price is that the
model refuses more harmless requests.

## Results

These are guard-measured rates in % on 291 MultiJail prompts per language and 250 safe XSTest prompts per language.
Answers are greedy and capped at 256 new tokens. The task score is 1 to 7, judged by Qwen3-14B on 300 held-out
prompts. The plain and mixing columns are seed 42 (this adapter); the base model has no seed.

| | Base model | Plain LoRA | This adapter |
|---|---:|---:|---:|
| Korean attack success | 5.2 | 19.2 | 8.9 |
| English attack success | 6.9 | 6.9 | 3.4 |
| Chinese attack success | 8.6 | 13.4 | 7.9 |
| Swahili attack success (never trained) | 6.2 | 7.6 | 4.8 |
| English over-refusal on safe prompts | 9.6 | 7.2 | 13.2 |
| Task score, mean of three languages | 6.32 | 5.94 | 5.94 |

**Registered comparisons.** These were decided at seeds 1 and 2, with rules fixed before those runs were read; seed
42 had been seen before.

- **Against SafeGrad, gradient surgery at the same seed:** "This mixing arm protected Korean better than SafeGrad",
  −2.7 [−5.2, −0.5] points.
  - It raised English over-refusal above SafeGrad's, by +5.2 [+2.8, +8.0].
  - No difference in task score was detected (+0.02 [−0.06, +0.09]).
- **On a second benchmark,** the 200 unsafe XSTest prompts machine-translated: five copies minus plain fine-tuning
  in Korean is −8.5 [−12.2, −4.8] over three seeds. The refusal examples do not come from this benchmark.

## What these numbers are not

- **Every safety number is one guard's label, not shown harm.**
  - The guard also flags more of a fine-tuned model's answers to *harmless* prompts.
  - No second automatic judge reached our agreement bar in Korean.
  - How much of the measured rise is harm is open; the paper's section 3.3 gives the human readings and how they
    came back.
- **The refusal examples come from MultiJail, the benchmark that measures attack success.** Their prompt families
  are excluded from every evaluation.
- **This adapter refuses more safe requests** than the base model and than SafeGrad.
- **This adapter is below the base model on the general task score** in each of the three languages, as plain
  fine-tuning is.
- **The scope is narrow.** This is one model family, one LoRA configuration, and one seed for this adapter.
- **The training record lists a commit with uncommitted changes** (`run.json`). The data and settings are recorded
  there by hash.

## Training

| | |
|---|---|
| LoRA | r = 16, alpha = 32, dropout 0, on q, k, v, o, gate, up, down (30.2M trainable parameters) |
| Data | `bactrian_train_mix5.jsonl`: 1,800 Bactrian-X rows + 5 × 47 refusal examples = 2,035 rows (sha256 in `run.json`); loss on assistant tokens only |
| Optimizer | AdamW, peak LR 2e-4, cosine, 3% warm-up, weight decay 0, gradient clip 1.0 |
| Batch | effective 32 (4 × 8 accumulation), seed 42, 3 epochs, 192 optimizer steps |
| Hardware | one Apple M5 Pro laptop (48 GB), PyTorch on MPS, bfloat16 base weights; 82.2 minutes |

`train_log.jsonl` has every logged step.

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id, rev = "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324"
tok = AutoTokenizer.from_pretrained(base_id, revision=rev)
base = AutoModelForCausalLM.from_pretrained(base_id, revision=rev, torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "lost-in-refusal/tiny-aya-global-bactrian-lora-mix5")
```

## More

- **Code, preregistration and every result table:** [github.com/lost-in-refusal/lexsi-multilingual-safety](https://github.com/lost-in-refusal/lexsi-multilingual-safety)
- **Every generation and guard verdict:** [lost-in-refusal-eval-outputs](https://huggingface.co/datasets/lost-in-refusal/lost-in-refusal-eval-outputs)
- **The other adapters,** at three seeds and in every setting: [tiny-aya-lora-stage2](https://huggingface.co/lost-in-refusal/tiny-aya-lora-stage2)

## Citation

```bibtex
@misc{kang2026lostinrefusal,
  title  = {Lost in Refusal: Uneven Guard-Measured Safety Drift After Benign Multilingual Fine-Tuning, and How Much of It Is the Guard},
  author = {Kang, Mingyu and Nikkumbh, Aishwarya and Pham, Hannah},
  year   = {2026},
  howpublished = {\url{https://github.com/lost-in-refusal/lexsi-multilingual-safety}}
}
```
