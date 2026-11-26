---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: peft
language: [en, ko, zh]
tags: [lora, safety-drift, multilingual, lost-in-refusal]
datasets: [lost-in-refusal/lost-in-refusal-data]
---

# Tiny Aya Global + Bactrian-X LoRA (plain)

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


The **control arm** of Lost in Refusal: an ordinary LoRA fine-tune of
[Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (revision `af89d219`) on 1,800 benign
Bactrian-X instruction/response pairs (600 instructions × English, Korean, Mandarin). No safety data and no safety
method. It exists to measure how much refusal behaviour benign fine-tuning removes, per language.

**Do not deploy this adapter.** It is a research artifact whose purpose is to show safety drift.

## Training

| | |
|---|---|
| LoRA | r=16, alpha=32, dropout 0, all attention and MLP projections (q,k,v,o,gate,up,down) |
| Data | `bactrian_train.jsonl` (sha256 in `run.json`), loss on assistant tokens only, default Tiny Aya chat template |
| Optimizer | AdamW, peak LR 2e-4, cosine, 3% warmup, weight decay 0, grad-norm clip 1.0 |
| Batch | effective 32 (micro-batch 4 × 8 accumulation), length-grouped sampler, seed 42 |
| Epochs | 3 |
| Hardware | Apple M5 Pro (48 GB), PyTorch MPS, bf16 base weights, fp32 adapters |

The learning rate is the usual LoRA value, not Tiny Aya's full-fine-tuning recipe (2.5e-5): full fine-tuning does
not fit in 48 GB, and 2.5e-5 is too small for LoRA to move the model. `train_log.jsonl` has every logged step;
`run.json` records runtime, data hashes and versions.

## Evaluation

See the [results table](https://github.com/lost-in-refusal/lexsi-multilingual-safety/tree/main/results/lir) and
[all generations and judge verdicts](https://huggingface.co/datasets/lost-in-refusal/lost-in-refusal-eval-outputs).

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", revision="af89d219b53ed9b13b8a4645f9c8028973510324")
model = PeftModel.from_pretrained(base, "lost-in-refusal/tiny-aya-global-bactrian-lora-plain")
```
