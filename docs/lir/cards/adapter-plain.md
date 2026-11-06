---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
library_name: peft
language: [en, ko, zh]
tags: [lora, safety-drift, multilingual, lexsi-hackathon, lost-in-refusal]
datasets: [lost-in-refusal/lost-in-refusal-data]
---

# Tiny Aya Global + Bactrian-X LoRA (plain)

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
