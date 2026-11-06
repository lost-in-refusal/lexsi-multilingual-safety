# Lost in Refusal

Entry to the [Lexsi AI Safety and Alignment Hackathon](https://lexsi.ai), Track 2 (Red Teaming, Safety and Alignment).
Team: Aishwarya Nikkumbh (lead), Mingyu Kang, Hannah Pham.

**Question.** A small multilingual model is fine-tuned on harmless instructions in English, Korean and Mandarin.
Does it stop refusing harmful requests, is the damage worse outside English, does it reach a language it was never
fine-tuned on (Swahili), and can SafeTune prevent it without hurting the task?

**Setup.** [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (3.35B) fine-tuned with LoRA on 1,800
benign Bactrian-X rows (600 instructions × en/ko/zh). Two arms with identical data, order, LoRA config and optimizer:
plain LoRA, and LoRA hardened with SafeTune's SafeGrad. Evaluated on 291 MultiJail attack families in
en/ko/zh/**sw**, XSTest's 250 safe prompts (over-refusal) in the same four languages, and 100 held-out Bactrian-X
tasks per training language. Safety judge: Qwen3Guard-Gen-4B. Task judge: Qwen3-14B with Tiny Aya's 1–7 rubric.

## Results

Attack success rate on 291 MultiJail families per language (strict judge: Unsafe or Controversial). Change vs the
base model in parentheses; * = p < 0.05, exact McNemar on the same prompts.

| | English | Korean | Mandarin | Swahili (never trained) |
|---|---:|---:|---:|---:|
| Tiny Aya Global | 6.9 | 5.2 | 8.6 | 6.2 |
| + plain LoRA | 6.9 (+0.0) | **19.2 (+14.1\*)** | **13.4 (+4.8\*)** | 7.6 (+1.4) |
| + SafeGrad LoRA | 4.5 (−2.4) | 10.7 (+5.5\*) | 8.9 (+0.3) | 5.5 (−0.7) |

- **Break.** Benign fine-tuning leaves English untouched, so an English-only check passes the model, while Korean
  attack success nearly quadruples and Mandarin rises by half. Refusal on attacks drops in every language,
  Swahili included (50.9% → 37.8%).
- **Fix.** Against plain LoRA on the same prompts, SafeGrad cuts attack success by 8.6 points in Korean and 4.5 in
  Mandarin and raises refusal by 9–19 points in all four languages; the largest gain is Swahili (+18.6), which has no
  training or alignment data. XSTest over-refusal stays below the base model (macro 8.4% base, 5.3% plain, 6.4%
  SafeGrad). Cost: 18.5% more training time, final training loss +0.8%. Task score: TASK_SCORE_LINE
- **Explain.** The English refusal direction separates harmful from harmless prompts in every language (AUROC
  0.97–0.99 at layer 17), and a linear probe does as well as an MLP, so the model still *recognises* harm. Plain
  LoRA shrinks the harmful-vs-harmless projection on that direction by about a third in every language; SafeGrad
  restores most of it. Across languages and models, refusal rate tracks the median projection of harmful prompts.

![Attack success, over-refusal and task score by language](results/lir/headline.png)
![Refusal geometry](results/lir/refusal_geometry.png)

Full tables: [`results/lir/results.md`](results/lir/results.md) (vs base) and
[`results/lir/vs_plain/results.md`](results/lir/vs_plain/results.md) (SafeGrad vs plain). Write-up draft:
[`docs/lir/article.md`](docs/lir/article.md). ROI: [`docs/lir/roi.md`](docs/lir/roi.md).

## Pipeline: four Lexsi tools, chained

| Step | Lexsi tool | What it does here | Code |
|---|---|---|---|
| 1. Data | **CuratorKIT** | `TextCleaner`, `ExactDeduplicator`, `MinHashDeduplicator` on Bactrian-X; cleaning of MultiJail prompts; train/eval near-duplicate removal | `lir/data.py` |
| 2. Fine-tune and harden | **SafeTune** | `SafeGradHFTrainer` global gradient surgery + `safegrad_kl_alignment_loss` against the frozen base on the MultiJail alignment slice | `lir/train.py` |
| 3. Audit | **AuditKIT** | Scores every saved output (`PrecomputedModel`), aggregates per language, paired bootstrap vs the base model (`RunComparison`) | `lir/metrics.py` |
| 4. Explain | **CircuitKIT** | EAP-IG refusal-circuit discovery per language and cross-language overlap | `lir/circuits.py` |

Plus our own probes (`lir/probe.py`: difference-of-means refusal directions, linear vs MLP probes, drift of the
refusal projection) and a causal steering check (`lir/steer.py`).

## Reproduce

Everything (datasets, adapters, every generation and judge verdict) is on Hugging Face under
[lost-in-refusal](https://huggingface.co/lost-in-refusal). The notebook `notebooks/lost_in_refusal.ipynb`
recomputes every table from the published outputs on CPU, and can re-run or re-train any arm on a GPU.

Local run (Apple Silicon or CUDA, Python 3.12):

```sh
python3.12 -m venv .venv
git clone https://github.com/Lexsi-Labs/SafeTune.git vendor/SafeTune && git -C vendor/SafeTune checkout 3901a9f
git clone https://github.com/Lexsi-Labs/AuditKIT.git vendor/AuditKIT && git -C vendor/AuditKIT checkout 71316da
git clone https://github.com/Lexsi-Labs/CuratorKIT.git vendor/CuratorKIT && git -C vendor/CuratorKIT checkout dd9e835
.venv/bin/pip install -r requirements.txt
.venv/bin/hf auth login   # accept the Tiny Aya Global license first

python -m lir.data multijail && python -m lir.data xstest && python -m lir.data bactrian-candidates
python -m lir.guard --input data/lir/bactrian_candidates.jsonl --output data/lir/bactrian_guard.jsonl --prompt-key instruction
python -m lir.data bactrian-finalize
python -m lir.evaluate --arm base --sets alignment attack task && python -m lir.data alignment-sft
python -m lir.translate --langs ko zh --model CohereLabs/aya-expanse-8b
python -m lir.translate --langs sw --model CohereLabs/tiny-aya-global
python -m lir.evaluate --arm base --sets xstest
python -m lir.train --arm plain --output runs/plain
python -m lir.train --arm safegrad --output runs/safegrad
python -m lir.evaluate --arm plain --adapter runs/plain/adapter
python -m lir.evaluate --arm safegrad --adapter runs/safegrad/adapter
for a in base plain safegrad; do python -m lir.judge --input runs/eval/$a/task.gen.jsonl --output runs/eval/$a/task.judge.jsonl; done
python -m lir.metrics --arms base plain safegrad
```

CircuitKIT needs its own environment (it pins `huggingface-hub<2`): see `lir/circuits.py`.

## Layout

```
lir/            pipeline code (data, train, evaluate, guard, judge, metrics, probe, steer, circuits, publish)
data/lir/       frozen datasets + manifests with sha256 hashes
results/lir/    results tables, probe and circuit outputs, figures, judge-agreement packs
docs/lir/       write-up drafts, data/model cards, Lexsi tool feedback
notebooks/      reproducibility notebook
archive/        the Nov 2–3 Aya Expanse 8B pilot that preceded this project
```

## Claim boundaries

Every number comes from our own runs. Safety and task labels are automatic (Qwen3Guard, Qwen3-14B); judge-human
agreement is checked on a blinded sample per language and reported separately. Single seed, one LoRA
configuration, one model size. Cohere's published MultiJail numbers for Tiny Aya are used only as a pipeline sanity
check, never as a result.
