"""Write notebooks/lost_in_refusal.ipynb (the reproducibility notebook)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Lost in Refusal: reproducibility notebook

Tiny Aya Global fine-tuned on benign English/Korean/Mandarin instructions (Bactrian-X), evaluated on MultiJail in
English, Korean, Mandarin and Swahili (Swahili is never trained on), repaired with SafeTune's SafeGrad, and audited
with AuditKIT. CircuitKIT and linear probes look at where refusal lives.

Three ways to use this notebook:

1. **Recompute every number from the published outputs** (CPU, a few minutes). Section 2.
2. **Re-run an evaluation arm** from the published adapters (GPU, ~1 h per arm on a T4/L4). Section 3.
3. **Re-train** the plain and SafeGrad adapters (GPU, several hours). Section 4.

Pinned versions: Tiny Aya Global `af89d219`, Qwen3Guard-Gen-4B `6ec42827`, Qwen3-14B `40c06982`, SafeTune `3901a9f`,
AuditKIT `71316da`, CuratorKIT `dd9e835`, CircuitKIT `e71d5e1`."""),
    md("## 1. Setup"),
    code("""%%capture
!git clone https://github.com/lost-in-refusal/lexsi-multilingual-safety.git lir-repo
%cd lir-repo
!git clone -q https://github.com/Lexsi-Labs/SafeTune.git vendor/SafeTune && git -C vendor/SafeTune checkout -q 3901a9f
!git clone -q https://github.com/Lexsi-Labs/AuditKIT.git vendor/AuditKIT && git -C vendor/AuditKIT checkout -q 71316da
!git clone -q https://github.com/Lexsi-Labs/CuratorKIT.git vendor/CuratorKIT && git -C vendor/CuratorKIT checkout -q dd9e835
!pip install -q -e vendor/SafeTune -e vendor/AuditKIT -e vendor/CuratorKIT "transformers>=5.22,<6" peft scikit-learn matplotlib"""),
    md("""Gated models: accept the Tiny Aya Global license on Hugging Face with your own account, then log in
(sections 3-4 only). Put the token in Colab Secrets as `HF_TOKEN`; never paste it into a cell."""),
    code("""import os
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    pass"""),
    md("## 2. Recompute the results table from published outputs (no GPU)"),
    code("""from huggingface_hub import snapshot_download
snapshot_download("lost-in-refusal/lost-in-refusal-eval-outputs", repo_type="dataset", local_dir="runs/eval",
                  allow_patterns=["base/*", "plain/*", "safegrad/*"])
!python -m lir.metrics --arms base plain safegrad
from IPython.display import Markdown
Markdown(open("results/lir/results.md").read())"""),
    md("""## 3. Re-run one evaluation arm (GPU)

Downloads the frozen data and a published adapter, generates greedily on all sets, and scores with Qwen3Guard.
The task judge (Qwen3-14B) needs ~30 GB; on a smaller GPU skip `lir.judge` and compare safety metrics only."""),
    code("""snapshot_download("lost-in-refusal/lost-in-refusal-data", repo_type="dataset", local_dir="data/lir")
snapshot_download("lost-in-refusal/tiny-aya-global-bactrian-lora-plain", local_dir="runs/plain/adapter")
!python -m lir.evaluate --arm plain-rerun --adapter runs/plain/adapter --sets attack xstest task
!python -m lir.metrics --arms base plain-rerun --out results/lir-rerun"""),
    md("""## 4. Re-train (GPU, several hours)

Same data order, LoRA config and optimizer for both arms. The SafeGrad arm uses SafeTune's `SafeGradHFTrainer`
(global gradient surgery, KL to the frozen base on the MultiJail alignment slice), run once per optimizer step on
the accumulated gradient."""),
    code("""!python -m lir.train --arm plain --output runs/plain-retrain
!python -m lir.train --arm safegrad --output runs/safegrad-retrain"""),
    md("""## 5. Refusal geometry (GPU for extraction, CPU for analysis)"""),
    code("""!python -m lir.probe extract --arm base
!python -m lir.probe extract --arm plain --adapter runs/plain/adapter
!python -m lir.probe extract --arm safegrad --adapter runs/safegrad/adapter
!python -m lir.probe analyze --arms base plain safegrad
from IPython.display import Image
Image("results/lir/refusal_geometry.png")"""),
    md("""## 6. Refusal circuits with CircuitKIT

The comparison below recomputes cross-language and cross-model overlap from the published node scores (CPU).
CircuitKIT pins `huggingface-hub<2` and `transformer-lens==3.8.0`, so discovery runs in its own environment;
each language takes ~15 minutes on a 24 GB GPU or an M-series Mac."""),
    code("""snapshot_download("lost-in-refusal/lost-in-refusal-eval-outputs", repo_type="dataset", local_dir="results/lir",
                  allow_patterns=["circuits/*"])
!git clone -q https://github.com/Lexsi-Labs/CircuitKIT.git vendor/CircuitKIT && git -C vendor/CircuitKIT checkout -q e71d5e1
!python -m venv .venv-circuit && .venv-circuit/bin/pip install -q -e vendor/CircuitKIT peft scipy
!.venv-circuit/bin/python -m lir.circuits compare --tags base plain safegrad"""),
    code("""# full re-run (GPU): discovery, patching faithfulness, multi-token check on the metric
!.venv-circuit/bin/python -m lir.circuits discover --tag base --device cuda
!.venv-circuit/bin/python -m lir.circuits discover --tag plain --adapter runs/plain/adapter --device cuda
!.venv-circuit/bin/python -m lir.circuits faithfulness --tag base --device cuda
!.venv-circuit/bin/python -m lir.circuits margin --device cuda"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"accelerator": "GPU", "colab": {"gpuType": "L4"},
                                                "kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).resolve().parents[1] / "notebooks" / "lost_in_refusal.ipynb"
nbf.write(nb, out)
print("wrote", out)
