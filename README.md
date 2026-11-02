# Lexsi multilingual safety

GitHub: [xtnayk](https://github.com/xtnayk) · Hugging Face: [xtnayk](https://huggingface.co/xtnayk)

Current model: **Aya Expanse 8B**, selected by Mingyu on November 2, 2026 for local work. The original team brief used Tiny Aya; the model choice here supersedes that setup default. Current work verifies tooling and a tiny training loop, not a safety improvement.

## Start here

Use Python 3.12 on Apple Silicon with enough unified memory. The tested machine has 48 GiB. From a checkout of this repository:

```sh
python3.12 scripts/bootstrap_local.py
.venv/bin/hf auth login
.venv/bin/python scripts/cache_model.py
.venv/bin/python scripts/bootstrap_local.py --verify-only
HF_HUB_OFFLINE=1 .venv/bin/python scripts/auditkit_smoke_check.py \
  --output results/aya-auditkit-first-run
```

Accept the [Aya Expanse 8B model conditions](https://huggingface.co/CohereLabs/aya-expanse-8b) in your own account first. Enter a read token only into the interactive login prompt. The cache script uses the immutable model revision in `configs/runtime.json`; it does not follow a changing model HEAD. Choose a new results directory for every experiment.

`bootstrap_local.py` installs only SafeTune and AuditKIT with the tested local dependency pins. Existing source changes are never overwritten. `--verify-only` checks source revisions, editable import paths, Python, package pins, and dependency consistency. CircuitKIT is source-reviewed but excluded from this environment.

`notebooks/02_local_aya_auditkit.ipynb` wraps the same scripts for local notebook use. Its underlying runner is verified; the notebook is not a separate training or safety result. The older Colab notebook below remains a small-model fallback; unquantized Aya 8B is not validated on a free T4.

Generated `results/` files and model weights are local and excluded from Git. A compact verified summary is in `evidence/setup-summary.json`; full logs and adapters still need publication to the team's Hugging Face destination. The private repository is a staging workspace, not the final public submission.

## Mingyu's work package

Dates and assignments come from the updated team brief; model selection follows Mingyu's explicit choice above. The machine-readable checklist is `configs/mingyu-milestones.json`.

| Due | Deliverable | Current boundary |
|---|---|---|
| Nov 3 | GitHub repo and working environment | Pinned local setup; saved Colab Qwen reference |
| Nov 4 | SafeTune + AuditKIT smoke test | Actual Aya 8B generation connected to AuditKIT; benign fixtures only |
| Nov 7 | Plain fine-tuning config and baseline | Requires the agreed task, training data and held-out evaluation sets |
| Nov 10 | Plain and hardened runs; adapters on HF | Ordinary two-step LoRA works; SafeGrad and repair are untested |
| Nov 12 | Probe/CircuitKIT analysis and one figure | Source compatibility reviewed; actual CircuitKIT checkpoint execution untested |
| Nov 14 | Reproducibility notebook | Initial local wrapper prepared; final experiment reproduction remains |

The first team handoff is the agreed task plus labeled scenario families and splits. Keep translated variants of a scenario in the same split. Preserve language, scenario ID, dataset revision and scoring provenance. Aish owns attack design/metric definitions and Hannah owns data/evaluation; the six prompts in `data/fixtures/` are integration fixtures only. Test whether fine-tuning causes drift rather than assuming it does.

## SafeTune and AuditKIT integration

The runner reuses one SafeTune-loaded checkpoint through AuditKIT's public `CallableModel` API, avoiding a second full model in memory. It scores six benign English/Korean/Mandarin prompts using exact match and token F1, saves every prompt/output/score, records immutable model/tool revisions, and forces a fresh run with a unique cache identity. Generation limits are set on the SafeTune backend as well as AuditKIT.

All six fixtures passed on November 2. These two simple metrics verify scoring integration and are not the hackathon's attack-success, over-refusal or meaningful task-quality results. Real evaluation will need reviewed labels, separate development/test scenarios, and the team's chosen metrics.

## CircuitKIT setup boundary

Reviewed source: `e71d5e17fe6e2abfde47ec9a7fbaf31649601b78`. The source includes experimental Aya Expanse/Cohere support, but we have not run it on this Mac. It requires `transformer-lens==3.8.0` and `huggingface-hub>=1.5,<2`, conflicting with the working environment's Hub 2.4.0. Use a separate environment and explicit `device="mps"`.

The next compatibility test should load HF and CircuitKIT sequentially in bfloat16, compare outputs on fixed short prompts, and extract only selected last-token residuals to CPU. Avoid the stock parity test's two simultaneous float32 models on this machine. The linear-versus-nonlinear probe comparison is custom research work: split by scenario family, score held-out data, and use a causal intervention before claiming that a feature explains refusal. The shipped probe demo's training accuracy does not establish generalization.

## Local setup on this Mac

Verified hardware: Apple M5 Pro, 20 GPU cores, 48 GiB unified memory, macOS 26.5.1. The local environment uses Python 3.12 in `.venv`. The runner supports Apple Metal through `--device mps`, as well as CUDA for Colab. Each model check runs in a separate process.

Verified on November 2, 2026: dependency consistency, SafeTune imports, bfloat16 forward/backward on MPS, and one LoRA optimizer step on miniature random `cohere` and `cohere2` models. Evidence is in `results/local-environment.json` and `results/mps-architecture-check.json`; rerun the architecture check with `.venv/bin/python scripts/check_mps.py`. PyTorch reports a recommended MPS budget of 37.44 GiB on this machine. These synthetic checks do not validate actual Aya weights or full SafeGrad training. Local Hugging Face sign-in and Aya Expanse 8B access are verified as `xtnayk`. Actual Aya Expanse 8B inference passed through the pinned SafeTune loader and generation backend on November 2, 2026. The verified report is in [results/local-aya-8b-20261102-verified/report.json](results/local-aya-8b-20261102-verified/report.json): model revision `5062468bf9bc0c6035fd64e06274333ec127d980`, 32 decoder layers, bfloat16, and actual parameter device `mps:0`.

On a new environment, authenticate locally as `xtnayk` using the interactive Hugging Face CLI. Type the token into the CLI yourself; do not put it in chat, a command argument, or a project file. A token saved in Colab Secrets does not authenticate this Mac.

```sh
cd lexsi-multilingual-safety
.venv/bin/hf auth login
.venv/bin/hf auth whoami
```

Accept each model's conditions on Hugging Face before downloading it. To test Aya Expanse 8B locally:

```sh
.venv/bin/python scripts/smoke_check.py \
  --model CohereLabs/aya-expanse-8b --device mps \
  --source vendor/SafeTune --output results/local-aya-8b-smoke
```

For the Tiny Aya baseline, replace the model with `CohereLabs/tiny-aya-global` and use a different output directory. The report includes model revision, dtype, load/generation time, and allocated MPS memory after generation. The verified cached run loaded in 23.535 seconds (excluding download and tokenizer setup) and completed its first short generation call in 2.889 seconds. It returned a coherent explanation about experiment logs, truncated by the 32-new-token cap. After generation, PyTorch reported 14.97 GiB of live tensor allocations and 15.03 GiB of driver allocations. These are post-generation snapshots, not peak or total unified-memory use; no tokens-per-second measurement was taken. These are setup measurements, not a full throughput or safety benchmark.

Single-model Aya Expanse 8B inference is verified on this hardware. At 16-bit precision, nominal 8B weights alone use about 14.9 GiB. SafeTune's SafeGrad runner holds two complete models (about 29.8 GiB of weights); RESTA holds three (about 44.7 GiB), before activations, temporary tensors, and the OS. CPU placement shares the same physical memory on this Mac. Do not assume an inference pass establishes that either training or three-model repair fits. RESTA also requires a verified matching pre-alignment base checkpoint.

Aya Expanse 8B is the selected local baseline. Tiny Aya remains an untested fallback. Aya Expanse supports 23 languages and is a different model family, not a scaled-up Tiny Aya; longer training and repair still need measured fit checks.

## Local LoRA training check

On November 2, 2026, a two-step ordinary LoRA check passed on the actual Aya Expanse 8B checkpoint. It used SafeTune's model loader and LoRA wrapper with a bounded PyTorch AdamW loop. This does not validate SafeTune's high-level training runner, SafeGrad, or repair. Evidence: [report](results/aya-8b-lora-smoke-20261102/report.json), [benign examples](results/aya-8b-lora-smoke-20261102/benign_examples.json), and the exact [experiment script](results/aya-8b-lora-smoke-20261102/run_script.py).

- Two authored benign examples, 37 and 27 tokens including chat formatting; batch size 1. Only assistant-answer tokens contributed to the loss.
- Rank-4 LoRA, alpha 8, learning rate 0.0001, bfloat16 base and float32 adapters, gradient checkpointing. It trained 10,485,760 adapter parameters, approximately 0.13% of the wrapped model.
- Both losses and gradients were finite. All 448 adapter tensors changed across the two steps. Base parameters were frozen, received no gradients, and samples from all 258 frozen tensors remained unchanged.
- The two instrumented optimizer steps took 2.779 seconds, including synchronization and diagnostic checks. The losses came from different examples, so their values are not a learning curve or evidence of improvement.
- Largest MPS driver allocation observed at synchronized stage boundaries: 16.183 GiB. This is not a continuously measured peak. Total system swap was already 10.46 GiB and reached 12.91 GiB during the run; other applications were active, so neither value is isolated model memory usage.
- The 40.06 MiB adapter was saved separately in `results/aya-8b-lora-smoke-20261102/adapter/`. Its reloaded tensors matched exactly, and it reproduced the same post-training response using the existing frozen base. Responses were coherent but capped at 64 new tokens; this is not a utility evaluation or a fresh-process reload test.

To repeat the check, use a new output directory; it uses only the cached, pinned checkpoint:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/lora_smoke_check.py \
  --output results/aya-8b-lora-next-check
```

This establishes that short ordinary LoRA training works locally. Longer sequences, larger batches, useful task learning, SafeGrad, RESTA, and multilingual safety still need separate tests. The original model cache is unchanged; no merged model was saved or uploaded.

## Colab reference and fallback

[Saved Colab notebook](https://colab.research.google.com/drive/17gPZr9vhU5D69ClSVTjbGT3hM8toMLA-)

The Colab dependency check and official Qwen quickstart passed on November 2, 2026 using a Tesla T4. Captured evidence is in `results/colab-reference.json`. Tiny Aya was not run in that session. The local `.ipynb` now embeds the updated runner with MPS support; the saved Colab copy contains the earlier CUDA runner used for that successful reference run.

To recreate the Colab setup:

1. Open <https://colab.research.google.com/> and sign in with your Google account.
2. Choose **Upload notebook** and select `notebooks/01_safetune_setup.ipynb` from this folder.
3. Choose **Runtime → Change runtime type → T4 GPU** (or another available GPU). Use Python 3.12+.
4. Run sections 1–5 to install SafeTune and try its official small-model quickstart.
5. Sign in to Hugging Face as `xtnayk` and accept the conditions on [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global).
6. Create a read token in [Hugging Face token settings](https://huggingface.co/settings/tokens). Put it in Colab's **Secrets** panel as `HF_TOKEN` and enable notebook access. Keep the token out of chat and code cells.
7. Run sections 6–7 for model access and Tiny Aya compatibility. Section 8 is an optional longer upstream quickstart trial.
8. Download the results archive in section 9 before disconnecting, including after a failure.

The notebook is self-contained: upload just the `.ipynb` file. There is no need to upload the local source checkout or helper script. GPU availability is not guaranteed; no paid runtime is required by this starter.

## What is prepared

- `notebooks/01_safetune_setup.ipynb`: installation, access checks, isolated runs, and downloadable logs.
- `scripts/smoke_check.py`: readable copy of the runner embedded in the notebook.
- `scripts/lora_smoke_check.py`: separate local two-step benign LoRA experiment with adapter save/reload verification.
- `vendor/SafeTune/`: local source checkout for reference, excluded from version control.

SafeTune source is pinned to `26f59e6d1bdcacafe5f2be3b8d982fb214369659`. Each model run records its resolved model revision, package versions, runtime details, and actual pass/fail status. Each notebook attempt has a separate results directory. The Aya smoke check uses one model, batch size 1, and at most 32 new tokens; memory fit must be measured on the selected runtime.

The Colab reference confirms that the small-model setup runs. Separate local reports verify Aya Expanse 8B inference and a two-step ordinary LoRA check on MPS. Full fine-tuning experiments, safety hardening, and repair remain unvalidated, and Tiny Aya's actual checkpoint has not been run. Local verification evidence is recorded under `results/`.

## What the checks mean

The official Qwen quickstart demonstrates refusal-direction ablation. It is an installation/mechanism check, not safety repair or a hackathon result. The team brief specifies Cohere models for the project; Qwen is only the dependency check.

The Aya smoke check tests SafeTune loading, decoder-layer discovery, chat formatting, and nonempty benign generation. It does not test fine-tuning, repair, multilingual safety, or CircuitKIT compatibility. The optional unmodified Tiny Aya quickstart uses a longer layer sweep and may hit memory or formatting issues; keep any failure logs.

SafeTune's English refusal-phrase matcher is not sufficient for multilingual evaluation. The eventual experiment needs separate calibration/test sets and safety plus utility measures agreed with Aish and Hannah.

## Reading for Mingyu

- [SafeTune README at the pinned revision](https://github.com/Lexsi-Labs/SafeTune/tree/26f59e6d1bdcacafe5f2be3b8d982fb214369659): train-time hardening, weight-space recovery, and inference-time steering.
- [Tiny Aya Global model card](https://huggingface.co/CohereLabs/tiny-aya-global): access and model details.
- [CircuitKIT docs](https://circuitkit.lexsi.ai/): start with its refusal-localization tutorial and supported-model documentation; Tiny Aya compatibility remains unverified.

For later training, SafeTune's README warns that Tiny Aya's chat preamble is long: leave `max_len` automatic or use at least 512 and verify supervised tokens. Keep CircuitKIT and three-model recovery experiments out of the initial environment until their needs are clear.

## Troubleshooting

- **No GPU:** select a GPU runtime, then reconnect. If free GPUs are unavailable, retry later or discuss the brief's Kaggle fallback.
- **Import error:** inspect `imports.log`; verify Python 3.12+ and rerun installation in a fresh runtime.
- **401/403:** confirm you accepted the model conditions as `xtnayk`; check token access and the Colab Secrets notebook-access switch.
- **Xet/CAS download error:** retry in a fresh process with `HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60` before the Python command to use Hugging Face's standard HTTP downloader. This fallback successfully downloaded Aya Expanse 8B on this Mac after a Xet response-decoding error. Completed cached files are reused; preserve the failed attempt's logs and choose a new output directory.
- **Out of memory:** restart the runtime, rerun the setup, and try only the single-model smoke check. The larger optional sweep is not necessary to establish basic loading.
- **Crash or disconnect:** preserve the current attempt's logs. A missing or `running` report is incomplete, not a pass.
