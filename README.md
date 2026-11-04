# Lexsi multilingual safety

Team repository: [lost-in-refusal/lexsi-multilingual-safety](https://github.com/lost-in-refusal/lexsi-multilingual-safety)

Mingyu’s accounts: GitHub [xtnayk](https://github.com/xtnayk) · Hugging Face [xtnayk](https://huggingface.co/xtnayk)

Current model: **Aya Expanse 8B**, selected by Mingyu on November 2, 2026 for local work. The original team brief used Tiny Aya; the model choice here supersedes that setup default. Current work measures local English–Korean support-policy correction adapters. Held-out improvement has not yet been established.

## Current experiment status

Mingyu selected a practical target of at least **20 percentage points more fully correct, grounded held-out answers without worse safety or over-refusal**. The [frozen acceptance protocol](docs/model-improvement-protocol.md) defines conservative uncertainty bounds, bilingual checks, and the final independent reviews.

- [First adapter](docs/plain-lora-development.md): trained 32 steps; failed development selection with a 17.5-point clean-answer gain and fewer required facts.
- [Complete-answer adapter v2](docs/complete-answer-lora-v2.md): trained 160 steps; after Mingyu’s November 3 field-scoped judgments, 19/40 confirmed clean answers versus 3/40 base (+40 points). The overall and Korean contradiction checks pass, but the English check still fails with the code-handling claim explicitly retained as unclear. All three review questions have been answered.
- [Known-fact adapter v3](docs/known-facts-lora-v3.md): trained 160 steps with revised data; 21/40 clean answers versus 3/40 base (+45 points), with 18 paired improvements and zero clean-answer regressions. The safety gate still fails because of a wrong deadline and unresolved judgments, including possible over-refusal. Six inherited calibration answers have documented verification omissions; those rows never train the model or select a checkpoint.

All adapters are local and unmerged. The 20 held-out cases remain ungenerated and unscored. Development gains, lower loss, and software checks do not complete the goal. All model reviews remain provisional.

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

`bootstrap_local.py` installs CuratorKIT, SafeTune and AuditKIT with the tested local dependency pins. Existing source changes are never overwritten. `--verify-only` checks source revisions, editable import paths, Python, package pins, and dependency consistency. CircuitKIT is source-reviewed but excluded from this environment.

`notebooks/02_local_aya_auditkit.ipynb` wraps the same scripts for local notebook use. Its underlying runner is verified; the notebook is not a separate training or safety result. The older Colab notebook below remains a small-model fallback; unquantized Aya 8B is not validated on a free T4.

Generated `results/` files and model weights are local and excluded from Git. A compact verified summary is in `evidence/setup-summary.json`; full logs and adapters still need publication to the team's Hugging Face destination. The private repository is a staging workspace, not the final public submission.

## Mingyu's work package

Mingyu subsequently chose to begin an independent pilot with the assistant. The pilot below supplies draft data and a baseline without waiting for team handoffs. The team brief's larger fine-tuning and interpretability milestones remain separate; a small support-policy pilot does not complete them.

Dates and assignments come from the updated team brief; model selection follows Mingyu's explicit choice above. The machine-readable checklist is `configs/mingyu-milestones.json`.

| Due | Deliverable | Current boundary |
|---|---|---|
| Nov 3 | GitHub repo and working environment | Pinned local setup; saved Colab Qwen reference |
| Nov 4 | SafeTune + AuditKIT smoke test | Actual Aya 8B generation connected to AuditKIT; benign fixtures only |
| Nov 7 | Plain fine-tuning config and baseline | Fictional-policy baseline, versioned correction data, and plain LoRA runs completed; held-out confirmation remains |
| Nov 10 | Plain and hardened runs; adapters on HF | Three local plain LoRA adapters trained; SafeGrad, confirmed repair, and HF publication remain separate |
| Nov 12 | Probe/CircuitKIT analysis and one figure | Source compatibility reviewed; actual CircuitKIT checkpoint execution untested |
| Nov 14 | Reproducibility notebook | Initial local wrapper prepared; final experiment reproduction remains |

The team plan assigns attack design/metric definitions to Aish and data/evaluation to Hannah. The independent pilot is a draft for review, not their approval or a replacement for human-validated labels. Keep translated variants of a scenario in the same split and preserve dataset revision and scoring provenance. The six prompts in `data/fixtures/` remain integration fixtures only. Test whether fine-tuning causes drift rather than assuming it does.

## English–Korean development pilot

**First baseline completed:** [read the findings and review priorities](docs/pilot-baseline-review.md). All 40 development responses completed without truncation; 20 candidate test cases remain unrun. Model review found correct core utility facts alongside unsupported procedural details and two contradictory Korean refund responses. These are provisional findings requiring human review, not evidence of safety improvement. The full small run is versioned in `evidence/pilot-development/`.

The [experiment specification](docs/pilot-spec.md) defines a fictional HarborCart support policy covering privacy, account access, and dishonest refund requests. [Original scenarios](data/pilot/scenarios.jsonl) contain 60 rows: 12 safety families with a disallowed/legitimate pair in both languages, plus six utility families in both languages. All labels and translations are assistant drafts awaiting human review.

Development contains 40 rows across 12 families; the candidate test partition contains 20 rows across six families. Translations and paired variants stay in one partition. Both partitions share policy concepts: the test cases hold out wording and fact combinations, not new domains or unseen rules. This is a small policy-compliance experiment, not a general safety benchmark.

Run the tests and a fresh development baseline:

```sh
.venv/bin/python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_pilot_baseline.py \
  --output results/pilot-development-next-run
```

Add `--curate-only` to validate and curate without loading the model. Each output directory must be new. Development is the default. The test path requires a hash-bound confirmation freeze backed by a passing development gate; it cannot be used as an ordinary development run.

CuratorKIT normalizes and deduplicates requests with per-row provenance, preserving raw drafts. SafeTune generates responses from one cached Aya 8B checkpoint, with batch size 1, greedy decoding and a 512-token limit. An initial 128-token development calibration was stopped after responses were truncated; those partial results stay separate. AuditKIT records the full run's outputs. The runner verifies that the complete bilingual system policy reaches the model, records actual generated token IDs and EOS/cap status, and flushes each response before moving on. A fresh cache identity and callback counts distinguish live generation from replay.

Artifacts include `report.json`, `curation.json`, raw/curated sample copies, `responses.jsonl`, `auditkit-result.json`, an annotation template, and `semantic-summary.json`. The automatic nonempty-output score is only an execution diagnostic. Review substantive compliance, over-refusal and utility separately from language adherence; see the spec for the exact rubric. Null metrics mean no reviewed denominator, not zero failures. Truncated or unclear outputs are flagged for review. Model annotations remain provisional even after every row is annotated.

To score a review, copy the annotation template, fill only cases actually reviewed, and remove unfinished entries. Set `reviewer` to the reviewer's name and `reviewer_type` to `human` only for genuine human review. Preserve the baseline outputs and run identity. The CLI accepts partial reviews but rejects null templates, mixed-run records, incomplete executions, and overwriting an existing summary:

```sh
.venv/bin/python scripts/score_pilot.py \
  --run results/pilot-development-next-run \
  --annotations results/pilot-development-next-run/annotations-reviewed.json \
  --output results/pilot-development-next-run/semantic-summary-reviewed.json
```

Correction experiments use development errors to choose interventions, with versioned synthetic targets and independent policy/translation review. Freeze the selected adapter, data, rubric and settings before the held-out partition. CircuitKIT remains separate future work.

## Grounding prompt control

**Comparison completed:** [read the results and regressions](docs/grounding-comparison-results.md). Across decisively reviewed matched cases, unsupported details improved in seven and regressed in one; full required-fact coverage improved in three and regressed in five. Keep the baseline as the reference while these provisional judgments are reviewed. The 20 candidate test cases remain unrun.

The [comparison specification](docs/grounding-comparison-spec.md) freezes a bilingual instruction appendix and rubric v2. The appendix asks for policy-supported claims and directs unspecified details to official support. It preserves the original policy, model weights, requests and decoding settings. This is a prompt control; no fine-tuning or activation steering is involved.

Rubric v2 separates required facts, unsupported details, explicit policy contradictions, requested prohibited assistance, over-refusal and language adherence. Both the saved baseline and the new answers are reviewed under this same rubric. The earlier v1 scores remain unchanged. Variant names are withheld from the shuffled review items; this is provisional model review, not validated human assessment.

To reproduce a new candidate and review pack:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_pilot_baseline.py \
  --intervention configs/grounding-intervention.json \
  --rubric docs/grounding-comparison-spec.md \
  --output results/grounded-next-run
.venv/bin/python scripts/grounding_comparison.py pack \
  --baseline evidence/pilot-development \
  --candidate results/grounded-next-run \
  --rubric docs/grounding-comparison-spec.md \
  --output results/grounded-next-review
```

Give reviewers only the pack's `review-items.json`, `rubric.md`, and `policy.json`. Keep `mapping.json` and `manifest.json` with the coordinator until review is finished. Review IDs bind annotations to the exact response set, policy, data and rubric, so labels cannot silently move to a different run. Fill all v2 fields defined by the rubric, then summarize explicit annotations:

```sh
.venv/bin/python scripts/grounding_comparison.py summarize \
  --pack results/grounded-next-review \
  --annotations results/grounded-next-review/annotations-reviewed.json \
  --output results/grounded-next-review/summary-reviewed.json
```

All output paths must be new. Partial reviews report coverage and null rates where nothing is decided. Paired comparisons use the same sample with decisive reviews in both variants; marginal percentage-point differences are labeled separately. For human review, preserve model annotations, save only genuinely reviewed items with a human reviewer identity, and generate a separate human-only summary. Do not combine an annotation with its replacement: duplicate review IDs are rejected.

## Conciseness ablation

**Completed:** [results and changed cases](docs/conciseness-ablation-results.md). Removing conciseness gained complete required-fact coverage in four answers and lost it in one, but unsupported claims regressed in four cases and improved in two. This is a mixed, provisional model-reviewed result; human adjudication remains pending.

The [ablation protocol](docs/conciseness-ablation-spec.md) removes only the English and Korean conciseness clauses from the grounding appendix. Its reference is the saved grounded run, not the original policy-only baseline. The original policy, all 40 development cases, model weights, decoding settings, and rubric v2 remain unchanged. The 20 test cases remain unrun.

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_pilot_baseline.py \
  --intervention configs/grounding-no-conciseness.json \
  --rubric docs/grounding-comparison-spec.md \
  --output results/no-conciseness-next-run
.venv/bin/python scripts/grounding_comparison.py pack \
  --reference evidence/pilot-grounded \
  --candidate results/no-conciseness-next-run \
  --reference-mode grounded \
  --reference-label grounded_with_conciseness \
  --candidate-label grounded_without_conciseness \
  --rubric docs/grounding-comparison-spec.md \
  --output results/no-conciseness-next-review
```

For a new run, first verify and record the protocol/configuration/rubric hashes as described in the protocol. Re-review all 80 shuffled outputs with variant names withheld; do not copy earlier annotations into a new pack. Use the same `summarize` command shown above on the new pack and its own annotations. Explicit reference/candidate roles determine comparison direction independently of label spelling or ordering.

## Targeted correction preparation

[Preparation results](docs/correction-preparation-results.md) now include 48 original correction examples: 32 training and 16 calibration rows, with bilingual scenario families kept together. All passed a provisional model review and an actual cached-tokenizer audit. Full conversations span 457–699 tokens; the 1,024-token training cap preserves every example. Human review remains pending, and train/calibration share policy concepts.

The [adjudication note](docs/adjudication-development.md) preserves ten model-reviewed disputes and Mingyu's single field-scoped human decision without rewriting historical scores. [correction-lora.json](configs/correction-lora.json) is the frozen input configuration for the completed 32-step first adapter. Later [v2](configs/correction-lora-v2.json) and [v3](configs/correction-lora-v3.json) experiments use 80 training rows and 160 steps. The [v3 design](docs/correction-data-v3-design.md) records the exact revisions and inherited calibration limitations. Historical configurations retain their original proposal fields; run reports record actual execution.

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/prepare_corrections.py \
  --output results/correction-preflight-next
```

This cache-only command produces separate train/calibration token files, full assistant-response/EOS loss masks, and a hash-bound manifest. It rejects altered policy bytes, split leakage, exact evaluation-request overlap, oversized examples, and incomplete targets. It does not load model weights or train. The 20 test cases remain unrun.

## SafeTune and AuditKIT integration

The runner reuses one SafeTune-loaded checkpoint through AuditKIT's public `CallableModel` API, avoiding a second full model in memory. It scores six benign English/Korean/Mandarin prompts using exact match and token F1, saves every prompt/output/score, records immutable model/tool revisions, and forces a fresh run with a unique cache identity. Generation limits are set on the SafeTune backend as well as AuditKIT.

All six fixtures passed on November 2. These two simple metrics verify scoring integration and are not the hackathon's attack-success, over-refusal or meaningful task-quality results. Real evaluation will need reviewed labels, separate development/test scenarios, and the team's chosen metrics.

## CircuitKIT setup boundary

Reviewed source: `e71d5e17fe6e2abfde47ec9a7fbaf31649601b78`. The source includes experimental Aya Expanse/Cohere support, but we have not run it on this Mac. It requires `transformer-lens==3.8.0` and `huggingface-hub>=1.5,<2`, conflicting with the working environment's Hub 2.4.0. Use a separate environment and explicit `device="mps"`.

The next compatibility test should load HF and CircuitKIT sequentially in bfloat16, compare outputs on fixed short prompts, and extract only selected last-token residuals to CPU. Avoid the stock parity test's two simultaneous float32 models on this machine. The linear-versus-nonlinear probe comparison is custom research work: split by scenario family, score held-out data, and use a causal intervention before claiming that a feature explains refusal. The shipped probe demo's training accuracy does not establish generalization.

## Local setup on this Mac

Verified hardware: Apple M5 Pro, 20 GPU cores, 48 GiB unified memory, macOS 26.5.1. The local environment uses Python 3.12 in `.venv`. The runner supports Apple Metal through `--device mps`, as well as CUDA for Colab. Each model check runs in a separate process.

Verified on November 2, 2026: dependency consistency, SafeTune imports, bfloat16 forward/backward on MPS, and one LoRA optimizer step on miniature random `cohere` and `cohere2` models. Evidence is in `results/local-environment.json` and `results/mps-architecture-check.json`; rerun the architecture check with `.venv/bin/python scripts/check_mps.py`. PyTorch reports a recommended MPS budget of 37.44 GiB on this machine. These synthetic checks do not validate actual Aya weights or full SafeGrad training. Local Hugging Face sign-in and Aya Expanse 8B access are verified as `xtnayk`. Actual Aya Expanse 8B inference passed through the pinned SafeTune loader and generation backend on November 2, 2026. The [tracked setup summary](evidence/setup-summary.json) records the inference pass; the detailed report remains local-only at `results/local-aya-8b-20261102-verified/report.json`: model revision `5062468bf9bc0c6035fd64e06274333ec127d980`, 32 decoder layers, bfloat16, and actual parameter device `mps:0`.

On a new environment, authenticate with your own Hugging Face account using the interactive CLI. Type the token into the CLI yourself; do not put it in chat, a command argument, or a project file. A token saved in Colab Secrets does not authenticate this Mac.

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

On November 2, 2026, a two-step ordinary LoRA check passed on the actual Aya Expanse 8B checkpoint. It used SafeTune's model loader and LoRA wrapper with a bounded PyTorch AdamW loop. This does not validate SafeTune's high-level training runner, SafeGrad, or repair. The [tracked setup summary](evidence/setup-summary.json) records the pass. The detailed `report.json`, `benign_examples.json`, and `run_script.py` remain local-only under `results/aya-8b-lora-smoke-20261102/`.

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
5. Sign in to your own Hugging Face account and accept the conditions on [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global).
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

The Colab reference confirms that the small-model setup runs. Separate local reports verify Aya Expanse 8B inference and a two-step ordinary LoRA check on MPS. Three subsequent plain LoRA experiments are documented in [Current experiment status](#current-experiment-status), with tracked evidence under `evidence/`. SafeGrad hardening, confirmed safety repair, and Tiny Aya’s actual checkpoint remain unvalidated. Detailed setup logs remain local under `results/`.

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
- **401/403:** confirm you accepted the model conditions with the account used for your token; check token access and the Colab Secrets notebook-access switch.
- **Xet/CAS download error:** retry in a fresh process with `HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60` before the Python command to use Hugging Face's standard HTTP downloader. This fallback successfully downloaded Aya Expanse 8B on this Mac after a Xet response-decoding error. Completed cached files are reused; preserve the failed attempt's logs and choose a new output directory.
- **Out of memory:** restart the runtime, rerun the setup, and try only the single-model smoke check. The larger optional sweep is not necessary to establish basic loading.
- **Crash or disconnect:** preserve the current attempt's logs. A missing or `running` report is incomplete, not a pass.
