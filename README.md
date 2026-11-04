# Multilingual instruction fine-tuning and safety preservation

This experiment compares the same Aya Expanse 8B checkpoint before fine-tuning,
after ordinary LoRA, and after a SafeGrad adaptation. Languages are English,
Korean, and Mandarin Chinese. Safety deterioration and improvement are measured
outcomes, not assumptions.

**Execution status:** the ordinary LoRA arm has completed all 1,000 updates.
The matched SafeGrad arm is training; three-arm generation and blinded semantic
review are pending. No model has passed the practical improvement criterion.

## Data and splits

Task examples come from the human-authored [Aya dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset),
revision `f9ea04583f02a8f86404ff6c58bf75fe637df8a2`, licensed Apache-2.0.
Safety prompts come from [MultiJail](https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail),
revision `93d65c778925d55973b9ff30c7bccc8545494eb6`, licensed MIT.
The preparation manifest records source-file hashes and all exclusions.

| Partition | English | Korean | Chinese | Purpose |
|---|---:|---:|---:|---|
| Instruction training | 450 | 100 | 450 | Optimizer updates |
| Task calibration | 6 | 6 | 6 | Loss diagnostics only |
| Task development | 12 | 12 | 12 | Intervention comparison |
| Safety development | 12 | 12 | 12 | Intervention comparison |
| Safety alignment | 24 | 24 | 24 | Reference distribution preservation |
| Task confirmation | 100 | 100 | 100 | Reserved final evaluation |
| Safety confirmation | 100 | 100 | 100 | Reserved final evaluation |

CuratorKIT cleans and deduplicates requests before deterministic splitting.
Exact normalized requests do not cross partitions. All three translations of
each MultiJail family stay together. Task examples have no supplied translation
group IDs, so semantic or translated-paraphrase independence is not certified.
The task splits are drawn from Aya's original training partition and are held
out from this experiment's fine-tuning only. Base-model pretraining contamination
cannot be ruled out. The Korean training pool is smaller; report language-level
and macro-averaged results instead of hiding this imbalance in pooled accuracy.

A lexical screen excludes potentially unsafe task examples. This is an auditable
filter, not a guarantee that every retained task or reference answer is correct
or harmless. Semantic review remains separate.

An input-only automated audit flagged unreliable references, ambiguous task
targets, benign requests in the safety partition, and some cross-language meaning
changes. These are provisional annotations, not externally verified labels.
Reference agreement is therefore not the correctness metric, and safety-probe
labels do not make refusal mandatory. Translation-family IDs do not guarantee
semantic equivalence; language gaps cannot automatically be attributed to the
model. All fixed cases remain in the primary denominators.

## Training comparison

Both arms start from the same frozen base, use the same data order and one
complete epoch, rank-4/alpha-8 LoRA, learning rate `1e-5`, and batch size one.
Both arms and all evaluations use explicitly pinned eager attention. On the
Apple MPS environment, an SDPA diagnostic produced different logits with and
without autograd at zero adapter weights (maximum absolute difference 0.5625);
eager attention produced bitwise identical logits. A runtime preflight therefore
requires exact reference/student equivalence before any optimizer update.
Only complete assistant responses plus EOS receive task loss. No sequence is
silently truncated. Memory preflight performs no optimizer update. Adapter
weights are saved separately and reloaded exactly; base weights remain frozen.

The SafeGrad arm computes a global task/alignment gradient conflict, projects a
conflicting task gradient, then adds the alignment gradient with rho=1. It uses
SafeTune's reference-to-student KL loss. The frozen reference is evaluated
sequentially with adapters disabled, avoiding a second full base allocation.
Zero adapter dropout and unchanged base weights are required for that equivalence.

Alignment uses the complete **prompt-only** MultiJail sequences, including the
generation prefix, with batchmean KL. This differs from prompt/response alignment
protocols and is explicitly a scoped adaptation, not an exact reproduction of
all paper settings. Failure to obtain finite alignment gradients must abort;
silently continuing as plain training would invalidate the comparison.

## Evaluation and claim boundary

Generation consumes only the system/user prefix; task reference answers are
never supplied to the model. Greedy decoding uses at most 512 new tokens, with
EOS, truncation, elapsed time, and actual output token IDs recorded. AuditKIT's
nonempty-output score is only an execution diagnostic.

An initial 256-token execution probe reached its cap in four of nine outputs.
Before semantic review, the cap was fixed at 512 for every main comparison arm.
That interrupted length probe is excluded from the model comparison.

Semantic review must separately report harmful assistance on safety prompts,
over-refusal on legitimate tasks, task correctness/completeness, and language
adherence. Unclear or capped responses are not silently counted as safe or
correct. Review variants blind and preserve disagreement. Macro results and
per-language counts are required. Confirmation generation is disabled until
selection rules and a passing candidate have been independently reviewed.

The 72-case development set is exploratory. It cannot establish superiority to
published work, and this repository makes no state-of-the-art claim. A research
comparison requires the same model, data, threat model, metrics, and compute
budget, plus independent confirmation and uncertainty estimates.

## Related research

- [Qi et al., 2023](https://arxiv.org/abs/2310.03693) found that fine-tuning can
  weaken alignment, including on benign datasets. It does not establish that
  every model or benign dataset must exhibit drift.
- [Deng et al., ICLR 2024](https://arxiv.org/abs/2310.06474) introduced MultiJail
  and studied multilingual attacks and a self-defense training approach.
- [Yi et al., 2025](https://arxiv.org/abs/2508.07172) proposed SafeGrad's global
  gradient projection and reference-distribution alignment. This experiment
  adapts its method to adapter-only training and a three-language setting.
- [Li et al., 2026](https://arxiv.org/abs/2609.22144) study multilingual safety
  signals across multiple layers and propose MMSAFE filtering. A multilayer
  observation alone therefore does not establish a new contribution.
- [Zhang et al., 2026](https://arxiv.org/abs/2602.11157) report that distilling
  apparently safe teacher responses can still worsen multilingual jailbreak
  resistance. Teacher-generated targets require behavioral validation too.

Recent primary work also includes [ASCENT](https://arxiv.org/abs/2610.08061),
[a factorial replay study](https://arxiv.org/abs/2610.38001),
[MSafeTemp](https://arxiv.org/abs/2610.40516), and
[shared safety pathways](https://arxiv.org/abs/2608.09095). The targeted literature
audit records their distinct supervision, update rules, and compute requirements.
A matched KL-only control is needed before assigning any combined-method benefit
specifically to projection. No such attribution is made by this three-arm study.

## Preparation

Use the pinned Python 3.12 environment and accept the model's conditions with
your own Hugging Face account. Cache the exact model revision in `config.json`.
From a fresh repository root, install the pinned Lexsi sources and dependencies:

```sh
git clone https://github.com/Lexsi-Labs/SafeTune.git vendor/SafeTune
git -C vendor/SafeTune checkout --detach 26f59e6d1bdcacafe5f2be3b8d982fb214369659
git clone https://github.com/Lexsi-Labs/AuditKIT.git vendor/AuditKIT
git -C vendor/AuditKIT checkout --detach 71316da345784ff220d5ed59337441a59ab31526
git clone https://github.com/Lexsi-Labs/CuratorKIT.git vendor/CuratorKIT
git -C vendor/CuratorKIT checkout --detach dd9e835301fcddc74a37501d9e207a5a060d209f
python3.12 -m venv .venv
.venv/bin/python -m pip install -r experiments/multilingual_repair/requirements.txt
.venv/bin/hf auth login
.venv/bin/hf download CohereLabs/aya-expanse-8b --revision 5062468bf9bc0c6035fd64e06274333ec127d980
```

Then invoke the experiment modules from the repository root using that environment:

```sh
.venv/bin/python -m experiments.multilingual_repair.prepare --output results/new-prepared
.venv/bin/python -m experiments.multilingual_repair.train --prepared results/new-prepared --arm plain --output results/plain-run
.venv/bin/python -m experiments.multilingual_repair.train --prepared results/new-prepared --arm safegrad --output results/safegrad-run
.venv/bin/python -m experiments.multilingual_repair.evaluate --prepared results/new-prepared --split development --output results/base-eval
.venv/bin/python -m experiments.multilingual_repair.evaluate --prepared results/new-prepared --split development --training-run results/plain-run --output results/plain-eval
```

Every output directory must be new. Original source data, model revisions,
prepared-file hashes, actual step identities, failures, and execution metadata
are retained. Short feasibility runs must be identified separately from complete
training. Run one MPS process at a time.

## Publication export

The publication exporter copies only the experiment modules, configuration,
requirements, scientific documentation, and multilingual tests. Scientific
prepared-data and run directories must be supplied explicitly. It creates a
new ignored directory, preserves each selected file byte-for-byte, and records
relative source paths, sizes, and SHA-256 hashes in `publication-manifest.json`.
Declared response, step, and frozen training-source hashes must match their
copied artifacts; available configuration and prepared-manifest bindings are
checked too.
It performs no Git commit or upload and includes no historical Git objects.

```sh
.venv/bin/python -m experiments.multilingual_repair.export_publication \
  --output results/publication-new \
  --prepared corpus=results/new-prepared \
  --run plain=results/plain-run \
  --run safegrad=results/safegrad-run \
  --run baseline=results/base-eval
```

Output retains the `experiments/multilingual_repair/` and `tests/` package paths.
The root README is an exact copy of this experiment README, and the repository's
ignore rules are retained. Dataset attribution and the three selected license
documents are included explicitly.
Selected data and reports appear under `data/corpus/` and `runs/<alias>/`.
Unlisted files and binary model weights are excluded. Confirmation text is
never read or copied; the unchanged preparation manifest retains its file hash.
Regenerate the complete prepared bundle from the pinned public sources before
training, because the publication copy deliberately lacks that reserved file.

JSON, JSONL, and documentation are checked for disallowed metadata and local
filesystem paths. A failed check rejects the export before its directory is
created; the exporter never rewrites a scientific artifact to make it pass.
Python source and tests are exempt from generic content-marker checks so they
can define and test the rejection rules. Additional private terms can be supplied
with `--deny-terms-file results/export-deny-terms.json`, an ignored local JSON
array of strings that is never exported. Those terms are checked in every file,
including Python source. Plausible credential tokens are rejected in every file.
Source selection still follows the fixed package boundary.

After independent review, add `--review-pack study=results/review-pack` and
`--review-scores study=results/review-scores` to export the blinded inputs,
validated original reviewer parts, released review key, and scored results.
The private mapping is excluded. Score exports require their matching pack;
file hashes, released-payload identity, and reviewer/protocol bindings are
verified before copying. Review the resulting files before a separate
publication step.
