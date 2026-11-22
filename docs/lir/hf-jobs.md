# Running the section 12 jobs on a Hugging Face GPU

`python -m lir.hfjob` starts one Hugging Face Job per block of section 12 of the preregistration (Aish's section,
parts 12.0 to 12.4). Each job clones this repository at the commit you are on, installs `requirements-gpu.txt`,
runs the same `lir.train` and `lir.evaluate` as the laptop, and uploads everything to a private staging dataset.

## Before the first job

- Use a commit that is pushed. The job clones from GitHub and refuses to start from uncommitted changes.
- Be logged in (`hf auth login`) or `export HF_TOKEN=...` in the shell. The token needs Jobs rights on
  `lost-in-refusal-lexsi`, the Tiny Aya Global licence accepted on its account, and write access to
  `lost-in-refusal`. The command passes the token by name only.
- Print the job first: add `--dry-run`. Nothing is submitted.

## The jobs

| Command | Section | What it runs | Arms it writes |
|---|---|---|---|
| `python -m lir.hfjob launch --job control` | 12.0 | Base and the laptop's seed 42 plain adapter, attack set and XSTest | `gpu-base`, `gpu-plain` |
| `python -m lir.hfjob launch --job only-ko` | 12.1 | Trains plain LoRA on the 600 Korean rows, then evaluates | `plain-ko` |
| `python -m lir.hfjob launch --job only-en`, `--job only-zh` | 12.1 | The same on the English or Chinese rows | `plain-en`, `plain-zh` |
| `python -m lir.hfjob launch --job nf4` | 12.2 | SafeGrad in bfloat16, then base, plain and SafeGrad with NF4 base weights | `gpu-safegrad`, `gpu-base-nf4`, `gpu-plain-nf4`, `gpu-safegrad-nf4` |
| `python -m lir.hfjob launch --job unmerged` | not registered | Plain and SafeGrad in bfloat16 with the adapter left unmerged | `gpu-plain-unmerged`, `gpu-safegrad-unmerged` |

Defaults: `--flavor l4x1`, `--timeout 3h`, `--namespace lost-in-refusal-lexsi`. Section 12.0 asks for one GPU type for
every job, so change the flavor for all of them or none. Follow a job with `hf jobs logs <id>` and `hf jobs inspect <id>`.

## What a job uploads

To the private dataset `lost-in-refusal/lost-in-refusal-gpu-runs`, under `<job>/<UTC time>-<commit>/`:

- `job.json`: the commit, the GPU type and driver, the library versions, and for each step the command, the seconds it
  took, the peak GPU memory in use and the generation speed per set.
- `logs/`: the full output of every step.
- `<arm>/`: `attack.gen.jsonl`, `attack.guard.jsonl`, `xstest.gen.jsonl` and `xstest.guard.jsonl`, as on the laptop.
- `<arm>/run/` for a trained arm: the adapter, its run record and its training log.

Nothing goes to the public outputs repository from a job. Results are copied there after they have been read.

## Not covered, and three points on section 12 for Aish

- **The task set is not run.** Sections 12.1 and 12.3 ask for it, but its judge (Qwen3-14B in bfloat16, about 30 GB)
  does not fit the 24 GB of an L4, and 12.0 fixes one GPU type. Either the task set is dropped for these runs or the
  judge runs on the laptop afterwards from uploaded answers; say which in 12.1 before the run.
- **Merged against unmerged.** `lir.evaluate` merges the adapter into the weights, on the laptop and in the control.
  Section 12.2 loads NF4 with the adapter unmerged. So NF4 minus bfloat16 for plain and SafeGrad also contains the
  difference between a merged and an unmerged adapter. The `unmerged` job measures that difference in bfloat16. It is
  not registered: add it to 12.2 before running it, or state the confound.
- **The seed 42 adapters are the two stage 1 repositories** (`tiny-aya-global-bactrian-lora-plain` and `-safegrad`),
  not `tiny-aya-lora-stage2` as the brief says. The job downloads them from there.
- **AlignTune (12.3) is not scripted.** It needs its own entry point once someone has checked that AlignTune can
  fine-tune Tiny Aya Global.
- **The laptop already has the single-language runs** (`plain-only-en`, `plain-only-ko`, `plain-only-zh`; section 12a on
  `mingyu/stage2`). Section 12.1 is no longer blind and should say so before `only-ko` is launched.

This script has been checked with `--dry-run` and unit tests only. No job has been submitted with it yet: the account
it was written on has no Jobs rights on the namespace. Run `control` first and read its log before anything else.
