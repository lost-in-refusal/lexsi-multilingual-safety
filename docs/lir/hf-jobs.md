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
| `python -m lir.hfjob launch --job unmerged` | 14f (the arms the review note of Nov 22 under 12.2 names) | Plain and SafeGrad at seed 42 in bfloat16 with the adapter left unmerged | `gpu-plain-unmerged`, `gpu-safegrad-unmerged` |
| `python -m lir.hfjob launch --job unmerged-s1`, `--job unmerged-s2` | 14f | The same at seed 1 or 2, with the laptop's adapter of that seed | `gpu-plain-s1-unmerged`, `gpu-safegrad-s1-unmerged`, and the same with `s2` |
| `python -m lir.hfjob launch --job nf4-s1`, `--job nf4-s2` | 14f | Plain and SafeGrad at seed 1 or 2 with NF4 base weights | `gpu-plain-s1-nf4`, `gpu-safegrad-s1-nf4`, and the same with `s2` |

Defaults: `--flavor l4x1`, `--timeout 3h`, `--namespace lost-in-refusal-lexsi`. Section 12.0 asks for one GPU type for
every job, so change the flavor for all of them or none. Follow a job with `hf jobs logs <id>` and `hf jobs inspect <id>`.

## What a job uploads

To the private dataset `lost-in-refusal/lost-in-refusal-gpu-runs`, under `<job>/<UTC time>-<commit>/`:

- `job.json`: the commit, the GPU type and driver, the library versions, and for each step the command, the seconds it
  took, the peak GPU memory in use and the generation speed per set. Generation and the guard run as two steps (the
  two halves of `lir.evaluate`, same functions), so the memory of the `generate` step is the model's alone. That is
  the figure for the ROI table; the `guard` step's memory is the safety judge's.
- `logs/`: the full output of every step.
- `<arm>/`: `attack.gen.jsonl`, `attack.guard.jsonl`, `xstest.gen.jsonl` and `xstest.guard.jsonl`, as on the laptop.
- `<arm>/run/` for a trained arm: the adapter, its run record and its training log.

Nothing goes to the public outputs repository from a job. Results are copied there after they have been read.

To read them: `python -m lir.gpu fetch` copies every job's answers, verdicts, records and logs (no adapters) from the
staging dataset into `runs/gpu/`, and `python -m lir.gpu` writes `results/lir/stage2/gpu.md` with the readings of 12.0,
12.1 and 12.2. Neither loads a model.

## Section 14f: NF4 at three seeds

The five jobs of section 14f are `unmerged`, `unmerged-s1`, `nf4-s1`, `unmerged-s2` and `nf4-s2`: two evaluations each,
about 1.1 hours on an L4, and they can run side by side. The seed 1 and 2 adapters come from the private stage 2
adapter repository, so the token also needs read access to `lost-in-refusal/tiny-aya-lora-stage2`. To read them:
`python -m lir.gpu fetch`, then `python -m lir.nf4seeds`, which writes `results/lir/stage2/nf4_seeds.md` once all
three seeds are in (`--partial` writes it earlier, with no sentence). Do not run `python -m lir.gpu` for this: it
would rewrite the table of section 12.

## A model outside the Tiny Aya family: the `granite41` jobs

Not registered when this was written, and not launched. The main experiment (base, plain LoRA and SafeGrad at seeds
42, 1 and 2) on `ibm-granite/granite-4.1-3b` at revision `c0650403`, picked because `lir/train.py` runs on it
unchanged: the same seven LoRA layer names, a chat template that takes one user turn with no arguments, not gated,
Apache 2.0, and Korean and Chinese on its card. The key is `granite41` (`LIR_MODEL=granite41`), not `granite`, which
is the name of the Granite Guardian second judge in this repository.

| Command | What it runs | Arm it writes |
|---|---|---|
| `python -m lir.hfjob launch --job granite41-base` | The base model on the held-out prompts, the attack set and XSTest; then writes the model's own refusal set `alignment_sft_granite41.jsonl` from its safe refusals, as was done for Earth | `granite41-base` |
| `python -m lir.hfjob launch --job granite41-plain-s42`, `-s1`, `-s2` | Trains plain LoRA at that seed on `bactrian_train.jsonl`, then evaluates on the attack set and XSTest | `granite41-plain-s42`, `-s1`, `-s2` |
| `python -m lir.hfjob launch --job granite41-safegrad-s42`, `-s1`, `-s2` | Trains SafeGrad at that seed against the refusal set of the base job, then evaluates | `granite41-safegrad-s42`, `-s1`, `-s2` |

- **Order.** The base job and the three plain jobs need nothing from each other and can start together. A SafeGrad
  job takes the refusal set from the newest upload of `granite41-base` in the staging dataset, so it starts after
  that job has finished, and stops at once if the file is missing or empty. The three SafeGrad jobs can run side by
  side. The file a SafeGrad job trained against is uploaded with its record, and its hash is in `job.json` and in
  the run record.
- **SafeTune.** Only the SafeGrad jobs install it: `requirements-gpu-safegrad.txt`, then SafeTune without its
  dependencies at commit `3901a9f`, the one in the README. The other jobs install what they installed before.
- **Token.** Granite 4.1 is not gated. The token needs what the other jobs need: Jobs rights on the namespace and
  write access to the staging dataset.
- **Reading.** `python -m lir.gpu fetch`, then `python -m lir.granite41` writes
  `results/lir/stage2/granite41/seeds.md` once the base run and all six arms are in (`--partial` writes it earlier,
  with every verdict pending). It applies `analyse()` of `lir/seeds.py` to these arms, against this model's own
  base run, and stops if a run record does not show the model revision, seed, training file, settings and refusal
  file these jobs use. It does not write the tables of section 12 or 14f.
- **What differs from Tiny Aya.** Five Korean training rows are over the 1,024-token limit under Granite's tokenizer,
  so a run trains on 1,795 rows (600 English, 595 Korean, 600 Chinese). Granite's template adds no system prompt,
  where Tiny Aya's adds a safety preamble. Swahili is not among the languages on Granite's card.
- **Time and cost**, on one L4 at $0.80 an hour, sized from the step seconds of the section 12 job records
  (`runs/gpu/*/job.json`): plain training of Tiny Aya on 600 rows took 1,031 to 1,070 seconds (912 training tokens a
  second, 18.4 seconds an optimizer step), and an evaluation with its guard pass 1,880 to 2,117 seconds. Granite has
  the same 171 optimizer steps and 51% of Tiny Aya's training tokens, which bounds plain training between 27 minutes
  (by tokens) and 52 minutes (by steps); SafeGrad cost 7% to 19% more on the laptop. With four to six minutes to
  clone and install, which no record holds: the base job 37 to 43 minutes ($0.49 to $0.57), a plain job 63 to 94
  minutes ($0.83 to $1.25), a SafeGrad job 64 to 104 minutes ($0.86 to $1.38). All seven: 7.0 to 10.6 GPU hours,
  $5.57 to $8.46. Each stays under the 3 hour timeout.
- **Not checked on a GPU.** No Granite job has run. SafeGrad has never trained on an L4, where plain training of
  Tiny Aya peaked at 21.8 of 23.0 GB. Granite's smaller vocabulary and shorter sequences should need less, but that
  is an expectation. Launch `granite41-base` and one plain job first and read their logs, then one SafeGrad job
  before the other two. What was checked without a GPU: the unit tests, `--dry-run`, and the unchanged
  `lir/train.py` (plain and SafeGrad) and `lir/generate.py` on a random model of Granite's architecture with its real
  tokenizer and template, on a CPU, with only the packages a SafeGrad job installs.

## Suggested order tonight

`control`, then `nf4`, then `unmerged` if Aish agrees. NF4 is the only run that fills the organizers' write-up line
"results on the quantized / deployed model (4-bit)". The three `only-*` jobs repeat fine-tunes the laptop already has
(`single_language.md`), so they are the first to drop.

## Not covered, and three points on section 12 for Aish

- **The task set is not run.** Sections 12.1 and 12.3 ask for it, but its judge (Qwen3-14B in bfloat16, about 30 GB)
  does not fit the 24 GB of an L4, and 12.0 fixes one GPU type. Either the task set is dropped for these runs or the
  judge runs on the laptop afterwards from uploaded answers; say which in 12.1 before the run.
- **Merged against unmerged.** `lir.evaluate` merges the adapter into the weights, on the laptop and in the control.
  Section 12.2 loads NF4 with the adapter unmerged. So NF4 minus bfloat16 for plain and SafeGrad also contains the
  difference between a merged and an unmerged adapter. The `unmerged` job measures that difference in bfloat16. It is
  written into 12.2 as a review note (Nov 22); run it once Aish agrees, or state the confound.
- **The seed 42 adapters are the two stage 1 repositories** (`tiny-aya-global-bactrian-lora-plain` and `-safegrad`),
  not `tiny-aya-lora-stage2` as the brief says. The job downloads them from there at a pinned revision and stops if
  the weights do not have the hash of the file the laptop evaluated.
- **AlignTune (12.3) is not scripted.** It needs its own entry point once someone has checked that AlignTune can
  fine-tune Tiny Aya Global.
- **The laptop already has the single-language runs** (`plain-only-en`, `plain-only-ko`, `plain-only-zh`; section 13a on
  `mingyu/stage2`). Section 12.1 is no longer blind and should say so before `only-ko` is launched.

This script has been checked with `--dry-run` and unit tests only. No job has been submitted with it yet: the account
it was written on has no Jobs rights on the namespace. Run `control` first and read its log before anything else.
