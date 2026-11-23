"""A plain LoRA fine-tune of Tiny Aya Global run through Lexsi's AlignTune (D13; sections 12.3 and 14d of the preregistration).

  .venv-d13/bin/python -m lir.train_aligntune train --output runs/plain-aligntune
  .venv/bin/python -m lir.train_aligntune report

AlignTune lives in its own environment (.venv-d13), because it needs trl 1.7.1 and torch below 2.14 and the main
environment has torch 2.15. Everything after training (answers, guard, task judge, this report) runs in the main
environment on the adapter that AlignTune saved, as for every other arm.

What goes through AlignTune: the trainer is built by aligntune.core.backend_factory.create_sft_trainer, the model, the
tokenizer and the LoRA adapter are built by its setup_model, the training rows are loaded by its setup_data, and the
training itself is its own train(), which builds a trl.SFTTrainer and saves the adapter.

What we add between setup_data and train: each row's prompt and completion strings are wrapped as a user message and an
assistant message. With plain strings TRL tokenizes the text as it is, with no chat template; if the template is applied
to the strings beforehand, TRL adds a second start-of-text token (this is what stopped the first attempt). With message
lists TRL applies Tiny Aya's own chat template, and the token IDs and the prompt boundary come out the same as in
lir/train.py. That is checked on every row before the first training step; training does not start on a difference.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from lir.common import DATA, EVAL_LANGS, MODEL, MODEL_REV, ROOT, RUNS, git_commit, read_jsonl, sha256_file, write_json

ARM = "plain-aligntune"
CONFIG = ROOT / "configs" / "d13" / "aligntune.json"
ALIGNTUNE_COMMIT = "f53b41d0e50495278caea7c99a89505d9ddfae23"   # the commit Hannah's attempts used
OUT = ROOT / "results" / "lir" / "stage2"

# Settings of lir/train.py that this run does not reproduce. Written into the run record and into section 14d.
DIFFERENCES = [
    "library versions: trl 1.7.1 and torch 2.13.0 here (AlignTune requires torch below 2.14); lir/train.py ran on torch 2.15.0 "
    "with transformers' Trainer and no trl",
    "optimizer: AlignTune asks for adamw_torch; lir/train.py used the Trainer default, adamw_torch_fused. Same algorithm and "
    "the same betas, epsilon and weight decay",
    "loss: trl computes the next-token loss from logits at every position and masks the prompt; lir/train.py computes logits "
    "at the response positions only. Both are the mean over the response tokens of a micro-batch",
    "gradient checkpointing: switched on by the Trainer here; lir/train.py switches it on with use_reentrant=False",
    "batch order and LoRA initialisation: same seed (42) and the same grouping by length, but the two programs draw from the "
    "random generators in a different order, so the batches and the initial adapter weights are not the same",
    "data path: the rows pass through AlignTune's data manager (CuratorKIT), with its schema gate, cleaning and "
    "deduplication switched off; row count, order and text are checked against the training file",
]

T1_ROWS = ["Run through AlignTune, plain fine-tuning raised Korean attack success significantly and by at least the bar: "
           "the Korean drift is not specific to our training code",
           "Run through AlignTune, plain fine-tuning raised Korean attack success significantly, but by less than the bar",
           "Run through AlignTune, plain fine-tuning did not raise Korean attack success significantly: the drift was not "
           "reproduced through AlignTune with these settings"]


def load_config(path: Path = CONFIG) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    need = ["model", "model_revision", "dataset", "expected_rows", "epochs", "learning_rate", "batch_size",
            "gradient_accumulation_steps", "max_seq_length", "seed", "precision", "lora"]
    missing = [k for k in need if k not in cfg]
    if missing:
        raise ValueError(f"missing in {path}: {', '.join(missing)}")
    return cfg


def trainer_kwargs(cfg: dict, model_ref: str, output: Path, micro_batch: int | None = None) -> dict:
    """The arguments for aligntune.core.backend_factory.create_sft_trainer.

    micro_batch: a smaller micro-batch than the configured one, with the accumulation raised so that the effective batch
    stays the same (the fallback registered in section 14d for a run that does not fit in memory).
    Three things differ from the first attempt's arguments, each found by printing the SFTConfig AlignTune builds:
    warmup_steps is not passed (passing 0 overrode the 3% warm-up), the batch order is asked for as
    train_sampling_strategy (group_by_length is ignored by transformers 5), and no checkpoint is saved per epoch."""
    lora = cfg["lora"]
    batch, accum = cfg["batch_size"], cfg["gradient_accumulation_steps"]
    if micro_batch is not None and micro_batch != batch:
        if (batch * accum) % micro_batch:
            raise ValueError(f"micro-batch {micro_batch} does not divide the effective batch {batch * accum}")
        batch, accum = micro_batch, batch * accum // micro_batch
    return dict(
        model_name=model_ref, dataset_name=str((ROOT / cfg["dataset"]).resolve()), task_type="instruction_following",
        backend="trl", output_dir=str(output), num_epochs=cfg["epochs"], batch_size=batch,
        learning_rate=cfg["learning_rate"], max_seq_length=cfg["max_seq_length"], seed=cfg["seed"], data_seed=cfg["seed"],
        split_seed=cfg["seed"], precision=cfg["precision"], attn_implementation="eager",
        use_peft=True, lora_r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        lora_target_modules=list(lora["target_modules"]),
        gradient_accumulation_steps=accum, completion_only_loss=True, assistant_only_loss=False, packing=False,
        curator_schema_gate=False, curator_clean=False, curator_dedup="none", keep_columns=True,
        val_split_ratio=None, test_split_ratio=None,
        warmup_ratio=0.03, weight_decay=0.0, max_grad_norm=1.0, lr_scheduler="cosine",
        train_sampling_strategy="group_by_length", gradient_checkpointing=True, logging_steps=1,
        save_strategy="no", eval_strategy="no", load_best_model_at_end=False, report_to="none",
        bf16=False, fp16=False,   # the model is loaded in bfloat16; no autocast, as in lir/train.py
    )


SLOW_AFTER_STEPS = 5          # optimizer steps after which the training time is projected
SLOW_LIMIT_S = 5 * 3600       # a projection above this stops the run (section 14d)
NOT_FIT = 4                   # exit code: out of memory, or projected to take too long; the queue then tries the fallback once


def settings_of(args) -> dict[str, str]:
    """The training settings as trl's SFTConfig holds them, as strings."""
    names = ("num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type",
             "warmup_steps", "weight_decay", "max_grad_norm", "optim", "seed", "data_seed", "bf16", "fp16",
             "train_sampling_strategy", "gradient_checkpointing", "save_strategy", "max_length", "packing",
             "completion_only_loss", "assistant_only_loss", "loss_type")
    return {k: str(getattr(getattr(args, k), "value", getattr(args, k))) for k in names}


def expected_settings(cfg: dict, kw: dict) -> dict[str, str]:
    """What settings_of must return for the registered run (kw: the arguments from trainer_kwargs)."""
    return {"num_train_epochs": str(cfg["epochs"]), "per_device_train_batch_size": str(kw["batch_size"]),
            "gradient_accumulation_steps": str(kw["gradient_accumulation_steps"]), "learning_rate": str(cfg["learning_rate"]),
            "lr_scheduler_type": "cosine", "warmup_steps": "0.03", "weight_decay": "0.0", "max_grad_norm": "1.0",
            "optim": "adamw_torch", "seed": str(cfg["seed"]), "data_seed": str(cfg["seed"]), "bf16": "False", "fp16": "False",
            "train_sampling_strategy": "group_by_length", "gradient_checkpointing": "True", "save_strategy": "no",
            "max_length": str(cfg["max_seq_length"]), "packing": "False", "completion_only_loss": "True",
            "assistant_only_loss": "False", "loss_type": "nll"}


def projected_seconds(elapsed_s: float, steps_done: int, steps_total: int) -> float:
    """Training time if every step took as long as the steps so far. The longest batches come first, so this is high."""
    return elapsed_s / steps_done * steps_total


class NotFit(Exception):
    """The run does not fit on this machine as configured: out of memory, or too slow."""


def to_messages(row: dict) -> dict:
    """One row of AlignTune's dataset, with prompt and completion as message lists, so that TRL applies the chat template."""
    return {"prompt": [{"role": "user", "content": row["prompt"]}],
            "completion": [{"role": "assistant", "content": row["completion"]}]}


def compare_prepared(prepared, baseline: list[dict]) -> list[str]:
    """Problems between what TRL prepared and lir/train.py's encoding, row by row. An empty list means they are the same.

    prepared: rows with input_ids and completion_mask, in training-file order. baseline: lir.train.encode of each row."""
    if len(prepared) != len(baseline):
        return [f"{len(prepared)} prepared rows against {len(baseline)} baseline rows"]
    problems = []
    for i, base in enumerate(baseline):
        row = prepared[i]
        ids, mask = list(row["input_ids"]), list(row["completion_mask"])
        want = [0 if label == -100 else 1 for label in base["labels"]]
        if ids != base["input_ids"]:
            problems.append(f"row {i}: token IDs differ ({len(ids)} against {len(base['input_ids'])})")
        elif mask != want:
            problems.append(f"row {i}: the prompt boundary differs ({mask.count(0)} against {want.count(0)} prompt tokens)")
        if len(problems) >= 5:
            break
    return problems


def cached_revision(model: str) -> str | None:
    """The revision the local Hugging Face cache resolves `model` to, or None if it is not cached."""
    import os

    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    ref = hub / f"models--{model.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def train(a) -> None:
    import torch
    import transformers
    from transformers import TrainerCallback

    from lir.train import LORA, LogCallback, encode

    out = Path(a.output)
    if a.before and time.time() >= a.before:
        print(f"not launched: {ARM}. It is {time.strftime('%F %T')} and the limit was "
              f"{time.strftime('%F %T', time.localtime(a.before))}.")
        raise SystemExit(3)
    cfg = load_config(Path(a.config))
    rows = read_jsonl(ROOT / cfg["dataset"])
    if len(rows) != cfg["expected_rows"]:
        raise SystemExit(f"{cfg['dataset']}: {len(rows)} rows, expected {cfg['expected_rows']}")
    if (cfg["model"], cfg["model_revision"]) != (MODEL, MODEL_REV):
        raise SystemExit(f"{a.config} names {cfg['model']}@{cfg['model_revision'][:8]}, the pipeline uses {MODEL}@{MODEL_REV[:8]}")
    if (cfg["lora"]["r"], cfg["lora"]["alpha"], cfg["lora"]["dropout"], list(cfg["lora"]["target_modules"])) != (
            LORA["r"], LORA["lora_alpha"], LORA["lora_dropout"], LORA["target_modules"]):
        raise SystemExit(f"{a.config}: the LoRA settings are not those of lir/train.py")
    rehearsal = a.rehearsal is not None
    if rehearsal:
        model_ref = a.rehearsal   # a small random model with the real tokenizer, to try the whole path without the weights
    else:
        model_ref = MODEL       # the adapter then names the model the way lir/generate.py expects
        if cached_revision(MODEL) != MODEL_REV:
            raise SystemExit(f"the cache resolves {MODEL} to {cached_revision(MODEL)}, not to the pinned {MODEL_REV}")
    if out.exists() and not (out / "run.json").exists():   # an attempt that did not finish: kept beside, under another name
        out.rename(out.with_name(f"{out.name}.failed-{time.strftime('%H%M%S')}"))
    out.mkdir(parents=True, exist_ok=False)

    import aligntune
    import peft
    import trl
    from aligntune.core.backend_factory import create_sft_trainer

    kw = trainer_kwargs(cfg, model_ref, out / "adapter", micro_batch=a.micro_batch)
    if rehearsal:
        kw.update(device_map="cpu", max_steps=a.max_steps or 1, precision="fp32")   # bfloat16 is very slow on a CPU
    elif a.max_steps:
        raise SystemExit("--max-steps is for a rehearsal only")
    trainer = create_sft_trainer(**kw)
    kind = f"{type(trainer).__module__}.{type(trainer).__name__}"
    if kind != "aligntune.backends.trl.sft.sft_generation.TRLSFTTrainer":
        raise SystemExit(f"AlignTune chose {kind}, not its TRL trainer")

    trainer.setup_model()   # AlignTune loads the model and the tokenizer and applies the LoRA adapter
    model, tok = trainer.model, trainer.tokenizer
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lcfg = model.peft_config["default"]
    got = (lcfg.r, lcfg.lora_alpha, lcfg.lora_dropout, sorted(lcfg.target_modules))
    if got != (LORA["r"], LORA["lora_alpha"], LORA["lora_dropout"], sorted(LORA["target_modules"])):
        raise SystemExit(f"AlignTune built a different adapter: {got}")
    if not rehearsal:
        if lcfg.base_model_name_or_path != MODEL:   # lir/generate.py refuses an adapter that names another model
            raise SystemExit(f"the adapter names {lcfg.base_model_name_or_path} as its base model, not {MODEL}")
        loaded = getattr(model.config, "_commit_hash", None)   # not always set; the cache check above is the gate
        if loaded not in (None, MODEL_REV):
            raise SystemExit(f"the model was loaded at revision {loaded}, not {MODEL_REV}")
        expect = json.loads((RUNS / "plain" / "run.json").read_text())["trainable_params"]
        if n_train != expect:
            raise SystemExit(f"{n_train:,} trainable parameters, where the plain run has {expect:,}")
        if next(model.parameters()).dtype != torch.bfloat16:
            raise SystemExit(f"the model is in {next(model.parameters()).dtype}, not bfloat16")

    trainer.setup_data()    # AlignTune loads the training file
    loaded_rows = trainer.train_dataset
    if len(loaded_rows) != len(rows) or any(
            loaded_rows[i]["prompt"] != r["instruction"] or loaded_rows[i]["completion"] != r["response"] for i, r in enumerate(rows)):
        raise SystemExit("AlignTune's data manager did not return the training file row for row")
    as_messages = loaded_rows.map(to_messages, remove_columns=loaded_rows.column_names, desc="prompt and completion as messages")
    trainer.train_dataset = trainer.dataset = as_messages

    baseline = [encode(tok, r["instruction"], r["response"]) for r in rows]
    if any(b is None for b in baseline):
        raise SystemExit("a row is over the length limit in lir/train.py's encoding; the usual runs drop none")
    checked = {}

    class CheckBeforeTraining(TrainerCallback):
        """Stops before the first step unless the rows TRL prepared are the rows lir/train.py would train on."""

        def on_train_begin(self, args, state, control, train_dataloader=None, **_):
            problems = compare_prepared(train_dataloader.dataset, baseline)
            if problems:
                raise SystemExit("training not started: " + "; ".join(problems))
            batch = train_dataloader.collate_fn([train_dataloader.dataset[i] for i in range(args.per_device_train_batch_size)])
            for i in range(args.per_device_train_batch_size):
                k = len(baseline[i]["input_ids"])
                want = baseline[i]["labels"] + [-100] * (batch["labels"].shape[1] - k)
                if batch["labels"][i].tolist() != want or batch["input_ids"][i, :k].tolist() != baseline[i]["input_ids"]:
                    raise SystemExit(f"training not started: the collated batch differs from lir/train.py's at row {i}")
            got, want = settings_of(args), expected_settings(cfg, kw)
            wrong = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
            if wrong:
                raise SystemExit(f"training not started: settings differ from the registered ones (got, registered): {wrong}")
            if not rehearsal and str(args.device) != "mps":
                raise SystemExit(f"training not started: the device is {args.device}, not mps")
            checked.update(rows=len(baseline), settings=got, device=str(args.device), t0=time.time())
            print(f"checked before training: all {len(baseline)} rows have lir/train.py's token IDs and prompt boundary; "
                  f"device {args.device}", flush=True)

    class StopIfTooSlow(TrainerCallback):
        """After the first optimizer steps, stops a run whose projected training time is over the registered limit."""

        def on_step_end(self, args, state, control, **_):
            if state.global_step == SLOW_AFTER_STEPS and not rehearsal:
                need = projected_seconds(time.time() - checked["t0"], state.global_step, state.max_steps)
                checked["projected_s_after_first_steps"] = round(need)
                print(f"after {state.global_step} optimizer steps: projected training time {need / 3600:.1f} h", flush=True)
                if need > SLOW_LIMIT_S:
                    raise NotFit(f"projected training time {need / 3600:.1f} h, over the limit of {SLOW_LIMIT_S / 3600:.0f} h")

    log_cb = LogCallback(out / "train_log.jsonl")
    own = trainer.get_hf_callbacks
    trainer.get_hf_callbacks = lambda: own() + [CheckBeforeTraining(), StopIfTooSlow(), log_cb]

    t0 = time.time()
    try:
        result = trainer.train()   # AlignTune's own training call
    except NotFit as e:
        print(f"stopped, does not fit: {e}", flush=True)
        raise SystemExit(NOT_FIT)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        print(f"stopped, does not fit: {str(e)[:300]}", flush=True)
        raise SystemExit(NOT_FIT)
    elapsed = time.time() - t0
    if not checked:
        raise SystemExit("the check before training did not run")
    adapter = out / "adapter"
    if not (adapter / "adapter_model.safetensors").exists() or not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"AlignTune did not leave an adapter in {adapter}")
    named = json.loads((adapter / "adapter_config.json").read_text()).get("base_model_name_or_path")
    if not rehearsal and named != MODEL:
        raise SystemExit(f"the adapter names {named} as its base model, not {MODEL}")
    write_json(out / "run.json", {
        "arm": ARM, "rehearsal": rehearsal, "model": MODEL, "model_revision": MODEL_REV, "lora": LORA, "trainable_params": n_train,
        "train_file": str(ROOT / cfg["dataset"]), "train_sha256": sha256_file(ROOT / cfg["dataset"]), "train_rows": len(rows),
        "dropped_over_max_len": 0, "epochs": cfg["epochs"], "lr": cfg["learning_rate"], "micro_batch": kw["batch_size"],
        "grad_accum": kw["gradient_accumulation_steps"], "effective_batch": kw["batch_size"] * kw["gradient_accumulation_steps"],
        "optimizer_steps": result.get("steps"), "scheduler": "cosine, 3% warmup", "seed": cfg["seed"], "commit": git_commit(),
        "alignment": None, "train_loss": result.get("final_loss"), "train_runtime_s": round(elapsed, 1),
        "device": checked["device"], "platform": platform.platform(), "torch": torch.__version__,
        "through": "aligntune.core.backend_factory.create_sft_trainer -> TRLSFTTrainer.setup_model, setup_data, train -> trl.SFTTrainer",
        "aligntune": {"version": aligntune.__version__ if hasattr(aligntune, "__version__") else None, "commit": ALIGNTUNE_COMMIT},
        "versions": {"trl": trl.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "rows_checked_against_lir_train": checked["rows"], "settings_passed_to_trl": checked["settings"],
        "projected_s_after_first_steps": checked.get("projected_s_after_first_steps"),
        "differences_from_lir_train": DIFFERENCES + ([] if kw["batch_size"] == cfg["batch_size"] else [
            f"micro-batch {kw['batch_size']} with accumulation {kw['gradient_accumulation_steps']}, where lir/train.py has "
            f"{cfg['batch_size']} with {cfg['gradient_accumulation_steps']} (the same effective batch): the fallback of section 14d, "
            "taken after the run at the registered micro-batch did not fit in memory or was projected to take too long"]),
    })
    print(f"{ARM}: {result.get('steps')} optimizer steps, training loss {result.get('final_loss')}, {elapsed / 60:.1f} min")


# ---------------------------------------------------------------- the report (main environment)

def result(att: dict[str, dict], xs: dict[str, dict], task: dict[str, dict], arm: str = ARM, plain: str = "plain") -> dict:
    """T1 and the figures beside it. att, xs, task: run name -> {uid: row} for the attack set, XSTest and the task set.

    T1 is the K1 table of section 6 applied to this arm: Korean strict attack success against base, with the bar of
    half the average Korean rise of the three plain seeds. Everything else is reported with no rule."""
    from lir.repeat import pooled_scores, score_diff
    from lir.seeds import REFUSAL, by_family, change, control_result, rate

    t1 = control_result(att, arm, plain=plain)
    refuse = {a: by_family(xs[a], REFUSAL) for a in ("base", plain, arm) if a in xs}
    out = {"arm": arm, "plain_arm": plain, "t1": t1, "t1_report": T1_ROWS[t1["outcome_row"] - 1]}
    if len(refuse) == 3:
        out["over_refusal"] = {l: {"base": rate(refuse["base"], l), "plain": rate(refuse[plain], l), "aligntune": rate(refuse[arm], l),
                                   "vs_base": change(refuse["base"], refuse[arm], l), "vs_plain": change(refuse[plain], refuse[arm], l)}
                               for l in EVAL_LANGS}
    if all(a in task for a in ("base", plain, arm)):
        mean = lambda a: sum(r["task_score"] for r in task[a].values()) / len(task[a])
        out["task_score"] = {"base": mean("base"), "plain": mean(plain), "aligntune": mean(arm),
                             "vs_plain": pooled_scores([score_diff(task[plain], task[arm])]),
                             "vs_base": pooled_scores([score_diff(task["base"], task[arm])])}
    return out


def report_md(res: dict, record: dict | None) -> str:
    pv = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    t1 = res["t1"]
    ko, s = t1["vs_base"]["ko"], t1["strict"]
    lines = ["### A plain fine-tune through AlignTune (T1)", "",
             "Plain LoRA on the usual 1,800 rows at seed 42, trained through Lexsi's AlignTune (TRL backend) on the laptop, and "
             "evaluated like every other arm. Sections 12.3 and 14d of the preregistration fixed the reading before the run. "
             "One run, one seed. Answers capped at 256 tokens on the attack set and XSTest.", "",
             "| Arm | " + " | ".join(EVAL_LANGS) + " |", "|---|" + "---:|" * len(EVAL_LANGS),
             "| base | " + " | ".join(f"{s['base'][l]:.1f}" for l in EVAL_LANGS) + " |",
             f"| plain, our trainer (`{res['plain_arm']}`) | " + " | ".join(f"{s['plain'][l]:.1f}" for l in EVAL_LANGS) + " |",
             f"| plain, through AlignTune (`{res['arm']}`) | " + " | ".join(f"{s['control'][l]:.1f}" for l in EVAL_LANGS) + " |", "",
             "Strict attack success in %, 291 prompt families per language.", "",
             f"- **T1. Korean strict attack success against base: {ko['delta']:+.1f} points ({pv(ko['p'])}); the bar is "
             f"{t1['bar']:.2f}. Row {t1['outcome_row']} of the K1 table. {res['t1_report']}.**",
             "- Beside, no rule. Against base in the other languages: " + ", ".join(
                 f"{l} {t1['vs_base'][l]['delta']:+.1f} ({pv(t1['vs_base'][l]['p'])})" for l in EVAL_LANGS if l != "ko") + ".",
             "- Beside, no rule. Against plain at seed 42 from our own trainer, on the same laptop and the same prompts: " + ", ".join(
                 f"{l} {t1['vs_plain'][l]['delta']:+.1f} ({pv(t1['vs_plain'][l]['p'])})" for l in EVAL_LANGS) + ".",
             f"- Beside, no rule. Swahili refusal rate against base: {t1['sw_refusal_vs_base']['delta']:+.1f} points "
             f"({pv(t1['sw_refusal_vs_base']['p'])}). Korean on the guard's Unsafe class alone: "
             f"{t1['unsafe_class_only_ko']['rate']['base']:.1f}% for base, {t1['unsafe_class_only_ko']['rate']['plain']:.1f}% for "
             f"plain and {t1['unsafe_class_only_ko']['rate']['control']:.1f}% through AlignTune."]
    if "over_refusal" in res:
        o = res["over_refusal"]
        lines.append("- Beside, no rule. Over-refusal of safe prompts (XSTest), base, plain and through AlignTune: " + "; ".join(
            f"{l} {o[l]['base']:.1f}, {o[l]['plain']:.1f} and {o[l]['aligntune']:.1f} (against plain {o[l]['vs_plain']['delta']:+.1f}, "
            f"{pv(o[l]['vs_plain']['p'])})" for l in EVAL_LANGS) + ".")
    if "task_score" in res:
        t = res["task_score"]
        d = t["vs_plain"]
        lines.append(f"- Beside, no rule. Task score (1 to 7): {t['base']:.2f} for base, {t['plain']:.2f} for plain and "
                     f"{t['aligntune']:.2f} through AlignTune; against plain {d['mean']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}] "
                     f"on {d['prompts']} prompts.")
    if record:
        lines += ["", f"Run record: {record['train_rows']:,} rows, {record['optimizer_steps']} optimizer steps, training loss "
                  f"{record['train_loss']:.3f} (the plain run: 171 steps, 1.425), micro-batch {record['micro_batch']} with accumulation "
                  f"{record['grad_accum']}, {record['train_runtime_s'] / 60:.0f} minutes on {record['device']}, commit "
                  f"`{str(record['commit'])[:7]}`. AlignTune at commit `{record['aligntune']['commit'][:7]}`, trl "
                  f"{record['versions']['trl']}, torch {record['torch']}. Before the first step all "
                  f"{record['rows_checked_against_lir_train']:,} rows were checked to have the token IDs and the prompt boundary "
                  "of our own trainer.", "",
                  "Settings of our trainer that this run does not reproduce:", ""] + [f"- {d[0].upper() + d[1:]}." for d in record["differences_from_lir_train"]]
    return "\n".join(lines)


def report(a) -> None:
    from lir.metrics import load
    from lir.repeat import judged
    from lir.seeds import SEEDS

    runs = ["base", ARM] + [p for p, _ in SEEDS.values()]
    n = len(read_jsonl(DATA / "multijail_attack.jsonl"))
    att = judged(runs, "attack", n)
    if ARM not in att or any(r not in att for r in runs):
        print("pending, nothing written: " + ", ".join(r for r in runs if r not in att) + " not fully judged on the attack set")
        return
    xs = judged(["base", "plain", ARM], "xstest", len(load("base", "xstest")))
    task = judged(["base", "plain", ARM], "task", len(load("plain", "task")))
    record = json.loads((RUNS / ARM / "run.json").read_text())
    if record.get("rehearsal") or (record["model"], record["model_revision"], record["seed"], record["train_rows"]) != (MODEL, MODEL_REV, 42, 1800):
        raise SystemExit(f"runs/{ARM}/run.json is not the registered run")
    res = result(att, xs, task)
    res["run"] = {k: record[k] for k in ("train_rows", "optimizer_steps", "train_loss", "micro_batch", "grad_accum", "train_runtime_s",
                                         "device", "commit", "torch", "aligntune", "versions", "rows_checked_against_lir_train",
                                         "settings_passed_to_trl", "differences_from_lir_train")}
    md = report_md(res, record)
    write_json(OUT / "aligntune.json", res)
    (OUT / "aligntune.md").write_text(md + "\n")
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "report"])
    ap.add_argument("--output", default=str(RUNS / ARM))
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--before", type=float, default=None, help="epoch seconds; after it the run is not launched")
    ap.add_argument("--micro-batch", type=int, default=None, help="a smaller micro-batch with the same effective batch")
    ap.add_argument("--rehearsal", default=None, help="path of a small random model: tries the whole path on the CPU, one step")
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else report(a)


if __name__ == "__main__":
    main()
