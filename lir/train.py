"""LoRA fine-tuning of Tiny Aya Global: plain vs SafeTune SafeGrad.

Both arms share data, order, LoRA config and optimizer settings. The SafeGrad
arm uses SafeTune's SafeGradHFTrainer (global gradient surgery + KL to a frozen
copy of the base model on the MultiJail alignment slice). SafeTune's trainer
refuses gradient accumulation, which we need on a laptop GPU, so the subclass
below runs the surgery once per optimizer step on the accumulated gradient.

  python -m lir.train --arm plain --output runs/plain
  python -m lir.train --arm safegrad --output runs/safegrad
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from itertools import cycle
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from lir.common import DATA, MODEL, MODEL_REV, SEED, device, read_jsonl, sha256_file, write_json

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
MAX_LEN = 1024


def encode(tok, prompt: str, response: str) -> dict | None:
    user = [{"role": "user", "content": prompt}]
    prefix = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(user + [{"role": "assistant", "content": response}], tokenize=False)
    assert full.startswith(prefix)
    p_ids = tok(prefix, add_special_tokens=False).input_ids
    ids = tok(full, add_special_tokens=False).input_ids
    if len(ids) > MAX_LEN:
        return None
    assert ids[:len(p_ids)] == p_ids
    return {"input_ids": ids, "labels": [-100] * len(p_ids) + ids[len(p_ids):]}


def collate(pad_id: int):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id)
        labels = torch.full((len(batch), n), -100)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return fn


class MaskedLoss:
    """Next-token loss computed only where labels exist. Projecting the 262k-vocab
    head over the ~370-token system preamble wastes memory and time, so logits are
    taken just for the response positions (logits_to_keep with a boolean mask).
    Returns the micro-batch token mean; the Trainer divides by accumulation steps."""

    loss_is_scaled_for_ga = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        tgt = labels[:, 1:] != -100
        keep = torch.zeros_like(labels, dtype=torch.bool)
        keep[:, :-1] = tgt
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], logits_to_keep=keep)
        loss = torch.nn.functional.cross_entropy(out.logits.float(), labels[:, 1:][tgt])
        return (loss, out) if return_outputs else loss


class PlainTrainer(MaskedLoss, Trainer):
    pass


def make_safegrad_trainer():
    from safetune.core.optim.safegrad import safegrad_kl_alignment_loss
    from safetune.interventions.harden.safegrad import SafeGradHFTrainer

    class AccumulatingSafeGrad(MaskedLoss, SafeGradHFTrainer):
        def __init__(self, *args, safety_dataset, rho=1.0, kl_temperature=1.0, **kw):
            Trainer.__init__(self, *args, **kw)  # skip the accumulation check, see module docstring
            self._safety_dataset = safety_dataset
            self._safety_iter = cycle(iter(safety_dataset))
            self._rho, self._kl_temperature = float(rho), float(kl_temperature)
            # The frozen reference is the student with its LoRA adapters switched off: with
            # dropout 0 and frozen base weights that is exactly the base model, without a
            # second 6.7 GB copy in memory.
            self._reference_model = None
            self.surgery_log = []

        def _reference_logits(self, model, **kw):
            with torch.no_grad(), model.disable_adapter():
                return model(**kw).logits

        def _alignment_gradient(self, model, batch):
            # Same as SafeTune's KL mode, but padding positions are excluded from
            # the KL (SafeTune's loss sums over every position).
            batch = {k: v.to(model.device) for k, v in batch.items()}
            keep = batch["attention_mask"].bool()
            student = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            logits_to_keep=keep).logits  # (non-pad tokens, vocab)
            ref = self._reference_logits(model, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                         logits_to_keep=keep)
            # SafeTune's KL runs softmax in the input dtype; in bf16 two identical distributions
            # come out at KL ~0.6 per sequence, so upcast to fp32. Chunks keep the 262k-vocab
            # fp32 tensors small; the KL is a sum over positions, so chunking is exact.
            loss = 0.0
            for i in range(0, student.shape[0], 512):
                loss = loss + safegrad_kl_alignment_loss(student[i:i + 512].float().unsqueeze(0),
                                                         ref[i:i + 512].float().unsqueeze(0),
                                                         temperature=self._kl_temperature)
            loss = loss / keep.shape[0]
            params = self._trainable_params(model)
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            self._last_kl = float(loss.detach())
            return [g.detach() if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = Trainer.training_step(self, model, inputs, num_items_in_batch)
            if self.accelerator.sync_gradients:
                g_align = self._alignment_gradient(model, next(self._safety_iter))
                params = self._trainable_params(model)
                gu = torch.cat([p.grad.reshape(-1).float() for p in params])
                ga = torch.cat([g.reshape(-1).float() for g in g_align])
                cos = float((gu * ga).sum() / (gu.norm() * ga.norm() + 1e-12))
                self._apply_global_surgery(model, g_align)
                self.surgery_log.append({"step": self.state.global_step + 1, "kl": self._last_kl, "cos": cos,
                                         "projected": cos < 0, "g_user_norm": float(gu.norm()),
                                         "g_align_norm": float(ga.norm())})
            return loss

    return AccumulatingSafeGrad


class LogCallback(TrainerCallback):
    def __init__(self, path: Path):
        self.path, self.rows, self.t0 = path, [], time.time()

    def on_step_end(self, args, state, control, **kw):
        # Long-sequence batches grow the MPS cache; on a 48 GB machine that tips into swap.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def on_log(self, args, state, control, logs=None, **kw):
        row = dict(logs or {}, step=state.global_step, elapsed_s=round(time.time() - self.t0, 1))
        if torch.backends.mps.is_available():
            row["mps_driver_gib"] = round(torch.mps.driver_allocated_memory() / 2**30, 2)
        self.rows.append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["plain", "safegrad"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--align-batch", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--train-file", default=str(DATA / "bactrian_train.jsonl"))
    ap.add_argument("--align-file", default=str(DATA / "alignment_sft.jsonl"))
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED)

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    rows = read_jsonl(Path(a.train_file))
    enc = [encode(tok, r["instruction"], r["response"]) for r in rows]
    dropped = sum(e is None for e in enc)
    train_ds = [e for e in enc if e is not None]

    from peft import LoraConfig, get_peft_model

    dev = device()
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REV, dtype=torch.bfloat16,
                                                 attn_implementation="eager").to(dev)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    accum = a.batch // a.micro_batch
    args = TrainingArguments(
        output_dir=str(out / "hf"), per_device_train_batch_size=a.micro_batch, gradient_accumulation_steps=accum,
        num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=0.03, weight_decay=0.0, max_grad_norm=1.0, logging_steps=1, save_strategy="no",
        report_to=[], seed=SEED, data_seed=SEED, remove_unused_columns=False, dataloader_num_workers=0,
        bf16=False, fp16=False, use_cpu=False, train_sampling_strategy="group_by_length")
    log_cb = LogCallback(out / "train_log.jsonl")
    common = dict(model=model, args=args, train_dataset=train_ds, data_collator=collate(tok.pad_token_id),
                  callbacks=[log_cb])

    align_info = None
    if a.arm == "plain":
        trainer = PlainTrainer(**common)
    else:
        arows = read_jsonl(Path(a.align_file))
        aenc = [encode(tok, r["prompt"], r["response"]) for r in arows]
        aenc = [e for e in aenc if e is not None]
        g = torch.Generator().manual_seed(SEED)
        loader = torch.utils.data.DataLoader(aenc, batch_size=a.align_batch, shuffle=True, generator=g,
                                             collate_fn=collate(tok.pad_token_id))
        trainer = make_safegrad_trainer()(**common, safety_dataset=loader, rho=a.rho)
        align_info = {"file": a.align_file, "sha256": sha256_file(Path(a.align_file)), "rows": len(aenc),
                      "batch": a.align_batch, "rho": a.rho, "reference": "base model = student with adapters disabled"}

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    if a.arm == "safegrad":
        with open(out / "surgery_log.jsonl", "w") as f:
            for r in trainer.surgery_log:
                f.write(json.dumps(r) + "\n")
    steps = trainer.state.global_step
    write_json(out / "run.json", {
        "arm": a.arm, "model": MODEL, "model_revision": MODEL_REV, "lora": LORA, "trainable_params": n_train,
        "train_file": a.train_file, "train_sha256": sha256_file(Path(a.train_file)), "train_rows": len(train_ds),
        "dropped_over_max_len": dropped, "epochs": a.epochs, "lr": a.lr, "micro_batch": a.micro_batch,
        "grad_accum": accum, "effective_batch": a.batch, "optimizer_steps": steps,
        "scheduler": "cosine, 3% warmup", "seed": SEED, "alignment": align_info,
        "surgery_steps": len(getattr(trainer, "surgery_log", [])),
        "projected_steps": sum(r["projected"] for r in getattr(trainer, "surgery_log", [])),
        "train_loss": result.training_loss, "train_runtime_s": round(elapsed, 1),
        "device": dev, "platform": platform.platform(), "torch": torch.__version__,
        "safetune_api": None if a.arm == "plain" else
        "safetune.interventions.harden.safegrad.SafeGradHFTrainer._apply_global_surgery + "
        "safetune.core.optim.safegrad.safegrad_kl_alignment_loss",
    })
    if a.arm == "safegrad":
        assert len(trainer.surgery_log) == steps, "surgery did not run on every optimizer step"
    print("done", steps, "steps in", round(elapsed), "s")


if __name__ == "__main__":
    main()
