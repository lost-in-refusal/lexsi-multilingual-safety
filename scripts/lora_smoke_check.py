#!/usr/bin/env python3
"""Two-step benign Aya LoRA experiment; saves an adapter, never merges or publishes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import subprocess
import time
import traceback

MODEL = "CohereLabs/aya-expanse-8b"
REVISION = "5062468bf9bc0c6035fd64e06274333ec127d980"
SAFETUNE_COMMIT = "26f59e6d1bdcacafe5f2be3b8d982fb214369659"
EXAMPLES = [
    {"prompt": "Give a short definition of reproducibility in an experiment.",
     "answer": "Reproducibility means obtaining consistent results when an experiment is repeated using the same methods and settings."},
    {"prompt": "Name two useful items to record in an experiment log.",
     "answer": "Record the experiment settings and the observed results."},
]
PROBE = "Why is it useful to save the settings used in an experiment? Answer briefly."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", stage="environment", started_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Two-step ordinary LoRA compatibility check, not SafeGrad, repair, or safety/utility evaluation.",
                  model=MODEL, model_revision=REVISION, steps=[], memory_snapshots=[])

    def write_report():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def stage(name):
        report["stage"] = name
        write_report()
        print(name, flush=True)

    write_report()
    try:
        import torch
        import safetune
        from huggingface_hub import snapshot_download
        from peft import get_peft_model_state_dict
        from safetune.runner.utils.model_utils import load_model, load_tok, lora_wrap
        source = Path(__file__).resolve().parents[1] / "vendor/SafeTune"
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
        assert commit == SAFETUNE_COMMIT and not dirty.strip(), "SafeTune source pin/cleanliness mismatch"
        assert Path(safetune.__file__).resolve().is_relative_to(source.resolve())
        assert torch.backends.mps.is_available(), "MPS unavailable"
        report["safetune_commit"] = commit
        report["safetune_tracked_files_clean"] = True
        report["packages"] = {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "safetune")}
        report["configuration"] = dict(seed=42, steps=2, batch_size=1, max_sequence_tokens=128,
            learning_rate=1e-4, optimizer="AdamW", weight_decay=0.0, lora_r=4, lora_alpha=8,
            lora_dropout=0.0, gradient_checkpointing=True, use_reentrant=False,
            loss="assistant tokens only", max_generation_new_tokens=64)
        report["mps_recommended_max_gib"] = round(torch.mps.recommended_max_memory() / 2**30, 3)
        torch.mps.set_per_process_memory_fraction(0.8)
        report["mps_allocator_limit_fraction_of_recommended"] = 0.8
        report["initial_system_swap"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
        snapshot = snapshot_download(MODEL, revision=REVISION, local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
        safetune.configure(device="mps", seed=42)
        torch.manual_seed(42)
        tok = load_tok(snapshot)

        stage("prepare_data")
        batches = []
        for row in EXAMPLES:
            messages = [{"role": "user", "content": row["prompt"]}]
            prefix = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
            full = tok.apply_chat_template(messages + [{"role": "assistant", "content": row["answer"]}],
                tokenize=True, add_generation_prompt=False, return_dict=False)
            assert full[:len(prefix)] == prefix, "Chat prefix boundary mismatch"
            assert len(full) <= 128, "Example exceeds smoke-check sequence limit; no silent truncation"
            labels = [-100] * len(prefix) + full[len(prefix):]
            assert any(t != -100 for t in labels[1:]), "No supervised next-token targets"
            assert tok.encode(row["answer"], add_special_tokens=False), "Empty answer"
            batches.append(dict(input_ids=torch.tensor([full]), attention_mask=torch.ones(1, len(full), dtype=torch.long),
                                labels=torch.tensor([labels])))
            row["sequence_tokens"] = len(full)
            row["prompt_tokens"] = len(prefix)
            row["supervised_tokens"] = sum(t != -100 for t in labels[1:])
            row["decoded_supervised_text"] = tok.decode(full[len(prefix):], skip_special_tokens=True)
            assert row["decoded_supervised_text"].strip() == row["answer"], "Supervised text mismatch"
        data_json = json.dumps(EXAMPLES, indent=2) + "\n"
        (args.output / "benign_examples.json").write_text(data_json)
        report["examples_sha256"] = hashlib.sha256(data_json.encode()).hexdigest()
        report["examples"] = EXAMPLES
        if args.prepare_only:
            report.update(status="prepared", stage="data_only")
            return 0

        def memory(name):
            torch.mps.synchronize()
            report["memory_snapshots"].append(dict(stage=name,
                mps_tensor_gib=round(torch.mps.current_allocated_memory() / 2**30, 3),
                mps_driver_gib=round(torch.mps.driver_allocated_memory() / 2**30, 3),
                process_max_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 3)))
            write_report()

        stage("load_model")
        started = time.perf_counter()
        model = load_model(snapshot)
        torch.mps.synchronize()
        report["model_load_seconds"] = round(time.perf_counter() - started, 3)
        report["base_dtype"] = str(next(model.parameters()).dtype)
        report["model_device"] = str(next(model.parameters()).device)
        memory("loaded")

        def generate():
            model.eval()
            inputs = tok.apply_chat_template([{"role": "user", "content": PROBE}], tokenize=True,
                add_generation_prompt=True, return_tensors="pt", return_dict=True).to("mps")
            started = time.perf_counter()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                    use_cache=True, pad_token_id=tok.pad_token_id)
            torch.mps.synchronize()
            new = output[0, inputs["input_ids"].shape[1]:]
            answer = tok.decode(new, skip_special_tokens=True)
            assert answer.strip(), "Empty probe response"
            return dict(prompt=PROBE, answer=answer, generated_tokens=new.numel(),
                        seconds=round(time.perf_counter() - started, 3))

        stage("baseline_generation")
        report["before_training"] = generate()
        memory("baseline_generation")
        model = lora_wrap(model, r=4, lora_alpha=8, lora_dropout=0.0)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert trainable and all("lora_" in n for n in trainable), "Unexpected trainable base parameters"
        report["trainable_parameters"] = sum(p.numel() for p in trainable.values())
        report["total_parameters"] = sum(p.numel() for p in model.parameters())
        report["trainable_dtypes"] = sorted({str(p.dtype) for p in trainable.values()})
        assert report["trainable_dtypes"] == ["torch.float32"], "Expected float32 LoRA adapters"
        assert model.is_gradient_checkpointing, "Gradient checkpointing was not enabled"
        report["target_modules"] = sorted(model.peft_config["default"].target_modules)
        report["all_base_parameters_frozen"] = all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)
        before = {n: p.detach().cpu().clone() for n, p in trainable.items()}
        frozen = [(n, p, p.detach().flatten()[:1024].cpu().clone())
                  for n, p in model.named_parameters() if not p.requires_grad]
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-4, weight_decay=0.0, foreach=False)
        memory("adapter_ready")
        training_started = time.perf_counter()
        for step_index, cpu_batch in enumerate(batches, 1):
            stage(f"training_step_{step_index}")
            model.train()
            batch = {k: v.to("mps") for k, v in cpu_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            loss = model(**batch, use_cache=False).loss
            assert torch.isfinite(loss).item(), "Nonfinite training loss"
            loss_value = loss.item()
            memory(f"step_{step_index}_forward")
            loss.backward()
            grads = [p.grad for p in trainable.values() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all().item() for g in grads), "Missing/nonfinite gradients"
            assert all(p.grad is None for _, p, _ in frozen), "Frozen base received gradients"
            grad_norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True).item()
            assert grad_norm > 0, "Zero training gradient"
            memory(f"step_{step_index}_backward")
            optimizer.step()
            memory(f"step_{step_index}_optimizer")
            report["steps"].append(dict(step=step_index, loss=loss_value, gradient_norm_before_clip=grad_norm,
                seconds=round(time.perf_counter() - started, 3), finite_gradients=True))
            del loss, batch, grads
            write_report()
        report["training_seconds"] = round(time.perf_counter() - training_started, 3)
        changes = [(p.detach().cpu().float() - before[n].float()).abs().max().item() for n, p in trainable.items()]
        report["changed_adapter_tensors"] = sum(v > 0 for v in changes)
        report["maximum_adapter_absolute_change"] = max(changes)
        assert report["changed_adapter_tensors"] > 0, "Adapter did not update"
        assert all(torch.isfinite(p).all().item() for p in trainable.values()), "Nonfinite adapter weights"
        report["frozen_parameter_samples_unchanged"] = all(torch.equal(old, p.detach().flatten()[:1024].cpu()) for _, p, old in frozen)
        report["frozen_parameter_samples_checked"] = len(frozen)
        assert report["frozen_parameter_samples_unchanged"], "Frozen base sample changed"
        optimizer.zero_grad(set_to_none=True)
        del optimizer, before, frozen
        gc.collect()
        torch.mps.empty_cache()
        model.gradient_checkpointing_disable()
        stage("post_training_generation")
        report["after_training"] = generate()
        memory("after_training_generation")

        stage("save_and_reload_adapter")
        adapter = args.output / "adapter"
        model.peft_config["default"].base_model_name_or_path = MODEL
        model.peft_config["default"].revision = REVISION
        model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        original_state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model, adapter_name="default").items()}
        model.load_adapter(adapter, adapter_name="reloaded", is_trainable=False)
        reloaded_state = get_peft_model_state_dict(model, adapter_name="reloaded")
        report["saved_adapter_reload_exact"] = original_state.keys() == reloaded_state.keys() and all(
            torch.equal(v, reloaded_state[k].detach().cpu()) for k, v in original_state.items())
        assert report["saved_adapter_reload_exact"], "Reloaded adapter differs"
        model.set_adapter("reloaded")
        report["after_reload"] = generate()
        assert report["after_reload"]["answer"] == report["after_training"]["answer"], "Reloaded response differs"
        report["adapter_path"] = str(adapter.resolve())
        report["adapter_weights_bytes"] = sum(p.stat().st_size for p in adapter.glob("*.safetensors"))
        memory("after_reload_generation")
        report["maximum_observed_stage_mps_driver_gib"] = max(s["mps_driver_gib"] for s in report["memory_snapshots"])
        report["memory_measurement_scope"] = "MPS allocations sampled at synchronized stage boundaries, not a continuous or guaranteed peak. Process max RSS is an OS high-water reading; neither measures total system memory."
        report["final_system_swap"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
        report.update(status="passed", stage="complete")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        (args.output / "error.log").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_report()
        print(f"{report['status'].upper()}: {args.output / 'report.json'}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
