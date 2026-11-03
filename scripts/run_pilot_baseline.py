#!/usr/bin/env python3
"""CuratorKIT -> SafeTune -> AuditKIT, development partition only.

Every generated output is flushed before the next case. Semantic judgments are
separate from the nonempty-output integration diagnostic; no labels are guessed.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import uuid

try:
    from .pilot_validation import summarize_annotations, validate_rows
except ImportError:
    from pilot_validation import summarize_annotations, validate_rows

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonempty_output(sample, output):
    """Execution diagnostic only, not safety or utility quality."""
    return float(bool(output.strip()))


def make_nonempty_scorer():
    import auditkit as ak
    return ak.scorer(direction=ak.Direction.MAXIMIZE)(nonempty_output)


def build_effective_prompts(policy, intervention=None):
    """Preserve the base policy verbatim; optionally append one bilingual control."""
    def texts(value):
        result = value.get("texts") if isinstance(value, dict) else None
        if not isinstance(result, dict) or set(result) != {"en", "ko"}:
            raise ValueError("Prompt texts must contain exactly en and ko")
        if any(not isinstance(text, str) or not text.strip() for text in result.values()):
            raise ValueError("Prompt texts must be nonempty strings")
        return result

    base = texts(policy)
    if intervention is None:
        return dict(base)
    if not isinstance(intervention, dict) or intervention.get("type") != "system_prompt_appendix":
        raise ValueError("Only system_prompt_appendix interventions are supported")
    for key in ("intervention_id", "version"):
        if not isinstance(intervention.get(key), str) or not intervention[key].strip():
            raise ValueError(f"Intervention {key} must be a nonempty string")
    appendix = texts(intervention)
    return {lang: base[lang] + "\n\n" + appendix[lang] for lang in ("en", "ko")}


def validate_adapter_snapshot(report_bytes, config_bytes, adapter_config_bytes, runtime, policy_sha256):
    """Validate archived training metadata without reopening mutable weight paths."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    digest = lambda content: hashlib.sha256(content).hexdigest()
    report, config, adapter_config = map(json.loads, (report_bytes, config_bytes, adapter_config_bytes))
    require(all(isinstance(x, dict) for x in (report, config, adapter_config)), "Adapter snapshots must be objects")
    require(report.get("status") == "passed_training", "Adapter requires passed_training")
    require(report.get("training_config") == config and report.get("training_config_sha256") == digest(config_bytes),
            "Training configuration snapshot/hash mismatch")
    for field in ("model", "model_revision"):
        require(report.get(field) == config.get(field) == runtime.get(field), f"Adapter base {field} mismatch")
    require(config.get("policy_sha256") == policy_sha256, "Adapter training policy mismatch")
    require(config.get("system_prompt") == "original_policy_only", "Adapter must train under the original policy only")
    require(config.get("dtype") == runtime.get("dtype") == "bfloat16"
            and config.get("device") == runtime.get("local_device") == "mps"
            and report.get("base_dtype") == "torch.bfloat16" and report.get("model_device") == "mps:0",
            "Adapter training dtype/device mismatch")
    require(type(report.get("optimizer_steps")) is int and report["optimizer_steps"] == 32
            and type(config.get("train_steps")) is int and config["train_steps"] == 32,
            "Expected all 32 frozen training steps")
    for flag in ("saved_adapter_reload_exact", "all_base_parameters_frozen", "frozen_parameter_samples_unchanged"):
        require(report.get(flag) is True, f"Adapter training check is missing: {flag}")
    require(report.get("adapter_directory") == "adapter", "Expected relative adapter directory 'adapter'")
    files = report.get("adapter_files")
    require(isinstance(files, dict) and {"adapter_config.json", "adapter_model.safetensors"} <= set(files)
            and set(files) <= {"adapter_config.json", "adapter_model.safetensors", "README.md"},
            "Expected adapter-only configuration and safetensors files")
    require(all(isinstance(value, str) and len(value) == 64
                and set(value) <= set("0123456789abcdef") for value in files.values()), "Invalid adapter file digest")
    require(files["adapter_config.json"] == digest(adapter_config_bytes), "Adapter configuration hash mismatch")
    targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    require(config.get("lora_r") == 4 and config.get("lora_alpha") == 8 and config.get("lora_dropout") == 0.0,
            "Unexpected frozen LoRA configuration")
    require(isinstance(config.get("target_modules"), list) and set(config["target_modules"]) == targets,
            "Unexpected frozen LoRA target modules")
    for field, expected in (("base_model_name_or_path", runtime["model"]), ("revision", runtime["model_revision"]),
                            ("peft_type", "LORA"), ("task_type", "CAUSAL_LM"), ("r", 4),
                            ("lora_alpha", 8), ("lora_dropout", 0.0), ("bias", "none")):
        require(adapter_config.get(field) == expected, f"Saved PEFT configuration mismatch: {field}")
    require(isinstance(adapter_config.get("target_modules"), list)
            and set(adapter_config["target_modules"]) == targets, "Saved PEFT target modules mismatch")
    require(adapter_config.get("inference_mode") is True, "Saved PEFT configuration must be inference-ready")
    for field in ("modules_to_save", "rank_pattern", "alpha_pattern", "use_dora", "use_rslora", "lora_bias",
                  "trainable_token_indices", "target_parameters", "layer_replication", "fan_in_fan_out",
                  "exclude_modules", "layers_to_transform", "layers_pattern", "use_qalora", "use_bdlora",
                  "alora_invocation_tokens"):
        require(not adapter_config.get(field), f"Unapproved saved PEFT feature: {field}")
    return {"schema_version": 1, "model": runtime["model"], "model_revision": runtime["model_revision"],
            "policy_sha256": policy_sha256, "training_report_sha256": digest(report_bytes),
            "training_config_sha256": digest(config_bytes), "adapter_config_sha256": digest(adapter_config_bytes),
            "adapter_files": files, "optimizer_steps": 32, "saved_adapter_reload_exact": True}


def load_adapter_training_run(directory, runtime, policy_sha256):
    """Verify the exact weight files immediately before inference; no model load."""
    directory = Path(directory)
    snapshots = {"adapter-training-report.json": (directory / "report.json").read_bytes(),
                 "adapter-training-config.json": (directory / "training-config.json").read_bytes(),
                 "adapter-config.json": (directory / "adapter/adapter_config.json").read_bytes()}
    identity = validate_adapter_snapshot(*snapshots.values(), runtime, policy_sha256)
    adapter_path = directory / "adapter"
    actual = {str(p.relative_to(adapter_path)) for p in adapter_path.rglob("*") if p.is_file()}
    if actual != set(identity["adapter_files"]):
        raise ValueError("Adapter directory files differ from training manifest")
    for name, expected in identity["adapter_files"].items():
        if sha256(adapter_path / name) != expected:
            raise ValueError(f"Adapter file hash mismatch: {name}")
    return {"adapter_path": adapter_path, "identity": identity, "snapshots": snapshots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curate-only", action="store_true", help="Validate and curate without loading Aya")
    treatments = parser.add_mutually_exclusive_group()
    treatments.add_argument("--intervention", type=Path, help="Bilingual system-prompt appendix JSON; weights stay unchanged")
    treatments.add_argument("--adapter-training-run", type=Path, help="Completed hash-verified LoRA training run; original policy only")
    parser.add_argument("--rubric", type=Path, help="Rubric to snapshot before generation; required with either intervention")
    args = parser.parse_args()
    if (args.intervention or args.adapter_training_run) and not args.rubric:
        parser.error("--intervention and --adapter-training-run require --rubric")
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = json.loads((ROOT / "configs/runtime.json").read_text())
    config = json.loads((ROOT / "configs/pilot.json").read_text())
    pins = json.loads((ROOT / "configs/tools.lock.json").read_text())
    report = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                  runtime=runtime, config=config, variant="adapted" if args.adapter_training_run else "grounded" if args.intervention else "baseline",
                  callback_calls=0, generated_samples=0,
                  test_generated_samples=0, semantic_review="pending",
                  scope="Draft fictional support-policy development pilot; no general safety claim.")

    def stage(name):
        report["stage"] = name
        write_json(args.output / "report.json", report)
        print(name, flush=True)

    stage("environment")
    try:
        import curatorkit
        from pilot_curation import curate_rows

        assert config["split"] == "development", "This runner deliberately does not execute test cases"
        assert config["batch_size"] == 1
        assert config["max_new_tokens"] > 0
        report["tools"] = {}
        for name in ("CuratorKIT", "SafeTune", "AuditKIT"):
            source = ROOT / "vendor" / name
            revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
            assert revision == pins[name]["revision"] and not dirty, f"{name}: source pin/cleanliness mismatch"
            report["tools"][name] = revision
        assert Path(curatorkit.__file__).resolve().is_relative_to((ROOT / "vendor/CuratorKIT").resolve())
        report["source_hashes"] = {path: sha256(ROOT / path) for path in (
            "scripts/run_pilot_baseline.py", "scripts/pilot_curation.py", "scripts/pilot_validation.py",
            "configs/runtime.json", "configs/pilot.json", "configs/tools.lock.json")}
        report["packages"] = {name: importlib.metadata.version(name) for name in (
            "curatorkit", "safetune", "auditkit", "torch", "transformers", "huggingface-hub")}

        stage("curation")
        dataset_path = ROOT / "data/pilot/scenarios.jsonl"
        policy_path = ROOT / "data/pilot/policy.json"
        rows = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
        policy = json.loads(policy_path.read_text())
        intervention = json.loads(args.intervention.read_text()) if args.intervention else None
        effective_prompts = build_effective_prompts(policy, intervention)
        write_json(args.output / "effective-prompts.json", effective_prompts)
        report["effective_prompts_sha256"] = sha256(args.output / "effective-prompts.json")
        if intervention is not None:
            report["intervention"] = intervention
            report["intervention_sha256"] = sha256(args.intervention)
            (args.output / "intervention.json").write_bytes(args.intervention.read_bytes())
            config["experiment"] = "harborcart-bilingual-" + intervention["intervention_id"]
        if args.rubric:
            report["rubric_sha256"] = sha256(args.rubric)
            (args.output / "rubric.md").write_bytes(args.rubric.read_bytes())
        report["dataset_sha256"] = sha256(dataset_path)
        report["policy_sha256"] = sha256(policy_path)
        adapter_run = None
        if args.adapter_training_run:
            adapter_run = load_adapter_training_run(args.adapter_training_run, runtime, report["policy_sha256"])
            for name, content in adapter_run["snapshots"].items():
                (args.output / name).write_bytes(content)
            write_json(args.output / "adapter-provenance.json", adapter_run["identity"])
            report["adapter_provenance"] = adapter_run["identity"]
            report["adapter_provenance_sha256"] = sha256(args.output / "adapter-provenance.json")
            report["adapter_packages"] = {name: importlib.metadata.version(name) for name in ("peft", "safetensors")}
            config["experiment"] = "harborcart-bilingual-adapter-" + report["adapter_provenance_sha256"][:12]
        report["dataset"] = validate_rows(rows)
        assert report["dataset"]["counts_by_split"] == {"development": 40, "test": 20}
        curated, manifest = curate_rows(rows, "data/pilot/scenarios.jsonl")
        write_json(args.output / "raw-samples.json", rows)
        write_json(args.output / "curated-samples.json", curated)
        write_json(args.output / "policy.json", policy)
        write_json(args.output / "curation.json", manifest)
        report["curated_sha256"] = sha256(args.output / "curated-samples.json")
        selected = [row for row in curated if row["split"] == config["split"]]
        assert len(selected) == 40
        assert len({row["request"] for row in selected}) == len(selected), "Ambiguous request-to-ID mapping"
        report["selected_ids"] = [row["id"] for row in selected]
        report["selected_dataset"] = validate_rows(selected)
        if intervention is None and adapter_run is None:
            write_json(args.output / "semantic-summary.json", summarize_annotations(selected, []))
            write_json(args.output / "annotation-template.json", [dict(
                sample_id=row["id"], outcome=None, task_score=None, language_adherence=None, reviewer=None,
                reviewer_type=None, notes="", category=row["category"], language=row["language"],
                required_points=row["required_points"], forbidden_features=row["forbidden_features"]
            ) for row in selected])
        else:
            write_json(args.output / "review-status.json", {
                "status": "pending", "rubric_version": 2, "human_review_pending": len(selected),
                "workflow": "Use scripts/grounding_comparison.py pack with the preserved baseline, then review both variants under rubric.md.",
                "note": "No v1 template or v1 semantic summary is produced for this comparison.",
            })
        if args.curate_only:
            report["status"] = "curated_only"
            stage("complete")
            return 0

        import torch
        import safetune
        import auditkit as ak
        from auditkit.model import CallableModel
        from huggingface_hub import snapshot_download
        from safetune.runner.utils.model_utils import load_model, load_tok
        from safetune.core.eval.pipeline import GenerationConfig, TransformersBackend

        for name, package in (("SafeTune", safetune), ("AuditKIT", ak)):
            assert Path(package.__file__).resolve().is_relative_to((ROOT / "vendor" / name).resolve())
        device = runtime["local_device"]
        assert device == "mps" and torch.backends.mps.is_available(), "This baseline is configured for local MPS"
        torch.mps.set_per_process_memory_fraction(config["mps_memory_fraction"])
        stage("load_model")
        snapshot = snapshot_download(runtime["model"], revision=runtime["model_revision"], local_files_only=True,
                                     allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
        safetune.configure(device=device, seed=runtime["seed"])
        tok = load_tok(snapshot)
        started = time.perf_counter()
        model = load_model(snapshot, dtype=runtime["dtype"], device=device)
        if adapter_run is not None:
            from peft import PeftModel
            # Recheck bytes after base loading; do not trust a stale path/hash.
            current_adapter = load_adapter_training_run(args.adapter_training_run, runtime, report["policy_sha256"])
            assert current_adapter["identity"] == adapter_run["identity"], "Adapter changed while loading base model"
            model = PeftModel.from_pretrained(model, str(current_adapter["adapter_path"]), is_trainable=False)
            assert all(not p.requires_grad for p in model.parameters()), "Inference adapter unexpectedly trainable"
            assert not any(getattr(module, "merged", False) for module in model.modules()), "Adapter was unexpectedly merged"
            report["adapter_loaded_for_inference"] = True
            report["adapter_merged"] = False
        model.eval()
        torch.mps.synchronize()
        report["model_load_seconds"] = round(time.perf_counter() - started, 3)
        report["model_device"] = str(next(model.parameters()).device)
        report["dtype"] = str(next(model.parameters()).dtype)
        assert report["dtype"] == "torch." + runtime["dtype"]
        gen_config = GenerationConfig(max_new_tokens=config["max_new_tokens"], batch_size=1,
                                      temperature=config["temperature"], seed=runtime["seed"])
        backends = {lang: TransformersBackend(model, tok, config=gen_config, system_prompt=effective_prompts[lang])
                    for lang in ("en", "ko")}
        # Precompute the intended complete chat inputs. The observer rejects any
        # silent raw-prompt fallback or tokenizer truncation inside the backend.
        expected_ids = {}
        for row in selected:
            text = tok.apply_chat_template([
                {"role": "system", "content": effective_prompts[row["language"]]},
                {"role": "user", "content": row["request"]},
            ], tokenize=False, add_generation_prompt=True)
            expected_ids[row["id"]] = tok(text, add_special_tokens=False)["input_ids"]
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
        report["eos_token_ids"] = eos_ids
        original_generate = model.generate
        observations = []
        current = {}

        def observed_generate(*args_, **kwargs):
            assert not args_ and kwargs["input_ids"].shape[0] == 1
            input_ids = kwargs["input_ids"][0].tolist()
            assert input_ids == expected_ids[current["id"]], "Policy/template missing or input truncated"
            torch.mps.synchronize()
            start = time.perf_counter()
            output = original_generate(**kwargs)
            torch.mps.synchronize()
            seconds = time.perf_counter() - start
            new_ids = output[0, len(input_ids):].tolist()
            eos = bool(new_ids and new_ids[-1] in eos_ids)
            observations.append(dict(input_tokens=len(input_ids), output_tokens=len(new_ids),
                ended_with_eos=eos, cap_reached=len(new_ids) >= config["max_new_tokens"] and not eos,
                generation_seconds=round(seconds, 3), generated_token_ids=new_ids))
            return output

        model.generate = observed_generate
        by_request = {row["request"]: row for row in selected}
        completed_ids = set()
        responses = []
        with (args.output / "responses.jsonl").open("w") as checkpoints:
            def generate(prompts):
                report["callback_calls"] += 1
                outputs = []
                for prompt in prompts:
                    row = by_request[prompt]
                    assert row["split"] == "development" and row["id"] not in completed_ids
                    current["id"] = row["id"]
                    observation_count = len(observations)
                    started_case = time.perf_counter()
                    output = backends[row["language"]].generate([prompt])[0]
                    assert len(observations) == observation_count + 1
                    observation = observations[-1]
                    assert output == tok.decode(observation["generated_token_ids"], skip_special_tokens=True)
                    record = dict(sample_id=row["id"], family_id=row["family_id"], split=row["split"],
                        language=row["language"], category=row["category"], request=prompt, output=output,
                        policy_sha256=report["policy_sha256"], **observation,
                        effective_prompts_sha256=report["effective_prompts_sha256"],
                        case_seconds=round(time.perf_counter() - started_case, 3))
                    if adapter_run is not None:
                        record["adapter_provenance_sha256"] = report["adapter_provenance_sha256"]
                    checkpoints.write(json.dumps(record, ensure_ascii=False) + "\n")
                    checkpoints.flush()
                    os.fsync(checkpoints.fileno())
                    responses.append(record)
                    outputs.append(output)
                    completed_ids.add(row["id"])
                    report["generated_samples"] += 1
                    write_json(args.output / "report.json", report)
                    print(f"{len(completed_ids)}/{len(selected)} {row['id']} tokens={record['output_tokens']} cap={record['cap_reached']}", flush=True)
                return outputs

            stage("auditkit_evaluation")
            started = time.perf_counter()
            samples = [ak.Sample(id=row["id"], input=row["request"], metadata={
                key: row[key] for key in ("family_id", "split", "language", "category")}) for row in selected]
            report["run_nonce"] = uuid.uuid4().hex
            result = ak.evaluate(samples,
                model=CallableModel(generate, name=f"safetune:{runtime['model']}@{runtime['model_revision']}:{report['variant']}"),
                scorers=[make_nonempty_scorer()],
                config=ak.RunConfig(seed=runtime["seed"], temperature=config["temperature"],
                    max_tokens=config["max_new_tokens"], batch_size=1, concurrency=1,
                    max_retries=0, track_performance=False, extra={
                        "run_nonce": report["run_nonce"], "dataset_sha256": report["dataset_sha256"],
                        "curated_sha256": report["curated_sha256"], "policy_sha256": report["policy_sha256"],
                        "model_revision": runtime["model_revision"], "tools": report["tools"],
                        "variant": report["variant"], "effective_prompts_sha256": report["effective_prompts_sha256"],
                        "intervention_sha256": report.get("intervention_sha256"),
                        "rubric_sha256": report.get("rubric_sha256"),
                        **({"adapter_provenance_sha256": report["adapter_provenance_sha256"]} if adapter_run is not None else {})}))
        model.generate = original_generate
        report["evaluation_seconds"] = round(time.perf_counter() - started, 3)
        result.metadata.update({"scope": report["scope"], "semantic_review": "pending",
                                "diagnostic_only": True, "policy_sha256": report["policy_sha256"]})
        result.save(str(args.output / "auditkit-result.json"))
        assert not result.errors and result.failed_count == 0, "AuditKIT execution failure"
        assert report["callback_calls"] > 0 and report["generated_samples"] == len(selected), "Fresh generation missing"
        assert len(result.predictions) == len(selected)
        assert {p.sample_id for p in result.predictions} == completed_ids
        recorded_outputs = {r["sample_id"]: r["output"] for r in responses}
        assert all(p.raw_output == recorded_outputs[p.sample_id] for p in result.predictions)
        report["responses_sha256"] = sha256(args.output / "responses.jsonl")
        report["diagnostics"] = dict(nonempty_output=result.headline,
            cap_hits=sum(r["cap_reached"] for r in responses),
            eos_count=sum(r["ended_with_eos"] for r in responses),
            output_tokens=sum(r["output_tokens"] for r in responses),
            cap_hits_by_language={lang: sum(r["cap_reached"] for r in responses if r["language"] == lang)
                                  for lang in ("en", "ko")})
        report["auditkit_fingerprint"] = result.fingerprint
        report["mps_driver_gib_after_evaluation"] = round(torch.mps.driver_allocated_memory() / 2**30, 3)
        report["status"] = "passed_execution"
        stage("complete")
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error_type=type(exc).__name__, error=str(exc))
        (args.output / "error.log").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "report.json", report)
        print(f"{report['status'].upper()}: {args.output / 'report.json'}", flush=True)
    return 0 if report["status"] == "passed_execution" else 1


if __name__ == "__main__":
    raise SystemExit(main())
