#!/usr/bin/env python3
"""CuratorKIT -> SafeTune -> AuditKIT, with explicitly frozen test confirmation.

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
RUNNER_SOURCE_BYTES = Path(__file__).read_bytes()


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
    experiment = config.get("experiment_id", "harborcart_targeted_plain_lora_v1")
    require(experiment in {"harborcart_targeted_plain_lora_v1", "harborcart_complete_answer_lora_v2"},
            "Unknown adapter experiment; declare and review its schedule first")
    expected_steps = 32
    if experiment == "harborcart_complete_answer_lora_v2":
        expected_steps = 160
        for field, expected in (("epochs", 2), ("learning_rate", 0.00005),
                                ("expected_rows", {"train": 80, "calibration": 16}),
                                ("expected_families", {"train": 24, "calibration": 4})):
            require(type(config.get(field)) is type(expected) and config[field] == expected,
                    f"Unexpected frozen v2 training setting: {field}")
    require(type(report.get("optimizer_steps")) is int and report["optimizer_steps"] == expected_steps
            and type(config.get("train_steps")) is int and config["train_steps"] == expected_steps,
            f"Expected all {expected_steps} frozen training steps")
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
            "adapter_files": files, "optimizer_steps": expected_steps, "saved_adapter_reload_exact": True}


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


def validate_confirmation_snapshot(blobs, runtime, config, *, dataset_sha256, policy_sha256, rubric_sha256,
                                   adapter_identity=None):
    """Validate an archived selection decision, including its development evidence."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    digest = lambda data: hashlib.sha256(data).hexdigest()
    canonical = lambda value: (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    freeze = json.loads(blobs["confirmation-freeze.json"])
    gate = json.loads(blobs["development-gate-report.json"])
    manifest = json.loads(blobs["development-pack-manifest.json"])
    goal = json.loads(blobs["confirmation-goal-config.json"])
    require(all(isinstance(x, dict) for x in (freeze, gate, manifest, goal)), "Confirmation snapshots must be objects")
    require(freeze.get("schema_version") == 1 and freeze.get("phase") == "heldout_confirmation",
            "Expected a held-out confirmation freeze")
    require(isinstance(freeze.get("confirmation_id"), str) and len(freeze["confirmation_id"]) >= 16,
            "Confirmation requires a unique ID of at least 16 characters")
    selected = freeze.get("selected_test_ids")
    require(isinstance(selected, list) and len(selected) == 20 and all(isinstance(x, str) and x for x in selected)
            and len(set(selected)) == 20 and freeze.get("family_count") == 6, "Freeze must select exactly 20 test IDs and six families")
    for field, expected in (("dataset_sha256", dataset_sha256), ("policy_sha256", policy_sha256),
                            ("rubric_sha256", rubric_sha256), ("model", runtime.get("model")),
                            ("model_revision", runtime.get("model_revision"))):
        require(freeze.get(field) == expected, f"Confirmation {field} mismatch")
    settings = {"seed": runtime.get("seed"), **{key: config.get(key) for key in (
        "temperature", "max_new_tokens", "batch_size", "mps_memory_fraction")}}
    require(settings == freeze.get("generation_settings") == {
        "seed": 42, "temperature": 0.0, "max_new_tokens": 512, "batch_size": 1, "mps_memory_fraction": 0.8},
        "Confirmation generation settings mismatch")
    require(runtime.get("dtype") == "bfloat16" and runtime.get("local_device") == "mps",
            "Confirmation requires the frozen bf16 MPS runtime")
    require(freeze.get("goal_config_sha256") == digest(blobs["confirmation-goal-config.json"])
            and freeze.get("protocol_sha256") == digest(blobs["confirmation-protocol.md"]),
            "Confirmation goal or protocol snapshot hash mismatch")
    development = freeze.get("development_gate_report")
    require(isinstance(development, dict) and isinstance(development.get("path"), str)
            and development.get("sha256") == digest(blobs["development-gate-report.json"]),
            "Development gate report hash mismatch")
    decision = gate.get("gate", {})
    require(decision.get("phase") == "development_selection_only" and decision.get("passed") is True
            and decision.get("failed_checks") == [] and isinstance(decision.get("checks"), list)
            and bool(decision["checks"]) and all(check.get("passed") is True for check in decision["checks"]),
            "Frozen development selection gate did not pass")
    require(gate.get("goal_complete") is False and gate.get("heldout_confirmed") is False,
            "Development gate cannot already claim held-out confirmation")
    provenance = gate.get("provenance", {})
    require(provenance.get("pack_manifest_sha256") == digest(blobs["development-pack-manifest.json"]),
            "Development pack manifest hash mismatch")
    require(provenance.get("goal_config_sha256") == freeze["goal_config_sha256"]
            and gate.get("goal_configuration") == goal, "Development gate used a different goal configuration")
    require(manifest.get("schema_version") == 3 and manifest.get("sample_count") == 40
            and manifest.get("review_item_count") == 80 and manifest.get("family_count") == 12,
            "Selection must originate from the 40-case adapted development pack")
    development_ids = manifest.get("selected_ids")
    require(isinstance(development_ids, list) and len(development_ids) == 40
            and all(isinstance(x, str) for x in development_ids) and len(set(development_ids)) == 40
            and set(development_ids).isdisjoint(selected), "Frozen test IDs must be disjoint from development selection")
    require(manifest.get("files", {}).get("policy.json") == policy_sha256
            and manifest.get("files", {}).get("rubric.md") == rubric_sha256, "Development policy/rubric file binding mismatch")
    comparison = manifest.get("comparison", {})
    reference, candidate = comparison.get("reference", {}), comparison.get("candidate", {})
    require(reference.get("mode") == "baseline" and candidate.get("mode") == "adapted"
            and gate.get("comparison") == {"reference": reference.get("label"), "candidate": candidate.get("label")},
            "Development gate comparison does not select an adapter against baseline")
    context = manifest.get("review_context", {})
    require(context.get("version") == 3 and context.get("comparison") == comparison
            and digest(canonical(context)) == manifest.get("review_context_sha256") == provenance.get("review_context_sha256"),
            "Development review context mismatch")
    require(provenance.get("pack_files") == manifest.get("files"), "Development gate pack file hashes mismatch")
    for field in ("dataset_sha256", "policy_sha256", "rubric_sha256"):
        require(manifest.get(field) == freeze[field], f"Development selection {field} mismatch")
    require(context.get("policy_sha256") == policy_sha256 and context.get("rubric_sha256") == rubric_sha256
            and context.get("curated_sha256") == manifest.get("curated_sha256")
            and context.get("ordered_selected_ids") == manifest.get("selected_ids"), "Development context data mismatch")
    identity = freeze.get("selected_adapter_provenance")
    require(isinstance(identity, dict) and identity.get("model") == runtime["model"]
            and identity.get("model_revision") == runtime["model_revision"]
            and identity.get("policy_sha256") == policy_sha256, "Frozen selected adapter identity mismatch")
    require(adapter_identity is None or adapter_identity == identity, "Inference adapter differs from frozen selection")
    for role in (reference, candidate):
        run = manifest.get("runs", {}).get(role.get("label"), {})
        require(run.get("config", {}).get("split") == "development", "Selection source was not development")
        require({"seed": run.get("runtime", {}).get("seed"), **{key: run.get("config", {}).get(key)
                    for key in settings if key != "seed"}} == settings, "Development and confirmation settings differ")
        require(all(run.get("runtime", {}).get(key) == runtime.get(key) for key in (
            "model", "model_revision", "dtype", "local_device", "batch_size")), "Development and confirmation runtimes differ")
    run = manifest["runs"][candidate["label"]]
    bound = context.get("runs", {}).get(candidate["label"], {})
    for key, filename, expected in (
        ("adapter_provenance_sha256", "adapter-provenance.json", digest(canonical(identity))),
        ("training_report_sha256", "adapter-training-report.json", identity.get("training_report_sha256")),
        ("training_config_sha256", "adapter-training-config.json", identity.get("training_config_sha256")),
        ("adapter_config_sha256", "adapter-config.json", identity.get("adapter_config_sha256"))):
        require(isinstance(expected, str) and run.get("files", {}).get(filename) == bound.get(key) == expected,
                "Selected adapter is not bound to the development source pack")
    return freeze


def load_confirmation_freeze(path, runtime, config, **expected):
    """Read mutable source paths once; future evidence validation uses snapshots."""
    path = Path(path)
    freeze_bytes = path.read_bytes()
    freeze = json.loads(freeze_bytes)
    gate_path = Path(freeze["development_gate_report"]["path"])
    if not gate_path.is_absolute():
        gate_path = path.parent / gate_path
    gate_bytes = gate_path.read_bytes()
    gate = json.loads(gate_bytes)
    pack_path = Path(gate["provenance"]["pack"])
    if not pack_path.is_absolute():
        pack_path = gate_path.parent / pack_path
    snapshots = {
        "confirmation-freeze.json": freeze_bytes,
        "development-gate-report.json": gate_bytes,
        "development-pack-manifest.json": (pack_path / "manifest.json").read_bytes(),
        "confirmation-goal-config.json": (ROOT / "configs/model-improvement-goal.json").read_bytes(),
        "confirmation-protocol.md": (ROOT / "docs/model-improvement-protocol.md").read_bytes(),
    }
    validated = validate_confirmation_snapshot(snapshots, runtime, config, **expected)
    return {"freeze": validated, "snapshots": snapshots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curate-only", action="store_true", help="Validate and curate without loading Aya")
    treatments = parser.add_mutually_exclusive_group()
    treatments.add_argument("--intervention", type=Path, help="Bilingual system-prompt appendix JSON; weights stay unchanged")
    treatments.add_argument("--adapter-training-run", type=Path, help="Completed hash-verified LoRA training run; original policy only")
    parser.add_argument("--rubric", type=Path, help="Rubric to snapshot before generation; required with either intervention")
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--confirmation-freeze", type=Path, help="Frozen development selection; required for test")
    args = parser.parse_args()
    if (args.intervention or args.adapter_training_run) and not args.rubric:
        parser.error("--intervention and --adapter-training-run require --rubric")
    if args.split == "test" and (not args.confirmation_freeze or not args.rubric or args.intervention):
        parser.error("--split test requires --confirmation-freeze and --rubric; prompt appendices are forbidden")
    if args.split != "test" and args.confirmation_freeze:
        parser.error("--confirmation-freeze is only valid with --split test")
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = json.loads((ROOT / "configs/runtime.json").read_text())
    config = json.loads((ROOT / "configs/pilot.json").read_text())
    config["split"] = args.split
    pins = json.loads((ROOT / "configs/tools.lock.json").read_text())
    report = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                  runtime=runtime, config=config, variant="adapted" if args.adapter_training_run else "grounded" if args.intervention else "baseline",
                  callback_calls=0, generated_samples=0,
                  test_generated_samples=0, semantic_review="pending",
                  scope="Draft fictional support-policy development pilot; no general safety claim.")

    if args.split == "test":
        report.update(development_generated_samples=0, phase="heldout_confirmation",
                      scope="Frozen 20-case held-out confirmation of a development-selected adapter.")

    def stage(name):
        report["stage"] = name
        write_json(args.output / "report.json", report)
        print(name, flush=True)

    stage("environment")
    try:
        import curatorkit
        from pilot_curation import curate_rows

        assert config["split"] in ("development", "test")
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
        confirmation = None
        if args.split == "test":
            confirmation = load_confirmation_freeze(args.confirmation_freeze, runtime, config,
                dataset_sha256=sha256(dataset_path), policy_sha256=sha256(policy_path), rubric_sha256=sha256(args.rubric))
            for name, content in confirmation["snapshots"].items():
                (args.output / name).write_bytes(content)
            report["confirmation_freeze_sha256"] = sha256(args.output / "confirmation-freeze.json")
            report["confirmation_id"] = confirmation["freeze"]["confirmation_id"]
            report["confirmation_snapshot_hashes"] = {name: sha256(args.output / name) for name in confirmation["snapshots"]}
            (args.output / "generation-runner.py").write_bytes(RUNNER_SOURCE_BYTES)
            report["generation_runner_sha256"] = sha256(args.output / "generation-runner.py")
            report["full_input_token_verification"] = "exact_full_chat_template_token_ids"
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
            if confirmation is not None and adapter_run["identity"] != confirmation["freeze"]["selected_adapter_provenance"]:
                raise ValueError("Inference adapter differs from frozen selection")
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
        if confirmation is not None:
            development_manifest = json.loads(confirmation["snapshots"]["development-pack-manifest.json"])
            assert report["curated_sha256"] == development_manifest["curated_sha256"], "Curated dataset changed since selection"
        selected = [row for row in curated if row["split"] == config["split"]]
        assert len(selected) == (20 if args.split == "test" else 40)
        if confirmation is not None:
            frozen_ids = confirmation["freeze"]["selected_test_ids"]
            assert set(frozen_ids) == {row["id"] for row in selected}, "Frozen test IDs differ from dataset"
            by_id = {row["id"]: row for row in selected}
            selected = [by_id[sample_id] for sample_id in frozen_ids]
            assert len({row["family_id"] for row in selected}) == 6, "Expected six held-out families"
        assert len({row["request"] for row in selected}) == len(selected), "Ambiguous request-to-ID mapping"
        report["selected_ids"] = [row["id"] for row in selected]
        report["selected_dataset"] = validate_rows(selected)
        if intervention is None and adapter_run is None and confirmation is None:
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
                    assert row["split"] == args.split and row["id"] not in completed_ids
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
                    if confirmation is not None:
                        record.update(confirmation_freeze_sha256=report["confirmation_freeze_sha256"],
                                      run_nonce=report["run_nonce"],
                                      full_input_token_ids_sha256=hashlib.sha256(json.dumps(expected_ids[row["id"]]).encode()).hexdigest())
                    checkpoints.write(json.dumps(record, ensure_ascii=False) + "\n")
                    checkpoints.flush()
                    os.fsync(checkpoints.fileno())
                    responses.append(record)
                    outputs.append(output)
                    completed_ids.add(row["id"])
                    report["generated_samples"] += 1
                    if args.split == "test":
                        report["test_generated_samples"] += 1
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
                        **({"confirmation_freeze_sha256": report["confirmation_freeze_sha256"], "split": "test",
                            "generation_runner_sha256": report["generation_runner_sha256"]} if confirmation is not None else {}),
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
