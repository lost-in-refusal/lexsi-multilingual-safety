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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curate-only", action="store_true", help="Validate and curate without loading Aya")
    parser.add_argument("--intervention", type=Path, help="Bilingual system-prompt appendix JSON; weights stay unchanged")
    parser.add_argument("--rubric", type=Path, help="Rubric to snapshot before generation; required with --intervention")
    args = parser.parse_args()
    if args.intervention and not args.rubric:
        parser.error("--intervention requires --rubric")
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = json.loads((ROOT / "configs/runtime.json").read_text())
    config = json.loads((ROOT / "configs/pilot.json").read_text())
    pins = json.loads((ROOT / "configs/tools.lock.json").read_text())
    report = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                  runtime=runtime, config=config, variant="grounded" if args.intervention else "baseline",
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
        if intervention is None:
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
                        "rubric_sha256": report.get("rubric_sha256")}))
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
