#!/usr/bin/env python3
"""Run real SafeTune generation through AuditKIT on six benign fixture prompts."""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cuda"), default="mps")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = json.loads((ROOT / "configs/runtime.json").read_text())
    pins = json.loads((ROOT / "configs/tools.lock.json").read_text())
    report = dict(status="running", stage="environment", started_utc=datetime.now(timezone.utc).isoformat(),
        scope="Benign generation/scoring integration only; not multilingual safety or task-quality evidence.",
        runtime=runtime, device=args.device, callback_calls=0, generated_samples=0)

    def stage(name):
        report["stage"] = name
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, flush=True)

    stage("environment")
    try:
        import torch
        import safetune
        import auditkit as ak
        from auditkit.model import CallableModel
        from huggingface_hub import snapshot_download
        from safetune.runner.utils.model_utils import load_model, load_tok
        from safetune.core.eval.pipeline import GenerationConfig, TransformersBackend
        for name, package in (("SafeTune", safetune), ("AuditKIT", ak)):
            source = ROOT / "vendor" / name
            revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
            assert revision == pins[name]["revision"] and not dirty, f"{name} source pin/cleanliness mismatch"
            assert Path(package.__file__).resolve().is_relative_to(source.resolve())
        report["tools"] = {n: pins[n]["revision"] for n in ("SafeTune", "AuditKIT")}
        report["packages"] = {n: importlib.metadata.version(n) for n in ("safetune", "auditkit", "torch", "transformers")}
        if args.device == "mps":
            assert torch.backends.mps.is_available()
            torch.mps.set_per_process_memory_fraction(0.8)
        else:
            assert torch.cuda.is_available()
        fixture = ROOT / "data/fixtures/benign_smoke.jsonl"
        rows = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]
        assert len(rows) == 6 and len({r["id"] for r in rows}) == 6
        samples = [ak.Sample(id=r["id"], input=r["input"], target=r["target"],
            metadata={k: r[k] for k in ("language", "scenario_id", "split")}) for r in rows]
        report["fixture_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
        (args.output / "samples.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")

        stage("load_model")
        snapshot = snapshot_download(runtime["model"], revision=runtime["model_revision"], local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
        safetune.configure(device=args.device, seed=runtime["seed"])
        tok = load_tok(snapshot)
        started = time.perf_counter()
        model = load_model(snapshot, dtype=runtime["dtype"], device=args.device)

        def sync():
            if args.device == "mps":
                torch.mps.synchronize()
            else:
                torch.cuda.synchronize()

        sync()
        report["model_load_seconds"] = round(time.perf_counter() - started, 3)
        report["model_device"] = str(next(model.parameters()).device)
        report["dtype"] = str(next(model.parameters()).dtype)
        assert report["dtype"] == "torch." + runtime["dtype"], "Loaded dtype differs from runtime config"
        backend = TransformersBackend(model, tok, config=GenerationConfig(
            max_new_tokens=runtime["smoke_max_new_tokens"], batch_size=runtime["batch_size"],
            temperature=0, seed=runtime["seed"]))

        def generate(prompts):
            report["callback_calls"] += 1
            outputs = backend.generate(prompts)
            report["generated_samples"] += len(outputs)
            return outputs

        stage("auditkit_evaluation")
        started = time.perf_counter()
        result = ak.evaluate(samples,
            model=CallableModel(generate, name=f"safetune:{runtime['model']}@{runtime['model_revision']}:baseline"),
            scorers=["exact_match", "f1_score"],
            config=ak.RunConfig(seed=runtime["seed"], temperature=0.0,
                max_tokens=runtime["smoke_max_new_tokens"], batch_size=1, concurrency=1,
                max_retries=0, track_performance=False,
                extra={"run_nonce": uuid.uuid4().hex, "fixture_sha256": report["fixture_sha256"],
                       "model_revision": runtime["model_revision"], "tools": report["tools"]}))
        sync()
        report["evaluation_seconds"] = round(time.perf_counter() - started, 3)
        result.metadata.update({"scope": report["scope"], "inputs": [
            {"kind": "model", "ref": runtime["model"], "revision": runtime["model_revision"]},
            {"kind": "dataset", "ref": "data/fixtures/benign_smoke.jsonl", "sha256": report["fixture_sha256"]}]})
        result.save(str(args.output / "auditkit-result.json"))
        assert not result.errors and result.failed_count == 0, "AuditKIT recorded execution failures"
        assert report["callback_calls"] > 0 and report["generated_samples"] == len(rows), "Live generation was not performed"
        assert len(result.predictions) == len(rows)
        assert {p.sample_id for p in result.predictions} == {r["id"] for r in rows}
        assert all(p.raw_output and p.raw_output.strip() for p in result.predictions)
        assert set(result.headline) == {"exact_match", "f1_score"}
        assert all(math.isfinite(v) for v in result.headline.values())
        report["headline"] = result.headline
        report["failed_count"] = result.failed_count
        report["auditkit_fingerprint"] = result.fingerprint
        report["by_language"] = {}
        for language in ("en", "ko", "zh"):
            ids = {r["id"] for r in rows if r["language"] == language}
            predictions = [p for p in result.predictions if p.sample_id in ids]
            report["by_language"][language] = {metric: sum(
                next(s["value"] for s in p.metadata["scores"] if s["name"] == metric)
                for p in predictions) / len(predictions) for metric in result.headline}
        if args.device == "mps":
            report["mps_driver_gib_after_evaluation"] = round(torch.mps.driver_allocated_memory() / 2**30, 3)
        report["status"] = "passed"
        stage("complete")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        (args.output / "error.log").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{report['status'].upper()}: {args.output / 'report.json'}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
