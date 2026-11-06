#!/usr/bin/env python3
"""Run one isolated setup check; save evidence, never publish or train."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

SAFETUNE_COMMIT = "26f59e6d1bdcacafe5f2be3b8d982fb214369659"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--official-quickstart", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "mode": "official_quickstart" if args.official_quickstart else "benign_generation",
        "status": "running",
        "stage": "environment",
        "python": sys.version,
        "scope": "Setup check only; not evidence of safety improvement.",
    }
    # A terminated process must never leave an earlier successful report behind.
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        if sys.version_info < (3, 12):
            raise RuntimeError("SafeTune requires Python 3.12 or newer.")
        report["safetune_commit"] = subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip()
        if report["safetune_commit"] != SAFETUNE_COMMIT:
            raise RuntimeError("SafeTune source does not match the reviewed revision.")
        dirty = subprocess.check_output(
            ["git", "-C", str(args.source), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
        report["safetune_tracked_files_clean"] = not bool(dirty.strip())
        if dirty.strip():
            raise RuntimeError("SafeTune has edited tracked files; use a clean checkout for this setup check.")
        report["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("safetune", "torch", "transformers", "accelerate", "peft", "trl", "huggingface-hub")
        }
        import torch
        import safetune
        from safetune.config import resolve_device
        from huggingface_hub import HfApi, snapshot_download

        if not Path(safetune.__file__).resolve().is_relative_to(args.source.resolve()):
            raise RuntimeError("Imported SafeTune is not the pinned source checkout. Re-run installation.")
        device = resolve_device(args.device)
        report["device"] = device
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch cannot access this Mac's Metal GPU.")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access a CUDA GPU.")
        if device == "cuda":
            report["gpu"] = torch.cuda.get_device_name(0)
            report["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        if args.model.lower().startswith("coherelabs/") and device == "cpu":
            raise RuntimeError("Use --device mps on Apple Silicon or --device cuda for this Aya check.")
        if device == "mps":
            report["mps_recommended_max_gib"] = round(torch.mps.recommended_max_memory() / 2**30, 2)

        report["stage"] = "model_download"
        info = HfApi().model_info(args.model)
        report["model_revision"] = info.sha
        if not info.sha:
            raise RuntimeError("Could not resolve an immutable model revision.")
        snapshot = snapshot_download(
            args.model,
            revision=info.sha,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        )
        if args.official_quickstart:
            report["stage"] = "official_quickstart"
            script = args.source / "examples/quickstart/quickstart.py"
            with (args.output / "quickstart.log").open("w") as log:
                result = subprocess.run(
                    [sys.executable, "-u", str(script), "--model", snapshot,
                     "--device", device, "--max-new-tokens", "32"],
                    stdout=log, stderr=subprocess.STDOUT,
                )
            report["quickstart_exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError("Official quickstart failed; inspect quickstart.log.")
        else:
            report["stage"] = "model_load"
            from safetune.runner.utils import load_model, load_tok
            from safetune._refusal_helpers import _get_decoder_layers
            from safetune.core.eval.pipeline import GenerationConfig, TransformersBackend

            safetune.configure(device=device, seed=42)
            tokenizer = load_tok(snapshot)
            load_started = time.perf_counter()
            model = load_model(snapshot)
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            report["model_load_seconds"] = round(time.perf_counter() - load_started, 3)
            model.eval()
            layers = _get_decoder_layers(model)
            if not len(layers):
                raise RuntimeError("SafeTune did not discover any decoder layers.")
            report["decoder_layers"] = len(layers)
            report["model_type"] = model.config.model_type
            report["dtype"] = str(next(model.parameters()).dtype)
            report["model_device"] = str(next(model.parameters()).device)
            prompt = "Explain in one sentence why keeping experiment logs is useful."
            token_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, return_dict=False,
            )
            report["templated_prompt_tokens"] = len(token_ids)
            report["stage"] = "benign_generation"
            backend = TransformersBackend(
                model, tokenizer,
                config=GenerationConfig(max_new_tokens=32, batch_size=1, temperature=0, seed=42),
            )
            generation_started = time.perf_counter()
            answer = backend.generate([prompt])[0]
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            report["generation_seconds"] = round(time.perf_counter() - generation_started, 3)
            if not answer.strip():
                raise RuntimeError("Model generated an empty answer.")
            report["benign_answer"] = answer
            if device == "cuda":
                report["peak_allocated_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            elif device == "mps":
                report["mps_allocated_gib_after_generation"] = round(torch.mps.current_allocated_memory() / 2**30, 2)
                report["mps_driver_allocated_gib_after_generation"] = round(torch.mps.driver_allocated_memory() / 2**30, 2)
        report["status"] = "passed"
        report["stage"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        (args.output / "error.log").write_text(traceback.format_exc())
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        target = args.output / "report.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(f"{report['status'].upper()}: {args.model}; stage={report['stage']}")
        print(f"Evidence: {target}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
