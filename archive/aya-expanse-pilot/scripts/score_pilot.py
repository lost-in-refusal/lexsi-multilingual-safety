#!/usr/bin/env python3
"""Summarize explicit annotations against a verified 40-sample development run.

This performs no generation or automatic judging. Null annotation templates and
incomplete runs are rejected; partial actual reviews keep explicit denominators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .pilot_validation import summarize_annotations, validate_rows
except ImportError:  # Direct execution from scripts/.
    from pilot_validation import summarize_annotations, validate_rows


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _require_text(mapping: dict, key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {key} must be a nonempty string")
    return value


def score_run(run: Path | str, annotations: Path | str) -> dict:
    """Verify run evidence and return provenance plus semantic summary.

    Hashes identify the exact bytes read during this call. Response identity is
    checked against curated rows and the run's policy hash. Output text is never
    judged or changed here. The run report does not contain a response-file hash;
    the new summary records one to bind this review to those responses thereafter.
    """
    run = Path(run)
    annotations = Path(annotations)
    report_bytes = (run / "report.json").read_bytes()
    report = json.loads(report_bytes)
    if not isinstance(report, dict) or report.get("status") != "passed_execution":
        raise ValueError("Run report must have status passed_execution")
    if report.get("generated_samples") != 40 or report.get("test_generated_samples") != 0:
        raise ValueError("Run must contain 40 generated development samples and no generated test samples")
    runtime, config = report.get("runtime"), report.get("config")
    if not isinstance(runtime, dict) or not isinstance(config, dict):
        raise ValueError("Run report must include runtime and config objects")
    if config.get("split") != "development":
        raise ValueError("Only the development split can be scored by this CLI")
    for key in ("model", "model_revision"):
        _require_text(runtime, key, "runtime")
    for key in ("run_nonce", "scope", "dataset_sha256", "policy_sha256", "curated_sha256"):
        _require_text(report, key, "run report")
    _require_text(config, "experiment", "config")

    curated_bytes = (run / "curated-samples.json").read_bytes()
    if _sha256(curated_bytes) != report["curated_sha256"]:
        raise ValueError("curated-samples.json SHA-256 does not match the run report")
    rows = json.loads(curated_bytes)
    validate_rows(rows)
    rows_by_id = {row["id"]: row for row in rows}
    development_ids = {row["id"] for row in rows if row["split"] == "development"}
    selected_ids = report.get("selected_ids")
    if (not isinstance(selected_ids, list) or len(selected_ids) != 40
            or any(not isinstance(sample_id, str) for sample_id in selected_ids)
            or len(set(selected_ids)) != 40):
        raise ValueError("selected_ids must contain exactly 40 unique sample IDs")
    if set(selected_ids) != development_ids:
        raise ValueError("selected_ids do not exactly match curated development sample IDs")
    selected_rows = [rows_by_id[sample_id] for sample_id in selected_ids]

    response_bytes = (run / "responses.jsonl").read_bytes()
    responses = [json.loads(line) for line in response_bytes.splitlines() if line.strip()]
    response_ids: set[str] = set()
    for index, response in enumerate(responses):
        if not isinstance(response, dict):
            raise ValueError(f"Response {index} must be an object")
        sample_id = response.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in development_ids:
            raise ValueError(f"Response {index} has an unknown or unexecuted sample_id")
        if sample_id in response_ids:
            raise ValueError(f"Duplicate response sample_id: {sample_id!r}")
        response_ids.add(sample_id)
        row = rows_by_id[sample_id]
        for field in ("family_id", "split", "language", "category", "request"):
            if response.get(field) != row[field]:
                raise ValueError(f"Response {sample_id!r}: {field} differs from curated sample")
        if response.get("policy_sha256") != report["policy_sha256"]:
            raise ValueError(f"Response {sample_id!r}: policy hash differs from run report")
        if not isinstance(response.get("output"), str):
            raise ValueError(f"Response {sample_id!r}: output must be a string")
    if response_ids != development_ids:
        raise ValueError("responses.jsonl does not exactly cover the 40 selected development samples")

    annotation_bytes = annotations.read_bytes()
    labels = json.loads(annotation_bytes)
    if not isinstance(labels, list) or not labels:
        raise ValueError("Annotations must be a nonempty JSON list of explicit reviews, not an empty/template file")
    # This also rejects null-template fields, duplicate/unknown IDs, invalid
    # category outcomes and invalid scores. Only selected rows can be annotated.
    summary = summarize_annotations(selected_rows, labels)
    provenance = {
        "run": str(run.resolve()),
        "annotations": str(annotations.resolve()),
        "annotation_sha256": _sha256(annotation_bytes),
        "responsefile_sha256": _sha256(response_bytes),
        "run_report_sha256": _sha256(report_bytes),
        "curated_sha256": report["curated_sha256"],
        "dataset_sha256": report["dataset_sha256"],
        "policy_sha256": report["policy_sha256"],
        "run_nonce": report["run_nonce"],
        "model": runtime["model"],
        "model_revision": runtime["model_revision"],
        "experiment": config["experiment"],
        "split": "development",
        "selected_ids": selected_ids,
        "scope": report["scope"],
        "tools": report.get("tools", {}),
        "run_source_hashes": report.get("source_hashes", {}),
        "scoring_source_hashes": {
            "scripts/score_pilot.py": _sha256(Path(__file__).read_bytes()),
            "scripts/pilot_validation.py": _sha256(Path(__file__).with_name("pilot_validation.py").read_bytes()),
        },
    }
    return {"provenance": provenance, **summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed pilot run directory")
    parser.add_argument("--annotations", type=Path, required=True, help="JSON list of explicit reviews")
    parser.add_argument("--output", type=Path, required=True, help="New summary file; existing paths are refused")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        summary = score_run(args.run, args.annotations)
        text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        # Exclusive creation also guards against another process creating the
        # output after the earlier existence check.
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Saved semantic summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
