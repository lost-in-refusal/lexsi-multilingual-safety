#!/usr/bin/env python3
"""Validate correction partitions and audit cached Aya tokens without training.

Only mechanical overlap is checked here. Policy fidelity, translation quality,
and semantic independence require separate review; no evaluation is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("en", "ko")
SPLITS = ("train", "calibration")
CATEGORIES = ("benign", "disallowed")
V1 = "harborcart_targeted_plain_lora_v1"
V2 = "harborcart_complete_answer_lora_v2"
EXPERIMENTS = {
    V1: {"data": "data/corrections/examples.jsonl", "config": "configs/correction-lora.json",
         "expected_rows": {"train": 32, "calibration": 16},
         "expected_families": {"train": 8, "calibration": 4},
         "epochs": 1, "train_steps": 32, "learning_rate": 0.0001},
    V2: {"data": "data/corrections-v2/examples.jsonl", "config": "configs/correction-lora-v2.json",
         "expected_rows": {"train": 80, "calibration": 16},
         "expected_families": {"train": 24, "calibration": 4},
         "epochs": 2, "train_steps": 160, "learning_rate": 0.00005},
}
ROW_FIELDS = {
    "id", "family_id", "split", "language", "category", "policy_rule_ids",
    "request", "target_response", "scenario_summary_en", "provenance", "human_review_status",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def experiment_spec(experiment_id):
    """Choose only declared experiments; an identifier is never a file path."""
    _require(isinstance(experiment_id, str) and experiment_id in EXPERIMENTS,
             "Unknown correction experiment_id")
    return EXPERIMENTS[experiment_id]


def canonical_input_paths(experiment_id):
    spec = experiment_spec(experiment_id)
    return {"data": spec["data"], "config": spec["config"], "policy": "data/pilot/policy.json",
            "evaluation": "data/pilot/scenarios.jsonl", "runtime": "configs/runtime.json"}


def _text(value, name):
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")
    return value


def _normalized(value):
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def _policy_texts(policy):
    _require(isinstance(policy, dict), "Policy must be an object")
    texts = policy.get("texts")
    _require(isinstance(texts, dict) and set(texts) == set(LANGUAGES), "Policy requires exactly en/ko texts")
    for lang, value in texts.items():
        _text(value, f"policy.texts.{lang}")
    return texts


def validate_correction_rows(rows, policy, evaluation_rows, *, experiment_id=V1):
    """Reject schema/pair leakage and exact normalized evaluation overlap.

    Evaluation rows are inspected only for IDs, families, and request strings.
    No labels, outputs, or semantic outcomes from either split are used.
    """
    experiment_spec(experiment_id)
    categories = CATEGORIES + (("utility",) if experiment_id == V2 else ())
    _policy_texts(policy)
    rules = policy.get("rules")
    _require(isinstance(rules, list) and bool(rules), "Policy rules must be a nonempty list")
    rule_ids = []
    for rule in rules:
        _require(isinstance(rule, dict), "Each policy rule must be an object")
        rule_ids.append(_text(rule.get("id"), "policy rule ID"))
    _require(len(set(rule_ids)) == len(rule_ids), "Duplicate policy rule IDs")
    known_rules = set(rule_ids)
    _require(isinstance(rows, list) and bool(rows), "Correction rows must be a nonempty list")
    _require(isinstance(evaluation_rows, list), "Evaluation rows must be a list")
    evaluation_ids, evaluation_families, evaluation_requests = set(), set(), set()
    for row in evaluation_rows:
        _require(isinstance(row, dict), "Evaluation row must be an object")
        evaluation_ids.add(_text(row.get("id"), "evaluation ID"))
        evaluation_families.add(_text(row.get("family_id"), "evaluation family ID"))
        evaluation_requests.add(_normalized(_text(row.get("request"), "evaluation request")))

    ids, requests, families = set(), set(), {}
    for index, row in enumerate(rows):
        _require(isinstance(row, dict) and set(row) == ROW_FIELDS, f"Row {index}: expected exactly the correction schema fields")
        for field in ROW_FIELDS - {"policy_rule_ids"}:
            _text(row[field], f"row {index}.{field}")
        sample_id = row["id"]
        _require(sample_id not in ids, f"Duplicate correction ID: {sample_id}")
        _require(sample_id not in evaluation_ids, f"Evaluation ID overlap: {sample_id}")
        _require(row["family_id"] not in evaluation_families, f"Evaluation family overlap: {row['family_id']}")
        for field, allowed in (("split", SPLITS), ("language", LANGUAGES), ("category", categories)):
            _require(row[field] in allowed, f"{sample_id}: invalid {field}")
        _require(row["provenance"] == "original_assistant_draft", f"{sample_id}: expected assistant-draft provenance")
        _require(row["human_review_status"] == "pending", f"{sample_id}: this draft schema cannot assert human approval")
        normalized = _normalized(row["request"])
        _require(bool(normalized), f"{sample_id}: request must contain letters or numbers")
        _require(normalized not in requests, f"Duplicate normalized correction request: {sample_id}")
        _require(normalized not in evaluation_requests, f"Normalized evaluation request overlap: {sample_id}")
        references = row["policy_rule_ids"]
        _require(isinstance(references, list) and bool(references)
                 and all(isinstance(r, str) and r in known_rules for r in references), f"{sample_id}: invalid policy rule IDs")
        _require(len(set(references)) == len(references), f"{sample_id}: duplicate policy rule IDs")
        family = families.setdefault(row["family_id"], {
            "split": row["split"], "scenario_summary_en": row["scenario_summary_en"], "variants": {},
        })
        _require(family["split"] == row["split"], f"Family {row['family_id']} crosses splits")
        _require(family["scenario_summary_en"] == row["scenario_summary_en"], f"Family {row['family_id']} has inconsistent scenario summary")
        variant = (row["category"], row["language"])
        _require(variant not in family["variants"], f"Family {row['family_id']} has duplicate variant")
        family["variants"][variant] = set(references)
        ids.add(sample_id)
        requests.add(normalized)
    expected_variants = {(category, language) for category in CATEGORIES for language in LANGUAGES}
    for family_id, family in families.items():
        utility = any(category == "utility" for category, _ in family["variants"])
        if utility:
            _require(experiment_id == V2 and family["split"] == "train",
                     f"Family {family_id}: utility pairs are allowed only in v2 training")
        required = {("utility", language) for language in LANGUAGES} if utility else expected_variants
        _require(set(family["variants"]) == required, f"Family {family_id} lacks complete bilingual category pairs")
        for category in (("utility",) if utility else CATEGORIES):
            _require(family["variants"][(category, "en")] == family["variants"][(category, "ko")],
                     f"Family {family_id}: translated {category} policy references differ")
    return {
        "total_rows": len(rows), "total_families": len(families),
        "counts_by_split": {split: sum(r["split"] == split for r in rows) for split in SPLITS},
        "family_counts_by_split": {split: sum(f["split"] == split for f in families.values()) for split in SPLITS},
        "counts_by_language": {lang: sum(r["language"] == lang for r in rows) for lang in LANGUAGES},
        "counts_by_category": {category: sum(r["category"] == category for r in rows) for category in categories},
        "families_by_split": {split: sorted(k for k, f in families.items() if f["split"] == split) for split in SPLITS},
        "human_review_pending": len(rows),
        "evaluation_rows_checked_mechanically": len(evaluation_rows),
        "evaluation_overlap_check": "passed_exact_normalized_only",
        "overlap_normalization": "Unicode NFKC, casefold, retain alphanumeric characters only",
        "limitation": "Exact mechanical disjointness does not prove semantic independence; policy concepts overlap.",
    }


def prepare_example(row, policy, tokenizer, max_sequence_tokens):
    """Tokenize an intact conversation and mask exactly its complete prefix."""
    texts = _policy_texts(policy)
    _require(isinstance(row, dict), "Example must be an object")
    sample_id = _text(row.get("id"), "example ID")
    _require(row.get("language") in LANGUAGES, f"{sample_id}: invalid language")
    request = _text(row.get("request"), f"{sample_id}.request")
    answer = _text(row.get("target_response"), f"{sample_id}.target_response")
    _require(type(max_sequence_tokens) is int and max_sequence_tokens > 1, "Sequence limit must be an integer greater than one")
    for token in tokenizer.all_special_tokens:
        if token:
            _require(token not in request and token not in answer, f"{sample_id}: literal tokenizer special token in authored text")
    messages = [{"role": "system", "content": texts[row["language"]]}, {"role": "user", "content": request}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}],
                                         tokenize=True, add_generation_prompt=False, return_dict=False)
    for name, value in (("prefix", prefix), ("full", full)):
        _require(isinstance(value, list) and bool(value) and all(type(t) is int and t >= 0 for t in value),
                 f"{sample_id}: {name} must contain integer token IDs")
    _require(full[:len(prefix)] == prefix, f"{sample_id}: chat prefix boundary mismatch")
    _require(len(full) <= max_sequence_tokens,
             f"{sample_id}: sequence length {len(full)} exceeds limit {max_sequence_tokens}; no truncation permitted")
    suffix = full[len(prefix):]
    _require(len(suffix) >= 2, f"{sample_id}: no complete supervised response plus EOS")
    _require(type(tokenizer.eos_token_id) is int and suffix[-1] == tokenizer.eos_token_id,
             f"{sample_id}: supervised response must end in tokenizer EOS")
    decoded = tokenizer.decode(suffix, skip_special_tokens=True)
    _require(decoded == answer, f"{sample_id}: supervised target text mismatch or truncation")
    labels = [-100] * len(prefix) + suffix
    supervised = sum(token != -100 for token in labels[1:])
    _require(supervised == len(suffix), f"{sample_id}: supervised next-token count mismatch")
    return {"id": sample_id, "input_ids": full, "attention_mask": [1] * len(full), "labels": labels,
            "prompt_tokens": len(prefix), "sequence_tokens": len(full), "supervised_tokens": supervised}


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _validate_config(config, runtime, validation, policy_sha256):
    _require(isinstance(config, dict) and isinstance(runtime, dict), "Configuration and runtime must be objects")
    spec = experiment_spec(config.get("experiment_id"))
    for field in ("model", "model_revision", "dtype", "seed", "batch_size"):
        _require(config.get(field) == runtime.get(field), f"Config/runtime mismatch: {field}")
    _text(config.get("model"), "model")
    _require(isinstance(config.get("model_revision"), str) and bool(re.fullmatch(r"[0-9a-f]{40}", config["model_revision"])),
             "Model revision must be a full immutable commit")
    _require(config.get("device") == runtime.get("local_device") == "mps", "Expected recorded MPS configuration")
    _require(config.get("system_prompt") == "original_policy_only", "Only unchanged original-policy messages are supported")
    _require(isinstance(config.get("policy_sha256"), str) and bool(re.fullmatch(r"[0-9a-f]{64}", config["policy_sha256"])),
             "Configuration must pin the original policy with a SHA-256 hash")
    _require(config["policy_sha256"] == policy_sha256, "Policy bytes differ from the frozen original-policy hash")
    _require(config.get("loss") == "assistant_response_and_eos_only", "Expected assistant-response/EOS-only supervision")
    _require(config.get("human_review_status") == "pending" and config.get("status") == "proposed_not_trained",
             "Configuration must remain an untrained proposal pending human review")
    for field, actual in (("expected_rows", "counts_by_split"), ("expected_families", "family_counts_by_split")):
        _require(config.get(field) == spec[field] == validation[actual]
                 and all(type(value) is int for value in config[field].values()),
                 f"Expected fixed {field}={spec[field]}")
    for field, expected in (("train_steps", spec["train_steps"]), ("epochs", spec["epochs"]),
                            ("batch_size", 1), ("gradient_accumulation_steps", 1)):
        _require(type(config.get(field)) is int and config[field] == expected, f"Expected fixed {field}={expected}")
    _require(type(config.get("learning_rate")) is float and config["learning_rate"] == spec["learning_rate"],
             f"Expected fixed learning_rate={spec['learning_rate']}")
    _require(type(config.get("max_sequence_tokens")) is int and config["max_sequence_tokens"] == 1024,
             "This draft uses a fixed 1024-token total sequence limit")
    _require(config.get("max_generation_new_tokens") == 512, "Keep inference limit separately at 512 new tokens")


def load_cached_tokenizer(config):
    """Local-cache-only tokenizer load; this function never loads model weights."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    snapshot = Path(snapshot_download(config["model"], revision=config["model_revision"], local_files_only=True,
        allow_patterns=["config.json", "tokenizer*", "special_tokens_map.json", "added_tokens.json",
                        "vocab.*", "merges.txt", "*.model", "*.jinja"]))
    _require(snapshot.name == config["model_revision"], "Cached snapshot does not match the exact requested commit")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    selected = [p for p in snapshot.rglob("*") if p.is_file() and (
        p.name.startswith("tokenizer") or p.name in {"config.json", "special_tokens_map.json", "added_tokens.json", "merges.txt"}
        or p.name.startswith("vocab.") or p.suffix in {".model", ".jinja"})]
    _require(bool(selected), "No tokenizer source files in cached snapshot")
    template = tokenizer.get_chat_template()
    _text(template, "resolved chat template")
    provenance = {
        "model": config["model"], "requested_revision": config["model_revision"],
        "actual_snapshot_commit": snapshot.name, "snapshot_path": str(snapshot.resolve()),
        "class": type(tokenizer).__name__, "eos_token_id": tokenizer.eos_token_id,
        "chat_template_sha256": _sha(template.encode("utf-8")),
        "files": {str(p.relative_to(snapshot)): _sha(p.read_bytes()) for p in sorted(selected)},
        "packages": {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "huggingface-hub")},
        "local_files_only": True, "trust_remote_code": False,
    }
    return tokenizer, template, provenance


def _token_stats(records):
    return {"rows": len(records), **{
        field: {"minimum": min(r[field] for r in records), "maximum": max(r[field] for r in records),
                "total": sum(r[field] for r in records), "mean": sum(r[field] for r in records) / len(records)}
        for field in ("prompt_tokens", "sequence_tokens", "supervised_tokens")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Defaults to the canonical corpus for the config's known experiment_id")
    parser.add_argument("--policy", type=Path, default=ROOT / "data/pilot/policy.json")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "data/pilot/scenarios.jsonl")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/correction-lora.json")
    parser.add_argument("--runtime", type=Path, default=ROOT / "configs/runtime.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes)
        _require(isinstance(config, dict), "Configuration must be an object")
        defaults = canonical_input_paths(config.get("experiment_id"))
        if args.data is None:
            args.data = ROOT / defaults["data"]
        paths = {name: getattr(args, name) for name in ("data", "policy", "evaluation", "config", "runtime")}
        blobs = {name: config_bytes if name == "config" else path.read_bytes() for name, path in paths.items()}
        rows = [json.loads(line) for line in blobs["data"].splitlines() if line.strip()]
        evaluation_rows = [json.loads(line) for line in blobs["evaluation"].splitlines() if line.strip()]
        policy, config, runtime = (json.loads(blobs[name]) for name in ("policy", "config", "runtime"))
        validation = validate_correction_rows(rows, policy, evaluation_rows, experiment_id=config["experiment_id"])
        _validate_config(config, runtime, validation, _sha(blobs["policy"]))
        tokenizer, template, tokenizer_provenance = load_cached_tokenizer(config)
        prepared = [dict(prepare_example(row, policy, tokenizer, config["max_sequence_tokens"]),
                         family_id=row["family_id"], split=row["split"], language=row["language"], category=row["category"])
                    for row in rows]
        files = {f"{split}.jsonl": b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode()
                                          for row in prepared if row["split"] == split) for split in SPLITS}
        files.update({"examples.jsonl": blobs["data"], "policy.json": blobs["policy"],
                      "config.json": blobs["config"], "runtime.json": blobs["runtime"], "chat-template.txt": template.encode("utf-8")})
        manifest = {
            "schema_version": 1, "status": "passed_tokenization_preflight", "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "CPU-only tokenizer audit; no model weights loaded, memory compatibility test, generation, or training.",
            "inputs": {name: {"path": str(paths[name].resolve()), "sha256": _sha(blobs[name])} for name in paths},
            "script_sha256": _sha(Path(__file__).read_bytes()), "validation": validation,
            "tokenizer": tokenizer_provenance, "max_sequence_tokens": config["max_sequence_tokens"],
            "original_policy_only": True, "truncated_examples": 0,
            "supervision": "Complete assistant response and terminal EOS; full system/user/generation prefix masked with -100.",
            "optimizer_steps": 0, "model_weights_loaded": False, "test_generated_samples": 0, "test_scored_samples": 0,
            "human_review_pending": len(rows), "memory_preflight": "not_run", "training": "not_run",
            "statistics": {"overall": _token_stats(prepared),
                "by_split": {s: _token_stats([r for r in prepared if r["split"] == s]) for s in SPLITS},
                "by_language": {lang: _token_stats([r for r in prepared if r["language"] == lang]) for lang in LANGUAGES},
                "by_split_language": {s: {lang: _token_stats([r for r in prepared if r["split"] == s and r["language"] == lang])
                                           for lang in LANGUAGES} for s in SPLITS}},
            "examples": [{k: v for k, v in row.items() if k not in {"input_ids", "attention_mask", "labels"}} for row in prepared],
            "files": {name: _sha(content) for name, content in files.items()},
        }
        args.output.mkdir(parents=True, exist_ok=False)
        for name, content in files.items():
            with (args.output / name).open("xb") as handle:
                handle.write(content)
        with (args.output / "manifest.json").open("xb") as handle:
            handle.write(_json_bytes(manifest))
        print(f"Prepared {len(rows)} examples; maximum sequence {manifest['statistics']['overall']['sequence_tokens']['maximum']} tokens; zero optimizer steps.")
        print(f"Saved preflight: {args.output / 'manifest.json'}")
        return 0
    except (ValueError, OSError, TypeError) as exc:
        print(f"Correction preflight failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
