"""Contract tests for leakage checks and complete assistant-only supervision."""
from copy import deepcopy
import importlib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


POLICY = {"texts": {"en": "Exact English policy.\nKeep it.", "ko": "정확한 정책입니다."},
          "rules": [{"id": "secrets"}, {"id": "refunds"}]}
V2 = "harborcart_complete_answer_lora_v2"


def family(name="train_a", split="train"):
    return [{"id": f"{name}_{category}_{language}", "family_id": name,
             "split": split, "language": language, "category": category,
             "policy_rule_ids": ["secrets"],
             "request": f"{name} {category} {language}: Explain this distinct request.",
             "target_response": "Enter credentials only in the official portal.",
             "scenario_summary_en": f"The circumstance for {name}.",
             "provenance": "original_assistant_draft", "human_review_status": "pending"}
            for category in ("benign", "disallowed") for language in ("en", "ko")]


def utility_family(name="utility_a", split="train"):
    rows = family(name, split)[:2]
    for row in rows:
        row["category"] = "utility"
        row["id"] = f"{name}_utility_{row['language']}"
        row["request"] = f"{name} utility {row['language']}: Explain this distinct request."
    return rows


def v2_config(config):
    result = deepcopy(config)
    result.update(experiment_id=V2, expected_rows={"train": 80, "calibration": 16},
                  expected_families={"train": 24, "calibration": 4},
                  train_steps=160, epochs=2, learning_rate=0.00005)
    return result


class FakeTokenizer:
    eos_token_id = 2
    all_special_tokens = ["<EOS>", "<ROLE>"]
    chat_template = "fixture template"

    def __init__(self, defect=None):
        self.defect = defect
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        self.calls.append(deepcopy(messages))
        assert tokenize and return_dict is False
        prefix = [1] + [ord(c) + 10 for c in messages[0]["content"] + messages[1]["content"] + "ASSISTANT:"]
        if add_generation_prompt:
            return prefix
        suffix = [ord(c) + 10 for c in messages[2]["content"]]
        if self.defect == "prefix":
            prefix[1] += 1
        if self.defect == "truncated_target":
            suffix = suffix[:-1]
        if self.defect == "normalized_whitespace":
            suffix = [ord(c) + 10 for c in messages[2]["content"].strip()]
        return prefix + suffix + ([3] if self.defect == "eos" else [2])

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        return "".join(chr(i - 10) for i in ids if i >= 10)


class ContractTest(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("scripts.prepare_corrections")
        except ModuleNotFoundError as exc:
            self.fail(f"Correction preflight module is missing: {exc}")


class ValidateCorrectionRowsTests(ContractTest):
    def test_v2_allows_complete_utility_pairs_alongside_full_category_pairs(self):
        rows = family() + utility_family() + family("cal_a", "calibration")
        result = self.api.validate_correction_rows(rows, POLICY, [], experiment_id=V2)
        self.assertEqual(result["counts_by_split"], {"train": 6, "calibration": 4})
        self.assertEqual(result["family_counts_by_split"], {"train": 2, "calibration": 1})
        self.assertEqual(result["counts_by_category"], {"benign": 4, "disallowed": 4, "utility": 2})

    def test_utility_requires_v2_and_cannot_replace_calibration_pairs(self):
        with self.assertRaises(ValueError):
            self.api.validate_correction_rows(utility_family(), POLICY, [])
        with self.assertRaises(ValueError):
            self.api.validate_correction_rows(utility_family(split="calibration"), POLICY, [], experiment_id=V2)
        with self.assertRaises(ValueError):
            self.api.validate_correction_rows(family(), POLICY, [], experiment_id="arbitrary_experiment")

    def test_v2_rejects_missing_mixed_duplicate_or_mismatched_utility_pair(self):
        missing = utility_family()[:1]
        mixed = utility_family(); mixed[1]["category"] = "benign"
        duplicate = utility_family(); duplicate[1]["language"] = "en"
        rules = utility_family(); rules[1]["policy_rule_ids"] = ["refunds"]
        extra = utility_family() + family("utility_a")
        for rows in (missing, mixed, duplicate, rules, extra, family()[:2]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.api.validate_correction_rows(rows, POLICY, [], experiment_id=V2)

    def test_complete_bilingual_families_have_explicit_split_counts(self):
        rows = family() + family("cal_a", "calibration")
        result = self.api.validate_correction_rows(rows, POLICY, [])
        self.assertEqual(result["total_rows"], 8)
        self.assertEqual(result["counts_by_split"], {"train": 4, "calibration": 4})
        self.assertEqual(result["family_counts_by_split"], {"train": 1, "calibration": 1})
        self.assertEqual(result["counts_by_language"], {"en": 4, "ko": 4})
        self.assertEqual(result["human_review_pending"], 8)
        self.assertEqual(result["evaluation_overlap_check"], "passed_exact_normalized_only")

    def test_schema_missing_extra_and_false_human_approval_are_rejected(self):
        for mutation in (lambda r: r.pop("target_response"), lambda r: r.update(extra="oops"),
                         lambda r: r.update(human_review_status="approved"),
                         lambda r: r.update(provenance="human_authored")):
            rows = family(); mutation(rows[0])
            with self.subTest(row=rows[0]), self.assertRaises(ValueError):
                self.api.validate_correction_rows(rows, POLICY, [])

    def test_empty_or_nontext_fields_are_rejected(self):
        for key in ("request", "target_response", "scenario_summary_en", "family_id", "id"):
            rows = family(); rows[0][key] = "  "
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.validate_correction_rows(rows, POLICY, [])
        with self.assertRaises(ValueError):
            self.api.validate_correction_rows([], POLICY, [])

    def test_family_leakage_missing_pair_and_duplicate_variant_are_rejected(self):
        leaked = family(); leaked[0]["split"] = "calibration"
        duplicate = family(); duplicate[0]["category"] = "disallowed"
        for rows in (leaked, family()[:-1], duplicate):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.api.validate_correction_rows(rows, POLICY, [])

    def test_duplicate_ids_and_normalized_requests_are_rejected(self):
        rows = family(); rows[1]["id"] = rows[0]["id"]
        with self.assertRaises(ValueError):
            self.api.validate_correction_rows(rows, POLICY, [])
        rows = family(); rows[0]["request"] = "Ａccount CODE!"; rows[1]["request"] = "account code"
        with self.assertRaises(ValueError):
            self.api.validate_correction_rows(rows, POLICY, [])

    def test_evaluation_id_family_and_normalized_request_collisions_are_rejected(self):
        row = family()[0]
        for evaluation in (
            {"id": row["id"], "family_id": "other", "request": "different"},
            {"id": "other", "family_id": row["family_id"], "request": "different"},
            {"id": "other", "family_id": "other", "request": row["request"].upper().replace(" ", "-")},
        ):
            with self.subTest(evaluation=evaluation), self.assertRaises(ValueError):
                self.api.validate_correction_rows(family(), POLICY, [evaluation])

    def test_unknown_duplicate_and_translation_mismatched_rule_ids_are_rejected(self):
        for rules in (["unknown"], ["secrets", "secrets"], [], ["refunds"]):
            rows = family(); rows[0]["policy_rule_ids"] = rules
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                self.api.validate_correction_rows(rows, POLICY, [])

    def test_inconsistent_family_summary_is_rejected(self):
        rows = family(); rows[0]["scenario_summary_en"] = "A different circumstance."
        with self.assertRaises(ValueError):
            self.api.validate_correction_rows(rows, POLICY, [])

    def test_bad_language_category_split_and_policy_are_rejected(self):
        for key,value in (("language", "fr"), ("category", "utility"), ("split", "test")):
            rows = family(); rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.validate_correction_rows(rows, POLICY, [])
        bad = deepcopy(POLICY); del bad["texts"]["ko"]
        with self.assertRaises(ValueError):
            self.api.validate_correction_rows(family(), bad, [])


class PrepareExampleTests(ContractTest):
    def test_exact_policy_and_prefix_mask_with_response_and_eos_supervised(self):
        row = family()[0]; tok = FakeTokenizer()
        result = self.api.prepare_example(row, POLICY, tok, 1024)
        n = result["prompt_tokens"]
        self.assertEqual(result["labels"][:n], [-100] * n)
        self.assertEqual(result["labels"][n:], result["input_ids"][n:])
        self.assertEqual(result["labels"][-1], tok.eos_token_id)
        self.assertEqual(result["attention_mask"], [1] * len(result["input_ids"]))
        self.assertEqual(result["supervised_tokens"], len(row["target_response"]) + 1)
        self.assertEqual(result["sequence_tokens"], len(result["input_ids"]))
        self.assertEqual(tok.calls[0][0], {"role": "system", "content": POLICY["texts"]["en"]})
        self.assertEqual(tok.calls[1][-1], {"role": "assistant", "content": row["target_response"]})

    def test_korean_policy_is_selected_without_rewriting(self):
        row = family()[1]; tok = FakeTokenizer()
        self.api.prepare_example(row, POLICY, tok, 1024)
        self.assertEqual(tok.calls[0][0]["content"], POLICY["texts"]["ko"])

    def test_overlength_is_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "limit|length|exceed"):
            self.api.prepare_example(family()[0], POLICY, FakeTokenizer(), 20)

    def test_mismatched_prefix_incomplete_target_and_missing_eos_are_rejected(self):
        for defect in ("prefix", "truncated_target", "eos"):
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.api.prepare_example(family()[0], POLICY, FakeTokenizer(defect), 1024)

    def test_special_token_injection_in_either_request_or_answer_is_rejected(self):
        for key in ("request", "target_response"):
            row = family()[0]; row[key] += "<ROLE>"
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.prepare_example(row, POLICY, FakeTokenizer(), 1024)

    def test_tokenizer_cannot_silently_change_authored_target_whitespace(self):
        row = family()[0]; row["target_response"] = " " + row["target_response"]
        with self.assertRaises(ValueError):
            self.api.prepare_example(row, POLICY, FakeTokenizer("normalized_whitespace"), 1024)

    def test_empty_answer_and_invalid_limit_are_rejected(self):
        row = family()[0]; row["target_response"] = " "
        with self.assertRaises(ValueError):
            self.api.prepare_example(row, POLICY, FakeTokenizer(), 1024)
        for limit in (True, 0, 1, 5.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.api.prepare_example(family()[0], POLICY, FakeTokenizer(), limit)

    def test_cli_refuses_existing_directory_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "keep.txt"; marker.write_text("unchanged")
            result = self.api.main(["--output", directory, "--data", "/nonexistent"])
            self.assertEqual(result, 1)
            self.assertEqual(marker.read_text(), "unchanged")


class PreflightCliTests(ContractTest):
    def _inputs(self, directory, *, version=1):
        rows = [row for i in range(8) for row in family(f"train_{i}")]
        rows += [row for i in range(4) for row in family(f"cal_{i}", "calibration")]
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/correction-lora.json").read_text())
        config["policy_sha256"] = hashlib.sha256(json.dumps(POLICY).encode()).hexdigest()
        if version == 2:
            rows += [row for i in range(8, 16) for row in family(f"train_{i}")]
            rows += [row for i in range(8) for row in utility_family(f"utility_{i}")]
            config = v2_config(config)
        runtime = {k: config[k] for k in ("model", "model_revision", "dtype", "seed", "batch_size")}
        runtime["local_device"] = "mps"
        values = {"data": "\n".join(json.dumps(r) for r in rows) + "\n", "evaluation": "",
                  "policy": json.dumps(POLICY), "config": json.dumps(config), "runtime": json.dumps(runtime)}
        paths = {}
        for name, value in values.items():
            paths[name] = directory / name
            paths[name].write_text(value)
        return paths

    def _args(self, paths, output):
        return [arg for name, path in paths.items() for arg in ("--" + name, str(path))] + ["--output", str(output)]

    def test_cli_writes_hash_bound_separate_splits_with_no_training_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); paths = self._inputs(root); output = root / "prepared"
            provenance = {"actual_snapshot_commit": "fixture", "local_files_only": True}
            with patch.object(self.api, "load_cached_tokenizer", return_value=(FakeTokenizer(), "fixture template", provenance)) as load:
                self.assertEqual(self.api.main(self._args(paths, output)), 0)
            load.assert_called_once()
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["optimizer_steps"], 0)
            self.assertFalse(manifest["model_weights_loaded"])
            self.assertEqual(manifest["human_review_pending"], 48)
            self.assertEqual(manifest["test_generated_samples"], 0)
            self.assertEqual(manifest["test_scored_samples"], 0)
            self.assertEqual(manifest["memory_preflight"], "not_run")
            for name, digest in manifest["files"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            for name, metadata in manifest["inputs"].items():
                self.assertEqual(hashlib.sha256(paths[name].read_bytes()).hexdigest(), metadata["sha256"])
            for split, count in (("train", 32), ("calibration", 16)):
                rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), count)
                self.assertTrue(all(row["split"] == split for row in rows))
                for row in rows:
                    n = row["prompt_tokens"]
                    self.assertEqual(row["labels"][:n], [-100] * n)
                    self.assertEqual(row["labels"][n:], row["input_ids"][n:])
                    self.assertEqual(row["labels"][-1], 2)

    def test_v2_cli_selects_versioned_default_data_and_writes_80_16_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); paths = self._inputs(root, version=2); output = root / "prepared"
            canonical = root / "data/corrections-v2/examples.jsonl"
            data_bytes = paths["data"].read_bytes(); paths["data"].unlink()
            canonical.parent.mkdir(parents=True); canonical.write_bytes(data_bytes)
            del paths["data"]
            with patch.object(self.api, "ROOT", root), patch.object(self.api, "load_cached_tokenizer",
                    return_value=(FakeTokenizer(), "fixture template", {"actual_snapshot_commit": "fixture"})):
                self.assertEqual(self.api.main(self._args(paths, output)), 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["validation"]["counts_by_split"], {"train": 80, "calibration": 16})
            self.assertEqual(manifest["validation"]["family_counts_by_split"], {"train": 24, "calibration": 4})
            self.assertEqual(manifest["validation"]["counts_by_category"], {"benign": 40, "disallowed": 40, "utility": 16})
            self.assertEqual(manifest["inputs"]["data"]["path"], str(canonical.resolve()))
            self.assertEqual(manifest["optimizer_steps"], 0)

    def test_v2_rejects_schedule_or_unknown_experiment_before_tokenization(self):
        for field, value in (("epochs", 1), ("epochs", True), ("train_steps", 159),
                             ("learning_rate", 0.0001), ("experiment_id", "../rogue")):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); paths = self._inputs(root, version=2); output = root / "prepared"
                config = json.loads(paths["config"].read_text()); config[field] = value
                paths["config"].write_text(json.dumps(config))
                with patch.object(self.api, "load_cached_tokenizer", side_effect=AssertionError("Tokenizer must not load")):
                    self.assertEqual(self.api.main(self._args(paths, output)), 1)
                self.assertFalse(output.exists())

    def test_wrong_counts_and_changed_model_config_fail_before_tokenizer_load(self):
        for defect in ("count", "model", "revision", "cap", "steps", "policy", "supervision", "policy_bytes", "policy_hash"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); paths = self._inputs(root); output = root / "prepared"
                if defect == "count":
                    paths["data"].write_text("\n".join(paths["data"].read_text().splitlines()[:-4]) + "\n")
                elif defect == "policy_bytes":
                    changed = json.loads(paths["policy"].read_text())
                    changed["texts"]["en"] += " Changed policy."
                    paths["policy"].write_text(json.dumps(changed))
                else:
                    config = json.loads(paths["config"].read_text())
                    key, value = {"model": ("model", "wrong/model"), "revision": ("model_revision", "main"),
                                  "cap": ("max_sequence_tokens", 512), "steps": ("train_steps", 24),
                                  "policy": ("system_prompt", "extra_appendix"),
                                  "policy_hash": ("policy_sha256", "not-a-hash"),
                                  "supervision": ("loss", "all_tokens")}[defect]
                    config[key] = value; paths["config"].write_text(json.dumps(config))
                with patch.object(self.api, "load_cached_tokenizer") as load:
                    self.assertEqual(self.api.main(self._args(paths, output)), 1)
                load.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
