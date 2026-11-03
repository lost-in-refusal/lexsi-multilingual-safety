"""Adapter identity and original-policy inference guards; no GPU needed."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def blob(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture():
    runtime = {"model": "CohereLabs/aya-expanse-8b", "model_revision": "a" * 40,
               "dtype": "bfloat16", "local_device": "mps"}
    config = {"model": runtime["model"], "model_revision": runtime["model_revision"],
              "dtype": "bfloat16", "device": "mps", "policy_sha256": "b" * 64,
              "system_prompt": "original_policy_only", "train_steps": 32,
              "lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
              "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
    adapter = {"base_model_name_or_path": runtime["model"], "revision": runtime["model_revision"],
               "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 4, "lora_alpha": 8,
               "lora_dropout": 0.0, "target_modules": list(reversed(config["target_modules"])),
               "bias": "none", "modules_to_save": None, "rank_pattern": {}, "alpha_pattern": {},
               "inference_mode": True, "use_dora": False, "use_rslora": False}
    weights = b"fixture adapter weights, not a real tensor file"
    report = {"status": "passed_training", **{k: runtime[k] for k in ("model", "model_revision")},
              "training_config": config, "training_config_sha256": sha(blob(config)),
              "optimizer_steps": 32, "adapter_directory": "adapter",
              "adapter_files": {"adapter_config.json": sha(blob(adapter)), "adapter_model.safetensors": sha(weights)},
              "saved_adapter_reload_exact": True, "all_base_parameters_frozen": True,
              "frozen_parameter_samples_unchanged": True, "model_device": "mps:0", "base_dtype": "torch.bfloat16"}
    return runtime, config, adapter, report, weights


class AdapterValidationTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module("scripts.run_pilot_baseline")
        self.runtime, self.config, self.adapter, self.report, self.weights = fixture()

    def validate(self):
        return self.api.validate_adapter_snapshot(blob(self.report), blob(self.config), blob(self.adapter),
                                                  self.runtime, self.config["policy_sha256"])

    def test_valid_snapshot_binds_report_config_and_weights(self):
        identity = self.validate()
        self.assertEqual(identity["training_report_sha256"], sha(blob(self.report)))
        self.assertEqual(identity["adapter_files"], self.report["adapter_files"])
        self.assertEqual(identity["policy_sha256"], self.config["policy_sha256"])

    def test_rejects_failed_partial_unverified_or_wrong_model_training(self):
        for key, value in [("status", "running"), ("optimizer_steps", 31),
                           ("saved_adapter_reload_exact", False), ("all_base_parameters_frozen", False),
                           ("frozen_parameter_samples_unchanged", False), ("model_revision", "c" * 40),
                           ("training_config_sha256", "0" * 64), ("adapter_directory", "../elsewhere")]:
            with self.subTest(key=key):
                saved = deepcopy(self.report); self.report[key] = value
                with self.assertRaises(ValueError): self.validate()
                self.report = saved

    def test_rejects_wrong_policy_and_adapter_features_even_with_updated_hash(self):
        with self.assertRaises(ValueError):
            self.api.validate_adapter_snapshot(blob(self.report), blob(self.config), blob(self.adapter), self.runtime, "c" * 64)
        for key, value in [("revision", "c" * 40), ("r", 8), ("modules_to_save", ["lm_head"]),
                           ("target_modules", ["q_proj"]), ("rank_pattern", {"q_proj": 8}), ("use_dora", True),
                           ("fan_in_fan_out", True), ("exclude_modules", ["k_proj"])]:
            with self.subTest(key=key):
                saved = deepcopy(self.adapter); self.adapter[key] = value
                self.report["adapter_files"]["adapter_config.json"] = sha(blob(self.adapter))
                with self.assertRaises(ValueError): self.validate()
                self.adapter = saved

    def test_loader_verifies_actual_adapter_weights_and_does_not_require_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder); (p / "adapter").mkdir()
            for name, data in [("report.json", blob(self.report)), ("training-config.json", blob(self.config)),
                               ("adapter/adapter_config.json", blob(self.adapter)), ("adapter/adapter_model.safetensors", self.weights)]:
                (p / name).write_bytes(data)
            loaded = self.api.load_adapter_training_run(p, self.runtime, self.config["policy_sha256"])
            self.assertEqual(loaded["adapter_path"], p / "adapter")
            (p / "adapter/adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Adapter file hash"):
                self.api.load_adapter_training_run(p, self.runtime, self.config["policy_sha256"])

    def test_cli_rejects_missing_rubric_and_mixed_prompt_intervention_before_output(self):
        script = Path(self.api.__file__)
        with tempfile.TemporaryDirectory() as folder:
            for options in [["--adapter-training-run", folder],
                            ["--adapter-training-run", folder, "--intervention", "unused", "--rubric", "unused"]]:
                out = Path(folder) / "never-created"
                result = subprocess.run([sys.executable, str(script), "--output", str(out), *options], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(out.exists())


if __name__ == "__main__": unittest.main()
