"""Synthetic fixtures for explicit test-split authorization and immutable packs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_pilot_baseline as runner
from scripts import grounding_comparison as comparison
from tests import test_grounding_comparison as fixtures
from tests.test_grounding_comparison import write_json, digest


SNAPSHOT_NAMES = ("confirmation-freeze.json", "development-gate-report.json", "development-pack-manifest.json",
                  "confirmation-goal-config.json", "confirmation-protocol.md")


def make_confirmation_fixture(case):
    """Only synthetic requests; never reads the actual benchmark."""
    case.make_modern_baseline()
    case.make_adapted_candidate()
    rows = []
    for family in range(6):
        for category in (("disallowed", "benign") if family < 4 else ("utility",)):
            for language in ("en", "ko"):
                row = deepcopy(case.rows[0])
                row.update(id=f"synthetic-heldout-{family}-{category}-{language}", family_id=f"synthetic-heldout-{family}",
                           split="test", category=category, language=language,
                           request=f"Synthetic confirmation request {family} {category} {language}")
                rows.append(row)
    for path in (case.baseline, case.candidate):
        write_json(path / "curated-samples.json", case.rows + rows)
        write_json(path / "raw-samples.json", case.rows + rows)
        case.mutate_report(path, ("curated_sha256",), digest(path / "curated-samples.json"))
    development_pack = case.root / "development-pack"
    manifest = case.create_adapter_comparison(output=development_pack)
    report = case.read(case.candidate / "report.json")
    goal = {"primary_absolute_gain_pp": 20, "minimum_language_gain_pp": 10,
            "minimum_dimension_coverage": 0.9, "complete_annotation_required": True}
    goal_bytes = comparison._json_bytes(goal)
    protocol_bytes = b"Frozen synthetic practical confirmation protocol.\n"
    sha = comparison._sha
    gate = {"schema_version": 1, "comparison": {"reference": "baseline", "candidate": "plain_lora"},
            "goal_configuration": goal, "goal_complete": False, "heldout_confirmed": False,
            "gate": {"phase": "development_selection_only", "passed": True, "failed_checks": [],
                     "checks": [{"name": "synthetic gate", "passed": True}]},
            "provenance": {"pack": str(development_pack), "pack_manifest_sha256": digest(development_pack / "manifest.json"),
                           "review_context_sha256": manifest["review_context_sha256"], "pack_files": manifest["files"],
                           "goal_config_sha256": sha(goal_bytes)}}
    gate_path = case.root / "development-gate.json"
    write_json(gate_path, gate)
    freeze = {"schema_version": 1, "phase": "heldout_confirmation", "confirmation_id": "fixture-confirmation-0001",
              "selected_adapter_provenance": report["adapter_provenance"],
              "development_gate_report": {"path": str(gate_path), "sha256": digest(gate_path)},
              "goal_config_sha256": sha(goal_bytes), "protocol_sha256": sha(protocol_bytes),
              **{key: report[key] for key in ("dataset_sha256", "policy_sha256", "rubric_sha256")},
              **{key: report["runtime"][key] for key in ("model", "model_revision")},
              "generation_settings": {"seed": 42, "temperature": 0.0, "max_new_tokens": 512,
                                      "batch_size": 1, "mps_memory_fraction": 0.8},
              "selected_test_ids": [row["id"] for row in rows], "family_count": 6}
    snapshots = {"confirmation-freeze.json": comparison._json_bytes(freeze),
                 "development-gate-report.json": gate_path.read_bytes(),
                 "development-pack-manifest.json": (development_pack / "manifest.json").read_bytes(),
                 "confirmation-goal-config.json": goal_bytes, "confirmation-protocol.md": protocol_bytes}
    return freeze, snapshots, rows


def convert_to_heldout(case, freeze, snapshots, rows):
    """Make two fake completed test runs without executing any model."""
    for path in (case.baseline, case.candidate):
        report = case.read(path / "report.json")
        for name, content in snapshots.items():
            (path / name).write_bytes(content)
        runner_bytes = b"# Synthetic archived runner fixture\n"
        (path / "generation-runner.py").write_bytes(runner_bytes)
        report.update(generated_samples=20, test_generated_samples=20, development_generated_samples=0,
                      phase="heldout_confirmation", confirmation_id=freeze["confirmation_id"],
                      confirmation_freeze_sha256=digest(path / "confirmation-freeze.json"),
                      confirmation_snapshot_hashes={name: digest(path / name) for name in SNAPSHOT_NAMES},
                      generation_runner_sha256=digest(path / "generation-runner.py"),
                      source_hashes={"scripts/run_pilot_baseline.py": digest(path / "generation-runner.py")},
                      full_input_token_verification="exact_full_chat_template_token_ids",
                      selected_ids=freeze["selected_test_ids"])
        report["config"]["split"] = "test"
        all_rows = case.rows + rows
        write_json(path / "curated-samples.json", all_rows)
        write_json(path / "raw-samples.json", all_rows)
        report["curated_sha256"] = digest(path / "curated-samples.json")
        (path / "rubric.md").write_bytes(case.rubric.read_bytes())
        report["rubric_sha256"] = digest(path / "rubric.md")
        responses = []
        for row in rows:
            response = {key: row[key] for key in ("family_id", "split", "language", "category", "request")}
            response.update(sample_id=row["id"], output="Synthetic held-out answer", cap_reached=False, ended_with_eos=True,
                            policy_sha256=report["policy_sha256"], effective_prompts_sha256=report["effective_prompts_sha256"],
                            run_nonce=report["run_nonce"], confirmation_freeze_sha256=report["confirmation_freeze_sha256"],
                            input_tokens=100, full_input_token_ids_sha256="a" * 64)
            if path == case.candidate:
                response["adapter_provenance_sha256"] = report["adapter_provenance_sha256"]
            responses.append(response)
        case.write_responses(path, responses)
        report["responses_sha256"] = digest(path / "responses.jsonl")
        write_json(path / "report.json", report)
        audit = case.read(path / "auditkit-result.json")
        audit["config"]["extra"].update(confirmation_freeze_sha256=report["confirmation_freeze_sha256"], split="test",
                                         model_revision=report["runtime"]["model_revision"])
        audit["config"]["extra"].update({field: report[field] for field in (
            "dataset_sha256", "curated_sha256", "policy_sha256", "tools", "variant",
            "effective_prompts_sha256", "rubric_sha256", "generation_runner_sha256")})
        audit["predictions"] = [{"sample_id": r["sample_id"], "prompt": r["request"], "raw_output": r["output"]} for r in responses]
        write_json(path / "auditkit-result.json", audit)


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.GroundingComparisonTests(methodName="runTest")
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.freeze, self.snapshots, self.rows = make_confirmation_fixture(self.case)
        self.report = self.case.read(self.case.candidate / "report.json")

    def validate(self, snapshots=None, identity=None):
        return runner.validate_confirmation_snapshot(snapshots or self.snapshots, self.report["runtime"], self.report["config"],
            dataset_sha256=self.report["dataset_sha256"], policy_sha256=self.report["policy_sha256"],
            rubric_sha256=self.report["rubric_sha256"], adapter_identity=identity)

    def create_pack(self, **options):
        return comparison.create_review_pack(self.case.baseline, self.case.candidate, self.case.rubric, self.case.pack,
                                            candidate_mode="adapted", candidate_label="plain_lora", **options)

    def test_confirmation_binds_development_source_and_archive_does_not_need_original_paths(self):
        self.assertEqual(self.validate(), self.freeze)
        (self.case.root / "development-gate.json").unlink()
        (self.case.root / "development-pack/manifest.json").unlink()
        self.assertEqual(self.validate(), self.freeze)

    def test_live_loader_checks_goal_protocol_and_gate_files_then_snapshots_them(self):
        root = self.case.root / "source"
        (root / "configs").mkdir(parents=True)
        (root / "docs").mkdir()
        (root / "configs/model-improvement-goal.json").write_bytes(self.snapshots["confirmation-goal-config.json"])
        (root / "docs/model-improvement-protocol.md").write_bytes(self.snapshots["confirmation-protocol.md"])
        freeze_path = self.case.root / "freeze.json"
        freeze_path.write_bytes(self.snapshots["confirmation-freeze.json"])
        expected = {key: self.report[key] for key in ("dataset_sha256", "policy_sha256", "rubric_sha256")}
        with patch.object(runner, "ROOT", root):
            loaded = runner.load_confirmation_freeze(freeze_path, self.report["runtime"], self.report["config"], **expected)
            self.assertEqual(loaded["snapshots"], self.snapshots)
            (root / "docs/model-improvement-protocol.md").write_text("Changed protocol")
            with self.assertRaisesRegex(ValueError, "protocol snapshot hash"):
                runner.load_confirmation_freeze(freeze_path, self.report["runtime"], self.report["config"], **expected)

    def test_rejects_wrong_adapter_settings_ids_scope_and_gate_even_when_freeze_rehashed(self):
        for key, value in [("selected_adapter_provenance", {**self.freeze["selected_adapter_provenance"], "training_report_sha256": "0" * 64}),
                           ("selected_test_ids", self.freeze["selected_test_ids"][:-1]), ("family_count", 5),
                           ("phase", "development_selection_only"), ("model_revision", "other"),
                           ("generation_settings", {**self.freeze["generation_settings"], "temperature": 0.1})]:
            with self.subTest(key=key):
                mutated = {**self.freeze, key: value}
                with self.assertRaises(ValueError):
                    self.validate({**self.snapshots, "confirmation-freeze.json": comparison._json_bytes(mutated)})
        with self.assertRaisesRegex(ValueError, "Inference adapter"):
            self.validate(identity={**self.freeze["selected_adapter_provenance"], "model_revision": "wrong"})
        for field, value in [("passed", False), ("phase", "heldout_confirmation_review"),
                             ("checks", [{"name": "bad", "passed": False}])]:
            gate = json.loads(self.snapshots["development-gate-report.json"])
            gate["gate"][field] = value
            gate_bytes = comparison._json_bytes(gate)
            frozen = deepcopy(self.freeze)
            frozen["development_gate_report"]["sha256"] = comparison._sha(gate_bytes)
            with self.assertRaises(ValueError):
                self.validate({**self.snapshots, "development-gate-report.json": gate_bytes,
                               "confirmation-freeze.json": comparison._json_bytes(frozen)})

    def test_development_manifest_must_bind_selected_weights_not_just_passed_boolean(self):
        manifest = json.loads(self.snapshots["development-pack-manifest.json"])
        manifest["runs"]["plain_lora"]["files"]["adapter-provenance.json"] = "0" * 64
        manifest_bytes = comparison._json_bytes(manifest)
        gate = json.loads(self.snapshots["development-gate-report.json"])
        gate["provenance"]["pack_manifest_sha256"] = comparison._sha(manifest_bytes)
        gate_bytes = comparison._json_bytes(gate)
        freeze = deepcopy(self.freeze)
        freeze["development_gate_report"]["sha256"] = comparison._sha(gate_bytes)
        with self.assertRaisesRegex(ValueError, "Selected adapter"):
            self.validate({**self.snapshots, "development-pack-manifest.json": manifest_bytes,
                           "development-gate-report.json": gate_bytes, "confirmation-freeze.json": comparison._json_bytes(freeze)})

    def test_explicit_test_pack_roundtrip_has_20_pairs_6_families_and_bound_freeze(self):
        convert_to_heldout(self.case, self.freeze, self.snapshots, self.rows)
        manifest = self.create_pack(split="test")
        self.assertEqual((manifest["schema_version"], manifest["sample_count"], manifest["review_item_count"], manifest["family_count"]), (4, 20, 40, 6))
        self.assertEqual(manifest["phase"], "heldout_confirmation")
        loaded, _, items, pairs = comparison._load_pack(self.case.pack)
        self.assertEqual(len(items), 40)
        self.assertEqual([len(rows) for rows in pairs.values()], [20, 20])
        self.assertEqual(loaded["confirmation_freeze_sha256"], comparison._sha(self.snapshots["confirmation-freeze.json"]))
        empty_labels = self.case.root / "empty-labels.json"
        write_json(empty_labels, [])
        summary = comparison.summarize_review(self.case.pack, [empty_labels], self.case.root / "heldout-summary.json")
        self.assertEqual(summary["provenance"]["phase"], "heldout_confirmation_review")
        self.assertEqual(summary["paired"]["overall"]["language_adherence"]["total_pairs"], 20)

    def test_wrong_run_split_selected_ids_nonce_counter_or_freeze_rejected(self):
        convert_to_heldout(self.case, self.freeze, self.snapshots, self.rows)
        with self.assertRaises(ValueError): self.create_pack()  # Explicit test mode required.
        report = self.case.read(self.case.candidate / "report.json")
        mutations = [("selected_ids", report["selected_ids"][::-1]), ("development_generated_samples", 1),
                     ("test_generated_samples", 0), ("confirmation_id", "another-confirmation-id"),
                     ("confirmation_freeze_sha256", "0" * 64), ("run_nonce", "baseline-nonce")]
        for key, value in mutations:
            with self.subTest(key=key):
                write_json(self.case.candidate / "report.json", {**report, key: value})
                with self.assertRaises(ValueError): self.create_pack(split="test")
        write_json(self.case.candidate / "report.json", report)

    def test_schema4_loader_rejects_tampered_scope_and_selected_ids(self):
        convert_to_heldout(self.case, self.freeze, self.snapshots, self.rows)
        manifest = self.create_pack(split="test")
        for key, value in [("selected_ids", manifest["selected_ids"][::-1]), ("family_count", 12),
                           ("phase", "development_selection_only"), ("sample_count", 40)]:
            with self.subTest(key=key):
                write_json(self.case.pack / "manifest.json", {**manifest, key: value})
                with self.assertRaises(ValueError): comparison._load_pack(self.case.pack)
        write_json(self.case.pack / "manifest.json", manifest)

    def test_each_run_valid_but_different_confirmation_freezes_cannot_be_paired(self):
        convert_to_heldout(self.case, self.freeze, self.snapshots, self.rows)
        candidate = self.case.candidate
        freeze = deepcopy(self.freeze)
        freeze["confirmation_id"] = "fixture-confirmation-0002"
        write_json(candidate / "confirmation-freeze.json", freeze)
        new_sha = digest(candidate / "confirmation-freeze.json")
        report = self.case.read(candidate / "report.json")
        report.update(confirmation_id=freeze["confirmation_id"], confirmation_freeze_sha256=new_sha)
        report["confirmation_snapshot_hashes"]["confirmation-freeze.json"] = new_sha
        responses = [json.loads(line) for line in (candidate / "responses.jsonl").read_text().splitlines()]
        for response in responses:
            response["confirmation_freeze_sha256"] = new_sha
        self.case.write_responses(candidate, responses)
        report["responses_sha256"] = digest(candidate / "responses.jsonl")
        write_json(candidate / "report.json", report)
        audit = self.case.read(candidate / "auditkit-result.json")
        audit["config"]["extra"]["confirmation_freeze_sha256"] = new_sha
        write_json(candidate / "auditkit-result.json", audit)
        comparison._load_run(candidate, "adapted", report["rubric_sha256"], "test")
        with self.assertRaisesRegex(ValueError, "different freezes"):
            self.create_pack(split="test")

    def test_existing_development_pack_ids_remain_frozen(self):
        expected = {1: "cabcab87d3d7cff2530174d60b8ef68b6cf3f3fec8e02fadfeb0610c562b321c",
                    2: "45d12b94bc489fece4775b29e124ad5b22e59bc37912cca10521b09cecb7ba92",
                    3: "f3daecbdcb0ebeb3cd7eb363babc0dbc6d21d9ffa2777313d4dc12ad8f25aca5"}
        for schema, expected_sha in expected.items():
            case = fixtures.GroundingComparisonTests(methodName="runTest")
            case.setUp()
            try:
                reference, options = case.baseline, {}
                if schema == 2:
                    reference = case.make_grounded_reference()
                    options = dict(reference_mode="grounded", reference_label="grounded_with_conciseness",
                                   candidate_label="grounded_without_conciseness")
                elif schema == 3:
                    case.make_adapted_candidate()
                    options = dict(candidate_mode="adapted", candidate_label="plain_lora")
                manifest = comparison.create_review_pack(reference, case.candidate, case.rubric, case.pack, **options)
                self.assertEqual(manifest["schema_version"], schema)
                self.assertEqual((manifest["sample_count"], manifest["review_item_count"]), (40, 80))
                self.assertEqual(digest(case.pack / "review-items.json"), expected_sha)
            finally:
                case.doCleanups()

    def test_test_cli_requires_explicit_freeze_and_rubric_and_forbids_appendix(self):
        for options in [["--split", "test"], ["--split", "test", "--confirmation-freeze", "unused"],
                        ["--split", "test", "--confirmation-freeze", "unused", "--rubric", "unused", "--intervention", "unused"],
                        ["--confirmation-freeze", "unused"]]:
            with self.subTest(options=options):
                out = self.case.root / "never-created"
                result = subprocess.run([sys.executable, runner.__file__, "--output", str(out), *options], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(out.exists())


if __name__ == "__main__": unittest.main()
