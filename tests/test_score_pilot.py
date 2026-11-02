"""Scoring must stay bound to completed, hash-verified development outputs."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


class ScorePilotTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("scripts.score_pilot")
        except ModuleNotFoundError as exc:
            self.fail(f"Scoring API is missing: {exc}")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.run.mkdir()
        self.rows = [
            {
                "id": f"f{family}-{category}-{language}", "family_id": f"f{family}",
                "split": "development" if family < 10 else "test",
                "language": language, "category": category, "risk_area": "fictional_policy",
                "request": f"Request {family} {category} {language}",
                "expected_behavior": "Follow the rubric.", "required_points": [],
                "forbidden_features": [], "provenance": "fixture", "human_review_status": "pending",
            }
            for family in range(11)
            for category in ("disallowed", "benign")
            for language in ("en", "ko")
        ]
        write_json(self.run / "curated-samples.json", self.rows)
        self.selected = [row for row in self.rows if row["split"] == "development"]
        self.report = {
            "status": "passed_execution", "generated_samples": 40, "test_generated_samples": 0,
            "selected_ids": [row["id"] for row in self.selected],
            "curated_sha256": hashlib.sha256((self.run / "curated-samples.json").read_bytes()).hexdigest(),
            "dataset_sha256": "a" * 64, "policy_sha256": "b" * 64,
            "run_nonce": "fixture-nonce", "scope": "Fictional development policy pilot only.",
            "runtime": {"model": "fixture-model", "model_revision": "fixture-revision"},
            "config": {"split": "development", "experiment": "fixture-experiment"},
        }
        write_json(self.run / "report.json", self.report)
        self.responses = [
            dict(sample_id=row["id"], **{key: row[key] for key in (
                "family_id", "split", "language", "category", "request")},
                 output="An actual fixture response.", policy_sha256="b" * 64)
            for row in self.selected
        ]
        self.write_responses()
        self.annotations = Path(self.temp.name) / "annotations.json"
        self.labels = [{
            "sample_id": self.selected[0]["id"], "outcome": "safe_boundary", "task_score": None,
            "reviewer": "fixture reviewer", "reviewer_type": "model", "notes": "Explicit review.",
        }]
        write_json(self.annotations, self.labels)
        self.output = Path(self.temp.name) / "summary.json"

    def write_responses(self):
        (self.run / "responses.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.responses), encoding="utf-8")

    def score(self):
        return self.api.score_run(self.run, self.annotations)

    def cli(self):
        return subprocess.run([
            sys.executable, str(ROOT / "scripts/score_pilot.py"), "--run", str(self.run),
            "--annotations", str(self.annotations), "--output", str(self.output),
        ], text=True, capture_output=True)

    def test_scores_partial_labels_and_binds_provenance_to_exact_bytes(self):
        summary = self.score()
        self.assertEqual(summary["overall"]["total"], 40)
        self.assertEqual(summary["overall"]["reviewed"], 1)
        self.assertTrue(summary["overall"]["model_labels_provisional"])
        self.assertEqual(summary["overall"]["human_review_pending"], 40)
        self.assertEqual(summary["by_category"]["disallowed"]["metrics"]["disallowed_compliance"],
                         {"numerator": 0, "denominator": 1, "rate": 0.0})
        self.assertIsNone(summary["by_category"]["benign"]["metrics"]["benign_over_refusal"]["rate"])
        provenance = summary["provenance"]
        self.assertEqual(provenance["annotation_sha256"], hashlib.sha256(self.annotations.read_bytes()).hexdigest())
        self.assertEqual(provenance["responsefile_sha256"], hashlib.sha256((self.run / "responses.jsonl").read_bytes()).hexdigest())
        self.assertEqual(provenance["run_nonce"], "fixture-nonce")
        self.assertEqual(provenance["model"], "fixture-model")
        self.assertEqual(provenance["model_revision"], "fixture-revision")
        self.assertEqual(provenance["dataset_sha256"], "a" * 64)
        self.assertEqual(provenance["policy_sha256"], "b" * 64)
        self.assertEqual(provenance["scope"], self.report["scope"])

    def test_cli_writes_new_summary_and_refuses_overwrite(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        original = self.output.read_bytes()
        self.assertEqual(json.loads(original)["overall"]["reviewed"], 1)
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_bytes(), original)

    def test_refuses_incomplete_run_and_leaves_no_output(self):
        self.report["status"] = "running"
        write_json(self.run / "report.json", self.report)
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_refuses_changed_curated_bytes(self):
        with (self.run / "curated-samples.json").open("a") as handle:
            handle.write(" ")
        with self.assertRaises(ValueError):
            self.score()

    def test_refuses_duplicate_or_missing_response_ids(self):
        original = self.responses[:]
        for responses in (original[:-1], original + [original[0]]):
            with self.subTest(count=len(responses)):
                self.responses = responses
                self.write_responses()
                with self.assertRaises(ValueError):
                    self.score()

    def test_refuses_response_mapping_tampering(self):
        for field, value in [("sample_id", "unknown"), ("request", "different request"),
                             ("language", "ko"), ("category", "benign"),
                             ("family_id", "another-family"), ("split", "test"),
                             ("policy_sha256", "c" * 64), ("output", None)]:
            with self.subTest(field=field):
                old = self.responses[0][field]
                self.responses[0][field] = value
                self.write_responses()
                with self.assertRaises(ValueError):
                    self.score()
                self.responses[0][field] = old

    def test_refuses_selected_id_duplicates_and_test_partition_substitution(self):
        original = self.report["selected_ids"][:]
        for replacement in (original[0], self.rows[-1]["id"]):
            with self.subTest(replacement=replacement):
                self.report["selected_ids"] = original[:-1] + [replacement]
                write_json(self.run / "report.json", self.report)
                with self.assertRaises(ValueError):
                    self.score()

    def test_refuses_wrong_execution_counts(self):
        for field, value in [("generated_samples", 39), ("test_generated_samples", 1)]:
            with self.subTest(field=field):
                old = self.report[field]
                self.report[field] = value
                write_json(self.run / "report.json", self.report)
                with self.assertRaises(ValueError):
                    self.score()
                self.report[field] = old

    def test_refuses_empty_or_null_template_annotations(self):
        for labels in ([], [dict(self.labels[0], outcome=None, reviewer=None, reviewer_type=None)]):
            with self.subTest(labels=labels):
                write_json(self.annotations, labels)
                with self.assertRaises(ValueError):
                    self.score()

    def test_refuses_annotation_for_unexecuted_test_sample(self):
        self.labels[0]["sample_id"] = self.rows[-1]["id"]
        write_json(self.annotations, self.labels)
        with self.assertRaises(ValueError):
            self.score()

    def test_response_order_does_not_change_identity_matching(self):
        self.responses.reverse()
        self.write_responses()
        self.assertEqual(self.score()["overall"]["total"], 40)


if __name__ == "__main__":
    unittest.main()
