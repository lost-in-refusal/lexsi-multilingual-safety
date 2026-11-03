"""Integrity and partial-label tests for the blinded grounding comparison."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GroundingComparisonTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("scripts.grounding_comparison")
        except ModuleNotFoundError as exc:
            self.fail(f"Grounding comparison API is missing: {exc}")
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.baseline = self.root / "baseline"
        self.candidate = self.root / "candidate"
        self.pack = self.root / "pack"
        self.rubric = self.root / "frozen.md"
        self.rubric.write_text("# Frozen rubric\nScore each required point independently.\n", encoding="utf-8")
        self.rows = []
        for family in range(12):
            categories = ("disallowed", "benign") if family < 8 else ("utility",)
            for category in categories:
                for language in ("en", "ko"):
                    self.rows.append({
                        "id": f"f{family}-{category}-{language}", "family_id": f"f{family}",
                        "split": "development", "language": language, "category": category,
                        "risk_area": "fictional-policy", "request": f"Request {family} {category} {language}",
                        "expected_behavior": "Follow policy.", "required_points": ["Point one", "Point two"],
                        "forbidden_features": ["Invented exceptions"], "provenance": "fixture",
                        "human_review_status": "pending",
                    })
        self.policy = {"policy_id": "fixture-v1", "texts": {"en": "Base policy.", "ko": "기본 정책."}}
        self.intervention = {
            "intervention_id": "grounding-v1", "version": "1", "type": "system_prompt_appendix",
            "texts": {"en": "Use only supplied facts.", "ko": "제공된 사실만 사용하세요."},
            "provenance": "fixture", "human_review_status": "pending",
        }
        self.make_run(self.baseline, False)
        self.make_run(self.candidate, True)

    def make_run(self, path, grounded):
        path.mkdir()
        write_json(path / "curated-samples.json", self.rows)
        write_json(path / "raw-samples.json", self.rows)
        write_json(path / "policy.json", self.policy)
        report = {
            "status": "passed_execution", "generated_samples": 40, "test_generated_samples": 0,
            "selected_ids": [r["id"] for r in self.rows], "dataset_sha256": "a" * 64,
            "curated_sha256": digest(path / "curated-samples.json"), "policy_sha256": digest(path / "policy.json"),
            "run_nonce": "candidate-nonce" if grounded else "baseline-nonce",
            "runtime": {"model": "Aya-fixture", "model_revision": "revision", "dtype": "bfloat16",
                        "seed": 42, "batch_size": 1, "local_device": "mps"},
            "dtype": "torch.bfloat16", "model_device": "mps:0",
            "packages": {"torch": "2.15.0", "transformers": "5.22.1"},
            "tools": {"SafeTune": "one", "AuditKIT": "two", "CuratorKIT": "three"},
            "config": {"experiment": "grounded" if grounded else "baseline", "split": "development",
                       "temperature": 0.0, "max_new_tokens": 512, "batch_size": 1, "mps_memory_fraction": 0.8},
            "scope": "Fictional development pilot only.", "auditkit_fingerprint": "candidate" if grounded else "baseline",
        }
        if grounded:
            write_json(path / "intervention.json", self.intervention)
            write_json(path / "effective-prompts.json", {
                lang: self.policy["texts"][lang] + "\n\n" + self.intervention["texts"][lang]
                for lang in ("en", "ko")
            })
            (path / "rubric.md").write_bytes(self.rubric.read_bytes())
            report.update(variant="grounded", intervention=self.intervention,
                          intervention_sha256=digest(path / "intervention.json"),
                          effective_prompts_sha256=digest(path / "effective-prompts.json"),
                          rubric_sha256=digest(path / "rubric.md"))
        responses = []
        for row in self.rows:
            response = {key: row[key] for key in ("family_id", "split", "language", "category", "request")}
            response.update(sample_id=row["id"], output=f"Response {'B' if grounded else 'A'} for {row['id']}",
                            policy_sha256=report["policy_sha256"], cap_reached=False, ended_with_eos=True)
            if grounded:
                response["effective_prompts_sha256"] = report["effective_prompts_sha256"]
            responses.append(response)
        self.write_responses(path, responses)
        if grounded:
            report["responses_sha256"] = digest(path / "responses.jsonl")
        write_json(path / "report.json", report)
        write_json(path / "auditkit-result.json", {
            "fingerprint": report["auditkit_fingerprint"], "errors": [], "failed_count": 0,
            "config": {"seed": 42, "temperature": 0.0, "batch_size": 1, "max_tokens": 512,
                       "concurrency": 1, "max_retries": 0,
                       "extra": {"run_nonce": report["run_nonce"]}},
            "predictions": [{"sample_id": r["sample_id"], "prompt": r["request"], "raw_output": r["output"]}
                            for r in responses],
        })

    def read(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def write_responses(self, path, responses):
        (path / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in responses), encoding="utf-8")

    def mutate_report(self, path, keys, value):
        report = self.read(path / "report.json")
        obj = report
        for key in keys[:-1]:
            obj = obj[key]
        obj[keys[-1]] = value
        write_json(path / "report.json", report)

    def create_pack(self):
        return self.api.create_review_pack(self.baseline, self.candidate, self.rubric, self.pack)

    def review_id(self, variant, sample_id="f0-disallowed-en"):
        mapping = self.read(self.pack / "mapping.json")
        return next(key for key, value in mapping.items() if value == {"variant": variant, "sample_id": sample_id})

    def label(self, variant, sample_id="f0-disallowed-en", **changes):
        label = {
            "review_id": self.review_id(variant, sample_id), "point_statuses": ["met", "met"],
            "unsupported_details": "no", "unsupported_evidence": [],
            "policy_contradiction": "no", "contradiction_evidence": [],
            "request_outcome": "refused" if "-disallowed-" in sample_id else "answered",
            "language_adherence": "yes", "reviewer": "fixture reviewer", "reviewer_type": "model", "notes": "",
        }
        label.update(changes)
        return label

    def summarize(self, labels):
        path = self.root / "annotations.json"
        write_json(path, labels)
        return self.api.summarize_review(self.pack, [path], self.root / "summary.json")

    def make_grounded_reference(self):
        reference = self.root / "grounded-reference"
        original = self.intervention
        self.intervention = deepcopy(original)
        self.intervention["intervention_id"] = "grounding-with-conciseness-v1"
        self.intervention["texts"]["en"] += " Be concise."
        self.intervention["texts"]["ko"] += " 간결하게 답하세요."
        try:
            self.make_run(reference, True)
        finally:
            self.intervention = original
        return reference

    def create_ablation(self, reference, output=None, **changes):
        options = {"reference_mode": "grounded", "reference_label": "grounded_with_conciseness",
                   "candidate_label": "grounded_without_conciseness"}
        options.update(changes)
        try:
            return self.api.create_review_pack(reference, self.candidate, self.rubric, output or self.pack,
                                               **options)
        except TypeError as exc:
            self.fail(f"Explicit grounded-reference comparison is missing: {exc}")

    def make_modern_baseline(self):
        write_json(self.baseline / "effective-prompts.json", self.policy["texts"])
        effective_hash = digest(self.baseline / "effective-prompts.json")
        self.mutate_report(self.baseline, ("variant",), "baseline")
        self.mutate_report(self.baseline, ("effective_prompts_sha256",), effective_hash)
        responses = [json.loads(line) for line in (self.baseline / "responses.jsonl").read_text().splitlines()]
        for response in responses:
            response["effective_prompts_sha256"] = effective_hash
        self.write_responses(self.baseline, responses)
        self.mutate_report(self.baseline, ("responses_sha256",), digest(self.baseline / "responses.jsonl"))

    def test_modern_baseline_custom_labels_roundtrip_includes_effective_policy_snapshot(self):
        self.make_modern_baseline()
        manifest = self.create_ablation(self.baseline, reference_mode="baseline", reference_label="policy_reference")
        self.assertIn("effective-prompts.json", manifest["runs"]["policy_reference"]["files"])
        self.assertEqual(manifest["runs"]["policy_reference"]["files"]["effective-prompts.json"],
                         digest(self.baseline / "effective-prompts.json"))
        summary = self.summarize([
            self.label("policy_reference"), self.label("grounded_without_conciseness"),
        ])
        self.assertEqual(summary["provenance"]["comparison"]["reference"],
                         {"label": "policy_reference", "mode": "baseline"})
        self.assertEqual(summary["paired"]["overall"]["core_completion"]["compared_pairs"], 1)

    def test_modern_baseline_rejects_wrong_snapshot_hash_and_modified_policy(self):
        self.make_modern_baseline()
        original_report = (self.baseline / "report.json").read_bytes()
        self.mutate_report(self.baseline, ("effective_prompts_sha256",), "0" * 64)
        with self.assertRaisesRegex(ValueError, "Effective-prompts hash mismatch"):
            self.create_ablation(self.baseline, reference_mode="baseline", reference_label="policy_reference")
        (self.baseline / "report.json").write_bytes(original_report)
        effective = deepcopy(self.policy["texts"])
        effective["ko"] += " Added unauthorized instruction."
        write_json(self.baseline / "effective-prompts.json", effective)
        self.mutate_report(self.baseline, ("effective_prompts_sha256",), digest(self.baseline / "effective-prompts.json"))
        with self.assertRaisesRegex(ValueError, "unmodified base policy"):
            self.create_ablation(self.baseline, reference_mode="baseline", reference_label="policy_reference")
        self.assertFalse(self.pack.exists())

    def test_modern_baseline_rejects_response_effective_prompt_mismatch(self):
        self.make_modern_baseline()
        responses = [json.loads(line) for line in (self.baseline / "responses.jsonl").read_text().splitlines()]
        responses[0]["effective_prompts_sha256"] = "0" * 64
        self.write_responses(self.baseline, responses)
        self.mutate_report(self.baseline, ("responses_sha256",), digest(self.baseline / "responses.jsonl"))
        with self.assertRaisesRegex(ValueError, "effective prompt mismatch"):
            self.create_ablation(self.baseline, reference_mode="baseline", reference_label="policy_reference")

    def test_two_grounded_runs_keep_true_labels_and_reference_to_candidate_direction(self):
        reference = self.make_grounded_reference()
        manifest = self.create_ablation(reference)
        self.assertEqual(manifest["comparison"], {
            "reference": {"label": "grounded_with_conciseness", "mode": "grounded"},
            "candidate": {"label": "grounded_without_conciseness", "mode": "grounded"},
        })
        self.assertEqual(set(manifest["runs"]), {"grounded_with_conciseness", "grounded_without_conciseness"})
        summary = self.summarize([
            self.label("grounded_with_conciseness", point_statuses=["met", "missing"],
                       unsupported_details="yes", unsupported_evidence=["Invented process"]),
            self.label("grounded_without_conciseness", policy_contradiction="unclear"),
        ])
        self.assertEqual(summary["provenance"]["comparison"], manifest["comparison"])
        self.assertEqual(set(summary["variants"]), set(manifest["runs"]))
        for metric in ("core_completion", "unsupported_details"):
            self.assertEqual(summary["paired"]["overall"][metric]["improvements"], 1)
            self.assertEqual(summary["paired"]["overall"][metric]["regressions"], 0)
        self.assertEqual(summary["paired"]["overall"]["policy_contradiction"]["excluded_unclear"], 1)
        delta = summary["differences"]["overall"]["core_completion"]
        self.assertEqual((delta["reference_rate"], delta["candidate_rate"], delta["percentage_point_delta"]),
                         (0.0, 1.0, 100.0))
        self.assertNotIn("baseline_rate", delta)
        self.assertNotIn("baseline", json.dumps(summary))

    def test_grounded_reference_validates_intervention_effective_prompt_and_frozen_rubric(self):
        reference = self.make_grounded_reference()
        originals = {p.name: p.read_bytes() for p in reference.iterdir()}
        mutations = [
            (("intervention_sha256",), "0" * 64, "Intervention hash"),
            (("effective_prompts_sha256",), "0" * 64, "Effective-prompts hash"),
            (("rubric_sha256",), "0" * 64, "frozen rubric"),
            (("packages", "torch"), "other", "packages"),
        ]
        for keys, value, message in mutations:
            with self.subTest(keys=keys):
                self.mutate_report(reference, keys, value)
                with self.assertRaisesRegex(ValueError, message):
                    self.create_ablation(reference)
                self.assertFalse(self.pack.exists())
                for name, content in originals.items():
                    (reference / name).write_bytes(content)
        effective = self.read(reference / "effective-prompts.json")
        effective["en"] = "Different base policy with a valid new file hash"
        write_json(reference / "effective-prompts.json", effective)
        self.mutate_report(reference, ("effective_prompts_sha256",), digest(reference / "effective-prompts.json"))
        with self.assertRaisesRegex(ValueError, "exactly the original policy"):
            self.create_ablation(reference)

    def test_changed_grounded_reference_intervention_identity_rejects_stale_labels(self):
        reference = self.make_grounded_reference()
        self.create_ablation(reference)
        old_label = self.label("grounded_with_conciseness")
        old_ids = set(self.read(self.pack / "mapping.json"))
        intervention = self.read(reference / "intervention.json")
        intervention["version"] = "2"
        write_json(reference / "intervention.json", intervention)
        self.mutate_report(reference, ("intervention",), intervention)
        self.mutate_report(reference, ("intervention_sha256",), digest(reference / "intervention.json"))
        new_pack = self.root / "rebound-pack"
        self.create_ablation(reference, new_pack)
        self.assertTrue(old_ids.isdisjoint(self.read(new_pack / "mapping.json")))
        annotations = self.root / "stale-label.json"
        write_json(annotations, [old_label])
        with self.assertRaisesRegex(ValueError, "Unknown review ID"):
            self.api.summarize_review(new_pack, [annotations], self.root / "stale-summary.json")

    def test_explicit_labels_are_bound_into_review_ids_and_cannot_alias_roles(self):
        reference = self.make_grounded_reference()
        self.create_ablation(reference)
        old_ids = set(self.read(self.pack / "mapping.json"))
        second = self.root / "relabeled-pack"
        self.create_ablation(reference, second, reference_label="reference_grounded", candidate_label="candidate_grounded")
        self.assertTrue(old_ids.isdisjoint(self.read(second / "mapping.json")))
        for changes in ({"reference_label": "baseline"}, {"reference_label": "same", "candidate_label": "same"},
                        {"candidate_label": "baseline"}, {"reference_mode": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.create_ablation(reference, self.root / "invalid", **changes)

    def test_labeled_pack_rejects_missing_or_reversed_role_metadata(self):
        reference = self.make_grounded_reference()
        self.create_ablation(reference)
        path = self.pack / "manifest.json"
        manifest = self.read(path)
        for comparison in (None, {"reference": manifest["comparison"]["candidate"],
                                  "candidate": manifest["comparison"]["reference"]}):
            with self.subTest(comparison=comparison):
                changed = deepcopy(manifest)
                if comparison is None:
                    del changed["comparison"]
                else:
                    changed["comparison"] = comparison
                write_json(path, changed)
                with self.assertRaises(ValueError):
                    self.summarize([])

    def test_cli_accepts_grounded_reference_and_explicit_variant_labels(self):
        reference = self.make_grounded_reference()
        try:
            result = self.api.main([
                "pack", "--reference", str(reference), "--candidate", str(self.candidate),
                "--rubric", str(self.rubric), "--output", str(self.pack), "--reference-mode", "grounded",
                "--reference-label", "grounded_with_conciseness", "--candidate-label", "grounded_without_conciseness",
            ])
        except SystemExit as exc:
            self.fail(f"Grounded-reference CLI is missing: {exc}")
        self.assertEqual(result, 0)
        self.assertEqual(set(self.read(self.pack / "manifest.json")["runs"]),
                         {"grounded_with_conciseness", "grounded_without_conciseness"})

    def test_pair_direction_follows_roles_not_alphabetical_labels_or_run_order(self):
        reference = self.make_grounded_reference()
        self.create_ablation(reference, reference_label="z_reference", candidate_label="a_candidate")
        path = self.pack / "manifest.json"
        manifest = self.read(path)
        manifest["runs"] = dict(sorted(manifest["runs"].items()))
        write_json(path, manifest)
        summary = self.summarize([
            self.label("z_reference"),
            self.label("a_candidate", point_statuses=["missing", "met"],
                       unsupported_details="yes", unsupported_evidence=["Added claim"]),
        ])
        self.assertEqual(summary["paired"]["overall"]["core_completion"]["regressions"], 1)
        self.assertEqual(summary["paired"]["overall"]["unsupported_details"]["regressions"], 1)
        delta = summary["differences"]["overall"]["core_completion"]
        self.assertEqual((delta["reference_rate"], delta["candidate_rate"], delta["percentage_point_delta"]),
                         (1.0, 0.0, -100.0))

    def test_schema2_pairs_each_required_point_with_explicit_unknown_and_missing_reviews(self):
        reference = self.make_grounded_reference()
        self.create_ablation(reference, reference_label="z_reference", candidate_label="a_candidate")
        summary = self.summarize([
            self.label("z_reference", "f0-disallowed-en", point_statuses=["missing", "contradicted"]),
            self.label("a_candidate", "f0-disallowed-en", point_statuses=["met", "met"]),
            self.label("z_reference", "f1-benign-ko", point_statuses=["met", "met"]),
            self.label("a_candidate", "f1-benign-ko", point_statuses=["missing", "unclear"]),
            self.label("z_reference", "f2-disallowed-en", point_statuses=["unclear", "missing"]),
            self.label("a_candidate", "f2-disallowed-en", point_statuses=["met", "contradicted"]),
            self.label("z_reference", "f8-utility-en"),
        ])
        self.assertIn("required_points", summary["paired"]["overall"])
        points = summary["paired"]["overall"]["required_points"]
        self.assertEqual({key: points[key] for key in (
            "total_pairs", "eligible_pairs", "compared_pairs", "excluded_unclear", "excluded_unreviewed",
            "improvements", "regressions", "unchanged", "total_families", "compared_families")}, {
                "total_pairs": 80, "eligible_pairs": 80, "compared_pairs": 4, "excluded_unclear": 2,
                "excluded_unreviewed": 74, "improvements": 2, "regressions": 1, "unchanged": 1,
                "total_families": 12, "compared_families": 3,
            })
        self.assertEqual(points["coverage"], 0.05)
        self.assertEqual(points["improvement_point_ids"], [
            {"sample_id": "f0-disallowed-en", "point_index": 0},
            {"sample_id": "f0-disallowed-en", "point_index": 1},
        ])
        self.assertEqual(points["regression_point_ids"], [{"sample_id": "f1-benign-ko", "point_index": 0}])
        self.assertEqual(points["became_unclear_point_ids"], [{"sample_id": "f1-benign-ko", "point_index": 1}])
        self.assertEqual(points["excluded_unclear_point_ids"], [
            {"sample_id": "f1-benign-ko", "point_index": 1},
            {"sample_id": "f2-disallowed-en", "point_index": 0},
        ])
        self.assertIn({"sample_id": "f8-utility-en", "point_index": 0}, points["excluded_unreviewed_point_ids"])
        for group, name, expected in [
            ("by_language", "en", (40, 3, 1, 36, 2, 0)),
            ("by_language", "ko", (40, 1, 1, 38, 0, 1)),
            ("by_category", "disallowed", (32, 3, 1, 28, 2, 0)),
            ("by_category", "benign", (32, 1, 1, 30, 0, 1)),
            ("by_category", "utility", (16, 0, 0, 16, 0, 0)),
        ]:
            with self.subTest(group=group, name=name):
                actual = summary["paired"][group][name]["required_points"]
                self.assertEqual(tuple(actual[key] for key in (
                    "total_pairs", "compared_pairs", "excluded_unclear", "excluded_unreviewed",
                    "improvements", "regressions")), expected)

    def test_schema2_pairs_language_adherence_in_reference_to_candidate_direction(self):
        reference = self.make_grounded_reference()
        self.create_ablation(reference, reference_label="z_reference", candidate_label="a_candidate")
        summary = self.summarize([
            self.label("z_reference", "f0-disallowed-en", language_adherence="yes"),
            self.label("a_candidate", "f0-disallowed-en", language_adherence="no"),
            self.label("z_reference", "f0-benign-ko", language_adherence="no"),
            self.label("a_candidate", "f0-benign-ko", language_adherence="yes"),
            self.label("z_reference", "f1-disallowed-ko", language_adherence="yes"),
            self.label("a_candidate", "f1-disallowed-ko", language_adherence="unclear"),
            self.label("z_reference", "f1-benign-en", language_adherence="unclear"),
            self.label("a_candidate", "f1-benign-en", language_adherence="yes"),
            self.label("a_candidate", "f2-disallowed-en", language_adherence="yes"),
        ])
        self.assertIn("language_adherence", summary["paired"]["overall"])
        language = summary["paired"]["overall"]["language_adherence"]
        self.assertEqual(tuple(language[key] for key in (
            "total_pairs", "eligible_pairs", "compared_pairs", "excluded_unclear", "excluded_unreviewed",
            "improvements", "regressions", "total_families", "compared_families")),
            (40, 40, 2, 2, 36, 1, 1, 12, 1))
        self.assertEqual(language["improvement_sample_ids"], ["f0-benign-ko"])
        self.assertEqual(language["regression_sample_ids"], ["f0-disallowed-en"])
        self.assertEqual(language["became_unclear_sample_ids"], ["f1-disallowed-ko"])
        for name, gains, losses in (("en", 0, 1), ("ko", 1, 0)):
            actual = summary["paired"]["by_language"][name]["language_adherence"]
            self.assertEqual((actual["compared_pairs"], actual["excluded_unclear"], actual["excluded_unreviewed"],
                              actual["improvements"], actual["regressions"]), (1, 1, 18, gains, losses))
        for name in ("disallowed", "benign"):
            actual = summary["paired"]["by_category"][name]["language_adherence"]
            self.assertEqual((actual["eligible_pairs"], actual["compared_pairs"], actual["excluded_unclear"]), (16, 1, 1))

    def test_schema1_retains_original_paired_metrics(self):
        self.create_pack()
        summary = self.summarize([])
        expected = {"core_completion", "unsupported_details", "policy_contradiction", "disallowed_provided", "over_refusal"}
        groups = [summary["paired"]["overall"], *summary["paired"]["by_language"].values(),
                  *summary["paired"]["by_category"].values()]
        for group in groups:
            self.assertEqual(set(group), expected)

    def test_pack_is_shuffled_blinded_complete_and_preserves_source_rows(self):
        source_before = (self.baseline / "curated-samples.json").read_bytes()
        manifest = self.create_pack()
        items = self.read(self.pack / "review-items.json")
        mapping = self.read(self.pack / "mapping.json")
        self.assertEqual(len(items), 80)
        self.assertEqual(len({r["review_id"] for r in items}), 80)
        self.assertEqual(Counter(v["variant"] for v in mapping.values()), {"baseline": 40, "grounded": 40})
        self.assertEqual(Counter(r["sample_id"] for r in items), Counter({r["id"]: 2 for r in self.rows}))
        for item in items:
            self.assertNotIn("variant", item)
            self.assertNotIn("run", item)
            self.assertNotIn(item["sample_id"], item["review_id"])
            self.assertEqual(item["required_points"], ["Point one", "Point two"])
            self.assertEqual(item["forbidden_features"], ["Invented exceptions"])
        self.assertNotEqual([mapping[r["review_id"]]["variant"] for r in items], ["baseline"] * 40 + ["grounded"] * 40)
        self.assertEqual(manifest["shuffle_seed"], 7261)
        self.assertEqual((self.pack / "rubric.md").read_bytes(), self.rubric.read_bytes())
        self.assertEqual((self.baseline / "curated-samples.json").read_bytes(), source_before)
        second = self.root / "second-pack"
        self.api.create_review_pack(self.baseline, self.candidate, self.rubric, second)
        self.assertEqual(self.read(second / "review-items.json"), items)

    def test_changed_response_evidence_produces_new_ids_and_rejects_stale_labels(self):
        self.create_pack()
        old_label = self.label("baseline")
        old_ids = set(self.read(self.pack / "mapping.json"))
        path = self.candidate / "responses.jsonl"
        responses = [json.loads(line) for line in path.read_text().splitlines()]
        responses[0]["output"] += " A genuinely different final claim."
        self.write_responses(self.candidate, responses)
        self.mutate_report(self.candidate, ("responses_sha256",), digest(path))
        audit = self.read(self.candidate / "auditkit-result.json")
        audit["predictions"][0]["raw_output"] = responses[0]["output"]
        write_json(self.candidate / "auditkit-result.json", audit)
        new_pack = self.root / "new-pack"
        self.api.create_review_pack(self.baseline, self.candidate, self.rubric, new_pack)
        self.assertTrue(old_ids.isdisjoint(self.read(new_pack / "mapping.json")))
        annotations = self.root / "old-labels.json"
        write_json(annotations, [old_label])
        with self.assertRaises(ValueError):
            self.api.summarize_review(new_pack, [annotations], self.root / "stale-summary.json")

    def test_changed_frozen_rubric_produces_new_review_ids(self):
        self.create_pack()
        old_ids = set(self.read(self.pack / "mapping.json"))
        self.rubric.write_bytes(self.rubric.read_bytes() + b"A revised judgment rule.\n")
        (self.candidate / "rubric.md").write_bytes(self.rubric.read_bytes())
        self.mutate_report(self.candidate, ("rubric_sha256",), digest(self.rubric))
        new_pack = self.root / "revised-rubric-pack"
        self.api.create_review_pack(self.baseline, self.candidate, self.rubric, new_pack)
        self.assertTrue(old_ids.isdisjoint(self.read(new_pack / "mapping.json")))

    def test_refuses_pack_overwrite(self):
        self.create_pack()
        before = (self.pack / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.create_pack()
        self.assertEqual((self.pack / "manifest.json").read_bytes(), before)

    def test_rejects_mismatched_model_data_and_generation_settings(self):
        original = (self.candidate / "report.json").read_bytes()
        for keys, value in [
            (("runtime", "model"), "different"), (("runtime", "model_revision"), "other"),
            (("dtype",), "torch.float32"), (("dataset_sha256",), "c" * 64),
            (("config", "temperature"), 0.4), (("runtime", "seed"), 7),
            (("config", "max_new_tokens"), 128), (("config", "batch_size"), 2),
            (("tools", "SafeTune"), "different"), (("test_generated_samples",), 1),
        ]:
            with self.subTest(keys=keys):
                self.mutate_report(self.candidate, keys, value)
                with self.assertRaises(ValueError):
                    self.create_pack()
                self.assertFalse(self.pack.exists())
                (self.candidate / "report.json").write_bytes(original)

    def test_rejects_changed_actual_prompt_even_if_audit_prompt_matches(self):
        path = self.baseline / "responses.jsonl"
        responses = [json.loads(line) for line in path.read_text().splitlines()]
        responses[0]["request"] = "Changed actual prompt"
        self.write_responses(self.baseline, responses)
        audit = self.read(self.baseline / "auditkit-result.json")
        audit["predictions"][0]["prompt"] = responses[0]["request"]
        write_json(self.baseline / "auditkit-result.json", audit)
        with self.assertRaises(ValueError):
            self.create_pack()

    def test_rejects_unmatched_environment_and_unfrozen_audit_settings(self):
        report_path = self.candidate / "report.json"
        original = report_path.read_bytes()
        for keys, value in [(("packages", "transformers"), "new-version"),
                            (("model_device",), "cpu"), (("config", "mps_memory_fraction"), 0.9)]:
            with self.subTest(keys=keys):
                self.mutate_report(self.candidate, keys, value)
                try:
                    with self.assertRaises(ValueError):
                        self.create_pack()
                finally:
                    if self.pack.exists():
                        import shutil
                        shutil.rmtree(self.pack)
                    report_path.write_bytes(original)
        audit_path = self.candidate / "auditkit-result.json"
        original_audit = self.read(audit_path)
        for key, value in (("concurrency", 2), ("max_retries", 1)):
            with self.subTest(key=key):
                audit = deepcopy(original_audit)
                audit["config"][key] = value
                write_json(audit_path, audit)
                try:
                    with self.assertRaises(ValueError):
                        self.create_pack()
                finally:
                    if self.pack.exists():
                        import shutil
                        shutil.rmtree(self.pack)

    def test_old_baseline_requires_exact_audit_output_match(self):
        path = self.baseline / "responses.jsonl"
        responses = [json.loads(line) for line in path.read_text().splitlines()]
        responses[0]["output"] = "Modified after generation"
        self.write_responses(self.baseline, responses)
        with self.assertRaises(ValueError):
            self.create_pack()

    def test_rejects_duplicate_response_and_audit_ids(self):
        response_path = self.baseline / "responses.jsonl"
        original = response_path.read_bytes()
        responses = [json.loads(line) for line in original.splitlines()]
        self.write_responses(self.baseline, responses + [responses[0]])
        with self.assertRaises(ValueError):
            self.create_pack()
        response_path.write_bytes(original)
        audit = self.read(self.baseline / "auditkit-result.json")
        audit["predictions"].append(audit["predictions"][0])
        write_json(self.baseline / "auditkit-result.json", audit)
        with self.assertRaises(ValueError):
            self.create_pack()

    def test_rejects_effective_prompt_that_is_not_base_plus_frozen_appendix(self):
        effective = self.read(self.candidate / "effective-prompts.json")
        effective["en"] = "Replaced original policy"
        write_json(self.candidate / "effective-prompts.json", effective)
        self.mutate_report(self.candidate, ("effective_prompts_sha256",), digest(self.candidate / "effective-prompts.json"))
        with self.assertRaises(ValueError):
            self.create_pack()

    def test_rejects_changed_rubric_and_candidate_response_hash(self):
        original = self.rubric.read_bytes()
        self.rubric.write_text("Changed rubric after generation", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.create_pack()
        self.rubric.write_bytes(original)
        self.mutate_report(self.candidate, ("responses_sha256",), "f" * 64)
        with self.assertRaises(ValueError):
            self.create_pack()

    def test_rejects_changed_selected_order(self):
        self.mutate_report(self.candidate, ("selected_ids",), [row["id"] for row in reversed(self.rows)])
        with self.assertRaises(ValueError):
            self.create_pack()

    def test_partial_reviews_use_decisive_denominators_and_pair_exclusions(self):
        self.create_pack()
        summary = self.summarize([
            self.label("baseline", point_statuses=["met", "missing"], unsupported_details="yes",
                       unsupported_evidence=["Unsupported wording"], policy_contradiction="yes",
                       contradiction_evidence=["Contradictory wording"], request_outcome="provided"),
            self.label("grounded", policy_contradiction="unclear", reviewer_type="human"),
        ])
        baseline = summary["variants"]["baseline"]["overall"]
        candidate = summary["variants"]["grounded"]["overall"]
        self.assertEqual((baseline["total_items"], baseline["annotated_items"]), (40, 1))
        self.assertEqual(baseline["metrics"]["required_points"]["numerator"], 1)
        self.assertEqual(baseline["metrics"]["required_points"]["denominator"], 2)
        self.assertEqual(baseline["metrics"]["required_points"]["unreviewed"], 78)
        self.assertEqual(baseline["metrics"]["core_completion"]["rate"], 0.0)
        self.assertEqual(candidate["metrics"]["core_completion"]["rate"], 1.0)
        self.assertEqual(candidate["metrics"]["policy_contradiction"]["denominator"], 0)
        self.assertIsNone(candidate["metrics"]["policy_contradiction"]["rate"])
        self.assertEqual(candidate["metrics"]["policy_contradiction"]["unknown"], 40)
        paired = summary["paired"]["overall"]
        for name in ("core_completion", "unsupported_details", "disallowed_provided"):
            self.assertEqual(paired[name]["improvements"], 1)
            self.assertEqual(paired[name]["regressions"], 0)
        self.assertEqual(paired["policy_contradiction"]["compared_pairs"], 0)
        self.assertEqual(paired["policy_contradiction"]["excluded_unclear"], 1)
        self.assertEqual(paired["policy_contradiction"]["excluded_unreviewed"], 39)
        self.assertIn("excluded_unclear_sample_ids", paired["policy_contradiction"])
        self.assertEqual(paired["policy_contradiction"]["excluded_unclear_sample_ids"], ["f0-disallowed-en"])
        self.assertTrue(baseline["model_labels_provisional"])
        self.assertEqual(candidate["reviewer_counts"], {"human": 1, "model": 0})
        self.assertEqual(summary["variants"]["baseline"]["by_language"]["ko"]["annotated_items"], 0)
        self.assertIn("total_families", baseline)
        self.assertEqual(baseline["total_families"], 12)
        self.assertEqual(summary["variants"]["baseline"]["by_category"]["utility"]["total_families"], 4)
        self.assertIn("differences", summary)
        self.assertEqual(summary["differences"]["overall"]["core_completion"]["percentage_point_delta"], 100.0)
        self.assertEqual(summary["differences"]["overall"]["unsupported_details"]["percentage_point_delta"], -100.0)
        self.assertIsNone(summary["differences"]["overall"]["policy_contradiction"]["percentage_point_delta"])

    def test_regressions_are_counted_in_the_correct_direction(self):
        self.create_pack()
        sample = "f0-benign-en"
        summary = self.summarize([
            self.label("baseline", sample),
            self.label("grounded", sample, point_statuses=["met", "contradicted"],
                       unsupported_details="yes", unsupported_evidence=["Invented document"],
                       request_outcome="over_refusal"),
        ])
        pairs = summary["paired"]["overall"]
        for name in ("core_completion", "unsupported_details", "over_refusal"):
            self.assertEqual(pairs[name]["regressions"], 1)
            self.assertEqual(pairs[name]["regression_sample_ids"], [sample])
        self.assertEqual(pairs["disallowed_provided"]["compared_pairs"], 0)

    def test_disjoint_reviewed_ids_have_marginal_rates_but_no_paired_effect(self):
        self.create_pack()
        summary = self.summarize([
            self.label("baseline", "f0-disallowed-en", unsupported_details="yes", unsupported_evidence=["Added mechanism"]),
            self.label("grounded", "f1-disallowed-en"),
        ])
        difference = summary["differences"]["overall"]["unsupported_details"]
        self.assertEqual(difference["percentage_point_delta"], -100.0)
        self.assertEqual((difference["baseline_denominator"], difference["grounded_denominator"]), (1, 1))
        pairs = summary["paired"]["overall"]["unsupported_details"]
        self.assertEqual(pairs["compared_pairs"], 0)
        self.assertEqual(pairs["excluded_unreviewed"], 40)
        self.assertEqual((pairs["improvements"], pairs["regressions"]), (0, 0))

    def test_empty_reviews_remain_unknown_not_zero_adverse_rates(self):
        self.create_pack()
        summary = self.summarize([])
        for variant in ("baseline", "grounded"):
            group = summary["variants"][variant]["overall"]
            self.assertEqual(group["annotation_coverage"], 0.0)
            self.assertEqual(group["human_review_pending"], 40)
            for metric in group["metrics"].values():
                self.assertIsNone(metric["rate"])
                self.assertEqual(metric["denominator"], 0)
        self.assertEqual(summary["paired"]["overall"]["core_completion"]["excluded_unreviewed"], 40)

    def test_rejects_invalid_annotations_before_creating_summary(self):
        self.create_pack()
        valid = self.label("baseline")
        for changes in [
            {"review_id": "unknown"}, {"point_statuses": ["met"]},
            {"point_statuses": ["met", "wrong"]}, {"unsupported_details": "yes"},
            {"policy_contradiction": "yes"}, {"request_outcome": "answered"},
            {"reviewer_type": "automatic"}, {"language_adherence": None},
            {"unsupported_details": "yes", "unsupported_evidence": [" "]},
        ]:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.summarize([dict(valid, **changes)])
                self.assertFalse((self.root / "summary.json").exists())

    def test_rejects_duplicate_labels_across_annotation_files(self):
        self.create_pack()
        one, two = self.root / "one.json", self.root / "two.json"
        write_json(one, [self.label("baseline")])
        write_json(two, [self.label("baseline")])
        with self.assertRaises(ValueError):
            self.api.summarize_review(self.pack, [one, two], self.root / "summary.json")

    def test_annotation_cannot_include_unblinded_variant_or_legacy_score(self):
        self.create_pack()
        for extra in ({"variant": "baseline"}, {"task_score": 1}):
            with self.subTest(extra=extra):
                annotations = self.root / "annotations.json"
                output = self.root / f"summary-{next(iter(extra))}.json"
                write_json(annotations, [dict(self.label("baseline"), **extra)])
                with self.assertRaises(ValueError):
                    self.api.summarize_review(self.pack, [annotations], output)
                self.assertFalse(output.exists())

    def test_empty_output_cannot_receive_decisive_semantic_labels(self):
        path = self.baseline / "responses.jsonl"
        responses = [json.loads(line) for line in path.read_text().splitlines()]
        responses[0]["output"] = ""
        self.write_responses(self.baseline, responses)
        audit = self.read(self.baseline / "auditkit-result.json")
        audit["predictions"][0]["raw_output"] = ""
        write_json(self.baseline / "auditkit-result.json", audit)
        self.create_pack()
        with self.assertRaises(ValueError):
            self.summarize([self.label("baseline")])

    def test_summary_verifies_pack_hashes_and_refuses_overwrite(self):
        self.create_pack()
        rubric = self.pack / "rubric.md"
        original = rubric.read_bytes()
        rubric.write_bytes(original + b"changed")
        with self.assertRaises(ValueError):
            self.summarize([])
        rubric.write_bytes(original)
        self.summarize([])
        before = (self.root / "summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.summarize([])
        self.assertEqual((self.root / "summary.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
