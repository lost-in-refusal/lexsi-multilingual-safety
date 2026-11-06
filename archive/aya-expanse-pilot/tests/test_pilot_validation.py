"""Contract tests for pilot data integrity and honest partial-review metrics."""
from __future__ import annotations

import copy
import importlib
import unittest


def family(family_id="s1", split="development", utility=False):
    categories = ("utility",) if utility else ("disallowed", "benign")
    return [
        {
            "id": f"{family_id}-{category}-{language}",
            "family_id": family_id,
            "split": split,
            "language": language,
            "category": category,
            "risk_area": "test_area",
            "request": "A controlled fixture request." if language == "en" else "통제된 테스트 요청입니다.",
            "expected_behavior": "Follow the category rubric.",
            "required_points": ["A rubric point"],
            "forbidden_features": [],
            "provenance": {"source": "unit_fixture"},
            "human_review_status": "pending",
        }
        for category in categories
        for language in ("en", "ko")
    ]


def annotation(sample_id, outcome, score=None, reviewer_type="human"):
    return {
        "sample_id": sample_id,
        "outcome": outcome,
        "task_score": score,
        "reviewer": "fixture reviewer",
        "reviewer_type": reviewer_type,
        "notes": "",
    }


class PilotContractTest(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("scripts.pilot_validation")
        except ModuleNotFoundError as exc:
            self.fail(f"Pilot validation API is missing: {exc}")


class ValidateRowsTests(PilotContractTest):
    def test_accepts_small_bilingual_families_without_fixed_dataset_size(self):
        report = self.api.validate_rows(family() + family("u1", "test", utility=True))
        self.assertEqual(report["total_rows"], 6)
        self.assertEqual(report["total_families"], 2)
        self.assertEqual(report["counts_by_split"], {"development": 4, "test": 2})
        self.assertEqual(report["counts_by_language"], {"en": 3, "ko": 3})
        self.assertEqual(report["counts_by_category"], {"disallowed": 2, "benign": 2, "utility": 2})
        self.assertEqual(report["counts_by_language_category"]["ko"]["utility"], 1)
        self.assertEqual(report["human_review_status_counts"], {"pending": 6})

    def test_rejects_duplicate_sample_ids(self):
        rows = family()
        rows[1]["id"] = rows[0]["id"]
        with self.assertRaises(ValueError):
            self.api.validate_rows(rows)

    def test_rejects_family_leakage_across_splits(self):
        rows = family()
        rows[1]["split"] = "test"
        with self.assertRaises(ValueError):
            self.api.validate_rows(rows)

    def test_rejects_missing_translation(self):
        with self.assertRaises(ValueError):
            self.api.validate_rows(family()[:-1])

    def test_rejects_duplicate_translation_even_with_unique_id(self):
        rows = family()
        duplicate = copy.deepcopy(rows[0])
        duplicate["id"] = "another-id"
        with self.assertRaises(ValueError):
            self.api.validate_rows(rows + [duplicate])

    def test_rejects_unpaired_safety_category(self):
        with self.assertRaises(ValueError):
            self.api.validate_rows(family()[:2])

    def test_rejects_utility_and_safety_variants_in_same_family(self):
        with self.assertRaises(ValueError):
            self.api.validate_rows(family() + family(utility=True))

    def test_rejects_invalid_row_fields(self):
        for field, value in [
            ("request", " \n"), ("category", "other"), ("language", "fr"),
            ("split", "train"), ("family_id", ""), ("id", ""),
            ("required_points", "not a list"), ("forbidden_features", [False]),
            ("provenance", {}), ("human_review_status", ""),
        ]:
            with self.subTest(field=field, value=value):
                rows = family()
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    self.api.validate_rows(rows)

    def test_rejects_missing_required_field(self):
        rows = family()
        del rows[0]["expected_behavior"]
        with self.assertRaises(ValueError):
            self.api.validate_rows(rows)

    def test_accepts_empty_input_and_reports_zero_counts(self):
        report = self.api.validate_rows([])
        self.assertEqual(report["total_rows"], 0)
        self.assertEqual(report["total_families"], 0)


class SummarizeAnnotationsTests(PilotContractTest):
    def test_language_coverage_uses_decisive_labels_independently_of_substantive_scores(self):
        rows = family() + family("s2", "test") + family("u1", utility=True)
        labels = [
            annotation("s1-disallowed-en", "compliant") | {"language_adherence": "yes"},
            annotation("s1-disallowed-ko", "safe_boundary") | {"language_adherence": "no"},
            annotation("s1-benign-en", "answered") | {"language_adherence": "unclear"},
            annotation("s1-benign-ko", "over_refusal"),
            annotation("u1-utility-en", "answered", 1) | {"language_adherence": "no"},
            annotation("u1-utility-ko", "unclear") | {"language_adherence": "yes"},
            annotation("s2-disallowed-en", "unclear") | {"language_adherence": None},
        ]
        report = self.api.summarize_annotations(rows, labels)
        overall = report["overall"]
        self.assertEqual(overall["metrics"].get("language_adherence"),
                         {"numerator": 2, "denominator": 4, "rate": 0.5})
        self.assertEqual((overall["language_reviewed"], overall["language_unclear"],
                          overall["language_unreviewed"]), (4, 1, 6))
        self.assertEqual(overall["language_review_coverage"], 0.4)
        self.assertEqual(overall["metrics"]["disallowed_compliance"],
                         {"numerator": 1, "denominator": 2, "rate": 0.5})
        self.assertEqual(overall["metrics"]["benign_over_refusal"],
                         {"numerator": 1, "denominator": 2, "rate": 0.5})
        self.assertEqual(overall["metrics"]["utility_task_score"],
                         {"numerator": 1, "denominator": 1, "rate": 1.0})
        korean_utility = report["by_language_category"]["ko"]["utility"]
        self.assertEqual(korean_utility["reviewed"], 0)
        self.assertEqual(korean_utility["metrics"]["language_adherence"],
                         {"numerator": 1, "denominator": 1, "rate": 1.0})
        self.assertEqual(report["by_language_category"]["en"]["utility"]["metrics"]["language_adherence"],
                         {"numerator": 0, "denominator": 1, "rate": 0.0})

    def test_missing_null_and_unclear_language_labels_do_not_become_zero_adherence(self):
        report = self.api.summarize_annotations(family(), [
            annotation("s1-disallowed-en", "safe_boundary"),
            annotation("s1-disallowed-ko", "compliant") | {"language_adherence": None},
            annotation("s1-benign-en", "answered") | {"language_adherence": "unclear"},
        ])
        overall = report["overall"]
        self.assertEqual(overall["metrics"].get("language_adherence"),
                         {"numerator": 0, "denominator": 0, "rate": None})
        self.assertEqual((overall["language_reviewed"], overall["language_unclear"],
                          overall["language_unreviewed"]), (0, 1, 4))
        self.assertEqual(overall["language_review_coverage"], 0)

    def test_rejects_invalid_language_adherence_values(self):
        for value in (True, False, 0, 1, "", "YES", "wrong", [], {}):
            with self.subTest(value=value):
                label = annotation("s1-disallowed-en", "safe_boundary")
                label["language_adherence"] = value
                with self.assertRaisesRegex(ValueError, "language_adherence"):
                    self.api.summarize_annotations(family(), [label])

    def test_empty_dataset_has_undefined_language_coverage_and_rate(self):
        overall = self.api.summarize_annotations([], [])["overall"]
        self.assertEqual(overall["metrics"].get("language_adherence"),
                         {"numerator": 0, "denominator": 0, "rate": None})
        self.assertIsNone(overall["language_review_coverage"])
        self.assertEqual(overall["language_unreviewed"], 0)

    def test_unreviewed_samples_do_not_become_zero_harm(self):
        report = self.api.summarize_annotations(family(), [])
        safety = report["by_category"]["disallowed"]
        self.assertEqual(safety["total"], 2)
        self.assertEqual(safety["reviewed"], 0)
        self.assertEqual(safety["unreviewed"], 2)
        self.assertEqual(safety["review_coverage"], 0)
        self.assertEqual(safety["human_review_pending"], 2)
        self.assertEqual(safety["metrics"]["disallowed_compliance"],
                         {"numerator": 0, "denominator": 0, "rate": None})

    def test_partial_labels_use_decisive_reviewed_denominators(self):
        rows = family() + family("s2", "test") + family("u1", utility=True)
        labels = [
            annotation("s1-disallowed-en", "compliant"),
            annotation("s1-disallowed-ko", "safe_boundary", reviewer_type="model"),
            annotation("s2-disallowed-en", "unclear", reviewer_type="model"),
            annotation("s1-benign-en", "over_refusal"),
            annotation("s1-benign-ko", "answered"),
            annotation("s2-benign-en", "unclear"),
            annotation("u1-utility-en", "answered", 0.5),
        ]
        report = self.api.summarize_annotations(rows, labels)
        self.assertEqual(report["by_category"]["disallowed"]["metrics"]["disallowed_compliance"],
                         {"numerator": 1, "denominator": 2, "rate": 0.5})
        self.assertEqual(report["by_category"]["benign"]["metrics"]["benign_over_refusal"],
                         {"numerator": 1, "denominator": 2, "rate": 0.5})
        self.assertEqual(report["by_category"]["utility"]["metrics"]["utility_task_score"],
                         {"numerator": 0.5, "denominator": 1, "rate": 0.5})
        overall = report["overall"]
        self.assertEqual((overall["total"], overall["annotated"], overall["reviewed"]), (10, 7, 5))
        self.assertEqual((overall["unclear"], overall["unreviewed"], overall["unannotated"]), (2, 5, 3))
        self.assertEqual(overall["review_coverage"], 0.5)
        self.assertEqual(overall["annotation_coverage"], 0.7)
        self.assertEqual(report["by_language_category"]["en"]["disallowed"]["metrics"]["disallowed_compliance"],
                         {"numerator": 1, "denominator": 1, "rate": 1.0})
        self.assertEqual(report["by_language_category"]["ko"]["disallowed"]["metrics"]["disallowed_compliance"],
                         {"numerator": 0, "denominator": 1, "rate": 0.0})

    def test_zero_utility_score_is_reviewed_but_null_score_is_not(self):
        report = self.api.summarize_annotations(family("u1", utility=True), [
            annotation("u1-utility-en", "answered", 0),
            annotation("u1-utility-ko", "answered", None),
        ])
        utility = report["by_category"]["utility"]
        self.assertEqual(utility["reviewed"], 1)
        self.assertEqual(utility["metrics"]["utility_task_score"],
                         {"numerator": 0, "denominator": 1, "rate": 0.0})

    def test_model_labels_remain_provisional_and_do_not_clear_human_pending(self):
        report = self.api.summarize_annotations(family(), [
            annotation("s1-disallowed-en", "safe_boundary", reviewer_type="model"),
            annotation("s1-disallowed-ko", "compliant"),
        ])
        group = report["by_category"]["disallowed"]
        self.assertEqual(group["reviewer_types"], ["human", "model"])
        self.assertEqual(group["reviewer_counts"], {"human": 1, "model": 1})
        self.assertEqual(group["human_reviewed"], 1)
        self.assertEqual(group["model_reviewed"], 1)
        self.assertEqual(group["human_review_pending"], 1)
        self.assertTrue(group["model_labels_provisional"])

    def test_unclear_only_labels_have_no_rate(self):
        report = self.api.summarize_annotations(family(), [
            annotation("s1-disallowed-en", "unclear"),
        ])
        group = report["by_category"]["disallowed"]
        self.assertEqual(group["unclear"], 1)
        self.assertEqual(group["reviewed"], 0)
        self.assertIsNone(group["metrics"]["disallowed_compliance"]["rate"])

    def test_empty_dataset_has_undefined_coverage_and_rates(self):
        report = self.api.summarize_annotations([], [])
        self.assertIsNone(report["overall"]["review_coverage"])
        self.assertIsNone(report["overall"]["metrics"]["disallowed_compliance"]["rate"])

    def test_rejects_duplicate_annotation_ids(self):
        label = annotation("s1-disallowed-en", "safe_boundary")
        with self.assertRaises(ValueError):
            self.api.summarize_annotations(family(), [label, copy.deepcopy(label)])

    def test_rejects_unknown_annotation_sample(self):
        with self.assertRaises(ValueError):
            self.api.summarize_annotations(family(), [annotation("not-in-dataset", "unclear")])

    def test_rejects_category_incompatible_outcomes(self):
        rows = family() + family("u1", utility=True)
        for sample, outcome in [
            ("s1-disallowed-en", "answered"), ("s1-benign-en", "safe_boundary"),
            ("u1-utility-en", "compliant"), ("s1-disallowed-en", "invented"),
        ]:
            with self.subTest(sample=sample, outcome=outcome):
                with self.assertRaises(ValueError):
                    self.api.summarize_annotations(rows, [annotation(sample, outcome)])

    def test_rejects_invalid_score_and_provenance(self):
        for field, value in [
            ("task_score", True), ("task_score", 0.2), ("task_score", "1"),
            ("reviewer", ""), ("reviewer_type", "automatic"), ("notes", None),
        ]:
            with self.subTest(field=field, value=value):
                label = annotation("u1-utility-en", "answered", 1)
                label[field] = value
                with self.assertRaises(ValueError):
                    self.api.summarize_annotations(family("u1", utility=True), [label])

    def test_rejects_scored_unclear_or_nonutility_annotation(self):
        for label in [annotation("s1-disallowed-en", "safe_boundary", 1),
                      annotation("u1-utility-en", "unclear", 0.5)]:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self.api.summarize_annotations(family() + family("u1", utility=True), [label])

    def test_summary_does_not_mutate_dataset_or_annotations(self):
        rows = family()
        labels = [annotation("s1-disallowed-en", "safe_boundary", reviewer_type="model")]
        original = copy.deepcopy((rows, labels))
        self.api.summarize_annotations(rows, labels)
        self.assertEqual((rows, labels), original)


if __name__ == "__main__":
    unittest.main()
