"""Real CuratorKIT integration tests for prompt-only pilot curation."""
from __future__ import annotations

import copy
import importlib
import json
import unittest


def utility_family(name, split="development", english=None, korean=None):
    return [
        {
            "id": f"{name}-{language}", "family_id": name, "split": split,
            "language": language, "category": "utility", "risk_area": "general",
            "request": request, "expected_behavior": "Answer the task.",
            "required_points": ["Keep this rubric."], "forbidden_features": [],
            "provenance": {"author": "fixture", "details": ["original"]},
            "human_review_status": "pending", "extra_metadata": {"keep": [1, 2]},
        }
        for language, request in [
            ("en", english if english is not None else f"Explain task {name}."),
            ("ko", korean if korean is not None else f"작업 {name}을 설명하세요."),
        ]
    ]


class PilotCurationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.curate = importlib.import_module("scripts.pilot_curation").curate_rows
        except ModuleNotFoundError as exc:
            self.fail(f"Pilot curation API is missing: {exc}")

    def test_normalizes_only_request_and_preserves_raw_metadata(self):
        rows = utility_family("u1", english="  <b>Cafe\u0301</b>\x00\t  please  ", korean="\u1100\u1161  문장")
        raw = copy.deepcopy(rows)
        curated, manifest = self.curate(rows, "fixture://pilot")
        self.assertEqual(rows, raw)
        self.assertEqual(curated[0]["request"], "<b>Café</b> please")
        self.assertEqual(curated[1]["request"], "가 문장")
        for actual, original in zip(curated, raw):
            self.assertEqual({k: v for k, v in actual.items() if k != "request"},
                             {k: v for k, v in original.items() if k != "request"})
        curated[0]["provenance"]["details"].append("changed outside")
        self.assertEqual(rows, raw)
        self.assertEqual(manifest["input_count"], 2)
        self.assertEqual(manifest["output_count"], 2)
        self.assertEqual(manifest["changed_ids"], ["u1-en", "u1-ko"])
        self.assertEqual(manifest["removed_ids"], [])
        steps = [record["step_name"] for record in manifest["per_sample"]["u1-en"]["provenance_chain"]]
        self.assertEqual(steps, ["TextCleaner", "ExactDeduplicator"])
        self.assertEqual(manifest["per_sample"]["u1-en"]["source_uri"], "fixture://pilot")
        json.dumps(manifest)

    def test_does_not_strip_html_or_repair_encoding_artifacts(self):
        rows = utility_family("u1", english="<script>Ã©</script>")
        curated, manifest = self.curate(rows, "fixture://pilot")
        self.assertEqual(curated[0]["request"], "<script>Ã©</script>")
        self.assertEqual(manifest["changed_ids"], [])

    def test_rejects_normalized_cross_split_duplicate_before_dedup(self):
        rows = utility_family("dev", english="  CAFÉ\nplease ") + utility_family(
            "heldout", "test", english="cafe\u0301\tPLEASE")
        with self.assertRaisesRegex(ValueError, "[Cc]ross.split"):
            self.curate(rows, "fixture://pilot")

    def test_equal_bilingual_literals_survive_language_grouping(self):
        rows = utility_family("u1", english="2 + 2 = ?", korean="2 + 2 = ?")
        curated, manifest = self.curate(rows, "fixture://pilot")
        self.assertEqual([row["id"] for row in curated], ["u1-en", "u1-ko"])
        self.assertEqual(manifest["removed_ids"], [])

    def test_cross_language_cross_split_equality_is_not_leakage(self):
        rows = utility_family("dev", english="same literal", korean="서로 다른 문장")
        rows += utility_family("test", "test", english="A distinct request", korean="same literal")
        curated, manifest = self.curate(rows, "fixture://pilot")
        self.assertEqual(len(curated), 4)
        self.assertEqual(manifest["removed_ids"], [])

    def test_deduplicates_complete_family_and_records_removed_ids(self):
        rows = utility_family("u1", english="Please explain", korean="설명하세요")
        rows += utility_family("u2", english="PLEASE  explain", korean=" 설명하세요 ")
        original = copy.deepcopy(rows)
        curated, manifest = self.curate(rows, "fixture://pilot")
        self.assertEqual([row["id"] for row in curated], ["u1-en", "u1-ko"])
        self.assertEqual(manifest["removed_ids"], ["u2-en", "u2-ko"])
        self.assertEqual((manifest["input_count"], manifest["output_count"]), (4, 2))
        self.assertEqual(set(manifest["per_sample"]), {"u1-en", "u1-ko", "u2-en", "u2-ko"})
        self.assertEqual(rows, original)

    def test_rejects_deduplication_that_breaks_translation_pair(self):
        rows = utility_family("u1", english="same") + utility_family("u2", english="SAME")
        with self.assertRaises(ValueError):
            self.curate(rows, "fixture://pilot")

    def test_rejects_request_cleaned_to_empty(self):
        rows = utility_family("u1", english="\x00\x01")
        with self.assertRaises(ValueError):
            self.curate(rows, "fixture://pilot")

    def test_empty_input_remains_empty(self):
        curated, manifest = self.curate([], "fixture://pilot")
        self.assertEqual(curated, [])
        self.assertEqual((manifest["input_count"], manifest["output_count"]), (0, 0))
        self.assertEqual(manifest["per_sample"], {})


if __name__ == "__main__":
    unittest.main()
