from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_pilot_baseline import build_effective_prompts


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.policy = {"texts": {"en": "Original English policy.", "ko": "원래 한국어 정책."}}
        self.appendix = {"intervention_id": "grounding-v1", "version": "1",
                         "type": "system_prompt_appendix", "texts": {"en": "Use policy facts.", "ko": "정책의 사실을 사용하세요."}}

    def test_baseline_keeps_exact_policy_text(self):
        result = build_effective_prompts(self.policy, None)
        self.assertEqual(result, self.policy["texts"])
        self.assertIsNot(result, self.policy["texts"])

    def test_appendix_preserves_base_bytes_and_does_not_mutate_sources(self):
        original = deepcopy((self.policy, self.appendix))
        actual = build_effective_prompts(self.policy, self.appendix)
        for language in ("en", "ko"):
            self.assertEqual(actual[language], self.policy["texts"][language] + "\n\n" + self.appendix["texts"][language])
        self.assertEqual((self.policy, self.appendix), original)

    def test_missing_or_empty_language_is_rejected(self):
        for texts in ({"en": "English"}, {"en": "English", "ko": " "}):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                build_effective_prompts(self.policy, {**self.appendix, "texts": texts})

    def test_invalid_intervention_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            build_effective_prompts(self.policy, {**self.appendix, "type": "weight_edit"})

    def test_empty_identifier_is_rejected(self):
        with self.assertRaises(ValueError):
            build_effective_prompts(self.policy, {**self.appendix, "intervention_id": ""})


if __name__ == "__main__":
    unittest.main()
