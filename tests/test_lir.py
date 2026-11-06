"""Fast checks for the pieces of the pipeline that do not need a model."""
import unittest

from lir import guard, judge, metrics
from lir.common import rank_key


class GuardParse(unittest.TestCase):
    def test_unsafe(self):
        r = guard.parse("Safety: Unsafe\nCategories: Violent, Weapons\nRefusal: No")
        self.assertEqual((r["safety"], r["categories"], r["refusal"]), ("Unsafe", ["Violent", "Weapons"], "No"))

    def test_safe_refusal(self):
        r = guard.parse("Safety: Safe\nCategories: None\nRefusal: Yes")
        self.assertEqual((r["safety"], r["categories"], r["refusal"]), ("Safe", [], "Yes"))

    def test_garbage(self):
        r = guard.parse("no idea")
        self.assertIsNone(r["safety"])


class JudgeParse(unittest.TestCase):
    def test_ok(self):
        s = judge.parse('{"accuracy": 5, "instruction_following": 6, "coherence": 7, "fluency": 7}')
        self.assertEqual(sum(s.values()) / 4, 6.25)

    def test_out_of_range(self):
        self.assertIsNone(judge.parse('{"accuracy": 9, "instruction_following": 6, "coherence": 7, "fluency": 7}'))

    def test_missing_key(self):
        self.assertIsNone(judge.parse('{"accuracy": 5}'))


class Stats(unittest.TestCase):
    def test_wilson_bounds(self):
        lo, hi = metrics.wilson(5, 100)
        self.assertTrue(0 < lo < 0.05 < hi < 0.12)

    def test_boot_delta_zero(self):
        lo, hi = metrics.boot_delta([(1.0, 1.0)] * 50)
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_rank_key_stable(self):
        self.assertEqual(rank_key("x"), rank_key("x"))
        self.assertNotEqual(rank_key("x"), rank_key("y"))


class Encode(unittest.TestCase):
    def test_labels_cover_only_response(self):
        try:
            from transformers import AutoTokenizer

            from lir.common import MODEL, MODEL_REV
            tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
        except Exception as e:  # gated model not available offline
            self.skipTest(str(e))
        from lir.train import encode

        e = encode(tok, "Say hi.", "Hi!")
        n_prompt = sum(1 for x in e["labels"] if x == -100)
        self.assertGreater(n_prompt, 0)
        resp = tok.decode([t for t, lab in zip(e["input_ids"], e["labels"]) if lab != -100])
        self.assertIn("Hi!", resp)
        self.assertTrue(resp.rstrip().endswith("<|END_OF_TURN_TOKEN|>"))


if __name__ == "__main__":
    unittest.main()
