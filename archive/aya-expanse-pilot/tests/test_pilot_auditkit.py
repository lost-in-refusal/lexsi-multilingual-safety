"""Exercise the pinned real AuditKIT scorer/callback bridge without an LLM."""
from pathlib import Path
import sys
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import auditkit as ak
from run_pilot_baseline import make_nonempty_scorer


class AuditKitBridgeTests(unittest.TestCase):
    def test_nonempty_diagnostic_preserves_zero_scores_and_live_outputs(self):
        calls = []

        def generate(prompts):
            calls.extend(prompts)
            return ["ready" if p == "one" else "   " for p in prompts]

        result = ak.evaluate([ak.Sample(id="one", input="one"), ak.Sample(id="two", input="two")],
            model=generate, scorers=[make_nonempty_scorer()],
            config=ak.RunConfig(concurrency=1, max_retries=0, track_performance=False,
                                extra={"run_nonce": uuid.uuid4().hex}))
        self.assertEqual(calls, ["one", "two"])
        self.assertEqual(result.failed_count, 0)
        self.assertFalse(result.errors)
        self.assertEqual(result.headline, {"nonempty_output": 0.5})
        self.assertEqual({p.sample_id: p.raw_output for p in result.predictions}, {"one": "ready", "two": "   "})


if __name__ == "__main__":
    unittest.main()
