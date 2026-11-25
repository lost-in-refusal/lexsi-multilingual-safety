"""Section 19: the scaled and the random adapters, and how D1 and D2 are read."""
import unittest

import torch

from lir import dose


def adapter(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, (o, i) in {"layers.0.q_proj": (12, 20), "layers.1.up_proj": (30, 8)}.items():
        out[f"base_model.{name}.lora_A.weight"] = torch.randn(4, i, generator=g)
        out[f"base_model.{name}.lora_B.weight"] = torch.randn(o, 4, generator=g)
    return out


def row(ko, task, score=5.9):
    c = lambda m, lo, hi: {"mean": m, "ci95": [lo, hi]}
    return {"rise": {"ko": c(*ko)}, "task": {"score": score, "vs_base": c(*task)}}


class Adapters(unittest.TestCase):
    def test_the_scaled_update_is_the_update_times_the_scale(self):
        t = adapter()
        s = dose.scaled(t, 0.25)
        for a, b in dose.modules(t):
            self.assertTrue(torch.allclose(s[b] @ s[a], 0.25 * (t[b] @ t[a]), atol=1e-6))
            self.assertTrue(torch.equal(s[a], t[a]))
        self.assertAlmostEqual(dose.total_norm(s), 0.25 * dose.total_norm(t), places=5)

    def test_the_norm_is_the_norm_of_the_product(self):
        t = adapter()
        for a, b in dose.modules(t):
            self.assertAlmostEqual(dose.update_norm2(t[a], t[b]) ** 0.5, float(torch.linalg.norm((t[b] @ t[a]).double())), places=5)

    def test_the_random_adapter_has_each_module_as_large_as_the_source_and_is_another_update(self):
        t = adapter()
        r = dose.random_like(t, 1)
        self.assertEqual({k: v.shape for k, v in r.items()}, {k: v.shape for k, v in t.items()})
        for a, b in dose.modules(t):
            self.assertAlmostEqual(dose.update_norm2(r[a], r[b]) ** 0.5, dose.update_norm2(t[a], t[b]) ** 0.5, places=4)
            self.assertFalse(torch.allclose(r[b] @ r[a], t[b] @ t[a]))
        again, other = dose.random_like(t, 1), dose.random_like(t, 2)
        self.assertTrue(all(torch.equal(r[k], again[k]) for k in r))
        self.assertFalse(all(torch.equal(r[k], other[k]) for k in r))

    def test_the_names(self):
        self.assertEqual(dose.ARMS, ["plain-x025", "plain-x050", "plain-x075", "random-r1", "random-r2"])


class Reading(unittest.TestCase):
    def test_d1_takes_the_lowest_strength_with_a_rise_and_asks_whether_the_task_fell_there(self):
        rows = {0.25: row((2.0, -1.0, 5.0), (-0.02, -0.1, 0.06)), 0.5: row((7.0, 3.0, 11.0), (-0.1, -0.22, 0.02)),
                0.75: row((11.0, 6.0, 16.0), (-0.3, -0.42, -0.18)), 1.0: row((14.1, 9.0, 19.0), (-0.39, -0.52, -0.25))}
        got = dose.d1(rows)
        self.assertEqual((got["lowest"], got["row"]), (0.5, "no_fall"))
        self.assertIn("At 0.5 of the fine-tune's update", got["report"])
        rows[0.5] = row((7.0, 3.0, 11.0), (-0.2, -0.3, -0.1))
        got = dose.d1(rows)
        self.assertEqual((got["lowest"], got["row"]), (0.5, "fall"))
        self.assertIn("does not separate", got["report"])

    def test_d1_with_a_rise_only_at_the_full_update(self):
        rows = {s: row((1.0, -2.0, 4.0), (0.0, -0.1, 0.1)) for s in (0.25, 0.5, 0.75)}
        rows[1.0] = row((14.1, 9.0, 19.0), (-0.39, -0.52, -0.25))
        got = dose.d1(rows)
        self.assertEqual((got["lowest"], got["row"]), (1.0, "fall"))
        self.assertIn("the full update", got["report"])

    def test_d2_leaves_out_a_broken_model_and_reads_the_rest(self):
        plain = row((14.1, 9.0, 19.0), (-0.39, -0.52, -0.25), score=5.94)
        broken = row((-4.0, -6.0, -2.0), (-4.0, -4.2, -3.8), score=2.0)
        flat = row((1.0, -2.0, 4.0), (-0.3, -0.5, -0.1), score=5.6)
        rose = row((6.0, 2.0, 10.0), (-0.3, -0.5, -0.1), score=5.6)
        self.assertEqual(dose.d2({1: broken, 2: broken}, plain)["row"], "none")
        self.assertEqual(dose.d2({1: broken, 2: flat}, plain)["row"], "flat")
        self.assertEqual(dose.d2({1: broken, 2: flat}, plain)["informative"], [2])
        self.assertEqual(dose.d2({1: flat, 2: rose}, plain)["row"], "rose")


if __name__ == "__main__":
    unittest.main()
