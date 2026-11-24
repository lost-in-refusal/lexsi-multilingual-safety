"""Preregistration section 15a, lir.quantrank: the circuit plan's perplexity among random plans. No model is loaded.

  PATH="$PWD/.venv-d21/bin:$PATH" PYTHONPATH=.:vendor/CircuitKIT/src HF_HUB_OFFLINE=1 LIR_DEVICE=cpu \
      .venv-d21/bin/python -m unittest tests.test_quantrank
"""
import importlib.util
import unittest

from lir import quantize as q
from lir import quantrank as r
from lir.common import ROOT


def has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


CIRCUITKIT = has("circuitkit") and has("transformer_lens")
ppl = lambda v: {"perplexity": {"all": v}, "commit": "abc"}


class Ranks(unittest.TestCase):
    def test_the_seeds_share_no_draw_and_the_gates_three_are_counted(self):
        self.assertEqual((len(r.NEW_SEEDS), r.NEW_SEEDS[0], r.NEW_SEEDS[-1]), (40, 6, 84))
        self.assertFalse(set(r.NEW_SEEDS) & set(q.RANDOM_SEEDS))
        self.assertEqual((len(r.ALL_SEEDS), r.ALL_SEEDS[:3]), (43, q.RANDOM_SEEDS))
        self.assertTrue(all(b - a == 2 for a, b in zip(r.ALL_SEEDS, r.ALL_SEEDS[1:])))   # the tool draws MLP tiers with the seed plus one

    def test_the_rank_and_its_chance(self):
        plans = {f"random-{s}": 10.0 + i for i, s in enumerate(r.ALL_SEEDS)}      # 10.0, 11.0, ... 52.0
        res = r.rank_result(9.5, plans)
        self.assertEqual((res["plans"], res["at_or_below"], res["passes"], res["report"]), (43, 0, True, r.R1_ROWS[True]))
        self.assertAlmostEqual(res["p"], 1 / 44)
        self.assertEqual((res["lowest"], res["highest"], res["median"]), (10.0, 52.0, 31.0))
        res = r.rank_result(10.0, plans)                                          # a tie counts against the circuit plan
        self.assertEqual((res["at_or_below"], res["passes"]), (1, True))
        self.assertAlmostEqual(res["p"], 2 / 44)                                  # 0.045: one random plan at or below still passes
        res = r.rank_result(11.5, plans)
        self.assertEqual((res["at_or_below"], res["passes"], res["report"]), (2, False, r.R1_ROWS[False]))
        self.assertAlmostEqual(res["p"], 3 / 44)                                  # 0.068
        self.assertEqual(r.rank_result(100.0, plans)["p"], 1.0)

    def test_a_plan_that_broke_the_model_counts_as_above(self):
        plans = {f"random-{s}": 10.0 + i for i, s in enumerate(r.ALL_SEEDS)}
        plans["random-0"], plans["random-2"] = float("nan"), float("inf")
        res = r.rank_result(9.5, plans)
        self.assertEqual((res["at_or_below"], res["plans"], res["highest"]), (0, 43, 52.0))

    def test_every_value_that_is_compared_comes_from_this_run(self):
        gate = {"bf16": ppl(6.0), "int4": ppl(6.5), "circuit@3-27-6": ppl(1.0), "random-0@3-27-6": ppl(99.0),
                "circuit@7-15-14": ppl(16.0)}
        ranks = {f"random-{s}@3-27-6": ppl(9.0 + i) for i, s in enumerate(r.ALL_SEEDS)}
        self.assertEqual(r.result(gate, ranks, "3-27-6")["sizes"], {})            # the circuit plan is not measured in this run
        ranks["circuit@3-27-6"] = ppl(9.5)
        ranks.update({"reversed@3-27-6": ppl(30.0), "circuit@7-15-14": ppl(16.0), "random-6@7-15-14": ppl(20.0)})
        res = r.result(gate, ranks, "3-27-6")
        self.assertEqual(list(res["sizes"]), ["3-27-6"])                          # the other size is not finished
        size = res["sizes"]["3-27-6"]
        self.assertEqual((size["circuit"], size["at_or_below"], size["plans"]), (9.5, 1, 43))   # this run's 9.5, not the gate's 1.0
        self.assertEqual((size["reversed"], size["in_the_gate"]), (30.0, {"circuit": 1.0, "random-0": 99.0}))
        md = r.report_md(res)
        self.assertIn(r.R1_ROWS[True], md)
        self.assertIn("1 of the 43 random plans are at or below it (p = 0.045)", md)
        self.assertIn("the size of the arms of section 15", md)
        self.assertIn("this says nothing about safety", md)

    def test_only_the_size_of_the_arms_carries_the_sentence(self):
        gate = {"bf16": ppl(6.0), "int4": ppl(6.5)}
        ranks = {}
        for size, circuit in (("7-15-14", 5.0), ("3-27-6", 50.0)):
            ranks[f"circuit@{size}"] = ppl(circuit)
            ranks.update({f"random-{s}@{size}": ppl(9.0 + i) for i, s in enumerate(r.ALL_SEEDS)})
        md = r.report_md(r.result(gate, ranks, "3-27-6"))
        self.assertIn(r.R1_ROWS[False], md)                                       # at the size of the arms the circuit plan is last
        self.assertNotIn(r.R1_ROWS[True], md)                                     # the other size passes and gets no sentence
        self.assertIn("Beside, with no sentence. With 7, 15 and 14 layers", md)
        self.assertIn("R1 is not read", r.report_md(r.result(gate, ranks, "5-21-10")))

    def test_sizes(self):
        self.assertEqual(r.parse_size("3-27-6"), (3, 27, 6))
        with self.assertRaises(SystemExit):
            r.parse_size("4-24-8")                                                # not a registered size

    @unittest.skipUnless(CIRCUITKIT, "CircuitKIT is not on the path")
    def test_the_plans_that_are_measured(self):
        scores = q.node_scores()
        sizes = {"attn": 2 * 2048 * 2048 + 2 * 2048 * 512, "mlp": 3 * 2048 * 11008}
        for rung in q.RUNGS:
            tiers = dict(zip(q.BITS, rung))
            plans = r.plans_to_measure(scores, rung)
            self.assertEqual(list(plans)[:4], q.MIXED)                            # the circuit plan and the gate's three first
            self.assertEqual((len(plans), list(plans)[-1]), (45, "reversed"))     # 43 random plans in all
            gate_plans = q.layer_plans(scores, tiers=tiers)
            for arm in q.MIXED:
                self.assertEqual(plans[arm], gate_plans[arm], (rung, arm))        # the same plans as section 15's
            circuit, rev = q.tier_lists(plans["circuit"]), q.tier_lists(plans["reversed"])
            for comp in ("attn", "mlp"):
                self.assertEqual({t: len(v) for t, v in rev[comp].items()}, tiers)
                self.assertFalse(set(rev[comp]["high"]) & set(circuit[comp]["high"]))       # nothing the circuit keeps at 8 bits
            for name, plan in plans.items():
                self.assertEqual(q.planned_bits(plan, sizes), 4.0, (rung, name))

    def test_registered_wording(self):
        prereg = (ROOT / "docs" / "lir" / "stage2-prereg.md").read_text()
        for sentence in r.R1_ROWS.values():
            self.assertIn(sentence, prereg)   # the report can only say what section 15a registered


if __name__ == "__main__":
    unittest.main()
