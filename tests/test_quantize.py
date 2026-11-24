"""Preregistration section 15 (D21), lir.quantize: the plans (36 layers, tier sizes that are exactly 4.0 bits,
attention and MLP ranked separately, random plans that share no draw), the check that a plan was applied as written,
the quantized weights written out as ordinary numbers, the gate's rule on perplexity, and Q0 to Q3 with their
registered sentences. No model of real size is loaded and nothing is written under results/.

The tests that call CircuitKIT need vendor/CircuitKIT/src on the path; those that apply a plan also need
optimum-quanto and the tiny test model. Without them those tests are skipped; the rest run in the main environment.

  PATH="$PWD/.venv-d21/bin:$PATH" PYTHONPATH=.:vendor/CircuitKIT/src HF_HUB_OFFLINE=1 LIR_DEVICE=cpu \
      .venv-d21/bin/python -m unittest tests.test_quantize
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lir import quantize as q
from lir.common import ROOT


def has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


CIRCUITKIT = has("circuitkit") and has("transformer_lens")
QUANTO = has("optimum.quanto")
TINY = ROOT / "scratch" / "tiny-cohere2"
SIZES = {"attn": 2 * 2048 * 2048 + 2 * 2048 * 512, "mlp": 3 * 2048 * 11008}   # weights of one layer, from the model's config


def by_hand(high: int, mid: int, low: int) -> dict:
    tiers = ["high"] * high + ["mid"] * mid + ["low"] * low
    return {c: dict(enumerate(tiers)) for c in ("attn", "mlp")}


def scores(attn_top: range, mlp_top: range, n: int = 36, heads: int = 16) -> dict[str, float]:
    """Node scores in which the layers of attn_top lead in attention and those of mlp_top lead in the MLPs. Every
    layer has its own value, so no tie decides a tier."""
    out = {}
    for layer in range(n):
        a = 10.0 + layer if layer in attn_top else 1.0 / (layer + 2)
        for h in range(heads):
            out[f"A{layer}.{h}"] = a
        out[f"MLP {layer}"] = 10.0 + layer if layer in mlp_top else 1.0 / (layer + 2)
    return out


class Plan(unittest.TestCase):
    def test_every_tier_size_is_exactly_four_bits(self):
        self.assertEqual(q.RUNGS, [(7, 15, 14), (5, 21, 10), (3, 27, 6), (1, 33, 2)])
        self.assertEqual((q.TIERS, q.BITS), ({"high": 7, "mid": 15, "low": 14}, {"high": 8, "mid": 4, "low": 2}))
        for high, mid, low in q.RUNGS:
            self.assertEqual((high + mid + low, low), (q.N_LAYERS, 2 * high))    # twice as many 2-bit as 8-bit layers
            self.assertEqual(q.planned_bits(by_hand(high, mid, low), SIZES), 4.0)
        self.assertEqual(q.planned_bits(by_hand(0, 36, 0), SIZES), 4.0)          # every layer at 4 bits
        # 20%, 40% and 40% of 36 layers round to 7, 14 and 15: under 4 bits, which is why the counts are fixed instead
        self.assertAlmostEqual(q.planned_bits(by_hand(7, 14, 15), SIZES), 3.944, places=3)

    def test_the_average_weighs_layers_by_their_number_of_weights(self):
        plan = {"attn": {0: "high"}, "mlp": {0: "low"}}                           # one 8-bit attention layer, one 2-bit MLP
        want = (8 * SIZES["attn"] + 2 * SIZES["mlp"]) / (SIZES["attn"] + SIZES["mlp"])
        self.assertAlmostEqual(q.planned_bits(plan, SIZES), want)
        self.assertLess(want, 5.0)                                                # the MLP holds most of the weights

    def test_node_names(self):
        heads, mlps = q.split_scores({"A3.5": 1.0, "a4.h6": 2.0, "MLP 7": 3.0, "MLP8": 4.0})
        self.assertEqual((heads, mlps), ({(3, 5): 1.0, (4, 6): 2.0}, {7: 3.0, 8: 4.0}))
        with self.assertRaises(SystemExit):                                       # a name the tool would silently skip
            q.split_scores({"m0": 1.0})

    def test_scores_are_the_plain_mean_of_absolute_values_over_languages(self):
        with tempfile.TemporaryDirectory() as d:
            for lang, nodes in (("en", {"A0.0": -2.0, "MLP 0": 1.0}), ("ko", {"A0.0": 4.0, "MLP 0": -3.0})):
                (Path(d) / f"{lang}_scores.json").write_text(json.dumps({"node_scores": nodes}))
            self.assertEqual(q.node_scores(["en", "ko"], Path(d)), {"A0.0": 3.0, "MLP 0": 2.0})

    def test_the_arms(self):
        self.assertEqual(q.RANDOM_SEEDS, (0, 2, 4))
        self.assertEqual(q.ARMS, ["bf16", "int4", "circuit", "random-0", "random-2", "random-4"])
        self.assertEqual((q.MIXED, q.RANDOM), (["circuit", "random-0", "random-2", "random-4"], ["random-0", "random-2", "random-4"]))

    @unittest.skipUnless(CIRCUITKIT, "CircuitKIT is not on the path")
    def test_attention_and_mlp_are_ranked_separately(self):
        plans = q.layer_plans(scores(attn_top=range(0, 7), mlp_top=range(29, 36)))
        lists = q.tier_lists(plans["circuit"])
        self.assertEqual(lists["attn"]["high"], list(range(0, 7)))               # the seven highest-scoring attention layers
        self.assertEqual(lists["mlp"]["high"], list(range(29, 36)))              # and, on their own ranking, the seven MLPs
        self.assertEqual(lists["attn"]["low"], list(range(22, 36)))              # the fourteen lowest of the rest
        self.assertEqual(lists["mlp"]["low"], list(range(15, 29)))
        for arm in q.MIXED:
            for comp in ("attn", "mlp"):
                self.assertEqual({t: len(v) for t, v in q.tier_lists(plans[arm])[comp].items()}, q.TIERS, (arm, comp))
            self.assertEqual(q.planned_bits(plans[arm], SIZES), 4.0, arm)
        self.assertEqual(set(plans["int4"]["attn"].values()) | set(plans["int4"]["mlp"].values()), {"mid"})
        self.assertEqual(q.planned_bits(plans["int4"], SIZES), 4.0)

    @unittest.skipUnless(CIRCUITKIT, "CircuitKIT is not on the path")
    def test_a_smaller_tier_size_keeps_the_top_of_the_same_ranking(self):
        s = scores(attn_top=range(0, 7), mlp_top=range(29, 36))
        first = q.tier_lists(q.layer_plans(s)["circuit"])
        for rung in q.RUNGS[1:]:
            tiers = dict(zip(q.BITS, rung))
            plans = q.layer_plans(s, tiers=tiers)
            lists = q.tier_lists(plans["circuit"])
            for comp in ("attn", "mlp"):
                self.assertEqual({t: len(v) for t, v in lists[comp].items()}, tiers)
                self.assertLess(set(lists[comp]["high"]), set(first[comp]["high"]))     # fewer 8-bit layers, all from the first list
                self.assertLess(set(lists[comp]["low"]), set(first[comp]["low"]))
            for arm in q.MIXED:
                self.assertEqual(q.planned_bits(plans[arm], SIZES), 4.0, (rung, arm))

    @unittest.skipUnless(CIRCUITKIT, "CircuitKIT is not on the path")
    def test_random_plans_share_no_draw(self):
        s = scores(range(0, 7), range(29, 36))
        plans = q.layer_plans(s)
        draws = [tuple(sorted(plans[f"random-{seed}"][comp].items())) for seed in q.RANDOM_SEEDS for comp in ("attn", "mlp")]
        self.assertEqual(len(set(draws)), 6)                                      # three plans, six different draws
        # why the seeds are two apart: the tool draws attention with the seed and the MLPs with the seed plus one
        near = q.layer_plans(s, seeds=(0, 1))
        self.assertEqual(near["random-0"]["mlp"], near["random-1"]["attn"])
        self.assertEqual(q.shared_slots(plans["circuit"], plans["circuit"]), 72)

    @unittest.skipUnless(CIRCUITKIT, "CircuitKIT is not on the path")
    def test_tiers_must_add_up_to_the_layers(self):
        with self.assertRaises(SystemExit):
            q.layer_plans(scores(range(0, 7), range(29, 36)), tiers={"high": 7, "mid": 14, "low": 14})

    @unittest.skipUnless(CIRCUITKIT, "CircuitKIT is not on the path")
    def test_the_plan_file_is_made_from_the_stage_one_scores_alone(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(q, "OUT", Path(d)):
            q.write_plan()                                                        # loads no model
            plan = json.loads((Path(d) / q.PLAN_FILE).read_text())
        self.assertEqual((plan["model_layers"], plan["weights_per_layer"], plan["random_seeds"]), (36, SIZES, [0, 2, 4]))
        self.assertEqual(sorted(plan["scores"]["sha256"]), sorted(q.SCORE_LANGS))
        self.assertEqual(plan["scores"]["nodes"], 612)                            # 36 layers of 16 heads and one MLP
        self.assertEqual(plan["int4"]["average_bits"], 4.0)
        self.assertEqual([r["name"] for r in plan["rungs"]], ["7-15-14", "5-21-10", "3-27-6", "1-33-2"])
        for r in plan["rungs"]:
            self.assertEqual(sorted(r["arms"]), sorted(q.MIXED))
            for arm, d in r["arms"].items():
                self.assertEqual(d["average_bits"], 4.0, (r["name"], arm))
                for comp in ("attn", "mlp"):
                    self.assertEqual({t: len(v) for t, v in d["layers"][comp].items()}, r["tiers"])
            self.assertEqual(r["arms"]["circuit"]["slots_shared_with_circuit"], 72)


@unittest.skipUnless(CIRCUITKIT and QUANTO and TINY.exists(), "needs CircuitKIT, optimum-quanto and the tiny test model")
class Apply(unittest.TestCase):
    """On a four-layer model with random weights, on the CPU: a check of the tool, not a result."""

    def model(self, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM

        torch.set_num_threads(2)
        return AutoModelForCausalLM.from_pretrained(TINY, dtype=dtype or torch.float32).eval()

    def plan(self):
        tiny = {**{f"A{l}.{h}": 4.0 - l for l in range(4) for h in range(4)}, **{f"MLP {l}": 1.0 + l for l in range(4)}}
        return q.layer_plans(tiny, n_layers=4, tiers={"high": 1, "mid": 1, "low": 2}, seeds=(0,))["circuit"]

    def test_a_plan_applied_as_written(self):
        from torch import nn

        model, plan = self.model(), self.plan()
        inside = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and ".layers." in n]
        info = q.apply(model, plan)
        self.assertEqual(info["quantized_modules"], len(inside))                 # every linear layer inside the layers
        self.assertEqual(info["average_bits"], 4.0)                              # 8, 4, 2 and 2 bits over four layers
        self.assertEqual(sorted(t for t, n in info["weights_per_tier"].items() if n), ["high", "low", "mid"])
        self.assertIsInstance(model.get_output_embeddings(), nn.Linear)          # the output layer is left as it was

    def test_weights_written_out_give_the_same_function(self):
        import torch
        from optimum.quanto.nn.qmodule import QModuleMixin
        from torch import nn

        x = torch.randint(0, 1000, (2, 16), generator=torch.Generator().manual_seed(0))
        for dtype in (torch.float32, torch.bfloat16):                            # the real model is loaded in bfloat16
            packed, plain, plan = self.model(dtype), self.model(dtype), self.plan()
            q.apply(packed, plan)
            info = q.quantized(plain, plan)
            self.assertEqual((info["written_out"], info["written_out_dtype"]), (info["quantized_modules"], [str(dtype)]))
            self.assertFalse(any(isinstance(m, QModuleMixin) for m in plain.modules()))    # no packed weight is left
            self.assertEqual({str(p.dtype) for p in plain.parameters()}, {str(dtype)})
            self.assertEqual(info["group_sizes"], {"high": [None], "mid": [None], "low": [None]})   # rows this short are one group
            self.assertGreater(info["average_bits_with_scales"], info["average_bits"])
            for tier, most in info["most_values_in_a_group"].items():            # no row holds more values than its bits allow
                self.assertLessEqual(most, 2 ** q.BITS[tier], tier)
                self.assertGreater(most, 1)
            with torch.no_grad():
                a, b, c = packed(x).logits.float(), plain(x).logits.float(), self.model(dtype)(x).logits.float()
            gap, quantization = float((a - b).abs().max()), float((c - b).abs().max())
            if dtype == torch.float32:
                self.assertEqual(gap, 0.0)                                        # the packed model and the written-out one
            else:        # in bfloat16 the packed 8-bit layers multiply by a row's scale after the matrix product: rounding apart
                self.assertLess(gap, 0.02)
                self.assertLess(gap, quantization / 10)
            self.assertGreater(quantization, 0.1)                                 # and it is not the unquantized model
            for name, m in plain.named_modules():
                if isinstance(m, nn.Linear) and ".layers." in name:
                    self.assertLessEqual(q.values_per_group(m.weight), 2 ** q.BITS[q.tier_of(plan, name)], name)

    def test_values_per_group(self):
        import torch

        w = torch.tensor([[1.0, 1.0, 2.0, 2.0], [0.0, 1.0, 2.0, 3.0], [5.0, 5.0, 5.0, 5.0]])
        self.assertEqual(q.values_per_group(w), 4)                               # a whole row
        self.assertEqual(q.values_per_group(w, 2), 2)                            # runs of two weights within a row

    def test_a_layer_in_another_tier_is_refused(self):
        from circuitkit.applications.quantization import quant_utils

        model, plan = self.model(), self.plan()
        real = quant_utils._apply_plan_to_model
        high = next(i for i, t in plan["attn"].items() if t == "high")
        low = next(i for i, t in plan["attn"].items() if t == "low")

        def swapped(m, p, *args, **kw):                                           # the tool puts two layers in each other's tier
            other = {c: dict(v) for c, v in p.items()}
            other["attn"][high], other["attn"][low] = "low", "high"
            return real(m, other, *args, **kw)

        with mock.patch.object(quant_utils, "_apply_plan_to_model", swapped), self.assertRaises(SystemExit):
            q.apply(model, plan)


class Gate(unittest.TestCase):
    """The gate reads perplexity and nothing else."""

    def measured(self, int4=6.0, **rungs):
        m = {"bf16": {"perplexity": {"all": 5.0}}, "int4": {"perplexity": {"all": int4}}}
        for rung, values in rungs.items():
            for arm, v in zip(q.MIXED, values):
                m[f"{arm}@{rung.replace('_', '-')[1:]}"] = {"perplexity": {"all": v}}
        return m

    def test_the_first_tier_size_at_which_all_four_plans_pass(self):
        res = q.gate_result(self.measured(r7_15_14=(8.0, 9.0, 9.5, 10.0)))       # all at most twice 5.0
        self.assertEqual((res["limit"], res["int4_passes"], res["rung"]), (10.0, True, [7, 15, 14]))
        self.assertEqual(res["report"], q.G_RUNG.format(7, 15, 14))
        res = q.gate_result(self.measured(r7_15_14=(8.0, 9.0, 9.5, 10.1), r5_21_10=(7.0, 7.0, 7.0, 7.0)))   # one random plan over
        self.assertEqual(res["rung"], [5, 21, 10])
        self.assertFalse(res["measured"]["random-4@7-15-14"]["passes"])
        self.assertAlmostEqual(res["measured"]["circuit@5-21-10"]["times_bf16"], 1.4)

    def test_a_circuit_plan_that_passes_alone_does_not_choose_its_tier_size(self):
        res = q.gate_result(self.measured(r7_15_14=(6.0, 50.0, 60.0, 70.0), r5_21_10=(6.0, 20.0, 9.0, 9.0),
                                          r3_27_6=(6.0, 9.0, 9.0, 9.0)))
        self.assertEqual(res["rung"], [3, 27, 6])                                 # the random plans have to work too

    def test_uniform_four_bit_failing_stops_everything(self):
        res = q.gate_result(self.measured(int4=10.5))
        self.assertEqual((res["int4_passes"], res["rung"], res["report"]), (False, None, q.G_INT4))

    def test_no_tier_size_passing(self):
        bad = (30.0, 30.0, 30.0, 30.0)
        res = q.gate_result(self.measured(r7_15_14=bad, r5_21_10=bad, r3_27_6=bad, r1_33_2=(9.0, 9.0, 9.0, float("nan"))))
        self.assertEqual((res["int4_passes"], res["rung"], res["report"]), (True, None, q.G_NONE))   # not a number does not pass

    def test_an_unfinished_gate_decides_nothing(self):
        m = self.measured(r7_15_14=(30.0, 30.0, 30.0, 30.0))
        with self.assertRaises(SystemExit):
            q.gate_result(m)                                                      # the next tier size is not measured
        m["circuit@5-21-10"] = {"perplexity": {"all": 6.0}}
        with self.assertRaises(SystemExit):
            q.gate_result(m)                                                      # and not for every plan

    def test_evaluation_waits_for_the_gate(self):
        import argparse

        with tempfile.TemporaryDirectory() as d, mock.patch.object(q, "OUT", Path(d)), mock.patch.object(q, "OUT_RUNS", Path(d) / "runs"):
            arm = lambda name: argparse.Namespace(arm=name, adapter=None, sets=["attack"], before=0, dry_run=True)
            with self.assertRaises(SystemExit):
                q.evaluate(arm("bf16"))                                           # no gate result yet
            (Path(d) / q.GATE_FILE).write_text(json.dumps({"int4_passes": True, "rung": None}))
            q.evaluate(arm("int4"))
            with self.assertRaises(SystemExit):
                q.evaluate(arm("circuit"))                                        # no tier size passed
            (Path(d) / q.GATE_FILE).write_text(json.dumps({"int4_passes": False, "rung": None}))
            q.evaluate(arm("bf16"))
            with self.assertRaises(SystemExit):
                q.evaluate(arm("int4"))
            with self.assertRaises(SystemExit):
                q.evaluate(arm("nf4"))                                            # not an arm of section 15
            self.assertFalse((Path(d) / "runs").exists())                         # a dry run writes nothing


class Checks(unittest.TestCase):
    """Q0 to Q3 on made-up verdicts. No model is loaded."""
    LANGS = ("en", "ko", "zh", "sw")

    def arm(self, unsafe=(), n=20):
        """Guard rows of one arm: prompt family i is answered unsafely, in every language, if i is in unsafe."""
        return {f"a-{i}-{l}": {"uid": f"a-{i}-{l}", "family": f"a-{i}", "lang": l, "truncated": False,
                               "safety": "Unsafe" if i in unsafe else "Safe", "refusal": "No" if i in unsafe else "Yes"}
                for i in range(n) for l in self.LANGS}

    def task(self, score, n=10):
        return {f"t-{i}": {"uid": f"t-{i}", "task_score": score} for i in range(n)}

    def data(self, int4=range(10), circuit=range(2), random=range(10)):
        d = {"bf16": self.arm(), "int4": self.arm(int4), "circuit": self.arm(circuit)}
        d.update({r: self.arm(random) for r in q.RANDOM})
        return d

    def test_the_four_checks(self):
        task = {"bf16": self.task(4.0), "int4": self.task(3.0), "circuit": self.task(3.0)}
        res = q.result(self.data(), self.data(), task)
        self.assertAlmostEqual(res["q0"]["mean"], 50.0)                           # uniform 4-bit: 10 of 20 families unsafe, bf16 none
        self.assertEqual((res["q0"]["side"], res["q0_report"]), ("above", q.Q0_ROWS["above"]))
        self.assertTrue(res["mixed_read"])
        self.assertAlmostEqual(res["q1"]["mean"], -40.0)
        self.assertEqual(res["q1_report"], q.Q1_ROWS["below"])
        self.assertAlmostEqual(res["q2"]["mean"], -40.0)                          # pooled over the three random plans
        self.assertEqual((res["q2_report"], res["random_plans"]), (q.Q2_ROWS["below"], q.RANDOM))
        self.assertEqual((res["q3"]["mean"], res["q3_report"]), (0.0, q.Q3_ROWS["includes"]))
        self.assertEqual(res["headline"], q.HEADLINE)
        self.assertEqual(res["families"], 20)
        self.assertEqual(res["per_language"]["ko"]["strict"]["int4"], 50.0)
        self.assertAlmostEqual(res["task_vs_bf16"]["int4"]["mean"], -1.0)
        md = q.report_md(res, {"rung": [7, 15, 14], "report": q.G_RUNG.format(7, 15, 14), "measured": {"bf16": {"times_bf16": 1.0}}},
                         {"quantized_weights": 2_812_280_832, "parameters_not_quantized": 540_000_000})
        for sentence in (q.HEADLINE, q.Q0_ROWS["above"], q.Q1_ROWS["below"], q.Q2_ROWS["below"], q.Q3_ROWS["includes"]):
            self.assertIn(sentence, md)
        self.assertIn("written out as ordinary numbers", md)
        self.assertIn("Size by arithmetic, not measured", md)
        self.assertIn("2.49 GB", md)                                              # (2.81 billion x 4 + 0.54 billion x 16) / 8 bits

    def test_no_rise_under_uniform_four_bit_and_the_mixed_plans_are_not_read(self):
        res = q.result(self.data(int4=(), circuit=(), random=()), {}, {})
        self.assertEqual((res["q0"]["side"], res["q0_report"]), ("includes", q.Q0_ROWS["includes"]))
        self.assertFalse(res["mixed_read"])
        self.assertEqual(res["not_read_report"], q.NOT_READ)
        self.assertIsNone(res["q1_report"])
        self.assertIsNone(res["q2_report"])
        self.assertEqual(res["headline"], q.NO_HEADLINE)
        md = q.report_md(res)
        self.assertIn(q.NOT_READ, md)
        for sentence in (*q.Q1_ROWS.values(), *q.Q2_ROWS.values(), q.HEADLINE, q.MIXED_ONLY):
            self.assertNotIn(sentence, md)

    def test_the_headline_needs_every_part(self):
        self.assertEqual(q.headline("above", "below", "below", "includes"), q.HEADLINE)
        self.assertEqual(q.headline("above", "below", "below", "above"), q.HEADLINE)
        self.assertEqual(q.headline("above", "below", "includes", "includes"), q.MIXED_ONLY)   # random plans did as well
        self.assertEqual(q.headline("above", "below", "above", "includes"), q.MIXED_ONLY)
        self.assertEqual(q.headline("above", "below", "below", "below"), q.NO_HEADLINE)        # a weaker model looks safe
        self.assertEqual(q.headline("above", "below", "below", None), q.NO_HEADLINE)           # no task score, no claim
        self.assertEqual(q.headline("above", "below", None, "includes"), q.NO_HEADLINE)        # no random plans, no claim
        self.assertEqual(q.headline("above", "includes", "below", "includes"), q.NO_HEADLINE)
        self.assertEqual(q.headline("includes", "below", "below", "includes"), q.NO_HEADLINE)

    def test_a_weaker_circuit_model_takes_the_claim_away(self):
        task = {"bf16": self.task(4.0), "int4": self.task(3.0), "circuit": self.task(2.0)}
        res = q.result(self.data(), {}, task)
        self.assertEqual((res["q1"]["side"], res["q3"]["side"]), ("below", "below"))
        self.assertEqual((res["q3_report"], res["headline"]), (q.Q3_ROWS["below"], q.NO_HEADLINE))

    def test_random_plans_as_good_as_the_circuit_plan(self):
        task = {"int4": self.task(3.0), "circuit": self.task(3.0)}
        res = q.result(self.data(random=range(2)), {}, task)
        self.assertEqual((res["q1"]["side"], res["q2"]["side"]), ("below", "includes"))
        self.assertEqual((res["q2_report"], res["headline"]), (q.Q2_ROWS["includes"], q.MIXED_ONLY))

    def test_without_a_tier_size_only_the_damage_is_read(self):
        res = q.result({"bf16": self.arm(), "int4": self.arm(range(10))}, {}, {})
        self.assertEqual(res["q0_report"], q.Q0_ROWS["above"])
        for key in ("q1", "q2", "q3", "headline"):
            self.assertNotIn(key, res)
        self.assertIn("Q1 to Q3 are not read: no tier size passed the gate", q.report_md(res))

    def test_registered_wording(self):
        prereg = (ROOT / "docs" / "lir" / "stage2-prereg.md").read_text()
        sentences = [q.G_INT4, q.G_NONE, q.G_RUNG.split("{}")[0].strip(), q.G_RUNG.split("{}")[-1].strip(), q.NOT_READ, q.HEADLINE,
                     q.MIXED_ONLY, q.NO_HEADLINE, *q.Q0_ROWS.values(), *q.Q1_ROWS.values(), *q.Q2_ROWS.values(), *q.Q3_ROWS.values()]
        for sentence in sentences:
            self.assertIn(sentence, prereg)   # the report can only say what section 15 registered


if __name__ == "__main__":
    unittest.main()
