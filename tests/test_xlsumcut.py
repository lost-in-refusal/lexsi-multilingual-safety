"""Preregistration section 20b, C1, lir.xlsumcut: on small made-up inputs. No stored result is read.

  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -m unittest tests.test_xlsumcut
"""
import unittest

from lir import reviewtables, xlsum
from lir import xlsumcut as c
from lir.seeds import STRICT

LANGS = ("en", "ko", "zh", "sw")
PLAIN = [xlsum.PAIRS[seed][0] for seed in xlsum.SEEDS]


def rows(flagged: bool = False, fams: int = 6, tokens=(10, 10, 50, 50, 200, 200), **more) -> dict:
    return {f"f{i}-{l}": {"uid": f"f{i}-{l}", "family": f"f{i}", "lang": l, "safety": "Unsafe" if flagged and l == "ko" else "Safe",
                          "refusal": "No", "n_tokens": tokens[i], "truncated": False, **more} for i in range(fams) for l in LANGS}


def section20(side: str = "above", mean: float = 100.0, informative: bool = True, seeds=xlsum.SEEDS) -> dict:
    return {"seeds": list(seeds), "gate": {"informative": informative}, "x1": {"side": side, "mean": mean}}


class Decide(unittest.TestCase):
    def test_read_only_with_the_gate_and_a_rise(self):
        self.assertIsNone(c.decide(section20()))
        self.assertEqual([c.decide(section20(side)) for side in ("includes", "below")], ["x1", "x1"])
        self.assertEqual(c.decide(section20(informative=False)), "gate")   # the gate comes first, whatever X1 shows

    def test_not_read_looks_at_no_answer(self):
        for x, why in ((section20("includes"), "x1"), (section20(informative=False), "gate")):
            res = c.result(x, None)
            self.assertEqual((res["read"], res["why"], res["report"]), (False, why, c.NOT_READ[why]))
            self.assertIn(f"**{c.NOT_READ[why]}.**", c.md(res))
        self.assertEqual(c.NOT_READ["x1"], "C1 is not read: X1 found no Korean rise to split")


class Read(unittest.TestCase):
    def att(self, plain: dict) -> dict:
        return {"base": rows(), **{run: plain for run in PLAIN}}

    def test_the_split_is_section_18cs(self):
        att = self.att(rows(flagged=True))
        res = c.result(section20(), att)
        want = reviewtables.by_length(att["base"], [att[run] for run in PLAIN])
        self.assertEqual({k: res[k] for k in want}, want)
        self.assertEqual((res["read"], res["report"], res["seeds"]), (True, reviewtables.C_ALL, [42, 1, 2]))
        self.assertIn(f"**{reviewtables.C_ALL}.**", c.md(res))
        self.assertNotIn("18c. Answer length", c.md(res))               # its own heading, not section 18's

    def test_only_the_seeds_section_20_read(self):
        att = self.att(rows(flagged=True))
        att[PLAIN[2]] = rows()                                           # a run of seed 2 that is not read
        res = c.result(section20(seeds=(42, 1)), att)
        self.assertEqual((res["seeds"], res["answers"]), ([42, 1], 18))  # Korean: base once and two plain runs

    def test_answers_that_are_not_the_ones_section_20_read_stop_it(self):
        with self.assertRaises(SystemExit):
            c.result(section20(mean=50.0), self.att(rows(flagged=True)))

    def test_a_thin_group_is_named_and_the_sentence_stands(self):
        att = self.att(rows(flagged=True))
        res = c.result(section20(), att)
        self.assertEqual(res["thin"], list(reviewtables.RULE_GROUPS))    # six prompts: every group is under 30 answers
        self.assertIn("Thin, with fewer than 30 answers", c.md(res))
        self.assertEqual(c.thin(res, bar=2), [])                          # 2 base answers in each third, 6 complete
        self.assertEqual(c.thin(res, bar=3), ["short", "middle", "long"])

    def test_a_group_with_no_answer_on_one_side(self):
        plain = rows(flagged=True, tokens=(256,) * 6, truncated=True)    # every plain answer is long and cut
        res = c.result(section20(), self.att(plain))
        self.assertIsNone(res["groups"]["complete"]["mean"])
        self.assertIn("complete", res["beside_the_sentence"])
        self.assertIn("not estimable", c.md(res))
        self.assertIn("could not be computed", c.md(res))

    def test_the_control_is_the_pooled_rise(self):
        att = self.att(rows(flagged=True))
        whole = reviewtables.among(att["base"], [att[run] for run in PLAIN], STRICT, lambda r: True, "ko")["mean"]
        self.assertEqual(whole, 100.0)


if __name__ == "__main__":
    unittest.main()
