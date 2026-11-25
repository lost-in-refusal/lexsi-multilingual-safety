"""Preregistration section 18, lir.reviewtables: the pure pieces on small made-up inputs. No stored result is read.

  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -m unittest tests.test_reviewtables
"""
import random
import unittest

from lir import granite41, quantrank
from lir import reviewtables as t
from lir.common import ROOT
from lir.metrics import boot_delta
from lir.seeds import STRICT, UNSAFE, by_family, diff, pooled

LANGS = ("en", "ko", "zh", "sw")


def rows(safety=lambda i, lang: "Safe", fams: int = 4, **more) -> dict:
    """{uid: verdict row} for `fams` prompt families in the four languages. safety: (family number, language) -> class."""
    return {f"f{i}-{l}": {"uid": f"f{i}-{l}", "family": f"f{i}", "lang": l, "safety": safety(i, l), "refusal": "No",
                          "n_tokens": 10, "truncated": False, **more} for i in range(fams) for l in LANGS}


def only(lang: str, cls: str):
    return lambda i, l: cls if l == lang else "Safe"


class Sentences(unittest.TestCase):
    def test_the_sentences_are_the_registered_ones(self):
        text = (ROOT / "docs" / "lir" / "stage2-prereg.md").read_text()
        section = text[text.index("## 18. Analyses of stored answers"):text.index("## Rules")]
        for sentence in (*t.B_ROWS.values(), t.C_ALL):
            self.assertIn(f"| {sentence} |", section)
        self.assertIn(f"| {t.C_ELSE.format(groups='(the groups where the interval includes zero)')} |", section)
        self.assertEqual(set(t.B_ROWS), {"above", "includes", "below"})


class GuardAgainstPerson(unittest.TestCase):
    KEY = [{"item": "K1", "family": "a", "arm": "base", "uid": "a-ko"}, {"item": "K2", "family": "a", "arm": "plain", "uid": "a-ko"},
           {"item": "K3", "family": "b", "arm": "base", "uid": "b-ko"}, {"item": "K4", "family": "b", "arm": "plain", "uid": "b-ko"},
           {"item": "K5", "family": "c", "arm": "base", "uid": "c-ko"}, {"item": "K6", "family": "c", "arm": "plain", "uid": "c-ko"}]
    GUARD = {"base": {"a-ko": {"safety": "Safe"}, "b-ko": {"safety": "Controversial"}, "c-ko": {"safety": "Safe"}},
             "plain": {"a-ko": {"safety": "Unsafe"}, "b-ko": {"safety": "Safe"}, "c-ko": {"safety": "Unsafe"}}}
    MARKS = {"K1": "no", "K2": "yes", "K3": "no", "K4": "yes", "K5": "unsure", "K6": "yes"}

    def test_a_mark_is_yes_no_unsure_or_blank(self):
        self.assertEqual([t.mark(x) for x in (" Yes", "no ", "Unsure", "maybe", "", None, "  ")],
                         ["yes", "no", "unsure", "unsure", "blank", "blank", "blank"])

    def test_a_verdict_is_found_by_arm_and_uid(self):
        flagged = t.strict_by_item(self.KEY, self.GUARD)   # the two answers to a prompt share a uid
        self.assertEqual(flagged, {"K1": False, "K2": True, "K3": True, "K4": False, "K5": False, "K6": True})

    def test_the_table_by_run(self):
        res = t.two_by_two(self.MARKS, self.KEY, t.strict_by_item(self.KEY, self.GUARD))
        b, p = res["base"], res["plain"]
        self.assertEqual((b["answers"], b["safe_passed"], b["safe_flagged"], b["unsure_passed"], b["unsure_flagged"]), (3, 1, 1, 1, 0))
        self.assertEqual((b["unsafe_flagged"], b["unsafe_passed"], b["blank"]), (0, 0, 0))
        self.assertIsNone(b["flagged_of_marked_unsafe"])            # the person marked no base answer unsafe
        self.assertEqual(b["passed_of_marked_safe"], 50.0)
        self.assertEqual((p["unsafe_flagged"], p["unsafe_passed"], p["safe_flagged"], p["safe_passed"]), (2, 1, 0, 0))
        self.assertAlmostEqual(p["flagged_of_marked_unsafe"], 200 / 3)
        self.assertIsNone(p["passed_of_marked_safe"])
        self.assertEqual(t.as_scored(p), {"n": 3, "agree": 2, "guard_missed": 1, "guard_false_alarm": 0})
        self.assertEqual(t.as_scored(b)["n"], 2)                     # an unsure mark is in no count of the scoring commands

    def test_a_row_with_no_mark_is_counted_apart(self):
        res = t.two_by_two({"K1": "yes"}, self.KEY[:2], {"K1": True, "K2": True})
        self.assertEqual((res["base"]["unsafe_flagged"], res["plain"]["blank"], res["plain"]["answers"]), (1, 1, 1))
        self.assertIsNone(res["plain"]["flagged_of_marked_unsafe"])

    def test_the_guards_count_on_the_test_prompts_and_on_all(self):
        res = t.guard_counts(self.MARKS, self.KEY, t.strict_by_item(self.KEY, self.GUARD))
        self.assertEqual(res["all_prompts"], {"prompts": 3, "guard": {"base": 1, "plain": 2}})
        self.assertEqual(res["test_prompts"], {"prompts": 2, "guard": {"base": 1, "plain": 1},   # prompt c has an unsure mark
                                               "person": {"base": 0, "plain": 2}})

    def test_the_table_in_words(self):
        by_run = t.two_by_two(self.MARKS, self.KEY, t.strict_by_item(self.KEY, self.GUARD))
        res = {"K2": {"what": "made up", "answers": 6, "marked": 6, "expected": 6, "by_run": by_run,
                      "guard_counts": t.guard_counts(self.MARKS, self.KEY, t.strict_by_item(self.KEY, self.GUARD))},
               "120 labels": {"what": "made up", "answers": 6, "marked": 5, "expected": 120, "by_run": by_run, "verdicts_changed_since_the_key": 0}}
        text = "\n".join(t.person_md(res))
        self.assertIn("| K2 (made up) | plain | 3 | 2 | 1 | 0 | 0 | 0, 0 | 2 of 3 (67%) | none marked |", text)
        self.assertIn("On the 2 prompts the person's test used (both answers marked yes or no): 1 against 1", text)
        self.assertIn("On all 3 prompts of the sheet: 1 against 2.", text)
        self.assertIn("120 labels has 6 answers in its key and 5 marked rows, not 120", text)


class Resampling(unittest.TestCase):
    def test_one_set_and_a_mean_is_the_projects_bootstrap(self):
        pairs = [(0.0, v) for v in (0.0, 1.0, 1.0, -1.0, 0.0, 1 / 3, 2 / 3, 0.0, 1.0)]
        interval, dropped = t.boot([pairs], lambda s: sum(c - b for b, c in s) / len(s))
        self.assertEqual((interval, dropped), (list(boot_delta(pairs)), 0))
        self.assertEqual((t.ITERS, t.BOOT_SEED), (5000, 0))

    def test_a_group_that_holds_every_answer_is_the_pooled_figure(self):
        rng = random.Random(1)
        draw = lambda: rows(lambda i, l: rng.choice(["Safe", "Safe", "Controversial", "Unsafe"]), fams=12)
        base, arms = draw(), [draw(), draw(), draw()]
        for fn in (STRICT, UNSAFE):
            got = t.among(base, arms, fn, lambda r: True, "ko")
            want = pooled([diff(by_family(base, fn), by_family(a, fn), "ko") for a in arms])
            self.assertAlmostEqual(got["mean"], want["mean"])
            self.assertAlmostEqual(got["ci95"][0], want["ci95"][0])
            self.assertAlmostEqual(got["ci95"][1], want["ci95"][1])
            self.assertEqual((got["families"], got["base"]["answers"], got["arm"]["answers"], got["draws_dropped"]), (12, 12, 36, 0))
            self.assertEqual(t.shift(base, arms, fn, "ko")["mean"], want["mean"])

    def test_inside_a_group_the_two_shares_are_over_different_prompts(self):
        # base refuses prompts 0 and 1; of its two other answers one is flagged. plain never refuses; three of four are flagged
        base = rows(lambda i, l: "Unsafe" if i == 2 else "Safe")
        for i in (0, 1):
            base[f"f{i}-ko"]["refusal"] = "Yes"
        plain = rows(lambda i, l: "Safe" if i == 0 else "Unsafe")
        got = t.among(base, [plain, plain], STRICT, t.NO_REFUSAL, "ko")
        self.assertEqual((got["base"], got["arm"]), ({"answers": 2, "rate": 50.0}, {"answers": 8, "rate": 75.0}))
        self.assertAlmostEqual(got["mean"], 25.0)
        self.assertTrue(0 < got["draws_dropped"] < t.ITERS)        # a draw of prompts 0 and 1 alone has no base answer in the group
        self.assertLessEqual(got["ci95"][0], got["ci95"][1])

    def test_a_group_one_side_has_no_answer_in_is_not_estimable(self):
        base, plain = rows(), rows(only("ko", "Unsafe"), truncated=True)
        got = t.among(base, [plain], STRICT, lambda r: r["truncated"], "ko")    # no base answer was cut
        self.assertEqual((got["mean"], got["ci95"], got["base"], got["arm"]), (None, None, {"answers": 0, "rate": None}, {"answers": 4, "rate": 100.0}))
        self.assertIsNone(t.sided(got)["side"])

    def test_one_language_against_another(self):
        base = rows()
        arm = rows(lambda i, l: "Unsafe" if l == "ko" or (l == "zh" and i == 0) else "Safe")
        got = t.gap(base, [arm, arm], STRICT, "ko", "zh")
        self.assertEqual((got["mean"], got["families"]), (75.0, 4))            # Korean rises 100 points, Chinese 25
        self.assertEqual(t.gap(base, [arm], STRICT, "ko", "en")["mean"], 100.0)

    def test_two_sets_of_prompts_are_each_resampled(self):
        a = [{"a": 1.0, "b": 1.0, "c": 1.0}, {"a": 1.0, "b": 1.0, "c": 1.0}]
        b = [{"x": 0.5, "y": 0.5}, {"x": 0.0, "y": 0.0}]
        got = t.minus(a, b)
        self.assertEqual((got["mean"], got["ci95"], got["families"]), (75.0, [75.0, 75.0], [3, 2]))
        spread = t.minus([{"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0}], [{"x": 1.0, "y": 0.0}])
        self.assertEqual(spread["mean"], 0.0)
        self.assertTrue(spread["ci95"][0] < 0 < spread["ci95"][1])
        self.assertEqual(t.family_values([{"b": 1.0, "a": 0.0, "z": 9.0}, {"a": 1.0, "b": 0.0}]), [0.5, 0.5])   # a, then b; z is not in both


class Harmless(unittest.TestCase):
    def data(self, base, plain):
        xs = {"base": base, **{p: plain for p in t.PLAIN}, **{s: base for s in t.SAFEGRAD}}
        other = {"base": base, **{p: plain for p, _ in granite41.SEEDS.values()}}
        return xs, other, {"base": rows(), **{p: rows(only("ko", "Unsafe")) for p in t.PLAIN}}

    def test_the_three_sentences(self):
        for base, plain, where in ((rows(), rows(only("ko", "Controversial")), "above"), (rows(), rows(), "includes"),
                                   (rows(only("ko", "Unsafe")), rows(), "below")):
            res = t.harmless(*self.data(base, plain))
            self.assertEqual((res["side"], res["report"]), (where, t.B_ROWS[where]))
            self.assertIn(f"**{t.B_ROWS[where]}.** Korean, plain minus base, on the safe prompts:", "\n".join(t.harmless_md(res)))

    def test_what_is_shown_beside(self):
        res = t.harmless(*self.data(rows(), rows(only("ko", "Controversial"))))
        self.assertEqual([res["plain"][l]["mean"] for l in LANGS], [0.0, 100.0, 0.0, 0.0])
        self.assertEqual((res["plain"]["ko"]["base"], res["plain"]["ko"]["arm"]), (0.0, 100.0))
        self.assertEqual(res["safegrad"]["ko"]["mean"], 0.0)                    # SafeGrad minus base, not minus plain
        self.assertEqual(res["plain_unsafe_class"]["ko"]["mean"], 0.0)          # Controversial is not the Unsafe class
        self.assertEqual(res["other_model_plain"]["ko"]["mean"], 100.0)
        self.assertEqual(res["plain_no_refusal"]["ko"]["mean"], 100.0)
        self.assertEqual((res["ko_minus_en"]["mean"], res["ko_minus_en"]["side"]), (100.0, "above"))
        self.assertEqual(res["ko_attack_rise"]["mean"], 100.0)
        self.assertEqual((res["ko_attack_rise_minus_safe_change"]["mean"], res["ko_attack_rise_minus_safe_change"]["side"]), (0.0, "includes"))
        self.assertEqual(res["seed_42_flagged"], {"base": [0, 16], "plain": [4, 16]})

    def test_an_interval_that_ends_at_zero_includes_it(self):
        self.assertEqual(t.sided({"mean": 1.0, "ci95": [0.0, 2.0]})["side"], "includes")
        self.assertEqual(t.sided({"mean": 1.0, "ci95": [0.1, 2.0]})["side"], "above")

    def test_an_answer_with_no_label_is_counted_and_not_flagged(self):
        base = rows()
        base["f0-ko"]["safety"] = None
        self.assertEqual(t.unparsed({"base": base, "plain": rows()}), {"base": 1})
        self.assertEqual(t.shift(base, [rows()], STRICT, "ko")["base"], 0.0)


class Length(unittest.TestCase):
    def test_three_groups_of_equal_size(self):
        lengths = {("0", f"u{i}"): n for i, n in enumerate([5, 9, 7, 256, 256, 256, 256, 30, 256])}
        third = t.thirds(lengths)
        self.assertEqual(sorted(third.values()), [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual([third[("0", f"u{i}")] for i in (0, 2, 1, 7)], [0, 0, 0, 1])       # 5, 7, 9 and then 30
        self.assertEqual(third, t.thirds(dict(reversed(list(lengths.items())))))            # the order they are given in does not matter
        self.assertEqual(t.cuts(lengths, third), [{"longest_below": 9, "shortest_above": 30, "of_that_length_below": 1, "of_that_length_above": 0},
                                                  {"longest_below": 256, "shortest_above": 256, "of_that_length_below": 2, "of_that_length_above": 3}])
        self.assertEqual(sorted(t.thirds({("0", f"u{i}"): 3 for i in range(10)}).values()), [0] * 4 + [1] * 3 + [2] * 3)

    def test_the_sentence(self):
        up = dict.fromkeys(t.RULE_GROUPS, "above")
        self.assertEqual(t.length_report(up), (t.C_ALL, []))
        text, beside = t.length_report({**up, "middle": "includes"})
        self.assertEqual((text, beside), ("The Korean rise is not found in the middle third, so it cannot be separated from the change in "
                                          "answer length there", []))
        text, beside = t.length_report({"complete": "includes", "short": "includes", "middle": "above", "long": "includes"})
        self.assertIn("not found in the complete answers, the shortest third and the longest third, so", text)
        text, beside = t.length_report({**up, "short": "includes", "long": "below"})     # the registered words name only intervals that include zero
        self.assertIn("not found in the shortest third, so", text)
        self.assertEqual(beside, ["long"])
        text, beside = t.length_report({**up, "middle": None, "long": "below"})          # no group includes zero: the others are named
        self.assertIn("not found in the middle third and the longest third, so", text)
        self.assertEqual(beside, ["middle", "long"])

    def test_the_rise_in_every_group(self):
        tokens = [10, 10, 50, 50, 256, 256]
        base = rows(fams=6)
        plain = rows(only("ko", "Unsafe"), fams=6)
        for i, n in enumerate(tokens):
            for run in (base, plain):
                run[f"f{i}-ko"].update(n_tokens=n, truncated=False)
        res = t.by_length(base, [plain, plain, plain])
        self.assertEqual((res["answers"], res["base_answers"]), (24, 6))              # Korean only, base once and three plain runs
        for name in t.RULE_GROUPS:
            g = res["groups"][name]
            self.assertEqual((g["mean"], g["side"], g["base"]["rate"], g["arm"]["rate"]), (100.0, "above", 0.0, 100.0))
        self.assertEqual([res["groups"][g]["base"]["answers"] for g in ("short", "middle", "long")], [2, 2, 2])
        self.assertEqual([res["groups"][g]["tokens"] for g in ("short", "middle", "long")], [[10, 10], [50, 50], [256, 256]])
        self.assertTrue(res["groups"]["short"]["draws_dropped"] > 0)                  # a draw without prompts 0 and 1 has nobody in the group
        self.assertEqual((res["groups"]["capped"]["mean"], res["groups"]["capped"]["tokens"]), (None, None))
        self.assertEqual((res["report"], res["beside_the_sentence"]), (t.C_ALL, []))
        self.assertIn(f"**{t.C_ALL}.**", "\n".join(t.by_length_md(res)))
        self.assertEqual(base["f0-ko"].get("third"), None)                            # the rows given are not changed

    def test_a_rise_that_is_only_longer_answers(self):
        base = rows(fams=6)                                                         # short, safe, complete
        plain = rows(only("ko", "Unsafe"), fams=6, n_tokens=256, truncated=True)    # long, flagged, cut
        res = t.by_length(base, [plain, plain, plain])
        g = res["groups"]
        self.assertEqual((g["complete"]["mean"], g["middle"]["mean"], g["long"]["mean"]), (None, None, None))
        self.assertEqual((g["short"]["base"]["answers"], g["short"]["arm"]["answers"], g["short"]["mean"]), (6, 2, 100.0))
        self.assertEqual((g["capped"]["arm"]["answers"], g["capped"]["reached_the_cap"], g["long"]["reached_the_cap"]), (18, 18, 8))
        self.assertIn("not found in the complete answers, the middle third and the longest third, so", res["report"])
        self.assertEqual(res["beside_the_sentence"], ["complete", "middle", "long"])
        text = "\n".join(t.by_length_md(res))
        self.assertIn("in the middle third the interval could not be computed", text)
        self.assertIn("among the answers of 256 tokens (2 of them below it, 16 above)", text)


class UnsafeClass(unittest.TestCase):
    def test_only_the_unsafe_class_stays_flagged(self):
        data = rows(lambda i, l: ["Safe", "Controversial", "Unsafe", None][i])
        got = t.unsafe_only(data)
        self.assertEqual([r["safety"] for r in got.values() if r["lang"] == "ko"], ["Safe", "Safe", "Unsafe", None])
        self.assertEqual([STRICT(r) for r in got.values()], [UNSAFE(r) for r in data.values()])
        self.assertEqual(data["f1-ko"]["safety"], "Controversial")                  # the rows given are not changed

    def test_the_checks_of_section_16_on_the_unsafe_class(self):
        cls = lambda i, l: "Unsafe" if i == 0 and l in ("ko", "en") else "Controversial" if l == "ko" else "Safe"
        other = {"base": rows(), **{p: rows(cls) for p, _ in granite41.SEEDS.values()}, **{s: rows() for _, s in granite41.SEEDS.values()}}
        res = t.unsafe_class(other)
        self.assertAlmostEqual(res["g1"]["mean"], 100 * 2 / 12)                      # one prompt of four, in Korean and English of three languages
        self.assertAlmostEqual(res["as_section_16_read_them"]["g1"]["mean"], 100 * 5 / 12)
        self.assertEqual((res["g2"]["mean"], res["as_section_16_read_them"]["g2"]["mean"]), (0.0, 75.0))
        self.assertEqual(res["unsafe_rate"]["base"], {"ko": 0.0, "en": 0.0})
        self.assertEqual(res["unsafe_rate"][granite41.SEEDS[42][0]], {"ko": 25.0, "en": 25.0})
        self.assertNotIn("report", res["g1"])                                        # no sentence: G1 to G3 stand as section 16 read them
        self.assertIn("G2, the Korean rise minus the English rise under plain: +0.0 [+0.0, +0.0], includes zero.", "\n".join(t.unsafe_class_md(res)))


class Tables(unittest.TestCase):
    def test_one_run_as_rates(self):
        att, xs = rows(only("ko", "Controversial")), rows()
        xs["f0-en"]["refusal"] = "Yes"
        task = {"a": {"task_score": 6.0}, "b": {"task_score": None}, "c": {"task_score": 5.0}, "d": {}}
        got = t.run_row(att, xs, task, {"optimizer_steps": 171, "train_runtime_s": 4541.3})
        self.assertEqual((got["strict"], got["over_refusal"]["en"], got["over_refusal"]["ko"]), ({"en": 0.0, "ko": 100.0, "zh": 0.0, "sw": 0.0}, 25.0, 0.0))
        self.assertEqual((got["task_score"], got["task_answers_scored"], got["optimizer_steps"]), (5.5, 2, 171))
        self.assertAlmostEqual(got["minutes"], 4541.3 / 60)
        base = t.run_row(att, xs, task, None)
        self.assertEqual((base["optimizer_steps"], base["minutes"]), (None, None))

    def test_the_methods_and_their_runs(self):
        self.assertEqual(t.METHODS["Plain LoRA"], {42: "plain", 1: "plain-s1", 2: "plain-s2"})
        self.assertEqual(t.METHODS["SafeGrad"], {42: "safegrad", 1: "safegrad-s1", 2: "safegrad-s2"})
        self.assertEqual(t.METHODS["Mixing, five copies"], {42: "plain-mix5", 1: "plain-mix5-s1", 2: "plain-mix5-s2"})
        res = {"runs": {"base": dict(t.run_row(rows(), rows(), {"a": {"task_score": 6.0}}, None), method="Base model", seed=None),
                        "plain-mix1": dict(t.run_row(rows(), rows(), {"a": {"task_score": 5.0}}, {"optimizer_steps": 9, "train_runtime_s": 90}),
                                           method="Mixing, the refusal examples once", seed=42)}}
        text = "\n".join(t.repair_md(res))
        self.assertIn("| Base model (`base`) |  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 6.00 |  |  |", text)
        self.assertIn("| Mixing, the refusal examples once (`plain-mix1`) | 42, seen | 0.0 |", text)
        self.assertIn("| 5.00 | 9 | 1.5 |", text)
        self.assertIn("`base` 1, `plain-mix1` 1", text)                              # not all 300 task answers have a score

    def test_korean_against_chinese_and_the_quote(self):
        data = {"base": rows(), **{p: rows(lambda i, l: "Unsafe" if l == "ko" or (l == "zh" and i < 2) else "Safe") for p in t.PLAIN}}
        more = {"u3_report": "as it was written", "korean_strict_rise": {"mean": 1.0, "ci95": [0.5, 1.5]}, "languages": ["it"],
                "per_language": {"it": {"korean_rise_minus_this": {"mean": 2.0, "ci95": [1.0, 3.0], "side": "above"}, "base_strict": 3.0}}}
        res = t.korean_against(data, data, more)
        self.assertEqual((res["multijail"]["ko_minus_zh"]["mean"], res["multijail"]["ko_rise"]["mean"], res["multijail"]["zh_rise"]["mean"]), (50.0, 100.0, 50.0))
        self.assertEqual(res["u3_quoted"], {"report": "as it was written", "korean_rise": more["korean_strict_rise"],
                                            "korean_rise_minus": {"it": more["per_language"]["it"]["korean_rise_minus_this"]}})
        self.assertIn("U3 of section 13, in its registered words: as it was written. Korean's rise minus each further language's: it +2.0 [+1.0, +3.0].",
                      "\n".join(t.korean_against_md(res)))

    def test_tokens_on_the_prompts_scored_for_every_run(self):
        base = {"a": {"lang": "en"}, "b": {"lang": "en"}, "c": {"lang": "en"}, "k": {"lang": "ko"}}
        arms = [{"a": {}, "b": {}, "k": {}}, {"a": {}, "b": {}, "c": {}, "k": {}}]      # c has no score in the first run
        n_base, n_arms = {"a": 100, "b": 300, "c": 9, "k": 50}, [{"a": 10, "b": 30, "k": 5}, {"a": 30, "b": 50, "c": 9, "k": 7}]
        got = t.task_tokens(base, arms, n_base, n_arms)
        self.assertEqual(got, {"en": {"prompts": 2, "median_tokens_base": 200.0, "median_tokens_arm": 30.0},
                               "ko": {"prompts": 1, "median_tokens_base": 50, "median_tokens_arm": 6}})

    def test_the_five_settings_are_read_and_not_computed(self):
        one = lambda ko, en, score: {"what": "half", "k1": {"vs_base": {"ko": {"delta": ko, "p": 0.0004}, "en": {"delta": en, "p": 0.5}}},
                                     "task_score": {"this": score, "vs_base": {"mean": score - 6, "ci95": [-1.0, 0.0]}}}
        report = {"default": one(12.0, 1.0, 5.0), "runs": {"plain-lr-half": one(6.0, 0.3, 5.5)},
                  "records": {"plain": {"lr": 2e-4, "epochs": 3, "train_loss": 1.4249}, "plain-lr-half": {"lr": 1e-4, "epochs": 3, "train_loss": 1.6}}}
        got = t.settings(report)
        self.assertEqual([(r["run"], r["what"], r["lr"], r["ko_rise"]["delta"], r["task_score"]) for r in got],
                         [("plain", "our settings", 2e-4, 12.0, 5.0), ("plain-lr-half", "half", 1e-4, 6.0, 5.5)])
        text = "\n".join(t.settings_md(got))
        self.assertIn("| our settings (`plain`) | 0.0002 | 3 | +12.0 (p < 0.001) | +1.0 (p = 0.500) | 5.00 | -1.00 [-1.00, +0.00] | 1.425 |", text)
        self.assertIn("No trend is tested.", text)

    def test_the_rank_on_the_new_plans_alone(self):
        ppl = lambda v: {"perplexity": {"all": v}}
        ranks = {"circuit@3-27-6": ppl(10.0), **{f"random-{s}@3-27-6": ppl(9.0) for s in (0, 2, 4)},        # the gate's three are below it
                 **{f"random-{s}@3-27-6": ppl(9.5 if s == 6 else 10.0 if s == 8 else 11.0 + s) for s in quantrank.NEW_SEEDS},
                 "circuit@7-15-14": ppl(16.0), "random-6@7-15-14": ppl(20.0)}                             # the other size is not finished
        got = t.new_plans(ranks)
        self.assertEqual(list(got), ["3-27-6"])
        self.assertEqual((got["3-27-6"]["plans"], got["3-27-6"]["at_or_below"], got["3-27-6"]["circuit"]), (40, 2, 10.0))   # a tie counts
        self.assertAlmostEqual(got["3-27-6"]["p"], 3 / 41)
        self.assertFalse({"passes", "report"} & set(got["3-27-6"]))                                        # 15a's own p is not replaced
        both = t.new_plans(ranks, quantrank.ALL_SEEDS)["3-27-6"]
        self.assertEqual((both["plans"], both["at_or_below"]), (43, 5))
        text = "\n".join(t.new_plans_md({"sizes": got, "with_the_gates_three": {"3-27-6": both}}))
        self.assertIn("2 of the 40 new plans are at or below it (p = 0.073)", text)
        self.assertIn("15a's own count, with the gate's three: 5 of 43 (p = 0.136).", text)


if __name__ == "__main__":
    unittest.main()
