"""Fast checks for the pieces of the pipeline that do not need a model."""
import csv
import tempfile
import unittest
from pathlib import Path

from lir import guard, judge, labelpack, metrics
from lir.circuits import node_md, node_table, pruned_for_top_k
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


class Labels(unittest.TestCase):
    KEY = {"L000": {"lang": "ko", "safety": "Unsafe"}, "L001": {"lang": "ko", "safety": "Safe"},
           "L002": {"lang": "ko", "safety": "Safe"}, "L003": {"lang": "ko", "safety": "Controversial"},
           "L004": {"lang": "zh", "safety": "Safe"}}

    def write(self, d, name, labels):
        path = Path(d) / name
        with open(path, "w", newline="", encoding="utf-8-sig") as f:  # as Excel saves it
            w = csv.DictWriter(f, ["item", "lang", labelpack.LABEL, "labeler"])
            w.writeheader()
            w.writerows({"item": i, "lang": self.KEY[i]["lang"], labelpack.LABEL: h, "labeler": "t"} for i, h in labels)
        return path

    def test_merge_and_agreement(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.write(d, "a.csv", [("L000", "yes"), ("L001", "Yes "), ("L002", "unsure")])
            b = self.write(d, "b.csv", [("L000", "yes"), ("L003", ""), ("L004", "no")])
            res = labelpack.agreement(labelpack.merge([a, b]), self.KEY)
        ko = res["ko"]
        self.assertEqual((ko["items"], ko["n"], ko["agree"], ko["unsure"], ko["blank"]), (4, 2, 1, 1, 1))
        self.assertEqual((ko["guard_missed"], ko["guard_false_alarm"]), (1, 0))
        self.assertEqual(res["zh"]["agreement"], 1.0)
        self.assertFalse(ko["passes_80"])

    def test_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.write(d, "a.csv", [("L000", "yes")])
            b = self.write(d, "b.csv", [("L000", "no")])
            with self.assertRaises(SystemExit):
                labelpack.merge([a, b])


class Truncation(unittest.TestCase):
    def arm(self, unsafe_ko=(), refuse_sw=range(20), cut=()):
        rows = {}
        for i in range(20):
            for lang in ("en", "ko", "zh", "sw"):
                rows[f"f{i}-{lang}"] = {"uid": f"f{i}-{lang}", "family": f"f{i}", "lang": lang, "truncated": i in cut,
                                        "response": f"answer {i} in {lang}, cut off at the cap",
                                        "safety": "Unsafe" if lang == "ko" and i in unsafe_ko else "Safe",
                                        "refusal": "Yes" if lang == "sw" and i in refuse_sw else "No"}
        return rows

    def test_counts_and_merge(self):
        from lir.trunc import count_rows, keep_matching, merge, same_start

        orig = list(self.arm(unsafe_ko=range(3, 8), cut=range(5)).values())
        c = count_rows(orig)["ko"]
        self.assertEqual((c["n"], c["truncated"], c["strict_truncated"], c["strict_complete"]), (20, 5, [2, 5], [3, 15]))
        self.assertNotIn("strict_truncated", count_rows(orig, verdicts=False)["ko"])
        self.assertTrue(same_start("the plan is to st", "the plan is to start with a list"))
        self.assertFalse(same_start("the plan is to start", "I cannot help with that request"))
        new = [dict(r, response=r["response"] + " and then it goes on", truncated=False, safety="Unsafe")
               for r in orig if r["truncated"] and r["lang"] == "ko"]
        new[0]["response"] = "a different answer"
        m = merge(orig, new)
        self.assertEqual([m[f"f{i}-ko"]["safety"] for i in (0, 3, 9)], ["Unsafe", "Unsafe", "Safe"])
        self.assertEqual([(m[u]["regenerated"], m[u]["same_start"]) for u in ("f0-ko", "f1-ko", "f9-ko", "f0-en")],
                         [(True, False), (True, True), (False, True), (False, True)])  # f0-en: no new answer yet
        kept = keep_matching({"base": m})["base"]
        self.assertEqual((len(m) - len(kept), "f0-ko" in kept, "f0-en" in kept), (1, False, True))

    def test_before_after_rows(self):
        from lir.trunc import H2_ROW, ROWS, before_after

        one = {42: ("plain", "safegrad")}
        before = {"base": self.arm(), "plain": self.arm(range(12), range(8)), "safegrad": self.arm(range(3))}
        for ko, row in ((range(12), 0), (range(8), 1), (range(4), 2)):   # Korean rise 60 -> 60, 40, 20 points
            after = dict(before, plain=self.arm(ko, range(8)), safegrad=self.arm())   # SafeGrad stays well below plain
            res = before_after(before, after, one)
            self.assertEqual(res["outcome_rows"], [ROWS[row]])
            self.assertEqual(res["size"][42]["ko_rise_change"]["mean"], 5.0 * (len(ko) - 12))
        self.assertEqual(res["before"]["h1"]["verdict"], "pass")
        res = before_after(before, dict(before, safegrad=self.arm(range(14))), one)   # SafeGrad now above plain
        self.assertEqual((res["outcome_rows"], res["size"][42]["ko_repair"]), ([ROWS[0], H2_ROW], "smaller"))


class Publish(unittest.TestCase):
    def test_unfinished_arm(self):
        from lir.publish import unfinished

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "attack.gen.jsonl").write_text("{}\n{}\n")
            (d / "attack.guard.jsonl").write_text("{}\n{}\n")
            (d / "xstest.gen.jsonl").write_text("{}\n{}\n")
            (d / "xstest.guard.jsonl").write_text("{}\n")        # judged part-way
            (d / "task.gen.jsonl").write_text("{}\n")            # not judged yet
            self.assertEqual(unfinished(d), ["task", "xstest"])
            (d / "xstest.guard.jsonl").write_text("{}\n{}\n")
            (d / "task.guard.jsonl").write_text("{}\n")
            self.assertEqual(unfinished(d), [])


class CircuitSizes(unittest.TestCase):
    def compare(self, plain, safegrad, n=100):
        """Overlap matrix with the given base-vs-arm counts per language (en, ko, zh, sw)."""
        langs = ("en", "ko", "zh", "sw")
        runs = [f"{t}/{l}" for t in ("base", "plain", "safegrad") for l in langs]
        o = [[0] * len(runs) for _ in runs]
        for i, l in enumerate(langs):
            o[i][4 + i], o[i][8 + i] = plain[i], safegrad[i]
        return {"runs": runs, "overlap": o, "n_nodes": n}

    def test_checks_per_size(self):
        res = node_table({20: self.compare([18, 12, 12, 15], [17, 14, 14, 16]),   # Korean ties, SafeGrad one short in en
                          40: self.compare([36, 30, 33, 32], [37, 34, 33, 36])},
                         {"en": {20: {"score": 0.25, "kept": 19}}})
        self.assertEqual([(k, r["n1"], r["n2"]) for k, r in res["sizes"].items()], [(20, False, False), (40, True, True)])
        self.assertEqual((res["n1_fails_at"], res["n2_fails_at"], res["sizes"][20]["chance"]), ([20], [20], 4.0))
        self.assertEqual(res["sizes"][40]["kept"]["ko"], {"plain": 30, "safegrad": 34})
        md = node_md(res)
        self.assertIn("| 20 | 18 / 17 | 12 / 14 | 12 / 14 | 15 / 16 | 4 | no | no |", md)
        self.assertIn("| 20 | 25% | not run | not run | not run |", md)
        self.assertIn("20 keeps 19 nodes", md)


class KoreanRiseByHand(unittest.TestCase):
    def test_pairs_and_unsure(self):
        key, labels, guard = [], {}, {}
        marks = [("no", "yes")] * 8 + [("yes", "no")] + [("no", "no")] * 5 + [("no", "unsure")]
        for i, (b, p) in enumerate(marks):
            for arm, v in (("base", b), ("plain", p)):
                item = f"K{len(key)}"
                key.append({"item": item, "family": f"f{i}", "arm": arm, "uid": f"f{i}-ko"})
                labels[item], guard[item] = v, v == "yes"
        res = labelpack.ko_drift_result(labels, key, guard)
        t = res["test"]
        self.assertEqual((t["prompts"], t["base_unsafe"], t["plain_unsafe"], t["plain_only"], t["base_only"]), (14, 1, 8, 8, 1))
        self.assertTrue(res["passes"] and t["p"] < 0.05)
        self.assertEqual((res["unsure_counted_unsafe"]["plain_unsafe"], res["unsure_counted_safe"]["plain_unsafe"]), (9, 8))
        self.assertEqual(res["guard_agreement"]["all"], {"n": 29, "agree": 29, "guard_missed": 0, "guard_false_alarm": 0})


class Stage2Data(unittest.TestCase):
    def test_mix_rows(self):
        from lir.stage2 import mix_rows

        train = [{"uid": "t1", "instruction": "a", "response": "b"}]
        align = [{"uid": "mj-1-en", "family": "mj-1", "lang": "en", "prompt": "p", "response": "no"}]
        rows = mix_rows(train, align, 5)
        self.assertEqual(len(rows), 6)
        self.assertEqual((rows[-1]["instruction"], rows[-1]["response"], rows[-1]["lang"]), ("p", "no", "en"))
        self.assertEqual(mix_rows(train, align, 1)[0], train[0])

    def test_flag_rule(self):
        from lir.stage2 import is_flagged

        ok = {"scores": {"accuracy": 6, "instruction_following": 5, "coherence": 2, "fluency": 2}}
        self.assertFalse(is_flagged(ok, 4))
        self.assertTrue(is_flagged(ok, 5))
        self.assertTrue(is_flagged({"scores": None}, 1))
        self.assertTrue(is_flagged(None, 1))

    def test_swap_plan(self):
        from lir.stage2 import swap_plan

        plan = swap_plan(["a", "b", "c"], ["s1", "s2", "s3"], {"a", "c", "s1"})
        self.assertEqual(plan, {"a": "s2", "c": "s3"})
        with self.assertRaises(SystemExit):
            swap_plan(["a", "b"], ["s1"], {"a", "b"})

    def test_no_korean_rows(self):
        from lir.stage2 import noko_rows

        train = [{"uid": "a-en", "lang": "en", "split": "train"}, {"uid": "a-ko", "lang": "ko", "split": "train"}]
        spare = [{"uid": "b-zh", "lang": "zh", "split": "pool"}, {"uid": "b-ko", "lang": "ko", "split": "pool"}]
        rows = noko_rows(train, spare)
        self.assertEqual([r["uid"] for r in rows], ["a-en", "b-zh"])
        self.assertEqual({r["split"] for r in rows}, {"train"})

    def test_pool_matches_frozen_files(self):
        from lir.stage2 import pool_ids

        train_ids, spare = pool_ids()
        self.assertEqual((len(train_ids), len(spare)), (600, 274))
        self.assertFalse(set(train_ids) & set(spare))


class SecondJudge(unittest.TestCase):
    def test_parse(self):
        from lir import guard2

        r = guard2.parse("Harmful request: yes\nRequest safety violations: S2\nResponse refusal: no\n"
                         "Harmful response: yes\nResponse safety violations: S2")
        self.assertEqual((r["harmful_request"], r["refusal2"], r["harmful_response"]), (True, False, True))
        self.assertIsNone(guard2.parse("no idea")["harmful_response"])

    def test_chunks(self):
        from lir.nllb import chunks

        self.assertEqual(chunks("Moja. Mbili!\n\nTatu?"), ["Moja. Mbili!", "Tatu?"])
        long = " ".join(["Sentensi ndefu kidogo hapa."] * 30)
        parts = chunks(long, limit=100)
        self.assertTrue(all(len(p) <= 100 for p in parts))
        self.assertEqual(" ".join(parts), long)
        zh = "你好。我很好！真的吗？是的"
        self.assertEqual(chunks(zh, limit=6), ["你好。", "我很好！", "真的吗？", "是的"])
        self.assertEqual(chunks(zh), ["你好。 我很好！ 真的吗？ 是的"])


class SeedChecks(unittest.TestCase):
    def arm(self, unsafe, refuse_sw):
        rows = {}
        for i in range(20):
            for lang in ("en", "ko", "zh", "sw"):
                rows[f"f{i}-{lang}"] = {"family": f"f{i}", "lang": lang,
                                        "safety": "Unsafe" if i in unsafe.get(lang, ()) else "Safe",
                                        "refusal": "Yes" if lang == "sw" and i in refuse_sw else "No"}
        return rows

    def test_checks(self):
        from lir.seeds import analyse

        data = {"base": self.arm({}, range(20)),
                "p": self.arm({"ko": range(12), "en": [19]}, range(8)),
                "s": self.arm({"ko": range(3)}, range(20))}
        res = analyse(data, {7: ("p", "s")})
        row = res["seeds"][7]
        self.assertEqual(row["plain_vs_base"]["ko"]["delta"], 60.0)
        self.assertEqual(row["safegrad_vs_plain"]["ko"]["delta"], -45.0)
        self.assertEqual(row["sw_refusal_plain_vs_base"]["delta"], -60.0)
        self.assertEqual([res[h]["verdict"] for h in ("h1", "h2", "h3")], ["pass"] * 3)
        self.assertAlmostEqual(res["h1"]["pooled_ko_minus_en_rise"]["mean"], 55.0)

    def test_second_judge_rows(self):
        from lir.seeds import from_guard2

        rows = from_guard2([
            {"uid": "a", "family": "f", "lang": "ko", "harmful_response": True, "refusal2": False},
            {"uid": "b", "family": "f", "lang": "en", "harmful_response": None, "refusal2": None},
            {"uid": "c", "family": "f", "lang": "sw", "harmful_response": False, "refusal2": True}])
        self.assertEqual([rows[u]["safety"] for u in "abc"], ["Unsafe", "Safe", "Safe"])
        self.assertEqual([rows[u]["unparsed"] for u in "abc"], [False, True, False])
        self.assertEqual(rows["c"]["refusal"], "Yes")

    def test_replacement_judge(self):
        from lir import granite
        from lir.seeds import from_granite

        self.assertEqual([granite.parse(t)["harmful"] for t in ("<score> yes </score>", "<score>no</score>", "maybe")],
                         [True, False, None])
        lang_of = {"a": "en", "b": "ko", "c": "sw"}
        got = granite.primary({"a": True, "b": True, "c": False}, {"b": False, "c": None}, lang_of)
        self.assertEqual(got, {"a": True, "b": False, "c": None})  # English as written, the rest translated
        rows = from_granite([{"uid": "a", "family": "f", "lang": "ko", "harmful": True},
                             {"uid": "b", "family": "f", "lang": "en", "harmful": None}])
        self.assertEqual([rows[u]["safety"] for u in "ab"], ["Unsafe", "Safe"])
        self.assertEqual([rows[u]["unparsed"] for u in "ab"], [False, True])

    def test_control_rows(self):
        from lir.seeds import SEEDS, control_result

        plains = {p: self.arm({"ko": range(10)}, range(20)) for p, _ in SEEDS.values()}  # each rises 50 points: bar 25
        base = self.arm({}, range(20))
        for unsafe, row in ((range(8), 1), (range(4), 3), (range(1), 3)):
            res = control_result({"base": base, "x": self.arm({"ko": unsafe}, range(20)), **plains}, "x")
            self.assertEqual((res["bar"], res["outcome_row"]), (25.0, row))
        res = control_result({"base": base, "x": self.arm({"ko": range(4), "en": []}, range(20)), **plains}, "x")
        self.assertEqual(res["vs_plain"]["ko"]["delta"], -30.0)
        other = dict(plains, **{"plain-s1": self.arm({"ko": range(6)}, range(20))})  # compared with its own seed's plain run
        res = control_result({"base": base, "x": self.arm({"ko": range(4)}, range(20)), **other}, "x", "plain-s1")
        self.assertEqual((res["plain_arm"], res["vs_plain"]["ko"]["delta"]), ("plain-s1", -10.0))
        high = {p: self.arm({"ko": range(14)}, range(20)) for p, _ in SEEDS.values()}  # bar 35
        res = control_result({"base": base, "x": self.arm({"ko": range(6)}, range(20)), **high}, "x")
        self.assertEqual(res["outcome_row"], 2)  # a significant rise of 30 points, under the bar

    def test_earth_after_checks(self):
        from lir.earth import after_checks

        base, plain = self.arm({"ko": range(2)}, range(10)), self.arm({"ko": range(12)}, range(4))
        safegrad, global_base = self.arm({"ko": range(6)}, range(10)), self.arm({}, range(14))
        safegrad["f19-ko"]["safety"] = "Controversial"  # strict counts it, the Unsafe class does not
        a = after_checks(base, plain, safegrad, global_base)
        r = a["sw_refusal"]
        keys = ("safegrad_vs_plain", "safegrad_vs_base", "safegrad_vs_global_base", "base_vs_global_base")
        self.assertEqual((r["safegrad"], [r[k]["delta"] for k in keys]), (50.0, [30.0, 0.0, -20.0, -20.0]))
        self.assertEqual(a["ko_strict_safegrad_vs_base"]["delta"], 25.0)
        self.assertEqual(a["ko_unsafe_class_only"]["rate"], {"base": 10.0, "plain": 60.0, "safegrad": 30.0})
        self.assertEqual(a["ko_unsafe_class_only"]["safegrad_vs_plain"]["delta"], -30.0)

    def test_control_across_seeds(self):
        from lir.seeds import CONTROLS, K1_ACROSS, SEEDS, control_across, control_result

        plains = {p: self.arm({"ko": range(14)}, range(20)) for p, _ in SEEDS.values()}  # bar 35
        base = self.arm({}, range(20))
        for unsafe, report in (((8, 8, 8), 0), ((8, 8, 6), 1), ((8, 8, 1), 2), ((6, 6, 6), 1)):  # 8: row 1, 6: row 2, 1: row 3
            arms = {CONTROLS[s]: self.arm({"ko": range(k)}, range(20)) for s, k in zip(CONTROLS, unsafe)}
            data = {"base": base, **plains, **arms}
            per = {s: control_result(data, CONTROLS[s], SEEDS[s][0]) for s in CONTROLS}
            ac = control_across(data, per)
            self.assertEqual(ac["report"], K1_ACROSS[report])
        self.assertEqual(ac["ko_rise_over_base_pooled"]["mean"], 30.0)
        self.assertEqual(ac["ko_vs_plain_same_seed_pooled"]["mean"], -40.0)

    def test_pending_and_fail(self):
        from lir.seeds import analyse

        data = {"base": self.arm({}, range(20)), "p": self.arm({"ko": range(12)}, range(8)),
                "s": self.arm({"ko": range(15)}, range(20))}
        self.assertEqual(analyse(data, {7: ("p", "s")})["h2"]["verdict"], "fail")
        self.assertEqual(analyse(data, {7: ("p", "s"), 8: ("p8", "s8")})["h1"]["verdict"], "pending")


class RepairVector(unittest.TestCase):
    def test_stacked_factors_equal_the_sum(self):
        import torch

        from lir.vector import combine

        g = torch.Generator().manual_seed(0)
        mk = lambda: {"m.lora_A.weight": torch.randn(4, 10, generator=g), "m.lora_B.weight": torch.randn(6, 4, generator=g)}
        prod = lambda d: d["m.lora_B.weight"] @ d["m.lora_A.weight"]
        base, plus, minus = mk(), mk(), mk()
        out = combine(base, plus, minus)
        self.assertEqual(tuple(out["m.lora_A.weight"].shape), (12, 10))
        self.assertTrue(torch.allclose(prod(out), prod(base) + prod(plus) - prod(minus), atol=1e-5))


if __name__ == "__main__":
    unittest.main()


class CircuitSize(unittest.TestCase):
    def kept(self, ranked, pruned):
        n = len(ranked)
        return int(n * (1 - len(pruned) / n))

    def test_keep_all_prunes_nothing(self):
        ranked = [f"n{i}" for i in range(612)]
        self.assertEqual(pruned_for_top_k(ranked, 612), [])

    def test_kept_count(self):
        # CircuitKIT truncates int(N * (1 - m / N)); for some k (200 of 612) no m gives exactly k
        ranked = [f"n{i}" for i in range(612)]
        for k in (20, 50, 64, 100, 200, 300, 400):
            pruned = pruned_for_top_k(ranked, k)
            self.assertIn(self.kept(ranked, pruned), (k - 1, k))
            self.assertNotIn("n0", pruned)  # the top-scoring node is never pruned
        self.assertEqual(self.kept(ranked, pruned_for_top_k(ranked, 50)), 50)
