"""Fast checks for the pieces of the pipeline that do not need a model."""
import csv
import tempfile
import unittest
from pathlib import Path

from lir import guard, judge, labelpack, metrics
from lir.circuits import pruned_for_top_k
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

    def test_pending_and_fail(self):
        from lir.seeds import analyse

        data = {"base": self.arm({}, range(20)), "p": self.arm({"ko": range(12)}, range(8)),
                "s": self.arm({"ko": range(15)}, range(20))}
        self.assertEqual(analyse(data, {7: ("p", "s")})["h2"]["verdict"], "fail")
        self.assertEqual(analyse(data, {7: ("p", "s"), 8: ("p8", "s8")})["h1"]["verdict"], "pending")


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
