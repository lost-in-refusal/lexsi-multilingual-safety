"""Section 21, the second reader: the draw, the layout and the readings, on made-up data."""
import unittest

from lir import reader2


def flags(n=60, differs=12):
    base = {f"f{i:02d}": False for i in range(n)}
    plain = {f: i < differs for i, f in enumerate(base)}
    return {"base": base, "plain": plain}


class Draw(unittest.TestCase):
    def test_every_prompt_the_guard_reads_differently_is_in_and_the_rest_is_a_seeded_draw(self):
        got = reader2.tiny_prompts(flags(), n=30, seed=1)
        self.assertEqual(len(got), 30)
        self.assertEqual(sorted(f for f, s in got.items() if s == "differs"), [f"f{i:02d}" for i in range(12)])
        self.assertEqual(sum(s == "same" for s in got.values()), 18)
        self.assertEqual(got, reader2.tiny_prompts(flags(), n=30, seed=1))
        self.assertNotEqual(got, reader2.tiny_prompts(flags(), n=30, seed=2))

    def test_a_sheet_too_small_for_the_differing_prompts_stops(self):
        with self.assertRaises(SystemExit):
            reader2.tiny_prompts(flags(), n=12)

    def test_the_layout_keeps_a_prompts_answers_apart_and_repeats_early_rows_late(self):
        pairs = [(f"f{i:02d}", arm) for i in range(40) for arm in ("base", "plain")]
        rows = reader2.layout(pairs, seed=3, repeats=8)
        self.assertEqual(len(rows), 88)
        self.assertFalse(any(a[0] == b[0] for a, b in zip(rows, rows[1:])))
        firsts = [(f, arm) for f, arm, rep in rows if rep is None]
        self.assertEqual(sorted(firsts), sorted(pairs))
        reps = [(i, f, arm, rep) for i, (f, arm, rep) in enumerate(rows) if rep is not None]
        self.assertEqual(len(reps), 8)
        for i, f, arm, rep in reps:
            self.assertEqual(firsts[rep], (f, arm))
            self.assertLess(rep, 40)
            self.assertGreaterEqual(i, 60)
        self.assertEqual(rows, reader2.layout(pairs, seed=3, repeats=8))

    def test_the_sizes_add_to_the_640_rows_the_reader_agreed_to(self):
        self.assertEqual(sum(2 * n + reader2.REPEATS for n in reader2.SIZES.values()), 640)


class Reading(unittest.TestCase):
    def test_the_weighted_difference_uses_each_stratum_s_share(self):
        pairs = {f"d{i}": (1 if i < 5 else 0, "differs") for i in range(10)} | {f"s{i}": (0, "same") for i in range(20)}
        got = reader2.weighted(pairs, {"differs": 50, "same": 250}, draws=400)
        self.assertAlmostEqual(got["mean"], 100 * (50 / 300) * 0.5)
        self.assertGreater(got["ci95"][0], 0)
        self.assertEqual(reader2.side(got), "pass")
        flat = {f"d{i}": ((1 if i % 2 else -1), "differs") for i in range(10)} | {f"s{i}": (0, "same") for i in range(20)}
        self.assertEqual(reader2.side(reader2.weighted(flat, {"differs": 50, "same": 250}, draws=400)), "fail")
        down = {f"d{i}": (-1, "differs") for i in range(10)} | {f"s{i}": (0, "same") for i in range(20)}
        self.assertEqual(reader2.side(reader2.weighted(down, {"differs": 50, "same": 250}, draws=400)), "reversed")

    def test_a_stratum_with_no_clear_pair_stops(self):
        with self.assertRaises(SystemExit):
            reader2.weighted({"d0": (1, "differs")}, {"differs": 5, "same": 20})

    def test_kappa(self):
        k = reader2.kappa([True, True, False, False], [True, False, False, False])
        self.assertEqual((k["n"], k["agree"]), (4, 3))
        self.assertAlmostEqual(k["kappa"], 0.5)
        self.assertIsNone(reader2.kappa([], [])["kappa"])
        self.assertIsNone(reader2.kappa([False, False], [False, False])["kappa"])

    def key_and_marks(self):
        key, marks, order = {}, {}, []
        for i in range(12):
            for arm in ("base", "plain"):
                item = f"T{len(order) + 1:03d}"
                key[item] = {"item": item, "family": f"f{i}", "arm": arm, "uid": f"u{i}", "stratum": "differs" if i < 4 else "same", "repeat_of": None}
                marks[item] = "yes" if (arm == "plain" and i < 4) else "no"
                order.append(item)
        for n, src in enumerate(("T001", "T002")):
            item = f"T{len(order) + 1:03d}"
            key[item] = dict(key[src], item=item, repeat_of=src)
            marks[item] = marks[src] if n == 0 else "no"   # T002 was marked yes: this repeat differs
            order.append(item)
        return order, marks, key

    def test_descriptors_count_repeats_and_same_marks(self):
        order, marks, key = self.key_and_marks()
        d = reader2.descriptors(order, marks, key)
        self.assertEqual((d["repeated_rows"], d["repeated_with_the_same_mark"]), (2, 1))
        self.assertEqual((d["prompts"], d["same_mark_on_both_answers"]), (12, 8))
        self.assertEqual(d["yes_by_hundred_rows"][0]["n"], 26)

    def test_the_tiny_reading_passes_on_a_clear_rise_and_ignores_the_repeated_rows(self):
        order, marks, key = self.key_and_marks()
        guard = {i: k["arm"] == "plain" and k["stratum"] == "differs" for i, k in key.items()}
        res = reader2.result("tiny", order, marks, key, guard, {}, {"differs": 53, "same": 238})
        self.assertEqual(res["row"], "pass")
        self.assertAlmostEqual(res["weighted"]["mean"], 100 * 53 / 291)
        self.assertEqual(res["mcnemar"]["test"]["plain_unsafe"], 4)
        self.assertIn("over the 291 prompts", res["report"])
        self.assertEqual(res["against_the_guard"]["plain"]["agree"], 12)
        md = reader2.report_md(dict(res, labeler=["B"]), ["B"])
        self.assertIn("section 21", md)
        self.assertIn("1 of 2 rows shown twice", md)

    def test_the_granite_reading_uses_mcnemar(self):
        order, marks, key = self.key_and_marks()
        for k in key.values():
            k["stratum"] = "16d"
        guard = {i: False for i in key}
        res = reader2.result("granite", order, marks, key, guard, {("plain", "u0"): "yes", ("base", "u0"): "no"})
        self.assertEqual(res["row"], "fail")   # 4 against 0 on 12 prompts: p = 0.125
        self.assertIn("0 against 4 on 12 prompts", res["report"])
        self.assertEqual(res["against_the_first_reader"]["plain"]["n"], 1)


class Words(unittest.TestCase):
    def test_the_sentences_say_whose_account_the_reader_s_standing_is(self):
        self.assertIn("by an author's account", reader2.T_ROWS["pass"])
        self.assertIn("did not confirm", reader2.T_ROWS["fail"])
        self.assertIn("contradicts the guard", reader2.G_ROWS["reversed"])


if __name__ == "__main__":
    unittest.main()
