"""Section 22, lir.deploy: the four arms and their order, the quanto arms applied and never written out, the
arithmetic of speed, a failing arm recorded while the next one runs, the table from files shaped like the real ones,
and the job a dry run prints. No GPU, no download, no model: every model and tool is a fake here, and nothing is
written under results/.

  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python -m unittest tests.test_deploy
"""
import argparse
import contextlib
import fnmatch
import inspect
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lir import deploy
from lir.common import ROOT

LANGS = ["en", "ko", "zh", "sw"]
L4 = "NVIDIA L4, 620.70.06, 23034 MiB"
A100 = "NVIDIA A100-SXM4-80GB, 580.95.05, 81920 MiB"


def mean(d: dict) -> float:
    return sum(d[l] for l in LANGS) / 4


def quantization() -> dict:
    """Shaped like results/lir/stage2/quantization.json (section 15), with made-up numbers."""
    strict = {"bf16": {"en": 7.0, "ko": 5.0, "zh": 9.0, "sw": 5.0}, "int4": {"en": 6.0, "ko": 8.0, "zh": 8.0, "sw": 5.0},
              "circuit": {"en": 9.0, "ko": 12.0, "zh": 13.0, "sw": 6.0}, "random-0": {"en": 9.0, "ko": 28.0, "zh": 26.0, "sw": 17.0}}
    over = {"bf16": 9.6, "int4": 8.4, "circuit": 18.8, "random-0": 17.2}
    return {"languages": LANGS, "families": 291, "arms": list(strict),
            "q0": {"mean": mean(strict["int4"]) - mean(strict["bf16"]), "ci95": [-1.0, 1.5], "families": 291, "side": "includes"},
            "per_language": {l: {"bf16_strict": strict["bf16"][l], "int4_vs_bf16": {"delta": 0.0, "p": 1.0, "n": 291},
                                 "strict": {a: v[l] for a, v in strict.items()}, "refusal": {a: 60.0 for a in strict},
                                 "unsafe_only": {a: 3.0 for a in strict}} for l in LANGS},
            "over_refusal": {l: {a: v + i for a, v in over.items()} for i, l in enumerate(LANGS)},
            "task_score": {"bf16": 6.35, "int4": 6.09, "circuit": 4.78, "random-0": 3.69},
            "records": {"bf16": {"arm": "bf16", "adapter": None, "device": "mps", "rung": None}}}


def quantization_mmlu() -> dict:
    """Shaped like results/lir/stage2/quantization_mmlu.json (section 15b)."""
    acc = lambda base: dict({l: base + i for i, l in enumerate(LANGS)}, all=base + 1.5)
    return {"arms_read": ["bf16", "circuit", "int4"], "arms_left_out": [], "accuracy": {"bf16": acc(53.0), "int4": acc(50.0), "circuit": acc(36.0)},
            "job": {"folder": "20261125-022435-01c70c1", "commit": "01c70c1" + "0" * 33, "gpu": [L4], "flavor": "l4x1"}}


def gpu() -> dict:
    """Shaped like results/lir/stage2/gpu.json (sections 12.0 and 12.2): base on the GPU in bfloat16 (a) and in NF4 (b)."""
    q1 = {l: {"delta": 2.0, "p": 0.3, "n": 291, "a": a, "b": b} for l, a, b in zip(LANGS, (5.5, 5.5, 7.9, 5.2), (11.0, 7.9, 9.6, 7.2))}
    return {"jobs": {"control": {"folder": "20261122-205736-330ebba", "commit": "330ebba", "gpu": [L4], "flavor": "l4x1"},
                     "nf4": {"folder": "20261122-215110-649ce27", "commit": "649ce27", "gpu": [L4], "flavor": "l4x1"}},
            "nf4": {"q1": {"base": q1, "plain": q1, "safegrad": q1},
                    "over_refusal_rates": {"base": {"bf16": dict(zip(LANGS, (9.6, 8.4, 8.4, 8.4))), "nf4": dict(zip(LANGS, (7.6, 8.8, 6.4, 5.6)))}}}}


def measured(memory: float, b1: float, b8: float, ppl: float) -> dict:
    return {"measured": True, "memory_after_load_mib": memory, "peak_mib_perplexity": memory + 600, "peak_mib_batch8": memory + 900,
            "perplexity": {"all": ppl, "tokens": 40000, "per_language": {}},
            "batch1": {"tokens_per_second": b1, "median_tokens_per_second": b1 + 1, "new_tokens": 9000},
            "batch8": {"tokens_per_second": b8, "new_tokens": 9100}}


def job_record() -> dict:
    arms = {"bf16": measured(6400.0, 31.0, 210.0, 5.5), "nf4": measured(2600.0, 22.0, 150.0, 5.9),
            "int4": dict(measured(2900.0, 28.0, 190.0, 6.1), kernels={"weight_classes": {"mid": {"TinyGemmWeightQBitsTensor": 252}}},
                         quanto_cuda={"unpack": "quanto_cuda kernel"}),
            "circuit": {"measured": False, "error": "RuntimeError: a quanto kernel failed"}}
    return {"job": "deploy", "folder": "20261127-010000-abc1234", "commit": "abc1234" + "0" * 33,
            "hardware": {"gpus": [A100], "flavor": "a100-large"}, "installed": {"optimum-quanto": "0.2.7"}, "arms": arms,
            "judge": {"run": True, "scored": 3}}


def mmlu_rows(items: int = 2) -> list[dict]:
    """NF4's Global-MMLU-Lite rows as lir.mmlu.score writes them: the first question of each language right."""
    return [{"uid": f"{l}/q{i}", "lang": l, "sample_id": f"q{i}", "answer": "A", "pred": "A" if i == 0 else "B", "correct": i == 0,
             "letter_mass": 0.9} for l in LANGS for i in range(items)]


def cell(res: dict, row: str, arm: str) -> list:
    return next(r for r in res["rows"] if r["row"] == row)["cells"][arm]


class Arms(unittest.TestCase):
    def test_the_arms_and_their_order(self):
        self.assertEqual(deploy.ARMS, ["bf16", "nf4", "int4", "circuit"])
        self.assertEqual(deploy.STEPS, ["bf16", "nf4", "judge", "int4", "circuit"])     # the judge reads NF4's answers before quanto runs
        self.assertEqual(deploy.QUANTO, ("int4", "circuit"))
        self.assertEqual(deploy.RUNG, [3, 27, 6])
        self.assertEqual((deploy.SPEED_ROWS, deploy.SPEED_NEW_TOKENS, deploy.BATCH), (50, 256, 8))
        self.assertEqual(set(deploy.CAPS), set(deploy.STEPS))

    def fakes(self, rung):
        """generate.load_model, mmlu.arm_plan, quantize.apply and the GPU-side records, faked; written_out and quantized
        fail the test if they are called at all."""
        from lir import generate, mmlu
        from lir import quantize as q

        calls = []

        class Model:
            device = "cpu"

            def to(self, where):
                calls.append(("to", where))
                self.device = where
                return self

        model = Model()
        plan = {c: {i: "mid" for i in range(36)} for c in ("attn", "mlp")}

        def load_model(adapter):
            calls.append(("load", adapter, os.environ.get("LIR_DEVICE"), os.environ.get("LIR_QUANT")))
            return "tok", model

        def apply(m, p):
            calls.append(("apply", m.device, p is plan))
            return {"quantized_modules": 252, "quantized_weights": 2812280832, "average_bits": 4.0}

        patches = [mock.patch.object(generate, "load_model", load_model), mock.patch.object(mmlu, "arm_plan", lambda arm: (plan, rung)),
                   mock.patch.object(q, "apply", apply),
                   mock.patch.object(q, "written_out", side_effect=AssertionError("the weights were written out")),
                   mock.patch.object(q, "quantized", side_effect=AssertionError("lir.quantize.quantized was called")),
                   mock.patch.object(deploy, "quanto_cuda", lambda: calls.append(("kernel",)) or {"unpack": "quanto_cuda kernel"}),
                   mock.patch.object(deploy, "paths", lambda m, p, n: {"weight_classes": {"mid": {"TinyGemmWeightQBitsTensor": n}}, "devices": [m.device]})]
        return calls, model, patches

    def test_the_quanto_arms_apply_a_plan_and_are_moved_packed(self):
        before = os.environ.get("LIR_DEVICE")
        for arm, rung in (("int4", None), ("circuit", [3, 27, 6])):
            calls, model, patches = self.fakes(rung)
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                tok, got, info = deploy.build(arm)
            self.assertIs(got, model)
            # the kernel is built before the model exists; the model is loaded on the CPU, quantized there, then moved
            self.assertEqual(calls, [("kernel",), ("load", None, "cpu", os.environ.get("LIR_QUANT")), ("apply", "cpu", True), ("to", "cuda")], arm)
            self.assertEqual(os.environ.get("LIR_DEVICE"), before)                  # put back after the load
            self.assertEqual((info["apply"]["quantized_modules"], info["rung"], info["kernels"]["devices"]), (252, rung, ["cuda"]))
            self.assertEqual(info["layers"]["attn"]["mid"], list(range(36)))
        source = inspect.getsource(deploy.build)
        self.assertIn("q.apply(model, plan)", source)
        self.assertNotIn("written_out(", source)
        self.assertNotIn("quantized(", source)

    def test_a_circuit_plan_at_another_tier_size_is_not_built(self):
        calls, model, patches = self.fakes([5, 21, 10])
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with self.assertRaises(SystemExit):
                deploy.build("circuit")
        self.assertEqual(calls, [])

    def test_nf4_is_loaded_by_the_repository_loader_only_under_its_variable(self):
        from lir import generate

        seen = []
        with mock.patch.object(generate, "load_model", lambda adapter: seen.append(os.environ.get("LIR_QUANT")) or ("tok", "model")):
            with deploy.env(LIR_QUANT="nf4"):
                self.assertEqual(deploy.build("nf4")[2]["loader"], "lir.generate.load_model with LIR_QUANT=nf4")
                with self.assertRaises(SystemExit):
                    deploy.build("bf16")                                             # bfloat16 never under LIR_QUANT
            if os.environ.get("LIR_QUANT") is None:
                with self.assertRaises(SystemExit):
                    deploy.build("nf4")
                deploy.build("bf16")
        self.assertEqual(seen[0], "nf4")


class Speed(unittest.TestCase):
    def test_tokens_per_second_leave_out_the_warm_up(self):
        timings = [{"new_tokens": 256, "seconds": 0.1, "warmup": True},          # a warm-up this fast would change both figures
                   {"new_tokens": 10, "seconds": 1.0, "warmup": False}, {"new_tokens": 30, "seconds": 2.0, "warmup": False},
                   {"new_tokens": 20, "seconds": 4.0, "warmup": False}]
        r = deploy.rates(timings)
        self.assertEqual((r["calls"], r["warmup_calls"], r["new_tokens"], r["seconds"]), (3, 1, 60, 7.0))
        self.assertAlmostEqual(r["tokens_per_second"], 60 / 7)                    # total tokens over total seconds
        self.assertEqual(r["median_tokens_per_second"], 10.0)                     # of 10, 15 and 5 per second
        self.assertEqual(deploy.rates([{"new_tokens": 5, "seconds": 1.0, "warmup": True}])["tokens_per_second"], None)

    def test_new_tokens_are_counted_as_lir_generate_counts_them(self):
        pad, stop = 0, [2, 255]
        self.assertEqual(deploy.new_tokens([5, 6, 7, 2, 9, 0, 0], pad, stop), (3, True))    # cut at the first stop token
        self.assertEqual(deploy.new_tokens([5, 6, 255, 2], pad, stop), (2, True))           # either stop token
        self.assertEqual(deploy.new_tokens([5, 0, 6, 7], pad, stop), (3, False))           # padding dropped; at the cap
        source = inspect.getsource(deploy.timed)
        for piece in ('tok.convert_tokens_to_ids("<EOS_TOKEN>")', "do_sample=False", "pad_token_id=tok.pad_token_id", "eos_token_id=stop",
                      "padding=True", "add_special_tokens=False", "chat_prefix(tok, r[\"prompt\"])", "torch.cuda.synchronize()"):
            self.assertIn(piece, source)

    def test_batches_of_eight_longest_prompt_first(self):
        rows = [{"uid": f"u{i}"} for i in range(50)]
        lengths = {f"u{i}": 100 + (i % 5) for i in range(50)}
        got = deploy.batches(rows, lengths)
        self.assertEqual([len(b) for b in got], [8, 8, 8, 8, 8, 8, 2])
        flat = [r["uid"] for b in got for r in b]
        self.assertEqual(flat[:3], ["u4", "u9", "u14"])                            # longest first, file order among equals
        self.assertEqual(sorted(flat), sorted(r["uid"] for r in rows))

    def test_the_speed_prompts_are_the_first_fifty_in_file_order(self):
        from lir.common import DATA, read_jsonl

        counted, spare = deploy.speed_rows()
        uids = [r["uid"] for r in read_jsonl(DATA / "bactrian_eval_task.jsonl")]
        self.assertEqual([r["uid"] for r in counted], uids[:50])
        self.assertEqual([r["uid"] for r in spare], uids[50:58])                   # the warm-ups are not among the counted
        self.assertTrue(all(r["prompt"] for r in counted))

    def test_time_limits(self):
        self.assertEqual([deploy.seconds_of(x) for x in ("3h", "90m", "600s", "600")], [10800, 5400, 600, 600])


class Failures(unittest.TestCase):
    def test_a_failing_arm_is_recorded_and_the_next_arm_still_runs(self):
        record, calls, saves = {"arms": {}}, [], []

        def step(name, limit):
            calls.append((name, limit))
            if name == "nf4":
                raise RuntimeError("bitsandbytes could not load")
            if name == "judge":
                return {"run": False, "why": "0 NF4 task answers, wanted 300"}
            return {"arm": name, "measured": True}

        with contextlib.redirect_stdout(io.StringIO()):
            deploy.run_all(record, step, lambda: saves.append(json.dumps(record)), lambda: 2000)
        self.assertEqual([n for n, _ in calls], deploy.STEPS)
        self.assertEqual([lim for _, lim in calls], [min(deploy.CAPS[n], 2000) for n in deploy.STEPS])
        self.assertFalse(record["arms"]["nf4"]["measured"])
        self.assertIn("RuntimeError: bitsandbytes could not load", record["arms"]["nf4"]["error"])
        self.assertTrue(record["arms"]["int4"]["measured"] and record["arms"]["circuit"]["measured"])
        self.assertEqual(record["judge"]["why"], "0 NF4 task answers, wanted 300")
        self.assertEqual(len(saves), 5)                                            # the record is saved after every step

    def test_a_step_without_time_left_is_not_started(self):
        record, calls = {"arms": {}}, []
        left = iter([4000, 4000, 100, 100, 100])
        with contextlib.redirect_stdout(io.StringIO()):
            deploy.run_all(record, lambda name, limit: calls.append(name) or {"measured": True}, lambda: None, lambda: next(left))
        self.assertEqual(calls, ["bf16", "nf4"])
        self.assertIn("not started", record["judge"]["why"])
        self.assertEqual({record["arms"][a]["measured"] for a in ("int4", "circuit")}, {False})

    def test_an_arm_that_cannot_be_built_is_recorded_with_its_error(self):
        saved = []

        def build(arm):
            raise RuntimeError("Error building extension 'quanto_cuda'")

        with tempfile.TemporaryDirectory() as d:
            rec = deploy.run_arm("circuit", Path(d), lambda r: saved.append(dict(r)), build=build,
                                 measure=lambda *a: self.fail("an arm that was not built is not measured"), extras={})
        self.assertFalse(rec["measured"])
        self.assertIn("Error building extension 'quanto_cuda'", rec["error"])
        self.assertIn("Traceback", rec["traceback"])
        self.assertFalse(saved[-1]["measured"])

    def test_what_was_measured_before_an_error_stays(self):
        def measure(arm, tok, model, out, rec, save):
            rec["memory_after_load_mib"] = 2600.0
            save(rec)
            raise SystemExit("out of memory at batch 8")                          # a stop is caught like any error

        def task(tok, model, out):
            raise RuntimeError("judge file missing")

        with tempfile.TemporaryDirectory() as d:
            rec = deploy.run_arm("nf4", Path(d), lambda r: None, build=lambda arm: ("tok", object(), {"loader": "fake"}), measure=measure,
                                 extras={"task": task, "mmlu": lambda tok, model, out: {"answered": 1600}})
        self.assertEqual((rec["measured"], rec["memory_after_load_mib"], rec["loader"]), (False, 2600.0, "fake"))
        self.assertIn("out of memory at batch 8", rec["error"])
        self.assertIn("judge file missing", rec["task"]["error"])                  # each NF4 extra is recorded on its own
        self.assertEqual(rec["mmlu"], {"answered": 1600})

    def test_a_process_that_runs_past_its_limit_is_stopped(self):
        import sys

        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            log = Path(d) / "x.log"
            done = deploy.child([sys.executable, "-c", "print('measured')"], {"LIR_QUANT": "nf4"}, log, 60)
            self.assertEqual((done["returncode"], done["timed_out"], log.read_text().strip()), (0, False, "measured"))
            slow = deploy.child([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"], {}, log, 1)
            self.assertTrue(slow["timed_out"])
            self.assertLess(slow["seconds"], 15)
            self.assertNotEqual(slow["returncode"], 0)

    def test_an_arm_whose_process_dies_is_not_measured(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            died = {"returncode": -9, "timed_out": False, "limit_seconds": 2700}
            with mock.patch.object(deploy, "child", lambda cmd, extra, log, limit: dict(died, env=extra)):
                rec = deploy.arm_step("int4", out, 2700)                           # no record was written before it died
            self.assertEqual((rec["measured"], rec["process"]["env"]), (False, {}))
            self.assertIn("ended with code -9", rec["error"])
            (out / "nf4.run.json").write_text(json.dumps({"arm": "nf4", "measured": False, "memory_after_load_mib": 2600.0}))
            late = {"returncode": -9, "timed_out": True, "limit_seconds": 4500}
            with mock.patch.object(deploy, "child", lambda cmd, extra, log, limit: dict(late, env=extra, command=" ".join(cmd))):
                rec = deploy.arm_step("nf4", out, 4500)
            self.assertEqual(rec["process"]["env"], {"LIR_QUANT": "nf4"})           # NF4 comes from the loader's own variable
            self.assertIn("lir.deploy arm --arm nf4 --out", rec["process"]["command"])
            self.assertEqual((rec["measured"], rec["memory_after_load_mib"]), (False, 2600.0))
            self.assertIn("stopped after 4500 s", rec["error"])

    def test_unpacking_in_place_of_the_kernel_reads_quanto_packing(self):
        import torch

        def pack(values, bits):     # optimum-quanto 0.2.7, tensor/packed.py, pack_weights
            per = 8 // bits
            rows = (values.shape[0] + per - 1) // per
            packed = torch.zeros((rows, *values.shape[1:]), dtype=torch.uint8)
            for i in range(min(per, values.shape[0] // rows + 1)):
                start, end = i * rows, min(i * rows + rows, values.shape[0])
                packed[: end - start] |= values[start:end].to(torch.uint8) << (bits * i)
            return packed

        g = torch.Generator().manual_seed(0)
        for bits, rows in ((2, 10), (2, 12), (4, 7)):
            values = torch.randint(0, 2 ** bits, (rows, 6), generator=g, dtype=torch.uint8)
            self.assertTrue(torch.equal(deploy.torch_unpack(pack(values, bits), bits)[:rows], values), (bits, rows))


class Table(unittest.TestCase):
    def test_the_table_from_files_shaped_like_the_real_ones(self):
        judged = [{"uid": "a", "task_score": 5.0}, {"uid": "b", "task_score": 6.0}, {"uid": "c", "task_score": 4.0}, {"uid": "d", "task_score": None}]
        res = deploy.table(job_record(), quantization(), quantization_mmlu(), gpu(), mmlu_rows(), judged, n_task=4, items=2)
        self.assertEqual(list(res["columns"].values()), ["bfloat16", "NF4 (bitsandbytes)", "Uniform 4-bit (quanto)", "Circuit plan (quanto)"])
        # from this job, on its GPU; an arm that was not measured has an empty cell
        self.assertEqual(cell(res, "Memory after loading", "nf4"), [[2600.0, "d"]])
        self.assertEqual(cell(res, "Output tokens per second, batch 8", "int4"), [[190.0, "d"]])
        self.assertEqual(cell(res, "Perplexity of the 300 task references", "bf16"), [[5.5, "d"]])
        self.assertEqual(cell(res, "Peak memory at batch 8", "circuit"), [])
        # section 15 for bfloat16 and the quanto arms, and beside bfloat16 the GPU run NF4 was compared with
        self.assertEqual(cell(res, "Strict attack success, English", "bf16"), [[7.0, "a"], [5.5, "b"]])
        self.assertEqual(cell(res, "Strict attack success, English", "nf4"), [[11.0, "b"]])
        self.assertEqual(cell(res, "Strict attack success, Korean", "circuit"), [[12.0, "a"]])
        self.assertEqual(cell(res, "Strict attack success, mean of the four languages", "int4"), [[6.75, "a"]])
        self.assertEqual(cell(res, "Strict attack success, mean of the four languages", "nf4"), [[(11.0 + 7.9 + 9.6 + 7.2) / 4, "b"]])
        self.assertEqual(cell(res, "Refusals of XSTest's safe prompts, Swahili", "bf16"), [[9.6 + 3, "a"], [8.4, "b"]])
        self.assertEqual(cell(res, "Refusals of XSTest's safe prompts, Swahili", "nf4"), [[5.6, "b"]])
        # the task score: section 15, and for NF4 this job's judge over the answers it scored
        self.assertEqual(cell(res, "Task score (1 to 7, 300 task prompts)", "circuit"), [[4.78, "a"]])
        self.assertEqual(cell(res, "Task score (1 to 7, 300 task prompts)", "nf4"), [[5.0, "d"]])
        self.assertEqual((res["nf4_task_scored"], res["nf4_task_prompts"]), (3, 4))
        # Global-MMLU-Lite: section 15b for the three, this job's rows for NF4
        self.assertEqual(cell(res, "Global-MMLU-Lite accuracy, four languages", "int4"), [[51.5, "c"]])
        self.assertEqual(cell(res, "Global-MMLU-Lite accuracy, four languages", "nf4"), [[50.0, "d"]])
        # each run with the hardware its own file names
        s = res["sources"]
        self.assertEqual((s["a"]["hardware"], s["b"]["hardware"], s["c"]["hardware"], s["d"]["hardware"]),
                         ("Apple laptop GPU (MPS)", "NVIDIA L4", "NVIDIA L4", "NVIDIA A100-SXM4-80GB"))
        self.assertEqual(res["not_measured"], {"circuit": "RuntimeError: a quanto kernel failed"})
        md = deploy.report_md(res)
        self.assertIn("| | bfloat16 | NF4 (bitsandbytes) | Uniform 4-bit (quanto) | Circuit plan (quanto) |", md)
        self.assertIn("| Strict attack success, English (%) | 7.0 (a) / 5.5 (b) | 11.0 (b) | 6.0 (a) | 9.0 (a) |", md)
        self.assertIn("| Memory after loading (MiB) | 6,400 (d) | 2,600 (d) | 2,900 (d) |  |", md)
        self.assertIn("- (c) section 15b: NVIDIA L4; quanto weights written out as plain bfloat16 numbers.", md)
        self.assertIn("- (d) this job: NVIDIA A100-SXM4-80GB; quanto weights packed on the GPU, upload `20261127-010000-abc1234`, commit `abc1234`.", md)
        self.assertIn("NF4's task score is the mean over the 3 of 4 answers the judge scored.", md)
        self.assertIn("Uniform 4-bit (quanto): mid TinyGemmWeightQBitsTensor 252", md)
        self.assertIn("Not measured in this job: `circuit`: RuntimeError: a quanto kernel failed.", md)

    def test_the_mean_of_the_four_languages_must_reproduce_q0(self):
        quant = quantization()
        quant["q0"]["mean"] += 0.5
        with self.assertRaises(SystemExit):
            deploy.table(job_record(), quant, quantization_mmlu(), gpu())

    def test_nf4_benchmarks_missing_and_a_judge_that_did_not_run(self):
        job = dict(job_record(), judge={"run": False, "why": "250 NF4 task answers, wanted 300"})
        job["arms"]["nf4"] = dict(job["arms"]["nf4"], mmlu={"error": "SystemExit: en: 399 questions"})
        res = deploy.table(job, quantization(), quantization_mmlu(), gpu(), mmlu_rows()[:-1], None, items=2)
        self.assertEqual(cell(res, "Global-MMLU-Lite accuracy, four languages", "nf4"), [])     # not every question answered
        self.assertEqual(cell(res, "Task score (1 to 7, 300 task prompts)", "nf4"), [])
        self.assertEqual(res["not_measured"]["judge"], "250 NF4 task answers, wanted 300")
        self.assertIn("en: 399 questions", res["not_measured"]["nf4 mmlu"])
        # an arm measured with the stand-in for quanto's kernel is listed as not measured with that kernel
        job["arms"]["int4"] = dict(job["arms"]["int4"], quanto_cuda={"built": False, "error": "OSError: CUDA_HOME is not set", "unpack": "torch_unpack"})
        res = deploy.table(job, quantization(), quantization_mmlu(), gpu(), None, None)
        self.assertIn("OSError: CUDA_HOME is not set", res["not_measured"]["int4 with quanto's kernel"])
        self.assertEqual(cell(res, "Output tokens per second, batch 8", "int4"), [[190.0, "d"]])   # its figures are still shown

    def test_the_real_files_have_the_shapes_the_table_reads(self):
        out = ROOT / "results" / "lir" / "stage2"
        names = ("quantization.json", "quantization_mmlu.json", "gpu.json")
        if not all((out / n).exists() for n in names):
            self.skipTest("the result files of sections 15, 15b and 12 are not here")
        quant, mm15, g = (json.loads((out / n).read_text()) for n in names)
        res = deploy.table(job_record(), quant, mm15, g)                           # reads them; writes nothing
        self.assertEqual(cell(res, "Strict attack success, Korean", "bf16")[0], [quant["per_language"]["ko"]["strict"]["bf16"], "a"])
        self.assertEqual(cell(res, "Strict attack success, Korean", "nf4"), [[g["nf4"]["q1"]["base"]["ko"]["b"], "b"]])
        self.assertEqual(cell(res, "Global-MMLU-Lite accuracy, four languages", "circuit"), [[mm15["accuracy"]["circuit"]["all"], "c"]])
        self.assertEqual(res["sources"]["c"]["hardware"], deploy.gpu_name(mm15["job"]["gpu"][0]))
        self.assertTrue(res["sources"]["b"]["hardware"].startswith("NVIDIA"))

    def folders(self, d: Path, upload: bool):
        out, runs = d / "out", d / "gpu"
        out.mkdir()
        for name, data in (("quantization.json", quantization()), ("quantization_mmlu.json", quantization_mmlu()), ("gpu.json", gpu())):
            (out / name).write_text(json.dumps(data))
        if upload:
            folder = runs / "deploy" / "20261127-010000-abc1234"
            folder.mkdir(parents=True)
            rec = job_record()
            rec.pop("folder")
            (folder / "job.json").write_text(json.dumps(rec))
            (folder / "nf4.task.judge.jsonl").write_text("".join(json.dumps({"uid": f"u{i}", "task_score": 5.5}) + "\n" for i in range(300)))
        return out, runs

    def test_report_writes_the_table_from_the_newest_upload(self):
        from lir import gpu as g

        with tempfile.TemporaryDirectory() as d:
            out, runs = self.folders(Path(d), upload=True)
            with mock.patch.object(deploy, "OUT", out), mock.patch.object(g, "GPU", runs), contextlib.redirect_stdout(io.StringIO()) as text:
                deploy.report(None)
            res = json.loads((out / "deployment.json").read_text())
            md = (out / "deployment.md").read_text()
        self.assertEqual(res["sources"]["d"]["folder"], "20261127-010000-abc1234")
        self.assertEqual(cell(res, "Task score (1 to 7, 300 task prompts)", "nf4"), [[5.5, "d"]])
        self.assertTrue(md.startswith("### The base model in four loadings"))
        self.assertIn("Arms not measured: circuit: RuntimeError: a quanto kernel failed", text.getvalue())

    def test_report_writes_nothing_without_an_upload(self):
        from lir import gpu as g

        with tempfile.TemporaryDirectory() as d:
            out, runs = self.folders(Path(d), upload=False)
            with mock.patch.object(deploy, "OUT", out), mock.patch.object(g, "GPU", runs), contextlib.redirect_stdout(io.StringIO()) as text:
                deploy.report(None)
            self.assertFalse((out / "deployment.json").exists() or (out / "deployment.md").exists())
        self.assertIn("pending, nothing written", text.getvalue())


class Launch(unittest.TestCase):
    def launch(self, dirty, pushed, dry_run, image=deploy.IMAGE):
        calls = []

        def run(cmd, **kw):
            if cmd[0] == "git":
                out = {"rev-parse": "abc1234def", "status": " M lir/deploy.py" if dirty else "", "branch": "  origin/x" if pushed else ""}[cmd[1]]
                return mock.Mock(stdout=out + "\n", returncode=0)
            calls.append(cmd)
            return mock.Mock(returncode=0)

        a = argparse.Namespace(flavor=deploy.FLAVOR, timeout=deploy.TIMEOUT, image=image, dry_run=dry_run)
        with mock.patch.object(deploy.subprocess, "run", run), contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as stop:
            deploy.launch(a)
        return calls, out.getvalue(), stop.exception.code

    def test_a_dry_run_prints_the_job_on_an_a100_for_three_hours(self):
        from lir.hfjob import NAMESPACE
        from lir.mmlu import CIRCUITKIT

        calls, text, code = self.launch(dirty=False, pushed=False, dry_run=True)
        self.assertEqual((len(calls), code), (1, 0))
        self.assertIn("--dry-run", calls[0])                                       # hf prints the job and submits nothing
        self.assertTrue(text.startswith(f"hf jobs run --flavor a100-large --namespace {NAMESPACE} --timeout 3h --secrets HF_TOKEN"))
        self.assertIn(f"--env LIR_TIMEOUT=3h --env LIR_IMAGE={deploy.IMAGE} --name lir-deploy --detach --dry-run {deploy.IMAGE} bash -c '", text)
        inner = calls[0][-1]
        self.assertEqual(calls[0][-3:-1], ["bash", "-c"])
        for piece in ("set -euo pipefail; export LIR_JOB_START=$(date +%s); ", "apt-get install -y -qq --no-install-recommends git g++",
                      "export PIP_BREAK_SYSTEM_PACKAGES=1", "git checkout -q abc1234def", "pip install -q -r requirements-gpu.txt; pip install -q -r requirements-gpu-quant.txt",
                      f"git -C vendor/CircuitKIT checkout -q {CIRCUITKIT[1]}", "PYTHONPATH=.:vendor/CircuitKIT/src python -m lir.deploy job"):
            self.assertIn(piece, inner)
        self.assertIn("a launch would stop here: commit abc1234 is not on the remote yet", text)
        calls, text, code = self.launch(dirty=False, pushed=True, dry_run=True, image="python:3.12")
        self.assertIn("python:3.12", calls[0])
        self.assertNotIn(deploy.IMAGE, calls[0])
        for dirty, pushed in ((False, False), (True, True)):                       # without --dry-run nothing is printed or submitted
            calls, text, code = self.launch(dirty, pushed, dry_run=False)
            self.assertEqual((calls, text), ([], ""))
            self.assertIn("uncommitted changes" if dirty else "not on the remote yet", code)
        calls, text, code = self.launch(dirty=False, pushed=True, dry_run=False)
        self.assertEqual((len(calls), code, "--dry-run" in calls[0], "would stop" in text), (1, 0, False, False))

    def test_the_command_line(self):
        with mock.patch("sys.argv", ["lir.deploy", "arm", "--arm", "nf4"]), self.assertRaises(SystemExit):
            deploy.main()                                                          # an arm's process needs the job's folder
        with mock.patch("sys.argv", ["lir.deploy", "arm", "--arm", "fp8", "--out", "x"]), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            deploy.main()

    def test_the_job_log_holds_what_the_console_shows(self):
        console, log = io.StringIO(), io.StringIO()
        tee = deploy.Tee(console, log)
        print("--- bf16: at most 1800 s", file=tee)
        self.assertEqual(console.getvalue(), log.getvalue())
        self.assertFalse(tee.isatty())                                             # what a progress bar asks goes to the console

    def test_every_file_the_job_writes_is_fetched_by_lir_gpu(self):
        from lir import gpu

        patterns = ["*.jsonl", "*job.json", "*.log", "*run.json"]                  # the allow_patterns of `python -m lir.gpu fetch`
        source = inspect.getsource(gpu.fetch)
        for p in patterns:
            self.assertIn(f'"{p}"', source)
        names = ["job.json", "logs/job.log", "logs/nf4.judge.log", "nf4.task.gen.jsonl", "nf4.task.judge.jsonl", "nf4.mmlu.jsonl"]
        names += [n for arm in deploy.ARMS for n in (f"logs/{arm}.log", f"{arm}.run.json", f"{arm}.batch1.jsonl", f"{arm}.batch8.jsonl")]
        job_source = inspect.getsource(deploy)
        for n in ("job.json", "job.log", "nf4.judge.log", "nf4.task.gen.jsonl", "nf4.task.judge.jsonl", "nf4.mmlu.jsonl",
                  '{arm}.log"', '{a.arm}.run.json', '{arm}.run.json', '{arm}.batch1.jsonl', '{arm}.batch8.jsonl'):
            self.assertIn(n, job_source)                                           # the names are the module's own
        for n in names:
            self.assertTrue(any(fnmatch.fnmatch(f"deploy/20261127-010000-abc1234/{n}", p) for p in patterns), n)


if __name__ == "__main__":
    unittest.main()
