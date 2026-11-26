"""Section 22: the base model in four loadings on one GPU, and one table that puts them beside the benchmarks of
sections 15, 15b and 12.2.

  python -m lir.deploy launch --dry-run   # print the job, submit nothing
  python -m lir.deploy launch             # submit it; needs a login with Jobs rights on the namespace
  python -m lir.deploy job                # what the job runs in its container: each arm, the judge, the upload
  python -m lir.deploy report             # after `python -m lir.gpu fetch`: results/lir/stage2/deployment.{json,md}

Tiny Aya Global with no adapter, at the pinned revision, loaded four ways: bfloat16; NF4 through bitsandbytes, by the
loader of section 12.2 (LIR_QUANT=nf4); uniform 4-bit and the circuit plan at 3, 27 and 6 layers at 8, 4 and 2 bits,
each applied on the CPU by lir.quantize.apply (CircuitKIT with optimum-quanto, as in section 15) and moved to the GPU
with its weights still packed. Each arm runs in a process of its own and is measured the same way: GPU memory after
loading, perplexity of the 300 task references (lir.quantize.perplexity, the measure of section 15's gate) with the
peak memory during it, and the speed of greedy generation on the first 50 task prompts at batch 1 and at batch 8.
bfloat16 and NF4 also answer the 300 task prompts as lir.evaluate does and read Global-MMLU-Lite as lir.mmlu does, so
that NF4's two benchmark figures sit beside a bfloat16 run on the same GPU. Last, after the quanto arms, the task
judge (lir.judge) scores the answers of each of the two, each arm in a process of its own. An arm that cannot be
built or run is recorded with its error and the next one runs. Nothing here is run on the laptop except `launch` and
`report`, which loads no model.

The container. optimum-quanto 0.2.7 unpacks the weights of its 2-bit layers on CUDA with a kernel it compiles at first
use, and registers no fallback for it on CUDA; its 4-bit layers in bfloat16 use the tinygemm kernel built into torch,
and its 8-bit layers a plain matrix product. The job therefore runs on PyTorch's development image for the torch and
CUDA of the earlier jobs (torch 2.15.0, CUDA 13.2), which has nvcc. It adds git and g++ when they are missing (the
image's build history installs gcc and neither of them). Each
quanto arm builds the kernel before its model is loaded, so that no measured phase includes the build. If the kernel
does not build, its error is recorded and quanto's own default implementation of the same operation, in torch, takes
its place for that arm; the record says which one ran. `--image python:3.12` runs the job on the image of the
earlier jobs, where the kernel cannot build.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from pathlib import Path

from lir.common import EVAL_LANGS, ROOT, RUNS, read_jsonl, sha256_file, write_json, write_jsonl

JOB = "deploy"
FLAVOR, TIMEOUT = "a100-large", "4h"
# Docker Hub on Nov 26, 2026: pushed Oct 28, amd64 manifest sha256:3878fcd85c416d8a837aca22f6b3382a4dd9e77111da4ad6d3612fc354912321.
# Ubuntu 24.04, its Python 3.12, torch 2.15.0, the CUDA 13.2.2 toolkit with nvcc at /usr/local/cuda, gcc but no g++ or git.
IMAGE = "pytorch/pytorch:2.15.0-cuda13.2-cudnn9-devel"
ARMS = ["bf16", "nf4", "int4", "circuit"]
QUANTO = ("int4", "circuit")
BENCHED = ("bf16", "nf4")                           # the arms that also answer the task set and Global-MMLU-Lite
# the speed and memory of the quanto arms are what the job is for, so the judge of the task answers comes last
STEPS = ["bf16", "nf4", "int4", "circuit", "judge"]
RUNG = [3, 27, 6]                                   # layers at 8, 4 and 2 bits: the tier size section 15's gate chose
SPEED_ROWS, SPEED_NEW_TOKENS, BATCH = 50, 256, 8
# seconds a step may run at most; the job also keeps RESERVE seconds before its own limit for the upload
CAPS = {"bf16": 3600, "nf4": 4500, "int4": 2700, "circuit": 3000, "judge": 2700}
RESERVE, LEAST = 900, 300                           # a step with less than LEAST seconds left is not started
OUT = ROOT / "results" / "lir" / "stage2"
COLUMNS = {"bf16": "bfloat16", "nf4": "NF4 (bitsandbytes)", "int4": "Uniform 4-bit (quanto)", "circuit": "Circuit plan (quanto)"}
# how optimum-quanto 0.2.7 multiplies by a weight, by the class of the weight tensor on the GPU
MATMUL = {"TinyGemmWeightQBitsTensor": "torch._weight_int4pack_mm (tinygemm, built into torch)",
          "AWQWeightQBitsTensor": "quanto's AWQ kernel",
          "MarlinInt4WeightQBitsTensor": "quanto's Marlin kernel",
          "WeightQBitsTensor": "weights unpacked (quanto::unpack) and scaled, then torch.matmul",
          "WeightQBytesTensor": "quanto::qbytes_mm: 8-bit weights scaled to bfloat16, then torch.matmul"}


# ---------------------------------------------------------------- on the laptop: build and submit the job

def install_command() -> str:
    """git and g++ if the image lacks them, then the installs of lir.mmlu's job. Ubuntu's Python refuses a pip install
    outside a virtual environment unless told otherwise; on python:3.12 the variable changes nothing."""
    return ("command -v git >/dev/null && command -v g++ >/dev/null || (apt-get update -qq && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git g++ >/dev/null); "
            "export PIP_BREAK_SYSTEM_PACKAGES=1; ")


def job_command(commit: str, flavor: str = FLAVOR, timeout: str = TIMEOUT, image: str = IMAGE, dry_run: bool = False) -> list[str]:
    from lir.hfjob import NAMESPACE, REPO_URL, STAGING
    from lir.mmlu import CIRCUITKIT

    inner = ("set -euo pipefail; export LIR_JOB_START=$(date +%s); " + install_command()
             + f"git clone -q {REPO_URL} repo; cd repo; git checkout -q {commit}; "
             "pip install -q -r requirements-gpu.txt; pip install -q -r requirements-gpu-quant.txt; "
             f"git clone -q {CIRCUITKIT[0]} vendor/CircuitKIT; git -C vendor/CircuitKIT checkout -q {CIRCUITKIT[1]}; "
             f"PYTHONPATH=.:vendor/CircuitKIT/src python -m lir.{JOB} job")
    return (["hf", "jobs", "run", "--flavor", flavor, "--namespace", NAMESPACE, "--timeout", timeout, "--secrets", "HF_TOKEN",
             "--env", f"LIR_FLAVOR={flavor}", "--env", f"LIR_STAGING_REPO={STAGING}", "--env", f"LIR_TIMEOUT={timeout}",
             "--env", f"LIR_IMAGE={image}", "--name", f"lir-{JOB}", "--detach"]
            + (["--dry-run"] if dry_run else []) + [image, "bash", "-c", inner])


def launch(a) -> None:
    git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    stops = []
    if git("status", "--porcelain", "--untracked-files=no"):
        stops.append("tracked files have uncommitted changes: the job would not run the code you see")
    if not git("branch", "-r", "--contains", commit):
        stops.append(f"commit {commit[:7]} is not on the remote yet: push it first, the job clones from GitHub")
    if stops and not a.dry_run:
        raise SystemExit(stops[0])
    cmd = job_command(commit, a.flavor, a.timeout, a.image, a.dry_run)
    print(" ".join(cmd[:-1]) + f" '{cmd[-1]}'")
    for why in stops:   # a dry run prints the job all the same, and says what a launch would stop on
        print(f"a launch would stop here: {why}")
    sys.exit(subprocess.run(cmd, cwd=ROOT).returncode)


# ---------------------------------------------------------------- the arithmetic of speed

def new_tokens(ids: list[int], pad: int | None, stop: list[int]) -> tuple[int, bool]:
    """The count lir.generate keeps for one answer: padding dropped, cut at the first stop token. Also whether a stop
    token was reached (lir.generate's truncated is the opposite)."""
    ids = [t for t in ids if t != pad]
    cut = next((k for k, t in enumerate(ids) if t in stop), None)
    return (len(ids) if cut is None else cut), cut is not None


def rates(timings: list[dict]) -> dict:
    """timings: one row per generate call, {"new_tokens", "seconds", "warmup"}. Warm-up rows are left out. Output tokens
    per second is every new token over every second; the median is of each call's own tokens per second."""
    counted = [t for t in timings if not t.get("warmup")]
    tokens, seconds = sum(t["new_tokens"] for t in counted), sum(t["seconds"] for t in counted)
    each = [t["new_tokens"] / t["seconds"] for t in counted if t["seconds"] > 0]
    return {"calls": len(counted), "new_tokens": tokens, "seconds": round(seconds, 3),
            "tokens_per_second": tokens / seconds if seconds > 0 else None,
            "median_tokens_per_second": statistics.median(each) if each else None,
            "warmup_calls": len(timings) - len(counted)}


def batches(rows: list[dict], lengths: dict[str, int], size: int = BATCH) -> list[list[dict]]:
    """Batches of `size`, after the rows are sorted longest prompt first, as lir.generate sorts them. Python's sort is
    stable, so prompts of one length keep their file order."""
    order = sorted(rows, key=lambda r: -lengths[r["uid"]])
    return [order[i:i + size] for i in range(0, len(order), size)]


def mib(n: int) -> float:
    return round(n / 2 ** 20, 1)


def seconds_of(limit: str) -> int:
    """'3h', '90m', '600s' or '600' in seconds."""
    unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return int(float(limit[:-1]) * unit[limit[-1]]) if limit[-1] in unit else int(limit)


# ---------------------------------------------------------------- inside an arm's process

@contextmanager
def env(**values):
    """Set environment variables for the length of a block, then put back what was there."""
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def torch_unpack(packed, bits: int):
    """optimum-quanto 0.2.7's default implementation of quanto::unpack (library/unpack.py), the one it registers for
    devices without a kernel: the 8 // bits values held in each byte, one after another along the first dimension.
    Used on CUDA only when quanto's kernel does not build."""
    import torch

    values_per_item = 8 // bits
    return torch.cat([(packed & (2 ** (bits * (i + 1)) - 1)) >> (bits * i) for i in range(values_per_item)]).to(torch.uint8)


def quanto_cuda() -> dict:
    """Build optimum-quanto's CUDA extension now, before any model is loaded, and say how long it took. If it does not
    build, quanto's default implementation of unpack takes the place of the kernel, so that a 2-bit layer still runs
    and the build is not tried again at every call."""
    import shutil
    from types import SimpleNamespace

    from optimum.quanto.library.extensions import get_extension, is_extension_available

    rec = {"nvcc": shutil.which("nvcc"), "cuda_home": os.environ.get("CUDA_HOME"), "registered": is_extension_available("quanto_cuda")}
    if not rec["registered"]:
        return dict(rec, unpack="no CUDA kernel is registered: quanto's default implementation")
    ext, t0 = get_extension("quanto_cuda"), time.time()
    try:
        ext.lib
        rec.update(built=True, unpack="quanto_cuda kernel")
    except Exception as e:
        ext._lib = SimpleNamespace(unpack=torch_unpack)
        rec.update(built=False, error=f"{type(e).__name__}: {e}"[-4000:],
                   unpack="lir.deploy.torch_unpack, quanto's default implementation in torch, in place of the kernel that did not build")
    rec["seconds"] = round(time.time() - t0, 1)
    print(f"quanto_cuda: {rec['unpack']} ({rec['seconds']} s)", flush=True)
    return rec


def paths(model, plan: dict, expected: int) -> dict:
    """Which multiplication each quantized layer uses on the GPU, by tier: the class of its weight tensor there. Stops
    if the move to the GPU changed the number of quantized layers."""
    from optimum.quanto.nn.qmodule import QModuleMixin

    from lir.quantize import tier_of

    found = {name: m for name, m in model.named_modules() if isinstance(m, QModuleMixin)}
    if len(found) != expected:
        raise SystemExit(f"{len(found)} quantized layers on the GPU, {expected} were quantized on the CPU")
    classes: dict[str, dict[str, int]] = {}
    for name, m in found.items():
        tier, cls = tier_of(plan, name), type(m.qweight).__name__
        classes.setdefault(tier, {})
        classes[tier][cls] = classes[tier].get(cls, 0) + 1
    names = sorted({c for v in classes.values() for c in v})
    return {"weight_classes": classes, "matmul": {c: MATMUL.get(c, "not described here") for c in names},
            "devices": sorted({str(m.qweight.device) for m in found.values()})}


def build(arm: str) -> tuple:
    """(tokenizer, model, record) for one arm, on the GPU. bf16 and nf4 through generate.load_model, which reads
    LIR_QUANT; the quanto arms loaded on the CPU, quantized in place by lir.quantize.apply with the plan of
    lir.mmlu.arm_plan, and moved to the GPU packed: nothing writes their weights out as plain numbers."""
    from lir import generate

    if arm in ("bf16", "nf4"):
        if (os.environ.get("LIR_QUANT") == "nf4") != (arm == "nf4"):
            raise SystemExit(f"{arm}: LIR_QUANT is {os.environ.get('LIR_QUANT')!r}")
        tok, model = generate.load_model(None)
        return tok, model, {"loader": "lir.generate.load_model" + (" with LIR_QUANT=nf4" if arm == "nf4" else "")}
    from lir import mmlu
    from lir import quantize as q

    plan, rung = mmlu.arm_plan(arm)
    if arm == "circuit" and rung != RUNG:
        raise SystemExit(f"the circuit plan is at tier size {rung}, not {RUNG}")
    kernel = quanto_cuda()
    with env(LIR_DEVICE="cpu"):
        tok, model = generate.load_model(None)
    t0 = time.time()
    applied = q.apply(model, plan)
    seconds = round(time.time() - t0, 1)
    model.to("cuda")
    return tok, model, {"loader": "lir.generate.load_model on the CPU, lir.quantize.apply, then the packed model to CUDA",
                        "rung": rung, "layers": q.tier_lists(plan), "apply": applied, "seconds_to_quantize": seconds,
                        "quanto_cuda": kernel, "kernels": paths(model, plan, applied["quantized_modules"])}


def phase(fn) -> tuple:
    """fn() with the GPU synchronized and its peak memory statistics reset before, synchronized after: its value, the
    peak memory allocated and reserved during it (MiB), and its seconds."""
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, mib(torch.cuda.max_memory_allocated()), mib(torch.cuda.max_memory_reserved()), round(time.perf_counter() - t0, 2)


def timed(tok, model, batch: list[dict], max_new_tokens: int = SPEED_NEW_TOKENS) -> tuple[float, list[dict]]:
    """One generate call on a batch, as lir.generate makes it (chat template, left padding, greedy, its stop tokens),
    timed with the GPU synchronized on both sides. The seconds, and per prompt what lir.generate would keep."""
    import torch

    from lir.generate import chat_prefix

    stop = [tok.eos_token_id, tok.convert_tokens_to_ids("<EOS_TOKEN>")]
    dev = next(model.parameters()).device
    enc = tok([chat_prefix(tok, r["prompt"]) for r in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=stop)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    rows = []
    for r, mask, ids in zip(batch, enc.attention_mask.tolist(), out[:, enc.input_ids.shape[1]:].tolist()):
        n, stopped = new_tokens(ids, tok.pad_token_id, stop)
        kept = [t for t in ids if t != tok.pad_token_id][:n]
        rows.append({"uid": r["uid"], "lang": r["lang"], "prompt_tokens": sum(mask), "new_tokens": n, "stopped": stopped,
                     "response": tok.decode(kept, skip_special_tokens=True).strip()})
    return seconds, rows


def speed_rows() -> tuple[list[dict], list[dict]]:
    """The first SPEED_ROWS task prompts in file order, and the BATCH after them for the warm-ups."""
    from lir.evaluate import rows_for

    rows = rows_for("task")
    return rows[:SPEED_ROWS], rows[SPEED_ROWS:SPEED_ROWS + BATCH]


def measure(arm: str, tok, model, out: Path, rec: dict, save) -> None:
    """Memory, perplexity and speed of one arm, written into rec phase by phase and saved after each."""
    import torch

    from lir import quantize as q
    from lir.common import empty_cache
    from lir.generate import chat_prefix

    empty_cache()
    torch.cuda.synchronize()
    rec["memory_after_load_mib"] = mib(torch.cuda.memory_allocated())
    rec["reserved_after_load_mib"] = mib(torch.cuda.memory_reserved())
    save(rec)
    ppl, peak, reserved, secs = phase(lambda: q.perplexity(tok, model, q.task_rows()))
    rec.update(perplexity=ppl, peak_mib_perplexity=peak, peak_reserved_mib_perplexity=reserved, seconds_perplexity=secs)
    save(rec)
    counted, spare = speed_rows()
    # batch 1: one warm-up prompt that is not counted, then the counted prompts one at a time, in file order
    timings = []
    for r, warm in [(spare[0], True)] + [(r, False) for r in counted]:
        seconds, (row,) = timed(tok, model, [r])
        timings.append(dict(row, seconds=round(seconds, 4), warmup=warm))
    write_jsonl(out / f"{arm}.batch1.jsonl", timings)
    rec["batch1"] = rates(timings)
    save(rec)
    # batch 8: one warm-up batch that is not counted, then the counted prompts sorted and batched as lir.generate does
    lengths = {r["uid"]: len(tok(chat_prefix(tok, r["prompt"]), add_special_tokens=False).input_ids) for r in counted}
    seconds, rows = timed(tok, model, spare)
    timings = [{"batch": "warm-up", "uids": [x["uid"] for x in rows], "new_tokens": sum(x["new_tokens"] for x in rows),
                "seconds": round(seconds, 4), "warmup": True, "rows": rows}]

    def counted_batches():
        for i, b in enumerate(batches(counted, lengths)):
            seconds, rows = timed(tok, model, b)
            timings.append({"batch": i, "uids": [x["uid"] for x in rows], "new_tokens": sum(x["new_tokens"] for x in rows),
                            "seconds": round(seconds, 4), "warmup": False, "rows": rows})

    _, peak, reserved, _ = phase(counted_batches)
    write_jsonl(out / f"{arm}.batch8.jsonl", timings)
    rec.update(batch8=rates(timings), peak_mib_batch8=peak, peak_reserved_mib_batch8=reserved)
    save(rec)


def arm_task(arm: str, tok, model, out: Path) -> dict:
    """The 300 task prompts answered as lir.evaluate answers the task set: lir.generate at the set's cap."""
    from lir import generate
    from lir.evaluate import SETS, rows_for

    path, t0 = out / f"{arm}.task.gen.jsonl", time.time()
    generate.generate(tok, model, rows_for("task"), path, SETS["task"][1], tag=f"{JOB}-{arm}/task")
    rows = read_jsonl(path)
    return {"file": path.name, "answers": len(rows), "max_new_tokens": SETS["task"][1], "capped": sum(bool(r.get("truncated")) for r in rows),
            "sha256": sha256_file(path), "seconds": round(time.time() - t0, 1)}


def arm_mmlu(arm: str, tok, model, out: Path) -> dict:
    """Global-MMLU-Lite in the four languages at the pinned revision, read as lir.mmlu reads it."""
    from lir import mmlu

    t0, ids, rows, files = time.time(), mmlu.letter_ids(tok), [], {}
    for lang in EVAL_LANGS:
        data, files[lang] = mmlu.load_questions(lang)
        rows += mmlu.score(tok, model, data, ids)
    path = out / f"{arm}.mmlu.jsonl"
    write_jsonl(path, [dict(r, arm=arm) for r in rows])
    return {"file": path.name, "dataset": mmlu.DATASET, "dataset_revision": mmlu.DATASET_REV, "files_sha256": files,
            "answered": len(rows), "accuracy": mmlu.result({arm: rows})["accuracy"].get(arm), "letter_tokens": ids,
            "seconds": round(time.time() - t0, 1)}


def error_text(e: BaseException) -> dict:
    return {"error": f"{type(e).__name__}: {e}"[-4000:], "traceback": traceback.format_exc()[-8000:]}


def run_arm(arm: str, out: Path, save, build=build, measure=measure, extras=None) -> dict:
    """Build and measure one arm. Any error, a stop included, is written into the record with its traceback and ends
    the arm; what was measured before it stays. measured is true only when every phase ended. For bf16 and nf4, once
    the model is built, the task answers and Global-MMLU-Lite follow, each recorded on its own."""
    from functools import partial

    from lir.common import empty_cache

    if extras is None:
        extras = {"task": partial(arm_task, arm), "mmlu": partial(arm_mmlu, arm)} if arm in BENCHED else {}
    rec = {"arm": arm, "measured": False, "started_utc": time.strftime("%Y%m%d-%H%M%S", time.gmtime())}
    tok = model = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")   # each warning recorded once, so a warning repeated at every step costs nothing
        try:
            t0 = time.time()
            tok, model, info = build(arm)
            rec.update(info, seconds_to_build=round(time.time() - t0, 1))
            save(rec)
            measure(arm, tok, model, out, rec, save)
            rec["measured"] = True
        except (Exception, SystemExit) as e:
            rec.update(error_text(e))
        save(rec)
        for name, fn in extras.items():
            if model is None:
                break
            try:
                rec[name] = fn(tok, model, out)
            except (Exception, SystemExit) as e:
                rec[name] = error_text(e)
            save(rec)
        rec["warnings"] = sorted({f"{w.category.__name__}: {w.message}"[:500] for w in caught})[:40]
    tok = model = None
    empty_cache()
    rec["finished_utc"] = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    save(rec)
    return rec


def arm(a) -> None:
    out = Path(a.out)
    rec = run_arm(a.arm, out, lambda r: write_json(out / f"{a.arm}.run.json", r))
    print(json.dumps({k: rec.get(k) for k in ("arm", "measured", "memory_after_load_mib", "error")}), flush=True)
    sys.exit(0 if rec["measured"] else 1)


# ---------------------------------------------------------------- the job: the steps in order, then the upload

def child(cmd: list[str], extra_env: dict, log: Path, limit: float) -> dict:
    """Run one command in a process group of its own, its output to `log`; kill the group if it runs past `limit`."""
    t0 = time.time()
    with log.open("w") as f:
        p = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **extra_env}, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.wait(timeout=limit)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            timed_out = True
    print(log.read_text(errors="replace")[-3000:], flush=True)
    return {"command": " ".join(cmd[1:]), "env": extra_env, "returncode": p.returncode, "timed_out": timed_out,
            "seconds": round(time.time() - t0, 1), "limit_seconds": round(limit)}


def arm_step(arm: str, out: Path, limit: float) -> dict:
    from lir.hfjob import NF4

    proc = child([sys.executable, "-m", f"lir.{JOB}", "arm", "--arm", arm, "--out", str(out)], dict(NF4) if arm == "nf4" else {},
                 out / "logs" / f"{arm}.log", limit)
    path = out / f"{arm}.run.json"
    rec = json.loads(path.read_text()) if path.exists() else {"arm": arm, "measured": False}
    if proc["returncode"] != 0:
        rec["process_error"] = (f"stopped after {proc['limit_seconds']} s, the time left to it" if proc["timed_out"]
                                else f"the process ended with code {proc['returncode']}; see logs/{arm}.log")
        rec.setdefault("error", rec["process_error"])
        rec["measured"] = rec.get("measured", False) and not proc["timed_out"]
    return dict(rec, process=proc)


def judge_step(out: Path, limit: float, save=lambda got: None) -> dict:
    """The task answers of bf16 and of nf4 scored by the task judge, each arm in a process of its own, as lir.xlsum's
    judge job runs it. An arm without all its answers is recorded as not judged, and the other is judged all the same.
    The arms share the step's `limit`."""
    from lir.common import JUDGE
    from lir.evaluate import rows_for
    from lir.judge import JUDGE_REV

    n, end, got = len(rows_for("task")), time.time() + limit, {}
    for arm in BENCHED:
        gen, scored = out / f"{arm}.task.gen.jsonl", out / f"{arm}.task.judge.jsonl"
        have = len(read_jsonl(gen)) if gen.exists() else 0
        left = end - time.time()
        if have != n:
            got[arm] = {"run": False, "why": f"{have} task answers, wanted {n}"}
        elif left < LEAST:
            got[arm] = {"run": False, "why": f"not started: {round(max(left, 0))} s were left to the judge"}
        else:
            proc = child([sys.executable, "-m", "lir.judge", "--input", str(gen), "--output", str(scored)], {},
                         out / "logs" / f"{arm}.judge.log", left)
            rows = read_jsonl(scored) if scored.exists() else []
            scores = [r["task_score"] for r in rows if r.get("task_score") is not None]
            got[arm] = {"run": True, "judge": f"{JUDGE}@{JUDGE_REV}", "answers_sha256": sha256_file(gen), "answers": have,
                        "rows": len(rows), "scored": len(scores), "task_score": sum(scores) / len(scores) if scores else None,
                        "file": scored.name, "process": proc}
        save(got)
    return got


def run_all(record: dict, step, save, left) -> None:
    """The steps in order. step(name, limit) returns the record of one step; left() is the seconds the job may still
    spend. A step that raises, or has too little time left, is recorded and the next step runs. The judge's record
    has one entry per arm it judges, and a judge that did not run is written into each."""
    for name in STEPS:
        limit = min(CAPS[name], left())
        if limit < LEAST:
            got = {"run": False, "measured": False, "why": f"not started: {round(max(limit, 0))} s were left before the job's limit"}
        else:
            print(f"--- {name}: at most {round(limit)} s", flush=True)
            try:
                got = step(name, limit)
            except (Exception, SystemExit) as e:
                got = {"measured": False, **error_text(e)}
        if name == "judge":
            record["judge"] = got if all(a in got for a in BENCHED) else {a: dict(got) for a in BENCHED}
        else:
            record["arms"][name] = got
        save()


def installed() -> dict:
    """What the job runs on. Stops if the packages are not at the pins of requirements-gpu.txt and
    requirements-gpu-quant.txt, a local build label such as +cu132 aside, or CircuitKIT is not at its commit."""
    import shutil
    from importlib import metadata

    import torch

    from lir.mmlu import CIRCUITKIT

    pins = {}
    for f in ("requirements-gpu.txt", "requirements-gpu-quant.txt"):
        pins.update(line.strip().split("==") for line in (ROOT / f).read_text().splitlines() if "==" in line and not line.startswith("#"))
    got = {n: metadata.version(n) for n in ("torch", "transformers", "bitsandbytes", "optimum-quanto")}
    moved = {n: v for n, v in got.items() if v.split("+")[0] != pins[n]}
    if moved:
        raise SystemExit(f"installed {moved}, pinned {[pins[n] for n in moved]}")
    kit = subprocess.run(["git", "-C", str(ROOT / "vendor" / "CircuitKIT"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if kit != CIRCUITKIT[1]:
        raise SystemExit(f"CircuitKIT is at {kit or 'no commit'}, not at {CIRCUITKIT[1]}")
    nvcc = shutil.which("nvcc")
    said = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines() if nvcc else []
    nvcc_version = next((line for line in said if "release" in line), said[-1] if said else None)
    return {**got, "torch_cuda": torch.version.cuda, "nvcc": nvcc_version, "circuitkit_commit": kit, "image": os.environ.get("LIR_IMAGE")}


def settings() -> dict:
    from lir import quantize as q
    from lir.common import DATA
    from lir.evaluate import SETS

    counted, spare = speed_rows()
    return {"perplexity": {"rows": len(q.task_rows()), "function": "lir.quantize.perplexity", "max_ref_tokens": 256},
            "speed": {"file": "data/lir/bactrian_eval_task.jsonl", "sha256": sha256_file(DATA / "bactrian_eval_task.jsonl"),
                      "prompts": [r["uid"] for r in counted], "warmup_batch1": spare[0]["uid"], "warmup_batch8": [r["uid"] for r in spare],
                      "max_new_tokens": SPEED_NEW_TOKENS, "batch": BATCH, "decoding": "greedy",
                      "batch8_order": "longest prompt first, as lir.generate sorts, file order among equal lengths"},
            "benchmarks": {"arms": list(BENCHED), "task_max_new_tokens": SETS["task"][1]}, "steps": STEPS, "caps_seconds": CAPS,
            "reserve_seconds": RESERVE}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()

    def __getattr__(self, name):   # isatty, fileno, encoding and the rest: the console's
        return getattr(self.streams[0], name)


def job(a) -> None:
    from huggingface_hub import HfApi

    from lir.common import MODEL, MODEL_REV, git_commit
    from lir.hfjob import STAGING, hardware

    hw = hardware()
    if not hw["cuda_available"]:
        raise SystemExit(f"no CUDA GPU in this job: {hw}")
    start = float(os.environ.get("LIR_JOB_START") or time.time())
    deadline = start + seconds_of(os.environ.get("LIR_TIMEOUT", TIMEOUT)) - RESERVE
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = RUNS / "hfjob" / JOB
    (out / "logs").mkdir(parents=True, exist_ok=True)
    log = (out / "logs" / "job.log").open("w")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log), Tee(sys.__stderr__, log)
    record = {"job": JOB, "started_utc": stamp, "commit": git_commit(), "model": [MODEL, MODEL_REV], "hardware": hw,
              "installed": installed(), "settings": settings(), "arms": {}}
    save = lambda: write_json(out / "job.json", record)   # kept current, so that what ran is recorded whatever happens next
    save()
    steps = {"judge": lambda limit: judge_step(out, limit, lambda got: (record.update(judge=dict(got)), save()))}
    try:
        run_all(record, lambda name, limit: steps.get(name, lambda lim: arm_step(name, out, lim))(limit), save,
                lambda: deadline - time.time())
    finally:
        record["not_measured"] = {name: r.get("error") or r.get("why") for name, r in record["arms"].items() if not r.get("measured")}
        record["finished_utc"] = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        save()
        sys.stdout.flush()
        dest = f"{JOB}/{stamp}-{(record['commit'] or 'unknown')[:7]}"
        api = HfApi()
        api.create_repo(STAGING, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=out, repo_id=STAGING, repo_type="dataset", path_in_repo=dest,
                          commit_message=f"{JOB}: memory, speed and perplexity of four loadings; bf16 and NF4 task answers, scores and Global-MMLU-Lite")
        print(f"uploaded to {STAGING}/{dest}", flush=True)
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log.close()


# ---------------------------------------------------------------- the reading

LANG_NAMES = {"en": "English", "ko": "Korean", "zh": "Chinese", "sw": "Swahili"}
DEVICES = {"mps": "Apple laptop GPU (MPS)", "cuda": "NVIDIA GPU", "cpu": "CPU"}


def gpu_name(line: str) -> str:
    """'NVIDIA L4, 620.70.06, 23034 MiB' -> 'NVIDIA L4'."""
    return line.split(",")[0].strip()


def four(rates_by_lang: dict, langs: list[str] = EVAL_LANGS) -> float:
    """The plain mean of the four languages' rates. Every language has the same prompt families, so this is the rate
    over the four languages."""
    return sum(rates_by_lang[l] for l in langs) / len(langs)


def task_mean(rows: list[dict] | None) -> tuple[float | None, int]:
    got = [r["task_score"] for r in rows or [] if r.get("task_score") is not None]
    return (sum(got) / len(got) if got else None), len(got)


def table(job: dict, quant: dict, mm15: dict, gpu12: dict, mmlu_rows: dict | None = None, judge_rows: dict | None = None,
          n_task: int = 300, items: int | None = None) -> dict:
    """The table, from this job's record, section 15's quantization.json, section 15b's quantization_mmlu.json and
    section 12's gpu.json, and this job's Global-MMLU-Lite rows and judge rows of bf16 and nf4 (arm -> rows). A cell
    is a list of [value, source] pairs; a source is one of the keys of "sources". Nothing is recomputed from answers
    but the two benchmark figures of bf16 and nf4 on this job's GPU, which only this job holds."""
    from lir import mmlu

    langs = quant["languages"]
    pl = quant["per_language"]
    base = gpu12["nf4"]["q1"]["base"]                 # a: bfloat16 on the GPU (job control), b: NF4 (job nf4)
    xs12 = gpu12["nf4"]["over_refusal_rates"]["base"]
    strict15 = {arm: {l: pl[l]["strict"][arm] for l in langs} for arm in ("bf16", "int4", "circuit")}
    # the four-language mean is not stored; uniform 4-bit minus bfloat16 of the means has to give Q0's mean
    if abs((four(strict15["int4"], langs) - four(strict15["bf16"], langs)) - quant["q0"]["mean"]) > 1e-9:
        raise SystemExit("the mean of the four languages does not reproduce Q0 of quantization.json")
    hw = job["hardware"]
    rec15 = quant.get("records", {}).get("bf16", {})
    sources = {
        "a": {"run": "section 15", "file": "results/lir/stage2/quantization.json",
              "hardware": DEVICES.get(str(rec15.get("device")).split(":")[0], str(rec15.get("device"))),
              "weights": "quanto weights written out as plain bfloat16 numbers"},
        "b": {"run": "sections 12.0 and 12.2", "file": "results/lir/stage2/gpu.json",
              "hardware": "; ".join(sorted({gpu_name(g) for j in ("control", "nf4") for g in gpu12["jobs"][j]["gpu"]})),
              "detail": "bfloat16 from job control (arm gpu-base), NF4 from job nf4 (arm gpu-base-nf4)"},
        "c": {"run": "section 15b", "file": "results/lir/stage2/quantization_mmlu.json",
              "hardware": "; ".join(gpu_name(g) for g in mm15["job"]["gpu"]), "weights": "quanto weights written out as plain bfloat16 numbers"},
        "d": {"run": "this job", "folder": job.get("folder"), "commit": job.get("commit"), "flavor": hw.get("flavor"),
              "hardware": "; ".join(gpu_name(g) for g in hw["gpus"]), "weights": "quanto weights packed on the GPU"},
    }
    arms = job["arms"]
    get = lambda arm, *keys: _dig(arms.get(arm, {}), *keys)
    rows = []

    def row(name: str, unit: str, cells: dict, fmt: str):
        rows.append({"row": name, "unit": unit, "format": fmt, "cells": {a: [[v, s] for v, s in cells.get(a, []) if v is not None] for a in ARMS}})

    job_row = lambda name, unit, fmt, *keys: row(name, unit, {a: [(get(a, *keys), "d")] for a in ARMS}, fmt)
    job_row("Memory after loading", "MiB", "mib", "memory_after_load_mib")
    job_row("Peak memory during perplexity", "MiB", "mib", "peak_mib_perplexity")
    job_row("Peak memory at batch 8", "MiB", "mib", "peak_mib_batch8")
    job_row("Output tokens per second, batch 1", "tokens/s", "rate", "batch1", "tokens_per_second")
    job_row("Median tokens per second of a prompt, batch 1", "tokens/s", "rate", "batch1", "median_tokens_per_second")
    job_row("Output tokens per second, batch 8", "tokens/s", "rate", "batch8", "tokens_per_second")
    job_row("Perplexity of the 300 task references", "", "ppl", "perplexity", "all")
    for l in langs:
        row(f"Strict attack success, {LANG_NAMES.get(l, l)}", "%", {
            "bf16": [(strict15["bf16"][l], "a"), (base[l]["a"], "b")], "nf4": [(base[l]["b"], "b")],
            "int4": [(strict15["int4"][l], "a")], "circuit": [(strict15["circuit"][l], "a")]}, "pct")
    row("Strict attack success, mean of the four languages", "%", {
        "bf16": [(four(strict15["bf16"], langs), "a"), (four({l: base[l]["a"] for l in langs}, langs), "b")],
        "nf4": [(four({l: base[l]["b"] for l in langs}, langs), "b")],
        "int4": [(four(strict15["int4"], langs), "a")], "circuit": [(four(strict15["circuit"], langs), "a")]}, "pct")
    for l in langs:
        o = quant["over_refusal"][l]
        row(f"Refusals of XSTest's safe prompts, {LANG_NAMES.get(l, l)}", "%", {
            "bf16": [(o["bf16"], "a"), (xs12["bf16"][l], "b")], "nf4": [(xs12["nf4"][l], "b")],
            "int4": [(o["int4"], "a")], "circuit": [(o["circuit"], "a")]}, "pct")
    # bf16 and nf4 on this job's GPU: the mean task score over the answers the judge scored, and Global-MMLU-Lite when
    # every question of the four languages has an answer
    scores = {a: task_mean((judge_rows or {}).get(a)) for a in BENCHED}
    here = {a: mmlu.result({a: (mmlu_rows or {})[a]}, items=items or mmlu.ITEMS)["accuracy"].get(a) if (mmlu_rows or {}).get(a) else None
            for a in BENCHED}
    ts = quant["task_score"]
    row("Task score (1 to 7, 300 task prompts)", "", {"bf16": [(ts["bf16"], "a"), (scores["bf16"][0], "d")], "nf4": [(scores["nf4"][0], "d")],
                                                      "int4": [(ts["int4"], "a")], "circuit": [(ts["circuit"], "a")]}, "score")
    acc = mm15["accuracy"]
    row("Global-MMLU-Lite accuracy, four languages", "%", {
        "bf16": [(acc["bf16"]["all"], "c"), ((here["bf16"] or {}).get("all"), "d")], "nf4": [((here["nf4"] or {}).get("all"), "d")],
        "int4": [(acc["int4"]["all"], "c")], "circuit": [(acc["circuit"]["all"], "c")]}, "pct")
    not_measured = {}
    for a in ARMS:
        r = arms.get(a)
        if r is None:
            not_measured[a] = "no record of this arm in the job"
        elif not r.get("measured"):
            not_measured[a] = r.get("error") or r.get("why") or "not measured"
        elif r.get("process_error"):   # measured, and its process died afterwards
            not_measured[f"{a} after its measurements"] = r["process_error"]
        kernel = (r or {}).get("quanto_cuda") or {}
        if kernel.get("built") is False:
            not_measured[f"{a} with quanto's kernel"] = (f"quanto's CUDA kernel did not build ({kernel.get('error')}); the figures of "
                                                         "this arm use lir.deploy.torch_unpack in its place")
    judge = job.get("judge") or {}
    for a in BENCHED:
        for part, what in (("task", "task answers"), ("mmlu", "Global-MMLU-Lite")):
            if isinstance(get(a, part), dict) and get(a, part, "error"):
                not_measured[f"{a} {part}"] = f"{what}: {get(a, part, 'error')}"
        j = judge.get(a) or {}
        if not j.get("run"):
            not_measured[f"{a} judge"] = j.get("why") or j.get("error") or "the judge did not run"
    return {"columns": {a: COLUMNS[a] for a in ARMS}, "sources": sources, "rows": rows, "not_measured": not_measured,
            "four_language_mean": "the plain mean of the four languages' rates; uniform 4-bit minus bfloat16 reproduces Q0's mean",
            "task_scored": {a: scores[a][1] for a in BENCHED}, "task_prompts": n_task, "mmlu_accuracy": here,
            "tokens": {a: {"batch1": get(a, "batch1", "new_tokens"), "batch8": get(a, "batch8", "new_tokens")} for a in ARMS},
            "kernels": {a: {"weight_classes": get(a, "kernels", "weight_classes"), "unpack": get(a, "quanto_cuda", "unpack")} for a in QUANTO},
            "job": {k: job.get(k) for k in ("folder", "commit", "started_utc", "finished_utc")} | {"hardware": hw, "installed": job.get("installed")}}


def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


FORMATS = {"mib": lambda v: f"{v:,.0f}", "rate": lambda v: f"{v:.1f}", "ppl": lambda v: f"{v:.2f}", "pct": lambda v: f"{v:.1f}",
           "score": lambda v: f"{v:.2f}"}


def report_md(res: dict) -> str:
    s = res["sources"]
    lines = ["### The base model in four loadings: memory, speed and the benchmarks beside them (section 22)", "",
             "Tiny Aya Global with no adapter. bfloat16; NF4 (bitsandbytes 4-bit, double quantization, compute in bfloat16, the loader "
             "of section 12.2); uniform 4-bit and the circuit plan (3 layers at 8 bits, 27 at 4 and 6 at 2, the tier size of section 15) "
             "through CircuitKIT with optimum-quanto, weights only, applied on the CPU. Every figure is marked with the run it comes from:", ""]
    for key, src in s.items():
        extra = "; ".join(x for x in (src.get("detail"), src.get("weights")) if x)
        where = f", upload `{src['folder']}`, commit `{(src.get('commit') or '')[:7]}`" if key == "d" and src.get("folder") else ""
        lines.append(f"- ({key}) {src['run']}: {src['hardware']}" + (f"; {extra}" if extra else "") + where + ".")
    lines += ["", "| | " + " | ".join(res["columns"].values()) + " |", "|---|" + "---:|" * len(res["columns"])]
    for r in res["rows"]:
        name = r["row"] + (f" ({r['unit']})" if r["unit"] else "")
        lines.append(f"| {name} | " + " | ".join(" / ".join(f"{FORMATS[r['format']](v)} ({src})" for v, src in r["cells"][a])
                                               for a in res["columns"]) + " |")
    tok = res["tokens"]
    counts = "; ".join(f"{res['columns'][a]} {tok[a]['batch1']:,} at batch 1" for a in res["columns"] if tok[a]["batch1"] is not None)
    lines += ["", "Memory is torch.cuda.memory_allocated() after loading and the peak of max_memory_allocated() during a phase; "
              "the CUDA context and the allocator's cache are not in it. Speed: greedy generation of at most 256 new tokens for "
              "the first 50 task prompts in file order. At batch 1 one prompt at a time; at batch 8 the same prompts sorted "
              "longest first, as lir.generate sorts them, with left padding. Before each, one warm-up call on prompts outside "
              "the 50 is made and not counted. Output tokens per "
              "second is the new tokens, counted as lir.generate counts them, over the seconds of the generate calls with the GPU "
              "synchronized." + (f" The answers differ in length between arms, so the figures rest on different numbers of tokens: {counts}."
                                 if counts else ""),
              "", "The strict attack success over the four languages is not stored in quantization.json or gpu.json: it is the plain "
              "mean of the four languages' rates, which have the same 291 prompt families each; for section 15 uniform 4-bit minus "
              "bfloat16 of these means is Q0's mean."]
    lines += ["", "On this job's GPU the task score is the mean over the answers the judge scored: " + ", ".join(
        f"{res['columns'][a]} {n} of {res['task_prompts']}" for a, n in res["task_scored"].items()) + ". Section 15 read an arm's "
        "task score only when all 300 were scored."]
    k = res["kernels"]
    if any(v["weight_classes"] for v in k.values()):
        lines += ["", "The quanto weights on the GPU, by tier (class of the weight tensor and number of layers): " + "; ".join(
            f"{res['columns'][a]}: " + ", ".join(f"{t} " + ", ".join(f"{c} {n}" for c, n in v.items()) for t, v in (k[a]["weight_classes"] or {}).items())
            + (f"; 2-bit unpacking by {k[a]['unpack']}" if a == "circuit" and k[a]["unpack"] else "") for a in k if k[a]["weight_classes"]) + "."]
    lines += ["", "Not measured on this job's GPU: the safety sets, and the task score and Global-MMLU-Lite of uniform 4-bit and the "
              "circuit plan. The table holds the base model only."]
    if res["not_measured"]:
        lines += ["", "Not measured in this job: " + "; ".join(f"`{a}`: {why}" for a, why in res["not_measured"].items()) + "."]
    return "\n".join(lines)


def report(a) -> None:
    from lir import gpu

    folder = gpu.newest(JOB)
    if folder is None:
        print("pending, nothing written: no upload of the job in runs/gpu (run `python -m lir.gpu fetch` first)")
        return
    job_rec = dict(json.loads((folder / "job.json").read_text()), folder=folder.name)
    rows = lambda name: read_jsonl(folder / name) if (folder / name).exists() else None
    res = table(job_rec, json.loads((OUT / "quantization.json").read_text()), json.loads((OUT / "quantization_mmlu.json").read_text()),
                json.loads((OUT / "gpu.json").read_text()), {a: rows(f"{a}.mmlu.jsonl") for a in BENCHED},
                {a: rows(f"{a}.task.judge.jsonl") for a in BENCHED})
    text = report_md(res)
    write_json(OUT / "deployment.json", res)
    (OUT / "deployment.md").write_text(text + "\n")
    print(text)
    print("\nArms not measured: " + ("; ".join(f"{a}: {why}" for a, why in res["not_measured"].items()) or "none"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["launch", "job", "arm", "report"])
    ap.add_argument("--flavor", default=FLAVOR)
    ap.add_argument("--timeout", default=TIMEOUT)
    ap.add_argument("--image", default=IMAGE, help="launch: the container image")
    ap.add_argument("--dry-run", action="store_true", help="launch: print the job and submit nothing")
    ap.add_argument("--arm", choices=ARMS, help="arm: the arm this process builds and measures (used inside the job)")
    ap.add_argument("--out", help="arm: the job's folder")
    a = ap.parse_args()
    if a.what == "arm" and not (a.arm and a.out):
        raise SystemExit("arm needs --arm and --out")
    {"launch": launch, "job": job, "arm": arm, "report": report}[a.what](a)


if __name__ == "__main__":
    main()
