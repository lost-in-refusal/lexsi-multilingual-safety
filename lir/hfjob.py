"""Runs on a Hugging Face GPU for preregistration sections 12.0 to 12.2 on Aish's branch (deliverable D20).

  python -m lir.hfjob launch --job control --dry-run    # print the job, submit nothing
  python -m lir.hfjob launch --job control              # submit it; needs HF_TOKEN with Jobs rights on the namespace
  python -m lir.hfjob run --job control                 # what the job runs inside the container

A job clones this repository at one pushed commit, installs requirements-gpu.txt, trains or downloads each arm's
adapter, evaluates it with the same lir.evaluate as on the laptop, and uploads answers, verdicts, run records, logs
and a hardware record to a private staging dataset. Nothing goes to the public outputs repository from a job.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from lir.common import DATA, HF_ORG, ROOT, RUNS, write_json

NAMESPACE = "lost-in-refusal-lexsi"   # the organization whose credits pay for the job
FLAVOR, TIMEOUT, IMAGE = "l4x1", "3h", "python:3.12"
REPO_URL = "https://github.com/lost-in-refusal/lexsi-multilingual-safety.git"
STAGING = os.environ.get("LIR_STAGING_REPO", f"{HF_ORG}/lost-in-refusal-gpu-runs")   # private dataset repository
# The seed 42 adapters trained on the laptop are the two stage 1 repositories, not the stage 2 one.
LAPTOP = {"plain": f"{HF_ORG}/tiny-aya-global-bactrian-lora-plain", "safegrad": f"{HF_ORG}/tiny-aya-global-bactrian-lora-safegrad"}
SETS = ["attack", "xstest"]
NF4 = {"LIR_QUANT": "nf4"}

# job -> steps. A step has an arm name, where its adapter comes from (no key: the base model; "hub": a laptop adapter;
# "train": a training file under data/lir) and the environment that changes how the model is loaded.
# Arm names differ from every laptop arm, so a GPU result can never be read as a laptop one.
JOBS = {
    "control": [{"arm": "gpu-base"}, {"arm": "gpu-plain", "hub": "plain"}],                                   # 12.0
    "only-ko": [{"arm": "plain-ko", "train": "bactrian_train_only_ko.jsonl"}],                                  # 12.1
    "only-en": [{"arm": "plain-en", "train": "bactrian_train_only_en.jsonl"}],
    "only-zh": [{"arm": "plain-zh", "train": "bactrian_train_only_zh.jsonl"}],
    "nf4": [{"arm": "gpu-safegrad", "hub": "safegrad"}, {"arm": "gpu-base-nf4", "env": NF4},                    # 12.2
            {"arm": "gpu-plain-nf4", "hub": "plain", "env": NF4}, {"arm": "gpu-safegrad-nf4", "hub": "safegrad", "env": NF4}],
    # Not registered. bfloat16 with the adapter left unmerged, as under NF4; run only if it is added to 12.2 first.
    "unmerged": [{"arm": "gpu-plain-unmerged", "hub": "plain", "env": {"LIR_MERGE": "0"}},
                 {"arm": "gpu-safegrad-unmerged", "hub": "safegrad", "env": {"LIR_MERGE": "0"}}],
}


# ---------------------------------------------------------------- on the laptop: build and submit the job

def job_command(job: str, commit: str, flavor: str = FLAVOR, timeout: str = TIMEOUT, namespace: str = NAMESPACE,
                dry_run: bool = False) -> list[str]:
    """The `hf jobs run` command for one job. The token is passed by name only and is read from the environment."""
    inner = (f"set -euo pipefail; git clone -q {REPO_URL} repo; cd repo; git checkout -q {commit}; "
             f"pip install -q -r requirements-gpu.txt; python -m lir.hfjob run --job {job}")
    return (["hf", "jobs", "run", "--flavor", flavor, "--namespace", namespace, "--timeout", timeout, "--secrets", "HF_TOKEN",
             "--env", f"LIR_FLAVOR={flavor}", "--env", f"LIR_STAGING_REPO={STAGING}", "--name", f"lir-{job}", "--detach"]
            + (["--dry-run"] if dry_run else []) + [IMAGE, "bash", "-c", inner])


def launch(a) -> None:
    git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked files have uncommitted changes: the job would not run the code you see")
    if not git("branch", "-r", "--contains", commit):
        raise SystemExit(f"commit {commit[:7]} is not on the remote yet: push it first, the job clones from GitHub")
    cmd = job_command(a.job, commit, a.flavor, a.timeout, a.namespace, a.dry_run)
    print(" ".join(cmd[:-1]) + f" '{cmd[-1]}'")
    if not a.dry_run and not os.environ.get("HF_TOKEN"):
        raise SystemExit("set HF_TOKEN in this shell first (a token with Jobs rights on the namespace and access to Tiny Aya)")
    sys.exit(subprocess.run(cmd, cwd=ROOT).returncode)


# ---------------------------------------------------------------- inside the job

def hardware() -> dict:
    """GPU type, driver and library versions for the run record (section 12.0 asks for them in every record)."""
    import platform
    from importlib import metadata

    import torch

    def smi(fields: str) -> list[str]:
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"], capture_output=True, text=True)
            return [line.strip() for line in out.stdout.splitlines() if line.strip()]
        except FileNotFoundError:
            return []

    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {"gpus": smi("name,driver_version,memory.total"), "flavor": os.environ.get("LIR_FLAVOR"),
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "python": platform.python_version(), "platform": platform.platform(),
            "libraries": {n: version(n) for n in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors",
                                                  "tokenizers", "huggingface_hub", "numpy")}}


def speeds(log: str) -> dict[str, float]:
    """Generation speed per set, from the last progress line lir.generate printed for it."""
    out = {}
    for tag, rate in re.findall(r"\[([^\]]+)\] \d+/\d+ bs=\d+ batch [\d.]+s \| ([\d.]+) tok/s overall", log):
        out[tag] = float(rate)
    return out


def run_step(cmd: list[str], env: dict, log: Path) -> dict:
    """Run one command, keep its output, and record seconds and the peak GPU memory in use while it ran."""
    peak, stop = [0], threading.Event()

    def watch() -> None:
        while not stop.wait(5):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True).stdout.split()
                peak[0] = max(peak[0], *(int(x) for x in out)) if out else peak[0]
            except (FileNotFoundError, ValueError):
                return

    threading.Thread(target=watch, daemon=True).start()
    t0 = time.time()
    with log.open("w") as f:
        code = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT).returncode
    stop.set()
    text = log.read_text()
    print(text[-2000:], flush=True)
    if code:
        raise SystemExit(f"{' '.join(cmd)} failed with code {code}; see {log.name}")
    return {"command": " ".join(cmd), "env": env, "seconds": round(time.time() - t0, 1), "peak_gpu_mib": peak[0] or None,
            "tokens_per_second": speeds(text)}


def run(a) -> None:
    from huggingface_hub import HfApi, snapshot_download

    from lir.common import git_commit

    hw = hardware()
    if not hw["cuda_available"] and not os.environ.get("LIR_ALLOW_CPU"):
        raise SystemExit(f"no CUDA GPU in this job: {hw}")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = ROOT / "runs" / "hfjob" / a.job
    (out / "logs").mkdir(parents=True, exist_ok=True)
    record = {"job": a.job, "started_utc": stamp, "commit": git_commit(), "sets": SETS, "hardware": hw, "steps": []}
    py = [sys.executable, "-m"]
    for step in JOBS[a.job]:
        arm, env, adapter = step["arm"], step.get("env", {}), None
        if "hub" in step:   # an adapter trained on the laptop, not retrained
            adapter = snapshot_download(LAPTOP[step["hub"]], local_dir=RUNS / arm / "adapter",
                                        allow_patterns=["adapter_config.json", "adapter_model.safetensors"])
        if "train" in step:
            done = run_step(py + ["lir.train", "--arm", "plain", "--output", str(RUNS / arm), "--train-file",
                                  str(DATA / step["train"])], {}, out / "logs" / f"{arm}.train.log")
            record["steps"].append(dict(done, arm=arm, what="train"))
            adapter = str(RUNS / arm / "adapter")
        cmd = py + ["lir.evaluate", "--arm", arm, "--sets", *SETS] + (["--adapter", str(adapter)] if adapter else [])
        done = run_step(cmd, env, out / "logs" / f"{arm}.evaluate.log")
        record["steps"].append(dict(done, arm=arm, what="evaluate", adapter_from=step.get("hub") or step.get("train")))
        write_json(out / "job.json", record)   # kept current, so a job that dies later still leaves a record
    record["finished_utc"] = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    write_json(out / "job.json", record)
    api, dest = HfApi(), f"{a.job}/{stamp}-{(record['commit'] or 'unknown')[:7]}"
    api.create_repo(STAGING, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=STAGING, repo_type="dataset", path_in_repo=dest, commit_message=f"{a.job}: job record and logs")
    for step in JOBS[a.job]:
        arm = step["arm"]
        api.upload_folder(folder_path=RUNS / "eval" / arm, repo_id=STAGING, repo_type="dataset", path_in_repo=f"{dest}/{arm}",
                          commit_message=f"{a.job}: answers and verdicts of {arm}")
        if "train" in step:   # a new adapter with its run record and training log; tokenizer files are the base model's
            api.upload_folder(folder_path=RUNS / arm, repo_id=STAGING, repo_type="dataset", path_in_repo=f"{dest}/{arm}/run",
                              allow_patterns=["run.json", "train_log.jsonl", "adapter/adapter_config.json",
                                              "adapter/adapter_model.safetensors"], commit_message=f"{a.job}: adapter of {arm}")
    print(f"uploaded to {STAGING}/{dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["launch", "run"])
    ap.add_argument("--job", required=True, choices=sorted(JOBS))
    ap.add_argument("--flavor", default=FLAVOR)
    ap.add_argument("--timeout", default=TIMEOUT)
    ap.add_argument("--namespace", default=NAMESPACE)
    ap.add_argument("--dry-run", action="store_true", help="launch: print the job and submit nothing")
    a = ap.parse_args()
    (launch if a.what == "launch" else run)(a)


if __name__ == "__main__":
    main()
