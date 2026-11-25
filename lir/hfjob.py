"""Runs on a Hugging Face GPU for preregistration sections 12.0 to 12.2 and 14f (deliverable D20), and the main
experiment on a model outside the Tiny Aya family (the granite41 jobs; read by lir.granite41).

  python -m lir.hfjob launch --job control --dry-run    # print the job, submit nothing
  python -m lir.hfjob launch --job control              # submit it; needs a login with Jobs rights on the namespace
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
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from lir.common import DATA, HF_ORG, ROOT, RUNS, read_jsonl, sha256_file, write_json

NAMESPACE = "lost-in-refusal-lexsi"   # the organization whose credits pay for the job
FLAVOR, TIMEOUT, IMAGE = "l4x1", "3h", "python:3.12"
REPO_URL = "https://github.com/lost-in-refusal/lexsi-multilingual-safety.git"
STAGING = os.environ.get("LIR_STAGING_REPO", f"{HF_ORG}/lost-in-refusal-gpu-runs")   # private dataset repository
# The seed 42 adapters trained on the laptop are the two stage 1 repositories, not the stage 2 one. Each is pinned
# to a revision, and the downloaded weights must have the hash of the file the laptop evaluated.
LAPTOP = {"plain": (f"{HF_ORG}/tiny-aya-global-bactrian-lora-plain", "0e57114e1804beb4c81a3afad0076d9733067010",
                    "f244c893ba78532f94e68658b86a290abd90ef319536854908ac867b5b5b3f83"),
          "safegrad": (f"{HF_ORG}/tiny-aya-global-bactrian-lora-safegrad", "8b2687f885d388e4a3a370fb04252a21c3f13e99",
                       "dca9b81486c46d1ba683fbafaa0ca933058467ade7a7baa09eedab36e10fac16")}
# The seed 1 and 2 adapters trained on the laptop are folders of the stage 2 adapter repository, pinned the same way:
# one revision of the repository, and for each folder the hash of the weights the laptop evaluated.
STAGE2_REPO, STAGE2_REV = f"{HF_ORG}/tiny-aya-lora-stage2", "292f8ba92163b3a2ac58ce625046f16516d0c70b"
STAGE2 = {"plain-s1": "02374672ba14d67abb79f5fab8a46a0abfe41f35a3301d428554cbcb3a8b2a17",
          "safegrad-s1": "7c8b94797bd44e8000be4e272e9330d2b4bdca71d414a8d8ac2b676e195d0e04",
          "plain-s2": "d71a63e21def479901fd7adee5f840960bdb22a1c69f51eededb90bea6e5e10f",
          "safegrad-s2": "ccc2d4868b5d2479fb83545981057c75ebba1547c63678a4804aafd428a36848"}
SETS = ["attack", "xstest"]
NF4 = {"LIR_QUANT": "nf4"}
UNMERGED = {"LIR_MERGE": "0"}
# The main experiment on a model outside the Tiny Aya family, by its key in lir.common.MODELS: the base model, then
# plain and SafeGrad at three seeds, one training a job, so that the six can run side by side.
OTHER, OTHER_SEEDS = "granite41", (42, 1, 2)
# SafeGrad needs SafeTune, which requirements-gpu.txt leaves out. A job that trains SafeGrad installs
# requirements-gpu-safegrad.txt and then SafeTune without its dependencies, at the commit the laptop trains with.
SAFETUNE = ("https://github.com/Lexsi-Labs/SafeTune.git", "3901a9f48a88984a181c8c9493be2d3a20ab160f")

# job -> steps. A step has an arm name, where its adapter comes from (no key: the base model; "hub": a laptop adapter
# of seed 42; "stage2": a laptop adapter of seed 1 or 2; "train": a training file under data/lir) and the environment
# that changes how the model is loaded.
# Arm names differ from every laptop arm, so a GPU result can never be read as a laptop one.
# A step on another model also has "model", its key, set as LIR_MODEL for training and generation. It may have "sets"
# (the sets it answers; SETS without it), "method" and "seed" (what lir.train is called with; plain at its own seed
# without them), "refusals" (the file its safe refusals to the held-out prompts are written to) and "align" (the job
# whose refusal file a SafeGrad step trains against).
JOBS = {
    "control": [{"arm": "gpu-base"}, {"arm": "gpu-plain", "hub": "plain"}],                                   # 12.0
    "only-ko": [{"arm": "plain-ko", "train": "bactrian_train_only_ko.jsonl"}],                                  # 12.1
    "only-en": [{"arm": "plain-en", "train": "bactrian_train_only_en.jsonl"}],
    "only-zh": [{"arm": "plain-zh", "train": "bactrian_train_only_zh.jsonl"}],
    "nf4": [{"arm": "gpu-safegrad", "hub": "safegrad"}, {"arm": "gpu-base-nf4", "env": NF4},                    # 12.2
            {"arm": "gpu-plain-nf4", "hub": "plain", "env": NF4}, {"arm": "gpu-safegrad-nf4", "hub": "safegrad", "env": NF4}],
    # Section 14f. bfloat16 with the adapter left unmerged, as under NF4: the review note under 12.2 names these arms.
    "unmerged": [{"arm": "gpu-plain-unmerged", "hub": "plain", "env": UNMERGED},
                 {"arm": "gpu-safegrad-unmerged", "hub": "safegrad", "env": UNMERGED}],
    # Section 14f. Seeds 1 and 2, plain and SafeGrad: bfloat16 with the adapter unmerged, and NF4. Two arms a job.
    **{f"{how}-s{seed}": [{"arm": f"gpu-{arm}-s{seed}-{how}", "stage2": f"{arm}-s{seed}", "env": env} for arm in ("plain", "safegrad")]
       for seed in (1, 2) for how, env in (("unmerged", UNMERGED), ("nf4", NF4))},
    # The model outside the Tiny Aya family. The base job also answers the held-out prompts and writes the model's own
    # refusal set from them. Each SafeGrad job trains against that file, taken from the base job's upload.
    f"{OTHER}-base": [{"arm": f"{OTHER}-base", "model": OTHER, "sets": ["alignment", *SETS], "refusals": f"alignment_sft_{OTHER}.jsonl"}],
    **{f"{OTHER}-{method}-s{seed}": [{"arm": f"{OTHER}-{method}-s{seed}", "model": OTHER, "train": "bactrian_train.jsonl", "method": method,
                                      "seed": seed, **({"align": f"{OTHER}-base"} if method == "safegrad" else {})}]
       for method in ("plain", "safegrad") for seed in OTHER_SEEDS},
    # Section 16b. Plain LoRA on that model with no Korean training row, at the three seeds.
    **{f"{OTHER}-noko-s{seed}": [{"arm": f"{OTHER}-noko-s{seed}", "model": OTHER, "train": "bactrian_train_noko.jsonl", "method": "plain",
                                  "seed": seed}] for seed in OTHER_SEEDS},
}


# ---------------------------------------------------------------- on the laptop: build and submit the job

def trains_safegrad(job: str) -> bool:
    return any(step.get("method") == "safegrad" for step in JOBS[job])


def install_command(job: str) -> str:
    """What a job installs before it runs. Only a job that trains SafeGrad adds SafeTune, at its pinned commit."""
    extra = (f"pip install -q -r requirements-gpu-safegrad.txt; pip install -q --no-deps git+{SAFETUNE[0]}@{SAFETUNE[1]}; "
             if trains_safegrad(job) else "")
    return "pip install -q -r requirements-gpu.txt; " + extra


def job_command(job: str, commit: str, flavor: str = FLAVOR, timeout: str = TIMEOUT, namespace: str = NAMESPACE,
                dry_run: bool = False) -> list[str]:
    """The `hf jobs run` command for one job. The token is passed by name only and is read from the environment."""
    inner = (f"set -euo pipefail; git clone -q {REPO_URL} repo; cd repo; git checkout -q {commit}; "
             f"{install_command(job)}python -m lir.hfjob run --job {job}")
    return (["hf", "jobs", "run", "--flavor", flavor, "--namespace", namespace, "--timeout", timeout, "--secrets", "HF_TOKEN",
             "--env", f"LIR_FLAVOR={flavor}", "--env", f"LIR_STAGING_REPO={STAGING}", "--name", f"lir-{job}", "--detach"]
            + (["--dry-run"] if dry_run else []) + [IMAGE, "bash", "-c", inner])


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
    cmd = job_command(a.job, commit, a.flavor, a.timeout, a.namespace, a.dry_run)
    print(" ".join(cmd[:-1]) + f" '{cmd[-1]}'")
    for why in stops:   # a dry run prints the job all the same, and says what a launch would stop on
        print(f"a launch would stop here: {why}")
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
        while True:   # the first sample is taken at once, then one every 5 seconds until the command ends
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True).stdout.split()
                peak[0] = max(peak[0], *(int(x) for x in out)) if out else peak[0]
            except (FileNotFoundError, ValueError):
                return
            if stop.wait(5):
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


def laptop_adapter(step: dict, arm: str, download=None) -> str | None:
    """The folder of the adapter a step evaluates when it was trained on the laptop, downloaded at its pinned revision;
    None for the base model and for a step that trains. Stops if the weights are not the file the laptop evaluated."""
    if "hub" in step:
        repo, revision, sha = LAPTOP[step["hub"]]
        folder, sub = RUNS / arm / "adapter", ""
    elif "stage2" in step:
        repo, revision, sha = STAGE2_REPO, STAGE2_REV, STAGE2[step["stage2"]]
        folder, sub = RUNS / arm / "hub", step["stage2"] + "/"
    else:
        return None
    if download is None:
        from huggingface_hub import snapshot_download as download
    got = Path(download(repo, revision=revision, local_dir=folder,
                        allow_patterns=[sub + "adapter_config.json", sub + "adapter_model.safetensors"])) / sub
    if sha256_file(got / "adapter_model.safetensors") != sha:
        raise SystemExit(f"{repo}@{revision[:8]} {sub}: the adapter weights are not the file the laptop evaluated")
    return str(got)


def model_env(step: dict) -> dict:
    """LIR_MODEL for a step on another model; nothing for Tiny Aya Global, which every script loads by default."""
    return {"LIR_MODEL": step["model"]} if "model" in step else {}


def step_sets(step: dict) -> list[str]:
    return step.get("sets", SETS)


def train_command(step: dict, align: Path | None = None) -> list[str]:
    """The lir.train call of a step that trains. align: the refusal file of a SafeGrad step."""
    cmd = ["lir.train", "--arm", step.get("method", "plain"), "--output", str(RUNS / step["arm"]), "--train-file", str(DATA / step["train"])]
    if "seed" in step:
        cmd += ["--seed", str(step["seed"])]
    if step.get("method") == "safegrad":
        cmd += ["--align-file", str(align)]
    return cmd


def refusal_set(step: dict, out: Path, files: list[str] | None = None, download=None) -> tuple[Path, dict]:
    """The refusal file a SafeGrad step trains against: the one the newest upload of the base job holds in the staging
    dataset, copied into the job's folder so that it is uploaded with the record. Stops if there is none or it is empty.
    files, download: the paths in the staging dataset and a function that fetches one, for tests."""
    job = step["align"]
    name = JOBS[job][0]["refusals"]
    if files is None:
        from huggingface_hub import HfApi, hf_hub_download

        files = HfApi().list_repo_files(STAGING, repo_type="dataset")
        download = lambda path: hf_hub_download(STAGING, path, repo_type="dataset", local_dir=RUNS / "staging")
    found = sorted(f for f in files if f.startswith(f"{job}/") and f.endswith(f"/{name}") and f.count("/") == 2)
    if not found:
        raise SystemExit(f"no {name} under {job}/ in {STAGING}: job {job} writes it, so run that job first")
    got = Path(download(found[-1]))   # folders are named <UTC start>-<commit>, so the last name is the newest
    rows = read_jsonl(got)
    if not rows:
        raise SystemExit(f"{STAGING}/{found[-1]} holds no refusal: SafeGrad has nothing to train against")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(got, out / name)
    return out / name, {"file": name, "from": f"{STAGING}/{found[-1]}", "rows": len(rows), "sha256": sha256_file(out / name)}


def run(a) -> None:
    from huggingface_hub import HfApi

    from lir.common import git_commit

    hw = hardware()
    if not hw["cuda_available"] and not os.environ.get("LIR_ALLOW_CPU"):
        raise SystemExit(f"no CUDA GPU in this job: {hw}")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = ROOT / "runs" / "hfjob" / a.job
    (out / "logs").mkdir(parents=True, exist_ok=True)
    record = {"job": a.job, "started_utc": stamp, "commit": git_commit(), "sets": SETS, "hardware": hw, "steps": []}
    if trains_safegrad(a.job):
        from importlib import metadata

        record["safetune"] = {"repository": SAFETUNE[0], "commit": SAFETUNE[1], "installed_version": metadata.version("safetune")}
    py = [sys.executable, "-m"]
    for step in JOBS[a.job]:
        arm, env = step["arm"], {**model_env(step), **step.get("env", {})}
        adapter = laptop_adapter(step, arm)   # an adapter trained on the laptop, not retrained
        if "train" in step:
            align = None
            if step.get("method") == "safegrad":   # the model's own refusals, from the base job's upload
                align, record["refusal_set"] = refusal_set(step, out)
            done = run_step(py + train_command(step, align), model_env(step), out / "logs" / f"{arm}.train.log")
            record["steps"].append(dict(done, arm=arm, what="train"))
            adapter = str(RUNS / arm / "adapter")
        # lir.evaluate generates and then judges. Here its two halves run as two processes, with the same functions,
        # so that the peak GPU memory of the first is the model's alone and does not include the guard.
        cmd = py + ["lir.evaluate", "--arm", arm, "--sets", *step_sets(step), "--skip-guard"] + (["--adapter", str(adapter)] if adapter else [])
        done = run_step(cmd, env, out / "logs" / f"{arm}.generate.log")
        record["steps"].append(dict(done, arm=arm, what="generate", adapter_from=step.get("hub") or step.get("stage2") or step.get("train"),
                                    **({"sets": step["sets"]} if "sets" in step else {})))
        done = run_step(py + ["lir.hfjob", "guard", "--job", a.job, "--arm", arm], {}, out / "logs" / f"{arm}.guard.log")
        record["steps"].append(dict(done, arm=arm, what="guard"))
        if "refusals" in step:   # last in its job: it rewrites a tracked file, the manifest under data/lir
            name = step["refusals"]
            done = run_step(py + ["lir.stage2", "refusals", "--arm", arm, "--out", name], {}, out / "logs" / f"{arm}.refusals.log")
            shutil.copyfile(DATA / name, out / name)   # uploaded with the record; the SafeGrad jobs take it from there
            record["steps"].append(dict(done, arm=arm, what="refusals", file=name, rows=len(read_jsonl(out / name)), sha256=sha256_file(out / name)))
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
                              allow_patterns=["run.json", "train_log.jsonl", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"]
                                             + (["surgery_log.jsonl"] if step.get("method") == "safegrad" else []),
                              commit_message=f"{a.job}: adapter of {arm}")
    print(f"uploaded to {STAGING}/{dest}")


def guard_sets(a) -> None:
    """The second half of lir.evaluate: the guard's verdict on every answer the arm generated."""
    from lir import guard
    from lir.common import read_jsonl

    tok, model = guard.load()
    for name in next((step_sets(step) for step in JOBS[a.job] if step["arm"] == a.arm), SETS):
        guard.judge(tok, model, read_jsonl(RUNS / "eval" / a.arm / f"{name}.gen.jsonl"), RUNS / "eval" / a.arm / f"{name}.guard.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["launch", "run", "guard"])
    ap.add_argument("--job", required=True, choices=sorted(JOBS))
    ap.add_argument("--arm", help="guard: the arm whose answers are judged (used inside a job)")
    ap.add_argument("--flavor", default=FLAVOR)
    ap.add_argument("--timeout", default=TIMEOUT)
    ap.add_argument("--namespace", default=NAMESPACE)
    ap.add_argument("--dry-run", action="store_true", help="launch: print the job and submit nothing")
    a = ap.parse_args()
    {"launch": launch, "run": run, "guard": guard_sets}[a.what](a)


if __name__ == "__main__":
    main()
