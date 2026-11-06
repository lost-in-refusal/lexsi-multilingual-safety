"""Upload datasets, adapters and evaluation outputs to the lost-in-refusal HF org.

Repos are created private; making them public is a separate, deliberate step
(`--public`), planned for the Nov 13 freeze.

  python -m lir.publish data
  python -m lir.publish evals --arms base plain safegrad
  python -m lir.publish adapter --arm plain
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from lir.common import DATA, HF_ORG, ROOT, RUNS

DATA_REPO = f"{HF_ORG}/lost-in-refusal-data"
EVAL_REPO = f"{HF_ORG}/lost-in-refusal-eval-outputs"
CARDS = ROOT / "docs" / "lir" / "cards"


def _repo(api: HfApi, repo_id: str, kind: str, public: bool) -> None:
    api.create_repo(repo_id, repo_type=kind, private=not public, exist_ok=True)


def push_data(api: HfApi, public: bool) -> None:
    _repo(api, DATA_REPO, "dataset", public)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for p in DATA.glob("*.json*"):
            if p.name.startswith(("bactrian_candidates", "bactrian_guard")):
                (t / "screening").mkdir(exist_ok=True)
                shutil.copy(p, t / "screening" / p.name)
            else:
                shutil.copy(p, t / p.name)
        shutil.copy(CARDS / "data.md", t / "README.md")
        api.upload_folder(folder_path=t, repo_id=DATA_REPO, repo_type="dataset",
                          commit_message="Frozen datasets")
    print("pushed", DATA_REPO)


def push_evals(api: HfApi, arms: list[str], public: bool) -> None:
    _repo(api, EVAL_REPO, "dataset", public)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for arm in arms:
            shutil.copytree(RUNS / "eval" / arm, t / arm)
        res = ROOT / "results" / "lir"
        for p in res.glob("*.json"):
            shutil.copy(p, t / p.name)
        if (res / "results.md").exists():
            shutil.copy(res / "results.md", t / "results.md")
        shutil.copy(CARDS / "evals.md", t / "README.md")
        api.upload_folder(folder_path=t, repo_id=EVAL_REPO, repo_type="dataset",
                          commit_message=f"Evaluation outputs: {', '.join(arms)}")
    print("pushed", EVAL_REPO)


def push_adapter(api: HfApi, arm: str, public: bool) -> None:
    repo_id = f"{HF_ORG}/tiny-aya-global-bactrian-lora-{arm}"
    _repo(api, repo_id, "model", public)
    run = RUNS / arm
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        shutil.copytree(run / "adapter", t, dirs_exist_ok=True)
        for name in ["run.json", "train_log.jsonl", "surgery_log.jsonl"]:
            if (run / name).exists():
                shutil.copy(run / name, t / name)
        shutil.copy(CARDS / f"adapter-{arm}.md", t / "README.md")
        api.upload_folder(folder_path=t, repo_id=repo_id, commit_message=f"{arm} LoRA adapter")
    print("pushed", repo_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "evals", "adapter"])
    ap.add_argument("--arm")
    ap.add_argument("--arms", nargs="+", default=["base", "plain", "safegrad"])
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    api = HfApi()
    if a.what == "data":
        push_data(api, a.public)
    elif a.what == "evals":
        push_evals(api, a.arms, a.public)
    else:
        push_adapter(api, a.arm, a.public)


if __name__ == "__main__":
    main()
