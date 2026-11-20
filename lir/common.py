"""Shared settings and small helpers for the Lost in Refusal pipeline."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lir"
RUNS = ROOT / "runs"

SEED = 42
TRAIN_LANGS = ["en", "ko", "zh"]
EVAL_LANGS = ["en", "ko", "zh", "sw"]  # sw is evaluation-only

# LIR_MODEL=earth switches every script to Tiny Aya Earth; the default is Tiny Aya Global
MODELS = {"global": ("CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324"),
          "earth": ("CohereLabs/tiny-aya-earth", "0624fafbe200f930b89566fc4ec4da314ba48d89")}
MODEL_KEY = os.environ.get("LIR_MODEL", "global")
MODEL, MODEL_REV = MODELS[MODEL_KEY]
GUARD = "Qwen/Qwen3Guard-Gen-4B"
GUARD_REV = "6ec42827da0c1ff11e7a49dc269d2e810d27e108"
JUDGE = "Qwen/Qwen3-14B"

BACTRIAN = ("MBZUAI/Bactrian-X", "3698480a001cb3f62c61c314d4acb181eb31e983")
MULTIJAIL = ("DAMO-NLP-SG/MultiJail", "93d65c778925d55973b9ff30c7bccc8545494eb6")
XSTEST = ("Paul/XSTest", None)  # revision resolved at download time and recorded

HF_ORG = "lost-in-refusal"


def rank_key(source_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{source_id}".encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def git_commit() -> str | None:
    """Current commit of the repo, with -dirty if tracked files have changes."""
    import subprocess

    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return rev + ("-dirty" if dirty else "")


def device() -> str:
    import torch

    if os.environ.get("LIR_DEVICE"):
        return os.environ["LIR_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
