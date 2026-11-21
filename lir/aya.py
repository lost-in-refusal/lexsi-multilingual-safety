"""A second training set from the Cohere Aya Dataset (preregistration section 10).

  python -m lir.aya candidates    # text filters, cleaning, dedup, length: data/lir/aya_candidates.jsonl
  python -m lir.guard --input data/lir/aya_candidates.jsonl --output data/lir/aya_guard.jsonl --prompt-key instruction
  python -m lir.aya finalize      # after the guard screen: aya_train.jsonl, aya_eval_task.jsonl, the manifest
  python -m lir.aya task --arm plain --adapter runs/plain/adapter   # answers to the held-out Aya prompts

Aya is written by people in each language and is not parallel, so every step runs per language (the Bactrian-X
pipeline in lir.data works on ids shared across languages). The filters, CuratorKIT cleaning and dedup, the
length limit and the guard screen are the same as for Bactrian-X.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from lir.common import DATA, MODEL, MODEL_KEY, MODEL_REV, RUNS, SEED, TRAIN_LANGS, rank_key, read_jsonl, sha256_file, \
    write_json, write_jsonl
from lir.data import REFUSAL_PATTERNS, _cleaner, _samples

AYA = ("CohereLabs/aya_dataset", "f9ea04583f02a8f86404ff6c58bf75fe637df8a2")
NAMES = {"en": "English", "ko": "Korean", "zh": "Simplified Chinese", "sw": "Swahili"}  # sw is evaluation-only
N_EVAL = 100
N_TRAIN_MAX = 600
N_CANDIDATES = 1000


def drop_reason(lang: str, prompt: str, answer: str) -> str | None:
    """The Bactrian-X text filters; Swahili has no refusal pattern, as in stage 1."""
    if not prompt.strip():
        return "empty_prompt"
    if len(answer.strip()) < (20 if lang in ("en", "sw") else 8):
        return "near_empty"
    if lang in REFUSAL_PATTERNS and re.search(REFUSAL_PATTERNS[lang], answer, flags=re.I):
        return "refusal_or_disclaimer"
    return None


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def split_sizes(passed: dict[str, int], n_eval: int = N_EVAL, cap: int = N_TRAIN_MAX) -> int:
    """Training rows per language: the same number in each, set by the scarcest training language after the hold-out."""
    return max(0, min(cap, *(passed[l] - n_eval for l in TRAIN_LANGS)))


def _load() -> tuple[dict[str, list[dict]], str]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo, rev = AYA
    path = Path(hf_hub_download(repo, "data/train-00000-of-00001.parquet", repo_type="dataset", revision=rev))
    t = pq.read_table(path).to_pydict()
    code = {v: k for k, v in NAMES.items()}
    rows = {l: [] for l in NAMES}
    for i, name in enumerate(t["language"]):
        if name in code:
            lang = code[name]
            rows[lang].append({"uid": f"aya-{i}-{lang}", "id": f"aya-{i}", "lang": lang, "instruction": t["inputs"][i],
                               "output": t["targets"][i], "annotation_type": t["annotation_type"][i]})
    return rows, sha256_file(path)


def build_candidates() -> None:
    from curatorkit.normalizers import ExactDeduplicator, MinHashDeduplicator
    from transformers import AutoTokenizer

    from lir.train import encode

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    rows, source_hash = _load()
    taken = {l: {norm(r["instruction"]) for name in ("bactrian_train.jsonl", "bactrian_eval_task.jsonl")
                 for r in read_jsonl(DATA / name) if r["lang"] == l} for l in TRAIN_LANGS}
    cand, report = [], {}
    for lang, rs in rows.items():
        reasons = Counter()
        keep = []
        for r in sorted(rs, key=lambda r: rank_key(r["id"])):
            why = drop_reason(lang, r["instruction"], r["output"])
            if why is None and norm(r["instruction"]) in taken.get(lang, ()):
                why = "same_prompt_as_bactrian"
            if why:
                reasons[why] += 1
            else:
                keep.append(r)
        clean = _cleaner().run(_samples(keep, f"hf://datasets/{AYA[0]}@{AYA[1]}"))
        n = len(clean)
        clean = MinHashDeduplicator(threshold=0.7).run(ExactDeduplicator().run(clean))
        reasons["dedup_removed"] = n - len(clean)
        fits = [s for s in clean if encode(tok, s.instruction, s.output) is not None]
        reasons["over_max_len"] = len(clean) - len(fits)
        for s in fits[:N_CANDIDATES]:
            cand.append({"uid": s.id, "id": s.metadata["id"], "lang": lang, "instruction": s.instruction,
                         "response": s.output, "annotation_type": s.metadata["annotation_type"],
                         "source": f"{AYA[0]}@{AYA[1]}:{NAMES[lang]}"})
        report[lang] = {"source_rows": len(rs), "after_filters": len(fits), "candidates": min(len(fits), N_CANDIDATES),
                        "drop_reasons": dict(reasons)}
        print(lang, report[lang], flush=True)
    write_jsonl(DATA / "aya_candidates.jsonl", cand)
    write_json(DATA / "aya_candidates_manifest.json", {
        "source": AYA[0], "revision": AYA[1], "split": "train", "license": "apache-2.0", "source_sha256": source_hash,
        "languages": NAMES, "per_language": report, "ranking": f"sha256('{SEED}:aya-<row>') ascending",
        "curation": "text filters as for Bactrian-X; CuratorKIT TextCleaner, ExactDeduplicator, MinHashDeduplicator(0.7) "
                    "per language; rows over the trainer's 1,024-token limit dropped",
        "sha256": sha256_file(DATA / "aya_candidates.jsonl")})


def build_final() -> None:
    from curatorkit.normalizers import MinHashDeduplicator
    from curatorkit.schema import DataSample

    cand = read_jsonl(DATA / "aya_candidates.jsonl")
    screen = {r["uid"]: r for r in read_jsonl(DATA / "aya_guard.jsonl")}
    missing = [r["uid"] for r in cand if not (screen.get(r["uid"]) or {}).get("safety")]
    if missing:
        raise SystemExit(f"{len(missing)} candidates have no guard verdict; nothing written")
    passed, flagged = {l: [] for l in NAMES}, {l: Counter() for l in NAMES}
    for r in cand:
        v = screen[r["uid"]]
        if v["safety"] != "Safe":
            flagged[r["lang"]]["guard_not_safe"] += 1
        elif v["refusal"] == "Yes":
            flagged[r["lang"]]["guard_refusal"] += 1
        else:
            passed[r["lang"]].append(r)
    evals, pools = {}, {}
    for lang, rs in passed.items():
        if len(rs) < N_EVAL:
            raise SystemExit(f"{lang}: only {len(rs)} rows passed the screen, fewer than the {N_EVAL} held-out prompts")
        evals[lang], pool = rs[:N_EVAL], rs[N_EVAL:]
        probe = [DataSample(id=f"e-{r['uid']}", source_uri="eval", instruction=r["instruction"]) for r in evals[lang]]
        rest = [DataSample(id=f"t-{r['uid']}", source_uri="train", instruction=r["instruction"]) for r in pool]
        kept = {s.id[2:] for s in MinHashDeduplicator(threshold=0.7).run(probe + rest) if s.id.startswith("t-")}
        flagged[lang]["train_near_dup_of_eval"] = len(pool) - len(kept)
        pools[lang] = [r for r in pool if r["uid"] in kept]
    n = split_sizes({l: len(pools[l]) + N_EVAL for l in TRAIN_LANGS})
    train = [dict(r, split="train") for l in TRAIN_LANGS for r in pools[l][:n]]
    task = [dict(r, split="eval_task") for l in NAMES for r in evals[l]]
    assert not {r["uid"] for r in train} & {r["uid"] for r in task}
    write_jsonl(DATA / "aya_train.jsonl", train)
    write_jsonl(DATA / "aya_eval_task.jsonl", task)
    # Bactrian-X at the same size: the first n ids of the usual training file, all three languages
    bact = read_jsonl(DATA / "bactrian_train.jsonl")
    ids = list(dict.fromkeys(r["id"] for r in bact))[:n]
    small = DATA / "bactrian_train_ayasize.jsonl"
    write_jsonl(small, [r for r in bact if r["id"] in set(ids)])
    files = ["aya_train.jsonl", "aya_eval_task.jsonl", small.name]
    write_json(DATA / "aya_manifest.json", {
        "source": AYA[0], "revision": AYA[1], "license": "apache-2.0",
        "train_rows_per_language": n, "train_languages": TRAIN_LANGS, "eval_task_rows_per_language": N_EVAL,
        "eval_languages": list(NAMES), "passed_screen": {l: len(passed[l]) for l in NAMES},
        "screening": {l: dict(c) for l, c in flagged.items()},
        "annotation_types_train": {l: dict(Counter(r["annotation_type"] for r in train if r["lang"] == l)) for l in TRAIN_LANGS},
        "guard": "Qwen3Guard-Gen-4B response moderation on (prompt, answer); kept when Safe and not a refusal",
        "size_matched_bactrian": {"file": small.name, "ids": len(ids)},
        "files": {p: sha256_file(DATA / p) for p in files}})
    print("train rows per language", n, "| passed", {l: len(passed[l]) for l in NAMES}, "|", {l: dict(c) for l, c in flagged.items()})


def task(a) -> None:
    """Answers to the held-out Aya prompts for one arm; the task judge (lir.judge) scores them afterwards."""
    if a.arm.startswith("earth-") != (MODEL_KEY == "earth"):
        raise SystemExit(f"arm {a.arm} does not match the loaded model {MODEL} (set LIR_MODEL)")
    rows = [{"uid": r["uid"], "id": r["id"], "lang": r["lang"], "prompt": r["instruction"], "reference": r["response"]}
            for r in read_jsonl(DATA / "aya_eval_task.jsonl")]
    from lir import generate

    tok, model = generate.load_model(a.adapter)
    # kept apart from runs/eval so an arm's folder there stays exactly what the guard and the tables read
    generate.generate(tok, model, rows, RUNS / "ayatask" / a.arm / "gen.jsonl", 1024, tag=f"{a.arm}/ayatask")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("candidates")
    sub.add_parser("finalize")
    t = sub.add_parser("task")
    t.add_argument("--arm", required=True)
    t.add_argument("--adapter")
    a = ap.parse_args()
    {"candidates": lambda a: build_candidates(), "finalize": lambda a: build_final(), "task": task}[a.cmd](a)


if __name__ == "__main__":
    main()
