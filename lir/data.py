"""Build the frozen datasets.

  python -m lir.data multijail          # attack + alignment prompts
  python -m lir.data xstest             # English XSTest safe prompts (translations: lir.translate)
  python -m lir.data bactrian-candidates
  python -m lir.data bactrian-finalize  # after lir.guard has screened candidates

CuratorKIT does the text cleaning, exact dedup and MinHash near-dup removal.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

from lir.common import (BACTRIAN, DATA, EVAL_LANGS, MULTIJAIL, SEED, TRAIN_LANGS, XSTEST,
                        rank_key, read_jsonl, sha256_file, write_json, write_jsonl)

N_ALIGN_FAMILIES = 24
N_TRAIN_IDS = 600
N_EVAL_IDS = 100
N_CANDIDATES = 1000

# Responses that refuse or disclaim are dropped before sampling (Nov 4 rule).
REFUSAL_PATTERNS = {
    "en": r"\bas an ai\b|\bi(?:'m| am) (?:sorry|unable|not able)\b|\bi can(?:no|')t\b|\blanguage model\b",
    "ko": r"죄송|AI로서|인공지능으로서|언어 모델|할 수 없습니다",
    "zh": r"抱歉|作为AI|作为一个AI|作为人工智能|语言模型|我无法|我不能",
}


def _cleaner():
    from curatorkit.normalizers import TextCleaner

    # keep newlines and markdown; just fix unicode/control chars
    return TextCleaner(transforms={"strip_html": False, "collapse_whitespace": False,
                                   "fix_encoding_artifacts": True})


def _samples(rows, source_uri, instr="instruction", inp="input", out="output"):
    from curatorkit.schema import DataSample

    return [DataSample(id=r["uid"], source_uri=source_uri, instruction=r.get(instr, ""),
                       input=r.get(inp, "") or "", output=r.get(out, "") or "",
                       metadata={k: v for k, v in r.items() if k not in (instr, inp, out)})
            for r in rows]


# ---------------------------------------------------------------- MultiJail

def build_multijail() -> None:
    repo, rev = MULTIJAIL
    path = Path(hf_hub_download(repo, "MultiJail.csv", repo_type="dataset", revision=rev))
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fams = sorted(rows, key=lambda r: rank_key(f"multijail-{r['id']}"))
    align_ids = {r["id"] for r in fams[:N_ALIGN_FAMILIES]}

    flat = []
    for r in fams:
        split = "alignment" if r["id"] in align_ids else "attack"
        langs = TRAIN_LANGS if split == "alignment" else EVAL_LANGS  # sw never used for training
        for lang in langs:
            flat.append({"uid": f"mj-{r['id']}-{lang}", "family": f"mj-{r['id']}", "lang": lang,
                         "instruction": r[lang], "split": split, "source": r["source"],
                         "tags": ast.literal_eval(r["tags"])})
    cleaned = _cleaner().run(_samples(flat, f"hf://datasets/{repo}@{rev}"))
    out = []
    for s in cleaned:
        m = s.metadata
        out.append({"uid": s.id, "family": m["family"], "lang": m["lang"], "split": m["split"],
                    "prompt": s.instruction, "harm_category": m["tags"], "source": m["source"]})
    attack = [r for r in out if r["split"] == "attack"]
    align = [r for r in out if r["split"] == "alignment"]
    write_jsonl(DATA / "multijail_attack.jsonl", attack)
    write_jsonl(DATA / "multijail_alignment.jsonl", align)
    write_json(DATA / "multijail_manifest.json", {
        "source": repo, "revision": rev, "source_sha256": sha256_file(path),
        "ranking": f"sha256('{SEED}:multijail-<id>') ascending",
        "alignment_families": N_ALIGN_FAMILIES, "attack_families": len(rows) - N_ALIGN_FAMILIES,
        "attack_rows_per_lang": dict(Counter(r["lang"] for r in attack)),
        "alignment_rows_per_lang": dict(Counter(r["lang"] for r in align)),
        "curation": "CuratorKIT TextCleaner (unicode NFKC, control chars, encoding artifacts)",
        "files": {p: sha256_file(DATA / p) for p in ["multijail_attack.jsonl", "multijail_alignment.jsonl"]},
    })
    print(len(attack), "attack rows,", len(align), "alignment rows")


# ---------------------------------------------------------------- XSTest

def build_xstest() -> None:
    repo, _ = XSTEST
    from huggingface_hub import HfApi

    rev = HfApi().dataset_info(repo).sha
    path = Path(hf_hub_download(repo, "xstest_prompts.csv", repo_type="dataset", revision=rev))
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] == "safe"]
    out = [{"uid": f"xs-{r['id']}-en", "family": f"xs-{r['id']}", "lang": "en", "prompt": r["prompt"],
            "type": r["type"], "focus": r.get("focus", "")} for r in rows]
    write_jsonl(DATA / "xstest_en.jsonl", out)
    write_json(DATA / "xstest_manifest.json", {"source": repo, "revision": rev, "license": "cc-by-4.0",
                                               "source_sha256": sha256_file(path), "safe_prompts": len(out)})
    print(len(out), "safe XSTest prompts")


# ---------------------------------------------------------------- Bactrian-X

def _load_bactrian():
    repo, rev = BACTRIAN
    by_lang, hashes = {}, {}
    for lang in TRAIN_LANGS:
        p = Path(hf_hub_download(repo, f"data/{lang}.json.gz", repo_type="dataset", revision=rev))
        hashes[lang] = sha256_file(p)
        by_lang[lang] = {r["id"]: r for r in json.load(gzip.open(p))}
    return by_lang, hashes


def _drop_reason(lang: str, row: dict) -> str | None:
    out = (row.get("output") or "").strip()
    if len(out) < (20 if lang == "en" else 8):
        return "near_empty"
    if re.search(REFUSAL_PATTERNS[lang], out, flags=re.I):
        return "refusal_or_disclaimer"
    if (row.get("input") or "").strip():
        return "has_input"
    return None


def build_bactrian_candidates() -> None:
    from curatorkit.normalizers import ExactDeduplicator, MinHashDeduplicator

    by_lang, hashes = _load_bactrian()
    shared = set.intersection(*(set(d) for d in by_lang.values()))
    reasons = Counter()
    keep = []
    for sid in sorted(shared, key=lambda s: rank_key(f"bactrian-{s}")):
        bad = next((f"{lang}:{r}" for lang in TRAIN_LANGS
                    if (r := _drop_reason(lang, by_lang[lang][sid]))), None)
        if bad:
            reasons[bad.split(":")[1]] += 1
            continue
        keep.append(sid)
        if len(keep) >= 3 * N_CANDIDATES:
            break

    # clean every language, then dedup on the English side (ids are shared)
    rows = {lang: [{"uid": f"{sid}-{lang}", "id": sid, "lang": lang,
                    "instruction": by_lang[lang][sid]["instruction"],
                    "output": by_lang[lang][sid]["output"]} for sid in keep] for lang in TRAIN_LANGS}
    cleaned = {lang: {s.metadata["id"]: s for s in _cleaner().run(_samples(rows[lang], f"bactrian-{lang}"))}
               for lang in TRAIN_LANGS}
    en = [cleaned["en"][sid] for sid in keep]
    en = ExactDeduplicator().run(en)
    en = MinHashDeduplicator(threshold=0.7).run(en)
    survivors = [s.metadata["id"] for s in en][:N_CANDIDATES]
    reasons["dedup_removed"] = len(keep) - len(en)

    cand = []
    for sid in survivors:
        for lang in TRAIN_LANGS:
            s = cleaned[lang][sid]
            cand.append({"uid": f"{sid}-{lang}", "id": sid, "lang": lang, "instruction": s.instruction,
                         "response": s.output, "source": f"MBZUAI/Bactrian-X@{BACTRIAN[1]}:{lang}"})
    write_jsonl(DATA / "bactrian_candidates.jsonl", cand)
    write_json(DATA / "bactrian_candidates_manifest.json", {
        "source": BACTRIAN[0], "revision": BACTRIAN[1], "source_sha256": hashes,
        "shared_ids": len(shared), "candidate_ids": len(survivors), "drop_reasons": dict(reasons),
        "ranking": f"sha256('{SEED}:bactrian-<id>') ascending"})
    print(len(survivors), "candidate ids;", dict(reasons))


def build_bactrian_final() -> None:
    from curatorkit.normalizers import MinHashDeduplicator
    from curatorkit.schema import DataSample

    cand = read_jsonl(DATA / "bactrian_candidates.jsonl")
    screen = {r["uid"]: r for r in read_jsonl(DATA / "bactrian_guard.jsonl")}
    flagged = Counter()
    ok_ids = []
    order = list(dict.fromkeys(r["id"] for r in cand))
    by_uid = {r["uid"]: r for r in cand}
    for sid in order:
        verdicts = [screen[f"{sid}-{lang}"] for lang in TRAIN_LANGS]
        if any(v["safety"] != "Safe" for v in verdicts):
            flagged["guard_not_safe"] += 1
        elif any(v["refusal"] == "Yes" for v in verdicts):
            flagged["guard_refusal"] += 1
        else:
            ok_ids.append(sid)
    need = N_TRAIN_IDS + N_EVAL_IDS
    assert len(ok_ids) >= need, f"only {len(ok_ids)} ids passed screening"
    eval_ids, train_pool = ok_ids[:N_EVAL_IDS], ok_ids[N_EVAL_IDS:]

    # drop training ids that are near-duplicates of any eval instruction
    probe = [DataSample(id=f"e-{s}", source_uri="eval", instruction=by_uid[f"{s}-en"]["instruction"])
             for s in eval_ids]
    pool = [DataSample(id=f"t-{s}", source_uri="train", instruction=by_uid[f"{s}-en"]["instruction"])
            for s in train_pool]
    kept = MinHashDeduplicator(threshold=0.7).run(probe + pool)
    kept_train = [s.id[2:] for s in kept if s.id.startswith("t-")]
    flagged["train_near_dup_of_eval"] = len(train_pool) - len(kept_train)
    train_ids = kept_train[:N_TRAIN_IDS]
    assert len(train_ids) == N_TRAIN_IDS

    def rows(ids, split):
        return [dict(by_uid[f"{s}-{lang}"], split=split) for s in ids for lang in TRAIN_LANGS]

    write_jsonl(DATA / "bactrian_train.jsonl", rows(train_ids, "train"))
    write_jsonl(DATA / "bactrian_eval_task.jsonl", rows(eval_ids, "eval_task"))
    write_json(DATA / "bactrian_manifest.json", {
        "train_ids": len(train_ids), "eval_task_ids": len(eval_ids), "screening": dict(flagged),
        "guard": "Qwen3Guard-Gen-4B response moderation on (instruction, response), all three languages",
        "files": {p: sha256_file(DATA / p) for p in ["bactrian_train.jsonl", "bactrian_eval_task.jsonl"]}})
    print("train", len(train_ids), "eval", len(eval_ids), dict(flagged))


# ---------------------------------------------------------------- SafeGrad alignment set

def build_alignment_sft() -> None:
    """Base model's own answers to the 24 alignment families, kept only if the
    guard calls them safe refusals. These feed SafeGrad's KL term."""
    from lir.common import RUNS

    judged = read_jsonl(RUNS / "eval" / "base" / "alignment.guard.jsonl")
    keep = [{"uid": r["uid"], "family": r["family"], "lang": r["lang"], "prompt": r["prompt"],
             "response": r["response"]} for r in judged
            if r["safety"] == "Safe" and r["refusal"] == "Yes" and not r["truncated"]]
    keep.sort(key=lambda r: r["uid"])
    write_jsonl(DATA / "alignment_sft.jsonl", keep)
    write_json(DATA / "alignment_manifest.json", {
        "source": "Tiny Aya Global (base) greedy responses to multijail_alignment.jsonl",
        "filter": "Qwen3Guard Safety=Safe and Refusal=Yes, not truncated",
        "rows": len(keep), "of": len(judged), "per_lang": dict(Counter(r["lang"] for r in keep)),
        "sha256": sha256_file(DATA / "alignment_sft.jsonl")})
    print(len(keep), "of", len(judged), "alignment rows kept")


def main() -> None:
    steps = {"multijail": build_multijail, "xstest": build_xstest, "bactrian-candidates": build_bactrian_candidates,
             "bactrian-finalize": build_bactrian_final, "alignment-sft": build_alignment_sft}
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=list(steps))
    steps[ap.parse_args().step]()


if __name__ == "__main__":
    main()
