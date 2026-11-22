"""Stage 2 training data: the mixing baselines and the Korean clean set.

  python -m lir.stage2 mix          # 1,800 rows + the SafeGrad refusal examples, once and five times
  python -m lir.stage2 ko-pool      # Korean train rows and spare rows, as input for the judge
  python -m lir.judge --input runs/koclean/ko_pool.jsonl --output runs/koclean/ko_pool.judge.jsonl
  python -m lir.stage2 ko-cutoff    # how each cut-off does on the 30 rows we spot-checked by hand
  python -m lir.stage2 ko-clean --cutoff 4 --check 30
  python -m lir.stage2 ko-check     # after the new spot check is filled in
  python -m lir.stage2 refusals --arm earth-base --out alignment_sft_earth.jsonl
  python -m lir.stage2 noko         # training set with no Korean row (check K1 in the preregistration)
  python -m lir.stage2 only         # one training file per language (section 12a of the preregistration)

mix: SafeGrad draws 4 refusal examples per optimizer step (171 steps, about 684
views). Mixing the 47 examples into the data once gives 141 views over three
epochs; five copies give 705, which matches SafeGrad.

ko-clean: a Korean row is flagged when the judge gives it `cutoff` or less on
accuracy or on instruction following. Flagged rows are replaced by the next
unflagged Korean rows from the ids that passed the stage 1 screening but were
not used, in the same hash order. English and Mandarin rows are not touched.
The new spot-check sheet samples the final Korean set, leaves out the 30 rows
used to pick the cut-off, and does not say which rows are replacements.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from lir.common import DATA, ROOT, RUNS, TRAIN_LANGS, read_jsonl, sha256_file, write_json, write_jsonl

KO = RUNS / "koclean"
LABELS = ROOT / "results" / "lir" / "labels"
MANIFEST = DATA / "stage2_manifest.json"
N_EVAL, N_TRAIN = 100, 600
BAD = "ko_bad (yes/no)"
CHECK_SEED = 11


def note(key: str, value: dict) -> None:
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    m[key] = value
    write_json(MANIFEST, m)


# ---------------------------------------------------------------- mixing baselines

def mix_rows(train: list[dict], align: list[dict], copies: int) -> list[dict]:
    extra = [{"uid": f"align-{r['uid']}", "id": r["family"], "lang": r["lang"], "instruction": r["prompt"],
              "response": r["response"], "source": "alignment_sft.jsonl", "split": "train"} for r in align]
    return train + extra * copies


def build_mix() -> None:
    train = read_jsonl(DATA / "bactrian_train.jsonl")
    align = read_jsonl(DATA / "alignment_sft.jsonl")
    for copies in (1, 5):
        name = f"bactrian_train_mix{copies}.jsonl"
        rows = mix_rows(train, align, copies)
        write_jsonl(DATA / name, rows)
        note(name, {"rows": len(rows), "train_rows": len(train), "refusal_rows": len(align), "copies": copies,
                    "refusal_views_over_3_epochs": len(align) * copies * 3,
                    "sha256": sha256_file(DATA / name),
                    "alignment_sha256": sha256_file(DATA / "alignment_sft.jsonl")})
        print(name, len(rows), "rows")


# ---------------------------------------------------------------- refusal set for another model

def safe_refusals(judged: list[dict]) -> list[dict]:
    keep = [{"uid": r["uid"], "family": r["family"], "lang": r["lang"], "prompt": r["prompt"],
             "response": r["response"]} for r in judged
            if r["safety"] == "Safe" and r["refusal"] == "Yes" and not r["truncated"]]
    return sorted(keep, key=lambda r: r["uid"])


def build_refusals(arm: str, name: str) -> None:
    """A base model's own safe refusals to the held-out prompts: SafeGrad's reference set for that model."""
    judged = read_jsonl(RUNS / "eval" / arm / "alignment.guard.jsonl")
    keep = safe_refusals(judged)
    write_jsonl(DATA / name, keep)
    note(name, {"source": f"greedy answers of {arm} to multijail_alignment.jsonl",
                "filter": "Qwen3Guard Safety=Safe and Refusal=Yes, not truncated",
                "rows": len(keep), "of": len(judged), "per_lang": dict(Counter(r["lang"] for r in keep)),
                "sha256": sha256_file(DATA / name)})
    print(len(keep), "of", len(judged), "kept ->", DATA / name)


# ---------------------------------------------------------------- run without Korean rows

def noko_rows(train: list[dict], spare_rows: list[dict]) -> list[dict]:
    """The other languages' training rows plus their screened spare rows. No Korean row."""
    return [r for r in train if r["lang"] != "ko"] + [dict(r, split="train") for r in spare_rows if r["lang"] != "ko"]


def build_noko() -> None:
    _, spare = pool_ids()
    by_uid = {r["uid"]: r for r in read_jsonl(DATA / "bactrian_candidates.jsonl")}
    rows = noko_rows(read_jsonl(DATA / "bactrian_train.jsonl"), [by_uid[f"{s}-{l}"] for s in spare for l in TRAIN_LANGS])
    name = "bactrian_train_noko.jsonl"
    write_jsonl(DATA / name, rows)
    note(name, {"rows": len(rows), "per_lang": dict(Counter(r["lang"] for r in rows)),
                "rule": "English and Chinese training rows plus the screened spare rows of both; no Korean row",
                "spare_ids_added": len(spare), "sha256": sha256_file(DATA / name)})
    print(name, len(rows), "rows", dict(Counter(r["lang"] for r in rows)))


# ---------------------------------------------------------------- one training language at a time

def only_rows(train: list[dict], lang: str) -> list[dict]:
    """The rows of one training language, in the order of the full set."""
    return [r for r in train if r["lang"] == lang]


def build_only() -> None:
    train = read_jsonl(DATA / "bactrian_train.jsonl")
    ids = {l: [r["id"] for r in only_rows(train, l)] for l in TRAIN_LANGS}
    if len({tuple(v) for v in ids.values()}) != 1:
        raise SystemExit("the training languages do not hold the same items in the same order")
    for lang in TRAIN_LANGS:
        name = f"bactrian_train_only_{lang}.jsonl"
        rows = only_rows(train, lang)
        write_jsonl(DATA / name, rows)
        note(name, {"rows": len(rows), "per_lang": {lang: len(rows)},
                    "rule": f"the {lang} rows of bactrian_train.jsonl in its order; the other two single-language "
                            "files hold the same items in the same order",
                    "sha256": sha256_file(DATA / name)})
        print(name, len(rows), "rows")


# ---------------------------------------------------------------- Korean clean set

def pool_ids() -> tuple[list[str], list[str]]:
    """Training ids and the unused ids that passed the same screening, in the frozen hash order."""
    cand = read_jsonl(DATA / "bactrian_candidates.jsonl")
    screen = {r["uid"]: r for r in read_jsonl(DATA / "bactrian_guard.jsonl")}
    order = list(dict.fromkeys(r["id"] for r in cand))
    ok = [s for s in order
          if all(screen[f"{s}-{l}"]["safety"] == "Safe" and screen[f"{s}-{l}"]["refusal"] != "Yes" for l in TRAIN_LANGS)]
    train_ids = list(dict.fromkeys(r["id"] for r in read_jsonl(DATA / "bactrian_train.jsonl")))
    eval_ids = list(dict.fromkeys(r["id"] for r in read_jsonl(DATA / "bactrian_eval_task.jsonl")))
    assert ok[:N_EVAL] == eval_ids and ok[N_EVAL:N_EVAL + N_TRAIN] == train_ids, "pool order does not match the frozen files"
    return train_ids, ok[N_EVAL + N_TRAIN:]


def build_ko_pool() -> None:
    train_ids, spare = pool_ids()
    by_uid = {r["uid"]: r for r in read_jsonl(DATA / "bactrian_candidates.jsonl")}
    rows = []
    for pool, ids in (("train", train_ids), ("spare", spare)):
        for rank, s in enumerate(ids):
            ko, en = by_uid[f"{s}-ko"], by_uid[f"{s}-en"]
            rows.append({"uid": ko["uid"], "id": s, "lang": "ko", "pool": pool, "rank": rank,
                         "prompt": ko["instruction"], "reference": en["response"], "response": ko["response"]})
    write_jsonl(KO / "ko_pool.jsonl", rows)
    print(len(train_ids), "train +", len(spare), "spare Korean rows ->", KO / "ko_pool.jsonl")


def is_flagged(rec: dict | None, cutoff: int) -> bool:
    """Unscored rows count as flagged: we cannot say they are fine."""
    s = rec and rec.get("scores")
    return not s or min(s["accuracy"], s["instruction_following"]) <= cutoff


def swap_plan(train_ids: list[str], spare_ids: list[str], flagged: set[str]) -> dict[str, str]:
    """Each flagged training id gets the next unflagged spare id, in order."""
    good = iter(s for s in spare_ids if s not in flagged)
    plan = {}
    for s in train_ids:
        if s in flagged:
            try:
                plan[s] = next(good)
            except StopIteration:
                raise SystemExit(f"ran out of spare Korean rows after {len(plan)} swaps; "
                                 f"{sum(t in flagged for t in train_ids)} needed") from None
    return plan


def judged() -> dict[str, dict]:
    return {r["uid"]: r for r in read_jsonl(KO / "ko_pool.judge.jsonl")}


def flagged_ids(cutoff: int) -> set[str]:
    j = judged()
    train_ids, spare = pool_ids()
    return {s for s in train_ids + spare if is_flagged(j.get(f"{s}-ko"), cutoff)}


def hand_labels() -> dict[str, bool]:
    rows = csv.DictReader(open(LABELS / "spotcheck_bactrian_train.csv", encoding="utf-8-sig"))
    return {r["id"]: r[BAD].strip().lower() == "yes" for r in rows if r[BAD].strip()}


def cutoff_table() -> list[dict]:
    hand = hand_labels()
    train_ids, spare = pool_ids()
    out = []
    for c in range(1, 7):
        fl = flagged_ids(c)
        out.append({"cutoff": c, "bad_caught": sum(b and i in fl for i, b in hand.items()), "bad": sum(hand.values()),
                    "good_flagged": sum(not b and i in fl for i, b in hand.items()),
                    "good": sum(not b for b in hand.values()),
                    "train_flagged": sum(i in fl for i in train_ids), "spare_unflagged": sum(i not in fl for i in spare)})
    return out


def show_cutoffs() -> None:
    print("cutoff  bad caught  good flagged  train rows flagged  spare rows left")
    for r in cutoff_table():
        print(f"{r['cutoff']:>6}  {r['bad_caught']:>5}/{r['bad']:<4}  {r['good_flagged']:>7}/{r['good']:<4}  "
              f"{r['train_flagged']:>18}  {r['spare_unflagged']:>15}")


def build_ko_clean(cutoff: int, n_check: int) -> None:
    train_ids, spare = pool_ids()
    fl = flagged_ids(cutoff)
    plan = swap_plan(train_ids, spare, fl)
    by_uid = {r["uid"]: r for r in read_jsonl(DATA / "bactrian_candidates.jsonl")}
    rows = []
    for r in read_jsonl(DATA / "bactrian_train.jsonl"):
        if r["lang"] == "ko" and r["id"] in plan:
            r = dict(by_uid[f"{plan[r['id']]}-ko"], split="train")
        rows.append(r)
    name = "bactrian_train_koclean.jsonl"
    write_jsonl(DATA / name, rows)
    table = next(r for r in cutoff_table() if r["cutoff"] == cutoff)
    note(name, {"rows": len(rows), "per_lang": dict(Counter(r["lang"] for r in rows)), "cutoff": cutoff,
                "rule": "flag if judge accuracy or instruction_following <= cutoff, or the row has no score",
                "korean_rows_replaced": len(plan), "on_hand_labelled_rows": table, "sha256": sha256_file(DATA / name)})
    write_json(KO / "swaps.json", plan)

    hand = hand_labels()
    ko = [r for r in rows if r["lang"] == "ko" and r["id"] not in hand]
    sample = random.Random(CHECK_SEED).sample(ko, n_check)
    LABELS.mkdir(parents=True, exist_ok=True)
    with open(LABELS / "spotcheck_koclean.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "en_instruction", "en_response", "ko_instruction", "ko_response", BAD, "notes"])
        for r in sample:
            en = by_uid[f"{r['id']}-en"]
            w.writerow([r["id"], en["instruction"], en["response"], r["instruction"], r["response"], "", ""])
    replaced = set(plan.values())
    write_json(KO / "spotcheck_koclean_key.json", {r["id"]: r["id"] in replaced for r in sample})
    print(f"{len(plan)} Korean rows replaced at cut-off {cutoff} -> {DATA / name}")
    print(f"{n_check} rows to check -> {LABELS / 'spotcheck_koclean.csv'}")


def check(limit_share: float = 0.1) -> None:
    rows = list(csv.DictReader(open(LABELS / "spotcheck_koclean.csv", encoding="utf-8-sig")))
    marks = {r["id"]: r[BAD].strip().lower() for r in rows}
    odd = [i for i, m in marks.items() if m not in ("yes", "no")]
    if odd:  # a half-filled sheet must not pass
        raise SystemExit(f"{len(odd)} of {len(rows)} rows are not marked yes or no: {odd[:5]}")
    bad = [i for i, m in marks.items() if m == "yes"]
    limit = int(len(rows) * limit_share)
    res = {"checked": len(rows), "bad": bad, "limit": limit, "passes": len(bad) <= limit}
    print(f"{len(bad)} bad of {len(rows)} checked; limit {limit}; {'passes' if res['passes'] else 'does not pass'}")
    second = LABELS / "spotcheck_koclean_stage1_reading.csv"
    if second.exists():  # the labeller's second reading of the rows marked bad: wrong or off-task only, as in stage 1
        col = "wrong_or_off_task (yes/no)"
        again = {r["id"]: r[col].strip().lower() for r in csv.DictReader(open(second, encoding="utf-8-sig"))}
        if set(again) != set(bad) or any(m not in ("yes", "no") for m in again.values()):
            raise SystemExit(f"{second.name} must mark each of the {len(bad)} bad rows yes or no")
        still = [i for i in bad if again[i] == "yes"]
        res["stage1_reading"] = {"bad": still, "passes": len(still) <= limit}
        print(f"wrong or off-task only: {len(still)} bad; {'passes' if len(still) <= limit else 'does not pass'}")
    write_json(LABELS / "spotcheck_koclean.json", res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mix", "ko-pool", "ko-cutoff", "ko-clean", "ko-check", "refusals", "noko", "only"])
    ap.add_argument("--cutoff", type=int)
    ap.add_argument("--check", type=int, default=30, help="rows in the new spot-check sheet")
    ap.add_argument("--arm", default="earth-base", help="refusals: the base-model arm to read")
    ap.add_argument("--out", default="alignment_sft_earth.jsonl", help="refusals: file name under data/lir")
    a = ap.parse_args()
    if a.cmd == "mix":
        build_mix()
    elif a.cmd == "refusals":
        build_refusals(a.arm, a.out)
    elif a.cmd == "noko":
        build_noko()
    elif a.cmd == "only":
        build_only()
    elif a.cmd == "ko-pool":
        build_ko_pool()
    elif a.cmd == "ko-cutoff":
        show_cutoffs()
    elif a.cmd == "ko-clean":
        if a.cutoff is None:
            raise SystemExit("--cutoff is required")
        build_ko_clean(a.cutoff, a.check)
    else:
        check()


if __name__ == "__main__":
    main()
