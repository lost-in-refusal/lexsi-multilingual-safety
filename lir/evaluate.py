"""Generate and guard-score one model arm on every evaluation set.

  python -m lir.evaluate --arm base
  python -m lir.evaluate --arm plain --adapter runs/plain/adapter
  python -m lir.evaluate --arm base --sets attack   # subset

Writes runs/eval/<arm>/<set>.gen.jsonl and <set>.guard.jsonl. The task judge
(lir.judge) runs separately because Qwen3-14B needs the whole GPU.
"""
from __future__ import annotations

import argparse
from lir.common import DATA, EVAL_LANGS, RUNS, empty_cache, read_jsonl

SETS = {
    # name: (files, max_new_tokens)
    "attack": (["multijail_attack.jsonl"], 256),
    "xstest": ([f"xstest_{l}.jsonl" for l in EVAL_LANGS], 256),
    "task": (["bactrian_eval_task.jsonl"], 1024),
    "alignment": (["multijail_alignment.jsonl"], 256),
}


def rows_for(name: str) -> list[dict]:
    files, _ = SETS[name]
    rows = []
    for fn in files:
        for r in read_jsonl(DATA / fn):
            if name == "task":
                r = {"uid": r["uid"], "id": r["id"], "lang": r["lang"], "prompt": r["instruction"],
                     "reference": r["response"]}
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--sets", nargs="+", default=["attack", "xstest", "task"])
    ap.add_argument("--skip-guard", action="store_true")
    a = ap.parse_args()
    out = RUNS / "eval" / a.arm
    out.mkdir(parents=True, exist_ok=True)

    from lir import generate

    tok, model = generate.load_model(a.adapter)
    for name in a.sets:
        generate.generate(tok, model, rows_for(name), out / f"{name}.gen.jsonl", SETS[name][1], tag=f"{a.arm}/{name}")
    del model
    empty_cache()
    if a.skip_guard:
        return

    from lir import guard

    gtok, gmodel = guard.load()
    for name in a.sets:
        gens = read_jsonl(out / f"{name}.gen.jsonl")
        guard.judge(gtok, gmodel, gens, out / f"{name}.guard.jsonl")


if __name__ == "__main__":
    main()
