"""ROI table: what the measured runs cost, self-hosted vs a closed API.

Throughput and training time are read from our own run logs. Prices are
inputs (list prices on the date given), kept in one place so they are easy to
update; nothing here is estimated from a benchmark we did not run.

  python -m lir.roi
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from lir.common import ROOT, RUNS, read_jsonl, write_json

PRICES = {
    "as_of": "2026-11-06",
    # closed API, USD per 1M tokens (input, output); check the provider page before quoting
    "api": {
        "OpenAI GPT-5 mini": (0.25, 2.00),
        "OpenAI GPT-5 nano": (0.05, 0.40),
    },
    "source": "OpenAI list prices as reported for Oct 2026; confirm on openai.com/api/pricing before quoting",
}


def measured_throughput() -> dict:
    """Aggregate output tokens/s from the generation logs of every eval run."""
    out = {}
    for log in sorted((ROOT / "logs").glob("*.log")):
        for m in re.finditer(r"\[([^\]/]+)/([^\]/]+)\] done: (\d+) tokens in (\d+)s", log.read_text(errors="ignore")):
            arm, name, tok, sec = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            if sec > 0:
                out.setdefault(f"{arm}/{name}", []).append(tok / sec)
    return {k: max(v) for k, v in out.items()}


def prompt_tokens(arm: str = "base") -> dict:
    from transformers import AutoTokenizer

    from lir.common import MODEL, MODEL_REV
    from lir.generate import chat_prefix

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    res = {}
    for name in ["attack", "xstest", "task"]:
        p = RUNS / "eval" / arm / f"{name}.gen.jsonl"
        if not p.exists():
            continue
        rows = read_jsonl(p)
        n_in = sum(len(tok(chat_prefix(tok, r["prompt"]), add_special_tokens=False).input_ids) for r in rows)
        n_out = sum(r["n_tokens"] for r in rows)
        res[name] = {"requests": len(rows), "input_tokens": n_in, "output_tokens": n_out}
    return res


def main() -> None:
    tp = measured_throughput()
    usage = prompt_tokens()
    train = {}
    for arm in ["plain", "safegrad"]:
        p = RUNS / arm / "run.json"
        if p.exists():
            r = json.loads(p.read_text())
            train[arm] = {"runtime_h": r["train_runtime_s"] / 3600, "steps": r["optimizer_steps"]}
    api_cost = {}
    tot_in = sum(u["input_tokens"] for u in usage.values())
    tot_out = sum(u["output_tokens"] for u in usage.values())
    for name, (pin, pout) in PRICES["api"].items():
        api_cost[name] = {"per_eval_pass_usd": tot_in / 1e6 * pin + tot_out / 1e6 * pout}
    # wall-clock generation time of one pass on our hardware, from the base arm's measured throughput
    gen_s = sum(usage[n]["output_tokens"] / tp[f"base/{n}"] for n in usage if f"base/{n}" in tp)
    for name in api_cost:
        api_cost[name]["break_even_usd_per_hour"] = api_cost[name]["per_eval_pass_usd"] / (gen_s / 3600)
    res = {"prices": PRICES, "throughput_tok_s": tp, "one_eval_pass_tokens": usage,
           "one_eval_pass_generation_hours": gen_s / 3600,
           "one_eval_pass_totals": {"input": tot_in, "output": tot_out}, "api_cost": api_cost, "training": train}
    write_json(ROOT / "results" / "lir" / "roi.json", res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
