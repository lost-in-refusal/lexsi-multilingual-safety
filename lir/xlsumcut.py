"""Preregistration section 20b, C1: the split of section 18c on the summarisation fine-tune of section 20.

The Korean attack answers of the base run and of the plain runs that section 20 read, split by whether the answer
reached the 256-token cap and into three groups of equal size by length. The split and its sentence are
lir.reviewtables.by_length and length_report, called unchanged. This module decides whether the check is read
(only when section 20's gate holds and X1 is above zero), loads the uploads section 20 read, and names a thin group.

    python -m lir.xlsumcut      # after `python -m lir.xlsum report`
"""
from __future__ import annotations

import json

from lir import reviewtables, xlsum
from lir.common import write_json
from lir.repeat import WHERE, ci
from lir.seeds import STRICT

OUT = xlsum.OUT
THIN = 30   # a group with fewer answers than this on either side is named beside the sentence
NOT_READ = {"gate": "C1 is not read: by section 20's gate the model no longer answered general requests",
            "x1": "C1 is not read: X1 found no Korean rise to split"}


def decide(x: dict) -> str | None:
    """Why C1 is not read, given what `python -m lir.xlsum report` wrote, or None when it is read."""
    if not x["gate"]["informative"]:
        return "gate"
    return None if x["x1"]["side"] == "above" else "x1"


def thin(res: dict, bar: int = THIN) -> list[str]:
    """The groups the sentence reads that have fewer than `bar` answers of the base run or of the plain runs."""
    return [g for g in reviewtables.RULE_GROUPS if min(res["groups"][g]["base"]["answers"], res["groups"][g]["arm"]["answers"]) < bar]


def result(x: dict, att: dict[str, dict] | None) -> dict:
    """x: section 20's result. att: run -> the guard's rows on the attack set by uid, the base run under "base"; it is
    not looked at when C1 is not read."""
    why = decide(x)
    if why:
        return {"read": False, "why": why, "report": NOT_READ[why], "seeds": x["seeds"]}
    plains = [att[xlsum.PAIRS[seed][0]] for seed in x["seeds"]]
    whole = reviewtables.among(att["base"], plains, STRICT, lambda r: True, "ko")["mean"]
    if abs(whole - x["x1"]["mean"]) > 1e-9:   # with every answer in one group the figure is X1's
        raise SystemExit(f"the Korean rise over every answer is {whole:.4f} here and {x['x1']['mean']:.4f} in section 20's table: "
                         "these are not the answers it read, nothing written")
    res = reviewtables.by_length(att["base"], plains)
    return dict(res, read=True, seeds=x["seeds"], thin=thin(res), thin_bar=THIN)


def md(res: dict) -> str:
    head = ["### The Korean rise on the summarisation fine-tune, by answer length and the cap (section 20b: C1)", ""]
    if not res["read"]:
        return "\n".join(head + [f"**{res['report']}.**"])
    one = lambda v: "none" if v is None else f"{v:.1f}"
    G = reviewtables.GROUPS
    lines = head + [
        f"The Korean attack answers of `{xlsum.BASE}` and of the plain runs at seed{'s' if len(res['seeds']) > 1 else ''} "
        f"{xlsum.listed(res['seeds'])}, split as in section 18c: by whether the answer reached the 256-token cap, and into three groups of "
        "equal size by its length in tokens. Inside a group: strict attack success of the plain answers, the seeds put together, minus "
        "that of the base answers, with a 95% interval from a bootstrap over the prompt families.", "",
        "| Group | Tokens | Of them at the cap | Base answers | flagged (%) | Plain answers | flagged (%) | Plain minus base (points) | Interval |",
        "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for name, g in res["groups"].items():
        tok = f"{g['tokens'][0]} to {g['tokens'][1]}" if g["tokens"] else "none"
        lines.append(f"| {G[name][0]} | {tok} | {g['reached_the_cap']} | {g['base']['answers']} | {one(g['base']['rate'])} | "
                     f"{g['arm']['answers']} | {one(g['arm']['rate'])} | {ci(g) if g['ci95'] else 'not estimable'} | "
                     f"{WHERE[g['side']] if g['side'] else 'none'} |")
    lines += ["", f"**{res['report']}.**"]
    if res["beside_the_sentence"]:
        lines += ["", "The registered sentence names the groups whose interval includes zero. Beside it: "
                  + "; ".join(f"in {G[g][1]} the interval " + ("lies below zero" if res["groups"][g]["side"] == "below" else "could not be computed")
                              for g in res["beside_the_sentence"]) + "."]
    if res["thin"]:
        lines += ["", f"Thin, with fewer than {res['thin_bar']} answers of the base run or of the plain runs: "
                  + ", ".join(G[g][1] for g in res["thin"]) + ". The sentence stands as the rule gives it; a thin group's interval says little."]
    dropped = {G[g][1]: v["draws_dropped"] for g, v in res["groups"].items() if v["draws_dropped"]}
    lines += ["", f"The thirds are cut over the {res['answers']:,} answers together ({res['base_answers']} of the base run, counted once). "
              "A group is defined by the answer, which the fine-tune changes, so this compares different prompts in the two runs: a "
              "description, not a control. No answer of this block was generated again past the cap."
              + (" Bootstrap draws left out because a side had no answer in the group: " + ", ".join(f"{g} {k}" for g, k in dropped.items()) + "."
                 if dropped else "")]
    return "\n".join(lines)


def main() -> None:
    table = OUT / "xlsum.json"
    if not table.exists():
        raise SystemExit("section 20 is not read yet (`python -m lir.xlsum report`), nothing written")
    x = json.loads(table.read_text())
    att = None
    if decide(x) is None:
        got = xlsum.read(xlsum.STAGING_DIR)
        runs = [xlsum.BASE] + [xlsum.PAIRS[seed][0] for seed in x["seeds"]]
        other = [run for run in runs if run in got["missing"] or got["runs"][run]["upload"] != x["runs"][run]["upload"]]
        if other:
            raise SystemExit(f"not the uploads section 20 read: {', '.join(other)}; nothing written")
        att = got["att"]
    res = result(x, att)
    text = md(res)
    write_json(OUT / "xlsum_cut.json", res)
    (OUT / "xlsum_cut.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
