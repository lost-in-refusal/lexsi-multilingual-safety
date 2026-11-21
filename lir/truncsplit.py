"""Truncation audit, a reading added on Nov 21: answers continued and answers that came out different.

  python -m lir.truncsplit      # results/lir/stage2/truncation/split.{json,md}

The batch an answer is generated in depends on the token cap, so a re-generated answer is either the original
continued or a different answer to the same prompt. For every arm whose re-generation is finished this counts the
guard's verdict changes in each group. For Tiny Aya Earth it also reads E1 on the prompts where every arm's answer
was complete or continued. Reads files only; preregistration section 8 has the notes this follows.
"""
from __future__ import annotations

from lir import trunc
from lir.common import EVAL_LANGS, write_json
from lir.seeds import REFUSAL, STRICT


def split(before: dict[str, dict], after: dict[str, dict]) -> dict:
    """before, after: uid -> guard row. Verdict changes among re-generated answers, by whether the new answer
    starts with the original."""
    out = {}
    for group, keep in (("continued", True), ("different", False)):
        ids = [u for u, r in after.items() if r["regenerated"] and r["same_start"] == keep]
        moves = lambda fn: {"gained": sum(fn(after[u]) > fn(before[u]) for u in ids),
                            "lost": sum(fn(after[u]) < fn(before[u]) for u in ids)}
        out[group] = {"answers": len(ids), "flagged": moves(STRICT), "refusal": moves(REFUSAL)}
    return out


def earth_matching(att0: dict, att1: dict) -> dict:
    """E1 on the prompts where every Earth arm's answer was complete or continued. att0, att1: arm -> {uid: row}."""
    name = dict(zip(trunc.EARTH, ("base", "plain", "safegrad")))
    m1 = trunc.keep_matching({name[arm]: att1[arm] for arm in trunc.EARTH})
    m0 = {arm: {u: att0[inv][u] for u in m1[arm]} for inv, arm in name.items()}
    return dict(trunc.earth_e1(m0, m1), prompts={l: sum(r["lang"] == l for r in m1["base"].values()) for l in EVAL_LANGS})


def fmt(res: dict) -> str:
    lines = ["### Re-generated answers: the original continued, or a different answer", "",
             "Counted after the seed 42 re-generation had been read. A verdict change is the guard's strict flag "
             "(Unsafe or Controversial) appearing or going away.", "",
             "| Arm | Set | Continued | to flagged | to safe | Different | to flagged | to safe |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm, sets in res["arms"].items():
        for name, s in sets.items():
            c, d = s["continued"], s["different"]
            lines.append(f"| {arm} | {name} | {c['answers']} | {c['flagged']['gained']} | {c['flagged']['lost']} "
                         f"| {d['answers']} | {d['flagged']['gained']} | {d['flagged']['lost']} |")
    if "earth_matching_only" in res:
        e = res["earth_matching_only"]
        lines += ["", "### Tiny Aya Earth, E1 on prompts where every arm's answer was complete or continued", "",
                  "Prompts: " + ", ".join(f"{e['prompts'][l]} {l}" for l in EVAL_LANGS) + " of 291.", "",
                  "| | " + " | ".join(EVAL_LANGS) + " |", "|---|" + "---:|" * len(EVAL_LANGS)]
        for when in ("before", "after"):
            s = e[when]["seeds"][42]
            for arm in ("base", "plain", "safegrad"):
                lines.append(f"| {arm}, {when} | " + " | ".join(f"{s['strict'][arm][l]:.1f}" for l in EVAL_LANGS) + " |")
        lines.append("")
        for when in ("before", "after"):
            s, x = e[when]["seeds"][42], e[when]["e1"]
            lines.append(f"- {when.capitalize()}: Korean rise {s['plain_vs_base']['ko']['delta']:+.1f} points "
                         f"(p = {s['plain_vs_base']['ko']['p']:.3g}), SafeGrad against plain "
                         f"{s['safegrad_vs_plain']['ko']['delta']:+.1f} (p = {s['safegrad_vs_plain']['ko']['p']:.3g}). "
                         f"E1 {'holds' if x['holds'] else 'does not hold'}"
                         + ("" if x["korean_rise_significant"] else " (the Korean rise is not significant)") + ".")
    return "\n".join(lines)


def main() -> None:
    ready = [arm for arm in trunc.ARMS if all(trunc.after(arm, n) is not None for n in trunc.SETS)]
    if not ready:
        raise SystemExit("no arm has been generated again yet")
    before = {n: {arm: {r["uid"]: r for r in trunc.judged(arm, n)} for arm in ready} for n in trunc.SETS}
    after = {n: {arm: trunc.after(arm, n) for arm in ready} for n in trunc.SETS}
    res = {"cap": trunc.CAP, "arms": {arm: {n: split(before[n][arm], after[n][arm]) for n in trunc.SETS} for arm in ready}}
    if all(arm in ready for arm in trunc.EARTH):
        res["earth_matching_only"] = earth_matching(before["attack"], after["attack"])
    trunc.OUT.mkdir(parents=True, exist_ok=True)
    write_json(trunc.OUT / "split.json", res)
    md = fmt(res)
    (trunc.OUT / "split.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
