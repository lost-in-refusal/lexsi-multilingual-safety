"""Truncation audit, a reading added on Nov 21: answers continued and answers that came out different.

  python -m lir.truncsplit      # results/lir/stage2/truncation/split.{json,md}

The batch an answer is generated in depends on the token cap, so a re-generated answer is either the original
continued or a different answer to the same prompt. For every arm whose re-generation is finished this counts the
guard's verdict changes in each group. For Tiny Aya Earth it also reads E1 on the prompts where every arm's answer
was complete or continued, and repeats the unregistered readings of earth/checks.md on the re-generated answers.
Reads files only; preregistration section 8 has the notes this follows.
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


def earth_beside(att0: dict, att1: dict) -> dict:
    """The readings of earth/checks.md that were not registered, before and after, and how many of each Earth arm's
    Korean flags sit on answers that reached the old cap. att0, att1: arm -> {uid: row}, Global base under "base".
    Added after the Earth re-generation had been read."""
    from lir.earth import after_checks

    side = lambda att: after_checks(*(att[arm] for arm in trunc.EARTH), att["base"])
    cap = {}
    for arm in trunc.EARTH:
        ko = [u for u, r in att0[arm].items() if r["lang"] == "ko"]
        cut = [u for u in ko if att1[arm][u]["regenerated"]]
        flags = lambda att, ids: sum(int(STRICT(att[arm][u])) for u in ids)
        cap[arm] = {"answers": len(ko), "cut": len(cut), "flagged_cut_before": flags(att0, cut),
                    "flagged_cut_after": flags(att1, cut), "flagged_complete": flags(att0, ko) - flags(att0, cut)}
    return {"before": side(att0), "after": side(att1), "ko_cap": cap}


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
    if "earth_not_registered" in res:
        x = res["earth_not_registered"]
        b, a = x["before"], x["after"]
        pv = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        d = lambda c: f"{c['delta']:+.1f} ({pv(c['p'])})"
        two = lambda get: f"{d(get(b))} before, {d(get(a))} after"
        rates = lambda s: ", ".join(f"{s['ko_unsafe_class_only']['rate'][k]:.1f}%" for k in ("base", "plain", "safegrad"))
        lines += ["", "### Tiny Aya Earth, the readings that were not registered, before and after", "",
                  "These are the readings under \"Added after the runs were read\" in `earth/checks.md`, repeated on the "
                  "re-generated answers. They were added to this file after the Earth re-generation had been read. They "
                  "are not registered and decide nothing. Changes in points with the exact McNemar p-value.", "",
                  "- Korean strict attack success, SafeGrad against base: "
                  f"{two(lambda s: s['ko_strict_safegrad_vs_base'])}.",
                  f"- Korean counting only the guard's Unsafe class, base, plain and SafeGrad: {rates(b)} before, "
                  f"{rates(a)} after. Plain against base {two(lambda s: s['ko_unsafe_class_only']['plain_vs_base'])}. "
                  f"SafeGrad against plain {two(lambda s: s['ko_unsafe_class_only']['safegrad_vs_plain'])}. "
                  f"SafeGrad against base {two(lambda s: s['ko_unsafe_class_only']['safegrad_vs_base'])}.",
                  f"- Row C. SafeGrad's Swahili refusal rate on Earth: {b['sw_refusal']['safegrad']:.1f}% before, "
                  f"{a['sw_refusal']['safegrad']:.1f}% after. Against Earth plain "
                  f"{two(lambda s: s['sw_refusal']['safegrad_vs_plain'])}. Against Earth base "
                  f"{two(lambda s: s['sw_refusal']['safegrad_vs_base'])}. Against Global base "
                  f"{two(lambda s: s['sw_refusal']['safegrad_vs_global_base'])}. Earth base against Global base "
                  f"{two(lambda s: s['sw_refusal']['base_vs_global_base'])}.",
                  "", "Korean attack answers that reached the 256-token cap, and the strict flags on them.", "",
                  "| Arm | Korean answers | Reached the cap | Flagged among those, before | after | Flagged among the complete |",
                  "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {arm} | {c['answers']} | {c['cut']} | {c['flagged_cut_before']} | {c['flagged_cut_after']} "
                  f"| {c['flagged_complete']} |" for arm, c in x["ko_cap"].items()]
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
        if "base" in ready:
            res["earth_not_registered"] = earth_beside(before["attack"], after["attack"])
    trunc.OUT.mkdir(parents=True, exist_ok=True)
    write_json(trunc.OUT / "split.json", res)
    md = fmt(res)
    (trunc.OUT / "split.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
