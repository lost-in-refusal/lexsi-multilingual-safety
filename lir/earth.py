"""Tiny Aya Earth: the two preregistered checks and which rows of the outcome table apply.

  python -m lir.earth        # writes results/lir/stage2/earth/checks.{json,md}

E1: plain fine-tuning raises Korean strict attack success on Earth, and SafeGrad stays below plain.
E2: the same fine-tune lowers Earth's Swahili refusal rate on attack prompts by more than it
    lowered Global's at seed 42.
One seed per arm, so every number here is a single run.
"""
from __future__ import annotations

from lir.common import DATA, EVAL_LANGS, ROOT, read_jsonl, write_json
from lir.metrics import load
from lir.seeds import REFUSAL, analyse, by_family, change, diff, pooled, rate

ARMS = ("earth-base", "earth-plain", "earth-safegrad")
GLOBAL = ("base", "plain")  # seed 42


def checks(arms: tuple[str, str, str] = ARMS) -> dict:
    base, plain, safegrad = arms
    data = {"base": load(base, "attack"), plain: load(plain, "attack"), safegrad: load(safegrad, "attack")}
    ref = {a: load(a, "attack") for a in GLOBAL}
    n = len(read_jsonl(DATA / "multijail_attack.jsonl"))
    judged = {**dict(zip(arms, data.values())), **ref}
    short = [f"{a} {sum(r['safety'] is not None for r in d.values())}/{n}" for a, d in judged.items()
             if sum(r["safety"] is not None for r in d.values()) != n]
    if short:  # a guard pass that stopped part-way would give rates on a skewed subset
        raise SystemExit("attack verdicts incomplete, nothing written: " + ", ".join(short))
    row = analyse(data, {42: (plain, safegrad)})["seeds"][42]
    eb, ep = by_family(data["base"], REFUSAL), by_family(data[plain], REFUSAL)
    gb, gp = (by_family(ref[a], REFUSAL) for a in GLOBAL)
    e_drop, g_drop = change(eb, ep, "sw"), change(gb, gp, "sw")
    e, g = diff(eb, ep, "sw"), diff(gb, gp, "sw")
    gap = pooled([{f: e[f] - g[f] for f in e if f in g}])  # Earth's change minus Global's, same prompts
    rise, fix = row["plain_vs_base"]["ko"], row["safegrad_vs_plain"]["ko"]
    drop = e_drop["delta"] < 0 and e_drop["p"] < 0.05
    table = []
    if drop and e_drop["delta"] < g_drop["delta"]:
        table.append("A. A larger Swahili refusal drop than Global: E2 confirmed")
    elif drop:
        table.append("A significant Swahili refusal drop, but not larger than Global's: E2 not confirmed")
    else:
        table.append("B. No significant Swahili refusal drop: E2 refuted")
    if rate(eb, "sw") < rate(gb, "sw"):
        table.append("C. Base Earth refuses fewer Swahili attack prompts than base Global")
    if fix["delta"] >= 0:
        table.append("D. SafeGrad is not below plain on Korean: E1 refuted for the repair")
    res = {"arms": list(arms), "strict": row["strict"], "macro": row["macro"], "plain_vs_base": row["plain_vs_base"],
           "safegrad_vs_plain": row["safegrad_vs_plain"],
           "sw_refusal": {"earth_base": rate(eb, "sw"), "earth_plain": rate(ep, "sw"), "global_base": rate(gb, "sw"),
                          "global_plain": rate(gp, "sw"), "earth_change": e_drop, "global_change": g_drop,
                          "earth_minus_global": gap},
           "e1": {"korean_rises": rise["delta"] > 0, "korean_rise_significant": rise["delta"] > 0 and rise["p"] < 0.05,
                  "safegrad_below_plain": fix["delta"] < 0, "holds": rise["delta"] > 0 and fix["delta"] < 0},
           "e2": {"holds": drop and e_drop["delta"] < g_drop["delta"]}, "outcome_rows": table}
    align = DATA / "alignment_sft_earth.jsonl"
    if align.exists():
        res["refusal_examples"] = {"earth": len(read_jsonl(align)), "global": len(read_jsonl(DATA / "alignment_sft.jsonl"))}
    return res


def fmt(res: dict) -> str:
    star = lambda c: "*" if c["p"] < 0.05 else ""
    lines = ["### Tiny Aya Earth, seed 42: strict attack success (%)", "",
             "| Arm | en | ko | zh | sw | macro |", "|---|---:|---:|---:|---:|---:|",
             "| base | " + " | ".join(f"{res['strict']['base'][l]:.1f}" for l in EVAL_LANGS) + f" | {res['macro']['base']:.1f} |",
             "| plain | " + " | ".join(
                 f"{res['strict']['plain'][l]:.1f} ({res['plain_vs_base'][l]['delta']:+.1f}{star(res['plain_vs_base'][l])})"
                 for l in EVAL_LANGS) + f" | {res['macro']['plain']:.1f} |",
             "| safegrad | " + " | ".join(
                 f"{res['strict']['safegrad'][l]:.1f} ({res['safegrad_vs_plain'][l]['delta']:+.1f}{star(res['safegrad_vs_plain'][l])})"
                 for l in EVAL_LANGS) + f" | {res['macro']['safegrad']:.1f} |",
             "", "Plain: change against base. SafeGrad: change against plain. * = p < 0.05, exact McNemar. Single runs.", ""]
    s, e1 = res["sw_refusal"], res["e1"]
    ko, fix = res["plain_vs_base"]["ko"], res["safegrad_vs_plain"]["ko"]
    pv = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    e1_word = "does not hold" if not e1["holds"] else (
        "holds" if e1["korean_rise_significant"] else "holds in direction only (the Korean rise is not significant)")
    lines += ["### Preregistered checks", "",
              f"- **E1** {e1_word}: Korean attack success changes by {ko['delta']:+.1f} points "
              f"under plain ({pv(ko['p'])}); SafeGrad against plain is {fix['delta']:+.1f} points ({pv(fix['p'])}).",
              f"- **E2** {'holds' if res['e2']['holds'] else 'does not hold'}: Earth's Swahili refusal rate goes from "
              f"{s['earth_base']:.1f}% to {s['earth_plain']:.1f}% ({s['earth_change']['delta']:+.1f} points, "
              f"{pv(s['earth_change']['p'])}); Global's went from {s['global_base']:.1f}% to {s['global_plain']:.1f}% "
              f"({s['global_change']['delta']:+.1f}). Earth's change minus Global's on the same prompts: "
              f"{s['earth_minus_global']['mean']:+.1f} points [{s['earth_minus_global']['ci95'][0]:+.1f}, "
              f"{s['earth_minus_global']['ci95'][1]:+.1f}].", "", "Rows of the outcome table that apply:", ""]
    lines += [f"- {t}" for t in res["outcome_rows"]]
    if "refusal_examples" in res:
        n = res["refusal_examples"]
        lines += ["", f"SafeGrad's refusal set on Earth keeps {n['earth']} examples; Global's kept {n['global']}."]
    return "\n".join(lines)


def main() -> None:
    res = checks()
    out = ROOT / "results" / "lir" / "stage2" / "earth"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "checks.json", res)
    md = fmt(res)
    (out / "checks.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
