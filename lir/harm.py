"""Harm-category breakdown (preregistration section 9).

  python -m lir.harm             # results/lir/stage2/harm_categories.{json,md}
  python -m lir.harm --figure    # and harm_categories.png

MultiJail tags each prompt with one or more harm categories, so a prompt counts in each of its categories.
Rates are averaged over the three seeds; intervals are a bootstrap over the category's prompts. A description:
categories with fewer than 20 prompts are listed and not interpreted, and no test is run between categories.
"""
from __future__ import annotations

import argparse

from lir.common import EVAL_LANGS, ROOT, write_json
from lir.metrics import load
from lir.seeds import SEEDS, STRICT, UNSAFE, by_family, pooled

OUT = ROOT / "results" / "lir" / "stage2"
MIN_PROMPTS = 20
ALL = "all"  # the four languages together: a prompt's value is its mean over the languages


def per_prompt(rows: dict[str, dict], fn) -> dict[str, dict[str, float]]:
    """lang -> {family: value}, with the four-language mean under ALL."""
    v = by_family(rows, fn)
    out = {l: {f: x for (f, lang), x in v.items() if lang == l} for l in EVAL_LANGS}
    out[ALL] = {f: sum(out[l][f] for l in EVAL_LANGS) / len(EVAL_LANGS) for f in out[EVAL_LANGS[0]]
                if all(f in out[l] for l in EVAL_LANGS)}
    return out


def breakdown(data: dict[str, dict[str, dict]], seeds: dict[int, tuple[str, str]] = SEEDS) -> dict:
    """data: arm -> {uid: guard row} for the attack set; rows carry `harm_category` (a list of tags)."""
    tags: dict[str, set] = {}
    for r in data["base"].values():
        for t in r["harm_category"]:
            tags.setdefault(t, set()).add(r["family"])
    vals = {fn_name: {arm: per_prompt(rows, fn) for arm, rows in data.items()}
            for fn_name, fn in (("strict", STRICT), ("loose", UNSAFE))}
    mean = lambda d, fams: 100 * sum(d[f] for f in fams) / len(fams)
    cats = {}
    for tag, fams in sorted(tags.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        fams = sorted(fams)
        cell = {}
        for lang in (*EVAL_LANGS, ALL):
            s, lo = vals["strict"], vals["loose"]
            over = lambda a, b: pooled([{f: s[y][lang][f] - s[x][lang][f] for f in fams}
                                        for x, y in ((a(p, g), b(p, g)) for p, g in seeds.values())])
            avg = lambda v, i: sum(mean(v[pair[i]][lang], fams) for pair in seeds.values()) / len(seeds)
            cell[lang] = {
                "strict": {"base": mean(s["base"][lang], fams), "plain": avg(s, 0), "safegrad": avg(s, 1)},
                "loose": {"base": mean(lo["base"][lang], fams), "plain": avg(lo, 0), "safegrad": avg(lo, 1)},
                "plain_vs_base": over(lambda p, g: "base", lambda p, g: p),
                "safegrad_vs_plain": over(lambda p, g: p, lambda p, g: g),
                "safegrad_vs_base": over(lambda p, g: "base", lambda p, g: g),
            }
        cats[tag] = {"prompts": len(fams), "interpreted": len(fams) >= MIN_PROMPTS, "langs": cell}
    big = [t for t, c in cats.items() if c["interpreted"]]
    flat = [t for t in big if cats[t]["langs"]["ko"]["plain_vs_base"]["mean"] <= 0]
    return {"min_prompts": MIN_PROMPTS, "seeds": sorted(seeds), "categories": cats,
            "korean_broad": not flat, "korean_no_rise_in": flat, "interpreted": big}


def fmt(res: dict) -> str:
    ci = lambda c: f"{c['mean']:+.1f} [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}]"
    names = {"en": "English", "ko": "Korean", "zh": "Chinese", "sw": "Swahili", ALL: "Four languages together"}
    lines = [f"### Strict attack success by harm category, averaged over seeds {', '.join(map(str, res['seeds']))}", "",
             "A prompt with several tags counts in each of its categories, so rows overlap. Changes are in points "
             "with a 95% interval over the category's prompts. Loose attack success is in brackets after each rate.", ""]
    if res["korean_broad"]:
        lines.append(f"Korean strict attack success rises under plain in every category with {res['min_prompts']} or more prompts.")
    else:
        lines.append(f"Korean strict attack success does not rise under plain in: {', '.join(res['korean_no_rise_in'])}.")
    for lang in ("ko", ALL, "en", "zh", "sw"):
        lines += ["", f"#### {names[lang]}", "",
                  "| Category | Prompts | Base | Plain | SafeGrad | Plain against base | SafeGrad against plain |",
                  "|---|---:|---:|---:|---:|---|---|"]
        small = False
        for tag, c in res["categories"].items():
            if not c["interpreted"] and not small:
                small = True
                lines.append(f"| *Fewer than {res['min_prompts']} prompts: listed, not interpreted* | | | | | | |")
            v = c["langs"][lang]
            rate = lambda arm: f"{v['strict'][arm]:.1f} ({v['loose'][arm]:.1f})"
            lines.append(f"| {tag} | {c['prompts']} | {rate('base')} | {rate('plain')} | {rate('safegrad')} | "
                         f"{ci(v['plain_vs_base'])} | {ci(v['safegrad_vs_plain'])} |")
    lines += ["", "Swahili cells are the guard's raw verdicts. The guard missed our agreement bar in Swahili, and the "
              "translator-corrected numbers are not split by category."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    arms = ["base"] + [arm for pair in SEEDS.values() for arm in pair]
    data = {arm: load(arm, "attack") for arm in arms}
    short = [arm for arm, d in data.items() if len(d) != len(data["base"]) or any(r["safety"] is None for r in d.values())]
    if short:
        raise SystemExit("attack verdicts incomplete, nothing written: " + ", ".join(short))
    res = breakdown(data)
    write_json(OUT / "harm_categories.json", res)
    md = fmt(res)
    (OUT / "harm_categories.md").write_text(md + "\n")
    print(md)
    if a.figure:
        from lir.figures import harm_figure

        harm_figure(res, OUT / "harm_categories.png")


if __name__ == "__main__":
    main()
