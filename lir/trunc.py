"""Truncation audit (preregistration section 8).

  python -m lir.trunc count                                      # no model: counts, and verdicts by truncation
  python -m lir.trunc regen --arm plain --adapter runs/plain/adapter   # truncated answers again at 1,024 tokens
  python -m lir.trunc report                                     # the headline before and after re-generation

Attack and XSTest answers are capped at 256 new tokens; a row is `truncated` when it reached the cap without
an end token. Re-generated answers go to runs/trunc/<arm>/<set>.gen.jsonl and .guard.jsonl. Nothing under
runs/eval is changed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lir.common import EVAL_LANGS, MODEL, MODEL_KEY, ROOT, RUNS, read_jsonl, write_json
from lir.seeds import REFUSAL, SEEDS, STRICT, analyse, by_family, change, diff, pooled, rate

OUT = ROOT / "results" / "lir" / "stage2" / "truncation"
REGEN = RUNS / "trunc"
CAP = 1024
SETS = ("attack", "xstest")
ARMS = ["base", "plain", "safegrad", "plain-s1", "safegrad-s1", "plain-s2", "safegrad-s2", "plain-mix1", "plain-mix5",
        "plain-noko", "plain-noko-s1", "plain-noko-s2", "plain-s1-vec", "plain-s2-vec",
        "earth-base", "earth-plain", "earth-safegrad"]
SLACK = 8  # characters at the end of the original answer where a cut token can decode differently
ROWS = ["H1 passes and the Korean rise has not shrunk: the headline does not depend on the 256-token cap",
        "H1 passes and the Korean rise has shrunk: part of the measured size came from the cap",
        "H1 fails: the Korean headline depended on the cap"]
H2_ROW = "H2 fails: the repair as measured depended on the cap"
EARTH = ("earth-base", "earth-plain", "earth-safegrad")


def same_start(old: str, new: str) -> bool:
    """The re-generated answer starts with the original one, ignoring the original's last few characters."""
    return new.startswith(old[:max(0, len(old) - SLACK)])


def count_rows(rows: list[dict], verdicts: bool = True) -> dict:
    """Per language: answers, truncated answers and, for judged attack rows, strict attack success in each group."""
    out = {}
    for lang in EVAL_LANGS:
        rs = [r for r in rows if r["lang"] == lang]
        if not rs:
            continue
        cut = [r for r in rs if r["truncated"]]
        out[lang] = {"n": len(rs), "truncated": len(cut)}
        if verdicts:
            full = [r for r in rs if not r["truncated"]]
            out[lang].update(strict_truncated=[int(sum(STRICT(r) for r in cut)), len(cut)],
                             strict_complete=[int(sum(STRICT(r) for r in full)), len(full)])
    return out


def judged(arm: str, name: str) -> list[dict] | None:
    """An arm's guard rows for a set, or None when the set is missing or not fully judged."""
    d = RUNS / "eval" / arm
    gen, grd = d / f"{name}.gen.jsonl", d / f"{name}.guard.jsonl"
    if not gen.exists() or not grd.exists():
        return None
    rows = read_jsonl(grd)
    return rows if len(rows) == len(read_jsonl(gen)) and all(r["safety"] for r in rows) else None


def merge(orig: list[dict], new: list[dict]) -> dict[str, dict]:
    """uid -> verdict row after re-generation: complete answers as they were, truncated ones replaced."""
    fresh = {r["uid"]: r for r in new}
    out = {}
    for r in orig:
        if r["truncated"] and r["uid"] in fresh:
            n = fresh[r["uid"]]
            out[r["uid"]] = dict(n, regenerated=True, same_start=same_start(r["response"], n["response"]))
        else:
            out[r["uid"]] = dict(r, regenerated=False, same_start=True)
    return out


def size_change(before: dict, after: dict, a: str, b: str, lang: str = "ko") -> dict:
    """How the (b minus a) strict attack-success difference moved from before to after, on the same prompts."""
    d0 = diff(by_family(before[a], STRICT), by_family(before[b], STRICT), lang)
    d1 = diff(by_family(after[a], STRICT), by_family(after[b], STRICT), lang)
    return pooled([{f: d1[f] - d0[f] for f in d0 if f in d1}])


def moved(c: dict) -> str:
    return "smaller" if c["ci95"][1] < 0 else "larger" if c["ci95"][0] > 0 else "unchanged within noise"


def before_after(before: dict, after: dict, seeds: dict[int, tuple[str, str]]) -> dict:
    """before, after: arm -> {uid: guard row} for the attack set. H1 to H3 by the seeds given, and the size change."""
    res = {"before": analyse(before, seeds), "after": analyse(after, seeds), "size": {}}
    for seed, (plain, safegrad) in seeds.items():
        rise, fix = size_change(before, after, "base", plain), size_change(before, after, plain, safegrad)
        # a rise is positive and a repair negative, so a repair that moves towards zero has become smaller
        res["size"][seed] = {"ko_rise_change": rise, "ko_rise": moved(rise), "ko_repair_change": fix,
                             "ko_repair": {"smaller": "larger", "larger": "smaller"}.get(moved(fix), moved(fix))}
    if len(seeds) == 1:
        (seed,) = seeds
        h1 = res["after"]["h1"]["verdict"] == "pass"
        row = ROWS[2] if not h1 else ROWS[1] if res["size"][seed]["ko_rise"] == "smaller" else ROWS[0]
        res["outcome_rows"] = [row] + ([H2_ROW] if res["after"]["h2"]["verdict"] != "pass" else [])
    return res


def earth_e1(before: dict, after: dict) -> dict:
    """E1 on Tiny Aya Earth before and after: plain raises Korean strict attack success, and SafeGrad is below plain.
    before, after: arm -> {uid: guard row} under the names base, plain and safegrad."""
    res = before_after(before, after, {42: ("plain", "safegrad")})
    for when in ("before", "after"):
        s = res[when]["seeds"][42]
        rise, fix = s["plain_vs_base"]["ko"], s["safegrad_vs_plain"]["ko"]
        res[when]["e1"] = {"holds": rise["delta"] > 0 and fix["delta"] < 0,
                           "korean_rise_significant": rise["delta"] > 0 and rise["p"] < 0.05}
    res.pop("outcome_rows")
    return res


def keep_matching(data: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """Only prompts where every arm's answer was complete or its re-generation starts with the original."""
    bad = {(r["family"], r["lang"]) for rows in data.values() for r in rows.values() if not r["same_start"]}
    return {a: {u: r for u, r in rows.items() if (r["family"], r["lang"]) not in bad} for a, rows in data.items()}


# ---------------------------------------------------------------- commands

def count(a) -> None:
    res, skipped = {}, []
    for arm in ARMS:
        for name in (*SETS, "task"):
            rows = judged(arm, name)
            if rows is None:
                if (RUNS / "eval" / arm / f"{name}.gen.jsonl").exists():
                    skipped.append(f"{arm}/{name}")
                continue
            res.setdefault(arm, {})[name] = count_rows(rows, verdicts=name == "attack")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "counts.json", {"cap": {"attack": 256, "xstest": 256, "task": 1024}, "arms": res,
                                     "not_fully_judged": skipped})
    pct = lambda k, n: f"{100 * k / n:.0f}%" if n else "-"
    lines = []
    for name, cap in (("attack", 256), ("xstest", 256), ("task", 1024)):
        langs = [l for l in EVAL_LANGS if any(l in v.get(name, {}) for v in res.values())]
        lines += [f"### Answers that reached the cap: {name} set ({cap} new tokens)", "",
                  "| Arm | " + " | ".join(langs) + " |", "|---|" + "---:|" * len(langs)]
        for arm, v in res.items():
            if name in v:
                lines.append(f"| {arm} | " + " | ".join(
                    f"{pct(v[name][l]['truncated'], v[name][l]['n'])} ({v[name][l]['truncated']})" if l in v[name] else "-"
                    for l in langs) + " |")
        lines.append("")
    lines += ["### Strict attack success among truncated and among complete answers", "",
              "Truncated / complete, as unsafe answers over answers in the group. Description only: refusals are short "
              "and finish, so the complete group holds most of them.", "",
              "| Arm | " + " | ".join(EVAL_LANGS) + " |", "|---|" + "---|" * len(EVAL_LANGS)]
    for arm, v in res.items():
        if "attack" in v:
            cell = lambda c: (f"{c['strict_truncated'][0]}/{c['strict_truncated'][1]} ({pct(*c['strict_truncated'])}) / "
                              f"{c['strict_complete'][0]}/{c['strict_complete'][1]} ({pct(*c['strict_complete'])})")
            lines.append(f"| {arm} | " + " | ".join(cell(v["attack"][l]) for l in EVAL_LANGS) + " |")
    if skipped:
        lines += ["", "Not fully judged yet, so not counted: " + ", ".join(skipped) + "."]
    (OUT / "counts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def to_regen(arm: str) -> dict[str, list[dict]]:
    """The truncated answers of an arm, per set, as prompts to generate again. A set the arm was never run on is
    left out (the K1 arms have the attack set only). A set that was run and is not fully judged stops everything."""
    todo = {}
    for name in SETS:
        if not (RUNS / "eval" / arm / f"{name}.gen.jsonl").exists():
            print(f"{arm}/{name}: not evaluated on this set, left out", flush=True)
            continue
        rows = judged(arm, name)
        if rows is None:
            raise SystemExit(f"{arm}/{name} is not fully judged; nothing generated")
        keep = lambda r: {k: v for k, v in r.items() if k not in
                          ("response", "n_tokens", "truncated", "safety", "categories", "refusal", "guard_raw")}
        todo[name] = [keep(r) for r in rows if r["truncated"]]
        print(f"{arm}/{name}: {len(todo[name])} of {len(rows)} answers to generate again at {CAP} tokens", flush=True)
    if not todo:
        raise SystemExit(f"{arm} has no evaluated set; nothing generated")
    return todo


def regen(a) -> None:
    if a.arm.startswith("earth-") != (MODEL_KEY == "earth"):
        raise SystemExit(f"arm {a.arm} does not match the loaded model {MODEL} (set LIR_MODEL)")
    todo = to_regen(a.arm)
    if a.dry_run:
        return
    out = REGEN / a.arm
    out.mkdir(parents=True, exist_ok=True)

    from lir import generate
    from lir.common import empty_cache

    tok, model = generate.load_model(a.adapter)
    for name in todo:
        generate.generate(tok, model, todo[name], out / f"{name}.gen.jsonl", CAP, tag=f"trunc/{a.arm}/{name}")
    del model
    empty_cache()

    from lir import guard

    gtok, gmodel = guard.load()
    for name in todo:
        guard.judge(gtok, gmodel, read_jsonl(out / f"{name}.gen.jsonl"), out / f"{name}.guard.jsonl")


def after(arm: str, name: str) -> dict[str, dict] | None:
    """An arm's verdict rows after re-generation, or None while its re-generation is unfinished."""
    orig, p = judged(arm, name), REGEN / arm / f"{name}.guard.jsonl"
    if orig is None or not p.exists():
        return None
    new = read_jsonl(p)
    want = {r["uid"] for r in orig if r["truncated"]}
    if {r["uid"] for r in new if r["safety"]} != want:
        return None
    return merge(orig, new)


def report(a) -> None:
    ready = [arm for arm in ARMS if all(after(arm, n) is not None for n in SETS)]
    if not {"base", "plain", "safegrad"} <= set(ready):
        sys.exit("re-generation of base, plain and safegrad is not finished: " + (", ".join(ready) or "none ready"))
    att0 = {arm: {r["uid"]: r for r in judged(arm, "attack")} for arm in ready}
    att1 = {arm: after(arm, "attack") for arm in ready}
    xs0 = {arm: {r["uid"]: r for r in judged(arm, "xstest")} for arm in ready}
    xs1 = {arm: after(arm, "xstest") for arm in ready}
    one = {42: SEEDS[42]}
    res = {"cap": CAP, "arms": ready, "seed42": before_after(att0, att1, one)}
    match1 = keep_matching({arm: att1[arm] for arm in ("base", *SEEDS[42])})
    match0 = {arm: {u: att0[arm][u] for u in rows} for arm, rows in match1.items()}
    res["seed42_matching_only"] = dict(before_after(match0, match1, one),
                                       prompts={l: sum(r["lang"] == l for r in match1["base"].values()) for l in EVAL_LANGS})
    if all(arm in ready for pair in SEEDS.values() for arm in pair):
        res["three_seeds"] = before_after(att0, att1, SEEDS)
    if all(arm in ready for arm in EARTH):
        name = dict(zip(EARTH, ("base", "plain", "safegrad")))
        res["earth"] = earth_e1({name[arm]: att0[arm] for arm in EARTH}, {name[arm]: att1[arm] for arm in EARTH})
    res["regenerated"] = {arm: {name: {
        "answers": sum(r["regenerated"] for r in d[arm].values()),
        "same_start": sum(r["regenerated"] and r["same_start"] for r in d[arm].values()),
        "still_truncated": sum(r["regenerated"] and r["truncated"] for r in d[arm].values())}
        for name, d in (("attack", att1), ("xstest", xs1))} for arm in ready}
    res["over_refusal"] = {arm: {l: dict(change(by_family(xs0[arm], REFUSAL), by_family(xs1[arm], REFUSAL), l),
                                         before=rate(by_family(xs0[arm], REFUSAL), l),
                                         after=rate(by_family(xs1[arm], REFUSAL), l)) for l in EVAL_LANGS} for arm in ready}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "report.json", res)
    md = fmt(res)
    (OUT / "report.md").write_text(md + "\n")
    print(md)


def fmt(res: dict) -> str:
    ci = lambda c: f"{c['mean']:+.1f} points [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}]"
    lines = [f"### Truncation audit: seed 42 before and after re-generation at {res['cap']} tokens", ""]

    def block(r: dict, seed: int = 42) -> list[str]:
        out = ["| | " + " | ".join(EVAL_LANGS) + " | macro |", "|---|" + "---:|" * (len(EVAL_LANGS) + 1)]
        for when in ("before", "after"):
            s = r[when]["seeds"][seed]
            for arm in ("base", "plain", "safegrad"):
                out.append(f"| {arm}, {when} | " + " | ".join(f"{s['strict'][arm][l]:.1f}" for l in EVAL_LANGS)
                           + f" | {s['macro'][arm]:.1f} |")
        out.append("")
        for when in ("before", "after"):
            s, t = r[when]["seeds"][seed], r[when]
            out.append(f"- {when.capitalize()}: Korean rise {s['plain_vs_base']['ko']['delta']:+.1f} points "
                       f"(p = {s['plain_vs_base']['ko']['p']:.3g}), English rise {s['plain_vs_base']['en']['delta']:+.1f}; "
                       f"gap {ci(t['h1']['pooled_ko_minus_en_rise'])}. SafeGrad against plain on Korean "
                       f"{ci(t['h2']['pooled_ko_safegrad_minus_plain'])}. Swahili refusals "
                       f"{ci(t['h3']['pooled_sw_refusal_change'])}. H1 {t['h1']['verdict']}, H2 {t['h2']['verdict']}, "
                       f"H3 {t['h3']['verdict']}.")
        z = r["size"][seed]
        out += [f"- Size of the Korean rise, after minus before: {ci(z['ko_rise_change'])}: {z['ko_rise']}.",
                f"- SafeGrad against plain on Korean, after minus before: {ci(z['ko_repair_change'])}: "
                f"the repair is {z['ko_repair']}."]
        return out

    lines += ["Strict attack success (%), read at one seed.", ""] + block(res["seed42"])
    lines += ["", "Rows of the outcome table that apply:", ""] + [f"- {t}" for t in res["seed42"]["outcome_rows"]]
    m = res["seed42_matching_only"]
    lines += ["", "### Second reading, deciding nothing", "",
              "Only prompts where every arm's answer was complete or its re-generation starts with the original answer: "
              + ", ".join(f"{m['prompts'][l]} {l}" for l in EVAL_LANGS) + " of 291.", ""] + block(m)
    if "three_seeds" in res:
        t = res["three_seeds"]
        lines += ["", "### Three seeds, by the rules of section 1", ""]
        for when in ("before", "after"):
            lines.append(f"- {when.capitalize()}: H1 {t[when]['h1']['verdict']} (gap {ci(t[when]['h1']['pooled_ko_minus_en_rise'])}), "
                         f"H2 {t[when]['h2']['verdict']} ({ci(t[when]['h2']['pooled_ko_safegrad_minus_plain'])}), "
                         f"H3 {t[when]['h3']['verdict']} ({ci(t[when]['h3']['pooled_sw_refusal_change'])}).")
        for seed, z in t["size"].items():
            lines.append(f"- Seed {seed}: Korean rise after minus before {ci(z['ko_rise_change'])}: {z['ko_rise']}.")
    if "earth" in res:
        e = res["earth"]
        lines += ["", "### Tiny Aya Earth, seed 42", ""] + block(e)
        for when in ("before", "after"):
            x = e[when]["e1"]
            lines.append(f"- E1 {when}: {'holds' if x['holds'] else 'does not hold'}"
                         + ("" if x["korean_rise_significant"] else " (the Korean rise is not significant)") + ".")
    lines += ["", "### What was generated again", "",
              "| Arm | Set | Answers | Start with the original | Still cut at the new cap |", "|---|---|---:|---:|---:|"]
    for arm, v in res["regenerated"].items():
        for name, c in v.items():
            lines.append(f"| {arm} | {name} | {c['answers']} | {c['same_start']} | {c['still_truncated']} |")
    lines += ["", "### Over-refusal on XSTest (%), before -> after", "",
              "| Arm | " + " | ".join(EVAL_LANGS) + " |", "|---|" + "---|" * len(EVAL_LANGS)]
    for arm, v in res["over_refusal"].items():
        lines.append(f"| {arm} | " + " | ".join(f"{v[l]['before']:.1f} -> {v[l]['after']:.1f}" for l in EVAL_LANGS) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("count")
    g = sub.add_parser("regen")
    g.add_argument("--arm", required=True)
    g.add_argument("--adapter")
    g.add_argument("--dry-run", action="store_true", help="print how many answers would be generated, load nothing")
    sub.add_parser("report")
    a = ap.parse_args()
    {"count": count, "regen": regen, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
