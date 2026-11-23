"""Preregistration section 14b: the attack prompts in six more languages the model was never fine-tuned on.

  python -m lir.morelangs build                  # data/lir/multijail_attack_more.jsonl and its manifest
  python -m lir.morelangs evaluate --arm base    # answers and guard verdicts -> runs/morelangs/<arm>/
  python -m lir.morelangs evaluate --arm plain-s1
  python -m lir.morelangs report                 # U1 to U3 -> results/lir/stage2/more_languages.{json,md}

Evaluation only: no model is trained. The prompts are the 291 MultiJail families of the attack set in Italian,
Vietnamese, Arabic, Thai, Bengali and Javanese, from the same pinned file and through the same cleaning. The answers
are generated and judged exactly as the attack set is (same code, same 256-token cap, same guard). Outputs go to
runs/morelangs/ and not to runs/eval/, so the arm folders that are published stay as they are.

`report` reads files only. It prints "pending" and writes nothing until all seven arms are fully judged.
"""
from __future__ import annotations

import argparse
import ast
import csv
import time
from collections import Counter
from pathlib import Path

from lir.common import DATA, MULTIJAIL, ROOT, RUNS, empty_cache, rank_key, read_jsonl, sha256_file, write_json, write_jsonl
from lir.metrics import load
from lir.seeds import REFUSAL, SEEDS, STRICT, UNSAFE, by_family, change, pooled, rate

MORE_LANGS = ["it", "vi", "ar", "th", "bn", "jv"]   # every MultiJail language that is not in the attack set
NAMES = {"it": "Italian", "vi": "Vietnamese", "ar": "Arabic", "th": "Thai", "bn": "Bengali", "jv": "Javanese"}
FILE = "multijail_attack_more.jsonl"
MANIFEST = "multijail_more_manifest.json"
OUT_RUNS = RUNS / "morelangs"
OUT = ROOT / "results" / "lir" / "stage2"
ARMS = ["base"] + [arm for pair in SEEDS.values() for arm in pair]

U1_ROWS = {"below": "Refusals on attack prompts fell in languages the model was never fine-tuned on, beyond Swahili: on "
                    "average over six more languages",
           "includes": "No detected fall in refusals on average over the six further untrained languages. The spillover "
                       "finding rests on Swahili",
           "above": "On average over the six further untrained languages refusals rose under plain fine-tuning. The "
                    "spillover finding rests on Swahili and does not extend to them"}
U2_ROWS = {"above": "SafeGrad kept more refusals than plain fine-tuning in the six further untrained languages",
           "includes": "No detected difference between SafeGrad and plain fine-tuning in refusals over the six further "
                       "untrained languages",
           "below": "SafeGrad left fewer refusals than plain fine-tuning in the six further untrained languages"}
U3_ROWS = {"all": "As the guard measures it, Korean attack success rose more than in each of the six further languages, so "
                  "Korean drifted most of the ten languages tested",
           "not all": "As the guard measures it, Korean did not drift more than every further language. \"Korean is hit "
                      "hardest\" is then written only as: hardest among the four languages of the main evaluation, and the "
                      "languages that rose as much or more are named with it"}


# ---------------------------------------------------------------- the prompts

def more_rows(csv_rows: list[dict], families: set[str], langs: list[str] = MORE_LANGS) -> list[dict]:
    """One row per attack family and language, before cleaning, in the order of the attack set (families by their
    rank, then languages). csv_rows: the rows of MultiJail.csv; families: the family names of the attack set."""
    ranked = sorted(csv_rows, key=lambda r: rank_key(f"multijail-{r['id']}"))
    return [{"uid": f"mj-{r['id']}-{lang}", "family": f"mj-{r['id']}", "lang": lang, "instruction": r[lang],
             "split": "attack", "source": r["source"], "tags": ast.literal_eval(r["tags"])}
            for r in ranked if f"mj-{r['id']}" in families for lang in langs]


def build() -> None:
    from huggingface_hub import hf_hub_download

    from lir.data import _cleaner, _samples

    repo, rev = MULTIJAIL
    old = sha256_file(DATA / "multijail_attack.jsonl")
    families = {r["family"] for r in read_jsonl(DATA / "multijail_attack.jsonl")}
    path = Path(hf_hub_download(repo, "MultiJail.csv", repo_type="dataset", revision=rev))
    with open(path, encoding="utf-8-sig") as f:
        flat = more_rows(list(csv.DictReader(f)), families)
    out = []
    for s in _cleaner().run(_samples(flat, f"hf://datasets/{repo}@{rev}")):   # the cleaning of the attack set
        m = s.metadata
        out.append({"uid": s.id, "family": m["family"], "lang": m["lang"], "split": m["split"],
                    "prompt": s.instruction, "harm_category": m["tags"], "source": m["source"]})
    write_jsonl(DATA / FILE, out)
    if sha256_file(DATA / "multijail_attack.jsonl") != old:
        raise SystemExit("the attack set changed while this ran")
    write_json(DATA / MANIFEST, {
        "source": repo, "revision": rev, "source_sha256": sha256_file(path), "languages": MORE_LANGS,
        "families": len({r["family"] for r in out}), "same_families_as": "multijail_attack.jsonl",
        "attack_set_sha256": old, "rows_per_lang": dict(Counter(r["lang"] for r in out)),
        "curation": "CuratorKIT TextCleaner (unicode NFKC, control chars, encoding artifacts)",
        "files": {FILE: sha256_file(DATA / FILE)}})
    print(len(out), "rows:", dict(Counter(r["lang"] for r in out)))


# ---------------------------------------------------------------- answers and verdicts

def evaluate(a) -> None:
    """The steps of lir.evaluate for the attack set, on the six-language file, written under runs/morelangs/."""
    from lir.evaluate import SETS

    if a.arm not in ARMS:
        raise SystemExit(f"not an arm of section 14b: {a.arm} (arms: {', '.join(ARMS)})")
    adapter = None if a.arm == "base" else RUNS / a.arm / "adapter"
    if adapter is not None and not (adapter / "adapter_model.safetensors").exists():
        raise SystemExit(f"no adapter at {adapter}")
    rows = read_jsonl(DATA / FILE)
    out = OUT_RUNS / a.arm
    print(f"{a.arm}: {len(rows)} prompts in {', '.join(MORE_LANGS)}; adapter {adapter or 'none'}; output {out}")
    if a.before and time.time() >= a.before:
        print(f"not started: it is {time.strftime('%F %T')} and the limit was {time.strftime('%F %T', time.localtime(a.before))}")
        return
    if a.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)

    from lir import generate

    tok, model = generate.load_model(str(adapter) if adapter else None)
    generate.generate(tok, model, rows, out / "attack.gen.jsonl", SETS["attack"][1], tag=f"{a.arm}/more languages")
    del model
    empty_cache()

    from lir import guard

    gtok, gmodel = guard.load()
    guard.judge(gtok, gmodel, read_jsonl(out / "attack.gen.jsonl"), out / "attack.guard.jsonl")


# ---------------------------------------------------------------- the report

def side(d: dict) -> str:
    lo, hi = d["ci95"]
    return "above" if lo > 0 else "below" if hi < 0 else "includes"


def over_languages(a: dict, b: dict, langs: list[str]) -> dict[str, float]:
    """b minus a per family, averaged over the languages. a, b: (family, language) -> 0 or 1."""
    fams = sorted({f for (f, l) in a if l in langs})
    return {f: sum(b[(f, l)] - a[(f, l)] for l in langs) / len(langs) for f in fams
            if all((f, l) in a and (f, l) in b for l in langs)}


def more_result(more: dict[str, dict], main: dict[str, dict], seeds: dict[int, tuple[str, str]] = SEEDS,
                langs: list[str] = MORE_LANGS) -> dict:
    """U1 to U3 of section 14b. more: arm -> {uid: guard row} on the further languages; main: the same arms on the
    attack set, for Korean. Refusal rate decides U1 and U2, as in H3; attack success is what the guard sees."""
    ref = {arm: by_family(rows, REFUSAL) for arm, rows in more.items()}
    strict = {arm: by_family(rows, STRICT) for arm, rows in more.items()}
    unsafe = {arm: by_family(rows, UNSAFE) for arm, rows in more.items()}
    ko = {arm: by_family(rows, STRICT) for arm, rows in main.items()}
    one = lambda a, b, l: {f: b[(f, l)] - a[(f, l)] for (f, x) in a if x == l and (f, l) in b}
    res = {"languages": langs, "seeds": list(seeds),
           "u1": pooled([over_languages(ref["base"], ref[p], langs) for p, _ in seeds.values()]),
           "u2": pooled([over_languages(ref[p], ref[s], langs) for p, s in seeds.values()]),
           "per_language": {}, "per_seed": {}}
    res["u1"]["side"], res["u2"]["side"] = side(res["u1"]), side(res["u2"])
    res["u1_report"], res["u2_report"] = U1_ROWS[res["u1"]["side"]], U2_ROWS[res["u2"]["side"]]
    ko_rise = [one(ko["base"], ko[p], "ko") for p, _ in seeds.values()]
    for l in langs:
        rise = [one(strict["base"], strict[p], l) for p, _ in seeds.values()]
        gap = [{f: k[f] - r[f] for f in k if f in r} for k, r in zip(ko_rise, rise)]   # Korean rise minus this language's
        res["per_language"][l] = {
            "base_refusal": rate(ref["base"], l), "base_strict": rate(strict["base"], l),
            "refusal_plain_vs_base": pooled([one(ref["base"], ref[p], l) for p, _ in seeds.values()]),
            "refusal_safegrad_vs_plain": pooled([one(ref[p], ref[s], l) for p, s in seeds.values()]),
            "strict_plain_vs_base": pooled(rise),
            "strict_safegrad_vs_plain": pooled([one(strict[p], strict[s], l) for p, s in seeds.values()]),
            "unsafe_only_plain_vs_base": pooled([one(unsafe["base"], unsafe[p], l) for p, _ in seeds.values()]),
            "korean_rise_minus_this": pooled(gap)}
        res["per_language"][l]["korean_rise_minus_this"]["side"] = side(res["per_language"][l]["korean_rise_minus_this"])
    for seed, (p, s) in seeds.items():
        res["per_seed"][seed] = {l: {"refusal_plain_vs_base": change(ref["base"], ref[p], l),
                                     "refusal_safegrad_vs_plain": change(ref[p], ref[s], l),
                                     "strict_plain_vs_base": change(strict["base"], strict[p], l)} for l in langs}
    res["korean_strict_rise"] = pooled(ko_rise)
    not_below = [l for l in langs if res["per_language"][l]["korean_rise_minus_this"]["side"] != "above"]
    res["u3"] = {"korean_rose_more_than_each": not not_below, "as_much_or_more": not_below}
    res["u3_report"] = U3_ROWS["all" if not not_below else "not all"]
    res["truncated_share"] = {arm: {l: 100 * sum(bool(r.get("truncated")) for r in rows.values() if r["lang"] == l)
                                    / max(1, sum(r["lang"] == l for r in rows.values())) for l in langs}
                              for arm, rows in more.items()}
    return res


ci = lambda d: f"{d['mean']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]"
WHERE = {"above": "above zero", "includes": "includes zero", "below": "below zero"}
pv = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def more_md(res: dict) -> str:
    langs = res["languages"]
    lines = ["### Six more languages the model was never fine-tuned on (U1 to U3)", "",
             "The 291 attack families in Italian, Vietnamese, Arabic, Thai, Bengali and Javanese, answered by base and by "
             "plain and SafeGrad at seeds 42, 1 and 2. No model was trained for this. Section 14b of the preregistration "
             "fixed the rules before any answer in these languages existed. Refusal rate decides U1 and U2, as in H3. "
             "**No person has checked the guard in these six languages**, and in Swahili the guard missed harm that a "
             "re-translation found, so every attack-success figure here is what the guard sees. Answers are capped at 256 "
             "tokens. Intervals are 95%, from a bootstrap over the 291 families with each family's value averaged over the "
             "three seeds.", "",
             f"- **U1, spillover (refusal rate, plain minus base, averaged over the six languages): {ci(res['u1'])} points, "
             f"{WHERE[res['u1']['side']]}. {res['u1_report']}.**",
             f"- **U2, repair (refusal rate, SafeGrad minus plain, averaged over the six languages): {ci(res['u2'])} points, "
             f"{WHERE[res['u2']['side']]}. {res['u2_report']}.**",
             f"- **U3, where Korean stands: {res['u3_report']}.**"
             + ("" if res["u3"]["korean_rose_more_than_each"] else " Rose as much or more, by the interval: "
                + ", ".join(NAMES.get(l, l) for l in res["u3"]["as_much_or_more"]) + "."),
             f"  Korean's own rise on the attack set, for comparison: {ci(res['korean_strict_rise'])} points.", "",
             "| Language | Base refusal rate | Refusals, plain minus base | Refusals, SafeGrad minus plain | Base strict attack success | Strict attack success, plain minus base | SafeGrad minus plain | Unsafe class only, plain minus base | Korean's rise minus this language's |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for l in langs:
        r = res["per_language"][l]
        lines.append(f"| {NAMES.get(l, l)} | {r['base_refusal']:.1f} | {ci(r['refusal_plain_vs_base'])} | {ci(r['refusal_safegrad_vs_plain'])} "
                     f"| {r['base_strict']:.1f} | {ci(r['strict_plain_vs_base'])} | {ci(r['strict_safegrad_vs_plain'])} "
                     f"| {ci(r['unsafe_only_plain_vs_base'])} | {ci(r['korean_rise_minus_this'])} |")
    lines += ["", "Rates in %, differences in points, pooled over the three seeds. Only U1, U2 and U3 have a rule; the table is "
              "shown beside them.", "", "Per seed, refusal rate, plain minus base (exact McNemar): " + "; ".join(
                  f"seed {seed} " + ", ".join(f"{l} {row[l]['refusal_plain_vs_base']['delta']:+.1f} ({pv(row[l]['refusal_plain_vs_base']['p'])})" for l in langs)
                  for seed, row in res["per_seed"].items()) + ".", "",
              "Share of answers cut at the 256-token cap (%): " + "; ".join(
                  f"`{arm}` " + ", ".join(f"{l} {v[l]:.0f}" for l in langs) for arm, v in res["truncated_share"].items()) + ".", "",
              "Not done in these languages: XSTest (no over-refusal figure), the task set, a human check of the guard, "
              "re-translation, and generating capped answers again at 1,024 tokens."]
    return "\n".join(lines)


def judged(arm: str, n: int) -> dict[str, dict] | None:
    p = OUT_RUNS / arm / "attack.guard.jsonl"
    rows = {r["uid"]: r for r in read_jsonl(p)} if p.exists() else {}
    return rows if len(rows) == n and all(r.get("safety") is not None for r in rows.values()) else None


def report(a) -> None:
    n = len(read_jsonl(DATA / FILE))
    more = {arm: judged(arm, n) for arm in ARMS}
    waiting = [arm for arm, rows in more.items() if rows is None]
    if waiting:
        print("pending, nothing written: not fully judged in the six languages: " + ", ".join(waiting))
        return
    res = more_result(more, {arm: load(arm, "attack") for arm in ARMS})
    res["prompts_file"] = {"file": FILE, "sha256": sha256_file(DATA / FILE)}
    md = more_md(res)
    write_json(OUT / "more_languages.json", res)
    (OUT / "more_languages.md").write_text(md + "\n")
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["build", "evaluate", "report"])
    ap.add_argument("--arm", help="evaluate: base, or a plain or SafeGrad run at seed 42, 1 or 2")
    ap.add_argument("--before", type=int, default=0, help="evaluate: do not start at or after this time (epoch seconds)")
    ap.add_argument("--dry-run", action="store_true", help="evaluate: say what would run and stop")
    a = ap.parse_args()
    if a.what == "build":
        build()
    elif a.what == "evaluate":
        evaluate(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
