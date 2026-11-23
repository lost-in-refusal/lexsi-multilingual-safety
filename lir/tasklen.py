"""Preregistration 12h: is the task-score drop after fine-tuning a matter of shorter answers?

  python -m lir.tasklen      # -> results/lir/stage2/task_length.{json,md}

Reads the task answers and the task judge's scores that already exist for base, plain and SafeGrad at seeds 42, 1
and 2 (300 held-out Bactrian-X prompts). No model is run. For every fine-tuned answer the length ratio is its
length in characters over the base model's answer to the same prompt, and the score difference is its task score
minus the base model's on that prompt.
"""
from __future__ import annotations

from statistics import mean, median

from lir.common import ROOT, RUNS, read_jsonl, write_json
from lir.metrics import boot_delta
from lir.seeds import SEEDS

OUT = ROOT / "results" / "lir" / "stage2"
SAME = (0.8, 1.25)     # "about the same length": the fine-tuned answer has 80% to 125% of the base answer's characters
HALF = 0.5             # "under half": fewer than half of the base answer's characters
ENOUGH = 30            # prompts needed in the same-length group before it is read
SUBS = ("accuracy", "instruction_following", "coherence", "fluency")
LANGS = ("en", "ko", "zh")
READINGS = {"level": "Where the fine-tuned answer is about as long as the base model's, its task score is not detectably different",
            "lower": "The fine-tuned models score lower even where their answers are about as long as the base model's, so the drop "
                     "is not only a matter of length",
            "higher": "Where the fine-tuned answer is about as long as the base model's, it scores higher",
            "too few": "Too few fine-tuned answers are about as long as the base model's to tell"}


def load(arm: str) -> dict[str, dict]:
    """uid -> language, answer length in characters, task score and sub-scores."""
    gen = {r["uid"]: r for r in read_jsonl(RUNS / "eval" / arm / "task.gen.jsonl")}
    out = {}
    for r in read_jsonl(RUNS / "eval" / arm / "task.judge.jsonl"):
        if r.get("task_score") is None or any(r["scores"].get(k) is None for k in SUBS):
            continue   # an answer the judge's reply could not be parsed for
        g = gen[r["uid"]]
        out[r["uid"]] = {"lang": r["lang"], "chars": len(g["response"]), "ref_chars": len(g["reference"]),
                         "score": r["task_score"], "subs": r["scores"]}
    return out


def group_of(ratio: float) -> str:
    return "same" if SAME[0] <= ratio <= SAME[1] else "longer" if ratio > SAME[1] else "under half" if ratio < HALF else "shorter"


def pooled(values: dict[str, list[float]]) -> dict:
    """Mean over prompts of each prompt's mean value, with a bootstrap 95% interval over prompts."""
    per = [mean(v) for v in values.values()]
    lo, hi = boot_delta([(0.0, x) for x in per])
    return {"mean": mean(per), "ci95": [lo, hi], "prompts": len(per)}


def by_length(base: dict, arms: list[dict]) -> dict:
    """Score difference against base by length group, over the arms (one per seed)."""
    groups: dict[str, dict[str, list[float]]] = {g: {} for g in ("same", "shorter", "under half", "longer")}
    rows = {g: 0 for g in groups}
    for arm in arms:
        for uid, a in arm.items():
            if uid not in base:
                continue
            g = group_of(a["chars"] / max(1, base[uid]["chars"]))
            groups[g].setdefault(uid, []).append(a["score"] - base[uid]["score"])
            rows[g] += 1
    out = {g: dict(pooled(v), answers=rows[g]) if v else {"prompts": 0, "answers": 0} for g, v in groups.items()}
    same = out["same"]
    kind = ("too few" if same["prompts"] < ENOUGH else "lower" if same["ci95"][1] < 0 else "higher" if same["ci95"][0] > 0 else "level")
    return {"groups": out, "reading": kind, "report": READINGS[kind]}


def describe(base: dict, arms: list[dict]) -> dict:
    """Description with no rule: scores, sub-score changes and lengths per language, averaged over the arms."""
    out = {}
    for lang in LANGS:
        ids = [u for u, r in base.items() if r["lang"] == lang and all(u in a for a in arms)]
        if not ids:
            continue
        out[lang] = {"score_base": mean(base[u]["score"] for u in ids),
                     "score_arm": mean(mean(a[u]["score"] for u in ids) for a in arms),
                     "median_chars_base": median(base[u]["chars"] for u in ids),
                     "median_chars_arm": mean(median(a[u]["chars"] for u in ids) for a in arms),
                     "median_chars_reference": median(base[u]["ref_chars"] for u in ids),
                     "sub_score_change": {s: mean(mean(a[u]["subs"][s] - base[u]["subs"][s] for u in ids) for a in arms) for s in SUBS}}
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def result() -> dict:
    base = load("base")
    res = {"same_length": list(SAME), "under_half": HALF, "methods": {}}
    for i, name in enumerate(("plain", "safegrad")):
        arms = [load(pair[i]) for pair in SEEDS.values()]
        res["methods"][name] = dict(by_length(base, arms), by_language=describe(base, arms))
    res["base_length_against_score"] = {l: spearman([r["chars"] for r in base.values() if r["lang"] == l],
                                                    [r["score"] for r in base.values() if r["lang"] == l]) for l in LANGS}
    return res


def md(res: dict) -> str:
    ci = lambda c: f"{c['mean']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]" if c.get("prompts") else "none"
    names = {"plain": "Plain LoRA", "safegrad": "SafeGrad LoRA"}
    lines = ["### Answer length and the task score", "",
             "Task score (1 to 7, Qwen3-14B judge) on the 300 held-out Bactrian-X prompts, for plain and SafeGrad at seeds 42, 1 "
             "and 2 against the base model. Each fine-tuned answer is grouped by its length in characters relative to the base "
             f"model's answer to the same prompt: about the same ({res['same_length'][0]:.0%} to {res['same_length'][1]:.0%}), shorter, "
             f"under half, or longer. Section 12h of the preregistration fixed the groups and the reading before this was computed. "
             "The groups are formed after generation, so this locates the drop; it does not show what causes it.", "",
             "Task score minus the base model's on the same prompt, by length group (95% interval over prompts):", "",
             "| Method | About the same length | Shorter (50% to 80%) | Under half | Longer |", "|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        g = r["groups"]
        cell = lambda k: f"{ci(g[k])}, {g[k]['answers']} answers" if g[k].get("prompts") else "none"
        lines.append(f"| {names[m]} | {cell('same')} | {cell('shorter')} | {cell('under half')} | {cell('longer')} |")
    for m, r in res["methods"].items():
        lines += ["", f"**{names[m]}: {r['report']}.**"]
    lines += ["", "#### Description, with no rule", "",
              "| Method | Language | Task score, base to fine-tuned | Median characters: base, fine-tuned, reference | Accuracy | Instruction following | Coherence | Fluency |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for m, r in res["methods"].items():
        for l, d in r["by_language"].items():
            s = d["sub_score_change"]
            lines.append(f"| {names[m]} | {l} | {d['score_base']:.2f} to {d['score_arm']:.2f} | {d['median_chars_base']:.0f}, "
                         f"{d['median_chars_arm']:.0f}, {d['median_chars_reference']:.0f} | " + " | ".join(f"{s[k]:+.2f}" for k in SUBS) + " |")
    lines += ["", "Sub-score columns are the change against the base model, averaged over the three seeds.", "",
              "Within the base model's own answers, the rank correlation between answer length and task score is "
              + ", ".join(f"{v:+.2f} in {l}" for l, v in res["base_length_against_score"].items()) + "."]
    return "\n".join(lines)


def main() -> None:
    res = result()
    write_json(OUT / "task_length.json", res)
    (OUT / "task_length.md").write_text(md(res) + "\n")
    print(md(res))


if __name__ == "__main__":
    main()
