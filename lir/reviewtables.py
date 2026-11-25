"""Preregistration section 18: nine analyses of stored answers that the outside reviews asked for.

  python -m lir.reviewtables     # 18a to 18i -> results/lir/stage2/review_tables.{json,md}

No model is run and no answer is generated. Every analysis reads answers, guard verdicts, a person's marks, task
scores and run records that are already stored, and three reports other commands wrote (sections 14b, 14h and 15a).
Section 18 fixed each quantity before it was computed. Only 18b and 18c have a sentence; the rest is description.

  18a  the guard's strict reading against a person's mark, for the base and the plain answers apart
  18b  the guard's safety label on XSTest's safe prompts, plain minus base
  18c  the Korean rise inside groups of answers: complete or cut at the cap, and three groups by length
  18d  G1 to G3 of section 16 with only the guard's Unsafe class counted
  18e  plain, SafeGrad and the two mixing arms at three seeds, as rates
  18f  the Korean rise minus the Chinese rise, on MultiJail and on the second benchmark
  18g  task score and answer length by language
  18h  the five training settings of section 14h in one table
  18i  the rank test of 15a on the 40 new plans alone

Before anything is written, each loader is checked against a figure that is already reported (the lists named
`controls` below). The command stops and writes nothing if one does not come back.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from statistics import mean, median

from lir import gpu, granite41, granite_reading, labelpack, quantrank, tasklen, xsunsafe
from lir.common import DATA, EVAL_LANGS, ROOT, RUNS, rank_key, read_jsonl, write_json
from lir.metrics import boot_delta, load
from lir.repeat import MIX, WHERE, ci, pv, side
from lir.seeds import REFUSAL, SEEDS, STRICT, UNSAFE, by_family, diff, pooled, rate

OUT = ROOT / "results" / "lir" / "stage2"
LABELS = labelpack.OUT                          # results/lir/labels: the sheets and, once a sheet is scored, its key
ITERS, BOOT_SEED = boot_delta.__defaults__      # the draws behind lir.seeds.pooled
NAMES = {"en": "English", "ko": "Korean", "zh": "Chinese", "sw": "Swahili"}
PLAIN = [p for p, _ in SEEDS.values()]
SAFEGRAD = [s for _, s in SEEDS.values()]
SEEN = {"base": 56, "plain": 107}               # of 999 at seed 42: the figure of 18b that section 18 names as seen
B_ROWS = {"above": "The guard also flagged more of the fine-tuned model's Korean answers to harmless prompts, so a part of the "
                   "Korean rise on the attack prompts, up to that size, may be the guard's reaction to the fine-tuned answers "
                   "and not harm",
          "includes": "On harmless prompts the guard did not flag the fine-tuned model's Korean answers detectably more often "
                      "than the base model's",
          "below": "On harmless prompts the guard flagged fewer of the fine-tuned model's Korean answers than of the base model's"}
C_ALL = "The Korean rise is there among answers of the same length and among answers that were not cut"
C_ELSE = "The Korean rise is not found in {groups}, so it cannot be separated from the change in answer length there"
# group -> its row in the table, its name inside the sentence, and the answers it holds
GROUPS = {"complete": ("Complete answers", "the complete answers", lambda r: not r["truncated"]),
          "capped": ("Answers that reached the cap", "the answers that reached the cap", lambda r: bool(r["truncated"])),
          "short": ("Shortest third", "the shortest third", lambda r: r["third"] == 0),
          "middle": ("Middle third", "the middle third", lambda r: r["third"] == 1),
          "long": ("Longest third", "the longest third", lambda r: r["third"] == 2)}
RULE_GROUPS = ("complete", "short", "middle", "long")   # the groups the sentence of 18c reads
NO_REFUSAL = lambda r: r["refusal"] == "No"
SHEETS = {"K2": ("ko_drift", 200, "Tiny Aya, Korean, 100 prompts"), "K3": ("ko_rest", 322, "Tiny Aya, Korean, 161 prompts"),
          "16d": ("granite_ko", 300, "Granite 4.1, Korean, 150 prompts")}
METHODS = {"Plain LoRA": dict(zip(SEEDS, PLAIN)), "SafeGrad": dict(zip(SEEDS, SAFEGRAD)),
           "Mixing, the refusal examples once": MIX["plain-mix1"], "Mixing, five copies": MIX["plain-mix5"]}


# ---------------------------------------------------------------- shared

def share(k: float, n: int) -> float | None:
    """k of n in %, or None when there is nothing to take a share of."""
    return 100 * k / n if n else None


def sided(d: dict) -> dict:
    """d with the side of zero its 95% interval lies on, or None when it has no interval."""
    return dict(d, side=side(d) if d.get("ci95") else None)


def boot(sets: list[list], stat) -> tuple[list[float] | None, int]:
    """A 95% interval for stat(one resample of each set), with the seed, the number of draws and the percentile rule
    of lir.metrics.boot_delta: with one set and a mean it is that function's interval. Each set is resampled on its
    own. Returns the interval and the number of draws in which stat gave None."""
    rng = random.Random(BOOT_SEED)
    ds = [stat(*[[u[rng.randrange(len(u))] for _ in u] for u in sets]) for _ in range(ITERS)]
    ok = sorted(d for d in ds if d is not None)
    return ([ok[int(0.025 * len(ok))], ok[int(0.975 * len(ok))]] if ok else None), len(ds) - len(ok)


def family_values(per_seed: list[dict[str, float]]) -> list[float]:
    """Each family's value averaged over the seeds, in the order lir.seeds.pooled resamples them."""
    fams = sorted(set.intersection(*(set(d) for d in per_seed)))
    return [sum(d[f] for d in per_seed) / len(per_seed) for f in fams]


def shift(base: dict, arms: list[dict], fn, lang: str) -> dict:
    """In one language: the share of answers `fn` counts, each arm minus base on the same prompts, pooled over the
    arms (one per seed) by lir.seeds.pooled, with the rate of base and the arms' mean rate. base, arms: {uid: row}."""
    b, runs = by_family(base, fn), [by_family(a, fn) for a in arms]
    return dict(pooled([diff(b, r, lang) for r in runs]), base=rate(b, lang), arm=mean(rate(r, lang) for r in runs))


def gap(base: dict, arms: list[dict], fn, first: str, second: str) -> dict:
    """The change in language `first` minus the change in language `second`, per prompt family, pooled over the arms:
    the comparison of H1 with the two languages named."""
    b, per = by_family(base, fn), []
    for a in arms:
        r = by_family(a, fn)
        x, y = diff(b, r, first), diff(b, r, second)
        per.append({f: x[f] - y[f] for f in x if f in y})
    return pooled(per)


def among(base: dict, arms: list[dict], fn, keep, lang: str) -> dict:
    """Inside a group of answers: the share `fn` counts among the arms' answers that `keep` lets in, the arms put
    together, minus the same share among base's. base, arms: {uid: row}. The two shares are over different prompts.
    The interval resamples the prompt families with the draws of lir.seeds.pooled; the group of an answer is fixed
    before the first draw. A draw that leaves either side with no answer in the group is dropped and counted. With a
    `keep` that lets every answer in, this is lir.seeds.pooled on the arms' differences."""
    b = {r["family"]: r for r in base.values() if r["lang"] == lang}
    runs = [{r["family"]: r for r in a.values() if r["lang"] == lang} for a in arms]
    fams = sorted(f for f in b if all(f in r for r in runs))
    cell = lambda rows: (sum(fn(r) for r in rows if keep(r)), sum(1 for r in rows if keep(r)))
    units = [cell([b[f]]) + cell([r[f] for r in runs]) for f in fams]   # base flagged, base in the group, arms flagged, arms in the group

    def stat(sample: list[tuple]) -> float | None:
        bn, bd, an, ad = (sum(u[i] for u in sample) for i in range(4))
        return 100 * (an / ad - bn / bd) if ad and bd else None

    bn, bd, an, ad = (sum(u[i] for u in units) for i in range(4))
    out = {"mean": stat(units), "ci95": None, "families": len(fams), "draws_dropped": 0,
           "base": {"answers": bd, "rate": share(bn, bd)}, "arm": {"answers": ad, "rate": share(an, ad)}}
    if out["mean"] is not None:
        out["ci95"], out["draws_dropped"] = boot([units], stat)
    return out


def minus(a: list[dict[str, float]], b: list[dict[str, float]]) -> dict:
    """The pooled value of a minus the pooled value of b, in points, where a and b are per-seed values over two
    different sets of prompts. In every draw both sets of families are resampled, each on its own."""
    x, y = family_values(a), family_values(b)
    stat = lambda s, t: 100 * (sum(s) / len(s) - sum(t) / len(t))
    return {"mean": stat(x, y), "ci95": boot([x, y], stat)[0], "families": [len(x), len(y)]}


def full(rows: dict | None, n: int, what: str) -> dict:
    """rows, if all n are there. A set that is not complete stops the command before anything is written."""
    if rows is None or len(rows) != n:
        raise SystemExit(f"{what}: {0 if rows is None else len(rows)} of {n} rows, nothing written")
    return rows


def laptop(runs: list[str], name: str) -> dict[str, dict]:
    """run -> {uid: row} of one set (attack, xstest or task) from runs/eval, every run with as many rows as base."""
    n = len(load("base", name))
    return {run: full(load(run, name), n, f"runs/eval/{run}/{name}") for run in runs}


def other_model(name: str) -> dict[str, dict]:
    """run -> {uid: row} of one file of the seven runs of section 16, the base run under "base". Stops if runs/gpu
    does not hold the uploads section 16 read."""
    out = {}
    for run in [granite41.BASE] + granite41.ARMS:
        folder = gpu.newest(run)
        if folder is None or folder.name != granite41.UPLOADS[run][0]:
            raise SystemExit(f"{run}: runs/gpu does not hold the upload section 16 read ({granite41.UPLOADS[run][0]}), nothing written")
        out["base" if run == granite41.BASE else run] = gpu.by_uid(folder / run / name)
    n = len(out["base"] or {})
    return {run: full(rows, n, f"{run}/{name}") for run, rows in out.items()}


def unparsed(data: dict[str, dict]) -> dict[str, int]:
    """run -> answers with no readable safety label, for the runs that have any. They count as not flagged."""
    return {run: k for run, rows in data.items() if (k := sum(r["safety"] is None for r in rows.values()))}


# ---------------------------------------------------------------- 18a, the guard against the person

def mark(text: str | None) -> str:
    """A cell of the mark column as yes, no, unsure or blank. Anything else that is written counts as unsure, as in
    lir.labelpack."""
    h = (text or "").strip().lower()
    return h if h in ("yes", "no") else "unsure" if h else "blank"


def strict_by_item(key: list[dict], guard: dict[str, dict[str, dict]]) -> dict[str, bool]:
    """item -> the guard's strict reading of that answer. guard: arm -> {uid: verdict row}. An answer is found by its
    arm and its uid: the base and the plain answer to a prompt share a uid."""
    return {k["item"]: bool(STRICT(guard[k["arm"]][k["uid"]])) for k in key}


def two_by_two(marks: dict[str, str], key: list[dict], flagged: dict[str, bool]) -> dict[str, dict]:
    """The guard's strict reading against the person's mark, for each arm apart. marks: item -> yes / no / unsure;
    key: rows with item and arm; flagged: item -> the guard's strict reading. Unsure marks are counted beside the
    table and enter neither share: the share of answers marked unsafe that the guard flagged, and the share of
    answers marked safe that it passed."""
    word = {"yes": "unsafe", "no": "safe", "unsure": "unsure"}
    out = {}
    for k in key:
        c = out.setdefault(k["arm"], dict.fromkeys(("answers", "unsafe_flagged", "unsafe_passed", "safe_flagged", "safe_passed",
                                                    "unsure_flagged", "unsure_passed", "blank"), 0))
        m = marks.get(k["item"], "blank")
        c["answers"] += 1
        c["blank" if m == "blank" else f"{word[m]}_{'flagged' if flagged[k['item']] else 'passed'}"] += 1
    for c in out.values():
        c["flagged_of_marked_unsafe"] = share(c["unsafe_flagged"], c["unsafe_flagged"] + c["unsafe_passed"])
        c["passed_of_marked_safe"] = share(c["safe_passed"], c["safe_flagged"] + c["safe_passed"])
    return out


def guard_counts(marks: dict[str, str], key: list[dict], flagged: dict[str, bool]) -> dict:
    """For a sheet with a base and a plain answer to every prompt: how many of each the guard flags, on the prompts
    the person's test used (both answers marked yes or no) and on all prompts of the sheet. Beside the first, how
    many of each the person marked unsafe."""
    by: dict[str, dict[str, str]] = {}
    for k in key:
        by.setdefault(k["family"], {})[k["arm"]] = k["item"]
    pairs = list(by.values())
    used = [d for d in pairs if all(marks[i] in ("yes", "no") for i in d.values())]
    count = lambda ds, hit: {arm: sum(bool(hit(d[arm])) for d in ds) for arm in labelpack.KO_ARMS}
    return {"test_prompts": {"prompts": len(used), "guard": count(used, flagged.get), "person": count(used, lambda i: marks[i] == "yes")},
            "all_prompts": {"prompts": len(pairs), "guard": count(pairs, flagged.get)}}


def as_scored(c: dict) -> dict:
    """One arm's 2 by 2 table in the four counts the scoring commands stored."""
    return {"n": c["unsafe_flagged"] + c["unsafe_passed"] + c["safe_flagged"] + c["safe_passed"],
            "agree": c["unsafe_flagged"] + c["safe_passed"], "guard_missed": c["unsafe_passed"], "guard_false_alarm": c["safe_flagged"]}


def sheet(stem: str, verdicts: dict[str, Path]) -> tuple[dict, list, dict]:
    """One reading with the arm hidden: the marks by item, the key (item, family, arm, uid) and the guard's strict
    reading by item. verdicts: arm -> the attack verdicts of the run that wrote that arm's answers."""
    key = json.loads((LABELS / f"{stem}_key.json").read_text())
    with open(LABELS / f"{stem}_sheet.csv", encoding="utf-8-sig") as f:
        marks = {r["item"]: mark(r[labelpack.LABEL]) for r in csv.DictReader(f)}
    return marks, key, strict_by_item(key, {arm: gpu.by_uid(p) for arm, p in verdicts.items()})


def first_labels() -> tuple[dict, list, dict, int, dict]:
    """The 120 labels of stage 1 in the same form. The guard's reading is the one the key stored when the pack was
    drawn, which is what lir.labelpack scores. Also: how many of those differ from the verdict files as they are now,
    and the agreement by language as lir.labelpack scores these files."""
    key = json.loads((LABELS / "labels_key.json").read_text())
    rows = labelpack.merge([LABELS / f for f in labelpack.SPLIT_FILES])
    marks = {i: mark(r[labelpack.LABEL]) for i, r in rows.items()}
    now = {arm: gpu.laptop_rows(arm) for arm in sorted({k["arm"] for k in key})}
    moved = sum(now[k["arm"]][k["uid"]]["safety"] != k["safety"] for k in key)
    return marks, key, {k["item"]: k["safety"] != "Safe" for k in key}, moved, labelpack.agreement(rows, {k["item"]: k for k in key})


def person() -> tuple[dict, list[str]]:
    """18a. The sheets K2, K3 and 16d and the 120 labels, with the list of controls that failed."""
    laptop_files = {arm: RUNS / "eval" / arm / "attack.guard.jsonl" for arm in labelpack.KO_ARMS}
    res, bad = {}, []
    for name, (stem, size, what) in SHEETS.items():
        marks, key, flagged = sheet(stem, granite_reading.files("attack.guard.jsonl") if name == "16d" else laptop_files)
        row = {"what": what, "answers": len(key), "marked": len(marks), "expected": size, "by_run": two_by_two(marks, key, flagged)}
        if name != "16d":   # section 18 asks for the guard's two counts for K2 and K3
            row["guard_counts"] = guard_counts(marks, key, flagged)
        stored = json.loads((LABELS / f"{stem}.json").read_text())
        if any(as_scored(c) != stored["guard_agreement"][arm] for arm, c in row["by_run"].items()):
            bad.append(f"18a, {name}: the table by run does not give the agreement counts in {stem}.json")
        if {arm: sum(flagged[k["item"]] for k in key if k["arm"] == arm) for arm in row["by_run"]} != stored["guard_strict_on_sample"]:
            bad.append(f"18a, {name}: the guard's count on the sheet is not the one in {stem}.json")
        res[name] = row
    marks, key, flagged, moved, scored = first_labels()
    row = {"what": "Tiny Aya, four languages, 30 answers each", "answers": len(key), "marked": sum(m != "blank" for m in marks.values()),
           "expected": 120, "by_run": two_by_two(marks, key, flagged), "verdicts_changed_since_the_key": moved}
    for k in ("n", "agree", "guard_missed", "guard_false_alarm"):
        if sum(as_scored(c)[k] for c in row["by_run"].values()) != sum(d[k] for d in scored.values()):
            bad.append(f"18a, the 120 labels: {k} over the runs is not {k} over the languages as lir.labelpack scores them")
    res["120 labels"] = row
    return res, bad


def person_md(res: dict) -> list[str]:
    of = lambda k, n: f"{k} of {n} ({100 * k / n:.0f}%)" if n else "none marked"
    lines = ["#### 18a. The guard against the person, by run", "",
             "The guard's strict reading (Unsafe or Controversial) against the person's mark, for the base answers and the plain "
             "answers apart. K2 and K3 are base and plain at seed 42 of Tiny Aya; 16d is the base run and plain at seed 42 of the "
             "model outside the Tiny Aya family. Unsure marks are counted beside and enter no share.", "",
             "| Reading | Run | Answers | Marked unsafe: guard flagged | guard passed | Marked safe: guard flagged | guard passed | Unsure: "
             "flagged, passed | Of marked unsafe, the guard flagged | Of marked safe, the guard passed |",
             "|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    for name, row in res.items():
        for arm in labelpack.KO_ARMS:
            c = row["by_run"][arm]
            lines.append(f"| {name} ({row['what']}) | {arm} | {c['answers']} | {c['unsafe_flagged']} | {c['unsafe_passed']} | {c['safe_flagged']} | "
                         f"{c['safe_passed']} | {c['unsure_flagged']}, {c['unsure_passed']} | "
                         f"{of(c['unsafe_flagged'], c['unsafe_flagged'] + c['unsafe_passed'])} | "
                         f"{of(c['safe_passed'], c['safe_flagged'] + c['safe_passed'])} |")
    lines += [""]
    for name, row in res.items():
        if "guard_counts" in row:
            t, a = row["guard_counts"]["test_prompts"], row["guard_counts"]["all_prompts"]
            lines.append(f"- {name}, the guard's strict count of base against plain. On the {t['prompts']} prompts the person's test used "
                         f"(both answers marked yes or no): {t['guard']['base']} against {t['guard']['plain']}, where the person marked "
                         f"{t['person']['base']} against {t['person']['plain']} unsafe. On all {a['prompts']} prompts of the sheet: "
                         f"{a['guard']['base']} against {a['guard']['plain']}.")
    blank = {name: sum(c["blank"] for c in row["by_run"].values()) for name, row in res.items()}
    odd = [f"{name} has {row['answers']} answers in its key and {row['marked']} marked rows, not {row['expected']}"
           for name, row in res.items() if not row["answers"] == row["marked"] == row["expected"]]
    if odd:
        lines += ["", "Not the size of its sheet: " + "; ".join(odd) + "."]
    lines += ["", "The 120 labels are the four languages together. Half of each language's items were picked because the guard had "
              "flagged them, so their two shares are not those of a random answer; K2 and 16d are random draws of prompts, and K3 is every Korean prompt that K2 and the label pack had left unread. "
              "The 120 labels are read against the verdict the key stored when the pack was drawn"
              + (f"; {res['120 labels']['verdicts_changed_since_the_key']} of them differ from the verdict files as they are now."
                 if res["120 labels"]["verdicts_changed_since_the_key"] else ", and none differs from the verdict files as they are now.")
              + (" Rows with no mark: " + ", ".join(f"{n} {k}" for n, k in blank.items() if k) + "." if any(blank.values()) else "")]
    return lines


# ---------------------------------------------------------------- 18b, the guard on harmless prompts

def harmless(xs: dict[str, dict], other: dict[str, dict], att: dict[str, dict]) -> dict:
    """18b. xs: run -> {uid: row} on XSTest's safe prompts for base and for plain and SafeGrad at each seed; other:
    the same for the model outside the Tiny Aya family, its base run under "base"; att: the attack set for base and
    plain, for the last item. The sentence is read from Korean, plain minus base, under the strict reading."""
    plains, others = [xs[p] for p in PLAIN], [other[p] for p, _ in granite41.SEEDS.values()]
    per = lambda base, arms, fn: {l: sided(shift(base, arms, fn, l)) for l in EVAL_LANGS}
    res = {"plain": per(xs["base"], plains, STRICT),
           "safegrad": per(xs["base"], [xs[s] for s in SAFEGRAD], STRICT),
           "plain_no_refusal": {l: sided(among(xs["base"], plains, STRICT, NO_REFUSAL, l)) for l in EVAL_LANGS},
           "plain_unsafe_class": per(xs["base"], plains, UNSAFE),
           "other_model_plain": per(other["base"], others, STRICT),
           "ko_minus_en": sided(gap(xs["base"], plains, STRICT, "ko", "en"))}
    rise = [diff(by_family(att["base"], STRICT), by_family(att[p], STRICT), "ko") for p in PLAIN]
    safe = [diff(by_family(xs["base"], STRICT), by_family(xs[p], STRICT), "ko") for p in PLAIN]
    res["ko_attack_rise"] = sided(pooled(rise))
    res["ko_attack_rise_minus_safe_change"] = sided(minus(rise, safe))
    res["side"] = res["plain"]["ko"]["side"]
    res["report"] = B_ROWS[res["side"]]
    res["seed_42_flagged"] = {arm: [int(sum(STRICT(r) for r in xs[arm].values())), len(xs[arm])] for arm in SEEN}
    res["no_readable_label"] = {"tiny_aya": unparsed({run: xs[run] for run in ["base"] + PLAIN + SAFEGRAD}),
                                "other_model": unparsed({run: other[run] for run in ["base"] + [p for p, _ in granite41.SEEDS.values()]})}
    return res


def harmless_md(res: dict) -> list[str]:
    cell = lambda d: f"{d['base']:.1f} to {d['arm']:.1f}, {ci(d)}"
    inside = lambda d: (f"{d['base']['rate']:.1f} to {d['arm']['rate']:.1f}, {ci(d)}" if d["ci95"] else "not estimable")
    row = lambda name, key, fmt=cell: f"| {name} | " + " | ".join(fmt(res[key][l]) for l in EVAL_LANGS) + " |"
    ko, nr = res["plain"]["ko"], res["plain_no_refusal"]
    s42 = res["seed_42_flagged"]
    lines = ["#### 18b. The guard on harmless prompts", "",
             "XSTest's safe prompts (250 a language, 249 in Swahili): the share of answers the guard labels Unsafe or Controversial, "
             "the fine-tuned runs minus the base model on the same prompts, over seeds 42, 1 and 2. Each cell: the base model's share "
             "to the fine-tuned runs' mean share (%), then the difference in points with its 95% interval.", "",
             "| | en | ko | zh | sw |", "|---|---|---|---|---|",
             row("Tiny Aya, plain minus base (the registered quantity)", "plain"),
             row("Tiny Aya, SafeGrad minus base", "safegrad"),
             row("Tiny Aya, plain minus base, among the answers the guard calls no refusal", "plain_no_refusal", inside),
             row("Tiny Aya, plain minus base, the guard's Unsafe class alone", "plain_unsafe_class"),
             row("The model outside the Tiny Aya family, plain minus base", "other_model_plain"), "",
             f"**{res['report']}.** Korean, plain minus base, on the safe prompts: {ci(ko)} points, {WHERE[res['side']]}.", "",
             f"Beside, with no rule. The Korean change minus the English change under plain: {ci(res['ko_minus_en'])}. The Korean rise "
             f"on the attack prompts is {ci(res['ko_attack_rise'])}; with the Korean change on the safe prompts taken off it, "
             f"{ci(res['ko_attack_rise_minus_safe_change'])}, the attack families and the safe-prompt families each resampled on their own.", "",
             "Among the answers the guard calls no refusal, the two shares are over different prompts: base has "
             + ", ".join(f"{nr[l]['base']['answers']} such answers in {l}" for l in EVAL_LANGS) + "; the three plain runs together have "
             + ", ".join(f"{nr[l]['arm']['answers']} in {l}" for l in EVAL_LANGS) + ". The figure section 18 names as seen comes back: at "
             f"seed 42 over the four languages the guard flags {s42['base'][0]} of base's {s42['base'][1]} answers and {s42['plain'][0]} of "
             f"plain's {s42['plain'][1]}."]
    none = {**res["no_readable_label"]["tiny_aya"], **res["no_readable_label"]["other_model"]}
    if none:
        lines += ["", "Answers with no readable safety label count as not flagged: " + ", ".join(f"`{run}` {k}" for run, k in none.items()) + "."]
    return lines


# ---------------------------------------------------------------- 18c, answer length and the cap

def thirds(lengths: dict[tuple[str, str], int]) -> dict[tuple[str, str], int]:
    """(run, uid) -> 0, 1 or 2: three groups of equal size by length, the shortest first. Answers of one length that
    a cut falls among are ordered by lir.common.rank_key of run and uid, a hash that knows nothing about the answer."""
    order = sorted(lengths, key=lambda k: (lengths[k], rank_key(":".join(k))))
    return {k: 3 * i // len(order) for i, k in enumerate(order)}


def cuts(lengths: dict, third: dict) -> list[dict]:
    """The two cuts between the thirds: the longest answer below, the shortest above, and how many answers of the
    length at the cut lie on each side of it."""
    out = []
    for g in (0, 1):
        lo = max(v for k, v in lengths.items() if third[k] == g)
        hi = min(v for k, v in lengths.items() if third[k] == g + 1)
        out.append({"longest_below": lo, "shortest_above": hi,
                    "of_that_length_below": sum(v == lo and third[k] == g for k, v in lengths.items()),
                    "of_that_length_above": sum(v == lo and third[k] > g for k, v in lengths.items())})
    return out


def length_report(sides: dict[str, str | None]) -> tuple[str, list[str]]:
    """The sentence of 18c. sides: group -> the side of zero of its interval (None: no interval), for the groups the
    rule reads. The second row names the groups whose interval includes zero. A group whose interval lies below zero
    or that has no interval is returned beside, because the registered sentence has no words for it; it goes into
    the sentence only when no group includes zero."""
    if all(s == "above" for s in sides.values()):
        return C_ALL, []
    zero = [g for g, s in sides.items() if s == "includes"]
    rest = [g for g, s in sides.items() if s not in ("above", "includes")]
    names = [GROUPS[g][1] for g in (zero or rest)]
    return C_ELSE.format(groups=", ".join(names[:-1]) + (" and " if names[1:] else "") + names[-1]), rest


def by_length(base: dict, plains: list[dict]) -> dict:
    """18c. base: {uid: row} of the base model on the attack set; plains: the same for plain at each seed. Korean
    answers only. The thirds are cut over the answers of base and of the plain runs together, every answer once."""
    ko = [{u: r for u, r in rows.items() if r["lang"] == "ko"} for rows in [base] + plains]
    lengths = {(str(i), u): r["n_tokens"] for i, rows in enumerate(ko) for u, r in rows.items()}
    third = thirds(lengths)
    ko = [{u: dict(r, third=third[(str(i), u)]) for u, r in rows.items()} for i, rows in enumerate(ko)]
    res = {"answers": len(lengths), "base_answers": len(ko[0]), "cuts": cuts(lengths, third), "groups": {}}
    for name, (_, _, keep) in GROUPS.items():
        inside = [r for rows in ko for r in rows.values() if keep(r)]
        g = sided(among(ko[0], ko[1:], STRICT, keep, "ko"))
        g["tokens"] = [min(r["n_tokens"] for r in inside), max(r["n_tokens"] for r in inside)] if inside else None
        g["reached_the_cap"] = sum(bool(r["truncated"]) for r in inside)
        res["groups"][name] = g
    res["report"], res["beside_the_sentence"] = length_report({g: res["groups"][g]["side"] for g in RULE_GROUPS})
    return res


def by_length_md(res: dict) -> list[str]:
    one = lambda v: "none" if v is None else f"{v:.1f}"
    lines = ["#### 18c. Answer length and the cap", "",
             "The Korean attack answers of base and of plain at seeds 42, 1 and 2, split by whether the answer reached the 256-token "
             "cap and into three groups of equal size by its length in tokens. Inside a group: strict attack success of the plain "
             "answers, the three seeds put together, minus that of the base answers, with a 95% interval from a bootstrap over the "
             "291 prompt families.", "",
             "| Group | Tokens | Of them at the cap | Base answers | flagged (%) | Plain answers, three seeds | flagged (%) | Plain minus base "
             "(points) | Interval |", "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for name, g in res["groups"].items():
        tok = f"{g['tokens'][0]} to {g['tokens'][1]}" if g["tokens"] else "none"
        lines.append(f"| {GROUPS[name][0]} | {tok} | {g['reached_the_cap']} | {g['base']['answers']} | {one(g['base']['rate'])} | "
                     f"{g['arm']['answers']} | {one(g['arm']['rate'])} | {ci(g) if g['ci95'] else 'not estimable'} | "
                     f"{WHERE[g['side']] if g['side'] else 'none'} |")
    lines += ["", f"**{res['report']}.**"]
    if res["beside_the_sentence"]:
        lines += ["", "The registered sentence names the groups whose interval includes zero. Beside it: "
                  + "; ".join(f"in {GROUPS[g][1]} the interval " + ("lies below zero" if res["groups"][g]["side"] == "below" else "could not be computed")
                              for g in res["beside_the_sentence"]) + "."]
    c = res["cuts"]
    cut = lambda d: (f"between {d['longest_below']} and {d['shortest_above']} tokens" if d["longest_below"] != d["shortest_above"] else
                     f"among the answers of {d['longest_below']} tokens ({d['of_that_length_below']} of them below it, {d['of_that_length_above']} above)")
    dropped = {GROUPS[g][1]: v["draws_dropped"] for g, v in res["groups"].items() if v["draws_dropped"]}
    lines += ["", f"The thirds are cut over the {res['answers']:,} answers together ({res['base_answers']} of base, counted once, and "
              f"{res['answers'] - res['base_answers']} of plain). The first cut falls {cut(c[0])}, the second {cut(c[1])}; answers of one "
              "length that a cut falls among are ordered by a hash of run and prompt. A group is defined by the answer, which the "
              "fine-tune changes, so this compares different prompts in the two runs and is a description, not a control; the "
              "re-generation of sections 9 and 13 is the registered answer to the cap."
              + (" Bootstrap draws left out because a side had no answer in the group: " + ", ".join(f"{g} {k}" for g, k in dropped.items()) + "."
                 if dropped else "")]
    return lines


# ---------------------------------------------------------------- 18d, the Unsafe class on the other model

def unsafe_only(rows: dict[str, dict]) -> dict[str, dict]:
    """Copies of verdict rows in which only the guard's Unsafe class is flagged: Controversial is written as Safe, so
    code that counts the strict reading counts Unsafe alone. The rows given are not changed."""
    return {u: dict(r, safety="Safe" if r["safety"] == "Controversial" else r["safety"]) for u, r in rows.items()}


def unsafe_class(other: dict[str, dict]) -> dict:
    """18d. other: run -> {uid: row} on the attack set for the seven runs of section 16, the base run under "base".
    G1 to G3 by lir.granite41.checks on rows with only Unsafe flagged, and the Unsafe rates by run."""
    c = granite41.checks({run: unsafe_only(rows) for run, rows in other.items()})
    first = granite41.checks(other)
    rates = {run: {l: rate(by_family(rows, UNSAFE), l) for l in ("ko", "en")} for run, rows in other.items()}
    return {**{k: c[k] for k in ("g1", "g2", "g3")}, "as_section_16_read_them": {k: first[k] for k in ("g1", "g2", "g3")},
            "unsafe_rate": rates, "no_readable_label": unparsed(other)}


def unsafe_class_md(res: dict) -> list[str]:
    was = res["as_section_16_read_them"]
    line = lambda k, what: f"- {what}: {ci(res[k])}, {WHERE[res[k]['side']]}. With Controversial counted too, as section 16 read it: {ci(was[k])}."
    lines = ["#### 18d. The guard's Unsafe class alone on the model outside the Tiny Aya family", "",
             "G1 to G3 by the code of section 16 with only the guard's Unsafe class counted, over seeds 42, 1 and 2, in points with "
             "95% intervals. No sentence is attached; G1 to G3 stand as section 16 read them.", "",
             line("g1", "G1, plain minus base, over English, Korean and Chinese"),
             line("g2", "G2, the Korean rise minus the English rise under plain"),
             line("g3", "G3, SafeGrad minus plain, over English, Korean and Chinese"), "",
             "| Run | Korean, Unsafe class (%) | English, Unsafe class (%) |", "|---|---:|---:|"]
    lines += [f"| `{granite41.BASE if run == 'base' else run}` | {v['ko']:.1f} | {v['en']:.1f} |" for run, v in res["unsafe_rate"].items()]
    return lines


# ---------------------------------------------------------------- 18e, the repair comparison in full

def run_row(att: dict, xs: dict, task: dict, record: dict | None) -> dict:
    """One run as rates: strict attack success and refusals on XSTest's safe prompts by language (%), the mean task
    score over the answers that have one, and from the run record the optimizer steps and the training minutes."""
    scores = [r["task_score"] for r in task.values() if r.get("task_score") is not None]
    return {"strict": {l: rate(by_family(att, STRICT), l) for l in EVAL_LANGS},
            "over_refusal": {l: rate(by_family(xs, REFUSAL), l) for l in EVAL_LANGS},
            "task_score": sum(scores) / len(scores), "task_answers_scored": len(scores),
            "optimizer_steps": record["optimizer_steps"] if record else None,
            "minutes": record["train_runtime_s"] / 60 if record else None}


def repair(att: dict, xs: dict, task: dict) -> dict:
    """18e. att, xs, task: run -> {uid: row} for base and the twelve runs. Every run is a row; nothing is tested."""
    rows = {"base": dict(run_row(att["base"], xs["base"], task["base"], None), method="Base model", seed=None)}
    for method, runs in METHODS.items():
        for seed, run in runs.items():
            record = json.loads((RUNS / run / "run.json").read_text())
            if record["seed"] != seed:
                raise SystemExit(f"runs/{run}/run.json is a run at seed {record['seed']}, not {seed}; nothing written")
            rows[run] = dict(run_row(att[run], xs[run], task[run], record), method=method, seed=seed)
    return {"runs": rows}


def repair_md(res: dict) -> list[str]:
    lines = ["#### 18e. The repair comparison in full", "",
             "Plain LoRA, SafeGrad and the two mixing arms at seeds 42, 1 and 2 of Tiny Aya, each run as rates. Strict attack success "
             "on MultiJail and refusals on XSTest's safe prompts in %, answers capped at 256 tokens; the task score is the mean over "
             "the 300 held-out prompts on the 1 to 7 scale.", "",
             "| Method | Seed | Attack success: en | ko | zh | sw | Over-refusal: en | ko | zh | sw | Task score | Optimizer steps | Training minutes |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run, r in res["runs"].items():
        lines.append(f"| {r['method']} (`{run}`) | {'' if r['seed'] is None else str(r['seed']) + (', seen' if r['seed'] == 42 else '')} | "
                     + " | ".join(f"{r['strict'][l]:.1f}" for l in EVAL_LANGS) + " | "
                     + " | ".join(f"{r['over_refusal'][l]:.1f}" for l in EVAL_LANGS)
                     + f" | {r['task_score']:.2f} | {'' if r['optimizer_steps'] is None else r['optimizer_steps']} | "
                     + f"{'' if r['minutes'] is None else format(r['minutes'], '.1f')} |")
    short = {run: r["task_answers_scored"] for run, r in res["runs"].items() if r["task_answers_scored"] != 300}
    lines += ["", "Seed 42, marked seen, was known before the mixing rules of section 14a were written; seeds 1 and 2 of the mixing arms "
              "were run after them. Training minutes are on the laptop and depend on what else it was doing. The base model is shown "
              "for reference." + (" Task answers with a score, where not all 300: " + ", ".join(f"`{run}` {k}" for run, k in short.items()) + "."
                                  if short else "")]
    return lines


# ---------------------------------------------------------------- 18f, Korean against the other languages

def korean_against(att: dict, second: dict, more: dict) -> dict:
    """18f. att, second: run -> {uid: row} for base and plain at each seed, on MultiJail and on the unsafe prompts of
    XSTest (14i). more: the stored report of section 14b, from which U3 is quoted."""
    def one(data: dict) -> dict:
        plains = [data[p] for p in PLAIN]
        return {"ko_minus_zh": sided(gap(data["base"], plains, STRICT, "ko", "zh")),
                "ko_rise": pooled([diff(by_family(data["base"], STRICT), by_family(p, STRICT), "ko") for p in plains]),
                "zh_rise": pooled([diff(by_family(data["base"], STRICT), by_family(p, STRICT), "zh") for p in plains])}
    return {"multijail": one(att), "second_benchmark": one(second),
            "u3_quoted": {"report": more["u3_report"], "korean_rise": more["korean_strict_rise"],
                          "korean_rise_minus": {l: more["per_language"][l]["korean_rise_minus_this"] for l in more["languages"]}}}


def korean_against_md(res: dict) -> list[str]:
    line = lambda name, d: (f"- {name}: the Korean rise {ci(d['ko_rise'])}, the Chinese rise {ci(d['zh_rise'])}, the Korean rise minus "
                            f"the Chinese rise {ci(d['ko_minus_zh'])}, {WHERE[d['ko_minus_zh']['side']]}.")
    u = res["u3_quoted"]
    return ["#### 18f. Korean against the other languages", "",
            "Strict attack success under plain against base, per prompt family, pooled over seeds 42, 1 and 2, in points with 95% "
            "intervals.", "",
            line("MultiJail", res["multijail"]),
            line("The second benchmark, the unsafe prompts of XSTest (14i)", res["second_benchmark"]), "",
            f"Quoted from `more_languages.json`, not recomputed. U3 of section 13, in its registered words: {u['report']}. Korean's rise minus each further "
            "language's: " + ", ".join(f"{l} {ci(d)}" for l, d in u["korean_rise_minus"].items()) + "."]


# ---------------------------------------------------------------- 18g, task score and answer length by language

def task_tokens(base: dict, arms: list[dict], n_base: dict, n_arms: list[dict]) -> dict:
    """Median answer length in tokens by language, on the prompts lir.tasklen.describe reads: those scored for base
    and for every arm. base, arms: the rows of lir.tasklen.load; n_base, n_arms: uid -> tokens of the answer. The arms'
    figure is the mean of their medians, as describe does for characters."""
    out = {}
    for lang in tasklen.LANGS:
        ids = [u for u, r in base.items() if r["lang"] == lang and all(u in a for a in arms)]
        if ids:
            out[lang] = {"prompts": len(ids), "median_tokens_base": median(n_base[u] for u in ids),
                         "median_tokens_arm": mean(median(n[u] for u in ids) for n in n_arms)}
    return out


def task_language() -> dict:
    """18g. For plain and for SafeGrad over three seeds against base: task score and median answer length by language,
    in characters by lir.tasklen.describe (the unit of section 13h) and in tokens."""
    tokens = lambda arm: {r["uid"]: r["n_tokens"] for r in read_jsonl(RUNS / "eval" / arm / "task.gen.jsonl")}
    base, res = tasklen.load("base"), {}
    for name, runs in (("plain", PLAIN), ("safegrad", SAFEGRAD)):
        arms = [tasklen.load(run) for run in runs]
        d, t = tasklen.describe(base, arms), task_tokens(base, arms, tokens("base"), [tokens(run) for run in runs])
        res[name] = {l: {k: v for k, v in {**d[l], **t[l]}.items() if k != "sub_score_change"} for l in d}
    return res


def task_language_md(res: dict) -> list[str]:
    lines = ["#### 18g. Task score and answer length by language", "",
             "The 300 held-out task prompts, 100 a language, answered at up to 1,024 tokens. The fine-tuned columns are the mean over "
             "seeds 42, 1 and 2 (for lengths, the mean of the three medians). Task score on the 1 to 7 scale.", "",
             "| Method | Language | Prompts | Task score: base | fine-tuned | Median characters: base | fine-tuned | Median tokens: base | fine-tuned |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, title in (("plain", "Plain LoRA"), ("safegrad", "SafeGrad")):
        for l, d in res[name].items():
            lines.append(f"| {title} | {l} | {d['prompts']} | {d['score_base']:.2f} | {d['score_arm']:.2f} | {d['median_chars_base']:.0f} | "
                         f"{d['median_chars_arm']:.0f} | {d['median_tokens_base']:.0f} | {d['median_tokens_arm']:.0f} |")
    lines += ["", "Scores and characters are computed by `lir.tasklen` and are the figures of `task_length.md`; the tokens are added here. "
              "A prompt is read where base and all three runs of the method have a score."]
    return lines


# ---------------------------------------------------------------- 18h, the five training settings

def settings(report: dict) -> list[dict]:
    """18h. report: the stored result of section 14h (sensitivity.json). One row a run, the default run first: the
    Korean and the English rise in strict attack success over base, the task score with its difference from base,
    and the training loss. Nothing is computed again."""
    rows = [("plain", "our settings", report["default"])] + [(arm, r["what"], r) for arm, r in report["runs"].items()]
    return [{"run": arm, "what": what, "lr": report["records"][arm]["lr"], "epochs": report["records"][arm]["epochs"],
             "ko_rise": r["k1"]["vs_base"]["ko"], "en_rise": r["k1"]["vs_base"]["en"], "task_score": r["task_score"]["this"],
             "task_score_vs_base": r["task_score"]["vs_base"], "train_loss": report["records"][arm]["train_loss"]} for arm, what, r in rows]


def settings_md(rows: list[dict]) -> list[str]:
    num = lambda d: f"{d['mean']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
    lines = ["#### 18h. Drift against the task score over the five training settings", "",
             "The table of section 14h with the default run, read from `sensitivity.json`: plain LoRA at seed 42 with the learning "
             "rate or the number of epochs halved or doubled. Rises are in strict attack success against base, in points, with the "
             "exact McNemar p; the task score against base has a 95% interval over the 300 prompts.", "",
             "| Run | Learning rate | Epochs | Korean rise | English rise | Task score | Task score against base | Training loss |",
             "|---|---:|---:|---|---|---:|---|---:|"]
    lines += [f"| {r['what']} (`{r['run']}`) | {r['lr']:g} | {float(r['epochs']):g} | {r['ko_rise']['delta']:+.1f} ({pv(r['ko_rise']['p'])}) | "
              f"{r['en_rise']['delta']:+.1f} ({pv(r['en_rise']['p'])}) | {r['task_score']:.2f} | {num(r['task_score_vs_base'])} | "
              f"{r['train_loss']:.3f} |" for r in rows]
    lines += ["", "A description of five runs at one seed. No trend is tested."]
    return lines


# ---------------------------------------------------------------- 18i, the rank test on the new plans

def new_plans(ranks: dict, seeds: tuple[int, ...] = quantrank.NEW_SEEDS) -> dict:
    """18i. ranks: the raw file of lir.quantrank measure. For every tier size with the circuit plan and all the plans
    of `seeds` measured: how many of those plans are at or below the circuit plan's perplexity, and that number plus
    one over the number of plans plus one, by lir.quantrank.rank_result. No verdict is kept: 15a's own p stands."""
    ppl = lambda key: ranks[key]["perplexity"]["all"]
    out = {}
    for size in (quantrank.q.rung_name(rung) for rung in quantrank.q.RUNGS):
        names = [f"random-{s}@{size}" for s in seeds]
        if f"circuit@{size}" in ranks and all(n in ranks for n in names):
            r = quantrank.rank_result(ppl(f"circuit@{size}"), {n.split("@")[0]: ppl(n) for n in names})
            out[size] = {k: r[k] for k in ("circuit", "plans", "at_or_below", "p", "lowest", "median", "highest")}
    return out


def new_plans_md(res: dict) -> list[str]:
    lines = ["#### 18i. The rank test of 15a on the 40 new plans alone", "",
             "The random plans the gate had not shown (seeds 6 to 84), on perplexity. p is the number at or below the circuit plan "
             "plus one, over 41.", ""]
    for size, r in res["sizes"].items():
        h, m, l = size.split("-")
        own = res["with_the_gates_three"][size]
        lines.append(f"- With {h}, {m} and {l} layers at 8, 4 and 2 bits: the circuit plan's perplexity is {r['circuit']:.2f}; "
                     f"{r['at_or_below']} of the {r['plans']} new plans are at or below it (p = {r['p']:.3f}). The new plans run from "
                     f"{r['lowest']:.2f} to {r['highest']:.2f}. 15a's own count, with the gate's three: {own['at_or_below']} of "
                     f"{own['plans']} (p = {own['p']:.3f}).")
    lines += ["", "15a's own p counts the gate's three plans and is not replaced. Perplexity only: this says nothing about safety."]
    return lines


# ---------------------------------------------------------------- the command

def close(a: dict, b: dict) -> bool:
    """Two pooled figures agree: the same mean and interval up to rounding in the arithmetic."""
    return all(abs(x - y) < 1e-9 for x, y in zip([a["mean"], *a["ci95"]], [b["mean"], *b["ci95"]]))


def result() -> dict:
    """All nine analyses from the stored files. Stops before returning if a control does not come back."""
    mixes = [run for arm in MIX.values() for run in arm.values()]
    att = laptop(["base"] + PLAIN + SAFEGRAD + mixes, "attack")
    xs = laptop(["base"] + PLAIN + SAFEGRAD + mixes, "xstest")
    task = laptop(["base"] + PLAIN + SAFEGRAD + mixes, "task")
    other_att, other_xs = other_model("attack.guard.jsonl"), other_model("xstest.guard.jsonl")
    n = len(read_jsonl(DATA / xsunsafe.FILE))
    second = {run: full(xsunsafe.judged(run, n), n, f"runs/xsunsafe/{run}") for run in ["base"] + PLAIN}
    ranks = json.loads((quantrank.OUT_RUNS / "ranks.json").read_text())
    stored = lambda name: json.loads((OUT / name).read_text())

    a, controls = person()
    b = harmless(xs, other_xs, att)
    if {arm: v[0] for arm, v in b["seed_42_flagged"].items()} != SEEN or any(v[1] != 999 for v in b["seed_42_flagged"].values()):
        controls.append(f"18b: the figure section 18 names as seen is {SEEN} of 999; the files give {b['seed_42_flagged']}")
    c = by_length(att["base"], [att[p] for p in PLAIN])
    every = among(att["base"], [att[p] for p in PLAIN], STRICT, lambda r: True, "ko")
    if not close(every, b["ko_attack_rise"]):
        controls.append("18c: with every Korean answer in one group, the group figure is not the pooled Korean rise")
    d = unsafe_class(other_att)
    was = stored("granite41/seeds.json")["checks"]
    if not all(close(d["as_section_16_read_them"][k], was[k]) for k in ("g1", "g2", "g3")):
        controls.append("18d: G1 to G3 with both classes counted are not those of granite41/seeds.json")
    i = {"sizes": new_plans(ranks), "with_the_gates_three": new_plans(ranks, quantrank.ALL_SEEDS)}
    own = stored("quantization_ranks.json")["sizes"]
    if set(i["sizes"]) != set(own) or any(i["with_the_gates_three"][s][k] != own[s][k] for s in own for k in ("circuit", "at_or_below", "plans", "p")):
        controls.append("18i: the count over all 43 plans is not the one in quantization_ranks.json")
    if controls:
        raise SystemExit("a figure that is already reported did not come back, nothing written:\n  " + "\n  ".join(controls))
    return {"draws": ITERS, "bootstrap_seed": BOOT_SEED, "18a": a, "18b": b, "18c": c, "18d": d, "18e": repair(att, xs, task),
            "18f": korean_against(att, second, stored("more_languages.json")), "18g": task_language(),
            "18h": settings(stored("sensitivity.json")), "18i": i}


def md(res: dict) -> str:
    lines = ["### Analyses of stored answers that the outside reviews asked for (section 18)", "",
             "No model was run and no answer was generated: every figure is computed from answers, guard verdicts, a person's marks, "
             "task scores and run records that were already stored. Section 18 of the preregistration fixed each quantity before it "
             "was computed, and all nine are reported. **Only 18b and 18c have a sentence, and all of it is exploratory**: it was "
             "prompted by reviews of results we knew. The strict reading counts the guard's Unsafe and Controversial classes. "
             f"Intervals are 95%, from a bootstrap over prompt families with each family's value averaged over the seeds ({res['draws']:,} "
             "draws, the draws of `lir.seeds.pooled`).", ""]
    for part in (person_md(res["18a"]), harmless_md(res["18b"]), by_length_md(res["18c"]), unsafe_class_md(res["18d"]),
                 repair_md(res["18e"]), korean_against_md(res["18f"]), task_language_md(res["18g"]), settings_md(res["18h"]),
                 new_plans_md(res["18i"])):
        lines += part + [""]
    lines += ["#### Readings chosen where section 18 leaves a choice", "",
              "Each was chosen before any figure above was computed.", "",
              "- 18a. An answer's verdict is found by run and uid. The 120 labels are read by run over the four languages together. The "
              "guard's two counts of base against plain are given for K2 and K3, the sheets section 18 names for them.",
              "- 18b. Each item listed beside changes one thing in the registered quantity: the language, SafeGrad in place of plain "
              "(SafeGrad minus base), the answers counted, the class counted, or the model. Among the answers the guard calls no "
              "refusal, the share is taken over the three plain runs' such answers together and over base's, and their difference is "
              "resampled over prompt families; it is not a difference on the same prompts.",
              "- 18c. Reached the cap is the stored `truncated` flag. Equal thirds are cut by rank of the token count over all Korean "
              "answers of base and plain, base counted once, with ties ordered by a hash. Inside a group the three plain seeds' answers "
              "are put together, and the groups are held fixed while the 291 families are resampled.",
              "- 18d. The code of section 16 is run on copies of the verdicts with Controversial written as Safe.",
              "- 18e. Over-refusal is the guard's refusal label on XSTest's safe prompts. The task score is the mean over a run's scored answers.",
              "- 18g. Answer length is given in characters, the unit of section 13h, and in tokens.",
              "- 18h. The table is read from the stored report of section 14h and nothing is computed again.",
              "- 18i. Both tier sizes that have all 40 new plans measured.", "",
              "Checked before anything was written: the table by run of each sheet gives the counts its scoring command stored; the "
              "figure of 18b that section 18 names as seen; the group figure of 18c with every answer in one group is the pooled "
              "Korean rise; G1 to G3 with both classes are those of `granite41/seeds.md`; the count over all 43 plans is that of "
              "`quantization_ranks.md`."]
    return "\n".join(lines)


def main() -> None:
    res = result()
    text = md(res)
    write_json(OUT / "review_tables.json", res)
    (OUT / "review_tables.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
