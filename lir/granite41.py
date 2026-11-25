"""The main experiment on a model outside the Tiny Aya family: IBM Granite 4.1 3B, on a Hugging Face GPU.

  python -m lir.gpu fetch      # the files of every job, from the staging dataset -> runs/gpu
  python -m lir.granite41      # the three-seed checks -> results/lir/stage2/granite41/seeds.{json,md}

The base model, plain LoRA and SafeGrad at seeds 42, 1 and 2, trained and evaluated by the granite41 jobs of
lir.hfjob. The checks are the three of lir.seeds, computed by its analyse() and so by the same rules: the Korean rise
under plain, SafeGrad below plain, and the Swahili refusal rate, each against this model's own base run. Every
comparison is between two runs on the GPU. Nothing here loads a model, and nothing here writes the tables of
sections 12 or 14f. This module is not lir.granite, which is the Granite Guardian second judge.
"""
from __future__ import annotations

import argparse
import json

from lir import gpu, hfjob
from lir.common import DATA, EVAL_LANGS, MODELS, ROOT, read_jsonl, sha256_file, write_json
from lir.gpu import gpu_rows, newest, versus
from lir.repeat import ci, pv, star
from lir.seeds import REFUSAL, STRICT, analyse, by_family, by_language, change, fmt

KEY = hfjob.OTHER
MODEL, REVISION = MODELS[KEY]
OUT = ROOT / "results" / "lir" / "stage2" / KEY
BASE = f"{KEY}-base"   # for every run here the job and its arm have one name
SEEDS = {seed: (f"{KEY}-plain-s{seed}", f"{KEY}-safegrad-s{seed}") for seed in hfjob.OTHER_SEEDS}
ARMS = [arm for pair in SEEDS.values() for arm in pair]
REFUSALS = hfjob.JOBS[BASE][0]["refusals"]
TRAIN = "bactrian_train.jsonl"
SETTINGS = {"epochs": 3, "lr": 2e-4, "effective_batch": 32, "micro_batch": 4}   # the defaults of lir.train, as on Tiny Aya
NAMES = ("H1", "H2", "H3")   # how the report names the three checks; the registration may give them names of its own


def run_problems(runs: dict[str, dict | None], refusals_sha: str | None, train_sha: str) -> list[str]:
    """Run records that do not show the runs this module reads. runs: arm -> its run.json, None if there is none."""
    bad = []
    for seed, pair in SEEDS.items():
        for method, arm in zip(("plain", "safegrad"), pair):
            r = runs.get(arm)
            if r is None:
                bad.append(f"{arm}: no run record")
                continue
            got = {"method": r.get("arm"), "model": [r.get("model"), r.get("model_revision")], "seed": r.get("seed"),
                   "training file": r.get("train_sha256"), "refusal examples": (r.get("alignment") or {}).get("sha256"),
                   **{k: r.get(k) for k in SETTINGS}}
            want = {"method": method, "model": [MODEL, REVISION], "seed": seed, "training file": train_sha,
                    "refusal examples": refusals_sha if method == "safegrad" else None, **SETTINGS}
            bad += [f"{arm}: {k} is {got[k]}, expected {want[k]}" for k in want if got[k] != want[k]]
            if str(r.get("commit") or "").endswith("-dirty"):
                bad.append(f"{arm}: trained from a commit with uncommitted changes ({r['commit']})")
    return bad


def result(data: dict[str, dict], xstest: dict[str, dict] | None = None) -> dict:
    """The three checks by the rules of lir.seeds. data: arm -> {uid: guard row} for the attack set, the base model
    under "base", holding only arms whose answers are all judged. xstest: the same for XSTest, shown with no rule."""
    res = analyse(data, SEEDS)
    base = by_family(data["base"], STRICT)
    for seed, row in res["seeds"].items():   # beside the checks, no rule
        row["safegrad_vs_base_ko"] = change(base, by_family(data[SEEDS[seed][1]], STRICT), "ko")
    if res["seeds"]:
        res["by_language"] = by_language(data, SEEDS)
    if xstest and "base" in xstest:
        res["over_refusal"] = {arm: versus(xstest["base"], rows, REFUSAL) for arm, rows in xstest.items() if arm != "base"}
    return res


def report_md(res: dict) -> str:
    done = [str(s) for s in res["seeds"]]
    lines = [f"### A model outside the Tiny Aya family: {MODEL}, seeds 42, 1 and 2", "",
             f"`{MODEL}` at revision `{REVISION[:8]}`: the base model, plain LoRA and SafeGrad at three seeds, trained with "
             "the settings of the Tiny Aya runs and evaluated on one NVIDIA L4. Strict attack success counts the guard's "
             "Unsafe and Controversial classes, on 291 prompts per language. The three checks follow the rules of H1 to H3 "
             "in `lir.seeds`, each against this model's own base run. Per-seed tests are exact McNemar, two-sided; a pooled "
             "figure averages each prompt family over the seeds, then bootstraps the 291 families (95% interval). Every "
             "comparison is between two runs on the GPU.", ""]
    if res.get("jobs"):
        lines += ["| Job | Upload | Commit | GPU | Finished (UTC) |", "|---|---|---|---|---|"]
        lines += [f"| {job} | `{j['folder']}` | `{(j['commit'] or '')[:7]}` | {'; '.join(j['gpu'])} | {j['finished_utc']} |" for job, j in res["jobs"].items()]
        lines += ["", ("The library versions are the same in every job." if not res.get("libraries_differ") else
                       "The library versions are not the same in every job: " + "; ".join(res["libraries_differ"]) + "."), ""]
    if not res["complete"]:
        lines += [f"**The checks are not read: they need the base run and both arms at all three seeds.** Seeds with both arms: "
                  f"{', '.join(done) or 'none'}. Their figures are shown below and every verdict stays pending.", ""]
    lines += [fmt(res, NAMES)]
    if res["seeds"]:
        lines += ["", "Beside the checks, with no rule. SafeGrad against the base model on Korean: " + "; ".join(
            f"seed {s} {r['safegrad_vs_base_ko']['delta']:+.1f} ({pv(r['safegrad_vs_base_ko']['p'])})" for s, r in res["seeds"].items()) + "."]
    if "by_language" in res:
        b = res["by_language"]["strict"]
        lines += ["", f"Change in strict attack success against the base model, pooled over the seeds read ({', '.join(done)}), with no rule:", "",
                  "| Arm | en | ko | zh | sw |", "|---|---|---|---|---|",
                  "| Base rate (%) | " + " | ".join(f"{b['base'][l]:.1f}" for l in EVAL_LANGS) + " |"]
        lines += [f"| {name} | " + " | ".join(ci(b[key][l]) for l in EVAL_LANGS) + " |" for key, name in (("plain", "Plain LoRA"), ("safegrad", "SafeGrad"))]
    if res.get("over_refusal"):
        cell = lambda v: f"{v['a']:.1f} to {v['b']:.1f} ({v['delta']:+.1f}{star(v)})"
        lines += ["", "Refusals on the safe prompts of XSTest (%), the base model to each run, with no rule:", "",
                  "| Run | en | ko | zh | sw |", "|---|---|---|---|---|"]
        lines += [f"| `{arm}` | " + " | ".join(cell(v[l]) for l in EVAL_LANGS) + " |" for arm, v in res["over_refusal"].items()]
    if res.get("runs"):
        lines += ["", "Training, from the run records and the job records:", "",
                  "| Run | Seed | Rows trained | Rows over the length limit | Optimizer steps | Minutes | Peak GPU memory (MiB) | Refusal examples |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        lines += [f"| `{arm}` | {r['seed']} | {r['train_rows']:,} | {r['dropped_over_max_len']} | {r['optimizer_steps']} | {r['minutes']:.1f} | "
                  f"{r['peak_gpu_mib'] or ''} | {r['refusal_examples'] or ''} |" for arm, r in res["runs"].items()]
    if res.get("refusals"):
        f = res["refusals"]
        lines += ["", f"SafeGrad's refusal set is this model's own: {f['rows']} safe refusals of the base model to the 72 held-out prompts "
                  f"(`{REFUSALS}`, sha256 `{f['sha256'][:8]}`, " + ", ".join(f"{l} {n}" for l, n in sorted(f["per_lang"].items())) + ")."]
    if res.get("unjudged"):
        lines += ["", "Left out, because not every answer has a verdict: " + ", ".join(res["unjudged"]) + "."]
    lines += ["", "Not measured: the task score (its judge does not fit an L4), any second judge, and answers past the 256-token "
              "cap. The verdicts are the guard's; no person has read these answers."]
    return "\n".join(lines)


def read() -> dict:
    """Everything of these runs that is in runs/gpu, by the newest upload of each job."""
    n = len(read_jsonl(DATA / "multijail_attack.jsonl"))
    data, xstest, unjudged = {}, {}, []
    for arm in [BASE] + ARMS:
        key = "base" if arm == BASE else arm
        rows = gpu_rows(arm, arm)
        if rows is not None and len(rows) == n:
            data[key] = rows
            x = gpu_rows(arm, arm, "xstest.guard.jsonl")
            if x is not None:
                xstest[key] = x
        elif rows is not None:
            unjudged.append(f"{arm} ({len(rows)} of {n})")
    if "base" not in data:
        return {"seeds": {}, "complete": False, "unjudged": unjudged, "jobs": {}}
    counts = {arm: {"n": len(rows), "unparsed": sum(r["safety"] is None for r in rows.values())} for arm, rows in data.items()}
    for rows in data.values():
        for r in rows.values():
            r["unparsed"] = r["safety"] is None
    res = result(data, xstest)
    res["rows"], res["unjudged"] = counts, unjudged
    records, res["jobs"] = {}, {}
    for job in [BASE] + ARMS:
        folder = newest(job)
        if folder:
            records[job] = json.loads((folder / "job.json").read_text())
            hw = records[job]["hardware"]
            res["jobs"][job] = {"folder": folder.name, "commit": records[job]["commit"], "gpu": hw["gpus"], "flavor": hw["flavor"],
                                "libraries": hw["libraries"], "finished_utc": records[job].get("finished_utc")}
    libs = {job: j["libraries"] for job, j in res["jobs"].items()}
    res["libraries_differ"] = [f"{name} " + ", ".join(f"{job} {v.get(name)}" for job, v in libs.items())
                               for name in sorted({x for v in libs.values() for x in v}) if len({v.get(name) for v in libs.values()}) > 1]
    refusals = newest(BASE) / REFUSALS
    if refusals.exists():
        rows = read_jsonl(refusals)
        res["refusals"] = {"rows": len(rows), "sha256": sha256_file(refusals), "per_lang": {l: sum(r["lang"] == l for r in rows) for l in {r["lang"] for r in rows}}}
    runs = {}
    for arm in ARMS:
        p = newest(arm) / arm / "run" / "run.json" if newest(arm) else None
        runs[arm] = json.loads(p.read_text()) if p and p.exists() else None
    used = [arm for seed in res["seeds"] for arm in SEEDS[seed]]
    res["run_problems"] = [line for line in run_problems(runs, (res.get("refusals") or {}).get("sha256"), sha256_file(DATA / TRAIN))
                           if line.split(":")[0] in used]
    peak = {s["arm"]: s.get("peak_gpu_mib") for rec in records.values() for s in rec["steps"] if s.get("what") == "train"}
    res["runs"] = {arm: {"seed": r["seed"], "train_rows": r["train_rows"], "dropped_over_max_len": r["dropped_over_max_len"],
                         "optimizer_steps": r["optimizer_steps"], "minutes": r["train_runtime_s"] / 60, "peak_gpu_mib": peak.get(arm),
                         "refusal_examples": (r.get("alignment") or {}).get("rows"), "commit": r.get("commit")}
                   for arm, r in runs.items() if r is not None}
    if (gpu.GPU / "revision.txt").exists():
        res["staging_revision"] = (gpu.GPU / "revision.txt").read_text().strip()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="write the table although a run is missing; every verdict then stays pending")
    a = ap.parse_args()
    res = read()
    if res.get("run_problems"):
        raise SystemExit("run records do not match these runs, nothing written:\n  " + "\n  ".join(res["run_problems"]))
    if not res["complete"] and not a.partial:
        print(f"pending, nothing written: seeds with both arms: {sorted(res['seeds']) or 'none'}; jobs found: {sorted(res['jobs'])}"
              + (f"; not fully judged: {', '.join(res['unjudged'])}" if res.get("unjudged") else ""))
        return
    if not res["seeds"]:
        print("nothing to write: no seed has both arms and the base run")
        return
    text = report_md(res)
    write_json(OUT / "seeds.json", res)
    (OUT / "seeds.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
