"""Preregistration section 15b: Global-MMLU-Lite on the arms of section 15, on a Hugging Face GPU.

  python -m lir.mmlu launch --dry-run   # print the job, submit nothing
  python -m lir.mmlu launch             # submit it; needs a login with Jobs rights on the namespace
  python -m lir.mmlu job                # what the job runs in its container: every arm, then the upload
  python -m lir.mmlu report             # after `python -m lir.gpu fetch`: results/lir/stage2/quantization_mmlu.{json,md}

Section 15 measures capability by the task score, which a judge model gives. This adds a measure with no judge: the
400 multiple-choice questions of Global-MMLU-Lite in each of English, Korean, Chinese and Swahili. Each arm is built
as in section 15 (lir.quantize.quantized: the plan applied on the CPU and written out as plain weights) and reads
each question once. The answer is the option letter with the highest probability for the next token. Nothing here
is run on the laptop except `report`, which loads no model.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from lir.common import EVAL_LANGS, ROOT, RUNS, read_jsonl, sha256_file, write_json, write_jsonl

DATASET, DATASET_REV = "CohereLabs/Global-MMLU-Lite", "36c2fd756f19ccf13a9a96c8e53ccecc02192b8b"
ITEMS = 400                      # test questions per language
LETTERS = "ABCD"
INSTRUCTION = "Answer with the letter of the correct option only."
JOB = "mmlu"
# The quantization code needs CircuitKIT, which the job takes from GitHub at the commit the laptop's arms used, and
# the packages of requirements-gpu-quant.txt.
CIRCUITKIT = ("https://github.com/Lexsi-Labs/CircuitKIT", "e71d5e17fe6e2abfde47ec9a7fbaf31649601b78")
OUT = ROOT / "results" / "lir" / "stage2"
B1_ROWS = {"below": "The circuit plan scored below uniform 4-bit on Global-MMLU-Lite",
           "includes": "No difference in Global-MMLU-Lite accuracy between the circuit plan and uniform 4-bit was detected",
           "above": "The circuit plan scored above uniform 4-bit on Global-MMLU-Lite"}
NOT_READ = "B1 is not read: it needs every question of the four languages answered by the circuit plan and by uniform 4-bit"


# ---------------------------------------------------------------- the questions and the score

def question(row: dict) -> str:
    """The user turn for one question: the question, its four options by letter, and one instruction in English."""
    return (f"{row['question']}\n\nA. {row['option_a']}\nB. {row['option_b']}\nC. {row['option_c']}\nD. {row['option_d']}"
            f"\n\n{INSTRUCTION}")


def letter_ids(tok) -> dict[str, list[int]]:
    """For each option letter, the single tokens that write it at the start of an answer: the letter alone and the
    letter after a space. Stops if a letter has no such token or two letters share one."""
    out = {}
    for letter in LETTERS:
        ids = {enc[0] for enc in (tok.encode(text, add_special_tokens=False) for text in (letter, " " + letter)) if len(enc) == 1}
        if not ids:
            raise SystemExit(f"the tokenizer writes {letter} with no single token")
        out[letter] = sorted(ids)
    flat = [i for ids in out.values() for i in ids]
    if len(flat) != len(set(flat)):
        raise SystemExit(f"two option letters share a token: {out}")
    return out


def load_questions(lang: str, download=None) -> tuple[list[dict], str]:
    """The test questions of one language at the pinned revision, and the sha256 of the file they came from."""
    import pyarrow.parquet as pq

    if download is None:
        from huggingface_hub import hf_hub_download

        download = lambda name: hf_hub_download(DATASET, name, repo_type="dataset", revision=DATASET_REV)
    path = Path(download(f"{lang}/test-00000-of-00001.parquet"))
    rows = [dict(r, lang=lang) for r in pq.read_table(path).to_pylist()]
    bad = [r["sample_id"] for r in rows if r["answer"] not in LETTERS]
    if len(rows) != ITEMS or bad or len({r["sample_id"] for r in rows}) != ITEMS:
        raise SystemExit(f"{lang}: {len(rows)} questions (wanted {ITEMS}), {len(bad)} with an answer that is not a letter")
    return rows, sha256_file(path)


def score(tok, model, rows: list[dict], ids: dict[str, list[int]]) -> list[dict]:
    """One forward pass a question. The prediction is the letter whose best token has the highest log-probability
    for the next token; a tie goes to the earlier letter. letter_mass is the probability on all letter tokens."""
    import torch

    from lir.generate import chat_prefix

    out = []
    for r in rows:
        enc = tok(chat_prefix(tok, question(r)), return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logp = torch.log_softmax(model(**enc).logits[0, -1].float(), dim=-1)
        per = {letter: max(logp[i].item() for i in ids[letter]) for letter in LETTERS}
        pred = max(LETTERS, key=lambda letter: per[letter])
        out.append({"uid": f"{r['lang']}/{r['sample_id']}", "lang": r["lang"], "sample_id": r["sample_id"], "answer": r["answer"],
                    "pred": pred, "correct": pred == r["answer"], "logprob": {k: round(v, 4) for k, v in per.items()},
                    "letter_mass": round(sum(torch.exp(logp[i]).item() for v in ids.values() for i in v), 4)})
    return out


# ---------------------------------------------------------------- the job

def job_command(commit: str, flavor: str = "l4x1", timeout: str = "2h", dry_run: bool = False) -> list[str]:
    from lir.hfjob import IMAGE, NAMESPACE, REPO_URL, STAGING

    inner = (f"set -euo pipefail; git clone -q {REPO_URL} repo; cd repo; git checkout -q {commit}; "
             "pip install -q -r requirements-gpu.txt; pip install -q -r requirements-gpu-quant.txt; "
             f"git clone -q {CIRCUITKIT[0]} vendor/CircuitKIT; git -C vendor/CircuitKIT checkout -q {CIRCUITKIT[1]}; "
             "PYTHONPATH=.:vendor/CircuitKIT/src python -m lir.mmlu job")
    return (["hf", "jobs", "run", "--flavor", flavor, "--namespace", NAMESPACE, "--timeout", timeout, "--secrets", "HF_TOKEN",
             "--env", f"LIR_FLAVOR={flavor}", "--env", f"LIR_STAGING_REPO={STAGING}", "--name", f"lir-{JOB}", "--detach"]
            + (["--dry-run"] if dry_run else []) + [IMAGE, "bash", "-c", inner])


def launch(a) -> None:
    git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    stops = []
    if git("status", "--porcelain", "--untracked-files=no"):
        stops.append("tracked files have uncommitted changes: the job would not run the code you see")
    if not git("branch", "-r", "--contains", commit):
        stops.append(f"commit {commit[:7]} is not on the remote yet: push it first, the job clones from GitHub")
    if stops and not a.dry_run:
        raise SystemExit(stops[0])
    cmd = job_command(commit, a.flavor, a.timeout, a.dry_run)
    print(" ".join(cmd[:-1]) + f" '{cmd[-1]}'")
    for why in stops:
        print(f"a launch would stop here: {why}")
    sys.exit(subprocess.run(cmd, cwd=ROOT).returncode)


def arm_plan(arm: str) -> tuple[dict | None, list | None]:
    """The plan of an arm exactly as lir.quantize.evaluate builds it, and its tier size. None for the unquantized arm.
    Stops if the plan is not the one in the committed plan file."""
    from lir import quantize as q

    if arm == "bf16":
        return None, None
    g = q.chosen()
    rung = tuple(g["rung"]) if arm in q.MIXED else None
    plan = q.layer_plans(q.node_scores(), tiers=dict(zip(q.BITS, rung or q.RUNGS[0])))[arm]
    if rung:
        filed = json.loads((q.OUT / q.PLAN_FILE).read_text())
        want = next(r for r in filed["rungs"] if r["name"] == q.rung_name(rung))["arms"][arm]["layers"]
        if q.tier_lists(plan) != want:
            raise SystemExit(f"{arm}: the plan built here is not the plan in {q.PLAN_FILE}")
    return plan, list(rung) if rung else None


def installed() -> dict:
    """What the quantization code runs on in the job: optimum-quanto's version and CircuitKIT's commit. Stops if the
    second install moved torch or transformers away from the pins of requirements-gpu.txt."""
    from importlib import metadata

    pins = dict(line.strip().split("==") for line in (ROOT / "requirements-gpu.txt").read_text().splitlines()
                if "==" in line and not line.startswith("#"))
    moved = {name: metadata.version(name) for name in ("torch", "transformers") if metadata.version(name) != pins[name]}
    if moved:
        raise SystemExit(f"the quantization packages changed {moved}; requirements-gpu.txt pins {[pins[n] for n in moved]}")
    kit = subprocess.run(["git", "-C", str(ROOT / "vendor" / "CircuitKIT"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if kit != CIRCUITKIT[1]:
        raise SystemExit(f"CircuitKIT is at {kit or 'no commit'}, not at {CIRCUITKIT[1]}")
    return {"optimum-quanto": metadata.version("optimum-quanto"), "circuitkit_commit": kit}


def job(a) -> None:
    from huggingface_hub import HfApi

    from lir import generate
    from lir import quantize as q
    from lir.common import empty_cache, git_commit
    from lir.hfjob import STAGING, hardware

    hw = hardware()
    if not hw["cuda_available"]:
        raise SystemExit(f"no CUDA GPU in this job: {hw}")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = RUNS / "hfjob" / JOB
    out.mkdir(parents=True, exist_ok=True)
    data, files = {}, {}
    for lang in EVAL_LANGS:
        data[lang], files[lang] = load_questions(lang)
    record = {"job": JOB, "started_utc": stamp, "commit": git_commit(), "hardware": hw, "quantization": installed(),
              "dataset": DATASET, "dataset_revision": DATASET_REV,
              "files_sha256": files, "items_per_language": ITEMS, "instruction": INSTRUCTION, "arms": {}}
    for arm in q.ARMS:
        t0 = time.time()
        plan, rung = arm_plan(arm)
        tok, model = generate.load_model(None)
        info = {"rung": rung, "device": str(model.device)}
        if plan is not None:
            info.update(q.quantized(model, plan), layers=q.tier_lists(plan))
        ids = letter_ids(tok)
        rows = [r for lang in EVAL_LANGS for r in score(tok, model, data[lang], ids)]
        write_jsonl(out / f"{arm}.jsonl", [dict(r, arm=arm) for r in rows])
        info.update(letter_tokens=ids, seconds=round(time.time() - t0, 1), answered=len(rows))
        record["arms"][arm] = info
        write_json(out / "job.json", record)   # kept current, so a job that dies later still leaves a record
        print(f"{arm}: {len(rows)} questions in {info['seconds']} s", flush=True)
        del model
        empty_cache()
    record["finished_utc"] = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    write_json(out / "job.json", record)
    dest = f"{JOB}/{stamp}-{(record['commit'] or 'unknown')[:7]}"
    api = HfApi()
    api.create_repo(STAGING, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=out, repo_id=STAGING, repo_type="dataset", path_in_repo=dest, commit_message=f"{JOB}: answers and record")
    print(f"uploaded to {STAGING}/{dest}")


# ---------------------------------------------------------------- the reading

def side(c: dict) -> str:
    return "above" if c["ci95"][0] > 0 else "below" if c["ci95"][1] < 0 else "includes"


def result(rows: dict[str, list[dict]], langs: list[str] = EVAL_LANGS, items: int = ITEMS) -> dict:
    """rows: arm -> the scored questions of that arm. An arm is read only if it answered every question of every
    language. A difference averages a question over the languages that hold it, then bootstraps the questions."""
    from lir.seeds import pooled

    full = {arm: r for arm, r in rows.items()
            if all(sum(x["lang"] == lang for x in r) == items for lang in langs) and len(r) == items * len(langs)}
    C = {arm: {lang: {x["sample_id"]: float(x["correct"]) for x in r if x["lang"] == lang} for lang in langs} for arm, r in full.items()}
    acc = {arm: {**{lang: 100 * sum(v[lang].values()) / items for lang in langs},
                 "all": 100 * sum(sum(v[lang].values()) for lang in langs) / (items * len(langs))} for arm, v in C.items()}
    shared = all(set(v[lang]) == set(v[langs[0]]) for v in C.values() for lang in langs)   # the same questions in every language?
    def minus(a: str, b: str) -> dict:   # a minus b
        per = [{k: C[a][lang][k] - C[b][lang][k] for k in C[a][lang]} for lang in langs]
        if not shared:   # every question is its own unit
            per = [{f"{lang}/{k}": v for lang, d in zip(langs, per) for k, v in d.items()}]
        return pooled(per)
    res = {"arms_read": sorted(full), "arms_left_out": sorted(set(rows) - set(full)), "accuracy": acc, "questions_shared_across_languages": shared,
           "letter_mass": {arm: sum(x["letter_mass"] for x in r) / len(r) for arm, r in full.items()},
           "vs_bf16": {arm: minus(arm, "bf16") for arm in full if arm != "bf16" and "bf16" in full}}
    if "circuit" in full and "int4" in full:
        b1 = minus("circuit", "int4")
        res["b1"] = dict(b1, side=side(b1), report=B1_ROWS[side(b1)])
    randoms = sorted(arm for arm in full if arm.startswith("random-"))
    if "circuit" in full and randoms:
        res["circuit_minus_random"] = {arm: minus("circuit", arm) for arm in randoms}
    return res


def report_md(res: dict) -> str:
    from lir.repeat import ci

    langs = [l for l in EVAL_LANGS]
    lines = ["### Global-MMLU-Lite on the arms of section 15 (section 15b: B1)", "",
             f"`{DATASET}` at revision `{DATASET_REV[:8]}`: the {ITEMS} test questions of each of English, Korean, Chinese and Swahili, "
             "four options each. Each arm is built as in section 15, the plan applied on the CPU and written out as plain weights, and "
             "run on one GPU in a Hugging Face job; the laptop's arms are not these processes. The answer is the option letter with the "
             "highest probability for the next token after the question, with one instruction in English. No judge model reads "
             "anything. Chance is 25%. A difference averages a question over its languages, then bootstraps the questions (95% "
             "interval)." + ("" if res.get("questions_shared_across_languages", True) else
                             " The four languages do not hold the same questions, so every question is its own unit."), ""]
    if res.get("job"):
        j = res["job"]
        lines += [f"Job upload `{j['folder']}`, commit `{(j['commit'] or '')[:7]}`, {'; '.join(j['gpu'])}.", ""]
    if "b1" in res:
        lines += [f"**B1. {res['b1']['report']}.** Circuit plan minus uniform 4-bit, accuracy over the four languages: {ci(res['b1'])} points.", ""]
    else:
        lines += [f"**{NOT_READ}.**", ""]
    lines += ["Accuracy (%), with no rule:", "", "| Arm | " + " | ".join(langs) + " | Four languages | Against the unquantized model |",
              "|---|" + "---|" * (len(langs) + 2)]
    for arm in [a for a in ("bf16", "int4", "circuit", "random-0", "random-2", "random-4") if a in res["accuracy"]]:
        v = res["accuracy"][arm]
        lines.append(f"| `{arm}` | " + " | ".join(f"{v[l]:.1f}" for l in langs) + f" | {v['all']:.1f} | "
                     + (ci(res["vs_bf16"][arm]) if arm in res.get("vs_bf16", {}) else "") + " |")
    if res.get("circuit_minus_random"):
        lines += ["", "The circuit plan minus each random plan, with no rule: "
                  + "; ".join(f"`{arm}` {ci(v)}" for arm, v in res["circuit_minus_random"].items()) + "."]
    lines += ["", "Probability on the four option letters, mean over the questions: "
              + ", ".join(f"`{arm}` {v:.2f}" for arm, v in res["letter_mass"].items()) + ". A low figure would mean the model wants to "
              "write something other than a letter, and the score would then say less."]
    if res.get("arms_left_out"):
        lines += ["", "Left out, because not every question has an answer: " + ", ".join(res["arms_left_out"]) + "."]
    lines += ["", "This does not enter Q0 to Q3 or the table of section 15. Not measured: the fine-tuned models, IFEval, and any "
              "question outside these four languages."]
    return "\n".join(lines)


def report(a) -> None:
    from lir import gpu

    folder = gpu.newest(JOB)
    if folder is None:
        print("pending, nothing written: no upload of the job in runs/gpu (run `python -m lir.gpu fetch` first)")
        return
    rows = {p.stem: read_jsonl(p) for p in sorted(folder.glob("*.jsonl"))}
    res = result(rows)
    rec = json.loads((folder / "job.json").read_text())
    res["job"] = {"folder": folder.name, "commit": rec["commit"], "gpu": rec["hardware"]["gpus"], "flavor": rec["hardware"]["flavor"],
                  "libraries": rec["hardware"]["libraries"], "quantization": rec.get("quantization"), "files_sha256": rec["files_sha256"], "finished_utc": rec.get("finished_utc")}
    text = report_md(res)
    write_json(OUT / "quantization_mmlu.json", res)
    (OUT / "quantization_mmlu.md").write_text(text + "\n")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["launch", "job", "report"])
    ap.add_argument("--flavor", default="l4x1")
    ap.add_argument("--timeout", default="2h")
    ap.add_argument("--dry-run", action="store_true", help="launch: print the job and submit nothing")
    a = ap.parse_args()
    {"launch": launch, "job": job, "report": report}[a.what](a)


if __name__ == "__main__":
    main()
