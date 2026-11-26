"""Rebuild the stage 2 tables from the published files, in a fresh clone of this repository.

  python -m lir.rebuild fetch      # the public outputs at a pinned revision -> runs/ (no login, about 455 MB)
  python -m lir.rebuild records    # the training records, no weights -> runs/<arm>/
  python -m lir.rebuild tables     # runs each report whose inputs are there and compares its tables with the committed ones

`fetch` is for a clone with no runs/ folder. It stops if runs/eval exists, so it cannot touch the folder the runs
were made in, and the other two commands only work in a folder that `fetch` made. The pinned revision is the upload
of Nov 25, 2026. Besides the arm folders it holds the answers generated again, the result files of the Hugging Face
GPU jobs, the six further languages, the unsafe prompts of XSTest and the quantization arms, and `layout` puts each
where its report reads it.

`tables` deletes each table before the command that writes it runs, so a command that writes nothing cannot pass.
A table counts as rebuilt only if it comes back byte for byte the same as the committed file. The report commands
run with the Hub switched off and need no model and no GPU. Most need nothing outside the standard library; the
figures need matplotlib, and the five tables written by `lir.metrics` need AuditKIT, which has no dependencies of
its own (the README gives the commit). A command whose package or input is missing is skipped and named.

`records` reads the seed 42 records from the two public stage 1 adapter repositories and the others from the stage 2
adapter repository, which has been public since Nov 25, 2026. If it cannot be read, the reports that check a
training record are skipped and named. Its pinned revision is the upload of Nov 25, 5:02 PM ET, which added the nine
laptop runs made after Nov 22 (47 folders), so the five reports that read one of their records are in COMMANDS.

NOT_REBUILT names every other table with the reason: an input that is not published, a package or a model its
command needs, or what the command writes from the published files that the committed file does not hold.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from lir.common import HF_ORG, ROOT, RUNS

OUTPUTS = f"{HF_ORG}/lost-in-refusal-eval-outputs"
OUTPUTS_REV = "cf19782b9445b2aa56f3ab069fe269888d39a3f6"    # the upload of Nov 25, 2026 (979 files)
ADAPTERS = f"{HF_ORG}/tiny-aya-lora-stage2"
ADAPTERS_REV = "d79340e8e5cae22266eb0cd5210cf81671297d37"   # the upload of Nov 25, 2026 (47 folders)
STAGE1 = {"plain": (f"{HF_ORG}/tiny-aya-global-bactrian-lora-plain", "0e57114e1804beb4c81a3afad0076d9733067010"),   # seed 42, public
          "safegrad": (f"{HF_ORG}/tiny-aya-global-bactrian-lora-safegrad", "8b2687f885d388e4a3a370fb04252a21c3f13e99")}
RECORD_FILES = ("run.json", "train_log.jsonl", "surgery_log.jsonl")
MARK = "rebuild.json"                       # written by fetch into runs/; says where the files came from
S2 = Path("results") / "lir" / "stage2"

# What each report command writes under results/lir/stage2, and what it needs beyond the public outputs and the
# standard library: nothing, AuditKIT ("auditkit") or the training records ("records"). Images are written by the
# command but not compared.
COMMANDS = [
    {"args": ["lir.seeds"], "tables": ["seeds.json", "seeds.md"]},
    {"args": ["lir.seeds", "--figure"], "tables": ["drift_by_language.json", "korean_by_setting.json"],
     "images": ["drift_by_language.png", "korean_by_setting.png"]},
    {"args": ["lir.metrics", "--arms", "base", "plain", "safegrad", "plain-s1", "safegrad-s1", "plain-s2", "safegrad-s2",
              "plain-mix1", "plain-mix5", "--out", str(S2)], "tables": ["results.json", "results.md"], "needs": "auditkit"},
    {"args": ["lir.metrics", "--arms", "safegrad", "plain-mix1", "plain-mix5", "--ref", "safegrad", "--out", str(S2 / "mixing")],
     "tables": ["mixing/results.json", "mixing/results.md"], "needs": "auditkit"},
    {"args": ["lir.metrics", "--arms", "earth-base", "earth-plain", "earth-safegrad", "--ref", "earth-base", "--out", str(S2 / "earth")],
     "tables": ["earth/results.json", "earth/results.md"], "needs": "auditkit"},
    {"args": ["lir.metrics", "--arms", "base", "aya-plain", "aya-safegrad", "--out", str(S2 / "aya")],
     "tables": ["aya/results.json", "aya/results.md"], "needs": "auditkit"},
    {"args": ["lir.metrics", "--arms", "base", "plain-s1", "safegrad-s1", "plain-s1-vec", "plain-s2", "safegrad-s2", "plain-s2-vec",
              "--out", str(S2 / "vector")], "tables": ["vector/results.json", "vector/results.md"], "needs": "auditkit"},
    {"args": ["lir.earth"], "tables": ["earth/checks.json", "earth/checks.md"]},
    {"args": ["lir.vector", "report"], "tables": ["vector.json", "vector.md"]},
    {"args": ["lir.harm"], "tables": ["harm_categories.json", "harm_categories.md"]},
    {"args": ["lir.harm", "--regenerated"], "tables": ["harm_categories_regenerated.json", "harm_categories_regenerated.md"]},
    {"args": ["lir.trunc", "count"], "tables": ["truncation/counts.json", "truncation/counts.md"]},
    {"args": ["lir.trunc", "report"], "tables": ["truncation/report.json", "truncation/report.md"]},
    {"args": ["lir.truncsplit"], "tables": ["truncation/split.json", "truncation/split.md"]},
    {"args": ["lir.tasklen"], "tables": ["task_length.json", "task_length.md"]},
    {"args": ["lir.loose"], "tables": ["unsafe_only.json", "unsafe_only.md"]},
    {"args": ["lir.circuits", "nodes"], "tables": ["circuit_nodes.json", "circuit_nodes.md"]},
    # From the folders added on Nov 25, 2026. A command named `again` reads the table of the `report` before it.
    {"args": ["lir.morelangs", "report"], "tables": ["more_languages.json", "more_languages.md"]},
    {"args": ["lir.morelangs", "figure"], "tables": ["ten_languages.json"], "images": ["ten_languages.png"]},
    {"args": ["lir.morelangs", "again"], "tables": ["more_languages_regen.json", "more_languages_regen.md"]},
    {"args": ["lir.xsunsafe", "report"], "tables": ["second_benchmark.json", "second_benchmark.md"]},
    {"args": ["lir.xsunsafe", "again"], "tables": ["second_benchmark_regen.json", "second_benchmark_regen.md"]},
    {"args": ["lir.quantize", "report"], "tables": ["quantization.json", "quantization.md"]},
    {"args": ["lir.quantrank", "report"], "tables": ["quantization_ranks.json", "quantization_ranks.md"]},
    {"args": ["lir.mmlu", "report"], "tables": ["quantization_mmlu.json", "quantization_mmlu.md"]},
    {"args": ["lir.granite41", "noko"], "tables": ["granite41/noko.json", "granite41/noko.md"]},
    {"args": ["lir.granite41", "task"], "tables": ["granite41/task.json", "granite41/task.md"]},
    {"args": ["lir.seeds", "--control", "plain-noko", "plain-noko-s1", "plain-noko-s2"], "tables": ["noko.json", "noko.md"], "needs": "records"},
    {"args": ["lir.seeds", "--times"], "tables": ["train_time.json", "train_time.md"], "needs": "records"},
    {"args": ["lir.repeat", "aya"], "tables": ["aya/seeds.json", "aya/seeds.md"], "needs": "records"},
    {"args": ["lir.repeat", "k1"], "tables": ["truncation/k1.json", "truncation/k1.md"], "needs": "records"},
    {"args": ["lir.repeat", "earth"], "tables": ["earth/seeds.json", "earth/seeds.md"], "needs": "records"},
    {"args": ["lir.section13", "only"], "tables": ["single_language.json", "single_language.md"], "needs": "records"},
    {"args": ["lir.section13", "seeds"], "tables": ["single_language_seeds.json", "single_language_seeds.md"], "needs": "records"},
    # With the records of the nine runs added to the adapter repository on Nov 25, 2026. `mix-again` reads the table of
    # `mix`, and `lir.reviewtables` reads tables that `lir.morelangs`, `lir.quantrank` and `lir.sensitivity` write before it.
    {"args": ["lir.repeat", "mix"], "tables": ["mixing/seeds.json", "mixing/seeds.md"], "needs": "records"},
    {"args": ["lir.repeat", "mix-again"], "tables": ["mixing/seeds_regen.json", "mixing/seeds_regen.md"], "needs": "records"},
    {"args": ["lir.sensitivity", "report"], "tables": ["sensitivity.json", "sensitivity.md"], "needs": "records"},
    {"args": ["lir.train_aligntune", "report"], "tables": ["aligntune.json", "aligntune.md"], "needs": "records"},
    {"args": ["lir.reviewtables"], "tables": ["review_tables.json", "review_tables.md"], "needs": "records"},
]

# Every other file in results/lir/stage2 (the status page aside), and why no command above rebuilds it. Where a
# reason says how a rebuilt file differs, the command was run on the published files on Nov 25, 2026.
NOT_REBUILT = [
    (("vector_aya.json", "vector_aya.md"),
     "`python -m lir.section13 vector` checks the hashes of the adapters the edit was built from, so it needs the adapter weights"),
    (("refusal_direction.json", "refusal_direction.md"),
     "`python -m lir.direction` reads the activation files, which are not published"),
    (("gpu.json", "gpu.md"),
     "`python -m lir.gpu` reads Q1 of section 12.2 against the bfloat16 arms with the adapter left unmerged when the files "
     "of job `unmerged` are in runs/gpu, which they are since the upload of Nov 25, and names the staging revision it finds "
     "there. The committed table is the reading of Nov 23, made before that job ran, so the command now writes another "
     "table; the unmerged arms are read in nf4_seeds.md (section 14f)"),
    (("aya/task.json", "aya/task.md"),
     "`python -m lir.aya taskreport` reads the answers to the held-out Aya prompts, which are not published because they quote Aya rows"),
    (("aya/checks.json", "aya/checks.md"),
     "`python -m lir.aya report` copies a file path from the seed 42 plain record, and the public copy of that record still names "
     "a folder on the training machine. The .md comes out the same and the .json differs in that one field"),
    (("swahili_nllb.json", "swahili_nllb_seeds.json"),
     "written by `python -m lir.nllb backtranslate`, which runs the translation model and the guard. The back-translations are "
     "published; the step that judges them is not run again here"),
    (("ko_drift.md", "ko_rest.md", "second_judge_check.md"),
     "a person's labels and the write-up of the two second-judge checks; their inputs are in results/lir/labels"),
    (("dose.json", "dose.md"),
     "`python -m lir.dose report` reads the answers, verdicts and task scores of the five arms of section 19 and their adapters' "
     "records, which are on the laptop and not published"),
    (("dose_seeds.json", "dose_seeds.md"),
     "`python -m lir.dose seeds` reads the answers, verdicts and task scores of the six arms of section 19b and their adapters' "
     "records, which are on the laptop and not published"),
    (("cloud_jobs.json", "cloud_jobs.md"),
     "`python -m lir.cloudhours` reads the job records of the organisation's Hugging Face account, which need a login with access"),
    (("deployment.json", "deployment.md"),
     "`python -m lir.deploy report` reads the job records and timing files of section 22, which are in the private staging dataset"),
    (("xlsum_premise.json",),
     "`python -m lir.xlsum premise` reads the ROUGE-L and task scores of the base run and of plain at seed 42 of section 20, "
     "which are in the private staging dataset"),
    (("xlsum.json", "xlsum.md"),
     "`python -m lir.xlsum report` reads the answers, verdicts and scores of the seven runs of section 20, which are in the "
     "private staging dataset: the answers quote news articles, and only scores are published"),
    (("xlsum_mix.json", "xlsum_mix.md"),
     "`python -m lir.xlsum mix` reads the answers, verdicts and scores of the three mixing runs of section 20b and of the runs "
     "of section 20, which are in the private staging dataset"),
    (("xlsum_regen.json", "xlsum_regen.md"),
     "`python -m lir.xlsumregen again` reads the verdicts of the runs of sections 20 and 20b and of the answers generated again, "
     "which are in the private staging dataset"),
    (("xlsum_cut.json", "xlsum_cut.md"),
     "`python -m lir.xlsumcut` reads the Korean attack verdicts of the base run and the plain runs of section 20, which are in "
     "the private staging dataset"),
    (("third_judge_check.md",),
     "`python -m lir.judge3 check-a` reads the verdicts of the Hugging Face job of section 17, which are in the private staging "
     "dataset; the judge's verdicts on the 120 labelled answers are in results/lir/labels"),
    (("third_judge_inputs.json",),
     "`python -m lir.judge3 inputs` writes its hashes under runs/judge3 and not into this folder, and the committed copy "
     "also names the revision of the private staging dataset that the upload of the two input files returned. From the "
     "published answers the command gives the same hashes"),
    (("second_reader_tiny.md", "second_reader_granite.md"),
     "`python -m lir.reader2 score` reads its keys from runs/humancheck, a folder that is not published, and writes two files "
     "in results/lir/labels. The marks, a copy of each key and the results are in results/lir/labels"),
    (("granite41/person.md",),
     "`python -m lir.granite_reading score` reads its key from runs/humancheck, a folder that is not published, and also "
     "writes two files in results/lir/labels. The answers and verdicts it reads are published, and the marks and a copy "
     "of the key are in results/lir/labels"),
    (("granite41/seeds_regen.json", "granite41/seeds_regen.md"),
     "`python -m lir.granite41 again` writes the languages of its `per_lang` counts in the order of a Python set, which is "
     "not the same from one run of the command to the next. The .md comes out the same, and the .json holds the same "
     "numbers with those keys in another order"),
    (("granite41/seeds.json", "granite41/seeds.md"),
     "`python -m lir.granite41` copies the revision of the private staging dataset that runs/gpu was read from, and the "
     "published files name another one than the committed table (`7c210da3`, not `9070ad38`). It also writes the three "
     "languages of the refusal set in the order of a Python set. The .md comes out the same, and the .json differs in "
     "that field and that order"),
    (("nf4_seeds.json", "nf4_seeds.md"),
     "`python -m lir.nf4seeds` copies the revision of the private staging dataset that runs/gpu was read from, and the "
     "published files name another one than the committed table (`7c210da3`, not `9ae17d8a`). The .md comes out the same "
     "and the .json differs in that one field"),
    (("quantization_gate.json", "quantization_gate.md"),
     "`python -m lir.quantize gate` loads the model once for every plan it measures, in the environment with CircuitKIT "
     "and optimum-quanto"),
    (("quantization_plan.json",),
     "`python -m lir.quantize plan` reads the stage 1 score files and loads no model, but it needs CircuitKIT, which is "
     "not in this repository, in an environment of its own"),
    (("drift_by_language.png", "korean_by_setting.png", "harm_categories.png", "ten_languages.png"),
     "images are not compared; the numbers beside them are"),
]


# Folders of the outputs repository that are not arm folders -> the folder under runs/ their reports read. Each is
# copied as it is, sub-folders included. It is `stage_stage2`, `gpu_files` and FAMILIES of lir.publish read backwards.
FOLDERS = {"truncation-regenerated": "trunc", "gpu": "gpu", "judge-check-backtranslations": "swcheck",
           "more-languages": "morelangs", "more-languages-regenerated": "morelangs-trunc",
           "xstest-unsafe": "xsunsafe", "xstest-unsafe-regenerated": "xsunsafe-trunc", "quantization": "quant"}


def layout(hub: Path, runs: Path) -> dict[str, int]:
    """Copy a download of the outputs repository into the folders the report commands read: each arm folder to
    runs/eval/ and each folder of FOLDERS to its folder under runs/ (truncation-regenerated/ to runs/trunc/, gpu/ to
    runs/gpu/, the back-translations to runs/swcheck/, the six further languages to runs/morelangs/, the unsafe
    prompts of XSTest to runs/xsunsafe/, the answers of those two generated again to runs/morelangs-trunc/ and
    runs/xsunsafe-trunc/, and the quantization arms to runs/quant/). Returns the number of files per folder."""
    n = {"eval": 0, **{key: 0 for key in FOLDERS.values()}}

    def put(src: Path, dest: Path, key: str) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        n[key] += 1

    for d in sorted(p for p in hub.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name not in FOLDERS):
        if any(d.glob("*.gen.jsonl")):   # an arm folder: answers and verdicts side by side
            for p in sorted(x for x in d.iterdir() if x.is_file()):
                put(p, runs / "eval" / d.name / p.name, "eval")
    for name, key in FOLDERS.items():
        for p in sorted(x for x in (hub / name).rglob("*") if x.is_file()):
            rel = p.relative_to(hub / name)
            # the outputs repository names the staging revision file differently from `python -m lir.gpu fetch`
            put(p, runs / key / ("revision.txt" if key == "gpu" and str(rel) == "staging_revision.txt" else rel), key)
    return n


def fetch(revision: str) -> None:
    from huggingface_hub import snapshot_download

    if (RUNS / "eval").exists():
        raise SystemExit(f"{RUNS / 'eval'} exists. This command is for a fresh clone and does not write over a runs folder.")
    with tempfile.TemporaryDirectory() as tmp:
        hub = Path(snapshot_download(OUTPUTS, repo_type="dataset", revision=revision, token=False, local_dir=tmp,
                                     ignore_patterns=["circuits/*", "stage2/*"]))
        n = layout(hub, RUNS)
    (RUNS / MARK).write_text(json.dumps({"outputs": OUTPUTS, "outputs_revision": revision, "files": n, "records": None}, indent=1) + "\n")
    print(f"{OUTPUTS} at {revision[:12]} -> {RUNS}: " + ", ".join(f"{k} {v} files" for k, v in n.items()))


def mark() -> dict:
    p = RUNS / MARK
    if not p.exists():
        raise SystemExit(f"no {p}. Run `python -m lir.rebuild fetch` in a fresh clone first; this command does not use a "
                         "runs folder that was not made by it.")
    return json.loads(p.read_text())


def copy_records(src: Path, runs: Path, arm: str | None = None) -> int:
    """Copy run records and training logs from a download of an adapter repository into runs/<arm>/. With arm, the
    repository holds one adapter at its top level; without, one folder per arm."""
    n = 0
    for p in sorted(src.glob("*" if arm else "*/*")):
        if p.is_file() and p.name in RECORD_FILES:
            dest = runs / (arm or p.parent.name) / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dest)
            n += 1
    return n


def records() -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    m = mark()
    got = {}
    with tempfile.TemporaryDirectory() as tmp:
        for arm, (repo, revision) in STAGE1.items():
            d = snapshot_download(repo, revision=revision, token=False, allow_patterns=list(RECORD_FILES), local_dir=Path(tmp) / arm)
            got[arm] = copy_records(Path(d), RUNS, arm)
        try:   # with the login this machine has, if any
            d = snapshot_download(ADAPTERS, revision=ADAPTERS_REV, allow_patterns=[f"*/{k}" for k in RECORD_FILES],
                                  local_dir=Path(tmp) / "stage2")
            got["stage2"] = copy_records(Path(d), RUNS)
        except HfHubHTTPError as e:
            got["stage2"] = 0
            print(f"{ADAPTERS} did not open ({type(e).__name__}); "
                  "the reports that check a training record will be skipped.")
    m["records"] = {"stage1": got["plain"] + got["safegrad"], "stage2": got["stage2"],
                    "stage2_revision": ADAPTERS_REV if got["stage2"] else None}
    (RUNS / MARK).write_text(json.dumps(m, indent=1) + "\n")
    print(f"record files -> {RUNS}: seed 42 {m['records']['stage1']}, stage 2 {m['records']['stage2']}")


def verdict(committed: bytes | None, rebuilt: bytes | None) -> str:
    if rebuilt is None:
        return "not written"
    if committed is None:
        return "not a committed file"
    return "identical" if rebuilt == committed else "differs"


def committed(rel: Path) -> bytes | None:
    r = subprocess.run(["git", "show", f"HEAD:{rel.as_posix()}"], cwd=ROOT, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def run_command(cmd: dict) -> dict:
    """Delete the command's tables and images, run it, and compare each table with the committed file. A table
    that did not come back, and every image, is restored from the commit so the working tree shows only real
    differences."""
    files = [S2 / t for t in cmd["tables"]]
    images = [S2 / t for t in cmd.get("images", [])]
    for rel in files + images:
        (ROOT / rel).unlink(missing_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", MPLBACKEND="Agg", PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-m", *cmd["args"]], cwd=ROOT, env=env, capture_output=True, text=True)
    out = {"command": "python -m " + " ".join(cmd["args"]), "rc": r.returncode, "tables": {}}
    for rel in files:
        p = ROOT / rel
        out["tables"][rel.relative_to(S2).as_posix()] = verdict(committed(rel), p.read_bytes() if p.exists() else None)
    if r.returncode or "not written" in out["tables"].values():
        out["last_line"] = ((r.stderr.strip() or r.stdout.strip()).splitlines() or [""])[-1][:300]
    restore = [rel.as_posix() for rel in files if not (ROOT / rel).exists()] + [rel.as_posix() for rel in images]
    if restore:
        subprocess.run(["git", "checkout", "-q", "--", *restore], cwd=ROOT, check=True)
    return out


def tables() -> int:
    import importlib.util

    m = mark()
    have = bool(m.get("records") and m["records"].get("stage2"))
    here = {None: True, "records": have, "auditkit": importlib.util.find_spec("auditkit") is not None}
    done, skipped = [], []
    for cmd in COMMANDS:
        if not here[cmd.get("needs")]:
            skipped.append(cmd)
            continue
        res = run_command(cmd)
        done.append(res)
        print(f"{res['command']}: " + ", ".join(f"{t} {v}" for t, v in res["tables"].items())
              + (f" (exit {res['rc']}: {res.get('last_line', '')})" if res["rc"] or "last_line" in res else ""))
    counts = {}
    for res in done:
        for v in res["tables"].values():
            counts[v] = counts.get(v, 0) + 1
    bad = [res for res in done if res["rc"] or set(res["tables"].values()) != {"identical"}]
    print(f"\n{counts.get('identical', 0)} tables from {len(done)} commands came out identical to the committed files"
          + (f"; {sum(v for k, v in counts.items() if k != 'identical')} did not" if bad else "") + ".")
    print(f"Outputs: {m['outputs']} at {m['outputs_revision'][:12]}." + (f" Training records: {ADAPTERS} at {m['records']['stage2_revision'][:12]}." if have else ""))
    for need, why in (("auditkit", "AuditKIT is not installed (the README gives the commit; it has no dependencies)"),
                      ("records", "the training records of the stage 2 runs are not here (`python -m lir.rebuild records`)")):
        if any(cmd.get("needs") == need for cmd in skipped):
            print(f"\nSkipped, because {why}:")
            for cmd in skipped:
                if cmd.get("needs") == need:
                    print(f"  python -m {' '.join(cmd['args'])}: {', '.join(cmd['tables'])}")
    print("\nNot rebuilt by this command:")
    for names, why in NOT_REBUILT:
        print(f"  {', '.join(names)}: {why}.")
    left = subprocess.run(["git", "status", "--short", "--", "results"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print("\n`git status --short results` prints nothing: the folder is as committed." if not left
          else "\nChanged under results/ (see `git diff`):\n" + left)
    return 1 if bad or left else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fetch", "records", "tables"])
    ap.add_argument("--revision", default=OUTPUTS_REV, help="fetch: a revision of the outputs repository (default: the pinned one)")
    a = ap.parse_args()
    if a.what == "fetch":
        fetch(a.revision)
    elif a.what == "records":
        records()
    else:
        sys.exit(tables())


if __name__ == "__main__":
    main()
