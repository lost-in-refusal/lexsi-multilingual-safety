"""Rebuild the stage 2 tables from the published files, in a fresh clone of this repository.

  python -m lir.rebuild fetch      # the public outputs at a pinned revision -> runs/ (no login, about 220 MB)
  python -m lir.rebuild records    # the training records, no weights -> runs/<arm>/
  python -m lir.rebuild tables     # runs each report whose inputs are there and compares its tables with the committed ones

`fetch` is for a clone with no runs/ folder. It stops if runs/eval exists, so it cannot touch the folder the runs
were made in, and the other two commands only work in a folder that `fetch` made.

`tables` deletes each table before the command that writes it runs, so a command that writes nothing cannot pass.
A table counts as rebuilt only if it comes back byte for byte the same as the committed file. The report commands
run with the Hub switched off and need no model and no GPU. Most need nothing outside the standard library; the
figures need matplotlib, and the five tables written by `lir.metrics` need AuditKIT, which has no dependencies of
its own (the README gives the commit). A command whose package or input is missing is skipped and named.

`records` reads the seed 42 records from the two public stage 1 adapter repositories and the others from the stage 2
adapter repository, which is private until the paper is submitted. Without access to it, the reports that check a
training record are skipped and named.
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
OUTPUTS_REV = "dd4400715312dd9a200cf0c8f072dd291a9c76a2"    # the upload of Nov 23, 2026
ADAPTERS = f"{HF_ORG}/tiny-aya-lora-stage2"
ADAPTERS_REV = "292f8ba92163b3a2ac58ce625046f16516d0c70b"   # the upload of Nov 23, 2026
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
    {"args": ["lir.gpu"], "tables": ["gpu.json", "gpu.md"]},
    {"args": ["lir.circuits", "nodes"], "tables": ["circuit_nodes.json", "circuit_nodes.md"]},
    {"args": ["lir.seeds", "--control", "plain-noko", "plain-noko-s1", "plain-noko-s2"], "tables": ["noko.json", "noko.md"], "needs": "records"},
    {"args": ["lir.seeds", "--times"], "tables": ["train_time.json", "train_time.md"], "needs": "records"},
    {"args": ["lir.repeat", "aya"], "tables": ["aya/seeds.json", "aya/seeds.md"], "needs": "records"},
    {"args": ["lir.repeat", "k1"], "tables": ["truncation/k1.json", "truncation/k1.md"], "needs": "records"},
    {"args": ["lir.repeat", "earth"], "tables": ["earth/seeds.json", "earth/seeds.md"], "needs": "records"},
    {"args": ["lir.section13", "only"], "tables": ["single_language.json", "single_language.md"], "needs": "records"},
    {"args": ["lir.section13", "seeds"], "tables": ["single_language_seeds.json", "single_language_seeds.md"], "needs": "records"},
]

# Every other file in results/lir/stage2 (the status page aside), and why no command above rebuilds it.
NOT_REBUILT = [
    (("vector_aya.json", "vector_aya.md"),
     "`python -m lir.section13 vector` checks the hashes of the adapters the edit was built from, so it needs the adapter weights"),
    (("refusal_direction.json", "refusal_direction.md"),
     "`python -m lir.direction` reads the activation files, which are not published"),
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
    (("mixing/seeds.json", "mixing/seeds.md"),
     "`python -m lir.repeat mix` reads the mixing runs at seeds 1 and 2, whose answers and verdicts are not published yet"),
    (("more_languages.json", "more_languages.md", "ten_languages.json"),
     "`python -m lir.morelangs report` and `figure` read the answers and verdicts in the six further languages, which are not "
     "published yet"),
    (("more_languages_regen.json", "more_languages_regen.md"),
     "`python -m lir.morelangs again` reads the answers generated again in the six further languages and their verdicts, "
     "which are not published yet"),
    (("aligntune.json", "aligntune.md"),
     "`python -m lir.train_aligntune report` reads the answers and verdicts of the run through AlignTune and its training record, "
     "which are not published yet"),
    (("drift_by_language.png", "korean_by_setting.png", "harm_categories.png", "ten_languages.png"),
     "images are not compared; the numbers beside them are"),
]


def layout(hub: Path, runs: Path) -> dict[str, int]:
    """Copy a download of the outputs repository into the folders the report commands read: each arm folder to
    runs/eval/, truncation-regenerated/ to runs/trunc/, gpu/ to runs/gpu/ and the back-translations to runs/swcheck/.
    Returns the number of files per folder."""
    n = {"eval": 0, "trunc": 0, "gpu": 0, "swcheck": 0}

    def put(src: Path, dest: Path, key: str) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        n[key] += 1

    for d in sorted(p for p in hub.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if any(d.glob("*.gen.jsonl")):   # an arm folder: answers and verdicts side by side
            for p in sorted(x for x in d.iterdir() if x.is_file()):
                put(p, runs / "eval" / d.name / p.name, "eval")
    for name, key in (("truncation-regenerated", "trunc"), ("gpu", "gpu"), ("judge-check-backtranslations", "swcheck")):
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
        except HfHubHTTPError as e:   # a private repository answers "not found" to anyone without access
            got["stage2"] = 0
            print(f"{ADAPTERS} did not open ({type(e).__name__}). It is private until the paper is submitted; "
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
