"""Upload datasets, adapters and evaluation outputs to the lost-in-refusal HF org.

Repos are created private; making them public is a separate, deliberate step
(`--public`), planned for the Nov 13 freeze.

  python -m lir.publish data
  python -m lir.publish evals --arms base plain safegrad
  python -m lir.publish adapter --arm plain
  python -m lir.publish stage2 --dry-run      # stage 2 arm folders and tables; lists the files, uploads nothing
  python -m lir.publish adapters --dry-run    # stage 2 adapters, one folder each in a single model repo
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from lir.common import DATA, HF_ORG, ROOT, RUNS

DATA_REPO = f"{HF_ORG}/lost-in-refusal-data"
EVAL_REPO = f"{HF_ORG}/lost-in-refusal-eval-outputs"
CARDS = ROOT / "docs" / "lir" / "cards"


def _repo(api: HfApi, repo_id: str, kind: str, public: bool) -> None:
    api.create_repo(repo_id, repo_type=kind, private=not public, exist_ok=True)


def push_data(api: HfApi, public: bool) -> None:
    _repo(api, DATA_REPO, "dataset", public)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for p in DATA.glob("*.json*"):
            if p.name.startswith(("bactrian_candidates", "bactrian_guard")):
                (t / "screening").mkdir(exist_ok=True)
                shutil.copy(p, t / "screening" / p.name)
            else:
                shutil.copy(p, t / p.name)
        shutil.copy(CARDS / "data.md", t / "README.md")
        api.upload_folder(folder_path=t, repo_id=DATA_REPO, repo_type="dataset",
                          commit_message="Frozen datasets")
    print("pushed", DATA_REPO)


def push_evals(api: HfApi, arms: list[str], public: bool) -> None:
    _repo(api, EVAL_REPO, "dataset", public)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for arm in arms:
            shutil.copytree(RUNS / "eval" / arm, t / arm)
        if (RUNS / "swcheck").exists():  # back-translations behind the judge check
            shutil.copytree(RUNS / "swcheck", t / "judge-check-backtranslations")
        res = ROOT / "results" / "lir"
        for p in list(res.glob("*.json")) + list(res.glob("*.md")) + list(res.glob("*.png")):
            shutil.copy(p, t / p.name)
        for sub in ["vs_plain", "steering", "replication", "circuits"]:
            if (res / sub).exists():
                shutil.copytree(res / sub, t / sub)
        shutil.copy(CARDS / "evals.md", t / "README.md")
        api.upload_folder(folder_path=t, repo_id=EVAL_REPO, repo_type="dataset",
                          commit_message=f"Evaluation outputs: {', '.join(arms)}")
    print("pushed", EVAL_REPO)


def push_adapter(api: HfApi, arm: str, public: bool) -> None:
    repo_id = f"{HF_ORG}/tiny-aya-global-bactrian-lora-{arm}"
    _repo(api, repo_id, "model", public)
    run = RUNS / arm
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        shutil.copytree(run / "adapter", t, dirs_exist_ok=True)
        for name in ["run.json", "train_log.jsonl", "surgery_log.jsonl"]:
            if (run / name).exists():
                shutil.copy(run / name, t / name)
        shutil.copy(CARDS / f"adapter-{arm}.md", t / "README.md")
        api.upload_folder(folder_path=t, repo_id=repo_id, commit_message=f"{arm} LoRA adapter")
    print("pushed", repo_id)


ADAPTER_REPO = f"{HF_ORG}/tiny-aya-lora-stage2"
# One folder each. The seed 42 plain and SafeGrad runs on Tiny Aya Global are the two stage 1 repos.
STAGE2_ADAPTERS = ["plain-s1", "safegrad-s1", "plain-s2", "safegrad-s2", "plain-mix1", "plain-mix5", "plain-noko",
                   "plain-noko-s1", "plain-noko-s2", "plain-ayasize", "aya-plain", "aya-safegrad", "earth-plain",
                   "earth-safegrad", "plain-s1-vec", "plain-s2-vec"]


def _relative(x):
    """A run record with every path under the repository written relative to it."""
    if isinstance(x, dict):
        return {k: _relative(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_relative(v) for v in x]
    if isinstance(x, str) and x.startswith(f"{ROOT}/"):
        return x[len(f"{ROOT}/"):]
    return x


def _text(p: Path) -> str:
    """What a staged file says in text: the whole file, or the header of a safetensors file."""
    if p.suffix != ".safetensors":
        return p.read_text()
    with p.open("rb") as f:
        return f.read(int.from_bytes(f.read(8), "little")).decode("utf-8", "replace")


def stage_adapters(t: Path, arms: list[str]) -> list[Path]:
    """Copy each arm's adapter weights and config, run record and logs to t/<arm>/. The tokenizer files are the
    base model's and are left out. Stops if anything staged names the home directory."""
    import json

    for arm in arms:
        run, dest = RUNS / arm, t / arm
        dest.mkdir(parents=True)
        for name in ["adapter_config.json", "adapter_model.safetensors"]:
            shutil.copy(run / "adapter" / name, dest / name)
        record = _relative(json.loads((run / "run.json").read_text()))
        (dest / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        for name in ["train_log.jsonl", "surgery_log.jsonl"]:
            if (run / name).exists():
                shutil.copy(run / name, dest / name)
    files = sorted(p for p in t.rglob("*") if p.is_file())
    for p in files:
        if str(Path.home()) in _text(p):
            raise SystemExit(f"{p.relative_to(t)} names the home directory")
    return files


def push_adapters(api: HfApi, arms: list[str] | None, public: bool, dry_run: bool) -> None:
    """The stage 2 adapters as folders of one model repo, with the card as its README."""
    arms = arms or STAGE2_ADAPTERS
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        files = stage_adapters(t, arms)
        shutil.copy(CARDS / "adapters-stage2.md", t / "README.md")
        files = [t / "README.md"] + files
        print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e9:.2f} GB -> {ADAPTER_REPO}")
        if dry_run:
            for p in files:
                print(" ", p.relative_to(t))
            print("dry run: nothing uploaded")
            return
        _repo(api, ADAPTER_REPO, "model", public)
        api.upload_folder(folder_path=t, repo_id=ADAPTER_REPO,
                          commit_message=f"Stage 2 LoRA adapters: {', '.join(arms)}")
    print("pushed", ADAPTER_REPO)


# Stage 2 arms. The second list was still running when the first went up; each joins an upload once it is fully judged.
STAGE2_ARMS = ["plain-s1", "safegrad-s1", "plain-s2", "safegrad-s2", "plain-mix1", "plain-mix5", "plain-noko",
               "earth-base", "earth-plain", "earth-safegrad"]
STAGE2_LATER = ["plain-s1-vec", "plain-s2-vec", "plain-noko-s1", "plain-noko-s2", "aya-plain", "aya-safegrad", "plain-ayasize"]
TRUNC_ARMS = ["base", "plain", "safegrad"] + STAGE2_ARMS  # arms whose capped answers may have been generated again


def unfinished(arm_dir: Path) -> list[str]:
    """Sets of an arm whose guard file is missing or has fewer rows than its generation file."""
    rows = lambda p: sum(1 for line in p.read_text().splitlines() if line.strip())
    bad = []
    for gen in sorted(arm_dir.glob("*.gen.jsonl")):
        guard = gen.with_name(gen.name.replace(".gen.", ".guard."))
        if not guard.exists() or rows(guard) < rows(gen):
            bad.append(gen.name.split(".")[0])
    task, scores = arm_dir / "task.gen.jsonl", arm_dir / "task.judge.jsonl"   # the task judge runs as its own step
    if task.exists() and "task" not in bad and (not scores.exists() or rows(scores) < rows(task)):
        bad.append("task scores")
    return bad


def push_stage2(api: HfApi, arms: list[str] | None, dry_run: bool, aya_task: bool = False) -> None:
    """Add stage 2 outputs to the outputs repo: arm folders, re-generated answers, the NLLB re-translations and
    the committed stage 2 tables. The answers to the held-out Aya prompts go up only with aya_task, because
    each row quotes an Aya prompt and its reference answer.

    Nothing already in the repo is removed, and only the README card and files whose content changed are
    replaced. The repo is public, so the files are public as soon as they are uploaded. With no arms given, every
    finished stage 2 arm goes up and the ones still running are named and left out.
    """
    from lir import trunc

    waiting = []
    if arms is None:
        arms = list(STAGE2_ARMS)
        for arm in STAGE2_LATER:
            d = RUNS / "eval" / arm
            if d.is_dir() and not unfinished(d):
                arms.append(arm)
            else:
                waiting.append(arm)
    for arm in arms:
        d = RUNS / "eval" / arm
        if not d.is_dir():
            sys.exit(f"{arm}: no outputs at {d}")
        if bad := unfinished(d):
            sys.exit(f"{arm}: not fully judged yet ({', '.join(bad)}); nothing uploaded")
    rows = lambda p: sum(1 for line in p.read_text().splitlines() if line.strip())
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for arm in arms:
            shutil.copytree(RUNS / "eval" / arm, t / arm)
        again = [arm for arm in TRUNC_ARMS if all(trunc.after(arm, name) is not None for name in trunc.SETS)]
        for arm in again:
            shutil.copytree(trunc.REGEN / arm, t / "truncation-regenerated" / arm)
        aya = [d.name for d in sorted((RUNS / "ayatask").glob("*")) if aya_task and (d / "gen.jsonl").exists()
               and (d / "judge.jsonl").exists() and rows(d / "judge.jsonl") == rows(d / "gen.jsonl")]
        for arm in aya:
            shutil.copytree(RUNS / "ayatask" / arm, t / "aya-task" / arm)
        bt = t / "judge-check-backtranslations"
        bt.mkdir()
        for p in sorted((RUNS / "swcheck").glob("*.nllb.bt*.jsonl")):
            shutil.copy(p, bt / p.name)
        # only tables that are committed: a result file a queue step has just written has not been read yet
        res = Path("results") / "lir" / "stage2"
        tracked = subprocess.run(["git", "ls-files", str(res)], cwd=ROOT, capture_output=True, text=True, check=True)
        for rel in tracked.stdout.splitlines():
            if Path(rel).name != "README.md":
                dest = t / "stage2" / Path(rel).relative_to(res)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / rel, dest)
        shutil.copy(CARDS / "evals.md", t / "README.md")
        files = sorted(p for p in t.rglob("*") if p.is_file())
        size = sum(p.stat().st_size for p in files)
        print(f"{len(files)} files, {size / 1e6:.1f} MB -> {EVAL_REPO}")
        print("arms:", ", ".join(arms))
        print("re-generated answers:", ", ".join(again) or "none yet", "| Aya task answers:",
              (", ".join(aya) or "none yet") if aya_task else "left out (--aya-task adds them)")
        if waiting:
            print("not finished, left out:", ", ".join(waiting))
        if dry_run:
            for p in files:
                print(" ", p.relative_to(t))
            print("dry run: nothing uploaded")
            return
        api.upload_folder(folder_path=t, repo_id=EVAL_REPO, repo_type="dataset",
                          commit_message=f"Stage 2 evaluation outputs: {', '.join(arms)}")
    print("pushed", EVAL_REPO)


SPACE_REPO = f"{HF_ORG}/lost-in-refusal-demo"


def push_space(api: HfApi, public: bool) -> None:
    """Static Space (free for organisations): index.html plus the outputs it shows, bundled as JSON."""
    import json

    from lir.common import read_jsonl

    arms = ["base", "plain", "safegrad"]
    data = {}
    for set_name in ["attack", "xstest"]:
        rows = {}
        for arm in arms:
            for r in read_jsonl(RUNS / "eval" / arm / f"{set_name}.guard.jsonl"):
                x = rows.setdefault(r["uid"], {"f": r["family"], "l": r["lang"], "p": r["prompt"]})
                x[arm] = {"s": r["safety"], "r": r["refusal"], "t": r["response"]}
        data[set_name] = sorted(rows.values(), key=lambda x: (x["l"], x["f"]))
    res = json.loads((ROOT / "results" / "lir" / "results.json").read_text())
    asr = {a: {l: 100 * res[a]["asr_strict"][l]["mean"] for l in ["en", "ko", "zh", "sw"]} for a in arms}
    sw = json.loads((ROOT / "results" / "lir" / "swahili_backtranslation_check.json").read_text())["corrected_sw_asr"]
    for a in arms:
        asr[a]["sw"] = 100 * sw[a]["asr_strict_corrected"]
    data["stats"] = {"asr": asr}
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="static", private=not public, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        shutil.copy(ROOT / "space" / "index.html", t / "index.html")
        shutil.copy(ROOT / "space" / "README.md", t / "README.md")
        (t / "examples.json").write_text(json.dumps(data, ensure_ascii=False))
        api.upload_folder(folder_path=t, repo_id=SPACE_REPO, repo_type="space", commit_message="Demo Space",
                          delete_patterns=["style.css"])
    print("pushed", SPACE_REPO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "evals", "adapter", "adapters", "space", "stage2"])
    ap.add_argument("--arm")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="stage2 and adapters: list the files and upload nothing")
    ap.add_argument("--aya-task", action="store_true",
                    help="stage2 only: also upload the answers to the held-out Aya prompts (they quote the Aya rows)")
    a = ap.parse_args()
    api = HfApi()
    if a.what == "stage2":
        push_stage2(api, a.arms, a.dry_run, a.aya_task)
    elif a.what == "data":
        push_data(api, a.public)
    elif a.what == "evals":
        push_evals(api, a.arms or ["base", "plain", "safegrad"], a.public)
    elif a.what == "space":
        push_space(api, a.public)
    elif a.what == "adapters":
        push_adapters(api, a.arms, a.public, a.dry_run)
    else:
        push_adapter(api, a.arm, a.public)


if __name__ == "__main__":
    main()
