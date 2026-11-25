"""Upload datasets, adapters and evaluation outputs to the lost-in-refusal HF org.

Repos are created private; making them public is a separate, deliberate step
(`--public`), planned for the Nov 13 freeze. `create_repo` leaves a repository that exists as it is, so only
`adapters --public` changes the setting of one that is already there.

  python -m lir.publish data
  python -m lir.publish evals --arms base plain safegrad
  python -m lir.publish evals --dry-run       # the stage 1 folders, tables and circuit files; lists them, uploads nothing
  python -m lir.publish adapter --arm plain
  python -m lir.publish stage2 --dry-run      # stage 2 arm folders and tables; lists the files, uploads nothing
  python -m lir.publish adapters --dry-run    # stage 2 adapters, one folder each in a single model repo

A dry run reads the repository on the Hub and says which staged files are new, changed or the same there. It
writes nothing to the Hub. With HF_HUB_OFFLINE=1 it lists the files without that comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from lir.common import DATA, HF_ORG, ROOT, RUNS, read_jsonl, sha256_file

DATA_REPO = f"{HF_ORG}/lost-in-refusal-data"
EVAL_REPO = f"{HF_ORG}/lost-in-refusal-eval-outputs"
CARDS = ROOT / "docs" / "lir" / "cards"


def _repo(api: HfApi, repo_id: str, kind: str, public: bool) -> None:
    api.create_repo(repo_id, repo_type=kind, private=not public, exist_ok=True)


TEXT = {".json", ".jsonl", ".md", ".txt", ".csv", ".py", ".html", ".yaml", ".yml", ".log"}
HOME_DIR = r"(?<![\w.~-])/(?:Users|home|root)/[\w.-]+"   # a home directory on any machine
TOKEN = r"hf_[A-Za-z0-9]{30,}"                           # a Hugging Face token; hf_hub_download in a log is not one
# The fields of a JSON row that quote a dataset or a model. Some of that text merely looks like a path ("on macOS it
# may be /Users/John/Documents/file.txt") and must go up unchanged.
QUOTED = {"prompt", "prompt_en", "instruction", "reference", "response"}


def _text(p: Path) -> str:
    """What a staged file says in text: the whole file, or the header of a safetensors file."""
    if p.suffix != ".safetensors":
        return p.read_text()
    with p.open("rb") as f:
        return f.read(int.from_bytes(f.read(8), "little")).decode("utf-8", "replace")


def _strings(x, key=None):
    """Every string in a parsed JSON value with the key it is under. A key is itself a string, under no key."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield None, k
            yield from _strings(v, k)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v, key)
    elif isinstance(x, str):
        yield key, x


def leaks(p: Path) -> list[str]:
    """What a staged file holds that must not be published, in words. This machine's home directory and a Hugging Face
    token count wherever they are. A home directory of any other machine (/Users/x, /home/x, /root/x) counts everywhere
    but in the QUOTED fields of JSON rows. A file with a suffix in TEXT is read as text, a safetensors file by its
    header and any other file as bytes."""
    home = str(Path.home())
    if p.suffix not in TEXT:
        raw = _text(p).encode() if p.suffix == ".safetensors" else p.read_bytes()
        other = re.search(HOME_DIR.encode(), raw.replace(home.encode(), b""))
        return (["this machine's home directory"] if home.encode() in raw else []) \
            + (["a Hugging Face token"] if re.search(TOKEN.encode(), raw) else []) \
            + ([f"a home directory ({other.group(0).decode()})"] if other else [])
    text = p.read_text(errors="ignore")
    found = (["this machine's home directory"] if home in text else []) + (["a Hugging Face token"] if re.search(TOKEN, text) else [])
    text = text.replace(home, "")   # named already; what is looked for from here on is another machine's
    other = re.search(HOME_DIR, text.replace("\\/", "/"))
    if other and p.suffix in (".json", ".jsonl"):   # look again field by field, so that quoted text is let through
        try:
            docs = [json.loads(line) for line in (text.split("\n") if p.suffix == ".jsonl" else [text]) if line.strip()]
            other = next((m for key, s in _strings(docs) if key not in QUOTED and (m := re.search(HOME_DIR, s))), None)
        except ValueError:
            pass   # not JSON after all: the whole text counts
    return found + ([f"a home directory ({other.group(0)})"] if other else [])


def no_home_paths(folder: Path) -> None:
    """Stop an upload if a staged file names a home directory or holds a token. See `leaks` for what counts."""
    bad = [f"{p.relative_to(folder)} ({', '.join(found)})" for p in sorted(folder.rglob("*")) if p.is_file() and (found := leaks(p))]
    if bad:
        raise SystemExit("these staged files name a home directory or hold a token, nothing uploaded: " + "; ".join(bad[:10]))


# Section 20 trains on BBC articles (XL-Sum, CC BY-NC-SA 4.0). Its training file, its held-out file and every answer
# file of its runs quote them, and none of that is published: it stays under runs/xlsum and in the private staging
# dataset. Its tables in results/lir/stage2 (xlsum.json, xlsum.md, xlsum_premise.json) hold numbers only and go up.
PRIVATE = "xlsum"


def private(rel: Path) -> bool:
    """Whether a path is one of section 20's row files: under a folder named xlsum or one whose name starts with
    xlsum- (its runs), or a .jsonl file whose own name starts with xlsum."""
    *folders, name = rel.parts
    return any(f == PRIVATE or f.startswith(PRIVATE + "-") for f in folders) or (name.startswith(PRIVATE) and name.endswith(".jsonl"))


def no_private(folder: Path) -> None:
    """Stop an upload if a staged file is one of section 20's row files. See `private` for what counts."""
    bad = [str(p.relative_to(folder)) for p in sorted(folder.rglob("*")) if p.is_file() and private(p.relative_to(folder))]
    if bad:
        raise SystemExit("these staged files quote the BBC articles of section 20, which are not published; nothing uploaded: "
                         + "; ".join(bad[:10]))


def blob_id(p: Path) -> str:
    """The id git gives the content of a file. The Hub lists it for every file that is not in LFS."""
    h = hashlib.sha1(f"blob {p.stat().st_size}\0".encode())
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def against_hub(t: Path, files: list[Path], hub: dict[str, str]) -> dict[str, list[str]]:
    """Staged files sorted into new, changed and unchanged against a listing of the repository (path -> the sha256 of
    an LFS file, 64 characters, or the git blob id of any other file, 40 characters), and the files of the repository
    that are not staged. An upload leaves those as they are: nothing is deleted."""
    out = {"new": [], "changed": [], "unchanged": []}
    for p in files:
        rel = p.relative_to(t).as_posix()
        if rel not in hub:
            out["new"].append(rel)
        else:
            same = (sha256_file(p) if len(hub[rel]) == 64 else blob_id(p)) == hub[rel]
            out["unchanged" if same else "changed"].append(rel)
    staged = {p.relative_to(t).as_posix() for p in files}
    out["not staged"] = sorted(set(hub) - staged)
    return out


def hub_listing(api: HfApi, repo_id: str, kind: str) -> tuple[str, dict[str, str]] | None:
    """The head revision of a repository on the Hub and the content id of each file in it, or None if it cannot be
    read (the Hub is switched off, or this login does not see the repository). Reads only."""
    try:
        info = api.repo_info(repo_id, repo_type=kind, files_metadata=True)
    except OSError as e:   # huggingface_hub raises its HTTP and offline errors as OSError
        print(f"{repo_id} was not read ({type(e).__name__}): no comparison with the Hub")
        return None
    return info.sha, {s.rfilename: (s.lfs.sha256 if s.lfs else s.blob_id) for s in info.siblings}


def show_dry_run(api: HfApi, repo_id: str, kind: str, t: Path, files: list[Path]) -> None:
    """Print what an upload of the staged files would do to the repository as it is now, then every staged file:
    `+` new, `~` changed, no mark for a file that is the same on the Hub (or when the Hub was not read)."""
    mark = {}
    if listing := hub_listing(api, repo_id, kind):
        sha, hub = listing
        d = against_hub(t, files, hub)
        mark = {rel: "+" for rel in d["new"]} | {rel: "~" for rel in d["changed"]}
        top = lambda names: sorted({n.split("/")[0] for n in names if "/" in n})
        count = lambda names: "; ".join(f"{k} {sum(1 for n in names if n.split('/')[0] == k)}" for k in sorted({n.split("/")[0] for n in names})) or "none"
        print(f"against {repo_id} at {sha} ({len(hub)} files): {len(d['new'])} new, {len(d['changed'])} changed, "
              f"{len(d['unchanged'])} unchanged; {len(d['not staged'])} files there are not staged and stay as they are")
        print("new top-level folders:", ", ".join(sorted(set(top(d["new"])) - set(top(hub)))) or "none")
        print("new files by top-level folder:", count(d["new"]))
        print("changed files by top-level folder:", count(d["changed"]))
        print("changed:", ", ".join(d["changed"]) or "none")
    for p in files:
        rel = p.relative_to(t).as_posix()
        print(mark.get(rel, " "), rel)
    print("dry run: nothing uploaded")


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
        no_home_paths(t)
        no_private(t)
        api.upload_folder(folder_path=t, repo_id=DATA_REPO, repo_type="dataset",
                          commit_message="Frozen datasets")
    print("pushed", DATA_REPO)


def stage_evals(t: Path, arms: list[str]) -> None:
    """Copy what `evals` uploads into t: the stage 1 arm folders, the back-translations, the stage 1 tables with
    their sub-folders (the circuit files among them) and the card."""
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


def push_evals(api: HfApi, arms: list[str], public: bool, dry_run: bool = False) -> None:
    """The stage 1 outputs. Files already in the repo are replaced only where their content changed, and nothing is
    removed; the versions a file had before stay in the repository's history."""
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        stage_evals(t, arms)
        no_home_paths(t)
        no_private(t)
        files = sorted(p for p in t.rglob("*") if p.is_file())
        print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB -> {EVAL_REPO}")
        if dry_run:
            show_dry_run(api, EVAL_REPO, "dataset", t, files)
            return
        _repo(api, EVAL_REPO, "dataset", public)
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
        no_home_paths(t)   # a run record written on the laptop holds absolute paths; write them relative before uploading
        api.upload_folder(folder_path=t, repo_id=repo_id, commit_message=f"{arm} LoRA adapter")
    print("pushed", repo_id)


ADAPTER_REPO = f"{HF_ORG}/tiny-aya-lora-stage2"
# One folder each. The seed 42 plain and SafeGrad runs on Tiny Aya Global are the two stage 1 repos.
STAGE2_ADAPTERS = ["plain-s1", "safegrad-s1", "plain-s2", "safegrad-s2", "plain-mix1", "plain-mix5", "plain-noko",
                   "plain-noko-s1", "plain-noko-s2", "plain-ayasize", "aya-plain", "aya-safegrad", "earth-plain",
                   "earth-safegrad", "plain-s1-vec", "plain-s2-vec",
                   # sections 11 and 13 of the preregistration, Nov 21 and 22
                   "aya-plain-s1", "aya-safegrad-s1", "aya-plain-s2", "aya-safegrad-s2", "earth-plain-s1", "earth-safegrad-s1",
                   "earth-plain-s2", "earth-safegrad-s2", "plain-only-en", "plain-only-ko", "plain-only-zh",
                   "aya-plain-s1-vec", "aya-plain-s2-vec", "aya-plain-vec",
                   # seeds 1 and 2 of the single-language runs and of the 519-row run (sections 13a and 13e), Nov 22
                   "plain-only-en-s1", "plain-only-ko-s1", "plain-only-zh-s1", "plain-only-en-s2", "plain-only-ko-s2", "plain-only-zh-s2",
                   "plain-ayasize-s1", "plain-ayasize-s2",
                   # the mixing runs at seeds 1 and 2 (14a), the run through AlignTune (14d) and the four runs with the
                   # learning rate or the number of epochs changed (14h), Nov 23 and 24
                   "plain-mix1-s1", "plain-mix1-s2", "plain-mix5-s1", "plain-mix5-s2", "plain-aligntune",
                   "plain-lr-half", "plain-lr-double", "plain-ep-half", "plain-ep-double"]


def _relative(x):
    """A run record with every path under the repository written relative to it."""
    if isinstance(x, dict):
        return {k: _relative(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_relative(v) for v in x]
    if isinstance(x, str) and x.startswith(f"{ROOT}/"):
        return x[len(f"{ROOT}/"):]
    return x


def stage_adapters(t: Path, arms: list[str]) -> list[Path]:
    """Copy each arm's adapter weights and config, run record and logs to t/<arm>/. Only those named files are taken:
    the tokenizer files are the base model's, and whatever else a trainer left in the adapter folder stays behind.
    Stops if anything staged names a home directory or holds a token."""
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
        if found := leaks(p):
            raise SystemExit(f"{p.relative_to(t)}: {', '.join(found)}; nothing uploaded")
    return files


def push_adapters(api: HfApi, arms: list[str] | None, public: bool, dry_run: bool) -> None:
    """The stage 2 adapters as folders of one model repo, with the card as its README. Folders that are not staged
    stay as they are. With public, the repository is made public once the files are up."""
    arms = arms or STAGE2_ADAPTERS
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        files = stage_adapters(t, arms)
        shutil.copy(CARDS / "adapters-stage2.md", t / "README.md")
        files = [t / "README.md"] + files
        print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e9:.2f} GB -> {ADAPTER_REPO}")
        print("visibility:", "the repository is made public after the upload (--public)" if public else "left as it is")
        if dry_run:
            show_dry_run(api, ADAPTER_REPO, "model", t, files)
            return
        _repo(api, ADAPTER_REPO, "model", public)
        api.upload_folder(folder_path=t, repo_id=ADAPTER_REPO,
                          commit_message=f"Stage 2 LoRA adapters: {', '.join(arms)}")
    print("pushed", ADAPTER_REPO)
    if public:   # create_repo leaves a repository that exists as it is, so the setting is changed here, with the files and the card in place
        api.update_repo_settings(ADAPTER_REPO, private=False)
        print("made public:", ADAPTER_REPO)


# Stage 2 arms. The second list was still running when the first went up; each joins an upload once it is fully judged.
STAGE2_ARMS = ["plain-s1", "safegrad-s1", "plain-s2", "safegrad-s2", "plain-mix1", "plain-mix5", "plain-noko",
               "earth-base", "earth-plain", "earth-safegrad"]
STAGE2_LATER = ["plain-s1-vec", "plain-s2-vec", "plain-noko-s1", "plain-noko-s2", "aya-plain", "aya-safegrad", "plain-ayasize",
                # sections 11 and 13 of the preregistration, Nov 21 and 22. Attack set only
                "aya-plain-s1", "aya-safegrad-s1", "aya-plain-s2", "aya-safegrad-s2", "earth-plain-s1", "earth-safegrad-s1",
                "earth-plain-s2", "earth-safegrad-s2", "plain-only-en", "plain-only-ko", "plain-only-zh",
                "aya-plain-s1-vec", "aya-plain-s2-vec", "aya-plain-vec",
                # seeds 1 and 2 of the single-language runs and of the 519-row run (sections 13a and 13e). Attack set only
                "plain-only-en-s1", "plain-only-ko-s1", "plain-only-zh-s1", "plain-only-en-s2", "plain-only-ko-s2", "plain-only-zh-s2",
                "plain-ayasize-s1", "plain-ayasize-s2",
                # the mixing runs at seeds 1 and 2 (14a), the run through AlignTune (14d) and the four runs with the
                # learning rate or the number of epochs changed (14h). All three sets
                "plain-mix1-s1", "plain-mix1-s2", "plain-mix5-s1", "plain-mix5-s2", "plain-aligntune",
                "plain-lr-half", "plain-lr-double", "plain-ep-half", "plain-ep-double"]
STAGE1_ARMS = ["base", "plain", "safegrad"]  # already in the repo; their capped answers were generated again too


def regenerated(arm: str) -> bool:
    """Every capped answer of the arm was generated again and judged, on each set the arm was evaluated on."""
    from lir import trunc

    sets = [name for name in trunc.SETS if (RUNS / "eval" / arm / f"{name}.gen.jsonl").exists()]
    return bool(sets) and (trunc.REGEN / arm).is_dir() and all(trunc.after(arm, name) is not None for name in sets)


def unfinished(arm_dir: Path, task_guard: bool = True) -> list[str]:
    """Sets of an arm whose guard file is missing or has fewer rows than its generation file. With task_guard off the
    task answers need their scores only: the guard did not read the task answers of the quantization arms."""
    rows = lambda p: sum(1 for line in p.read_text().splitlines() if line.strip())
    bad = []
    for gen in sorted(arm_dir.glob("*.gen.jsonl")):
        guard = gen.with_name(gen.name.replace(".gen.", ".guard."))
        if (task_guard or gen.name != "task.gen.jsonl") and (not guard.exists() or rows(guard) < rows(gen)):
            bad.append(gen.name.split(".")[0])
    task, scores = arm_dir / "task.gen.jsonl", arm_dir / "task.judge.jsonl"   # the task judge runs as its own step
    if task.exists() and "task" not in bad and (not scores.exists() or rows(scores) < rows(task)):
        bad.append("task scores")
    return bad


# The Hugging Face GPU jobs whose result files are published: section 12 (the control, the single-language runs, 4-bit
# loading and the unmerged arms), 14f (the same two at seeds 1 and 2), 15b (Global-MMLU-Lite on the quantization arms)
# and 16 to 16c (Granite 4.1). Not d13, a check of one training batch that no report reads, and not judge3 (section 17).
GPU_JOBS = ("control", "only-ko", "only-en", "only-zh", "nf4", "unmerged",
            "unmerged-s1", "unmerged-s2", "nf4-s1", "nf4-s2", "mmlu",
            "granite41-base", "granite41-plain-s42", "granite41-plain-s1", "granite41-plain-s2",
            "granite41-safegrad-s42", "granite41-safegrad-s1", "granite41-safegrad-s2", "granite41-regen",
            "granite41-noko-s42", "granite41-noko-s1", "granite41-noko-s2", "granite41-task")


def gpu_files(root: Path) -> list[tuple[Path, Path]]:
    """The result files of the Hugging Face GPU jobs (answers, verdicts, records and logs), each with its path in the
    outputs repo. root is runs/gpu as `python -m lir.gpu fetch` leaves it. A job folder with no job record is left
    out, and so is a job that is not in GPU_JOBS. Nothing under a folder named `adapter` is taken, whatever its
    suffix: the adapters trained in the jobs do not belong in the outputs repo, their configs included. Nothing of
    section 20 is taken, wherever it lies (see `private`)."""
    out = []
    for job in sorted(p for p in root.glob("*/*") if (p / "job.json").exists() and p.parent.name in GPU_JOBS):
        out += [(p, Path("gpu") / p.relative_to(root)) for p in sorted(job.rglob("*"))
                if p.is_file() and p.suffix in (".jsonl", ".json", ".log") and "adapter" not in p.relative_to(job).parts
                and not private(p.relative_to(root))]
    if out and (root / "revision.txt").exists():   # the revision of the staging dataset the files were read from
        out.append((root / "revision.txt", Path("gpu") / "staging_revision.txt"))
    return out


# Outputs that are not arm folders of runs/eval. Each family goes under a folder of its own in the outputs repo, with
# its arms one level down: `lir.rebuild` takes every top-level folder with a *.gen.jsonl in it for an arm of runs/eval,
# and these arms share names with those (base, plain, ...).
# Folder under runs/ -> (folder in the repo, the arms its report reads, the folder of the first pass if the answers are
# ones generated again).
SEEDED = STAGE1_ARMS + ["plain-s1", "safegrad-s1", "plain-s2", "safegrad-s2"]
MIX5 = ["plain-mix5", "plain-mix5-s1", "plain-mix5-s2"]
FAMILIES = {"morelangs": ("more-languages", SEEDED, None),                                   # 14b, six further languages
            "morelangs-trunc": ("more-languages-regenerated", SEEDED, "morelangs"),          # 14e
            "xsunsafe": ("xstest-unsafe", SEEDED + MIX5, None),                              # 14i, the unsafe prompts of XSTest
            "xsunsafe-trunc": ("xstest-unsafe-regenerated", SEEDED + MIX5, "xsunsafe"),      # 14j
            "quant": ("quantization", ["bf16", "int4", "circuit", "random-0", "random-2", "random-4"], None)}   # 15, 15a
QUANT_RECORDS = ("check.json", "gate.json", "ranks.json")   # beside the arm folders of runs/quant
ARM_FILES = (".gen.jsonl", ".guard.jsonl", ".judge.jsonl")  # what is taken from an arm folder, with quantization.json


def cut_again(first: Path, again: Path) -> bool:
    """Every answer of the first pass that reached the cap, and no other, has one verdict in the second pass. This is
    the rule of `after` in lir.trunc, lir.morelangs and lir.xsunsafe, read from the verdict files of two arm folders."""
    for guard in sorted(first.glob("*.guard.jsonl")):
        if not (again / guard.name).exists():
            return False
        new = read_jsonl(again / guard.name)
        done = [r["uid"] for r in new if r.get("safety")]
        if len(done) != len(new) or len(set(done)) != len(done) or set(done) != {r["uid"] for r in read_jsonl(guard) if r["truncated"]}:
            return False
    return True


def more_files(runs: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    """The files of the FAMILIES, each with its path in the outputs repo, and the families left out, each with what is
    missing. A family goes up whole or not at all, because its table reads every arm, and answers generated again go
    only with their first pass. Only answers, verdicts, scores and the quantization records are taken, whatever else
    is in a folder."""
    out, left, gone = [], [], set()
    for name, (hub, arms, first) in FAMILIES.items():
        files, bad = [], []
        for arm in arms:
            d = runs / name / arm
            if not any(d.glob("*.gen.jsonl")):
                bad.append(f"{arm}: no answers")
            elif why := unfinished(d, task_guard=name != "quant"):
                bad.append(f"{arm}: {', '.join(why)} not judged")
            elif first and not cut_again(runs / first / arm, d):
                bad.append(f"{arm}: not every capped answer has one new verdict")
            elif name == "quant" and not (d / "quantization.json").exists():
                bad.append(f"{arm}: no quantization.json")
            else:
                files += [(p, Path(hub) / arm / p.name) for p in sorted(d.iterdir())
                          if p.is_file() and (p.name.endswith(ARM_FILES) or p.name == "quantization.json")]
        if name == "quant":
            bad += [f"no {f}" for f in QUANT_RECORDS if not (runs / name / f).exists()]
            files += [(runs / name / f, Path(hub) / f) for f in QUANT_RECORDS]
        if first and first not in gone:   # answers generated again are read beside the first pass
            bad = ["its first pass is left out"]
        if bad:
            left.append(f"{hub} ({'; '.join(bad)})")
        else:
            out += files
            gone.add(name)
    return out, left


# Committed tables that do not go up, by the start of their file name. Section 17, one more automatic judge, is not
# part of what is published: its inputs and verdicts are under runs/judge3, which no function here reads, and its
# tables in results/lir/stage2 are all named third_judge*.
HELD_TABLES = ("third_judge",)


def stage_stage2(t: Path, arms: list[str], aya_task: bool = False) -> dict[str, list[str]]:
    """Copy what `stage2` uploads into t, laid out as in the outputs repo. Returns what went in, by kind, the
    families of `more_files` that were left out and the committed tables that were held back (HELD_TABLES)."""
    from lir import trunc

    rows = lambda p: sum(1 for line in p.read_text().splitlines() if line.strip())
    for arm in arms:
        shutil.copytree(RUNS / "eval" / arm, t / arm)
    again = [arm for arm in STAGE1_ARMS + arms if regenerated(arm)]
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
    held = []
    for rel in tracked.stdout.splitlines():
        if Path(rel).name.startswith(HELD_TABLES):
            held.append(Path(rel).relative_to(res).as_posix())
        elif Path(rel).name != "README.md":
            dest = t / "stage2" / Path(rel).relative_to(res)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, dest)
    gpu = gpu_files(RUNS / "gpu") if f"{res}/gpu.md" in tracked.stdout.splitlines() else []   # once their table is committed
    more, left = more_files(RUNS)
    for src, rel in gpu + more:
        (t / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, t / rel)
    shutil.copy(CARDS / "evals.md", t / "README.md")
    return {"again": again, "aya": aya, "gpu": sorted({rel.parts[1] for _, rel in gpu if len(rel.parts) > 2}),
            "more": sorted({rel.parts[0] for _, rel in more}), "left": left, "held": held}


def push_stage2(api: HfApi, arms: list[str] | None, dry_run: bool, aya_task: bool = False) -> None:
    """Add stage 2 outputs to the outputs repo: arm folders, re-generated answers, the NLLB re-translations, the
    result files of the GPU jobs, the families of `more_files` and the committed stage 2 tables but those of
    HELD_TABLES. The answers to the held-out Aya prompts go up only with aya_task, because each row quotes an Aya
    prompt and its reference answer.

    Nothing already in the repo is removed, and only the README card and files whose content changed are
    replaced. The repo is public, so the files are public as soon as they are uploaded. With no arms given, every
    finished stage 2 arm goes up and the ones still running are named and left out.
    """
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
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        staged = stage_stage2(t, arms, aya_task)
        no_home_paths(t)
        no_private(t)
        files = sorted(p for p in t.rglob("*") if p.is_file())
        size = sum(p.stat().st_size for p in files)
        print(f"{len(files)} files, {size / 1e6:.1f} MB -> {EVAL_REPO}")
        print("arms:", ", ".join(arms))
        print("re-generated answers:", ", ".join(staged["again"]) or "none yet", "| Aya task answers:",
              (", ".join(staged["aya"]) or "none yet") if aya_task else "left out (--aya-task adds them)")
        print("GPU jobs:", ", ".join(staged["gpu"]) or "none")
        print("in folders of their own:", ", ".join(staged["more"]) or "none")
        print("committed tables held back:", ", ".join(staged["held"]) or "none")
        if waiting or staged["left"]:
            print("not finished, left out:", ", ".join(waiting + staged["left"]))
        if dry_run:
            show_dry_run(api, EVAL_REPO, "dataset", t, files)
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
        no_home_paths(t)
        api.upload_folder(folder_path=t, repo_id=SPACE_REPO, repo_type="space", commit_message="Demo Space",
                          delete_patterns=["style.css"])
    print("pushed", SPACE_REPO)


DRY_RUNS = ("stage2", "evals", "adapters")   # the commands that have a dry run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "evals", "adapter", "adapters", "space", "stage2"])
    ap.add_argument("--arm")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--public", action="store_true",
                    help="a new repository is created public; adapters: the stage 2 adapter repository is made public after the upload")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage2, evals and adapters: list the files against the Hub and upload nothing")
    ap.add_argument("--aya-task", action="store_true",
                    help="stage2 only: also upload the answers to the held-out Aya prompts (they quote the Aya rows)")
    a = ap.parse_args()
    if a.dry_run and a.what not in DRY_RUNS:   # the flag would be ignored and the upload would run
        ap.error(f"{a.what} has no dry run; nothing uploaded")
    api = HfApi()
    if a.what == "stage2":
        push_stage2(api, a.arms, a.dry_run, a.aya_task)
    elif a.what == "data":
        push_data(api, a.public)
    elif a.what == "evals":
        push_evals(api, a.arms or ["base", "plain", "safegrad"], a.public, a.dry_run)
    elif a.what == "space":
        push_space(api, a.public)
    elif a.what == "adapters":
        push_adapters(api, a.arms, a.public, a.dry_run)
    else:
        push_adapter(api, a.arm, a.public)


if __name__ == "__main__":
    main()
