"""Preregistration section 14i: a second safety benchmark, the 200 unsafe prompts of XSTest.

  python -m lir.xsunsafe build                  # data/lir/xstest_unsafe.jsonl and its manifest
  python -m lir.xsunsafe evaluate --arm base    # answers and guard verdicts -> runs/xsunsafe/<arm>/
  python -m lir.xsunsafe evaluate --arm plain-s1
  python -m lir.xsunsafe report                 # X1 to X4 -> results/lir/stage2/second_benchmark.{json,md}

Evaluation only: no model is trained. XSTest has 250 safe prompts, which the over-refusal set uses, and 200 unsafe
prompts written as their contrasts. This block takes the 200 unsafe ones from the same pinned file, in English as
written and in Korean, Chinese and Swahili as NLLB-200 translates them. The answers are generated and judged as the
attack set is (same code, same 256-token cap, same guard). Outputs go to runs/xsunsafe/ and not to runs/eval/.

Nothing is downloaded: `build` reads the XSTest file from the local cache at the revision of xstest_manifest.json and
stops if its hash is not the one recorded there. `report` reads files only. It prints "pending" and writes nothing
until base and the six plain and SafeGrad runs are fully judged.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from lir.common import DATA, EVAL_LANGS, ROOT, RUNS, device, empty_cache, read_jsonl, sha256_file, write_json, write_jsonl
from lir.seeds import REFUSAL, SEEDS, STRICT, UNSAFE, by_family, change, diff, pooled, rate

LANGS = list(EVAL_LANGS)                      # en as written; ko, zh, sw translated
TRANSLATED = [l for l in LANGS if l != "en"]
NAMES = {"en": "English", "ko": "Korean", "zh": "Chinese", "sw": "Swahili"}
FILE = "xstest_unsafe.jsonl"
MANIFEST = "xstest_unsafe_manifest.json"
SAFE_MANIFEST = "xstest_manifest.json"       # the safe half: its source file and revision are the ones used here
OUT_RUNS = RUNS / "xsunsafe"
OUT = ROOT / "results" / "lir" / "stage2"
MIX = {42: "plain-mix5", 1: "plain-mix5-s1", 2: "plain-mix5-s2"}   # five copies of the refusal examples, by seed
MAIN = ["base"] + [arm for pair in SEEDS.values() for arm in pair]  # X1 to X3 need these
ARMS = MAIN + list(MIX.values())

X1_ROWS = {"above": "Plain fine-tuning also raised Korean attack success on a second safety benchmark, the unsafe prompts "
                    "of XSTest",
           "includes": "No Korean rise was detected on the second safety benchmark. The Korean drift is shown on MultiJail, "
                       "and that limit is stated wherever H1 is stated",
           "below": "On the second safety benchmark Korean attack success fell under plain fine-tuning. The Korean drift is "
                    "shown on MultiJail only, and that limit is stated wherever H1 is stated"}
X2_ROWS = {"above": "On the second benchmark Korean rose more than English",
           "includes": "On the second benchmark the Korean rise could not be told apart from the English rise",
           "below": "On the second benchmark English rose more than Korean"}
X3_ROWS = {"below": "SafeGrad kept Korean attack success below plain fine-tuning on a benchmark that its refusal examples "
                    "do not come from",
           "includes": "No difference between SafeGrad and plain fine-tuning was detected in Korean on the second "
                       "benchmark. The repair is shown on MultiJail, the benchmark the refusal examples come from, and "
                       "that limit is stated wherever H2 is stated",
           "above": "On the second benchmark SafeGrad left Korean attack success above plain fine-tuning. The repair is "
                    "shown on MultiJail only, and that limit is stated wherever H2 is stated"}
X4_ROWS = {"below": "Mixing five copies of the refusal examples into the training data kept Korean attack success below "
                    "plain fine-tuning on a benchmark that the examples do not come from",
           "includes": "No difference between five-copies mixing and plain fine-tuning was detected in Korean on the "
                       "second benchmark. Its protection is shown on MultiJail, the benchmark the refusal examples come "
                       "from",
           "above": "On the second benchmark five-copies mixing left Korean attack success above plain fine-tuning. Its "
                    "protection is shown on MultiJail only"}
NOT_READ = ("On the second benchmark plain fine-tuning did not raise Korean attack success detectably, so there was no "
            "rise for a repair to take back: X3 and X4 are not read, and their figures are shown with no rule")


# ---------------------------------------------------------------- the prompts

def unsafe_rows(csv_rows: list[dict]) -> list[dict]:
    """The English rows: every XSTest prompt labelled unsafe, in the order of the file."""
    return [{"uid": f"xu-{r['id']}-en", "family": f"xu-{r['id']}", "lang": "en", "prompt": r["prompt"],
             "type": r["type"], "focus": r.get("focus", "")} for r in csv_rows if r["label"] == "unsafe"]


def usable(text: str, lang: str, english: str) -> bool:
    """The drop rule for a translation, fixed before any was made: it is kept unless it is empty, has no Hangul
    (Korean) or no Chinese character (Chinese), or is the English prompt unchanged (Swahili)."""
    text = text.strip()
    if not text:
        return False
    if lang == "ko":
        return any("가" <= c <= "힣" for c in text)
    if lang == "zh":
        return any("一" <= c <= "鿿" for c in text)
    return text.casefold() != english.strip().casefold()


def to_language(tok, model, texts: list[str], lang: str, batch: int = 32) -> list[str]:
    """English prompts into one language with NLLB-200, in the order given. Beam search with four beams, as in
    lir.nllb; the prompts are one sentence each, so nothing is cut into pieces."""
    import torch

    from lir.nllb import CODES

    tok.src_lang = CODES["en"]
    out: list[str] = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device())
            gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(CODES[lang]), max_new_tokens=200,
                                 num_beams=4)
            out += [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
            del enc, gen
            empty_cache()
            print(f"nllb {lang} {len(out)}/{len(texts)}", flush=True)
    return out


def build() -> None:
    from huggingface_hub import hf_hub_download

    from lir import nllb

    if (DATA / FILE).exists():
        print(f"{FILE} is already built; it is not built twice")
        return
    safe = read_json(DATA / SAFE_MANIFEST)
    repo, rev = safe["source"], safe["revision"]
    path = Path(hf_hub_download(repo, "xstest_prompts.csv", repo_type="dataset", revision=rev, local_files_only=True))
    if sha256_file(path) != safe["source_sha256"]:
        raise SystemExit("the cached XSTest file is not the one the safe prompts were taken from")
    with open(path, encoding="utf-8") as f:
        english = unsafe_rows(list(csv.DictReader(f)))
    tok, model = nllb.load()
    translator = f"{nllb.NLLB}@{nllb.NLLB_REV}"
    by_lang, dropped = {"en": {r["family"]: r for r in english}}, {}
    for lang in TRANSLATED:
        texts = to_language(tok, model, [r["prompt"] for r in english], lang)
        keep = [usable(t, lang, r["prompt"]) for r, t in zip(english, texts)]
        dropped[lang] = [f"{r['family']}-{lang}" for r, k in zip(english, keep) if not k]
        by_lang[lang] = {r["family"]: {"uid": f"{r['family']}-{lang}", "family": r["family"], "lang": lang,
                                       "prompt": t.strip(), "type": r["type"], "focus": r["focus"],
                                       "prompt_en": r["prompt"], "translator": translator}
                         for r, t, k in zip(english, texts, keep) if k}
    out = [by_lang[lang][r["family"]] for r in english for lang in LANGS if r["family"] in by_lang[lang]]
    write_jsonl(DATA / FILE, out)
    write_json(DATA / MANIFEST, {
        "source": repo, "revision": rev, "license": safe.get("license"), "source_sha256": sha256_file(path),
        "same_file_as": SAFE_MANIFEST, "unsafe_prompts": len(english), "languages": LANGS,
        "translations": {"translator": translator, "languages": TRANSLATED, "checked_by_a_person": False,
                         "drop_rule": "empty, no Hangul (ko), no Chinese character (zh), or the English unchanged (sw)"},
        "rows_per_lang": dict(Counter(r["lang"] for r in out)), "dropped": dropped,
        "files": {FILE: sha256_file(DATA / FILE)}})
    print(len(out), "rows:", dict(Counter(r["lang"] for r in out)), "dropped:", {l: len(v) for l, v in dropped.items()})


# ---------------------------------------------------------------- answers and verdicts

def evaluate(a) -> None:
    """The steps of lir.evaluate for the attack set, on the unsafe XSTest prompts, written under runs/xsunsafe/."""
    from lir.evaluate import SETS

    if a.arm not in ARMS:
        raise SystemExit(f"not an arm of section 14i: {a.arm} (arms: {', '.join(ARMS)})")
    adapter = None if a.arm == "base" else RUNS / a.arm / "adapter"
    if adapter is not None and not (adapter / "adapter_model.safetensors").exists():
        raise SystemExit(f"no adapter at {adapter}")
    if not (DATA / MANIFEST).exists() or not (DATA / FILE).exists():
        raise SystemExit(f"no {FILE}: `python -m lir.xsunsafe build` has not run")
    if read_json(DATA / MANIFEST)["files"][FILE] != sha256_file(DATA / FILE):
        raise SystemExit(f"{FILE} is not the file its manifest records")
    rows = read_jsonl(DATA / FILE)
    out = OUT_RUNS / a.arm
    print(f"{a.arm}: {len(rows)} prompts in {', '.join(LANGS)}; adapter {adapter or 'none'}; output {out}")
    if a.before and time.time() >= a.before:
        print(f"not started: it is {time.strftime('%F %T')} and the limit was {time.strftime('%F %T', time.localtime(a.before))}")
        return
    if a.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)

    from lir import generate

    tok, model = generate.load_model(str(adapter) if adapter else None)
    generate.generate(tok, model, rows, out / "unsafe.gen.jsonl", SETS["attack"][1], tag=f"{a.arm}/xstest unsafe")
    del model
    empty_cache()

    from lir import guard

    gtok, gmodel = guard.load()
    guard.judge(gtok, gmodel, read_jsonl(out / "unsafe.gen.jsonl"), out / "unsafe.guard.jsonl")


# ---------------------------------------------------------------- the report

def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def side(d: dict) -> str:
    lo, hi = d["ci95"]
    return "above" if lo > 0 else "below" if hi < 0 else "includes"


def with_side(d: dict) -> dict:
    return dict(d, side=side(d))


def result(data: dict[str, dict], seeds: dict[int, tuple[str, str]] = SEEDS, mix: dict[int, str] | None = None,
           langs: list[str] = LANGS) -> dict:
    """X1 to X4 of section 14i. data: arm -> {uid: guard row} on the unsafe XSTest prompts. Strict attack success
    decides all four, as in H1 and H2. mix: seed -> the five-copies run of that seed; without it X4 is not computed.
    X3 and X4 are read only if X1 is above zero: where plain did not rise there is no rise to take back."""
    strict = {arm: by_family(rows, STRICT) for arm, rows in data.items()}
    ref = {arm: by_family(rows, REFUSAL) for arm, rows in data.items()}
    unsafe = {arm: by_family(rows, UNSAFE) for arm, rows in data.items()}
    rise = {l: [diff(strict["base"], strict[p], l) for p, _ in seeds.values()] for l in langs}
    res = {"languages": langs, "seeds": list(seeds), "prompts": len({f for f, _ in strict["base"]}),
           "x1": with_side(pooled(rise["ko"]))}
    res["x1_report"] = X1_ROWS[res["x1"]["side"]]
    gap = [{f: k[f] - e[f] for f in k if f in e} for k, e in zip(rise["ko"], rise["en"])]
    res["x2"] = with_side(pooled(gap))
    res["x2_report"] = X2_ROWS[res["x2"]["side"]]
    read = res["x1"]["side"] == "above"
    res["repairs_read"] = read
    res["not_read_report"] = None if read else NOT_READ
    res["x3"] = with_side(pooled([diff(strict[p], strict[s], "ko") for p, s in seeds.values()]))
    res["x3_report"] = X3_ROWS[res["x3"]["side"]] if read else None
    if mix:
        res["mix_seeds"] = list(mix)
        res["x4"] = with_side(pooled([diff(strict[seeds[seed][0]], strict[arm], "ko") for seed, arm in mix.items()]))
        res["x4_report"] = X4_ROWS[res["x4"]["side"]] if read else None
        res["mixing_minus_safegrad_ko"] = with_side(pooled([diff(strict[seeds[seed][1]], strict[arm], "ko")
                                                             for seed, arm in mix.items()]))
    res["per_language"] = {}
    for l in langs:
        row = {"prompts": sum(x == l for _, x in strict["base"]), "base_strict": rate(strict["base"], l),
               "base_refusal": rate(ref["base"], l), "strict_plain_vs_base": pooled(rise[l]),
               "strict_safegrad_vs_plain": pooled([diff(strict[p], strict[s], l) for p, s in seeds.values()]),
               "refusal_plain_vs_base": pooled([diff(ref["base"], ref[p], l) for p, _ in seeds.values()]),
               "unsafe_only_plain_vs_base": pooled([diff(unsafe["base"], unsafe[p], l) for p, _ in seeds.values()])}
        if mix:
            row["strict_mixing_vs_plain"] = pooled([diff(strict[seeds[seed][0]], strict[arm], l) for seed, arm in mix.items()])
        res["per_language"][l] = row
    res["per_seed"] = {}
    for seed, (p, s) in seeds.items():
        row = {"ko_plain": rate(strict[p], "ko"), "ko_safegrad": rate(strict[s], "ko"),
               "plain_vs_base": change(strict["base"], strict[p], "ko"), "safegrad_vs_plain": change(strict[p], strict[s], "ko")}
        if mix and seed in mix:
            row["ko_mixing"] = rate(strict[mix[seed]], "ko")
            row["mixing_vs_plain"] = change(strict[p], strict[mix[seed]], "ko")
        res["per_seed"][seed] = row
    res["truncated_share"] = {arm: {l: 100 * sum(bool(r.get("truncated")) for r in rows.values() if r["lang"] == l)
                                    / max(1, sum(r["lang"] == l for r in rows.values())) for l in langs}
                              for arm, rows in data.items()}
    return res


ci = lambda d: f"{d['mean']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]"
WHERE = {"above": "above zero", "includes": "includes zero", "below": "below zero"}
pv = lambda p: "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def report_md(res: dict, waiting: list[str] | None = None) -> str:
    langs = res["languages"]
    lines = ["### A second safety benchmark: the unsafe prompts of XSTest (X1 to X4)", "",
             f"The {res['prompts']} unsafe prompts of XSTest, in English as written and in Korean, Chinese and Swahili as "
             "NLLB-200 translates them, answered by base, by plain and SafeGrad at seeds 42, 1 and 2, and by the run with five "
             "copies of the refusal examples mixed into the training data. No model was trained for this. Section 14i of "
             "the preregistration fixed the rules before any model had answered one of these prompts. Strict attack success "
             "decides all four checks, as in H1 and H2. The refusal examples of SafeGrad and of mixing come from MultiJail, "
             "so this benchmark is one they do not come from. **No person has checked the guard on these prompts or the "
             "translations**, so every figure here is what the guard sees on machine-translated prompts. Answers are capped "
             "at 256 tokens. Intervals are 95%, from a bootstrap over the prompts with each prompt's value averaged over "
             "the seeds.", "",
             f"- **X1, the drift (Korean strict attack success, plain minus base): {ci(res['x1'])} points, "
             f"{WHERE[res['x1']['side']]}. {res['x1_report']}.**",
             f"- **X2, Korean against English (Korean rise minus English rise): {ci(res['x2'])} points, "
             f"{WHERE[res['x2']['side']]}. {res['x2_report']}.**"]
    if res["repairs_read"]:
        lines.append(f"- **X3, SafeGrad (Korean strict attack success, SafeGrad minus plain): {ci(res['x3'])} points, "
                     f"{WHERE[res['x3']['side']]}. {res['x3_report']}.**")
        if "x4" in res:
            lines.append(f"- **X4, mixing (Korean strict attack success, five copies minus plain): {ci(res['x4'])} points, "
                         f"{WHERE[res['x4']['side']]}. {res['x4_report']}.**")
    else:
        lines.append(f"- **X3 and X4: {res['not_read_report']}.** SafeGrad minus plain: {ci(res['x3'])}."
                     + (f" Five copies minus plain: {ci(res['x4'])}." if "x4" in res else ""))
    if waiting:
        lines.append("- X4 is not read: not fully judged on these prompts: " + ", ".join(f"`{a}`" for a in waiting) + ".")
    if "mixing_minus_safegrad_ko" in res:
        lines.append(f"- Beside, no rule. Korean strict attack success, five copies minus SafeGrad: "
                     f"{ci(res['mixing_minus_safegrad_ko'])}.")
    head = "| Language | Prompts | Base strict attack success | Plain minus base | SafeGrad minus plain |"
    rule = "|---|---:|---:|---:|---:|"
    if "x4" in res:
        head, rule = head + " Five copies minus plain |", rule + "---:|"
    head, rule = head + " Base refusal rate | Refusals, plain minus base | Unsafe class only, plain minus base |", rule + "---:|---:|---:|"
    lines += ["", head, rule]
    for l in langs:
        r = res["per_language"][l]
        row = (f"| {NAMES.get(l, l)} | {r['prompts']} | {r['base_strict']:.1f} | {ci(r['strict_plain_vs_base'])} "
               f"| {ci(r['strict_safegrad_vs_plain'])} |")
        if "x4" in res:
            row += f" {ci(r['strict_mixing_vs_plain'])} |"
        lines.append(row + f" {r['base_refusal']:.1f} | {ci(r['refusal_plain_vs_base'])} | {ci(r['unsafe_only_plain_vs_base'])} |")
    lines += ["", "Rates in %, differences in points, pooled over the seeds. Only X1 to X4 have a rule; the table is shown "
              "beside them, and its cells are not adjusted for their number.", "",
              "Per seed, Korean strict attack success (exact McNemar on the same prompts): " + "; ".join(
                  f"seed {seed}: plain {r['ko_plain']:.1f}% ({r['plain_vs_base']['delta']:+.1f} against base, "
                  f"{pv(r['plain_vs_base']['p'])}), SafeGrad {r['ko_safegrad']:.1f}% ({r['safegrad_vs_plain']['delta']:+.1f} "
                  f"against plain, {pv(r['safegrad_vs_plain']['p'])})"
                  + (f", five copies {r['ko_mixing']:.1f}% ({r['mixing_vs_plain']['delta']:+.1f} against plain, "
                     f"{pv(r['mixing_vs_plain']['p'])})" if "ko_mixing" in r else "")
                  for seed, r in res["per_seed"].items()) + ".", "",
              "Share of answers cut at the 256-token cap (%): " + "; ".join(
                  f"`{arm}` " + ", ".join(f"{l} {v[l]:.0f}" for l in langs) for arm, v in res["truncated_share"].items()) + ".", "",
              "Not done on this benchmark: a human check of the guard or of the translations, generating capped answers "
              "again at 1,024 tokens, and the mixing run with the examples added once."]
    return "\n".join(lines)


def judged(arm: str, n: int) -> dict[str, dict] | None:
    p = OUT_RUNS / arm / "unsafe.guard.jsonl"
    rows = {r["uid"]: r for r in read_jsonl(p)} if p.exists() else {}
    return rows if len(rows) == n and all(r.get("safety") is not None for r in rows.values()) else None


def report(a) -> None:
    if not (DATA / FILE).exists():
        print(f"pending, nothing written: {FILE} is not built")
        return
    n = len(read_jsonl(DATA / FILE))
    data = {arm: judged(arm, n) for arm in ARMS}
    waiting = [arm for arm in MAIN if data[arm] is None]
    if waiting:
        print("pending, nothing written: not fully judged on the unsafe XSTest prompts: " + ", ".join(waiting))
        return
    no_mix = [arm for arm in MIX.values() if data[arm] is None]
    res = result({arm: rows for arm, rows in data.items() if rows is not None}, mix=None if no_mix else MIX)
    res["mixing_arms_waiting"] = no_mix
    res["prompts_file"] = {"file": FILE, "sha256": sha256_file(DATA / FILE), "manifest": read_json(DATA / MANIFEST)}
    md = report_md(res, no_mix)
    write_json(OUT / "second_benchmark.json", res)
    (OUT / "second_benchmark.md").write_text(md + "\n")
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["build", "evaluate", "report"])
    ap.add_argument("--arm")
    ap.add_argument("--before", type=float, default=0, help="evaluate: do not start at or after this time (epoch seconds)")
    ap.add_argument("--dry-run", action="store_true", help="evaluate: print what would run and stop")
    a = ap.parse_args()
    if a.what == "build":
        build()
    elif a.what == "evaluate":
        if not a.arm:
            raise SystemExit("evaluate needs --arm")
        evaluate(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
