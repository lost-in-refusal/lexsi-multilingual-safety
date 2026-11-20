"""Swahili re-translation with a translator that is not the model under test.

  python -m lir.nllb backtranslate --arms base plain safegrad   # translate, re-judge, corrected ASR
  python -m lir.nllb labels --langs sw ko zh                    # human-label rows in English
  python -m lir.nllb arms --arms base plain --langs ko          # every attack answer of an arm in English

Stage 1 translated Swahili answers with base Tiny Aya (lir.swcheck). This does the
same check with NLLB-200 3.3B: every Swahili attack answer the guard called Safe
without a refusal is translated to English and judged again with MultiJail's
English prompt. Outputs go to their own files; the stage 1 files are not touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from lir.common import DATA, ROOT, RUNS, device, empty_cache, read_jsonl, write_json, write_jsonl

NLLB = "facebook/nllb-200-3.3B"
NLLB_REV = "1a07f7d195896b2114afcb79b7b57ab512e7b43e"
CODES = {"sw": "swh_Latn", "ko": "kor_Hang", "zh": "zho_Hans", "en": "eng_Latn"}
OUT = ROOT / "results" / "lir" / "stage2"
LABELS = ROOT / "results" / "lir" / "labels"


def chunks(text: str, limit: int = 300) -> list[str]:
    """Split an answer into pieces the translator handles well: by line, then by sentence."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cur = ""
        for sent in re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", line):  # Chinese has no space after a full stop
            if not sent:
                continue
            if cur and len(cur) + len(sent) + 1 > limit:
                out.append(cur)
                cur = sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur:
            out.append(cur)
    return out


def load():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(NLLB, revision=NLLB_REV)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB, revision=NLLB_REV, dtype=torch.float32).to(device()).eval()
    return tok, model


def translate(tok, model, texts: list[str], src: str, batch: int = 24, token_budget: int = 3000) -> list[str]:
    """Translate whole answers to English, chunk by chunk, and join the chunks back in order."""
    import torch

    tok.src_lang = CODES[src]
    pieces = [(i, c) for i, t in enumerate(texts) for c in chunks(t, 150 if src == "zh" else 300)]
    n_tok = [len(tok(c, truncation=True, max_length=512).input_ids) for _, c in pieces]
    order = sorted(range(len(pieces)), key=lambda k: -n_tok[k])
    done: dict[int, str] = {}
    with torch.no_grad():
        s = 0
        while s < len(order):
            # beam search keeps a cache per beam that grows with every token, so long pieces go in small batches
            longest = n_tok[order[s]]
            idx = order[s:s + max(1, min(batch, token_budget // longest))]
            enc = tok([pieces[k][1] for k in idx], return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(device())
            gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(CODES["en"]),
                                 max_new_tokens=min(400, 3 * longest + 50), num_beams=4)
            for k, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                done[k] = text.strip()
            del enc, gen
            empty_cache()
            s += len(idx)
            print(f"nllb {s}/{len(order)}", flush=True)
    out = [[] for _ in texts]
    for k, (i, _) in enumerate(pieces):
        out[i].append(done[k])
    return ["\n".join(p) for p in out]


def backtranslate(arms: list[str]) -> dict:
    from lir import guard
    from lir.metrics import boot_delta, mcnemar_p, wilson

    en_prompt = {r["family"]: r["prompt"] for r in read_jsonl(DATA / "multijail_attack.jsonl") if r["lang"] == "en"}
    out_dir = RUNS / "swcheck"
    passed = {arm: [r for r in read_jsonl(RUNS / "eval" / arm / "attack.guard.jsonl")
                    if r["lang"] == "sw" and r["safety"] == "Safe" and r["refusal"] != "Yes"] for arm in arms}
    todo = {arm: rows for arm, rows in passed.items() if not (out_dir / f"{arm}.nllb.bt.jsonl").exists()}
    if todo:
        tok, model = load()
        for arm, rows in todo.items():
            en = translate(tok, model, [r["response"] for r in rows], "sw")
            write_jsonl(out_dir / f"{arm}.nllb.bt.jsonl",
                        [{"uid": r["uid"], "family": r["family"], "lang": "sw", "response": t,
                          "translator": f"{NLLB}@{NLLB_REV}"} for r, t in zip(rows, en)])
        del model
        empty_cache()
    gtok, gmodel = guard.load()
    lab, res = {}, {}
    for arm in arms:
        bt = read_jsonl(out_dir / f"{arm}.nllb.bt.jsonl")
        pairs = [{"uid": r["uid"], "prompt": en_prompt[r["family"]], "response": r["response"]} for r in bt]
        guard.judge(gtok, gmodel, pairs, out_dir / f"{arm}.nllb.bt.guard.jsonl")
        again = {r["uid"]: r for r in read_jsonl(out_dir / f"{arm}.nllb.bt.guard.jsonl")}
        g = {r["uid"]: r for r in read_jsonl(RUNS / "eval" / arm / "attack.guard.jsonl") if r["lang"] == "sw"}
        lab[arm] = {u: float(r["safety"] != "Safe" or (u in again and again[u]["safety"] != "Safe")) for u, r in g.items()}
        k = sum(lab[arm].values())
        res[arm] = {"n_safe_nonrefusal": len(again),
                    "unsafe_after_translation_strict": sum(r["safety"] != "Safe" for r in again.values()),
                    "asr_strict_corrected": k / len(g), "ci95": wilson(k, len(g)), "n": len(g)}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            pairs = [(lab[a][u], lab[b][u]) for u in sorted(lab[a])]
            res[f"{b}_vs_{a}"] = {"delta": sum(y - x for x, y in pairs) / len(pairs), "ci95": boot_delta(pairs),
                                  "mcnemar_p": mcnemar_p(pairs)}
    res["translator"] = f"{NLLB}@{NLLB_REV}"
    write_json(OUT / "swahili_nllb.json", res)
    return res


def labels(langs: list[str]) -> None:
    """English versions of the human-label rows in these languages, for the second judge. One file per language."""
    key = {k["item"]: k for k in json.loads((LABELS / "labels_key.json").read_text())}
    en_prompt = {r["family"]: r["prompt"] for r in read_jsonl(DATA / "multijail_attack.jsonl") if r["lang"] == "en"}
    family = {r["uid"]: r["family"] for r in read_jsonl(DATA / "multijail_attack.jsonl")}
    todo = [l for l in langs if not (LABELS / f"labels_{l}_nllb.jsonl").exists()]
    if not todo:
        print("label rows already translated for", ", ".join(langs))
        return
    tok, model = load()
    for lang in todo:
        rows = [r for r in csv.DictReader(open(LABELS / "labels_blank.csv", encoding="utf-8")) if r["lang"] == lang]
        en = translate(tok, model, [r["response"] for r in rows], lang)
        write_jsonl(LABELS / f"labels_{lang}_nllb.jsonl",
                    [{"uid": r["item"], "item": r["item"], "lang": lang,
                      "prompt": en_prompt[family[key[r["item"]]["uid"]]], "response": t,
                      "translator": f"{NLLB}@{NLLB_REV}"} for r, t in zip(rows, en)])
        print(len(rows), lang, "label rows ->", LABELS / f"labels_{lang}_nllb.jsonl")


def arms(names: list[str], langs: list[str]) -> None:
    """Every attack answer of each arm in these languages, in English: runs/nllb/<arm>.<lang>.jsonl.

    Swahili answers already translated for the re-judging check are reused, so both checks read the same text."""
    out_dir = RUNS / "nllb"
    tok = model = None
    for lang in langs:
        for arm in names:
            path = out_dir / f"{arm}.{lang}.jsonl"
            rows = [r for r in read_jsonl(RUNS / "eval" / arm / "attack.gen.jsonl") if r["lang"] == lang]
            if path.exists() and {r["uid"] for r in read_jsonl(path)} >= {r["uid"] for r in rows}:
                print(f"{arm} {lang}: already translated")
                continue
            old = RUNS / "swcheck" / f"{arm}.nllb.bt.jsonl"
            have = {r["uid"]: r["response"] for r in read_jsonl(old)} if lang == "sw" and old.exists() else {}
            new = [r for r in rows if r["uid"] not in have]
            if tok is None:
                tok, model = load()
            have.update(zip((r["uid"] for r in new), translate(tok, model, [r["response"] for r in new], lang)))
            write_jsonl(path, [{"uid": r["uid"], "family": r["family"], "lang": lang, "response": have[r["uid"]],
                                "translator": f"{NLLB}@{NLLB_REV}"} for r in rows])
            print(f"{arm} {lang}: {len(new)} answers translated, {len(rows) - len(new)} reused -> {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["backtranslate", "labels", "arms"])
    ap.add_argument("--arms", nargs="+", default=["base", "plain", "safegrad"])
    ap.add_argument("--langs", nargs="+", default=["sw"], choices=["sw", "ko", "zh"])
    a = ap.parse_args()
    if a.cmd == "labels":
        labels(a.langs)
    elif a.cmd == "arms":
        arms(a.arms, a.langs)
    else:
        print(json.dumps(backtranslate(a.arms), indent=2))


if __name__ == "__main__":
    main()
