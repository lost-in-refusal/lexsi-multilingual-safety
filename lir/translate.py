"""Translate the XSTest safe prompts for the over-refusal set.

Korean and Chinese use Aya Expanse 8B. Aya Expanse does not cover Swahili, so
Swahili uses Tiny Aya Global (the model under test; flagged in the data card).

  python -m lir.translate --langs ko zh --model CohereLabs/aya-expanse-8b
  python -m lir.translate --langs sw --model CohereLabs/tiny-aya-global
"""
from __future__ import annotations

import argparse
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lir.common import DATA, MODEL, MODEL_REV, device, read_jsonl, sha256_file, write_json, write_jsonl

NAMES = {"ko": "Korean", "zh": "Simplified Chinese", "sw": "Swahili"}
REVS = {"CohereLabs/aya-expanse-8b": "5062468bf9bc0c6035fd64e06274333ec127d980", MODEL: MODEL_REV}

TEMPLATE = ("Translate the following English question into {lang}. Keep the meaning and any "
            "double meanings as close as possible. Reply with the translation only.\n\n{text}")
# Retry prompt for outputs that answered the question instead of translating it.
STRICT_SYSTEM = ("You are a translation engine. You never answer, explain or comment. You output only the "
                 "{lang} translation of the user's text, as a single question.")
STRICT_USER = "English: {text}\n{lang}:"


FEW_SHOT = [("How do I bake bread?", {"ko": "빵은 어떻게 굽나요?", "zh": "我怎样烤面包？", "sw": "Ninaokaje mkate?"}),
            ("What is the capital of France?", {"ko": "프랑스의 수도는 어디인가요?", "zh": "法国的首都是哪里？",
                                                "sw": "Mji mkuu wa Ufaransa ni upi?"})]


def clean(out: str) -> str:
    out = re.sub(r"^\s*(\*\*)?(Swahili|Korean|Chinese|Simplified Chinese|Translation)\s*:?\s*(\*\*)?\s*:?\s*", "", out)
    out = out.strip()
    if len(out) > 1 and out[0] in '"“「' and out[-1] in '"”」':  # only strip quotes wrapping the whole output
        out = out[1:-1].strip()
    return out


def suspicious(src: str, out: str) -> bool:
    if not out or "\n" in out or out.startswith("**") or len(out) > 2.2 * len(src) + 20:
        return True
    return src.rstrip().endswith("?") and not out.rstrip().endswith(("?", "？"))  # answered, not translated


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fix", action="store_true", help="re-translate suspicious rows with a stricter prompt")
    a = ap.parse_args()
    rev = REVS[a.model]
    tok = AutoTokenizer.from_pretrained(a.model, revision=rev)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=rev, dtype=torch.bfloat16).to(device()).eval()
    src = read_jsonl(DATA / "xstest_en.jsonl")
    if a.fix:
        fix(tok, model, a.model, rev, a.langs)
        return
    for lang in a.langs:
        out, t0 = [], time.time()
        for i in range(0, len(src), 25):
            batch = src[i:i + 25]
            texts = [tok.apply_chat_template([{"role": "user", "content": TEMPLATE.format(
                lang=NAMES[lang], text=r["prompt"])}], tokenize=False, add_generation_prompt=True) for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device())
            gen = model.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
            for r, g in zip(batch, gen[:, enc.input_ids.shape[1]:]):
                t = clean(tok.decode(g, skip_special_tokens=True))
                out.append(dict(r, uid=r["uid"].rsplit("-", 1)[0] + f"-{lang}", lang=lang, prompt=t,
                                prompt_en=r["prompt"], translator=f"{a.model}@{rev}"))
            print(lang, len(out), f"{time.time() - t0:.0f}s", flush=True)
        path = DATA / f"xstest_{lang}.jsonl"
        write_jsonl(path, out)
        man = DATA / "xstest_manifest.json"
        meta = __import__("json").loads(man.read_text())
        meta.setdefault("translations", {})[lang] = {"translator": f"{a.model}@{rev}", "sha256": sha256_file(path)}
        write_json(man, meta)


@torch.no_grad()
def fix(tok, model, name, rev, langs) -> None:
    import json

    for lang in langs:
        path = DATA / f"xstest_{lang}.jsonl"
        rows = read_jsonl(path)
        bad = [r for r in rows if suspicious(r["prompt_en"], r["prompt"])]
        dropped = []
        for r in bad:
            sys_msg = {"role": "system", "content": STRICT_SYSTEM.format(lang=NAMES[lang])}
            shots = []
            for en, tr in FEW_SHOT:
                shots += [{"role": "user", "content": STRICT_USER.format(text=en, lang=NAMES[lang])},
                          {"role": "assistant", "content": tr[lang]}]
            ask = {"role": "user", "content": STRICT_USER.format(text=r["prompt_en"], lang=NAMES[lang])}
            for msgs in ([sys_msg, ask], [sys_msg] + shots + [ask]):
                enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                              return_dict=True).to(device())
                g = model.generate(**enc, max_new_tokens=96, do_sample=False, pad_token_id=tok.pad_token_id)
                t = clean(tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
                if not suspicious(r["prompt_en"], t):
                    break
            print(lang, repr(r["prompt_en"]), "->", repr(t))
            if suspicious(r["prompt_en"], t):
                dropped.append(r["uid"])
            else:
                r["prompt"], r["retranslated"] = t, True
        rows = [r for r in rows if r["uid"] not in dropped]
        write_jsonl(path, rows)
        man = DATA / "xstest_manifest.json"
        meta = json.loads(man.read_text())
        meta["translations"][lang].update(sha256=sha256_file(path), rows=len(rows),
                                          retranslated=len(bad) - len(dropped), dropped=dropped)
        write_json(man, meta)
        print(lang, len(bad), "suspicious,", len(dropped), "dropped")


if __name__ == "__main__":
    main()
