"""Task-quality judge: Qwen3-14B (pinned, thinking off, greedy).

Scores each answer 1-7 on accuracy, instruction following, coherence and
fluency (Tiny Aya report rubric) and averages them. The Bactrian-X answer is
shown as a reference, not as the only acceptable answer.

  python -m lir.judge --input gens.jsonl --output judged.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lir.common import JUDGE, device, read_jsonl

JUDGE_REV = "40c069824f4251a91eefaf281ebe4c544efd3e18"
LANG = {"en": "English", "ko": "Korean", "zh": "Chinese", "sw": "Swahili"}
DIMS = ["accuracy", "instruction_following", "coherence", "fluency"]

PROMPT = """You are grading an AI assistant's answer. The user wrote in {lang}; the answer should be in {lang}.

[User request]
{instruction}

[Reference answer, for guidance only; other correct answers are fine]
{reference}

[Assistant answer to grade]
{response}

Score the assistant answer from 1 (very poor) to 7 (excellent) on each criterion:
- accuracy: is the content correct and does it actually solve the request?
- instruction_following: does it do exactly what was asked, in the right language and format?
- coherence: is it logically organised and consistent?
- fluency: is the {lang} natural and grammatical?

Reply with only a JSON object like {{"accuracy": 5, "instruction_following": 6, "coherence": 6, "fluency": 7}}."""


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        scores = {k: int(d[k]) for k in DIMS}
        if all(1 <= v <= 7 for v in scores.values()):
            return scores
    except (ValueError, KeyError, TypeError):
        return None
    return None


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="generations with instruction/reference/response")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    out_path = Path(a.output)
    rows = read_jsonl(Path(a.input))
    done = {r["uid"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    rows = [r for r in rows if r["uid"] not in done]
    tok = AutoTokenizer.from_pretrained(JUDGE, revision=JUDGE_REV)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(JUDGE, revision=JUDGE_REV, dtype=torch.bfloat16).to(device()).eval()

    def text(r):
        msg = PROMPT.format(lang=LANG[r["lang"]], instruction=r["prompt"], reference=r["reference"],
                            response=r["response"] or "(empty)")
        return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    texts = {r["uid"]: text(r) for r in rows}
    rows.sort(key=lambda r: -len(texts[r["uid"]]))
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(rows), a.batch):
            batch = rows[i:i + a.batch]
            enc = tok([texts[r["uid"]] for r in batch], return_tensors="pt", padding=True).to(device())
            gen = model.generate(**enc, max_new_tokens=48, do_sample=False)
            for r, g in zip(batch, gen[:, enc.input_ids.shape[1]:]):
                raw = tok.decode(g, skip_special_tokens=True)
                s = parse(raw)
                rec = {"uid": r["uid"], "lang": r["lang"], "judge": f"{JUDGE}@{JUDGE_REV}", "judge_raw": raw,
                       "scores": s, "task_score": sum(s.values()) / 4 if s else None}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"judge {i + len(batch)}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
