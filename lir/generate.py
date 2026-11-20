"""Greedy batched generation for Tiny Aya (optionally with a LoRA adapter).

Output is appended to a JSONL file one batch at a time, so an interrupted run
resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lir.common import MODEL, MODEL_REV, device, read_jsonl


def load_model(adapter: str | None = None, merge: bool = True, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=MODEL_REV, dtype=dtype, attn_implementation="eager"
    )
    if adapter:
        from peft import PeftModel

        # every Tiny Aya variant has the same shapes, so a wrong pairing would load without an error
        base = json.loads((Path(adapter) / "adapter_config.json").read_text()).get("base_model_name_or_path")
        if base != MODEL:
            raise SystemExit(f"{adapter} was trained on {base}, but the loaded model is {MODEL}")
        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    return tok, model.to(device()).eval()


def chat_prefix(tok, prompt: str, system: str | None = None) -> str:
    msgs = [{"role": "user", "content": prompt}]
    if system is not None:
        msgs.insert(0, {"role": "system", "content": system})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(tok, model, rows: list[dict], out_path: Path, max_new_tokens: int,
             token_budget: int = 64000, max_batch: int = 96, tag: str = "") -> None:
    """rows need `uid` and `prompt`. Already-written uids are skipped."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["uid"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["uid"] not in done]
    if not todo:
        print(f"[{tag}] nothing to do ({len(done)} done)")
        return
    texts = {r["uid"]: chat_prefix(tok, r["prompt"]) for r in todo}
    lens = {u: len(tok(t, add_special_tokens=False).input_ids) for u, t in texts.items()}
    todo.sort(key=lambda r: -lens[r["uid"]])
    dev = device()
    stop = [tok.eos_token_id, tok.convert_tokens_to_ids("<EOS_TOKEN>")]
    t_start, n_tok = time.time(), 0
    i = 0
    with open(out_path, "a", encoding="utf-8") as f:
        while i < len(todo):
            longest = lens[todo[i]["uid"]] + max_new_tokens
            bs = max(1, min(max_batch, token_budget // longest))
            batch = todo[i:i + bs]
            enc = tok([texts[r["uid"]] for r in batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(dev)
            t0 = time.time()
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=stop)
            if dev == "mps":
                torch.mps.synchronize()
            dt = time.time() - t0
            gen = out[:, enc.input_ids.shape[1]:].tolist()
            for r, ids in zip(batch, gen):
                ids = [t for t in ids if t != tok.pad_token_id]
                cut = next((k for k, t in enumerate(ids) if t in stop), None)
                hit_eos = cut is not None
                if hit_eos:
                    ids = ids[:cut]
                n_tok += len(ids)
                rec = {k: v for k, v in r.items() if k != "prompt"}
                rec.update(prompt=r["prompt"], response=tok.decode(ids, skip_special_tokens=True).strip(),
                           n_tokens=len(ids), truncated=not hit_eos)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            i += bs
            el = time.time() - t_start
            print(f"[{tag}] {min(i, len(todo))}/{len(todo)} bs={bs} batch {dt:.1f}s | {n_tok / el:.1f} tok/s overall",
                  flush=True)
    print(f"[{tag}] done: {n_tok} tokens in {time.time() - t_start:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL with uid + prompt")
    ap.add_argument("--output", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    rows = read_jsonl(Path(a.input))[: a.limit]
    tok, model = load_model(a.adapter)
    generate(tok, model, rows, Path(a.output), a.max_new_tokens, tag=Path(a.output).stem)


if __name__ == "__main__":
    main()
