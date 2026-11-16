"""Per-language refusal circuits with CircuitKIT (EAP-IG, node level).

  python -m lir.circuits discover --tag base --device mps
  python -m lir.circuits discover --tag plain --adapter runs/plain/adapter --device mps
  python -m lir.circuits discover --tag base --dry-run       # pairs + answer tokens only
  python -m lir.circuits compare --tags base plain safegrad
  python -m lir.circuits faithfulness --tag base --ks 612 300 100 50 --device mps   # held-out pairs
  python -m lir.circuits margin --device mps        # the metric with the top-5 opening tokens per side

Run with .venv-circuit/bin/python. Every language gets the same MultiJail
families. Clean = harmful prompt, corrupt = a same-language XSTest prompt with
the same token length: CircuitKIT aligns clean/corrupt by left padding, so any
length gap shifts the whole system preamble and adds spurious activation
differences. Metric = logit(refusal token) - logit(compliance token) at the
first response position, prompts wrapped in the model's chat template.
Answer tokens default to the base model's own first tokens, so base and
fine-tuned runs share one metric. Families depend on --langs and the XSTest
files, tokens on the generation/guard files present: run all arms against the
same inputs (compare warns if they differ).

Faithfulness is CircuitKIT's patching pillar on the global top-k nodes, scored on
the next n families in the same order (held out from discovery, paired the same
way). Writes results/lir/circuits/<tag>_<lang>.json, raw CircuitKIT artifacts
under results/lir/circuits/raw/<tag>/, compare_<tags>.json and margin_check.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from lir.common import DATA, EVAL_LANGS, MODEL, MODEL_REV, ROOT, RUNS, SEED, rank_key, read_jsonl, write_json

OUT = ROOT / "results" / "lir" / "circuits"

# Fallbacks when no generations exist yet; each must be a single token in context.
REFUSAL = {"en": "I'm", "ko": "죄", "zh": "抱", "sw": "Sam"}
COMPLY = {"en": "Sure", "ko": "물론", "zh": "当然", "sw": "Kwa"}
# Only used to split responses until Qwen3Guard verdicts exist.
REFUSAL_RE = {"en": r"sorry|can't|cannot|unable|won't", "ko": r"죄송|수 없|드릴 수",
              "zh": r"抱歉|无法|不能", "sw": r"samahani|siwezi|haiwezekani"}


def log(msg: str) -> None:
    print(msg, flush=True)


def resolve(model: str) -> str:
    """Hub id -> pinned local snapshot dir; local dirs and TransformerLens names pass through."""
    if Path(model).expanduser().is_dir():
        return str(Path(model).expanduser().resolve())
    if model != MODEL:
        return model
    from huggingface_hub import snapshot_download

    pats = ["*.json", "*.safetensors", "*.jinja"]
    try:
        return snapshot_download(MODEL, revision=MODEL_REV, allow_patterns=pats, local_files_only=True)
    except Exception:
        return snapshot_download(MODEL, revision=MODEL_REV, allow_patterns=pats)


class Prompter:
    """Chat-template wrapping and in-context token ids for one tokenizer."""

    def __init__(self, tok, chat: bool):
        self.tok, self.chat = tok, chat

    def __call__(self, text: str) -> str:
        if not self.chat:
            return text
        return self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                            add_generation_prompt=True)

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def length(self, text: str) -> int:
        return len(self.ids(self(text)))

    def continuation(self, answer: str, ctx: str = "x") -> list[int]:
        # tokens of `answer` as they follow the generation prompt
        p = self.ids(self(ctx))
        full = self.ids(self(ctx) + answer)
        return full[len(p):] if full[:len(p)] == p else self.ids(answer)

    def single(self, s: str) -> int:
        ids = self.continuation(s)
        if len(ids) != 1:
            sys.exit(f"answer token {s!r} is {len(ids)} tokens here: {self.tok.convert_ids_to_tokens(ids)}")
        return ids[0]

    def show(self, i: int) -> str:
        return f"{self.tok.decode([i])!r} (id {i}, piece {self.tok.convert_ids_to_tokens(i)!r})"


def _responses(gen_dir: Path, name: str):
    guard, gen = gen_dir / f"{name}.guard.jsonl", gen_dir / f"{name}.gen.jsonl"
    if guard.exists():
        return read_jsonl(guard), "guard"
    if gen.exists():
        return read_jsonl(gen), "regex"
    return [], None


def _refused(r: dict, src: str) -> bool:
    if src == "guard":
        return r.get("refusal") == "Yes"
    pat = REFUSAL_RE.get(r["lang"], REFUSAL_RE["en"])
    return bool(re.search(pat, r.get("response", "")[:120], re.I))


def pick_tokens(p: Prompter, langs, gen_dir: Path, cli_ref: dict, cli_comp: dict) -> dict:
    """Most common first response token: base refusals to MultiJail vs compliant answers to XSTest."""
    plan = [("refusal", "attack", *_responses(gen_dir, "attack"), cli_ref, REFUSAL),
            ("compliance", "xstest", *_responses(gen_dir, "xstest"), cli_comp, COMPLY)]
    out = {}
    for lang in langs:
        counts = {}
        for side, _, rows, src, _, _ in plan:
            counts[side] = Counter(p.continuation(t)[0] for r in rows
                                   if r.get("lang") == lang and (t := r.get("response", "").lstrip())
                                   and _refused(r, src) == (side == "refusal"))
        sides = {}
        for side, fname, _, src, cli, default in plan:
            taken = sides.get("refusal", {}).get("id")
            ranked = [(i, c) for i, c in counts[side].most_common() if i != taken]
            if lang in cli:
                tid, how = p.single(cli[lang]), "cli"
            elif ranked:
                tid, how = ranked[0][0], f"{fname}.{src} ({ranked[0][1]}/{sum(counts[side].values())})"
            else:
                tid, how = p.single(default[lang]), "default"
                log(f"WARNING [{lang}] no usable {side} generations in {gen_dir}, falling back to {default[lang]!r}")
            other = counts["compliance" if side == "refusal" else "refusal"]
            # how often the token opens the other kind of answer; high means a weak contrast
            share = round(other[tid] / sum(other.values()), 3) if other else None
            sides[side] = {"id": tid, "token": p.tok.decode([tid]), "piece": p.tok.convert_ids_to_tokens(tid),
                           "source": how, "candidates": [[p.tok.decode([i]), c] for i, c in ranked[:5]],
                           "share_of_other_side": share}
            log(f"[{lang}] {side} token {p.show(tid)} from {how}; top: {sides[side]['candidates']}"
                + (f"; opens {share:.0%} of the other side's answers" if share is not None else ""))
        if sides["refusal"]["id"] == sides["compliance"]["id"]:
            sys.exit(f"[{lang}] refusal and compliance tokens are identical")
        out[lang] = sides
    return out


def build_pairs(p: Prompter, langs, n: int, max_tokens: int, max_gap: int):
    """Same MultiJail families in every language, each paired with an equal-length XSTest prompt.

    Families go in hash order; one is skipped when some language has no unused
    control within max_gap tokens of it.
    """
    fam = defaultdict(dict)
    for r in read_jsonl(DATA / "multijail_attack.jsonl"):
        fam[r["family"]][r["lang"]] = r["prompt"]
    cands = sorted((f for f, d in fam.items() if all(l in d and len(p.ids(d[l])) <= max_tokens for l in langs)),
                   key=rank_key)
    pools, files = {}, {}
    for lang in langs:
        path = DATA / f"xstest_{lang}.jsonl"
        if not path.exists():
            log(f"WARNING {path.name} missing, using English XSTest controls for {lang}")
            path = DATA / "xstest_en.jsonl"
        files[lang] = path.name
        pools[lang] = [(r["uid"], r["prompt"], p.length(r["prompt"])) for r in read_jsonl(path)]
    free = {lang: set(range(len(pools[lang]))) for lang in langs}
    rows, skipped = {lang: [] for lang in langs}, 0
    for f in cands:
        if len(rows[langs[0]]) == n:
            break
        pick = {}
        for lang in langs:
            pool, hl = pools[lang], p.length(fam[f][lang])
            j = min(free[lang], key=lambda i: (abs(pool[i][2] - hl), i), default=None)
            if j is None or abs(pool[j][2] - hl) > max_gap:
                break
            pick[lang] = (j, hl)
        if len(pick) < len(langs):
            skipped += 1
            continue
        for lang, (j, hl) in pick.items():
            free[lang].discard(j)
            uid, text, bl = pools[lang][j]
            rows[lang].append({"family": f, "harmful": fam[f][lang], "benign_uid": uid, "benign": text,
                               "tokens": hl, "gap": bl - hl})
    got = len(rows[langs[0]])
    log(f"{got} families ({skipped} skipped for length gap > {max_gap}, {len(cands)} under {max_tokens} tokens)")
    if got < n:
        log(f"WARNING wanted {n} families, found {got}; raise --max-gap or --max-prompt-tokens")
    pairs = {}
    for lang in langs:
        rs = rows[lang]
        exact = sum(r["gap"] == 0 for r in rs)
        log(f"[{lang}] controls from {files[lang]}: {exact}/{got} exact-length, max |gap| "
            f"{max(abs(r['gap']) for r in rs)}, sequence length {min(r['tokens'] for r in rs)}-"
            f"{max(r['tokens'] for r in rs)} tokens")
        pairs[lang] = {"control_file": files[lang], "exact": exact, "rows": rs}
    return [r["family"] for r in rows[langs[0]]], pairs


def register(name: str, rows, p: Prompter, tokens: dict, cache_dir: str) -> None:
    from circuitkit.data.normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset
    from circuitkit.data.normalized_task import NormalizedTaskSpec
    from circuitkit.tasks.registry import register_task

    class ChatTask(NormalizedTaskSpec):
        # prompts arrive templated (BOS included); the stock loader would prepend a second BOS
        def _build_eap_dataloader(self, model, batch_size, seed=42):
            dl = super()._build_eap_dataloader(model, batch_size, seed)
            dl.templated = p.chat
            return dl

    ref, comp = tokens["refusal"], tokens["compliance"]
    labels = {"clean_label_id": ref["id"], "corrupt_label_id": comp["id"]}
    recs = [ContrastiveRecord(record_id=r["family"], clean_prompt=p(r["harmful"]), clean_answer=ref["token"],
                              corrupt_prompt=p(r["benign"]), corrupt_answer=comp["token"],
                              target_field="refusal_vs_compliance", contrast_source=ContrastSource.EXTERNAL,
                              meta={"_precomputed_labels": labels}) for r in rows]
    ds = NormalizedDataset(name=name, shape=DatasetShape.REFUSAL, records=recs, source="lir")
    register_task(ChatTask(ds, name=name, cache_dir=cache_dir))


def load_model(path: str, a):
    import circuitkit as ck
    import torch

    kw = {}
    if a.adapter:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, a.dtype))
        hf = PeftModel.from_pretrained(hf, a.adapter).merge_and_unload()
        if a.save_merged:
            dst = RUNS / "merged" / a.tag
            hf.save_pretrained(dst)
            AutoTokenizer.from_pretrained(path).save_pretrained(dst)
            log(f"merged checkpoint -> {dst}")
        kw["hf_model"] = hf
        log(f"merged adapter {a.adapter}")
    model = ck.load_model(path, dtype=a.dtype, device=a.device, **kw)
    model.eval()
    # tokenize_plus left-pads but builds the mask from tokenizer.padding_side
    # (right for Tiny Aya), which leaves the pads unmasked
    model.tokenizer.padding_side = "left"
    return model


def logit_gap(model, prompts, rid: int, cid: int, templated: bool, bs: int) -> float:
    """Mean logit(refusal) - logit(compliance) at the first response position."""
    import torch
    from circuitkit.backends.eap.eap_utils import tokenize_plus

    vals = []
    with torch.no_grad():
        for i in range(0, len(prompts), bs):
            toks, mask, _, _ = tokenize_plus(model, prompts[i:i + bs], padding_side="left", templated=templated)
            last = model(toks.to(model.cfg.device), attention_mask=mask.to(model.cfg.device))[:, -1].float()
            vals += (last[:, rid] - last[:, cid]).tolist()
    return sum(vals) / len(vals)


def peak_mem_gb(device: str) -> dict:
    import torch

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    out = {"max_rss_gb": round(rss / (1 << 30 if sys.platform == "darwin" else 1 << 20), 2)}
    if device == "mps":
        out["mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / (1 << 30), 2)
    if device.startswith("cuda"):
        out["cuda_peak_gb"] = round(torch.cuda.max_memory_allocated() / (1 << 30), 2)
    return out


def discover(a) -> None:
    import circuitkit as ck  # before anything pulls in transformer_lens
    from transformers import AutoTokenizer

    path = resolve(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    chat = bool(getattr(tok, "chat_template", None)) and not a.no_chat_template
    if not chat:
        log("WARNING running without a chat template")
    p = Prompter(tok, chat)
    tokens = pick_tokens(p, a.langs, Path(a.gen_dir), dict(a.refusal_token), dict(a.compliance_token))
    fams, pairs = build_pairs(p, a.langs, a.n_examples, a.max_prompt_tokens, a.max_gap)
    if a.dry_run:
        return

    from circuitkit.api import discover_circuit
    from circuitkit.circuit import Circuit
    from circuitkit.quick import build_discovery_config

    model = load_model(path, a)
    n_nodes = model.cfg.n_layers * (model.cfg.n_heads + 1)
    out = Path(a.out)
    raw = out / "raw" / a.tag
    raw.mkdir(parents=True, exist_ok=True)
    for lang in a.langs:
        rows, t = pairs[lang]["rows"], tokens[lang]
        rid, cid = t["refusal"]["id"], t["compliance"]["id"]
        name = f"lir_refusal_{a.tag}_{lang}"
        t0 = time.time()
        gap_h = logit_gap(model, [p(r["harmful"]) for r in rows], rid, cid, chat, a.batch_size)
        gap_b = logit_gap(model, [p(r["benign"]) for r in rows], rid, cid, chat, a.batch_size)
        log(f"[{lang}] logit(refusal)-logit(compliance): harmful {gap_h:+.3f}, benign {gap_b:+.3f}")
        with tempfile.TemporaryDirectory() as cache:
            register(name, rows, p, t, cache)
            cfg = build_discovery_config(model, name, algorithm="eap-ig", level="node", n_examples=len(rows),
                                         batch_size=a.batch_size, sparsity=1 - a.k / n_nodes, scope="both",
                                         output_path=str(raw / f"{lang}.pt"), ig_steps=a.ig_steps, seed=SEED,
                                         mlp_hook=a.mlp_hook)
            cfg["model"]["device"] = a.device  # discover_circuit would otherwise auto-pick MPS
            discover_circuit(cfg, _model=model)
            circuit = Circuit.from_artifact(raw / f"{lang}.pt")
            faith = faithfulness(ck, model, circuit, name, len(rows), a, raw / f"{lang}_faithfulness.json")
        secs = time.time() - t0
        scores = dict(sorted(circuit.scores.items(), key=lambda kv: -kv[1]))
        top = list(scores.items())[:a.k]
        log(f"[{lang}] done in {secs:.0f}s; top-5 {[(n, round(s, 4)) for n, s in top[:5]]}")
        write_json(out / f"{a.tag}_{lang}.json", {
            "tag": a.tag, "lang": lang, "model": a.model, "revision": MODEL_REV if a.model == MODEL else None,
            "adapter": a.adapter, "algorithm": "eap-ig", "level": "node", "ig_steps": a.ig_steps,
            "mlp_hook": a.mlp_hook, "batch_size": a.batch_size, "dtype": a.dtype, "device": a.device,
            "chat_template": chat, "n_examples": len(rows), "k": a.k, "n_nodes": len(scores),
            "answer_tokens": t, "logit_gap": {"harmful": gap_h, "benign": gap_b},
            "faithfulness": faith, "seconds": round(secs, 1), "memory": peak_mem_gb(a.device),
            "pairs": {"control_file": pairs[lang]["control_file"], "exact_length": pairs[lang]["exact"],
                      "rows": [{k: r[k] for k in ("family", "benign_uid", "tokens", "gap")} for r in rows]},
            "top_k": [[n, s] for n, s in top],
            "node_scores": scores,  # |EAP-IG| per node; CircuitKIT drops the sign
            "artifact": str(raw / f"{lang}.pt"),
        })


def pruned_for_top_k(ranked: list, k: int) -> list:
    """Lowest-scoring nodes to hand CircuitKIT as the pruned list so that it keeps exactly the top k.

    CircuitKIT keeps int(N * (1 - pruned/N)) nodes, which can round one short.
    """
    total = len(ranked)
    m = next((m for m in (total - k, total - k - 1) if m >= 0 and int(total * (1 - m / total)) == k), total - k)
    return ranked[total - m:]  # not ranked[-m:], which is everything when m == 0


def faithfulness(ck, model, circuit, name, n, a, path: Path):
    """CircuitKIT's patching pillar on the global top-k nodes.

    The artifact from discover_circuit caps head pruning at 50% per layer, so it
    is not the top-k circuit. ck.faithfulness keeps int(N * (1 - pruned/N))
    nodes, so hand it a pruned list whose length rounds to exactly k.
    """
    if not a.faithfulness:
        return None
    from circuitkit.circuit import Circuit

    ranked = sorted(circuit.scores, key=circuit.scores.get, reverse=True)
    total = len(ranked)
    pruned = pruned_for_top_k(ranked, a.k)
    m = len(pruned)
    top = Circuit(pruned, circuit.scores, algorithm="eap-ig", level="node")  # nodes = the pruned ones
    try:
        rep = ck.faithfulness(model, top, name, pillars=["patching"], n_examples=n, batch_size=a.batch_size,
                              device=a.device)
        rep.to_json(path)
        out = json.loads(path.read_text())
        out["n_nodes_kept"] = int(total * (1 - m / total))
        return out
    except Exception as e:  # report and keep the discovery result
        log(f"WARNING faithfulness failed: {e!r}")
        return {"error": repr(e)}


def faithful(a) -> None:
    """Patching faithfulness for circuits already discovered, with the discovery run's answer tokens, on the
    next n_examples families in the same order (held out from discovery, same exact-length pairing)."""
    import circuitkit as ck  # noqa: F401  (before anything pulls in transformer_lens)
    from circuitkit.circuit import Circuit
    from transformers import AutoTokenizer

    path = resolve(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    p = Prompter(tok, bool(getattr(tok, "chat_template", None)))
    out = Path(a.out)
    saved = {lang: json.loads((out / f"{a.tag}_{lang}.json").read_text()) for lang in a.langs}
    tokens = {lang: s["answer_tokens"] for lang, s in saved.items()}  # the discovery run's metric
    # discovery picked families for all evaluation languages at once; the first n reproduce it
    n = a.n_examples
    _, pairs = build_pairs(p, EVAL_LANGS, 2 * n, a.max_prompt_tokens, a.max_gap)
    for lang, s in saved.items():
        rows = pairs[lang]["rows"]
        if [r["family"] for r in rows[:n]] != [r["family"] for r in s["pairs"]["rows"]]:
            sys.exit(f"[{lang}] families differ from the discovery run; pass the same --n-examples/--max-gap")
        if len(rows) < 2 * n:
            sys.exit(f"[{lang}] only {len(rows) - n} held-out families")
        pairs[lang]["rows"] = rows[n:]
    a.adapter = a.adapter or saved[a.langs[0]].get("adapter")
    a.save_merged = False
    model = load_model(path, a)
    a.faithfulness = True
    for lang in a.langs:
        name = f"lir_refusal_{a.tag}_{lang}_heldout"
        circuit = Circuit.from_artifact(out / "raw" / a.tag / f"{lang}.pt")
        s = saved[lang]
        held = (s.get("faithfulness") or {}).get("held_out_by_k", {})
        with tempfile.TemporaryDirectory() as cache:
            register(name, pairs[lang]["rows"], p, tokens[lang], cache)
            for k in a.ks:
                a.k = k
                t0 = time.time()
                res = faithfulness(ck, model, circuit, name, len(pairs[lang]["rows"]), a,
                                   out / "raw" / a.tag / f"{lang}_faithfulness_heldout_k{k}.json")
                res["seconds"] = round(time.time() - t0, 1)
                held[str(k)] = res
                log(f"[{lang}] k={k} faithfulness {res.get('patching_score')} ({res['seconds']}s)")
                s["faithfulness"] = {"held_out_families": [r["family"] for r in pairs[lang]["rows"]],
                                     "held_out_by_k": dict(sorted(held.items(), key=lambda kv: int(kv[0])))}
                write_json(out / f"{a.tag}_{lang}.json", s)


def margin(a) -> None:
    """Robustness check on the circuit metric: the same harmful/benign pairs, scored with each language's
    top-n refusal and compliance opening tokens (log-sum-exp of their log-probs) instead of one token each."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve(MODEL)
    tok = AutoTokenizer.from_pretrained(path)
    p = Prompter(tok, True)
    _, pairs = build_pairs(p, EVAL_LANGS, a.n_examples, a.max_prompt_tokens, a.max_gap)
    gen = Path(a.gen_dir)
    plan = [("refusal", *_responses(gen, "attack")), ("compliance", *_responses(gen, "xstest"))]
    sets = {}
    for lang in EVAL_LANGS:
        counts = {side: Counter(p.continuation(t)[0] for r in rows if r.get("lang") == lang
                                and (t := r.get("response", "").lstrip()) and _refused(r, src) == (side == "refusal"))
                  for side, rows, src in plan}
        tot = {side: sum(c.values()) for side, c in counts.items()}
        sets[lang] = {}
        for side, c in counts.items():
            o = counts["compliance" if side == "refusal" else "refusal"]
            other = "compliance" if side == "refusal" else "refusal"
            # a token counts for the side whose answers it opens more often, relative to that side's size
            top = [(i, n) for i, n in c.most_common() if n / tot[side] > o[i] / tot[other]][:a.top_n]
            sets[lang][side] = {"ids": [i for i, _ in top], "tokens": [tok.decode([i]) for i, _ in top],
                                "share": round(sum(n for _, n in top) / sum(c.values()), 3)}
    dev = a.device
    res = {"top_n": a.top_n, "sets": sets, "arms": {}}
    for tag, adapter in a.arms:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager")
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        model.to(dev).eval()
        res["arms"][tag] = {}
        for lang in EVAL_LANGS:
            r_ids = torch.tensor(sets[lang]["refusal"]["ids"], device=dev)
            c_ids = torch.tensor(sets[lang]["compliance"]["ids"], device=dev)
            out = {}
            for kind in ("harmful", "benign"):
                vals = []
                with torch.no_grad():
                    for row in pairs[lang]["rows"]:
                        ids = torch.tensor([p.ids(p(row[kind]))], device=dev)
                        lp = model(ids).logits[0, -1].float().log_softmax(-1)
                        vals.append((lp[r_ids].logsumexp(0) - lp[c_ids].logsumexp(0)).item())
                out[kind] = sum(vals) / len(vals)
            out["separation"] = out["harmful"] - out["benign"]
            res["arms"][tag][lang] = out
            log(f"[{tag}/{lang}] set contrast harmful {out['harmful']:+.2f} benign {out['benign']:+.2f} "
                f"separation {out['separation']:.2f}")
        del model
        if dev == "mps":
            torch.mps.empty_cache()
    write_json(Path(a.out) / "margin_check.json", res)


def spearman(x: dict, y: dict) -> float:
    from scipy.stats import spearmanr

    common = sorted(set(x) & set(y))
    return float(spearmanr([x[n] for n in common], [y[n] for n in common]).statistic)


def compare(a) -> None:
    from circuitkit.analysis.cross_method_jaccard import cross_method_jaccard

    runs, fams, toks = {}, {}, {}
    for tag in a.tags:
        for lang in a.langs:
            f = Path(a.out) / f"{tag}_{lang}.json"
            if f.exists():
                d = json.loads(f.read_text())
                runs[f"{tag}/{lang}"] = d["node_scores"]
                fams[f"{tag}/{lang}"] = [r["family"] for r in d["pairs"]["rows"]]
                toks[f"{tag}/{lang}"] = (d["answer_tokens"]["refusal"]["id"], d["answer_tokens"]["compliance"]["id"])
            else:
                log(f"missing {f}, skipped")
    if len(runs) < 2:
        sys.exit("need at least two runs to compare")
    # family and token choice depend on --langs, the XSTest translations and the guard files present
    if len({tuple(v) for v in fams.values()}) > 1:
        log("WARNING runs were built on different MultiJail families; rerun with the same --langs and control files")
    for lang in a.langs:
        if len({v for r, v in toks.items() if r.endswith(f"/{lang}")}) > 1:
            log(f"WARNING [{lang}] answer tokens differ across tags, so the metrics differ; pin them with "
                "--refusal-token/--compliance-token")
    k = a.k
    jac = cross_method_jaccard(runs, top_k=k)
    names = jac.methods
    tops = {m: set(jac.top_nodes[m]) for m in names}
    overlap = [[len(tops[x] & tops[y]) for y in names] for x in names]
    rho = [[spearman(runs[x], runs[y]) for y in names] for x in names]
    n_nodes = len(next(iter(runs.values())))
    chance = (k * k / n_nodes) / (2 * k - k * k / n_nodes)
    shared = {}
    for tag in a.tags:
        sets = [tops[f"{tag}/{l}"] for l in a.langs if f"{tag}/{l}" in tops]
        if sets:
            shared[tag] = sorted(set.intersection(*sets))
    res = {"k": k, "n_nodes": n_nodes, "runs": names, "jaccard": jac.matrix, "overlap": overlap,
           "spearman": rho, "jaccard_chance": chance, "shared_by_all_langs": shared, "top_k": jac.top_nodes}
    path = Path(a.out) / f"compare_{'-'.join(a.tags)}{'' if k == 50 else f'_k{k}'}.json"
    write_json(path, res)
    w = max(len(x) for x in names) + 2
    for title, m in [(f"top-{k} Jaccard (chance {chance:.3f})", jac.matrix), ("Spearman rho of node scores", rho)]:
        log(f"\n{title}\n{'':{w}}" + "".join(f"{x:>{w}}" for x in names))
        for x, row in zip(names, m):
            log(f"{x:{w}}" + "".join(f"{v:{w}.3f}" for v in row))
    for tag, nodes in shared.items():
        log(f"\n{tag}: {len(nodes)} nodes in every language's top-{k}: {nodes}")
    log(f"\n-> {path}")


def kv(s: str):
    lang, _, tok = s.partition("=")
    if not tok:
        raise argparse.ArgumentTypeError(f"expected lang=token, got {s!r}")
    return lang, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--model", default=MODEL)
    d.add_argument("--adapter", help="PEFT LoRA adapter, merged before analysis")
    d.add_argument("--save-merged", action="store_true", help="also write runs/merged/<tag>")
    d.add_argument("--tag", required=True)
    d.add_argument("--langs", nargs="+", default=EVAL_LANGS)
    d.add_argument("--n-examples", type=int, default=64)
    d.add_argument("--max-prompt-tokens", type=int, default=32)
    d.add_argument("--max-gap", type=int, default=0, help="allowed clean/corrupt token-length gap")
    d.add_argument("--k", type=int, default=50)
    d.add_argument("--ig-steps", type=int, default=5)
    d.add_argument("--batch-size", type=int, default=1)
    # CircuitKIT's default mlp_out pairs the MLP output delta with the MLP input
    # gradient; post_act scores delta . grad at the MLP neurons, which equals the
    # usual output-side EAP score, at ~5x the activation buffer
    d.add_argument("--mlp-hook", choices=["post_act", "mlp_out"], default="post_act")
    d.add_argument("--device", default="cpu")
    d.add_argument("--dtype", default="bfloat16")
    d.add_argument("--gen-dir", default=str(RUNS / "eval" / "base"), help="generations used to pick answer tokens")
    d.add_argument("--refusal-token", type=kv, nargs="*", default=[], metavar="LANG=TOK")
    d.add_argument("--compliance-token", type=kv, nargs="*", default=[], metavar="LANG=TOK")
    d.add_argument("--faithfulness", action="store_true", help="CircuitKIT patching pillar on the top-k circuit")
    d.add_argument("--no-chat-template", action="store_true")
    d.add_argument("--dry-run", action="store_true")
    d.add_argument("--out", default=str(OUT))
    f = sub.add_parser("faithfulness", help="patching pillar for circuits already discovered")
    f.add_argument("--tag", required=True)
    f.add_argument("--model", default=MODEL)
    f.add_argument("--adapter", help="defaults to the adapter recorded by the discovery run")
    f.add_argument("--langs", nargs="+", default=EVAL_LANGS)
    f.add_argument("--n-examples", type=int, default=64)
    f.add_argument("--max-prompt-tokens", type=int, default=32)
    f.add_argument("--max-gap", type=int, default=0)
    f.add_argument("--ks", type=int, nargs="+", default=[50], help="circuit sizes (top-k nodes) to evaluate")
    f.add_argument("--batch-size", type=int, default=1)
    f.add_argument("--device", default="cpu")
    f.add_argument("--dtype", default="bfloat16")
    f.add_argument("--out", default=str(OUT))
    m = sub.add_parser("margin", help="harmful/benign contrast over the top-n opening tokens per side")
    m.add_argument("--arms", type=lambda s: tuple(s.split("=", 1)) if "=" in s else (s, None), nargs="+",
                   default=[("base", None), ("plain", "runs/plain/adapter"), ("safegrad", "runs/safegrad/adapter")],
                   metavar="TAG[=ADAPTER]")
    m.add_argument("--top-n", type=int, default=5)
    m.add_argument("--n-examples", type=int, default=64)
    m.add_argument("--max-prompt-tokens", type=int, default=32)
    m.add_argument("--max-gap", type=int, default=0)
    m.add_argument("--device", default="mps")
    m.add_argument("--gen-dir", default=str(RUNS / "eval" / "base"))
    m.add_argument("--out", default=str(OUT))
    c = sub.add_parser("compare")
    c.add_argument("--tags", nargs="+", required=True)
    c.add_argument("--langs", nargs="+", default=EVAL_LANGS)
    c.add_argument("--k", type=int, default=50)
    c.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    # CircuitKIT's opt-in TransformerLens patch drops per-forward weight copies
    # (W_U, gated MLP); same scores to ~1e-6 here, ~30% lower peak memory.
    # Read when circuitkit is first imported, so set it before that.
    os.environ.setdefault("CIRCUITKIT_TL_MEMORY_PATCH", "1")
    {"discover": discover, "faithfulness": faithful, "margin": margin, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
