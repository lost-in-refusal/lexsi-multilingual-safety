"""Activation steering with refusal directions (causal check for lir.probe).

Adds coef * (difference-of-means refusal direction) to the residual stream
after one decoder layer, at every position, while generating. Directions come
from the base model's activations (runs/acts/base.npz).

  python -m lir.steer --arm plain --adapter runs/plain/adapter --direction en --coef 1.0
  python -m lir.steer --arm plain --adapter runs/plain/adapter --direction native --coef 1.0

`--direction en` uses the English direction for every language; `native`
uses each prompt's own-language direction. Families are subsampled to keep
each config under an hour; the same families are used for every config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lir.common import EVAL_LANGS, RUNS, empty_cache, rank_key, read_jsonl
from lir.evaluate import rows_for
from lir.probe import ACT, OUT, is_test


def directions(layer: int) -> dict[str, np.ndarray]:
    z = np.load(ACT / "base.npz")
    acts = z["acts"][:, layer].astype(np.float32)
    lang, harm, fam = z["lang"], z["harmful"], z["family"]
    train = np.array([not is_test(f) for f in fam])
    out = {}
    for l in EVAL_LANGS:
        k = (lang == l) & train
        out[l] = acts[k & (harm == 1)].mean(0) - acts[k & (harm == 0)].mean(0)
    return out


def subsample(rows: list[dict], n_families: int) -> list[dict]:
    fams = sorted({r["family"] for r in rows if is_test(r["family"])}, key=lambda f: rank_key(f"steer-{f}"))
    keep = set(fams[:n_families])
    return [r for r in rows if r["family"] in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--direction", choices=["en", "native"], required=True)
    ap.add_argument("--coef", type=float, default=1.0)
    ap.add_argument("--layer", type=int)
    ap.add_argument("--families", type=int, default=80)
    a = ap.parse_args()
    layer = a.layer if a.layer is not None else json.loads((OUT / "probe.json").read_text())["best_layer"]
    dirs = directions(layer)
    name = f"{a.arm}+steer-{a.direction}-L{layer}-c{a.coef:g}"
    out = RUNS / "eval" / name
    out.mkdir(parents=True, exist_ok=True)

    from lir import generate, guard

    tok, model = generate.load_model(a.adapter)
    blocks = model.model.layers
    current = {"vec": None}

    def hook(_m, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + current["vec"].to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = blocks[layer - 1].register_forward_hook(hook)  # hidden_states[layer] = output of block layer-1
    sets = {"attack": subsample(rows_for("attack"), a.families), "xstest": subsample(rows_for("xstest"), a.families)}
    try:
        for set_name, rows in sets.items():
            for l in EVAL_LANGS:
                v = dirs["en" if a.direction == "en" else l]
                current["vec"] = torch.tensor(a.coef * v, device=model.device)
                generate.generate(tok, model, [r for r in rows if r["lang"] == l], out / f"{set_name}.gen.jsonl",
                                  256, tag=f"{name}/{set_name}/{l}")
    finally:
        handle.remove()
    del model
    empty_cache()
    gtok, gmodel = guard.load()
    for set_name in sets:
        guard.judge(gtok, gmodel, read_jsonl(out / f"{set_name}.gen.jsonl"), out / f"{set_name}.guard.jsonl")
    (out / "steer.json").write_text(json.dumps({"layer": layer, "direction": a.direction, "coef": a.coef,
                                                "families": a.families, "adapter": a.adapter}, indent=2))


if __name__ == "__main__":
    main()
