"""Where does refusal live, and why do other languages slip past it?

  python -m lir.probe extract --arm base
  python -m lir.probe extract --arm plain --adapter runs/plain/adapter
  python -m lir.probe analyze --arms base plain safegrad

extract: last-token residual stream (the <|START_RESPONSE|> position) at every
layer for every attack / XSTest-safe / task prompt.
analyze: difference-of-means refusal direction per language and layer, how
well the English direction separates harmful from harmless prompts in each
language, linear vs MLP probes trained on English and tested on held-out
families in every language, and how much fine-tuning shrinks each language's
projection onto the base model's English refusal direction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lir.common import EVAL_LANGS, ROOT, RUNS, device, rank_key, read_jsonl, write_json
from lir.evaluate import rows_for

ACT = RUNS / "acts"
OUT = ROOT / "results" / "lir"
PREFIX = ""


def is_test(family: str) -> bool:
    return rank_key(f"probe-{family}")[0] in "01234567"


@torch.no_grad()
def extract(arm: str, adapter: str | None, limit: int | None = None, out_name: str | None = None) -> None:
    from lir.generate import chat_prefix, load_model

    tok, model = load_model(adapter)
    rows = []
    for name, harmful in [("attack", 1), ("xstest", 0), ("task", 0)]:
        for r in rows_for(name):
            fam = r.get("family") or f"bx-{r['id']}"
            rows.append({"uid": r["uid"], "lang": r["lang"], "family": fam, "harmful": harmful, "set": name,
                         "text": chat_prefix(tok, r["prompt"])})
    if limit:  # quick test: first `limit` families of each set
        keep = {}
        for r in rows:
            keep.setdefault(r["set"], [])
            if r["family"] not in keep[r["set"]] and len(keep[r["set"]]) < limit:
                keep[r["set"]].append(r["family"])
        rows = [r for r in rows if r["family"] in keep[r["set"]]]
    acts = []
    for i in range(0, len(rows), 16):
        batch = rows[i:i + 16]
        enc = tok([r["text"] for r in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device())
        hs = model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states
        acts.append(torch.stack([h[:, -1, :] for h in hs], 1).float().cpu().numpy().astype(np.float16))
        print(f"extract {arm} {i + len(batch)}/{len(rows)}", flush=True)
    ACT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ACT / f"{out_name or arm}.npz", acts=np.concatenate(acts), uid=[r["uid"] for r in rows],
                        lang=[r["lang"] for r in rows], family=[r["family"] for r in rows],
                        harmful=[r["harmful"] for r in rows], set=[r["set"] for r in rows])


def _load(arm):
    z = np.load(ACT / f"{PREFIX}{arm}.npz")
    meta = {k: z[k] for k in ["uid", "lang", "family", "harmful", "set"]}
    meta["test"] = np.array([is_test(f) for f in meta["family"]])
    return z["acts"].astype(np.float32), meta


def _auroc(scores, labels):
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) != 2 or not np.isfinite(scores).all():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _unit(v):
    # layer 0 is the embedding of <|START_RESPONSE|>, identical for every prompt, so the
    # difference of means there is exactly zero
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def analyze(arms: list[str]) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    X, m = _load("base")
    n_layers = X.shape[1]
    # harmful = MultiJail, harmless = XSTest safe (+ task prompts for en/ko/zh)
    res = {"layers": n_layers, "per_lang": {}}

    def mask(lang, test=None, harmful=None):
        k = m["lang"] == lang
        if test is not None:
            k &= m["test"] == test
        if harmful is not None:
            k &= m["harmful"] == harmful
        return k

    def direction(acts, lang):
        return acts[mask(lang, False, 1)].mean(0) - acts[mask(lang, False, 0)].mean(0)  # [layers, d]

    dirs = {l: direction(X, l) for l in EVAL_LANGS}
    unit_en = _unit(dirs["en"])

    # 1. cosine between each language's refusal direction and English's, per layer
    for l in EVAL_LANGS:
        cos = (_unit(dirs[l]) * unit_en).sum(-1)
        res["per_lang"].setdefault(l, {})["cos_to_en"] = cos.tolist()

    # 2. separation with the English direction on held-out families
    for l in EVAL_LANGS:
        k = mask(l, True)
        proj = np.einsum("nld,ld->nl", X[k], unit_en)
        res["per_lang"][l]["auroc_en_direction"] = [_auroc(proj[:, j], m["harmful"][k]) for j in range(n_layers)]
        own = _unit(dirs[l])
        projo = np.einsum("nld,ld->nl", X[k], own)
        res["per_lang"][l]["auroc_own_direction"] = [_auroc(projo[:, j], m["harmful"][k]) for j in range(n_layers)]

    # pick the layer where the English direction separates English best (ignoring the first quarter)
    en_auc = np.nan_to_num(np.array(res["per_lang"]["en"]["auroc_en_direction"]), nan=0.0)
    lo = n_layers // 4
    best = int(lo + np.argmax(en_auc[lo:]))
    res["best_layer"] = best

    # 3. linear vs MLP probe at the best layer: train on English, test everywhere
    tr = mask("en", False)
    sc = StandardScaler().fit(X[tr, best])
    probes = {"linear": LogisticRegression(max_iter=2000, C=0.1),
              "mlp": MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, alpha=1e-3, random_state=0)}
    res["probes"] = {}
    for name, clf in probes.items():
        clf.fit(sc.transform(X[tr, best]), m["harmful"][tr])
        res["probes"][name] = {l: _auroc(clf.predict_proba(sc.transform(X[mask(l, True), best]))[:, 1],
                                         m["harmful"][mask(l, True)]) for l in EVAL_LANGS}

    # 4. drift: harmful prompts' projection on the base English direction, per arm and language
    res["drift"] = {}
    for arm in arms:
        Xa, ma = (X, m) if arm == "base" else _load(arm)
        assert (ma["uid"] == m["uid"]).all()
        for l in EVAL_LANGS:
            k = (ma["lang"] == l) & (ma["harmful"] == 1)
            kb = (ma["lang"] == l) & (ma["harmful"] == 0)
            p = np.einsum("nld,ld->nl", Xa[k], unit_en)
            pb = np.einsum("nld,ld->nl", Xa[kb], unit_en)
            res["drift"].setdefault(arm, {})[l] = {
                "harmful_proj": p.mean(0).tolist(), "harmless_proj": pb.mean(0).tolist(),
                "gap_at_best": float(p[:, best].mean() - pb[:, best].mean())}
    # 5. per-language margin: harmful prompts' projection vs the arm's own refusal rate
    res["margin"] = {}
    for arm in arms:
        Xa, ma = (X, m) if arm == "base" else _load(arm)
        g_path = RUNS / "eval" / arm / "attack.guard.jsonl"
        if not g_path.exists():
            continue
        g = {r["uid"]: r for r in read_jsonl(g_path)}
        for l in EVAL_LANGS:
            k = (ma["lang"] == l) & (ma["harmful"] == 1)
            p = Xa[k, best] @ unit_en[best]
            ref = np.array([g[u]["refusal"] == "Yes" for u in ma["uid"][k]])
            res["margin"].setdefault(arm, {})[l] = {"median_proj": float(np.median(p)), "refusal_rate": float(ref.mean()),
                                                    "auroc_proj_to_refusal": _auroc(p, ref.astype(int))}
    write_json(OUT / "probe.json", res)
    return res


def figure(res: dict, results: dict | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"en": "#1f4e79", "ko": "#c0504d", "zh": "#9bbb59", "sw": "#8064a2"}
    names = {"en": "English", "ko": "Korean", "zh": "Mandarin", "sw": "Swahili (never trained)"}
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    L = range(res["layers"])
    for l in EVAL_LANGS:
        ax[0].plot(L, res["per_lang"][l]["cos_to_en"], color=colors[l], label=names[l])
        ax[1].plot(L, res["per_lang"][l]["auroc_en_direction"], color=colors[l], label=names[l])
    ax[0].set(title="Refusal direction: cosine to English", xlabel="layer", ylabel="cosine")
    ax[1].set(title="English direction separates harmful/harmless\n(held-out families)", xlabel="layer",
              ylabel="AUROC")
    ax[1].axvline(res["best_layer"], ls=":", c="grey")
    ax[0].legend(frameon=False)
    arms = list(res["drift"])
    w = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        vals = [res["drift"][arm][l]["gap_at_best"] for l in EVAL_LANGS]
        ax[2].bar(np.arange(4) + i * w, vals, w, label={"base": "Tiny Aya Global", "plain": "+ plain LoRA",
                                                        "safegrad": "+ SafeGrad LoRA"}.get(arm, arm),
                  color={"base": "#9aa5b1", "plain": "#d1495b", "safegrad": "#2e86ab"}.get(arm))
    ax[2].set_xticks(np.arange(4) + w * (len(arms) - 1) / 2, [names[l].split()[0] for l in EVAL_LANGS])
    ax[2].set(title=f"Harmful − harmless projection on base English\nrefusal direction (layer {res['best_layer']})",
              ylabel="projection gap")
    ax[2].legend(frameon=False)
    arm_c = {"base": "#9aa5b1", "plain": "#d1495b", "safegrad": "#2e86ab"}
    for arm, by_l in res.get("margin", {}).items():
        xs = [by_l[l]["median_proj"] for l in EVAL_LANGS]
        ys = [100 * by_l[l]["refusal_rate"] for l in EVAL_LANGS]
        ax[3].scatter(xs, ys, color=arm_c.get(arm), s=40, zorder=3)
        for l, x, y in zip(EVAL_LANGS, xs, ys):
            ax[3].annotate(l, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax[3].set(title="Refusal rate tracks the refusal signal\n(harmful prompts, median projection)",
              xlabel=f"median projection on base English direction (layer {res['best_layer']})",
              ylabel="refusal rate on attacks (%)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "analyze"])
    ap.add_argument("--arm")
    ap.add_argument("--adapter")
    ap.add_argument("--arms", nargs="+", default=["base"])
    ap.add_argument("--limit", type=int, help="quick test on the first N families per set")
    a = ap.parse_args()
    global PREFIX, OUT
    if a.limit:
        PREFIX, OUT = "test-", RUNS / "acts" / "test-results"
    if a.cmd == "extract":
        extract(a.arm, a.adapter, a.limit, f"{PREFIX}{a.arm}")
    else:
        res = analyze(a.arms)
        figure(res, None, OUT / "refusal_geometry.png")
        print({k: v for k, v in res.items() if k in ("best_layer", "probes")})


if __name__ == "__main__":
    main()
