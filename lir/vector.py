"""Exploratory, not preregistered: SafeGrad's repair applied as a weight edit, with no training.

  python -m lir.vector build --seed 1      # runs/plain-s1-vec/adapter
  python -m lir.vector report              # results/lir/stage2/vector.{json,md}

The repair vector is the SafeGrad adapter's weight change minus the plain adapter's, both
from seed 42. It is added at full size to the plain adapter of another seed, which never saw
the refusal examples. The design was fixed before the first run: the vector comes from seed
42 only, it is tested on seeds 1 and 2, and nothing is tuned on the attack set.

The sum of three rank-16 changes is written as one rank-48 LoRA adapter (the factors side by
side, same scaling), so the edited model goes through the same evaluation code as every arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lir.common import EVAL_LANGS, ROOT, RUNS, git_commit, sha256_file, write_json

SOURCE = ("safegrad", "plain")  # seed 42: the vector is the first minus the second
TEST_SEEDS = (1, 2)
WEIGHTS = "adapter_model.safetensors"


def combine(base: dict, plus: dict, minus: dict) -> dict:
    """LoRA factors whose product is base + plus - minus, for adapters that share one scaling."""
    import torch

    out = {}
    for k in base:
        if k.endswith("lora_A.weight"):
            out[k] = torch.cat([base[k], plus[k], minus[k]], dim=0)
        elif k.endswith("lora_B.weight"):
            out[k] = torch.cat([base[k], plus[k], -minus[k]], dim=1)
        else:
            raise SystemExit(f"unexpected tensor in the adapter: {k}")
    return out


def build(seed: int) -> Path:
    from safetensors.torch import load_file, save_file

    names = {"base": f"plain-s{seed}", "plus": SOURCE[0], "minus": SOURCE[1]}
    dirs = {k: RUNS / v / "adapter" for k, v in names.items()}
    cfgs = {k: json.loads((d / "adapter_config.json").read_text()) for k, d in dirs.items()}
    same = ("base_model_name_or_path", "r", "lora_alpha", "use_dora", "use_rslora", "target_modules")
    for k in ("plus", "minus"):
        if any(cfgs[k][f] != cfgs["base"][f] for f in same if f != "target_modules") or \
                sorted(cfgs[k]["target_modules"]) != sorted(cfgs["base"]["target_modules"]):
            raise SystemExit(f"{names[k]} and {names['base']} were not trained with the same adapter settings")
    if cfgs["base"]["use_dora"] or cfgs["base"]["use_rslora"]:
        raise SystemExit("the factors can only be stacked for plain LoRA scaling")
    tensors = {k: load_file(d / WEIGHTS) for k, d in dirs.items()}
    if not (tensors["base"].keys() == tensors["plus"].keys() == tensors["minus"].keys()):
        raise SystemExit("the adapters do not cover the same modules")
    out = RUNS / f"plain-s{seed}-vec" / "adapter"
    out.mkdir(parents=True, exist_ok=True)
    save_file(combine(**tensors), out / WEIGHTS, metadata={"format": "pt"})
    cfg = dict(cfgs["base"], r=3 * cfgs["base"]["r"], lora_alpha=3 * cfgs["base"]["lora_alpha"])
    write_json(out / "adapter_config.json", cfg)
    write_json(out.parent / "run.json", {
        "arm": f"plain-s{seed}-vec", "edit": "plain adapter + (SafeGrad adapter - plain adapter, seed 42)",
        "sources": {k: {"run": names[k], "sha256": sha256_file(dirs[k] / WEIGHTS)} for k in names},
        "sha256": sha256_file(out / WEIGHTS), "commit": git_commit()})
    print("wrote", out)
    return out


def report() -> dict:
    from lir.metrics import load
    from lir.seeds import STRICT, by_family, change, macro, rate

    base = by_family(load("base", "attack"), STRICT)
    res = {"note": "Exploratory, single runs, not preregistered.", "seeds": {}}
    for seed in TEST_SEEDS:
        arms = {"plain": f"plain-s{seed}", "edited": f"plain-s{seed}-vec", "safegrad": f"safegrad-s{seed}"}
        v = {k: by_family(load(a, "attack"), STRICT) for k, a in arms.items()}
        ko = {k: rate(d, "ko") for k, d in v.items()}
        gap = ko["plain"] - ko["safegrad"]
        res["seeds"][seed] = {
            "strict": {"base": {l: rate(base, l) for l in EVAL_LANGS}, **{k: {l: rate(d, l) for l in EVAL_LANGS} for k, d in v.items()}},
            "macro": {"base": macro(base), **{k: macro(d) for k, d in v.items()}},
            "edited_vs_plain": {l: change(v["plain"], v["edited"], l) for l in EVAL_LANGS},
            "edited_vs_safegrad": {l: change(v["safegrad"], v["edited"], l) for l in EVAL_LANGS},
            "edited_vs_base": {l: change(base, v["edited"], l) for l in EVAL_LANGS},
            "ko_share_of_safegrad_repair": (ko["plain"] - ko["edited"]) / gap if gap else None}
    star = lambda c: "*" if c["p"] < 0.05 else ""
    lines = ["### Repair vector added to another seed's plain model: strict attack success (%)", "",
             "Exploratory and not preregistered. Single runs. The vector is SafeGrad minus plain at seed 42; "
             "no training and no tuning on these prompts.", "",
             "| Seed | Arm | en | ko | zh | sw | macro |", "|---|---|---:|---:|---:|---:|---:|"]
    for seed, r in res["seeds"].items():
        for arm in ("base", "plain", "safegrad"):
            lines.append(f"| {seed} | {arm} | " + " | ".join(f"{r['strict'][arm][l]:.1f}" for l in EVAL_LANGS)
                         + f" | {r['macro'][arm]:.1f} |")
        lines.append(f"| {seed} | plain + vector | " + " | ".join(
            f"{r['strict']['edited'][l]:.1f} ({r['edited_vs_plain'][l]['delta']:+.1f}{star(r['edited_vs_plain'][l])})"
            for l in EVAL_LANGS) + f" | {r['macro']['edited']:.1f} |")
    lines += ["", "Parentheses: change against the plain model of the same seed. * = p < 0.05, exact McNemar.", ""]
    for seed, r in res["seeds"].items():
        s, d = r["ko_share_of_safegrad_repair"], r["edited_vs_safegrad"]["ko"]
        lines.append(f"Seed {seed}, Korean: the edit gives {'n/a' if s is None else f'{100 * s:.0f}%'} of the drop that "
                     f"training with SafeGrad gave; edited against SafeGrad {d['delta']:+.1f} points (p = {d['p']:.3f}).")
    out = ROOT / "results" / "lir" / "stage2"
    write_json(out / "vector.json", res)
    (out / "vector.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "report"])
    ap.add_argument("--seed", type=int, choices=TEST_SEEDS)
    a = ap.parse_args()
    if a.cmd == "build":
        if a.seed is None:
            raise SystemExit("--seed is required for build")
        build(a.seed)
    else:
        report()


if __name__ == "__main__":
    main()
