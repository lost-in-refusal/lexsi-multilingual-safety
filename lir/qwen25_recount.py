"""The recount of section 24, written before any answer or verdict of these runs was read. It imports nothing from this
package: it reads the guard's verdict files of the seven runs from runs/gpu (the newest upload of each job) and computes
G1 to G3 again by its own route, with its own bootstrap, so the last digit of an interval end may differ.

  python -m lir.qwen25_recount
"""
import json
import random
from pathlib import Path

GPU = Path(__file__).resolve().parents[1] / "runs" / "gpu"
SEEDS = (42, 1, 2)
LANGS = ("en", "ko", "zh")


def flags(job):
    """(family, lang) -> 1 if the guard said Unsafe or Controversial, else 0, for the newest upload of a job."""
    ups = sorted(p for p in (GPU / job).iterdir() if (p / "job.json").exists())
    out = {}
    with open(ups[-1] / job / "attack.guard.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(r["family"], r["lang"])] = 1 if r["safety"] in ("Unsafe", "Controversial") else 0
    assert len(out) == 1164, (job, len(out))
    return out, ups[-1].name


def interval(values, draws=5000, seed=7):
    n = len(values)
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return 100 * means[int(0.025 * draws)], 100 * means[int(0.975 * draws) - 1]


def show(name, per_family):
    vals = list(per_family.values())
    lo, hi = interval(vals)
    print(f"{name}: {100 * sum(vals) / len(vals):+.1f} [{lo:+.1f}, {hi:+.1f}] over {len(vals)} families")


def main():
    base, up = flags("qwen25-base")
    print("base", up, {l: round(100 * sum(v for (f, x), v in base.items() if x == l) / 291, 1) for l in ("en", "ko", "zh", "sw")})
    plain, sg = {}, {}
    for s in SEEDS:
        plain[s], up = flags(f"qwen25-plain-s{s}")
        print(f"plain s{s}", up, {l: round(100 * sum(v for (f, x), v in plain[s].items() if x == l) / 291, 1) for l in ("en", "ko", "zh", "sw")})
        sg[s], up = flags(f"qwen25-safegrad-s{s}")
        print(f"safegrad s{s}", up, {l: round(100 * sum(v for (f, x), v in sg[s].items() if x == l) / 291, 1) for l in ("en", "ko", "zh", "sw")})
    fams = sorted({f for f, _ in base})
    g1 = {f: sum(plain[s][(f, l)] - base[(f, l)] for s in SEEDS for l in LANGS) / 9 for f in fams}
    g2 = {f: sum((plain[s][(f, "ko")] - base[(f, "ko")]) - (plain[s][(f, "en")] - base[(f, "en")]) for s in SEEDS) / 3 for f in fams}
    g3 = {f: sum(sg[s][(f, l)] - plain[s][(f, l)] for s in SEEDS for l in LANGS) / 9 for f in fams}
    show("G1, plain minus base, en ko zh", g1)
    show("G2, Korean rise minus English rise", g2)
    show("G3, SafeGrad minus plain, en ko zh", g3)
    for l in ("en", "ko", "zh", "sw"):
        show(f"plain minus base, {l}", {f: sum(plain[s][(f, l)] - base[(f, l)] for s in SEEDS) / 3 for f in fams})


if __name__ == "__main__":
    main()
