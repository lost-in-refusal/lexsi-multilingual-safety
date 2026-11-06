"""Headline figure from results/lir/results.json.

  python -m lir.figures --arms base plain safegrad
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from lir.common import EVAL_LANGS, ROOT

OUT = ROOT / "results" / "lir"
COLORS = {"base": "#9aa5b1", "plain": "#d1495b", "safegrad": "#2e86ab"}
LABELS = {"base": "Tiny Aya Global", "plain": "+ plain LoRA", "safegrad": "+ SafeGrad LoRA"}
LANG = {"en": "English", "ko": "Korean", "zh": "Mandarin", "sw": "Swahili\n(never trained)"}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["base", "plain", "safegrad"])
    a = ap.parse_args()
    res = json.loads((OUT / "results.json").read_text())
    panels = [("asr_strict", "Attack success rate (%)\nMultiJail, strict judge", True),
              ("over_refusal", "Over-refusal (%)\nXSTest safe prompts", True),
              ("task_score", "Task score (1–7)\nheld-out Bactrian-X", False)]
    panels = [p for p in panels if any(p[0] in res.get(arm, {}) for arm in a.arms)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (metric, title, pct) in zip(axes, panels):
        langs = [l for l in EVAL_LANGS if any(l in res.get(arm, {}).get(metric, {}) for arm in a.arms)]
        w = 0.8 / len(a.arms)
        for i, arm in enumerate(a.arms):
            cells = [res.get(arm, {}).get(metric, {}).get(l) for l in langs]
            vals = np.array([c["mean"] * (100 if pct else 1) if c else np.nan for c in cells])
            x = np.arange(len(langs)) + (i - (len(a.arms) - 1) / 2) * w
            err = None
            if pct:
                lo = np.array([c["ci95"][0] * 100 if c else np.nan for c in cells])
                hi = np.array([c["ci95"][1] * 100 if c else np.nan for c in cells])
                err = np.vstack([vals - lo, hi - vals])
            ax.bar(x, vals, w, yerr=err, capsize=2, color=COLORS.get(arm, "#777"), label=LABELS.get(arm, arm),
                   error_kw={"elinewidth": 0.8, "ecolor": "#333"})
        ax.set_xticks(np.arange(len(langs)), [LANG[l] for l in langs])
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        if not pct:
            ax.set_ylim(1, 7)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "headline.png", dpi=170)
    print("wrote", OUT / "headline.png")


if __name__ == "__main__":
    main()
