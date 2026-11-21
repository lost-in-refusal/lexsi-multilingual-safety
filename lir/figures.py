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


# Stage 2 figures. Series colours were checked for colour-blind separation and contrast on this surface.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STAGE2 = {"plain": "#e34948", "safegrad": "#2a78d6"}
SHORT = {"Non-violent unethical behavior (e.g. lying, cheating, etc.)": "Non-violent unethical behavior",
         "Soliciting personally identifiable information": "Soliciting personal information",
         "Sexual exploitation & human trafficking": "Sexual exploitation & trafficking",
         "Conspiracy theories & misinformation": "Conspiracy & misinformation",
         "Substance abuse & banned substances": "Substance abuse"}


def harm_figure(res: dict, path) -> None:
    """Change in strict attack success against base, by harm category, under plain and under SafeGrad."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cats = list(res["categories"].items())
    n_big = sum(c["interpreted"] for _, c in cats)
    ys = [i + (1.2 if i >= n_big else 0) for i in range(len(cats))]  # a gap before the small categories
    series = [("plain_vs_base", "Plain LoRA", STAGE2["plain"], -0.18), ("safegrad_vs_base", "SafeGrad LoRA", STAGE2["safegrad"], 0.18)]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 7.4), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, (lang, title) in zip(axes, (("ko", "Korean"), ("all", "Four languages together"))):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        for key, _, color, dy in series:
            for (_, c), y in zip(cats, ys):
                v = c["langs"][lang][key]
                if c["interpreted"]:
                    ax.plot(v["ci95"], [y + dy] * 2, color=color, lw=1.6, solid_capstyle="round", zorder=2)
                    ax.plot(v["mean"], y + dy, "o", ms=8.5, mfc=color, mec=SURFACE, mew=1.6, zorder=3)
                else:
                    ax.plot(v["mean"], y + dy, "o", ms=6.5, mfc=SURFACE, mec=color, mew=1.5, zorder=3)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
        ax.tick_params(colors=INK2, length=0, labelsize=9)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    ax0 = axes[0]
    ax0.set_yticks(ys, [f"{SHORT.get(t, t)}  ({c['prompts']})" for t, c in cats])
    ax0.tick_params(axis="y", labelsize=9, labelcolor=INK)
    ax0.set_ylim(ys[-1] + 0.8, -0.8)
    mid = (ys[n_big - 1] + ys[n_big]) / 2
    for ax in axes:
        ax.axhline(mid, color=GRID, lw=1)
    ax0.annotate(f"fewer than {res['min_prompts']} prompts: shown without intervals, not interpreted", (0, mid),
                 xycoords=("axes fraction", "data"), xytext=(4, -5), textcoords="offset points", fontsize=8, color=INK2,
                 va="top", zorder=5, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    fig.supxlabel("Change in strict attack success against the base model, in points\n"
                  f"mean of seeds {', '.join(map(str, res['seeds']))}; lines are 95% intervals over the category's prompts; "
                  "prompts per category in brackets", fontsize=9, color=INK2)
    handles = [Line2D([], [], marker="o", ms=8.5, mfc=c, mec=SURFACE, mew=1.6, color=c, lw=1.6, label=lab)
               for _, lab, c, _ in series]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.985, 0.995))
    fig.tight_layout(rect=(0, 0.01, 1, 0.965))
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    print("wrote", path)
