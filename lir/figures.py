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


LANG2 = {"en": "English", "ko": "Korean", "zh": "Chinese", "sw": "Swahili\n(not trained on)"}
# H1: Korean against English under plain. H2: Korean, SafeGrad against plain. H3: Swahili refusals under plain.
READ = {("strict", "ko", "plain"), ("strict", "en", "plain"), ("strict", "ko", "safegrad"), ("refusal", "sw", "plain")}


def drift_figure(res: dict, path) -> None:
    """Change against base by language under plain and under SafeGrad: the mean of the seeds with its interval, and
    each seed as a tick. Left panel strict attack success, right panel the refusal rate on the same prompts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    panels = (("strict", "Strict attack success", "more attacks succeed  \u2192"),
              ("refusal", "Refusal rate on attack prompts", "\u2190  fewer refusals"))
    series = (("plain", "Plain LoRA", STAGE2["plain"], -0.17), ("safegrad", "SafeGrad LoRA", STAGE2["safegrad"], 0.17))
    langs = list(LANG2)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0), sharey=True, facecolor=SURFACE)
    for ax, (metric, title, hint) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        for key, _, color, dy in series:
            for y, lang in enumerate(langs):
                c = res[metric][key][lang]
                ax.plot(c["ci95"], [y + dy] * 2, color=color, lw=1.6, solid_capstyle="round", zorder=2)
                ax.plot(list(c["seeds"].values()), [y + dy] * len(c["seeds"]), "|", ms=12, mew=1.4, color=color, zorder=3)
                ax.plot(c["mean"], y + dy, "o", ms=8.5, mfc=color, mec=SURFACE, mew=1.6, zorder=4)
                if (metric, lang, key) in READ:   # the marks H1 to H3 compare
                    right = c["mean"] >= 0
                    ax.annotate(f"{c['mean']:+.1f}".replace("-", "\u2212"), (c["ci95"][1] if right else c["ci95"][0], y + dy),
                                xytext=(7 if right else -7, 0), textcoords="offset points", fontsize=9, color=INK,
                                va="center", ha="left" if right else "right")
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
        ax.set_xlabel(hint, fontsize=8.5, color=INK2)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.margins(x=0.12)
        ax.tick_params(colors=INK2, length=0, labelsize=9)
        for edge in ("top", "right", "left", "bottom"):
            ax.spines[edge].set_visible(False)
    axes[0].set_yticks(range(len(langs)), [LANG2[l] for l in langs])
    axes[0].tick_params(axis="y", labelsize=10, labelcolor=INK)
    axes[0].set_ylim(len(langs) - 0.45, -0.55)
    n = len(res["seeds"])
    fig.supxlabel("Change against the base model, in points. Tiny Aya Global, 291 MultiJail prompts per language, answers "
                  f"capped at 256 tokens.\nDots: mean of seeds {', '.join(map(str, res['seeds']))}. Lines: 95% intervals over "
                  "prompts. Ticks: single seeds. The numbered marks are the ones H1 to H3 compare.", fontsize=8.5, color=INK2)
    handles = [Line2D([], [], marker="o", ms=8.5, mfc=c, mec=SURFACE, mew=1.6, color=c, lw=1.6, label=lab)
               for _, lab, c, _ in series]
    handles.append(Line2D([], [], marker="|", ms=12, mew=1.4, color=INK2, lw=0, label=f"one seed ({n} per arm)"))
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.985, 0.995))
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    print("wrote", path)


def settings_figure(rows: list[dict], path) -> None:
    """Korean strict attack success against the setting's own base model, one row per training setting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    series = (("plain", "Plain LoRA", STAGE2["plain"], -0.17), ("safegrad", "SafeGrad LoRA", STAGE2["safegrad"], 0.17))
    fig, ax = plt.subplots(figsize=(10.6, 1.6 + 0.66 * len(rows)), facecolor=SURFACE)   # 4.9 high for five rows
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    for key, _, color, dy in series:
        for y, row in enumerate(rows):
            c = row[key]
            if c is None:
                continue
            ax.plot(c["ci95"], [y + dy] * 2, color=color, lw=1.6, solid_capstyle="round", zorder=2)
            if len(c["runs"]) > 1:
                ax.plot(c["runs"], [y + dy] * len(c["runs"]), "|", ms=12, mew=1.4, color=color, zorder=3)
            ax.plot(c["mean"], y + dy, "o", ms=8.5, mfc=color, mec=SURFACE, mew=1.6, zorder=4)
            ax.annotate(f"{c['mean']:+.1f}".replace("-", "\u2212"), (c["ci95"][1], y + dy), xytext=(7, 0),
                        textcoords="offset points", fontsize=9, color=INK, va="center", ha="left")
    runs = lambda row: len(row["plain"]["runs"])
    ax.set_yticks(range(len(rows)), [f"{row['setting']}\n{runs(row)} seed{'s' if runs(row) > 1 else ''}, base "
                                     f"{row['base']:.1f}%" for row in rows])
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.tick_params(axis="y", labelsize=9.5, labelcolor=INK)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.margins(x=0.1)
    ax.set_xlim(left=min(-2.5, ax.get_xlim()[0]))
    for edge in ("top", "right", "left", "bottom"):
        ax.spines[edge].set_visible(False)
    ax.set_title("Korean strict attack success after fine-tuning, against the same model before it", fontsize=11,
                 color=INK, loc="left", pad=8)
    fig.supxlabel("Change in points on 291 Korean MultiJail prompts, answers capped at 256 tokens. Dots: mean of the "
                  "setting's seeds. Lines: 95% intervals over prompts.\nTicks: single seeds. Rows without a blue mark had "
                  "no SafeGrad run. Each row's own registered test is in its table.", fontsize=8.5, color=INK2)
    handles = [Line2D([], [], marker="o", ms=8.5, mfc=c, mec=SURFACE, mew=1.6, color=c, lw=1.6, label=lab)
               for _, lab, c, _ in series]
    handles.append(Line2D([], [], marker="|", ms=12, mew=1.4, color=INK2, lw=0, label="one seed"))
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.985, 0.995))
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    print("wrote", path)


LANG10 = {"en": "English", "ko": "Korean", "zh": "Chinese", "sw": "Swahili \u2021", "it": "Italian \u2020", "vi": "Vietnamese \u2020",
          "ar": "Arabic \u2020", "th": "Thai \u2020", "bn": "Bengali \u2020", "jv": "Javanese \u2020"}
TRAINED = ("en", "ko", "zh")


def ten_figure(res: dict, path) -> None:
    """The drift figure over ten languages: the three the model was fine-tuned on, then the seven it was not. Change
    against base under plain and under SafeGrad, mean of the seeds with its interval and each seed as a tick."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    panels = (("strict", "Strict attack success, as the guard measures it", "more attacks succeed  \u2192"),
              ("refusal", "Refusal rate on attack prompts", "\u2190  fewer refusals"))
    series = (("plain", "Plain LoRA", STAGE2["plain"], -0.17), ("safegrad", "SafeGrad LoRA", STAGE2["safegrad"], 0.17))
    langs = [l for l in res["languages"] if l in TRAINED] + [l for l in res["languages"] if l not in TRAINED]
    n_trained = sum(l in TRAINED for l in langs)
    ys = {l: i + (1.2 if i >= n_trained else 0) for i, l in enumerate(langs)}       # a gap between the two groups
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 1.9 + 0.56 * len(langs)), sharey=True, facecolor=SURFACE)
    for ax, (metric, title, hint) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        for key, _, color, dy in series:
            for lang in langs:
                c, y = res[metric][key][lang], ys[lang] + dy
                ax.plot(c["ci95"], [y] * 2, color=color, lw=1.6, solid_capstyle="round", zorder=2)
                ax.plot(list(c["seeds"].values()), [y] * len(c["seeds"]), "|", ms=11, mew=1.4, color=color, zorder=3)
                ax.plot(c["mean"], y, "o", ms=8, mfc=color, mec=SURFACE, mew=1.6, zorder=4)
                if key == "plain":   # one number per language and panel: the change under plain fine-tuning
                    right = c["mean"] >= 0
                    ax.annotate(f"{c['mean']:+.1f}".replace("-", "\u2212"), (c["ci95"][1] if right else c["ci95"][0], y),
                                xytext=(7 if right else -7, 0), textcoords="offset points", fontsize=8.5, color=INK,
                                va="center", ha="left" if right else "right")
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
        ax.set_xlabel(hint, fontsize=8.5, color=INK2)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.margins(x=0.14)
        ax.tick_params(colors=INK2, length=0, labelsize=9)
        for edge in ("top", "right", "left", "bottom"):
            ax.spines[edge].set_visible(False)
    axes[0].set_yticks([ys[l] for l in langs], [LANG10.get(l, l) for l in langs])
    axes[0].tick_params(axis="y", labelsize=10, labelcolor=INK)
    axes[0].set_ylim(max(ys.values()) + 0.55, -1.25)
    for y, text in ((-0.85, "Fine-tuned on"), (n_trained + 0.35, "Never fine-tuned on")):
        axes[0].text(-0.02, y, text, transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontsize=8.5,
                     color=INK2, style="italic")
    n = len(res["seeds"])
    fig.supxlabel("Change against the base model, in points. Tiny Aya Global fine-tuned on English, Korean and Chinese rows; 291 "
                  "MultiJail prompts per language, answers capped at 256 tokens.\n"
                  f"Dots: mean of seeds {', '.join(map(str, res['seeds']))}. Lines: 95% intervals over prompts. Ticks: single "
                  "seeds. Numbers: plain LoRA.\n"
                  "\u2020 No person has checked the guard in this language.  \u2021 Against a person's labels the guard "
                  "missed our agreement bar in Swahili.",
                  fontsize=8.5, color=INK2)
    handles = [Line2D([], [], marker="o", ms=8, mfc=c, mec=SURFACE, mew=1.6, color=c, lw=1.6, label=lab) for _, lab, c, _ in series]
    handles.append(Line2D([], [], marker="|", ms=11, mew=1.4, color=INK2, lw=0, label=f"one seed ({n} per arm)"))
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.985, 0.998))
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    print("wrote", path)
