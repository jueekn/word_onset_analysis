"""plot_mixed.py -- figures from mixed_models.R (word-on contrast; run that first).

  latency   estimated high-gamma latency per lobe (mixed model, mean +/- SE), lobes ordered by
            latency; brackets: lobe pairs with Holm p < .05
            -> figures/latency/hg_latency_word_on_mixed.png
  bands     left: estimated d per lobe for alpha and high gamma (mean +/- SE; stars: d != 0, Holm
            over the 12); right: lobe x lobe matrix of the alpha-vs-high-gamma pattern difference
            (lobe x band interaction contrast); stars: Holm p
            -> figures/burke_roi_power/roi_power_word_on_mixed.png

    python plot_mixed.py
"""
from __future__ import annotations

from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fc_comparison_functions as fc

LAT = join("figures", "latency", "hg_latency_word_on_mixed")
BANDS = join("figures", "burke_roi_power", "roi_power_word_on_mixed")


def save(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"[saved] {stem}.png")


def latency() -> None:
    e = pd.read_csv(f"{LAT}_lobes.csv").sort_values("emmean").reset_index(drop=True)
    pr = pd.read_csv(f"{LAT}_pairs.csv")
    pos = {lb: i for i, lb in enumerate(e["lobe"])}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(e.index, e["emmean"], yerr=e["SE"], capsize=4, edgecolor="0.15",
           color=[fc.LOBE_COLORS[lb] for lb in e["lobe"]])
    sig = []
    for c, p in zip(pr["contrast"], pr["p.value"]):
        a, b = sorted((pos[x] for x in c.split(" - ")))
        if p < 0.05:
            sig.append((b - a, a, b, p))
    y0, step = float((e["emmean"] + e["SE"]).max()) * 1.05, float(e["emmean"].max()) * 0.08
    for k, (_, a, b, p) in enumerate(sorted(sig)):   # shortest brackets lowest
        y = y0 + k * step
        ax.plot([a, a, b, b], [y - step * 0.25, y, y, y - step * 0.25], color="0.2", lw=1)
        ax.annotate(fc.stars(p), ((a + b) / 2, y), ha="center", va="bottom", fontsize=11)
    ax.set_xticks(e.index, [lb.capitalize() for lb in e["lobe"]], rotation=30, ha="right", fontsize=11)
    for lab, lb in zip(ax.get_xticklabels(), e["lobe"]):
        lab.set_color(fc.LOBE_COLORS[lb])
    ax.set_ylabel("High-gamma half-max latency (ms)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, LAT)


def bands() -> None:
    from statsmodels.stats.multitest import multipletests
    e = pd.read_csv(f"{BANDS}_lobes.csv")
    e["q"] = multipletests(e["p.value"], method="holm")[1]
    pr = pd.read_csv(f"{BANDS}_pairs.csv")
    lobes = list(fc.LOBES)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for k, (band, hatch) in enumerate([("alpha", ""), ("high_gamma", "//")]):
        s = e[e["band"] == band].set_index("lobe").reindex(lobes)
        x = np.arange(len(lobes)) + (k - 0.5) * 0.38
        a1.bar(x, s["emmean"], 0.38, yerr=s["SE"], capsize=3, edgecolor="0.15", hatch=hatch,
               color=[fc.LOBE_COLORS[lb] for lb in lobes], label={"alpha": "Alpha", "high_gamma": "High gamma"}[band])
        for xi, m, se, q in zip(x, s["emmean"], s["SE"], s["q"]):
            a1.annotate(fc.stars(q), (xi, m + np.sign(m) * se), textcoords="offset points",
                        xytext=(0, 3 if m >= 0 else -12), ha="center", fontsize=10)
    a1.axhline(0, color="0.6", lw=0.8)
    a1.set_xticks(range(len(lobes)), [lb.capitalize() for lb in lobes], rotation=30, ha="right", fontsize=11)
    for lab, lb in zip(a1.get_xticklabels(), lobes):
        lab.set_color(fc.LOBE_COLORS[lb])
    a1.set_ylabel("Power change, Cohen's d (mixed model)", fontsize=11)
    a1.legend(frameon=False, fontsize=10)
    a1.spines[["top", "right"]].set_visible(False)

    M = np.full((len(lobes), len(lobes)), np.nan)
    q = M.copy()
    for pair, est, p in zip(pr["lobe_pairwise"], pr["estimate"], pr["p.value"]):
        a, b = (lobes.index(x) for x in pair.split(" - "))
        i, j = max(a, b), min(a, b)
        M[i, j], q[i, j] = abs(est), p   # lower triangle: size of the pattern difference
    M, q = M[1:, :-1], q[1:, :-1]   # drop the empty first row / last column
    rows, cols = lobes[1:], lobes[:-1]
    im = a2.imshow(M, cmap="magma_r", vmin=0)
    for (i, j), p in np.ndenumerate(q):
        if np.isfinite(p) and fc.stars(p):
            a2.annotate(fc.stars(p), (j, i), ha="center", va="center", fontsize=10,
                        color="white" if M[i, j] > np.nanmax(M) / 2 else "black")
    a2.set_xticks(range(len(cols)), [lb.capitalize() for lb in cols], rotation=45, ha="right", fontsize=11)
    a2.set_yticks(range(len(rows)), [lb.capitalize() for lb in rows], fontsize=11)
    for lab, lb in zip([*a2.get_xticklabels(), *a2.get_yticklabels()], cols + rows):
        lab.set_color(fc.LOBE_COLORS[lb])
    a2.spines[:].set_visible(False)
    fig.colorbar(im, ax=a2, shrink=0.8).set_label("Alpha vs high-gamma pattern difference (|d|)", fontsize=10)
    save(fig, BANDS)


if __name__ == "__main__":
    latency()
    bands()
