"""plot_mixed.py -- figures from mixed_models.R (word-on contrast; run that first).

  latency   estimated high-gamma latency per lobe (mixed model, +/- SE), ordered by latency;
            brackets: neighbouring lobes (Holm over all 15 pairs; full table in the _pairs CSV)
            -> figures/latency/hg_latency_word_on_mixed.png
  bands     lobe x lobe matrix of the alpha-vs-high-gamma pattern difference (lobe x band
            interaction contrast, |d|); stars: Holm p
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
    e = pd.read_csv(fc.table_path(f"{LAT}_lobes.csv")).sort_values("emmean").reset_index(drop=True)
    pr = pd.read_csv(fc.table_path(f"{LAT}_pairs.csv"))
    pos = {lb: i for i, lb in enumerate(e["lobe"])}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(e.index, e["emmean"], yerr=e["SE"], capsize=4, edgecolor="0.15",
           color=[fc.LOBE_COLORS[lb] for lb in e["lobe"]])
    pval = {tuple(sorted(pos[x] for x in c.split(" - "))): p for c, p in zip(pr["contrast"], pr["p.value"])}
    top, pad = (e["emmean"] + e["SE"]).to_numpy(), float(e["emmean"].max()) * 0.05
    for a in range(len(e) - 1):   # neighbours in the latency order
        y = max(top[a], top[a + 1]) + pad
        ax.plot([a + 0.05, a + 0.05, a + 0.95, a + 0.95], [y - pad * 0.4, y, y, y - pad * 0.4], color="0.2", lw=1)
        ax.annotate(fc.stars(pval[(a, a + 1)]) or "n.s.", (a + 0.5, y), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(e.index, [lb.capitalize() for lb in e["lobe"]], rotation=30, ha="right", fontsize=11)
    for lab, lb in zip(ax.get_xticklabels(), e["lobe"]):
        lab.set_color(fc.LOBE_COLORS[lb])
    ax.set_ylabel("High-gamma half-max latency (ms)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, LAT)


def bands() -> None:
    pr = pd.read_csv(fc.table_path(f"{BANDS}_pairs.csv"))
    lobes = list(fc.LOBES)
    fig, a2 = plt.subplots(figsize=(6, 4.8))
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
