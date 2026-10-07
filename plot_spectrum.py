"""plot_spectrum.py -- t of word on vs off power per Burke ROI x log-spaced frequency bin.

Per electrode the Cohen's d of multitaper power in 10 log-spaced bins (5-100 Hz,
48-52 / 58-62 Hz left out; compute.py); sessions averaged per electrode, electrodes
per subject and ROI; one-sample t vs 0 across subjects per cell, BH-FDR over all
ROI x bin cells (fc.bin_stats). -> figures/burke_roi_power/power_spectrum_word_on_rois.png

    python plot_spectrum.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fc_comparison_functions as fc
from compute import FREQ_EDGES, power_dir

BEH = "word_on"
CENTRES = np.sqrt(FREQ_EDGES[:-1] * FREQ_EDGES[1:])


def spectrum_table(save_root: str, lobe_of: dict, n_sessions=None) -> pd.DataFrame:
    """Per (subject, electrode, bin): d, with its Burke ROI and anatomical label (fine)."""
    rows = []
    for f in fc.session_files(power_dir(save_root, BEH, "spectrum"), "_power.pkl", n_sessions):
        p = fc.load_pickle(str(f))
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        for lab, r, fine, dd in zip(p["labels"], roi, p["reg_full"], p["cohens_d_freq"]):
            rows += [(str(p["sid"][0]), str(lab), r, fine if isinstance(fine, str) else None, k, float(v))
                     for k, v in enumerate(dd)]
    return pd.DataFrame(rows, columns=["sub", "label", "roi", "fine", "bin", "d"])


def bin_t(d: pd.DataFrame, col: str, min_electrodes: int) -> pd.DataFrame:
    """sessions -> electrode, electrodes -> subject x `col` region; t per region x bin (fc.bin_stats)."""
    x = d.dropna(subset=[col]).groupby(["sub", "label", col, "bin"], as_index=False)["d"].mean()
    g = x.groupby(["sub", col, "bin"]).agg(d=("d", "mean"), n=("label", "nunique")).reset_index()
    g = g[g["n"] >= min_electrodes].rename(columns={col: "roi"})
    g["freq_hz"] = CENTRES[g["bin"]]
    return fc.bin_stats(g, "freq_hz", "d")


def t_label() -> str:
    c = fc.contrast(BEH)
    return f"t ({c['hi_label']} vs. {c['lo_label']})"


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "burke_roi_power"))
    st = bin_t(spectrum_table(save_root, fc.load_burke_maps(), args.n_sessions), "roi", args.min_electrodes)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"power_spectrum_{BEH}_rois")
    st.to_csv(f"{stem}_stats.csv", index=False)
    T = st.pivot(index="roi", columns="freq_hz", values="t").reindex(fc.ROI_ORDER)
    Q = st.pivot(index="roi", columns="freq_hz", values="q").reindex(fc.ROI_ORDER)
    v = float(np.nanmax(np.abs(T.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(11, 6.5))
    im = ax.imshow(T.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    for (i, j), q in np.ndenumerate(Q.to_numpy()):
        if fc.stars(q):
            ax.text(j, i, fc.stars(q), ha="center", va="center", fontsize=12)
    ax.set_xticks(range(len(T.columns)), [f"{x:.0f}" for x in T.columns], fontsize=13)
    ax.set_yticks(range(len(T.index)), [fc.pretty_roi(x) for x in T.index], fontsize=13)
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    fig.colorbar(im, ax=ax).set_label(t_label(), fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"[saved] {stem}.png / .pdf")


if __name__ == "__main__":
    main()
