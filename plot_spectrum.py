"""plot_spectrum.py -- word on vs off power per Burke ROI x log-spaced frequency bin.

Per electrode the Cohen's d of multitaper power in 11 log-spaced bins (5-150 Hz,
mains 50/60/100/120 +/- 2 Hz left out; compute.py); sessions averaged per electrode, electrodes
per subject and ROI. As DeHaan et al. 2025: one-sample t vs 0 across subjects per cell (colour),
BH-FDR over all ROI x bin cells (q < .05); outlines enclose connected significant cells.
-> figures/burke_roi_power/power_spectrum_word_on_rois.png
Also per anatomical subregion (hemispheres pooled, >= MIN_SUBJECTS_ROI subjects), rows grouped
by lobe, FDR over all subregion x bin cells -> power_spectrum_word_on_fine.png (supplement)

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


def subject_table(d: pd.DataFrame, col: str, min_electrodes: int) -> pd.DataFrame:
    """sessions -> electrode, electrodes -> subject x `col` region x bin mean d."""
    x = d.dropna(subset=[col]).groupby(["sub", "label", col, "bin"], as_index=False)["d"].mean()
    g = x.groupby(["sub", col, "bin"]).agg(d=("d", "mean"), n=("label", "nunique")).reset_index()
    g = g[g["n"] >= min_electrodes].rename(columns={col: "roi"})
    g["freq_hz"] = CENTRES[g["bin"]]
    return g


def bin_t(d: pd.DataFrame, col: str, min_electrodes: int) -> pd.DataFrame:
    """t per region x bin, BH-FDR over all cells (fc.bin_stats)."""
    return fc.bin_stats(subject_table(d, col, min_electrodes), "freq_hz", "d")


def outline(ax, sig: np.ndarray) -> None:
    """Border around connected significant cells: every cell edge between a significant
    cell and a non-significant (or missing) neighbour."""
    from matplotlib.collections import LineCollection
    S = np.pad(sig, 1)
    segs = []
    for i, j in zip(*np.nonzero(sig)):
        y, x = i - 0.5, j - 0.5
        if not S[i, j + 1]: segs.append([(x, y), (x + 1, y)])           # top
        if not S[i + 2, j + 1]: segs.append([(x, y + 1), (x + 1, y + 1)])   # bottom
        if not S[i + 1, j]: segs.append([(x, y), (x, y + 1)])           # left
        if not S[i + 1, j + 2]: segs.append([(x + 1, y), (x + 1, y + 1)])   # right
    ax.add_collection(LineCollection(segs, colors="black", linewidths=1.5))


def t_label() -> str:
    c = fc.contrast(BEH)
    return f"t ({c['delta_label']})"


def heatmap(M: np.ndarray, q: np.ndarray, labels: list, stem: str, figsize: tuple,
            lobes: list, gaps: bool = False) -> None:
    """t map with outlined q < .05 cells; labels coloured by `lobes` (one per row);
    gaps: white lines between lobes."""
    v = float(np.nanmax(np.abs(M))) or 1.0
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    outline(ax, q < 0.05)
    ax.set_xticks(range(len(CENTRES)), [f"{x:.0f}" for x in CENTRES], fontsize=13)
    ax.set_yticks(range(len(labels)), labels, fontsize=13 if not gaps else 11)
    for lab, lb in zip(ax.get_yticklabels(), lobes):
        lab.set_color(fc.LOBE_COLORS[lb])
        lab.set_fontweight("bold")
    lobes = np.array(lobes)
    for i in np.flatnonzero(lobes[1:] != lobes[:-1]) if gaps else []:
        ax.axhline(i + 0.5, color="white", lw=3)
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    fig.colorbar(im, ax=ax).set_label(t_label(), fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {stem}.png / .pdf")


def run(g: pd.DataFrame, rows: list, stem: str) -> tuple:
    """Per-cell t across subjects, BH-FDR over all rows x bins; writes <stem>_stats.csv;
    returns (t, q) as rows x bins."""
    st = fc.bin_stats(g[g["roi"].isin(rows)], "freq_hz", "d", fdr="fdr_bh")
    st.to_csv(fc.table_path(f"{stem}_stats.csv"), index=False)
    st["bin"] = np.searchsorted(CENTRES, st["freq_hz"])
    M, q = (st.pivot(index="roi", columns="bin", values=c).reindex(index=rows, columns=range(len(CENTRES))).to_numpy(float)
            for c in ("t", "q"))
    print(f"[fdr] {os.path.basename(stem)}: {int(np.nansum(q < 0.05))} of {int(np.isfinite(q).sum())} cells q < .05")
    return M, q


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "burke_roi_power"))
    d = spectrum_table(save_root, fc.load_burke_maps(), args.n_sessions)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = [f"{h}-{lobe}" for lobe in fc.LOBES for h in "LR"]   # left / right of each region together
    stem = join(args.out_dir, f"power_spectrum_{BEH}_rois")
    M, q = run(subject_table(d, "roi", args.min_electrodes), rows, stem)
    heatmap(M, q, [fc.pretty_roi(x) for x in rows], stem, (7.5, 5.2), [r.split("-", 1)[1] for r in rows])

    # anatomical subregions (hemispheres pooled), >= MIN_SUBJECTS_ROI subjects, grouped by lobe
    d["region"] = d["fine"].str.replace(r"^[LR] ", "", regex=True)
    d["lobe"] = d["roi"].str.split("-", n=1).str[1]
    g = subject_table(d, "region", args.min_electrodes)
    n = g.groupby("roi")["sub"].nunique()
    lobe_of = d.dropna(subset=["region"]).groupby("region")["lobe"].first()
    rows = [r for lb in fc.LOBES for r in n[n >= fc.MIN_SUBJECTS_ROI].sort_values(ascending=False).index
            if lobe_of[r] == lb]
    stem = join(args.out_dir, f"power_spectrum_{BEH}_fine")
    M, q = run(g, rows, stem)
    heatmap(M, q, [f"{r} ({n[r]})" for r in rows], stem, (6.5, 0.24 * len(rows) + 1.3),
            [lobe_of[r] for r in rows], gaps=True)


if __name__ == "__main__":
    main()
