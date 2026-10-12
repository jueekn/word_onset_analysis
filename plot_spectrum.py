"""plot_spectrum.py -- t of word on vs off power per Burke ROI x log-spaced frequency bin.

Per electrode the Cohen's d of multitaper power in 10 log-spaced bins (5-100 Hz,
48-52 / 58-62 Hz left out; compute.py); sessions averaged per electrode, electrodes
per subject and ROI; one-sample t vs 0 across subjects per cell. Significance by
threshold-free cluster enhancement along frequency (TFCE, E=0.5, H=2) with
sign-flip permutations of whole subjects and max-statistic family-wise correction
over all ROI x bin cells (as Rao et al. 2025 / Ezzyat et al. 2017; rows are not
adjacent). Outlines mark runs of cells with p_FWE < .05.
-> figures/burke_roi_power/power_spectrum_word_on_rois.png

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


def nan_t(Y: np.ndarray) -> np.ndarray:
    """One-sample t over subjects (axis 0), ignoring NaN; NaN where < MIN_SUBJECTS_ROI subjects."""
    n = np.isfinite(Y).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.nanmean(Y, 0) / (np.nanstd(Y, 0, ddof=1) / np.sqrt(n))
    return np.where(n >= fc.MIN_SUBJECTS_ROI, t, np.nan)


def tfce_rows(T: np.ndarray, E: float = 0.5, H: float = 2.0, dh: float = 0.1) -> np.ndarray:
    """TFCE along the last axis (frequency) of each row, positive and negative separately:
    sum over thresholds h of extent(h)^E * h^H * dh, extent = contiguous run length."""
    out = np.zeros_like(T)
    for sign in (1, -1):
        X = np.nan_to_num(sign * T, nan=0.0)
        for h in np.arange(dh, X.max() + dh, dh) if X.max() > 0 else []:
            above = X >= h
            for r in range(X.shape[0]):
                j = 0
                while j < X.shape[1]:
                    if above[r, j]:
                        k = j
                        while k < X.shape[1] and above[r, k]:
                            k += 1
                        out[r, j:k] += sign * (k - j) ** E * h ** H * dh
                        j = k
                    else:
                        j += 1
    return out


def tfce_test(g: pd.DataFrame, rows: list, n_perm: int = 1000, seed: int = 0):
    """(t map, TFCE map, family-wise p map) over rows x bins; sign-flip permutation of whole
    subjects (each subject's map x +1/-1), max |TFCE| per permutation."""
    subs = sorted(g["sub"].unique())
    Y = np.full((len(subs), len(rows), len(CENTRES)), np.nan)
    si, ri = {s: i for i, s in enumerate(subs)}, {r: i for i, r in enumerate(rows)}
    for s, r, b, v in g[g["roi"].isin(ri)][["sub", "roi", "bin", "d"]].itertuples(index=False):
        Y[si[s], ri[r], b] = v
    T = nan_t(Y)
    obs = tfce_rows(T)
    rng = np.random.default_rng(seed)
    null = np.array([np.nanmax(np.abs(tfce_rows(nan_t(Y * rng.choice([-1.0, 1.0], len(subs))[:, None, None]))))
                     for _ in range(n_perm)])
    p = (1 + (null[None, None, :] >= np.abs(obs)[..., None]).sum(-1)) / (n_perm + 1)
    return T, obs, np.where(np.isfinite(T), p, np.nan)


def outline(ax, sig: np.ndarray) -> None:
    """Rectangle around each row's run of contiguous significant cells."""
    from matplotlib.patches import Rectangle
    for i, row in enumerate(sig):
        j = 0
        while j < len(row):
            if row[j]:
                k = j
                while k < len(row) and row[k]:
                    k += 1
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), k - j, 1, fill=False, lw=2.2, ec="black"))
                j = k
            else:
                j += 1


def t_label() -> str:
    c = fc.contrast(BEH)
    return f"t ({c['hi_label']} vs. {c['lo_label']})"


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "burke_roi_power"))
    g = subject_table(spectrum_table(save_root, fc.load_burke_maps(), args.n_sessions), "roi", args.min_electrodes)
    rows = [f"{h}-{lobe}" for lobe in fc.LOBES for h in "LR"]   # left / right of each region together
    T, tfce, p = tfce_test(g, rows)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"power_spectrum_{BEH}_rois")
    st = fc.bin_stats(g, "freq_hz", "d")
    tf = pd.DataFrame([(r, CENTRES[b], tfce[i, b], p[i, b]) for i, r in enumerate(rows) for b in range(len(CENTRES))],
                      columns=["roi", "freq_hz", "tfce", "p_fwe_tfce"])
    st.merge(tf, on=["roi", "freq_hz"], how="left").to_csv(f"{stem}_stats.csv", index=False)
    print(f"[tfce] {int(np.nansum(p < 0.05))} of {int(np.isfinite(p).sum())} cells p_FWE < .05")
    v = float(np.nanmax(np.abs(T))) or 1.0
    fig, ax = plt.subplots(figsize=(11, 6.5))
    im = ax.imshow(T, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    outline(ax, p < 0.05)
    ax.set_xticks(range(len(CENTRES)), [f"{x:.0f}" for x in CENTRES], fontsize=13)
    ax.set_yticks(range(len(rows)), [fc.pretty_roi(x) for x in rows], fontsize=13)
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    fig.colorbar(im, ax=ax).set_label(t_label(), fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"[saved] {stem}.png / .pdf")


if __name__ == "__main__":
    main()
