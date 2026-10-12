"""plot_anticipation.py -- does the high-gamma response depend on how long the
participant waited for the word (the jittered blank before it)?

Responsive electrodes as plot_latency.py. Per electrode, trials are split at the
session's median blank; for short- and long-wait trials, the peak (0-800 ms) of the
trial-mean high-gamma envelope minus the (common) word-off mean. Long - short per
electrode -> subject mean per ROI -> mean +/- SEM, one-sample t vs 0, BH-FDR over
ROIs (fc.roi_stats).
-> figures/latency/hg_anticipation_word_on.png (peak change per ROI)
   figures/latency/wait_times_word_on.png (blank durations, dashed line at the median)

    python plot_anticipation.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir
from plot_latency import PEAK_WIN_MS, SMOOTH_MS
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

BEH = "word_on"


def peak(curve: np.ndarray, t: np.ndarray, k: int) -> float:
    """Peak of a smoothed response curve within PEAK_WIN_MS."""
    c = np.convolve(curve, np.ones(k) / k, mode="same")
    return float(c[(t >= PEAK_WIN_MS[0]) & (t < PEAK_WIN_MS[1])].max())


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "latency"))
    lobe_of = fc.load_burke_maps()
    rows, waits = [], []
    for f in tqdm(fc.session_files(power_dir(save_root, BEH, "high_gamma"), "_power.pkl", args.n_sessions),
                  desc="sessions"):
        p = fc.load_pickle(str(f))
        lo, hi = np.asarray(p["trial_pow_lo"], float), np.asarray(p["trial_pow_hi"], float)
        _, pv = ttest_ind(hi, lo, axis=0, equal_var=False, alternative="greater", nan_policy="omit")
        ok = np.isfinite(pv)
        resp = np.zeros(len(pv), bool)
        if ok.any():
            resp[ok] = multipletests(pv[ok], method="fdr_bh")[0]
        env_hi = np.asarray(p["env_hi"], float)
        t = p["env_hi_t0_ms"] + p["env_step_ms"] * (np.arange(env_hi.shape[-1]) + 0.5)
        k = max(1, int(round(SMOOTH_MS / p["env_step_ms"])))
        blank = np.asarray(p["blank_ms_hi"], float)
        waits.append(blank[np.isfinite(blank)])
        split = {"short": blank < np.nanmedian(blank), "long": blank > np.nanmedian(blank)}
        base = np.nanmean(np.asarray(p["env_lo"], float), axis=(0, 2))   # common word-off baseline
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        for c in np.flatnonzero(resp):
            if roi[c] is None:
                continue
            r = {"sub": str(p["sid"][0]), "label": str(p["labels"][c]), "roi": roi[c]}
            r["peak"] = peak(np.nanmean(env_hi[split["long"], c], 0) - base[c], t, k) \
                - peak(np.nanmean(env_hi[split["short"], c], 0) - base[c], t, k)
            rows.append(r)
    el = pd.DataFrame(rows).groupby(["sub", "label", "roi"], as_index=False)["peak"].mean()
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"hg_anticipation_{BEH}")
    el.to_csv(f"{stem}_per_electrode.csv", index=False)

    tbl = el.groupby(["sub", "roi"], as_index=False)["peak"].mean()
    st = fc.roi_stats(tbl, "peak")
    st.to_csv(f"{stem}_peak_stats.csv", index=False)
    allsub = el.groupby("sub")["peak"].mean()
    print(f"[anticipation] peak, all electrodes: {allsub.mean():+.4g} ± "
          f"{allsub.std() / np.sqrt(len(allsub)):.3g} SEM (n={len(allsub)} subjects)")
    fc.roi_figure([("High-gamma response peak\nlong − short wait", lambda ax: fc.roi_bar_panel(ax, st))],
                  args.out_dir, f"hg_anticipation_{BEH}")

    # wait (blank) durations, all words; trials are split at each session's own median
    w = np.concatenate(waits)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    lo, hi = np.percentile(w, [0.1, 99.9])   # a few pauses of several seconds would stretch the axis
    # bins of one 60 Hz screen frame: word timing is frame-locked, finer bins alias
    ax.hist(w[(w >= lo) & (w <= hi)], bins=np.arange(lo, hi + 1000 / 60, 1000 / 60), color="0.6", edgecolor="white", lw=0.3)
    ax.axvline(np.median(w), color="black", ls="--", lw=1.5, label=f"median {np.median(w):.0f} ms")
    ax.set_xlabel("Blank before the word (ms)", fontsize=12)
    ax.set_ylabel("Words", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    wstem = join(args.out_dir, f"wait_times_{BEH}")
    fig.savefig(f"{wstem}.png", dpi=300)
    fig.savefig(f"{wstem}.pdf")
    plt.close(fig)
    print(f"[saved] {stem}.png, {wstem}.png")


if __name__ == "__main__":
    main()
