"""plot_decoding.py -- when, along the word-onset response, does information about later
recall appear, and where?

Per trial and electrode: the high-gamma response (plot_latency.response: envelope minus
its word-off mean, smoothed) averaged in 100 ms windows, 0-1000 ms; z-scored within
session. Per subject, Burke lobe (hemispheres pooled) and window: L2 logistic regression
on the lobe's electrodes (C as the lab's RAM encoding classifiers, Ezzyat et al. 2017;
balanced class weights), leave-one-session-out (single-session subjects: 5 contiguous
blocks of trials); AUC of the pooled held-out predictions. Subjects need
>= MIN_TRIALS recalled and forgotten words. Across subjects: t of AUC vs 0.5, BH-FDR
over lobe x window (one-sided: above chance); stars q < .05.
-> figures/decoding/recall_auc_time_word_on.png (+ _per_subject.csv, _stats.csv)

    python plot_decoding.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir
from plot_latency import response

BEH = "word_on"
WIN_MS = 100
WINDOWS = range(0, 1000, WIN_MS)
MIN_TRIALS = 10
C = 7.2e-4   # L2 strength of the lab's encoding classifiers


def session_features(p: dict, lobe_of: dict) -> pd.DataFrame:
    """(trials x '<lobe>|<electrode>|<window start>') z-scored within the session, + recalled, session."""
    roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
    x, t = response(p)
    cols = {f"{r.split('-', 1)[1]}|{lab}|{a}": x[:, c, (t >= a) & (t < a + WIN_MS)].mean(1)
            for c, (r, lab) in enumerate(zip(roi, p["labels"])) if r is not None for a in WINDOWS}
    f = pd.DataFrame(cols)
    f = (f - f.mean()) / f.std()
    f["recalled"], f["session"] = p["recalled_hi"], str(p["sid"])
    return f


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "decoding"))
    lobe_of = fc.load_burke_maps()
    subs: dict[str, list] = {}
    for f in tqdm(fc.session_files(power_dir(save_root, BEH, "high_gamma"), "_power.pkl", args.n_sessions),
                  desc="sessions"):
        p = fc.load_pickle(str(f))
        subs.setdefault(str(p["sid"][0]), []).append(session_features(p, lobe_of))

    rows = []
    for sub, fs in tqdm(subs.items(), desc="subjects"):
        d = pd.concat(fs, ignore_index=True).dropna(subset=["recalled"])
        y = d["recalled"].to_numpy(int)
        if min(y.sum(), len(y) - y.sum()) < MIN_TRIALS:
            continue
        groups = d["session"] if d["session"].nunique() > 1 else np.arange(len(d)) * 5 // len(d)
        for lobe in fc.LOBES:
            for a in WINDOWS:
                cols = [c for c in d.columns if c.startswith(f"{lobe}|") and c.endswith(f"|{a}")]
                if not cols:
                    continue
                prob = cross_val_predict(LogisticRegression(C=C, class_weight="balanced", max_iter=1000),
                                         d[cols].fillna(0).to_numpy(), y, groups=groups,
                                         cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
                rows.append({"sub": sub, "lobe": lobe, "window_ms": a, "auc": roc_auc_score(y, prob)})
    df = pd.DataFrame(rows)
    st = df.groupby(["lobe", "window_ms"])["auc"].agg(["mean", "sem", "size"]).reset_index()
    st["p"] = [ttest_1samp(g["auc"], 0.5, alternative="greater").pvalue for _, g in df.groupby(["lobe", "window_ms"])]
    st["q"] = multipletests(st["p"], method="fdr_bh")[1]
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"recall_auc_time_{BEH}")
    df.to_csv(f"{stem}_per_subject.csv", index=False)
    st.to_csv(f"{stem}_stats.csv", index=False)
    print(st.round(4).to_string(index=False))

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for lobe, ax in zip(fc.LOBES, axes.flat):
        s = st[st["lobe"] == lobe]
        x, col = s["window_ms"] + WIN_MS / 2, fc.LOBE_COLORS[lobe]
        ax.fill_between(x, s["mean"] - s["sem"], s["mean"] + s["sem"], color=col, alpha=0.3, lw=0)
        ax.plot(x, s["mean"], "o-", color=col, lw=2, ms=4)
        sig = s["q"] < 0.05
        ax.plot(x[sig], (s["mean"] + s["sem"])[sig] + 0.004, "*", color=col, ms=8)
        ax.axhline(0.5, color="black", ls="--", lw=1)
        ax.set_title(f"{lobe.capitalize()} (n={int(s['size'].max())})", fontsize=12, color=col)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Time after word onset (ms)", fontsize=13)
    fig.supylabel("Recall AUC", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"[saved] {stem}.png")


if __name__ == "__main__":
    main()
