"""plot_decoding.py -- does the single-trial word-onset response predict later recall?

Per trial and Burke ROI (LOBES: those with a clear spectral effect), from compute.py's
power pickles, mean over the ROI's electrodes; one group per paper section:
  power        word on (0-600 ms): log alpha, log high-gamma power         (power figure)
  time course  high-gamma response (plot_latency.response) in 100 ms bins,
               0-1000 ms                                                  (time course)
  pre-word     word off (the blank before the word): log alpha, log high-gamma
               power; and the blank's duration                           (anticipation)
z-scored within session; missing values (ROI absent) -> 0.

Per subject (>= MIN_TRIALS recalled and forgotten words): L2 logistic regression
(C as the lab's RAM encoding classifiers, Ezzyat et al. 2017), balanced class weights,
leave-one-session-out (single-session subjects: 5 contiguous blocks of trials); AUC of
the pooled held-out predictions. Models add the groups in turn. Across subjects:
one-sample t of AUC vs 0.5; paired t of each model vs the one before it (does the
group add information?).
-> figures/decoding/recall_auc_word_on.png (+ _per_subject.csv)

    python plot_decoding.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir
from plot_latency import response

BEH = "word_on"
LOBES = ("frontal", "occipital", "hippocampus")   # both hemispheres
GROUPS = {"power": ["alpha", "hg_power"], "time course": ["hg_t"], "pre-word": ["alpha_pre", "hg_pre", "wait"]}
MODELS = {"power": GROUPS["power"], "+ time course": GROUPS["power"] + GROUPS["time course"],
          "+ pre-word": sum(GROUPS.values(), [])}
BIN_MS = 100
MIN_TRIALS = 10
C = 7.2e-4   # L2 strength of the lab's encoding classifiers; C = 1 overfits the time-course features


def session_features(p: dict, p_alpha: dict, lobe_of: dict) -> pd.DataFrame:
    """(trials x '<feature>|<roi>') z-scored within the session, + recalled, session."""
    roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
    x, t = response(p)
    log = lambda q, k: np.log(np.asarray(q[k], float))
    cols = {"wait": p["blank_ms_hi"]}
    for r in {r for r in roi if r is not None and r.split("-", 1)[1] in LOBES}:
        e = roi == r
        cols[f"alpha|{r}"], cols[f"hg_power|{r}"] = log(p_alpha, "trial_pow_hi")[:, e].mean(1), log(p, "trial_pow_hi")[:, e].mean(1)
        cols[f"alpha_pre|{r}"], cols[f"hg_pre|{r}"] = log(p_alpha, "trial_pow_lo")[:, e].mean(1), log(p, "trial_pow_lo")[:, e].mean(1)
        for a in range(0, 1000, BIN_MS):
            cols[f"hg_t|{r}|{a}"] = x[:, e][..., (t >= a) & (t < a + BIN_MS)].mean((1, 2))
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
        p_alpha = fc.load_pickle(str(power_dir(save_root, BEH, "alpha") / f.name))
        subs.setdefault(str(p["sid"][0]), []).append(session_features(p, p_alpha, lobe_of))

    rows = []
    for sub, fs in tqdm(subs.items(), desc="subjects"):
        d = pd.concat(fs, ignore_index=True).dropna(subset=["recalled"])
        y = d["recalled"].to_numpy(int)
        if min(y.sum(), len(y) - y.sum()) < MIN_TRIALS:
            continue
        groups = d["session"] if d["session"].nunique() > 1 else np.arange(len(d)) * 5 // len(d)
        r = {"sub": sub, "n_sessions": d["session"].nunique(), "n_trials": len(y), "recall_rate": y.mean()}
        for m, feats in MODELS.items():
            X = d[[c for c in d if c.split("|")[0] in feats]].fillna(0).to_numpy()
            if X.shape[1]:
                prob = cross_val_predict(LogisticRegression(C=C, class_weight="balanced", max_iter=1000), X, y,
                                         groups=groups, cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
                r[m] = roc_auc_score(y, prob)
        rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"recall_auc_{BEH}")
    df.to_csv(f"{stem}_per_subject.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    for i, m in enumerate(MODELS):
        a = df[m].dropna()
        t, pv = ttest_1samp(a, 0.5)
        print(f"[decoding] {m}: AUC {a.mean():.3f} ± {a.sem():.3f} SEM, t({len(a) - 1}) = {t:.2f}, p = {pv:.2g}")
        ax.bar(i, a.mean(), yerr=a.sem(), color=plt.cm.Blues(0.4 + 0.25 * i), edgecolor="0.15", capsize=4)
    for m0, m1 in zip(MODELS, list(MODELS)[1:]):
        t, pv = ttest_rel(df[m1], df[m0], nan_policy="omit")
        print(f"[decoding] {m1} vs {m0}: t = {t:.2f}, p = {pv:.2g}")
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.set_ylim(0.45, None)
    ax.set_xticks(range(len(MODELS)), list(MODELS), fontsize=11)
    ax.set_xlabel("Features used", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"[saved] {stem}.png")


if __name__ == "__main__":
    main()
