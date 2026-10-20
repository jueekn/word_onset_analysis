"""plot_decoding.py -- when, along the word-onset response, and where does information
about later recall appear?

Encoding contrast only (recalled vs not-recalled words; the lab's RAM classifiers,
ram_utils, also train on retrieval events). Per trial and electrode: the high-gamma
response (plot_latency.response: envelope minus its word-off mean, smoothed) averaged in
100 ms windows, 0-1000 ms; z-scored within session. Per subject (>= MIN_TRIALS recalled and
forgotten words), Burke lobe (hemispheres pooled; >= MIN_ELECTRODES electrodes) and window:
L2 logistic regression on the lobe's electrodes (C as ram_utils / Ezzyat et al. 2017; balanced class weights),
leave-one-session-out (single-session subjects: leave-one-list-out); AUC of the pooled
held-out predictions. Chance: the mean AUC over N_PERM within-session label shuffles per
subject (as ram_utils' permuted_loso_cross_validation; ~0.49: pooled held-out AUC is biased
low), reported only. Across subjects: one-sided t of AUC > 0.5 (stricter than the permutation
level), BH-FDR over lobe x window; stars q < .05; dashed: 0.5.
-> figures/decoding/recall_auc_time_word_on.png (+ _per_subject.csv, _stats.csv)

    python plot_decoding.py [--from-csv] [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

from joblib import Parallel, delayed

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
MIN_ELECTRODES = 5   # per subject and lobe: features per classifier
C = 7.2e-4   # L2 strength of the lab's encoding classifiers
N_PERM = 20  # per subject; enough for the group chance level (ram_utils uses 200 for per-subject p)


def session_features(p: dict, lobe_of: dict) -> pd.DataFrame:
    """(trials x '<lobe>|<electrode>|<window start>') z-scored within the session, +
    recalled, session, list (NaN when trials don't match the events)."""
    roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
    x, t = response(p)
    f = pd.DataFrame({f"{r.split('-', 1)[1]}|{lab}|{a}": x[:, c, (t >= a) & (t < a + WIN_MS)].mean(1)
                      for c, (r, lab) in enumerate(zip(roi, p["labels"])) if r is not None for a in WINDOWS})
    f = (f - f.mean()) / f.std()
    w = fc.load_events(fc.dfrow_from_sid(p["sid"]), BEH)
    w = w[np.asarray(w.attrs["mask"], bool)]   # word-on events, in the stored trial order
    f["recalled"], f["session"] = p["recalled_hi"], str(p["sid"])
    f["list"] = w["trial"].to_numpy(float) if len(w) == len(f) else np.nan
    return f


def auc(d: pd.DataFrame, cols: list[str], y: np.ndarray, groups, perms) -> tuple[float, float]:
    """(held-out AUC, mean held-out AUC over the label permutations)."""
    X = d[cols].fillna(0).to_numpy()
    def one(yy):
        prob = cross_val_predict(LogisticRegression(C=C, class_weight="balanced", max_iter=1000), X, yy,
                                 groups=groups, cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
        return roc_auc_score(yy, prob)
    return one(y), float(np.mean([one(y[i]) for i in perms]))


def decode_subject(sub: str, fs: list) -> list:
    """Per lobe x window AUCs (and their permutation means) for one subject."""
    d = pd.concat(fs, ignore_index=True).dropna(subset=["recalled"])
    y = d["recalled"].to_numpy(int)
    if min(y.sum(), len(y) - y.sum()) < MIN_TRIALS:
        return []
    sess = d["session"].to_numpy()
    if len(set(sess)) > 1:
        groups = sess
    else:
        groups = d["list"] if d["list"].notna().all() else np.arange(len(d)) * 5 // len(d)
    rng = np.random.default_rng(0)
    perms = []
    for _ in range(N_PERM):   # shuffle labels within each session
        i = np.arange(len(d))
        for s in set(sess):
            i[sess == s] = rng.permutation(i[sess == s])
        perms.append(i)
    tres = []
    for lobe in fc.LOBES:
        for a in WINDOWS:
            cols = [c for c in d.columns if c.startswith(f"{lobe}|") and c.endswith(f"|{a}")]
            if len(cols) >= MIN_ELECTRODES:
                r, nl = auc(d, cols, y, groups, perms)
                tres.append({"sub": sub, "lobe": lobe, "window_ms": a, "n_electrodes": len(cols), "auc": r, "null": nl})
    return tres


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "decoding"), **{
        "--from-csv": dict(action="store_true", help="redraw from the saved per-subject CSV (no decoding)")})
    stem = join(args.out_dir, f"recall_auc_time_{BEH}")
    if args.from_csv:
        tres = pd.read_csv(f"{stem}_per_subject.csv")
    else:
        lobe_of = fc.load_burke_maps()
        subs: dict[str, list] = {}
        for f in tqdm(fc.session_files(power_dir(save_root, BEH, "high_gamma"), "_power.pkl", args.n_sessions),
                      desc="sessions"):
            p = fc.load_pickle(str(f))
            subs.setdefault(str(p["sid"][0]), []).append(session_features(p, lobe_of))
        out = Parallel(n_jobs=max(1, os.cpu_count() // 2))(
            delayed(decode_subject)(sub, fs) for sub, fs in tqdm(subs.items(), desc="subjects"))
        tres = pd.DataFrame([t for r in out for t in r])
        os.makedirs(args.out_dir, exist_ok=True)
        tres.to_csv(f"{stem}_per_subject.csv", index=False)

    g = tres.groupby(["lobe", "window_ms"])
    st = g["auc"].agg(["mean", "sem", "size"]).join(g["null"].mean()).reset_index()
    st["p"] = [ttest_1samp(x["auc"], 0.5, alternative="greater").pvalue for _, x in g]
    st["q"] = multipletests(st["p"], method="fdr_bh")[1]
    st.to_csv(f"{stem}_stats.csv", index=False)
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
    print(f"[decoding] mean permutation chance level {st['null'].mean():.3f}; "
          f"{int((st['q'] < 0.05).sum())} of {len(st)} lobe x window cells AUC > 0.5 at q < .05")
    print(f"[saved] {stem}.png")


if __name__ == "__main__":
    main()
