"""plot_reliability.py -- is the word-onset response stable across a subject's sessions?

Per session (compute.py power pickles): per-electrode Cohen's d (word on vs off) for alpha
and high gamma. As riley-thesis compute_retest_reliability.py (its channel-level run):
each subject's sessions split at random into two halves (seed 0), >= MIN_SESS_PER_HALF
sessions per half; per session, d averaged per anatomical region (reg_full), then sessions
averaged within each half; test-retest = Pearson r across the subject's regions between the
halves (>= MIN_FINITE regions); dashed: medians. Synchrony (alpha ciPLV, high-gamma AEC-c;
word on - word off): the same on node strength (fc.electrode_sync: each electrode's mean
over all partners, as the synchrony summary figure; riley-thesis hub reliability).
-> figures/reliability/reliability_word_on.png (+ test_retest_word_on.csv)

    python plot_reliability.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import cond_dir, power_dir

BEH = "word_on"
BANDS = {"alpha": "Alpha d", "high_gamma": "High-gamma d"}
MIN_SESS_PER_HALF = 2   # riley-thesis defaults
MIN_FINITE = 10
SYNC = {"alpha": ("ciplv", "Alpha ciPLV"), "high_gamma": ("aec_c", "High-gamma AEC-c")}


def retest(d: pd.DataFrame, key: str, cols: list[str], order: pd.DataFrame) -> pd.DataFrame:
    """Per subject and column: Pearson r across `key` (region) between the two halves'
    session means; NaN with < MIN_FINITE finite values."""
    s = d.groupby(["sub", "sess", key], as_index=False)[cols].mean()   # per-session region means
    h = s.merge(order, on=["sub", "sess"]).groupby(["sub", "half", key])[cols].mean()
    rows = []
    for sub, g in h.groupby("sub"):
        r = {"sub": sub}
        for c in cols:
            w = g[c].unstack("half").dropna()
            r[c] = pearsonr(w[False], w[True])[0] if len(w) >= MIN_FINITE else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "reliability"))
    el = []
    for f in tqdm(fc.session_files(power_dir(save_root, BEH, "high_gamma"), "_power.pkl", args.n_sessions),
                  desc="sessions"):
        p = {b: fc.load_pickle(str(power_dir(save_root, BEH, b) / f.name)) for b in BANDS}
        sub, exp, n = p["alpha"]["sid"]
        reg = np.asarray(p["alpha"]["reg_full"], str)
        fcm = {b: fc.load_pickle(str(cond_dir(save_root, BEH, "diff", b) / f.name.replace("_power.pkl", "_fc_mats.pkl")))
               for b in SYNC}
        el.append(pd.DataFrame({"sub": sub, "sess": f"{exp}_{n}", "region": reg,
                                **{b: p[b]["cohens_d"] for b in BANDS},
                                **{m: fc.electrode_sync(fcm[b][m]) for b, (m, _) in SYNC.items()}}))
    el = pd.concat(el, ignore_index=True)
    el = el[~el["region"].isin(["None", "nan"])]

    rng = np.random.default_rng(0)
    order = el[["sub", "sess"]].drop_duplicates().sort_values(["sub", "sess"])
    order["half"] = order.groupby("sub")["sess"].transform(lambda x: rng.permutation(len(x)) >= len(x) // 2)
    order = order[order.groupby("sub")["half"].transform(lambda x: min(x.sum(), (~x).sum())) >= MIN_SESS_PER_HALF]
    sync_cols = [m for m, _ in SYNC.values()]
    rt = retest(el, "region", list(BANDS) + sync_cols, order)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"reliability_{BEH}")
    rt.to_csv(fc.table_path(join(args.out_dir, f"test_retest_{BEH}.csv")), index=False)
    print("[reliability] test-retest r (median): " + ", ".join(f"{k} {rt[k].median():.2f}" for k in list(BANDS) + sync_cols)
          + f"  (n={len(rt)} subjects)")

    panels = [("Power", "regions", BANDS),
              ("Synchrony (node strength)", "regions", dict(SYNC.values()))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (title, unit, cols) in zip(axes, panels):
        for (k, label), col in zip(cols.items(), ("C0", "C3")):
            ax.hist(rt[k].dropna(), bins=np.linspace(-1, 1, 41), color=col, alpha=0.6, label=label)
            ax.axvline(rt[k].median(), color=col, ls="--", lw=1.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"Test-retest r across {unit}", fontsize=11)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Subjects", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"[saved] {stem}.png")


if __name__ == "__main__":
    main()
