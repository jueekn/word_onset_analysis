"""plot_reliability.py -- is the word-onset response stable across a subject's sessions?

Per session (compute.py power pickles): per-electrode Cohen's d (word on vs off) for alpha
and high gamma. As riley-thesis compute_retest_reliability.py (its channel-level run):
each subject's sessions split at random into two halves (seed 0), >= MIN_SESS_PER_HALF
sessions per half; d averaged per electrode within each half; test-retest = Pearson r
across the subject's electrodes between the halves (>= MIN_FINITE electrodes); dashed: means.
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
from compute import power_dir

BEH = "word_on"
BANDS = {"alpha": "Alpha d", "high_gamma": "High-gamma d"}
MIN_SESS_PER_HALF = 2   # riley-thesis defaults
MIN_FINITE = 10


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "reliability"))
    el = []
    for f in tqdm(fc.session_files(power_dir(save_root, BEH, "high_gamma"), "_power.pkl", args.n_sessions),
                  desc="sessions"):
        p = {b: fc.load_pickle(str(power_dir(save_root, BEH, b) / f.name)) for b in BANDS}
        sub, exp, n = p["alpha"]["sid"]
        el.append(pd.DataFrame({"sub": sub, "sess": f"{exp}_{n}", "label": p["alpha"]["labels"],
                                **{b: p[b]["cohens_d"] for b in BANDS}}))
    el = pd.concat(el, ignore_index=True)

    rng = np.random.default_rng(0)
    order = el[["sub", "sess"]].drop_duplicates().sort_values(["sub", "sess"])
    order["half"] = order.groupby("sub")["sess"].transform(lambda x: rng.permutation(len(x)) >= len(x) // 2)
    order = order[order.groupby("sub")["half"].transform(lambda x: min(x.sum(), (~x).sum())) >= MIN_SESS_PER_HALF]
    h = el.merge(order, on=["sub", "sess"]).groupby(["sub", "half", "label"], as_index=False)[list(BANDS)].mean()
    rt = []
    for sub, g in h.groupby("sub"):
        w = g.pivot(index="label", columns="half", values=list(BANDS)).dropna()
        if len(w) >= MIN_FINITE:
            rt.append({"sub": sub, **{b: pearsonr(w[(b, False)], w[(b, True)])[0] for b in BANDS}})
    rt = pd.DataFrame(rt)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"reliability_{BEH}")
    rt.to_csv(join(args.out_dir, f"test_retest_{BEH}.csv"), index=False)
    print("[reliability] test-retest r (mean): " + ", ".join(f"{k} {rt[k].mean():.2f}" for k in BANDS)
          + f"  (n={len(rt)} subjects)")

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for k, col in zip(BANDS, ("C0", "C3")):
        ax.hist(rt[k].dropna(), bins=np.linspace(-1, 1, 41), color=col, alpha=0.6, label=BANDS[k])
        ax.axvline(rt[k].mean(), color=col, ls="--", lw=1.5)
    ax.set_xlabel("Test-retest r across electrodes", fontsize=11)
    ax.set_ylabel("Subjects", fontsize=11)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"[saved] {stem}.png")


if __name__ == "__main__":
    main()
