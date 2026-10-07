"""plot_timecourse.py -- high-gamma word on vs off power over time per Burke ROI.

Per electrode, Cohen's d of sub-band Hilbert high-gamma power in each 50 ms bin of
0-600 ms against the whole word-off window (compute.py `cohens_d_bins`); sessions
averaged per electrode, electrodes per subject and ROI; one-sample t vs 0 across
subjects, BH-FDR over all ROI x bin cells (fc.bin_stats).
-> figures/burke_roi_power/power_timecourse_word_on_high_gamma.png

    python plot_timecourse.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import numpy as np
import pandas as pd

import fc_comparison_functions as fc
from compute import power_dir

BEH, BAND = "word_on", "high_gamma"


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "burke_roi_power"))
    lobe_of = fc.load_burke_maps()
    rows = []
    for f in fc.session_files(power_dir(save_root, BEH, BAND), "_power.pkl", args.n_sessions):
        p = fc.load_pickle(str(f))
        centres = np.asarray(p["bin_centers_ms"], float)
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        rows += [(str(p["sid"][0]), str(lab), r, float(c), float(v))
                 for lab, r, dd in zip(p["labels"], roi, np.asarray(p["cohens_d_bins"], float)) if r is not None
                 for c, v in zip(centres, dd) if np.isfinite(v)]
    el = (pd.DataFrame(rows, columns=["sub", "label", "roi", "bin_ms", "d"])
            .groupby(["sub", "label", "roi", "bin_ms"], as_index=False)["d"].mean())
    tbl = el.groupby(["sub", "roi", "bin_ms"]).agg(d=("d", "mean"), n_elec=("label", "nunique")).reset_index()
    tbl = tbl[tbl["n_elec"] >= args.min_electrodes]
    stats = fc.bin_stats(tbl, "bin_ms", "d")
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"power_timecourse_{BEH}_{BAND}")
    stats.to_csv(f"{stem}_stats.csv", index=False)
    tbl.to_csv(f"{stem}_per_subject.csv", index=False)
    path = fc.roi_curve_figure(stats, "bin_ms", stem, "Time after word onset (ms)", "Cohen's d")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
