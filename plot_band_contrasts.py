"""Word on - word off contrast for power, synchrony and power-synchrony, one
figure per analysis with an alpha row and a high-gamma row.

Reads the per-subject CSVs the three plot stages write, so it runs after them:
    python plot_band_contrasts.py --fig-dir figures
Each panel: per-subject contrast per ROI; stars from fc.roi_stats (paired t,
i.e. one-sample t on the within-subject contrast, BH-FDR over ROIs).
"""
import argparse
from os.path import join

import numpy as np
import pandas as pd

import fc_comparison_functions as fc

BANDS = ("alpha", "high_gamma")
# analysis -> (subfolder, per-subject CSV stem, contrast column, y label)
ANALYSES = {
    "power": ("burke_roi_power", "roi_power_{beh}_{band}", "cohens_d", "power (Cohen's d)"),
    "synchrony": ("burke_roi_synchrony", "roi_synchrony_{beh}_{band}_{metric}", "sync_diff", "{metric} on - off\n(z within distance bin)"),
    "power_synchrony": ("power_synchrony", "power_synchrony_{beh}_{band}_{metric}", "r", "power-sync r (on - off)"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fig-dir", default="figures")
    p.add_argument("--beh", default="word_on")
    p.add_argument("--metric", default="ppc")
    a = p.parse_args()
    c = fc.contrast(a.beh)
    for name, (sub, stem, col, ylab) in ANALYSES.items():
        panels = []
        for band in BANDS:
            f = join(a.fig_dir, sub, stem.format(beh=a.beh, band=band, metric=a.metric) + "_per_subject.csv")
            tbl = pd.read_csv(f)
            if "cond" in tbl:   # power-synchrony: r of the on - off differences
                tbl = tbl[tbl["cond"] == "diff"]
            stats = fc.roi_stats(tbl, col)
            fc.print_roi_stats(stats, f"{band} {name}: {c['hi_label']} - {c['lo_label']}")
            panels.append((f"{band}\n{ylab.format(metric=a.metric.upper())}",
                           lambda ax, tbl=tbl, stats=stats: fc.roi_panel(
                               ax, tbl, col, np.random.default_rng(0), stats=stats)))
        fc.roi_figure(panels, join(a.fig_dir, "band_contrasts"), f"{name}_{a.beh}")


if __name__ == "__main__":
    main()
