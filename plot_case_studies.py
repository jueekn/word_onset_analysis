"""plot_case_studies.py -- t of word on vs off power vs frequency for anatomical
subregions, hemispheres pooled, all in one plot.

Default: the 4 occipital subregions (Burke occipital ROI) with the most subjects
(hemispheres pooled, >= MIN_SUBJECTS_ROI). Per-electrode spectrum d as
plot_spectrum.py; a subject's value is the mean over its electrodes in the region,
both hemispheres. Significance: TFCE along frequency with sign-flip permutations
and max-statistic correction over all region x bin cells (plot_spectrum.tfce_test);
filled markers p_FWE < .05.
-> figures/burke_roi_power/power_spectrum_word_on_<name>.png

    python plot_case_studies.py [--regions "fusiform gyrus" ...] [--name occipital]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd

import fc_comparison_functions as fc
from plot_spectrum import BEH, CENTRES, spectrum_table, subject_table, t_label, tfce_test


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "burke_roi_power"), **{
        "--regions": dict(nargs="+", default=None,
                          help="anatomical labels without hemisphere (default: top 4 occipital)"),
        "--name": dict(default="occipital", help="figure name suffix")})
    d = spectrum_table(save_root, fc.load_burke_maps(), args.n_sessions).dropna(subset=["fine"])
    d["region"] = d["fine"].str.split(" ", n=1).str[1]          # hemispheres pooled
    if args.regions is None:
        occ = d[d["roi"].astype(str).str.endswith("occipital")]
        n = occ.groupby("region")["sub"].nunique().sort_values(ascending=False)
        args.regions = list(n[n >= fc.MIN_SUBJECTS_ROI].index[:4])
    d = d[d["region"].isin(args.regions)]
    g = subject_table(d, "region", args.min_electrodes)
    T, _, p = tfce_test(g, args.regions)
    n_sub = g.groupby("roi")["sub"].nunique()
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"power_spectrum_{BEH}_{args.name}")
    pd.DataFrame([(r, CENTRES[b], T[i, b], p[i, b]) for i, r in enumerate(args.regions) for b in range(len(CENTRES))],
                 columns=["region", "freq_hz", "t", "p_fwe_tfce"]).to_csv(f"{stem}_stats.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, r in enumerate(args.regions):
        line, = ax.plot(CENTRES, T[i], lw=2, label=f"{r[:1].upper() + r[1:]} (n={n_sub.get(r, 0)})")
        sig = p[i] < 0.05
        ax.plot(CENTRES[~sig], T[i][~sig], "o", ms=6, mfc="white", mec=line.get_color(), mew=1.5)
        ax.plot(CENTRES[sig], T[i][sig], "o", ms=7, color=line.get_color())
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(CENTRES, [f"{x:.0f}" for x in CENTRES], fontsize=12)
    ax.minorticks_off()
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel(t_label(), fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=11, frameon=False, title="filled: p$_{FWE}$ < .05 (TFCE)", title_fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"[saved] {stem}.png / .pdf  regions: {args.regions}")


if __name__ == "__main__":
    main()
