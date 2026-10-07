"""plot_case_studies.py -- t of word on vs off power vs frequency for chosen anatomical regions.

One panel per region (hemispheres as separate lines), from the same per-electrode
spectrum d as plot_spectrum.py; BH-FDR over all region x bin cells of the figure.
Default regions: fusiform gyrus and lateral occipital cortex.
-> figures/burke_roi_power/power_spectrum_word_on_<name>.png

    python plot_case_studies.py [--regions "fusiform gyrus" "lateral occipital cortex"] [--name occipital]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt

import fc_comparison_functions as fc
from plot_spectrum import BEH, CENTRES, bin_t, spectrum_table, t_label


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "burke_roi_power"), **{
        "--regions": dict(nargs="+", default=["fusiform gyrus", "lateral occipital cortex"],
                          help="anatomical labels without hemisphere"),
        "--name": dict(default="occipital", help="figure name suffix")})
    d = spectrum_table(save_root, fc.load_burke_maps(), args.n_sessions)
    split = d["fine"].str.split(" ", n=1)
    d["region"] = [f"{r[1]}|{r[0]}" if isinstance(r, list) and r[1] in args.regions else None for r in split]
    st = bin_t(d, "region", args.min_electrodes)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"power_spectrum_{BEH}_{args.name}")
    st.to_csv(f"{stem}_stats.csv", index=False)
    panels = [r for r in args.regions if st[st["roi"].str.startswith(r + "|")]["t"].notna().any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5), sharey=True, squeeze=False)
    for ax, panel in zip(axes[0], panels):
        for name, s in st[st["roi"].str.startswith(panel + "|")].groupby("roi"):
            s = s.sort_values("freq_hz")
            line, = ax.plot(s["freq_hz"], s["t"], marker="o", lw=2,
                            label=f"{name.split('|')[1]} (n={int(s['n'].max())})")
            for x, y, q in zip(s["freq_hz"], s["t"], s["q"]):
                if fc.stars(q):
                    ax.annotate(fc.stars(q), (x, y), textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=13, color=line.get_color())
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(CENTRES, [f"{x:.0f}" for x in CENTRES], fontsize=12)
        ax.minorticks_off()
        ax.tick_params(axis="y", labelsize=13)
        ax.set_title(panel[:1].upper() + panel[1:], fontsize=15)
        ax.set_xlabel("Frequency (Hz)", fontsize=14)
        ax.legend(fontsize=12, frameon=False)
    axes[0, 0].set_ylabel(t_label(), fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    print(f"[saved] {stem}.png / .pdf")


if __name__ == "__main__":
    main()
