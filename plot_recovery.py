"""Recovery curves: measured vs planted effect for the simulation sweeps.

Reads the per-session pickles `snakemake simulations` writes for every
`sweeps:` tag in config/simulation_config.yaml and plots, per sweep, the effect
the pipeline measured against the one planted. Target-lobe electrodes should sit
on the identity line; all others at 0.

  power      post/pre band power (or amplitude, hilbert) in dB, per electrode; planted 20*log10(gain)
  synchrony  raw post-onset PPC between pairs of electrodes; planted = PPC

    python plot_recovery.py [--root-dir DIR] [--out-dir DIR] [--fc-mode MODE]
"""
import argparse
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

import fc_comparison_functions as fc
from build_roi_power import power_dir
from build_roi_synchrony import cond_dir
from figure_io import SaveFigure
from simulate_eeg import SWEEPS, _target_lobe_mask, simulation_parameters

BEH, BAND = "word_on", "high_gamma"


def session_values(root, tag, fc_mode):
    """[(target, other)] per session: mean measured effect on / off the plant."""
    p = simulation_parameters[tag]
    sim_root = join(root, "sim", tag)
    out = []
    if p.get("data_generating_process") == "hg_power":
        for f in sorted(power_dir(sim_root, BEH, BAND, fc_mode).glob("*_power.pkl")):
            d = fc.load_pickle(str(f))
            m = _target_lobe_mask(d["reg_full"], p["target_lobes"])
            db = (20 if d.get("measure") == "amplitude" else 10) * np.log10(d["pow_hi"] / d["pow_lo"])
            out.append((np.nanmean(db[m]), np.nanmean(db[~m])))
    else:
        hi = fc.beh_conds(BEH)[1]
        for f in sorted(cond_dir(sim_root, BEH, hi, BAND).glob("*_fc_mats.pkl")):
            d = fc.load_pickle(str(f))
            m = _target_lobe_mask(d["reg_full"], p["target_lobes"])
            M = np.asarray(d["ppc"], float)
            tt, oo = np.outer(m, m), ~np.outer(m, m)
            np.fill_diagonal(tt, False); np.fill_diagonal(oo, False)
            out.append((np.nanmean(M[tt]), np.nanmean(M[oo])))
    return np.array(out)


def planted(tag):
    p = simulation_parameters[tag]
    return 20 * np.log10(p["planted"]) if p.get("data_generating_process") == "hg_power" else p["planted"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root-dir", default=fc.root_dir)
    ap.add_argument("--out-dir", default=join("figures", "simulations"))
    ap.add_argument("--fc-mode", default="multitaper", help="power estimator whose pickles to read")
    args = ap.parse_args()

    kinds = {"power": ("Power change (dB)", []), "synchrony": ("PPC after onset", [])}
    for sweep, tags in SWEEPS.items():
        kind = "power" if simulation_parameters[tags[0]].get("data_generating_process") == "hg_power" else "synchrony"
        kinds[kind][1].append((sweep, tags))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (kind, (ylab, sweeps)) in zip(axes, kinds.items()):
        lim = [np.inf, -np.inf]
        for sweep, tags in sweeps:
            x = np.array([planted(t) for t in tags])
            v = [session_values(args.root_dir, t, args.fc_mode) for t in tags]
            if not any(len(s) for s in v):   # sweep not run (e.g. no synchrony under longetal)
                continue
            noise = sweep.split("_")[-1]
            for col, lab, ls in ((0, f"target, {noise}", "-"), (1, f"other, {noise}", ":")):
                y = np.array([s[:, col].mean() if len(s) else np.nan for s in v])
                e = np.array([s[:, col].std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan for s in v])
                ax.errorbar(x, y, yerr=e, marker="o", ls=ls, capsize=3, label=lab)
            lim = [min(lim[0], x.min()), max(lim[1], x.max())]
        if not np.isfinite(lim[0]):   # nothing run for this kind
            ax.set_visible(False)
            continue
        ax.plot(lim, lim, "k--", lw=1, label="identity")
        ax.set(xlabel=f"planted {ylab.lower()}", ylabel=f"measured {ylab.lower()}", title=kind)
        ax.legend(fontsize=8)
    fig.suptitle(f"Recovery ({BEH}, {BAND}; mean ± SEM over sessions)")
    fig.tight_layout()
    SaveFigure(fig, "recovery", args.out_dir)
    print(f"[saved] {args.out_dir}/recovery.png / .pdf")


if __name__ == "__main__":
    main()
