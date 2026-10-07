"""Recovery curves: measured vs planted effect for the simulation sweeps.

Reads the per-session pickles `snakemake simulations` writes for every
`sweeps:` tag in config/simulation_config.yaml and plots, per sweep, the effect
the pipeline measured against the one planted. Target-lobe electrodes should sit
on the identity line; all others at 0.

  power   high gamma post/pre band power (or amplitude, hilbert) in dB; planted 20*log10(gain)
  ciPLV   alpha word on - word off ciPLV between electrode pairs; planted = post phase consistency
  AEC-c   high-gamma word on - word off AEC-c; planted = depth of the shared envelope
Synchrony plants are not on the metric's scale: those curves should rise
monotonically for target pairs and stay at 0 for all other pairs.

    python plot_recovery.py [--root-dir DIR] [--out-dir DIR]
"""
import argparse
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

import fc_comparison_functions as fc
from compute import cond_dir, power_dir
from simulate_eeg import SWEEPS, _target_lobe_mask, simulation_parameters

BEH = "word_on"
# data-generating process -> (panel, band, metric; None = power)
KIND = {"hg_power": ("power", "high_gamma", None), "osc_lag": ("ciPLV", "alpha", "ciplv"),
        "hg_envelope": ("AEC-c", "high_gamma", "aec_c")}


def session_values(root, tag):
    """[(target, other)] per session: mean measured effect on / off the plant."""
    p = simulation_parameters[tag]
    _, band, metric = KIND[p["data_generating_process"]]
    sim_root = join(root, "sim", tag)
    out = []
    if metric is None:
        for f in fc.session_files(power_dir(sim_root, BEH, band), "_power.pkl"):
            d = fc.load_pickle(str(f))
            m = _target_lobe_mask(d["reg_full"], p["target_lobes"])
            db = (20 if d.get("measure") == "amplitude" else 10) * np.log10(d["pow_hi"] / d["pow_lo"])
            out.append((np.nanmean(db[m]), np.nanmean(db[~m])))
    else:
        for f in fc.session_files(cond_dir(sim_root, BEH, "diff", band), "_fc_mats.pkl"):
            d = fc.load_pickle(str(f))
            m = _target_lobe_mask(d["reg_full"], p["target_lobes"])
            M = fc.symmetrize_dense(np.asarray(d[metric], float), diag_value=np.nan)
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
    args = ap.parse_args()

    kinds = {"power": ("Power change (dB)", []), "ciPLV": ("ciPLV change", []), "AEC-c": ("AEC-c change", [])}
    for sweep, tags in SWEEPS.items():
        kinds[KIND[simulation_parameters[tags[0]]["data_generating_process"]][0]][1].append((sweep, tags))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (kind, (ylab, sweeps)) in zip(axes, kinds.items()):
        lim = [np.inf, -np.inf]
        for sweep, tags in sweeps:
            x = np.array([planted(t) for t in tags])
            v = [session_values(args.root_dir, t) for t in tags]
            if not any(len(s) for s in v):   # sweep not run (e.g. no synchrony under longetal)
                continue
            noise = sweep.split("_")[-1] if sweep.endswith(("noisy", "clean")) else ""
            for col, lab, ls in ((0, f"target {noise}", "-"), (1, f"other {noise}", ":")):
                y = np.array([np.nanmean(s[:, col]) if len(s) else np.nan for s in v])   # NaN: session without target pairs
                e = np.array([np.nanstd(s[:, col], ddof=1) / np.sqrt(np.isfinite(s[:, col]).sum())
                              if len(s) > 1 else np.nan for s in v])
                ax.errorbar(x, y, yerr=e, marker="o", ls=ls, capsize=3, label=lab)
            lim = [min(lim[0], x.min()), max(lim[1], x.max())]
        if not np.isfinite(lim[0]):   # nothing run for this kind
            ax.set_visible(False)
            continue
        if kind == "power":
            ax.plot(lim, lim, "k--", lw=1, label="identity")
        ax.axhline(0, color="0.6", lw=0.8)
        planted_lab = {"power": "planted power change (dB)", "ciPLV": "planted phase consistency (post)",
                       "AEC-c": "planted shared-envelope depth"}[kind]
        ax.set(xlabel=planted_lab, ylabel=f"measured {ylab}", title=kind)
        ax.legend(fontsize=8)
    fig.suptitle(f"Recovery ({BEH}; mean ± SEM over sessions)")
    fig.tight_layout()
    os.makedirs(args.out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(join(args.out_dir, f"recovery.{ext}"), dpi=200, bbox_inches="tight")
    print(f"[saved] {args.out_dir}/recovery.png / .pdf")


if __name__ == "__main__":
    main()
