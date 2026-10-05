"""
build_power_synchrony.py

Power-synchrony correlation per Burke ROI, computed the way Rao et al. 2025
(J Neurosci) computed it: the contrast only (Cohen's d of power vs the saved
hi - lo connectivity), Rao's own contrast-vs-contrast correlation. One row per
band already plotted into --out-dir (fc.band_contrast_figure).

Rao's version (notebook cell 189, `compute_power_synchrony_correlations`):

    per electrode:  power    = Cohen's d of log power        -> (C,)
                    synchrony = mean PPC to ALL partners     -> (C,)
    per session:    pearsonr(power, synchrony) across electrodes
    per subject:    mean r across that subject's sessions
    group:          one-sample t of the subject r's vs 0

This script keeps that structure exactly -- synchrony = mean change to ALL
partners (fc.electrode_sync), correlate across ELECTRODES within a session,
average r within subject, test across subjects -- but computes r within each of
the 12 Burke ROIs rather than over the whole montage.

Reading the SAVED diff matrix keeps Rao's order of operations (difference at
the pair level, then aggregate over partners).

CAVEAT: phase metrics are amplitude-independent but
not SNR-independent -- a cleaner signal gives a better-conditioned phase estimate
and so higher phase locking. The contrast cancels the STATIC part of that, but not the
dynamic part: an electrode whose power rises also gets a better phase estimate,
so its PPC rises. Check a positive r against a surrogate that
preserves the power change but destroys true connectivity.

Inputs (both must already exist):
    <root>/<beh>/power/<band>/<ftag>_power.pkl          build_roi_power.py
    <root>/<beh>/fc_mats/<cond>/<band>/<ftag>_fc_mats.pkl
                                                       build_roi_synchrony.py

Usage:
    python build_power_synchrony.py
    python build_power_synchrony.py --band alpha --n-sessions 30
"""
from __future__ import annotations

import argparse
import os
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import fc_comparison_functions as fc


def collect(save_root, beh, band, metric, args, lobe_of):
    """Per (session, ROI): r across the ROI's electrodes between power d and
    synchrony change (mean over all partners)."""
    from scipy.stats import pearsonr
    files = fc.session_files(Path(save_root) / beh / "power" / band, "_power.pkl", args.n_sessions)
    rows = []
    for f in tqdm(files, desc="load sessions"):
        P = fc.load_pickle(str(f))
        fpath = Path(save_root) / beh / "fc_mats" / "diff" / band / f.name.replace("_power.pkl", "_fc_mats.pkl")
        if not fpath.exists() or metric not in (M := fc.load_pickle(str(fpath))):
            continue
        sync = fc.electrode_sync(M[metric])
        power = np.asarray(P["cohens_d"], float)
        if len(sync) != len(power):
            print(f"[skip] {f.name}: {len(sync)} FC vs {len(power)} power channels")
            continue
        roi = fc.roi_of_reg_full(P["reg_full"], lobe_of)
        for r_name in fc.ROI_ORDER:
            m = (roi == r_name) & np.isfinite(power) & np.isfinite(sync)
            if m.sum() >= fc.MIN_ELEC_CORR:
                rows.append({"sub": str(P["sid"][0]), "sess": int(P["sid"][2]), "roi": r_name,
                             "cond": "diff", "n_elec": int(m.sum()),
                             "r": float(pearsonr(power[m], sync[m])[0])})
    if not rows:
        raise SystemExit(f"no (session, ROI) cell had enough electrodes ({save_root}/{beh}, {band})")
    return pd.DataFrame(rows)


def to_subject(df, keys):
    """Sessions -> subjects: plain mean of r, as Rao does (no Fisher z)."""
    return df.groupby(["sub", *keys], as_index=False)["r"].mean()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fc.add_common_args(p, compute=False)
    p.add_argument("--metric", default=None, choices=fc.PHASE_METRICS,
                   help="default: the band's metric in config runs")
    p.add_argument("--out-dir", default=join("figures", "power_synchrony"))
    args = p.parse_args()
    args.metric = args.metric or fc.RUN_METRIC[args.band]
    return args


def main():
    args = parse_args()

    _, save_root = fc.resolve_roots(args)
    print(f"[setup] {args.beh} {args.band} {args.metric}")
    sub_r = to_subject(collect(save_root, args.beh, args.band, args.metric, args,
                               fc.load_burke_maps()), ["roi", "cond"])

    # One-sample t of the subject r's vs 0, BH-FDR across the 12 ROIs.
    stats = {"diff": fc.roi_stats(sub_r, "r")}
    c = fc.contrast(args.beh)
    fc.print_roi_stats(stats["diff"], f"{c['hi_label']} - {c['lo_label']}: power-synchrony r per ROI "
                       f"(across electrodes within session, mean over sessions/subject)")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.beh}_{args.band}_{args.metric}"
    sub_r.to_csv(join(args.out_dir, f"power_synchrony_{tag}_per_subject.csv"),
                 index=False)
    pd.concat([s.assign(cond=c) for c, s in stats.items()]).to_csv(
        join(args.out_dir, f"power_synchrony_{tag}_stats.csv"), index=False)
    print(f"[saved] per-subject / stats CSVs in {args.out_dir}")
    fc.band_contrast_figure(args.out_dir, f"power_synchrony_{args.beh}_{{band}}_{{metric}}", "r",
                            f"r, power d vs. {{metric}} ({c['hi_label']} vs. {c['lo_label']})")


if __name__ == "__main__":
    main()
