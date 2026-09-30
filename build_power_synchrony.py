"""
build_power_synchrony.py

Power-synchrony correlation per Burke ROI, computed the way Rao et al. 2025
(J Neurosci) computed it: one panel for the behavior's `lo` arm, one for its
`hi` arm, and one for the contrast (Cohen's d of power vs the saved delta
connectivity), which is the panel that matches Rao's own contrast-vs-contrast
correlation.

Rao's version (notebook cell 189, `compute_power_synchrony_correlations`):

    per electrode:  power    = Cohen's d of log power        -> (C,)
                    synchrony = mean PPC to ALL partners     -> (C,)
    per session:    pearsonr(power, synchrony) across electrodes
    per subject:    mean r across that subject's sessions
    group:          one-sample t of the subject r's vs 0

This script keeps that structure exactly -- correlate across ELECTRODES within a
session, average r within subject, test across subjects -- and changes two things:

  1. the correlation is computed within each of the 12 Burke ROIs, not over the
     whole montage, so `r` is reported per ROI;
  2. synchrony is collapsed across distance bins rather than averaged flat over
     all partners (fc.collapsed_synchrony; see below).

Why the distance collapse
-------------------------
Rao averages each electrode's PPC over every partner. That is only fair when
electrodes sample the same partner distances -- and they do not: a mesial depth
and a lateral grid contact have very different distance distributions, and PPC
falls off steeply with distance, so a flat mean partly reports geometry rather
than physiology. Binning by distance and z-scoring each bin across the session's
electrodes before averaging the bins makes the per-electrode score read "how much
more (or less) synchronized than expected given the distances it happens to
sample". The full argument for both moments of that z-score is in
fc.collapsed_synchrony / fc.zscore_bins.

Per-bin correlations are always reported (`*_per_bin.csv`): power vs each bin's
z-scored column separately. If r is flat across distance the collapse choice does
not matter; if it is not, that is the more informative result.

The diff panel is the Rao-exact one: his correlation is CONTRAST vs CONTRAST --
power Cohen's d against a raw connectivity difference. Reading the SAVED diff
matrix keeps his order of operations (difference at the pair level, then
aggregate over partners); differencing two collapsed scores would not be
equivalent, since bins are z-scored per condition.

CAVEAT, and it applies to the diff panel too: PPC is amplitude-independent but
not SNR-independent -- a cleaner signal gives a better-conditioned phase estimate
and so a higher PPC. The contrast cancels the STATIC part of that, but not the
dynamic part: an electrode whose power rises also gets a better phase estimate,
so its PPC rises. Check a positive diff-panel r against a surrogate that
preserves the power change but destroys true connectivity.

Inputs (both must already exist):
    <root>/<beh>/power/<band>/<ftag>_power.pkl          build_roi_power.py
    <root>/<beh>/fc_mats/<cond>/<band>/<ftag>_fc_mats.pkl
                                                       build_roi_synchrony.py

Usage:
    python build_power_synchrony.py
    python build_power_synchrony.py --n-sessions 30 --metric ppc
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


def power_key_of_cond(beh):
    """{saved cond dir: column in the power pickle to correlate against}.

    The two arms use the mean power in that condition; the diff arm uses
    the per-electrode Cohen's d, which is the same contrast Rao correlates.
    """
    lo, hi, diff = fc.beh_conds(beh)
    return {lo: "pow_lo", hi: "pow_hi", diff: "cohens_d"}


def collect(save_root, fc_root, beh, band, metric, edges, args, lobe_of):
    """Per (session, ROI, cond): r, per-bin r, and the ROI-mean bar values."""
    from scipy.stats import pearsonr

    conds = power_key_of_cond(beh)
    pow_dir = Path(save_root) / beh / "power" / fc.band_dirname(band, args.fc_mode)
    files = sorted(pow_dir.glob("*_power.pkl"))
    if args.n_sessions is not None:
        files = files[:args.n_sessions]
    if not files:
        raise SystemExit(
            f"no power pickles in {pow_dir}\n"
            f"run: python build_roi_power.py --stage compute --beh {beh} "
            f"--band {band}")

    missing = [c for c in conds
               if not (Path(fc_root) / beh / "fc_mats" / c
                       / fc.band_dirname(band, args.fc_mode)).is_dir()]
    if missing:
        raise SystemExit(
            f"missing condition dir(s) {missing} under "
            f"{fc_root}/{beh}/fc_mats/*/{band}\n"
            f"run: python build_roi_synchrony.py --stage compute --beh {beh} "
            f"--band {band}")

    nb = len(edges) - 1
    rows, binrows, bars = [], [], []

    for f in tqdm(files, desc="load sessions"):
        try:
            P = fc.load_pickle(str(f))
        except Exception as e:
            print(f"[skip] {f.name}: {e!r}")
            continue
        dfrow = fc.dfrow_from_sid(P["sid"])
        sub, sess = str(dfrow["sub"]), int(dfrow["sess"])
        fc_name = f.name.replace("_power.pkl", "_fc_mats.pkl")

        try:
            xyz, lead = fc.pair_xyz_lead(dfrow)
        except Exception as e:
            print(f"[skip] {f.name}: get_pairs failed ({e!r})")
            continue
        n_ch = xyz.shape[0]
        if n_ch != len(P["labels"]):
            print(f"[skip] {f.name}: {n_ch} pairs vs "
                  f"{len(P['labels'])} power channels")
            continue

        iu, dist, keep = fc.pair_distance_mask(
            xyz, lead, args.rmin, args.rmax, args.exclude_same_shank)
        roi = fc.roi_of_reg_full(P["reg_full"], lobe_of)

        for cond, pow_key in conds.items():
            fpath = (Path(fc_root) / beh / "fc_mats" / cond
                     / fc.band_dirname(band, args.fc_mode) / fc_name)
            if not fpath.exists():
                continue
            try:
                M = fc.load_pickle(str(fpath))[metric]
            except Exception as e:
                print(f"[skip] {fc_name} ({cond}): {e!r}")
                continue

            got = fc.collapsed_synchrony(M, iu, dist, keep, n_ch, edges)
            if got is None:
                continue
            sync, Sc = got

            power = np.asarray(P[pow_key], float)      # per-electrode power
            # bars only: put both on the session's own SD scale (r is invariant)
            zp, zs = fc.zscore_across(power), fc.zscore_across(sync)

            for r_name in fc.ROI_ORDER:
                m = (roi == r_name) & np.isfinite(power) & np.isfinite(sync)
                if m.sum():
                    bars.append({"sub": sub, "sess": sess, "roi": r_name,
                                 "cond": cond, "n_elec": int(m.sum()),
                                 "power_z": float(np.nanmean(zp[m])),
                                 "sync_z": float(np.nanmean(zs[m]))})
                if m.sum() >= fc.MIN_ELEC_CORR:
                    rows.append({"sub": sub, "sess": sess, "roi": r_name,
                                 "cond": cond, "n_elec": int(m.sum()),
                                 "r": float(pearsonr(power[m], sync[m])[0])})
                # per-bin r: power vs that bin's z-scored column
                for bi in range(nb):
                    mb = ((roi == r_name) & np.isfinite(power)
                          & np.isfinite(Sc[:, bi]))
                    if mb.sum() >= fc.MIN_ELEC_CORR:
                        binrows.append({
                            "sub": sub, "sess": sess, "roi": r_name,
                            "cond": cond, "bin_lo": edges[bi],
                            "bin_hi": edges[bi + 1], "n_elec": int(mb.sum()),
                            "r": float(pearsonr(power[mb], Sc[mb, bi])[0])})

    if not rows:
        raise SystemExit("no (session, ROI) cell had enough electrodes")
    return pd.DataFrame(rows), pd.DataFrame(binrows), pd.DataFrame(bars)


def to_subject(df, keys):
    """Sessions -> subjects: plain mean of r, as Rao does (no Fisher z)."""
    return df.groupby(["sub", *keys], as_index=False)["r"].mean()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fc.add_common_args(p, compute=False)
    fc.add_distance_args(p)
    p.add_argument("--fc-mode", default=fc.FC_MODE,
                   choices=list(fc.FC_MODES), dest="fc_mode",
                   help="spectral estimator for the phase metrics. multitaper (default): one band-averaged estimate, no time axis. cwt_morlet: Morlet wavelets, time-resolved, enables latency analyses. Outputs land in a separate <band>__cwt_morlet directory so the two estimators never overwrite each other.")
    p.add_argument("--fc-root", default=None,
                   help="where the phase-FC pickles live (build_roi_synchrony.py); "
                        "defaults to --save-root")
    p.add_argument("--out-dir", default=join("figures", "power_synchrony"))
    p.add_argument("--conds", nargs="+", default=None,
                   help="conditions to draw (default all: lo, hi, diff); stats CSV keeps all")
    return p.parse_args()


def main():
    args = parse_args()

    # Morlet figures go to their own subfolder so the two estimators' figures
    # never overwrite each other, mirroring the <band>__cwt_morlet split on the
    # compute side. An explicit --out-dir is still honoured as the parent.
    if args.fc_mode != "multitaper":
        args.out_dir = join(args.out_dir, args.fc_mode)
    _, save_root = fc.resolve_roots(args)
    fc_root = args.fc_root or save_root
    edges = np.arange(args.rmin, args.rmax + 1e-9, args.bin_w)
    lobe_of = fc.load_burke_maps()
    labels = fc.cond_labels(args.beh)

    print(f"[setup] power={save_root}\n[setup] fc   ={fc_root}")
    print(f"[setup] {args.beh} {args.band} {args.metric}  "
          f"bins {args.rmin}-{args.rmax} mm / {args.bin_w} mm  "
          f"(each bin z-scored across electrodes)")

    sess_r, bin_r, bars = collect(save_root, fc_root, args.beh, args.band,
                                  args.metric, edges, args, lobe_of)

    sub_r = to_subject(sess_r, ["roi", "cond"])
    sub_bin = to_subject(bin_r, ["roi", "cond", "bin_lo", "bin_hi"])
    bar_sub = (bars.groupby(["sub", "roi", "cond"], as_index=False)
                   [["power_z", "sync_z"]].mean())

    # One-sample t of the subject r's vs 0, BH-FDR across the 12 ROIs -- the same
    # per-ROI test the power and synchrony figures run, on `r` instead of a
    # per-subject measure.
    conds = fc.beh_conds(args.beh)
    stats = {c: fc.roi_stats(sub_r[sub_r["cond"] == c].copy(), "r")
             for c in conds}
    for cond in conds:
        fc.print_roi_stats(
            stats[cond],
            f"{labels[cond]}: power-synchrony r per ROI (across electrodes "
            f"within session, mean over sessions/subject)")

    fc.roi_figure(
        [(f"r ({labels[cond]})",
          lambda ax, cond=cond: fc.roi_bar_panel(ax, stats[cond]))
         for cond in (args.conds or conds)],
        args.out_dir, f"power_synchrony_{args.beh}_{args.band}_{args.metric}",
        height=4.0)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.beh}_{args.band}_{args.metric}"
    sub_r.to_csv(join(args.out_dir, f"power_synchrony_{tag}_per_subject.csv"),
                 index=False)
    sub_bin.to_csv(join(args.out_dir, f"power_synchrony_{tag}_per_bin.csv"),
                   index=False)
    bar_sub.to_csv(join(args.out_dir, f"power_synchrony_{tag}_bars.csv"),
                   index=False)
    pd.concat([s.assign(cond=c) for c, s in stats.items()]).to_csv(
        join(args.out_dir, f"power_synchrony_{tag}_stats.csv"), index=False)
    print(f"[saved] per-subject / per-bin / bars / stats CSVs in {args.out_dir}")


if __name__ == "__main__":
    main()
