"""
build_roi_synchrony.py

Phase synchrony per Burke ROI, end to end: compute the phase-FC matrices, then
collapse them to one score per electrode and draw the 12-ROI figure. Two stages,
the same shape as build_roi_power.py:

  compute  multitaper phase connectivity for every session (--workers N to
           parallelise), word on - word off:
               <save_root>/<beh>/fc_mats/diff/<band>/<ftag>_fc_mats.pkl
               {"sid", "reg_full", <metric>: electrode x electrode matrix}
  plot     aggregate those pickles -> per-(subject, ROI) table -> box plots

Compute stage
-------------
`fc.compute_session_fc`: mne_connectivity `spectral_connectivity_epochs(
mode="multitaper", faverage=True)` on the word off and word on windows, loaded
separately, band collapsed to one value; only their difference is saved.

Plot stage
----------
Pairs are binned by seed-target distance; an electrode's score is the mean over
the bins it populates of its mean connectivity to partners in that bin
(fc.collapsed_synchrony), read from the SAVED hi - lo difference matrix
(fc_mats/diff/): difference at the pair level, then aggregate over partners.
Raw metric units by default, so whole-brain changes show; --zscore standardizes
each bin across the session's electrodes (relative, ROI-vs-ROI only).

Per ROI, a one-sample t of the subject values vs 0 (= paired t), FDR across the
12 ROIs (fc.roi_stats). The figure has one row per band already plotted into
--out-dir (fc.band_contrast_figure). A second figure per band shows the same
contrast per distance bin before the collapse (roi_synchrony_distance_*), FDR
over all ROI x bin cells.

Usage:
    python build_roi_synchrony.py                        # compute + plot
    python build_roi_synchrony.py --n-sessions 2         # smoke test
    python build_roi_synchrony.py --workers 4            # 4 sessions at a time
    python build_roi_synchrony.py --stage plot           # replot from pickles
    python build_roi_synchrony.py --band high_gamma --metric plv
"""
from __future__ import annotations

from typing import Any, Sequence

import argparse
import os
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import fc_comparison_functions as fc



def cond_dir(root: str, beh: str, cond: str, band: str) -> Path:
    return Path(root) / beh / "fc_mats" / cond / band


# ------------------------------ compute stage --------------------------------
def run_sess_phase_fc(
    dfrow: pd.Series, save_root: str, beh: str, band: str,
    metrics: Sequence[str], root_dir: str, simulation_tag: str | None = None,
) -> str:
    """Compute FC for one session and write the hi - lo (diff) matrices.
    Returns a short status string for the dispatcher's progress bar."""
    import helper
    import fc_comparison_functions as fc

    fc.root_dir = root_dir
    helper.root_dir = root_dir
    sid = fc.ftag(dfrow)
    path = cond_dir(save_root, beh, "diff", band) / f"{sid}_fc_mats.pkl"
    if path.exists():
        try:
            if all(m in fc.load_pickle(str(path)) for m in metrics):
                return f"{sid}: cached"
        except Exception:
            pass

    pairs = helper.get_pairs(dfrow)
    mat = fc.compute_session_fc(dfrow, beh, band, metrics,
                                overlap_mask=fc.make_overlap_mask(pairs, n_ch=len(pairs)),
                                simulation_tag=simulation_tag)
    if mat is None:
        return f"{sid}: no events ({beh})"
    out: dict[str, Any] = {"sid": (dfrow["sub"], dfrow["exp"], int(dfrow["sess"])),
                           "reg_full": helper.regionalize_electrodes_by_type(pairs)}
    out.update({m: np.squeeze(mat[m]["diff"]) for m in metrics})
    os.makedirs(path.parent, exist_ok=True)
    fc.save_pickle(str(path), out)
    return f"{sid}: wrote diff"


# ------------------------------- plot stage ----------------------------------
def collect_electrode_table(
    save_root: str, beh: str, band: str, metric: str, edges: np.ndarray,
    rmin: float, rmax: float, exclude_same_shank: bool,
    n_sessions: int | None, lobe_of: dict[str, str], zscore: bool = False,
) -> pd.DataFrame:
    """Per (subject, electrode): collapsed hi - lo synchrony (`sync_diff`), and
    per (subject, electrode, distance bin) the uncollapsed value (`dist_mm` =
    bin centre). Sessions of the same subject are averaged per electrode label,
    so a subject with 4 sessions does not outweigh one with 1.
    """
    import helper
    d = cond_dir(save_root, beh, "diff", band)
    files = sorted(d.glob("*_fc_mats.pkl"))[:n_sessions]
    if not files:
        raise SystemExit(
            f"no FC pickles in {d}\nrun the compute stage first: "
            f"python build_roi_synchrony.py --stage compute "
            f"--band {band}")

    rows, bin_rows = [], []
    centres = (edges[:-1] + edges[1:]) / 2
    for f in tqdm(files, desc="load sessions"):
        try:
            mat = fc.load_pickle(str(f))
        except Exception as e:
            print(f"[skip] {f.name}: {e!r}")
            continue
        dfrow = fc.dfrow_from_sid(mat["sid"])
        try:
            xyz, lead = fc.pair_xyz_lead(dfrow)
        except Exception as e:
            print(f"[skip] {f.name}: get_pairs failed ({e!r})")
            continue
        labels = helper.get_pairs(dfrow)["label"].astype(str).to_numpy()
        n_ch = xyz.shape[0]
        roi = fc.roi_of_reg_full(mat["reg_full"], lobe_of)
        iu, dist, keep = fc.pair_distance_mask(
            xyz, lead, rmin, rmax, exclude_same_shank)
        if mat.get(metric) is None:
            continue
        got = fc.collapsed_synchrony(mat[metric], iu, dist, keep, n_ch, edges, zscore)
        if got is None:
            continue
        sub = str(dfrow["sub"])
        for e, (v, vb) in enumerate(zip(*got)):
            if roi[e] is not None and np.isfinite(v):
                rows.append((sub, labels[e], roi[e], float(v)))
                bin_rows += [(sub, labels[e], roi[e], c, float(x))
                             for c, x in zip(centres, vb) if np.isfinite(x)]

    if not rows:
        raise SystemExit("no electrodes with a Burke ROI and finite synchrony")
    keys = ["sub", "label", "roi"]
    df = pd.DataFrame(rows, columns=[*keys, "sync_diff"])
    bins = pd.DataFrame(bin_rows, columns=[*keys, "dist_mm", "sync_diff"])
    return (df.groupby(keys, as_index=False)[["sync_diff"]].mean(),
            bins.groupby([*keys, "dist_mm"], as_index=False)[["sync_diff"]].mean())


def run_plot_stage(
    save_root: str, beh: str, band: str, metric: str, edges: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    elec_df, bin_df = collect_electrode_table(
        save_root, beh, band, metric, edges, args.rmin, args.rmax,
        args.exclude_same_shank, args.n_sessions, fc.load_burke_maps(), args.zscore)
    tbl = fc.subject_roi_means(elec_df, ["sync_diff"],
                               min_electrodes=args.min_electrodes)
    print(f"[collect] {elec_df['sub'].nunique()} subjects, "
          f"{len(elec_df)} ROI-assigned electrodes")
    stats = {"sync_diff": fc.roi_stats(tbl, "sync_diff")}
    c = fc.contrast(beh)
    fc.print_roi_stats(
        stats["sync_diff"],
        f"{c['hi_label']} - {c['lo_label']} collapsed {metric.upper()}, per ROI:")
    stem = f"roi_synchrony_{beh}_{{band}}_{metric}"
    fc.write_roi_csvs(args.out_dir, stem.format(band=band), tbl, elec_df, stats)
    ylab = (f"{metric.upper()} {c['hi_label']} - {c['lo_label']}"
            + (" (z within distance bin)" if args.zscore else ""))
    fc.band_contrast_figure(args.out_dir, stem, "sync_diff", ylab)

    # the same contrast before collapsing over distance: per subject and ROI, the
    # mean over its electrodes in each distance bin
    g = (bin_df.groupby(["sub", "roi", "dist_mm"])
               .agg(sync_diff=("sync_diff", "mean"), n_elec=("label", "nunique")).reset_index())
    dstats = fc.bin_stats(g[g["n_elec"] >= args.min_electrodes], "dist_mm", "sync_diff")
    dstem = join(args.out_dir, f"roi_synchrony_distance_{beh}_{band}_{metric}")
    dstats.to_csv(f"{dstem}_stats.csv", index=False)
    print(f"[saved] {fc.roi_curve_figure(dstats, 'dist_mm', dstem, 'Seed-partner distance (mm)', ylab)}")
    return tbl


# ---------------------------------- CLI --------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="both", choices=("compute", "plot", "both"))
    fc.add_common_args(p)
    fc.add_distance_args(p)
    p.add_argument("--metrics", nargs="+", default=list(fc.PHASE_METRICS),
                   choices=list(fc.PHASE_METRICS),
                   help="metrics the compute stage stores (the plot stage draws "
                        "the single --metric)")
    p.add_argument("--min-electrodes", type=int, default=3,
                   help="min electrodes for a subject to contribute an ROI")
    p.add_argument("--out-dir", default=join("figures", "burke_roi_synchrony"))
    p.add_argument("--zscore", action="store_true",
                   help="z-score each distance bin across the session's electrodes "
                        "(ROI-vs-ROI only; default raw)")
    args = p.parse_args()

    if args.n_sessions is not None and args.n_subjects is not None:
        raise ValueError("pass only one of --n-sessions / --n-subjects")
    return args


def main() -> None:
    args = parse_args()
    root_dir, save_root = fc.resolve_roots(args)
    c = fc.contrast(args.beh)                    
    edges = np.arange(args.rmin, args.rmax + 1e-9, args.bin_w)
    print(f"[setup] beh={args.beh}  band={args.band} {fc.bands[args.band]} Hz  "
          f"metric={args.metric}")
    print(f"[setup] {c['hi_label']} - {c['lo_label']}")
    print(f"[setup] bins {args.rmin}-{args.rmax} mm / {args.bin_w} mm "
          f"({'each bin z-scored across electrodes' if args.zscore else 'raw'})")

    if args.stage in ("compute", "both"):
        fc.run_compute_stage(
            run_sess_phase_fc, desc="phase FC", root_dir_=root_dir,
            n_sessions=args.n_sessions, n_subjects=args.n_subjects,
            workers=args.workers, save_root=save_root, beh=args.beh, band=args.band,
            metrics=tuple(args.metrics), root_dir=root_dir, simulation_tag=args.simulation_tag)

    if args.stage in ("plot", "both"):
        run_plot_stage(save_root, args.beh, args.band, args.metric, edges, args)


if __name__ == "__main__":
    main()
