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
An electrode's score is its mean hi - lo change over all its partners
(fc.electrode_sync, as Rao et al. 2025; contact-sharing pairs excluded), read
from the SAVED difference matrix (fc_mats/diff/): difference at the pair level,
then aggregate over partners. Distance bins are used only in the distance
figure.

Per ROI, a one-sample t of the subject values vs 0 (= paired t), FDR across the
12 ROIs (fc.roi_stats). The figure has one row per band already plotted into
--out-dir (fc.band_contrast_figure). A second figure per band shows the same
contrast per seed-partner distance bin (roi_synchrony_distance_*), FDR over all
ROI x bin cells; a third, per ROI pair and 200 ms epoch, is drawn on the brain
(roi_synchrony_epochs_*).

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
    """Compute FC for one session and add the hi - lo (diff) matrix and the
    per-epoch changes (`<metric>_epochs`) of each metric to the session pickle
    (metrics already in it are kept). Returns a short status string."""
    import helper
    import fc_comparison_functions as fc

    fc.root_dir = root_dir
    helper.root_dir = root_dir
    sid = fc.ftag(dfrow)
    path = cond_dir(save_root, beh, "diff", band) / f"{sid}_fc_mats.pkl"
    out: dict[str, Any] = fc.load_pickle(str(path)) if path.exists() else {}
    if all(m in out for m in metrics) and out.get("subtract_erp") == fc.SUBTRACT_ERP:
        return f"{sid}: cached"

    pairs = helper.get_pairs(dfrow)
    mat = fc.compute_session_fc(dfrow, beh, band, metrics,
                                overlap_mask=fc.make_overlap_mask(pairs, n_ch=len(pairs)),
                                simulation_tag=simulation_tag)
    if mat is None:
        return f"{sid}: no events ({beh})"
    out.update({"subtract_erp": fc.SUBTRACT_ERP, "sid": (dfrow["sub"], dfrow["exp"], int(dfrow["sess"])),
                "reg_full": helper.regionalize_electrodes_by_type(pairs)})
    for m in metrics:
        out[m] = np.squeeze(mat[m]["diff"])
        if "epochs" in mat[m]:
            out[f"{m}_epochs"] = mat[m]["epochs"]
    os.makedirs(path.parent, exist_ok=True)
    fc.save_pickle(str(path), out)
    return f"{sid}: wrote diff"


# ------------------------------- plot stage ----------------------------------
def collect_electrode_table(
    save_root: str, beh: str, band: str, metric: str, edges: np.ndarray,
    rmin: float, rmax: float, exclude_same_shank: bool,
    n_sessions: int | None, lobe_of: dict[str, str],
) -> pd.DataFrame:
    """Per (subject, electrode): hi - lo synchrony to all partners (`sync_diff`,
    fc.electrode_sync), and -- for the distance figure only -- per distance bin
    (`dist_mm` = bin centre; pairs within [rmin, rmax]). Sessions of the same
    subject are averaged per electrode label.
    """
    import helper
    d = cond_dir(save_root, beh, "diff", band)
    files = fc.session_files(d, "_fc_mats.pkl", n_sessions)
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
        M = np.asarray(mat[metric], float)
        sel = keep & np.isfinite(M[iu])
        S = fc.electrode_bin_matrix(M[iu][sel], dist[sel], (iu[0][sel], iu[1][sel]), n_ch, edges)
        sub = str(dfrow["sub"])
        for e, (v, vb) in enumerate(zip(fc.electrode_sync(M), S)):
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


def collect_epoch_table(
    save_root: str, beh: str, band: str, metric: str,
    n_sessions: int | None, lobe_of: dict[str, str], fine: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (subject, region pair 'a|b', epoch start): mean per-epoch change over
    all the pair's electrode pairs (sessions averaged per subject); and the region
    centroids (mean MNI of their electrodes) with their lobe. Regions are the 12
    ROIs, or with `fine` the anatomical labels (reg_full) within them."""
    rows, xyz_rows = [], []
    for f in fc.session_files(cond_dir(save_root, beh, "diff", band), "_fc_mats.pkl", n_sessions):
        mat = fc.load_pickle(str(f))
        E = mat.get(f"{metric}_epochs")
        if E is None:
            continue
        xyz, _ = fc.pair_xyz_lead(fc.dfrow_from_sid(mat["sid"]))
        roi = fc.roi_of_reg_full(mat["reg_full"], lobe_of)
        lobe = [r and r.split("-", 1)[1] for r in roi]
        if fine:
            roi = [f if r is not None else None for f, r in zip(mat["reg_full"], roi)]
        a, b = np.triu_indices(len(roi), 1)
        ok = np.array([roi[i] is not None and roi[j] is not None for i, j in zip(a, b)], bool)
        a, b = a[ok], b[ok]
        d = pd.DataFrame(E[:, a, b].T, columns=[w[0] for w in fc.EPOCHS])
        d["roi"] = ["|".join(sorted((roi[i], roi[j]))) for i, j in zip(a, b)]
        d = d.groupby("roi").mean().stack().rename("sync_diff").reset_index()
        rows.append(d.rename(columns={"level_1": "epoch_ms"}).assign(sub=str(mat["sid"][0])))
        xyz_rows += [(r, lb, *c) for r, lb, c in zip(roi, lobe, xyz) if r is not None and np.isfinite(c).all()]
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    tbl = pd.concat(rows).groupby(["sub", "roi", "epoch_ms"], as_index=False)["sync_diff"].mean()
    cen = (pd.DataFrame(xyz_rows, columns=["roi", "lobe", "x", "y", "z"])
             .groupby("roi").agg(lobe=("lobe", "first"), x=("x", "mean"), y=("y", "mean"), z=("z", "mean")))
    return tbl, cen


def run_plot_stage(
    save_root: str, beh: str, band: str, metric: str, edges: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    elec_df, bin_df = collect_electrode_table(
        save_root, beh, band, metric, edges, args.rmin, args.rmax,
        args.exclude_same_shank, args.n_sessions, fc.load_burke_maps())
    tbl = fc.subject_roi_means(elec_df, ["sync_diff"],
                               min_electrodes=args.min_electrodes)
    print(f"[collect] {elec_df['sub'].nunique()} subjects, "
          f"{len(elec_df)} ROI-assigned electrodes")
    stats = {"sync_diff": fc.roi_stats(tbl, "sync_diff")}
    c = fc.contrast(beh)
    fc.print_roi_stats(
        stats["sync_diff"],
        f"{c['hi_label']} - {c['lo_label']} {metric.upper()} (mean over partners), per ROI:")
    stem = f"roi_synchrony_{beh}_{{band}}_{{metric}}"
    fc.write_roi_csvs(args.out_dir, stem.format(band=band, metric=metric), tbl, elec_df, stats)
    ylab = f"{{metric}} ({c['hi_label']} vs. {c['lo_label']})"
    fc.band_contrast_figure(args.out_dir, stem, "sync_diff", ylab)

    # the same contrast before collapsing over distance: per subject and ROI, the
    # mean over its electrodes in each distance bin
    g = (bin_df.groupby(["sub", "roi", "dist_mm"])
               .agg(sync_diff=("sync_diff", "mean"), n_elec=("label", "nunique")).reset_index())
    dstats = fc.bin_stats(g[g["n_elec"] >= args.min_electrodes], "dist_mm", "sync_diff")
    dstem = join(args.out_dir, f"roi_synchrony_distance_{beh}_{band}_{metric}")
    dstats.to_csv(f"{dstem}_stats.csv", index=False)
    print(f"[saved] {fc.roi_curve_figure(dstats, 'dist_mm', dstem, 'Seed-partner distance (mm)', ylab.format(metric=fc.METRIC_LABELS.get(metric, metric)))}")

    # time-resolved network: per region pair and 200 ms epoch, FDR over pairs x epochs;
    # 12 ROIs, and the fine anatomical regions within them
    for level, fine in (("", False), ("fine_", True)):
        etbl, cen = collect_epoch_table(save_root, beh, band, metric, args.n_sessions,
                                        fc.load_burke_maps(), fine)
        if not len(etbl):
            continue
        estem = f"roi_synchrony_epochs_{level}{beh}_{{band}}_{{metric}}"
        epath = join(args.out_dir, estem.format(band=band, metric=metric))
        est = fc.bin_stats(etbl, "epoch_ms", "sync_diff", min_subjects=fc.MIN_SUBJECTS_PAIR)
        est.to_csv(f"{epath}_stats.csv", index=False)
        cen.to_csv(f"{epath}_centroids.csv")
        # hubs (Rao): per subject, a region's mean change over all its connections
        # that pass the region-pair floor
        n_pair = etbl.groupby("roi")["sub"].nunique()
        htbl = etbl[etbl["roi"].isin(n_pair.index[n_pair >= fc.MIN_SUBJECTS_PAIR])]
        ab = htbl["roi"].str.split("|", expand=True)
        long = pd.concat([htbl.assign(roi=ab[0]), htbl[ab[0] != ab[1]].assign(roi=ab[1])])
        hub = fc.bin_stats(long.groupby(["sub", "roi", "epoch_ms"], as_index=False)["sync_diff"].mean(),
                           "epoch_ms", "sync_diff", fdr="fdr_tsbky")
        hub.to_csv(f"{epath}_hubs.csv", index=False)
        print(f"[epochs {level or 'roi_'}] {int(hub['t'].notna().sum() / len(fc.EPOCHS))} regions tested, "
              f"{int((hub['q'] < 0.05).sum())} hub x epoch cells at q < .05")
        fc.epoch_network_figure(args.out_dir, estem, ylab)
    return tbl


# ---------------------------------- CLI --------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="both", choices=("compute", "plot", "both"))
    fc.add_common_args(p)
    fc.add_distance_args(p)
    p.add_argument("--metrics", nargs="+", default=None, choices=list(fc.PHASE_METRICS),
                   help="metrics the compute stage stores (default: --metric)")
    p.add_argument("--min-electrodes", type=int, default=1,
                   help="min electrodes for a subject to contribute an ROI")
    p.add_argument("--out-dir", default=join("figures", "burke_roi_synchrony"))
    args = p.parse_args()
    args.metric = args.metric or fc.RUN_METRIC[args.band]
    args.metrics = args.metrics or [args.metric]
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
    print(f"[setup] distance figure bins {args.rmin}-{args.rmax} mm / {args.bin_w} mm")

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
