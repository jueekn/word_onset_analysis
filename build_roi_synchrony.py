"""
build_roi_synchrony.py

Phase synchrony per Burke ROI, end to end: compute the phase-FC matrices, then
collapse them to one score per electrode and draw the 12-ROI figure. Two stages,
the same shape as build_roi_power.py:

  compute  multitaper phase connectivity for every session (--workers N to
           parallelise), one pickle per condition:
               <save_root>/<beh>/fc_mats/<cond>/<band>/<ftag>_fc_mats.pkl
               {"sid", "reg_full", <metric>: electrode x electrode matrix}
           Layout is identical to the main pipeline's output, so the existing
           loaders (fc.load_results, build_subj_region_mats, ...) and the
           distance figures read these unchanged.
  plot     aggregate those pickles -> per-(subject, ROI) table -> box plots

Compute stage
-------------
Drives the SAME canonical `fc.compute_session_fc` path as the main pipeline's
`run_sess_fc` (mne_connectivity `spectral_connectivity_epochs(mode="multitaper",
faverage=True)` with the band collapsed to a single value), but calls it ONCE per
session and fans the result out to every condition, so the two arms and their
difference all come from one compute.

Plot stage
----------
Pairs are binned by seed-target distance, each bin is Z-SCORED across that
session's electrodes, then an electrode's score is the mean of the bins it
populates (fc.collapsed_synchrony):

    s[e, b]   = mean connectivity of electrode e to its partners in bin b
    s'[e, b]  = (s[e, b] - mean_e s[:, b]) / sd_e s[:, b]
    S[e]      = mean over the bins e populates of s'[e, b]

The z-score is WITHIN a distance bin, ACROSS electrodes. It corrects the LEVEL
and SPREAD artifacts of ragged distance coverage; it is not a z-score of one
electrode across its own bins (that would force every electrode to 0 and delete
the signal). Because the score is standardized on the session's own electrodes,
an ROI mean reads "how much more (or less) synchronized than this subject's
average electrode" -- directly comparable across subjects, the same footing as
the z_lo / z_hi panels in build_roi_power.py.

Measures (one panel each, matching the power figure's two-levels-plus-contrast
layout):

  sync_lo     collapsed synchrony in the behavior's `lo` condition
  sync_hi     collapsed synchrony in its `hi` condition
  sync_diff   the saved hi - lo difference matrix, collapsed

`sync_diff` is read from the SAVED difference matrix (fc_mats/diff/), not built
by differencing two collapsed scores. That keeps Rao's order of operations --
difference at the pair level, then aggregate over partners -- and matches
build_power_synchrony.py. Differencing two collapsed scores would NOT be
equivalent, since each condition is standardized by its own per-bin SDs.

word_on / voc contrast two TIME WINDOWS of the same events (lo = baseline);
en / rm contrast two EVENT GROUPS in one window (lo = fail). Condition names and
labels come from fc.BEH_CONTRASTS.

Per ROI, a one-sample t of the subject values vs 0, FDR-corrected across the 12
ROIs -- the same fc.roi_stats call build_roi_power.py makes, meaning two
different things: on sync_diff (already a within-subject difference) it IS the
paired t; on sync_lo / sync_hi it asks whether the ROI departs from the
session's own electrode mean, which is where 0 sits after the z-score.

Usage:
    python build_roi_synchrony.py                        # compute + plot
    python build_roi_synchrony.py --n-sessions 2         # smoke test
    python build_roi_synchrony.py --workers 4            # 4 sessions at a time
    python build_roi_synchrony.py --stage plot           # replot from pickles
    python build_roi_synchrony.py --stage compute --band theta_6_12
    python build_roi_synchrony.py --metric plv --beh en
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

MEASURE_KEYS: tuple[str, ...] = ("sync_lo", "sync_hi", "sync_diff")


def measure_of_cond(beh: str) -> dict[str, str]:
    """{saved cond dir: measure column} -- lo/hi/diff in this behavior's names."""
    lo, hi, diff = fc.beh_conds(beh)
    return {lo: "sync_lo", hi: "sync_hi", diff: "sync_diff"}


def panel_labels(beh: str, metric: str) -> dict[str, str]:
    """{measure column: y-axis label}.

    The axis is in SD UNITS, not in the metric's own units: each distance bin is
    z-scored across the session's electrodes before the bins are averaged, so 1
    means "one SD more synchronized than this subject's average electrode at the
    same distances". The [0, 1] bound of PPC/PLV -- and the [-1, 1] bound of
    their difference -- does not apply to the plotted value, which is why points
    run past 1. The label says so, the same way build_roi_power's z_lo / z_hi
    panels do.
    """
    labels = fc.cond_labels(beh)
    return {measure_of_cond(beh)[cond]:
            f"{lab}\ncollapsed {metric.upper()} (z within distance bin)"
            for cond, lab in labels.items()}


def cond_dir(root: str, beh: str, cond: str, band: str,
             fc_mode: str = "multitaper") -> Path:
    return Path(root) / beh / "fc_mats" / cond / fc.band_dirname(band, fc_mode)


# ------------------------------ compute stage --------------------------------
def run_sess_phase_fc(
    dfrow: pd.Series, save_root: str, beh: str, band: str,
    metrics: Sequence[str], root_dir: str,
) -> str:
    """Compute FC for one session and write one pickle per condition.

    Mirrors `fc.run_sess_fc` (same EEG load + overlap mask + compute_session_fc
    multitaper path), but calls compute_session_fc ONCE and fans the result out
    to every condition of the behavior instead of a single one.

    Returns a short status string for the dispatcher's progress bar.
    """
    import helper
    import fc_comparison_functions as fc

    fc.root_dir = root_dir
    helper.root_dir = root_dir

    metrics = tuple(metrics)
    conds = fc.beh_conds(beh)
    sid = f"{dfrow['sub']}_{dfrow['exp']}_{dfrow['sess']}"

    def out_path(cond: str) -> str:
        return str(cond_dir(save_root, beh, cond, band) /
                   f"{fc.ftag(dfrow)}_fc_mats.pkl")

    def complete(cond: str) -> bool:
        p = out_path(cond)
        if not os.path.exists(p):
            return False
        try:
            cached = fc.load_pickle(p)
        except Exception:
            return False
        return all(m in cached for m in metrics)

    if all(complete(c) for c in conds):
        return f"{sid}: cached"

    pairs = helper.get_pairs(dfrow)
    n_ch_pairs = len(pairs)

    events = fc.load_events(dfrow, beh)
    if events is None:
        return f"{sid}: no events ({beh})"

    mat_eeg, buffer_mask = fc.get_beh_eeg(dfrow, events, save=False)
    n_ch_eeg = np.asarray(mat_eeg.data).shape[1]
    if n_ch_pairs != n_ch_eeg:
        raise ValueError(
            f"{sid}: len(pairs)={n_ch_pairs} but eeg n_ch={n_ch_eeg}; "
            "channel order misaligned for overlap mask.")

    overlap_mask = fc.make_overlap_mask(pairs, n_ch=n_ch_eeg)

    mat = fc.compute_session_fc(
        dfrow, beh=beh, band=band, metrics=metrics,
        overlap_mask=overlap_mask, eeg=mat_eeg, mask=buffer_mask)
    if mat is None:
        return f"{sid}: compute_session_fc returned None"

    reg_full = helper.regionalize_electrodes_by_type(pairs)
    sid_tuple = (dfrow["sub"], dfrow["exp"], int(dfrow["sess"]))

    written, empty = 0, []
    for cond in conds:
        out: dict[str, Any] = {"sid": sid_tuple, "reg_full": reg_full}
        for m in metrics:
            if m in mat["metrics"] and cond in mat["metrics"][m]:
                out[m] = np.squeeze(mat["metrics"][m][cond])
        if not any(m in out for m in metrics):
            # cond not produced for this behavior; writing a metric-less pickle
            # would look like valid output to every downstream reader.
            empty.append(cond)
            continue
        os.makedirs(cond_dir(save_root, beh, cond, band), exist_ok=True)
        fc.save_pickle(out_path(cond), out)
        written += 1
    msg = f"{sid}: wrote {written} cond(s)"
    if empty:
        msg += f"; SKIPPED {empty} (not produced for beh={beh})"
    return msg


# ------------------------------- plot stage ----------------------------------
def collect_electrode_table(
    save_root: str, beh: str, band: str, metric: str, edges: np.ndarray,
    rmin: float, rmax: float, exclude_same_shank: bool,
    n_sessions: int | None, lobe_of: dict[str, str],
) -> pd.DataFrame:
    """Per (subject, electrode): collapsed synchrony in each condition.

    Sessions of the same subject are averaged per electrode label, so a subject
    with 4 sessions does not outweigh one with 1.
    """
    conds = measure_of_cond(beh)
    _, hi_cond, _ = fc.beh_conds(beh)          # the arm every behavior has
    hi_dir = cond_dir(save_root, beh, hi_cond, band)

    # every condition dir must exist, else each session is silently skipped
    missing = [c for c in conds
               if not cond_dir(save_root, beh, c, band).is_dir()]
    if missing:
        raise SystemExit(
            f"missing condition dir(s) {missing} under "
            f"{save_root}/{beh}/fc_mats/*/{band}\nrun the compute stage first: "
            f"python build_roi_synchrony.py --stage compute --beh {beh} "
            f"--band {band}")

    files = sorted(hi_dir.glob("*_fc_mats.pkl"))
    if n_sessions is not None:
        files = files[:n_sessions]
    if not files:
        raise SystemExit(
            f"no FC pickles in {hi_dir}\nrun the compute stage first: "
            f"python build_roi_synchrony.py --stage compute --beh {beh} "
            f"--band {band}")

    cols = list(conds.values())
    rows = []

    for f in tqdm(files, desc="load sessions"):
        mats, ok = {}, True
        for cond in conds:
            p = cond_dir(save_root, beh, cond, band) / f.name
            if not p.exists():
                ok = False
                break
            try:
                mats[cond] = fc.load_pickle(str(p))
            except Exception as e:
                print(f"[skip] {f.name} ({cond}): {e!r}")
                ok = False
                break
        if not ok:
            continue

        dfrow = fc.dfrow_from_sid(mats[hi_cond]["sid"])
        sub = str(dfrow["sub"])
        try:
            xyz, lead = fc.pair_xyz_lead(dfrow)
        except Exception as e:
            print(f"[skip] {f.name}: get_pairs failed ({e!r})")
            continue

        import helper
        labels = helper.get_pairs(dfrow)["label"].astype(str).to_numpy()
        n_ch = xyz.shape[0]
        roi = fc.roi_of_reg_full(mats[hi_cond]["reg_full"], lobe_of)
        iu, dist, keep = fc.pair_distance_mask(
            xyz, lead, rmin, rmax, exclude_same_shank)

        sync = {}
        for cond in conds:
            M = mats[cond].get(metric)
            if M is None:
                break
            got = fc.collapsed_synchrony(M, iu, dist, keep, n_ch, edges)
            if got is None:
                break
            sync[cond], _ = got
        if len(sync) != len(conds):
            continue

        for e in range(n_ch):
            if roi[e] is None:
                continue
            vals_e = [sync[c][e] for c in conds]
            if not all(np.isfinite(v) for v in vals_e):
                continue
            rows.append((sub, labels[e], roi[e], *[float(v) for v in vals_e]))

    if not rows:
        raise SystemExit("no electrodes with a Burke ROI and finite synchrony")
    df = pd.DataFrame(rows, columns=["sub", "label", "roi", *cols])
    return df.groupby(["sub", "label", "roi"], as_index=False)[cols].mean()


def run_plot_stage(
    save_root: str, beh: str, band: str, metric: str, edges: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    lobe_of = fc.load_burke_maps()
    elec_df = collect_electrode_table(
        save_root, beh, band, metric, edges, args.rmin, args.rmax,
        args.exclude_same_shank, args.n_sessions, lobe_of)
    tbl = fc.subject_roi_means(elec_df, MEASURE_KEYS,
                               min_electrodes=args.min_electrodes)
    print(f"[collect] {elec_df['sub'].nunique()} subjects, "
          f"{len(elec_df)} ROI-assigned electrodes")

    # every measure is tested and written to the stats CSV, whether or not it
    # got a panel in this figure
    stats = {m: fc.roi_stats(tbl, m) for m in MEASURE_KEYS}
    c = fc.contrast(beh)
    fc.print_roi_stats(
        stats["sync_diff"],
        f"{c['hi_label']} - {c['lo_label']} collapsed {metric.upper()}, per ROI:")

    labels = panel_labels(beh, metric)
    rng = np.random.default_rng(0)
    fc.roi_figure(
        [(labels[m],
          lambda ax, m=m: fc.roi_panel(ax, tbl, m, rng, stats=stats[m],
                                       style=args.style))
         for m in args.measures],
        args.out_dir, f"roi_synchrony_{beh}_{band}_{metric}")

    fc.write_roi_csvs(args.out_dir, f"roi_synchrony_{beh}_{band}_{metric}",
                      tbl, elec_df, stats)
    return tbl


# ---------------------------------- CLI --------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="both", choices=("compute", "plot", "both"))
    p.add_argument("--fc-mode", default=fc.FC_MODE,
                   choices=list(fc.FC_MODES), dest="fc_mode",
                   help="spectral estimator for the phase metrics. multitaper (default): one band-averaged estimate, no time axis. cwt_morlet: Morlet wavelets, time-resolved, enables latency analyses. Outputs land in a separate <band>__cwt_morlet directory so the two estimators never overwrite each other.")
    fc.add_common_args(p)
    fc.add_distance_args(p)
    p.add_argument("--metrics", nargs="+", default=list(fc.PHASE_METRICS),
                   choices=list(fc.PHASE_METRICS),
                   help="metrics the compute stage stores (the plot stage draws "
                        "the single --metric)")
    p.add_argument("--measures", nargs="+", default=list(MEASURE_KEYS),
                   choices=list(MEASURE_KEYS))
    p.add_argument("--style", default="box", choices=("box", "ci"),
                   help="box: median/IQR + whiskers + mean diamond (default). "
                        "ci: mean + 95%% CI, clearer on small subsets")
    p.add_argument("--min-electrodes", type=int, default=3,
                   help="min electrodes for a subject to contribute an ROI")
    p.add_argument("--out-dir", default=join("figures", "burke_roi_synchrony"))
    args = p.parse_args()

    # Morlet figures go to their own subfolder so the two estimators' figures
    # never overwrite each other, mirroring the <band>__cwt_morlet split on the
    # compute side. An explicit --out-dir is still honoured as the parent.
    if args.fc_mode != "multitaper":
        args.out_dir = join(args.out_dir, args.fc_mode)

    # --fc-mode is WIRED BUT NOT YET LIVE for compute. The flag currently reaches
    # the output-path helper only; the estimator itself is still multitaper
    # (band_power / compute_session_fc take no fc_mode yet, and the path helper
    # is called here without it). Running cwt_morlet would therefore compute
    # MULTITAPER numbers and write them into the multitaper directory -- silently
    # mislabelled. Fail loudly until the threading in CHANGELOG.md section 6 is done.
    if args.fc_mode != "multitaper":
        raise SystemExit(
            f"--fc-mode {args.fc_mode} is not implemented for compute yet.\n"
            "The flag is parsed and the output paths understand it, but the\n"
            "estimator is still multitaper, so this run would produce multitaper\n"
            "numbers labelled as %s. See CHANGELOG.md section 6 (fc_mode threading).\n"
            "Use --fc-mode multitaper for now." % args.fc_mode)
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
    print(f"[setup] {c['lo_label']} (lo) vs {c['hi_label']} (hi); "
          f"conds {list(fc.beh_conds(args.beh))}")
    print(f"[setup] bins {args.rmin}-{args.rmax} mm / {args.bin_w} mm "
          f"(each bin z-scored across electrodes)")

    if args.stage in ("compute", "both"):
        fc.run_compute_stage(
            run_sess_phase_fc, desc="phase FC", root_dir_=root_dir,
            n_sessions=args.n_sessions, n_subjects=args.n_subjects,
            workers=args.workers, save_root=save_root, beh=args.beh, band=args.band,
            metrics=tuple(args.metrics), root_dir=root_dir)

    if args.stage in ("plot", "both"):
        run_plot_stage(save_root, args.beh, args.band, args.metric, edges, args)


if __name__ == "__main__":
    main()
