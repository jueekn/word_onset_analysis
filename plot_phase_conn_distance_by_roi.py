"""
plot_phase_conn_distance_by_roi.py

Burke-ROI version of plot_phase_conn_distance.py. Identical across-subjects
pipeline -- reads the SAVED phase-FC matrices (build_roi_synchrony.py --stage compute output), bins
connectivity by seed-target distance within each subject, then takes mean +/- SEM
ACROSS SUBJECTS -- with ONE addition: pairs are split by the Burke ROI of the
SEED electrode, giving one Word-Off/Word-On distance curve per ROI.

    pair (i,j)  ->  contributes to ROI[i]'s curve (seed = i, target = j)
                and to ROI[j]'s curve (seed = j, target = i)   [M is symmetric]
    per seed:        mean connectivity per distance bin over that seed's targets
    within subject:  mean over SEEDS per bin, per ROI  (equal weight per seed)
    across subjects: mean +/- SEM per bin, per ROI     (equal weight per subject)

Three-level hierarchy (pair -> seed -> subject -> group): a subject's ROI curve
is the average of its per-seed curves, and the group curve averages those across
subjects. This weights every subject equally and every seed equally within a
subject, and makes the across-subject SEM / paired t a valid subject-level test
(no pair/seed pseudo-replication).

Electrode -> ROI uses `reg_full` stored in each pickle (the canonical
regionalize_electrodes_by_type output: volumetric cascade for depths, surface
for grid/strip) mapped through region_to_burke_lobe.csv + hemisphere.

Coordinates, distance filtering, same-shank / cross-type exclusion are all
reused verbatim from plot_phase_conn_distance.py.

Usage:
    python plot_phase_conn_distance_by_roi.py                     # all subjects, theta
    python plot_phase_conn_distance_by_roi.py --n-sessions 60     # quick test
    python plot_phase_conn_distance_by_roi.py --metrics plv
"""
from __future__ import annotations

from typing import Sequence
import argparse
import os
from collections import defaultdict
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import helper
import fc_comparison_functions as fc
import plot_phase_conn_distance as ppcd
import burke_roi_connectivity as b   

# The 4 main task contrasts (config.yaml `behaviors_main`). Each contrast is two
# saved conditions -> two lines; `lo` is the reference/blue line, `hi` the
# active/red line, and the paired test is hi vs lo (matches diff = hi - lo).
# Condition dir names come straight from compute_session_fc:
#   word_on / voc -> PREPOST_SPEC  -> baseline / succ
#   en / rm       -> succ/fail SME -> fail / succ
CONTRASTS = {
    "word_on":      {"beh": "word_on", "lo": "baseline", "hi": "succ",
                     "lo_label": "Word Off",         "hi_label": "Word On",
                     "title": "Word Presentation vs Pre-Word"},
    "encoding":     {"beh": "en",      "lo": "fail",     "hi": "succ",
                     "lo_label": "Not Recalled",     "hi_label": "Recalled",
                     "title": "Encoding"},
    "retrieval":    {"beh": "rm",      "lo": "fail",     "hi": "succ",
                     "lo_label": "Deliberation",           "hi_label": "Recalled",
                     "title": "Retrieval"},
    "vocalization": {"beh": "voc",     "lo": "baseline", "hi": "succ",
                     "lo_label": "Pre-Vocalization", "hi_label": "Vocalization",
                     "title": "Vocalization vs Pre-Vocalization"},
}


def _styles_for(contrast):
    """{cond_dir_name: (line label, color)} for a contrast — lo=blue, hi=red."""
    return {contrast["lo"]: (contrast["lo_label"], "tab:blue"),
            contrast["hi"]: (contrast["hi_label"], "tab:red")}

def _roi_of_reg_full(reg_full, lobe_of):
    """Map each 'L amygdala'/'R hippocampus'/nan label to a Burke ROI (e.g.
    'R-temporal') or None."""
    out = []
    for v in reg_full:
        if not isinstance(v, str) or " " not in v:
            out.append(None)
            continue
        hemi, region = v.split(" ", 1)
        lobe = lobe_of.get(region, "")
        out.append(f"{hemi}-{lobe}" if (lobe in b.LOBES and hemi in ("L", "R")) else None)
    return np.array(out, dtype=object)


def collect_per_subject_by_roi(
    save_root, beh, band, metrics, rmin, rmax, exclude_same_shank,
    n_sessions, drop_cross_type, lobe_of, cond_lo, cond_hi,
    print_bins=False, edges=None,
):
    """per_subject[sub][(roi, metric, cond)][seed_label] = list of
    (dist_vec, conn_vec).

    Same walk as ppcd.collect_per_subject, but each kept pair is filed under the
    Burke ROI of each of its endpoints that has one (seed = that endpoint) AND
    under that seed electrode's label, so per-seed curves can be averaged within
    a subject before averaging across subjects."""
    hi_dir = Path(save_root) / beh / "fc_mats" / cond_hi / band
    lo_dir = Path(save_root) / beh / "fc_mats" / cond_lo / band
    files = sorted(hi_dir.glob("*_fc_mats.pkl"))
    if n_sessions is not None:
        files = files[:n_sessions]
    if not files:
        raise SystemExit(
            f"no pickles in {hi_dir}\n(only word_on is built in this save_root; "
            f"build '{beh}' first via build_roi_synchrony.py --stage compute --beh {beh} "
            f"--conds {cond_lo} {cond_hi} diff)")

    mni_cache_dir = join(save_root, ppcd.MNI_CACHE_SUBDIR)
    conds = {cond_lo: lo_dir, cond_hi: hi_dir}
    # per_subject[sub][(roi,metric,cond)] -> {seed_label: [(dist,conn), ...]}
    per_subject = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for f in tqdm(files, desc="load sessions"):
        sid = ppcd._sid_from_filename(f.name)
        sub = str(sid["sub"])
        dfrow = pd.Series({**sid, "loc": 0, "mon": 0})
        try:
            xyz, lead, is_depth = ppcd._pair_xyz_lead(dfrow, mni_cache_dir)
            labels = helper.get_pairs(dfrow)["label"].astype(str).to_numpy()
        except Exception as e:
            print(f"[skip] {f.name}: get_pairs failed ({e!r})")
            continue

        n = xyz.shape[0]
        iu = np.triu_indices(n, 1)
        i_idx, j_idx = iu
        dist_full = np.linalg.norm(xyz[i_idx] - xyz[j_idx], axis=1)
        same_shank = lead[i_idx] == lead[j_idx]

        keep = np.isfinite(dist_full) & (dist_full >= rmin) & (dist_full <= rmax)
        if exclude_same_shank:
            keep &= ~same_shank
        if drop_cross_type:
            keep &= (is_depth[i_idx] == is_depth[j_idx])

        # load matrices + reg_full for both conditions
        mats, ok = {}, True
        for cond, d in conds.items():
            p = d / f.name
            if not p.exists():
                ok = False
                break
            mats[cond] = fc.load_pickle(str(p))
        if not ok:
            continue

        reg_full = np.asarray(mats[cond_hi].get("reg_full"))
        if reg_full is None or len(reg_full) != n:
            print(f"[skip] {f.name}: reg_full missing/mismatched")
            continue
        roi = _roi_of_reg_full(reg_full, lobe_of)          # (n,) ROI or None
        roi_i, roi_j = roi[i_idx], roi[j_idx]

        if print_bins and edges is not None:            # once, first session
            binidx = np.clip(np.digitize(dist_full, edges) - 1, 0, len(edges) - 2)
            def _tag(idx):                              # "R-frontal | inferior frontal gyrus"
                reg = reg_full[idx]
                reg = reg.split(" ", 1)[1] if isinstance(reg, str) and " " in reg else "unmapped"
                return f"{roi[idx] or 'no-ROI'} | {reg}"
            print(f"\n# seed–target pairs -> region/ROI + distance bin  "
                  f"({sub} {sid['exp']} sess {sid['sess']})")
            kept = np.flatnonzero(keep)
            for k in kept[np.argsort(dist_full[kept])]:  # ascending distance = bin order
                lo, hi = int(edges[binidx[k]]), int(edges[binidx[k] + 1])
                print(f"{labels[i_idx[k]]:>13} [{_tag(i_idx[k]):<38}] -- "
                      f"{labels[j_idx[k]]:<13} [{_tag(j_idx[k]):<38}]  "
                      f"{dist_full[k]:6.1f} mm  bin {lo}-{hi} mm")
            print_bins = False                           # only the first session

        # combined "ROI|seed_label" group key per endpoint (independent of
        # metric/cond); "" where the seed endpoint has no ROI so it is dropped.
        gkey_i = np.array([f"{r}|{labels[i_idx[k]]}" if r is not None else ""
                           for k, r in enumerate(roi_i)], dtype=object)
        gkey_j = np.array([f"{r}|{labels[j_idx[k]]}" if r is not None else ""
                           for k, r in enumerate(roi_j)], dtype=object)

        for m in metrics:
            for cond in conds:
                M = mats[cond].get(m)
                if M is None:
                    continue
                vals = np.asarray(M)[iu]
                finite = keep & np.isfinite(vals)
                # each endpoint is a seed for its own ROI (M is symmetric)
                for gkey in (gkey_i, gkey_j):
                    have = finite & (gkey != "")
                    if not have.any():
                        continue
                    d_sel = dist_full[have]
                    v_sel = vals[have]
                    g_sel = gkey[have].astype(str)
                    order = np.argsort(g_sel, kind="stable")
                    d_sel, v_sel, g_sel = d_sel[order], v_sel[order], g_sel[order]
                    uniq, starts = np.unique(g_sel, return_index=True)
                    bounds = list(starts) + [len(g_sel)]
                    for u, s0, s1 in zip(uniq, bounds[:-1], bounds[1:]):
                        roi_u, seed_lab = u.split("|", 1)
                        per_subject[sub][(roi_u, m, cond)][seed_lab].append(
                            (d_sel[s0:s1], v_sel[s0:s1]))
    return per_subject


def _subject_roi_row(seed_dict, edges):
    """One subject's ROI curve: bin each SEED's targets, average per bin over
    that seed's targets (ppcd._subject_row), then average ACROSS SEEDS. Returns
    (n_bins,) with NaN where no seed contributes."""
    n_bins = len(edges) - 1
    if not seed_dict:
        return np.full(n_bins, np.nan)
    per_seed = np.vstack([
        ppcd._subject_row(entries, edges) for entries in seed_dict.values()
    ])                                                   # (n_seeds, n_bins)
    with np.errstate(invalid="ignore"):
        return np.nanmean(per_seed, axis=0)


def roi_curve(per_subject, roi, metric, cond, subjects, edges):
    """(n_subjects, n_bins) per-subject SEED-AVERAGED bin means for one
    (roi, metric, cond), row-aligned to `subjects`. Each row is that subject's
    mean OVER SEEDS (not pooled over pairs), so across-subject averaging weights
    every subject equally and every seed equally within a subject."""
    return np.vstack([
        _subject_roi_row(per_subject.get(s, {}).get((roi, metric, cond), {}), edges)
        for s in subjects
    ])


def report_roi_counts(per_subject, roi_sub, metric, cond, subjects, edges):
    """Across-subjects count, per distance bin, of connections whose SEED is in
    an ROI matching `roi_sub` (substring, e.g. 'hippocampus' catches L- and R-).
    Session repeats of the same seed->target are deduped by (seed, distance)."""
    nbin = len(edges) - 1
    counts = np.zeros((len(subjects), nbin), int)         # (n_subj, nbin)
    for si, s in enumerate(subjects):
        seen = set()
        for (roi, m, c), seed_dict in per_subject.get(s, {}).items():
            if m != metric or c != cond or roi_sub not in roi:
                continue
            for seed_lab, entries in seed_dict.items():
                for dvec, _ in entries:
                    for d in dvec:
                        key = (roi, seed_lab, round(float(d), 2))
                        if key in seen:
                            continue
                        seen.add(key)
                        bi = int(np.clip(np.digitize(d, edges) - 1, 0, nbin - 1))
                        counts[si, bi] += 1
    print(f"\n# seed-in-'{roi_sub}' connection counts per bin, across subjects "
          f"({metric}/{cond})")
    print(f"{'bin (mm)':>10}  {'#subjects':>9}  {'#connections':>12}  {'median/subj':>11}")
    for b in range(nbin):
        col = counts[:, b]; nz = col[col > 0]
        nsubj = int(nz.size); nconn = int(col.sum())
        med = int(np.median(nz)) if nsubj else 0
        print(f"{int(edges[b]):>4}-{int(edges[b+1]):<4}  {nsubj:>9}  {nconn:>12}  {med:>11}")


def plot_by_roi(per_subject, metric, subjects, edges, centers,
                out_dir, contrast, band, exclude_same_shank):
    beh, cond_lo, cond_hi = contrast["beh"], contrast["lo"], contrast["hi"]
    styles = _styles_for(contrast)                       # {cond: (label, color)}
    fig, axes = plt.subplots(2, 6, figsize=(22, 8), sharex=True)
    for ax, roi in zip(axes.ravel(), b.ROI_ORDER):
        # subjects contributing >=1 pair to this ROI (in the active condition)
        Bs = {c: roi_curve(per_subject, roi, metric, c, subjects, edges)
              for c in styles}
        n_subj = int(np.sum(np.any(np.isfinite(Bs[cond_hi]), axis=1)))
        if n_subj == 0:
            ax.set_title(f"{roi}\n(no subjects)", fontsize=14)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        binmean, binsem = {}, {}
        for cond, (labl, color) in styles.items():
            mean = np.nanmean(Bs[cond], axis=0)
            n_eff = np.sum(np.isfinite(Bs[cond]), axis=0)
            sem = np.nanstd(Bs[cond], axis=0, ddof=1) / np.sqrt(np.maximum(n_eff, 1))
            sem = np.where(n_eff > 1, sem, np.nan)
            binmean[cond], binsem[cond] = mean, sem
            ax.errorbar(centers, mean, yerr=sem, color=color, marker="o", ms=4,
                        lw=1.6, capsize=2, label=labl)

        # paired hi vs lo across subjects, per bin, FDR across bins
        _, q, _, _ = ppcd.bin_paired_test(Bs[cond_lo], Bs[cond_hi])
        for bi in range(len(centers)):
            st = ppcd.stars(q[bi])
            if st:
                y = np.nanmax([binmean[cond_lo][bi], binmean[cond_hi][bi]])
                e = np.nanmax([np.nan_to_num(binsem[cond_lo][bi]),
                               np.nan_to_num(binsem[cond_hi][bi])])
                ax.annotate(st, (centers[bi], y + e), textcoords="offset points",
                            xytext=(0, 4), ha="center", va="bottom",
                            fontsize=11, fontweight="bold", color="k")
        ax.set_title(f"{roi}  ({n_subj} subj)", fontsize=14)
        ax.axhline(0, color="0.8", lw=0.7)
    for ax in axes[-1]:
        ax.set_xlabel("seed–target distance (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ppcd.METRIC_LABELS.get(metric, metric))
    axes[0, 0].legend(fontsize=8)
    shank = "excl. same-shank" if exclude_same_shank else "incl. same-shank"
    fig.suptitle(
        f"{ppcd.METRIC_LABELS.get(metric, metric)} vs distance per Burke ROI — "
        f"{contrast['title']}  (mean ± SEM across subjects; stars = paired t + "
        f"FDR/ROI; {band}, {shank})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(out_dir, exist_ok=True)
    stem = join(out_dir, f"phase_conn_distance_by_roi_{metric}_{beh}_{band}")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"[saved] {stem}.png / .pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save-root", default=fc.root_dir)
    p.add_argument("--contrasts", nargs="+", default=["word_on"],
                   choices=list(CONTRASTS),
                   help="which task contrasts to plot (default: word_on)")
    p.add_argument("--band", default="theta_6_12")
    p.add_argument("--metrics", nargs="+", default=list(ppcd.METRIC_LABELS))
    p.add_argument("--rmin", type=float, default=10.0)
    p.add_argument("--rmax", type=float, default=100.0)
    p.add_argument("--bin-w", type=float, default=10.0, dest="bin_w")
    p.add_argument("--exclude-same-shank", action="store_true", default=True)
    p.add_argument("--drop-cross-type", action="store_true", default=True,
                   help="drop depth<->grid pairs (distance spans MNI vs native frames)")
    p.add_argument("--n-sessions", type=int, default=None)
    p.add_argument("--print-bins", action="store_true",
                   help="print each seed–target pair and its distance bin (first session)")
    p.add_argument("--count-roi", default=None,
                   help="print across-subjects connection counts per bin for seeds in "
                        "this ROI (substring, e.g. 'hippocampus', 'L-hippocampus')")
    p.add_argument("--out-dir", default=join("figures", "burke_roi_group"))
    args = p.parse_args()

    import helper
    helper.root_dir = args.save_root
    fc.root_dir = args.save_root

    lobe_of, _ = b._load_burke_maps()
    edges = np.arange(args.rmin, args.rmax + 1e-9, args.bin_w)
    centers = 0.5 * (edges[:-1] + edges[1:])

    for cname in args.contrasts:
        contrast = CONTRASTS[cname]
        print(f"\n=== contrast: {cname} ({contrast['title']}) ===")
        per_subject = collect_per_subject_by_roi(
            args.save_root, contrast["beh"], args.band, args.metrics,
            args.rmin, args.rmax, args.exclude_same_shank, args.n_sessions,
            args.drop_cross_type, lobe_of, contrast["lo"], contrast["hi"],
            print_bins=args.print_bins, edges=edges)
        subjects = sorted(per_subject)
        print(f"[collect] {len(subjects)} subjects with usable pairs")

        print(f"subjects per ROI ({contrast['hi']}):")
        for roi in b.ROI_ORDER:
            ns = sum(1 for s in subjects
                     if any(k[0] == roi and k[2] == contrast["hi"]
                            for k in per_subject[s]))
            print(f"  {roi:14s} {ns}")

        if args.count_roi:
            report_roi_counts(per_subject, args.count_roi, args.metrics[0],
                              contrast["hi"], subjects, edges)

        for m in args.metrics:
            plot_by_roi(per_subject, m, subjects, edges, centers,
                        args.out_dir, contrast, args.band, args.exclude_same_shank)


if __name__ == "__main__":
    main()
