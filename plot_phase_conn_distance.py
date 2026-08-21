"""
plot_phase_conn_distance.py

Across-subjects version of the notebook's "connectivity vs distance" cell, built
from the SAVED phase-FC matrices (build_roi_synchrony.py --stage compute output) instead of recomputing
per-event angles.

WHAT THIS CAN AND CANNOT REPRODUCE
----------------------------------
The saved pickles hold, per session, one AGGREGATE connectivity value per
electrode-pair x electrode-pair for each metric and condition:

    <save_root>/word_on/fc_mats/<cond>/<band>/<sub>_<exp>_<sess>_fc_mats.pkl
        -> {"sid", "reg_full", "plv": MxM, "ppc": MxM, "ciplv": MxM, "pli": MxM}

    cond = baseline  ->  FC over the Word-Off (pre, -700..-100 ms) window
    cond = succ      ->  FC over the Word-On  (post, 0..600 ms)  window
    cond = diff      ->  succ - baseline

M[i, j] is PLV/PPC/ciPLV/PLI between bipolar pair i and pair j (helper.get_pairs
order), pooled over ALL word_on events in that session.

Because each entry is already collapsed across events, these matrices let you make
the connectivity-vs-distance scatter, region heatmaps, seed profiles, and any
across-subject aggregate of the connectivity VALUES -- exactly this script.

They do NOT contain per-event phase angles, so they CANNOT reproduce the polar
"rose" phase-angle histograms (cstat.rose): those need one angle per event, which
is discarded once compute_session_fc reduces events to a single PLV/PPC/... value.
To get rose plots across subjects you must save the per-event angles too (see the
note at the bottom of this file / ask to extend the batch).

Aggregation: connectivity is averaged within distance bins per subject, then mean
+/- SEM is taken ACROSS subjects (mirrors the paper's across-subject confidence
bands), so a subject with many sessions/contacts does not dominate.

Usage:
    python plot_phase_conn_distance.py
    python plot_phase_conn_distance.py --n-sessions 60      # quick test subset
    python plot_phase_conn_distance.py --exclude-same-shank
    python plot_phase_conn_distance.py --rmin 15 --rmax 120 --bin-w 10
"""
from __future__ import annotations

from typing import Sequence
import argparse
import json
import os
import re
from collections import defaultdict
from os.path import join, basename
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import helper
import fc_comparison_functions as fc

# Metric key on disk (lowercase) -> display label used in the notebook.
METRIC_LABELS = {"plv": "PLV", "ppc": "PPC", "ciplv": "ciPLV", "pli": "PLI"}

# Minimum paired subjects in a distance bin before it is tested.
MIN_SUBJECTS = 5


def stars(q: float) -> str:
    """Significance stars from a (FDR-corrected) p-value."""
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def _sid_from_filename(fname: str) -> dict[str, int | str]:
    """R1495J_catFR1_0_fc_mats.pkl -> {sub, exp, sess}."""
    stem = basename(fname).replace("_fc_mats.pkl", "")
    m = re.match(r"^(.+)_([A-Za-z0-9]+)_(\d+)$", stem)
    if not m:
        raise ValueError(f"cannot parse sid from {fname}")
    return {"sub": m.group(1), "exp": m.group(2), "sess": int(m.group(3))}


def _coord_block(p0: pd.DataFrame, prefix: str) -> np.ndarray | None:
    """(N,3) coords for a coordinate-space prefix, or None if absent.

    A prefix of "" means the bare x/y/z columns the old pyFR .mat localizations
    carry (Talairach) -- those subjects have no mni/avg/ind blocks at all.
    """
    cols = ["x", "y", "z"] if prefix == "" else [
        f"{prefix}.x", f"{prefix}.y", f"{prefix}.z"]
    if not all(c in p0.columns for c in cols):
        return None
    return p0[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def _first_available(p0: pd.DataFrame, prefixes: Sequence[str]) -> np.ndarray | None:
    """First coordinate block present, trying `prefixes` in order."""
    for pref in prefixes:
        block = _coord_block(p0, pref)
        if block is not None:
            return block
    return None


# Native-space cascade for grid/strip contacts. `ind` is native FreeSurfer RAS
# and is NOT surface-projected (the snapped variants are ind.corrected /
# ind.snap / ind.dural), so it is a genuine 3D position. `stein.*` and `vox.*`
# are deliberately absent: stein columns exist for ~89% of subjects but are
# entirely NaN, and vox exists for ~1%.
_NATIVE_PREFERENCE = ["ind", "avg", "tal", ""]

# Per-session cache of contacts-derived pair MNI coords, relative to save_root.
MNI_CACHE_SUBDIR = join("electrode_information", "pairs_mni")


def _pair_mni_from_contacts(
    dfrow: pd.Series, cache_dir: str,
) -> dict[str, list[float]] | None:
    """{pair_label: [x, y, z]} MNI centroid per bipolar pair, cached per session.

    The bipolar `pairs` table carries `mni.*` for only ~2% of subjects, but the
    monopolar `contacts` table has it for ~94% (99.9% finite). A bipolar's MNI
    coord is the midpoint of its two contacts' MNI coords -- the same
    construction the pair-level avg.*/ind.* columns already use. Joins on
    contacts.label <-> pairs.contact_label_{1,2}.

    Returns None when contacts cannot be read or carry no MNI.
    """
    tag = fc.ftag(dfrow)
    path = Path(cache_dir) / f"{tag}_pairs_mni.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass  

    try:
        import cmlreaders as cml
        c = cml.CMLReader(str(dfrow["sub"]), str(dfrow["exp"]),
                          int(dfrow["sess"])).load("contacts")
    except Exception:
        return None
    if c is None or not all(col in c.columns
                            for col in ("label", "mni.x", "mni.y", "mni.z")):
        return None

    xyz = c[["mni.x", "mni.y", "mni.z"]].apply(
        pd.to_numeric, errors="coerce").to_numpy(float)
    by_contact = {str(lab): xyz[i] for i, lab in enumerate(c["label"].astype(str))}

    p0 = helper.get_pairs(dfrow)
    out: dict[str, list[float]] = {}
    for lab, c1, c2 in zip(p0["label"].astype(str),
                           p0["contact_label_1"].astype(str),
                           p0["contact_label_2"].astype(str)):
        a, b = by_contact.get(c1), by_contact.get(c2)
        if a is None or b is None:
            continue
        mid = (a + b) / 2.0
        if np.isfinite(mid).all():
            out[lab] = [float(v) for v in mid]

    if out:
        os.makedirs(cache_dir, exist_ok=True)
        path.write_text(json.dumps(out))
    return out or None


def _pair_xyz_lead(
    dfrow: pd.Series, mni_cache_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N,3) pair centroid coords, (N,) lead prefix, (N,) is-depth mask.

    Coordinate space is chosen per electrode type (`type_1`, verified identical
    to `type_2` for every pair in this dataset):

        depth ('D','uD')     -> MNI, derived from the monopolar contacts table
        grid/strip ('G','S') -> native ind.* (see _NATIVE_PREFERENCE)

    Depth pairs with no MNI available fall back to the native cascade, so no
    subject is dropped for lack of MNI.

    NOTE: mixing spaces means a depth<->grid distance spans two frames with
    different origin/scaling. `is_depth` is returned so those cross-type pairs
    can be excluded downstream (--drop-cross-type).
    """
    p0 = helper.get_pairs(dfrow)
    xyz = _first_available(p0, _NATIVE_PREFERENCE)
    if xyz is None:
        raise KeyError(f"no usable coordinate columns {_NATIVE_PREFERENCE}")
    xyz = xyz.copy()

    if "type_1" in p0.columns:
        is_depth = p0["type_1"].astype(str).str.upper().isin(("D", "UD")).to_numpy()
    else:
        is_depth = np.zeros(len(p0), bool)

    if is_depth.any():
        mni = _pair_mni_from_contacts(dfrow, mni_cache_dir)
        if mni:
            labels = p0["label"].astype(str).to_numpy()
            for i in np.flatnonzero(is_depth):
                v = mni.get(labels[i])
                if v is not None:
                    xyz[i] = v

    lead = p0["label"].str.extract(r"^([A-Za-z]+)")[0].to_numpy()
    return xyz, lead, is_depth


def collect_per_subject(
    save_root: str, beh: str, band: str, metrics: Sequence[str],
    rmin: float, rmax: float, exclude_same_shank: bool,
    n_sessions: int | None, drop_cross_type: bool = False,
) -> dict[str, dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]]:
    """Walk the saved pickles and gather (distance, connectivity) per subject.

    Returns per_subject[sub][(metric, cond)] = list of (dist_vec, conn_vec), one
    entry per session, covering every finite upper-triangle electrode-pair whose
    seed-target distance falls in [rmin, rmax] (and, optionally, excluding
    same-shank pairs and depth<->grid cross-type pairs).
    """
    succ_dir = Path(save_root) / beh / "fc_mats" / "succ" / band
    base_dir = Path(save_root) / beh / "fc_mats" / "baseline" / band
    files = sorted(succ_dir.glob("*_fc_mats.pkl"))
    if n_sessions is not None:
        files = files[:n_sessions]
    if not files:
        raise SystemExit(f"no pickles in {succ_dir}")

    mni_cache_dir = join(save_root, MNI_CACHE_SUBDIR)
    conds = {"baseline": base_dir, "succ": succ_dir}
    per_subject: dict[str, dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]] = \
        defaultdict(lambda: defaultdict(list))

    for f in tqdm(files, desc="load sessions"):
        sid = _sid_from_filename(f.name)
        sub = str(sid["sub"])
        dfrow = pd.Series({**sid, "loc": 0, "mon": 0})
        try:
            xyz, lead, is_depth = _pair_xyz_lead(dfrow, mni_cache_dir)
        except Exception as e:
            print(f"[skip] {f.name}: get_pairs failed ({e!r})")
            continue

        n = xyz.shape[0]
        iu = np.triu_indices(n, 1)
        dist_full = np.linalg.norm(xyz[iu[0]] - xyz[iu[1]], axis=1)
        same_shank = lead[iu[0]] == lead[iu[1]]

        keep = np.isfinite(dist_full) & (dist_full >= rmin) & (dist_full <= rmax)
        if exclude_same_shank:
            keep &= ~same_shank
        if drop_cross_type:
            # depth<->grid distances span two coordinate frames; drop them.
            keep &= (is_depth[iu[0]] == is_depth[iu[1]])

        mats: dict[str, dict[str, np.ndarray]] = {}
        ok = True
        for cond, d in conds.items():
            p = d / f.name
            if not p.exists():
                ok = False
                break
            mats[cond] = fc.load_pickle(str(p))
        if not ok:
            continue

        for m in metrics:
            for cond in conds:
                M = mats[cond].get(m)
                if M is None:
                    continue
                vals = np.asarray(M)[iu]
                sel = keep & np.isfinite(vals)
                if sel.any():
                    per_subject[sub][(m, cond)].append(
                        (dist_full[sel], vals[sel]))
    return per_subject


def _subject_row(
    entries: list[tuple[np.ndarray, np.ndarray]], edges: np.ndarray,
) -> np.ndarray:
    """One subject's mean connectivity per distance bin (NaN where empty)."""
    n_bins = len(edges) - 1
    row = np.full(n_bins, np.nan)
    if not entries:
        return row
    d = np.concatenate([e[0] for e in entries])
    v = np.concatenate([e[1] for e in entries])
    idx = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        vv = v[(idx == b) & np.isfinite(v)]
        if vv.size:
            row[b] = vv.mean()
    return row


def _subject_count_row(
    entries: list[tuple[np.ndarray, np.ndarray]], edges: np.ndarray,
) -> np.ndarray:
    """One subject's pair-pair connection COUNT per distance bin.

    NaN (not 0) where the subject contributes nothing to a bin, so averaging
    across subjects covers only the subjects that actually enter that bin --
    the same set the reported `n` counts.
    """
    n_bins = len(edges) - 1
    row = np.full(n_bins, np.nan)
    if not entries:
        return row
    d = np.concatenate([e[0] for e in entries])
    v = np.concatenate([e[1] for e in entries])
    idx = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        cnt = int(((idx == b) & np.isfinite(v)).sum())
        if cnt:
            row[b] = cnt
    return row


def aligned_bin_means(
    per_subject: dict[str, dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]],
    metric: str, cond: str, subjects: list[str], edges: np.ndarray,
) -> np.ndarray:
    """(len(subjects), n_bins) bin means, ROW-ALIGNED to `subjects`.

    Row order follows `subjects` (not dict order) so the baseline and succ
    arrays are paired subject-for-subject for the within-bin paired test.
    """
    return np.vstack([
        _subject_row(per_subject.get(s, {}).get((metric, cond), []), edges)
        for s in subjects
    ])


def aligned_bin_counts(
    per_subject: dict[str, dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]],
    metric: str, cond: str, subjects: list[str], edges: np.ndarray,
) -> np.ndarray:
    """(len(subjects), n_bins) pair-pair connection counts, aligned to `subjects`."""
    return np.vstack([
        _subject_count_row(per_subject.get(s, {}).get((metric, cond), []), edges)
        for s in subjects
    ])


def bin_paired_test(
    B_base: np.ndarray, B_succ: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per distance bin, paired t-test of Word-On vs Word-Off ACROSS subjects.

    B_base / B_succ are (n_subjects, n_bins), row-aligned by subject. Within each
    bin, subjects finite in BOTH conditions are paired and `ttest_rel` (== a
    1-sample t of the per-subject differences vs 0) gives the raw p, then
    Benjamini-Hochberg FDR is applied across bins. Returns
    (p_raw, q, n_subjects, mean_delta) per bin.
    """
    n_bins = B_base.shape[1]
    p_raw = np.full(n_bins, np.nan)
    dmean = np.full(n_bins, np.nan)
    n = np.zeros(n_bins, int)
    for b in range(n_bins):
        off, on = B_base[:, b], B_succ[:, b]
        ok = np.isfinite(off) & np.isfinite(on)
        off, on = off[ok], on[ok]
        n[b] = off.size
        if off.size:
            dmean[b] = np.mean(on - off)
        if off.size >= MIN_SUBJECTS and np.any((on - off) != 0):
            try:
                p_raw[b] = ttest_rel(on, off).pvalue
            except ValueError:
                p_raw[b] = np.nan
    q = np.full(n_bins, np.nan)
    fin = np.isfinite(p_raw)
    if fin.any():
        q[fin] = fdrcorrection(p_raw[fin], alpha=0.05)[1]
    return p_raw, q, n, dmean


# Per-metric colors for the overlaid diff figure.
METRIC_COLORS = {"plv": "tab:blue", "ppc": "tab:orange",
                 "ciplv": "tab:green", "pli": "tab:red"}


def plot_diff_figure(
    per_subject: dict[str, dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]],
    metrics: Sequence[str], subjects: list[str],
    edges: np.ndarray, centers: np.ndarray,
    out_dir: str, beh: str, band: str, exclude_same_shank: bool, n_sub: int,
) -> None:
    """Single panel overlaying the Word-On - Word-Off diff line for all metrics.

    The diff is the per-subject paired difference (succ - baseline) averaged
    ACROSS subjects, with SEM; stars mark bins where the paired t-test (the same
    test as the main figure) survives FDR, colored to each metric's line.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in metrics:
        color = METRIC_COLORS.get(m, "k")
        B_base = aligned_bin_means(per_subject, m, "baseline", subjects, edges)
        B_succ = aligned_bin_means(per_subject, m, "succ", subjects, edges)
        D = B_succ - B_base                       # per-subject paired diff
        mean = np.nanmean(D, axis=0)
        n_eff = np.sum(np.isfinite(D), axis=0)
        sem = np.nanstd(D, axis=0, ddof=1) / np.sqrt(np.maximum(n_eff, 1))
        ax.errorbar(centers, mean, yerr=sem, marker="o", ms=5, lw=1.8,
                    capsize=3, color=color, label=METRIC_LABELS.get(m, m))

        _, q, _, _ = bin_paired_test(B_base, B_succ)
        for b in range(len(centers)):
            if stars(q[b]):
                ax.annotate(stars(q[b]), (centers[b], mean[b] + sem[b]),
                            textcoords="offset points", xytext=(0, 4),
                            ha="center", va="bottom", fontsize=9,
                            fontweight="bold", color=color)

    ax.axhline(0, color="0.6", lw=1.0)
    ax.set_xlabel("seed–target distance (mm)")
    ax.set_ylabel("Word-On − Word-Off  (Δ connectivity)")
    ax.legend(fontsize=9, title="paired Δ, mean ± SEM across subjects")
    shank = "excl. same-shank" if exclude_same_shank else "incl. same-shank"
    fig.suptitle(
        f"Word-On − Word-Off phase connectivity vs distance across {n_sub} "
        f"subjects  ({beh}, {band}, {shank})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(out_dir, exist_ok=True)
    stem = join(out_dir, f"phase_conn_distance_diff_{beh}_{band}")
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=400)
    print(f"[saved] {stem}.pdf / .png")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save-root", default=fc.root_dir)
    p.add_argument("--beh", default="word_on")
    p.add_argument("--band", default="theta_6_12")
    p.add_argument("--metrics", nargs="+", default=list(METRIC_LABELS))
    p.add_argument("--rmin", type=float, default=10.0)
    p.add_argument("--rmax", type=float, default=110.0)
    p.add_argument("--bin-w", type=float, default=10.0, dest="bin_w")
    p.add_argument("--exclude-same-shank", action="store_true")
    p.add_argument("--drop-cross-type", action="store_true",
                   help="drop depth<->grid pairs, whose distance spans the MNI "
                        "and native frames")
    p.add_argument("--n-sessions", type=int, default=None,
                   help="limit number of sessions (quick test)")
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args()

    helper.root_dir = args.save_root
    fc.root_dir = args.save_root

    per_subject = collect_per_subject(
        args.save_root, args.beh, args.band, args.metrics,
        args.rmin, args.rmax, args.exclude_same_shank, args.n_sessions,
        drop_cross_type=args.drop_cross_type)
    subjects = sorted(per_subject)
    n_sub = len(subjects)
    print(f"[collect] {n_sub} subjects with usable pairs")

    edges = np.arange(args.rmin, args.rmax + 1e-9, args.bin_w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    styles = {"baseline": ("Word Off", "tab:blue"), "succ": ("Word On", "tab:red")}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, m in zip(axes.ravel(), args.metrics):
        B = {c: aligned_bin_means(per_subject, m, c, subjects, edges)
             for c in styles}
        binmean: dict[str, np.ndarray] = {}
        binsem: dict[str, np.ndarray] = {}
        for cond, (labl, color) in styles.items():
            mean = np.nanmean(B[cond], axis=0)
            n_eff = np.sum(np.isfinite(B[cond]), axis=0)
            sem = np.nanstd(B[cond], axis=0, ddof=1) / np.sqrt(np.maximum(n_eff, 1))
            binmean[cond], binsem[cond] = mean, sem
            ax.errorbar(centers, mean, yerr=sem, color=color, marker="o", ms=5,
                        lw=1.8, capsize=3, label=f"{labl}")

        # Pair-pair connections underlying each subject's bin mean. Counts are
        # per-condition but effectively identical (same pairs finite in both);
        # succ is reported.
        C = aligned_bin_counts(per_subject, m, "succ", subjects, edges)
        conn_mean = np.nanmean(C, axis=0)
        conn_tot = np.nansum(C, axis=0)

        # Paired Word-On vs Word-Off across subjects, per bin, FDR across bins.
        p_raw, q, n_pair, dmean = bin_paired_test(B["baseline"], B["succ"])
        print(f"\n[{METRIC_LABELS.get(m, m)}] Word On - Word Off, paired "
              f"t-test across subjects:")
        for b in range(len(centers)):
            if n_pair[b]:
                st = stars(q[b]) or "ns"
                praw_s = f"{p_raw[b]:.3f}" if np.isfinite(p_raw[b]) else "  -  "
                q_s = f"{q[b]:.3f}" if np.isfinite(q[b]) else "  -  "
                print(f"   {int(edges[b]):>3}-{int(edges[b+1]):<3}mm  "
                      f"n={n_pair[b]:>3}  conn/subj={conn_mean[b]:>8.1f}  "
                      f"conn_tot={int(conn_tot[b]):>8d}  "
                      f"Δmean={dmean[b]:+.4f}  p={praw_s}  q={q_s}  {st}")
        for b in range(len(centers)):
            st = stars(q[b])
            if st:
                y = np.nanmax([binmean["baseline"][b], binmean["succ"][b]])
                e = np.nanmax([np.nan_to_num(binsem["baseline"][b]),
                               np.nan_to_num(binsem["succ"][b])])
                ax.annotate(st, (centers[b], y + e), textcoords="offset points",
                            xytext=(0, 5), ha="center", va="bottom",
                            fontsize=13, fontweight="bold", color="k")

        ax.axhline(0, color="0.75", lw=0.8)
        ax.set_title(METRIC_LABELS.get(m, m))
        ax.set_xlabel("seed–target distance (mm)")
        ax.set_ylabel(METRIC_LABELS.get(m, m))
        ax.legend(fontsize=8, title="mean ± SEM across subjects")
    shank = "excl. same-shank" if args.exclude_same_shank else "incl. same-shank"
    fig.suptitle(
        f"Phase connectivity vs distance across {n_sub} subjects  "
        f"({args.beh}, {args.band}, {shank})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, f"phase_conn_distance_{args.beh}_{args.band}")
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=400)
    print(f"[saved] {stem}.pdf / .png")

    # Companion figure: just the Word-On - Word-Off diff line, all metrics.
    plot_diff_figure(
        per_subject, args.metrics, subjects, edges, centers,
        args.out_dir, args.beh, args.band, args.exclude_same_shank, n_sub)


if __name__ == "__main__":
    main()
