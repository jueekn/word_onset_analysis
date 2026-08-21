"""
Burke-ROI seed connectivity vs distance (Word On vs Word Off).

Scales up the single-seed "Solomon-style" distance analysis to EVERY electrode
as a seed, then averages the per-seed distance curves within each of Burke
(2013)'s ROIs (6 categories x 2 hemispheres = 12).

Method is identical to the single-seed notebook cell:
  - multitaper cross-spectrum -> per-event phase angle (fc.mt_event_angles math)
  - across-event connectivity via fc.conn_from_z  (PLV / PPC / ciPLV / PLI)
  - targets binned by seed->target distance
Only the scale changes: all seeds, then a two-level average
(per-seed curve, then mean across seeds within an ROI), exactly as the PI asked.

Efficiency: PLV/PPC/ciPLV/PLI are all invariant under swapping seed<->target,
so each unordered pair is computed once and credited to BOTH endpoints' curves.
The multitaper spectra are computed once per session (all channels) instead of
once per pair.

Coordinate policy (per-electrode, for distance geometry):
  - depth (type D / uD): mni.x/y/z, fall back to tal.x/y/z if mni missing
  - grid/strip (type G / S): ind.x/y/z
"""
from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fc_comparison_functions as fc  # sets up root_dir etc. via helper
import helper

# ----------------------------- config ---------------------------------------
SUBJECTS = [("R1065J", "FR1"), ("R1121M", "FR1"), ("R1616S", "catFR1")]

FMIN, FMAX = 6.0, 12.0
MT_BANDWIDTH = 2.0
RMIN, RMAX = 10.0, 100.0
BIN_W = 10.0
EXCLUDE_SAME_SHANK = True
PRE_WIN, POST_WIN = (-700.0, -100.0), (0.0, 600.0)
METRICS = ["PLV", "PPC", "ciPLV", "PLI"]

# Distance bins (shared across everything)
EDGES = np.arange(RMIN, RMAX + 1e-9, BIN_W)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])
NBIN = len(CENTERS)

# 12 Burke ROIs, fixed plotting order
LOBES = ["frontal", "temporal", "parietal", "occipital", "limbic", "hippocampus"]
ROI_ORDER = [f"{h}-{lobe}" for h in ("L", "R") for lobe in LOBES]

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures", "burke_roi_connectivity")

# ------------------------- region / ROI assignment --------------------------
# Region LABELS come from the repo's canonical type-aware cascade
# (helper.regionalize_electrodes_by_type):
#   - depth (D / uD) -> VOLUMETRIC atlases (stein/das/MTL/whole-brain/mni/tal)
#   - grid / strip   -> SURFACE atlases (dk / ind / avg)
# Depths are NOT labeled from the individual surface parcellation (`ind.region`),
# which would snap a deep contact to the nearest cortical gyrus.
#
# NOTE: this is independent of the DISTANCE coordinate policy below
# (mni/tal for depths, ind for grid/strip) -- labels and geometry are separate.


def _load_burke_maps():
    lobe = pd.read_csv(os.path.join(HERE, "region_to_burke_lobe.csv"))
    lobe_of = lobe.set_index("region")["burke_lobe"].to_dict()
    return lobe_of, None


def _finite(v):
    return isinstance(v, (int, float)) and np.isfinite(v)


def _coords_for_electrode(row):
    """Per-electrode coordinate per the user's policy. Returns (x,y,z) or None."""
    t = str(row.get("type_1", "")).upper()
    if t in ("G", "S"):  # grid / strip
        order = [("ind.x", "ind.y", "ind.z")]
    else:                # depth (D / uD) and anything else -> mni, then tal
        order = [("mni.x", "mni.y", "mni.z"), ("tal.x", "tal.y", "tal.z")]
    for cx, cy, cz in order:
        if cx in row and _finite(row[cx]) and _finite(row[cy]) and _finite(row[cz]):
            return float(row[cx]), float(row[cy]), float(row[cz])
    return None


def assign_rois(p0, lobe_of, reg_of=None, sub=None, exp=None):
    """Add columns: _region, _lobe, _hemi, _roi, _src, _x/_y/_z (chosen coords).

    Region label + hemisphere come from the repo's canonical type-aware
    cascade (volumetric for depths, surface for grid/strip). `_src` records
    which atlas column supplied the label, for auditing.

    Pass sub/exp to also merge the session's localization (adds MTL atlas
    columns like stein/das when available); optional.
    """
    p0 = p0.copy()

    # distance coordinates (per-electrode policy) -- needed first so the
    # hemisphere fallback in regionalize can use them if an atlas is silent
    xyz = [_coords_for_electrode(r) for _, r in p0.iterrows()]
    p0["_x"] = [c[0] if c else np.nan for c in xyz]
    p0["_y"] = [c[1] if c else np.nan for c in xyz]
    p0["_z"] = [c[2] if c else np.nan for c in xyz]

    # canonical type-aware region label: 'L amygdala' / 'R hippocampus' / nan
    loc = None
    if sub is not None and exp is not None:
        _loc = helper.get_localization(
            pd.Series({"sub": sub, "exp": exp, "sess": 0, "loc": 0, "mon": 0}))
        loc = _loc if (_loc is not None and len(_loc)) else None
    labels = pd.Series(helper.regionalize_electrodes_by_type(p0, loc), index=p0.index)
    # source atlas for each pair, for transparency
    src = helper.get_atlas_labels_by_type(p0, loc)["atlas"]
    p0["_src"] = src.values

    def parse_hemi(v):
        return v.split(" ", 1)[0] if isinstance(v, str) else None

    def parse_region(v):
        return v.split(" ", 1)[1] if isinstance(v, str) and " " in v else "nan"

    p0["_hemi"] = labels.map(parse_hemi)
    p0["_region"] = labels.map(parse_region)
    p0["_lobe"] = p0["_region"].map(lambda r: lobe_of.get(r, ""))

    def roi(r):
        lobe, hemi = r["_lobe"], r["_hemi"]
        if lobe in LOBES and hemi in ("L", "R"):
            return f"{hemi}-{lobe}"
        return None
    p0["_roi"] = p0.apply(roi, axis=1)
    return p0


# --------------------------- connectivity core ------------------------------
def event_angles_multi(X, w, seed, tgs):
    """Per-event phase angle of seed vs each target in `tgs`.

    Identical math to fc.mt_event_angles but keeps targets separate (that
    function collapses them with a final .mean(1)). Returns (n_events, n_tgs).
    """
    Xs = X[:, [seed]]
    Xt = X[:, tgs]
    csd = np.sum((w * Xs) * np.conj(w * Xt), axis=-2)          # (E, K, F)
    csd = csd * (2.0 / (w * np.conj(w)).real.sum(axis=-2))
    return np.angle(csd.mean(-1))                              # (E, K)


def conn_from_accum(Sz, n, Ssign, metric):
    """conn_from_z, but from accumulated sufficient statistics across events."""
    if n < 2:
        return np.nan
    R = abs(Sz) / n
    if metric == "PLV":
        return R
    if metric == "PPC":
        return (n * R * R - 1) / (n - 1)
    re, im = Sz.real / n, Sz.imag / n
    if metric == "ciPLV":
        return abs(im) / np.sqrt(max(1 - re * re, 1e-12))
    if metric == "PLI":
        return abs(Ssign / n)
    raise ValueError(metric)


def build_eligible_pairs(p0):
    """Unordered eligible pairs (i<j): distance in [RMIN,RMAX], no shared
    contact, not same shank (if configured), both with valid coords."""
    labels = p0["label"].values
    contacts = [set(str(l).split("-")) for l in labels]
    lead = pd.Series(labels, index=p0.index).astype(str).str.extract(r"^([A-Za-z]+)")[0].values
    xyz = p0[["_x", "_y", "_z"]].to_numpy(float)
    valid = np.isfinite(xyz).all(1)
    n = len(p0)

    pairs = []  # (i, j, dist)
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(i + 1, n):
            if not valid[j]:
                continue
            if contacts[i] & contacts[j]:
                continue
            if EXCLUDE_SAME_SHANK and lead[i] == lead[j]:
                continue
            d = float(np.linalg.norm(xyz[i] - xyz[j]))
            if RMIN <= d <= RMAX:
                pairs.append((i, j, d))
    return pairs


def accumulate_session(ses_row, pairs, n_chan):
    """Return per-pair (Sz, n, Ssign) accumulators for pre and post windows,
    for one session. pairs is the list of (i,j,d)."""
    data, t, sf = fc.session_eeg(ses_row, (PRE_WIN[0], POST_WIN[1]))
    # group targets by seed for vectorized angle computation
    by_seed = {}
    for k, (i, j, _) in enumerate(pairs):
        by_seed.setdefault(i, []).append((j, k))

    acc = {
        key: (np.zeros(len(pairs), complex), np.zeros(len(pairs), int),
              np.zeros(len(pairs), float))
        for key in ("pre", "post")
    }
    for key, win in (("pre", PRE_WIN), ("post", POST_WIN)):
        m = (t >= win[0]) & (t <= win[1])
        from mne.time_frequency import psd_array_multitaper
        X, _, w = psd_array_multitaper(
            data[:, :, m], sf, fmin=FMIN, fmax=FMAX, bandwidth=MT_BANDWIDTH,
            adaptive=False, low_bias=False, output="complex", verbose=False)
        Sz, N, Ssign = acc[key]
        for i, tgs_k in by_seed.items():
            tgs = [tg for tg, _ in tgs_k]
            ks = [k for _, k in tgs_k]
            ang = event_angles_multi(X, w, i, tgs)            # (E, K)
            for col, k in enumerate(ks):
                a = ang[:, col]
                a = a[np.isfinite(a)]
                if a.size:
                    z = np.exp(1j * a)
                    Sz[k] += z.sum()
                    N[k] += a.size
                    Ssign[k] += np.sign(z.imag).sum()
    return acc


# ------------------------------ per subject ---------------------------------
def analyze_subject(sub, exp, lobe_of, reg_of, verbose=True):
    dfrow0 = pd.Series({"sub": sub, "exp": exp, "sess": 0, "loc": 0, "mon": 0})
    p0 = helper.get_pairs(dfrow0)
    p0 = assign_rois(p0, lobe_of, reg_of, sub=sub, exp=exp)
    n = len(p0)

    pairs = build_eligible_pairs(p0)
    if verbose:
        n_assigned = p0["_roi"].notna().sum()
        print(f"[{sub} {exp}] {n} electrodes, {n_assigned} in a Burke ROI, "
              f"{len(pairs)} eligible pairs")

    sessions = [s for s in range(20)
                if helper.load_events(
                    pd.Series({"sub": sub, "exp": exp, "sess": s, "loc": 0, "mon": 0}),
                    "word_on") is not None]
    if verbose:
        print(f"[{sub} {exp}] sessions: {sessions}")

    # accumulate across sessions
    Sz = {k: np.zeros(len(pairs), complex) for k in ("pre", "post")}
    N = {k: np.zeros(len(pairs), int) for k in ("pre", "post")}
    Ssign = {k: np.zeros(len(pairs), float) for k in ("pre", "post")}
    for s in sessions:
        ses_row = pd.Series({"sub": sub, "exp": exp, "sess": s, "loc": 0, "mon": 0})
        acc = accumulate_session(ses_row, pairs, n)
        for k in ("pre", "post"):
            sz, nn, ss = acc[k]
            Sz[k] += sz; N[k] += nn; Ssign[k] += ss
        if verbose:
            print(f"    session {s} done")

    # per-pair connectivity per metric per window
    conn = {k: {m: np.array([
        conn_from_accum(Sz[k][p], N[k][p], Ssign[k][p], m) for p in range(len(pairs))
    ]) for m in METRICS} for k in ("pre", "post")}

    dists = np.array([d for _, _, d in pairs])
    binidx = np.clip(np.digitize(dists, EDGES) - 1, 0, NBIN - 1)

    # ---- per-seed curves: each pair credits BOTH endpoints ----
    # seed_curve[metric][window] = (n_elec, NBIN) mean over that seed's targets
    seed_curve = {m: {k: np.full((n, NBIN), np.nan) for k in ("pre", "post")}
                  for m in METRICS}
    # collect, per seed, the (bin, value) contributions
    seed_bin_vals = {s: {k: {m: [[] for _ in range(NBIN)] for m in METRICS}
                         for k in ("pre", "post")} for s in range(n)}
    for p, (i, j, _) in enumerate(pairs):
        b = binidx[p]
        for k in ("pre", "post"):
            for m in METRICS:
                v = conn[k][m][p]
                if np.isfinite(v):
                    seed_bin_vals[i][k][m][b].append(v)
                    seed_bin_vals[j][k][m][b].append(v)
    for s in range(n):
        for k in ("pre", "post"):
            for m in METRICS:
                for b in range(NBIN):
                    vv = seed_bin_vals[s][k][m][b]
                    if vv:
                        seed_curve[m][k][s, b] = np.mean(vv)

    # ---- average per-seed curves within each ROI ----
    roi_of = p0["_roi"].values
    roi_result = {}  # roi -> {metric -> {window -> (mean, sem, n_seeds)}}
    for roi in ROI_ORDER:
        seeds = np.where(roi_of == roi)[0]
        if seeds.size == 0:
            continue
        entry = {}
        for m in METRICS:
            entry[m] = {}
            for k in ("pre", "post"):
                curves = seed_curve[m][k][seeds]           # (n_seeds, NBIN)
                with np.errstate(invalid="ignore"):
                    mean = np.nanmean(curves, axis=0)
                    cnt = np.sum(np.isfinite(curves), axis=0)
                    sd = np.nanstd(curves, axis=0, ddof=1)
                    sem = np.where(cnt > 1, sd / np.sqrt(cnt), np.nan)
                entry[m][k] = (mean, sem, cnt)
        roi_result[roi] = {"n_seeds": int(seeds.size), **entry}

    return {
        "sub": sub, "exp": exp, "p0": p0, "pairs": pairs,
        "conn": conn, "dists": dists, "binidx": binidx,
        "seed_curve": seed_curve, "roi_result": roi_result,
        "centers": CENTERS,
    }


# ------------------------------- plotting -----------------------------------
STYLES = {"pre": ("Word Off", "tab:blue"), "post": ("Word On", "tab:red")}


def plot_subject(res, metric="PLV", save=True):
    sub, exp = res["sub"], res["exp"]
    rr = res["roi_result"]
    fig, axes = plt.subplots(2, 6, figsize=(22, 8), sharex=True)
    for ax, roi in zip(axes.ravel(), ROI_ORDER):
        if roi not in rr:
            ax.set_title(f"{roi}\n(no electrodes)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        e = rr[roi][metric]
        for k, (labl, color) in STYLES.items():
            mean, sem, _ = e[k]
            ax.errorbar(CENTERS, mean, yerr=sem, color=color, marker="o",
                        ms=4, lw=1.6, capsize=2, label=labl)
        ax.set_title(f"{roi}  (n={rr[roi]['n_seeds']} seeds)", fontsize=16)
        ax.axhline(0, color="0.8", lw=0.7)
    for ax in axes[-1]:
        ax.set_xlabel("Seed–target distance (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel(metric)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"{sub} {exp} - {metric} vs. distance", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        os.makedirs(FIGDIR, exist_ok=True)
        path = os.path.join(FIGDIR, f"{sub}_{exp}_{metric}_burke_rois.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print("saved", path)
    return fig


def main(metrics=("PLV",), save_results=True):
    lobe_of, reg_of = _load_burke_maps()
    all_res = {}
    for sub, exp in SUBJECTS:
        res = analyze_subject(sub, exp, lobe_of, reg_of)
        all_res[(sub, exp)] = res
        for m in metrics:
            plot_subject(res, metric=m, save=True)
            plt.close("all")
    if save_results:
        os.makedirs(FIGDIR, exist_ok=True)
        with open(os.path.join(FIGDIR, "results.pkl"), "wb") as f:
            # drop the big per-seed arrays' redundancy is fine; keep everything
            pickle.dump(all_res, f)
        print("saved results.pkl")
    return all_res


if __name__ == "__main__":
    main(metrics=METRICS)
