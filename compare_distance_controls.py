"""
compare_distance_controls.py

Which distance correction removes the most geometry from the per-electrode
synchrony score?

The pipeline under test is the one in build_roi_synchrony.py /
build_power_synchrony.py: pair-level connectivity -> one number per electrode ->
ROI means -> across-subject test. The distance problem lives entirely in the
first arrow. Electrodes do not sample the same partner distances -- a mesial
depth contact and a lateral grid contact have completely different partner
distance distributions -- and connectivity falls off with distance, so a badly
collapsed score reports where the electrode sits rather than what it does.

METHODS COMPARED  (--methods)
-----------------------------
raw      Flat mean over partners (Rao et al. 2025). No correction; the reference
         point that says how big the problem is to begin with.
binz     CURRENT METHOD (fc.collapsed_synchrony). 10 mm bins, z-score each bin
         across the session's electrodes, average the bins an electrode
         populates.
expres   Solomon et al. 2017. Fit a*exp(-d/lambda)+c to the session's pairs,
         keep the residuals, average them per electrode.
localz   Continuous version of binz at the PAIR level: kernel-smoothed local
         mean and SD over log-distance, plus inverse-density weighting so an
         electrode's near partners cannot outvote its far ones.

THE METRIC
----------
    bias share = manufactured VARIANCE / real variance of the electrode score

One number per method. Lower is better; 0.16 means 16% of the electrode-to-
electrode variance a method shows you could be produced by geometry alone.

The null replaces connectivity with pure geometry (real montage, real decay,
real heteroscedasticity, all from that session's own data):

    v_k = mu(d_k) + eps_k,   eps_k ~ N(0, sd(d_k))

so nothing whatsoever differs between electrodes -- the true score of every
electrode is identical, and any spread a method reports there is manufactured.
Dividing by the variance the SAME method reports on the real recordings makes it
scale-free, which is what stops a method from winning by being conservative: one
that shrinks everything toward zero shrinks both terms equally.

Separating BIAS from NOISE, which is the only subtle part. A method can invent
electrode differences that are reproducible (real bias -- more data will never
remove it) or that scatter run to run (sampling noise). The numerator wants the
first. Reps are dealt alternately into two halves and the systematic variance is
the COVARIANCE across electrodes between the two half-means: noise is
independent between halves, so it contributes zero in expectation and drops out
with no noise model at all.

    Do NOT go back to var_between - var_within/n_reps. That needs the noise to
    be independent across ELECTRODES, and it is not -- electrodes share pairs,
    and binz z-scores across electrodes within each bin. It gave estimates that
    kept shrinking as reps were added.

Reported as a variance share rather than its square root on purpose: sqrt of a
noisy near-zero variance is asymmetric (the clip at 0 truncates only the low
side), which inflates exactly the methods that are working. The SD-scale column
is printed for intuition; read `bias share` +/- SEM for anything careful, and
treat an estimate within 2 SEM of 0 as "consistent with no residual bias".

What it does NOT tell you: whether the remaining real-data variance is signal. A
low bias share says the method is not inventing structure; it does not prove the
structure it keeps is physiology. The previous version of this script also ran a
signal-injection pass measuring detection power at a matched false-positive
rate; that is in compare_distance_controls_full.py.bak if the sensitivity
question comes back.

USAGE
    python compare_distance_controls.py --n-sessions 100
    python compare_distance_controls.py --methods raw binz localz --n-reps 30
    python compare_distance_controls.py --band theta_6_12 --cond succ
    python compare_distance_controls.py --workers 8 --n-reps 200
    python compare_distance_controls.py --self-test              # no data needed

One session is one unit of work: each task loads its own pickle and returns a
handful of rows, and a session's seed comes from its position in the file list,
so any --workers value produces bit-identical results.
"""
from __future__ import annotations

import argparse
import os
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import fc_comparison_functions as fc

# Kernel bandwidth in log-mm: ~20% distance resolution (+/-3 mm at 15 mm,
# +/-18 mm at 90 mm). Log-distance keeps the resolution constant in RELATIVE
# terms, which matches both where the decay curves and where the pairs are.
KDE_BW = 0.18


# ----------------------------------------------------------------------------
# local (kernel) moments of connectivity vs distance
# ----------------------------------------------------------------------------
def local_moments(dist, vals, bw=KDE_BW, n_grid=64):
    """Locally-linear smoothed mean and SD of `vals` as a function of log-distance.

    LOCAL LINEAR, not local constant: a Nadaraya-Watson mean is biased wherever
    the curve has slope and the kernel window is one-sided, i.e. at both ends of
    the distance range and anywhere the pair density changes. That bias is
    itself distance-dependent, so it leaks straight back into the score it is
    supposed to clean. Local linear removes the first-order term.

    Evaluated on a grid and interpolated back, so cost is O(n_grid * n_pairs).
    The SD is the local conditional SD of the residuals about that line -- it
    shrinks with distance in real data, which is exactly the second-moment
    artifact that centring alone leaves behind. Returns (mu, sd) per pair.
    """
    x = np.log(np.maximum(dist, 1e-6))
    vals = np.asarray(vals, float)
    if x.size < 20 or not (x.max() > x.min()):
        m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
        return np.full(x.size, m), np.full(x.size, max(s, 1e-12))
    grid = np.linspace(x.min(), x.max(), n_grid)
    U = x[None, :] - grid[:, None]                     # (n_grid, n_pairs)
    W = np.exp(-0.5 * (U / bw) ** 2)
    S0, S1, S2 = W.sum(1), (W * U).sum(1), (W * U ** 2).sum(1)
    T0, T1 = W @ vals, (W * U) @ vals
    det = S0 * S2 - S1 ** 2
    bad = ~np.isfinite(det) | (np.abs(det) < 1e-30) | (S0 < 3)
    det = np.where(bad, 1.0, det)
    a = (S2 * T0 - S1 * T1) / det                      # local intercept = mu
    bslope = (S0 * T1 - S1 * T0) / det
    a = np.where(bad, T0 / np.maximum(S0, 1e-12), a)   # fall back to NW
    bslope = np.where(bad, 0.0, bslope)
    R = vals[None, :] - a[:, None] - bslope[:, None] * U
    var_g = (W * R ** 2).sum(1) / np.maximum(S0, 1e-12)
    # floor the SD at a small fraction of the global SD so a flat stretch of the
    # curve cannot blow up the z-scores
    sd_g = np.maximum(np.sqrt(np.maximum(var_g, 0.0)),
                      0.05 * np.std(vals, ddof=1) + 1e-12)
    return np.interp(x, grid, a), np.interp(x, grid, sd_g)


def _kde_weights(x_e, bw=KDE_BW):
    """1 / (kernel density of this electrode's own partner log-distances).

    Turns "one weight per partner" into "one weight per unit of distance
    covered", so an electrode with 40 partners at 15 mm and 3 at 90 mm does not
    have its score decided by the 15 mm crowd. The continuous version of what
    binz achieves by averaging bins instead of pairs.
    """
    if x_e.size < 3:
        return np.ones_like(x_e)
    D = np.exp(-0.5 * ((x_e[:, None] - x_e[None, :]) / bw) ** 2)
    dens = D.sum(1) / (x_e.size * bw)
    w = 1.0 / np.maximum(dens, np.percentile(dens, 5) * 0.25 + 1e-12)
    return w / w.sum() * x_e.size


# ----------------------------------------------------------------------------
# the four collapse methods:  (vals, sess, ctx) -> S, one score per electrode
# ----------------------------------------------------------------------------
def m_raw(vals, sess, ctx):
    """Flat mean over partners (Rao). Every pair credits both endpoints."""
    n = sess["n_ch"]
    tot = np.bincount(sess["ii"], vals, n) + np.bincount(sess["jj"], vals, n)
    cnt = np.bincount(sess["ii"], None, n) + np.bincount(sess["jj"], None, n)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def electrode_bin_matrix(vals, sess, edges):
    """(n_ch, n_bins) mean connectivity per electrode per distance bin.

    Vectorised equivalent of fc.electrode_bin_matrix (asserted in --self-test).
    """
    n, nb = sess["n_ch"], len(edges) - 1
    b = sess["bin"]
    flat_i, flat_j = sess["ii"] * nb + b, sess["jj"] * nb + b
    tot = np.bincount(flat_i, vals, n * nb) + np.bincount(flat_j, vals, n * nb)
    cnt = np.bincount(flat_i, None, n * nb) + np.bincount(flat_j, None, n * nb)
    tot, cnt = tot.reshape(n, nb), cnt.reshape(n, nb)
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def m_binz(vals, sess, ctx):
    """CURRENT METHOD: per-bin z across electrodes, then mean over bins."""
    S = electrode_bin_matrix(vals, sess, ctx["edges"])
    Z = np.full_like(S, np.nan)
    for b in range(S.shape[1]):
        col = S[:, b]
        ok = np.isfinite(col)
        if ok.sum() < ctx["min_elec_bin"]:
            continue
        sd = np.nanstd(col, ddof=1)
        if not (np.isfinite(sd) and sd > 0):
            continue
        Z[:, b] = np.where(ok, (col - np.nanmean(col)) / sd, np.nan)
    n_ok = np.isfinite(Z).sum(axis=1)
    return np.where(n_ok > 0, np.nansum(Z, axis=1) / np.maximum(n_ok, 1), np.nan)


def _exp_model(d, a, lam, c):
    return a * np.exp(-d / lam) + c


def fit_exp(dist, vals, binned=True, n_bins=60):
    """Least-squares a*exp(-d/lambda)+c, Solomon-style.

    `binned` fits quantile-bin means weighted by count instead of all pairs:
    same estimate to 3 decimals for a smooth 3-parameter model, ~50x faster,
    which matters because the null refits it thousands of times. Falls back to a
    quadratic in log-distance if the fit will not converge.
    """
    d, v = np.asarray(dist, float), np.asarray(vals, float)
    if binned and d.size > 4 * n_bins:
        q = np.quantile(d, np.linspace(0, 1, n_bins + 1))
        q[-1] += 1e-9
        idx = np.clip(np.digitize(d, q) - 1, 0, n_bins - 1)
        cnt = np.bincount(idx, None, n_bins)
        keep = cnt > 0
        dfit = (np.bincount(idx, d, n_bins) / np.maximum(cnt, 1))[keep]
        vfit = (np.bincount(idx, v, n_bins) / np.maximum(cnt, 1))[keep]
        sig = 1.0 / np.sqrt(cnt[keep])
    else:
        dfit, vfit, sig = d, v, None

    span = max(np.ptp(vfit), 1e-9)
    p0 = (span, max(np.ptp(d) / 3.0, 1.0), float(np.min(vfit)))
    try:
        popt, _ = curve_fit(_exp_model, dfit, vfit, p0=p0, sigma=sig,
                            bounds=([-10 * span, 1.0, -10 * span],
                                    [10 * span, 1e4, 10 * span]),
                            maxfev=20000)
        return _exp_model(d, *popt)
    except Exception:
        co = np.polyfit(np.log(np.maximum(dfit, 1e-6)), vfit, 2)
        return np.polyval(co, np.log(np.maximum(d, 1e-6)))


def m_expres(vals, sess, ctx):
    """Solomon: subtract one fitted exponential, average the residuals."""
    return m_raw(vals - fit_exp(sess["dist"], vals, binned=ctx["exp_binned"]),
                 sess, ctx)


def m_localz(vals, sess, ctx):
    """Continuous local z at the pair level + distance-stratified weighting."""
    mu, sd = local_moments(sess["dist"], vals, bw=ctx["bw"])
    z = (vals - mu) / sd
    if not ctx["localz_weight"]:
        return m_raw(z, sess, ctx)
    out = np.full(sess["n_ch"], np.nan)
    for e, idx in enumerate(sess["elec_pairs"]):
        if idx.size == 0:
            continue
        w = _kde_weights(sess["logd"][idx], bw=ctx["bw"])
        out[e] = float(np.sum(w * z[idx]) / np.sum(w))
    return out


def pair_level(name, vals, sess, ctx):
    """The method's correction expressed PER PAIR, for the distance curves.

    The per-electrode collapse is what the methods are for; this is the value
    that goes into it, before the collapse, so the familiar connectivity vs
    distance curve can be drawn after each correction. Units differ by method
    (raw and expres are in the metric's own units, binz and localz in SDs), so
    the curves get one panel each rather than one shared axis.

      raw     the pair value itself -- the uncorrected decay curve
      expres  v - fitted exponential. Flat at 0 unless the exponential is the
              wrong shape; systematic curvature here IS the misfit
      localz  (v - mu(d)) / sd(d). Flat at 0 unless the kernel fit misses --
              boundaries and sharp curvature are where to look
    binz is NOT here: it produces no pair-level value at all. It corrects at the
    (electrode, bin) level, and its curve is built directly from those values by
    `binz_bin_means` below.
    """
    if name == "raw":
        return vals
    if name == "expres":
        return vals - fit_exp(sess["dist"], vals, binned=ctx["exp_binned"])
    if name == "localz":
        mu, sd = local_moments(sess["dist"], vals, bw=ctx["bw"])
        return (vals - mu) / sd
    return None


def binz_bin_means(vals, sess, ctx):
    """(n_bins,) mean over electrodes of the z-scored value in each bin.

    These are literally the numbers that go into the ROI synchrony figure --
    binz's per-electrode score is the mean of Z[e, :] over the bins e populates,
    so Z IS the corrected connectivity and this is its distance profile.

    It is EXACTLY ZERO in every bin, and that is the correction working rather
    than a bug: binz subtracts each bin's own across-electrode mean, so the
    across-electrode mean of what remains is zero by construction. (Its SD is
    exactly 1 for the same reason.) expres and localz are not forced this way --
    they subtract a curve fitted across the whole distance range, so their
    per-bin means are free to be non-zero, and the wiggle in those panels is
    genuine misfit.

    The consequence worth keeping in mind: no bin-level summary can say anything
    about binz. Its information is entirely in WHICH electrode holds which
    z-score, never in the bin marginal.
    """
    S = electrode_bin_matrix(vals, sess, ctx["edges"])
    out = np.full(S.shape[1], np.nan)
    for b in range(S.shape[1]):
        col = S[:, b]
        ok = np.isfinite(col)
        if ok.sum() < ctx["min_elec_bin"]:
            continue
        sd = np.nanstd(col, ddof=1)
        if not (np.isfinite(sd) and sd > 0):
            continue
        out[b] = float(np.mean((col[ok] - np.mean(col[ok])) / sd))
    return out


METHODS = {"raw": m_raw, "binz": m_binz, "expres": m_expres, "localz": m_localz}
METHOD_DESC = {
    "raw":    "flat mean over partners (no correction)",
    "binz":   "10 mm bins, z within bin across electrodes, mean of bins",
    "expres": "Solomon a*exp(-d/lam)+c residuals, mean over pairs",
    "localz": "kernel-local z per pair + inverse-density weighting",
}


# ----------------------------------------------------------------------------
# the null: real geometry, connectivity that depends on nothing but distance
# ----------------------------------------------------------------------------
def simulate(sess, rng):
    """v = mu(d) + eps,  eps ~ N(0, sd(d)), both from this session's own data.

    The decay shape and its heteroscedasticity are real; the electrode identity
    of a pair's endpoints is irrelevant to its value. So the true score of every
    electrode is identical and any spread a method reports is manufactured.
    """
    return sess["mu"] + rng.normal(0.0, 1.0, size=sess["mu"].size) * sess["sd"]


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def _finish_session(s, edges, ctx):
    """Attach the per-session structures the methods reuse."""
    nb = len(edges) - 1
    s["bin"] = np.clip(np.digitize(s["dist"], edges) - 1, 0, nb - 1).astype(int)
    s["logd"] = np.log(np.maximum(s["dist"], 1e-6))
    order = np.argsort(np.concatenate([s["ii"], s["jj"]]), kind="stable")
    both = np.concatenate([s["ii"], s["jj"]])[order]
    pidx = np.concatenate([np.arange(s["dist"].size)] * 2)[order]
    bounds = np.searchsorted(both, np.arange(s["n_ch"] + 1))
    s["elec_pairs"] = [pidx[bounds[e]:bounds[e + 1]] for e in range(s["n_ch"])]
    s["mu"], s["sd"] = local_moments(s["dist"], s["vals"], bw=ctx["bw"])
    return s


def session_files(args):
    """The FC pickles this run covers, in a fixed order.

    The order matters beyond tidiness: a session's index in this list seeds its
    null draws, so the same --seed reproduces the same numbers whether the run
    is serial or spread over the cluster.
    """
    fc_root = args.fc_root or fc.root_dir
    d = Path(fc_root) / args.beh / "fc_mats" / args.cond / args.band
    files = sorted(d.glob("*_fc_mats.pkl"))
    if not files:
        raise SystemExit(
            f"no FC pickles in {d}\nrun: python build_roi_synchrony.py "
            f"--stage compute --beh {args.beh} --band {args.band}")
    if args.n_sessions is not None:
        files = files[:args.n_sessions]
    return [(i, str(f)) for i, f in enumerate(files)]


def load_session(path, cfg, ctx, edges):
    """One FC pickle -> flat pair arrays, or None if it cannot be used."""
    P = fc.load_pickle(path)
    dfrow = fc.dfrow_from_sid(P["sid"])
    M = P.get(cfg["metric"])
    xyz, lead = fc.pair_xyz_lead(dfrow)
    n_ch = xyz.shape[0]
    if M is None or np.asarray(M).shape[0] != n_ch:
        return None

    iu, dist, keep = fc.pair_distance_mask(
        xyz, lead, cfg["rmin"], cfg["rmax"], cfg["exclude_same_shank"])
    vals = np.asarray(M, float)[iu]
    keep &= np.isfinite(vals)
    if keep.sum() < cfg["min_pairs"]:
        return None
    return _finish_session({
        "sub": str(dfrow["sub"]),
        "sess": f"{dfrow['sub']}_{dfrow['exp']}_{dfrow['sess']}",
        "n_ch": n_ch,
        "ii": iu[0][keep].astype(np.int32), "jj": iu[1][keep].astype(np.int32),
        "dist": dist[keep].astype(float), "vals": vals[keep].astype(float),
    }, edges, ctx)


# ----------------------------------------------------------------------------
# the comparison -- one session is one unit of work
# ----------------------------------------------------------------------------
def session_bias_rows(item, cfg=None, ctx=None, edges=None, methods=(),
                      n_reps=20, seed=0, root_dir=None, curves=False):
    """Load one session, run the null `n_reps` times, return this session's rows.

    Returns {"bias": [...], "curves": [...]} -- one bias row per method, and
    (when `curves` is set) the per-distance-bin mean of that method's corrected
    pair values, which is what the distance-curve figure averages over subjects.

    Everything a worker needs arrives as arguments, and it loads its own pickle,
    so the only thing crossing the wire is this handful of rows. `item` is the
    (index, path) pair from session_files -- the index seeds the null draws.
    """
    if root_dir is not None:                 # workers get their own module state
        fc.root_dir = root_dir
        import helper
        helper.root_dir = root_dir

    si, path = item
    sess = load_session(path, cfg, ctx, edges)
    if sess is None:
        return {"bias": [], "curves": []}

    # SPLIT-HALF: reps are dealt alternately into two independent halves, and
    # the systematic variance is the COVARIANCE across electrodes between the
    # two half-means. Noise is independent between halves, so it contributes
    # zero in expectation and drops out with no noise model at all --
    # E[cov] = Var_e(true bias), whatever the noise does.
    #
    # The obvious alternative, var_between - var_within/n_reps, needs the noise
    # to be independent ACROSS ELECTRODES, and here it is not: electrodes share
    # pairs, and binz z-scores across electrodes within each bin. That
    # correction is therefore wrong by an unknown factor, which showed up as a
    # bias estimate that kept shrinking as reps were added.
    n = sess["n_ch"]
    half = {m: [np.zeros(n), np.zeros(n)] for m in methods}
    cnt = {m: [np.zeros(n), np.zeros(n)] for m in methods}
    for rep in range(n_reps):
        v = simulate(sess, np.random.default_rng((seed + rep) * 100003 + si))
        h = rep % 2
        for m in methods:
            S = METHODS[m](v, sess, ctx)
            ok = np.isfinite(S)
            half[m][h] += np.where(ok, S, 0.0)
            cnt[m][h] += ok

    rows = []
    for m in methods:
        S_real = METHODS[m](sess["vals"], sess, ctx)
        # Both terms of the ratio must describe the SAME electrodes: an electrode
        # that scores in the null but not on the real data (or the reverse) would
        # otherwise sit in one variance and not the other.
        good = ((cnt[m][0] >= max(2, 0.4 * n_reps))
                & (cnt[m][1] >= max(2, 0.4 * n_reps))
                & np.isfinite(S_real))
        sd_real = float(np.nanstd(S_real[good], ddof=1)) if good.sum() > 1 else 0.0
        if good.sum() < 8 or not (sd_real > 0):
            continue
        a = half[m][0][good] / cnt[m][0][good]
        b = half[m][1][good] / cnt[m][1][good]
        # signed by construction and deliberately left unclipped: for a method
        # with no real bias it lands negative about half the time, and clipping
        # per session would truncate only the low side, inflating exactly the
        # methods that work. The aggregation clips once, at the end.
        sys_var = float(np.cov(a, b, ddof=1)[0, 1])
        rows.append({"method": m, "sub": sess["sub"], "sess": sess["sess"],
                     "n_elec": int(good.sum()), "n_pairs": int(sess["dist"].size),
                     "sys_var": sys_var, "real_var": sd_real ** 2,
                     "var_ratio": sys_var / sd_real ** 2, "real_sd": sd_real,
                     # per-session view only; the clip makes it upward-biased
                     "bias_fraction": float(np.sqrt(max(sys_var, 0.0))) / sd_real})

    curve_rows = []
    if curves:
        for m in methods:
            # binz corrects per (electrode, bin), so its curve comes straight
            # from those values; the others correct per pair.
            zb = binz_bin_means(sess["vals"], sess, ctx) if m == "binz" else None
            pv = None if m == "binz" else pair_level(m, sess["vals"], sess, ctx)
            if zb is None and pv is None:
                continue
            for b in range(len(edges) - 1):
                sel = sess["bin"] == b
                if sel.sum() < ctx.get("min_pairs_bin", 20):
                    continue
                val = zb[b] if m == "binz" else float(np.nanmean(pv[sel]))
                if not np.isfinite(val):
                    continue
                curve_rows.append(
                    {"method": m, "sub": sess["sub"], "sess": sess["sess"],
                     "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                     "center": float(0.5 * (edges[b] + edges[b + 1])),
                     "n_pairs": int(sel.sum()), "value": float(val)})
    return {"bias": rows, "curves": curve_rows}


def bias_fraction(args, ctx, edges):
    """Run session_bias_rows over every session."""
    fc_root = args.fc_root or fc.root_dir
    fc.root_dir = fc_root
    items = session_files(args)
    cfg = {"metric": args.metric, "rmin": args.rmin, "rmax": args.rmax,
           "exclude_same_shank": args.exclude_same_shank,
           "min_pairs": args.min_pairs}
    kw = dict(cfg=cfg, ctx=ctx, edges=edges, methods=tuple(args.methods),
              n_reps=args.n_reps, seed=args.seed, root_dir=fc_root,
              curves=args.plot_curves)

    print(f"[stage] {len(items)} sessions, {args.n_reps} null reps, "
          f"{args.workers} worker(s)")
    out = fc.run_sessions(session_bias_rows, items, desc="distance controls",
                          workers=args.workers, collect=True, **kw)

    rows = [r for chunk in out for r in chunk["bias"]]
    curves = [r for chunk in out for r in chunk["curves"]]
    if not rows:
        raise SystemExit("no session produced a usable score")
    return pd.DataFrame(rows), pd.DataFrame(curves)


def summarize(per_sess, methods):
    """Sessions -> subjects -> one bias fraction per method.

    Averaged in VARIANCE units (`var_ratio`, which is signed), and only then
    square-rooted. Averaging the per-session SD ratios instead would inherit
    their per-session clipping at zero and overstate every method whose true
    bias is small -- at 20 reps that inflated binz and localz by roughly a
    third. The mean over subjects, not the median, because the signed variance
    ratio is the quantity that averages to the truth.

    A negative pooled ratio means the systematic component is not distinguishable
    from zero at this number of reps; it is reported as 0.0 with a note rather
    than silently clipped upward.
    """
    sub = per_sess.groupby(["method", "sub"], as_index=False)["var_ratio"].mean()
    out = []
    for m in methods:
        v = sub.loc[sub["method"] == m, "var_ratio"].to_numpy(float)
        pooled = float(np.mean(v)) if v.size else np.nan
        sem = (float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan)
        out.append({"method": m, "n_subjects": int(v.size),
                    "bias_share": pooled, "sem": sem,
                    # SD-scale companion; the sqrt of a noisy near-zero variance
                    # is asymmetric, so read `bias_share` for anything careful
                    "bias_fraction": float(np.sqrt(max(pooled, 0.0)))})
    return pd.DataFrame(out)


def plot_bars(summary, out_dir, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(summary))
    ax.bar(x, summary["bias_share"], width=0.6, color="#0072B2", alpha=0.85,
           yerr=summary["sem"], capsize=4, error_kw=dict(lw=1, ecolor="0.3"))
    ax.axhline(0, color="0.7", lw=0.9, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], fontsize=10)
    ax.set_ylabel("bias share\nnull variance / real variance", fontsize=15)
    #ax.set_title("share of the reported electrode-score variance\n"
    #             "that geometry alone can manufacture (+/- SEM)", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    stem = join(out_dir, f"distance_controls_{tag}")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"[saved] {stem}.png / .pdf")
    plt.close(fig)


def plot_curves(curve_df, methods, out_dir, tag, metric):
    """Connectivity vs seed-target distance after each correction.

    Aggregated the way plot_phase_conn_distance.py does it: bin the pairs,
    average within a session, average sessions within a subject, then take the
    mean +/- SEM ACROSS SUBJECTS -- so a subject with 4 sessions counts once and
    the error band is between-subject.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = (curve_df.groupby(["method", "sub", "center"], as_index=False)["value"]
                   .mean())
    fig, axes = plt.subplots(1, len(methods), figsize=(3.6 * len(methods), 3.4),
                             squeeze=False)
    for ax, m in zip(axes[0], methods):
        g = sub[sub["method"] == m].groupby("center")["value"]
        c = np.asarray(sorted(g.groups))
        mu = g.mean().reindex(c).to_numpy(float)
        n = g.size().reindex(c).to_numpy(float)
        se = (g.std(ddof=1).reindex(c).to_numpy(float) / np.sqrt(np.maximum(n, 1)))
        ax.plot(c, mu, "-o", ms=3.5, lw=1.6, color="#0072B2")
        ax.fill_between(c, mu - se, mu + se, color="#0072B2", alpha=0.25, lw=0)
        if m != "raw":
            ax.axhline(0, color="0.7", lw=0.9, zorder=0)
        ax.set_title(f"{m}", fontsize=12)
        ax.set_xlabel("seed-target distance (mm)", fontsize=9)
        ax.set_ylabel(metric.upper() if m in ("raw", "expres") else "SD units",
                      fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    stem = join(out_dir, f"distance_curves_{tag}")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"[saved] {stem}.png / .pdf")
    plt.close(fig)


def self_test():
    """binz here must match the production collapse in fc exactly."""
    rng = np.random.default_rng(0)
    n_ch = 40
    iu = np.triu_indices(n_ch, 1)
    dist = rng.uniform(10, 100, iu[0].size)
    vals = rng.normal(size=dist.size)
    edges = np.arange(10, 101, 10.0)
    sess = {"n_ch": n_ch, "ii": iu[0], "jj": iu[1], "dist": dist, "vals": vals,
            "bin": np.clip(np.digitize(dist, edges) - 1, 0, len(edges) - 2)}
    ctx = {"edges": edges, "min_elec_bin": fc.MIN_ELEC_BIN}

    A = fc.electrode_bin_matrix(vals, dist, iu, n_ch, edges)
    B = electrode_bin_matrix(vals, sess, edges)
    assert np.allclose(np.nan_to_num(A), np.nan_to_num(B), atol=1e-10), \
        "bin matrix mismatch"
    Sa = fc.mean_over_bins(fc.zscore_bins(A))
    Sb = m_binz(vals, sess, ctx)
    assert np.allclose(np.nan_to_num(Sa), np.nan_to_num(Sb), atol=1e-10), \
        "binz mismatch"
    print("[self-test] binz matches fc.collapsed_synchrony exactly")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fc-root", default=None,
                   help="where the FC pickles live; default: fc.root_dir")
    p.add_argument("--beh", default="word_on", choices=fc.BEHAVIORS)
    p.add_argument("--band", default="alpha")
    p.add_argument("--cond", default="diff",
                   help="condition dir to read (succ / baseline / diff)")
    p.add_argument("--metric", default="ppc", choices=fc.PHASE_METRICS)
    p.add_argument("--methods", nargs="+", default=list(METHODS),
                   choices=list(METHODS))
    p.add_argument("--rmin", type=float, default=10.0)
    p.add_argument("--rmax", type=float, default=110.0)
    p.add_argument("--bin-w", type=float, default=10.0, dest="bin_w")
    p.add_argument("--exclude-same-shank", action="store_true", default=True)
    p.add_argument("--min-pairs", type=int, default=200)
    p.add_argument("--n-sessions", type=int, default=None)
    p.add_argument("--n-reps", type=int, default=20,
                   help="null repetitions; more sharpens the bias/noise split")
    p.add_argument("--bw", type=float, default=KDE_BW,
                   help="kernel bandwidth in log-mm (localz)")
    p.add_argument("--no-localz-weight", action="store_true",
                   help="ablate localz's inverse-density weighting")
    p.add_argument("--exp-fit", default="binned", choices=("binned", "full"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=join("figures", "distance_controls"))
    p.add_argument("--plot-curves", action="store_true",
                   help="also plot connectivity vs seed-target distance after "
                        "each correction (mean +/- SEM across subjects)")
    p.add_argument("--min-pairs-bin", type=int, default=5,
                   help="min pairs in a distance bin for that session to "
                        "contribute it to the curves")
    p.add_argument("--self-test", action="store_true")
    # A session here is ~0.5 s for the full rep set, so the whole dataset is
    # ~10 min in one process; --workers earns its keep when --n-reps goes into
    # the hundreds (cost is linear in reps).
    p.add_argument("--workers", type=int, default=1,
                   help="sessions processed at once in separate processes")
    return p.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return

    edges = np.arange(args.rmin, args.rmax + 1e-9, args.bin_w)
    ctx = {"edges": edges, "min_elec_bin": fc.MIN_ELEC_BIN, "bw": args.bw,
           "localz_weight": not args.no_localz_weight,
           "exp_binned": args.exp_fit == "binned",
           "min_pairs_bin": args.min_pairs_bin}
    tag = f"{args.beh}_{args.band}_{args.cond}_{args.metric}"

    print(f"[setup] {args.beh} {args.band} {args.cond} {args.metric}, "
          f"{args.rmin:.0f}-{args.rmax:.0f} mm")
    for m in args.methods:
        print(f"[method] {m:8s} {METHOD_DESC[m]}")

    per_sess, curve_df = bias_fraction(args, ctx, edges)
    summary = summarize(per_sess, args.methods)

    one = per_sess[per_sess["method"] == args.methods[0]]
    print(f"\n[data] {len(one)} sessions, {one['sub'].nunique()} subjects, "
          f"{one['n_pairs'].sum():,} pairs")

    print(f"\n=== bias share: manufactured VARIANCE / real variance of the "
          f"electrode score ({args.n_reps} null reps, split-half) ===")
    print(f"{'method':8s} {'#subj':>6} {'bias share':>11} {'+/- SEM':>9} "
          f"{'(SD scale)':>11}")
    for _, r in summary.iterrows():
        print(f"{r['method']:8s} {r['n_subjects']:>6d} {r['bias_share']:>11.4f} "
              f"{r['sem']:>9.4f} {r['bias_fraction']:>11.3f}")
    best = summary.loc[summary["bias_share"].idxmin(), "method"]
    bs = summary.loc[summary["method"] == best]
    print(f"\n  lower = less geometry left in the score. Best: {best} "
          f"({bs['bias_share'].iloc[0]:.1%} of its score variance).")
    if not (bs["bias_share"].iloc[0] > 2 * bs["sem"].iloc[0]):
        print("  NOTE: that estimate is within 2 SEM of zero -- it is consistent "
              "with no residual\n  bias, but the run is too small to pin it down. "
              "Raise --n-reps / --n-sessions.")

    plot_bars(summary, args.out_dir, tag)
    if args.plot_curves and not curve_df.empty:
        plot_curves(curve_df, args.methods, args.out_dir, tag, args.metric)
        curve_df.to_csv(
            join(args.out_dir, f"distance_curves_{tag}_per_session.csv"),
            index=False)
    os.makedirs(args.out_dir, exist_ok=True)
    per_sess.to_csv(join(args.out_dir, f"distance_controls_{tag}_per_session.csv"),
                    index=False)
    summary.to_csv(join(args.out_dir, f"distance_controls_{tag}_summary.csv"),
                   index=False)
    print(f"[saved] per-session / summary CSVs in {args.out_dir}")


if __name__ == "__main__":
    main()
