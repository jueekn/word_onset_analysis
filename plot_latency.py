"""plot_latency.py -- order of activation across ROIs from high-gamma power.

Reads compute.py's high-gamma power pickles (per-trial sub-band Hilbert envelopes,
10 ms steps, each sub-band normalized by its word-off mean). Per electrode:

  responsive  per-trial power, word on (0-600 ms) > word off: one-sided Welch t,
              BH-FDR over the session's electrodes, q < .05
  latency     trial-mean envelope minus its word-off mean, smoothed (30 ms); the
              first time it reaches half its peak within 0-800 ms (half-maximum)

Sessions are averaged per electrode, each subject contributes its median latency
per region, and regions are compared by the across-subject mean +/- 95% CI (regions
with >= MIN_SUBJECTS_ROI subjects). Figure: regions ordered by latency, and each
region's group-mean response (scaled to its peak) over time.

    python plot_latency.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

import fc_comparison_functions as fc
from compute import power_dir

PEAK_WIN_MS = (0, 800)
SMOOTH_MS = 30


def electrode_rows(p: dict, lobe_of: dict) -> list[tuple]:
    """(sub, label, roi, latency_ms, trial-mean response curve, fine label) per responsive electrode."""
    lo, hi = np.asarray(p["trial_pow_lo"], float), np.asarray(p["trial_pow_hi"], float)
    t, pv = ttest_ind(hi, lo, axis=0, equal_var=False, alternative="greater", nan_policy="omit")
    ok = np.isfinite(pv)
    resp = np.zeros(len(pv), bool)
    if ok.any():
        resp[ok] = multipletests(pv[ok], method="fdr_bh")[0]
    step = float(p["env_step_ms"])
    curve = np.nanmean(np.asarray(p["env_hi"], float), axis=0)                 # (C, T)
    curve -= np.nanmean(np.asarray(p["env_lo"], float), axis=(0, 2))[:, None]  # minus word-off mean
    k = max(1, int(round(SMOOTH_MS / step)))
    curve = np.apply_along_axis(lambda c: np.convolve(c, np.ones(k) / k, mode="same"), 1, curve)
    times = p["env_hi_t0_ms"] + step * (np.arange(curve.shape[1]) + 0.5)
    win = (times >= PEAK_WIN_MS[0]) & (times < PEAK_WIN_MS[1])
    roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
    out = []
    for c in np.flatnonzero(resp):
        if roi[c] is None or not np.isfinite(curve[c]).all():   # float16 overflow in compute.py v1 envelopes
            continue
        w = curve[c, win]
        pk = int(np.argmax(w))
        if w[pk] <= 0:
            continue
        lat = times[win][int(np.argmax(w[:pk + 1] >= w[pk] / 2))]   # first crossing of half max
        out.append((str(p["sid"][0]), str(p["labels"][c]), roi[c], float(lat), curve[c], str(p["reg_full"][c])))
    return out


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "latency"), **{
        "--fine": dict(action="store_true", help="anatomical labels within the Burke ROIs instead of the 12 ROIs"),
        "--pool-hemispheres": dict(action="store_true", help="with --fine: pool left and right")})
    lobe_of = fc.load_burke_maps()
    rows, times = [], None
    n_elec = 0
    for f in tqdm(fc.session_files(power_dir(save_root, "word_on", "high_gamma"), "_power.pkl", args.n_sessions),
                  desc="sessions"):
        p = fc.load_pickle(str(f))
        if p.get("env_hi") is None:
            continue
        n_elec += len(p["labels"])
        if times is None:
            times = p["env_hi_t0_ms"] + p["env_step_ms"] * (np.arange(np.shape(p["env_hi"])[-1]) + 0.5)
        rows += electrode_rows(p, lobe_of)
    # region: the Burke ROI, or (--fine) the anatomical label, optionally without hemisphere
    region = [(r[5].split(" ", 1)[1] if args.pool_hemispheres else r[5]) if args.fine else r[2] for r in rows]
    lobe = {g: r[2].split("-", 1)[1] for g, r in zip(region, rows)}   # region -> Burke lobe (colour)
    el = pd.DataFrame([(r[0], r[1], g, r[3]) for r, g in zip(rows, region)],
                      columns=["sub", "label", "roi", "latency_ms"])
    curves = np.array([r[4] for r in rows])
    print(f"[latency] {len(el)} responsive electrode-sessions of {n_elec}, {el['sub'].nunique()} subjects")

    # sessions -> electrode (mean), electrodes -> subject x ROI (median)
    el = el.groupby(["sub", "label", "roi"], as_index=False)["latency_ms"].mean()
    tbl = el.groupby(["sub", "roi"], as_index=False).agg(latency_ms=("latency_ms", "median"),
                                                          n_elec=("label", "size"))
    st = []
    for roi, g in tbl.groupby("roi"):
        m, ci = fc.mean_ci(g["latency_ms"].to_numpy())
        st.append({"roi": roi, "n_subjects": len(g), "n_electrodes": int(g["n_elec"].sum()),
                   "mean_ms": m, "ci95_ms": ci, "sem_ms": float(g["latency_ms"].std(ddof=1) / np.sqrt(len(g))),
                   "median_ms": float(g["latency_ms"].median())})
    st = pd.DataFrame(st).sort_values("mean_ms")
    os.makedirs(args.out_dir, exist_ok=True)
    stem = join(args.out_dir, "hg_latency_word_on" + ("_fine" if args.fine else "") + ("_pooled" if args.pool_hemispheres else ""))
    st.to_csv(f"{stem}_stats.csv", index=False)
    tbl.to_csv(f"{stem}_per_subject.csv", index=False)
    print(st.round(1).to_string(index=False))

    shown = st[st["n_subjects"] >= fc.MIN_SUBJECTS_ROI]
    # group-mean response per ROI: electrode curves -> subject mean -> mean over subjects
    cdf = pd.DataFrame(curves)
    cdf[["sub", "roi"]] = [(r[0], g) for r, g in zip(rows, region)]
    prof = cdf.groupby(["roi", "sub"]).mean().groupby("roi").mean().reindex(shown["roi"])
    shown_t = (times >= 0) & (times < 1000)              # the plotted range
    prof = prof.loc[:, shown_t]
    prof = prof.div(prof.max(axis=1), axis=0)

    # bars and heatmap share the rows (and one set of row labels, on the bars)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 0.45 * len(shown) + 2.2), sharey=True,
                                 gridspec_kw={"width_ratios": [1, 1.8], "wspace": 0.05})
    y = np.arange(len(shown))
    xmax = float((shown["mean_ms"] + shown["sem_ms"]).max()) * 1.25
    for i, r in enumerate(shown.itertuples()):
        a1.barh(i, r.mean_ms, xerr=r.sem_ms, height=0.7, capsize=3, edgecolor="0.15", ecolor="0.2",
                color=fc.LOBE_COLORS.get(lobe[r.roi], "0.5"))
        a1.annotate(f"n={r.n_subjects}", (xmax * 0.98, i), ha="right", va="center", fontsize=9, color="0.4")
    a1.set_yticks(y, [fc.pretty_roi(r) for r in shown["roi"]], fontsize=12)
    for lab, r in zip(a1.get_yticklabels(), shown["roi"]):   # region names in their lobe colour
        lab.set_color(fc.LOBE_COLORS.get(lobe[r], "k"))
    a1.set_ylim(len(shown) - 0.5, -0.5)   # first row at the top, for both panels
    a1.set_xlim(0, xmax)
    a1.set_xlabel("Half-max latency (ms)", fontsize=13)
    a1.spines[["top", "right"]].set_visible(False)
    im = a2.imshow(prof.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1,
                   extent=[0, 1000, len(shown) - 0.5, -0.5])
    a2.tick_params(axis="y", left=False, labelleft=False)
    a2.set_xlabel("Time after word onset (ms)", fontsize=13)
    fig.colorbar(im, ax=a2, shrink=0.8).set_label("High-gamma response (fraction of peak)", fontsize=12)
    fc.lobe_key(fig, y=0.9)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")   # keeps the row labels
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {stem}.png / .pdf")


if __name__ == "__main__":
    main()
