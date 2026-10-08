"""plot_boxplots.py -- word on vs word off per Burke ROI, alpha and high gamma rows.

  power      per-electrode Cohen's d of band power (compute.py power pickles)
             -> figures/burke_roi_power/roi_power_word_on(_ci).png
  synchrony  per-electrode mean on - off change over all partners (fc.electrode_sync,
             as Rao et al. 2025): alpha ciPLV, high-gamma AEC-c
             -> figures/burke_roi_synchrony/roi_synchrony_word_on(_ci).png
  pac        theta (3-8 Hz) phase x gamma (70-110 Hz) amplitude coupling, on - off:
             within each electrode, and each electrode's mean over all partners
             (both directions) -> figures/burke_roi_pac/roi_pac_word_on(_ci).png

Sessions are averaged per electrode, electrodes per subject and ROI; one-sample t
vs 0 across subjects, BH-FDR over the 12 ROIs (fc.roi_stats). Two versions of each
figure: subject distributions, and mean +/- 95% CI (`_ci`).

    python plot_boxplots.py [--measure power synchrony pac] [--simulation-tag TAG]
"""
from __future__ import annotations

from os.path import join

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import fc_comparison_functions as fc
import helper
from compute import cond_dir, power_dir

BEH, BANDS = "word_on", ("alpha", "high_gamma")
MEASURES = ("cohens_d", "change_dB")


def power_table(save_root: str, band: str, lobe_of: dict, n_sessions=None) -> pd.DataFrame:
    """Per (subject, electrode): word-off / word-on power and Cohen's d, sessions averaged."""
    rows = []
    for f in tqdm(fc.session_files(power_dir(save_root, BEH, band), "_power.pkl", n_sessions), desc=f"power {band}"):
        p = fc.load_pickle(str(f))
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        rows += [(str(p["sid"][0]), str(lab), r, float(lo), float(hi), float(d))
                 for lab, r, lo, hi, d in zip(p["labels"], roi, p["pow_lo"], p["pow_hi"], p["cohens_d"])
                 if r is not None and np.isfinite(lo) and np.isfinite(hi)]
    df = pd.DataFrame(rows, columns=["sub", "label", "roi", "lo", "hi", "cohens_d"])
    df = df.groupby(["sub", "label", "roi"], as_index=False)[["lo", "hi", "cohens_d"]].mean()
    return df.assign(change_dB=10 * np.log10(df["hi"] / df["lo"]))


def sync_table(save_root: str, band: str, metric: str, lobe_of: dict, n_sessions=None) -> pd.DataFrame:
    """Per (subject, electrode): on - off synchrony to all partners, sessions averaged."""
    rows = []
    for f in tqdm(fc.session_files(cond_dir(save_root, BEH, "diff", band), "_fc_mats.pkl", n_sessions),
                  desc=f"synchrony {band}"):
        mat = fc.load_pickle(str(f))
        pairs = helper.get_pairs(fc.dfrow_from_sid(mat["sid"]))
        if pairs is None or mat.get(metric) is None:
            continue
        roi = fc.roi_of_reg_full(mat["reg_full"], lobe_of)
        rows += [(str(mat["sid"][0]), lab, r, float(v))
                 for lab, r, v in zip(pairs["label"].astype(str), roi, fc.electrode_sync(np.asarray(mat[metric], float)))
                 if r is not None and np.isfinite(v)]
    df = pd.DataFrame(rows, columns=["sub", "label", "roi", "sync_diff"])
    return df.groupby(["sub", "label", "roi"], as_index=False)[["sync_diff"]].mean()


def pac_table(save_root: str, lobe_of: dict, n_sessions=None) -> pd.DataFrame:
    """Per (subject, electrode): on - off PAC within the electrode (`within`) and its mean
    over all partners, as phase or amplitude electrode (`between`); sessions averaged."""
    rows = []
    for f in tqdm(fc.session_files(cond_dir(save_root, BEH, "diff", "pac"), "_fc_mats.pkl", n_sessions), desc="pac"):
        p = fc.load_pickle(str(f))
        pairs = helper.get_pairs(fc.dfrow_from_sid(p["sid"]))
        P = np.asarray(p["pac"], float)
        between = fc.electrode_sync(np.nanmean(np.stack([P, P.T]), axis=0))
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        rows += [(str(p["sid"][0]), lab, r, float(w), float(b))
                 for lab, r, w, b in zip(pairs["label"].astype(str), roi, p["pac_local"], between)
                 if r is not None and np.isfinite(w)]
    df = pd.DataFrame(rows, columns=["sub", "label", "roi", "within", "between"])
    return df.groupby(["sub", "label", "roi"], as_index=False)[["within", "between"]].mean()


def pac_figure(out: str, ci: bool) -> None:
    """Rows: within-electrode and between-electrode PAC change per ROI."""
    panels = []
    for m, lab in (("within", "within electrode"), ("between", "between electrodes")):
        tbl = pd.read_csv(join(out, f"roi_pac_{BEH}_{m}_per_subject.csv"))
        st = fc.roi_stats(tbl, m)
        panels.append((f"PAC, {lab}\n(Word On vs. Word Off)",
                       (lambda ax, st=st: fc.roi_ci_panel(ax, st)) if ci else
                       (lambda ax, tbl=tbl, st=st, m=m: fc.roi_panel(ax, tbl, m, np.random.default_rng(0), stats=st))))
    fc.roi_figure(panels, out, f"roi_pac_{BEH}" + ("_ci" if ci else ""))


def main() -> None:
    args, save_root = fc.plot_args(__doc__, "figures", **{
        "--measure": dict(nargs="+", default=["power", "synchrony", "pac"], choices=["power", "synchrony", "pac"])})
    lobe_of = fc.load_burke_maps()
    c = fc.contrast(BEH)
    vs = f"({c['hi_label']} vs. {c['lo_label']})"
    if "power" in args.measure:
        out, stem = join(args.out_dir, "burke_roi_power"), f"roi_power_{BEH}_{{band}}"
        for band in BANDS:
            el = power_table(save_root, band, lobe_of, args.n_sessions)
            tbl = fc.subject_roi_means(el, list(MEASURES), min_electrodes=args.min_electrodes)
            stats = {m: fc.roi_stats(tbl, m) for m in MEASURES}
            fc.print_roi_stats(stats["cohens_d"], f"{band} power, Cohen's d {vs}:")
            fc.write_roi_csvs(out, stem.format(band=band), tbl, el, stats)
        for ci in (False, True):
            fc.band_contrast_figure(out, stem, "cohens_d", f"Power, Cohen's d {vs}", ci=ci)
    if "synchrony" in args.measure:
        out, stem = join(args.out_dir, "burke_roi_synchrony"), f"roi_synchrony_{BEH}_{{band}}_{{metric}}"
        for band in BANDS:
            metric = fc.RUN_METRIC[band]
            el = sync_table(save_root, band, metric, lobe_of, args.n_sessions)
            tbl = fc.subject_roi_means(el, ["sync_diff"], min_electrodes=args.min_electrodes)
            stats = {"sync_diff": fc.roi_stats(tbl, "sync_diff")}
            fc.print_roi_stats(stats["sync_diff"], f"{band} {metric}, mean over partners {vs}:")
            fc.write_roi_csvs(out, stem.format(band=band, metric=metric), tbl, el, stats)
        for ci in (False, True):
            fc.band_contrast_figure(out, stem, "sync_diff", f"{{metric}} {vs}", ci=ci)
    if "pac" in args.measure:
        out = join(args.out_dir, "burke_roi_pac")
        el = pac_table(save_root, lobe_of, args.n_sessions)
        for m in ("within", "between"):
            tbl = fc.subject_roi_means(el, [m], min_electrodes=args.min_electrodes)
            stats = {m: fc.roi_stats(tbl, m)}
            fc.print_roi_stats(stats[m], f"PAC {m} {vs}:")
            fc.write_roi_csvs(out, f"roi_pac_{BEH}_{m}", tbl, el, stats)
        for ci in (False, True):
            pac_figure(out, ci)


if __name__ == "__main__":
    main()
