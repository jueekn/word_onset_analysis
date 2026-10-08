"""plot_epoch_network.py -- time-resolved synchrony network (Rao et al. 2025 hubs,
Solomon et al. 2017 style), alpha ciPLV, high-gamma AEC-c and theta-gamma PAC rows,
200 ms epochs (PAC between electrodes, both directions averaged).

Per subject and region pair, the mean per-epoch change (epoch minus the mean of
three word-off epochs; compute.py) over the pair's electrode pairs. Hubs: a
region's mean change over its connections with >= MIN_SUBJECTS_PAIR subjects,
one-sample t across subjects, two-stage FDR over regions x epochs; edges: each
hub's top-5 connections (fc.epoch_network_figure). 12 ROIs, and the fine
anatomical regions within them.
-> figures/burke_roi_synchrony/roi_synchrony_epochs_(fine_)word_on.png
Also within-electrode PAC per ROI and epoch (t vs 0, BH-FDR over ROI x epoch)
-> figures/burke_roi_synchrony/roi_pac_within_epochs_word_on.png

    python plot_epoch_network.py [--simulation-tag TAG]
"""
from __future__ import annotations

import os
from os.path import join

import numpy as np
import pandas as pd

import fc_comparison_functions as fc
from compute import cond_dir

BEH, BANDS = "word_on", ("alpha", "high_gamma")
PAC_LABEL = "PAC (θ 3–8 Hz × γ 70–110 Hz)"


def epoch_table(save_root: str, band: str, metric: str, lobe_of: dict, n_sessions=None, fine: bool = False):
    """(per subject x region pair 'a|b' x epoch start mean change, region centroids
    = mean MNI of their electrodes, with lobe). Regions: 12 ROIs or (fine) reg_full labels."""
    rows, xyz_rows = [], []
    for f in fc.session_files(cond_dir(save_root, BEH, "diff", band), "_fc_mats.pkl", n_sessions):
        mat = fc.load_pickle(str(f))
        E = mat.get(f"{metric}_epochs")
        if E is None:
            continue
        if metric == "pac":   # directed: phase electrode x amplitude electrode -> both directions
            E = np.nanmean(np.stack([E, E.transpose(0, 2, 1)]), axis=0)
        xyz, _ = fc.pair_xyz_lead(fc.dfrow_from_sid(mat["sid"]))
        roi = fc.roi_of_reg_full(mat["reg_full"], lobe_of)
        lobe = [r and r.split("-", 1)[1] for r in roi]
        if fine:
            roi = [g if r is not None else None for g, r in zip(mat["reg_full"], roi)]
        a, b = np.triu_indices(len(roi), 1)
        ok = np.array([roi[i] is not None and roi[j] is not None for i, j in zip(a, b)], bool)
        a, b = a[ok], b[ok]
        d = pd.DataFrame(E[:, a, b].T, columns=[w[0] for w in fc.EPOCHS])
        d["roi"] = ["|".join(sorted((roi[i], roi[j]))) for i, j in zip(a, b)]
        d = d.groupby("roi").mean().stack().rename("sync_diff").reset_index()
        rows.append(d.rename(columns={"level_1": "epoch_ms"}).assign(sub=str(mat["sid"][0])))
        xyz_rows += [(r, lb, *c) for r, lb, c in zip(roi, lobe, xyz) if r is not None and np.isfinite(c).all()]
    tbl = pd.concat(rows).groupby(["sub", "roi", "epoch_ms"], as_index=False)["sync_diff"].mean()
    cen = (pd.DataFrame(xyz_rows, columns=["roi", "lobe", "x", "y", "z"])
             .groupby("roi").agg(lobe=("lobe", "first"), x=("x", "mean"), y=("y", "mean"), z=("z", "mean")))
    return tbl, cen


def hubs(etbl: pd.DataFrame) -> pd.DataFrame:
    """Region's mean change over its connections with >= MIN_SUBJECTS_PAIR subjects;
    t across subjects, two-stage FDR over regions x epochs."""
    n_pair = etbl.groupby("roi")["sub"].nunique()
    htbl = etbl[etbl["roi"].isin(n_pair.index[n_pair >= fc.MIN_SUBJECTS_PAIR])]
    ab = htbl["roi"].str.split("|", expand=True)
    long = pd.concat([htbl.assign(roi=ab[0]), htbl[ab[0] != ab[1]].assign(roi=ab[1])])
    return fc.bin_stats(long.groupby(["sub", "roi", "epoch_ms"], as_index=False)["sync_diff"].mean(),
                        "epoch_ms", "sync_diff", fdr="fdr_tsbky")


def network_csvs(save_root, band, metric, path, lobe_of, n_sessions, fine, tag) -> None:
    """Write <path>_stats / _centroids / _hubs.csv for one row of the network figure."""
    etbl, cen = epoch_table(save_root, band, metric, lobe_of, n_sessions, fine)
    fc.bin_stats(etbl, "epoch_ms", "sync_diff", min_subjects=fc.MIN_SUBJECTS_PAIR).to_csv(
        f"{path}_stats.csv", index=False)
    cen.to_csv(f"{path}_centroids.csv")
    hub = hubs(etbl)
    hub.to_csv(f"{path}_hubs.csv", index=False)
    print(f"[epochs {tag}] {int(hub['t'].notna().sum() / len(fc.EPOCHS))} regions tested, "
          f"{int((hub['q'] < 0.05).sum())} hub x epoch cells at q < .05")


def pac_within_figure(save_root, lobe_of, out_dir, n_sessions, min_electrodes) -> None:
    """Within-electrode PAC change per ROI and epoch: subject means, t vs 0."""
    rows = []
    for f in fc.session_files(cond_dir(save_root, BEH, "diff", "pac"), "_fc_mats.pkl", n_sessions):
        p = fc.load_pickle(str(f))
        if p.get("pac_local_epochs") is None:
            continue
        roi = fc.roi_of_reg_full(p["reg_full"], lobe_of)
        rows += [(str(p["sid"][0]), e, r, w[0], float(v)) for e, r in enumerate(roi) if r is not None
                 for w, v in zip(fc.EPOCHS, p["pac_local_epochs"][:, e]) if np.isfinite(v)]
    el = pd.DataFrame(rows, columns=["sub", "e", "roi", "epoch_ms", "d"])
    tbl = el.groupby(["sub", "roi", "epoch_ms"]).agg(d=("d", "mean"), n_elec=("e", "nunique")).reset_index()
    stats = fc.bin_stats(tbl[tbl["n_elec"] >= min_electrodes], "epoch_ms", "d")
    stem = join(out_dir, f"roi_pac_within_epochs_{BEH}")
    stats.to_csv(f"{stem}_stats.csv", index=False)
    print(f"[saved] {fc.roi_curve_figure(stats, 'epoch_ms', stem, 'Epoch start (ms after word onset)', 'PAC change, within electrode')}")


def main() -> None:
    args, save_root = fc.plot_args(__doc__, join("figures", "burke_roi_synchrony"))
    lobe_of = fc.load_burke_maps()
    c = fc.contrast(BEH)
    os.makedirs(args.out_dir, exist_ok=True)
    for level, fine in (("", False), ("fine_", True)):
        stem = f"roi_synchrony_epochs_{level}{BEH}_{{band}}_{{metric}}"
        for band in BANDS:
            metric = fc.RUN_METRIC[band]
            network_csvs(save_root, band, metric, join(args.out_dir, stem.format(band=band, metric=metric)),
                         lobe_of, args.n_sessions, fine, f"{level or 'roi_'}{band}")
        pac_path = join(args.out_dir, f"roi_pac_epochs_{level}{BEH}")
        network_csvs(save_root, "pac", "pac", pac_path, lobe_of, args.n_sessions, fine, f"{level or 'roi_'}pac")
        fc.epoch_network_figure(args.out_dir, stem, f"{{metric}} ({c['hi_label']} vs. {c['lo_label']})",
                                extra_rows=[(PAC_LABEL, "pac", f"{pac_path}_hubs.csv")])
    pac_within_figure(save_root, lobe_of, args.out_dir, args.n_sessions, args.min_electrodes)


if __name__ == "__main__":
    main()
