"""
Subregion composition of a Burke ROI: which regions get pooled, and how many
subjects / bipolar pairs each one contributes.

The ROIwise scripts (burke_roi_connectivity / build_roi_synchrony /
build_roi_power) pool every electrode whose region maps to a Burke lobe
("occipital", "hippocampus", ...) via region_to_burke_lobe.csv. This script
opens that pooling up: for one ROI it lists the constituent regions and plots

    - number of SUBJECTS with >=1 bipolar pair in that region
    - number of bipolar PAIRS (electrodes) in that region, summed over subjects

Labels come from the same canonical type-aware cascade the ROIwise scripts use
(helper.regionalize_electrodes_by_type: volumetric atlases for D/uD depths,
surface atlases for G/S grids-strips), so the counts describe exactly what gets
pooled downstream -- no separate labeling path.

Sessions are unioned within a subject: a region counts for a subject if any of
that subject's sessions has a pair there. Pair counts use the session with the
most pairs in that region (montages differ across sessions; summing would
double-count the same physical contacts).

Usage
-----
    python roi_subregion_counts.py --roi occipital
    python roi_subregion_counts.py --roi hippocampus --split-hemi
    python roi_subregion_counts.py --roi all              # one figure per ROI
    python roi_subregion_counts.py --roi occipital --exp FR1
    python roi_subregion_counts.py --roi occipital --refresh   # rebuild cache

The per-session tabulation is cached (scratch/roi_subregion_counts_cache.csv)
because labeling all ~1.4k sessions takes a few minutes; subsequent runs with
different --roi / --exp / --split-hemi reuse it. Use --refresh to rebuild.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import helper
from project_paths import SCRATCH_DIR
from figure_io import SaveFigure

helper.root_dir = str(SCRATCH_DIR)

HERE = Path(__file__).resolve().parent
PAIRS_DIR = Path(SCRATCH_DIR) / "electrode_information" / "pairs"
CACHE_PATH = Path(SCRATCH_DIR) / "roi_subregion_counts_cache.csv"
FIGDIR = HERE / "figures" / "roi_subregion_counts"

# Same six ROIs the ROIwise scripts pool into (burke_roi_connectivity.LOBES).
LOBES = ["frontal", "temporal", "parietal", "occipital", "limbic", "hippocampus"]

# <sub>_<exp>_<sess>_pairs.json
_FNAME_RE = re.compile(r"^(?P<sub>[^_]+)_(?P<exp>.+)_(?P<sess>\d+)_pairs\.json$")


# ------------------------------- tabulation ---------------------------------
def _session_files():
    """(sub, exp, sess, path) for every pairs.json on disk, sorted."""
    out = []
    for p in sorted(PAIRS_DIR.glob("*_pairs.json")):
        m = _FNAME_RE.match(p.name)
        if m:
            out.append((m["sub"], m["exp"], int(m["sess"]), p))
    return out


def _label_session(sub: str, exp: str, sess: int, path: Path) -> pd.DataFrame:
    """One row per (region, hemisphere) present in this session, with n_pairs."""
    pairs = pd.read_json(path).fillna("nan")
    if not len(pairs):
        return pd.DataFrame()

    labels = pd.Series(helper.regionalize_electrodes_by_type(pairs))
    labels = labels.dropna()
    if not len(labels):
        return pd.DataFrame()

    df = pd.DataFrame({
        "hemi": labels.map(lambda v: v.split(" ", 1)[0]),
        "region": labels.map(lambda v: v.split(" ", 1)[1] if " " in v else "nan"),
    })
    df = df[(df["region"] != "nan") & df["hemi"].isin(["L", "R"])]
    if not len(df):
        return pd.DataFrame()

    out = df.groupby(["region", "hemi"], as_index=False).size()
    out = out.rename(columns={"size": "n_pairs"})
    out.insert(0, "sub", sub)
    out.insert(1, "exp", exp)
    out.insert(2, "sess", sess)
    return out


def build_table(refresh: bool = False) -> pd.DataFrame:
    """Long table: sub, exp, sess, region, hemi, n_pairs (cached on disk)."""
    if CACHE_PATH.exists() and not refresh:
        return pd.read_csv(CACHE_PATH)

    files = _session_files()
    print(f"labeling {len(files)} sessions from {PAIRS_DIR} ...", flush=True)
    rows, failed = [], []
    for i, (sub, exp, sess, path) in enumerate(files, 1):
        try:
            r = _label_session(sub, exp, sess, path)
            if len(r):
                rows.append(r)
        except Exception as e:  # a handful of montages have malformed pairs.json
            failed.append((sub, exp, sess, repr(e)))
        if i % 100 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    if failed:
        print(f"WARNING: {len(failed)} sessions failed to label, e.g. {failed[:3]}",
              file=sys.stderr)
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["sub", "exp", "sess", "region", "hemi", "n_pairs"])
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CACHE_PATH, index=False)
    print(f"cached -> {CACHE_PATH}  ({len(table)} rows, "
          f"{table['sub'].nunique()} subjects)")
    return table


def load_lobe_map() -> dict[str, str]:
    lobe = pd.read_csv(HERE / "region_to_burke_lobe.csv")
    return lobe.set_index("region")["burke_lobe"].to_dict()


# ------------------------------ summarizing ---------------------------------
def summarize(table: pd.DataFrame, lobe_of: dict[str, str], roi: str,
              exp: str | None = None, split_hemi: bool = False) -> pd.DataFrame:
    """Per-subregion subject / pair counts for one ROI.

    Subjects: unique subs with >=1 pair in that subregion (union over sessions).
    Pairs   : per subject, the max over that subject's sessions (montages
              overlap across sessions, so summing would double-count), then
              summed across subjects.
    """
    df = table.copy()
    if exp:
        df = df[df["exp"] == exp]
    df["lobe"] = df["region"].map(lambda r: lobe_of.get(r, ""))
    df = df[df["lobe"] == roi]
    if not len(df):
        return pd.DataFrame(columns=["subregion", "n_subjects", "n_pairs"])

    df["subregion"] = (df["hemi"] + "-" + df["region"]) if split_hemi else df["region"]

    # collapse sessions within a subject
    per_sub = (df.groupby(["subregion", "sub"], as_index=False)["n_pairs"].max())
    out = (per_sub.groupby("subregion")
                  .agg(n_subjects=("sub", "nunique"), n_pairs=("n_pairs", "sum"))
                  .reset_index()
                  .sort_values(["n_subjects", "n_pairs"], ascending=False)
                  .reset_index(drop=True))
    return out


# -------------------------------- plotting ----------------------------------
def plot_counts(summary: pd.DataFrame, roi: str, total_subs: int,
                exp: str | None, split_hemi: bool) -> plt.Figure:
    n = max(len(summary), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * n + 2.2), sharey=True)
    y = np.arange(len(summary))[::-1]  # largest at top

    axes[0].barh(y, summary["n_subjects"], color="#4C72B0", height=0.7)
    axes[0].set_xlabel("Subjects with ≥1 pairs")
    for yi, v in zip(y, summary["n_subjects"]):
        axes[0].text(v, yi, f" {v}", va="center", fontsize=8)

    axes[1].barh(y, summary["n_pairs"], color="#DD8452", height=0.7)
    axes[1].set_xlabel("Total pairs count")
    for yi, v in zip(y, summary["n_pairs"]):
        axes[1].text(v, yi, f" {v}", va="center", fontsize=8)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(summary["subregion"])
    for ax in axes:
        ax.margins(x=0.12)

    sub_tot = summary["n_subjects"].max() if len(summary) else 0
    title = (f'"{roi}" ROI — {len(summary)} pooled subregions')
    if exp:
        title += f"  [{exp}]"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------- main ------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roi", default="occipital",
                    help=f"one of {LOBES} (or 'subcortical'), or 'all'")
    ap.add_argument("--exp", default=None, help="restrict to one experiment, e.g. FR1")
    ap.add_argument("--split-hemi", action="store_true",
                    help="count L and R subregions separately")
    ap.add_argument("--refresh", action="store_true", help="rebuild the cache")
    ap.add_argument("--figdir", default=str(FIGDIR))
    args = ap.parse_args()

    table = build_table(refresh=args.refresh)
    lobe_of = load_lobe_map()
    total_subs = table["sub"].nunique() if len(table) else 0

    rois = LOBES if args.roi == "all" else [args.roi]
    for roi in rois:
        summary = summarize(table, lobe_of, roi, exp=args.exp,
                            split_hemi=args.split_hemi)
        if not len(summary):
            print(f"\n[{roi}] no labeled electrodes found.")
            continue

        print(f"\n=== {roi} ===")
        print(summary.to_string(index=False))

        tag = roi + ("_byhemi" if args.split_hemi else "") + \
            (f"_{args.exp}" if args.exp else "")
        Path(args.figdir).mkdir(parents=True, exist_ok=True)
        summary.to_csv(Path(args.figdir) / f"{tag}_subregion_counts.csv", index=False)
        fig = plot_counts(summary, roi, total_subs, args.exp, args.split_hemi)
        png, _ = SaveFigure(fig, f"{tag}_subregion_counts", args.figdir)
        plt.close(fig)
        print(f"saved -> {png}")


if __name__ == "__main__":
    main()
