"""counts_helper.py — electrode-pair / subject coverage tabulations and viz.

Used by data_visualization.ipynb (and unit tests). Not part of the live FC
compute / plot pipeline. Dominated by pandas + seaborn + matplotlib method
chains; file-level narrowing for library-stub noise. Real signal still
caught (the strict rules other than the four narrowed ones stay on).
"""
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportAttributeAccessIssue=warning
from __future__ import annotations

from typing import Any, Sequence, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import seaborn as sns  # pyright: ignore[reportMissingTypeStubs]

# `from X import *` preserves the legacy module namespace for notebook callers.
from cstat import *  # noqa: F401,F403
from misc import *  # noqa: F401,F403
from misc import get_username_from_working_directory  # explicit for type checker
from matrix_operations import *  # noqa: F401,F403

from project_paths import SCRATCH_DIR as _SCRATCH_DIR
USERNAME: str = get_username_from_working_directory(index=2)  # pyright: ignore[reportConstantRedefinition]
root_dir: str = str(_SCRATCH_DIR)

import helper
from helper import *  # noqa: F401,F403  # pyright: ignore[reportAssignmentType]
helper.root_dir = root_dir
from fc_comparison_functions import *  # noqa: F401,F403
from figure_io import SaveFigure as _SaveFigure  # noqa: F401 — re-exported for notebook callers

from pathlib import Path
import pickle

NDArrayAny = npt.NDArray[Any]


def load_pickle(path: str) -> Any:
    """Convenience pickle.load wrapper. Returns Any (caller-known).

    Same signature (including parameter name) as misc.load_pickle, which
    this shadows via star-import.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def load_results(root_dir: str, beh: str, cond: str | None = None, band: str = "theta") -> list[Any]:
    """Load every *_fc_mats.pkl in <root_dir>/<beh>/fc_mats/<band>/ in sorted order.

    Matches helper.load_results's positional signature (which this shadows
    via star-import); the `cond` parameter is accepted but unused — this
    implementation walks the legacy <band>/ layout without a cond subdir.
    """
    d = Path(root_dir) / beh / "fc_mats" / band
    files = sorted(d.glob("*_fc_mats.pkl"))
    return [load_pickle(str(p)) for p in files]

metrics = ["coh", "plv", "ppc", "ciplv", "pli", "wpli", "dpli", "aec"]
asymm = {"dpli"}               
#multivar = {"mim", "gc", "gc_tr"} 

from figure_io import METRIC_LABELS as correct_labels  # noqa: E402
# CFG_FLOW_VERIFY: counts_helper.correct_labels sourced from figure_io
# (counts pre-migration omitted 'aec_c','pac','gc','gc_tr','narrow_*' from its
# correct_labels — those keys are now exposed via the centralized dict; no
# runtime consumer in counts breaks since lookup is by-key.)
assert correct_labels["coh"] == "Coh" and correct_labels["aec"] == "AEC", (  # CFG_FLOW_VERIFY
    f"correct_labels drift: {correct_labels}"
)

from project_paths import BANDS as bands

# CFG_FLOW_VERIFY: counts_helper.bands must equal project_paths.BANDS
# Per-key pins: pre-migration counts_helper.bands had {"theta": (3,9),
# "alpha": (8,13), "gamma": (80,160)}. theta was reconciled (3,9) -> (4,9)
# at unification (canonical is fc_comparison_functions value (4,9); counts
# is exploratory, no production consumer of counts.bands["theta"]).
from project_paths import BANDS as _CFG_BANDS  # noqa: E402
assert bands is _CFG_BANDS, "counts_helper.bands drift from project_paths.BANDS"
assert bands["theta"] == (4.0, 9.0), f"theta drift: {bands['theta']}"     # CFG_FLOW_VERIFY
assert bands["alpha"] == (8.0, 13.0), f"alpha drift: {bands['alpha']}"    # CFG_FLOW_VERIFY
assert bands["gamma"] == (70.0, 110.0), f"gamma drift: {bands['gamma']}"  # CFG_FLOW_VERIFY

bad_regions = ("Clear Label", "head", "tail", "misc")

atlas_cols = [
    "stein.region", "das.region", "atlases.mtl", "atlases.wholebrain", "wb.region",
    "mni.region", "atlases.dk", "dk.region",
    "ind.corrected.region", "mat.ind.corrected.region",
    "ind.snap.region", "mat.ind.snap.region",
    "ind.dural.region", "mat.ind.dural.region",
    "ind.region", "mat.ind.region",
    "avg.corrected.region", "avg.mat.corrected.region",
    "avg.snap.region", "avg.mat.snap.region",
    "avg.dural.region", "avg.mat.dural.region",
    "avg.region", "avg.mat.region",
    "mat.tal.region",
]

regionlabels = list(helper.get_region_information("region_labels"))  
reg2i = {r: i for i, r in enumerate(regionlabels)}


def electrode_pair_coverage_from_results(
    results_list: Sequence[dict[str, Any] | None],
    regionlabels: Sequence[str],
    reg2i: dict[str, int],
    min_elec_per_region: int = 1,
    aggregate_sessions: str = "max",
) -> tuple[pd.DataFrame, list[str]]:
    """Build a (region × region) pair-coverage matrix across subjects.

    For each session result with a 'reg_full' electrode→region label array,
    bincount electrodes per region, aggregate across a subject's sessions
    via max (default) or sum, then take the outer product of per-subject
    counts and sum across subjects. Off-diagonal entries count i·j electrode
    pairs; diagonal counts n(n-1) within-region pairs.
    """
    R = len(regionlabels)

    subj_counts = {}

    for r in results_list:
        if r is None:
            continue
        sub = r["sid"][0]
        reg_full = np.asarray(r["reg_full"], dtype=object)

        idx = np.array([reg2i.get(lbl, -1) for lbl in reg_full], dtype=int)
        idx = idx[idx >= 0]
        if idx.size == 0:
            continue

        counts = np.bincount(idx, minlength=R).astype(np.int64) 

        counts[counts < min_elec_per_region] = 0

        subj_counts.setdefault(sub, []).append(counts)

    subs = sorted(subj_counts.keys())
    if not subs:
        raise ValueError("No subjects with valid electrode->region assignments.")

    Ns = []
    for sub in subs:
        stack = np.stack(subj_counts[sub], axis=0)  
        if aggregate_sessions == "max":
            n = stack.max(axis=0)
        elif aggregate_sessions == "sum":
            n = stack.sum(axis=0)
        else:
            raise ValueError("aggregate_sessions must be 'max' or 'sum'")
        Ns.append(n.astype(np.int64))

    N = np.stack(Ns, axis=0)  

    E = np.einsum("si,sj->ij", N, N).astype(np.int64)  

    diag = np.einsum("si->i", N * (N - 1)).astype(np.int64)
    np.fill_diagonal(E, diag)

    E_df = pd.DataFrame(E, index=regionlabels, columns=regionlabels)  # pyright: ignore[reportCallIssue, reportArgumentType]
    return E_df, subs

def plot_electrodecounts_matrix(
    E_df: pd.DataFrame,
    sorted_regionlabels: Sequence[str],
    centers: Sequence[float],
    group_names: Sequence[str],
    starts: Sequence[float],
    out_png: str | None = None,
    out_dir: str = ".",
    title: str | None = None,
    cmap: str = "viridis",
    sym_log: bool = True,
    linthresh: float = 100,
    vmin: float | None = None,
    vmax: float | None = None,
    annot: bool = True,
    fmt: str = ".0f",
    annot_kws: dict[str, Any] | None = None,
    cbar_label: str = "Total electrode counts per region-pair",
    separated_groups: bool = True,
    show_region_names: bool = False,
    show_hemispheres: bool = True,
    hemi_split: int | None = None,
    max_region_labels: int = 40,
    region_label_fontsize: float = 6,
    group_label_fontsize: float = 10,
    figsize: tuple[float, float] = (20, 20),
    show: bool = True,
    save: bool = False,
) -> Any:
    """Heatmap of electrode-pair counts per region pair (square, region-major)."""

    E_df = E_df.loc[sorted_regionlabels, sorted_regionlabels]

    data = np.asarray(E_df, float)
    if vmax is None:
        vmax = np.nanmax(data) if np.isfinite(data).any() else 1.0
    if vmin is None:
        vmin = 0.0

    norm = None
    if sym_log:
        norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)

    if annot_kws is None:
        annot_kws = {"size": 4.5}

    fig, ax = plt.subplots(figsize=figsize)

    _hm = sns.heatmap(
        E_df,
        ax=ax,
        cmap=cmap,
        norm=norm,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        square=True,
        annot=annot,
        fmt=fmt,
        annot_kws=annot_kws,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": cbar_label},
    )

    ax.set_xticks(centers)
    ax.set_xticklabels(group_names, rotation=45, fontsize=group_label_fontsize, ha="right")
    ax.set_yticks(centers)
    ax.set_yticklabels(group_names, rotation=45, fontsize=group_label_fontsize)

    if show_region_names:
        nreg = len(sorted_regionlabels)
        tick_pos = np.arange(nreg) + 0.5

        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)

        ax.set_xticklabels(
            list(sorted_regionlabels),
            rotation=90,
            ha="center",
            va="top",
            fontsize=region_label_fontsize,
        )
        ax.set_yticklabels(
            list(sorted_regionlabels),
            rotation=0,
            ha="right",
            va="center",
            fontsize=region_label_fontsize,
        )

        ax.tick_params(axis="both", which="major", length=0, pad=1)

    if separated_groups:
        for b in starts[1:]:
            ax.axvline(b, color="white", lw=2)
            ax.axhline(b, color="white", lw=2)

    if show_hemispheres:
        nreg = len(sorted_regionlabels)
        split = hemi_split
        if split is None:
            split = next((k for k, r in enumerate(sorted_regionlabels) if str(r).startswith("R")), nreg // 2)

        trans = ax.get_xaxis_transform()
        y = 1.03
        ax.plot([0, split], [y, y], transform=trans, clip_on=False, color="k", lw=2)
        ax.plot([split, nreg], [y, y], transform=trans, clip_on=False, color="k", lw=2)
        ax.text(split / 2, y + 0.01, "L", transform=trans, ha="center", va="bottom", fontsize=12)
        ax.text((split + nreg) / 2, y + 0.01, "R", transform=trans, ha="center", va="bottom", fontsize=12)

    ax.set_xlabel("Region Group")
    ax.set_ylabel("Region Group")
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()

    if save and out_png is not None:
        _SaveFigure(fig, out_png, out_dir, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def build_subject_region_counts(
    sess_list_df: pd.DataFrame,
    region_translator: Any,
    regionlabels: Sequence[str],
) -> pd.DataFrame:
    """Wide (subject × region) electrode counts via helper.get_pairs per session."""
    import helper

    rows = []
    for _, dfrow in sess_list_df.iterrows():
        pairs = helper.get_pairs(dfrow)
        if pairs is None or len(pairs) == 0:
            continue

        localization = helper.get_localization(dfrow)
        reg_full = helper.regionalize_electrodes_by_type(pairs, localization)
        s = pd.Series(reg_full, dtype="object").dropna()

        s = s[s.isin(regionlabels)]

        counts = s.value_counts()
        sub = dfrow["sub"]

        for reg, n in counts.items():
            rows.append({"sub": sub, "region": reg, "n_elec": int(n)})

    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError("No electrode counts found")

    long = long.groupby(["sub", "region"], as_index=False)["n_elec"].sum()

    wide = long.pivot_table(index="sub", columns="region", values="n_elec",
                            aggfunc="sum", fill_value=0)

    wide = wide.reindex(columns=regionlabels, fill_value=0)

    return wide

def subject_counts_region_pairs(
    results_list: Sequence[dict[str, Any] | None],
    regionlabels: Sequence[str],
    reg2i: dict[str, int],
    metric: str = "coh",
    min_elec_per_region: int = 1,
) -> tuple[pd.DataFrame, list[str]]:
    """Count distinct subjects with at least one finite electrode-pair value per region pair."""
    R = len(regionlabels)
    subj_has_pair = {}  

    for r in results_list:
        if r is None:
            continue

        sub = r["sid"][0]
        reg_full = np.asarray(r["reg_full"], dtype=object)

        m_arr = np.asarray(r[metric])
        if m_arr.ndim != 2 or m_arr.shape[0] != m_arr.shape[1]:
            continue
        if len(reg_full) != m_arr.shape[0]:
            continue

        idx = np.array([reg2i.get(lbl, -1) for lbl in reg_full], dtype=int)
        keep_elec = idx >= 0
        if keep_elec.sum() == 0:
            continue

        idx = idx[keep_elec]
        m_arr = m_arr[np.ix_(keep_elec, keep_elec)]

        elec_by_region = [np.where(idx == i)[0] for i in range(R)]
        n_by_region = np.array([e.size for e in elec_by_region], dtype=int)
        present = n_by_region >= min_elec_per_region

        finite = np.isfinite(m_arr)

        sess_pair = np.zeros((R, R), dtype=bool)
        regions_present = np.flatnonzero(present)

        for ii, i in enumerate(regions_present):
            ai = elec_by_region[i]
            if ai.size == 0:
                continue
            for j in regions_present[ii:]:
                bj = elec_by_region[j]
                if bj.size == 0:
                    continue
                if finite[np.ix_(ai, bj)].any():
                    sess_pair[i, j] = True
                    sess_pair[j, i] = True

        if sub not in subj_has_pair:
            subj_has_pair[sub] = sess_pair
        else:
            subj_has_pair[sub] |= sess_pair

    subs = sorted(subj_has_pair.keys())
    if len(subs) == 0:
        raise ValueError("No subjects produced valid region-pair matrices")

    c_counts = np.zeros((R, R), dtype=int)
    for sub in subs:
        c_counts += subj_has_pair[sub].astype(int)

    C_df = pd.DataFrame(c_counts, index=regionlabels, columns=regionlabels)  # pyright: ignore[reportCallIssue, reportArgumentType]
    return C_df, subs

def subjects_per_region(
    sess_list_df: pd.DataFrame,
    region_translator: Any,
    regionlabels: Sequence[str],
) -> pd.DataFrame:
    """Single-column DataFrame ('n_subjects') indexed by region, with the
    count of distinct subjects that have ≥1 electrode in each region."""
    import helper

    rows = []
    for _, dfrow in sess_list_df.iterrows():
        pairs = helper.get_pairs(dfrow)
        if pairs is None or len(pairs) == 0:
            continue

        localization = helper.get_localization(dfrow)
        reg_full = helper.regionalize_electrodes_by_type(pairs, localization)
        s = pd.Series(reg_full, dtype="object").dropna()
        s = s[s.isin(regionlabels)]

        sub = dfrow["sub"]

        for reg in s.unique():
            rows.append({"sub": sub, "region": reg})

    long = cast(pd.DataFrame, pd.DataFrame(rows).drop_duplicates())
    subj_counts = cast(
        pd.Series,
        long.groupby("region")["sub"].nunique().reindex(regionlabels, fill_value=0))
    out = subj_counts.rename("n_subjects").to_frame()  # pyright: ignore[reportCallIssue, reportArgumentType]
    return out

def plot_subjectcounts_matrix(
    C_df: pd.DataFrame,
    sorted_regionlabels: Sequence[str],
    centers: Sequence[float],
    group_names: Sequence[str],
    starts: Sequence[float],
    out_png: str | None = None,
    out_dir: str = ".",
    title: str | None = None,
    cmap: str = "viridis",
    sym_log: bool = True,
    linthresh: float = 10,
    vmin: float | None = None,
    vmax: float | None = None,
    annot: bool = True,
    fmt: str = ".0f",
    annot_kws: dict[str, Any] | None = None,
    cbar_label: str = "Total subject counts per region-pair",
    separated_groups: bool = True,
    show_region_names: bool = False,
    show_hemispheres: bool = True,
    hemi_split: int | None = None,
    max_region_labels: int = 40,
    region_label_fontsize: float = 6,
    group_label_fontsize: float = 10,
    figsize: tuple[float, float] = (20, 20),
    show: bool = True,
    save: bool = False,
) -> Any:
    """Heatmap of distinct-subject counts per region pair (square, region-major)."""

    C_df = C_df.loc[sorted_regionlabels, sorted_regionlabels]

    _data = np.asarray(C_df, float)
    vmax = 383
    vmin = 0

    norm = None
    if sym_log:
        norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)

    if annot_kws is None:
        annot_kws = {"size": 4.5}

    fig, ax = plt.subplots(figsize=figsize)

    _hm = sns.heatmap(
        C_df,
        ax=ax,
        cmap=cmap,
        norm=norm,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        square=True,
        annot=annot,
        fmt=fmt,
        annot_kws=annot_kws,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": cbar_label},
    )

    ax.set_xticks(centers)
    ax.set_xticklabels(group_names, rotation=45, fontsize=group_label_fontsize, ha="right")
    ax.set_yticks(centers)
    ax.set_yticklabels(group_names, rotation=45, fontsize=group_label_fontsize)

    if show_region_names:
        nreg = len(sorted_regionlabels)
        tick_pos = np.arange(nreg) + 0.5

        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)

        ax.set_xticklabels(
            list(sorted_regionlabels),
            rotation=90,
            ha="center",
            va="top",
            fontsize=region_label_fontsize,
        )
        ax.set_yticklabels(
            list(sorted_regionlabels),
            rotation=0,
            ha="right",
            va="center",
            fontsize=region_label_fontsize,
        )

        ax.tick_params(axis="both", which="major", length=0, pad=1)

    if separated_groups:
        for b in starts[1:]:
            ax.axvline(b, color="white", lw=2)
            ax.axhline(b, color="white", lw=2)

    if show_hemispheres:
        nreg = len(sorted_regionlabels)
        split = hemi_split
        if split is None:
            split = next((k for k, r in enumerate(sorted_regionlabels) if str(r).startswith("R")), nreg // 2)

        trans = ax.get_xaxis_transform()
        y = 1.03
        ax.plot([0, split], [y, y], transform=trans, clip_on=False, color="k", lw=2)
        ax.plot([split, nreg], [y, y], transform=trans, clip_on=False, color="k", lw=2)
        ax.text(split / 2, y + 0.01, "L", transform=trans, ha="center", va="bottom", fontsize=12)
        ax.text((split + nreg) / 2, y + 0.01, "R", transform=trans, ha="center", va="bottom", fontsize=12)

    ax.set_xlabel("Region Group")
    ax.set_ylabel("Region Group")
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()

    if save and out_png is not None:
        _SaveFigure(fig, out_png, out_dir, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
