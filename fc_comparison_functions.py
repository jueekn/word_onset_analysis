"""fc_comparison_functions.py — per-session FC compute + cross-metric aggregation.

Centerpiece of the project. Loads per-session EEG + pairs, runs each FC
metric, regionalizes to (region × region) matrices, aggregates across
subjects, computes hub scores, bootstrap-evaluated subsample similarity,
and case-study contrast tests.

Dominated by pandas + mne_connectivity + matplotlib pyplot method chains
— all weakly stubbed. File-level narrowing for the library-stub-noise
rules; the other strict-mode rules stay on. (compute_pac uses pactools
via a function-scope import; compute_hub_scores' eigencent path now uses
numpy.linalg.eigh, having dropped its NetworkX dependency.)
"""
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportAttributeAccessIssue=warning, reportConstantRedefinition=warning
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import ttest_1samp  # pyright: ignore[reportMissingTypeStubs]
from scipy.stats import t as tdist  # pyright: ignore[reportMissingTypeStubs]
from statsmodels.stats.multitest import multipletests  # pyright: ignore[reportMissingTypeStubs]

import functools
import os
from os.path import join

import pandas as pd

import matplotlib.pyplot as plt

from misc import *  # noqa: F401,F403
from mne_connectivity import (  # pyright: ignore[reportMissingTypeStubs]
    spectral_connectivity_epochs, envelope_correlation)
# pactools is imported at function scope inside compute_pac (the only
# caller). The high-level Comodulogram API was removed in favor of
# pactools.bandpass_filter.multiple_band_pass + a direct Ozkurt MI
# matmul; no top-level import is needed.

from project_paths import (
    SCRATCH_DIR as _SCRATCH_DIR,
    MT_BANDWIDTH, COMPUTATION_METRICS, RUN_METRIC, SUBTRACT_ERP,
    REAL_DATA_BUFFER_MS,
    TIME_BIN_MS, MT_WINDOW_MS, LONGETAL,  # noqa: F401 (re-exported to the build scripts)
)

root_dir: str = str(_SCRATCH_DIR)

import helper
from helper import *  # noqa: F401,F403  # pyright: ignore[reportAssignmentType]
helper.root_dir = root_dir

from pathlib import Path
import pickle

NDArrayAny = npt.NDArray[Any]


def load_pickle(path: str) -> Any:
    """pickle.load. A session result's `reg_full` is refreshed from the session's
    CURRENT pairs table, so atlas fixes apply without recomputing."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    if isinstance(d, dict) and "sid" in d and "reg_full" in d:
        new = session_regions(*map(str, d["sid"][:3]))
        if new is not None and len(new) == len(d["reg_full"]):
            d["reg_full"] = new
    return d


@functools.lru_cache(maxsize=None)
def session_regions(sub: str, exp: str, sess: str) -> NDArrayAny | None:
    """Region label per pair from the session's pairs table (None if absent).
    Depth pairs with no volumetric atlas label get a neighbour label
    (mni_neighbour_labels)."""
    pairs = helper.get_pairs(pd.Series({"sub": sub, "exp": exp, "sess": int(sess)}))
    if pairs is None:
        return None
    fine = np.array(helper.regionalize_electrodes_by_type(pairs), dtype=object)   # writable copy
    todo = (pairs["type_1"].isin(["D", "UD"]).to_numpy()
            & (helper.get_atlas_labels_by_type(pairs)["atlas"].astype(str) == "no atlas").to_numpy())
    if todo.any():
        fine[todo] = mni_neighbour_labels(sub, pairs.loc[todo, ["mni.x", "mni.y", "mni.z"]].to_numpy(float))
    return fine


MNI_LABEL_RADIUS_MM, MNI_LABEL_MIN_NEIGHBOURS = 5.0, 3


@functools.lru_cache(maxsize=1)
def _depth_label_reference() -> tuple[Any, NDArrayAny, NDArrayAny]:
    """(KD-tree of MNI coords, region label, subject) for every depth pair with a
    stein/das/wb label, over all sessions' pairs tables (one row per subject x pair)."""
    from glob import glob
    from scipy.spatial import cKDTree
    rows = []
    for f in glob(join(root_dir, "electrode_information", "pairs", "*_pairs.json")):
        sub = os.path.basename(f).rsplit("_", 3)[0]
        p = pd.read_json(f).fillna("nan")
        ok = (p["type_1"].isin(["D", "UD"]).to_numpy()
              & helper.get_atlas_labels_by_type(p)["atlas"].isin(["stein.region", "das.region", "wb.region"]).to_numpy())
        fine = np.asarray(helper.regionalize_electrodes_by_type(p), dtype=object)
        for i in np.flatnonzero(ok):
            if isinstance(fine[i], str):
                rows.append((sub, p["label"].iloc[i], fine[i], *p[["mni.x", "mni.y", "mni.z"]].iloc[i]))
    d = pd.DataFrame(rows, columns=["sub", "label", "region", "x", "y", "z"]).drop_duplicates(["sub", "label"])
    d = d[np.isfinite(d[["x", "y", "z"]].astype(float)).all(axis=1)]
    return cKDTree(d[["x", "y", "z"]].to_numpy(float)), d["region"].to_numpy(object), d["sub"].to_numpy(object)


def mni_neighbour_labels(sub: str, xyz: NDArrayAny) -> NDArrayAny:
    """Region of each MNI point: majority label of expert/atlas-labelled depth pairs
    of OTHER subjects within MNI_LABEL_RADIUS_MM (>= MNI_LABEL_MIN_NEIGHBOURS), else
    NaN. Leave-one-subject-out on labelled pairs (2026-10-03): 89% 12-ROI / 79%
    fine-region agreement, 70% of pairs covered."""
    from collections import Counter
    tree, region, subs = _depth_label_reference()
    out = np.full(len(xyz), np.nan, dtype=object)
    for i, nb in enumerate(tree.query_ball_point(np.nan_to_num(xyz, nan=1e6), MNI_LABEL_RADIUS_MM)):
        labs = [region[j] for j in nb if subs[j] != sub]
        if len(labs) >= MNI_LABEL_MIN_NEIGHBOURS:
            out[i] = Counter(labs).most_common(1)[0][0]
    return out


def _assert_time_axis_covers(
    t: NDArrayAny, min_pre: float, max_post: float,
    contrast_label: str, tol: float = 10.0,
) -> None:
    """Raise ValueError if the EEG time axis (ms) does not cover [min_pre, max_post]
    within `tol` ms on either side. `tol` accommodates the slowest native sample
    rate (>= 100 Hz) that could leave the resampled axis short by one native-rate
    sample at either end."""
    if t[0] > min_pre + tol or t[-1] < max_post - tol:
        raise ValueError(
            f"Time axis {t[0]} to {t[-1]} does not cover required "
            f"window of {min_pre} to {max_post} ms for '{contrast_label}' contrast "
            f"(tolerance {tol:.1f} ms)."
        )


def assert_windows_disjoint(
    pre_win: tuple[float, float], post_win: tuple[float, float],
) -> None:
    """Raise ValueError if two (lo, hi) ms analysis windows overlap.

    The word off / word on contrast must compare temporally disjoint real-data
    epochs. Touching at a shared boundary (hi == lo) is allowed.
    """
    (lo1, hi1), (lo2, hi2) = pre_win, post_win
    if not (hi1 <= lo2 or hi2 <= lo1):
        raise ValueError(
            f"analysis windows overlap: pre={pre_win} post={post_win}; "
            "pre/post must be disjoint."
        )


def equalize_time_length(*arrays: NDArrayAny) -> tuple[NDArrayAny, ...]:
    """Truncate the last (time) axis of each array to the common minimum length.

    Resolves the #88 off-by-one (e.g. 150 vs 149 samples from inclusive time
    masks + sub-sample resample grid phase) so paired pre/post epochs carry
    identical sample counts before FC estimation. Dropped samples are split
    SYMMETRICALLY across both ends (floor(d/2) from the start, the remainder
    from the end) so the kept window stays centred; an odd surplus drops the
    extra sample from the end.
    """
    if not arrays:
        raise ValueError("equalize_time_length requires at least one array")
    n = min(int(a.shape[-1]) for a in arrays)
    if n <= 0:
        raise ValueError(f"empty time axis after equalization (n={n})")
    out: list[NDArrayAny] = []
    for a in arrays:
        drop = int(a.shape[-1]) - n
        left = drop // 2
        out.append(a[..., left:int(a.shape[-1]) - (drop - left)])
    return tuple(out)


from project_paths import BANDS as bands


def symmetrize_dense(mat: NDArrayAny, diag_value: float = 1.0, eps: float = 1e-12) -> NDArrayAny:
    '''Creates symmetrical regionxregion matrices if unsymmetrical (mne outputs are lower triangular by default)'''
    mat = np.asarray(mat)
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1]
    n = mat.shape[0]

    iu = np.triu_indices(n, 1)
    upper = mat[iu]

    upper_empty = (
        np.all(~np.isfinite(upper)) or
        np.nanmax(np.abs(np.nan_to_num(upper))) < eps
    )

    if upper_empty:
        out = mat + mat.T         
    else:
        out = 0.5 * (mat + mat.T)  

    np.fill_diagonal(out, diag_value)
    return out
 

def band_analytic(x: NDArrayAny, sf: float, fmin: float, fmax: float) -> NDArrayAny:
    """Band-pass + Hilbert over the whole (buffered) clip, so the filter edge sits
    in the buffer (as riley-thesis compute_aec_buffered)."""
    import mne
    from scipy.signal import hilbert
    from mne.filter import next_fast_len
    filt = mne.filter.filter_data(np.asarray(x, float), sf, fmin, fmax, verbose=False)
    n = filt.shape[-1]
    return hilbert(filt, N=next_fast_len(n), axis=-1)[..., :n]


def compute_aec_c(analytic: NDArrayAny) -> NDArrayAny:
    """Orthogonalized AEC of a complex analytic signal (trials, ch, time):
    mne_connectivity.envelope_correlation (pairwise, signed r, as riley-thesis),
    mean over trials, symmetric, NaN diagonal. |r| > 1 comes from (near-)flat
    epochs; as in riley-thesis the session is excluded (raises)."""
    out = np.squeeze(np.asarray(envelope_correlation(
        analytic, orthogonalize="pairwise", absolute=False, verbose=False).get_data(output="dense")))
    if out.ndim == 3:
        out = np.nanmean(out, axis=0)
    out = symmetrize_dense(out, diag_value=np.nan)
    if np.any(np.abs(out[np.isfinite(out)]) > 1) or np.isinf(out).any():
        raise ValueError("AEC-c |r| > 1 (near-flat epochs); session excluded")
    return out


def window_fc(x: NDArrayAny, sf: float, t0: float, win: tuple[float, float], m: str,
              fmin: float, fmax: float, nw_ms: float, n_buf: int,
              analytic: NDArrayAny | None = None) -> NDArrayAny:
    """Metric `m` over window `win` (ms) of `x`, whose first sample is at t0 ms.
    aec_c: from the whole-clip `analytic`; pac: the window plus n_buf buffer
    samples each side; phase metrics: MNE multitaper on the window, bandwidth
    scaled so the time-bandwidth product equals that of an nw_ms window."""
    i = int(round((win[0] - t0) * sf / 1000.0))
    n = int(round((win[1] - win[0]) * sf / 1000.0))
    if m == "aec_c":
        return compute_aec_c(analytic[..., i:i + n])
    if m == "pac":
        return compute_pac(x[..., i - n_buf:i + n + n_buf], sf,
                           buffer_left_samples=n_buf, buffer_right_samples=n_buf)
    return compute_spectral_fc(x[..., i:i + n], sf, method=m, fmin=fmin, fmax=fmax,
                               bandwidth=MT_BANDWIDTH * nw_ms / (win[1] - win[0]))

FC_MODES = ("multitaper", "hilbert")   # hilbert: power only (longetal)
POWER_MODE = "hilbert" if LONGETAL else "multitaper"


def band_dirname(band: str, fc_mode: str = "multitaper") -> str:
    """On-disk directory segment for a (band, fc_mode) pair.

    multitaper -> "high_gamma", hilbert -> "high_gamma__hilbert", so the two
    estimators never overwrite each other's pickles.
    """
    if fc_mode not in FC_MODES:
        raise ValueError(f"fc_mode must be one of {FC_MODES}, got {fc_mode!r}")
    return band if fc_mode == "multitaper" else f"{band}__{fc_mode}"


def compute_spectral_fc(
    data: NDArrayAny,
    sfreq: float,
    method: str,
    fmin: float,
    fmax: float,
    faverage: bool = True,
    bandwidth: float | None = None,
) -> NDArrayAny:
    """MNE multitaper phase connectivity (non-multivariate methods), symmetric
    (n_ch, n_ch) with a NaN diagonal. bandwidth defaults to config mt_bandwidth."""
    con = spectral_connectivity_epochs(
        data, method=method, mode="multitaper",
        sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=faverage,
        mt_adaptive=False, mt_bandwidth=bandwidth or MT_BANDWIDTH, n_jobs=1, verbose=False,
    )
    out = con.get_data(output="dense")
    if faverage:
        out = out[..., 0]
    out = np.asarray(out)
    if out.ndim == 2:
        out = symmetrize_dense(out, diag_value=np.nan)
    return out


# Pre/post spec: PRE_WORD events load at the pre window and WORD events at the
# post window (disjoint), each widened by the real-data buffer. PAC keeps the
# buffer through the envelope; the phase metrics drop it then equalize lengths.
PREPOST_SPEC: dict[str, dict[str, Any]] = {
    "word_on": {"pre_type": "PRE_WORD", "post_type": "WORD",
                "pre_win": (-700.0, -100.0), "post_win": (0.0, 600.0)},
}
# 200 ms epochs of the word presentation, each minus the mean of the same-length
# epochs of the word-off window (the time-resolved synchrony figure).
EPOCHS: tuple[tuple[float, float], ...] = tuple((s, s + 200.0) for s in range(0, 1600, 200))
EPOCH_BASELINE: tuple[tuple[float, float], ...] = tuple((s, s + 200.0) for s in range(-700, -100, 200))
if LONGETAL:   # Long et al.: blank screen vs word on screen
    PREPOST_SPEC["word_on"].update(pre_win=tuple(map(float, LONGETAL["pre_win"])),
                                   post_win=tuple(map(float, LONGETAL["post_win"])))
    if LONGETAL.get("baseline") == "full_isi":   # longest blank; each trial masked to its own blank later
        PREPOST_SPEC["word_on"]["pre_win"] = (-1000.0, 0.0)


def compute_prepost_separate(
    dfrow: pd.Series, beh: str, events: pd.DataFrame,
    metric_list: Sequence[str], fmin: float, fmax: float,
    overlap_mask: NDArrayAny | None, real_data_buffer_ms: float,
    simulation_tag: str | None = None,
    load_fn: Any = None,
) -> dict[str, dict[str, NDArrayAny]]:
    """Word off vs word on FC from SEPARATE event-locked pre/post loads. The
    pre-type events (events.attrs['mask'] == False) load at the pre window,
    post-type (mask == True) at the post window extended to the last EPOCHS end,
    each widened by a real-data buffer (see below). With config subtract_erp, each
    window's evoked response (across-trial mean, per channel and sample) is
    subtracted first. Returns
    {metric: {baseline, succ, diff, epochs}}: diff = FC(post) - FC(pre); epochs =
    FC(each EPOCHS window) - mean FC(EPOCH_BASELINE windows), (8, ch, ch).

    `load_fn(dfrow, events, window, real_data_buffer_ms, simulation_tag)`
    defaults to get_beh_eeg.

    Guards (enforced just before the events are consumed):
      * both POST (mask=True) and PRE (mask=False) groups are present — catches
        stale pre-item-4 event files that lack the PRE_WORD/PRE_REC_WORD copies;
      * PRE and POST share onsets row-for-row (PRE_* are copies of WORD/REC_WORD);
      * each loaded clip's time axis covers its analysis window (catches a clip
        clipped at the recording boundary).
    """
    if load_fn is None:
        load_fn = get_beh_eeg          # module-level (helper.* via star import)
    spec = PREPOST_SPEC[beh]
    pre_win = tuple(spec["pre_win"]); post_win = tuple(spec["post_win"])
    assert_windows_disjoint(pre_win, post_win)

    full_mask = np.asarray(events.attrs["mask"], dtype=bool)   # True=post, False=pre
    if not full_mask.any() or full_mask.all():
        raise ValueError(
            f"{beh}: event file must contain BOTH post (mask=True) and pre "
            f"(mask=False) events; got {int(full_mask.sum())} post / "
            f"{int((~full_mask).sum())} pre. Regenerate events via load_events "
            f"(needs the PRE_{spec['post_type']} copies) — looks like a "
            "pre-item-4 event file."
        )
    post_events = events[full_mask].copy(); post_events.attrs = dict(events.attrs)
    post_events.attrs["mask"] = full_mask[full_mask]
    pre_events = events[~full_mask].copy(); pre_events.attrs = dict(events.attrs)
    pre_events.attrs["mask"] = full_mask[~full_mask]

    post_off = np.asarray(post_events["eegoffset"])
    pre_off = np.asarray(pre_events["eegoffset"])
    if post_off.shape != pre_off.shape or not np.array_equal(post_off, pre_off):
        raise ValueError(
            f"{beh}: PRE and POST events must share onsets row-for-row (PRE_* are "
            f"copies of {spec['post_type']}); {post_off.shape[0]} post vs "
            f"{pre_off.shape[0]} pre, content mismatch — malformed event file."
        )

    post_load = (post_win[0], max(post_win[1], EPOCHS[-1][1]))
    post_eeg, _ = load_fn(dfrow, post_events, post_load, real_data_buffer_ms, simulation_tag)
    pre_eeg, _ = load_fn(dfrow, pre_events, pre_win, real_data_buffer_ms, simulation_tag)
    # coverage guard: the loaded clip must span its analysis window (the buffer
    # extends beyond it, so a short axis means boundary clipping).
    _assert_time_axis_covers(np.asarray(post_eeg.time), post_load[0], post_load[1], f"{beh} post")
    _assert_time_axis_covers(np.asarray(pre_eeg.time), pre_win[0], pre_win[1], f"{beh} pre")
    sf = float(post_eeg.samplerate)
    post = np.asarray(post_eeg.data, float); pre = np.asarray(pre_eeg.data, float)
    if SUBTRACT_ERP:   # subtract the evoked response (across-trial mean) per window
        post -= post.mean(axis=0, keepdims=True)
        pre -= pre.mean(axis=0, keepdims=True)
    arms = {"post": (post, float(post_eeg.time[0])), "pre": (pre, float(pre_eeg.time[0]))}
    n_buf = int(round(real_data_buffer_ms * sf / 1000.0))
    nw_ms = post_win[1] - post_win[0]

    out_m: dict[str, dict[str, NDArrayAny]] = {}
    for m in metric_list:
        # AEC-c on bands.gamma (70-110 Hz, as riley-thesis): the envelope must stay
        # below the band's lower edge (Bedrosian), which a 70-150 band violates.
        an = ({k: band_analytic(x, sf, *bands["gamma"]) for k, (x, _) in arms.items()}
              if m == "aec_c" else {})
        def fc_(arm: str, win: tuple[float, float]) -> NDArrayAny:
            x, t0 = arms[arm]
            return apply_overlap_mask(window_fc(x, sf, t0, win, m, fmin, fmax, nw_ms, n_buf,
                                                an.get(arm)), overlap_mask)
        C_post, C_pre = fc_("post", post_win), fc_("pre", pre_win)
        out_m[m] = {"baseline": C_pre, "succ": C_post, "diff": C_post - C_pre}
        if m != "pac":
            base = np.nanmean([fc_("pre", w) for w in EPOCH_BASELINE], axis=0)
            out_m[m]["epochs"] = np.stack([fc_("post", w) - base for w in EPOCHS])
    return out_m


def compute_session_fc(
    dfrow: pd.Series, beh: str, band: str, metrics: Sequence[str],
    overlap_mask: NDArrayAny | None = None, simulation_tag: str | None = None,
) -> dict[str, dict[str, NDArrayAny]] | None:
    """{metric: {baseline, succ, diff}} for one session, or None without events."""
    ev = load_events(dfrow, beh)
    if ev is None:
        return None
    fmin, fmax = bands[band]
    return compute_prepost_separate(dfrow, beh, ev, metrics, fmin, fmax, overlap_mask,
                                    REAL_DATA_BUFFER_MS, simulation_tag=simulation_tag)


def compute_pac(
    data: NDArrayAny,
    sfreq: float,
    phase_band: tuple[float, float] = bands["low"],
    amp_band: tuple[float, float] = bands["gamma"],
    buffer_left_samples: int = 0,
    buffer_right_samples: int = 0,
    keep_diagonal: bool = False,
) -> NDArrayAny:
    """Ozkurt phase-amplitude coupling, channel × channel (phase channel i, amplitude
    channel j). The diagonal (within-electrode PAC) is NaN unless `keep_diagonal`.

    Mathematically identical to looping ``pactools.Comodulogram(method='ozkurt',
    n_surrogates=0).fit(low_sig=data[:, i], high_sig=data[:, j])`` for every
    (i, j) pair, but hoists pactools' band-pass filtering out of the inner
    loop: the n_ch² calls to ``fit`` each triggered ``multiple_band_pass`` on
    BOTH bands, so the original implementation did 2*n_ch² channel filterings.
    Here we filter each channel ONCE per band (2*n_ch filterings total) and
    reuse the cached analytic signals to evaluate the closed-form Ozkurt MI
    for every off-diagonal (i, j).

    The Ozkurt modulation index for a single (low, high) channel pair, with
    epochs concatenated along time (matching pactools' internal reshape in
    ``_comodulogram`` when no mask is given), is::

        phi = angle(bandpass_hilbert(low_sig))     # phase of low band
        a   = abs(bandpass_hilbert(high_sig))      # amplitude envelope
        MI  = |mean(a * exp(1j*phi))| * sqrt(N) / sqrt(sum(a**2))

    See ``pactools/comodulogram.py::_one_modulation_index`` (method='ozkurt')
    and the ``filtered_low.reshape(filtered_low.shape[0], -1)`` flatten step
    in ``_comodulogram`` for the reference implementation that this matches.
    """
    from pactools.bandpass_filter import multiple_band_pass

    n_ep, n_ch, n_t = data.shape
    data_ep = data.transpose(1, 0, 2)  # (n_ch, n_ep, n_t)
    low_fq_range = np.array([float(np.mean(phase_band))])
    high_fq_range = np.array([float(np.mean(amp_band))])
    low_fq_width = float(phase_band[1] - phase_band[0])
    high_fq_width = float(amp_band[1] - amp_band[0])

    # Filter all channels in one shot per band by flattening (channel × epoch)
    # into rows. pactools' BandPassFilter.transform applies fftconvolve
    # row-by-row, so flattened rows stay independent.
    flat = data_ep.reshape(n_ch * n_ep, n_t)
    # multiple_band_pass returns shape (n_frequencies, n_signals, n_points)
    # with n_frequencies==1 here.
    filt_low = multiple_band_pass(
        flat, sfreq, low_fq_range, low_fq_width,
    )[0].reshape(n_ch, n_ep, n_t)
    filt_high = multiple_band_pass(
        flat, sfreq, high_fq_range, high_fq_width,
    )[0].reshape(n_ch, n_ep, n_t)

    # Real-data buffer: the band-pass + Hilbert above ran on the FULL (buffered)
    # window; crop the buffer off the analytic signals so the Ozkurt MI is
    # evaluated only on the interior. Mathematically identical to pactools'
    # Comodulogram.fit(..., mask=<buffer True>) ("PAC evaluated where mask is
    # False"). buffer_left==buffer_right==0 is a no-op (bit-identical to the
    # unbuffered path; guarded by the test_compute_pac_*_snapshot_pin tests).
    if buffer_left_samples < 0 or buffer_right_samples < 0:
        raise ValueError(
            f"buffer samples must be >= 0, got left={buffer_left_samples} "
            f"right={buffer_right_samples}"
        )
    if buffer_left_samples or buffer_right_samples:
        if buffer_left_samples + buffer_right_samples >= n_t:
            raise ValueError(
                f"buffer_left_samples={buffer_left_samples} + "
                f"buffer_right_samples={buffer_right_samples} >= n_times={n_t}; "
                "nothing left after crop."
            )
        end = n_t - buffer_right_samples if buffer_right_samples > 0 else n_t
        filt_low = filt_low[:, :, buffer_left_samples:end]
        filt_high = filt_high[:, :, buffer_left_samples:end]
        n_t = filt_low.shape[-1]

    # Concatenate epochs into time (matches the no-mask reshape in
    # pactools.comodulogram._comodulogram).
    # phase[i]: shape (n_ep*n_t,) complex unit phasor
    # amp[j]:   shape (n_ep*n_t,) real envelope
    phase_phasor = np.exp(
        1j * np.angle(filt_low.reshape(n_ch, n_ep * n_t))
    )  # shape (n_ch, N)
    amp = np.abs(filt_high.reshape(n_ch, n_ep * n_t))  # shape (n_ch, N)
    # norm_a[j] = sqrt(sum(amp[j]**2))  (== np.linalg.norm)
    norm_a = np.sqrt(np.einsum("jt,jt->j", amp, amp))  # shape (n_ch,)

    # MI[i, j] = |mean_t(amp[j,t] * exp(1j * phi[i,t]))| * sqrt(N) / norm_a[j]
    # Vectorized: form Z[i, j] = sum_t(amp[j, t] * phasor[i, t]) / N
    # Then MI = |Z| * sqrt(N) / norm_a[j].
    N = phase_phasor.shape[1]
    # phase_phasor: (n_ch_i, N), amp: (n_ch_j, N) → Z: (n_ch_i, n_ch_j)
    Z = phase_phasor @ amp.T  # complex (n_ch, n_ch); sum over time
    Z /= N
    M = np.abs(Z) * (np.sqrt(N) / norm_a[np.newaxis, :])
    if not keep_diagonal:
        np.fill_diagonal(M, np.nan)
    return M.astype(float, copy=False)


def stars(q):
    """Significance stars from a (corrected) p-value."""
    if not np.isfinite(q): return ""
    if q < 0.001: return "***"
    if q < 0.01:  return "**"
    if q < 0.05:  return "*"
    return ""


# =============================================================================
# ROI analysis layer
# =============================================================================
# Shared machinery for the ROI-level build scripts:
#
#   build_roi_power.py       local spectral power per Burke ROI
#   build_roi_synchrony.py   phase FC (compute) + collapsed synchrony per ROI
#
# Everything those scripts have in common lives here -- the behavior registry,
# the Burke ROI vocabulary, pair geometry,
# the per-ROI test, the session dispatcher and the ROI figure --
# so each script contains only what is unique to it and none of them has to
# import a plotting script to get a constant.
#
# Nothing in this section imports the build or plot scripts; the dependency
# runs one way (scripts -> here).

# --- contrast registry -------------------------------------------------------
# `lo` is the reference arm, `hi` the active one, diff = hi - lo; the names are
# the saved condition dirs (compute_prepost_separate).
BEH_CONTRASTS: dict[str, dict[str, str]] = {
    "word_on": {"beh": "word_on", "lo": "baseline", "hi": "succ",
                "lo_label": "Word Off", "hi_label": "Word On",
                "title": "Word Presentation vs Pre-Word"},
}
PHASE_METRICS: tuple[str, ...] = tuple(COMPUTATION_METRICS)


def contrast(beh: str) -> dict[str, str]:
    """{lo, hi, lo_label, hi_label, title} for `beh`. Raises on an unknown beh."""
    try:
        return BEH_CONTRASTS[beh]
    except KeyError:
        raise KeyError(
            f"unknown behavior {beh!r}; known: {list(BEH_CONTRASTS)}") from None


# --- Burke ROI vocabulary ----------------------------------------------------
# 12 Burke ROIs, fixed plotting order (left block then right block).
LOBES: list[str] = ["frontal", "temporal", "parietal", "occipital", "limbic",
                    "hippocampus"]
ROI_ORDER: list[str] = [f"{h}-{lobe}" for h in ("L", "R") for lobe in LOBES]

# Okabe-Ito, one hue per Burke lobe (CVD-safe). Hemisphere is NOT color-coded:
# it is carried by x position (Left block / Right block), so identity never
# rests on color alone.
LOBE_COLORS: dict[str, str] = {
    "frontal": "#0072B2",
    "temporal": "#E69F00",
    "parietal": "#009E73",
    "occipital": "#CC79A7",
    "limbic": "#56B4E9",
    "hippocampus": "#D55E00",
}

# Minimum subjects contributing an ROI before it is tested / drawn.
# Minimum subjects per test; a subject
# needs only 1 electrode in a region. Simulations (20 subjects) use 5 (resolve_roots).
MIN_SUBJECTS_ROI: int = 30     # region-level tests (ROIs, subregions, bins)
MIN_SUBJECTS_PAIR: int = 100   # region pairs (epoch network edges and hubs), as riley-thesis

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_burke_maps() -> dict[str, str]:
    """{region label: burke lobe} from region_to_burke_lobe.csv."""
    lobe = pd.read_csv(join(_HERE, "region_to_burke_lobe.csv"))
    return lobe.set_index("region")["burke_lobe"].to_dict()


def roi_of_reg_full(reg_full: Any, lobe_of: dict[str, str]) -> NDArrayAny:
    """Map each 'L amygdala' / 'R hippocampus' / nan label to a Burke ROI
    (e.g. 'R-temporal'), or None when the region has no Burke lobe.

    Region LABELS come from the repo's canonical type-aware cascade
    (helper.regionalize_electrodes_by_type): depths from VOLUMETRIC atlases,
    grid/strip from SURFACE atlases. That is independent of the coordinate
    policy in pair_xyz_lead -- labels and geometry are separate.
    """
    out: list[str | None] = []
    for v in np.asarray(reg_full, dtype=object):
        if not isinstance(v, str) or " " not in v:
            out.append(None)
            continue
        hemi, region = v.split(" ", 1)
        lobe = lobe_of.get(region, "")
        if LONGETAL and (lobe in LONGETAL.get("exclude_lobes", ())            # Long et al.: no limbic/sublobar
                         or region in LONGETAL.get("exclude_regions", ())):
            lobe = ""
        out.append(f"{hemi}-{lobe}"
                   if (lobe in LOBES and hemi in ("L", "R")) else None)
    return np.array(out, dtype=object)


# --- pair geometry -----------------------------------------------------------
def pair_xyz_lead(dfrow: pd.Series) -> tuple[NDArrayAny, NDArrayAny]:
    """(N,3) pair centroid MNI coords and (N,) lead prefix for a session's pairs.

    The BIDS electrode table gives every contact in one space
    (MNI152NLin6ASym), so every pair centroid -- depth, grid or strip -- lives
    in the same frame and any two are directly comparable.
    """
    p0 = helper.get_pairs(dfrow)
    xyz = p0[["mni.x", "mni.y", "mni.z"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    lead = p0["label"].str.extract(r"^([A-Za-z]+)")[0].to_numpy()
    return xyz, lead


def dfrow_from_sid(sid: Sequence[Any]) -> pd.Series:
    """Stored `sid` tuple -> the dfrow the loaders expect."""
    return pd.Series({"sub": str(sid[0]), "exp": str(sid[1]), "sess": int(sid[2])})


def electrode_sync(M: Any) -> NDArrayAny:
    """Per-electrode synchrony: mean over all its partners (as Rao et al. 2025);
    NaN entries (diagonal, contact-sharing pairs) are skipped."""
    M = np.asarray(M, float)
    n = np.isfinite(M).sum(axis=1)
    return np.where(n > 0, np.nansum(M, axis=1) / np.maximum(n, 1), np.nan)


# --- per-ROI statistics ------------------------------------------------------
def mean_ci(v: Any, conf: float = 0.95) -> tuple[float, float]:
    """(mean, half-width of the t-based CI). Half-width is nan for n < 2."""
    v = np.asarray(v, float)
    if v.size < 2:
        return (float(v.mean()) if v.size else np.nan), np.nan
    sem = float(np.std(v, ddof=1) / np.sqrt(v.size))
    return float(np.mean(v)), float(tdist.ppf(0.5 + conf / 2, v.size - 1) * sem)


def roi_stats(
    tbl: pd.DataFrame, measure: str, roi_order: Sequence[str] = tuple(ROI_ORDER),
    min_subjects: int | None = None,
) -> pd.DataFrame:
    """Per-ROI n / mean / CI / median / IQR + one-sample t vs 0, BH-FDR over ROIs.

    `tbl` is one row per (subject, ROI). One `ttest_1samp(v, 0)` per ROI, but it
    means two different things depending on the measure:

      contrast measures (a Cohen's d, a dB change, a saved diff matrix, an r)
        already difference the two conditions inside each subject, so the
        one-sample t on them IS the paired t -- the same test the connectivity
        figures run as ttest_rel(hi, lo).
      level measures (z-scored power or synchrony in one condition) are not
        differences and nothing is paired; 0 is simply where that subject's own
        whole-brain mean sits after the z-score, so the test asks whether the
        ROI departs from it.
    """
    rows: list[dict[str, Any]] = []
    tested: list[str] = []
    pv: list[float] = []
    for roi in roi_order:
        v = tbl.loc[tbl["roi"] == roi, measure].dropna().to_numpy(float)
        t = p = np.nan
        # t is undefined at zero variance (every subject identical) -- rare, but
        # it would come back as a nan + RuntimeWarning rather than an error
        if v.size >= (min_subjects or MIN_SUBJECTS_ROI) and np.std(v, ddof=1) > 0:
            res = ttest_1samp(v, 0.0)
            t, p = float(res.statistic), float(res.pvalue)
            tested.append(roi)
            pv.append(p)
        mean, half = mean_ci(v)
        rows.append({
            "roi": roi, "n_subjects": int(v.size),
            "mean": mean,
            "sem": (float(np.std(v, ddof=1) / np.sqrt(v.size))
                    if v.size > 1 else np.nan),
            "ci95_lo": mean - half, "ci95_hi": mean + half,
            "median": float(np.median(v)) if v.size else np.nan,
            "q25": float(np.percentile(v, 25)) if v.size else np.nan,
            "q75": float(np.percentile(v, 75)) if v.size else np.nan,
            "t": t, "p": p, "q": np.nan,
        })
    out = pd.DataFrame(rows).set_index("roi")
    if pv:
        q = multipletests(np.asarray(pv), method="fdr_bh")[1]
        out.loc[tested, "q"] = q
    return out.reset_index()


def print_roi_stats(st: pd.DataFrame, title: str) -> None:
    """The per-ROI table every build script prints before drawing."""
    print(f"\n{title}")
    print(f"{'ROI':16s} {'#subj':>6} {'mean':>9} {'median':>9} {'t':>7} "
          f"{'p':>9} {'q':>9}")
    for _, r in st.iterrows():
        print(f"{r['roi']:16s} {r['n_subjects']:>6d} {r['mean']:>9.4f} "
              f"{r['median']:>9.4f} {r['t']:>7.2f} {r['p']:>9.3g} "
              f"{r['q']:>9.3g}")


# --- ROI figures -------------------------------------------------------------
def _roi_positions(roi_order: Sequence[str], lobes: Sequence[str]) -> list[float]:
    """x positions with a gap between the left-hemisphere and right blocks."""
    return [i + (0.8 if i >= len(lobes) else 0.0) for i in range(len(roi_order))]


def _roi_axis_furniture(
    ax: Any, pos: Sequence[float], roi_order: Sequence[str],
    lobes: Sequence[str],
) -> None:
    """Shared axis dress: 0 line, hemisphere divider, tick labels, LEFT/RIGHT."""
    ax.axhline(0, color="0.7", lw=0.9, zorder=0)
    ax.axvline(len(lobes) - 0.1, color="0.85", lw=1.0, zorder=0)
    ax.set_xlim(-0.7, len(roi_order) + 0.5)
    ax.set_xticks(list(pos))
    ax.set_xticklabels([r.split("-", 1)[1] for r in roi_order], rotation=35,
                       ha="right", fontsize=9)
    for lab, r in zip(ax.get_xticklabels(), roi_order):   # lobe names double as the colour key
        lab.set_color(LOBE_COLORS.get(r.split("-", 1)[1], "k"))
        lab.set_fontweight("bold")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for name, sl in (("LEFT", slice(None, len(lobes))),
                     ("RIGHT", slice(len(lobes), None))):
        ax.annotate(name, (float(np.mean(pos[sl])), 1.02),
                    xycoords=("data", "axes fraction"), ha="center",
                    fontsize=9, color="0.35", fontweight="bold")


def roi_panel(
    ax: Any, tbl: pd.DataFrame, measure: str, rng: Any,
    roi_order: Sequence[str] = tuple(ROI_ORDER),
    lobes: Sequence[str] = tuple(LOBES),
    stats: pd.DataFrame | None = None,
) -> list[float]:
    """One 12-ROI panel over the per-subject values in `tbl[measure]`:
    median/IQR box + 1.5*IQR whiskers + mean diamond + subject points."""
    data: list[NDArrayAny] = []
    colors: list[str] = []
    pos = _roi_positions(roi_order, lobes)
    for roi in roi_order:
        data.append(tbl.loc[tbl["roi"] == roi, measure].dropna().to_numpy(float))
        colors.append(LOBE_COLORS.get(roi.split("-", 1)[1], "0.5"))

    drawn = [i for i, v in enumerate(data) if v.size >= MIN_SUBJECTS_ROI]

    if drawn:          # boxplot([]) raises in matplotlib >= 3.9
        bp = ax.boxplot([data[i] for i in drawn],
                        positions=[pos[i] for i in drawn],
                        widths=0.62, showfliers=False, patch_artist=True,
                        medianprops=dict(color="0.15", lw=1.6),
                        whiskerprops=dict(color="0.45", lw=1.0),
                        capprops=dict(color="0.45", lw=1.0))
        for patch, i in zip(bp["boxes"], drawn):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.35)
            patch.set_edgecolor(colors[i])
            patch.set_linewidth(1.4)

    # subject points first, so the estimate sits on top of them. Size and alpha
    # scale down as n grows: readable as individual subjects on a small subset,
    # a density cloud at the full 120-290 subjects per ROI.
    n_max = max((v.size for v in data), default=0)
    s_dot, a_dot = (9, 0.55) if n_max <= 40 else (4, 0.25)
    for i, v in enumerate(data):
        if v.size == 0:
            continue
        x = pos[i] + rng.uniform(-0.16, 0.16, v.size)
        ax.scatter(x, v, s=s_dot, color=colors[i], alpha=a_dot, linewidths=0,
                   zorder=3)

    for i in drawn:
        ax.scatter([pos[i]], [float(np.mean(data[i]))], marker="D", s=26,
                   facecolor="white", edgecolor="0.15", linewidths=1.2, zorder=4)

    _roi_axis_furniture(ax, pos, roi_order, lobes)
    _annotate_counts_and_stars(
        ax, pos, roi_order,
        n=[v.size for v in data],
        tops=[float(np.max(v)) if v.size else np.nan for v in data],
        stats=stats)
    return pos


def roi_bar_panel(
    ax: Any, stats: pd.DataFrame, roi_order: Sequence[str] = tuple(ROI_ORDER),
    lobes: Sequence[str] = tuple(LOBES),
) -> list[float]:
    """One 12-ROI panel of bars, across-subject mean +/- SEM (roi_stats), coloured by
    lobe; FDR stars in a fixed row along the top, n along the bottom."""
    pos = _roi_positions(roi_order, lobes)
    s = stats.set_index("roi")
    for i, roi in enumerate(roi_order):
        if roi not in s.index or s.loc[roi, "n_subjects"] < MIN_SUBJECTS_ROI:
            continue
        r = s.loc[roi]
        ax.bar(pos[i], r["mean"], yerr=r["sem"], width=0.7, capsize=3, edgecolor="0.15",
               ecolor="0.2", color=LOBE_COLORS.get(roi.split("-", 1)[1], "0.5"), zorder=3)
        ax.annotate(f"{int(r['n_subjects'])}", (pos[i], 0.015), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7, color="0.45")
        if stars(float(r["q"])):
            ax.annotate(stars(float(r["q"])), (pos[i], 0.93), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
    lo, hi = np.nanmin(stats["mean"] - stats["sem"]), np.nanmax(stats["mean"] + stats["sem"])
    pad = 0.12 * (max(hi, 0) - min(lo, 0))
    ax.set_ylim(min(lo, 0) - 1.4 * pad, max(hi, 0) + 1.6 * pad)   # room for n and stars
    _roi_axis_furniture(ax, pos, roi_order, lobes)
    return pos


def _annotate_counts_and_stars(
    ax: Any, pos: Sequence[float], roi_order: Sequence[str],
    n: Sequence[int], tops: Sequence[float], stats: pd.DataFrame | None,
) -> None:
    """n inside the axes at the bottom, FDR stars just past each ROI's estimate."""
    s = None if stats is None else stats.set_index("roi")
    for i, roi in enumerate(roi_order):
        if not n[i]:
            continue
        ax.annotate(f"{int(n[i])}", (pos[i], 0.015),
                    xycoords=("data", "axes fraction"), ha="center",
                    va="bottom", fontsize=7, color="0.45")
        if s is None or n[i] < MIN_SUBJECTS_ROI or roi not in s.index \
                or not np.isfinite(tops[i]):
            continue
        star = stars(float(s.loc[roi, "q"]))
        if star:
            ax.annotate(star, (pos[i], tops[i]), textcoords="offset points",
                        xytext=(0, 5), ha="center", va="bottom", fontsize=11,
                        fontweight="bold", color="k")


def roi_figure(
    panels: Sequence[tuple[str, Callable[[Any], Any]]],
    out_dir: str, stem: str, height: float = 3.6,
) -> None:
    """Stack one panel per measure, label each y axis, save .png + .pdf.

    `panels` is [(y-axis label, draw(ax)), ...]."""
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, height * n + 0.8))
    for ax, (ylabel, draw) in zip(np.atleast_1d(axes), panels):
        draw(ax)
        ax.set_ylabel(ylabel, fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    os.makedirs(out_dir, exist_ok=True)
    path = join(out_dir, stem)
    fig.savefig(f"{path}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    print(f"[saved] {path}.png / .pdf")
    plt.close(fig)


METRIC_LABELS: dict[str, str] = {"coh": "Coherence", "plv": "PLV", "ppc": "PPC", "ciplv": "ciPLV",
                                  "pli": "PLI", "wpli": "wPLI", "aec_c": "AEC-c", "pac": "PAC"}


def band_label(band: str) -> str:
    """'high_gamma' -> 'High Gamma (70–150 Hz)'."""
    lo, hi = bands[band]
    return f"{band.replace('_', ' ').title()} ({lo:g}–{hi:g} Hz)"


def band_stems(out_dir: str, stem: str, suffix: str) -> list[tuple[str, str, str]]:
    """[(band, its run metric, path)] for each band whose <stem><suffix> file exists
    in out_dir; stem has a {band} and optionally a {metric} field (config runs)."""
    out = []
    for band in bands:
        metric = RUN_METRIC.get(band, "")
        f = join(out_dir, stem.format(band=band, metric=metric) + suffix)
        if os.path.exists(f):
            out.append((band, metric, f))
    return out


def band_contrast_figure(out_dir: str, stem: str, col: str, ylabel: str, bars: bool = False) -> None:
    """hi - lo contrast, one row per band whose `<stem>_per_subject.csv` exists
    (band_stems); `{metric}` in ylabel becomes that band's metric label. Saved as
    stem without its `_{band}` / `_{metric}` fields; `bars`: bars of the mean +/- SEM
    (roi_bar_panel) instead of subject distributions, saved with suffix `_sem`."""
    panels = []
    for band, metric, f in band_stems(out_dir, stem, "_per_subject.csv"):
        tbl = pd.read_csv(f)
        st = roi_stats(tbl, col)
        panels.append((f"{band_label(band)}\n{ylabel.format(metric=METRIC_LABELS.get(metric, metric))}",
                       (lambda ax, st=st: roi_bar_panel(ax, st)) if bars else
                       (lambda ax, tbl=tbl, st=st: roi_panel(ax, tbl, col, np.random.default_rng(0), stats=st))))
    if panels:   # none when no band's run metric has been plotted yet
        roi_figure(panels, out_dir, stem.replace("_{band}", "").replace("_{metric}", "") + ("_sem" if bars else ""))


def epoch_network_figure(out_dir: str, stem: str, ylabel: str, top_n: int = 5,
                         max_lw: float = 6.0, extra_rows: Sequence[tuple[str, str, str]] = ()) -> None:
    """Rao et al. 2025 / Solomon et al. 2017 style network per EPOCHS window
    (columns) for each band with `<stem>_hubs.csv` (rows; band_stems).

    Hubs: regions whose mean on - off change over all their connections differs
    from 0 across subjects (two-stage BKY FDR over regions x epochs, as Rao);
    red node = positive hub, blue = negative, grey = not a hub. Edges: each hub's
    `top_n` strongest connections of either sign (ranked by |mean change|, as
    Rao), from region pairs with >= MIN_SUBJECTS_PAIR subjects; red =
    synchronized, blue = desynchronized, width proportional to |t| (Solomon)."""
    from matplotlib import patheffects
    from matplotlib.lines import Line2D
    from nilearn import plotting
    # rows: (row label, metric, hubs csv); `extra_rows` (e.g. PAC) are appended as given
    rows = [(band_label(b), m, f) for b, m, f in band_stems(out_dir, stem, "_hubs.csv")] + list(extra_rows)
    if not rows:
        return
    nE = len(EPOCHS)
    fig, axes = plt.subplots(len(rows), nE, figsize=(2.7 * nE + 2.5, 3.2 * len(rows) + 0.6),
                             squeeze=False)
    for r, (row_label, metric, f) in enumerate(rows):
        hubs = pd.read_csv(f)
        pairs = pd.read_csv(f.replace("_hubs.csv", "_stats.csv")).dropna(subset=["t"])
        pairs[["a", "b"]] = pairs["roi"].str.split("|", expand=True)
        pairs = pairs[pairs["a"] != pairs["b"]]
        cen = pd.read_csv(f.replace("_hubs.csv", "_centroids.csv"), index_col="roi")
        rois = sorted(cen.index, key=lambda x: (ROI_ORDER.index(x) if x in ROI_ORDER else 99, x))
        idx = {x: i for i, x in enumerate(rois)}
        xyz = cen.loc[rois, ["x", "y", "z"]].to_numpy()
        small = len(rois) > 12
        tmax = float(np.nanmax(np.abs(pairs["t"]))) if len(pairs) else 1.0
        for c, (w0, w1) in enumerate(EPOCHS):
            h = hubs[(hubs["epoch_ms"] == w0) & (hubs["q"] < 0.05)].set_index("roi")["t"]
            color = ["red" if h.get(x, 0) > 0 else "blue" if h.get(x, 0) < 0 else "0.75" for x in rois]
            size = [(30 if small else 90) if x in h.index else (6 if small else 25) for x in rois]
            disp = plotting.plot_connectome(np.zeros((len(rois), len(rois))), xyz, axes=axes[r, c],
                                            display_mode="z", node_color=color, node_size=size,
                                            node_kwargs={"edgecolors": "black", "linewidths": 0.6},
                                            colorbar=False, annotate=False)
            pe = pairs[pairs["epoch_ms"] == w0]
            for hub, ht in h.items():
                mine = pe[(pe["a"] == hub) | (pe["b"] == hub)]
                for _, e in mine.reindex(mine["mean"].abs().sort_values(ascending=False).index).head(top_n).iterrows():
                    if e["a"] not in idx or e["b"] not in idx:
                        continue
                    A = np.zeros((len(rois), len(rois)))
                    A[idx[e["a"]], idx[e["b"]]] = A[idx[e["b"]], idx[e["a"]]] = np.sign(e["mean"])
                    disp.add_graph(A, xyz, node_size=0, edge_cmap="bwr", edge_vmin=-1, edge_vmax=1,
                                   edge_kwargs={"linewidth": max_lw * abs(e["t"]) / tmax})
            if not small:   # 12 ROIs: shorthand label on top of everything ('L-frontal' -> 'LFr')
                gax = list(disp.axes.values())[0].ax
                for x, (px, py, _) in zip(rois, xyz):
                    hemi, lobe = x.split("-", 1)
                    below = lobe == "occipital"   # occipital labels under the node, away from parietal
                    gax.text(px, py + (-7 if below else 7), hemi + lobe[:2].title(), fontsize=9,
                             fontweight="bold", ha="center", va="top" if below else "bottom", zorder=1000,
                             path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])
            if r == 0:
                axes[r, c].set_title(f"{w0:g}–{w1:g} ms", fontsize=15)
        handles = [Line2D([], [], marker="o", ls="", color="red", ms=9, label="Positive hub"),
                   Line2D([], [], marker="o", ls="", color="blue", ms=9, label="Negative hub"),
                   Line2D([], [], color="red", lw=3, label="Coupled" if metric == "pac" else "Synchronized"),
                   Line2D([], [], color="blue", lw=3, label="Decoupled" if metric == "pac" else "Desynchronized")]
        axes[r, -1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.5),
                           frameon=False, fontsize=11, title_fontsize=12,
                           title=ylabel.format(metric=METRIC_LABELS.get(metric, metric)).replace(" (", "\n("))
        axes[r, 0].text(-0.08, 0.5, row_label, transform=axes[r, 0].transAxes,
                        rotation=90, ha="right", va="center", multialignment="center", fontsize=15)
    path = join(out_dir, stem.replace("_{band}", "").replace("_{metric}", ""))
    fig.savefig(f"{path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    print(f"[saved] {path}.png / .pdf")
    plt.close(fig)


def pretty_roi(name: Any) -> str:
    """'L-occipital' -> 'L-Occipital'; other labels just get a capital first letter."""
    t = str(name)
    if "-" in t and len(t.split("-", 1)[0]) == 1:
        hemi, rest = t.split("-", 1)
        return f"{hemi.upper()}-{rest[:1].upper()}{rest[1:]}"
    return t[:1].upper() + t[1:]


def bin_stats(tbl: pd.DataFrame, x: str, y: str, min_subjects: int | None = None,
              fdr: str = "fdr_bh") -> pd.DataFrame:
    """Per (ROI, x bin): one-sample t of the subject values `y` vs 0, BH-FDR over
    ALL roi x bin cells (the family is the whole map). Columns: roi, x, n, mean,
    sem, ci95 (t-based half-width, fc.mean_ci), t, p, q."""
    rows = []
    present = set(tbl["roi"])   # ROIs in plotting order; anything else (ROI pairs) sorted
    for roi in [r for r in ROI_ORDER if r in present] + sorted(present - set(ROI_ORDER)):
        sub = tbl[tbl["roi"] == roi]
        for b in sorted(sub[x].unique()):
            v = sub.loc[sub[x] == b, y].dropna().to_numpy(float)
            t = p = np.nan
            if v.size >= (min_subjects or MIN_SUBJECTS_ROI) and np.ptp(v) > 0:
                t, p = ttest_1samp(v, 0.0)
            rows.append({"roi": roi, x: b, "n": v.size,
                         "mean": float(np.mean(v)) if v.size else np.nan,
                         "sem": float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan,
                         "ci95": mean_ci(v)[1] if v.size > 1 else np.nan,
                         "t": float(t), "p": float(p)})
    out = pd.DataFrame(rows)
    m = out["p"].notna().to_numpy()
    out["q"] = np.nan
    if m.any():
        out.loc[m, "q"] = multipletests(out.loc[m, "p"].to_numpy(float), method=fdr)[1]
    return out


def lobe_key(fig: Any, y: float = 1.0, fontsize: float = 11) -> None:
    """One line of lobe names, each in its LOBE_COLORS colour, centred at figure height `y`."""
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], ls="") for _ in LOBE_COLORS], [l.capitalize() for l in LOBE_COLORS],
               labelcolor=list(LOBE_COLORS.values()), loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=len(LOBE_COLORS), frameon=False, handlelength=0, handletextpad=0, columnspacing=1.5,
               prop={"size": fontsize, "weight": "bold"})


def roi_curve_figure(stats: pd.DataFrame, x: str, path: str, xlabel: str, ylabel: str) -> str:
    """One small-multiple panel per ROI: mean +/- 95% CI across subjects vs `x`,
    FDR stars per bin (stats from bin_stats). Saves <path>.png."""
    rois = [r for r in ROI_ORDER if r in set(stats["roi"])]
    ncol = 4
    nrow = int(np.ceil(len(rois) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.4 * nrow), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    ylim = np.nanmax(np.abs(stats["mean"].to_numpy(float))) * 1.35 or 1.0
    for ax, roi in zip(axes, rois):
        s = stats[stats["roi"] == roi].sort_values(x)
        xv, y, e = (s[c].to_numpy(float) for c in (x, "mean", "ci95"))
        col = LOBE_COLORS.get(roi.split("-", 1)[1], "C0")
        ax.axhline(0, color="0.6", lw=0.8, zorder=1)
        ax.fill_between(xv, y - e, y + e, alpha=0.25, lw=0, color=col, zorder=2)
        ax.plot(xv, y, lw=1.6, color=col, zorder=3)
        for xi, qi in zip(xv, s["q"].to_numpy(float)):
            if stars(qi):
                ax.text(xi, ylim * 0.80, stars(qi), ha="center", va="center", fontsize=9, zorder=4)
        ax.set_title(f"{pretty_roi(roi)}  (n={int(np.nanmax(s['n']))})", fontsize=9, color=col, fontweight="bold")
        ax.set_ylim(-ylim, ylim)
    for ax in axes[len(rois):]:
        ax.axis("off")
    for ax in axes[-ncol:]:
        ax.set_xlabel(xlabel)
    for r in range(nrow):
        axes[r * ncol].set_ylabel(ylabel)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(f"{path}.png", dpi=160)
    plt.close(fig)
    return f"{path}.png"


# --- session dispatch ---------------------------------------------------------
def load_sess_list(
    root_dir_: str, n_sessions: int | None = None,
    n_subjects: int | None = None,
) -> list[pd.Series]:
    """Included sessions from sess_list_df.json, optionally subset.

    --n-sessions N: the first N rows. --n-subjects K: every session of the first
    K subjects (so a subject is never half-computed).
    """
    df = (pd.read_json(join(root_dir_, "sess_list_df.json"))
            .query("include == True"))
    if n_subjects is not None:
        first_k = df["sub"].drop_duplicates().iloc[:n_subjects]
        df = df[df["sub"].isin(first_k)]
    elif n_sessions is not None:
        df = df.iloc[:n_sessions]
    return [row for _, row in df.iterrows()]


def session_files(d: Any, suffix: str, n_sessions: int | None = None) -> list[Path]:
    """Sorted per-session files `<ftag><suffix>` in d, for sessions INCLUDED in
    root_dir's sess_list_df.json only, so stale files of excluded sessions are
    never aggregated."""
    inc = pd.read_json(join(root_dir, "sess_list_df.json")).query("include == True")
    keep = {f"{r.sub}_{r.exp}_{r.sess}" for r in inc.itertuples()}
    return [f for f in sorted(Path(d).glob(f"*{suffix}")) if f.name[:-len(suffix)] in keep][:n_sessions]


def _sid_str(row: Any) -> str:
    """Label one work item for an error message.

    Items are usually session rows, but the dispatchers also take plain values
    (a file path, an (index, path) tuple) when a script's unit of work is not a
    dfrow -- fall back to str() rather than raising inside the error handler.
    """
    if row is None:
        return "<unknown>"
    try:
        return f"{row['sub']}_{row['exp']}_{row['sess']}"
    except Exception:
        return str(row)


def _run_item(args: tuple[Callable[..., Any], Any, dict[str, Any]]) -> tuple[Any, Any, Exception | None]:
    """One work item in a worker process; the exception is returned, not raised."""
    fn, item, kwargs = args
    try:
        return item, fn(item, **kwargs), None
    except Exception as e:
        return item, None, e


def run_sessions(
    fn: Callable[..., Any], sess_list: Sequence[Any], desc: str,
    workers: int = 1, collect: bool = False, quiet: bool = False, **kwargs: Any,
) -> list[Any]:
    """Run `fn(item, **kwargs)` over `sess_list`, `workers` items at a time.

    workers=1 runs in this process; more spreads the items over that many
    worker processes (`fn` must then be importable, i.e. live in a .py file).
    A failing item is reported and skipped, never fatal. collect=True returns
    each call's result (failures excluded), for callers whose work returns data
    rather than writing a file. quiet=True drops the per-item [ok] line.
    """
    from tqdm.auto import tqdm

    jobs = [(fn, item, kwargs) for item in sess_list]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=min(workers, max(len(jobs), 1)))
        results: Iterable[Any] = pool.map(_run_item, jobs)
    else:
        pool = None
        results = map(_run_item, jobs)

    out: list[Any] = []
    n_ok, n_err = 0, 0
    # Overwritten each run, one capped line per failure: bounded, and never stale.
    log_path = Path(root_dir, "logs", "_".join(desc.lower().split()) + "_errors.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    pbar = tqdm(results, total=len(jobs), desc=f"{desc} ({workers} worker{'s' if workers > 1 else ''})")
    for item, res, err in pbar:
        if err is None:
            n_ok += 1
            if collect:
                out.append(res)
            elif not quiet:
                pbar.write(f"[ok] {res}")
        else:
            n_err += 1
            line = f"sess={_sid_str(item)}: {err!r}"[:500]
            pbar.write(f"[error] {line}")
            log.write(line + "\n")
        pbar.set_postfix(ok=n_ok, err=n_err)
    log.close()
    if pool is not None:
        pool.shutdown()
    print(f"[{desc}] done: ok={n_ok}, err={n_err}" + (f" (errors: {log_path})" if n_err else ""))
    return out


def run_compute_stage(
    fn: Callable[..., str], desc: str, root_dir_: str,
    n_sessions: int | None = None, n_subjects: int | None = None,
    workers: int = 1, **kwargs: Any,
) -> None:
    """Select the sessions and run `fn` over them."""
    sess_list = load_sess_list(root_dir_, n_sessions=n_sessions,
                               n_subjects=n_subjects)
    n_subj = len({r["sub"] for r in sess_list})
    print(f"[stage] {len(sess_list)} sessions ({n_subj} subjects)")
    if workers > 1:
        helper.prefetch_bids(sess_list)
    run_sessions(fn, sess_list, desc, workers=workers, **kwargs)


def add_common_args(p: Any, compute: bool = True) -> Any:
    """CLI flags shared by the build scripts (paths, subsetting, parallelism)."""
    p.set_defaults(beh="word_on")
    p.add_argument("--band", default="alpha",
                   help="band key in config.yaml (default: alpha = 8-13 Hz)")
    p.add_argument("--root-dir", default=None,
                   help="source data dir; defaults to fc.root_dir (SCRATCH_DIR)")
    p.add_argument("--save-root", default=None,
                   help="where per-session pickles live; defaults to --root-dir")
    p.add_argument("--n-sessions", type=int, default=None,
                   help="use only the first N sessions")
    p.add_argument("--simulation-tag", default=None,
                   help="replace the EEG with this config/simulation_config.yaml DGP; "
                        "pickles go to <root-dir>/sim/<tag>")
    if compute:
        p.add_argument("--n-subjects", type=int, default=None,
                       help="compute all sessions of the first K subjects")
        p.add_argument("--workers", type=int, default=1,
                       help="sessions computed at once in separate processes "
                            "(1 = in this process); budget a few GB of RAM each")
    return p


def plot_args(doc: str, out_dir: str, **extra: Any) -> tuple[Any, str]:
    """CLI shared by the plot_*.py scripts -> (args, save_root). `extra`:
    {flag: argparse kwargs} for script-specific options."""
    import argparse
    p = argparse.ArgumentParser(description=doc.splitlines()[0])
    add_common_args(p, compute=False)
    p.add_argument("--out-dir", default=out_dir)
    p.add_argument("--min-electrodes", type=int, default=1,
                   help="min electrodes a subject needs in a region to enter its test")
    for flag, kw in extra.items():
        p.add_argument(flag, **kw)
    args = p.parse_args()
    return args, resolve_roots(args)[1]


def resolve_roots(args: Any) -> tuple[str, str]:
    """(root_dir, save_root) from --root-dir / --save-root; point helper at them."""
    global root_dir, MIN_SUBJECTS_ROI, MIN_SUBJECTS_PAIR
    root_dir = args.root_dir or root_dir
    tag = getattr(args, "simulation_tag", None)   # simulated pickles never share the real cache
    if tag:   # simulation runs use ~20 subjects
        MIN_SUBJECTS_ROI = MIN_SUBJECTS_PAIR = 5
    save_root = getattr(args, "save_root", None) or (join(root_dir, "sim", tag) if tag else root_dir)
    helper.root_dir = root_dir
    print(f"[setup] root_dir  = {root_dir}")
    print(f"[setup] save_root = {save_root}")
    return root_dir, save_root


def subject_roi_means(
    per_elec: pd.DataFrame, measures: Sequence[str], min_electrodes: int = 1,
) -> pd.DataFrame:
    """Per-electrode measures -> per-(subject, ROI) means; the unit of analysis.

    `per_elec` is one row per (sub, label, roi) with a column per measure. ROIs
    a subject covers with fewer than `min_electrodes` contacts are dropped.
    """
    tbl = (per_elec.groupby(["sub", "roi"])
                   .agg(n_elec=("label", "size"),
                        **{m: (m, "mean") for m in measures})
                   .reset_index())
    return tbl[tbl["n_elec"] >= min_electrodes].reset_index(drop=True)


def write_roi_csvs(
    out_dir: str, stem: str, tbl: pd.DataFrame, per_elec: pd.DataFrame,
    stats: dict[str, pd.DataFrame], key: str = "measure",
) -> None:
    """The three CSVs every ROI figure ships with: per-subject, per-electrode,
    and the per-ROI stats for EVERY measure (drawn or not), stacked."""
    os.makedirs(out_dir, exist_ok=True)
    tbl.to_csv(join(out_dir, f"{stem}_per_subject.csv"), index=False)
    per_elec.to_csv(join(out_dir, f"{stem}_per_electrode.csv"), index=False)
    pd.concat([s.assign(**{key: m}) for m, s in stats.items()]).to_csv(
        join(out_dir, f"{stem}_stats.csv"), index=False)
    print(f"[saved] per-subject / per-electrode / stats CSVs in {out_dir}")
