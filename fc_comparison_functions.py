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
from scipy.stats import ttest_1samp, pearsonr  # pyright: ignore[reportMissingTypeStubs]
from scipy.stats import t as tdist  # pyright: ignore[reportMissingTypeStubs]
from statsmodels.stats.multitest import fdrcorrection, multipletests  # pyright: ignore[reportMissingTypeStubs]

import os
import warnings
from os.path import join

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns  # pyright: ignore[reportMissingTypeStubs]

from cstat import *  # noqa: F401,F403
from misc import *  # noqa: F401,F403
from matrix_operations import *  # noqa: F401,F403
from mne_connectivity import (  # pyright: ignore[reportMissingTypeStubs]
    spectral_connectivity_epochs, envelope_correlation)
from mne.time_frequency import psd_array_multitaper
# pactools is imported at function scope inside compute_pac (the only
# caller). The high-level Comodulogram API was removed in favor of
# pactools.bandpass_filter.multiple_band_pass + a direct Ozkurt MI
# matmul; no top-level import is needed.

from project_paths import (
    SCRATCH_DIR as _SCRATCH_DIR,
    MT_BANDWIDTH, GC_N_LAGS, COMPUTATION_METRICS,
    REAL_DATA_BUFFER_MS,
    FC_MODE, CWT_FNUM, CWT_MORLET_REPS, CWT_BUFFER_N_SIGMA, TIME_BIN_MS, MT_WINDOW_MS,
    NOTCH_HARMONICS_UP_TO_HZ,
)

root_dir: str = str(_SCRATCH_DIR)

import helper
from helper import *  # noqa: F401,F403  # pyright: ignore[reportAssignmentType]
helper.root_dir = root_dir

# figure_io is imported lazily inside functions that produce figures.
from figure_io import SaveFigure as _SaveFigure  # noqa: F401

from pathlib import Path
import pickle

NDArrayAny = npt.NDArray[Any]


def load_pickle(path: str) -> Any:
    """Convenience pickle.load wrapper. Shadows misc.load_pickle via star-import."""
    with open(path, "rb") as f:
        return pickle.load(f)


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

    A pre/post contrast (word_on, voc) must compare temporally disjoint real-data
    epochs — unlike en/rm, which split distinct event sets and never share
    samples. Touching at a shared boundary (hi == lo) is allowed.
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


def assert_equal_time_length(*arrays: NDArrayAny, tol: int = 0) -> int:
    """Assert all arrays share the same last-axis (time) length and return it.

    `tol` is the maximum allowed spread (max-min) in samples: tol=1 permits the
    #88 resample grid-phase off-by-one that equalize_time_length resolves, while
    a larger spread signals a genuine windowing bug and raises. Call this to
    validate pre/post epochs are equal-length (tol=0 after equalize_time_length,
    tol=1 on the raw loaded epochs).
    """
    if not arrays:
        raise ValueError("assert_equal_time_length requires at least one array")
    lengths = [int(a.shape[-1]) for a in arrays]
    spread = max(lengths) - min(lengths)
    if spread > tol:
        raise ValueError(
            f"time-axis lengths differ by {spread} > tol {tol}: {lengths}"
        )
    return min(lengths)


def load_results(root_dir: str, beh: str, cond: str, band: str = "low") -> list[Any]:
    d = Path(root_dir) / beh / "fc_mats" / cond / band
    files = sorted(d.glob("*_fc_mats.pkl"))
    return [load_pickle(str(p)) for p in files]

metrics = list(COMPUTATION_METRICS)
# CFG_FLOW_VERIFY: metrics list pinned to the live computation menu (dpli is
# permanently retired — see project_paths.BANNED_METRICS).
assert metrics == ["coh", "plv", "ppc", "ciplv", "pli", "wpli", "aec", "aec_c",
                   "pac", "gc", "gc_tr"], (  # CFG_FLOW_VERIFY
    f"metrics drift: {metrics}"
)
# `asymm` is the symmetrize-decision set: any asymmetric metric is reduced to its
# symmetric part before comparison. `dpli` stays listed as a defensive default
# (it is banned upstream, so it never actually flows) alongside the live `pac`
# and the directional GC family.
asymm = {"dpli", "gc", "gc_tr", "pac"}
#multivar = {"gc", "gc_tr"}
# Genuinely directional measures whose HUB convention is undecided: hub/graph
# measures raise NotImplementedError for these (rather than computing on the
# symmetric part like the other asymmetric metrics) to force a decision when GC
# hubs are enabled.
_DIRECTED_HUB_UNDECIDED = {"gc", "gc_tr"}

from figure_io import METRIC_LABELS as correct_labels  # noqa: E402
# CFG_FLOW_VERIFY: fc_comparison.correct_labels sourced from figure_io.METRIC_LABELS
assert correct_labels["coh"] == "Coh" and correct_labels["gc_tr"] == "GC-TR", (  # CFG_FLOW_VERIFY
    f"correct_labels drift: {correct_labels}"
)

from project_paths import BANDS as bands

# CFG_FLOW_VERIFY: fc_comparison_functions.bands must equal project_paths.BANDS
# Per-key pins: pre-migration had 6 keys {theta, alpha, gamma, low, narrow_5hz,
# narrow_7hz}; all unchanged at unification.
from project_paths import BANDS as _CFG_BANDS  # noqa: E402
assert bands is _CFG_BANDS, "fc.bands drift from project_paths.BANDS"
assert bands["theta"]      == (4.0, 9.0), f"theta drift: {bands['theta']}"           # CFG_FLOW_VERIFY
assert bands["alpha"]      == (8.0, 13.0), f"alpha drift: {bands['alpha']}"          # CFG_FLOW_VERIFY
assert bands["gamma"]      == (70.0, 110.0), f"gamma drift: {bands['gamma']}"        # CFG_FLOW_VERIFY
assert bands["low"]        == (3.0, 8.0), f"low drift: {bands['low']}"              # CFG_FLOW_VERIFY
assert bands["narrow_3hz"] == (2.5, 3.5), f"narrow_3hz drift: {bands['narrow_3hz']}"  # CFG_FLOW_VERIFY
assert bands["narrow_5hz"] == (4.5, 5.5), f"narrow_5hz drift: {bands['narrow_5hz']}"  # CFG_FLOW_VERIFY
assert bands["narrow_7hz"] == (6.5, 7.5), f"narrow_7hz drift: {bands['narrow_7hz']}"  # CFG_FLOW_VERIFY

regionlabels = list(helper.get_region_information("region_labels"))  
reg2i = {r: i for i, r in enumerate(regionlabels)}


def split_reg_full(s: str) -> tuple[str | None, str | None]:
    """Split a 'L hippocampus' / 'R fusiform gyrus' string into (hemi, region).
    Returns (None, None) when the input lacks the hemisphere prefix.
    """
    parts = str(s).split(maxsplit=1)
    return (parts[0], parts[1].strip().lower()) if len(parts) == 2 else (None, None)

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
 

def compute_metric_matrix(
    data: NDArrayAny,
    sfreq: float,
    m: str,
    fmin: float,
    fmax: float,
    gc_n_lags: int,
    mode: str = "multitaper",
    buffer_left_ms: float = 0.0,
    buffer_right_ms: float = 0.0,
    buffer_left_samples: int = 0,
    buffer_right_samples: int = 0,
    cwt_freqs: NDArrayAny | None = None,
    cwt_n_cycles: NDArrayAny | float | None = 5,
) -> NDArrayAny:
    """`mode` is one of {'multitaper', 'cwt_morlet'}. Only phase-based
    metrics (coh, plv, ppc, ciplv, pli, wpli, dpli) honor the mode flag.
    aec/pac/gc/gc_tr ignore it and use their own internal computation.

    `buffer_{left,right}_ms` trim that many milliseconds from the left /
    right of the time axis BEFORE averaging across time. Asymmetric —
    symmetric is the special case left == right. Only relevant for
    time-resolved modes (cwt_morlet); multitaper has no time axis to
    trim. Sample count is derived from buffer_ms * sfreq / 1000 using
    int() truncation to match helper.mirror_buffer's convention.

    `buffer_{left,right}_samples` is the REAL-data edge buffer (samples). For
    AEC/AEC-c/PAC the buffer is kept through the band-pass + Hilbert and cropped
    off the analytic signal before estimation (compute_aec_buffered /
    compute_pac). For every other (spectral / directed) metric the buffer is
    plain real data only needed to anchor the resample/notch edges, so it is
    sliced off BEFORE the spectral estimate. 0/0 reproduces the unbuffered path.
    """
    if m in ("aec", "aec_c"):
        orthogonalize = "pairwise" if m == "aec_c" else False
        return compute_aec_buffered(
            data, sfreq, orthogonalize=orthogonalize,
            buffer_left_samples=buffer_left_samples,
            buffer_right_samples=buffer_right_samples)
    if m == "pac":
        return compute_pac(
            data, sfreq,
            buffer_left_samples=buffer_left_samples,
            buffer_right_samples=buffer_right_samples)
    # Spectral / directed metrics: the real-data buffer only anchors the
    # resample/notch edges; remove it before the (multitaper) estimate.
    if buffer_left_samples or buffer_right_samples:
        nt = data.shape[-1]
        end = nt - buffer_right_samples if buffer_right_samples > 0 else nt
        data = data[..., buffer_left_samples:end]
    if m in asymm:
        try:
            return compute_spectral_fc_directed(
                data, sfreq, method=m, fmin=fmin, fmax=fmax,
                gc_n_lags=gc_n_lags, faverage=True
            )
        except ValueError as e:
            # GC (and gc_tr) enforce n_lags < 2*(n_freqs-1). Sessions sampled
            # at sub-500 Hz rates (e.g. 499.7071 BioSemi) lose one sample
            # under the resample step, dropping the frequency-bin count and
            # tripping this check. Other metrics on the same session are
            # unaffected, so we NaN this metric instead of killing the session.
            if m in ("gc", "gc_tr") and "number of lags" in str(e):
                n_ch = data.shape[1]
                return np.full((n_ch, n_ch), np.nan)
            raise
    return compute_spectral_fc(
        data, sfreq, method=m, fmin=fmin, fmax=fmax, faverage=True,
        mode=mode,
        buffer_left_ms=buffer_left_ms,
        buffer_right_ms=buffer_right_ms,
        cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles,
    )

FC_MODES = ("multitaper", "cwt_morlet")


def band_dirname(band: str, fc_mode: str = "multitaper") -> str:
    """On-disk directory segment for a (band, fc_mode) pair.

    Outputs must not collide across estimators: a Morlet run and a multitaper
    run of the same band/behaviour produce different numbers, and without this
    the second would silently overwrite the first's pickles.

    'multitaper' returns the bare band name, so every EXISTING path and pickle
    stays exactly where it is and nothing needs recomputing or migrating. Only
    the new mode gets a suffix:

        multitaper -> "high_gamma"
        cwt_morlet -> "high_gamma__cwt_morlet"
    """
    if fc_mode not in FC_MODES:
        raise ValueError(f"fc_mode must be one of {FC_MODES}, got {fc_mode!r}")
    return band if fc_mode == "multitaper" else f"{band}__{fc_mode}"


def default_cwt_freqs(
    fmin: float, fmax: float, fnum: int, morlet_reps: int = 5,
) -> NDArrayAny:
    """Log-spaced Morlet centre frequencies, from the project's Wavelet bank.

    Delegates to `wavelet.Wavelet` (Rao's module, the single source of truth for
    the Morlet definition here) so the cwt grid and the simulation/audit code
    cannot drift apart.

    Exists because the MNE default in `compute_spectral_fc` is a 1 Hz arange,
    which is catastrophically oversampled for a wide high-frequency band:
    `np.arange(70, 150.5, 1)` is 81 wavelets, but a 5-rep Morlet at 100 Hz has a
    ~40 Hz half-power bandwidth, so those estimates are near-duplicates costing
    ~13x the compute of the handful that carry independent information.

    Log spacing is the right geometry because Morlet bandwidth is proportional
    to centre frequency — the same reason 70-150 is best read as centred at its
    GEOMETRIC centre, sqrt(70*150) = 102.5 Hz.
    """
    from wavelet import Wavelet  # local: keeps matplotlib off the compute path
    freqs = Wavelet(fmin=fmin, fmax=fmax, fnum=fnum,
                    morlet_reps=morlet_reps).freqs
    return nudge_off_notches(freqs)


def nudge_off_notches(freqs, half_width_hz=2.0, notch_up_to=None):
    """Move any wavelet centre frequency that sits inside a notch stopband.

    A wavelet CENTRED on a notched line frequency measures the hole we dug
    rather than the signal. With a 6-point log grid over 70-150 this is not
    hypothetical: the grid's top point is exactly 150.0 Hz, which is the German
    3rd harmonic, so one sixth of the estimate would be noise floor.

    Each offending centre is shifted to the nearer edge of the stopband (plus a
    hair), which keeps the grid spanning the band while moving the wavelet's
    peak sensitivity off the hole. Its skirts still overlap the notch -- a
    5-cycle Morlet at 150 Hz has a ~30 Hz bandwidth, so a 4 Hz stopband removes
    a small fraction of what it integrates -- but that is unavoidable and is the
    same partial loss multitaper takes.

    No-op when harmonic notching is off (notch_up_to None -> config value).
    """
    up_to = NOTCH_HARMONICS_UP_TO_HZ if notch_up_to is None else notch_up_to
    if not up_to:
        return freqs
    lines = sorted({f * k for f in (50., 60.)
                    for k in range(2, int(up_to // f) + 1)})
    out = np.asarray(freqs, float).copy()
    for i, f in enumerate(out):
        for L in lines:
            if abs(f - L) < half_width_hz:
                # shift to whichever stopband edge is nearer, + 1% margin
                edge = L - half_width_hz if f <= L else L + half_width_hz
                out[i] = edge * (0.99 if f <= L else 1.01)
                break
    return out


def morlet_buffer_ms(fmin: float, morlet_reps: int = 5,
                     n_sigma: float = 5.0) -> float:
    """Edge buffer (ms per side) a Morlet transform needs at its LOWEST frequency.

    Uses `Wavelet.get_morlet_width` — sigma = morlet_reps / (2*pi*f) seconds —
    so the buffer follows the same wavelet definition as everything else here.
    `n_sigma` is how far into the Gaussian tail to keep; 5 is the project's
    existing implied convention, since 5 sigma at 4 Hz with morlet_reps=5 is
    995 ms, i.e. the configured `mirror_buffer_ms: 1000`.

    The frequency dependence is severe and is the reason one fixed constant
    cannot serve every band: 5 sigma is ~995 ms at 4 Hz but only ~57 ms at
    70 Hz. Note this also means the configured `real_data_buffer_ms: 50` is
    marginally short for a 70 Hz lower edge (~4.4 sigma).
    """
    from wavelet import Wavelet  # local: keeps matplotlib off the compute path
    # fmax/fnum are irrelevant to get_morlet_width; only morlet_reps and f are.
    w = Wavelet(fmin=fmin, fmax=fmin * 2, fnum=2, morlet_reps=morlet_reps)
    return 1000.0 * w.get_morlet_width(float(fmin)) * float(n_sigma)


def assert_windows_separable(
    pre_win: tuple[float, float], post_win: tuple[float, float],
    fmin: float, morlet_reps: int = 5, n_sigma: float = 5.0,
) -> float:
    """Check the pre/post gap is wide enough that Morlet support cannot bleed.

    `compute_prepost_separate` loads the pre and post windows as separate
    epochs, each widened by an edge buffer. If the buffered windows meet, the
    same samples inform both arms of the contrast — manufacturing agreement
    between conditions that are supposed to be independent. That is the exact
    bleed-through the buffer exists to prevent, so it raises rather than warns.

    Returns the per-side buffer actually required (ms).
    """
    need = morlet_buffer_ms(fmin, morlet_reps=morlet_reps, n_sigma=n_sigma)
    gap = float(post_win[0]) - float(pre_win[1])
    if 2.0 * need > gap:
        raise ValueError(
            f"Morlet support at fmin={fmin} Hz needs {need:.0f} ms of buffer on "
            f"each side ({2 * need:.0f} ms total), but the pre/post gap is only "
            f"{gap:.0f} ms (pre ends {pre_win[1]}, post starts {post_win[0]}). "
            f"The buffered windows would overlap and bleed signal across the "
            f"contrast. Widen the gap, lower morlet_reps, or raise fmin."
        )
    return need


def compute_spectral_fc(
    data: NDArrayAny,
    sfreq: float,
    method: str,
    fmin: float,
    fmax: float,
    faverage: bool = True,
    mode: str = "multitaper",
    buffer_left_ms: float = 0.0,
    buffer_right_ms: float = 0.0,
    cwt_freqs: NDArrayAny | None = None,
    cwt_n_cycles: NDArrayAny | float | None = 5,
) -> NDArrayAny:
    '''Computes all MNE phase based functions except multivariate ones.

    Supported modes:
      'multitaper'  — current default. Single freq-averaged estimate per
                      channel pair; no time axis. mt_adaptive=False.
      'cwt_morlet'  — Morlet wavelet at `cwt_freqs` (default
                      np.arange(fmin, fmax + 0.5, 1)) with `cwt_n_cycles`
                      cycles (default 5, matching the synthetic
                      generator's morlet_reps). MNE returns time-resolved
                      connectivity. We trim buffer_left_samples /
                      buffer_right_samples from each side of the time
                      axis, then average across time. faverage collapses
                      the freq dim at the end.
    '''
    if mode == "cwt_morlet":
        if cwt_freqs is None:
            # Log-spaced via the project's Wavelet bank, NOT a 1 Hz arange:
            # arange(70, 150.5, 1) is 81 near-duplicate wavelets (see
            # default_cwt_freqs). fnum from config so the choice stays visible.
            cwt_freqs = default_cwt_freqs(
                fmin, fmax, fnum=CWT_FNUM,
                morlet_reps=int(cwt_n_cycles) if cwt_n_cycles else 5)
        cwt_freqs = np.asarray(cwt_freqs, dtype=float)
        kw = {} if cwt_n_cycles is None else {"cwt_n_cycles": cwt_n_cycles}
        con = spectral_connectivity_epochs(
            data, method=method, mode="cwt_morlet",
            sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=faverage,
            cwt_freqs=cwt_freqs,
            n_jobs=1, verbose=False,
            **kw,  # pyright: ignore[reportArgumentType]
        )
        # cwt_morlet output: (n_ch, n_ch, n_freqs, n_times). Trim the
        # buffer region from the time axis BEFORE averaging — this is
        # the whole point of running time-resolved.
        out = np.asarray(con.get_data(output="dense"))
        # Convert ms -> samples with int() truncation (matches
        # helper.mirror_buffer's convention so the trim region exactly
        # matches the buffer region that was added at generation time).
        buf_left_n = int(buffer_left_ms * sfreq / 1000.0)
        buf_right_n = int(buffer_right_ms * sfreq / 1000.0)
        if buf_left_n > 0 or buf_right_n > 0:
            n_t = out.shape[-1]
            if buf_left_n + buf_right_n >= n_t:
                raise ValueError(
                    f"buffer_left_ms={buffer_left_ms} + "
                    f"buffer_right_ms={buffer_right_ms} (= "
                    f"{buf_left_n + buf_right_n} samples) >= n_times={n_t}; "
                    "nothing left after trim."
                )
            end = n_t - buf_right_n if buf_right_n > 0 else n_t
            out = out[..., buf_left_n:end]
        out = out.mean(axis=-1)   # average across time -> (n_ch, n_ch, n_freqs)
        if faverage:
            out = out[..., 0]
    elif mode == "multitaper":
        # CFG_FLOW_VERIFY: mt_bandwidth pinned at decision value (2)
        assert MT_BANDWIDTH == 2.0, f"mt_bandwidth drift: {MT_BANDWIDTH}"  # CFG_FLOW_VERIFY
        con = spectral_connectivity_epochs(
            data, method=method, mode="multitaper",
            sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=faverage,
            mt_adaptive=False, mt_bandwidth=MT_BANDWIDTH, n_jobs=1, verbose=False,
        )
        out = con.get_data(output="dense")
        if faverage:
            out = out[..., 0]
    else:
        raise ValueError(f"mode must be 'multitaper' or 'cwt_morlet', got {mode!r}")
    out = np.asarray(out)
    if out.ndim == 2:
        out = symmetrize_dense(out, diag_value=np.nan)
    return out


def compute_aec(
    data: NDArrayAny,
    orthogonalize: str | bool = "pairwise",
    absolute: bool = False,
) -> NDArrayAny:
    """Amplitude-envelope correlation via mne_connectivity.

    Parameters
    ----------
    data
        Shape (n_epochs, n_channels, n_times) real-valued EEG.
    orthogonalize
        "pairwise" (default, AEC-c) or False (standard AEC).
    absolute
        Whether to take |signed per-epoch r| before symmetrizing and averaging.
        Only used when orthogonalize="pairwise". MNE's default is True; we
        default to False to match Hipp 2012, Brookes 2012, Colclough 2015 —
        all of which use signed r. See code_issues #84.
    """
    conn = envelope_correlation(
        data,
        orthogonalize=orthogonalize,  # pyright: ignore[reportArgumentType]
        absolute=absolute,
        verbose=False,
    )
    out = np.asarray(conn.get_data(output="dense"))
    out = np.squeeze(out)

    if out.ndim == 3:
        # envelope_correlation returns shape (n_epochs, n_ch, n_ch) — one
        # correlation matrix per epoch. Aggregate to a single session-level
        # matrix by averaging across epochs. nanmean (not mean) so an epoch
        # whose row/column is NaN for one channel (e.g. epoch had a flat
        # channel) doesn't poison the cross-channel pairs from other epochs.
        out = np.nanmean(out, axis=0)

    out = symmetrize_dense(out, diag_value=np.nan)

    # code_issues #71: MNE's envelope_correlation occasionally returns
    # |r| > 1 (and sometimes inf) — neither is valid for a correlation.
    # Root cause is flat-epoch contamination (code_issues #80): when an
    # (epoch, ch) segment has var==0, MNE's `data_conj_scaled /= data_mag`
    # divides by zero and the per-epoch corr blows up. Detect at the
    # channel-pair level (most upstream possible), warn with cell coords
    # + values, and NaN them so downstream can see the anomaly. Both
    # finite |r|>1 and infs are caught; existing NaNs are passed through.
    oor_mask = (~np.isnan(out)) & (np.abs(out) > 1.0)
    if oor_mask.any():
        bad_idx = np.argwhere(oor_mask)
        bad_vals = out[oor_mask].copy()
        # Confirm the suspected root cause: scan the input for near-flat
        # (epoch, channel) segments. MNE's data_mag_std[==0]=1 guard catches
        # strict zero, but NOT near-zero std (~1e-12 from integer-rounded
        # signals + float noise), which still drives the divisor near 0 and
        # blows the per-epoch corr up to ~1e8. Threshold 1e-6 is comfortably
        # above the float-noise floor on quantized iEEG; well-behaved
        # channels have std >> 1.
        seg_std = np.asarray(data).std(axis=-1)  # (n_epochs, n_channels)
        flat_threshold = 1e-6
        flat_mask = seg_std < flat_threshold
        n_flat = int(flat_mask.sum())
        flat_msg = ""
        if n_flat > 0:
            flat_idx = np.argwhere(flat_mask).tolist()
            flat_msg = (
                f" Input has {n_flat} near-flat (std<{flat_threshold:g}) "
                f"(epoch, channel) cell(s) — likely root cause. "
                f"flat_cells={flat_idx[:20]}"
                f"{'... (truncated)' if n_flat > 20 else ''}."
            )
        raise ValueError(
            f"compute_aec: {int(oor_mask.sum())} cell(s) with |r| > 1 "
            f"(orthogonalize={orthogonalize!r}). "
            f"cells={bad_idx.tolist()[:20]}"
            f"{'... (truncated)' if len(bad_idx) > 20 else ''} "
            f"vals_minmax=[{float(np.min(bad_vals)):.3g}, {float(np.max(bad_vals)):.3g}]. "
            f"Root cause likely near-flat epochs (code_issues #80)."
            f"{flat_msg}"
            f" To proceed: add these channels to config/excluded_eeg_channels.csv "
            f"(re-run get_excluded_channels.py to refresh from the cohort scan, "
            f"or hand-add via config/manual_excluded_channels.csv which is merged "
            f"into the final exclusion list). compute_aec refuses to silently "
            f"NaN this — explicit exclusion is required."
        )
    return out


def compute_aec_buffered(
    data: NDArrayAny,
    sfreq: float,
    orthogonalize: str | bool = "pairwise",
    absolute: bool = False,
    buffer_left_samples: int = 0,
    buffer_right_samples: int = 0,
) -> NDArrayAny:
    """Gamma AEC with a real-data buffer cropped AFTER the Hilbert envelope.

    `data` is real EEG (n_epochs, n_channels, n_times) covering the analysis
    window PLUS `buffer_left_samples` / `buffer_right_samples` of real adjacent
    samples. We gamma band-pass + Hilbert the FULL buffered window, slice the
    buffer off the analytic signal, then hand the clean interior (complex) to
    `compute_aec` → `mne_connectivity.envelope_correlation`. Because the input
    is complex, envelope_correlation skips its own Hilbert (it cannot crop edge
    samples itself), so the FIR + Hilbert edge transient — which lives in the
    buffer region — is removed before the correlation.

    The Hilbert call (`hilbert(x, N=next_fast_len(n), axis=-1)[..., :n]`) is the
    exact one envelope_correlation runs internally, so
    `buffer_left_samples == buffer_right_samples == 0` reproduces
    `compute_aec(mne.filter.filter_data(data, *bands["gamma"]), ...)` bit-for-bit.
    """
    import mne
    from scipy.signal import hilbert
    from mne.filter import next_fast_len
    # AEC shares the single `gamma` band with PAC's amplitude band.
    gamma_band = bands["gamma"]
    # CFG_FLOW_VERIFY: AEC gamma band tracks config bands.gamma
    assert gamma_band == _CFG_BANDS["gamma"], (  # CFG_FLOW_VERIFY
        f"AEC gamma band drift: {gamma_band} != config={_CFG_BANDS['gamma']}")
    if buffer_left_samples < 0 or buffer_right_samples < 0:
        raise ValueError(
            f"buffer samples must be >= 0, got left={buffer_left_samples} "
            f"right={buffer_right_samples}"
        )
    x = np.asarray(data, dtype=float)
    filt = mne.filter.filter_data(
        x, sfreq=sfreq, l_freq=gamma_band[0], h_freq=gamma_band[1], verbose=False)
    n_times = filt.shape[-1]
    if buffer_left_samples + buffer_right_samples >= n_times:
        raise ValueError(
            f"buffer_left_samples={buffer_left_samples} + "
            f"buffer_right_samples={buffer_right_samples} >= n_times={n_times}; "
            "nothing left after crop."
        )
    # exact replica of envelope_correlation's internal analytic-signal step
    analytic = np.asarray(hilbert(filt, N=next_fast_len(n_times), axis=-1))[..., :n_times]
    end = n_times - buffer_right_samples if buffer_right_samples > 0 else n_times
    analytic = analytic[..., buffer_left_samples:end]
    return compute_aec(analytic, orthogonalize=orthogonalize, absolute=absolute)


# Item 4 separate-load spec: each contrast loads its PRE-type events at the pre
# window and its POST-type events at the post window (shifted, disjoint), each
# widened by the real-data buffer. AEC/AEC-c/PAC keep the buffer through the
# envelope; the spectral/directed metrics drop it then equalize pre/post length.
AMPLITUDE_METRICS = ("aec", "aec_c", "pac")
PREPOST_SPEC: dict[str, dict[str, Any]] = {
    "word_on": {"pre_type": "PRE_WORD", "post_type": "WORD",
                "pre_win": (-700.0, -100.0), "post_win": (0.0, 600.0)},
    "voc": {"pre_type": "PRE_REC_WORD", "post_type": "REC_WORD",
            "pre_win": (-1100.0, -100.0), "post_win": (0.0, 1000.0)},
}


def compute_prepost_separate(
    dfrow: pd.Series, beh: str, events: pd.DataFrame,
    metric_list: Sequence[str], fmin: float, fmax: float, gc_n_lags: int,
    overlap_mask: NDArrayAny | None, real_data_buffer_ms: float,
    simulation_tag: str | None = None,
    load_fn: Any = None,
) -> dict[str, dict[str, NDArrayAny]]:
    """Baseline-vs-active FC contrast from SEPARATE event-locked pre/post loads
    (item 4). The pre-type events (events.attrs['mask'] == False) load at the pre
    window, post-type (mask == True) at the post window, each widened by
    real_data_buffer_ms. The buffer is removed before the spectral/directed
    metrics (pre/post then equalized to one sample count) and kept through the
    band-pass + Hilbert for AEC/AEC-c/PAC (cropped off the analytic). Returns
    {metric: {baseline, succ, diff}} with diff = FC(post) - FC(pre).

    `load_fn(dfrow, events, save, simulation_tag, real_data_buffer_ms, window)`
    defaults to the module-level get_beh_eeg; injectable for testing.

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

    post_eeg, _ = load_fn(dfrow, post_events, save=False, simulation_tag=simulation_tag,
                          real_data_buffer_ms=real_data_buffer_ms, window=post_win)
    pre_eeg, _ = load_fn(dfrow, pre_events, save=False, simulation_tag=simulation_tag,
                         real_data_buffer_ms=real_data_buffer_ms, window=pre_win)
    # coverage guard: the loaded clip must span its analysis window (the buffer
    # extends beyond it, so a short axis means boundary clipping).
    _assert_time_axis_covers(np.asarray(post_eeg.time), post_win[0], post_win[1], f"{beh} post")
    _assert_time_axis_covers(np.asarray(pre_eeg.time), pre_win[0], pre_win[1], f"{beh} pre")
    sf = float(post_eeg.samplerate)
    post = np.asarray(post_eeg.data); pre = np.asarray(pre_eeg.data)
    nL = int(round(real_data_buffer_ms * sf / 1000.0))

    out_m: dict[str, dict[str, NDArrayAny]] = {}
    for m in metric_list:
        if m in AMPLITUDE_METRICS:
            # keep the buffer through band-pass + Hilbert; cropped off the analytic
            C_post = compute_metric_matrix(post, sf, m, fmin, fmax, gc_n_lags=gc_n_lags,
                                           buffer_left_samples=nL, buffer_right_samples=nL)
            C_pre = compute_metric_matrix(pre, sf, m, fmin, fmax, gc_n_lags=gc_n_lags,
                                          buffer_left_samples=nL, buffer_right_samples=nL)
        else:
            # drop the buffer, then equalize pre/post to one sample count so the
            # multitaper frequency grid matches across the contrast.
            post_i = post[..., nL:post.shape[-1] - nL] if nL else post
            pre_i = pre[..., nL:pre.shape[-1] - nL] if nL else pre
            pre_i, post_i = equalize_time_length(pre_i, post_i)
            assert_equal_time_length(pre_i, post_i, tol=0)
            C_post = compute_metric_matrix(post_i, sf, m, fmin, fmax, gc_n_lags=gc_n_lags)
            C_pre = compute_metric_matrix(pre_i, sf, m, fmin, fmax, gc_n_lags=gc_n_lags)
        C_post = apply_overlap_mask(C_post, overlap_mask)
        C_pre = apply_overlap_mask(C_pre, overlap_mask)
        out_m[m] = {"baseline": C_pre, "succ": C_post, "diff": C_post - C_pre}
    return out_m


def compute_session_fc(
    dfrow: pd.Series,
    beh: str,
    band: str = "low",
    metrics: Sequence[str] = ("coh", "plv", "ppc", "ciplv", "pli", "wpli",
                              "aec", "aec_c", "pac", "gc", "gc_tr"),
    do_aec: bool = False,
    simulation_tag: str | None = None,
    overlap_mask: NDArrayAny | None = None,
    eeg: Any = None,
    mask: Any = None,
) -> Any:
    """
    Main pipeline to load EEG + events and compute FC for a session.

    Parameters
    ----------
    eeg, mask : optional
        Pre-loaded TimeSeries (or compatible) and buffer mask. When BOTH are
        supplied (mask may be None if the caller has no buffer mask), the
        internal `get_beh_eeg` call is skipped — saving a redundant PTSA
        read + resample + notch when the caller has already loaded the EEG
        (e.g. `run_sess_fc`). When neither is supplied, behavior is
        unchanged (load EEG internally via `get_beh_eeg`).

        Pass `eeg` alone (without `mask`) is rejected with ValueError — the
        two must be supplied together to avoid silent buffer-mask drops.
    """
    # gc_n_lags from config.yaml. Pre-migration was hardcoded =5. At sfreq=250
    # Hz each lag = 4 ms → 20 ms VAR memory. This is the empirical max allowed
    # by MNE's n_lags < 2*(n_freqs-1) constraint for the 600-ms word_on slice
    # (most constrained window); 1000-ms windows could go up to 11 but we use
    # the same value everywhere so GC magnitudes are comparable across
    # behaviors. Flagged for methodological revisit in code_issues.md TOP.
    gc_n_lags = GC_N_LAGS
    # CFG_FLOW_VERIFY: gc_n_lags pinned at pre-migration value (5)
    assert gc_n_lags == 5, f"gc_n_lags drift: {gc_n_lags}"  # CFG_FLOW_VERIFY

    eeg_passed = eeg is not None  # track caller-supplied vs internally-loaded
    if eeg is None:
        if mask is not None:
            # Defensive: require BOTH or NEITHER. A bare `mask=` with no eeg
            # cannot be honored (we'd reload the EEG and discard the caller's
            # mask), so this is also an error.
            raise ValueError(
                "compute_session_fc: eeg= and mask= must be supplied together"
            )
        events = load_events(dfrow, beh)
        # helper.load_events typed as DataFrame, but legacy callers tolerate None
        # (no events file on disk). Keep the runtime guard.
        if events is None:  # pyright: ignore[reportUnnecessaryComparison]
            return None
        eeg, mask = get_beh_eeg(dfrow, events, save=False, simulation_tag=simulation_tag)
    elif mask is None:
        # Defensive: require BOTH or NEITHER.
        raise ValueError(
            "compute_session_fc: eeg= and mask= must be supplied together"
        )
    sfreq = float(eeg.samplerate)
    data = np.asarray(eeg.data)
    _n_ep, n_ch, _ = data.shape

    if overlap_mask is not None and overlap_mask.shape != (n_ch, n_ch):
        raise ValueError(f"overlap_mask shape {overlap_mask.shape} does not match n_ch={n_ch}")

    # When the caller hands in a pre-loaded eeg, defensively verify it
    # actually belongs to this session: compare its channel count to the
    # pairs.json count for `dfrow`. Catches the mismatched-session case
    # (e.g. caller passing the wrong subject's eeg) — the overlap_mask
    # check above misses this when overlap_mask=None. Skipped under
    # simulation_tag (the EEG comes from a synthetic generator that may
    # legitimately not have a pairs.json yet). Skipped silently when
    # pairs.json can't be loaded (tests can monkeypatch helper.get_pairs
    # to return a matching-channel-count DataFrame, OR make_overlap_mask
    # callers can rely on the overlap_mask shape check above).
    if eeg_passed and simulation_tag is None:
        try:
            n_ch_pairs = len(helper.get_pairs(dfrow))  # pyright: ignore[reportArgumentType]
        except Exception:
            n_ch_pairs = None  # missing pairs.json — fall back to overlap_mask check alone
        if n_ch_pairs is not None and n_ch_pairs != n_ch:
            raise ValueError(
                f"compute_session_fc: pre-loaded eeg has n_ch={n_ch} but "
                f"pairs.json for {dfrow.get('sub', '?')}_"
                f"{dfrow.get('exp', '?')}_sess{dfrow.get('sess', '?')} "
                f"reports {n_ch_pairs} bipolar pairs; eeg may belong to a "
                f"different session."
            )

    fmin, fmax = bands[band]
    # CFG_FLOW_VERIFY: confirm band tuple at compute_session_fc consumption
    assert (fmin, fmax) == _CFG_BANDS[band], (
        f"compute_session_fc bands[{band!r}] drift: "
        f"({fmin},{fmax}) != config={_CFG_BANDS[band]}"
    )

    out = {
        "meta": {
            "sub": dfrow["sub"],
            "exp": dfrow["exp"],
            "sess": dfrow["sess"],
            "beh": beh,
            "band": band,
            "sfreq": sfreq,
            "n_epochs": data.shape[0],
            "n_ch": n_ch,
            "simulation_tag": simulation_tag,
        },
        "metrics": {},
    }

    if beh in PREPOST_SPEC:
        # word_on / voc: separate event-locked pre/post loads (item 4). PRE_WORD/
        # PRE_REC_WORD events load at the pre window, WORD/REC_WORD at the post
        # window, each widened by the real-data buffer (removed before multitaper;
        # kept through the envelope for AEC/AEC-c/PAC). diff = FC(post) - FC(pre).
        # The upfront `eeg` (union load) is used only for the n_ch/meta context.
        ev = load_events(dfrow, beh)
        if ev is None:  # pyright: ignore[reportUnnecessaryComparison]
            return None
        out["metrics"] = compute_prepost_separate(
            dfrow, beh, ev, metrics, fmin, fmax, gc_n_lags, overlap_mask,
            REAL_DATA_BUFFER_MS, simulation_tag=simulation_tag)
        out["meta"]["n_epochs"] = int(np.asarray(ev.attrs["mask"]).sum())
        return out

    if beh == "rm_all":
        t = np.asarray(eeg.time)
        pre_idx = (t >= -1000) & (t <= 0)
        data_pre = data[:, :, pre_idx]

        for m in metrics:
            C_pre = compute_metric_matrix(data_pre, sfreq, m, fmin, fmax, gc_n_lags=gc_n_lags)
            C_pre = apply_overlap_mask(C_pre, overlap_mask)
            out["metrics"][m] = {"succ": C_pre}

        return out

    if beh == "en_all":
        t = np.asarray(eeg.time)
        pre_idx = (t >= 250) & (t <= 1250)
        data_pre = data[:, :, pre_idx]

        for m in metrics:
            C_pre = compute_metric_matrix(data_pre, sfreq, m, fmin, fmax, gc_n_lags=gc_n_lags)
            C_pre = apply_overlap_mask(C_pre, overlap_mask)
            out["metrics"][m] = {"succ": C_pre}

        return out

    data_succ = data[mask]
    data_fail = data[~mask]

    for m in metrics:
        C_succ = compute_metric_matrix(data_succ, sfreq, m, fmin, fmax, gc_n_lags=gc_n_lags)
        C_fail = compute_metric_matrix(data_fail, sfreq, m, fmin, fmax, gc_n_lags=gc_n_lags)

        C_succ = apply_overlap_mask(C_succ, overlap_mask)
        C_fail = apply_overlap_mask(C_fail, overlap_mask)

        out["metrics"][m] = {
            "succ": C_succ,
            "fail": C_fail,
            "diff": C_succ - C_fail,
        }

    return out

def _all_ordered_pairs(n_ch: int) -> tuple[NDArrayAny, NDArrayAny]:
    '''Only for directed methods'''
    seeds, targets = np.where(~np.eye(n_ch, dtype=bool))
    return seeds, targets  

def compute_spectral_fc_directed(
    data: NDArrayAny,
    sfreq: float,
    method: str,
    fmin: float,
    fmax: float,
    gc_n_lags: int,
    faverage: bool = True,
    **kwargs: Any,
) -> NDArrayAny:
    """Directed methods, GC handled as pairwise multivariate"""
    n_ch = data.shape[1]

    if method in ("gc", "gc_tr"):
        seeds = []
        targets = []
        for i in range(n_ch):
            for j in range(n_ch):
                if i == j:
                    continue
                seeds.append(np.array([i], int))
                targets.append(np.array([j], int))
        indices = (np.array(seeds, dtype=object), np.array(targets, dtype=object))

        con = spectral_connectivity_epochs(
            data,
            method=method,
            mode="multitaper",
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=faverage,
            indices=indices,
            mt_adaptive=False,
            mt_bandwidth=MT_BANDWIDTH,
            n_jobs=1,
            verbose=False,
            gc_n_lags=gc_n_lags,
            rank=None,
            **kwargs,
        )
        vals = con.get_data()     
        if faverage:
            vals = vals[:, 0]

        M = np.full((n_ch, n_ch), np.nan, float)
        for k, (seed_grp, target_grp) in enumerate(zip(*con.indices)):
            i = int(seed_grp[0])
            j = int(target_grp[0])
            M[i, j] = vals[k]
        np.fill_diagonal(M, np.nan)
        return M

    seeds, targets = _all_ordered_pairs(n_ch)
    con = spectral_connectivity_epochs(
        data,
        method=method,
        mode="multitaper",
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=faverage,
        indices=(seeds, targets),
        mt_adaptive=False,
        mt_bandwidth=MT_BANDWIDTH,
        n_jobs=1,
        verbose=False,
        **kwargs,
    )
    vals = con.get_data()
    if faverage:
        vals = vals[:, 0]
    M = np.full((n_ch, n_ch), np.nan, float)
    M[seeds, targets] = vals
    np.fill_diagonal(M, np.nan)
    return M

def compute_pac(
    data: NDArrayAny,
    sfreq: float,
    phase_band: tuple[float, float] = bands["low"],
    amp_band: tuple[float, float] = bands["gamma"],
    buffer_left_samples: int = 0,
    buffer_right_samples: int = 0,
) -> NDArrayAny:
    # CFG_FLOW_VERIFY: PAC defaults match config bands.low / bands.gamma
    assert phase_band == _CFG_BANDS["low"], (
        f"compute_pac phase_band drift: {phase_band} != config low={_CFG_BANDS['low']}"
    )
    assert amp_band == _CFG_BANDS["gamma"], (
        f"compute_pac amp_band drift: {amp_band} != config gamma={_CFG_BANDS['gamma']}"
    )
    """Ozkurt phase-amplitude coupling, channel × channel.

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

    np.fill_diagonal(M, np.nan)
    return M.astype(float, copy=False)


def normalize_region_label(x: str) -> str:
    return str(x).replace("\xa0"," ").strip().replace("  "," ")

def electrode_to_region_mean(
    el_mat: NDArrayAny,
    reg_full: Sequence[str] | NDArrayAny,
    reg_order: Sequence[str],
    fill_value: float = np.nan,
) -> NDArrayAny:
    n = el_mat.shape[0]
    assert el_mat.shape == (n, n)
    assert len(reg_full) == n

    reg2i_local = {normalize_region_label(r): i for i, r in enumerate(reg_order)}
    idx = np.array([reg2i_local.get(normalize_region_label(r), -1) for r in reg_full], int)
    keep = idx >= 0

    R = len(reg_order)
    out = np.full((R, R), fill_value, dtype=float)

    # No mapped electrodes → all blocks are empty → return fill_value grid.
    if not np.any(keep):
        return out

    # Restrict to mapped electrodes; unmapped rows/cols feed no region.
    kept_idx = np.where(keep)[0]
    sub = el_mat[np.ix_(kept_idx, kept_idx)]
    sub_regions = idx[kept_idx]  # region index for each kept electrode

    # Scatter-add finite entries into (R × R) sum and count grids,
    # keyed by (region_of_row, region_of_col).
    finite = np.isfinite(sub)
    row_reg = np.broadcast_to(sub_regions[:, None], sub.shape)
    col_reg = np.broadcast_to(sub_regions[None, :], sub.shape)

    sums = np.zeros((R, R), dtype=float)
    counts = np.zeros((R, R), dtype=np.int64)
    if finite.any():
        flat_mask = finite.ravel()
        flat_vals = sub.ravel()[flat_mask]
        flat_rr = row_reg.ravel()[flat_mask]
        flat_cc = col_reg.ravel()[flat_mask]
        np.add.at(sums, (flat_rr, flat_cc), flat_vals)
        np.add.at(counts, (flat_rr, flat_cc), 1)

    # Mirror original NaN semantics:
    #   - Block is "empty" (→ fill_value) iff one of the regions has no
    #     mapped electrodes at all.
    #   - Block exists but all entries were NaN (count == 0) → NaN.
    region_present = np.zeros(R, dtype=bool)
    region_present[sub_regions] = True
    block_exists = region_present[:, None] & region_present[None, :]

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    out = np.where(block_exists, means, fill_value)

    return out

def pool_connection_electrode_mean(
    el_mat: NDArrayAny,
    reg_full: Sequence[str] | NDArrayAny,
    region_pairs: Sequence[tuple[str, str]],
    reg_order: Sequence[str],
) -> float:
    """Electrode-weighted mean of one (pooled) case-study connection.

    Pools every finite electrode-pair entry whose two electrodes fall in the
    ordered region-pairs given by `region_pairs`, then takes a single nanmean
    over the union — so each contributing electrode pair carries equal weight,
    *as if all the pooled region-pairs came from one unified region pair*. This
    differs from averaging the region-level FC-matrix entries (which would
    weight each region-pair equally regardless of how many electrode pairs back
    it), which is why the case-study pooled values are re-derived here from the
    raw channel-level matrix rather than from `electrode_to_region_mean`'s
    output.

    Each region-pair (A, B) is read in the canonical orientation used by
    `electrode_to_region_mean` + `vec_upper`: rows from the region with the
    smaller `reg_order` index, columns from the larger. For a single
    region-pair this reproduces `electrode_to_region_mean(...)[lo, hi]` exactly
    (the basis of the single-edge identity test); for symmetric metrics the
    orientation is immaterial.

    `el_mat` must already carry the same preprocessing as in
    `build_subj_region_mats` (symmetric metrics symmetrized, asymmetric metrics
    with a NaN diagonal). Returns NaN if no region-pair contributes a finite
    entry (e.g. a region absent from this session's montage).
    """
    el_mat = np.asarray(el_mat, float)
    n = el_mat.shape[0]
    assert el_mat.shape == (n, n)
    assert len(reg_full) == n

    reg2i_local = {normalize_region_label(r): i for i, r in enumerate(reg_order)}
    elec_reg = np.array(
        [reg2i_local.get(normalize_region_label(r), -1) for r in reg_full], int
    )

    chunks: list[NDArrayAny] = []
    for ra, rb in region_pairs:
        ia = reg2i_local.get(normalize_region_label(ra), -1)
        ib = reg2i_local.get(normalize_region_label(rb), -1)
        if ia < 0 or ib < 0:
            missing = [r for r, i in ((ra, ia), (rb, ib)) if i < 0]
            raise ValueError(f"region(s) not in reg_order: {missing}")
        lo, hi = (ia, ib) if ia <= ib else (ib, ia)
        rows = np.where(elec_reg == lo)[0]
        cols = np.where(elec_reg == hi)[0]
        if rows.size and cols.size:
            block = el_mat[np.ix_(rows, cols)]
            chunks.append(block[np.isfinite(block)].ravel())

    if not chunks:
        return float("nan")
    pooled = np.concatenate(chunks)
    if pooled.size == 0:
        return float("nan")
    return float(np.mean(pooled))


def run_sess_fc(
    dfrow: pd.Series,
    save_dir: str,
    beh: str,
    regionlabels: Sequence[str],
    cond: str,
    band: str = "low",
    simulation_tag: str | None = None,
    sess_metrics: Sequence[str] | None = None,
) -> Any:
    """Runs FC computation on one session.

    Parameters
    ----------
    simulation_tag : str or None
        If set, EEG is replaced with simulated EEG matching the named parameter
        set in `simulate_eeg.simulation_parameters`. Passed through to
        `helper.get_beh_eeg` and `compute_session_fc`.
    sess_metrics : iterable[str] or None
        Subset of metrics to compute. Defaults to the module-level `metrics`
        tuple (all 12). Only the requested metrics are written to the output
        pickle; consumers handle missing keys via the `m in subj_mat` guard.
    """
    import helper
    helper.root_dir = root_dir

    sess_metrics = tuple(sess_metrics) if sess_metrics is not None else metrics

    pairs = helper.get_pairs(dfrow)
    # Caller invariant: pairs is non-None here. The original code crashed
    # with TypeError on len(None) if not — pragma preserves that behavior.
    n_ch_pairs = len(pairs)  # pyright: ignore[reportArgumentType]

    # Load EEG once for both the channel-count invariant check and the
    # downstream compute_session_fc pass. Pre-refactor, compute_session_fc
    # re-loaded internally; now we hand the pre-loaded eeg+mask down.
    events = load_events(dfrow, beh)
    if events is None:  # pyright: ignore[reportUnnecessaryComparison]
        return None
    mat_eeg, buffer_mask = get_beh_eeg(dfrow, events, save=False,
                                       simulation_tag=simulation_tag)
    n_ch_eeg = np.asarray(mat_eeg.data).shape[1]

    if n_ch_pairs != n_ch_eeg:
        raise ValueError(f"len(pairs)={n_ch_pairs} but eeg has n_ch={n_ch_eeg}; "
                         "need channel order alignment for overlap mask.")

    overlap_mask = make_overlap_mask(pairs, n_ch=n_ch_eeg)  # pyright: ignore[reportArgumentType]

    mat = compute_session_fc(dfrow, beh=beh, band=band,
                             metrics=sess_metrics, simulation_tag=simulation_tag,
                             overlap_mask=overlap_mask,
                             eeg=mat_eeg, mask=buffer_mask)
    if mat is None:
        return None

    os.makedirs(join(save_dir, "fc_mats", cond, band), exist_ok=True)
    out_file = join(save_dir, "fc_mats", cond, band, f"{ftag(dfrow)}_fc_mats.pkl")

    if os.path.exists(out_file):
        # Recompute if the cached pickle predates the requested metric set
        # (covers cache from a smaller-metric run). A complete cache is one
        # whose keys cover every requested metric.
        try:
            cached = load_pickle(out_file)
            if all(m in cached for m in sess_metrics):
                return
        except Exception:
            pass  # corrupt → fall through and recompute

    reg_full = helper.regionalize_electrodes_by_type(pairs)

    out: dict[str, Any] = {
        "sid": (dfrow["sub"], dfrow["exp"], int(dfrow["sess"])),  # pyright: ignore[reportArgumentType]
        "reg_full": reg_full,
    }
    for m in sess_metrics:
        if m in mat["metrics"] and cond in mat["metrics"][m]:
            out[m] = np.squeeze(mat["metrics"][m][cond])
    save_pickle(out_file, out)
    return out

def session_phase(dfrow, pre_win, post_win, buffer_ms, freqs, morlet_width):
    ev = load_events(dfrow, "word_on")
    words = ev[np.asarray(ev.attrs["mask"], bool)].reset_index(drop=True)
    words.attrs = dict(ev.attrs)
    eeg, _ = get_beh_eeg(dfrow, words, save=False,
                            window=(pre_win[0] - buffer_ms, post_win[1] + buffer_ms))
    wf = MorletWaveletFilter(freqs=freqs, width=morlet_width, output="phase", complete=True)
    ph = wf.filter(timeseries=eeg).transpose("event", "channel", "frequency", "time")
    return np.asarray(ph), np.asarray(eeg.time, float)

def event_angles(phase, t, seed, tg, win):
    m  = (t >= win[0]) & (t <= win[1])
    z  = np.exp(1j * phase[:, :, :, m])
    pd_ = z[:, [seed]] * np.conj(z[:, tg])
    return np.angle(pd_.mean((2, 3)).mean(1))

def pairwise_corr_nan(x: NDArrayAny, y: NDArrayAny) -> float:
    '''Pairwise correlation between 2 vectors'''
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return pearsonr(x[mask], y[mask]).statistic

def vec_upper(A: NDArrayAny) -> NDArrayAny:
    '''Vectorizes upper triangle of matrix'''
    A = np.asarray(A)
    iu = np.triu_indices(A.shape[0], 1)
    return A[iu]

def vec_upper_indices(n: int) -> tuple[NDArrayAny, NDArrayAny]:
    """Return the (i,j) indices for the upper triangle of a matrix."""
    return np.triu_indices(n, 1)

def symmetric_part(M: NDArrayAny) -> NDArrayAny:
    """Symmetric part (M + M.T)/2 of a square matrix.

    For a directed/asymmetric FC matrix this collapses the two directed
    edges of each region pair to their average — the only component of a
    directed measure a symmetric measure can express (the antisymmetric
    part (M - M.T)/2 is orthogonal to every symmetric matrix). Use this for
    any comparison BETWEEN a symmetric and an asymmetric measure, so the
    result does not depend on the arbitrary region ordering. A symmetric
    input is returned unchanged (up to floating error); region matrices
    share coverage across the two directions, so a NaN pair stays NaN.
    """
    M = np.asarray(M, float)
    return (M + M.T) / 2.0

def maybe_symmetrize(M: NDArrayAny, metric: str) -> NDArrayAny:
    """`symmetric_part(M)` for an asymmetric metric, else `M` unchanged.

    The project compares ALL FC measures in symmetric-part space: a symmetric
    measure cannot express the antisymmetric (directional) component of a
    directed measure, and the raw upper triangle of an asymmetric matrix
    depends on the arbitrary region ordering. Using the symmetric part for
    asymmetric metrics everywhere (cross- AND within-measure) is order-
    invariant and avoids the variance drop of full off-diagonal correlations.
    Directed (asymmetric) matrices are otherwise left intact on disk for
    future directional analyses.
    """
    return symmetric_part(M) if metric in asymm else np.asarray(M, float)

def maybe_symmetrize_nested(
    nested: dict[Any, dict[Any, NDArrayAny]], metric: str,
) -> dict[Any, dict[Any, NDArrayAny]]:
    """Apply `symmetric_part` to every matrix in a two-level `{outer: {inner:
    R x R}}` dict when `metric` is asymmetric; return `nested` unchanged
    otherwise. Lets per-metric within-subject analyses (split-half, test-retest,
    ICC) consume directed metrics in symmetric-part space without changing the
    inner functions' signatures."""
    if metric not in asymm:
        return nested
    return {ok: {ik: symmetric_part(iv) for ik, iv in inner.items()}
            for ok, inner in nested.items()}

def vec_upper_metric(M: NDArrayAny, metric: str) -> NDArrayAny:
    """Upper-triangle connection vector, taking the symmetric part first for
    asymmetric metrics (see maybe_symmetrize). The single metric-aware
    vectorizer for per-metric connection-level analyses."""
    return vec_upper(maybe_symmetrize(M, metric))

def vec_offdiag(A: NDArrayAny) -> NDArrayAny:
    """Flatten all off-diagonal entries (both triangles), row-major.

    Length R*(R-1). For a directed matrix this keeps BOTH directions of
    every region pair, so correlating two matrices' `vec_offdiag` is
    order-invariant (the set of ordered pairs is fixed) and uses all
    directed edges — the non-arbitrary WITHIN-measure comparison for
    asymmetric metrics. For a symmetric matrix each pair value appears
    twice; the correlation is unchanged vs the upper triangle.
    """
    A = np.asarray(A)
    return A[~np.eye(A.shape[0], dtype=bool)]

def vec_offdiag_indices(n: int) -> tuple[NDArrayAny, NDArrayAny]:
    """(row, col) indices of all off-diagonal entries, matching `vec_offdiag`."""
    ii, jj = np.where(~np.eye(n, dtype=bool))
    return ii, jj

def corr_nan(a: NDArrayAny, b: NDArrayAny, min_n: int = 10) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < min_n:
        return np.nan
    return pearsonr(a[mask], b[mask]).statistic

def summarize_matrix(M: NDArrayAny, metric_name: str) -> float:
    """Mean of off-diagonal (asymmetric metrics) or upper-tri (symmetric) entries."""
    M = np.asarray(M)
    if metric_name in asymm:
        mask = ~np.eye(M.shape[0], dtype=bool)
        v = M[mask]
    else:
        iu = np.triu_indices(M.shape[0], 1)
        v = M[iu]
    return float(np.nanmean(v))

def build_subj_region_mats(
    results: Sequence[Any],
    beh: str,
    metrics: Sequence[str],
    asymm: Sequence[str] | set[str],
    regionlabels: Sequence[str],
    out_pkl: str | None = None,
) -> tuple[dict[str, dict[str, NDArrayAny]], list[str]]:
    by_sub = {m: {} for m in metrics}

    for r in results:
        if r is None:
            continue

        sub = r["sid"][0]
        reg_full = r["reg_full"]

        for m in metrics:
            if m not in r:
                # Per-session pickle predates the current metrics list (e.g.
                # cached from a smaller-metric run). Honor the docstring
                # promise of silent skip; caller treats the metric as absent
                # from this session.
                continue
            el = np.asarray(r[m])

            if m in asymm:
                el = el.copy()
                np.fill_diagonal(el, np.nan)
            else:
                # mne outputs are lower triangular by default for non-directed methods; symmetrize to get the 
                # full matrix. Diagonal is set to nan since it doesn't represent meaningful connectivity and 
                # can interfere with some analyses (e.g. bootstrapping).
                el = symmetrize_dense(el, diag_value=np.nan)

            reg_mat = electrode_to_region_mean(el, reg_full, reg_order=regionlabels)

            # Code-issues #62: directed metrics (PAC, GC, GC-TR, dPLI) have
            # no meaningful self-region value at the region level. The
            # electrode-pair diagonal was already NaN above; aggregating
            # within-region electrode pairs into a single value here would
            # invent a number that the metric's mathematical definition
            # doesn't support. Drop it.
            if m in asymm:
                np.fill_diagonal(reg_mat, np.nan)

            # code_issues #71: AEC (and AEC-c) sometimes return |r| > 1 from
            # MNE — mathematically impossible for a correlation. Replace
            # offenders with NaN so downstream can see the anomaly, and warn
            # with the full sid + cell coords + values for later diagnosis.
            if m in ("aec", "aec_c"):
                out_of_range = np.isfinite(reg_mat) & (np.abs(reg_mat) > 1.0)
                n_invalid = int(out_of_range.sum())
                if n_invalid > 0:
                    bad_idx = np.argwhere(out_of_range)
                    bad_vals = reg_mat[out_of_range]
                    warnings.warn(
                        f"[{m}] {n_invalid} out-of-range region-pair value(s) "
                        f"for sid={tuple(r['sid'])}: "
                        f"cells={bad_idx.tolist()} vals={bad_vals.tolist()}",
                        stacklevel=2,
                    )
                    reg_mat = np.where(out_of_range, np.nan, reg_mat)

            by_sub[m].setdefault(sub, []).append(reg_mat)

    subj_mat = {m: {} for m in metrics}
    for m in metrics:
        for sub, mats in by_sub[m].items():
            arr = np.stack(mats, axis=0)
            subj_mat[m][sub] = np.nanmean(arr, axis=0)

    if out_pkl is None:
        out_pkl = f"subj_mat_{beh}.pkl"
    pd.to_pickle(subj_mat, out_pkl)

    subs_common = sorted(set.intersection(*[set(subj_mat[m].keys()) for m in metrics]))
    return subj_mat, subs_common


def _prep_session_metric_matrix(
    el: NDArrayAny, m: str, asymm: Sequence[str] | set[str],
) -> NDArrayAny:
    """Channel-level preprocessing shared by region aggregation and pooling.

    Mirrors the per-metric branch in `build_subj_region_mats`: asymmetric
    metrics keep their orientation with a NaN diagonal; symmetric metrics are
    symmetrized (mne lower-triangular -> full) with a NaN diagonal. AEC/AEC-c
    channel entries with |r| > 1 (the MNE bug behind code_issues #71) are
    dropped to NaN so they neither feed a region mean nor a pooled mean.
    """
    el = np.asarray(el, float)
    if m in asymm:
        el = el.copy()
        np.fill_diagonal(el, np.nan)
    else:
        el = symmetrize_dense(el, diag_value=np.nan)
    if m in ("aec", "aec_c"):
        bad = np.isfinite(el) & (np.abs(el) > 1.0)
        if bad.any():
            el = np.where(bad, np.nan, el)
    return el


def build_subj_pooled_connections(
    results: Sequence[Any],
    cases: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    metrics: Sequence[str],
    asymm: Sequence[str] | set[str],
    regionlabels: Sequence[str],
    out_pkl: str | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-subject pooled case-study connection values, re-averaged from the
    raw channel-level FC matrices.

    `cases` is a sequence of ``(case_name, region_pairs)``; ``region_pairs`` is
    the list of ordered (region_a, region_b) pairs pooled into that case (one
    pair => a plain single edge, several => a pooled connection). For each
    session and metric the channel matrix is preprocessed exactly as in
    `build_subj_region_mats`, then `pool_connection_electrode_mean` reduces each
    case to a single electrode-weighted scalar. Per subject these are averaged
    across sessions with `np.nanmean` (matching the region-level pipeline).

    Returns ``pooled[case_name][metric][subject] = float`` and (if `out_pkl`)
    pickles the same dict. A single-edge case reproduces the legacy region-level
    extraction value for that connection exactly (see tests).
    """
    by_sub: dict[str, dict[str, dict[str, list[float]]]] = {
        name: {m: {} for m in metrics} for name, _ in cases
    }
    for r in results:
        if r is None:
            continue
        sub = r["sid"][0]
        reg_full = r["reg_full"]
        for m in metrics:
            if m not in r:
                continue
            el = _prep_session_metric_matrix(np.asarray(r[m]), m, asymm)
            # Asymmetric metrics pooled via their symmetric part — matches the
            # region-level path (prep_regpair_vectors), and pooling the
            # channel-level (el+el.T)/2 equals the symmetric part of the pooled
            # directed means (order-invariant).
            el = maybe_symmetrize(el, m)
            for name, region_pairs in cases:
                val = pool_connection_electrode_mean(
                    el, reg_full, list(region_pairs), regionlabels,
                )
                by_sub[name][m].setdefault(sub, []).append(float(val))

    pooled: dict[str, dict[str, dict[str, float]]] = {
        name: {m: {} for m in metrics} for name, _ in cases
    }
    for name, _ in cases:
        for m in metrics:
            for sub, vals in by_sub[name][m].items():
                arr = np.asarray(vals, float)
                pooled[name][m][sub] = (
                    float(np.nanmean(arr)) if np.any(np.isfinite(arr))
                    else float("nan")
                )

    if out_pkl is not None:
        pd.to_pickle(pooled, out_pkl)
    return pooled


def build_session_region_mats(
    results: Sequence[Any],
    metrics: Sequence[str],
    regionlabels: Sequence[str],
) -> Any:
    sess_mat = {m: {} for m in metrics}
    # Tolerate per-session results that lack some metrics (legacy 8-metric
    # cache, smokescreen metric subsets). See code_issues #30.
    _warned_missing = set()

    for r in results:
        if r is None:
            continue

        subj, _exp, sess = r["sid"]
        reg_full = r["reg_full"]

        for m in metrics:
            if m not in r:
                if m not in _warned_missing:
                    print(f"[build_session_region_mats] metric {m!r} absent "
                          f"from per-session results; skipping")
                    _warned_missing.add(m)
                continue
            el = np.asarray(r[m])
            if m in asymm:
                el = el.copy()
                np.fill_diagonal(el, np.nan)
            else:
                el = symmetrize_dense(el, diag_value=np.nan)

            reg_mat = electrode_to_region_mean(el, reg_full, reg_order=regionlabels)
            # Code-issues #62: directed metrics — drop region diagonal.
            if m in asymm:
                np.fill_diagonal(reg_mat, np.nan)
            sess_mat[m].setdefault(subj, {})[sess] = reg_mat

    # Drop metrics that ended up with zero subjects (truly absent everywhere).
    sess_mat = {m: v for m, v in sess_mat.items() if v}
    return sess_mat

def build_session_channel_mats(
    results: Iterable[Any],
    metrics: Sequence[str],
) -> dict[str, dict[str, dict[Any, NDArrayAny]]]:
    """Channel-level analog of `build_session_region_mats`.

    Returns ``sess_mat[m][subj][sess]`` electrode x electrode matrices, skipping
    the `electrode_to_region_mean` aggregation so every electrode (bipolar pair)
    is retained. Symmetrization / diagonal-NaN handling is identical to the
    region path, so the same downstream test-retest code runs unchanged on the
    finer node set. Within a subject the montage (electrode order) is fixed
    across sessions, so the per-session matrices share shape and node order.
    """
    sess_mat: dict[str, dict[str, dict[Any, NDArrayAny]]] = {m: {} for m in metrics}
    _warned_missing: set[str] = set()

    for r in results:
        if r is None:
            continue

        subj, _exp, sess = r["sid"]

        for m in metrics:
            if m not in r:
                if m not in _warned_missing:
                    print(f"[build_session_channel_mats] metric {m!r} absent "
                          f"from per-session results; skipping")
                    _warned_missing.add(m)
                continue
            el = np.asarray(r[m])
            if m in asymm:
                el = el.copy()
                np.fill_diagonal(el, np.nan)
            else:
                el = symmetrize_dense(el, diag_value=np.nan)
            sess_mat[m].setdefault(subj, {})[sess] = el

    sess_mat = {m: v for m, v in sess_mat.items() if v}
    return sess_mat

# -------------------bootstrap functions--------------------------------------------------------------

def get_connections_w_min_subjects(
    subj_mat: dict[str, dict[str, NDArrayAny]],
    subjects_all: Sequence[str],
    metrics: Sequence[str],
    Smin: int,
) -> tuple[NDArrayAny, NDArrayAny]:
    """Boolean mask + count of connections with finite values from >= Smin
    subjects across all requested metrics. Asymmetric metrics are reduced to
    their symmetric part first (see _symmetrize_asymmetric_metrics), so every
    analysis built on this mask compares in the same order-invariant space."""
    subj_mat = _symmetrize_asymmetric_metrics(subj_mat, metrics)
    V_full_by_m = []
    for m in metrics:
        X = np.stack([subj_mat[m][s] for s in subjects_all], axis=0)
        V = np.stack([vec_upper(X[i]) for i in range(X.shape[0])], axis=0)
        V_full_by_m.append(V)

    # get boolean mask of connections with finite values across all metrics
    finite_all_full = np.logical_and.reduce([np.isfinite(V) for V in V_full_by_m])
    # filter for connections with at least Smin subjects
    counts_full = finite_all_full.sum(axis=0)
    ok_use_full = counts_full >= int(Smin)
    return ok_use_full, counts_full

def prep_regpair_vectors(
    subj_mat: dict[str, dict[str, NDArrayAny]],
    subjects: Sequence[str],
    metrics: Sequence[str],
    ok_pairs_full: NDArrayAny,
) -> dict[str, NDArrayAny]:
    # Asymmetric metrics compared via their symmetric part (order-invariant).
    subj_mat = _symmetrize_asymmetric_metrics(subj_mat, metrics)
    V_by_m = {}
    for m in metrics:
        X = np.stack([subj_mat[m][s] for s in subjects], axis=0)
        V = np.stack([vec_upper(X[i]) for i in range(X.shape[0])], axis=0)
        V_by_m[m] = V[:, ok_pairs_full]
    return V_by_m

def bootstrapped_pair_means(
    subj_mat: dict[str, dict[str, NDArrayAny]],
    subjects_all: Sequence[str],
    metrics: Sequence[str],
    ok_use: NDArrayAny | None,
    N: int = 20,
    n_resamples: int = 2000,
    rng_seed: int = 0,
) -> tuple[dict[str, NDArrayAny], dict[str, NDArrayAny]]:
    """Per-connection paired bootstrap of per-metric means.

    For each retained connection rp and each bootstrap iteration b, draws
    a SINGLE set of N subjects (with replacement) from the per-connection
    pool — the subjects finite at rp across both the upper and lower
    triangle of EVERY metric — and uses those same N picks to compute the
    per-metric per-connection mean. This preserves the within-subject
    cross-metric coupling that downstream measurecorr computations
    (`compute_measurecorrs_bootstrap`) rely on.

    Pre-2026-06 the picks were drawn independently for each (metric,
    connection, iteration) triple. That attenuated cross-metric Pearson
    by a factor of ~N/S (see code_issues #72). Switching to per-
    connection paired picks matches the thesis description (juee_thesis
    §5.1 "N subjects were drawn with replacement from this pool").
    """
    rng = np.random.default_rng(rng_seed)
    if not metrics:
        return {}, {}

    # Asymmetric metrics enter via their symmetric part, so upper == lower
    # triangle below (the directional component is intentionally dropped —
    # symmetric measures cannot express it). The two triangles are kept for
    # structural symmetry with the heatmap fill; they are identical here.
    subj_mat = _symmetrize_asymmetric_metrics(subj_mat, metrics)
    X_by_m = {m: np.stack([subj_mat[m][s] for s in subjects_all], axis=0)
              for m in metrics}
    sample = next(iter(X_by_m.values()))
    S, R, _ = sample.shape
    iu = vec_upper_indices(R)
    P_full = len(iu[0])

    V_u_by_m = {m: np.stack([X_by_m[m][i][iu] for i in range(S)], axis=0)
                for m in metrics}
    V_l_by_m = {m: np.stack([X_by_m[m][i][(iu[1], iu[0])] for i in range(S)], 0)
                for m in metrics}

    if ok_use is None:
        ok_use_edges = np.ones(P_full, dtype=bool)
    else:
        ok_use_edges = np.asarray(ok_use, bool)
        if ok_use_edges.shape[0] != P_full:
            raise ValueError(
                f"ok_use length {ok_use_edges.shape[0]} != number of edges {P_full}"
            )

    pair_means_u = {m: np.full((n_resamples, P_full), np.nan, float) for m in metrics}
    pair_means_l = {m: np.full((n_resamples, P_full), np.nan, float) for m in metrics}

    for rp in range(P_full):
        if not ok_use_edges[rp]:
            continue
        finite_all = np.ones(S, dtype=bool)
        for m in metrics:
            finite_all &= np.isfinite(V_u_by_m[m][:, rp])
            finite_all &= np.isfinite(V_l_by_m[m][:, rp])
        pool = np.where(finite_all)[0]
        if pool.size < N:
            continue
        picks = rng.choice(pool, size=(n_resamples, N), replace=True)
        for m in metrics:
            pair_means_u[m][:, rp] = V_u_by_m[m][picks, rp].mean(axis=1)
            pair_means_l[m][:, rp] = V_l_by_m[m][picks, rp].mean(axis=1)

    boot_means_upper = {m: pair_means_u[m][:, ok_use_edges] for m in metrics}
    boot_means_lower = {m: pair_means_l[m][:, ok_use_edges] for m in metrics}
    return boot_means_upper, boot_means_lower

def bootstrapped_pair_means_2beh(
    V1: dict[str, NDArrayAny],
    V2: dict[str, NDArrayAny],
    metrics: Sequence[str],
    N: int = 20,
    n_resamples: int = 2000,
    rng_seed: int = 0,
) -> tuple[dict[str, NDArrayAny], dict[str, NDArrayAny]]:
    """Per-connection paired bootstrap of per-metric per-behavior means.

    Same shape semantics as `bootstrapped_pair_means` but with two
    behavior vectors per metric (V1, V2). Per connection and iteration,
    a single set of N subject picks is drawn from the per-connection
    pool — the subjects finite at that connection across BOTH behaviors
    AND ALL metrics — and those same picks compute the per-metric per-
    behavior mean for every metric. This preserves cross-metric, cross-
    behavior subject coupling in the Fig 7 cross-contrast measurecorr
    (see code_issues #72).

    Pre-2026-06-04 the picks were drawn independently per (metric,
    connection, iteration), which attenuated cross-metric cells in the
    cross-contrast heatmap for the same N/S reason as the within-
    contrast case.
    """
    rng = np.random.default_rng(rng_seed)
    if not metrics:
        return {}, {}
    sample = next(iter(V1.values()))
    S, P_use = sample.shape

    b1 = {m: np.full((n_resamples, P_use), np.nan, float) for m in metrics}
    b2 = {m: np.full((n_resamples, P_use), np.nan, float) for m in metrics}

    for rp in range(P_use):
        finite_all = np.ones(S, dtype=bool)
        for m in metrics:
            finite_all &= np.isfinite(V1[m][:, rp])
            finite_all &= np.isfinite(V2[m][:, rp])
        pool = np.where(finite_all)[0]
        if pool.size < N:
            continue
        picks = rng.choice(pool, size=(n_resamples, N), replace=True)
        for m in metrics:
            b1[m][:, rp] = V1[m][picks, rp].mean(axis=1)
            b2[m][:, rp] = V2[m][picks, rp].mean(axis=1)

    return b1, b2


def _symmetrize_asymmetric_metrics(
    subj_mat: dict[str, dict[str, NDArrayAny]], metrics: Sequence[str],
) -> dict[str, dict[str, NDArrayAny]]:
    """Copy of `subj_mat` (over `metrics`) with asymmetric-metric matrices
    replaced by their symmetric part (M+M.T)/2; symmetric metrics untouched.

    Cross-measure correlations compare in symmetric-part space: a symmetric
    measure cannot express the antisymmetric (directional) component of a
    directed measure, and the raw upper triangle of an asymmetric matrix depends
    on the arbitrary region ordering (see symmetric_part). Symmetric metrics are
    returned by reference (byte-identical), so only directed measures change.
    The raw directed matrices on disk are untouched — directional analyses can
    still use them.
    """
    return {
        m: ({s: symmetric_part(subj_mat[m][s]) for s in subj_mat[m]}
            if m in asymm else subj_mat[m])
        for m in metrics if m in subj_mat
    }


def compute_measurecorrs_bootstrap(
    subj_mat: dict[str, dict[str, NDArrayAny]],
    metrics: Sequence[str],
    N: int = 20,
    Smin: int = 100,
    n_resamples: int = 2000,
    rng_seed: int = 0,
    estimator: str = "POM",
) -> dict[str, Any]:
    """Bootstrap cross-metric correlation estimator.

    Two `estimator` modes implementing the same scientific quantity with
    very different CI behavior — see code_issues #75 for the trade-off:

    - "POM" (default; matches thesis §5.1 prose): per iteration, average
      per-connection FC across N picks → length-P mean vector per metric
      → Pearson across connections → C_stack[b, i, j]. Within-seed
      iteration SD ≈ Fisher SE = (1 − ρ²)/√(P − 2) regardless of N (only
      collapses slowly at N ≫ S).
    - "MOP": per subject, Pearson(metric_i, metric_j) across that
      subject's connections. Per iteration, average across N picked
      subjects. Within-seed iteration SD = Fisher SE / √N — classical
      bootstrap-mean 1/√N tightening because per-subject Pearsons are
      averaged at the scalar level (not at the vector-then-correlate
      level POM uses).

    For both modes: `C_stack[b, i, j]` is the per-iteration r,
    `C_mean`/`C_lo`/`C_hi` are mean / 2.5 / 97.5 percentile across iters,
    `C_ciw` = hi − lo, `sig_mask[i, j]` is True iff 0 ∉ [C_lo, C_hi].

    Returns a dict with: C_stack, C_mean, C_lo, C_hi, C_ciw, sig_mask,
    ok_use, subjects_all, metrics, estimator. Pure compute — no plotting,
    no I/O.
    """
    metrics = [m for m in metrics if m in subj_mat]
    if len(metrics) < 2:
        raise ValueError(f"need >=2 metrics present in subj_mat; have {len(metrics)}")
    if estimator not in ("POM", "MOP"):
        raise ValueError(f"estimator must be 'POM' or 'MOP'; got {estimator!r}")
    # Asymmetric metrics are reduced to their symmetric part inside the
    # vectorization helpers (get_connections_w_min_subjects / bootstrapped_pair_means
    # / prep_regpair_vectors), so the heatmap is order-invariant and symmetric.
    subjects_all = sorted(set.intersection(*[set(subj_mat[m].keys()) for m in metrics]))
    ok_use, _ = get_connections_w_min_subjects(
        subj_mat, subjects_all, list(metrics), Smin=Smin,
    )
    n = len(metrics)
    C_stack = np.full((n_resamples, n, n), np.nan)

    if estimator == "POM":
        boot_upper, boot_lower = bootstrapped_pair_means(
            subj_mat=subj_mat, subjects_all=subjects_all, metrics=metrics,
            ok_use=ok_use, N=N, n_resamples=n_resamples, rng_seed=rng_seed,
        )
        for b in range(n_resamples):
            for i, mi in enumerate(metrics):
                for j, mj in enumerate(metrics):
                    if i <= j:
                        vi, vj = boot_upper[mi][b], boot_upper[mj][b]
                    else:
                        vi, vj = boot_lower[mi][b], boot_lower[mj][b]
                    ok = np.isfinite(vi) & np.isfinite(vj)
                    if ok.sum() >= 3:
                        C_stack[b, i, j] = pairwise_corr_nan(vi[ok], vj[ok])
    else:  # MOP
        V = prep_regpair_vectors(subj_mat, subjects_all, list(metrics), ok_use)
        n_subs = len(subjects_all)
        per_subj_r = np.full((n_subs, n, n), np.nan)
        for s_idx in range(n_subs):
            for i, mi in enumerate(metrics):
                for j, mj in enumerate(metrics):
                    if i == j:
                        per_subj_r[s_idx, i, j] = 1.0
                        continue
                    vi, vj = V[mi][s_idx], V[mj][s_idx]
                    ok = np.isfinite(vi) & np.isfinite(vj)
                    if ok.sum() >= 3:
                        per_subj_r[s_idx, i, j] = pairwise_corr_nan(vi[ok], vj[ok])
        rng = np.random.default_rng(rng_seed)
        for b in range(n_resamples):
            picks = rng.choice(n_subs, size=N, replace=True)
            C_stack[b] = np.nanmean(per_subj_r[picks], axis=0)

    C_mean = np.nanmean(C_stack, axis=0)
    C_lo = np.nanpercentile(C_stack, 2.5, axis=0)
    C_hi = np.nanpercentile(C_stack, 97.5, axis=0)
    C_ciw = C_hi - C_lo
    C_se = np.nanstd(C_stack, axis=0, ddof=1)
    sig_mask = (C_lo > 0) | (C_hi < 0)
    return {
        "C_stack": C_stack,
        "C_mean": C_mean,
        "C_lo": C_lo,
        "C_hi": C_hi,
        "C_ciw": C_ciw,
        "C_se": C_se,
        "sig_mask": sig_mask,
        "ok_use": ok_use,
        "subjects_all": subjects_all,
        "metrics": list(metrics),
        "estimator": estimator,
    }


# Project-wide FDR knob. "bh" reproduces the historical `fdrcorrection`
# default (Benjamini-Hochberg, independence/positive-dependence); "by_twostage"
# is the Benjamini-Yekutieli two-stage adaptive linear step-up procedure used by
# Rao et al. (2025) for hub/connection significance (statsmodels 'tsbky').
_FDR_METHODS: dict[str, str] = {"bh": "fdr_bh", "by_twostage": "fdr_tsbky"}


def fdr_adjust(
    pvals: NDArrayAny, alpha: float = 0.05, method: str = "bh"
) -> tuple[NDArrayAny, NDArrayAny]:
    """Multiple-comparison correction with a selectable procedure.

    Returns ``(reject, p_adj)`` over the supplied 1-D p-value array.
    ``method="bh"`` matches the project's historical ``fdrcorrection`` call;
    ``method="by_twostage"`` is Rao et al.'s Benjamini-Yekutieli two-stage
    adaptive step-up. Unknown methods raise ``KeyError`` (no silent fallback).
    """
    p = np.asarray(pvals, float)
    if p.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    sm_method = _FDR_METHODS[method]
    reject, p_adj, _, _ = multipletests(p, alpha=alpha, method=sm_method)
    return np.asarray(reject, dtype=bool), np.asarray(p_adj, dtype=float)


def bootstrap_reverse_percentile_pvalues(
    stack: NDArrayAny, estimate: NDArrayAny, null: float = 0.0
) -> NDArrayAny:
    """Two-sided reverse-percentile (basic-bootstrap) p-value per cell.

    `stack` is the bootstrap distribution with the resample axis first
    (e.g. ``(B, n, n)``); `estimate` is the point estimate over the
    remaining axes (e.g. ``(n, n)``). Tests H0: ``theta == null``.

    The basic-bootstrap CI is ``[2*est - q(1-a/2), 2*est - q(a/2)]``; ``null``
    lies inside it iff ``F*(2*est - null)`` falls in ``[a/2, 1-a/2]`` (``F*``
    = bootstrap CDF), so the two-sided achieved significance level is
    ``2 * min(F*, 1 - F*)`` evaluated at the reflected point. Cells whose
    `estimate` is non-finite return ``nan``.
    """
    reflected = 2.0 * estimate - null
    cdf = np.mean(stack <= reflected[None, ...], axis=0)
    p = 2.0 * np.minimum(cdf, 1.0 - cdf)
    p = np.clip(p, 0.0, 1.0)
    return np.where(np.isfinite(estimate), p, np.nan)


def reverse_percentile_fdr_mask(
    stack: NDArrayAny, estimate: NDArrayAny, *,
    alpha: float = 0.05, null: float = 0.0, exclude_diagonal: bool = True,
) -> NDArrayAny:
    """Boolean significance mask at ``p_FDR < alpha`` for a square estimate.

    Computes the reverse-percentile p-value per cell
    (`bootstrap_reverse_percentile_pvalues`), then Benjamini-Hochberg FDR
    (`fdrcorrection`) across the finite, off-diagonal cells (the diagonal of
    a correlation matrix is a trivial self-correlation and is excluded by
    default from both the test pool and the returned mask).
    """
    p = bootstrap_reverse_percentile_pvalues(stack, estimate, null=null)
    n = p.shape[0]
    pool = np.isfinite(p)
    if exclude_diagonal:
        pool &= ~np.eye(n, dtype=bool)
    mask = np.zeros((n, n), dtype=bool)
    if pool.any():
        reject, _ = fdrcorrection(p[pool], alpha=alpha)
        mask[pool] = reject
    return mask


def compute_measurecorrs_2beh_bootstrap(
    subj_mat1: dict[str, dict[str, NDArrayAny]],
    subj_mat2: dict[str, dict[str, NDArrayAny]],
    metrics: Sequence[str],
    N: int = 20,
    Smin: int = 100,
    n_resamples: int = 2000,
    rng_seed: int = 0,
    estimator: str = "POM",
) -> dict[str, Any]:
    """Cross-CONTRAST measurecorr bootstrap.

    Two estimator modes (same trade-off as `compute_measurecorrs_bootstrap`):

    - "POM" (default; matches thesis §5.1 prose): per iteration, average
      per-connection FC across N picks for each behavior → length-P mean
      vectors per metric per behavior → cell (i, j) = Pearson between
      b1's metric-i mean vector and b2's metric-j mean vector across
      connections.
    - "MOP": per subject compute Pearson(metric_i on b1, metric_j on b2)
      across that subject's connections. Per iteration, average across N
      picked subjects.

    Cells are asymmetric: (i, j) measures "metric i on b1 vs metric j on
    b2" which is generally not equal to (j, i).

    Returns C_stack / C_mean / C_lo / C_hi / C_ciw / sig_mask / ok_use /
    subjects_all / metrics / estimator. Pure compute — no plotting, no
    I/O.
    """
    metrics = [m for m in metrics if m in subj_mat1 and m in subj_mat2]
    if len(metrics) < 2:
        raise ValueError(f"need >=2 metrics present in both subj_mats; have {len(metrics)}")
    if estimator not in ("POM", "MOP"):
        raise ValueError(f"estimator must be 'POM' or 'MOP'; got {estimator!r}")
    # Symmetric-part reduction of asymmetric metrics happens inside the
    # vectorization helpers (see compute_measurecorrs_bootstrap).
    subjects_all = sorted(
        set.intersection(*[set(subj_mat1[m].keys()) for m in metrics])
        & set.intersection(*[set(subj_mat2[m].keys()) for m in metrics])
    )
    ok_use1, _ = get_connections_w_min_subjects(
        subj_mat1, subjects_all, list(metrics), Smin=Smin,
    )
    ok_use2, _ = get_connections_w_min_subjects(
        subj_mat2, subjects_all, list(metrics), Smin=Smin,
    )
    ok_use = ok_use1 & ok_use2
    V1 = prep_regpair_vectors(subj_mat1, subjects_all, list(metrics), ok_use)
    V2 = prep_regpair_vectors(subj_mat2, subjects_all, list(metrics), ok_use)
    n = len(metrics)
    C_stack = np.full((n_resamples, n, n), np.nan)

    if estimator == "POM":
        b1, b2 = bootstrapped_pair_means_2beh(
            V1, V2, list(metrics),
            N=N, n_resamples=n_resamples, rng_seed=rng_seed,
        )
        for b in range(n_resamples):
            for i, mi in enumerate(metrics):
                for j, mj in enumerate(metrics):
                    vi = b1[mi][b]
                    vj = b2[mj][b]
                    ok = np.isfinite(vi) & np.isfinite(vj)
                    if ok.sum() >= 3:
                        C_stack[b, i, j] = pairwise_corr_nan(vi[ok], vj[ok])
    else:  # MOP
        n_subs = len(subjects_all)
        per_subj_r = np.full((n_subs, n, n), np.nan)
        for s_idx in range(n_subs):
            for i, mi in enumerate(metrics):
                for j, mj in enumerate(metrics):
                    vi, vj = V1[mi][s_idx], V2[mj][s_idx]
                    ok = np.isfinite(vi) & np.isfinite(vj)
                    if ok.sum() >= 3:
                        per_subj_r[s_idx, i, j] = pairwise_corr_nan(vi[ok], vj[ok])
        rng = np.random.default_rng(rng_seed)
        for b in range(n_resamples):
            picks = rng.choice(n_subs, size=N, replace=True)
            C_stack[b] = np.nanmean(per_subj_r[picks], axis=0)

    C_mean = np.nanmean(C_stack, axis=0)
    C_lo = np.nanpercentile(C_stack, 2.5, axis=0)
    C_hi = np.nanpercentile(C_stack, 97.5, axis=0)
    C_ciw = C_hi - C_lo
    C_se = np.nanstd(C_stack, axis=0, ddof=1)
    sig_mask = (C_lo > 0) | (C_hi < 0)
    return {
        "C_stack": C_stack,
        "C_mean": C_mean,
        "C_lo": C_lo,
        "C_hi": C_hi,
        "C_ciw": C_ciw,
        "C_se": C_se,
        "sig_mask": sig_mask,
        "ok_use": ok_use,
        "subjects_all": subjects_all,
        "metrics": list(metrics),
        "estimator": estimator + "_2beh",
    }


def per_subject_metric_z(
    subj_mat: dict[str, dict[str, NDArrayAny]],
    subjects: Sequence[str],
    metrics: Sequence[str],
    ok_use: NDArrayAny,
) -> NDArrayAny:
    V = prep_regpair_vectors(subj_mat, subjects, metrics, ok_use)
    nS, nM = len(subjects), len(metrics)
    Z = np.full((nS, nM, nM), np.nan)
    for i, mi in enumerate(metrics):
        for j, mj in enumerate(metrics):
            if i == j:
                continue
            Vi, Vj = V[mi], V[mj]
            for s in range(nS):
                vi, vj = Vi[s], Vj[s]
                ok = np.isfinite(vi) & np.isfinite(vj)
                if ok.sum() >= 3:
                    r = pairwise_corr_nan(vi[ok], vj[ok])
                    if np.isfinite(r) and abs(r) < 1.0:
                        Z[s, i, j] = np.arctanh(r)
    return Z

def metric_corr_pvals(Z: NDArrayAny) -> Any:
    nM = Z.shape[1]
    P = np.full((nM, nM), np.nan)
    for i in range(nM):
        for j in range(nM):
            if i == j:
                continue
            z = Z[:, i, j]
            z = z[np.isfinite(z)]
            if z.size >= 3:
                _, P[i, j] = ttest_1samp(z, 0.0)
    mask = np.isfinite(P)
    Q = np.full_like(P, np.nan)
    Q[mask] = fdrcorrection(P[mask])[1]
    return P, Q

def _stars(q: float) -> str:  # pyright: ignore[reportUnusedFunction]
    """p-value-to-stars: ***/**/*/'' for q < 0.001 / 0.01 / 0.05 / else."""
    if not np.isfinite(q): return ""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""

def metric_corr_detection_rate(
    Z: NDArrayAny,
    N: int,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> Any:
    rng = np.random.default_rng(rng_seed)
    nS, nM, _ = Z.shape
    hits = np.zeros((nM, nM))
    runs = np.zeros((nM, nM))
    for _ in range(n_resamples):
        rows = rng.integers(0, nS, size=N)
        Zb = Z[rows]
        Pb = np.full((nM, nM), np.nan)
        for i in range(nM):
            for j in range(nM):
                z = Zb[:, i, j]
                z = z[np.isfinite(z)]
                if z.size >= 3:
                    _, Pb[i, j] = ttest_1samp(z, 0.0)
        mask = np.isfinite(Pb)
        if not mask.any():
            continue
        Qb = np.full_like(Pb, np.nan)
        Qb[mask] = fdrcorrection(Pb[mask])[1]
        ok = np.isfinite(Qb)
        hits[ok] += (Qb[ok] < alpha)
        runs[ok] += 1
    return np.where(runs > 0, hits / runs, np.nan)

def _rate_stars(r: float) -> str:  # pyright: ignore[reportUnusedFunction]
    if not np.isfinite(r): return ""
    return "***" if r >= 0.99 else "**" if r >= 0.95 else "*" if r >= 0.80 else ""


# -------------------- phase connectivity helpers --------------------------
def session_eeg(dfrow, win):
    ev = load_events(dfrow, "word_on")
    words = ev[np.asarray(ev.attrs["mask"], bool)].reset_index(drop=True)
    words.attrs = dict(ev.attrs)
    eeg, _ = get_beh_eeg(dfrow, words, save=False, window=win)
    return np.asarray(eeg.data), np.asarray(eeg.time, float), float(eeg.samplerate)

def mt_event_angles(data, t, sf, seed, tg, win, fmin, fmax, bandwidth):
    m = (t >= win[0]) & (t <= win[1])
    X, freqs, w = psd_array_multitaper(
        data[:, :, m], sf, fmin=fmin, fmax=fmax, bandwidth=bandwidth,
        adaptive=False, low_bias=False, output="complex", verbose=False)
    Xs, Xt = X[:, [seed]], X[:, tg]                        
    csd = np.sum((w * Xs) * np.conj(w * Xt), axis=-2)      
    csd = csd * (2.0 / (w * np.conj(w)).real.sum(axis=-2))
    return np.angle(csd.mean(-1).mean(1))     

def contacts_of(label):
    return set(label.split("-"))

def conn_from_z(ang, metric):
    """Across-event connectivity of one seed→target pair from per-event angles."""
    ang = np.asarray(ang); ang = ang[np.isfinite(ang)]
    n = ang.size
    if n < 2: return np.nan 
    z = np.exp(1j * ang); R = abs(z.mean())
    if metric == "PLV":   return R
    if metric == "PPC":   return (n * R * R - 1) / (n - 1)
    re, im = z.real.mean(), z.imag.mean()
    if metric == "ciPLV": return abs(im) / np.sqrt(max(1 - re * re, 1e-12))
    if metric == "PLI":   return abs(np.sign(z.imag).mean())

def stars(q):
    """Significance stars from a (corrected) p-value."""
    if not np.isfinite(q): return ""
    if q < 0.001: return "***"
    if q < 0.01:  return "**"
    if q < 0.05:  return "*"
    return ""

def binstats(v, centers, binidx):
    mean = np.full(len(centers), np.nan); sem = np.full(len(centers), np.nan)
    for b in range(len(centers)):
        vv = v[(binidx == b) & np.isfinite(v)]
        if vv.size:
            mean[b] = vv.mean()
            if vv.size > 1:
                sem[b] = vv.std(ddof=1) / np.sqrt(vv.size)
    return mean, sem

def bin_wilcoxon(m):
    """Return raw p, FDR-corrected q, n, and median Δ (On-Off) per distance bin."""
    p_raw = np.full(len(centers), np.nan)
    dmed  = np.full(len(centers), np.nan)
    npair = np.zeros(len(centers), int)
    off_all, on_all = conn["pre"][m], conn["post"][m]
    for b in range(len(centers)):
        sel = (binidx == b)
        off, on = off_all[sel], on_all[sel]
        ok = np.isfinite(off) & np.isfinite(on)
        off, on = off[ok], on[ok]
        npair[b] = off.size
        dmed[b]  = np.median(on - off) if off.size else np.nan
        if off.size >= MIN_PAIRS and np.any((on - off) != 0):
            try:
                p_raw[b] = wilcoxon(on, off).pvalue
            except ValueError:
                p_raw[b] = np.nan
    q = np.full(len(centers), np.nan)
    fin = np.isfinite(p_raw)
    if fin.any():
        q[fin] = fdrcorrection(p_raw[fin], alpha=0.05)[1]   
    return p_raw, q, npair, dmed

# =============================================================================
# ROI / distance analysis layer
# =============================================================================
# Shared machinery for the three ROI-level build scripts:
#
#   build_roi_power.py       local spectral power per Burke ROI
#   build_roi_synchrony.py   phase FC (compute) + collapsed synchrony per ROI
#   build_power_synchrony.py power-synchrony correlation per Burke ROI
#
# Everything those scripts have in common lives here -- the behavior registry,
# the Burke ROI vocabulary, pair geometry, the distance-bin synchrony collapse,
# the per-ROI test, the session dispatcher and the ROI figure --
# so each script contains only what is unique to it and none of them has to
# import a plotting script to get a constant.
#
# Nothing in this section imports the build or plot scripts; the dependency
# runs one way (scripts -> here).

# --- behavior registry -------------------------------------------------------
# The 4 main task contrasts (config.yaml `behaviors_main`), keyed by behavior.
# Each contrast is two SAVED conditions: `lo` is the reference arm, `hi` the
# active one, and diff = hi - lo. Condition dir names come straight from
# compute_session_fc:
#   word_on / voc -> PREPOST_SPEC (two TIME WINDOWS)  -> baseline / succ
#   en / rm       -> succ/fail SME (two EVENT GROUPS) -> fail / succ
BEH_CONTRASTS: dict[str, dict[str, str]] = {
    "word_on": {"beh": "word_on", "lo": "baseline", "hi": "succ",
                "lo_label": "Word Off", "hi_label": "Word On",
                "title": "Word Presentation vs Pre-Word"},
    "en":      {"beh": "en", "lo": "fail", "hi": "succ",
                "lo_label": "Not Recalled", "hi_label": "Recalled",
                "title": "Encoding"},
    "rm":      {"beh": "rm", "lo": "fail", "hi": "succ",
                "lo_label": "Deliberation", "hi_label": "Recalled",
                "title": "Retrieval"},
    "voc":     {"beh": "voc", "lo": "baseline", "hi": "succ",
                "lo_label": "Pre-Vocalization", "hi_label": "Vocalization",
                "title": "Vocalization vs Pre-Vocalization"},
}

BEHAVIORS: tuple[str, ...] = tuple(BEH_CONTRASTS)

PHASE_METRICS: tuple[str, ...] = ("plv", "ppc", "ciplv", "pli")


def contrast(beh: str) -> dict[str, str]:
    """{lo, hi, lo_label, hi_label, title} for `beh`. Raises on an unknown beh."""
    try:
        return BEH_CONTRASTS[beh]
    except KeyError:
        raise KeyError(
            f"unknown behavior {beh!r}; known: {list(BEH_CONTRASTS)}") from None


def beh_conds(beh: str) -> tuple[str, str, str]:
    """(lo, hi, 'diff') -- the saved condition dirs this behavior writes/reads."""
    c = contrast(beh)
    return (c["lo"], c["hi"], "diff")


def cond_labels(beh: str) -> dict[str, str]:
    """{saved cond dir: panel label}, including the diff arm."""
    c = contrast(beh)
    return {c["lo"]: c["lo_label"], c["hi"]: c["hi_label"],
            "diff": f"{c['hi_label']} - {c['lo_label']}"}


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
MIN_SUBJECTS_ROI: int = 5
# Minimum electrodes in a (session, ROI) cell before its correlation is taken,
# and minimum electrodes in a distance bin before that bin can be standardized.
MIN_ELEC_CORR: int = 5
MIN_ELEC_BIN: int = 5

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


def pair_distance_mask(
    xyz: NDArrayAny, lead: NDArrayAny, rmin: float, rmax: float,
    exclude_same_shank: bool = False,
) -> tuple[tuple[NDArrayAny, NDArrayAny], NDArrayAny, NDArrayAny]:
    """(upper-triangle indices, pair distances, eligibility mask).

    A pair is eligible when its distance is finite and inside [rmin, rmax], and
    -- optionally -- when the two contacts are on different shanks.
    """
    n_ch = xyz.shape[0]
    iu = np.triu_indices(n_ch, 1)
    dist = np.linalg.norm(xyz[iu[0]] - xyz[iu[1]], axis=1)
    keep = np.isfinite(dist) & (dist >= rmin) & (dist <= rmax)
    if exclude_same_shank:
        keep &= lead[iu[0]] != lead[iu[1]]
    return iu, dist, keep


def dfrow_from_sid(sid: Sequence[Any]) -> pd.Series:
    """Stored `sid` tuple -> the dfrow the loaders expect."""
    return pd.Series({"sub": str(sid[0]), "exp": str(sid[1]), "sess": int(sid[2])})


# --- distance-bin synchrony collapse ----------------------------------------
def electrode_bin_matrix(
    vals: NDArrayAny, dist: NDArrayAny, iu: tuple[NDArrayAny, NDArrayAny],
    n_ch: int, edges: NDArrayAny,
) -> NDArrayAny:
    """(n_ch, n_bins) mean connectivity per electrode per distance bin.

    Every eligible pair credits BOTH endpoints, exactly as the seed-based
    distance curves in burke_roi_connectivity.py do.
    """
    nb = len(edges) - 1
    tot = np.zeros((n_ch, nb))
    cnt = np.zeros((n_ch, nb), int)
    b = np.clip(np.digitize(dist, edges) - 1, 0, nb - 1)
    for k in range(vals.size):
        i, j, bb, v = iu[0][k], iu[1][k], b[k], vals[k]
        tot[i, bb] += v; cnt[i, bb] += 1
        tot[j, bb] += v; cnt[j, bb] += 1
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def zscore_bins(S: NDArrayAny, min_elec: int = MIN_ELEC_BIN) -> NDArrayAny:
    """Z-score each distance bin across that session's electrodes.

    WITHIN a bin, ACROSS electrodes: each bin column is standardized against its
    own electrodes, then (by the caller) the bins are averaged. This is not a
    z-score of one electrode across its own bins, which would force every
    electrode to 0 and delete the signal.

    Both moments matter, and both correct the same artifact -- ragged distance
    coverage. Centring removes the LEVEL difference (electrodes sample different
    distances, and connectivity falls off with distance). Dividing by the SD
    removes the SPREAD difference: across-electrode variability also shrinks with
    distance, so after centring alone an electrode covered only by far bins
    carries systematically smaller deviations and its score is compressed toward
    zero for purely geometric reasons.

    A bin is dropped when it has fewer than `min_elec` electrodes (its mean and
    SD would be too noisy to standardize against) or a non-positive SD.
    """
    out = np.full_like(S, np.nan)
    for b in range(S.shape[1]):
        col = S[:, b]
        ok = np.isfinite(col)
        if ok.sum() < min_elec:
            continue
        sd = np.nanstd(col, ddof=1)
        if not (np.isfinite(sd) and sd > 0):
            continue
        out[:, b] = np.where(ok, (col - np.nanmean(col)) / sd, np.nan)
    return out


def mean_over_bins(Sc: NDArrayAny) -> NDArrayAny:
    """(n_ch,) mean of each electrode's usable z-scored bins.

    Written as sum/count rather than nanmean so an electrode with no usable bin
    yields nan quietly instead of an all-NaN-slice warning.
    """
    n_ok = np.isfinite(Sc).sum(axis=1)
    return np.where(n_ok > 0, np.nansum(Sc, axis=1) / np.maximum(n_ok, 1),
                    np.nan)


def collapsed_synchrony(
    M: Any, iu: tuple[NDArrayAny, NDArrayAny], dist: NDArrayAny,
    keep: NDArrayAny, n_ch: int, edges: NDArrayAny,
) -> tuple[NDArrayAny, NDArrayAny] | None:
    """Connectivity matrix -> (per-electrode collapsed synchrony, z-scored bins).

        s[e, b]  = mean connectivity of electrode e to its partners in bin b
        s'[e, b] = (s[e, b] - mean_e s[:, b]) / sd_e s[:, b]
        S[e]     = mean over the bins e populates of s'[e, b]

    S[e] reads "how much more (or less) synchronized is this electrode than
    expected given the distances it happens to sample" -- a correction for
    RAGGED BIN COVERAGE, not for the distance decay itself (if every electrode
    populated every bin the centring would subtract the same constant from
    everyone). Standardizing also gives every distance range equal influence,
    without which the high-variance short-range bins decide the result.

    Returns None when no eligible pair has a finite value.
    """
    vals = np.asarray(M, float)[iu]
    sel = keep & np.isfinite(vals)
    if not sel.any():
        return None
    S = electrode_bin_matrix(vals[sel], dist[sel], (iu[0][sel], iu[1][sel]),
                             n_ch, edges)
    Sc = zscore_bins(S)
    return mean_over_bins(Sc), Sc


def zscore_across(v: Any) -> NDArrayAny:
    """z across electrodes; nan-vector when the SD is undefined or zero."""
    v = np.asarray(v, float)
    sd = np.nanstd(v, ddof=1)
    if not (np.isfinite(sd) and sd > 0):
        return np.full_like(v, np.nan)
    return (v - np.nanmean(v)) / sd


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
    min_subjects: int = MIN_SUBJECTS_ROI,
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
        if v.size >= min_subjects and np.std(v, ddof=1) > 0:
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
        _, q = fdr_adjust(np.asarray(pv))
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
    stats: pd.DataFrame | None = None, style: str = "box",
) -> list[float]:
    """One 12-ROI panel over the per-subject values in `tbl[measure]`.

    style="box" (default) median/IQR box + 1.5*IQR whiskers + mean diamond.
                Over the full session list every ROI has ~120-290 subjects, so
                the quartiles are well estimated and the distribution's shape
                and tails are worth showing.
    style="ci"  mean + 95% CI -- the estimate the t-test is run on, so an
                interval clear of the 0 line IS the (uncorrected) result. Useful
                on small subsets (a --n-sessions smoke test leaves single-digit
                subject counts, where quartiles are interpolated between
                adjacent points and the whiskers just reach the extremes), but
                at full n the interval shrinks to near-invisible and the
                distribution is the more informative thing to draw.
    """
    data: list[NDArrayAny] = []
    colors: list[str] = []
    pos = _roi_positions(roi_order, lobes)
    for roi in roi_order:
        data.append(tbl.loc[tbl["roi"] == roi, measure].dropna().to_numpy(float))
        colors.append(LOBE_COLORS.get(roi.split("-", 1)[1], "0.5"))

    drawn = [i for i, v in enumerate(data) if v.size >= MIN_SUBJECTS_ROI]

    if style == "box" and drawn:          # boxplot([]) raises in matplotlib >= 3.9
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
        m, h = mean_ci(data[i])
        if style == "box":                  # mean as a diamond beside median
            ax.scatter([pos[i]], [m], marker="D", s=26, facecolor="white",
                       edgecolor="0.15", linewidths=1.2, zorder=4)
        else:
            ax.errorbar(pos[i], m, yerr=h, color=colors[i], marker="o", ms=7,
                        lw=2.0, capsize=4, markeredgecolor="0.15",
                        markeredgewidth=0.8, zorder=5)

    _roi_axis_furniture(ax, pos, roi_order, lobes)
    _annotate_counts_and_stars(
        ax, pos, roi_order,
        n=[v.size for v in data],
        tops=[float(np.max(v)) if v.size else np.nan for v in data],
        stats=stats)
    return pos


def roi_bar_panel(
    ax: Any, st: pd.DataFrame, value: str = "mean", err: str = "sem",
    roi_order: Sequence[str] = tuple(ROI_ORDER),
    lobes: Sequence[str] = tuple(LOBES),
) -> list[float]:
    """One bar per ROI (mean +/- SEM) from a `roi_stats` table."""
    pos = _roi_positions(roi_order, lobes)
    s = st.set_index("roi")
    mu = [float(s.loc[r, value]) if r in s.index else np.nan for r in roi_order]
    se = [float(s.loc[r, err]) if r in s.index else np.nan for r in roi_order]
    cols = [LOBE_COLORS.get(r.split("-", 1)[1], "0.5") for r in roi_order]
    ax.bar(pos, mu, width=0.62, yerr=se, color=cols, alpha=0.85, capsize=3,
           error_kw=dict(lw=1, ecolor="0.3"))

    _roi_axis_furniture(ax, pos, roi_order, lobes)
    tops = [m + (np.nan_to_num(e) if m >= 0 else -np.nan_to_num(e))
            if np.isfinite(m) else np.nan for m, e in zip(mu, se)]
    _annotate_counts_and_stars(
        ax, pos, roi_order,
        n=[int(s.loc[r, "n_subjects"]) if r in s.index else 0 for r in roi_order],
        tops=tops, stats=st, below=[m < 0 for m in mu])
    return pos


def _annotate_counts_and_stars(
    ax: Any, pos: Sequence[float], roi_order: Sequence[str],
    n: Sequence[int], tops: Sequence[float], stats: pd.DataFrame | None,
    below: Sequence[bool] | None = None,
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
            down = bool(below[i]) if below is not None else False
            ax.annotate(star, (pos[i], tops[i]), textcoords="offset points",
                        xytext=(0, -12 if down else 5), ha="center",
                        va="top" if down else "bottom", fontsize=11,
                        fontweight="bold", color="k")


def roi_figure(
    panels: Sequence[tuple[str, Callable[[Any], Any]]],
    out_dir: str, stem: str, height: float = 3.6,
) -> None:
    """Stack one panel per measure, label each y axis, save .png + .pdf.

    `panels` is [(y-axis label, draw(ax)), ...] -- the caller supplies the
    per-panel draw call (roi_panel or roi_bar_panel with its own arguments), so
    the box figures and the bar figure share one layout and one save path.
    """
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, height * n + 0.8))
    for ax, (ylabel, draw) in zip(np.atleast_1d(axes), panels):
        draw(ax)
        ax.set_ylabel(ylabel, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    os.makedirs(out_dir, exist_ok=True)
    path = join(out_dir, stem)
    fig.savefig(f"{path}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    print(f"[saved] {path}.png / .pdf")
    plt.close(fig)


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
            pbar.write(f"[error] sess={_sid_str(item)}: {err!r}")
        pbar.set_postfix(ok=n_ok, err=n_err)
    if pool is not None:
        pool.shutdown()
    print(f"[{desc}] done: ok={n_ok}, err={n_err}")
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
    p.add_argument("--beh", default="word_on", choices=BEHAVIORS,
                   help="contrast (default: word_on). word_on/voc contrast two "
                        "time windows; en/rm contrast two event groups")
    p.add_argument("--band", default="alpha",
                   help="band key in config.yaml (default: alpha = 8-13 Hz)")
    p.add_argument("--root-dir", default=None,
                   help="source data dir; defaults to fc.root_dir (SCRATCH_DIR)")
    p.add_argument("--save-root", default=None,
                   help="where per-session pickles live; defaults to --root-dir")
    p.add_argument("--n-sessions", type=int, default=None,
                   help="use only the first N sessions")
    if compute:
        p.add_argument("--n-subjects", type=int, default=None,
                       help="compute all sessions of the first K subjects")
        p.add_argument("--workers", type=int, default=1,
                       help="sessions computed at once in separate processes "
                            "(1 = in this process); budget a few GB of RAM each")
    return p


def add_distance_args(p: Any) -> Any:
    """CLI flags for the pair-distance binning shared by the synchrony scripts."""
    p.add_argument("--metric", default="ppc", choices=PHASE_METRICS)
    p.add_argument("--rmin", type=float, default=10.0)
    p.add_argument("--rmax", type=float, default=110.0)
    p.add_argument("--bin-w", type=float, default=10.0, dest="bin_w")
    p.add_argument("--exclude-same-shank", action="store_true")
    return p


def resolve_roots(args: Any) -> tuple[str, str]:
    """(root_dir, save_root) from --root-dir / --save-root; point helper at them."""
    global root_dir
    root_dir = args.root_dir or root_dir
    save_root = getattr(args, "save_root", None) or root_dir
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
