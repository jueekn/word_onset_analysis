"""misc.py — small utilities shared across the pipeline.

Session-label helpers, pickle / scipy.io IO wrappers, t-test-with-pretty-print
shortcuts, BH and BY FDR procedures, a TFCE-on-2D-t-map helper, and a few
scalar conversion primitives.
"""
from __future__ import annotations

# Basic
from typing import Any, Callable, Sequence, cast
import os
from os.path import join

import numpy as np
import numpy.typing as npt
import scipy
import scipy.stats  # pyright: ignore[reportMissingTypeStubs]

# Data Handling
import pickle
import h5py  # pyright: ignore[reportMissingTypeStubs]

# Data Analysis
import pandas as pd

NDArrayAny = npt.NDArray[Any]


def ftag(dfrow: pd.Series) -> str:
    """Filename tag for a session.

    Args:
        dfrow: pandas.Series with at least keys (sub, exp, sess, loc, mon).

    Returns:
        '{sub}_{exp}_{sess}' when loc == mon == 0; otherwise
        '{sub}_{exp}_{sess}_{loc}_{mon}'.
    """
    # pandas Series indexing returns Unknown via stubs; cast at boundary.
    sub = cast(Any, dfrow['sub'])
    exp = cast(Any, dfrow['exp'])
    sess = cast(Any, dfrow['sess'])
    loc = cast(Any, dfrow['loc'])
    mon = cast(Any, dfrow['mon'])
    if (int(loc) == 0) and (int(mon) == 0):
        return f'{sub}_{exp}_{sess}'
    else:
        return f'{sub}_{exp}_{sess}_{loc}_{mon}'


def get_dfrow(dfrow: Sequence[Any] | NDArrayAny) -> pd.Series:
    """Coerce a (sub, exp, sess[, loc, mon]) tuple/list/array to a Series.

    Args:
        dfrow: length-3 (sub, exp, sess) or length-5 (sub, exp, sess, loc, mon).

    Returns:
        pandas.Series with the same keys as the input length.
    """
    if len(dfrow) == 3:
        sub, exp, sess = dfrow
        return pd.Series({'sub': sub,
                          'exp': exp,
                          'sess': sess})
    else:
        sub, exp, sess, loc, mon = dfrow
        return pd.Series({'sub': sub,
                          'exp': exp,
                          'sess': sess,
                          'loc': loc,
                          'mon': mon})


def load_pickle(path: str) -> Any:
    """Load any pickle file. Return type is genuinely Any (caller-known)."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(path: str, obj: Any) -> None:
    """Pickle obj to path (binary, default protocol)."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def print_header(text: str) -> None:
    """Pretty-print a section header surrounded by dashes."""
    print(f'---------{text}---------')


def print_whether_significant(p: float, alpha: float = 0.05) -> None:
    """Print 'Statistically Significant' / 'NOT ...' based on p vs alpha."""
    if p < alpha:
        print(f'Statistically Significant (p < {alpha})')
    else:
        print(f'NOT Statistically Significant (p >= {alpha})')

def finitize(x: NDArrayAny) -> NDArrayAny:
    """Return the unraveled subset of x containing only finite values."""
    return x[np.isfinite(x)]

def print_ttest_1samp(
    vals: NDArrayAny,
    header: str | None = None,
    alternative: str = 'two-sided',
    popmean: float = 0,
) -> tuple[float, float]:
    """One-sample t-test against popmean; prints a formatted result, returns (t, p)."""
    vals = finitize(vals)
    # scipy.stats stubs return Unknown overload; cast at boundary.
    t, p = cast("tuple[float, float]", scipy.stats.ttest_1samp(  # pyright: ignore[reportUnknownMemberType]
        vals, popmean=popmean, alternative=alternative))
    sem_v = cast(float, scipy.stats.sem(vals))  # pyright: ignore[reportUnknownMemberType]
    if header is not None:
        print_header(header)
    print_whether_significant(p)
    print(f't_{len(vals)-1} = {t:.3}, p = {p:.3}, '
          f'Mean: {np.mean(vals):.3} ± {sem_v:.3}')
    return t, p


def print_ttest_rel(
    a: NDArrayAny,
    b: NDArrayAny,
    header: str | None = None,
    alternative: str = 'two-sided',
) -> tuple[float, float]:
    """Paired t-test (a, b); finite-pairs only; prints + returns (t, p)."""
    where_finite = np.isfinite(a) & np.isfinite(b)
    a = a[where_finite]
    b = b[where_finite]
    # scipy.stats stubs return Unknown overload; cast at boundary.
    t, p = cast("tuple[float, float]", scipy.stats.ttest_rel(  # pyright: ignore[reportUnknownMemberType]
        a, b, alternative=alternative))
    sem_a = cast(float, scipy.stats.sem(a))  # pyright: ignore[reportUnknownMemberType]
    sem_b = cast(float, scipy.stats.sem(b))  # pyright: ignore[reportUnknownMemberType]
    sem_d = cast(float, scipy.stats.sem(a - b))  # pyright: ignore[reportUnknownMemberType]
    if header is not None:
        print_header(header)
    print_whether_significant(p)
    print(f't_{len(a)-1} = {t:.3}, p = {p:.3}, '
          f'Mean_A: {np.mean(a):.3} ± {sem_a:.3}, '
          f'Mean_B: {np.mean(b):.3} ± {sem_b:.3}, '
          f'Mean_Diff: {np.mean(a-b):.3} ± {sem_d:.3}')
    return t, p


def jzs_bayes_factor(t: float, N: int) -> float:
    """JZS Bayes factor for a one-sample t against zero.

    Args:
        t: observed t-statistic.
        N: sample size.

    Returns:
        BF₁₀ in the Rouder et al. (2009) JZS prior parameterization.
    """
    from scipy.integrate import quad as integral  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
    v = N - 1
    numerator = (1 + t**2/v)**(-(v+1)/2)
    integrand: Callable[[float], float] = lambda g: (
        (1 + N*g)**(-1/2)
        * (1 + t**2/((1 + N*g)*v))**(-(v+1)/2)
        * (2*np.pi)**(-1/2)
        * (g**(-3/2))
        * ((np.e)**(-1/(2*g)))
    )
    # scipy.integrate.quad returns (value, error_estimate); cast both.
    denominator = cast(
        "tuple[float, float]",
        integral(integrand, 0, np.inf))[0]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return float(numerator / denominator)


def one_stage_linear_step_up(ps: NDArrayAny, alpha: float = 0.05) -> pd.Series:
    """Benjamini-Hochberg (1995) FDR procedure.

    Args:
        ps: 1D array of raw p-values.
        alpha: FDR target.

    Returns:
        pandas.Series with keys 'ps_corr' (BH-adjusted p-values) and
        'rejected' (boolean mask, ps_corr < alpha).
    """
    m = len(ps)
    # Code-issues #65: stable sort so ranks are deterministic across runs
    # / platforms when ps contains ties (common after p-value clamping).
    ranks = np.argsort(np.argsort(ps, kind="stable"), kind="stable") + 1
    ps_ = np.asarray([(m*p)/j for j, p in zip(ranks, ps)])
    ps_corr = np.asarray(
        [np.min(ps_[ranks >= i]) for i in ranks])  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    ps_corr[ps_corr > 1] = 1
    return pd.Series({'ps_corr': ps_corr,
                      'rejected': ps_corr < alpha})


def two_stage_linear_step_up(ps: NDArrayAny, alpha: float = 0.05) -> pd.Series:
    """Benjamini-Yekutieli (2006) two-stage linear step-up FDR.

    Per Definition 6 of BY 2006.

    Args:
        ps: 1D array of raw p-values.
        alpha: FDR target.

    Returns:
        pandas.Series with keys 'ps_corr' and 'rejected'.
    """
    m = len(ps)
    # Code-issues #65: stable sort so ranks are deterministic across runs
    # / platforms when ps contains ties (common after p-value clamping).
    ranks = np.argsort(np.argsort(ps, kind="stable"), kind="stable") + 1
    ps_ = np.asarray([(m*p*(1+alpha))/j for j, p in zip(ranks, ps)])
    ps_corr = np.asarray(
        [np.min(ps_[ranks >= i]) for i in ranks])  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    ps_corr[ps_corr > 1] = 1
    r1 = int(np.sum(ps_corr < alpha))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    m0 = m - r1
    if r1 not in [0, m]:
        ps_ = np.asarray([(m0*p*(1+alpha))/j for j, p in zip(ranks, ps)])
        ps_corr = np.asarray(
            [np.min(ps_[ranks >= i]) for i in ranks])  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        ps_corr[ps_corr > 1] = 1
    return pd.Series({'ps_corr': ps_corr,
                      'rejected': ps_corr < alpha})


def tfce(ts: NDArrayAny) -> NDArrayAny:
    """Threshold-free cluster enhancement on a 2D t-statistic map.

    Args:
        ts: 2D ndarray of t-statistics.

    Returns:
        2D ndarray of TFCE-enhanced t-statistics (same shape as ts).
    """
    import skimage  # pyright: ignore[reportMissingTypeStubs]

    def one_sided_tfce(
        ts: NDArrayAny,
        thresholds: NDArrayAny,
        comparison_function: Callable[[NDArrayAny, float], NDArrayAny],
    ) -> NDArrayAny:
        ts_tfce = np.zeros_like(ts)
        for threshold in thresholds:
            ts_beyond_threshold = comparison_function(ts, threshold)
            # skimage stubs return Unknown for label; cast at boundary.
            ts_clustered = cast(NDArrayAny, skimage.measure.label(  # pyright: ignore[reportUnknownMemberType]
                ts_beyond_threshold, connectivity=2))
            cluster_sizes = np.zeros_like(ts_clustered)
            n_clusters = int(np.max(ts_clustered))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportCallIssue]
            for iCluster in np.arange(1, n_clusters + 1):  # pyright: ignore[reportUnknownMemberType]
                cluster_sizes[ts_clustered == iCluster] = int(
                    np.sum(ts_clustered == iCluster))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            for m in np.arange(ts.shape[0]):  # pyright: ignore[reportUnknownMemberType]
                for n in np.arange(ts.shape[1]):  # pyright: ignore[reportUnknownMemberType]
                    ts_tfce[m, n] = ts_tfce[m, n] + cluster_sizes[m, n]**0.5 * threshold**2
        return ts_tfce

    positive_thresholds = np.arange(  # pyright: ignore[reportUnknownMemberType]
        0, float(np.max(ts)) + 0.05, 0.05)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportCallIssue]
    negative_thresholds = np.arange(  # pyright: ignore[reportUnknownMemberType]
        float(np.min(ts)), 0.00 + 0.05, 0.05)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportCallIssue]
    positive_ts_tfce = one_sided_tfce(ts, positive_thresholds, np.greater)
    negative_ts_tfce = one_sided_tfce(ts, negative_thresholds, np.less)
    return positive_ts_tfce + negative_ts_tfce


def duration_to_samples(duration_s: float, sample_rate_Hz: float) -> int:
    """Number of samples in a duration_s window at sample_rate_Hz, +1 inclusive.

    TODO(types): code_issues #48 — the +1 is the "inclusive of endpoint"
    convention; arange(0, duration_s, 1/sample_rate_Hz) produces N samples
    where N = duration_s * sample_rate_Hz (no +1). Mismatched contracts.
    """
    return int(duration_s * sample_rate_Hz) + 1


from typing import overload as _overload

@_overload
def get_time_offset(phase_offset: float, frequency: float) -> float: ...
@_overload
def get_time_offset(phase_offset: NDArrayAny, frequency: float) -> NDArrayAny: ...
def get_time_offset(
    phase_offset: float | NDArrayAny, frequency: float
) -> float | NDArrayAny:
    """Convert a phase offset (radians) to a time offset (seconds) at frequency Hz.

    Vectorized: scalar in / scalar out, array in / array out.
    """
    return phase_offset / (2 * np.pi * frequency)


def get_username_from_working_directory(index: int = 2) -> str:
    """Extract the username segment from the cwd path at `index`.

    Defaults to index=2, matching '/home/user/...' or '/home1/user/...'.
    Raises ValueError if the path has fewer segments than `index`.
    """
    try:
        working_directory = os.getcwd()
        path_parts = working_directory.split(os.sep)
        username = path_parts[index]
        return username
    except IndexError:
        raise ValueError("Unable to extract username from working directory.")
