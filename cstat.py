"""cstat.py — circular statistics primitives.

Phase-domain helpers used across FC compute, simulation, and validation:
- vectorization of phases (cos/sin packing)
- circular mean / difference / regularization to (-pi, pi]
- mean resultant vector length and PLV / PPC primitives
- wrapped-normal moment relations (PLV ↔ sigma ↔ PPC)
- sampling from a wrapped multivariate normal.
"""
from __future__ import annotations

from typing import Any, overload

import numpy as np
import numpy.typing as npt
import scipy.stats  # pyright: ignore[reportMissingTypeStubs]

NDArrayAny = npt.NDArray[Any]


def rectangularize(m: NDArrayAny) -> NDArrayAny:
    """Pack phases into (cos, sin) along a leading axis.

    Returns an array of shape (2, *m.shape): row 0 is cos(m), row 1 is sin(m).
    """
    m = np.expand_dims(m, axis=0)
    return np.vstack([np.cos(m), np.sin(m)])


def circ_mean(x: NDArrayAny, axis: int) -> NDArrayAny:
    """Circular mean of phases x along `axis` (NaN-tolerant)."""
    rect_x = rectangularize(x)
    if axis >= 0:
        axis += 1
    mean = np.nanmean(rect_x, axis=axis)
    return np.arctan2(mean[1, ...], mean[0, ...])


def circ_diff(x: NDArrayAny, y: NDArrayAny) -> NDArrayAny:
    """Element-wise circular difference x - y, regularized to (-pi, pi]."""
    return regularize(x - y)


def regularize(x: NDArrayAny) -> NDArrayAny:
    """Wrap phases to (-pi, pi] using mod-2pi."""
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def mean_resultant_vector_length(x: NDArrayAny, axis: int) -> NDArrayAny:
    """L2 norm of the (cos-mean, sin-mean) vector along `axis`. R ∈ [0, 1]."""
    rect_x = rectangularize(x)
    if axis >= 0:
        axis += 1
    mean = np.nanmean(rect_x, axis=axis)
    return np.linalg.norm(mean, axis=0)


def dstat(
    phase_wavelet_recalled_diff: NDArrayAny,
    phase_wavelet_not_recalled_diff: NDArrayAny,
    axis: int,
) -> NDArrayAny:
    """PLV-difference statistic between recalled and not-recalled conditions.

    Uses the PLV = exp(-sigma^2 / 2) identity with sigma estimated via
    scipy.stats.circstd. Returns PLV_recalled - PLV_not_recalled.
    """
    cstd_recalled = scipy.stats.circstd(phase_wavelet_recalled_diff, axis=axis)  # pyright: ignore[reportUnknownMemberType]
    cstd_not_recalled = scipy.stats.circstd(phase_wavelet_not_recalled_diff, axis=axis)  # pyright: ignore[reportUnknownMemberType]
    PLVs_recalled = np.power(np.e, np.divide(np.power(cstd_recalled, 2), -2))
    PLVs_not_recalled = np.power(np.e, np.divide(np.power(cstd_not_recalled, 2), -2))
    return PLVs_recalled - PLVs_not_recalled


def ppc(phase: NDArrayAny) -> NDArrayAny:
    """Pairwise phase consistency along axis 0.

    Args:
        phase: shape (n, ...), n >= 2.

    Returns:
        PPC value(s) over the trailing axes. Raises ValueError for n < 2.
    """
    n = phase.shape[0]
    if n < 2:
        raise ValueError(
            f"ppc requires at least 2 samples along axis 0; got {n}. "
            "Pairwise phase consistency is undefined for a single sample."
        )
    sin_phase, cos_phase = np.sin(phase), np.cos(phase)
    distance_sum: list[NDArrayAny] = []
    for j in np.arange(n - 1):  # pyright: ignore[reportUnknownMemberType]
        d = np.sum(  # pyright: ignore[reportUnknownMemberType]
            cos_phase[j:(j + 1), ...] * cos_phase[(j + 1):, ...]
            + sin_phase[j:(j + 1), ...] * sin_phase[(j + 1):, ...],
            axis=0,
        )
        distance_sum.append(d)
    distance_sum_arr = np.sum(distance_sum, axis=0)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    return (2 / (n * (n - 1))) * distance_sum_arr


def plv(phase: NDArrayAny) -> float:
    """PLV via the PLV = exp(-sigma^2/2) identity where sigma = circular std."""
    circ_std = scipy.stats.circstd(phase)  # pyright: ignore[reportUnknownMemberType]
    return float(np.e ** ((-circ_std ** 2) / 2))

def ciplv(phase: NDArrayAny) -> NDArrayAny:
    """Corrected imaginary PLV (Bruña et al. 2018) along axis 0.

    Args:
        phase: shape (n, ...), n >= 2. 
    """
    n = phase.shape[0]
    if n < 2:
        raise ValueError(
            f"ciplv requires at least 2 samples along axis 0; got {n}."
        )
    z = np.mean(np.exp(1j * phase), axis=0)
    re, im = z.real, z.imag
    denom = np.sqrt(1.0 - re**2)
    return np.abs(im) / denom

def pli(phase: NDArrayAny) -> NDArrayAny:
    """Phase lag index (Stam et al. 2007) along axis 0.

    Args:
        phase: shape (n, ...), n >= 2. 

    Returns:
        PLI value(s) over the trailing axes.
    """
    n = phase.shape[0]
    if n < 2:
        raise ValueError(
            f"pli requires at least 2 samples along axis 0; got {n}."
        )
    return np.abs(np.mean(np.sign(np.sin(phase)), axis=0))


def plv_to_kappa(R: float) -> float:
    """Invert R = I_1(kappa)/I_0(kappa) numerically to recover kappa from PLV.

    Iterative scaling against the Bessel-function ratio. Seeds with a uniform
    random initial guess (np.random.seed pinned for reproducibility — note
    this mutates the global RNG state).

    TODO(types): the random-init pattern (np.random.seed + rand()) is
    suspicious for a deterministic inversion; surface but don't refactor here.
    """
    np.random.seed(202410)
    from scipy.special import iv  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

    close = False
    rtol, atol = 1e-10, 1e-10
    kappa = float(np.random.random())

    while not close:
        A = float(iv(1, kappa) / iv(0, kappa))
        close = bool(np.isclose(A, R, rtol=rtol, atol=atol))
        if not close:
            kappa *= R / A
    return kappa


def wrapped_normal_first_moment(mu: float, sigma: float) -> complex:
    """First circular moment of a wrapped normal: E[e^{i theta}] = e^{i mu - sigma^2/2}."""
    return complex(np.exp(1j * mu - sigma ** 2 / 2))


def wrapped_normal_sigma_to_plv(sigma: float) -> complex:
    """PLV (magnitude of first moment) for a wrapped normal with given sigma at mu=0.

    TODO(types): returns complex because wrapped_normal_first_moment returns
    complex; for sigma > 0 the imaginary part is 0 so this is effectively a
    real number. Most callers treat it as float.
    """
    return wrapped_normal_first_moment(0, sigma)


def wrapped_normal_sigma_to_ppc(sigma: float) -> complex:
    """PPC = PLV^2 for a wrapped normal with given sigma at mu=0."""
    return wrapped_normal_first_moment(0, sigma) ** 2


@overload
def plv_to_wrapped_normal_sigma(plv: float) -> float: ...
@overload
def plv_to_wrapped_normal_sigma(plv: NDArrayAny) -> NDArrayAny: ...
def plv_to_wrapped_normal_sigma(plv: float | NDArrayAny) -> float | NDArrayAny:
    """Invert PLV = exp(-sigma^2/2) → sigma = sqrt(-2 log PLV).

    Vectorized: scalar in / scalar out, array in / array out.
    """
    return np.sqrt(-2 * np.log(plv))


@overload
def ppc_to_wrapped_normal_sigma(ppc: float) -> float: ...
@overload
def ppc_to_wrapped_normal_sigma(ppc: NDArrayAny) -> NDArrayAny: ...
def ppc_to_wrapped_normal_sigma(ppc: float | NDArrayAny) -> float | NDArrayAny:
    """Invert PPC = exp(-sigma^2) → sigma = sqrt(-log PPC).

    Vectorized: scalar in / scalar out, array in / array out.
    """
    return np.sqrt(-np.log(ppc))


def sample_wrapped_multivariate_normal(
    mean: NDArrayAny,
    cov: NDArrayAny,
    size: int,
    rng: None | int | np.random.Generator = None,
    **args: Any,
) -> NDArrayAny:
    """Sample from a wrapped multivariate normal on (-pi, pi].

    Args:
        mean: shape (n,) — per-channel mean phases.
        cov: shape (n, n) — covariance of the underlying Gaussian.
        size: number of trials.
        rng: None / int seed / np.random.Generator. When provided, draws go
            through np.random.default_rng(rng) and the global numpy state is
            left untouched. When None, falls back to
            np.random.multivariate_normal (legacy global state).
        **args: forwarded to multivariate_normal (e.g. method, tol).

    Returns:
        ndarray of shape (size, n), values in (-pi, pi].
    """
    if rng is not None:
        rng = np.random.default_rng(rng)
        draws = rng.multivariate_normal(mean=mean + np.pi, cov=cov, size=size, **args)
    else:
        draws = np.random.multivariate_normal(mean=mean + np.pi, cov=cov, size=size, **args)
    return draws % (2 * np.pi) - np.pi


def rose(ax, a, title, n_bins, stats):
    a = np.ravel(a)
    if len(a) == 0:
        ax.set_title(f"{title}\n(no data)", fontsize=9, pad=14); ax.set_yticklabels([]); return
    edges = np.linspace(-np.pi, np.pi, n_bins + 1); c = (edges[:-1] + edges[1:]) / 2
    h, _ = np.histogram(a, bins=edges, density=True)
    ax.bar(c, h, width=np.diff(edges), color="#4C72B0", edgecolor="w", alpha=.85)

    z = np.mean(np.exp(1j * a))
    R, ang = abs(z), np.angle(z)
    ax.annotate("", xy=(ang, R * h.max()), xytext=(0, 0),
                arrowprops=dict(color="black", width=2, headwidth=8))
    ax.set_theta_zero_location("E"); ax.set_yticklabels([])

    n = len(a)
    plv_ = f"PLV={R:.2f}"
    stat = f"p={np.exp(-n * R * R):.1e}" if stats else ""   
    ppc_ = f"PPC={float(ppc(a)):.3f}"
    ciplv_ = f"ciPLV={float(ciplv(a)):.3f}"
    pli_ = f"PLI={float(pli(a)):.3f}"
    ax.set_title(f"{title}\n{plv_}  {stat}  \n{ppc_}\n{ciplv_}\n{pli_} n={n}".strip(), fontsize=9, pad=14)
    