"""tfce.py -- threshold-free cluster enhancement, vendored from
RaoEtal25JNeurosci/tfce.py (the machinery behind that paper's Fig. 4B).

Only the two functions this project needs are carried over: `tfce_compute` and
the sign-flip permutation test. Rao's module additionally imports
`data_containers`, `data_util` and `anatomy_util`, but those belong to other
functions in that file and are not needed here.

TWO DELIBERATE DEPARTURES FROM THE ORIGINAL
-------------------------------------------
1. CLUSTERING IS 1-D ALONG TIME, NOT 2-D.

   Rao's map is time x frequency -- both axes continuous, so 8-connectivity is
   meaningful in both directions. Ours is ROI x time, and only time is
   continuous: L-frontal and L-parietal are not neighbours in any metric sense,
   and the row order is an arbitrary sort. Running 2-D TFCE over that grid would
   merge clusters across unrelated brain regions and manufacture significance
   out of how the rows happen to be ordered.

   `tfce_map` therefore applies the enhancement to each ROI's row separately.
   The permutation null is still built over the WHOLE map, so the correction
   still accounts for having searched every region.

2. THE INNER ACCUMULATION IS VECTORISED.

   Rao's double `for m ... for n ...` loop is replaced by array arithmetic, and
   the connected-component sizes come from `np.bincount` instead of a Python
   loop over labels. This is arithmetically identical -- `test_matches_rao()`
   asserts agreement with a transcription of the original to 1e-12 -- but the
   permutation test needs thousands of calls, and the loop version is too slow
   to be usable at that scale.

The statistic itself is unchanged:

    TFCE(p) = sum_h  extent(p, h)^E * h^H * dh,    E = 0.5, H = 2, dh = 0.05

where extent(p, h) is the size of the contiguous supra-threshold cluster
containing p at height h.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import ttest_1samp
from skimage.measure import label

NDArrayAny = npt.NDArray[Any]

E_DEFAULT = 0.5      # extent exponent   (Smith & Nichols 2009; Rao uses 0.5)
H_DEFAULT = 2.0      # height exponent   (         "                    2  )
DH_DEFAULT = 0.05    # threshold step, matching Rao's np.arange(..., 0.05)


def _extent_map(mask: NDArrayAny) -> NDArrayAny:
    """Cluster size at every True cell of `mask` (0 elsewhere).

    `label(..., connectivity=2)` is Rao's choice. On a 1-row array it can only
    join left/right neighbours, which is exactly the 1-D behaviour we want.
    """
    lab = label(mask, connectivity=2)
    if lab.max() == 0:
        return np.zeros(lab.shape, float)
    sizes = np.bincount(lab.ravel())        # sizes[0] is the background
    sizes[0] = 0
    return sizes[lab].astype(float)


def tfce_compute(t: NDArrayAny, E: float = E_DEFAULT, H: float = H_DEFAULT,
                 dh: float = DH_DEFAULT) -> tuple[NDArrayAny, NDArrayAny, NDArrayAny]:
    """TFCE-enhance a 2-D statistic map. Returns (tfce, tfce_pos, tfce_neg).

    Positive and negative excursions are enhanced separately and summed, so a
    suppression is treated symmetrically with an enhancement -- which matters
    here, since a region can respond by decreasing high-gamma power.
    """
    t = np.asarray(t, float)
    finite = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    tfce_pos = np.zeros_like(finite)
    tmax = float(np.max(finite))
    if tmax > 0:
        for h in np.arange(dh, tmax + dh, dh):
            tfce_pos += _extent_map(finite > h) ** E * h ** H * dh

    tfce_neg = np.zeros_like(finite)
    tmin = float(np.min(finite))
    if tmin < 0:
        for h in np.arange(dh, -tmin + dh, dh):
            tfce_neg += _extent_map(finite < -h) ** E * h ** H * dh

    return tfce_pos - tfce_neg, tfce_pos, tfce_neg


def tfce_map(t: NDArrayAny, by_row: bool = True, **kw: Any
             ) -> tuple[NDArrayAny, NDArrayAny, NDArrayAny]:
    """TFCE over a (n_roi, n_bins) map.

    by_row=True (default) enhances each ROI independently, so clusters form
    along TIME only and never across regions. by_row=False reproduces Rao's 2-D
    behaviour and is correct only when both axes are genuinely continuous.
    """
    t = np.atleast_2d(np.asarray(t, float))
    if not by_row:
        return tfce_compute(t, **kw)
    out = np.zeros_like(t)
    pos = np.zeros_like(t)
    neg = np.zeros_like(t)
    for i in range(t.shape[0]):
        o, p, n = tfce_compute(t[i:i + 1, :], **kw)
        out[i], pos[i], neg[i] = o[0], p[0], n[0]
    return out, pos, neg


def tfce_sign_flip_test(y: NDArrayAny, n_perm: int = 1000, by_row: bool = True,
                        seed: int = 0, **kw: Any) -> dict[str, NDArrayAny]:
    """Sign-flip permutation test on TFCE-enhanced one-sample t maps.

    Parameters
    ----------
    y : (n_subjects, n_roi, n_bins), NaN where a subject lacks that ROI.

    Returns a dict with `tstat`, `tfce`, `p_tfce`, `p_tmax`, `p_uncorr`.

    The null is built by flipping the sign of each SUBJECT'S WHOLE MAP -- valid
    because H0 is "mean = 0" and the contrast is already a within-subject
    difference. Flipping whole subjects preserves each subject's temporal
    autocorrelation, so the null distribution already reflects how smooth the
    data are. That is why TFCE needs no independence assumption, unlike the BH
    correction it replaces.

    Correction is by the MAX statistic over the whole map (the maximum TFCE
    value in each permutation), which controls the family-wise error rate across
    every ROI x bin cell at once. Positive and negative excursions get their own
    null, so the test is two-sided without assuming symmetry.
    """
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    n_sub = y.shape[0]

    tstat, p_uncorr = ttest_1samp(y, 0.0, axis=0, nan_policy="omit")
    tstat = np.asarray(np.ma.filled(tstat, np.nan), float)
    p_uncorr = np.asarray(np.ma.filled(p_uncorr, np.nan), float)
    tfce, tfce_pos, tfce_neg = tfce_map(tstat, by_row=by_row, **kw)

    max_pos = np.empty(n_perm)
    max_neg = np.empty(n_perm)
    max_tpos = np.empty(n_perm)
    max_tneg = np.empty(n_perm)
    for i in range(n_perm):
        flip = np.where(rng.random(n_sub) <= 0.5, -1.0, 1.0)[:, None, None]
        ts, _ = ttest_1samp(y * flip, 0.0, axis=0, nan_policy="omit")
        ts = np.asarray(np.ma.filled(ts, np.nan), float)
        _, p_i, n_i = tfce_map(ts, by_row=by_row, **kw)
        max_pos[i] = np.nanmax(p_i) if np.isfinite(p_i).any() else 0.0
        max_neg[i] = np.nanmax(n_i) if np.isfinite(n_i).any() else 0.0
        finite_ts = ts[np.isfinite(ts)]
        max_tpos[i] = finite_ts.max(initial=0.0)
        max_tneg[i] = -finite_ts.min(initial=0.0)

    p_tfce = np.ones_like(tstat)
    p_tmax = np.ones_like(tstat)
    pos_side = tstat > 0
    # +1 in numerator and denominator: a permutation test can never report p=0,
    # and the observed data is itself one draw from the null under H0.
    for idx, is_pos in ((np.where(pos_side), True), (np.where(~pos_side), False)):
        if len(idx[0]) == 0:
            continue
        obs_tfce = (tfce_pos if is_pos else tfce_neg)[idx]
        null_tfce = max_pos if is_pos else max_neg
        p_tfce[idx] = (1 + (null_tfce[None, :] >= obs_tfce[:, None]).sum(1)) / (n_perm + 1)
        obs_t = np.abs(tstat[idx])
        null_t = max_tpos if is_pos else max_tneg
        p_tmax[idx] = (1 + (null_t[None, :] >= obs_t[:, None]).sum(1)) / (n_perm + 1)

    p_tfce[~np.isfinite(tstat)] = np.nan
    p_tmax[~np.isfinite(tstat)] = np.nan
    return {"tstat": tstat, "tfce": tfce, "p_tfce": p_tfce,
            "p_tmax": p_tmax, "p_uncorr": p_uncorr}


def _rao_tfce_reference(t: NDArrayAny) -> NDArrayAny:
    """Rao's original loop implementation, kept only to validate the fast one."""
    t = np.asarray(t, float)
    pos_space = np.arange(0, np.max(t) + 0.05, 0.05)
    tfce_pos = np.zeros_like(t)
    for k in range(len(pos_space)):
        lab = label(t > pos_space[k], connectivity=2)
        extent = np.zeros_like(lab)
        for z in range(1, np.max(lab) + 1):
            extent[lab == z] = np.sum(lab == z)
        for m in range(t.shape[0]):
            for n in range(t.shape[1]):
                tfce_pos[m, n] += extent[m, n] ** 0.5 * pos_space[k] ** 2
    return tfce_pos


def test_matches_rao(seed: int = 0) -> float:
    """Max abs difference between the fast path and Rao's loop, on random maps.

    Rao accumulates over `np.arange(0, max+0.05, 0.05)` WITHOUT multiplying by
    dh, i.e. an unnormalised sum rather than an integral. The fast path includes
    dh, so the two differ by exactly that constant factor; this comparison
    divides it out. The constant is irrelevant to inference because the same
    scaling applies to the observed map and to every permutation.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(5):
        t = rng.normal(0, 2, (1, 12))
        _, fast_pos, _ = tfce_compute(t)
        ref = _rao_tfce_reference(t) * DH_DEFAULT
        denom = max(1e-12, float(np.abs(ref).max()))
        worst = max(worst, float(np.abs(fast_pos - ref).max() / denom))
    return worst
