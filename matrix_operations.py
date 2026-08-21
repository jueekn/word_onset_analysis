"""matrix_operations.py — pure-numpy helpers for FC matrices.

Symmetry / triangle extraction / sorting / xarray-metadata checks. Used
throughout the FC compute and aggregation paths. All numpy-typed; the
xarray helpers (`sort_multi_index_coord`, `compare_xarray_metadata`) accept
DataArray instances directly.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr  # pyright: ignore[reportMissingTypeStubs]

NDArrayAny = npt.NDArray[Any]


def any_finite(x: NDArrayAny) -> bool:
    """True if ANY value in x is finite (not NaN/Inf), else False."""
    return bool(np.any(np.isfinite(x)))


def all_finite(x: NDArrayAny) -> bool:
    """True if ALL values in x are finite (not NaN/Inf), else False."""
    return bool(np.all(np.isfinite(x)))


def get_fraction_finite(x: NDArrayAny) -> float:
    """Fraction of values in x that are finite (not NaN/Inf).

    Note: name says 'count' historically (see code_issues #42); actual
    behavior is fraction-of-total. The contract here matches the
    implementation.
    """
    # numpy stubs make np.sum / np.prod overload-ambiguous on our generic
    # input shape. Keep the division inside numpy so the empty-array
    # behavior (0/0 -> NaN with RuntimeWarning) is preserved — pulling
    # operands out to Python int would convert that to ZeroDivisionError.
    return float(np.sum(np.isfinite(x)) / np.prod(x.shape))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def finitize(x: NDArrayAny) -> NDArrayAny:
    """Return the unraveled subset of x containing only finite values."""
    return x[np.isfinite(x)]


def symmetrize(mx: NDArrayAny) -> NDArrayAny:
    """Symmetrize a matrix whose upper or lower triangle is NaN.

    Computes nan-mean of mx and its transpose, filling NaN entries from the
    populated triangle.
    """
    return np.nanmean([mx, np.swapaxes(mx, 0, 1)], axis=0)


def apply_function_expand_dims(
    data: NDArrayAny, func: Callable[[Any], NDArrayAny]
) -> NDArrayAny:
    """Apply `func` element-wise; `func` returns a 1D vector per element.

    Returns an ndarray with one extra trailing dimension equal to the
    returned-vector length. Restricted to numpy.ndarray (code_issues #64:
    xarray branch was unused).
    """
    # Runtime defensive check despite the type annotation — callers from
    # untyped scripts can still pass non-ndarrays.
    if not isinstance(data, np.ndarray):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"apply_function_expand_dims: data must be numpy.ndarray, "
            f"got {type(data).__name__}."
        )
    vector_length = len(func(data.flat[0]))
    result_array = np.apply_along_axis(
        lambda x: func(x[0]), -1, data[..., None])  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    return result_array.reshape(*data.shape, vector_length)


def is_square(matrix: NDArrayAny, require: bool = False) -> bool:
    """True if matrix is 2D and shape[0] == shape[1]. If require, raises on False."""
    cond = len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    if require and not cond:
        raise ValueError(f'Matrix of shape {matrix.shape} is not square!')
    return cond


def is_symmetric(matrix: NDArrayAny, require: bool = False) -> bool:
    """True if matrix ≈ matrix.T within numpy float tolerance. If require, raises on False.

    TODO(types): see code_issues — on non-square input matrix.T differs from
    matrix in shape and the subtraction raises before the check; the
    intended contract was likely to return False on non-square.
    """
    # running into floating point issues just comparing matrix and matrix.T
    cond = bool(np.isclose(matrix - matrix.T, np.zeros_like(matrix)).all())
    if require and not cond:
        raise ValueError(f'Matrix is not symmetric!')
    return cond


def is_positive_definite(matrix: NDArrayAny, require: bool = False) -> bool:
    """True if matrix has a Cholesky factorization (and thus is PD).

    Note: numpy.linalg.cholesky reads only the lower triangle and treats
    its input as Hermitian. This function does NOT symmetry-guard the
    input, so a non-Hermitian matrix can be silently accepted as PD.
    """
    is_square(matrix, require=True)
    try:
        np.linalg.cholesky(matrix)
        cond = True
    except np.linalg.LinAlgError:
        cond = False
    if require and not cond:
        raise ValueError(f'Matrix is not positive definite!')
    return cond


def strict_triu(arr: NDArrayAny) -> NDArrayAny:
    """Upper-triangular part of arr with the diagonal zeroed out."""
    arr = np.triu(arr)
    np.fill_diagonal(arr, 0)
    return arr


def upper_tri_values(
    matrix: NDArrayAny, include_diagonal: bool = True
) -> NDArrayAny:
    """1D array of upper-triangular values of a square matrix.

    Args:
        matrix: square 2D ndarray.
        include_diagonal: if True, include the diagonal; if False, k=1 offset.
    """
    if include_diagonal:
        return matrix[np.triu_indices(matrix.shape[0])]
    else:
        return matrix[np.triu_indices(matrix.shape[0], k=1)]


def off_diagonal_values(matrix: NDArrayAny) -> NDArrayAny:
    """1D array of off-diagonal values of a square matrix."""
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def apply_indexing(
    arr: NDArrayAny, indices: NDArrayAny, axis: int
) -> NDArrayAny:
    """Apply 1D fancy indexing to arr along the given axis (returns a view/copy)."""
    all_indices: list[Any] = [slice(None)] * arr.ndim
    all_indices[axis] = indices
    return arr[tuple(all_indices)]


def sort_array_across_order(
    arr: NDArrayAny,
    order: NDArrayAny | Sequence[Any],
    axis: int | Sequence[int] = 0,
    invert_sort: bool = False,
) -> NDArrayAny:
    """Sort arr by the rank-order of `order` along one or more axes.

    Args:
        arr: ndarray to sort.
        order: 1D sequence of ranking values; arr is reordered to ascend
            in `order`.
        axis: int or sequence of ints; reorder along each.
        invert_sort: if True, apply the inverse permutation (undoing a prior
            forward sort).

    Note: np.argsort default kind is 'quicksort' (unstable). For duplicate
    keys, tie order is not specified.
    """
    if isinstance(order, list):
        order = np.array(order)
    assert isinstance(order, np.ndarray) and len(order.shape) == 1
    sort_idx = np.argsort(order)
    if invert_sort:
        sort_idx = np.argsort(sort_idx)
    axes: Sequence[int] = [axis] if isinstance(axis, int) else axis
    for ax in axes:
        arr = apply_indexing(arr, sort_idx, ax)
    return arr


def sort_multi_index_coord(da: xr.DataArray, dim: str, level: str) -> xr.DataArray:
    """Sort an xarray DataArray along a multi-index coordinate.

    Args:
        da: DataArray with a multi-index on `dim`.
        dim: the dimension carrying the multi-index.
        level: the multi-index level (column name) to sort by.
    """
    sorted_indices = np.array(np.argsort(da[level]))
    return da.isel({dim: sorted_indices})


def compare_xarray_metadata(da1: xr.DataArray, da2: xr.DataArray) -> bool:
    """True if two DataArrays match in shape, dims, coords, and attrs.

    Values are NOT compared — only metadata. Used as a precondition check
    in code that aligns two arrays by coordinate before doing arithmetic.
    """
    if da1.shape != da2.shape:
        return False
    if da1.dims != da2.dims:
        return False
    # xarray's DataArrayCoordinates is typed with Unknown; library-stub
    # limitation, not a real type bug.
    if da1.coords.keys() != da2.coords.keys():  # pyright: ignore[reportUnknownMemberType]
        return False
    for coord_name in da1.coords:  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if not da1[coord_name].equals(da2[coord_name]):
            return False
    if da1.attrs != da2.attrs:
        return False
    return True
