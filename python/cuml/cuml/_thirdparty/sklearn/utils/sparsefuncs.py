# Original authors from Sckit-Learn:
#          Manoj Kumar
#          Thomas Unterthiner
#          Giorgio Patrini
#
# License: BSD 3 clause


# This code originates from the Scikit-Learn library,
# it was since modified to allow GPU acceleration.
# This code is under BSD 3 clause license.
# Authors mentioned above do not endorse or promote this production.

import cupy as np
import numpy as cpu_np
import scipy.sparse as cpu_sp
from cupyx.scipy import sparse as gpu_sp

from ....thirdparty_adapters.sparsefuncs_fast import (
    csc_mean_variance_axis0 as _csc_mean_var_axis0,
)
from ....thirdparty_adapters.sparsefuncs_fast import (
    csr_mean_variance_axis0 as _csr_mean_var_axis0,
)


def iscsr(X):
    return isinstance(X, cpu_sp.csr_matrix) \
        or isinstance(X, gpu_sp.csr_matrix)


def iscsc(X):
    return isinstance(X, cpu_sp.csc_matrix) \
        or isinstance(X, gpu_sp.csc_matrix)


def issparse(X):
    return iscsr(X) or iscsc(X)


def _raise_typeerror(X):
    """Raises a TypeError if X is not a CSR or CSC matrix"""
    input_type = X.format if issparse(X) else type(X)
    err = "Expected a CSR or CSC sparse matrix, got %s." % input_type
    raise TypeError(err)


def _raise_error_wrong_axis(axis):
    if axis not in (0, 1):
        raise ValueError(
            "Unknown axis value: %d. Use 0 for rows, or 1 for columns" % axis)


def inplace_csr_column_scale(X, scale):
    """Inplace column scaling of a CSR matrix.

    Scale each feature of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : CSR matrix with shape (n_samples, n_features)
        Matrix to normalize using the variance of the features.

    scale : float array with shape (n_features,)
        Array of precomputed feature-wise values to use for scaling.
    """
    assert scale.shape[0] == X.shape[1]
    indices_copy = X.indices.copy()
    indices_copy[indices_copy >= X.shape[1]] = X.shape[1] - 1
    X.data *= scale.take(indices_copy)


def inplace_csr_row_scale(X, scale):
    """ Inplace row scaling of a CSR matrix.

    Scale each sample of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : CSR sparse matrix, shape (n_samples, n_features)
        Matrix to be scaled.

    scale : float array with shape (n_samples,)
        Array of precomputed sample-wise values to use for scaling.
    """
    assert scale.shape[0] == X.shape[0]
    X.data *= np.repeat(scale, np.diff(X.indptr).tolist())


def inplace_column_scale(X, scale):
    """Inplace column scaling of a CSC/CSR matrix.

    Scale each feature of the data matrix by multiplying with specific scale
    provided by the caller assuming a (n_samples, n_features) shape.

    Parameters
    ----------
    X : CSC or CSR matrix with shape (n_samples, n_features)
        Matrix to normalize using the variance of the features.

    scale : float array with shape (n_features,)
        Array of precomputed feature-wise values to use for scaling.
    """
    if iscsc(X):
        inplace_csr_row_scale(X.T, scale)
    elif iscsr(X):
        inplace_csr_column_scale(X, scale)
    else:
        _raise_typeerror(X)


def mean_variance_axis(X, axis):
    """Compute mean and variance along an axix on a CSR or CSC matrix

    Parameters
    ----------
    X : CSR or CSC sparse matrix, shape (n_samples, n_features)
        Input data.

    axis : int (either 0 or 1)
        Axis along which the axis should be computed.

    Returns
    -------

    means : float array with shape (n_features,)
        Feature-wise means

    variances : float array with shape (n_features,)
        Feature-wise variances

    """
    _raise_error_wrong_axis(axis)

    if iscsr(X):
        if axis == 0:
            return _csr_mean_var_axis0(X)
        else:
            return _csc_mean_var_axis0(X.T)
    elif iscsc(X):
        if axis == 0:
            return _csc_mean_var_axis0(X)
        else:
            return _csr_mean_var_axis0(X.T)
    else:
        _raise_typeerror(X)


ufunc_dic = {
    'min': np.min,
    'max': np.max,
    'nanmin': np.nanmin,
    'nanmax': np.nanmax
}


def _minor_reduce(X, min_or_max):
    fminmax = ufunc_dic[min_or_max]

    major_index = np.flatnonzero(np.diff(X.indptr))
    values = cpu_np.zeros(major_index.shape[0], dtype=X.dtype)
    ptrs = X.indptr[major_index]

    start = ptrs[0]
    for i, end in enumerate(ptrs[1:]):
        values[i] = fminmax(X.data[start:end])
        start = end
    values[-1] = fminmax(X.data[end:])

    return major_index, np.array(values)


def _min_or_max_axis(X, axis, min_or_max):
    N = X.shape[axis]
    if N == 0:
        raise ValueError("zero-size array to reduction operation")
    M = X.shape[1 - axis]
    mat = X.tocsc() if axis == 0 else X.tocsr()
    mat.sum_duplicates()
    major_index, value = _minor_reduce(mat, min_or_max)
    not_full = np.diff(mat.indptr)[major_index] < N
    if 'min' in min_or_max:
        fminmax = np.fmin
    else:
        fminmax = np.fmax
    is_nan = np.isnan(value)
    value[not_full] = fminmax(value[not_full], 0)
    if 'nan' not in min_or_max:
        value[is_nan] = np.nan
    mask = value != 0
    major_index = np.compress(mask, major_index)
    value = np.compress(mask, value)

    if axis == 0:
        res = gpu_sp.coo_matrix((value, (np.zeros(len(value)), major_index)),
                                dtype=X.dtype, shape=(1, M))
    else:
        res = gpu_sp.coo_matrix((value, (major_index, np.zeros(len(value)))),
                                dtype=X.dtype, shape=(M, 1))
    return res.A.ravel()


def _sparse_min_or_max(X, axis, min_or_max):
    if axis is None:
        if 0 in X.shape:
            raise ValueError("zero-size array to reduction operation")
        if X.nnz == 0:
            return X.dtype.type(0)
        fminmax = ufunc_dic[min_or_max]
        m = fminmax(X.data)
        if np.isnan(m):
            if 'nan' in min_or_max:
                m = 0
        elif X.nnz != cpu_np.prod(X.shape):
            if 'min' in min_or_max:
                m = m if m <= 0 else 0
            else:
                m = m if m >= 0 else 0
        return X.dtype.type(m)
    if axis < 0:
        axis += 2
    if (axis == 0) or (axis == 1):
        return _min_or_max_axis(X, axis, min_or_max)
    else:
        raise ValueError("invalid axis, use 0 for rows, or 1 for columns")


def _sparse_min_max(X, axis):
    return (_sparse_min_or_max(X, axis, 'min'),
            _sparse_min_or_max(X, axis, 'max'))


def _sparse_nan_min_max(X, axis):
    return(_sparse_min_or_max(X, axis, 'nanmin'),
           _sparse_min_or_max(X, axis, 'nanmax'))


def min_max_axis(X, axis, ignore_nan=False):
    """Compute minimum and maximum along an axis on a CSR or CSC matrix and
    optionally ignore NaN values.

    Parameters
    ----------
    X : CSR or CSC sparse matrix, shape (n_samples, n_features)
        Input data.

    axis : int (either 0 or 1)
        Axis along which the axis should be computed.

    ignore_nan : bool, default is False
        Ignore or passing through NaN values.

    Returns
    -------

    mins : float array with shape (n_features,)
        Feature-wise minima

    maxs : float array with shape (n_features,)
        Feature-wise maxima
    """
    if issparse(X):
        if ignore_nan:
            return _sparse_nan_min_max(X, axis=axis)
        else:
            return _sparse_min_max(X, axis=axis)
    else:
        _raise_typeerror(X)
