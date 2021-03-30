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


from scipy import sparse as cpu_sp
from cupy import sparse as gpu_sp
import cupy as np
import numpy as cpu_np

from ....thirdparty_adapters.sparsefuncs_fast import (
    csr_mean_variance_axis0 as _csr_mean_var_axis0,
    csc_mean_variance_axis0 as _csc_mean_var_axis0)


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
    X.indices[X.indices >= X.shape[1]] = X.shape[1] - 1
    X.data *= scale.take(X.indices)


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


def _minor_reduce(X, min_or_max):
    if min_or_max == 'min':
        min_or_max = np.min
    else:
        min_or_max = np.max

    major_index = np.flatnonzero(np.diff(X.indptr))

    # reduceat tries casts X.indptr to intp, which errors
    # if it is int64 on a 32 bit system.
    # Reinitializing prevents this where possible, see #13737
    X = type(X)((X.data, X.indices, X.indptr), shape=X.shape)

    value = cpu_np.zeros(len(X.indptr)-1, dtype=X.dtype)

    start = X.indptr[0]
    for i, end in enumerate(X.indptr[1:]):
        value[i] = min_or_max(X.data[start:end])
        start = end

    value = np.array(value)
    return major_index, value


def _min_or_max_axis(X, axis, min_or_max):
    N = X.shape[axis]
    if N == 0:
        raise ValueError("zero-size array to reduction operation")
    M = X.shape[1 - axis]
    mat = X.tocsc() if axis == 0 else X.tocsr()
    mat.sum_duplicates()
    major_index, value = _minor_reduce(mat, min_or_max)
    not_full = np.diff(mat.indptr)[major_index] < N
    if min_or_max == 'min':
        min_or_max = np.fmin
    else:
        min_or_max = np.fmax
    value[not_full] = min_or_max(value[not_full], 0)
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
        zero = X.dtype.type(0)
        if X.nnz == 0:
            return zero
        if min_or_max == 'min':
            fminmax = np.min
        else:
            fminmax = np.max
        m = fminmax.reduce(X.data.ravel())
        if X.nnz != np.product(X.shape):
            m = fminmax(zero, m)
        return m
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
    return(_sparse_min_or_max(X, axis, 'min'),
           _sparse_min_or_max(X, axis, 'max'))


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
