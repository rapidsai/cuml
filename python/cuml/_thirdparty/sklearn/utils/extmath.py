"""
Extended math utilities.
"""
# Original authors from Sckit-Learn:
#          Gael Varoquaux
#          Alexandre Gramfort
#          Alexandre T. Passos
#          Olivier Grisel
#          Lars Buitinck
#          Stefan van der Walt
#          Kyle Kastner
#          Giorgio Patrini
# License: BSD 3 clause


# This code originates from the Scikit-Learn library,
# it was since modified to allow GPU acceleration.
# This code is under BSD 3 clause license.
# Authors mentioned above do not endorse or promote this production.


import cupy as np
import cupyx
from cupy import sparse

from ....internals import _deprecate_pos_args


def row_norms(X, squared=False):
    """Row-wise (squared) Euclidean norm of X.

    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.

    Performs no input validation.

    Parameters
    ----------
    X : array_like
        The input array
    squared : bool, optional (default = False)
        If True, return squared norms.

    Returns
    -------
    array_like
        The row-wise (squared) Euclidean norm of X.
    """
    if sparse.issparse(X):
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)
        # norms = csr_row_norms(X)
    else:
        norms = np.einsum('ij,ij->i', X, X)

    if not squared:
        np.sqrt(norms, norms)
    return norms


def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
    """Calculate mean update and a Youngs and Cramer variance update.

    last_mean and last_variance are statistics computed at the last step by the
    function. Both must be initialized to 0.0. In case no scaling is required
    last_variance can be None. The mean is always required and returned because
    necessary for the calculation of the variance. last_n_samples_seen is the
    number of samples encountered until now.

    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to use for variance update

    last_mean : array-like, shape: (n_features,)

    last_variance : array-like, shape: (n_features,)

    last_sample_count : array-like, shape (n_features,)

    Returns
    -------
    updated_mean : array, shape (n_features,)

    updated_variance : array, shape (n_features,)
        If None, only mean is computed

    updated_sample_count : array, shape (n_features,)

    Notes
    -----
    NaNs are ignored during the algorithm.

    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247

    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    new_sum = _safe_accumulator_op(np.nansum, X, axis=0)

    new_sample_count = np.sum(~np.isnan(X), axis=0)
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        new_unnormalized_variance = (
            _safe_accumulator_op(np.nanvar, X, axis=0) * new_sample_count)
        last_unnormalized_variance = last_variance * last_sample_count

        with cupyx.errstate(divide=None, invalid=None):
            last_over_new_count = last_sample_count / new_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance + new_unnormalized_variance +
                last_over_new_count / updated_sample_count *
                (last_sum / last_over_new_count - new_sum) ** 2)

        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count


@_deprecate_pos_args(version="0.20")
def weighted_mode(a, w, *, axis=0):
    """Returns an array of the weighted modal (most common) value in a

    If there is more than one such value, only the first is returned.
    The bin-count for the modal bins is also returned.

    This is an extension of the algorithm in scipy.stats.mode.

    Parameters
    ----------
    a : array_like
        n-dimensional array of which to find mode(s).
    w : array_like
        n-dimensional array of weights for each value
    axis : int, optional
        Axis along which to operate. Default is 0, i.e. the first axis.

    Returns
    -------
    vals : ndarray
        Array of modal values.
    score : ndarray
        Array of weighted counts for each mode.

    Examples
    --------
    >>> from sklearn.utils.extmath import weighted_mode
    >>> x = [4, 1, 4, 2, 4, 2]
    >>> weights = [1, 1, 1, 1, 1, 1]
    >>> weighted_mode(x, weights)
    (array([4.]), array([3.]))

    The value 4 appears three times: with uniform weights, the result is
    simply the mode of the distribution.

    >>> weights = [1, 3, 0.5, 1.5, 1, 2]  # deweight the 4's
    >>> weighted_mode(x, weights)
    (array([2.]), array([3.5]))

    The value 2 has the highest score: it appears twice with weights of
    1.5 and 2: the sum of these is 3.5.

    See Also
    --------
    scipy.stats.mode
    """
    if axis is None:
        a = np.ravel(a)
        w = np.ravel(w)
        axis = 0
    else:
        a = np.asarray(a)
        w = np.asarray(w)

    if a.shape != w.shape:
        w = np.full(a.shape, w, dtype=w.dtype)

    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)
    for score in scores:
        template = np.zeros(a.shape)
        ind = (a == score)
        template[ind] = w[ind]
        counts = np.expand_dims(np.sum(template, axis), axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent
    return mostfrequent, oldcounts


# Use at least float64 for the accumulating functions to avoid precision issue
# see https://github.com/numpy/numpy/issues/9393. The float64 is also retained
# as it is in case the float overflows
def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.

    Parameters
    ----------
    op : function
        A numpy accumulator function such as np.mean or np.sum
    x : numpy array
        A numpy array to apply the accumulator function
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function

    Returns
    -------
    result : The output of the accumulator function passed to this function
    """
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result
