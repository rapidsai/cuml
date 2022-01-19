#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# distutils: language = c++

import numpy as np
import warnings
import math
import inspect

from numba import cuda
from cupy import linalg
import cupy as cp
from cupyx import lapack, geterr, seterr
from cuml import Handle
from cuml.common.array import CumlArray
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.base import Base
from cuml.common.mixins import RegressorMixin
from cuml.common.doc_utils import generate_docstring
from cuml.common import input_to_cuml_array

from sklearn.metrics.pairwise import pairwise_kernels


@cuda.jit(device=True)
def linear_kernel(x, y):
    sum = 0.0
    for i in range(len(x)):
        sum += x[i] * y[i]
    return sum


@cuda.jit(device=True)
def additive_chi2_kernel(x, y):
    res = 0.0
    for i in range(len(x)):
        denom = x[i] - y[i]
        nom = x[i] + y[i]
        if nom != 0.0:
            res += denom * denom / nom
    return -res


@cuda.jit(device=True)
def chi2_kernel(x, y, gamma=1.0):
    if gamma is None:
        gamma = 1.0
    k = additive_chi2_kernel(x, y)
    k *= gamma
    return math.exp(k)


@cuda.jit(device=True)
def cosine_similarity(x, y):
    z = linear_kernel(x, y)
    x_norm = 0.0
    y_norm = 0.0
    for i in range(len(x)):
        x_norm += x[i] * x[i]
        y_norm += y[i] * y[i]
    return z / math.sqrt(x_norm * y_norm)


@cuda.jit(device=True)
def laplacian_kernel(x, y, gamma=None):
    if gamma is None:
        gamma = 1.0 / len(x)
    manhattan = 0.0
    for i in range(len(x)):
        manhattan += abs(x[i] - y[i])
    return math.exp(-gamma * manhattan)


@cuda.jit(device=True)
def polynomial_kernel(x, y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / len(x)
    return (gamma * linear_kernel(x, y) + coef0) ** degree


@cuda.jit(device=True)
def rbf_kernel(x, y, gamma=None):
    if gamma is None:
        gamma = 1.0 / len(x)
    sum = 0.0
    for i in range(len(x)):
        sum += (x[i] - y[i]) ** 2
    return math.exp(-gamma * sum)


@cuda.jit(device=True)
def sigmoid_kernel(x, y, gamma=None, coef0=1.0):
    if gamma is None:
        gamma = 1.0 / len(x)
    return math.tanh(gamma * linear_kernel(x, y) + coef0)


PAIRWISE_KERNEL_FUNCTIONS = {
    "linear": linear_kernel,
    "additive_chi2": additive_chi2_kernel,
    "chi2": chi2_kernel,
    "cosine": cosine_similarity,
    "laplacian": laplacian_kernel,
    "polynomial": polynomial_kernel,
    "poly": polynomial_kernel,
    "rbf": rbf_kernel,
    "sigmoid": sigmoid_kernel,
}


def _solve_cholesky_kernel(K, y, alpha, sample_weight=None, copy=False):
    # dual_coef = inv(X X^t + alpha*Id) y
    n_samples = K.shape[0]
    n_targets = y.shape[1]

    if copy:
        K = K.copy()

    alpha = cp.atleast_1d(alpha)
    one_alpha = (alpha == alpha[0]).all()
    has_sw = sample_weight not in [1.0, None]

    if has_sw:
        # Unlike other solvers, we need to support sample_weight directly
        # because K might be a pre-computed kernel.
        sw = cp.sqrt(cp.atleast_1d(sample_weight))
        y = y * sw[:, cp.newaxis]
        K *= cp.outer(sw, sw)

    if one_alpha:
        # Only one penalty, we can solve multi-target problems in one time.
        K.flat[:: n_samples + 1] += alpha[0]

        try:
            # we need to set the error mode of cupy to raise
            # otherwise we silently get an array of NaNs
            err_mode = geterr()["linalg"]
            seterr(linalg="raise")
            dual_coef = lapack.posv(K, y)
            seterr(linalg=err_mode)
        except Exception as err:
            warnings.warn(
                "Singular matrix in solving dual problem. Using "
                "least-squares solution instead."
            )
            dual_coef = linalg.lstsq(K, y, rcond=None)[0]

        # K is expensive to compute and store in memory so change it back in
        # case it was user-given.
        K.flat[:: n_samples + 1] -= alpha[0]

        if has_sw:
            dual_coef *= sw[:, np.newaxis]

        return dual_coef
    else:
        # One penalty per target. We need to solve each target separately.
        dual_coefs = np.empty([n_targets, n_samples], K.dtype)

        for dual_coef, target, current_alpha in zip(dual_coefs, y.T, alpha):
            K.flat[:: n_samples + 1] += current_alpha

            dual_coef[:] = linalg.solve(
                K, target, sym_pos=True, overwrite_a=False
            ).ravel()

            K.flat[:: n_samples + 1] -= current_alpha

        if has_sw:
            dual_coefs *= sw[np.newaxis, :]

        return dual_coefs.T


# Check if we have a valid kernel function correctly specified arguments
# Returns keyword arguments formed as a tuple (numba kernels cannot deal with kwargs as a dict)
def _validate_kernel_function(func, filter_params=False, **kwds):
    if not hasattr(func, "py_func"):
        raise TypeError("Kernel function should be a numba device function.")

    # get all the possible extra function arguments, excluding x, y
    all_func_kwargs = list(inspect.signature(func.py_func).parameters.values())
    if len(all_func_kwargs) < 2:
        raise ValueError("Expected at least two arguments to kernel function.")

    all_func_kwargs = all_func_kwargs[2:]
    if any(p.default is inspect.Parameter.empty for p in all_func_kwargs):
        raise ValueError("Extra kernel parameters must be passed as keyword arguments.")
    all_func_kwargs = [(k.name, k.default) for k in all_func_kwargs]
    if all_func_kwargs and not filter_params:
        # kwds must occur in the function signature
        available_kwds = set(list(zip(*all_func_kwargs))[0])
        # is kwds a subset of the valid func keyword arguments?
        if not set(kwds.keys()) <= available_kwds:
            raise ValueError("kwds contains arguments not used by kernel function")

    filtered_kwds_tuple = tuple(
        kwds[k] if k in kwds.keys() else v for (k, v) in all_func_kwargs
    )
    return filtered_kwds_tuple


_kernel_cache = {}


def pairwise_kernels(X, Y=None, metric="linear", *, filter_params=False, **kwds):
    """Compute the kernel between arrays X and optional array Y.
    This method takes either a vector array or a kernel matrix, and returns
    a kernel matrix. If the input is a vector array, the kernels are
    computed. If the input is a kernel matrix, it is returned instead.
    This method provides a safe way to take a kernel matrix as input, while
    preserving compatibility with many other algorithms that take a vector
    array.
    If Y is given (default is None), then the returned matrix is the pairwise
    kernel between the arrays from both X and Y.
    Valid values for metric are:
        ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf',
        'laplacian', 'sigmoid', 'cosine']
    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_features)
        Array of pairwise kernels between samples, or a feature array.
        The shape of the array should be (n_samples_X, n_samples_X) if
        metric == "precomputed" and (n_samples_X, n_features) otherwise.
    Y : ndarray of shape (n_samples_Y, n_features), default=None
        A second feature array only if X has shape (n_samples_X, n_features).
    metric : str or callable (numba device function), default="linear"
        The metric to use when calculating kernel between instances in a
        feature array.
        If metric is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two rows from X as input and return the corresponding
        kernel value as a single number.
    filter_params : bool, default=False
        Whether to filter invalid parameters or not.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the kernel function.
    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_samples_Y)
        A kernel matrix K such that K_{i, j} is the kernel between the
        ith and jth vectors of the given matrix X, if Y is None.
        If Y is not None, then K_{i, j} is the kernel between the ith array
        from X and the jth array from Y.
    Notes
    -----
    If metric is 'precomputed', Y is ignored and X is returned.
    """
    if metric == "precomputed":
        return X
    elif metric in PAIRWISE_KERNEL_FUNCTIONS:
        func = PAIRWISE_KERNEL_FUNCTIONS[metric]
    elif isinstance(metric, str):
        raise ValueError("Unknown kernel %r" % metric)
    else:
        func = metric

    filtered_kwds_tuple = _validate_kernel_function(func, filter_params, **kwds)

    def evaluate_pairwise_kernels(X, Y, K):
        idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        X_m = X.shape[0]
        Y_m = Y.shape[0]
        row = idx // Y_m
        col = idx % Y_m
        if idx < X_m * Y_m:
            if X is Y and row <= col:
                # matrix is symmetric, reuse half the evaluations
                k = func(X[row], Y[col], *filtered_kwds_tuple)
                K[row, col] = k
                K[col, row] = k
            else:
                k = func(X[row], Y[col], *filtered_kwds_tuple)
                K[row, col] = k

    if Y is None:
        Y = X
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y have different dimensions.")

    threadsperblock = 256
    blockspergrid = (X.shape[0] * Y.shape[0] + (threadsperblock - 1)) // threadsperblock

    # Here we force K to use 64 bit, even if the input is 32 bit
    # 32 bit K results in serious numerical stability problems
    K = cp.zeros((X.shape[0], Y.shape[0]), dtype=np.float64)

    key = (metric, filtered_kwds_tuple,X.dtype, Y.dtype)
    if key in _kernel_cache:
        compiled_kernel = _kernel_cache[key]
    else:
        compiled_kernel = cuda.jit(evaluate_pairwise_kernels)
        _kernel_cache[key] = compiled_kernel
    compiled_kernel[blockspergrid, threadsperblock](X, Y, K)
    return K


class KernelRidge(Base, RegressorMixin):
    dual_coef_ = CumlArrayDescriptor()
    X_fit_ = CumlArrayDescriptor()

    def __init__(
        self,
        *,
        alpha=1,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        handle=None,
        output_type=None,
        verbose=False
    ):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def _get_kernel(self, X, Y=None):
        if isinstance(self.kernel,str):
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        else:
            params = self.kernel_params or {}
        return pairwise_kernels(X, metric=self.kernel, filter_params=True, **params)

    @generate_docstring()
    def fit(self, X, y, sample_weight=None, convert_dtype=True) -> "KernelRidge":
        """
        Fit the model with X and y.

        """

        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True

        X_m, n_rows, self.n_cols, self.dtype = input_to_cuml_array(
            X, check_dtype=[np.float32, np.float64]
        )

        y_m, _, _, _ = input_to_cuml_array(
            y,
            check_dtype=self.dtype,
            convert_to_dtype=(self.dtype if convert_dtype else None),
            check_rows=n_rows,
        )

        if self.n_cols < 1:
            msg = "X matrix must have at least a column"
            raise TypeError(msg)

        K = self._get_kernel(X_m, self.kernel)
        copy = self.kernel == "precomputed"
        self.dual_coef_ = _solve_cholesky_kernel(
            K, cp.asarray(y_m), self.alpha, sample_weight, copy
        )

        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()
        self.X_fit_ = X_m
        return self

    def predict(self, X):
        """Predict using the kernel ridge model.
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Samples. If kernel == "precomputed" this is instead a
                precomputed kernel matrix, shape = [n_samples,
                n_samples_fitted], where n_samples_fitted is the number of
                samples used in the fitting for this estimator.
            Returns
            -------
            C : array of shape (n_samples,) or (n_samples, n_targets)
                Returns predicted values.
            """
        X_m, _, _, _ = input_to_cuml_array(
            X, check_dtype=[np.float32, np.float64]
        )

        K = self._get_kernel(X_m, self.X_fit_)
        return cp.dot(K, self.dual_coef_)

