#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
import math
import inspect
from numba import cuda
import cupy as cp
import numpy as np
import cuml.internals


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


def _validate_kernel_function(func, filter_params=False, **kwds):
    # Check if we have a valid kernel function correctly specified arguments
    # Returns keyword arguments formed as a tuple
    # (numba kernels cannot deal with kwargs as a dict)
    if not hasattr(func, "py_func"):
        raise TypeError("Kernel function should be a numba device function.")

    # get all the possible extra function arguments, excluding x, y
    all_func_kwargs = list(inspect.signature(
        func.py_func).parameters.values())
    if len(all_func_kwargs) < 2:
        raise ValueError(
            "Expected at least two arguments to kernel function.")

    all_func_kwargs = all_func_kwargs[2:]
    if any(p.default is inspect.Parameter.empty for p in all_func_kwargs):
        raise ValueError(
            "Extra kernel parameters must be passed as keyword arguments.")
    all_func_kwargs = [(k.name, k.default) for k in all_func_kwargs]
    if all_func_kwargs and not filter_params:
        # kwds must occur in the function signature
        available_kwds = set(list(zip(*all_func_kwargs))[0])
        # is kwds a subset of the valid func keyword arguments?
        if not set(kwds.keys()) <= available_kwds:
            raise ValueError(
                "kwds contains arguments not used by kernel function")

    filtered_kwds_tuple = tuple(
        kwds[k] if k in kwds.keys() else v for (k, v) in all_func_kwargs
    )
    return filtered_kwds_tuple


_kernel_cache = {}


@cuml.internals.api_return_array(get_output_type=True)
def pairwise_kernels(X, Y=None, metric="linear", *,
                     filter_params=False, convert_dtype=True, **kwds):
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
    X : Dense matrix (device or host) of shape (n_samples_X, n_samples_X) or \
            (n_samples_X, n_features)
        Array of pairwise kernels between samples, or a feature array.
        The shape of the array should be (n_samples_X, n_samples_X) if
        metric == "precomputed" and (n_samples_X, n_features) otherwise.
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy
    Y : Dense matrix (device or host) of shape (n_samples_Y, n_features),
        default=None
        A second feature array only if X has shape (n_samples_X, n_features).
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy
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
    convert_dtype : bool, optional (default = True)
        When set to True, the method will, when necessary, convert
        Y to be the same data type as X if they differ. This
        will increase memory used for the method.
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

    Examples
    --------

    .. code-block:: python

        import cupy as cp
        from cuml.metrics import pairwise_kernels
        from numba import cuda
        import math

        X = cp.array([[2, 3], [3, 5], [5, 8]])
        Y = cp.array([[1, 0], [2, 1]])

        pairwise_kernels(X, Y, metric='linear')

        @cuda.jit(device=True)
        def custom_rbf_kernel(x, y, gamma=None):
            if gamma is None:
                gamma = 1.0 / len(x)
            sum = 0.0
            for i in range(len(x)):
                sum += (x[i] - y[i]) ** 2
            return math.exp(-gamma * sum)

        pairwise_kernels(X, Y, metric=custom_rbf_kernel)

    """
    if metric == "precomputed":
        return X
    elif metric in PAIRWISE_KERNEL_FUNCTIONS:
        func = PAIRWISE_KERNEL_FUNCTIONS[metric]
    elif isinstance(metric, str):
        raise ValueError("Unknown kernel %r" % metric)
    else:
        func = metric

    filtered_kwds_tuple = _validate_kernel_function(
        func, filter_params, **kwds)

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
    blockspergrid = (X.shape[0] * Y.shape[0] +
                     (threadsperblock - 1)) // threadsperblock

    # Here we force K to use 64 bit, even if the input is 32 bit
    # 32 bit K results in serious numerical stability problems
    K = cp.zeros((X.shape[0], Y.shape[0]), dtype=np.float64)

    key = (metric, filtered_kwds_tuple, X.dtype, Y.dtype)
    if key in _kernel_cache:
        compiled_kernel = _kernel_cache[key]
    else:
        compiled_kernel = cuda.jit(evaluate_pairwise_kernels)
        _kernel_cache[key] = compiled_kernel
    compiled_kernel[blockspergrid, threadsperblock](X, Y, K)
    return K
