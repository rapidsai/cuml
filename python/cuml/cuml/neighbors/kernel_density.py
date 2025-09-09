#
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import cupy as cp
import numpy as np
from cupyx.scipy.special import gammainc
from numba import cuda

from cuml.common.exceptions import NotFittedError
from cuml.internals.base import Base
from cuml.internals.input_utils import input_to_cuml_array, input_to_cupy_array
from cuml.metrics import pairwise_distances

VALID_KERNELS = [
    "gaussian",
    "tophat",
    "epanechnikov",
    "exponential",
    "linear",
    "cosine",
]


@cp.fuse()
def gaussian_log_kernel(x, h):
    return -(x * x) / (2 * h * h)


@cp.fuse()
def tophat_log_kernel(x, h):
    """
    if x < h:
        return 0.0
    else:
        return -FLOAT_MIN
    """
    y = (x >= h) * np.finfo(x.dtype).min
    return y


@cp.fuse()
def _epanechnikov_log_kernel(x, h, h_squared):
    # don't call log(0) otherwise we get NaNs
    z = cp.maximum(1.0 - (x * x) / h_squared, 1e-30)
    y = (x < h) * cp.log(z)
    y += (x >= h) * np.finfo(y.dtype).min
    return y


def epanechnikov_log_kernel(x, h):
    # TODO: Due to https://github.com/cupy/cupy/issues/8536 cupy.fuse errors when trying
    # to compile the elementwise operation in epanechnikov. Handling `h * h` on host
    # (where `h` is a host scalar) seems to work around the bug completely. Once the upstream
    # issue is fixed this can be reverted.
    return _epanechnikov_log_kernel(x, h, h * h)


@cp.fuse()
def exponential_log_kernel(x, h):
    return -x / h


@cp.fuse()
def linear_log_kernel(x, h):
    # don't call log(0) otherwise we get NaNs
    z = cp.maximum(1.0 - x / h, 1e-30)
    y = (x < h) * cp.log(z)
    y += (x >= h) * np.finfo(y.dtype).min
    return y


@cp.fuse()
def cosine_log_kernel(x, h):
    # don't call log(0) otherwise we get NaNs
    z = cp.maximum(cp.cos(0.5 * np.pi * x / h), 1e-30)
    y = (x < h) * cp.log(z)
    y += (x >= h) * np.finfo(y.dtype).min
    return y


log_probability_kernels_ = {
    "gaussian": gaussian_log_kernel,
    "tophat": tophat_log_kernel,
    "epanechnikov": epanechnikov_log_kernel,
    "exponential": exponential_log_kernel,
    "linear": linear_log_kernel,
    "cosine": cosine_log_kernel,
}


def logVn(n):
    return 0.5 * n * np.log(np.pi) - math.lgamma(0.5 * n + 1)


def logSn(n):
    return np.log(2 * np.pi) + logVn(n - 1)


def norm_log_probabilities(log_probabilities, kernel, h, d):
    if kernel == "gaussian":
        factor = 0.5 * d * np.log(2 * np.pi)
    elif kernel == "tophat":
        factor = logVn(d)
    elif kernel == "epanechnikov":
        factor = logVn(d) + np.log(2.0 / (d + 2.0))
    elif kernel == "exponential":
        factor = logSn(d - 1) + math.lgamma(d)
    elif kernel == "linear":
        factor = logVn(d) - np.log(d + 1.0)
    elif kernel == "cosine":
        factor = 0.0
        tmp = 2.0 / np.pi
        for k in range(1, d + 1, 2):
            factor += tmp
            tmp *= -(d - k) * (d - k - 1) * (2.0 / np.pi) ** 2
        factor = np.log(factor) + logSn(d - 1)
    else:
        raise ValueError("Unsupported kernel.")

    return log_probabilities - (factor + d * np.log(h))


@cuda.jit()
def logsumexp_kernel(distances, log_probabilities):
    row = cuda.blockIdx.x
    if row >= distances.shape[0]:
        return
    tid = cuda.threadIdx.x
    ncols = distances.shape[1]
    stride = cuda.blockDim.x

    sdata = cuda.shared.array(256, dtype=distances.dtype)

    # phase 1: block-wide max
    local_max = 0.0
    for col in range(tid, ncols, stride):
        val = distances[row, col]
        if val > local_max:
            local_max = val
    sdata[tid] = local_max
    cuda.syncthreads()

    offset = stride // 2
    while offset > 0:
        if tid < offset:
            other = sdata[tid + offset]
            if other > sdata[tid]:
                sdata[tid] = other
        cuda.syncthreads()
        offset //= 2

    block_max = sdata[0]

    # phase 2: block-wide sum of exp(x - max)
    local_sum = 0.0
    for col in range(tid, ncols, stride):
        local_sum += math.exp(distances[row, col] - block_max)
    sdata[tid] = local_sum
    cuda.syncthreads()

    offset = stride // 2
    while offset > 0:
        if tid < offset:
            sdata[tid] += sdata[tid + offset]
        cuda.syncthreads()
        offset //= 2

    if tid == 0:
        log_probabilities[row] = math.log(sdata[0]) + block_max


class KernelDensity(Base):
    """
    Kernel Density Estimation. Computes a non-parametric density estimate
    from a finite data sample, smoothing the estimate according to a
    bandwidth parameter.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth of the kernel.
    kernel : {'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', \
                 'cosine'}, default='gaussian'
        The kernel to use.
    metric : str, default='euclidean'
        The distance metric to use.  Note that not all metrics are
        valid with all algorithms. Note that the normalization of the density
        output is correct only for the Euclidean distance metric. Default
        is 'euclidean'.
    metric_params : dict, default=None
        Additional parameters to be passed to the tree for use with the
        metric.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the
        CUDA stream that will be used for the model's computations, so
        users can run different models concurrently in different streams
        by creating handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Examples
    --------

    .. code-block:: python

        >>> from cuml.neighbors import KernelDensity
        >>> import cupy as cp
        >>> rng = cp.random.RandomState(42)
        >>> X = rng.random_sample((100, 3))
        >>> kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
        >>> log_density = kde.score_samples(X[:3])

    """

    def __init__(
        self,
        *,
        bandwidth=1.0,
        kernel="gaussian",
        metric="euclidean",
        metric_params=None,
        output_type=None,
        handle=None,
        verbose=False,
    ):
        super(KernelDensity, self).__init__(
            verbose=verbose, handle=handle, output_type=output_type
        )
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.metric_params = metric_params

        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if kernel not in VALID_KERNELS:
            raise ValueError("invalid kernel: '{0}'".format(kernel))

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "bandwidth",
            "kernel",
            "metric",
            "metric_params",
        ]

    def fit(self, X, y=None, sample_weight=None, *, convert_dtype=True):
        """Fit the Kernel Density model on the data.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        y : None
            Ignored.
        sample_weight : array-like of shape (n_samples,), default=None
            List of sample weights attached to the data X.

        Returns
        -------

        self : object
            Returns the instance itself.
        """
        if sample_weight is not None:
            self.sample_weight_ = input_to_cupy_array(
                sample_weight,
                convert_to_dtype=(np.float32 if convert_dtype else None),
                check_dtype=[cp.float32, cp.float64],
            ).array
            if self.sample_weight_.min() <= 0:
                raise ValueError("sample_weight must have positive values")
        else:
            self.sample_weight_ = None

        self.X_ = input_to_cupy_array(
            X,
            order="C",
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[cp.float32, cp.float64],
        ).array

        return self

    def score_samples(self, X, *, convert_dtype=True):
        """Compute the log-likelihood of each sample under the model.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        Returns
        -------

        density : ndarray of shape (n_samples,)
            Log-likelihood of each sample in `X`. These are normalized to be
            probability densities, so values will be low for high-dimensional
            data.
        """
        if not hasattr(self, "X_"):
            raise NotFittedError()
        X_cuml = input_to_cuml_array(
            X,
            convert_to_dtype=(np.float32 if convert_dtype else None),
        )
        if self.metric_params:
            if len(self.metric_params) != 1:
                raise ValueError(
                    "Cuml only supports metrics with a single arg."
                )
            metric_arg = list(self.metric_params.values())[0]
            distances = pairwise_distances(
                X_cuml.array,
                self.X_,
                metric=self.metric,
                metric_arg=metric_arg,
            )
        else:
            distances = pairwise_distances(
                X_cuml.array, self.X_, metric=self.metric
            )

        distances = cp.asarray(distances)

        h = self.bandwidth
        if self.kernel in log_probability_kernels_:
            distances = log_probability_kernels_[self.kernel](distances, h)
        else:
            raise ValueError("Unsupported kernel.")

        log_probabilities = cp.zeros(distances.shape[0], dtype=distances.dtype)
        if self.sample_weight_ is not None:
            distances += cp.log(self.sample_weight_)

        logsumexp_kernel[(log_probabilities.size,),(256,)](
            distances, log_probabilities
        )
        # Note that sklearns user guide is wrong
        # It says the (unnormalised) probability output for
        #  the kernel density is sum(K(x,h)).
        # In fact what they implement is (1/n)*sum(K(x,h))
        # Here we divide by n in normal probability space
        # Which becomes -log(n) in log probability space
        sum_weights = (
            cp.sum(self.sample_weight_)
            if self.sample_weight_ is not None
            else distances.shape[1]
        )
        log_probabilities -= np.log(sum_weights)

        # norm
        if len(X_cuml.array.shape) == 1:
            # if X is one dimensional, we have 1 feature
            dimension = 1
        else:
            dimension = X_cuml.array.shape[1]
        log_probabilities = norm_log_probabilities(
            log_probabilities, self.kernel, h, dimension
        )

        return log_probabilities

    def score(self, X, y=None):
        """Compute the total log-likelihood under the model.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        y : None
            Ignored.

        Returns
        -------

        logprob : float
            Total log-likelihood of the data in X. This is normalized to be a
            probability density, so the value will be low for high-dimensional
            data.
        """
        return cp.sum(self.score_samples(X))

    def sample(self, n_samples=1, random_state=None):
        """
        Generate random samples from the model.
        Currently, this is implemented only for gaussian and tophat kernels,
        and the Euclidean metric.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        random_state : int, cupy RandomState instance or None, default=None

        Returns
        -------
        X : cupy array of shape (n_samples, n_features)
            List of samples.
        """
        if not hasattr(self, "X_"):
            raise NotFittedError()

        supported_kernels = ["gaussian", "tophat"]
        if self.kernel not in supported_kernels or self.metric != "euclidean":
            raise NotImplementedError(
                "Only {} kernels, and the euclidean"
                " metric are supported.".format(supported_kernels)
            )

        if isinstance(random_state, cp.random.RandomState):
            rng = random_state
        else:
            rng = cp.random.RandomState(random_state)

        u = rng.uniform(0, 1, size=n_samples)
        if self.sample_weight_ is None:
            i = (u * self.X_.shape[0]).astype(np.int64)
        else:
            cumsum_weight = cp.cumsum(self.sample_weight_)
            sum_weight = cumsum_weight[-1]
            i = cp.searchsorted(cumsum_weight, u * sum_weight)
        if self.kernel == "gaussian":
            return cp.atleast_2d(rng.normal(self.X_[i], self.bandwidth))

        elif self.kernel == "tophat":
            # we first draw points from a d-dimensional normal distribution,
            # then use an incomplete gamma function to map them to a uniform
            # d-dimensional tophat distribution.
            dim = self.X_.shape[1]
            X = rng.normal(size=(n_samples, dim))
            s_sq = cp.einsum("ij,ij->i", X, X)

            correction = (
                gammainc(0.5 * dim, 0.5 * s_sq) ** (1.0 / dim)
                * self.bandwidth
                / cp.sqrt(s_sq)
            )
            return self.X_[i] + X * correction[:, np.newaxis]
