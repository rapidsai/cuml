#
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import math

import cupy as cp
import numpy as np
from cupyx.scipy.special import gammainc

from cuml.common.exceptions import NotFittedError
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.input_utils import input_to_cuml_array, input_to_cupy_array
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.outputs import reflect, run_in_internal_context
from cuml.internals.utils import check_random_seed
from cuml.metrics import pairwise_distances
from cuml.metrics.pairwise_distances import (
    PAIRWISE_DISTANCE_METRICS as SUPPORTED_METRICS,
)

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
def epanechnikov_log_kernel(x, h):
    # don't call log(0) otherwise we get NaNs
    z = cp.maximum(1.0 - (x * x) / (h * h), 1e-30)
    y = (x < h) * cp.log(z)
    y += (x >= h) * np.finfo(y.dtype).min
    return y


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


def norm_factor(kernel, h, d):
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

    return factor + d * np.log(h)


# Implements a faster (but simpler) version of `cupyx.scipy.special.logsumexp`
logsumexp = cp.ReductionKernel(
    "T d",
    "T out",
    "exp(d)",
    "a + b",
    "out = log(a)",
    "0",
    "logsumexp",
)


class KernelDensity(Base, InteropMixin):
    """
    Kernel Density Estimation. Computes a non-parametric density estimate
    from a finite data sample, smoothing the estimate according to a
    bandwidth parameter.

    Parameters
    ----------
    bandwidth : float or {"scott", "silverman"}, default=1.0
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
    handle : cuml.Handle or None, default=None

        .. deprecated:: 26.02
            The `handle` argument was deprecated in 26.02 and will be removed
            in 26.04. There's no need to pass in a handle, cuml now manages
            this resource automatically.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    bandwidth_ : float
        Value of the bandwidth used, either given directly via ``bandwidth`` or
        estimated with ``bandwidth="scott"`` or ``bandwidth="silverman"``.

    Examples
    --------
    >>> from cuml.neighbors import KernelDensity
    >>> import cupy as cp
    >>> rng = cp.random.RandomState(42)
    >>> X = rng.random_sample((100, 3))
    >>> kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    >>> log_density = kde.score_samples(X[:3])
    """

    _cpu_class_path = "sklearn.neighbors.KernelDensity"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "bandwidth",
            "kernel",
            "metric",
            "metric_params",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.metric not in SUPPORTED_METRICS:
            raise UnsupportedOnGPU(
                f"`metric={model.metric!r}` is not supported"
            )
        return {
            "bandwidth": model.bandwidth,
            "kernel": model.kernel,
            "metric": model.metric,
            "metric_params": model.metric_params,
        }

    def _params_to_cpu(self):
        return {
            "bandwidth": self.bandwidth,
            "kernel": self.kernel,
            "metric": self.metric,
            "metric_params": self.metric_params,
        }

    def _attrs_from_cpu(self, model):
        X = cp.asarray(model.tree_.data, dtype=cp.float64)
        sample_weight = (
            None
            if model.tree_.sample_weight is None
            else cp.asarray(model.tree_.sample_weight, dtype=cp.float32)
        )
        return {
            "bandwidth_": model.bandwidth_,
            "_X": X,
            "_sample_weight": sample_weight,
            **super()._attrs_from_cpu(model),
        }

    def _sync_attrs_to_cpu(self, model):
        # There's no way to create a `sklearn.neighbors.KernelDensity` instance
        # from our stored state without fitting a new KernelDensity (since
        # sklearn always uses a pre-fit tree rather than a brute-force method).
        # As such, syncing to CPU is effectively just a refit. Fitting isn't
        # that expensive here, and unlike other estimators it's not clear that
        # a fit-on-gpu, predict-on-cpu model is useful given typical
        # KernelDensity use cases. What we have here should be fine for now.
        X = cp.asnumpy(self._X)
        sample_weight = (
            None
            if self._sample_weight is None
            else cp.asnumpy(self._sample_weight)
        )
        model.fit(X, sample_weight=sample_weight)

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
        super().__init__(
            verbose=verbose, handle=handle, output_type=output_type
        )
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.metric = metric
        self.metric_params = metric_params

    @reflect(reset=True)
    def fit(
        self, X, y=None, sample_weight=None, *, convert_dtype=True
    ) -> "KernelDensity":
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
        self
            Returns the instance itself.
        """
        if isinstance(self.bandwidth, str):
            if self.bandwidth == "scott":
                self.bandwidth_ = X.shape[0] ** (-1 / (X.shape[1] + 4))
            elif self.bandwidth == "silverman":
                self.bandwidth_ = (X.shape[0] * (X.shape[1] + 2) / 4) ** (
                    -1 / (X.shape[1] + 4)
                )
            else:
                raise ValueError(
                    f"Expected bandwidth in ['scott', 'silverman'], got {self.bandwidth!r}"
                )
        elif self.bandwidth <= 0:
            raise ValueError(f"Expected bandwidth > 0, got {self.bandwidth}")
        else:
            self.bandwidth_ = self.bandwidth

        if self.kernel not in VALID_KERNELS:
            raise ValueError(f"kernel={self.kernel!r} is not supported")

        self._X, n_rows, n_cols, _ = input_to_cupy_array(
            X,
            order="C",
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[cp.float32, cp.float64],
        )

        if n_rows < 1:
            raise ValueError(
                f"Found array with 0 sample(s) (shape={self._X.shape}) while "
                f"a minimum of 1 is required by KernelDensity"
            )
        if n_cols < 1:
            raise ValueError(
                f"Found array with 0 feature(s) (shape={self._X.shape}) while "
                f"a minimum of 1 is required by KernelDensity"
            )

        if sample_weight is not None:
            self._sample_weight = input_to_cupy_array(
                sample_weight,
                convert_to_dtype=(np.float32 if convert_dtype else None),
                check_dtype=[cp.float32, cp.float64],
                check_cols=1,
                check_rows=self._X.shape[0],
            ).array
            if self._sample_weight.min() < 0:
                raise ValueError("sample_weight must have positive values")
        else:
            self._sample_weight = None

        return self

    @reflect
    def score_samples(self, X, *, convert_dtype=True) -> CumlArray:
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
        if not hasattr(self, "_X"):
            raise NotFittedError()

        X = input_to_cuml_array(
            X,
            convert_to_dtype=(self._X.dtype if convert_dtype else None),
            check_dtype=[self._X.dtype],
            check_cols=self.n_features_in_,
        ).array
        if self.metric_params:
            if len(self.metric_params) != 1:
                raise ValueError(
                    "Cuml only supports metrics with a single arg."
                )
            metric_arg = list(self.metric_params.values())[0]
            distances = pairwise_distances(
                X,
                self._X,
                metric=self.metric,
                metric_arg=metric_arg,
            )
        else:
            distances = pairwise_distances(X, self._X, metric=self.metric)

        distances = cp.asarray(distances)

        h = distances.dtype.type(self.bandwidth_)
        if self.kernel in log_probability_kernels_:
            # XXX: passing `h` as a 0-dim array works around dtype inference
            # issues in cupy.fuse. See https://github.com/cupy/cupy/issues/9400
            distances = log_probability_kernels_[self.kernel](
                distances, cp.array(h, dtype=distances.dtype)
            )
        else:
            raise ValueError("Unsupported kernel.")

        if self._sample_weight is not None:
            distances += cp.log(self._sample_weight)

        # To avoid overflow, we apply
        # log(exp(x).sum()) -> log(exp(x - x.max())) + x.max()
        # We subtract the max inplace to avoid an extra allocation,
        # since `distances` is no longer needed after this point.
        max_distances = distances.max(axis=1)
        distances -= max_distances[:, None]
        log_probabilities = logsumexp(distances, axis=1)
        log_probabilities += max_distances

        # Note that sklearns user guide is wrong
        # It says the (unnormalised) probability output for
        #  the kernel density is sum(K(x,h)).
        # In fact what they implement is (1/n)*sum(K(x,h))
        # Here we divide by n in normal probability space
        # Which becomes -log(n) in log probability space
        sum_weights = (
            cp.sum(self._sample_weight)
            if self._sample_weight is not None
            else distances.shape[1]
        )
        log_probabilities -= np.log(sum_weights)

        # norm
        if len(X.shape) == 1:
            # if X is one dimensional, we have 1 feature
            dimension = 1
        else:
            dimension = X.shape[1]
        log_probabilities -= norm_factor(self.kernel, h, dimension)

        return log_probabilities

    @run_in_internal_context
    def score(self, X, y=None) -> float:
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
        return float(cp.sum(self.score_samples(X).to_output("cupy")))

    @reflect
    def sample(self, n_samples=1, random_state=None) -> CumlArray:
        """Generate random samples from the model.

        Currently, this is implemented only for gaussian and tophat kernels.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.
        random_state : int, RandomState instance or None, default=None
            Determines random number generation used to generate
            random samples.

        Returns
        -------
        X : cupy array of shape (n_samples, n_features)
            List of samples.
        """
        if not hasattr(self, "_X"):
            raise NotFittedError()

        supported_kernels = ["gaussian", "tophat"]
        if self.kernel not in supported_kernels:
            raise NotImplementedError(
                f"Only {supported_kernels} kernels are supported."
            )

        rng = cp.random.RandomState(check_random_seed(random_state))

        u = rng.uniform(0, 1, size=n_samples)
        if self._sample_weight is None:
            i = (u * self._X.shape[0]).astype(np.int64)
        else:
            cumsum_weight = cp.cumsum(self._sample_weight)
            sum_weight = cumsum_weight[-1]
            i = cp.searchsorted(cumsum_weight, u * sum_weight)
        if self.kernel == "gaussian":
            return cp.atleast_2d(rng.normal(self._X[i], self.bandwidth_))

        elif self.kernel == "tophat":
            # we first draw points from a d-dimensional normal distribution,
            # then use an incomplete gamma function to map them to a uniform
            # d-dimensional tophat distribution.
            dim = self._X.shape[1]
            X = rng.normal(size=(n_samples, dim))
            s_sq = cp.einsum("ij,ij->i", X, X)

            correction = (
                gammainc(0.5 * dim, 0.5 * s_sq) ** (1.0 / dim)
                * self.bandwidth_
                / cp.sqrt(s_sq)
            )
            return self._X[i] + X * correction[:, np.newaxis]
