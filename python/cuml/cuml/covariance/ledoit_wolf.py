#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np

from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.interop import InteropMixin, to_cpu, to_gpu


def _ledoit_wolf_shrinkage(X, assume_centered=False, block_size=1000):
    """Estimate the shrunk Ledoit-Wolf covariance matrix.

    Parameters
    ----------
    X : cupy.ndarray of shape (n_samples, n_features)
        Data from which to compute the covariance estimate.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split
        for memory efficiency.

    Returns
    -------
    shrinkage : float
        The optimal shrinkage coefficient.
    emp_cov : cupy.ndarray of shape (n_features, n_features)
        The empirical covariance matrix.
    mu : float
        The trace of the empirical covariance divided by n_features.
    """
    n_samples, n_features = X.shape

    # Handle single feature case
    if n_features == 1:
        if assume_centered:
            emp_cov = cp.dot(X.T, X) / n_samples
        else:
            emp_cov = cp.cov(X.T, ddof=0).reshape(1, 1)
        return 0.0, emp_cov, float(emp_cov[0, 0])

    # Center data if needed
    if not assume_centered:
        X = X - cp.mean(X, axis=0, keepdims=True)

    # Empirical covariance (biased estimator, ddof=0)
    emp_cov = cp.dot(X.T, X) / n_samples

    # Compute mu = trace(emp_cov) / n_features
    # Using X^2 for numerical stability (matches sklearn)
    X2 = X**2
    emp_cov_trace = cp.sum(X2, axis=0) / n_samples
    mu = float(cp.sum(emp_cov_trace) / n_features)

    # Compute beta_ and delta_ using blocked computation for memory efficiency
    # This follows the Ledoit-Wolf formula as implemented in sklearn
    beta_ = 0.0
    delta_ = 0.0

    for i in range(0, n_features, block_size):
        i_end = min(i + block_size, n_features)
        for j in range(0, n_features, block_size):
            j_end = min(j + block_size, n_features)

            # beta_ accumulates sum of X2.T @ X2
            beta_ += float(cp.sum(cp.dot(X2[:, i:i_end].T, X2[:, j:j_end])))

            # delta_ accumulates sum of squared elements of X.T @ X
            delta_ += float(
                cp.sum(cp.dot(X[:, i:i_end].T, X[:, j:j_end]) ** 2)
            )

    delta_ /= n_samples**2

    # Final beta and delta computation (Ledoit-Wolf formula)
    beta = (1.0 / (n_features * n_samples)) * (beta_ / n_samples - delta_)
    delta = (
        delta_ - 2.0 * mu * float(cp.sum(emp_cov_trace)) + n_features * mu**2
    )
    delta /= n_features

    # Shrinkage is min(beta, delta) / delta, with edge case handling
    beta = min(beta, delta)
    if beta == 0:
        shrinkage = 0.0
    else:
        shrinkage = beta / delta

    return shrinkage, emp_cov, mu


class LedoitWolf(Base, InteropMixin):
    """LedoitWolf Estimator for covariance matrix estimation.

    Computes the Ledoit-Wolf shrinkage estimator for the covariance matrix.
    This estimator regularizes the empirical covariance by shrinking it
    towards a scaled identity matrix, with the shrinkage coefficient
    determined by the Ledoit-Wolf formula.

    The regularized covariance is:
    ``(1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)``

    where ``mu = trace(cov) / n_features`` and ``shrinkage`` is computed
    to minimize the Mean Squared Error between the regularized estimate
    and the true covariance.

    Parameters
    ----------
    store_precision : bool, default=True
        Specifies if the estimated precision matrix is stored.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero. If False (default), data will be centered before computation.
    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split
        during its Ledoit-Wolf estimation. This is purely a memory
        optimization and does not affect results.
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
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
            'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.
    location_ : ndarray of shape (n_features,)
        Estimated location, i.e., the estimated mean.
    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix. Only stored if ``store_precision``
        is True.
    shrinkage_ : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.covariance import LedoitWolf
    >>> rng = cp.random.RandomState(42)
    >>> X = rng.randn(100, 5)
    >>> lw = LedoitWolf().fit(X)
    >>> lw.covariance_.shape
    (5, 5)
    >>> lw.shrinkage_  # doctest: +SKIP
    0.123...

    See Also
    --------
    sklearn.covariance.LedoitWolf : The scikit-learn CPU implementation.

    References
    ----------
    O. Ledoit and M. Wolf, "A Well-Conditioned Estimator for
    Large-Dimensional Covariance Matrices", Journal of Multivariate
    Analysis, Volume 88, Issue 2, February 2004, pages 365-411.
    """

    covariance_ = CumlArrayDescriptor()
    location_ = CumlArrayDescriptor()
    precision_ = CumlArrayDescriptor()

    _cpu_class_path = "sklearn.covariance.LedoitWolf"

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "store_precision",
            "assume_centered",
            "block_size",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        return {
            "store_precision": model.store_precision,
            "assume_centered": model.assume_centered,
            "block_size": model.block_size,
        }

    def _params_to_cpu(self):
        return {
            "store_precision": self.store_precision,
            "assume_centered": self.assume_centered,
            "block_size": self.block_size,
        }

    def _attrs_from_cpu(self, model):
        attrs = {
            "covariance_": to_gpu(model.covariance_),
            "location_": to_gpu(model.location_),
            "shrinkage_": model.shrinkage_,
            **super()._attrs_from_cpu(model),
        }
        if self.store_precision and hasattr(model, "precision_"):
            attrs["precision_"] = to_gpu(model.precision_)
        return attrs

    def _attrs_to_cpu(self, model):
        attrs = {
            "covariance_": to_cpu(self.covariance_),
            "location_": to_cpu(self.location_),
            "shrinkage_": self.shrinkage_,
            **super()._attrs_to_cpu(model),
        }
        if self.store_precision and self.precision_ is not None:
            attrs["precision_"] = to_cpu(self.precision_)
        return attrs

    def __init__(
        self,
        *,
        store_precision=True,
        assume_centered=False,
        block_size=1000,
        handle=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            handle=handle,
            verbose=verbose,
            output_type=output_type,
        )
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.block_size = block_size
        self.shrinkage_ = None

    @generate_docstring()
    def fit(self, X, y=None) -> "LedoitWolf":
        """Fit the Ledoit-Wolf shrunk covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : LedoitWolf
            Returns the instance itself.
        """
        X_m, n_samples, n_features, dtype = input_to_cuml_array(
            X,
            check_dtype=[np.float32, np.float64],
            order="C",
        )

        self._set_n_features_in(X_m)
        self._set_output_type(X)

        X_arr = cp.asarray(X_m)

        # Compute location (mean)
        if self.assume_centered:
            location = cp.zeros(n_features, dtype=dtype)
        else:
            location = cp.mean(X_arr, axis=0)

        # Compute shrinkage and empirical covariance
        shrinkage, emp_cov, mu = _ledoit_wolf_shrinkage(
            X_arr,
            assume_centered=self.assume_centered,
            block_size=self.block_size,
        )

        # Compute shrunk covariance:
        # (1 - shrinkage) * emp_cov + shrinkage * mu * I
        shrunk_cov = (1.0 - shrinkage) * emp_cov
        shrunk_cov.flat[:: n_features + 1] += shrinkage * mu

        self.shrinkage_ = shrinkage
        self.location_ = CumlArray(data=location)
        self.covariance_ = CumlArray(data=shrunk_cov)

        # Compute precision matrix if requested
        if self.store_precision:
            self.precision_ = CumlArray(data=cp.linalg.pinv(shrunk_cov))
        else:
            self.precision_ = None

        return self

    def get_precision(self):
        """Getter for the precision matrix.

        Returns
        -------
        precision_ : ndarray of shape (n_features, n_features)
            The precision matrix associated to the current covariance object.
        """
        if self.store_precision:
            return self.precision_
        else:
            covariance = cp.asarray(self.covariance_)
            precision = cp.linalg.pinv(covariance)
            return CumlArray(data=precision).to_output(
                self._get_output_type(self.covariance_)
            )

    def score(self, X_test, y=None):
        """Compute the log-likelihood of X_test under the estimated model.

        The log-likelihood is computed using the Gaussian model.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data of which we compute the likelihood.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        log_likelihood : float
            Log-likelihood of the data under the fitted Gaussian model.
        """
        X_m = input_to_cuml_array(
            X_test,
            check_dtype=[np.float32, np.float64],
            check_cols=self.n_features_in_,
            order="C",
        ).array

        X_arr = cp.asarray(X_m)
        n_samples, n_features = X_arr.shape

        # Get precision matrix
        precision = cp.asarray(self.get_precision())
        location = cp.asarray(self.location_)

        # Center the data
        X_centered = X_arr - location

        # Compute log-likelihood
        # log_likelihood = -0.5 * (n_features * log(2*pi)
        #                         - log|precision|
        #                         + (X - mu)^T @ precision @ (X - mu))
        log_det_precision = cp.linalg.slogdet(precision)[1]

        # Compute sum of Mahalanobis distances
        # (X - mu) @ precision @ (X - mu)^T for each sample
        mahal = cp.sum(cp.dot(X_centered, precision) * X_centered, axis=1)

        log_likelihood = -0.5 * (
            n_features * np.log(2 * np.pi) - log_det_precision + cp.mean(mahal)
        )

        return float(log_likelihood)

    def error_norm(
        self, comp_cov, norm="frobenius", scaling=True, squared=True
    ):
        """Compute the Mean Squared Error between two covariance estimators.

        Parameters
        ----------
        comp_cov : array-like of shape (n_features, n_features)
            The covariance to compare with.
        norm : {"frobenius", "spectral"}, default="frobenius"
            The type of norm used to compute the error.
        scaling : bool, default=True
            If True, the squared error is scaled by n_features.
        squared : bool, default=True
            If True, return squared error. If False, return error.

        Returns
        -------
        error : float
            The Mean Squared Error (in the sense of the Frobenius norm)
            between `self` and `comp_cov`.
        """
        comp_cov_m = input_to_cuml_array(
            comp_cov,
            check_dtype=[np.float32, np.float64],
            order="C",
        ).array

        comp_cov_arr = cp.asarray(comp_cov_m)
        self_cov = cp.asarray(self.covariance_)

        diff = self_cov - comp_cov_arr

        if norm == "frobenius":
            error = cp.sum(diff**2)
        elif norm == "spectral":
            error = cp.linalg.norm(diff, ord=2) ** 2
        else:
            raise ValueError(
                f"Invalid norm '{norm}'. Must be 'frobenius' or 'spectral'."
            )

        if scaling:
            n_features = self_cov.shape[0]
            error = error / n_features

        if not squared:
            error = cp.sqrt(error)

        return float(error)

    def mahalanobis(self, X):
        """Compute the squared Mahalanobis distances of given observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The observations, the Mahalanobis distances of which we compute.

        Returns
        -------
        mahalanobis_distances : ndarray of shape (n_samples,)
            Squared Mahalanobis distances of the observations.
        """
        X_m = input_to_cuml_array(
            X,
            check_dtype=[np.float32, np.float64],
            check_cols=self.n_features_in_,
            order="C",
        ).array

        X_arr = cp.asarray(X_m)
        precision = cp.asarray(self.get_precision())
        location = cp.asarray(self.location_)

        # Center the data
        X_centered = X_arr - location

        # Compute Mahalanobis distance: (X - mu) @ precision @ (X - mu)^T
        mahal = cp.sum(cp.dot(X_centered, precision) * X_centered, axis=1)

        return CumlArray(data=mahal).to_output(self._get_output_type(X))
