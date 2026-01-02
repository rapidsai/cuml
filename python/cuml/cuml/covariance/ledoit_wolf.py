#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np

from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals import reflect, run_in_internal_context
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.input_utils import input_to_cupy_array
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

    if not assume_centered:
        X = X - cp.mean(X, axis=0, keepdims=True)

    emp_cov = cp.dot(X.T, X) / n_samples

    X2 = X**2
    emp_cov_trace = cp.sum(X2, axis=0) / n_samples
    mu = float(cp.sum(emp_cov_trace) / n_features)

    beta_ = 0.0
    delta_ = 0.0

    for i in range(0, n_features, block_size):
        i_end = min(i + block_size, n_features)
        for j in range(0, n_features, block_size):
            j_end = min(j + block_size, n_features)

            beta_ += float(cp.sum(cp.dot(X2[:, i:i_end].T, X2[:, j:j_end])))
            delta_ += float(
                cp.sum(cp.dot(X[:, i:i_end].T, X[:, j:j_end]) ** 2)
            )

    delta_ /= n_samples**2

    beta = (1.0 / (n_features * n_samples)) * (beta_ / n_samples - delta_)
    delta = (
        delta_ - 2.0 * mu * float(cp.sum(emp_cov_trace)) + n_features * mu**2
    )
    delta /= n_features

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
    handle : cuml.Handle or None, default=None

        .. deprecated:: 26.02
            The `handle` argument was deprecated in 26.02 and will be removed
            in 26.04. There's no need to pass in a handle, cuml now manages
            this resource automatically.
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
        return {
            "covariance_": to_gpu(model.covariance_),
            "location_": to_gpu(model.location_),
            "precision_": to_gpu(model.precision_)
            if self.store_precision
            else None,
            "shrinkage_": model.shrinkage_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "covariance_": to_cpu(self.covariance_),
            "location_": to_cpu(self.location_),
            "precision_": to_cpu(self.precision_)
            if self.store_precision
            else None,
            "shrinkage_": self.shrinkage_,
            **super()._attrs_to_cpu(model),
        }

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

    @reflect(reset=True)
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
        X_arr, _, n_features, dtype = input_to_cupy_array(
            X,
            check_dtype=[np.float32, np.float64],
            order="C",
        )

        if self.assume_centered:
            location = cp.zeros(n_features, dtype=dtype)
        else:
            location = cp.mean(X_arr, axis=0)

        shrinkage, emp_cov, mu = _ledoit_wolf_shrinkage(
            X_arr,
            assume_centered=self.assume_centered,
            block_size=self.block_size,
        )

        shrunk_cov = (1.0 - shrinkage) * emp_cov
        shrunk_cov.flat[:: n_features + 1] += shrinkage * mu

        self.shrinkage_ = shrinkage
        self.location_ = CumlArray(data=location)
        self.covariance_ = CumlArray(data=shrunk_cov)

        if self.store_precision:
            self.precision_ = CumlArray(data=cp.linalg.pinv(shrunk_cov))
        else:
            self.precision_ = None

        return self

    @reflect
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
            return precision

    @run_in_internal_context
    def score(self, X_test, y=None) -> float:
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
        X_arr, _, n_features, _ = input_to_cupy_array(
            X_test,
            check_dtype=[np.float32, np.float64],
            check_cols=self.n_features_in_,
            order="C",
        )

        precision = cp.asarray(self.get_precision())
        location = cp.asarray(self.location_)

        X_centered = X_arr - location
        log_det_precision = cp.linalg.slogdet(precision)[1]
        mahal = cp.sum(cp.dot(X_centered, precision) * X_centered, axis=1)

        log_likelihood = -0.5 * (
            n_features * np.log(2 * np.pi) - log_det_precision + cp.mean(mahal)
        )

        return float(log_likelihood)

    @run_in_internal_context
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
        comp_cov_arr, _, _, _ = input_to_cupy_array(
            comp_cov,
            check_dtype=[np.float32, np.float64],
            order="C",
        )
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

    @reflect
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
        X_arr, _, _, _ = input_to_cupy_array(
            X,
            check_dtype=[np.float32, np.float64],
            check_cols=self.n_features_in_,
            order="C",
        )
        precision = cp.asarray(self.get_precision())
        location = cp.asarray(self.location_)

        X_centered = X_arr - location
        mahal = cp.sum(cp.dot(X_centered, precision) * X_centered, axis=1)

        return mahal
