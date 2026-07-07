#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import cupy as cp
import numpy as np

from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals import reflect, run_in_internal_context
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.interop import InteropMixin, to_cpu, to_gpu
from cuml.internals.validation import check_inputs, check_is_fitted


def _empirical_covariance(X, assume_centered=False):
    """Compute the maximum likelihood covariance estimator."""
    if assume_centered:
        return cp.dot(X.T, X) / X.shape[0]

    location = cp.mean(X, axis=0, keepdims=True)
    X_centered = X - location
    return cp.dot(X_centered.T, X_centered) / X.shape[0]


def _log_likelihood(emp_cov, precision):
    """Compute the sample mean log-likelihood under a covariance model."""
    sign, log_det_precision = cp.linalg.slogdet(precision)
    if float(sign) <= 0:
        log_det_precision = -cp.inf

    n_features = precision.shape[0]
    log_likelihood = -cp.sum(emp_cov * precision) + log_det_precision
    log_likelihood -= n_features * np.log(2 * np.pi)
    log_likelihood /= 2.0
    return float(log_likelihood)


class EmpiricalCovariance(InteropMixin, Base):
    """Maximum likelihood covariance estimator.

    Parameters
    ----------
    store_precision : bool, default=True
        Specifies if the estimated precision matrix is stored.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero. If False (default), data will be centered before computation.
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
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.covariance import EmpiricalCovariance
    >>> rng = cp.random.RandomState(42)
    >>> X = rng.randn(100, 5)
    >>> cov = EmpiricalCovariance().fit(X)
    >>> cov.covariance_.shape
    (5, 5)

    See Also
    --------
    sklearn.covariance.EmpiricalCovariance
        The scikit-learn CPU implementation.
    """

    covariance_ = CumlArrayDescriptor()
    location_ = CumlArrayDescriptor()
    precision_ = CumlArrayDescriptor()

    _cpu_class_path = "sklearn.covariance.EmpiricalCovariance"

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "store_precision",
            "assume_centered",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        return {
            "store_precision": model.store_precision,
            "assume_centered": model.assume_centered,
        }

    def _params_to_cpu(self):
        return {
            "store_precision": self.store_precision,
            "assume_centered": self.assume_centered,
        }

    def _attrs_from_cpu(self, model):
        return {
            "covariance_": to_gpu(model.covariance_),
            "location_": to_gpu(model.location_),
            "precision_": to_gpu(model.precision_)
            if self.store_precision
            else None,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "covariance_": to_cpu(self.covariance_),
            "location_": to_cpu(self.location_),
            "precision_": to_cpu(self.precision_)
            if self.store_precision
            else None,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        store_precision=True,
        assume_centered=False,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.store_precision = store_precision
        self.assume_centered = assume_centered

    @reflect(reset=True)
    def fit(
        self, X, y=None, *, convert_dtype="deprecated"
    ) -> "EmpiricalCovariance":
        """Fit the maximum likelihood covariance estimator to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Not used, present for API consistency.
        convert_dtype : bool, default="deprecated"
            .. deprecated:: 26.08
                `convert_dtype` was deprecated in version 26.08 and will be
                removed in version 26.10. cuML only copies input arrays when
                necessary (e.g. to unify dtypes), there is no reason to provide
                this keyword going forward.

        Returns
        -------
        self : EmpiricalCovariance
            Returns the instance itself.
        """
        X = check_inputs(
            self,
            X,
            dtype=("float32", "float64"),
            convert_dtype=convert_dtype,
            reset=True,
        )
        if X.shape[0] == 1:
            warnings.warn(
                "Only one sample available. "
                "You may want to reshape your data array"
            )

        if self.assume_centered:
            location = cp.zeros(X.shape[1], dtype=X.dtype)
        else:
            location = cp.mean(X, axis=0)

        covariance = _empirical_covariance(
            X, assume_centered=self.assume_centered
        )

        self.location_ = CumlArray(data=location)
        self.covariance_ = CumlArray(data=covariance)

        if self.store_precision:
            self.precision_ = CumlArray(data=cp.linalg.pinv(covariance))
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
        check_is_fitted(self)

        if self.store_precision:
            return self.precision_

        covariance = self.covariance_.to_output("cupy")
        precision = cp.linalg.pinv(covariance)
        return precision

    @run_in_internal_context
    def score(self, X_test, y=None) -> float:
        """Compute the log-likelihood of X_test under the estimated model.

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
        check_is_fitted(self)
        X_test = check_inputs(
            self,
            X_test,
            dtype=("float32", "float64"),
        )
        precision = self.get_precision().to_output("cupy")
        location = self.location_.to_output("cupy")

        test_cov = _empirical_covariance(
            X_test - location, assume_centered=True
        )

        return _log_likelihood(test_cov, precision)

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
        check_is_fitted(self)

        comp_cov = check_inputs(self, comp_cov, dtype=("float32", "float64"))
        self_cov = self.covariance_.to_output("cupy")
        diff = comp_cov - self_cov

        if norm == "frobenius":
            error = cp.sum(diff**2)
        elif norm == "spectral":
            error = cp.linalg.norm(diff, ord=2) ** 2
        else:
            raise NotImplementedError(
                "Only spectral and frobenius norms are implemented"
            )

        if scaling:
            error = error / self_cov.shape[0]

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
        check_is_fitted(self)
        X = check_inputs(self, X, dtype=("float32", "float64"))
        precision = self.get_precision().to_output("cupy")
        location = self.location_.to_output("cupy")

        X_centered = X - location
        mahal = cp.sum(cp.dot(X_centered, precision) * X_centered, axis=1)

        return mahal
