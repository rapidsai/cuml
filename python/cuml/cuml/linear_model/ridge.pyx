#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray, cuda_ptr
from cuml.internals.base import Base
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import FMajorInputTagMixin, RegressorMixin
from cuml.internals.outputs import reflect
from cuml.linear_model.base import (
    LinearPredictMixin,
    check_deprecated_normalize,
)

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM" nogil:

    cdef void ridgeFit(handle_t& handle,
                       float *input,
                       size_t n_rows,
                       size_t n_cols,
                       float *labels,
                       float *alpha,
                       int n_alpha,
                       float *coef,
                       float *intercept,
                       bool fit_intercept,
                       bool normalize,
                       int algo,
                       float *sample_weight) except +

    cdef void ridgeFit(handle_t& handle,
                       double *input,
                       size_t n_rows,
                       size_t n_cols,
                       double *labels,
                       double *alpha,
                       int n_alpha,
                       double *coef,
                       double *intercept,
                       bool fit_intercept,
                       bool normalize,
                       int algo,
                       double *sample_weight) except +


_ridge_transform = cp.ElementwiseKernel(
    "T x, T s, T alpha",
    "T out",
    "out = s < 1e-10 ? 0 : x * s / (s * s + alpha)",
    "_ridge_transform"
)


_SOLVER_CUML_TO_SKLEARN = {
    "auto": "auto",
    "svd": "svd",
    "eig": "cholesky",
}


class Ridge(Base,
            InteropMixin,
            RegressorMixin,
            LinearPredictMixin,
            FMajorInputTagMixin):
    """Linear least squares with L2 regularization.

    Ridge extends LinearRegression by providing L2 regularization on the
    coefficients when predicting response y with a linear combination of the
    predictors in X. It can reduce the variance of the predictors, and improves
    the conditioning of the problem.

    Parameters
    ----------
    alpha : float or array of shape (n_targets,), default=1.0
        Regularization strength - must be a positive float. Larger values
        specify stronger regularization.
    solver : {'auto', 'eig', 'svd'}, default='auto'
        The solver to use when fitting:

        - 'auto': will select 'eig' if supported, and 'svd' otherwise.

        - 'eig': uses an eigendecomposition of the covariance matrix. It is
          fast but potentially unstable. It also doesn't support multi-target
          ``y`` or array-like ``alpha``.

        - 'svd': uses an SVD decomposition. It's slower, but stable and
          supports all options.
    fit_intercept : bool, default=True
        If True, Ridge tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    copy_X: bool, default=True
        If True, X will never be mutated. Setting to False may reduce memory
        usage, at the cost of potentially mutating X.
    normalize : boolean, default=False

        .. deprecated:: 25.12
            ``normalize`` is deprecated and will be removed in 26.02. When
            needed, please use a ``StandardScaler`` to normalize your data
            before passing to ``fit``.

    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        The estimated coefficients for the linear regression model.
    intercept_ : float or array, shape (n_targets,)
        The independent term. If `fit_intercept` is False, will be 0. Will be
        an array when fit on multi-target y, otherwise will be a float.
    solver_ : str
        The solver that was used at fit time.

    Notes
    -----
    For additional docs, see `Scikit-learn's Ridge Regression
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_.

    Examples
    --------
    >>> import cupy as cp
    >>> import cudf
    >>> from cuml import Ridge

    >>> X = cudf.DataFrame()
    >>> X['col1'] = cp.array([1,1,2,2], dtype = cp.float32)
    >>> X['col2'] = cp.array([1,2,2,3], dtype = cp.float32)
    >>> y = cudf.Series(cp.array([6.0, 8.0, 9.0, 11.0], dtype=cp.float32))

    >>> ridge = Ridge(alpha=1e-5).fit(X, y)
    >>> print(ridge.coef_) # doctest: +SKIP
    0 1.000...
    1 1.999...
    >>> print(ridge.intercept_) #doctest: +SKIP
    3.0...
    >>> X_new = cudf.DataFrame()
    >>> X_new['col1'] = cp.array([3,2], dtype=cp.float32)
    >>> X_new['col2'] = cp.array([5,5], dtype=cp.float32)
    >>> preds = ridge.predict(X_new)
    >>> print(preds) # doctest: +SKIP
    0 15.999...
    1 14.999...
    """
    coef_ = CumlArrayDescriptor()
    intercept_ = CumlArrayDescriptor()

    _cpu_class_path = "sklearn.linear_model.Ridge"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "alpha",
            "fit_intercept",
            "solver",
            "copy_X",
            "normalize",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.positive:
            raise UnsupportedOnGPU("`positive=True` is not supported")

        if model.solver == "svd":
            solver = "svd"
        elif model.solver == "lbfgs":
            # lbfgs only works in sklearn for positive=True, since we don't
            # support that parameter we want to fallback so sklearn
            # can error appropriately
            raise UnsupportedOnGPU(f"`solver={model.solver!r}` is not supported")
        else:
            solver = "auto"

        return {
            "alpha": model.alpha,
            "fit_intercept": model.fit_intercept,
            "solver": solver,
            "copy_X": model.copy_X,
        }

    def _params_to_cpu(self):
        solver = _SOLVER_CUML_TO_SKLEARN[self.solver]
        return {
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "solver": solver,
            "copy_X": self.copy_X,
        }

    def _attrs_from_cpu(self, model):
        return {
            "intercept_": to_gpu(model.intercept_),
            "coef_": to_gpu(model.coef_),
            "solver_": "svd" if model.solver_ == "svd" else "eig",
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        if model.solver == "auto":
            solver = _SOLVER_CUML_TO_SKLEARN[self.solver_]
        else:
            solver = model.solver

        return {
            "intercept_": to_cpu(self.intercept_),
            "coef_": to_cpu(self.coef_),
            "solver_": solver,
            "n_iter_": None,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        solver="auto",
        copy_X=True,
        normalize=False,
        handle=None,
        output_type=None,
        verbose=False,
    ):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.copy_X = copy_X
        self.normalize = normalize

    def _fit_svd(
        self,
        X_m,
        y_m,
        sample_weight_m,
        *,
        alpha,
        X_is_copy,
        y_is_copy,
    ):
        """Fit a Ridge regression using SVD."""
        if self.normalize:
            raise ValueError(
                "The normalize option is not supported for solver='svd'"
            )

        X = X_m.to_output("cupy")
        y = y_m.to_output("cupy")
        sample_weight = (
            None if sample_weight_m is None else sample_weight_m.to_output("cupy")
        )

        # Ensure 2D
        if X.ndim == 1:
            X = X[:, None]
        if (y_1d := y.ndim == 1):
            y = y[:, None]

        # Normalize alpha to a cupy array of shape (n_targets,)
        if cp.isscalar(alpha):
            alpha = cp.full(y.shape[1], alpha, dtype=X.dtype)

        if self.fit_intercept:
            if sample_weight is not None:
                # Offset by weighted mean
                den = sample_weight.sum()
                X_offset = (X * sample_weight[:, None]).sum(axis=0) / den
                y_offset = (y * sample_weight[:, None]).sum(axis=0) / den
            else:
                # Offset by mean
                X_offset = X.mean(axis=0)
                y_offset = y.mean(axis=0)
            # Subtract offset, reusing existing buffers when possible
            X = cp.subtract(
                X,
                X_offset,
                out=X if X_is_copy or not self.copy_X else None,
            )
            y = cp.subtract(y, y_offset, out=y if y_is_copy else None)
            X_is_copy = y_is_copy = True

        if sample_weight is not None:
            # Weights are always copied, can mutate buffer
            sqrt_weight = cp.sqrt(sample_weight, out=sample_weight)
            # Multiply by sqrt(weight), reusing existing buffers when possible
            X = cp.multiply(
                X,
                sqrt_weight[:, None],
                out=X if X_is_copy or not self.copy_X else None,
            )
            y = cp.multiply(y, sqrt_weight[:, None], out=y if y_is_copy else None)

        # Solve using SVD method
        u, s, vh = cp.linalg.svd(X, full_matrices=False)
        temp = _ridge_transform(u.T.dot(y), s[:, None], alpha)
        coef = vh.T.dot(temp).T

        if self.fit_intercept:
            intercept = y_offset - cp.dot(X_offset, coef.T)
            if y_1d:
                intercept = coef.dtype.type(intercept.item())
            else:
                intercept = CumlArray(data=intercept)
        else:
            intercept = 0.0
        coef = CumlArray(data=(coef.ravel() if y.shape[1] == 1 else coef))

        return coef, intercept

    def _fit_eig(
        self,
        X_m,
        y_m,
        sample_weight_m,
        *,
        alpha,
        X_is_copy,
        y_is_copy,
    ):
        """Fit a Ridge regression using the Eig solver."""
        cdef int n_rows = X_m.shape[0]
        cdef int n_cols = X_m.shape[1]

        # The `eig` solver requires X be F-contiguous. Additionally, all inputs
        # are mutated when weighted or `fit_intercept=True`, so we'll copy if
        # required (note that sample_weight is always already copied).
        mutates = self.fit_intercept or sample_weight_m is not None
        X_m = input_to_cuml_array(
            X_m, order="F", deepcopy=(mutates and self.copy_X and not X_is_copy)
        ).array
        if mutates and not y_is_copy:
            y_m = input_to_cuml_array(y_m, deepcopy=True).array

        # Allocate outputs
        coef = CumlArray.zeros(n_cols, dtype=X_m.dtype)

        cdef float intercept_f32
        cdef double intercept_f64
        cdef float alpha_f32 = alpha
        cdef double alpha_f64 = alpha

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t y_ptr = y_m.ptr
        cdef uintptr_t coef_ptr = coef.ptr
        cdef uintptr_t sample_weight_ptr = (
            0 if sample_weight_m is None else sample_weight_m.ptr
        )
        cdef bool normalize = self.normalize
        cdef bool fit_intercept = self.fit_intercept
        cdef bool use_float32 = X_m.dtype == np.float32

        with nogil:
            if use_float32:
                ridgeFit(
                    handle_[0],
                    <float*>X_ptr,
                    n_rows,
                    n_cols,
                    <float*>y_ptr,
                    &alpha_f32,
                    1,
                    <float*>coef_ptr,
                    &intercept_f32,
                    fit_intercept,
                    normalize,
                    1,
                    <float*>sample_weight_ptr,
                )
            else:
                ridgeFit(
                    handle_[0],
                    <double*>X_ptr,
                    n_rows,
                    n_cols,
                    <double*>y_ptr,
                    &alpha_f64,
                    1,
                    <double*>coef_ptr,
                    &intercept_f64,
                    fit_intercept,
                    normalize,
                    1,
                    <double*>sample_weight_ptr,
                )
        self.handle.sync()

        if self.fit_intercept:
            intercept = intercept_f32 if use_float32 else intercept_f64
            if y_m.ndim == 1:
                intercept = coef.dtype.type(intercept)
            else:
                intercept = CumlArray(data=cp.array([intercept], dtype=coef.dtype))
        else:
            intercept = 0.0

        return coef, intercept

    @generate_docstring()
    @reflect(reset=True)
    def fit(self, X, y, sample_weight=None, *, convert_dtype=True) -> "Ridge":
        """
        Fit the model with X and y.
        """
        check_deprecated_normalize(self)

        X_m, n_rows, n_cols, dtype = input_to_cuml_array(
            X,
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[np.float32, np.float64],
            order="K",
        )

        if n_cols < 1:
            raise ValueError(
                f"Found array with {n_cols} feature(s) (shape={X_m.shape}) while "
                f"a minimum of 1 is required."
            )

        if n_rows < 2:
            raise ValueError(
                f"Found array with {n_rows} sample(s) (shape={X_m.shape}) while a "
                f"minimum of 2 is required."
            )

        y_m, _, n_targets, _ = input_to_cuml_array(
            y,
            check_dtype=dtype,
            convert_to_dtype=(dtype if convert_dtype else None),
            check_rows=n_rows,
            order="K",
        )

        if sample_weight is not None:
            sample_weight_m = input_to_cuml_array(
                sample_weight,
                check_dtype=dtype,
                convert_to_dtype=(dtype if convert_dtype else None),
                check_rows=n_rows,
                check_cols=1,
                deepcopy=True,
            ).array
        else:
            sample_weight_m = None

        X_is_copy = cuda_ptr(X) != X_m.ptr
        y_is_copy = cuda_ptr(y) != y_m.ptr

        # Validate alpha
        if cp.isscalar(self.alpha):
            alpha = self.alpha
            if self.alpha < 0.0:
                raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        else:
            alpha = cp.asarray(self.alpha, dtype=dtype).ravel()
            if (alpha < 0).any():
                raise ValueError(f"alpha must be non-negative, got {self.alpha}")
            if alpha.shape[0] == 1:
                alpha = alpha.item()
            elif alpha.shape[0] != n_targets:
                raise ValueError(
                    f"Number of targets and number of penalties do not correspond: "
                    f"{n_targets} != {alpha.shape[0]}"
                )

        # Validate and select solver
        _SUPPORTED_SOLVERS = ["auto", "eig", "svd"]
        if (solver := self.solver) not in _SUPPORTED_SOLVERS:
            raise ValueError(
                f"Expected `solver` to be one of {_SUPPORTED_SOLVERS}, got {solver!r}"
            )

        if solver == "eig":
            if n_cols == 1:
                raise ValueError(
                    "solver='eig' doesn't support X with 1 column, please select "
                    "solver='svd' or solver='auto' instead"
                )
            if n_targets != 1:
                raise ValueError(
                    "solver='eig' doesn't support multi-target y, please select "
                    "solver='svd' or solver='auto' instead"
                )
        elif solver == "auto":
            solver = "svd" if n_cols == 1 or n_targets != 1 else "eig"

        # Perform fit
        solver_func = self._fit_svd if solver == "svd" else self._fit_eig
        coef, intercept = solver_func(
            X_m,
            y_m,
            sample_weight_m,
            alpha=alpha,
            X_is_copy=X_is_copy,
            y_is_copy=y_is_copy,
        )

        self.coef_ = coef
        self.intercept_ = intercept
        self.solver_ = solver

        return self
