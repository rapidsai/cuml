#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
import warnings

import numpy as np

from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU, to_gpu
from cuml.internals.mixins import FMajorInputTagMixin, RegressorMixin
from cuml.linear_model.base import LinearPredictMixin

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


_SOLVER_SKLEARN_TO_CUML = {
    "auto": "auto",
    "svd": "svd",
    "cholesky": "eig",
    "lsqr": "eig",
    "sag": "eig",
    "saga": "eig",
    "sparse_cg": "eig"
}
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

    """
    Ridge extends LinearRegression by providing L2 regularization on the
    coefficients when predicting response y with a linear combination of the
    predictors in X. It can reduce the variance of the predictors, and improves
    the conditioning of the problem.

    cuML's Ridge can take array-like objects, either in host as
    NumPy arrays or in device (as Numba or `__cuda_array_interface__`
    compliant), in addition to cuDF objects. It provides 2
    algorithms: SVD and Eig to fit a linear model. In general SVD uses
    significantly more memory and is slower than Eig. If using CUDA 10.1,
    the memory difference is even bigger than in the other supported CUDA
    versions. However, SVD is more stable than Eig (default).

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> import cudf

        >>> # Both import methods supported
        >>> from cuml import Ridge
        >>> from cuml.linear_model import Ridge

        >>> alpha = 1e-5
        >>> ridge = Ridge(alpha=alpha, fit_intercept=True, normalize=False,
        ...               solver="eig")

        >>> X = cudf.DataFrame()
        >>> X['col1'] = cp.array([1,1,2,2], dtype = cp.float32)
        >>> X['col2'] = cp.array([1,2,2,3], dtype = cp.float32)

        >>> y = cudf.Series(cp.array([6.0, 8.0, 9.0, 11.0], dtype=cp.float32))

        >>> result_ridge = ridge.fit(X, y)
        >>> print(result_ridge.coef_) # doctest: +SKIP
        0 1.000...
        1 1.999...
        >>> print(result_ridge.intercept_)
        3.0...
        >>> X_new = cudf.DataFrame()
        >>> X_new['col1'] = cp.array([3,2], dtype=cp.float32)
        >>> X_new['col2'] = cp.array([5,5], dtype=cp.float32)
        >>> preds = result_ridge.predict(X_new)
        >>> print(preds) # doctest: +SKIP
        0 15.999...
        1 14.999...

    Parameters
    ----------
    alpha : float (default = 1.0)
        Regularization strength - must be a positive float. Larger values
        specify stronger regularization. Array input will be supported later.
    solver : {'auto', 'eig', 'svd'} (default = 'auto')
        Eig uses a eigendecomposition of the covariance matrix, and is much
        faster. SVD is slower, but guaranteed to be stable.
    fit_intercept : boolean (default = True)
        If True, Ridge tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        If True, the predictors in X will be normalized by dividing by the
        column-wise standard deviation.
        If False, no scaling will be done.
        Note: this is in contrast to sklearn's deprecated `normalize` flag,
        which divides by the column-wise L2 norm; but this is the same as if
        using sklearn's StandardScaler.
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
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : float
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    -----
    Ridge provides L2 regularization. This means that the coefficients can
    shrink to become very small, but not zero. This can cause issues of
    interpretability on the coefficients.
    Consider using Lasso, or thresholding small coefficients to zero.

    **Applications of Ridge**

        Ridge Regression is used in the same way as LinearRegression, but does
        not suffer from multicollinearity issues.  Ridge is used in insurance
        premium prediction, stock market analysis and much more.


    For additional docs, see `Scikit-learn's Ridge Regression
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_.
    """

    coef_ = CumlArrayDescriptor(order='F')

    _cpu_class_path = "sklearn.linear_model.Ridge"

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + ['solver', 'fit_intercept', 'normalize', 'alpha']

    @classmethod
    def _params_from_cpu(cls, model):
        if model.positive:
            raise UnsupportedOnGPU("`positive=True` is not supported")

        solver = _SOLVER_SKLEARN_TO_CUML.get(model.solver)
        if solver is None:
            raise UnsupportedOnGPU(f"`solver={model.solver!r}` is not supported")

        return {
            "alpha": model.alpha,
            "fit_intercept": model.fit_intercept,
            "solver": solver,
        }

    def _params_to_cpu(self):
        solver = _SOLVER_CUML_TO_SKLEARN[self.solver]
        return {
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "solver": solver,
        }

    def _attrs_from_cpu(self, model):
        solver = _SOLVER_SKLEARN_TO_CUML.get(model.solver_)
        if solver is None:
            raise UnsupportedOnGPU(f"`solver={model.solver_!r}` is not supported")

        return {
            "intercept_": float(model.intercept_),
            "coef_": to_gpu(model.coef_, order="F"),
            "solver_": solver,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        if model.solver == "auto":
            solver = _SOLVER_CUML_TO_SKLEARN[self.solver_]
        else:
            solver = model.solver

        return {
            "intercept_": np.float64(self.intercept_),
            "coef_": self.coef_.to_output("numpy"),
            "solver_": solver,
            **super()._attrs_to_cpu(model),
        }

    def __init__(self, *, alpha=1.0, solver='auto', fit_intercept=True,
                 normalize=False, handle=None, output_type=None,
                 verbose=False):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.alpha = alpha
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.normalize = normalize

    def _pre_fit(self):
        """Validate hyperparameters, set `solver_` attribute, and get algo value."""
        if self.alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")

        _SUPPORTED_SOLVERS = ["auto", "eig", "svd"]
        if (solver := self.solver) not in _SUPPORTED_SOLVERS:
            raise ValueError(
                f"Expected `solver` to be one of {_SUPPORTED_SOLVERS}, got {solver!r}"
            )

        if self.n_features_in_ == 1:
            # Only `svd` supports 1 column data currently. Warn if `eig`
            # explicitly selected and fallback to `svd`
            if self.solver == "eig":
                warnings.warn(
                    (
                        "Changing solver to 'svd' as 'eig' solver does not support "
                        "training data with 1 column currently"
                    ),
                    UserWarning
                )
            solver = "svd"
        else:
            solver = "eig" if self.solver == "auto" else self.solver

        self.solver_ = solver

        return {"svd": 0, "eig": 1}[solver]

    @generate_docstring()
    def fit(self, X, y, sample_weight=None, *, convert_dtype=True) -> "Ridge":
        """
        Fit the model with X and y.
        """
        cdef size_t n_rows, n_cols
        X, n_rows, n_cols, dtype = input_to_cuml_array(
            X,
            deepcopy=True,
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[np.float32, np.float64]
        )

        if n_cols < 1:
            raise ValueError(
                f"Found array with {n_cols} feature(s) (shape={X.shape}) while "
                f"a minimum of 1 is required."
            )

        if n_rows < 2:
            raise ValueError(
                f"Found array with {n_rows} sample(s) (shape={X.shape}) while a "
                f"minimum of 2 is required."
            )

        y = input_to_cuml_array(
            y,
            check_dtype=dtype,
            convert_to_dtype=(dtype if convert_dtype else None),
            check_rows=n_rows,
            check_cols=1,
        ).array

        if sample_weight is not None:
            sample_weight = input_to_cuml_array(
                sample_weight,
                check_dtype=dtype,
                convert_to_dtype=(dtype if convert_dtype else None),
                check_rows=n_rows,
                check_cols=1,
            ).array

        # Validate hyperparameters and set `solver_`
        cdef int algo = self._pre_fit()

        # Allocate outputs
        coef = CumlArray.zeros(n_cols, dtype=dtype)

        cdef float intercept_f32
        cdef double intercept_f64
        cdef float alpha_f32 = self.alpha
        cdef double alpha_f64 = self.alpha

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t X_ptr = X.ptr
        cdef uintptr_t y_ptr = y.ptr
        cdef uintptr_t coef_ptr = coef.ptr
        cdef uintptr_t sample_weight_ptr = 0 if sample_weight is None else sample_weight.ptr
        cdef bool normalize = self.normalize
        cdef bool fit_intercept = self.fit_intercept
        cdef bool use_float32 = dtype == np.float32

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
                    algo,
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
                    algo,
                    <double*>sample_weight_ptr,
                )
        self.handle.sync()

        self.coef_ = coef
        self.intercept_ = intercept_f32 if use_float32 else intercept_f64

        return self
