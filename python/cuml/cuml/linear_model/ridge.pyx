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

# distutils: language = c++

import warnings

import numpy as np

from libc.stdint cimport uintptr_t

from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base, deprecate_non_keyword_only
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_gpu,
    warn_legacy_device_interop,
)
from cuml.internals.mixins import FMajorInputTagMixin, RegressorMixin
from cuml.linear_model.base import LinearPredictMixin

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
    "cd": "sag",
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
    compliant), in addition to cuDF objects. It provides 3
    algorithms: SVD, Eig and CD to fit a linear model. In general SVD uses
    significantly more memory and is slower than Eig. If using CUDA 10.1,
    the memory difference is even bigger than in the other supported CUDA
    versions. However, SVD is more stable than Eig (default). CD uses
    Coordinate Descent and can be faster when data is large.

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
    solver : {'auto', 'eig', 'svd', 'cd'} (default = 'auto')
        Eig uses a eigendecomposition of the covariance matrix, and is much
        faster.
        SVD is slower, but guaranteed to be stable.
        CD or Coordinate Descent is very fast and is suitable for large
        problems.
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
    intercept_ : array
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
    intercept_ = CumlArrayDescriptor(order='F')

    _cpu_class_path = "sklearn.linear_model.Ridge"

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + ['solver', 'fit_intercept', 'normalize', 'alpha']

    @classmethod
    def _params_from_cpu(cls, model):
        if model.positive:
            raise UnsupportedOnGPU

        solver = _SOLVER_SKLEARN_TO_CUML.get(model.solver)
        if solver is None:
            raise UnsupportedOnGPU

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
            raise UnsupportedOnGPU

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
        """
        Initializes the linear ridge regression class.

        Parameters
        ----------
        solver : Type: string. 'auto' (default), 'eig', 'svd', and 'cd' are supported
        algorithms. The 'auto' option will use 'eig' as the solver.
        fit_intercept: boolean. For more information, see `scikitlearn's OLS
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.
        normalize: boolean. For more information, see `scikitlearn's OLS
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.

        """
        self._check_alpha(alpha)
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

        # internal array attributes
        self.coef_ = None
        self.intercept_ = None

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.solver = solver

        self.intercept_value = 0.0

    def _check_alpha(self, alpha):
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

    def _select_solver(self, solver):
        """
        Select the solver based on the input solver and set both solver_ and algo attributes.
        """
        if solver == 'auto':
            self.solver_ = 'eig'
        elif solver in ("eig", "svd", "cd"):
            self.solver_ = solver
        else:
            # TODO (csadorf): this should be a ValueError, but using using
            # TypeError for backwards compatibility
            raise TypeError(f"solver {solver!r} is not supported")

        self.algo = {'svd': 0, 'eig': 1, 'cd': 2}[self.solver_]

    @generate_docstring()
    @warn_legacy_device_interop
    @deprecate_non_keyword_only("convert_dtype")
    def fit(self, X, y, convert_dtype=True, sample_weight=None) -> "Ridge":
        """
        Fit the model with X and y.
        """
        self._select_solver(self.solver)

        cdef uintptr_t _X_ptr, _y_ptr, _sample_weight_ptr
        X_m, n_rows, self.n_features_in_, self.dtype = \
            input_to_cuml_array(X, deepcopy=True,
                                convert_to_dtype=(np.float32 if convert_dtype
                                                  else None),
                                check_dtype=[np.float32, np.float64])
        _X_ptr = X_m.ptr
        self.feature_names_in_ = X_m.index

        y_m, _, _, _ = \
            input_to_cuml_array(y, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_rows=n_rows, check_cols=1)
        _y_ptr = y_m.ptr

        if sample_weight is not None:
            sample_weight_m, _, _, _ = \
                input_to_cuml_array(sample_weight, check_dtype=self.dtype,
                                    convert_to_dtype=(
                                        self.dtype if convert_dtype else None),
                                    check_rows=n_rows, check_cols=1)
            _sample_weight_ptr = sample_weight_m.ptr
        else:
            _sample_weight_ptr = 0

        if self.n_features_in_ < 1:
            msg = "X matrix must have at least a column"
            raise TypeError(msg)

        if n_rows < 2:
            msg = "X matrix must have at least two rows"
            raise TypeError(msg)

        if self.n_features_in_ == 1 and self.algo != 0:
            warnings.warn("Changing solver to 'svd' as 'eig' or 'cd' " +
                          "solvers do not support training data with 1 " +
                          "column currently.", UserWarning)
            self._select_solver('svd')

        self.n_alpha = 1

        # Initialize coef_ with correct shape based on y
        if len(y_m.shape) == 2:
            assert y_m.shape == (n_rows, 1)
            self.coef_ = CumlArray.zeros((1, self.n_features_in_), dtype=self.dtype)
        else:
            self.coef_ = CumlArray.zeros(self.n_features_in_, dtype=self.dtype)
        cdef uintptr_t _coef_ptr = self.coef_.ptr

        cdef float _c_intercept_f32
        cdef double _c_intercept_f64
        cdef float _c_alpha_f32
        cdef double _c_alpha_f64

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            _c_alpha_f32 = self.alpha
            ridgeFit(handle_[0],
                     <float*>_X_ptr,
                     <size_t>n_rows,
                     <size_t>self.n_features_in_,
                     <float*>_y_ptr,
                     <float*>&_c_alpha_f32,
                     <int>self.n_alpha,
                     <float*>_coef_ptr,
                     <float*>&_c_intercept_f32,
                     <bool>self.fit_intercept,
                     <bool>self.normalize,
                     <int>self.algo,
                     <float*>_sample_weight_ptr)

            self.intercept_ = _c_intercept_f32
        else:
            _c_alpha_f64 = self.alpha
            ridgeFit(handle_[0],
                     <double*>_X_ptr,
                     <size_t>n_rows,
                     <size_t>self.n_features_in_,
                     <double*>_y_ptr,
                     <double*>&_c_alpha_f64,
                     <int>self.n_alpha,
                     <double*>_coef_ptr,
                     <double*>&_c_intercept_f64,
                     <bool>self.fit_intercept,
                     <bool>self.normalize,
                     <int>self.algo,
                     <double*>_sample_weight_ptr)

            self.intercept_ = _c_intercept_f64

        self.handle.sync()

        del X_m
        del y_m
        if sample_weight is not None:
            del sample_weight_m

        return self
