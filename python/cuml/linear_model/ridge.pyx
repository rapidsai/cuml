#
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

import ctypes
import cudf
import numpy as np
from collections import defaultdict
from numba import cuda
import warnings

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.base import Base
from cuml.common.mixins import RegressorMixin
from cuml.common.array import CumlArray
from cuml.common.doc_utils import generate_docstring
from cuml.linear_model.base import LinearPredictMixin
from cuml.raft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from cuml.common.mixins import FMajorInputTagMixin

cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM":

    cdef void ridgeFit(handle_t& handle,
                       float *input,
                       int n_rows,
                       int n_cols,
                       float *labels,
                       float *alpha,
                       int n_alpha,
                       float *coef,
                       float *intercept,
                       bool fit_intercept,
                       bool normalize,
                       int algo) except +

    cdef void ridgeFit(handle_t& handle,
                       double *input,
                       int n_rows,
                       int n_cols,
                       double *labels,
                       double *alpha,
                       int n_alpha,
                       double *coef,
                       double *intercept,
                       bool fit_intercept,
                       bool normalize,
                       int algo) except +


class Ridge(Base,
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

        import numpy as np
        import cudf

        # Both import methods supported
        from cuml import Ridge
        from cuml.linear_model import Ridge

        alpha = np.array([1e-5])
        ridge = Ridge(alpha = alpha, fit_intercept = True, normalize = False,
                      solver = "eig")

        X = cudf.DataFrame()
        X['col1'] = np.array([1,1,2,2], dtype = np.float32)
        X['col2'] = np.array([1,2,2,3], dtype = np.float32)

        y = cudf.Series( np.array([6.0, 8.0, 9.0, 11.0], dtype = np.float32) )

        result_ridge = ridge.fit(X, y)
        print("Coefficients:")
        print(result_ridge.coef_)
        print("Intercept:")
        print(result_ridge.intercept_)

        X_new = cudf.DataFrame()
        X_new['col1'] = np.array([3,2], dtype = np.float32)
        X_new['col2'] = np.array([5,5], dtype = np.float32)
        preds = result_ridge.predict(X_new)

        print("Predictions:")
        print(preds)

    Output:

    .. code-block:: python

        Coefficients:

                    0 1.0000001
                    1 1.9999998

        Intercept:
                    3.0

        Preds:

                    0 15.999999
                    1 14.999999

    Parameters
    -----------
    alpha : float (default = 1.0)
        Regularization strength - must be a positive float. Larger values
        specify stronger regularization. Array input will be supported later.
    solver : {'eig', 'svd', 'cd'} (default = 'eig')
        Eig uses a eigendecomposition of the covariance matrix, and is much
        faster.
        SVD is slower, but guaranteed to be stable.
        CD or Coordinate Descent is very fast and is suitable for large
        problems.
    fit_intercept : boolean (default = True)
        If True, Ridge tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        If True, the predictors in X will be normalized by dividing by it's L2
        norm.
        If False, no scaling will be done.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    Attributes
    -----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    ------
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

    coef_ = CumlArrayDescriptor()
    intercept_ = CumlArrayDescriptor()

    def __init__(self, alpha=1.0, solver='eig', fit_intercept=True,
                 normalize=False, handle=None, output_type=None,
                 verbose=False):
        """
        Initializes the linear ridge regression class.

        Parameters
        ----------
        solver : Type: string. 'eig' (default) and 'svd' are supported
        algorithms.
        fit_intercept: boolean. For more information, see `scikitlearn's OLS
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.
        normalize: boolean. For more information, see `scikitlearn's OLS
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.

        """
        self._check_alpha(alpha)
        super(Ridge, self).__init__(handle=handle, verbose=verbose,
                                    output_type=output_type)

        # internal array attributes
        self.coef_ = None
        self.intercept_ = None

        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize

        if solver in ['svd', 'eig', 'cd']:
            self.solver = solver
            self.algo = self._get_algorithm_int(solver)
        else:
            msg = "solver {!r} is not supported"
            raise TypeError(msg.format(solver))
        self.intercept_value = 0.0

    def _check_alpha(self, alpha):
        if alpha <= 0.0:
            msg = "alpha value has to be positive"
            raise TypeError(msg.format(alpha))

    def _get_algorithm_int(self, algorithm):
        return {
            'svd': 0,
            'eig': 1,
            'cd': 2
        }[algorithm]

    @generate_docstring()
    def fit(self, X, y, convert_dtype=True) -> "Ridge":
        """
        Fit the model with X and y.

        """
        cdef uintptr_t X_ptr, y_ptr
        X_m, n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64])
        X_ptr = X_m.ptr

        y_m, _, _, _ = \
            input_to_cuml_array(y, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_rows=n_rows, check_cols=1)
        y_ptr = y_m.ptr

        if self.n_cols < 1:
            msg = "X matrix must have at least a column"
            raise TypeError(msg)

        if n_rows < 2:
            msg = "X matrix must have at least two rows"
            raise TypeError(msg)

        if self.n_cols == 1 and self.algo != 0:
            warnings.warn("Changing solver to 'svd' as 'eig' or 'cd' " +
                          "solvers do not support training data with 1 " +
                          "column currently.", UserWarning)
            self.algo = 0

        self.n_alpha = 1

        self.coef_ = CumlArray.zeros(self.n_cols, dtype=self.dtype)
        cdef uintptr_t coef_ptr = self.coef_.ptr

        cdef float c_intercept1
        cdef double c_intercept2
        cdef float c_alpha1
        cdef double c_alpha2
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            c_alpha1 = self.alpha
            ridgeFit(handle_[0],
                     <float*>X_ptr,
                     <int>n_rows,
                     <int>self.n_cols,
                     <float*>y_ptr,
                     <float*>&c_alpha1,
                     <int>self.n_alpha,
                     <float*>coef_ptr,
                     <float*>&c_intercept1,
                     <bool>self.fit_intercept,
                     <bool>self.normalize,
                     <int>self.algo)

            self.intercept_ = c_intercept1
        else:
            c_alpha2 = self.alpha

            ridgeFit(handle_[0],
                     <double*>X_ptr,
                     <int>n_rows,
                     <int>self.n_cols,
                     <double*>y_ptr,
                     <double*>&c_alpha2,
                     <int>self.n_alpha,
                     <double*>coef_ptr,
                     <double*>&c_intercept2,
                     <bool>self.fit_intercept,
                     <bool>self.normalize,
                     <int>self.algo)

            self.intercept_ = c_intercept2

        self.handle.sync()

        del X_m
        del y_m

        return self

    def get_param_names(self):
        return super().get_param_names() + \
            ['solver', 'fit_intercept', 'normalize', 'alpha']
