#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
import numpy as np
import cupy as cp
import warnings

from numba import cuda
from collections import defaultdict

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml import Handle
from cuml.internals.array import CumlArray
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.base import UniversalBase
from cuml.internals.mixins import RegressorMixin, FMajorInputTagMixin
from cuml.common.doc_utils import generate_docstring
from cuml.linear_model.base import LinearPredictMixin
from pylibraft.common.handle cimport handle_t
from pylibraft.common.handle import Handle
from cuml.common import input_to_cuml_array
from cuml.internals.api_decorators import device_interop_preparation
from cuml.internals.api_decorators import enable_device_interop

cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM":

    cdef void olsFit(handle_t& handle,
                     float *input,
                     size_t n_rows,
                     size_t n_cols,
                     float *labels,
                     float *coef,
                     float *intercept,
                     bool fit_intercept,
                     bool normalize,
                     int algo,
                     float *sample_weight) except +

    cdef void olsFit(handle_t& handle,
                     double *input,
                     size_t n_rows,
                     size_t n_cols,
                     double *labels,
                     double *coef,
                     double *intercept,
                     bool fit_intercept,
                     bool normalize,
                     int algo,
                     double *sample_weight) except +


def divide_non_zero(x1, x2):
    # Value chosen to be consistent with the RAFT implementation in
    # linalg/detail/lstsq.cuh
    eps = 1e-10

    # Do not divide by values of x2 that are smaller than eps
    mask = abs(x2) < eps
    x2[mask] = 1.

    return x1 / x2


def fit_multi_target(X, y, fit_intercept=True, sample_weight=None):
    X = CumlArray.from_input(X)
    y = CumlArray.from_input(y)
    assert X.ndim == 2
    assert y.ndim == 2
    if sample_weight is not None:
        sample_weight = CumlArray.from_input(sample_weight)

    x_rows, x_cols = X.shape
    if x_cols == 0:
        raise ValueError(
            "Number of columns cannot be less than one"
        )
    if x_rows < 2:
        raise ValueError(
            "Number of rows cannot be less than two"
        )
    X_arr = X.to_output('array')
    y_arr = y.to_output('array')

    if fit_intercept:
        # Add column containg ones to fit intercept.
        nrow, ncol = X.shape
        X_wide = X.mem_type.xpy.empty_like(
            X_arr, shape=(nrow, ncol + 1)
        )
        X_wide[:, :ncol] = X_arr
        X_wide[:, ncol] = 1.
        X_arr = X_wide

    if sample_weight is not None:
        sample_weight = X.mem_type.xpy.sqrt(sample_weight)
        X_arr = sample_weight[:, None] * X_arr
        y_arr = sample_weight[:, None] * y_arr

    u, s, vh = X.mem_type.xpy.linalg.svd(X_arr, full_matrices=False)

    params = vh.T @ divide_non_zero(u.T @ y_arr, s[:, None])

    coef = params[:-1] if fit_intercept else params
    intercept = params[-1] if fit_intercept else None

    return (
        CumlArray.from_input(coef),
        None if intercept is None else CumlArray.from_input(intercept)
    )


class LinearRegression(LinearPredictMixin,
                       UniversalBase,
                       RegressorMixin,
                       FMajorInputTagMixin):
    """
    LinearRegression is a simple machine learning model where the response y is
    modelled by a linear combination of the predictors in X.

    cuML's LinearRegression expects either a cuDF DataFrame or a NumPy matrix
    and provides 2 algorithms SVD and Eig to fit a linear model. SVD is more
    stable, but Eig (default) is much faster.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> import cudf

        >>> # Both import methods supported
        >>> from cuml import LinearRegression
        >>> from cuml.linear_model import LinearRegression
        >>> lr = LinearRegression(fit_intercept = True, normalize = False,
        ...                       algorithm = "eig")
        >>> X = cudf.DataFrame()
        >>> X['col1'] = cp.array([1,1,2,2], dtype=cp.float32)
        >>> X['col2'] = cp.array([1,2,2,3], dtype=cp.float32)
        >>> y = cudf.Series(cp.array([6.0, 8.0, 9.0, 11.0], dtype=cp.float32))
        >>> reg = lr.fit(X,y)
        >>> print(reg.coef_)
        0   1.0
        1   2.0
        dtype: float32
        >>> print(reg.intercept_)
        3.0...

        >>> X_new = cudf.DataFrame()
        >>> X_new['col1'] = cp.array([3,2], dtype=cp.float32)
        >>> X_new['col2'] = cp.array([5,5], dtype=cp.float32)
        >>> preds = lr.predict(X_new)
        >>> print(preds) # doctest: +SKIP
        0   15.999...
        1   14.999...
        dtype: float32


    Parameters
    ----------
    algorithm : {'svd', 'eig', 'qr', 'svd-qr', 'svd-jacobi'}, (default = 'eig')
        Choose an algorithm:

          * 'svd' - alias for svd-jacobi;
          * 'eig' - use an eigendecomposition of the covariance matrix;
          * 'qr'  - use QR decomposition algorithm and solve `Rx = Q^T y`
          * 'svd-qr' - compute SVD decomposition using QR algorithm
          * 'svd-jacobi' - compute SVD decomposition using Jacobi iterations.

        Among these algorithms, only 'svd-jacobi' supports the case when the
        number of features is larger than the sample size; this algorithm
        is force-selected automatically in such a case.

        For the broad range of inputs, 'eig' and 'qr' are usually the fastest,
        followed by 'svd-jacobi' and then 'svd-qr'. In theory, SVD-based
        algorithms are more stable.
    fit_intercept : boolean (default = True)
        If True, LinearRegression tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        This parameter is ignored when `fit_intercept` is set to False.
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
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    -----
    LinearRegression suffers from multicollinearity (when columns are
    correlated with each other), and variance explosions from outliers.
    Consider using Ridge Regression to fix the multicollinearity problem, and
    consider maybe first DBSCAN to remove the outliers, or statistical analysis
    to filter possible outliers.

    **Applications of LinearRegression**

        LinearRegression is used in regression tasks where one wants to predict
        say sales or house prices. It is also used in extrapolation or time
        series tasks, dynamic systems modelling and many other machine learning
        tasks. This model should be first tried if the machine learning problem
        is a regression task (predicting a continuous variable).

    For additional information, see `scikitlearn's OLS documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`__.

    For an additional example see `the OLS notebook
    <https://github.com/rapidsai/cuml/blob/main/notebooks/linear_regression_demo.ipynb>`__.
    """

    _cpu_estimator_import_path = 'sklearn.linear_model.LinearRegression'
    coef_ = CumlArrayDescriptor(order='F')
    intercept_ = CumlArrayDescriptor(order='F')

    @device_interop_preparation
    def __init__(self, *, algorithm='eig', fit_intercept=True, normalize=False,
                 handle=None, verbose=False, output_type=None):
        if handle is None and algorithm == 'eig':
            # if possible, create two streams, so that eigenvalue decomposition
            # can benefit from running independent operations concurrently.
            handle = Handle(n_streams=2)
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

        # internal array attributes
        self.coef_ = None
        self.intercept_ = None

        self.fit_intercept = fit_intercept
        self.normalize = normalize
        if algorithm in ['svd', 'eig', 'qr', 'svd-qr', 'svd-jacobi']:
            self.algorithm = algorithm
            self.algo = self._get_algorithm_int(algorithm)
        else:
            msg = "algorithm {!r} is not supported"
            raise TypeError(msg.format(algorithm))

        self.intercept_value = 0.0

    def _get_algorithm_int(self, algorithm):
        return {
            'svd': 0,
            'eig': 1,
            'qr': 2,
            'svd-qr': 3,
            'svd-jacobi': 0
        }[algorithm]

    @generate_docstring()
    @enable_device_interop
    def fit(self, X, y, convert_dtype=True,
            sample_weight=None) -> "LinearRegression":
        """
        Fit the model with X and y.

        """
        cdef uintptr_t X_ptr, y_ptr, sample_weight_ptr
        X_m, n_rows, self.n_features_in_, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64])
        X_ptr = X_m.ptr
        self.feature_names_in_ = X_m.index

        y_m, _, y_cols, _ = \
            input_to_cuml_array(y, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_rows=n_rows)
        y_ptr = y_m.ptr

        if sample_weight is not None:
            sample_weight_m, _, _, _ = \
                input_to_cuml_array(sample_weight, check_dtype=self.dtype,
                                    convert_to_dtype=(
                                        self.dtype if convert_dtype else None),
                                    check_rows=n_rows, check_cols=1)
            sample_weight_ptr = sample_weight_m.ptr
        else:
            sample_weight_ptr = 0

        if self.n_features_in_ < 1:
            msg = "X matrix must have at least a column"
            raise TypeError(msg)

        if n_rows < 2:
            msg = "X matrix must have at least two rows"
            raise TypeError(msg)

        if self.n_features_in_ == 1 and self.algo != 0:
            warnings.warn("Changing solver from 'eig' to 'svd' as eig " +
                          "solver does not support training data with 1 " +
                          "column currently.", UserWarning)
            self.algo = 0

        if 1 < y_cols:
            if sample_weight is None:
                sample_weight_m = None

            return self._fit_multi_target(
                X_m, y_m, convert_dtype, sample_weight_m
            )

        self.coef_ = CumlArray.zeros(self.n_features_in_, dtype=self.dtype)
        cdef uintptr_t coef_ptr = self.coef_.ptr

        cdef float c_intercept1
        cdef double c_intercept2
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:

            olsFit(handle_[0],
                   <float*>X_ptr,
                   <size_t>n_rows,
                   <size_t>self.n_features_in_,
                   <float*>y_ptr,
                   <float*>coef_ptr,
                   <float*>&c_intercept1,
                   <bool>self.fit_intercept,
                   <bool>self.normalize,
                   <int>self.algo,
                   <float*>sample_weight_ptr)

            self.intercept_ = c_intercept1
        else:
            olsFit(handle_[0],
                   <double*>X_ptr,
                   <size_t>n_rows,
                   <size_t>self.n_features_in_,
                   <double*>y_ptr,
                   <double*>coef_ptr,
                   <double*>&c_intercept2,
                   <bool>self.fit_intercept,
                   <bool>self.normalize,
                   <int>self.algo,
                   <double*>sample_weight_ptr)

            self.intercept_ = c_intercept2

        self.handle.sync()

        del X_m
        del y_m
        if sample_weight is not None:
            del sample_weight_m

        return self

    def _fit_multi_target(self, X, y, convert_dtype=True, sample_weight=None):
        # In the cuml C++ layer, there is no support yet for multi-target
        # regression, i.e., a y vector with multiple columns.
        # We implement the regression in Python here.

        X = CumlArray.from_input(
            X,
            convert_to_dtype=(self.dtype if convert_dtype else None)
        )
        y = CumlArray.from_input(
            y,
            convert_to_dtype=(self.dtype if convert_dtype else None)
        )
        try:
            y_cols = y.shape[1]
        except IndexError:
            y_cols = 1

        if self.algo != 0:
            warnings.warn("Changing solver to 'svd' as this is the " +
                          "only solver that support multiple targets " +
                          "currently.", UserWarning)
            self.algo = 0
        if self.normalize:
            raise ValueError(
                "The normalize option is not supported when `y` has "
                "multiple columns."
            )

        if sample_weight is not None:
            sample_weight = CumlArray.from_input(
                sample_weight,
                convert_to_dtype=(self.dtype if convert_dtype else None),
            )
        coef, intercept = fit_multi_target(
            X,
            y,
            fit_intercept=self.fit_intercept,
            sample_weight=sample_weight
        )
        self.coef_ = CumlArray.from_input(
            coef,
            check_dtype=self.dtype,
            check_rows=self.n_features_in_,
            check_cols=y_cols
        )
        if self.fit_intercept:
            self.intercept_ = CumlArray.from_input(
                intercept,
                check_dtype=self.dtype,
                check_rows=y_cols,
                check_cols=1
            )
        else:
            self.intercept_ = CumlArray.zeros(y_cols, dtype=self.dtype)

        return self

    def _predict(self, X, convert_dtype=True) -> CumlArray:
        self.dtype = self.coef_.dtype
        self.features_in_ = self.coef_.shape[0]
        # Adding UniversalBase here skips it in the Method Resolution Order
        # (MRO) Since UniversalBase and LinearPredictMixin now both have a
        # `predict` method
        return super()._predict(X, convert_dtype=convert_dtype)

    def get_param_names(self):
        return super().get_param_names() + \
            ['algorithm', 'fit_intercept', 'normalize']

    def get_attr_names(self):
        return ['coef_', 'intercept_', 'n_features_in_', 'feature_names_in_']

    @staticmethod
    def _more_static_tags():
        return {"multioutput": True}
