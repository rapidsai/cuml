#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import cupy as cp
import numpy as np

from cuml.internals import logger

from cuml.internals cimport logger

import cuml.internals

from libc.stdint cimport uintptr_t
from libcpp cimport nullptr

from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.mixins import RegressorMixin

from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/solvers/lars.hpp" namespace "ML::Solver::Lars" nogil:

    cdef void larsFit[math_t](
        const handle_t& handle, math_t* X, int n_rows, int n_cols,
        const math_t* y, math_t* beta, int* active_idx, math_t* alphas,
        int* n_active, math_t* Gram, int max_iter, math_t* coef_path,
        logger.level_enum verbosity, int ld_X, int ld_G, math_t epsilon) except +

    cdef void larsPredict[math_t](
        const handle_t& handle, const math_t* X, int n_rows, int n_cols,
        int ld_X, const math_t* beta, int n_active, int* active_idx,
        math_t intercept, math_t* preds) except +


class Lars(Base, RegressorMixin):
    """
    Least Angle Regression

    Least Angle Regression (LAR or LARS) is a model selection algorithm. It
    builds up the model using the following algorithm:

    1. We start with all the coefficients equal to zero.
    2. At each step we select the predictor that has the largest absolute
       correlation with the residual.
    3. We take the largest step possible in the direction which is equiangular
       with all the predictors selected so far. The largest step is determined
       such that using this step a new predictor will have as much correlation
       with the residual as any of the currently active predictors.
    4. Stop if `max_iter` reached or all the predictors are used, or if the
       correlation between any unused predictor and the residual is lower than
       a tolerance.

    The solver is based on [1]_. The equations referred in the comments
    correspond to the equations in the paper.

    .. note:: This algorithm assumes that the offset is removed from `X` and
        `y`, and each feature is normalized:

        .. math::

            sum_i y_i = 0, sum_i x_{i,j} = 0,sum_i x_{i,j}^2=1 \
            for j=0..n_{col}-1

    Parameters
    ----------
    fit_intercept : boolean (default = True)
        If True, Lars tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    normalize : boolean (default = False)
        This parameter is ignored when `fit_intercept` is set to False.
        If True, the predictors in X will be normalized by removing its mean
        and dividing by it's variance. If False, then the solver expects that
        the data is already normalized.

        .. versionchanged:: 24.06
            The default of `normalize` changed from `True` to `False`.

    copy_X : boolean (default = True)
        The solver permutes the columns of X. Set `copy_X` to True to prevent
        changing the input data.
    fit_path : boolean (default = True)
        Whether to return all the coefficients along the regularization path
        in the `coef_path_` attribute.
    precompute : bool, 'auto', or array-like with shape = (n_features, \
            n_features). (default = 'auto')
        Whether to precompute the Gram matrix. The user can provide the Gram
        matrix as an argument.
    n_nonzero_coefs : int (default 500)
        The maximum number of coefficients to fit. This gives an upper limit of
        how many features we select for prediction. This number is also an
        upper limit of the number of iterations.
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
    alphas_ : array of floats or doubles, shape = [n_alphas + 1]
        The maximum correlation at each step.
    active_ : array of ints shape = [n_alphas]
        The indices of the active variables at the end of the path.
    beta_ : array of floats or doubles [n_asphas]
        The active regression coefficients (same as `coef_` but zeros omitted).
    coef_path_ : array of floats or doubles, shape = [n_alphas, n_alphas + 1]
        The coefficients along the regularization path. Stored only if
        `fit_path` is True. Note that we only store coefficients for indices
        in the active set (i.e. :py:`coef_path_[:,-1] == coef_[active_]`)
    coef_ : array, shape (n_features)
        The estimated coefficients for the regression model.
    intercept_ : scalar, float or double
        The independent term. If `fit_intercept_` is False, will be 0.
    n_iter_ : int
        The number of iterations taken by the solver.

    Notes
    -----
    For additional information, see `scikitlearn's OLS documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html>`__.

    References
    ----------
    .. [1] `B. Efron, T. Hastie, I. Johnstone, R Tibshirani, Least Angle
       Regression The Annals of Statistics (2004) Vol 32, No 2, 407-499
       <http://statweb.stanford.edu/~tibs/ftp/lars.pdf>`_

    """

    alphas_ = CumlArrayDescriptor()
    active_ = CumlArrayDescriptor()
    beta_ = CumlArrayDescriptor()
    coef_path_ = CumlArrayDescriptor()
    coef_ = CumlArrayDescriptor()
    intercept_ = CumlArrayDescriptor()

    def __init__(self, *, fit_intercept=True, normalize=True,
                 handle=None, verbose=False, output_type=None, copy_X=True,
                 fit_path=True, n_nonzero_coefs=500, eps=None,
                 precompute='auto'):
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.eps = eps
        self.fit_path = fit_path
        self.n_nonzero_coefs = n_nonzero_coefs  # this corresponds to max_iter
        self.precompute = precompute

    def _preprocess_data(self, X_m, y_m):
        """ Remove mean and scale each feature column. """
        x_mean = cp.zeros(self.n_cols, dtype=self.dtype)
        x_scale = cp.ones(self.n_cols, dtype=self.dtype)
        y_mean = self.dtype.type(0.0)
        X = cp.asarray(X_m)
        y = cp.asarray(y_m)
        if self.fit_intercept:
            y_mean = cp.mean(y)
            y = y - y_mean
            if self.normalize:
                x_mean = cp.mean(X, axis=0)
                x_scale = cp.sqrt(cp.var(X, axis=0) *
                                  self.dtype.type(X.shape[0]))
                x_scale[x_scale==0] = 1
                X = (X - x_mean) / x_scale
        return X, y, x_mean, x_scale, y_mean

    def _set_intercept(self, x_mean, x_scale, y_mean):
        """ Set the intercept value and scale coefficients. """
        if self.fit_intercept:
            with cuml.using_output_type('cupy'):
                self.coef_ = self.coef_ / x_scale
                self.intercept_ = y_mean - cp.dot(x_mean, self.coef_.T)
                self.intercept_ = self.intercept_.item()
        else:
            self.intercept_ = self.dtype.type(0.0)

    def _calc_gram(self, X):
        """
        Return the Gram matrix, or None if it is not applicable.
        """
        Gram = None
        X = cp.asarray(X)
        if self.precompute is True or (self.precompute == 'auto' and
                                       self.n_cols < X.shape[0]):
            logger.debug('Calculating Gram matrix')
            try:
                Gram = cp.dot(X.T, X)
            except MemoryError as err:
                if self.precompute:
                    logger.debug("Not enough memory to store the Gram matrix."
                                 " Proceeding without it.")
        return Gram

    def _fit_cpp(self, X, y, Gram, x_scale, convert_dtype):
        """ Fit lars model using cpp solver"""
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        X_m, _, _, _ = \
            input_to_cuml_array(X,
                                convert_to_dtype=(np.float32 if convert_dtype
                                                  else None),
                                check_dtype=self.dtype,
                                order='F')
        cdef uintptr_t X_ptr = X_m.ptr
        cdef int n_rows = X.shape[0]
        cdef uintptr_t y_ptr = \
            input_to_cuml_array(y,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_dtype=self.dtype).array.ptr
        cdef int max_iter = self.n_nonzero_coefs
        self.beta_ = CumlArray.zeros(max_iter, dtype=self.dtype)
        cdef uintptr_t beta_ptr = self.beta_.ptr
        self.active_ = CumlArray.zeros(max_iter, dtype=np.int32)
        cdef uintptr_t active_idx_ptr = self.active_.ptr
        self.alphas_ = CumlArray.zeros(max_iter+1, dtype=self.dtype)
        cdef uintptr_t alphas_ptr = self.alphas_.ptr
        cdef int n_active
        cdef uintptr_t Gram_ptr = <uintptr_t> nullptr
        if Gram is not None:
            Gram_m, _, _, _ = input_to_cuml_array(Gram)
            Gram_ptr = Gram_m.ptr
        cdef uintptr_t coef_path_ptr = <uintptr_t> nullptr
        if (self.fit_path):
            try:
                self.coef_path_ = CumlArray.zeros((max_iter, max_iter+1),
                                                  dtype=self.dtype, order='F')
            except MemoryError as err:
                raise MemoryError("Not enough memory to store coef_path_. "
                                  "Try to decrease n_nonzero_coefs or set "
                                  "fit_path=False.") from err
            coef_path_ptr = self.coef_path_.ptr
        cdef int ld_X = n_rows
        cdef int ld_G = self.n_cols

        if self.dtype == np.float32:
            larsFit(handle_[0], <float*> X_ptr, n_rows, <int> self.n_cols,
                    <float*> y_ptr, <float*> beta_ptr, <int*> active_idx_ptr,
                    <float*> alphas_ptr, &n_active, <float*> Gram_ptr,
                    max_iter, <float*> coef_path_ptr, self.verbose, ld_X,
                    ld_G, <float> self.eps)
        else:
            larsFit(handle_[0], <double*> X_ptr, n_rows, <int> self.n_cols,
                    <double*> y_ptr, <double*> beta_ptr, <int*> active_idx_ptr,
                    <double*> alphas_ptr, &n_active, <double*> Gram_ptr,
                    max_iter, <double*> coef_path_ptr, self.verbose,
                    ld_X, ld_G, <double> self.eps)
        self.n_active = n_active
        self.n_iter_ = n_active

        with cuml.using_output_type("cupy"):
            self.active_ = self.active_[:n_active]
            self.beta_ = self.beta_[:n_active]
            self.alphas_ = self.alphas_[:n_active+1]

            self.coef_ = cp.zeros(self.n_cols, dtype=self.dtype)
            self.coef_[self.active_] = self.beta_

            if self.fit_intercept:
                self.beta_ = self.beta_ / x_scale[self.active_]

    @generate_docstring(y='dense_anydtype')
    def fit(self, X, y, convert_dtype=True) -> 'Lars':
        """
        Fit the model with X and y.

        """
        self._set_n_features_in(X)
        self._set_output_type(X)

        X_m, n_rows, self.n_cols, self.dtype = input_to_cuml_array(
            X, check_dtype=[np.float32, np.float64], order='F')

        conv_dtype = self.dtype if convert_dtype else None
        y_m, _, _, _ = input_to_cuml_array(
            y, order='F', check_dtype=self.dtype, convert_to_dtype=conv_dtype,
            check_rows=n_rows, check_cols=1)

        X, y, x_mean, x_scale, y_scale = self._preprocess_data(X_m, y_m)

        if hasattr(self.precompute, '__cuda_array_interface__') or \
                hasattr(self.precompute, '__array_interface__'):
            Gram, _, _, _ = \
                input_to_cuml_array(self.precompute, order='F',
                                    check_dtype=[np.float32, np.float64],
                                    convert_to_dtype=conv_dtype,
                                    check_rows=self.n_cols,
                                    check_cols=self.n_cols)
            logger.debug('Using precalculated Gram matrix')
        else:
            Gram = self._calc_gram(X)

        if Gram is None and self.copy_X and not isinstance(X, np.ndarray):
            # Without Gram matrix, the solver will permute columns of X
            # We make a copy here, and work on the copy.
            X = cp.copy(X)

        if self.eps is None:
            self.eps = np.finfo(float).eps

        self._fit_cpp(X, y, Gram, x_scale, convert_dtype)

        self._set_intercept(x_mean, x_scale, y_scale)

        self.handle.sync()

        del X_m
        del y_m
        del Gram

        return self

    def predict(self, X, convert_dtype=True) -> CumlArray:
        """
        Predicts `y` values for `X`.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = True)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.

        Returns
        -------
        y: cuDF DataFrame
           Dense vector (floats or doubles) of shape (n_samples, 1)

        """
        conv_dtype=(self.dtype if convert_dtype else None)
        X_m, n_rows, _n_cols, _dtype = input_to_cuml_array(
            X, check_dtype=self.dtype, convert_to_dtype=conv_dtype,
            check_cols=self.n_cols, order='F')

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t X_ptr = X_m.ptr
        cdef int ld_X = n_rows
        cdef uintptr_t beta_ptr = input_to_cuml_array(self.beta_).array.ptr
        cdef uintptr_t active_idx_ptr = \
            input_to_cuml_array(self.active_).array.ptr

        preds = CumlArray.zeros(n_rows, dtype=self.dtype, index=X_m.index)

        if self.dtype == np.float32:
            larsPredict(handle_[0], <float*> X_ptr, <int> n_rows,
                        <int> self.n_cols, ld_X, <float*> beta_ptr,
                        <int> self.n_active, <int*> active_idx_ptr,
                        <float> self.intercept_,
                        <float*><uintptr_t> preds.ptr)
        else:
            larsPredict(handle_[0], <double*> X_ptr, <int> n_rows,
                        <int> self.n_cols, ld_X, <double*> beta_ptr,
                        <int> self.n_active, <int*> active_idx_ptr,
                        <double> self.intercept_,
                        <double*><uintptr_t> preds.ptr)

        self.handle.sync()
        del X_m

        return preds

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + \
            ['copy_X', 'fit_intercept', 'fit_path', 'n_nonzero_coefs',
             'normalize', 'precompute', 'eps']
