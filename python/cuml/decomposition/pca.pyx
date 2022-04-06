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
import cudf
import numpy as np
import cupy as cp
import cupyx
import scipy

from enum import IntEnum

import rmm

from libcpp cimport bool
from libc.stdint cimport uintptr_t

from cython.operator cimport dereference as deref

import cuml.internals
from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.doc_utils import generate_docstring
from raft.common.handle cimport handle_t
from raft.common.handle import Handle
import cuml.common.logger as logger
from cuml.decomposition.utils cimport *
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.input_utils import input_to_cupy_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common import using_output_type
from cuml.prims.stats import cov
from cuml.common.input_utils import sparse_scipy_to_cp
from cuml.common.exceptions import NotFittedError
from cuml.common.mixins import FMajorInputTagMixin
from cuml.common.mixins import SparseInputTagMixin


cdef extern from "cuml/decomposition/pca.hpp" namespace "ML":

    cdef void pcaFit(handle_t& handle,
                     float *input,
                     float *components,
                     float *explained_var,
                     float *explained_var_ratio,
                     float *singular_vals,
                     float *mu,
                     float *noise_vars,
                     const paramsPCA &prms) except +

    cdef void pcaFit(handle_t& handle,
                     double *input,
                     double *components,
                     double *explained_var,
                     double *explained_var_ratio,
                     double *singular_vals,
                     double *mu,
                     double *noise_vars,
                     const paramsPCA &prms) except +

    cdef void pcaInverseTransform(handle_t& handle,
                                  float *trans_input,
                                  float *components,
                                  float *singular_vals,
                                  float *mu,
                                  float *input,
                                  const paramsPCA &prms) except +

    cdef void pcaInverseTransform(handle_t& handle,
                                  double *trans_input,
                                  double *components,
                                  double *singular_vals,
                                  double *mu,
                                  double *input,
                                  const paramsPCA &prms) except +

    cdef void pcaTransform(handle_t& handle,
                           float *input,
                           float *components,
                           float *trans_input,
                           float *singular_vals,
                           float *mu,
                           const paramsPCA &prms) except +

    cdef void pcaTransform(handle_t& handle,
                           double *input,
                           double *components,
                           double *trans_input,
                           double *singular_vals,
                           double *mu,
                           const paramsPCA &prms) except +


class Solver(IntEnum):
    COV_EIG_DQ = <underlying_type_t_solver> solver.COV_EIG_DQ
    COV_EIG_JACOBI = <underlying_type_t_solver> solver.COV_EIG_JACOBI


class PCA(Base,
          FMajorInputTagMixin,
          SparseInputTagMixin):

    """
    PCA (Principal Component Analysis) is a fundamental dimensionality
    reduction technique used to combine features in X in linear combinations
    such that each new component captures the most information or variance of
    the data. N_components is usually small, say at 3, where it can be used for
    data visualization, data compression and exploratory analysis.

    cuML's PCA expects an array-like object or cuDF DataFrame, and provides 2
    algorithms Full and Jacobi. Full (default) uses a full eigendecomposition
    then selects the top K eigenvectors. The Jacobi algorithm is much faster
    as it iteratively tries to correct the top K eigenvectors, but might be
    less accurate.

    Examples
    --------

    .. code-block:: python

        >>> # Both import methods supported
        >>> from cuml import PCA
        >>> from cuml.decomposition import PCA

        >>> import cudf
        >>> import cupy as cp

        >>> gdf_float = cudf.DataFrame()
        >>> gdf_float['0'] = cp.asarray([1.0,2.0,5.0], dtype = cp.float32)
        >>> gdf_float['1'] = cp.asarray([4.0,2.0,1.0], dtype = cp.float32)
        >>> gdf_float['2'] = cp.asarray([4.0,2.0,1.0], dtype = cp.float32)

        >>> pca_float = PCA(n_components = 2)
        >>> pca_float.fit(gdf_float)
        PCA()

        >>> print(f'components: {pca_float.components_}') # doctest: +SKIP
        components: 0           1           2
        0  0.69225764  -0.5102837 -0.51028395
        1 -0.72165036 -0.48949987  -0.4895003
        >>> print(f'explained variance: {pca_float.explained_variance_}')
        explained variance: 0   8.510...
        1 0.489...
        dtype: float32
        >>> exp_var = pca_float.explained_variance_ratio_
        >>> print(f'explained variance ratio: {exp_var}')
        explained variance ratio: 0   0.9456...
        1 0.054...
        dtype: float32

        >>> print(f'singular values: {pca_float.singular_values_}')
        singular values: 0 4.125...
        1 0.989...
        dtype: float32
        >>> print(f'mean: {pca_float.mean_}')
        mean: 0 2.666...
        1 2.333...
        2 2.333...
        dtype: float32
        >>> print(f'noise variance: {pca_float.noise_variance_}')
        noise variance: 0  0.0
        dtype: float32
        >>> trans_gdf_float = pca_float.transform(gdf_float)
        >>> print(f'Inverse: {trans_gdf_float}') # doctest: +SKIP
        Inverse: 0           1
        0   -2.8547091 -0.42891636
        1 -0.121316016  0.80743366
        2    2.9760244 -0.37851727
        >>> input_gdf_float = pca_float.inverse_transform(trans_gdf_float)
        >>> print(f'Input: {input_gdf_float}') # doctest: +SKIP
        Input: 0         1         2
        0 1.0 4.0 4.0
        1 2.0 2.0 2.0
        2 5.0 1.0 1.0

    Parameters
    ----------
    copy : boolean (default = True)
        If True, then copies data then removes mean from data. False might
        cause data to be overwritten with its mean centered version.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    iterated_power : int (default = 15)
        Used in Jacobi solver. The more iterations, the more accurate, but
        slower.
    n_components : int (default = None)
        The number of top K singular vectors / values you want.
        Must be <= number(columns). If n_components is not set, then all
        components are kept:

            ``n_components = min(n_samples, n_features)``

    random_state : int / None (default = None)
        If you want results to be the same when you restart Python, select a
        state.
    svd_solver : 'full' or 'jacobi' or 'auto' (default = 'full')
        Full uses a eigendecomposition of the covariance matrix then discards
        components.
        Jacobi is much faster as it iteratively corrects, but is less accurate.
    tol : float (default = 1e-7)
        Used if algorithm = "jacobi". Smaller tolerance can increase accuracy,
        but but will slow down the algorithm's convergence.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    whiten : boolean (default = False)
        If True, de-correlates the components. This is done by dividing them by
        the corresponding singular values then multiplying by sqrt(n_samples).
        Whitening allows each component to have unit variance and removes
        multi-collinearity. It might be beneficial for downstream
        tasks like LinearRegression where correlated features cause problems.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    components_ : array
        The top K components (VT.T[:,:n_components]) in U, S, VT = svd(X)
    explained_variance_ : array
        How much each component explains the variance in the data given by S**2
    explained_variance_ratio_ : array
        How much in % the variance is explained given by S**2/sum(S**2)
    singular_values_ : array
        The top K singular values. Remember all singular values >= 0
    mean_ : array
        The column wise mean of X. Used to mean - center the data first.
    noise_variance_ : float
        From Bishop 1999's Textbook. Used in later tasks like calculating the
        estimated covariance of X.

    Notes
    ------
    PCA considers linear combinations of features, specifically those that
    maximize global variance structure. This means PCA is fantastic for global
    structure analyses, but weak for local relationships. Consider UMAP or
    T-SNE for a locally important embedding.

    **Applications of PCA**

        PCA is used extensively in practice for data visualization and data
        compression. It has been used to visualize extremely large word
        embeddings like Word2Vec and GloVe in 2 or 3 dimensions, large
        datasets of everyday objects and images, and used to distinguish
        between cancerous cells from healthy cells.


    For additional docs, see `scikitlearn's PCA
    <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_.
    """

    components_ = CumlArrayDescriptor()
    explained_variance_ = CumlArrayDescriptor()
    explained_variance_ratio_ = CumlArrayDescriptor()
    singular_values_ = CumlArrayDescriptor()
    mean_ = CumlArrayDescriptor()
    noise_variance_ = CumlArrayDescriptor()
    trans_input_ = CumlArrayDescriptor()

    def __init__(self, *, copy=True, handle=None, iterated_power=15,
                 n_components=None, random_state=None, svd_solver='auto',
                 tol=1e-7, verbose=False, whiten=False,
                 output_type=None):
        # parameters
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)
        self.copy = copy
        self.iterated_power = iterated_power
        self.n_components = n_components
        self.random_state = random_state
        self.svd_solver = svd_solver
        self.tol = tol
        self.whiten = whiten
        self.c_algorithm = self._get_algorithm_c_name(self.svd_solver)

        # internal array attributes
        self.components_ = None
        self.trans_input_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.noise_variance_ = None

        # This variable controls whether a sparse model was fit
        # This can be removed once there is more inter-operability
        # between cuml.array and cupy.ndarray
        self._sparse_model = None

    def _get_algorithm_c_name(self, algorithm):
        algo_map = {
            'full': Solver.COV_EIG_DQ,
            'auto': Solver.COV_EIG_DQ,
            # 'arpack': NOT_SUPPORTED,
            # 'randomized': NOT_SUPPORTED,
            'jacobi': Solver.COV_EIG_JACOBI
        }
        if algorithm not in algo_map:
            msg = "algorithm {!r} is not supported"
            raise TypeError(msg.format(algorithm))

        return algo_map[algorithm]

    def _build_params(self, n_rows, n_cols):
        cpdef paramsPCA *params = new paramsPCA()
        params.n_components = self._n_components
        params.n_rows = n_rows
        params.n_cols = n_cols
        params.whiten = self.whiten
        params.n_iterations = self.iterated_power
        params.tol = self.tol
        params.algorithm = <solver> (<underlying_type_t_solver> (
            self.c_algorithm))

        return <size_t>params

    def _initialize_arrays(self, n_components, n_rows, n_cols):

        self.components_ = CumlArray.zeros((n_components, n_cols),
                                           dtype=self.dtype)
        self.explained_variance_ = CumlArray.zeros(n_components,
                                                   dtype=self.dtype)
        self.explained_variance_ratio_ = CumlArray.zeros(n_components,
                                                         dtype=self.dtype)
        self.mean_ = CumlArray.zeros(n_cols, dtype=self.dtype)

        self.singular_values_ = CumlArray.zeros(n_components,
                                                dtype=self.dtype)
        self.noise_variance_ = CumlArray.zeros(1, dtype=self.dtype)

    def _sparse_fit(self, X):

        self._sparse_model = True

        self.n_rows = X.shape[0]
        self.n_cols = X.shape[1]
        self.dtype = X.dtype

        # NOTE: All intermediate calculations are done using cupy.ndarray and
        # then converted to CumlArray at the end to minimize conversions
        # between types
        covariance, self.mean_, _ = cov(X, X, return_mean=True)

        self.explained_variance_, self.components_ = \
            cp.linalg.eigh(covariance, UPLO='U')

        # NOTE: We reverse the eigen vector and eigen values here
        # because cupy provides them in ascending order. Make a copy otherwise
        # it is not C_CONTIGUOUS anymore and would error when converting to
        # CumlArray
        self.explained_variance_ = self.explained_variance_[::-1]

        self.components_ = cp.flip(self.components_, axis=1)

        self.components_ = self.components_.T[:self._n_components, :]

        self.explained_variance_ratio_ = self.explained_variance_ / cp.sum(
            self.explained_variance_)

        if self._n_components < min(self.n_rows, self.n_cols):
            self.noise_variance_ = \
                self.explained_variance_[self._n_components:].mean()
        else:
            self.noise_variance_ = cp.array([0.0])

        self.explained_variance_ = \
            self.explained_variance_[:self._n_components]

        self.explained_variance_ratio_ = \
            self.explained_variance_ratio_[:self._n_components]

        # Truncating negative explained variance values to 0
        self.singular_values_ = \
            cp.where(self.explained_variance_ < 0, 0,
                     self.explained_variance_)
        self.singular_values_ = \
            cp.sqrt(self.singular_values_ * (self.n_rows - 1))

        return self

    @generate_docstring(X='dense_sparse')
    def fit(self, X, y=None) -> "PCA":
        """
        Fit the model with X. y is currently ignored.

        """
        if self.n_components is None:
            logger.warn(
                'Warning(`fit`): As of v0.16, PCA invoked without an'
                ' n_components argument defauts to using'
                ' min(n_samples, n_features) rather than 1'
            )
            n_rows = X.shape[0]
            n_cols = X.shape[1]
            self._n_components = min(n_rows, n_cols)
        else:
            self._n_components = self.n_components

        if cupyx.scipy.sparse.issparse(X):
            return self._sparse_fit(X)
        elif scipy.sparse.issparse(X):
            X = sparse_scipy_to_cp(X, dtype=None)
            return self._sparse_fit(X)

        X_m, self.n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64])
        cdef uintptr_t input_ptr = X_m.ptr

        cdef paramsPCA *params = <paramsPCA*><size_t> \
            self._build_params(self.n_rows, self.n_cols)

        if params.n_components > self.n_cols:
            raise ValueError('Number of components should not be greater than'
                             'the number of columns in the data')

        # Calling _initialize_arrays, guarantees everything is CumlArray
        self._initialize_arrays(params.n_components,
                                params.n_rows, params.n_cols)

        cdef uintptr_t comp_ptr = self.components_.ptr

        cdef uintptr_t explained_var_ptr = \
            self.explained_variance_.ptr

        cdef uintptr_t explained_var_ratio_ptr = \
            self.explained_variance_ratio_.ptr

        cdef uintptr_t singular_vals_ptr = \
            self.singular_values_.ptr

        cdef uintptr_t _mean_ptr = self.mean_.ptr

        cdef uintptr_t noise_vars_ptr = \
            self.noise_variance_.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        if self.dtype == np.float32:
            pcaFit(handle_[0],
                   <float*> input_ptr,
                   <float*> comp_ptr,
                   <float*> explained_var_ptr,
                   <float*> explained_var_ratio_ptr,
                   <float*> singular_vals_ptr,
                   <float*> _mean_ptr,
                   <float*> noise_vars_ptr,
                   deref(params))
        else:
            pcaFit(handle_[0],
                   <double*> input_ptr,
                   <double*> comp_ptr,
                   <double*> explained_var_ptr,
                   <double*> explained_var_ratio_ptr,
                   <double*> singular_vals_ptr,
                   <double*> _mean_ptr,
                   <double*> noise_vars_ptr,
                   deref(params))

        # make sure the previously scheduled gpu tasks are complete before the
        # following transfers start
        self.handle.sync()

        return self

    @generate_docstring(X='dense_sparse',
                        return_values={'name': 'trans',
                                       'type': 'dense_sparse',
                                       'description': 'Transformed values',
                                       'shape': '(n_samples, n_components)'})
    @cuml.internals.api_base_return_array_skipall
    def fit_transform(self, X, y=None) -> CumlArray:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        """

        return self.fit(X).transform(X)

    @cuml.internals.api_base_return_array_skipall
    def _sparse_inverse_transform(self, X, return_sparse=False,
                                  sparse_tol=1e-10) -> CumlArray:

        # NOTE: All intermediate calculations are done using cupy.ndarray and
        # then converted to CumlArray at the end to minimize conversions
        # between types

        if self.whiten:
            cp.multiply(self.components_,
                        (1 / cp.sqrt(self.n_rows - 1)), out=self.components_)
            cp.multiply(self.components_,
                        self.singular_values_.reshape((-1, 1)),
                        out=self.components_)

        X_inv = cp.dot(X, self.components_)
        cp.add(X_inv, self.mean_, out=X_inv)

        if self.whiten:
            self.components_ /= self.singular_values_.reshape((-1, 1))
            self.components_ *= cp.sqrt(self.n_rows - 1)

        if return_sparse:
            X_inv = cp.where(X_inv < sparse_tol, 0, X_inv)

            X_inv = cupyx.scipy.sparse.csr_matrix(X_inv)

            return X_inv

        return X_inv

    @generate_docstring(X='dense_sparse',
                        return_values={'name': 'X_inv',
                                       'type': 'dense_sparse',
                                       'description': 'Transformed values',
                                       'shape': '(n_samples, n_features)'})
    def inverse_transform(self, X, convert_dtype=False,
                          return_sparse=False, sparse_tol=1e-10) -> CumlArray:
        """
        Transform data back to its original space.

        In other words, return an input X_original whose transform would be X.

        """

        self._check_is_fitted('components_')
        if cupyx.scipy.sparse.issparse(X):
            return self._sparse_inverse_transform(X,
                                                  return_sparse=return_sparse,
                                                  sparse_tol=sparse_tol)
        elif scipy.sparse.issparse(X):
            X = sparse_scipy_to_cp(X, dtype=None)
            return self._sparse_inverse_transform(X,
                                                  return_sparse=return_sparse,
                                                  sparse_tol=sparse_tol)
        elif self._sparse_model:
            X, _, _, _ = \
                input_to_cupy_array(X, order='K',
                                    check_dtype=[cp.float32, cp.float64])
            return self._sparse_inverse_transform(X,
                                                  return_sparse=return_sparse,
                                                  sparse_tol=sparse_tol)

        X_m, n_rows, _, dtype = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None)
                                )

        cdef uintptr_t _trans_input_ptr = X_m.ptr

        # todo: check n_cols and dtype
        cpdef paramsPCA params
        params.n_components = self._n_components
        params.n_rows = n_rows
        params.n_cols = self.n_cols
        params.whiten = self.whiten

        input_data = CumlArray.zeros((params.n_rows, params.n_cols),
                                     dtype=dtype.type)

        cdef uintptr_t input_ptr = input_data.ptr
        cdef uintptr_t components_ptr = self.components_.ptr
        cdef uintptr_t singular_vals_ptr = self.singular_values_.ptr
        cdef uintptr_t _mean_ptr = self.mean_.ptr

        cdef handle_t* h_ = <handle_t*><size_t>self.handle.getHandle()
        if dtype.type == np.float32:
            pcaInverseTransform(h_[0],
                                <float*> _trans_input_ptr,
                                <float*> components_ptr,
                                <float*> singular_vals_ptr,
                                <float*> _mean_ptr,
                                <float*> input_ptr,
                                params)
        else:
            pcaInverseTransform(h_[0],
                                <double*> _trans_input_ptr,
                                <double*> components_ptr,
                                <double*> singular_vals_ptr,
                                <double*> _mean_ptr,
                                <double*> input_ptr,
                                params)

        # make sure the previously scheduled gpu tasks are complete before the
        # following transfers start
        self.handle.sync()

        return input_data

    @cuml.internals.api_base_return_array_skipall
    def _sparse_transform(self, X) -> CumlArray:

        # NOTE: All intermediate calculations are done using cupy.ndarray and
        # then converted to CumlArray at the end to minimize conversions
        # between types
        with using_output_type("cupy"):

            if self.whiten:
                self.components_ *= cp.sqrt(self.n_rows - 1)
                self.components_ /= self.singular_values_.reshape((-1, 1))

            X = X - self.mean_
            X_transformed = X.dot(self.components_.T)

            if self.whiten:
                self.components_ *= self.singular_values_.reshape((-1, 1))
                self.components_ *= (1 / cp.sqrt(self.n_rows - 1))

        return X_transformed

    @generate_docstring(X='dense_sparse',
                        return_values={'name': 'trans',
                                       'type': 'dense_sparse',
                                       'description': 'Transformed values',
                                       'shape': '(n_samples, n_components)'})
    def transform(self, X, convert_dtype=False) -> CumlArray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        """

        self._check_is_fitted('components_')
        if cupyx.scipy.sparse.issparse(X):
            return self._sparse_transform(X)
        elif scipy.sparse.issparse(X):
            X = sparse_scipy_to_cp(X, dtype=None)
            return self._sparse_transform(X)
        elif self._sparse_model:
            X, _, _, _ = \
                input_to_cupy_array(X, order='K',
                                    check_dtype=[cp.float32, cp.float64])
            return self._sparse_transform(X)

        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)

        cdef uintptr_t input_ptr = X_m.ptr

        # todo: check dtype
        cpdef paramsPCA params
        params.n_components = self._n_components
        params.n_rows = n_rows
        params.n_cols = n_cols
        params.whiten = self.whiten

        t_input_data = \
            CumlArray.zeros((params.n_rows, params.n_components),
                            dtype=dtype.type, index=X_m.index)

        cdef uintptr_t _trans_input_ptr = t_input_data.ptr
        cdef uintptr_t components_ptr = self.components_.ptr
        cdef uintptr_t singular_vals_ptr = \
            self.singular_values_.ptr
        cdef uintptr_t _mean_ptr = self.mean_.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        if dtype.type == np.float32:
            pcaTransform(handle_[0],
                         <float*> input_ptr,
                         <float*> components_ptr,
                         <float*> _trans_input_ptr,
                         <float*> singular_vals_ptr,
                         <float*> _mean_ptr,
                         params)
        else:
            pcaTransform(handle_[0],
                         <double*> input_ptr,
                         <double*> components_ptr,
                         <double*> _trans_input_ptr,
                         <double*> singular_vals_ptr,
                         <double*> _mean_ptr,
                         params)

        # make sure the previously scheduled gpu tasks are complete before the
        # following transfers start
        self.handle.sync()

        return t_input_data

    def get_param_names(self):
        return super().get_param_names() + \
            ["copy", "iterated_power", "n_components", "svd_solver", "tol",
                "whiten", "random_state"]

    def _check_is_fitted(self, attr):
        if not hasattr(self, attr) or (getattr(self, attr) is None):
            msg = ("This instance is not fitted yet. Call 'fit' "
                   "with appropriate arguments before using this estimator.")
            raise NotFittedError(msg)
