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

from enum import IntEnum

import rmm
from libcpp cimport bool
from libc.stdint cimport uintptr_t


from cuml.internals.array import CumlArray
from cuml.internals.base import UniversalBase
from pylibraft.common.handle cimport handle_t
from cuml.decomposition.utils cimport *
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.mixins import FMajorInputTagMixin
from cuml.internals.api_decorators import device_interop_preparation
from cuml.internals.api_decorators import enable_device_interop

from cython.operator cimport dereference as deref


cdef extern from "cuml/decomposition/tsvd.hpp" namespace "ML":

    cdef void tsvdFit(handle_t& handle,
                      float *input,
                      float *components,
                      float *singular_vals,
                      const paramsTSVD &prms) except +

    cdef void tsvdFit(handle_t& handle,
                      double *input,
                      double *components,
                      double *singular_vals,
                      const paramsTSVD &prms) except +

    cdef void tsvdFitTransform(handle_t& handle,
                               float *input,
                               float *trans_input,
                               float *components,
                               float *explained_var,
                               float *explained_var_ratio,
                               float *singular_vals,
                               const paramsTSVD &prms) except +

    cdef void tsvdFitTransform(handle_t& handle,
                               double *input,
                               double *trans_input,
                               double *components,
                               double *explained_var,
                               double *explained_var_ratio,
                               double *singular_vals,
                               const paramsTSVD &prms) except +

    cdef void tsvdInverseTransform(handle_t& handle,
                                   float *trans_input,
                                   float *components,
                                   float *input,
                                   const paramsTSVD &prms) except +

    cdef void tsvdInverseTransform(handle_t& handle,
                                   double *trans_input,
                                   double *components,
                                   double *input,
                                   const paramsTSVD &prms) except +

    cdef void tsvdTransform(handle_t& handle,
                            float *input,
                            float *components,
                            float *trans_input,
                            const paramsTSVD &prms) except +

    cdef void tsvdTransform(handle_t& handle,
                            double *input,
                            double *components,
                            double *trans_input,
                            const paramsTSVD &prms) except +


class Solver(IntEnum):
    COV_EIG_DQ = <underlying_type_t_solver> solver.COV_EIG_DQ
    COV_EIG_JACOBI = <underlying_type_t_solver> solver.COV_EIG_JACOBI


class TruncatedSVD(UniversalBase,
                   FMajorInputTagMixin):
    """
    TruncatedSVD is used to compute the top K singular values and vectors of a
    large matrix X. It is much faster when n_components is small, such as in
    the use of PCA when 3 components is used for 3D visualization.

    cuML's TruncatedSVD an array-like object or cuDF DataFrame, and provides 2
    algorithms Full and Jacobi. Full (default) uses a full eigendecomposition
    then selects the top K singular vectors. The Jacobi algorithm is much
    faster as it iteratively tries to correct the top K singular vectors, but
    might be less accurate.

    Examples
    --------

    .. code-block:: python

        >>> # Both import methods supported
        >>> from cuml import TruncatedSVD
        >>> from cuml.decomposition import TruncatedSVD

        >>> import cudf
        >>> import cupy as cp

        >>> gdf_float = cudf.DataFrame()
        >>> gdf_float['0'] = cp.asarray([1.0,2.0,5.0], dtype=cp.float32)
        >>> gdf_float['1'] = cp.asarray([4.0,2.0,1.0], dtype=cp.float32)
        >>> gdf_float['2'] = cp.asarray([4.0,2.0,1.0], dtype=cp.float32)

        >>> tsvd_float = TruncatedSVD(n_components = 2, algorithm = "jacobi",
        ...                           n_iter = 20, tol = 1e-9)
        >>> tsvd_float.fit(gdf_float)
        TruncatedSVD()
        >>> print(f'components: {tsvd_float.components_}') # doctest: +SKIP
        components:           0         1         2
        0  0.587259  0.572331  0.572331
        1  0.809399 -0.415255 -0.415255
        >>> exp_var = tsvd_float.explained_variance_
        >>> print(f'explained variance: {exp_var}')
        explained variance: 0    0.494...
        1    5.505...
        dtype: float32
        >>> exp_var_ratio = tsvd_float.explained_variance_ratio_
        >>> print(f'explained variance ratio: {exp_var_ratio}')
        explained variance ratio: 0    0.082...
        1    0.917...
        dtype: float32
        >>> sing_values = tsvd_float.singular_values_
        >>> print(f'singular values: {sing_values}')
        singular values: 0    7.439...
        1    4.081...
        dtype: float32

        >>> trans_gdf_float = tsvd_float.transform(gdf_float)
        >>> print(f'Transformed matrix: {trans_gdf_float}') # doctest: +SKIP
        Transformed matrix:           0         1
        0  5.165910 -2.512643
        1  3.463844 -0.042223
        2  4.080960  3.216484
        >>> input_gdf_float = tsvd_float.inverse_transform(trans_gdf_float)
        >>> print(f'Input matrix: {input_gdf_float}')
        Input matrix:      0    1    2
        0  1.0  4.0  4.0
        1  2.0  2.0  2.0
        2  5.0  1.0  1.0

    Parameters
    ----------
    algorithm : 'full' or 'jacobi' or 'auto' (default = 'full')
        Full uses a eigendecomposition of the covariance matrix then discards
        components.
        Jacobi is much faster as it iteratively corrects, but is less accurate.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    n_components : int (default = 1)
        The number of top K singular vectors / values you want.
        Must be <= number(columns).
    n_iter : int (default = 15)
        Used in Jacobi solver. The more iterations, the more accurate, but
        slower.
    random_state : int / None (default = None)
        If you want results to be the same when you restart Python, select a
        state.
    tol : float (default = 1e-7)
        Used if algorithm = "jacobi". Smaller tolerance can increase accuracy,
        but but will slow down the algorithm's convergence.
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
    components_ : array
        The top K components (VT.T[:,:n_components]) in U, S, VT = svd(X)
    explained_variance_ : array
        How much each component explains the variance in the data given by S**2
    explained_variance_ratio_ : array
        How much in % the variance is explained given by S**2/sum(S**2)
    singular_values_ : array
        The top K singular values. Remember all singular values >= 0

    Notes
    -----
    TruncatedSVD (the randomized version [Jacobi]) is fantastic when the number
    of components you want is much smaller than the number of features. The
    approximation to the largest singular values and vectors is very robust,
    however, this method loses a lot of accuracy when you want many, many
    components.

    **Applications of TruncatedSVD**

    TruncatedSVD is also known as Latent Semantic Indexing (LSI) which
    tries to find topics of a word count matrix. If X previously was
    centered with mean removal, TruncatedSVD is the same as TruncatedPCA.
    TruncatedSVD is also used in information retrieval tasks,
    recommendation systems and data compression.

    For additional documentation, see `scikitlearn's TruncatedSVD docs
    <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`_.

    """

    _cpu_estimator_import_path = 'sklearn.decomposition.TruncatedSVD'
    components_ = CumlArrayDescriptor(order='F')
    explained_variance_ = CumlArrayDescriptor(order='F')
    explained_variance_ratio_ = CumlArrayDescriptor(order='F')
    singular_values_ = CumlArrayDescriptor(order='F')

    @device_interop_preparation
    def __init__(self, *, algorithm='full', handle=None, n_components=1,
                 n_iter=15, random_state=None, tol=1e-7,
                 verbose=False, output_type=None):
        # params
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)
        self.algorithm = algorithm
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol
        self.c_algorithm = self._get_algorithm_c_name(self.algorithm)

        # internal array attributes
        self.components_ = None
        self.explained_variance_ = None

        self.explained_variance_ratio_ = None

        self.singular_values_ = None

    def _get_algorithm_c_name(self, algorithm):
        algo_map = {
            'full': Solver.COV_EIG_DQ,
            'auto': Solver.COV_EIG_DQ,
            'jacobi': Solver.COV_EIG_JACOBI
        }
        if algorithm not in algo_map:
            msg = "algorithm {!r} is not supported"
            raise TypeError(msg.format(algorithm))
        return algo_map[algorithm]

    def _build_params(self, n_rows, n_cols):
        cpdef paramsTSVD *params = new paramsTSVD()
        params.n_components = self.n_components
        params.n_rows = n_rows
        params.n_cols = n_cols
        params.n_iterations = self.n_iter
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
        self.singular_values_ = CumlArray.zeros(n_components,
                                                dtype=self.dtype)

    @generate_docstring()
    @enable_device_interop
    def fit(self, X, y=None) -> "TruncatedSVD":
        """
        Fit LSI model on training cudf DataFrame X. y is currently ignored.

        """

        self.fit_transform(X)

        return self

    @generate_docstring(return_values={'name': 'trans',
                                       'type': 'dense',
                                       'description': 'Reduced version of X',
                                       'shape': '(n_samples, n_components)'})
    @enable_device_interop
    def fit_transform(self, X, y=None) -> CumlArray:
        """
        Fit LSI model to X and perform dimensionality reduction on X.
        y is currently ignored.

        """
        X_m, self.n_rows, self.n_features_in_, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64])
        cdef uintptr_t input_ptr = X_m.ptr

        cdef paramsTSVD *params = <paramsTSVD*><size_t> \
            self._build_params(self.n_rows, self.n_features_in_)

        self._initialize_arrays(self.n_components, self.n_rows,
                                self.n_features_in_)

        cdef uintptr_t comp_ptr = self.components_.ptr

        cdef uintptr_t explained_var_ptr = \
            self.explained_variance_.ptr

        cdef uintptr_t explained_var_ratio_ptr = \
            self.explained_variance_ratio_.ptr

        cdef uintptr_t singular_vals_ptr = \
            self.singular_values_.ptr

        _trans_input_ = CumlArray.zeros((params.n_rows, params.n_components),
                                        dtype=self.dtype, index=X_m.index)
        cdef uintptr_t t_input_ptr = _trans_input_.ptr

        if self.n_components> self.n_features_in_:
            raise ValueError(' n_components must be < n_features')

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        if self.dtype == np.float32:
            tsvdFitTransform(handle_[0],
                             <float*> input_ptr,
                             <float*> t_input_ptr,
                             <float*> comp_ptr,
                             <float*> explained_var_ptr,
                             <float*> explained_var_ratio_ptr,
                             <float*> singular_vals_ptr,
                             deref(params))
        else:
            tsvdFitTransform(handle_[0],
                             <double*> input_ptr,
                             <double*> t_input_ptr,
                             <double*> comp_ptr,
                             <double*> explained_var_ptr,
                             <double*> explained_var_ratio_ptr,
                             <double*> singular_vals_ptr,
                             deref(params))

        # make sure the previously scheduled gpu tasks are complete before the
        # following transfers start
        self.handle.sync()

        return _trans_input_

    @generate_docstring(return_values={'name': 'X_original',
                                       'type': 'dense',
                                       'description': 'X in original space',
                                       'shape': '(n_samples, n_features)'})
    @enable_device_interop
    def inverse_transform(self, X, convert_dtype=False) -> CumlArray:
        """
        Transform X back to its original space.
        Returns X_original whose transform would be X.

        """
        dtype = self.components_.dtype
        X_m, n_rows, _, dtype = \
            input_to_cuml_array(X, check_dtype=dtype,
                                convert_to_dtype=(dtype if convert_dtype
                                                  else None))

        cpdef paramsTSVD params
        params.n_components = self.n_components
        params.n_rows = n_rows
        params.n_cols = self.n_features_in_

        input_data = CumlArray.zeros((params.n_rows, params.n_cols),
                                     dtype=dtype, index=X_m.index)

        cdef uintptr_t trans_input_ptr = X_m.ptr
        cdef uintptr_t input_ptr = input_data.ptr
        cdef uintptr_t components_ptr = self.components_.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if dtype.type == np.float32:
            tsvdInverseTransform(handle_[0],
                                 <float*> trans_input_ptr,
                                 <float*> components_ptr,
                                 <float*> input_ptr,
                                 params)
        else:
            tsvdInverseTransform(handle_[0],
                                 <double*> trans_input_ptr,
                                 <double*> components_ptr,
                                 <double*> input_ptr,
                                 params)

        # make sure the previously scheduled gpu tasks are complete before the
        # following transfers start
        self.handle.sync()

        return input_data

    @generate_docstring(return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Reduced version of X',
                                       'shape': '(n_samples, n_components)'})
    @enable_device_interop
    def transform(self, X, convert_dtype=False) -> CumlArray:
        """
        Perform dimensionality reduction on X.

        """
        dtype = self.components_.dtype
        self.n_features_in_ = self.components_.shape[1]

        X_m, n_rows, _, dtype = \
            input_to_cuml_array(X, check_dtype=dtype,
                                convert_to_dtype=(dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_features_in_)

        cpdef paramsTSVD params
        params.n_components = self.n_components
        params.n_rows = n_rows
        params.n_cols = self.n_features_in_

        t_input_data = \
            CumlArray.zeros((params.n_rows, params.n_components),
                            dtype=dtype, index=X_m.index)

        cdef uintptr_t input_ptr = X_m.ptr
        cdef uintptr_t trans_input_ptr = t_input_data.ptr
        cdef uintptr_t components_ptr = self.components_.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if dtype.type == np.float32:
            tsvdTransform(handle_[0],
                          <float*> input_ptr,
                          <float*> components_ptr,
                          <float*> trans_input_ptr,
                          params)
        else:
            tsvdTransform(handle_[0],
                          <double*> input_ptr,
                          <double*> components_ptr,
                          <double*> trans_input_ptr,
                          params)

        # make sure the previously scheduled gpu tasks are complete before the
        # following transfers start
        self.handle.sync()

        return t_input_data

    def get_param_names(self):
        return super().get_param_names() + \
            ["algorithm", "n_components", "n_iter", "random_state", "tol"]

    def get_attr_names(self):
        return ['components_', 'explained_variance_',
                'explained_variance_ratio_', 'singular_values_',
                'n_features_in_', 'feature_names_in_']
