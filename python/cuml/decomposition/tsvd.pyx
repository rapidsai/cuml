#
# Copyright (c) 2019, NVIDIA CORPORATION.
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

import ctypes
import cudf
import numpy as np

import rmm
from libcpp cimport bool
from libc.stdint cimport uintptr_t

import cuml

from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.decomposition.utils cimport *
from cuml.utils import input_to_cuml_array

from cython.operator cimport dereference as deref

cdef extern from "cuml/decomposition/tsvd.hpp" namespace "ML":

    cdef void tsvdFit(cumlHandle& handle,
                      float *input,
                      float *components,
                      float *singular_vals,
                      const paramsTSVD &prms) except +

    cdef void tsvdFit(cumlHandle& handle,
                      double *input,
                      double *components,
                      double *singular_vals,
                      const paramsTSVD &prms) except +

    cdef void tsvdFitTransform(cumlHandle& handle,
                               float *input,
                               float *trans_input,
                               float *components,
                               float *explained_var,
                               float *explained_var_ratio,
                               float *singular_vals,
                               const paramsTSVD &prms) except +

    cdef void tsvdFitTransform(cumlHandle& handle,
                               double *input,
                               double *trans_input,
                               double *components,
                               double *explained_var,
                               double *explained_var_ratio,
                               double *singular_vals,
                               const paramsTSVD &prms) except +

    cdef void tsvdInverseTransform(cumlHandle& handle,
                                   float *trans_input,
                                   float *components,
                                   float *input,
                                   const paramsTSVD &prms) except +

    cdef void tsvdInverseTransform(cumlHandle& handle,
                                   double *trans_input,
                                   double *components,
                                   double *input,
                                   const paramsTSVD &prms) except +

    cdef void tsvdTransform(cumlHandle& handle,
                            float *input,
                            float *components,
                            float *trans_input,
                            const paramsTSVD &prms) except +

    cdef void tsvdTransform(cumlHandle& handle,
                            double *input,
                            double *components,
                            double *trans_input,
                            const paramsTSVD &prms) except +


class TruncatedSVD(Base):
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
    ---------

    .. code-block:: python

        # Both import methods supported
        from cuml import TruncatedSVD
        from cuml.decomposition import TruncatedSVD

        import cudf
        import numpy as np

        gdf_float = cudf.DataFrame()
        gdf_float['0'] = np.asarray([1.0,2.0,5.0], dtype = np.float32)
        gdf_float['1'] = np.asarray([4.0,2.0,1.0], dtype = np.float32)
        gdf_float['2'] = np.asarray([4.0,2.0,1.0], dtype = np.float32)

        tsvd_float = TruncatedSVD(n_components = 2, algorithm = "jacobi",
                                  n_iter = 20, tol = 1e-9)
        tsvd_float.fit(gdf_float)

        print(f'components: {tsvd_float.components_}')
        print(f'explained variance: {tsvd_float._explained_variance_}')
        exp_var = tsvd_float._explained_variance_ratio_
        print(f'explained variance ratio: {exp_var}')
        print(f'singular values: {tsvd_float._singular_values_}')

        trans_gdf_float = tsvd_float.transform(gdf_float)
        print(f'Transformed matrix: {trans_gdf_float}')

        input_gdf_float = tsvd_float.inverse_transform(trans_gdf_float)
        print(f'Input matrix: {input_gdf_float}')

    Output:

    .. code-block:: python

        components:            0           1          2
        0 0.58725953  0.57233137  0.5723314
        1 0.80939883 -0.41525528 -0.4152552
        explained variance:
        0  55.33908
        1 16.660923

        explained variance ratio:
        0  0.7685983
        1 0.23140171

        singular values:
        0  7.439024
        1 4.0817795

        Transformed Matrix:
        0           1         2
        0   5.1659107    -2.512643
        1   3.4638448    -0.042223275
        2    4.0809603   3.2164836

        Input matrix:           0         1         2
        0       1.0  4.000001  4.000001
        1 2.0000005 2.0000005 2.0000007
        2  5.000001 0.9999999 1.0000004

    Parameters
    -----------
    algorithm : 'full' or 'jacobi' or 'auto' (default = 'full')
        Full uses a eigendecomposition of the covariance matrix then discards
        components.
        Jacobi is much faster as it iteratively corrects, but is less accurate.
    handle : cuml.Handle
        If it is None, a new one is created just for this class
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
    verbose : bool
        Whether to print debug spews

    Attributes
    -----------
    components_ : array
        The top K components (VT.T[:,:n_components]) in U, S, VT = svd(X)
    explained_variance_ : array
        How much each component explains the variance in the data given by S**2
    explained_variance_ratio_ : array
        How much in % the variance is explained given by S**2/sum(S**2)
    singular_values_ : array
        The top K singular values. Remember all singular values >= 0

    Notes
    ------
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

    For additional examples, see `the Truncated SVD  notebook
    <https://github.com/rapidsai/notebooks/blob/master/cuml/tsvd_demo.ipynb>`_.
    For additional documentation, see `scikitlearn's TruncatedSVD docs
    <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`_.
    """

    def __init__(self, algorithm='full', handle=None, n_components=1,
                 n_iter=15, random_state=None, tol=1e-7, verbose=False,
                 output_type=None):
        # params
        super(TruncatedSVD, self).__init__(handle=handle, verbose=verbose,
                                           output_type=output_type)
        self.algorithm = algorithm
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol
        self.c_algorithm = self._get_algorithm_c_name(self.algorithm)

        # internal array attributes
        self._components_ = None  # accessed via estimator.components_
        self._explained_variance_ = None
        # accessed via estimator.explained_variance_

        self._explained_variance_ratio_ = None
        # accessed via estimator.explained_variance_ratio_

        self._singular_values_ = None
        # accessed via estimator.singular_values_

    def _get_algorithm_c_name(self, algorithm):
        algo_map = {
            'full': COV_EIG_DQ,
            'auto': COV_EIG_DQ,
            'jacobi': COV_EIG_JACOBI
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
        params.algorithm = self.c_algorithm

        return <size_t>params

    def _initialize_arrays(self, n_components, n_rows, n_cols):

        self._components_ = CumlArray.zeros((n_components, n_cols),
                                            dtype=self.dtype)
        self._explained_variance_ = CumlArray.zeros(n_components,
                                                    dtype=self.dtype)
        self._explained_variance_ratio_ = CumlArray.zeros(n_components,
                                                          dtype=self.dtype)
        self._mean_ = CumlArray.zeros(n_cols, dtype=self.dtype)
        self._singular_values_ = CumlArray.zeros(n_components,
                                                 dtype=self.dtype)
        self._noise_variance_ = CumlArray.zeros(1, dtype=self.dtype)

    def fit(self, X, y=None):
        """
        Fit LSI model on training cudf DataFrame X.

        Parameters
        ----------
       X : array-like (device or host) shape = (n_samples, n_features)
           Dense matrix (floats or doubles) of shape (n_samples, n_features).
           Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
           ndarray, cuda array interface compliant array like CuPy

        y : ignored

        """

        self.fit_transform(X)

        return self

    def fit_transform(self, X, y=None):
        """
        Fit LSI model to X and perform dimensionality reduction on X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        y : ignored

        Returns
        -------
        X_new : cuDF DataFrame, shape (n_samples, n_components)
            Reduced version of X as a dense cuDF DataFrame
        """
        self._set_output_type(X)

        X_m, self.n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64])
        cdef uintptr_t input_ptr = X_m.ptr

        cdef paramsTSVD *params = <paramsTSVD*><size_t> \
            self._build_params(self.n_rows, self.n_cols)

        self._initialize_arrays(self.n_components, self.n_rows, self.n_cols)

        cdef uintptr_t comp_ptr = self._components_.ptr

        cdef uintptr_t explained_var_ptr = \
            self._explained_variance_.ptr

        cdef uintptr_t explained_var_ratio_ptr = \
            self._explained_variance_ratio_.ptr

        cdef uintptr_t singular_vals_ptr = \
            self._singular_values_.ptr

        _trans_input_ = CumlArray.zeros((params.n_rows, params.n_components),
                                        dtype=self.dtype)
        cdef uintptr_t t_input_ptr = _trans_input_.ptr

        if self.n_components> self.n_cols:
            raise ValueError(' n_components must be < n_features')

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
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

        out_type = self._get_output_type(X)
        return _trans_input_.to_output(out_type)

    def inverse_transform(self, X, convert_dtype=False):
        """
        Transform X back to its original space.
        Returns a cuDF DataFrame X_original whose transform would be X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
           Dense matrix (floats or doubles) of shape (n_samples, n_features).
           Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
           ndarray, cuda array interface compliant array like CuPy
        convert_dtype : bool, optional (default = False)
            When set to True, the inverse_transform method will automatically
            convert the input to the data type which was used to train the
            model. This will increase memory used for the method.

        Returns
        -------
        X_original : cuDF DataFrame, shape (n_samples, n_features)
            Note that this is always a dense cuDF DataFrame.
        """

        trans_input, n_rows, _, dtype = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None))

        cpdef paramsTSVD params
        params.n_components = self.n_components
        params.n_rows = n_rows
        params.n_cols = self.n_cols

        input_data = CumlArray.zeros((params.n_rows, params.n_cols),
                                     dtype=self.dtype)

        cdef uintptr_t trans_input_ptr = trans_input.ptr
        cdef uintptr_t input_ptr = input_data.ptr
        cdef uintptr_t components_ptr = self._components_.ptr

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

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

        out_type = self._get_output_type(X)
        return input_data.to_output(out_type)

    def transform(self, X, convert_dtype=False):
        """
        Perform dimensionality reduction on X.
        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        convert_dtype : bool, optional (default = False)
            When set to True, the transform method will automatically
            convert the input to the data type which was used to train the
            model.

        Returns
        -------
        X_new : cuDF DataFrame, shape (n_samples, n_components)
            Reduced version of X. This will always be a dense DataFrame.
        """
        input, n_rows, _, dtype = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)

        cpdef paramsTSVD params
        params.n_components = self.n_components
        params.n_rows = n_rows
        params.n_cols = self.n_cols

        t_input_data = \
            CumlArray.zeros((params.n_rows, params.n_components),
                            dtype=self.dtype)

        cdef uintptr_t input_ptr = input.ptr
        cdef uintptr_t trans_input_ptr = t_input_data.ptr
        cdef uintptr_t components_ptr = self._components_.ptr

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

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

        out_type = self._get_output_type(X)
        return t_input_data.to_output(out_type)

    def get_param_names(self):
        return ["algorithm", "n_components", "n_iter", "random_state", "tol"]
