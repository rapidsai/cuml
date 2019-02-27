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

from numba import cuda

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libcpp cimport bool
from libc.stdint cimport uintptr_t

from cuml.decomposition.utils cimport *

cdef extern from "pca/pca_c.h" namespace "ML":

    cdef void pcaFit(float *input,
                     float *components,
                     float *explained_var,
                     float *explained_var_ratio,
                     float *singular_vals,
                     float *mu,
                     float *noise_vars,
                     paramsPCA prms)

    cdef void pcaFit(double *input,
                     double *components,
                     double *explained_var,
                     double *explained_var_ratio,
                     double *singular_vals,
                     double *mu,
                     double *noise_vars,
                     paramsPCA prms)

    cdef void pcaFitTransform(float *input,
                              float *trans_input,
                              float *components,
                              float *explained_var,
                              float *explained_var_ratio,
                              float *singular_vals,
                              float *mu,
                              float *noise_vars,
                              paramsPCA prms)

    cdef void pcaFitTransform(double *input,
                              double *trans_input,
                              double *components,
                              double *explained_var,
                              double *explained_var_ratio,
                              double *singular_vals,
                              double *mu,
                              double *noise_vars,
                              paramsPCA prms)

    cdef void pcaInverseTransform(float *trans_input,
                                  float *components,
                                  float *singular_vals,
                                  float *mu,
                                  float *input,
                                  paramsPCA prms)

    cdef void pcaInverseTransform(double *trans_input,
                                  double *components,
                                  double *singular_vals,
                                  double *mu,
                                  double *input,
                                  paramsPCA prms)

    cdef void pcaTransform(float *input,
                           float *components,
                           float *trans_input,
                           float *singular_vals,
                           float *mu,
                           paramsPCA prms)

    cdef void pcaTransform(double *input,
                           double *components,
                           double *trans_input,
                           double *singular_vals,
                           double *mu,
                           paramsPCA prms)

class PCAparams:
    def __init__(self, n_components, copy, whiten, tol, iterated_power,
                 random_state, svd_solver):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.n_cols = None
        self.n_rows = None


class PCA:
    """
    Create a DataFrame, fill it with data, and compute PCA:

    .. code-block:: python

        import cudf
        from cuml import PCA
        import numpy as np

        gdf_float = cudf.DataFrame()
        gdf_float['0']=np.asarray([1.0,2.0,5.0],dtype=np.float32)
        gdf_float['1']=np.asarray([4.0,2.0,1.0],dtype=np.float32)
        gdf_float['2']=np.asarray([4.0,2.0,1.0],dtype=np.float32)

        pca_float = PCA(n_components = 2)
        pca_float.fit(gdf_float)

        print(f'components: {pca_float.components_}')
        print(f'explained variance: {pca_float.explained_variance_}')
        print(f'explained variance ratio: {pca_float.explained_variance_ratio_}')

        print(f'singular values: {pca_float.singular_values_}')
        print(f'mean: {pca_float.mean_}')
        print(f'noise variance: {pca_float.noise_variance_}')

        trans_gdf_float = pca_float.transform(gdf_float)
        print(f'Inverse: {trans_gdf_float}')

        input_gdf_float = pca_float.inverse_transform(trans_gdf_float)
        print(f'Input: {input_gdf_float}')

    Output:

    .. code-block:: python

          components:
                      0           1           2
                      0  0.69225764  -0.5102837 -0.51028395
                      1 -0.72165036 -0.48949987  -0.4895003

          explained variance:

                      0   8.510402
                      1 0.48959687

          explained variance ratio:

                       0   0.9456003
                       1 0.054399658

          singular values:

                     0 4.1256275
                     1 0.9895422

          mean:

                    0 2.6666667
                    1 2.3333333
                    2 2.3333333

          noise variance:

                0  0.0

          transformed matrix:
                       0           1
                       0   -2.8547091 -0.42891636
                       1 -0.121316016  0.80743366
                       2    2.9760244 -0.37851727

          Input Matrix:
                    0         1         2
                    0 1.0000001 3.9999993       4.0
                    1       2.0 2.0000002 1.9999999
                    2 4.9999995 1.0000006       1.0


    For an additional example see `the PCA notebook <https://github.com/rapidsai/cuml/blob/master/python/notebooks/pca_demo.ipynb>`_. For additional docs, see `scikitlearn's PCA <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_.

    """

    def __init__(self, n_components=1, copy=True, whiten=False, tol=1e-7,
                 iterated_power=15, random_state=None, svd_solver='auto'):
        if svd_solver in ['full', 'auto', 'jacobi']:
            c_algorithm = self._get_algorithm_c_name(svd_solver)
        else:
            msg = "algorithm {!r} is not supported"
            raise TypeError(msg.format(svd_solver))
        self.params = PCAparams(n_components, copy, whiten, tol,
                                iterated_power, random_state, c_algorithm)
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.noise_variance_ = None
        self.components_ptr = None
        self.explained_variance_ptr = None
        self.explained_variance_ratio_ptr = None
        self.singular_values_ptr = None
        self.mean_ptr = None
        self.noise_variance_ptr = None

    def _get_algorithm_c_name(self, algorithm):
        return {
            'full': COV_EIG_DQ,
            'auto': COV_EIG_DQ,
            # 'arpack': NOT_SUPPORTED,
            # 'randomized': NOT_SUPPORTED,
            'jacobi': COV_EIG_JACOBI
        }[algorithm]

    def _initialize_arrays(self, n_components, n_rows, n_cols):

        self.trans_input_ = cuda.to_device(np.zeros(n_rows*n_components,
                                                    dtype=self.gdf_datatype))
        self.components_ = cuda.to_device(np.zeros(n_components*n_cols,
                                                   dtype=self.gdf_datatype))
        self.explained_variance_ = cudf.Series(
                                      np.zeros(n_components,
                                               dtype=self.gdf_datatype))
        self.explained_variance_ratio_ = cudf.Series(
                                            np.zeros(n_components,
                                                     dtype=self.gdf_datatype))
        self.mean_ = cudf.Series(np.zeros(n_cols, dtype=self.gdf_datatype))
        self.singular_values_ = cudf.Series(np.zeros(n_components,
                                                     dtype=self.gdf_datatype))
        self.noise_variance_ = cudf.Series(np.zeros(1,
                                                    dtype=self.gdf_datatype))

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def fit(self, X, _transform=False):
        """
        Fit the model with X.

        Parameters
        ----------
        X : cuDF DataFrame
          Dense matrix (floats or doubles) of shape (n_samples, n_features)

        Returns
        -------
        cluster labels

        """
        # c params

        cdef uintptr_t input_ptr
        if (isinstance(X, cudf.DataFrame)):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            # PCA expects transpose of the input
            X_m = X.as_gpu_matrix()
            self.params.n_rows = len(X)
            self.params.n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            self.gdf_datatype = X.dtype
            X_m = cuda.to_device(np.array(X, order='F'))
            self.params.n_rows = X.shape[0]
            self.params.n_cols = X.shape[1]

        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        input_ptr = self._get_ctype_ptr(X_m)

        cpdef paramsPCA params
        params.n_components = self.params.n_components
        params.n_rows = self.params.n_rows
        params.n_cols = self.params.n_cols
        params.whiten = self.params.whiten
        params.n_iterations = self.params.iterated_power
        params.tol = self.params.tol
        params.algorithm = self.params.svd_solver

        self._initialize_arrays(params.n_components,
                                params.n_rows, params.n_cols)

        cdef uintptr_t components_ptr = self._get_ctype_ptr(self.components_)

        cdef uintptr_t explained_var_ptr = self._get_column_ptr(
                                                self.explained_variance_)
        cdef uintptr_t explained_var_ratio_ptr = self._get_column_ptr(
                                                self.explained_variance_ratio_)
        cdef uintptr_t singular_vals_ptr = self._get_column_ptr(
                                                self.singular_values_)
        cdef uintptr_t mean_ptr = self._get_column_ptr(self.mean_)
        cdef uintptr_t noise_vars_ptr = self._get_column_ptr(
                                            self.noise_variance_)
        cdef uintptr_t trans_input_ptr = self._get_ctype_ptr(self.trans_input_)

        if not _transform:
            if self.gdf_datatype.type == np.float32:
                pcaFit(<float*> input_ptr,
                             <float*> components_ptr,
                             <float*> explained_var_ptr,
                             <float*> explained_var_ratio_ptr,
                             <float*> singular_vals_ptr,
                             <float*> mean_ptr,
                             <float*> noise_vars_ptr,
                             params)
            else:
                pcaFit(<double*> input_ptr,
                             <double*> components_ptr,
                             <double*> explained_var_ptr,
                             <double*> explained_var_ratio_ptr,
                             <double*> singular_vals_ptr,
                             <double*> mean_ptr,
                             <double*> noise_vars_ptr,
                             params)
        else:

            if self.gdf_datatype.type == np.float32:
                pcaFitTransform(<float*> input_ptr,
                                      <float*> trans_input_ptr,
                                      <float*> components_ptr,
                                      <float*> explained_var_ptr,
                                      <float*> explained_var_ratio_ptr,
                                      <float*> singular_vals_ptr,
                                      <float*> mean_ptr,
                                      <float*> noise_vars_ptr,
                                      params)
            else:
                pcaFitTransform(<double*> input_ptr,
                                      <double*> trans_input_ptr,
                                      <double*> components_ptr,
                                      <double*> explained_var_ptr,
                                      <double*> explained_var_ratio_ptr,
                                      <double*> singular_vals_ptr,
                                      <double*> mean_ptr,
                                      <double*> noise_vars_ptr,
                                      params)

        components_gdf = cudf.DataFrame()
        for i in range(0, params.n_cols):
            components_gdf[str(i)] = self.components_[i*params.n_components:(i+1)*params.n_components]

        self.components_ = components_gdf
        self.components_ptr = components_ptr
        self.explained_variance_ptr = explained_var_ptr
        self.explained_variance_ratio_ptr = explained_var_ratio_ptr
        self.singular_values_ptr = singular_vals_ptr
        self.mean_ptr = mean_ptr
        self.noise_variance_ptr = noise_vars_ptr

        if (isinstance(X, cudf.DataFrame)):
            del(X_m)

        return self

    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : cuDF DataFrame, shape (n_samples, n_features)
          training data (floats or doubles), where n_samples is the number of samples, and n_features is the number of features.

        Returns
        -------
        X_new : cuDF DataFrame, shape (n_samples, n_components)
        """
        self.fit(X, _transform=True)
        X_new = cudf.DataFrame()
        num_rows = self.params.n_rows

        for i in range(0, self.params.n_components):
            X_new[str(i)] = self.trans_input_[i*num_rows:(i+1)*num_rows]

        return X_new

    def inverse_transform(self, X):
        """
        Transform data back to its original space.

        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        X : cuDF DataFrame, shape (n_samples, n_components)
            New data (floats or doubles), where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        X_original : cuDF DataFrame, shape (n_samples, n_features)

        """
        cdef uintptr_t trans_input_ptr
        if (isinstance(X, cudf.DataFrame)):
            gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix()
        elif (isinstance(X, np.ndarray)):
            gdf_datatype = X.dtype
            X_m = cuda.to_device(np.array(X, order='F'))
        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        trans_input_ptr = self._get_ctype_ptr(X_m)

        cpdef paramsPCA params
        params.n_components = self.params.n_components
        params.n_rows = len(X)
        params.n_cols = self.params.n_cols
        params.whiten = self.params.whiten

        input_data = cuda.to_device(np.zeros(params.n_rows*params.n_cols,
                                             dtype=gdf_datatype.type))

        cdef uintptr_t input_ptr = input_data.device_ctypes_pointer.value

        cdef uintptr_t components_ptr = self.components_ptr
        cdef uintptr_t singular_vals_ptr = self.singular_values_ptr
        cdef uintptr_t mean_ptr = self.mean_ptr

        if gdf_datatype.type == np.float32:
            pcaInverseTransform(<float*> trans_input_ptr,
                                      <float*> components_ptr,
                                      <float*> singular_vals_ptr,
                                      <float*> mean_ptr,
                                      <float*> input_ptr,
                                      params)
        else:
            pcaInverseTransform(<double*> trans_input_ptr,
                                      <double*> components_ptr,
                                      <double*> singular_vals_ptr,
                                      <double*> mean_ptr,
                                      <double*> input_ptr,
                                      params)

        X_original = cudf.DataFrame()
        for i in range(0, params.n_cols):
            X_original[str(i)] = input_data[i*params.n_rows:(i+1)*params.n_rows]

        del(X_m)

        return X_original

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted from a training set.

        Parameters
        ----------
        X : cuDF DataFrame, shape (n_samples, n_features)
            New data (floats or doubles), where n_samples is the number of samples and n_features is the number of features.

        Returns
        -------
        X_new : cuDF DataFrame, shape (n_samples, n_components)

        """


        cdef uintptr_t input_ptr
        if (isinstance(X, cudf.DataFrame)):
            gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix()
            n_rows = len(X)
            n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            gdf_datatype = X.dtype
            X_m = cuda.to_device(np.array(X, order='F'))
            n_rows = X.shape[0]
            n_cols = X.shape[1]

        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        input_ptr = self._get_ctype_ptr(X_m)

        cpdef paramsPCA params
        params.n_components = self.params.n_components
        params.n_rows = n_rows
        params.n_cols = n_cols
        params.whiten = self.params.whiten

        trans_input_data = cuda.to_device(
                              np.zeros(params.n_rows*params.n_components,
                                       dtype=gdf_datatype.type))

        cdef uintptr_t trans_input_ptr = self._get_ctype_ptr(trans_input_data)
        cdef uintptr_t components_ptr = self.components_ptr
        cdef uintptr_t singular_vals_ptr = self.singular_values_ptr
        cdef uintptr_t mean_ptr = self.mean_ptr

        if gdf_datatype.type == np.float32:
            pcaTransform(<float*> input_ptr,
                               <float*> components_ptr,
                               <float*> trans_input_ptr,
                               <float*> singular_vals_ptr,
                               <float*> mean_ptr,
                               params)
        else:
            pcaTransform(<double*> input_ptr,
                               <double*> components_ptr,
                               <double*> trans_input_ptr,
                               <double*> singular_vals_ptr,
                               <double*> mean_ptr,
                               params)

        X_new = cudf.DataFrame()
        for i in range(0, params.n_components):
            X_new[str(i)] = trans_input_data[i*params.n_rows:(i+1)*params.n_rows]

        del(X_m)
        return X_new
