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

from libcpp cimport bool
from libc.stdint cimport uintptr_t

from cuml.decomposition.utils cimport *

cdef extern from "tsvd/tsvd_spmg.h" namespace "ML":

    cdef void tsvdFitSPMG(float *h_input,
                          float *h_components,
                          float *h_singular_vals,
                  paramsTSVD prms,
                          int *gpu_ids,
                          int n_gpus)

    cdef void tsvdFitSPMG(double *h_input,
                          double *h_components,
                          double *h_singular_vals,
                  paramsTSVD prms,
                          int *gpu_ids,
                          int n_gpus)

    cdef void tsvdFitTransformSPMG(float *h_input,
                                   float *h_trans_input,
                           float *h_components,
                                   float *h_explained_var,
                           float *h_explained_var_ratio,
                                   float *h_singular_vals,
                                   paramsTSVD prms,
                           int *gpu_ids,
                                   int n_gpus)

    cdef void tsvdFitTransformSPMG(double *h_input,
                                   double *h_trans_input,
                           double *h_components,
                                   double *h_explained_var,
                           double *h_explained_var_ratio,
                                   double *h_singular_vals,
                                   paramsTSVD prms,
                           int *gpu_ids,
                                   int n_gpus)

    cdef void tsvdInverseTransformSPMG(float *h_trans_input,
                                       float *h_components,
                                       bool trans_comp,
                                       float *input,
                                       paramsTSVD prms,
                                       int *gpu_ids,
                                       int n_gpus)

    cdef void tsvdInverseTransformSPMG(double *h_trans_input,
                                       double *h_components,
                                       bool trans_comp,
                                       double *input,
                                       paramsTSVD prms,
                                       int *gpu_ids,
                                       int n_gpus)

    cdef void tsvdTransformSPMG(float *h_input,
                                float *h_components,
                                bool trans_comp,
                                float *h_trans_input,
                                paramsTSVD prms,
                                int *gpu_ids,
                                int n_gpus)

    cdef void tsvdTransformSPMG(double *h_input,
                                double *h_components,
                                bool trans_comp,
                                double *h_trans_input,
                                paramsTSVD prms,
                                int *gpu_ids,
                                int n_gpus)

class TSVDparams:
    def __init__(self,n_components,tol,iterated_power,random_state,svd_solver):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.n_cols = None
        self.n_rows = None

class TruncatedSVDSPMG:
    """
    Create a DataFrame, fill it with data, and compute Truncated Singular Value
    Decomposition:

    .. code-block:: python

            from cuml import TruncatedSVD
            import cudf
            import numpy as np

            gdf_float = cudf.DataFrame()
            gdf_float['0']=np.asarray([1.0,2.0,5.0],dtype=np.float32)
            gdf_float['1']=np.asarray([4.0,2.0,1.0],dtype=np.float32)
            gdf_float['2']=np.asarray([4.0,2.0,1.0],dtype=np.float32)

            tsvd_float = TruncatedSVD(n_components = 2, algorithm="jacobi",
                                      n_iter=20, tol=1e-9)
            tsvd_float.fit(gdf_float)

            print(f'components: {tsvd_float.components_}')
            print(f'explained variance: {tsvd_float.explained_variance_}')
            exp_var = tsvd_float.explained_variance_ratio_
            print(f'explained variance ratio: {exp_var}')
            print(f'singular values: {tsvd_float.singular_values_}')

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

            Transformed matrix:           0            1
            0 5.1659107    -2.512643
            1 3.4638448 -0.042223275
            2 4.0809603    3.2164836

            Input matrix:           0         1         2
            0       1.0  4.000001  4.000001
            1 2.0000005 2.0000005 2.0000007
            2  5.000001 0.9999999 1.0000004

    For additional examples, see the Truncated SVD  notebook
    <https://github.com/rapidsai/cuml/blob/master/python/notebooks/tsvd_demo.ipynb>`_.
    For additional documentation, see `scikitlearn's TruncatedSVD docs
    <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`_.

    """

    def __init__(self, n_components=1, tol=1e-7, n_iter=15, random_state=None,
                 algorithm='full'):
        if algorithm in ['full', 'auto', 'jacobi']:
            c_algorithm = self._get_algorithm_c_name(algorithm)
        else:
            msg = "algorithm {!r} is not supported"
            raise TypeError(msg.format(algorithm))
        self.params = TSVDparams(n_components, tol, n_iter, random_state,
                                 c_algorithm)
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.components_ptr = None
        self.explained_variance_ptr = None
        self.explained_variance_ratio_ptr = None
        self.singular_values_ptr = None

        self.algo_dict = {
                'full': COV_EIG_DQ,
                'auto': COV_EIG_DQ,
                'jacobi': COV_EIG_JACOBI
            }

    def _get_algorithm_c_name(self, algorithm):
        return self.algo_dict[algorithm]

    def _initialize_arrays(self, n_components, n_rows, n_cols):

        self.trans_input_ = cudf.utils.cudautils.zeros(n_rows*n_components,
                                                       self.gdf_datatype)

        self.components_ = cudf.utils.cudautils.zeros(n_cols*n_components,
                                                      self.gdf_datatype)

        self.explained_variance_ = cudf.Series(cudf.utils.cudautils.zeros(
                                                    n_components,
                                                    self.gdf_datatype))

        self.explained_variance_ratio_ = cudf.Series(np.zeros(
                                                       n_components,
                                                       self.gdf_datatype))

        self.mean_ = cudf.Series(cudf.utils.cudautils.zeros(n_cols,
                                                            self.gdf_datatype))

        self.singular_values_ = cudf.Series(cudf.utils.cudautils.zeros(
                                                    n_components,
                                                    self.gdf_datatype))

        self.noise_variance_ = cudf.Series(np.zeros(1,
                                                    dtype=self.gdf_datatype))

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def _get_gdf_as_matrix_ptr(self, gdf):
        return self._get_ctype_ptr(gdf.as_gpu_matrix())

    def _fit_spmg(self, X, _transform=True, gpu_ids=[]):

        if (not isinstance(X, np.ndarray)):
            msg = "X matrix must be a Numpy ndarray. Dask will be supported" \
                  + " in the next version."
            raise TypeError(msg)

        n_gpus = len(gpu_ids)

        n_rows = X.shape[0]
        n_cols = X.shape[1]
        self.params.n_rows = n_rows
        self.params.n_cols = n_cols

        if (n_rows < 2):
            msg = "X must have at least two rows."
            raise TypeError(msg)

        if (n_cols < 2):
            msg = "X must have at least two columns."
            raise TypeError(msg)

        if (not np.isfortran(X)):
            X = np.array(X, order='F', dtype=X.dtype)

        cpdef paramsTSVD params
        params.n_components = self.params.n_components
        params.n_rows = n_rows
        params.n_cols = n_cols
        params.n_iterations = self.params.iterated_power
        params.tol = self.params.tol
        params.algorithm = self.params.svd_solver

        cdef uintptr_t X_ptr, components_ptr, explained_variance_ptr
        cdef uintptr_t explained_variance_ratio_ptr, singular_values_ptr,
        cdef uintptr_t trans_input_ptr, gpu_ids_ptr

        self.gdf_datatype = X.dtype

        self.components_ = np.zeros((n_cols, self.params.n_components),
                                    dtype=X.dtype, order='F')
        self.explained_variance_ = np.zeros(self.params.n_components,
                                            dtype=X.dtype, order='F')
        self.explained_variance_ratio_ = np.zeros(self.params.n_components,
                                                  dtype=X.dtype, order='F')
        self.singular_values_ = np.zeros(self.params.n_components,
                                         dtype=X.dtype, order='F')
        self.trans_input_ = np.zeros((n_rows, self.params.n_components),
                                     dtype=X.dtype, order='F')

        X_ptr = X.ctypes.data
        components_ptr = self.components_.ctypes.data
        explained_variance_ptr = self.explained_variance_ratio_.ctypes.data
        exp_vr = self.explained_variance_ratio_
        explained_variance_ratio_ptr = exp_vr.ctypes.data
        singular_values_ptr = self.singular_values_.ctypes.data
        trans_input_ptr = self.trans_input_.ctypes.data
        gpu_ids_32 = np.array(gpu_ids, dtype=np.int32)
        gpu_ids_ptr = gpu_ids_32.ctypes.data

        if not _transform:
            if self.gdf_datatype.type == np.float32:
                tsvdFitSPMG(<float*>X_ptr,
                            <float*>components_ptr,
                            <float*>singular_values_ptr,
                            params,
                            <int*>gpu_ids_ptr,
                            <int>n_gpus)

            else:
                tsvdFitSPMG(<float*>X_ptr,
                            <float*>components_ptr,
                            <float*>singular_values_ptr,
                            params,
                            <int*>gpu_ids_ptr,
                            <int>n_gpus)
        else:
            if self.gdf_datatype.type == np.float32:
                tsvdFitTransformSPMG(<float*>X_ptr,
                                     <float*>trans_input_ptr,
                                     <float*>components_ptr,
                                     <float*>explained_variance_ptr,
                                     <float*>explained_variance_ratio_ptr,
                                     <float*>singular_values_ptr,
                                     params,
                                     <int*>gpu_ids_ptr,
                                     <int>n_gpus)

            else:
                tsvdFitTransformSPMG(<double*>X_ptr,
                                     <double*>trans_input_ptr,
                                     <double*>components_ptr,
                                     <double*>explained_variance_ptr,
                                     <double*>explained_variance_ratio_ptr,
                                     <double*>singular_values_ptr,
                                     params,
                                     <int*>gpu_ids_ptr,
                                     <int>n_gpus)

        self.components_ = np.transpose(self.components_)

        return self

    def _fit_transform_spmg(self, X, gpu_ids):
        self._fit_spmg(X, True, gpu_ids)
        return self.trans_input_

    def _inverse_transform_spmg(self, X, gpu_ids=[]):
        n_gpus = len(gpu_ids)

        if (not np.isfortran(X)):
            X = np.array(X, order='F')

        if (not np.isfortran(self.components_)):
            self.components_ = np.array(self.components_, order='F',
                                        dtype=X.dtype)

        n_rows = X.shape[0]
        n_cols = X.shape[1]

        if (n_rows < 2):
            msg = "X must have at least two rows."
            raise TypeError(msg)

        if (n_cols < 2):
            msg = "X must have at least two columns."
            raise TypeError(msg)

        cpdef paramsTSVD params
        params.n_components = self.params.n_components
        params.n_rows = n_rows
        params.n_cols = self.params.n_cols

        original_X = np.zeros((n_rows, self.params.n_cols), dtype=X.dtype,
                              order='F')

        cdef uintptr_t X_ptr, original_X_ptr, gpu_ids_ptr, components_ptr

        self.gdf_datatype = X.dtype

        X_ptr = X.ctypes.data
        original_X_ptr = original_X.ctypes.data
        gpu_ids_32 = np.array(gpu_ids, dtype=np.int32)
        gpu_ids_ptr = gpu_ids_32.ctypes.data
        components_ptr = self.components_.ctypes.data

        if self.gdf_datatype.type == np.float32:
            tsvdInverseTransformSPMG(<float*>X_ptr,
                                     <float*>components_ptr,
                                     <bool>False,
                                     <float*>original_X_ptr,
                                     params,
                                     <int*>gpu_ids_ptr,
                                     <int>n_gpus)

        else:
            tsvdInverseTransformSPMG(<double*>X_ptr,
                                     <double*>components_ptr,
                                     <bool>False,
                                     <double*>original_X_ptr,
                                     params,
                                     <int*>gpu_ids_ptr,
                                     <int>n_gpus)

        return original_X

    def _transform_spmg(self, X, gpu_ids=[]):
        n_gpus = len(gpu_ids)

        if (not np.isfortran(X)):
            X = np.array(X, order='F')

        if (not np.isfortran(self.components_)):
            self.components_ = np.array(self.components_, order='F',
                                        dtype=X.dtype)

        n_rows = X.shape[0]
        n_cols = X.shape[1]

        if (n_rows < 2):
            msg = "X must have at least two rows."
            raise TypeError(msg)

        if (n_cols < 2):
            msg = "X must have at least two columns."
            raise TypeError(msg)

        cpdef paramsTSVD params
        params.n_components = self.params.n_components
        params.n_rows = n_rows
        params.n_cols = self.params.n_cols

        trans_X = np.zeros((n_rows, self.params.n_components), dtype=X.dtype,
                           order='F')

        cdef uintptr_t X_ptr, trans_X_ptr, gpu_ids_ptr, components_ptr

        self.gdf_datatype = X.dtype

        X_ptr = X.ctypes.data
        trans_X_ptr = trans_X.ctypes.data
        gpu_ids_32 = np.array(gpu_ids, dtype=np.int32)
        gpu_ids_ptr = gpu_ids_32.ctypes.data
        components_ptr = self.components_.ctypes.data

        if self.gdf_datatype.type == np.float32:
            tsvdTransformSPMG(<float*>X_ptr,
                              <float*>components_ptr,
                              <bool>True,
                              <float*>trans_X_ptr,
                              params,
                              <int*>gpu_ids_ptr,
                              <int>n_gpus)

        else:
            tsvdTransformSPMG(<double*>X_ptr,
                              <double*>components_ptr,
                              <bool>True,
                              <double*>trans_X_ptr,
                              params,
                              <int*>gpu_ids_ptr,
                              <int>n_gpus)

        return trans_X

    def fit(self, X, n_gpus=1, gpu_ids=[]):
        """
        Fit LSI model on training cudf DataFrame X.

        Parameters
        ----------
        X : cuDF DataFrame, dense matrix, shape (n_samples, n_features)
            Training data (floats or doubles)

        n_gpus : int
                 Number of gpus to be used for prediction. If gpu_ids parameter
                 has more than element, this parameter is ignored.

        gpu_ids: int array
                 GPU ids to be used for prediction.

        """

        if (len(gpu_ids) > 1):
            return self._fit_spmg(X, True, gpu_ids)
        elif (n_gpus > 1):
            for i in range(0, n_gpus):
                gpu_ids.append(i)
            return self._fit_spmg(X, True, gpu_ids)
        else:
            raise ValueError('Number of GPUS should be 2 or more'
                             'For single GPU, use the normal TruncatedSVD')

    def fit_transform(self, X, n_gpus=1, gpu_ids=[]):
        """
        Fit LSI model to X and perform dimensionality reduction on X.

        Parameters
        ----------
        X GDF : cuDF DataFrame, dense matrix, shape (n_samples, n_features)
                Training data (floats or doubles)

        n_gpus : int
                 Number of gpus to be used for prediction. If gpu_ids parameter
                 has more than element, this parameter is ignored.

        gpu_ids: int array
                 GPU ids to be used for prediction.

        Returns
        ----------
        X_new : cuDF DataFrame, shape (n_samples, n_components)
                Reduced version of X. This will always be a dense cuDF
                DataFrame

        """

        if (len(gpu_ids) > 1):
            return self._fit_transform_spmg(X, gpu_ids)
        elif (n_gpus > 1):
            for i in range(0, n_gpus):
                gpu_ids.append(i)
            return self._fit_transform_spmg(X, gpu_ids)
        else:
            raise ValueError('Number of GPUS should be 2 or more'
                             'For single GPU, use the normal TruncatedSVD')

    def inverse_transform(self, X, n_gpus=1, gpu_ids=[]):
        """
        Transform X back to its original space.

        Returns a cuDF DataFrame X_original whose transform would be X.

        Parameters
        ----------
        X : cuDF DataFrame, shape (n_samples, n_components)
            New data.

        n_gpus : int
                 Number of gpus to be used for prediction. If gpu_ids parameter
                 has more than element, this parameter is ignored.

        gpu_ids: int array
                 GPU ids to be used for prediction.

        Returns
        ----------
        X_original : cuDF DataFrame, shape (n_samples, n_features)
                     Note that this is always a dense cuDF DataFrame.

        """

        if (len(gpu_ids) > 1):
            return self._inverse_transform_spmg(X, gpu_ids)
        elif (n_gpus > 1):
            for i in range(0, n_gpus):
                gpu_ids.append(i)
            return self._inverse_transform_spmg(X, gpu_ids)
        else:
            raise ValueError('Number of GPUS should be 2 or more'
                             'For single GPU, use the normal TruncatedSVD')

    def transform(self, X, n_gpus=1, gpu_ids=[]):
        """
        Perform dimensionality reduction on X.

        Parameters
        ----------
        X : cuDF DataFrame, dense matrix, shape (n_samples, n_features)
            New data.

        n_gpus : int
                 Number of gpus to be used for prediction. If gpu_ids parameter
                 has more than element, this parameter is ignored.

        gpu_ids: int array
                 GPU ids to be used for prediction.

        Returns
        ----------
        X_new : cuDF DataFrame, shape (n_samples, n_components)
            Reduced version of X. This will always be a dense DataFrame.

        """

        if (len(gpu_ids) > 1):
            return self._transform_spmg(X, gpu_ids)
        elif (n_gpus > 1):
            for i in range(0, n_gpus):
                gpu_ids.append(i)
            return self._transform_spmg(X, gpu_ids)
        else:
            raise ValueError('Number of GPUS should be 2 or more'
                             'For single GPU, use the normal TruncatedSVD')

