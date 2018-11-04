 # Copyright (c) 2018, NVIDIA CORPORATION.
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

cimport c_tsvd
import numpy as np
from numba import cuda
import cudf
from libcpp cimport bool
import ctypes
from libc.stdint cimport uintptr_t
from c_tsvd cimport *


class TSVDparams:
    def __init__(self,n_components,tol,iterated_power,random_state,svd_solver):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.n_cols = None
        self.n_rows = None

class TruncatedSVD:
    """
    Create a DataFrame, fill it with data, and compute Truncated Singular Value Decomposition:

    .. code-block:: python

            from cuml import TruncatedSVD
            import cudf
            import numpy as np

            gdf_float = cudf.DataFrame()
            gdf_float['0']=np.asarray([1.0,2.0,5.0],dtype=np.float32)
            gdf_float['1']=np.asarray([4.0,2.0,1.0],dtype=np.float32)
            gdf_float['2']=np.asarray([4.0,2.0,1.0],dtype=np.float32)

            tsvd_float = TruncatedSVD(n_components = 2, algorithm="jacobi", n_iter=20, tol=1e-9)
            tsvd_float.fit(gdf_float)

            print(f'components: {tsvd_float.components_}')
            print(f'explained variance: {tsvd_float.explained_variance_}')
            print(f'explained variance ratio: {tsvd_float.explained_variance_ratio_}')
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
            1 3.4638448 -0.042223275                                                                                                                     2 4.0809603    3.2164836

            Input matrix:           0         1         2
            0       1.0  4.000001  4.000001
            1 2.0000005 2.0000005 2.0000007
            2  5.000001 0.9999999 1.0000004

    For additional examples, see `the Truncated SVD  notebook <https://github.com/rapidsai/cuml/blob/master/python/notebooks/tsvd_demo.ipynb>`_. For additional documentation, see `scikitlearn's TruncatedSVD docs <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`_.

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

    def _get_algorithm_c_name(self, algorithm):
        return {
            'full': COV_EIG_DQ,
            'auto': COV_EIG_DQ,
            'jacobi': COV_EIG_JACOBI
        }[algorithm]

    def _initialize_arrays(self, input_gdf, n_components, n_rows, n_cols):

        x = []
        for col in input_gdf.columns:
            x.append(input_gdf[col]._column.dtype)
            break
        self.gdf_datatype = np.dtype(x[0])

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
        self.singular_values_ = cudf.Series(np.zeros(n_components,
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


    def fit(self, X, _transform=True):
        """
            Fit LSI model on training cudf DataFrame X.

            Parameters
            ----------
            X : cuDF DataFrame, dense matrix, shape (n_samples, n_features)
                Training data (floats or doubles)

        """

        # c params
        cpdef c_tsvd.paramsTSVD params
        params.n_components = self.params.n_components
        params.n_rows = len(X)
        params.n_cols = len(X._cols)
        params.n_iterations = self.params.iterated_power
        params.tol = self.params.tol
        params.algorithm = self.params.svd_solver

        # python params
        self.params.n_rows = len(X)
        self.params.n_cols = len(X._cols)

        self._initialize_arrays(X, self.params.n_components,
                                self.params.n_rows, self.params.n_cols)

        X_m = X.as_gpu_matrix()
        cdef uintptr_t input_ptr = self._get_ctype_ptr(X_m)

        cdef uintptr_t components_ptr = self._get_ctype_ptr(self.components_)

        cdef uintptr_t explained_var_ptr = self._get_column_ptr(
                                                    self.explained_variance_)
        cdef uintptr_t explained_var_ratio_ptr = self._get_column_ptr(
                                                self.explained_variance_ratio_)
        cdef uintptr_t singular_vals_ptr = self._get_column_ptr(
                                                self.singular_values_)
        cdef uintptr_t trans_input_ptr = self._get_ctype_ptr(self.trans_input_)

        if not _transform:
            if self.gdf_datatype.type == np.float32:
                c_tsvd.tsvdFit(<float*> input_ptr,
                               <float*> components_ptr,
                               <float*> singular_vals_ptr,
                               params)
            else:
                c_tsvd.tsvdFit(<double*> input_ptr,
                               <double*> components_ptr,
                               <double*> singular_vals_ptr,
                               params)
        else:
            if self.gdf_datatype.type == np.float32:
                c_tsvd.tsvdFitTransform(<float*> input_ptr,
                                        <float*> trans_input_ptr,
                                        <float*> components_ptr,
                                        <float*> explained_var_ptr,
                                        <float*> explained_var_ratio_ptr,
                                        <float*> singular_vals_ptr,
                                        params)
            else:
                c_tsvd.tsvdFitTransform(<double*> input_ptr,
                                        <double*> trans_input_ptr,
                                        <double*> components_ptr,
                                        <double*> explained_var_ptr,
                                        <double*> explained_var_ratio_ptr,
                                        <double*> singular_vals_ptr,
                                        params)


        components_gdf = cudf.DataFrame()
        for i in range(0, params.n_cols):
            components_gdf[str(i)] = self.components_[i*params.n_components:(i+1)*params.n_components]

        self.components_ = components_gdf
        self.components_ptr = components_ptr
        self.explained_variance_ptr = explained_var_ptr
        self.explained_variance_ratio_ptr = explained_var_ratio_ptr
        self.singular_values_ptr = singular_vals_ptr

        del(X_m)


    def fit_transform(self, X):
        """
            Fit LSI model to X and perform dimensionality reduction on X.

            Parameters
            ----------
            X GDF : cuDF DataFrame, dense matrix, shape (n_samples, n_features)
                Training data (floats or doubles)

            Returns
            ----------
            X_new : cuDF DataFrame, shape (n_samples, n_components)
                Reduced version of X. This will always be a dense cuDF DataFrame

        """
        self.fit(X, _transform=True)
        X_new = cudf.DataFrame()
        num_rows = self.params.n_rows

        for i in range(0, self.params.n_components):
            X_new[str(i)] = self.trans_input_[i*num_rows:(i+1)*num_rows]

        return X_new


    def inverse_transform(self, X):
        """
            Transform X back to its original space.

            Returns a cuDF DataFrame X_original whose transform would be X.

            Parameters
            ----------
            X : cuDF DataFrame, shape (n_samples, n_components)
                New data.

            Returns
            ----------
            X_original : cuDF DataFrame, shape (n_samples, n_features)
                Note that this is always a dense cuDF DataFrame.

        """

        cpdef c_tsvd.paramsTSVD params
        params.n_components = self.params.n_components
        params.n_rows = len(X)
        params.n_cols = self.params.n_cols

        x = []
        for col in X.columns:
            x.append(X[col]._column.dtype)
            break
        gdf_datatype = np.dtype(x[0])

        input_data = cuda.to_device(np.zeros(params.n_rows*params.n_cols,dtype=gdf_datatype.type))

        cdef uintptr_t input_ptr = input_data.device_ctypes_pointer.value
        cdef uintptr_t trans_input_ptr = X.as_gpu_matrix().device_ctypes_pointer.value
        cdef uintptr_t components_ptr = self.components_ptr

        if gdf_datatype.type == np.float32:
            c_tsvd.tsvdInverseTransform(<float*> trans_input_ptr,
                                        <float*> components_ptr,
                                        <float*> input_ptr,
                                        params)
        else:
            c_tsvd.tsvdInverseTransform(<double*> trans_input_ptr,
                                        <double*> components_ptr,
                                        <double*> input_ptr,
                                        params)

        X_original = cudf.DataFrame()
        for i in range(0, params.n_cols):
            X_original[str(i)] = input_data[i*params.n_rows:(i+1)*params.n_rows]

        return X_original



    def transform(self, X):
        """
            Perform dimensionality reduction on X.

            Parameters
            ----------
            X : cuDF DataFrame, dense matrix, shape (n_samples, n_features)
                New data.

            Returns
            ----------
            X_new : cuDF DataFrame, shape (n_samples, n_components)
                Reduced version of X. This will always be a dense DataFrame.

        """

        cpdef c_tsvd.paramsTSVD params
        params.n_components = self.params.n_components
        params.n_rows = len(X)
        params.n_cols = len(X._cols)

        x = []
        for col in X.columns:
            x.append(X[col]._column.dtype)
            break
        gdf_datatype = np.dtype(x[0])

        trans_input_data = cuda.to_device(
                              np.zeros(params.n_rows*params.n_components,
                                       dtype=gdf_datatype.type))

        X_m = X.as_gpu_matrix()
        cdef uintptr_t input_ptr = self._get_ctype_ptr(X_m)

        cdef uintptr_t trans_input_ptr = self._get_ctype_ptr(trans_input_data)
        cdef uintptr_t components_ptr = self.components_ptr

        if gdf_datatype.type == np.float32:
            c_tsvd.tsvdTransform(<float*> input_ptr,
                                 <float*> components_ptr,
                                 <float*> trans_input_ptr,
                                 params)
        else:
            c_tsvd.tsvdTransform(<double*> input_ptr,
                                 <double*> components_ptr,
                                 <double*> trans_input_ptr,
                                 params)

        X_new = cudf.DataFrame()
        for i in range(0, params.n_components):
            X_new[str(i)] = trans_input_data[i*params.n_rows:(i+1)*params.n_rows]

        del(X_m)
        return X_new

