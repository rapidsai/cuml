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
import pygdf
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
        self.explained_variance_ = pygdf.Series(
                                      np.zeros(n_components,
                                               dtype=self.gdf_datatype))
        self.explained_variance_ratio_ = pygdf.Series(
                                            np.zeros(n_components,
                                                     dtype=self.gdf_datatype))
        self.singular_values_ = pygdf.Series(np.zeros(n_components,
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

    def fit(self, input_gdf, _transform=True):
        # c params
        cpdef c_tsvd.paramsTSVD params
        params.n_components = self.params.n_components
        params.n_rows = len(input_gdf)
        params.n_cols = len(input_gdf._cols)
        params.n_iterations = self.params.iterated_power
        params.tol = self.params.tol
        params.algorithm = self.params.svd_solver

        # python params
        self.params.n_rows = len(input_gdf)
        self.params.n_cols = len(input_gdf._cols)

        self._initialize_arrays(input_gdf, self.params.n_components,
                                self.params.n_rows, self.params.n_cols)

        cdef uintptr_t input_ptr = self._get_gdf_as_matrix_ptr(input_gdf)

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


        components_gdf = pygdf.DataFrame()
        for i in range(0, params.n_cols):
            components_gdf[str(i)] = self.components_[i*params.n_components:(i+1)*params.n_components]

        self.components_ = components_gdf
        self.components_ptr = components_ptr
        self.explained_variance_ptr = explained_var_ptr
        self.explained_variance_ratio_ptr = explained_var_ratio_ptr
        self.singular_values_ptr = singular_vals_ptr

    def fit_transform(self, input_gdf):
        self.fit(input_gdf, _transform=True)
        trans_input_gdf = pygdf.DataFrame()
        num_rows = self.params.n_rows

        for i in range(0, self.params.n_components):
            trans_input_gdf[str(i)] = self.trans_input_[i*num_rows:(i+1)*num_rows]

        return trans_input_gdf

    def inverse_transform(self, trans_input_gdf):

        cpdef c_tsvd.paramsTSVD params
        params.n_components = self.params.n_components
        params.n_rows = len(trans_input_gdf)
        params.n_cols = self.params.n_cols

        x = []
        for col in trans_input_gdf.columns:
            x.append(trans_input_gdf[col]._column.dtype)
            break
        gdf_datatype = np.dtype(x[0])

        input_data = cuda.to_device(np.zeros(params.n_rows*params.n_cols,dtype=gdf_datatype.type))

        cdef uintptr_t input_ptr = input_data.device_ctypes_pointer.value
        cdef uintptr_t trans_input_ptr = trans_input_gdf.as_gpu_matrix().device_ctypes_pointer.value
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

        input_gdf = pygdf.DataFrame()
        for i in range(0, params.n_cols):
            input_gdf[str(i)] = input_data[i*params.n_rows:(i+1)*params.n_rows]

        return input_gdf


    def transform(self, input_gdf):

        cpdef c_tsvd.paramsTSVD params
        params.n_components = self.params.n_components
        params.n_rows = len(input_gdf)
        params.n_cols = len(input_gdf._cols)

        x = []
        for col in input_gdf.columns:
            x.append(input_gdf[col]._column.dtype)
            break
        gdf_datatype = np.dtype(x[0])

        trans_input_data = cuda.to_device(
                              np.zeros(params.n_rows*params.n_components,
                                       dtype=gdf_datatype.type))

        cdef uintptr_t trans_input_ptr = self._get_ctype_ptr(trans_input_data)
        cdef uintptr_t input_ptr = self._get_gdf_as_matrix_ptr(input_gdf)
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

        trans_input_gdf = pygdf.DataFrame()
        for i in range(0, params.n_components):
            trans_input_gdf[str(i)] = trans_input_data[i*params.n_rows:(i+1)*params.n_rows]

        return trans_input_gdf

