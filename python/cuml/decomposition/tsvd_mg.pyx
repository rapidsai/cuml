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

from libc.stdlib cimport malloc, free

from libcpp cimport bool
from libc.stdint cimport uintptr_t, uint32_t, uint64_t
from cython.operator cimport dereference as deref

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.decomposition.utils cimport *
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros

from cuml.decomposition import TruncatedSVD

cdef extern from "cumlprims/opg/matrix/data.hpp" \
                 namespace "MLCommon::Matrix":

    cdef cppclass floatData_t:
        floatData_t(float *ptr, size_t totalSize)
        float *ptr
        size_t totalSize

    cdef cppclass doubleData_t:
        doubleData_t(double *ptr, size_t totalSize)
        double *ptr
        size_t totalSize

cdef extern from "cumlprims/opg/matrix/part_descriptor.hpp" \
                 namespace "MLCommon::Matrix":

    cdef cppclass RankSizePair:
        int rank
        size_t size

cdef extern from "cumlprims/opg/tsvd.hpp" namespace "ML::TSVD::opg":

    cdef void fit_transform(cumlHandle& handle,
                            RankSizePair **rank_sizes,
                            size_t n_parts,
                            floatData_t **input,
                            floatData_t **trans_input,
                            float *components,
                            float *explained_var,
                            float *explained_var_ratio,
                            float *singular_vals,
                            paramsTSVD prms,
                            bool verbose) except +

    cdef void fit_transform(cumlHandle& handle,
                            RankSizePair **rank_sizes,
                            size_t n_parts,
                            doubleData_t **input,
                            doubleData_t **trans_input,
                            double *components,
                            double *explained_var,
                            double *explained_var_ratio,
                            double *singular_vals,
                            paramsTSVD prms,
                            bool verbose) except +

    cdef void transform(cumlHandle& handle,
                        RankSizePair **rank_sizes,
                        size_t n_parts,
                        floatData_t **input,
                        float *components,
                        floatData_t **trans_input,
                        paramsTSVD prms,
                        bool verbose) except +

    cdef void transform(cumlHandle& handle,
                        RankSizePair **rank_sizes,
                        size_t n_parts,
                        doubleData_t **input,
                        double *components,
                        doubleData_t **trans_input,
                        paramsTSVD prms,
                        bool verbose) except +

    cdef void inverse_transform(cumlHandle& handle,
                                RankSizePair **rank_sizes,
                                size_t n_parts,
                                floatData_t **trans_input,
                                float *components,
                                floatData_t **input,
                                paramsTSVD prms,
                                bool verbose) except +

    cdef void inverse_transform(cumlHandle& handle,
                                RankSizePair **rank_sizes,
                                size_t n_parts,
                                doubleData_t **trans_input,
                                double *components,
                                doubleData_t **input,
                                paramsTSVD prms,
                                bool verbose) except +


class TSVDMG(TruncatedSVD):

    def __init__(self, **kwargs):
        super(TSVDMG, self).__init__(**kwargs)

    def _initialize_arrays(self, n_components, n_rows, n_cols):

        self.components_ary = rmm.to_device(zeros(n_components*n_cols,
                                                  dtype=self.dtype))

        self.explained_variance_ = cudf.Series(zeros(n_components,
                                               dtype=self.dtype))

        self.explained_variance_ratio_ = cudf.Series(zeros(n_components,
                                                     dtype=self.dtype))

        self.singular_values_ = cudf.Series(zeros(n_components,
                                                  dtype=self.dtype))

    def _build_dataFloat(self, arr_interfaces):
        cdef floatData_t ** dataF = < floatData_t ** > \
            malloc(sizeof(floatData_t *)
                   * len(arr_interfaces))

        cdef uintptr_t input_ptr
        for x_i in range(len(arr_interfaces)):
            x = arr_interfaces[x_i]
            input_ptr = x["data"]
            dataF[x_i] = < floatData_t * > malloc(sizeof(floatData_t))
            dataF[x_i].ptr = < float * > input_ptr
            dataF[x_i].totalSize = < size_t > x["shape"][0]
        return <size_t>dataF

    def _build_dataDouble(self, arr_interfaces):
        cdef doubleData_t ** dataD = < doubleData_t ** > \
            malloc(sizeof(doubleData_t *)
                   * len(arr_interfaces))

        cdef uintptr_t input_ptr
        for x_i in range(len(arr_interfaces)):
            x = arr_interfaces[x_i]
            input_ptr = x["data"]
            dataD[x_i] = < doubleData_t * > malloc(sizeof(doubleData_t))
            dataD[x_i].ptr = < double * > input_ptr
            dataD[x_i].totalSize = < size_t > x["shape"][0]
        return <size_t>dataD

    def _freeDoubleD(self, data, arr_interfaces):
        cdef uintptr_t data_ptr = data
        cdef doubleData_t **d = <doubleData_t**>data_ptr
        for x_i in range(len(arr_interfaces)):
            free(d[x_i])
        free(d)

    def _freeFloatD(self, data, arr_interfaces):
        cdef uintptr_t data_ptr = data
        cdef floatData_t **d = <floatData_t**>data_ptr
        for x_i in range(len(arr_interfaces)):
            free(d[x_i])
        free(d)

    def _build_transData(self, partsToRanks, rnk, n_cols, dtype):
        arr_interfaces_trans = []
        for idx, rankSize in enumerate(partsToRanks):
            rank, size = rankSize
            if rnk == rank:
                trans_ary = rmm.to_device(zeros((size, n_cols),
                                                order="F",
                                                dtype=dtype))

                trans_ptr = get_dev_array_ptr(trans_ary)
                arr_interfaces_trans.append({"obj": trans_ary,
                                             "data": trans_ptr,
                                             "shape": (size, n_cols)})

        return arr_interfaces_trans

    def fit(self, X, M, N, partsToRanks, rnk, _transform=False):
        """
        Fit function for PCA MG. This not meant to be used as
        part of the public API.
        :param X: array of local dataframes / array partitions
        :param M: total number of rows
        :param N: total number of cols
        :param partsToRanks: array of tuples in the format: [(rank,size)]
        :return: self
        """

        arr_interfaces = []
        for arr in X:
            X_m, input_ptr, n_rows, self.n_cols, self.dtype = \
                input_to_dev_array(arr, check_dtype=[np.float32, np.float64])
            arr_interfaces.append({"obj": X_m,
                                   "data": input_ptr,
                                   "shape": (n_rows, self.n_cols)})


        self.n_cols = N
        cpdef paramsTSVD params
        params.n_components = self.n_components
        params.n_rows = M
        params.n_cols = N
        params.n_iterations = self.n_iter
        params.tol = self.tol
        params.algorithm = self.c_algorithm
        

        n_total_parts = 0
        for idx, rankSize in enumerate(partsToRanks):
            rank, size = rankSize
            if rnk == rank:
                n_total_parts = n_total_parts + 1

        
        cdef RankSizePair **rankSizePair = <RankSizePair**> \
            malloc(sizeof(RankSizePair**)
                   * n_total_parts)

        indx = 0
        n_part_row = 0

        for idx, rankSize in enumerate(partsToRanks):
            rank, size = rankSize
            if rnk == rank:
                rankSizePair[indx] = <RankSizePair*> \
                    malloc(sizeof(RankSizePair))
                rankSizePair[indx].rank = <int>rank
                rankSizePair[indx].size = <size_t>size
                n_part_row = n_part_row + rankSizePair[indx].size
                indx = indx + 1

        self._initialize_arrays(params.n_components,
                                params.n_rows, params.n_cols)

        cdef uintptr_t comp_ptr = get_dev_array_ptr(self.components_ary)
        cdef uintptr_t explained_var_ptr = \
            get_cudf_column_ptr(self.explained_variance_)
        cdef uintptr_t explained_var_ratio_ptr = \
            get_cudf_column_ptr(self.explained_variance_ratio_)
        cdef uintptr_t singular_vals_ptr = \
            get_cudf_column_ptr(self.singular_values_)
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef uintptr_t data
        cdef uintptr_t trans_data
        arr_interfaces_trans = []

        if self.dtype == np.float32:
            data = self._build_dataFloat(arr_interfaces)
            arr_interfaces_trans = self._build_transData(partsToRanks,
                                                         rnk,
                                                         self.n_components,
                                                         np.float32)
            trans_data = self._build_dataFloat(arr_interfaces_trans)

            fit_transform(handle_[0],
                          <RankSizePair**>rankSizePair,
                          <size_t> n_total_parts,
                          <floatData_t**> data,
                          <floatData_t**> trans_data,
                          <float*> comp_ptr,
                          <float*> explained_var_ptr,
                          <float*> explained_var_ratio_ptr,
                          <float*> singular_vals_ptr,
                          params,
                          False)
            
        else:
            data = self._build_dataDouble(arr_interfaces)
            arr_interfaces_trans = self._build_transData(partsToRanks,
                                                         rnk,
                                                         self.n_components,
                                                         np.float64)
            trans_data = self._build_dataDouble(arr_interfaces_trans)

            fit_transform(handle_[0],
                          <RankSizePair**>rankSizePair,
                          <size_t> n_total_parts,
                          <doubleData_t**> data,
                          <doubleData_t**> trans_data,
                          <double*> comp_ptr,
                          <double*> explained_var_ptr,
                          <double*> explained_var_ratio_ptr,
                          <double*> singular_vals_ptr,
                          params,
                          False)

        self.handle.sync()

        for idx in range(n_total_parts):
            free(<RankSizePair*>rankSizePair[idx])
        free(<RankSizePair**>rankSizePair)
        

        self.components_ = cudf.DataFrame()
        for i in range(0, params.n_cols):
            n_c = params.n_components
            self.components_[str(i)] = self.components_ary[i*n_c:(i+1)*n_c]

        del(X_m)

        trans_cudf = []
        if _transform:
            for x_i in arr_interfaces_trans:
                trans_cudf.append(cudf.DataFrame.from_gpu_matrix(x_i["obj"]))

            if self.dtype == np.float32:
                self._freeFloatD(trans_data, arr_interfaces_trans)
                self._freeFloatD(data, arr_interfaces)
            else:
                self._freeDoubleD(trans_data, arr_interfaces_trans)
                self._freeDoubleD(data, arr_interfaces)

            return trans_cudf
                

    def transform(self, X, M, N, partsToRanks, rnk):
        """
        Transform function for PCA MG. This not meant to be used as
        part of the public API.
        :param X: array of local dataframes / array partitions
        :param M: total number of rows
        :param N: total number of cols
        :param partsToRanks: array of tuples in the format: [(rank,size)]
        :return: self
        """

        if N != self.n_cols:
            raise Exception("Number of columns of the X has to match with "
                            "number of columns of the data was fit to model.")

        arr_interfaces = []
        for arr in X:
            X_m, input_ptr, n_rows, n_cols, self.dtype = \
                input_to_dev_array(arr, check_dtype=[np.float32, np.float64])
            arr_interfaces.append({"obj": X_m,
                                   "data": input_ptr,
                                   "shape": (n_rows, n_cols)})

        cpdef paramsTSVD params
        params.n_components = self.n_components
        params.n_rows = M
        params.n_cols = self.n_cols
        params.n_iterations = self.n_iter
        params.tol = self.tol
        params.algorithm = self.c_algorithm

        n_total_parts = 0
        for idx, rankSize in enumerate(partsToRanks):
            rank, size = rankSize
            if rnk == rank:
                n_total_parts = n_total_parts + 1

        cdef RankSizePair **rankSizePair = <RankSizePair**> \
            malloc(sizeof(RankSizePair**)
                   * n_total_parts)

        indx = 0
        n_part_row = 0

        for idx, rankSize in enumerate(partsToRanks):
            rank, size = rankSize
            if rnk == rank:
                rankSizePair[indx] = <RankSizePair*> \
                    malloc(sizeof(RankSizePair))
                rankSizePair[indx].rank = <int>rank
                rankSizePair[indx].size = <size_t>size
                n_part_row = n_part_row + rankSizePair[indx].size
                indx = indx + 1

        cdef uintptr_t comp_ptr = get_dev_array_ptr(self.components_ary)
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef uintptr_t data
        cdef uintptr_t trans_data
        arr_interfaces_trans = []

        if self.dtype == np.float32:
            data = self._build_dataFloat(arr_interfaces)
            arr_interfaces_trans = self._build_transData(partsToRanks,
                                                         rnk,
                                                         self.n_components,
                                                         np.float32)
            trans_data = self._build_dataFloat(arr_interfaces_trans)

            transform(handle_[0],
                      <RankSizePair**>rankSizePair,
                      <size_t> n_total_parts,
                      <floatData_t**> data,
                      <float*> comp_ptr,
                      <floatData_t**> trans_data,
                      params,
                      False)

        else:
            data = self._build_dataDouble(arr_interfaces)
            arr_interfaces_trans = self._build_transData(partsToRanks,
                                                         rnk,
                                                         self.n_components,
                                                         np.float64)
            trans_data = self._build_dataDouble(arr_interfaces_trans)

            transform(handle_[0],
                      <RankSizePair**>rankSizePair,
                      <size_t> n_total_parts,
                      <doubleData_t**> data,
                      <double*> comp_ptr,
                      <doubleData_t**> trans_data,
                      params,
                      False)

        self.handle.sync()

        for idx in range(n_total_parts):
            free(<RankSizePair*>rankSizePair[idx])
        free(<RankSizePair**>rankSizePair)

        del(X_m)

        trans_cudf = []
        for x_i in arr_interfaces_trans:
            trans_cudf.append(cudf.DataFrame.from_gpu_matrix(x_i["obj"]))

        if self.dtype == np.float32:
            self._freeFloatD(trans_data, arr_interfaces_trans)
            self._freeFloatD(data, arr_interfaces)
        else:
            self._freeDoubleD(trans_data, arr_interfaces_trans)
            self._freeDoubleD(data, arr_interfaces)

        return trans_cudf
        
    def inverse_transform(self, X, M, N, partsToRanks, rnk):
        """
        Transform function for PCA MG. This not meant to be used as
        part of the public API.
        :param X: array of local dataframes / array partitions
        :param M: total number of rows
        :param N: total number of cols
        :param partsToRanks: array of tuples in the format: [(rank,size)]
        :return: self
        """

        if N != self.n_components:
            raise Exception("Number of columns of the X has to match with "
                            "number of principal components.")

        arr_inter_tran = []
        for arr in X:
            X_m, input_ptr, n_rows, n_cols, self.dtype = \
                input_to_dev_array(arr, check_dtype=[np.float32, np.float64])
            arr_inter_tran.append({"obj": X_m,
                                   "data": input_ptr,
                                   "shape": (n_rows, n_cols)})

        cpdef paramsTSVD params
        params.n_components = self.n_components
        params.n_rows = M
        params.n_cols = self.n_cols
        params.n_iterations = self.n_iter
        params.tol = self.tol
        params.algorithm = self.c_algorithm

        n_total_parts = 0
        for idx, rankSize in enumerate(partsToRanks):
            rank, size = rankSize
            if rnk == rank:
                n_total_parts = n_total_parts + 1

        cdef RankSizePair **rankSizePair = <RankSizePair**> \
            malloc(sizeof(RankSizePair**)
                   * n_total_parts)

        indx = 0
        n_part_row = 0

        for idx, rankSize in enumerate(partsToRanks):
            rank, size = rankSize
            if rnk == rank:
                rankSizePair[indx] = <RankSizePair*> \
                    malloc(sizeof(RankSizePair))
                rankSizePair[indx].rank = <int>rank
                rankSizePair[indx].size = <size_t>size
                n_part_row = n_part_row + rankSizePair[indx].size
                indx = indx + 1

        cdef uintptr_t comp_ptr = get_dev_array_ptr(self.components_ary)
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef uintptr_t data
        cdef uintptr_t trans_data
        arr_interfaces = []

        if self.dtype == np.float32:
            trans_data = self._build_dataFloat(arr_inter_tran)
            arr_interfaces = self._build_transData(partsToRanks,
                                                   rnk,
                                                   self.n_cols,
                                                   np.float32)
            data = self._build_dataFloat(arr_interfaces)

            inverse_transform(handle_[0],
                              <RankSizePair**>rankSizePair,
                              <size_t> n_total_parts,
                              <floatData_t**> trans_data,
                              <float*> comp_ptr,
                              <floatData_t**> data,
                              params,
                              False)
        else:
            trans_data = self._build_dataDouble(arr_inter_tran)
            arr_interfaces = self._build_transData(partsToRanks,
                                                   rnk,
                                                   self.n_cols,
                                                   np.float64)
            data = self._build_dataDouble(arr_interfaces)

            inverse_transform(handle_[0],
                              <RankSizePair**>rankSizePair,
                              <size_t> n_total_parts,
                              <doubleData_t**> trans_data,
                              <double*> comp_ptr,
                              <doubleData_t**> data,
                              params,
                              False)

        self.handle.sync()

        for idx in range(n_total_parts):
            free(<RankSizePair*>rankSizePair[idx])
        free(<RankSizePair**>rankSizePair)

        del(X_m)

        trans_cudf = []
        for x_i in arr_interfaces:
            trans_cudf.append(cudf.DataFrame.from_gpu_matrix(x_i["obj"]))

        if self.dtype == np.float32:
            self._freeFloatD(trans_data, arr_interfaces)
            self._freeFloatD(data, arr_inter_tran)
        else:
            self._freeDoubleD(trans_data, arr_interfaces)
            self._freeDoubleD(data, arr_inter_tran)

        return trans_cudf
