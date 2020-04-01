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

from cuml.common.array import CumlArray


from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.decomposition.utils cimport *
from cuml.utils import input_to_dev_array, zeros

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


class BaseDecompositionMG(object):

    def __init__(self, **kwargs):
        super(BaseDecompositionMG, self).__init__(**kwargs)

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
                trans_ary = CumlArray.zeros((size, n_cols),
                                            order="F",
                                            dtype=dtype)

                arr_interfaces_trans.append({"obj": trans_ary,
                                             "data": trans_ary.ptr,
                                             "shape": (size, n_cols)})

        return arr_interfaces_trans

    def _fit(self, X, total_rows, n_cols, partsToRanks, rank,
             _transform=False):
        """
        Fit function for PCA MG. This not meant to be used as
        part of the public API.
        :param X: array of local dataframes / array partitions
        :param total_rows: total number of rows
        :param n_cols: total number of cols
        :param partsToRanks: array of tuples in the format: [(rank,size)]
        :return: self
        """

        self._set_output_type(X)

        arr_interfaces = []
        for arr in X:
            X_m, input_ptr, n_rows, self.n_cols, self.dtype = \
                input_to_dev_array(arr, check_dtype=[np.float32, np.float64])
            arr_interfaces.append({"obj": X_m,
                                   "data": input_ptr,
                                   "shape": (n_rows, self.n_cols)})

        n_total_parts = len(X)
        cdef RankSizePair **rank_size_pair = <RankSizePair**> \
            malloc(sizeof(RankSizePair**)
                   * n_total_parts)

        p2r = []

        n_rows = 0
        for i in range(len(X)):
            rank_size_pair[i] = <RankSizePair*> \
                malloc(sizeof(RankSizePair))
            rank_size_pair[i].rank = <int>rank
            n_rows += len(X[i])
            rank_size_pair[i].size = <size_t>len(X[i])
            p2r.append((rank, len(X[i])))

        self._initialize_arrays(self.n_components, total_rows, n_cols)

        arg_rank_size_pair = <size_t>rank_size_pair
        decomp_params = self._build_params(total_rows, n_cols)

        arr_interfaces_trans, data, trans_data = self._call_fit(
            arr_interfaces, p2r, rank, arg_rank_size_pair, n_total_parts,
            decomp_params)

        for idx in range(n_total_parts):
            free(<RankSizePair*>rank_size_pair[idx])
        free(<RankSizePair**>rank_size_pair)

        del(X_m)

        trans_cudf = []
        if _transform:
            for x_i in arr_interfaces_trans:
                trans_cudf.append(x_i["obj"].to_output(
                    output_type=self._get_output_type(X)))

            if self.dtype == np.float32:
                self._freeFloatD(trans_data, arr_interfaces_trans)
                self._freeFloatD(data, arr_interfaces)
            else:
                self._freeDoubleD(trans_data, arr_interfaces_trans)
                self._freeDoubleD(data, arr_interfaces)

            return trans_cudf

        return self
