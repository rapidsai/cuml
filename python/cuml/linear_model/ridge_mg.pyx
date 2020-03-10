#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
from cuml.common.array import CumlArray
from cuml.common.handle cimport cumlHandle
from cuml.decomposition.utils cimport *
from cuml.utils import input_to_cuml_array

from cuml.linear_model import Ridge

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

cdef extern from "cumlprims/opg/ridge.hpp" namespace "ML::Ridge::opg":

    cdef void fit(cumlHandle& handle,
                  RankSizePair **rank_sizes,
                  size_t n_parts,
                  floatData_t **input,
                  size_t n_rows,
                  size_t n_cols,
                  floatData_t **labels,
                  float *alpha,
                  int n_alpha,
                  float *coef,
                  float *intercept,
                  bool fit_intercept,
                  bool normalize,
                  int algo,
                  bool verbose) except +

    cdef void fit(cumlHandle& handle,
                  RankSizePair **rank_sizes,
                  size_t n_parts,
                  doubleData_t **input,
                  size_t n_rows,
                  size_t n_cols,
                  doubleData_t **labels,
                  double *alpha,
                  int n_alpha,
                  double *coef,
                  double *intercept,
                  bool fit_intercept,
                  bool normalize,
                  int algo,
                  bool verbose) except +


class RidgeMG(Ridge):

    def __init__(self, **kwargs):
        super(RidgeMG, self).__init__(**kwargs)

    def fit(self, input_data, n_rows, n_cols, partsToSizes, rank):
        """
        Fit function for MNMG Ridge Regression.
        This not meant to be used as
        part of the public API.
        :param X: array of local dataframes / array partitions
        :param n_rows: total number of rows
        :param n_cols: total number of cols
        :param partsToSizes: array of tuples in the format: [(rank,size)]
        :return: self
        """

        arr_interfaces = []
        arr_interfaces_y = []

        for i in range(len(input_data)):
            X_m, n_rows_X, self.n_cols, self.dtype = \
                input_to_cuml_array(input_data[i][0],
                                    check_dtype=[np.float32, np.float64])

            arr_interfaces.append({"obj": X_m,
                                   "data": X_m.ptr,
                                   "shape": (n_rows_X, self.n_cols)})

            y_m, n_rows_y, n_cols_y, self.dtype = \
                input_to_cuml_array(input_data[i][1],
                                    check_dtype=[np.float32, np.float64])

            arr_interfaces_y.append({"obj": y_m,
                                     "data": y_m.ptr,
                                     "shape": (n_rows_y, n_cols_y)})

        n_total_parts = len(input_data)
        cdef RankSizePair **rankSizePair = <RankSizePair**> \
            malloc(sizeof(RankSizePair**)
                   * n_total_parts)

        for i in range(len(input_data)):
            rankSizePair[i] = <RankSizePair*> \
                malloc(sizeof(RankSizePair))
            rankSizePair[i].rank = <int>rank
            rankSizePair[i].size = <size_t>len(input_data[i][0])

        self._coef_ = CumlArray.zeros(self.n_cols,
                                      dtype=self.dtype)
        cdef uintptr_t coef_ptr = self._coef_.ptr

        cdef float float_intercept
        cdef double double_intercept
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        cdef uintptr_t data
        cdef uintptr_t labels
        cdef float float_alpha
        cdef double double_alpha
        # Only one alpha is supported.
        self.n_alpha = 1

        if self.dtype == np.float32:
            data = _build_dataFloat(arr_interfaces)
            labels = _build_dataFloat(arr_interfaces_y)
            float_alpha = self.alpha

            fit(handle_[0],
                <RankSizePair**>rankSizePair,
                <size_t> n_total_parts,
                <floatData_t**>data,
                <size_t>n_rows,
                <size_t>n_cols,
                <floatData_t**>labels,
                <float*>&float_alpha,
                <int>self.n_alpha,
                <float*>coef_ptr,
                <float*>&float_intercept,
                <bool>self.fit_intercept,
                <bool>self.normalize,
                <int>self.algo,
                False)

            self.intercept_ = float_intercept

        else:
            data = _build_dataDouble(arr_interfaces)
            labels = _build_dataDouble(arr_interfaces_y)
            double_alpha = self.alpha

            fit(handle_[0],
                <RankSizePair**>rankSizePair,
                <size_t> n_total_parts,
                <doubleData_t**>data,
                <size_t>n_rows,
                <size_t>n_cols,
                <doubleData_t**>labels,
                <double*>&double_alpha,
                <int>self.n_alpha,
                <double*>coef_ptr,
                <double*>&double_intercept,
                <bool>self.fit_intercept,
                <bool>self.normalize,
                <int>self.algo,
                False)

            self.intercept_ = double_intercept

        self.handle.sync()

        del(X_m)
        del(y_m)

        for idx in range(n_total_parts):
            free(<RankSizePair*>rankSizePair[idx])
        free(<RankSizePair**>rankSizePair)

        if self.dtype == np.float32:
            _freeFloatD(data, arr_interfaces)
            _freeFloatD(labels, arr_interfaces_y)
        else:
            _freeDoubleD(data, arr_interfaces)
            _freeDoubleD(labels, arr_interfaces_y)


# Util functions, will be moved to their own file as the other methods are
# refactored
# todo: use cuda_array_interface instead of arr_interfaces for building this

def _build_dataFloat(arr_interfaces):
    cdef floatData_t **dataF = <floatData_t **> \
        malloc(sizeof(floatData_t *)
               * len(arr_interfaces))

    cdef uintptr_t input_ptr
    for x_i in range(len(arr_interfaces)):
        x = arr_interfaces[x_i]
        input_ptr = x["data"]
        dataF[x_i] = <floatData_t *> malloc(sizeof(floatData_t))
        dataF[x_i].ptr = <float *> input_ptr
        dataF[x_i].totalSize = <size_t> x["shape"][0]
    return <size_t>dataF


def _build_dataDouble(arr_interfaces):
    cdef doubleData_t **dataD = <doubleData_t **> \
        malloc(sizeof(doubleData_t *)
               * len(arr_interfaces))

    cdef uintptr_t input_ptr
    for x_i in range(len(arr_interfaces)):
        x = arr_interfaces[x_i]
        input_ptr = x["data"]
        dataD[x_i] = <doubleData_t *> malloc(sizeof(doubleData_t))
        dataD[x_i].ptr = <double *> input_ptr
        dataD[x_i].totalSize = <size_t> x["shape"][0]
    return <size_t>dataD


def _freeDoubleD(data, arr_interfaces):
    cdef uintptr_t data_ptr = data
    cdef doubleData_t **d = <doubleData_t**>data_ptr
    for x_i in range(len(arr_interfaces)):
        free(d[x_i])
    free(d)


def _freeFloatD(data, arr_interfaces):
    cdef uintptr_t data_ptr = data
    cdef floatData_t **d = <floatData_t**>data_ptr
    for x_i in range(len(arr_interfaces)):
        free(d[x_i])
    free(d)


def _build_predData(partsToSizes, rank, n_cols, dtype):
    arr_interfaces_trans = []
    for idx, rankSize in enumerate(partsToSizes):
        rk, size = rankSize
        if rank == rk:
            trans_ary = CumlArray.zeros((size, n_cols),
                                        order="F",
                                        dtype=dtype)

            arr_interfaces_trans.append({"obj": trans_ary,
                                         "data": trans_ary.ptr,
                                         "shape": (size, n_cols)})

    return arr_interfaces_trans
