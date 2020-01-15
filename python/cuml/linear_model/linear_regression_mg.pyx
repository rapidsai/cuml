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

from cuml.linear_model import LinearRegression

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

cdef extern from "cumlprims/opg/ols.hpp" namespace "ML::OLS::opg":

    cdef void fit(cumlHandle& handle,
                  RankSizePair **rank_sizes,
                  size_t n_parts,
                  floatData_t **input,
                  size_t n_rows,
                  size_t n_cols,
                  floatData_t **labels,
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
                  double *coef,
                  double *intercept,
                  bool fit_intercept,
                  bool normalize,
                  int algo,
                  bool verbose) except +

    cdef void predict(cumlHandle& handle,
                      RankSizePair **rank_sizes,
                      size_t n_parts,
                      floatData_t **input,
                      size_t n_rows,
                      size_t n_cols,
                      float *coef,
                      float intercept,
                      floatData_t **preds,
                      bool verbose) except +

    cdef void predict(cumlHandle& handle,
                      RankSizePair **rank_sizes,
                      size_t n_parts,
                      doubleData_t **input,
                      size_t n_rows,
                      size_t n_cols,
                      double *coef,
                      double intercept,
                      doubleData_t **preds,
                      bool verbose) except +


class LinearRegressionMG(LinearRegression):

    def __init__(self, **kwargs):
        super(LinearRegressionMG, self).__init__(**kwargs)

    def _build_dataFloat(self, arr_interfaces):
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

    def _build_dataDouble(self, arr_interfaces):
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

    def _build_predData(self, partsToSizes, rank, n_cols, dtype):
        arr_interfaces_trans = []
        for idx, rankSize in enumerate(partsToSizes):
            rk, size = rankSize
            if rank == rk:
                trans_ary = rmm.to_device(zeros((size, n_cols),
                                                order="F",
                                                dtype=dtype))

                trans_ptr = get_dev_array_ptr(trans_ary)
                arr_interfaces_trans.append({"obj": trans_ary,
                                             "data": trans_ptr,
                                             "shape": (size, n_cols)})

        return arr_interfaces_trans

    def fit(self, X, y, n_rows, n_cols, partsToSizes, rank):
        arr_interfaces = []
        for arr in X:
            X_m, input_ptr, n_rows_X, self.n_cols, self.dtype = \
                input_to_dev_array(arr, check_dtype=[np.float32, np.float64])
            arr_interfaces.append({"obj": X_m,
                                   "data": input_ptr,
                                   "shape": (n_rows_X, self.n_cols)})

        arr_interfaces_y = []
        for arr in y:
            y_m, input_ptr, n_rows_y, n_cols_y, self.dtype = \
                input_to_dev_array(arr, check_dtype=[np.float32, np.float64])
            arr_interfaces_y.append({"obj": y_m,
                                     "data": input_ptr,
                                     "shape": (n_rows_y, n_cols_y)})

        n_total_parts = 0
        for idx, rankSize in enumerate(partsToSizes):
            rk, size = rankSize
            if rank == rk:
                n_total_parts = n_total_parts + 1

        cdef RankSizePair **rankSizePair = <RankSizePair**> \
            malloc(sizeof(RankSizePair**)
                   * n_total_parts)

        indx = 0
        n_part_row = 0

        for idx, rankSize in enumerate(partsToSizes):
            rk, size = rankSize
            if rank == rk:
                rankSizePair[indx] = <RankSizePair*> \
                    malloc(sizeof(RankSizePair))
                rankSizePair[indx].rank = <int>rank
                rankSizePair[indx].size = <size_t>size
                n_part_row = n_part_row + rankSizePair[indx].size
                indx = indx + 1

        self.coef_ = cudf.Series(zeros(self.n_cols,
                                       dtype=self.dtype))
        cdef uintptr_t coef_ptr = get_cudf_column_ptr(self.coef_)

        cdef float float_intercept
        cdef double double_intercept
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        cdef uintptr_t data
        cdef uintptr_t labels

        if self.dtype == np.float32:
            data = self._build_dataFloat(arr_interfaces)
            labels = self._build_dataFloat(arr_interfaces_y)

            fit(handle_[0],
                <RankSizePair**>rankSizePair,
                <size_t> n_total_parts,
                <floatData_t**>data,
                <size_t>n_rows,
                <size_t>n_cols,
                <floatData_t**>labels,
                <float*>coef_ptr,
                <float*>&float_intercept,
                <bool>self.fit_intercept,
                <bool>self.normalize,
                <int>self.algo,
                False)

            self.intercept_ = float_intercept
        else:
            data = self._build_dataDouble(arr_interfaces)
            labels = self._build_dataDouble(arr_interfaces_y)

            fit(handle_[0],
                <RankSizePair**>rankSizePair,
                <size_t> n_total_parts,
                <doubleData_t**>data,
                <size_t>n_rows,
                <size_t>n_cols,
                <doubleData_t**>labels,
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
            self._freeFloatD(data, arr_interfaces)
            self._freeFloatD(labels, arr_interfaces_y)
        else:
            self._freeDoubleD(data, arr_interfaces)
            self._freeDoubleD(labels, arr_interfaces_y)

    def fit_colocated(self, input_data, n_rows, n_cols, partsToSizes, rank):
        """
        Fit function for MNMG Linear Regression.
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
            X_m, input_ptr, n_rows_X, self.n_cols, self.dtype = \
                input_to_dev_array(input_data[i][0],
                                   check_dtype=[np.float32, np.float64])

            arr_interfaces.append({"obj": X_m,
                                   "data": input_ptr,
                                   "shape": (n_rows_X, self.n_cols)})

            y_m, input_ptr, n_rows_y, n_cols_y, self.dtype = \
                input_to_dev_array(input_data[i][1],
                                   check_dtype=[np.float32, np.float64])

            arr_interfaces_y.append({"obj": y_m,
                                     "data": input_ptr,
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

        self.coef_ = cudf.Series(zeros(self.n_cols,
                                       dtype=self.dtype))
        cdef uintptr_t coef_ptr = get_cudf_column_ptr(self.coef_)

        cdef float float_intercept
        cdef double double_intercept
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        cdef uintptr_t data
        cdef uintptr_t labels

        if self.dtype == np.float32:
            data = self._build_dataFloat(arr_interfaces)
            labels = self._build_dataFloat(arr_interfaces_y)

            fit(handle_[0],
                <RankSizePair**>rankSizePair,
                <size_t> n_total_parts,
                <floatData_t**>data,
                <size_t>n_rows,
                <size_t>n_cols,
                <floatData_t**>labels,
                <float*>coef_ptr,
                <float*>&float_intercept,
                <bool>self.fit_intercept,
                <bool>self.normalize,
                <int>self.algo,
                False)

            self.intercept_ = float_intercept
        else:
            data = self._build_dataDouble(arr_interfaces)
            labels = self._build_dataDouble(arr_interfaces_y)

            fit(handle_[0],
                <RankSizePair**>rankSizePair,
                <size_t> n_total_parts,
                <doubleData_t**>data,
                <size_t>n_rows,
                <size_t>n_cols,
                <doubleData_t**>labels,
                <double*>coef_ptr,
                <double*>&double_intercept,
                <bool>self.fit_intercept,
                <bool>self.normalize,
                <int>self.algo,
                False)

            self.intercept_ = double_intercept

        self.handle.sync()

        for idx in range(n_total_parts):
            free(<RankSizePair*>rankSizePair[idx])
        free(<RankSizePair**>rankSizePair)

        if self.dtype == np.float32:
            self._freeFloatD(data, arr_interfaces)
            self._freeFloatD(labels, arr_interfaces_y)
        else:
            self._freeDoubleD(data, arr_interfaces)
            self._freeDoubleD(labels, arr_interfaces_y)

    def predict(self, X, n_rows, n_cols, partsToSizes, rank):
        """
        Transform function for Linear Regression MG.
        This not meant to be used as
        part of the public API.
        :param X: array of local dataframes / array partitions
        :param n_rows: total number of rows
        :param n_cols: total number of cols
        :param partsToSizes: array of tuples in the format: [(rank,size)]
        :return: self
        """

        if n_cols != self.n_cols:
            raise Exception("Number of columns of the X has to match with "
                            "number of columns of the data was fit to model.")

        arr_interfaces = []
        for arr in X:
            X_m, input_ptr, n_rows_X, n_cols_X, self.dtype = \
                input_to_dev_array(arr, check_dtype=[np.float32, np.float64])
            arr_interfaces.append({"obj": X_m,
                                   "data": input_ptr,
                                   "shape": (n_rows_X, n_cols_X)})

        n_total_parts = 0
        for idx, rankSize in enumerate(partsToSizes):
            rk, size = rankSize
            if rank == rk:
                n_total_parts = n_total_parts + 1

        cdef RankSizePair **rankSizePair = <RankSizePair**> \
            malloc(sizeof(RankSizePair**)
                   * n_total_parts)

        indx = 0
        for idx, rankSize in enumerate(partsToSizes):
            rk, size = rankSize
            if rank == rk:
                rankSizePair[indx] = <RankSizePair*> \
                    malloc(sizeof(RankSizePair))
                rankSizePair[indx].rank = <int>rank
                rankSizePair[indx].size = <size_t>size
                indx = indx + 1

        cdef uintptr_t coef_ptr = get_cudf_column_ptr(self.coef_)
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef uintptr_t data
        cdef uintptr_t pred_data
        arr_interfaces_pred = []

        if self.dtype == np.float32:
            data = self._build_dataFloat(arr_interfaces)
            arr_interfaces_pred = self._build_predData(partsToSizes,
                                                       rank,
                                                       1,
                                                       np.float32)
            pred_data = self._build_dataFloat(arr_interfaces_pred)

            predict(handle_[0],
                    <RankSizePair**>rankSizePair,
                    <size_t> n_total_parts,
                    <floatData_t**> data,
                    <size_t>n_rows,
                    <size_t>n_cols,
                    <float*> coef_ptr,
                    <float>self.intercept_,
                    <floatData_t**> pred_data,
                    False)

        else:
            data = self._build_dataDouble(arr_interfaces)
            arr_interfaces_pred = self._build_predData(partsToSizes,
                                                       rank,
                                                       1,
                                                       np.float64)
            pred_data = self._build_dataDouble(arr_interfaces_pred)

            predict(handle_[0],
                    <RankSizePair**>rankSizePair,
                    <size_t> n_total_parts,
                    <doubleData_t**> data,
                    <size_t>n_rows,
                    <size_t>n_cols,
                    <double*> coef_ptr,
                    <double>self.intercept_,
                    <doubleData_t**> pred_data,
                    False)

        self.handle.sync()

        for idx in range(n_total_parts):
            free(<RankSizePair*>rankSizePair[idx])
        free(<RankSizePair**>rankSizePair)

        del(X_m)

        pred_cudf = []
        for x_i in arr_interfaces_pred:
            pred_cudf.append(cudf.DataFrame.from_gpu_matrix(x_i["obj"]))

        if self.dtype == np.float32:
            self._freeFloatD(pred_data, arr_interfaces_pred)
            self._freeFloatD(data, arr_interfaces)
        else:
            self._freeDoubleD(pred_data, arr_interfaces_pred)
            self._freeDoubleD(data, arr_interfaces)

        return pred_cudf
